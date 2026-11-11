"""exp_grid (home, R2): a static price grid maker that never re-prices, so its orders age to the front.

Why (round2 anatomy): market orders live ~0.3 s, so an order resting at one price for >= 1 s has 0 lots ahead
84 % of the time when its level becomes the touch (2 s: 97 %). The front-of-queue oracle turns the round-2 plain
maker from -217k into +170k; every cancel/re-quote sends us to the back of a ~700-lot queue instead.

Per side keep one order of grid_size lots at each price touch, touch-1, ..., touch-(grid_levels-1) (asks
mirrored), where touch = the best MARKET price (our orders excluded). Orders are never moved: a price that
leaves the band by more than grid_slack ticks is cancelled, a missing price is filled from the free capacity,
nearest the touch first. The touch level itself is joined only when its market queue is <= join_touch_max lots
(a freshly formed level: we stand near its front); deeper levels are always joined (they age before the price
arrives). An order of ours alone at a better price than the market touch (its level was emptied around it)
is kept unless keep_alone is 0.

Trend gate (as exp_r2_ladder): drift = robust mid change over gate_window_us (sampled while spread <=
gate_max_spread); |drift| >= gate_thr cancels the whole against-trend side. gate_thr 0 = off.
flow_pull (as mm_r2_insurance): signed taker flow of the last flow_window_ev taker orders beyond +-flow_pull
cancels the side the flow runs into. 0 = off.

HARD INVARIANT (as mm.py): with OUT_k = remaining of every not-yet-dead order on side k,
(B) I + OUT_buy <= M and (S) -I + OUT_sell <= M. `_send` is the only place an order is created and refuses
anything that would break it; fills move lots from OUT to I one for one; cancels/rejects only lower OUT.
So |I| <= M <= 100 at all times: no breach penalty, and the 1000 reject rule cannot fire.
"""

from collections import deque

from simulator.models import (
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    Trade,
)
from simulator.strategy import BaseStrategy

EXCHANGE_MAX_POSITION = 100

_BUY = Side.BUY
_SELL = Side.SELL
_LIMIT = OrderType.LIMIT
_ACCEPTED = OrderStatus.ACCEPTED
_PARTIALLY_FILLED = OrderStatus.PARTIALLY_FILLED
_FILLED = OrderStatus.FILLED
_CANCELED = OrderStatus.CANCELED
_MODIFIED = OrderStatus.MODIFIED
_REJECTED = OrderStatus.REJECTED
_NEVER_US = -(1 << 62)


class _Order:
    """Our record of one resting order; `remaining` bounds what the exchange can still fill."""

    __slots__ = ('order_id', 'k', 'sign', 'price', 'remaining', 'accepted', 'cancel_sent')

    def __init__(self, order_id: str, k: int, sign: int, price: int, quantity: int) -> None:
        self.order_id = order_id
        self.k = k  # 0 = buy side, 1 = sell side
        self.sign = sign
        self.price = price
        self.remaining = quantity
        self.accepted = False
        self.cancel_sent = False


class GridMaker(BaseStrategy):
    """Static per-side price grid; see the module docstring."""

    PARAMS = {
        'max_position': 100,
        # Lots per grid price and number of prices per side (touch .. touch - levels + 1).
        'grid_size': 25,
        'grid_levels': 4,
        # Extra ticks beyond the band before a grid order is cancelled.
        'grid_slack': 1,
        # Join the touch price only while its market queue is at most this many lots (-1: always).
        'join_touch_max': -1,
        # Keep an order of ours that is alone at a better price than the market touch.
        'keep_alone': 1,
        # Smallest order worth sending when capacity is short.
        'min_lots': 1,
        # Trend gate: robust mid drift over the window; pull the against-trend side. 0 = off.
        'gate_thr': 0.0,
        'gate_window_us': 30000000,
        'gate_max_spread': 3,
        # Flow pull: signed taker flow of the last flow_window_ev taker orders. 0 = off.
        'flow_pull': 0,
        'flow_window_ev': 30,
    }

    def __init__(self) -> None:
        super().__init__()
        p = self.PARAMS
        self.max_pos = max(0, min(int(p['max_position']), EXCHANGE_MAX_POSITION))
        self.grid_size = max(1, int(p['grid_size']))
        self.grid_levels = max(1, int(p['grid_levels']))
        self.grid_slack = max(0, int(p['grid_slack']))
        self.join_touch_max = int(p['join_touch_max'])
        self.keep_alone = int(p['keep_alone'])
        self.min_lots = max(1, int(p['min_lots']))
        self.gate_thr = float(p['gate_thr'])
        self.gate_window = int(p['gate_window_us'])
        self.gate_max_spread = int(p['gate_max_spread'])
        self.gate_pts: deque = deque()
        self.gate_last_ts = _NEVER_US
        self.gate_mid2 = 0
        self.flow_pull = int(p['flow_pull'])
        self.flow_window_ev = max(1, int(p['flow_window_ev']))
        self.flow_q: deque = deque()
        self.flow = 0
        self.flow_last_taker = ''

        self.inv = 0
        self.orders: dict[str, _Order] = {}
        self.out = [0, 0]
        self.grid: list[dict[int, str]] = [{}, {}]  # price -> order id of live (not cancelling) orders

        self.n_orders = 0
        self.n_fills = 0
        self.fill_volume = 0
        self.n_cancels = 0
        self.n_rejects = 0
        self.n_blocked = 0
        self.invariant_breaks = 0
        self.max_abs_inv = 0

    # ------------------------------------------------------------------ market feed

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        if self.flow_pull > 0:
            fq = self.flow_q
            for t in trades:
                sq = t.quantity if t.book_side == 'ask' else -t.quantity
                self.flow += sq
                if t.taker_order_id == self.flow_last_taker:
                    fq[-1] += sq
                    continue
                self.flow_last_taker = t.taker_order_id
                fq.append(sq)
                if len(fq) > self.flow_window_ev:
                    self.flow -= fq.popleft()
        bids = snapshot.bids
        asks = snapshot.asks
        if not bids or not asks:
            return
        # Market touch and its market queue (our orders skipped).
        bb = -1
        qb = 0
        for v in bids:
            if v.owner != 'market':
                continue
            if bb < 0:
                bb = v.price
            elif v.price != bb:
                break
            qb += v.quantity
        ba = -1
        qa = 0
        for v in asks:
            if v.owner != 'market':
                continue
            if ba < 0:
                ba = v.price
            elif v.price != ba:
                break
            qa += v.quantity
        if bb < 0 or ba < 0:
            return
        now = snapshot.timestamp_us

        pull_bid = pull_ask = False
        if self.gate_thr > 0.0:
            drift = self._drift(now, bb, ba)
            if drift <= -self.gate_thr:
                pull_bid = True
            elif drift >= self.gate_thr:
                pull_ask = True
        if self.flow_pull > 0:
            pull_bid = pull_bid or self.flow < -self.flow_pull
            pull_ask = pull_ask or self.flow > self.flow_pull
        self._side(0, 1, _BUY, bb, qb, ba, pull_bid)
        self._side(1, -1, _SELL, ba, qa, bb, pull_ask)

    def _side(self, k: int, sign: int, side: Side, touch: int, touch_q: int, other: int, pulled: bool) -> None:
        grid = self.grid[k]
        orders = self.orders
        far = self.grid_levels - 1 + self.grid_slack
        if grid:
            for px in list(grid):
                d = (touch - px) * sign  # ticks behind the market touch; < 0 = alone in front of it
                if pulled or d > far or (d < 0 and not self.keep_alone):
                    self._cancel(orders[grid.pop(px)])
        if pulled:
            return
        for j in range(self.grid_levels):
            px = touch - sign * j
            if px in grid or px <= 0 or (other - px) * sign <= 0:
                continue
            if j == 0 and 0 <= self.join_touch_max < touch_q:
                continue
            free = self.max_pos - sign * self.inv - self.out[k]
            if free < self.min_lots:
                return
            self._send(k, sign, side, px, min(self.grid_size, free))

    def _drift(self, now: int, bb: int, ba: int) -> float:
        """Mid change (ticks) over the gate window, from samples taken while the spread is normal."""
        pts = self.gate_pts
        horizon = now - self.gate_window
        if ba - bb <= self.gate_max_spread:
            self.gate_mid2 = bb + ba
            if now - self.gate_last_ts >= self.gate_window // 200:
                pts.append((now, bb + ba))
                self.gate_last_ts = now
        while len(pts) > 1 and pts[1][0] <= horizon:
            pts.popleft()
        if not pts or pts[0][0] > horizon + self.gate_window // 10:
            return 0.0
        return (self.gate_mid2 - pts[0][1]) / 2.0

    def _cancel(self, order: _Order) -> None:
        if not order.cancel_sent:
            order.cancel_sent = True
            self.cancel_order(order.order_id)
            self.n_cancels += 1

    def _send(self, k: int, sign: int, side: Side, px: int, qty: int) -> None:
        """The only place orders are created: refuse anything that could break the invariant."""
        if qty <= 0 or px <= 0 or sign * self.inv + self.out[k] + qty > self.max_pos:
            self.n_blocked += 1
            return
        order_id = self.send_order(OrderRequest(side=side, price=px, quantity=qty, order_type=_LIMIT))
        self.orders[order_id] = _Order(order_id, k, sign, px, qty)
        self.out[k] += qty
        self.grid[k][px] = order_id
        self.n_orders += 1

    # ------------------------------------------------------------------ private feed

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        status = event.status
        order = self.orders.get(event.order_id)
        if order is None:
            return
        if status is _FILLED or status is _PARTIALLY_FILLED:
            qty = event.filled_quantity
            self.inv += order.sign * qty
            self.n_fills += 1
            self.fill_volume += qty
            order.remaining -= qty
            self.out[order.k] -= qty
            if order.remaining <= 0:
                self._forget(order)
            inv = self.inv
            if inv + self.out[0] > self.max_pos or self.out[1] - inv > self.max_pos:
                self.invariant_breaks += 1
            if abs(inv) > self.max_abs_inv:
                self.max_abs_inv = abs(inv)
        elif status is _ACCEPTED:
            order.accepted = True
        elif status is _CANCELED:
            self.out[order.k] -= order.remaining
            order.remaining = 0
            self._forget(order)
        elif status is _MODIFIED:
            drop = order.remaining - event.remaining_quantity
            if drop > 0:
                order.remaining -= drop
                self.out[order.k] -= drop
        elif status is _REJECTED:
            if not order.accepted:
                self.n_rejects += 1
            elif event.remaining_quantity > 0:
                return
            self.out[order.k] -= order.remaining
            order.remaining = 0
            self._forget(order)

    def _forget(self, order: _Order) -> None:
        self.orders.pop(order.order_id, None)
        grid = self.grid[order.k]
        if grid.get(order.price) == order.order_id:
            del grid[order.price]
