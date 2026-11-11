"""exp_r2_grid2 (home, R2): exp_grid + the hidden-fair filter on every grid price, small orders, several orders per
price, and an optional over-cap resting mode with a breach guard.

Why: the round-1 winner made 12x the fills of the other top teams with many SMALL orders that aged to the front of
the queue at many prices and took a share of every taker sweep. exp_grid (static grid touch..touch-3, never
re-priced) lost on round2 because its orders become the touch when the price RETREATS onto them, and those front
fills lose -0.57 ticks/lot in the first 200 ms (round2_anatomy s.7). The hidden-fair signal of exp_r2_fair
(sig > 0: fair price above the mid) says which retreats continue, so it decides per price which orders may rest.

  * fair_thr / fair_tau_ms / fair_levels / fair_mid_any: the signal as exp_r2_taker._fair_update (fair_levels 20
    and the previous mid taken at any two-sided spread find the deep inserts that carry much of the signal;
    fair_levels 3 + fair_mid_any 0 = exp_r2_fair's version). fair = robust mid (spread <= 3) + sig.
    While sig < -fair_thr every bid above fair - grid_fair_margin is cancelled and none is placed there (asks
    mirrored while sig > fair_thr). fair_thr 0 = off (identical to exp_grid).
  * grid_orders_per_level / grid_add_ms / grid_add_touch: keep up to that many orders at one grid price; one more is
    added when the newest one there is >= grid_add_ms old and capacity allows (grid_add_touch 1: only while the
    price is the market touch; 0: at every grid price). 1 = off (identical to exp_grid).
  * rest_cap / breach_guard / breach_ioc_ticks / breach_target: rest_cap 0 = the hard invariant below (identical to
    exp_grid). rest_cap > 0 lets a side rest up to max(rest_cap, max_position - s*I) lots: the exchange rejects only
    when |I + resting same side| > 1000, and the 10,000/s penalty runs only while the ACTUAL |I| > 100. Then
    whenever s*I > breach_guard the side that grows |I| is cancelled at once (also from the fill callback) and not
    quoted, and whenever |I| > 100 an IOC breach_ioc_ticks through the market touch for |I| - breach_target lots
    (minus unwind lots in flight) is sent from the next snapshot. Counters: n_breach, breach_us (penalty =
    breach_us / 100), n_unwinds, unwind_lots.

exp_grid docstring:
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

HARD INVARIANT with rest_cap 0 (as mm.py): with OUT_k = remaining of every not-yet-dead order on side k,
(B) I + OUT_buy <= M and (S) -I + OUT_sell <= M. `_send` is the only place a grid order is created and refuses
anything that would break it; fills move lots from OUT to I one for one; cancels/rejects only lower OUT.
So |I| <= M <= 100 at all times: no breach penalty, and the 1000 reject rule cannot fire. With rest_cap > 0 only
s*I + OUT_k <= 900 is kept (no rejects); |I| may pass 100 and the guard above limits the time spent there.
"""

from collections import deque
from math import exp

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
# The exchange rejects an order when |I + s * (resting same side + q)| > 1000; we stay 100 below it.
REST_HARD_CAP = 900

_BUY = Side.BUY
_SELL = Side.SELL
_LIMIT = OrderType.LIMIT
_IOC = OrderType.IOC
_ACCEPTED = OrderStatus.ACCEPTED
_PARTIALLY_FILLED = OrderStatus.PARTIALLY_FILLED
_FILLED = OrderStatus.FILLED
_CANCELED = OrderStatus.CANCELED
_MODIFIED = OrderStatus.MODIFIED
_REJECTED = OrderStatus.REJECTED
_NEVER_US = -(1 << 62)


class _Order:
    """Our record of one order; `remaining` bounds what the exchange can still fill."""

    def __init__(self, order_id: str, k: int, sign: int, price: int, quantity: int, ioc: bool) -> None:
        self.order_id = order_id
        self.k = k  # 0 = buy side, 1 = sell side
        self.sign = sign
        self.price = price
        self.remaining = quantity
        self.accepted = False
        self.cancel_sent = False
        self.ioc = ioc  # a breach unwind, not part of the grid


class GridMaker2(BaseStrategy):
    """Static per-side price grid with a hidden-fair filter; see the module docstring."""

    PARAMS = {
        'max_position': 100,
        # Lots per grid order and number of prices per side (touch .. touch - levels + 1).
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
        # Several orders per grid price, placed at different times (1 = off).
        'grid_orders_per_level': 1,
        'grid_add_ms': 500,
        'grid_add_touch': 1,
        # Trend gate: robust mid drift over the window; pull the against-trend side. 0 = off.
        'gate_thr': 0.0,
        'gate_window_us': 30000000,
        'gate_max_spread': 3,
        # Flow pull: signed taker flow of the last flow_window_ev taker orders. 0 = off.
        'flow_pull': 0,
        'flow_window_ev': 30,
        # Hidden-fair filter (exp_r2_fair). 0 = off.
        'fair_thr': 0.0,
        'fair_tau_ms': 50.0,
        'fair_levels': 20,
        'fair_mid_any': 1,
        'grid_fair_margin': 0.5,
        # Over-cap resting (0 = off: the hard invariant) and its breach guard.
        'rest_cap': 0,
        'breach_guard': 80,
        'breach_ioc_ticks': 0,
        'breach_target': 95,
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
        self.opl = max(1, int(p['grid_orders_per_level']))
        self.add_us = int(float(p['grid_add_ms']) * 1000.0)
        self.add_touch = int(p['grid_add_touch'])
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
        self.fair_thr = float(p['fair_thr'])
        self.fair_tau_us = max(1.0, float(p['fair_tau_ms']) * 1000.0)
        self.fair_levels = max(1, int(p['fair_levels']))
        self.fair_mid_any = int(p['fair_mid_any'])
        self.fair_margin = float(p['grid_fair_margin'])
        self.fair_anchor2 = 0  # 2 x the last mid seen with spread <= 3: where the fair price is measured from
        self.fair_max_id = 0
        self.fair_s = 0.0
        self.fair_w = 0.0
        self.fair_t = 0
        self.fair_sig = 0.0
        self.fair_off = [-1.2, 1.2]  # running mean of (insert price - mid before) for bids / asks
        self.fair_n = [0, 0]
        self.fair_prev_mid2 = 0
        self.rest_cap = max(0, min(int(p['rest_cap']), REST_HARD_CAP))
        self.breach_guard = int(p['breach_guard'])
        self.breach_ioc_ticks = max(0, int(p['breach_ioc_ticks']))
        self.breach_target = max(0, min(int(p['breach_target']), EXCHANGE_MAX_POSITION))

        self.inv = 0
        self.orders: dict[str, _Order] = {}
        self.out = [0, 0]
        self.grid: list[dict[int, list[str]]] = [{}, {}]  # price -> ids of live (not cancelling) grid orders
        self.added: list[dict[int, int]] = [{}, {}]  # price -> time the newest grid order there was sent
        self.bb = -1
        self.ba = -1
        self.unw_qty = 0  # unwind IOC lots in flight

        self.n_orders = 0
        self.n_fills = 0
        self.fill_volume = 0
        self.n_cancels = 0
        self.n_rejects = 0
        self.n_blocked = 0
        self.invariant_breaks = 0
        self.max_abs_inv = 0
        self.n_breach = 0
        self.breach_us = 0
        self.breach_t0 = -1
        self.n_unwinds = 0
        self.unwind_lots = 0
        self.n_guard = 0

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
        self.bb = bb
        self.ba = ba

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
        lim_bid = lim_ask = None
        if self.fair_thr > 0.0:
            self._fair_update(now, bids, asks, bb, ba)
            if ba - bb <= 3:
                self.fair_anchor2 = bb + ba
            sig = self.fair_sig
            if self.fair_anchor2 <= 0:
                pass
            elif sig < -self.fair_thr:
                lim_bid = self.fair_anchor2 / 2.0 + sig - self.fair_margin
            elif sig > self.fair_thr:
                lim_ask = self.fair_anchor2 / 2.0 + sig + self.fair_margin
        if self.rest_cap > 0:
            inv = self.inv
            if inv > self.breach_guard:
                pull_bid = True
            elif -inv > self.breach_guard:
                pull_ask = True
            if inv > EXCHANGE_MAX_POSITION or -inv > EXCHANGE_MAX_POSITION:
                self._unwind()
        self._side(0, 1, _BUY, bb, qb, ba, pull_bid, lim_bid, now)
        self._side(1, -1, _SELL, ba, qa, bb, pull_ask, lim_ask, now)

    def _side(
        self, k: int, sign: int, side: Side, touch: int, touch_q: int, other: int, pulled: bool, lim, now: int
    ) -> None:
        grid = self.grid[k]
        orders = self.orders
        far = self.grid_levels - 1 + self.grid_slack
        if grid:
            for px in list(grid):
                d = (touch - px) * sign  # ticks behind the market touch; < 0 = alone in front of it
                if (
                    pulled
                    or d > far
                    or (d < 0 and not self.keep_alone)
                    or (lim is not None and (px - lim) * sign > 0)
                ):
                    for order_id in grid.pop(px):
                        self._cancel(orders[order_id])
        if pulled:
            return
        for j in range(self.grid_levels):
            px = touch - sign * j
            if px in grid or px <= 0 or (other - px) * sign <= 0:
                continue
            if lim is not None and (px - lim) * sign > 0:
                continue
            if j == 0 and 0 <= self.join_touch_max < touch_q:
                continue
            free = self._free(k, sign)
            if free < self.min_lots:
                return
            self._send(k, sign, side, px, min(self.grid_size, free), now)
        if self.opl > 1:
            added = self.added[k]
            for j in range(1 if self.add_touch else self.grid_levels):
                px = touch - sign * j
                ids = grid.get(px)
                if not ids or len(ids) >= self.opl or now - added.get(px, _NEVER_US) < self.add_us:
                    continue
                if j == 0 and 0 <= self.join_touch_max < touch_q:
                    continue
                free = self._free(k, sign)
                if free < self.min_lots:
                    return
                self._send(k, sign, side, px, min(self.grid_size, free), now)

    def _free(self, k: int, sign: int) -> int:
        free = self.max_pos - sign * self.inv - self.out[k]
        if self.rest_cap > 0 and self.rest_cap - self.out[k] > free:
            free = self.rest_cap - self.out[k]
        return free

    def _fair_update(self, now: int, bids, asks, bb: int, ba: int) -> None:
        """Fold a new market limit order (if this row inserted one) into the hidden-fair signal."""
        prev2 = self.fair_prev_mid2
        if self.fair_mid_any or ba - bb <= 3:
            self.fair_prev_mid2 = bb + ba
        if prev2 <= 0:
            return
        found = None
        for k, views in ((0, bids), (1, asks)):
            last_px = -1
            last_view = None
            nlev = 0
            for v in views:
                if v.price != last_px:
                    if last_view is not None and last_view.owner == 'market':
                        oid = int(last_view.order_id)
                        if oid > self.fair_max_id:
                            self.fair_max_id = oid
                            found = (k, last_view.price)
                    nlev += 1
                    if nlev > self.fair_levels:
                        last_view = None
                        break
                    last_px = v.price
                last_view = v
            if last_view is not None and last_view.owner == 'market':
                oid = int(last_view.order_id)
                if oid > self.fair_max_id:
                    self.fair_max_id = oid
                    found = (k, last_view.price)
        if found is None:
            return
        k, px = found
        x = px - prev2 / 2.0
        n = self.fair_n[k] + 1
        self.fair_n[k] = n
        self.fair_off[k] += (x - self.fair_off[k]) / min(n + 50, 20000)
        dev = x - self.fair_off[k]
        a = exp(-(now - self.fair_t) / self.fair_tau_us)
        self.fair_t = now
        self.fair_s = self.fair_s * a + dev
        self.fair_w = self.fair_w * a + 1.0
        self.fair_sig = self.fair_s / self.fair_w

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

    def _clear_side(self, k: int) -> None:
        """Cancel every grid order of side k (breach guard)."""
        grid = self.grid[k]
        if not grid:
            return
        self.n_guard += 1
        orders = self.orders
        for px in list(grid):
            for order_id in grid.pop(px):
                self._cancel(orders[order_id])

    def _send(self, k: int, sign: int, side: Side, px: int, qty: int, now: int) -> None:
        """The only place grid orders are created: refuse anything that could break the invariant (or the cap)."""
        if qty <= 0 or px <= 0:
            self.n_blocked += 1
            return
        if self.rest_cap > 0:
            if sign * self.inv + self.out[k] + qty > REST_HARD_CAP or qty > self._free(k, sign):
                self.n_blocked += 1
                return
        elif sign * self.inv + self.out[k] + qty > self.max_pos:
            self.n_blocked += 1
            return
        order_id = self.send_order(OrderRequest(side=side, price=px, quantity=qty, order_type=_LIMIT))
        self.orders[order_id] = _Order(order_id, k, sign, px, qty, False)
        self.out[k] += qty
        ids = self.grid[k].get(px)
        if ids is None:
            self.grid[k][px] = [order_id]
        else:
            ids.append(order_id)
        self.added[k][px] = now
        self.n_orders += 1

    def _unwind(self) -> None:
        """|I| > 100: one IOC through the market touch for the excess over breach_target (minus what is in flight)."""
        inv = self.inv
        qty = abs(inv) - self.breach_target - self.unw_qty
        if qty <= 0:
            return
        if inv > 0:
            k, sign, side, px = 1, -1, _SELL, self.bb - self.breach_ioc_ticks
        else:
            k, sign, side, px = 0, 1, _BUY, self.ba + self.breach_ioc_ticks
        if px <= 0:
            return
        order_id = self.send_order(OrderRequest(side=side, price=px, quantity=qty, order_type=_IOC))
        self.orders[order_id] = _Order(order_id, k, sign, px, qty, True)
        self.out[k] += qty
        self.unw_qty += qty
        self.n_unwinds += 1
        self.unwind_lots += qty

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
            if order.ioc:
                self.unw_qty -= qty
            if order.remaining <= 0:
                self._forget(order)
            inv = self.inv
            if self.rest_cap > 0:
                if inv > EXCHANGE_MAX_POSITION or -inv > EXCHANGE_MAX_POSITION:
                    if self.breach_t0 < 0:
                        self.breach_t0 = event.timestamp_us
                        self.n_breach += 1
                elif self.breach_t0 >= 0:
                    self.breach_us += event.timestamp_us - self.breach_t0
                    self.breach_t0 = -1
                if inv > self.breach_guard:
                    self._clear_side(0)
                elif -inv > self.breach_guard:
                    self._clear_side(1)
            elif inv + self.out[0] > self.max_pos or self.out[1] - inv > self.max_pos:
                self.invariant_breaks += 1
            if abs(inv) > self.max_abs_inv:
                self.max_abs_inv = abs(inv)
        elif status is _ACCEPTED:
            order.accepted = True
        elif status is _CANCELED:
            self._drop(order)
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
            self._drop(order)

    def _drop(self, order: _Order) -> None:
        """The order is dead with `remaining` unfilled: release it from OUT."""
        self.out[order.k] -= order.remaining
        if order.ioc:
            self.unw_qty -= order.remaining
        order.remaining = 0
        self._forget(order)

    def _forget(self, order: _Order) -> None:
        self.orders.pop(order.order_id, None)
        grid = self.grid[order.k]
        ids = grid.get(order.price)
        if ids and order.order_id in ids:
            ids.remove(order.order_id)
            if not ids:
                del grid[order.price]
