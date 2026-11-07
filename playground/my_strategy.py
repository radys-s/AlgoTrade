"""Playground baseline: a simple passive market maker with a trend filter.

THIS is the file to experiment with. Everything you may want to change is in PARAMS or in the
clearly marked "STRATEGY LOGIC" section. Run it with:

    uv run python -X utf8 playground/run.py --data data/round2.csv
    uv run python -X utf8 playground/run.py --data data/round2.csv --frac 0.25          # quick: first 25 %
    uv run python -X utf8 playground/run.py --data data/round2.csv --set '{"quote_size": 30}'

How the exchange works (what the strategy can rely on):
  * on_order_book_change(snapshot, trades) is called after every market event. snapshot.bids / .asks are
    lists of single orders (best price first, then time priority) with .price, .quantity, .owner
    ('market' or 'user' = us), .order_id. `trades` are the trades of that event.
  * send_order(OrderRequest(...)) -> order_id. Orders become active 5 ms later (latency).
  * cancel_order(id); modify_order(id, smaller_qty) keeps the queue place (only DOWN).
  * on_private_order_update(event): our order's status (ACCEPTED / PARTIALLY_FILLED / FILLED / CANCELED / REJECTED),
    event.filled_quantity, event.fill_price, event.remaining_quantity.
  * Score = cash + inventory * final mark-out price. |inventory| > 100 costs 10,000 $ per second.
    Orders whose worst case would exceed |1000| are rejected. No fees.

Baseline idea (deliberately simple, lots of room to improve):
  1. Join the best bid and best ask with `quote_size` lots each (passive, earns the spread).
  2. Never let inventory + resting buys exceed max_position (same for sells) -> no penalty, ever.
  3. Trend filter: a fast and a slow exponential average of the mid price. If the fast one is
     `trend_thr` ticks below the slow one, the price is falling -> stop buying (no bid). Rising -> no ask.
     In round 2 the price falls 450 ticks at one point; a market maker without this keeps buying into it.
  4. When the touch price changes we cancel and re-send (simple, but it loses queue position; the champion
     `strategies/mm_ml_r1tuned.py` keeps queue with modify_order "parking" instead -> idea to port).
"""

from simulator.models import OrderRequest, OrderStatus, OrderType, Side
from simulator.strategy import BaseStrategy


class PlaygroundStrategy(BaseStrategy):
    PARAMS = {
        'quote_size': 40,       # lots per side at the touch
        'max_position': 60,     # |inventory + resting same-side lots| never above this (penalty starts > 100)
        'trend_fast_s': 5.0,    # fast EMA time constant of the mid, seconds
        'trend_slow_s': 60.0,   # slow EMA time constant, seconds
        'trend_thr': 3.0,       # ticks: |fast - slow| above this = trending, quote only with the trend
        'skew_lots': 0,         # 0 = off. If > 0: stop quoting the side that increases |inventory| beyond this
    }

    def __init__(self) -> None:
        super().__init__()
        p = self.PARAMS
        self.qsize = int(p['quote_size'])
        self.maxpos = min(100, int(p['max_position']))
        self.fast_tau = float(p['trend_fast_s']) * 1e6
        self.slow_tau = float(p['trend_slow_s']) * 1e6
        self.thr = float(p['trend_thr'])
        self.skew_lots = int(p['skew_lots'])
        self.inv = 0                      # our inventory (lots), from fills
        self.cash = 0.0
        self.orders = {}                  # order_id -> [side, price, remaining, cancel_sent]
        self.fast = self.slow = None      # EMAs of mid
        self.last_ts = None

    # ------------------------------------------------------------------ helpers
    def _resting(self, side: Side) -> int:
        """Lots we have working on one side (including orders still in flight)."""
        return sum(o[2] for o in self.orders.values() if o[0] is side)

    def _keep_quote(self, side: Side, price: int | None) -> None:
        """Make our quote on `side` sit at `price` (None = no quote on this side)."""
        for oid, o in list(self.orders.items()):
            if o[0] is side and not o[3] and o[1] != price:
                self.cancel_order(oid)
                o[3] = True
        if price is None:
            return
        live = sum(o[2] for o in self.orders.values() if o[0] is side and not o[3] and o[1] == price)
        sign = 1 if side is Side.BUY else -1
        room = self.maxpos - sign * self.inv - self._resting(side)   # invariant: no penalty possible
        qty = min(self.qsize - live, room)
        if qty > 0:
            oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=OrderType.LIMIT))
            self.orders[oid] = [side, price, qty, False]

    # ------------------------------------------------------------------ callbacks
    def on_order_book_change(self, snapshot, trades) -> None:
        if not snapshot.bids or not snapshot.asks:
            return
        best_bid = snapshot.bids[0].price
        best_ask = snapshot.asks[0].price
        mid = (best_bid + best_ask) / 2
        now = snapshot.timestamp_us

        # EMAs of the mid in continuous time
        if self.fast is None:
            self.fast = self.slow = mid
        else:
            dt = now - self.last_ts
            self.fast += (mid - self.fast) * min(1.0, dt / self.fast_tau)
            self.slow += (mid - self.slow) * min(1.0, dt / self.slow_tau)
        self.last_ts = now
        trend = self.fast - self.slow     # > 0 rising, < 0 falling (ticks)

        # ======================= STRATEGY LOGIC (experiment here) =======================
        want_bid = best_bid
        want_ask = best_ask
        if trend < -self.thr:             # falling: do not buy
            want_bid = None
        elif trend > self.thr:            # rising: do not sell
            want_ask = None
        if self.skew_lots:
            if self.inv >= self.skew_lots:
                want_bid = None
            if self.inv <= -self.skew_lots:
                want_ask = None
        # =================================================================================

        self._keep_quote(Side.BUY, want_bid)
        self._keep_quote(Side.SELL, want_ask)

    def on_private_order_update(self, event) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        if event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            q = event.filled_quantity
            sign = 1 if o[0] is Side.BUY else -1
            self.inv += sign * q
            self.cash -= sign * q * event.fill_price
            o[2] = event.remaining_quantity
        if event.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            del self.orders[event.order_id]
