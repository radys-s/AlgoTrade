"""Passive market maker steered by top-of-book imbalance.

Idea (from research/analyze_tape.py on sample_v2): spread 1-2 ticks, deep touch, no momentum after
trades (slight reversion), so a resting quote earns ~half-spread per fill. L1 imbalance predicts the
next mid move (+0.35 corr at 100 ms), so we pull the quote on the side that is about to be run over:
- imbalance strongly positive (more bids) -> price likely to rise -> don't sell at the ask
- imbalance strongly negative -> price likely to fall -> don't buy at the bid

Risk: potential inventory (inventory + all our resting/in-flight quantity on that side) never exceeds
MAX_POS, so latency can't push us over the +/-100 penalty band. Inventory is counted from fills only.
All parameters are class constants (research/grid.py overrides them).
"""

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

_BUY = Side.BUY
_SELL = Side.SELL
_DONE = (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)
_FILLS = (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)


class MMImbalanceStrategy(BaseStrategy):
    QTY = 20  # lots per quote
    MAX_POS = 60  # cap on potential inventory per side (penalty starts above 100)
    IMB_PULL = 30  # pull a side when |imbalance| > IMB_PULL / 100 against it
    SKEW_INV = 40  # stop adding on a side once |inventory| reaches this
    IMB_JUMP = 101  # when spread >= 2 and imbalance > IMB_JUMP/100, improve by 1 tick (101 = off)

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        # order_id -> [side, price, remaining_qty, cancel_sent]
        self.orders: dict[str, list] = {}

    # ---- helpers -------------------------------------------------------------------------------
    def _side_exposure(self, side: Side) -> int:
        total = 0
        for o in self.orders.values():
            if o[0] is side:
                total += o[2]
        return total

    def _sync(self, side: Side, want: int | None) -> None:
        has_live_at_want = False
        for oid, o in list(self.orders.items()):
            if o[0] is not side or o[3]:
                continue
            if want is not None and o[1] == want:
                has_live_at_want = True
            else:
                self.cancel_order(oid)
                o[3] = True
        if want is None or has_live_at_want:
            return
        exposure = self._side_exposure(side)
        if side is _BUY:
            room = self.MAX_POS - self.inv - exposure
        else:
            room = self.MAX_POS + self.inv - exposure
        qty = min(self.QTY, room)
        if qty <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=want, quantity=qty, order_type=OrderType.LIMIT))
        self.orders[oid] = [side, want, qty, False]

    # ---- callbacks -----------------------------------------------------------------------------
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
        bb = bids[0].price
        ba = asks[0].price
        bq = 0
        for o in bids:
            if o.price != bb:
                break
            if o.owner == 'market':
                bq += o.quantity
        aq = 0
        for o in asks:
            if o.price != ba:
                break
            if o.owner == 'market':
                aq += o.quantity
        tot = bq + aq
        imb = (bq - aq) * 100 / tot if tot else 0.0

        want_bid = bb if (imb > -self.IMB_PULL and self.inv < self.SKEW_INV) else None
        want_ask = ba if (imb < self.IMB_PULL and self.inv > -self.SKEW_INV) else None
        if ba - bb >= 2:
            if imb > self.IMB_JUMP and want_bid is not None:
                want_bid = bb + 1
            elif imb < -self.IMB_JUMP and want_ask is not None:
                want_ask = ba - 1
        self._sync(_BUY, want_bid)
        self._sync(_SELL, want_ask)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st in _FILLS:
            q = event.filled_quantity
            self.inv += q if o[0] is _BUY else -q
            o[2] = event.remaining_quantity
        if st in _DONE:
            del self.orders[event.order_id]
