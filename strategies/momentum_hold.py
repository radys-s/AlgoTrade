"""Momentum s držením pozice: vstup IOC po směru čistého toku obchodů, výstup po HOLD_US.

Parametry jsou konstanty třídy (grid je přepisuje v research/grid.py).
Inventář se počítá výhradně z fillů; strop |I| <= MAX_POS drží rezervu pod pokutovým pásmem ±100.
"""

from __future__ import annotations

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


class MomentumHoldStrategy(BaseStrategy):
    QTY = 20  # lotů na vstup
    MAX_POS = 80  # tvrdý strop inventáře (pokuta nad 100)
    MIN_FLOW = 1  # minimální |čistý tok| v callbacku pro vstup
    HOLD_US = 3_000_000  # jak dlouho držet pozici
    SLIP_TICKS = 2  # o kolik ticků přes touch smí IOC sweepnout
    FLATTEN_AT_US = 3_590_000_000  # od kdy jen zavírat (konec dat)

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.in_flight = False
        self.entry_us = 0

    def _ioc(self, side: Side, price: int, qty: int) -> None:
        self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=OrderType.IOC))
        self.in_flight = True

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        if self.in_flight or not snapshot.bids or not snapshot.asks:
            return
        now = snapshot.timestamp_us
        bid, ask = snapshot.bids[0].price, snapshot.asks[0].price

        # Výstup: po uplynutí HOLD_US nebo na konci dat
        if self.inv != 0 and (now - self.entry_us >= self.HOLD_US or now >= self.FLATTEN_AT_US):
            if self.inv > 0:
                self._ioc(Side.SELL, max(1, bid - self.SLIP_TICKS), self.inv)
            else:
                self._ioc(Side.BUY, ask + self.SLIP_TICKS, -self.inv)
            return
        if now >= self.FLATTEN_AT_US:
            return

        flow = 0
        for t in trades:
            flow += t.quantity if t.book_side == 'ask' else -t.quantity
        if abs(flow) < self.MIN_FLOW:
            return
        if flow > 0 and self.inv >= 0:
            qty = min(self.QTY, self.MAX_POS - self.inv)
            if qty > 0:
                self._ioc(Side.BUY, ask + self.SLIP_TICKS, qty)
        elif flow < 0 and self.inv <= 0:
            qty = min(self.QTY, self.MAX_POS + self.inv)
            if qty > 0:
                self._ioc(Side.SELL, max(1, bid - self.SLIP_TICKS), qty)
        elif self.inv != 0:  # tok proti pozici: zavřít
            if self.inv > 0:
                self._ioc(Side.SELL, max(1, bid - self.SLIP_TICKS), self.inv)
            else:
                self._ioc(Side.BUY, ask + self.SLIP_TICKS, -self.inv)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        if st in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            # znaménko fillu podle toho, kterou stranu jsme posílali, řešíme přes poslední záměr
            self._apply_fill(event)
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.in_flight = False

    _pending_side: Side | None = None

    def send_order(self, order: OrderRequest) -> str:
        self._pending_side = order.side
        return super().send_order(order)

    def _apply_fill(self, event: PrivateOrderEvent) -> None:
        was_flat = self.inv == 0
        self.inv += event.filled_quantity if self._pending_side == Side.BUY else -event.filled_quantity
        if was_flat and self.inv != 0:
            self.entry_us = event.timestamp_us
