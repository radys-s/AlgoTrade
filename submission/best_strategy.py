"""Momentum s držením pozice: vstup IOC po směru čistého toku obchodů, výstup po HOLD_US.

Parametry jsou konstanty třídy (grid je přepisuje v research/grid.py i research/run.py).
Inventář se počítá výhradně z fillů; strop |I| <= MAX_POS drží rezervu pod pokutovým pásmem ±100.

v2 oproti momentum_hold: konec dat. Původní FLATTEN_AT_US byl absolutní čas pásky (3 590 s), takže
na delším kole strategie po hodině přestala obchodovat a na pásce, která nezačíná v čase 0, by
neobchodovala vůbec. FLATTEN_AFTER_US se měří od první události a je ve výchozím stavu vypnutý:
markout na konci je sám IOC likvidace pozice do knihy, tedy totéž, co by udělalo zavření, a díky
|I| <= MAX_POS a HOLD_US je ta pozice na konci malá.
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


class MomentumHoldV2Strategy(BaseStrategy):
    QTY = 50  # lotů na vstup (grid na samplu: nad ~25 už málo pomáhá, likvidita na touch ~25)
    MAX_POS = 80  # tvrdý strop inventáře (pokuta nad 100)
    MIN_FLOW = 1  # minimální |čistý tok| v callbacku pro vstup
    HOLD_US = 5_000_000  # jak dlouho držet pozici (grid: 5 s nejlepší)
    SLIP_TICKS = 3  # o kolik ticků přes touch smí IOC sweepnout
    FLATTEN_AFTER_US = 0  # jen zavírat po tolika µs od první události; 0 = vypnuto (délka kola není známá)

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.in_flight = False
        self.entry_us = 0
        self.start_us = -1

    def _ioc(self, side: Side, price: int, qty: int) -> None:
        self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=OrderType.IOC))
        self.in_flight = True

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        if self.start_us < 0:
            self.start_us = now
        if self.in_flight or not snapshot.bids or not snapshot.asks:
            return
        bid, ask = snapshot.bids[0].price, snapshot.asks[0].price
        flatten = self.FLATTEN_AFTER_US > 0 and now - self.start_us >= self.FLATTEN_AFTER_US

        # Výstup: po uplynutí HOLD_US, nebo ve fázi zavírání
        if self.inv != 0 and (now - self.entry_us >= self.HOLD_US or flatten):
            if self.inv > 0:
                self._ioc(Side.SELL, max(1, bid - self.SLIP_TICKS), self.inv)
            else:
                self._ioc(Side.BUY, ask + self.SLIP_TICKS, -self.inv)
            return
        if flatten:
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
