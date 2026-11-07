"""Signal taker: IOC in the direction of one feature (or a pasted linear model), hold HOLD_US, IOC out.

Same state machine as `research/taker_scan.py` (the fast npz proxy that scans thousands of
configs); confirm a proxy winner here with

    uv run python -m research.run --strategy strategies/signal_taker.py --data data/round1.csv \
        --set '{"SIGNAL": "flow_ev10", "THRESHOLD": 20, "HOLD_US": 5000000, "QTY": 50, "SLIP_TICKS": 3}'

Signals (definitions as in research/signals.py `OnlineFeatures`, computed only for SIGNAL):
imb1 imb2 imb3 imb5 micro cnt_imb1 flow_ev<N> flow_ms<T> ret_ms<T> ltp_dev last_dmid, or
'linear' (LINEAR_MODEL = an `export_linear` dict over the full OnlineFeatures set; slower).
Trade flow and the last trade price count market takers only: our own IOC fills are left out,
so the online value equals the offline (no-trading) feature the proxy uses.

Logic per event: when flat, or long/short with |I| < MAX_POS, signal > THRESHOLD -> IOC buy QTY
at ask + SLIP_TICKS, signal < -THRESHOLD -> IOC sell at bid - SLIP_TICKS. The whole position is
closed with an IOC once HOLD_US has passed since the fill that opened it (and on a reverse signal
when EXIT_ON_REVERSE). One order in flight at a time, inventory from fills only, no flatten at
the end (the final markout liquidates a position of at most MAX_POS lots).
"""

from __future__ import annotations

import math
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

K_IMB, K_MICRO, K_CNT, K_FLOW_EV, K_FLOW_MS, K_RET, K_LTP, K_DMID, K_LIN = range(9)


def _depth(views: list, levels: int) -> int:
    """Quantity at the best `levels` distinct prices of one side (best first)."""
    total = 0
    level = 0
    price = views[0].price
    for view in views:
        if view.price != price:
            level += 1
            if level == levels:
                break
            price = view.price
        total += view.quantity
    return total


def _imb(b: float, a: float) -> float:
    return (b - a) / (b + a) if b + a > 0 else 0.0


class _FullFeatures:
    """Pasted copy of research.signals.OnlineFeatures (default lookbacks), used only for SIGNAL='linear'."""

    def __init__(self) -> None:
        self.flow_events = (1, 10, 100)
        self.flow_ms = (10, 100, 1000, 10000)
        self.ret_ms = (100, 1000, 10000)
        self._ev_hist = deque(maxlen=100)
        self._ev_sums = [0.0, 0.0, 0.0]
        self._flow_windows = [(t * 1000, deque(), [0.0]) for t in self.flow_ms]
        self._fills = deque()
        self._fills_sum = 0
        self._events = deque()
        self._mid_hist = [(t * 1000, deque()) for t in self.ret_ms]
        self._mid_ff = math.nan
        self._last_dmid = 0.0
        self._last_change_us = None
        self._last_trade_px = 0

    def update(self, snapshot: OrderBookSnapshot, trades: list) -> dict | None:
        t = snapshot.timestamp_us
        if self._last_change_us is None:
            self._last_change_us = t
        flow = 0.0
        for trade in trades:
            flow += trade.quantity if trade.book_side == 'ask' else -trade.quantity
            self._last_trade_px = trade.price
        hist = self._ev_hist
        for k, n in enumerate(self.flow_events):
            self._ev_sums[k] += flow
            if len(hist) >= n:
                self._ev_sums[k] -= hist[-n]
        hist.append(flow)
        for window, items, total in self._flow_windows:
            if flow:
                items.append((t, flow))
                total[0] += flow
            while items and items[0][0] <= t - window:
                total[0] -= items.popleft()[1]
        if trades:
            self._fills.append((t, len(trades)))
            self._fills_sum += len(trades)
        while self._fills and self._fills[0][0] <= t - 1_000_000:
            self._fills_sum -= self._fills.popleft()[1]
        self._events.append(t)
        while self._events[0] <= t - 1_000_000:
            self._events.popleft()
        bids, asks = snapshot.bids, snapshot.asks
        two_sided = bool(bids) and bool(asks)
        if two_sided:
            mid = (bids[0].price + asks[0].price) / 2
            if mid != self._mid_ff:
                if self._mid_ff == self._mid_ff:
                    self._last_dmid = mid - self._mid_ff
                    self._last_change_us = t
                self._mid_ff = mid
        rets = []
        for window, items in self._mid_hist:
            items.append((t, self._mid_ff))
            while len(items) >= 2 and items[1][0] <= t - window:
                items.popleft()
            rets.append(self._mid_ff - items[0][1] if items[0][0] <= t - window else math.nan)
        if not two_sided:
            return None
        b = [_depth(bids, k) for k in (1, 2, 3, 5)]
        a = [_depth(asks, k) for k in (1, 2, 3, 5)]
        bp, ap = bids[0].price, asks[0].price
        nb = 0
        for view in bids:
            if view.price != bp:
                break
            nb += 1
        na = 0
        for view in asks:
            if view.price != ap:
                break
            na += 1
        spread = float(ap - bp)
        imb1 = _imb(b[0], a[0])
        out = {
            'spread': spread, 'imb1': imb1, 'imb2': _imb(b[1], a[1]), 'imb3': _imb(b[2], a[2]),
            'imb5': _imb(b[3], a[3]), 'micro': spread * imb1 / 2, 'cnt_imb1': _imb(nb, na),
            'depth1_log': math.log1p(b[0] + a[0]),
        }  # fmt: skip
        for n, total in zip(self.flow_events, self._ev_sums):
            out['flow_ev' + str(n)] = total
        for t_ms, (_, _, total) in zip(self.flow_ms, self._flow_windows):
            out['flow_ms' + str(t_ms)] = total[0]
        out['ntr_ms1000'] = float(self._fills_sum)
        out['nev_ms1000'] = float(len(self._events))
        for t_ms, value in zip(self.ret_ms, rets):
            out['ret_ms' + str(t_ms)] = value
        out['last_dmid'] = self._last_dmid
        out['dt_mid_log'] = math.log1p((t - self._last_change_us) / 1000)
        out['ltp_dev'] = self._last_trade_px - (bp + ap) / 2 if self._last_trade_px > 0 else 0.0
        return out


def _predict_linear(model: dict, x: dict) -> float:
    total = model['intercept']
    lo, hi, mean, scale = model['lo'], model['hi'], model['mean'], model['scale']
    for name, coef in model['coef'].items():
        value = x[name]
        if value < lo[name]:
            value = lo[name]
        elif value > hi[name]:
            value = hi[name]
        total += coef * (value - mean[name]) / scale[name]
    return total


class SignalTakerStrategy(BaseStrategy):
    SIGNAL = 'flow_ev1'  # feature name (see module docstring) or 'linear'
    SIGN = 1  # +1 trade with the signal, -1 fade it (signal is multiplied by SIGN)
    THRESHOLD = 0.0  # enter when SIGN * signal > THRESHOLD (buy) or < -THRESHOLD (sell)
    HOLD_US = 5_000_000  # close the position this long after the fill that opened it
    QTY = 50  # lots per entry IOC
    SLIP_TICKS = 3  # IOC limit = touch + SLIP_TICKS
    MAX_POS = 80  # |I| cap (penalty above 100)
    EXIT_ON_REVERSE = False  # also close when the signal crosses the opposite threshold
    COOLDOWN_US = 0  # minimum time between two entry orders
    LINEAR_MODEL = {}  # export_linear dict for SIGNAL = 'linear'

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.in_flight = False
        self.side = Side.BUY
        self.entry_us = 0
        self.start_us = -1
        self.last_entry_us = -(10**18)
        name = self.SIGNAL
        self.n = 0
        self.full = None
        if name in ('imb1', 'imb2', 'imb3', 'imb5'):
            self.kind, self.n = K_IMB, int(name[3:])
        elif name == 'micro':
            self.kind = K_MICRO
        elif name == 'cnt_imb1':
            self.kind = K_CNT
        elif name.startswith('flow_ev'):
            self.kind, self.n = K_FLOW_EV, int(name[7:])
        elif name.startswith('flow_ms'):
            self.kind, self.n = K_FLOW_MS, int(name[7:]) * 1000
        elif name.startswith('ret_ms'):
            self.kind, self.n = K_RET, int(name[6:]) * 1000
        elif name == 'ltp_dev':
            self.kind = K_LTP
        elif name == 'last_dmid':
            self.kind = K_DMID
        elif name == 'linear':
            self.kind = K_LIN
            self.full = _FullFeatures()
        else:
            raise ValueError('unknown SIGNAL ' + str(name))
        self.hist = deque()  # flow_ev: last n flows; flow_ms: (ts, flow); ret_ms: (ts, mid_ff)
        self.acc = 0  # running sum for flow windows
        self.mid_ff = math.nan
        self.last_dmid = 0.0
        self.ltp = 0
        self.sig = math.nan

    # ------------------------------------------------------------------ signal

    def _signal(self, now: int, flow: int, bids: list, asks: list) -> float:
        """Advance the selected feature by one event; NaN when not defined (one-sided book etc.)."""
        kind = self.kind
        if kind == K_FLOW_EV:
            hist = self.hist
            self.acc += flow
            if len(hist) >= self.n:
                self.acc -= hist.popleft()
            hist.append(flow)
            return self.acc
        if kind == K_FLOW_MS:
            hist = self.hist
            if flow:
                hist.append((now, flow))
                self.acc += flow
            edge = now - self.n
            while hist and hist[0][0] <= edge:
                self.acc -= hist.popleft()[1]
            return self.acc
        two_sided = bool(bids) and bool(asks)
        if kind == K_IMB:
            if not two_sided:
                return math.nan
            if self.n == 1:
                return _imb(_depth(bids, 1), _depth(asks, 1))
            return _imb(_depth(bids, self.n), _depth(asks, self.n))
        if kind == K_MICRO:
            if not two_sided:
                return math.nan
            return (asks[0].price - bids[0].price) * _imb(_depth(bids, 1), _depth(asks, 1)) / 2
        if kind == K_CNT:
            if not two_sided:
                return math.nan
            bp, ap = bids[0].price, asks[0].price
            nb = 0
            for view in bids:
                if view.price != bp:
                    break
                nb += 1
            na = 0
            for view in asks:
                if view.price != ap:
                    break
                na += 1
            return _imb(nb, na)
        # mid-based features
        if two_sided:
            mid = (bids[0].price + asks[0].price) / 2
            if mid != self.mid_ff:
                if self.mid_ff == self.mid_ff:
                    self.last_dmid = mid - self.mid_ff
                self.mid_ff = mid
        if kind == K_RET:
            items = self.hist
            items.append((now, self.mid_ff))
            edge = now - self.n
            while len(items) >= 2 and items[1][0] <= edge:
                items.popleft()
            return self.mid_ff - items[0][1] if items[0][0] <= edge else math.nan
        if kind == K_LTP:
            return self.ltp - self.mid_ff if self.ltp > 0 and two_sided else 0.0
        return self.last_dmid  # K_DMID

    # ------------------------------------------------------------------ trading

    def _ioc(self, side: Side, price: int, qty: int) -> None:
        self.side = side
        self.in_flight = True
        self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=OrderType.IOC))

    def _exit(self, bid: int, ask: int) -> None:
        if self.inv > 0:
            self._ioc(Side.SELL, max(1, bid - self.SLIP_TICKS), self.inv)
        else:
            self._ioc(Side.BUY, ask + self.SLIP_TICKS, -self.inv)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        if self.start_us < 0:
            self.start_us = now
        bids, asks = snapshot.bids, snapshot.asks
        if self.kind == K_LIN:
            market = [t for t in trades if t.taker_owner == 'market']
            x = self.full.update(snapshot, market)
            s = math.nan if x is None else _predict_linear(self.LINEAR_MODEL, x)
        else:
            flow = 0
            for t in trades:
                if t.taker_owner == 'market':
                    flow += t.quantity if t.book_side == 'ask' else -t.quantity
                    self.ltp = t.price
            s = self._signal(now, flow, bids, asks)
        if self.SIGN < 0:
            s = -s
        self.sig = s
        if self.in_flight or not bids or not asks:
            return
        bid, ask = bids[0].price, asks[0].price
        th = self.THRESHOLD
        inv = self.inv
        if inv != 0:
            if now - self.entry_us >= self.HOLD_US or (self.EXIT_ON_REVERSE and (s < -th if inv > 0 else s > th)):
                self._exit(bid, ask)
                return
        if now - self.last_entry_us < self.COOLDOWN_US:
            return
        if s > th and inv >= 0:
            qty = min(self.QTY, self.MAX_POS - inv)
            if qty > 0:
                self.last_entry_us = now
                self._ioc(Side.BUY, ask + self.SLIP_TICKS, qty)
        elif s < -th and inv <= 0:
            qty = min(self.QTY, self.MAX_POS + inv)
            if qty > 0:
                self.last_entry_us = now
                self._ioc(Side.SELL, max(1, bid - self.SLIP_TICKS), qty)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            was_flat = self.inv == 0
            q = event.filled_quantity
            self.inv += q if self.side == Side.BUY else -q
            if was_flat and self.inv != 0:
                self.entry_us = event.timestamp_us
        if st == OrderStatus.FILLED or st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
            self.in_flight = False
