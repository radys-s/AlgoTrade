"""exp_ml_taker (H10): exp_full_capacity + an IOC overlay that crosses the spread on a large 1 s forecast.

Hypothesis: a linear forecast of the 1 s mid move (ridge, IC ~0.40 walk-forward on round1; top 1 %
predicted moves ~ +/-0.95 ticks) is worth the half-spread when it is large, because the inventory an
IOC at the touch creates is unwound PASSIVELY by the maker side (earning the spread), whereas
research.taker_scan paid the spread twice (IOC in and out) and found nothing.
Model: 23 OnlineFeatures (book imbalances, signed flow windows, mid returns, ...) on market orders
only, horizon 1000 ms after the 5 ms latency, fit on the first 75 % of round1 in time (compare's
holdout stays clean). Overlay is off at taker_thr = 0 (identical to exp_full_capacity); the
invariant gate in `_send` still decides every quantity.

Base docstring, exp_full_capacity (home): mm_v2 with quote_size 200 = quote ALL free capacity on each side.

Why: the invariant (B) I + OUT_buy <= M and (S) -I + OUT_sell <= M lets the side that REDUCES
|I| carry up to M + |I| lots (200 when |I| = 100), but quote_size 100 capped it at 100. With
quote_size above max_position, `_quote` sends min(quote_size - have, free) lots, so when flat both
sides quote 100 as before and when long 60 the ask carries 160: more volume exactly where the
strategy is stuck at the cap. sample_v2: 237,306 -> 263,618 (+11 %), holdout 54,929 -> 61,917,
8/8 chunks, breach 0; 200 and 300 are identical (saturated at M + |I|). Nothing else changed.

Original mm_v2 docstring: sample_v2 sweep best baked in (quote 100, max_pos 100, skew 0, filters off).
Built from exp_imb_pull: Honza's mm.py + adverse-selection pull filter.

Change vs mm.py (both off by default = identical to mm.py):
  * imb_pull: when L1 imbalance (qb - qa)/(qb + qa) < -imb_pull, pull the bid (price about to
    fall through it); when > +imb_pull, pull the ask. sample_v2: L1 imbalance corr with the
    100 ms forward mid is +0.35, so the side we pull is the one about to be run over.
  * inv_stop: stop quoting the side that would grow |I| once |I| >= inv_stop.

Original mm.py docstring:
Parametric two-sided market maker whose position limit is enforced by construction.

Quotes one passive LIMIT order per side at (or one tick inside) the market touch, shifted
by an inventory skew and an optional L1-imbalance alpha. Every tunable lives in
`MarketMaker.PARAMS` (the team runner overrides it by subclassing); `__init__` reads it
only through `self.PARAMS[...]`.

HARD INVARIANT (why the 10,000/s penalty cannot happen)
-------------------------------------------------------
Let M = max_position (clamped to <= 100, the exchange limit), I = our inventory, and for
side k let OUT_k be the sum of `remaining` over every order of ours on that side that is
not yet known to be dead: in flight to the exchange, resting, or resting with a cancel in
flight (a cancel can lose the race against a fill, so it counts until CANCELED arrives).
With s_buy = +1 and s_sell = -1 we keep, at every moment our code can observe,

    (B)  I + OUT_buy  <= M        and        (S)  -I + OUT_sell <= M.

Proof by induction over the only events that change I or OUT:
  * `_send` (the only place an order is created) refuses any q with s*I + OUT_k + q > M,
    so a new order keeps (B)/(S).
  * Buy fill of f: I += f, OUT_buy -= f, so (B) is unchanged and -I + OUT_sell drops by f.
    Sell fill symmetric. The simulator reports each fill synchronously, after updating its
    own inventory and before anything else can happen, so our I equals the exchange's I.
  * CANCELED, REJECTED, MODIFIED and IOC residuals only lower OUT_k.
  * A market cancel never touches our orders, and an order's remaining quantity never grows.
`remaining` here is an upper bound on what the exchange can still fill, so every fill that
can ever happen is already counted in OUT. Since OUT_k >= 0, (B) and (S) give
-M <= I <= M at all times: |I| <= M <= 100, hence zero breach penalty, even if every
outstanding order fills.
The reject rule |I + s(R + q)| > 1000 can never fire either: R + q <= OUT_k (+q), so
s*I + R + q <= M <= 100, and s*I >= -M, so the value lies in [-100, 100].
"""

import math
from collections import deque
from math import floor

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

# Exchange rules, not tunables: |I| > 100 costs 10,000 per second.
EXCHANGE_MAX_POSITION = 100

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

_FLOW_EVENTS = (1, 10, 100)
_FLOW_MS = (10, 100, 1000, 10000)
_RET_MS = (100, 1000, 10000)
_N_LEVELS = 5


def _agg(views):
    """(best price, order count at level 1, quantities of the first 5 prices), market orders only."""
    q = [0] * _N_LEVELS
    best = price = -1
    level = 0
    count1 = 0
    for view in views:
        if view.owner != 'market':
            continue
        vp = view.price
        if best < 0:
            best = price = vp
        elif vp != price:
            level += 1
            if level == _N_LEVELS:
                break
            price = vp
        q[level] += view.quantity
        if level == 0:
            count1 += 1
    return (best if best > 0 else 0), count1, q


class OnlineFeatures:
    """Pure-Python, O(1)-per-event version of `build_matrix` for use inside a strategy.

    Call `update(snapshot, trades)` from `on_order_book_change`; it returns the feature dict (None
    while the book is one-sided). Same names and definitions as the vectorized code.
    """

    def __init__(
        self, flow_events=_FLOW_EVENTS, flow_ms=_FLOW_MS, ret_ms=_RET_MS
    ) -> None:
        self.flow_events = flow_events
        self.flow_ms = flow_ms
        self.ret_ms = ret_ms
        self._ev_hist = deque(maxlen=max(flow_events))
        self._ev_sums = [0.0] * len(flow_events)
        self._flow_windows = [(t * 1000, deque(), [0.0]) for t in flow_ms]  # (window, deque of (ts, v), [sum])
        self._fills = deque()
        self._fills_sum = 0
        self._events = deque()
        self._mid_hist = [(t * 1000, deque()) for t in ret_ms]  # (window, deque of (ts, mid))
        self._mid_ff = math.nan
        self._last_dmid = 0.0
        self._last_change_us = None
        self._last_trade_px = 0

    def update(self, snapshot, trades):
        """Advance by one event and return its features (None when the book is one-sided)."""
        t = snapshot.timestamp_us
        if self._last_change_us is None:
            self._last_change_us = t
        flow = 0.0
        for trade in trades:
            if trade.taker_owner != 'market' or trade.maker_owner != 'market':
                continue  # the model was fit on the market-only tape
            flow += trade.quantity if trade.book_side == 'ask' else -trade.quantity
            self._last_trade_px = trade.price

        # event-count windows: sums over the last N values of a shared history
        hist = self._ev_hist
        for k, n in enumerate(self.flow_events):
            self._ev_sums[k] += flow
            if len(hist) >= n:
                self._ev_sums[k] -= hist[-n]
        hist.append(flow)

        # time windows (t - T, t]
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

        bid_px, bid_n, bq = _agg(snapshot.bids)
        ask_px, ask_n, aq = _agg(snapshot.asks)
        two_sided = bid_px > 0 and ask_px > 0
        if two_sided:
            mid = (bid_px + ask_px) / 2
            if mid != self._mid_ff:
                if self._mid_ff == self._mid_ff:  # not NaN: a real change
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

        def imb(b: float, a: float) -> float:
            return (b - a) / (b + a) if b + a > 0 else 0.0

        spread = float(ask_px - bid_px)
        imb1 = imb(bq[0], aq[0])
        out = {
            'spread': spread,
            'imb1': imb1,
            'imb2': imb(bq[0] + bq[1], aq[0] + aq[1]),
            'imb3': imb(bq[0] + bq[1] + bq[2], aq[0] + aq[1] + aq[2]),
            'imb5': imb(sum(bq), sum(aq)),
            'micro': spread * imb1 / 2,
            'cnt_imb1': imb(bid_n, ask_n),
            'depth1_log': math.log1p(bq[0] + aq[0]),
        }
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
        out['ltp_dev'] = self._last_trade_px - mid if self._last_trade_px > 0 else 0.0
        return out


LINEAR_MODEL = {
    'intercept': 0.00362456,
    'coef': {
        'spread': -0.00313078,
        'imb1': -0.00868267,
        'imb2': 0.108489,
        'imb3': -0.0224505,
        'imb5': -0.0922771,
        'micro': -0.0451887,
        'cnt_imb1': 0.310264,
        'depth1_log': 0.00542437,
        'flow_ev1': 0.00194793,
        'flow_ev10': 0.0090444,
        'flow_ev100': 0.0908511,
        'flow_ms10': -0.000540677,
        'flow_ms100': 0.040903,
        'flow_ms1000': 0.0993554,
        'flow_ms10000': 0.0114559,
        'ntr_ms1000': 0.0119599,
        'nev_ms1000': 0.00341265,
        'ret_ms100': 0.00825009,
        'ret_ms1000': -0.053335,
        'ret_ms10000': -0.0252344,
        'last_dmid': 0.0518657,
        'dt_mid_log': 0.00825181,
        'ltp_dev': 0.0802835,
    },
    'mean': {
        'spread': 1.06999,
        'imb1': 0.000475875,
        'imb2': 0.00356199,
        'imb3': 0.00423584,
        'imb5': 0.00432833,
        'micro': 0.000209107,
        'cnt_imb1': 0.000966727,
        'depth1_log': 7.6231,
        'flow_ev1': 0.0239112,
        'flow_ev10': 0.0937916,
        'flow_ev100': 0.664073,
        'flow_ms10': 0.00380114,
        'flow_ms100': -0.268813,
        'flow_ms1000': 1.29352,
        'flow_ms10000': 11.3065,
        'ntr_ms1000': 31.4203,
        'nev_ms1000': 369.008,
        'ret_ms100': 0.00033933,
        'ret_ms1000': 0.0035171,
        'ret_ms10000': 0.0366733,
        'last_dmid': 0.00119356,
        'dt_mid_log': 5.95607,
        'ltp_dev': 6.65946e-05,
    },
    'scale': {
        'spread': 0.255129,
        'imb1': 0.63368,
        'imb2': 0.391897,
        'imb3': 0.39072,
        'imb5': 0.394236,
        'micro': 0.332395,
        'cnt_imb1': 0.553213,
        'depth1_log': 0.725397,
        'flow_ev1': 22.9349,
        'flow_ev10': 116.464,
        'flow_ev100': 418.813,
        'flow_ms10': 75.0446,
        'flow_ms100': 272.69,
        'flow_ms1000': 839.767,
        'flow_ms10000': 2660.64,
        'ntr_ms1000': 16.4536,
        'nev_ms1000': 25.6225,
        'ret_ms100': 0.299193,
        'ret_ms1000': 0.921958,
        'ret_ms10000': 2.89269,
        'last_dmid': 0.503511,
        'dt_mid_log': 1.49714,
        'ltp_dev': 0.554515,
    },
    'lo': {
        'spread': 1.0,
        'imb1': -0.999228,
        'imb2': -0.942137,
        'imb3': -0.948878,
        'imb5': -0.952993,
        'micro': -0.909409,
        'cnt_imb1': -0.958333,
        'depth1_log': 4.11087,
        'flow_ev1': -358.0,
        'flow_ev10': -1263.83,
        'flow_ev100': -2693.81,
        'flow_ms10': -961.81,
        'flow_ms100': -2262.0,
        'flow_ms1000': -4239.0,
        'flow_ms10000': -10206.0,
        'ntr_ms1000': 2.0,
        'nev_ms1000': 294.0,
        'ret_ms100': -1.5,
        'ret_ms1000': -3.0,
        'ret_ms10000': -9.0,
        'last_dmid': -1.0,
        'dt_mid_log': 0.0,
        'ltp_dev': -1.5,
    },
    'hi': {
        'spread': 2.0,
        'imb1': 0.999223,
        'imb2': 0.94965,
        'imb3': 0.950978,
        'imb5': 0.954728,
        'micro': 0.910879,
        'cnt_imb1': 0.96,
        'depth1_log': 10.0205,
        'flow_ev1': 367.405,
        'flow_ev10': 1272.0,
        'flow_ev100': 2692.0,
        'flow_ms10': 930.025,
        'flow_ms100': 2071.0,
        'flow_ms1000': 3950.0,
        'flow_ms10000': 9135.13,
        'ntr_ms1000': 98.0,
        'nev_ms1000': 450.0,
        'ret_ms100': 1.5,
        'ret_ms1000': 3.0,
        'ret_ms10000': 9.0,
        'last_dmid': 1.0,
        'dt_mid_log': 8.63287,
        'ltp_dev': 1.5,
    },
    'horizon_ms': 1000,
    'latency_ms': 5,
    'source': 'data\round1.csv',
    'rows': 1651786,
}


def _predict(model, x):
    """Linear forecast in ticks (NaN while the return windows are still warming up)."""
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



class _Order:
    """Our record of one of our orders; `remaining` bounds what the exchange can still fill."""

    def __init__(self, order_id: str, k: int, sign: int, price: int, quantity: int, ioc: bool) -> None:
        self.order_id = order_id
        self.k = k  # 0 = buy side, 1 = sell side
        self.sign = sign  # +1 buy, -1 sell
        self.price = price
        self.remaining = quantity
        self.ioc = ioc
        self.accepted = False
        self.cancel_sent = False
        self.active = not ioc  # part of the side's current quote (not cancelling, not an IOC)


class MarketMaker(BaseStrategy):
    """Two-sided passive quoting with inventory skew; see the module docstring for the invariant."""

    PARAMS = {
        # Lots per side we try to keep resting at the quote price.
        'quote_size': 200,
        # Hard cap M on inventory + same-side outstanding quantity (clamped to <= 100). Below the
        # exchange's 100 so a stuck inventory costs less to mark out at the end.
        'max_position': 100,
        # Improve the touch by one tick on both sides when the market spread is at least this
        # many ticks; 0 disables (default: join only). Values 1-2 act as 3, so our bid stays
        # strictly below our ask.
        'improve_min_spread': 0,
        # Fair-value shift in ticks per lot of inventory: both quotes move by -round(k * I)
        # (rounded half away from zero, together with the alpha term below).
        'skew_per_lot': 0.0,
        # Fair-value shift in ticks per unit of L1 imbalance (qb - qa) / (qb + qa), market orders only.
        'alpha_imbalance': 0.0,
        # Minimum time between moving a quote towards the market / topping it up, per side.
        # Retreats (our quote more aggressive than desired) are never throttled.
        'min_requote_us': 0,
        # Keep a quote that is at most this many ticks less aggressive than desired (queue priority).
        'requote_tolerance': 0,
        # Top a partly filled quote back up only when it is short by at least this many lots.
        'topup_min': 1,
        # When |I| >= this, send an IOC at the touch towards zero; 0 disables.
        'unwind_threshold': 0,
        # Lots per unwind IOC.
        'unwind_size': 5,
        # Max market orders per side summed into the L1 quantity for the imbalance (ours are skipped).
        'book_scan_orders': 1000,
        # Pull a side when |L1 imbalance| > imb_pull against it; 0 disables.
        'imb_pull': 0.0,
        # Stop quoting the side that grows |I| once |I| >= inv_stop; 0 disables.
        'inv_stop': 0,
        # Cross the spread with an IOC when the 1 s forecast (ticks) exceeds this; 0 disables.
        'taker_thr': 0.0,
        'taker_qty': 50,
        'taker_cooldown_us': 200000,
        # Only send an IOC when |I| after a full fill would stay within this.
        'taker_max_inv': 100,
        # Evaluate the forecast on every N-th event (features are still updated every event).
        'taker_every': 1,
        # Lots per side the quotes leave free for IOCs (the invariant counts resting quotes, so
        # full-capacity quotes would leave no room to cross); -1 = taker_qty. Only when taker_thr > 0.
        'taker_reserve': -1,
    }

    def __init__(self) -> None:
        super().__init__()
        params = self.PARAMS
        self.quote_size = max(0, int(params['quote_size']))
        self.max_pos = max(0, min(int(params['max_position']), EXCHANGE_MAX_POSITION))
        improve_min_spread = int(params['improve_min_spread'])
        self.improve_min_spread = max(3, improve_min_spread) if improve_min_spread > 0 else 1 << 62
        self.skew_per_lot = float(params['skew_per_lot'])
        self.alpha_imbalance = float(params['alpha_imbalance'])
        self.min_requote_us = int(params['min_requote_us'])
        self.requote_tolerance = max(0, int(params['requote_tolerance']))
        self.topup_min = max(1, int(params['topup_min']))
        self.unwind_threshold = int(params['unwind_threshold'])
        self.unwind_size = int(params['unwind_size'])
        self.book_scan_orders = max(1, int(params['book_scan_orders']))
        self.imb_pull = float(params['imb_pull'])
        self.inv_stop = int(params['inv_stop'])
        self.taker_thr = float(params['taker_thr'])
        self.taker_qty = int(params['taker_qty'])
        self.taker_cooldown_us = int(params['taker_cooldown_us'])
        self.taker_max_inv = int(params['taker_max_inv'])
        self.taker_every = max(1, int(params['taker_every']))
        reserve = int(params['taker_reserve'])
        self.reserve = (self.taker_qty if reserve < 0 else reserve) if self.taker_thr > 0.0 else 0
        self.feat = OnlineFeatures() if self.taker_thr > 0.0 else None
        self.taker_last_us = [_NEVER_US, _NEVER_US]
        self.taker_tick = 0
        self.n_taker = 0
        self.taker_lots = 0

        # Position and order state (index k: 0 = buy, 1 = sell).
        self.inv = 0
        self.cash = 0.0
        self.orders: dict[str, _Order] = {}
        self.out = [0, 0]  # OUT_k: remaining of every not-yet-dead order on the side
        self.act_qty = [0, 0]  # remaining of the side's active quote orders
        self.active: list[list[str]] = [[], []]  # ids of the side's active quote orders, all at quote_px[k]
        self.quote_px = [0, 0]
        self.last_action_us = [_NEVER_US, _NEVER_US]
        self.unwind_id: str | None = None

        # Debug counters.
        self.n_orders = 0
        self.n_fills = 0
        self.fill_volume = 0
        self.n_cancels = 0
        self.n_rejects = 0  # order placements rejected by the exchange
        self.n_cancel_rejects = 0  # cancels that lost the race against a fill
        self.n_blocked = 0  # sends refused by the invariant gate
        self.n_unwinds = 0
        self.invariant_breaks = 0  # must stay 0
        self.max_abs_inv = 0

    # ------------------------------------------------------------------ market feed

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Recompute the desired bid/ask from the market touch and move our quotes if needed."""
        bids = snapshot.bids
        asks = snapshot.asks
        feat = self.feat
        forecast = 0.0
        if feat is not None:
            # Updated before the one-sided early return so the windows see every event.
            x = feat.update(snapshot, trades)
            self.taker_tick += 1
            if x is not None and self.taker_tick % self.taker_every == 0:
                forecast = _predict(LINEAR_MODEL, x)
                if forecast != forecast:
                    forecast = 0.0
        if not bids or not asks:
            return
        scan = self.book_scan_orders
        need_qty = self.alpha_imbalance != 0.0 or self.imb_pull > 0.0

        # Market touch and its quantity, skipping our own orders (only market orders count
        # towards the scan limit, so our own fragments can never hide the touch).
        bb = -1
        qb = 0
        n = 0
        for view in bids:
            if view.owner != 'market':
                continue
            if bb < 0:
                bb = view.price
                if not need_qty:
                    break
            elif view.price != bb:
                break
            qb += view.quantity
            n += 1
            if n >= scan:
                break
        if bb < 0:
            return
        ba = -1
        qa = 0
        n = 0
        for view in asks:
            if view.owner != 'market':
                continue
            if ba < 0:
                ba = view.price
                if not need_qty:
                    break
            elif view.price != ba:
                break
            qa += view.quantity
            n += 1
            if n >= scan:
                break
        if ba < 0:
            return

        if ba - bb >= self.improve_min_spread:
            bid = bb + 1
            ask = ba - 1
        else:
            bid = bb
            ask = ba
        inv = self.inv
        fair_shift = -self.skew_per_lot * inv
        if need_qty and qb + qa > 0:
            fair_shift += self.alpha_imbalance * (qb - qa) / (qb + qa)
        # Round half away from zero: equal step widths, and the same for long and short.
        shift = floor(fair_shift + 0.5) if fair_shift >= 0.0 else -floor(0.5 - fair_shift)
        if shift:
            bid += shift
            ask += shift
        # Never cross the market (clamping only widens our own spread, so bid < ask holds).
        if bid >= ba:
            bid = ba - 1
        if ask <= bb:
            ask = bb + 1

        now = snapshot.timestamp_us
        if self.unwind_threshold > 0 and self.unwind_id is None and abs(inv) >= self.unwind_threshold:
            self._unwind(bb, ba)
        if forecast > self.taker_thr or forecast < -self.taker_thr:
            self._take(forecast, bb, ba, now)
        pull_bid = pull_ask = False
        if self.imb_pull > 0.0 and qb + qa > 0:
            imb = (qb - qa) / (qb + qa)
            pull_bid = imb < -self.imb_pull
            pull_ask = imb > self.imb_pull
        if self.inv_stop > 0:
            pull_bid = pull_bid or inv >= self.inv_stop
            pull_ask = pull_ask or inv <= -self.inv_stop
        if pull_bid:
            self._pull(0)
        else:
            self._quote(0, 1, _BUY, bid, now)
        if pull_ask:
            self._pull(1)
        else:
            self._quote(1, -1, _SELL, ask, now)

    def _take(self, forecast: float, bb: int, ba: int, now: int) -> None:
        """IOC at the touch in the direction of a large forecast; the maker side unwinds it passively."""
        if forecast > 0.0:
            k, sign, side, px = 0, 1, _BUY, ba
        else:
            k, sign, side, px = 1, -1, _SELL, bb
        if now - self.taker_last_us[k] < self.taker_cooldown_us:
            return
        room = self.taker_max_inv - sign * self.inv - self.out[k]
        qty = min(self.taker_qty, self.max_pos - sign * self.inv - self.out[k], room)
        if qty <= 0:
            return
        if self._send(k, sign, side, px, qty, True) is not None:
            self.taker_last_us[k] = now
            self.n_taker += 1
            self.taker_lots += qty

    def _pull(self, k: int) -> None:
        """Cancel the side's active quote (retreats are never throttled)."""
        act = self.active[k]
        if not act:
            return
        orders = self.orders
        for order_id in act:
            order = orders[order_id]
            order.active = False
            order.cancel_sent = True
            self.cancel_order(order_id)
        self.n_cancels += len(act)
        act.clear()
        self.act_qty[k] = 0

    def _quote(self, k: int, sign: int, side: Side, px: int, now: int) -> None:
        """Keep the side's active quote at `px` with up to quote_size lots, within the invariant."""
        act = self.active[k]
        if act and self.quote_px[k] != px:
            behind = (px - self.quote_px[k]) * sign  # > 0: our quote is less aggressive than desired
            if behind > 0:
                if behind <= self.requote_tolerance or now - self.last_action_us[k] < self.min_requote_us:
                    return
            orders = self.orders
            for order_id in act:
                order = orders[order_id]
                order.active = False
                order.cancel_sent = True
                self.cancel_order(order_id)
            self.n_cancels += len(act)
            act.clear()
            self.act_qty[k] = 0
            self.last_action_us[k] = now

        have = self.act_qty[k]
        want = self.quote_size - have
        if want <= 0 or px <= 0:
            return
        if have > 0 and (want < self.topup_min or now - self.last_action_us[k] < self.min_requote_us):
            return
        free = self.max_pos - self.reserve - sign * self.inv - self.out[k]
        if want > free:
            if self.out[k] > have:
                return  # in-flight cancels/IOCs hold the room; wait for them rather than fragment the quote
            want = free
            if want <= 0:
                return
        if self._send(k, sign, side, px, want, False) is not None:
            self.quote_px[k] = px
            self.last_action_us[k] = now

    def _unwind(self, bb: int, ba: int) -> None:
        """Send one IOC at the touch towards zero inventory."""
        inv = self.inv
        if inv > 0:
            k, sign, side, px = 1, -1, _SELL, bb
        else:
            k, sign, side, px = 0, 1, _BUY, ba
        qty = min(self.unwind_size, abs(inv), self.max_pos - sign * inv - self.out[k])
        if qty > 0:
            self.unwind_id = self._send(k, sign, side, px, qty, True)
            if self.unwind_id is not None:
                self.n_unwinds += 1

    def _send(self, k: int, sign: int, side: Side, px: int, qty: int, ioc: bool) -> str | None:
        """The only place orders are created: refuse anything that could break the invariant."""
        if qty <= 0 or px <= 0 or sign * self.inv + self.out[k] + qty > self.max_pos:
            self.n_blocked += 1
            return None
        order_id = self.send_order(
            OrderRequest(side=side, price=px, quantity=qty, order_type=_IOC if ioc else _LIMIT)
        )
        self.orders[order_id] = _Order(order_id, k, sign, px, qty, ioc)
        self.out[k] += qty
        if not ioc:
            self.active[k].append(order_id)
            self.act_qty[k] += qty
        self.n_orders += 1
        return order_id

    # ------------------------------------------------------------------ private feed

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Book fills into inventory/cash and retire dead orders from the outstanding totals."""
        status = event.status
        order = self.orders.get(event.order_id)
        if order is None:
            if status is _REJECTED:
                self.n_cancel_rejects += 1  # cancel of an order that had already filled
            return

        if status is _FILLED or status is _PARTIALLY_FILLED:
            qty = event.filled_quantity
            sign = order.sign
            self.inv += sign * qty
            self.cash -= sign * qty * event.fill_price
            self.n_fills += 1
            self.fill_volume += qty
            self._shrink(order, qty)
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
            self._shrink(order, order.remaining)
            self._forget(order)
        elif status is _MODIFIED:
            drop = order.remaining - event.remaining_quantity
            if drop > 0:
                self._shrink(order, drop)
        elif status is _REJECTED:
            if not order.accepted:
                self.n_rejects += 1  # placement rejected: it never rested
            elif event.remaining_quantity > 0:
                return  # a rejected modify: the order is still resting unchanged
            else:
                self.n_cancel_rejects += 1  # exchange says it is no longer resting
            self._shrink(order, order.remaining)
            self._forget(order)

    def _shrink(self, order: _Order, qty: int) -> None:
        """Lower an order's remaining quantity and the side totals that include it."""
        order.remaining -= qty
        self.out[order.k] -= qty
        if order.active:
            self.act_qty[order.k] -= qty

    def _forget(self, order: _Order) -> None:
        """Drop a dead order from the book-keeping."""
        self.orders.pop(order.order_id, None)
        if order.active:
            order.active = False
            self.active[order.k].remove(order.order_id)
            self.act_qty[order.k] -= order.remaining
        if order.order_id == self.unwind_id:
            self.unwind_id = None
