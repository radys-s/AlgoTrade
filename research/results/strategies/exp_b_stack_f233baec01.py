"""mm_ml_b (home): mm_ml with the loss-aligned ridge target B (mid move after the first taker hit x [hit within 100 ms]), ml_thr 0.105, ml_stub 3 baked. round1 503,148 vs mm_ml 490,460 (+12,688, 7/8 paired chunks, holdout +3,397); sample_v2 258,042 (+3,981, 6/8). Plateau thr 0.097-0.115 (499-503k).

mm_ml (radys): mm_park + ridge 100 ms signal shrinks the side predicted to be run over to a 5-lot stub
(ml_thr 0.15 ticks, ml_stub 5) baked in. round1 local pnl 490,460 vs mm_park 452,651, 8/8 chunks.

exp_ml_shrink (radys): exp_shrink + a walk-forward ridge model that SHRINKS (never cancels) the side it
predicts will be run over.

  * ml_thr / ml_stub: pred = predicted mid change over the next 100 ms (ticks, ridge on research.signals
    features, IC +0.38 in all 7 walk-forward folds on round1). pred < -ml_thr: cap the bid quote at ml_stub
    lots (modify the oldest order down, cancel newer ones), pred > +ml_thr: same for the ask. The oldest
    order keeps its queue place; the quote is topped up again (at the back) once the signal fades.
    ml_thr 0 = off. Features: pasted copy of research.signals.OnlineFeatures (via signal_taker.py).

exp_shrink (radys): exp_queue + park a stub instead of cancelling when the market improves past us.

  * park_size / park_depth / park_max: when the touch quote must move to a better price, the old order is
    shrunk with modify_order to park_size lots (keeps its queue priority, simulator reduces in place)
    instead of being cancelled; the freed room goes to the new touch quote. A parked stub is cancelled
    once it is more than park_depth ticks behind the market touch, or when the side has more than
    park_max stubs (oldest first). round1: fills with 0 lots ahead earn +0.21 $/lot, every cancel
    sends us to the back of a ~900-lot queue. park_size 0 = off.

Built from exp_queue (radys): exp_flow_pull + thin-level pull.

  * thin_pull: pull a side when the MARKET quantity at its touch (our own orders excluded) is below
    thin_pull lots, and quote it again only once it is back above thin_pull * thin_rejoin (hysteresis).
    round1 fill analysis: our fills on levels with < 100 market lots have 1 s edge -0.12..-0.49 $/lot
    (the level is about to be depleted and the price moves through us), > 1000 lots +0.32. 0 = off.

Built from exp_flow_pull (radys): exp_layers + pull after big aggressive trades.

  * flow_pull_qty / flow_pull_us: when one book update carries >= flow_pull_qty lots of
    buyer-initiated volume (book_side 'ask'), pull our ask for flow_pull_us (and the bid after
    seller-initiated volume). round1: mid moves +0.52 tick in the aggressor's direction within 1 s
    after p90+ trades, so the side the flow hits next is adversely selected. 0 = off.

Built from exp_layers (radys): exp_imb_pull + a second resting level per side.

  * layer_size / layer_offset: besides the touch quote, keep up to layer_size lots per side
    layer_offset ticks behind it (bid - offset / ask + offset). Big sweeps that reach it revert
    (sample_v2: -0.13 tick after big trades), so the deeper fill is cheaper. The invariant is
    unchanged: both levels count in OUT_k; the touch quote is served first. 0 = off.

Built from exp_imb_pull (radys): Honza's mm.py + adverse-selection pull filter.

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



class _Order:
    """Our record of one of our orders; `remaining` bounds what the exchange can still fill."""

    def __init__(self, order_id: str, k: int, sign: int, price: int, quantity: int, ioc: bool, slot: int) -> None:
        self.order_id = order_id
        self.k = k  # 0 = buy side, 1 = sell side
        self.slot = slot  # 0/1 touch quote, 2/3 second level (side = slot & 1)
        self.parked = False  # a shrunk stub left behind the touch for its queue priority
        self.target = -1  # pending modify target (-1 = none)
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
        # Spread == 2 and |imbalance| > jump_imb: improve ONLY the favoured side by 1 tick
        # (bid if bid-heavy, ask if ask-heavy), so bid < ask still holds; 0 disables.
        'jump_imb': 0.0,
        # Second level per side: lots and ticks behind the touch quote; layer_size 0 disables.
        'layer_size': 0,
        'layer_offset': 1,
        # Pull the side in the aggressor's way for flow_pull_us after >= flow_pull_qty lots; 0 disables.
        'flow_pull_qty': 0,
        'flow_pull_us': 0,
        # Pull a side while its market touch quantity < thin_pull lots; rejoin above thin_pull * thin_rejoin.
        'thin_pull': 0,
        'thin_rejoin': 1.0,
        # Park a stub of park_size lots instead of cancelling a quote the market moved past; 0 disables.
        'park_size': 20,
        'park_depth': 1,
        'park_max': 3,
        # ML shrink: cap the side predicted to be run over at ml_stub lots when |pred| > ml_thr ticks; 0 disables.
        'ml_thr': 0.105,
        'ml_stub': 3,
        # graded shrink: for ml_thr_lo < against-side |pred| <= ml_thr cap at ml_frac_lo x quote_size; 0 = off.
        'ml_thr_lo': 0.0,
        # ML stub only while the market spread is 1 tick (a wide-spread fill earns a full tick); 0 = always.
        'ml_tight_only': 0,
        'ml_frac_lo': 0.5,
        'ml_target': 'B',  # base | A | B | C: which LINEAR_MODEL(S) entry feeds the shrink
        # Thin-level cap: while a side's market touch qty < thin_cap lots, cap its quote at thin_cap_size (shrink,
        # never cancel the oldest order); 0 disables.
        'thin_cap': 0,
        'thin_cap_size': 20,
    }

    # H18 loss-aligned targets (research.fit_target, round1 full fit): A = weighted MSE k=10, B = move x hit, C = adverse-fill score
    LINEAR_MODELS: dict = {'A': {'intercept': -0.00011633, 'coef': {'spread': 0.0013991, 'imb1': 0.0792149, 'imb2': 0.043925, 'imb3': 0.0119941, 'imb5': -0.0593541, 'micro': -0.0344156, 'cnt_imb1': 0.0879723, 'depth1_log': 0.000281984, 'flow_ev1': 0.00118888, 'flow_ev10': 0.00368276, 'flow_ev100': 0.025835, 'flow_ms10': 0.000803595, 'flow_ms100': 0.00198066, 'flow_ms1000': 0.0341407, 'flow_ms10000': 0.00355985, 'ntr_ms1000': -0.000549619, 'nev_ms1000': 0.000324458, 'ret_ms100': 0.0133359, 'ret_ms1000': -0.014181, 'ret_ms10000': -0.00374638, 'last_dmid': 0.0166229, 'dt_mid_log': -0.000235338, 'ltp_dev': 0.0228302}, 'mean': {'spread': 1.0693, 'imb1': 0.00226445, 'imb2': 0.00180018, 'imb3': 0.00210598, 'imb5': 0.00196122, 'micro': 0.00110094, 'cnt_imb1': 0.00302462, 'depth1_log': 7.62745, 'flow_ev1': 0.00831957, 'flow_ev10': 0.109338, 'flow_ev100': 0.75778, 'flow_ms10': 0.0611908, 'flow_ms100': -0.133823, 'flow_ms1000': 1.45438, 'flow_ms10000': 14.4329, 'ntr_ms1000': 31.394, 'nev_ms1000': 369.081, 'ret_ms100': 0.000137964, 'ret_ms1000': 0.00183982, 'ret_ms10000': 0.0215042, 'last_dmid': -0.00232995, 'dt_mid_log': 5.96466, 'ltp_dev': -0.00145316}, 'scale': {'spread': 0.253972, 'imb1': 0.634089, 'imb2': 0.391809, 'imb3': 0.391049, 'imb5': 0.39463, 'micro': 0.332689, 'cnt_imb1': 0.552367, 'depth1_log': 0.724809, 'flow_ev1': 23.1121, 'flow_ev10': 115.745, 'flow_ev100': 414.842, 'flow_ms10': 75.3632, 'flow_ms100': 269.873, 'flow_ms1000': 829.43, 'flow_ms10000': 2626.82, 'ntr_ms1000': 16.4881, 'nev_ms1000': 25.5999, 'ret_ms100': 0.297793, 'ret_ms1000': 0.916508, 'ret_ms10000': 2.89814, 'last_dmid': 0.503705, 'dt_mid_log': 1.49873, 'ltp_dev': 0.55481}, 'lo': {'spread': 1.0, 'imb1': -0.999197, 'imb2': -0.944827, 'imb3': -0.948755, 'imb5': -0.952257, 'micro': -0.918063, 'cnt_imb1': -0.959184, 'depth1_log': 4.07754, 'flow_ev1': -365.0, 'flow_ev10': -1244.13, 'flow_ev100': -2659.0, 'flow_ms10': -972.0, 'flow_ms100': -2155.92, 'flow_ms1000': -3955.96, 'flow_ms10000': -9906.0, 'ntr_ms1000': 2.0, 'nev_ms1000': 294.0, 'ret_ms100': -1.5, 'ret_ms1000': -3.0, 'ret_ms10000': -9.0, 'last_dmid': -1.0, 'dt_mid_log': 0.0, 'ltp_dev': -1.5}, 'hi': {'spread': 2.0, 'imb1': 0.999224, 'imb2': 0.95081, 'imb3': 0.952935, 'imb5': 0.955548, 'micro': 0.91591, 'cnt_imb1': 0.959184, 'depth1_log': 10.1089, 'flow_ev1': 370.0, 'flow_ev10': 1261.13, 'flow_ev100': 2641.0, 'flow_ms10': 939.131, 'flow_ms100': 2090.0, 'flow_ms1000': 3824.0, 'flow_ms10000': 8906.0, 'ntr_ms1000': 98.0, 'nev_ms1000': 451.0, 'ret_ms100': 1.5, 'ret_ms1000': 3.0, 'ret_ms10000': 9.0, 'last_dmid': 1.0, 'dt_mid_log': 8.66873, 'ltp_dev': 1.5}, 'horizon_ms': 100, 'latency_ms': 5}, 'B': {'intercept': 0.000441121, 'coef': {'spread': 0.00133452, 'imb1': 0.0824714, 'imb2': 0.0149258, 'imb3': 0.0144349, 'imb5': -0.0312584, 'micro': -0.0520587, 'cnt_imb1': 0.0505402, 'depth1_log': -0.000421541, 'flow_ev1': -0.000244005, 'flow_ev10': -0.000286522, 'flow_ev100': 0.0151363, 'flow_ms10': -0.000948543, 'flow_ms100': 0.00504899, 'flow_ms1000': 0.0168779, 'flow_ms10000': 0.00168168, 'ntr_ms1000': -0.000859284, 'nev_ms1000': 0.000417404, 'ret_ms100': 0.00504642, 'ret_ms1000': -0.00788903, 'ret_ms10000': -0.0023049, 'last_dmid': 0.00922577, 'dt_mid_log': 0.000422585, 'ltp_dev': 0.00995797}, 'mean': {'spread': 1.0692, 'imb1': 0.00224751, 'imb2': 0.00174989, 'imb3': 0.00204918, 'imb5': 0.0019074, 'micro': 0.00107808, 'cnt_imb1': 0.00305099, 'depth1_log': 7.6274, 'flow_ev1': 0.0138849, 'flow_ev10': 0.0796944, 'flow_ev100': 0.633488, 'flow_ms10': -0.0203193, 'flow_ms100': -0.199753, 'flow_ms1000': 1.38836, 'flow_ms10000': 14.3423, 'ntr_ms1000': 31.3946, 'nev_ms1000': 369.079, 'ret_ms100': 0.000105061, 'ret_ms1000': 0.00183437, 'ret_ms10000': 0.0214286, 'last_dmid': -0.00237398, 'dt_mid_log': 5.96541, 'ltp_dev': -0.00162312}, 'scale': {'spread': 0.253788, 'imb1': 0.634086, 'imb2': 0.391867, 'imb3': 0.391076, 'imb5': 0.394647, 'micro': 0.332665, 'cnt_imb1': 0.552431, 'depth1_log': 0.724585, 'flow_ev1': 23.0386, 'flow_ev10': 115.5, 'flow_ev100': 414.757, 'flow_ms10': 74.7879, 'flow_ms100': 269.905, 'flow_ms1000': 829.333, 'flow_ms10000': 2626.78, 'ntr_ms1000': 16.4882, 'nev_ms1000': 25.5991, 'ret_ms100': 0.297626, 'ret_ms1000': 0.916424, 'ret_ms10000': 2.89809, 'last_dmid': 0.503711, 'dt_mid_log': 1.49741, 'ltp_dev': 0.554799}, 'lo': {'spread': 1.0, 'imb1': -0.999197, 'imb2': -0.944827, 'imb3': -0.948755, 'imb5': -0.952257, 'micro': -0.918063, 'cnt_imb1': -0.959184, 'depth1_log': 4.07754, 'flow_ev1': -365.0, 'flow_ev10': -1244.13, 'flow_ev100': -2659.0, 'flow_ms10': -972.0, 'flow_ms100': -2155.92, 'flow_ms1000': -3955.96, 'flow_ms10000': -9906.0, 'ntr_ms1000': 2.0, 'nev_ms1000': 294.0, 'ret_ms100': -1.5, 'ret_ms1000': -3.0, 'ret_ms10000': -9.0, 'last_dmid': -1.0, 'dt_mid_log': 0.0, 'ltp_dev': -1.5}, 'hi': {'spread': 2.0, 'imb1': 0.999224, 'imb2': 0.95081, 'imb3': 0.952935, 'imb5': 0.955548, 'micro': 0.91591, 'cnt_imb1': 0.959184, 'depth1_log': 10.1089, 'flow_ev1': 370.0, 'flow_ev10': 1261.13, 'flow_ev100': 2641.0, 'flow_ms10': 939.131, 'flow_ms100': 2090.0, 'flow_ms1000': 3824.0, 'flow_ms10000': 8906.0, 'ntr_ms1000': 98.0, 'nev_ms1000': 451.0, 'ret_ms100': 1.5, 'ret_ms1000': 3.0, 'ret_ms10000': 9.0, 'last_dmid': 1.0, 'dt_mid_log': 8.66873, 'ltp_dev': 1.5}, 'horizon_ms': 100, 'latency_ms': 5}, 'C': {'intercept': 0.00102837, 'coef': {'spread': -0.000957659, 'imb1': 0.00304119, 'imb2': 0.0136326, 'imb3': 0.0041803, 'imb5': -0.0203731, 'micro': -0.0167489, 'cnt_imb1': 0.084174, 'depth1_log': -6.01452e-05, 'flow_ev1': 0.000299281, 'flow_ev10': 0.002634, 'flow_ev100': 0.0215704, 'flow_ms10': -0.0010119, 'flow_ms100': 0.0118347, 'flow_ms1000': 0.0290867, 'flow_ms10000': 0.00208868, 'ntr_ms1000': 0.00189649, 'nev_ms1000': 0.000922184, 'ret_ms100': 0.0020093, 'ret_ms1000': -0.0172664, 'ret_ms10000': -0.00404831, 'last_dmid': 0.0139123, 'dt_mid_log': 0.00298355, 'ltp_dev': 0.0226772}, 'mean': {'spread': 1.0692, 'imb1': 0.00224751, 'imb2': 0.00174989, 'imb3': 0.00204918, 'imb5': 0.0019074, 'micro': 0.00107808, 'cnt_imb1': 0.00305099, 'depth1_log': 7.6274, 'flow_ev1': 0.0138849, 'flow_ev10': 0.0796944, 'flow_ev100': 0.633488, 'flow_ms10': -0.0203193, 'flow_ms100': -0.199753, 'flow_ms1000': 1.38836, 'flow_ms10000': 14.3423, 'ntr_ms1000': 31.3946, 'nev_ms1000': 369.079, 'ret_ms100': 0.000105061, 'ret_ms1000': 0.00183437, 'ret_ms10000': 0.0214286, 'last_dmid': -0.00237398, 'dt_mid_log': 5.96541, 'ltp_dev': -0.00162312}, 'scale': {'spread': 0.253788, 'imb1': 0.634086, 'imb2': 0.391867, 'imb3': 0.391076, 'imb5': 0.394647, 'micro': 0.332665, 'cnt_imb1': 0.552431, 'depth1_log': 0.724585, 'flow_ev1': 23.0386, 'flow_ev10': 115.5, 'flow_ev100': 414.757, 'flow_ms10': 74.7879, 'flow_ms100': 269.905, 'flow_ms1000': 829.333, 'flow_ms10000': 2626.78, 'ntr_ms1000': 16.4882, 'nev_ms1000': 25.5991, 'ret_ms100': 0.297626, 'ret_ms1000': 0.916424, 'ret_ms10000': 2.89809, 'last_dmid': 0.503711, 'dt_mid_log': 1.49741, 'ltp_dev': 0.554799}, 'lo': {'spread': 1.0, 'imb1': -0.999197, 'imb2': -0.944827, 'imb3': -0.948755, 'imb5': -0.952257, 'micro': -0.918063, 'cnt_imb1': -0.959184, 'depth1_log': 4.07754, 'flow_ev1': -365.0, 'flow_ev10': -1244.13, 'flow_ev100': -2659.0, 'flow_ms10': -972.0, 'flow_ms100': -2155.92, 'flow_ms1000': -3955.96, 'flow_ms10000': -9906.0, 'ntr_ms1000': 2.0, 'nev_ms1000': 294.0, 'ret_ms100': -1.5, 'ret_ms1000': -3.0, 'ret_ms10000': -9.0, 'last_dmid': -1.0, 'dt_mid_log': 0.0, 'ltp_dev': -1.5}, 'hi': {'spread': 2.0, 'imb1': 0.999224, 'imb2': 0.95081, 'imb3': 0.952935, 'imb5': 0.955548, 'micro': 0.91591, 'cnt_imb1': 0.959184, 'depth1_log': 10.1089, 'flow_ev1': 370.0, 'flow_ev10': 1261.13, 'flow_ev100': 2641.0, 'flow_ms10': 939.131, 'flow_ms100': 2090.0, 'flow_ms1000': 3824.0, 'flow_ms10000': 8906.0, 'ntr_ms1000': 98.0, 'nev_ms1000': 451.0, 'ret_ms100': 1.5, 'ret_ms1000': 3.0, 'ret_ms10000': 9.0, 'last_dmid': 1.0, 'dt_mid_log': 8.66873, 'ltp_dev': 1.5}, 'horizon_ms': 100, 'latency_ms': 5}}

    # research.signals data/features/round1.npz --horizon-ms 100 --export-linear (walk-forward ridge, IC +0.38)
    LINEAR_MODEL: dict = {'intercept': -0.00015815,
     'coef': {'spread': 0.000997825,
              'imb1': 0.0255576,
              'imb2': 0.0367096,
              'imb3': 0.0078605,
              'imb5': -0.0458227,
              'micro': -0.00206035,
              'cnt_imb1': 0.0705746,
              'depth1_log': 0.000425994,
              'flow_ev1': 0.00152928,
              'flow_ev10': 0.00413217,
              'flow_ev100': 0.0178795,
              'flow_ms10': 0.00143799,
              'flow_ms100': -0.00128259,
              'flow_ms1000': 0.0282251,
              'flow_ms10000': 0.00346013,
              'ntr_ms1000': -0.000652296,
              'nev_ms1000': 0.000189557,
              'ret_ms100': 0.0141121,
              'ret_ms1000': -0.0116711,
              'ret_ms10000': -0.0036369,
              'last_dmid': 0.0128142,
              'dt_mid_log': -9.06298e-05,
              'ltp_dev': 0.0202146},
     'mean': {'spread': 1.06919,
              'imb1': 0.00227439,
              'imb2': 0.00174741,
              'imb3': 0.00204263,
              'imb5': 0.00190086,
              'micro': 0.00109152,
              'cnt_imb1': 0.00307936,
              'depth1_log': 7.62739,
              'flow_ev1': 0.0138823,
              'flow_ev10': 0.0794335,
              'flow_ev100': 0.633076,
              'flow_ms10': -0.0203602,
              'flow_ms100': -0.20129,
              'flow_ms1000': 1.38783,
              'flow_ms10000': 14.4327,
              'ntr_ms1000': 31.3942,
              'nev_ms1000': 369.078,
              'ret_ms100': 0.000105055,
              'ret_ms1000': 0.00181113,
              'ret_ms10000': 0.0214824,
              'last_dmid': -0.0024013,
              'dt_mid_log': 5.96546,
              'ltp_dev': -0.00161418},
     'scale': {'spread': 0.253782,
               'imb1': 0.634088,
               'imb2': 0.391863,
               'imb3': 0.391071,
               'imb5': 0.394643,
               'micro': 0.332665,
               'cnt_imb1': 0.552434,
               'depth1_log': 0.724594,
               'flow_ev1': 23.038,
               'flow_ev10': 115.496,
               'flow_ev100': 414.749,
               'flow_ms10': 74.7858,
               'flow_ms100': 269.898,
               'flow_ms1000': 829.313,
               'flow_ms10000': 2626.73,
               'ntr_ms1000': 16.4879,
               'nev_ms1000': 25.5992,
               'ret_ms100': 0.297618,
               'ret_ms1000': 0.916412,
               'ret_ms10000': 2.89802,
               'last_dmid': 0.503711,
               'dt_mid_log': 1.49739,
               'ltp_dev': 0.554796},
     'lo': {'spread': 1.0,
            'imb1': -0.999197,
            'imb2': -0.944827,
            'imb3': -0.948754,
            'imb5': -0.952257,
            'micro': -0.918056,
            'cnt_imb1': -0.959184,
            'depth1_log': 4.07754,
            'flow_ev1': -365.0,
            'flow_ev10': -1244.1,
            'flow_ev100': -2659.0,
            'flow_ms10': -972.0,
            'flow_ms100': -2155.71,
            'flow_ms1000': -3955.51,
            'flow_ms10000': -9906.0,
            'ntr_ms1000': 2.0,
            'nev_ms1000': 294.0,
            'ret_ms100': -1.5,
            'ret_ms1000': -3.0,
            'ret_ms10000': -9.0,
            'last_dmid': -1.0,
            'dt_mid_log': 0.0,
            'ltp_dev': -1.5},
     'hi': {'spread': 2.0,
            'imb1': 0.999224,
            'imb2': 0.95081,
            'imb3': 0.952933,
            'imb5': 0.955548,
            'micro': 0.915903,
            'cnt_imb1': 0.959184,
            'depth1_log': 10.1089,
            'flow_ev1': 370.0,
            'flow_ev10': 1261.1,
            'flow_ev100': 2641.0,
            'flow_ms10': 939.101,
            'flow_ms100': 2090.0,
            'flow_ms1000': 3824.0,
            'flow_ms10000': 8906.0,
            'ntr_ms1000': 98.0,
            'nev_ms1000': 451.0,
            'ret_ms100': 1.5,
            'ret_ms1000': 3.0,
            'ret_ms10000': 9.0,
            'last_dmid': 1.0,
            'dt_mid_log': 8.66873,
            'ltp_dev': 1.5},
     'horizon_ms': 100,
     'latency_ms': 5,
     'source': 'data\\round1.csv',
     'rows': 2203598}

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
        self.jump_imb = float(params['jump_imb'])
        self.layer_size = max(0, int(params['layer_size']))
        self.layer_offset = max(1, int(params['layer_offset']))
        self.flow_pull_qty = int(params['flow_pull_qty'])
        self.flow_pull_us = int(params['flow_pull_us'])
        self.pull_until = [_NEVER_US, _NEVER_US]  # [bid, ask]
        self.thin_pull = int(params['thin_pull'])
        self.thin_rejoin = float(params['thin_rejoin'])
        self.thin_out = [False, False]  # side currently pulled by the thin-level rule
        self.park_size = max(0, int(params['park_size']))
        self.park_depth = max(0, int(params['park_depth']))
        self.park_max = max(1, int(params['park_max']))
        self.parked: list[list[str]] = [[], []]  # ids of parked stubs per side, oldest first
        self.parked_qty = [0, 0]
        self.ml_thr = float(params['ml_thr'])
        self.ml_stub = max(0, int(params['ml_stub']))
        self.ml_thr_lo = float(params['ml_thr_lo'])
        self.ml_tight_only = int(params['ml_tight_only'])
        self.ml_frac_lo = float(params['ml_frac_lo'])
        self.thin_cap = int(params['thin_cap'])
        self.thin_cap_size = max(1, int(params['thin_cap_size']))
        self.full = _FullFeatures() if self.ml_thr > 0 else None
        self.pred = 0.0
        tgt = str(params['ml_target'])
        self.model = self.LINEAR_MODEL if tgt == 'base' else self.LINEAR_MODELS[tgt]

        # Position and order state (index k: 0 = buy, 1 = sell).
        self.inv = 0
        self.cash = 0.0
        self.orders: dict[str, _Order] = {}
        self.out = [0, 0]  # OUT_k: remaining of every not-yet-dead order on the side
        self.act_qty = [0, 0, 0, 0]  # remaining of the side's active quote orders
        self.active: list[list[str]] = [[], [], [], []]  # ids of the side's active quote orders, all at quote_px[k]
        self.quote_px = [0, 0, 0, 0]
        self.last_action_us = [_NEVER_US, _NEVER_US, _NEVER_US, _NEVER_US]
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
        if self.full is not None:
            x = self.full.update(snapshot, [t for t in trades if t.taker_owner == 'market'])
            self.pred = 0.0 if x is None else _predict_linear(self.model, x)
        if not bids or not asks:
            return
        scan = self.book_scan_orders
        need_qty = (self.alpha_imbalance != 0.0 or self.imb_pull > 0.0 or self.jump_imb > 0.0 or self.thin_pull > 0
                    or self.thin_cap > 0)

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
        if self.jump_imb > 0.0 and ba - bb == 2 and qb + qa > 0:
            jimb = (qb - qa) / (qb + qa)
            if jimb > self.jump_imb:
                bid = bb + 1
            elif jimb < -self.jump_imb:
                ask = ba - 1
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
        pull_bid = pull_ask = False
        if self.imb_pull > 0.0 and qb + qa > 0:
            imb = (qb - qa) / (qb + qa)
            pull_bid = imb < -self.imb_pull
            pull_ask = imb > self.imb_pull
        if self.park_size > 0:
            for k, touch, sg in ((0, bb, 1), (1, ba, -1)):
                for order_id in list(self.parked[k]):
                    if (touch - self.orders[order_id].price) * sg > self.park_depth:
                        self._unpark(order_id)
        if self.flow_pull_qty > 0:
            buy_vol = 0
            sell_vol = 0
            for t in trades:
                if t.book_side == 'ask':
                    buy_vol += t.quantity
                else:
                    sell_vol += t.quantity
            if buy_vol >= self.flow_pull_qty:
                self.pull_until[1] = now + self.flow_pull_us
            if sell_vol >= self.flow_pull_qty:
                self.pull_until[0] = now + self.flow_pull_us
            pull_bid = pull_bid or now < self.pull_until[0]
            pull_ask = pull_ask or now < self.pull_until[1]
        if self.thin_pull > 0:
            lo = self.thin_pull
            hi = lo * self.thin_rejoin
            for k, qk in ((0, qb), (1, qa)):
                if self.thin_out[k]:
                    if qk >= hi:
                        self.thin_out[k] = False
                elif qk < lo:
                    self.thin_out[k] = True
            pull_bid = pull_bid or self.thin_out[0]
            pull_ask = pull_ask or self.thin_out[1]
        if self.inv_stop > 0:
            pull_bid = pull_bid or inv >= self.inv_stop
            pull_ask = pull_ask or inv <= -self.inv_stop
        lay = self.layer_size
        off = self.layer_offset
        if pull_bid:
            self._pull(0)
            self._pull(2)
        else:
            sz = self.quote_size if self.ml_tight_only and ba - bb > 1 else self._ml_cap(-self.pred)
            if self.thin_cap > 0 and qb < self.thin_cap and sz > self.thin_cap_size:
                sz = self.thin_cap_size
            self._quote(0, 1, _BUY, bid, now, sz)
            if lay:
                self._quote(2, 1, _BUY, bid - off, now, lay)
        if pull_ask:
            self._pull(1)
            self._pull(3)
        else:
            sz = self.quote_size if self.ml_tight_only and ba - bb > 1 else self._ml_cap(self.pred)
            if self.thin_cap > 0 and qa < self.thin_cap and sz > self.thin_cap_size:
                sz = self.thin_cap_size
            self._quote(1, -1, _SELL, ask, now, sz)
            if lay:
                self._quote(3, -1, _SELL, ask + off, now, lay)

    def _ml_cap(self, against: float) -> int:
        """Quote size for a side given the signed prediction against it (>0 = predicted to be run over)."""
        if self.full is None:
            return self.quote_size
        if against > self.ml_thr:
            return self.ml_stub
        if self.ml_thr_lo > 0 and against > self.ml_thr_lo:
            return max(self.ml_stub, int(self.ml_frac_lo * self.quote_size))
        return self.quote_size

    def _pull(self, k: int) -> None:  # k = slot
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

    def _quote(self, j: int, sign: int, side: Side, px: int, now: int, size: int) -> None:
        """Keep slot j's active quote at `px` with up to `size` lots, within the invariant (side k = j & 1)."""
        k = j & 1
        act = self.active[j]
        if act and self.quote_px[j] != px:
            behind = (px - self.quote_px[j]) * sign  # > 0: our quote is less aggressive than desired
            if behind > 0:
                if behind <= self.requote_tolerance or now - self.last_action_us[j] < self.min_requote_us:
                    return
            orders = self.orders
            park = self.park_size > 0 and behind > 0 and j < 2
            for order_id in act:
                order = orders[order_id]
                order.active = False
                if park:
                    if order.remaining > self.park_size:
                        self.modify_order(order_id, self.park_size)
                    order.parked = True
                    self.parked[k].append(order_id)
                    self.parked_qty[k] += order.remaining
                else:
                    order.cancel_sent = True
                    self.cancel_order(order_id)
                    self.n_cancels += 1
            act.clear()
            self.act_qty[j] = 0
            self.last_action_us[j] = now
            if park:
                while len(self.parked[k]) > self.park_max:
                    self._unpark(self.parked[k][0])

        have = self.act_qty[j]
        if have > size and act:
            self._cap(j, size)
            return
        want = size - have
        if want <= 0 or px <= 0:
            return
        if have > 0 and (want < self.topup_min or now - self.last_action_us[j] < self.min_requote_us):
            return
        free = self.max_pos - sign * self.inv - self.out[k]
        if want > free:
            if self.out[k] > self.act_qty[k] + self.act_qty[k + 2] + self.parked_qty[k]:
                return  # in-flight cancels/IOCs hold the room; wait for them rather than fragment the quote
            want = free
            if want <= 0:
                return
        if self._send(k, sign, side, px, want, False, j) is not None:
            self.quote_px[j] = px
            self.last_action_us[j] = now

    def _unwind(self, bb: int, ba: int) -> None:
        """Send one IOC at the touch towards zero inventory."""
        inv = self.inv
        if inv > 0:
            k, sign, side, px = 1, -1, _SELL, bb
        else:
            k, sign, side, px = 0, 1, _BUY, ba
        qty = min(self.unwind_size, abs(inv), self.max_pos - sign * inv - self.out[k])
        if qty > 0:
            self.unwind_id = self._send(k, sign, side, px, qty, True, k)
            if self.unwind_id is not None:
                self.n_unwinds += 1

    def _send(self, k: int, sign: int, side: Side, px: int, qty: int, ioc: bool, j: int) -> str | None:
        """The only place orders are created: refuse anything that could break the invariant."""
        if qty <= 0 or px <= 0 or sign * self.inv + self.out[k] + qty > self.max_pos:
            self.n_blocked += 1
            return None
        order_id = self.send_order(
            OrderRequest(side=side, price=px, quantity=qty, order_type=_IOC if ioc else _LIMIT)
        )
        self.orders[order_id] = _Order(order_id, k, sign, px, qty, ioc, j)
        self.out[k] += qty
        if not ioc:
            self.active[j].append(order_id)
            self.act_qty[j] += qty
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
            self.act_qty[order.slot] -= qty
        if order.parked:
            self.parked_qty[order.k] -= qty

    def _cap(self, j: int, size: int) -> None:
        """Shrink slot j's active quote to `size` lots: keep the oldest order (queue place), modify it down,
        cancel newer ones. Pending modifies are remembered so they are not resent every event."""
        act = self.active[j]
        orders = self.orders
        keep = size
        for order_id in list(act):
            order = orders[order_id]
            eff = order.remaining if order.target < 0 else min(order.remaining, order.target)
            if keep <= 0:
                order.active = False
                act.remove(order_id)
                self.act_qty[j] -= order.remaining
                order.cancel_sent = True
                self.cancel_order(order_id)
                self.n_cancels += 1
            elif eff > keep:
                if order.target != keep:
                    order.target = keep
                    self.modify_order(order_id, keep)
                keep = 0
            else:
                keep -= eff

    def _unpark(self, order_id: str) -> None:
        """Cancel a parked stub (it stays counted in OUT until CANCELED arrives)."""
        order = self.orders[order_id]
        order.parked = False
        self.parked[order.k].remove(order_id)
        self.parked_qty[order.k] -= order.remaining
        if not order.cancel_sent:
            order.cancel_sent = True
            self.cancel_order(order_id)
            self.n_cancels += 1

    def _forget(self, order: _Order) -> None:
        """Drop a dead order from the book-keeping."""
        self.orders.pop(order.order_id, None)
        if order.parked:
            order.parked = False
            self.parked[order.k].remove(order.order_id)
            self.parked_qty[order.k] -= order.remaining
        if order.active:
            order.active = False
            self.active[order.slot].remove(order.order_id)
            self.act_qty[order.slot] -= order.remaining
        if order.order_id == self.unwind_id:
            self.unwind_id = None
