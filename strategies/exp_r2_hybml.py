"""exp_r2_hybml (home, R2): mm_r2_hybrid (maker + hidden-fair filter + fair IOC taker) + the ML shrink of exp_r2_ml.

exp_r2_ml lifted the gate-only maker from 86k to 115-117k on round2 by pulling (ml_stub 0) or shrinking the maker
quote on the side a ridge model predicts will be run over (target B: mid move after the next taker hit; model B75 =
fitted on round2's first 75 %). Here it is stacked on the hybrid.
  * ml_thr / ml_stub: pred < -ml_thr caps the bid quote at ml_stub lots (modify the oldest order down, cancel newer
    ones), pred > +ml_thr the ask. ml_thr 0 = off (identical to mm_r2_hybrid). Features: research.signals copy.
  * take_ml_agree / take_ml_min: 1 = the taker enters only when s * pred > take_ml_min (s = +1 buy, -1 sell),
    i.e. the ML prediction agrees with the hidden-fair signal. 0 = off.
Result (ledger r2-hybml, -agree, -ctrl): parity ML off = 204,054; take/fair off = exp_r2_ml 116,287. Best ml_thr 0.12
ml_stub 0 = 212,375 (+8.3k) but all of it in the train window the B75 model was fitted on (hold -1.3k, 4/8 paired
chunks), sample_v2 -36k (0/8), round1 +0.7k: compare REJECT. take_ml_agree: noise (+-1.5k). Defaults stay ML off.

mm_r2_hybrid (home, R2 candidate): exp_r2_taker baked: maker (30 s gate thr 6 + hidden-fair filter 0.45, fair over 20
levels, mid at any spread) + IOC taker on |fair_sig| > 1.5 that exits when the signal fades. round2 204,054 (8/8 chunks).

exp_r2_taker (home, R2): hidden-fair TAKER (+ optional maker) on top of exp_r2_fair.

Offline (research/diag/fair_taker.py + scratch taker2): entering with an IOC when |fair_sig| > 1.0-1.5 (tau 50 ms) and
leaving with an IOC once the signal fades below take_exit or after take_maxhold_ms made +118-121k on round2, 8/8 chunks.
  * take_thr (0 = off), take_exit, take_maxhold_ms, take_qty, take_slip (entry limit ticks through the touch),
    exit_slip, take_cool_ms; maker_off 1 = no maker quotes at all (pure taker).
One taker position at a time; the position is the inventory itself (pure-taker mode), entries and exits go through
_send(ioc=True), so the I + OUT <= M invariant still holds.

exp_r2_fair (home, R2): mm_r2_gate + the hidden-fair filter (radys findings s.11).

The generator places new limit orders around a hidden fair price, so every new market limit order is a noisy look at
it: dev = price - side offset - mid before the insert (offsets ~ -1.2 bids / +1.2 asks on round2, estimated online).
sig = time-decayed mean of dev (fair_tau_ms) over new inserts, found as market order ids above the highest id seen
in the first fair_levels price levels of each side (a new limit order is the last order of its level).
research/diag/fair_fills.py: our maker fills with the signal against us (buy with sig < 0, sell with sig > 0) lose
-305k at 1 s on the plain maker, those with it on our side win +106k.
  * fair_thr: pull the bid while sig < -fair_thr, the ask while sig > +fair_thr. 0 = off (identical to mm_r2_gate).

mm_r2_gate (home, R2 candidate): exp_r2_ladder with gate_thr 4, gate_max_spread 3, park_size 0, gate_rungs 1 baked in.
round2 86,333 (8/8 chunks, chunk_min +95, holdout 44,739). A passive trend follower: the against-trend side is pulled
while the robust 30 s mid drift is beyond 4 ticks, so inventory rides the minute-scale drift.

exp_r2_ladder (home, R2): exp_ladder_home (mm_park + standing rungs) + the round-2 trend gate.

Why (round2 anatomy, research/diag/queue_age.py): market orders live ~0.3 s, so an order parked at a
level >= 1 s before that level becomes the touch is at the very FRONT of the queue 84 % of the time
(2 s: 97 %). The front-of-queue oracle on round2 is worth ~+150k for the gate maker (84k -> 232k).
  * gate_thr / gate_window_us / gate_max_spread: drift = mid change over gate_window_us, sampled only
    while the spread is <= gate_max_spread (round2 has transient one-sided gaps of 10-80 ticks that are
    63 % of the mid variance; they must not trip the gate). |drift| >= gate_thr pulls the against-trend
    side (bid when falling, ask when rising). gate_thr 0 = off (identical to exp_ladder_home).
  * gate_rungs: 1 = the gate also cancels that side's rungs and parked stubs.

exp_ladder (H11): mm_park + standing "ladder" rungs behind the touch that are never moved.

  * ladder_size / ladder_depth / ladder_reserve: per side keep one ladder_size-lot order at each of
    touch-1 .. touch-ladder_depth (mirrored on the ask), placed once and then left alone while inside the
    window, so when the price arrives they are already at the front of the new touch queue (queue position
    is the edge: 0 lots ahead +0.21 $/lot, 500+ ahead loses). Rungs deeper than the window, or better than
    the touch, are cancelled. Rungs count in OUT_k; ladder_reserve 1 = touch quote first, rungs from what is
    left; 0 = rungs first, the touch quote takes the rest. ladder_size 0 = off (identical to mm_park).
    The invariant gate in _send is untouched.

mm_park (radys): exp_full_capacity + parked 20-lot stubs (park_size 20, park_depth 1, park_max 3) baked in.
round1 local pnl 452,651 vs champion 425,708.

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
from math import exp, floor

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
        self.rung = False  # a standing ladder order
        self.parked = False  # a shrunk stub left behind the touch for its queue priority
        self.target = -1  # pending modify target of the ML cap (-1 = none)
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
        'park_size': 0,
        'park_depth': 1,
        'park_max': 3,
        'ladder_size': 0,
        'ladder_depth': 3,
        'ladder_reserve': 1,
        'gate_thr': 6.0,
        'gate_window_us': 30000000,
        'gate_max_spread': 3,
        'gate_rungs': 1,
        'fair_thr': 0.45,
        'fair_tau_ms': 50.0,
        'fair_levels': 20,
        'take_thr': 1.5,
        'take_exit': 0.0,
        'take_maxhold_ms': 100000000,
        'take_qty': 100,
        'take_slip': 0,
        'exit_slip': 0,
        'take_cool_ms': 0,
        'maker_off': 0,
        'fair_mid_any': 1,
        'fair_off_fixed': 0,
        # ML shrink (model LINEAR_MODEL_B75): cap the side predicted to be run over at ml_stub lots; 0 = off.
        'ml_thr': 0.0,
        'ml_stub': 0,
        # Taker entry only when s * pred > take_ml_min (ML agrees with the fair signal); 0 = off.
        'take_ml_agree': 0,
        'take_ml_min': 0.0,
    }

    # Ridge on research.signals features, target B (mid move after the first taker hit x [hit within 100 ms]),
    # fitted on round2's first 75 % (copied from exp_r2_ml.py).
    LINEAR_MODEL_B75: dict = {'intercept': -0.00525626, 'coef': {'spread': -0.00101443, 'imb1': 0.0613574, 'imb2': -0.00715758, 'imb3': -0.00645482, 'imb5': 0.00561995, 'micro': -0.0628429, 'cnt_imb1': 0.0863342, 'depth1_log': 0.00442554, 'flow_ev1': 0.000861316, 'flow_ev10': 0.00669514, 'flow_ev100': 0.0240596, 'flow_ms10': 0.0051096, 'flow_ms100': 0.0381025, 'flow_ms1000': -0.0116887, 'flow_ms10000': -0.00310393, 'ntr_ms1000': 0.0134946, 'nev_ms1000': 0.0117272, 'ret_ms100': 0.0704035, 'ret_ms1000': -0.00215148, 'ret_ms10000': -0.00231434, 'last_dmid': 0.040987, 'dt_mid_log': 0.0150266, 'ltp_dev': 0.0290398}, 'mean': {'spread': 1.20442, 'imb1': -0.00467601, 'imb2': -0.000496138, 'imb3': -0.000165381, 'imb5': -0.00148991, 'micro': -0.000539375, 'cnt_imb1': -0.00982661, 'depth1_log': 7.31927, 'flow_ev1': -0.033544, 'flow_ev10': -0.32983, 'flow_ev100': -4.22505, 'flow_ms10': -0.206216, 'flow_ms100': -1.93961, 'flow_ms1000': -18.06, 'flow_ms10000': -165.104, 'ntr_ms1000': 33.5031, 'nev_ms1000': 368.57, 'ret_ms100': -0.00982258, 'ret_ms1000': -0.10587, 'ret_ms10000': -1.2045, 'last_dmid': 0.000384619, 'dt_mid_log': 5.55252, 'ltp_dev': 0.00408715}, 'scale': {'spread': 0.540603, 'imb1': 0.635533, 'imb2': 0.425312, 'imb3': 0.405798, 'imb5': 0.405819, 'micro': 0.41103, 'cnt_imb1': 0.53239, 'depth1_log': 1.01757, 'flow_ev1': 23.4874, 'flow_ev10': 117.203, 'flow_ev100': 419.502, 'flow_ms10': 76.8783, 'flow_ms100': 279.574, 'flow_ms1000': 839.807, 'flow_ms10000': 2647.6, 'ntr_ms1000': 18.2601, 'nev_ms1000': 30.8893, 'ret_ms100': 0.671553, 'ret_ms1000': 2.98278, 'ret_ms10000': 11.6427, 'last_dmid': 0.552562, 'dt_mid_log': 1.72892, 'ltp_dev': 0.665426}, 'lo': {'spread': 1.0, 'imb1': -0.999154, 'imb2': -0.979798, 'imb3': -0.960764, 'imb5': -0.961447, 'micro': -1.83515, 'cnt_imb1': -0.958333, 'depth1_log': 2.48491, 'flow_ev1': -373.573, 'flow_ev10': -1239.0, 'flow_ev100': -2660.0, 'flow_ms10': -967.0, 'flow_ms100': -2108.0, 'flow_ms1000': -3772.15, 'flow_ms10000': -9205.0, 'ntr_ms1000': 1.0, 'nev_ms1000': 111.0, 'ret_ms100': -6.0, 'ret_ms1000': -33.2865, 'ret_ms10000': -102.0, 'last_dmid': -2.0, 'dt_mid_log': 0.0, 'ltp_dev': -3.0}, 'hi': {'spread': 6.0, 'imb1': 0.999106, 'imb2': 0.983439, 'imb3': 0.967306, 'imb5': 0.966805, 'micro': 1.89895, 'cnt_imb1': 0.957447, 'depth1_log': 10.0216, 'flow_ev1': 360.0, 'flow_ev10': 1227.0, 'flow_ev100': 2377.0, 'flow_ms10': 915.0, 'flow_ms100': 1997.0, 'flow_ms1000': 3845.0, 'flow_ms10000': 8813.0, 'ntr_ms1000': 128.0, 'nev_ms1000': 450.0, 'ret_ms100': 5.5, 'ret_ms1000': 22.5, 'ret_ms10000': 71.5, 'last_dmid': 2.0, 'dt_mid_log': 8.60845, 'ltp_dev': 3.5}, 'horizon_ms': 100, 'latency_ms': 5}

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
        self.ladder_size = max(0, int(params['ladder_size']))
        self.ladder_depth = max(1, int(params['ladder_depth']))
        self.ladder_reserve = int(params['ladder_reserve'])
        self.rungs: list[dict[int, str]] = [{}, {}]  # price -> order id of standing rungs
        self.rung_qty = [0, 0]
        self.gate_thr = float(params['gate_thr'])
        self.gate_window = int(params['gate_window_us'])
        self.gate_max_spread = int(params['gate_max_spread'])
        self.gate_rungs = int(params['gate_rungs'])
        self.gate_pts: deque = deque()
        self.gate_last_ts = _NEVER_US
        self.gate_mid2 = 0
        self.fair_thr = float(params['fair_thr'])
        self.fair_tau_us = max(1.0, float(params['fair_tau_ms']) * 1000.0)
        self.fair_levels = max(1, int(params['fair_levels']))
        self.fair_max_id = 0
        self.fair_s = 0.0
        self.fair_w = 0.0
        self.fair_t = 0
        self.fair_sig = 0.0
        self.fair_off = [-1.2, 1.2]  # running mean of (insert price - mid before) for bids / asks
        self.fair_n = [0, 0]
        self.fair_prev_mid2 = 0
        self.take_thr = float(params['take_thr'])
        self.take_exit = float(params['take_exit'])
        self.take_maxhold_us = int(params['take_maxhold_ms']) * 1000
        self.take_qty = max(1, int(params['take_qty']))
        self.take_slip = int(params['take_slip'])
        self.exit_slip = int(params['exit_slip'])
        self.take_cool_us = int(params['take_cool_ms']) * 1000
        self.maker_off = int(params['maker_off'])
        self.fair_mid_any = int(params['fair_mid_any'])
        if int(params['fair_off_fixed']):
            self.fair_off = [-1.22, 1.20]
            self.fair_n = [1 << 40, 1 << 40]
        self.tk_dir = 0
        self.tk_t = 0
        self.tk_last = _NEVER_US
        self.tk_pending = None
        self.n_takes = 0
        self.ml_thr = float(params['ml_thr'])
        self.ml_stub = max(0, int(params['ml_stub']))
        self.take_ml_agree = int(params['take_ml_agree'])
        self.take_ml_min = float(params['take_ml_min'])
        self.full = _FullFeatures() if self.ml_thr > 0.0 or self.take_ml_agree else None
        self.model = self.LINEAR_MODEL_B75
        self.pred = 0.0

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
        need_qty = self.alpha_imbalance != 0.0 or self.imb_pull > 0.0 or self.jump_imb > 0.0 or self.thin_pull > 0

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

        if self.fair_thr > 0.0 or self.take_thr > 0.0:
            self._fair_update(snapshot.timestamp_us, bids, asks, bb, ba)
        if self.take_thr > 0.0:
            self._taker(snapshot.timestamp_us, bb, ba)
        if self.maker_off:
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
        if self.fair_thr > 0.0:
            if self.fair_sig < -self.fair_thr:
                pull_bid = True
            elif self.fair_sig > self.fair_thr:
                pull_ask = True
        if self.gate_thr > 0.0:
            drift = self._drift(now, bb, ba)
            if drift <= -self.gate_thr:
                pull_bid = True
                if self.gate_rungs:
                    self._clear_side(0, 1)
            elif drift >= self.gate_thr:
                pull_ask = True
                if self.gate_rungs:
                    self._clear_side(1, -1)
        lay = self.layer_size
        off = self.layer_offset
        ladder = self.ladder_size > 0
        if ladder and not self.ladder_reserve:
            self._ladder(0, 1, _BUY, bb, ask, pull_bid)
            self._ladder(1, -1, _SELL, ba, bid, pull_ask)
        if pull_bid:
            self._pull(0)
            self._pull(2)
        else:
            sz = self.ml_stub if self.ml_thr > 0.0 and self.pred < -self.ml_thr else self.quote_size
            self._quote(0, 1, _BUY, bid, now, sz)
            if lay:
                self._quote(2, 1, _BUY, bid - off, now, lay)
        if pull_ask:
            self._pull(1)
            self._pull(3)
        else:
            sz = self.ml_stub if self.ml_thr > 0.0 and self.pred > self.ml_thr else self.quote_size
            self._quote(1, -1, _SELL, ask, now, sz)
            if lay:
                self._quote(3, -1, _SELL, ask + off, now, lay)
        if ladder and self.ladder_reserve:
            self._ladder(0, 1, _BUY, bb, ask, pull_bid)
            self._ladder(1, -1, _SELL, ba, bid, pull_ask)

    def _taker(self, now: int, bb: int, ba: int) -> None:
        """One IOC position at a time: enter on a strong hidden-fair signal, leave when it fades."""
        if self.tk_pending is not None:
            if self.tk_pending in self.orders:
                return  # the last IOC is still in flight
            self.tk_pending = None
        inv = self.inv
        if self.tk_dir != 0:
            s = self.tk_dir
            if inv * s <= 0:
                self.tk_dir = 0  # flat again (exit done, or the entry never filled)
                self.tk_last = now
                return
            if s * self.fair_sig < self.take_exit or now - self.tk_t >= self.take_maxhold_us:
                if s > 0:
                    self.tk_pending = self._send(1, -1, _SELL, max(1, bb - self.exit_slip), inv, True, 1)
                else:
                    self.tk_pending = self._send(0, 1, _BUY, ba + self.exit_slip, -inv, True, 0)
            return
        if now - self.tk_last < self.take_cool_us or ba - bb > 3:
            return
        sig = self.fair_sig
        agree = self.take_ml_agree
        if sig > self.take_thr and (not agree or self.pred > self.take_ml_min):
            oid = self._send(0, 1, _BUY, ba + self.take_slip, self.take_qty, True, 0)
            s = 1
        elif sig < -self.take_thr and (not agree or self.pred < -self.take_ml_min):
            oid = self._send(1, -1, _SELL, max(1, bb - self.take_slip), self.take_qty, True, 1)
            s = -1
        else:
            return
        if oid is not None:
            self.tk_pending = oid
            self.tk_dir = s
            self.tk_t = now
            self.n_takes += 1

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
            return 0.0  # not enough history yet
        return (self.gate_mid2 - pts[0][1]) / 2.0

    def _clear_side(self, k: int, sign: int) -> None:
        """Cancel the side's rungs and parked stubs (the gate says the price runs through them)."""
        rungs = self.rungs[k]
        orders = self.orders
        for px in list(rungs):
            order = orders[rungs[px]]
            order.rung = False
            del rungs[px]
            self.rung_qty[k] -= order.remaining
            if not order.cancel_sent:
                order.cancel_sent = True
                self.cancel_order(order.order_id)
                self.n_cancels += 1
        for order_id in list(self.parked[k]):
            self._unpark(order_id)

    def _ladder(self, k: int, sign: int, side: Side, touch: int, other_quote: int, pulled: bool) -> None:
        """Cancel rungs outside touch..touch-depth, then fill missing rungs from the free capacity."""
        rungs = self.rungs[k]
        depth = self.ladder_depth
        if rungs:
            orders = self.orders
            for px in list(rungs):
                d = (touch - px) * sign
                if d < 0 or d > depth:
                    order = orders[rungs[px]]
                    order.rung = False
                    del rungs[px]
                    self.rung_qty[k] -= order.remaining
                    if not order.cancel_sent:
                        order.cancel_sent = True
                        self.cancel_order(order.order_id)
                        self.n_cancels += 1
        if pulled:
            return
        for i in range(1, depth + 1):
            px = touch - sign * i
            if px in rungs or px <= 0 or (other_quote - px) * sign <= 0:
                continue
            if self.active[k] and self.quote_px[k] == px:
                continue
            if self.parked[k] and any(self.orders[o].price == px for o in self.parked[k]):
                continue
            free = self.max_pos - sign * self.inv - self.out[k]
            if free <= 0:
                return
            order_id = self._send(k, sign, side, px, min(self.ladder_size, free), False, 4)
            if order_id is not None:
                rungs[px] = order_id

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
            if self.out[k] > self.act_qty[k] + self.act_qty[k + 2] + self.parked_qty[k] + self.rung_qty[k]:
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
        if j == 4:
            o = self.orders[order_id]
            o.active = False
            o.rung = True
            self.rung_qty[k] += qty
        elif not ioc:
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
        if order.rung:
            self.rung_qty[order.k] -= qty

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
        if order.rung:
            order.rung = False
            del self.rungs[order.k][order.price]
            self.rung_qty[order.k] -= order.remaining
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
