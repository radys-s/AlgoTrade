"""exp_r3_mlp2 = mm_r3_mlp + flag mlp2_on (seed ensemble / extra MLP inputs / 2 horizons; off = identical).
mm_r3_mlp (home, R3 upload candidate): mm_r3_g2 with the taker ridge replaced by an MLP (22-32-16-1 ReLU on the same 22 features,
200 ms target, trained on round3 + mirror first 75 %), pred_blend 0.85, MLP evaluated when |ridge| >= 0.3.
Subagent: round3 549,721 / mirror 556,404 / round2 426,333; holdout 135,145 / 130,359 (g2 117,658 / 117,867).

mm_r3_g2 (home, R3 upload candidate): mm_r3_pms + passive filters retuned for R3 (rt_ms 100, fq_k_sp2 1.5, gate_thr 10, qs_k 0.6)
+ taker ridge refitted for a 200 ms target on round3 + mirror (first 75 %, rfit 19) at pred_blend 0.7.
round3 478,572 / mirror 477,772 / round2 426,845; holdout 117,658 / 117,867 (mm_r3_pms 104,902 / 105,585), 8/8 chunks.

mm_r3_pms (home, R3 upload candidate): mm_r3_ins + a ridge predictor refitted on round 3 (first 75 %, 150 ms target, lean
22 features) driving the IOC taker (pms_on 1, pred_blend 0.5, pred_scale 1.0; passive quotes keep fair_sig).
Subagent pms: round3 413,750 / mirror 411,861 (base 339,285), holdout 104,902 / 105,585 (base 87,007), 4/4 chunks.

mm_r3_ins (home, R3 insurance): our R2 final (mm_r2_final, R2 leaderboard 974,031) with the round-2 ridge switched off (pred_on 0).
round3 339,285 / round3_mirror 339,285 (holdout 87,007, 7/8 chunks); with the R2 ridge 338,970 / 342,176 but holdout
80,507 / 79,129; without the R2 passive filters 273-279k.

mm_r2_final (home, R2 last upload): mm_r2_rth (leaderboard 970,101) + passive refinements only, taker unchanged
(pred_blend 0.5): qs_k 0.3, fq_k_sp2 0.75, fq_k_sp3 0.15, hk_k 0.6. round2 434,477 / mirror 432,324 in the subagent grid
(+3.7k / +3.0k over mm_r2_rth).

mm_r2_rth2 (home, R2 leaderboard candidate): mm_r2_rth with qs_k 0.3 and fq_k_sp3 0.3.

round2 + round2_mirror 863,925 vs 860,077 for mm_r2_rth (leaderboard 970,101); both tapes up (+0.8k..+2.3k), i.e. a
small refinement: the 200-lot flip quote from a 0.3-tick lean (was 0.45), and at a spread >= 3 the lean threshold 0.3
(spread 2 keeps 0.7).

mm_r2_rth (home, R2 leaderboard candidate): mm_r2_rts + a stricter quote over a hole behind the touch.

hk_gap 3 / hk_k 0.8: when the next market level behind our side's touch is >= 3 ticks away, the touch quote needs
F_hat >= 0.8 ticks inside it (fq_k 0.05 otherwise): a sweep that takes the level runs through the hole (fills where the
sweep went deeper mark out -1.5 ticks/lot). round2 + round2_mirror 860,077 vs 840,149 for mm_r2_rts (both tapes ~+10k);
hk_gap 2 858,284. mm_r2_rts: fq_k_sp2 0.7, rt_ms 20, qs_k 0.45 (leaderboard 957,133).

exp_r2_rt (home, R2): mm_r2_zd + no passive quote on a touch that just retreated (+ qs_k / qs_weak as exp_r2_qsw).

round2 anatomy: market fills at a level the touch RETREATED onto mark out -0.57 / -0.42 ticks (0-50 / 50-200 ms), at a
fresh level made by an improving insert +0.64. rt_ms: after the bid touch falls (ask touch rises) the side's passive
quote is pulled for rt_ms; 0 = off (identical to mm_r2_zd).

mm_r2_zd (home, R2 leaderboard candidate): mm_r2_pred (gate 8 / 30 s) + zero-delay take (the flip IOC goes out in the
same callback as the cancel of our blocking passive quote; a guard IOC trims any |I| > 100 at once) + zero-delay
requote toward the market (zd_quote 2), quote_size 200. Subagent A. round2 local: see progress.md (2026-11-07 22:55).
Leaderboard: mm_r2_pred 797,904.

exp_r2_new_a: mm_r2_pred2 + zero-delay take (zd_take). Base = mm_r2_pred2 (home, R2 candidate): mm_r2_pred with gate_window 20 s, gate_thr 10 (fq_take_k 0.2): robust across regimes:
round2 378,507 (+6.5k), round1 480,854 (+39k), sample_v2 267,440 vs mm_r2_pred. Leaderboard: mm_r2_fq2 740,067, mm_r2_pred 797,904.

mm_r2_pred (home, R2 candidate): exp_r2_pred baked with pred_on 1, pred_mode 1 (the prediction drives only the
IOC taker test; passive quotes keep fair_sig), pred_blend 0.5 (taker signal = 0.5 ridge + 0.5 fair_sig), pred_scale 1.0,
fq_take_k 0.2, pred_model 0 (ridge fitted on the first 75 % of round2). round2 371,961 (hold 106,650, 8/8),
mirror 369,664, round1 441,626, sample_v2 265,868.

exp_r2_pred (home, R2): mm_r2_fq2 with a better F_hat - mid predictor fed to the fair taker/quoter.

research/diag/oracle_fair.py: with the REAL 150 ms move instead of fair_sig the same strategy makes 935k, so the
predictor is the bottleneck. research/diag/pred_r2.py fits a ridge (no winsorizing: the edge is in the tails) on
fair_sig at taus 20/50/70/150/300 ms, its decayed insert count, touch queues/imbalance, trade flow, last trade and
the book-hole offset, target = robust mid (spread <= 3) at t+5ms+150ms minus the current mid (ticks, the unit _fq
adds to the mid). Holdout (last 25 % of round2) corr 0.417 vs fair_sig 0.392; mean move in the top 1/3/6 % of
|pred| 3.45/2.15/1.47 ticks vs 2.91/1.86/1.31 for |fair_sig|.
  * pred_on: 1 = _fq uses pred_scale * prediction instead of fair_sig. 0 = identical to mm_r2_fq2.
  * pred_model: 0 = ridge fitted on the first 75 % of round2 (honest holdout), 1 = fitted on 100 %,
    2 = the same features with a 100 ms target (first 75 %).
  * pred_blend: the signal used is pred_blend * prediction + (1 - pred_blend) * fair_sig (1 = the prediction only).
  * pred_mode: 0 = the prediction drives both the IOC taker test and the passive quotes, 1 = taker test only
    (quotes keep fair_sig), 2 = quotes only (the taker keeps fair_sig).
Features use only market flow (our own IOC trades are skipped) and market orders at the touch.

mm_r2_fq2 (home, R2 candidate): mm_r2_fq with tau 70 ms, fq_take 200 (one IOC can reverse a full position),
fq_take_k 0.2, gate_thr 8. round2 349,077 (8/8, chunk_min +15.8k).

mm_r2_fq (home, R2 candidate): exp_r2_fq baked with the subagent's best cell: aggressive hidden-fair IOC taker (100 lots
whenever F_hat is beyond the opposite touch by 0.25 ticks, re-entry on every book update) + touch quotes only on the side F favours,
fair over 20 levels, mid at any spread, 30 s gate thr 6. round2 341,105 (8/8 chunks, chunk_min +14.6k), round1 377k, sample_v2 253k, check PASS 1.3 min.

exp_r2_fq (home, R2): exp_r2_fair turned into a FAIR-CENTRIC quoter.

The generator centres new limit orders and IOC limit prices on a hidden fair price F, and the visible mid lags it.
Instead of joining the touch and only pulling the side the signal is against, quote around the estimate
F_hat = mid + fq_scale * fair_sig (mid = the last normal mid, spread <= 3, so book holes do not move it):
  * fq_k: desired bid = floor(F_hat - fq_k), ask = ceil(F_hat + fq_k). Passive quotes are clamped to at most fq_inside
    ticks inside the market touch (never crossing it); a side whose price would be more than fq_max_behind ticks
    behind the market touch is not quoted. fq_k 0 = off (identical to exp_r2_fair under the same params).
  * fq_take / fq_slip / fq_take_cooldown_ms: when the desired bid reaches the market ask (F_hat far above), the
    passive bid is pulled (its room frees once CANCELED arrives) and an IOC buy at ask + fq_slip for up to fq_take
    lots goes out, at most once per cooldown, within the invariant; mirror for the ask. 0 = never cross (the quote
    stays clamped inside the spread).
  * fq_tol: keep a resting quote whose desired price is within fq_tol ticks (a cancel costs the ~700-lot queue).
  * fq_take_k: > 0 = the crossing test uses this half-width instead of fq_k (decouples the taker threshold).
  * fair_mid_any: 1 = the pre-insert mid is updated at any two-sided spread (lead's fix, strategies/exp_r2_taker.py);
    with fair_levels 20 the online signal matches the offline one. 0 = only while the spread is <= 3 (as exp_r2_fair).
  * fq_level: 1 = use (time-decayed mean of insert price - side offset) - mid instead of fair_sig, i.e. the
    devs are not measured against the mid at their own insert time (the mid catching up does not double count).

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

from collections import deque
from math import exp, floor, log1p

import numpy as np

# Ridge predictor of (robust mid at t+5ms+150ms) - mid now, folded to raw-feature weights: (c0, weights in PRED_NAMES order).
PRED_NAMES = ('fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fs70_x_lw', 'fs70_cube', 'sgnflow100_x_absfs', 'fs70_x_spread',
              'spread', 'imb1', 'cnt_imb1', 'lq_b', 'lq_a', 'ltr_sign', 'ltr_lsz', 'ltp_dev', 'hole', 'flow_ev10',
              'flow_ms100', 'flow_ms1000', 'dt_tr_log')
PRED_MODELS = (
    # fitted on the first 75 % of round2 (time), holdout corr 0.417
    (-0.1640373, (0.1537846, 1.5590139, -1.9111093, 0.5027856, -0.1723518, 0.1470229, 0.0134986, 0.2770717, 0.014809,
                  0.0323918, -0.1716682, 0.006399, 0.0677491, -0.0504011, -0.106348, 0.0074704, 0.0968138, -0.2549203,
                  0.0004209, 0.0001058, -1.2e-06, 0.0018721)),
    # fitted on 100 % of round2
    (-0.156208, (0.2649571, 1.9563702, -2.2357974, 0.6931888, -0.2452375, 0.0289781, 0.0098034, 0.2456548, 0.0117925,
                 0.0415263, -0.1733876, 0.0089595, 0.0649292, -0.0500369, -0.0969171, 0.0070827, 0.0767696, -0.3871174,
                 0.0004106, 7.41e-05, 6.9e-06, 0.0021099)),
    # H = 100 ms instead of 150, fitted on the first 75 % of round2, holdout corr 0.421
    (-0.1376293, (0.1241886, 1.0000457, -0.8574578, 0.1962215, -0.0459786, -0.0031503, 0.0115308, 0.2154807, 0.0168906,
                  0.0406975, -0.1544921, -0.0163024, 0.0655148, -0.0529128, -0.0901387, 0.0057237, 0.0798153, -0.2998945,
                  0.0003081, 5.95e-05, -3e-06, 0.001068)),
    # exp_r3_pms (research/diag/pred_ms_r3.py): lean ridge, H = 150 ms, fitted on the first 75 % of round3, holdout corr 0.456
    (-0.062417, (0.2775507, -1.5860022, 4.0092883, -1.1545062, 0.4269749, -0.5366721, 0.001776, 0.0732531, 0.0104478,
                 0.0365482, -0.1613575, 0.1527298, 0.0518662, -0.0493269, -0.054811, 0.0044655, 0.1031784, -0.4176847,
                 0.0003904, 0.0003088, -1.74e-05, -0.0021945)),
    # same, H = 500 ms, holdout corr 0.450
    (0.0014319, (0.4915197, -3.4882645, 5.7144612, -3.4437006, 1.7689845, 0.0203486, 0.0034531, 0.2038764, 0.0073592,
                 0.0532544, -0.2642959, 0.4456907, 0.0198291, -0.0235174, -0.07914, 0.022941, 0.1142459, -0.548584,
                 0.0006258, 0.0006661, -5.49e-05, -0.0171894)),
    # rfit 5 (parity_r3_75); params rfit=5
    (-0.0624031, (0.277322, -1.5824591, 4.0031018, -1.1521293, 0.4264759, -0.5362993, 0.0017923, 0.0731271, 0.0104396, 0.0367361, -0.1611373, 0.1525277, 0.0518743, -0.0493497, -0.0542882, 0.004415, 0.1031497, -0.4180602, 0.0003906, 0.0003078, -1.76e-05, -0.0022234)),
    # rfit 6 (r3_100); params rfit=6
    (-0.0312048, (0.2524195, -1.3480718, 3.3785846, -0.861668, 0.3252749, -0.4616473, 0.0039907, 0.0887642, 0.0104336, 0.0330804, -0.1469838, 0.1544691, 0.0474075, -0.0491369, -0.0678913, 0.0067726, 0.1081369, -0.4207363, 0.0003946, 0.0002871, -1.46e-05, -0.0018087)),
    # rfit 7 (r3m_75); params rfit=7
    (0.0057834, (0.2815741, -1.5935843, 3.9275782, -1.1594923, 0.4303528, -0.510283, 0.005897, 0.0708377, 0.0107017, -0.0030404, -0.1612667, 0.1574112, 0.049974, -0.050176, -0.0588904, 0.005253, 0.1088707, -0.4565042, 0.0003878, 0.0003099, -1.47e-05, -0.0002002)),
    # rfit 8 (r3m_100); params rfit=8
    (0.0054826, (0.2466326, -1.2869831, 3.2095648, -0.8381711, 0.3205404, -0.433942, 0.0090329, 0.0897958, 0.0105591, -0.0036631, -0.1470923, 0.1603695, 0.0478484, -0.0478539, -0.0724708, 0.0077466, 0.1125127, -0.4387961, 0.0003906, 0.0002869, -1.14e-05, -0.0002533)),
    # rfit 9 (r3r2_75); params rfit=9
    (-0.0713038, (0.2382439, -0.5503573, 2.2009459, -0.6307873, 0.2446267, -0.3812074, 0.0104099, 0.1187834, 0.0122256, 0.0135575, -0.1686974, 0.1091626, 0.0579816, -0.049788, -0.0780315, 0.0067838, 0.1084385, -0.3902581, 0.0003895, 0.0002508, -8.2e-06, -0.0010775)),
    # rfit 10 (r3r2_100); params rfit=10
    (-0.0648742, (0.2105307, -0.1880777, 1.8586417, -0.4405044, 0.1869755, -0.4230065, 0.0032603, 0.1277431, 0.0134098, 0.0247087, -0.1629833, 0.1038543, 0.0551304, -0.0499972, -0.08014, 0.0072563, 0.098058, -0.3955948, 0.0003957, 0.000219, -6e-06, -0.0005508)),
    # rfit 11 (h100); params rfit=11
    (-0.0535764, (0.1986696, -1.1499214, 3.5870151, -0.8245487, 0.2840145, -0.6206259, 0.0026813, 0.0370241, 0.0109065, 0.0338854, -0.1289682, 0.0896141, 0.048413, -0.0468462, -0.0542142, 0.0021219, 0.0965979, -0.4151024, 0.0003238, 0.0002183, -1.65e-05, -0.0013505)),
    # rfit 12 (h200); params rfit=12
    (-0.0827942, (0.3240314, -1.8918189, 4.2843136, -1.4612609, 0.5873722, -0.4526961, 0.0015868, 0.1012547, 0.0097485, 0.044972, -0.166947, 0.2014094, 0.0470965, -0.0439305, -0.0558036, 0.0069273, 0.109682, -0.4199382, 0.0004335, 0.0003967, -1.95e-05, -0.0010397)),
    # rfit 13 (a30); params rfit=13
    (-0.0622792, (0.2243555, -1.1074354, 3.3663358, -0.8674549, 0.3337896, -0.5294242, 0.0018603, 0.0728563, 0.0105099, 0.0367516, -0.1610352, 0.1528795, 0.0518183, -0.0493092, -0.0542123, 0.0044774, 0.1029307, -0.4178398, 0.0003941, 0.0003062, -1.79e-05, -0.0022288)),
    # rfit 14 (a3); params rfit=14
    (-0.0624589, (0.3015645, -1.7992775, 4.2923061, -1.2817882, 0.4686366, -0.5389994, 0.0017672, 0.0732306, 0.0104078, 0.0367303, -0.1611726, 0.1523779, 0.0518954, -0.049364, -0.054311, 0.0043852, 0.1032468, -0.4181489, 0.000389, 0.0003085, -1.74e-05, -0.0022209)),
    # rfit 15 (h250); params rfit=15
    (-0.07794, (0.3595536, -2.2248901, 4.6752248, -1.821058, 0.7794161, -0.3920195, 0.0019532, 0.1302642, 0.008737, 0.0422214, -0.170975, 0.2523107, 0.0404407, -0.0357294, -0.0664491, 0.0117513, 0.1115327, -0.4640593, 0.0004835, 0.0004559, -2.33e-05, -0.0042474)),
    # rfit 16 (h300); params rfit=16
    (-0.0724841, (0.3900014, -2.5147352, 4.8607743, -2.1605277, 0.9738315, -0.2807812, 0.0026442, 0.1481326, 0.0081509, 0.0536237, -0.1987224, 0.303423, 0.0365457, -0.0342982, -0.066235, 0.014328, 0.1113762, -0.4866288, 0.0005272, 0.0005067, -2.73e-05, -0.0055818)),
    # rfit 17 (h400); params rfit=17
    (-0.0282405, (0.466667, -3.2114534, 5.4724355, -2.8877678, 1.3836155, -0.1034287, 0.003558, 0.180911, 0.0075644, 0.0506587, -0.2457532, 0.3890844, 0.0306282, -0.0309191, -0.0744028, 0.0204671, 0.111792, -0.5352834, 0.0005708, 0.0005952, -4e-05, -0.0130527)),
    # rfit 18 (h200_r3r2_75); params rfit=18
    (-0.0973287, (0.2980089, -0.8502869, 2.4510317, -0.906685, 0.37287, -0.2955706, 0.0099235, 0.1505303, 0.0114378, 0.0158284, -0.1828066, 0.154566, 0.0555681, -0.044164, -0.0873357, 0.010265, 0.1145985, -0.3912, 0.0004421, 0.0003158, -8.7e-06, -0.0008097)),
    # rfit 19 (h200_r3m_75); params rfit=19
    (0.005415, (0.3303941, -1.9259269, 4.2404875, -1.4825975, 0.5955986, -0.4271199, 0.0057687, 0.0997469, 0.0100156, -0.0028265, -0.1671455, 0.2063794, 0.0448473, -0.0450412, -0.0610664, 0.0078129, 0.11599, -0.4641638, 0.0004312, 0.0003986, -1.65e-05, -0.0001859)),
)
# exp_r3_mlp: 22-32-16-1 ReLU MLP on the same 22 features, target = 200 ms (robust mid at t+5ms+200ms - mid now), fitted on the
# first 75 % of round3 + mirror (research/diag/mlp_r3.py); holdout corr 0.49 vs 0.463 for the ridge. Inputs clipped at +-8 sd.
MLP_MEAN = [0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233382, -0.0, -0.0, 5.906055, 5.906348, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057909]
MLP_SCALE = [1.019957, 0.879338, 0.830501, 0.723114, 0.628298, 2.28085, 12.9744, 0.673953, 10.69429, 0.757186, 0.651131, 0.545643, 1.651632, 1.651633, 0.999999, 4.399308, 1.05635, 0.427403, 127.451202, 323.701813, 938.297913, 1.343849]
MLP_LAYERS = [
    ([[0.066943, 0.120272, -0.358368, -0.115884, 0.010945, -0.175679, -0.261411, 0.112239, 0.1838, -0.114866, -0.031451, -0.188784, 0.022319, -0.149227, 0.267015, -0.132292, -0.076388, -0.41682, 0.115046, 0.201321, -0.458292, 0.162697, -0.017254, 0.008661, 0.536648, -0.250936, 0.146941, 0.062188, 0.481761, -0.221041, 0.14004, -0.049273], [0.049527, 0.165229, -0.239083, -0.042354, -0.066629, 0.171265, 0.111111, -0.316688, 0.045589, 0.025842, 0.140061, 0.121268, -0.035111, 0.166081, -0.061191, -0.202608, 0.138511, -0.094053, -0.030576, 0.335219, -0.271021, -0.113142, 0.220937, 0.135911, -0.409814, -0.085793, -0.027404, 0.105903, 0.136293, 0.185961, -0.016854, 0.015552], [-0.291672, -0.119207, 0.030961, 0.122247, 0.162517, -0.094942, 0.253666, 0.014278, 0.172065, 0.15009, 0.000265, 0.284292, 0.071912, 0.026779, -0.134713, 0.259852, 0.100663, 0.353949, -0.17241, 0.146077, -0.115363, 0.022841, 0.222134, -0.084045, -0.273249, 0.304072, 0.020471, -0.134488, 0.218031, 0.028553, 0.187728, 0.151106], [-0.000635, 0.089779, 0.184897, 0.234758, 0.233159, 0.197303, -0.01192, -0.023077, -0.234368, 0.319151, 0.03369, -0.048972, 0.095157, 0.048649, -0.08427, 0.080902, 0.048699, 0.44223, 0.101248, -0.178398, 0.191833, 0.172732, -0.034094, 0.12217, 0.079339, -0.030245, 0.243988, 0.179318, -0.185257, -0.080207, -0.311141, -0.000833], [0.02545, 0.052024, 0.083945, -0.174566, 0.154057, 0.081128, -0.086149, 0.056167, -0.055505, -0.062265, -0.273824, 0.004291, 0.398352, -0.115718, 0.13195, -0.158535, 0.013204, 0.368176, 0.465274, 0.018586, -0.167041, 0.070921, -0.116818, -0.138021, 0.283977, 0.015354, -0.037041, 0.221298, -0.432361, 0.116947, 0.09942, -0.205754], [-0.046082, -0.213961, 0.017157, 0.016043, 0.291605, -0.152497, -0.017401, -0.098684, 0.067304, -0.117707, 0.091451, 0.021718, 0.179773, 0.047944, 0.019562, -0.060467, 0.116437, 0.25667, 0.103636, 0.148801, 0.20462, -0.072123, -0.020666, 0.04013, -0.551843, 0.124648, -0.116716, 0.190634, 0.145153, 0.111042, 0.234161, -0.266948], [-0.028178, -0.366293, -0.235355, -0.112522, 0.095321, 0.514204, -0.46427, -0.196664, -0.001375, -0.428219, -0.426284, 0.696893, -0.087232, 0.609089, -0.240243, 0.362304, 0.199522, 0.29372, -0.169118, 0.154267, 0.083371, -0.019951, -0.06784, 0.19711, 0.025124, 0.024469, 0.509556, 0.141625, 0.075955, 0.434384, 0.241584, 0.398111], [0.026493, -0.025249, 0.182736, 0.092406, 0.042804, -0.024718, 0.143291, -0.101047, 0.120779, 0.138127, -0.100367, 0.053303, -0.2141, -0.118666, 0.025511, 0.11286, 0.062464, 0.058981, 0.015699, -0.046808, 0.060583, 0.044302, -0.017981, -0.077249, 0.103769, 0.222212, 0.117276, -0.101792, -0.034874, 0.181032, 0.020078, -0.134994], [-0.242958, -0.416172, -0.276257, -0.101694, -0.209015, -0.275512, 0.229838, -0.588181, -0.427902, 0.175556, -0.204039, 0.022584, 0.000452, -0.222633, -0.082669, 0.574998, -0.678333, 0.558933, -0.408714, 0.075776, -0.285633, -0.319221, 0.195898, 0.081872, -0.206058, 0.023249, -0.312369, 0.244347, -0.012093, 0.129399, 0.266651, 0.445851], [0.32444, 0.071472, -0.14634, 0.261524, -0.62179, 0.131166, 0.251585, -0.65535, 0.188753, -0.094319, 0.018025, 0.12533, 0.178684, -0.396574, 0.226798, 0.242977, -0.631209, -0.172422, -0.085186, -0.600798, -0.357081, 0.342211, -0.288159, 0.10646, 0.065198, 0.145692, -0.162816, -0.045557, -0.073317, 0.379005, 0.102916, -0.077891], [0.088014, -0.072944, 0.081135, 0.048031, -0.118287, 0.210904, -0.004548, 0.086928, 0.057418, 0.023458, -0.145608, 0.039913, -0.104034, 0.049363, 0.019398, -0.040156, -0.028357, -0.068551, 0.277338, -0.088502, -0.00418, 0.124905, 0.18883, 0.01386, -0.152409, 0.036733, -0.261399, 0.008652, 0.185204, -0.166706, -0.176721, 0.10537], [-0.322693, -0.158956, -0.401976, -0.129965, -0.402451, -0.190516, -0.011472, -0.330192, -0.49554, 0.31314, -0.022773, -0.121285, 0.069603, -0.418982, 0.01482, 0.209771, -0.027682, 0.177813, -0.067498, 0.172307, -0.042609, -0.143668, -0.101161, 0.026613, 0.120705, -0.196704, 0.024583, -0.191531, -0.073524, -0.132547, 0.279965, 0.400684], [-0.085093, -0.081297, 0.325137, -0.176173, -0.026384, -0.227667, -0.047251, -0.191921, 0.022097, 0.163785, 0.240984, -0.286312, -0.007035, -0.027705, -0.092092, 0.233661, 0.311685, -0.036284, 0.032067, -0.163136, 0.05885, 0.011786, -0.204477, 0.228237, -0.106065, -0.136709, -0.223084, -0.16489, -0.075028, -0.084014, -0.247336, 0.010806], [-0.282997, -0.12814, -0.19534, -0.334562, 0.428732, 0.038923, -0.040778, -0.04184, 0.344938, 0.069975, -0.066574, -0.385734, -0.176927, 0.086314, 0.157292, 0.128472, 0.268988, 0.019416, 0.08695, -0.041762, 0.06346, -0.092785, -0.189507, -0.047271, 0.010201, 0.186727, -0.085047, -0.106782, 0.023611, -0.128958, -0.026165, -0.363271], [-0.065195, -0.153132, -0.080385, -0.688909, -0.027196, -0.292153, 0.712369, -0.328982, 0.042757, 0.01859, 0.055972, -0.251503, -0.05227, -0.105975, -0.615423, -0.423421, 0.12091, -0.256202, -0.377161, 0.237194, -0.183856, 0.154538, -0.161818, 0.24888, -0.041748, -0.151389, -0.304998, -0.231785, 0.105836, 0.203381, -0.116649, 0.179649], [-0.18632, 0.241947, 0.097802, 0.473644, 0.038817, -0.20453, -0.191199, -0.076638, -0.058033, 0.084998, -0.076714, -0.071364, 0.235598, -0.028088, 0.16679, 0.246116, -0.152069, -0.090617, -0.089045, 0.143986, 0.1488, 0.2621, 0.13354, -0.263948, -0.116339, 0.078125, -0.075966, -0.080283, -0.200206, -0.238513, 0.248191, -0.104852], [-0.024095, 0.113596, -0.123797, 0.358121, -0.100889, -0.125164, -0.090714, 0.520833, -0.13413, 0.195057, -0.063276, 0.018434, 0.120175, -0.694796, 0.261976, 0.487608, -0.038586, -0.010077, 0.567651, -0.202001, -0.142432, -0.605223, -0.20975, -0.192642, 0.034903, -0.203975, 0.590032, -0.255833, 0.035756, -0.125958, 0.440037, -0.094081], [0.154172, 0.114625, 0.569564, -0.111379, -0.610456, -0.613659, -0.478289, -0.190994, -0.009435, -0.055415, -0.04209, -0.210513, -0.2225, 0.161215, 0.048415, 0.026395, 0.460373, -0.18949, -0.184556, 0.010055, 0.06885, 0.219711, 0.23177, 0.127199, 0.34185, -0.062599, 0.240196, -0.073394, 0.475098, -0.403108, -0.285911, -0.105996], [-0.108529, 0.096269, -0.025211, -0.055042, -0.139981, 0.050161, -0.15435, -0.015445, 0.053185, -0.043506, -0.127909, -0.077368, -0.061149, -0.022121, 0.016729, -0.0872, -0.043714, 0.044695, -0.109019, -0.057356, 0.032643, 0.050282, -0.035206, -0.008852, -0.027579, 0.044184, -0.00963, 0.270621, 0.025517, -0.068444, 0.08516, -0.008622], [-0.222509, -0.92691, 0.094995, -0.03933, -0.112769, -0.193575, -0.030115, 0.017401, -0.058151, -0.002767, -0.061206, 0.270805, 0.052331, -0.086804, 0.185191, 0.168124, 0.161172, 0.018984, -0.25976, 0.007301, -0.03831, -0.233665, 0.329697, -0.082291, 0.070424, 0.328462, -0.496405, 0.101706, 0.021723, 0.304964, 0.181378, 0.141366], [0.164095, -0.051133, 0.195924, -0.078633, -0.266317, -0.194948, 0.154941, -0.10852, -0.117282, 0.199036, 0.254971, 0.090201, -0.030095, 0.055339, 0.049931, -0.062985, 0.086792, -0.181168, -0.202497, 0.045323, -0.071063, -0.103218, 0.138162, 0.082975, 0.078019, 0.421609, -0.160835, -0.178847, -0.043429, -0.057394, -0.038515, 0.038923], [0.070086, -0.139566, -0.061042, -0.292309, 0.041117, 0.032878, -0.005407, -0.031323, 0.168689, -0.1224, 0.498416, 0.393255, 0.079576, -0.096018, -0.479617, 0.28723, -0.266583, 0.032518, 0.008717, -0.130674, -0.003462, -0.048226, -0.435315, -0.629183, -0.011066, -0.059616, 0.111921, 0.405622, -0.054787, -0.085482, -0.095232, 0.261683]], [-0.278376, -0.099034, -0.204593, -0.053975, -0.317732, -0.091683, -0.047354, -0.174534, -0.609226, -0.704569, 0.499123, -0.472611, 0.247864, 0.067568, 0.046678, 0.155329, 0.341443, -0.088468, -0.009622, -0.350855, -0.295885, -0.24397, 0.113148, -0.169626, -0.001047, -0.296195, 0.034764, 0.312772, -0.120348, 0.103988, -0.091402, 0.082136]),
    ([[0.168585, -0.252286, 0.039887, -0.109907, 0.12596, -0.167051, 0.246393, -0.141371, -0.030026, 0.061707, 0.194228, 0.128337, 0.154953, -0.144466, -0.231496, -0.158408], [-0.11185, -0.001954, -0.65748, -0.182059, -0.130215, -0.272908, 0.144975, -0.485061, 0.083155, -0.127845, 0.105343, 0.033351, 0.012089, 0.038348, 0.233256, 0.046223], [0.024713, -0.003143, 0.275231, -0.060463, -0.017949, 0.334513, 0.250913, 0.346796, 0.339438, 0.160482, 0.332899, 0.175774, -0.150747, 0.211427, -0.002518, -0.014795], [-0.639437, -0.2416, -0.044399, 0.032352, -0.251727, 0.198805, -0.183777, -0.061502, 0.036179, 0.057994, 0.031361, 0.127759, -0.055203, -0.106212, 0.319384, 0.326432], [-0.454435, 0.080872, -0.268573, -0.007907, -0.122726, -0.313976, -0.476468, -0.122549, 0.012155, -0.238251, -0.023788, -0.505511, -0.714493, 0.285875, 0.009196, 0.221654], [-0.002674, 0.189375, 0.278, 0.258679, 0.018169, 0.326685, 0.267519, -0.120731, 0.02031, 0.181177, 0.200836, 0.302716, 0.040863, -0.609574, -0.25271, 0.319428], [0.108734, 0.15706, 0.250209, 0.043576, 0.129074, 0.156547, 0.323675, 0.143479, 0.353787, 0.107248, 0.185543, 0.321341, -0.235009, -0.170051, -0.024423, -0.551516], [-0.113804, -0.3113, -0.11749, -0.104168, -0.010707, 0.30636, 0.158643, -0.111952, 0.148734, 0.194378, 0.208802, 0.004262, -0.460504, 0.139871, -0.000776, 0.065859], [0.181902, 0.068307, 0.026174, -0.216895, -0.245778, -0.13641, 0.246353, 0.124476, 0.327407, 0.336649, 0.338545, 0.100827, 0.251603, -0.167972, 0.373931, -0.116822], [0.239059, -0.197664, 0.080324, 0.126201, 0.162274, -0.011076, -0.010038, 0.100613, -0.45117, 0.229271, 0.00641, -0.816061, 0.598643, 0.146778, -0.073125, 0.00545], [0.097529, -0.057735, -0.1484, -0.040611, -0.176287, -0.487833, -0.097353, -0.108813, -0.227546, 0.116581, -0.055772, -0.125936, -0.076689, 0.064261, -0.137101, -0.478306], [-0.033589, -0.106713, -0.262553, -0.04954, -0.170773, 0.019786, -0.323806, -0.068054, -0.264491, 0.194995, 0.033952, -0.326182, 0.3498, -0.495059, -0.40789, -0.048604], [-0.127255, 0.1263, 0.132122, -0.012348, 0.143074, 0.180594, 0.115362, 0.134678, 0.115873, -0.409901, 0.17746, -0.176669, -0.013763, -0.117479, 0.22609, 0.187761], [0.054243, 0.351503, -0.186892, 0.074319, 0.077268, -0.150439, -0.117804, 0.128678, 0.090459, 0.180649, 0.251682, 0.042095, 0.266992, -0.689368, -0.441449, -0.421321], [-0.215694, -0.024014, 0.145645, 0.06306, -0.108096, 0.161536, 0.006547, 0.066546, -0.054053, -0.477606, 0.027263, 0.170556, 0.112535, 0.397616, 0.059122, 0.322156], [0.181385, 0.199542, -0.161469, 0.035065, 0.032752, -0.142462, 0.157537, 0.000695, -0.627874, -0.368862, 0.120877, 0.054108, 0.010247, 0.032214, -0.118763, 0.310466], [0.261721, -0.075117, -0.31792, -0.005442, -0.132271, 0.012447, 0.160078, 0.033721, -0.324435, -0.236227, 0.047451, -0.516395, -0.122214, -0.12113, -0.126514, -0.469225], [0.093794, -0.562656, 0.307276, 0.140741, 0.155311, 0.063097, 0.028136, 0.28139, 0.275088, 0.54642, -0.204701, 0.636974, 0.217433, 0.157309, 0.022828, 0.015885], [0.127768, 0.132281, -0.386891, -0.072621, -0.230695, 0.143519, -0.469635, -0.453128, -0.023589, -0.187015, 0.205327, -0.430509, 0.124684, -0.09954, -0.001156, -0.33405], [0.137309, -0.147078, 0.070003, 0.216889, 0.261085, 0.064633, -0.012219, 0.134411, 0.194118, -0.022313, -0.344139, 0.290876, 0.173418, 0.321086, -0.363999, -0.223875], [0.042691, -0.089243, 0.230801, -0.101536, 0.249435, -0.040305, 0.027876, -0.104094, 0.170332, 0.218709, -0.07791, -0.036683, -0.110513, -0.443043, 0.016472, -0.409957], [0.205812, 0.130191, -0.097385, -0.106529, 0.005118, -0.424658, -0.164644, -0.260955, 0.046359, 0.126803, 0.047614, 0.107673, 0.280215, -0.384282, -0.192837, 0.22982], [-0.038826, -0.047256, 0.058717, 0.141466, 0.181242, 0.115117, -0.036607, 0.136449, -0.184992, -0.005515, 0.12189, 0.000646, -0.535637, -0.156596, -0.084076, -0.170541], [-0.223482, 0.15001, -0.052376, -0.110291, -0.267031, -0.330386, -0.166686, -0.13269, 0.153282, 0.403131, 0.116884, 0.03204, -0.180861, -0.192244, 0.187456, -0.175297], [0.003585, -4e-06, 0.07922, -0.163196, 0.074003, 0.028042, 0.013547, 0.335226, 0.371282, -0.229818, 0.179316, 0.030298, 0.013467, 0.249659, -0.010651, -0.152682], [0.032085, 0.102724, -0.165203, -0.223382, -0.114059, -0.094106, -0.541972, -0.280941, -0.170173, 0.000357, 0.062926, -0.835212, -0.487093, -0.053907, 0.132584, 0.08954], [-0.023315, 0.119127, -0.102696, 0.290801, -0.093167, -0.744975, 0.123413, -0.277015, 0.035289, 0.101998, 0.089249, -0.114926, 0.166078, -0.347571, -0.090007, 0.022492], [0.044123, 0.089042, -0.075957, -0.12383, 0.073883, 0.070634, 0.114846, -0.029639, -0.029193, -0.432353, -0.11187, -0.296778, -0.113306, -0.199852, -0.015916, -0.101034], [0.166041, -0.346702, 0.277411, 0.239204, 0.185581, 0.175612, 0.292734, 0.182268, 0.295251, -0.003075, 0.007127, 0.308381, -0.029198, 0.024698, -0.014589, 0.107169], [0.078143, -0.537069, 0.121968, 0.149884, 0.210873, -0.207563, -0.346699, 0.206404, -0.13718, 0.171427, 0.155044, -0.267504, -0.131033, 0.001403, -0.066493, -0.118575], [-0.050594, 0.259558, 0.143369, -0.021736, 0.006612, 0.091827, 0.185095, 0.029657, -0.236907, -0.35728, 0.191691, -0.451339, -0.041276, 0.285251, -0.123816, 0.025568], [0.402898, -0.17066, -0.077135, 0.040564, 0.174057, -0.153129, 0.122233, 0.173263, -0.085006, -0.159128, -0.147667, -0.08058, 0.127474, 0.091702, -0.177335, -0.331555]], [0.043458, 0.149123, 0.110197, 0.342115, -0.086261, -0.190143, 0.149109, 0.146057, -0.040206, -0.240387, 0.123344, 0.219386, -0.231043, -0.214568, 0.421145, -0.260841]),
    ([[0.298042], [0.399064], [0.435127], [0.414081], [0.317696], [0.666317], [-0.287826], [0.446264], [-0.610079], [-0.767302], [-0.174593], [-0.626661], [0.825489], [0.968076], [0.43161], [0.946813]], [-0.338221]),
]
# MLP2 block
MLP2_COLS = ['fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fs70_x_lw', 'fs70_cube', 'sgnflow100_x_absfs', 'fs70_x_spread', 'spread', 'imb1', 'cnt_imb1', 'lq_b', 'lq_a', 'ltr_sign', 'ltr_lsz', 'ltp_dev', 'hole', 'flow_ev10', 'flow_ms100', 'flow_ms1000', 'dt_tr_log', 'ioc_sweep50', 'spread_max100', 'nev200']
MLP2_MEAN = [-0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.233382, 0.0, -0.0, 5.906633, 5.906633, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.05788, 0.0, 2.165361, 4.415074]
MLP2_SCALE = [1.020368, 0.879681, 0.830847, 0.723417, 0.628551, 2.281788, 12.978411, 0.674084, 10.704477, 0.755279, 0.651221, 0.545623, 1.651871, 1.651871, 0.999999, 4.399956, 1.05635, 0.427403, 127.463814, 323.861115, 938.76239, 1.34419, 1.069721, 2.097923, 0.19608]
MLP2_NETS = [
    [([[0.47814, 0.033182, 0.126547, 0.248644, 0.145869, 0.147435, -0.187832, 0.304538, 0.050776, 0.003358, -0.234962, -0.257905, -0.31, 0.099343, -0.08825, -0.414409, -0.020177, -0.145715, 0.199235, -0.117144, -0.271229, 0.018359, -0.054285, 0.203533, -0.115823, -0.159059, 0.243232, -0.068599, -0.213962, -0.061111, -0.282288, -0.107259], [0.296911, 0.148113, 0.147509, -0.002641, 0.147741, -0.121081, -0.302814, -0.295312, 0.161227, -0.102775, -0.061945, 0.03344, 0.058482, 0.139789, 0.061621, 0.163774, 0.0965, 0.000862, 0.022786, -0.050222, 0.234153, 0.285555, -0.210543, -0.09528, -0.043123, 0.088384, -0.027042, 0.102161, 0.103414, 0.257232, 0.058605, 0.215223], [-0.148906, -0.250973, 0.076566, -0.27526, -0.173485, -0.251744, -0.181755, -0.425875, -0.145945, 0.105618, 0.137007, -0.329571, -0.003236, 0.141775, -0.117052, 0.266613, 0.0165, -0.133753, 0.032346, 0.156937, 0.271711, 0.081257, 0.101664, -0.173739, -0.123691, -0.085166, 0.227328, 0.23727, -0.044676, -0.000188, 0.236792, -0.046047], [-0.371274, -0.119766, -0.084161, 0.136686, -0.0778, 0.12841, 0.161472, 0.029328, -0.137833, 0.000696, 0.095986, -0.064129, 0.168776, 0.211678, 0.128943, 0.228418, 0.066451, 0.024667, -0.097426, 0.014117, 0.133042, 0.017114, -0.135597, -0.026958, 0.063349, -0.145684, -0.165771, 0.301043, 0.061713, 0.187299, 0.367918, 0.380537], [-0.27273, -0.085341, 0.17139, 0.064828, -0.26857, -0.024065, 0.211661, -0.198487, -0.011207, 0.000513, -0.128406, 0.109668, 0.03514, 0.007087, 0.145004, -0.281757, -0.228287, 0.221962, 0.101672, 0.200973, 0.211235, -0.045222, -0.082083, 0.096131, -0.269786, -0.025958, -0.005438, 0.059382, -0.029207, -0.264812, -0.140398, 0.006548], [-0.014278, 0.046079, -0.303088, -0.201786, 0.086852, -0.050088, -0.066049, 0.009298, 0.076287, 0.010244, 0.163585, 0.058868, -0.059168, -0.303483, 0.107835, 0.172232, -0.24741, 0.041443, 0.209127, 0.100163, -0.233803, 0.032226, 0.17841, -0.177613, -0.312871, 0.146149, -0.201031, 0.157038, -0.034321, -0.145141, 0.111729, 0.018912], [0.344733, 0.618338, 0.127635, -0.008598, -0.654451, -0.274135, -0.095576, -0.440326, 0.017681, 0.141911, -0.001041, -0.353389, -0.457057, -0.064064, -0.168538, 0.000405, 0.166475, 0.034736, -0.219777, 0.073844, 0.192566, -0.082121, -0.126511, -0.174158, 0.376046, 0.743402, -0.492596, -0.142389, 0.187884, 0.546064, 0.272517, 0.875458], [0.250794, -0.135407, -0.076891, -0.012086, -0.053955, -0.096888, -0.136035, 0.05216, -0.041267, -0.050143, -0.275737, -0.016839, 0.156486, 0.117464, 0.113376, -0.245753, 0.137521, 0.209101, 0.155245, -0.161337, 0.072367, 0.144031, -0.057353, -0.272292, -0.157916, 0.091604, -0.027026, -0.186131, 0.000812, -0.168732, -0.010251, 0.016166], [-0.022863, -0.206191, -0.135045, -0.206426, -0.374369, -0.375198, -0.557198, -0.225797, 0.347011, 0.303009, 0.118394, -0.097882, -0.169152, 0.172414, -0.564107, 0.439736, 0.338165, -0.333958, -0.800276, -0.004479, 0.227853, -0.255163, 0.013315, 0.343786, 0.418019, 0.338254, 0.046404, 0.420454, 0.337435, 0.234116, -0.070165, 0.067092], [-0.013988, -0.104364, 0.177505, 0.082413, 0.176221, 0.194758, -0.041334, 0.125798, 0.267712, 0.161947, -0.142732, -0.1412, -0.015629, -0.092842, -0.008664, 0.239615, -0.051998, 0.020888, -0.244166, 0.054445, 0.270944, -1.023264, -1.056698, -0.0345, 0.14123, 0.220595, -0.14172, 0.188792, 0.261946, -0.216624, -0.26873, 0.137509], [0.168842, 0.067954, 0.112543, 0.016776, 0.043967, 0.017947, 0.073144, -0.001734, -0.035535, -0.017252, 0.154103, 0.101987, -0.35765, -0.100041, -0.030725, -0.000926, -0.22591, 0.069273, -0.017615, 0.236647, -0.115517, 0.101228, -0.177804, -0.192984, -0.121699, -0.076868, -0.091445, 0.036003, 0.098855, 0.073877, 0.027735, -0.23299], [-0.045144, -0.178243, 0.032034, -0.174317, -0.089389, 0.07569, -0.373565, -0.244152, 0.031337, 0.206401, 0.105769, -0.379698, 0.036653, 0.383171, -0.386883, 0.054038, 0.024836, -0.371686, -0.120743, -0.485035, 0.412727, 0.051303, -0.075628, 0.124432, 0.537548, -0.185653, 0.032718, 0.232028, -0.016541, 0.197338, 0.078222, -0.176784], [-0.205574, 0.105393, -0.068805, -0.071574, 0.035339, -0.116248, -0.026101, -0.162471, 0.025965, -0.143156, -0.016987, 0.054751, 0.00152, 0.021084, -0.312852, 0.221415, 0.189649, -0.181931, 0.282645, 0.180562, -0.117752, -0.111725, -0.079842, 0.044941, 0.065946, -0.313748, 0.255336, -0.19517, -0.170431, -0.06551, 0.002104, -0.289351], [0.024201, 0.170564, 0.067075, -0.023375, -0.027722, -0.199816, -0.251883, -0.108999, -0.037627, -0.041882, 0.418158, 0.090346, -0.169592, 0.147653, 0.107152, -0.169338, -0.083975, -0.04646, 0.025343, -0.138649, -0.116612, 0.022104, -0.049952, 0.181923, -0.257378, -0.110018, 0.033549, -0.038406, -0.292386, -0.082159, 0.016592, 0.146345], [0.035738, 0.027524, 0.035176, 0.056277, 0.10099, -0.029236, -0.142313, -0.296432, 0.046642, -0.151563, 0.381628, 0.265866, 0.075004, -0.006608, 0.015839, 0.335992, -0.265183, -0.018529, -0.258433, 0.186009, 0.001029, 0.037098, -0.459469, 0.108484, 0.260911, -0.316356, -0.186324, 0.088382, 0.045865, 0.06334, 0.254207, -0.238676], [-0.18417, -0.042514, -0.190292, 0.246032, -0.052478, -0.013753, -0.022033, 0.081526, -0.254048, 0.112124, -0.073198, -0.038354, 0.000753, -0.083576, 0.082909, -0.222529, -0.256871, -0.229301, 0.011898, -0.115462, 0.096598, 0.178072, -0.070095, 0.160068, 0.150224, 0.064713, 0.167021, 0.154695, -0.061633, -0.178256, -0.369588, 0.050881], [-0.001011, -0.351274, 0.124254, 0.158476, -0.249233, 0.059845, -0.169314, 0.507623, 0.066684, 0.529724, -0.498199, -0.134946, -0.227752, 0.124443, 0.340839, -0.250438, -0.174063, -0.812665, 0.460535, -0.351298, -0.067798, -0.298057, 0.440517, -0.147956, -0.139255, 0.438348, 0.431474, 0.01174, 0.292121, -0.006133, 0.129312, 0.176543], [0.694677, -0.058903, -0.217622, 0.221007, -0.075733, -0.066253, 0.170429, -0.6687, -0.085665, -0.148414, -0.015374, 0.123695, -0.216226, 0.575141, 0.10382, -0.393336, -0.192823, 0.323568, -0.040886, 0.009027, 0.135723, 0.060182, -0.116025, -0.201031, -0.313091, 0.291493, 0.403551, 0.195953, -0.103291, 0.311926, 0.247728, 0.297745], [0.002176, 0.024295, -0.016183, 0.021187, -0.072363, -0.124429, 0.080423, 0.120268, -0.025461, -0.057309, 0.050157, -0.077297, 0.033414, 0.14428, -0.094286, -0.100898, -0.049162, 0.100307, -0.080733, -0.028549, 0.037109, -0.078908, -0.007956, -0.078841, -0.006573, -0.05072, -0.008197, -0.036841, -0.033727, 0.003371, 0.01612, 0.148414], [0.02076, -0.009871, 0.246092, -0.953144, 0.02652, -0.597455, -0.03082, -0.012105, -0.113066, 0.121221, 0.16928, -0.156339, -0.091433, -0.095503, -0.092197, -0.180812, 0.576608, -0.28591, -0.483311, 0.252955, 0.066037, 0.103549, -0.078139, 0.124749, 0.06351, -0.236795, 0.291471, -0.010394, 0.228107, -0.066926, -0.107021, 0.202398], [-0.034163, -0.337758, -0.271078, -0.161453, 0.24903, -0.026736, 0.175207, 0.084878, 0.121021, -0.058098, 0.423052, 0.012307, 0.058012, 0.319252, -0.171311, 0.059765, 0.09417, 0.156796, -0.037357, 0.009926, -0.08358, 0.282329, 0.054359, 0.338264, -0.110399, 0.283852, -0.085725, -0.123056, 0.227709, 0.089999, 0.057253, -0.027379], [-0.052116, -0.177253, -0.16464, -0.165332, -0.042848, 0.033281, 0.0265, 0.012582, 0.684206, -0.157621, -0.065121, -0.124613, 0.00669, -0.098786, 0.708716, -0.227422, -0.269031, -0.150835, -0.208996, -0.152347, -0.023071, -0.021395, 0.069992, -0.420432, 0.161307, -0.386303, -0.174701, 0.109668, 0.158353, 0.595492, -0.207395, 0.17889], [0.036599, 0.074849, 0.014542, -0.01661, 0.009991, 0.025258, 0.296249, -0.07781, -0.179706, -0.085246, 0.026156, -0.15441, -0.143733, 0.037983, -0.046956, -0.139602, -0.02414, 0.00196, -0.030121, -0.037854, 0.065294, 0.105944, -0.120589, 0.1985, -0.040908, 0.13051, -0.154246, -0.066356, 0.03588, -0.060987, 0.178037, -0.14818], [-0.142235, -0.254316, 0.479258, 0.074535, 0.700061, -0.218516, 0.267153, -0.239287, 0.308486, 0.466148, -0.690903, 0.07578, 0.435068, -0.232875, -0.176805, -0.190531, -0.606811, 0.255734, -0.678589, 0.197999, -0.137824, 0.00938, 0.07608, 0.496101, -0.763004, -0.463385, -0.028018, -0.47607, 0.481477, -0.226635, 0.491033, -0.233272], [-0.096527, 0.335096, 0.028938, 0.026906, -0.151063, -0.01055, -0.266855, 0.06814, -0.069122, -0.25887, -0.171214, -0.09196, 0.277259, -0.077003, 0.189162, -0.313453, -0.061092, -0.081826, -0.305729, 0.317673, -0.046812, 0.052502, 0.105381, 0.108099, 0.15636, -0.085661, 0.242285, 0.254914, 0.025278, -0.071619, -0.14327, -0.083437]], [-0.026562, -0.155693, 0.318717, -0.028426, -0.295439, 0.192919, -0.148268, -0.297645, -0.174143, -0.175841, 0.145365, -0.299341, -0.272106, -0.400794, -0.120102, 0.020181, 0.100106, -0.247649, 0.111566, 0.07234, 0.060596, -0.074546, -0.025885, -0.287164, -0.206094, 0.370456, 0.096172, -0.30292, 0.082985, -0.229062, -0.010188, -0.49282]), ([[0.003125, 0.345108, 0.009493, 0.334028, 0.171428, 0.057013, 0.126851, 0.169162, 0.130452, 0.089163, -0.197632, 0.06555, 0.536941, 0.279364, 0.03241, -0.013674], [0.165815, -0.150182, 0.069369, -0.122801, 0.251651, -0.065822, -0.345543, 0.039076, 0.168762, 0.10205, 0.036582, -0.085585, 0.075809, 0.150494, -0.280415, -0.513], [-0.078487, 0.07026, 0.074538, 0.266495, 0.137208, 0.214128, 0.175281, 0.146772, -0.232517, -0.050532, 0.164399, 0.046407, -0.214388, -0.094023, -0.283732, 0.149791], [-0.363613, -0.414634, 0.020508, -0.696249, -0.578086, -0.162554, -0.007292, -0.126936, 0.101178, 0.030649, -0.09928, 0.204712, -0.013857, 0.09815, 0.006574, -0.053463], [-0.806247, -0.201693, 0.382272, -0.156695, 0.171872, -0.07457, -0.18033, 0.094894, 0.197935, -0.203049, 0.022741, -0.065474, 0.190803, 0.098603, 0.28385, -0.285887], [0.013878, -0.377605, 0.18517, -0.276259, -0.014471, -0.003406, -0.253765, 0.021377, 0.191882, 0.147781, 0.223062, -0.069827, -0.02199, -0.196274, 0.08, -0.070724], [0.00237, 0.110527, 0.200033, 0.286342, 0.361345, 0.100968, 0.122146, 0.201806, 0.016033, -0.150915, 0.001137, -0.053736, 0.025963, 0.208565, -0.150015, -0.264093], [0.315686, 0.20769, 0.306232, 0.152144, -0.15917, -0.043244, 0.388662, 0.355868, 0.28224, -0.085288, 0.264293, 0.055451, -0.081622, 0.170905, 0.331871, 0.391239], [0.057179, -0.460345, -0.000354, 0.005145, -0.017762, 0.020511, -0.138855, 0.016408, 0.00558, 0.162035, -0.315625, 0.013213, -0.091198, -0.057543, -0.100315, -0.191841], [0.206618, 0.059179, 0.028469, -0.115571, 0.149688, -0.184853, 0.0258, -0.326799, -0.373008, -0.053258, -0.324948, 0.298975, -0.21644, 0.085674, 0.069302, 0.254765], [-0.022896, -0.180927, -0.359862, -0.621021, 0.154419, 0.079566, -0.42171, 0.279545, -0.130341, -0.328827, -0.113673, -0.545797, 0.110055, -0.090641, -0.099134, -0.34328], [0.122173, -0.037281, 0.072834, -0.083826, -0.039834, -0.073828, -0.078224, 0.00554, 0.07858, 0.295216, 0.090017, -0.009276, 0.045842, 0.4755, 0.105499, 0.085071], [0.292089, 0.239108, 0.029894, 0.301911, 0.178423, -0.005261, -0.181717, 0.110832, 0.088625, -0.009409, 0.297624, 0.182796, 0.18667, -0.211165, 0.002414, -0.390384], [0.06563, 0.322836, -0.233047, -0.144777, 0.232552, 0.205855, 0.080792, 0.295738, 0.148497, 0.307795, -0.304332, -0.24154, 0.08171, -0.186769, 0.191389, 0.062485], [-0.72765, -0.177858, 0.005997, -0.254815, -0.740553, -0.172048, -0.281754, 0.064618, -0.040067, -0.030881, -0.170094, 0.199876, -0.100165, 0.208421, -0.020371, -0.179568], [0.089861, 0.089817, 0.170592, 0.187231, 0.14677, -0.066937, 0.247088, -0.240841, 0.203241, 0.002902, 0.315869, -0.137147, 0.4072, -0.180527, 0.040596, -0.16079], [0.067206, -0.023043, -0.055364, -0.200455, 0.020069, 0.151428, -0.110964, 0.234845, -0.016685, -0.025083, -0.417938, -0.292259, -0.387227, 0.124258, 0.314888, -0.209287], [-0.583148, -0.251436, 0.070451, -0.125098, 0.045836, -0.22332, 0.130638, -0.12633, 0.005103, -0.376097, 0.005313, 0.426708, -0.039719, 0.016299, 0.081967, -0.226783], [-0.14462, -0.319897, -0.028738, -0.413576, -0.160314, -0.053716, -0.204227, 0.063388, -0.363077, 0.127918, -0.647582, -0.339002, -0.29295, 0.178476, 0.070684, -0.499012], [0.185374, 0.164194, 0.122035, 0.017418, 0.076198, 0.110181, -0.114735, 0.073294, -0.103379, -0.137976, 0.167424, -0.307986, 0.123805, 0.302868, -0.431917, -0.384334], [0.099654, 0.21824, -0.154468, 0.029059, 0.118884, 0.22494, 0.146131, -0.242537, -0.052467, 0.145274, 0.076235, -0.002656, 0.301373, 0.193266, 0.150653, 0.035057], [-0.001053, 0.069982, -0.460462, 0.100362, 0.060398, 0.004202, 0.083657, -0.326227, 0.109619, 0.063224, -0.325241, -0.006813, -0.753657, -0.170389, 0.125956, 0.202773], [-0.296545, 0.010444, 0.054133, -0.29319, -0.117295, 0.118268, 0.110387, 0.134548, 0.000977, -0.059232, 0.169577, -0.375433, 0.336032, 0.111015, 0.191255, -0.609363], [0.017193, -0.10628, 0.177996, 0.185156, -0.387417, 0.049726, 0.33263, 0.083664, -0.11955, 0.079158, 0.003648, 0.228267, -0.213632, 0.004829, 0.316777, 0.379785], [-0.067734, -0.145363, 0.187218, -0.364513, 0.167784, 0.27592, -0.577813, 0.093647, 0.002087, 0.239831, -0.022733, -0.055342, -0.240293, -0.558896, 0.034729, -0.041427], [-0.043623, 0.306814, -0.194563, -0.135929, -0.11661, -0.078456, 0.003244, -0.2167, 0.286834, -0.192112, 0.163918, -0.211259, 0.029234, -0.202185, -0.884536, -0.164807], [-0.120874, -0.053255, 0.000648, 0.332281, -0.19338, 0.010029, 0.256673, -0.065242, -0.077125, -0.050395, -0.094849, -0.106958, 0.281207, -0.407848, 0.213056, 0.139874], [-0.040807, 0.304782, 0.187529, 0.090977, 0.093017, 0.181123, -0.119371, -0.047688, -0.370371, -0.142, -0.095781, -0.112704, -0.247882, -0.0294, 0.200959, 0.031067], [0.171398, 0.02335, 0.216696, -0.051357, -0.025889, 0.15963, -0.016409, 0.006466, -0.058116, 0.090483, -0.225666, -0.003606, -0.387181, 0.142502, -0.108696, -0.226729], [-0.025855, 0.131653, -0.122096, 0.126972, -0.040704, 0.054321, 0.34484, -0.017667, -0.403089, -0.030545, -0.361555, -0.012014, -0.32376, -0.335628, 0.16438, 0.530492], [0.025605, -0.151698, 0.066659, 0.142133, -0.054451, -0.056083, -0.207071, 0.114066, -0.06699, 0.251543, 0.335496, -0.204731, 0.248724, 0.229243, 0.308255, -0.132896], [-0.212705, 0.040417, -0.16327, 0.022115, -0.432548, -0.029544, -0.010967, 0.233695, 0.122142, -0.468453, -0.565763, -0.312019, -0.337382, 0.086334, -0.408295, 0.298514]], [-0.385946, 0.041463, 0.088571, -0.043708, -0.371423, 0.151719, -0.161805, 0.236703, 0.138739, -0.145876, -0.087221, 0.088015, -0.237212, -0.631908, -0.614395, -0.104507]), ([[1.087752], [0.534728], [-0.29415], [0.615198], [0.409238], [0.199265], [0.442961], [-0.187374], [-0.364365], [0.38639], [-0.528867], [0.46719], [-0.949453], [-0.705297], [0.87909], [0.684799]], [0.053818])],
    [([[0.42441, 0.0512, -0.674451, 0.113884, 0.257393, -0.044391, 0.146597, 0.131113, -0.004053, 0.012757, 0.292674, -0.013012, -0.108821, -0.007469, 0.04068, -0.001023, -0.054371, -0.061957, -0.277134, 0.036089, 0.199513, -0.07346, -0.021881, -0.086936, 0.419772, 0.168138, 0.069976, 0.079861, -0.205437, 0.173089, -0.061282, -0.04428], [-0.069802, 0.135116, -0.199451, -0.042652, 0.109758, -0.106962, 0.081976, -0.036728, -0.003709, -0.036098, -0.056107, -0.386483, -0.013229, -0.294995, 0.000538, 0.055391, 0.023577, -0.156667, -0.068571, -0.161114, -0.239646, 0.141826, -0.208462, -0.029792, 0.077258, 0.043654, 0.035131, -0.09188, 0.13614, 0.121004, 0.152143, -0.042127], [-0.101521, 0.160455, 0.18575, -0.071023, 0.224315, 0.087975, -0.191699, -0.147495, 0.406105, 0.100136, 0.045626, -0.291349, 0.095138, -0.156719, -0.108716, 0.27004, -0.156301, -0.238398, 0.06667, 0.054719, -0.368879, 0.094896, 0.029725, -0.009645, 0.164072, 0.269538, -0.254894, -0.074681, 0.261271, -0.008247, 0.085338, -0.155876], [-0.17245, -0.182629, 0.216696, -0.126615, 0.016447, 0.027112, 0.01448, -0.0559, 0.057442, 0.181602, 0.012829, -0.18875, -0.01121, -0.089658, -0.120989, -0.021186, -0.085669, 0.114501, 0.05628, 0.196837, -0.080218, -0.229324, -0.168224, 0.074305, -0.209915, -0.050602, 0.100406, 0.125502, 0.259859, 0.083316, -0.157113, 0.091276], [-0.292899, 0.183337, 0.154609, 0.220356, 0.064577, 0.195349, -0.084499, 0.295908, 0.233507, 0.341925, 0.202152, 0.053961, -0.107311, 0.078796, 0.346108, -0.191451, -0.097392, -0.16594, 0.022281, -0.269972, 0.07614, -0.184707, 0.057549, 0.066036, -0.25953, 0.181323, -0.066386, -0.070805, -0.116039, -0.170489, -0.18083, 0.109478], [0.041986, -0.046116, 0.202222, 0.005764, -0.36653, -0.131196, -0.261333, -0.080375, -0.009663, 0.119807, -0.276418, 0.060943, -0.027745, 0.193212, 0.065336, -0.082282, -0.065172, 0.216986, 0.127793, 0.228684, -0.08809, -0.112481, 0.187744, 0.14056, -0.274097, 0.013124, -0.020782, -0.106204, -0.000522, 0.077773, 0.280459, 0.128382], [-0.493639, -0.264489, -0.338959, -0.049731, 0.813075, 0.96133, -0.453988, -0.012675, 0.082352, -0.035966, -0.091392, 0.163525, 0.076493, 0.028322, 0.551341, -0.405776, -0.227626, -0.46093, 0.378352, -0.62727, -0.210489, -0.042887, -0.29627, -0.327521, 0.394329, -0.165948, 0.007892, -0.011061, -0.120606, 0.271503, 0.375368, -0.1917], [0.035178, -0.033924, 0.003913, 0.005182, -0.030687, 0.080731, 0.083413, -0.289177, -0.02911, 0.077971, 0.135267, 0.098423, 0.276324, 0.120322, -0.064345, 0.047824, -0.215349, 0.038858, -0.013853, 0.045463, -0.324188, 0.035608, -0.052352, -0.062101, 0.171309, 0.059526, 0.088257, -0.127302, 0.07038, -0.119311, 0.123592, 0.04031], [0.008289, -0.103851, 0.271534, -0.511352, 0.043694, 0.198904, 0.070393, 0.210921, 0.058606, -0.251877, -0.265489, 0.157539, 0.078591, -0.246544, -0.34914, 0.156602, -0.019728, -0.395849, 0.1734, -0.093132, -0.043522, 0.209769, 0.06376, -0.02507, -0.23671, -0.18933, 0.049933, 0.087363, 0.482234, -0.155669, -0.062978, -0.148248], [-0.008116, -0.628206, 0.049305, 0.111769, -0.054427, 0.430365, 0.284961, -0.34775, 0.315802, 0.159643, 0.195165, 0.017778, 0.14861, 0.243674, 0.142518, -0.313182, 0.061775, -0.149234, -0.106704, 0.624296, -0.014525, 0.138135, 0.453614, 0.278743, 0.211619, -0.377537, 0.091156, 0.205723, -0.235331, 0.221766, 0.132576, -0.26716], [-0.092071, 0.140241, -0.31556, 0.002841, 0.015681, 0.095022, -0.089165, 0.034344, -0.057845, 0.171519, -0.043361, 0.162841, 0.026018, 0.025133, 0.09917, 0.024735, -0.088802, -0.060129, 0.135292, -0.047345, 0.089857, -0.185311, 0.033763, -0.139324, 0.07544, -0.007515, -0.065495, 0.08044, -0.012144, 0.165121, 0.070517, -0.130952], [0.182601, 0.095158, 0.316191, -0.207734, 0.218706, -0.024696, -0.029907, -0.247404, 0.280187, -0.440119, 0.21306, -0.562091, -0.162216, -0.277335, -0.068552, 0.224362, 0.014958, -0.029046, -0.511463, 0.087181, 0.190097, 0.153092, -0.243175, -0.012988, 0.010237, 0.05667, -0.212973, 0.133194, 0.16482, -0.101139, -0.25383, -0.34967], [-0.034202, 0.0578, 0.136124, 0.066165, 0.245811, -0.224074, 0.102221, -0.02307, -0.156437, 0.079355, -0.028017, 0.067289, -0.229028, -0.438991, -0.067642, 0.076695, 0.02176, 0.388448, 0.381858, -0.008058, -0.114282, 0.061059, -0.039778, 0.279731, -0.233697, -0.345928, 0.02569, -0.064551, 0.235459, 0.101448, -0.02923, -0.351506], [-0.01407, 0.085073, -0.067523, 0.044035, -0.021795, -0.136627, -0.164709, -0.097991, 0.031939, 0.393134, 0.143142, -0.012772, -0.066261, -0.206197, 0.01065, 0.103943, -0.402043, 0.001872, 0.021634, -0.085137, -0.005225, -0.225547, 0.070226, 0.014096, 0.020276, -0.152227, -0.283223, -0.15076, -0.234832, -0.191892, -0.223059, 0.138877], [-0.103413, 0.056747, 0.13167, 0.063518, -0.121523, -0.188955, -0.150358, -0.274086, 0.019789, -0.599459, -0.196486, 0.104537, -0.125258, -0.069825, 0.299301, 0.205627, 0.378106, 0.1684, 0.317668, 0.429633, 0.020365, 0.356838, -0.291635, 0.139106, -0.031892, 0.023188, -0.086235, 0.070438, 0.141665, -0.311267, 0.069254, -0.175881], [0.046661, 0.326965, -0.010225, 0.126126, 0.148408, 0.078921, 0.098533, -0.123377, -0.057072, -0.088707, -0.113841, 0.065211, -0.329175, -0.058164, 0.305482, 0.156482, -0.195459, -0.242614, 0.088253, 0.182425, -0.038503, 0.24696, -0.100112, 0.337229, 0.056625, 0.008677, -0.13958, 0.082896, 0.09913, 0.0481, -0.110119, 0.160443], [0.395513, -0.109719, -0.13147, 0.015542, 0.291536, 0.391077, 0.214342, 0.073835, 0.152163, -0.287088, 0.457133, 0.193632, 0.368862, 0.169754, -0.792589, 0.118728, -0.564853, 0.282767, -0.38662, -0.30549, -0.559558, -0.332647, -0.207803, 0.538528, -0.026791, -0.134797, 0.418844, -0.171197, -0.283601, -0.265413, -0.407986, 0.236956], [-0.185046, 0.265017, 0.005058, 0.271226, 0.41168, -0.236856, -0.24867, 0.113714, 0.029501, 0.633983, -0.251749, 0.336003, -0.124772, -0.378503, 0.445444, -0.281183, 0.098431, -0.102274, -0.430296, 0.102258, -0.403922, 0.016122, -0.492927, 0.102066, 0.378527, 0.059447, -0.092725, -0.354823, 0.10433, 0.183017, 0.187356, -0.166925], [0.020722, -0.015697, -0.05617, 0.014406, -0.003323, -0.051864, -0.157488, 0.004892, 0.012316, 0.042038, -0.030148, -0.036055, -0.006511, 0.148957, 0.048606, -0.010265, -0.030631, -0.053097, -0.038633, -0.025085, -0.000697, 0.017756, 0.074034, -0.026587, 0.032177, 0.01615, -0.167979, -0.007298, 0.007531, -0.097543, 0.099231, 0.057825], [-0.13299, -0.014765, -0.28377, -0.771456, -0.297539, 0.578628, -0.05574, 0.119556, 0.173179, 0.07413, 0.186112, -0.088265, 0.063758, -0.006977, -0.189424, 0.242052, -0.290827, 0.173161, -0.077807, -0.097503, 0.027959, 0.030492, -0.275332, -0.100197, 0.000284, 0.35326, -0.726937, -0.033239, -0.112856, 0.253033, -0.066339, 0.108437], [-0.059197, 0.084667, 0.201676, -0.178757, -0.238738, -0.162668, -0.227848, 0.054032, -0.053366, 0.251651, -0.01481, 0.06927, 0.130445, -0.088512, -0.201299, 0.142368, -0.062087, 0.431007, 0.038116, 0.291905, -0.240266, 0.377841, 0.012113, 0.057737, -0.068049, 0.037942, 0.06294, -0.120949, 0.211986, 0.125808, -0.009549, 0.014904], [-0.217319, -0.142681, -0.054646, -0.056378, 0.138091, 0.118466, 0.417596, 0.121501, 0.134327, -0.210277, -0.222107, 0.059911, 0.642158, 0.035229, 0.376247, 0.141767, -0.005126, -0.015973, -0.229357, 0.071886, -0.188785, 0.050794, -0.130907, -0.273448, 0.104669, -0.013349, -0.125663, 0.600923, 0.030523, -0.030934, -0.124729, 0.62812], [-0.045463, 0.093706, -0.024382, -0.03867, -0.087401, -0.020468, 0.077669, 0.132747, -0.012927, 0.06041, 0.006338, -0.003125, -0.001046, -0.065767, 0.184793, -0.002926, 0.007499, -0.020811, -0.032542, 0.038462, 0.034635, 0.01788, -0.010871, -0.095213, 0.135356, 0.028906, -0.076475, -0.22879, 0.041, 0.092671, -0.223177, -0.011658], [0.149702, -0.076729, 0.096618, 0.042623, -0.863891, 0.20097, -0.370504, 0.777195, -0.548835, -0.444424, 0.615661, 0.018392, 0.055125, -0.585424, 0.164601, 0.561254, 0.278654, -0.708875, 0.114868, -0.148138, -0.405877, 0.222437, 0.204576, -0.329153, 0.243864, -0.325945, 0.458659, 0.338553, -0.494877, 0.389748, 0.530643, 0.030727], [0.033373, -0.067038, 0.096916, 0.026536, -0.252407, 0.042324, 0.32861, -0.231556, 0.264213, 0.146604, -0.096401, 0.053221, 0.122729, 0.091087, 0.118418, 0.267042, -0.052027, -0.098692, 0.242548, -0.051407, 0.171968, 0.255644, 0.025621, 0.235072, 0.342661, 0.127561, 0.07624, 0.088098, -0.009832, -0.091713, 0.459657, 0.059092]], [-0.296778, 0.518413, 0.064003, 0.481045, -0.050593, -0.003714, -0.098083, 0.007822, -0.452724, -0.258398, -0.262707, -0.565902, 0.120472, 0.038692, -0.193805, -0.080594, 0.08795, 0.248892, -0.09708, -0.369343, 0.254715, 0.117438, -0.482709, -0.011796, -0.129337, -0.216172, -0.011302, -0.491539, -0.298702, -0.028857, 0.178447, 0.065766]), ([[-0.144212, 0.208324, -0.077127, 0.117578, 0.341737, 0.113698, -0.111155, 0.143522, 0.158247, 0.103776, -0.104272, 0.098995, 0.295221, -0.341901, -0.009221, 0.188507], [-0.010026, 0.12119, 0.30895, 0.237877, -0.513009, -0.406333, 0.066144, -0.047006, 0.098083, 0.214929, -0.226006, -0.052909, -0.167781, -0.196155, 0.045901, -0.171796], [0.290934, 0.285229, 0.023502, 0.174843, 0.013063, -0.138169, 0.095527, 0.251709, 0.014871, 0.165702, 0.085855, 0.062412, 0.341151, 0.427519, 0.297873, 0.086521], [-0.206286, -0.517054, -0.190475, -0.354144, -0.193646, 0.058559, -0.251708, -0.043771, 0.106213, -0.022486, -0.05828, 0.038877, -0.084953, 0.035491, -0.525289, -0.013991], [0.27985, 0.193336, -0.627674, 0.032336, 0.325269, -0.13259, -0.212098, 0.211856, -0.35851, -0.296128, 0.001485, 0.316518, -0.380459, -0.025241, 0.145569, -0.164719], [-0.210478, 0.039003, -0.082565, 0.025404, 0.319269, 0.048358, -0.164282, 0.144661, -0.476636, -0.590239, -0.00992, 0.136362, -0.509714, -0.243281, -0.172076, 0.337267], [0.016325, 0.01985, 0.106941, -0.185013, -0.026546, -0.034707, -0.370951, -0.478508, 0.149843, -0.237472, 0.000937, 0.150057, -0.152822, -0.172627, 0.27103, -0.104411], [-0.236358, 0.138988, -0.209321, 0.072293, -0.046225, 0.433193, 0.113106, 0.128367, -0.264191, -0.140399, 0.030508, 0.100549, 0.128585, 0.078399, 0.010348, -0.658776], [-0.021968, 0.14811, 0.062777, 0.14884, 0.216445, -0.433408, 0.181994, -0.169302, 0.315393, -0.282019, -0.383334, 0.159027, 0.358657, -0.276047, 0.08154, -0.077808], [-0.315197, -0.589408, -0.128603, -0.161829, 0.175542, -0.193612, -0.164316, -0.070449, -0.058276, 0.014397, 0.157438, -0.135719, -0.218031, -0.174606, -0.174389, 0.257562], [-0.488413, -0.10383, -0.372947, 0.207922, 0.072639, 0.101388, 0.333939, 0.265278, 0.038237, -0.127102, 0.073328, -0.03067, -0.558932, 0.314403, 0.10161, -0.153394], [-0.400066, 0.255429, 0.182296, 0.103241, 0.156067, 0.2453, 0.08307, -0.02213, 0.20045, 0.25198, 0.136748, 0.326553, 0.208517, -0.001812, 0.256043, 0.326295], [0.32432, -0.209294, -0.204698, -0.117798, -0.142781, -0.035805, -0.374051, -0.071971, -0.089945, -0.249005, -0.118025, 0.050047, -0.165945, -0.345044, -0.209791, 0.065698], [-0.628715, 0.15581, 0.271357, 0.226452, -0.052269, 0.077183, 0.261866, 0.050319, -0.1114, 0.37917, 0.231393, -0.064579, -0.095689, 0.189794, 0.024374, 0.090127], [-0.428397, -0.148115, -0.321947, 0.017322, 0.195836, -0.073667, -0.404265, -0.077593, -0.092891, -0.115257, -0.32191, -0.533577, -0.39197, -0.080333, -0.245345, -0.104813], [0.119713, -0.03042, 0.272431, 0.09477, -0.379945, 0.330073, 0.080669, 0.182479, 0.094426, 0.031922, 0.178646, 0.261932, -0.2589, 0.049902, 0.250309, -0.228803], [0.315397, -0.618817, 0.012622, 0.262916, 0.267589, -0.118952, -0.202309, -0.159118, -0.012207, 0.208196, -0.141174, 0.02681, 0.111088, 0.211331, -0.204175, 0.104816], [-0.158006, -0.277757, 0.163884, -0.287588, -0.192602, -0.033591, -0.0092, -0.347669, 0.1945, -0.136882, -0.102686, 0.207177, -0.012443, -0.686164, -0.461797, -0.241248], [-0.210865, 0.013736, -0.102837, -0.103749, -0.243857, 0.35283, 0.025963, -0.222028, 0.013537, -0.133211, -0.008155, 0.179218, 0.225395, 0.263747, 0.408291, -0.102356], [0.35817, 0.068355, 0.255598, -0.12579, -0.075326, 0.406262, 0.344011, -0.436467, -0.133435, 0.406176, 0.267497, 0.166288, 0.673121, 0.027136, -0.533576, 0.054352], [0.235147, -0.011818, 0.26125, -0.049124, -0.026255, 0.23817, 0.030326, -0.357339, 0.140618, -0.191653, 0.225645, 0.155408, -0.138834, -0.235336, -0.136836, -0.227271], [0.126222, -0.138839, 0.002516, 0.065844, -0.042161, 0.155896, 0.237129, 0.065708, 0.034685, 0.12576, -0.000486, -0.25207, -0.530731, 0.143353, 0.094417, -0.002581], [-0.575784, 0.201933, 0.171499, 0.252723, 0.136104, 0.1894, 0.209666, 0.227578, -0.1966, 0.021074, 0.222307, 0.054322, -0.032153, -0.212459, 0.173158, -0.33465], [-0.039071, 0.002095, 0.050609, -0.117783, 0.060013, -0.140041, 0.024634, 0.09059, -0.053249, 0.033132, -0.461463, 0.135489, -0.595795, -0.187804, -0.139062, 0.064742], [0.273608, 0.160673, 0.04447, 0.142577, -0.035293, 0.032868, 0.033038, 0.281681, -0.009999, -0.094662, -0.019061, -0.14918, 0.416688, 0.248767, 0.178489, 0.06873], [-0.39891, 0.140314, 0.102143, 0.226514, -0.124362, -0.031544, 0.004073, -0.002323, 0.32622, -0.043253, -0.031007, -0.237777, -0.623501, 0.019825, 0.057701, -0.157692], [-0.259764, -0.112768, -0.037153, -0.256164, -0.184913, 0.09155, -0.40281, 0.219959, -0.076964, -0.006182, -0.021186, 0.104868, 0.029299, 0.055539, 0.025128, -0.485203], [0.036259, 0.131024, 0.256625, -0.042162, 0.108283, -0.056886, -0.383607, 0.316841, 0.146101, -0.508521, -0.027271, 0.056548, -0.13274, -0.525425, 0.271357, 0.117678], [0.112974, 0.18901, -0.035301, 0.134518, -0.502426, -1.172771, 0.179495, -0.040651, 0.008592, 0.190726, -0.102872, -0.388344, 0.045118, -0.461261, 0.013807, -0.201048], [-0.043828, 0.147717, -0.304305, 0.044465, 0.164217, 0.260067, 0.022809, 0.063331, 0.074303, 0.138766, 0.303791, -0.074413, -0.111747, 0.180845, -0.045429, -0.006033], [0.024889, -0.063632, 0.041022, 0.163933, -0.424999, -0.086157, 0.294451, -0.026788, -0.455279, -0.053914, 0.078927, 0.1345, 0.236912, 0.160495, -0.210249, -0.356523], [-0.306114, -0.079812, 0.092366, -0.31468, -0.088752, 0.022155, 0.143394, -0.288754, 0.075888, 0.155706, 0.128494, 0.146445, 0.250054, -0.341623, 0.03489, -0.147985]], [-0.293413, -0.380699, -0.180045, -0.086319, -0.312896, -0.249778, 0.208349, -0.055448, -0.328568, 0.242473, 0.179862, 0.048711, -0.11491, 0.055285, -0.073198, 0.068476]), ([[0.362975], [0.819697], [-0.426407], [0.384474], [0.810402], [-0.429812], [0.582754], [0.437075], [0.664763], [-0.399483], [-0.34687], [-0.221031], [-0.989736], [-0.437394], [0.902527], [0.751002]], [0.186188])],
    [([[0.202887, 0.076849, 0.255971, -0.117217, 0.041718, -0.05696, 0.080628, 0.081052, 0.30634, -0.06429, -0.121748, -0.007308, -0.186706, -0.104221, -0.275616, 0.335643, -0.171496, -0.233586, 0.106254, -0.468023, 0.040707, 0.037105, -0.05731, 0.008612, 0.11884, -0.387256, -0.15913, 0.262616, -0.002918, 0.098282, 0.118731, 0.250777], [-0.07847, -0.046739, 0.34046, 0.017151, 0.026011, -0.195514, -0.05096, -0.210183, -0.220356, -0.11953, -0.03604, 0.096609, -0.248941, 0.156213, -0.083223, -0.179698, 0.135024, 0.165598, 0.094329, -0.212518, 0.081327, -0.264551, 0.181565, -0.010743, 0.009908, -0.258123, 0.098671, -0.249882, 0.096082, -0.152778, -0.097714, 0.031575], [0.087141, -0.153278, 0.254768, -0.202453, -0.267413, -0.025271, -0.286621, -0.106708, -0.332913, 0.062566, 0.066173, -0.133361, 0.120218, 0.178402, -0.3094, -0.267358, 0.091523, 0.178792, -0.019921, 0.048018, -0.000158, -0.316196, 0.369039, 0.197774, -0.222012, 0.287528, 0.049806, -0.038344, -0.085032, -0.1772, -0.424587, 0.279019], [-0.058725, -0.087336, 0.007652, -0.033124, -0.047723, -0.27547, 0.008164, 0.123497, -0.40173, 0.060834, -0.061159, -0.334973, -0.0342, -0.045814, -0.047649, -0.156158, 0.36409, 0.005543, -0.076952, 0.302455, -0.169551, 0.284347, 0.144901, -0.143341, 0.134148, 0.412263, -0.023008, -0.058023, 0.155848, 0.088934, -0.031093, -0.017442], [0.080322, -0.126359, -0.265714, -0.297146, -0.118082, -0.216258, -0.175148, 0.137582, 0.146199, -0.134516, -0.329539, -0.116598, -0.066651, -0.038463, 0.091507, 0.188927, 0.112413, 0.048678, 0.072097, -0.049959, -0.12182, -0.008284, -0.314895, -0.022295, 0.087038, 0.464758, -0.134235, -0.22128, -0.298954, 0.11016, -0.127325, -0.359367], [-0.058717, -0.116741, -0.083504, 0.184898, 0.110211, 0.005489, -0.042272, 0.05133, 0.179335, 0.118942, 0.300214, 0.138217, 0.011246, -0.0695, 0.228032, -0.029171, 0.313052, -0.246332, -0.100398, 0.111905, -0.131431, 0.171889, -0.081781, -0.063732, -0.085487, 0.090022, 0.180284, -0.095681, 0.251104, -0.076197, -0.103294, 0.069718], [-0.18461, -0.62099, -0.239865, 0.312824, 0.037291, 0.09254, -0.456469, 0.294063, -0.304308, -0.177228, -0.103269, -0.230783, 0.222765, -0.257144, -0.038697, 0.172508, -0.044273, 0.431162, 0.18964, 0.69891, -0.164757, 0.375581, 0.278236, -0.393657, -0.173705, -0.009008, 0.187804, 0.002274, -0.501152, -0.153274, -0.174229, 0.341852], [0.039928, 0.062436, 0.222974, -0.014527, 0.033181, -0.028099, -0.247695, -0.016577, -0.004267, 0.021463, 0.115434, -0.033869, -0.251182, -0.090507, -0.195849, -0.190387, -0.030549, 0.031817, -0.011339, -0.110354, -0.086811, 0.019627, -0.010882, 0.018261, 0.033286, 0.095339, 0.042061, -0.028206, -0.276243, 0.051881, 0.058223, -0.092808], [0.027857, 0.192048, -0.314097, 0.222634, -0.249035, -0.503981, -0.141414, -0.291972, -0.041477, -0.195944, -0.203178, -0.043874, 0.267394, 0.754605, 0.179674, 0.21893, -0.00952, 0.220439, 0.293933, 0.086763, 0.067653, -0.35881, 0.413933, 0.289777, -0.53295, 0.166676, -0.280746, -0.046172, -0.135753, 0.391541, -0.229634, 0.265995], [-0.130795, -0.030763, 0.488927, -0.022899, 0.120743, 0.447424, 0.038609, -0.656694, -0.221655, -0.615441, -0.124439, -0.140787, 0.094597, 0.05192, 0.064464, -0.160795, 0.142084, -0.450793, 0.274385, -0.099222, 0.188062, 0.140393, -0.694875, 0.2315, 0.301835, 0.237912, 0.346862, 0.083183, 0.179849, -0.195, 0.052921, 0.066516], [-0.06914, 0.216492, 0.045187, -0.13131, 0.200873, 0.24327, -0.113418, -0.128869, 0.057396, -0.042044, -0.050472, 0.055672, 0.131298, 0.087537, 0.327252, 0.180823, -0.164229, 0.044707, -0.111795, -0.099229, 0.246197, 0.190263, 0.061295, -0.236336, -0.064661, -0.118624, 0.076332, 0.007452, 0.049981, 0.089681, 0.043918, -0.13621], [0.208127, -0.089422, 0.079304, 0.210356, -0.130688, -0.187656, 0.220258, -0.190921, -0.446684, -0.116405, -0.252138, -0.009276, 0.074146, 0.212022, -0.262754, -0.14049, -0.484394, 0.283762, 0.395168, 0.055124, 0.156215, -0.20937, 0.152177, 0.123495, -0.41086, -0.005904, 0.001745, 0.225492, -0.287885, -0.07405, -0.260235, 0.566449], [-0.047164, -0.04345, -0.152192, 0.046881, -0.039107, -0.158099, 0.011513, -0.021118, 0.076801, -0.057873, 0.169195, -0.063772, -0.063847, -0.288116, 0.111096, -0.22595, -0.218775, -0.254096, -0.030404, 0.114709, 0.339216, -0.014391, 0.210926, -0.042457, -0.130297, -0.076229, -0.340243, -0.457303, 0.181207, 0.141665, -0.194556, -0.197363], [0.216409, -0.203475, 0.067282, -0.014204, 0.005116, 0.050047, -0.218553, 0.202439, -0.079329, -0.03616, -0.061224, 0.014579, -0.107701, -0.015379, 0.052947, -0.197956, 0.275448, -0.214768, 0.11815, -0.125654, -0.105747, -0.07911, -0.233929, -0.047937, -0.160784, -0.097283, -0.083373, -0.14574, -0.064453, 0.313128, 0.155408, 0.151986], [-0.376757, -0.131832, 0.415098, 0.21406, -0.04367, -0.219327, 0.175236, -0.256884, 0.129056, -0.377372, -0.428472, -0.108422, 0.19255, 0.203594, -0.001575, -0.490917, -0.296174, -0.087941, -0.015667, 0.270783, 0.04022, -0.058057, 0.100698, -0.162509, -0.168545, -0.137095, -0.133342, -0.195433, 0.156878, 0.148934, 0.293733, -0.061326], [0.218056, 0.005208, -0.232153, -0.232366, -0.13561, -0.04434, 0.236049, -0.080387, 0.323603, -0.135503, 0.037674, -0.331221, -0.064532, 0.020691, 0.027691, 0.001857, 0.127473, 0.117396, 0.002719, -0.067202, -0.291189, -0.110125, 0.077835, 0.157289, 0.220967, -0.292556, -0.070737, 0.134585, 0.149513, 0.324665, -0.271456, -0.074481], [0.294207, -0.508225, -0.026329, -0.543122, -0.571291, 0.045749, -0.340007, 0.058193, 0.261673, 0.800709, 0.129849, -0.155636, -0.461571, -0.53739, -0.095673, 0.360952, 0.070227, 0.310125, -0.116515, -0.212202, 0.652607, 0.262588, -0.21781, 0.0811, -0.011821, 0.464128, -0.314206, 0.175883, -0.202642, -0.265209, 0.111621, 0.030846], [-0.348993, 0.23539, 0.007977, 0.232345, 0.156048, 0.083645, 0.024903, 0.812721, 0.686403, 0.121015, -0.236271, -0.22312, -0.345872, 0.26817, 0.304712, 0.504176, 0.046105, 0.134369, -0.062014, -0.795215, 0.150939, -0.170263, 0.117947, -0.226556, -0.199946, 0.1754, 0.068229, 0.257843, 0.099753, 0.577522, -0.524267, 0.154349], [-0.040818, 0.155888, 0.078309, -0.028255, 0.015008, 0.047154, -0.063485, -0.006046, -0.023535, 0.003716, 0.045979, 0.141303, -0.002625, 0.060442, -0.029124, 0.012039, -0.061626, 0.146886, 0.118303, -0.014533, -0.103469, -0.129887, -0.104405, 0.059031, -0.114351, 0.00346, 0.058622, 0.052141, -0.004951, 0.055237, 0.102492, -0.006291], [0.062162, -0.110955, 0.099136, 0.162468, 0.020295, -0.042236, -0.585687, 0.046412, 0.105198, 0.002517, 0.160259, 0.109148, 0.499006, 1.030211, -0.032618, 0.192784, -0.017348, 0.071359, 0.24921, -0.006142, -0.034018, -0.191724, 0.03466, 0.203253, -0.237871, 0.123437, -0.096396, -0.265331, -0.110204, -0.106135, 0.123404, 0.149921], [-0.083671, 0.164561, -0.04078, 0.307055, 0.090336, 0.070783, 0.002814, -0.270408, -0.127139, 0.026962, 0.472256, -0.281141, -0.070136, 0.049504, -0.107591, 0.069087, -0.091707, 0.064603, -0.367637, 0.095901, -0.125828, 0.124089, 0.225352, 0.306682, -0.017297, -0.224108, -0.06471, -0.129438, -0.010734, -0.43409, -0.015608, -0.051621], [-0.493449, 0.133239, -0.189981, -0.408404, -0.145602, -0.226363, -0.072251, -0.157899, -0.043864, 0.14474, 0.465374, -0.05807, -0.065435, -0.086195, -0.124917, -0.084904, 0.016688, -0.055441, -0.029456, -0.140204, 0.248663, -0.209805, 0.134286, -0.078494, 0.120026, -0.035311, -0.125496, 0.400732, -0.107277, -0.441653, 0.111446, -0.079702], [-0.001068, -0.086082, 0.090951, -0.020873, 0.036817, -0.21125, -0.017733, 0.006694, -0.110982, -0.095366, 0.045367, -0.074347, 0.104499, -0.052525, 0.145533, 0.211737, 0.051966, 0.083679, 0.3245, -0.054058, -0.217572, -0.126745, 0.03626, -0.049773, 0.055411, -0.175612, 0.130814, -0.046528, -0.156934, -0.015363, -0.022231, -0.16532], [0.381186, -0.092117, -0.00911, -0.284386, 0.79064, -0.128895, -0.057364, -0.011263, 0.069498, -0.004529, -0.449612, -0.638838, 0.053273, 0.017827, 0.045311, 0.190145, -0.376868, -0.738656, -0.139285, 0.35162, -0.447155, -0.299413, -0.01845, 0.222563, 0.385827, 0.134981, 0.210935, 0.122058, 0.280449, -0.619215, -0.686486, 0.02533], [-0.042136, -0.030263, 0.132078, -0.104749, -0.133232, -0.093716, -0.161058, 0.27641, -0.088385, 0.058625, -0.20509, 0.125599, -0.011301, 0.096338, -0.203243, 0.120098, 0.057693, 0.174697, -0.191717, 0.223411, 0.09398, 0.280971, -0.0395, 0.486274, 0.040154, -0.163335, 0.257119, 0.296464, 0.297799, -0.107843, 0.136188, 0.014243]], [-0.310198, -0.372137, -0.207293, 0.109899, -0.264229, -0.359288, 0.276752, -0.13003, -0.121299, 0.226291, 0.101189, -0.27066, -0.359914, -0.015127, -0.157886, -0.035, -0.171728, -0.295476, -0.193534, -0.288382, 0.409599, -0.010165, -0.05026, 0.117222, -0.254124, -0.025873, 0.045884, -0.450153, -0.071666, 0.094885, -0.178771, -0.075099]), ([[-0.476064, -0.018304, -0.091795, 0.167505, 0.041367, -0.203881, 0.048468, -0.033269, 0.067774, 0.267768, -0.117085, 0.390723, -0.212042, -0.087064, -0.016266, -0.281724], [0.080546, -0.291907, -0.263633, 0.050697, 0.309572, -0.196418, -0.345368, -0.033408, -0.141288, 0.130408, -0.036244, -0.048377, 0.00627, 0.35609, 0.26207, -0.067274], [0.22406, 0.164671, -0.000728, -0.054464, -0.230832, 0.05995, 0.111584, 0.23813, 0.083968, -0.001183, 0.011016, 0.342372, 0.032251, 0.277091, 0.291527, 0.096776], [-0.204422, -0.396721, 0.178289, 0.065836, -0.314764, -0.198851, -0.32741, -0.066146, -0.054931, 0.049496, 0.039224, -0.260922, -0.13572, -0.006988, 0.107476, -0.210112], [-0.117321, -0.496461, 0.068753, -0.143694, 0.323839, -0.01174, -0.703883, 0.212637, 0.144446, 0.289533, 0.317074, -0.043851, -0.007301, -0.067021, 0.196659, 0.186856], [0.304429, 0.355375, -0.184755, -0.120533, 0.050246, -0.014084, 0.570672, 0.086947, 0.129368, 0.097801, 0.149283, 0.108868, -0.039094, 0.034099, -0.183092, 0.055071], [-0.143986, -0.844072, -0.078926, 0.177128, 0.012094, -0.049479, -0.244892, 0.236719, 0.283393, -0.19877, -0.066257, -0.045151, 0.175847, 0.096317, -0.003908, 0.175855], [0.067078, 0.119214, -0.278475, -0.18378, 0.08745, -0.152274, 0.101872, 0.362726, 0.033745, -0.592668, 0.26628, -0.061179, 0.089889, 0.155912, 0.114564, 0.26246], [0.121458, 0.01736, -0.346249, 0.063251, -0.162978, -0.104592, 0.486099, 0.02827, 0.063526, -0.031827, 0.11972, 0.250985, 0.322313, 0.292581, 0.298428, 0.088799], [-0.060155, -0.258402, 0.146522, -0.047623, -0.321697, -0.163919, -0.443488, -0.242084, 0.156137, 0.031342, -0.074912, -0.351324, -0.08509, 0.164116, 0.296721, 0.022973], [-0.00815, -0.55534, 0.015035, -0.488203, -0.07573, -0.226335, -0.81285, 0.220906, -0.179684, 0.039984, 0.225265, -0.141331, 0.121826, 0.003835, -0.042177, -0.150826], [-0.031561, 0.111383, -0.106064, -0.077884, -0.382187, -0.058608, -0.219349, -0.258963, 0.132423, -0.069985, 0.0698, -0.439011, 0.032692, 0.22841, 0.030783, 0.117691], [0.192173, -0.25699, 0.145186, 0.147316, 0.172427, 0.080302, -0.003062, -0.078158, -0.267599, 0.022331, -0.20298, -0.119951, -0.224111, -0.38488, -0.275899, -0.31723], [0.007584, 0.184664, 0.200704, 0.000356, -0.395423, -0.166463, -0.081962, -0.136663, -0.397439, 0.040142, -0.455221, 0.049377, -0.179048, -0.855056, -0.243756, -0.613256], [0.126221, -0.070319, -0.121375, -0.041693, -0.077818, -0.034832, 0.045141, 0.137013, 0.111512, -0.051565, 0.048976, 0.201626, 0.029306, 0.055385, 0.195874, 0.285023], [-0.063851, 0.116773, -0.076234, -0.033786, -0.39406, -0.137051, 0.265009, 0.226152, 0.260476, -0.143327, 0.114631, 0.1793, 0.034067, -0.142819, -0.42283, 0.215559], [0.063175, 0.070752, 0.008703, 0.21297, -0.396306, -0.047091, -0.143389, -0.18607, 0.323294, -0.234763, -0.092555, -0.177016, -0.033877, -0.237657, -0.118273, -0.421679], [-0.020087, -0.097968, -0.410068, -0.499469, 0.246817, 0.086698, -0.00612, 0.140939, 0.200473, 0.179444, -0.076219, 0.1027, 0.023334, 0.031524, 0.000831, 0.149609], [0.360847, -0.102116, -0.021078, 0.283703, 0.040756, -0.052774, 0.159878, -0.175352, -0.160028, 0.172203, -0.258205, 0.113354, -0.259873, -0.020003, -0.307904, 0.171756], [0.116523, 0.063803, 0.049484, 0.130802, -0.071514, -0.138239, 0.730305, 0.115704, 0.212066, -0.239337, 0.120731, 0.17128, 0.135767, 0.282985, 0.300268, 0.138701], [-0.002918, -0.085862, 0.074825, 0.226395, -0.194333, -0.206167, -0.25418, -0.1327, -0.355295, 0.062659, 0.113359, -0.051161, -0.11069, -0.467256, -0.09304, 0.007236], [0.403127, 0.306413, -0.299826, -0.072443, -0.115022, -0.110924, -0.320597, 0.096307, 0.135653, -0.143025, 0.134195, -0.159989, 0.152747, 0.013767, 0.089589, 0.087111], [-0.201642, 0.211144, -0.108063, -0.388207, 0.317044, 0.052651, 0.008595, -0.36595, 0.079855, 0.305383, -0.204168, 0.159419, -0.1273, -0.450249, 0.072757, -0.057135], [-0.045765, 0.010054, -0.09625, -0.188648, 0.129566, -0.140176, 0.180105, 0.033058, -0.317846, -0.105627, 0.084577, 0.158919, 0.131384, -0.013573, -0.549841, -0.105246], [-0.007626, -0.244113, 0.315486, -0.021795, 0.195249, -0.204536, -0.103097, 0.216491, -0.434396, -0.452536, 0.12421, -0.064662, 0.122628, 0.158263, -0.217159, 0.082455], [-0.632879, 0.202008, 0.061216, 0.208311, 0.246595, -0.121272, 0.001508, 0.340141, 0.085592, -0.107271, 0.111602, 0.314529, -0.082031, 0.283874, 0.133479, 0.053028], [-0.042485, 0.343203, 0.201795, -0.156227, 0.005923, -0.02055, -0.242237, 0.051941, 0.116977, -0.117208, 0.049852, 0.097205, 0.00793, -0.028111, 0.243145, 0.094162], [0.172421, -0.103277, -0.308355, 0.299593, 0.189763, 0.12793, -0.392522, 0.063697, 0.053613, 0.161892, 0.00197, 0.14144, 0.049769, -0.307959, -0.278934, 0.065078], [-0.079634, 0.111506, 0.058955, 0.184158, -0.002268, -0.00572, -0.440567, 0.320845, 0.017599, -0.108355, 0.146682, 0.060837, 0.065343, -0.272029, 0.272298, 0.190144], [-0.190196, -0.161735, 0.211358, -0.106524, 0.18115, 0.015465, -0.322861, -0.37698, -0.093549, -0.101082, -0.162117, -0.410081, 0.111505, -0.204092, -0.289106, -0.157634], [0.233642, -0.531548, -0.236268, 0.189305, -0.400466, -0.168089, 0.12861, -0.09997, 0.205312, 0.441813, 0.049811, -0.315054, 0.314013, 0.067845, -0.02467, 0.112756], [0.085327, 0.237965, -0.087776, -0.219634, 0.187206, -0.113305, 0.177718, 0.273511, 0.046499, 0.277655, 0.3041, 0.140762, -0.112901, 0.190323, 0.025714, 0.207749]], [-0.066431, -0.181023, 0.362283, -0.473532, -0.237118, -0.12457, -0.012467, -0.317921, -0.032149, -0.523591, 0.055889, 0.062144, 0.390576, -0.18859, -0.360459, 0.019912]), ([[-0.319395], [0.410102], [-0.315907], [0.697582], [1.030763], [0.0201], [0.894659], [-0.391123], [-0.299239], [0.676481], [-0.180945], [0.480738], [-0.190892], [-0.707668], [-0.928175], [-0.228891]], [0.272089])],
    [([[-0.053316, -0.166743, 0.006307, -0.061658, -0.156686, -0.023836, -0.019704, 0.005581, -0.417443, 0.044307, -0.060543, 0.016693, -0.133012, -0.241671, 0.011121, 0.109073, -0.442457, -0.296235, -0.084857, -0.041979, -0.02077, -0.103558, -0.010754, 0.038989, -0.07805, -0.144211, 0.181018, -0.160984, 0.207422, 0.56173, -0.038354, -0.065564], [-0.285617, -0.307463, -0.091336, 0.006959, 0.077498, -0.0119, 0.243338, -0.03859, 0.257595, 0.038429, 0.12451, -0.073832, -0.03347, -0.101622, -0.096638, -0.057058, -0.111758, 0.035199, 0.29731, -0.152963, 0.022906, 0.242363, -0.131063, -0.003721, 0.213622, -0.088846, 0.172091, 0.173546, -0.234461, 0.21167, -0.040441, -0.166647], [-0.257449, -0.146937, -0.05242, 0.256489, 0.034999, -0.099086, 0.077949, 0.229775, 0.378184, 0.140852, 0.261708, -0.100177, -0.000581, -0.069257, -0.179591, 0.091955, 0.071312, 0.083941, 0.091614, -0.014517, -0.13824, 0.066989, -0.200601, -0.071475, 0.030096, 0.125866, 0.184033, 0.242752, -0.011196, 0.06033, 0.246619, 0.113474], [-0.131667, -0.007988, 0.172888, 0.018021, 0.030074, 0.057898, 0.111432, 0.072392, 0.37381, -0.007685, 0.031274, 0.04446, -0.150599, -0.014551, -0.062927, 0.204368, 0.170801, -0.069398, 0.155015, 0.028753, 0.209522, -0.018599, 0.211275, 0.040386, 0.095847, -0.014138, -0.064866, 0.112529, -0.351819, -0.338369, -0.255845, 0.119392], [0.103478, 0.044268, 0.172487, 0.229129, -0.132338, 0.189758, 0.154331, -0.189351, 0.16055, -0.255706, -0.034179, 0.077377, -0.280148, -0.143955, 0.281044, -0.008458, 0.196918, -0.081602, 0.175162, 0.328072, -0.035221, -0.109463, -0.110414, -0.230788, -0.051709, 0.154013, -0.204333, 0.024794, -0.17118, -0.29034, 0.090892, -0.080542], [-0.01483, -0.039565, 0.112435, -0.204445, -0.126116, 0.267772, -0.288922, 0.102578, -0.09776, -0.112683, 0.044306, -0.169719, -0.084609, 0.038866, 0.178778, -0.087713, -0.013775, -0.181022, -0.010001, -0.099969, -0.002105, -0.043371, 0.080743, 0.042825, 0.107367, 0.092341, -0.052012, 0.172912, -0.118422, -0.149653, -0.087868, 0.264175], [-0.520546, -0.150099, 0.311103, 0.050466, -0.187415, 0.037271, 0.501812, -0.376523, 0.15368, 0.251797, 0.583959, 0.011903, 0.123315, -0.28251, -0.314662, -0.253661, -0.176051, -0.058296, -0.032547, 0.344589, 0.714322, -0.101094, -0.003442, -0.356309, -0.28063, 0.155288, -0.248636, 0.154246, -0.216719, -0.732343, -0.068931, -0.379821], [-0.046039, -0.030479, 0.067798, 0.035359, -0.194586, -0.056433, 0.139664, 0.008847, 0.010326, 0.094727, 0.115447, 0.141515, -0.073484, 0.192961, 0.013182, -0.094479, -0.045924, -0.205003, -0.0477, 0.053964, -0.212553, -0.011223, 0.107066, -0.171942, 0.202153, 0.042493, 0.148386, 0.176204, 0.078423, 0.216952, 0.104189, -0.132663], [0.015884, -0.264709, -0.399117, 0.206347, 0.770512, -0.437277, 0.249792, 0.365336, 0.64158, 0.225127, -0.218372, 0.141556, 0.555926, -0.343812, -0.044507, -0.211403, 0.039747, 0.310679, 0.195767, -0.251112, -0.335326, 0.655581, 0.229856, -0.157712, -0.231074, -0.20976, -0.143788, 0.069182, -0.31828, 0.005423, 0.122332, -0.453767], [0.309886, -0.632669, -0.027166, 0.107676, -0.037804, -0.140442, -0.233708, -0.440195, -0.200295, 0.024738, -0.552722, 0.027123, -0.164993, 0.232561, -0.340353, -0.045604, -0.254595, -0.209228, 0.229364, 0.04787, 0.284134, -0.400331, -0.239357, 0.267042, 0.28275, -0.09564, -0.494786, 0.242485, -0.397018, 0.106528, 0.226983, -0.193533], [-0.123453, 0.05968, 0.094988, -0.062095, -0.056076, -0.210467, -0.00732, 0.17327, -0.111306, 0.357786, 0.000835, -0.160266, 0.047275, 0.209187, -0.198949, -0.100054, -0.222344, 0.023533, 0.093557, -0.036796, 0.064671, 0.358799, 0.088242, 0.041001, -0.078713, 0.081727, -0.013838, 0.010245, 0.288073, 0.048719, -0.085565, -0.12134], [-0.040005, -0.264512, -0.106314, -0.014402, 0.178266, -0.041485, 0.307854, 0.386409, 0.173985, -0.047726, -0.248063, 0.071994, 0.183951, -0.43962, -0.177877, 0.275354, 0.145532, -0.138956, 0.189491, 0.094259, -0.046501, -0.272568, -0.253203, -0.030748, -0.140231, -0.126046, 0.289886, 0.08773, -0.48117, 0.131866, 0.079603, -0.169417], [-0.182606, 0.003555, -0.340237, -0.168003, -0.114908, 0.257549, 0.127909, 0.104206, 0.010114, -0.160101, 0.008084, 0.145896, 0.329267, 0.08915, -0.236095, -0.237194, 0.193233, 0.033051, -0.092374, -0.068185, 0.148286, -0.162609, 0.043887, 0.001806, 0.038691, 0.013381, -0.033934, 0.076832, 0.039358, -0.1041, 0.050293, -0.166484], [0.01049, -0.144066, 0.088464, -0.078297, -0.177762, 0.194722, 0.12341, -0.131998, -0.047908, -0.127211, 0.209966, 0.334543, -0.008055, 0.01781, 0.032427, -0.074467, 0.019411, 0.154025, -0.038477, -0.258082, -0.121283, -0.131446, 0.050993, -0.154743, -0.217774, -0.001862, -0.082876, 0.42859, 0.068624, 0.175096, 0.044874, -0.031875], [-0.459437, 0.051046, -0.506593, -0.194187, 0.049649, -0.282302, -0.131387, 0.013005, 0.01856, 0.108738, -0.438801, 0.01593, -0.062935, 0.279181, -0.285099, -0.15574, -0.010328, -0.038577, -0.021124, 0.151478, -0.273023, 0.437034, -0.148641, -0.113112, -0.053894, 0.446165, 0.105348, 0.178801, 0.118115, 0.057375, -0.390352, -0.063792], [-0.093969, 0.069781, -0.296509, -0.071242, -0.059622, -0.248811, 0.132012, 0.327007, -0.344583, -0.220451, -0.179345, 0.215507, -0.160527, -0.026602, 0.094279, 0.313922, -0.043573, -0.220358, 0.004413, -0.229709, -0.246579, -0.02104, -0.184709, -0.043247, -0.003183, -0.10405, -0.154295, 0.061034, -0.299939, 0.064915, -0.319159, -0.081006], [0.293635, -0.123065, -0.048942, 0.032506, 0.003598, -0.296032, -0.006782, -0.046558, -0.156938, 0.115695, 0.377119, 0.296581, -0.578528, -0.138827, 0.337341, 0.095502, 0.309528, 0.046275, 0.149484, 0.119618, 0.513858, -0.141168, 0.515351, -0.612212, 0.431783, -0.582236, 0.001062, -0.19563, 0.320758, -0.231294, 0.555341, -0.451811], [0.671509, -0.307682, -0.140191, -0.092959, 0.621203, -0.015829, 0.413994, 0.239724, -0.101589, -0.404068, -0.632896, -0.259884, -0.180005, -0.257086, -0.082795, -0.172572, -0.487286, -0.044682, 0.253031, 0.15347, -0.734234, -0.005649, 0.231087, 0.030553, 0.36384, 0.374972, -0.197439, 0.065085, -0.331202, -0.210399, 0.044249, 0.282229], [0.162798, -0.094887, -0.018679, 0.067639, -0.341132, 0.097039, 0.026652, -0.072916, 0.016591, -0.042229, -0.06697, -0.114581, 0.086596, -0.173485, 0.010297, -0.058909, 0.055842, -0.036416, -0.02784, -0.009332, -0.002892, 0.038182, -0.138789, 0.001866, -0.061438, 0.054332, -0.021879, 0.038266, 0.116261, -0.077564, 0.08908, 0.017864], [-0.055784, 0.164424, 0.351061, 0.709266, -0.026306, -0.357478, -0.359816, 0.050429, -0.010372, 0.291156, -0.005836, 0.134919, 0.451694, -0.086191, 0.174202, 0.199854, -0.268386, 0.466589, 0.172837, 0.132504, 0.202035, 0.188164, -0.583195, 0.112143, 0.228283, -0.103639, 0.063667, 0.229093, -0.115562, 0.107866, 0.383506, -0.092544], [0.084858, -0.015011, -0.113048, -0.022276, 0.326956, -0.222744, -0.061024, 0.109908, -0.109346, 0.136081, 0.392896, -0.269812, 0.202928, 0.024598, -0.104801, 0.027489, 0.194268, -0.109815, 0.001306, -0.564347, 0.057145, 0.170422, -0.070479, 0.082716, 0.389109, 0.035219, 0.070481, 0.324618, 0.004146, -0.049226, 0.18468, -0.026194], [-0.129967, -0.131883, 0.490203, -0.2046, -0.1696, -0.104721, 0.742191, -0.075074, -0.034678, -0.150091, -0.356281, -0.583482, -0.284822, -0.155995, 0.023515, -0.28037, -0.06999, 0.058348, 0.114233, 0.267652, -0.293645, -0.042915, -0.200064, -0.090978, 0.093454, -0.53484, -0.131465, 0.313625, 0.069628, 0.012257, -0.151243, -0.03114], [0.083025, 0.055445, -0.010515, 0.043658, -0.074663, -0.186032, -0.220643, 0.127988, 0.016854, 0.190379, 0.028834, 0.049798, 0.078772, 0.017221, 0.04586, -0.257126, -0.025355, 0.036365, 0.000337, 0.095701, -0.118249, 0.175606, 0.031113, 0.133701, -0.014741, -0.085307, -0.001211, 0.012247, -0.180259, 0.076828, -0.046209, -0.003239], [-0.580572, 0.144898, -0.145419, 0.100326, -0.466261, 0.106618, -0.297868, -0.120136, 0.046141, -0.059508, -0.726695, -0.168791, -0.205972, -0.5815, 0.398472, 0.220585, 0.495136, -0.902059, -0.606647, 0.294345, 0.053782, -0.411151, -0.281122, 0.537981, -0.008614, 0.257722, 0.302745, -0.042996, -0.537394, 0.217888, 0.105688, 0.428672], [0.00134, 0.478461, 0.22584, -0.176834, -0.18811, 0.263848, -0.106819, 0.152436, 0.221188, 0.032087, -0.004251, -0.062651, 0.080186, 0.067732, -0.165948, -0.009606, 0.149476, -0.11932, 0.208351, 0.158247, 0.099806, 0.171781, 0.0631, 0.22155, 0.002184, -0.01358, -0.085202, -0.137607, 0.174495, 0.258317, 0.167305, -0.364467]], [-0.445286, -0.225438, 0.025184, 0.158574, 0.170726, -0.314791, -0.127705, -0.101413, -0.151783, -0.237737, -0.247355, -0.102433, 0.126725, -0.268566, 0.052589, -0.109547, -0.362924, 0.561663, -0.134865, 0.04476, 0.232665, 0.335728, 0.396954, -0.428101, -0.646737, -0.357709, -0.154987, -0.020825, -0.528612, -0.367521, -0.1245, -0.043874]), ([[0.149869, 0.0129, 0.203329, 0.123707, 0.236402, -0.174827, -0.331994, 0.196775, 0.052358, 0.082776, 0.143553, 0.101903, 0.312649, 0.355851, 0.182002, 0.120874], [-0.66712, 0.241877, 0.026824, -0.270751, -0.004355, 0.230996, 0.544841, 0.073349, 0.223661, 0.212638, 0.113189, -0.008043, 0.101263, 0.167232, 0.129856, 0.191405], [0.054547, 0.015838, 0.260993, -0.140147, 0.034706, -0.120414, -0.178793, -0.402902, -0.263905, 0.324533, -0.334122, 0.023966, -0.185746, 0.022405, -0.185999, -0.246435], [0.027617, 0.123834, 0.163667, -0.080701, 0.199441, 0.245357, -0.06251, 0.332494, -0.186163, -0.1209, 0.195215, 0.145513, 0.078868, -0.038583, 0.117629, -0.316716], [0.304835, -0.10149, 0.189555, 0.025607, 0.197004, -0.332394, 0.007328, 0.194611, 0.274365, -0.000983, 0.125893, -0.056302, -0.552691, -0.005946, 0.11789, -0.0627], [-0.22871, -0.452556, -0.037335, 0.016721, -0.117364, -0.326563, -0.828855, -0.283981, 0.133192, -0.808981, -0.116804, -0.038414, 0.254448, -0.039045, 0.079852, -0.02828], [-0.333093, 0.458485, -0.287471, -0.238986, 0.163853, -0.011015, -0.157153, 0.07223, -0.178795, -0.144772, 0.281839, -0.155577, -0.182964, -0.345084, -0.019109, -0.168965], [-0.402597, -0.462091, -0.153978, -0.008758, -0.010636, 0.005821, 0.16549, 0.108157, -0.165889, 0.129655, 0.243077, 0.123052, -0.023096, -0.02424, 0.005176, -0.205473], [-0.020986, 0.172782, 0.14201, -0.154099, 0.226242, 0.018116, 0.298613, 0.095807, 0.223712, -0.316769, 0.066814, 0.282766, 0.219608, 0.050183, 0.060557, 0.412293], [0.360352, 0.104648, 0.227762, 0.093181, -0.152627, 0.122338, 0.011422, -0.244258, 0.156627, -0.335221, -0.138786, -0.026285, -0.03944, 0.209091, 0.230249, -0.096027], [-0.2373, 0.276203, -0.151207, -0.183459, 0.113146, -0.316957, -0.323729, -0.214443, -0.549166, -0.545722, -0.306007, 0.013794, -0.231855, -0.184029, -0.213022, -0.345674], [0.105008, -0.033385, 0.020257, -0.042803, 0.114494, -0.087162, 0.174969, 0.155772, -0.270111, 0.368529, -0.028362, -0.129637, 0.22464, -0.15346, 0.112076, -0.132197], [-0.326655, -0.311393, 0.09321, -0.290837, 0.196464, -0.124869, -0.41587, -0.006736, -0.094195, -0.259213, -0.12144, -0.104175, -0.293479, -0.353942, -0.14919, -0.187726], [-0.137724, -0.172003, 0.399535, 0.055754, 0.301901, 0.231581, -0.013695, 0.036687, 0.297032, 0.051529, -0.025814, 0.034241, -0.710089, 0.122877, 0.085432, 0.272571], [-0.165261, -0.376782, -0.060239, 0.169418, -0.364879, 0.040563, 0.176788, -0.02571, 0.384875, -0.091007, -0.27605, 0.066861, 0.065832, 0.229137, 0.115694, 0.183925], [-0.092381, -0.292151, -0.321807, -0.135226, -0.002335, 0.006753, 0.038484, 0.115931, 0.072099, 0.012939, 0.127127, 0.164236, 0.203467, -0.012251, -0.181627, -0.129643], [-0.702438, 0.108929, 0.216788, 0.082903, 0.210626, 0.298581, 0.020791, 0.13643, 0.203811, 0.119922, 0.275091, -0.038992, 0.463125, 0.239866, 0.149395, 0.140013], [-0.3485, -0.620465, -0.016324, -0.157392, -0.032008, -0.250392, -0.641746, -0.270533, -0.450058, -0.916671, -0.084576, -0.046823, -0.587817, -0.085664, -0.194546, -0.102932], [0.246504, -0.114673, -0.224397, 0.062509, 0.23047, -0.137993, 0.047405, -0.157992, -0.383559, 0.184029, 0.21613, 0.242877, 0.148347, -0.041713, 0.267084, -0.277398], [0.15723, 0.160964, 0.024406, -0.007466, 0.121244, -0.226976, -0.326937, 0.157817, -0.116221, -0.214374, -0.167452, -0.050253, -0.044943, -0.020099, -0.566416, -0.157809], [-0.18016, 0.230478, 0.057695, 0.13208, -0.231676, 0.067684, -0.378068, 0.144551, 0.047685, -0.023444, -0.109432, -0.170885, 0.020752, 0.095672, 0.079399, 0.323492], [0.048883, -0.613161, -0.280796, 0.180664, -0.135343, -0.038502, 0.106481, -0.506259, -0.122277, 0.07387, -0.374753, 0.122646, 0.069817, 0.110234, 0.03066, -0.213079], [-0.172144, -0.049929, -0.035852, 0.099888, -0.418266, -0.211003, -0.53248, 0.10639, 0.020706, -0.024278, -0.18335, -0.206435, -0.415792, 0.13469, 0.044063, -0.21895], [-0.672608, 0.108809, 0.258901, -0.686894, 0.022315, -0.178744, 0.059231, -0.02346, 0.156348, -0.134402, 0.081674, 0.238632, 0.187131, 0.018118, -0.364292, 0.114662], [0.068445, -0.52844, 0.065403, 0.085357, 0.062741, -0.286067, 0.03819, -0.109817, 0.42996, 0.038811, 0.131041, -0.240868, -0.108942, 0.113798, 0.131209, 0.410191], [-0.649597, -0.583565, 0.176629, 0.057532, -0.304859, -0.363149, -0.016213, -0.27584, 0.05859, -0.247187, -0.057839, 0.010513, -0.086741, 0.115465, 0.12998, 0.195588], [-0.184562, 0.114213, -0.139852, 0.265131, 0.121709, 0.032297, 0.345461, 0.141177, 0.262222, 0.2721, 0.198301, 0.20502, 0.186772, 0.068783, 0.027751, -0.134918], [0.055493, -0.075657, 0.027155, 0.207544, -0.29735, -0.196854, 0.123179, 0.198473, -0.359918, -0.030321, -0.150516, -0.065933, -0.116781, 0.28125, -0.023237, -0.245008], [-0.788052, -0.019831, 0.033564, 0.305192, 0.07342, 0.255512, -0.029762, 0.32929, 0.20775, -0.606138, 0.530125, -0.629288, -0.6646, 0.351856, -0.19336, 0.318337], [-0.239917, 0.115766, 0.362643, -0.112908, 0.322164, 0.204796, 0.167827, 0.086099, 0.562873, 0.025406, 0.075343, 0.208854, 0.146914, 0.109855, 0.198312, 0.194904], [0.008603, -0.494123, -0.006431, 0.138564, -0.001211, 0.100063, -0.162548, -0.126419, -0.480373, -0.057593, -0.093269, 0.034244, 0.252594, -0.190247, 0.252206, -0.194272], [0.044285, 0.17934, 0.060316, -0.333937, -0.089352, -0.137312, 0.0309, -0.551808, -0.079208, 0.139583, 0.007606, 0.121473, 0.274735, 0.037158, -0.426159, 0.268233]], [0.187298, -0.305378, -0.173129, 0.27415, 0.029081, 0.308112, -0.026996, -0.070173, 0.057811, 0.169795, -0.45212, 0.02027, -0.169478, 0.124479, -0.171391, 0.261725]), ([[0.57972], [1.515585], [-0.237382], [0.371942], [0.333244], [0.717839], [0.531928], [0.389126], [-0.461194], [-0.519862], [0.805715], [0.398752], [0.618192], [-0.459359], [0.430841], [-0.64049]], [-0.032288])],
    [([[0.106878, 0.038348, 0.122825, 0.130447, -0.139024, -0.036941, -0.092239, -0.180223, 0.051699, -0.066312, -0.109457, 0.077089, -0.125999, -0.018087, 0.227331, -0.31715, 0.031669, 0.195974, 0.019187, -0.781108, -0.049449, 0.092355, 0.370729, 0.219907, -0.024972, 0.167634, 0.139818, 0.098711, 0.011782, -0.006387, 0.036762, 0.073598], [-0.083787, 0.007519, -0.02032, 0.013749, 0.188775, -0.034623, 0.170542, -0.036378, -0.240906, 0.152677, 0.04487, 0.022508, 0.167803, 0.109144, -0.085943, -0.021437, -0.070392, -0.052169, 0.061829, 0.240302, 0.245615, 0.102448, 0.19132, -0.289094, 0.033961, 0.261712, -0.007953, 0.042828, -0.204132, 0.067768, 0.214017, -0.09085], [-0.314889, 0.066606, 0.047045, 0.060837, 0.247421, 0.18058, -0.128331, -0.099459, -0.037176, -0.050328, -0.123672, -0.097767, 0.040269, -0.143214, -0.01754, -0.196278, -0.1381, -0.079381, -0.044046, 0.285996, 0.154674, 0.317547, 0.150368, -0.495587, 0.016788, -0.157493, -0.102177, 0.240385, -0.324491, -0.101265, 0.265642, -0.329995], [-0.208795, 0.321408, -0.122575, -0.092684, -0.01996, 0.037118, 0.074828, -0.134698, 0.19063, -0.052123, -0.075679, -0.10813, 0.179332, -0.08669, 0.031303, -0.132714, -0.239242, 0.034208, 0.213777, 0.429991, 0.162359, 0.172912, -0.333225, -0.146851, -0.162144, 0.147657, -0.149256, 0.207874, -0.165729, -0.021782, -0.112078, -0.060371], [-0.52415, 0.119469, 0.074868, -0.1597, -0.163518, 0.08846, -0.033035, -0.02021, -0.224957, 0.08555, -0.214459, 0.215108, -0.299147, -0.11906, -0.02068, 0.246664, 0.122174, 0.22085, 0.127032, 0.046643, -0.019693, 0.09991, -0.237198, 0.226731, -0.162422, 0.107008, 0.088166, -0.094484, -0.074657, -0.295744, -0.28942, 0.183388], [-0.076686, 0.118289, -0.065934, -0.096984, -0.032785, -0.119009, 0.149998, 0.212206, 0.133591, -0.149871, -0.373125, -0.111646, 0.051104, 0.135657, -0.068898, -0.119343, -0.048246, 0.213269, -0.136178, 0.081742, -0.097118, -0.154291, 0.014532, -0.209779, -0.007839, -0.210551, -0.035153, -0.002019, 0.222637, 0.165043, 0.194161, 0.003994], [0.029684, 0.043718, -0.281033, 0.410697, -0.715509, -0.032006, -0.120122, -0.070894, -0.290049, 0.255228, 0.062702, -0.74072, 0.116938, 0.032425, -0.47507, 0.042726, 0.122574, 0.291472, -0.002562, 0.520563, -0.440144, 0.013742, -0.153311, -0.020803, -0.456493, 0.103088, -0.189494, -0.186353, -0.029488, -0.260636, 0.204899, -0.466447], [-0.013194, 0.061537, 0.067995, -0.133499, -0.020607, 0.084614, 0.236155, -0.312235, -0.037342, 0.197453, -0.145938, -0.09448, 0.111591, 0.161024, 0.055163, 0.184077, 0.128068, 0.266478, 0.010349, -0.053968, 0.14558, 0.146507, 0.124623, 0.068634, 0.03875, 0.171811, 0.000994, -0.020134, -0.113005, -0.108145, 0.024704, 0.165191], [-0.102668, -0.0287, 0.172795, 0.084322, 0.36213, 0.596378, -0.13899, -0.385187, -0.161239, 0.607317, 0.337194, -0.055051, 0.001314, -0.072104, -0.321449, -0.246528, -0.055279, -0.584969, 0.31146, 0.168117, 0.343564, 0.459972, -0.063265, -0.354825, 0.223966, -0.299686, -0.200256, 0.324312, -0.142979, 0.124107, -0.335716, -0.283227], [0.240353, 0.002671, 0.066619, -0.052363, -0.054081, 0.264623, -0.008309, -0.248231, 0.177734, 0.199735, -0.319199, 0.520819, -0.233176, 0.218145, 0.122154, -0.655692, 0.270848, -0.147512, 0.288342, -0.241161, -0.138065, 0.338091, -0.514302, 0.221146, -0.175599, 0.353568, -0.116451, -0.946109, -0.317287, 0.064831, 0.374002, -0.67604], [0.149633, -0.17297, -0.210283, -0.014124, -0.027941, -0.141659, 0.030888, -0.001852, 0.052091, 0.090934, 0.196572, -0.175623, 0.037504, 0.168598, 0.002755, 0.095932, -0.015421, 0.106886, -0.048575, -0.132442, -0.030031, 0.047036, 0.076974, 0.180065, -0.025348, -0.029491, -0.153113, 0.024773, -0.109361, -0.070255, -0.068568, 0.10088], [-0.46051, -0.017343, 0.188029, 0.385587, 0.121162, 0.283293, -0.115371, -0.086798, -0.304531, 0.077977, 0.262823, -0.556929, 0.014393, -0.26542, -0.123115, -0.089151, -0.232782, -0.312241, 0.411148, 0.023367, -0.045265, 0.164995, -0.059875, -0.114794, 0.251725, 0.184351, 0.125041, 0.235324, 0.133772, -0.021359, 0.344182, -0.359994], [0.343023, -0.100615, 0.026635, 0.101527, -0.077002, -0.056267, -0.298515, -0.082199, 0.003165, -0.209927, 0.459716, -0.130853, 0.06453, 0.227693, -0.084176, 0.063949, -0.003179, -0.078132, 0.098951, 0.059087, -0.207103, -0.04355, -0.11598, -0.326504, -0.409708, 0.00581, 0.089202, -0.078696, -0.097789, 0.030468, -0.300037, 0.007804], [-0.221201, 0.139703, 0.28159, -0.273267, -0.109815, -0.024641, 0.037235, -0.002088, 0.15196, -0.138516, 0.050482, -0.180625, 0.383071, -0.17497, -0.061343, -0.050854, 0.020819, 0.319511, -0.058234, -0.035883, -0.067958, 0.002177, 0.078402, -0.111593, 0.01813, -0.122035, 0.375688, 0.012053, -0.283207, -0.048203, -0.338014, 0.105029], [0.189268, -0.214395, -0.533816, -0.089668, 0.031796, -0.031729, -0.291685, -0.114405, -0.211747, 0.076283, 0.236249, -0.14703, -0.06025, -0.120275, -0.133556, -0.134558, 0.283723, -0.334529, -0.106356, 0.317084, -0.016678, -0.117394, 0.063868, -0.199293, -0.132474, 0.685977, 0.249588, 0.140182, -0.05487, 0.141705, -0.068682, 0.18161], [0.146244, 0.037678, 0.214074, 0.163582, 0.042316, 0.115586, 0.19059, -0.024424, -0.266997, -0.282206, -0.160513, -0.013144, -0.059215, -0.232835, -0.018853, 0.097374, -0.461485, 0.023176, 0.048968, -0.264151, -0.058606, -0.207998, -0.038633, -0.070344, 0.001495, 0.122762, 0.249388, 0.205729, 0.008229, -0.162769, 0.037835, 0.044811], [0.107171, 0.313069, 0.496124, 0.414196, -0.100934, 0.015894, 0.274669, 0.105158, -0.269172, -0.100891, 0.347637, -0.041601, -0.60284, 0.143928, 0.266152, -0.080858, -0.599377, 0.066048, -0.131197, -0.32206, 0.155931, 0.256665, -0.191276, 0.300163, -0.006711, -0.396329, 0.162769, -0.317516, 0.041695, -0.668636, -0.533371, 0.124434], [-0.479778, 0.486037, -0.33258, -0.13999, -0.224098, 0.13997, -0.000765, 0.405713, -0.30751, 0.134296, -0.234963, -0.208273, 0.376483, -0.149509, 0.110574, 0.080623, -0.025651, -0.469742, 0.375215, -0.163261, -0.247472, 0.159518, -0.046663, -0.059115, -0.128228, 0.354522, 0.378757, 0.192914, -0.209791, -0.325868, -0.301269, -0.18724], [-0.010367, 0.120142, -0.029952, 0.074001, 0.008468, 0.014789, -0.018829, -0.022215, 0.056314, 0.072842, -0.015575, 0.046096, 0.028359, -0.020423, -0.041206, 0.011115, -0.041318, -0.033287, -0.415773, -0.015985, -0.027547, 0.035754, -0.009906, -0.043967, 0.003056, -0.094669, 0.087766, -0.078823, 0.036291, 0.043613, 0.054972, -0.015285], [-0.09255, 0.259167, 0.026862, 0.1598, 0.532342, -0.048612, -0.031209, 0.031404, -0.041335, 0.312801, 0.108459, 0.232249, 0.047629, -0.092229, -1.12032, 0.033402, -0.028561, 0.345658, 0.041623, -0.053279, 0.144045, -0.057592, 0.087148, -0.122294, -0.307067, 0.036023, -0.073025, 0.008933, -0.248078, 0.306057, 0.106501, -0.094384], [0.394195, 0.039631, -0.094724, 0.072239, 0.224088, -0.043559, 0.212183, 0.166896, 0.196986, 0.105002, 0.142228, -0.068385, 0.141239, 0.050843, 0.054581, 0.12128, -0.024288, 0.25618, 0.271733, -0.101658, 0.140477, 0.085324, -0.016798, 0.099319, 0.136854, 0.053682, 0.234819, 0.169831, -0.04541, 0.544844, 0.107654, 0.028799], [-0.030157, 0.273768, -0.481133, -0.281851, -0.16625, 0.009523, 0.379978, -0.18957, -0.305623, 0.033496, -0.117641, -0.120185, -0.466763, -0.388763, -0.110504, -0.044323, -0.28571, -0.399847, 0.256283, -0.091242, -0.0211, 0.078213, -0.025794, -0.094434, 0.319591, -0.252941, -0.013329, 0.058187, -0.109392, 0.30906, -0.081364, 0.260004], [-0.060186, -0.027271, 0.033302, -0.02582, -0.061505, -0.048017, 0.137427, -0.012891, 0.003748, 0.034325, -0.014072, -0.194783, 0.006254, -0.039119, 0.0071, -0.047648, -0.012037, 0.015912, 0.056766, -0.005352, -0.08452, 0.110025, 0.02247, 0.227174, 0.040136, 0.032675, 0.016197, 0.130192, -0.125856, -0.098963, -0.097843, 0.040436], [-0.058199, -0.386101, -0.00793, 0.470388, 0.080986, 0.582296, 0.430014, 0.545278, -0.387257, 0.334253, -0.478126, -0.125269, -0.855784, 0.023169, -0.034037, 0.30856, 0.736295, -0.025691, -0.215114, 0.367427, -0.517343, -0.676841, 0.341022, -0.075891, 0.475137, 0.079898, 0.273196, -0.013324, -0.425219, -0.194677, -0.103658, -0.343369], [0.192081, 0.434016, -0.030693, -0.403283, 0.123618, -0.052643, 0.07945, -0.240637, 0.211519, 0.080388, 0.115298, -0.079018, -0.098056, -0.158116, -0.049387, 0.219952, 0.081185, -0.146997, -0.121063, 0.012009, 0.360192, 0.203912, 0.160498, 0.14848, -0.146376, 0.035386, 0.305164, -0.208054, 0.317901, 0.090564, -0.102582, -0.178737]], [-0.459468, -0.162747, -0.216455, -0.342552, -0.054231, -0.458899, -0.35936, -0.095977, 0.147399, 0.125987, -0.035017, -0.616601, 0.173552, 0.088721, 0.173153, -0.274045, -0.124094, 0.240653, -0.463743, -0.321713, -0.262066, -0.223624, -0.308072, 0.18158, -0.890883, -0.635092, -0.784682, -0.241284, -0.251249, 0.109613, 0.12957, -0.170336]), ([[-0.195961, 0.260379, -0.084089, 0.021898, 0.174804, -0.031999, 0.145724, 0.401914, 0.028461, -0.264018, 0.097084, 0.113466, 0.315323, -0.315862, -0.323618, -0.026394], [0.155468, 0.234046, -0.086172, 0.105494, -0.34157, 0.362973, 0.241782, 0.103554, -0.203304, 0.03065, -0.124399, -0.339855, 0.006589, 0.06382, 0.11503, 0.231977], [-0.146738, 0.211522, 0.321057, 0.17389, -0.175873, 0.429677, -0.045936, -0.120435, -0.029211, 0.270912, 0.128444, -0.140486, 0.32585, -0.076079, 0.004335, -0.238139], [-0.272922, -0.123734, 0.118015, -0.067765, 0.160913, -0.423793, -0.294927, -0.144006, 0.08147, 0.049988, 0.048749, 0.042929, 0.134598, -0.085213, 0.051051, -0.617248], [0.002545, -0.132471, 0.124598, -0.248433, 0.303395, 0.08366, -0.630957, -0.324599, -0.170236, 0.135385, 0.037924, -0.482851, -0.041486, 0.163898, 0.128943, -0.044427], [-0.128184, 0.027727, 0.029459, -0.383722, -0.141236, -0.703624, -0.388282, -0.102, -0.272296, 0.251764, -0.007694, -0.193414, 0.049635, 0.211915, 0.054642, -0.242398], [0.121342, -0.059732, -0.258745, -0.232894, 0.076058, -0.146201, 0.093438, 0.04118, 0.160211, -0.221752, -0.240506, -0.04792, -0.424084, -0.24284, -0.261955, 0.005982], [0.139147, -0.271381, -0.025143, -0.366204, 0.240984, -0.022464, 0.161716, 0.106953, 0.25777, -0.085209, -0.242558, 0.418111, -0.006953, 0.092276, 0.094275, -0.147108], [0.233203, -0.135593, -0.013533, 0.115654, 0.259729, -0.067703, 0.136301, 0.265742, 0.267315, -0.226372, -0.175499, 0.19856, 0.03121, -0.044142, 0.179114, -0.019343], [0.027791, -0.109418, 0.041724, 0.022185, 0.048819, -0.072671, -0.126371, 0.046473, -0.319963, 0.044999, 0.224375, -0.00946, 0.049172, 0.010435, 0.012662, -0.238095], [-0.582836, 0.028083, 0.330065, 0.139204, -0.168175, -0.126214, -0.141178, 0.087856, -0.164527, -0.074159, -0.372967, -0.252837, 0.126282, -0.02951, 0.009572, -0.465607], [0.004085, 0.002239, -0.00195, 0.108217, -0.027999, -0.088606, 0.120209, -0.354987, 0.23219, 0.336695, 0.170537, -0.470621, -0.007296, -0.0514, 0.304998, 0.331907], [-0.124239, -0.202757, 0.023684, 0.033201, -0.232734, -0.337768, -0.180941, -0.291656, 0.047487, -0.090673, -0.752711, -0.136871, -0.304335, -0.20746, 0.131721, -0.150441], [-0.057748, -0.11902, -0.035313, -0.149442, -0.138893, -0.294997, -0.05855, 0.135247, 0.019898, -0.431252, 0.19335, -0.083758, -0.145234, -0.049196, -0.013717, 0.230522], [-0.049357, -0.58009, -0.260393, -0.17541, -0.045152, -0.035545, -0.065715, 0.033847, 0.02702, -0.226242, -0.44343, -0.037142, -0.257934, -0.228759, -0.598006, -0.093119], [0.337768, 0.269328, 0.298752, 0.214014, -0.072734, 0.172327, 0.090761, 0.143151, 0.168616, 0.306565, 0.385855, 0.099756, -0.139203, 0.168435, 0.056143, 0.163447], [-0.703262, -0.254656, -0.207074, -0.316171, 0.210021, 0.049989, -0.125772, 0.100508, 0.102255, -0.565563, -0.334128, -0.029511, 0.18666, -0.077119, 0.175681, 0.006066], [-0.121312, 0.21761, -0.192843, -0.039872, -0.309471, -0.257279, 0.033868, -0.002633, -0.30513, -0.143711, 0.243604, -0.111732, -0.003973, 0.0964, -0.264934, 0.155535], [0.293928, 0.192477, 0.13389, 0.157831, 0.001264, -0.578917, 0.091144, -0.070413, 0.026578, 0.387159, -0.003877, -0.124933, -0.162341, 0.012009, -0.094274, 0.293381], [0.170017, 0.35088, 0.13634, 0.098621, 0.108779, -0.401813, 0.332209, 0.318615, 0.158995, 0.148824, 0.216127, 0.418511, 0.271217, 0.011873, 0.139489, 0.12395], [0.101603, -0.046943, -0.032791, -0.092734, -0.405539, 0.058951, 0.105473, -0.002045, -0.050945, 0.049488, -0.193881, -0.111758, 0.308775, -0.067698, 0.139889, -0.137966], [0.199459, 0.191711, 0.007159, -0.08528, 0.036132, -0.068894, 0.094751, -0.286663, -0.36915, -0.087138, -0.194608, -0.36725, 0.22625, -0.187906, 0.197414, 0.119355], [-0.037061, 0.246719, 0.364043, 0.009573, -0.010809, 0.345514, 0.162805, 0.014845, 0.33403, 0.355645, 0.386355, 0.374769, 0.185003, -0.087305, 0.189907, 0.188308], [0.071237, 0.121855, 0.111966, 0.27227, 0.019975, 0.00635, 0.29477, -0.140823, -0.186164, 0.304287, 0.202817, 0.132051, -0.046146, -0.10288, 0.123311, 0.06473], [0.017727, -0.077775, 0.046942, -0.951012, 0.024385, 0.404466, 0.134838, -0.675637, 0.056567, -0.231944, -0.135273, -0.344316, 0.293998, 0.093166, 0.345512, 0.295911], [0.421972, 0.030157, -0.075836, 0.048864, 0.28157, -0.011017, 0.069452, 0.39461, 0.511567, -0.003121, 0.209268, 0.263033, 0.044214, -0.044384, 0.011475, -0.10162], [-0.138873, -0.395515, 0.286528, 0.340393, -0.126383, -0.013561, -0.633348, 0.091548, -0.686841, 0.332323, 0.244015, 0.056949, 0.110502, -0.403819, 0.21836, -0.230072], [0.072182, 0.23264, 0.014947, 0.035153, -0.177362, -0.376001, 0.042834, 0.103203, -0.459376, 0.216021, -0.060137, -0.311256, 0.191539, 0.140599, 0.166023, 0.062131], [-0.14908, -0.16683, -0.216009, -0.052367, 0.133275, 0.062219, 0.271886, -0.170537, -0.022368, -0.610607, -0.465274, 0.264796, 0.039456, -0.393996, -0.220775, 0.103485], [-0.079143, -0.100953, -0.15713, -0.199918, -0.014988, -0.637186, -0.252498, -0.213624, -0.15976, -0.339217, -0.417628, -0.278785, -0.082255, -0.175145, 0.035811, 0.145255], [-0.032049, 0.291058, 0.200441, -0.055094, 0.278222, -0.028172, -0.194986, 0.242713, 0.057192, -0.205054, 0.014219, 0.196461, 0.28783, 0.0466, 0.077497, 0.109224], [0.28503, 0.346687, -0.199762, 0.174797, 0.180254, -0.24555, 0.334749, -0.241026, 0.158956, -0.307914, -0.166767, 0.018019, -0.279753, -0.121416, -0.367712, 0.250178]], [-0.089714, 0.117509, 0.115517, 0.230247, -0.002988, -0.086154, -0.073055, -0.201175, -0.103189, -0.20243, 0.130551, -0.172452, -0.205447, -0.231288, -0.248363, -0.241575]), ([[0.352139], [0.265273], [0.161688], [-0.282702], [-0.252091], [-1.001606], [-0.372343], [-0.748058], [-0.682336], [0.867097], [0.604244], [-0.403009], [0.532768], [0.284304], [0.277077], [-0.633788]], [0.030465])],
]
# MLP2 end
PRED_TAUS_US = (20000.0, 50000.0, 70000.0, 150000.0, 300000.0)

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


class _Order:
    """Our record of one of our orders; `remaining` bounds what the exchange can still fill."""

    def __init__(self, order_id: str, k: int, sign: int, price: int, quantity: int, ioc: bool, slot: int) -> None:
        self.order_id = order_id
        self.k = k  # 0 = buy side, 1 = sell side
        self.slot = slot  # 0/1 touch quote, 2/3 second level (side = slot & 1)
        self.rung = False  # a standing ladder order
        self.parked = False  # a shrunk stub left behind the touch for its queue priority
        self.sign = sign  # +1 buy, -1 sell
        self.price = price
        self.remaining = quantity
        self.ioc = ioc
        self.accepted = False
        self.cancel_sent = False
        self.active = not ioc  # part of the side's current quote (not cancelling, not an IOC)
        self.zq = False  # cancelled because the quote moved TOWARD the market (zd_quote 2 credits only these)


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
        'gate_thr': 10.0,
        'gate_window_us': 30000000,
        'gate_max_spread': 3,
        'gate_rungs': 1,
        'fair_thr': 0.0,
        'fair_tau_ms': 70.0,
        'fair_levels': 20,
        # Fair-centric quoting (see the module docstring); fq_k 0 = off.
        'fq_k': 0.05,
        'fq_scale': 1.0,
        'fq_max_behind': 0,
        'fq_inside': 0,
        'fq_take': 200,
        'fq_slip': 0,
        'fq_take_cooldown_ms': 0.0,
        'fq_tol': 0,
        'fq_level': 0,
        'fq_take_k': 0.2,
        'fair_mid_any': 1,
        'pred_on': 0,
        'pred_scale': 1.0,
        'pred_model': 0,
        'pred_mode': 1,
        'pred_blend': 0.85,
        # exp_r3_pms: 1 = the IOC taker uses the round-3 150 ms ridge (pred_on 1, pred_mode 1, pred_model 3), 2 = the
        # 500 ms one (pred_model 4); blended with fair_sig by pred_blend, scaled by pred_scale. 0 = off (base).
        'pms_on': 1,
        # exp_r3_rfit: 0 = champion model 3; else the PRED_MODELS index of a refit variant (5 parity, 6 r3_100, 7 r3m_75, 8 r3m_100,
        # 9 r3r2_75, 10 r3r2_100, 11 h100, 12 h200, 13 alpha x3, 14 alpha /3, 15 h250, 16 h300, 17 h400, 18 h200+r2, 19 h200+mirror).
        'rfit': 19,
        # exp_r3_mlp: mlp_on 1 = the taker prediction is (1 - mlp_mix) * ridge + mlp_mix * MLP (before pred_blend with fair_sig);
        # the MLP runs only when |ridge| >= mlp_gate ticks (runtime). 0 = off (identical to the champion).
        'mlp_on': 1,
        'mlp_mix': 1.0,
        'mlp_gate': 0.3,
        # exp_r3_mlp2: 1 = the MLP is the average of the MLP2_NETS ensemble (seed ensemble / 2 horizons / extra inputs MLP2_COLS)
        # instead of MLP_LAYERS. 0 = off (identical to mm_r3_mlp).
        'mlp2_on': 0,
        # exp_r2_new_a, zero-delay take: the IOC taker counts the room of the favoured-side quote it cancels in the
        # same callback (and of cancels still in flight), so IOC and cancel land together after 5 ms instead of the
        # IOC waiting for CANCELED (>= 10 ms). A race fill of the cancelled quote can push |I| over max_position;
        # the guard then sends an IOC back to the cap at once (penalty ~50 per 5 ms). 0 = off (champion).
        'zd_take': 1,
        'zd_guard_slip': 3,
        # Credit a cancelling quote only when at least this much market quantity is queued ahead of it (a race fill
        # in the 5 ms window then needs a sweep through all of it); 0 = always credit.
        'zd_min_ahead': 0,
        # Zero-delay requote: a passive quote moved to a new price is re-placed in the same callback as the cancel of
        # the old one (crediting the cancelling quantity) instead of after CANCELED: it reaches the new level >= 5 ms
        # earlier, i.e. further forward in its queue. Same over-cap guard. 0 = off.
        'zd_quote': 2,
        'rt_ms': 100.0,
        'ri_ms': 0.0,
        'qa_max': 0,
        'fq_k_sp2': 1.5,
        'of_ms': 0.0,
        'fq_k_sp3': 0.15,
        'hk_gap': 3,
        'hk_k': 0.6,
        'qs_k': 0.6,
        'qs_weak': 100,
        # zd_quote 2: credit only quotes cancelled because the desired price moved toward the market (following an
        # improving touch onto a fresh level), never retreats, whose old quote is the one at risk of a race fill.
        # Guard also pulls the over-cap side's resting quotes (1) or only IOCs the excess (0).
        'zd_guard_pull': 1,
        # Take-and-rest: the taker is a LIMIT at the far touch instead of an IOC; when the touch is thinner than the
        # order, the residual rests at the swept price, alone at the front of a fresh best level inside the spread,
        # and stays while floor(F_t - tr_keep_k) >= its price (asks mirrored). 0 = off.
        'tr_rest': 0,
        'tr_keep_k': 0.0,
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
        self.fair_ls = 0.0  # time-decayed sum of (insert price - side offset), same weights as fair_s
        self.fq_k = float(params['fq_k'])
        self.fq_on = self.fq_k > 0.0
        self.fq_scale = float(params['fq_scale'])
        self.fq_max_behind = int(params['fq_max_behind'])
        self.fq_inside = max(0, int(params['fq_inside']))
        self.fq_take = max(0, int(params['fq_take']))
        self.fq_slip = max(0, int(params['fq_slip']))
        self.fq_cool_us = int(float(params['fq_take_cooldown_ms']) * 1000.0)
        self.fq_tol = max(0, int(params['fq_tol']))
        self.fq_level = int(params['fq_level'])
        self.fq_take_k = float(params['fq_take_k'])
        self.fair_mid_any = int(params['fair_mid_any'])
        self.fq_next_take = [0, 0]  # earliest time of the next IOC take per side
        if int(params.get('pms_on', 0)):
            params = dict(params, pred_on=1, pred_mode=1, pred_model=2 + int(params['pms_on']))
            if int(params.get('rfit', 0)):
                params = dict(params, pred_model=int(params['rfit']))
        self.pred_on = int(params['pred_on'])
        self.pred_scale = float(params['pred_scale'])
        self.pred_mode = int(params['pred_mode'])
        self.pred_blend = float(params['pred_blend'])
        self.pred_c0, self.pred_w = PRED_MODELS[int(params['pred_model'])]
        self.mlp_on = int(params['mlp_on'])
        self.mlp_mix = float(params['mlp_mix'])
        self.mlp_gate = float(params['mlp_gate'])
        if self.mlp_on:
            self.mlp_mean = np.array(MLP_MEAN)
            self.mlp_scale = np.array(MLP_SCALE)
            self.mlp_layers = [(np.array(w), np.array(b)) for w, b in MLP_LAYERS]
        self.mlp2_on = int(params['mlp2_on'])
        if self.mlp2_on:
            self.mlp2_mean = np.array(MLP2_MEAN)
            self.mlp2_scale = np.array(MLP2_SCALE)
            self.mlp2_nets = [[(np.array(w), np.array(b)) for w, b in net] for net in MLP2_NETS]
            self.mlp2_x = [c in ('ioc_sweep50', 'spread_max100', 'nev200') for c in MLP2_COLS]
        self.sw_w: deque = deque()  # (t, signed levels traded through) of the last 50 ms, as mlp2_r3.sweep_event
        self.sw_s = 0.0
        self.sw_pbb = 0  # market touch before the current event
        self.sw_pba = 0
        self.nv_w: deque = deque()  # event times of the last 200 ms
        self.sp_w: deque = deque(maxlen=200)  # touch spreads of the last 200 events
        self.pred_sig = 0.0
        self.pf_s = [0.0] * 5  # decayed sums of insert devs at PRED_TAUS_US
        self.pf_w = [0.0] * 5
        self.pf_sig = [0.0] * 5
        self.pf_t = 0
        self.pf_seen = False
        self.pr_hist: deque = deque()  # signed market flow of the last 10 events
        self.pr_ev10 = 0.0
        self.pr_w100: deque = deque()
        self.pr_s100 = 0.0
        self.pr_w1000: deque = deque()
        self.pr_s1000 = 0.0
        self.pr_ltp = 0
        self.pr_ltv = 0.0
        self.pr_ttr = -1
        self.pr_rm2 = 0
        self.n_takes = 0
        self.zd_take = int(params['zd_take'])
        self.zd_guard_slip = max(0, int(params['zd_guard_slip']))
        self.zd_min_ahead = max(0, int(params['zd_min_ahead']))
        self.last_views = [None, None]
        self.zd_quote = int(params['zd_quote'])
        self.rt_us = int(float(params['rt_ms']) * 1000.0)
        self.qs_k = float(params['qs_k'])
        self.qs_weak = int(params['qs_weak'])
        self.qs_f = 0.0
        self.rt_t = [-10**15, -10**15]
        self.ri_us = int(float(params['ri_ms']) * 1000.0)
        self.ri_t = [-10**15, -10**15]
        self.qa_max = int(params['qa_max'])
        self.fq_k_sp2 = float(params['fq_k_sp2'])
        self.of_us = int(float(params['of_ms']) * 1000.0)
        self.fq_k_sp3 = float(params['fq_k_sp3'])
        self.hk_gap = int(params['hk_gap'])
        self.hk_k = float(params['hk_k'])
        self.hole = [0, 0]
        self.of_t = [-10**15, -10**15]
        self.of_now = 0
        self.zd_guard_pull = int(params['zd_guard_pull'])
        self.tr_rest = int(params['tr_rest'])
        self.tr_keep_k = float(params['tr_keep_k'])
        self.last_bb = 0
        self.last_ba = 0
        self.n_guard = 0

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
        if self.pred_on:
            self._pred_flow(snapshot.timestamp_us, trades)
        if not bids or not asks:
            return
        scan = self.book_scan_orders
        need_qty = self.alpha_imbalance != 0.0 or self.imb_pull > 0.0 or self.jump_imb > 0.0 or self.thin_pull > 0 or self.qa_max > 0

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
        self.of_now = snapshot.timestamp_us
        if self.hk_gap > 0:
            g = 99
            for view in bids:
                if view.owner == 'market' and view.price < bb:
                    g = bb - view.price
                    break
            self.hole[0] = g
            g = 99
            for view in asks:
                if view.owner == 'market' and view.price > ba:
                    g = view.price - ba
                    break
            self.hole[1] = g
        if self.ri_us > 0:
            if self.last_ba > 0 and ba < self.last_ba:
                self.ri_t[0] = snapshot.timestamp_us  # the ask improved: F likely lower, our bid is at risk
            if 0 < self.last_bb < bb:
                self.ri_t[1] = snapshot.timestamp_us
        if self.rt_us > 0:
            if 0 < bb < self.last_bb:
                self.rt_t[0] = snapshot.timestamp_us
            if self.last_ba > 0 and ba > self.last_ba:
                self.rt_t[1] = snapshot.timestamp_us
        self.last_bb = bb
        self.last_ba = ba
        if self.zd_take:
            self.last_views[0] = bids
            self.last_views[1] = asks
            if self.inv > self.max_pos or self.inv < -self.max_pos:
                self._guard()  # a guard IOC that found too little liquidity: try again

        if self.fair_thr > 0.0 or self.fq_on:
            self._fair_update(snapshot.timestamp_us, bids, asks, bb, ba)
            if self.pred_on:
                self._pred_update(snapshot.timestamp_us, bids, asks, bb, ba)
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
        if self.of_us > 0:
            if now - self.of_t[0] < self.of_us:
                pull_bid = True
            if now - self.of_t[1] < self.of_us:
                pull_ask = True
        if self.qa_max > 0:
            # do not START a quote behind a thick market queue: only sweeps reach its back (markout -0.4..-0.7)
            if not self.active[0] and qb > self.qa_max:
                pull_bid = True
            if not self.active[1] and qa > self.qa_max:
                pull_ask = True
        if self.ri_us > 0:
            if now - self.ri_t[0] < self.ri_us:
                pull_bid = True
            if now - self.ri_t[1] < self.ri_us:
                pull_ask = True
        if self.rt_us > 0:
            if now - self.rt_t[0] < self.rt_us:
                pull_bid = True
            if now - self.rt_t[1] < self.rt_us:
                pull_ask = True
        if self.fq_on:
            bid, ask, pull_bid, pull_ask = self._fq(now, bb, ba, fair_shift, pull_bid, pull_ask)
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
            qsz = self.quote_size
            if self.qs_k > 0.0 and self.qs_f - bid < self.qs_k:
                qsz = self.qs_weak
            self._quote(0, 1, _BUY, bid, now, qsz)
            if lay:
                self._quote(2, 1, _BUY, bid - off, now, lay)
        if pull_ask:
            self._pull(1)
            self._pull(3)
        else:
            qsz = self.quote_size
            if self.qs_k > 0.0 and ask - self.qs_f < self.qs_k:
                qsz = self.qs_weak
            self._quote(1, -1, _SELL, ask, now, qsz)
            if lay:
                self._quote(3, -1, _SELL, ask + off, now, lay)
        if ladder and self.ladder_reserve:
            self._ladder(0, 1, _BUY, bb, ask, pull_bid)
            self._ladder(1, -1, _SELL, ba, bid, pull_ask)

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
        if self.pred_on:
            dtp = now - self.pf_t if self.pf_seen else 0
            ps = self.pf_s
            pw = self.pf_w
            pg = self.pf_sig
            for i in range(5):
                ai = exp(-dtp / PRED_TAUS_US[i])
                ps[i] = ps[i] * ai + dev
                pw[i] = pw[i] * ai + 1.0
                pg[i] = ps[i] / pw[i]
            self.pf_t = now
            self.pf_seen = True
        a = exp(-(now - self.fair_t) / self.fair_tau_us)
        self.fair_t = now
        self.fair_s = self.fair_s * a + dev
        self.fair_ls = self.fair_ls * a + (px - self.fair_off[k])
        self.fair_w = self.fair_w * a + 1.0
        self.fair_sig = self.fair_s / self.fair_w

    def _pred_flow(self, now: int, trades: list[Trade]) -> None:
        """Market trade flow windows (our own IOC trades are not market flow), as research.signals.OnlineFeatures."""
        fl = 0.0
        traded = False
        for t in trades:
            if t.taker_owner == 'user':
                continue
            traded = True
            fl += t.quantity if t.book_side == 'ask' else -t.quantity
            self.pr_ltp = t.price
        if self.mlp2_on:
            sw = self.sw_w
            if traded and fl != 0.0:
                if fl > 0.0:
                    x = min(max(self.pr_ltp - self.sw_pba + 1, 0), 11)
                else:
                    x = -min(max(self.sw_pbb - self.pr_ltp + 1, 0), 11)
                if x:
                    sw.append((now, x))
                    self.sw_s += x
            lim = now - 50000
            while sw and sw[0][0] <= lim:
                self.sw_s -= sw.popleft()[1]
            nv = self.nv_w
            nv.append(now)
            lim = now - 200000
            while nv[0] <= lim:
                nv.popleft()
        if traded:
            self.pr_ltv = fl
            self.pr_ttr = now
        h = self.pr_hist
        h.append(fl)
        self.pr_ev10 += fl
        if len(h) > 10:
            self.pr_ev10 -= h.popleft()
        if fl:
            self.pr_w100.append((now, fl))
            self.pr_s100 += fl
            self.pr_w1000.append((now, fl))
            self.pr_s1000 += fl
        w = self.pr_w100
        lim = now - 100000
        while w and w[0][0] <= lim:
            self.pr_s100 -= w.popleft()[1]
        w = self.pr_w1000
        lim = now - 1000000
        while w and w[0][0] <= lim:
            self.pr_s1000 -= w.popleft()[1]

    def _pred_update(self, now: int, bids, asks, bb: int, ba: int) -> None:
        """Evaluate the baked ridge on this event's features (see PRED_NAMES) into pred_sig (ticks)."""
        qb = nb = 0
        for v in bids:
            if v.owner != 'market':
                continue
            if v.price != bb:
                break
            qb += v.quantity
            nb += 1
        qa = na = 0
        for v in asks:
            if v.owner != 'market':
                continue
            if v.price != ba:
                break
            qa += v.quantity
            na += 1
        spread = ba - bb
        mid2 = bb + ba
        if spread <= 3:
            self.pr_rm2 = mid2
        rm2 = self.pr_rm2 if self.pr_rm2 > 0 else mid2
        pg = self.pf_sig
        f70 = pg[2]
        af = f70 if f70 >= 0.0 else -f70
        fw = self.pf_w[2] * exp(-(now - self.pf_t) / 70000.0) if self.pf_seen else 0.0
        s100 = self.pr_s100
        ltv = self.pr_ltv
        ltp = self.pr_ltp
        w = self.pred_w
        p = (self.pred_c0 + w[0] * pg[0] + w[1] * pg[1] + w[2] * f70 + w[3] * pg[3] + w[4] * pg[4]
             + w[5] * f70 * log1p(fw) + w[6] * f70 * af + w[8] * f70 * spread + w[9] * spread
             + w[17] * (mid2 - rm2) * 0.5 + w[18] * self.pr_ev10 + w[19] * s100 + w[20] * self.pr_s1000)
        if s100 > 0.0:
            p += w[7] * af
        elif s100 < 0.0:
            p -= w[7] * af
        if qb + qa > 0:
            p += w[10] * (qb - qa) / (qb + qa) + w[11] * (nb - na) / (nb + na)
        p += w[12] * log1p(qb) + w[13] * log1p(qa)
        if ltv > 0.0:
            p += w[14] + w[15] * log1p(ltv)
        elif ltv < 0.0:
            p -= w[14] + w[15] * log1p(-ltv)
        if ltp > 0:
            p += w[16] * (ltp - mid2 * 0.5)
        p += w[21] * (log1p((now - self.pr_ttr) / 1000.0) if self.pr_ttr >= 0 else 10.0)
        if self.mlp2_on:
            self.sw_pbb = bb
            self.sw_pba = ba
            self.sp_w.append(spread)
        if self.mlp_on and (p >= self.mlp_gate or p <= -self.mlp_gate):
            # same 22 features as the ridge, in PRED_NAMES order
            ltr_sign = 1.0 if ltv > 0.0 else (-1.0 if ltv < 0.0 else 0.0)
            ltr_lsz = ltr_sign * log1p(ltv if ltv > 0.0 else -ltv)
            sg100 = 1.0 if s100 > 0.0 else (-1.0 if s100 < 0.0 else 0.0)
            if qb + qa > 0:
                imb1 = (qb - qa) / (qb + qa)
                cnt1 = (nb - na) / (nb + na)
            else:
                imb1 = cnt1 = 0.0
            h = np.array((pg[0], pg[1], f70, pg[3], pg[4], f70 * log1p(fw), f70 * af, sg100 * af, f70 * spread, spread,
                          imb1, cnt1, log1p(qb), log1p(qa), ltr_sign, ltr_lsz, (ltp - mid2 * 0.5) if ltp > 0 else 0.0,
                          (mid2 - rm2) * 0.5, self.pr_ev10, s100, self.pr_s1000,
                          log1p((now - self.pr_ttr) / 1000.0) if self.pr_ttr >= 0 else 10.0))
            if self.mlp2_on:
                ex = {'ioc_sweep50': self.sw_s, 'spread_max100': float(max(self.sp_w)), 'nev200': log1p(len(self.nv_w))}
                h = np.concatenate((h, [ex[c] for c in MLP2_COLS[22:]]))
                h = np.clip((h - self.mlp2_mean) / self.mlp2_scale, -8.0, 8.0)
                m = 0.0
                for net in self.mlp2_nets:
                    g = np.maximum(h @ net[0][0] + net[0][1], 0.0)
                    g = np.maximum(g @ net[1][0] + net[1][1], 0.0)
                    m += float((g @ net[2][0] + net[2][1])[0])
                m /= len(self.mlp2_nets)
            else:
                h = np.clip((h - self.mlp_mean) / self.mlp_scale, -8.0, 8.0)
                ls = self.mlp_layers
                h = np.maximum(h @ ls[0][0] + ls[0][1], 0.0)
                h = np.maximum(h @ ls[1][0] + ls[1][1], 0.0)
                m = float((h @ ls[2][0] + ls[2][1])[0])
            p = (1.0 - self.mlp_mix) * p + self.mlp_mix * m
        if self.pred_blend != 1.0:
            p = self.pred_blend * p + (1.0 - self.pred_blend) * self.fair_sig
        self.pred_sig = p

    def _fq(self, now: int, bb: int, ba: int, fshift: float, pull_bid: bool, pull_ask: bool):
        """Fair-centric prices: quote around F_hat, cross with an IOC when F_hat - k is beyond the far touch."""
        ref2 = self.fair_prev_mid2
        mid = ref2 / 2.0 if ref2 > 0 else (bb + ba) / 2.0
        if self.fq_level:
            sig = self.fair_ls / self.fair_w - mid if self.fair_w > 0.0 else 0.0
        elif self.pred_on and self.pred_mode != 1:
            sig = self.pred_scale * self.pred_sig
        else:
            sig = self.fair_sig
        f = mid + self.fq_scale * sig + fshift
        self.qs_f = f
        k = self.fq_k
        if self.fq_k_sp2 >= 0.0 and ba - bb >= 2:
            k = self.fq_k_sp2
        if self.fq_k_sp3 >= 0.0 and ba - bb >= 3:
            k = self.fq_k_sp3
        kb = k
        ka = k
        if self.hk_gap > 0:
            # a touch with a hole right behind it: a sweep that takes our level runs far (markout -1.5/lot)
            if self.hole[0] >= self.hk_gap and self.hk_k > kb:
                kb = self.hk_k
            if self.hole[1] >= self.hk_gap and self.hk_k > ka:
                ka = self.hk_k
        dbid = floor(f - kb)
        dask = -floor(-(f + ka))
        tk = self.fq_take_k
        if tk > 0.0:
            ft = f
            if self.pred_on and self.pred_mode:
                ft = mid + (self.pred_scale * self.pred_sig if self.pred_mode == 1 else self.fq_scale * self.fair_sig) + fshift
            take_buy = floor(ft - tk) >= ba
            take_sell = -floor(-(ft + tk)) <= bb
            if self.tr_rest:
                for px in list(self.rungs[0]):
                    if floor(ft - self.tr_keep_k) < px:
                        self._cancel_rung(0, px)
                for px in list(self.rungs[1]):
                    if -floor(-(ft + self.tr_keep_k)) > px:
                        self._cancel_rung(1, px)
        else:
            take_buy = dbid >= ba
            take_sell = dask <= bb
        ins = self.fq_inside
        cap_b = bb + ins if bb + ins < ba else ba - 1
        cap_a = ba - ins if ba - ins > bb else bb + 1
        bid = dbid if dbid < cap_b else cap_b
        ask = dask if dask > cap_a else cap_a
        tol = self.fq_tol
        mb = self.fq_max_behind
        if tol and not take_buy and not take_sell:
            # Keep a resting quote that is close enough and still valid (queue priority), unless we are crossing.
            if self.active[0]:
                p = self.quote_px[0]
                if -tol <= p - bid <= tol and p <= cap_b and bb - p <= mb:
                    bid = p
            if self.active[1]:
                p = self.quote_px[1]
                if -tol <= p - ask <= tol and p >= cap_a and p - ba <= mb:
                    ask = p
            if bid >= ask:  # never let kept prices cross each other: fall back to the desired (strictly ordered) ones
                bid = dbid if dbid < cap_b else cap_b
                ask = dask if dask > cap_a else cap_a
        if bb - bid > mb:
            pull_bid = True
        if ask - ba > mb:
            pull_ask = True
        take = self.fq_take
        if take > 0:
            if take_buy:
                pull_bid = True  # its room goes to the IOC once CANCELED arrives
                if now >= self.fq_next_take[0]:
                    if self.tr_rest:
                        self._clear_side(0, 1)  # an older residual gives its room to this take
                    cred = self._cxl_credit(0) if self.zd_take else 0
                    q = min(take, self.max_pos - self.inv - self.out[0] + cred)
                    if q > 0 and self._take(0, 1, _BUY, ba + self.fq_slip, q, cred) is not None:
                        self.fq_next_take[0] = now + self.fq_cool_us
                        self.n_takes += 1
            elif take_sell:
                pull_ask = True
                if now >= self.fq_next_take[1]:
                    if self.tr_rest:
                        self._clear_side(1, -1)
                    cred = self._cxl_credit(1) if self.zd_take else 0
                    q = min(take, self.max_pos + self.inv - self.out[1] + cred)
                    if q > 0 and self._take(1, -1, _SELL, bb - self.fq_slip, q, cred) is not None:
                        self.fq_next_take[1] = now + self.fq_cool_us
                        self.n_takes += 1
        return bid, ask, pull_bid, pull_ask

    def _cxl_credit(self, k: int) -> int:
        """Resting quantity on side k that is being cancelled now (active quotes, pulled by the caller) or already."""
        c = 0
        mina = self.zd_min_ahead
        for o in self.orders.values():
            if o.k == k and not o.ioc and (o.active or o.cancel_sent):
                if mina and self._ahead(o) < mina:
                    continue
                c += o.remaining
        return c

    def _ahead(self, o: _Order) -> int:
        """Market quantity queued ahead of our order o in the last snapshot (large if it is not listed)."""
        views = self.last_views[o.k]
        if views is None:
            return 1 << 30
        px = o.price
        oid = o.order_id
        a = 0
        sg = o.sign
        for v in views:
            if (v.price - px) * sg > 0:
                continue
            if v.price != px:
                break
            if v.order_id == oid:
                return a
            if v.owner == 'market':
                a += v.quantity
        return 1 << 30

    def _take(self, k: int, sign: int, side: Side, px: int, q: int, cred: int) -> str | None:
        """The taker's order: an IOC, or with tr_rest a LIMIT whose residual rests as a rung at the swept price."""
        if self.tr_rest and px not in self.rungs[k]:
            oid = self._send(k, sign, side, px, q, False, 4, cred)
            if oid is not None:
                self.rungs[k][px] = oid
            return oid
        return self._send(k, sign, side, px, q, True, k, cred)

    def _cancel_rung(self, k: int, px: int) -> None:
        """Cancel one resting take residual (it stays in OUT until CANCELED arrives)."""
        rungs = self.rungs[k]
        order = self.orders[rungs[px]]
        order.rung = False
        del rungs[px]
        self.rung_qty[k] -= order.remaining
        if not order.cancel_sent:
            order.cancel_sent = True
            self.cancel_order(order.order_id)
            self.n_cancels += 1

    def _cxl_only(self, k: int) -> int:
        """Resting quantity on side k whose cancel is already in flight."""
        c = 0
        only_up = self.zd_quote == 2
        for o in self.orders.values():
            if o.k == k and o.cancel_sent and not o.ioc and (o.zq or not only_up):
                c += o.remaining
        return c

    def _guard(self) -> None:
        """|I| is over the cap after a race fill: IOC the excess (minus reducing IOCs in flight) back at once."""
        inv = self.inv
        if inv > self.max_pos:
            k, sign, side = 1, -1, _SELL
            px = self.last_bb - self.zd_guard_slip
            ex = inv - self.max_pos
        elif inv < -self.max_pos:
            k, sign, side = 0, 1, _BUY
            px = self.last_ba + self.zd_guard_slip
            ex = -inv - self.max_pos
        else:
            return
        if self.zd_guard_pull:
            self._pull(1 - k)
            self._pull(3 - k)
        for o in self.orders.values():
            if o.k == k and o.ioc:
                ex -= o.remaining
        if ex > 0 and px > 0 and self._send(k, sign, side, px, ex, True, k) is not None:
            self.n_guard += 1

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
                    if behind > 0:
                        order.zq = True
                    self.cancel_order(order_id)
                    self.n_cancels += 1
            act.clear()
            self.act_qty[j] = 0
            self.last_action_us[j] = now
            if park:
                while len(self.parked[k]) > self.park_max:
                    self._unpark(self.parked[k][0])

        have = self.act_qty[j]
        want = size - have
        if want <= 0 or px <= 0:
            return
        if have > 0 and (want < self.topup_min or now - self.last_action_us[j] < self.min_requote_us):
            return
        cred = self._cxl_only(k) if self.zd_quote else 0
        free = self.max_pos - sign * self.inv - self.out[k] + cred
        if want > free:
            if self.out[k] - cred > self.act_qty[k] + self.act_qty[k + 2] + self.parked_qty[k] + self.rung_qty[k]:
                return  # in-flight cancels/IOCs hold the room; wait for them rather than fragment the quote
            want = free
            if want <= 0:
                return
        if self._send(k, sign, side, px, want, False, j, cred) is not None:
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

    def _send(self, k: int, sign: int, side: Side, px: int, qty: int, ioc: bool, j: int, credit: int = 0) -> str | None:
        """The only place orders are created: refuse anything that could break the invariant."""
        if qty <= 0 or px <= 0 or sign * self.inv + self.out[k] - credit + qty > self.max_pos:
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
            if self.of_us > 0 and not order.ioc:
                self.of_t[order.k] = self.of_now
            self._shrink(order, qty)
            if order.remaining <= 0:
                self._forget(order)
            inv = self.inv
            if inv + self.out[0] > self.max_pos or self.out[1] - inv > self.max_pos:
                self.invariant_breaks += 1
            if abs(inv) > self.max_abs_inv:
                self.max_abs_inv = abs(inv)
            if (self.zd_take or self.zd_quote) and (inv > self.max_pos or inv < -self.max_pos):
                self._guard()
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
