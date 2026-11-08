"""mm_r3_mlp2 (home, R3 candidate): mm_r3_mlp with the g3 passive thresholds (fq_k_sp2 2.0, fq_k_sp3 1.5, rt_ms 130, gate_thr 11) and
pred_blend 0.7: round3 540,341 / mirror 560,237 / round2 427,786; holdout 138,223 / 137,230 / 108,183 (mm_r3_mlp 135,145 /
130,359 / 107,581): better on the honest holdout, -9k on the full round3.

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
# exp_r3_mlpf: 3-seed ReLU ensembles (outputs averaged); _75 = first 75 % of round3 + mirror, _100 = all of it.
MLP_ENS_75 = [
    ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233382, -0.0, -0.0, 5.906055, 5.906348, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057909], [1.019957, 0.879338, 0.830501, 0.723114, 0.628298, 2.28085, 12.9744, 0.673953, 10.69429, 0.757186, 0.651131, 0.545643, 1.651632, 1.651633, 0.999999, 4.399308, 1.05635, 0.427403, 127.451202, 323.701813, 938.297913, 1.343849], [
        ([[0.076256, 0.095056, -0.257383, -0.136574, -0.047013, -0.122071, -0.331709, 0.088837, 0.094979, -0.066308, 0.006336, -0.186056, -0.050357, -0.205137, 0.197952, -0.159716, 0.045897, -0.460958, 0.10192, 0.185652, -0.630492, 0.15212, -0.155949, -0.023345, 0.736884, -0.239635, 0.202161, 0.099693, 0.526011, -0.217592, 0.068837, -0.01199], [-0.003814, 0.178836, -0.264097, -0.061809, -0.074539, 0.171046, 0.181152, -0.346835, 0.070003, 0.035506, 0.13549, 0.096317, -0.046456, 0.245902, -0.077171, -0.171964, 0.142913, -0.099869, -0.003217, 0.362221, -0.29089, -0.112574, 0.236327, 0.209315, -0.467235, -0.080843, -0.032084, 0.065341, 0.178757, 0.15274, -0.04988, 0.056033], [-0.387299, -0.172328, -0.002471, 0.062198, 0.172414, -0.14043, 0.28325, -0.054015, 0.087772, 0.215443, 0.019808, 0.253126, 0.138631, 0.051233, -0.045147, 0.350837, 0.106646, 0.422998, -0.199136, 0.202953, -0.143813, -0.062095, 0.3046, -0.093035, -0.401388, 0.314425, -0.038284, -0.151192, 0.246006, 0.060156, 0.229164, 0.220322], [0.108564, 0.158467, 0.163198, 0.243027, 0.276555, 0.152135, 0.004477, 0.005634, -0.255629, 0.3523, 0.123357, -0.073799, 0.034046, 0.05746, -0.103358, 0.130721, 0.10735, 0.533869, 0.044227, -0.181114, 0.207617, 0.161467, -0.077084, 0.132332, 0.107776, 0.019011, 0.261356, 0.187692, -0.249002, -0.148871, -0.357384, -0.041931], [-0.009506, 0.023822, 0.209915, -0.11993, 0.129754, 0.110319, -0.130037, -9.5e-05, 0.036067, -0.153933, -0.351131, -0.081129, 0.46732, -0.14769, 0.063589, -0.181259, 0.051913, 0.389125, 0.633091, 0.051884, -0.180419, 0.109649, -0.129743, -0.170289, 0.573553, -0.030192, -0.07262, 0.253373, -0.499212, 0.20987, 0.118721, -0.259178], [-0.067993, -0.131449, 0.049375, 0.000217, 0.35782, -0.171594, 0.025428, -0.021121, 0.090519, -0.107135, 0.11142, 0.090277, 0.227124, -0.015814, 0.098175, -0.141499, 0.058065, 0.277217, 0.133897, 0.168732, 0.307898, -0.044524, 0.039521, 0.030636, -0.490246, 0.091048, -0.12028, 0.131542, 0.112007, 0.069406, 0.292868, -0.321842], [0.028305, -0.53031, -0.190032, -0.002158, 0.3523, 0.876362, -0.780823, -0.292942, 0.131236, -0.446235, -0.709431, 0.829671, -0.050421, 0.526926, -0.356027, 0.472441, -0.013346, 0.341398, -0.328855, 0.181067, 0.279786, -0.016846, 0.179288, 0.212822, -0.077578, 0.044785, 0.530679, 0.485238, 0.295026, 0.701926, 0.357568, 0.407737], [-0.063926, -0.008476, 0.021339, 0.065951, 0.058729, -0.021255, 0.187223, -0.134368, 0.106252, 0.173935, -0.08692, 0.113864, -0.251515, -0.091794, 0.069953, 0.078711, 0.12634, 0.001678, -0.062552, -0.012659, -0.026454, 0.0208, -0.063729, -0.079775, 0.045075, 0.201677, 0.173453, -0.133036, -0.003778, 0.125369, 0.060614, -0.189834], [-0.185826, -0.503433, -0.338888, -0.232691, -0.123976, -0.291772, 0.088953, -0.708127, -0.557415, 0.257482, -0.649072, -0.039675, -0.062728, -0.023351, -0.350529, 0.57868, -0.785039, 0.683235, -0.578609, -0.089939, -0.013115, -0.24914, 0.050933, 0.199006, -0.43241, -0.195075, -0.253744, 0.244196, -0.184096, -0.029924, 0.27331, 0.465826], [0.366871, 0.075939, -0.198002, 0.398182, -0.710495, 0.268434, 0.23243, -0.808721, 0.50078, -0.148366, 0.076447, 0.187884, 0.207591, -0.465115, 0.301484, 0.177234, -0.80282, -0.176051, -0.187946, -0.805666, -0.461866, 0.376144, -0.223681, 0.142584, 0.087085, 0.201094, -0.206618, 0.006418, -0.127083, 0.433566, 0.150837, -0.133107], [0.173383, -0.057487, 0.039453, 0.09931, -0.133021, 0.191501, -0.12049, 0.090186, 0.023407, -0.003404, -0.199659, -0.00272, -0.01473, 0.105855, 0.02289, -0.139732, 0.063133, 0.03323, 0.308092, -0.18509, 0.037082, 0.170858, 0.231764, 0.056308, -0.035149, 0.138315, -0.36704, 0.136897, 0.15627, -0.035663, -0.185969, 0.070926], [-0.45304, -0.114551, -0.518753, -0.185216, -0.443337, -0.154296, 0.049482, -0.29099, -0.56959, 0.425879, -0.075384, -0.067122, 0.087854, -0.464991, -0.132492, 0.153374, -0.057456, 0.150131, -0.05443, 0.277794, -0.021784, -0.177282, -0.223941, -0.045058, 0.084185, -0.278151, 0.072123, -0.212223, -0.075904, -0.154103, 0.366831, 0.459394], [-0.058456, -0.13899, 0.482147, -0.159727, -0.02395, -0.267484, -0.007049, -0.186884, 0.014452, 0.175576, 0.194937, -0.251995, -0.045606, -0.038256, -0.009849, 0.426605, 0.194614, -0.052614, 0.105374, -0.299497, 0.047046, -0.014983, -0.146226, 0.280854, -0.102504, -0.125047, -0.193959, -0.256733, -0.11285, -0.06721, -0.352201, -0.03785], [-0.310106, -0.076824, -0.258406, -0.421955, 0.480027, 0.075175, -0.098995, 0.000987, 0.247318, 0.094727, -0.034331, -0.525945, -0.21732, 0.129436, 0.058146, 0.068171, 0.252316, 0.012129, 0.128523, -0.023534, 0.020724, -0.101162, -0.221943, -0.081063, -0.024609, 0.172095, -0.146708, -0.065048, 0.004583, -0.048383, 0.047624, -0.46479], [-0.114234, -0.212166, -0.307842, -0.762015, 0.002309, -0.293566, 0.911371, -0.349565, -0.025115, -0.121425, 0.042928, -0.25514, -0.032567, -0.01251, -0.694718, -0.56829, 0.158182, -0.268924, -0.381514, 0.204901, -0.220519, 0.311288, -0.339978, 0.279422, -0.121435, -0.155265, -0.407827, -0.30184, 0.192235, 0.277599, -0.166883, 0.132843], [-0.260782, 0.261064, 0.242908, 0.470463, 0.014655, -0.26398, -0.308356, -0.051278, 0.026136, 0.225281, -0.062372, 0.000471, 0.304697, -0.134008, 0.198167, 0.418037, -0.092743, -0.127707, -0.155509, 0.113251, 0.218652, 0.240769, 0.158968, -0.360664, 0.056107, 0.069742, -0.064758, -0.159051, -0.271316, -0.387494, 0.235471, -0.053123], [0.133007, 0.145027, -0.07949, 0.456253, -0.125601, -0.163765, -0.141651, 0.467812, -0.268872, 0.347851, -0.12505, 0.096041, 0.103965, -0.883469, 0.306878, 0.356934, -0.077121, -0.008509, 0.66341, -0.412143, -0.143135, -0.745072, -0.219867, -0.22444, 0.043359, -0.264765, 0.677752, -0.345313, 0.056628, -0.054725, 0.481806, -0.048019], [0.364672, 0.124113, 0.71747, -0.147183, -0.72338, -0.683308, -0.521849, -0.214741, 0.252406, -0.129597, -0.023056, -0.369688, -0.239059, 0.157767, 0.13684, 0.032038, 0.574654, -0.116028, -0.027078, 0.096911, 0.072673, 0.263436, 0.278922, 0.172495, 0.305816, -0.102193, 0.383503, 0.217703, 0.564107, -0.373832, -0.202163, -0.01183], [-0.020694, 0.126161, -0.033526, -0.020829, -0.205702, 0.051392, -0.036615, 0.01809, 0.031687, -0.088786, -0.055537, -0.127238, -0.00194, -0.034953, 0.038887, -0.067312, 0.045209, 0.083546, -0.048202, -0.070226, 0.017595, 0.005255, -0.106439, -0.001821, -0.018593, 0.069186, -0.010281, 0.17637, 0.016765, 0.040456, 0.036864, 0.114714], [-0.155053, -1.116513, 0.140091, -0.003388, -0.037801, -0.159542, -0.060692, 0.075007, 0.008575, -0.011999, -0.056697, 0.244703, 0.046469, -0.039604, 0.203679, 0.072987, 0.081744, 0.027762, -0.178149, 0.024829, -0.094311, -0.231873, 0.373926, -0.067195, 0.009422, 0.34861, -0.476612, 0.141995, -0.021383, 0.347549, 0.194836, 0.161123], [0.219718, -0.072521, 0.279643, -0.029583, -0.234738, -0.255619, 0.175676, -0.076867, -0.069908, 0.185197, 0.326484, 0.106115, -0.194362, 0.068368, 0.120085, -0.045952, 0.17818, -0.202977, -0.228095, 0.054989, 0.066117, -0.16244, 0.144222, 0.094387, -0.035317, 0.495162, -0.142426, -0.125857, 0.012933, -0.082131, -0.020681, 0.126816], [0.126954, -0.17177, 0.034013, -0.291674, 0.082846, -0.00533, -0.042848, -0.018406, 0.141814, -0.112816, 0.621714, 0.216513, -0.006378, -0.162688, -0.55429, 0.247889, -0.292465, -0.008193, 0.017565, -0.155171, 0.052875, -0.108185, -0.290224, -0.640044, -0.012323, -0.077949, 0.112203, 0.476245, -0.001784, -0.125446, -0.115175, 0.273241]], [-0.365384, -0.097393, -0.233975, -0.054104, -0.423877, -0.089224, -0.029717, -0.283379, -0.610956, -0.751637, 0.286126, -0.602071, 0.177335, 0.054576, 0.03176, 0.265322, 0.275297, -0.118646, 0.126406, -0.245665, -0.188046, -0.286467, 0.207251, -0.217441, -0.065376, -0.298394, 0.043653, 0.134775, -0.070343, 0.147536, -0.078361, -0.00874]),
        ([[0.315466, -0.293441, -0.181393, -0.149397, 0.361132, -0.401533, 0.223075, -0.195688, -0.119023, -0.016922, 0.049345, 0.188661, 0.222089, -0.060069, -0.4721, -0.054931], [-0.187053, 0.149865, -0.682088, -0.220401, 0.002627, -0.318626, 0.038279, -0.744824, 0.069387, -0.06753, -0.029482, 0.00744, 0.006904, 0.094486, 0.294376, 0.102868], [0.178634, 0.125194, 0.307575, 0.202237, 0.097786, 0.225469, 0.272938, 0.203297, 0.341226, 0.279105, 0.490617, 0.196382, -0.202281, 0.313633, -0.135912, 0.349685], [-0.712156, -0.441201, -0.100033, 0.048928, -0.330842, 0.239626, -0.245236, 0.058855, 0.067034, 0.012955, -0.067922, 0.241265, -0.187824, -0.145817, 0.253296, 0.265517], [-0.346503, 0.147177, -0.237069, 0.029903, -0.223918, -0.211764, -0.483804, -0.132308, 0.069853, -0.220352, 0.134818, -0.960497, -0.959274, 0.292406, -0.090232, 0.161351], [0.039518, 0.189488, 0.199583, 0.434823, -0.047509, 0.360918, 0.274744, -0.08659, 0.045525, 0.136798, 0.193592, 0.310636, 0.098039, -1.5008, -0.279427, 0.276469], [0.024996, 0.024899, 0.165326, 0.091014, 0.267119, 0.159337, 0.381951, 0.294113, 0.402951, -0.086985, 0.113155, 0.474511, -0.324636, -0.1267, -0.008756, -0.734186], [-0.176908, -0.338697, -0.269699, -0.122728, -0.125873, 0.315494, 0.139619, -0.086679, 0.142283, 0.146859, 0.123966, -0.026597, -0.458459, 0.26693, -0.11378, 0.094439], [0.240981, -0.078057, 0.059751, -0.017603, -0.068914, -0.102458, 0.663734, 0.178397, 0.223423, 0.338852, 0.460762, 0.014639, 0.134922, -0.255053, 0.43171, -0.058095], [-0.04816, -0.196112, 0.137424, 0.151046, 0.198361, -0.113074, -0.139843, 0.146935, -0.48058, 0.416188, 0.028582, -0.800226, 0.645554, 0.229794, -0.089728, 0.277008], [0.072085, 0.087571, -0.059693, -0.075313, -0.169218, -0.412171, -0.269632, -0.164407, -0.199103, 0.22618, 0.040981, -0.211608, 0.051173, -0.092752, -0.118974, -0.527342], [0.128357, -0.468384, -0.270886, -0.080559, -0.126928, 0.081863, -0.129789, -0.055726, -0.119179, 0.319778, 0.018558, -0.449406, 0.370465, -0.500052, -0.31361, -0.241532], [-0.130518, 0.078225, 0.090006, 0.005053, 0.131338, 0.165972, 0.052188, 0.074825, 0.134644, -0.575445, 0.039532, -0.195432, 0.046676, -0.182655, 0.216416, 0.28033], [0.033135, 0.378602, -0.255592, -0.009952, 0.123687, -0.261466, -0.210796, 0.154029, 0.126227, 0.175981, 0.382447, -0.004583, 0.234669, -0.955446, -0.412229, -0.527373], [-0.271604, 0.027439, 0.189541, 0.177602, -0.294894, 0.121117, 0.001626, 0.146727, -0.15739, -0.588154, 0.101998, 0.154591, 0.276962, 0.394131, 0.011773, 0.269996], [0.38071, -0.078847, -0.222234, 0.091933, 0.081342, -0.259401, 0.053272, 0.001814, -0.620461, -0.374856, 0.189485, 0.035961, -0.296844, 0.099756, -0.145003, 0.263938], [0.278222, 0.069093, -0.351902, -0.017026, -0.042579, 0.083201, 0.188107, 0.001626, -0.296504, -0.368021, 0.148249, -0.547134, -0.047327, -0.183963, -0.086464, -0.874994], [0.017429, -0.345179, 0.363949, 0.069358, 0.170628, 0.058633, 0.014412, 0.311786, 0.223258, 0.662686, -0.40426, 0.610973, 0.305925, 0.107134, -0.199974, 0.015328], [0.110657, -0.0501, -0.690835, -0.01921, -0.324375, 0.100772, -0.418672, -0.386259, -0.023884, -0.174042, 0.208156, -0.538472, 0.125879, 0.02647, 0.047903, -0.427893], [0.059528, -0.446516, 0.070459, 0.156002, 0.289369, 0.040696, 0.096663, 0.077459, -0.068557, -0.027093, -0.681903, 0.54962, 0.084098, 0.359125, -0.224661, -0.548318], [0.127724, -0.055043, 0.41864, -0.163687, 0.188952, 0.20894, -0.023026, -0.05036, 0.23878, 0.220216, -0.054159, -0.025239, 0.001164, -0.611931, -0.147304, -0.836496], [0.193466, 0.128115, -0.156233, -0.088719, 0.116417, -0.622392, -0.314088, -0.267106, 0.034855, 0.165041, 0.021416, 0.095589, 0.203351, -0.362333, -0.367086, 0.237846], [-0.01766, -0.081718, 0.116429, 0.049991, 0.217196, 0.174532, 0.028363, 0.161584, -0.168535, 0.086404, 0.039935, -0.093287, -0.618418, -0.250792, -0.013457, -0.114211], [-0.108958, 0.111828, 0.010411, -0.140731, -0.24308, -0.381053, -0.130768, -0.111391, 0.166321, 0.322293, 0.127969, 0.058459, -0.104277, -0.245639, 0.214109, -0.341441], [0.041157, 0.009232, 0.213923, -0.25919, 0.314344, 0.228419, 0.034706, 0.549224, 0.581894, -0.161856, 0.018362, 0.157, 0.033501, 0.169412, 0.10348, -0.106372], [-0.011316, 0.119968, -0.264948, -0.262987, -0.275451, 0.019832, -0.586978, -0.282328, -0.338419, -0.055016, -0.007576, -0.658228, -0.302153, -0.040144, 0.036684, 0.124917], [-0.013638, -0.050098, 0.188871, 0.356642, -0.222045, -0.762154, 0.070381, -0.36496, 0.11225, 0.173249, 0.189874, -0.152273, 0.110666, -0.340318, -0.117019, -0.005848], [-0.048426, 0.129227, -0.007635, -0.124321, -0.107116, 0.023719, -0.02769, -0.128862, -0.030932, -0.499692, -0.117367, -0.264866, 0.019466, -0.236439, -0.161111, -0.060169], [0.103284, -0.241821, 0.260937, 0.255666, 0.230204, 0.322519, 0.395358, 0.139227, 0.460606, -0.02037, -0.021989, 0.185656, -0.143111, 0.043433, 0.114556, 0.183706], [0.288102, -0.683951, 0.009052, 0.103974, 0.213305, -0.324456, -0.344513, 0.198979, -0.092721, 0.095412, 0.063319, -0.325848, -0.051928, 0.058031, 0.023469, -0.181239], [-0.047267, 0.542549, 0.215878, -0.050414, -0.039606, 0.071336, 0.317833, -0.024793, -0.341091, -0.235816, 0.275246, -0.532568, -0.117996, 0.227668, 0.028281, 0.088882], [0.52398, -0.184812, -0.004669, 0.107128, 0.03956, -0.246002, 0.141017, 0.124732, -0.287232, -0.121502, 0.067724, -0.031233, 0.092532, 0.180303, -0.155676, -0.413514]], [-0.094101, 0.181249, 0.038877, 0.391917, -0.140709, -0.235028, 0.226778, 0.203167, -0.105966, -0.343398, 0.189366, 0.164868, -0.408924, -0.247565, 0.536585, -0.320329]),
        ([[0.31582], [0.392017], [0.531229], [0.371966], [0.410373], [0.690949], [-0.339193], [0.388448], [-0.617814], [-0.848938], [-0.251756], [-0.690837], [0.774422], [0.937782], [0.386555], [1.11243]], [-0.32521]),
    ]),
    ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233382, -0.0, -0.0, 5.906055, 5.906348, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057909], [1.019957, 0.879338, 0.830501, 0.723114, 0.628298, 2.28085, 12.9744, 0.673953, 10.69429, 0.757186, 0.651131, 0.545643, 1.651632, 1.651633, 0.999999, 4.399308, 1.05635, 0.427403, 127.451202, 323.701813, 938.297913, 1.343849], [
        ([[0.445534, -0.152837, -0.081079, -0.032797, -0.300493, -0.028892, 0.017492, -0.549281, 0.110433, 0.421017, 0.048019, 0.259486, 0.005849, -0.22185, -0.027847, -0.026558, -0.571332, 0.336392, -0.671366, -0.120421, 0.03929, 0.285516, 0.049476, -0.229598, -0.108576, -0.036796, -0.104018, -0.20616, -0.313748, -0.576954, -0.25865, 0.182407], [0.123025, 0.063228, 0.095492, 0.050983, 0.273365, 0.361879, 0.119122, -0.196277, 0.000922, -0.316684, 0.173504, -0.020986, -0.257023, 0.141568, 0.27763, 0.170763, 0.073913, -0.208716, -0.06011, -0.036376, 0.201029, 0.054108, -0.049351, -0.106147, -0.094312, -0.229651, -0.042891, 0.082311, 0.163347, -0.3883, 0.276621, -0.065729], [0.136726, 0.177622, 0.001881, -0.029263, 0.155523, 0.385928, 0.270605, -0.115791, -0.211431, -0.336112, -0.256661, -0.163268, 0.191842, -0.142066, -0.117505, 0.179835, 0.546562, -0.237151, 0.303994, -0.004914, -0.12595, -0.068088, -0.204601, 0.258941, 0.039895, -0.015647, 0.063813, -0.042446, 0.382495, -0.202989, 0.075624, -0.021985], [-0.086611, 0.012759, -0.081415, -0.157567, -0.168013, -0.484371, 0.081335, 0.031303, 0.287367, -0.303217, 0.064207, 0.042793, -0.255918, 0.099723, 0.058662, -0.129155, 0.388831, 0.287358, 0.227575, 0.308375, 0.083573, -0.02633, 0.37457, 0.112829, -0.062556, -0.020304, -0.020602, 0.045792, 0.084096, 0.190593, 0.054454, -0.105882], [-0.46089, -0.145313, -0.070922, 0.180229, -0.166103, -0.134726, -0.056111, 0.182047, 0.065815, 0.358535, -0.017023, 0.055535, 0.210375, -0.222227, -0.452655, -0.100446, 0.031145, 0.272654, -0.269616, 0.191975, -0.08663, -0.018302, -0.066725, -0.069221, -0.103172, -0.267101, -0.319111, 0.157006, -0.212883, 0.399144, -0.049813, -0.012336], [0.191601, 0.298715, 0.099164, 0.03419, 0.105803, 0.235269, 0.132727, 0.209951, 0.061505, -0.143065, -0.212874, -0.246324, 0.097515, -0.038047, -0.166216, 0.086668, 0.214215, -0.098829, 0.045616, 0.176207, 0.04055, -0.07618, -0.060784, 0.057522, 0.145949, 0.030422, -0.073039, 0.363686, -0.099, 0.231847, 0.095419, -0.034323], [-0.137926, 0.035046, -0.300291, 0.864471, -0.641089, 0.527829, -0.305077, -0.186448, -0.217194, -0.426002, -0.329945, 0.523287, 0.256556, 0.042019, -0.827029, -0.208473, -0.123009, -0.007693, 0.228958, -0.257829, -0.287679, -0.092557, 0.462102, 0.11678, -0.12781, 0.211725, 0.073437, 1.10472, 0.320387, -0.281745, 0.054513, 0.842305], [0.031166, 0.200891, -0.095243, -0.093478, 0.03941, 0.003386, -0.030146, -0.121261, 0.093139, 0.092203, 0.078528, 0.037009, 0.087151, -0.030417, -0.183885, 0.172484, -0.025562, -0.077591, -0.030407, 0.007688, 0.200081, -0.141033, 0.069369, 0.291403, -0.245798, -0.012869, -0.003781, 0.053727, 0.221044, 0.038043, -0.271668, -0.194541], [0.238424, -0.300081, 0.634889, 0.301611, 0.113545, -0.038649, 0.285593, 0.411875, -0.821319, -0.122994, -0.332596, 0.267572, 0.318959, 0.325669, 0.705441, -0.036365, 0.311839, 0.113611, 0.892858, 0.060117, -0.460133, -0.220385, -0.041923, -0.064526, 0.735745, -0.170452, 1.713572, -0.367093, 0.525968, -0.190918, -0.296992, 0.152681], [0.221523, 0.463518, 0.001217, 0.411813, 0.391765, -0.740986, -0.093802, 0.336443, 0.21432, -0.130761, 0.319474, 0.171447, 0.176685, -0.930353, 0.551126, -0.330781, -0.208861, -0.006733, -0.037948, -0.202359, -0.408434, 0.014159, -0.080127, -0.297122, 0.139627, -0.002073, 0.046116, -0.169519, -0.019486, 0.097849, -0.262075, -0.177638], [-0.007687, -0.309872, 0.143088, 0.185619, 0.215554, -0.002706, -0.147134, -0.321654, -0.016013, 0.1461, 0.058825, 0.074386, 0.032419, 0.005054, 0.120455, 0.198796, 0.071595, -0.446773, -0.179594, 0.120721, -0.202776, -0.221149, -0.073986, -0.014862, 0.156471, -0.096606, -0.308296, -0.057496, -0.232765, 0.104403, 0.263743, 0.045063], [0.27539, 0.033815, 0.17209, 0.092528, -0.169095, 0.09351, 0.117602, -0.131274, -0.350353, -0.157215, -0.164638, 0.102873, -0.043342, -0.12686, -0.373263, -0.388719, 0.102407, 0.284299, 0.138376, 0.107026, 0.141934, 0.350048, -0.066158, 0.060377, 0.032903, -0.015685, 0.206541, 0.037759, 0.341297, -0.210646, -0.185354, -0.192329], [-0.240518, -0.45575, -0.19412, -0.137749, 0.410681, -0.044222, -0.243559, 0.276486, -0.191194, 0.040515, -0.083561, -0.096339, -0.209419, -0.085563, 0.020344, -0.243746, -0.028192, 0.176138, 0.163963, 0.110676, 0.813552, 0.101469, 0.03665, -0.033704, -0.114313, -0.462521, 0.142622, 0.096133, 0.039985, -0.092191, 0.142618, -0.141189], [-0.01032, -0.350419, -0.068015, -0.125453, -0.034801, -0.181655, -0.084693, -0.099278, 0.103476, 0.157171, -0.078404, -0.162298, 0.247011, -0.000819, -0.360651, 0.450102, -0.007815, 0.251741, 0.00156, -0.166401, 0.066743, 0.091381, -0.120032, -0.076367, -0.209174, -0.035063, -0.051637, 0.651883, 0.133389, -0.065514, 0.714229, -0.228324], [0.123114, 0.033875, 0.08277, -0.040501, 0.462849, 0.235806, 0.063486, -0.492376, -0.159317, -0.243936, -0.194606, 0.455698, -0.502822, -0.193048, -0.182403, 0.076318, -0.07327, -0.285034, 0.390476, -0.264572, 0.267152, -0.248199, 0.194743, -0.891296, 0.488324, -0.236626, 0.01687, -0.44364, 0.041879, -0.020552, -0.026068, -0.260177], [-0.312029, -0.197277, 0.193681, -0.062942, -0.187815, -0.113627, 0.070929, 0.407937, 0.112203, -0.095905, -0.030988, 0.244604, 0.278107, -0.134565, 0.133502, -0.383, 0.118025, 0.03004, 0.097008, 0.251184, 0.103288, 0.151964, 0.298188, -0.205624, -0.15845, -0.198189, -0.193856, 0.11769, -0.185324, -0.11987, -0.267101, -0.037854], [0.070423, 0.486132, -0.240427, -0.048821, -0.359, -0.619449, 0.054031, 0.081247, -0.193345, 0.510726, -0.541477, -0.075649, 0.215471, 0.321066, 0.21011, -0.181073, -0.210883, 0.326118, -9.7e-05, -0.319686, -0.30037, 0.39697, -0.695852, 0.838565, -0.893955, 0.205623, -0.329762, -0.064227, -0.491433, 0.018587, 0.018068, 0.303091], [0.033443, -0.218083, -0.087161, -0.114353, -0.112018, -0.004271, -0.003551, 0.241623, -0.038307, 0.25362, 0.279229, 0.250953, -0.039708, -0.010442, -0.644729, -0.966205, -0.574218, 0.233196, -0.468277, -0.882789, -0.435687, 0.115757, -0.328725, 0.496948, 0.283014, 0.18535, -0.181381, -0.186907, -0.1117, -0.447548, 0.227522, 0.88953], [0.055699, 0.046161, 0.036698, 0.031362, 0.048064, -0.0265, -0.009621, 0.045298, 0.078105, -0.026658, 0.247969, -0.080952, 0.185853, 0.043654, -0.117996, -0.016454, -0.030068, -0.045419, -0.050787, 0.036542, 0.020096, -0.103332, 0.076656, 0.040253, 0.06019, 0.082871, 0.000817, -0.097977, -0.050674, -0.080419, 0.105046, -0.190766], [0.084405, -0.09197, 0.97159, 0.017004, -0.114212, 0.051756, 0.54374, 0.011022, -0.03046, -0.047849, -0.582065, 0.159584, 0.224656, -0.167954, 0.123605, 0.073472, -0.0772, 0.000951, 0.44778, 0.267354, -0.079562, -0.687971, -0.424515, 0.040521, 0.218131, -0.040396, 0.376877, 0.110217, -0.548249, 0.059509, 0.21139, 0.064245], [-0.170418, 0.158193, -0.025992, -0.122082, 0.268719, 0.043755, 0.250451, 0.097888, -0.025557, -0.030545, -0.058815, 0.278946, 0.048574, -0.130366, 0.165007, -0.16169, -0.079436, 0.040707, -0.065842, 0.351447, 0.289693, -0.221518, 0.009765, -0.14881, -0.110244, 0.13704, 0.315864, -0.098716, -0.225575, -0.176615, 0.208352, -0.122118], [0.023229, 0.015461, -0.121047, 0.607604, -0.07058, -0.092379, 0.123355, -0.078218, 0.814608, -0.011324, -0.046783, -0.119985, -0.453795, 0.054961, 0.161561, -0.106935, -0.111449, 0.051786, 0.179576, 0.253852, -0.236384, 0.062838, -0.40998, -0.096444, -0.117609, -0.150148, 0.074535, -0.063539, -0.013849, 0.054162, -0.269024, -0.036529]], [-0.126404, -0.392567, 0.107366, 0.164799, 0.005566, -0.395223, 0.222138, -0.507895, 0.071141, -0.582736, -0.143736, -0.115434, -0.041626, 0.035416, -0.308534, -0.395963, -0.193671, 0.006778, 0.076955, -0.574377, 0.130247, 0.099183, 0.088373, -0.528332, -0.317276, -0.300754, -0.06808, 0.281877, -0.046952, -0.556685, 0.332498, 0.056861]),
        ([[0.259431, 0.081003, -0.029598, 0.230914, 0.063782, -0.065359, 0.265071, -0.246626, 0.285609, 0.15858, 0.068993, -0.048485, 0.311513, 0.278637, 0.262521, 0.076766], [-0.094786, -0.469669, 0.031284, -0.335826, 0.26039, 0.189959, 0.018148, 0.020856, -0.082565, 0.192793, -0.262104, 0.243619, 0.076452, -0.594259, 0.099537, -0.129064], [-0.015298, -0.047321, 0.323459, 0.200999, -0.594619, -0.121078, -0.632307, -0.390567, -0.122273, -0.080115, 0.083995, 0.073404, -0.374062, 0.082099, 0.162621, 0.010298], [-0.279485, -0.98616, 0.169358, 0.152647, -0.301012, 0.156133, -0.354611, -0.068228, 0.196948, 0.079644, 0.069707, 0.173583, 0.055477, -0.00885, 0.000134, -0.17532], [-0.380062, -0.173389, -0.025393, 0.021716, 0.381053, -0.543197, 0.000812, -0.261355, -0.381557, -0.166279, 0.316011, -0.326811, -0.166175, 0.049076, 0.30869, -0.116957], [-0.277832, 0.314714, 0.148651, -0.056596, 0.086468, 0.132845, 0.098552, -0.133258, -0.40676, -0.037154, 0.137064, -0.087788, -0.14417, -0.662031, 0.321411, -0.723521], [0.071538, 0.152155, 0.040907, 0.083918, -0.489047, 0.113845, -0.135029, 0.034225, -0.540249, -0.072851, -0.279109, 0.216083, -0.156061, -0.017897, 0.081318, 0.171753], [0.031152, -0.458293, 0.133558, 0.554196, -0.126959, -0.583949, 0.286941, -0.044455, 0.107655, 0.022402, 0.120737, -0.080742, 0.294679, -0.094635, 0.270196, 0.242852], [-0.465668, -1.055522, 0.058132, -0.119785, -0.000186, -0.114975, 0.022756, 0.192202, 0.050255, -0.02653, -0.318117, 0.20136, -0.229754, -0.025676, -0.188226, 0.103031], [0.490553, -0.119163, -0.279879, 0.063462, 0.082846, -0.741852, 0.334596, 0.232471, 0.218626, -0.502283, 0.080268, -0.07466, 0.185275, 0.094557, -0.320896, 0.246314], [-0.023974, 0.309003, -0.457308, -0.30208, -0.002154, 0.346557, 0.008925, 0.227584, 0.033575, 0.249927, -0.201362, -0.287634, 0.13831, 0.188967, -0.118321, 0.039681], [0.052585, 0.041901, -0.017618, 0.148792, 0.394916, -0.120583, 0.347871, -0.153374, -0.16223, -0.031, -0.612819, -0.219567, 0.165932, -0.016455, 0.007217, 0.163272], [0.317439, 0.027623, -0.082034, 0.195006, -0.184166, 0.089534, -0.106832, -0.155433, 0.354255, -0.136877, -0.326345, 0.01832, -0.271903, 0.067851, 0.239495, -0.006661], [-0.1335, 0.095046, 0.058502, -0.273071, 0.324559, 0.18536, 0.258983, 0.179284, 0.195575, 0.129417, -0.084637, 0.210759, 0.136782, 0.122791, -0.351915, -0.034194], [0.331803, 0.122051, -0.391688, -0.106917, -0.00299, -0.502936, 0.06457, 0.143768, 0.25816, -0.59241, 0.251683, -0.118235, 0.082502, -0.074751, -0.002117, 0.394251], [0.189355, -0.681516, 0.131602, -0.181509, -0.444727, 0.070797, 0.246267, -0.337341, -0.285384, -0.00449, 0.086255, 0.305551, -0.114778, 0.071021, 0.057186, 0.367531], [0.393823, 0.29062, 0.164557, 0.042133, 0.263145, 0.07147, 0.195199, 0.425662, 0.235816, 0.002942, 0.203743, -0.23423, 0.353729, 0.038626, 0.29351, -0.459514], [0.218399, -0.009453, 0.418807, 0.248221, 0.162984, 0.045644, 0.234143, 0.151512, -0.354479, 0.112263, 0.237798, 0.002748, 0.082523, 0.419092, 0.177605, -0.096587], [0.178611, -0.103339, 0.198462, -0.588009, -0.310345, 0.136215, -0.14275, 0.021506, 0.162404, -0.05415, -0.217514, 0.150936, 0.096198, 0.257442, -0.019153, 0.207696], [0.350329, -0.298207, -0.056052, -0.293052, 0.432215, 0.158919, -0.215183, 0.093274, 0.162857, 0.441962, -0.456157, 0.183252, -0.285987, -0.273575, -0.173709, 0.224824], [0.117128, 0.303659, 0.138093, -0.067883, -0.46252, 0.218792, -0.137678, -0.296461, 0.245519, -0.117615, -0.065127, 0.353681, -0.332855, 0.255896, 0.09532, -0.068406], [0.192128, -0.498593, 0.274652, -0.63375, -0.316646, -0.460653, -0.065254, 0.073254, 0.075628, 0.081624, 0.24405, 0.067243, 0.056733, -0.26571, -0.463236, -0.13761], [-0.104675, -0.275994, -0.214527, -0.191274, 0.093954, -0.292545, 0.16453, -0.071135, -0.099142, 0.413563, 0.240516, -0.283991, 0.32119, -0.028493, -0.42037, 0.009134], [-0.220042, -1.150711, -0.002453, -0.328242, 0.083541, -0.957938, -0.731011, 0.145363, 0.052428, -0.248046, -0.149875, 0.146079, -0.388705, -0.001765, -0.358496, -0.554501], [-0.226785, 0.265204, -0.003564, -0.518515, 0.064053, 0.078354, -0.096538, 0.191231, 0.510653, -0.460195, -0.04094, 0.178807, -0.144838, 0.044371, -0.113916, 0.115061], [0.020712, 0.25313, 0.069758, 0.256968, -0.045559, 0.053398, 0.300274, 0.077618, -0.07839, -0.476151, 0.182795, 0.049095, 0.157904, 0.219996, -0.318834, -0.105841], [0.102604, 0.275663, -0.037263, -0.197458, -0.138809, -0.409836, -0.4376, -0.018934, 0.083994, -0.133312, 0.121697, 0.255391, -0.198139, -0.585862, -0.483288, -0.024218], [-0.23661, -0.046329, -0.086125, 0.202777, -0.084224, -0.112593, -0.367474, 0.105396, -0.171345, -0.612306, 0.025518, 0.375263, -0.406745, -0.33495, -0.055766, -0.289457], [0.03715, 0.559024, 0.121387, 0.364903, 0.351885, 0.429707, 0.27135, -0.553632, -0.21915, 0.189268, 0.457516, -0.457652, -0.047095, -0.102403, 0.334046, 0.229536], [0.164938, 0.142421, -0.133105, -0.572972, 0.230216, 0.331935, 0.138038, 0.110919, 0.307402, 0.034491, -0.016839, 0.092893, -0.001715, 0.560415, 0.277731, -0.118377], [-0.247109, -0.112468, 0.144208, -0.384405, 0.19768, 0.266211, 0.137218, 0.077509, -0.20134, 0.093733, -0.036408, 0.018882, -0.251669, -0.150205, -0.285553, 0.206533], [-0.125542, -0.522947, 0.214411, 0.207176, 0.000585, 0.229955, -0.116187, 0.548578, -0.035343, -0.360074, -0.054469, -0.177567, 0.060825, 0.223032, -0.103939, 0.218401]], [-0.186065, -0.610742, 0.4558, 0.018079, -0.515436, -0.466551, 0.08303, 0.040094, -0.270013, -0.213767, 0.301963, -0.049863, 0.132597, 0.129003, 0.104046, -0.283578]),
        ([[0.652798], [1.282073], [0.398127], [0.915777], [-1.051385], [0.684454], [-0.397262], [-0.259645], [0.697369], [0.708945], [-0.336509], [-0.259282], [-0.381891], [-0.52804], [0.390768], [-0.554288]], [-0.023953]),
    ]),
    ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233382, -0.0, -0.0, 5.906055, 5.906348, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057909], [1.019957, 0.879338, 0.830501, 0.723114, 0.628298, 2.28085, 12.9744, 0.673953, 10.69429, 0.757186, 0.651131, 0.545643, 1.651632, 1.651633, 0.999999, 4.399308, 1.05635, 0.427403, 127.451202, 323.701813, 938.297913, 1.343849], [
        ([[-0.142669, 0.217257, -0.060058, 0.08921, 0.040708, 0.818366, 0.028648, -0.341732, 0.023584, 0.268708, 0.001831, -0.113434, 0.073367, -0.16996, -0.059564, -0.473679, -0.118501, -0.220972, -0.132148, 0.097386, -0.081579, -0.048202, -0.197693, 0.063331, -0.156471, -0.49224, 0.228429, -0.024113, 0.013335, 0.560537, 0.001884, -0.450937], [-0.07781, 0.199529, 0.072438, -0.130225, -0.378213, 0.087876, -0.251525, -0.272044, 0.158338, 0.077098, -0.085704, -0.083892, -0.033063, -0.052989, 0.191116, 0.363348, 0.065568, 0.135157, 0.126993, -0.204299, -0.070028, -0.182712, 0.199831, -0.279846, -0.245678, -0.124723, -0.142567, -0.011343, -0.080029, -0.214163, 0.137696, -0.067782], [0.147917, 0.097311, 0.180793, -0.022517, -0.368827, -0.122218, -0.011197, 0.07151, 0.094996, -0.231301, 0.144927, 0.23031, -0.085569, 0.251869, 0.260116, 0.395163, 0.143322, -0.119777, 0.091091, -0.057819, 0.07339, -0.121938, 0.417098, -0.100531, 0.197307, 0.210181, 0.0848, 0.123426, -0.069124, -0.431178, -0.238264, 0.298476], [0.20574, 0.016906, -0.12428, 0.277657, -0.153819, -0.200679, 0.008419, 0.506224, 0.187671, 0.004106, 0.152122, 0.150972, -0.200371, -0.231927, -0.286307, 0.010656, 0.151986, -0.092912, 0.133637, -0.139382, 0.102276, -0.035545, 0.138828, 0.167526, 0.050862, 0.535026, 0.047178, -0.113057, 0.169587, 0.05476, 0.119541, 0.086189], [-0.058515, 0.036449, 0.233247, -0.011252, 0.179202, -0.510265, 0.089751, 0.420447, -0.138338, 0.036212, -0.161152, -0.023742, 0.068954, -0.298356, -0.313359, -0.601699, -0.107509, 0.087397, 0.23279, 0.079882, -0.432303, 0.049663, -0.187008, 0.183016, 0.018047, -0.064898, -0.003961, 0.120722, -0.055953, 0.13404, -0.1587, 0.163898], [-0.165871, -0.174186, 0.048815, 0.229796, -0.01996, -0.256122, -0.249472, 0.286423, -0.199551, -0.418031, -0.113145, 0.302892, 0.24531, -0.018471, 0.179353, 0.16428, -0.026697, 0.074009, 0.104912, -0.112716, -0.048364, -0.150709, 0.19822, -0.246665, 0.14014, 0.027379, -0.037238, -0.021473, -0.015175, -0.179423, -0.070234, 0.044108], [0.138289, -0.190219, 0.43944, -0.384277, 0.328463, -0.363119, 0.153866, 0.05851, 0.412074, -0.022198, 0.085475, -0.638961, -0.360042, -0.3369, 0.063596, -0.588607, 0.464665, -0.167144, -0.200978, 0.137062, 0.218617, -0.237401, 0.915307, 0.375053, 1.043349, -0.432215, -0.071536, -0.194637, 0.414972, 0.830619, -0.296071, 0.969236], [-0.340766, 0.011378, 0.164544, 0.027376, 0.053347, 0.060449, -0.060817, 0.253237, -0.188296, 0.179527, 0.303168, 0.002986, 0.098725, -0.048056, -0.014999, 0.051349, 0.004318, -0.065067, 0.0201, -0.10602, 0.132742, -0.040512, -0.065222, 0.00506, -0.087285, -0.093433, -0.337409, 0.348035, -0.000516, -0.201809, 0.248769, -0.237375], [0.196524, -0.27526, 0.152537, -0.351138, -0.641412, -0.300057, -0.02359, 0.468947, -0.264656, -0.342211, -0.556049, -0.512451, -0.012128, 1.095233, -0.015416, 0.020771, 0.207563, -0.047501, -1.167052, -0.104894, 0.79636, -0.455362, 0.048838, -0.04067, 0.38252, 0.381963, 0.187488, -0.117278, -0.43272, -0.028653, -0.217032, -0.160075], [-0.47619, -0.90799, 0.021579, -0.107901, -0.397629, 0.333187, -0.041774, -0.222501, 0.043068, -0.005451, -0.487512, -0.102076, 0.012702, -0.087898, -0.626677, 0.024847, -0.498534, 0.21609, -0.08087, -0.316436, -0.087827, 0.099225, 0.09156, -0.484499, 0.054807, 0.244699, -0.041542, 0.502659, -0.138592, 0.29251, -0.552517, 0.136138], [0.14243, -0.004382, -0.108015, 0.031365, -0.059748, 0.062279, 0.184548, 0.077893, 0.055899, -0.34774, 0.039284, -0.142786, -0.121293, -0.092717, -0.129649, 0.259926, -0.199352, -0.272234, 0.083834, 0.088773, -0.202783, 0.271947, -0.199141, -0.01983, 0.015142, -0.077118, 0.148135, -0.270193, -0.149703, 0.058303, -0.109083, 0.041922], [-0.113049, 0.207015, 0.081726, -0.191643, -0.175189, 0.035788, 0.149989, 0.09111, -0.106468, -0.285887, -0.136632, -0.451994, -0.002898, 0.342393, 0.166486, -0.282688, -0.235442, 0.505597, -0.26335, -0.05296, 0.080446, -0.410957, 0.31102, 0.301771, 0.001515, -0.211171, 0.270237, 0.226777, 0.069831, -0.124213, -0.44552, 0.523884], [0.004931, -0.147651, -0.027483, -0.081541, -0.107673, 0.041108, 0.103514, -0.051556, 0.151291, -0.416412, -0.05515, -0.666028, 0.066576, 0.043392, -0.297731, -0.040004, -0.001147, -0.182264, 0.054365, -0.149319, 0.09319, -0.01612, 0.084952, 0.32413, -0.180764, 0.123119, 0.054507, -0.181789, 0.088084, -0.286393, 0.051343, -0.217341], [0.53589, 0.129654, 0.218997, 0.025643, -0.113381, 0.179242, 0.103273, 0.040147, 0.073616, -0.298097, -0.201578, 0.104538, 0.142356, -0.153532, -0.117033, -0.095007, 0.276647, -0.192922, 0.18365, -0.016895, 0.178518, -0.208118, 0.013466, 0.009559, -0.067695, -0.237651, -0.473657, -0.134673, -0.061731, 0.189418, 0.037709, 0.108336], [0.095281, 0.493209, -0.053337, 0.153527, -0.243133, 0.101339, 0.441359, -0.069929, 0.216198, -0.184516, -0.314566, 0.058203, -0.572952, -0.150076, 0.163551, 0.476458, -0.286936, 0.490121, -0.573942, -0.355484, 0.004852, -0.488545, -0.289601, -0.428012, 0.009344, -0.406846, -0.108688, 0.54165, 0.407525, -0.08347, 0.367796, 0.094758], [-0.001216, 0.015209, 0.138114, 0.285625, 0.02269, -0.157902, 0.519841, -0.393922, -0.447085, 0.076562, -0.152285, 0.155754, 0.216645, -0.435231, 0.142705, -0.295566, -0.367345, -0.058728, -0.436349, 0.002571, -0.004685, 0.05564, 0.254468, 0.347828, -0.091262, 0.353512, -0.302618, -0.338186, -0.009054, -0.136802, -0.13303, 0.345019], [-0.287451, -0.19819, 0.460512, -0.947028, 0.705878, 0.283173, 0.41116, -0.057373, 0.08135, 0.065143, -0.352441, 0.42614, 0.440426, 0.536434, -0.338138, -0.287014, -0.079889, -0.224041, 0.779156, -0.006069, -0.395984, 0.509778, 0.044266, -0.177531, -0.137513, -0.196422, 0.263167, -0.143167, -0.328768, 0.161037, -0.452227, -0.183171], [-0.245201, 0.295275, -0.337526, 0.179834, 0.073403, 0.227909, -0.134863, 0.287289, 0.026367, 0.092278, -0.3867, 0.110641, -0.190951, 0.078129, -0.837733, -0.082954, 0.246148, 0.760746, 0.20638, 0.450969, -0.425326, -0.439047, 0.294169, -0.737498, -0.118098, -0.155887, -0.02623, -0.361531, -0.29938, -0.054, 0.64157, 0.028604], [-0.01827, -0.143082, -0.066307, 0.307185, 0.101697, 0.080759, -0.033197, 0.113127, -0.055995, -0.039548, 0.018776, -0.050185, 0.045704, -0.008642, 0.05839, 0.015741, -0.126069, -0.044673, -0.013585, -0.04005, -0.009291, 0.033174, 0.071252, -0.064151, -0.006082, -0.0348, 0.025385, 0.067682, 0.083728, 0.008493, 0.019268, 0.004874], [0.393507, 0.115002, 0.357014, -0.270557, -0.062607, -0.001468, -0.198903, 0.093127, -0.028212, -0.214462, 0.004836, -0.071998, 0.044749, 0.551524, 0.053705, -0.004301, -0.055056, 0.142469, -0.153097, -0.349521, 0.337254, -0.053488, 0.131605, 0.015069, 1.389787, -0.11869, 0.070494, 0.005311, -0.505159, 0.137352, -0.023239, 0.111328], [-0.166475, 0.190088, 0.201901, 0.084644, -0.0425, -0.009993, -0.299073, -0.183593, 0.03401, 0.004842, -0.226069, -0.002992, 0.007018, 0.249207, 0.01658, 0.042353, -0.236675, 0.269673, -0.1625, -0.369911, 0.18914, 0.227358, -0.17359, 0.033901, -0.034842, 0.117723, -0.240678, -0.167024, -0.021643, -0.139854, 0.047361, -0.028378], [-0.231462, -0.080355, -0.181921, -0.044221, -0.12022, -0.134886, 0.089311, -0.067832, -0.726076, -0.003863, 0.030773, 0.30726, -0.768886, 0.006019, -0.000233, -0.024239, -0.399728, -0.379457, 0.063113, -0.208293, 0.117237, 0.105876, -0.076244, 0.074141, -0.172984, -0.169272, 0.054017, -0.031715, -0.447459, -0.109751, 0.201704, 0.005133]], [-0.025484, 0.251334, 0.104723, -0.142982, -0.233478, -0.327582, 0.079628, -0.118794, 0.13545, -0.496526, 0.1852, -0.106731, -0.090822, -0.118707, -0.176573, -0.120487, -0.095217, 0.051146, 0.129812, 0.046392, 0.323941, 0.088879, -0.728237, -0.496082, -0.088499, -0.122527, 0.295088, -0.287941, -0.094498, 0.243415, -0.513194, -0.227825]),
        ([[-0.086119, -0.086202, -0.099694, -0.026216, -0.007103, -0.239189, 0.171824, -0.312081, -0.6706, -0.190784, -0.22951, -0.208022, 0.05851, -0.011723, -0.097212, 0.17957], [-0.323074, 0.243197, 0.042596, 0.348297, -0.234098, -0.142118, -0.007007, 0.379815, -0.058472, 0.173403, 0.226683, -0.277472, 0.065353, -0.18339, 0.167582, -0.560054], [-0.198024, -0.147216, -0.259876, 0.19016, 0.175038, 0.10594, 0.194619, -0.134582, 0.113766, 0.047319, -0.03424, 0.09709, -0.358307, -1.263666, 0.004739, -0.128705], [-0.010489, -0.048695, 0.137201, -0.021532, 0.071228, -0.096497, -0.070319, -0.381038, -0.138409, 0.074056, -0.182016, -0.27308, -0.356567, -0.706902, -0.777485, 0.029981], [-0.086233, 0.275082, -0.029595, -0.12806, 0.367437, 0.166152, 0.142551, 0.235857, 0.10072, 0.288516, 0.138397, 0.231829, 0.216821, 0.122409, 0.165458, 0.072939], [0.156491, -0.150662, -0.152383, 0.034998, -0.445123, -0.017861, -0.084162, 0.133434, -0.068508, -0.183561, 0.412237, 0.305739, 0.33623, 0.196072, 0.245885, 0.173867], [-0.001328, -0.03948, -0.222463, -0.356146, -0.107567, -0.104463, 0.106694, -0.073635, -0.10774, -0.239122, -0.084664, -0.05922, -0.376197, -0.250099, -0.179474, -0.186972], [-0.010137, 0.255884, -0.050774, 0.232179, 0.39789, 0.101227, -0.074555, -0.063745, 0.120742, 0.266842, 0.184139, 0.126739, 0.112303, 0.458758, 0.119977, 0.329144], [0.086428, -0.27546, 0.141227, -0.322998, -0.159198, -0.249501, -0.018473, -0.664365, -0.453806, 0.048405, -0.316809, -0.012857, 0.07033, 0.124253, -0.156531, 0.197585], [-0.08771, 0.04177, 0.055039, 0.163926, 0.033939, 0.026493, -0.096845, -0.246073, -0.701516, -0.156594, 0.254673, 0.034192, -0.076337, -0.140613, 0.077947, -0.658588], [0.118707, -0.128601, 0.055966, 0.069749, -0.093154, -0.044142, -0.344137, 0.152612, -0.471926, -0.063472, -0.225786, -0.012228, 0.021475, 0.133981, -0.970738, -0.003436], [-0.364331, -0.240384, -0.033195, -0.43741, -0.011652, -0.204477, 0.172532, -0.019929, 0.164219, -0.268588, -0.269558, 0.105366, -0.203962, 0.079969, -0.432937, 0.53581], [-0.15162, 0.05062, -0.184157, 0.111662, 0.045768, 0.01532, -0.013269, 0.527472, 0.466344, -0.205832, 0.400606, -0.197055, -0.240772, -0.6577, -0.06636, -0.390888], [-0.581701, 0.07584, -0.081185, -0.082057, -0.055084, 0.039022, -0.101934, -0.332239, -0.251182, -0.479307, 0.01378, -0.258814, -0.352849, -0.57221, -0.116128, 0.019263], [0.211959, 0.366246, -0.319216, 0.272402, -0.015995, -0.211313, -0.187825, 0.00694, 0.153997, 0.46827, -0.011729, 0.205662, 0.340627, 0.148607, 0.306573, -0.968137], [0.288598, -0.192285, 0.397057, 0.24219, 0.122256, 0.311173, -0.310563, -0.022259, 0.085533, 0.333479, 0.63636, 0.314169, 0.329827, 0.40138, 0.545878, 0.103], [-0.187534, 0.144556, -0.038615, 0.032052, 0.042011, 0.331232, 0.050296, 0.031781, 0.434549, 0.134063, -0.293735, 0.226675, 0.064127, 0.163245, -0.180338, 0.229838], [0.269676, 0.205176, -0.07983, -0.317846, -0.373588, -0.327375, -0.133409, -0.087945, 0.23243, 0.076376, 0.085374, 0.060684, 0.220697, 0.415766, 0.377733, -0.193775], [0.043618, -0.486619, -0.215636, -0.486565, 0.057822, -0.051415, 0.118495, -0.293275, 0.031171, -0.434071, -0.086242, -0.422209, -0.127794, -0.269511, -0.082617, -0.478561], [-0.133342, 0.289359, -0.147177, -0.07234, 0.156348, 0.109386, 0.004996, -0.394995, -0.270391, -0.184993, -0.142621, -0.050838, -0.189809, 0.064242, 0.182038, 0.118726], [-0.13986, -0.131033, -0.003139, -0.151913, -0.097137, -0.291572, 0.383778, -0.220296, -0.267248, -0.189198, -0.22158, -0.166864, -0.376495, 0.106095, 0.109361, -0.491751], [-0.139632, 0.168619, 0.319394, 0.166291, 0.147553, -0.16287, 0.00223, -0.003638, -0.135514, 0.24138, 0.011697, 0.10218, 0.267141, -0.072178, -0.297257, 0.375684], [-0.097552, -0.697787, -0.496918, 0.152648, 0.380937, -0.189405, -0.528195, 0.020288, 0.222489, 0.228986, -0.044898, 0.01234, 0.188615, 0.015935, 0.217878, 0.191112], [-0.195504, -0.362839, 0.039033, 0.168372, -0.365918, 0.084309, 0.356509, 0.317026, 0.26243, -0.056861, 0.041489, -0.335712, -0.221029, -0.054519, 0.503498, 0.289479], [0.251979, -0.155364, 0.210302, 0.041181, 0.05225, -0.110257, -0.200686, 0.171315, -0.068146, 0.087689, 0.090149, -0.524096, -0.729739, -0.373392, -0.06175, -0.911391], [0.361523, 0.057449, -0.176103, 0.142404, 0.093063, 0.356705, 0.208872, 0.575184, 0.066562, 0.002201, 0.280971, 0.436724, 0.008751, 0.023279, 0.043823, -0.019733], [-0.185148, -0.119186, 0.142135, 0.396776, -0.312824, 0.08377, -0.198103, -0.044818, 0.027808, -0.062145, 0.311371, 0.134726, 0.038936, -0.140755, 0.039862, -0.189915], [0.299485, -0.222258, -0.381493, 0.269951, -0.18181, -0.592717, 0.140967, -0.317089, 0.160312, 0.055921, 0.08924, -0.07211, 0.205538, 0.014157, -0.068397, 0.182309], [0.099915, -0.032867, 0.164292, 0.091374, 0.062417, -0.081418, 0.134691, -0.015548, -0.417104, -0.419239, -0.17257, 0.026368, 0.287253, 0.313053, 0.052458, 0.110569], [-0.033434, -0.148267, 0.360807, 0.096242, -0.090166, 0.264554, 0.04759, 0.520709, 0.405421, 0.050818, 0.143147, 0.220448, 0.260386, 0.00178, 0.300036, -0.317592], [-0.047193, -0.682155, -0.116511, 0.199592, 0.326565, -0.120191, 0.611395, -0.259902, -1.120686, -0.459647, 0.382418, 0.084569, 0.463114, 0.181867, -0.654492, 0.034468], [-0.397992, -0.275709, 0.366214, -0.074449, 0.275014, -0.230772, -0.251541, 0.19148, -0.186539, -0.001089, 0.013014, -0.130819, 0.228488, 0.099641, 0.219448, -0.305791]], [-0.106957, 0.375073, 0.294825, -0.181401, -0.417762, 0.373345, 0.257844, -0.3388, 0.128276, 0.378739, -0.271588, 0.106319, -0.079732, -0.321548, -0.31641, -0.258769]),
        ([[-0.706572], [-0.364071], [-0.315937], [0.39173], [-0.570358], [-0.331269], [-0.220367], [0.909621], [0.463592], [0.373077], [0.344678], [-0.423594], [-0.617789], [-0.627983], [0.502521], [-1.406175]], [0.251914]),
    ]),
]
MLP_ENS_100 = [
    ([-0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 1.2325, -0.0, -0.0, 5.906742, 5.90694, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.000779, 4.050925], [1.010813, 0.870736, 0.822972, 0.71822, 0.624642, 2.270075, 11.568464, 0.66481, 9.419287, 0.719369, 0.651959, 0.544833, 1.650221, 1.65021, 0.999999, 4.406554, 1.009487, 0.379223, 127.856239, 324.762512, 940.834229, 1.341255], [
        ([[0.117386, 0.076776, -0.233311, 0.011426, 0.041248, -0.19437, -0.535708, 0.020058, 0.115788, -0.06117, 0.037933, -0.208905, 0.226073, -0.104969, 0.161826, -0.059027, 0.031073, -0.512534, 0.032453, 0.156172, -0.029081, -0.118845, 0.030692, 0.12851, 0.397701, -0.47504, 0.143548, 0.006675, 0.234315, -0.016803, 0.256494, -0.038872], [0.140564, 0.204182, -0.421213, -0.17431, -0.063159, 0.275503, 0.08873, -0.341392, 0.163398, -0.148686, 0.103448, 0.150234, -0.144027, 0.124541, -0.157789, -0.260678, 0.117053, -0.265603, 0.016621, 0.373898, -0.277323, -0.103362, 0.157512, 0.085658, -0.245849, -0.017609, -0.093849, 0.035433, 0.292238, 0.168059, -0.024312, 0.072365], [-0.362528, -0.191481, -0.208895, 0.02757, 0.156919, -0.02729, 0.281562, -0.079464, 0.158062, 0.095634, -0.16339, 0.311664, -0.026628, -0.077902, -0.120345, 0.3198, 0.199649, 0.35632, -0.159782, 0.249384, -0.27899, 0.041698, 0.25828, -0.259579, -0.111655, 0.355932, -0.082282, -0.171594, 0.356616, -0.031058, 0.190744, 0.241146], [-0.052157, 0.187308, 0.176694, 0.210977, 0.263686, 0.194945, 0.000571, -0.018858, -0.369642, 0.518919, 0.085771, -0.066506, -0.023369, -0.009381, -0.032583, 0.157916, 0.243187, 0.581291, 0.000928, -0.133093, 0.064011, 0.270922, 0.035231, 0.075813, 0.052517, 0.024182, 0.279241, 0.212976, -0.234551, -0.323704, -0.320137, 0.062109], [-0.081818, 0.021899, 0.336983, 0.01842, 0.112689, -0.115835, -0.13216, 0.153575, -0.071066, 0.064602, -0.016999, -0.123327, 0.345577, -0.087314, -0.088058, -0.243891, -0.047563, 0.365334, 0.489849, 0.028324, 0.189155, 0.204726, -0.031784, 0.013424, -0.19574, 0.082834, -0.099124, 0.303033, -0.517018, 0.316242, 0.259867, -0.173616], [0.066858, -0.106532, 0.072349, -0.167228, 0.328814, -0.164726, 0.25381, -0.008164, -0.094244, -0.146165, -0.03928, 0.031252, 0.138312, 0.286163, -0.034827, -0.04252, -0.059502, 0.277454, 0.139674, 0.127597, 0.135208, 0.023457, -0.115413, -0.145023, -0.531737, 0.230372, 0.002423, 0.098367, 0.144413, 0.05675, 0.241353, -0.357718], [-0.652284, -0.955876, -0.283988, 0.883463, 0.565601, -0.279889, -0.371819, 0.057483, 0.681519, -0.923691, -0.96676, 1.045576, -0.455989, -0.113316, -0.562528, 0.364761, 0.204437, 0.111539, -0.081351, 0.134928, -0.062663, -0.136941, -0.293822, 0.428485, -0.444016, 0.061632, -0.212703, 0.288706, 0.27096, 0.576967, 0.114079, 0.310067], [-0.030239, 0.025895, 0.162592, -0.006119, -0.145012, -0.053953, 0.095747, -0.197397, 0.009088, 0.064848, -0.009999, 0.116517, -0.318544, -0.028522, -0.086867, 0.068013, 0.056404, -0.002928, 0.109738, 0.082408, -0.211368, 0.00785, 0.027763, -0.001188, -0.260023, 0.267671, 0.423137, -0.297519, 0.183773, 0.040826, 0.270902, -0.228903], [-0.121322, -0.152994, -0.587303, 0.030813, -0.235587, 0.070468, 0.207161, -0.761296, -0.045885, 0.521151, -0.032266, 0.146412, -0.169913, -0.303958, -0.119143, 0.270374, -0.659131, 0.810695, -0.619849, 0.097103, -0.619763, -0.530358, 0.317802, -0.103179, 0.449595, 0.085612, -0.855491, 0.290367, 0.01997, -0.303852, 0.131905, 0.54973], [0.599378, 0.117773, -0.10836, 0.431418, -0.3703, -0.228338, 0.319583, -0.705806, 0.213285, -0.138958, 0.089111, 0.088456, -0.258645, 0.046427, 0.224082, 0.562247, -0.831921, -0.34838, -0.259917, -0.781322, -0.41752, 0.053371, -0.077031, 0.039165, 0.132265, -0.022964, -0.032678, 0.175434, -0.343878, 0.555722, 0.158704, -0.275533], [-0.001022, 0.057452, -0.120646, -0.047719, 0.193364, 0.100666, 0.284096, 0.067773, 0.220839, -0.214394, -0.27161, -0.127879, 0.15965, 0.021743, -0.069375, -0.032516, -0.048453, 0.112202, 0.048683, -0.042406, 0.284145, 0.189125, -0.031511, -0.135464, -0.444942, 0.160012, -0.185123, 0.233299, 0.198503, -0.224672, -0.151198, -0.108567], [-0.334639, -0.141687, -0.525509, -0.286611, -0.380444, -0.151333, -0.219685, -0.313106, -0.55069, 0.455284, 0.086048, -0.149395, -0.195252, -0.449671, 0.167921, -0.014798, 0.065446, -0.00802, 0.254064, 0.171461, -0.115036, -0.391055, 0.50798, 0.122347, 0.56879, -0.146768, -0.125964, 0.038163, 0.051403, 0.066124, 0.256338, 0.360991], [0.072537, -0.200372, 0.187751, -0.141122, -0.134255, 0.019554, 0.004334, -0.218715, 0.029531, 0.117845, 0.096109, -0.388231, -0.1126, 0.203866, -0.19336, 0.160854, 0.246086, 0.021889, 0.323294, -0.051643, -0.162262, 0.023538, -0.34403, 0.170057, -0.043282, -0.212597, -0.223076, -0.305523, -0.294175, -0.08628, -0.278906, 0.004612], [-0.173554, -0.046768, -0.195204, -0.162391, 0.396612, -0.065089, -0.071488, -0.027833, 0.279577, 0.029669, -0.197867, -0.512964, -0.320714, 0.01239, -0.063646, 0.131826, 0.344184, 0.172953, 0.118008, -0.214869, 0.373086, -0.055249, -0.393125, -0.122182, 0.110504, 0.217391, -0.067763, -0.162988, 0.230452, -0.17433, 0.247826, -0.328865], [-0.040813, -0.055343, -0.010338, -0.844938, -0.145071, -0.416878, 0.871803, -0.523113, 0.327436, -0.094384, 0.324826, -0.160462, -0.127779, -0.732269, -0.565329, -0.543248, 0.057043, -0.059754, -0.364145, 0.249428, -0.315488, 0.453944, -0.10062, 0.478429, 0.036437, 0.015111, -0.153258, -0.137077, 0.121177, 0.44079, -0.186835, 0.199656], [0.033651, 0.073005, 0.135353, 0.354134, -0.159763, 0.147502, -0.391618, -0.096331, -0.147495, 0.100174, -0.12862, -0.040368, 0.535207, 0.032513, 0.363621, 0.308241, 0.109298, -0.121953, -0.054465, 0.381602, -0.230957, 0.314513, 0.227258, -0.212716, -0.139601, 0.064512, 0.234307, 0.235746, -0.267169, -0.506825, 0.297219, -0.292982], [-0.208933, 0.233886, -0.210323, 0.224241, -0.882826, -0.658442, -0.163379, 0.762946, -0.493992, 0.0172, -0.135197, -0.017697, 0.237473, 0.323094, 0.59286, 0.282525, -0.255531, -0.380071, 0.910196, -0.344232, 0.247436, -0.623995, -0.064925, -0.254598, -0.118236, -0.079373, 0.227819, -0.301248, -0.055706, 0.026841, 0.352642, 0.413686], [0.354988, 0.145023, 0.390595, -0.179815, 0.143241, -0.044413, -0.414829, 0.128801, 0.274409, 0.058937, -0.049629, -0.133466, -0.572061, 0.03283, 0.035914, 0.331672, 0.542529, -0.197883, -0.327936, 0.144728, -0.162343, 0.6067, 0.853846, 0.11147, -0.109151, -0.121326, -0.025643, 0.021621, 0.74609, -0.346221, -0.255643, -0.040588], [-0.012246, -0.001215, 0.014509, 0.008112, 0.040122, -0.035339, -0.104028, 0.078355, 0.131751, -0.081592, 0.041336, -0.01581, -0.000985, -0.154644, -0.057006, -0.000207, 0.01708, 0.047401, -0.126432, -0.123304, 0.024711, 0.060006, -0.069217, -0.092517, -0.011952, -0.015253, -0.084705, 0.033008, 0.041887, 0.051997, 0.078303, -0.012036], [-0.106019, -1.365649, 0.324956, 0.021688, 0.072656, -0.184397, -0.105353, -0.256758, -0.016021, -0.003241, -0.07869, 0.219655, 0.082542, -0.204361, 0.286067, 0.221121, 0.078525, 0.076319, -0.231723, 0.013966, -0.301493, -0.366022, 0.064409, -0.53511, 0.199726, 0.488856, -0.591729, -0.017095, 0.011203, 0.353797, 0.047111, 0.328193], [0.086613, -0.149349, 0.119059, -0.181796, -0.226816, -0.149233, 0.127359, -0.036585, -0.113096, 0.129736, 0.032792, 0.136234, -0.183689, 0.321317, 0.292518, -0.003212, 0.149334, -0.193114, -0.244672, 0.077408, -0.431788, -0.183964, -0.014629, 0.050848, 0.201423, 0.342899, 0.085622, -0.155947, -0.066948, -0.058551, 0.082443, -0.018476], [-0.018676, -0.131005, 0.048969, -0.319958, 0.110206, 0.00974, -0.059148, -0.082573, 0.228896, 0.023361, 0.596966, 0.477948, -0.11615, 0.172519, -0.199969, 0.111038, -0.257158, -0.003354, 0.014813, -0.116368, -0.160522, -0.086765, -0.077793, -0.45342, -0.089624, -0.050177, 0.199555, 0.867641, 0.022894, -0.085977, 0.034005, 0.231205]], [-0.289077, -0.111489, -0.225053, -0.085433, -0.080495, -0.167083, -0.102631, -0.124919, -0.720543, -0.989643, 0.398296, -0.476557, -0.273427, -0.350983, -0.352406, 0.010732, 0.565628, -0.090387, -0.053658, -0.40087, -0.115615, -0.054217, -0.038003, -0.156962, -0.11924, -0.147075, -0.243157, 0.044072, -0.171753, -0.061342, 0.09398, -0.006905]),
        ([[0.297063, -0.211467, 0.094884, -0.287094, -0.034659, 0.187175, 0.190131, -0.273369, -0.059185, 0.027785, 0.231499, 0.252206, -0.061469, 0.185255, -0.428796, -0.206324], [0.166681, 0.315511, -0.736572, -0.136056, -0.231513, -0.778506, 0.010429, -0.505652, 0.022651, -0.035795, -0.056042, -0.011415, -0.169855, 0.319185, -0.104148, -0.147844], [0.073628, -0.162591, 0.302258, 0.049513, 0.092616, 0.447565, 0.321836, 0.402823, 0.126733, 0.150278, 0.499942, 0.123287, -0.114293, -0.068094, -0.019048, -0.001877], [-0.547732, -0.337058, -0.130288, 0.169827, -0.346376, 0.182597, -0.04365, 0.002174, 0.187839, -0.144193, -0.011853, 0.045377, -0.010195, -0.205018, 0.49795, 0.14496], [-0.338042, 0.363075, -0.43101, 0.112133, -0.129653, -0.291616, -0.395679, -0.045771, 0.001516, -0.22745, 0.295857, -0.499568, -0.18769, -0.055964, -0.180088, 0.447951], [0.185028, 0.164195, 0.177306, 0.10649, 0.210347, 0.289068, 0.223131, 0.047583, 0.068132, 0.191354, 0.369997, 0.226858, -0.036983, 0.26107, -0.233203, -0.245185], [0.101462, -0.160687, 0.132371, 0.073065, 0.247355, 0.215834, 0.417284, 0.400837, 0.218841, 0.298092, 0.231315, 0.260448, -0.120702, -0.844653, 0.244032, -0.614281], [0.273553, -0.25623, -0.018366, 0.006087, 0.195777, -0.021161, 0.411454, 0.036147, 0.249903, 0.357518, 0.273837, 0.120999, -0.129582, 0.070099, 0.03917, 0.369559], [-0.078623, 0.083261, -0.087857, 0.024519, -0.236785, 0.534733, -0.01983, 0.108181, 0.241157, 0.62931, 0.173017, -0.084694, -0.028867, -0.286964, -0.58052, 0.043418], [0.182561, -0.642502, 0.631488, 0.439174, -0.020335, 0.711846, -0.836358, 0.241976, -0.037209, 0.861178, -0.50371, -0.266812, -0.13174, -0.61582, -0.471565, 0.562712], [0.040882, 0.128906, -0.163287, -0.108548, -0.031774, -0.224265, -0.1729, -0.10398, -0.155717, 0.026817, 0.059329, -0.010772, -0.212866, -0.150789, -0.505907, -0.270455], [-0.058883, -0.10345, -0.513957, 0.124419, -0.199136, 0.089488, -0.055659, -0.189539, -0.437308, 0.249994, 0.043599, -0.221756, 0.078579, -0.38527, 0.039282, -0.206189], [0.007473, -0.018727, 0.022343, -0.037054, 0.332189, 0.132843, 0.407273, -0.075771, 0.327271, -0.492678, -0.051941, -0.030167, 0.00311, 0.16345, 0.655906, -0.185881], [0.249227, 0.023976, -0.503994, 0.121457, -0.05598, -0.896503, -0.482147, -0.176664, 0.066144, 0.036596, 0.211288, 0.054981, 0.142679, -0.162598, 0.151392, -0.386482], [-0.141622, -0.096796, 0.104371, 0.007007, -0.27309, 0.153011, 0.266352, 0.164545, -0.385359, -0.206848, 0.256817, 0.149553, -0.091409, 0.297435, 0.240309, 0.430006], [0.053147, 0.238974, -0.100063, 0.015838, 0.000628, 0.030569, -0.007903, 0.001112, -0.909316, 0.001063, 0.125832, 0.003952, -0.124032, -0.118678, -0.002353, 0.469629], [0.435012, 0.112071, -0.335214, -0.084874, -0.322014, 0.113165, -0.068868, -0.02496, -0.523699, -0.06914, 0.289929, -0.408306, -0.061391, -0.068185, -0.115506, -1.126027], [0.009173, -0.742217, 0.424233, 0.212581, 0.143786, 0.076628, 0.135651, 0.29247, 0.287569, 0.43184, -0.563084, 0.778553, -0.027639, 0.268763, 0.188446, 0.028585], [0.045488, 0.31314, -0.331205, 0.017114, -0.609766, -0.033358, -0.512489, -0.227382, -0.073966, -0.255276, 0.384395, -0.5452, 0.039418, 0.082369, -0.133908, 0.104251], [0.138746, -0.562892, 0.198643, 0.184079, 0.239438, 0.25418, 0.005158, 0.128517, 0.152978, -0.170368, -0.476627, 0.330504, -0.246558, 0.300948, -0.298835, -0.374427], [-0.1889, 0.280773, 0.21901, -0.098506, 0.283062, -0.516692, 0.071634, -0.347835, -0.01131, 0.148354, -0.049359, -0.071964, -0.025949, -0.422366, 0.07535, -0.696326], [-0.015949, 0.084496, -0.218139, -0.073622, 0.101669, -0.534506, -0.53646, -0.157076, 0.016431, -0.08762, -0.166335, 0.12308, 0.062566, -0.346167, -0.435479, 0.108263], [-0.030815, 0.193687, 0.19914, 0.261739, 0.219947, -0.018166, 0.292074, 0.139891, -0.25173, 0.253447, -0.086351, 0.150127, -0.147732, -0.273382, -0.052436, 0.045444], [-0.520359, 0.180871, 0.192648, -0.174823, -0.424789, -0.460737, -0.109551, 0.165927, 0.310674, 0.151839, 0.11186, 0.045781, -0.144169, -0.071202, 0.120154, -0.090853], [-0.247518, 0.249612, -0.021392, -0.141111, 0.092822, 0.23271, -0.130529, 0.281852, 0.325305, -0.127282, 0.237626, -0.014499, 0.051638, 0.229372, -0.095817, -0.853891], [0.021278, 0.321776, -0.005575, -0.29872, -0.04611, -0.384947, -0.699708, -0.215088, -0.051982, -0.096159, 0.012692, -0.756801, -0.20353, -0.14651, -0.157695, -0.072229], [-0.242613, 0.192414, -0.162521, 0.122493, 0.089908, -0.325181, 0.249189, -0.19171, 0.190317, -0.087218, 0.333173, -0.37894, 0.000984, -0.162572, 0.138284, -0.291877], [0.103061, 0.094748, -0.140019, -0.216056, -0.033763, 0.230509, 0.114247, -0.119081, -0.010094, -0.397851, -0.167013, -0.393023, -0.190214, -0.688196, -0.184966, -0.025271], [-0.024577, -0.417293, 0.239492, 0.383261, 0.190249, 0.266292, 0.383653, 0.082208, 0.297756, 0.163852, -0.011432, 0.543008, 0.042501, 0.190626, 0.130117, -0.035098], [0.239107, -0.434539, -0.124275, 0.339015, 0.200733, -0.185212, -0.27841, 0.384617, 0.263111, 0.3455, -0.121739, -0.227556, -0.213051, 0.145699, -0.185724, -0.028465], [-0.291974, 0.25225, 0.345674, -0.020379, 0.083638, -0.031825, 0.155716, 0.136837, 0.078591, -0.254187, 0.126587, -0.519252, 0.056675, 0.243486, 0.139314, 0.017854], [0.159939, -0.095072, -0.055859, 0.311884, -0.002705, -0.04485, 0.049036, 0.13208, -0.28846, -0.461625, 0.149301, -0.063914, -0.190483, -0.030087, -0.451551, -0.542118]], [0.18776, -0.06465, 0.023418, 0.343671, 0.065159, -0.303939, -0.001909, 0.303895, 0.053394, -0.598805, 0.179316, 0.290441, -0.252894, -0.387403, 0.248313, -0.209034]),
        ([[0.243336], [0.349438], [0.487939], [0.306287], [0.298289], [0.871111], [-0.385603], [0.301567], [-0.444585], [-0.771279], [-0.260966], [-0.436658], [0.18871], [0.844053], [0.5532], [0.924923]], [-0.204009]),
    ]),
    ([-0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 1.2325, -0.0, -0.0, 5.906742, 5.90694, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.000779, 4.050925], [1.010813, 0.870736, 0.822972, 0.71822, 0.624642, 2.270075, 11.568464, 0.66481, 9.419287, 0.719369, 0.651959, 0.544833, 1.650221, 1.65021, 0.999999, 4.406554, 1.009487, 0.379223, 127.856239, 324.762512, 940.834229, 1.341255], [
        ([[0.778275, -0.122597, -0.073003, -0.010624, 0.019345, -0.043047, 0.078506, -0.225899, -0.000521, 0.330492, 0.061486, 0.558401, 0.071463, -0.182197, -0.117348, 0.068321, -0.571917, 0.147883, -0.441759, -0.179558, -0.116459, 0.428101, -0.043719, 0.002076, -0.115108, 0.257702, -0.109578, -0.11254, -0.211792, -0.591739, -0.304844, 0.122666], [0.110494, -0.026225, 0.173129, 0.055356, 0.156435, 0.285746, 0.086693, -0.258661, 0.09434, -0.24179, 0.213233, 0.081335, -0.196382, 0.145209, 0.133985, 0.064234, -0.142703, 0.105739, 0.108254, -0.069695, 0.224654, -0.015554, -0.104836, -0.051586, -0.029032, -0.257444, 0.016305, 0.100685, 0.364513, -0.283898, 0.358419, -0.135322], [0.060844, 0.091241, 0.094431, 0.027023, -0.001918, 0.373578, 0.234367, -0.092199, -0.143553, -0.30076, -0.312312, -0.267378, 0.234918, -0.071273, -0.074631, 0.007242, 0.345399, 0.10264, 0.546842, 0.043558, -0.060671, -0.080039, -0.213865, 0.131253, 0.150886, -0.134634, 0.153799, -0.08443, 0.421857, -0.111801, 0.07281, -0.090022], [-0.12846, 0.137289, -0.12891, 0.021699, -0.352653, -0.248573, 0.039203, 0.14194, 0.142316, -0.339644, -0.086586, -0.402463, -0.263612, 0.143485, 0.305282, -0.311579, 0.4501, 0.211182, 0.299749, 0.39926, 0.119916, -0.019272, 0.376367, 0.048445, -0.010189, -0.039364, -0.031019, 0.008846, -0.202812, 0.047255, -0.095007, -0.059683], [-0.264597, 0.021238, -0.038094, 0.197362, -0.12229, 0.132808, -0.067871, 0.109167, -0.213791, 0.416119, -0.084463, -0.340416, 0.221031, -0.092938, -0.007435, -0.063145, 0.228422, -0.108599, -0.129208, 0.093749, -0.24701, 0.061723, -0.128768, -0.196892, -0.071006, -0.204302, -0.254819, 0.176642, -0.426929, 0.296639, -0.125189, 0.360725], [0.077395, 0.224471, -0.022385, 0.000166, -0.099768, 0.174648, 0.105148, 0.090455, 0.186031, 0.048454, -0.140613, -0.299622, -0.023696, -0.147706, -0.189795, -0.055961, 0.099211, 0.003043, 0.172159, 0.179096, 0.121529, 0.059945, -0.00598, 0.222597, 0.046171, -0.147437, -0.109851, 0.338117, -0.086517, 0.195713, 0.151443, -0.143395], [-0.210312, -0.203378, -0.009971, 0.948955, -0.672039, 0.285649, -0.253793, -0.267181, -0.286348, -1.282305, -0.322779, 0.33721, 0.909469, 0.249592, -0.333181, 0.187974, 0.222561, -0.100201, -0.327304, -0.354058, -0.854122, -0.190363, 0.134866, 0.212865, -0.010924, -0.015129, 0.197924, 0.886014, 0.54252, -0.068708, -0.098172, 0.613662], [0.020833, 0.321208, -0.032882, -0.00881, 0.154587, 0.051021, -0.088079, -0.036704, 0.119801, -0.04831, 0.135609, -0.013313, 0.017638, -0.094745, -0.228454, 0.0097, 0.125173, 0.171192, -0.018859, 0.049969, 0.183549, 0.012136, -0.139101, 0.176778, 0.059369, 0.096508, 0.207792, 0.27183, 0.1758, 0.227365, 0.06134, -0.049807], [-0.025315, 0.247977, 0.371501, 0.329068, 0.4339, 0.115898, -0.057721, 0.306054, -0.47843, -0.680037, -0.241247, 0.750093, 0.065739, -0.222435, 0.154251, -0.325018, 0.797604, 0.113408, 0.409563, 0.086747, -0.352159, -0.791677, 0.006404, -0.089874, 0.087134, -0.381548, 0.763101, -0.42253, 0.370248, 0.132112, -0.376049, -0.289244], [0.166378, 0.176382, -0.114433, 0.35442, 0.235402, -0.357654, -0.003174, 0.514681, 0.104514, -0.46072, 0.472403, -0.142739, 0.04947, -0.581939, 0.537395, 0.10338, 0.247439, -0.045253, 0.054124, -0.467757, -0.588198, -0.124477, -0.084827, -0.746619, 0.104397, -0.318448, 0.205453, -0.32857, -0.034307, -0.052483, -0.223176, 0.044799], [0.021213, -0.037254, 0.102732, 0.028294, -0.125433, -0.171167, -0.204096, -0.136231, -0.081305, 0.025602, 0.195567, -0.349381, 0.069045, -0.126723, 0.113905, 0.020927, 0.126447, -0.475864, 0.023764, -0.011497, -0.254579, -0.255553, -0.149834, -0.310781, -0.120884, 0.112262, -0.155383, -0.130201, 0.039916, -0.040855, 0.223781, 0.140038], [0.13896, -0.366727, 0.231448, 0.115937, -0.045548, 0.051487, 0.145145, 0.171053, -0.36409, -0.333089, -0.342154, 0.112152, -0.107515, -0.065287, 0.033681, -0.455083, -0.151013, 0.523992, 0.20107, 0.274896, 0.242243, 0.266615, -0.160628, 0.037739, 0.087267, -0.276286, 0.222708, -0.055547, 0.403111, -0.207181, -0.50078, -0.320791], [-0.152219, -0.547887, -0.224532, -0.292537, 0.33687, -0.017248, -0.334461, -0.078529, -0.045668, -0.035041, -0.038782, -0.132709, -0.23466, 0.065055, -0.385282, 0.155333, -0.193753, 0.12661, 0.082343, 0.038765, 0.189522, 0.104748, 0.232011, -0.183302, -0.096412, -0.428941, 0.33903, 0.124687, -0.064907, 0.259155, -0.025842, -0.41906], [0.165549, -0.300764, -0.192957, -0.140289, -0.165278, -0.372824, 0.038178, -0.006939, -0.029173, 0.074613, -0.162093, -0.142397, 0.144395, -0.051093, -0.005912, -0.018567, -0.102652, 0.221717, 0.308656, 0.005319, -0.143525, 0.035742, -0.072156, -0.083529, -0.1224, -0.094969, 0.07464, 0.627054, 0.222925, 0.064006, 0.380544, -0.141144], [-0.125762, -0.120274, 0.11954, -0.055092, 0.74289, 0.478599, 0.166851, -0.730909, 0.353658, -0.091907, -0.091604, 0.258425, -0.570401, -0.430295, 0.441313, -0.29121, -0.007439, -0.10491, 0.120929, -0.193804, 0.572386, -0.333413, 0.373569, -0.559029, 0.273744, -0.048962, -0.427021, -0.354353, 0.123765, -0.094747, 0.145193, 0.026429], [-0.271192, 0.078801, -0.010836, 0.079773, -0.368616, 0.103645, 0.158739, 0.701597, -0.121852, -0.099729, -0.100598, -0.011762, 0.271708, -0.11821, -0.02154, 0.019231, -0.175358, -0.323809, -0.030943, 0.264163, 0.149563, 0.080664, 0.23797, 0.045074, -0.259526, -0.152601, -0.219208, 0.061797, -0.115391, -0.18193, -0.073608, -0.122142], [0.259176, 0.206714, -0.130504, -0.454478, -0.552728, -0.270837, -0.063177, 0.106468, -0.145247, 0.400426, -0.363979, 0.10979, 0.231401, 0.73166, -0.44729, -0.206796, 0.020859, 0.040848, -0.281375, -0.042624, -0.291929, 0.583011, -0.715848, 0.022143, -0.53276, 0.140819, 0.280116, 0.017295, -0.304048, -0.241074, -0.684838, 0.021606], [-0.152857, -0.239418, -0.048008, -0.205266, -0.128201, -0.299828, -0.054185, -0.031618, -0.155737, 0.240448, 0.30114, 0.246153, -0.106003, -0.072764, -0.267703, 0.3838, -0.161083, 0.637709, -0.198975, -0.854875, -0.625297, 0.195221, -0.3888, 0.95523, -0.020585, -0.144682, -0.238878, -0.213883, -0.363131, -0.511363, -0.144038, 0.269477], [0.051525, 0.017527, 0.031491, 0.131352, -0.002815, -0.081876, -0.186247, -0.00378, 0.054973, -0.038428, 0.063401, -0.02914, 0.174678, 0.081232, -0.02332, -0.074144, -0.006139, -0.104561, 0.176184, -0.020264, -0.044459, -0.017659, -0.0085, -0.041346, -0.09527, 0.120301, -0.063037, -0.029191, -0.047722, 0.048187, 0.197255, -0.076988], [0.116429, -0.201364, 0.777938, -0.051024, -0.06385, -0.04748, 0.846264, -0.179949, 0.131185, -0.042517, -0.596286, 0.21303, 0.491966, -0.192408, -0.347413, 0.155613, 0.018454, -0.198625, 0.593227, 0.15346, -0.035417, -0.596836, -0.195214, -0.288629, 0.740886, 0.09777, 0.576819, -0.215237, -0.425418, 0.058511, 0.228799, 0.375891], [-0.088174, 0.281886, 0.177192, -0.125077, 0.243599, -0.054122, 0.253566, 0.075224, 0.031889, 0.008369, 0.159416, -0.101369, -5.2e-05, -0.19414, 0.007024, -0.268922, -0.118165, 0.295517, -0.212324, 0.016996, 0.289, -0.062904, 0.08387, 0.041331, 0.011245, 0.167131, 0.287556, 0.05776, -0.023122, -0.089423, 0.040426, 0.034542], [0.030173, 0.026965, -0.031641, 0.870402, -0.040158, -0.1126, 0.169468, -0.031387, 1.001498, -0.00737, -0.004507, 0.057058, -0.355073, 0.124742, -0.044169, -0.356078, 0.032762, 0.075511, -0.14529, 0.034122, -0.164205, 0.063377, -0.399909, -0.034, -0.010045, -0.213657, 0.047077, -0.063328, 0.202934, -0.044157, 0.034164, -0.079683]], [-0.175419, -0.449857, 0.124997, 0.201463, 0.197678, -0.765656, 0.007428, 0.071597, -0.360409, -0.758175, -0.029627, -0.735706, -0.135825, 0.310102, -0.320126, -0.762192, 0.04666, 0.019334, -0.069499, -0.847828, 0.084065, -0.112614, 0.191814, -0.214224, -0.49972, 0.001084, 0.253886, 0.520722, -0.136723, -0.79084, -0.192559, 0.262119]),
        ([[0.22876, -0.035042, 0.141379, 0.344974, -0.144112, 0.104442, 0.392314, 0.310276, 0.215166, -0.131298, 0.424207, -0.318478, 0.248654, -0.01528, 0.056069, 0.111368], [-0.060869, -0.014452, -0.054808, -0.37848, -0.051869, -0.094971, -0.072133, -0.009784, -0.171409, -0.165088, -0.347358, 0.12256, -0.048638, -0.732087, 0.011676, 0.086741], [-0.032006, -0.147082, 0.175569, 0.015544, -0.18866, 0.077193, -0.182175, -0.772503, -0.339502, -0.042512, -0.347146, -0.045992, -0.17607, 0.1405, 0.100618, 0.020075], [-0.290547, -0.14525, 0.437827, -0.080648, -0.167084, -0.110985, -0.381877, 0.008856, 0.188997, 0.13712, -0.195534, 0.424149, 0.040127, -0.168703, -0.201238, -0.17442], [-0.295404, -0.214583, -0.412071, -0.006287, 0.367138, -0.968625, -0.269175, -0.060496, 0.191811, -0.294259, -0.026813, -0.517862, -0.103493, 0.017703, 0.183962, -0.100671], [0.13699, 0.03406, 0.066377, 0.119921, 0.452348, -0.002028, 0.268918, 0.12268, -0.272121, -0.101407, -0.01695, -0.443839, 0.17396, -0.43008, 0.127022, -0.591582], [0.179716, 0.102789, -0.042455, 0.079452, -0.842668, -0.120921, -0.30346, -0.162434, -0.696637, -0.049311, -0.309669, 0.201759, -0.130943, -0.130328, 0.062552, 0.120019], [0.375421, -0.287394, 0.392805, 0.31841, 0.108581, -0.616508, -0.038996, 0.141406, 0.087951, 0.0968, 0.151505, 0.230294, 0.345284, 0.151984, 0.285125, 0.22927], [-0.501273, -0.031941, 0.145104, -0.314346, 0.207264, -0.44251, 0.276607, 0.086317, 0.123508, -0.248546, -0.19947, 0.265931, -0.167468, 0.059556, -0.122798, 0.201982], [0.402428, -0.17791, -0.375601, 0.373287, 0.259204, -0.310532, 0.002433, 0.439515, 0.358545, 0.469341, 0.146423, 0.326263, 0.367399, 0.34215, -0.365477, 0.156579], [0.246955, -0.196401, -0.033047, -0.178209, -0.032054, 0.356306, 0.217353, 0.096919, -0.034108, 0.233079, -0.001053, -0.081135, 0.249474, 0.19484, -0.392498, 0.33628], [0.325578, -0.114203, -0.265393, -0.103416, -0.953441, -1.133573, 0.260002, 0.115444, 0.158721, -0.012302, -0.271858, -0.202392, 0.049575, 0.064662, 0.342417, 0.261503], [0.130856, -0.144133, -0.03236, 0.371687, -0.133763, -0.217064, -0.520846, 0.112172, 0.239435, -0.056975, -0.293725, -0.118591, -0.381883, -0.054408, 0.137841, -0.115556], [-0.485581, -0.092799, 0.144778, -0.661121, 0.133935, 0.229749, 0.048523, 0.303671, 0.255226, 0.109794, 0.074247, -0.061156, 0.141585, 0.255601, -0.103048, -0.054189], [0.035068, 0.055617, -0.156874, -0.520504, -0.188856, -0.039991, 0.375096, -0.373113, 0.218468, -0.433339, 0.224967, -0.225849, -0.077823, -0.037066, -0.060574, 0.038288], [0.145007, -0.044816, 0.247012, 0.11298, -0.10162, 0.404212, 0.100492, 0.059915, -0.138908, -0.939194, 0.204974, -0.00504, 0.007728, 0.253877, 0.139969, 0.255528], [0.218908, -0.200092, -0.123976, 0.187698, 0.448699, 0.067673, 0.226203, 0.38803, 0.026013, 0.097975, 0.073695, -0.25253, 0.313093, -0.061113, 0.32118, -0.176466], [0.177891, -0.169433, 0.071651, 0.076602, 0.132754, 0.213925, 0.397038, -0.258122, -0.853311, 0.258827, 0.20325, -0.067308, -0.146501, 0.291831, 0.162632, 0.132841], [-0.026247, -0.207663, 0.105131, -0.180683, -0.054651, 0.232792, -0.271062, 0.004607, 0.280308, -0.261087, -0.140149, -0.014702, -0.124464, 0.454783, 0.357757, 0.397689], [0.259202, -0.167308, -0.286264, -0.567029, -0.047561, 0.304852, -0.078214, 0.333241, 0.425699, 0.643023, -1.169719, -0.286344, -0.505459, -0.653954, -0.17678, -0.352206], [0.280542, 0.054294, 0.320678, 0.05613, -0.462896, 0.343924, 0.467038, -0.057014, -0.352788, 0.136667, -0.008248, 0.361136, -0.032327, 0.315873, 0.219568, -0.316002], [0.250311, 0.065248, 0.271128, -0.66923, -0.705499, -0.342588, 0.074285, 0.074613, 0.038694, 0.301752, 0.107538, 0.022855, 0.269967, -0.075561, 0.013134, -0.240756], [0.085725, -0.238798, -0.359635, -0.392957, 0.303909, -0.131927, -0.047708, -0.131778, -0.32091, 0.211943, -0.075446, -0.273584, 0.019416, -0.182004, -0.256246, 0.110416], [-0.555366, -0.220543, 0.268945, -0.425648, 0.389444, 0.10427, -0.026692, 0.295194, -0.18022, -0.242256, -0.163209, 0.327563, -0.024597, 0.218745, 0.090064, -0.190518], [-0.147446, 0.017612, -0.087965, -0.390808, 0.133365, 0.039475, 0.162932, -0.52105, 0.385485, 0.389279, -0.491223, -0.201343, -0.640036, -0.306478, -0.005611, 0.079773], [-0.036379, 0.113405, -0.1504, 0.115842, 0.029572, -0.063817, 0.224795, -0.017304, 0.178046, -0.664646, 0.217236, 0.193183, 0.109842, 0.029782, -0.08022, -0.215028], [0.040446, 0.145692, 0.012779, -0.168698, -0.268243, -0.329804, -0.739176, -0.083174, -0.053163, -0.067332, -0.151099, 0.227237, -0.100805, -0.435242, -0.21188, 0.113607], [-0.273969, -0.054579, -0.057211, 0.031004, -0.209481, 0.085764, -0.309944, 0.194473, -0.375349, -0.535658, -0.00022, -0.019425, -0.44322, -0.237548, -0.38166, -0.274139], [0.188726, 0.02328, 0.101666, 0.361754, 0.341755, 0.319663, 0.201143, -0.312415, -0.813324, 0.172473, 0.488971, 0.069105, -0.152764, -0.086694, 0.330226, 0.033942], [0.051554, -0.12808, -0.213156, -0.935224, -0.030342, -0.30403, 0.354004, -0.024392, 0.137822, -0.16261, 0.233541, -0.309445, 0.048493, 0.416088, 0.71644, -0.071988], [0.220191, -0.19423, 0.143625, -0.429314, -0.12919, -0.064763, 0.016625, 0.063374, -0.267535, -0.271362, -0.10747, 0.016304, 0.054367, 0.029622, -0.179038, 0.371928], [-0.230955, -0.155525, 0.126847, 0.299908, -0.208141, 0.311086, -0.085535, 0.234743, -0.05168, 0.180328, 0.077638, -0.188028, 0.294582, 0.22584, -0.022342, 0.199744]], [-0.466058, -0.241744, 0.454456, -0.05827, -0.280497, -0.284884, -0.189155, -0.174519, -0.313246, -0.273523, 0.310224, 0.335375, 0.073425, 0.001095, 0.226701, -0.18162]),
        ([[0.646707], [-0.052808], [0.375285], [0.855692], [-1.055729], [0.337312], [-0.426642], [-0.276781], [0.857047], [0.948275], [-0.292147], [-0.390829], [-0.338269], [-0.390219], [0.343749], [-0.319016]], [0.036063]),
    ]),
    ([-0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 1.2325, -0.0, -0.0, 5.906742, 5.90694, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.000779, 4.050925], [1.010813, 0.870736, 0.822972, 0.71822, 0.624642, 2.270075, 11.568464, 0.66481, 9.419287, 0.719369, 0.651959, 0.544833, 1.650221, 1.65021, 0.999999, 4.406554, 1.009487, 0.379223, 127.856239, 324.762512, 940.834229, 1.341255], [
        ([[0.227885, 0.092682, 0.024383, 0.056234, 0.073979, 0.847132, -0.009252, -0.059334, -0.135127, -0.033506, -0.062555, -0.088016, 0.042997, -0.072548, 0.197849, -0.471529, 0.040263, -0.222635, -0.106641, 0.157394, 0.032482, -0.236896, -0.299175, 0.046299, -0.189287, -0.635265, 0.157041, -0.092593, -0.05924, 0.203386, 0.310156, -0.061812], [-0.153671, 0.192622, -0.06625, -0.209734, -0.298983, -0.065084, -0.206088, -0.609998, 0.084304, 0.311714, -0.119788, -0.122628, 0.036128, -0.005061, 0.078531, 0.005719, 0.13824, 0.127326, 0.188549, -0.070625, -0.117008, -0.136627, 0.249337, -0.356011, -0.250466, 0.240612, -0.059661, 0.0677, -0.142528, -0.110549, 0.204451, -0.075307], [-0.073555, 0.134689, 0.165348, -0.095883, -0.361806, -0.402298, 0.030457, -0.289736, 0.105342, -0.130899, 0.196374, 0.168874, -0.011552, 0.228078, 0.154095, 0.045767, 0.205921, -0.216729, 0.096749, -0.035714, -0.012344, -0.052829, 0.506531, -0.102191, 0.247926, 0.585997, 0.095671, 0.265233, -0.064498, -0.333363, -0.237848, 0.205973], [-0.0978, 0.069408, 0.116378, 0.315159, -0.232054, -0.52633, -0.014153, 0.607184, 0.265605, -0.156072, 0.061458, 0.080155, -0.176738, -0.177769, -0.239527, 0.153172, 0.077269, 0.023138, 0.105093, -0.115505, 0.041349, -0.092631, 0.142123, 0.362379, 0.003288, 0.404582, -0.004598, -0.018393, 0.13853, 0.217035, -0.17389, 0.121574], [-0.201913, -0.041168, 0.129426, -0.003755, 0.179386, -0.06306, 0.462032, 0.570194, -0.173634, -0.006262, -0.338782, 0.022117, 0.023038, -0.321852, -0.166199, -0.386623, -0.254486, 0.098469, 0.318383, 0.037746, -0.46971, 0.115347, -0.325184, 0.157643, 0.018212, -0.776686, -0.111441, 0.034507, 0.028631, 0.01518, -0.357262, 0.44516], [-0.133836, 0.007357, 0.095594, 0.099224, 0.055468, -0.275915, -0.172848, 0.181242, -0.080285, 0.053015, -0.156421, 0.319783, 0.183782, -0.028098, 0.091452, 0.136869, -0.058183, 0.022042, 0.201481, -0.060828, 0.003805, 0.020046, 0.253225, -0.128819, 0.174228, 0.207104, -0.102918, 0.05174, 0.051701, -0.036803, -0.095819, -0.175788], [-0.306394, -0.238168, 0.589254, 0.074784, 0.184545, -0.136155, -0.083592, -0.026443, 0.102613, 0.25358, 0.183949, -0.696107, -0.308378, -0.136986, -0.072134, -0.506652, 0.88601, -0.18278, -0.342014, -0.212535, 0.222658, -0.246728, 1.11016, 0.267898, 1.039252, -0.167286, 0.02985, 0.1299, 0.145087, -0.284229, 0.183498, 0.387043], [-0.262546, 0.057418, -0.108954, 0.178966, 0.085746, 0.293746, 0.311561, 0.125641, -0.087014, 0.281885, 0.260258, -0.091695, 0.017312, 0.024971, -0.050151, 0.09517, -0.083536, -0.123289, -0.10489, 0.02261, 0.089562, -0.204243, -0.012206, -0.063579, -0.081906, 0.063733, -0.300915, 0.192757, -0.244103, -0.447149, 0.148838, -0.123681], [0.498174, -0.477983, -0.342448, -0.028621, -0.550228, -0.562251, -0.172681, 0.371289, 0.099761, -0.523235, -0.765336, -0.706649, 0.039478, 0.973255, 0.000374, 0.197294, 0.071199, 0.103607, -1.338717, -0.166976, -0.330687, -0.050062, 0.255407, -0.058908, 0.107425, -0.091341, 0.615157, 0.308281, -0.009783, -0.657784, -0.425576, 0.563211], [-0.144081, -0.97804, 0.098817, -0.05682, -0.397991, 0.07305, -0.216952, -0.262161, 0.060335, -0.679244, -0.292534, -0.204599, -0.055807, -0.315379, -0.875957, -0.028483, -0.321508, 0.150215, -0.22298, -0.152473, -0.314609, 0.004357, -0.060268, -0.119197, 0.07112, -0.316082, 0.048738, 0.621785, -0.67375, 0.398337, -0.432161, 0.192137], [-0.214866, -0.032953, -0.077433, 0.06, -0.076323, 0.161106, 0.142045, 0.081517, 0.050341, -0.239293, 0.03398, -0.047143, -0.261383, -0.088882, 0.123881, 0.291678, -0.024376, -0.329434, 0.24262, 0.112977, 0.268378, 0.329544, -0.083211, -0.222131, 0.092704, 0.151004, -0.124024, -0.003303, -0.234695, 0.294415, 0.122199, 0.021716], [0.482929, 0.117693, 0.301562, -0.120116, -0.325841, -0.082457, 0.264016, -0.044777, -0.127296, 0.0966, -0.041325, -0.412917, 0.240467, 0.200793, 0.256497, -0.466338, -0.105041, 0.43242, -0.216795, -0.200567, -0.356046, -0.606883, 0.558294, 0.734626, 0.126697, -0.065381, 0.493319, 0.209234, 0.416764, -0.316026, -0.261624, 0.223461], [0.519957, 0.010231, -0.189032, -0.197273, -0.12693, -0.052809, 0.190492, -0.00637, 0.181514, -0.16838, 0.089901, -0.234, 0.048175, 0.12156, -0.412558, 0.045261, -0.050951, -0.150005, -0.029858, -0.098878, 0.108699, 0.142741, -0.142695, 0.152048, -0.159763, -0.009779, 0.11685, -0.030282, 0.07171, -0.138462, -0.012751, -0.131775], [0.140735, 0.063554, 0.26231, 0.230001, -0.19631, 0.102162, 0.041376, 0.072541, 0.215584, -0.26006, -0.340211, -0.055434, 0.198747, -0.210302, -0.146394, -0.194732, 0.18641, -0.184111, 0.182754, 0.196558, 0.357897, -0.059215, 0.165875, -0.108579, 0.026822, -0.020486, -0.171379, -0.038056, 0.118553, -0.144195, 0.312575, -0.132579], [-0.064248, 0.323122, -0.111597, 0.659258, -0.291979, 0.268428, 0.341992, -0.122778, 0.433609, -0.425411, -0.291496, -0.221272, -0.662115, -0.201982, 0.261916, 0.227938, -0.263274, 0.322547, -0.714105, 0.067298, -0.08863, -0.627582, -0.453288, -0.156638, -0.10282, 0.024231, -0.160589, 0.489186, -0.258193, -0.33721, 0.214897, 0.210307], [0.090161, 0.197687, 0.345164, 0.106174, 0.169502, -0.163116, 0.213264, -0.165679, -0.324536, -0.068104, 0.086595, 0.206132, 0.191929, -0.206279, -0.014523, -0.293166, -0.337851, -0.003903, 0.043031, 0.305347, -0.301844, 0.115105, 0.100834, 0.260674, -0.024848, 0.159238, -0.182967, -0.513331, 0.154705, 0.008174, -0.231089, 0.125352], [0.321004, -0.357779, 0.044204, -1.049186, 0.526495, 0.327864, 0.175723, 0.374783, -0.596226, 0.263059, -0.238361, 0.318087, 0.580147, 0.242684, -0.294187, -0.793754, -0.314334, -0.073697, 0.552978, -0.268428, -0.021277, 0.446659, 0.103478, 0.072038, -0.078927, -0.209883, 0.341703, 0.231537, -0.550983, -0.064907, -0.310395, -0.48302], [0.36202, -0.559585, -0.197162, 0.369248, 0.171508, 0.690409, 0.256439, 0.159621, 0.177787, 0.741553, -0.429533, 0.007498, 0.320568, 0.223496, -0.643096, -0.209651, 0.074287, 0.66164, 0.076493, -0.169113, -0.023662, -0.40368, 0.127192, 0.184902, -0.165358, 0.368766, 0.025545, -0.133136, 0.239965, 0.49255, 0.771486, -0.165294], [-0.056072, -0.16012, 0.132661, 0.260584, -0.038666, 0.021253, 0.036218, 0.013694, -0.049168, -0.008883, 0.112403, -0.120363, 0.009131, -0.036726, 0.061116, 0.022304, -0.014898, -0.094453, -0.02452, 0.086641, -0.037384, 0.032721, 0.059891, -0.073263, -0.00844, -0.012317, -0.076789, 0.004352, 0.123036, -0.023339, 0.227464, 0.230543], [0.017227, 0.302431, 0.508596, -0.263469, -0.046784, 0.082487, -0.10556, 0.107723, -0.012419, -0.067458, -0.087346, 0.113537, 0.13533, 0.894548, 0.029454, 0.042566, -0.113977, 0.055259, -0.225943, -0.637528, 0.29133, -0.050901, 0.137407, 0.023844, 1.052034, -0.007988, 0.100453, 0.111335, -0.326916, -0.095589, -0.118918, -0.053905], [-0.262052, 0.297955, 0.04196, -0.039766, 0.121292, -0.067303, -0.548985, -0.291112, -0.037108, -0.165568, -0.018341, 0.176021, -0.058523, 0.404492, -0.037586, 0.22591, -0.346314, 0.244631, -0.155678, -0.050389, 0.084996, 0.076429, -0.069446, 0.002928, 0.0056, -0.054432, 0.02972, -0.153199, 0.077997, 0.014293, -0.015031, -0.119802], [-0.180885, -0.116278, -0.247711, 0.009386, -0.054667, -0.051488, 0.12973, -0.049544, -0.637206, -0.088901, 0.338289, 0.780544, -0.68054, -0.032351, -0.130107, 0.071034, -0.147417, -0.277833, -0.040895, -0.28484, 0.376794, -0.071716, 0.026412, 0.081376, -0.04482, 0.01046, 0.783298, -0.085724, -0.369405, 0.171629, -0.117479, -0.15216]], [0.08822, 0.198387, -0.50455, -0.093159, -0.047268, -0.005069, 0.029889, -0.077622, 0.233903, -0.516125, 0.097272, -0.096737, -0.140187, -0.080582, -0.432913, -0.462368, 0.121763, 0.08695, -0.051695, 0.119582, 0.564057, -0.205984, -0.689772, -1.078219, 0.337254, -0.153406, -0.180165, -0.378696, -0.682074, 0.166396, -0.724249, 0.028156]),
        ([[-0.197347, 0.034063, -0.155753, -0.03183, -0.191244, -0.037446, 0.141231, -0.391226, -0.429441, -1.481537, 0.027941, -0.038827, -0.028699, 0.008715, 0.012664, 0.07974], [-0.055595, 0.146183, 0.172414, 0.33858, -0.213227, -0.190175, -0.076059, 0.426089, -0.121964, 0.03169, 0.334766, -0.333892, 0.264074, -0.072529, 0.16951, -0.727178], [-0.221534, -0.02691, -0.221126, 0.162243, 0.347263, 0.216056, 0.217307, -0.065114, 0.126981, -0.0799, 0.006329, -0.073353, -0.10557, -0.773628, 0.135846, -0.801717], [-0.368611, -0.333635, 0.406175, -0.548296, -0.234569, 0.14521, -0.235038, -0.222419, 0.095785, -0.154068, 0.107751, -0.265195, -0.781173, -0.029674, -0.705572, -0.075035], [-0.092191, 0.011833, -0.147868, 0.040563, 0.245893, 0.437011, 0.287749, -0.016665, -0.063648, 0.243595, 0.206178, 0.209823, 0.09541, 0.232107, 0.130477, -0.075636], [0.256136, -0.222304, 0.179758, 0.414724, 0.064541, -0.089524, -0.254462, 0.138205, 0.257174, 0.069669, -0.004196, 0.374976, 0.369092, 0.197828, 0.474045, -0.002849], [-0.169577, -0.347392, 0.030005, -0.158028, -0.113588, 0.409925, -0.13797, -0.128077, 0.133628, -0.329343, 0.126613, -0.276851, -0.425026, -0.130717, -0.339831, -0.168201], [0.206659, 0.578059, -0.344947, 0.225687, 0.334581, -0.056994, -0.007312, 0.370686, -0.186318, 0.203989, 0.258858, 0.336235, -0.040542, 0.547732, 0.264331, 0.250981], [0.068835, -0.469591, 0.19679, -0.179885, -0.058052, 0.148526, -0.071571, -0.489321, -0.368177, 0.086583, -0.246732, -0.140595, -0.27765, 0.069524, -0.450443, 0.440751], [-0.301783, 0.583415, -0.242766, -0.269467, 0.219847, -0.668022, 0.135785, -0.288921, -0.552838, 0.628836, -0.363666, 0.351885, -0.019955, -0.108069, -0.665989, -0.259315], [0.167077, 0.119907, 0.062031, 0.016783, -0.267874, 0.101133, -0.408728, 0.051467, -0.268927, -0.075962, 0.175649, -0.237871, -0.027114, 0.142708, -0.553305, 0.082119], [-0.203922, -0.375268, 0.122383, -0.147217, 0.103145, -0.166126, 0.217073, -0.213177, 0.15286, -0.066355, -0.518117, 0.088425, -0.354456, 0.136204, -0.120113, 0.542293], [-0.145004, 0.191111, -0.187276, -0.059915, -0.078892, 0.061774, -0.09516, 0.449535, 0.755191, -0.31309, 0.091456, -0.079744, -0.046882, -0.38179, 0.058969, -0.544245], [-1.32645, 0.094963, -0.211325, -0.11297, -0.080785, 0.157359, -0.166243, -0.066226, -0.223368, -0.06002, 0.099835, -0.141645, 0.009719, -0.393663, -0.3158, 0.108053], [0.240103, 0.005472, 0.043878, 0.250146, -0.208374, -0.186738, -0.312525, 0.190052, 0.20407, -0.164225, 0.055274, 0.023869, 0.001696, -0.147526, -0.005608, -1.369374], [0.098593, 0.30544, 0.155017, 0.347455, 0.231366, 0.176203, -0.226115, 0.04166, 0.01218, -0.462964, 0.398491, -0.040046, 0.074423, 0.225655, 0.688052, 0.402852], [0.17585, 0.18473, -0.011716, 0.431431, 0.060687, 0.116324, 0.117113, 0.388672, 0.065985, 0.186382, 0.066768, 0.09456, 0.298659, 0.141208, -0.221894, 0.006893], [0.329373, 0.055672, 0.076458, -0.253298, -0.390762, 0.028931, -0.157375, -0.166881, 0.232216, -0.073833, -0.047232, 0.004125, 0.199903, 0.263154, 0.119518, 0.095482], [0.050549, -0.207342, 0.103695, -0.17459, 0.00286, -0.351291, -0.088871, -0.352128, 0.060568, -0.488603, -0.519278, -0.561293, -0.189643, -0.355371, -0.033021, -0.182724], [0.047864, 0.201672, -0.037193, -0.128218, 0.16399, 0.01544, 0.035382, 0.076797, -0.69448, -0.1952, 0.000957, 0.036413, 0.015565, 0.125805, 0.231612, 0.081664], [-0.20491, -0.101878, -0.114473, -0.298697, 0.114934, -0.384351, 0.345202, -0.175841, -0.284158, -0.411727, -0.516751, -0.21596, -0.282209, -0.027152, -0.306137, -0.30674], [-0.125814, 0.153827, 0.121318, 0.527992, 0.148408, 0.068192, -0.131039, -0.11122, 0.167794, 0.64363, -0.203676, 0.013404, 0.367319, 0.184056, 0.025929, 0.127425], [-0.239791, -0.035392, -0.072895, 0.11267, 0.549543, -0.187689, -0.239309, 0.374263, 0.242926, 0.003389, -0.057409, 0.03338, -0.134567, -0.252293, -0.342334, -0.385866], [0.334575, -0.307776, -0.424175, 0.25314, 1.08022, -0.128457, -0.031138, 0.253073, -0.04137, 0.454032, 0.492681, -0.305957, -0.684042, -0.077518, 0.735035, 0.341348], [-0.304095, -0.166354, 0.072594, -0.09013, 0.004206, 0.054715, 0.038215, 0.076137, -0.143882, 0.134671, -0.006589, -0.165392, -0.55753, -0.405282, 0.175115, -0.408311], [0.633364, 0.02373, 0.327682, 0.01146, 0.437209, -0.52715, 0.14193, 0.634088, -0.273087, 0.104107, 0.32207, 0.416753, 0.325734, -0.152372, 0.569929, 0.256871], [-0.096216, -0.184672, -0.177719, 0.235026, -0.445846, 0.009767, 0.104808, -0.060565, -0.049285, -0.320827, 0.139044, -0.007165, -0.265776, -0.318823, -0.086836, 0.032735], [0.244644, -0.371879, -0.702122, 0.04728, -0.165257, -0.481785, 0.227947, 0.041176, 0.322576, -0.055012, 0.107868, 0.150415, 0.001532, 0.430737, -0.143248, 0.140117], [-0.018217, -0.132872, 0.065853, -0.44789, 0.141587, -0.229342, 0.279506, -0.229257, -0.520772, -1.096049, 0.178544, 0.06102, 0.086248, 0.432673, 0.361718, 0.080517], [-0.08668, 0.044946, 0.025668, -0.801932, -0.1052, 0.025177, 0.082441, 0.294821, 0.037165, -0.402297, 0.324284, 0.37758, 0.164237, -0.175543, 0.001064, -0.109696], [0.38332, -0.251293, -0.232412, 0.360641, 0.164656, -0.646754, 0.487388, -0.757062, -0.461521, 0.119942, 0.122208, -0.039069, 0.155259, 0.160541, 0.338589, -0.52566], [0.152321, 0.056959, 0.176362, 0.092733, -0.016164, 0.047796, -0.250492, 0.22993, -0.157674, 0.284086, 0.262361, -0.006214, 0.309658, 0.027064, 0.20905, -0.017269]], [-0.13756, 0.414025, 0.280066, -0.377607, -0.540329, 0.309984, 0.194863, -0.285348, 0.184839, 0.29054, -0.115651, 0.336824, -0.115742, -0.513667, -0.243379, -0.387955]),
        ([[-0.750269], [-0.263785], [-0.172191], [0.662331], [-0.790382], [-0.2937], [-0.252997], [0.414447], [0.471626], [0.588941], [0.280218], [-0.415186], [-0.452128], [-0.492892], [0.676183], [-0.770261]], [0.239368]),
    ]),
]
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
        'gate_thr': 11.0,
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
        'pred_blend': 0.7,
        # exp_r3_pms: 1 = the IOC taker uses the round-3 150 ms ridge (pred_on 1, pred_mode 1, pred_model 3), 2 = the
        # 500 ms one (pred_model 4); blended with fair_sig by pred_blend, scaled by pred_scale. 0 = off (base).
        'pms_on': 1,
        # exp_r3_rfit: 0 = champion model 3; else the PRED_MODELS index of a refit variant (5 parity, 6 r3_100, 7 r3m_75, 8 r3m_100,
        # 9 r3r2_75, 10 r3r2_100, 11 h100, 12 h200, 13 alpha x3, 14 alpha /3, 15 h250, 16 h300, 17 h400, 18 h200+r2, 19 h200+mirror).
        'rfit': 19,
        # exp_r3_mlp: mlp_on 1 = the taker prediction is (1 - mlp_mix) * ridge + mlp_mix * MLP (before pred_blend with fair_sig);
        # the MLP runs only when |ridge| >= mlp_gate ticks (runtime). 0 = off (identical to the champion).
        'mlp_on': 1,
        'mlp_w': 100,
        'mlp_mix': 1.0,
        'mlp_gate': 0.3,
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
        'rt_ms': 130.0,
        'ri_ms': 0.0,
        'qa_max': 0,
        'fq_k_sp2': 2.0,
        'of_ms': 0.0,
        'fq_k_sp3': 1.5,
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
        self.mlp_w = int(params['mlp_w'])
        if self.mlp_on:
            # mlp_w 0 = the single 75 % net of mm_r3_mlp2 (parity), 75 / 100 = the 3-seed ensembles
            ens = {0: [(MLP_MEAN, MLP_SCALE, MLP_LAYERS)], 75: MLP_ENS_75, 100: MLP_ENS_100}[self.mlp_w]
            self.mlp_nets = [(np.array(mu), np.array(sc), [(np.array(w), np.array(b)) for w, b in ly]) for mu, sc, ly in ens]
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
            h0 = np.array((pg[0], pg[1], f70, pg[3], pg[4], f70 * log1p(fw), f70 * af, sg100 * af, f70 * spread, spread,
                           imb1, cnt1, log1p(qb), log1p(qa), ltr_sign, ltr_lsz, (ltp - mid2 * 0.5) if ltp > 0 else 0.0,
                           (mid2 - rm2) * 0.5, self.pr_ev10, s100, self.pr_s1000,
                           log1p((now - self.pr_ttr) / 1000.0) if self.pr_ttr >= 0 else 10.0))
            m = 0.0
            for mu, sc, ls in self.mlp_nets:
                h = np.clip((h0 - mu) / sc, -8.0, 8.0)
                h = np.maximum(h @ ls[0][0] + ls[0][1], 0.0)
                h = np.maximum(h @ ls[1][0] + ls[1][1], 0.0)
                m += float((h @ ls[2][0] + ls[2][1])[0])
            m /= len(self.mlp_nets)
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
