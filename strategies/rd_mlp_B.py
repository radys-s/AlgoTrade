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
MLP_MEAN = [-2.7e-05, -1.9e-05, -1.5e-05, -9e-06, 2e-06, -9.2e-05, 0.000129, -9e-05, 0.00191, 1.269499, 0.000186, 0.000166, 5.946043, 5.945807, 5.9e-05, 0.000896, 0.000137, -3.8e-05, 0.005368, 0.035902, 0.192797, 4.055568]
MLP_SCALE = [0.958608, 0.817782, 0.772926, 0.679613, 0.595651, 2.159911, 3.896705, 0.634826, 8.697445, 0.812542, 0.643248, 0.53615, 1.627199, 1.627058, 1.0, 4.391737, 1.059952, 0.431619, 129.053833, 325.410004, 940.443115, 1.342963]
MLP_LAYERS = [
    ([[0.080357, -0.122567, -0.476799, 0.145257, 0.148375, 0.160456, -0.085945, -0.01824, 0.088775, -0.19474, 0.644604, -0.038174, 0.206754, 0.357946, -0.267891, -0.54826, 0.053998, 0.270478, 0.193454, 0.198089, -0.038722, 0.083849, 0.260309, 0.168045, -0.119679, -0.228917, -0.019837, 0.127669, -0.052105, -0.013825, 0.116105, 0.33081, -0.014447, 0.113136, -0.054395, -0.032498, -0.06118, 0.226592, 0.034782, 0.004912, 0.119369, -0.149875, 0.238196, -0.122571, 0.229483, 0.10922, -0.070333, -0.197458, -0.076467, -0.001528, -0.267085, 0.013975, -0.078953, -0.099952, -0.192322, -0.161045, -0.081714, -0.161852, -0.316394, -0.032478, -0.142635, -0.061391, 0.080286, 0.054033], [-0.163803, -0.073551, 0.291482, -0.189471, 0.08647, -0.131415, 0.489005, 0.146479, 0.280712, -0.262111, 0.013719, -0.296229, -0.010792, -0.206792, 0.007231, -0.184544, 0.044039, -0.130276, 0.008309, 0.00377, 0.095394, -0.16137, -0.004193, -0.427158, 0.056941, 0.369882, -0.08274, -0.009567, -0.305036, -0.139034, -0.236629, -0.198325, 0.195591, -0.363451, -0.037142, -0.269895, 0.082186, -0.234628, 0.190667, 0.276774, -0.019456, 0.053566, 0.040797, 0.00566, -0.189846, 0.171387, 0.070727, 0.310721, 0.130639, -0.108595, 0.157879, 0.116854, 0.143334, 0.189783, 0.170547, 0.195719, 0.088286, -0.088871, 0.112936, 0.034515, -0.028438, 0.056205, -0.350133, -0.205328], [-0.036056, 0.086498, 0.296441, -0.095809, -0.333235, -0.141735, 0.124856, 0.263463, 0.060864, 0.008284, -0.154815, -0.105948, 0.289434, -0.101324, 0.020676, 0.439541, 0.190311, -0.082896, -0.273563, 0.219587, 0.127851, 0.280967, -0.242576, 0.18819, -0.191204, 0.162093, -0.119042, 0.085428, -0.020209, 0.047288, -0.321362, -0.314723, 0.15275, 0.02424, 0.05084, 0.119516, 0.307932, 0.190598, -0.231735, -0.114727, 0.197927, -0.024035, -0.169018, -0.259243, -0.062025, -0.226729, 0.25508, -0.085207, -0.096266, 0.287525, 0.125623, -0.18033, -0.187407, 0.10946, 0.01531, 0.023539, 0.356753, 0.117474, 0.091072, 0.031895, 0.230203, -0.056397, -0.15987, -0.340055], [-0.031854, 0.332974, 0.191192, 0.216998, -0.436301, 0.151347, -0.411407, -0.021384, -0.265203, 0.40461, -0.289337, 0.007137, -0.164282, -0.128603, 0.039519, 0.253215, 0.048038, 0.174777, 0.253636, -0.152579, -0.17817, -0.331449, -0.288021, 0.277592, -0.198321, -0.198986, 0.159457, 0.031756, 0.204373, 0.210557, -0.181078, 0.105887, -0.307292, 0.325872, 0.104781, -0.203227, 0.326919, 0.076191, -0.06355, -0.241832, -0.103436, -0.124815, -0.095831, -0.042411, -0.2142, -0.166422, -0.210581, 0.237196, 0.106443, -0.195179, 0.065259, -0.135327, -0.01371, -0.006106, 0.155082, -0.03474, -0.275115, 0.017899, -0.114751, 0.250005, 0.035645, 0.259151, 0.223464, -0.0992], [-0.059504, -0.052401, 0.153103, -0.036938, 0.259186, -0.188025, -0.5963, 0.150321, 0.214914, 0.231567, -0.010342, -0.036395, -0.345303, 0.29184, 0.455233, -0.060865, -0.131356, 0.241564, -0.168275, 0.391335, 0.474452, 0.208185, 0.160219, 0.374953, -0.218439, 0.396214, -0.192068, -0.139641, 0.155731, -0.259775, 0.393442, 0.218874, 0.086042, 0.044163, -0.005048, -0.119633, 0.145524, -0.082086, -0.173395, -0.246606, -0.398167, 0.047257, 0.266266, 0.427171, 0.10011, 0.193212, -0.16668, -0.328032, 0.031531, -0.289181, -0.382318, -0.396424, 0.204799, -0.377646, 0.04896, -0.401143, -0.386141, -0.016864, 0.142812, -0.107303, -0.134129, -0.059024, 0.190827, 0.195418], [-0.219563, -0.130727, -0.146964, -0.105686, -0.023597, -0.2474, 0.294161, 0.303129, -0.334467, -0.239323, -0.337918, 0.238885, -0.226818, 0.040083, 0.000944, 0.179807, -0.257879, -0.197479, 0.075708, -0.091369, 0.14735, -0.139791, 0.000945, -0.231714, 0.189232, -0.03723, 0.063407, 0.072723, 2.9e-05, 0.272692, -0.00727, -0.378461, -0.143314, 0.009467, -0.028842, 0.109906, -0.205474, -0.187933, 0.054269, 0.11929, -0.16157, 0.14124, -0.168289, 0.041076, 0.13215, 0.03263, 0.050463, -0.190697, 0.174736, 0.233413, 0.1565, 0.086584, 0.286058, 0.242707, -0.149647, -0.055164, 0.283169, 0.147873, -0.002299, 0.162478, 0.016229, -0.110708, 0.094729, -0.120058], [0.020118, 0.547544, 0.493158, -0.255385, -0.527512, -0.08867, -0.137953, -0.152863, 0.289129, 0.285507, -0.26706, 0.549691, -0.546736, -0.156524, -0.063991, 0.151727, 0.02332, 0.195496, -0.340104, 0.565028, 0.165708, -0.003103, -0.573346, -0.170656, 0.136796, -0.234791, -0.351695, -0.147532, -0.295198, -0.124217, 0.325058, -0.103714, -0.25505, -0.157093, -0.405341, 0.081326, 0.085253, 0.260348, 0.250005, -0.375865, -0.155974, 0.533623, 0.104983, -0.149367, 0.648536, -0.019599, 0.025536, 0.172658, -0.008849, -0.327907, 0.139922, -0.029041, -0.184492, -0.212912, 0.515791, 0.226429, -0.077974, -0.229814, 0.51887, -0.020213, 0.226127, -0.030597, 0.04189, 0.407669], [0.17176, 0.138941, -0.011802, 0.113203, -0.145666, -0.101178, 0.312436, -0.065903, 0.060809, -0.291507, 0.126483, 0.047081, 0.001073, -0.003307, -0.066135, 0.154235, -0.167228, 0.051815, 0.072436, -0.077815, 0.261934, 0.022624, 0.029877, 0.07688, 0.099971, 0.146722, 0.242327, 0.138123, -0.113654, 0.182123, 0.219285, 0.321284, -0.14449, -0.051096, -0.315585, -0.190163, -0.342059, 0.084504, -0.210171, 0.068126, -0.018101, 0.053477, -0.060885, -0.045947, 0.143529, -0.082093, -0.00143, -0.029259, 0.170548, -0.164275, -0.178239, -0.030164, 0.064209, -0.119672, 0.173439, 0.048671, -0.022562, 0.2532, 0.102354, -0.066932, -0.01397, 0.03172, -0.16496, -0.271726], [0.113031, 0.363673, 0.132254, -0.321457, -0.360225, -0.032723, -0.296689, 0.462787, 0.109411, -0.21902, -0.622922, 0.253698, -0.539839, 0.220227, -0.380279, 0.485209, 0.080478, -0.423878, -0.702148, -0.13652, -0.42919, 0.284206, -0.259268, 0.205114, 0.297389, -0.015702, 0.875253, 0.287641, -0.376392, 0.498934, -0.419796, 0.333195, 0.238504, -0.201854, 0.379283, 0.342841, 0.01373, 0.104704, -0.096636, -0.281873, -0.227575, -0.238762, -0.5237, -0.891312, -0.072801, 0.005841, -0.517552, 0.26549, 0.374373, 0.151984, 0.260598, 0.190971, 0.772147, 0.054207, 0.029899, 0.219287, 0.290787, 0.367022, 0.959114, -0.294124, -0.021924, -0.538414, 0.498834, -0.389256], [-0.262826, 0.2789, -0.189553, 0.353385, -0.232258, 0.132521, -0.002064, -0.136463, -0.117459, -0.214073, -0.07356, -0.11778, -0.0262, 0.593777, -0.11669, -0.049527, 0.187378, 0.106443, 0.066699, 0.017408, -0.116846, 0.242591, 0.37193, 0.094424, -0.367911, 0.001996, 0.33057, -0.112426, -0.243729, 0.10182, -0.15446, -0.119937, 0.034994, 0.400089, -0.174949, -0.157588, 0.365807, -0.322738, -0.222653, -0.142644, 0.216558, 0.16481, 0.31383, 0.339415, -0.045444, 0.063679, 0.16954, 0.412571, 0.493355, 0.255225, 0.37936, 0.022833, -0.257646, -0.322404, -0.0882, 0.046671, -0.011335, 0.012685, 0.238532, 0.521606, -0.111838, -0.023737, -0.174391, -0.017786], [0.025203, 0.224324, -0.00736, 0.018842, -0.012364, -0.241678, 0.163665, -0.431561, 0.376424, -0.06642, 0.126611, -0.175047, 0.089155, -0.144122, 0.079193, -0.135828, 0.085467, -0.067316, 0.05059, 0.329123, -0.098886, 0.225111, 0.235505, -0.185385, 0.138795, 0.084125, 0.031276, -0.095081, 0.140282, 0.184251, 0.042212, -0.069784, 0.042334, -0.431921, 0.273989, 0.067641, -0.055736, -0.00902, 0.019527, 0.036603, -0.111622, -0.194735, -0.423731, -0.083637, -0.237219, -0.221133, 0.105627, 0.156028, 0.158433, -0.201696, 0.13781, 0.085445, 0.013878, -0.087364, -0.077877, 0.129221, -0.052796, -0.318409, -0.031139, -0.195287, -0.064129, -0.223245, -0.130318, -0.199838], [-0.156248, -0.292279, 0.156578, -0.544502, -0.228795, 0.050249, -0.064762, 0.270903, 0.097343, -0.043957, -0.14396, -0.223026, -0.325234, -0.220718, -0.079714, -0.017421, 0.004087, 0.327924, -0.100746, -0.326167, -0.241896, 0.237214, 0.173819, 0.556518, 0.095324, -0.196406, -0.164722, -0.093291, -0.177125, -0.20609, -0.107509, 0.232384, -0.052676, 0.047602, -0.212903, -0.213052, -0.519967, 0.255803, -0.200902, -0.247046, 0.229926, 0.184933, 0.419298, -0.192787, 0.36803, 0.210633, 0.124679, -0.187838, 0.44085, 0.259725, 0.043911, 0.107079, -0.109692, -0.025581, -0.300782, -0.228629, 0.191109, 0.158349, 0.118957, 0.200106, 0.059242, 0.496623, 0.336152, -0.010424], [0.512525, 0.366434, 0.139521, -0.166592, -0.146053, -0.014053, 0.159504, 0.128197, -0.094149, 0.17216, -0.074047, 0.339564, -0.176859, 0.115698, 0.074846, 0.104818, -0.342312, -0.388065, 0.061886, -0.529662, 0.004118, -0.342431, 0.180816, -0.102186, -0.039783, -0.548555, 0.274972, 0.259959, -0.076698, -0.179731, -0.43694, 0.175889, -0.112565, 0.109866, -0.029534, -0.043069, 0.41791, -0.022527, 0.220711, -0.32076, 0.646613, 0.169796, -0.204528, -0.028357, 0.097034, 0.460482, -0.217467, -0.076157, -0.296249, -0.200441, -0.085818, 0.004735, -0.272115, -0.181465, 0.124648, -0.046374, -0.356433, 0.178721, -0.201659, -0.0522, -0.069161, -0.401104, -0.032727, -0.134815], [0.109394, 0.043817, 0.173589, -0.165457, 0.108498, -0.425266, 0.240455, -0.129258, 0.15119, -0.226562, -0.284426, -0.077663, -0.344053, 0.252355, 0.061949, -0.065181, -0.478478, 0.100421, 0.027986, -0.185917, -0.098488, -0.047117, 0.448843, 0.099088, -0.173657, 0.143497, -0.140129, -0.041821, 0.337609, 0.360833, -0.127766, -0.050355, -0.109105, 0.135031, -0.087026, -0.076815, 0.13614, -0.037883, 0.490922, -0.134504, -0.092383, -0.109371, -0.075778, -0.10966, 0.042438, 0.049178, 0.126467, 0.066859, -0.030749, -0.163969, -0.085992, 0.070881, -0.066619, 0.016264, 0.182519, -0.063956, -0.138304, 0.359979, -0.05663, -0.200387, 0.033377, 0.038272, 0.054975, -0.051744], [0.11138, 0.180395, 0.130015, 0.051343, 0.359352, -0.153773, 0.019899, -0.125337, 0.402806, -0.212102, -0.083764, -0.198622, -0.160485, -0.6539, 0.555602, -0.328671, 0.184208, -0.455323, 0.152635, -0.218373, 0.157408, 0.054825, 0.388267, -0.21127, 0.435986, 0.254998, 0.315381, 0.03952, 0.201497, 0.340309, -0.072516, -0.128323, 0.752262, -0.270585, 0.08205, -0.466701, -0.421842, -0.250882, -0.028536, 0.122915, -0.078919, -0.360837, -0.092301, 0.467709, -0.499767, 0.015826, -0.367618, 0.175802, 0.651096, -0.195677, -0.033248, -0.105339, 0.130459, 0.117239, -0.570778, 0.400282, 0.038613, -0.402298, 0.232972, -0.695735, 0.439982, -0.052761, 0.24092, 0.159874], [-0.126992, 0.067322, 0.1842, -0.053807, -0.047138, -0.160167, -0.317043, -0.019393, -0.497893, 0.474392, 0.142572, -0.476707, 0.154238, 0.050227, 0.44927, -0.107072, 0.375142, -0.281082, 0.020897, 0.212313, 0.217699, -0.237684, 0.129886, 0.190452, 0.084329, 0.251835, 0.074179, 0.446076, -0.135282, -0.031836, -0.164322, -0.18882, -0.471776, 0.208017, 0.326644, -0.155195, 0.489642, 0.307343, -0.123065, 0.456417, 0.052761, -0.385504, 0.026845, 0.156286, -0.155023, 0.091288, -0.269856, -0.336491, -0.475463, 0.449129, -0.037328, -0.447728, -0.278562, 0.124296, 0.008122, -0.071011, -0.010995, -0.052396, -0.188947, 0.503729, -0.649817, -0.196312, 0.268733, 0.333907], [0.219476, 0.559239, 0.190798, -0.376989, 0.381214, -0.226821, 0.301174, -0.377033, 0.090742, 0.478678, 0.456969, -0.330873, 0.159919, -0.289394, -0.485285, -0.495616, -0.370984, 0.103539, -0.116765, 0.109266, -0.611621, -0.029842, 0.20209, 0.049589, 0.255449, -0.140925, -0.438431, -0.698279, 0.467506, -0.207719, -0.129152, -0.401849, -0.389702, 0.251243, 0.325292, 0.038697, 0.082601, 0.233385, -0.294132, -0.587067, -0.180566, -0.395276, -0.270999, 0.080968, 0.293864, -0.210022, -0.898373, 0.075295, -0.273984, -0.508647, 0.038933, -0.411316, -0.104831, 0.672247, 0.228103, -0.176669, -0.130555, 0.478139, -0.455724, 0.494324, -0.010272, 0.169587, -0.541041, 0.145862], [0.4621, 0.333126, -0.52567, -0.117864, 0.216567, 0.262798, -0.193408, 0.521453, -0.442193, 0.499391, 0.428496, 0.102488, -0.025372, 0.263896, 0.094382, 0.308066, 0.497894, 0.410982, 0.191288, -0.166873, -0.268776, 0.33325, -0.130078, -0.129992, -0.479793, -0.335189, -0.246598, -0.073061, -0.133268, -0.235697, 0.788811, 0.15167, -0.074123, 0.277085, 0.401448, -0.174674, 0.037007, 0.208801, 0.228226, -0.292241, -0.199249, 0.050313, -0.306166, -0.124517, 0.092531, -0.118342, 0.357481, -0.260139, 0.341758, 0.244299, -0.365082, 0.404812, -0.070701, -0.217174, 0.129415, -0.054329, -0.524233, 0.101711, 0.172147, 0.161577, 0.069952, 0.019477, -0.723655, 0.147522], [-0.076058, -0.132234, 0.000743, 0.008217, -0.046862, 0.117719, -0.038261, 0.060579, 0.048989, -0.134778, -0.014381, 0.09305, -0.040392, -0.027774, 0.150625, 0.01208, 0.04316, 0.248868, -0.03172, 0.110461, 0.108345, -0.018627, 0.083652, 0.055484, 0.024481, 0.050141, -0.085072, -0.156536, -0.070921, 0.134983, 0.057287, 0.032645, 0.010725, -0.109421, 0.11328, 0.012149, -0.011665, -0.132188, 0.017443, 0.076366, -0.023583, 0.157694, 0.028702, 0.003005, 0.156464, -0.076184, -0.084249, -0.016656, -0.138797, 0.081857, 0.024604, 0.042177, -0.034369, 0.076257, 0.081355, 0.128031, -0.02599, 0.207017, -0.038283, 0.02882, 0.240763, -0.162341, 0.018853, -0.057799], [-0.22696, -0.174228, 0.189895, -0.061452, 0.386738, 0.202132, 0.206252, 0.263854, 0.343789, -0.123575, 0.029805, 0.245924, -0.235768, 0.174976, -0.152011, -0.181647, -0.322597, -0.428959, -0.879325, -0.136047, 0.34418, 0.10647, -0.130054, -0.299732, -0.105151, -0.103011, 0.34921, -0.512781, 0.246947, 0.093022, 0.017174, 0.160188, 0.03093, -0.177528, 0.332288, 0.059518, -0.096361, 0.446994, 0.241475, -0.394792, -0.046801, 0.343943, 0.093591, 0.358856, -0.107486, 0.28195, -0.180055, -0.067629, 0.151506, 0.365661, 0.436207, -0.017992, 0.070963, 0.213849, 0.159045, -0.438841, 0.02625, 0.037621, 0.614253, 0.017847, -0.017892, -0.06044, 0.29173, -0.176497], [0.000595, 0.051513, -0.191164, -0.06462, -0.065434, 0.199997, -0.144059, -0.298104, -0.27733, 0.026222, 0.223598, 0.308284, 0.308858, -0.077919, -0.201425, 0.228837, 0.054028, -0.260758, 0.113352, -0.148736, 0.156073, -0.154408, -0.102872, -0.041755, 0.242537, 0.219684, -0.065591, -0.07392, 0.004629, 0.151637, 0.169311, -0.336933, 0.06152, -0.333869, 0.086098, 0.061418, -0.072199, 0.245025, 0.056804, 0.427602, -0.013763, -0.062832, -0.217901, -0.10725, -0.071225, 0.319652, 0.409481, -0.431967, 0.108679, 0.224859, 0.183181, -0.007117, 0.001885, -0.052819, 0.021548, -0.021005, 0.046671, -0.05843, -0.06274, -0.127205, 0.011915, 0.035849, -0.03632, 0.055576], [-0.147516, 0.094392, -0.145494, -0.09308, -0.038718, -0.089894, -0.079999, -0.043162, -0.001278, 0.035524, 0.019104, -0.281993, -0.197105, 0.055205, 0.125858, -0.129271, -0.011473, 0.342354, -0.137766, 0.028665, -0.066037, 0.492233, 0.104552, -0.041374, 0.267343, 0.050909, 0.004282, 0.174931, 0.205416, -0.316683, -0.091503, -0.049491, -0.232887, -0.14909, -0.110304, -0.18629, 0.139805, -0.095562, -0.068205, -0.027321, 0.023589, 0.139021, 0.118002, 0.177075, -0.20797, -0.016219, -0.343861, -0.004548, -0.083679, 0.216455, 0.110866, -0.048197, 0.367018, 0.642849, 0.319321, 0.014468, 0.032656, -0.118845, -0.152898, -0.127018, 0.557951, 0.208955, -0.345662, -0.186209]], [0.481747, -0.175111, -0.488667, -0.197188, 0.076169, -0.559147, -0.61397, -0.516299, -0.029159, -0.393331, -0.410558, -0.195646, -0.563746, -0.31232, -0.23202, -0.253203, -0.595739, -0.545543, 0.183711, -0.05336, 0.024471, 0.11899, -0.120412, -0.556012, 0.067902, -0.37422, 0.165503, 0.20492, -0.068669, 0.306583, 0.008077, 0.083893, 0.063214, -0.368904, 0.103131, 0.038542, -0.750613, -0.082392, 0.528669, -0.267364, 0.01998, 0.227898, -0.122995, 0.041084, 0.52776, 0.473474, -0.18341, 0.043442, -0.259658, -0.283969, 0.081824, -0.453627, -0.179856, -0.419816, 0.06968, 0.235895, 0.155407, 0.096911, -0.259437, -0.943161, 0.117066, 0.253774, -0.303503, -0.01954]),
    ([[0.082639, -0.011999, -0.065876, 0.135819, 0.102854, -0.214559, -0.175161, 0.280942, 0.070203, -0.06042, -0.085384, 0.251941, -0.065958, 0.219346, -0.298735, 0.010888, -0.174176, -1.380097, -0.359859, 0.314748, -0.173681, 0.130634, 0.217145, -0.341739, -0.107508, -0.381236, -0.098873, -0.072023, -0.250998, -0.150969, -0.160176, -0.103094], [0.330086, 0.164977, 0.241283, 0.055001, 0.406896, 0.206568, -0.691306, -0.054254, 0.024761, 0.00849, 0.135305, -0.043554, 0.021274, 0.108353, -0.267026, -0.078015, 0.283484, 0.140894, -0.019354, -0.014919, 0.198602, 0.33458, -0.520439, 0.079373, 0.10147, 0.340212, 0.293323, 0.217551, 0.226771, -0.109969, -0.092982, 0.105574], [-0.018543, 0.162332, 0.081247, -0.324463, 0.029072, -0.208934, 0.350415, -0.106588, -0.155864, -0.242994, 0.238662, 0.074832, -0.083329, 0.208366, 0.246747, 0.335457, 0.080542, 0.07113, -0.161863, 0.156761, -0.11632, -0.231897, 0.093594, 0.280648, 0.086713, 0.068357, -0.332774, 0.293352, -1.067165, -0.169191, -0.849717, -0.051996], [-0.37819, -0.069216, 0.229193, 0.058328, -0.205021, -0.419535, -0.248148, -0.382141, -0.38693, 0.144402, -0.004953, 0.138978, -0.183356, 0.099464, 0.194138, 0.267558, 0.173813, -0.010865, -0.163912, -0.024131, 0.021804, -0.162384, 0.080526, -0.004868, -0.180184, 0.252695, -0.304674, 0.343599, 0.07321, 0.069742, 0.116428, -0.217069], [0.108961, 0.031177, 0.001627, 0.153139, 0.287729, 0.030503, 0.057413, -0.304428, -0.121658, 0.4354, 0.098331, -0.013292, -0.00651, -0.092859, 0.153707, 0.131726, 0.395676, -0.105213, 0.09259, -0.221228, 0.018255, -0.176755, -0.010113, 0.087598, 0.299325, 0.27292, 0.352532, 0.066645, 0.27719, 0.30937, -0.117965, 0.166641], [-0.53716, -0.151077, -0.41924, -0.165145, -0.028704, -0.839197, 0.062291, 0.243686, -0.174083, 0.078915, -0.248254, -0.000522, -0.07516, 0.05392, 0.099064, -0.201679, 0.088107, -0.051416, -0.259704, -0.02651, -0.222409, 0.045453, 0.070271, -0.01612, 0.401563, -0.337354, 0.271767, 0.210763, 0.103661, -0.014898, 0.028277, -0.080416], [-0.189911, 0.353527, 0.264009, -0.193029, -0.26251, 0.281582, 0.202595, 0.073603, -0.427387, 0.191015, 0.584036, 0.334385, -0.116949, -0.008025, -0.039492, -0.213589, 0.131789, -0.400776, -0.11133, -0.595443, -0.90566, 0.009531, 0.00193, 0.488963, 0.276434, 0.210909, 0.617441, 0.110712, 0.294481, 0.08409, -0.119039, 0.508887], [-0.560532, 0.055476, -0.05581, 0.037511, -0.215416, 0.035638, 0.123348, -0.335313, -0.047403, 0.03307, -0.138461, 0.006885, 0.027388, 0.110963, -0.724904, -0.177207, -0.506981, 0.090028, -0.128452, -0.127921, 0.002328, 0.004122, -0.112563, 0.347216, 0.307661, 0.364953, -0.291149, -0.10962, -0.692511, -0.436432, 0.058675, -0.362785], [0.189348, 0.356235, -0.641493, -0.276587, -0.085727, -0.105734, -0.12274, -0.379176, -0.011632, 0.306588, 0.242707, -0.038854, -0.012898, 0.011984, 0.22385, 0.083857, 0.337952, 0.489079, -0.235971, -0.383091, -0.324534, -0.26671, -0.066716, 0.144016, -0.009941, 0.146788, -0.389511, 0.024469, 0.165417, 0.300967, 0.093921, 0.324217], [0.001165, 0.080085, -0.459608, 0.132955, -0.134443, 0.405428, -0.281885, 0.061216, 0.171808, -0.115701, 0.445842, -0.062746, -0.131802, 0.069426, -0.484465, -0.174188, 0.000927, 0.048886, -0.235656, 0.080515, 0.349136, 0.173589, 0.429874, -0.317883, 0.29949, 0.057354, 0.065752, 0.250702, 0.682583, -0.794534, 0.034406, 0.224389], [0.160371, 0.00017, 0.188789, 0.007766, 0.311581, 0.392332, 0.508437, -0.290775, 0.42306, -0.201409, 0.239906, 0.103831, -0.028501, 0.140952, 0.094109, -0.202844, -0.160315, 0.327153, -0.403989, 0.551954, -0.167231, 0.147677, 0.115852, 0.124519, 0.302235, -0.119789, 0.039018, 0.383161, 0.289984, -0.820969, -0.241584, 0.353981], [0.275692, 0.176964, 0.191835, 0.103744, -0.340751, -0.211474, -0.318471, -0.028495, -0.367755, 0.296333, 0.087215, -0.198345, -0.184698, -0.14669, -0.095223, -0.165054, 0.225968, 0.241246, -0.190654, -0.070602, -0.191077, 0.00585, 0.077716, 0.166555, -0.043832, -0.866429, 0.343649, -0.398487, -0.225257, -0.078672, 0.217986, 0.111107], [-0.234569, 0.06712, 0.260107, -0.403609, 0.284471, -1.438717, -0.114023, -0.024211, -0.729381, 0.227153, 0.046136, 0.122513, 0.349463, 0.143749, 0.031276, 0.11932, -0.225649, -1.255648, 0.093963, 0.244952, -0.136475, -0.257117, 0.050939, -0.281155, 0.253645, -0.250235, -0.126949, -0.278491, 0.210233, -0.19694, -0.001354, 0.136223], [-0.261907, 0.148217, 0.334373, 0.025037, -0.010758, 0.26123, -0.004726, -0.015929, 0.079733, 0.270526, 0.06927, 0.612108, 0.078599, 0.152123, 0.223043, 0.04008, -0.007066, 0.045314, 0.421409, 0.014613, 0.449451, 0.061227, -0.091593, 0.05005, -0.425441, -0.168354, -0.042464, 0.272057, 0.247453, 0.315401, -0.209395, 0.100562], [-0.201727, 0.304523, 0.194958, -0.426533, 0.136362, -0.043826, 0.18245, -0.085835, 0.202091, -0.087688, -0.164044, -0.333026, 0.248234, -0.186576, -0.128806, 0.342499, -0.326198, 0.072887, -0.148549, -0.456327, -0.22648, 0.310676, -0.265829, 0.141432, -0.129021, -0.284968, -0.055382, -0.069615, 0.152869, 0.130125, 0.126585, -0.131385], [-0.133043, -0.062868, -0.043146, 0.035504, 0.277393, 0.383289, 0.014474, -0.294373, -0.131502, 0.095306, 0.339708, -0.232575, -0.093902, 0.007024, -0.008524, -0.101177, 0.519151, 0.046813, 0.167207, -0.466324, 0.026944, -0.262805, -0.24854, 0.169726, 0.23206, 0.288563, 0.608454, 0.099169, -0.037174, 0.010829, 0.63415, 0.031583], [0.120206, 0.229478, -0.190983, -0.031036, 0.06835, 0.147586, -0.414542, -0.121122, -0.047771, 0.032123, -0.253806, 0.100333, 0.200918, -0.019972, 0.084243, 0.119667, 0.343621, -0.337237, -0.032528, -0.267407, 0.008444, 0.080229, 0.291458, -0.446224, -0.210505, 0.423225, 0.136952, -0.12985, -0.066158, -0.465492, -0.17604, -0.138094], [-0.063913, -0.206455, 0.031414, 0.474317, -0.199657, 0.159898, -0.310383, -0.050253, 0.308533, -0.079549, -0.281485, -0.57368, 0.048728, -0.011776, -0.033276, 0.296596, -0.134961, 0.041798, -0.200436, 0.340274, 0.277936, 0.107898, 0.455788, -0.064286, -0.340929, 0.167046, -0.026325, -0.249837, -0.353169, -0.004336, -0.160118, 0.19908], [0.169928, 0.065499, 0.301702, -0.48601, 0.088083, -0.222067, -0.0044, -0.163626, -0.417074, 0.045197, 0.006824, -0.374311, -0.551387, -0.135594, 0.055892, 0.102068, 0.268055, -0.404995, 0.225499, -0.139888, 0.101211, -0.112522, 0.162334, -0.030028, -0.289346, -0.714841, -0.207255, 0.040838, -0.114886, -0.139628, 0.098791, 0.388075], [0.125, 0.013488, 0.290408, 0.182115, 0.186767, 0.132942, 0.400643, -0.07886, 0.172125, -0.187618, -0.008156, 0.071291, 0.057642, 0.09615, 0.104071, 0.000295, 0.131733, -0.196836, 0.305743, -0.10102, -0.247443, 0.01335, 0.196291, -0.056719, 0.252018, -0.040181, 0.135492, 0.345836, -0.079591, -0.077725, -0.299612, -0.401176], [-0.078328, -0.335476, 0.242249, 0.046204, 0.122586, 0.179076, -0.289989, -0.0618, 0.246354, -0.070505, -0.220633, -0.129401, 0.320029, 0.065044, -0.20311, 0.318452, -0.024478, -0.064653, -0.306385, 0.018202, -0.129668, 0.202156, -0.129414, -0.307438, 0.191351, 0.007894, 0.144354, -0.019482, 0.265269, -0.171338, -0.222126, -0.056599], [0.337102, -0.395815, 0.017432, 0.013129, 0.230023, 0.050567, -0.048305, 0.195777, -0.174888, 0.163841, -0.179831, -0.20045, 0.34494, -0.020247, -0.048559, -0.133675, -0.044307, 0.2941, 0.075709, 0.305178, 0.298907, 0.363878, -0.156877, -0.275895, 0.068747, 0.045406, 0.049027, -0.029089, -0.100257, 0.000952, -0.57376, -0.143562], [0.165323, 0.166899, 0.201328, -0.000968, 0.406025, -0.632257, 0.018072, 0.025073, 0.292103, -0.05226, -0.221803, 0.106932, -0.129696, -0.131336, 0.456654, -0.212944, 0.30041, 0.376008, -0.123562, 0.103088, 0.322544, 0.321501, -0.505683, -0.217803, -0.197614, 0.104577, -0.374981, -0.157353, 0.025406, -0.101067, 0.099311, -0.029263], [0.212558, 0.189822, 0.16978, 0.099862, -0.317853, 0.180345, 0.603148, -0.118356, -0.253578, -0.180377, 0.349063, 0.221839, 0.136033, 0.191131, -0.057081, 0.44341, 0.420982, -0.351653, 0.120937, -0.189698, 0.484056, 0.344201, 0.058317, 0.216605, 0.376269, 0.45976, 0.027665, 0.169071, 0.400281, 0.458441, 0.119855, -0.13406], [-0.120924, -0.194954, -0.089716, -0.083872, -0.200953, -0.196841, -0.23033, 0.15085, 0.096016, -0.392579, 0.137422, -0.30649, 0.114116, -0.511788, -0.118797, 0.296753, -0.161614, 0.141016, 0.122292, 0.237324, 0.208842, -0.027164, 0.128185, -0.046361, -0.25707, -0.025909, -0.096901, 0.387294, -0.178431, 0.041634, -0.51918, 0.109644], [0.234885, -0.284769, -0.00754, 0.101551, -0.1776, -0.082755, -0.106474, 0.253931, -0.014165, 0.000978, 0.034422, 0.023213, -0.075038, 0.067289, -0.197705, 0.215879, -0.283949, 0.206709, -0.595525, 0.105917, -0.087706, 0.223433, -0.517755, 0.134907, -0.276748, -0.048282, -0.011808, 0.416046, 0.048505, 0.289726, -0.602489, 0.21404], [-0.35034, -0.211336, 0.146749, -0.105212, 0.197415, 0.166504, 0.084536, 0.130274, -0.119284, -0.263995, 0.198128, -0.038465, 0.043615, -0.092488, 0.136067, -0.176374, 0.20474, -0.274939, 0.31786, 0.398663, -0.240099, 0.048554, -0.092904, 0.113558, 0.072791, -0.287402, -0.023347, -0.376947, -0.034946, -0.104067, 0.276438, -0.26643], [0.2098, -0.239675, -0.243215, -0.016534, -0.187895, -0.057941, 0.292165, -0.004511, 0.157858, 0.246613, -0.167863, -0.596152, 0.121233, -0.491202, -0.253741, -0.154388, 0.006431, -0.416747, -0.001662, -0.086988, 0.314194, 0.261215, -0.312404, 0.063551, -0.044933, 0.01516, 0.054122, -0.031574, -0.260321, -0.001799, -0.214733, -0.152985], [-0.05758, 2.7e-05, 0.145598, -0.324421, -0.045258, -0.471227, -1.168385, 0.235511, 0.091699, -0.125773, -0.05004, 0.099821, -0.513358, 0.039706, -0.229546, -0.147578, -0.135354, 0.192212, -0.051855, 0.050937, -0.178267, 0.14166, 0.089139, -0.064651, -0.139606, -0.03839, -0.150017, -0.075688, -0.270601, -0.479883, -0.206391, -0.051927], [0.164823, 0.146139, -0.209628, -0.026305, -0.05112, 0.278393, -0.084329, 0.106617, -0.132597, -0.135904, 0.09309, 0.239944, 0.214841, -0.468804, 0.026386, -0.410794, -0.157874, -0.123892, 0.076339, -0.22258, 0.137063, 0.139571, -0.303401, 0.251431, -0.332606, -0.001276, -0.20356, 0.041387, -0.283156, 0.177326, 0.207507, -0.074738], [0.12517, 0.289417, -0.046866, 0.300039, -0.065826, -0.043751, 0.033807, -0.11403, -0.199072, 0.100914, 0.087566, 0.076893, -0.155404, 0.528132, 0.013494, -0.203556, 0.118589, -0.342389, -0.067893, 0.0395, 0.255316, -0.15765, 0.24517, 0.117543, 0.403568, -0.472356, -0.283759, 0.024933, 0.12432, 0.030642, -0.100495, 0.230021], [-0.034016, -0.028815, -0.955819, 0.041847, 0.487021, 0.22326, -0.363872, -0.174437, 0.255045, 0.004415, -0.305401, 0.182259, -0.456058, -0.109023, -0.455579, -0.193265, -0.148427, -0.182175, 0.056444, 0.300928, -0.058834, -0.316709, 0.125699, 0.219666, -0.165536, -0.262073, 0.092955, 0.02621, 0.049247, -0.162075, 0.321943, 0.187144], [-0.00203, 0.44883, -0.090113, 0.276586, 0.022805, -0.06152, 0.194499, 0.183022, 0.165287, 0.056565, -0.312972, -0.395156, -0.118867, 0.213996, 0.069282, 0.268218, -0.180869, 0.145491, 0.234029, -0.159176, 0.01269, -0.001783, -0.241349, 0.038663, -0.073254, -0.029397, -0.082286, 0.302022, 0.084709, -0.204469, 0.340091, -0.239155], [0.190498, -0.530737, -0.524283, -0.177544, 0.046907, 0.275636, -0.075004, -0.07857, 0.137371, -0.03763, -0.052655, -0.354813, 0.419452, 0.200842, -0.37679, 0.27038, -0.08393, -0.10145, -0.058791, 0.408533, -0.652679, -0.092752, 0.027288, 0.282725, -0.016241, 0.312021, -0.145469, 0.196087, 0.269657, 0.068615, -0.059261, -0.217545], [0.079277, 0.109592, -0.223301, -0.05391, -0.060128, -0.145353, -0.089078, -0.075593, -0.11204, 0.206928, -0.019785, 0.193512, -0.012752, -0.357933, 0.291746, -0.095525, 0.19198, -0.308402, -0.060298, 0.149238, -0.111001, 0.031157, 0.164994, 0.298063, 0.11736, 0.12085, -0.385858, -0.282749, -0.079736, 0.031369, -0.238555, -0.050374], [-0.319562, 0.07878, -0.740009, 0.135108, -0.001983, -0.007435, 0.171999, 0.330142, 0.022139, 0.172013, 0.035587, 0.272043, -0.350729, 0.154033, 0.099844, 0.269897, -0.157511, -0.38922, 0.436589, -0.006327, 0.079805, -0.258092, 0.16871, 0.074931, -0.030102, -0.064319, 0.179047, -0.129595, -0.131566, 0.022212, 0.074927, 0.18157], [-0.0283, 0.133808, 0.1774, -0.019875, -0.077295, -0.058172, -0.189505, -0.338798, 0.140292, -0.203582, 0.053549, 0.068286, -0.283155, -0.215472, -0.025231, -0.570343, -0.044773, 0.508536, -0.041118, -0.057483, 0.291677, -0.08374, 0.447661, 0.11559, 0.266898, -0.396059, 0.485175, 0.199743, 0.285368, -0.445653, 0.394147, 0.394059], [0.017226, 0.172542, 0.022718, -0.00358, -0.120246, -0.147434, -0.369969, 0.030234, -0.122599, -0.154091, 0.155475, -0.078217, 0.016352, -0.042117, 0.316019, 0.072354, -0.225012, 0.1815, 0.118928, 0.14804, 0.372318, -0.047067, 0.270061, 0.039148, 0.125392, 0.312913, -0.182081, -0.383675, 0.29302, 0.252583, -1.359428, 0.036508], [0.242315, 0.145211, -0.208846, -0.017015, -0.062803, -0.218365, -0.176624, -0.051562, -0.192743, 0.015335, 0.149331, -0.532864, -0.50528, 0.089769, -0.021101, 0.183596, 0.246017, 0.017287, -0.445607, 0.205443, 0.095978, 0.174551, 0.090003, 0.194517, -0.3219, -0.153509, -0.189803, 0.077653, 0.106845, -0.016075, -0.15469, -0.028459], [0.354059, 0.016709, 0.203756, -0.142474, 0.214479, -0.206406, -0.310712, 0.197854, 0.112784, -0.12054, -0.009054, -0.234283, 0.117787, -0.25935, 0.081868, 0.065738, -0.12831, 0.284773, -0.013399, 0.503181, -0.28725, -0.057529, -0.060766, -0.005594, -0.296632, 0.256133, 0.37334, 0.084637, -0.278362, 0.409637, -0.041553, -0.381841], [-0.198448, -0.109387, 0.137416, -0.015046, -0.284062, -0.846944, 0.12728, 0.246193, -0.092025, 0.021323, -0.086694, -0.152073, 0.469296, 0.035499, -0.121887, 0.054988, -0.134105, -0.343269, 0.134612, -0.238176, 0.036578, -0.250878, -0.252208, -0.025659, 0.159815, 0.225645, -0.71388, -0.102296, -0.295953, -0.232169, -0.006994, -0.016841], [0.175689, -0.048438, -0.150285, 0.074338, 0.012156, -0.11563, 0.313325, -0.024098, 0.140953, -0.390022, -0.326728, 0.356976, -0.294917, 0.143636, 0.313113, -0.138596, 0.099039, -0.155294, -0.434053, -0.102956, 0.103616, 0.267571, -0.082193, 0.116837, -0.169083, -0.356215, -0.2531, 0.095245, 0.146705, 0.0914, 0.124026, -0.075047], [0.186224, -0.038686, 0.026848, -0.017264, 0.205992, 0.428766, 0.382262, 0.164971, 0.100855, -0.272797, -0.031872, 0.050042, 0.185062, 0.252952, 0.006282, -0.229184, -0.04235, 0.217125, -0.614904, -0.0421, 0.085382, -0.103248, -0.194186, -0.329064, 0.033171, 0.32898, 0.109187, -0.144694, -0.008135, 0.10748, 0.306327, 0.235207], [0.075009, -0.053841, -0.227156, 0.203464, -0.367009, 0.073738, -0.152558, -0.047708, 0.091807, 0.211129, 0.175918, 0.246456, -0.127576, 0.219874, 0.265483, -0.330079, -0.028618, 0.378103, -0.119895, 0.136004, 0.011116, -0.109707, -0.053429, 0.233255, -0.14964, -0.45196, -0.043965, 0.019599, -0.403114, -0.084072, 0.100822, -0.073218], [0.046798, -0.178907, 0.426201, 0.115369, -0.261282, 0.118458, -0.002543, 0.369397, 0.09566, -0.401387, 0.046647, -0.252308, 0.152163, 0.051724, 0.055605, 0.312883, 0.111889, -0.219831, 0.346633, 0.267839, -0.29951, -0.177519, 0.014277, -0.143594, 0.266845, -0.254458, 0.165559, -0.053213, -0.00587, -0.044549, -0.395885, -0.382083], [0.048373, -0.134632, -0.196885, -0.207763, 0.228516, -0.016356, -0.210826, 0.191249, 0.146593, 0.3345, 0.288165, -0.44747, -0.175531, -0.353957, -0.219619, -0.057252, -0.198017, -0.079038, 0.167475, 0.215272, -0.347476, 0.046265, -0.336648, 0.118789, 0.219876, 0.071038, -0.764304, 0.160713, 0.221663, 0.04334, 0.26634, -0.069346], [0.296297, 0.347986, -0.003956, -0.15615, -0.395809, -0.079819, 0.33284, -0.123784, -0.065062, -0.32201, -0.642521, 0.180744, -0.151498, -0.338633, 0.280046, -0.078859, 0.052399, 0.187207, -0.161255, 0.198741, -0.292644, -0.348033, -0.316883, 0.229057, -0.24542, 0.110065, -0.363789, 0.083844, -0.047629, -0.159578, -0.308356, -0.249943], [-0.252214, 0.044519, -0.204894, 0.356965, -0.090639, -0.840563, 0.116065, 0.083363, -0.2217, 0.159651, 0.029873, -0.54515, 0.078819, 0.108669, -0.325078, -0.037509, -0.158748, -0.130563, -0.38734, -0.038811, 0.112781, 0.142623, 0.08366, 0.158471, 0.218729, 0.282042, 0.188706, 0.252375, 0.220858, 0.003847, 0.162106, -0.140299], [-0.356637, 0.169789, -0.063089, 0.310388, 0.11948, -0.061763, -0.028549, -0.160466, -0.11726, 0.524693, 0.267011, 0.063042, 0.11429, -0.037788, 0.14721, 0.148117, 0.106466, 0.049041, -0.038737, -0.221008, 0.221425, 0.246006, -0.470948, 0.250024, -0.26288, -0.084327, 0.045479, 0.107361, -0.501371, -0.193842, 0.104043, 0.113835], [0.305019, -0.357148, 0.345857, -0.289536, -0.083843, 0.06383, -0.53125, -0.012243, -0.464006, -0.065477, -0.056975, -0.207829, 0.296421, -0.106021, -0.059204, -0.105712, 0.045589, 0.193496, 0.236772, 0.0912, 0.19399, 0.164102, 0.146477, -0.206365, 0.208203, -0.06212, 0.168038, -0.440128, 0.089543, -0.274989, -0.126891, 0.070358], [-0.126329, -0.101477, 0.10231, 0.061532, -0.11902, 0.055265, -0.130799, 0.465273, 0.24952, -0.202368, 0.190908, 0.095506, -0.065558, -0.290639, -0.380643, -0.026572, 0.275686, -0.067011, 0.266796, 0.143457, 0.001664, 0.312392, 0.34596, 0.122106, 0.170721, -0.035122, 0.012954, -0.265373, -0.515085, 0.053393, -0.229688, 0.393106], [0.003365, -0.00304, 0.347255, 0.019726, 0.08091, -0.588123, -0.081349, 0.069625, 0.185179, 0.377754, 0.161393, -0.140223, -0.454086, 0.103203, 0.268014, -0.242895, 0.107425, 0.428001, -0.116402, -0.021691, -0.032993, -0.02934, -0.07031, -0.012656, 0.061489, -0.210527, -0.049668, -0.112469, 0.198525, 0.302923, 0.001136, 0.305308], [0.257587, 0.035949, 0.232355, -0.159751, -0.239883, 0.099974, -0.551185, -0.290351, -0.035207, 0.143931, -0.099182, 0.059416, 0.223976, -0.099258, -0.201621, 0.18938, -0.097241, -0.03231, 0.160271, 0.035532, -0.105647, 0.177559, 0.01256, 0.110848, -0.196205, -0.138583, -0.228881, -0.350237, -0.385606, -0.236251, -0.016633, 0.566738], [-0.133495, -0.164254, 0.397485, -0.198507, 0.100159, -0.213685, 0.077758, -0.083984, -0.20012, 0.084455, -0.068249, -0.148073, -0.229326, -0.145265, -0.319246, 0.293009, 0.174731, -0.496836, -0.94517, 0.129327, -0.323298, -0.900432, 0.108744, -0.124379, -0.106804, -0.055756, -0.073996, 0.145572, 0.174401, -0.287123, 0.294879, 0.029923], [0.231447, -0.075177, 0.093434, -0.13275, -0.105746, -0.379024, -0.082135, 0.333846, -0.116766, -0.039837, -0.247777, 0.072654, -0.077144, -0.214708, -0.15667, 0.030143, 0.325829, 0.147621, 0.126, -0.389745, -0.571258, 0.03646, -0.003349, -0.478189, -0.130943, -0.162416, -0.195556, 0.025376, 0.033147, -0.004827, -0.211493, -0.146333], [0.076579, 0.314875, -0.277256, -0.100279, 0.142865, 0.235295, 0.242665, 0.041387, -0.294408, -0.207815, 0.182681, -0.146641, -0.047824, -0.040665, -0.015415, 0.079815, 0.250031, -0.180106, 0.035023, -0.509076, 0.22823, -0.039991, 0.189116, -0.128951, 0.085896, 0.097958, 0.061331, -0.101495, 0.257127, 0.187278, 0.309421, 0.398086], [-0.22429, 0.22076, -0.344476, -0.254631, 0.295598, 0.311031, -0.042623, -0.009787, 0.272008, 0.220421, 0.054386, 0.276512, 0.238519, 0.097503, 0.054393, 0.062319, -0.365363, -0.037108, -0.2967, -0.226746, -0.293802, -0.097358, 0.126319, 0.085026, 0.273395, 0.215294, 0.106595, -0.232786, 0.094806, 0.313222, 0.338755, 0.054386], [-0.056361, 0.176945, -0.318809, 0.136584, -0.234342, -0.080768, -0.091994, 0.341435, 0.077317, 0.031209, 0.216243, 0.002029, 0.043116, -0.313065, 0.312932, -0.317803, 0.331028, 0.265175, 0.116739, -0.007542, 0.032543, -0.241262, -0.235815, 0.016349, -0.00493, 0.281722, 0.0357, -0.046068, -0.194933, 0.115382, -0.169542, -0.029201], [0.062061, 0.430119, 0.321112, 0.282182, -0.152448, 0.096185, -0.288023, 0.229406, 0.410912, 0.173517, -0.027356, -0.166993, 0.21373, 0.029543, -0.229702, -0.132708, 0.033596, 0.254447, 0.30542, -0.134679, 0.031694, 0.033935, -0.188625, -0.047984, 0.223652, -0.179036, -0.938431, -0.288136, 0.080952, -0.117548, -0.181804, -0.217856], [0.340489, -0.098629, 0.401913, 0.129591, -0.152289, -0.372955, -0.298438, 0.093511, -0.362073, 0.287633, -0.595491, 0.450015, -0.475124, -0.442348, 0.553642, 0.067358, 0.270262, -0.583044, 0.375179, -0.341633, -0.220878, 0.519997, -0.306435, -0.281521, 0.041742, -0.024751, -0.46695, -0.201169, 0.329138, 0.071219, -0.257673, -0.15503], [-0.202151, 0.030695, -0.04491, 0.225119, 0.152429, -0.129952, 0.003044, 0.014629, -0.05954, 0.070786, -0.272448, -0.219939, 0.1508, -0.083649, 0.426677, -0.470929, -0.356715, -0.215584, 0.036863, -0.001103, 0.312968, -0.303132, -0.197391, -0.290126, 0.267289, 0.258752, -0.071696, -0.561169, 0.215853, 0.489737, 0.218832, -0.04874], [-0.352674, -0.064812, 0.098223, 0.086037, 0.187954, -0.279618, -0.257302, -0.101662, -0.015171, -0.162061, 0.140471, 0.137117, 0.067871, 0.12533, -0.177165, 0.071438, 0.029207, -1.034268, 0.047417, -0.316836, -0.184694, 0.189609, 0.080189, -0.236668, 0.009366, -0.143824, 0.221258, -0.180481, 0.004666, -0.163888, 0.128278, -0.15195], [-0.132783, 0.057194, -0.249576, 0.087752, -0.066349, -0.026882, -0.194163, 0.348727, -0.064103, 0.03384, 0.315545, 0.202567, -0.545364, 0.058321, -0.153256, 0.133068, 0.171463, -0.23483, -0.486533, -5.5e-05, 0.239896, 0.27109, 0.102504, -0.068483, 0.008177, 0.049417, 0.613324, -0.109838, 0.238238, 0.18636, -0.22536, -0.499539], [0.075331, 0.054697, 0.093631, 0.220296, 0.180824, -0.330524, -0.024957, 0.122492, 0.139875, -0.244927, -0.163665, -0.247349, 0.430705, 0.023164, 0.126199, -0.200303, 0.237816, 0.031783, 0.180532, 0.062445, 0.11355, -0.116011, 0.278717, -0.22711, 0.224436, -0.022643, -0.129072, 0.093697, -0.091542, -0.016169, 0.175439, 0.197757]], [0.079467, -0.304166, -0.06097, 0.173663, -0.150363, 0.167679, 0.025559, 0.138413, 0.097978, -0.39019, -0.055792, -0.183848, 0.050629, 0.326021, -0.269579, -0.038994, 0.192596, -0.455742, -0.160565, 0.048632, -0.091873, -0.053434, -0.201091, 0.230257, -0.149661, -0.295653, 0.150355, 0.023134, 0.013267, -0.128236, -0.226314, -0.24243]),
    ([[0.349153], [-0.366179], [-0.501398], [0.291375], [0.395809], [0.317971], [-0.705896], [0.142028], [0.204235], [-0.582977], [0.365152], [0.845254], [0.233932], [-0.418229], [0.444544], [0.429756], [-0.318567], [1.270515], [0.703626], [-0.411677], [0.47394], [-0.297165], [-0.458939], [0.345101], [0.346086], [0.564022], [-0.955919], [-0.315662], [0.61402], [-0.466529], [-0.640288], [-0.541408]], [-0.179815]),
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
        if self.mlp_on:
            self.mlp_mean = np.array(MLP_MEAN)
            self.mlp_scale = np.array(MLP_SCALE)
            self.mlp_layers = [(np.array(w), np.array(b)) for w, b in MLP_LAYERS]
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
            h = (np.array((pg[0], pg[1], f70, pg[3], pg[4], f70 * log1p(fw), f70 * af, sg100 * af, f70 * spread, spread,
                           imb1, cnt1, log1p(qb), log1p(qa), ltr_sign, ltr_lsz, (ltp - mid2 * 0.5) if ltp > 0 else 0.0,
                           (mid2 - rm2) * 0.5, self.pr_ev10, s100, self.pr_s1000,
                           log1p((now - self.pr_ttr) / 1000.0) if self.pr_ttr >= 0 else 10.0)) - self.mlp_mean) / self.mlp_scale
            h = np.clip(h, -8.0, 8.0)
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
