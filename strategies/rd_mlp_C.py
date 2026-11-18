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
    ([[-0.123187, -0.08257, -0.243233, 0.12784, 0.220133, 0.192647, -0.108078, 0.060527, 0.296485, -0.034311, 0.020509, 0.0771, -0.023871, 0.208447, -0.135548, -0.18924, 0.042695, 0.118714, 0.227257, 0.055697, -0.075305, -0.226905, 0.044711, -0.065326, 0.037806, 0.06456, -0.052514, 0.289618, 0.038138, -0.008701, -0.00495, 0.161755, 0.090572, 0.044012, -0.051333, -0.012894, -0.170036, -0.046984, 0.267311, 0.058619, -0.044745, 0.089378, -0.034484, -0.084189, 0.098386, 0.192301, 0.022522, 0.051039, 0.100952, -0.025926, -0.067483, 0.178629, 0.145502, 0.046818, -0.17976, -0.115745, 0.017086, -0.016048, 0.030148, -0.163093, -0.066851, 0.056975, 0.072822, -0.218533, 0.073775, 0.126849, 0.066153, -0.043054, 0.560807, -0.137646, 0.208858, -0.195467, 0.150151, -0.006913, 0.116135, -0.177329, 0.084532, 0.017925, -0.249539, -0.122647, -0.016399, -0.223049, -0.098142, 0.011893, 0.147219, -0.063162, 0.135355, -0.233833, -0.091887, 0.212293, -0.06024, 0.001445, -0.279776, -0.017907, -0.137152, -0.053971, 0.091357, -0.18599, 0.119919, -0.143743, 0.023415, -0.142369, 0.00425, 0.218799, -0.404001, 0.166441, 0.052983, 0.001438, -0.042657, 0.141756, 0.030223, 0.213121, -0.321652, -0.030302, -0.012807, 0.190797, 0.111725, -0.02509, -0.008087, 0.003595, -0.018767, -0.073658, 0.053505, -0.006355, -0.177813, 0.073103, -0.181986, -0.127586], [-0.023803, -0.076206, -0.072726, -0.059384, -0.245049, -0.000546, 0.197746, -0.041891, 0.116451, 0.084372, 0.240428, -0.165885, 0.241721, -0.132157, -0.056312, 0.18084, 0.145271, 0.0687, -0.234624, 0.204364, 0.184834, 0.1687, -0.144991, 0.173042, -0.190543, 0.103047, -0.176784, 0.140288, 0.189379, 0.048668, -0.034072, -0.211286, -0.129433, 0.18307, 0.115962, 0.220186, 0.129702, 0.081738, -0.066656, 0.152794, 0.134976, -0.125837, -0.04, -0.220194, -0.110232, -0.040986, 0.11746, -0.111361, 0.048149, 0.312192, -0.029338, -0.121843, -0.098015, 0.016669, -0.189232, -0.156282, 0.143504, 0.070633, -0.131635, 0.068241, 0.070074, -0.04834, -0.003671, -0.209778, -0.059036, 0.199321, 0.229578, -0.005702, -0.18138, 0.015234, -0.101464, 0.197454, -0.068891, 0.245423, 0.023123, -0.153149, -0.144759, -0.3089, 0.187231, -0.1097, 0.070182, 0.123563, 0.319745, -0.144997, 0.127623, -0.117465, -0.25684, 0.088684, -0.039082, -0.112473, -0.147896, 0.180664, 0.034884, -0.048315, -0.003595, 0.284616, 0.039542, 0.338494, 0.222868, -0.169378, 0.188969, 0.046638, -0.14921, -0.242089, -0.054252, -0.107396, -0.164959, -0.014267, -0.109742, 0.096156, -0.117156, -0.022401, 0.309941, -0.268011, -0.048516, -0.238692, 0.137643, 0.110798, 0.181952, -0.118005, -0.10532, -0.150728, -0.174815, 0.032076, 0.286473, 0.093228, 0.101438, -0.15072], [-0.018291, 0.017401, 0.240956, -0.152373, -0.292281, -0.210753, -0.184207, 0.048733, -0.05164, 0.125646, 0.027227, 0.040368, -0.183311, -0.247498, 0.193474, 0.240675, -0.032675, 0.120131, -0.220823, 0.261574, 0.347807, 0.148105, 0.146056, 0.236076, -0.131176, 0.152126, 0.124793, -0.337032, 0.022661, -0.170335, 0.330144, -0.002902, -0.198122, 0.129192, -0.160941, -0.196225, 0.036917, -0.07591, 0.05797, -0.146942, -0.150802, 0.077163, 0.287418, 0.17066, 0.259536, 0.057433, -0.238728, -0.247861, -0.05411, -0.160849, -0.215219, -0.281791, 0.102882, -0.360376, 0.175595, -0.082361, -0.124195, -0.015187, 0.041053, -0.082934, -0.009759, 0.094672, 0.164272, 0.130964, -0.110168, -0.241412, -0.177406, -0.122214, -0.301783, -0.032541, 0.037785, 0.302838, -0.108884, 0.002636, -0.170644, 0.122542, -0.196307, 0.084143, 0.295194, 0.08992, -0.046653, 0.073174, 0.077133, -0.022234, -0.000261, -0.168234, -0.052507, -0.162228, 0.073404, -0.228289, -0.118027, -0.064029, 0.008435, 0.106168, 0.219502, -0.219753, -0.008732, 0.106211, 0.040437, 0.047015, -0.079332, -0.210867, -0.010591, 0.091527, -0.066516, 0.272651, 0.167766, -0.106269, 0.109769, 0.079391, 0.172445, -0.274514, 0.379678, 0.123475, 0.208083, -0.11052, 0.064258, 0.175694, -0.126949, -0.06119, 0.231513, 0.168284, -0.178367, -0.041257, 0.337964, -0.079082, 0.155554, 0.124046], [0.225286, 0.128286, 0.362037, -0.147412, 0.07028, -0.100031, -0.08556, 0.016209, -0.333001, 0.323688, -0.103792, 0.281575, 0.092549, -0.149782, -0.194123, -0.060822, -0.028585, 0.192872, -0.290448, 0.049047, 0.263634, 0.301302, -0.117826, -0.064079, 0.037698, -0.127106, -0.241935, -0.200796, -0.291567, -0.066538, -0.064527, -0.169623, -0.209994, -0.42496, -0.167263, -0.075633, 0.231358, 0.177618, -0.071471, -0.12284, -0.193788, 0.041932, 0.190089, -0.090305, 0.175669, -0.145873, -0.284564, -0.073418, -0.295888, 0.131162, -0.125979, -0.002671, -0.045714, -0.058907, 0.508252, 0.007966, 0.04049, 0.14866, 0.256266, -0.180674, -0.208008, -0.135373, 0.064383, 0.576349, 0.305845, 0.012897, -0.132177, 0.356293, -0.120627, -0.111343, 0.19639, -0.114151, 0.19788, -0.341173, 0.099832, 0.522811, 0.178286, 0.04735, -0.032451, 0.033659, -0.066507, -0.089493, -0.214906, -0.178949, 0.095227, -0.34466, 0.005961, 0.293918, 0.168975, 0.00151, 0.507581, 0.048913, 0.266448, 0.356554, 0.15869, -0.112383, -0.213784, -0.047664, -0.247556, -0.057194, -0.207514, -0.033902, 0.045828, 0.23786, 0.19086, 0.208246, -0.023999, 0.107787, 0.009519, -0.060929, 0.14237, -0.182165, -0.089626, 0.166599, -0.15876, -0.030523, -0.157676, -0.155084, -0.087057, 0.267448, -0.233883, 0.07097, 0.218933, -0.10695, -0.280986, -0.169625, -0.239807, 0.14022], [0.144643, -0.100505, -0.00909, 0.35991, 0.016063, 0.073913, 0.061879, 0.421124, 0.092113, -0.106414, -0.246872, 0.06473, -0.255552, 0.108896, 0.03713, 0.388582, -0.17571, 0.150875, -0.274546, -0.281705, 0.302613, -0.298338, -0.186845, -0.316578, 0.03707, 0.260279, 0.123307, 0.492214, -0.361383, -0.032435, -0.075698, -0.096913, 0.66129, -0.622143, 0.166251, 0.214664, -0.054266, -0.021803, 0.051733, -0.003077, 0.049907, 0.184544, 0.042606, -0.290338, -0.294786, 0.145937, 0.540835, 0.386602, -0.157114, -0.106745, 0.368543, 0.201329, -0.232586, 0.318036, 0.132996, 0.198251, 0.307027, 0.21293, 0.13203, -0.395423, -0.36999, -0.067229, 0.145502, -0.159658, -0.352593, 0.191714, -0.086577, 0.362593, 0.084581, 0.156955, 0.02025, -0.142409, 0.056737, -0.406511, -0.20272, -0.526763, -0.236484, 0.042831, 0.013709, 0.421056, 0.413891, 0.205538, -0.159526, 0.34055, -0.313339, -0.153196, -0.393762, 0.18609, -0.002499, -0.080764, 0.21044, -0.576721, -0.049189, 0.175794, -0.415619, -0.226256, 0.185736, -0.214533, 0.01638, 0.13347, 0.450776, 0.008857, -0.285268, -0.273079, 0.100704, 0.059536, -0.194909, 0.403611, -0.062757, 0.150211, -0.053562, 0.494864, 0.103489, 0.209355, 0.102228, -0.012735, 0.086206, -0.294206, -0.14496, -0.21453, -0.167554, -0.086423, 0.324187, 0.442461, -0.511002, -0.283324, 0.191264, 0.183416], [-0.030435, 0.182744, 0.483013, -0.21503, 0.016528, -0.182433, 0.095192, -0.035057, 0.065874, -0.243687, -0.105399, -0.094149, -0.036167, -0.169553, 0.334602, -0.396897, 0.036643, -0.335424, 0.10177, 0.215073, -0.026459, 0.344156, -0.200807, 0.253375, -0.046975, -0.012433, 0.115483, -0.06981, 0.102351, 0.032488, -0.111044, -0.080854, 0.005575, -0.171058, 0.01997, 0.113784, 0.176506, 0.102735, -0.146792, -0.006484, 0.174536, -0.131802, 0.011138, 3.7e-05, -0.268572, -0.040419, 0.110137, -0.062639, 0.217057, -0.198109, 0.249445, -0.267878, -0.049232, -0.010104, 0.072205, -0.050149, -0.063448, -0.249965, 0.153848, 0.371348, 0.051995, -0.469186, -0.346901, -0.192704, -0.119716, -0.285516, -0.131425, -0.312517, -0.336267, 0.19063, -0.115009, 0.206034, -0.029439, 0.1011, -0.137126, -0.142593, -0.080033, -0.128327, -0.075717, 0.142027, 0.036619, 0.283519, -0.146089, 0.244742, -0.11782, 0.302299, 0.174569, 0.301308, 0.042708, 0.0324, -0.013318, 0.217881, -0.026837, -0.378448, 0.270113, 0.246995, -0.183697, 0.024001, -0.068388, -0.360686, -0.188854, 0.355032, -0.346978, -0.215545, 0.309371, -0.284219, 0.21976, 0.035796, 0.123222, -0.126389, -0.151139, -0.271514, 0.207895, 0.102351, 0.015371, 0.085847, -0.016558, -0.021177, -0.041633, -0.054711, 0.304506, -0.102423, -0.160099, -0.049945, 0.214719, 0.288304, 0.171588, -0.110951], [0.303996, 0.076097, -0.000663, -0.172136, -0.369318, -0.292132, 0.071446, -0.524913, -0.240405, 0.154161, 0.03396, 0.16715, -0.219612, -0.037305, 0.156243, 0.140573, -0.563786, -0.503312, -0.168799, -0.062619, -0.176834, -0.251975, 0.153732, -0.127823, 0.034631, -0.337198, 0.391365, -0.081794, -0.326874, -0.639885, -0.137695, -0.044278, -0.070261, 0.281944, -0.118301, 0.219346, 0.209968, 0.223379, 0.152885, -0.61003, 0.217772, -0.381575, -0.045848, 0.185127, 0.158337, -0.021517, -0.32208, 0.130979, -0.63758, -0.454287, -0.443663, -0.06631, 0.452769, -0.519408, 0.122339, -0.12544, -0.35194, -0.114729, -0.100455, -0.023284, 0.118045, 0.179579, 0.213032, -0.303692, 0.075272, 0.238367, 0.062048, -0.075204, -0.040592, -0.432671, -0.191168, 0.059665, 0.231326, -0.359615, -0.008446, 0.72977, 0.045748, 0.446991, 0.185246, 0.241166, -0.122279, 0.511475, 0.199296, 0.169765, -0.204219, 0.295958, 0.036756, 0.432623, -0.345741, -0.056498, -0.17753, -0.25805, 0.37528, 0.794887, -0.297353, -0.253215, 0.084663, -0.009488, -0.294806, -0.174031, -0.032762, 0.031906, 0.539093, -0.507626, -0.102879, -0.305112, -0.253204, -0.061899, -0.209655, 0.060729, 0.167842, 0.354669, 0.038578, -0.151583, 0.230278, 0.113813, 0.262917, -0.243851, 0.213228, 0.178164, -0.020126, 0.217849, -0.126122, -0.150513, 0.245983, -0.150799, 0.03758, 0.001268], [0.058635, 0.062534, -0.019606, -0.062164, 0.20544, -0.245389, -0.186366, -0.001192, 0.072909, 0.095547, -0.27882, 0.077351, -0.092513, -0.072358, 0.259436, -0.301841, 0.188822, -0.209516, 0.193312, -0.05235, -0.210011, -0.11212, 0.076893, -0.132211, 0.092809, -0.042685, -0.019598, 0.139726, -0.029002, 0.183499, 0.076772, -0.178423, 0.037155, -0.177041, 0.191453, -0.204443, -0.198988, -0.219385, -0.005704, 0.079818, -0.13739, -0.328097, -0.043719, 0.217115, -0.144658, -0.18266, -0.253422, -0.044971, 0.146699, 0.011431, 0.034544, -0.105029, 0.185967, 0.230622, -0.230226, 0.095375, -0.105334, -0.053813, -0.198782, -0.070228, -0.077345, 0.087447, 0.126717, 0.084794, -0.000408, 0.190261, -0.076156, -0.080598, -0.100335, -0.097239, -0.066609, 0.257657, -0.108478, 0.239707, 0.247268, -0.000458, 0.005295, -0.124184, 0.166472, -0.054827, 0.171072, 0.065804, -0.139531, -0.020576, 0.0437, -0.014015, 0.273943, 0.044336, 0.201038, 0.063964, -0.017165, 0.168179, -0.259053, 0.232501, -0.306262, -0.09102, -0.090687, 0.027916, 0.157107, -0.208371, -0.138695, 0.148824, -0.14802, 0.041666, 0.124121, -0.297724, -0.076309, 0.052041, -0.261723, 0.070897, 0.036095, -0.128406, -0.082921, 0.082648, -0.149985, -0.324198, -0.059385, -0.180269, -0.159423, 0.248517, 0.02819, -0.011793, 0.264306, 0.140144, -0.046879, 0.063411, -0.052847, 0.229288], [0.234503, 0.161939, 0.032888, 0.068555, 0.187802, -0.272384, -0.012036, -0.219328, -0.008951, 0.155032, 0.623459, 0.099738, -0.276433, -0.271792, -0.540337, 0.365465, -0.387539, -0.1234, 0.05817, 0.202737, -0.269247, 0.151504, 0.134317, -0.192243, -0.012543, -0.100192, 0.068591, -0.27784, 0.064142, -0.124222, 0.441809, -0.112725, -0.268291, 0.210025, -0.073336, 0.307458, 0.459073, 0.040887, -0.266778, -0.209265, 0.237335, -0.22316, -0.014277, 0.14086, 0.42467, -0.566715, -0.420111, 0.051093, -0.125276, 0.151918, 0.099994, -0.338124, 0.096271, 0.144227, -0.032063, -0.196491, 0.169906, 0.584817, 0.008919, -0.117904, 0.352767, 0.252162, -0.155007, 0.133475, 0.099517, -0.036847, -0.11568, -0.168795, -0.254542, -0.04601, -0.105084, 0.150138, -0.228572, -0.21411, -0.341755, 0.063022, -0.341172, 0.108266, -0.205198, -0.167625, 0.374095, 0.239145, -0.165601, 0.090627, 0.145027, 0.235571, -0.190142, -0.055593, 0.111784, -0.620345, 0.267203, -0.422693, -0.127736, -0.219395, -0.020347, 0.298003, -0.293598, 0.254703, 0.502907, -0.253438, 0.061679, 0.062217, -0.093218, -0.044199, 0.078915, 0.265151, -0.456881, -0.305169, 0.102077, -0.281643, 0.002684, -0.345964, 0.449155, 0.040533, 0.421387, 0.180516, -0.518477, -0.41858, -0.050115, 0.167281, -0.068678, 0.189348, -0.160818, -0.001064, 0.0358, -0.045979, -0.187543, 0.082488], [-0.166975, -0.046132, 0.120831, 0.272894, -0.121521, 0.172827, -0.411713, 0.019695, 0.052325, -0.084845, -0.253481, 0.168163, -0.095326, -0.229323, 0.029191, -0.403227, 0.228197, 0.272409, -0.099706, -0.071673, -0.288525, 0.195776, 0.227538, 0.220362, 0.146088, -0.012501, 0.250285, -0.037349, 0.15327, 0.087747, -0.021199, -0.208167, -0.0392, 0.139119, 0.098437, -0.040658, 0.284915, 0.193549, 0.260291, 0.42749, -0.165575, 0.086747, -0.470363, 0.003696, 0.216632, -0.068307, -0.192172, -0.132969, -0.663085, 0.259854, 0.237021, 0.25703, 0.16777, -0.338227, -0.038698, 0.319855, -0.033305, 0.303326, -0.112921, -0.009316, 0.258446, -0.212536, 0.216514, -0.019474, -0.405813, -0.52051, 0.387336, 0.038652, 0.11149, 0.289502, 0.278402, -0.333948, 0.24157, -0.729196, 0.039005, 0.143754, -0.080296, -0.088155, -0.301907, -0.078868, -0.322361, 0.240539, -0.232839, 0.079996, 0.213573, 0.09571, -0.076502, -0.760943, -0.325419, -0.590637, 0.41738, -0.104668, 0.101621, 0.189498, -0.074743, 0.049238, 0.120266, -0.100927, 0.199708, -0.013026, 0.052753, 0.3059, 0.104208, -0.195466, 0.148933, -0.089348, 0.137625, 0.280014, -0.176738, 0.047169, 0.102224, -0.085085, 0.248059, 0.518021, 0.289497, -0.147397, 0.165856, -0.082031, 0.135422, -0.064929, 0.111601, -0.090248, 0.24338, 0.455725, -0.048297, -0.180042, 0.02804, -0.129519], [-0.29771, 0.196069, -0.014021, -0.061043, -0.127497, 0.22281, -0.258655, 0.078084, -0.191898, -0.140278, 0.245441, 0.317361, -0.017617, 0.0934, -0.264696, 0.256434, 0.468451, -0.147385, 0.194859, -0.249862, 0.067654, -0.095892, 0.129818, 0.064063, -0.025424, 0.21637, -0.078739, -0.17644, 0.330057, 0.048018, 0.074735, -0.27544, 0.094669, 0.013886, 0.184256, 0.05407, 0.246583, 0.22914, -0.079176, 0.216927, -0.171501, -0.172621, -0.243597, -0.17749, 0.024643, 0.072352, 0.018428, -0.390245, -0.078976, 0.384224, -0.066097, 0.045916, 0.136025, 0.046005, -0.163112, 0.000265, 0.317383, 0.02647, -0.162298, -0.216034, -0.307173, -0.12543, 0.162412, 0.198893, 0.224803, 0.089721, -0.058741, 0.121107, 0.011459, 0.307521, -0.173986, -0.203324, 0.094849, 0.166581, 0.324417, -0.334625, -0.035155, 0.073253, 0.212875, -0.155774, 0.213106, 0.021989, -0.151206, 0.21631, 0.103722, -0.060648, 0.248058, -0.174166, -0.028749, 0.250139, 0.049171, 0.017423, 0.335225, -0.389288, -0.39881, 0.083365, -0.410543, -0.14812, -0.397493, 0.145567, 0.008285, -0.178457, 0.265676, -0.042527, 0.010049, -0.232582, -0.058693, 0.001699, 0.090832, -0.262875, -0.374543, -0.026357, -0.200099, -0.0737, 0.146035, -0.053293, 0.216133, 0.143981, 0.178864, 0.129744, 0.08409, -0.205307, -0.245273, 0.21692, 0.294088, 0.18013, -0.447148, -0.326509], [0.4363, -0.192668, -0.482807, 0.249386, 0.206997, -0.337495, 0.175176, -0.170049, -0.031581, -0.46177, -0.389925, 0.176332, -0.240856, -0.213648, -0.358327, -0.211279, -0.277302, -0.178198, 0.22717, 0.370859, -0.021018, 0.165084, -0.089762, -0.331451, 0.225509, -0.203021, 0.024064, 0.073254, 0.276205, 0.307647, 0.324318, 0.413762, -0.023721, -0.241444, -0.054266, -0.141409, -0.226728, 0.067215, 0.331834, -0.030168, 0.003913, 0.197471, 0.198963, 0.132291, 0.236693, -0.040571, 0.02251, 0.009863, 0.392875, 0.087858, 0.442389, -0.184265, 0.147228, 0.083814, 0.041897, 0.233872, 0.078066, -0.011142, 0.142974, -0.322702, 0.232957, 0.180971, -0.220338, 0.117231, -0.156265, 0.067977, -0.155015, -0.014541, -0.111222, -0.241667, 0.186202, 0.183744, -0.239967, -0.123898, -0.26987, 0.192437, 0.147532, 0.039817, 0.139725, 0.348737, -0.218395, -0.065014, -0.039992, 0.006741, -0.271997, 0.235488, 0.194157, -0.213657, -0.054183, -0.136495, -0.311767, -0.072977, -0.090999, 0.101851, 0.083836, -0.195519, 0.322395, 0.317353, 0.413437, 0.04447, 0.167085, 0.007543, -0.443069, -0.201541, -0.017704, -0.055313, 0.24199, -0.035229, 0.121634, 0.119086, -0.255181, 0.161131, 0.088217, 0.18248, 0.138839, 0.192749, -0.146117, 0.400251, 0.11945, 0.094148, 0.002141, 0.097479, -0.034892, 0.088661, 0.055057, -0.234653, 0.072862, 0.3536], [0.030711, 0.288143, 0.114435, -0.226382, -0.336245, 0.038009, -0.195813, -0.126012, -0.081051, -0.200192, 0.136882, 0.193841, -0.005713, 0.451605, 0.040147, 0.282584, 0.132695, -0.10724, -0.269191, 0.013121, -0.203161, 0.014383, -0.06668, 0.161309, -0.226572, 0.102723, -0.315115, 0.044853, 0.38004, -0.136961, -0.295717, -0.183809, -0.032004, 0.150856, -0.076554, 0.454619, -0.278017, -0.388439, -0.169955, -0.294936, -0.128636, 0.179991, 0.036821, 0.096879, 0.239169, 0.315223, 0.066271, 0.446598, 0.186154, 0.066657, -0.073618, -0.081859, -0.099757, -0.183115, 0.233261, 0.107498, -0.267541, -0.221721, -0.450247, 0.225213, 0.080064, -0.117402, 0.151713, -0.200407, 0.228083, 0.044416, -0.235284, 0.148332, 0.010653, -0.233142, -0.196216, -0.214492, 0.184468, 0.386159, 0.332749, -0.111326, -0.136498, 0.049022, 0.100473, 0.782657, 0.088394, -0.049126, -0.12576, -0.257197, -0.029907, 0.019733, 0.242312, -0.00784, 0.16729, 0.771055, 0.009478, 0.180637, -0.023312, 0.101727, -0.340898, 0.060641, 0.071981, -0.179671, -0.159538, 0.112665, 0.130816, -0.276201, 0.070492, 0.346318, -0.09654, -0.008844, 0.012319, -0.184618, 0.2964, 0.491783, -0.04674, -0.356506, 0.069147, -0.169179, -0.35841, -0.115858, -0.075139, 0.323843, -0.021797, -0.091156, 0.256646, -0.214213, 0.109373, 0.348403, 0.079105, -0.052892, 0.020924, 0.024035], [0.074415, -0.166848, -0.064, -0.150949, -0.163096, -0.292585, 0.055532, 0.153725, -0.242308, 0.012181, 0.477187, 0.163031, 0.037572, -0.135453, -0.360234, -0.340028, 0.071382, 0.217115, 0.018874, -0.143179, -0.060562, -0.042406, -0.411096, -0.105499, 0.230303, -0.089672, 0.080018, -0.183783, 0.277098, -0.211387, -0.206235, 0.268777, -0.22977, -0.068565, -0.036167, 0.135405, 0.030742, 0.217669, 0.036532, -0.273052, -0.125841, 0.115452, 0.152793, -0.135244, -0.079161, 0.061806, -0.227999, 0.083216, -0.125719, 0.002772, -0.031844, -0.143537, -0.241087, 0.332575, -0.154005, 0.255515, 0.15315, 0.646171, -0.342039, -0.276969, -0.069328, -0.454655, 0.154799, -0.151661, -0.303877, -0.203509, -0.305657, -0.062205, -0.052278, -0.146565, -0.314998, -0.210703, 0.107546, 0.03097, 0.524056, -0.281747, 0.022245, 0.349029, -0.098288, 0.1985, 0.305133, 0.586081, -0.164287, -0.397239, -0.173703, -0.143739, 0.055013, -0.058249, 0.26073, 0.068706, -0.153916, -0.109724, 0.121232, 0.336549, -0.164449, 0.182734, -0.276777, 0.164981, -0.234208, 0.360674, -0.057047, 0.186835, -0.275159, 0.6818, -0.050234, 0.007526, -0.076391, -0.193592, -0.124997, 0.560875, -0.37982, -0.035834, -0.038938, -0.101075, -0.121681, 0.321221, 0.327337, -0.176674, 0.466866, -0.197764, 0.032539, -0.253798, 0.084616, 0.001709, 0.113191, -0.107776, 0.712088, -0.072381], [-0.248422, -0.082767, -0.137523, 0.530474, -0.009638, -0.181946, -0.033473, 0.058832, -0.323318, -0.33803, 0.593433, -0.13438, -0.146042, -0.067179, 0.158344, -0.140628, -0.015052, 0.300139, 0.005886, -0.19973, 0.110435, 0.295028, 0.266595, 0.098507, 0.01087, -0.249412, 0.406927, 0.257524, 0.031602, 0.152592, -0.040176, 0.240734, -0.159915, 0.497912, 0.055088, -0.054276, -0.385397, -0.224741, -0.572481, 0.181687, -0.150751, 0.13131, 0.061031, 0.002905, 0.110405, -0.171261, 0.084086, -0.253491, 0.327155, 0.476898, -0.034842, -0.149482, 0.21791, 0.058355, -0.073202, 0.231514, -0.149363, -0.137705, -0.370995, 0.00478, -0.024906, -0.0681, 0.410816, -0.009735, -0.455463, 0.042297, 0.257447, -0.208375, -0.189984, -0.479395, 0.080589, 0.000847, 0.050079, -0.219864, -0.123669, 0.25269, -0.013982, -0.11116, 0.029528, 0.089539, -0.11733, -0.334838, 0.110724, 0.233165, -0.331755, -0.358744, 0.057288, -0.043145, 0.385938, -0.041051, -0.569034, -0.43606, 0.30303, -0.258829, 0.003947, -0.020004, -0.209028, 0.359605, 0.027504, -0.141551, 0.384337, -0.017166, 0.144846, 0.156859, 0.165123, 0.229183, -0.201779, 0.271671, 0.056577, 0.195346, 0.092708, -0.237632, 0.062657, 0.110902, 0.152702, 0.209939, 0.124159, -0.012517, -0.196965, -0.007385, -0.147922, -0.118532, 0.044929, -0.355292, 0.141779, 0.176842, 0.066929, -0.226078], [-0.340665, 0.506463, 0.045277, -0.409303, 0.210658, 0.170584, -0.142733, 0.191852, 0.350408, -0.025797, -0.082726, -0.248782, -0.151736, 0.11312, -0.475594, 0.289712, -0.58011, 0.209301, 0.143263, -0.210214, -0.167001, 0.080157, -0.392712, 0.126975, 0.220004, -0.154756, -0.415641, -0.308323, 0.070704, 0.162769, -0.002999, 0.429215, 0.048867, 0.036591, -0.131832, 0.512945, 0.258464, 0.369115, -0.174456, -0.247239, -0.262584, 0.330597, 0.263045, -0.614203, -0.061175, -0.073574, -0.247918, 0.065311, 0.227211, -0.306603, 0.563446, -0.206036, -0.220757, -0.266827, -0.31754, 0.021717, -0.204013, 0.180615, 0.055998, -0.075872, -0.029088, 0.046246, 0.022261, 0.012, -0.111839, -0.031381, -0.05907, 0.121981, 0.032673, 0.335864, -0.232883, 0.071397, -0.463267, 0.14976, -0.25573, -0.028003, 0.049926, -0.107892, 0.055408, -0.239352, 0.224005, -0.135386, 0.086198, -0.115631, 0.129286, 0.100314, 0.182093, -0.395088, -0.520367, 0.02552, 0.129814, -0.362819, -0.251465, -0.143182, -0.367265, -0.395217, -0.081078, -0.229389, 0.09356, 0.173981, -0.407047, -0.14927, -0.137663, 0.10411, 0.21147, -0.129117, -0.110542, -0.49376, 0.48826, -0.293262, 0.038165, 0.362085, -0.262411, -0.212371, 0.201968, 0.156601, 0.181558, -0.252473, 0.410389, -0.050126, -0.082953, -0.223317, 0.008914, 0.213569, -0.299947, 0.040148, -0.111444, -0.133398], [0.378354, -0.901587, 0.312475, -0.650474, 0.076282, -0.021755, -0.119468, -0.576087, 0.080559, 0.674925, -0.302895, -0.172335, -0.069461, 0.191112, -0.097425, 0.091038, 0.18681, 0.194792, 0.141126, -0.077331, 0.270298, 0.051042, -0.058229, -0.219615, -0.023331, -0.006595, -0.322272, -0.121055, 0.066741, -0.284813, -0.16537, 0.01491, 0.438633, -0.364615, 0.132115, -0.114127, 0.201542, 0.219659, 0.651183, -0.34155, 0.48168, -0.074127, -0.11618, 0.041878, -0.196481, 0.178879, 0.013595, -0.518835, 0.122362, -0.210898, -0.157238, 0.152811, 0.014734, 0.072947, 0.175658, 0.181789, 0.430027, 0.245831, 0.129888, 0.511145, -0.118512, 0.207422, -0.446176, -0.210292, 0.403375, -0.450844, 0.134018, 0.076577, 0.637632, 0.278084, -0.128397, -0.086524, -0.124463, 0.072994, -0.115677, -0.143713, 0.328401, -0.066353, -0.123624, 0.205766, 0.145677, -0.42721, -0.527033, 0.053716, 0.230741, 0.542712, -0.151385, -0.195817, 0.186792, -0.180002, -0.21266, -0.207116, 0.136628, 0.235031, -0.072285, -0.321868, 0.16332, 0.020766, 0.189732, 0.196349, 0.154627, 0.198376, -0.185675, 0.314516, 0.089675, 0.503533, -0.430199, -0.275799, -0.280868, -0.187598, -0.298257, -0.160176, -0.104089, 0.2513, 0.154564, 0.448126, 0.266218, 0.577054, -0.382905, -0.244981, 0.410541, 0.2022, -0.423093, 0.052441, -0.114525, 0.497863, -0.249449, 0.028198], [-0.390999, -0.090461, 0.260221, 0.10227, 0.450923, -0.12725, -0.167901, -0.046333, 0.379531, 0.248273, -0.214599, -0.546083, 0.003492, -0.180027, 0.195469, 0.307238, 0.380295, -0.043564, -0.057847, -0.234475, 0.010433, 0.076607, -0.390343, -0.381751, -0.361018, 0.23568, -0.05556, 0.09417, -0.04239, -0.235645, 0.099945, 0.221359, 0.422355, -0.087481, -0.004684, -0.016903, 0.48891, -0.201627, 0.542491, -0.243239, 0.041482, -0.284819, -0.629154, 0.144137, 0.477794, 0.050595, -0.333299, 0.288302, -0.098907, 0.044337, -0.222882, 0.055097, 0.253245, 0.462944, 0.134209, -0.032287, -0.083984, 0.015857, 0.189082, -0.186033, -0.014715, 0.174088, 0.074487, 0.088938, 0.688257, 0.483963, -0.293826, -0.679668, -0.010177, -0.026485, -0.050486, 0.115662, -0.146399, -0.166341, 0.342563, -0.001558, -0.068899, -0.564296, -0.133938, 0.112205, -0.500649, 0.200834, 0.040674, -0.04503, -0.238477, -0.256411, 0.296259, 0.04033, 0.098852, 0.268667, -0.039449, 0.240841, -0.40089, -0.238501, -0.344587, 0.334783, 0.423401, -0.264132, -0.280461, -0.029776, -0.011845, -0.048526, 0.334486, 0.207488, 0.025927, 0.469521, 0.224974, -0.008173, -0.076458, 0.318051, 0.314449, 0.279399, -0.338223, 0.061665, 0.06965, -0.363902, -0.135501, -0.233629, -0.157155, -0.118122, -0.077029, -0.098726, 0.03796, -0.098392, 0.222754, 0.265031, 0.136823, 0.142607], [-0.230892, -0.00593, 0.032407, -0.033446, -0.040729, -0.017524, 0.165747, 0.246348, 0.136865, -0.133829, 0.035135, 0.090721, 0.057516, 0.028348, 0.000459, -0.163469, -0.098146, 0.027824, 0.012155, 0.272004, -0.015915, 0.034843, -0.028109, 0.049143, -0.067239, 0.045375, -0.020553, -0.047586, -0.008501, -0.203161, 0.037195, -0.103939, 0.002641, -0.119819, -0.025339, 0.031008, -0.118583, -0.119473, 0.117939, -0.064314, 0.018768, 0.280887, 0.028616, -0.183154, 0.060948, -0.167886, -0.210648, -0.108938, -0.029411, -0.014043, 0.009772, -0.064246, 0.191785, -0.025139, -0.022256, -0.154531, -0.060566, -0.061333, 0.060352, 0.098816, -0.025723, -0.069254, 0.14183, -0.049774, 0.059406, 0.040215, 0.01329, -0.009869, -0.021142, 0.038426, -0.095337, 0.004451, 0.173164, -0.027014, -0.140558, 0.111994, -0.051863, -0.017897, -0.144305, -0.076983, -0.009591, -0.035566, -0.003977, -0.015485, -0.038173, 0.154546, 0.075751, 0.007067, -0.011461, -0.022921, -0.043693, -0.167728, -0.074906, 0.031136, -0.105224, 0.174411, 0.000976, -0.154839, -0.093636, -0.082045, -0.027253, 0.047629, 0.101772, 0.036253, -0.251991, -0.177041, 0.041193, 0.13264, -0.004173, -0.147251, -0.111953, -0.176025, 0.002542, -0.107737, 0.085137, -0.03846, 0.030719, 0.061811, -0.097989, 0.173175, 0.088193, 0.026896, -0.267504, 0.00756, -0.00308, 0.065838, 0.088654, 0.002106], [0.212293, 0.060937, -0.007315, 0.493402, 0.072865, -0.154946, -0.53055, -0.175537, -0.059075, -0.00602, 0.139956, 0.160389, -0.258132, 0.240863, 0.116708, -0.063573, -0.088158, -0.295295, 0.267493, -0.079047, -0.155983, -0.321193, -0.085836, 0.018533, 0.533345, -0.291624, -0.142588, 0.077265, -0.105679, -0.141272, 0.040028, 0.300528, 0.208583, 0.021565, -0.205823, -0.169066, 0.125629, 0.143502, -0.139199, -0.112153, 0.626243, -0.080761, -0.215574, 0.250192, -0.062194, -0.537144, 0.141287, 0.390965, -0.501036, 0.13921, -0.346203, 0.256194, -0.421329, 0.070984, 0.058235, 0.283528, -0.176108, 0.429809, 0.071416, -0.417833, 0.441166, -0.096506, 0.353876, -0.066805, 0.032809, 0.129108, -0.338076, 0.005153, 0.081515, -0.030911, 0.411526, 0.051422, 0.060465, 0.139731, -0.294857, -0.003473, -0.156682, 0.602043, 0.047463, 0.10624, -0.488313, -0.020909, 0.384676, 0.297734, -0.17178, 0.184914, -0.235087, -0.017449, -0.103075, -0.349359, 0.284746, 0.287002, 0.07815, -0.021343, 0.119171, -0.007944, 0.067881, -0.359338, -0.02862, 0.271561, 0.16615, 0.148251, -0.008298, 0.141382, 0.247465, 0.11959, 0.214982, -0.154087, 0.104032, 0.213506, -0.187568, 0.121797, -0.063657, 0.037841, 0.229733, -0.028863, -0.122235, 0.214083, -0.138391, 0.402854, 0.417087, 0.507425, -0.17277, 0.31249, 0.249316, 0.023064, -0.044398, -0.656206], [0.228, 0.098619, -0.152623, -0.150819, -0.062687, -0.210495, -0.269019, 0.009076, -0.199471, -0.035553, 0.229926, 0.417021, 0.255655, -0.249706, 0.248923, -0.114628, 0.055134, -0.14942, -0.335657, 0.140563, -0.098968, 0.256069, 0.435386, 0.33363, 0.113825, 0.339579, -0.131687, -0.205119, -0.100154, -0.031599, -0.267689, 0.032319, -0.306705, 0.070688, -0.388214, -0.050794, 0.002124, -0.343436, 0.00048, 0.215964, -0.190603, 0.138071, 0.280691, 0.160247, -0.239239, 0.204675, 0.130283, 0.370487, 0.362663, 0.063402, 0.365462, 0.127592, 0.08193, -0.087482, 0.250431, 0.120605, -0.374296, -0.316767, -0.036692, 0.138175, -0.432036, 0.141129, 0.060419, 0.025873, 0.057582, 0.28502, 0.072051, -0.181696, 0.069639, 0.000924, -0.263224, -0.306026, 0.160422, 0.142731, 0.0264, 0.089463, -0.077, -0.249669, -0.202006, -0.451037, -0.092946, 0.420419, -0.025764, -0.044952, 0.475848, 0.011013, 0.254487, 0.047712, 0.003438, -0.093451, 0.102986, 0.192583, -0.252928, -0.285689, 0.113125, -0.384137, -0.044455, -0.356123, 0.295537, 0.308346, 0.02087, -0.053893, 0.076369, 0.258956, -0.03395, -0.274795, 0.092436, -0.126607, 0.115376, -0.096604, -0.217574, 0.088026, 0.055489, 0.103996, 0.229457, 0.339344, 0.456198, 0.138081, -0.173681, 0.040935, 0.199058, 0.279499, 0.019378, -0.118592, 0.162392, 0.301179, -0.137115, 0.33571], [0.314505, -0.103089, 0.216999, -0.019286, -0.034155, 0.143101, -0.145035, 0.124223, -0.104158, 0.135331, 0.156653, -0.187654, -0.074314, 0.153575, 0.102983, -0.080115, 0.341919, -0.023432, 0.105929, 0.035596, 0.147998, 0.06623, 0.058795, 0.51948, -0.156732, -0.19615, -0.013214, -0.181544, 0.075395, 0.267313, 0.006281, 0.014691, 0.040575, -0.067445, 0.018494, -0.046742, -0.048349, -0.196019, 0.098798, 0.215664, -0.072349, -0.02527, -0.064067, -0.28385, 0.411939, -0.213366, 0.332315, -0.087646, 0.014356, -0.157271, -0.146287, -0.037624, -0.318429, 0.011002, 0.178467, 0.023224, -0.258615, -0.165242, 0.40516, -0.186339, -0.018918, 0.216513, -0.238025, -0.072229, -0.069982, -0.144995, 0.163853, -0.098089, -0.108538, -0.121103, 0.271424, 0.185303, -0.179733, -0.011938, 0.006216, -0.148712, 0.126327, 0.121678, -0.018921, 0.128734, -0.169258, 0.003451, 0.317682, 0.024981, -0.267334, 0.148495, -0.209791, -0.011091, 0.16134, -0.017674, 0.078925, -0.059442, -0.088884, 0.089583, 0.022399, -0.063044, 0.066191, -0.165763, -0.028262, 0.120772, 0.280286, -0.088471, 0.123141, -0.008544, 0.134248, 0.026645, -0.097355, -0.026026, -0.386981, -0.094456, 0.018618, 0.09785, -0.115232, -0.188014, -0.098567, -0.135182, 0.498448, -0.008406, -0.160459, 0.561547, -0.156064, -0.024583, -0.183614, 0.110497, 0.112504, -0.211437, 0.0393, 0.050341]], [-0.343589, -0.363136, -0.706175, -0.399546, -0.289998, -0.256612, -0.209371, -0.063939, -0.04133, -0.278008, -0.011238, -0.151659, 0.308831, -0.008194, -0.21117, -0.390503, -0.123581, -0.302413, -0.532168, -0.035326, -0.064372, -0.291758, -0.108648, -0.301625, -0.26903, -0.248112, 0.088161, -0.027394, -0.031454, -0.021918, 0.092078, -0.001041, -0.116872, -0.339451, -0.011061, 0.402402, -0.200077, -0.390759, 0.023816, -0.413438, 0.058328, 0.212331, 0.152169, 0.012698, -0.125588, 0.202569, -0.338311, -0.161924, -0.341458, 0.031517, -0.428726, 0.169895, -0.074314, -0.143746, -0.327457, 0.159623, -0.437621, -0.30314, -0.619684, 0.11029, -0.833486, -0.385187, -0.036862, -0.195607, 0.126784, -0.115953, -0.084563, 0.214825, -0.284642, -0.147051, -0.174782, 0.023097, 0.180999, -0.571556, -0.062256, -0.199917, 0.099481, 0.17571, 0.282803, 0.000145, -0.199857, -0.165979, 0.179959, -0.449674, -0.352601, -0.28529, 0.059503, -0.005282, 0.346666, -0.031001, -0.135865, -0.200604, -0.036743, -0.034617, -0.607144, -0.378379, -0.003108, -0.598475, -0.392833, -0.587691, 0.170464, 0.083231, -0.490601, -0.080748, 0.05648, 0.268999, 0.234925, -0.258105, -0.183099, 0.115589, -0.010182, -0.15449, -0.294649, -0.451079, -0.11483, 0.250099, -0.087033, -0.337616, 0.206738, -0.208639, 0.156064, 0.122802, -0.075553, -0.173921, -0.181287, -0.101157, 0.001089, -0.274476]),
    ([[0.522435, 0.026543, -0.079884, -0.114884, -0.296492, 0.151704, 0.118309, 0.028005, 0.094094, 0.240054, -0.249973, 0.081084, 0.291362, -0.192474, 0.125878, -0.362893, 0.127758, 0.377259, 0.113806, 0.300473, -0.248017, -0.205526, -0.240403, 0.043918, -0.635438, 0.354792, -0.486243, 0.216974, -0.026942, -0.121011, 0.030739, -0.058041, 0.196226, 0.037888, -0.164697, -0.149101, 0.164191, -0.082669, 0.058046, 0.106162, -0.343953, 0.266043, -0.079639, 0.545032, 0.325873, 0.006818, -0.207064, -0.17655, -0.300394, -0.125666, -0.460613, 0.120449, 0.158054, 0.189113, -0.007112, 0.174763, 0.067183, -0.156739, 0.170796, 0.198665, -0.90152, 0.083711, 0.129006, -0.054864], [-0.328588, 0.121392, -0.559326, -0.478117, -0.035022, 0.197095, 0.227476, -0.401636, -0.059298, 0.064111, -0.024105, 0.003264, 0.181374, -0.058195, 0.263801, 0.194569, 0.276987, 0.10683, -0.014247, -0.222339, 0.134387, 0.088404, -0.112722, 0.126888, -0.084398, -0.002237, 0.005058, 0.078344, -0.323836, -0.411876, -0.043523, 0.193784, -0.055711, -0.02116, 0.207913, -0.18233, -0.378326, -0.121262, -0.198588, -0.00559, -0.483888, -0.1344, -0.07045, -1.30475, 0.148149, 0.319971, 0.380397, 0.140398, 0.085557, 0.060781, 0.138945, -0.23198, -0.303303, 0.152684, 0.122242, -0.120231, 0.031561, 0.115893, 0.411216, -0.012266, 0.324825, -0.057408, -0.48594, -0.115515], [-0.093465, 0.224743, 0.121215, 0.195981, 0.035074, -0.293363, 0.016673, 0.493182, 0.136044, -0.336052, 0.186724, -0.35553, -0.295125, -0.39919, -0.387011, -0.020069, 0.14401, 0.098412, 0.195706, 0.017705, -0.143695, 0.30152, 0.137095, -0.064387, -0.030071, 0.502204, 0.317669, 0.722435, -0.010728, 0.155187, -0.072549, 0.149705, -0.020134, -0.166803, 0.172747, 0.422719, 0.012813, 0.112351, -0.179986, 0.1982, -0.273102, 0.247451, -0.178397, 0.080394, -0.056878, 0.142839, 0.423946, -0.070812, 0.332197, 0.249461, 0.188117, -0.116896, 0.156637, -0.473191, 0.005601, 0.090552, 0.140462, 0.196904, 0.11468, 0.103448, 0.342647, 0.032544, 0.090726, 0.161266], [0.088165, 0.292722, 0.41998, -0.084645, -0.05635, 0.29337, 0.08042, 0.502768, -0.035711, 0.487255, 0.20431, 0.287096, 0.184297, -0.355045, 0.271052, -0.057759, 0.439166, 0.163767, 0.012498, -1.290519, 0.303331, 0.02121, -0.153657, 0.290712, -0.416814, -0.257612, -0.18009, -1.147735, -0.071872, 0.001401, 0.044151, 0.111004, 0.138983, 0.045522, 0.077214, -0.119474, -0.11465, -0.217452, -0.083329, -0.296257, 0.129636, -0.181311, -0.29055, -0.408512, 0.252846, -0.259578, 0.187036, 0.358335, 0.455126, -0.063523, -0.451815, -0.213099, -0.041865, -0.376531, -0.202697, -0.103961, -0.164486, 0.198088, 0.028425, 0.033394, 0.233132, 0.145125, 0.005903, -0.139645], [0.217861, -0.025596, 0.162367, -0.00969, -0.129362, -0.430795, -0.132222, 0.061524, 0.351028, -0.231191, -0.143457, -0.070678, -0.645555, -0.00939, -0.03817, -0.19077, -0.057806, -0.251065, 0.029369, -0.099032, -0.056284, 0.024051, 0.220155, -0.0221, -0.107095, 0.260966, -0.077996, 0.349526, 0.037566, -0.159849, -0.098595, -0.009647, 0.093579, -0.103828, -0.070248, -0.125971, 0.158777, 0.028468, 0.020934, 0.038158, -0.196007, -0.581102, 0.240433, 0.332477, -0.06278, -0.31244, -0.011969, 0.23143, 0.092814, -0.175019, 0.115537, -0.424169, 0.096855, -0.197101, 0.239225, 0.130552, 0.137226, 0.097453, 0.149478, -0.048857, -0.08341, -0.22842, 0.076338, 0.129161], [-0.091185, 0.223939, -0.134928, -0.125196, 0.25116, 0.285931, -0.236266, 0.081013, -0.028447, -0.021263, 0.018677, 0.212982, 0.172073, -0.034742, 0.205612, 0.146563, 0.06012, -0.163352, 0.02303, -0.166983, 0.30898, -0.049925, 0.218073, 0.013654, 0.237267, 0.164942, 0.035786, -0.419548, 0.111764, 0.164282, 0.179627, -0.4399, -0.035891, 0.033138, -0.795523, -0.01372, -0.054066, -0.243357, 0.002814, 0.209972, -0.018932, 0.061286, -0.150119, 0.157672, 0.088104, 1.358958, 0.109896, -0.05254, 0.229373, -0.030074, 0.130305, -0.067479, -0.031028, -0.119409, -0.227923, 0.003078, 0.083747, -0.465358, -0.003067, 0.168589, 0.000928, 0.204961, 0.164927, -0.419805], [-0.18839, -0.165443, 0.277377, 0.176298, -0.373987, -0.210635, -0.048903, -0.099053, -0.051932, -0.033633, 0.034722, -0.052499, 0.131094, -0.04332, 0.163443, -0.166053, 0.006057, 0.21889, 0.305816, -0.075875, -0.348275, 0.033124, -0.691381, 0.127957, 0.05836, -0.071718, -0.014986, -0.7569, 0.189083, 0.009823, -0.056145, -0.451298, 0.270673, -0.165705, 0.062315, 0.206364, -0.096107, -0.127797, -0.019125, 0.093065, 0.039378, 0.157876, -0.322518, 0.258183, -0.096575, -0.76433, 0.154251, -0.091895, 0.215573, 0.018464, -0.101706, -0.094028, 0.169303, 0.097467, 0.077753, -0.051808, -0.038124, 0.03018, -0.317089, 0.125339, 0.059851, -0.05552, 0.18905, 0.078539], [0.241807, -0.039645, -0.196749, 0.414338, -0.204998, 0.217362, 0.066104, -0.222801, 0.041621, 0.085302, -0.043647, -0.363011, -0.003555, 0.204636, 0.121745, -0.002875, -0.27803, -0.042729, 0.252987, 0.288698, 0.189771, -0.072221, -0.174404, 0.000313, -0.094462, -0.030619, -0.204096, -0.138936, -0.122128, 0.147204, 0.2671, 0.185425, -0.009957, 0.038612, -0.082909, -0.017982, 0.534143, 0.069609, 0.073035, 0.224182, 0.060303, 0.240331, 0.231838, -0.632911, 0.347119, -0.274536, -0.294461, -0.13281, -0.15282, 0.013475, -0.056935, 0.082629, -0.021489, 0.148411, -0.102208, -0.05482, 0.106426, -0.171654, 0.042616, -0.284368, 0.057272, -0.06303, 0.462585, -0.070447], [0.034468, -0.25089, 0.096402, -0.008201, 0.245789, -0.153858, 0.073033, 0.012398, 0.167655, -0.080601, -0.124114, 0.037025, 0.166879, -0.249253, 0.188086, 0.024276, 0.156421, -0.140908, -0.243614, 0.022607, 0.279981, 0.132747, -0.101166, -0.260794, 0.126785, 0.110683, 0.304916, 0.186763, 0.36002, -0.037691, -0.08297, -0.060241, -0.008684, -0.05163, 0.365902, 0.180569, 0.211888, -0.157759, -0.18974, 0.187408, -0.073282, -0.275396, -0.060003, 0.15282, -0.196136, 0.101721, 0.089788, 0.018375, 0.186127, 0.124905, 0.064393, -0.070513, 0.074208, 0.148414, 0.259928, 0.130965, 0.15889, -0.500423, -0.168197, 0.216439, 0.122818, 0.241926, -0.342386, -0.008823], [-0.074347, -0.323279, 0.041468, 0.100703, 0.126202, 0.232804, 0.164131, -0.014347, 0.110102, -0.335673, -0.011564, -0.508454, -0.109906, 0.211246, -0.166048, -0.028164, 0.357125, 0.016465, -0.016481, 0.512003, -0.195342, 0.059287, 0.275711, 0.113726, 0.110201, -0.039888, 0.092675, -0.626414, 0.034094, 0.343708, 0.040017, -0.159632, 0.334032, -0.925745, 0.175231, 0.148724, 0.285456, 0.080841, 0.165668, 0.215481, -0.072813, -0.021416, 0.251436, -0.086395, 0.291951, 0.013735, 0.078161, 0.225481, 0.051806, 0.031215, 0.215914, 0.051705, 0.055929, 0.015676, -0.139213, 0.05797, 0.440361, 0.194318, 0.081838, -0.10016, 0.108978, -0.120416, -0.498187, -0.149877], [-0.242045, -0.084226, -0.040106, -0.478591, -0.528956, 0.278338, 0.152991, -0.202727, -0.138223, 0.059088, -0.136595, -0.290231, -0.395711, 0.034728, 0.114949, -0.058896, -0.20656, 0.34401, 0.060818, -0.056857, -0.070662, -0.249643, 0.007431, 0.22188, -0.242455, 0.102164, -0.123715, -0.055203, -0.18787, -0.101922, -0.067696, -0.154204, 0.05938, 0.029626, 0.045381, 0.035212, -0.056167, 0.102426, -0.123041, -0.32129, 0.25857, -0.223744, -0.486058, 0.257939, 0.123386, -0.00239, -0.384271, -0.056208, -0.005511, -0.263394, 0.028835, -0.268177, -0.14917, 0.024969, -0.165829, -0.059328, -0.214362, -0.153861, 0.096547, 0.274202, 0.624007, 0.215071, -0.009397, -0.216841], [-0.566521, 0.03677, 0.04256, 0.447475, -0.33049, 0.022148, -0.037623, 0.116972, 0.195041, 0.089394, -0.337442, -0.06918, -0.091905, 0.299599, -0.282776, 0.008143, 0.218576, 0.149295, -0.011444, -0.2764, 0.049725, 0.200275, 0.061528, 0.206055, 0.236811, 0.024924, -0.066641, 0.08116, -0.201444, 0.250414, 0.001514, 0.19381, -0.086304, -0.32039, -0.032346, -0.2023, -0.219284, -0.238134, -0.293111, -0.015281, 0.118527, 0.010306, -0.451619, -0.121793, 0.197266, 0.240358, -0.617004, 0.183695, -0.233304, -0.229704, -0.172392, 0.176653, 0.188522, -0.319357, -0.087112, 0.180312, -0.088838, 0.184606, 0.249823, -0.092551, 0.293977, 0.012013, -0.218094, -0.008705], [-0.198459, 0.036973, -0.071115, 0.111686, -0.097449, -0.030312, -0.176534, 0.015372, -0.429782, 0.038014, 0.032156, -0.099345, 0.007762, -0.256298, 0.259311, -0.365655, 0.128939, -0.002997, -0.098302, 0.184484, -0.047812, -0.167741, 0.017328, -0.002677, -0.335043, 0.001167, 0.068505, 0.298198, -0.296183, -0.106595, -0.253367, 0.167662, -0.795084, 0.020425, 0.070194, -0.052906, 0.208622, -0.107018, 0.071627, -0.051207, 0.072158, 0.316028, 0.012272, 0.077623, -0.539754, -0.322487, 0.229568, 0.273175, -0.131264, -0.002255, -0.072831, 0.152749, 0.057581, 0.08622, -0.060921, -0.061465, -0.16674, -0.159972, 0.125967, 0.242733, -0.131803, 0.105526, -0.365022, -0.323539], [-0.224186, -0.286684, -0.031732, -0.10569, 0.278745, -0.573246, -0.016912, 0.002514, -0.168002, -0.188056, 0.397409, 0.003509, -0.074328, 0.045439, -0.03876, -0.215484, -0.424446, 0.073097, -0.438764, 0.075874, 0.259853, -0.009906, -0.122986, -0.224525, -0.23593, -0.020485, 0.045437, -0.520863, -0.129966, -0.033073, -0.289961, -0.066323, -0.295737, 0.271032, -0.593208, -0.043633, -0.023753, 0.223467, -0.237574, -0.090671, 0.028299, 0.061848, 0.105087, -0.070084, -0.076471, -0.659452, -0.068621, 0.139013, -0.074294, -0.017906, -0.136298, 0.058166, 0.109981, 0.126973, 0.109546, 0.120789, 0.044475, -0.404349, 0.05047, -0.013851, -0.085128, 0.086612, 0.29212, 0.015886], [-0.922095, 0.115191, 0.117391, 0.373752, -0.284555, -0.147192, -0.126288, -0.080883, -0.301922, -0.004689, -0.189814, 0.047667, 0.088367, -0.02888, -0.273557, -0.102499, -0.217821, 0.162128, -0.297697, -0.061968, -0.064406, -0.064117, -0.136377, -0.019546, -0.204247, 0.150727, -0.101271, 0.129467, -0.000123, 0.185423, -0.16479, 0.102102, 0.054123, 0.013662, 0.120032, -0.17779, -0.139193, 0.235264, -0.038986, -0.22735, 0.037999, 0.174826, 0.467864, -0.344802, 0.354644, 0.228374, 0.106828, -0.211995, -0.176846, -0.296245, -0.142115, 0.01094, 0.189447, 0.073276, -0.112541, -0.13604, -0.421043, 0.147568, -0.111293, -0.060156, -0.997511, -0.099858, -0.453324, -0.101015], [0.356388, 0.010064, 0.021178, 0.021321, 0.193932, 0.044153, -0.00438, 0.55577, 0.443928, -0.083334, -0.017575, -0.009108, -0.215819, -0.035471, 0.195524, 0.188848, -0.168156, 0.242311, -0.554368, -0.167694, -0.431151, 0.062651, -0.033481, -0.09629, -0.025942, 0.248, -0.078176, -0.134491, 0.131206, 0.427797, 0.057334, -0.102422, 0.258965, 0.506455, -0.231241, -0.180175, 0.519093, 0.200423, -0.079871, 0.331905, 0.221423, 0.088388, -0.082494, -0.472857, -0.4107, -0.215971, 0.304666, 0.175361, -0.07634, -0.335267, 0.067025, 0.032922, 0.081559, -0.113165, -0.108745, -0.161615, 0.097327, -0.112474, 0.022693, -0.160865, -0.063453, 0.279429, -0.218569, 0.136845], [-0.439941, 0.307443, -0.336192, 0.309322, 0.11091, -0.128659, -0.298905, -0.056388, 0.29581, 0.201746, -0.140299, -0.272811, -0.067722, -0.053364, 0.035724, 0.284364, 0.044813, -0.178494, 0.036248, -0.372868, -0.163249, 0.001706, 0.247088, -0.041282, -0.422218, -0.060821, -0.363366, 0.066585, -0.095935, 0.154799, 0.1489, 0.072086, 0.088413, 0.09956, -0.217226, -0.103659, -0.159849, 0.009762, -0.104104, -0.010734, 0.153812, 0.225646, -0.211274, -0.043829, -0.075349, -0.188433, -0.282337, 0.321839, -0.549847, 0.052706, 0.208917, -0.163381, -0.074431, 0.276095, 0.175413, 0.012862, 0.031094, -0.495711, 0.003724, -0.111789, 0.022197, 0.132698, -0.075433, -0.280629], [-0.06044, 0.063409, -0.546723, 0.00795, 0.026659, -0.328593, -0.205366, 0.067097, -0.075735, -0.157089, -0.029959, -0.252937, -0.22061, 0.054102, 0.246339, 0.285077, -0.154492, -0.089412, 0.273441, -0.048173, -0.035495, 0.162707, -0.624157, 0.057754, 0.258737, -0.059925, 0.123559, 0.250837, 0.137695, 0.081432, 0.254532, -0.05822, -0.007555, 0.019833, -0.004787, 0.081742, 0.237306, 0.166458, -0.184293, -0.074168, 0.169397, -0.21253, -0.13306, -0.006116, 0.013928, -0.593725, -0.463684, 0.087948, 0.078461, -0.053399, 0.027255, 0.00816, -0.051416, -0.523929, -0.012317, 0.17996, -0.222304, -0.125798, 0.230769, -0.000875, -0.421875, -0.163128, -0.200839, -0.11821], [-0.043459, 0.044751, -0.284741, -0.361426, -0.089495, 0.094336, 0.199336, 0.165052, 0.340529, -0.234173, -0.005489, 0.117904, -0.683667, 0.262057, 0.161009, -0.054534, 0.049147, 0.344396, 0.273318, 0.203022, 0.358457, 0.144756, -0.09079, 0.086914, -0.216406, -0.071326, -0.282821, -1.590309, 0.496404, 0.095043, 0.174572, 0.168286, 0.168716, 0.034357, -0.116177, 0.040019, 0.019633, 0.1026, -0.203917, 0.109149, -0.199361, -0.418663, 0.232204, 0.006211, 0.104206, -0.103452, -0.387134, 0.160712, 0.108072, -0.050196, -0.132955, 0.165925, 0.169451, 0.047471, 0.092205, 0.421247, -0.105784, -0.061621, 0.110429, 0.087492, 0.135706, -0.190931, -0.002642, 0.148441], [-0.518433, -0.063444, -0.096129, -0.011392, -0.052023, -0.122205, 0.039766, -0.07353, -0.03876, 0.051575, 0.027254, -0.213436, -0.058314, -0.024189, 0.267274, -0.288476, 0.096198, -0.148183, 0.17715, 0.239893, 0.007094, 0.062417, -0.135707, -0.208994, -0.249585, -0.030991, -0.169508, -0.265299, 0.048498, 0.115335, 0.182451, 0.031179, -0.218687, 0.196426, 0.009268, -0.059151, -0.032931, 0.136109, -0.075631, 0.097066, -0.070592, 0.005197, -0.106458, -0.329658, -0.020267, 0.107901, 0.00216, 0.001469, 0.164618, -0.371217, 0.246268, 0.252029, 0.041985, -0.188347, -0.002729, 0.321898, -0.112679, 0.062469, -0.126726, 0.065189, 0.237067, 0.177589, 0.230251, 0.193918], [0.355046, 0.081357, 0.307887, 0.076676, 0.227078, 0.072929, 0.124101, -0.128154, -0.119777, -0.030045, -2.7e-05, -0.113093, 0.241521, -0.009546, -0.05832, 0.132931, 0.172458, -0.128895, 0.179518, -0.040219, 0.173059, 0.013117, 0.204025, 0.125084, -0.048769, 0.012006, -0.29628, -0.049605, 0.044594, 0.097506, 0.20686, -0.12199, 0.00599, 0.204224, 0.019485, 0.085287, 0.179806, -0.009009, -0.094772, 0.209217, 0.214162, 0.12715, -0.063796, 0.942905, -0.273958, 0.037483, 0.099728, 0.019947, -0.002557, -0.301394, 0.116416, 0.029374, -0.16342, -0.034976, 0.06861, -0.024359, 0.136514, -0.124329, 0.058362, 0.002223, 0.005686, -0.116903, -0.256733, -0.002507], [0.225524, -0.149736, 0.031055, 0.106072, 0.219145, -0.242394, 0.076857, 0.185809, 0.091936, -0.033702, 0.211548, -0.121871, 0.059576, -0.396584, 0.108926, -0.03989, 0.209298, 0.008768, -0.183224, -0.21697, -0.036622, -0.235125, -0.148408, 0.174254, -0.086031, 0.158797, 0.000931, -0.088004, 0.161929, 0.007802, -0.054744, 0.044706, -0.319899, -0.358884, 0.152434, -0.320993, -0.143003, -0.205709, 0.012247, -0.06222, 0.180978, -0.220135, -0.432454, -0.019284, -0.052105, 0.110934, -0.135207, 0.131963, 0.079659, 0.406556, -0.112035, 0.224844, -0.248255, 0.192391, 0.133474, 0.487743, 0.104269, 0.25665, 0.292382, -0.002138, -0.577132, -0.11697, -0.071486, 0.06309], [-0.178754, 0.261845, -0.135563, 0.078485, -0.005868, 0.079138, 0.124134, 0.121201, 0.106364, 0.102498, -0.098731, 0.019018, 0.058607, 0.270927, 0.462695, 0.176503, 0.111081, 0.107211, -0.350355, -0.456065, -0.8295, -0.380388, -0.079854, 0.115541, -0.090723, -0.032465, 0.11743, 0.132741, 0.096442, -0.024733, -0.082948, 0.027099, -1.13619, 0.014954, -0.103158, 0.077644, -0.013447, -0.291148, 0.285903, 0.197264, 0.286988, 0.016935, 0.115818, -0.166663, -0.109144, 0.36417, -0.209083, -0.098992, -0.221636, -0.043509, 0.006779, 0.174337, -0.243795, -0.050115, -0.013775, -0.06685, -0.448079, -0.3369, -0.034924, -0.01859, 0.061205, 0.193577, -0.08795, 0.216305], [0.040888, -0.144099, -0.453322, -0.197347, -0.090798, -0.214583, 0.219459, 0.026413, -0.307553, -0.110726, -0.284374, 0.124184, 0.021732, 0.040137, -0.623081, 0.197548, -0.000558, -0.094393, -0.043097, 0.276769, -0.025573, 0.150366, -0.044611, 0.063202, 0.22357, 0.010868, -0.213792, 0.069254, 0.001224, 0.097107, -0.217177, 0.206744, -0.009928, 0.023811, 0.01635, -0.406629, 0.030481, -0.156958, -0.086415, 0.087843, 0.285288, 0.244536, 0.292582, -0.314112, 0.136129, -0.277294, -0.57539, -0.62374, 0.004199, 0.443053, 0.258074, -0.087196, -0.294593, -0.168871, 0.221469, -0.269129, 0.101674, 0.137301, 0.079115, 0.00052, -0.899411, 0.124189, 0.290904, -0.056661], [0.122955, -0.076647, -0.118068, -0.195462, -0.104886, -0.166921, 0.142714, -0.142905, 0.030602, 0.010803, -0.277741, 0.156107, -0.085796, -0.014404, 0.154636, 0.182731, -0.096537, 0.132308, -0.645991, 0.049967, -0.034976, 0.154515, 0.00901, 0.124439, 0.182544, -0.03042, -0.154579, -0.287672, -0.350173, -0.043089, 0.071763, 0.154741, 0.005752, -0.751391, -0.299281, -0.34512, -0.180984, 0.232781, -0.005493, -0.203471, 0.020215, -0.07201, 0.095878, -0.138843, 0.232915, 0.111527, -0.370915, -0.03939, -0.020358, -0.248334, -0.136295, -0.120707, 0.080401, -0.092962, -0.073484, -0.33118, 0.018866, 0.134424, -0.140897, 0.24929, 0.049718, -0.133371, 0.222145, 0.045328], [-0.76709, 0.076614, 0.038809, 0.073365, 0.296488, 0.100601, -0.005389, -0.62625, -0.197658, -0.144597, -0.023465, 0.017328, 0.136332, 0.260725, -0.163584, 0.248337, -0.202696, -0.118607, -0.157107, -0.114236, 0.038794, -0.186186, 0.13163, 0.055155, -0.168143, 0.08218, 0.044299, -0.102717, -0.072562, 0.055034, -0.041724, 0.212256, -0.224999, -0.132086, 0.412391, 0.019551, 0.176129, 0.022751, 0.132684, -0.10603, 0.077941, -0.141842, 0.121409, 0.033584, 0.090524, 0.066449, 0.124685, -0.111583, 0.117206, 0.062151, -0.076222, 0.302861, -0.377814, 0.060061, 0.017439, -0.262164, -0.410221, -0.166777, -0.224709, -0.105003, -0.55014, -0.035369, -0.753821, -0.162388], [-0.195968, 0.037294, 0.118874, 0.105048, -0.400965, 0.036164, -0.020629, -0.161114, -0.196287, 0.04517, 0.12002, 0.052405, 0.004419, -0.172168, 0.089787, -0.068549, 0.327677, -0.001697, 0.029143, -0.625494, -0.016207, 0.224659, 0.270086, 0.113164, -0.084601, 0.066481, -0.093753, 0.120142, -0.311583, 0.198556, -0.014007, 0.147887, -0.49394, 0.05697, -0.115819, -0.180875, -0.051701, -0.092511, -0.043749, -0.093053, 0.232211, 0.11292, -0.055465, -0.163062, -0.239745, 0.048667, -0.315899, 0.218264, -0.02621, -0.187658, -0.043896, -0.164558, -0.233629, -0.199014, 0.14353, -0.126157, 0.02605, -0.261574, -0.229694, -0.271309, 0.088494, 0.188584, -0.116173, -0.125466], [0.015685, -0.130471, -0.0388, -0.216224, -0.044799, -0.011419, 0.019182, 0.209753, 0.067947, -0.124565, 0.057904, -0.227982, -0.037899, -0.033036, 0.119889, -0.195379, -0.132733, -0.304411, -0.499068, -0.250686, -0.14683, -0.193713, 0.021395, 0.192735, -0.183868, 0.181252, 0.058682, 0.289308, 0.102179, -0.391221, 0.127739, -0.058727, 0.093843, -0.478538, 0.067669, -0.235249, -0.194471, -0.1083, 0.034558, 0.07402, 0.169551, -0.047245, 0.242243, 0.343523, 0.072144, -0.120057, -0.538674, 0.110856, 0.126198, -0.072085, 0.104686, -0.163148, -0.125763, 0.046732, 0.243128, 0.111647, -0.16577, -0.335915, -0.013148, 0.091087, -0.503992, -0.095967, 0.439413, -0.075511], [0.058873, 0.428795, -0.25419, -0.035148, -0.119797, -0.116326, -0.255788, -0.196131, -0.052095, 0.228499, 0.218739, 0.185711, -0.015296, 0.090685, 0.06608, -0.329139, 0.184527, -0.179412, 0.31926, -0.186421, -0.049382, -0.0669, -0.07448, -0.072016, -0.528872, -0.253072, 0.263121, -0.096654, -0.105862, -0.092336, 0.052801, -0.083697, 0.017508, 0.056985, -0.200213, 0.002057, 0.091584, 0.255054, -0.211496, -0.090829, -0.197215, 0.000434, -0.589674, -0.07608, -0.658238, 0.321221, -0.362836, -0.257744, -0.602753, -0.004551, -0.101277, 0.488665, -0.127961, -0.079043, -0.046337, 0.193189, -0.142597, 0.145344, 0.178098, -0.04489, 0.006606, -0.22956, -0.170371, 0.193508], [0.124494, 0.070459, 0.023034, -0.013068, 0.23602, -0.161868, -0.048909, 0.138247, 0.105051, 0.166055, -0.000146, 0.162727, -0.074183, 0.020474, 0.067751, -0.18591, -0.129902, -0.16893, -0.163886, 0.043391, -0.311854, -0.098349, -0.133372, 0.221507, -0.30172, 0.009825, 0.275601, 0.14323, 0.171879, -0.109929, 0.124573, -0.196675, -0.195788, -0.068887, 0.020831, 0.108124, -0.268772, -0.170341, -0.015058, 0.059462, -0.099504, 0.189749, -0.15678, 0.13738, -0.079379, -0.085257, 0.101878, 0.15411, 0.285729, 0.28301, 0.059962, -0.25843, -0.131345, -0.065444, 0.27713, -0.150177, 0.159502, 0.062431, 0.047699, 0.06819, 0.086208, 0.146798, 0.232585, 0.20098], [-0.207689, -0.139626, -0.016894, 0.285981, 0.036632, 0.192302, -0.182949, -0.017829, -0.035046, 0.255341, -0.154024, 0.123645, 0.202314, 0.157783, 0.052894, 0.047546, 0.057933, 0.350163, 0.114195, -0.148471, 0.148777, 0.085198, 0.118569, 0.033883, 0.025781, 0.016227, 0.016704, -0.190027, 0.288671, 0.036161, -0.064434, -0.13398, -0.124625, -0.128502, 0.246588, -0.15478, 0.099579, -0.008537, 0.02653, -0.064651, 0.060934, 0.003765, 0.034448, -0.115529, -0.024132, -0.115228, 0.197653, 0.034221, -0.055752, 0.192875, 0.030671, -0.048711, -0.240913, 0.004373, -0.242532, 0.046428, -0.329268, 0.214659, -0.193201, 0.28649, 0.003458, 0.247027, 0.1109, 0.130514], [0.032171, -0.223927, 0.070422, 0.254501, -0.019793, 0.138456, -0.124539, -0.043842, 0.057026, 0.141153, -0.289055, 0.003129, 0.076987, 0.067437, -0.038232, -0.401803, -0.074748, -0.035164, 0.050824, -0.211582, 0.343728, 0.033129, -0.111047, 0.11439, -0.224982, 0.122655, -0.051001, -0.4344, -0.064071, -0.310365, 0.212907, 0.031357, -0.224974, -0.095524, -0.36936, 0.108225, -0.209984, -0.392236, 0.015452, -0.021765, -0.288986, -0.293599, -0.095101, -0.428904, 0.363374, 0.08484, 0.131079, -0.094593, 0.060307, 0.057609, -0.084804, -0.768352, 0.276553, 0.124238, -0.11566, 0.423988, -0.123271, 0.003798, 0.009367, 0.244747, -0.192115, -0.248956, -0.123553, 0.009735], [0.037943, -0.131734, 0.132391, -0.321613, -0.022678, 0.101008, -0.058783, 0.304233, -0.16149, 0.156971, -0.116693, 0.065314, 0.072506, 0.26122, 0.019389, -0.300304, 0.060567, -0.225475, 0.293549, 0.0081, 0.114544, -0.064766, -0.118256, -0.083331, -0.018873, 0.023411, 0.112736, -0.023623, -0.074497, -0.348361, -0.13909, 0.00591, 0.049305, -0.076827, -0.075338, -0.179683, -0.08104, 0.286509, 0.250997, 0.130105, -0.035549, -0.047001, -0.029461, -0.093738, -0.059918, -0.053583, -0.423882, 0.309795, -0.155378, -0.116338, -0.218681, -1.082212, -0.564532, 0.239304, 0.084621, -0.271828, -0.210689, -0.159965, 0.130659, 0.377359, -0.104948, 0.283033, 0.065177, -0.187511], [0.076632, 0.15688, 0.151457, -0.045672, 0.377513, -0.159691, 0.225114, 0.463071, 0.282249, 0.09726, -0.18492, 0.048897, -0.232554, -0.007646, 0.019547, -0.002818, 0.034182, 0.135232, 0.36075, 0.147351, -0.430436, -0.052649, 0.057568, -0.203357, -0.089206, 0.186177, -0.017425, -0.254883, 0.064505, 0.294329, -0.147501, 0.089667, 0.030904, 0.383463, -0.10263, 0.065552, -0.25851, -0.266613, 0.095187, -0.067676, -0.191361, 0.42285, 0.135983, 0.017483, 0.282214, 0.3011, 0.130835, 0.161989, -0.00305, -0.292553, -0.201868, -0.199118, -0.27465, 0.168357, -0.044997, -0.009361, 0.099787, 0.307459, 0.126451, 0.029598, 0.188039, -0.276953, -0.166137, 0.379107], [-0.086523, -0.045487, -0.088682, 0.156488, -0.029225, -0.251465, 0.068664, -0.231281, -0.210963, 0.110239, -0.000319, -0.178169, 0.159755, 0.195895, 0.207913, -0.404805, -0.287761, 0.005746, 0.067251, -0.151544, -0.535509, 0.162168, -0.42663, 0.030498, 0.173196, 0.008272, -0.052743, -0.441372, -0.055596, 0.049214, 0.171791, -0.619949, -0.021233, -0.090012, -0.610151, -0.105643, -0.002492, -0.001905, -0.040045, -0.090567, -0.03631, -0.092292, 0.284894, -0.054125, -0.214238, -0.153774, 0.1231, -0.041902, 0.039828, -0.0205, 0.06557, -0.013477, 0.091461, -0.013488, 0.151732, -0.275749, -0.101786, 0.224187, 0.069901, -0.02815, -0.020951, 0.056869, 0.202799, 0.149169], [0.108007, 0.128812, -0.187315, -0.128565, 0.066778, -0.193356, 0.043223, -0.135771, -0.056181, 0.086651, 0.059813, -0.344466, 0.194965, 0.144616, 0.134862, -0.207496, 0.070907, -0.18357, -0.296574, -0.007926, 0.018367, -0.038901, -0.069919, 0.259867, -0.167479, -0.39009, -0.141543, -0.043737, 0.088196, 0.1008, 0.088863, 0.114946, 0.018468, -0.259796, -0.084837, 0.620488, -0.04678, 0.064701, -0.00171, -0.199861, -0.11396, 0.036166, -0.325201, -0.43444, -0.11735, 0.117416, -0.209078, 0.070122, 0.007278, -0.21985, 0.192459, 0.313446, 0.230661, -1.140283, 0.294453, -0.067781, 0.117718, 0.12035, -0.126521, 0.079391, -0.115285, -0.005427, -0.056173, -0.125116], [0.103687, 0.334452, -0.130274, -0.255929, -0.069237, -0.318372, 0.208599, 0.310503, 0.07123, -0.085042, 0.056289, 0.157499, -0.012206, 0.034794, 0.220326, -0.018466, -0.09879, 0.074821, 0.269795, 0.036034, 0.062189, 0.238102, 0.115274, -0.140297, 0.233341, 0.132309, 0.270746, -0.130386, -0.139344, 0.157318, 0.103475, 0.116944, 0.092229, -0.160079, 0.195279, 0.205945, -0.032242, -0.407262, 0.082121, 0.205677, -0.217957, -0.095853, 0.081087, -0.402487, 0.135786, 0.177475, 0.298156, -0.044284, 0.033868, 0.375203, -0.049183, -0.187715, 0.113631, -0.26519, -0.13633, 0.11279, 0.016265, 0.129127, -0.137392, 0.137882, -0.050794, -0.026183, -0.08014, -0.096707], [0.202743, -0.367565, 0.117151, 0.02603, 0.110232, 0.023034, 0.244845, 0.299505, 0.210633, -0.3267, 0.507873, 0.101947, -0.036402, 0.183334, 0.478262, -0.080056, 0.095813, -0.213406, 0.254731, 0.179439, -0.318148, -0.128557, 0.150466, -0.0872, 0.00839, -0.420814, 0.173603, 0.285778, -0.037528, -0.240334, -0.258496, -0.114453, -0.377031, 0.276932, -0.285347, 0.339332, -0.037131, -0.411131, -0.107837, -0.034042, -0.212951, -0.340803, 0.040379, 0.051416, 0.111892, 0.298572, 0.19786, 0.069803, -0.04533, -0.307253, -0.22803, -0.037006, -0.121424, 0.172177, -0.024918, 0.028693, 0.370762, 0.008709, 0.091016, 0.189585, 0.368995, 0.040465, -0.047898, 0.117295], [-0.130014, 0.042797, -0.060047, 0.063241, -0.072642, -0.100868, -0.021094, -0.085387, -0.198024, -0.329531, 0.222057, 0.183971, -0.128143, 0.147597, -0.197087, -0.527518, 0.180569, -0.446539, -0.354979, -0.046742, -0.631285, 0.153474, -0.190263, -0.038945, 0.055655, 0.25247, 0.017055, 0.094913, -0.176389, -0.276573, -0.047937, 0.013805, -0.03144, 0.098244, -0.219379, 0.13517, 0.055796, 0.16775, -0.142009, 0.279761, -0.098364, -0.148378, -0.125933, 0.163321, -0.129966, -0.076408, -0.282696, -0.390865, 0.187735, 0.198563, 0.378084, -0.069032, -0.025997, -0.000731, 0.120514, -0.356109, 0.217368, 0.298083, 0.281962, -0.407198, -0.120885, -0.028204, 0.244459, -0.029316], [0.004376, 0.010665, -0.236692, 0.156807, -0.033037, 0.062018, 0.324472, 0.179291, -0.058155, 0.019844, -0.20383, 0.244262, -0.051908, -0.179409, -0.193096, -0.178151, 0.011493, -0.104267, -0.184092, 0.239975, -0.467889, 0.028642, 0.009745, 0.199877, 0.002205, 0.275025, 0.417507, -0.484317, -0.246352, 0.031282, 0.092442, -0.09639, -0.23439, 0.269995, -0.14207, -0.081803, 0.065801, -0.464063, -0.102776, 0.144407, 0.101759, 0.084197, 0.003362, 0.256928, 0.028721, -0.335516, 0.213648, -0.064414, -0.049084, 0.053912, -0.002142, 0.111692, -0.025113, 0.074996, 0.040975, 0.154664, -0.013792, 0.327896, 0.07129, -0.300654, 0.129126, 0.204701, 0.196377, -1.196113], [-0.185501, 0.189982, -0.29332, -0.128037, -0.283521, -0.265304, -0.302569, 0.062122, -0.065029, -0.121182, 0.127952, 0.060624, 0.240517, -0.320568, 0.05604, 0.022409, 0.29505, -0.395385, -0.326453, 0.053661, 0.357417, 0.00541, 0.011575, 0.280114, 0.171398, -0.040719, 0.196801, 0.274921, 0.178854, 0.106237, 0.220416, -0.149902, 0.158137, -0.012666, 0.29967, 0.0186, -0.180802, 0.098217, -0.336615, -0.096739, -0.141832, -0.592362, 0.128131, 0.092771, 0.218108, 0.179263, -0.091604, 0.066086, -0.008745, 0.205263, -0.169218, -0.104403, 0.113639, 0.248092, 0.071977, 0.11287, -0.218996, -0.093399, 0.021466, -0.158402, 0.072964, 0.258778, -0.281376, 0.089455], [-0.282285, -0.031001, -0.118505, -0.338133, 0.137877, -0.099471, 0.013519, -0.088312, -0.128276, 0.220086, -0.021617, 0.228118, -0.217523, 0.205859, -0.085613, -0.216783, 0.020658, -0.0838, -0.522817, 0.122632, -0.063184, 0.300525, -0.012899, -0.332247, -0.004524, 0.030135, 0.065754, -0.001152, 0.254771, 0.074843, -0.031744, 0.009266, -0.102334, -0.244957, 0.079943, 0.329037, 0.310764, -0.183335, 0.169022, -0.004329, -0.032548, 0.291015, 0.560751, -0.053791, -0.047579, 0.042565, 0.010103, -0.035134, -0.251086, 0.1017, 0.200093, -0.435676, 0.085182, -0.129424, -0.075175, 0.247739, 0.001916, -0.023237, -0.097812, 0.127106, -0.065688, -0.179064, 0.003276, -0.072184], [-0.446544, 0.046153, 0.057503, 0.11882, 0.202888, -0.168639, -0.248877, -0.177387, -0.076461, 0.032177, 0.375338, -0.004205, -0.133064, 0.027395, 0.241566, 0.005328, 0.051407, -0.433812, -0.115066, -0.142023, -0.12525, -0.053824, -0.11304, 0.064745, -0.183254, -0.272838, -0.44456, 0.065005, 0.197497, -0.068076, -0.135442, 0.129376, -0.308881, -0.070889, 0.05122, 0.031719, 0.194511, 0.20123, -0.010872, -0.102441, 0.000381, 0.002835, 0.030646, -0.372369, -0.04486, 0.092493, 0.175485, 0.069055, 0.093224, -0.499593, -0.06119, -0.062057, -0.153082, -0.202927, 0.029448, -0.09486, 0.138987, 0.060083, 0.182695, -0.481744, 0.346334, -0.244339, -0.082469, 0.266585], [0.22144, 0.100774, 0.201542, -0.283206, -0.076957, 0.060699, -0.018134, -0.41124, -0.234705, 0.062643, 0.158925, -0.227975, -0.116585, 0.163519, -0.074029, 0.194919, -0.263837, 0.130847, -0.43223, -0.197764, -0.435136, -0.332243, -0.234263, 0.370866, -0.385777, -0.014048, -0.131122, -0.049809, -0.30241, -0.123253, -0.276518, -0.30461, 0.042145, -0.113097, -0.200448, 0.141995, -0.031138, -0.012334, 0.125622, -0.257321, 0.086215, -0.248396, 0.069662, -0.415915, -0.048424, -0.224219, 0.181505, -0.034405, -0.105353, -0.085243, 0.11245, 0.067638, -0.073184, -0.061541, 0.188522, 0.171495, 0.147748, -0.097164, -0.116528, -0.063826, 0.065554, -0.170673, -0.027357, -0.035235], [0.063142, 0.103173, -0.094963, 0.31656, 0.175199, 0.014308, -0.569742, -0.027708, 0.197536, 0.25084, -0.178879, -0.157707, -0.012981, -0.301331, -0.245928, -0.030169, -0.009866, -0.104997, 0.144516, 0.037352, -0.275229, 0.001654, 0.077305, 0.257592, 0.115628, -0.126191, -0.099011, -0.042504, 0.070141, -0.244359, 0.403155, -0.180761, 0.398895, -0.063465, 0.037955, -0.407853, -0.109409, -0.234048, -0.439227, -0.08387, -0.035485, 0.211263, 0.463136, -0.107234, 0.074919, -0.174978, 0.277785, 0.229717, 0.004347, 0.108227, -0.009008, -0.351639, -0.035292, -0.019817, 0.15319, 0.304539, 0.337655, -0.190996, -0.28638, 0.016425, -0.068068, -0.074192, -0.068972, 0.141021], [-0.380869, -0.004429, 0.005665, -0.024038, -0.081961, -0.329054, 0.162707, 0.019993, -0.181855, 0.026603, -0.070737, -0.139836, -0.067783, -0.058798, -0.268694, -0.385825, -0.142104, -0.130863, -0.135842, -0.138497, 0.174471, -0.059366, -0.205921, -0.161565, -0.055351, -0.135158, 0.145966, 0.016636, 0.317248, -0.225842, -0.426795, -0.277907, -0.237276, 0.025947, 0.106852, -0.035397, -0.226014, -0.044957, -0.090624, -0.005149, -0.066719, -0.240948, 0.229627, 0.0019, 0.055999, -0.029564, -0.066797, 0.140041, -0.018606, -0.412289, -0.131724, 0.011607, -0.12083, 0.287188, 0.167499, 0.074988, -0.009352, 0.217078, -0.339922, 0.058173, -0.400275, 0.082424, 0.110077, 0.136657], [-0.534986, -0.196128, 0.248101, -0.066771, 0.054037, -0.441492, -0.013628, 0.326845, -0.030097, -0.142866, 0.206319, -0.458586, -0.416222, -0.199286, 0.252409, 0.044752, -0.094221, -0.484635, 0.40279, 0.176727, -0.047061, -0.256605, -0.165831, 0.026074, -0.322695, -0.030097, 0.0602, 0.070924, 0.017143, -0.166108, -0.262392, -0.010191, -0.25132, -0.263661, -0.193585, 0.48968, -0.405559, 0.428116, 0.023785, 0.167576, -0.023731, 0.038244, 0.025525, -0.986094, 0.043012, -0.033987, -0.148006, 0.018103, 0.156403, -0.073962, 0.075095, 0.085112, 0.151733, -0.105654, 0.072917, -0.049842, 0.052559, -0.03263, 0.023207, 0.228747, -0.031126, -0.255874, 0.122556, 0.108479], [-0.846235, -0.03348, 0.039594, -0.970742, 0.07601, -0.299024, -0.15618, -0.162309, 0.077034, 0.026362, -0.045369, 0.442307, -0.108344, -0.343959, -0.321942, 0.360636, 0.00944, 0.035387, -0.37343, -0.45513, -0.001447, -0.097443, -0.122745, 0.140164, 0.137207, 0.709903, -0.324766, 0.246556, 0.255788, 0.578401, 0.331475, 0.132524, 0.247372, -0.410387, -0.307386, -0.027843, 0.155946, 0.16399, -0.019386, 0.047557, 0.357177, 0.259985, -0.106799, 0.211332, -0.210406, -0.19742, 0.109398, -0.329854, -0.161366, 0.455564, 0.23526, 0.149281, -0.052727, -0.24036, -0.238511, -0.091807, 0.072245, 0.050618, -0.195894, 0.253845, 0.422567, 0.005942, -0.097082, 0.132221], [-0.037495, 0.099479, 0.398957, -0.255585, 0.125719, 0.065363, -0.56449, -0.109339, -0.059321, -0.074194, 0.062813, 0.121672, 0.20835, 0.163745, -0.366693, -1.364506, 0.485694, -0.07361, -0.431841, -0.1763, 0.333963, 0.164105, 0.216945, -0.138638, -0.222875, 0.23806, -0.454456, 0.329925, -0.130324, -0.229826, -0.208726, 0.020066, -0.409777, -0.230994, -0.147009, 0.219585, -0.036454, -0.018782, -0.015933, 0.258522, 0.092062, -0.090634, -0.261454, 0.073841, 0.177776, -0.257531, -0.140172, -0.129062, -0.222995, -0.005749, 0.219587, -0.463582, -0.278534, -0.335852, -0.297135, 0.225331, 0.237659, -0.080677, 0.272474, -0.006885, -0.18244, -0.095237, 0.143324, -0.63344], [0.074395, -0.243566, -0.231257, -0.076532, 0.026111, -0.296161, 0.16286, -0.229622, -0.035193, 0.215863, -0.21605, 0.20453, 0.004691, 0.025689, -0.093811, 0.196171, -0.273667, 0.305158, -0.131489, 0.000749, 0.319836, -0.064585, 0.003274, 0.081182, 0.176185, -0.187531, 0.173593, -0.2229, 0.005702, 0.241984, -0.172436, 0.112483, -0.335073, 0.006996, 0.251145, 0.372715, -0.106188, -0.071604, -0.01171, -0.431272, 0.109072, -0.330083, 0.010633, 0.415239, 0.147695, 0.154457, 0.114496, 0.277628, 0.218873, 0.074242, -0.087718, 0.079427, -0.199404, 0.08723, 0.04763, 0.160234, 0.05297, -0.036948, -0.121017, 0.129708, 0.093125, -0.174233, 0.173221, -0.335148], [-0.62062, 0.19074, -0.413022, 0.234691, -0.045612, 0.283715, -0.282928, -0.068394, -0.092579, 0.149763, 0.144607, -0.265663, -0.003347, -0.236213, -0.996557, 0.085144, 0.112851, 0.242959, 0.42621, 0.236109, 0.033204, -0.023635, 0.039393, -0.094711, 0.135436, -0.249228, 0.222602, 0.288256, -0.012036, 0.016797, 0.13887, -0.12598, 0.666644, 0.161433, -0.054094, 0.169537, 0.300205, 0.068876, -0.099828, -0.351272, 0.108205, -0.038052, 0.01589, 0.391274, -0.223098, -0.104296, -0.033435, -0.397313, 0.203633, -0.379279, 0.023841, 0.263485, -0.235323, -0.183492, -0.308342, 0.156237, 0.31127, 0.207767, -0.185436, -0.323556, -0.530908, -0.235161, -0.309887, 0.210531], [0.050993, -0.153881, -0.118873, -9e-06, -0.036233, 0.065785, -0.033224, -0.37494, -0.179014, -0.109167, 0.320572, 0.10192, -0.469379, -0.170329, -0.010437, -0.034558, 0.098, 0.056022, -0.052727, 0.069684, -0.360447, 0.213594, -0.026219, -0.112071, -0.287622, 0.157602, 0.071412, 0.229991, -0.144263, 0.067363, -0.257939, 0.206856, 0.141502, -0.019935, -0.525004, 0.175026, 0.106999, 0.085635, 0.130212, 0.159744, -0.129634, 0.039726, -0.059372, -0.083455, -0.030687, 0.015149, 0.12264, -0.039168, -0.066112, 0.143737, 6.8e-05, -0.008589, 0.158754, -0.149974, 0.004586, -0.070134, 0.00953, 0.249153, 0.026765, -0.065913, 0.110324, 0.200013, 0.114593, 0.34159], [-0.731178, 0.281317, 0.024566, 0.059551, 0.206034, 0.229664, 0.008589, -0.074304, 0.058357, 0.314898, 0.10936, 0.151305, 0.176745, 0.048759, -0.41352, -0.212608, -0.403018, 0.080874, 0.123472, -0.180626, -0.166208, -0.075621, 0.160258, 0.033267, -0.127225, 0.04222, -0.059294, -0.007189, 0.14047, -0.087846, -0.078653, 0.082351, -0.166739, 0.25977, 0.319831, 0.15661, 0.220199, 0.150384, -0.012763, -0.053795, 0.105017, -0.167681, -0.165637, 0.078685, -0.230209, -0.039808, -0.209089, 0.180803, -0.166085, 0.264214, 0.1022, 0.029852, 0.133739, -0.206801, 0.100814, 0.137239, -0.287139, -0.011494, 0.030324, -0.052017, 0.19163, -0.049729, 0.101467, -0.144479], [-0.213794, -0.052733, 0.352132, -0.011978, -0.183897, 0.060417, 0.122789, 0.012984, -0.12975, -0.057493, 0.046778, 0.374343, -0.193058, 0.249066, 0.294498, 0.086252, 0.340898, 0.064419, -0.643003, 0.071271, -0.634154, 0.007847, -0.350116, -0.41331, -0.373056, 0.110069, 0.301636, -0.707838, -0.525002, 0.047581, -0.154581, 0.228422, -0.108618, -0.202777, -0.45191, 0.096275, -0.021694, -0.269608, -0.210847, 0.09289, -0.106966, -0.04789, -0.170388, 0.138848, -0.47906, 0.587499, -0.102453, 0.367103, -0.044964, 0.082652, -0.204027, 0.171945, 0.106053, 0.112576, 0.081078, -0.106837, -0.239275, -0.16195, 0.345311, 0.084109, 0.099059, -0.248505, -0.171399, -0.151935], [-1.440275, -0.25234, 0.008088, 0.381536, -0.336181, -0.359077, -0.049187, -0.394912, 0.08626, -0.095212, -0.231965, -0.132459, -0.149224, -0.260757, 0.376855, 0.169104, 0.008558, 0.339244, 0.087576, 0.211491, -0.019797, -0.067047, 0.105057, 0.101182, 0.235357, -0.011799, -0.421488, -0.04762, -0.126078, 0.162051, 0.022095, 0.040059, -0.148576, 0.024529, 0.107159, 0.168966, -0.087962, 0.013411, 0.037468, -0.248811, -0.262939, -0.214819, 0.319074, -1.917395, 0.105953, 0.025456, -0.007183, 0.219603, 0.165103, 0.186901, -0.136792, 0.01673, 0.542684, -0.634198, 0.166839, -0.07489, -0.119012, 0.124354, 0.070802, 0.061053, -1.027609, 0.072214, 0.070277, 0.071223], [-0.087804, 0.202299, -0.142257, 0.083316, 0.180172, 0.083975, 0.346266, -0.065507, -0.110747, 0.139965, 0.190001, 0.075764, 0.101726, 0.139723, 0.049193, 0.326692, -0.132255, -0.014129, 0.015487, 0.053964, 0.023261, -0.169681, 0.039554, 0.17283, -0.090483, -0.189166, 0.028381, 0.0728, -0.241958, -0.136203, -0.070329, 0.001522, 0.022047, 0.151175, 0.462637, -0.053661, -0.198584, 0.144497, -0.263219, 0.106954, -0.004903, -0.127973, 0.01281, 0.204703, -0.100877, -0.057589, -0.556668, -0.106234, -0.56734, -0.059283, 0.054173, -0.036599, 0.110988, 0.107729, -0.180632, -0.118061, -0.087976, -0.003791, 0.012721, 0.100531, 0.081482, -0.042512, 0.28191, 0.151177], [-0.196091, -0.031276, -0.044675, 0.226225, -0.064882, 0.31266, -0.085752, -0.541267, 0.059126, -0.224594, 0.196708, -0.599233, -0.151657, 0.178492, 0.496225, 0.144678, 0.294393, -0.188898, 0.083475, 0.831764, -0.359108, 0.055915, -0.053012, -0.241117, -0.239718, -0.06653, -0.072579, -0.163743, -0.093462, -0.197799, -0.017705, -0.090705, 0.287674, 0.116522, -0.833773, 0.0838, -0.349535, 0.188983, -0.062907, 0.075307, -0.075804, 0.263113, 0.310831, 0.181666, 0.273957, 0.339211, -0.115267, -0.163215, 0.382122, -0.117387, -0.020075, -0.026095, 0.131541, -0.112822, 0.10278, -0.298319, -0.153243, 0.219036, -0.157952, 0.05538, -0.036769, -0.023908, 0.211947, -0.124041], [-0.108133, -0.134406, 0.101431, -0.092856, 0.378094, 0.332576, -0.287228, -0.308356, 0.083205, 0.076483, -0.168497, 0.016171, -0.005223, 0.414062, -0.085296, -0.130991, 0.169021, -0.029572, 0.309565, 0.078296, -0.141856, -0.309532, -0.098476, -0.783483, -0.252264, -0.014617, -0.040753, -0.098234, -0.27247, 0.106155, 0.136756, 0.013981, 0.342273, -0.243646, -0.120222, 0.026205, -0.158335, -0.008089, -0.286144, 0.128029, -0.042693, 0.481129, -0.035071, 0.254462, 0.277806, 0.255062, 0.163382, 0.062695, 0.09725, 0.100008, -0.005802, 0.031703, 0.181801, -0.331836, -0.037764, 0.297172, -0.289585, 0.232486, -0.241001, -0.062093, -0.679859, -0.256048, 0.224379, 0.132436], [-0.138731, -0.423842, 0.153369, 0.235024, -0.01131, -0.486906, 0.265107, -0.11384, 0.107369, -0.127213, 0.07722, 0.275984, 0.273081, 0.264829, 0.140308, 0.01648, 0.191775, -0.169838, -0.191221, 0.134072, 0.038618, 0.061052, 0.198565, 0.016244, 0.102365, 0.113228, -0.318933, 0.061282, -0.130711, 0.16697, -0.237005, 0.287756, -0.284304, 0.294186, 0.241744, -0.090068, -0.046434, -0.097805, 0.179251, -0.206219, 0.052294, -0.091835, -0.413704, -0.597532, 0.104489, 0.072193, 0.150891, -0.437512, -0.057413, 0.263839, -0.154759, 0.108948, 0.211777, 0.250065, -0.113862, -0.181372, -0.252583, 0.173389, -0.174757, -0.161256, 0.146164, 0.183248, -0.863747, -0.206557], [0.195077, -0.21182, -0.289829, 0.37141, -0.252473, -0.083339, 0.166978, -0.445304, -0.585166, 0.048693, 0.030455, -0.151473, 0.283966, 0.29204, -0.207789, -0.55047, 0.156426, -0.042879, -0.261388, -0.139533, 0.16878, -0.093326, -0.035809, 0.198731, -0.115122, -0.090026, -0.076647, -0.316458, -0.292617, 0.219506, 0.219403, 0.104526, -0.012512, -0.091047, -0.440463, 0.136114, 0.028583, 0.23021, 0.110593, -0.074133, 0.132653, -0.201904, 0.025122, 0.079355, -0.03632, -0.43287, 0.097077, -0.015533, -0.186335, 0.04096, 0.190723, 0.093249, 0.22606, 0.175942, -0.133244, 0.356853, -0.094848, -0.076093, 0.364052, 0.113777, 0.129493, 0.156444, 0.073633, 0.071202], [0.288756, 0.344505, 0.224043, 0.075659, 0.131845, 0.054915, -0.089312, -0.691302, 0.136206, -0.184259, -0.700381, 0.040561, -0.337615, -0.037745, 0.224105, 0.574222, -0.568319, -0.04001, 0.06469, 0.092348, 0.143677, -0.153209, -0.106753, 0.066253, 0.23843, -0.125798, -0.07323, -4.6e-05, 0.047842, -0.07293, 0.366284, 0.739045, -0.512183, -0.01824, -0.43681, 0.240991, 0.232841, 0.182002, -0.643773, 0.101942, 0.107705, -0.216293, -0.29754, 0.169424, -0.416477, -0.110636, 0.005206, -0.150923, -0.110744, 0.215117, -0.266771, 0.154322, -0.170433, 0.251081, 0.287451, -0.183329, -0.046218, -0.941897, -0.062838, 0.052883, -0.122284, -0.631348, -0.349492, 0.012644], [-0.403034, -0.007566, 0.251867, 0.268936, -0.201364, 0.140547, -0.023823, -0.080289, 0.21769, -0.863722, -0.32511, -0.074319, 0.206898, -0.099946, -0.152275, 0.235097, 0.191119, -0.087065, 0.783719, 0.126218, -0.128884, 0.115177, -0.091885, -0.060049, 0.363556, -0.088144, -0.195535, -0.17707, -0.244006, 0.15851, -0.240457, -0.39338, -0.059253, -0.078108, 0.077906, 0.133519, -0.120425, 0.32762, 0.339382, -0.15141, -0.00543, 0.302703, 0.028418, 0.075502, 0.189652, -0.262164, 0.058607, 0.069628, 0.194378, 0.364747, -0.176749, -0.005947, -0.193861, -0.014011, -0.095623, -0.144354, 0.152343, -0.285382, -0.2076, 0.086098, 0.180613, 0.163153, 0.117826, -0.084787], [-0.204525, 0.202632, 0.125474, -0.801379, -0.302531, 0.087034, 0.053142, -0.04617, 0.188752, -0.095467, 0.156721, 0.049997, 0.005711, 0.005765, -0.050751, 0.076009, -0.148969, 0.155306, -0.030799, -0.119222, -0.124629, 0.077081, -0.108749, 0.067441, -0.241101, -0.741684, 0.000924, -0.11854, 0.130291, -0.076795, 0.165565, 0.020091, -0.005758, 0.274584, -0.240578, -0.319706, -0.072511, -0.073017, 0.187275, -0.456333, -0.149173, 0.102033, -0.390512, -0.96259, 0.204342, -0.020765, 0.077914, 0.186898, 0.014097, -0.335514, 0.141302, 0.057335, 0.019714, 0.040052, -0.159617, -0.20063, 0.119072, -0.176982, 0.104083, -0.014057, 0.145516, 0.087474, -0.206781, 0.036632], [-0.383561, -0.158247, 0.041258, -0.048457, 0.034002, 0.006234, 0.105314, -0.05891, 0.238626, -0.135749, 0.086747, 0.385954, 0.003407, -0.550324, -0.087594, 0.246705, -0.360565, -0.012854, -0.03967, 0.133885, 0.17802, 0.323355, -0.187528, -0.001158, -0.28312, -0.073293, 0.13539, 0.120357, -0.08614, 0.401543, -0.149949, -0.006894, -0.502055, 0.318023, 0.021863, 0.050891, -0.067406, -0.43092, 0.154126, 0.011877, -0.209879, 0.04914, 0.104828, 0.16896, -0.172975, 0.439915, -0.063298, -0.174535, 0.072455, 0.05408, -0.104711, 0.088756, 0.0655, 0.169946, -0.036566, -0.002335, 0.100252, 0.234836, 0.151242, -0.114935, 0.099694, 0.087831, 0.071452, 0.51435], [-0.985389, 0.020578, -0.224246, -0.150462, -0.018019, -0.595647, 0.025864, -0.483997, -0.024224, -0.11166, -0.029371, -0.156443, -0.174779, -0.086354, -0.477916, -0.077849, -0.313679, -0.105236, -0.596617, -0.41741, 0.295962, -0.09466, 0.098673, 0.048749, 0.262629, 0.051507, 0.068286, 0.349428, -0.044077, -0.078903, -0.504948, -0.019207, 0.001154, -0.091844, 0.089891, 0.015446, 0.117238, 0.188665, -0.02733, 0.038303, 0.21782, -0.089122, 0.088079, -0.042591, 0.274834, -0.194667, -0.106768, -0.224154, -0.54593, 0.101739, -0.025539, -0.107983, 0.269755, -0.276126, 0.052122, 0.109143, 0.0978, -0.375614, -0.262816, 0.046059, -0.012985, -0.351835, -0.381819, 0.024975], [-1.287016, 0.012566, -0.060469, -0.18189, 0.045404, -0.012224, -0.20548, 0.305317, -0.032206, -0.1944, -0.235046, -0.160709, 0.152248, -0.19383, 0.075849, -0.106679, -0.218886, 0.001993, 0.191727, -0.274235, -0.143393, 0.272001, 0.068571, -0.04796, -0.405267, -0.053422, 0.152124, 0.179311, -0.036143, 0.057245, 0.160041, 0.193402, -0.017329, 0.344732, 0.075978, -0.288864, -0.2713, -0.04412, 0.278375, 0.008339, -0.204434, 0.292776, 0.053561, 0.181577, 0.101388, 0.020867, 0.323899, 0.057385, -0.06602, -0.027326, 0.253518, 0.021134, -0.041484, 0.61948, -0.15724, -0.374024, -0.090669, 0.148484, 0.149801, 0.014071, -0.242863, 0.03391, -0.549462, -0.003349], [-0.125845, 0.147479, 0.002423, 0.22135, 0.061586, 0.088552, 0.03342, 0.131859, -0.377213, -0.27415, 0.123269, -0.19729, 0.210223, -0.029552, 0.095861, 0.084856, 0.126331, -0.203269, 0.040913, -0.049815, 0.049359, 0.286656, -0.253596, 0.04296, 0.186546, 0.242363, 0.208876, -0.250696, -0.125647, 0.032774, 0.235006, 0.096183, -0.130597, 0.268854, 0.097099, -0.17918, 0.123032, 0.160301, 0.261959, 0.060285, -0.060499, 0.179421, -0.211665, -0.03281, -0.321958, -0.962831, 0.108044, -0.12292, 0.255661, 0.128433, 0.060383, -0.163307, 0.071681, -0.036544, 0.147128, -0.19791, -0.266139, -0.016588, -0.097629, -0.128265, -0.180787, 0.144193, -0.032456, -0.31008], [0.330865, 0.127743, -0.112742, -0.073753, 0.230221, -0.004686, 0.017443, -0.313934, -0.158839, -0.182159, -0.173088, -0.036449, -0.156738, -0.196506, 0.111952, -0.070624, -0.169539, 0.255868, 0.301789, -0.265017, -0.152799, 0.368584, 0.114117, -0.005995, 0.3497, 0.000216, -0.108247, -0.161451, 0.25925, 0.113715, -0.065927, -0.097803, -0.13488, -0.144131, -0.222068, -0.028876, 0.078831, 0.225338, 0.266057, 0.017361, -0.028296, -0.168928, 0.245508, -0.074789, -0.263603, 0.091098, -0.157502, 0.010151, -0.15338, 0.062984, -0.041286, 0.083655, -0.032862, -0.217548, 0.083094, 0.010529, -0.283203, -0.252998, -0.137219, 0.004547, -1.277218, 0.284304, 0.151976, 0.103129], [0.130698, 0.241014, -0.236329, 0.325248, 0.264587, 0.168655, 0.365057, 0.215493, 0.246212, -0.442163, 0.362208, 0.265961, 0.097254, 0.240969, 0.087392, -0.226064, 0.084609, 0.134185, 0.360422, 0.205912, -0.274409, 0.133427, -0.214125, -0.244167, 0.005477, 0.053517, -0.032446, 0.206583, 0.026631, 0.151617, 0.312711, 0.003808, 0.010059, -0.014373, -0.313609, -0.006669, -0.038538, -0.064676, -0.084406, 0.113995, 0.023907, 0.543068, 0.296155, 0.164584, -0.258109, -0.466396, 0.150977, 0.043382, 0.184103, 0.18628, -0.003203, 0.351395, 0.045425, -0.021147, 0.049764, 0.113845, -0.017028, 0.103512, -0.073384, 0.069446, 0.195981, 0.398854, -0.171106, 0.316082], [0.097874, -0.015758, -0.426924, 0.183205, 0.113885, 0.054955, 0.247138, 0.149796, -0.239479, 0.081557, 0.057982, 0.156204, -0.010496, -0.023438, -0.279249, 0.093373, -0.140252, 0.029847, -0.098982, 0.166333, 0.436414, -0.028693, -0.062854, -0.260036, 0.255407, -0.024659, -0.127809, 0.139821, 0.289721, 0.152136, 0.034596, -0.007466, 0.481455, 0.045787, -0.349119, 0.296939, 0.099664, 0.022147, 0.114281, -0.018291, 0.078454, -0.201508, -0.092297, -0.093388, -0.414252, -0.027537, 0.213067, -0.022594, 0.059882, -0.074991, 0.112991, -0.028167, -0.373714, 0.059742, 0.085999, 0.019323, 0.00349, -0.005596, -0.214448, 0.090347, 0.348341, 0.047476, -0.061073, -0.086652], [-0.28073, -0.171787, -0.109606, -0.363338, 0.169077, 0.225841, 0.080198, -0.123455, 0.237993, 0.093852, -0.477458, -0.145073, 0.200601, -0.178939, 0.020529, -0.082434, -0.085606, -0.033464, -0.000675, 0.233115, -0.103669, 0.146771, 0.120003, -0.197277, -0.004252, 0.224647, -0.088491, 0.194373, 0.105732, -0.519134, 0.138812, -0.069335, -0.087609, -0.164576, -0.132306, 0.01404, 0.340143, -0.178208, 0.210358, -0.088764, 0.161217, 0.02753, 0.05332, -0.08164, 0.139772, 0.068813, 0.155746, -0.020399, 0.270255, 0.077027, -0.020377, -0.237347, -0.069468, 0.025169, 0.165594, -0.192536, 0.10222, 0.184122, -0.08388, -0.145862, -0.298153, 0.141824, 0.059242, 0.21095], [0.214117, -0.364161, 0.005606, -0.07552, 0.034018, 0.423562, -0.257069, 0.010915, 0.001104, -0.160817, -0.21171, 0.156191, -0.000392, -0.096225, -0.109843, 0.088066, 0.08654, -0.003398, 0.041708, 0.121356, 0.223477, 0.039305, 0.15354, -0.022499, -0.056802, 0.070349, 0.315311, -0.174588, 0.131324, 0.280207, -0.122869, -0.178599, -0.382427, -0.15841, 0.127119, 0.041975, 0.018915, -0.389579, 0.445406, 0.021625, -0.205987, 0.05154, -0.148755, 0.299765, -0.015408, 0.03251, 0.229764, 0.092572, -0.181268, -0.221901, 0.155218, 0.176776, -0.22483, 0.142025, -0.133631, 0.295878, -0.248279, 0.170517, -0.019394, -0.129663, 0.393243, 0.120856, -0.132615, 0.069429], [0.113332, -0.067205, -0.05069, -0.198401, 0.080694, 0.072326, 0.10343, 0.169648, 0.057356, -0.135214, -0.076138, -0.128686, -0.047198, -0.220772, -0.076605, -0.08523, 0.136888, 0.257935, 0.259799, -0.30168, -0.082703, 0.063401, -0.015043, 0.218124, 0.051101, -0.117118, -0.17981, 0.322018, -0.316826, -0.028642, 0.161043, -0.035202, 0.164722, -0.083616, -0.066268, -0.231946, -0.187175, 0.089876, -0.033753, 0.303434, 0.019261, -0.18052, -0.085831, 0.655574, -0.145592, 0.076747, -0.240672, -0.182567, 0.039735, -0.002095, 0.102445, -0.064791, 0.069552, -0.009691, 0.05516, -0.144084, -0.277275, 0.117043, -0.065651, 0.028197, -0.252585, 0.054968, -0.119277, -0.025917], [0.354994, 0.066203, -0.555061, -0.069236, 0.064395, 0.621348, 0.258423, 0.217225, 0.505468, 0.350025, -0.241422, 0.086392, -0.451398, -0.143717, 0.234689, -0.255296, 0.300767, -0.087334, -2.148304, 0.086726, 0.041221, 0.412982, 0.246658, 0.137984, 0.058237, 0.138977, -0.238657, -0.947408, 0.512448, 0.566521, 0.567611, 0.247833, 0.098404, -0.019566, 0.489953, -0.161313, 0.340546, -0.093338, 0.708899, 0.094851, -0.08905, -0.105401, 0.193252, -0.482365, -0.234145, -0.18259, 0.536751, -0.036333, 0.278907, -0.001739, -0.375177, 0.051061, 0.034595, -0.082256, -0.265494, -0.127943, -0.106648, -0.082942, 0.083258, -0.137558, 0.228175, 0.05351, -0.289509, -0.225222], [-0.777928, -0.26103, -0.097547, 0.024581, -0.046536, 0.1441, -0.090417, 0.316284, -0.736817, -0.120763, -0.315536, -0.21254, -0.003276, 0.036563, 0.06153, -0.764881, -0.219991, 0.101106, -0.104132, 0.105808, 0.044376, 0.287201, -0.553404, -0.082354, -0.148505, -0.04524, -0.10121, -0.875736, -0.18973, -0.073631, -0.147148, -0.034217, -0.322249, -0.172917, -0.485174, -0.288296, 0.040434, 0.23919, -0.127983, -0.174192, 0.05209, -0.041214, -0.140284, 0.123487, -0.158503, -0.835491, 0.067054, -0.057277, 0.038994, 0.069224, -0.068936, 0.20921, 0.072727, -0.107284, -0.15276, -0.165339, 0.188499, -0.413636, 0.24433, 0.108704, 0.126243, -0.066074, -0.472308, -0.080768], [-0.475983, 0.092063, -0.196482, -0.396335, -0.693373, 0.057089, 0.081588, 0.093008, 0.157442, 0.293681, 0.069401, 0.000992, 0.055866, -0.192906, -0.53337, 0.218045, -0.10826, 0.483842, 0.074494, -0.485827, -0.131171, 0.310305, 0.076202, 0.044931, -0.255264, 0.071927, 0.357321, 0.065851, 0.023703, 0.137591, -0.109293, -0.19767, -0.23471, 0.084468, -0.201431, 0.265972, -0.051708, 0.226547, 0.070716, -0.524243, -0.211492, -0.04038, -0.418031, 0.184637, 0.11639, -0.012992, 0.155558, -0.074172, -0.044662, 0.028541, 0.208522, -0.225469, 0.003177, -0.176688, 0.001226, -0.301513, 0.235038, 0.076464, -0.080869, -0.104289, -0.389176, 0.153699, 0.299216, 0.132821], [-0.080718, -0.017235, -0.230252, 0.138248, -0.003063, 0.146142, 0.226816, -0.13794, -0.045508, 0.157192, 0.074305, -0.169271, -0.027338, 0.237688, -0.206168, 0.37892, -0.008304, 0.137995, 0.07243, 0.160406, -0.040429, 0.063902, -0.225782, -0.215923, 0.304381, 0.038858, 0.097068, -0.18084, -0.185529, -0.057628, 0.145319, -0.025407, -0.323974, -0.052276, 0.0981, -0.039096, 0.008385, -0.079552, 0.07485, 0.018485, 0.006923, 0.101431, -0.014228, -0.109455, -0.240846, -0.047145, -0.260285, 0.152475, -0.014189, 0.148553, 0.226297, 0.099645, 0.036722, -0.10738, -0.064129, 0.054424, 0.140887, -0.08528, 0.106478, 0.116298, -0.014523, -0.223432, -0.048077, -0.330234], [0.021633, -0.11332, -0.008836, -0.090256, -0.186581, -0.173301, -0.563659, -0.757098, 0.048595, -0.089559, 0.150157, -0.262816, -0.19877, 0.255493, -0.448852, -0.019183, -0.087398, 0.099447, 0.020815, -0.2665, 0.064709, -0.45298, 0.227038, -0.15823, -0.076014, -0.135039, 0.008486, 0.362725, -0.018381, -0.309238, -0.131818, 0.029879, -0.116768, 0.026195, 0.296746, -0.595553, -0.197406, 0.226772, -0.017736, -0.015921, 0.32764, -0.024477, 0.136065, -0.214638, 0.548836, 0.176665, -0.107289, 0.028241, -0.143439, -0.233837, -0.187261, -0.13741, 0.030775, -0.207214, 0.260486, -0.255539, 0.078362, -0.178416, -0.057857, -0.314433, -0.385653, -0.172993, 0.34466, -0.314539], [-0.770761, 0.212021, 0.071752, -0.071154, -0.009284, 0.10918, 0.135817, -0.019666, -0.106619, -0.085382, -0.382807, 0.248366, 0.004312, -0.263469, -0.135245, 0.218602, 0.184701, -0.028577, -0.110888, 0.104142, 0.048768, 0.116295, -0.141571, -0.091893, 0.201042, -0.262658, -0.211713, -0.271734, 0.0083, 0.10206, 0.289035, 0.057082, -0.211548, -0.166675, -0.086612, -0.163276, -0.035199, -0.113847, -0.241316, -0.151825, -0.176014, 0.140018, -0.320701, 0.464926, -0.047653, -0.012101, 0.15137, 0.167063, -0.21697, 0.03503, -0.064978, 0.034527, 0.155298, -0.141532, 0.108952, -0.057567, -0.015281, -0.060995, 0.043403, 0.335744, 0.317418, 0.123106, -0.313304, 0.040242], [-0.450572, -0.125083, -0.0953, 0.100556, -0.120294, -0.138288, 0.057223, -0.924438, 0.209961, 0.03978, -0.041856, 0.167107, -0.015534, -0.112378, -0.104382, -0.158174, 0.06814, -0.183201, 0.013599, 0.128688, -0.032722, -0.540959, 0.270165, 0.065095, -0.217808, -0.022304, -0.168162, 0.34671, -0.092836, 0.37389, -0.149105, -0.165276, 0.190172, -0.2033, 0.04834, -0.03286, -0.352554, 0.076071, 0.390925, 0.194985, -0.161943, 0.096878, -0.233761, -0.761047, 0.004459, -0.054242, 0.086649, 0.055294, 0.373506, -0.407584, -0.094837, -0.22603, 0.299795, -0.822797, 0.079813, 0.001786, -0.188416, -0.113077, -0.147216, 0.064781, 0.235935, -0.074268, -0.075264, 0.205844], [-0.428488, 0.010808, -0.231628, -0.130844, -0.455953, 0.180002, -0.235394, -0.262803, 0.248252, -0.074643, 0.290102, 0.193997, 0.155749, 0.09397, -1.206288, -0.069645, -0.12578, 0.007798, -0.437396, 0.33904, -0.224956, 0.141731, 0.092094, -0.068266, -0.412786, 0.030056, -0.216205, -0.48203, -0.363859, -0.368024, -0.072902, -0.146548, 0.061927, -0.72981, -0.153485, 0.033342, 0.1357, -0.028291, 0.076923, -0.113894, 0.002508, 0.116917, -0.05936, -0.684677, -0.053624, 0.30814, -0.663858, -0.02687, 0.316555, -0.336035, -0.049591, 0.093502, -0.014697, 0.253868, -0.155485, -0.390177, 0.064265, -0.064198, -0.258056, 0.075396, -1.354887, -0.115667, 0.490993, -0.090953], [0.117815, 0.016314, 0.049519, 0.033977, -0.241374, -0.006417, -0.141801, -0.460108, 0.029026, 0.080417, 0.111412, 0.113232, 0.051884, 0.120161, 0.069571, 0.251312, 0.037216, -0.288418, -0.212689, -0.144419, 0.109308, -0.099995, -0.300528, -0.037847, -0.103105, -0.618708, 0.414804, -0.299828, -0.254407, -0.143799, -0.265657, -0.063309, 0.159653, 0.095125, 0.018354, 0.099881, -0.135727, 0.194153, -0.318903, -0.026736, -0.221768, 0.136444, 0.080493, -2.18018, -0.001734, 0.101115, -0.988636, -0.121773, -0.044588, -0.278807, -0.07334, 0.231891, -0.036707, -0.366428, 0.167125, -0.043684, -0.240956, 0.368131, -0.045497, 0.053004, 0.09034, 0.108936, -0.307899, -0.292713], [0.077112, 0.051192, 0.053181, 0.215507, 0.015808, 0.033223, -0.047005, 0.163404, -0.297137, 0.090623, 0.008842, -0.014923, 0.053778, 0.09616, 0.184586, -0.115979, -0.16666, -0.014976, -0.074294, 0.250979, 0.014174, 0.231292, 0.083245, 0.231626, -0.734838, -0.107715, 0.038423, -0.733984, 0.216208, -0.075183, -0.088957, 0.24483, -0.631871, -0.014122, 0.324833, -0.030431, -0.089549, 0.051137, 0.041688, 0.035219, -0.071998, -0.383941, 0.242534, 0.116868, 0.08005, 0.187586, 0.030406, 0.148165, 0.179733, -0.136046, 0.061372, -0.184441, -0.070382, 0.070418, 0.311404, 0.050473, -0.092113, 0.243364, 0.148827, -0.46128, -0.096032, 0.127519, 0.201777, -0.403855], [0.287761, -0.20386, 0.173172, -0.317846, -0.054674, -0.079088, 0.129143, -0.168713, 0.057955, -0.24893, -0.38035, 0.075586, -0.063863, 0.186107, 0.007126, -0.120589, -0.021746, -0.033631, -0.082778, -0.066553, 0.229312, 0.03272, 0.031182, -0.101958, -0.120112, -0.237794, -0.406121, 0.118988, 0.23222, -0.437371, 0.195961, 0.015421, 0.279793, -0.225899, -0.024944, -0.386554, -0.132509, -0.065661, -0.376869, -0.592326, 0.075282, -0.031219, 0.256445, -0.418564, 0.171971, 0.165719, -0.16789, -0.124486, -0.034279, -0.27927, 0.176934, -0.49586, 0.085813, 0.143509, -0.19754, -0.279555, -0.288549, -0.135767, 0.191422, 0.14471, 0.072743, -0.021606, -0.306296, -0.090828], [-0.127415, -0.186303, -0.150727, 0.046275, 0.026317, 0.043461, 0.099364, -0.102088, -0.092496, 0.012746, 0.010727, -0.059609, 0.033355, -0.171413, 0.181029, -0.386325, -0.15207, 0.042007, -0.217677, -0.08036, -0.043298, -0.160136, 0.114567, 0.034764, 0.487792, -0.10359, -0.143213, -0.31799, -0.313122, -0.016185, -0.285914, -0.118336, -0.407526, 0.242367, 0.141578, 0.272271, 0.164631, -0.249411, -0.163204, -0.073394, 0.066875, -0.007502, -0.238733, 0.033899, 0.197233, 0.134248, -0.420574, -0.36587, 0.547634, -0.273883, -0.22131, 0.128153, -0.168179, 0.140061, -0.234874, -0.130062, 0.339949, 0.069339, -0.211089, -0.219444, 0.471878, 0.126846, 0.076867, -0.172722], [0.206819, -0.061094, 0.022929, 0.187583, -0.002702, 0.169122, 0.313247, 0.04376, -0.227259, 0.239744, -0.035094, -0.175501, 0.274695, 0.365282, -0.124204, 0.232402, 0.064287, 0.13963, -0.05362, 0.257673, 0.107297, 0.305839, -0.031972, -0.009475, 0.203544, -0.103259, -0.250471, -0.576431, -0.219595, -0.158295, 0.408759, -0.219057, -0.01062, 0.177451, 0.156754, -0.019014, 0.028992, 0.191698, 0.093868, -0.223947, 0.103551, -0.275968, 0.014925, 0.053037, 0.446398, -0.385248, 0.070985, -0.200289, 0.012662, 0.103539, 0.1202, -0.183509, -0.180991, 0.06658, 0.017394, -0.024676, -0.207063, 0.728907, -0.106643, 0.101554, -0.170064, 0.149272, -0.107086, 0.117928], [-0.259968, -0.023303, 0.122684, -0.233881, 0.042782, -0.098658, -0.412219, 0.001468, -0.355598, 0.268358, 0.038783, 0.085339, 0.116764, 0.157163, -0.193937, 0.394523, -0.001984, -0.211281, -0.144492, 0.091402, 0.020837, -0.223045, 0.1045, -0.03527, 0.182375, -0.028817, -0.157477, -0.014577, 0.099964, 0.054342, -0.243882, 0.055821, 0.249902, -0.303464, 0.110519, -0.193527, 0.243867, 0.217388, 0.200147, 0.093411, -0.026055, -0.35002, 0.20025, -0.317737, -0.153013, 0.017945, 0.22173, 0.104836, 0.149293, -0.133449, 0.174814, -0.323436, 0.178569, 0.021411, 0.19928, 0.129732, 0.152031, -0.365178, 0.046937, -0.067367, -0.546494, 0.024842, -0.203982, -0.257049], [-0.140609, -0.034477, -0.06736, -0.025521, -0.007047, 0.001041, 0.090919, 0.301485, -0.130383, -0.075842, 0.063981, 0.13168, -0.332926, 0.509271, -0.409048, 0.007733, -0.14509, 0.085341, -0.350008, -0.140103, -0.065318, 0.207909, -0.038978, -0.131877, 0.158509, -0.03558, 0.214977, 0.083313, -0.190576, -0.137454, -0.020072, -0.093878, -0.061363, 0.516454, -0.130306, -0.304883, 0.09337, -0.007508, -0.209814, -0.200386, -0.27235, -0.03713, -0.006765, -0.385803, 0.229367, 0.145641, 0.328707, 0.05062, -0.109727, 0.014399, -0.180965, 0.059117, 0.028396, -0.285431, 0.145493, 0.059516, 0.37522, -0.268106, 0.25817, -0.107635, -0.979292, -0.003733, 0.397041, 0.210911], [-0.416171, 0.073767, -0.06622, -0.211507, -0.25145, -0.013372, -0.097903, -0.557625, -0.047904, 0.205417, 0.511843, -0.239291, 0.238592, 0.088672, 0.240969, -0.412094, -0.430837, 0.224244, 0.368688, -0.027829, 0.075875, -0.177845, -0.161025, -0.1304, 0.350603, -0.054257, 0.108681, -0.352664, 0.349928, 0.00683, -0.082303, -0.078812, -0.110603, 0.03964, 0.021665, 0.297807, 0.102929, 0.27947, 0.112444, 0.118567, 0.118777, 0.051161, -0.516696, 0.475437, -0.091798, 0.00698, -0.006491, -0.186187, 0.097216, 0.400571, -0.167394, -0.160443, 0.048268, -0.054582, 0.024292, -0.071466, 0.240493, 0.254093, -0.062656, -0.320764, -0.057685, -0.557638, -0.098387, -0.056241], [0.893789, -0.277079, 0.378483, -0.340028, -0.268249, 0.310445, 0.033439, 0.103422, 0.296271, -0.217517, 0.033099, -0.113882, -0.012173, -0.304535, 0.141146, -0.148548, -0.235957, 0.053112, -0.878112, -0.030681, -0.054057, -0.168886, 0.04505, 0.150197, -0.877318, 0.250526, -0.120248, -0.34589, -0.045764, 0.001991, -0.301926, -0.142389, -0.923091, -0.181976, 0.064116, -0.011244, -0.127959, 0.032273, 0.116088, 0.103734, 0.252038, -0.02582, 0.064753, -0.377928, 0.526451, 0.196125, -0.115884, -0.363627, -0.72959, 0.003943, -0.179336, -0.202156, 0.167847, 0.319728, -0.198918, 0.080914, -0.173596, 0.058311, -0.314147, 0.149056, -0.035061, -0.297012, -0.043577, -0.168057], [0.264743, 0.279319, -0.088241, -0.432852, -0.158833, 0.143213, 0.279204, 0.084287, 0.106928, 0.040614, 0.047264, -0.026228, 0.24626, -0.161338, 0.102625, -0.024485, 0.076308, -0.06055, 0.174563, 0.08775, 0.212714, 0.245244, 0.257391, -0.010944, 0.213513, 0.275196, 0.015624, -0.067765, 0.252224, -0.124425, -0.074297, 0.134413, -0.052954, -0.201455, 0.127111, 0.06129, 0.146744, -0.064877, -0.210832, 0.255201, -0.071743, 0.186561, -0.512245, -0.659212, 0.316633, 0.105783, -0.40305, -0.251052, -0.191669, -0.000754, 0.270694, 0.150498, -0.019417, -0.341745, 0.072082, 0.275116, 0.089203, -0.087503, 0.572279, 0.057522, 0.320203, -0.211839, 0.058375, -0.071762], [-0.004598, -0.115283, 0.174352, -0.232406, -0.097917, -0.124658, 0.068757, -0.395706, 0.1935, 0.120774, 0.342427, 0.00352, -0.298209, -0.196691, 0.266298, -0.727859, -0.519807, -0.063668, -0.627028, -0.416098, 0.024767, -0.439436, -0.572303, 0.252866, 0.285467, -0.059423, 0.152604, -0.493611, -0.03508, 0.270785, 0.055158, -0.426014, 0.05184, -0.24909, -0.293123, 0.218734, -0.172348, 0.316557, 0.055989, -0.123701, 0.066645, -0.016191, 0.023333, 0.029423, 0.154811, 0.176365, -0.098898, -0.02649, 0.142028, -0.110973, 0.071018, -0.075111, 0.185451, 0.000872, 0.014995, 0.405179, -0.055823, -0.025669, 0.126303, 0.091329, -0.112923, 0.021162, 0.037167, -0.412564], [-0.138541, 0.390855, 0.161979, 0.01808, 0.208576, -0.317378, 0.003863, -0.080449, -0.101994, -0.176488, -0.245944, -0.040775, 0.061345, -0.22515, -0.547235, 0.233204, -0.134744, 0.036352, -0.153333, -0.369594, -0.127769, 0.069311, 0.259958, 0.31661, 0.036204, 0.188615, -0.176441, 0.18151, 0.053374, -0.018513, -0.302103, -0.013193, 0.220724, -0.017277, 0.351796, -0.135418, 0.375966, -0.505529, -0.266109, 0.080182, -0.151337, 0.200481, -0.575618, 0.219377, -0.304993, 0.259644, 0.160894, 0.048603, 0.137786, -0.068762, 0.040881, 0.043109, -0.010666, 0.068655, -0.087683, -0.168386, -0.286821, 0.246877, -0.012111, 0.145522, -0.24206, 0.145584, 0.028377, -0.197935], [0.212478, -0.36867, 0.171346, -0.132755, 0.036997, 0.205864, 0.336181, 0.034828, 0.094505, -0.113576, 0.011974, -0.241469, -0.27095, -0.087944, 0.007477, 0.2437, -0.157575, 0.008088, -0.104584, 0.069013, -0.232052, 0.086362, 0.05496, -0.116668, 0.22855, -0.004877, 0.056404, -2.065518, -0.131257, -0.094665, -0.05013, -0.517464, 0.078527, -1.12933, 0.006217, -0.034375, -0.100222, -0.09162, -0.271656, 0.149119, -0.329532, -0.237881, 0.066335, 0.408401, 0.023384, -0.327117, 0.081914, -0.091575, -0.128329, -0.254156, -0.013878, -0.343802, 0.173869, -0.628222, 0.22464, 0.241446, 0.32239, -0.077233, 0.034145, 0.160625, 0.24099, 0.249208, 0.247502, -0.181702], [-0.214672, -0.158198, 0.146897, 0.210166, -0.141703, -0.247195, 0.0141, 0.464759, -0.809049, 0.241065, -0.05821, 0.155397, 0.30222, -0.758507, -0.21716, 0.114122, 0.354555, 0.048654, -0.060396, -1.538497, 0.287448, -0.054809, -0.049277, -0.051047, 0.280292, -0.410735, -0.196074, 0.159916, 0.04825, 0.073501, -0.099857, -0.01465, -0.646753, 0.01693, -0.017257, 0.033437, 0.004698, -0.089289, 0.258294, 0.090162, 0.154221, 0.163267, -0.028726, -0.210503, -0.098551, 0.431177, -0.244605, -0.039763, -0.041536, 0.136486, 0.042758, 0.374468, 0.49403, 0.07832, -0.243817, -0.278075, -0.054946, 0.059379, -0.699583, 0.216461, 0.096436, -0.229221, -0.193646, -0.163606], [0.211911, 0.025055, 0.446618, -0.225866, -0.007307, 0.195665, 0.010578, -0.76778, 0.356245, 0.031735, -0.366027, -0.072009, 0.201685, -0.053629, 0.156955, -0.20217, -0.342659, 0.271347, -0.042258, -0.842408, -0.096964, -0.007879, -0.000954, -0.295876, 0.322469, 0.201504, 0.250956, -1.017935, -0.111192, 0.346166, 0.016278, -0.088092, 0.312948, 0.403294, -0.578743, 0.090184, -0.157352, 0.14141, -0.053084, -0.080386, 0.084131, -0.274154, -0.153028, 0.09741, -0.207683, 0.147464, -0.040347, 0.165531, 0.061123, -0.082292, -0.019306, 0.233635, -0.038554, 0.281026, -0.110045, -0.243119, -0.05944, -0.276967, 0.335495, -0.036921, 0.463332, -0.318766, -0.099882, -0.634934], [-0.131385, 0.230061, -0.004839, 0.016245, -0.206601, 0.07481, 0.151498, 0.076724, -0.498136, -0.12268, -0.002297, 0.187586, -0.130343, -0.228841, -0.234894, 0.017497, 0.275064, 0.178362, -0.118877, -0.009657, -0.072552, 0.158839, -0.199425, -0.147522, -0.203831, -0.104788, 0.004582, -0.256223, 0.132036, -0.059092, -0.113331, -0.362712, -0.263164, -0.058528, -0.254429, 0.157248, 0.039921, 0.078408, 0.154765, 0.221568, -0.042947, 0.051043, -0.152119, -0.053157, -0.021002, -0.213886, 0.107525, 0.045242, 0.204657, -0.137101, 0.206819, -0.022307, 0.148357, 0.045991, 0.29554, 0.204241, 0.339661, 0.07698, -0.158518, -0.011378, -0.369553, 0.090912, -0.116299, -0.268804], [0.558597, -0.22506, -0.334508, 0.062837, 0.208626, -0.121809, -0.158249, -1.48902, 0.27202, 0.09833, -0.273867, 0.122001, -0.237487, 0.160825, -0.29548, 0.033302, -0.096543, 0.204537, 0.13722, 0.143622, 0.198814, -0.01571, -0.373858, -0.244894, 0.171981, -0.034622, -0.061317, 0.350598, -0.225618, 0.311071, 0.322453, 0.053296, 0.42896, -0.476287, 0.199523, -0.285333, -0.003814, 0.278999, -0.074112, -0.130768, -0.043601, -0.086658, -0.187634, -0.249166, -1.148292, -0.31579, -0.188542, -0.261825, -0.094985, -0.271041, -0.009613, 0.102103, 0.032574, 0.287756, -0.078609, -0.335336, 0.097211, -0.244573, 0.428883, -0.134565, -0.550142, -0.299026, -0.328017, 0.024928], [-0.216519, 0.062039, -0.1893, -0.011648, -0.048156, 0.15031, 0.056262, 0.180625, 0.342184, -0.538793, -0.22983, 0.339616, -0.149204, -0.137837, 0.058971, -0.133809, 0.118638, 0.167873, -0.692737, 0.10122, 0.011184, -0.010718, 0.079727, -0.050166, -0.160777, -0.115646, 0.185455, 0.023024, 0.01941, 0.121143, -0.0665, 0.019519, -0.078243, 0.538359, -0.22721, 0.44741, 0.062672, -0.20185, -0.185673, -0.08759, 0.238574, 0.085972, 0.059515, -0.532361, -0.010338, -0.00858, 0.224425, 0.074825, 0.235773, 0.149542, -0.101206, 0.121404, -0.000924, 0.139307, 0.198667, 0.152506, -0.011624, 0.031122, -0.068423, 0.397449, -0.272956, 0.09922, 0.025004, -0.15426], [-0.482639, -0.215647, 0.040679, -0.073858, 0.296222, 0.14613, -0.120973, 0.062104, 0.077564, -0.38058, -0.086221, 0.09971, -0.132112, 0.022263, -0.06174, -0.011011, -0.232567, -0.3192, -0.639802, 0.15686, 0.300628, -0.032855, -0.486172, -0.152539, 0.066971, -0.13361, 0.073319, -0.511624, -0.171602, 0.08214, -0.105692, -0.550043, 0.487821, -0.107319, -0.643532, 0.165079, 0.27863, 0.149905, 0.038732, -0.035817, 0.327752, -0.074605, 0.217698, -0.22245, -0.318653, -0.163035, 0.158355, 0.116544, -0.203784, 0.204148, -0.165783, -0.075764, 0.063099, 0.094936, -0.245271, 0.188648, 0.207807, -0.059433, -0.157466, 0.132956, -0.023434, 0.089686, 0.075177, 0.126034], [-0.030771, 0.311482, 0.148751, 0.194281, 0.018518, 0.31222, -0.189776, 0.281853, -0.235318, -0.076365, 0.227504, -0.028788, 0.132624, 0.25485, -0.065765, -0.129629, -0.008604, 0.086407, 0.0736, 0.162962, 0.155069, -0.347626, -0.04984, 0.231126, -0.143901, -0.229259, -0.366719, 0.148694, -0.34652, -0.217886, 0.162854, -0.031154, -0.06928, 0.221686, 0.022831, -0.02974, 0.050741, 0.071365, -0.079128, -0.311134, 0.038856, 0.11396, 0.096298, -0.603675, 0.200335, -0.185132, -0.118593, 0.098878, -0.065656, -0.495306, -0.080142, -0.087335, -0.420212, 0.063439, 0.097938, 0.047403, -0.170906, 0.107176, 0.207921, -0.003723, 0.141858, -0.028259, 0.051934, -0.040504], [0.073123, -0.25751, -0.412157, -0.170428, 0.085124, -0.001746, -0.145396, -0.051225, -0.102871, -0.032695, -0.177511, 0.088153, 0.04247, -0.24128, -0.120055, 0.309651, 0.141064, 0.016976, 0.104997, -0.006417, 0.078323, 0.069562, -0.063109, -0.162092, -0.064504, 0.15607, -0.147264, -0.154703, 0.086914, 0.172643, 0.158828, 0.170728, 0.073393, 0.052934, 0.17986, 0.04952, 0.119594, 0.190135, -0.138287, 0.227612, 0.071117, -0.18604, 0.021917, -0.293391, 0.059815, -0.130549, 0.018487, -0.101321, -0.135377, -0.011748, 0.144318, -0.310306, 0.099286, 0.163911, -0.00293, -0.115447, -0.297473, 0.145549, 0.075463, 0.160607, -0.041844, 0.000373, -0.244699, 0.219839], [0.213049, 0.012996, 0.070282, -0.10829, 0.129157, 0.100451, 0.198036, -0.731948, 0.410444, -0.14037, 0.100678, 0.110028, -0.175862, 0.233466, 0.094828, 0.004249, -0.34006, 0.075775, 0.087471, 0.346911, -0.434959, -0.168728, 0.288077, -0.013027, 0.20043, -0.343057, -0.122441, 0.075625, 0.159124, 0.305301, -0.185683, -0.263797, 0.124538, -0.032254, -0.32267, 0.044885, -0.05085, -0.164917, 0.006011, -0.052179, -0.289514, 0.124831, -0.385673, -0.023511, 0.108105, -0.32777, 0.135462, -0.087268, 0.04826, 0.06316, -0.056912, 0.106304, -0.271304, 0.102515, -0.025997, 0.018956, 0.327573, -0.151995, 0.006404, 0.166828, -0.014747, 0.063135, -0.241346, 0.780772], [-0.116155, 0.119724, 0.059781, 0.072964, 0.154735, -0.43712, -0.354135, 0.112268, 0.113348, 0.257866, -0.104657, -0.125837, 0.12216, 0.204583, -0.003243, -0.740514, -0.030255, 0.042948, -0.039949, -0.064105, -0.067201, 0.334157, -0.144383, 0.165754, 0.239937, 0.100902, -0.266716, 0.059042, -0.208555, -0.283589, -0.176772, -0.084542, -0.286765, -0.106946, -0.201357, -0.093442, 0.166171, 0.113027, -0.331197, 0.288986, 0.08316, -0.289296, 0.123242, -0.831475, -0.15331, 0.033, 0.028416, 0.230291, -0.312099, -0.103775, 0.029055, 0.275639, 0.234689, 0.091926, -0.317364, 0.294199, -0.002929, -0.416633, 0.217132, -0.07389, -0.13698, -0.192093, 0.0741, -0.285792], [0.196034, -0.195855, 0.03925, -0.112864, 0.001119, 0.021847, 0.243702, 0.083054, -0.171707, -0.296087, 0.13216, -0.137396, 0.174814, -0.021059, -0.019514, 0.21603, -0.07417, -0.240088, -0.350616, 0.240097, 0.230084, -0.176137, 0.140772, -0.014776, 0.096954, -0.043616, -0.059162, -0.130719, 0.143916, -0.217882, 0.146018, -0.03667, -0.021426, 0.013841, 0.098468, -0.318629, -0.064936, -0.429415, -0.021393, 0.158455, 0.174692, 0.007776, 0.003425, -0.419285, 0.368597, 0.040364, -0.048629, -0.057175, 0.030565, -0.133824, 0.095217, -0.221536, -0.268238, 0.177666, -0.009804, 0.143232, 0.01849, -0.014521, -0.130893, -0.164295, -0.024357, 0.054905, 0.058983, 0.061056], [-0.119378, -0.281657, 0.2023, 0.182827, -0.178609, -0.133016, -0.042798, 0.026157, 0.012864, 0.206475, -0.007542, -0.264197, -0.134265, 0.146383, -0.036038, -0.211596, -0.154794, 0.065157, -0.284689, -0.079037, 0.045008, 0.040501, 0.082626, 0.220751, 0.29919, 0.075826, 0.225196, -0.532077, 0.000333, -0.157304, -0.105456, 0.024061, 0.166188, -0.218805, -0.103273, 0.110635, 0.16916, -0.132902, 0.178862, 0.163822, 0.364921, -0.415503, 0.23191, -0.14014, -0.221264, -0.12627, 0.041421, -0.227085, -0.22213, 0.103287, 0.221125, -0.771425, 0.134976, -0.125652, 0.023786, 0.11768, 0.170314, 0.093232, 0.190369, -0.054251, -0.197253, 0.03406, -0.027327, 0.010086], [-0.06962, -0.062819, 0.09617, -0.167482, 0.066728, 0.204015, 0.035744, -0.231716, -0.000711, 0.229044, 0.230258, 0.058112, 0.206038, -0.283883, -0.045495, 0.026653, -0.043459, -0.044839, 0.147757, 0.439295, -0.315912, -0.48384, 0.125132, 0.031767, 0.263228, -0.151566, 0.079507, 0.190335, 0.156473, -0.513312, 0.025816, 0.104809, -0.028786, 0.012624, 0.170446, 0.09933, 0.063112, 0.174715, 0.056816, 0.141291, 0.021718, 0.066456, 0.21841, -0.2357, 0.201907, -0.229078, 0.286726, -0.203065, -0.082197, -0.292979, -0.234457, 0.062206, 0.211934, 0.06268, -0.046424, 0.093471, 0.208621, 0.06199, -0.046143, 0.086727, -0.616235, 0.198232, 0.073266, -0.294166], [0.334973, -0.079259, 0.154729, 0.444824, 0.216351, 0.146152, -0.050781, -0.453687, -0.049184, 0.195177, 0.104199, -0.24531, -0.509244, -0.403433, -0.230758, 0.219264, -0.075202, 0.217418, -0.055634, -0.212662, 0.447037, 0.180715, -0.018074, -0.061526, -0.206132, 0.159473, -0.211496, 0.301706, -0.134594, 0.000253, 0.072861, -0.043582, 0.02208, -0.027833, -0.055795, -0.146689, 0.040917, -0.086059, 0.315552, 0.188962, -0.115746, 0.153935, 0.086737, -0.192925, 0.046741, -0.15356, -0.054697, 0.299244, 0.228784, -0.405663, 0.099039, -0.245231, -0.055848, -0.542279, -0.00609, -0.012923, -0.441411, -0.222431, 0.128883, 0.145285, -1.048317, 0.188429, -0.484181, 0.034195], [-0.056151, 0.150029, 0.019361, 0.15919, -0.101154, 0.226554, -0.020885, -0.12509, 0.194995, 0.049865, 0.030843, -0.165879, 0.052668, 0.12743, -0.025182, -0.043297, 0.022509, -0.247476, 0.355917, -0.062345, 0.087706, -0.303212, 0.122532, 0.153503, -0.158473, 0.038196, 0.317215, -0.140452, -0.00414, 0.0008, 0.123495, -0.130384, -0.84759, 0.449747, -0.192588, -0.176819, 0.186344, -0.007362, 0.241102, 0.125772, -0.188008, 0.150829, -0.151453, -0.191246, 0.250409, -0.119546, -0.077598, 0.094925, -0.140733, -0.139302, -0.188052, -0.108083, 0.067545, 0.088164, -0.112602, -0.03563, 0.065048, -0.223892, -0.059448, -0.20737, 0.099867, -0.036834, 0.142319, -0.210602], [0.163897, 0.440562, -0.482634, -0.167431, 0.367727, 0.052359, -0.234358, 0.340403, 0.174346, 0.232597, -0.10186, 0.356424, 0.054214, -0.15734, -0.471212, 0.077235, 0.475754, -0.199636, -0.036521, -0.056201, -0.091544, -0.052319, 0.265729, -0.031202, -0.010128, -0.032263, -0.021814, 0.057912, 0.247294, 0.061882, -0.093536, 0.014634, -0.106366, -0.077348, 0.139011, -0.058743, -0.182845, 0.072493, 0.249931, 0.008582, 0.070529, 0.185772, 0.126879, 0.128835, -0.023663, -0.04297, -0.107526, -0.422334, -0.106386, 0.354831, -0.035355, -0.027967, -0.313136, -0.559659, -0.006751, 0.246175, -0.265178, 0.017987, -0.173316, 0.193022, -0.202881, 0.16051, 0.022682, 0.310226], [0.142235, -0.338916, 0.225038, 0.244116, -0.24169, 0.285351, -0.095278, 0.056798, -0.228393, 0.120835, 0.172307, -0.112411, -0.25578, 0.128518, -0.557071, 0.22347, 0.288895, -0.027775, -0.023911, -0.268902, -0.16102, 0.143139, -0.052636, -0.132361, -0.297876, -0.029783, 0.096674, -0.340822, 0.053552, -0.079043, 0.217155, -0.3363, 0.085144, -0.128703, 0.149528, -0.136375, -0.125048, 0.137522, 0.014064, 0.15041, -0.033983, 0.266505, -0.014874, -0.107064, -0.258818, -0.462659, -0.085363, -0.073872, -0.207527, -0.018255, -0.065001, 0.324893, -0.096471, -0.209416, 0.187704, 0.125373, 0.268927, -0.252945, -0.075272, 0.235899, -0.229201, 0.083892, -0.468994, 0.310546], [0.191617, 0.268858, -0.265629, -0.232942, 0.079187, 0.14212, 0.074347, -0.137905, 0.041639, 0.309908, 0.088589, 0.274502, 0.132549, -0.036557, -0.03667, 0.094563, -0.165137, 0.246562, 0.242473, -0.094477, 0.030173, 0.481467, -0.097292, -0.484589, -0.030017, -0.164531, -0.075008, 0.204847, 0.210957, 0.34729, 0.027328, -0.054379, 0.286871, 0.247593, -0.160496, 0.180531, 0.014121, 0.256422, 0.109001, -0.172883, -0.205868, 0.213542, -0.057235, 0.017432, -0.224769, 0.052206, 0.355458, 0.153568, -0.292845, -0.033179, -0.337474, 0.409249, 0.159925, -0.043852, -0.116231, -0.118277, 0.348345, -0.327239, 0.143797, -0.198606, -0.906712, -0.141281, -0.185632, 0.35618], [-0.271204, -0.215922, -0.459958, -0.309433, 0.029832, 0.074352, -0.123112, -0.315954, 0.153565, -0.047216, 0.088231, 0.36854, -0.163905, 0.262593, 0.049504, 0.192988, 0.078196, 0.089772, -0.035207, -0.02883, -0.073778, 0.256933, 0.042081, -0.17552, -0.091627, 0.258456, 0.132238, -0.260846, 0.072982, 0.195423, 0.131364, -0.126479, -0.037982, -0.37784, -0.296709, 0.015478, -0.103254, 0.122313, 0.020755, -0.17914, -0.130973, 0.006108, 0.159426, 0.166914, -0.12794, 0.02428, -0.111274, 0.310397, 0.132253, 0.330218, 0.11842, -0.107055, 0.115331, -0.321966, 0.152094, -0.112312, -0.024563, -0.133524, -0.056405, -0.127243, -0.066505, 0.229196, -0.188614, -0.096439], [0.162451, 0.136963, -0.068486, 0.377369, 0.073912, -0.095028, 0.102688, -0.197709, 0.260059, -0.110663, -0.218869, 0.150601, 0.039723, 0.001867, 0.17408, 0.00154, 0.140215, 0.105758, 0.20508, 0.091819, 0.431532, 0.112411, -0.059931, 0.0972, 0.086334, -0.284545, -0.098759, 0.173521, 0.000543, -0.014055, 0.116181, -0.130944, 0.091667, -0.301718, -0.047654, 0.167066, -0.145209, -0.22278, -0.142456, -0.243211, 0.070216, -0.29074, -0.233146, -0.397047, -0.054236, -0.0845, 0.110436, 0.232436, -0.472373, -0.399316, -0.218456, -0.15349, -0.176307, 0.112749, -0.162076, 0.023989, -0.467561, -0.051369, 0.030143, 0.310278, -0.451587, 0.110838, 0.338189, -0.054339], [0.036865, -0.114043, -0.141613, -0.405174, 0.17615, -0.255272, -0.144574, -0.000474, 0.058261, -0.078578, -0.191697, 0.118159, 0.142336, -0.019023, 0.021821, -0.062194, -0.160743, 0.264692, -0.098016, 0.036182, 0.058724, -0.133425, 0.150817, -0.199211, 0.177208, -0.152196, 0.018643, -0.094466, 0.14459, -0.080703, 0.133728, 0.145123, -0.006935, -0.343292, -0.037993, -0.280473, -0.41807, 0.229815, -0.161283, -0.449827, 0.111937, 0.032652, 0.119917, -1.090145, 0.062732, -0.061848, 0.071073, 0.074178, -0.149019, -0.016454, -0.221544, -0.20348, 0.119583, 0.114798, -0.048258, -0.096508, -0.162087, -0.248858, -0.143233, 0.001374, 0.363487, -0.032225, -0.000971, 0.025472], [0.108276, -0.191692, -0.069517, -0.320872, -0.121421, 0.375827, 0.121144, 0.208542, -0.108214, -0.056421, 0.404498, 0.15817, -0.098788, 0.274888, 0.01683, 0.083017, -0.295268, -0.031962, -0.583303, 0.104572, -0.006363, -0.236996, 0.042255, -0.008982, -0.274503, 0.208081, -0.428488, 0.243893, -0.023374, -0.061235, 0.085886, 0.125387, -0.339298, -0.236068, 0.017168, -0.316919, -0.077736, -0.052189, 0.190686, -0.224362, -0.08483, 0.121421, -0.065474, 0.527868, 0.145771, 0.306036, -0.031051, 0.133061, -0.166493, -0.097883, 0.04882, -0.160459, -0.08738, -0.118989, -0.233607, -0.014873, 0.066746, -0.110509, -0.109918, 0.011785, -0.508401, -0.021111, -0.047442, 0.088175], [-0.000526, -0.050574, 0.138172, 0.32565, 0.104644, -0.207543, -0.070848, -0.029453, 0.231619, -0.012917, 0.247375, -0.063188, -0.053821, 0.080969, 0.036829, -0.104151, -0.136965, -0.162045, 0.330192, 0.103872, -0.102767, -0.110647, 0.0631, -0.289892, 0.167072, 0.007419, -0.33561, -0.141883, -0.086472, 0.158399, -0.173353, 0.194318, 0.395418, 0.144739, 0.185685, -0.458385, -0.273754, -0.171662, -0.144022, 0.263679, 0.122212, -0.032228, -0.250928, 0.305578, -0.05994, -0.119444, -0.136149, 0.177033, 0.114246, 0.070437, 0.080061, -0.00134, -0.130859, 0.635561, -0.154111, 0.10772, 0.079099, -0.159864, -0.125074, -0.075927, 0.528678, -0.002224, 0.078527, 4e-05], [0.015274, -0.105261, -0.08513, 0.0268, 0.122517, -0.207632, 0.053239, -0.013446, 0.125125, -0.344785, -0.081944, 0.037435, -0.059661, 0.194073, -0.233734, 0.485987, 0.243129, -0.179547, 0.454536, 0.167493, 0.144073, 0.048963, 0.336177, -0.107331, -0.276378, -0.070449, 0.204083, -0.069266, -0.298239, 0.004675, 0.241148, -0.057558, 0.349829, -0.084767, -0.348785, -0.142309, 0.082143, -0.02981, -0.05661, 0.114723, 0.19862, 0.024224, -0.932223, 0.038816, 0.181636, -0.154548, -0.444779, -0.123827, -0.227582, 0.491503, 0.178343, 0.11813, -0.04547, -0.060272, -0.228658, 0.006851, 0.190229, 0.174483, 0.144256, 0.083717, 0.286654, 0.181938, 0.227765, 0.198379], [0.128306, -0.464556, 0.075481, -0.114653, 0.051368, -0.121279, 0.143164, 0.123471, -0.149528, 0.054869, -0.166504, 0.260766, 0.122181, -0.105665, -0.250027, -0.208667, 0.045583, -0.02813, -0.209909, -0.423304, 0.037328, -0.240143, 0.035583, -0.031221, 0.110374, 0.234843, 0.021132, -0.186293, -0.139088, -0.160335, -0.252088, -0.004662, 0.025488, 0.235448, -0.090677, -0.058223, 0.138298, -0.091143, -0.283294, 0.077383, -0.026924, -0.119712, -0.057292, 0.011131, -0.070605, -1.026832, -0.191802, 0.123704, -0.015549, 0.082235, 0.169693, -0.019509, 0.041594, 0.269636, -0.272836, -0.234782, -0.103002, -0.236172, 0.30276, 0.033042, 0.20249, -0.297234, -0.170754, -0.432373], [0.336627, 0.042428, 0.096373, 0.186076, -0.07795, -0.059795, 0.046264, -0.215858, -0.154517, 0.00316, -0.047638, -0.144544, 0.080029, -0.086889, 0.18659, 0.073232, 0.191232, -0.278979, 0.489536, 0.300719, 0.043976, 0.167875, -0.028546, 0.157137, 0.010072, 0.12163, 0.356082, 0.035485, -0.270708, -0.28693, 0.159269, -0.17002, -0.014963, 0.06829, -0.263538, 0.480808, -0.141429, 0.014847, 0.175966, -0.067338, 0.146935, -0.083933, 0.225639, -0.163022, 0.092808, 0.259219, -0.261541, 0.195327, 0.041814, -0.044144, 0.057972, 0.06676, -0.087286, -0.05714, 0.212319, 0.261303, 0.239785, 0.202009, 0.139824, -0.424649, -0.086037, 0.086875, 0.198655, 0.087154], [0.017935, 0.03361, -0.372447, 0.01747, 0.217472, -0.099899, -0.330102, 0.294076, -0.192678, -0.14636, -0.145521, -0.054243, 0.10909, -0.436694, 0.297411, -0.292495, 0.133207, -0.126885, 0.019444, 0.106939, -0.022449, -0.407566, 0.130095, -0.13054, -0.183519, -0.56407, -0.194318, -0.263438, 0.163374, -0.353677, -0.075473, 0.191977, -0.075576, -0.341143, -0.012832, 0.087982, -0.000623, -0.01845, -0.158681, -0.346015, -0.141436, 0.005118, -0.166709, 0.202858, 0.158461, -0.222898, 0.110567, -0.436344, 0.102392, 0.057339, -0.279333, -0.211032, -0.051632, -0.020636, -0.077875, 0.046988, 0.036043, -0.053569, -0.182331, 0.17515, -0.032073, 0.076215, -0.025034, 0.269921], [0.126241, 0.043264, 0.033382, -0.557894, -0.321863, -0.002295, -0.237608, -0.051047, -0.04392, -0.02406, -0.144719, 0.203053, -0.078449, -0.386752, -0.018791, -0.056283, 0.037086, -0.293838, -0.363838, -0.006958, -0.116212, -0.23668, -0.079676, 0.091698, -0.01213, 0.056881, 0.011994, -0.26762, 0.007819, -0.452027, -0.121635, 0.014497, 0.082067, -0.256561, -0.011031, 0.231722, 0.219468, 0.033664, 0.146258, -0.210657, 0.247202, 0.114545, -0.099619, 0.209987, -0.028441, 0.00533, -0.201471, -0.107928, -0.036015, 0.178883, -0.24054, 0.243437, 0.21675, 0.117814, -0.221516, 0.111429, -0.052278, 0.05735, -0.180271, -0.102448, -0.280909, 0.187166, 0.210337, -0.263568], [0.014496, -0.129505, -0.10341, 0.154239, 0.000364, -0.029064, -0.189938, -0.476838, -0.027657, -0.20316, -0.047666, -0.15271, -0.104318, -0.355331, 0.417607, -0.355595, -0.467391, -0.024164, 0.340756, 0.019958, -0.030892, 0.28359, 0.060302, -0.147875, -0.031367, 0.089835, 0.1923, 0.482576, -0.036908, 0.022039, 0.028102, -0.001529, -0.148144, -0.286838, 0.33782, 0.106504, -0.11047, 0.01679, -0.135836, -0.03812, 0.012501, 0.020803, 0.040579, -0.089521, -0.197768, -0.446449, -0.105763, 0.139295, -0.153146, 0.091859, 0.231135, 0.110054, 0.013413, -0.355093, -0.084695, 0.098757, -0.329465, 0.162148, -0.127899, 0.108146, -0.417289, -0.081004, -0.343388, -0.157606], [0.025838, 0.002525, -0.020095, -0.322367, -0.323141, 0.236785, 0.236077, 0.068322, -0.044686, -0.01937, 0.197798, 0.259742, 0.146404, 0.17028, 0.073625, -0.09848, -0.188106, -0.167897, -0.303623, 0.02447, 0.106401, -0.030867, 0.005871, 0.133344, 0.228823, 0.353959, 0.25257, 0.050904, -0.207702, -0.048148, -0.133047, 0.244547, 0.145195, 0.339375, -0.048605, 0.21309, -0.333387, 0.072629, -0.540842, 0.273615, -0.046411, 0.214937, -0.085017, 0.396114, 0.145436, 0.146047, 0.040901, -0.278441, 0.199244, 0.159307, -0.044977, -0.074284, 0.254342, -0.11099, 0.219341, -0.346255, -0.033249, 0.220329, -0.20514, 0.141, 0.11339, -0.027817, 0.173551, -0.143088], [0.088885, -0.497186, 0.267945, 0.326275, 0.344596, 0.236016, -0.49306, 0.398487, 0.102145, 0.055104, -0.230763, 0.238687, -0.031697, -0.150469, -0.016538, 0.224029, 0.144273, 0.208529, 0.152597, 0.162864, -0.205668, -0.337812, -0.084771, -0.162751, 0.028044, 0.640925, 0.141251, -0.889896, 0.193689, -0.117142, 0.113759, -0.034072, 0.366982, 0.389358, 0.143745, -0.510569, -0.320194, -0.332534, -0.059623, 0.404076, 0.002943, -0.219194, -0.055218, 0.637556, 0.113691, -0.021973, -0.073372, 0.042401, 0.364835, 0.233152, -0.242085, -0.303304, 0.09473, 0.027412, -0.225224, -0.092584, 0.020075, 0.052583, -0.05307, -0.260302, 0.044674, 0.129459, 0.118804, -0.105084], [0.250339, 0.253288, 0.050537, -0.019787, 0.139649, -0.300958, 0.074127, 0.093665, 0.139259, -0.108668, -0.009018, -0.112225, -0.063449, -0.34919, 0.053294, 0.160226, -0.150352, -0.202394, 0.249649, -0.019262, 0.065979, -0.157467, 0.258261, 0.307047, 0.077662, -0.087915, 0.172466, 0.024486, 0.067664, 0.083389, -0.033204, 0.037747, 0.165775, -0.321592, 0.043824, -0.195339, 0.158994, 0.152823, -0.197742, -0.027911, 0.139989, -0.264962, -0.133854, -0.293805, 0.052677, -0.290054, 0.004747, -0.420783, 0.164362, 0.25953, -0.105197, -0.236452, -0.144029, -0.006129, 0.200749, -0.132942, -0.238585, -0.311345, -0.008593, -0.18826, 0.075154, 0.033973, 0.155549, 0.089607], [0.172057, 0.073711, 0.11891, -0.112042, -0.191333, -0.461807, 0.069038, 0.254162, 0.164195, 0.038841, -0.134186, -0.317062, 0.133407, -0.041317, -0.284932, 0.332113, -0.065728, -0.065767, -0.091117, -0.059797, -0.24706, 0.049295, 0.150124, -0.264472, 0.157366, -0.0599, -0.074439, -0.300707, -0.386331, 0.179672, -0.22332, 0.235872, -0.038865, -0.358156, 0.01345, -0.04305, 0.072984, 0.007331, 0.138291, 0.212084, 0.150914, -0.0237, 0.048748, -0.507083, 0.132323, 0.049899, -0.224653, 0.03076, -0.066364, -0.064947, 0.046424, -0.154458, -0.229915, -0.363392, -0.029122, 0.062977, 0.127114, 0.008679, -0.106796, -0.119657, 0.090543, -0.238013, -0.173116, 0.129462], [-0.257759, 0.057795, 0.122266, 0.147974, 0.287184, 0.010352, 0.233151, -0.333157, 0.383356, -0.024527, 0.030888, -0.473803, -0.152606, -0.119261, 0.34602, -0.07643, 0.146287, 0.288716, 0.12837, -0.124529, -0.457333, 0.453552, 0.218814, 0.250785, -0.196641, 0.183615, -0.073076, -0.314982, 0.183314, 0.032517, 0.317187, -0.092521, 0.487372, -0.501982, 0.003432, -0.052686, -0.121245, 0.018488, -0.055778, -0.197915, -0.108891, -0.128579, 0.323274, 0.023329, 0.002918, 0.26358, 0.28557, 0.408396, -0.138044, 0.015098, 0.055503, 0.318836, -0.01567, 0.471827, 0.100638, 0.086612, -0.477096, -0.47422, -0.310264, -0.023225, 0.973976, 0.097458, 0.041294, -0.220855]], [0.202636, -0.234612, 0.032284, -0.155239, 0.109213, -0.124146, 0.05238, -0.003188, -0.167228, 0.2714, 0.249855, -0.146687, 0.124119, 0.072869, -0.115406, -0.006054, -0.209929, -0.199762, -0.231904, 0.018565, 0.010671, -0.130669, 0.141357, -0.101494, -0.050345, -0.266612, -0.001067, 0.015935, 0.006099, -0.25276, -0.105881, 0.286078, -0.216712, 0.088169, 0.203321, -0.143626, -0.039749, -0.119561, 0.209337, 0.011921, 0.16257, -0.021249, 0.055418, -0.169305, -0.167891, -0.130032, 0.069862, -0.027198, 0.097648, 0.132957, 0.269915, -0.000249, -0.027775, -0.164765, 0.134299, -0.088993, -0.045623, -0.199469, -0.053836, -0.105123, -0.315368, 0.080361, -0.030097, -0.191633]),
    ([[-0.989666], [-0.621319], [0.693247], [0.58192], [-0.342145], [0.756535], [-0.497034], [1.02273], [0.40291], [-0.350178], [-0.338961], [0.330454], [0.274592], [0.424333], [0.851683], [0.718717], [0.528239], [-0.515074], [2.091876], [-0.48779], [-0.414597], [-0.315308], [0.251871], [0.294201], [0.741543], [-0.412863], [-0.470097], [1.719924], [0.360127], [-0.587228], [0.418757], [0.279071], [0.810229], [0.933466], [0.342024], [0.582669], [-0.22272], [0.378474], [-0.374916], [-0.286573], [-0.261638], [0.391328], [0.779781], [-2.281709], [-0.324444], [0.496547], [-0.673068], [-0.328234], [-0.703555], [-0.501387], [-0.15451], [-0.457536], [-0.255322], [-0.655013], [-0.165563], [0.268334], [0.536975], [-0.689327], [-0.345109], [-0.251401], [1.18438], [0.32063], [0.482454], [0.794516]], [0.327529]),
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
