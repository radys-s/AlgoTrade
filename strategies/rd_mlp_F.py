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
MLP_MEAN = [0.000149, 0.000208, 0.000232, 0.000293, 0.000322, 0.000862, -0.001537, -0.000924, 5.2e-05, 1.25645, -0.00136, -0.002965, 6.038898, 6.044026, -9.8e-05, -0.001155, 0.00191, -0.000678, -0.097957, -0.597011, -5.096642, 4.069438]
MLP_SCALE = [0.864121, 0.734825, 0.692527, 0.604736, 0.524, 1.91431, 3.531115, 0.568915, 7.646924, 0.75669, 0.638444, 0.531728, 1.610903, 1.612934, 1.0, 4.384642, 0.998702, 0.385728, 129.756088, 315.187134, 912.231628, 1.343312]
MLP_LAYERS = [
    ([[-0.004797, 0.058145, -0.122376, 0.131678, 0.396478, 0.458473, -0.230543, 0.26552, 0.386488, -0.375017, 0.268832, -0.109237, 0.082621, 0.634669, -0.307875, -0.444153, -0.017049, 0.341576, 0.217314, 0.350007, 0.002262, 0.070508, 0.149465, -0.02314, -0.147587, -0.189452, 0.064278, 0.187121, 0.000528, 0.064687, 0.01104, 0.403415, -0.10073, -0.013186, 0.04962, 0.048876, -0.08702, 0.105846, 0.148768, -0.088155, 0.036494, -0.159386, 0.06459, 0.048549, 0.139407, 0.086436, -0.009723, -0.315582, -0.219185, 0.056498, -0.313206, -0.019709, 0.137282, 0.005291, -0.070581, -0.166328, -0.105907, -0.199134, -0.192297, -0.072918, -0.2097, -0.088793, 0.029201, 0.228194], [-0.216096, 0.014285, 0.223515, -0.311131, 0.242008, -0.255614, 0.534656, 0.196894, 0.402911, -0.259149, 0.151298, -0.228772, -0.087952, -0.151463, 0.126472, -0.368859, -0.065505, -0.105842, 0.004285, 0.07454, -0.005198, -0.067865, 0.224989, -0.278886, -0.029517, 0.354095, -0.042075, 0.063395, -0.209723, -0.21844, -0.146637, -0.134544, 0.253173, -0.309135, -0.002536, -0.363353, -0.100348, -0.173114, 0.179603, 0.263148, -0.11035, 0.028739, 0.144967, -0.169118, -0.236973, 0.137438, -0.019369, 0.050526, 0.182962, -0.301643, 0.286464, 0.068265, -0.054227, -0.003563, 0.141955, 0.213033, 0.17452, -0.106933, 0.419461, 0.075502, 0.041041, -0.01478, -0.252732, -0.291171], [0.029182, 0.05253, 0.128712, -0.200816, -0.381541, -0.182975, 0.215145, 0.197991, 0.130207, -0.041487, 0.035623, -0.171102, 0.365359, -0.219163, 0.031608, 0.210594, 0.031813, -0.113601, -0.30567, 0.396908, 0.122752, 0.413244, -0.126419, 0.303057, -0.228585, 0.226971, -0.132427, 0.186676, 0.093753, 0.113019, -0.236897, -0.19478, 0.153919, 0.14259, 0.066243, 0.00769, 0.142037, 0.325545, -0.384465, -0.001838, 0.088688, -0.038753, -0.070679, -0.363317, -0.008469, -0.217882, 0.191767, -0.283678, -0.005078, 0.265091, 0.193859, -0.270758, -0.244215, -0.084837, -0.020297, 0.00024, 0.515464, 0.165255, 0.099302, 0.061092, 0.404873, -0.071103, -0.1619, -0.522502], [-0.005979, 0.196685, -0.089679, 0.166031, -0.435527, 0.312754, -0.157161, -0.180965, -0.177972, 0.295325, 0.039151, -0.357149, -0.32798, -0.292479, -0.133664, 0.21364, 0.050201, 0.065767, 0.355535, -0.13992, -0.156879, -0.258894, -0.460742, 0.232613, -0.274276, -0.217909, 0.06909, 0.08315, 0.269712, 0.225621, -0.098205, -0.012093, -0.208938, 0.414288, -0.085481, -0.130057, 0.446827, 0.123628, -0.319988, 0.025944, -0.078169, -0.058674, -0.002744, -0.140109, -0.12326, -0.149655, -0.269512, 0.115116, 0.149862, -0.11992, -0.064297, -0.039914, 0.22338, 0.044001, 0.100341, -0.002745, -0.368271, 0.170261, -0.038235, 0.006256, 0.264904, 0.273172, 0.226175, -0.27911], [0.076479, 0.045393, 0.606277, -0.008925, 0.07659, -0.284224, -0.449413, 0.160168, 0.127966, 0.319908, 0.034789, 0.012116, -0.273611, 0.136721, 0.426258, 0.046888, 0.059409, 0.425109, -0.157385, 0.102554, 0.195892, 0.132951, -0.062639, 0.68789, 0.001564, 0.271539, -0.116458, -0.083032, -0.216725, -0.079892, 0.164124, 0.27441, 0.119085, -0.146101, -0.249165, -0.137796, 0.591619, -0.29441, 0.396946, -0.29225, -0.320877, 0.128103, 0.224845, 0.502733, 0.112575, 0.260264, -0.207501, -0.104977, 0.027663, -0.211392, -0.380807, -0.249729, 0.186354, -0.066808, 0.068798, -0.287173, -0.396456, -0.16218, 0.077585, -0.298092, -0.403673, -0.149788, 0.212522, 0.349229], [-0.362939, -0.30054, -0.387579, -0.046125, -0.305918, -0.545855, 0.124392, 0.121368, -0.557575, 0.065394, -0.270503, 0.453405, -0.270898, 0.013068, 0.068405, 0.406229, -0.342099, -0.321216, 0.035684, -0.057015, 0.343202, -0.27914, -0.089734, -0.41337, 0.112972, -0.305928, 0.024696, -0.077093, 0.043889, 0.170588, 0.194592, -0.424041, -0.227392, 0.188633, 0.073213, 0.199539, -0.087283, -0.153098, 0.150482, 0.265356, -0.215302, 0.167808, -0.085184, 0.333074, -0.0508, 0.015595, 0.04671, -0.055848, -0.06965, -0.028851, 0.179502, -0.108347, -0.037596, 0.149707, -0.170114, -0.05315, 0.284186, -0.067448, -0.233475, 0.654118, 0.088789, -0.137549, 0.380285, 0.135075], [-0.150238, -0.084856, 0.571215, -0.170883, -0.216694, -0.221096, -0.62327, -0.146236, -0.498319, 0.182287, -0.266253, 0.439797, -0.319388, -0.073473, 0.225055, 0.061372, -0.055123, 0.01685, -0.513441, -0.246589, 0.156749, -0.045284, -0.621005, -0.148655, 0.019293, -0.39265, -0.197589, 0.00834, -0.221927, -0.137398, -0.063425, -0.128365, -0.022889, -0.1021, -0.228678, 0.111424, -0.302923, 0.221709, -0.167426, -0.739279, 0.349663, 0.855364, 0.225411, -0.119699, 0.922443, -0.172632, 0.314687, 0.215862, 0.079506, -0.183992, 0.213214, -0.110873, 0.231511, -0.336077, 0.66904, 0.024334, -0.217022, 0.030666, 0.634721, -0.867592, -0.121951, -0.518507, -0.004292, 0.079999], [0.082572, 0.199707, 0.042516, 0.1582, -0.203902, -0.003188, 0.215092, -0.164459, 0.080297, -0.292503, 0.42257, 0.048646, -0.029684, 0.050397, -0.082306, -0.01777, -0.160132, -0.082128, 0.062303, 0.145905, 0.312627, -0.067914, 0.070573, 0.014326, 0.073411, 0.109995, 0.119102, -0.01021, -0.144164, 0.21792, 0.32008, 0.389241, -0.257625, 0.014273, -0.127025, -0.113707, -0.226237, -0.034042, -0.173832, 0.185374, 0.011507, -0.161342, 0.041009, -0.03246, 0.101874, 0.00974, 0.017992, -0.0496, 0.112313, 0.010699, -0.266542, -0.095433, -0.126361, 0.001245, 0.23039, 0.039154, 0.033615, 0.257594, 0.565427, 0.030497, -0.052511, -0.313864, -0.100167, -0.068146], [0.357322, 0.352319, -0.153567, -0.147513, -0.075241, -0.214155, 0.071619, 0.447659, -0.138724, 0.134714, 0.002279, 0.485774, -0.061771, 0.34484, 0.358339, 0.503685, -0.227366, -0.589632, -0.721206, -0.286497, -0.192694, 0.395119, -0.379484, -0.226361, 0.368111, 0.191456, 0.710451, -0.61551, -0.316199, 0.145919, -0.452761, 0.201784, 0.452204, -1.088288, 0.045233, -0.098645, 0.054255, 0.369291, 0.602263, 0.405635, -0.303918, -0.383947, -0.273732, 0.257519, -0.478447, 0.201665, -0.20938, 0.039651, 0.383143, -0.415017, 0.098373, 0.238907, -0.599276, 0.240742, 0.027685, 0.130035, 0.568872, 0.110639, 0.786024, -0.312376, -0.160402, -0.109912, -0.143864, -0.076329], [-0.278973, 0.165511, -0.301445, 0.479794, 0.055834, 0.165694, -0.11537, -0.203259, -0.076203, -0.378607, -0.520898, -0.00599, -0.23971, 0.229559, -0.125956, 0.016082, 0.217238, 0.167882, 0.053301, 0.169661, -0.21321, 0.17248, 0.139943, 0.248559, -0.166192, -0.134975, 0.174662, -0.136546, -0.173825, 0.078891, 0.07372, -0.275581, 0.091425, 0.194242, -0.223885, 0.085318, 0.160173, -0.239664, -0.409945, 0.02766, 0.369014, 0.369338, 0.111357, 0.639862, 0.069833, 0.106423, 0.211816, 0.04163, 0.562981, 0.246713, 0.271841, -0.291512, 0.173932, -0.258077, -0.206736, -0.078076, -0.039186, -0.130201, 0.262801, 0.269765, 0.104152, 0.243416, -0.089604, 0.010517], [-0.092165, 0.070931, 0.297986, 0.079575, 0.07189, -0.134142, 0.168358, -0.136798, 0.278795, 0.189037, -0.370796, -0.054245, 0.01359, 0.035936, 0.368428, -0.27987, 0.231672, -0.215513, 0.026637, -0.166137, -0.25414, 0.082295, -0.047984, -0.096142, 0.143094, 0.106486, -0.004737, -0.217078, 0.19283, 0.111226, -0.196369, -0.049312, 0.281178, -0.304475, 0.337027, 0.164473, 0.168598, 0.040425, 0.217822, 0.062244, -0.06327, -0.101649, -0.119276, 0.243064, -0.241792, -0.116703, -0.055531, -0.046384, -0.029665, 0.063133, 0.164, 0.047851, -0.131597, -0.09417, -0.04522, 0.148604, -0.091831, -0.307304, 0.070908, 0.013605, 0.008144, -0.168236, 0.164353, -0.099891], [0.070068, -0.427297, -0.213849, -0.433953, -0.024556, 0.002497, -0.015879, 0.205664, 0.116841, -0.188813, 0.0605, -0.230923, -0.359927, -0.119376, -0.107716, 0.163107, -0.012848, 0.269762, -0.151859, 0.130451, -0.04792, 0.245609, 0.303478, 0.327103, 0.079146, 0.068349, 0.216651, 0.105989, -0.168579, -0.268452, -0.194086, 0.103429, -0.035867, 0.150666, -0.146578, -0.363982, -0.139815, 0.552177, -0.207139, -0.25154, -0.302014, -0.149143, 0.14567, 0.091659, 0.397354, 0.219956, -0.101488, -0.193447, 0.368424, 0.073293, -0.00148, 0.068508, 0.155558, 0.024818, -0.487319, -0.381165, 0.015506, -0.014879, 0.098647, 0.176247, 0.271767, 0.265404, -0.155751, -0.174223], [0.473941, 0.393357, 0.132536, -0.085197, -0.115575, -0.061854, 0.146324, -0.0574, -0.272101, -0.013836, -0.079553, 0.330103, -0.103898, 0.126656, -0.058023, 0.148197, -0.131392, -0.143984, 0.097403, -0.374738, -0.047615, -0.259095, 0.444599, 0.024616, 0.044179, -0.234759, 0.453678, 0.475328, -0.017919, -0.111233, -0.094573, -0.023061, -0.240089, 0.041993, -0.025056, 0.022945, 0.010047, 0.03857, 0.078557, -0.400545, 0.392185, 0.111999, -0.282997, -0.060139, -0.036462, 0.228433, -0.0831, 0.070758, -0.241389, -0.079086, 0.002803, -0.117605, -0.038901, -0.06883, 0.052537, -0.123924, -0.278583, 0.413772, -0.173511, -0.22049, 0.047038, -0.104753, -0.024021, -0.06804], [0.162248, -0.284889, -0.051592, -0.121726, 0.138764, -0.177766, 0.178793, -0.043478, -0.008347, -0.161701, -0.129392, 0.024308, -0.408715, 0.176515, 0.366563, -0.109607, -0.251565, -0.042613, -0.065616, -0.073407, -0.022621, -0.077099, 0.389003, 0.069814, -0.157302, 0.145206, -0.248536, 0.008534, 0.55881, 0.347959, -0.405418, 0.006528, 0.042478, 0.294892, 0.201855, -0.1882, 0.245107, 0.093599, 0.648914, -0.049619, -0.235866, -0.273222, 0.182528, 0.263181, -0.054517, 0.061508, -0.034258, 0.149503, 0.049215, -0.109312, -0.109768, -0.002344, -0.129501, -0.032635, 0.269852, -0.033503, -0.217737, 0.125203, -0.020717, 0.016434, 0.018127, -0.083938, 0.103235, -0.184301], [-0.092858, -0.054284, 0.067679, 0.145217, -0.045026, -0.313355, -0.035723, -0.177179, 0.339796, 0.060189, 0.042742, -0.028695, -0.357778, -0.579839, 0.198965, 1.1e-05, 0.607732, -0.514157, 0.119611, -0.088811, 0.294055, -0.038463, 0.512714, -0.294355, 0.580037, 0.142822, 0.23263, 0.009925, 0.006863, 0.150433, -0.305579, -0.055509, 0.755869, -0.36243, 0.056993, -0.320498, -0.412192, -0.077491, -0.129997, 0.283383, 0.277167, -0.442889, 0.007123, 0.223085, -0.751101, -0.03574, -0.023193, -0.078175, 0.506951, -0.037457, 0.228758, 0.033616, 0.124516, -0.120624, -0.567112, 0.286715, -0.015407, -0.321863, 0.231977, -0.594292, 0.488165, 0.124991, 0.429751, -0.220656], [-0.314974, -0.16725, 0.001031, 0.034876, -0.101501, -0.189148, 0.097454, 0.172921, -0.112319, 0.267697, -0.011771, -0.293883, 0.484963, 0.273575, 0.7633, 0.04076, -0.14985, 0.174995, 0.022219, -0.041668, 0.092938, -0.179362, 0.040365, 0.310296, -0.131076, 0.130016, -0.208427, 0.461088, -0.322375, 0.139338, -0.374231, -0.420705, -0.367762, 0.268541, 0.569952, -0.163864, 0.199927, 0.201189, -0.114447, 0.042377, -0.460597, -0.10647, 0.065307, -0.026409, 0.006752, 0.092276, -0.228798, -0.249676, -0.074366, 0.190834, 0.144074, -0.216183, 0.123322, 0.20308, 0.080486, 0.0655, -0.099251, -0.216343, -0.316497, 0.434015, -0.397699, 0.00959, 0.197174, 0.41286], [0.628451, 0.456218, 0.171819, -0.381221, 0.414547, 0.14998, -0.085564, -0.411676, -0.185428, 0.381369, 0.443569, -0.189443, 0.222879, -0.022398, -0.586162, -0.524356, -0.474066, 0.012879, -0.031977, 0.58864, -0.585949, 0.019749, 0.072208, 0.331803, 0.195606, 0.184715, -0.225777, -0.769753, 0.633418, -0.106277, 0.494761, -0.33151, -0.33592, 0.149202, 0.431574, 0.035265, 0.169909, -0.041444, -0.133873, -0.561421, -0.039023, -0.523588, -0.63678, 0.380676, 0.221979, -0.140964, -0.658736, 0.408405, -0.286613, -0.766935, -0.076606, -0.682839, -0.369424, 0.458068, 0.201409, -0.116976, -0.051363, 0.465786, -0.122107, 0.307859, -0.252158, 0.297443, -0.858005, 0.19899], [0.268157, 0.473483, -0.24597, -0.34047, 0.543338, 0.254694, 0.434897, 0.653223, -0.401885, 0.313953, -0.017205, 0.313599, -0.26227, 0.355963, 0.224768, 0.065767, 0.315518, 0.173592, 0.062735, -0.223158, -0.243043, 0.218364, -0.216499, -0.060426, -0.179671, -0.383452, 0.063862, -0.067908, -0.315843, -0.176544, 0.570988, 0.146971, 0.11971, 0.111015, 0.373308, -0.44222, 0.017955, 0.399919, 0.484076, -0.087333, -0.031472, 0.117739, -0.390218, -0.359933, 0.295552, -0.08558, 0.142274, -0.491746, 0.341038, 0.245632, -0.296294, 0.222815, -0.093805, 0.07293, -0.217041, 0.05794, -0.282825, 0.19551, -0.033454, -0.01875, 0.031199, 0.029393, -0.685379, 0.331398], [0.042177, -0.132197, 0.014481, -0.006212, 0.008914, 0.021088, 0.027239, 0.080348, 0.037614, -0.043481, 0.067539, -0.033286, -0.041916, -0.067331, 0.086311, -0.008116, 0.103353, 0.305856, -0.000606, 0.009752, 0.055789, -0.037375, 0.079008, 0.038722, -0.043033, -0.017566, 0.067446, -0.147621, 0.035397, 0.145305, 0.032368, -0.024608, -0.015148, -0.111625, 0.112653, 0.010968, 0.068565, -0.008719, 0.088379, 0.032355, -0.043295, 0.05164, 0.019017, 0.077624, 0.168745, -0.012389, -0.064189, 0.268034, -0.1639, 0.182139, 0.107511, 0.120864, 0.080179, 0.050479, 0.020347, -0.014645, -0.062024, -0.036513, -0.017124, 0.014058, 0.131322, -0.073821, 0.092911, -0.058295], [-0.015189, -0.278925, 0.400972, -0.047562, 0.223919, 0.123976, 0.030555, -0.027683, 0.075036, -0.102683, 0.110385, 0.218941, -0.159158, -0.254242, -0.252801, -0.273915, -0.344209, -0.528706, -0.814647, 0.028192, 0.460434, 0.074985, -0.194041, -0.35653, -0.18534, -0.141487, 0.595245, -0.377355, 0.355149, 0.098957, 0.092867, -0.00974, 0.214212, 0.103848, 0.408047, -0.134129, -0.465463, 0.561482, -0.026715, -0.512505, -0.014834, 0.369435, 0.10252, 0.211391, -0.024165, 0.459538, -0.399072, -0.234607, 0.391597, 0.111261, 0.620563, -0.055159, 0.987728, 0.150641, 0.025096, -0.532808, 0.023376, 0.193349, 0.137958, 0.016891, -0.050947, -0.271534, 0.134637, -0.213844], [0.01514, 0.278565, -0.153404, -0.197845, 0.057902, 0.253783, -0.391733, -0.172936, -0.188878, -0.013207, 0.020787, 0.121889, 0.387893, -0.016375, -0.073375, 0.097466, -0.100348, -0.316923, 0.137958, -0.000237, 0.322332, -0.125654, 0.018355, 0.059771, 0.278404, 0.350979, -0.027145, -0.094509, 0.119367, 0.084735, 0.177415, -0.288735, -0.146926, -0.417923, -0.180242, 0.171445, 0.055339, 0.000602, 0.038708, 0.332924, 0.499038, -0.344911, -0.36676, 0.082658, -0.070698, 0.217235, 0.554316, -0.395927, -0.228598, 0.298081, -0.158622, 0.084383, -0.050047, -0.114297, 0.050644, 0.115336, 0.025908, -0.051565, -0.011965, -0.151158, -0.152586, 0.119931, -0.060256, 0.289289], [-0.110404, 0.085532, -0.179452, -0.070913, -0.003241, 0.068546, -0.017271, -0.149561, 0.068464, 0.004749, 0.057675, -0.47726, -0.187308, -0.013323, 0.010806, -0.194156, -0.025241, 0.190384, -0.128539, 0.081422, -0.055912, 0.622211, 0.016612, 0.014913, 0.607907, -0.112595, 0.347625, 0.08448, 0.342725, -0.412802, 0.070119, 0.258062, -0.281342, -0.103587, -0.201692, -0.157949, 0.32632, -0.265711, -0.047415, 0.216525, 0.123176, 0.090222, -0.039876, -0.01811, -0.269639, -0.114803, -0.134943, 0.294409, -0.003707, 0.42187, 0.296101, -0.255973, -0.096061, 1.01463, 0.302123, 0.018331, 0.014919, -0.144828, 0.041768, 0.11564, 0.394667, 0.266879, -0.358016, -0.039129]], [0.525131, -0.035449, -0.784635, -0.377149, -0.128238, -0.674031, -0.245671, -0.590099, -0.180746, -0.482017, -0.443799, -0.012798, -0.594407, -0.318668, -0.080242, -0.329702, -0.537675, -0.367233, 0.247425, 0.063162, 0.151368, 0.155131, -0.181479, -0.666822, -0.131801, -0.079814, 0.157322, 0.374597, -0.188189, 0.196361, 0.054972, -0.078416, -0.32907, -0.099402, -0.242754, -0.051452, -0.688646, -0.247435, 0.500902, -0.260268, -0.164136, -0.057403, 0.096174, -0.080151, 0.282371, 0.385814, -0.225128, -0.728588, -0.298302, -0.194689, 0.115318, -0.143216, -0.042001, 0.013665, 0.152765, 0.243537, -0.02211, 0.426142, -0.379322, -0.73417, 0.020553, 0.089122, -0.178688, -0.024262]),
    ([[0.196677, -0.170786, -0.3901, -0.021448, 0.026623, -0.609702, -0.259184, 0.241598, 0.038949, -0.220149, -0.125947, 0.267121, -0.051716, 0.222198, -0.417122, -0.052432, 0.030944, -0.349618, -0.12573, 0.028628, -0.292861, 0.255677, 0.355783, -0.021723, -0.03116, -0.757974, -0.384797, -0.185189, -0.210941, 0.204988, -0.397188, 0.007984], [0.217519, 0.163554, 0.151148, 0.405655, 0.415676, 0.447574, -0.216082, -0.34872, -0.177065, -0.168707, 0.005055, -0.089506, -0.039622, 0.157937, 0.067222, 0.069237, 0.282926, 0.179299, 0.075577, 0.085095, -0.093452, 0.453232, -0.189414, -0.065079, -0.044444, -0.124551, -0.349827, 0.380034, 0.449985, -0.20026, 0.069331, -0.116893], [-0.27965, 0.153802, -0.138298, 0.163089, -0.16204, -0.016001, -0.96448, -0.227129, 0.062558, -0.15233, 0.165276, -0.326462, 0.126844, 0.408881, -1.11385, 0.438934, -0.021434, 0.009579, -0.5206, 0.103661, -0.466367, -0.204646, 0.236958, 0.384658, 0.147149, -0.003304, 0.393277, 0.238414, -0.339567, 0.333468, 0.346676, 1.13538], [-0.197553, 0.287987, 0.072669, -0.270737, -0.086507, -0.606336, -0.45287, -0.162248, -0.201322, 0.154326, 0.147315, 0.158801, -0.039851, 0.142536, 0.149088, 0.329172, 0.049831, 0.022963, -0.310804, 0.160217, 0.176178, 0.139984, 0.036815, -0.0205, 0.090419, -0.081172, -0.128572, 0.008498, 0.125197, 0.138384, 0.285166, -0.436847], [-0.14142, 0.195318, -0.042575, 0.174949, 0.362954, -0.405848, 0.311958, -0.350754, -0.122651, 0.317185, 0.144416, -0.127809, 0.20727, 0.021661, 0.299448, 0.2838, 0.352953, 0.094424, -0.01537, -0.392393, 0.277291, -0.172122, 0.031891, 0.050766, 0.291861, -0.172841, 0.219529, 0.310766, 0.22644, 0.240662, 0.131802, 0.188739], [-0.749503, -0.0881, 0.022252, 0.006175, 0.159088, -0.552693, 0.047338, 0.337492, -0.910724, -0.03052, -0.516879, 0.186982, 0.318498, 0.202865, 0.125676, -0.240816, 0.082297, -0.440248, -0.526995, -0.402483, -0.212716, -0.596996, 0.107459, -0.038843, 0.517057, -0.125026, 0.243681, 0.282949, 0.263019, 0.117758, -0.093281, 0.040199], [-0.088091, 0.061048, 0.020632, -0.047111, -0.800824, -0.179445, 0.196994, -0.320735, -0.055986, 0.203065, 0.072467, 0.001121, -0.060818, -0.466956, -0.156354, 0.14903, -0.039187, 0.025471, -0.396859, -0.06814, 0.004311, -0.256863, -0.242689, 0.268999, -0.170836, 0.03567, 0.64938, -0.152238, 0.377501, 0.080162, 0.162587, 0.217363], [-0.01498, 0.170115, 0.077697, 0.052379, 0.016291, -0.069112, -0.005354, -0.221753, 0.047999, 0.018872, -0.233305, 0.257529, -0.16041, 0.161526, -0.507246, 0.075027, -0.712489, 0.328941, -0.248522, -0.071732, -0.079096, 0.067814, 0.056219, 0.09134, 0.273179, 0.398818, -0.115187, -0.034549, -0.441714, -0.028462, -1.22577, 0.396016], [0.017673, -0.18921, -0.248562, -0.006418, -0.101937, 0.148367, 0.281603, -0.681011, 0.020434, 0.243626, 0.096228, 0.005948, 0.154265, 0.566977, 0.437487, 0.224968, 0.173573, 0.622757, -0.542391, -0.399657, -0.36065, -0.414878, -0.008593, 0.24611, -0.026263, -0.054939, 0.184347, -0.331761, 0.092826, -0.034555, 0.125551, 0.092236], [-0.268341, 0.3754, -0.001683, 0.205117, 0.03348, 0.296958, 0.174048, 0.148658, -0.142211, 0.172752, 0.129992, 0.307693, 0.425994, 0.193325, -0.128893, -0.253013, -0.065921, -0.087446, -0.357641, -0.119896, 0.375374, 0.018401, -0.03652, -0.425328, 0.486368, 0.204923, -0.009609, 0.001693, 0.280906, -0.650708, 0.110309, 0.326681], [-0.142083, 0.214678, -0.178667, 0.066777, 0.003462, -0.015834, 0.203719, -0.431713, 0.10539, -0.648133, 0.221953, -0.124351, 0.032006, -0.066093, 0.479051, -0.316304, 0.113365, 0.320638, -0.376244, 0.173292, 0.015224, -0.024953, 0.231145, 0.188876, -0.001695, 0.016342, 0.141551, 0.178552, 0.235243, -0.090429, -0.094401, 0.45148], [0.30756, -0.180179, 0.003851, -0.075833, -0.053309, -0.077165, -0.112599, 0.12888, -0.046655, 0.461152, -0.01662, -0.034035, -0.059158, -0.073511, -0.059857, -0.417133, 0.382377, 0.496159, -0.358362, -0.11046, 0.220964, 0.106089, 0.337072, 0.224453, 0.012227, -0.39261, -0.143519, -0.177639, -0.11812, -0.300848, 0.279825, -0.112297], [-0.059168, -0.150675, 0.521652, -0.251082, 0.01974, -1.476813, -0.13226, -0.241561, -0.437197, 0.064477, -0.009929, 0.165607, 0.216278, 0.172912, -0.140469, 0.335868, -0.353916, -0.165489, -0.404487, -0.889343, -0.283305, -0.255986, 0.136127, -0.277006, 0.128288, -0.320397, -0.098857, -0.236111, 0.044357, -0.29921, -0.190601, 0.17807], [-0.258847, 0.071057, 0.10281, 0.24714, 0.082659, 0.024954, 0.132317, 0.249072, 0.093843, 0.093172, -0.505968, 0.475693, 0.252714, 0.146391, 0.098522, -0.236951, -0.060547, 0.369626, 0.370182, 0.263123, 0.012787, 0.265249, 0.375367, -0.084382, -0.322319, -0.858558, 0.019615, 0.277301, 0.024214, 0.289875, -0.381131, 0.213724], [-0.203444, -0.066241, 0.039775, -0.234238, 0.315255, -0.036757, -0.247399, 0.130555, 0.324866, -0.114961, -0.272636, -0.483669, -0.070117, 0.287981, -0.190441, 0.135802, -0.204073, 0.108094, 0.106699, -0.203284, -0.583968, 0.008671, -0.11817, 0.081906, -0.103613, -0.204451, -2.052559, 0.123732, -0.417104, -0.13021, 0.047031, -0.159719], [-0.044666, 0.359415, 0.286464, 0.030415, 0.226819, 0.274128, 0.226173, -0.264883, 0.002702, 0.228885, -0.171996, 0.170169, 0.228973, 0.201724, 0.297783, -0.023876, 0.014137, 0.253305, 0.169846, -0.046202, 0.320516, -0.303741, -0.091967, 0.145688, 0.316033, 0.329305, 0.331782, -0.386826, -0.245554, 0.380949, 0.250987, 0.25219], [0.119222, 0.166697, -0.108445, 0.042277, -0.237709, -0.179714, 0.139928, -0.16174, -0.177478, 0.135334, -0.805716, 0.265221, 0.12424, -0.061744, -0.048685, -0.028979, 0.155538, -0.252792, -0.071414, -0.597762, 0.199159, -0.005797, 0.03518, -0.308206, -0.105767, -1.254435, -0.149565, 0.017782, 0.092492, -0.721205, -0.135122, -0.080436], [0.26297, -0.249752, 0.113937, -0.023777, -0.166043, -0.010236, -0.382304, 0.071471, -0.124548, -0.33951, -0.075601, -0.65604, 0.427199, 0.254478, 0.121344, 0.435075, -0.129992, -0.389258, -0.255636, 0.170949, 0.255864, 0.352861, 0.046955, 0.232256, 0.025888, 0.061419, 0.003267, -0.173805, -0.20196, 0.21659, 0.200959, 0.18232], [0.24155, -0.020416, 0.121834, -0.648046, -0.027501, -0.685864, 0.109227, -0.015898, -0.02839, 0.092437, -0.145641, -0.491815, -0.473836, -0.001186, -0.138557, 0.185635, 0.402645, -0.368344, 0.166224, -0.462449, 0.145137, -0.128037, 0.109053, -0.000264, -0.207071, -0.732947, 0.003235, -0.434518, -0.002667, -0.060635, -0.002622, 0.216355], [0.09196, -0.071738, 0.081318, 0.180256, 0.154707, 0.096395, 0.311765, 0.044419, 0.172935, -0.138973, 0.035396, -0.003756, 0.008941, 0.333713, 0.36319, -0.063127, -0.049905, -0.212787, 0.255488, 0.079372, -0.01881, -0.066713, 0.162735, -0.145508, 0.241495, 0.329128, -0.054182, 0.2154, -0.006771, 0.036567, -0.515657, -0.455434], [-0.091538, -0.336249, 0.213971, -0.092345, 0.092955, 0.095964, -0.406965, 0.063104, 0.159992, -0.069222, -0.221727, -0.186999, 0.292732, -0.040199, -0.30367, 0.396595, 0.11587, -0.021815, -0.261732, 0.156541, 0.268453, 0.195562, -0.325558, -0.185288, 0.086801, -0.086866, -0.266031, -0.120826, 0.29863, 0.035275, -0.348293, -0.192286], [0.342904, -0.334952, 0.038809, -0.043367, 0.16365, 0.221387, 0.028874, 0.161662, 0.050326, -0.107179, -0.121866, -0.196933, -0.159215, -0.074522, -0.112492, -0.204115, -0.233326, 0.105396, -0.168987, 0.296131, 0.135521, 0.399479, -0.194616, -0.061397, 0.216072, 0.306185, -0.179141, -0.119064, -0.245998, 0.246234, -0.508583, -0.17057], [0.346316, -0.037365, 0.595283, -0.145613, 0.230768, -2.111612, 0.237909, 0.043294, 0.452647, -0.141688, -0.389284, -0.154939, 0.045931, 0.177726, 0.190589, 0.05587, 0.181231, -0.11236, 0.095595, 0.142809, -0.02582, 0.113185, -0.484487, -0.491535, 0.038093, 0.171592, 0.032736, -0.10008, 0.111997, -0.304196, 0.036143, -0.200171], [0.141258, 0.145085, -0.114103, -0.102992, 0.362826, -0.126198, 0.677647, -0.76608, -0.07364, -0.033958, 0.083741, 0.2181, 0.004634, -0.232479, -0.244276, 0.337649, 0.478405, -0.316637, 0.025242, -0.426867, 0.472653, 0.204956, -0.118062, -0.173021, -0.164582, 0.480514, 0.099995, 0.055116, 0.55745, -0.809624, -0.096127, 0.044087], [0.198832, 0.154557, 0.039733, 0.06665, -0.266738, 0.057944, -0.004729, 0.214957, 0.096358, -0.249095, 0.068715, -0.23999, 0.148259, -0.626766, 0.185805, 0.135718, -0.21481, -0.181468, -0.214009, 0.267318, 0.266989, -0.076759, 0.283838, -0.101772, -0.32195, 0.366569, -0.396343, 0.249289, -0.191532, -0.021639, -0.206916, -0.052095], [0.018596, -0.119608, -0.127348, -0.082271, -0.277369, -0.373746, 0.146153, 0.106315, 0.098963, -0.109127, 0.190279, -0.309431, -0.074776, 0.028963, -0.138257, 0.286785, -0.167089, -0.001485, 0.31226, -0.184303, -0.242053, -0.009399, -0.004035, 0.086549, -0.109581, -0.035418, 0.019463, 0.1582, -0.267555, 0.340813, -0.472243, -0.45467], [-0.275859, -0.23048, 0.177142, -0.190517, 0.295114, 0.001349, -0.114769, 0.15252, -0.101506, -0.202362, 0.257917, -0.101359, 0.234945, 0.183268, -0.202768, -0.279887, -0.136854, -0.189507, 0.200511, 0.346131, -0.002236, -0.049197, -0.422291, 0.071601, 0.034536, -0.194213, -0.9367, -0.010148, 0.095869, -0.146322, 0.392517, -0.391137], [0.321324, -0.292355, -0.26738, 0.013032, -0.34076, -0.199515, 0.076415, 0.158847, 0.06441, 0.101662, 0.213174, -1.008948, 0.041122, -0.544189, -0.324799, -0.126916, 0.274627, -0.036433, 0.216607, 0.028616, 0.149666, 0.121149, -0.31337, -0.000731, -0.074434, 0.087868, -0.126135, 0.136, -0.090821, -0.073083, 0.053909, -0.525243], [0.021199, 0.278347, -0.068263, 0.103309, -0.254143, -1.097629, -0.117986, -0.109391, 0.195098, -0.369647, -0.364778, 0.064414, -0.675982, 0.07809, -0.264196, -0.37529, -0.342982, -0.297433, -0.170675, 0.499081, -0.383341, -0.596168, 0.057501, 0.053061, -0.269365, 0.147712, -0.110381, -0.481112, 0.069246, -0.014572, 0.412915, -0.074442], [0.028059, 0.259115, -0.231355, 0.095422, 0.189764, 0.243298, 0.048472, 0.048912, -0.229154, -0.303871, -0.104671, 0.21823, -0.00106, -0.251534, -0.022329, -0.276341, -0.178851, -0.196859, -0.01769, -0.305434, 0.062074, 0.291605, -0.189656, 0.171565, -0.129258, 0.16632, 0.182736, 0.118849, -0.221725, 0.090237, 0.434157, -0.348355], [-0.002508, 0.119339, 0.080609, 0.259833, -0.147848, 0.093557, 0.03585, -0.009553, -0.151637, 0.237315, 0.471377, 0.064796, -0.000726, 0.466191, 0.219365, -0.136317, -0.071405, -0.301461, 0.145552, 0.468236, -0.21092, -0.177499, 0.215091, 0.114202, 0.367272, -0.35504, -0.002669, 0.117162, -0.078369, 0.075615, -0.318439, 0.211395], [-0.178841, -0.163258, -0.680119, 0.049624, 0.25844, 0.132895, -0.252954, 0.110976, -0.320058, 0.259271, -0.639125, 0.251309, -0.379406, -0.066575, -0.005399, -0.365224, -0.32725, 0.039907, -0.328551, 0.272511, 0.28492, -0.243449, 0.189847, 0.127773, -0.08835, -0.238443, -0.28757, -0.356087, -0.083869, -0.245085, 0.019378, 0.149641], [0.240983, 0.034443, -0.340417, 0.122171, 0.288503, -0.156079, 0.054179, -0.056365, 0.281897, -0.021742, -0.15834, -0.324806, -0.026582, 0.109791, 0.048967, -0.075263, 0.059996, -0.231263, -0.288555, -0.001502, -0.107446, 0.270547, -0.960302, -0.20134, 0.031333, 0.013179, 0.010976, 0.581888, -0.007685, 0.016518, 0.093547, -0.136682], [-0.347207, -0.683552, -0.042779, 0.160244, 0.118368, 0.103061, -0.357844, 0.343808, 0.21441, 0.148987, 0.211937, -0.250448, 0.223398, 0.209235, 0.179484, 0.259784, 0.417796, 0.243946, -0.010942, 0.187398, -0.539644, 0.017886, 0.192527, -0.013008, 0.335594, -0.053438, -0.020347, -0.395471, 0.448987, 0.073219, -0.162919, 0.23856], [0.345259, -0.061007, -0.264732, -0.069228, 0.039207, -0.063788, -0.231961, -0.212193, -0.226191, 0.055247, -0.000938, 0.254785, 0.094331, -0.605011, 0.166699, -0.145667, 0.294049, -0.095984, -0.159948, 0.167284, 0.039171, -0.081039, 0.157972, 0.3387, -0.026798, 0.078318, -0.528582, -0.344797, 0.054074, 0.016961, -0.207982, 0.090521], [-0.532577, 0.101672, -0.758718, 0.186109, -0.084759, -0.048222, 0.137804, 0.166655, 0.08161, 0.178586, 0.266999, 0.109942, -0.242509, -0.021199, -0.121056, 0.259917, -0.095197, 0.216062, 0.539625, -0.082874, -0.271612, -0.340868, -0.04385, -0.051445, -0.031155, -0.035506, 0.069927, -0.034166, -0.123634, 0.187329, 0.16511, 0.231235], [-0.629516, -0.051737, 0.369821, -0.136478, 0.090267, 0.259889, 0.288913, 0.081753, 0.305638, 0.01054, 0.096119, 0.215147, -0.367492, -0.299307, 0.144632, -0.006332, 0.099495, 0.395305, 0.44629, -0.461714, 0.046468, 0.025822, 0.301785, 0.112622, 0.0496, -0.286583, -0.175468, 0.372095, 0.055147, -0.695457, 0.087275, 0.043337], [-0.176278, 0.429209, -0.028535, 0.08081, -0.014262, -0.275038, -0.317418, 0.101879, -0.018368, -0.048872, 0.13078, 0.016101, -0.264002, -0.185087, 0.239754, -0.03479, -0.015873, -0.010177, 0.462323, 0.204534, 0.348119, 0.045624, 0.461372, 0.069797, 0.099641, 0.422357, 0.211419, -0.223214, 0.312739, 0.17229, -0.82686, -0.607931], [0.196271, 0.010855, -0.302997, 0.155591, -0.186348, 0.097919, -0.367069, -0.204758, -0.26868, 0.149867, 0.041896, -0.57066, -0.055082, -0.060462, 0.148531, -0.199121, 0.334809, -0.47759, -0.272992, 0.200353, -0.396755, 0.15114, 0.153862, 0.483576, -0.29983, -0.158038, -0.304626, -0.370268, 0.177434, -0.06521, -0.270326, -0.198483], [0.387213, -0.416109, -0.040205, -0.105378, 0.025969, 0.078517, 0.019702, 0.398043, -0.340258, 0.082123, 0.029769, -0.345533, 0.27375, -0.321636, 0.086751, -0.231785, 0.047099, -0.431529, -0.085658, 0.246197, -0.173937, 0.151494, 0.045809, 0.492244, -0.431233, -0.041385, 0.152538, 0.126371, -0.206652, 0.589895, -0.252126, -0.038835], [0.198173, -0.104461, 0.050816, -0.208626, -0.274956, -0.838238, 0.167966, 0.193981, -0.219125, 0.14607, -0.031467, -0.331841, -0.131648, -0.054676, 0.026944, -0.220816, -0.05886, -0.056833, 0.270586, 0.037068, -0.205457, -0.22475, -0.223723, 0.045887, 0.407288, 0.391638, -1.082246, -0.357448, -0.316044, -0.126684, -0.094802, 0.074611], [-0.159487, 0.160758, -0.173292, -0.108593, 0.122243, -0.1094, 0.333274, 0.131293, 0.255577, -0.253978, -0.840763, 0.427016, -0.567236, 0.070279, 0.255464, -0.104026, 0.233273, -0.080986, -0.68628, -0.144382, 0.299457, -0.003799, 0.070418, 0.23467, 0.087791, -0.306961, 0.070357, 0.320589, 0.171431, -0.074957, 0.303766, -0.215748], [-0.045072, 0.089938, -0.236163, -0.157858, 0.113413, -0.003215, 0.106172, 0.24992, 0.141671, -0.220015, -0.036084, 0.005753, -0.325388, 0.123563, -0.152579, -0.446839, 0.250956, 0.025714, -0.559178, -0.023809, 0.185398, -0.049931, 0.106965, -0.106895, -0.468226, 0.201998, -0.200493, 0.171192, -0.089666, 0.072474, 0.080657, 0.054981], [0.094372, -0.163708, -0.329422, 0.459939, -0.320406, -0.228401, -0.236543, -0.098295, -0.135237, 0.402016, 0.156124, 0.36787, 0.007904, 0.008657, 0.011832, -0.36276, -0.297867, 0.255703, -0.194505, 0.224125, 0.070795, -0.055719, -0.036136, 0.091953, -0.16218, 0.105716, -0.484773, 0.155931, -0.337147, -0.500374, -0.016086, -0.05932], [0.330444, -0.189932, 0.109269, 0.205973, -0.342373, 0.131495, 0.133533, 0.253258, 0.117391, -0.461535, -0.101579, -0.101155, 0.462235, -0.070538, -0.135551, 0.282487, 0.160029, -0.426186, 0.311707, 0.219851, -0.591661, -0.134719, -0.001272, -0.104474, 0.184514, -0.309823, 0.107345, -0.307234, -0.076305, -0.244336, -0.450563, -0.049978], [-0.335643, -0.249234, -0.176703, -0.111037, 0.290639, 0.088942, 0.082173, -0.011034, 0.329017, 0.234969, 0.154876, -0.273557, -0.038143, -0.338441, 0.145746, -0.101051, -0.265956, -0.145383, -0.094773, 0.257314, -0.026373, -0.102579, -0.006232, 0.189252, 0.183572, 0.009159, -0.196107, 0.051168, 0.338302, -0.000663, -0.168607, -0.776711], [0.120919, 0.283359, 0.260728, 0.087056, -0.27575, 0.166211, 0.348575, -0.14581, -0.011597, -0.263728, -0.315245, 0.343864, -0.129687, -0.032147, 0.312888, -0.038676, -0.031472, -0.228397, -0.155152, 0.327369, -0.441654, -0.233331, -0.296746, 0.32757, -0.43287, 0.19485, -0.202676, 0.04585, 0.038498, -0.25678, -0.177791, -0.219627], [-0.145726, 0.271924, 0.223119, 0.272347, 0.162909, 0.546225, -0.014326, 0.071692, 0.322029, 0.426489, 0.319605, -0.111951, -0.265945, 0.114036, -0.331001, 0.172368, -0.158823, 0.415591, -0.302282, 0.162456, 0.182224, -0.885733, 0.090162, -0.214975, 0.259111, -0.055615, 0.148568, 0.35986, 0.075575, -0.219575, 0.210904, -0.158071], [-0.052434, 0.292664, -0.226797, 0.421661, -0.124568, -0.034311, 0.079854, 0.155158, 0.264586, 0.341765, 0.13102, -0.068611, -0.018233, -0.053245, -0.125642, 0.215386, 0.210667, 0.139191, -0.097932, -0.263154, 0.233803, 0.380283, -0.18673, 0.089692, -0.312624, -0.428994, 0.144982, -0.056052, -0.505198, 0.123398, -0.350374, -0.388325], [0.201454, -0.022413, 0.414003, -0.23485, -0.009536, 0.197941, -0.159349, 0.2627, -0.23721, -0.071089, 0.038081, -0.000175, 0.269043, 0.227728, 0.071292, -0.410593, -0.206622, 0.249155, 0.007269, 0.11792, 0.093063, 0.071097, 0.013857, 0.088493, 0.002374, 0.51966, 0.222002, -0.551139, 0.30858, -0.014072, -0.116278, 0.067769], [0.148937, -0.274944, 0.318175, -0.316864, -0.160237, 0.175215, -0.022721, 0.418933, 0.044016, -0.301345, 0.069022, -0.123007, 0.105166, -0.134231, -0.557943, -0.029662, 0.273407, 0.213204, 0.113182, 0.051334, 0.170653, 0.148327, 0.567942, 0.256695, 0.270889, 0.000258, -0.267066, 0.420897, -0.292223, -0.067285, -0.218481, 0.431901], [-0.295692, -0.010431, -0.29695, 0.022936, 0.187791, 0.026557, -0.244397, -0.052569, -0.639339, 0.195561, -0.179018, -0.234216, -0.079644, 0.141751, 0.200838, -0.282235, 0.195899, 0.121801, -0.159993, -0.105304, -0.07112, -0.234504, -0.151807, -0.155616, 0.00496, -0.221405, -0.152959, -0.220629, 0.165194, 0.297622, -0.15664, 0.290865], [0.044933, 0.059508, 0.362437, -0.186072, -0.150102, 0.184252, -0.437011, -0.266135, 0.089289, 0.272539, 0.075512, 0.146864, -0.048049, -0.00477, 0.333572, 0.154161, -0.089915, 0.191777, 0.054237, -0.20713, -0.446634, 0.101766, -0.063066, 0.238517, -0.069862, -0.081555, -0.751463, -0.391577, -0.114371, 0.190254, 0.471325, 0.085594], [-0.008583, 0.082551, 0.101727, -0.160283, -0.014856, -0.441082, -0.132812, -0.19236, -0.041673, 0.22539, -0.098451, -0.101367, -0.420342, -0.130746, -0.246225, 0.136191, -0.065498, -0.496776, -0.096029, -0.09728, 0.009643, -0.564865, -0.105734, -0.255257, -0.048867, -0.35991, 0.12803, -0.064, 0.198721, -0.167357, -0.048605, -0.010537], [0.133167, -0.032455, 0.289281, -0.058383, -0.151391, -0.64242, 0.025854, 0.214936, -0.223745, 0.06811, 0.20503, 0.06555, -0.16568, -0.297882, 0.024859, 0.182244, 0.29425, 0.18566, -0.439318, -0.30582, -0.562439, -0.277651, -0.022876, -0.38167, -0.313728, -0.067348, -0.101295, -0.098195, -0.102536, 0.145742, -0.160018, -0.25541], [0.028284, 0.248662, -0.224471, -0.433475, 0.349321, 0.184303, 0.130792, 0.222072, -0.125259, -0.278526, 0.147007, 0.111161, 0.104723, -0.234785, -0.088435, 0.040239, 0.397747, 0.002062, -0.138971, -0.484855, 0.223578, -0.022861, 0.078652, -0.181299, 0.030296, -0.124653, 0.115837, -0.04651, 0.142744, 0.122823, 0.242937, 0.336785], [-0.271187, 0.501497, -0.095931, -0.133711, 0.417388, 0.119888, -0.309354, 0.146742, -0.181159, 0.309887, -0.144747, 0.089551, 0.176406, -0.323297, -0.001019, 0.381899, -0.569767, -0.15239, -0.203084, -0.154518, -0.02521, 0.203399, 0.098196, -0.02753, 0.220876, 0.122724, 0.108911, -0.322164, -0.025299, 0.154498, 0.335101, 0.14659], [-0.067907, 0.367326, -0.179261, 0.039574, -0.131972, -0.342015, -0.052898, 0.409867, -0.053287, -0.189486, 0.134226, 0.121276, 0.11248, -0.154762, 0.192957, -0.179884, 0.258178, 0.22968, -0.09019, 0.196064, -0.034867, -0.017912, -0.519655, 0.039198, -0.152892, 0.632674, 0.181945, 0.138148, 0.156472, -0.011096, -0.074632, 0.154646], [-0.046302, 0.203722, 0.376415, 0.320976, -0.066045, -0.022109, 0.011289, 0.39117, 0.391317, 0.324563, -0.143602, -0.055039, 0.308365, -0.013179, -0.624845, -0.067036, 0.028055, 0.163815, 0.187492, -0.250699, -0.089389, -0.058597, -0.048725, -0.020802, 0.17003, -0.178866, -0.8416, 0.183258, -0.443796, -0.420841, 0.215767, -1.02218], [-0.180159, -0.254616, -0.007833, 0.049943, -0.266602, -0.369231, 0.509849, 0.136909, -0.390203, 0.26502, -0.691895, 0.721577, -0.295853, -0.618886, 0.297866, 0.132066, 0.135359, 0.188257, 0.15219, 0.135069, -0.294148, 0.020099, -0.165942, -0.276115, -0.220714, -0.042935, -1.415944, 0.044934, 0.009346, -0.772034, -0.337767, -0.108479], [-0.182998, 0.126406, 0.125028, 0.443131, -0.028992, -0.181089, 0.208515, 0.235857, -0.063176, 0.272273, -0.034905, -0.648328, 0.126849, 0.020771, 0.659754, 0.045936, -0.092636, 0.10912, 0.313788, -0.166794, -0.024176, -0.139847, -0.157302, 0.058992, 0.387409, 0.422779, 0.199833, -0.413416, -0.289988, 0.489542, 0.431878, 0.301313], [-0.171157, -0.266766, -0.22294, 0.199273, 0.292967, -0.162127, -0.044481, -0.199228, -0.20634, 0.012793, 0.166383, -0.171275, 0.156465, -0.051811, 0.184673, 0.001762, 0.061079, 0.036705, 0.066802, -0.078313, -0.332811, 0.403988, 0.013523, -0.153623, -0.118839, -0.351649, 0.389615, 0.15793, 0.099012, -0.390437, 0.299258, 0.057923], [-0.464311, -0.125666, 0.49907, -0.156491, -0.184387, 0.022933, 0.090608, 0.076937, 0.220881, 0.155819, 0.400452, 0.072178, -0.254896, 0.160857, -0.138862, 0.254208, -0.267475, -0.371354, -0.190664, 0.05318, 0.257576, 0.25984, -0.325396, -0.148368, 0.171103, -0.041379, -0.260261, -0.057916, 0.098104, -0.15611, -0.522328, -0.551852], [-0.248747, -0.068358, 0.46533, 0.105882, 0.064476, 0.672202, -0.15947, -0.457668, 0.445876, 0.078001, 0.10752, 0.025712, 0.154502, 0.081034, -0.056748, 0.005935, 0.069971, 0.72962, 0.219604, -0.139303, 0.311591, -0.179772, 0.360633, -0.196551, 0.040841, 0.060156, -0.118656, 0.384151, -0.273962, -0.075547, 0.19413, 0.058941]], [0.291867, -0.442745, -0.167776, 0.308974, -0.160586, 0.187953, -0.006169, 0.087022, 0.149252, -0.598222, -0.004143, -0.332386, -0.328824, 0.219318, -0.426263, -0.120471, 0.092101, -0.389409, -0.152715, 0.207709, -0.224423, -0.030474, -0.34584, 0.298297, -0.063497, -0.460205, 0.377731, -0.106305, 0.153888, -0.147343, 0.039629, -0.12533]),
    ([[0.295608], [-0.461136], [-0.397657], [0.267735], [0.342341], [0.395596], [-0.540792], [0.18837], [0.16844], [-0.551576], [0.347461], [0.714373], [0.235282], [-0.435047], [0.420865], [0.479766], [-0.29936], [0.819905], [0.516763], [-0.297484], [0.524297], [-0.18983], [-0.47469], [0.213333], [0.189759], [0.665278], [-0.850343], [-0.363045], [0.276668], [-0.399549], [-0.570649], [-0.475742]], [-0.220886]),
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
