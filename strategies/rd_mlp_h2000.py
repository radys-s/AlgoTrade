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
MLP_MEAN = [0.000121, 5.1e-05, 3e-05, 1e-05, 1.1e-05, -5e-06, 0.000873, -9.4e-05, 0.001838, 1.269711, 0.000212, 0.00019, 5.945857, 5.943495, -2.8e-05, 0.000535, 4e-05, 4.9e-05, 0.008506, 0.032405, 0.190195, 4.055322]
MLP_SCALE = [0.959699, 0.818775, 0.773899, 0.680568, 0.596571, 2.162757, 3.896449, 0.635368, 8.681322, 0.808209, 0.643569, 0.536525, 1.629162, 1.629007, 1.0, 4.39194, 1.060236, 0.43395, 129.11998, 325.775116, 942.022949, 1.345009]
MLP_LAYERS = [
    ([[0.16597, 0.229166, 0.034063, 0.035284, -0.028209, -0.022313, -0.216143, 0.264129, 0.178706, -0.106876, 0.351027, 0.090444, 0.049789, 0.355328, -0.265974, -0.043923, -0.284271, 0.040759, 0.213854, 0.053676, 0.198061, 0.027729, 0.082339, 0.063478, -0.228663, 0.21522, -0.073061, 0.215942, 0.095261, -0.077181, 0.038258, 0.090632], [0.140761, 0.118681, -0.428267, 0.089343, 0.009944, 0.141255, 0.212293, 0.119567, -0.243794, 0.050966, 0.21178, -0.2835, 0.120979, 0.183984, -0.127728, -0.167673, -0.205476, -0.241414, 0.037852, -0.031671, 0.213638, -0.271659, -0.072554, -0.216938, 0.099804, -0.106714, -0.032718, -0.199537, -0.117092, -0.259457, 0.108708, -0.169503], [-0.073409, -0.109347, 0.155153, -0.245181, 0.147741, -0.139291, 0.310598, -0.077653, 0.204109, 0.123501, 0.236939, -0.300737, -0.170723, -0.251452, -0.153301, -0.249245, -0.200512, -0.12714, -0.257473, 0.136341, -0.024295, -0.150905, 0.054245, -0.174741, 0.088233, 0.267702, -0.133157, 0.092908, -0.173182, 0.220206, -0.158818, -0.132349], [0.052427, -0.55601, 0.322361, -0.246457, 0.154201, 0.010871, 0.308587, 0.155933, -0.162739, -0.009543, 0.086181, 0.048127, -0.313912, 0.228942, -0.259501, 0.144861, 0.059491, -0.016821, 0.302848, -0.148825, 0.284328, -0.047585, 0.092188, 0.313712, 0.237452, -0.169923, 0.331235, 0.090369, -0.128469, 0.230344, -0.265525, -0.029334], [0.084308, -0.362925, 0.199035, 0.074463, 0.109609, 0.021384, 0.172413, 0.000602, 0.106404, 0.094117, -0.139148, -0.189567, -0.016263, 0.01425, -0.238125, 0.291133, -0.027529, 0.192183, -0.262393, 0.146502, 0.060968, 0.201826, -0.3861, 0.247448, -0.253455, 0.002723, -0.273313, 0.157914, 0.068791, 0.046863, 0.039172, 0.00897], [0.237049, 0.017061, 0.066423, 0.257862, 0.252351, 0.276878, -0.391173, -0.135724, 0.091045, -0.182701, 0.147502, -0.38139, -0.30761, -0.240814, 0.193935, -0.113429, -0.259161, 0.236814, 0.16416, -0.341887, -0.298981, -0.005162, 0.134422, -0.109789, 0.308461, -0.034289, -0.073052, 0.026909, 0.221038, -0.16083, 0.018788, -0.081111], [-0.19222, 0.046985, 0.131249, 0.242255, -0.192659, 0.356848, -0.112822, -0.181655, -0.2483, 0.391312, 0.025814, -0.331068, -0.756473, -0.761869, -0.199767, 0.066274, -0.145078, -0.147247, 0.375072, -0.456227, -0.175352, -0.103654, -0.473782, 0.21947, 0.010916, -0.39293, 0.381229, 0.175502, 0.365084, 0.362618, -0.15043, 0.360126], [-0.032274, 0.217927, -0.038948, -0.086834, 0.359268, -0.033607, -0.364111, -0.097599, -0.012843, -0.180312, 0.004276, -0.401187, -0.051979, 0.032812, -0.050002, -0.194018, -0.078827, -0.297704, 0.205524, -0.131431, 0.122065, -0.160166, -0.12783, -0.097398, -0.154602, -0.028101, -0.066626, 0.144075, 0.203783, -0.082605, 0.07234, -0.293436], [0.102588, 0.157705, 0.092948, -0.151063, -0.229254, 0.177955, -0.503832, 0.194474, -0.0965, -0.115497, -0.007983, 0.229865, -0.282982, -0.222098, 0.280198, 0.26067, -0.25175, 0.224626, -0.299537, 0.048154, 0.399822, 0.422909, 0.144779, 0.508886, -0.09493, 0.390799, 0.060676, -0.522677, 0.108448, 0.00606, 0.464356, -0.342702], [0.240651, -0.006705, -0.038231, -0.39079, 0.554581, 0.082431, 0.070485, -0.225774, -0.029523, 0.258336, 0.336999, 0.273493, 0.431206, -0.11731, -0.059027, 0.043909, -0.214207, 0.218461, -0.287215, -0.238939, 0.23979, -0.824124, -0.10821, -0.091358, -0.095301, -0.250587, 0.315369, -0.049858, 0.216735, 0.053961, 0.26219, 0.307204], [-0.245587, -0.052768, -0.314121, -0.141071, -0.060261, -0.006002, 0.130662, 0.166158, -0.208148, -0.123033, -0.191871, 0.115379, -0.348571, 0.217158, 0.217829, 0.120687, -0.339083, -0.34278, 0.181722, 0.075964, -0.281064, -0.380904, -0.133167, -0.267643, 0.07289, -0.343717, -0.058795, -0.29615, -0.078359, -0.013484, 0.093389, -0.262742], [-0.412528, -0.323765, 0.341634, 0.078474, -0.16083, -0.409723, 0.231893, 0.141486, -0.193575, 0.2431, -0.054715, 0.093271, 0.132221, -0.123568, -0.112114, -0.315165, 0.50978, 0.253239, 0.291712, -0.270798, 0.134577, -0.143193, -0.125469, -0.140238, 0.181509, 0.696338, 0.128414, 0.006122, 0.337519, -0.006888, 0.370828, -0.040648], [-0.002154, 0.469821, 0.214905, -0.099895, 0.193281, 0.138052, -0.114644, 0.077492, -0.170585, 0.10951, -0.270186, 0.010504, -0.185581, -0.082296, -0.400366, -0.357547, -0.129711, -0.044945, -0.066798, 0.301961, 0.253937, -0.02987, -0.056915, -0.266192, -0.218719, -0.076581, -0.503973, -0.297635, -0.2918, -0.241453, -0.088373, -0.089243], [-0.259433, -0.074292, -0.056416, 0.27451, -0.09905, 0.666118, 0.133289, -0.093319, 0.039766, -0.078344, 0.387576, -0.350929, 0.252958, -0.237812, -0.155351, 0.121577, -0.218311, 0.107619, -0.063288, -0.204424, 0.273427, -0.063783, 0.132262, -0.069682, -0.044475, 0.422104, -0.096795, -0.127551, -0.217103, -0.087834, 0.141283, 0.558833], [0.220274, 0.416622, -0.211318, 0.389096, -0.569751, -0.411243, 0.5251, -0.487701, 0.198511, -0.125856, -0.044375, 0.226285, -0.29103, -0.176765, -0.260169, -0.010787, 0.041795, -0.136783, 0.014714, -0.096243, 0.302635, -0.131791, -0.185253, 0.395369, 0.216458, 0.237861, 0.260746, 0.422383, -0.189034, 0.392626, 0.4191, 0.115736], [-0.23158, -0.064741, -0.19008, -0.036655, 0.019195, 0.219026, 0.113321, 0.045495, 0.305222, 0.182561, -0.17619, -0.033652, 0.072318, -0.08354, -0.050647, -0.118414, -0.097593, -0.107007, 0.11111, 0.007533, -0.084219, -0.225935, -0.118901, 0.190929, 0.063943, 0.096216, 0.094317, -0.163107, -0.166117, 0.250793, -0.268953, 0.181479], [0.097734, -0.101919, 0.286577, 0.713057, -0.222684, -0.115525, 0.102666, 0.128165, 0.286417, -0.450491, -0.200346, 0.090344, -0.607511, -0.455773, 0.164946, 1.145665, -0.308694, 0.046606, -0.222908, -0.395728, 0.152335, 0.089743, -0.140301, -0.551944, -0.077416, 0.062176, 0.194809, 0.735307, -0.591985, 0.205543, 0.361047, 0.260413], [0.404903, -0.014744, 0.169016, 0.116009, -0.181276, -0.312018, 0.062008, -0.396756, 0.189551, 0.215747, 0.025249, -0.314819, -0.158075, -0.04982, 0.130455, -0.119773, 0.3289, -0.337843, 0.445456, 0.225867, 0.161912, 0.887566, 0.07545, 0.318865, 0.327123, 0.187021, 0.203756, -0.358268, 0.328325, 0.082897, 0.238588, -0.624132], [-0.165537, -0.020082, 0.055951, 0.013365, -0.298082, 0.115016, -0.045632, -0.332803, 0.054995, -0.006171, -0.064543, -0.119434, -0.098334, 0.201, 0.006336, 0.387546, 0.174275, 0.00046, 0.059272, 0.129504, -0.079812, -0.186423, -0.078148, 0.306587, -0.253993, -0.100799, 0.322382, -0.075499, 0.037057, 0.18557, -0.056125, -0.361999], [-0.025039, -0.025243, -0.040954, -0.669199, 0.374164, 0.017822, -0.39409, -0.224373, 0.218655, -0.047583, 0.014832, 0.462658, -0.048304, -0.022003, 0.006597, 0.069336, 0.499765, 0.460746, 0.300863, 0.045205, -0.011623, -0.040596, -0.525684, 0.020169, -0.232165, 0.051339, 0.049801, 0.374411, -0.279196, -0.469985, 0.011193, 0.028004], [0.389514, -0.163835, 0.4093, -0.035419, -0.317928, -0.138205, 0.269663, 0.348231, 0.290429, -0.13835, -0.130503, 0.12676, -0.226802, -0.527127, 0.345628, -0.385153, -0.328964, -0.453427, 0.023076, 0.350858, -0.256498, 0.020777, -0.441351, -0.044603, -0.070046, -0.45944, -0.066372, -0.168957, 0.417824, 0.346548, 0.103683, 0.048007], [0.1435, -0.054959, -0.033378, 0.054184, 0.418576, -0.004199, -0.026613, -0.217511, -0.408821, -0.42636, -0.122889, 0.413193, -0.33119, 0.066217, 0.095596, 0.172002, 0.456625, -0.332089, -0.031703, -0.176508, 0.0829, -0.12247, -0.049157, 0.031619, -0.022536, -0.356415, 0.14968, 0.16383, 0.282637, 0.135111, -0.223625, -0.050446]], [-0.064594, -0.416463, -0.586657, 0.031412, -0.097808, 0.084222, -0.48737, 0.240289, 0.129136, 0.063364, -0.450314, -0.268337, 0.020648, -0.248961, -0.079625, 0.201436, 0.110828, 0.435428, -0.50372, 0.587843, -0.19062, 0.274218, 0.003293, 0.4248, 0.209583, 0.096499, -0.226894, -0.238402, -0.126356, -0.407466, -0.245678, 0.05642]),
    ([[-0.129156, -0.097422, -0.431891, 0.190926, -0.381552, 0.003563, -0.147502, -0.338709, -0.171639, 0.030736, -0.124593, -0.059855, 0.418677, 0.165426, -0.401357, 0.066109], [0.228186, -0.181583, -0.394513, 0.305308, -0.24463, -0.588046, -0.51275, 0.234933, 0.204871, 0.292501, -0.048717, -0.072267, 0.188431, 0.421746, 0.267215, -0.110766], [0.505105, 0.235624, 0.179153, -0.622513, -0.044057, -1.057126, 0.150354, -0.694567, 0.018038, -0.153048, -0.478426, 0.260646, -0.083578, 0.095571, 0.424572, 0.612863], [-0.520856, -0.240417, -0.279679, -0.135923, -0.054608, -0.22026, 0.304125, -0.674683, 0.382686, 0.043711, 0.364003, 0.339922, -0.067236, -0.310286, -0.055827, -0.312769], [-0.345941, 0.198878, 0.043409, 0.219235, 0.289278, 0.247665, 0.418245, -0.155728, 0.198392, 0.26385, -0.12328, 0.106628, -0.322963, 0.233467, -0.398982, 0.001926], [-0.314431, -0.03143, 0.164857, -0.031022, -0.184501, -0.632824, 0.007542, -0.139224, -0.410959, 0.343121, -0.33701, 0.209367, 0.111442, -0.220807, -0.050499, -0.607739], [0.298501, 0.103099, 0.18386, 0.207292, 0.443485, -0.67265, -0.04149, 0.435962, 0.066292, 0.155889, -1.390634, 0.081971, -0.099478, 0.406526, 0.122373, 0.101853], [-0.153861, 0.031766, 0.307416, 0.130175, -0.249003, 0.102964, 0.121651, 0.08023, 0.114045, -0.01481, -0.038725, -0.099928, 0.426804, 0.489637, -0.447698, -0.100205], [0.078715, 0.332216, -0.350558, -0.11054, 0.216192, -0.008737, 0.41495, -0.115068, 0.087435, -0.095054, -0.36048, -0.062572, 0.019064, -0.154588, 0.232292, -0.030708], [-0.000435, -0.351602, -0.021789, 0.294823, 0.017569, -0.034799, 0.263711, 0.15914, -0.160773, 0.534822, -0.190162, 0.010619, -0.008612, 0.010047, 0.028481, -0.391721], [0.299551, 0.082112, -0.027752, 0.043064, 0.283187, -0.088108, -0.181825, -0.176193, 0.160036, 0.223998, -0.062839, -0.036718, -0.19018, -0.39564, 0.056031, -0.40077], [0.386131, -0.513104, 0.31404, 0.281384, 0.024153, -0.261153, 0.006314, -0.148449, 0.140475, -0.047653, -0.132786, 0.236109, 0.187242, 0.261518, 0.299953, -0.027172], [0.162104, -0.221786, 0.25728, -0.416851, -0.245723, -0.277752, 0.175575, 0.380284, -0.072658, -0.149649, -0.135778, 0.24793, -0.311204, -0.160421, -0.228787, 0.128971], [-0.460206, -0.311281, 0.232724, -0.48479, 0.181252, 0.365433, -0.579193, 0.361207, -0.288256, -0.333999, -0.026441, -0.1868, 0.326337, 0.104787, 0.213559, -0.113333], [-0.21932, 0.401596, -0.266691, 0.3755, -0.01478, -0.134345, 0.048807, 0.256146, -0.347582, 0.305507, 0.185932, -0.265159, 0.041245, 0.157733, 0.379362, -0.158973], [0.002304, 0.022863, -0.473007, 0.221497, 0.317698, 0.11841, 0.241414, 0.137176, 0.345514, 0.149703, 0.389042, 0.086754, -0.241499, 0.424299, -0.42783, -0.028256], [-0.254235, -0.317686, 0.188562, -0.638025, 0.1408, 0.138957, -0.247289, -0.019921, -0.337286, -0.599342, -0.119012, 0.172928, -0.536609, 0.219415, 0.075393, -0.313908], [-0.141478, 0.233285, 0.013998, -0.548082, -0.132995, -0.0035, -0.407192, -0.253035, 0.160513, -0.487013, 0.003221, 0.1732, -0.074694, 0.169272, -0.243937, 0.464124], [0.225504, 0.284402, 0.011605, -0.51117, 0.411795, -0.461457, -0.100581, -0.404654, 0.332579, 0.1958, 0.154711, -0.31855, 0.071623, -0.222239, -0.485777, 0.051833], [-0.502555, 0.159723, 0.118306, 0.143486, -0.817207, -0.202062, 0.252365, 0.221431, 0.072379, 0.259501, 0.03053, -0.330681, 0.075046, -0.2524, 0.273372, -0.184794], [-0.087845, 0.196784, 0.150116, 0.316034, 0.219844, 0.208249, -0.401203, -0.626464, -0.068751, -0.256667, -0.022463, 0.367967, 0.068992, -0.143907, -0.346125, 0.541499], [-0.262921, -0.118851, 0.089436, -0.217726, -0.541447, 0.133481, 0.250202, 0.023874, -1.335151, 0.206788, -0.617409, 0.262788, 0.12316, 0.288155, -0.278205, -0.034177], [0.162418, -0.225005, -0.087822, -0.267539, 0.214429, 0.289889, -0.550951, 0.327064, -0.021741, 0.501536, 0.238228, 0.11217, -0.099377, 0.477698, -0.285018, 0.002631], [0.281416, 0.121181, -0.035664, -0.284763, -0.125115, 0.151236, 0.035326, -0.165903, -0.229714, 0.010756, -0.039477, -0.146573, -0.103222, -0.627006, 0.163085, 0.316149], [0.100078, 0.275961, 0.305685, -0.328107, 0.127324, 0.228398, 0.192265, 0.109786, -0.024963, 0.08141, -0.24972, -0.243526, 0.306538, 0.128829, 0.121964, 0.032867], [0.194729, 0.09158, 0.103143, 0.275098, -0.646746, -0.421476, -0.279035, -0.444073, -0.239981, -0.0811, -0.278686, 0.051201, 0.009326, 0.081105, -0.088194, 0.079163], [-0.280947, -0.216379, 0.323772, 0.265927, -0.195261, 0.188943, 0.058367, -0.287347, 0.23939, 0.085162, 0.113273, 0.227378, -0.664373, -0.36377, -0.005875, -0.064389], [0.224346, 0.257999, -0.298573, 0.200108, 0.032636, -0.117273, 0.47583, -0.023539, 0.079567, -0.178955, 0.150947, -0.400316, 0.132144, 0.32437, 0.272879, -0.049612], [-0.161391, -0.068436, 0.123697, 0.046867, 0.332592, 0.275998, 0.47297, 0.329355, -0.736055, 0.166785, -0.322868, -0.010008, 0.156376, -0.29602, -0.188957, -0.42476], [-0.00077, 0.479073, 0.215912, 0.252366, 0.095894, -0.311815, 0.159035, 0.445826, -0.283225, 0.612612, -0.765791, 0.090088, 0.186912, -0.358675, -0.024872, -0.039061], [-0.246556, -0.208548, 0.271531, -0.071809, 0.200949, 0.449808, 0.322796, -0.081468, 0.369777, -0.364075, -0.177747, 0.210973, -0.064998, 0.110402, -0.118858, -0.313841], [-0.427127, -0.182099, -0.403274, -0.08456, -0.145608, -0.037234, -0.150699, -0.144647, -0.181513, -0.123074, 0.283429, -0.319367, 0.331778, 0.106051, -0.188032, -0.116735]], [0.333072, 0.096775, 0.151961, 0.116626, -0.133549, 0.406704, 0.159309, -0.37991, -0.200401, 0.215925, 0.177267, 0.279184, -0.302572, 0.28379, -0.125914, -0.002455]),
    ([[0.660759], [0.624988], [0.548304], [-0.772783], [0.74874], [0.433715], [-0.285588], [-0.772062], [0.690429], [-0.221419], [0.786667], [0.272349], [0.482139], [-0.520431], [0.464689], [0.662246]], [-0.050308]),
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
