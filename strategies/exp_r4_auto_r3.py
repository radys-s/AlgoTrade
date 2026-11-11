"""exp_r4_auto: mm_r3_gru with the learned tables (ridge,mlp,mq,slow,seq) refitted by research/retrain.py on round3.csv + mirror (first 75 %).

mm_r3_gru (home, R3 upload candidate): mm_r3_msq + the streaming GRU taker predictor (seq_on 1): round3 634,519 / mirror 624,778 / round2 437,941; holdout 156,075 / 157,441 / 109,156 (msq 571,085 / 595,553 / 439,016; holdout 144,549 / 143,063 / 109,960); check PASS 6.7 min.

mm_r3_msqs (home, R3): mm_r3_msq + the streaming GRU taker predictor from exp_r3_seq (seq_on; 0 = mm_r3_msq).

mm_r3_msq (home, R3 upload candidate): mm_r3_ms (leaderboard 925,809) with the 500 ms MLP feeding the passive quote signal
(mq_h 500): round3 571,085 / mirror 595,553 / round2 439,016; holdout 144,549 / 143,063 / 109,960 (ms 143,885 / 142,614 / 107,749).

exp_r3_msq (home, R3): mm_r3_ms + a 500 / 1000 ms MLP option for the quote signal (mq_h, from exp_r3_mlpq; mq_h 0 = mm_r3_ms).

mm_r3_ms (home, R3 upload candidate): mm_r3_mlp2 (leaderboard 864,770) + the MLP in the passive quote signal (mq_on 1, mq_b 0.5,
fq_k 0.1) + a learned 5 s direction pull on top of the 30 s gate (slw_on 1, slw_h 5, slw_thr 2).
round3 573,111 / mirror 589,261 / round2 428,199; holdout 143,885 / 142,614 / 107,749 (mlp2 138,223 / 137,230 / 108,183).

exp_r3_seq (home): mm_r3_mlp2 with seq_on 1 = the taker MLP replaced by a GRU (13-dim event token -> 16 hidden,
state carried over every market event) + the same 22 hand features into a 38-64-32-1 head (research/diag/seq_r3.py,
lite tokens, T 128 training windows, run streaming). seq_on 0 = identical to mm_r3_mlp2.

mm_r3_mlp2 (home, R3 candidate): mm_r3_mlp with the g3 passive thresholds (fq_k_sp2 2.0, fq_k_sp3 1.5, rt_ms 130, gate_thr 11) and
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

from bisect import bisect_left, bisect_right
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
    # rfit 19 (auto: h200 on round3 + mirror, first 75 %, research/retrain.py); params rfit=19
    (0.005415, (0.3303941, -1.9259269, 4.2404875, -1.4825975, 0.5955986, -0.4271199, 0.0057687, 0.0997469, 0.0100156, -0.0028265, -0.1671455, 0.2063794, 0.0448473, -0.0450412, -0.0610664, 0.0078129, 0.11599, -0.4641638, 0.0004312, 0.0003986, -1.65e-05, -0.0001859)),
)
# exp_r3_mlp: 22-32-16-1 ReLU MLP on the same 22 features, target = 200 ms (robust mid at t+5ms+200ms - mid now), fitted on the
# first 75 % of round3 + mirror (research/diag/mlp_r3.py); holdout corr 0.49 vs 0.463 for the ridge. Inputs clipped at +-8 sd.
MLP_MEAN = [0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233382, -0.0, -0.0, 5.906055, 5.906348, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057909]
MLP_SCALE = [1.019957, 0.879338, 0.830501, 0.723114, 0.628298, 2.28085, 12.9744, 0.673953, 10.69429, 0.757186, 0.651131, 0.545643, 1.651632, 1.651633, 0.999999, 4.399308, 1.05635, 0.427403, 127.451202, 323.701813, 938.297913, 1.343849]
MLP_LAYERS = [
    ([[0.057589, 0.122838, -0.312254, -0.135112, 0.071813, -0.111099, -0.251455, 0.13492, 0.115257, -0.117717, -0.098615, -0.212675, 0.044512, -0.219254, 0.27896, -0.09905, -0.020231, -0.446123, 0.170767, 0.1623, -0.417038, 0.115868, 0.073885, 0.004259, 0.490328, -0.189138, 0.167676, 0.108073, 0.514745, -0.187065, 0.180692, -0.029679], [0.059135, 0.160024, -0.236551, -0.074541, -0.082012, 0.201047, 0.116887, -0.321176, 0.018902, 0.015247, 0.145444, 0.119104, -0.054295, 0.201154, -0.078952, -0.203143, 0.151009, -0.095378, -0.018543, 0.345313, -0.258851, -0.090327, 0.210597, 0.146658, -0.400083, -0.060905, -0.001304, 0.091981, 0.119383, 0.1767, -0.022521, 0.022412], [-0.284534, -0.134534, 0.01453, 0.090413, 0.135636, -0.071217, 0.29134, -0.0001, 0.151357, 0.141732, -0.006921, 0.283826, 0.054677, 0.073631, -0.160924, 0.262558, 0.13149, 0.356416, -0.170173, 0.160863, -0.120117, 0.052622, 0.220872, -0.067394, -0.236317, 0.317621, 0.042593, -0.150827, 0.182021, 0.025564, 0.174438, 0.159933], [-0.017443, 0.08395, 0.159786, 0.232655, 0.233472, 0.187114, -0.00732, -0.04437, -0.205541, 0.318049, -0.042162, -0.090026, 0.087022, 0.042514, -0.120529, 0.059765, 0.029197, 0.44226, 0.07016, -0.173992, 0.182735, 0.202473, 0.047107, 0.140367, 0.129453, -0.031801, 0.235545, 0.189183, -0.186417, -0.079833, -0.356184, -0.012267], [-0.024677, 0.076832, 0.111958, -0.095509, 0.181376, 0.017232, -0.113838, 0.063678, -0.010086, -0.061174, -0.329034, -0.027186, 0.347161, -0.201962, 0.121416, -0.163283, -0.018153, 0.351853, 0.422887, 0.016288, -0.176139, 0.097465, 0.012285, -0.141145, 0.359701, -0.005921, -0.09849, 0.263086, -0.399495, 0.127065, 0.03734, -0.219574], [-0.072199, -0.237139, 0.00623, -0.04055, 0.268269, -0.15715, -0.049119, -0.11415, 0.035169, -0.131229, 0.075321, -0.003081, 0.175085, 0.063884, 0.012368, -0.038518, 0.110876, 0.268346, 0.084694, 0.18181, 0.148851, -0.067333, -0.071255, 0.034126, -0.546523, 0.14067, -0.072768, 0.176468, 0.115991, 0.082117, 0.235356, -0.272543], [-0.078394, -0.284843, -0.277255, -0.081573, 0.102616, 0.548724, -0.071495, -0.202194, 0.067854, -0.374068, -0.197018, 0.709621, -0.036918, 0.393341, -0.195578, 0.254841, 0.274373, 0.313418, -0.15497, 0.180069, 0.089274, -0.106639, 0.028001, 0.223051, 0.132426, 0.081773, 0.26788, 0.160741, 0.087902, 0.409821, 0.223776, 0.347419], [0.031242, -0.008439, 0.15175, 0.092605, 0.074744, -0.022366, 0.150256, -0.102852, 0.105545, 0.101862, -0.017278, 0.03435, -0.239839, -0.085853, 0.090279, 0.128578, 0.040135, 0.052751, 0.103476, -0.072444, 0.07096, 0.013394, 0.021988, -0.097978, 0.163229, 0.191646, 0.170234, -0.150108, 0.054943, 0.207719, 0.030493, -0.137418], [-0.18501, -0.421422, -0.495397, 0.02295, -0.313991, -0.365513, 0.153737, -0.603943, -0.436434, 0.234106, -0.12154, 0.061615, 0.029133, -0.302601, -0.100926, 0.536998, -0.786785, 0.550864, -0.356058, 0.131291, -0.308733, -0.528007, 0.201675, 0.097651, -0.106802, -0.104194, -0.322152, 0.193566, 0.054002, 0.240654, 0.299513, 0.411727], [0.356023, 0.050796, -0.160607, 0.350455, -0.761844, 0.200499, 0.271168, -0.671874, 0.357522, -0.044118, -0.060612, 0.246771, 0.189507, -0.401699, 0.205403, 0.428362, -0.551816, -0.1357, -0.130569, -0.572364, -0.37036, 0.428012, -0.502518, 0.107646, 0.229021, 0.216697, -0.087387, 0.025702, -0.059239, 0.383857, 0.090964, -0.123193], [0.091286, -0.088685, 0.124653, 0.093103, -0.116216, 0.110973, 0.080349, 0.065817, 0.045582, -0.088034, -0.085837, 0.039737, -0.092361, -0.039677, 0.006887, -0.084014, -0.042732, -0.097266, 0.188361, -0.068734, 0.044697, 0.153509, 0.148199, -0.01746, -0.261151, 0.092095, -0.248776, 0.05572, 0.183769, -0.168534, -0.198515, 0.071436], [-0.334156, -0.142148, -0.443118, -0.053594, -0.401184, -0.178609, 0.057153, -0.320025, -0.521553, 0.38804, 0.073101, -0.102757, -0.003765, -0.321987, 0.01471, 0.140827, -0.00134, 0.092068, -0.037144, 0.141824, -0.019472, -0.191449, -0.1186, 0.036616, 0.180263, -0.302827, -0.024175, -0.18476, -0.075114, -0.081411, 0.141378, 0.429172], [-0.025909, -0.099854, 0.33011, -0.182015, -0.054118, -0.183252, -0.090951, -0.164531, 0.007646, 0.202964, 0.189531, -0.22793, -0.035388, 0.007073, -0.107701, 0.282554, 0.229269, 0.014371, 0.10121, -0.158994, 0.038039, 0.030395, -0.258151, 0.291649, -0.163878, -0.134999, -0.271707, -0.112681, -0.116628, -0.098378, -0.255187, 0.00302], [-0.232992, -0.147551, -0.182179, -0.391493, 0.395578, 0.061081, -0.055256, -0.05219, 0.258216, 0.043369, -0.017712, -0.357736, -0.160785, 0.172862, 0.208644, 0.083304, 0.236895, -0.001338, 0.148413, -0.042886, 0.057062, -0.126202, -0.0778, -0.040177, -0.046467, 0.149308, -0.073481, -0.055288, 0.069489, -0.100508, -0.16399, -0.413413], [-0.062343, -0.158, -0.015009, -0.637261, -0.054652, -0.329983, 0.714564, -0.348119, 0.005838, 0.003325, -0.036845, -0.25809, -0.019641, -0.050104, -0.63974, -0.437272, 0.102581, -0.272574, -0.359293, 0.266627, -0.262431, 0.184962, -0.164813, 0.205595, 0.04177, -0.068616, -0.350465, -0.315149, 0.075583, 0.203256, -0.150362, 0.173869], [-0.103839, 0.222216, 0.131739, 0.375276, -0.057765, -0.161765, -0.137625, -0.083176, 0.008817, 0.059743, -0.027592, -0.059666, 0.30249, -0.058637, 0.155752, 0.279219, -0.082624, -0.097359, -0.043959, 0.16877, -0.009006, 0.257136, 0.131592, -0.277183, -0.136834, 0.137328, -0.074653, -0.233095, -0.300436, -0.264784, 0.281856, -0.096653], [-0.019348, 0.111486, -0.22008, 0.434186, -0.119544, -0.229268, -0.180082, 0.540028, -0.159587, 0.193347, 0.132319, 0.068074, 0.117733, -0.666592, 0.285242, 0.343819, -0.11223, -0.055142, 0.64087, -0.198537, -0.057054, -0.493411, -0.024642, -0.130572, 0.02058, -0.336587, 0.61262, -0.242943, 0.077469, -0.024512, 0.322569, -0.086244], [-0.035897, 0.110723, 0.476737, -0.071828, 0.019153, -0.691842, -0.402745, -0.130959, 0.142999, -0.067757, -0.077475, -0.317723, -0.225146, 0.085876, 0.093168, -0.016728, -0.039735, -0.134602, -0.106161, 0.280156, 0.097536, 0.119726, 0.245767, 0.108419, 0.423291, -0.028069, 0.216893, 0.194542, 0.69178, -0.362038, -0.286756, -0.161874], [-0.141174, 0.065724, -0.062273, -0.037183, -0.050354, 0.048752, -0.127112, 0.014129, 0.07331, -0.033942, -0.103249, -0.116183, 0.04548, -0.036888, -0.007937, -0.067767, 0.000885, 0.068051, -0.09941, -0.037484, 0.035805, 0.011556, -0.036565, 0.01885, -0.057755, 0.053375, -0.072698, 0.209421, 0.024468, -0.079597, -0.004792, 0.013392], [-0.197514, -0.930538, 0.092857, -0.085624, -0.054034, -0.23802, -0.102593, 0.046106, -0.065719, -0.064056, -0.069524, 0.366505, 0.118799, -0.013143, 0.213902, -0.006343, 0.233168, 0.021892, -0.339944, -0.006444, -0.073358, -0.245854, 0.156314, -0.121712, -0.002497, 0.232121, -0.440015, 0.1262, 0.027223, 0.3223, 0.178419, 0.105054], [0.15592, -0.065832, 0.254361, -0.109425, -0.268785, -0.14788, 0.144875, -0.018832, -0.067713, 0.174457, 0.272353, 0.092446, -0.022687, 0.02031, 0.063475, -0.04643, 0.053661, -0.217227, -0.177221, 0.004524, -0.152515, -0.09746, 0.082181, 0.085632, 0.024788, 0.454411, -0.102295, -0.227245, -0.032106, -0.043917, 0.025131, 0.032214], [0.04726, -0.131572, -0.028031, -0.238386, 0.045726, 0.098223, -0.0058, -0.068766, 0.066615, -0.091557, 0.473352, 0.24707, 0.0335, -0.180531, -0.46923, 0.207437, -0.347738, 0.018969, 0.038301, -0.078745, 0.095246, -0.030164, -0.281397, -0.603509, 0.004744, -0.008124, 0.128879, 0.485672, -0.053473, -0.048539, -0.104477, 0.239069]], [-0.32663, -0.056943, -0.18609, -0.025908, -0.378897, -0.03282, -0.109793, -0.289295, -0.565711, -0.690349, 0.516985, -0.392105, 0.297331, -0.005679, 0.109346, 0.226425, 0.385954, -0.164186, 0.012863, -0.362455, -0.128863, -0.270656, 0.151589, -0.141836, -0.014206, -0.205831, 0.035625, 0.175025, -0.144726, 0.129538, -0.059536, 0.053323]),
    ([[0.146175, -0.127923, 0.008227, -0.161901, 0.191138, -0.183359, 0.20237, -0.119729, -0.033425, 0.075651, 0.185839, 0.108064, 0.066529, -0.008193, -0.396395, -0.114487], [-0.074395, 0.093295, -0.519957, -0.160442, -0.162009, -0.431383, 0.123293, -0.512053, 0.101171, -0.096581, 0.114777, 0.036462, 0.076774, 0.017873, 0.221021, 0.025355], [0.073298, 0.130692, 0.270469, -0.021085, 0.008883, 0.385567, 0.249307, 0.342311, 0.18474, 0.170665, 0.395948, 0.186187, -0.160424, 0.084331, -0.211812, -0.025598], [-0.563754, -0.262502, -0.056708, 0.082828, -0.14629, 0.237977, -0.143199, -0.074244, 0.021877, 0.026028, 0.022003, 0.17055, 0.109459, -0.360814, 0.243071, 0.283112], [-0.436404, 0.086724, -0.401848, -0.07362, -0.180135, -0.388113, -0.409719, -0.137895, -0.001342, -0.173225, 0.03409, -0.404683, -0.605133, 0.351307, 0.106554, 0.208377], [0.044695, 0.242539, 0.316373, 0.291567, -0.091823, 0.378459, 0.231577, -0.058864, -0.006964, 0.21042, 0.18169, 0.318474, 0.062024, -0.156981, -0.244928, 0.24866], [0.149822, -0.059632, 0.290113, -0.020068, 0.141003, 0.169664, 0.351492, 0.132098, 0.301121, 0.094704, 0.158439, 0.23075, -0.04943, -0.061903, 0.112753, -0.607229], [-0.110207, -0.382834, -0.151617, -0.14384, 0.020256, 0.318363, 0.166595, -0.083462, 0.18581, 0.167312, 0.254169, 0.003476, -0.591464, 0.249908, -0.076825, 0.045227], [0.208917, -0.048909, 0.126761, -0.241213, -0.328512, -0.125796, 0.336893, 0.070248, 0.267306, 0.343146, 0.268771, 0.078938, 0.046985, -0.33093, 0.234026, -0.011538], [0.253354, -0.520112, 0.028942, 0.190719, 0.203179, 0.067851, -0.190086, 0.15366, -0.50718, 0.224223, -0.073142, -0.807031, 0.191007, 0.068848, -0.08502, -0.031839], [0.136469, 0.074381, -0.163868, -0.067989, -0.200475, -0.356413, 0.001033, -0.072287, -0.165433, 0.088129, 0.046408, -0.139948, -0.688188, 0.087073, -0.121125, -0.601118], [0.10533, -0.314232, -0.310781, -0.001752, -0.13986, -0.053427, -0.308666, -0.048053, -0.253395, 0.173307, 0.079856, -0.384542, -0.316193, -0.411735, -0.459171, -0.032792], [-0.095014, -0.090291, 0.116588, 0.025497, 0.137814, 0.142484, 0.131668, 0.149193, 0.117252, -0.477149, 0.164584, -0.10264, -0.019917, -0.16967, 0.190621, 0.179556], [-0.014489, 0.258529, -0.164789, 0.065732, 0.17234, -0.397529, -0.203582, 0.141396, 0.044314, 0.222667, 0.322825, 0.025423, 0.199321, -0.73608, -0.349516, -0.317704], [-0.315164, -0.176475, 0.206842, 0.096411, -0.125326, 0.251086, -0.105011, 0.095681, -0.083799, -0.512004, -0.024374, 0.168383, -0.012735, 0.321775, 0.013508, 0.393787], [0.15519, -0.091705, -0.188147, 0.149174, 0.122663, -0.100283, 0.099811, 0.00749, -0.603514, -0.273261, 0.040035, 0.058854, -0.259004, 0.037376, 0.032256, 0.307518], [0.279016, -0.013793, -0.25532, 0.027616, -0.100064, 0.060567, 0.195863, 0.071502, -0.342406, -0.201187, -0.007267, -0.491303, -0.089041, -0.042938, -0.175671, -0.116041], [0.071113, -0.491195, 0.286151, 0.140649, 0.158217, 0.084099, -0.006667, 0.276253, 0.297112, 0.568834, -0.182358, 0.544314, 0.39311, 0.265733, -0.07173, -0.014035], [0.114398, -0.004511, -0.158189, -0.021968, -0.297537, 0.108993, -0.50131, -0.484662, -0.099223, -0.242377, 0.183856, -0.503296, -0.265826, -0.04226, -0.016132, -0.213891], [0.106359, 0.100962, 0.120828, 0.216343, 0.254771, -0.005036, -0.001085, 0.140967, 0.221838, 0.046745, -0.35735, 0.218255, -0.184207, 0.304264, -0.442769, -0.471558], [-0.015631, -0.136682, 0.154099, -0.108827, 0.216095, -0.180904, 0.037254, -0.046165, 0.132981, 0.169972, -0.065335, -0.015216, -0.058763, -0.468215, -0.068694, -0.227753], [0.216275, 0.215045, -0.197977, -0.050884, 0.08891, -0.306863, -0.197449, -0.266356, -0.015043, 0.160572, 0.027439, 0.135801, 0.3256, -0.754263, -0.293231, 0.120342], [-0.04651, 0.076099, 0.119079, 0.131601, 0.193961, 0.041882, -0.061397, 0.116764, -0.099602, 0.000459, 0.070817, 0.143228, -0.531528, -0.106321, -0.072849, -0.232369], [-0.212639, 0.099195, -0.117997, -0.106156, -0.185051, -0.354516, -0.157971, -0.122527, 0.128503, 0.364868, 0.135185, 0.021403, -0.144648, -0.165423, 0.229244, -0.233344], [0.160893, 0.004738, 0.082537, -0.151932, 0.035807, 0.108661, -0.045144, 0.261721, 0.417314, -0.264302, 0.13851, 0.01911, 0.085337, 0.271637, -0.035793, -0.084406], [-0.022023, 0.034247, -0.098616, -0.204918, -0.190697, -0.101302, -0.547549, -0.282, -0.178076, -0.001721, 0.077588, -0.704119, -0.737776, -0.109122, 0.168801, -0.005002], [-0.002614, 0.104105, -0.112299, 0.280096, -0.175211, -0.47606, 0.131176, -0.281624, -0.007499, 0.135361, 0.117847, -0.161935, 0.161416, -0.421039, -0.092102, 0.021079], [0.069787, 0.06229, -0.177803, -0.080941, 0.046386, 0.083283, 0.030653, -0.064642, -0.011301, -0.457697, -0.089128, -0.300211, 0.191114, -0.279054, -0.006488, -0.231552], [0.087382, -0.375622, 0.198364, 0.225488, 0.204542, 0.238528, 0.313006, 0.181098, 0.394407, -0.100549, -0.033482, 0.4254, 0.032172, 0.0822, 0.01635, 0.065973], [0.09597, -0.563422, 0.001148, 0.134499, 0.213915, -0.173942, -0.253038, 0.178038, -0.23184, 0.107144, 0.169314, -0.254948, -0.285226, -0.114688, -0.009387, -0.198874], [-0.102001, -0.023587, 0.180268, 0.015046, -0.010024, 0.13445, 0.244872, 0.072152, -0.278772, -0.205345, 0.119244, -0.337285, 0.316087, 0.281567, -0.021725, 0.03839], [0.412602, -0.20185, -0.093551, 0.121821, 0.123292, -0.088039, 0.169762, 0.210264, -0.12765, -0.134195, -0.123058, -0.103384, -0.02099, 0.142539, -0.155787, -0.210299]], [0.022569, 0.223395, 0.070785, 0.306243, -0.070732, -0.224149, 0.186668, 0.16079, 0.024459, -0.264014, 0.136755, 0.182315, -0.460452, -0.243845, 0.383995, -0.30438]),
    ([[0.317035], [0.403658], [0.428266], [0.408501], [0.377566], [0.684566], [-0.298961], [0.441487], [-0.660709], [-0.785168], [-0.167881], [-0.600672], [1.221852], [1.223876], [0.441834], [0.959501]], [-0.352126]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, -7.546108291001272e-19, 7.546108291001272e-19, 0.0, 0.0, 1.6978743654752861e-18, 0.0, 0.0, 0.0, 1.232240613980502, 0.0, 5.966187927815806, 14.89545073636175], [4.84636859198212, 11.75951390089026, 19.444828455429125, 25.35692689478354, 34.69348778951227, 1.023785965479704, 0.8835995739934038, 0.8341675367341286, 0.7258298308787036, 0.6303098602816982, 939.3928403585505, 1665.0692101965788, 2975.0770665754285, 0.5677248883233125, 0.43221930100946554, 10.133645385319308, 20.520703908685007], [-0.30162275415038114, 0.02823086719450577, -0.06996426376957159, -0.03961008184054012, 0.003341342517940403, 0.6996990466473364, -3.8994252920921424, 6.68434148334, -4.354653838744496, 2.4396694005261232, 0.231590644527058, -0.09355699340554056, -0.18760311208474853, -0.008353986196611391, 0.38980241135999855, 0.0015446629860642585, 0.0004657713673523611], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, -7.552361978829417e-19, 7.552361978829417e-19, 0.0, 0.0, 1.6992814452366187e-18, 0.0, 0.0, 0.0, 1.2323782729860528, 0.0, 5.969883379890588, 14.899700394933951], [4.848318109281616, 11.764292345260863, 19.451727900064906, 25.36585477477367, 34.7066439494089, 1.0240868561669956, 0.8838581897273222, 0.8344085493754315, 0.7260306565079572, 0.6304745466287232, 939.6747578125994, 1665.547042902018, 2974.5372827629403, 0.5678948771639639, 0.4323983605128832, 10.137014689510544, 20.528651114869255], [-0.15786874499637382, 0.2747852896221565, -0.7038221822907381, -0.2257156563518991, 0.0037097989813588183, 0.3887871050456339, -0.30905971967311574, 1.6286919405225677, -0.8860508726649116, 0.6063206771629822, 0.3573265603031138, -0.255600756384128, -1.057809738347582, -0.012269982157182408, -0.5156724413798738, 0.0022674487048791395, 0.0006840598087049105], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233398, -0.0, -0.0, 5.906047, 5.906341, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057889], [1.019985, 0.879362, 0.830525, 0.723134, 0.628317, 2.280914, 12.974841, 0.673973, 10.694654, 0.757209, 0.651131, 0.54564, 1.651657, 1.65166, 0.999999, 4.399385, 1.056378, 0.427418, 127.455505, 323.712799, 938.328491, 1.343882], [([[0.098303, 0.057711, -0.491203, -0.114987, -0.048539, -0.094077, -0.552011, 0.186135, 0.022372, -0.217187, -0.24767, -0.155052, 0.133155, -0.115068, 0.181657, -0.135483, -0.045124, -0.316095, -0.017267, 0.043359, -0.236652, 0.041261, -0.046999, 0.023219, 0.065861, -0.136947, 0.053953, 0.117413, 0.32917, -0.145253, 0.099174, -0.052308], [0.08299, 0.140149, -0.252298, 0.012884, -0.065465, 0.179998, 0.086653, -0.301672, 0.034858, 0.1016, 0.137058, 0.078267, -0.168516, -0.102559, 0.002982, -0.143646, 0.02419, -0.213814, -0.022661, 0.225605, -0.25625, -0.082534, 0.245047, 0.059619, -0.104507, -0.090843, -0.074837, 0.054127, 0.137068, 0.185046, -0.017284, -0.029461], [-0.277966, -0.095097, 0.042144, 0.105937, 0.192428, -0.106866, 0.438278, 0.004503, 0.272054, 0.200278, 0.069188, 0.231072, -0.139387, -0.231569, -0.188804, 0.293379, -0.039813, 0.173426, -0.10713, 0.054641, -0.111651, 0.141685, 0.262041, -0.087214, 0.209166, 0.285673, 0.04264, -0.214121, 0.230091, 0.009913, 0.13344, 0.072262], [-0.060176, 0.209934, 0.244784, 0.298859, 0.189299, 0.1818, 0.296994, -0.103869, -0.121785, 0.314504, 0.075275, -0.132725, 0.05892, 0.060994, -0.32952, 0.059487, -0.070305, 0.240704, 0.083377, -0.237832, 0.256724, 0.312193, 0.066242, 0.238201, 0.367943, -0.10437, 0.371235, 0.07125, -0.243838, -0.22157, -0.443189, 0.134296], [-0.147393, 0.07064, -0.171774, -0.232336, 0.19061, -0.086346, -0.113617, -0.018026, -0.187022, -0.262124, -0.25835, 0.037782, 0.430567, -0.042796, 0.209606, -0.097947, 0.034511, 0.482321, 0.385193, 0.136808, -0.215949, -0.232776, 0.037551, -0.208899, -0.016033, -0.124171, -0.087244, 0.192202, -0.594396, 0.228791, 0.308036, -0.022004], [-0.098979, -0.31658, -0.171743, 0.00168, 0.219468, -0.278021, 0.15605, -0.262726, 0.068599, -0.096626, 0.042126, -0.08442, 0.034468, -0.148125, 0.067853, 0.038653, 0.018843, 0.154791, 0.267701, 0.149889, 0.03971, -0.038328, 0.029697, -0.022962, -0.131148, -0.152999, -0.061884, 0.199084, 0.229205, 0.142972, 0.225251, -0.02338], [-0.067625, 0.091927, 0.227943, -0.255134, 0.388771, -0.007578, 0.31348, -0.120099, -0.122598, -0.280098, -0.168517, 0.541592, -0.384244, 0.140091, -0.228311, 0.140866, -0.033585, 0.276394, -0.363975, -0.040947, -0.114513, -0.283118, -0.123782, 0.252778, 0.377757, 0.379727, -0.150955, -0.100014, 0.036977, 0.643027, 0.031496, 0.320676], [0.02718, -0.046946, 0.181112, 0.077647, -0.102763, -0.004432, 0.24906, -0.072504, 0.047382, 0.263064, -0.177173, 0.033885, -0.068021, 0.023841, 0.027203, 0.149928, 0.09174, -0.011606, 0.113582, -0.061518, 0.147468, 0.137641, -0.028525, -0.09461, -0.007325, 0.260203, 0.078716, -0.239219, 0.037989, 0.195525, 0.001137, 0.100656], [0.051586, -0.306464, -0.34489, -0.327283, -0.231828, -0.006068, 0.687963, -0.548001, -0.529951, 0.032951, 0.099002, 0.05627, 0.084431, -0.05756, -0.220517, 0.581306, -0.038165, 0.41804, -0.107455, -0.170819, -0.071425, -0.252511, 0.181245, -0.021269, -0.377001, 0.209541, -0.405822, 0.029512, 0.570277, 0.161676, 0.334576, 0.087822], [0.387594, 0.185628, 0.06735, 0.213803, 0.07781, -0.149339, -0.198542, -0.723171, 0.294807, 0.092114, 0.197435, 0.2672, 0.013846, -0.417392, 0.174688, 0.196033, -0.201842, 0.292355, -0.015241, -0.785908, -0.125965, 0.440134, -0.360527, -0.213349, 0.090155, 0.018345, -0.036197, 0.267483, -0.012782, 0.428971, 0.214603, -0.042848], [0.001264, -0.07455, 0.205916, 0.00105, 0.003319, 0.022232, 0.080498, 0.016269, -0.122035, 0.095055, -0.046232, 0.061993, -0.047939, 0.014108, 0.119642, -0.221155, -0.179975, -0.263611, 0.143652, 0.121512, 0.132122, 0.281159, 0.118313, -0.100608, -0.170084, -0.217309, -0.192047, -0.120162, 0.060229, -0.162944, -0.345563, 0.077943], [-0.228855, -0.078803, -0.230574, -0.125001, -0.335518, -0.119214, 0.081795, -0.181077, -0.162328, 0.211829, 0.096505, -0.0021, -0.193928, -0.412322, 0.152324, 0.233327, 0.440951, 0.194998, 0.088109, 0.160412, -0.176643, -0.138184, 0.058964, 0.086276, 0.039782, -0.136471, -0.16705, 0.32378, 0.12169, 0.031334, 0.189257, 0.466472], [-0.056253, -0.163486, -0.022703, 0.104407, 0.058825, 0.059527, -0.030598, -0.220407, -0.138182, -0.074452, -0.019013, -0.225451, 0.057344, 0.164442, -0.075324, -0.045749, 0.459017, -0.092259, 0.11423, -0.163466, 0.252939, 0.088279, -0.149067, 0.17748, -0.514063, -0.089919, -0.172493, 0.202247, -0.143113, -0.134846, -0.170981, -0.214827], [-0.233587, -0.130949, -0.085217, -0.24271, 0.391335, 0.154725, -0.08546, 0.03555, 0.489674, -0.077016, 0.055517, -0.299755, -0.359469, -0.197653, -0.021517, -0.010759, 0.146891, -0.065236, 0.02962, -0.046704, 0.130111, -0.098411, -0.037473, -0.046173, -0.020866, -0.1151, -0.10612, -0.090792, 0.116242, -0.246655, 0.015928, -0.24502], [-0.045142, -0.132015, 0.064796, -0.506655, -0.386833, -0.408003, 0.142672, -0.145784, 0.321801, 0.339505, 0.129607, -0.306356, -0.047392, -0.059671, -0.391613, -0.276371, 0.073333, -0.203454, -0.167287, 0.095406, -0.190122, 0.219504, 0.014075, 0.152603, 0.135702, -0.17193, -0.333002, -0.427749, 0.162731, 0.055586, -0.060772, 0.071559], [-0.140493, 0.275982, 0.018443, 0.376945, -0.319416, -0.357993, -0.168332, -0.056305, -0.033778, 0.02586, -0.201024, -0.024139, 0.263798, -0.238686, 0.040643, 0.370781, -0.023928, 0.045982, -0.041277, 0.114839, -0.238259, -0.084534, 0.35321, -0.09185, -0.078687, -0.042764, 0.038105, 0.042793, -0.161811, -0.15528, 0.177346, -0.241931], [0.003534, -0.152364, -0.16458, 0.502584, 0.05759, -0.012804, -0.273985, 0.496553, -0.000804, 0.121322, 0.33473, -0.164127, 0.067585, -0.013097, 0.591291, 0.400488, -0.164037, -0.009503, 0.558954, -0.081662, -0.160088, -0.12174, 0.030626, -0.09797, -0.242537, -0.1494, 0.591559, 0.035849, 0.002023, 0.158978, 0.477467, 0.084869], [-0.05796, 0.33207, 0.239299, -0.840284, 0.199644, -0.333057, -0.098542, -0.345234, -0.127686, -0.049306, -0.528592, -0.227522, 0.018727, 0.14453, -0.000145, -0.001575, 0.018228, 0.060096, 0.398447, -0.081296, -0.042191, 0.103989, 0.177143, -0.296993, 0.187123, -0.227244, 0.096669, 0.048529, 0.042273, -0.186948, -0.000663, 0.407613], [-0.102815, 0.039536, -0.05116, -0.006278, -0.16895, -0.00763, -0.000499, 0.045987, 0.013073, -0.101614, -0.080264, -0.034195, -0.077076, 0.057366, -0.05327, -0.12007, -0.168223, -0.043822, -0.113095, -0.105285, -0.218238, -0.040518, 0.148178, -0.012771, -0.049423, -0.055004, -0.067523, 0.008216, -0.007422, 0.077495, 0.050371, 0.123689], [-0.277578, -0.390751, 0.055455, -0.062544, -0.018839, -0.015043, -0.039613, -0.020784, -0.131209, -0.096675, 0.105441, 0.307948, 0.195893, -0.23956, 0.167554, 0.156211, 0.625203, 0.015872, -0.252285, 0.075728, -0.060831, -0.262619, 0.261944, -0.239448, -0.046726, 0.103423, -0.491407, -0.058288, 0.013516, -0.027525, 0.031589, -0.042729], [0.228172, 0.054449, 0.190072, 0.007249, -0.172976, -0.111441, -0.134435, 0.020365, 0.050155, 0.340907, 0.208982, 0.167899, -0.050115, 0.119619, 0.236664, -0.189414, -0.079428, -0.034605, -0.146703, 0.243809, -0.308969, 0.138859, 0.185312, 0.117051, -0.387953, 0.610081, -0.267725, -0.205001, -0.005388, -0.157618, -0.055421, -0.192531], [0.130138, -0.237001, -0.002297, -0.245043, 0.030957, 0.172716, 0.005507, -0.113324, 0.224582, -0.290862, 0.310408, 0.373984, 0.132326, 0.290774, -0.144311, 0.336915, -0.060161, -0.042744, 0.053581, -0.366631, -0.021073, 0.084623, -0.078324, -0.512068, 0.144746, 0.032186, 0.109668, 0.609112, -0.027845, -0.012814, -0.049966, 0.413016]], [-0.399351, -0.472443, -0.17152, 0.021593, -0.141414, -0.022498, -0.092515, -0.459275, -0.877632, -0.527864, 0.30848, -0.083981, -0.220692, -0.251534, 0.071094, 0.113012, 0.440625, -0.045238, -0.031279, -0.066033, -0.221421, -0.080418, 0.258536, -0.193378, 0.207829, -0.31002, 0.052749, -0.019685, -0.160122, -0.213136, 0.06267, 0.067107]), ([[-0.072774, -0.176474, -0.368832, 0.002229, 0.274899, 0.308137, 0.22984, -0.032491, -0.004597, 0.065343, 0.169582, 0.083571, 0.078166, -0.318702, -0.252021, 0.359498], [-0.697428, -0.266623, -0.282634, -0.242552, -0.294826, 0.054387, 0.205283, -0.117512, 0.058941, 0.066424, 0.167134, -0.037819, 0.114915, -0.132686, 0.026529, 0.06898], [0.213175, 0.17273, 0.174379, -0.246393, 0.049295, 0.230412, 0.303925, 0.329422, 0.23686, -0.103189, 0.156942, 0.108845, -0.288547, 0.038997, -0.127808, -0.078996], [-0.411023, -0.556645, 0.018415, 0.159507, -0.215456, 0.245553, -0.138142, -0.075666, 0.078675, -0.179326, 0.039432, 0.144375, 0.043754, -0.300493, 0.364929, -0.38802], [-0.118181, 0.142207, -0.071956, 0.011754, -0.113004, 0.034887, -0.25241, -0.230888, 0.133977, -0.26638, -0.154214, -0.366981, -0.191228, 0.12265, 0.125512, -0.320911], [0.069386, 0.127605, 0.184008, 0.06545, -0.021142, 0.127181, 0.145167, 0.038522, 0.03914, 0.302174, 0.012898, 0.175681, -0.177642, -0.148509, -0.195555, -0.163058], [0.04173, -0.037896, 0.172795, 0.100042, 0.119309, 0.154409, 0.244694, 0.192313, 0.109409, 0.528514, 0.082711, 0.470222, -0.096242, -0.527914, -0.123805, 0.023449], [-0.179384, -0.198537, -0.175021, -0.119536, -0.105575, 0.354664, 0.213786, -0.172673, 0.070585, 0.141868, 0.180869, -0.045179, -0.716395, 0.124682, 0.162855, 0.248957], [-0.074396, 0.140662, 0.361301, -0.334012, -0.85314, -0.007384, 0.566131, -0.049466, 0.240916, 0.29298, 0.28196, 0.122069, -0.116465, -0.3247, 0.299417, -0.68783], [0.256367, -0.64244, -0.022152, 0.032088, 0.158745, -0.26372, -0.172707, 0.06292, -0.173969, 0.103824, -0.000407, 0.160628, -0.34997, -0.419336, -0.211326, 0.223268], [0.184118, 0.119384, -0.023389, -0.003154, -0.032112, -0.384593, 0.226922, -0.059888, -0.125473, -0.087787, -0.002118, -0.023942, -0.084541, -0.520623, 0.031509, -0.354582], [-0.177703, -0.158369, -0.188214, 0.068356, -0.074373, 0.071194, -0.16208, -0.073988, -0.221727, 0.044057, -0.051861, -0.427707, -0.416111, -0.17775, -0.175524, -0.147494], [-0.227678, -0.571815, 0.008567, 0.041118, 0.053933, 0.115815, 0.18562, 0.151725, 0.214234, -0.017092, 0.277528, 0.103894, -0.243442, 0.327018, 0.11734, 0.274389], [-0.050512, 0.150218, -0.346084, 0.037604, 0.264421, -0.388019, 0.010833, 0.075567, 0.267349, 0.132129, 0.088405, 0.024145, 0.232223, -0.422756, -0.252755, 0.088907], [-0.168011, 0.068638, 0.033175, 0.153779, -0.228247, 0.271492, -0.069632, -0.086618, -0.073474, -0.194006, -0.097693, 0.32559, -0.164367, 0.420491, 0.305698, 0.02289], [0.020269, 0.145283, 0.038002, 0.087968, 0.055165, -0.004199, 0.162911, 0.016273, -0.643011, -0.177843, 0.095303, -0.061406, 0.201561, 0.058098, -0.060637, 0.161485], [0.229628, -0.228746, -0.017119, 0.037526, -0.167265, 0.043381, -0.08441, 0.005602, -0.354639, -0.331123, -0.052387, -0.591844, -0.598134, -0.083193, -0.213455, -0.028419], [0.081376, -0.167458, 0.279144, -0.029086, 0.012914, 0.208922, -0.209604, 0.28436, 0.208153, 0.353162, -0.113016, 0.252013, -0.034742, 0.194274, 0.117964, 0.284723], [0.303773, -0.079525, -0.258864, -0.024008, -0.030641, -0.191109, -0.295361, -0.363084, 0.100641, -0.069616, 0.077685, -0.654144, 0.218069, -0.260196, -0.055177, 0.163408], [0.051198, 0.038268, 0.044571, 0.225099, 0.322262, 0.174909, -0.192346, 0.075619, 0.103966, -0.320096, -0.288858, 0.049745, 0.021037, 0.277122, -0.296604, 0.017985], [-0.122484, 0.024323, 0.005237, -0.129584, -0.027115, -0.496596, 0.089146, -0.080018, 0.052713, 0.240274, -0.238196, 0.047361, 0.012941, -0.014023, 0.018369, 0.182477], [0.139587, 0.198299, 0.094477, -0.272506, 0.027262, -0.293251, 0.27845, -0.019054, 0.138695, 0.092461, 0.107222, 0.395309, 0.246958, 0.041622, -0.133471, -0.098231], [-0.061003, -0.016057, 0.082052, 0.080951, 0.124474, 0.034299, -0.084258, 0.022208, -0.111079, -0.016322, 0.175612, -0.773984, -0.760019, -0.13164, -0.165707, 0.195158], [-0.07839, -0.108596, -0.23979, -0.066422, -0.103703, -0.279272, -0.01422, -0.157885, 0.17923, 0.240052, 0.032226, 0.171441, -0.099973, -0.299855, 0.326706, -0.047601], [0.011075, 0.279335, -0.131223, -0.082553, 0.041175, 0.073347, 0.066539, 0.129573, 0.233626, -0.159946, 0.158911, 0.202797, 0.106736, -0.153149, 0.040745, -0.56013], [-0.169413, 0.003888, -0.052616, -0.185792, 0.005042, -0.347873, -0.393068, -0.307095, -0.183415, 0.170932, 0.086151, -0.548202, 0.311305, -0.231545, 0.155855, 0.100159], [-0.11049, 0.027615, -0.234175, 0.097537, -0.279577, -0.264139, 0.31784, -0.542169, -0.152625, -0.030424, 0.084703, -0.213162, 0.193529, -0.329068, -0.032305, -0.248351], [0.257714, -0.046858, -0.011959, -0.113526, 0.228765, 0.164295, 0.270413, 0.029647, -0.039047, -0.269428, -0.254441, -0.136151, 0.141405, -0.045325, -0.377907, 0.189342], [0.060072, -0.322456, 0.349394, 0.265385, 0.110098, 0.114504, 0.28889, 0.181073, 0.122149, 0.089417, 0.030762, 0.614322, -0.058684, 0.075161, 0.22022, 0.300355], [-0.114853, 0.12449, 0.039397, 0.165739, 0.33679, -0.165241, -0.111377, 0.276027, -0.334082, 0.183517, 0.090443, -0.346536, -0.118248, 0.014467, 0.150667, -0.185539], [-0.001817, 0.192725, 0.279898, 0.028764, -0.036511, 0.132495, 0.211733, 0.102793, -0.312443, -0.417298, 0.125329, -0.218125, -0.046804, 0.28914, 0.009493, -0.095962], [0.233182, 0.045173, -0.302422, 0.010694, 0.062948, -0.217628, -0.118927, 0.047838, 0.231102, -0.186861, -0.097156, -0.121961, -0.079636, -0.342357, -0.125748, -0.142423]], [0.28942, 0.186538, 0.194067, 0.208555, -0.073376, -0.252187, 0.057213, 0.067053, 0.05928, -0.528846, -0.034787, 0.11915, -0.02996, -0.069813, 0.255294, -0.464429]), ([[0.520072], [0.741356], [0.508901], [0.552982], [0.729878], [1.371711], [-0.552157], [0.638925], [-0.869695], [-1.532902], [-0.187695], [-1.211239], [1.615587], [2.293539], [0.764205], [1.50529]], [-0.62891])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.73969, -1.6e-05, 1.6e-05, 0.0, 0.047376, 0.047376, 0.0, 0.0, 0.0, 1.231694, 5.907079, 5.907079, 0.0]
SEQ_TSD = [1.080247, 0.118222, 0.118222, 0.089617, 1.451691, 1.451691, 0.632855, 0.196772, 0.156216, 0.572964, 1.651027, 1.651027, 0.652184]
SEQ_XM = [-0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.233382, 0.0, -0.0, 5.906633, 5.906633, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.05788]
SEQ_XS = [1.020368, 0.879681, 0.830847, 0.723417, 0.628551, 2.281788, 12.978411, 0.674084, 10.704477, 0.755279, 0.651221, 0.545623, 1.651871, 1.651871, 0.999999, 4.399956, 1.05635, 0.427403, 127.463814, 323.861115, 938.76239, 1.34419]
SEQ_YSC = 1.490180253982544
SEQ_WI = [[-0.063672, -0.25791, -0.514315, -0.360631, -0.547316, -0.592458, -0.516213, -0.562651, -0.37759, -0.068542, -0.831015, -0.790268, -0.48112, -0.890628, -0.216625, -0.331622, 0.540606, -0.28143, -0.039745, -0.056916, -0.356588, -0.146426, 0.442499, -0.228928, -0.132944, -0.321226, -0.212894, -0.568153, -0.410465, 0.326723, -0.046889, 0.094519, -0.002737, -0.061397, -0.241799, 0.297741, -0.253589, -0.136411, 0.219342, 0.216864, 0.221509, -0.023243, -0.128762, 0.033807, -0.123511, 0.355129, 0.233694, -0.414913], [0.006224, -0.134661, -0.425047, 0.572864, 0.359771, 0.463249, -0.265426, 0.412036, -0.328978, -0.071944, -0.013751, -0.20735, 0.115186, 0.238729, -0.021339, -0.320621, -0.121781, 0.022823, -0.568014, -0.034603, 0.192208, -0.070949, 0.369053, -0.010141, 0.170818, -0.285707, 0.132043, -0.291841, 0.162712, 0.536923, -0.138395, 0.061627, -0.063597, -0.587077, 0.592772, 0.00586, 0.168347, -0.374395, 0.124168, 0.166315, 0.067165, -0.773557, -0.131824, -0.334535, -0.117845, 0.252964, -0.117867, 0.019755], [0.153219, 0.232041, -0.234232, -0.337705, 0.513845, -0.389489, -0.118561, 0.07636, -0.617485, -0.225896, 0.035093, -0.482726, 0.301143, -0.530021, -0.196016, 0.223054, 0.063042, 0.216843, -0.164767, -0.177893, 0.611343, -0.256263, -0.353494, -0.519862, -0.28825, 0.314482, -0.591019, 0.514152, 0.763723, -0.051458, -0.338221, -0.032346, 0.202419, 0.168341, -0.552118, -0.053129, 0.058666, 0.316639, -0.224092, -0.026529, 0.29137, 0.157961, 0.149976, -0.204095, 0.221082, -0.292693, 0.16873, -0.199449], [-0.320907, 0.055062, -0.202472, 0.209716, 0.230712, 0.184962, 0.569057, -0.04141, -0.789052, -0.213233, 0.044998, -0.25928, 0.029294, -0.053838, -0.053803, -0.016479, -0.045592, -0.097219, -0.640911, -0.192546, 0.464068, 0.330523, 0.055823, -0.071308, 0.047918, -0.133151, -0.166299, 0.039418, 0.011932, 0.187461, -0.323452, 0.360485, 0.192044, 0.162265, 0.029293, -0.232091, 0.462207, -0.073739, 0.169525, 0.039343, -0.233629, 0.018506, 0.436437, -0.20553, 0.334189, 0.200052, -0.335814, -0.016575], [-0.178744, 0.207778, 0.150889, -0.68076, 0.23617, 0.483356, -0.042754, 0.201705, -0.370955, -1.168121, -0.413086, 0.091422, 0.28931, 0.737936, 0.282414, 0.181165, -0.127599, 0.402004, -0.138279, 0.256455, -0.095942, 0.315855, -0.471182, -0.051644, 0.615891, -0.992221, 0.280511, 0.329508, 0.30129, -0.184783, -0.324388, 0.210409, 0.034057, 0.167407, -0.482107, 0.052334, -0.097679, 0.107892, 0.090255, 0.191599, -0.074688, -0.827665, -0.217972, 0.364669, 0.216146, 0.068106, -0.35026, 0.053259], [0.181438, -0.188658, -0.364514, -0.631256, -0.48935, -0.654386, 0.398538, 0.0788, -0.27278, -0.282457, 0.431688, -0.455667, -0.455701, 0.052239, 0.798088, -0.509646, -0.080218, -0.994397, 0.481035, 0.154483, -0.570768, 0.267661, 0.331701, 0.227495, 0.398978, -0.255482, 0.53568, 0.109212, -0.173211, -0.338867, 0.04764, -0.370934, -0.264809, -1.154929, 0.37074, 0.008044, 0.005886, 0.000428, 0.480796, 0.107909, -0.211513, -0.08416, -0.553512, -0.027387, -0.501645, -0.109696, 0.118335, 0.430564], [0.14895, 0.092904, -0.534278, 0.489348, 0.193821, 0.080739, -0.094977, 0.303447, -0.18597, 0.192544, 0.202808, 0.145392, 0.552374, 0.267402, -0.380841, 0.319356, 0.002112, -0.099182, 0.255639, -0.061903, -0.386021, 0.024836, -0.176092, 0.052265, 0.069249, 0.119538, 0.110137, 0.243696, -0.351582, 0.634823, -0.295187, 0.052809, 0.257477, -0.099402, 0.030893, -0.073023, 0.14669, 0.532316, -0.190914, -0.009661, 0.196981, 0.259232, -0.025478, -0.249401, 0.000264, -0.391238, -0.376806, -0.010007], [0.475296, 0.274703, -0.523477, 0.714445, 0.224376, 0.274094, -0.208391, -0.024609, 0.104578, -0.209213, 0.164277, 0.031993, 0.505082, 0.241787, -0.133816, -0.177461, 0.245513, -0.226302, 0.446375, -0.040503, -0.235314, 0.010941, 0.005167, 0.052499, -0.21253, 0.405609, 0.280623, -0.095877, 0.010099, 0.15468, -0.399213, -0.325704, 0.069948, 0.027662, -0.254429, 0.174562, -0.196908, 0.212963, 0.181034, 0.137092, -0.124806, -0.077478, -0.198203, -0.442545, 0.074494, -0.357877, -0.347834, 0.209368], [0.2017, 0.296916, -0.647647, 0.656524, 0.379506, 0.416344, -0.262744, -0.227967, 0.025111, -0.096349, 0.078212, 0.014524, 0.544212, -0.053127, -0.049152, -0.47086, -0.14673, -0.147464, 0.202903, -0.072676, -0.057238, -0.178764, -0.198555, 0.104543, 0.26017, 0.286792, 0.175267, 0.134032, -0.266324, 0.493595, -0.49871, -0.316656, 0.177853, -0.225592, 0.24988, -0.315075, -0.091052, 0.022753, 0.127384, -0.010851, -0.061672, 0.189835, 0.023992, -0.356182, -0.146887, 0.164171, -0.011024, -0.011513], [0.194479, -0.643113, -0.368612, -0.042366, 0.03895, -0.083954, 0.019552, -0.041605, -0.077208, 0.196624, -0.056471, -0.281979, 0.28387, 0.288079, 0.683795, -0.356908, -0.035653, -0.056504, -0.288508, 0.137835, 0.000623, -0.334101, -0.104089, -0.663584, -0.062685, 0.102605, -0.119579, -0.511736, -0.323237, 0.08166, 0.469239, 0.388078, -0.655274, 0.229843, 0.159968, 0.196238, 0.141753, 0.198038, 0.170402, -0.156963, -0.248489, -0.090911, -0.189746, 0.319414, -0.239676, 0.231616, -0.33394, -0.167326], [-0.461688, 0.103655, -0.38701, 0.163323, -0.048147, 0.082727, 0.269658, -0.385318, -0.361424, -0.037358, 0.239713, -0.237137, -0.102448, -0.092947, 0.123982, -0.221088, 0.319678, -0.284659, 0.133882, -0.266465, -0.703753, -0.068282, 0.023999, 0.090024, -0.955163, -0.395579, -0.050259, -0.109512, -0.408222, 0.350494, -0.273019, -0.105678, -0.048187, -0.120645, 0.211872, -0.080239, 0.013878, 0.160757, 0.295742, -0.141636, 0.402703, -0.036472, -0.103896, -0.074979, -0.131945, -0.075958, -0.106208, 0.20833], [-0.155312, -0.244739, 0.083517, -0.090946, 0.177891, -0.454639, -0.312972, -0.654708, 0.084801, 0.005477, -0.120123, 0.151806, 0.784792, 0.065475, 0.376589, -0.469982, -0.349376, -0.216134, -0.516303, -0.485907, -0.259902, 0.118921, -0.497738, 0.002537, -0.568232, -0.11003, -0.107836, -0.31648, -0.559233, -0.208856, 0.153663, -0.449763, -0.262334, -0.168734, 0.066447, -0.049687, 0.100479, 0.068513, -0.118114, 0.024082, 0.254192, 0.032102, -0.099585, 0.108906, -0.001234, -0.222729, -0.041283, 0.068026], [-0.541077, 0.13487, -0.047329, 0.531358, 0.022665, 0.039334, 0.356943, -0.155588, -0.387365, -0.356557, 0.227415, -0.330125, -0.850598, 0.159989, -0.043975, 0.037558, 0.002999, 0.297964, 0.149947, -0.307643, -0.729209, 0.558643, 0.101021, -0.401679, 0.060511, 0.081895, 0.365885, 0.573691, 0.284303, 0.440335, -0.081076, -0.122741, -0.092549, -0.137968, -0.19939, -0.288746, -0.212662, -0.14494, 0.082523, -0.036617, -0.14703, 0.112453, 0.08077, 0.085287, 0.003151, 0.106287, -0.058473, -0.279818]]
SEQ_BI = [0.02743, 0.440348, 0.481677, -0.041872, 0.254574, 0.109349, 0.302931, 0.189871, -0.121573, -0.206353, -0.027563, 0.024091, 0.112224, 0.288008, 0.057126, -0.065663, 0.263339, 0.559711, 0.59846, 0.377206, 0.620957, 1.001435, 1.145357, 0.718324, 1.234877, 0.338472, 0.3679, 0.485474, 0.570638, 0.288179, 0.412057, 0.385537, -0.287597, 0.10706, 0.187713, 0.151614, -0.242915, -0.34383, -0.151574, 0.127066, 0.305487, -0.089034, -0.148083, -0.228485, -0.0004, -0.369442, -0.231589, 0.336363]
SEQ_WH = [[-0.085105, 0.371297, -0.101456, 0.088684, 0.014511, -0.198044, -0.008425, 0.158212, 0.385623, -0.26304, -0.081505, 0.13491, -0.049853, -0.049345, -0.741293, 0.411361, 0.019276, -0.845054, 0.14039, 0.387264, -0.875292, -0.265394, -0.333366, -0.072376, -0.696693, -0.407477, -0.369424, 0.051049, 0.706938, -0.395274, -0.275479, -0.263132, 0.40499, -0.383208, 0.117523, -0.203643, 0.512445, 0.354144, -0.475209, 0.049595, 0.258206, -0.168336, 0.236893, 0.047025, 0.258431, 0.066448, 0.218902, -0.263293], [-0.257007, -0.359314, 0.442398, -0.129047, 0.029848, 0.074037, -0.574512, -0.428699, -0.568998, -0.059036, 0.201723, 0.214267, 0.283087, -0.203199, -0.129047, 0.289674, -0.333692, -0.219751, -0.150305, 0.45108, -0.052674, -0.139071, 0.419814, -0.326038, 0.588165, -0.097735, 0.567738, 0.053708, -0.045573, -0.650689, -0.300751, 0.266955, 0.03329, 0.60043, -0.018377, -0.462138, 0.331156, 0.097483, 0.016203, -0.079547, -0.345681, -0.006977, -0.185859, -0.390665, -0.198189, -0.733761, 0.157543, 0.057513], [-0.491008, 0.089239, -0.777616, 0.113723, 0.474734, -0.118139, -0.18127, -0.042223, -0.052715, 0.669709, 0.215357, 0.017999, 0.487876, -0.024301, 0.228283, -0.327862, -0.17529, 0.346297, 0.837078, 0.046751, 0.103126, 0.552467, -0.275244, 0.66981, 1.560628, -0.009858, 0.623082, 0.617206, -0.186782, 0.530227, -0.077303, -0.149024, -0.030838, 0.449002, 0.305012, -0.221365, 0.014546, -0.234532, 0.461918, 0.269, 0.564044, 0.592637, 0.684149, 0.119608, -0.300897, 0.113966, -0.007712, 0.114181], [-0.137931, 0.051719, -0.611991, -0.02373, -0.691776, -0.141086, -0.287083, -0.245475, 0.609692, -0.11475, -0.996425, -0.593037, -0.197002, -0.304296, -0.127827, -0.496979, 0.271586, 0.568124, 0.208491, -0.00183, 0.00652, -0.060788, -0.226951, -0.10966, -0.39988, 0.469283, 0.207204, -0.657875, -0.34122, 0.53573, 0.351892, 0.103468, -0.259835, -0.28964, 0.141576, 0.330273, -0.408823, -0.481219, -0.32473, 0.526036, 0.227009, 0.125655, -0.304808, 0.031764, -0.540903, -0.170782, 0.017905, -0.014769], [-0.293804, -0.286043, -0.993868, 0.066127, 0.101052, -0.116905, -0.265539, -0.291041, 0.119464, 0.341656, -0.110351, 0.222684, 0.600888, -0.241437, -0.414294, 0.199656, -0.419289, -0.373267, -0.562634, 0.251924, -0.688364, -0.626693, -0.573115, -0.477692, -0.371883, -0.771891, -0.157413, -0.643875, -0.680537, -0.088467, 0.228236, -0.491388, 0.089033, -0.161072, -0.178215, 0.102223, 0.215153, 0.463289, 0.122249, -0.064115, -0.013758, 0.207081, -0.071, -0.200928, 0.37413, 0.152808, -0.084645, -0.071667], [-0.089042, -0.038781, -0.474532, -0.051761, 0.219049, -0.159447, -0.306226, -0.089956, 0.32222, 0.183591, 0.079967, 0.301343, 0.319891, -0.105235, 0.175207, 0.009104, -0.252886, -0.769373, -0.797263, 0.202016, -0.183531, 0.01891, -0.244112, -0.550143, -0.8287, -0.32256, -0.730195, -0.216631, -0.707863, -0.292616, -0.598478, -0.366372, 0.214342, 0.20664, -0.290913, -0.380244, 0.05339, 0.469663, -0.236912, -0.409917, 0.389525, -0.100187, -0.064391, -0.103558, 0.553386, 0.345541, 0.120405, -0.348914], [0.159482, -0.659949, 0.427014, 0.188255, 0.101188, -0.124681, 0.638508, -0.034734, -0.365876, 0.139565, -0.103847, -0.077542, -0.435941, -0.270442, 0.121772, -0.1966, -0.011613, 0.199122, -0.1892, -0.415403, -0.574512, -0.871047, -0.011482, -0.526641, 0.208517, -0.06648, 0.488442, 0.233672, 0.002226, 0.076468, -0.764402, -0.361497, -0.172768, -0.207776, -0.26886, 0.052259, -0.281029, 0.029505, 0.813447, 0.267335, -0.220999, -0.053382, 0.172238, 0.437403, -0.445399, -0.227889, 0.074304, -0.171101], [0.277315, 0.619903, 0.292193, -0.274046, -0.567021, -0.30164, 0.163067, 0.039815, 0.404254, -0.71619, -0.341211, -0.865665, -0.442964, 0.119712, -0.564446, -0.116782, -0.147287, 0.388889, 0.525077, 0.288525, -0.018208, 0.497269, 0.047291, 0.019517, 0.489098, 0.860275, 1.089644, -0.25367, 0.435672, 0.422118, 0.609036, -0.073528, 0.012147, -0.018851, -0.303381, 0.518811, 0.241279, -0.44279, -0.217639, 0.399252, 0.469986, -0.125779, 0.179947, -0.037192, -0.285525, 0.507069, -0.155743, -0.101311], [-0.189589, 0.102354, -0.378382, -0.04241, -0.218885, -0.576426, -0.117002, 0.21606, 0.288898, -0.179733, -0.19903, -0.654257, -0.31446, 0.19761, -0.208573, -0.873625, 0.771493, 0.579898, 0.004741, 0.172476, -0.376273, 0.296912, 0.159166, 0.383207, 0.293806, -0.085767, -0.072886, -0.745916, 0.387065, 0.521616, 1.089366, 0.147232, 0.001021, -0.458323, 0.284943, 0.374401, -0.371613, -0.290023, -0.545555, 0.329856, 0.3195, -0.266214, 0.141938, -0.105461, 0.108935, 0.649946, -0.365161, 0.081846], [0.145536, -0.184017, 0.034779, 0.336114, 0.111346, -0.23467, -0.671252, -0.136213, 0.035012, 0.432642, 0.814537, 0.642999, 0.302423, -0.262612, 0.206915, 0.081285, -0.173492, 0.04, -0.304597, -0.24434, 0.204548, -0.02378, -0.223023, 0.431266, 0.457456, -0.683045, -0.074437, -0.486539, -0.331569, -0.493469, 0.266612, -0.395754, -0.273496, 0.137888, 0.321893, -0.287134, -0.258147, 0.176473, 0.02911, -0.542766, -0.248553, 0.580372, 0.526177, 0.373546, 0.103177, -0.198579, 0.646401, 0.389266], [0.12505, 0.509121, 0.597418, 0.3945, 0.483959, 0.953826, 0.081032, 0.361194, -0.143467, 0.56859, 0.372033, -0.328796, 0.109137, -0.341159, 0.097671, -0.331033, -0.159142, -0.494718, 0.232517, -0.171231, -0.656716, -0.45703, -0.282482, 0.342298, 0.254304, 0.088334, 0.324764, 0.311347, 0.045304, 0.202325, -0.445151, 0.283082, -0.029504, -0.204516, 0.512195, 0.08793, -0.471269, -0.309021, 0.595954, 0.055129, -0.05529, 0.556759, 0.595052, 0.143327, -0.042524, -0.010993, 0.05279, 0.331428], [0.070512, -0.230699, 0.222699, 0.105835, 0.212554, -0.133604, 0.120832, -0.221948, 0.366661, 0.447648, -0.065834, 0.027526, -0.026374, -0.186045, 0.435864, 0.028413, 0.228776, -0.380824, -0.429211, -0.44981, -0.178749, -0.290864, -0.538383, -0.241022, -0.060926, -0.54143, -0.284717, -0.274053, -1.070223, -0.120987, -0.780451, -0.180781, -0.337774, 0.069503, -0.236881, -0.252566, -0.191768, 0.259914, 0.045303, 0.233781, -0.28338, 0.626519, -0.052787, 0.187901, 0.070529, 0.228854, 0.249485, 0.11717], [-0.285545, 0.290201, -0.073843, 0.253534, -0.056598, 0.092673, -0.336244, 0.0929, 0.342467, -0.402253, 0.059541, 0.173902, 0.345129, 0.288817, 0.130652, 0.281285, 0.169294, -0.053774, 0.267913, 0.621728, -0.385262, 0.715966, 0.078203, 0.206662, -0.133563, -0.474508, -0.074418, 0.168496, 0.238267, -0.341012, 0.330806, 0.139567, 0.365734, 0.218655, 0.374722, 0.12707, 0.160678, 0.277776, -0.589399, -0.520546, 0.425558, -0.220288, -0.312654, -0.405477, 0.224866, 0.051392, -0.200607, -0.499511], [-0.108743, -0.082747, -0.567492, 0.229117, 0.190196, 0.126456, 0.705454, -0.26249, 0.278838, 0.644436, -0.505677, -0.579552, -0.505753, -0.013833, 0.490996, 0.084375, 0.54181, 0.230277, -0.622493, -0.587885, -0.503493, 0.842019, 0.155153, 0.665992, -0.456289, -0.693662, 0.434189, -0.126287, -0.527634, 0.216529, -0.19241, -0.070734, -0.327744, -0.375757, -0.144215, 0.059315, -0.426885, -0.099066, 0.045303, 0.08216, 0.376952, 0.259608, -0.227759, 0.271461, 0.083927, 0.301038, -0.214266, -0.05207], [-0.019828, -0.288202, -0.417203, -0.027927, -0.629975, -0.037873, -0.586854, -0.146101, -0.176209, 0.161174, 0.078502, 0.227174, -0.071186, -0.416552, -0.123689, 0.029541, -0.139094, -0.303436, 0.181532, -0.420825, 0.207969, -0.649702, -0.632246, -0.922756, -0.553966, -0.456005, -0.575313, -0.116762, 0.028056, -0.462826, -0.546873, -0.541345, 0.241699, -0.037006, 0.457691, 0.074282, -0.177898, 0.180116, -0.146268, 0.175106, 0.068884, -0.111705, 0.468838, 0.364176, -0.078292, -0.054915, 0.301103, 0.08553], [0.088726, -0.1922, 0.820167, -0.139657, 0.192402, 0.12731, 0.24915, 0.23478, -0.042131, 0.059229, -0.047547, -0.112433, 0.431716, 0.246146, 0.244857, -0.534239, 0.211312, 0.508024, -0.074718, -0.608069, 0.216629, -0.418199, -0.300257, 0.55377, 0.625806, 0.672728, 0.075238, 0.363687, 0.095735, 0.327617, -0.003029, 0.481824, -0.42786, 0.208165, -0.223478, -0.181289, -0.343575, 0.152889, -0.023607, -0.245263, 0.242749, -0.138618, 0.275694, 0.106559, 0.376106, 0.044494, 0.152977, 1.255427]]
SEQ_BH = [-0.232965, 0.177843, 0.218474, -0.098731, 0.137955, 0.288582, 0.077193, 0.20163, -0.144289, -0.281921, 0.375381, 0.131314, 0.22398, 0.184749, 0.018177, -0.194805, 0.113556, 0.646095, 0.459123, 0.538415, 0.333513, 0.755185, 0.857592, 0.473622, 1.047179, 0.376183, 0.658755, 0.573938, 0.802198, 0.399893, 0.534758, 0.550386, 0.201075, 0.32262, -0.152389, -0.1824, -0.314794, -0.210673, 0.278922, 0.193893, 0.297255, 0.233263, 0.000877, -0.321048, 0.185686, 0.081464, 0.093779, -0.034781]
SEQ_HEAD = [
    ([[0.04971, -0.190509, 0.295287, 0.107863, 0.14797, -0.171776, 0.191818, -0.014388, 0.022122, -0.09817, 0.114334, 0.036859, -0.065103, 0.017696, -0.173765, -0.077878, 0.051944, -0.043275, 0.122132, 0.087001, -0.036141, -0.280836, -0.218244, 0.266737, 0.074198, -0.168017, 0.225473, -0.009649, -0.052968, 0.058609, 0.024315, 0.032066, -0.183547, -0.137123, -0.122477, -0.032404, -0.017621, -0.293395, 0.0443, 0.052082, 0.163595, 0.09682, -0.120697, -0.137863, 0.066313, -0.001695, 0.024798, -0.05207, -0.021532, -0.064457, 0.076284, -0.077871, -0.148199, 0.059159, -0.028698, 0.101475, -0.109002, -0.036486, 0.007486, 0.126412, 0.467096, -0.00173, -0.067757, 0.076167], [0.017258, -0.180853, 0.013534, 0.022194, -0.174864, -0.084896, 0.064378, 0.00055, 0.075976, -0.133448, 0.112482, 0.024226, 0.01785, 0.023528, -0.087314, 0.069419, 0.049091, 0.046722, 0.036351, -0.146583, -0.09359, 0.069786, -0.008959, -0.150115, -0.013886, -0.020192, -0.106838, 0.042722, 0.023945, 0.009732, 0.01591, -0.009258, 0.050201, -0.187925, 0.113216, -0.057035, -0.133528, 0.099395, -0.029559, 0.083928, -0.004555, -0.001126, 0.048503, -0.150252, -0.05147, -0.116868, 0.076757, -0.118614, 0.167075, 0.080343, 0.134113, 0.161285, -0.110442, -0.1026, 0.206439, 0.018523, 0.070981, -0.154136, 0.015215, 0.157998, 0.005212, 0.053181, -0.030651, -0.070726], [0.133126, -0.181698, -0.026693, 0.016495, -0.139547, -0.164751, 0.142021, -0.061604, 0.111617, -0.095059, 0.174174, -0.087345, 0.037701, 0.070572, 0.022231, 0.05689, -0.150764, 0.12672, 0.009928, 0.044588, 0.042774, -0.064309, 0.062429, -0.174704, -0.112626, 0.047398, -0.240642, 0.080751, 0.141651, -0.178143, -0.121397, -0.009834, 0.032595, 0.023855, -0.039091, -0.114568, 0.033294, 0.089659, -0.070371, -0.134312, 0.15266, -0.120694, 0.0502, 0.098958, -0.107052, -0.076191, 0.055632, -0.04567, 0.003593, -0.109138, -0.065395, 0.071576, -0.112368, -0.111962, -0.023151, -0.175621, 0.022855, 0.071974, -0.113376, -0.021695, -0.096685, -0.010275, -0.067921, -0.183056], [0.04858, -0.184257, -0.281209, -0.101619, -0.156981, 0.1952, -0.145807, -0.020143, 0.022768, 0.029131, -0.087346, 0.14586, 0.188232, 0.003593, -0.015016, 0.092667, 0.144666, -0.186765, 0.163784, -0.190465, 0.20228, 0.002036, 0.156265, -0.087301, 0.117154, 0.102811, -0.041628, -0.189637, 0.107861, -0.105067, -0.229177, 0.031274, 0.153144, -0.048018, 0.074817, 0.144099, 0.154339, 0.108568, -0.117569, -0.040622, 0.018011, -0.177159, -0.092426, -0.078451, 0.074611, -0.047088, -0.106279, 0.035141, -0.127811, 0.034501, -0.064245, -0.03491, -0.134024, 0.091893, -0.032601, 0.183993, 0.137694, -0.051229, 0.136127, -0.084814, -0.262197, 0.056709, 0.121512, 0.085587], [0.06587, 0.189741, -0.192737, -0.088531, 0.171122, 0.052823, 0.118718, -0.03426, 0.125871, 0.130586, -0.167419, -0.052581, 0.313612, 0.153774, 0.090994, -0.011326, -0.177687, 0.058506, 0.231674, 0.113579, 0.178133, -0.019459, -0.015825, -0.044084, -0.141231, 0.007799, -0.150581, -0.115374, -0.279559, -0.219066, -0.326379, 0.007692, 0.136823, 0.02584, 0.041452, 0.113277, -0.006552, 0.297897, 0.060577, -0.207384, -0.088929, -0.069456, -0.085353, 0.037417, -0.034631, -0.046303, -0.018816, -0.096467, -0.083056, 0.058902, -0.220082, -0.031993, -0.012683, 0.035342, -0.159453, 0.031936, -0.023169, 0.095921, -0.207674, -0.158073, -0.383068, 0.056084, -0.071646, 0.065232], [-0.156531, -0.103761, 0.069622, 0.048193, -0.048651, -0.06344, -0.140263, 0.214004, 0.072046, -0.084117, 0.052749, -0.108417, -0.132611, -0.089328, 0.0317, 0.093185, -0.009256, -0.005455, -0.056019, 0.015273, 0.125195, 0.037786, 0.011434, -0.039534, -0.005241, -0.067871, -0.048087, -0.032855, -0.006271, 0.099424, 0.001479, 0.005634, 0.188679, -0.057637, -0.076685, 0.06948, 0.028196, 0.031836, 0.006888, -0.051279, 0.086027, -0.032269, 0.135555, -0.016874, 0.051757, -0.122433, -0.133318, 0.020969, 0.098884, 0.075034, 0.025001, 0.160153, 0.094927, 0.140827, 0.130541, -0.147173, -0.038313, 0.012943, -0.028717, -0.038278, -0.091105, -0.060513, -0.015418, -0.024033], [-0.059496, 0.173387, -0.189208, -0.202472, -0.235533, -0.165167, 0.153534, -0.160226, 0.010096, 0.073434, 0.162745, 0.403997, -0.025938, 0.115955, 0.154547, 0.181299, -0.165399, -0.141929, 0.116246, 0.162865, 0.031771, 0.250313, 0.268038, 0.003445, -0.114056, -0.296226, -0.194519, -0.025434, 0.070969, -0.067519, -0.037721, 0.203721, -0.14258, 0.150409, -0.060927, 0.292831, -0.084163, 0.214329, 0.270707, 0.169584, -0.062637, -0.208899, 0.024053, 0.144619, -0.119542, -0.213075, 0.14312, -0.071322, 0.079403, -0.159708, 0.149973, 0.274833, -0.184626, -0.221691, -0.173261, -0.165992, -0.044295, 0.037324, -0.303928, 0.243672, -0.040339, -0.249542, -0.053476, -0.207848], [-0.051574, 0.010215, -0.090505, 0.094075, -0.028634, -0.104226, -0.034881, 0.088905, 0.0047, 0.154191, -0.118885, -0.048911, -0.101484, 0.216737, -0.102382, 0.065407, -0.004616, 0.151612, -0.052147, -0.107121, -0.013042, -0.079337, -0.115373, -0.158867, -0.043107, 0.063255, 0.004794, 0.034224, 0.046311, -0.049346, 0.085918, 0.036616, 0.006732, 0.058218, 0.127597, -0.087041, 0.078177, 0.021005, 0.051692, 0.011597, 0.121667, -0.060085, 0.055417, -0.014383, -0.065809, -0.016846, 0.057216, 0.046605, -0.059822, 0.156482, -0.070254, 0.059976, -0.001509, -0.112872, 0.023081, -0.139634, -0.09414, -0.000362, -0.073181, 0.042435, -0.147982, -0.200748, -0.098381, 0.101106], [0.015968, 0.04706, 0.148829, -0.197939, -0.138716, -0.092995, -0.04725, -0.2066, 0.154169, 0.021348, 0.033266, 0.217441, 0.213072, -0.186944, -0.046246, 0.020206, 0.005234, -0.031457, 0.007053, -0.087679, 0.187232, 0.13137, 0.027608, -0.084465, -0.126299, -0.014983, -0.172195, 0.171363, -0.018043, -0.113856, -0.000211, -0.221106, -0.428727, 0.04316, 0.043427, 0.054537, -0.005225, 0.23513, 0.014669, 0.231736, 0.291663, -0.000959, 0.299314, -0.241499, -0.285, -0.091454, 0.132664, 0.006405, 0.137556, -0.166444, 0.163387, -0.040886, 0.008393, -0.093977, -0.087862, -0.185235, -0.02709, -0.067745, 0.024071, 0.226695, 0.070135, -0.182902, 0.130041, -0.117735], [-0.356528, -0.16518, 0.02947, 0.057694, 0.02169, 0.010632, 0.008786, 0.055169, -0.010272, 0.113738, -0.06629, 0.065323, 0.030823, -0.360682, 0.066096, -0.360306, 0.102335, -0.175988, -0.136953, -0.026004, 0.147483, -0.117306, -0.071902, -0.076256, 0.022106, 0.041641, -0.212761, -0.234698, -0.014977, 0.16737, 0.115083, 0.10014, -0.029155, -0.028209, -0.194368, -0.281214, 0.029774, -0.045998, 0.159282, 0.121911, -0.032302, -0.265256, 0.113578, -0.103385, 0.106151, -0.224633, -0.068319, 0.250224, -0.040328, 0.054837, 0.046927, 0.037342, -0.343592, 0.107149, 0.121305, 0.019764, -0.004481, 0.183066, -0.07045, -0.112586, -0.192344, 0.038258, 0.116572, -0.080489], [0.214331, -0.045245, 0.070196, -0.115076, 0.170053, -0.080549, 0.087095, 0.175703, -0.008946, 0.025442, -0.207637, -0.045963, -0.139746, 0.106684, -0.052205, -0.253158, 0.167693, -0.054762, 0.080586, 0.03608, -0.126123, -0.045172, 0.036932, 0.130713, -0.154626, -0.160161, 0.025884, 0.076022, 0.034305, 0.192693, 0.074302, 0.126567, -0.173967, 0.033686, -0.0441, -0.045448, 0.013794, 0.099522, 0.001649, 0.091103, 0.069782, 0.128154, -0.088502, 0.062567, 0.017664, 0.127385, -0.125821, -0.084504, 0.085428, -0.136845, 0.029062, 0.19854, 0.025015, 0.009437, 0.118545, 0.110515, -0.049811, -0.102223, -0.112865, 0.04535, 0.139444, -0.109807, 0.049179, 0.013037], [0.127759, 0.02778, -0.137099, 0.075433, 0.079735, -0.069812, -0.011711, 0.007018, 0.203815, -0.264528, 0.141249, 0.129634, 0.08733, -0.059905, 0.153125, -0.211349, -0.009773, 0.099067, 0.036829, -0.117205, 0.319319, 0.134736, 0.111816, 0.05395, 0.140539, -0.017578, -0.159191, -0.07924, -0.0049, -0.059486, -0.258035, -0.119322, -0.061539, -0.25849, -0.048631, -0.192539, -0.181233, -0.072351, 0.288125, 0.059159, 0.434217, 0.225731, 0.177317, -0.294125, -0.141548, 0.062109, 0.186769, -0.066522, -0.032674, -0.287185, 0.233175, -0.044848, -0.455098, -0.098689, -0.090028, -0.006177, -0.116935, 0.04873, -0.072993, 0.442468, -0.186206, -0.233945, 0.22268, -0.065109], [0.093835, 0.029579, 0.048883, -0.23705, -0.035572, 0.107903, 0.041576, -0.049115, -0.043601, -0.018129, 0.017288, 0.041685, 0.146451, 0.07672, -0.076947, -0.071908, 0.121539, 0.081566, -0.103611, -0.176297, 0.006732, 0.0911, -0.100455, -0.171753, 0.025961, 0.107649, -0.062233, 0.180146, -0.004007, -0.025927, 0.081616, -0.056903, -0.195998, -0.115615, -0.087672, -0.09414, -0.013553, 0.092454, 0.024727, 0.018452, -0.002336, 0.248504, -0.115975, 0.096098, 0.003058, 0.175536, -0.067004, 0.254859, 0.160729, -0.162474, -0.018548, -0.16234, 0.068083, -0.052079, 0.079733, 0.083479, -0.096614, 0.013125, 0.082903, 0.186789, 0.070116, 0.027323, -0.012341, -0.040003], [-0.251288, -0.088134, -0.102915, -0.042166, -0.257805, 0.016963, 0.002813, 0.135868, 0.047567, -0.035778, -0.06868, -0.054998, -0.071937, 0.09914, -0.049186, 0.127122, 0.044692, -0.077526, 0.102819, -0.082967, -0.019138, -0.214552, -0.097543, -0.015212, -0.036667, 0.139457, -0.025227, -0.107395, 0.09435, -0.273042, -0.00064, 0.049643, -0.083991, 0.135667, -0.136874, 0.029109, -0.082322, -0.122924, -0.081115, -0.223605, 0.044012, 0.148495, -0.042138, 0.395117, -0.165623, 0.00993, 0.080733, 0.050127, 0.020258, -0.050926, -0.216414, -0.102879, 0.008116, -0.022077, 0.053938, -0.083727, 0.056188, 0.16746, -0.094264, 0.200444, 0.113987, -0.039334, -0.096763, -0.179033], [0.082514, 0.095476, 0.189083, -0.004456, 0.040317, 0.017628, -0.002226, 0.128011, -0.097164, -0.216631, -0.095795, -0.206218, -0.20211, 0.006291, 0.035507, -0.160208, 0.110623, -0.401378, 0.023168, -0.149877, 0.156555, -0.077668, -0.073707, 0.000379, 0.190041, 0.003245, -0.034287, -0.064083, -0.344167, 0.075836, -0.0032, -0.151782, -0.007185, -0.14873, -0.019562, -0.03347, 0.089309, -0.037891, 0.310025, -0.282846, 0.045731, 0.054104, -0.224328, 0.086731, 0.090364, -0.068137, -0.07944, -0.112099, 0.165453, 0.05339, -0.158025, 0.045039, -0.009755, -0.439054, -0.017989, 0.19583, 0.167295, -0.205683, 0.248848, -0.009872, 0.025811, 0.152853, 0.087134, -0.126787], [0.047495, -0.204639, 0.107249, -0.146973, 0.158951, -0.369345, 0.310541, -0.133803, 0.002494, -0.266226, -0.265325, -0.124177, 0.065675, -0.04798, 0.035734, 0.023163, -0.141765, -0.043718, 0.149309, -0.23973, 0.148594, -0.081617, -0.145943, 0.039657, -0.091048, -0.182584, 0.054378, -0.053943, -0.375576, 0.047054, -0.105949, -0.199426, -0.015349, -0.058756, 0.042206, 0.042676, -0.201584, -0.242907, -0.023429, -0.020362, -0.069069, -0.013548, 0.009251, -0.08144, -0.190225, -0.076351, -0.086839, -0.091136, -0.218367, -0.0494, -0.123843, 0.189347, 0.012311, 0.212261, -0.034318, 0.010761, 0.104423, 0.03868, 0.03453, -0.006097, 0.242262, 0.173729, 0.136692, 0.280467], [0.041767, 0.126465, 0.020984, -0.261062, 0.024581, 0.018131, 0.23272, -0.195364, -0.128658, -0.110572, -0.083744, 0.052601, 0.18037, 0.033702, 0.019397, 0.078992, -0.114387, 0.4516, -0.026516, 0.158729, 0.381114, -0.222155, -0.249603, 0.252356, -0.137172, 0.034974, 0.22438, -0.045361, 0.267961, -0.11367, 0.186701, -0.259319, -0.078698, -0.089018, -0.097752, 0.136285, -0.10653, -0.280183, -0.247002, 0.000745, 0.060185, 0.026475, -0.133761, -0.007672, -0.021772, -0.029113, -0.033421, 0.310235, -0.292617, -0.13268, 0.110301, 0.000378, -0.040279, 0.296152, -0.008248, 0.0884, 0.206052, 0.074087, -0.462151, 0.071362, -0.022829, -0.02288, 0.278862, 0.234538], [0.211886, 0.35262, 0.037848, -0.135449, 0.028232, 0.024415, 0.359744, -0.027195, 0.217586, -0.030777, -0.238448, -0.057912, -0.116803, 0.28951, -0.211435, 0.089326, 0.217015, 0.169219, 0.020636, -0.143014, -0.197577, 0.000517, -0.125624, 0.062107, 0.056131, -0.369573, 0.497695, 0.102538, -0.084073, -0.041293, 0.030758, 0.141916, -0.007236, -0.13428, -0.17859, 0.083912, 0.266619, -0.064105, 0.052455, -0.146654, 0.050376, 0.040598, -0.136924, -0.046213, 0.068882, -0.009363, -0.153739, -0.03301, -0.136817, 0.048244, -0.025988, -0.161236, -0.170246, -0.027124, -0.046104, 0.113967, 0.044868, -0.073322, 0.146064, 0.12577, -0.431214, 0.135806, -0.015838, -0.02372], [0.03435, 0.022816, -0.041784, 0.226101, -0.067769, 0.03963, 0.101621, -0.019056, -0.038261, 0.145374, 0.018334, -0.087109, -0.086527, 0.067616, -0.011004, -0.009991, 0.007426, -0.011278, -0.088122, 0.032547, 0.026605, 0.024729, -0.074445, 0.084603, -0.132728, -0.040286, -0.091139, -0.093271, 0.156679, -0.115366, 0.0428, -0.166233, -0.047733, 0.071316, 0.062671, 0.060769, 0.013337, -0.115319, -0.095235, 0.065567, 0.053645, -0.153726, 0.041221, -0.053397, 0.003732, 0.150204, 0.001787, 0.051826, -0.000362, -0.074576, -0.037475, 0.000524, 0.073866, 0.157294, -0.077337, -0.154254, 0.003112, -0.135124, -0.005469, -0.231102, 0.007199, 0.061012, -0.136912, 0.098599], [0.074362, -0.011834, 0.015856, -0.206593, -0.121851, -0.042866, 0.077131, 0.100198, 0.062682, 0.137955, -0.055511, 0.256553, 0.200381, 0.010369, -0.028443, -0.164511, -0.134673, 0.027418, 0.125064, -0.096823, 0.151823, -0.025861, 0.296703, 0.15413, 0.072756, -0.092893, 0.030863, 0.143591, 0.234407, -0.2808, -0.016606, -0.021021, 0.160527, 0.074332, -0.015601, 0.120579, -0.281114, -0.007506, -0.104896, 0.303615, -0.092177, -0.170967, 0.436765, 0.151323, -0.33599, 0.011758, 0.25887, 0.088557, 0.227543, -0.092218, 0.232502, 0.184761, -0.091247, 0.277784, -0.317215, -0.418326, -0.451917, 0.201329, 0.010735, -0.071569, 0.023962, -0.336693, 0.075415, 0.059883], [0.073041, 0.045402, -0.011842, -0.228379, 0.161173, -0.076054, 0.065188, 0.034564, 0.064195, 0.165739, -0.296804, -0.025571, -0.070158, -0.169104, 0.055611, -0.187259, -0.164998, 0.023728, -0.069842, -0.132863, -0.010194, -0.007443, 0.011436, -0.175036, 0.07171, 0.217977, 0.069701, 0.169585, 0.26432, -0.076935, 0.048332, -0.222658, -0.023509, -0.003553, -0.185438, -0.094113, 0.062434, 0.124559, 0.340984, 0.152718, -0.014575, 0.169055, 0.241071, 0.15687, 0.037485, 0.016626, -0.013336, 0.010573, 0.121667, 0.066985, 0.018523, -0.29857, 0.052484, -0.181912, -0.013225, -0.41987, -0.259065, -0.2739, 0.046342, 0.022307, 0.153594, -0.118701, 0.053169, 0.262768], [-0.048598, -0.046301, -0.113277, 0.380904, -0.064232, 0.131794, -0.003708, 0.381157, 0.266619, -0.061565, 0.013787, 0.46283, -0.241786, -0.005136, -0.120451, -0.142143, -0.228168, -0.236932, 0.041434, -0.024588, -0.040064, -0.043505, 0.410819, 0.045177, -0.198116, 0.072901, 0.116768, -0.058665, 0.112095, 0.079494, 0.100189, -0.099537, 0.022504, -0.113184, -0.193806, 0.0273, -0.218858, -0.069053, -0.113184, 0.047984, 0.035935, -0.135623, -0.117639, 0.05473, 0.081193, 0.311007, -0.200253, 0.000899, -0.376819, -0.048565, 0.019736, -0.043713, 0.101448, -0.23871, -0.163167, -0.17575, -0.062327, 0.202583, -0.285112, 0.133424, 0.074749, 0.228742, 0.124081, 0.494722], [0.099304, -0.011913, -0.292982, 0.191841, 0.227385, 0.249626, 0.178659, -0.101554, 0.123094, 0.076388, -0.018376, 0.076681, -0.087939, -0.072035, 0.108871, -0.031399, 0.187324, -0.055034, 0.214367, 0.109253, 0.089711, 0.20533, 0.156484, 0.134767, -0.116578, -0.00139, -0.144892, 0.093913, -0.061484, -0.170865, 0.055204, 0.257783, -0.328013, 0.06009, 0.14033, -0.083533, 0.06683, -0.032745, 0.228921, -0.030359, 0.134661, 0.21398, -0.121974, -0.149244, -0.045113, 0.272301, -0.038376, -0.198194, -0.2141, -0.23265, 0.149737, -0.085504, 0.084819, 0.092091, -0.112908, -0.340439, -0.21943, -0.002371, 0.057223, 0.081784, 0.277641, 0.04409, 0.240571, -0.010307], [0.096225, 0.184485, 0.468153, 0.129508, -0.095332, -0.133454, 0.178855, 0.093999, 0.183712, 0.225218, 0.296349, 0.194601, -0.017792, -0.413526, 0.182404, -0.000138, -0.16777, -0.155472, -0.077547, -0.018358, 0.083104, 0.099563, 0.041309, 0.173628, 0.068902, 0.152649, 0.028617, 0.098598, -0.139606, 0.080911, 0.106848, -0.01126, -0.285493, -0.064849, 0.028886, -0.245923, 0.163357, 0.249696, 0.110161, -0.105106, -0.044132, -0.021668, 0.396834, -0.42079, 0.096758, 0.088944, 0.172095, -0.094698, -0.212385, 0.069746, 0.181646, 0.348818, -0.419564, 0.255371, 0.234663, 0.118321, -0.068668, 0.188186, 0.065178, -0.176976, -0.312663, 0.229078, -0.0098, 0.065578], [0.013993, -0.168615, 0.002395, 0.077001, 0.126668, 0.091575, -0.05142, 0.017388, 0.199662, 0.086761, 0.083362, -0.05138, -0.082146, -0.061484, 0.052549, -0.061058, 0.150143, -0.128343, -0.079474, -0.373526, 0.13302, 0.004653, 0.093609, -0.349315, 0.049861, 0.105506, 0.298461, -0.067167, 0.054098, 0.085807, 0.10182, 0.218206, -0.495031, -0.053294, -0.451103, -0.018388, -0.046245, -0.368626, -0.078084, 0.085725, 0.151773, 0.225569, 0.036279, -0.199406, -0.018271, 0.356401, 0.0642, -0.28054, 0.055037, -0.089859, 0.001363, -0.397668, -0.065792, -0.089517, -0.187154, -0.178811, -0.029083, 0.12332, -0.148351, 0.013529, -0.097579, -0.046753, 0.285548, -0.264805], [-0.190633, 0.019955, -0.106679, 0.002875, 0.0239, -0.246891, 0.018411, -0.02136, -0.026392, -0.019226, 0.091797, -0.291491, -0.066234, 0.166103, -0.009066, 0.236684, 0.055299, 0.048503, -0.104513, 0.135155, 0.199102, 0.123642, -0.174958, -0.074206, -0.052988, -0.148241, 0.050716, 0.04512, 0.129423, -0.009665, -0.177557, 0.045964, 0.06794, -0.018419, 0.23911, 0.097886, -0.309959, -0.061252, -0.019186, -0.186141, -0.033806, 0.065911, -0.087059, 0.111479, 0.036994, -0.297035, -0.167511, 0.207503, -0.243543, 0.069304, 0.171621, -0.159182, -0.107188, 0.140181, -0.008426, -0.138631, 0.035024, 0.089758, 0.00738, 0.090529, 0.051952, -0.169002, 0.025713, -0.2854], [-0.152348, 0.085279, -0.039783, 0.34647, -0.106673, 0.082139, 0.222756, 0.405631, 0.159247, 0.124051, -0.022303, 0.03217, -0.09617, -0.036044, 0.135806, -0.099025, -0.201477, 0.119425, 0.290105, 0.302078, 0.065076, -0.047432, 0.005587, 0.199388, -0.155153, 0.163469, -0.094118, -0.182406, 0.055983, -0.097389, 0.152766, -0.037692, -0.021174, 0.069269, 0.20802, -0.176983, 0.127091, 0.137929, 0.512832, 0.04399, 0.014161, 0.192744, -0.128108, -0.088107, -0.074302, -0.06527, -0.052114, -0.030235, 0.122432, 0.008348, -0.14107, -0.036976, 0.103552, 0.405815, -0.010196, 0.176062, -0.20894, 0.214953, -0.088823, 0.062064, -0.055756, 0.117804, 0.149353, 0.297086], [-0.057682, -0.167199, 0.00118, 0.338103, 0.133249, -0.017414, 0.386839, -0.06723, -0.001198, 0.013939, 0.076299, 0.077286, 0.117167, 0.130368, -0.003531, -0.065709, -0.082192, 0.0254, 0.225726, 0.119938, 0.213849, 0.29945, 0.092905, 0.148354, -0.069354, 0.151098, 0.179995, -0.134026, -0.146269, -0.314495, 0.077746, 0.066748, -0.029717, 0.072168, 0.069439, -0.129836, 0.083375, -0.000114, 0.116223, -0.129739, 0.269179, 0.258638, -0.120194, -0.240327, -0.047323, -0.033067, -0.044116, 0.177805, -0.115356, 0.086345, 0.059462, 0.140929, 0.23692, 0.083247, -0.14968, 0.216438, 0.07818, 0.000257, -0.17508, 0.365707, 0.183915, 0.274808, 0.179944, 0.256398], [0.133974, 0.09167, -0.126116, 0.044918, 0.170484, 0.012868, -0.240948, 0.238053, -0.076319, 0.059047, -0.053108, 0.415671, -0.06744, -0.04987, -0.176765, -0.139166, 0.014345, -0.225854, -0.167074, 0.143465, -0.084142, -0.073343, 0.304779, -0.310168, -0.099613, 0.267308, -0.173817, -0.172999, 0.010919, 0.078312, -0.074089, -0.053926, 0.298958, -0.02389, -0.039208, -0.373708, 0.190249, 0.068618, 0.229232, 0.283546, -0.053695, -0.285257, 0.021939, -0.133958, 0.004686, -0.12463, 0.069, -0.009139, 0.315021, -0.025011, 0.105147, -0.164732, -0.151286, -0.328616, -0.081678, 0.224262, 0.174857, -0.126597, 0.099182, 0.126061, -0.127789, -0.199822, 0.189116, 0.065883], [-0.052976, 0.107029, -0.038961, -0.281586, 0.09702, -0.325967, -0.241199, -0.140714, -0.238997, 0.017687, 0.033638, -0.136467, -0.145693, 0.117295, -0.178854, 0.107537, 0.058462, 0.026825, -0.219068, 0.046663, -0.215356, -0.075121, -0.306236, 0.010064, -0.112328, -0.233958, -0.18088, -0.045977, 0.081386, 0.204754, -0.227539, -0.018133, 0.133196, -0.18642, -0.038327, 0.150532, -0.130785, -0.268723, -0.034775, -0.271737, -0.224603, -0.031758, -0.242664, 0.107714, -0.087524, -0.1481, -0.08803, -0.07821, -0.228766, -0.110193, -0.156315, -0.08813, -0.251006, -0.336724, -0.183472, -0.162687, -0.174607, -0.051671, 0.133511, -0.103216, 0.141047, -0.352514, 0.022846, -0.015168], [0.065864, -0.376269, -0.150442, -0.051602, 0.203957, -0.059921, -0.06801, -0.08317, 0.163589, -0.340069, 0.107056, -0.379154, 0.064118, -0.063662, -0.288572, -0.30095, 0.124314, 0.018116, 0.012284, -0.291269, 0.121444, 0.055762, -0.187682, -0.2307, -0.325329, -0.547203, -0.021115, 0.102069, 0.092901, -0.209564, -0.073191, 0.238707, -0.360392, -0.344022, -0.146938, -0.102049, -0.619557, -0.230165, -0.082608, -0.258969, 0.133279, 0.482528, 0.117323, 0.166985, -0.17016, 0.221891, -0.191952, 0.126016, -0.043147, -0.14707, -0.150167, -0.324447, -0.065935, 0.014372, -0.431909, -0.271468, -0.13615, 0.081212, -0.008239, 0.217117, -0.06107, -0.278286, -0.063397, -0.151302], [-0.02369, 0.033933, 0.220877, 0.35539, 0.103073, 0.229727, 0.081335, -0.162622, 0.070612, 0.640818, 0.123571, -0.108873, 0.062524, 0.132608, 0.419908, 0.22014, 0.513469, 0.113941, 0.121446, -0.629936, 0.155919, 0.332264, -0.11675, 0.214191, 0.425373, 0.519706, 0.474162, -0.128785, 0.043819, 0.130271, 0.678388, 0.371858, -0.271941, 0.489328, -0.268789, 0.166817, 0.343559, 0.142852, -0.078072, -0.215154, -0.022344, -0.303881, -0.067279, -0.312361, 0.118916, 0.168625, -0.085324, 0.221292, 0.163001, 0.565657, 0.044514, 0.120656, 0.275022, 0.303555, 0.309903, -0.119913, 0.262157, -0.158479, -0.162854, 0.060832, 0.303289, -0.261893, 0.203094, -0.075968], [0.083305, 0.051764, -0.060863, -0.294965, -0.199195, 0.145804, -0.283578, 0.303653, -0.374212, 0.172096, -0.105299, -0.280165, -0.11441, 0.292038, -0.033978, -0.184244, 0.311537, -0.121565, -0.354395, -0.032174, -0.094393, -0.201498, 0.001465, -0.067983, 0.055796, 0.12877, -0.065373, 0.230854, 0.069507, 0.14622, -0.178922, 0.155333, -0.032605, -0.054437, -0.392658, 0.01116, 0.428006, 0.025764, 0.037443, -0.073819, -0.057575, 0.032742, -0.058389, -0.441276, 0.303729, 0.097793, -0.245676, -0.228795, 0.172891, 0.139745, -0.212781, 0.015056, -0.055433, 0.050436, 0.08077, -0.22641, 0.025451, -0.189403, -0.102324, -0.032482, -0.013434, -0.392122, 0.092764, -0.178666], [0.032603, -0.221037, -0.099264, 0.116301, -0.088796, -0.03416, -0.059727, 0.011661, -0.154221, 0.12852, 0.072281, 0.040151, 0.194262, 0.229313, -0.002741, -0.169248, -0.167355, -0.120292, 0.145606, -0.112687, 0.021287, 0.23023, -0.047477, -0.167387, 0.071935, 0.21804, 0.04149, 0.141787, 0.225305, 0.023772, 0.362427, 0.120197, 0.067155, 0.176797, -0.014024, 0.164272, -0.193944, 0.088508, 0.03869, 0.170442, -0.029477, -0.309884, -0.102356, 0.068441, 0.151556, -0.050503, -0.02436, 0.307127, -0.009351, -0.023263, -0.132332, -0.111562, 0.210922, -0.076878, 0.095641, -0.110257, 0.066681, 0.20801, -0.129946, 0.119264, 0.19053, -0.036463, -0.102892, 0.245291], [0.209017, 0.074857, 0.011909, 0.068764, 0.109888, 0.123427, 0.216772, -0.449287, 0.396011, -0.1237, 0.056729, -0.24815, 0.119836, -0.274257, -0.07362, -0.018015, -0.095947, 0.2717, 0.109938, 0.061194, 0.147323, -0.197982, 0.256586, 0.296272, 0.069852, -0.256104, 0.071122, 0.014772, -0.162818, -0.08548, 0.103092, 0.050663, -0.16189, 0.17295, 0.37854, -0.331696, -0.033877, 0.252802, 0.142806, 0.081677, 0.243076, 0.34319, 0.344381, 0.136918, -0.034723, 0.370205, 0.121645, 0.084197, -0.078501, -0.090852, 0.217625, 0.139118, 0.090326, -0.003955, -0.161334, 0.152405, 0.033126, -0.088487, -0.043232, 0.175014, 0.078696, -0.202291, 0.165088, -0.245716], [-0.193589, -0.043912, -0.258219, 0.128404, 0.236051, -0.123706, 0.091567, -0.024496, 0.009189, -0.428117, -0.005936, -0.174507, -0.271548, 0.255936, 0.033021, -0.276603, 0.052323, 0.035331, 0.131532, -0.062535, -0.022061, 0.041301, 0.319757, 0.155454, -0.024108, -0.007485, 0.022483, 0.224373, 0.116377, -0.0076, 0.023339, -0.105688, 0.241615, -0.001817, -0.118716, -0.046564, -0.047146, 0.16973, 0.162753, -0.168913, 0.013757, 0.154073, -0.093593, 0.251684, 0.232455, -0.102008, -0.023817, 0.354778, 0.258022, -0.029178, -0.061354, -0.0399, 0.141593, 0.111909, -0.133018, -0.014926, -0.251615, 0.05188, 0.208386, 0.224392, 0.22645, -0.343028, 0.045465, -0.090893], [0.237045, -0.129779, -0.111544, 0.592223, 0.234217, 0.065226, 0.140744, 0.245097, -0.196835, 0.12488, -0.034439, 0.149613, -0.146403, -0.03179, 0.122018, 0.182831, 0.003379, -0.25185, -0.02341, -0.157653, 0.00208, 0.205576, 0.063651, 0.016287, 0.253571, 0.233832, 0.143755, 0.139028, 0.043177, -0.02236, 0.190863, 0.172132, 0.314495, 0.192147, -0.070178, 0.160164, -0.165471, 0.155226, 0.052076, 0.133197, 0.092011, 0.122055, 0.043649, -0.125535, 0.047873, 0.148528, -0.032197, -0.1697, -0.109122, 0.072319, -0.003569, 0.09106, 0.190083, -0.204306, 0.085305, -0.174508, 0.095192, -0.106372, 0.174658, -0.024766, 0.080715, -0.016027, 0.03019, 0.141922], [0.05524, -0.101432, -0.021667, -0.122601, -0.318877, -0.039361, -0.137118, 0.240619, 0.015091, -0.097486, -0.366563, -0.145382, -0.216856, 0.082755, 0.131812, 0.046991, 0.044674, -0.133488, -0.444416, -0.197891, -0.322061, -0.038241, -0.661258, -0.371395, 0.031196, -0.125518, -0.112361, 0.07928, 0.228013, -0.090222, 0.012408, -0.187608, 0.273566, -0.119705, -0.348644, 0.065676, -0.048155, -0.435333, -0.281409, 0.125692, -0.298266, 0.19313, -0.341811, -0.133194, -0.032283, -0.281253, -0.487701, -0.008404, 0.215951, -0.067477, -0.208598, -0.404525, -0.130837, -0.266192, -0.104171, 0.006658, 0.032829, -0.341415, -0.197788, 0.069876, 0.001779, -0.006147, -0.17436, 0.000351]], [-0.152694, 0.044947, 0.2312, -0.247483, -0.010912, -0.191683, -0.060154, -0.114133, 0.033286, 0.180157, -0.065063, -0.18581, 0.136502, -0.103647, -0.035758, -0.10748, 0.021269, 0.251483, -0.306061, -0.50336, -0.167175, -0.083529, -0.051774, -0.13931, 0.121284, -0.093233, -0.2574, -0.091145, -0.052236, 0.051431, -0.092718, -0.005921, -0.433957, 0.072137, -0.048796, -0.19775, 0.044904, 0.006238, -0.323289, -0.123512, -0.326922, -0.00712, 0.154305, -0.08317, -0.072065, 0.119249, 0.071879, -0.220135, -0.110279, 0.106771, 0.110207, -0.081294, -0.336168, -0.162594, 0.14498, -0.156887, 0.147984, -0.072604, -0.111311, -0.29968, -0.275416, 0.184755, -0.212353, -0.168719]),
    ([[0.014999, -0.139066, -0.141143, 0.047187, -0.156661, 0.08814, 0.067517, 0.070875, -0.138591, -0.085334, -0.082611, -0.035534, 0.013668, -0.027359, -0.035458, 0.041817, 0.047124, 0.026227, 0.108789, -0.024052, -0.209314, -0.143578, 0.011734, -0.130804, -0.011684, 0.100408, -0.279287, 0.062267, -0.226622, 0.024132, -0.09629, 0.072224], [-0.009002, -0.065068, 0.080521, 0.03365, 0.11134, -0.031677, 0.097463, 0.053232, 0.254361, 0.085262, 0.034232, 0.006937, 0.051904, 0.177287, -0.112551, 0.194504, 0.259289, 0.121657, 0.086343, 0.213036, 0.003675, 0.015598, 0.320698, -0.180889, 0.074814, 0.154569, -0.094099, 0.166993, -0.061986, -0.086769, 0.111149, 0.120098], [-0.281618, 0.085545, 0.110973, -0.000519, 0.054514, 0.037841, -0.479111, 0.095274, -0.009556, 0.129392, -0.101762, -0.063602, 0.074967, 0.136313, -0.022388, 0.211822, 0.062807, 0.082525, 0.223399, 0.145451, 0.031179, -0.074051, 0.118409, -0.461456, -0.089898, 0.124485, 0.138686, 0.079445, 0.094097, -0.256049, -0.010247, 0.20306], [-0.077798, 0.054874, 0.180019, 0.069808, 0.285002, -0.139926, 0.069021, -0.046704, -0.065548, 0.180461, 0.26344, 0.265578, 0.119251, -0.10264, 0.138705, -0.135716, 0.1207, -0.088718, 0.07611, 0.202345, 0.10398, 0.153526, 0.184462, 0.487883, 0.017336, -0.124967, 0.094993, -0.143667, 0.149676, 0.073038, 0.125571, 0.146887], [-0.030071, -0.011847, -0.061698, -0.057226, 0.034336, 0.046663, 0.153568, -0.104162, -0.017528, -0.16832, -0.281292, -0.01012, -0.049252, -0.096423, 0.152625, -0.078506, 0.066709, -0.095632, -0.036928, 0.145611, -0.165577, -0.155403, -0.03506, 0.018283, -0.081598, -0.07329, -0.024945, -0.110538, -0.092581, 0.035407, 0.071889, 0.118485], [-0.133699, 0.018335, 0.124923, 0.041526, 0.071816, -0.077005, -0.24511, -0.020911, -0.057616, -0.024063, 0.16474, -0.240864, 0.099345, 0.113053, 0.05302, -0.169342, -0.054666, 0.065406, 0.019458, 0.114047, 0.12388, -0.051625, 0.045811, -0.172937, 0.085031, -0.000462, 0.094267, 0.164462, -0.049979, -0.054335, -0.036568, -0.013097], [0.078036, 0.048993, -0.023707, -0.067315, 0.122896, 0.041105, -0.01882, 0.050447, 0.135636, 0.143699, 0.174866, 0.198606, 0.088372, 0.14651, -0.272919, 0.098236, 0.124281, 0.138238, 0.132705, 0.101723, 0.067817, 0.04125, -0.01813, -0.116263, 0.158156, 0.080808, -0.050671, -0.167599, 0.089133, 0.136447, -0.050594, 0.152928], [-0.007736, -0.076135, -0.010312, 0.063365, -0.059239, 0.035014, -0.236393, 0.075085, 0.030011, 0.079746, 0.219468, -0.181439, 0.047771, -0.008724, 0.13803, -0.104077, -0.140798, -0.103772, -0.065872, -0.045792, -0.092128, -0.030454, 0.068871, -0.078716, -0.31289, -0.158102, 0.007636, -0.067124, 0.058636, 0.023858, -0.080531, -0.10119], [0.064151, -0.157694, 0.081302, -0.068793, -0.018202, -0.069592, -0.07119, 0.072671, 0.042504, -0.079898, 0.054082, -0.040623, -0.069332, -0.076764, 0.043193, 0.032621, -0.048129, 0.183544, 0.048122, -0.017157, -0.135615, 0.176455, 0.026401, 0.163558, -0.000531, 0.063899, 0.015681, -0.071729, -0.126234, 0.152557, -0.140478, -0.065761], [0.123482, 0.044853, 0.117084, -0.04263, 0.131454, 0.084908, 0.100866, 0.042213, -0.17064, -0.099098, 0.119775, -0.228963, 0.168568, 0.020934, 0.015689, 0.176989, 0.034739, -0.005022, -0.07927, 0.038147, 0.131836, 0.151817, -0.158157, -0.287574, 0.046168, 0.180441, 0.136985, -0.188461, 0.049158, 0.1332, -0.07397, 0.092543], [0.158726, 0.003233, -0.029805, 0.04967, 0.18616, 0.110328, 0.042966, 0.057043, 0.160157, 0.017302, -0.065736, 0.008616, 0.166512, 0.055629, -0.177677, 0.023323, 0.120121, 0.110928, -0.008339, 0.149959, 0.151884, 0.032099, -0.006471, 0.243832, -0.09836, 0.104912, -0.027026, -0.15159, 0.18612, 0.112866, 0.100943, 0.137044], [0.315519, 0.069906, 0.117222, -0.047871, -0.05032, 0.074767, -0.028342, -0.025464, 0.1816, -0.191322, -0.280134, -0.009053, -0.035409, 0.096798, 0.20641, -0.074879, -0.013596, -0.12114, 0.02522, -0.146872, -0.039049, -0.020029, 0.137296, 0.25807, 0.191056, -0.112849, 0.028514, -0.179999, -0.251664, 0.14628, -0.160075, -0.074816], [0.085009, -0.075614, -0.159973, 0.074614, -0.285225, 0.009625, 0.197566, 0.049781, 0.154987, 0.159443, 0.068149, 0.071658, -0.028583, -0.108238, -0.241176, 0.004243, 0.052856, 0.049426, 0.105596, -0.210954, 0.052995, -0.162343, -0.110202, -0.096865, -0.0777, 0.057675, -0.149909, -0.083641, -0.079234, 0.033187, -0.100246, -0.060849], [-0.169602, 0.059632, 0.027118, -0.116882, -0.241304, -0.023276, 0.068906, -0.109211, -0.0864, 0.009749, 0.236623, -0.753089, -0.160526, -0.324304, -0.272429, -0.172393, -0.176309, -0.112004, -0.044361, 0.037904, 0.035214, -0.007929, -0.015813, 0.121434, -0.054244, -0.033383, -0.038894, -0.110792, -0.039747, 0.108707, -0.032082, 0.027393], [0.128628, 0.110645, -0.123126, 0.12095, -0.029877, -0.105115, 0.060096, -0.179411, 0.145832, 0.091964, -0.113413, -0.141573, 0.119863, 0.007981, -0.122167, 0.081915, -0.010048, -0.019471, -0.025363, 0.104885, 0.061866, 0.046358, 0.007154, -0.147725, 0.11574, -0.041096, 0.025276, 0.088779, 0.127957, 0.040344, 0.140983, -0.009851], [0.003915, 0.145781, 0.062015, -0.021263, 0.017136, 0.059219, -0.465766, 0.086804, 0.161212, -0.063881, -0.004634, -0.150119, 0.18809, 0.002219, -0.108145, -0.018678, -0.061326, -0.187797, -0.353719, 0.003394, 0.178983, -0.017527, -0.250373, 0.226686, -0.123119, 0.059847, 0.168829, -0.081206, 0.081066, 0.071077, 0.187066, 0.039017], [-0.188037, 0.028443, -0.008736, -0.075276, 0.152438, -0.09116, -0.04436, -0.093041, -0.247035, -0.00256, 0.049377, -0.015411, -0.03016, -0.086753, 0.138487, -0.175887, -0.076247, 0.023105, 0.058275, -0.021866, 0.010507, 0.087528, -0.03867, 0.133146, -0.028211, -0.11876, -0.001756, -0.00815, -0.079176, -0.234932, 0.171723, -0.03303], [-0.065106, -0.002673, -0.051767, 0.004275, -0.175474, 0.130308, -0.061933, -0.084423, 0.068239, -0.244422, -0.055052, -0.086425, 0.033844, -0.214379, -0.400363, 0.11105, -0.057772, 0.043103, 0.070679, 0.009197, 0.088289, -0.024709, 0.114721, 0.044429, 0.062557, -0.069463, -0.212258, 0.077442, -0.024605, 0.025062, -0.028921, -0.083207], [0.103813, -0.10161, -0.114267, -0.101777, -0.034335, 0.08734, -0.111323, 0.060308, 0.129831, 0.343089, 0.212774, -0.161403, -0.00971, 0.057359, -0.346099, -0.03832, 0.043178, 0.067048, 0.129181, -0.115278, -0.110833, -0.008575, 0.141174, 0.124845, 0.153018, -0.028114, -0.035912, -0.038879, -0.010904, 0.040071, 0.128021, 0.128333], [0.117026, -0.280019, -0.246884, -0.216187, -0.167716, 0.027391, 0.246119, 0.300947, 0.214615, -0.10424, -0.122047, -0.003484, -0.25905, 0.239108, -0.150581, 0.04107, 0.003218, -0.085696, 0.075233, -0.105274, -0.225203, -0.073873, -0.16069, 0.06988, 0.302461, -0.146815, -0.233613, 0.214135, -0.097466, 0.197187, -0.030094, -0.058179], [0.197026, -0.055859, -0.171849, -0.095778, -0.255235, 0.07636, 0.100925, 0.030705, -0.066234, 0.165206, -0.009794, -0.017315, -0.059048, -0.148546, -0.041473, 0.168895, 0.113452, 0.143146, 0.021127, -0.293293, 0.002313, -0.284504, 0.112827, 0.138157, 0.029131, -0.022719, -0.193802, -0.085517, -0.036117, 0.131847, -0.112499, 0.104753], [-0.021189, -0.066441, -0.052692, 0.007496, -0.084908, 0.155943, -0.183436, -0.157919, -0.071714, 0.032945, -0.056919, 0.000313, 0.054516, -0.182015, -0.217364, -0.191031, -0.027759, 0.084773, -0.102112, 0.129155, 0.005686, -0.052051, -0.049986, 0.196522, -0.021191, 0.025301, 0.091827, -0.168203, -0.003918, -0.120689, 0.075536, -0.108009], [0.153333, -0.20015, -0.030904, 0.004847, -0.114639, 0.291951, 0.157145, 0.161517, 0.202775, -0.207774, 0.059774, -0.266424, -0.04466, 0.212403, -0.048628, -0.04565, -0.009304, 0.181463, 0.151923, 0.008414, -0.036962, 0.028254, 0.190605, 0.148818, 0.275042, 0.104736, -0.033218, -0.100046, 0.023136, 0.2452, 0.022493, -0.413899], [0.080778, 0.050865, 0.111561, -0.100976, 0.209431, -0.149456, -0.166372, 0.17176, 0.097638, -0.010983, 0.062514, -0.002912, 0.101448, 0.155974, -0.153455, -0.034923, -0.0162, 0.15938, 0.165142, 0.035675, -0.003215, 0.186801, 0.095695, -0.25584, 0.068938, 0.021269, 0.090627, 0.107629, 0.126893, 0.288823, 0.105761, -0.024677], [0.06291, 0.158873, -0.060842, 0.133313, -0.01705, 0.043463, -0.068455, -0.068938, -0.101187, -0.043377, -0.02506, -0.06843, 0.089804, -0.135461, -0.076032, -0.028372, -0.078628, 0.008501, 0.009451, -0.019111, 0.058101, -0.020501, -0.148962, -0.055785, 0.111917, 0.159089, 0.149289, -0.030953, 0.062472, -0.018969, 0.111323, 0.123023], [0.250623, 0.101177, 0.139043, 0.064698, 0.134302, -0.064553, -0.027584, -0.056902, 0.13326, 0.065289, 0.213697, -0.27214, 0.214027, -0.306636, 0.124635, 0.071718, -0.007949, -0.157406, -0.252464, 0.082236, 0.063095, 0.204937, -0.224094, -0.439095, -0.171125, -0.000141, 0.228232, 0.116835, 0.07887, 0.076962, 0.100736, 0.083321], [0.029213, 0.090497, 0.125168, -0.032123, -0.004873, -0.06944, 0.268403, -0.126975, -0.048845, 0.124609, 0.169386, -0.183604, 0.209634, -0.08491, 0.10165, 0.088256, -0.098016, 0.140993, 0.106937, 0.193338, 0.176124, 0.099116, -0.265971, -0.126782, 0.159968, -0.136762, -0.061757, 0.001035, 0.11898, -0.149104, 0.214769, 0.180519], [-0.16374, -0.098392, -0.033716, 0.014974, 0.064029, 0.1496, -0.010887, -0.002726, 0.004474, 0.166808, -0.056954, -0.285222, 0.05222, 0.1517, -0.012093, 0.012399, 0.001687, 0.127765, -0.097668, 0.012098, -0.044509, -0.089104, -0.01079, -0.028866, -0.019088, -0.007057, -0.164984, -0.106911, -0.009483, -0.195285, 0.066344, 0.043123], [-0.364206, 0.056685, 0.104944, -0.08491, -0.032153, -0.122658, -0.14823, -0.196706, -0.105947, -0.106013, 0.023398, -0.832389, 0.079962, -0.091355, -0.018256, -0.00714, -0.020478, -0.041555, -0.429883, 0.00802, 0.050071, 0.02806, -0.249987, -0.089464, -0.34294, -0.164542, -0.028562, -0.087923, -0.134943, -0.517374, -0.097415, -0.173267], [0.016987, 0.050542, 0.019823, 0.010118, 0.023337, -0.002827, -0.62151, 0.095906, 0.097094, 0.071114, -0.111456, 0.214279, 0.008101, 0.054438, 0.038237, -0.222165, -0.165729, -0.18573, -0.140268, 0.027549, 0.11536, 0.082261, -0.08609, 0.038697, -0.163712, -0.08421, -0.061141, 0.117746, 0.037546, -0.160598, 0.106201, 0.020917], [-0.065724, 0.035122, 0.094775, 0.060617, 0.021797, 0.104125, -0.006278, 0.006131, -0.243498, 0.177354, 0.049232, -0.538131, 0.119241, 0.109287, -0.079457, 0.158997, -0.123028, 0.128751, -0.017067, 0.106647, 0.147421, 0.056705, -0.166894, -0.370192, 0.149799, 0.118819, -0.00823, 0.006498, -0.056243, 0.068262, 0.099536, 0.033612], [0.030008, 0.156055, -0.044868, -0.045051, 0.115491, 0.102878, -0.045797, -0.001275, -0.026871, 0.086433, -0.048211, -0.012529, -0.023753, -0.161087, -0.378422, 0.048051, 0.078574, -0.062614, 0.018442, 0.022393, 0.164455, 0.165917, 0.053962, 0.295903, -0.06788, -0.04725, 0.129795, -0.003273, 0.038295, -0.109191, 0.067983, -0.075691], [-0.212501, -0.090884, 0.014495, 0.036987, -0.06357, -0.173982, -0.122128, -0.058898, -0.17488, -0.013972, -0.010339, 0.318402, -0.08987, -0.054572, 0.249907, -0.028778, -0.174357, -0.197199, -0.049718, 0.137621, -0.096371, -0.081725, -0.222013, 0.191227, -0.270177, -0.168364, -0.105762, 0.043972, -0.080675, 0.08745, 0.068297, 0.320634], [0.072115, 0.010254, 0.119776, 0.074897, 0.097896, 0.11725, -0.10931, 0.075532, 0.040197, -0.018061, -0.018484, -0.127142, 0.070173, -0.146572, -0.290691, 0.169092, -0.048586, 0.046809, -0.175991, -0.060166, -0.078575, -0.049654, -0.077022, -0.068527, -0.037388, 0.103515, 0.134927, -0.022339, 0.079359, -0.043786, 0.120558, -0.12588], [0.17388, -0.104914, -0.11778, -0.137882, -0.098921, 0.010076, 0.060639, 0.140233, 0.0721, 0.056607, -0.006704, 0.055142, -0.025608, -0.092374, -0.104116, -0.092047, 0.111678, -0.012902, -0.063041, -0.269151, -0.104896, 0.096918, 0.055208, 0.120201, -0.061486, -0.046082, 0.099927, 0.138076, -0.12451, 0.034204, -0.107921, -0.266511], [-0.342879, 0.15694, 0.059155, -0.083189, -0.084435, -0.228536, -0.298459, -0.142432, -0.058348, 0.062323, 0.05888, -0.480494, -0.037841, 0.004227, -0.127256, 0.006939, -0.059603, -0.362056, -0.364746, 0.071156, 0.048696, 0.021471, -0.212231, 0.030029, -0.223414, -0.068441, -0.096761, -0.212228, 0.008706, -0.10789, 0.028625, -0.023974], [-0.15482, 0.202599, 0.272274, 0.01131, 0.013462, 0.017472, 0.232378, 0.025118, -0.426743, 0.119956, 0.00963, 0.116779, 0.188506, -0.232515, -0.145134, 0.005739, 0.04561, -0.11819, -0.05395, 0.097879, 0.182964, 0.157901, 0.003058, -0.042837, -0.04734, -0.068824, 0.075376, 0.026059, 0.082926, -0.029203, 0.173345, 0.119405], [-0.013706, 0.13502, -0.169467, -0.046046, -0.054234, 0.048853, 0.229289, 0.141965, -0.113632, 0.15818, 0.263323, -0.055091, 0.041136, 0.080016, -0.026908, 0.144104, -0.080694, 0.1415, 0.101882, 0.190281, 0.072696, 0.186105, 0.108201, -0.382911, 0.061219, 0.062058, 0.097976, 0.219822, 0.073223, -0.035941, 0.026058, -0.009816], [0.084171, 0.157508, 0.018339, 0.085053, 0.295664, 0.039071, 0.026352, 0.034765, 0.011166, 0.06379, -0.007081, -0.293422, 0.109241, -0.003674, 0.018476, -0.076611, 0.069977, 0.045097, -0.019, 0.190878, 0.088354, 0.183611, 0.116082, -0.122931, -0.002261, 0.075264, 0.090658, -0.21127, 0.091537, -0.053783, 0.202198, 0.250502], [-0.023971, 0.012541, 0.055183, -0.028246, -0.102154, 0.042893, -0.333861, -0.159722, -0.086958, -0.295289, -0.236307, -0.425208, -0.004514, -0.012785, 0.235303, -0.046091, -0.01572, -0.113006, -0.226196, -0.141767, -0.121508, -0.107609, -0.360073, -0.331391, 0.035179, -0.039574, -0.076392, -0.010095, -0.259798, -0.027128, -0.133587, 0.068911], [0.079789, -0.159906, 0.033837, -0.123254, -0.068076, -0.016403, -0.049476, 0.079924, 0.053467, 0.113297, -0.077529, 0.122335, -0.131846, -0.053073, -0.011181, 0.213277, -0.003514, 0.074229, 0.005462, 0.035875, 0.039027, -0.112495, 0.188953, 0.301611, -0.099809, 0.098488, -0.092395, 0.180405, -0.079427, -0.049983, -0.027129, -0.168677], [0.101098, -0.115594, -0.004965, 0.024064, -0.022644, -0.099947, -0.208635, 0.012782, 0.104161, -0.420888, -0.344317, -0.048115, -0.159217, 0.259821, -0.201206, -0.089342, -0.078711, 0.083495, -0.033408, -0.234914, 0.058571, 0.126916, -0.047999, -0.079768, 0.11833, -0.148626, -0.099637, 0.332512, -0.168536, -0.074056, -0.05703, -0.417867], [-0.134113, -0.103902, -0.042181, 0.046835, -0.328634, 0.089047, -0.01077, 0.092399, -0.05664, -0.225379, -0.024655, -0.237671, -0.110125, 0.027903, 0.235135, 0.041555, -0.033826, 0.09902, -0.132289, -0.098253, -0.213582, -0.016162, 0.042345, 0.042576, 0.008306, 0.112558, 0.110142, 0.167411, -0.164013, -0.017936, -0.273906, -0.226959], [-0.120164, -0.044774, 0.099584, 0.071836, -0.021961, -0.226245, -0.020953, -0.306766, -0.350728, 0.047141, -0.072193, -0.19483, -0.047735, -0.261369, -0.090556, -0.12036, -0.048128, -0.155494, 0.014092, 0.002866, -0.007066, 0.158361, -0.019676, -0.14892, -0.265269, -0.01337, -0.001424, -0.310816, 0.045942, -0.150193, -0.125428, -0.069477], [-0.084251, 0.096097, -0.069609, -0.056378, 0.054143, -0.053938, -0.110785, 0.077516, 0.013555, 0.160102, 0.069532, -0.013737, -0.099178, 0.00454, -0.000386, -0.088677, -0.182206, -0.110612, -0.082405, 0.134896, -0.022433, -0.063584, -0.093375, 0.011059, -0.033644, -0.219391, -0.054455, -0.07362, 0.065445, -0.096747, 0.164539, -0.037527], [-0.418894, 0.043791, 0.026647, 0.084252, -0.033233, -0.051061, -0.312771, -0.00308, -0.17992, 0.089415, -0.006566, -0.056468, 0.015791, 0.04814, -0.039477, -0.259819, -0.161656, 0.097439, 0.019178, -0.068012, 0.012755, -0.014029, -0.059526, 0.270722, -0.015865, 0.071789, 0.016949, 0.038802, 0.025993, -0.400751, -0.06619, -0.034627], [0.054301, -0.019319, 0.075314, -0.194061, -0.052222, 0.062768, -0.096564, 0.010174, 0.090027, -0.099021, -0.170165, 0.182753, -0.000884, 0.157705, 0.030802, 0.070232, 0.127652, 0.058447, 0.082372, -0.22077, -0.192313, 0.115276, -0.027409, -0.104032, 0.003469, 0.105809, -0.055923, 0.025167, -0.131537, 0.022122, -0.073272, -0.282047], [-0.033373, -0.116293, -0.05078, -0.053534, 0.122316, 0.012789, -0.056635, 0.003046, 0.049298, 0.149291, -0.051786, 0.254818, -0.103745, -0.058754, 0.205609, 0.160591, 0.0179, 0.102097, 0.091002, -0.158615, -0.152565, -0.036682, -0.099362, -0.077372, -0.011649, 0.094444, 0.229174, -0.079941, -0.111178, -0.039858, -0.102594, -0.246508], [-0.237457, 0.090932, -0.103455, 0.016901, -0.041434, -0.136659, -0.07541, -0.148332, -0.391505, 0.149947, 0.027874, -0.031206, 0.050617, -0.167018, 0.211225, 0.100852, 0.046642, -0.079494, -0.065774, -0.108469, -0.185404, 0.146394, -0.050348, -0.006592, 0.016811, 0.165066, 0.124831, -0.239418, -0.204323, -0.427927, -0.027436, -0.142104], [-0.11335, 0.08973, 0.141534, 0.112663, 0.020717, 0.146538, 0.111254, -0.065577, -0.012755, 0.07138, -0.144874, 0.058262, 0.131938, -0.276851, -0.053308, 0.18068, 0.18781, -0.077145, -0.114497, 0.124053, 0.120259, 0.198468, -0.001907, 0.113099, 0.082598, 0.117879, 0.160768, 0.059412, 0.172037, 0.139891, 0.023499, -0.041284], [0.118733, -0.08346, 0.016254, 0.1209, 0.095398, 0.176051, -0.024289, -0.129508, -0.117983, -0.143635, -0.49308, -0.187452, 0.072549, -0.111448, 0.157966, -0.022573, 0.175713, 0.059816, 0.043522, -0.061294, -0.024293, -0.073064, 0.11292, -0.251233, 0.032277, 0.02809, 0.02639, -0.109982, 0.037077, -0.077046, -0.06326, -0.015442], [-0.001851, -0.005849, -0.025145, 0.067922, -0.28931, 0.077904, -0.06018, 0.000909, 0.055153, 0.143878, 0.103761, -0.171343, -0.137165, 0.098979, -0.030832, 0.015843, 0.155887, 0.163869, 0.03198, -0.143165, -0.162614, -0.162237, -0.033312, 0.020087, 0.101585, 0.070488, 0.055269, 0.129895, -0.110696, -0.049338, -0.03849, 0.114456], [0.039382, 0.10829, 0.15585, -0.028194, 0.12863, -0.038767, -0.079657, 0.010158, 0.052367, 0.161111, 0.234661, 0.070313, -0.015843, 0.131352, 0.275169, -0.28344, 0.101117, 0.121115, 0.195768, 0.087569, 0.10634, 0.029676, 0.037659, -0.154662, 0.0566, -0.217544, 0.03458, 0.169814, 0.054036, 0.009891, 0.182851, -0.020075], [0.301516, -0.150727, -0.037787, -0.091141, -0.102283, 0.102268, -0.194074, 0.146982, 0.242254, -0.186077, -0.101568, 0.187436, -0.153332, 0.178486, -0.203417, 0.050731, 0.184424, 0.053963, 0.273966, -0.173779, 0.013605, -0.052456, -0.158477, 0.055261, 0.126652, 0.033456, -0.239172, 0.04812, -0.077789, 0.396605, -0.097712, -0.349861], [0.01381, 0.062772, -0.046722, -0.063256, 0.05601, -0.055211, -0.100832, -0.03179, 0.115327, -0.030993, -0.009283, 0.029226, 0.064101, 0.057138, 0.035336, -0.022987, 0.01637, 0.054021, -0.15113, 0.124175, 0.16111, -0.002532, 0.099407, -0.141055, -0.098628, 0.061106, -0.021689, 0.171741, 0.016107, -0.158733, 0.072395, 0.061105], [-0.405826, -0.017392, -0.06792, -0.028121, -0.06942, -0.396404, -0.265332, -0.111507, -0.035377, 0.063078, 0.019754, 0.233768, -0.060596, -0.001048, 0.056646, -0.548017, -0.138967, -0.252096, 0.187265, -0.057108, 0.102284, 0.024415, 0.081796, 0.136434, -0.037438, -0.292919, 0.139437, 0.054217, 0.041174, -0.175229, -0.075196, -0.17352], [-0.240005, 0.169842, -0.024323, 0.065301, -0.194136, -0.283356, 0.032637, -0.083807, -0.021168, -0.085463, -0.038954, -0.000283, 0.109163, 0.083364, 0.123331, 0.010275, -0.167846, -0.242963, 0.045716, 0.039047, -0.078828, 0.002267, -0.146646, 0.030376, -0.144007, -0.109951, -0.036043, -0.076025, 0.006286, 0.031889, 0.099646, 0.129447], [0.088239, 0.051312, -0.091212, -0.045, 0.362579, 0.021264, 0.151016, 0.052283, 0.063221, -0.076685, -0.208746, 0.009763, -0.077091, 0.137534, -0.092049, 0.036206, -0.010619, 0.169965, 0.234265, -0.156216, 0.019372, -0.074862, 0.113182, -0.26195, 0.096011, 0.01567, -0.011927, -0.006163, -0.009712, 0.024991, -0.103902, -0.150409], [-0.183345, 0.077168, -0.047501, -0.120316, 0.035759, 0.052328, 0.025784, -0.042313, -0.15577, 0.101744, -0.058546, -0.091438, -0.101137, 0.161982, 0.17163, -0.271647, 0.038679, -0.057612, -0.048218, -0.025117, 0.001519, 0.123551, -0.04701, -0.053003, 0.111962, -0.072571, 0.106361, 0.054043, -0.138094, -0.272555, -0.063836, 0.039523], [0.220485, 0.041176, -0.024683, -1.6e-05, -0.05581, 0.196013, 0.004172, 0.027169, -0.052287, 0.141655, -0.09996, -0.430812, -0.065156, 0.055046, -0.148946, 0.082973, 0.11691, 0.105807, -0.135184, -0.081282, 0.073521, 0.076863, -0.060881, 0.064912, 0.232673, 0.013253, -0.054356, 0.169745, -0.036176, 0.135666, 0.00547, 0.036473], [0.262129, -0.106674, 0.116577, -0.089522, 0.153362, 0.103044, -0.137692, 0.153249, 0.154632, 0.164809, 0.099698, 0.217449, 0.154636, 0.189025, -0.401223, -0.109458, 0.098314, 0.305036, 0.154998, 0.192502, -0.003295, 0.047705, -0.001226, -0.161678, 0.268624, -0.065827, 0.021959, 0.205546, 0.084642, -0.045988, 0.031809, 0.079299], [-0.29381, -0.05001, 0.021632, 0.038528, -0.217106, -0.123448, -2.6e-05, 0.085026, -0.334032, 0.012673, 0.011975, 0.027499, -0.086456, -0.158925, -0.005552, 0.009851, -0.150869, -0.129293, -0.067031, -0.108837, 0.012188, 0.03255, 0.099723, 0.011093, -0.121433, 0.103566, -0.129813, 0.094522, 0.001096, -0.241152, -0.114396, -0.022028], [0.166077, -0.100893, 0.089759, 0.063815, -0.240205, -0.032354, 0.161282, -0.004894, -0.032884, -0.014425, -0.033501, -0.244705, 0.090582, -0.103013, 0.090345, -0.02242, -0.039296, 0.085799, -0.125034, 0.100676, -0.141416, -0.08864, 0.12338, -0.169665, 0.123952, 0.026097, -0.006031, -0.162155, 0.050997, 0.090228, 0.005274, 0.136181], [0.162162, 0.009519, 0.152835, 0.029935, 0.086949, 0.080303, 0.21596, 0.078093, 0.07859, 0.224662, 0.157377, 0.139856, -0.039408, 0.029333, 0.192322, -0.05264, 0.148968, -0.043011, 0.127574, 0.248626, -0.049596, 0.087233, -0.003195, -0.089919, 0.222012, 0.087364, 0.011276, -0.308983, 0.047941, 0.089749, 0.140691, 0.105858]], [-0.088882, 0.123563, 0.080655, 0.109122, 0.050122, 0.06006, 0.033225, 0.086767, -0.052283, -0.239468, -0.129399, -0.146761, 0.047567, -0.016211, -0.01783, 0.181639, 0.124545, 0.046701, -0.145332, -0.022393, 0.138729, 0.058349, 0.154006, -0.088758, -0.17783, 0.095622, 0.072058, 0.03934, -0.107525, -0.141332, 0.071798, -0.06465]),
    ([[0.347932], [-0.147188], [-0.093466], [-0.005907], [-0.186618], [0.15218], [0.417094], [0.177668], [0.183366], [-0.324353], [-0.272949], [0.239712], [-0.084076], [0.190046], [-0.217289], [0.250396], [0.165998], [0.241982], [0.165498], [-0.215202], [-0.083476], [-0.12136], [0.141479], [0.245653], [0.245069], [0.069803], [-0.094263], [0.126866], [-0.077331], [0.233032], [-0.156342], [-0.093046]], [0.114574]),
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
        'fq_k': 0.1,
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
        # exp_r3_mq: 1 = passive quotes use mid + mq_b * (MLP/ridge prediction before the fair_sig blend) + (1 - mq_b) * fair_sig. 0 = off.
        'mq_on': 1,
        'mq_b': 0.5,
        # exp_r3_mlpq: horizon (ms) of the MLP behind the quote signal: 0 = the taker's 200 ms MLP, 500 / 1000 = own MLPs (MLPQ below).
        'mq_h': 500,
        # exp_r3_rfit: 0 = champion model 3; else the PRED_MODELS index of a refit variant (5 parity, 6 r3_100, 7 r3m_75, 8 r3m_100,
        # 9 r3r2_75, 10 r3r2_100, 11 h100, 12 h200, 13 alpha x3, 14 alpha /3, 15 h250, 16 h300, 17 h400, 18 h200+r2, 19 h200+mirror).
        'rfit': 19,
        # exp_r3_mlp: mlp_on 1 = the taker prediction is (1 - mlp_mix) * ridge + mlp_mix * MLP (before pred_blend with fair_sig);
        # the MLP runs only when |ridge| >= mlp_gate ticks (runtime). 0 = off (identical to the champion).
        'mlp_on': 1,
        'mlp_mix': 1.0,
        'mlp_gate': 0.3,
        # exp_r3_seq: 1 = the gated MLP output is replaced by the streaming GRU model (see SEQ_*). 0 = off (mm_r3_mlp2).
        'seq_on': 1,
        # share of the GRU model in the gated prediction (1 = GRU only, else blended with the MLP)
        'seq_mix': 1.0,
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
        # exp_r3_slow: slw_on 1 = a learned slow-direction ridge ALSO pulls the quote against its predicted move (|pred| >= slw_thr ticks);
        # 2 = it REPLACES the hand gate. slw_h 1 / 5 = the 1 s / 5 s ridge, 6 = their sum. 0 = off (identical to mm_r3_mlp2).
        'slw_on': 1,
        'slw_thr': 2.0,
        'slw_h': 5,
        'slw_every_us': 20000,
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
        self.slw_on = int(params['slw_on'])
        self.slw_thr = float(params['slw_thr'])
        self.slw_h = int(params['slw_h'])
        self.slw_every = int(params['slw_every_us'])
        self.slw_t: list = []
        self.slw_m: list = []
        self.slw_c: list = []
        self.slw_head = 0
        self.slw_last = -10**12
        self.slw_cf = 0.0
        self.slw_rm2 = 0
        self.slw_pred = 0.0
        self.slw_p_t = -10**12
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
        self.mq_on = int(params['mq_on'])
        self.mq_b = float(params['mq_b'])
        self.mq_h = int(params['mq_h'])
        if self.mq_h:
            q = MLPQ[self.mq_h]
            self.mq_mean = np.array(q[0]); self.mq_scale = np.array(q[1]); self.mq_layers = [(np.array(w), np.array(b)) for w, b in q[2]]
        self.pred_raw = 0.0
        self.pred_blend = float(params['pred_blend'])
        self.pred_c0, self.pred_w = PRED_MODELS[int(params['pred_model'])]
        self.mlp_on = int(params['mlp_on'])
        self.mlp_mix = float(params['mlp_mix'])
        self.mlp_gate = float(params['mlp_gate'])
        if self.mlp_on:
            self.mlp_mean = np.array(MLP_MEAN)
            self.mlp_scale = np.array(MLP_SCALE)
            self.mlp_layers = [(np.array(w), np.array(b)) for w, b in MLP_LAYERS]
        self.seq_on = int(params['seq_on'])
        self.seq_mix = float(params['seq_mix'])
        if self.seq_on:
            self.sq_tm = np.array(SEQ_TM); self.sq_tsd = np.array(SEQ_TSD)
            self.sq_xm = np.array(SEQ_XM); self.sq_xs = np.array(SEQ_XS)
            self.sq_head = [(np.array(a), np.array(b)) for a, b in SEQ_HEAD]
            # one matmul per event: [token, h] @ W -> [r|z pre-activations (halved: sigmoid = 0.5 + 0.5 tanh(x/2)),
            # input part of n, hidden part of n]; the token normalisation is folded into W and the bias
            n = self.sq_n = len(SEQ_BH) // 3
            wi = np.array(SEQ_WI); bi = np.array(SEQ_BI); wh = np.array(SEQ_WH); bh = np.array(SEQ_BH)
            nt = len(SEQ_TM)
            wn = wi / self.sq_tsd[:, None]
            off = (self.sq_tm / self.sq_tsd) @ wi
            W = np.zeros((nt + n, 4 * n))
            W[:nt, :2 * n] = 0.5 * wn[:, :2 * n]; W[nt:, :2 * n] = 0.5 * wh[:, :2 * n]
            W[:nt, 2 * n:3 * n] = wn[:, 2 * n:]; W[nt:, 3 * n:] = wh[:, 2 * n:]
            self.sq_w = W
            self.sq_b = np.concatenate((0.5 * (bi[:2 * n] - off[:2 * n] + bh[:2 * n]), bi[2 * n:] - off[2 * n:], bh[2 * n:]))
            self.sq_lo = self.sq_tm - 8.0 * self.sq_tsd; self.sq_hi = self.sq_tm + 8.0 * self.sq_tsd
            self.sq_buf = np.zeros(nt + n)
            self.sq_tv = self.sq_buf[:nt]
            self.sq_h = self.sq_buf[nt:]  # the GRU state lives in the input buffer
            self.sq_t = -1
            self.sq_bb = self.sq_ba = -1
            self.sq_qb = self.sq_qa = 0
            self.sq_ltp = 0
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
        if self.seq_on:
            self._seq_step(snapshot.timestamp_us, bids, asks, trades)
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
        if self.slw_on:
            sp = self._slow(now, bb, ba)
            if sp <= -self.slw_thr:
                pull_bid = True
                if self.gate_rungs:
                    self._clear_side(0, 1)
            elif sp >= self.slw_thr:
                pull_ask = True
                if self.gate_rungs:
                    self._clear_side(1, -1)
        if self.gate_thr > 0.0 and self.slw_on != 2:
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

    def _seq_step(self, now: int, bids, asks, trades: list[Trade]) -> None:
        """One GRU step on this event's token (research/diag/seq_r3.tokens, lite columns), from the market-only touch."""
        bb = ba = -1
        qb = qa = 0
        for v in bids:
            if v.owner != 'market':
                continue
            if bb < 0:
                bb = v.price
            elif v.price != bb:
                break
            qb += v.quantity
        for v in asks:
            if v.owner != 'market':
                continue
            if ba < 0:
                ba = v.price
            elif v.price != ba:
                break
            qa += v.quantity
        if bb < 0 or ba < 0 or ba <= bb:  # one-sided: the features carry the last two-sided touch
            if self.sq_bb < 0:
                return
            bb = self.sq_bb; ba = self.sq_ba; qb = self.sq_qb; qa = self.sq_qa
        if self.sq_bb < 0:
            self.sq_bb = bb; self.sq_ba = ba; self.sq_qb = qb; self.sq_qa = qa; self.sq_t = now
        pbb = self.sq_bb; pba = self.sq_ba
        tv = 0
        tc = 0
        for t in trades:
            if t.taker_owner == 'user':
                continue
            tc += 1
            tv += t.quantity if t.book_side == 'ask' else -t.quantity
            self.sq_ltp = t.price
        ltp = self.sq_ltp
        sw = 0.0
        if tc and tv > 0:
            sw = min(max(ltp - pba + 1, 0), 11)
        elif tc and tv < 0:
            sw = -min(max(pbb - ltp + 1, 0), 11)
        pmid = (pbb + pba) * 0.5
        mid = (bb + ba) * 0.5
        db = qb - self.sq_qb if bb == pbb else 0
        da = qa - self.sq_qa if ba == pba else 0
        dt = now - self.sq_t
        tv_ = self.sq_tv
        tv_[:] = (log1p(dt / 100.0), min(max(bb - pbb, -5), 5), min(max(ba - pba, -5), 5), min(max(mid - pmid, -5.0), 5.0),
                        (log1p(db) if db > 0 else -log1p(-db)), (log1p(da) if da > 0 else -log1p(-da)),
                        (log1p(tv) if tv > 0 else -log1p(-tv)), sw, min(max(ltp - pmid, -10.0), 10.0) if tc else 0.0,
                        min(ba - bb, 20), log1p(qb), log1p(qa), (qb - qa) / max(qb + qa, 1))
        np.maximum(tv_, self.sq_lo, out=tv_)
        np.minimum(tv_, self.sq_hi, out=tv_)
        n = self.sq_n
        g = self.sq_buf @ self.sq_w + self.sq_b
        rz = 0.5 + 0.5 * np.tanh(g[:2 * n])
        c = np.tanh(g[2 * n:3 * n] + rz[:n] * g[3 * n:])
        h = self.sq_h
        h[:] = c + rz[n:] * (h - c)
        self.sq_bb = bb; self.sq_ba = ba; self.sq_qb = qb; self.sq_qa = qa; self.sq_t = now

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
        self.slw_cf += fl
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
            h0 = (np.array((pg[0], pg[1], f70, pg[3], pg[4], f70 * log1p(fw), f70 * af, sg100 * af, f70 * spread, spread,
                           imb1, cnt1, log1p(qb), log1p(qa), ltr_sign, ltr_lsz, (ltp - mid2 * 0.5) if ltp > 0 else 0.0,
                           (mid2 - rm2) * 0.5, self.pr_ev10, s100, self.pr_s1000,
                           log1p((now - self.pr_ttr) / 1000.0) if self.pr_ttr >= 0 else 10.0)))
            h = (h0 - self.mlp_mean) / self.mlp_scale
            if self.seq_on and self.seq_mix < 1.0:
                hm0 = np.clip(h, -8.0, 8.0)
                ls = self.mlp_layers
                hm0 = np.maximum(hm0 @ ls[0][0] + ls[0][1], 0.0)
                hm0 = np.maximum(hm0 @ ls[1][0] + ls[1][1], 0.0)
                m0 = float((hm0 @ ls[2][0] + ls[2][1])[0])
            if self.seq_on:
                h = np.clip((h * self.mlp_scale + self.mlp_mean - self.sq_xm) / self.sq_xs, -8.0, 8.0)
                ls = self.sq_head
                h = np.maximum(np.concatenate((h, self.sq_h)) @ ls[0][0] + ls[0][1], 0.0)
                h = np.maximum(h @ ls[1][0] + ls[1][1], 0.0)
                m = float((h @ ls[2][0] + ls[2][1])[0]) * SEQ_YSC
                if self.seq_mix < 1.0:
                    m = self.seq_mix * m + (1.0 - self.seq_mix) * m0
            else:
                h = np.clip(h, -8.0, 8.0)
                ls = self.mlp_layers
                h = np.maximum(h @ ls[0][0] + ls[0][1], 0.0)
                h = np.maximum(h @ ls[1][0] + ls[1][1], 0.0)
                m = float((h @ ls[2][0] + ls[2][1])[0])
            pq = p
            p = (1.0 - self.mlp_mix) * p + self.mlp_mix * m
            if self.mq_h:
                g = np.clip((h0 - self.mq_mean) / self.mq_scale, -8.0, 8.0)
                ls = self.mq_layers
                g = np.maximum(g @ ls[0][0] + ls[0][1], 0.0)
                g = np.maximum(g @ ls[1][0] + ls[1][1], 0.0)
                pq = float((g @ ls[2][0] + ls[2][1])[0])
            self.pred_raw = pq if self.mq_h else p
        else:
            self.pred_raw = p
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
        elif self.mq_on:
            sig = self.mq_b * self.pred_raw + (1.0 - self.mq_b) * self.fair_sig
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

    def _slow(self, now: int, bb: int, ba: int) -> float:
        """Slow-direction ridge (ticks): 1 s / 5 s robust-mid move from slow drifts, hidden-fair signals, flow, spread and hole."""
        if ba - bb <= 3:
            self.slw_rm2 = bb + ba
        rm2 = self.slw_rm2
        if rm2 <= 0:
            return 0.0
        ts = self.slw_t
        if now - self.slw_last >= 50000:
            ts.append(now)
            self.slw_m.append(rm2)
            self.slw_c.append(self.slw_cf)
            self.slw_last = now
            head = self.slw_head
            if ts[head] < now - 61000000:
                head = bisect_left(ts, now - 61000000, head)
                self.slw_head = head
                if head > 4000:
                    del ts[:head]
                    del self.slw_m[:head]
                    del self.slw_c[:head]
                    self.slw_head = 0
        if now - self.slw_p_t < self.slw_every:
            return self.slw_pred
        self.slw_p_t = now
        head = self.slw_head
        ms = self.slw_m
        cs = self.slw_c
        x = []
        for w in (1000000, 5000000, 15000000, 30000000, 60000000):
            j = bisect_right(ts, now - w, head) - 1
            x.append((rm2 - ms[j]) * 0.5 if j >= head else 0.0)
        x.extend(self.pf_sig)
        for w in (1000000, 3000000, 10000000):
            j = bisect_right(ts, now - w, head) - 1
            x.append(self.slw_cf - cs[j] if j >= head else self.slw_cf - (cs[head] if cs else 0.0))
        x.append(float(ba - bb if ba - bb < 10 else 10))
        x.append((bb + ba - rm2) * 0.5)
        x.append(abs(x[1]))
        x.append(abs(x[3]))
        xa = np.array(x)
        p = 0.0
        if self.slw_h in (1, 6):
            mn, sc, w, c = SLW_STD1
            p += float(np.clip((xa - mn) / sc, -8.0, 8.0) @ w) + c
        if self.slw_h in (5, 6):
            mn, sc, w, c = SLW_STD5
            p += float(np.clip((xa - mn) / sc, -8.0, 8.0) @ w) + c
        self.slw_pred = p
        return p

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
