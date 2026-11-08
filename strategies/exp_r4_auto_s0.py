"""exp_r4_auto: mm_r3_gru3 with the learned tables (ridge,mlp,mq,slow,seq) refitted by research/retrain.py on round4.csv + mirror (first 75 %).

mm_r3_gru3 (home, R3 upload candidate): mm_r3_gru2 (gate_thr 14) + rt_ms 150, fq_k_sp2 1.5 and a 50 ms hysteresis after passive flips (hy_only 1, hy_ms 50, hy_k 0.2; exp_r3_hyst): round3 653,327 / mirror 639,714 / round2 447,310; holdout 157,925 / 158,799 / 112,106 (gru2 638,024 / 625,673 / 447,035; holdout 157,152 / 158,439 / 112,310).

mm_r3_gru2 (home, R3 upload candidate): mm_r3_gru (leaderboard 976,148) with gate_thr 14 (from 11): round3 638,024 / mirror 625,673 / round2 447,035; holdout 157,152 / 158,439 / 112,310 (gru 634,519 / 624,778 / 437,941; holdout 156,075 / 157,441 / 109,156); gate 17 / 22 on the same plateau / lower.

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
    # rfit 19 (auto: h200 on round4 + mirror, first 75 %, research/retrain.py); params rfit=19
    (0.0, (0.4391014, -1.7195359, 3.922156, -2.2004199, 1.5018903, -0.3131962, 0.0088583, 0.0533107, -0.0117911, 0.0, -0.151299, 0.1926082, 0.0417247, -0.0417247, -0.032648, -0.0081436, 0.1781465, -0.2981396, 0.0004944, 0.0004517, -5.49e-05, 0.0)),
)
# exp_r3_mlp: 22-32-16-1 ReLU MLP on the same 22 features, target = 200 ms (robust mid at t+5ms+200ms - mid now), fitted on the
# first 75 % of round3 + mirror (research/diag/mlp_r3.py); holdout corr 0.49 vs 0.463 for the ridge. Inputs clipped at +-8 sd.
MLP_MEAN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.257423, 0.0, 0.0, 5.846345, 5.846663, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 3.98748]
MLP_SCALE = [1.124389, 0.990516, 0.944776, 0.844764, 0.756599, 2.576755, 18.120184, 0.790505, 15.923768, 0.695558, 0.651213, 0.540523, 1.677436, 1.677418, 0.999993, 4.403104, 1.087599, 0.310604, 132.394562, 334.616058, 1066.178833, 1.340722]
MLP_LAYERS = [
    ([[0.078451, -0.245987, -0.094275, -0.065844, -0.150776, 0.016341, -0.225209, 0.187321, 0.048394, -0.579863, 0.083233, -0.011789, 0.181566, 0.01931, 0.117911, -0.076926, -0.108904, -0.284799, 0.199397, 0.134051, -0.15207, -0.011793, 0.035088, -0.011441, 0.132351, -0.189822, 0.002204, -0.085158, 0.723718, 0.027766, 0.091602, -0.010808], [0.087542, 0.111289, -0.183102, -0.023346, -0.13434, 0.095984, -0.078783, -0.241673, -0.063707, -0.24873, 0.049337, 0.1039, -0.000889, 0.030073, 0.086275, -0.155032, 0.113538, -0.183969, -0.06362, 0.321771, -0.40485, -0.063957, 0.240793, 0.02335, 0.077478, -0.080368, -0.01195, 0.017946, 0.101281, 0.16817, 0.019552, 0.019059], [-0.20393, -0.04091, 0.018359, 0.016682, 0.14841, -0.23352, 0.120815, 0.061909, 0.177402, -0.020263, -0.126085, 0.216089, 0.008055, -0.12813, -0.129551, 0.250989, 0.140121, 0.203562, -0.180526, 0.135746, -0.26917, 0.198347, 0.233491, -0.08903, 0.321749, 0.334862, 0.106437, -0.215923, 0.151449, -0.081365, 0.128254, 0.107815], [-0.052581, 0.247743, 0.146537, 0.001899, 0.225162, -0.078693, -0.049215, -0.120802, -0.127956, 0.480834, 0.005633, -0.159038, 0.047573, -0.038637, -0.195419, 0.152974, 0.043029, 0.310776, 0.055954, -0.177045, 0.200801, 0.266295, 0.108355, 0.238519, 0.09744, -0.147432, 0.335795, 0.092077, -0.0673, -0.158692, -0.415067, 0.078676], [-0.475784, -0.098212, -0.113243, -0.117933, 0.340235, 0.087958, 0.013312, -0.001172, -0.094576, 0.316664, -0.126832, 0.235981, 0.270728, -0.238415, -0.063638, -0.064834, -0.113283, 0.525902, 0.235737, -0.100385, -0.036122, -0.066552, 0.122329, 0.111439, -0.446975, -0.225157, -0.069836, 0.455401, -0.036332, 0.275307, 0.054505, 0.057791], [-0.107733, -0.049608, -0.057498, -0.236105, 0.185581, -0.248475, 0.124313, -0.141768, -0.117766, 0.048676, -0.104029, -0.180963, 0.145696, -0.021415, 0.006567, -0.041084, 0.091938, 0.180822, -0.107971, 0.152794, -0.066203, -0.04639, -0.042612, 0.056292, -0.203984, 0.162576, -0.092516, 0.169549, 0.125053, -0.096502, 0.184832, -0.242452], [-0.148505, -0.222716, -0.242352, -0.24154, 0.074418, -0.02957, -0.121955, 0.088004, 0.372162, -0.516409, -0.946071, 0.736274, -0.087095, -0.053911, -0.324882, 0.442371, 0.11822, 0.169617, -0.227408, 0.106843, -0.038878, -0.035374, 0.259523, 0.165683, 0.202706, 0.203735, -0.04185, 0.058163, -0.12397, 0.303244, 0.09543, 0.558727], [0.073672, 0.102859, 0.313079, 0.216039, 0.245589, -0.169875, 0.22168, 0.049053, -0.00227, 0.068093, -0.082943, -0.288809, -0.226239, -0.118292, 0.334815, 0.357549, 0.131849, 0.125573, -0.065977, -0.241637, 0.131149, 0.232033, 0.137039, -0.085304, 0.060276, 0.147205, 0.293063, 0.046123, -0.098601, 0.314007, 0.24006, 0.187721], [0.114565, -0.329734, -0.155357, -0.025974, -0.173592, -0.637024, 0.086188, -0.415513, -0.228455, 0.021323, -0.150902, -0.1916, 0.148366, -0.139386, -0.215985, 0.459457, -0.32945, -0.037286, 0.172727, 0.190271, 0.032322, 0.072182, 0.548441, -0.088638, -0.113774, -0.030356, 0.02259, 0.1292, -0.099809, 0.068492, 0.065271, 0.105254], [0.183357, 0.030161, -0.134319, 0.042977, -0.155854, -0.128905, 0.162383, -0.499035, 0.192308, -0.376214, -0.02294, 0.006441, 0.100065, -0.177665, -0.016368, 0.311618, -1.013282, 0.113119, 0.12553, -0.361135, -0.290215, 0.350218, -0.345645, -0.042299, 0.045352, -0.03886, -0.049415, 0.192242, -0.131203, 0.114967, 0.187379, -0.011265], [0.076758, -0.063736, 0.136244, 0.082618, -0.075797, -0.194412, 0.082551, -0.085735, 0.042464, 0.07473, 0.016653, 0.291589, -0.198953, -0.111606, -0.071522, -0.051675, 0.026597, -0.110226, 0.121774, 0.114805, -0.02093, 0.032629, 0.124231, 0.068207, -0.156777, -0.054709, -0.040896, -0.110019, 0.162267, -0.145202, -0.206133, 0.146334], [-0.140129, 0.116427, -0.41274, -0.375605, -0.521303, -0.258014, 0.078708, -0.000569, -0.510568, -0.076416, 0.173077, 0.101263, 0.140839, -0.245843, 0.110279, 0.20002, 0.257364, 0.21132, 0.466035, 0.12602, 0.017503, 0.063863, -0.050707, -0.034107, 0.371472, -0.260382, -0.287337, 0.00172, -0.22685, 0.019849, 0.24928, 0.220265], [-0.070291, -0.480005, 0.074362, 0.088466, 0.134719, -0.094697, -0.163512, -0.160051, 0.013074, -0.019851, 0.364775, 0.146091, 0.180521, 0.302844, 0.081242, 0.026694, 0.186847, -0.113439, 0.098923, -0.015621, 0.175141, -0.067838, -0.084022, 0.255286, -0.196925, 0.123168, -0.123979, -0.132287, -0.088438, -0.094164, -0.209666, -0.169189], [-0.198697, -0.160795, -0.253107, -0.185136, 0.252972, -0.219821, -0.339768, 0.070877, 0.185767, 0.015177, 0.206271, -0.064501, -0.121565, -0.022398, 0.11627, 0.070261, 0.10058, 0.08149, 0.144298, 0.045992, -0.054364, -0.174686, 0.050658, -0.008743, 0.011496, 0.067037, -0.073784, -0.15777, -0.07251, -0.317252, 0.016995, -0.432246], [0.098876, -0.345276, -0.083545, -0.294475, -0.339165, -0.143713, 0.471088, -0.166609, 0.356541, -0.099307, 0.374427, -0.17385, -0.01247, -0.396526, -0.268376, -0.171617, -0.002985, -0.137283, -0.305625, 0.189708, -0.071935, -0.073967, 0.197053, 0.126883, 0.064796, -0.051431, -0.143358, -0.216281, 0.057387, 0.078866, 0.023515, 0.169824], [-0.264114, 0.210016, -0.098841, 0.30171, -0.182518, 0.024414, -0.260621, -0.040721, -0.128075, 0.115313, 0.005563, 0.158105, 0.149066, -0.330738, 0.069848, 0.12274, -0.079571, 0.068509, 0.023053, 0.012128, 0.03321, 0.001264, 0.083863, -0.132697, 0.140823, 0.38664, -0.036201, -0.032366, -0.264127, -0.20116, 0.503522, -0.063953], [0.058169, 0.042621, 0.08848, 0.303372, -0.137014, -0.360248, -0.145548, 0.342718, -0.412304, 0.152859, 0.074421, -0.25666, 0.039017, 0.117519, 0.627242, 0.500077, -0.104164, 0.124192, 0.563078, -0.30377, -0.144328, -0.254112, -0.21406, -0.203092, -0.020736, -0.217542, 0.727238, -0.290933, 0.32983, 0.006487, 0.135405, 0.190044], [0.14293, 0.226973, 0.557605, -0.381928, 0.185833, -0.258317, -0.282662, -0.482345, 0.1659, 0.300631, -0.005262, -0.165846, -0.148001, -0.107324, -0.083404, 0.177012, 0.035369, -0.168328, -0.088443, 0.033141, 0.031034, 0.03702, 0.31047, -0.157829, 0.269855, -0.074943, 0.176244, 0.058088, 0.203731, -0.150556, -0.195055, -0.20062], [-0.103638, 0.017813, -0.036412, 0.274604, -0.078596, 0.019251, -0.118186, 0.047663, -0.110256, -0.02175, 0.047423, -0.080131, 0.086044, 0.252408, -0.004836, -0.058483, -0.017767, 0.021483, -0.050888, 0.054773, 0.03749, -0.004679, 0.005587, 0.03536, -0.077475, 0.293438, -0.018663, 0.268926, 0.10189, 0.157227, 0.212046, -0.031859], [-0.269523, -0.114166, 0.362557, 0.039566, -0.00575, -0.084, -0.065431, -0.072915, -0.306711, 0.017563, -0.24903, -0.09507, 0.223838, -0.315852, -0.007625, 0.097132, 0.328929, -0.034477, -0.173159, 0.033899, -0.212389, 0.054685, 0.162436, -0.109066, 0.175264, 0.346817, -0.355687, -0.028061, 0.002169, 0.43527, -0.010869, 0.036122], [0.120651, 0.202566, -0.187476, -0.045681, -0.142332, 0.148542, 0.174635, -0.10894, 0.007575, -0.04272, -0.023909, 0.072144, -0.146294, 0.003166, 0.194142, -0.052503, -0.042784, -0.157164, -0.088095, 0.029585, -0.121034, -0.310549, 0.052089, -0.074788, -0.562197, 0.109229, -0.059271, -0.251706, -0.082938, -0.174757, 0.25332, -0.225708], [-0.002711, -0.0006, 0.18965, -0.322506, 0.065582, 0.14665, -0.001738, -0.026285, -0.062776, -0.041691, 0.045574, 0.085309, -0.064357, 0.128188, -0.378286, 0.169698, 0.052038, -0.086088, -0.124959, -0.0529, -0.224141, 0.073139, -0.238536, -0.600584, -0.056456, 0.124813, 0.005833, 0.221572, 0.029144, -0.004436, -0.183, 0.577781]], [-0.19368, 0.010201, -0.116148, -0.008439, 0.292785, 0.048636, 0.193931, -0.179325, -0.377219, -0.422478, 0.450333, -0.587121, 0.161811, 0.024793, 0.056129, -0.026736, 0.329609, -0.143801, -0.612868, -0.476912, 0.006884, -0.424727, 0.0951, -0.181641, -0.111374, -0.373756, -0.051795, 0.400307, -0.188276, 0.093295, -0.099634, 0.027146]),
    ([[-0.225299, -0.111374, -0.148352, -0.147114, 0.311201, -0.228302, 0.084993, -0.366238, 0.043384, 0.027188, -0.010173, 0.181191, -0.067686, -0.247082, -0.232246, 0.05846], [-0.260716, -0.202708, -0.163691, -0.05345, -0.177434, 0.043424, 0.253336, 0.018547, 0.149321, -0.056845, 0.193752, 0.08794, -0.175088, -0.064564, 0.211014, 0.139416], [0.124809, -0.128627, 0.268034, -0.10445, -0.111863, 0.236217, 0.232444, 0.216937, 0.362318, 0.151491, 0.218809, 0.033156, -0.126924, 0.286592, -0.042899, 0.121986], [-0.239123, -0.275614, 0.066778, 0.26554, -0.070556, 0.220416, -0.173881, 0.004641, 0.078659, -0.063558, -0.078278, 0.118698, -0.040728, 0.092364, 0.262524, 0.219497], [-0.485518, 0.166179, -0.216376, -0.147399, -0.01884, -0.005124, -0.600316, -0.188206, -0.215354, -0.03529, 0.070967, -0.522181, -0.167761, -0.091749, 0.041611, 0.187139], [0.237018, -0.087587, 0.196511, 0.151077, 0.131929, 0.374613, 0.175992, 0.144338, -0.003743, 0.144345, 0.183421, 0.116667, -0.041646, -0.104541, -0.190481, -0.258408], [0.215168, 0.196056, 0.167927, -0.013985, 0.12828, 0.204805, 0.204714, 0.124791, 0.245045, 0.17163, 0.106651, 0.035369, -0.123326, -0.447806, -0.160598, -0.255929], [0.071308, -0.290387, -0.023121, -0.34822, -0.079924, 0.078193, 0.123225, -0.093737, 0.013221, 0.189196, 0.298511, 0.021672, -0.137448, 0.150372, -0.049169, -0.106085], [0.013646, 0.063889, -0.039818, -0.421567, -0.31921, 0.011742, 0.153262, -7e-05, 0.006355, 0.136022, 0.12111, 0.341255, 0.024353, 0.030567, -0.116695, -0.814303], [0.258777, -0.461172, 0.198266, 0.36852, 0.000623, 0.479568, 0.077425, 0.281195, 0.320622, 0.673825, 0.202553, 0.166772, -0.134379, 0.534567, -0.217147, 0.005613], [0.008406, 0.188319, -0.216938, -0.074131, -0.154584, -0.367515, 0.094322, -0.072673, -0.190888, -0.032679, -0.052377, -0.011705, -0.201636, -0.371093, -0.098564, -0.159538], [0.082381, -0.216468, 0.13326, 0.213368, 0.119197, 0.031661, -0.257066, 0.220819, 0.091221, 0.387094, -0.33571, -0.028132, 0.099564, -0.08564, -0.240422, 0.172646], [-0.085993, -0.452364, 0.097787, 0.089104, 0.172642, 0.024511, 0.066114, 0.178686, -0.002222, -0.295704, 0.030589, 0.174081, 0.01127, -0.133985, 0.26059, 0.090499], [-0.000666, 0.216108, -0.286181, -0.219629, 0.070739, -0.564664, -0.047022, -0.079178, 0.102563, 0.220624, 0.237576, 0.138259, 0.141899, -0.013201, 0.014183, -0.536879], [-0.169943, 0.161274, -0.051055, 0.059493, 0.131696, -0.008033, 0.141285, 0.033258, -0.188095, -0.631203, 0.062849, 0.23043, -0.107435, 0.222883, -0.012578, 0.330972], [0.088631, -0.066376, 0.085701, 0.256906, 0.128445, 0.067502, 0.22144, 0.101767, -0.243502, 0.188903, 0.120471, 0.310523, -0.097477, 0.059643, -0.11475, 0.237203], [0.082001, -0.031056, -0.26555, -0.078956, -0.064255, 0.145106, 0.062544, 0.136453, -0.434292, -0.468239, 0.014135, -0.243082, -0.016157, 0.175733, -0.065834, -1.249322], [0.026512, -0.555467, 0.255108, 0.124473, 0.068844, 0.180122, -0.37274, 0.294289, 0.195984, 0.371438, -0.264833, 0.077233, 0.011477, 0.263745, 0.156854, -0.169495], [0.342917, 0.164738, -0.023176, -0.095263, -0.161269, 0.215997, -0.506091, -0.08106, -0.286715, -0.272839, 0.077687, -0.268352, 0.122467, -0.312008, -0.016881, 0.277041], [-0.027926, 0.353964, 0.020169, 0.096672, 0.171541, 0.005109, -0.263426, 0.087624, 0.38747, -0.075818, -0.244923, -0.400277, -0.193332, 0.339353, -0.283581, -0.849248], [0.027778, -0.149887, 0.125779, -0.247646, -0.056059, -0.654243, 0.004356, 0.032304, 0.080368, 0.030549, -0.114257, -0.01856, -0.029702, -0.336394, -0.084897, 0.183733], [0.190094, 0.099683, -0.181802, -0.139243, 0.111283, -0.299709, 0.043044, -0.240057, -0.148351, 0.178284, 0.086459, 0.247219, 0.054321, -0.091175, 0.101455, 0.17169], [0.058954, 0.077084, 0.14181, 0.049123, 0.168081, 0.142416, -0.100794, 0.094232, -0.043771, -0.161601, 0.088295, 0.072658, -0.117194, -0.218257, -0.297467, -0.091054], [-0.124974, -0.091577, -0.229158, -0.174872, -0.123596, -0.162928, -0.137485, -0.044479, 0.149723, 0.225717, 0.015097, -0.15289, -0.109988, -0.327838, 0.15418, -0.19973], [-0.072845, 0.161586, 0.189197, 0.021454, -0.190185, 0.262469, 0.124455, 0.1716, 0.222106, 0.010878, 0.099061, 0.103393, 0.039691, 0.215947, -0.034067, -0.532225], [0.099009, 0.142548, -0.276753, -0.334056, -0.002958, -0.380099, -0.280555, -0.20969, 0.16147, 0.546574, 0.062693, -0.085851, -0.167308, -0.178186, 0.087022, 0.044412], [-0.228299, 0.368354, -0.363967, -0.061528, -0.134038, -0.371186, -0.077636, -0.227155, -0.182143, -0.01431, 0.045802, -0.149808, 0.017189, -0.309673, 0.026015, -0.114752], [0.091932, 0.131326, -0.083217, -0.089329, 0.196052, 0.084255, 0.104457, 0.003227, 0.099105, -0.297881, -0.023874, -0.62416, -0.152837, -0.086973, -0.13386, 0.153484], [0.151863, -0.357767, 0.161858, 0.26856, 0.132848, 0.249575, 0.237671, 0.039052, 0.323195, 0.320046, -0.064367, 0.182433, 0.065143, 0.067857, 0.113556, 0.13617], [-0.095738, -0.310965, 0.026649, 0.122493, 0.170315, -0.106705, -0.218278, 0.152114, -0.435442, 0.307832, 0.149619, -0.132322, -0.201487, -0.092473, -0.213674, -0.193401], [-0.169946, 0.276485, 0.046463, 0.020948, -0.029841, 0.091287, -0.059025, -0.005189, -0.218402, -0.441387, 0.036421, -0.071236, 0.034114, 0.338159, -0.237777, -0.213401], [0.251577, 0.060411, -0.067335, -0.084942, 0.063778, -0.151803, 0.267788, 0.160547, -0.130979, -0.417956, 0.068316, 0.063976, -0.178366, -0.358697, -0.272506, -0.335144]], [0.202816, 0.155774, 0.248587, 0.284194, 0.021816, -0.223933, 0.182303, 0.322, 0.232097, -0.357611, 0.130497, 0.071612, -0.208839, -0.114577, 0.42595, -0.203525]),
    ([[0.330553], [0.459502], [0.375361], [0.586682], [0.342533], [0.852687], [-0.299813], [0.484974], [-0.816719], [-1.11092], [-0.140484], [-0.467586], [0.395152], [1.161542], [0.581855], [1.343458]], [-0.462647]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.562321557048493e-19, -1.1343482335572739e-18, -5.671741167786369e-19, -1.8905803892621232e-19, 1.1343482335572739e-18, 0.0, 0.0, 0.0, 1.2572534685440513, 0.0, 10.639801906763806, 28.799029822078616], [6.72849538126497, 19.8058956257327, 34.925161910299096, 44.72666958616459, 57.636846319296275, 1.1243487056677264, 0.9903909371630099, 0.9446913427593254, 0.8452682282888074, 0.7575381583058208, 1067.5606541222405, 2053.035725905977, 4046.2991308026644, 0.6179080587708422, 0.3127841039679992, 16.705331990766382, 34.22120473597514], [0.6490227662551226, 0.3600762349252582, -0.1832944003136108, -0.09733275171505404, -0.10066480452612983, 0.8482856190320313, -3.2486297440276823, 6.1619984101404714, -3.540934484781457, 3.1035095898512166, -0.6019915249010418, 0.06557241931064624, -0.1855095648918967, -0.002102755891520747, 0.4105229362630337, 0.0003153876484541001, 0.00020715056621090425], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.568942287397852e-19, -1.1353413431096778e-18, -5.676706715548389e-19, -1.892235571849463e-19, 1.1353413431096778e-18, 0.0, 0.0, 0.0, 1.2574354160321035, 0.0, 10.646654416476506, 28.81409587438229], [6.7313766144182345, 19.814344420832445, 34.939862329400235, 44.74491730719747, 57.65822173996363, 1.124692105470714, 0.9907164467252751, 0.9450073668099886, 0.8455601770714993, 0.7578101560190531, 1067.786637740178, 2053.787560401509, 4047.179122347928, 0.6181145992138308, 0.3129209935553393, 16.710984248831306, 34.23237508220454], [2.9421164343350763, 0.7964360421909548, -1.0744230210945729, -0.9234918162695617, -0.8475723430517554, 1.4775048291769728, -4.635873707445536, 9.213729195881914, -4.907517756528247, 6.23467901404692, -2.386919742393157, 0.4346797676233983, -1.5368709559241014, 0.001829056109893383, -0.22349228908983945, -0.0002742367847445596, -0.00018005766511644536], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.257441, 0.0, 0.0, 5.84623, 5.846549, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987547], [1.124422, 0.990545, 0.944803, 0.844787, 0.75662, 2.57683, 18.120787, 0.790527, 15.924298, 0.695578, 0.651228, 0.54053, 1.677429, 1.677372, 0.999993, 4.403117, 1.087628, 0.310614, 132.397858, 334.626434, 1066.212891, 1.340704], [([[-0.005992, -0.025224, -0.110018, -0.106359, 0.006085, -0.171442, -0.187738, 0.064849, 0.086749, 0.083156, 0.006536, -0.160726, 0.121319, -0.077854, -0.026678, -0.067273, -0.069217, -0.315064, 0.138989, 0.048036, -0.316149, 0.014142, -0.094388, -0.104547, -0.513554, -0.190908, 0.039054, -0.078057, 0.788193, -0.010232, -0.007736, 0.133935], [0.178565, 0.180104, -0.24781, -0.138369, -0.307894, 0.125673, 0.009063, -0.246298, -0.083457, -0.003639, 0.067612, 0.010382, -0.005552, -0.004285, 0.10022, -0.161484, 0.068697, -0.16246, -0.052096, 0.31713, -0.338132, -0.188281, 0.178396, -0.113489, -0.27142, -0.165519, 0.016581, -0.070404, 0.190119, 0.189464, 0.042602, -0.034313], [-0.174257, -0.049382, 0.016881, -0.048824, -0.040701, -0.224999, 0.224285, 0.062709, 0.225903, 0.043788, -0.081443, 0.143031, -0.017868, -0.098856, -0.079025, 0.276357, 0.067338, 0.231172, -0.15027, 0.129692, -0.153857, 0.109504, 0.189373, -0.125717, 0.127259, 0.322937, 0.123886, -0.346156, 0.143098, -0.052234, 0.144144, 0.01098], [-0.186045, 0.222256, 0.25327, 0.22, 0.276229, -0.208752, 0.106959, -0.103799, -0.019707, 0.20652, -0.0815, -0.377227, -0.017227, 0.101038, -0.216586, 0.122588, 0.012242, 0.248811, 0.112527, -0.186542, 0.211023, 0.454981, 0.034791, 0.340485, 0.347376, 0.010178, 0.236394, -0.099492, -0.302666, -0.22957, -0.490845, -0.029988], [-0.712089, 0.177028, -0.226895, -0.511401, 0.544317, -0.006925, 0.019535, -0.027704, -0.141614, -0.336925, -0.266475, 0.206922, 0.31895, -0.314912, -0.014438, -0.225887, -0.027002, 0.447892, 0.303683, 0.107319, -0.304081, -0.198918, 0.082749, 0.07732, -0.211132, -0.225086, -0.613192, 0.53403, -0.377982, 0.424424, 0.20625, 0.284516], [0.007684, -0.226317, -0.209705, -0.247826, 0.091687, -0.297589, 0.123201, -0.150796, -0.103658, 0.003585, -0.018093, -0.20889, 0.12149, -0.060777, 0.150437, 0.157238, 0.194798, 0.238355, 0.027451, 0.150682, 0.047499, -0.05378, -0.015377, -0.060086, -0.209113, 0.119945, 0.03336, 0.140552, -0.137027, -0.002792, 0.18552, -0.29712], [-0.396198, -0.418349, 0.291331, -0.550915, 0.006622, 0.15385, 0.271232, -0.276552, 0.133542, 0.079752, -0.254961, 0.809746, -0.211991, 0.152193, -0.219339, 0.479903, 0.087261, 0.486372, 0.052759, 0.202665, -0.394777, -0.389343, 0.087182, 0.338912, -0.088782, 0.074327, -0.138591, -0.364902, 0.314576, 0.024876, 0.170219, 0.307761], [0.039428, -0.00987, 0.332728, 0.166617, -0.254015, 0.034042, 0.077144, -0.127432, -0.076574, 0.36879, 0.005866, 0.185211, -0.213383, -0.178927, 0.080435, 0.319775, 0.090787, 0.065167, 0.138362, -0.105135, 0.017616, -0.028143, 0.012805, -0.038951, 0.185262, 0.136748, 0.013834, -0.064674, -0.052605, 0.230527, 0.224925, -0.105441], [0.402274, -0.262208, -0.16583, 0.49468, 0.10128, -0.272131, 0.139405, -0.2268, -0.441178, -0.135093, -0.319964, 0.029867, -0.243197, 0.119678, -0.190278, 0.36444, -0.246485, -0.036129, 0.19996, 0.291852, -0.007833, -0.112774, 0.341627, 0.028694, -0.328406, 0.192036, 0.041746, -0.233117, 0.066872, 0.029646, 0.023677, -0.198053], [0.074585, 0.030261, 0.157616, 0.298832, -0.2576, 0.32832, 0.090931, -0.738698, 0.194016, 0.058253, -0.121981, -0.063306, 0.169588, -0.250161, -0.136733, 0.23315, -0.944753, 0.081945, -0.084085, -0.192488, -0.13807, 0.284021, -0.041536, -0.183774, -0.253959, -0.039181, -0.103198, 0.238646, -0.023355, 0.231365, 0.232098, -0.00446], [-0.060683, -0.150633, 0.027196, 0.169085, -0.036897, -0.082394, 0.407254, 0.037667, -0.10878, 0.021899, -0.096281, 0.103112, 0.012528, -0.068274, -0.087173, -0.022402, -0.012038, -0.099187, 0.190007, -0.206697, 0.033999, -0.083887, 0.000678, -0.077058, -0.037138, -0.120371, -0.019295, -0.190849, -0.002822, 0.027189, -0.209315, 0.104372], [0.030729, 0.166314, -0.216045, -0.367623, -0.00047, -0.122185, -0.121759, -0.215555, -0.510999, -0.113201, 0.206843, 0.052052, 0.067433, -0.072882, -0.026467, 0.184279, 0.125202, 0.174866, 0.286099, 0.19616, -0.048308, 0.098496, 0.191978, -0.004602, 0.137661, -0.00317, -0.173563, -0.075052, 0.108685, -0.287491, 0.247948, 0.245852], [-0.027517, -0.510331, -0.066967, -0.018935, 0.411261, -0.038108, -0.188664, -0.024908, 0.020659, -0.057121, 0.639457, -0.097464, 0.092506, 0.054316, 0.03313, 0.208103, 0.036629, -0.012543, 0.061219, -0.112806, 0.265729, 0.040725, -0.213445, 0.032782, -0.250768, -0.030269, -0.001508, -0.055454, -0.083061, -0.087468, -0.114226, -0.075915], [-0.075738, -0.087884, -0.267778, -0.271692, 0.007744, 0.050386, -0.394602, -0.027124, 0.398011, -0.02924, 0.193743, 0.036269, -0.184257, -0.082228, 0.236603, -0.171171, -0.039957, 0.102112, 0.019638, -0.14232, 0.33987, -0.221442, -0.191033, 0.018203, -0.153197, -0.007551, -0.270205, -0.230056, 0.056547, -0.318216, 0.006627, -0.343441], [-0.02338, -0.179203, -0.199279, -0.146081, 0.093364, -0.214243, 0.296674, -0.110679, 0.097311, -0.079737, -0.013418, -0.313313, 0.04394, -0.137795, -0.397965, -0.051891, -0.030138, -0.14203, -0.019819, 0.183137, -0.055546, -0.117489, 0.118079, 0.050611, -0.165738, -0.45424, 0.027409, -0.172763, 0.149723, 0.106914, 0.005859, 0.512252], [0.013285, 0.187038, -0.323289, 0.340892, -0.04854, 0.092465, -0.057519, -0.09096, 0.057794, -0.331902, -0.058166, -0.146522, 0.117075, -0.063595, 0.230555, 0.539279, 0.025413, -0.033959, 0.038419, 0.0957, 0.001704, 0.177177, 0.245179, -0.13348, 0.056543, 0.317898, 0.317704, -0.054599, -0.144269, -0.08063, 0.100802, -0.112935], [0.085094, 0.117189, -0.044058, 0.144358, 0.094762, -0.020189, -0.155893, 0.335292, -0.166059, 0.537944, 0.480363, -0.535217, 0.260096, 0.01106, 0.221685, 0.073742, -0.126548, 0.074033, 0.173042, -0.213667, -0.470861, 0.01658, -0.374526, -0.202676, 0.054228, -0.197942, 0.252394, 0.149525, 0.189911, -0.235164, 0.344767, 0.116875], [0.06777, -0.083995, 0.010709, -0.086895, 0.104086, -0.110181, -0.070374, -0.175137, 0.043017, -0.057486, -0.081686, 0.138732, -0.126126, -0.142326, 0.201964, 0.080196, -0.000366, -0.015326, 0.199267, -0.032375, 0.15197, 0.235683, 0.162881, -0.198903, -0.179465, -0.105673, -0.236373, -0.019481, -0.117904, -0.050562, -0.226958, -0.015918], [-0.04407, -0.003762, 0.016822, -0.055661, -0.032376, 0.031306, -0.08586, 0.051245, 0.113684, -0.088912, -0.059255, -0.05701, -0.012638, -0.020848, 0.048854, -0.119568, 0.006647, -0.00607, -0.27274, -0.005314, -0.060516, -0.343457, 0.086831, 0.109503, 0.020141, 0.062097, -0.061173, -0.019622, 0.008031, -0.009406, 0.031838, -0.008353], [-0.216094, -0.197179, -0.073481, 0.009589, -0.057557, 0.016378, -0.069328, -0.077034, -0.330161, -0.102531, 0.033826, 0.007048, 0.15725, -0.170508, 0.192796, 0.05125, 0.190056, 0.155229, 0.156767, -0.117028, -0.027491, 0.061182, 0.063632, -0.243844, 0.187933, 0.336749, -0.110245, -0.043832, -0.01369, 0.336779, 0.176053, -0.068169], [0.318674, -0.050067, -0.008753, 0.053695, 0.075249, 0.108355, 0.162961, 0.010899, -0.061207, 0.196121, 0.078904, 0.135021, -0.066249, 0.138499, 0.084239, -0.082248, -0.015835, -0.344198, 0.173602, 0.046841, 0.002917, -0.003382, 0.072538, -0.094269, -0.039182, 0.269891, 0.225738, -0.071212, -0.105349, -0.370319, 0.185695, -0.471657], [-0.002932, 0.066387, -0.041822, 0.086864, 0.011142, 0.233413, 0.036945, -0.041676, 0.020573, -0.126182, -0.14659, 0.482459, -0.035635, 0.227627, -0.444662, 0.364811, 0.002751, -0.085482, 0.033149, -0.050927, -0.151747, 0.035573, -0.007325, -0.419548, 0.027875, 0.137221, 0.048172, 0.026268, -0.000641, 0.060473, 0.040172, 0.136357]], [-0.519849, -0.053506, 0.225777, -0.091865, -0.547026, 0.243778, 0.109952, -0.44742, -0.122928, -0.356762, 0.42867, -0.190008, 0.227092, 0.321637, 0.016598, -0.170458, 0.386311, -0.381055, -0.319496, -0.249621, -0.229018, -0.181973, -0.039327, -0.495972, 0.199274, -0.334865, -0.217146, 0.383266, -0.097141, 0.154346, -0.006999, -0.143528]), ([[-0.330032, -0.627623, -0.185326, 0.137084, 0.361323, 0.04057, 0.111318, -0.495112, 0.008722, 0.061047, 0.102264, -0.062857, -0.06975, 0.727375, -0.894368, -0.34435], [-0.125133, -0.051812, -0.270888, -0.055974, -0.246667, 0.100927, 0.075703, -0.013377, 0.004114, -0.207248, 0.18185, -0.100821, -0.152444, 0.109472, 0.08289, -0.357657], [0.186615, 0.164984, 0.075003, -0.267293, -0.122004, -0.029025, 0.38342, 0.186806, 0.188515, -0.023627, 0.214687, 0.136173, -0.043056, -0.195951, -0.11662, -0.454425], [-0.284728, -0.285744, -0.09756, 0.152483, -0.145516, 0.089909, -0.237624, -0.164556, 0.03352, 0.350062, -0.158687, 0.091876, -0.034235, -0.096379, 0.153786, -0.301289], [-0.117808, -0.214859, 0.012551, 0.214589, 0.019905, 0.146165, 0.003561, -0.113101, 0.744327, 0.343902, 0.041898, 0.014755, -0.190385, -0.409532, 0.248477, 0.377942], [0.175116, 0.110931, 0.230162, -0.037403, 0.003053, 0.279457, 0.117727, 0.210865, -0.002094, 0.159859, 0.274346, 0.211579, -0.069549, -0.053158, -0.075912, 0.242563], [0.115706, 0.093896, 0.193749, 0.094545, 0.133472, 0.160214, 0.248557, 0.157583, 0.269761, 0.264256, 0.221684, 0.03076, -0.193071, -0.255012, -0.003123, -0.07314], [-0.122389, -0.257439, -0.084701, -0.191018, -0.057647, 0.076646, 0.200343, -0.120088, 0.05897, 0.183138, 0.229573, 0.048622, -0.223089, 0.17378, 0.210462, 0.0962], [0.054335, 0.24019, -0.194194, -0.201966, -0.254661, -0.210107, 0.168013, 0.033348, -0.207634, 0.09243, -0.046294, 0.161811, -0.001442, -0.051042, -0.00402, -0.489585], [-0.098641, 0.371955, -0.251766, -0.062787, 0.111028, -0.318812, -0.076043, -0.13527, -0.490954, -0.116222, -0.149916, -0.052115, -0.152659, -0.174832, -0.125661, -0.087786], [-0.053219, 0.224432, -0.23967, 0.105254, -0.146041, -0.344539, 0.194001, 0.112581, -0.203663, -0.08874, -0.164805, 0.127466, -0.277544, -0.010443, -0.235144, -0.118515], [-0.126985, -0.212854, -0.167375, 0.276915, -0.064458, -0.107305, -0.433598, -0.008338, -0.354694, -0.05762, -0.235142, 0.092222, 0.122433, 0.484426, -0.204499, 0.281774], [0.06095, -0.522605, 0.056107, 0.106798, 0.195904, -0.074296, 0.060352, 0.215474, -0.112128, -0.100855, 0.138285, 0.074844, -0.001679, -0.063828, 0.272735, -0.013312], [-0.038766, 0.224278, -0.341981, -0.254293, 0.19021, -0.217476, 0.017296, 0.073451, 0.069228, 0.071577, 0.129268, 0.065111, 0.190186, -0.411841, -0.07406, -0.596123], [-0.152575, -0.058487, 0.177318, 0.059488, -0.062665, 0.315063, -0.193784, 0.061153, -0.177201, -0.174849, 0.012816, 0.232837, -0.07551, 0.135286, 0.230847, 0.091668], [-0.062297, 0.116977, 0.165176, 0.166629, -0.011704, 0.01519, 0.201984, 0.117456, -0.521012, 0.258333, 0.023458, 0.293517, -0.073301, -0.064972, 0.015227, 0.222046], [0.061268, -0.001929, -0.15604, 0.087473, -0.070226, 0.113776, 0.232241, 0.032839, -0.426441, -0.319593, -0.154162, -0.060071, 0.037317, 0.007774, 0.192398, -0.064646], [0.121, -0.031314, 0.097775, 0.046264, -0.090089, 0.158493, -0.218884, 0.113757, 0.065542, 0.441943, -0.671055, 0.067701, -0.013247, -0.05955, 0.231741, 0.203443], [0.04171, -0.113567, -0.004186, -0.014779, -0.076349, 0.265615, -0.167216, -0.105862, 0.070852, -0.104312, 0.063696, -0.445167, 0.1234, -0.019088, 0.095364, 0.20095], [0.11018, 0.052529, 0.094603, 0.214685, 0.307028, 0.078911, 0.059441, 0.153667, 0.126137, -0.138862, -0.190303, 0.073687, -0.185916, 0.094907, -0.099494, -0.053653], [-0.159142, -0.147753, 0.292183, 0.254441, 0.127327, -0.581165, 0.114539, -0.170669, 0.113914, 0.20793, -0.21542, 0.087296, -0.033749, -0.108534, -0.008609, -0.099709], [-0.036126, 0.131129, -0.214856, -0.171887, 0.0222, -0.256365, 0.04702, -0.242977, -0.016761, 0.123407, 0.246303, 0.158398, 0.067623, -0.436531, -0.094876, 0.339381], [0.013292, 0.105835, 0.073791, 0.101314, 0.151368, 0.120655, -0.072833, 0.072685, -0.246988, 0.054898, 0.272558, 0.045692, -0.113444, -0.081724, -0.06808, 0.028728], [-0.262485, 0.044093, -0.273823, -0.109777, -0.265722, -0.346301, -0.048629, -0.224088, 0.061793, 0.274167, -0.019294, -0.142728, -0.128085, -0.194975, -0.010077, -0.753269], [0.0353, 0.019638, 0.154883, -0.136386, 0.074817, 0.178335, 0.024113, 0.070591, 0.29221, -0.179478, 0.136452, 0.090624, -0.049704, -0.51702, 0.071611, 0.46086], [-0.02997, 0.288051, -0.312209, -0.270715, -0.121789, -0.060369, -0.744571, -0.294862, -0.272775, 0.270384, -0.066938, 0.248871, -0.312717, 0.068922, -0.119628, -0.118083], [-0.44844, 0.35433, -0.097958, 0.067082, -0.097877, -0.468057, 0.046896, -0.291607, -0.166498, -0.399672, -0.092591, 0.000475, -0.013141, -0.13335, -0.09293, -0.085278], [0.326361, -0.172453, -0.07955, 0.054581, 0.194762, 0.18883, -0.131097, 0.057985, 0.021017, -0.282014, -0.117818, -0.275654, -0.211595, 0.36109, 0.043323, 0.18156], [0.196383, -0.129852, 0.283134, 0.156512, 0.032442, 0.298593, 0.390985, 0.115384, 0.521905, 0.222369, 0.155238, 0.096593, 0.133413, 0.010527, 0.259433, 0.170109], [-0.104819, -0.514852, 0.058264, 0.29331, 0.149505, -0.155463, -0.354129, 0.060835, -0.012622, 0.205186, -0.341102, -0.215734, -0.131521, -0.057235, -0.124215, -0.240629], [-0.021703, 0.040749, 0.234601, 0.003003, -0.048173, 0.100386, -0.017204, 0.031148, -0.58833, -0.337915, 0.136361, -0.170949, 0.07669, 0.282742, -0.031342, -0.145395], [0.256468, 0.085447, -0.206715, -0.26677, -0.089843, -0.122665, 0.013355, 0.182653, -0.038935, -0.048304, -0.52355, -0.007001, -0.246102, 0.238058, -0.384404, -0.165138]], [0.253551, 0.142521, 0.169051, 0.139404, -0.022549, -0.321979, 0.146526, 0.295521, 0.159385, -0.438576, 0.194482, 0.343685, -0.219832, -0.271364, 0.304603, -0.177172]), ([[0.550099], [1.501285], [0.727039], [1.010358], [0.724405], [1.522565], [-0.672536], [0.927767], [-1.633072], [-1.982256], [-0.985318], [-0.660451], [0.793398], [3.081697], [1.022183], [1.764686]], [-0.904828])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.75191, -9.2e-05, 9.2e-05, 0.0, 0.052846, 0.052846, -0.0, 0.0, 0.0, 1.243668, 5.879185, 5.879185, 0.0]
SEQ_TSD = [1.079295, 0.132411, 0.132411, 0.099558, 1.45424, 1.45424, 0.666687, 0.219428, 0.17921, 0.60273, 1.666237, 1.666237, 0.649942]
SEQ_XM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.257423, 0.0, 0.0, 5.846561, 5.846561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987482]
SEQ_XS = [1.124955, 0.99101, 0.94523, 0.845186, 0.757041, 2.577918, 18.130564, 0.790721, 15.945622, 0.69697, 0.6513, 0.540497, 1.677766, 1.677766, 0.999993, 4.402977, 1.087599, 0.310604, 132.411407, 334.802429, 1066.728027, 1.341266]
SEQ_YSC = 1.939010739326477
SEQ_WI = [[-0.758383, -0.669594, 0.384014, -0.136853, -0.113843, -0.12439, 0.231447, -0.532042, 0.00227, 0.268303, -0.483038, -0.040032, -0.274465, 0.074687, 0.479293, 0.194889, -0.006673, 0.223532, -0.302111, -0.42198, -0.33636, -0.056236, -0.466671, -0.227084, -0.206486, 0.117503, -0.195636, 0.222658, -0.247971, 0.119527, 0.349652, -0.212184, 0.191825, -0.170331, -0.043397, 0.45139, 0.269742, -0.12462, 0.386072, -0.01967, 0.621079, -0.231145, 0.083621, 0.110642, -0.111072, -0.072252, 0.289322, 0.030094], [0.222009, -0.201261, -0.508712, 0.280412, -0.114877, 0.130905, 0.261768, 0.394272, 0.14902, -0.154841, 0.054618, -0.210901, 0.21141, 0.428908, 0.098853, 0.350556, -0.222312, -0.028778, -0.69019, -0.055226, -0.173728, 0.060209, 0.591509, -0.044138, -0.133052, -0.789058, 0.24267, -0.021424, 0.243781, 0.605264, -0.391519, 0.234403, -0.066986, -0.000558, 0.336579, 0.214986, 0.009855, -0.067061, 0.499249, 0.096635, -0.075265, -0.495171, 0.124182, 0.026801, 0.111075, -0.036908, 0.279778, 0.304904], [-0.320554, 0.279068, -0.144728, -0.008654, 0.129433, -0.304505, -0.607921, 0.186838, 0.289534, 0.170573, -0.010263, -0.000459, 0.292226, -0.133863, -0.500409, 0.081383, 0.349204, 0.248036, 0.061694, 0.02233, 0.231298, -0.076666, -0.183974, -0.505609, -0.349907, 0.065316, -0.383878, 0.033981, 0.832349, 0.12151, -0.063702, -0.287593, -0.040007, 0.122599, -0.654081, -0.190011, -0.043424, 0.378602, -0.248267, 0.036537, 0.20742, -0.020081, -0.101302, -0.383767, 0.32533, 0.332519, -0.011959, -0.179758], [-0.315938, 0.100905, -0.162087, 0.166655, -0.228723, 0.121267, 0.392168, -0.189764, 0.163483, 0.193049, -0.124567, 0.220836, 0.398717, 0.389477, -0.133846, -0.139171, -0.018231, -0.049828, -0.699662, -0.369604, 0.417287, 0.411132, 0.223188, -0.180864, 0.16774, -0.250546, -0.272246, 0.031166, 0.150382, 0.535344, -0.376197, 0.110873, 0.140488, 0.327028, 0.186528, -0.094096, 0.070427, 0.260842, 0.067294, -0.000656, -0.323476, -0.027535, 0.483269, -0.077969, 0.459027, 0.291361, -0.012122, 0.102299], [0.087376, -0.083593, -0.255019, 0.415066, 0.225017, 0.093537, -0.214805, -0.304748, -0.200433, -1.024138, 0.309071, -0.094216, -0.210502, 0.224316, -0.377562, 0.220744, 0.018374, 0.581203, 0.183826, -0.436006, 0.284819, 0.083754, -0.160467, -0.275256, 0.176047, -1.461676, 0.52172, -0.005052, 0.469969, -0.684956, -0.375537, 0.289853, 0.279075, 0.014013, 0.003671, -0.053322, 0.526784, 0.191528, 0.177285, 0.537096, 0.054163, -1.267624, -0.76135, 0.24713, -0.154193, 0.082322, -0.134051, 0.237344], [0.21109, -0.332664, -0.17327, -0.050025, -0.182033, 0.024962, -0.211771, -0.591734, 0.334511, -0.501225, -0.359599, -0.315734, -0.945123, -0.120604, 0.089593, 0.325167, -0.253084, -1.409704, 0.35733, 0.622567, -0.11662, -0.087483, 0.493932, 0.374229, -0.171766, -0.025805, 0.065952, 0.042948, -0.254841, 0.481663, 0.088083, -0.126541, -0.181176, -1.103872, 0.220141, -0.087305, -0.330124, -0.207713, 0.511611, 0.114258, -0.236537, -0.002885, -0.12438, -0.281077, -0.790927, -0.256289, 0.347802, 0.56437], [0.045874, -0.178586, -0.514624, -0.216339, -0.119629, 0.419934, 0.253584, 0.219804, -0.635617, 0.357103, -0.066154, 0.168236, -0.241214, -0.122837, -0.053645, 0.455213, 0.132535, -0.190376, 0.189383, -0.161868, -0.097265, -0.026696, -0.042879, -0.068631, 0.364365, -0.398078, -0.054888, 0.360417, -0.262993, 0.481312, 0.288439, -0.199466, -0.044483, -0.100583, -0.328362, -0.289724, 0.283631, 0.20339, -0.179121, 0.03268, 0.086232, -0.19736, 0.195402, -0.021023, -0.129025, -0.330346, -0.115624, 0.288023], [0.164945, -0.052258, -0.616811, 0.152482, -0.120766, 0.598971, -0.063299, 0.008797, -0.274005, 0.108381, -0.176252, -0.174653, -0.45962, -0.133968, 0.196165, 0.007486, 0.074454, -0.283858, 0.116335, -0.027416, 0.076016, 0.001108, -0.221277, -0.004962, 0.096834, 0.085671, 0.182932, -0.10166, 0.142552, 0.463639, -0.052617, -0.261864, 0.112046, 0.040111, -0.260068, 0.114617, -0.202948, -0.129299, 0.279121, 0.17911, -0.203653, -0.099873, -0.036328, -0.084764, 0.030464, -0.130367, 0.102066, 0.352902], [-0.016545, 0.287706, -0.403667, 0.147227, 0.052604, 0.559522, -0.153638, 0.035904, -0.621752, 0.106105, -0.216273, -0.0839, 0.428775, -0.106607, 0.240279, 0.047137, -0.340325, -0.156681, 0.203469, -0.170911, 0.225111, -0.271429, -0.320236, 0.145695, 0.240369, -0.074988, 0.30786, 0.290327, -0.290023, 0.90283, -0.087233, -0.201926, 0.268817, -0.189824, 0.150013, -0.282743, -0.038129, -0.00481, 0.12134, 0.101876, -0.028173, 0.156647, 0.040773, -0.341682, -0.087704, 0.527629, 0.244513, 0.099694], [0.071985, -0.601674, 0.049286, 0.087286, 0.183343, 0.592952, 0.262113, 0.007415, 0.622983, -0.124899, -0.029366, 0.212992, 0.057747, -0.145029, 0.197989, -0.253812, 0.179489, -0.249737, 0.028684, -0.416981, 0.275062, -0.417491, -0.041705, -0.684119, 0.114388, -0.119084, 0.098417, -0.312138, 0.046998, 0.050089, 0.419581, 0.420646, -0.670824, 0.232231, 0.108214, 0.139425, 0.200423, -0.023, 0.230302, -0.144807, -0.165337, -0.076937, -0.216138, 0.276287, -0.191883, 0.28867, -0.154915, -0.246622], [-0.158553, 0.171246, 0.396497, -0.213383, -0.147448, 0.10294, -0.151604, 0.174948, 0.195265, 0.352571, 0.448429, 0.388214, -0.064557, -0.135524, -0.321089, 0.127061, 0.274632, 0.386834, -0.011983, -0.149606, -0.322181, 0.259712, 0.242619, -0.167623, -0.58795, -0.401085, -0.26448, -0.258073, -0.324623, -0.316418, 0.374797, -0.418852, 0.002615, -0.111361, 0.308581, 0.09767, -0.022953, 0.217426, 0.125763, -0.009784, 0.1355, -0.124207, -0.060103, 0.226256, -0.133866, -0.153453, -0.027118, 0.24129], [-0.096611, -0.073598, 0.456229, -0.104783, 0.043612, 0.218696, -0.153401, 0.273693, 0.240336, 0.045945, 0.109192, 0.170224, 0.337169, 0.330953, -0.075312, -0.376305, -0.010695, 0.215785, -0.2925, -0.154312, -0.505122, -0.346233, -0.577279, 0.017259, -0.55751, -0.133701, 0.451091, -0.340968, -0.33229, -0.182499, -0.41343, 0.068749, -0.176467, -0.23127, 0.197301, -0.126324, 0.096071, 0.389313, -0.205919, 0.078652, 0.110663, -0.005398, 0.031938, 0.198208, 0.074753, -0.139493, -0.112974, 0.101448], [-0.176604, 0.110095, 0.379961, 0.340962, 0.036834, -0.530046, 0.010607, -0.331863, -0.303273, -0.17302, -0.015618, -0.423805, -0.091074, -0.073549, -0.258938, 0.370742, -0.074752, 0.343308, 0.109639, -0.111339, -0.417119, 0.701596, 0.453809, -0.541687, 0.598339, 0.089924, -0.240057, 0.096386, 0.476717, -0.362726, 0.480727, -0.588795, -0.116853, -0.188653, -0.048148, -0.185406, -0.085271, -0.139518, 0.042243, -0.009035, -0.089326, 0.041371, 0.184718, 0.013456, 0.202918, -0.011681, -0.071627, -0.308862]]
SEQ_BI = [0.234388, 0.424884, 0.126386, 0.115404, 0.207704, 0.31465, 0.250506, 0.252953, 0.008736, -0.191765, 0.010645, 0.142858, 0.074741, 0.197082, 0.332559, -0.079156, 0.676844, 0.494563, 0.646194, 0.421912, 0.790057, 1.137895, 0.874153, 0.94892, 0.939168, 0.417856, 0.465016, 0.403216, 0.843916, 0.487601, 0.111463, 0.340067, -0.336062, 0.117394, 0.065922, 0.147891, -0.190105, -0.327775, -0.194467, 0.103262, 0.235547, 0.040684, -0.160942, -0.176679, 0.086381, -0.185286, -0.292769, 0.268785]
SEQ_WH = [[0.076663, 0.562607, -0.567868, 0.083438, 0.543839, -0.431475, -0.097708, -0.005034, 0.069067, -0.137764, 0.302424, 0.290451, 0.4241, 0.29688, 0.524432, 0.49119, 0.014642, -0.336288, 0.275715, -0.254531, 0.222531, -0.305199, 0.12972, 0.212806, -0.569539, -0.112886, -0.680737, 0.283803, 0.012703, -0.322379, -0.420343, -0.779203, 0.584939, -0.222057, -0.007499, -0.518556, 0.102246, 0.093462, -0.506566, -0.04266, 0.184127, -0.275158, 0.386554, 0.101352, 0.076714, -0.411109, 0.294767, -0.225899], [0.167055, 0.010118, 0.228472, 0.464978, 0.519546, 0.164672, -0.260929, -0.173333, -0.344946, 0.12297, 0.34936, 0.225775, 0.314812, -0.545514, -0.074347, -0.470026, 0.507089, -0.32672, 0.601512, 0.633946, 0.09892, 0.557741, 0.317209, -0.397077, -0.078742, -0.232863, 0.584933, 0.470295, -0.33575, -0.056227, 0.054994, 0.132126, -0.168729, 0.554699, 0.04547, -0.437072, 0.357097, 0.248166, -0.331792, -0.335001, -0.473768, 0.254253, -0.014498, -0.26433, -0.00472, -0.577977, -0.025377, -0.271451], [0.372919, 0.425754, -0.427619, 0.692408, 0.568372, 0.950204, -0.115677, 0.546107, 0.193667, 0.089906, 0.165135, 0.300824, -0.368809, -0.219957, 0.34112, 0.145511, -0.400612, 0.659426, 0.676594, 0.681797, -0.232606, 0.452549, -0.243414, 0.895621, 0.828077, 0.15201, 1.045861, 0.699202, 0.127538, -0.180585, -0.281527, 1.07332, 0.095196, 0.65838, 0.576417, -0.150724, 0.683266, -0.261832, 0.322025, 0.019802, -0.033694, 0.342668, -0.133073, 0.246636, -0.078495, 0.076605, -0.342404, -0.012941], [-0.646727, -0.476003, 0.289788, -0.302431, -0.762081, 0.1036, -0.124946, -0.554172, -0.142133, -0.446602, -0.690329, -0.677649, -0.386753, -0.239709, 0.07225, -0.372153, 0.11699, 0.43768, -0.208864, -0.200783, 0.372211, -0.45753, 0.505239, -0.080779, 0.055512, 0.225027, -0.319076, 0.132514, -0.43157, 0.085079, 0.124874, -0.210192, -0.348185, -0.448513, 0.125812, 0.639184, 0.219767, -0.188614, 0.217445, 0.506673, 0.225877, 0.062146, -0.559885, 0.290233, -0.689265, -0.1052, 0.263908, 0.061205], [-0.352008, -0.191078, -0.219404, -0.000609, 0.3066, 0.279533, -0.125957, 0.057333, -0.428077, 0.143466, -0.094913, 0.158377, 0.415512, -0.103842, -0.196125, -0.062976, 0.086112, 0.30105, -0.182157, -0.022779, -0.079204, 0.011163, -0.348623, -0.9133, -0.646476, -0.701708, 0.01628, -0.47163, -0.480354, -0.282127, 0.114697, -0.278235, -0.181116, 0.010963, -0.370755, 0.056257, 0.190125, 0.74357, 0.083705, -0.065809, -0.063371, 0.100018, -0.347697, -0.284489, 0.279469, -0.213364, -0.228353, 0.092557], [-0.071271, 0.244151, 0.161218, -0.19265, 0.126763, 0.072989, 0.151652, 0.156732, 0.214584, 0.230482, -0.05352, 0.165927, 0.233653, -0.167005, -0.159564, 0.04648, -0.384307, -0.395137, -0.562647, -0.189597, -1.358837, 0.246159, -0.581012, -0.870522, -1.177618, -0.386846, -0.646474, -0.135368, -0.722382, 0.090458, -0.665504, -1.017896, 0.257077, 0.037701, -0.30123, -0.324434, 0.022594, 0.735319, -0.049032, -0.39161, 0.145544, -0.138523, -0.115798, -0.097724, 0.122322, 0.221302, 0.097664, -0.221389], [0.177416, -0.019879, 0.629078, 0.388223, -0.063339, 0.125226, 0.671288, -0.059657, -0.086004, 0.237265, -0.418257, 0.500378, -0.067345, -0.914679, -0.667538, 0.308951, -0.270888, -0.077455, -0.177408, -0.736469, -0.814662, -0.15483, -0.461453, -0.451888, 0.239888, -0.59656, 0.730261, 0.514975, 0.057352, -0.241201, 0.1319, 0.083337, -0.461202, -0.057591, -0.082892, -0.124581, 0.154995, -0.123653, 0.666304, 0.186145, -0.729818, 0.073064, -0.136381, 0.066666, -0.057604, -0.213076, -0.107477, -0.04735], [-0.465297, 0.204482, -0.283977, -0.591583, -0.156732, -0.217429, -0.675268, -0.109264, -0.034173, -0.298356, -0.651506, -0.273494, -0.376204, 0.098821, 0.08942, -0.18268, 0.209534, 0.658552, -0.703393, 0.05054, 0.301393, -0.360731, 0.408764, -0.04034, 0.313036, 0.78995, -0.013612, 0.406153, -0.356569, -0.049476, 0.851945, -0.354519, -0.063469, 0.035008, 0.144778, 0.558059, 0.152546, -0.300037, -0.001348, 0.608207, 0.542235, -0.06615, 0.539545, -0.077181, -0.089325, 0.33907, -0.13067, -0.181457], [-0.169546, -0.161355, -0.224583, -0.345307, -0.263311, -0.086518, 0.168639, 0.209202, 0.026786, 0.184982, -0.480165, -0.363033, 0.037037, 0.229787, 0.231232, -0.619002, 0.62436, 0.473026, 0.207195, 0.418377, 0.633107, 0.844272, 0.426365, -0.01543, 0.618525, 0.244032, -0.152194, -0.332583, 0.11556, 0.096536, -0.418231, -0.07426, 0.13575, -0.437458, -0.101155, 0.263416, -0.0015, 0.025271, -0.226135, 0.279953, -0.028937, -0.063455, 0.241097, -0.027952, 0.24123, 0.431267, -0.173417, 0.062599], [0.04774, -0.392927, 0.446157, 0.471134, -0.728409, -0.274827, 0.075173, 0.224331, -0.07498, 0.052055, -0.091287, -0.214822, 0.22207, -0.574108, -0.500776, -0.31867, 0.705699, -0.373458, -0.605525, 0.355985, -0.610898, 0.170467, -0.146957, -0.203466, 0.2826, -0.710459, -0.371797, 0.240395, 0.104373, -0.106404, 0.055176, -0.36849, -0.348752, 0.360846, 0.013645, -0.14379, -0.514994, 0.219568, 0.084375, -0.58973, -0.165213, 0.422705, 0.215625, 0.461132, -0.22834, -0.02821, 0.609663, 0.107645], [0.434697, 0.263356, 0.096213, 0.503097, 0.662963, -0.186329, 0.048015, -0.074943, 0.119332, 0.452593, -0.122392, 0.037045, -0.085597, 0.291544, 0.480256, 0.28681, 0.285481, 0.013897, 0.094371, -0.504985, -0.126674, -0.076424, -0.28712, 0.653688, 0.805521, -0.298568, 0.421751, 0.935544, 0.621689, -0.353256, 0.036058, 0.397185, 0.509109, -0.059753, 0.317297, -0.284379, -0.754999, -0.687809, 0.270192, 0.245684, -0.124109, 0.237098, 0.609679, 0.110556, 0.14316, -0.051483, 0.0669, -0.162977], [-0.248101, -0.185309, 0.531558, 0.043814, -0.000158, -0.051388, 0.365743, -0.19602, 0.525737, -0.043321, -0.06573, -0.15272, -0.201576, -0.2765, -0.117189, 0.08591, -0.611889, -0.311912, -0.000406, -0.624853, -0.234047, -0.880869, 0.288261, 0.071691, 0.246436, -0.462523, 0.100998, 0.122575, -0.49136, -0.37897, -0.101672, 0.553063, -0.053285, 0.143982, 0.212074, -0.333791, -0.015505, 0.235171, -0.038035, 0.111909, -0.442449, 0.637948, -0.271293, 0.38759, -0.018932, 0.322478, 0.411179, 0.021151], [-0.067016, 0.339041, -0.125147, -0.021043, 0.055053, -0.128927, -0.404272, 0.072004, -0.160033, -0.373554, 0.111349, 0.381207, 0.379493, 0.384892, 0.231388, 0.234668, 0.536853, 0.201405, 0.343106, 0.327131, 0.265263, 0.919333, -0.348487, -0.269503, 0.189165, -0.435651, 0.511773, -0.02987, 0.109051, -0.336842, -0.537815, -0.281774, 0.17297, 0.098455, -0.245005, 0.101729, 0.134208, 0.234976, -0.551654, -0.266534, 0.553777, -0.35405, -0.246908, -0.57792, 0.215938, -0.225761, -0.435115, -0.512172], [-0.702299, -0.353782, 0.632528, 0.209322, -0.073402, -0.256666, -0.515669, -0.185905, -0.050455, 0.381926, -1.36796, -0.562721, -0.534846, 0.104672, 0.165212, 0.50128, 0.07633, 0.267502, -0.532546, -0.945134, -0.244237, 0.440208, -0.400566, -1.245615, 0.734968, 0.268055, 0.850692, 0.315935, 0.110538, -0.213838, 0.108282, 0.578792, -0.231401, -0.313947, -0.366388, -0.295072, -0.500707, -0.218392, -0.178844, 0.273422, 0.52546, -0.078975, -0.169862, 0.153965, 0.141538, 0.313676, -0.334644, 0.02171], [-0.197514, -0.268871, -0.259276, -0.110322, 0.196089, 0.049147, 0.010476, -0.586045, 0.002371, 0.108166, 0.428262, 0.067731, 0.237522, -0.206328, 0.274944, -0.006462, -0.728461, -0.596087, 0.013565, -0.370311, -0.2622, -0.315154, -0.109934, -0.424425, -1.08423, -0.27847, -1.132552, 0.597627, -0.292506, -0.17384, -0.123686, -0.608466, 0.345882, 0.101765, 0.425654, 0.120171, -0.186973, 0.064746, -0.178802, 0.103965, -0.172538, -0.341976, 0.523923, 0.382011, -0.240519, -0.055148, 0.369633, 0.045984], [0.09975, 0.437804, -0.115722, -0.59905, -0.183979, 0.650377, -0.27303, 0.409596, 0.786698, 0.380379, 0.329116, 0.672282, -0.006348, 0.41127, 0.442613, -0.018476, 0.257835, 0.951318, -0.749462, -0.133529, 0.508616, 0.599568, 0.957389, 0.850083, -0.084233, 0.644638, -0.089606, 0.093579, 0.875352, 0.705226, -0.187065, -0.13829, -0.28251, 0.204602, 0.135167, -0.001496, -0.593251, 0.037465, -0.308304, -0.234442, -0.132784, -0.078455, 0.294893, -0.085507, 0.206104, 0.128536, 0.044707, 1.004879]]
SEQ_BH = [-0.026007, 0.162378, -0.136817, 0.058543, 0.091081, 0.493878, 0.024771, 0.264716, -0.013981, -0.26733, 0.413594, 0.250081, 0.186497, 0.093822, 0.29361, -0.2083, 0.527047, 0.580943, 0.506846, 0.583118, 0.502595, 0.891623, 0.586386, 0.704211, 0.751459, 0.455571, 0.755877, 0.491667, 1.075477, 0.599325, 0.23417, 0.504904, 0.132323, 0.320381, -0.234352, -0.190038, -0.251259, -0.223698, 0.245777, 0.17933, 0.311928, 0.30015, -0.039588, -0.274368, 0.299782, 0.246654, 0.044241, -0.093054]
SEQ_HEAD = [
    ([[0.078842, -0.097704, 0.102601, -0.068301, 0.205831, -0.132246, 0.130506, -0.128827, 0.089883, -0.248854, 0.184791, 0.026948, -0.048114, 0.018277, -0.205365, -0.08362, 0.039481, -0.07009, 0.16155, 0.125755, 0.015693, -0.335046, -0.122363, 0.345785, 0.00082, -0.1973, 0.059447, -0.170775, -0.109686, -0.035438, -0.061597, -0.03735, -0.144464, -0.196429, -0.031583, -0.008074, -0.133642, -0.034483, 0.03811, 0.124749, 0.20381, 0.149805, 0.001371, -0.041887, 0.056955, 0.02139, 0.041717, -0.498684, -0.101185, -0.137778, 0.321685, -0.0225, -0.161837, 0.047929, -0.068722, 0.229511, -0.264566, -0.134574, 0.089956, 0.198469, 0.157893, -0.077304, -0.004665, 0.086867], [0.01198, -0.078719, -0.029188, -0.107123, -0.133861, -0.009072, 0.109701, -0.061443, 0.06649, -0.19097, 0.127448, -0.008012, 0.073482, 0.021337, -0.116034, 0.029208, 0.061848, 0.076443, 0.079189, -0.102448, -0.077501, -0.141871, 0.027497, -0.05853, -0.064951, -0.035972, -0.083109, -0.02849, -0.117559, -0.010875, 0.021322, -0.058059, -0.017308, -0.204926, 0.137337, -0.0891, -0.204382, 0.143398, 0.013626, 0.02961, 0.009441, 0.046418, 0.062478, -0.129109, -0.014382, -0.113479, 0.11058, -0.176404, 0.10827, 0.085357, 0.241075, 0.160122, -0.104716, -0.117115, 0.261343, 0.01453, 0.070646, -0.104063, -0.021816, 0.160882, -0.10283, 0.101087, 0.000849, -0.058493], [0.119564, -0.086696, 0.006904, -0.045678, -0.131985, -0.089506, 0.199732, -0.037196, 0.063747, -0.147187, 0.173602, -0.133681, 0.05145, 0.096214, -0.000321, 0.003731, -0.109524, 0.129595, 0.039354, 0.052064, 0.022818, -0.181841, 0.061443, -0.093054, -0.098362, 0.056216, -0.132805, 0.065576, 0.008364, -0.106179, -0.051388, -0.014591, 0.008919, 0.024911, -0.060487, -0.162566, 0.027923, 0.043144, 0.001525, -0.225577, 0.13445, -0.124133, 0.017892, 0.096732, -0.043134, -0.075962, 0.079122, -0.049113, 0.00858, -0.072867, -0.046162, 0.026094, -0.089643, -0.176335, 0.076787, -0.211635, 0.107042, 0.131621, -0.14062, -0.040482, -0.113101, 0.086198, -0.058895, -0.176814], [0.074146, -0.175542, -0.209121, -0.022286, -0.208474, 0.132365, -0.067076, 0.096411, -0.003804, -0.085396, -0.033501, 0.172891, 0.125913, 0.076952, -0.065636, -0.026999, 0.153409, -0.200066, 0.139276, -0.190012, 0.17236, 0.160904, 0.149841, -0.038273, 0.217103, 0.111767, 0.082994, -0.097431, 0.039397, 0.030667, -0.105991, 0.058337, 0.213606, -0.034031, -0.002004, 0.046114, 0.251649, -0.032692, -0.065965, -0.087487, -0.048991, -0.267896, -0.033482, -0.154042, 0.060525, -0.066432, -0.041925, 0.140275, -0.049611, 0.051834, -0.124468, -0.143543, -0.153528, 0.046107, 0.027701, 0.062297, 0.323401, -0.001141, 0.149491, -0.070431, -0.150215, 0.176008, 0.152357, 0.063346], [0.119011, 0.112622, -0.372026, 0.083983, 0.267908, -0.210062, 0.180467, -0.028002, 0.215971, -0.012042, -0.049158, 0.161891, 0.155101, 0.303812, 0.0194, -0.171033, -0.229917, 0.083175, 0.116299, 0.260542, 0.223803, 0.24469, 0.054481, -0.100633, -0.103246, 0.009765, -0.16608, -0.014146, -0.200516, -0.23614, -0.364661, -0.108296, 0.212492, -0.011727, -0.079573, -0.093326, 0.208811, 0.32277, 0.060321, -0.088018, -0.186205, -0.234428, 0.158202, -0.071483, -0.203712, -0.137324, 0.028528, 0.102079, 0.038815, 0.00932, -0.272483, -0.324036, -0.123388, 0.101585, -0.270243, -0.14011, 0.155384, 0.12277, -0.065977, -0.151224, -0.306245, 0.106148, 0.039776, 0.242892], [-0.186986, -0.146739, 0.10728, -0.056885, -0.078635, -0.076083, -0.060816, 0.129455, 0.032189, -0.113709, 0.1024, -0.077911, -0.110262, -0.175501, 0.021873, 0.035417, 0.027162, 0.032874, -0.004201, 0.011574, 0.142825, -0.117876, 0.032011, -0.072131, 0.03096, -0.001441, -0.001293, -0.024577, -0.12724, 0.121422, 0.027666, -0.034572, 0.179235, -0.052979, -0.093412, 0.060076, 0.118915, 0.067057, 0.162692, -0.157195, 0.091511, -0.012705, 0.095966, -0.055944, 0.056257, -0.15219, -0.111782, 0.166048, 0.083305, 0.101896, -0.030042, 0.192824, 0.050837, 0.089238, 0.231238, -0.182813, 0.116831, 0.094219, -0.036382, -0.026389, -0.113763, 0.037065, 0.028669, -0.05021], [-0.03363, 0.041927, -0.333856, -0.208434, -0.029973, -0.055408, -0.064248, 0.18694, -0.00111, 0.057701, -0.024366, 0.307431, -0.170154, 0.233806, 0.160983, -0.033885, 0.191233, -0.19975, 0.06267, 0.178307, -0.034023, 0.213138, 0.094126, 0.349476, 0.016811, -0.221054, -0.050943, -0.088193, -0.14021, 0.027316, -0.03242, 0.002053, 0.076036, -0.006216, 0.097726, -0.044197, -0.303131, 0.079959, 0.089047, 0.035871, -0.048218, -0.020601, -0.006559, 0.08378, 0.007554, -0.169796, 0.02816, -0.187364, 0.113422, 0.008491, 0.233472, 0.099526, -0.163617, -0.133469, 0.041596, -0.162886, -0.127365, -0.150158, -0.030393, 0.134992, 0.04525, 0.065081, 0.020445, 0.178981], [0.063923, -0.034848, -0.055447, 0.107067, 0.083979, -0.076395, 0.027407, 0.1129, -0.010857, 0.126463, -0.103416, -0.107803, -0.01526, 0.133428, -0.118426, 0.140765, 0.022138, 0.130093, -0.072463, -0.010315, -0.099189, 0.13272, -0.115337, -0.029936, -0.196528, 0.080763, -0.038702, 0.115401, 0.027576, 0.178113, 0.176432, 0.019195, -0.146337, 0.098598, 0.379803, -0.088431, 0.002745, -0.019828, 0.002097, -0.091481, -0.004455, 0.243293, 0.085069, 0.037114, -0.05889, -0.008441, 0.022027, 0.126922, -0.339048, -0.000604, -0.08248, 0.180355, -0.018602, -0.072756, 0.085506, -0.225201, -0.026004, -0.018656, -0.050365, -0.0855, 0.074041, 0.01179, -0.039288, -0.008221], [-0.142767, -0.103152, 0.172783, 0.030638, -0.11856, -0.03295, 0.127731, 0.006738, 0.071521, -0.123589, 0.014702, 0.280639, 0.079762, 0.060985, -0.123495, -0.300475, 0.119697, -0.175683, 0.151132, -0.087109, 0.091093, -0.019387, -0.218312, 0.054309, -0.0937, -0.096486, -0.221734, 0.039531, -0.26026, 0.039964, -0.032476, -0.285917, -0.333058, -0.049496, -0.027578, -0.101206, 0.144983, 0.11514, 0.036942, 0.068618, 0.171591, 0.037324, 0.132441, -0.082011, -0.252408, -0.095675, -0.057708, -0.156034, 0.049446, 0.013168, 0.148032, 0.155658, -0.05993, -0.250761, -0.03555, -0.003497, 0.087362, -0.150571, -0.04708, 0.272881, -0.18181, -0.030641, 0.110257, 0.079915], [-0.257877, -0.112712, 0.014466, 0.143539, -0.057102, 0.086203, -0.009084, 0.017763, -0.127204, 0.198115, -0.144766, 0.096618, -0.058092, -0.260512, 0.073507, -0.168137, 0.189378, -0.266978, -0.125684, 0.024915, 0.118817, -0.049018, -0.390914, 0.066482, -0.025456, -0.101176, -0.188106, -0.120062, -0.230271, 0.114064, 0.126499, 0.14054, -0.09087, 0.018386, 0.107669, -0.176877, -0.073362, 0.13454, 0.044279, 0.018448, -0.038122, -0.187096, -0.096056, 0.056538, 0.104641, -0.134551, -0.182786, 0.029961, -0.164314, -0.067903, 0.016209, 0.065573, -0.201128, 0.097849, 0.076911, 0.003315, -0.126775, 0.06404, -0.077153, 0.12439, -0.125047, -0.033434, 0.165222, 0.095469], [0.166966, -0.040297, -0.055869, -0.211807, 0.076656, -0.000608, 0.053425, 0.058738, -0.145688, -0.131974, -0.158769, 0.129284, -0.138634, 0.097219, -0.135777, -0.179751, 0.015015, -0.080525, 0.02627, 0.026815, -0.129995, -0.011998, 0.005136, 0.207349, -0.04875, -0.081915, 0.11199, 0.103289, 0.089853, 0.323565, 0.027669, 0.063767, -0.212525, 0.017469, -0.003824, -0.077469, -0.075721, -0.051067, 0.082142, 0.065657, 0.043856, 0.029439, -0.066518, -0.004666, -0.012804, 0.045329, -0.175837, 0.016212, 0.130959, -0.050591, -0.127252, -0.043436, -0.041889, -0.042045, 0.099208, 0.06847, 0.001109, -0.023488, -0.044998, 0.051337, 0.092918, -0.000542, 0.084642, -0.104313], [0.275354, -0.037789, 0.010573, 0.25883, -0.104326, 0.117006, 0.160095, -0.20238, 0.053525, -0.30699, 0.053284, 0.279776, 0.018809, 0.009766, 0.117604, -0.314059, -0.167938, 0.140161, 0.043576, -0.071086, 0.326086, -0.044028, 0.301861, -0.250877, 0.17394, -0.187699, 0.08051, -0.153025, -0.087912, 0.03019, -0.158587, -0.026353, -0.019984, -0.16533, 0.070378, -0.212081, -0.191833, 0.123541, 0.220268, 0.20067, 0.46762, 0.158316, -0.079367, -0.149623, -0.215942, 0.063367, 0.256644, -0.080604, 0.095507, -0.030804, 0.099276, 0.133534, -0.282681, -0.238893, 0.077798, -0.09302, -0.214431, 0.243142, -0.072582, 0.213078, -0.128401, -0.316155, 0.183077, 0.2214], [0.110667, 0.108714, -0.027581, -0.177261, -0.072297, 0.07019, -0.087067, 0.119301, -0.091424, -0.004565, 0.157179, 0.391254, 0.101683, -0.095114, -0.126204, 0.003597, 0.151323, 0.215044, -0.048867, -0.0368, -0.083432, 0.152649, -0.029913, -0.103482, -0.116157, 0.0369, -0.226351, 0.169956, 0.072248, -0.043494, -0.082285, -0.066574, 0.011133, -0.02898, -0.031097, -0.138549, -0.150008, 0.001347, 0.097513, -0.06928, -0.03324, 0.052858, -0.059968, 0.075414, 0.035116, 0.172236, -0.137191, 0.074476, 0.300469, -0.0735, -0.010692, -0.144894, 0.112634, 0.070627, -0.056233, 0.106045, -0.107454, 0.219269, 0.276507, 0.238853, -0.055015, 0.160264, 0.091792, -0.038749], [-0.212842, 0.022217, -0.089318, -0.022054, -0.241489, -0.140489, -0.096595, 0.441717, 0.162701, -0.043123, -0.051132, 0.209101, 0.037996, 0.058339, -0.041685, 0.193891, 0.154989, 0.171254, 0.111323, -0.05363, -0.182475, 0.012016, 0.170755, -0.023287, -0.101311, -0.002929, -0.176315, -0.099699, 0.042897, -0.199563, -0.098536, 0.160766, 0.084529, 0.048417, 0.032702, -0.058501, -0.052329, -0.031942, -0.047903, -0.168134, 0.068013, 0.109392, -0.068562, 0.174756, -0.01277, 0.176147, 0.034432, -0.106282, 0.134372, -0.058806, -0.058807, -0.196102, 0.018476, -0.020185, 0.018616, 0.123929, -0.046611, 0.175985, 0.057914, 0.162833, -0.019096, 0.009068, -0.065629, 0.061442], [0.031845, 0.119115, 0.224582, 0.294505, 0.175766, 0.119102, 0.006968, 0.276589, -0.038203, -0.205819, -0.102113, -0.155182, -0.065853, 0.001824, -0.049861, -0.110985, 0.15138, -0.287086, -0.032337, -0.07847, 0.067017, -0.004997, -0.03174, -0.092449, 0.125486, -0.034014, -0.213696, -0.017377, -0.052935, -0.03334, 0.041784, -0.169647, -0.32294, -0.088665, -0.103227, 0.248531, -0.006152, -0.011744, 0.223261, -0.147234, 0.030522, -0.025469, -0.285847, -0.014202, 0.18595, -0.093177, 0.082893, 0.160701, 0.197146, -0.062669, -0.153195, 0.084917, 0.00094, -0.268104, -0.058643, 0.403815, -0.013284, -0.232744, 0.205561, -0.1254, -0.031069, 0.057437, 0.049593, -0.105845], [-0.142654, -0.037136, 0.231938, 0.157871, 0.187157, -0.010801, 0.11976, -0.030438, 0.016019, -0.108078, -0.074201, -0.126354, 0.20307, 0.056092, 0.08794, 0.139017, -0.050285, 0.015602, 0.025223, -0.247688, 0.057672, -0.193096, -0.179931, -0.079907, -0.281814, -0.045811, 0.0945, -0.061966, -0.034403, -0.063257, -0.021703, -0.163574, -0.368073, 0.081353, 0.227471, 0.189244, -0.197677, -0.03112, -0.009271, 0.051404, -0.00534, -0.115225, -0.104721, -0.132407, -0.156862, -0.169057, -0.086759, -0.023166, -0.079688, -0.056913, -0.003306, 0.238004, 0.062652, 0.040909, -0.027757, 0.043709, -0.093144, 0.008045, 0.042466, -0.272341, 0.212199, 0.060127, 0.027966, 0.171264], [-0.067878, 0.329194, -0.039293, -0.130373, 0.132721, 0.161638, 0.305141, -0.326242, -0.154895, -0.196974, -0.053439, -0.038329, 0.359654, -0.216212, -0.101905, -0.019354, -0.223223, 0.286131, 0.151484, 0.061544, 0.14234, 0.168834, 0.120253, 0.166696, 0.04292, -0.149828, 0.015634, -0.055125, 0.206547, -0.036446, -0.078915, -0.121901, -0.180625, -0.398551, -0.234739, 0.179841, -0.264162, 0.099404, -0.283459, 0.004959, 0.157547, 0.097063, -0.141057, 0.022057, -0.047497, -0.105337, 0.006976, 0.24657, -0.381237, -0.375136, 0.000212, -0.166513, -0.29195, 0.358558, 0.10884, 0.324617, 0.111335, 0.168304, -0.315987, 0.053191, -0.316534, -0.00831, 0.035726, -0.03931], [0.065504, 0.151295, 0.058285, 0.038775, -0.180038, 0.16182, 0.246709, -0.151454, 0.139952, -0.101676, -0.133031, 0.118424, 0.024581, 0.173004, 0.005523, -0.005074, 0.087806, 0.131623, 0.031843, -0.096775, -0.228625, -0.137485, -0.060626, 0.196246, -0.006609, -0.18596, 0.384388, 0.017145, -0.19338, -0.06233, 0.112094, 0.088444, 0.021602, -0.178653, 0.196337, -0.051137, 0.095242, -0.120179, 0.185021, -0.104984, 0.025419, 0.044781, -0.083487, 0.077932, 0.106541, -0.038221, -0.186351, 0.011968, -0.108516, -0.063777, 0.009156, 0.016258, -0.056135, -0.045045, -0.092696, -0.01952, -0.008127, -0.009985, -0.02664, -0.025189, -0.053824, 0.218024, -0.034418, -0.107075], [-0.063121, 0.008332, -0.085137, 0.134769, 0.097431, -0.067921, -0.007302, -0.001574, -0.212068, 0.089777, 0.200536, 0.039212, -0.126507, -0.058147, 0.039317, 0.036055, -0.027966, -0.015988, -0.103482, -0.015558, 0.02683, -0.133878, -0.071914, 0.067847, -0.288391, -0.090093, 0.028221, -0.06432, 0.183306, -0.035114, -0.009771, -0.144386, 0.071852, 0.02224, 0.193493, 0.028548, -0.111112, 0.077676, -0.139317, 0.027172, 0.069792, -0.13486, 0.136332, -0.131123, -0.013057, -0.007984, 0.038122, -0.057425, -0.059215, -0.043693, -0.143163, 0.108867, 0.11117, 0.116906, 0.00856, -0.18181, -0.044484, -0.094026, -0.015365, -0.116932, 0.022989, 0.067277, -0.185503, 0.161468], [0.082316, 0.252411, 0.000532, -0.239472, 0.025697, 0.164907, -0.000965, 0.104141, 0.1354, -0.001936, -0.175833, 0.254712, 0.222898, -0.097869, -0.086059, -0.137101, -0.083109, -0.15593, 0.137569, -0.178559, 0.159087, -0.120706, 0.043111, 0.382304, -0.036386, -0.151562, 0.022439, 0.203819, -0.034674, -0.12779, 0.063626, -0.009333, 0.3023, -0.118501, 0.062587, -0.123783, -0.11722, -0.174291, -0.407871, 0.073883, -0.192895, -0.016691, 0.108567, 0.240273, -0.381651, -0.236742, 0.015989, -0.104413, 0.101688, -0.075833, 0.179031, 0.362454, -0.190095, 0.10728, -0.251964, -0.460786, -0.013628, 0.066331, 0.153143, -0.028767, -0.061102, -0.076582, -0.043549, 0.083778], [-0.112739, -0.018817, 0.212727, -0.251643, -0.159116, -0.300202, -0.0233, 0.00465, 0.160405, 0.109984, 0.045092, -0.197522, -0.088712, 0.040581, -0.009254, 0.032459, -0.101264, -0.005041, -0.057506, -0.15428, 0.006962, 0.003909, -0.099704, -0.297978, 0.07608, 0.133551, 0.037021, 0.075413, -0.100369, -0.050744, 0.022988, -0.260815, -0.066629, 0.039161, -0.349717, 0.055876, 0.1603, 0.01703, 0.25224, 0.344851, 0.078165, 0.332008, -0.234568, 0.22123, -0.050739, 0.169614, 0.007595, 0.081251, -0.078822, 0.240325, -0.251374, -0.042328, 0.123609, -0.120514, 0.181606, 0.121133, -0.279975, -0.064422, 0.006328, 0.08716, 0.148101, -0.399062, 0.125756, 0.319535], [0.104312, -0.19178, 0.062089, 0.27275, 0.027234, 0.023526, 0.070822, -0.110035, 0.253997, 0.000797, 0.051121, 0.086631, -0.223694, 0.048772, -0.026247, 0.068755, -0.231795, -0.264537, 0.103931, 0.294177, -0.106947, -0.131882, 0.45388, 0.112705, -0.174223, 0.190084, 0.15663, 0.001868, 0.26238, -0.004326, 0.170401, -0.057928, -0.018794, -0.068984, -0.157443, -0.002849, -0.069483, 0.104372, -0.11469, -0.117475, -0.064246, -0.148555, -0.131624, 0.356956, 0.010568, -0.077445, -0.066906, 0.197308, -0.268899, -0.287037, -0.022484, -0.11252, -0.065536, -0.200045, -0.029536, -0.181666, -0.053878, -0.04165, -0.140638, 0.286107, 0.091706, 0.073064, 0.278129, 0.048524], [0.248558, 0.016908, -0.32161, 0.239569, 0.012804, 0.178588, 0.287617, -0.263447, 0.347874, 0.305644, 0.152954, -0.056753, 0.126814, 0.250235, 0.165563, 0.030077, 0.172678, 0.218998, 0.355589, -0.102917, 0.171605, 0.071146, 0.103101, 0.090966, -0.148975, 0.261941, 0.067459, -0.10761, -0.00725, -0.181074, 0.015213, -0.108876, -0.312512, 0.209408, -0.038443, 0.207614, -0.056996, 0.018336, 0.250394, 0.029425, 0.3576, 0.170883, 0.052774, -0.140363, -0.20836, 0.024397, 0.103994, 0.275387, -0.169213, -0.026732, 0.279074, -0.083877, 0.069951, -0.003953, -0.130172, -0.218301, -0.156825, 0.081988, 0.100638, -0.097997, 0.33529, -0.325702, 0.246506, 0.063269], [0.044457, 0.230092, 0.271342, 0.182425, 0.190585, -0.150294, -0.000262, -0.150811, 0.033314, -0.091318, 0.011381, 0.00301, 0.003063, -0.2634, 0.216006, -0.064245, 0.04945, -0.281654, -0.043783, -0.208714, 0.153598, 0.058287, 0.001114, 0.001268, 0.033191, -0.148478, 0.084441, -0.354198, -0.299048, -0.056135, 0.135145, -0.038203, -0.15833, -0.044677, 0.048632, -0.133984, -0.082852, 0.241959, 0.08446, -0.087073, -0.113288, -0.168558, 0.237336, -0.382202, 0.093828, 0.138032, 0.033816, -0.025184, -0.016321, 0.059328, 0.06243, -0.043852, -0.100341, 0.0109, 0.333047, -0.167944, -0.068212, -0.095047, -0.343357, 0.133717, -0.157354, 0.108787, -0.093734, 0.047814], [-0.065357, 0.00514, 0.079043, 0.236022, 0.146553, 0.10455, -0.039028, 0.071216, 0.101394, 0.258075, 0.151293, -0.08047, -0.174322, 0.086321, 0.081891, -0.161075, -0.028784, 0.012233, 0.115165, -0.147397, 0.105052, 0.067571, -0.008739, 0.032959, 0.150071, 0.187643, -0.12325, -0.160166, -0.026374, -0.133755, 0.12117, 0.046545, -0.228572, 0.160462, -0.189795, -0.140364, -0.127744, -0.119381, 0.001055, -0.034801, 0.068206, 0.079573, -0.01564, -0.231239, -0.060013, 0.076313, 0.110361, 0.094244, 0.054788, -0.102623, -0.133225, 0.011826, -0.240957, -0.050478, -0.013733, 0.024095, -0.214423, -0.09258, -0.186946, 0.157576, 0.075315, 0.03349, 0.143573, 0.184009], [-0.26384, -0.197135, 0.07223, -0.33309, -0.172355, -0.254525, -0.038109, -0.062792, -0.211798, 0.041676, -0.402789, -0.294661, -0.181276, -0.011393, -0.026511, 0.130881, -0.087874, -0.052142, -0.291073, 0.255664, 0.074017, 0.070139, -0.140772, -0.091676, -0.070724, -0.206125, -0.04389, -0.125565, 0.101843, -0.146306, -0.119086, 0.070693, 0.207238, 0.041174, 0.18262, -0.047041, 0.054633, -0.03685, -0.385941, -0.025421, -0.125175, -0.031165, -0.021194, 0.067129, -0.005081, -0.026422, -0.395868, -0.103167, -0.228057, -0.195893, 0.039992, -0.044093, -0.225232, -0.014022, 0.080869, -0.100899, -0.066655, -0.00494, 0.021227, 0.276803, -0.142293, 0.133252, -0.055829, -0.025133], [-0.214327, 0.531397, -0.059276, 0.346416, 0.162655, -0.0905, 0.070603, 0.024554, 0.174398, 0.38368, -0.267632, -0.181617, -0.285911, -0.360959, 0.14179, -0.320503, -0.282234, 0.230182, -0.02549, 0.397068, 0.243319, -0.304403, -0.152592, 0.227635, -0.485781, 0.190769, -0.255383, -0.121644, -0.179128, -0.094235, 0.002969, -0.097534, 0.034395, 0.152035, 0.118397, -0.3466, -0.083901, 0.286911, 0.031928, 0.140707, -0.061683, 0.268559, 0.16492, -0.12376, -0.072523, -0.395462, 0.070908, 0.192187, -0.180763, -0.213802, 0.07391, 0.167709, -0.155779, 0.355733, -0.031106, 0.261066, -0.365847, -0.114943, -0.341546, 0.402488, -0.143842, 0.021835, 0.012285, 0.24589], [-0.047826, 0.211349, -0.060572, 0.404385, 0.105447, -0.016511, 0.203075, 0.19613, 0.050823, 0.179977, 0.095717, 0.126496, 0.113262, 0.116017, -0.046252, 0.112736, 0.05568, 0.150491, 0.130987, 0.097873, 0.095098, 0.263722, 0.02823, 0.199377, -0.208474, 0.108222, 0.083496, -0.055318, -0.150751, -0.082065, -0.066288, 0.082206, 0.007699, 0.103608, -0.035246, -0.201582, -0.102297, -0.04197, -0.102442, -0.119131, 0.28134, 0.372388, -0.126819, -0.061402, 0.103043, -0.115493, 0.043111, -0.062912, -0.096374, 0.101456, 0.463059, 0.106733, 0.243176, 0.073341, -0.154632, 0.250807, 0.103543, 0.002888, -0.066723, 0.20567, 0.178075, 0.205292, 0.160902, 0.191298], [-0.07624, 0.076691, -0.121949, -0.066907, 0.18972, -0.001578, -0.122736, 0.291547, 0.016602, -0.159439, -0.027246, 0.291582, -0.175317, -0.079243, -0.1751, -0.167419, -0.076822, -0.305694, 0.140832, 0.162565, -0.061198, -0.077937, -0.037182, -0.244542, 0.076978, 0.256041, -0.185199, 0.021935, -0.042685, 0.281692, 0.080756, 0.21736, 0.242445, -0.157417, -0.027237, -0.0516, 0.294659, -0.033388, 0.245987, 0.139026, 0.062941, 0.070463, -0.012942, 0.237631, -0.11414, -0.048322, 0.120886, 0.002106, 0.120031, 0.039331, -0.31184, 0.278172, -0.058373, 0.012793, -0.143538, 0.101576, 0.378083, -0.151904, 0.173298, 0.31627, -0.124873, -0.126391, 0.244821, 0.102403], [0.035056, -0.223231, 0.157214, -0.367652, -0.014481, -0.189547, -0.079504, -0.237727, -0.141407, 0.05437, -0.244286, -0.181459, -0.185755, 0.173492, 0.007067, -0.234408, -0.110002, 0.175305, -0.251015, 0.152123, -0.103974, -0.224802, -0.024247, -0.025846, -0.00948, -0.162276, -0.134171, 0.033931, 0.190782, 0.020416, -0.109289, 0.031594, -0.014511, -0.04466, -0.003758, 0.149204, -0.085085, -0.013355, -0.198271, -0.056709, -0.116878, -0.219673, 0.117727, 0.119386, -0.159577, -0.119755, -0.0199, -0.251586, -0.31664, -0.310224, -0.172036, 0.036445, -0.284201, -0.476994, -0.220847, -0.144053, -0.399767, -0.117648, 0.251136, 0.04454, -0.114947, -0.002997, 0.056848, -0.029889], [0.028056, -0.013829, 0.197123, -0.189659, -0.262791, -0.307115, -0.095795, 0.011952, -0.060274, -0.248952, -0.261874, 0.226465, 0.256363, -0.118523, -0.262124, -0.089117, 0.000369, 0.172115, 0.120719, -0.411464, -0.035983, 0.346304, 0.003671, 0.188204, 0.215394, -0.326737, -0.188908, -0.358149, 0.03959, -0.157453, -0.046573, 0.037239, -0.28413, -0.24085, -0.18429, -0.007618, -0.284638, -0.032173, -0.889071, 0.137837, 0.046861, 0.030491, -0.209852, -0.448646, -0.035184, 0.236079, -0.282138, -0.564137, -0.181641, -0.210189, 0.002904, -0.320152, -0.178252, -0.048798, -0.221735, -0.058567, -0.163115, 0.22578, 0.077843, -0.201534, -0.246422, -0.15539, -0.203311, -0.134209], [-0.083148, 0.012124, -0.001009, -0.068126, -0.059272, -0.094222, -0.047291, -0.1058, -0.07031, 0.165548, 0.301005, 0.186954, 0.063529, -0.011258, 0.155585, 0.268302, 0.241829, -0.327344, 0.088249, -0.159404, 0.04168, 0.155826, 0.018521, 0.066382, 0.248114, 0.218184, 0.237648, -0.168634, -0.081016, 0.088036, 0.50995, 0.15457, -0.109425, 0.224629, -0.068738, 0.145909, 0.454058, -0.169001, -0.117837, -0.31049, -0.016273, -0.490278, -0.188595, -0.220796, -0.021694, 0.210188, -0.087085, 0.136013, 0.111779, 0.426552, 0.12878, -0.1966, 0.16523, 0.329722, 0.21475, -0.25362, 0.148446, 0.169983, -0.097605, -0.096534, 0.400886, -0.012725, 0.210406, 0.053008], [0.268951, -0.189778, -0.076009, -0.301788, -0.136996, 0.283178, 0.06678, 0.206856, -0.066044, 0.064102, 0.273322, -0.069844, 0.209971, 0.265477, 0.001846, -0.134248, 0.279672, -0.05725, -0.088348, -0.246257, -0.02816, -0.160265, 0.150765, -0.256315, 0.109873, 0.127591, -0.079605, 0.219421, -0.156577, 0.128841, -0.044674, 0.042335, -0.276258, -0.071, 0.008261, 0.124896, 0.229847, -0.028205, 0.438714, -0.027089, -0.041379, 0.063445, 0.135116, 0.097444, -0.054143, 0.336104, -0.059438, 0.193122, 0.099696, 0.144542, -0.247999, -0.01836, -0.163284, -0.106384, -0.114785, -0.379734, 0.074877, -0.02091, 0.387686, -0.258923, 0.065139, -0.247704, 0.217686, -0.071736], [0.017628, -0.116816, -0.021219, -0.102238, -0.042761, 0.079163, -0.036373, -0.071137, 0.024739, 0.183057, 0.220785, -0.057357, 0.097717, 0.078784, -0.123468, -0.157903, -0.09904, 0.076271, 0.163845, 0.027366, -0.060469, 0.276009, 0.026148, -0.048147, 0.00864, -0.031294, 0.00901, 0.299178, 0.155596, 0.21882, 0.183259, 0.107399, -0.089093, 0.18662, 0.182769, 0.022674, -0.143675, 0.120441, 0.318097, -0.162423, 0.101027, -0.055194, -0.221424, 0.075634, 0.174363, 0.074239, -0.035584, 0.000764, 0.030696, -0.080892, 0.106633, -0.09836, 0.281487, -0.016485, 0.007391, -0.063938, 0.249634, 0.16795, -0.022639, 0.411558, 0.33967, 0.010026, -0.018224, 0.241094], [0.12427, 0.319762, -0.034203, 0.0106, 0.141289, -0.126641, 0.140476, -0.16658, 0.287278, -0.034351, -0.211574, -0.240414, 0.364445, -0.157353, 0.095078, 0.200145, -0.117498, 0.106182, -0.006963, -0.18277, 0.174449, -0.079531, 0.15433, 0.193018, -0.108795, -0.085545, -0.038346, -0.174097, 0.011471, -0.239586, -0.016381, 0.053985, -0.126832, 0.221202, 0.100716, -0.186571, 0.250615, 0.391949, 0.070713, 0.142269, 0.116705, 0.141275, 0.308497, -0.125819, -0.091, 0.163892, 0.08947, 0.416451, 0.060242, 0.019632, -0.032658, -0.045828, 0.007121, -0.031444, -0.231737, -0.008155, -0.037878, 0.058199, -0.211339, -0.250098, -0.221485, -0.303934, 0.024468, 0.055544], [0.025256, -0.045548, 0.039722, -0.20062, -0.122357, -0.129715, 0.208347, 0.474867, 0.300665, -0.306632, -0.142123, -0.001362, -0.087925, 0.479546, 0.006387, -0.173423, 0.048352, 0.164251, 0.065122, -0.026022, -0.120407, 0.343103, 0.261521, 0.256499, -0.034722, -0.139382, -0.068599, 0.222669, 0.239301, 0.009295, 0.144082, 0.06302, -0.090595, 0.023153, -0.024917, -0.097268, 0.020534, -0.086596, -0.090786, -0.23184, -0.03621, 0.02839, -0.299361, 0.127512, 0.139908, -0.010647, -0.252685, 0.162613, 0.055284, 0.012256, -0.145878, -0.169697, -0.030261, -0.142989, -0.082623, 0.086471, -0.180138, 0.060139, 0.162272, 0.214581, -0.056931, -0.228006, -0.037018, -0.172078], [0.349956, -0.203418, -0.020619, 0.139794, 0.260234, 0.44712, 0.273165, -0.010801, -0.043013, -0.054264, 0.27539, -0.265161, -0.21998, 0.067663, 0.164646, -0.143852, 0.056936, -0.113364, -0.061394, 0.077229, -0.051923, 0.096796, 0.128311, -0.259901, 0.243232, 0.081455, 0.369995, 0.137602, -0.158333, 0.267195, 0.129351, 0.131111, 0.169714, -0.007065, -0.156288, 0.035737, -0.110749, -0.010555, 0.532348, 0.038901, 0.126794, 0.136915, 0.38875, 0.016204, 0.108512, -0.073643, 0.095581, -0.032704, -0.145531, 0.176523, 0.215419, 0.10479, -0.046859, -0.27041, 0.231454, -0.195752, 0.387354, -0.109833, 0.192531, 0.114147, -0.037827, 0.10032, -0.007954, -0.255699], [0.120836, -0.054605, -0.005689, -0.084642, -0.13479, -0.082294, -0.008571, 0.258499, 0.17204, -0.032625, 0.132509, -0.037552, -0.03722, 0.142053, 0.184807, -0.140899, 0.09, 0.201086, -0.068664, -0.240433, -0.104337, 0.139433, 0.025004, -0.037107, 0.010449, -0.163774, -0.105431, 0.249228, 0.311423, -0.281602, 0.061283, -0.226399, 0.181842, -0.029121, -0.273941, 0.123778, 0.08344, -0.045146, 0.114155, 0.046935, 0.048905, 0.095819, -0.109925, -0.130378, 0.002274, -0.060609, -0.165375, -0.000844, 0.242534, 0.173215, 0.175127, 0.001706, 0.074411, -0.108944, 0.02842, -0.069551, 0.185169, -0.080081, 0.06148, 0.093617, 0.142915, -0.001773, -0.135761, -0.044573]], [-0.196092, -0.012259, 0.246812, -0.205838, 0.171508, -0.176346, -0.074297, -0.112078, -0.076985, 0.05062, 0.069916, -0.097885, 0.041569, -0.062462, 0.00567, -0.084391, 0.008814, 0.081391, -0.250716, -0.327594, -0.08659, -0.069965, -0.044907, -0.064949, 0.2033, -0.071277, -0.100252, -0.148503, -0.065816, -0.070139, 0.109101, 0.052069, -0.303358, 0.093606, 0.006183, -0.016686, 0.068058, 0.074185, -0.254784, -0.139659, -0.308138, -0.262567, 0.065603, -0.097966, -0.095476, 0.148326, 0.014087, -0.230145, -0.092591, 0.153589, -0.101509, -0.137588, -0.26948, -0.117262, 0.249789, -0.169956, -0.085904, -0.123921, -0.200176, -0.356149, -0.175344, 0.081864, -0.264103, -0.072474]),
    ([[0.000551, -0.068661, -0.181825, -0.058584, -0.099284, 0.055493, 0.003691, 0.0644, 0.065503, -0.172517, -0.017356, -0.120795, -0.051217, -0.014017, -0.171934, 0.070505, -0.000477, 0.05418, 0.092817, -0.113989, -0.084415, -0.100586, -0.070704, 0.050839, 0.07748, 0.078369, -0.262615, 0.008265, -0.324979, 0.047019, -0.24341, -0.025437], [0.173095, -0.149491, 0.154807, 0.040802, 0.107833, 0.043523, 0.120391, 0.044107, 0.1982, 0.06853, 0.02837, -0.259331, -0.108155, 0.152991, -0.028788, 0.22348, 0.204317, 0.198855, 0.113038, 0.036389, -5.2e-05, 0.076423, 0.073792, 0.166731, 0.236874, 0.1477, -0.0785, 0.194309, -0.089093, 0.186693, 0.040418, -0.003382], [-0.337026, 0.027579, 0.030352, 0.004489, 0.028121, -0.029741, -0.269779, 0.105893, -0.104353, 0.157508, -0.003142, -0.007894, -0.026035, 0.087021, 0.04135, 0.14226, 0.077951, -0.275448, 0.152064, 0.146447, 0.006271, -0.110643, 0.158576, -0.35799, -0.140756, 0.058162, 0.096108, -0.112769, 0.027596, -0.461029, -0.131432, 0.020984], [0.021373, 0.013385, 0.150669, 0.050438, 0.244643, -0.068734, 0.196428, -0.034768, 0.330724, 0.159678, 0.20479, 0.507217, 0.055167, -0.08417, 0.200629, -0.047986, 0.117401, 0.064484, 0.099349, 0.217497, 0.11854, 0.144709, 0.048662, 0.566838, 0.086592, -0.083141, 0.063605, 0.06265, 0.136565, 0.303095, 0.073809, 0.169428], [-0.091278, 0.023784, -0.095635, -0.125688, -0.016676, 0.180859, 0.143271, -0.00816, 0.130462, -0.190802, -0.246498, 0.068963, -0.026643, -0.02053, 0.122631, 0.128597, 0.049375, 0.050039, -0.021992, 0.04057, -0.11685, -0.141928, 0.115447, 0.098471, 0.095053, 0.057761, 0.009454, -0.056358, -0.082678, 0.147824, 0.055466, 0.175205], [-0.251833, 0.014086, 0.124137, 0.049245, 0.054972, -0.085252, -0.158112, -0.078596, -0.060837, -0.062309, 0.142741, -0.019963, 0.087441, 0.026643, 0.055729, -0.208869, -0.226102, -0.100739, -0.029172, 0.100398, 0.107492, -0.071941, -0.102036, -0.069558, 0.006539, -0.102002, 0.077414, 0.069702, -0.069625, -0.252165, -0.053581, 0.09793], [0.059561, 0.092538, 0.049228, -0.046531, 0.260629, 0.021251, -0.054511, 0.046192, 0.020331, 0.0785, 0.199591, -0.041287, 0.13351, 0.071005, -0.410724, 0.086112, 0.129078, 0.090857, 0.09177, 0.202456, -0.00862, 0.051211, 0.027407, 0.030106, 0.009528, 0.074726, 0.006056, -0.068163, 0.204347, 0.133541, -0.090537, 0.190658], [0.115215, -0.005663, -0.025102, 0.028472, 0.120502, 0.117668, -0.256542, 0.043396, -0.030599, 0.110919, 0.072465, -0.301083, -0.004188, -0.057144, 0.148402, -0.107562, -0.108025, -0.202001, -0.191783, 0.067989, -0.128963, -0.026092, -0.171862, -0.574864, -0.396655, 0.131888, 0.009256, -0.061486, 0.009665, 0.03055, -0.13079, -0.25993], [0.069788, -0.053642, 0.071458, -0.085519, -0.193975, -0.06064, 0.029324, 0.050994, 0.143353, -0.20169, 0.079206, -0.012674, -0.015828, -0.021683, -0.288176, -0.018733, -0.065954, 0.135563, 0.045647, -0.10435, -0.261871, 0.099209, -0.030026, 0.102193, 0.057102, 0.043783, -0.014302, 0.043634, -0.291362, 0.113195, -0.242202, -0.511954], [0.25313, 0.022359, 0.104076, -0.057992, 0.063179, 0.032792, 0.235039, 0.079686, -0.045203, -0.151916, 0.134401, -0.058487, 0.110266, 0.079909, 0.140803, 0.173885, 0.007198, 0.133593, -0.094546, -0.003185, 0.138068, 0.162387, -0.104093, 0.009613, 0.046645, 0.191471, 0.131494, -0.004739, 0.066608, 0.184356, -0.081998, 0.04335], [0.121544, -0.005876, -0.055227, 0.151668, 0.254946, 0.055646, 0.048199, 0.012522, 0.192526, 0.035041, -0.000386, -0.177853, 0.202959, 0.079369, 0.325582, -0.05776, 0.085572, 0.062849, 0.046601, 0.253089, 0.091257, 0.007132, -0.10811, 0.00991, -0.06169, 0.097585, -0.024611, -0.091863, 0.287878, -0.003835, 0.091814, 0.303753], [0.287179, -0.023247, 0.169934, -0.113478, -0.148381, 0.117742, 0.054173, 0.011069, 0.030772, -0.031067, -0.083212, -0.002464, -0.088627, -0.016629, -0.326008, -0.197284, -0.094008, -0.14329, -0.026547, -0.251918, -0.10697, -0.031252, -0.031548, -0.215957, 0.016702, -0.080923, -0.055086, -0.147559, -0.19352, -0.070396, -0.213494, -0.232608], [0.124039, -0.084099, -0.060528, 0.101182, -0.355944, 0.041672, 0.100228, 0.094979, 0.243174, 0.002008, -0.070048, -0.009681, -0.081853, -0.021472, -0.413818, -0.075507, 0.074871, 0.044128, 0.13665, -0.2188, -0.061103, -0.104578, -0.053081, 0.018204, -0.129728, 0.104501, -0.047393, -0.117367, -0.193808, 0.08288, -0.129529, -0.492051], [0.02889, 0.089178, 0.069914, -0.003442, -0.336432, -0.006414, 0.049589, -0.061762, -0.008374, -0.287758, -0.010194, -0.191795, -0.172251, -0.011563, -0.450851, -0.091294, -0.060506, -0.072309, 0.035252, -0.168328, -0.057045, 0.120912, 0.021199, -0.504007, 0.192234, 0.006603, -0.051007, 0.017623, -0.377895, 0.109021, -0.081139, -0.525223], [0.046279, 0.099371, -0.09894, 0.09996, -0.04191, -0.040508, 0.267264, -0.165832, 0.193317, 0.052798, 0.023505, 0.15244, 0.091095, -0.048512, -0.087894, 0.239702, 0.007348, -0.003923, 0.018559, 0.082295, 0.061837, 0.055, 0.043772, 0.179792, 0.083298, -0.084061, 0.032794, 0.125797, 0.141809, 0.084534, 0.145702, 0.037195], [-0.213838, 0.14779, 0.053624, 0.002008, 0.009933, 0.11582, -0.202648, -0.046603, -0.26576, 0.007172, 0.032087, -0.227624, 0.105377, -0.066235, 0.086749, -0.085625, -0.187976, -0.17161, -0.297631, -0.116019, 0.149222, -0.001083, -0.296779, -0.088519, -0.230114, -0.006044, 0.158384, -0.131693, -0.032929, -0.093596, 0.168942, 0.017474], [-0.004427, 0.035846, -0.005413, -0.0522, 0.115215, -0.026506, 0.08414, -0.066942, -0.104801, 0.086799, 0.165671, 0.145924, -0.053339, -0.037498, 0.118894, -0.073757, -0.062063, -0.011291, 0.074285, -0.103194, 0.065137, 0.114832, 0.042484, -0.103011, -0.023504, -0.070365, -0.027991, -0.029809, -0.095639, -0.155212, 0.148207, 0.059405], [0.015611, 0.034052, -0.141388, -0.11668, -0.327604, 0.075815, -0.086577, -0.032621, -0.395703, -0.314382, -0.007363, -0.114306, -0.185083, -0.079343, -0.450086, 0.096, -0.098128, 0.086427, 0.159759, -0.078027, -0.034334, 0.049474, 0.028943, -0.075606, 0.008874, -0.166449, -0.27614, -0.018246, -0.185358, 0.090501, -0.130463, -0.137805], [0.117254, 0.012291, -0.105121, -0.014311, 0.06828, 0.02391, -0.103226, 0.028447, 0.269802, 0.096433, 0.058436, -0.029742, 0.096488, 0.031339, 0.014406, -0.119225, 0.012572, -0.006115, 0.096502, -0.300903, -0.154287, 0.016319, 0.110337, 0.287888, 0.177673, -0.078019, -0.020834, 0.028729, 0.035126, 0.055712, 0.098409, -0.306595], [0.02992, -0.176451, -0.099793, -0.127118, -0.227444, -0.051962, 0.08303, 0.165138, 0.465415, -0.062281, -0.232357, 0.010984, -0.265316, 0.313684, -0.101195, 0.102018, 0.093166, 0.105158, 0.173696, -0.153143, -0.12409, -0.053772, 0.173922, 0.365249, 0.188555, 0.005351, -0.261343, 0.165512, -0.062464, 0.372872, -0.004159, -0.108047], [0.1563, -0.045154, -0.106015, -0.092845, -0.428412, 0.104955, 0.04279, 0.058452, -0.071991, 0.11707, -0.055344, 0.067999, 0.02702, -0.050583, -0.243571, 0.105582, 0.113702, 0.114168, 0.049805, -0.034281, -0.092972, -0.281076, 0.113461, 0.138357, -0.043641, -0.016014, -0.067799, 0.000365, -0.15271, 0.144607, -0.201214, 0.253922], [-0.048254, -0.028003, 0.025496, 0.005025, -0.199829, 0.111967, -0.492454, -0.114418, 0.004703, -0.082772, -0.018285, -0.113835, 0.110223, -0.114756, 0.010741, -0.415891, -0.045257, 0.038691, -0.009757, 0.073192, 0.030764, 0.00355, 0.012422, 0.036808, -0.126257, -0.132795, 0.098071, -0.122609, 0.065541, -0.364568, 0.109349, -0.106034], [0.018195, -0.114576, 0.029092, -0.029021, -0.159904, 0.062073, 0.098417, 0.020775, 0.124547, -0.364162, 0.026776, -0.426795, 0.027203, 0.117685, -0.209822, -0.000654, -0.122383, 0.211817, -0.027536, 0.060694, -0.021252, 0.02426, -0.013481, -0.253339, 0.260407, -0.00815, -0.088467, 0.10908, 0.001645, 0.252668, -0.018691, -0.153142], [0.075158, 0.081094, 0.139642, -0.122522, 0.163688, -0.033027, 0.127202, 0.151244, 0.042436, 0.030661, 0.022387, -0.221539, 0.063571, 0.150481, 0.243175, 0.184465, 0.004277, 0.176627, 0.070922, -0.004232, 0.045439, 0.212976, 0.12752, 0.020911, 0.163794, 0.148176, 0.101531, 0.06059, 0.174914, 0.164338, 0.105138, 0.091352], [-0.046028, 0.135955, -0.049939, 0.100754, -0.124172, 0.001122, -0.041708, -0.022609, -0.112393, -0.026936, 0.088253, -0.034169, 0.078951, -0.261659, -0.192571, -0.078852, -0.128903, -0.166873, 0.066314, -0.119735, 0.07245, 0.003184, -0.173745, -0.130213, -0.042657, -0.013517, 0.167406, -0.095244, 0.064983, -0.065561, 0.106522, 0.061649], [0.195342, -0.009414, 0.043395, 0.026572, 0.143328, -0.126566, 0.090213, 0.051207, -0.059018, -0.033246, 0.184672, -0.166709, 0.115994, -0.172724, 0.121301, 0.067441, 0.149399, 0.107416, -0.072257, 0.090561, 0.028785, 0.140789, -0.157757, 0.051686, -0.110439, 0.033175, 0.142625, 0.296139, 0.075264, 0.203446, 0.030552, 0.259872], [-0.165776, 0.048253, 0.085397, -0.040843, -0.036011, 0.140181, 0.23748, -0.234781, 0.002038, -0.019174, 0.112186, -0.110043, 0.146683, -0.13744, -0.124232, 0.124577, -0.098303, -0.051496, -0.012431, 0.135904, 0.129104, 0.058076, -0.227372, -0.192767, 0.102234, -0.160899, -0.066892, -0.115061, 0.084167, -0.25735, 0.129452, 0.17459], [-0.079214, -0.053213, -0.006682, 0.010336, 0.085232, 0.059169, -0.17156, 0.019496, 0.163359, 0.151826, -0.051015, -0.140244, -0.015646, 0.124407, 0.190451, -0.305318, -0.122192, 0.075602, -0.047582, -0.032719, -0.052702, -0.020774, 0.067193, -0.255102, -0.063571, -0.050968, -0.130489, -0.057017, 0.049413, -0.035366, 0.09711, 0.095998], [-0.266787, 0.077968, 0.038473, -0.120475, 0.002277, -0.128798, -0.332355, -0.025723, -0.391772, -0.151499, -0.056852, -0.480436, 0.01873, -0.175663, 0.146159, -0.00053, 0.010218, -0.109114, -0.261511, -0.023951, 0.001071, 0.066243, -0.067098, -0.309191, -0.28824, -0.074132, -0.059485, -0.147923, 0.011999, -0.281785, -0.018546, -0.04087], [0.119286, 0.059308, 0.020606, 0.026894, 0.003689, -0.006632, -0.126918, 0.032512, 0.101616, 0.125727, -0.157227, 0.229821, 0.000449, -0.01498, 0.002029, -0.213603, -0.158871, -0.122142, -0.086913, -0.001673, 0.088346, 0.083056, -0.032499, -0.01664, -0.097133, -0.101204, -0.088024, 0.029754, 0.078562, -0.031551, 0.128448, -0.015221], [-0.005871, 0.078284, 0.1188, 0.086761, -0.020688, 0.110871, -0.0573, 0.038743, -0.098365, 0.15157, 0.027859, -0.088839, 0.122839, 0.174765, -0.008561, 0.158274, -0.091842, 0.200698, 0.039294, 0.064892, 0.183964, 0.088842, 0.065489, -0.25625, 0.244556, 0.111098, -0.007405, -0.089192, -0.031268, -0.039306, 0.099774, 0.035433], [0.08969, 0.137277, -0.032457, -0.083492, -0.055453, 0.1324, -0.038405, -0.051735, -0.112273, 0.055263, -0.047745, 0.022333, -0.036446, -0.236001, -0.105874, -0.148716, -0.051501, -0.14611, -0.00892, -0.028425, 0.149386, 0.147594, -0.081951, 0.10663, -0.21476, -0.027318, 0.122643, -0.154863, -0.005279, -0.192747, 0.014256, -0.022923], [-0.171798, -0.059612, 0.002249, 0.043351, -0.245085, -0.156948, -0.096544, -0.057627, -0.092176, 0.056544, -0.179241, -0.001743, -0.01941, -0.2814, -0.483637, -0.058389, -0.212223, -0.181186, -0.223991, -0.154572, -0.194431, 0.029626, -0.37039, -0.068585, -0.274074, -0.113571, -0.079928, -0.074998, -0.098604, -0.031457, 0.044441, 0.272899], [0.128833, 0.026629, 0.10134, 0.068884, 0.093548, 0.080228, -0.000528, 0.129678, 0.061007, 0.008756, 0.034228, -0.044639, 0.054189, -0.091626, 0.019648, 0.180341, -0.01107, 0.167613, -0.229916, -0.041934, -0.033871, -0.03276, -0.045069, 0.025145, -0.111357, 0.144241, 0.134766, 0.133194, 0.124496, 0.100586, 0.14513, -0.012632], [0.00445, -0.060061, -0.049501, -0.16085, -0.22416, -0.022488, 0.01277, 0.121029, -0.35206, -0.021552, -0.11598, 0.119145, 0.069003, 0.014928, -0.705494, -0.140495, 0.074062, -0.020336, -0.007354, -0.166136, -0.138837, 0.101256, 0.052758, -0.157793, -0.217202, -0.007071, 0.008033, -0.105317, -0.212264, -0.107543, -0.065729, -0.151386], [-0.278885, 0.185902, 0.074305, -0.016933, -0.025729, -0.208914, -0.00543, -0.237297, -0.403482, 0.004468, 0.080979, -0.316279, 0.072469, -0.067138, -0.03628, -0.068754, -0.027581, -0.270315, -0.219521, 0.093561, 0.014112, 0.03716, -0.010375, -0.265279, -0.09506, -0.063768, 0.102552, -0.215341, -0.000543, -0.223374, 0.1363, -0.002445], [-0.150215, 0.185525, 0.149544, 0.045268, -0.105503, 0.065207, 0.176083, 0.020837, -0.410763, -0.065759, -0.258318, 0.102066, 0.164269, -0.007587, 0.058811, 0.105736, 0.134683, -0.03313, 0.019545, -0.051828, 0.182748, 0.156482, -0.045397, -0.214087, 0.10562, -0.038636, 0.034353, 0.055966, 0.17854, 0.090313, 0.155498, -0.159225], [-0.003164, 0.113964, -0.264491, -0.033896, -0.192591, 0.096295, 0.165322, 0.202514, -0.073816, -0.113858, 0.082324, 0.03347, 0.006932, 0.134796, 0.026841, 0.147686, -0.086992, 0.122549, 0.059465, 0.195008, 0.017097, 0.017274, 0.087984, -0.19612, 0.01639, 0.089738, -0.007743, 0.092916, 0.108776, -0.046536, 0.008337, -0.099626], [0.194123, 0.1518, 0.148999, 0.161312, 0.584378, 0.131849, 0.136984, 0.048211, -0.029224, 0.091907, -0.03601, 0.149738, 0.082692, 0.027623, 0.486161, 0.004531, 0.137044, 0.189601, 0.04004, 0.261151, 0.131753, 0.165734, 0.152972, 0.157443, 0.040873, 0.084346, 0.115586, 0.029326, 0.174166, 0.067507, 0.214996, 0.33883], [-0.082638, -0.105601, 0.045178, -0.058563, -0.171221, -0.042228, -0.022824, 0.027547, -0.123516, -0.1177, -0.137258, -0.251763, -0.029495, -0.032206, -0.177588, 0.004903, 0.050534, -0.16695, -0.017894, -0.142023, -0.107863, -0.18117, -0.141135, 0.160601, 0.000111, -0.043502, -0.120315, 0.124829, -0.151525, -0.240311, -0.115546, -0.01135], [0.087947, -0.175301, -0.009514, -0.138764, -0.246153, -0.020466, -0.001445, 0.131308, -0.140517, 0.075485, -0.110189, 0.006964, -0.06482, 0.100006, -0.083263, 0.220343, -0.012296, 0.044669, 0.11629, -0.12795, -0.058702, -0.197163, 0.190461, 0.312138, -0.07639, 0.143031, -0.034878, 0.043236, -0.106982, -0.022458, -0.007591, -0.439295], [0.159526, -0.148745, -0.080172, -0.107793, -0.269827, -0.017427, -0.126771, 0.107384, -0.091401, -0.299555, -0.055968, -0.020604, -0.285631, 0.356791, -0.400139, -0.039758, 0.044801, 0.10547, 0.107757, -0.344855, -0.09609, 0.113676, 0.004815, 0.043797, 0.047581, 0.032966, -0.18971, 0.187834, -0.287811, 0.020594, -0.193755, -0.589057], [-0.11497, 0.02274, 0.023735, 0.096137, -0.185842, 0.062435, 0.079669, 0.089026, 0.065294, -0.039392, -0.064886, -0.007748, -0.006482, -0.038825, -0.24016, 0.075173, -0.104903, 0.045226, -0.09612, -0.231071, -0.111659, -0.034288, 0.019821, 0.128259, 0.032266, 0.088033, 0.126432, 0.022494, -0.277157, -0.014108, -0.140686, -0.112384], [-0.030983, -0.080721, 0.055871, 0.071488, 0.018009, -0.047488, -0.073106, -0.226849, -0.148689, -0.011477, -0.048529, -0.266102, -0.009023, -0.19508, 0.129957, -0.303332, 0.035216, -0.317454, -0.079605, 0.030739, -0.044193, 0.075904, -0.070344, -0.293225, -0.181468, 0.126499, -0.02304, -0.315409, 0.103968, -0.342158, -0.076034, 0.223565], [-0.147848, 0.127046, -0.039551, -0.082716, 0.030167, -0.284869, -0.080921, -0.070148, 0.005546, 0.177105, 0.042662, 0.047606, -0.139375, -0.077531, 0.009711, -0.063368, -0.315132, -0.263039, -0.025121, 0.077785, -0.016794, -0.044786, -0.088708, 0.150754, 0.06661, -0.32707, -0.097636, -0.100671, 0.019737, -0.141057, 0.107826, -0.172319], [-0.421685, 0.081631, 0.045455, 0.069153, -0.053311, 0.010835, -0.252844, -0.018962, -0.173353, 0.100536, 0.000484, -0.276511, 0.021884, -0.002517, -0.048466, -0.305935, -0.197789, 0.018492, 0.050884, -0.035894, 0.065437, 0.006756, -0.07319, -0.020252, -0.026547, 0.028611, 0.072138, -0.064582, 0.076309, -0.235285, -0.035766, 0.078802], [0.012628, 0.055687, 0.020132, -0.089893, 0.090412, 0.05158, -0.072717, 0.012396, 0.065695, -0.043759, -0.224136, -0.042255, 0.08011, 0.156376, -0.016515, 0.073969, 0.12749, 0.049294, 0.068411, -0.20531, -0.085078, 0.041987, -0.062937, 0.170453, 0.078308, 0.096599, -0.093782, 0.018428, -0.251313, 0.016998, 0.039955, -0.195606], [0.129327, -0.058465, 0.085498, 0.032542, 0.192599, 0.112293, 0.161835, 0.148039, 0.191917, 0.151084, 0.099923, 0.238972, -0.04, 0.050074, 0.245712, 0.070594, 0.121194, 0.131017, 0.048256, -0.018897, -0.088827, 0.092742, 0.142097, 0.000517, 0.05566, 0.182771, 0.15939, 0.158277, 0.086495, 0.201548, -0.056349, 0.036579], [-0.243865, 0.073537, -0.049861, 0.017255, -0.018534, -0.034799, 0.076902, -0.114385, -0.446978, 0.228714, -0.050424, -0.101057, 0.095818, -0.212021, 0.174499, 0.046488, 0.067852, -0.187723, -0.028087, -0.053719, -0.168966, 0.147364, -0.073877, -0.395607, 0.009097, 0.235997, 0.05419, -0.159508, -0.014502, -0.438043, 0.051281, -0.176525], [-0.123849, 0.042148, 0.147093, 0.155244, 0.104914, 0.023484, 0.226156, -0.193378, -0.087258, 0.117011, -0.173769, -0.034222, 0.112042, -0.188834, 0.199915, 0.056937, 0.179882, -0.088739, -0.155466, 0.049407, 0.095066, 0.211715, -0.163421, -0.137124, 0.017917, 0.080203, 0.123866, 0.093189, 0.116193, 0.050044, -0.011514, -0.161967], [0.203023, -0.003529, 0.055452, 0.126259, 0.248008, 0.147404, -0.004217, -0.026225, 0.088272, 0.147903, 0.042292, -0.02415, 0.109688, 0.044389, 0.449519, 0.070259, 0.275484, 0.142368, 0.188982, 0.06716, 0.016777, 0.017328, 0.28238, 0.027042, 0.279182, 0.136937, 0.07682, 0.00267, 0.308466, -0.0154, 0.048361, 0.286137], [-0.109575, 0.062342, 0.045787, 0.143829, -0.111203, -0.000221, -0.132928, -0.068022, -0.06123, 0.158007, -0.000301, -0.167637, -0.065857, 0.100933, -0.596476, -0.004394, 0.145377, 0.120775, 0.028139, -0.09966, -0.120799, -0.107428, -0.016848, -0.007139, 0.054297, 0.028046, 0.139263, 0.049221, -0.236249, -0.047593, -0.01649, 0.148824], [-0.018211, 0.078438, 0.130413, -0.031427, 0.085929, -0.178222, -0.224525, -0.014179, 0.079339, 0.080469, 0.126543, 0.115292, -0.069383, 0.114068, 0.112758, -0.296828, 0.037115, -0.05615, 0.179269, 0.037614, 0.078292, 0.000294, 0.084539, 0.02227, -0.037144, -0.240392, -0.002023, 0.111705, 0.036259, -0.026255, 0.15591, 0.051293], [0.182287, -0.089302, 0.055445, -0.05216, 0.03027, 0.109401, 0.109381, 0.097178, 0.114518, -0.211755, 0.029026, 0.110269, -0.036965, 0.074494, 0.052696, 0.061915, 0.207681, 0.074493, 0.265599, 0.00972, 0.142559, 0.089689, 0.05312, 0.082076, 0.057821, 0.130155, -0.095481, 0.063414, 0.070469, 0.258419, 0.01956, 0.018068], [0.006274, 0.080152, 0.016376, -0.008605, 0.107536, -0.162083, -0.045399, -0.022358, -0.070388, 0.025519, 0.042384, 0.088966, 0.0376, 0.096713, 0.210911, -0.001528, 0.044956, 0.017924, -0.061834, 0.142527, 0.188985, 0.050146, 0.09354, -0.059557, -0.248159, -0.038343, -0.000565, 0.05805, 0.135233, -0.14467, 0.132337, 0.115361], [-0.222955, -0.105912, -0.115902, -0.055167, -0.055058, -0.312061, -0.360098, -0.098045, 0.022588, 0.007898, -0.058791, 0.154552, -0.224887, 0.076632, -0.107709, -0.302821, -0.152609, -0.219699, 0.301791, -0.283721, 0.081621, 0.043, 0.007963, 0.506271, 0.122297, -0.339289, -0.007901, 0.080655, -0.085833, -0.173571, -0.213222, -0.280083], [-0.393124, 0.232339, -0.002416, 0.207218, -0.138369, -0.11095, 0.010062, -0.064898, -0.08996, 0.002374, -0.043437, -0.134724, 0.2939, 0.148553, -0.08881, 0.048742, -0.000567, -0.100088, 0.0511, 0.026327, -0.016783, 0.110386, -0.059339, 0.100786, -0.117645, -0.029171, -0.033857, -0.039828, 0.222408, 0.204511, 0.165044, 0.320888], [0.252956, 0.008328, 0.013705, -0.008681, 0.018863, 0.113247, 0.022334, 0.068155, 0.007724, -0.104644, -0.118236, -0.046388, -0.058966, 0.133029, -0.042371, 0.015489, 0.016319, 0.127708, 0.060575, -0.075612, 0.028129, 0.03514, 0.087541, -0.276578, 0.018497, 0.119407, -0.038932, 0.042882, -0.053212, 0.058021, -0.133675, -0.124489], [-0.148779, 0.050939, -0.037083, -0.150899, 0.056964, 0.007315, -0.071275, -0.027565, -0.110519, 0.187025, -0.019503, -0.173121, -0.129436, -0.017645, -0.029436, -0.168076, 0.038449, -0.211872, -0.054459, -0.143547, 0.068343, 0.108419, 0.035267, -0.11461, 0.094131, -0.073518, 0.052521, 0.016624, -0.069419, -0.214841, 0.03468, -0.02193], [0.205583, 0.108746, 0.021823, 0.043752, 0.066939, 0.118189, 0.168784, -0.023156, -0.020145, 0.176343, 6.7e-05, -0.281425, -0.122755, -0.004142, 0.125547, 0.082361, 0.064206, 0.073751, -0.072765, 0.031864, 0.199992, 0.059256, -0.173532, -0.341498, 0.182914, 0.033223, -0.118114, 0.148317, 0.151069, 0.106912, 0.158236, 0.094892], [0.139832, -0.055397, 0.129001, -0.072417, 0.10974, 0.070005, -0.087262, 0.067457, -0.059256, 0.159118, -0.042812, 0.164581, 0.151286, 0.143026, 0.176723, -0.172239, 0.043562, 0.319444, 0.130056, 0.161119, 0.03222, 0.071488, -0.071161, -0.038388, 0.201119, -0.155877, 0.037121, -0.009321, 0.101982, 0.03819, 0.055194, 0.105093], [-0.291388, -0.002148, 0.080826, 0.099173, -0.112514, -0.075372, -0.008756, 0.074392, -0.000614, 0.083352, -0.015093, 0.065575, 0.016768, -0.099549, -0.410645, -0.013366, -0.131584, -0.127971, -0.078077, -0.149636, 0.064617, 0.089413, 0.061246, 0.109669, -0.069966, 0.061483, -0.049795, 0.104622, 0.072338, 0.018163, -0.059644, 0.081376], [0.162253, -0.110265, 0.047919, 0.044791, -0.161517, 0.001775, 0.098978, 0.011151, 0.008051, -0.044533, -0.025618, -0.057009, 0.105291, -0.030834, -0.037229, -0.043845, -0.034243, 0.092558, -0.119872, 0.154301, -0.148943, -0.165236, 0.125468, 0.001302, 0.10259, 0.059071, 0.012284, -0.008397, -0.153059, 0.144602, -0.102593, 0.089529], [0.000978, -0.102169, 0.082534, -0.072299, -0.221786, 0.060758, 0.167943, 0.016751, -0.093477, 0.044039, 0.078665, 0.004689, -0.036042, 0.028293, -0.441974, -0.138203, 0.094796, -0.073792, -0.017407, 0.026065, -0.151562, -0.077657, -0.082993, -0.120177, 0.060283, 0.083518, -0.070233, -0.121254, -0.435696, -0.149552, -0.084978, -0.238648]], [-0.132691, 0.13069, 0.067112, 0.096591, 0.008805, 0.066987, 0.118492, 0.102196, -0.040939, -0.107185, 0.085979, -0.165935, 0.004374, 0.006252, -0.11942, 0.146511, 0.116654, 0.001878, -0.167246, -0.009038, 0.145619, 0.073546, 0.055452, -0.27498, -0.19733, 0.12675, 0.076055, 0.016317, -0.068519, -0.093648, 0.090887, 0.017497]),
    ([[0.345983], [-0.125179], [-0.073531], [-0.020711], [-0.280599], [0.099485], [0.254441], [0.171372], [0.255613], [-0.260052], [-0.104028], [0.191411], [-0.08051], [0.131313], [-0.367307], [0.282968], [0.202918], [0.242533], [0.091561], [-0.231661], [-0.085523], [-0.111645], [0.116327], [0.347273], [0.303128], [0.085997], [-0.073274], [0.061128], [-0.13809], [0.232706], [-0.181779], [-0.157622]], [0.106806]),
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
        'gate_thr': 14.0,
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
        # exp_r3_hyst (all off = base): hysteresis after a flip, direction memory, asymmetric re-entry take
        'hy_ms': 50.0, 'hy_k': 0.2, 'hy_kq': 0.0, 'hp_ms': 0.0, 'ad_k': 0.0, 'hy_only': 1,
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
        'rt_ms': 150.0,
        'ri_ms': 0.0,
        'qa_max': 0,
        'fq_k_sp2': 1.5,
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
        self.hy_us = int(float(params['hy_ms']) * 1000.0)
        self.hy_k = float(params['hy_k'])
        self.hy_kq = float(params['hy_kq'])
        self.hp_us = int(float(params['hp_ms']) * 1000.0)
        self.ad_k = float(params['ad_k'])
        self.hy_only = int(params['hy_only'])  # 0 any flip, 1 only after a passive flip, 2 only after a taker flip
        self.hy_side = 0  # last nonzero inventory sign
        self.hy_t = -(1 << 60)  # time of the last sign flip
        self.hy_dir = 0  # sign we flipped to
        self.hy_ioc = False  # the flip was an IOC take
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
        hy_on = self.hy_us > 0 or self.hp_us > 0
        if hy_on and self.hy_dir != 0:
            age = now - self.hy_t
            if age < self.hy_us and self.hy_kq > 0.0 and (self.hy_only == 0 or (self.hy_only == 2) == self.hy_ioc):
                if self.hy_dir < 0:
                    kb += self.hy_kq
                else:
                    ka += self.hy_kq
            if age < self.hp_us and self.hy_ioc:
                if self.hy_dir < 0:
                    kb += 100.0
                else:
                    ka += 100.0
        dbid = floor(f - kb)
        dask = -floor(-(f + ka))
        tk = self.fq_take_k
        if tk > 0.0:
            ft = f
            if self.pred_on and self.pred_mode:
                ft = mid + (self.pred_scale * self.pred_sig if self.pred_mode == 1 else self.fq_scale * self.fair_sig) + fshift
            tkb = tks = tk
            if self.hy_dir != 0 and now - self.hy_t < self.hy_us and self.hy_k != 0.0 and (self.hy_only == 0 or (self.hy_only == 2) == self.hy_ioc):
                if self.hy_dir < 0:
                    tkb = tk + self.hy_k
                else:
                    tks = tk + self.hy_k
            if self.ad_k > 0.0:
                if self.inv > 0:
                    tkb = self.ad_k
                elif self.inv < 0:
                    tks = self.ad_k
            take_buy = floor(ft - tkb) >= ba
            take_sell = -floor(-(ft + tks)) <= bb
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
            ns = 1 if self.inv > 0 else (-1 if self.inv < 0 else 0)
            if ns != 0:
                if self.hy_side != 0 and ns != self.hy_side:
                    self.hy_t = self.of_now
                    self.hy_dir = ns
                    self.hy_ioc = order.ioc
                self.hy_side = ns
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
