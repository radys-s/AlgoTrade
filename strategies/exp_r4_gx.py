"""mm_r4_ins (home, R4 insurance): mm_r3_gru3 with every learned table refitted on round4 + mirror first 75 % (research.retrain, GRU seed 2): round4 1,060,473 / mirror 1,045,062; holdout 211,755 / 209,103 (mm_r3_gru3 unrefitted: 1,027,483 / 1,036,706; holdout 189,536 / 203,350).

exp_r4_auto: mm_r3_gru3 with the learned tables (ridge,mlp,mq,slow,seq) refitted by research/retrain.py on round4.csv + mirror (first 75 %).

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
    ([[-0.089962, 0.071676, 0.088856, 0.011464, 0.188966, 0.267389, 0.314488, 0.086875, 0.133872, -0.054612, 0.129885, -0.037057, 0.085819, -0.009956, 0.320303, -0.217636, -0.039207, -0.43886, -0.059913, -0.100801, -0.059342, -0.343111, -0.029467, 0.004529, -0.048367, -0.20405, 0.111709, 0.199453, -0.125659, 0.226796, 0.216135, -0.118335], [-0.102902, 0.24559, -0.000929, -0.064679, -0.15012, -0.065132, -0.109088, -0.261142, -0.083204, 0.150888, -0.274254, -0.123232, 0.141765, -0.088469, 0.388418, -0.045825, -0.013092, -0.041389, 0.175862, -0.331834, -0.090137, -0.102053, 0.173239, -0.030476, -0.222402, 0.073855, -0.135047, 0.048498, -0.257753, -0.048489, 0.17491, -0.184788], [0.036004, 0.085467, -0.019277, 0.076447, -0.14399, 0.011547, 0.014398, -0.1834, -0.001407, -0.09308, 0.028406, 0.20059, -0.129351, 0.105731, 0.235186, 0.008208, 0.098029, -0.139835, 0.152386, 0.040211, 0.030399, 0.027189, 0.239553, 0.076118, 0.049421, 0.218143, -0.024327, 0.199249, -0.099198, -0.202867, -0.219193, -0.031398], [0.101172, 0.089781, -0.126779, 0.118935, -0.257823, 0.152781, -0.131412, 0.104399, 0.181096, 0.051238, 0.178275, 0.01551, -0.282567, -0.188408, -0.41647, 0.039664, 0.026291, 0.119126, 0.129507, 0.04179, 0.094929, -0.164789, 0.167672, 0.191705, 0.133311, 0.181784, -0.016413, 0.141997, 0.089007, 0.169191, -0.158809, 0.138646], [0.270521, 0.025846, 0.133787, -0.325331, 0.041973, -0.473318, -0.257824, 0.465915, -0.240773, -0.194882, -0.161834, -0.392572, 0.005472, -0.110639, -0.477257, -0.402242, -0.107429, -0.052575, 0.097933, 0.642674, -0.360747, 0.016253, 0.297683, 0.054407, 0.242243, -0.341264, 0.292866, 0.078587, -0.153993, -0.094364, -0.411528, 0.492584], [0.034649, -0.048402, 0.023962, 0.256311, -0.102866, -0.253661, -0.343926, 0.078197, -0.266941, 0.073406, -0.129997, 0.227176, 0.220091, -0.110719, 0.014614, 0.041224, -0.167173, 0.263146, 0.06885, -0.254481, 0.06685, -0.021455, 0.213877, -0.096814, 0.00395, -0.121895, -0.045083, 0.250277, 0.014816, -0.198848, -0.178523, -0.230652], [-0.061836, -0.22961, 0.207836, -0.22327, 0.320167, -0.099177, -0.09362, -0.175668, -0.316686, -0.447099, -0.058628, -0.584306, -0.335734, -0.187834, 0.357512, 0.024207, 0.540562, 0.231203, -0.358895, -0.292817, 0.145168, -0.006968, -0.214811, 0.297062, 0.234121, 0.16562, 0.420118, -0.15642, -0.043569, 0.458055, -0.003936, -0.02091], [-0.580289, -0.066732, 0.008257, 0.364866, 0.177612, 0.14222, 0.176611, 0.033115, -0.070205, 0.052028, 0.079636, -0.071416, 0.072362, -0.265633, -0.075685, -0.108429, 0.126501, -0.01213, 0.154202, 0.085309, 0.035911, 0.169567, 0.059427, 0.149624, 0.155532, 0.102662, -0.046413, -0.028972, 0.051631, -0.190055, 0.025966, -0.223394], [-0.232644, -0.2492, 0.464198, 0.297613, -0.100646, -0.371575, 0.02403, 0.438682, -0.301287, -0.322173, 0.044206, -0.53591, -0.038958, 0.328356, -0.444613, -0.203425, -0.271056, 0.155262, -0.402131, -0.413294, 0.099844, -0.434435, 0.110854, 0.12439, -0.122376, -0.083581, 0.096685, 0.096898, -0.397924, -0.061673, 0.275281, 0.640593], [-0.091171, -0.636743, 0.291846, -0.158396, -0.245109, 0.251333, 0.03289, -0.049968, 0.000199, 0.051313, -0.249775, -0.194953, 0.105525, -0.356097, -0.011955, 0.046214, -0.261826, 0.129925, -0.191846, -0.006944, -0.007104, 0.101131, -0.156719, -0.213236, 0.053204, 0.173509, 0.155346, 0.239906, -0.537444, 0.401803, -0.181451, -0.136187], [0.036905, -0.017074, -0.023679, -0.069212, -0.205748, 0.120716, 0.327626, -0.16731, -0.078223, -0.027409, 0.027684, -0.088541, 0.013484, 0.06151, -0.005702, 0.232783, -0.075434, 0.018523, -0.038155, 0.122852, -0.061139, 0.233475, 0.141579, 0.094716, 0.11365, 0.193274, -0.027804, 0.216412, -0.055063, 0.191127, 0.065462, -0.081518], [-0.045967, 0.03241, 0.212961, -0.030368, -0.219951, -0.286899, 0.417356, 0.218492, -0.010065, -0.007133, 0.008096, -0.325341, -0.041446, 0.161771, 0.242152, -0.248244, -0.380046, 0.243826, -0.251115, -0.069183, 0.11758, -0.31726, -0.175384, 0.565426, -0.078215, -0.216322, 0.191506, -0.031085, -0.052599, -0.130409, -0.053986, 0.131113], [0.114936, -0.119373, -0.158327, -0.034657, -0.154365, -0.069426, 0.060829, 0.021474, 0.293874, -0.203273, -0.017543, -0.141167, 0.174378, 0.460189, -0.104291, 0.182113, 0.126097, -0.168164, 0.249372, 0.018838, 0.040596, 0.113785, 0.009048, 0.313659, -0.019012, 0.063026, -0.026789, 0.06529, 0.126471, 0.254981, 0.035797, -0.015364], [0.34822, 0.056826, 0.086546, 0.072823, -0.050809, 0.171125, -0.029502, 0.217394, 0.047927, -0.045105, -0.126559, -0.039124, 0.210689, 0.363804, 0.090129, 0.099882, 0.209276, -0.271456, 0.464499, 0.163066, -0.033996, -0.120732, -0.271241, 0.032251, 0.030683, -0.114168, -0.259815, -0.132626, 0.029837, 0.217245, -0.014315, -0.002332], [0.282858, 0.301311, -0.118654, 0.054043, -0.427624, 0.289172, 0.164763, 0.031301, 0.299044, -0.195009, -0.327757, 0.03553, -0.210786, -0.048647, 0.036709, 0.27148, 0.160869, 0.15773, -0.196113, 0.076104, -0.293057, -0.231811, -0.012122, -0.36513, -0.489261, -0.035809, 0.160297, 0.002431, 0.071263, -0.018617, 0.032206, -0.233878], [0.005362, 0.003389, 0.086681, 0.239501, 0.055698, -0.315279, 0.185012, -0.077927, -0.205588, -0.048187, -0.200643, 0.067489, 0.298011, -0.226308, 0.044141, 0.029703, -0.148786, 0.085515, -0.218057, -0.305268, 0.200579, 0.118367, -0.199794, 0.033181, 0.053708, -0.100213, -0.225993, 0.052164, 0.14472, -0.057044, 0.126971, 0.035645], [-0.304622, -0.471118, 0.437666, -0.507329, 0.439246, -0.228926, 0.249459, 0.070846, 0.167477, 0.094623, 0.0066, 0.516924, 0.293368, 0.043234, 0.14712, -0.392171, -0.385753, -0.141776, 0.576312, 0.006367, -0.437061, 0.024283, 0.23431, 0.443216, -0.353457, 0.014877, -0.213233, -0.159259, -0.08601, -0.507555, -0.101624, 0.357582], [-0.072117, -0.31592, -0.237855, 0.088514, -0.028887, 0.226691, -0.233849, 0.472939, 0.009225, 0.102906, -0.115077, 0.005045, 0.089407, -0.370425, -0.53479, -0.173432, 0.006755, 0.250737, 0.161426, 0.249139, 0.074915, -0.11766, 0.490973, 0.106767, 0.038639, 0.027074, -0.134749, -0.045342, -0.013957, 0.04639, 0.470051, 0.041734], [0.123325, -0.041342, 0.014534, 0.284976, 0.159482, 0.013985, -0.052853, 0.199273, -0.039528, 0.011156, -0.084459, 0.034244, 0.039698, -0.046772, 0.019402, -0.098618, 0.00073, -0.102615, -0.075834, -0.070243, 0.039888, 0.047009, 0.033793, -0.15691, 0.044974, -0.051207, 0.018089, -0.00485, 0.165962, -0.02533, -0.003002, 0.121324], [0.240993, 0.148178, 0.159328, -0.137395, 0.201965, -0.36697, -0.115851, 0.139567, -0.24091, 0.163309, -0.387723, -0.090127, -0.126576, 0.136945, -0.055233, -0.001586, -0.198145, -0.211304, -0.06359, -0.450822, 0.460753, 0.168758, 0.052183, 0.319074, 0.702598, 0.086526, -0.014461, 0.040407, -0.267732, 0.021437, -0.121594, 0.101632], [-0.382984, 0.1649, 0.299605, 0.203085, -0.197667, 0.057709, -0.035022, -0.210921, 0.116261, 0.09742, 0.002051, 0.192115, -0.096023, 0.494463, -0.033445, 0.042531, 0.055574, 0.345467, -0.293271, -0.201078, 0.148855, -0.049946, -0.220067, -0.015709, -0.116926, -0.306297, -0.197636, 0.006148, 0.039292, 0.031164, 0.312956, -0.164906], [0.032163, 0.03067, 0.090235, -0.112208, -0.068562, -0.010277, 0.065168, -0.035514, -0.1565, 0.790293, 0.101466, 0.578844, -0.551932, -0.050626, -0.032238, -0.161629, -0.464671, -0.164676, 0.094794, -0.129138, -0.000647, 0.075863, -0.049422, -0.109879, -0.014828, -0.062967, 0.641412, -0.049106, 0.034163, -0.010114, -0.054157, 0.083797]], [-0.234445, 0.099795, -0.16079, -0.028827, 0.245438, 0.034169, 0.010354, -0.210037, 0.305644, -0.270991, 0.33408, 0.026519, -0.125302, 0.320126, -0.046415, -0.287816, 0.013544, 0.243424, 0.192331, -0.570373, -0.029512, 0.006022, -0.421865, -0.550891, 0.257785, -0.500393, -0.127368, -0.186365, 0.225873, 0.214236, -0.173442, -0.126699]),
    ([[0.032249, 0.007262, -0.17206, -0.352592, 0.013006, -0.119119, -0.058078, -0.748441, -0.220723, -0.25578, 0.046782, 0.113285, 0.029745, -0.210351, -0.20782, -0.6731], [-0.515443, 0.111253, -0.074504, 0.122065, -0.140235, -0.16076, 0.007417, 0.30784, -0.030634, 0.193386, 0.244787, -0.01925, 0.208215, -0.443021, 0.048415, -0.662306], [-0.179222, 0.124296, -0.343083, 0.128812, 0.163583, -0.178166, 0.131495, -0.040668, 0.21976, -0.062384, 0.031248, 0.11014, -0.128779, -0.543731, 0.349675, 0.101553], [-0.418956, -0.255862, -0.37332, -0.185457, -0.150376, 0.166875, 0.078387, -0.375334, -0.187104, 0.140928, 0.057485, 0.00804, -0.53657, -0.336239, -0.250286, -0.46942], [0.002154, 0.218319, -0.02388, 0.283764, 0.066198, 0.014506, 0.121675, 0.298712, 0.301305, -0.043745, 0.196959, 0.199353, 0.294514, 0.159398, -0.034325, -0.267241], [0.000547, -0.102192, -0.10184, -0.260297, -0.051191, -0.134117, -0.188087, -0.259177, -0.422208, -0.243784, 0.139737, 0.167034, 0.173203, -0.005621, -0.598117, -0.310424], [-0.194063, -0.041759, 0.225849, -0.306623, -0.254616, -0.032812, 0.065806, -0.059998, -0.318154, -0.132099, 0.179734, -0.031484, -0.229978, -0.345665, -0.482797, -0.203005], [-0.022779, -0.145, 0.485038, 0.231342, 0.326467, -0.123439, -0.130482, 0.11303, 0.004314, 0.180631, 0.192351, -0.350127, -0.047395, 0.392827, 0.268205, -0.314509], [0.034101, -0.15812, 0.110582, -0.156833, -0.081929, 0.089591, 0.135485, -0.298212, -0.414635, -0.210785, -0.028527, -0.083653, -0.042215, -0.152065, -0.54544, 0.192197], [0.121017, 0.22674, 0.086862, 0.295928, 0.09461, -0.377363, 0.079382, -0.185051, -0.124164, 0.046249, -0.13618, 0.139017, 0.222171, 0.221649, -0.12626, 0.314871], [0.099214, 0.075325, 0.198777, -0.269691, 0.04041, -0.015861, -0.140383, -0.276143, -0.502873, -0.169512, 0.151561, 0.007609, 0.149566, 0.178903, -0.648063, -0.029569], [-0.087239, -0.111048, -0.109737, -0.09699, 0.069089, -0.063573, 0.194915, -0.618376, -0.024735, -0.404594, -0.233924, 0.051038, -0.17747, -0.011965, -0.232775, 0.298501], [-0.085339, 0.039046, -0.411377, 0.260906, -0.32728, 0.135068, -0.114144, 0.261857, 0.170213, -0.24407, -0.277483, -0.245573, -0.152429, -0.14057, -0.165592, -0.452554], [-0.21403, -0.121609, -0.289261, 0.142095, 0.103835, 0.130284, 0.082723, -0.75683, -0.00943, -0.142139, -0.242905, -0.362525, 0.009534, -0.011698, -0.114883, -0.091231], [0.226984, 0.115472, 0.190149, 0.302865, 0.288014, -0.125444, -0.213425, 0.112155, 0.249586, 0.553002, -0.039604, 0.193076, 0.220772, 0.186632, 0.318595, -0.056211], [0.114901, 0.123616, -0.094221, 0.114483, 0.022981, 0.114173, -0.126092, 0.103871, -0.079825, 0.442085, 0.138538, 0.186344, 0.096293, 0.115787, 0.296768, 0.249033], [0.013895, 0.036567, -0.092247, -0.226785, 0.249313, 0.186673, -0.13469, -0.152576, -0.076694, -0.026374, -0.184652, 0.269736, -0.001729, 0.118811, 0.150371, 0.239675], [0.331999, 0.119468, -0.075342, -0.03664, -0.129685, -0.049557, -0.145402, -0.141327, 0.265893, 0.063087, 0.099778, 0.092569, 0.1927, 0.25979, 0.355968, 0.16648], [-0.292635, -0.048814, -0.09897, -0.144634, -0.183135, 0.076003, 0.085594, -0.227734, -0.040431, 0.030026, -0.225642, -0.14032, -0.219556, -0.274688, 0.139361, -0.329167], [-0.201468, 0.032149, 0.002718, -0.857334, 0.348105, 0.022822, 0.099794, -0.330075, -0.007625, 0.029069, 0.022434, -0.08759, -0.267794, 0.083701, 0.209052, 0.202731], [0.130767, -0.029811, -0.260633, -0.226747, -0.419615, -0.190597, 0.236864, 0.011343, -0.026694, -0.264834, -0.152517, -0.053905, -0.37528, 0.020581, -0.22875, 0.149585], [-0.011227, -0.00895, 0.069119, 0.469982, 0.124413, 0.008546, 0.089136, 0.214863, 0.186889, -0.017404, 0.151284, 0.088948, 0.214438, 0.050971, 0.26229, 0.092581], [-0.000878, -0.02705, 0.076931, 0.028662, 0.378875, 0.012073, -0.550542, 0.215937, 0.216963, 0.070995, -0.053571, -0.11053, 0.31893, 0.199134, 0.000533, -0.031843], [-0.431203, 0.086748, 0.20765, 0.249704, 0.225892, -0.085968, -0.260514, 0.071027, -0.020131, 0.096797, 0.360237, -0.240912, -0.313241, 0.100604, 0.156286, 0.029093], [-0.127261, -0.33493, 0.059178, 0.00259, 0.145823, 0.352954, 0.002821, 0.079307, -0.197514, 0.1505, -0.030392, -0.164905, -0.638865, 0.140678, 0.075548, 0.261611], [0.278962, -0.234888, -0.002485, -0.421954, 0.16642, 0.076936, 0.247175, -0.029571, -0.382948, -0.462765, -0.067853, 0.119798, 0.030897, -0.15027, -0.194625, -0.302183], [-0.112722, -0.134354, -0.071288, 0.138475, -0.292028, 0.087688, -0.059398, 0.009894, 0.183452, -0.037564, 0.29486, -0.103573, -0.185907, -0.211146, -0.20711, -0.221882], [0.244646, -0.016793, -0.375277, 0.100461, -0.11161, -0.19246, 0.175082, 0.004744, 0.185247, -0.023576, 0.170136, 0.140878, 0.090738, -0.064292, -0.028869, 0.217794], [0.215721, 0.057626, -0.104621, -0.222249, 0.180289, 0.019491, -0.037452, -0.530809, -0.300908, -0.213075, 0.204083, -0.006017, 0.043966, 0.095102, 0.184234, 0.160759], [0.062482, 0.057109, 0.127533, -0.230331, -0.093226, 0.164313, -0.033276, 0.419763, 0.115383, -0.208568, -0.143458, 0.341868, 0.140621, -0.024353, 0.243419, -0.038024], [0.367, -0.015241, 0.15114, 0.46193, 0.107623, -0.086542, 0.069133, -0.290564, 0.01306, 0.745471, -0.355838, -0.101514, 0.218096, 0.12986, -0.095527, -0.03079], [0.166667, -0.158874, 0.32757, 0.182931, 0.125651, -0.012773, 0.115027, 0.13591, 0.200343, 0.380473, 0.177946, 0.086432, 0.24283, 0.274347, 0.19671, -0.093317]], [0.020552, 0.463364, 0.152127, -0.183592, -0.44268, 0.20316, 0.186495, 0.045338, 0.350168, 0.039488, 0.012778, -0.039163, -0.051064, -0.140448, 0.004753, -0.355939]),
    ([[-0.791889], [-0.248185], [-0.458888], [0.684866], [-1.056487], [-0.200262], [-0.314857], [0.728128], [0.575355], [0.731124], [0.310544], [-0.549394], [-0.649493], [-0.327337], [0.623176], [-1.108892]], [0.279174]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.562321557048493e-19, -1.1343482335572739e-18, -5.671741167786369e-19, -1.8905803892621232e-19, 1.1343482335572739e-18, 0.0, 0.0, 0.0, 1.2572534685440513, 0.0, 10.639801906763806, 28.799029822078616], [6.72849538126497, 19.8058956257327, 34.925161910299096, 44.72666958616459, 57.636846319296275, 1.1243487056677264, 0.9903909371630099, 0.9446913427593254, 0.8452682282888074, 0.7575381583058208, 1067.5606541222405, 2053.035725905977, 4046.2991308026644, 0.6179080587708422, 0.3127841039679992, 16.705331990766382, 34.22120473597514], [0.6490227662551226, 0.3600762349252582, -0.1832944003136108, -0.09733275171505404, -0.10066480452612983, 0.8482856190320313, -3.2486297440276823, 6.1619984101404714, -3.540934484781457, 3.1035095898512166, -0.6019915249010418, 0.06557241931064624, -0.1855095648918967, -0.002102755891520747, 0.4105229362630337, 0.0003153876484541001, 0.00020715056621090425], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.568942287397852e-19, -1.1353413431096778e-18, -5.676706715548389e-19, -1.892235571849463e-19, 1.1353413431096778e-18, 0.0, 0.0, 0.0, 1.2574354160321035, 0.0, 10.646654416476506, 28.81409587438229], [6.7313766144182345, 19.814344420832445, 34.939862329400235, 44.74491730719747, 57.65822173996363, 1.124692105470714, 0.9907164467252751, 0.9450073668099886, 0.8455601770714993, 0.7578101560190531, 1067.786637740178, 2053.787560401509, 4047.179122347928, 0.6181145992138308, 0.3129209935553393, 16.710984248831306, 34.23237508220454], [2.9421164343350763, 0.7964360421909548, -1.0744230210945729, -0.9234918162695617, -0.8475723430517554, 1.4775048291769728, -4.635873707445536, 9.213729195881914, -4.907517756528247, 6.23467901404692, -2.386919742393157, 0.4346797676233983, -1.5368709559241014, 0.001829056109893383, -0.22349228908983945, -0.0002742367847445596, -0.00018005766511644536], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.257441, 0.0, 0.0, 5.84623, 5.846549, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987547], [1.124422, 0.990545, 0.944803, 0.844787, 0.75662, 2.57683, 18.120787, 0.790527, 15.924298, 0.695578, 0.651228, 0.54053, 1.677429, 1.677372, 0.999993, 4.403117, 1.087628, 0.310614, 132.397858, 334.626434, 1066.212891, 1.340704], [([[0.046726, -0.070283, 0.041148, -0.121056, 0.039105, 0.323323, 0.294197, 0.053627, 0.087933, 0.485461, 0.09108, 0.010768, 0.086605, 0.020133, 0.033459, -0.19678, -0.051887, -0.327683, -0.091245, 0.059482, -0.026663, -0.468369, -0.145178, -0.091014, -0.119923, 0.013569, -0.033703, -0.074339, -0.111327, 0.308164, 0.131393, -0.01286], [-0.018247, 0.179254, 0.108377, -0.312966, -0.144238, -0.159034, 0.044336, -0.066689, -0.134883, 0.383292, -0.267714, -0.163974, 0.223166, -0.054344, 0.124266, -0.049885, -0.163316, -0.085414, 0.08333, -0.043674, -0.0897, -0.166446, 0.097537, -0.092006, -0.308528, -0.134043, -0.0985, 0.004593, -0.136702, 0.04477, 0.126643, 0.11769], [0.060343, 0.034685, 0.051102, -0.004343, -0.114597, -0.04816, 0.164206, -0.050273, -0.022993, -0.039793, 0.077041, 0.165135, -0.130236, 0.152316, 0.093886, 0.040422, 0.012192, -0.195665, 0.068724, 0.162412, 0.029286, 0.030016, 0.188276, 0.005894, 0.062535, 0.05318, 0.068192, 0.237619, 0.037826, -0.117224, -0.216425, 0.219775], [0.056784, 0.063706, -0.23288, 0.325502, -0.303094, 0.342989, -0.105093, -0.128217, 0.2161, -0.281962, 0.321468, 0.088406, -0.429037, 0.000635, 0.054878, 0.156312, 0.198334, 0.229963, 0.236295, -0.212359, 0.261556, -0.031812, 0.102016, 0.171193, 0.345254, 0.402714, -0.003717, 0.301961, 0.134332, -0.058102, -0.033585, -0.012858], [-0.053472, 0.08487, 0.166099, -0.325135, 0.042199, -0.494835, -0.160986, 0.238469, -0.376027, -0.355049, -0.098431, -0.265254, -0.005909, -0.616651, -0.00694, -0.417258, 0.127451, -0.027628, 0.374165, 0.465594, -0.52367, 0.124393, 0.17643, 0.140068, 0.335121, -0.378651, 0.400159, 0.102075, -0.275253, 0.051326, -0.52399, 0.448393], [0.04383, 0.047423, 0.183752, 0.186189, -0.116421, -0.245354, -0.187333, 0.202777, -0.249675, -0.318201, -0.205374, 0.124098, 0.209447, 0.106466, 0.146861, -0.005395, -0.238684, 0.250551, -0.037352, -0.183619, -0.017429, -0.076869, 0.333949, -0.034592, 0.119412, -0.267705, 0.095111, 0.292149, 0.041204, -0.296508, -0.180482, 0.031922], [-0.04407, -0.073493, 0.030897, 0.210464, 0.123824, 0.074499, -0.013934, -0.026732, -0.253624, 0.505973, -0.00436, -0.777064, -0.286892, -0.201469, 0.010768, 0.166735, 0.624984, 0.49835, 0.424553, 0.003668, -0.665699, -0.147602, -0.224333, 0.12273, 0.370232, 0.258959, -0.191718, 0.003982, 0.057753, 0.502685, -0.114171, -0.115123], [-0.068748, -0.067945, 0.126717, 0.12799, -0.00631, 0.095026, -0.126606, 0.094097, -0.005338, 0.031623, 0.02333, -0.017556, 0.095651, 0.065094, -0.185893, -0.064115, -0.070795, -0.06781, 0.065334, -0.306332, -0.045535, 0.030989, 0.089167, -0.127908, 0.172704, -0.00725, -0.029574, 0.103892, 0.053858, 0.029602, 0.127946, -0.348197], [-0.00636, -0.062375, -0.282531, 0.147843, -0.358343, -0.152506, -0.307282, 0.201657, -0.401413, 0.377585, 0.19533, -0.417243, 0.128938, 0.441617, 0.001544, 0.004125, 0.237706, 0.198203, -0.66402, -0.423367, 0.420114, -0.433128, -0.138937, -0.161427, 0.052746, -0.070181, -0.130435, -0.059375, 0.018565, 0.118558, 0.061307, 0.295071], [-0.001725, -0.403287, 0.208408, 0.047313, -0.128452, 0.158715, 0.091097, -0.574843, -0.16251, -0.298903, -0.222127, -0.214706, 0.027656, -0.088885, -0.651408, -0.046781, -0.441506, 0.280515, -0.038066, 0.030339, 0.07909, -0.068275, -0.115343, -0.426609, 0.307031, 0.04441, -0.134621, 0.386591, -0.334403, 0.385677, 0.047022, 0.010233], [0.083363, 0.014196, -0.118503, -0.158549, -0.275936, 0.036007, 0.168695, -0.05803, -0.084412, 0.016763, -0.074895, 0.004966, 0.04377, 0.053899, 0.066311, 0.342986, -0.03009, 0.018228, 0.098999, 0.129944, -0.168558, 0.127362, -0.250269, -0.059566, 0.101194, -0.009494, -0.051472, 0.291634, -0.246393, 0.114293, -0.009125, 0.068291], [-0.097286, 0.176365, 0.146839, -0.132834, 0.188411, -0.094171, 0.392765, 0.266239, 0.151958, 0.165952, 0.190914, -0.389984, 0.080555, 0.041021, 0.237496, -0.502079, -0.37039, 0.181902, -0.407467, -0.067079, 0.207641, -0.263687, 0.204893, 0.482353, -0.084624, -0.190223, -0.130918, -0.143864, 0.053511, 0.097594, -0.082236, 0.075682], [-0.016879, 0.323432, -0.182702, 0.005552, -0.219787, -0.008781, 0.004571, -0.087638, 0.324135, -0.07908, -0.042053, -0.146782, 0.093997, -0.090584, -0.244659, 0.162149, 0.161379, -0.141971, 0.183166, 0.012815, 0.001334, -0.013777, -0.032322, 0.415655, 0.053596, -0.012791, 0.091039, 0.055558, 0.176856, -0.053558, 0.076861, -0.241239], [0.076282, -0.008022, 0.108856, -0.036803, -0.025875, 0.143641, -0.026587, -0.112541, 0.025854, 0.077376, -0.029039, -0.013879, 0.187862, 0.030281, -0.17116, -0.042914, 0.053773, -0.155706, 0.531015, 0.167942, -0.152778, -0.125183, 0.037548, 0.15689, 0.216465, -0.204053, -0.129169, -0.06824, 0.23032, 0.015629, 0.022928, -0.222082], [-0.134717, 0.119423, 0.009748, -0.202154, -0.031601, 0.381605, 0.410017, -0.221055, 0.062742, 0.178709, -0.132796, 0.058792, -0.222071, -0.086716, -0.183675, 0.113831, 0.134898, 0.124792, 0.107301, 0.23721, -0.128498, -0.086661, -0.045842, -0.097464, -0.283426, -0.238853, 0.0771, -0.011204, -0.074139, 0.181958, 0.017994, 0.189307], [-0.002615, 0.068814, 0.152245, 0.387217, -0.006896, -0.031261, 0.279206, -0.085283, -0.160993, -0.04787, -0.093637, 0.111275, 0.173788, 0.006131, 0.098549, -0.054146, -0.24049, 0.10361, -0.404844, -0.480137, 0.42209, 0.080775, -0.004535, 0.014502, -0.184472, -0.146698, -0.082395, 0.057444, -0.21253, -0.157121, 0.088981, 0.039516], [0.193902, 0.043775, 0.457243, -0.332318, 0.431129, 0.12056, 0.111643, -0.176667, 0.202674, -0.073154, -0.14912, 0.434362, 0.396393, 0.164748, 0.365064, -0.162222, -0.080704, 0.136494, 0.066481, 0.28702, 0.01451, -0.222717, 0.148548, 0.037566, -0.0156, 0.258732, 0.00669, -0.39626, -0.018976, -0.051809, -0.139348, -0.06972], [-0.085148, -0.01064, -0.330562, 0.061945, -0.090622, 0.037454, -0.00437, 0.451453, -0.060044, -0.068937, -0.10815, -0.028768, 0.104916, -0.032837, -0.087458, -0.13677, -0.332158, 0.283849, -0.21746, 0.12286, 0.02463, 0.094012, 0.457237, -0.313342, -0.124047, 0.192418, -0.180637, -0.154261, -0.280198, -0.001562, -0.005561, 0.068909], [0.003288, -0.011157, 0.083008, 0.21288, 0.38949, 0.003184, 0.023515, 0.026219, -0.02147, 0.017094, 0.04904, 0.019519, 0.026988, -0.009773, 0.035622, -0.025586, 0.014561, -0.245233, -0.052759, -0.14415, -0.089255, -0.038812, 0.010373, -0.240546, -0.012976, -0.008921, -0.000704, -0.06154, 0.177034, 0.050122, -0.022034, -0.000263], [0.257348, -0.353959, 0.105577, 0.053881, 0.004454, -0.693114, -0.214876, 0.154533, 0.055779, -0.068189, -0.153909, -0.057551, -0.047738, -0.148359, -0.070534, 0.060537, -0.22444, -0.124562, -0.06612, -0.682951, -0.009443, -0.034987, 0.025266, 0.45936, 0.464008, -0.232687, 0.222798, 0.153073, -0.263571, -0.165113, -0.137804, -0.10479], [-0.701938, 0.409276, 0.378837, -0.19179, -0.159157, 0.004084, -0.23514, -0.039385, 0.106859, 0.1136, -0.149203, 0.061055, 0.056961, 0.668402, -0.077168, -0.134731, -0.184068, 0.176844, -0.180225, -0.094645, 0.165921, 0.188961, -0.138236, -0.14378, 0.058163, -0.029898, -0.740528, -0.143123, -0.162265, -0.349346, 0.290777, 0.078152], [-0.041795, -0.029185, -0.007466, -0.047467, 0.180032, -0.153776, -0.017542, 0.05825, -0.109208, -0.009636, 0.005905, 0.596182, -0.431021, 0.042504, -0.083305, -0.169114, -0.343465, -0.15865, 0.032946, -0.006537, 0.35613, 0.129559, 0.01836, 0.155144, -0.009875, 0.036818, -0.074206, 0.011242, -0.046649, 0.163343, -0.030138, 0.063729]], [-0.445711, -0.312916, 0.02658, -0.21546, 0.123375, 0.113426, -0.028764, 0.130539, 0.370853, -0.181608, 0.329139, -0.138481, -0.323158, 0.255979, 0.13668, -0.096588, -0.02268, 0.240645, 0.233496, -0.343905, -0.111729, 0.069759, -0.344425, -0.097962, -0.55605, 0.103329, 0.315764, -0.31001, 0.014966, 0.281158, -0.212236, -0.012496]), ([[0.32707, 0.196739, 0.167507, 0.177138, 0.120947, -0.039183, 0.173026, -0.279511, -0.472498, 0.038118, 0.3885, 0.169026, 0.203648, -0.018926, 0.444712, 0.433805], [-0.316835, 0.026789, 0.052372, 0.152331, 0.138571, 0.360401, 0.197348, 0.066323, -0.092698, 0.145225, 0.411392, 0.081951, 0.043232, 0.084899, 0.31767, 0.042078], [-0.021481, 0.02808, -0.471527, 0.11871, 0.063415, -0.191772, 0.037611, 0.164537, 0.227075, 0.097513, 0.075234, 0.239797, -0.327381, -0.554826, 0.101539, 0.255342], [-0.078019, -0.018086, -0.136294, -0.281049, -0.361637, -0.123393, 0.100833, -0.391682, -0.336623, -0.091653, 0.092428, 0.012598, -0.554026, 0.188729, -0.377817, -0.077355], [-0.036869, 0.278858, -0.009924, 0.176558, 0.208599, 0.031904, 0.185578, 0.327216, 0.301605, -0.008394, 0.090874, 0.105983, 0.167728, 0.026123, 0.123173, -0.083007], [0.247847, -0.390363, -0.074547, -0.225035, -0.184073, -0.110591, -0.314015, 0.026289, -0.228657, -0.279522, 0.049037, 0.212228, 0.121305, -0.139904, -0.456313, -0.007225], [-0.130652, 0.07673, 0.065021, -0.143774, -0.214663, -0.123672, -0.005458, -0.104838, -0.2411, -0.273383, 0.117748, 0.092342, -0.332343, -0.253886, -0.318681, -0.101854], [-0.08961, 0.057756, 0.041928, 0.119545, 0.068464, -0.132005, -0.146168, -0.133524, 0.024531, 0.191747, 0.162663, -0.111263, -0.644577, 0.065091, -0.039061, -0.496593], [-0.141263, -0.26112, 0.054253, -0.560746, -0.032494, -0.078967, -0.064316, -0.104938, -0.562434, -0.349435, -0.073411, 0.031888, -0.083467, -0.0442, -0.051182, 0.120878], [0.193191, -0.255973, 0.018408, 0.880701, 0.336066, 0.165751, 0.027441, -0.115918, 0.378622, 0.262392, -0.129686, 0.106623, 0.212744, -0.150601, 0.805912, 0.365802], [0.090969, 0.206946, 0.261796, -0.423219, 0.066259, 0.131259, -0.020819, 0.052397, -0.178338, -0.171258, 0.066049, 0.065289, 0.162583, -0.086118, -0.310549, 0.053449], [-0.053247, -0.063315, 0.050376, -0.39466, 0.086858, -0.200668, 0.248913, 0.065811, -0.214231, -0.298657, -0.044444, -0.004158, -0.056758, 0.154601, -0.69074, 0.313708], [-0.187879, 0.022985, -0.160118, 0.444651, -0.542963, 0.109844, -0.576409, 0.028577, 0.038655, -0.075953, -0.052008, -0.294215, -0.046009, -0.192264, -0.073179, -0.63765], [-0.08261, 0.085046, -0.366499, -0.044462, -0.008435, 0.12696, 0.002484, -0.59167, -0.174682, -0.780679, -0.249572, -0.345113, -0.037266, 0.106428, -0.555212, -0.379373], [0.207183, 0.170083, -0.050496, 0.076059, 0.181551, -0.119148, -0.036301, -0.005134, 0.056912, 0.076901, 0.176393, 0.083554, 0.316413, -0.273958, -0.235229, -0.1975], [0.201486, -0.056115, 0.171467, 0.155076, -0.02381, 0.074593, -0.097109, 0.04571, 0.071643, 0.427999, 0.234674, 0.152661, 0.040499, 0.210755, -0.227706, 0.19692], [-0.164169, -0.211617, 0.155825, -0.341786, 0.316197, 0.135655, -0.032904, -0.131762, -0.198623, 0.064986, -0.060951, 0.228016, -0.024909, 0.163242, -0.666375, 0.120125], [0.315172, 0.145899, -0.003727, 0.083122, -0.046754, -0.002887, -0.046507, -0.042386, 0.343091, -0.003922, -0.039618, 0.006314, 0.220972, 0.135908, 0.268153, 0.109555], [-0.200322, -0.12895, -0.076761, -0.032306, -0.421921, -0.140573, 0.154629, -0.351686, -0.221697, 0.035676, -0.435448, -0.195786, -0.107802, -0.166911, 0.061165, -0.404476], [-0.355377, 0.122407, -0.084945, -0.827739, 0.132981, 0.01837, 0.079032, 0.084992, -0.410459, -0.274163, -0.089868, -0.092052, -0.216304, 0.154608, 0.076928, 0.159291], [0.116149, 0.02296, -0.095561, 0.069147, 0.015382, -0.502113, 0.287132, -0.364365, -0.324067, -0.435505, -0.268527, -0.263637, -0.270979, 0.262436, -0.23876, -0.009608], [-0.001077, 0.017951, 0.14405, 0.436041, 0.184164, -0.109113, 0.008642, 0.107939, 0.31287, 0.094694, -0.048924, 0.08314, 0.200211, 0.141879, 0.255333, 0.134957], [0.16825, -0.375908, -0.730111, 0.079182, 0.230666, -0.302576, -0.274343, 0.177149, 0.11784, 0.133233, -0.006977, -0.047954, 0.194586, 0.025137, 0.047121, 0.055607], [-0.245518, -0.238981, 0.279388, 0.026706, -0.207142, -0.023467, -0.177645, 0.03351, -0.090353, -0.073988, -0.005218, -0.094529, -0.035871, -0.075, 0.353605, -0.088637], [0.062248, -0.000545, -0.294897, -0.089576, -0.379056, 0.10471, 0.047582, 0.289601, -0.130252, 0.171408, -0.077738, -0.202673, -0.72205, 0.564953, 0.058429, 0.143743], [0.217699, -0.054062, 0.025348, -0.622171, 0.064239, 0.141935, 0.009163, 0.333353, -0.337735, -0.397098, -0.122028, 0.124841, -0.032356, -0.180456, -0.070708, -0.058796], [-0.325986, 0.07768, 0.058101, 0.156756, -0.040143, -0.156196, -0.248644, -0.258481, 0.240373, 0.118264, 0.267123, -0.012103, -0.245083, 0.083731, -0.532919, -0.432458], [0.338514, 0.004719, -0.057813, 0.148885, -0.423127, -0.021469, 0.075223, 0.043083, 0.203848, 0.039072, 0.188025, 0.185425, 0.168319, 0.277963, 0.061078, 0.116104], [0.224592, -0.123257, 0.056035, 0.065796, 0.150837, -0.027666, -0.025172, 0.092338, -0.314947, -0.867418, 0.249301, -0.062737, 0.03542, 0.199346, -0.020592, 0.17977], [0.023318, -0.241384, 0.111874, -0.048595, -0.05139, 0.102408, -0.147941, 0.417486, 0.23464, 0.139173, 0.161418, 0.174234, 0.192339, -0.040519, 0.031836, -0.049386], [0.169491, 0.062514, 0.143615, 0.442471, 0.043914, 0.259432, 0.026649, -0.347124, -0.009801, 0.426209, -0.50091, 0.017164, 0.336589, -0.142009, -0.320381, 0.117792], [-0.1826, -0.154497, -0.08558, 0.002175, -0.137601, -0.058533, -0.098255, 0.086392, 0.0555, 0.177388, 0.267771, -0.072484, 0.236072, 0.160374, 0.176211, -0.366813]], [-0.008335, 0.230648, 0.261899, -0.080139, -0.19433, 0.408965, 0.244419, -0.051797, 0.203951, 0.175548, 0.088867, 0.192808, 0.033814, -0.37295, -0.146194, -0.240087]), ([[-1.780876], [-0.593797], [-0.662906], [1.287111], [-1.147842], [-0.575447], [-0.694969], [1.137534], [1.222602], [1.244919], [0.678561], [-0.933028], [-1.481327], [-1.425531], [1.466509], [-1.04814]], [0.599795])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.773948, -5.5e-05, 5.5e-05, 0.0, 0.053355, 0.053355, -0.0, 0.0, 0.0, 1.225105, 5.993309, 5.99331, 0.0]
SEQ_TSD = [1.082759, 0.122898, 0.122898, 0.093216, 1.493898, 1.493898, 0.652177, 0.203421, 0.162404, 0.580392, 1.641103, 1.641103, 0.64588]
SEQ_XM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.22782, 0.0, 0.0, 6.001745, 6.001824, 0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, -0.000234, 4.052088]
SEQ_XS = [0.942675, 0.815864, 0.771275, 0.675172, 0.58908, 2.091917, 12.760581, 0.63238, 11.907714, 0.687206, 0.645335, 0.538489, 1.638116, 1.638066, 0.999998, 4.392854, 0.972478, 0.325792, 129.998306, 315.957825, 944.868652, 1.34238]
SEQ_YSC = 1.5540871620178223
SEQ_WI = [[-0.954827, 0.51269, 0.026343, 0.545962, -0.647234, -0.647348, -0.052977, -0.187686, 0.951712, 0.027716, -1.165617, -0.783228, -0.661173, -0.398408, -0.054142, -0.135148, -0.025709, -0.024701, -0.451868, -0.579132, -0.69037, -0.016603, -0.694868, -0.010011, -0.150276, -0.053205, -0.093241, -0.205923, 0.07864, -0.138007, -0.263317, 0.184372, 0.113734, -0.010977, -0.383985, 0.174715, -0.219434, -0.370961, 0.338876, -0.274913, -0.250662, -0.057272, -0.195169, -0.094629, -0.240608, 0.037821, 0.156137, -0.366259], [0.207585, -0.398455, -0.458938, 0.535824, -0.283137, 0.136351, -0.015631, 0.361266, 0.044661, -0.182835, -1.160273, 0.00296, -0.245926, -0.230164, 0.038935, -0.098067, -0.324744, -0.016102, -1.000487, 0.268437, -0.063729, -0.149495, -0.007669, 0.114475, 0.221077, -0.614309, 0.082456, -0.374899, 0.133677, -0.008036, -0.200353, 0.067089, -0.001478, -0.159917, 0.831239, 0.050489, -0.053357, -0.356132, 0.134204, -0.130413, -0.16638, -0.894697, -0.281679, -0.456651, 0.032699, -0.055946, -0.141295, 0.208406], [-0.11419, 0.564071, -0.833042, -0.240773, 0.497483, 0.031417, 0.396312, -0.220059, 0.248232, -0.293891, 0.369814, -0.329677, -0.056524, -0.498315, 0.788586, 0.207117, 0.514894, 0.493307, -0.135402, 0.170361, 0.46825, -0.31624, -0.130483, -0.005916, -0.238821, 0.273911, -0.36958, 0.923555, 0.42769, 0.524412, 0.029335, -0.446969, 0.215814, 0.133193, -0.199979, 0.164376, 0.179276, 0.011852, -0.173311, -0.025035, 0.262994, 0.180506, -0.007127, 0.02544, 0.600943, 0.098397, 0.336578, -0.611739], [-0.333153, -0.15823, -0.777103, 0.394895, -0.206995, 0.111442, 0.435735, -0.66095, 0.280338, -0.285987, -0.056784, 0.15739, -0.217511, -0.337754, 0.461053, 0.412966, -0.235108, 0.139854, -0.387854, 0.125641, 0.46165, 0.38335, -0.517952, -0.227449, -0.114748, -0.580046, -0.194834, 0.199159, -0.010666, 0.37615, 0.037426, 0.215767, 0.13844, 0.348029, 0.119301, -0.177382, 0.368363, -0.07068, 0.134012, 0.099882, 0.036743, -0.124823, 0.314356, -0.272717, 0.439874, 0.191081, -0.372579, -0.232987], [-0.119907, -0.513737, -0.955434, 0.697889, -0.242663, 1.116997, -0.129553, -0.168988, 0.065164, -0.90877, 0.319687, -0.646574, -0.243089, 0.159603, 0.263883, -0.07867, -0.061681, 0.652143, 0.152754, 0.221112, 1.429119, -0.139299, -0.572534, -0.079163, 0.538402, -1.313951, 0.394353, -0.244452, -0.072775, 0.372255, -0.798688, 0.331991, 0.381267, 0.043053, -0.128029, -0.248147, 0.036759, -0.16928, -0.007319, 0.129939, -0.236755, -1.01525, -0.705404, 0.419368, -0.196259, 0.150709, -0.983061, 0.018076], [-0.177768, -0.346827, 0.345101, -0.578564, -1.382822, -0.19465, 0.256576, -0.239628, 0.458665, -0.657179, 0.118413, 0.210438, -0.409198, 0.004081, 0.826159, -0.331323, -0.656659, -1.768404, 0.988396, 0.43541, -1.583045, 0.238613, -0.037067, -0.198926, 0.377776, 0.008331, -0.207302, 0.654781, -0.299084, 0.01359, -0.064768, -0.314957, -1.014186, -1.047665, 0.112134, -0.057926, -0.226072, -0.105581, 0.883393, 0.598324, -0.249057, -0.053539, -0.162116, 0.065009, -0.791879, -0.128071, 0.494825, 0.408859], [-0.081083, -0.545478, -0.394994, 0.312466, -0.391457, -0.328868, 0.420258, 0.218864, -0.710941, 0.389047, -0.149516, 0.314995, -0.311708, -0.402477, -0.065197, 0.703806, 0.298598, -0.444274, 0.123979, -0.212671, 0.355686, -0.093338, -0.166873, -0.302179, 0.144465, 0.39478, 0.224522, 0.382734, -0.572296, 0.444705, -0.382698, 0.089499, 0.153372, -0.282503, 0.15551, -0.132915, 0.096032, 0.418665, -0.509415, 0.127894, 0.199111, 0.38782, 0.197025, -0.288801, -0.231934, -0.563515, -0.085754, 0.20867], [0.105099, 0.06708, -0.249753, 0.055932, -0.533892, -0.024065, 0.080572, 0.100139, -0.139996, -0.379348, -0.199422, 0.218521, 0.070348, -0.504179, 0.164452, 0.342955, 0.360201, -0.009404, 0.332326, -0.191535, -0.13677, 0.078021, -0.174405, -0.148435, -0.139314, 0.394352, 0.065713, -0.097865, 0.120611, -0.09598, -0.379701, -0.093184, 0.14642, 0.011813, -0.308722, 0.013625, -0.041315, 0.053858, 0.178461, 0.13584, -0.134951, -0.036227, -0.080702, -0.515377, 0.068742, -0.150171, -0.132519, 0.093129], [0.157309, 0.315731, 0.050022, 0.012593, -0.385517, 0.23558, -0.02806, 0.200313, -0.698872, -0.150034, -0.517864, 0.307445, 0.178724, -0.68905, 0.150443, 0.204194, -0.26272, -0.284655, 0.163636, -0.283668, 0.195278, -0.064849, -0.066586, -0.020916, 0.240826, 0.569204, 0.366836, -0.002708, -0.324475, 0.108307, -0.342241, -0.039194, 0.225781, -0.229895, -0.458386, -0.273434, -0.207292, -0.076357, 0.066148, 0.064673, -0.021312, 0.122218, 0.021894, -0.580742, -0.143812, 0.415759, 0.150733, -0.155122], [0.162403, -0.638493, -1.225481, 0.503459, -0.008757, 0.117361, 0.01247, 0.046387, -0.176643, 0.182948, -0.019044, -0.267591, 0.132514, -0.042169, -0.667816, -0.038645, 0.095347, -0.7639, 0.103912, 0.097109, -0.146109, -0.453355, 0.443629, 0.213453, -0.020012, -0.104681, -0.034624, 0.00966, 0.335022, 0.530389, 0.238272, 0.46724, -0.709367, 0.251082, -0.159495, 0.151544, 0.007638, 0.076282, 0.207605, -0.195732, -0.369937, -0.063649, -0.038645, 0.218002, -0.085025, 0.064181, -0.527164, -0.219747], [-0.039512, 0.091603, -0.190876, 0.132655, -0.102212, 0.144217, 0.06517, -0.011172, 0.365084, 0.236898, -0.294816, -0.245945, -0.287485, -0.155378, -0.066729, 0.128837, 0.109742, -0.096312, 0.061998, -0.236032, 0.145155, 0.254815, -0.016575, 0.005627, -0.722162, -0.2843, -0.364044, -0.481451, -0.190436, -0.248182, 0.035579, -0.172043, 0.122397, -0.06683, 0.217994, -0.094911, -0.179212, 0.099227, 0.342225, -0.029546, 0.074711, -0.057667, 0.036405, 0.307922, -0.111425, -0.036289, -0.120351, 0.162337], [-0.325076, -0.192055, -0.960585, 0.431407, -0.115553, -0.473403, -0.196506, -0.414676, 0.466306, -0.028257, -0.232048, -0.048263, 0.069514, 0.126374, 0.044755, -0.247671, -0.18801, -0.154822, -0.425679, -0.498951, 0.165791, 0.076383, -0.183282, -0.351012, -0.10013, 0.014119, -0.148994, -0.185372, 0.047492, -0.153962, -0.072488, 0.096009, -0.083123, -0.066479, 0.176036, -0.160792, -0.082089, 0.275938, -0.068186, 0.054615, 0.159594, 0.027323, -0.156463, 0.084474, 0.008817, -0.151378, -0.081848, 0.024585], [-0.05965, 0.146364, 0.560293, 0.20117, 0.157485, -0.137247, 0.194854, -0.22913, -0.287442, -0.256305, -0.058494, 0.077041, -0.373402, -0.298709, 0.268869, 0.004688, -0.006442, 0.620558, 0.217293, -0.311972, -0.152754, 0.494916, 0.108152, -0.02525, -0.10186, -0.017013, 0.065693, 0.20884, 0.252919, 0.274118, -0.122626, -0.437601, -0.048835, -0.182014, -0.106001, -0.285013, -0.205589, -0.175316, 0.07058, 0.030377, 0.010833, 0.105684, 0.075272, -0.06761, 0.111464, -0.106165, -0.127436, -0.209264]]
SEQ_BI = [0.179317, 0.399786, 0.201187, -0.167412, 0.295789, 0.171774, 0.133977, 0.221932, -0.289068, -0.247602, -0.118981, 0.015824, 0.254041, 0.137173, 0.304074, -0.061449, 0.761129, 0.602312, 0.502724, 0.344152, 0.801456, 1.172461, 1.016322, 0.764539, 1.220626, 0.37124, 0.674682, 0.306764, 0.819897, 0.203428, 0.316776, 0.235306, -0.362429, -0.060496, 0.131408, 0.215122, -0.288395, -0.496012, -0.241503, 0.086928, 0.348569, 0.013927, -0.175036, -0.275465, 0.059976, -0.239111, -0.376557, 0.495058]
SEQ_WH = [[0.136656, -0.723369, -0.15576, -0.296572, 0.158671, -0.285327, 0.580611, 0.059747, 0.58238, -0.185549, -0.056502, 0.124771, 0.203363, 0.022864, -0.043287, 0.517351, -0.519656, -0.06589, -0.348979, 0.532676, -0.097152, -0.650719, -0.182457, 0.090765, -0.390188, -0.362436, -0.764027, 0.133838, 0.027925, 0.164165, -0.747144, 0.145944, 0.87307, -0.219094, -0.183349, -0.205699, 0.558599, 0.701261, -0.329481, 0.311873, 0.45016, -0.326264, 0.169862, 0.011019, 0.240382, -0.528991, 0.322735, -0.338098], [-0.072987, -0.118531, 0.50069, 0.050135, 0.453056, 0.415759, -0.579517, -0.539881, -1.412107, 0.23599, -0.148795, 0.750649, 0.473678, -0.224357, -0.103384, 0.225563, -0.16677, 0.199645, 0.582209, 0.532631, 0.336702, 0.342244, 0.449706, 0.322988, 0.591756, 0.049993, -0.071523, 0.563146, -0.649143, -0.566936, 0.280645, 0.423762, -0.061796, 0.649564, -0.011812, -0.736874, 0.233495, -0.080202, 0.020021, -0.43682, -0.510614, 0.273272, -0.295202, -0.282894, -0.139711, -0.775062, -0.003354, 0.016843], [0.459305, 0.372375, 0.350072, -0.670611, 0.451275, 0.355271, 0.245287, 0.474509, -0.029453, 0.265083, -0.230337, 0.703789, 0.339369, -0.155736, 0.686521, 0.333503, 0.172249, 0.283063, 0.191302, 0.354684, 0.097803, 0.469531, -0.353956, 1.167603, 0.41505, 0.387632, 1.04915, 0.625833, -0.226118, -0.28522, 0.109323, 0.82781, -0.27297, 0.647964, 0.673489, 0.414994, 0.241576, -0.122776, 0.69378, -0.004462, -0.218756, 0.19996, 0.543341, 0.167724, -0.307427, 0.28745, -0.163326, 0.157234], [-0.672838, -0.443575, -0.462324, 0.139192, -0.75185, -0.746462, -0.625534, -0.762957, 0.347855, -0.349867, -1.312053, -1.167732, -0.521791, -0.775205, 0.236358, -0.572565, -0.024587, -0.407341, 0.237086, 0.032274, -0.653982, -0.254812, 0.146539, 0.21078, -0.583277, 0.160975, 0.542571, 0.317483, -0.741074, 0.867619, -0.135023, 0.401073, -0.331449, -0.300397, 0.297015, 0.960044, -0.193372, -0.295758, 0.032663, 0.1572, 0.402794, 0.20961, -0.517378, 0.125867, -0.636824, -0.205968, -0.061514, 0.248878], [0.403386, -0.187853, -0.292387, 0.096361, 0.120242, 0.574231, 0.388177, -0.126757, -0.06973, 0.393993, -0.615372, 0.041223, 0.509043, 0.291266, 0.100834, 0.69921, -0.804731, 0.308219, 0.636924, 0.393514, -0.401902, -0.500365, 0.261481, -0.506444, -0.636411, -0.161288, -0.193061, -0.208611, -0.784649, -0.320062, 0.191914, 0.298166, 0.209383, -0.025747, -0.268599, 0.274189, 0.132841, 0.507286, 0.07226, -0.133985, -0.037094, 0.139186, 0.097042, -0.308954, 0.303455, -0.408641, -0.00577, -0.033334], [0.586049, 0.231968, -0.028032, 0.012708, -0.044101, 0.261564, 0.333415, 0.275369, 0.776842, 0.204511, -0.0127, 0.087661, -0.200969, -0.35859, 0.119821, 0.317791, -0.844006, 0.089114, -0.376012, 0.01153, -0.5189, 0.320929, -1.374012, -1.313963, -0.734338, -0.195154, -0.665814, 0.048929, -0.508339, -0.555204, -0.556119, -0.625327, 0.351941, -0.0554, -0.016635, -0.337802, 0.070731, 0.867268, -0.04772, -0.463084, 0.582484, -0.138208, -0.049934, 0.011674, 0.182629, 0.485622, 0.223301, -0.56107], [-0.134606, -0.339282, 0.458426, 0.231069, -0.17316, -0.527474, 0.21743, 0.044742, 0.259281, 0.074197, -0.555203, 0.762212, -0.332159, -0.288956, 0.110313, 0.479069, -0.404986, 0.019175, -0.044107, 0.078136, -0.504003, -1.115188, -0.299455, 0.430657, 0.10922, 0.346929, 0.965457, 0.285719, -0.270374, 0.10905, -0.556938, 0.039392, -0.498454, 0.118229, -0.802077, -0.108493, -0.192511, -0.183548, 0.650066, 0.244613, -0.187876, 0.058232, 0.245029, 0.324347, -0.505265, -0.254685, -0.037594, 0.020487], [0.484969, 0.793863, 0.094066, 0.231622, 0.224632, -0.667471, -0.085315, 0.356954, 0.503568, -0.142773, 0.574634, -0.156077, 0.044228, 1.100021, 0.541512, -0.032258, -0.560976, 0.532609, -0.566837, -0.199116, 0.089509, -1.031201, -0.213207, 0.082313, 0.652351, 0.765332, -0.116728, 0.632034, 0.214607, -0.772476, 0.775691, 0.017215, 0.025075, 0.278244, -0.163998, 0.167438, 0.43893, -0.238733, -0.555925, 0.468324, 0.388483, 0.000974, 0.459005, 0.316957, 0.192607, 0.532542, 0.221036, -0.371608], [0.462379, 0.243645, -0.615503, 0.124502, -0.758419, -0.297774, 0.397006, 1.030103, 0.757517, 0.006849, -0.122421, -0.97709, -0.238591, 0.301473, -0.159384, -0.682756, 0.338332, 0.600498, 0.178311, 0.000453, 0.115101, 0.344798, 1.005903, 0.55115, 0.171564, 0.107588, 0.160592, 0.237087, 0.039961, 0.143774, 0.605101, 0.789944, 0.530585, -0.676058, -0.035442, 0.451388, -0.23603, -0.285221, -0.701728, 0.212424, 0.405267, -0.393728, 0.348814, 0.009476, 0.234334, 0.759992, -0.31552, 0.334105], [0.390315, -0.34001, 0.292839, -0.259659, 0.074392, 0.313302, -0.33508, -0.60522, -0.416351, 0.392492, 0.451085, 0.562388, 0.451632, -0.651042, -0.027119, 0.141908, 0.10558, -0.964831, 0.063611, -0.202704, -0.045912, 0.46142, 0.336627, -0.767643, 0.975909, -0.301833, 0.298268, 0.570886, -0.223128, 0.706529, 0.104373, -0.348176, -0.176996, 0.759427, 0.911335, -0.015989, -0.451544, 0.248836, 0.092169, -0.612864, 0.046398, 0.555007, 0.394766, 0.540321, -0.324717, 0.203754, 0.468423, 0.12547], [0.830517, -0.4, -0.711278, -0.420256, 0.511228, 0.66401, 0.589666, 0.313723, -0.773408, 0.346727, 0.499433, 0.562926, 0.303837, 0.608208, 0.124495, 0.075517, 0.002257, -0.359485, 0.611659, -0.090229, 0.470905, -0.753977, -0.030285, 0.00799, -0.105108, -0.637095, 0.538979, 0.716989, 0.411392, -1.065742, 0.25205, 1.183652, 0.327987, 0.103577, -0.317028, -0.007698, -0.346596, -0.168088, 0.395408, 0.139001, -0.092027, 0.107503, 0.09376, 0.152868, -0.221014, -0.146207, 0.05141, -0.132481], [-0.134666, -0.245989, -0.430204, 0.340863, -0.019648, -0.60069, 0.049398, -0.130134, 0.745426, 0.438873, -0.03918, 0.233171, -0.064006, 0.146085, -0.156218, -0.022341, 0.080184, -0.916582, -0.240366, -0.705902, -0.7985, -0.509613, -0.130354, -0.185407, 0.292125, -0.544595, -0.544624, -0.218238, -0.740059, 0.276167, -0.735243, -0.33639, -0.114318, 0.051548, -0.247676, -0.238764, -0.23122, 0.289058, -0.108746, 0.156075, 0.543165, 0.210842, -0.126936, 0.318885, -0.148284, 0.19481, 0.49135, -0.179187], [0.16497, 0.21479, -0.595653, -0.513615, -0.498013, 0.16301, 0.363167, 0.135048, -0.644174, -0.093494, 0.190272, -0.180108, 0.447232, 0.0047, 0.022333, 0.095457, -0.108756, 0.061752, 0.732623, 0.281669, 0.652371, 1.298602, 0.139664, 0.173657, 0.329145, -0.645048, 0.401743, 0.253225, 0.374107, 0.054458, 0.057575, -0.126875, 0.3729, -0.050042, 0.192513, 0.089134, 0.36685, 0.186696, -0.441637, -0.54336, -0.210645, -0.036122, -0.277894, -0.159885, 0.067383, -0.017064, -0.190176, -0.223535], [-0.087867, 0.475218, -0.272952, 0.001508, -0.461178, 0.296087, 0.733006, 0.206404, 0.710548, 1.137033, -0.289098, -0.301852, 0.065014, 0.834141, 0.697817, 0.138991, -0.303422, 0.173921, -0.474249, -0.948332, 0.111823, 0.671656, 0.567389, 0.502574, -1.062042, 0.38194, 0.362703, 0.462732, -0.254589, 0.181859, 0.350606, 0.11007, -0.037035, -0.550868, -0.48455, 0.36791, -0.218961, 0.120489, 0.24257, -0.39434, 0.739377, 0.413783, 0.506009, 0.922685, -0.087643, 0.30401, -0.525909, 0.243116], [-0.265149, -0.579282, -0.328821, -0.088162, -0.148249, 0.051803, -0.745875, -0.269853, 0.043466, 0.196146, 0.184243, 0.541683, -0.067493, -0.10785, 0.379922, -0.219309, -0.280386, -0.380951, -0.377629, -0.254921, -0.142689, -0.350686, -0.324257, -0.557857, -0.483675, -0.406179, -1.182648, 0.507867, -0.14512, 0.010349, -0.69005, 0.022252, 0.135231, 0.058865, 0.271124, 0.153918, -0.373625, 0.362933, -0.10193, 0.343832, 0.326898, -0.061848, 0.540587, 0.473321, -0.25447, 0.19754, 0.83109, -0.040995], [0.043633, 0.030414, -0.158945, -0.68144, -0.11588, -0.254559, -0.027407, 0.150871, -0.400263, -0.2946, 0.068686, -0.074291, 0.361462, -0.136139, 0.177616, -0.234794, 0.566918, 0.110122, 0.01601, -0.1426, 0.656943, -0.991835, -0.730111, 0.58545, 0.944658, 0.601745, -0.040543, 0.168303, 0.512196, 1.124193, 0.65367, 0.2511, -0.514231, 0.404561, 0.015508, -0.118477, -0.546794, -0.500364, -0.212992, -0.353101, 0.110747, -0.172106, -0.153082, 0.049983, 0.042287, 0.01007, 0.238908, 0.882439]]
SEQ_BH = [-0.077552, 0.140834, -0.058455, -0.223499, 0.180749, 0.348582, -0.0887, 0.233531, -0.311475, -0.322142, 0.278501, 0.121599, 0.364303, 0.035311, 0.265644, -0.188842, 0.613378, 0.687534, 0.365298, 0.503152, 0.51787, 0.929515, 0.732533, 0.523139, 1.035404, 0.408437, 0.961583, 0.394023, 1.048277, 0.313624, 0.437818, 0.397926, 0.071559, 0.195363, 0.000531, -0.123022, -0.33997, -0.376299, 0.183181, 0.177411, 0.241299, 0.271997, -0.126888, -0.188997, 0.30408, 0.198185, -0.025738, 0.056621]
SEQ_HEAD = [
    ([[0.117212, -0.666779, 0.170595, 0.127177, 0.139579, -0.180788, 0.228389, -0.042609, 0.091853, -0.128077, 0.223486, -0.089949, -0.094815, 0.123181, -0.191903, -0.056112, 0.006297, -0.089847, 0.100046, 0.190257, -0.066157, -0.405298, -0.077723, 0.268059, 0.062084, -0.301255, 0.155254, -0.182768, -0.037734, -0.085769, -0.015812, 0.088965, -0.230547, -0.135322, -0.154061, 0.038042, -0.130921, -0.097749, 0.06826, 0.084478, 0.202396, 0.135534, -0.096162, 0.013013, 0.151643, 0.074268, 0.004769, -0.146447, -0.101375, -0.05326, 0.265561, -0.311205, -0.153743, 0.02057, -0.104633, 0.160926, 0.007125, 0.001292, -0.128078, 0.186581, 0.40042, -0.029879, -0.051336, -0.121277], [0.090273, -0.304853, 0.08896, -0.150088, -0.172008, -0.194252, 0.141454, 0.008994, 0.136717, -0.166405, 0.184558, -0.016162, 0.078633, -0.153302, -0.026382, -0.027806, 0.011444, 0.10755, 0.084637, -0.084782, -0.039301, -0.133834, 0.080416, -0.10016, -0.14332, -0.063157, -0.17315, 0.069967, -0.090163, -0.03026, -0.008138, -0.072326, 0.083987, -0.144886, 0.018341, -0.275413, -0.270699, 0.087798, 0.068483, 0.067361, 0.138757, 0.062643, 0.159752, -0.143719, -0.037925, -0.074171, 0.135981, -0.181221, 0.139709, 0.056885, 0.292695, 0.095201, -0.154243, -0.15631, 0.211803, -0.056186, 0.072948, 0.01149, -0.055273, 0.232836, -0.015347, -0.032262, -0.071552, -0.034649], [0.193256, -0.14615, 0.032051, -0.173129, -0.139804, -0.24738, 0.216856, 0.011378, 0.162778, -0.132314, 0.196492, -0.116674, 0.099418, -0.092184, 0.035856, -0.044166, -0.146566, 0.182725, 0.102092, 0.049704, 0.113355, -0.150442, 0.100914, -0.118006, -0.238409, 0.017068, -0.294599, 0.216368, 0.02899, -0.125526, -0.13363, -0.090284, 0.128559, 0.053322, -0.117385, -0.37044, -0.046326, 0.012048, 0.07807, -0.179844, 0.264477, -0.091801, 0.088847, 0.049696, -0.126965, -0.081579, 0.08266, -0.092273, 0.039935, -0.133859, 0.001826, 0.037848, -0.169046, -0.171687, 0.005447, -0.286044, 0.046837, 0.193622, -0.071372, 0.032808, -0.140183, -0.064048, -0.096184, -0.017764], [-0.006076, 0.07535, -0.41959, -0.11634, -0.187868, 0.27673, -0.154197, 0.148504, 0.044294, 0.016169, -0.070082, 0.141818, 0.142401, 0.076337, -0.147227, 0.103498, 0.246376, -0.221531, 0.292111, -0.248774, 0.204067, 0.132349, 0.070539, 0.001068, 0.240596, 0.065293, -0.074462, 0.028095, 0.167183, 0.091921, -0.191787, -0.033386, 0.272904, -0.025567, 0.152464, 0.065801, 0.215129, -0.0846, -0.014973, -0.107744, -0.094095, -0.274639, -0.321784, -0.09334, 0.049206, -0.093773, -0.183669, 0.079798, -0.041312, 0.042511, -0.143135, -0.095303, -0.210512, 3e-06, 0.039531, 0.113354, 0.224498, -0.064318, 0.340527, -0.097029, -0.257085, 0.087965, 0.157782, 0.308643], [-0.011706, 0.310592, -0.364875, 0.008869, 0.31861, 0.212887, 0.057804, 0.105231, 0.324762, 0.04763, -0.033476, 0.107971, 0.253475, 0.460215, -0.075537, 0.008486, -0.113919, 0.111692, 0.492693, 0.113132, 0.28584, 0.146108, -0.000787, 0.060701, -0.048863, -0.004545, -0.338948, 0.041419, -0.095212, -0.350564, -0.487169, -0.104152, 0.118204, -0.020037, 0.332797, 0.215618, 0.097688, 0.414071, 0.010076, 0.093232, -0.359904, -0.314891, -0.240718, 0.123023, -0.1831, -0.191183, 0.037679, 0.047501, -0.060722, -0.021302, -0.282237, -0.153667, -0.25291, 0.069824, -0.43011, 0.057933, -0.230733, 0.042052, -0.089092, -0.06778, -0.455449, 0.188527, 0.036192, 0.306308], [-0.271143, 0.193906, 0.132261, -0.037968, -0.051129, 0.029797, -0.307497, 0.275825, 0.034444, 0.092455, 0.02279, -0.146009, -0.229761, -0.268203, 0.194669, 0.18501, -0.133579, 0.069767, -0.149478, 0.079647, 0.116877, -0.182404, -0.167128, -0.113352, 0.02118, 0.252986, 0.002263, 0.138185, -0.017809, 0.163557, 0.092157, -0.034502, 0.238834, 0.114287, -0.089133, -0.012589, 0.195077, -0.061284, 0.020925, -0.216867, -0.04052, 0.016258, 0.284938, 0.031824, 0.016762, -0.09227, -0.212935, 0.191572, 0.186313, 0.112086, -0.157025, 0.061704, 0.141402, 0.074742, 0.273366, -0.286684, 0.06144, 0.051178, -0.078407, -0.098836, -0.025134, -0.134659, -0.068124, 0.006947], [-0.119163, -0.140007, -0.367447, -0.277623, -0.09138, -0.208181, 0.180166, 0.377663, 0.043544, 0.208008, -0.131825, 0.459119, 0.113077, 0.492679, -0.036571, -0.178328, -0.473113, -0.309231, 0.036496, 0.398281, -0.048011, 0.270128, 0.059436, -0.139971, -0.24802, -0.295, -0.140577, 0.320971, -0.208573, -0.097794, 0.047996, 0.108833, 0.067599, 0.262511, -0.127679, 0.017452, -0.37782, 0.119968, 0.032052, 0.154675, 0.024435, -0.380217, 0.165409, 0.093021, -0.013572, -0.351157, 0.239711, 0.155232, 0.244413, 0.252922, 0.277373, 0.096493, -0.172671, -0.053555, -0.464286, -0.423999, -0.439901, -0.165954, -0.430949, 0.123917, 0.01592, -0.179517, -0.273589, -0.252002], [-0.049591, 0.125281, -0.052423, 0.005699, 0.067549, -0.040527, -0.066849, -0.03934, -0.077492, 0.07578, -0.027186, -0.025567, -0.148032, -0.086503, -0.052426, 0.056836, -0.062559, 0.139095, 0.0295, -0.061492, -0.041719, 0.561162, -0.189615, -0.120462, 0.007443, 0.092043, -0.020513, 0.052956, -0.058859, 0.032702, 0.081077, 0.090373, -0.00747, -0.035899, 0.102967, -0.168549, -0.003871, 0.032878, 0.148143, -0.075006, -0.067936, 0.063706, 0.020136, -0.020313, -0.153745, 0.172256, 0.141288, -0.181602, -0.315316, 0.02631, 0.061858, 0.26477, -0.113976, -0.168947, 0.121412, -0.070383, 0.006267, 0.018999, -0.010806, 0.126579, 0.012156, -0.336137, -0.021243, -0.06019], [0.079187, -0.161057, 0.025566, -0.26762, 0.003584, 0.0189, 0.047079, -0.299014, 0.14202, -0.32295, -0.13596, 0.071165, 0.217763, -0.173136, -0.367851, -0.046455, -0.000422, 0.033821, -0.040032, -0.163725, 0.01068, 0.334481, -0.315813, -0.232207, 0.007262, 0.031689, -0.457494, 0.069296, -0.415847, -0.075683, -0.175571, -0.382398, -0.178572, -0.100609, 0.087482, -0.100385, 0.417678, 0.102666, 0.218446, -0.090316, 0.062577, -0.004548, 0.176727, -0.368618, -0.356561, -0.257537, -0.03017, 0.383909, 0.011812, -0.133961, -0.136766, -0.47303, -0.105767, -0.158851, 0.12623, 0.116829, 0.080239, -0.036313, -0.004845, 0.250501, -0.388568, -0.08284, 0.195217, 0.296534], [-0.243498, -0.477599, -0.091487, 0.177627, -0.033384, -0.145239, -0.001458, -0.42935, -0.055334, 0.087441, -0.28545, 0.060689, -0.117814, -0.312267, 0.154589, -0.58097, 0.228188, 0.034331, -0.177155, 0.055114, 0.240657, -0.057204, -0.455861, -0.10827, -0.042781, -0.136446, -0.120564, -0.493441, -0.17434, 0.140416, 0.173149, 0.23639, 0.092079, -0.057234, -0.232249, -0.350023, -0.046671, 0.09416, 0.089109, -0.002749, -0.411581, -0.367554, 0.071288, 0.034833, 0.196833, -0.097724, -0.178198, 0.376677, 0.040854, -0.143116, -0.028423, 0.113855, -0.179098, 0.148077, -0.02575, -0.023535, -0.10623, 0.151534, -0.131228, -0.361623, -0.155747, -0.059581, 0.248914, 0.053039], [0.188186, 0.004415, 0.095317, 0.104827, 0.017589, -0.207127, 0.083896, 0.065286, -0.19002, -0.093823, -0.239081, 0.166966, -0.141461, 0.090018, -0.153582, -0.076944, 0.016258, -0.029261, 0.068834, -0.147925, -0.027191, -0.083713, 0.020495, 0.007446, 0.01042, 0.048826, 0.063743, 0.157832, 0.032316, 0.326123, 0.078152, 0.243101, -0.277793, 0.047698, -0.042545, -0.002028, -0.122921, 0.010089, 0.129413, 0.026574, -0.012463, 0.043443, -0.002544, 0.062414, -0.127228, 0.008544, -0.18247, -0.100435, 0.173169, -0.107093, -0.117582, 0.049544, -0.036474, 0.029915, 0.119168, 0.10741, -0.086818, -0.123047, -0.016109, -0.032512, 0.115927, 0.090977, 0.066193, -0.076932], [0.486185, -0.11357, -0.154086, -0.594023, -0.216794, -0.141975, -0.047013, -0.279638, 0.155008, -0.510265, 0.341055, -0.077597, -0.039708, -0.186467, 0.303265, -0.504144, 0.064282, 0.320822, 0.13643, 0.047943, 0.406809, 0.137029, 0.353719, -0.126135, 0.128031, -0.04482, -0.059387, -0.203938, -0.075225, -0.010931, -0.322724, -0.053838, 0.049703, 0.059682, 0.030626, -0.243785, 0.065084, 0.113791, 0.200989, 0.105923, 0.651069, 0.370649, 0.03592, -0.277228, -0.367756, 0.498864, 0.365607, -0.478197, -0.075889, -0.28118, 0.040822, -0.042675, -0.533181, -0.265567, 0.07431, -0.036639, -0.046561, 0.38597, -0.001543, 0.359156, -0.319324, -0.251578, -0.05497, 0.510836], [0.30361, 0.152405, 0.009634, 0.069058, -0.139673, 0.108653, -0.006884, 0.11796, -0.005484, 0.019314, 0.087865, 0.149532, 0.196338, -0.007577, -0.069856, -0.131457, 0.108737, 0.026999, -0.162275, -0.173667, -0.095087, 0.151707, 0.017522, -0.078623, 0.166504, 0.005724, -0.061573, 0.309647, -0.080571, -0.139925, -0.137023, 0.038415, 0.027674, 0.001832, -0.179965, -0.021346, -0.115791, -0.048769, -0.007727, -0.029545, -0.01725, 0.331786, 0.042727, 0.259925, -0.00374, 0.255004, -0.00613, 0.184234, 0.225241, 0.026863, -0.106708, -0.056259, 0.046352, -0.068545, 0.034967, 0.154209, -0.102963, 0.303562, 0.123824, 0.254041, -0.141498, 0.303899, -0.03536, 0.143065], [-0.201994, 0.006417, -0.158644, 0.180207, -0.19585, -0.081624, 0.026927, 0.245951, 0.163468, 0.091161, -0.091027, 0.101346, 0.092452, 0.296368, 0.08102, 0.09444, -0.09365, -0.033379, 0.019312, -0.000614, -0.210886, -0.253307, -0.029765, 0.184637, -0.018007, -0.076924, -0.046827, 0.013059, 0.035855, -0.353518, -0.244923, -0.019391, 0.07314, 0.058836, 0.221954, 0.03422, -0.030003, -0.161034, -0.155223, -0.127938, 0.178428, 0.173608, -0.130247, 0.267262, -0.072714, 0.013684, 0.203813, 0.175757, 0.137051, 0.242003, -0.015169, -0.050727, -0.139035, -0.067938, 0.079673, 0.110306, -0.001569, 0.435128, -0.240389, 0.520757, -0.025201, 0.096255, -0.261971, -0.092241], [0.109134, 0.050965, 0.111274, 0.089868, 0.10024, 0.060015, -0.101022, -0.023001, -0.077945, -0.244342, -0.047025, -0.184332, -0.299231, -0.104516, 0.078396, -0.161315, 0.326562, -0.571128, -0.146048, -0.111028, 0.130437, -0.236911, -0.07292, -0.233967, 0.214655, 0.148617, -0.145193, -0.080674, -0.633309, 0.091141, 0.058913, 0.118178, -0.192082, 0.020531, -0.028648, 0.277278, 0.275526, -0.035771, 0.482818, -0.457926, 0.098365, 0.006213, 0.08266, -0.014163, 0.159743, -0.199349, 0.053118, -0.119412, 0.1707, 0.123057, -0.031435, -0.00093, -0.056813, -0.442698, 0.04604, 0.562159, 0.04867, -0.198189, 0.274442, 0.066223, -0.085693, 0.068416, 0.065566, -0.05176], [-0.118005, -0.107459, 0.097089, -0.012665, 0.188426, -0.290531, 0.470217, 0.163404, 0.037877, -0.212716, -0.358271, -0.213349, 0.212204, 0.108047, -0.196496, 0.264053, -0.285452, -0.015644, 0.008697, -0.048038, 0.139413, -0.295532, -0.355549, 0.168746, -0.1698, -0.149825, 0.210238, 0.080138, -0.339613, 0.011413, 0.049331, -0.044003, -0.157386, -0.080032, 0.282227, 0.050955, -0.385419, -0.346019, -0.139469, -0.104121, -0.012088, -0.20574, -0.130893, -0.230278, -0.178842, -0.243994, -0.02382, -0.066225, -0.243347, -0.015189, 0.139661, 0.293087, 0.035706, 0.231397, 0.135394, -0.036156, -0.004344, -0.07954, 0.102396, -0.383608, 0.181556, -0.024712, 0.065878, 0.027073], [-0.014168, 0.094191, -0.089744, -0.300431, -0.070713, -0.250724, 0.173134, -0.143614, -0.106339, -0.07022, 0.127649, -0.105544, 0.457109, 0.080209, -0.290298, 0.095471, -0.19393, 0.504301, 0.187023, 0.045025, 0.14064, 0.190349, -0.070656, 0.29289, -0.191175, -0.12417, 0.172216, -0.059997, 0.241365, -0.015571, -0.107493, -0.152063, 0.132622, -0.267602, -0.744397, -0.114979, -0.177926, 0.017166, -0.186815, 0.201417, 0.03316, 0.035586, 0.089235, 0.020012, 0.005379, -0.029575, 0.083109, 0.235937, -0.27638, -0.287524, -0.243863, -0.333278, -0.020891, 0.297511, -0.092151, -0.091214, 0.189943, 0.011348, -0.124852, 0.228313, -0.229015, -0.031009, 0.280018, 0.154759], [0.081727, 0.223577, -0.061692, -0.121437, -0.150998, -0.008917, 0.343313, -0.084312, 0.105681, -0.184449, -0.412095, -0.038038, -0.078406, 0.224006, 0.118444, 0.129518, 0.088123, 0.205927, -0.022115, -0.079282, -0.107954, -0.220315, 0.121209, 0.140858, 0.034423, -0.245794, 0.300463, -0.089448, 0.01177, -0.065163, -0.001373, 0.015978, 0.150494, -0.151058, 0.172404, 0.022939, 0.09373, -0.174387, 0.153552, -0.051545, 0.050908, -0.182754, -0.07339, -0.022553, 0.127169, -0.16226, -0.169, 0.065182, -0.017102, 0.04619, 0.020082, -0.037697, -0.122605, -0.121735, -0.037184, 0.056449, 0.0824, -0.176999, 0.201783, 0.164169, -0.019779, 0.035055, -0.09602, 0.343215], [-0.045609, -0.181777, 0.005371, 0.185154, -0.015047, 0.086835, 0.048395, 0.245573, -0.085527, 0.157834, -0.011116, -0.140898, -0.097842, -0.143705, 0.082389, -0.10535, -0.036943, 0.045494, -0.000319, -0.037492, 0.105227, -0.043343, 0.09748, 0.159593, -0.090503, -0.06181, -0.035567, -0.034959, 0.057369, -0.174962, 0.002066, -0.132378, -0.039733, -0.035321, 0.09825, 0.105373, -0.064206, 0.100318, -0.080628, 0.070677, 0.129695, -0.138358, -0.050206, 0.006899, -0.029383, 0.301522, 0.135354, 0.053942, -0.090184, -0.026777, -0.265927, 0.319182, 0.086425, 0.150585, 0.083486, -0.207962, 0.151276, -0.210941, 0.044051, -0.359409, 0.024497, -0.135755, -0.094289, -0.139608], [0.103146, 0.050418, -0.16731, -0.05192, 0.032152, -0.029963, 0.141615, 0.21395, 0.103463, 0.121676, -0.179689, 0.63485, 0.10484, 0.065609, -0.312089, -0.16629, -0.317749, -0.090684, 0.205637, -0.322117, 0.171422, -0.263712, 0.190778, 0.230337, 0.065892, -0.242045, -0.129298, 0.124327, 0.085074, -0.089559, 0.033871, -0.026498, 0.33246, -0.200365, -0.407498, 0.216712, -0.24522, -0.078336, -0.110961, 0.356076, -0.036688, -0.351509, 0.326448, 0.293815, -0.484621, -0.026557, 0.227839, -0.105748, 0.247487, -0.080216, 0.172977, 0.162081, -0.189746, 0.323037, -0.210176, -0.433833, -0.441457, 0.111927, 0.056602, -0.27826, -0.176938, -0.286868, 0.140211, -0.255169], [0.005261, -0.103488, 0.234638, -0.120415, 0.078541, -0.143117, 0.004405, 0.005243, 0.243618, -0.086057, -0.254301, -0.157736, -0.183339, -0.409065, -0.069407, 0.028003, 0.018757, 0.074943, -0.385203, -0.094453, -0.138975, 0.053142, -0.012075, -0.252073, 0.003546, 0.070724, 0.371521, -0.015835, 0.208305, 0.002036, 0.093069, -0.262357, -0.024492, 0.042647, 0.008066, -0.085907, 0.422471, -0.054452, 0.184211, 0.412774, 0.093531, 0.247829, -0.018573, 0.197402, -0.125119, 0.004679, 0.062469, 0.287885, 0.055777, -0.052967, -0.253337, -0.067994, 0.033272, -0.21267, 0.324044, -0.205634, -0.26499, -0.373267, -0.395044, -0.097916, 0.255922, -0.361091, 0.170253, 0.116687], [0.053832, 0.070217, 0.006452, 0.35048, 0.247085, -0.070502, 0.061667, -0.051709, 0.368174, -0.007721, 0.118644, 0.510984, -0.376948, 0.209639, -0.010671, -0.099529, -0.185544, -0.472202, -0.014559, -0.034117, 0.035126, -0.086868, 0.618461, -0.11144, -0.328559, 0.293187, 0.169244, 0.009617, 0.494324, 0.001307, 0.026782, 0.058116, -0.069813, -0.061809, -0.165614, -0.0295, -0.169711, 0.06815, -0.179094, -0.264168, 0.017277, -0.03155, -0.016778, 0.440339, -0.043516, 0.342212, -0.242872, 0.175517, -0.622408, -0.45439, 0.06025, 0.088987, 0.052384, -0.219742, 0.059443, -0.125494, 0.031448, 0.206555, -0.161031, 0.541678, 0.110216, 0.455707, 0.070672, 0.277833], [0.55948, 0.273149, 0.033654, 0.437085, 0.188857, 0.297461, 0.236343, -0.410369, 0.430391, 0.233573, 0.174014, 0.170379, -0.034579, -0.285704, 0.240324, -0.107195, 0.30481, 0.116197, -0.048866, 0.322823, 0.335359, 0.085, 0.544981, 0.115075, -0.302143, 0.210147, 0.015584, -0.399081, -0.269502, -0.187951, 0.080711, 0.216788, -0.477385, 0.191601, -0.000564, 0.154332, -0.003564, 0.130552, 0.223827, 0.093599, 0.261808, 0.405106, 0.231053, -0.147706, -0.17035, 0.11698, 0.01475, 0.043002, -0.265483, -0.148555, 0.093416, 0.055879, 0.169892, -0.004573, -0.252258, 0.004575, -0.052163, -0.092094, 0.050905, 0.415766, 0.480586, -0.109831, 0.484145, 0.180533], [0.028384, -0.068877, 0.343518, -0.005754, 0.057756, -0.084624, 0.29842, -0.221835, 0.070219, -0.161137, -0.157363, 0.24164, -0.08325, -0.03448, 0.15088, -0.201216, 0.139043, 0.046046, -0.024525, -0.216142, 0.174783, -0.047467, 0.255747, -0.062335, 0.002107, 0.010864, -0.144852, -0.12551, 0.238176, 0.076598, -0.016781, 0.040534, -0.223986, -0.097127, -0.340574, -0.486536, -0.017637, 0.203771, 0.273739, -0.169348, -0.005063, -0.272106, 0.258878, -0.397662, 0.154114, 0.130612, 0.306471, -0.21481, -0.12157, 0.096103, 0.090924, 0.125888, -0.122157, 0.189723, 0.23785, -0.090156, 0.088694, 0.127785, -0.04348, 0.261536, -0.130244, 0.053428, 0.111655, -0.24884], [-0.004289, 0.070041, -0.048602, -0.065481, -0.060749, -0.027524, -0.109897, -0.105234, 0.117757, 0.062974, 0.282328, -0.276731, -0.223829, 0.091105, -0.292651, -0.111553, -0.019248, -0.162068, -0.144255, -0.207488, -0.031573, 0.022077, 0.218054, -0.427156, 0.288245, -0.296455, 0.07386, 0.203908, 0.056624, 0.092564, 0.288357, -0.323678, 0.070103, -0.062646, -0.227325, -0.227992, -0.017767, -0.18306, 0.38724, -0.013233, -0.089835, 0.058194, 0.0302, -0.101439, -0.093861, -0.019291, 0.219006, -0.43408, 0.152469, 0.1398, -0.073963, -0.854726, -0.414293, 0.145696, 0.087388, -0.238063, -0.20633, 0.035692, -0.002909, -0.099784, 0.012363, -0.425211, 0.413888, -0.100743], [-0.056093, 0.122808, -0.038074, -0.054636, 0.165272, -0.118083, -0.295447, 0.150089, -0.023654, -0.014279, 0.167093, -0.500297, -0.186082, 0.107016, -0.427851, 0.148577, -0.003488, 0.107435, -0.244204, -0.042322, 0.034026, 0.171552, -0.073765, -0.546263, 0.013966, -0.165546, -0.066902, 0.071911, 0.126646, -0.310448, -0.169264, -0.05214, 0.334498, -0.153099, 0.033827, -0.069254, -0.216558, -0.09381, 0.060082, -0.225516, -0.013391, 0.119533, 0.047428, 0.206364, 0.220588, -0.166127, -0.297687, 0.346901, -0.172759, -0.091569, -0.246052, 0.040776, -0.093348, -0.047299, 0.077814, -0.001858, -0.164234, 0.161104, 0.031476, -0.232597, -0.110362, 0.059738, -0.263467, 0.124241], [-0.044144, 0.287385, 0.039806, 0.224282, -0.149935, 0.048136, 0.218556, 0.151084, 0.482245, 0.057042, -0.201179, -0.257683, -0.111838, -0.118607, 0.330898, -0.07123, 0.077169, 0.207223, 0.179007, 0.383849, 0.080126, 0.066515, 0.221264, 0.511843, -0.159287, 0.214884, -0.057465, -0.262438, 0.025083, -0.058167, 0.018555, -0.076981, 0.086283, 0.003539, 0.156868, -0.075322, 0.358004, 0.241964, 0.342146, -0.01149, 0.091397, 0.339769, 0.076159, -0.051873, 0.007836, -0.095805, -0.155964, 0.194699, 0.148706, -0.120572, -0.055285, 0.089761, 0.091325, 0.071933, 0.027707, 0.31174, 0.259234, -0.157349, -0.086507, 0.307357, 0.044704, 0.130347, 0.853672, 0.247048], [-0.018015, 0.022544, 0.05117, 0.268442, 0.156249, 0.016597, 0.64681, 0.019517, 0.156254, 0.119332, -0.055547, 0.076891, 0.37007, 0.194645, 0.330473, -0.047256, 0.088628, 0.167678, 0.674514, 0.525627, 0.269695, 0.54148, 0.115732, 0.645241, -0.025177, 0.167723, 0.283293, 0.030573, -0.093691, -0.283576, 0.002001, 0.186138, 0.310313, 0.142068, 0.083374, -0.06242, 0.122767, 0.159725, 0.106453, -0.043141, 0.542576, 0.245569, -0.13952, -0.249369, -0.04241, -0.000507, -0.047215, 0.288374, -0.150782, 0.114335, 0.483511, 0.485153, 0.334686, 0.172806, -0.209691, -0.013709, 0.134088, 0.154003, 0.247245, -0.083296, 0.330637, 0.480561, 0.628693, 0.416651], [0.112195, 0.200824, 0.214474, -0.041084, 0.393827, 0.295459, -0.297793, 0.256683, -0.345293, 0.220221, -0.111799, 0.307221, -0.265844, -0.118224, -0.080369, -0.074278, -0.046623, -0.096496, -0.239553, 0.453643, 0.005619, -0.140176, 0.02749, -0.277395, -0.048134, 0.422484, 0.059285, 0.100079, -0.265403, 0.448524, 0.256575, -0.111045, 1.017413, 0.043693, -0.06072, -0.296723, 0.368726, 0.137242, 0.431916, -0.097016, 0.014889, -0.063341, 0.098756, -0.08713, -0.248573, -0.134816, 0.191837, -0.427258, 0.336734, -0.085954, -0.376123, 0.208646, -0.21377, -0.282299, -0.009818, -0.263101, 0.014736, -0.073318, 0.607976, 0.578337, -0.298935, -0.342179, 0.292885, 0.350621], [-0.12126, -0.321638, -0.248607, -0.110997, -0.706312, -0.224821, -0.4123, 0.23385, -0.340058, -0.004915, 0.439459, -0.678296, -0.597082, 0.457955, -0.113986, -0.044093, 0.123221, -0.035294, -0.343949, 0.025945, -0.457219, -0.314658, -0.195518, 0.076391, -0.115848, 0.037287, -0.400526, 0.44444, -0.23091, 0.229964, -0.056033, 0.103199, 0.264232, 0.127886, 0.263803, -0.044401, -0.082529, -0.79317, 0.394184, 0.125402, -0.467883, 0.01441, -0.729468, 0.414465, 0.011345, -0.2231, -0.320429, 0.144432, 0.248101, 0.065164, -0.204839, 0.214414, -0.123477, -0.4882, -0.131267, -0.156606, 0.085107, -0.273977, 0.146133, -0.049686, 0.14848, -0.576529, -0.501336, 0.159369], [0.31374, -0.117959, 0.254233, 0.111259, -0.304614, -0.36049, -0.103094, 0.040146, 0.173394, -0.467271, -0.112626, -0.518842, 0.408043, 0.08721, -0.613515, -0.356347, -0.12672, 0.435513, 0.0828, -0.658384, 0.266442, 0.098738, -0.080988, -0.398805, 0.140315, -1.102368, 0.047005, -0.041532, 0.337366, -0.52247, -0.484793, 0.283105, -0.39578, -0.502153, 0.528599, 0.18255, -0.763095, 0.063765, -0.710373, -0.335343, -0.053582, 0.497039, 0.212883, 0.325216, -0.086379, 0.394752, -0.340416, 0.179342, 0.12385, -0.212908, -0.460742, -0.275922, 0.039942, -0.395899, -0.471154, -0.03695, -0.793548, 0.530046, -0.304556, -0.581246, 0.380411, 0.22589, -0.327089, 0.074987], [-0.059842, 0.014858, 0.068745, -0.070116, 0.038861, 0.242648, -0.013513, -0.041781, -0.099355, 0.650075, 0.119186, 0.337028, 0.101551, 0.069003, 0.194866, 0.238343, 0.508474, 0.123293, 0.057431, -0.18332, 0.029503, -0.150064, -0.31532, 0.188374, 0.726436, 0.443774, 0.202565, 0.265501, 0.179746, -0.001229, 0.855284, -0.167531, -0.185287, 0.624669, -0.388083, 0.031895, 0.781953, -0.172585, 0.210417, -0.163617, -0.104405, -0.513305, 0.102809, -0.344421, 0.338726, 0.240187, 0.019907, 0.402321, -0.074316, 0.959969, 0.149776, 0.001779, 0.244026, 0.351475, 0.498721, -0.425063, 0.142633, 0.22707, 0.146731, -0.331324, 0.521583, -0.279845, 0.101695, -0.264272], [0.286207, 0.183186, -0.14958, 0.317182, -0.355371, 1.169756, -0.270935, 0.11104, 0.161372, 0.902123, 0.063395, -0.077676, -0.088063, 0.405078, -0.182267, -0.385996, 0.340448, 0.038226, 0.107504, -0.398652, -0.027802, 0.360895, -0.071073, -0.17693, 0.382778, 0.431265, 0.309659, 0.152453, -0.05246, 0.572655, 0.142628, 0.271706, 0.054005, 0.114741, 0.199266, 0.135246, 0.52955, -0.154147, 0.636848, -0.468831, -0.008366, 0.180907, 0.522456, -0.006635, 0.133068, 0.302385, -0.284649, -0.003823, 0.167187, 0.433857, -0.19148, -0.246265, 0.412151, -0.413304, 0.051201, -0.475629, -0.347006, -0.176135, 0.284952, 0.00596, 0.13898, -0.68865, 0.2365, 0.310454], [-0.004096, -0.137934, -0.229506, 0.043958, 0.056614, -0.027918, -0.121712, 0.058182, 0.103112, 0.15604, 0.148075, -0.163494, -0.020886, -0.002869, -0.16526, -0.21135, -0.1164, 0.051034, 0.160332, 0.099044, -0.032101, 0.146758, -0.301478, 0.043297, 0.013165, 0.223392, 0.230335, 0.383371, 0.003232, 0.07192, 0.109275, 0.115837, 0.062602, 0.229657, 0.136798, -0.016259, -0.187865, -0.047348, 0.201552, 0.048324, 0.00471, -0.124412, -0.131372, 0.098169, 0.021248, 0.149937, -0.118981, 0.537642, 0.057624, 0.010567, 0.033199, 0.052579, 0.38508, 0.062418, 0.020489, -0.075722, 0.201882, 0.191146, -0.107453, 0.018176, 0.393647, 0.044666, -0.134799, 0.180847], [0.091988, 0.333558, 0.46161, 0.183751, -0.08136, -0.065333, 0.199721, -0.555101, 0.283492, -0.080665, -0.248651, -0.156505, 0.42376, -0.541122, 0.174415, 0.0217, -0.206094, 0.258745, 0.042978, -0.057857, 0.28996, -0.270852, 0.205108, 0.379561, -0.043781, -0.140758, -0.017608, -0.518647, 0.165535, -0.149156, -0.102866, 0.006994, -0.400956, 0.136554, 0.089718, -0.106929, 0.041952, 0.4242, 0.110659, 0.253642, 0.051811, 0.247037, 0.652077, 0.092158, -0.113353, -0.06626, 0.154159, 0.199731, -0.038166, -0.031809, 0.159744, -0.312272, 0.094984, 0.254345, -0.289799, 0.147276, -0.196022, 0.173095, -0.352334, -0.202803, 0.061482, -0.029953, 0.086424, -0.398819], [0.140661, -0.025952, -0.212627, 0.49031, -0.544413, -0.192388, 0.132834, 0.407984, 0.144067, -0.483309, -0.06599, 0.005514, 0.071414, 0.171618, 0.171304, -0.428953, 0.266627, -0.069462, 0.391804, -0.609392, -0.087287, 0.291945, 0.184191, 0.015969, 0.151063, 0.209591, -0.00683, 0.388172, -0.169873, 0.068817, -0.094343, 0.264089, 0.180184, 0.054308, 0.11667, 0.203185, 0.092599, -0.067962, 0.305296, -0.175234, -0.002753, 0.295083, -0.16321, 0.444176, 0.328328, -0.032493, -0.047552, 0.194408, 0.567554, 0.086282, 0.18602, 0.103072, 0.185406, -0.258488, -0.095346, -0.013009, -0.696591, -0.10091, -0.032748, -0.21224, 0.109225, -0.253224, -0.256715, -0.051303], [0.012838, 0.05155, -0.256142, 0.707826, 0.487553, 0.263046, 0.154848, 0.364982, -0.04119, 0.359003, 0.395717, 0.131738, -0.224506, 0.128647, 0.391236, 0.296862, 0.120081, -0.321068, 0.139827, -0.160942, -0.161863, 0.567577, -0.144657, 0.081768, 0.187123, 0.834327, 0.264169, 0.43454, 0.224591, 0.153343, 0.370035, 0.229772, 0.4133, 0.230353, -0.167391, 0.098898, -0.329248, -0.087125, 0.176758, 0.227946, 0.097584, -0.23439, -0.16453, 0.079783, 0.109185, 0.495158, -0.111894, 0.157633, 0.068067, 0.061517, 0.499449, 0.131293, 0.57291, 0.005085, 0.181491, -0.48341, 0.331938, 0.092433, 0.2095, 0.030821, 0.442339, -0.072006, 0.035613, 0.298674], [-0.069945, 0.13214, 0.142041, -0.104681, -0.109923, -0.170187, -0.108036, 0.293262, 0.322439, -0.083285, -0.02958, 0.009162, 0.221284, 0.127844, -0.092678, 0.051061, -0.266585, -0.095827, 0.01187, -0.647232, -0.170072, -0.063465, -0.164569, -0.20045, 0.06709, -0.351082, -0.02282, 0.078586, 0.193174, -0.126417, 0.010476, -0.423416, 0.084523, -0.042793, -0.040815, 0.095248, -0.083767, -0.192214, -0.253995, 0.103298, -0.117305, 0.054279, -0.224992, -0.194503, -0.122967, 0.065217, -0.05086, -0.050812, -0.069845, 0.195946, -0.014609, -0.422644, -0.21568, -0.199067, 0.007555, -0.344602, 0.002744, 0.099326, -0.361255, -0.203383, -0.059325, 0.171994, -0.579685, 0.130174]], [-0.124518, -0.123894, 0.044662, -0.404487, -0.017053, -0.317365, -0.12986, -0.248426, -0.183765, -0.037603, 0.071651, -0.087596, 0.025642, 0.018783, -0.229, -0.094093, -0.14565, 0.190787, -0.45614, -0.731114, -0.16564, -0.287598, -0.026362, -0.408595, 0.062659, -0.22687, -0.234333, -0.366056, 0.025929, 0.064454, 0.005667, -0.090651, -0.524582, -0.05048, -0.031182, -0.129738, -0.002919, -0.012139, -0.287846, -0.114737, -0.357705, -0.256667, 0.039857, -0.190678, -0.065871, -0.016188, 0.081143, -0.459061, -0.252909, 0.15874, -0.032639, -0.303528, -0.41119, -0.269006, 0.219815, -0.036479, 0.186302, -0.111478, -0.151374, -0.336791, -0.382106, -0.00558, -0.353316, -0.388055]),
    ([[-0.034463, -0.094625, -0.047703, 0.024906, -0.161043, 0.106038, 0.098657, 0.120023, -0.111823, -0.066316, -0.230534, 0.042996, 0.009894, -0.09028, -0.014076, 0.197139, -0.023168, 0.049681, 0.18236, -0.046004, -0.223655, -0.201535, 0.054535, -0.059835, 0.049039, 0.167705, -0.40266, -0.062986, -0.402666, -0.105141, -0.239749, 0.050641], [-0.08613, -0.091047, 0.098583, -0.070483, 0.162093, 0.231738, 0.104674, -0.036974, 0.367478, 0.220047, 0.483781, 0.328203, 0.05958, 0.178358, -0.253602, 0.09833, 0.174686, 0.412621, 0.25865, 0.17461, 0.061442, -0.022775, 0.376251, -0.171173, 0.230551, 0.179682, 0.075409, 0.356451, -0.084819, -0.124756, 0.249938, 0.15308], [-0.265201, 0.042024, 0.118085, 0.022194, 0.097892, 0.062254, -0.511962, 0.11361, 0.09207, 0.088532, -0.059759, 0.252545, -0.111223, 0.260879, -0.114851, 0.256403, 0.055348, -0.105504, 0.213113, 0.112772, 0.060477, 0.032589, 0.143207, -0.2612, -0.250376, -0.054667, 0.113742, 0.075066, -0.043658, -0.165288, -0.11071, 0.127786], [-0.34813, 0.140127, 0.176838, 0.055915, 0.23412, -0.385712, -0.096145, -0.080605, 0.134495, 0.01209, 0.161996, 0.4095, -0.008394, 0.050834, -0.236895, -0.231826, -0.160208, -0.181495, 0.227081, 0.202959, 0.07932, 0.180142, 0.138206, 0.228158, -0.304023, -0.221098, -0.103767, 0.170256, 0.19044, -0.123667, 0.184509, 0.090414], [0.050302, 0.037506, -0.127099, -0.34182, 0.012346, 0.294873, 0.067244, 0.079659, 0.016891, -0.348816, 0.259034, -0.138887, -0.153124, -0.000401, -0.078185, 0.08673, 0.050304, -0.156586, -0.127342, 0.326209, -0.046466, -0.29938, 0.021327, 0.209273, 0.08606, -0.045666, 0.046407, -0.047295, -0.359231, -0.012608, 0.049899, 0.239881], [0.028718, 0.214579, 0.223927, 0.093211, 0.152461, -0.273792, 0.022573, -0.112497, -0.101644, -0.085952, 0.279632, 0.277268, 0.269702, 0.118563, 0.065168, -0.128683, -0.105922, -0.106439, 0.342156, 0.149135, 0.235697, 0.049972, -0.050794, -0.192945, -0.010836, -0.233376, 0.10243, 0.125743, -0.115906, 0.062098, 0.174927, 0.089572], [0.170085, 0.00139, -0.370896, -0.123466, 0.157524, 0.009388, -0.067308, 0.024221, 0.241497, 0.214872, 0.319075, -0.103226, 0.145136, 0.255714, -0.018716, 0.120298, 0.135538, 0.200779, 0.138194, 0.054666, 0.190077, 0.358786, -0.101457, -0.551182, 0.113355, 0.159758, 0.019392, -0.010145, 0.309983, 0.243312, -0.003163, 0.348943], [-0.233493, -0.133556, -0.119808, -0.108381, -0.137704, 0.010602, -0.307999, -0.117739, 0.236772, 0.000952, 0.006794, -0.934989, -0.014281, -0.183356, -0.025502, -0.231734, -0.145337, -0.02529, -0.118964, -0.130182, 0.0361, 0.06252, -0.290009, -0.358595, -0.160441, -0.166679, -0.069431, 0.011568, 0.078717, 0.098671, -0.208972, -0.371256], [0.095424, -0.165935, 0.145985, -0.097579, -0.399991, -0.076597, 0.02747, 0.146645, 0.055066, -0.077926, 0.073922, -0.02462, -0.437711, -0.142797, -0.233066, 0.132125, -0.017946, 0.113099, 0.100152, -0.075195, -0.078162, 0.058369, -0.025783, -0.038844, 0.065978, 0.019534, -0.097752, 0.127396, -0.445743, 0.080913, -0.664172, -0.756295], [0.074717, 0.102647, 0.179627, -0.041018, 0.131924, 0.023173, 0.247891, 0.217194, -0.195859, -0.138384, 0.235888, -0.148381, 0.083058, 0.188472, 0.135975, 0.237989, 0.043042, 0.13362, -0.135168, 0.029061, 0.110948, 0.162465, -0.054928, -0.152694, -0.006442, 0.137012, 0.082445, -0.115124, -0.031835, 0.228236, -0.007512, 0.05584], [0.051782, -0.025092, 0.030013, 0.039115, 0.141031, 0.138725, 0.38789, -0.00156, 0.027074, 0.102548, -0.145853, -0.382029, 0.233068, -0.158274, -0.650469, -0.064751, -0.157231, 0.053316, -0.147468, 0.183921, 0.042559, -0.088513, 0.060609, 0.060037, 0.041556, 0.096354, -0.162088, 0.029283, 0.22841, 0.034489, -0.142969, 0.101625], [0.291637, 0.091676, 0.229456, -0.189815, -0.006665, 0.09207, 0.484102, -0.079752, 0.26856, -0.334844, -0.137674, -0.537816, 0.054543, 0.08824, -0.007574, 6.6e-05, -0.117303, -0.04375, 0.088366, -0.108163, -0.170682, 0.020995, 0.166608, -0.841298, 0.236167, 0.094245, -0.163842, -0.175281, -0.310341, 0.000957, -0.0596, -0.038654], [0.14369, -0.367387, -0.045557, -0.189552, -0.54192, 0.075281, 0.074696, 0.110004, 0.207589, 0.232588, 0.14007, -0.044578, -0.20876, 0.018889, -0.161517, 0.071441, 0.166146, 0.071529, 0.250044, -0.056663, 0.062089, -0.228669, -0.354461, -0.33507, -0.077738, 0.023958, -0.245652, -0.062561, -0.221304, 0.019211, -0.203429, -0.290777], [-0.503062, 0.006243, 0.031637, -0.30276, -0.096866, -0.156072, -0.310625, -0.058311, -0.109352, -0.327028, 0.155276, -0.412371, -0.090197, -0.036515, -0.534364, -0.257237, -0.146937, -0.449853, -0.089578, 0.126891, 0.164392, 0.056467, -0.193558, 0.172536, -0.302886, -0.018102, -0.074518, -0.165004, -0.222766, -0.006923, 0.047256, -0.045591], [0.135932, 0.216378, -0.180949, 0.247429, 0.029258, -0.070732, -0.073582, -0.180674, 0.198126, 0.236392, -0.048367, 0.313381, 0.17397, -0.029002, -0.351642, 0.216269, -0.087341, -0.020005, 0.156358, 0.243099, 0.056827, 0.0532, 0.175638, 0.10053, 0.301177, 0.080714, 0.127452, 0.336481, 0.182612, 0.304536, 0.236471, -0.024724], [0.157933, 0.235054, -0.000369, -0.062475, -0.147605, 0.131132, -0.050066, 0.090852, -0.011438, 0.033087, 0.18566, -0.385009, 0.025402, -0.102401, -0.216096, -0.029452, -0.015427, -0.105545, -0.308092, -0.028826, 0.153664, -0.040111, -0.107959, -0.303311, -0.106044, -0.176985, 0.170866, 0.047877, 0.052046, -0.183352, 0.151901, -0.014809], [-0.033581, 0.116837, -0.082701, -0.003172, 0.138918, -0.085634, -0.297885, -0.335555, 0.074668, -0.024792, 0.056543, 0.088483, -0.056994, 0.003304, 0.074461, -0.277951, -0.344916, 0.045187, 0.041335, -0.012255, 0.030283, 0.077651, -0.008805, 0.104402, 0.055479, 0.093125, 0.055215, -0.057393, 0.012856, -0.127044, 0.212153, -0.056147], [0.131656, -0.052163, 0.014064, -0.171149, -0.350043, 0.151676, -0.198091, 0.017085, -0.196313, -0.290193, -0.145543, -0.078353, -0.161803, -0.101095, -0.21208, 0.295519, 0.028539, 0.054094, 0.230605, -0.186106, 0.065716, -0.042017, -0.065651, -0.136723, -0.172085, -0.159419, -0.255262, 0.094436, -0.172033, -0.095496, -0.210999, -0.463371], [0.11177, -0.610367, -0.469099, -0.129054, 0.288118, 0.123399, -0.001295, 0.143652, 0.243178, 0.565092, 0.254583, 0.080038, -0.280736, 0.18101, -0.394408, 0.076502, 0.125111, 0.080822, 0.255349, -0.22588, -0.030516, -0.133664, 0.022148, 0.37434, 0.285661, -0.092448, -0.056212, 0.206353, -0.176383, 0.174556, 0.069587, -0.276151], [0.146161, -0.105379, -0.274606, -0.315005, -0.484144, -0.112632, 0.170037, 0.254767, 0.314227, -0.424925, -0.076914, 0.491149, -0.458437, 0.5771, 0.035695, -0.016797, -0.054299, 0.105278, 0.02856, -0.131923, -0.291617, -0.23898, 0.016042, 0.211115, 0.273735, -0.162156, -0.304449, 0.16322, 0.28319, 0.605005, -0.121791, -0.233425], [0.204589, 0.035375, -0.197196, -0.200552, -0.470482, 0.095645, 0.053503, 0.116451, -0.149698, 0.110979, -0.052897, 0.167143, 0.076708, -0.072726, 0.112076, 0.207537, 0.177131, 0.171141, 0.184045, -0.207895, -0.206681, -0.424582, 0.044635, 0.026388, -0.033007, 0.02297, -0.240872, -0.015699, 0.165815, -0.060961, 0.02804, -0.010053], [-0.194803, -0.092372, 0.032556, -0.101334, -0.156363, 0.236926, -0.715493, -0.333177, -0.064509, -0.139905, -0.025304, -0.126628, 0.050931, -0.187328, -0.459605, -0.55471, 0.055632, 0.225606, 0.111846, 0.135407, 0.018054, -0.11063, -0.18953, 0.05888, -0.02711, 0.015725, -0.043942, -0.388378, -0.117632, 0.12036, 0.04281, -0.149015], [-0.023603, -0.016731, -0.019199, -0.080046, -0.166234, 0.115161, 0.189162, 0.091372, 0.262604, -0.3915, -0.275949, -0.032923, 0.0205, 0.049636, 0.159943, 0.128919, -0.338709, 0.007051, -0.228184, -0.00423, -0.014566, -0.063444, 0.057413, 0.075177, 0.464383, 0.078011, 0.055401, 0.022674, -0.466373, 0.414071, -0.222275, -0.106302], [0.115331, -0.097944, -0.339212, -0.277768, -0.075442, -0.245139, -0.045457, -0.041035, 0.143429, -0.210214, -0.266927, 0.609684, -0.206158, 0.248233, -0.153573, -0.028579, -0.16615, 0.010575, 0.003688, -0.530122, 0.092832, 0.271139, -0.153311, -0.42285, 0.090989, -0.123626, 0.036134, 0.193769, 0.223921, 0.261002, -0.229566, -0.115858], [0.044015, 0.20534, 0.042048, -0.113656, 0.020822, 0.03368, -0.05388, -0.208101, 0.13773, -0.144171, -0.071445, 0.014104, 0.129038, -0.229183, -0.192209, -0.225424, -0.070954, 0.058698, 0.08301, -0.148859, 0.038134, 0.024626, -0.278131, -0.050797, 0.008442, 0.081552, 0.067542, -0.106396, 0.021807, 0.085079, 0.128052, -0.112474], [0.291053, 0.132341, 0.082764, -0.015164, 0.134555, -0.085693, -0.053537, 0.119478, 0.144412, 0.276634, 0.582415, -0.280853, 0.401837, -0.161977, -0.278816, 0.063692, 0.036552, -0.166681, -0.675707, 0.16945, 0.161536, 0.183052, -0.593238, -0.331675, -0.010588, -0.050731, 0.255268, 0.04526, 0.189409, 0.349279, 0.210318, 0.67105], [0.093852, 0.213921, 0.154839, 0.079105, 0.002556, -0.142289, 0.27938, -0.306096, -0.135734, 0.117657, 0.014912, 0.299761, 0.15206, -0.103638, 0.135578, -0.101627, -0.163967, 0.188291, 0.089554, 0.044303, 0.1493, 0.108821, -0.386113, -0.130223, 0.192849, -0.144021, -0.127528, -0.226084, 0.180243, 0.100135, 0.227505, 0.252114], [-0.201867, -0.295552, -0.021171, -0.27366, -0.185522, 0.026924, -0.139928, 0.117353, 0.283951, 0.191255, 0.073343, -0.024603, 0.116153, 0.088139, 0.410407, -0.054807, -0.090102, 0.146567, -0.039495, -0.086075, -0.128441, -0.096843, -0.012057, -0.116325, -0.217036, -0.072051, -0.141223, -0.045229, 0.034677, -0.256461, 0.044926, -0.016663], [-0.344565, 0.104777, 0.074065, -0.134774, 0.069819, -0.163661, -0.355347, 0.025019, -0.116338, -0.101613, -0.056285, -0.204894, 0.084459, -0.580726, -0.034199, 0.070079, 0.042371, -0.408414, -0.263177, 0.081529, 0.054183, -0.010898, -0.064219, -0.398186, -0.462871, -0.092616, -0.082764, 0.011378, -0.178389, -0.25942, -0.040644, -0.13056], [0.116112, 0.12323, 0.073305, -0.023482, 0.102984, 0.109056, -0.426661, 0.173178, 0.175194, 0.181368, -0.097181, -0.028973, 0.013739, 0.030665, 0.056204, -0.169524, -0.447571, -0.204341, -0.507125, 0.002565, 0.056119, 0.034074, 0.051567, 0.037364, -0.056893, 0.009888, -0.12586, -0.018889, 0.171316, 0.067619, 0.1232, -0.130856], [0.203125, 0.21769, 0.206167, 0.10944, 0.104749, -0.011246, -0.36165, 0.072739, -0.123424, 0.223553, -0.057755, -0.046404, 0.063815, 0.215342, 0.172834, 0.159496, -0.145756, 0.263329, -0.05534, 0.015416, 0.190599, 0.08283, -0.1367, -0.413295, 0.176888, 0.142486, -0.04544, -0.105384, -0.092453, 0.21925, 0.188062, 0.00248], [0.198185, 0.287909, -0.201876, 0.144017, 0.066764, 0.081458, -0.316726, -0.068334, 0.013035, 0.055475, -0.249156, 0.056737, -0.216928, -0.191562, -0.105788, -0.031012, -0.031567, -0.037271, 0.073177, -0.107138, 0.106199, 0.211827, 0.098741, 0.122977, -0.01121, -0.014582, 0.182517, -0.303267, -0.097054, 0.007285, 0.031331, -0.165689], [-0.368619, -0.048656, -0.164262, -0.124584, -0.082815, -0.229389, 0.112707, -0.162637, -0.246065, -0.014556, -0.13199, -0.319752, 0.199133, -0.414164, 0.462905, -0.105468, 0.075948, -0.057248, 0.153363, -0.00128, -0.153828, -0.004707, 0.038799, -0.047929, -0.419151, -0.150889, -0.184606, -0.252789, -0.250702, -0.104437, 0.061083, 0.313603], [0.070045, 0.039057, 0.080242, 0.032019, 0.089389, 0.093646, -0.257183, -0.028046, 0.114672, 0.002322, -0.036434, 0.067224, 0.075304, -0.00241, 0.041859, -0.096155, -0.001504, 0.037254, -0.231107, -0.108977, -0.064046, -0.067516, -0.143072, 0.050155, 0.001424, 0.200438, 0.102386, -0.05291, 0.16582, 0.182248, 0.162686, -0.050023], [-0.005141, 0.071319, -0.428735, 0.182443, -0.338151, -0.083227, 0.200872, 0.210197, -0.686324, -0.018511, -0.143132, 0.048032, -0.004944, -0.312476, -0.067187, -0.139087, 0.04533, -0.049044, -0.101243, -0.144333, 0.021759, 0.204609, 0.214707, -0.204903, -0.404761, -0.017564, 0.005021, -0.005415, -0.58693, -0.460821, -0.015517, -0.182473], [-0.740225, 0.077503, 0.014899, -0.310718, 0.037772, -0.151564, -0.458819, -0.087322, -0.302048, -0.100365, 0.017124, -0.134363, -0.067453, 0.153784, -0.481049, -0.191591, -0.081997, -0.434366, -0.070575, 0.108078, 0.00813, -0.042541, 0.199061, -0.503342, -0.435017, -0.016513, -0.12694, -0.279372, -0.197387, -0.07594, 0.099112, 0.119012], [-0.202061, 0.264579, 0.291027, -0.034043, 0.103686, 0.021554, 0.308146, 0.131198, -0.853388, 0.186597, -0.073532, 0.202731, 0.246311, -0.107972, -0.027141, 0.172643, 0.322662, -0.131681, 0.143571, -0.09001, 0.124946, 0.168606, 0.157416, -0.294205, 0.000768, -0.284471, -0.027722, -0.063516, 0.238217, -0.320073, 0.210277, 0.135563], [-0.009093, 0.176774, -0.456584, -0.02135, -0.112972, 0.045969, 0.154183, 0.203124, -0.09398, 0.024687, 0.10801, 0.124226, -0.021205, 0.147612, -0.357975, 0.160446, -0.115619, 0.018496, 0.014629, 0.000585, 0.082141, 0.111676, 0.05808, -0.020237, 0.02006, 0.027038, -0.045075, 0.136136, 0.427809, -0.075236, -0.057666, -0.129599], [0.114178, 0.183064, -0.095669, 0.236626, 0.46875, 0.079938, 0.075465, 0.135132, -0.136487, 0.007431, -0.058574, -0.125857, 0.156903, -0.079872, 0.157132, -0.03585, 0.157842, 0.123241, 0.006716, 0.131097, -0.124617, 0.172753, 0.232218, -0.217335, -0.097003, -0.030291, 0.03645, -0.020393, 0.218768, -0.142253, 0.202147, 0.546102], [0.005157, -0.062281, 0.044307, -0.129765, -0.064443, 0.089666, -0.296893, -0.027251, 0.330441, -0.105763, -0.389038, -0.279619, -0.049714, -0.010733, 0.036181, 0.110003, 0.115352, -0.151439, -0.087125, -0.183776, -0.161681, -0.114079, -0.236803, 0.003468, 0.152928, -0.0323, -0.223943, 0.343236, -0.337617, 0.176027, -0.218289, -0.002764], [0.121116, -0.090557, 0.311667, 0.154668, 0.553501, 0.126993, 0.175636, 0.215667, -0.08997, 0.072484, -0.28211, 0.412225, 0.019805, 0.059766, 0.185514, 0.413803, -0.090035, 0.230382, 0.178827, -0.047187, 0.091088, 0.146831, 0.113421, -0.158064, -0.289617, 0.204044, 0.043567, 0.170691, -0.591375, 0.004685, -0.258061, -0.384699], [0.204656, -0.300143, -0.120345, -0.081019, 0.024476, -0.018829, -0.224886, -0.128785, -0.016457, -0.456727, 0.026943, -0.006785, -0.288942, -0.111844, -0.464616, -0.07246, -0.391228, 0.191537, -0.118062, -0.502898, -0.048327, 0.049818, -0.166646, -0.502226, -0.174822, -0.426447, 0.027912, -0.490005, -0.300968, 0.000736, -0.362217, -0.533707], [-0.108238, -0.075662, 0.02427, 0.127958, -0.100579, 0.130155, -0.138485, 0.11835, -0.090371, -0.321989, -0.18833, -0.124324, -0.033851, 0.120471, 0.203243, 0.148325, -0.016657, -0.066176, 0.038115, -0.030844, -0.154219, -0.065327, -0.086654, -0.000947, -0.122098, 0.078396, 0.131832, 0.161556, -0.64886, -0.137841, -0.259721, -0.328794], [-0.187234, -0.015041, 0.135427, -0.08249, 0.047322, -0.128669, -0.005811, -0.308352, -0.119257, 0.117344, 0.000955, -0.007325, -0.218523, -0.572581, 0.080224, -0.09372, 0.05166, -0.176913, 0.117229, -0.099599, 0.045392, 0.109025, -0.000472, -0.286234, -0.263535, 0.099541, 0.03516, -0.986483, -0.002855, -0.274293, -0.174394, 0.183966], [-0.254189, 0.271331, -0.063338, -0.192573, 0.014496, -0.173733, -0.379012, 0.11273, -0.121997, 0.132747, 0.02176, 0.020204, -0.180905, 0.021048, 0.015665, 0.054377, -0.237499, -0.28528, -0.054377, 0.135601, -0.061854, -0.080988, -0.270093, 0.038076, -0.162789, -0.249606, 0.075957, -0.077007, 0.215417, 0.020151, 0.225301, -0.173229], [-0.637091, 0.072443, 0.037854, 0.03879, 0.096708, 0.112879, -0.349213, -0.161635, 0.015889, 0.027188, -0.225051, -0.045683, -0.02319, -0.176119, -0.492371, -0.195722, -0.048789, 0.173779, 0.149543, -0.019932, 0.023803, -0.016268, 0.242756, 0.05316, -0.168257, 0.180376, -0.000576, -0.053214, 0.07435, -0.547, -0.036948, 0.043947], [-0.059488, 0.024153, -0.005073, 0.012206, -0.019938, -0.033409, -0.026603, -0.057802, 0.136917, -0.300727, -0.25583, 0.165746, 0.090602, 0.266869, 0.110318, 0.050919, 0.10598, 0.044708, 0.094297, -0.231221, -0.110095, 0.135226, -0.162936, 0.07117, -0.099617, 0.007234, -0.088167, 0.171224, -0.538972, -0.034748, -0.041863, -0.498022], [0.075253, 0.068452, 0.010577, -0.154777, 0.135888, 0.272264, 0.283384, -0.03772, -0.071766, -0.064461, -0.091812, -0.110261, -0.305366, -0.471177, -0.398368, 0.07227, 0.331214, 0.019069, 0.198675, -0.208583, 0.050179, 0.089916, 0.106651, -0.445805, -0.107814, 0.212204, 0.048172, -0.501458, -0.006453, -0.11028, 0.189172, -0.290561], [-0.222803, 0.154697, -0.098246, -0.047419, -0.170306, -0.09737, -0.242899, -0.222528, -0.197311, 0.115099, 0.046202, -0.028601, 0.112723, -0.091298, 0.076764, -0.024142, 0.018296, -0.140988, -0.131714, -0.298953, -0.340477, 0.160696, -0.257771, -0.249563, 0.129404, 0.063263, 0.0748, -0.300459, -0.069215, -0.297348, 0.005011, 0.021524], [-0.166507, 0.079648, 0.291934, -0.02437, 0.069811, 0.115353, -0.037836, -0.200618, -0.285594, 0.12576, 0.006211, -0.150568, 0.074181, -0.039862, -0.18558, 0.005783, 0.043386, -0.059324, -0.321395, 0.197056, 0.107463, 0.176884, -0.11953, -0.128499, 0.170866, 0.045812, 0.027069, 0.027812, 0.115519, -0.040601, 0.108094, -0.046508], [0.177062, 0.120715, 0.086875, 0.219158, 0.229386, 0.159988, 0.120397, -0.10512, -0.141539, 0.147497, -0.10657, -0.03908, 0.155026, 0.065252, 0.116202, 0.164274, 0.127963, 0.151387, 0.17152, 0.102852, -0.041508, 0.215613, 0.071659, -0.259221, 0.282922, 0.093318, 0.019473, 0.023501, 0.352829, 0.175339, -0.013599, 0.347727], [0.066301, 0.070466, -0.299219, 0.384778, -0.235145, 0.054104, -0.191432, 0.01343, -0.024957, 0.278471, 0.02857, -0.450543, -0.129911, 0.099335, -0.008827, 0.037246, 0.0896, 0.137868, 0.079334, 0.057477, -0.228648, -0.19244, -0.211212, 0.150437, 0.184687, 0.081219, 0.16394, 0.054264, -0.110133, -0.192562, 0.005605, 0.202933], [-0.059382, 0.207482, 0.203306, -0.140348, 0.119732, -0.078727, -0.188127, -0.194887, 0.019154, 0.175783, 0.070268, 0.320489, -0.02992, 0.116275, -0.055966, -0.513795, 0.165149, 0.029529, 0.177206, 0.047721, 0.084264, 0.045466, 0.060709, -0.093691, -0.216642, -0.101976, -0.072663, -0.10179, -0.17664, -0.104761, 0.239808, 0.137881], [0.482163, -0.097552, 0.169238, -0.438757, 0.069824, 0.011723, 0.201796, 0.014889, -0.134269, -0.103929, -0.146992, -0.17336, -0.050403, -0.037322, 0.144773, 0.095526, 0.219325, -0.054822, 0.123578, 0.180401, 0.154127, 0.038459, -0.256142, -0.096699, 0.035085, -0.037054, -0.162916, 0.044452, -0.612797, 0.217948, -0.064699, -0.169145], [0.04153, 0.193899, 0.10341, -0.137347, 0.132869, -0.010456, -0.319734, -0.137252, -0.119662, -0.09795, 0.289886, 0.119448, 0.02471, 0.117412, 0.116269, -0.171231, 0.076208, 0.184915, -0.062223, 0.181409, 0.100544, 0.045428, 0.128178, -0.103381, -0.182297, 0.002088, 0.001946, -0.002807, 0.146736, -0.072055, 0.047703, 0.055446], [-0.447652, 0.029773, 0.046077, -0.088186, -0.061546, -0.205846, -0.436955, 0.068438, -0.202751, 0.027511, -0.222488, 0.231139, -0.135123, 0.11836, -0.120039, -0.233738, -0.067004, -0.150843, 0.174413, -0.320088, 0.009635, 0.049567, 0.174018, 0.389312, -0.156695, -0.150506, 0.391007, 0.243231, 0.108081, -0.171443, -0.154372, -0.333374], [-0.273749, 0.19095, 0.025437, -0.095205, -0.247041, -0.209334, 0.215143, 0.061463, -0.491033, -0.039623, 0.103862, -0.0132, 0.175621, -0.021936, 0.248704, 0.118899, -0.059254, -0.470297, -0.058111, -0.000256, -0.182116, -0.140858, -0.122558, 0.109994, -0.164871, -0.423725, -0.014145, -0.025198, 0.151196, 0.294937, 0.066768, 0.156854], [-0.019551, -0.018931, -0.01214, -0.069535, 0.235146, 0.153728, 0.097413, -0.158054, -0.312247, 0.005218, -0.188645, -0.465239, -0.246866, -0.222647, -0.292129, 0.137259, 0.236122, 0.033919, 0.082004, -0.291103, -0.166136, 0.173185, -0.043716, -0.036431, -0.20402, 0.128352, -0.008389, 0.075799, 0.027792, -0.277992, -0.148727, -0.131939], [-0.349895, 0.091255, -0.041415, -0.22353, 0.064713, 0.049239, -0.381069, -0.220774, 0.193359, 0.086897, 0.052221, -0.015209, -0.069415, 0.077418, 0.077978, -0.142168, -0.347156, -0.266416, -0.168049, -0.154801, 0.03252, 0.023667, -0.401545, 0.103389, 0.096598, 0.075641, -0.012752, -0.156892, -0.148395, -0.261216, 0.030272, 0.043378], [0.454576, 0.041179, 0.16247, -0.040862, -0.203452, 0.131861, 0.267476, -0.086567, 0.495728, 0.012599, -0.076433, -0.119589, -0.073305, 0.05072, -0.387368, 0.059656, -0.032294, 0.309525, -0.112677, -0.253366, -0.121412, 0.128113, -0.199943, -0.236958, 0.498799, 0.06592, 0.111679, 0.054705, -0.199954, 0.445398, -0.035939, -0.138075], [-0.053233, 0.106539, 0.246713, 0.071491, 0.193947, 0.163829, -0.188062, -0.136108, 0.138608, 0.299383, -0.051227, 0.516883, 0.195191, 0.284791, 0.008388, -0.342455, 0.364306, 0.381697, 0.178145, 0.15746, 0.067992, 0.137253, -0.039236, -0.258557, 0.016528, -0.255587, -0.057767, 0.169531, 0.210297, -0.143731, 0.11749, 0.324785], [-0.367371, -0.079089, 0.062941, -0.174699, -0.080589, -0.079017, -0.226543, 0.192241, -0.651307, 0.049358, 0.065756, -0.009644, -0.032863, -0.222591, -0.177322, 0.037004, -0.182024, -0.241739, -0.284802, -0.195201, -0.121593, 0.042655, 0.170663, 0.253767, -0.204095, 0.251717, -0.030974, 0.180447, -0.111774, -0.402758, -0.081421, -0.013363], [0.310765, 0.038844, -0.186251, 0.084459, 0.106954, 0.060204, 0.257721, 0.121347, -0.036576, -0.166233, 0.074731, 0.037511, 0.285855, -0.172229, 0.406864, 0.094846, -0.114306, 0.151604, -0.117634, 0.060842, -0.04757, -0.046702, 0.239409, -0.486722, 0.194978, 0.097936, 0.05995, 0.008076, 0.230092, 0.202047, 0.113665, 0.299767], [0.077268, -0.001403, 0.036189, -0.265938, -0.085098, 0.207115, 0.289478, 0.044969, -0.446809, 0.18203, 0.115908, 0.250227, 0.002605, -0.10251, 0.313655, -0.022539, 0.030441, 0.155361, 0.157887, -0.120563, -0.069027, -0.000493, -0.028981, -0.396135, 0.167273, 0.192671, -0.166734, -0.44388, 0.018572, -0.395519, 0.049666, -0.140576]], [-0.276293, 0.265319, 0.086724, -0.009056, 0.070992, 0.039324, -0.094055, 0.057027, 0.016384, -0.306612, -0.15909, -0.629285, 0.03945, -0.048776, -0.094925, 0.078354, 0.0978, -0.167351, -0.168439, 0.020477, 0.130543, 0.06631, 0.153761, -0.034117, -0.380944, 0.094414, 0.04355, 0.052198, -0.09047, -0.47542, 0.102963, -0.222773]),
    ([[0.250655], [-0.146832], [-0.10045], [-0.187177], [-0.087724], [0.045246], [0.315813], [0.091144], [0.174703], [-0.230023], [-0.140622], [0.266839], [-0.076944], [0.124388], [-0.271889], [0.170933], [0.104097], [0.142767], [0.101021], [-0.102107], [-0.050075], [-0.057343], [0.073597], [0.187404], [0.30656], [0.059526], [-0.05721], [0.100707], [-0.227636], [0.281276], [-0.114402], [-0.270528]], [0.044875]),
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
