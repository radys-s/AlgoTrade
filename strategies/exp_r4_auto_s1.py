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
    ([[0.651683, -0.041094, -0.115138, 0.050263, -0.034488, -0.084897, 0.207851, -0.485468, 0.051456, 0.327977, -0.097791, 0.193623, 0.029432, -0.047423, -0.112842, 0.004146, -0.240815, 0.16442, -0.12108, -0.102823, -0.223613, 0.286998, -0.03493, 0.029135, -0.030462, -0.023639, 0.063561, -0.182547, -0.01871, -0.206384, -0.343649, -0.09389], [0.079922, -0.045462, 0.221774, 0.055083, 0.024768, 0.09576, 0.256357, -0.298292, 0.034306, -0.187356, 0.251776, 0.077665, -0.017024, 0.024756, 0.081174, 0.115748, -0.105186, -0.075386, -0.020922, -0.077579, 0.226468, 0.077567, -0.220386, -0.099573, -0.25261, -0.172861, -0.007792, 0.103457, 0.24432, -0.134048, 0.098981, 0.037841], [0.030661, -0.011978, 0.003749, -0.13513, 0.004322, 0.218808, 0.246388, -0.100548, -0.063966, -0.09394, -0.137714, -0.079324, 0.180659, -0.236219, -0.05235, 0.185053, 0.235942, -0.190108, 0.167706, -0.164669, -0.149566, -0.078549, -0.243217, 0.233827, -0.042756, 0.05006, -0.009588, 0.016122, 0.303101, -0.003693, -0.064008, 0.184815], [0.088115, -0.066518, 0.007843, -0.175404, -0.039658, -0.201426, 0.061072, 0.314989, 0.356553, -0.319997, -0.154411, -0.070663, -0.407207, 0.000955, 0.315353, -0.269894, 0.339122, 0.14696, 0.1691, 0.087294, 0.107721, 0.05111, 0.292745, 0.089854, 0.089628, -0.149263, -0.086217, -0.014127, -0.00992, 0.022783, -0.091159, -0.001371], [-0.167992, 0.526978, -0.037672, 0.457747, 0.107021, 0.176286, 0.108068, 0.585649, -0.073128, 0.00841, -0.172844, -0.460885, 0.375513, 0.037313, -0.359646, -0.396297, 0.359877, 0.153841, -0.242328, -0.039182, -0.012983, 0.02774, 0.076884, 0.078916, 0.137833, -0.150955, -0.236863, 0.219457, -0.24492, 0.215872, 0.095007, 0.086542], [0.097894, 0.251967, 0.104377, -0.042408, 0.161986, 0.041091, 0.173335, 0.123149, 0.068917, -0.190053, -0.161467, -0.127657, -0.032641, -0.228441, -0.156061, -0.060325, 0.109353, -0.015856, 0.361994, 0.397136, 0.240086, -0.184728, 0.076929, 0.164233, -0.035387, -0.001104, 0.060699, 0.272146, -0.140266, 0.157896, 0.137893, 0.060209], [-0.065271, -0.410948, -0.049068, 0.126182, -0.272967, 0.261697, 0.025038, -0.489011, -0.161328, -0.109623, -0.076232, -0.168771, -0.141415, 0.134562, -0.074944, -0.040819, 0.471479, -0.070569, 0.051078, 0.095597, 0.088875, -0.438563, 0.889427, -0.180306, -0.221365, 0.103753, -0.323997, 0.501, 0.288808, 0.093533, -0.118741, -0.136879], [-0.020076, 0.100207, -0.228586, 0.066882, -0.530433, 0.023256, -0.289402, -0.010491, 0.125794, -0.235929, 0.147385, 0.315753, 0.052387, -0.070525, -0.083996, -0.079214, 0.040165, -0.100456, -0.338112, -0.025607, 0.107451, -0.119616, 0.045697, 0.200447, -0.325196, -0.147389, -0.071914, -0.066506, -0.024392, 0.199818, 0.062531, -0.174063], [0.057212, 0.186517, 0.30493, 0.149142, -0.091269, 0.14932, -0.02808, 0.276716, -0.131702, 0.064764, 0.4165, 0.11553, 0.303531, -0.284234, -0.044842, 0.001093, 0.123102, 0.327858, 0.171107, 0.183529, 0.124434, -0.003482, 0.331486, -0.096316, 0.217651, -0.285627, 0.163816, 0.132632, 0.178575, -0.357271, -0.349036, 0.662199], [0.018583, 0.361595, -0.162924, 0.249134, -0.024262, -0.668962, -0.251543, 0.36011, 0.396193, -0.067742, 0.270879, 0.125458, 0.253842, -0.209576, 0.261708, 0.163926, -0.292674, 0.065619, 0.041266, -0.15634, -0.063269, 0.082363, -0.153993, -0.141411, 0.059977, -0.383596, -0.124248, -0.037666, 0.173147, 0.236318, -0.265779, -0.511288], [0.026704, -0.023776, 0.233776, 0.077243, 0.012667, -0.047777, -0.178751, 0.03264, -0.088214, 0.00441, 0.097419, -0.12436, -0.205061, 0.002273, 0.156032, 0.042471, -0.110506, -0.25334, -0.020353, -0.085599, 0.02301, -0.008007, -0.010924, 0.017543, -0.314386, -0.106851, -0.38422, -0.143699, 0.068299, -0.028141, 0.157835, 0.011213], [-0.132069, 0.015245, 0.12913, 0.123763, -0.072098, 0.003725, 0.051317, -0.14013, -0.190134, -0.12638, -0.447955, 0.346705, 0.12474, -0.096243, -0.230242, -0.282768, 0.14836, 0.283782, -0.001074, 0.098206, 0.287399, 0.397917, 0.000771, -0.651102, 0.038078, -0.249836, 0.193866, -0.128833, 0.381596, -0.049596, -0.007047, -0.287348], [-0.071772, -0.139243, -0.153534, -0.035799, 0.182806, -0.065069, -0.165944, 0.038579, 0.006095, -0.047442, 0.063251, -0.187747, -0.119677, -0.079769, -0.30693, 0.111698, -0.025874, -0.11962, -0.159394, 0.003117, 0.102545, 0.044569, 0.169162, -0.344149, -0.113827, -0.088079, -0.076323, 0.135571, 0.071621, -0.028705, -0.16249, -0.164063], [-0.177344, -0.14396, -0.255195, 0.072097, 0.267684, -0.170704, 0.045041, -0.043468, -0.007125, 0.228945, -0.071193, 0.088651, -0.081506, -0.146912, -0.188134, 0.042763, 0.024781, 0.153613, -0.191507, -0.058122, -0.063572, 0.227169, -0.119006, -0.322577, -0.357494, -0.074919, -0.201712, 0.589767, 0.441356, 0.112894, 0.114484, -0.08471], [-0.170017, 0.14866, 0.28507, 0.080315, 0.244415, 0.274944, 0.063816, -0.319169, 0.138862, -0.049244, 0.068331, -0.20036, -0.345262, -0.186676, 0.153872, -0.229741, -0.601829, -0.30555, -0.202553, 0.043643, 0.238793, -0.250835, 0.240512, -0.194783, -0.078601, -0.074578, -0.225893, -0.402761, 0.326119, -0.349688, 0.461005, -0.118211], [-0.139339, -0.079053, 0.103544, -0.134634, -0.019474, -0.128106, 0.161287, 0.034318, -0.021209, -0.046358, 0.047, -0.030473, 0.266024, -0.110517, -0.252955, 0.195268, -0.121355, -0.07908, 0.181425, 0.345988, 0.101822, -0.187155, 0.118304, -0.025523, -0.475657, -0.105582, 0.102161, -0.122034, -0.226735, -0.080302, 0.160693, -0.260194], [0.170146, 0.068753, -0.073417, -0.23053, -0.536795, -0.573466, 0.110933, 0.158409, -0.382239, 0.023074, -0.584557, 0.167939, 0.032595, 0.101005, 0.124215, -0.382105, 0.089541, -0.075111, 0.327385, 0.027603, -0.542111, 0.662627, -0.481919, -0.055449, -0.304403, 0.016499, 0.543736, -0.02255, -0.411289, -0.705912, -0.292372, 0.014612], [0.069034, -0.263872, 0.264537, -0.212444, -0.301823, 0.12015, -0.017666, 0.054723, 0.294589, 0.391046, 0.177646, 0.079322, 0.06019, -0.336903, -0.022791, -0.273246, -0.221851, 0.154163, -0.210091, -0.466195, -0.028097, -0.051097, -0.178595, -0.153776, 0.206773, -0.264611, 0.006573, 0.137827, -0.106524, 0.058433, -0.142317, 0.481716], [0.074158, -0.066763, -0.026565, 0.03725, 0.03166, -0.105351, 0.037287, -0.020317, -0.08592, 0.013763, 0.024578, 0.014762, 0.31189, 0.126164, -0.088678, -0.039959, 0.031486, -0.041515, 0.05789, -0.021811, 0.155171, -0.112694, -0.062241, 0.086022, -0.075017, 0.052886, 0.024631, -0.02809, -0.044221, 0.060528, 0.205655, -0.014352], [-0.033745, 0.182646, 0.199334, -0.150818, 0.058934, 0.211146, 0.427821, 0.105493, 0.086129, -0.166403, -0.208937, 0.05575, 0.213689, -0.090949, -0.633319, -0.235136, 0.079347, -0.106051, 0.425067, 0.081299, 0.059936, -0.350437, -0.266163, -0.243711, 0.111779, -0.102697, -0.130612, 0.078088, -0.109903, 0.209996, -0.106473, -0.021945], [-0.074973, -0.092276, -0.023757, -0.261323, 0.088758, -0.065131, 0.030405, -0.111443, 0.0421, 0.018418, -0.113357, -0.08285, 0.011139, -0.285948, 0.057511, 0.288371, -0.218125, 0.375134, -0.018028, 0.321901, -0.030055, 0.005727, 0.035762, -0.02378, 0.163579, 0.085252, 0.372099, 0.023092, -0.022346, -0.09434, 0.535213, -0.166604], [0.006526, 0.025648, -0.046147, 0.471466, -0.112279, 0.002753, -0.163671, -0.010433, 0.175551, 0.006501, -0.069153, -0.003313, -0.104582, 0.341903, -0.024572, -0.032523, -0.063158, -0.037483, 0.256865, -0.0019, 0.105866, -0.062929, -0.377239, 0.049047, 0.115131, -0.03032, 0.023923, 0.025548, 0.174252, -0.293005, 0.380018, -0.01996]], [-0.131702, -0.043183, 0.133255, 0.148128, 0.225739, -0.299921, -0.05508, 0.155734, -0.217815, -0.725368, -0.221296, 0.010476, -0.077592, 0.103467, -0.066427, -0.167444, 0.025046, 0.027537, -0.374884, -0.607647, 0.146918, -0.165239, -0.023476, -0.230902, -0.234387, -0.37409, -0.429827, 0.376946, -0.431712, -0.529585, -0.292421, 0.105634]),
    ([[0.173325, -0.012774, 0.088257, 0.12741, -0.324658, -0.013055, 0.251965, -0.005211, 0.099322, 0.189327, 0.321472, -0.074244, 0.133105, 0.08802, 0.266747, 0.298324], [-0.264449, -0.060658, 0.290203, -0.083246, 0.178328, 0.267417, -0.014314, 0.076852, -0.153979, 0.082181, -0.290747, 0.165007, 0.035945, -0.010297, 0.195034, -0.000471], [0.064282, 0.034061, 0.110609, 0.150387, -0.491558, 0.07814, -0.138972, -0.060252, -0.146525, 0.149747, -0.239825, -0.227297, 0.135794, 0.179152, 0.065252, 0.131259], [-0.067426, -0.592756, 0.18137, -0.321433, -0.180657, 0.303672, -0.223894, -0.26964, 0.293872, -0.095604, 0.148951, 0.08922, 0.117227, -0.180604, -0.007136, -0.211665], [-0.311925, -0.064938, -0.008504, -0.563413, -0.120609, -0.421148, -0.22931, 0.1292, -0.038811, -0.479344, -0.005723, -0.377241, -0.125577, -0.034564, -0.080493, 0.006797], [-0.072114, -0.006072, 0.021772, 0.215397, 0.13679, 0.049584, -0.043532, -0.191845, -0.063714, -0.007714, -0.166612, -0.056825, -0.357794, -0.320991, 0.143463, -0.218044], [0.233749, 0.413963, 0.070058, 0.122417, -0.125362, 0.088119, -0.184178, -0.043003, -0.256656, 0.057387, -0.063463, 0.04467, -0.189382, 0.170875, 0.185956, -0.332071], [0.220169, -0.112502, 0.249813, 0.291121, 0.278971, 0.084989, 0.340279, 0.188523, -0.087944, 0.067038, 0.105137, 0.118885, 0.280188, 0.037818, 0.51538, 0.163972], [-0.311666, -0.083578, -0.172708, 0.094861, -0.006201, 0.056974, 0.11107, 0.333622, -0.088808, 0.175005, -0.015763, 0.092184, -0.07882, -0.113116, -0.345223, 0.264145], [0.26096, 0.146246, -0.359651, 0.179471, 0.179393, 0.167711, 0.063884, 0.153322, 0.240264, -0.23347, 0.091187, 0.104141, 0.151014, 0.216298, -0.634961, -0.054591], [0.236293, 0.095318, -0.043513, -0.459111, 0.144565, 0.211001, 0.018311, 0.143803, -0.028168, 0.288936, 0.032663, 0.025274, 0.102554, 0.229066, 0.015776, 0.112307], [0.019005, 0.028475, 0.033965, 0.152826, -0.139089, -0.60975, 0.273985, 0.132041, -0.128443, -0.263705, 0.053991, -0.076617, 0.106304, 0.048449, 0.412754, 0.177238], [-0.021409, 0.086958, -0.029925, 0.236819, -0.492605, -0.099316, -0.425123, 0.254417, 0.337435, -0.227557, -0.034131, -0.303042, -0.309901, -0.150875, 0.061208, -0.224096], [-0.195713, -0.198285, 0.065294, -0.196364, 0.13559, -0.037626, 0.222936, -0.177626, 0.199776, 0.265059, 0.070941, 0.065745, 0.177358, 0.157779, -0.292582, -0.038363], [0.186472, 0.166389, -0.225881, -0.548299, -0.088558, 0.068137, 0.158574, -0.15401, 0.201319, -0.406081, 0.278885, -0.2334, -0.01765, -0.045087, -0.081964, 0.223563], [0.097399, 0.247802, 0.14964, 0.568606, -0.119399, 0.277599, 0.191038, -0.235779, -0.078147, -0.424751, 0.256447, 0.080106, 0.064022, 0.106577, 0.209604, 0.346503], [0.157313, -0.020426, 0.039802, 0.101392, 0.168142, -0.111433, 0.341271, 0.048863, -0.252988, -0.052033, -0.062612, -0.615649, 0.137731, -0.755153, 0.117431, 0.162326], [-0.040366, 0.055941, 0.150063, 0.021667, -0.047251, 0.296198, 0.177261, -0.193396, -0.56207, 0.159425, 0.113522, -0.04119, 0.23969, 0.067098, 0.168687, -0.223619], [0.073725, -0.314866, 0.024307, -0.048422, 0.072746, 0.108027, 0.123246, 0.522533, 0.061325, -0.129212, -0.484881, 0.283718, -0.244145, 0.345993, 0.058505, 0.120144], [0.099724, -0.191606, 0.014022, -0.436945, -0.068966, 0.194879, -0.349016, 0.228482, 0.454408, 0.080461, -0.830759, 0.047598, -0.305047, -0.15107, 0.038746, -0.752577], [0.217935, 0.091641, 0.115968, -0.293091, -0.062084, 0.137783, -0.082355, -0.036829, -0.014416, 0.175092, 0.085711, 0.054988, -0.199706, 0.25253, -0.029471, -0.204381], [0.240235, -0.030535, 0.306642, -0.240529, 0.024021, -0.197196, -0.327401, 0.001016, 0.122789, 0.340422, 0.041478, 0.339987, 0.058379, 0.081608, -0.238702, -0.160659], [0.021708, -0.254869, -0.099097, -0.399687, 0.196533, -0.16898, 0.200369, -0.276159, -0.071965, 0.364963, 0.022458, -0.15024, 0.258348, 0.028213, -0.284588, 0.035864], [-0.300307, -0.52171, 0.185125, -0.151397, 0.032078, 0.095883, -0.229872, 0.217692, -0.001182, -0.007615, 0.016774, 0.27655, -0.031866, -0.348056, 0.002697, -0.103419], [-0.517886, -0.117006, 0.058423, -0.483733, -0.059299, 0.065037, 0.158775, -0.072807, 0.207495, 0.178889, 0.173697, -0.106266, 0.10333, -0.000241, -0.315563, -0.104306], [0.054974, 0.242319, -0.16125, 0.310461, 0.287893, -0.052564, 0.239784, 0.211154, -0.054224, -0.384176, 0.107455, 0.148608, 0.025955, 0.22115, -0.05268, -0.103131], [0.002952, 0.418443, -0.153705, -0.303866, 0.105538, -0.408904, -0.061905, -0.086264, 0.313876, -0.218928, 0.187322, 0.152019, -0.070285, 0.033842, 0.133067, 0.197667], [0.139863, 0.101696, -0.015701, -0.050768, -0.023728, 0.08134, -0.456829, 0.240889, -0.206582, -0.132232, -0.008925, 0.028174, -0.206838, -0.156098, -0.23932, -0.434482], [0.02343, -0.255867, 0.025943, 0.496833, 0.534441, 0.285324, 0.181812, -0.489213, -0.393174, 0.230292, 0.25989, -0.163104, 0.160659, 0.050231, 0.148168, 0.28959], [-0.140195, -0.791687, -0.095985, -0.095615, 0.053178, -0.275427, 0.04181, 0.09885, 0.122163, -0.61044, -0.120308, 0.051322, -0.33971, 0.067942, -0.030947, -0.282709], [0.081982, -0.547771, 0.166507, -0.063024, -0.051883, 0.073939, -0.020547, -0.060076, -0.857394, 0.05039, -0.04749, 0.26188, -0.140403, 0.160938, 0.024909, 0.301522], [-0.372648, -0.513988, 0.081091, -0.225415, 0.067802, 0.077653, -0.114788, 0.176522, -0.099284, -0.141894, -0.227571, 0.167518, -0.025749, 0.105895, -0.037711, 0.209725]], [-0.369031, -0.511931, 0.288235, -0.053117, -0.393741, -0.279725, -0.051633, -0.029288, -0.158093, -0.042467, 0.310595, 0.054714, 0.162042, -0.006935, -0.040821, -0.216739]),
    ([[1.068947], [1.215427], [0.441522], [1.225485], [-0.744741], [0.372128], [-0.490984], [-0.3477], [0.911378], [0.685212], [-0.428101], [-0.627215], [-0.442776], [-0.56147], [0.638281], [-0.905537]], [0.045546]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.562321557048493e-19, -1.1343482335572739e-18, -5.671741167786369e-19, -1.8905803892621232e-19, 1.1343482335572739e-18, 0.0, 0.0, 0.0, 1.2572534685440513, 0.0, 10.639801906763806, 28.799029822078616], [6.72849538126497, 19.8058956257327, 34.925161910299096, 44.72666958616459, 57.636846319296275, 1.1243487056677264, 0.9903909371630099, 0.9446913427593254, 0.8452682282888074, 0.7575381583058208, 1067.5606541222405, 2053.035725905977, 4046.2991308026644, 0.6179080587708422, 0.3127841039679992, 16.705331990766382, 34.22120473597514], [0.6490227662551226, 0.3600762349252582, -0.1832944003136108, -0.09733275171505404, -0.10066480452612983, 0.8482856190320313, -3.2486297440276823, 6.1619984101404714, -3.540934484781457, 3.1035095898512166, -0.6019915249010418, 0.06557241931064624, -0.1855095648918967, -0.002102755891520747, 0.4105229362630337, 0.0003153876484541001, 0.00020715056621090425], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.568942287397852e-19, -1.1353413431096778e-18, -5.676706715548389e-19, -1.892235571849463e-19, 1.1353413431096778e-18, 0.0, 0.0, 0.0, 1.2574354160321035, 0.0, 10.646654416476506, 28.81409587438229], [6.7313766144182345, 19.814344420832445, 34.939862329400235, 44.74491730719747, 57.65822173996363, 1.124692105470714, 0.9907164467252751, 0.9450073668099886, 0.8455601770714993, 0.7578101560190531, 1067.786637740178, 2053.787560401509, 4047.179122347928, 0.6181145992138308, 0.3129209935553393, 16.710984248831306, 34.23237508220454], [2.9421164343350763, 0.7964360421909548, -1.0744230210945729, -0.9234918162695617, -0.8475723430517554, 1.4775048291769728, -4.635873707445536, 9.213729195881914, -4.907517756528247, 6.23467901404692, -2.386919742393157, 0.4346797676233983, -1.5368709559241014, 0.001829056109893383, -0.22349228908983945, -0.0002742367847445596, -0.00018005766511644536], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.257441, 0.0, 0.0, 5.84623, 5.846549, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987547], [1.124422, 0.990545, 0.944803, 0.844787, 0.75662, 2.57683, 18.120787, 0.790527, 15.924298, 0.695578, 0.651228, 0.54053, 1.677429, 1.677372, 0.999993, 4.403117, 1.087628, 0.310614, 132.397858, 334.626434, 1066.212891, 1.340704], [([[0.148821, 0.051613, -0.034859, 0.029999, -0.052208, -0.139313, 0.206168, -0.657156, -0.081735, 0.32675, -0.111259, 0.542732, -0.043256, -0.165722, -0.110017, -0.067648, -0.093841, 0.079159, 0.01561, -0.327035, -0.042751, 0.16452, 0.019119, -0.03865, -0.000291, -0.053283, -0.029686, -0.216644, 0.105832, -0.183324, -0.342821, -0.171424], [-0.175429, 0.003613, 0.204809, 0.084812, 0.055741, 0.078918, 0.206523, -0.170567, -0.059173, -0.186757, 0.206958, 0.301486, -0.059462, 0.101371, 0.38131, 0.125687, -0.181806, -0.099857, -0.012101, -0.290891, 0.364971, -0.081629, -0.149367, -0.252478, -0.219036, -0.085576, -0.145825, 0.122842, 0.093219, -0.12012, 0.266671, -0.185132], [-0.126687, -0.059252, -0.094155, -0.14198, 0.037201, 0.217483, 0.157915, -0.002352, -0.084469, -0.14301, -0.138673, 0.069024, 0.165499, -0.074432, 0.183429, 0.19551, 0.08552, -0.225125, 0.137882, -0.244056, -0.084613, -0.151351, -0.111661, 0.111427, 0.023566, 0.132911, -0.096415, 0.022282, 0.117314, 0.003466, 0.048467, -0.00074], [0.370946, -0.37991, -0.119077, -0.348335, -0.052241, -0.008439, -0.058911, 0.239002, 0.442599, -0.411933, -0.043087, -0.042722, -0.50682, 0.131049, 0.249788, -0.231589, 0.243707, 0.009824, 0.051219, 0.303922, 0.062446, 0.336639, 0.384614, 0.129581, 0.1113, -0.148283, 0.154885, -0.06817, -0.216938, -0.015855, -0.135572, -0.051953], [-0.135901, 0.466797, -0.045729, 0.601885, -0.5048, 0.557152, 0.269835, 0.07363, -0.083476, -0.307217, -0.166878, -0.419522, 0.437269, -0.123798, -0.681876, -0.305264, 0.433459, 0.280238, -0.046002, 0.233319, 0.012417, -0.270396, 0.146141, -0.019843, -0.036618, -0.34731, -0.356061, 0.260783, 0.134467, -0.210154, 0.138781, 0.071655], [0.104323, 0.224631, 0.1702, -0.00052, 0.126283, 0.145895, 0.114301, 0.142297, 0.067401, -0.216422, -0.274856, -0.36127, -0.084616, -0.292582, 0.026623, -0.10269, 0.116374, 5.8e-05, 0.290425, 0.331202, 0.279239, -0.260628, -0.056611, -0.039294, -0.020529, -0.060312, -0.101565, 0.170201, -0.130465, 0.127333, 0.107676, -0.15195], [0.120888, -0.486125, 0.003294, 0.382246, -0.183674, 0.514682, -0.014351, -0.281751, -0.190543, -0.182053, -0.180052, 0.290338, 0.157621, 0.232792, -0.122808, -0.420051, 0.355973, -0.054373, -0.174796, 0.112805, 0.006825, 0.138537, 0.302197, -0.151972, -0.107648, 0.161947, -0.327038, 0.292044, 0.024231, -0.092794, -0.157579, 0.044257], [0.083184, 0.084295, -0.096871, -0.019058, 0.071474, 0.161685, -0.162073, 0.219409, 0.297013, -0.060328, 0.149787, 0.012514, 0.089362, -0.105673, -0.157599, -0.038268, -0.1292, -0.038125, -0.016027, 0.039222, 0.125989, -0.014307, -0.243578, 0.067013, -0.335135, -0.013797, -0.037686, -0.067978, -0.180631, 0.066882, -0.195234, -0.164039], [-0.052821, -0.03815, 0.267198, -0.077967, 0.359632, -0.010538, -0.034811, 0.113321, -0.036469, -0.057092, 0.511449, 0.289913, 0.143347, -0.063505, 0.350959, 0.217174, -0.133581, 0.172068, -0.261965, 0.344513, 0.131056, -0.085194, -0.052522, 0.095006, -0.147239, 0.03621, 0.496063, 0.223878, -0.261089, -0.108722, -0.574398, 0.37062], [0.172254, 0.259688, -0.375994, 0.204075, -0.014453, -0.218397, -0.084592, 0.372267, 0.097771, -0.063111, 0.252281, 0.024589, 0.156876, -0.110854, 0.127855, 0.172532, 0.054811, -0.285667, -0.735478, 0.117623, -0.378727, -0.253028, -0.212712, -0.615093, 0.205592, -0.248087, 0.255532, -0.214437, 0.203814, 0.430949, 0.190345, -0.63523], [-0.086374, 0.018625, -0.090155, -0.020146, -0.107087, -0.233639, -0.194882, 0.062582, -0.158166, -0.040011, 0.124005, -0.035129, -0.001418, 0.062441, 0.157479, -0.168357, 0.10227, -0.204037, 0.095522, -0.087501, -0.031177, -0.023364, -0.069301, -0.023018, -0.358504, 0.049772, -0.031558, -0.007411, 0.066184, 0.145817, 0.133838, -0.117005], [0.117471, -0.071359, 0.527303, 0.154198, 0.122802, 0.15199, 0.072927, -0.010181, -0.378509, -0.049482, -0.175733, 0.14814, 0.007097, -0.294487, -0.132782, 0.176717, 0.032411, 0.245473, -0.226756, -0.011557, 0.166447, -0.059229, 0.066017, -0.547978, 0.173346, -0.314933, 0.071884, -0.346941, -0.12021, -0.321121, -0.303138, -0.013691], [-0.210916, -0.172419, 0.322461, -0.089233, 0.076641, 0.078529, -0.125912, -0.141334, -0.258015, -0.025724, 0.053387, -0.104141, -0.150217, -0.044133, -0.091978, -0.149529, -0.108735, 0.019274, 0.010824, 0.110954, -0.037049, 0.079551, 0.12556, -0.061806, -0.05149, -0.159455, -0.249787, 0.07221, -0.130586, 0.17677, -0.233478, -0.150826], [-0.110337, -0.312383, 0.050243, -0.00409, -0.021718, -0.010809, -0.019498, 0.003165, -0.268968, 0.013309, -0.109455, 0.071687, -0.079025, -0.316944, 0.033214, 0.359601, -0.124321, 0.15862, 0.125091, -0.114239, 0.000809, 0.069696, 0.006359, 0.43075, -0.17577, -0.117109, -0.243712, 0.440934, 0.148051, 0.051992, -0.076284, -0.080082], [-0.045743, 0.155185, 0.165087, -0.041066, -0.077051, 0.022719, 0.080454, -0.309794, -0.286461, -0.14012, 0.117077, 0.052445, -0.156323, -0.123252, 0.152749, -0.134953, -0.24719, -0.027474, -0.022239, -0.142377, 0.103778, -0.162743, 0.220989, -0.235114, -0.000401, 0.103887, -0.183226, -0.286262, 0.310843, 0.03162, 0.040674, 0.020194], [-0.05184, -0.135228, 0.083599, -0.38327, 0.125353, -0.120663, 0.123306, 0.054019, -0.175269, 0.017994, 0.116138, -0.171795, 0.24074, -0.079193, -0.011594, -0.093894, -0.153856, -0.111556, 0.133146, 0.134568, 0.329415, -0.155215, 0.01239, -0.196883, 0.026951, -0.124767, 0.110944, -0.185646, -0.154072, -0.058674, 0.135233, -0.125436], [0.056681, -0.037601, 0.087036, -0.028337, -0.064876, -0.248361, 0.035786, 0.10742, -0.050539, 0.337516, -0.902754, 0.144283, -0.212807, 0.098281, 0.010195, -0.137025, -0.229336, 0.034153, -0.027544, 0.004709, -0.240087, 0.219179, -0.475315, 0.095961, -0.071578, -0.123427, 0.196722, -0.343436, 0.436032, -0.251351, -0.086743, 0.224176], [0.260826, -0.061752, 0.440724, -0.142266, -0.098512, -0.263715, -0.133391, 0.163247, 0.158927, 0.206479, 0.196576, 0.028638, -0.002481, -0.317143, -0.008284, -0.260119, -0.076814, 0.278834, -0.643528, -0.093937, 0.072514, -0.209752, -0.287423, -0.198582, 0.167814, 0.027573, -0.112064, 0.268627, 0.066346, -0.294245, -0.390338, 0.263598], [-0.021546, 0.059263, -0.015623, -0.015858, 0.012418, 0.025852, -0.047136, 0.020035, -0.029048, -0.075723, 0.112285, 0.029309, 0.141498, 0.085946, 0.016126, -0.049211, -0.052329, 0.001506, -0.036149, -0.057985, 0.136075, -0.082114, 0.025937, 0.06498, -0.013757, -0.013655, -0.137554, -0.03077, 0.005298, -0.188627, 0.230107, 0.018427], [-0.231392, -0.040911, 0.05524, -0.058549, -0.142928, 0.147993, 0.263043, 0.00704, 0.039502, -0.169408, -0.125514, -0.006658, 0.353939, -0.11257, -0.206039, -0.13332, -0.000149, -0.208617, 0.178817, -0.131053, 0.021797, -0.710375, -0.134601, -0.187668, 0.235621, 0.003719, -0.163218, 0.07701, -0.126563, 0.32391, 0.130073, 0.006932], [0.302431, 0.001573, 0.148681, -0.118691, 0.604831, -0.240806, 0.089312, 0.094568, 0.063187, 0.308012, -0.03626, -0.054899, -0.030435, -0.366797, 0.008089, -0.022032, -0.057426, 0.466757, -0.095576, 0.100425, 0.130384, 0.11295, -0.267939, -0.082465, -0.214124, 0.241904, 0.435489, 0.10469, -0.084242, -0.081599, 0.157823, -0.121994], [-0.059249, 0.058085, 0.181408, 0.198908, 0.006206, -0.084917, -0.161547, 0.089153, 0.070454, 0.020874, -0.021589, -0.016, -0.083466, 0.231305, 0.077858, -0.063439, -0.001914, -0.103558, -0.056802, 0.027692, -0.059235, -0.106584, -0.357396, 0.10537, 0.350447, -0.10035, -0.01909, -0.058671, -0.082659, -0.018931, 0.541865, 0.008395]], [0.363542, 0.016925, 0.102222, 0.012787, 0.221766, -0.819481, -0.006685, -0.01701, -0.049273, -0.794959, -0.015142, -0.181097, -0.062363, 0.030854, -0.21741, -0.125499, 0.379165, -5.9e-05, 0.042327, -0.645968, 0.187762, 0.382007, -0.1283, -0.206469, 0.120987, -0.105592, -0.168045, 0.139573, -0.601967, -0.290926, -0.21078, -0.325469]), ([[0.050347, 0.030883, 0.147721, 0.167428, 0.284709, -0.049603, -0.051134, -0.158584, 0.01907, 0.112226, 0.169327, -0.289739, 0.1417, 0.088481, -0.005572, 0.381696], [-0.199134, 0.140786, 0.314216, 0.0292, 0.095731, 0.294291, -0.005008, -0.092371, -0.18129, 0.022524, -0.139198, -0.136277, 0.165559, -0.106142, 0.213049, -0.091338], [0.075049, -0.232621, 0.11526, 0.015982, -0.025304, -0.164943, -0.683473, -0.647064, 0.115096, 0.18857, -0.371984, 0.181964, 0.076326, 0.112031, -0.062614, 0.073951], [-0.023911, -0.01572, 0.11705, -0.279642, -0.462938, 0.104163, -0.280891, -0.271927, 0.449428, -0.096829, -0.041407, 0.238901, -0.070654, -0.230239, 0.10367, -0.284739], [-0.164813, 0.397633, -0.081286, -0.385706, -0.055454, -1.122321, 0.115082, 0.43451, 0.075386, -0.265205, -0.142367, -0.137043, -0.025293, -0.037314, -0.101005, -0.051166], [0.045861, 0.071975, 0.016497, -0.181312, -0.013755, -0.096872, 0.390588, 0.382829, -0.718731, -0.016998, -0.275602, -0.727041, -0.494328, -0.575212, 0.118743, -0.502718], [0.159485, 0.058606, 0.041615, 0.164037, -0.05012, 0.085879, -0.073433, 0.037294, -0.035011, -0.043056, -0.045969, -0.08518, -0.210056, 0.046908, 0.162763, 0.063508], [0.170679, -0.170136, 0.088962, 0.391017, 0.034273, 0.073488, 0.299399, -0.112358, -0.048808, -0.334685, 0.088909, -0.31596, 0.147714, -0.069889, 0.318158, 0.22027], [-0.305772, 0.109813, -0.296191, -0.01653, -0.379366, -0.043941, 0.062445, 0.154067, 0.147585, 0.109348, -0.182874, -0.058176, -0.051962, -0.195959, -0.249342, 0.167496], [0.605961, 0.175408, -0.740104, -0.478647, 0.175748, -0.150646, 0.004604, 0.047177, 0.175752, -0.275073, 0.012423, 0.041056, 0.110126, 0.165503, -0.386288, -0.003138], [0.081125, 0.010755, -0.116271, -0.568313, 0.253211, 0.298313, 0.029579, -0.062727, -0.144288, 0.271934, 0.142711, -0.153351, 0.193142, 0.189534, -0.301786, 0.069613], [0.151111, -0.199254, 0.043087, 0.230299, 0.434179, -0.119126, 0.353026, -0.625696, -0.213384, -0.048231, 0.023929, -0.139077, 0.186429, -0.000158, 0.417496, 0.115655], [0.018043, -0.339568, 0.067122, 0.304393, -0.471038, -0.204042, -0.245655, 0.421466, -0.306541, -0.181044, -0.342454, 0.243472, -0.289987, -0.235664, 0.147832, -0.239384], [-0.429138, -0.07117, -0.106228, -0.112095, 0.102059, 0.182819, 0.142404, 0.09365, 0.092748, 0.303446, 0.085465, 0.215381, 0.2248, 0.165856, -0.136357, -0.025487], [0.161889, 0.516453, -0.103248, -0.538999, -0.038143, -0.133623, 0.276044, -0.300859, 0.238916, -0.205803, 0.257349, -0.025058, -0.075643, -0.048339, 0.163269, 0.149515], [0.215754, 0.043595, 0.175494, 0.314236, 0.014816, 0.079319, 0.099274, -0.355444, -0.078645, -0.038365, 0.107502, -0.150124, 0.105934, 0.019364, 0.3865, 0.242355], [0.183698, -0.126441, 0.106564, 0.088647, -0.047221, 0.08536, 0.332686, 0.076966, -0.302929, -0.054439, 0.257476, -0.264799, 0.155774, -0.295041, 0.176864, -0.024039], [-0.018414, -0.293442, 0.393107, 0.030021, -0.268729, -0.03236, 0.189102, -0.133897, -0.44607, 0.080749, 0.238872, 0.387853, -0.093009, -0.117124, 0.244388, -0.228131], [0.12503, -0.357257, 0.166769, 0.072216, -0.160487, 0.00255, -0.442327, -0.560864, -0.052951, -0.072905, -0.351701, 0.033043, -0.222125, 0.173925, 0.188821, 0.124957], [0.380562, -0.143827, 0.19866, -0.093875, 0.253552, 0.552224, 0.52441, 0.899129, -0.303434, 0.173239, -0.238006, 0.282267, -0.140561, -0.175016, 0.391283, -0.250605], [0.184133, 0.244556, 0.151033, -0.241165, -0.145616, 0.131464, -0.452673, -0.231134, 0.338535, 0.165197, 0.231347, 0.157527, -0.079097, 0.211335, 0.114182, 0.043883], [-0.09211, 0.141726, 0.128469, 0.018655, 0.044856, -0.138706, 0.062932, 0.139551, 0.266756, 0.125659, 0.181546, 0.153094, 0.220823, 0.06794, -0.843249, -0.018882], [-0.551556, -0.905451, -0.13942, -0.179562, -0.078611, 0.067188, 0.058387, 0.127474, 0.049561, 0.151631, 0.049968, -0.007514, 0.232448, -0.028834, -0.199385, -0.138422], [-0.523105, -0.279131, 0.256615, -0.094687, 0.225427, -0.362227, -0.119147, 0.038422, -0.012897, -0.063942, -0.095876, 0.118692, 0.004532, -0.186759, -0.22001, 0.086358], [-0.556426, -0.151738, -0.10101, -0.039121, -0.202114, 0.143749, 0.104682, 0.166051, 0.109978, 0.141911, 0.149281, -0.006196, 0.04465, -0.022111, -0.14287, -0.00343], [0.043459, 0.000496, -0.095597, -0.405067, 0.277584, 0.069075, 0.212969, -0.301386, -0.082408, -0.109402, 0.015044, 0.225071, 0.007238, 0.176831, 0.423207, -0.225573], [0.096453, 0.084387, -0.183859, -0.448889, 0.190044, -0.401571, -0.226217, -0.197144, 0.185762, -0.325745, 0.016106, -0.128658, -0.031143, 0.002385, 0.008843, 0.003478], [-0.002626, 0.013138, 0.096274, -0.135915, -0.139758, 0.044997, -0.290106, 0.171895, -0.097183, -0.192875, -0.045827, 0.340409, -0.351412, -0.218527, -0.198432, -0.17236], [-0.198322, 0.389952, -0.029152, -0.028175, -0.128815, 0.122764, 0.201705, -0.324016, 0.345616, 0.330517, 0.100977, -0.037574, -0.335092, -0.019101, -0.255309, -0.171763], [-0.141184, 0.08163, -0.062741, -0.026764, 0.136108, -0.178778, 0.113857, 0.214292, -0.262174, -0.369018, 0.031368, -0.232974, -0.125798, 0.203033, 0.11158, 0.016609], [-0.239864, -0.44684, 0.137188, -0.126418, -0.012348, -0.01527, -0.179089, -0.125654, -0.034529, 0.068335, -0.16854, -0.066235, -0.232107, 0.336195, -0.00988, 0.243262], [-0.218711, -0.374563, -0.089264, 0.280683, -0.089406, 0.281799, 0.048097, 0.368597, -0.057301, 0.056712, 0.030889, 0.161931, 0.16805, 0.073893, -0.090651, 0.191557]], [-0.4067, -0.374297, 0.249803, 0.004878, -0.134753, 0.108009, -0.035815, -0.481897, -0.292014, 0.236203, 0.198886, 0.23905, 0.118993, 0.04222, -0.014381, -0.07545]), ([[2.258867], [1.806835], [1.041466], [2.107353], [-1.24939], [1.020805], [-1.030591], [-2.641348], [1.595492], [1.01104], [-0.569611], [-1.025463], [-0.874394], [-0.825485], [1.267637], [-1.099364]], [-0.11302])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.75191, -9.2e-05, 9.2e-05, 0.0, 0.052846, 0.052846, -0.0, 0.0, 0.0, 1.243668, 5.879185, 5.879185, 0.0]
SEQ_TSD = [1.079295, 0.132411, 0.132411, 0.099558, 1.45424, 1.45424, 0.666687, 0.219428, 0.17921, 0.60273, 1.666237, 1.666237, 0.649942]
SEQ_XM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.257423, 0.0, 0.0, 5.846561, 5.846561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987482]
SEQ_XS = [1.124955, 0.99101, 0.94523, 0.845186, 0.757041, 2.577918, 18.130564, 0.790721, 15.945622, 0.69697, 0.6513, 0.540497, 1.677766, 1.677766, 0.999993, 4.402977, 1.087599, 0.310604, 132.411407, 334.802429, 1066.728027, 1.341266]
SEQ_YSC = 1.939010739326477
SEQ_WI = [[-0.350319, -0.692666, 0.370612, -0.564874, -0.281368, -0.014702, -0.61526, -0.749857, -0.217527, -0.010025, 0.019446, -0.348418, -0.26742, -0.393962, -0.581613, -0.05086, 0.062852, 0.070486, -0.104065, -0.30042, -0.310648, 0.575254, 0.200198, -0.456078, -0.199061, -0.320003, -0.131799, 0.158209, 0.112556, 0.256146, -0.111071, -0.079307, -0.240949, -0.359382, -0.385661, -0.027719, 0.14509, 0.036748, -0.196686, -0.427457, -0.305237, 0.063043, -0.255483, 0.070201, -0.246041, -0.107409, -0.237863, -0.001265], [-0.619132, 0.300881, -0.318454, -0.352453, 0.178915, 0.541686, 0.094534, 0.18133, -0.047189, 0.129831, -0.448892, -0.241981, -0.41987, -0.045412, -0.287703, -0.042537, -0.035228, -0.166093, 0.389391, -0.375412, 0.120207, -0.137444, 0.885915, -0.789393, -0.014427, -0.225953, -0.620358, 0.093129, 0.117604, -0.255333, 0.070025, 0.248233, 0.107114, 0.098896, 0.125514, -0.453398, 0.02275, -0.007604, -0.102698, 0.068833, -0.133942, -0.144991, 0.410651, -0.072413, -0.107529, 0.0716, 0.202736, -0.159407], [-0.055205, 0.327021, -0.52745, 0.225272, 0.430423, -0.02195, -0.169358, 0.183831, -0.533299, -0.181844, 0.229163, -0.172879, 0.608189, 0.002097, -0.676039, 0.024916, 0.048388, -0.638732, -0.458596, 0.121249, -0.17531, -0.04536, -0.137107, 0.125302, 0.451658, 0.034212, -0.015039, -0.243307, -0.343635, 0.102908, 0.227046, -0.485335, 0.031814, -0.089598, -0.029887, -0.325447, -0.0647, 0.270384, 0.207289, 0.133058, 0.209364, -0.245246, -0.12228, -0.091073, 0.073288, 0.202002, 0.020816, 0.285699], [-0.320956, 0.454855, -0.359732, 0.498686, 0.47289, 0.303978, -0.267866, 0.214686, -0.303321, 0.186358, 0.159799, -0.158643, 0.211225, 0.03981, -0.237732, 0.07841, -0.15079, 0.002868, 0.214875, 0.101472, 0.079628, -0.44902, 0.033784, 0.10481, -0.014638, -0.426787, -0.697258, 0.19518, -0.147329, -0.123257, -0.046854, -0.271364, -0.194107, 0.076961, 0.217829, 0.18459, 0.014927, 0.20163, -0.300005, -0.100367, 0.068135, 0.113405, 0.13088, -0.103671, -0.187543, 0.321143, 0.015376, 0.163485], [-0.865512, 0.310413, 0.432238, -0.190854, 0.032374, -0.290004, -0.503693, 0.238512, 0.398022, 0.041193, 0.217402, 0.154684, -0.011743, -0.119918, -0.036051, -0.47958, -0.464777, -1.070318, 0.194752, -0.58985, -0.592801, 0.270231, -0.043232, 0.31272, -0.344302, 0.166865, -1.180033, 0.329018, 0.074885, -0.916952, 0.370399, 0.624655, -0.707299, 0.053983, -0.195641, -1.142492, 0.144981, -0.008811, -0.105207, -0.170954, 0.040379, -0.04915, 0.657877, -0.12863, -0.112922, 0.78724, 0.490261, -0.073001], [-0.260255, 0.199082, 0.597759, -0.526922, -0.739593, 0.145142, -0.25343, 0.307952, -0.309123, -0.740096, 0.542177, -0.280617, -0.300967, 0.10361, -0.19149, 0.369213, 0.602567, -0.649925, -0.112846, 0.005332, -0.223587, -1.089813, -0.213758, -0.196731, -0.285764, -0.321053, 0.130788, -0.344541, -0.563416, 0.305127, -0.479249, 1.022094, 0.421693, 0.404572, 0.064106, 0.219088, 0.585364, -1.039046, -0.248941, 0.187273, -0.300321, -1.185669, -0.056171, 0.774178, 0.023025, 0.037367, -0.088853, 0.031206], [0.038054, 0.02539, 0.135593, 0.23674, 0.243868, -0.243524, -0.002965, -0.368521, 0.067106, 0.072556, -0.009452, -0.313744, -0.224693, -0.062942, -0.502432, -0.554744, 0.104528, -0.033054, 0.033987, -0.291416, -0.461859, 0.441367, 0.318595, 0.266074, -0.037874, 0.201038, -0.226652, -0.047688, -0.083459, 0.087128, 0.22154, 0.094429, 0.006379, -0.265832, -0.08226, -0.176303, 0.121003, 0.091223, -0.04068, -0.206321, -0.019041, 0.352825, 0.399595, 0.032499, 0.173065, -0.040569, -0.097974, 0.078779], [0.170611, -0.170904, -0.274723, 0.065722, 0.181837, -0.047449, -0.14977, -0.482619, -0.007797, 0.532945, -0.245029, -0.354585, -0.092244, 0.062587, -0.587888, 0.24239, -0.052723, 0.291776, 0.297691, -0.161744, -0.563894, 0.195249, -0.05901, -0.249453, -0.006551, 0.111918, -0.160214, -0.054609, 0.258345, -0.020359, 0.026015, 0.035774, 0.152899, 0.086127, -0.109251, 0.226277, 0.301812, -0.087706, -0.051615, 0.219599, -0.09897, 0.150158, -0.217941, -0.16568, -0.082198, -0.086574, -0.261991, 0.162546], [0.273975, -0.126243, -0.474407, 0.060683, -0.026967, -0.018746, -0.395391, -0.583813, -0.251191, 0.449749, -0.401119, 0.02415, -0.372606, -0.182484, -0.298079, 0.061601, 0.081955, 0.549928, 0.541161, -0.34047, -0.302608, 0.304274, -0.15591, 0.11994, -0.412422, -0.434478, -0.109261, -0.42125, 0.351473, 0.550353, 0.228701, 0.109245, -0.08723, 0.325562, 0.193342, 0.066877, -0.22466, 0.029362, -0.216754, -0.33394, 0.030249, -0.219372, -0.1476, -0.106566, 0.168433, -0.086719, 0.134388, 0.23871], [-0.347594, -0.649838, -0.154996, -0.792969, 0.329815, 0.323888, 0.187178, 0.806775, -0.103214, -0.318347, -0.886902, 0.193926, 0.135014, -0.505899, -0.078789, 0.041326, -0.670774, 0.045209, -0.094357, 0.61697, -0.171483, 0.236486, -0.391635, 0.21371, -0.419107, 0.628067, 0.107579, -0.334107, 0.117229, 0.020943, 0.400531, -0.04012, 0.279922, 0.363716, 0.1793, -0.012044, 0.0923, 0.002325, -0.030759, -0.587006, -0.048701, -0.297131, 0.101337, 0.070449, 0.100395, -0.077506, 0.302179, 0.124643], [0.013802, -0.101004, 0.291884, -0.120957, 0.261465, -0.007385, 0.158042, -0.057787, 0.235964, 0.303426, -0.047787, -0.300904, -0.46634, -0.173806, 0.195657, -0.623535, 0.283759, -0.390851, -0.420897, 0.295851, -0.118476, 0.272993, -0.305799, 0.365847, -0.067881, -0.407676, -0.470532, -0.106751, -0.197963, 0.137021, -0.445234, -0.28938, -0.115457, 0.18338, 0.256623, 0.091359, 0.162344, 0.089488, 0.085206, -0.101469, 0.070781, -0.125631, 0.173281, -0.084738, -0.058176, 0.005203, -0.078037, 0.060077], [0.009329, -0.052108, -0.260413, -0.203819, 0.079776, -0.107835, 0.073666, 0.016315, 0.578106, 0.136511, 0.05363, -0.55767, 0.020024, -0.069038, 0.043739, -0.388099, 0.203728, 0.480014, -0.247471, -0.136485, -0.005139, -0.517868, 0.058242, -0.103578, -0.264498, -0.294349, -0.184652, 0.924636, -0.145878, -0.090754, 0.301111, -0.222914, -0.19557, 0.025447, 0.139114, -0.160316, -0.13414, 0.250251, 0.048737, -0.132627, 0.003568, -0.119953, -0.109685, 0.002708, 0.287358, -0.04687, 0.056503, 0.267506], [0.017227, 0.042068, 0.443574, -0.146279, 0.239408, 0.188028, 0.394017, -0.060263, 0.06542, 0.420798, -0.188725, -0.380374, -0.711776, -0.256602, 0.225686, -0.241964, -0.151003, -0.213919, -0.205425, -0.055153, -0.416738, 0.384517, -0.590635, -0.013975, 0.51742, -0.24816, -0.325856, -0.281329, -0.364425, -0.072012, -0.113429, 0.533485, -0.077071, 0.041593, 0.31338, -0.018475, 0.014524, -0.330091, -0.124441, 0.075327, 0.168348, 0.021804, -0.120756, 0.082083, -0.11675, -0.140948, 0.090898, 0.00768]]
SEQ_BI = [0.062491, 0.229623, -0.284672, 0.26279, -0.150082, -0.355407, 0.284098, 0.036139, 0.029738, -0.083855, -0.046456, 0.004822, -0.32132, 0.158734, 0.267379, 0.102049, 0.871667, 0.650934, 0.765468, 0.923963, 0.43921, 0.48583, 0.506211, 0.658311, 0.638318, 0.702239, 0.682678, 0.318953, 0.041411, 0.399801, 0.842924, 0.350323, 0.038354, 0.143746, -0.001192, 0.222393, 0.143006, 0.270255, -0.070037, -0.139908, 0.194497, 0.06188, 0.257882, -0.005693, 0.413429, -0.233363, 0.109537, -0.263898]
SEQ_WH = [[0.252555, 0.213697, -0.130561, -0.144321, 0.491608, 0.290977, 0.361157, 0.536106, -0.045142, 0.290722, -0.435316, 0.55888, 0.107014, 0.210146, -0.474086, -0.149499, 0.844755, 0.160585, 0.206922, 0.379713, 0.369846, -0.028836, 0.486388, 0.349691, 0.005909, 0.146969, -0.026898, 0.445099, -0.323837, -0.354781, -0.049381, 0.202438, 0.710484, -0.120519, -0.10083, 0.2835, 0.102823, 0.235228, 0.335818, -0.132941, 0.460392, -0.044633, -0.303263, -0.394274, 0.132375, -0.400203, 0.232315, 0.158378], [0.051964, -0.245138, 0.261341, -0.243288, -0.172483, -0.370243, 0.283965, 0.155054, -0.815605, -0.107441, -0.640178, -0.049938, -0.268868, -0.176032, -0.278829, 0.39419, -0.28054, -0.015587, -0.058434, 0.365606, -0.132998, -0.266779, 0.257019, -0.847042, 0.552842, 0.04909, 0.180114, -0.142225, -0.468237, 0.454426, 0.335943, -0.01103, 0.108063, 0.125344, -0.350323, 0.012822, 0.468434, -0.218745, -0.446231, -0.434124, -0.099999, -0.229943, 0.207568, 0.367716, -0.211048, -0.073125, 0.101941, 0.162902], [-0.17136, 0.544019, 0.130056, -0.016117, 0.485779, 0.141692, 0.387332, 0.184771, 0.140908, -0.107482, -0.321305, 0.168225, -0.330148, -0.020929, 0.077289, -0.582561, 0.012644, -0.252211, 0.169883, 0.369897, -0.756441, -0.159036, 0.257119, -0.083401, 0.943848, -0.016026, -0.536769, -0.426688, -0.148814, 0.159385, 0.668155, -0.326474, -0.342697, -0.085798, 0.221478, -0.102698, 0.210645, -0.003513, 0.319587, 0.30966, 0.277917, -0.242498, 0.389505, -0.376907, -0.207768, -0.332125, -0.128552, -0.152778], [0.364076, 0.03275, 0.171922, 0.192241, -0.151767, 0.261027, -0.020101, 0.410989, -0.234038, -0.114556, 0.104674, 0.283225, -0.08696, -0.014349, -0.44126, 0.162387, 0.481623, -0.671345, 0.33271, -0.150235, -0.051254, -0.532966, 0.412834, 0.566685, 0.45843, 0.063013, 0.271538, 0.459912, -0.936804, 0.106665, -0.18284, -0.000726, 0.281924, 0.110775, 0.088496, 0.574043, 0.457978, -0.278255, -0.272312, 0.211337, 0.008956, -0.380561, -0.335424, -0.225777, 0.544792, -0.036888, 0.010107, 0.212911], [0.12706, -0.141173, -0.100821, 0.145891, -0.263811, 0.175071, 0.132137, 0.193299, 0.009294, -0.242569, 0.16679, 0.026117, 0.06921, 0.622149, -0.882746, 0.264745, 0.336661, -0.979002, 0.054493, 0.024507, -0.525387, 0.094885, -0.063063, 0.004457, 0.345701, 0.016603, 0.120593, 0.231788, -0.430298, -0.10759, 0.188225, 0.392951, -0.119044, 0.386884, 0.241865, 0.208245, 0.52086, -0.34637, -0.181474, 0.047051, -0.173532, -0.093148, 0.015057, -0.132799, -0.19924, 0.015436, 0.484754, 0.051813], [-0.033286, 0.151988, 0.11555, -0.257498, -0.062897, -0.194243, 0.128032, -0.748635, 0.028462, 0.08737, -0.376961, 0.167487, 0.607534, -0.169044, -0.044953, -0.138236, 0.849371, 0.065154, 0.19359, 0.697882, 0.438856, 0.408991, -0.142916, 0.234691, -0.44002, 0.43665, 0.752218, -0.005674, 0.29145, 0.29747, 0.543319, 0.540371, 0.365691, -0.271632, -0.437006, -0.204561, -0.465502, 0.16924, 0.050467, -0.348702, 0.367826, 0.377406, -0.144636, -0.030139, -0.10148, -0.354824, 0.180615, -0.104803], [0.116331, 0.084778, 0.242504, 0.1717, 0.068208, 0.557332, 0.042001, 0.628847, 0.371447, 0.564412, -0.108153, 0.253632, 0.314402, -0.321499, 0.157693, -0.32712, -0.005457, -0.381183, 0.547217, -0.814968, -0.382505, -0.114129, 0.054093, -0.537323, -0.930273, -0.879236, -0.441171, -0.065144, 0.401869, -0.619903, -0.801884, -0.377271, 0.306091, -0.217808, 0.239628, 0.411339, -0.260858, -0.277914, 0.591524, 0.597652, -0.019706, 0.145528, 0.027304, 0.047814, -0.191136, 0.1415, 0.212821, 0.459908], [0.356403, 0.460964, -0.124214, 0.50363, -0.03048, 0.746522, -0.046164, 0.052421, 0.457263, 0.307085, 1.252322, 0.761201, 0.16054, 0.171616, 0.769343, 0.503464, 0.081555, -1.183694, 0.379386, 0.602708, -0.516037, 0.625758, 0.532566, 0.437223, -0.62639, 0.821584, 0.148954, -0.040227, 0.330666, 0.064266, -0.049509, -0.514527, -0.442423, -0.924201, 0.270894, 0.201122, -0.134537, 0.107983, 0.23734, 0.880834, -0.060229, 0.345041, -0.219016, -0.355206, 0.428717, -0.098249, 0.081399, 0.357853], [-0.557795, -0.072975, 0.376869, -0.16633, 0.242125, 0.071424, -0.045088, -0.080652, -0.212239, -0.354593, -0.251089, -0.046339, -0.574027, -0.235258, 0.078653, -0.783562, -0.367918, 0.193341, -0.142014, -0.368075, -0.059004, 0.415781, -0.17902, 0.230112, -0.111395, 0.318915, -0.24534, -0.789428, 0.127228, -0.603769, 0.063708, -0.170111, 0.340297, -0.390909, -0.213344, -0.054999, -0.082433, 0.396635, 0.345416, 0.153964, 0.562179, 0.151151, -0.202779, -0.353121, 0.097421, -0.656669, 0.440753, 0.204885], [-0.086827, 0.127868, 0.159819, 0.277558, 0.021292, -0.134185, 0.011313, -0.272067, -0.136539, -0.071455, -0.423496, 0.106772, -0.789353, -0.267873, 0.296777, -0.010129, 0.492415, -0.512047, 0.129983, 0.420407, 0.439827, 0.386943, -0.004084, 0.052533, 0.191928, -0.082322, 0.752813, -0.288576, 0.329095, -0.033854, 0.714989, -0.398674, -0.115094, -0.064503, 0.292414, -0.291082, -0.498825, 0.076075, -0.359368, -0.133028, 0.297512, 0.415164, 0.34789, 0.062334, 0.241909, -0.130233, 0.12819, -0.712821], [0.088996, -0.101939, 0.134249, 0.032562, 0.389476, -0.331365, 0.584079, 0.144146, 0.539174, 0.08759, 0.18522, 0.278009, -0.230821, 0.233805, 0.162379, 0.14399, 1.091888, 0.625097, 0.624652, 0.927307, 0.551982, 0.809351, 0.725657, 0.20743, 0.501192, 0.447241, 0.601868, 0.5479, 0.645747, 0.413468, 0.932695, -0.118983, -0.249451, 0.167832, -0.131345, -0.172332, 0.062988, 0.036596, -0.080588, -0.148881, -0.178809, 0.288832, 0.134739, 0.126909, -0.205452, 0.155221, -0.152726, -0.378955], [0.05853, -0.301386, -0.307104, -0.403513, -0.241909, 0.14403, -0.417333, 0.001472, -0.237058, -0.435771, 0.135358, -0.352106, -0.208037, 0.026929, -0.907588, -0.103412, -0.996284, 0.15332, -0.051884, -0.358578, -0.154227, -0.663858, -0.22596, -0.615613, -0.357219, -0.399575, -0.633322, -0.042129, -0.406564, -0.280271, -0.112387, 0.357279, -0.020118, -0.002575, -0.501462, 0.349705, 0.267481, -0.08192, -0.32697, 0.10958, -0.374855, -0.200346, 0.120349, 0.629441, 0.030674, 0.201105, -0.208984, 0.34703], [0.562485, -0.250348, -0.660909, 0.065712, -0.557866, -0.689626, 0.184046, 0.109493, -0.019637, -0.318442, 0.373466, 0.240435, -0.541321, 0.289322, 0.216336, -0.139236, 1.037736, 0.293426, -0.060209, 0.630357, 1.006321, 0.520877, -0.219286, 0.495326, -0.632668, 0.864608, 0.904835, 0.176331, 0.469536, 0.787117, -0.043743, -0.911545, 0.03488, 0.104191, -0.144684, 0.400577, -0.304831, 0.347947, -0.137843, -0.062058, -0.213584, 0.439362, -0.164053, -0.182119, 0.396041, -0.21803, -0.533457, 0.138412], [-0.085458, 0.048062, -0.045818, -0.218189, -0.348476, 0.468961, -0.384911, -0.443435, 0.373792, 0.057265, 0.14142, -0.13898, 0.114732, -0.340146, 0.152483, 0.312492, -0.537923, -0.447801, -0.470549, -0.156416, 0.123855, -0.222473, -0.581796, -0.544505, -0.636073, -0.481727, -0.157626, -0.180058, 0.125823, 0.298201, -0.654614, 0.046744, -0.405785, 0.24901, 0.224062, -0.320375, -0.097994, -0.246921, -0.151807, 0.047068, -0.32192, 0.015803, 0.239305, 0.437912, -0.024437, 0.417517, 0.182891, -0.02598], [-0.488929, 0.085655, 0.572154, -0.171869, 0.405739, 0.261588, 0.115555, 1.047079, -0.07519, 0.006942, -0.284909, 0.22611, 0.319138, -0.144928, -0.331611, -0.538916, -0.727057, 0.017092, -0.26778, -0.114274, 0.060348, -0.649731, 0.170901, -0.095778, -0.263377, -0.691899, -0.040674, 0.416679, 0.451669, -0.484542, -0.541165, 0.442543, 0.254287, -0.105861, -0.103345, -0.100403, 0.094971, 0.132683, 0.376188, -0.212313, 0.43347, 0.185428, 0.042145, -0.10388, -0.734249, -0.132791, 0.792047, 0.07829], [-0.020056, 0.133387, 0.002071, -0.210613, 0.07925, 0.364114, -0.218112, 0.07463, 0.069983, -0.12371, -0.049459, 0.175813, 0.509667, -0.442438, 0.051586, -0.090193, -0.584984, -0.032778, -0.640454, -0.08377, -0.832994, -1.010925, -0.13528, -1.058591, -0.675689, 0.154665, -0.824476, -0.524255, -0.254252, -0.26124, -0.837936, 0.295655, 0.021306, 0.037425, 0.189484, 0.051668, -0.024106, -0.072444, 0.555483, 0.081453, -0.141213, 0.094115, -0.311449, -0.339052, -0.537041, 0.234498, 0.295429, 0.155696]]
SEQ_BH = [0.31439, 0.048089, 0.077669, -0.070009, -0.173946, -0.116953, 0.232432, 0.117389, -0.07741, -0.047059, 0.065571, 0.305816, -0.007325, 0.291444, -0.041407, 0.2149, 1.206614, 0.269264, 0.593014, 0.513197, 0.722453, 0.693122, 0.823584, 0.98796, 0.426247, 0.48462, 0.497113, 0.530344, -0.114522, 0.387261, 0.689682, 0.265845, 0.205069, -0.080531, 0.149605, -0.157557, 0.097195, -0.17864, -0.356301, -0.044028, -0.0983, 0.223859, 0.21846, -0.24316, 0.146621, -0.135102, -0.176791, -0.286821]
SEQ_HEAD = [
    ([[-0.011775, 0.047071, -0.283459, -0.125316, 0.066112, -0.126867, 0.078614, 0.091746, -0.178322, -0.03906, 0.15635, -0.135972, 0.026212, 0.132968, 0.064364, 0.051583, 0.151987, -0.007846, 0.014153, -0.026847, -0.168883, 0.221344, -0.051277, 0.127751, -0.111273, 0.002942, -0.074761, -0.281837, -0.087798, -0.096296, 0.025639, -0.017267, -0.177494, -4e-05, -0.119704, -0.123572, 0.128388, 0.054444, -0.067833, 0.0772, 0.057452, -0.149837, 0.060152, -0.144541, -0.185587, 0.022069, 0.044733, -0.029202, 0.171017, -0.179483, 0.080874, -0.009591, -0.054396, -0.181036, -0.075735, -0.062352, 0.102374, -0.078488, -0.040103, -0.046726, 0.216746, -0.242208, -0.130601, 0.095082], [0.105863, 0.129225, -0.033458, 0.097041, 0.168696, -0.077255, -0.0181, -0.15676, 0.010987, -0.121858, -0.10338, 0.045428, 0.013189, -0.070063, -0.0616, 0.043302, 0.062316, -0.04941, -0.067737, -0.042972, -0.138234, 0.169801, -0.048232, -0.010174, -0.007423, -0.054174, -0.10102, -0.145729, 0.028689, 0.096839, -0.143744, 0.098887, 0.070385, -0.023643, -0.029375, 0.094728, 0.097879, -0.148309, -0.069168, -0.014192, 0.226723, 0.070523, -0.096283, -0.070315, -0.089948, -0.008753, 0.16169, 0.12187, -0.030574, -0.068426, -0.068331, -0.224735, -0.105734, -0.000454, 0.149406, -0.088524, -0.116455, 0.046156, 0.108968, 0.000589, -0.108559, -0.077777, -0.182423, -0.007334], [0.093936, 0.157713, 0.037287, 0.102577, 0.141972, -0.119208, -0.020035, -0.188997, -0.113625, -0.069541, -0.077177, -0.095221, 0.007177, -0.134822, -0.114685, -0.170629, -0.052817, -0.116047, 0.032736, 0.09818, 0.074331, 0.136627, 0.102911, 0.008959, 0.036989, -0.122672, -0.094615, 0.086058, 0.006775, 0.160697, 0.100682, 0.04502, -0.065341, -0.239782, -0.081901, 0.163844, 0.095015, -0.160838, 0.017775, 0.123264, -0.074025, 0.145354, 0.113957, -0.086557, 0.100266, -0.117554, 0.066023, 0.131443, 0.023993, 0.081667, 0.046886, -0.154432, -0.198776, 0.016426, 0.089, -0.125675, 0.058639, -0.13478, 0.1301, -0.049496, 0.17277, 0.013765, -0.081427, 0.103405], [0.180042, -0.062372, 0.039178, -0.04092, -0.084662, -0.086432, 0.062067, -0.023477, 0.048606, 0.168262, -0.127563, 0.077423, -0.166173, -0.291331, 0.086508, -0.080444, -0.034315, -0.243837, -0.06283, 0.070372, 0.076054, -0.242023, 0.001119, -0.04965, 0.155275, 0.152996, -0.134256, -0.030787, 0.236549, -0.073081, 0.082015, -0.162181, 0.152159, -0.212674, 0.051031, 0.038244, 0.155424, -0.136944, 0.05479, -0.091541, -0.204205, 0.080519, 0.11486, -0.288185, 0.069888, 0.155449, -0.117065, -0.093323, 0.067071, -0.071415, 0.175201, -0.055082, -0.224163, 0.213175, 0.109916, 0.022937, -0.029375, 0.11375, 0.150007, -0.084992, 0.025709, 0.320675, -0.077151, 0.153477], [0.060428, 0.119869, 0.243305, 0.249564, -0.20235, 0.038208, 0.15187, -0.170555, 0.160152, -0.269553, -0.21994, -0.149012, -0.014494, 0.338362, -0.081503, -0.17585, -0.131284, 0.257116, 0.103684, 0.076806, 0.107119, -0.174109, 0.042032, -0.169466, 0.196921, -0.139101, -0.048379, 0.155267, 0.319214, -0.373294, 0.173133, 0.234736, 0.412981, -0.034064, 0.104705, 0.127146, -0.092509, 0.126195, 0.078421, -0.013784, -0.258689, -0.08904, 0.170608, -0.330534, -0.027914, 0.042522, 0.064336, -0.05734, -0.234776, 0.12681, 0.158823, 0.159651, 0.034736, 0.228717, -0.242092, -0.245148, 0.111675, 0.258061, 0.301346, -0.294031, 0.258526, -0.219295, 0.05518, 0.254042], [0.098075, -0.033693, 0.023119, 0.124376, -0.010871, 0.136862, 0.0252, 0.088443, -0.062169, -0.018101, 0.041969, 0.115285, -0.057503, 0.089502, -0.116954, 0.114855, 0.139736, -0.19119, 0.012938, -0.142223, -0.014797, -0.013934, 0.193138, 0.091478, -0.198023, 0.107029, 0.209604, 0.156587, -0.09587, 0.046875, -0.043904, 0.116373, 0.180642, -0.039599, -0.019668, 0.172349, 0.204181, -0.298682, 0.100437, -0.047652, -0.016468, 0.182992, -0.127607, -0.143074, -0.151933, -0.026837, 0.07371, 0.074914, 0.168717, -0.071898, 0.086715, -0.182179, -0.077807, -0.12343, -0.041907, -0.111832, -0.146742, 0.014051, -0.195492, 0.181358, 0.054394, 0.253111, 0.013702, -0.025418], [0.058589, -0.021731, -0.292216, -0.010941, -0.042292, -0.184537, 0.20537, 0.119079, 0.169218, 0.080731, -0.19113, 0.173963, -0.250613, 0.212661, -0.245534, -0.165684, 0.203065, -0.0306, -0.045459, 0.068711, 0.007258, 0.12853, -0.003148, 0.008784, -0.114744, 0.078418, -0.170051, -0.132735, 0.275451, 0.01422, 0.098433, -0.090443, 0.165161, 0.226805, -0.199086, 0.150933, -0.162731, 0.040155, 0.212435, -0.048189, 0.029781, 0.076449, 0.14065, -0.123151, 0.093019, -0.22443, 0.038507, 0.142554, -0.037805, -0.209646, -0.112571, -0.025504, -0.057105, 0.219743, -0.142366, -0.08151, -0.066894, -0.019926, 0.052573, -0.303775, -0.007127, 0.054576, -0.063516, -0.018836], [-0.097628, -0.108922, -0.013616, -0.035858, -0.070258, 0.037263, -0.128556, -0.09908, -0.067572, 0.047801, 0.082581, 0.151498, -0.001059, 0.274917, -0.025496, -0.042121, 0.072933, 0.023732, 0.128589, 0.221444, 0.043941, -0.086116, -0.210281, -0.238389, -0.056904, 0.163566, 0.077441, 0.057552, 0.067212, 0.063914, -0.319335, -0.280714, 0.008558, 0.161218, -0.133429, -0.183015, 0.00144, 0.005186, -0.121636, -0.108839, 0.010566, -0.092227, 0.108481, 0.086024, 0.109642, 0.027361, 0.075514, 0.012271, 0.040931, 0.171987, 0.074052, -0.061546, -0.031082, 0.076565, 0.233105, -0.052922, 0.079765, 0.147621, 0.089437, 0.102785, 0.058648, 0.052138, -0.142344, -0.022716], [0.159927, 0.093762, 0.106389, 0.240299, 0.012626, -0.321392, -0.002947, -0.176405, 0.135537, -0.079377, 0.027043, -0.334999, -0.166647, 0.167101, -0.154256, 0.143569, -0.180067, 0.082654, -0.083773, -0.03883, 0.026735, -0.005347, -0.032874, -0.149243, -0.007124, -0.028519, -0.210639, -0.193422, 0.180884, -0.047625, -0.083909, -0.025926, -0.232402, -0.257199, -0.106299, 0.089195, -0.025687, 0.043299, -0.038436, -0.157371, -0.01447, 0.18654, 0.271379, -0.074781, -0.225303, 0.013097, -0.107886, 0.204665, 0.141003, -0.092142, -0.266321, -0.198656, -0.028013, 0.338275, -0.094533, 0.039015, -0.022766, 0.055482, 0.14725, 0.068375, 0.101264, 0.016342, -0.10759, 0.236699], [-0.065694, -0.019316, -0.029078, -0.030908, -0.083255, -0.117377, 0.111918, 0.213735, 0.141154, 0.018347, 0.135499, -0.060938, 0.173297, 0.047273, 0.044087, 0.164635, -0.122912, 0.087489, -0.080359, 0.109951, -0.227094, -0.060282, 0.038221, -0.041161, -0.002006, -0.108195, -0.127639, -0.035124, -0.04873, 0.002834, 0.056534, 0.001853, -0.12408, 0.055236, 0.042117, -0.178851, -0.204816, 0.017545, 0.04876, 0.054125, -0.10875, 0.152989, -0.136619, -0.155939, -0.053385, 0.102544, 0.295298, -0.199683, -0.176085, -0.373677, -0.262768, -0.004747, 0.066241, -0.104541, 0.247818, -0.046456, 0.12, 0.01731, 0.217134, 0.030452, -0.037247, -0.059827, -0.317236, -0.028673], [-0.035597, -0.012083, 0.034327, -0.064291, -0.102056, 0.108425, -0.092394, -0.041107, -0.15327, 0.061659, 0.225939, -0.133912, -0.013631, 0.134702, 0.016386, -0.006203, -0.029132, -0.113303, 0.043213, -0.168966, 0.168615, -0.12323, -0.058927, 0.142297, 0.077281, 0.102336, -0.087734, 0.022403, -0.078152, 0.104017, 0.023168, 0.100503, 0.019678, -0.054998, 0.046157, -0.001323, 0.067052, 0.08828, 0.025829, -0.150276, -0.065306, 1.6e-05, 0.005115, -0.078818, 0.098718, -0.222563, -0.053495, 0.207568, -0.094982, 0.02608, -0.085086, 0.050914, -0.320085, 0.089971, 0.198367, 0.119403, -0.070101, -0.0351, 0.075211, -0.050123, -0.068692, 0.086893, -0.163817, -0.01961], [0.10092, 0.099167, 0.036056, 0.035979, -0.016025, -0.360558, 0.020252, -0.282279, 0.320785, -0.017005, -0.145012, -0.338368, -0.121459, -0.052703, 0.025345, -0.266128, -0.019711, 0.157742, -0.069004, 0.095118, -0.116684, 0.103332, 0.090308, -0.175945, 0.05156, -0.179151, -0.314128, -0.097775, -0.181946, -0.023845, 0.026585, 0.051885, 0.019845, 0.002028, -0.102411, -0.054018, -0.306554, -0.184332, 0.092179, -0.20252, -0.28828, 0.197106, 0.229055, 0.003802, -0.214786, 0.29277, 0.274165, 0.471096, -0.016507, 0.037128, -0.034892, -0.253436, 0.271539, 0.095049, -0.068147, -0.010976, -0.150955, 0.123752, 0.161291, 0.256379, 0.278003, -0.044692, -0.128496, 0.097708], [-0.150253, 0.072746, -0.066538, -0.075853, 0.088382, -0.174951, 0.015609, -0.166144, -0.138055, -0.034696, 0.076932, 0.070617, -0.061535, 0.002081, -0.089518, 0.164701, 0.07825, 0.072914, -0.022187, 0.002637, -0.163079, 0.013721, 0.197209, -0.042682, -0.088337, -0.09685, 0.003775, -0.048938, -0.066847, 0.010632, -0.011668, -0.142889, 0.020966, 0.070863, -0.21768, 0.03573, 0.072454, 0.076162, -0.094403, 0.007042, 0.085093, -0.075514, 0.200197, 0.046352, -0.020253, -0.231409, -0.107672, 0.04433, -0.018527, -0.081636, -0.180694, 0.044425, -0.096593, 0.091782, 0.135633, 0.112281, 0.126206, -0.059777, 0.096799, -0.154547, 0.083551, -0.091359, -0.155585, -0.075421], [-0.039448, 0.048977, -0.073319, -0.058687, -0.05533, -0.157057, -0.198734, -0.15264, 0.09445, -0.089935, 0.020915, 0.10029, 0.063014, 0.024899, -0.089122, -0.030626, 0.076531, 0.16267, -0.230436, 0.102288, -0.19889, -0.085353, 0.073759, 0.096399, 0.098703, -0.165732, 0.170164, -0.083664, -0.096638, -0.055646, -0.080609, -0.061367, 0.260222, 0.071641, 0.019629, 0.142825, -0.123934, -0.039433, -0.035895, 0.149374, 0.210235, -0.11375, -0.106578, -0.014833, 0.081917, 0.055232, -0.068027, -0.093242, -0.030737, 0.024796, 0.117221, 0.049207, -0.002421, 0.226741, -0.043488, 0.114369, 0.21284, 0.043149, 0.003505, -0.138582, -0.013766, 0.125068, 0.042202, 0.039245], [0.053823, 0.054641, 0.118594, 0.051077, -0.098136, 0.017603, -0.123386, 0.004649, -0.094022, -0.037194, -0.020224, 4.2e-05, 0.12369, -0.093754, 0.105395, -0.15656, 0.045314, -0.091814, 0.087935, 0.032987, -0.103528, 0.045937, 0.109639, 0.217261, 0.100315, -0.127611, -0.05387, 0.039045, -0.124228, 0.072071, -0.135913, -0.002268, -0.424541, -0.120095, -0.143912, -0.0304, 0.015167, 0.228054, -0.012586, -0.00629, -0.093554, 0.257371, -0.286919, 0.007927, -0.272588, 0.029185, 0.00205, 0.048004, -0.075788, 0.171356, -0.050063, 0.023498, -0.061743, 0.052862, 0.213627, 0.047129, 0.04712, -0.284947, -0.005812, 0.018504, -0.168014, 0.040881, 0.041427, -0.05987], [-0.12612, 0.044739, 0.091055, 0.047615, 0.069659, -0.165645, 0.264411, 0.075086, -0.00942, 0.15904, -0.075863, -0.09339, 0.081968, 0.142008, 0.024293, 0.064934, -0.037657, 0.060918, 0.013265, -0.02486, 0.058707, 0.056302, -0.056351, 0.231101, -0.112245, -0.016431, 0.273014, 0.025056, -0.046183, 0.10592, -0.094414, 0.068175, -0.014265, 0.159129, -0.161154, -0.01946, -0.133188, 0.025931, 0.077478, -0.02735, -0.012582, 0.138363, -0.025353, -0.064538, -0.061443, -0.065293, -0.076145, 0.050528, 0.092653, -0.245472, 0.016229, 0.014928, -0.062466, -0.086175, 0.023857, 0.006374, -0.068271, -0.249862, 0.065096, -0.126626, -0.037723, -0.121758, 0.029806, 0.016749], [0.366241, 0.094643, 0.247937, -0.036072, 0.048634, -0.465559, 0.089114, 0.190502, 0.143573, -0.342718, -0.064031, 0.020656, -0.068529, -0.104274, -0.114126, -0.437402, -0.04693, 0.091958, -0.001624, 0.080989, 0.070027, 0.121072, -0.275996, 0.268656, -0.123906, -0.091389, -0.071637, -0.244536, -0.166179, -0.052533, -0.043932, 0.177055, -0.428192, 0.086541, 0.019336, -0.403957, -0.154011, 0.29595, -0.34284, 0.061885, -0.098982, -0.355151, -0.241448, -0.056347, 0.090445, 0.049575, 0.036206, -0.072994, -0.263012, -0.15434, -0.030322, -0.252359, 0.030366, 0.173305, -0.093845, 0.001304, 0.210757, 0.019076, 0.202175, 0.042084, 0.272033, -0.013041, 0.197205, 0.17635], [0.225711, 0.067588, 0.009435, -0.317018, -0.132731, -0.308724, 0.008496, 0.104465, -0.040072, -0.251613, 0.102853, -0.025022, 0.098402, -0.072217, 0.056411, 0.19992, -0.11187, -0.031058, -0.103118, -0.076282, 0.381931, -0.291412, 0.071483, -0.017649, 0.058147, 0.330111, -0.114541, -0.19349, 0.151904, 0.046247, -0.039905, -0.088179, 0.048206, -0.159772, 0.071361, -0.127519, -0.067713, 0.085156, -0.205425, 0.115781, -0.171412, 0.1259, 0.115225, -0.087561, 0.166124, 0.064043, -0.0547, 0.122733, -0.01222, 0.230434, -0.215345, -0.000184, -0.102817, 0.083184, -0.081254, -0.123811, -0.089555, 0.005981, -0.126033, 0.009044, 0.052189, 0.0944, -0.121353, 0.10169], [0.058351, -0.030146, 0.050688, 0.030401, -0.055881, 0.111291, 0.155346, -0.002191, 0.026569, 0.000143, -0.177899, -0.025712, -0.066999, 0.097674, 0.107754, -0.051729, 0.097235, -0.073504, -0.01954, 0.065668, 0.057926, 0.034541, -0.073322, -0.035838, -0.043843, 0.063339, 0.106707, -0.089548, 0.070931, 0.073665, 0.164822, 0.081002, 0.125917, 0.286757, -0.067745, 0.031693, 0.047967, -0.103442, -0.05669, 0.076639, 0.020091, -0.046173, -0.091609, 0.037817, 0.014734, -0.014893, -0.132934, 0.00737, -0.042293, -0.243723, -0.090318, 0.074436, -0.090161, -0.084567, -0.040806, -0.01281, 0.069476, 0.136885, 0.100879, -0.276588, -0.078359, -0.191947, 0.191673, 0.184483], [0.194306, 0.187417, -0.204405, -0.101846, 0.092997, -0.102494, 0.02829, 0.130224, 0.151057, 0.035459, -0.250212, -0.138103, -0.329723, 0.188576, -0.265168, 0.029361, -0.211188, 0.079423, 0.059351, -0.051876, 0.061358, 0.076894, -0.029191, -0.491002, 0.367397, -0.11407, 0.184446, -0.16797, 0.242267, -0.184683, -0.018347, 0.018787, 0.26235, 0.093736, -0.224504, -0.11997, 0.142034, 0.09407, -0.142845, -0.294497, 0.107835, -0.427579, 0.084712, -0.107538, 0.380171, -0.180106, 0.186522, -0.008834, -0.179968, 0.115181, 0.007453, 0.087309, 0.113535, -0.003214, 0.087677, 0.120055, -0.227756, -0.03412, 0.153254, -0.21233, 0.112885, 0.072495, -0.03193, 0.124962], [0.049298, -0.199362, 0.154309, 0.036845, -0.259378, -0.017926, 0.057878, -0.017562, 0.133649, 0.121333, -0.253212, -0.133976, 0.105518, 0.035962, -0.125429, -0.014349, 0.137217, 0.178873, -0.333008, 0.137338, -0.183067, -0.027646, -0.00351, -0.014019, 0.047537, -0.14878, 0.222484, 0.243484, -0.281245, 0.171336, -0.225437, 0.091625, -0.254875, -0.117111, -0.047527, 0.012066, 0.019579, 0.049147, 0.134291, 0.08285, -0.076633, 0.249117, -0.123584, 0.221823, -0.095577, 0.13264, -0.0328, 0.029453, 0.182315, -0.042138, -0.065895, -0.03201, -0.05535, -0.009151, -0.000318, 0.043257, -0.230488, 0.095502, -0.027835, 0.147428, -0.165958, -0.025239, 0.00111, -0.061257], [0.086617, -0.200103, 0.159003, 0.012387, 0.076647, 0.309757, -0.110569, 0.033539, -0.195414, 0.088414, -0.11727, 0.178962, -0.060048, 0.113577, 0.105399, 0.092889, 0.146522, -0.156617, -0.060482, -0.130484, -0.004244, 0.074539, -0.132698, -0.116334, -0.190061, -0.091779, 0.459444, 0.084273, 0.063341, 0.20354, 0.131294, -0.120898, -0.144064, -0.031795, 0.093192, -0.468552, 0.062237, 0.003394, 0.499593, -0.14422, 0.139752, 0.172891, 0.304347, 0.15172, 0.137536, 0.253167, 0.017018, -0.027678, 0.184115, -0.045525, 0.110247, 0.034121, 0.042524, -0.064435, 0.026032, -0.179342, -0.031391, -0.020022, 0.066354, 0.093873, -0.046753, -0.015015, 0.089455, 0.064381], [0.00507, 0.087406, 0.002325, -0.21962, 0.219511, -0.096948, 0.144959, 0.175157, 0.253441, 0.35013, -0.032074, 0.048118, 0.024672, 0.126203, -0.084711, -0.031518, 0.253698, 0.326775, -0.041962, -0.142292, -0.023089, -0.017277, 0.10039, 0.263046, -0.301523, -0.131851, -0.476223, -0.001835, 0.096079, 0.041426, 0.148347, 0.245087, 0.174546, 0.242015, 0.039388, 0.185373, 0.381983, 0.155333, -0.209079, -0.042413, -0.129343, 0.17657, 0.096331, -0.01268, 0.131429, -0.051106, 0.080911, 0.079333, 0.145395, 0.070139, 0.015217, -0.294307, -0.260714, -0.059354, 0.046974, -0.458726, -0.021023, 0.099749, 0.174807, 0.130562, 0.140514, 0.220453, 0.027008, 0.227561], [-0.106008, 0.129322, -0.191481, -0.061351, -0.174321, -0.196313, -0.156914, 0.031752, -0.216945, -0.010071, -0.162367, 0.069848, 0.069688, 0.156467, 0.055089, 0.136746, -0.207413, 0.074245, -0.150806, 0.12691, -0.116908, -0.472067, 0.130297, -0.169987, 0.058128, -0.125842, 0.000386, -0.276505, -0.177355, 0.003159, -0.254072, -0.065619, 0.043499, -0.215734, -0.136052, 0.099748, 0.112616, -0.218644, 0.025729, 0.178734, 0.041032, -0.202017, -0.133972, -0.002031, -0.346103, 0.061539, 0.023526, -0.084664, -0.292963, 0.077998, 0.135836, -0.164635, -0.078625, -0.039366, -0.011664, -0.189857, 0.164531, -0.137184, -0.065441, -0.113296, -0.182013, -0.140756, -0.117009, -0.058619], [-0.097761, -0.250315, 0.283577, 0.153827, 0.019208, 0.023479, -0.051096, 0.150143, -0.060815, -0.073661, 0.079976, -0.231639, 0.188608, -0.041634, 0.138567, 0.194365, 0.252225, 0.18212, -0.129133, -0.068858, 0.028875, -0.212158, 0.193487, 0.056098, -0.238557, -0.173983, 0.033078, 0.219512, -0.124819, -0.058514, -0.143624, -0.221136, -0.351979, -0.148644, 0.008968, 0.251306, 0.096528, 0.066928, -0.047959, 0.05987, 0.210454, 0.507459, 0.193615, 0.161288, 0.042225, 0.059688, 0.034602, -0.074298, 0.088827, 0.261308, 0.095152, 0.135337, -0.092503, 0.373227, 0.112423, 0.400395, -0.170464, -0.126572, -0.091245, -0.111582, 0.206007, -0.188105, -0.040431, 0.193392], [-0.062366, -0.115849, 0.062177, 0.087412, 0.148698, -0.02321, -0.451394, 0.21021, -0.156204, 0.437591, 0.126572, 0.196569, 0.142605, -0.169236, -0.083208, 0.313966, 0.306653, -0.463942, -0.128087, 0.136974, 0.273704, -0.370295, 0.163219, -0.013258, 0.274874, 0.285639, -0.027072, -0.142284, 0.32045, 0.139118, -0.284635, -0.317242, 0.358652, -0.030416, 0.267259, 0.034617, 0.303845, -0.227659, -0.091989, 0.004384, -0.156681, 0.100046, 0.004323, -0.082022, -0.224144, 0.137253, -0.265655, 0.008914, 0.303462, -0.226703, -0.077407, -0.191775, -0.386163, -0.3429, 0.22486, -0.254533, -0.250663, 0.00017, -0.107254, 0.073734, -0.2495, 0.161089, 0.163381, -0.281602], [-0.117083, -0.223306, 0.084881, 0.170963, -0.213294, -0.146347, -0.254408, -0.019236, 0.000903, 0.126427, -0.029581, 0.103735, -0.132549, -0.007505, -0.189873, -0.007079, 0.189567, -0.236662, 0.174884, 0.046683, -0.033012, -0.659467, -0.069673, 0.061267, -0.059998, -0.074768, 0.027047, -0.054147, 0.047178, -0.023022, -0.091817, 0.074323, 0.051389, -0.133319, -0.098573, -0.050454, 0.105045, -0.236207, 0.104487, -0.079441, -0.312369, 0.168121, -0.097281, -0.085277, -0.336422, 0.153454, -0.205876, -0.314095, -0.046124, -0.242168, -0.129711, -0.252704, -0.280452, -0.273848, 0.201704, -0.178679, 0.082468, -0.075615, -0.180048, 0.04003, -0.146767, 0.226144, -0.237516, -0.067227], [0.162509, 0.141658, -0.211927, -0.235422, -0.015957, -0.046868, 0.226158, -0.015289, 0.169119, -0.117918, -0.036443, 0.113274, -0.044966, 0.06814, -0.236348, 0.050488, 0.184711, 0.067667, 0.192593, -0.208355, -0.122686, -0.036927, -0.176446, -0.021412, -0.223587, -0.052439, -0.121743, -0.002891, -0.069465, -0.043347, -0.027003, -0.040102, -0.155828, 0.009614, 0.136064, -0.16863, -0.268006, 0.150963, 0.003694, 0.18984, 0.117025, 0.035815, -0.061216, -0.083102, 0.146696, 0.130334, 0.170069, 0.193347, -0.064666, 0.189893, 0.093676, 0.077882, -0.054041, 0.202053, 0.039407, -0.085031, -0.173798, 0.105742, 0.085662, -0.005961, -0.020907, -0.01466, 0.153218, 0.106144], [0.2644, -0.128318, 0.28834, 0.144736, 0.256341, 0.279877, -0.128073, -0.093277, 0.078511, 0.188729, 0.072739, -0.031007, 0.112595, 0.235731, 0.084666, 0.234393, 0.297708, 0.112277, -0.152489, -0.112134, 0.046596, 0.19657, 0.126525, 0.141809, 0.278394, 0.03315, -0.215973, 0.215245, 0.155485, 0.296128, -0.032302, 0.094618, 0.174219, 0.055369, 0.067223, -0.000467, 0.227277, 0.379552, -0.081695, 0.264188, 0.074414, 0.35028, -0.035717, 0.211555, 0.087909, 0.080362, 0.186905, 0.017091, 0.236052, -0.015628, 0.015291, -0.008934, -0.006316, 0.042408, -0.004578, -0.060053, 0.03478, -0.017365, 0.046764, -0.051573, 0.261316, -0.070303, 0.036246, 0.241978], [0.365836, 0.077601, 0.234746, 0.179578, 0.357427, 0.229633, 0.080722, 0.44477, 0.23891, 0.1665, 0.059091, 0.069617, 0.260539, -0.160505, 0.336598, 0.104792, 0.226383, -0.236586, -0.144212, -0.106893, 0.436696, -0.096909, 0.039502, -0.084417, -0.007147, -0.146757, -0.174656, 0.302356, -0.043817, 0.317307, -0.195138, 0.041399, -0.065041, -0.261409, 0.033596, 0.286383, 0.383086, -0.105541, -0.056812, -0.015205, 0.151594, 0.48572, 0.210503, 0.291722, 0.160408, 0.014649, -0.059797, 0.390397, 0.14056, 0.101867, 0.178069, 0.137472, -0.052463, 0.174478, -0.161297, 0.364685, -0.35283, 0.077402, -0.147378, 0.088135, 0.201037, -0.225926, 0.261267, 0.363372], [-0.074511, 0.245741, 0.242969, -0.086633, -0.023565, -0.098296, 0.1815, -0.003346, -0.05424, 0.118169, -0.067948, 0.519828, 0.030266, 0.180384, 0.193507, 0.084686, -0.025349, 0.037327, 0.341447, 0.072257, -0.138506, 0.006434, 0.342447, -0.078971, -0.322847, -0.079986, -0.054186, 0.394396, 0.027145, 0.054436, 0.18748, 0.113101, 0.221357, -0.055485, 0.100879, 0.127274, -0.313086, 0.59795, -0.237334, 0.20959, 0.263095, 0.060496, -0.050444, 0.094727, 0.08641, 0.098494, -0.075603, 0.166952, 0.108181, 0.327479, 0.137742, 0.091004, 0.031469, 0.174603, -0.003446, -0.095366, 0.053378, 0.006214, 0.046557, 0.237157, 0.248902, 0.099443, 0.06466, 0.183694], [-0.106541, 0.212744, -0.108702, -0.106984, -0.132764, 0.006329, 0.456003, -0.271044, 0.129772, 0.08066, 0.147941, 0.205803, -0.179892, 0.132548, -0.003899, -0.365044, -0.285264, 0.300996, 0.095645, 0.135198, -0.429572, 0.02309, -0.130285, 0.091167, -0.270692, -0.102041, 0.099022, -0.249101, 0.084028, -0.297642, 0.292594, 0.275896, -0.068496, -0.093059, 0.044815, -0.133859, -0.255246, 0.404687, -0.189089, 0.029338, -0.049765, -0.344321, 0.031062, -0.099134, 0.277237, -0.297286, -0.053905, 0.101992, -0.155757, 0.279103, 0.073364, 0.082952, 0.062271, 0.089876, -0.245033, 0.08226, -0.122864, 0.12579, 0.016714, 0.002425, 0.108516, -0.520488, -0.006892, 0.218228], [-0.016377, 0.043392, -0.054313, -0.302184, -0.801447, -0.012152, 0.055966, -0.002284, 0.040915, -0.277678, 0.029719, -0.113037, -0.012661, 0.044619, -0.115865, -0.216089, -0.172634, 0.048127, 0.177942, -0.187848, -0.107588, -0.022514, -0.065736, 0.114854, -0.020053, -0.282855, -0.10932, -0.132229, -0.068869, -0.108765, 0.17741, -0.099743, -0.252194, -0.202087, -0.156491, -0.135251, -0.10277, 0.006154, 0.044836, -0.025219, -0.563817, -0.001137, -0.242737, -0.234763, 0.054981, -0.187271, 0.063997, -0.12051, -0.157072, 0.16229, 0.239976, -0.346933, -0.093403, -0.062027, 0.029427, 0.051601, -0.177465, 0.005401, -0.06968, 0.120057, -0.177546, -0.123307, -0.123943, 0.122447], [0.052726, -0.14775, 0.051812, 0.027844, -0.044982, -0.081289, -0.188527, -0.123999, -0.045371, -0.299195, 0.016958, -0.247235, 0.006218, -0.188442, -0.138357, -0.319954, -0.330105, 0.06956, -0.147813, 0.043054, -0.128624, 0.079301, -0.052807, -0.098945, 0.340571, -0.12559, 0.042998, 0.067251, -0.026661, -0.173838, -0.07734, -0.035976, 0.323907, -0.007457, -0.116069, -0.100053, -0.026642, -0.25591, 0.113363, 0.090062, 0.036813, -0.254656, -0.058709, 0.031562, -0.071305, 0.115296, -0.019308, -0.009347, -0.197356, -0.106152, -0.004949, 0.156435, 0.00603, 0.115469, -0.086837, -0.035091, 0.240648, 0.103602, -0.027565, -0.214552, -0.063892, -0.075512, -0.00385, -0.049788], [-0.400176, 0.162929, -0.066, -0.405415, -0.003705, -0.054541, 0.223383, 0.102719, -0.046786, 0.168481, 0.069035, 0.139676, -0.094403, 0.002543, 0.124664, -0.118766, -0.027939, -0.084318, -0.202029, -0.182406, -0.272874, -0.465167, 0.056308, 0.069515, -0.288502, 0.005712, -0.260291, -0.046399, 0.177341, -0.111482, 0.028128, -0.034977, 0.110363, 0.058399, 0.12922, 0.1586, -0.214526, -0.017364, -0.272718, -0.016253, 0.162219, -0.161752, 0.109046, -0.003021, -0.130809, 0.026826, -0.011255, -0.074217, 0.09389, -0.257165, -0.084972, -0.026149, -0.116739, 0.004428, -0.239991, 0.280321, -0.005777, 0.110577, -0.100719, -0.075477, 0.091946, -0.247644, -0.077948, 0.326521], [0.168423, 0.047353, -0.070875, 0.154243, -0.079367, -0.076315, -0.141706, -0.166892, 0.081208, -0.181049, 0.19314, -0.291214, -0.08382, -0.261241, -0.031863, -0.051117, -0.205931, -0.290136, -0.159676, 0.052382, -0.10871, -0.065051, -0.029418, -0.053844, 0.319138, -0.164917, 0.12895, -0.156397, -0.100395, 0.038116, -0.04546, -0.197708, 0.095774, -0.13354, -0.295534, -0.238752, -0.138209, 0.01107, 0.013301, 0.125638, 0.154013, 0.013614, 0.015779, 0.059174, 0.059621, -0.089929, -0.075402, 0.0843, -0.107703, -0.006266, 0.056359, 0.067935, 0.056204, 0.098611, -0.089509, 0.211509, -0.007806, -0.141131, 0.145973, -0.230341, 0.095902, 0.014848, 0.012933, -0.154833], [-0.034184, -0.016745, 0.259744, -0.19294, 0.292715, 0.096727, 0.129452, -0.29, 0.398407, 0.002848, -0.180762, -0.197366, -0.134817, 0.10738, 0.290044, -0.080805, 0.164899, 0.540789, 0.299064, 0.432693, -0.156288, 0.685272, -0.150682, 0.004876, 0.028295, -0.04874, -0.107794, 0.194803, -0.070811, 0.155492, 0.472556, 0.155242, 0.336586, 0.36659, -0.197431, -0.220265, 0.241781, 0.529185, -0.05705, -0.214448, 0.034608, -0.048304, -0.401595, 0.13211, -0.039738, 0.189312, -0.060396, -0.217271, 0.39374, 0.337871, 0.138826, -0.328277, 0.64526, -0.184815, -0.020362, -0.609309, 0.254833, 0.03869, 0.338538, -0.030476, -0.0292, 0.224099, -0.199785, 0.037407], [0.299163, 0.098063, 0.045572, 0.333655, 0.301902, -0.042808, -0.000768, 0.035225, 0.000168, -0.101909, -0.00267, 0.082561, -0.018138, -0.257027, 0.111128, -0.08358, 0.058829, -0.024187, 0.08027, 0.135917, 0.072236, 0.187042, 0.246226, -0.033787, 0.221306, -0.153818, 0.115482, 0.026822, 0.063455, 0.010629, 0.039632, -0.14955, 0.099396, 0.200296, -0.035658, -0.072223, 0.151866, -0.14624, 0.112343, 0.121626, 0.18016, 0.232886, 0.058275, -0.029112, 0.117687, -0.005518, -0.032461, 0.06485, 0.007704, -0.012596, 0.083894, 0.111898, -0.032799, -0.050477, -0.03484, -0.023802, 0.035488, 0.109847, 0.007511, -0.000585, 0.05154, -0.014998, 0.148333, -0.113172]], [0.0008, -0.094952, -0.064313, -0.183473, -0.244742, -0.205422, 0.082559, -0.067087, -0.120453, 0.206936, 0.011613, 0.162517, -0.153333, 0.082275, -0.079894, -0.066692, 0.088487, -0.013914, 0.05745, -0.154847, -0.16584, -0.356475, -0.117333, 0.002769, -0.239558, -0.160248, -0.01406, -0.106991, 0.180372, -0.07798, -0.036215, 0.143823, -0.129337, -0.19346, 0.160924, -0.077156, 0.003187, 0.047668, -0.175551, -0.176595, -0.113494, -0.073273, 0.138081, -0.055356, -0.085567, -0.015161, -0.026044, -0.231275, -0.034448, -0.069096, 0.182795, -0.355294, -0.286092, -0.163137, -0.022696, 0.191511, -0.226723, 0.040873, -0.06336, 0.219282, -0.212757, -0.136166, -0.015794, -0.104089]),
    ([[0.116224, 0.053465, -0.053851, 0.152129, 0.047707, -0.131567, 0.251075, -0.080454, 0.015676, 0.047021, 0.083158, 0.104382, -0.059455, -0.048702, 0.113724, -0.118168, 0.032252, 0.129694, 0.095603, 0.27076, -0.034018, -0.115348, 0.133696, 0.093937, 0.433847, 0.09875, 0.066685, 0.348153, 0.02625, 0.103396, -0.01567, 0.046897], [-0.027205, 0.11985, -0.019087, -0.064133, -0.161612, -0.04731, -0.069601, 0.161188, -0.323621, 0.015838, 0.079389, 0.141948, 0.013987, -0.132361, -0.046046, 0.096539, 0.009948, -0.269704, 0.092488, -0.021608, -0.127131, 0.109572, -0.150957, -0.144738, 0.000847, -0.045765, 0.164807, -0.092539, -0.041247, -0.204213, -0.049951, -0.085084], [0.030103, -0.048602, 0.144852, -0.027706, -0.001737, -0.25446, 0.151756, 0.13854, 0.067742, 0.018004, 0.075902, 0.06497, 0.007621, -0.102513, -0.052526, 0.038337, 0.189282, 0.076858, 0.171483, 0.018509, -0.118649, 0.088721, 0.162622, 0.161585, -0.035024, -0.013193, -0.060398, -0.59386, 0.130152, 0.147069, 0.150759, -0.122048], [-0.001876, 0.180587, 0.111632, 0.064377, -0.032609, 0.133232, -0.14473, 0.128426, -0.282993, 0.138296, 0.173182, -0.053609, -0.026061, -0.188836, -0.005415, 0.108708, -0.224828, 0.100299, 0.006067, -0.007063, -0.752319, 0.139325, 0.047827, -0.038231, -0.387751, -0.030082, 0.062271, 0.047509, -0.055236, 0.01226, -0.05542, 0.02425], [-0.197763, 0.240268, -0.073432, 0.072281, -0.150726, -0.244878, 0.253221, -0.205839, 0.264111, -0.02561, 0.263053, 0.134871, -0.025974, -0.1412, -0.123589, -0.059131, 0.084634, 0.176021, -0.010292, 0.21972, -0.135497, -0.06027, 0.21694, 0.102098, 0.158949, -0.086514, -0.138973, 0.532862, 0.238979, 0.33105, 0.232252, -0.161031], [0.009539, 0.127027, -0.075895, 0.047287, 0.162716, -0.420358, 0.105487, 0.123543, -0.03305, 0.203589, 0.120628, -0.031376, 0.127493, -0.074944, 0.119374, 0.10766, 0.080561, -0.106781, 0.03098, -0.073676, 0.00896, 0.118295, -0.040868, 0.239421, 0.239409, 0.236145, -0.021433, -0.065064, -0.104044, 0.158194, 0.141455, -0.245935], [0.012571, -0.166722, 0.043291, 0.067107, 0.055856, 0.043664, 0.031289, 0.103346, -0.275129, -0.055597, -0.181722, -0.080195, -0.008172, -0.117347, 0.032976, 0.110174, 0.010144, -0.087485, 0.003863, -0.213871, 0.098768, -0.066163, -0.124782, 0.080968, -0.214395, 0.128224, 0.035961, -0.232613, 0.051469, 0.032642, -0.139552, 0.153381], [-0.033449, 0.124238, -0.022201, -0.056285, -0.05742, -0.12467, 0.185696, 0.127458, 0.152143, 0.023949, 0.08938, 0.144561, 0.091594, -0.057816, -0.137608, 0.247227, 0.113788, 0.021131, 0.103639, 0.088178, -0.118584, 0.079446, 0.120813, 0.191379, 0.162887, 0.196164, 0.059501, 0.211947, 0.124046, 0.142723, 0.145551, -0.181399], [-0.05766, -0.015554, 0.054058, 0.118369, 0.019585, 0.116935, 0.069383, -0.047887, -0.137622, 0.18888, 0.035958, 0.017363, 0.025826, -0.212598, 0.103406, 0.006078, 0.043048, 0.077306, 0.009708, -0.026887, -0.088519, 0.014238, 0.004193, 0.049278, 0.20768, 0.115997, 0.17567, 0.272257, -0.158078, -0.090478, 0.095568, -0.064832], [0.031901, 0.005118, -0.253469, -0.076738, -0.120924, 0.341746, 0.179151, -0.036009, 0.099252, -0.176562, 0.127804, 0.168267, -0.041749, 0.024055, -0.165724, 0.146382, 0.22854, 0.215469, 0.08855, 0.087784, 0.105191, -0.193616, 0.135995, -0.169731, 0.117171, -0.188191, -0.15356, 0.122309, 0.108725, 0.060387, 0.006084, -0.309071], [-0.011897, 0.137414, 0.17263, -0.031428, 0.066578, -0.308588, -0.02639, -0.057743, 0.130986, -0.023593, -0.073092, 0.081578, -0.114431, -0.093696, -0.163135, -0.226722, 0.061921, 0.074893, -0.040621, 0.108531, -0.079501, -0.20804, -0.127, -0.168258, 0.042856, -0.103549, -0.100033, 0.088028, 0.098186, -0.192383, -0.119228, 0.050535], [-0.224794, -0.09415, -0.226546, 0.027442, -0.079951, 0.108167, -0.015935, 0.037718, 0.113374, -0.107355, 0.077742, 0.092484, 0.067182, 0.031393, -0.02425, -0.176406, 0.078853, -0.004572, -0.130677, -0.034936, -0.064736, -0.093056, -0.063739, -0.422787, -0.047603, -0.075827, -0.115305, 0.050617, 0.07062, -0.032754, 0.125217, -0.060981], [0.102089, 0.14674, 0.221916, -0.088976, 0.061332, -0.035456, 0.016816, -0.020731, 0.135644, 0.120182, -0.030329, 0.038697, -0.023232, 0.038569, -0.156672, -0.062933, 0.126886, -0.096458, -0.120052, -0.024547, -0.131045, -0.079729, -0.057065, -0.168975, 0.059826, -0.087077, -0.116127, -0.1154, 0.120237, 0.067772, -0.068433, 0.139376], [-0.03635, -0.028649, -0.181238, -0.00947, -0.04814, 0.049512, -0.175, 0.167135, -0.275372, -0.050222, 0.061894, -0.00684, 0.120778, -0.211246, -0.035713, -0.097323, -0.022675, -0.11136, 0.161081, -0.206873, -0.038287, -0.034286, -0.218775, 0.012322, -0.177403, -0.069172, 0.134596, -0.058875, 0.044078, -0.333107, 0.100702, -0.144977], [-0.05783, 0.086668, 0.154427, 0.097822, -0.117644, -0.066773, 0.079787, 0.105304, 0.134315, 0.046113, 0.011045, 0.081025, 0.090235, 0.097948, -0.080324, -0.177899, -0.034473, -0.108228, -0.04769, -0.022175, -0.131446, 0.052365, 0.05065, -0.27968, 0.043412, -0.060811, -0.217227, -0.074533, 0.12275, 0.024781, -0.081882, 0.089705], [0.009391, 0.093498, -0.038431, -0.264405, -0.080431, -0.114156, 0.0224, 0.039531, 0.128316, -0.047408, 0.083579, -0.016416, 0.109113, 0.118971, 0.017049, -0.137279, -0.016748, 0.032217, -0.068086, -0.023018, -0.098118, -0.166799, 0.125217, 0.232778, 0.110888, 0.135028, 0.104252, 0.154061, 0.175555, 0.210056, -0.004495, -0.207193], [0.12533, -0.060445, -0.115675, 0.247602, -0.157984, 0.021233, 0.052467, -0.066127, 0.113765, 0.02116, 0.07205, 0.014906, 0.129581, -0.094423, -0.007683, 0.159942, 0.160532, -0.008833, -0.048163, -0.074274, -0.129032, 0.196709, 0.01163, 0.056099, 0.236557, -0.063623, 0.159325, 0.319968, 0.139458, 0.358401, 0.003079, 0.12887], [-0.033723, 0.005446, 0.188868, -0.03461, -0.141284, 0.132558, 0.083449, 0.00494, -0.127481, 0.064712, -0.050231, -0.08905, 0.072754, 0.054652, 0.233787, -0.077804, -0.030443, 0.059709, 0.153458, -0.240902, -0.067283, 0.167051, -0.121386, 0.245066, -0.155615, 0.023839, 0.140659, 0.04803, 0.006406, 0.011271, 0.006502, -0.072112], [-0.177639, -0.102984, 0.03815, -0.094364, -0.132695, -0.112832, -0.042074, 0.060494, -0.086379, -0.074423, 0.066512, 0.030414, 0.168984, -0.265293, 0.074168, 0.170048, -0.034033, 0.137315, 0.016682, -0.025775, -0.154325, 0.08185, 0.075058, -0.09985, -0.30125, -0.012833, 0.10946, -0.081737, -0.040299, -0.179305, -0.096378, -0.204776], [-0.072119, -0.096167, 0.097815, 0.04932, 0.08469, 0.162026, 0.048621, -0.080677, -0.13191, -0.043535, 0.107522, -0.056667, 0.111466, -0.067097, 0.070708, -0.043948, 0.088741, 0.060198, 0.027726, 0.023704, 0.058339, 0.034258, -0.091021, -0.020477, -0.039797, -0.138635, 0.124879, 0.089091, -0.129999, -0.192556, -0.067821, 0.196432], [-0.277381, 0.015979, -0.199005, -0.296072, 0.144972, 0.128525, -0.001001, -0.033477, 0.07644, 0.023634, 0.130268, 0.119237, 0.14985, -0.22851, -0.196074, 0.072963, -0.09791, 0.117377, -0.227141, -0.034928, 0.130118, -0.228221, 0.147756, -0.126323, 0.06605, -0.093163, -0.244657, 0.086965, 0.018702, 0.054679, 0.095541, -0.13564], [0.352002, -0.249859, 0.244098, 0.330516, -0.079646, 0.122995, -0.026089, 0.329392, -0.314974, 0.355138, -0.383847, -0.251125, 0.225626, 0.021609, 0.314952, 0.346174, -0.009213, 0.092051, 0.149162, -0.134956, -0.02475, 0.268381, -0.007108, 0.480752, -0.009779, 0.275392, 0.134014, 0.209294, -0.112791, -0.037356, -0.236893, 0.090788], [-0.036024, 0.000285, -0.02819, -0.394347, -0.102785, -0.112422, 0.111985, -0.123937, -0.119323, -0.12752, 0.061279, -0.048127, -0.085931, 0.058218, 0.017316, -0.006031, -0.05095, -0.137103, -0.198294, 0.057984, 0.083103, -0.154312, 0.061388, -0.034903, -0.001855, -0.219146, 0.046399, -0.399152, 0.079524, -0.135479, 0.020307, -0.156054], [-0.005337, 0.090457, 0.161058, 0.10042, -0.179976, 0.036422, -0.01101, 0.163797, 0.085245, -0.064917, 0.017662, -0.040004, -0.076612, 0.103365, -0.097168, -0.105126, 0.040719, 0.0565, 0.027801, 0.012174, 0.119711, 0.000991, 0.07107, -0.37774, -0.082987, -0.422834, -0.333072, 0.09481, 0.050853, -0.025497, -0.04658, 0.04832], [-0.053419, 0.108563, 0.116759, 0.007067, -0.001873, 0.121597, -0.012634, -0.325929, -0.019055, -0.196714, -0.050282, -0.071118, -0.071961, -0.043159, -0.192148, -0.245562, 0.089131, 0.044404, -0.090417, -0.011631, 0.197798, -0.317255, 0.047211, -0.348004, -0.029625, -0.312587, -0.064353, 0.279435, 0.043349, -0.065979, 0.090565, 0.179908], [-0.080573, -0.069365, 0.095443, -0.055877, 0.061641, -0.021734, 0.12613, -0.094382, -0.053147, 0.225978, -0.093652, 0.089466, 0.133502, -0.038623, -0.081334, -0.067813, 0.105973, 0.071482, -0.032944, 0.116434, 0.07813, -0.090585, -0.108321, -0.135634, 0.022124, 0.10752, -0.220542, 0.119099, 0.137075, -0.037903, 0.127477, -0.004479], [-0.074319, 0.196759, -0.100155, 0.048857, 0.021319, -0.011418, -0.087139, -0.206491, -0.107597, -0.198492, 0.015817, 0.076715, 0.030962, -0.20277, 0.109367, -0.120568, 0.208068, 0.071625, -0.10443, -0.131692, 0.132058, -0.066638, 0.070836, -0.188669, 0.211042, -0.227727, 0.086097, -0.290554, 0.196456, 0.14963, 0.129014, 0.035206], [0.292055, 0.093731, -0.20163, 0.011814, 0.096898, -0.087933, 0.021311, 0.015452, 0.111696, -0.038793, 0.037832, 0.082904, -0.145441, 0.075534, 0.033252, 0.261455, 0.042926, -0.023173, 0.076743, 0.20695, -0.020346, 0.035695, 0.191415, 0.381803, 0.040122, 0.426712, 0.021982, -0.275231, 0.092344, 0.132444, -0.020767, -0.070921], [-0.060012, 0.091831, -0.060854, -0.087591, 0.077618, 0.106858, -0.157636, 0.147193, -0.013458, 0.136892, 0.124041, 0.121057, -0.014559, -0.019024, -0.043464, -0.049654, -0.081627, -0.033851, 0.072782, 0.150185, 0.049507, -0.007568, -0.072803, 0.058642, -0.295141, 0.178487, -0.041687, 0.049935, -0.064922, -0.523947, -0.054424, -0.116029], [0.264922, 0.080241, 0.14233, 0.073357, 0.078889, -0.045161, 0.174253, 0.031806, 0.056844, 0.033801, 0.087824, 0.06207, -0.206266, 0.082106, -0.074018, -0.154498, 0.157631, 0.201468, -0.046411, 0.199923, -0.068424, 0.000733, 0.11352, -0.295188, 0.127948, 0.017168, 0.003707, -0.060992, -0.008325, 0.081622, -0.011577, 0.220036], [0.010692, -0.085998, 0.076711, 0.138897, 0.060001, -0.419246, 0.145563, 0.012849, -0.128138, 0.082022, -0.23233, 0.060429, 0.083072, -0.189753, 0.164255, -0.072323, 0.023882, -0.067026, -0.041278, 0.10399, -0.301007, 0.054657, -0.005717, 0.116101, -0.235902, 0.157726, 0.01135, -0.289462, 0.093118, -0.194077, -0.039649, -0.102629], [-0.042238, -0.187536, 0.01643, -0.098005, 0.077901, -0.052348, 0.097117, -0.041236, -0.115886, 0.025466, 0.002055, -0.024192, 0.123132, -0.220344, 0.141461, 0.16547, 0.016326, -0.006962, -0.024294, -0.034644, -0.064794, -0.073396, -0.010379, -0.081552, -0.141962, 0.004142, 0.166293, -0.196642, -0.022721, 0.040462, 0.037595, -0.066114], [-0.143866, 0.108586, -0.333329, -0.018377, -0.134756, 0.049684, -0.20304, -0.257908, -0.014137, -0.131611, 0.124373, 0.103652, -0.043239, 0.150152, -0.00667, -0.186648, -0.163756, -0.244944, -0.040497, -0.187735, -0.035403, -0.232436, 0.096197, -0.375579, -0.311538, -0.217836, -0.004416, 0.032359, -0.213213, -0.403336, 0.129349, 0.02434], [0.009882, 0.076731, -0.093131, 0.081917, -0.269833, -0.043429, -0.087997, 0.115455, 0.11944, -0.016528, 0.021718, 0.070342, 0.028396, -0.009186, 0.080827, -0.026595, 0.047097, 0.047772, 0.013466, -0.100533, -0.171174, 0.124022, -0.045953, 0.323332, -0.0279, 0.249169, 0.171557, 0.064128, 0.100912, 0.169821, 0.097917, 0.036876], [-0.345016, -0.080175, -0.069078, 0.040435, 0.134261, -0.031795, 0.085527, 0.039299, 0.120896, 0.013254, 0.133756, 0.14183, -0.032991, 0.013829, -0.167223, -0.098153, -0.080035, 0.126714, -0.021116, -0.043067, -0.100223, 0.007645, 0.038997, -0.067359, -0.117385, -0.219829, -0.071672, -0.112777, 0.168264, -0.047623, 0.107323, -0.12401], [-0.138306, 0.26713, -0.383236, -0.193831, -0.059547, -0.07018, -0.041466, -0.153104, 0.132455, 0.027348, 0.185526, 0.080315, -0.077479, -0.032479, -0.101894, 0.051567, -0.141082, 0.109981, -0.157269, 0.102201, -0.031834, -0.16809, 0.268687, -0.000938, 0.132558, -0.06168, 0.088826, -0.20497, 0.080114, 0.219902, 0.170042, 0.088508], [-0.075044, 0.17261, -0.081623, 0.20987, -0.014304, 0.157946, 0.086146, 0.005513, 0.131402, -0.045487, 0.153091, 0.070982, 0.052147, -0.214681, 0.045914, 0.030235, -0.103124, 0.186306, 0.041729, -0.017521, -0.179141, -0.076814, 0.061413, 0.148024, 0.294019, 0.143807, 0.08333, 0.333793, 0.047446, 0.193422, 0.11437, 0.113373], [0.325483, 0.083844, 0.12638, 0.000234, 0.002518, 0.059911, -0.082074, 0.262022, -0.136529, 0.021034, 0.077772, -0.123222, 0.101875, 0.118737, 0.044429, 0.133909, -0.017814, 0.008904, -0.019816, 0.060922, -0.197317, 0.08153, -0.059602, 0.150538, 0.090818, 0.102005, 0.110185, -0.048726, 0.097549, 0.086061, -0.028492, 0.044223], [0.143523, -0.070582, 0.041254, 0.167757, 0.054833, -0.167381, 0.003446, 0.155001, 0.020207, 0.092274, -0.144665, 0.119324, -0.057266, 0.035588, 0.168527, -0.045726, -0.032238, 0.008669, 0.120864, 0.093498, 0.00713, 0.014314, -0.065148, 0.175875, -0.438932, 0.137419, 0.006151, -0.458988, -0.071331, -0.084368, 0.017988, -0.141943], [-0.138688, 0.121719, 0.079297, -0.04168, 0.154834, 0.135255, 0.112639, -0.166994, 0.039686, -0.047556, -0.026399, 0.040263, 0.056162, 0.014735, -0.039327, -0.169409, 0.025306, -0.005582, 0.044807, -0.060728, -0.01164, 0.023424, -0.125317, -0.141643, -0.110597, -0.245294, 0.02299, -0.068816, 0.032403, 0.011106, -0.034485, 0.109166], [0.236769, 0.041634, -0.089838, -0.249066, -0.362056, -0.156396, 0.063728, -0.080517, -0.053025, -0.0797, -0.036214, 0.063947, -0.027222, 0.075888, 0.009467, -0.167975, -0.071895, 0.052972, -0.056717, 0.02263, 0.019759, -0.117889, 0.07742, -0.595222, 0.311406, -0.327219, 0.119543, 0.282928, 0.072501, 0.117609, -0.065212, 0.176949], [0.109022, -0.005354, 0.034999, 0.020421, -0.021524, -0.291974, 0.137068, 0.047379, 0.218506, -0.156034, 0.006698, -0.054291, -0.051261, -0.3756, 0.217723, -0.017863, 0.165741, 0.209799, 0.06629, 0.150687, -0.115479, 0.057604, 0.056117, 0.210945, 0.363242, -0.050004, 0.054395, -0.041595, 0.163373, 0.376171, 0.044618, 0.04683], [0.035736, -0.146002, 0.05595, -0.014993, 0.129747, 0.158711, -0.1518, 0.115249, -0.180949, -0.032876, -0.027847, -0.179489, 0.129389, -0.073529, 0.114449, -0.019654, 0.022806, -0.347109, -0.046951, -0.141275, 0.013453, 0.089655, -0.130982, 0.091912, -0.294001, 0.051447, -0.166118, -0.21411, -0.110932, -0.038832, -0.15511, -0.100996], [0.371908, -0.053925, -0.02915, -0.122679, -0.322794, 0.006469, -0.040636, 0.08782, -0.051817, -0.023738, 0.140167, 0.150225, -0.134333, 0.111643, -0.003905, -0.225799, 0.005776, 0.168401, 0.068583, 0.145306, -0.027013, -0.188882, 0.148895, -0.05454, 0.173196, -0.119142, 0.111145, -0.116695, 0.00584, 0.088538, 0.142513, 0.08432], [0.081936, 0.059923, -0.062626, 0.050686, -0.01628, -0.423634, 0.102012, 0.133169, 0.075366, 0.172594, -0.0511, -0.026519, 0.018393, -0.010272, 0.253835, 0.19696, 0.063815, 0.003674, 0.143324, 0.0811, -0.079414, 0.044606, 0.058602, 0.400647, 0.12213, 0.179889, 0.284381, -0.087828, 0.146879, -0.056831, 0.103312, -0.046656], [-0.199296, -0.0489, 0.07649, -0.075609, 0.047914, 0.10281, -0.001755, -0.111809, 0.182679, 0.084813, 0.031744, -0.055683, 0.126102, -0.154927, 0.04039, 0.085035, 0.043247, 0.160071, -0.005133, -0.054182, -0.123349, 0.113008, 0.003973, 0.124896, 0.019535, 0.017066, -0.024669, -0.17902, 0.107549, -0.019306, 0.021775, 0.015697], [0.025168, -0.031134, -0.175244, -0.090057, 0.069208, -0.049908, 0.083473, 0.126052, 0.156302, -0.005136, 0.14738, 0.068132, -0.034213, -0.135588, 0.082616, -0.030973, 0.021276, 0.025328, 0.10077, 0.100289, 0.433433, 0.123809, 0.090814, 0.022344, 0.283965, 0.071808, 0.051738, 0.114349, -0.055496, -0.187901, -0.023887, 0.024318], [0.182199, -0.115388, 0.17161, 0.052483, -0.001502, -0.191681, 0.006683, 0.009647, -0.12298, 0.152722, -0.020482, -0.017464, 0.069061, -0.06781, 0.075192, 0.01542, 0.003632, -0.045473, 0.005296, 0.044228, 0.157479, 0.01335, -0.049337, -0.068298, -0.377001, -0.031449, 0.126686, -0.045147, 0.003287, -0.024282, -0.105756, 0.233623], [0.147639, 0.143009, -0.187253, -0.02631, -0.183478, 0.100102, 0.101855, -0.104245, -0.010761, 0.092677, 0.092162, 0.177341, 0.111082, -0.061484, 0.165513, 0.098911, 0.297658, -0.017521, -0.028452, 0.010917, 0.081297, 0.007765, -0.062813, 0.173671, 0.403327, 0.136298, 0.155265, 0.12561, 0.080914, 0.360125, 0.139482, 0.222692], [0.144205, -0.050181, -0.094046, 0.15931, 0.071822, 0.128696, -0.080469, 0.020215, 0.055506, 0.074052, -0.08923, -0.005214, 0.152322, -0.259939, 0.103383, 0.014591, 0.104093, -0.036598, 0.075171, 0.102581, -0.00511, 0.122631, 0.000974, 0.158443, -0.076829, 0.065566, 0.026964, -0.35983, -0.030644, 0.13909, 0.082371, -0.056924], [-0.05061, -0.033255, 0.001848, -0.069204, -0.004468, 0.11265, -0.037416, 0.002659, -0.224756, 0.160461, 0.017195, -0.082738, 0.106748, -0.161055, 0.001218, 0.069306, -0.050265, -0.130656, -0.108712, -0.179554, -0.172436, 0.011377, -0.147083, -0.056233, -0.363454, 0.153776, 0.107219, -0.112764, 0.042564, -0.224948, 0.132275, 0.029979], [0.018599, 0.243391, 0.043744, -0.169744, 0.125772, -0.313374, 0.172075, 0.115267, 0.256783, -0.149895, 0.124866, 0.155717, -0.339357, 0.250803, 0.003157, -0.143074, 0.194236, 0.004038, 0.019401, 0.121442, -0.05385, -0.023983, 0.108895, -0.241077, 0.2623, -0.166542, 0.033903, 0.157854, 0.221343, 0.034657, 0.154924, 0.112408], [0.145821, 0.045883, 0.129977, 0.255805, -0.250248, -0.235149, 0.008838, 0.018268, 0.134128, 0.155365, 0.044705, -0.003859, -0.177408, 0.127637, 0.095521, 0.02746, 0.07259, -0.025824, 0.109471, 0.039908, 0.090036, 0.139867, -0.008972, 0.443153, 0.118737, 0.133417, 0.053225, 0.093711, -0.107572, 0.066615, 0.079146, 0.08217], [0.168751, 0.067095, 0.136686, 0.103603, -0.106031, 0.15509, -0.109964, 0.156796, -0.201511, 0.18175, 0.052534, -0.065797, 0.047933, -0.063799, -0.00549, -0.004267, -0.190017, 0.044731, 0.036887, -0.341522, -0.021963, 0.065962, -0.111347, 0.079739, -0.508118, 0.226558, -0.017641, -0.060756, -0.137161, 0.069152, -0.105239, 0.056449], [0.04432, 0.050038, -0.336996, -0.169039, 0.075849, 0.086952, -0.052985, -0.096018, 0.152512, -0.210259, -0.148983, -0.209881, 0.04991, -0.311453, 0.099941, 0.076131, 0.013616, 0.166427, 0.11389, 0.087493, 0.049131, 0.086355, 0.077259, -0.107524, 0.152974, -0.077694, 0.058326, 0.185801, 0.033577, -0.067427, -0.150077, 0.046779], [0.006469, 0.114677, -0.118957, -0.537808, 0.0011, 0.099029, -0.122182, -0.205687, -0.094459, -0.487303, -0.159707, 0.07594, 0.002623, -0.009718, 0.067556, -0.360927, -0.003278, -0.146948, -0.066474, 0.096783, 0.0989, -0.07007, -0.111428, -0.55442, -0.108742, -0.363082, -0.040823, -0.052284, 0.050346, -0.144738, -0.096321, 0.115449], [-0.267764, 0.009571, 0.221723, 0.2834, 0.107468, -0.067324, 0.041469, 0.017195, 0.126967, 0.210118, -0.04271, -0.129243, 0.13928, -0.236676, -0.137409, -0.09004, -0.059658, -0.034225, 0.127615, -0.071407, -0.104264, 0.06824, -0.22671, -0.059226, 0.074962, -0.018268, 0.019669, 0.009201, 0.078573, -0.107718, -0.164736, 0.080492], [-0.124149, -0.087045, -0.111795, 0.02635, 0.160783, 0.071578, -0.080265, 0.027652, 0.158874, 0.155763, 0.064907, 0.175324, 0.097185, -0.032453, -0.074165, 0.028979, -0.101228, -0.010344, 0.093135, 0.153605, 0.05756, -0.138204, 0.135621, 0.040511, -0.203907, 0.062936, 0.107499, -0.024347, -0.135537, -0.139553, 0.100443, 0.09878], [0.113671, 0.034748, 0.048105, -0.0386, -0.010954, 0.023728, -0.131176, 0.135696, 0.083593, 0.000391, -0.125219, -0.123947, 0.083504, -0.097494, 0.018619, 0.056186, -0.084286, 0.125089, 0.098644, -0.042681, -0.039501, 0.16255, -0.084855, -0.035663, 0.2642, 0.102367, 0.002869, 0.19574, -0.137247, -0.022206, -0.009248, -0.057662], [0.059072, 0.03451, 0.081864, 0.098918, -0.214556, -0.080011, -0.022126, -0.032427, 0.05399, -0.007857, -0.069495, 0.092891, -0.11546, 0.100807, -0.207233, 0.04601, 0.083014, 0.119938, 0.078259, 0.101981, -0.040956, 0.097413, 0.103034, -0.40242, -0.11178, -0.314936, -0.171066, -0.172615, 0.12189, -0.135175, 0.121354, 0.145752], [0.239946, -0.111934, 0.225711, 0.078851, -0.107396, -0.012596, -0.09059, 0.127518, 0.000637, 0.203207, -0.062107, -0.045744, 0.071132, -0.085178, 0.079715, -0.041599, -0.163193, -0.230867, -0.058322, 0.049564, -0.474003, 0.185629, 0.036834, 0.073483, -0.069005, 0.036901, 0.17788, 0.077438, -0.156455, 0.003628, 0.00716, -0.030434], [-0.161463, 0.087847, -0.049083, 0.028928, 0.006271, 0.119965, 0.202185, -0.163435, 0.09199, -0.14673, -0.02166, 0.141689, -0.023135, -0.067062, 0.008247, -0.0785, 0.083686, 0.26759, -0.115855, 0.105163, -0.046387, -0.005637, 0.142868, -0.172495, 0.271195, -0.13507, -0.040704, 0.014607, 0.077656, 0.048455, -0.013087, 0.079233], [0.032223, 0.086702, -0.120242, 0.059099, 0.037341, -0.113693, 0.163956, 0.021151, 0.043536, -0.036412, 0.120313, 0.152666, -0.070439, 0.072058, 0.059582, 0.078982, 0.140579, -0.014423, -0.018496, 0.139833, 0.102464, -0.162882, 0.046497, -0.316673, -0.048723, -0.138091, -0.033711, -0.058919, -0.005003, 0.071911, 0.08999, -0.011318], [-0.021208, 0.033439, 0.085885, 0.037726, -0.242506, -0.032466, -0.0224, 0.083104, -0.1901, 0.190249, 0.13381, 0.143343, 0.033024, -0.048366, 0.033736, 0.084414, 0.07803, -0.31593, 0.063448, -0.222491, -0.102181, 0.088352, -0.212134, 0.146742, -0.300375, 0.000698, 0.088936, -0.104911, 0.081042, -0.075977, -0.024879, -0.054169]], [-0.24274, 0.082755, -0.337046, -0.231694, 0.070126, 0.113615, 0.154552, -0.089441, 0.055633, -0.094134, 0.021851, 0.059141, 0.168617, -0.224629, 0.012994, 0.117837, 0.111959, -0.116726, 0.043772, 0.079677, -0.079998, -0.057897, -0.162953, -0.189234, -0.214935, -0.071965, 0.015619, -0.184992, 0.215223, -0.120087, 0.046275, -0.047995]),
    ([[0.24401], [-0.102758], [0.362113], [0.162746], [-0.146376], [-0.210802], [-0.121357], [0.161102], [-0.148456], [0.172548], [-0.039107], [-0.058002], [0.089405], [0.251067], [0.106654], [0.110636], [-0.043503], [-0.085155], [0.069042], [-0.075451], [0.168933], [0.174218], [-0.258425], [0.233969], [-0.288825], [0.202559], [0.182447], [-0.176889], [-0.130052], [-0.21961], [-0.092056], [0.243795]], [0.090072]),
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
