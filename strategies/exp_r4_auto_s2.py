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
SEQ_TM = [2.75191, -9.2e-05, 9.2e-05, 0.0, 0.052846, 0.052846, -0.0, 0.0, 0.0, 1.243668, 5.879185, 5.879185, 0.0]
SEQ_TSD = [1.079295, 0.132411, 0.132411, 0.099558, 1.45424, 1.45424, 0.666687, 0.219428, 0.17921, 0.60273, 1.666237, 1.666237, 0.649942]
SEQ_XM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.257423, 0.0, 0.0, 5.846561, 5.846561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987482]
SEQ_XS = [1.124955, 0.99101, 0.94523, 0.845186, 0.757041, 2.577918, 18.130564, 0.790721, 15.945622, 0.69697, 0.6513, 0.540497, 1.677766, 1.677766, 0.999993, 4.402977, 1.087599, 0.310604, 132.411407, 334.802429, 1066.728027, 1.341266]
SEQ_YSC = 1.939010739326477
SEQ_WI = [[-0.434753, 0.031188, -0.264576, -1.117368, -0.495722, -0.531406, -1.108726, -0.287513, 0.010296, 0.114371, -0.396365, -0.496648, -0.36923, -0.197228, 0.140836, -0.024166, -0.419023, -0.406482, -0.153849, -0.552748, -0.070451, 0.350616, -0.591333, 0.341195, -0.355785, -0.504126, -0.309319, -0.212602, 0.177411, -0.27078, -0.245068, -0.065639, 0.233167, 0.286888, 0.176335, 0.240348, 0.161801, 0.086538, -0.11771, 0.002491, -0.14436, -0.283248, 0.234805, -0.082355, 0.425428, -0.217892, 0.256304, 0.162068], [-0.17339, 0.275333, -0.819949, -0.052572, -0.312871, 0.367341, -0.458874, 0.495385, -0.224863, 0.180963, 0.102233, 0.220196, -0.067877, -0.109692, 0.043735, -0.094751, -0.603071, 0.313795, 0.160814, -1.013308, 0.017401, -0.358811, 0.338349, 0.330873, -0.163679, -0.37371, -0.265529, -0.548234, -0.201491, -0.039469, 0.316544, -0.571329, 0.581944, -0.330524, 0.194984, -0.010412, 0.025134, 0.043664, -0.210897, 0.06854, 0.132443, 0.08325, 0.358854, 0.038163, -0.069392, 0.116398, 0.203487, 0.289133], [0.034305, -0.439031, 0.539729, -0.143061, 0.056448, -0.116568, 0.165666, -0.603136, 0.346861, 0.442464, -0.419395, -0.295968, 0.283708, 0.070228, 0.285363, 0.005984, 0.006032, 0.089336, -0.057843, 0.131227, 0.743081, 0.062604, 0.252597, -0.136852, 0.032432, -0.082359, -0.087142, 0.673483, -0.196473, -0.343769, 0.050956, 0.227344, -0.075954, -0.00099, -0.231331, 0.114595, -0.403531, 0.489884, 0.194692, 0.539557, 0.042951, 0.541019, 0.006629, -0.067031, 0.06293, -0.064342, 0.184411, 0.048609], [-0.089674, 0.179794, 0.006952, 0.089685, -0.080114, 0.130729, -0.248189, 0.294815, 0.172259, 0.492011, -0.340362, -0.52897, 0.118495, -0.046219, 0.318264, -0.015124, -0.239652, -0.195972, 0.1095, -0.115534, 0.499268, 0.006294, 0.179002, 0.340981, 0.252255, -0.414193, -0.250057, 0.379848, -0.627564, 0.104509, 0.20332, -0.132841, 0.305861, 0.122744, 0.08129, -0.066045, -0.062863, -0.117648, 0.052657, 0.432032, -0.220856, -0.082203, 0.244227, 0.282788, 0.016655, 0.005482, 0.354231, 0.083263], [-0.544032, 1.275211, -0.392316, -0.73622, -0.491874, 0.315365, -0.268823, 0.126285, -0.685962, -0.039999, 0.670788, 0.437718, -0.139366, -0.512738, -0.194261, -0.074335, -0.650526, 0.433565, 0.25607, -0.03278, 0.226758, 0.168794, 0.037235, 0.891948, -0.264112, -0.136892, 0.32386, 0.1931, -0.002009, -0.391036, 0.685521, -1.355545, 0.697502, -0.582054, -0.074466, 0.117797, 0.303841, 0.005504, -0.268954, 0.072038, -0.488976, 0.181277, 0.101994, -0.130734, 0.052777, -0.230625, -0.077144, 0.608686], [0.01236, -0.831926, -0.202052, 0.625973, -0.478593, 0.135512, 0.920227, -1.620611, -0.118657, 0.190691, 0.067154, 0.45338, 0.284754, -0.497704, -0.220415, 0.665075, 0.100328, -1.540524, 0.145605, 0.308634, 0.675478, 0.278092, -0.119885, -0.501102, -0.408829, -0.537936, 0.063587, -0.253695, -0.076951, -0.193341, -0.945436, 0.391654, -0.107919, 1.162795, -0.029269, 0.122107, -0.062284, 0.01017, -0.574422, 1.240275, 0.137862, 0.160453, 0.035748, -0.201368, 0.217775, 0.211236, 0.369475, -0.081928], [-0.465253, 0.004051, -0.010658, 0.098986, 0.106469, 0.515667, -0.514095, 0.330257, 0.133599, -0.143863, -0.042499, -0.039808, 0.249428, -0.482388, 0.276718, -0.335337, 0.084814, 0.106428, -0.006252, -0.447009, -0.122858, 0.101985, 0.160817, 0.103327, 0.372855, -0.129285, 0.323279, -0.317851, -0.062506, -0.041775, -0.064195, 0.547245, 0.003236, 0.359784, -0.415423, -0.094201, -0.102569, -0.021141, 0.409538, -0.160828, -0.353738, -0.068161, 0.298659, -0.013273, -0.541541, 0.100092, 0.187135, -0.280076], [-0.316302, -0.579609, 0.042682, 0.213172, -0.024454, 0.122605, 0.03867, 0.096023, 0.014955, -0.258203, -0.450786, 0.230877, -0.283112, -0.275863, 0.058931, -0.263895, 0.257566, 0.037759, 0.100739, -0.061098, -0.123799, 0.097405, 0.0787, -0.239403, 0.248785, -0.004425, -0.010999, 0.100118, -0.244155, 0.060676, 0.055043, 0.213928, -0.278047, 0.366932, -0.492034, -0.371314, -0.037077, -0.191916, 0.060447, 0.063432, -0.270144, 0.101036, 0.14948, -0.066726, 0.20287, 0.013503, -0.057888, -0.101757], [-0.144328, -0.355672, -0.042947, 0.028671, 0.013475, 0.352279, 0.17094, 0.014819, -0.197704, -0.446421, 0.004801, 0.118819, 0.188003, -0.000585, 0.122811, -0.270679, 0.230591, -0.113889, -0.451507, 0.105532, -0.158796, -0.321369, 0.345225, -0.359431, 0.261386, -0.015189, 0.072004, -0.307346, 0.345786, 0.336739, 0.296912, 0.441782, -0.124646, 0.118474, -0.21659, -0.179492, 0.058626, 0.248593, -0.295123, -0.16266, 0.041419, 0.32391, -0.080388, 0.016382, 0.109486, 0.553177, 0.077401, -0.04721], [-0.083635, 0.096018, 0.065432, 0.781015, -0.132178, 0.264469, 0.308398, -0.195604, -0.219514, 0.081902, 0.352968, -0.309608, 0.13511, 0.407885, -0.403524, 0.180701, 0.093333, -0.108239, -0.195224, 0.37577, -0.340143, 0.54734, -0.421579, 0.08755, -0.152667, 0.239706, 0.543437, 0.543326, 0.541813, 0.078319, 0.771557, -0.37577, -0.019851, -0.195997, 0.081399, 0.375796, -0.008832, -0.132326, 0.011446, 0.351492, -0.485257, -0.566493, 0.238016, -0.325437, 0.136929, 0.158953, -0.087484, -0.298008], [0.087266, -0.124276, 0.557799, 0.370631, 0.387917, -0.070675, -0.198284, 0.321934, -0.134489, 0.212755, 0.141874, -0.01676, -0.575123, -0.184315, -0.412657, 0.158753, -0.196973, -0.046514, 0.208128, 0.132056, 0.059841, 0.076724, 0.099597, -0.585246, -0.041684, -0.33572, -0.038474, 0.061077, -0.15137, -0.612218, -0.431124, 0.096221, 0.018461, -0.059464, -0.272445, -0.078022, 0.143827, 0.098834, -0.092666, 0.285637, 0.062644, -0.203979, -0.070701, -0.053712, -0.155152, -0.070584, -0.296293, -0.013296], [-0.261783, 0.09399, 0.065074, 0.322616, 0.162996, -0.112323, -0.524113, 0.198121, -0.216359, -0.187239, 0.095381, 0.124992, -0.021349, -0.471382, -0.097574, 0.05972, -0.091411, -0.468727, -0.308748, 0.135101, -0.244709, -0.401458, -0.576711, 0.211036, -0.151102, 0.065159, 0.079576, -0.17247, 0.125892, -0.033597, 0.12131, 0.063384, 0.032605, -0.093039, -0.128862, -0.128465, -0.014632, -0.252694, 0.260431, 0.163252, 0.293918, -0.073565, 0.039706, -0.308002, -0.026907, 0.138419, -0.002536, 0.179091], [-0.09282, -0.424307, 0.019804, 0.533931, 0.185422, -0.347703, 0.242222, 0.049507, -0.027974, -0.491603, 0.023928, -0.459772, -0.430256, 0.294436, -0.177924, -0.107926, 0.086952, 0.182371, 0.737344, 0.171245, 0.281804, 0.695457, 0.442503, -0.428072, 0.046587, -0.432163, -0.288875, -0.010562, 0.026259, -0.19814, -0.472095, -0.15477, -0.017789, 0.062794, -0.02027, -0.075654, -0.134696, -0.035546, -0.142659, 0.11169, -0.237505, 0.125286, -0.035459, 0.07749, 0.065657, -0.154292, -0.324969, 0.095573]]
SEQ_BI = [0.179891, 0.014829, 0.091233, 0.152213, 0.419257, 0.315906, -0.078977, -0.190461, -0.238968, 0.540061, -0.206012, 0.153894, 0.176323, 0.36672, 0.159489, -0.016709, 0.730349, 0.497299, 0.475154, 0.508978, 0.479423, 0.520436, 1.070878, 0.575235, 0.344922, 1.003528, 1.032946, 0.077054, 0.385301, 0.503744, 0.228931, 0.746418, 0.122904, 0.117205, 0.222498, 0.083426, -0.031255, -0.251613, 0.167796, -0.254979, -0.088538, -0.152452, -0.290295, -0.087268, -0.287676, 0.267931, 0.399791, 0.088683]
SEQ_WH = [[-0.494842, 0.048241, -0.422114, -0.136924, 0.078967, -0.040573, 0.134398, -0.12328, -0.387577, 0.005743, -0.602106, 0.050431, -0.347046, -0.541171, 0.058058, -0.360649, 0.078949, 0.679185, -0.233763, 0.414104, 1.17012, 0.169533, 0.312902, 0.058478, -0.842381, 0.264968, 0.32546, -0.028069, -0.569841, 0.343387, -0.945359, -0.562151, 0.473792, -0.325183, 0.392678, 0.065784, -0.16112, -0.023397, -0.34376, -0.203479, -0.479743, -0.075376, 0.213479, -0.565633, -0.515216, -0.097351, 0.226359, 0.444011], [-1.017622, 0.045628, -0.071553, -0.132892, -0.215793, -0.217669, -0.123927, -0.342, -0.065674, -0.360294, -0.003688, -0.012115, -0.465946, -0.448797, 0.125038, -0.232522, -0.116857, -0.332462, 0.226692, -1.092298, -0.692478, -0.315334, 0.059047, -0.697052, 0.036418, -0.358398, -0.778365, 0.088238, 0.132031, -0.568421, 0.080552, -0.036546, 0.131139, 0.551444, 0.568351, -0.043883, 0.740007, -0.036121, 0.125978, 0.360543, -0.079508, 0.692143, -0.085217, -0.082536, 0.109749, -0.662881, -0.09423, -0.168353], [-0.420088, -0.032529, -0.646594, 0.502806, -0.072527, 0.111613, 0.127856, -0.915175, -0.280581, 0.117042, -0.122919, -0.460471, -0.445366, 0.308805, -0.213222, 0.515712, -0.002191, 0.516848, -0.799957, 0.168053, 0.707694, -0.435635, -0.108443, 1.123508, 0.212997, -0.177942, 0.264692, -0.115809, 0.179598, 0.339563, -0.707474, -0.047094, 0.395311, 0.021301, 0.386287, 0.505228, 0.139007, 0.076572, -0.658161, 0.371382, -0.163189, -0.143047, 0.014187, -0.195272, -0.069148, 0.276741, 0.137249, 0.066927], [0.337883, -0.113849, 0.083333, -0.021336, -0.415945, 0.101769, 0.275126, -0.191929, -0.245222, 0.537839, 0.143564, -0.437804, -0.339432, 0.432306, -0.401372, 0.213394, 0.060793, 0.040332, 0.160479, -0.297925, 0.640022, -0.097823, 0.474325, 0.6506, 0.185831, -0.14972, -1.001793, 0.287865, -0.192022, -0.477902, 0.082068, 0.098571, -0.16429, -0.115665, 0.305574, 0.156612, 0.049719, -0.435633, -0.452551, -0.259987, 0.081846, -0.477593, 0.294637, 0.097167, 0.145278, 0.176127, 0.253872, -0.364167], [0.192015, 0.04508, 0.252518, -0.104322, 0.046482, -0.18596, 0.062548, 0.02835, -0.245987, 0.001696, 0.765016, 0.013215, -0.207865, -1.063156, 0.436323, 0.600034, -0.017221, -0.281595, 0.17693, -0.872809, 0.062074, 0.217236, -0.773315, 0.565515, -0.353866, -0.572165, -1.049021, 0.19608, -0.719192, 0.153569, -0.092579, 0.639032, -0.123747, 0.442804, 0.073905, -0.107266, 0.17982, -0.676472, -0.245186, -0.087591, -0.328561, 0.248731, -0.104163, -0.166906, 0.292633, -0.151327, 0.384594, 0.530408], [0.114982, -0.030838, 0.221069, 0.02357, 0.118611, -0.389467, 0.174594, 0.147703, 0.114998, -0.51214, -0.12899, 0.078933, 0.131342, -0.147035, 0.32808, -0.480839, -0.104723, -0.706518, -0.864491, -0.222345, -0.120391, -0.297018, -0.723997, -0.241481, -0.654862, -0.298883, -0.669877, -0.092516, -0.610981, 0.135116, -0.18462, -0.550138, 0.221913, 0.166334, 0.184504, -0.232711, -0.271017, 0.221408, 0.3347, 0.364233, -0.071837, -0.050863, -0.050617, 0.002226, 0.065756, -0.510721, -0.183545, 0.303742], [-0.047461, 0.188064, -0.209561, -0.04242, -0.021787, -0.418209, -0.169592, 0.091467, 0.184511, 0.157947, -0.338895, 0.599797, 0.162143, -0.070196, 0.068881, -0.094469, 0.542106, 0.133672, 0.275145, 0.157999, -0.308857, -0.751403, 0.236609, 0.146261, -0.417887, -0.765142, 0.008931, -0.046383, 0.078548, 0.171946, 0.03273, -0.362729, -0.093074, -0.358237, -0.15206, -0.332504, 0.079141, 0.648505, 0.164582, -0.070913, 0.083292, 0.186841, 0.081483, 0.649596, 0.443265, -0.115604, -0.095532, 0.128731], [0.316268, 0.123848, 0.727019, -0.18137, 0.058731, -0.048483, -0.075892, -0.318818, -0.024385, -0.159705, 0.425873, -0.086083, 0.034612, -0.108779, -0.4919, 0.055181, -1.150552, -0.101316, -0.230976, -0.680773, -1.624402, -0.255003, -0.690566, -0.36043, 0.335334, 0.415693, -0.269946, -0.150526, -0.255868, -1.099926, -0.133574, -0.788708, -0.169233, 0.375773, -0.176471, 0.207087, 0.449932, -0.095315, 0.064497, 0.410755, -0.041005, -0.026159, -0.293778, 0.058056, -0.026668, -0.347886, -0.451372, 0.031155], [0.131492, -0.053423, 0.071369, 0.264003, 0.139778, 0.017428, 0.082744, 0.152493, 0.223743, 0.003567, 0.332859, 0.645791, 0.468685, -0.011906, -0.056973, -0.276284, -0.153068, -0.037194, 0.713219, 0.080979, -0.085887, -0.635206, 0.752077, 0.014243, -0.008775, -0.7638, -0.006933, -0.642594, 0.193176, 0.407451, -0.560054, -0.061053, -0.644429, 0.112118, -0.142446, -0.123529, 0.336671, 0.078799, 0.191478, 0.007373, 0.468568, 0.447498, -0.281004, -0.266219, 0.390293, 0.167661, -0.157169, -0.001447], [0.27913, 0.107332, -0.197206, 0.291984, 0.072323, -0.391146, 0.15899, -0.026722, 0.270866, -0.39175, -0.018427, 0.490077, 0.244066, -0.08618, 0.913302, -0.079652, 0.021816, 0.017268, 0.824147, -0.376391, -0.249983, -0.518449, 0.309553, -0.425994, -0.136742, 0.565282, -0.07803, -0.408998, 0.054081, -0.463945, 0.324488, 0.176938, -0.334516, -0.004487, -0.501521, -0.505572, 0.245565, -0.073068, 0.012168, -0.029919, 0.207051, 0.819556, 0.148128, 0.430775, -0.252222, -0.466158, 0.48269, 0.120921], [-0.165011, 0.260511, -0.087074, -0.464909, 0.37697, 0.257416, 0.072498, -0.345818, -0.100383, -0.040439, -0.126134, 0.130492, -0.055841, 0.00656, -0.258344, 0.066622, -0.938849, -0.337982, -0.164243, 0.128447, 0.318202, 0.351474, -0.843502, -0.541566, -0.770595, 0.017287, -0.060261, -0.201955, -0.592563, 0.466348, -0.166847, -0.698084, 0.062293, 0.07785, -0.079303, -0.071085, -0.307331, 0.242849, 0.036312, 0.200104, 0.04342, -0.419944, 0.462059, -0.399178, 0.163003, -0.34651, -0.228899, -0.119217], [0.358527, 0.120436, 0.036693, 0.223324, -0.698031, -0.306605, -0.236977, 0.022113, 0.745643, -0.505796, -0.156348, -0.296895, 0.200951, -0.028892, 0.402276, -0.254481, -0.127848, -0.30589, -0.847948, -0.025429, -0.076729, -0.232009, -0.013352, -0.87961, 0.955071, 0.631959, 0.165791, -0.318603, 0.289612, -0.036444, 0.870929, -0.655687, -0.161989, 0.00853, 0.042555, -0.001706, -0.069738, 0.200417, 0.360536, 0.263725, -0.07113, 0.239409, -0.471887, 0.620591, 0.110243, -0.012926, 0.261297, -0.344681], [0.399634, -0.075489, 0.633403, -0.010774, -0.295219, -0.517786, -0.084735, -0.390541, 0.294294, 0.073882, 0.470229, -0.147694, 0.138932, 0.510444, -0.234812, 0.059464, -0.224387, -0.27414, 0.682509, -0.35048, -0.968462, 0.141634, -0.421018, -0.005177, 0.350937, -0.543163, -1.495415, -0.253101, -0.028094, -0.964326, 0.205585, -0.686166, -0.101515, -0.03929, 0.094344, 0.125198, -0.050084, -0.184979, 0.138896, 0.015749, -0.184779, 0.177733, 0.396441, -0.036136, 0.776655, 0.215527, -0.463836, -0.369463], [-0.147793, -0.16364, -0.14059, 0.152408, 0.042571, 0.321091, 0.167988, -0.263889, -0.120838, 0.562923, -0.052113, -0.431593, 0.265526, 0.684562, -0.558185, 0.455785, 0.283419, 0.470016, 0.341144, 0.773969, -0.367489, 0.667565, 1.180984, 0.032846, 0.42739, 0.232313, 0.57547, 0.001082, 0.794874, -0.400183, 0.443727, 0.927801, 0.196852, -0.281226, -0.094901, 0.309764, 0.108251, -0.602454, -0.471969, -0.362409, -0.045174, -0.160045, -0.205027, -0.070251, -0.020609, 0.345928, -0.102207, -0.516987], [-0.562328, -0.012744, -0.520505, 0.277439, -0.042864, 0.260657, -0.145827, 0.047615, 0.077158, 0.050192, -0.506073, -0.177263, -0.185191, 0.007863, 0.264533, 0.06685, 0.674522, 0.009833, -0.60512, 0.332997, 0.452516, -0.134478, 0.420324, -0.475259, -0.267854, 0.587602, 0.953922, 0.331076, 0.458805, -0.292562, 0.283253, 0.65586, 0.402333, -0.208234, 0.341425, -0.150072, 0.022283, -0.248279, 0.035641, -0.121584, 0.055511, 0.287502, -0.005376, 0.224956, -0.26409, -0.064095, 0.079739, 0.026285], [-0.389512, 0.164379, -0.073702, -0.895247, 0.227643, -0.283634, -0.218924, 0.080477, -0.024733, -0.088079, -0.064049, 0.731598, -0.017172, -1.115256, 0.408067, -0.174675, 0.462294, 0.637263, -0.040832, -0.062447, 1.064673, -0.241143, 0.032833, -0.048205, -0.420683, 0.153844, 0.470359, 0.353936, -0.591965, -0.123493, -0.698423, -0.254783, 0.186593, 0.368755, 0.285884, -0.326259, 0.54955, 0.245861, -0.20313, -0.13745, -0.083815, 0.140994, -0.152253, -0.076446, -0.246014, -0.501188, 0.100747, 0.717765]]
SEQ_BH = [-0.075939, 0.26149, -0.041254, -0.050286, 0.202777, 0.598467, -0.059199, -0.082347, -0.013155, 0.558875, 0.148293, 0.155041, 0.24961, 0.243397, 0.311651, -0.033192, 0.899228, 0.416516, 0.712488, 0.755648, 0.568651, 0.405542, 0.619756, 0.326522, 0.188637, 1.0101, 0.872908, 0.242106, 0.405023, 0.125138, 0.167206, 0.625426, -0.184164, -0.027867, -0.206469, 0.165025, 0.094767, -0.46031, 0.178275, -0.166741, 0.280949, -0.114414, -0.041187, -0.177193, 0.044837, 0.1966, 0.29557, -0.118024]
SEQ_HEAD = [
    ([[0.082942, -0.022584, -0.353576, -0.088689, -0.315811, 0.0931, -0.068238, -0.042783, 0.191853, 0.133934, -0.039388, -0.093539, 0.039677, 0.05573, -0.142187, 0.225227, -0.07065, 0.086327, 0.135102, -0.042669, 0.178754, 0.101317, -0.114496, 0.079385, -0.159263, -0.126016, 0.018554, 0.11866, 0.124236, -0.070215, -0.3468, -0.010317, 0.011442, 0.246221, -0.00417, 0.123414, -0.12517, -0.003456, 0.108144, 0.151233, -0.108306, 0.095503, 0.102894, 0.010406, 0.146352, 0.045063, -0.08248, 0.146369, -0.054723, 0.041376, 0.079339, -0.195512, 0.046085, -0.006329, 0.190546, -0.04267, 0.074438, 0.083254, 0.017151, -0.058268, -0.016187, -0.015025, 0.067764, -0.001069], [-0.17175, 0.053742, -0.032707, 0.115545, -0.298398, 0.007606, -0.130883, 0.077524, -0.03243, -0.180958, 0.104269, 0.146564, -0.04465, 0.134146, -0.048856, 0.145629, 0.112142, 0.010153, -0.089144, 0.061541, 0.038332, 0.082926, -0.063138, 0.11779, -0.077844, -0.005446, 0.114995, -0.140932, 0.198804, 0.028354, -0.165655, 0.035013, -0.131663, -0.078707, -0.054913, -0.040129, -0.002073, -0.132581, -0.038388, 0.025922, 0.008297, -0.071465, 0.123003, -0.020857, -0.084797, 0.015915, -0.042168, 0.079479, 0.099676, -0.073574, -0.023666, 0.052399, -0.102662, -0.044814, 0.226548, 0.001828, -0.035278, 0.055281, -0.069206, -0.007338, -0.17759, -0.131171, -0.058878, 0.157352], [0.116576, -0.075134, -0.030828, -0.001925, -0.094248, -0.001435, -0.061377, 0.022184, 0.135143, -0.039861, 0.112512, 0.029531, -0.004273, 0.135051, 0.002979, -0.086649, 0.153444, -0.050951, 0.074183, -0.108087, -0.137137, 0.227235, 0.194212, -0.069849, -0.217797, -0.003389, -0.171722, 0.112442, 0.133555, 0.064878, -0.023889, -0.156928, -0.032062, -0.022542, -0.032607, 0.096903, 0.090302, 0.051236, -0.078992, -0.06985, -0.000776, 0.026367, 0.15382, -0.034326, 0.009101, 0.049289, 0.079497, 0.132052, 0.090181, 0.014442, 0.126346, 0.08733, 0.111729, -0.116287, 0.087838, 0.087093, -0.168295, -0.13827, -0.014495, 0.140802, -0.055104, -0.131051, 0.119741, -0.077185], [0.151018, 0.167487, 0.278813, 0.069115, 0.162588, -0.034051, -0.020119, 0.195751, -0.028508, 0.011762, 0.056895, 0.18484, 0.065836, 0.159641, -0.015152, 0.078178, -0.011644, 0.06845, 0.040189, 0.156502, -0.021163, 0.015371, 0.063228, 0.01832, -0.148765, -0.046439, -0.004831, 0.225081, -0.103486, -0.02199, 0.228395, -0.044941, -0.21063, -0.346095, -0.127354, 0.037475, 0.128718, 0.067463, -0.095513, -0.181065, -0.144045, -0.216428, 0.126782, -0.100571, -0.046397, -0.176002, 0.165416, -0.195347, -0.078875, 0.108347, -0.116186, -0.153544, -0.037537, -0.106252, -0.117604, 0.111863, -0.085946, -0.22487, 0.254891, 0.341043, 0.0929, -0.121102, -0.155629, -0.183308], [-0.160671, -0.17961, -0.041781, 0.051685, 0.366473, 0.12418, -0.095154, -0.109375, 0.013734, -0.115093, 0.050713, 0.176029, -0.115849, 0.057044, -0.127347, -0.145427, 0.186071, -0.077389, -0.08224, -0.190934, -0.196817, -0.133838, -0.259753, 0.316444, 0.001955, 0.200952, -0.263038, -0.097938, -0.160736, -0.096937, 0.247723, -0.190931, 0.02682, 0.05219, -0.23893, 0.0027, -0.05661, 0.004849, -0.373158, 0.123579, -0.13641, -0.043579, 0.212294, 0.036522, -0.120969, 0.168889, 0.342043, -0.391795, 0.070754, 0.250666, 0.116202, -0.107005, -0.258564, -0.218817, -0.225301, -0.033658, -0.212922, -0.219242, 0.313598, 0.075326, -0.169476, 0.009391, -0.328964, 0.285431], [0.001978, -0.064696, 0.171158, 0.105841, 0.070262, -0.101078, 0.012463, 0.066147, -0.170747, 0.031631, -0.138044, 0.116372, 0.151705, 0.121866, -0.027658, 0.188581, 0.049039, 0.040135, 0.078658, 0.084117, 0.178276, -0.159843, 0.192439, -0.080047, -0.130071, 0.084762, -0.230981, -0.123422, -0.146222, -0.010316, 0.128535, 0.069794, -0.150575, -0.073631, 0.207493, -0.050396, -0.020383, -0.043259, -0.089686, -0.195278, 0.084831, 0.064038, 0.025142, -0.147194, -0.132881, 0.084507, 0.00512, 0.063239, -0.031723, -0.097721, -0.09092, 0.163086, 0.109103, 0.12238, 0.189937, -0.100776, 0.051449, -0.092702, -0.062579, -0.039959, -0.070948, -0.135991, 0.089976, 0.033237], [0.011799, -0.022569, -0.199227, 0.068112, -0.256349, 0.161707, -0.00953, 0.302829, 0.064696, 0.058981, -0.071606, 0.007419, 0.233733, 0.037419, -0.023515, 0.049825, 0.051499, 0.172127, -0.041367, 0.105555, -0.090427, 0.108976, -0.099916, 0.05502, 0.006479, 0.080006, -0.018564, 0.182046, 0.239246, 0.210585, -0.31249, -0.10718, 0.072215, 0.080214, -0.034989, -0.028623, -0.087645, -0.052995, -0.092035, 0.093352, 0.019504, 0.063939, -0.017861, -0.032416, -0.210661, -0.061703, 0.218195, -0.111983, -0.106407, 0.035238, -0.045291, 0.145299, 0.023154, -0.072923, 0.01825, -0.023895, -0.236766, 0.046032, 0.099299, 0.072149, 0.02435, 0.247823, -0.323255, -0.03312], [0.155843, -0.059512, 0.083347, -0.135566, 0.152418, 0.003738, 0.049814, 0.110115, -0.065001, -0.03203, -0.203537, 0.007979, -0.073683, 0.049927, -0.08858, 0.062712, 0.112242, -0.091024, 0.089108, -0.001121, -0.244866, 0.049886, 0.140258, -0.016229, 0.115837, -0.070622, 0.045512, 0.001726, -0.041347, -0.047021, -0.035356, -0.102586, -0.005681, -0.021098, 0.086548, -0.133926, -0.022596, 0.07183, 0.071307, -0.113586, -0.083358, 0.076229, -0.14651, -0.037167, 0.026133, 0.048448, -0.03868, 0.034359, -0.007635, -0.007694, -0.253376, 0.343314, -0.224904, 0.124987, -0.061424, -0.194561, -0.030419, -0.156879, -0.276486, 0.137304, 0.078227, 0.075384, -0.095154, 0.174768], [-0.137917, -0.062565, -0.224234, 0.001229, -0.244088, 0.190753, -0.099945, -0.005038, -0.087754, 0.034779, 0.199557, 0.002969, -0.080581, 0.034081, -0.300314, 0.146442, 0.01568, 0.221624, -0.05704, -0.019377, -0.019613, 0.272533, 0.101453, 0.17942, -0.114036, 0.044554, -0.021416, 0.141262, -0.180116, 0.084339, -0.213491, -0.005709, -0.315584, 0.061687, 0.048352, 0.157253, 0.194688, -0.020934, 0.107474, -0.189079, -0.095125, -0.182736, -0.058551, 0.057916, -0.115252, 0.111975, 0.049947, -0.266944, 0.084338, 0.086945, 0.036867, 0.03261, 0.137918, -0.075606, 0.090841, -0.304329, -0.3697, -0.163812, 0.005564, 0.167381, -0.207751, -0.047425, 0.183539, 0.291195], [0.026134, 0.093697, -0.042734, 0.137481, 0.112933, -0.175419, -0.054183, -0.005063, -0.423877, -0.103071, 0.016029, 0.11215, 0.149047, -0.061027, 0.011479, -0.007105, -0.241679, -0.11998, -0.031668, 0.211219, 0.063005, -0.289053, -0.12505, 0.05774, 0.006028, 0.165635, -0.146467, -0.126567, -0.249338, 0.057175, -0.311925, 0.130179, 0.114467, 0.053872, -0.067628, -0.075267, -0.2489, -0.108471, 0.04495, -0.020787, -0.449047, 0.007237, -0.040994, 0.158397, 0.127715, 0.199033, 0.147288, 0.083385, 0.021837, 0.074545, 0.152554, -0.029424, 0.015625, 0.111932, -0.120094, -0.185023, -0.117738, -0.251545, -0.132515, 0.057871, -0.053976, -0.035332, -0.030827, 0.103336], [-0.134137, 0.098496, 0.120945, 0.026098, -0.063972, 0.014174, -0.084038, -0.024848, 0.061047, -0.018559, 0.06785, -0.065148, -0.043644, 0.079181, -0.006369, -0.093911, 0.115553, 0.165705, 0.188762, 0.009546, 0.070529, 0.019384, -0.010562, -0.198518, -0.059365, -0.031651, 0.045439, -0.325607, -0.064808, -0.074994, 0.047599, -0.035296, -0.126121, -0.072106, -0.118171, -0.10996, -0.001769, 0.120248, 0.121223, 0.059766, 0.000117, 0.037384, -0.03809, -0.041254, -0.044035, -0.144069, -0.109282, 0.007393, -0.084272, -0.091038, 0.012316, -0.131843, 0.011101, 0.000108, -0.060483, 0.027561, -0.178635, 0.088825, -0.023174, 0.04959, 0.046278, -0.073089, 0.056047, -0.014158], [-0.192308, -0.340151, -0.029081, 0.09189, -0.031016, -0.136082, -0.331181, -0.026513, 0.136122, -0.163593, 0.114142, -0.00476, 0.073454, 0.077662, 0.031336, 0.366636, 0.401054, 0.310191, -0.025853, 0.124072, -0.186228, 0.314265, 0.058586, 0.202645, -0.176325, 0.018266, 0.191303, 0.195172, 0.264941, -0.051708, -0.340743, -0.230126, 0.040005, -0.152785, 0.077252, 0.076658, 0.089712, 0.173528, 0.267427, 0.061047, -0.29228, -0.082001, 0.237058, -0.024667, -0.156529, -0.004685, -0.122155, 0.007308, 0.032823, 0.104116, 0.223161, 0.157734, 0.017929, -0.210246, -0.172842, -0.343133, -0.1245, -0.235671, -0.042591, 0.090179, 0.112886, 0.06574, -0.165629, 0.28815], [0.137814, 0.322991, -0.088329, -0.108968, -0.064038, -0.052373, -0.070475, -0.118757, 0.044811, 0.055124, 0.140269, -0.164544, -0.009753, -0.116266, -0.094156, 0.000538, 0.172042, 0.160469, -0.10629, -0.115051, -0.082116, -0.138392, -0.057803, -0.062728, 0.002126, 0.154655, 0.134011, -0.04379, 0.034752, 0.074977, 0.182917, -0.065722, -0.044224, 0.04122, 0.16513, -0.059474, -0.041897, 0.148367, -0.213187, -0.194138, -0.110243, 0.27101, -0.12598, 0.233738, -0.022943, 0.026655, 0.058887, -0.079707, 0.104401, -0.119177, 0.30268, 0.08816, 0.075595, -0.126828, 0.038332, -0.091367, 0.008709, 0.009938, 0.123847, -0.130305, 0.162572, -0.096251, 0.072582, -0.007016], [0.115274, 0.025901, 0.085632, 0.103302, -0.114727, -0.177707, -0.106808, 0.057605, -0.190862, 0.114198, -0.093622, -0.116142, -0.001485, -0.00354, -0.132343, 0.099255, 0.13031, -0.098887, -0.061558, -0.053318, 0.059669, 0.080084, -0.099805, 0.079672, 0.152596, 0.141651, 0.20901, 0.080472, 0.016904, 0.080795, -0.032084, -0.141113, 0.023412, -0.080947, -0.008157, 0.019403, -0.055159, 0.084273, -0.247362, 0.042713, 0.047415, 0.234793, -0.156937, 0.001332, 0.102312, 0.018415, 0.209937, -0.010024, 0.291792, -0.06426, 0.03087, -0.017527, 0.069971, 0.060381, 0.015624, -0.197114, 0.064157, -0.007877, 0.062935, -0.128681, 0.069872, -0.143281, 0.114851, -0.079727], [0.122258, 0.340203, 0.164588, 0.085273, 0.043907, 0.213704, -0.075776, -0.148869, 0.398439, 0.472662, 0.210372, 0.108775, 0.249492, 0.06812, 0.097811, -0.108291, -0.105475, -0.04682, 0.275498, 0.119398, 0.080541, 0.048556, -0.08722, -0.263499, 0.10622, -0.00443, 0.113172, 0.075261, -0.198128, 0.128568, -0.141025, 0.02533, -0.072701, -0.155361, 0.092068, -0.118064, -0.128369, 0.185906, 0.079195, 0.141925, 0.096566, 0.147893, 0.136093, -0.049768, -0.214176, -0.202009, -0.201175, 0.032947, -0.059267, -0.072557, -0.061902, -0.052247, 0.154636, 0.028079, -0.150449, -0.182112, 0.0965, 0.106031, 0.037512, -0.081173, -0.065285, 0.102866, 0.340619, 0.027258], [-0.109515, -0.119048, 0.132849, -0.006291, -0.252606, -0.037859, -0.071652, 0.011389, 0.139286, 0.26413, -0.039575, 0.165542, 0.064066, -0.184593, 0.028628, 0.084708, 0.156621, -0.109704, 0.108786, 0.053146, 0.157541, 0.175841, 0.186343, 0.024578, 0.024677, 0.094795, -0.105782, 0.11183, 0.016907, -0.087772, 0.188454, -0.145918, 0.282806, 0.091109, 0.064899, 0.153017, -0.161165, 0.130387, -0.074613, 0.043041, 0.11658, 0.063814, -0.073541, 0.053486, -0.163307, 0.175986, -0.173499, 0.042448, 0.123058, -0.127126, -0.126799, 0.006747, -0.044231, -0.05034, 0.046341, -0.070047, 0.00573, 0.140952, 0.219357, -0.017431, 0.004546, -0.175695, 0.315329, 0.161801], [0.031635, -0.199766, -0.220236, 0.343698, -0.202802, -0.085251, 0.095621, -0.183789, 0.171634, 0.342748, 0.031559, 0.002134, -0.224668, 0.224793, 0.107963, 0.137836, -0.189275, 0.228099, 0.035654, -0.129382, 0.2552, -0.206669, 0.053638, 0.185415, -0.271928, 0.165666, 0.359267, 0.129881, 0.332663, -0.160302, 0.308202, 0.063173, 0.189316, 0.202675, 0.150061, 0.142397, -0.21605, 0.295654, 0.151497, -0.384449, -0.18184, 0.146453, -0.152374, -0.208424, -0.106424, 0.069699, 0.08053, -0.246384, -0.070821, -0.026326, 0.143121, -0.303095, -0.160491, -0.085478, -0.199399, -0.059613, 0.126291, -0.22427, -0.186349, 0.344716, 0.037314, 0.215317, 0.125611, -0.018642], [-0.133608, 0.000286, 0.204516, 0.119081, -0.041515, 0.202567, 0.279481, -0.00927, -0.071485, 0.11022, 0.059168, -0.154889, 0.022862, -0.0086, -0.376265, 0.110374, 0.082382, 0.023812, -0.057068, 0.026645, -0.023847, 0.090302, 0.057034, 0.13109, -0.052926, 0.052122, 0.035302, 0.15625, -0.024094, 0.20734, -0.11835, 0.137829, -0.168086, -0.169162, -0.042547, 0.287988, -0.057479, -0.052333, 0.017101, -0.025582, 0.139007, -0.079216, -0.218053, -0.088484, 0.281021, 0.048817, 0.297148, 0.136069, 0.178477, 0.001197, -0.127833, -0.043171, -0.126777, 0.102291, -0.152444, -0.300428, -0.079869, -0.179786, -0.024445, 0.209565, -0.13236, 0.000667, 0.117867, -0.126635], [-0.082572, -0.087862, 0.121985, -0.015594, 0.100257, -0.099684, 0.074475, 0.134877, -0.106816, 0.104374, 0.087414, -0.135137, -0.187001, 0.004244, 0.012345, -0.075186, -0.051258, -0.097595, 0.093926, -0.099262, -0.148438, -0.13896, -0.181278, 0.024792, -0.046374, 0.070374, -0.064088, 0.029632, -0.078347, -0.102443, -0.090434, -0.000432, 0.280369, 0.052912, -0.091966, -0.067342, -0.047243, 0.040186, -0.084959, -0.092334, 0.103538, 0.11315, -0.026285, -0.059653, 0.007192, 0.168223, -0.07524, -0.093263, -0.226588, 0.063762, -0.139221, 0.103914, 0.021796, -0.131681, -0.03177, -0.00014, 0.122636, 0.011165, -0.036499, 0.026067, 0.103889, -0.136149, 0.056428, 0.208344], [-0.142873, 0.145031, -0.167916, 0.050322, -0.292423, -0.108058, 0.131787, 0.073617, 0.122565, 0.060767, 0.185245, 0.040078, -0.336588, 0.093192, 0.093535, -0.312507, 0.059918, 0.210641, -0.223238, 0.028367, -0.329151, 0.044214, -0.059964, 0.078589, -0.081106, 0.131604, -0.131164, 0.401583, 0.120893, -0.045404, 0.038477, -0.111861, 0.05727, 0.195749, -0.061347, -0.068721, -0.025373, 0.11948, -0.06954, 0.353924, -0.084349, -0.178906, -0.100948, 0.292625, -0.037537, -0.231334, -0.024011, -0.129183, 0.01314, 0.392556, -0.125051, 0.489617, -0.025612, -0.176285, -0.20812, 0.136174, 0.102546, -0.034809, -0.20509, 0.181429, 0.024741, 0.158265, -0.003948, 0.272158], [-0.108868, -0.24951, 0.378494, 0.168489, 0.104937, -0.268517, -0.134727, -0.314383, 0.099538, 0.038219, 0.024809, 0.134301, -0.077817, -0.07119, -0.00774, 0.114859, -0.072884, -0.163185, 0.206441, 0.130704, -0.014555, -0.036875, 0.159846, -0.1476, 0.048825, 0.047009, 0.003733, -0.339216, -0.381268, -0.113476, -0.054374, 0.098517, -0.113587, -0.347709, -0.107493, -0.03313, -0.049518, 0.144177, 0.04393, 0.305583, -0.003589, -0.13955, 0.058919, 0.125223, -0.075481, -0.200746, 0.019572, 0.066784, 0.337183, 0.054364, -0.054765, 0.026161, 0.245362, -0.136101, 0.087249, 0.094469, 0.139877, 0.095207, -0.296778, -0.216602, -0.178921, -0.004624, 0.269743, 0.138226], [-0.085276, -0.055756, 0.140558, 0.105201, 0.002878, 0.412461, 0.347616, 0.050649, 0.03223, -0.103856, 0.27705, 0.010195, 0.060375, 0.070605, 0.143976, 0.144942, 0.152968, 0.214385, -0.296939, 0.16727, -0.150799, -0.040268, 0.136203, -0.114076, 0.012079, -0.164839, -0.264784, 0.065271, -0.151988, -0.198237, 0.113684, 0.083224, -0.015398, -0.101585, 0.219278, 0.128828, -0.076475, -0.103198, 0.066459, -0.02122, -0.0264, 0.09184, 0.019406, 0.107858, 0.015683, 0.085083, 0.158911, -0.025771, 0.217081, -0.011933, 0.302104, -0.232398, -0.283156, 0.094696, 0.248513, -0.017903, 0.341926, 0.083441, -0.154345, 0.044019, 0.018678, 0.268723, -0.031947, 0.014721], [-0.274678, 0.099199, -0.326991, -0.08889, -0.062353, -0.039486, -0.186598, -0.111885, -0.062824, -0.249022, -0.16882, 0.070024, 0.006228, -0.232199, -0.351416, -0.136389, -0.116755, -0.181948, -0.077986, 0.228903, 0.224797, -0.03726, -0.228419, -0.223895, -0.455861, 0.206047, 0.033382, -0.102001, 0.021733, -0.482813, 0.033865, -0.301338, -0.241319, 0.162637, 0.153879, -0.31046, -0.412474, 0.000455, 0.20519, -0.031175, -0.189377, -0.087267, 0.13723, 0.149141, -0.082516, 0.205306, 0.39313, -0.264437, 0.055726, 0.03897, 0.093735, -0.212175, -0.125371, -0.379315, -0.665402, -0.214451, 0.030606, -0.428111, -0.386474, -0.036894, -0.325407, 0.216262, 0.127041, 0.186301], [-0.065819, 0.0127, 0.093108, -0.25401, -0.037434, -0.080113, 0.06103, -0.150294, 0.180491, -0.110079, -0.118424, -0.120144, -0.165302, -0.114791, -0.181603, 0.065539, 0.043563, -0.197279, -0.032782, -0.256125, -0.04531, 0.003975, 0.002127, -0.280882, -0.038246, -0.092355, 0.071365, -0.364945, -0.406288, 0.204799, -0.031142, 0.060401, -0.061092, -0.119645, -0.065999, -0.094233, -0.079742, -0.195988, 0.245946, 0.062093, 0.081915, -0.13843, -0.081746, -0.005793, -0.137897, 0.116888, 0.065375, -0.1835, 0.298909, -0.142169, 0.179035, 0.161461, 0.051426, -0.094816, 0.035356, -0.058087, 0.319506, 0.033353, -0.021474, -0.032009, 0.179123, -0.458534, -0.077887, -0.143888], [-0.203547, -0.108008, -0.116787, -0.026198, -0.130041, -0.034795, -0.089944, -0.082383, -0.048353, -0.076581, 0.000975, 0.01902, -0.177706, -0.113122, -0.004652, 0.006738, 0.000385, 0.128256, -0.281501, 0.026891, -0.039849, 0.055939, 0.061075, -0.017908, -0.151361, 0.118041, 0.089704, -0.091808, 0.122688, 0.118439, -0.044526, -0.020264, -0.00176, 0.012956, 0.052893, -0.196636, -0.095528, -0.217746, -0.027505, 0.163829, 0.099971, -0.127169, 0.092573, -0.006346, 0.037701, -0.238905, -0.106006, -0.256766, -0.17825, 0.025951, 0.150067, 0.132605, 0.151097, -0.069543, -0.314774, 0.084142, 0.037571, -0.060164, -0.136779, -0.094047, -0.024422, -0.077407, 0.101087, 0.087871], [-0.046218, 0.096925, -0.027196, -0.035886, -0.088463, 0.189956, -0.031607, -0.127213, 0.152186, -0.363937, -0.198324, 0.135216, 0.014647, -0.237821, -0.323869, -0.250285, 0.009672, -0.354452, -0.073234, 0.129403, 0.067272, -0.072497, 0.134561, -0.083283, -0.081917, -0.174193, -0.164, -0.131865, -0.075001, 0.166791, -0.224644, 0.056978, -0.206571, -0.278579, 0.01659, -0.034262, 0.040144, -0.106952, 0.15838, -0.018753, -0.087292, -0.173194, -0.031215, 0.206305, 0.194214, -0.080324, 0.118759, 0.086568, -0.103123, 0.13356, -0.255618, 0.018309, 0.157145, 0.010664, -0.017357, -0.209043, -0.319275, -0.077232, -0.009556, 0.069945, -0.047666, -0.423257, 0.213454, -0.126686], [-0.272588, -0.178423, -0.112629, -0.532602, -0.189146, -0.273023, -0.005289, 0.129796, -0.275426, -0.645745, -0.686766, -0.176286, -0.127502, -0.154355, 0.271008, -0.037619, -0.247898, -0.655337, -0.502672, -0.413257, 0.059922, -0.277495, 0.121659, -0.564262, -0.002317, -0.186996, 0.053885, -0.433791, -0.267035, -0.072335, 0.03932, 0.140909, -0.020838, -0.453011, 0.15318, 0.180608, -0.113241, -0.44197, -0.164174, -0.047129, 0.152642, -0.228867, -0.240703, 0.373313, 0.046701, 0.292206, 0.189525, -0.006211, 0.119061, -0.439474, 0.353355, 0.413306, -0.013194, -0.05426, -0.148169, -0.199757, 0.410432, 0.135414, 0.230609, 0.204124, -0.105825, -0.39469, 0.139608, 0.020821], [-0.083249, -0.029812, -0.014365, 0.015524, 0.079018, 0.165541, 0.2702, 0.063108, -0.149776, 0.307445, 0.146397, -0.026723, 0.023001, 0.234225, 0.236877, 0.226957, -0.047415, 0.187124, 0.024951, -0.104672, 0.277855, 0.240914, -0.012431, 0.239234, 0.208177, 0.149342, 0.142213, 0.335738, -0.058024, -0.238034, 0.10277, 0.040529, 0.330564, 0.401034, 0.138443, 0.087747, 0.117154, 0.207171, -0.08251, 0.35789, 0.264843, 0.046091, 0.128657, 0.009118, -0.086676, 0.132181, -0.003941, 0.003031, -0.038486, 0.21253, 0.05603, -0.115669, -0.148367, -0.20681, 0.085599, 0.218484, 0.186719, 0.164655, 0.094932, 0.419627, 0.015105, 0.436801, 0.029954, 0.097047], [0.121512, -0.131593, -0.121834, 0.4865, -0.15122, 0.001784, 0.164792, -0.230441, -0.129222, 0.555977, 0.192945, 0.187391, 0.046607, 0.378336, 0.297435, 0.143726, 0.195931, 0.111317, 0.054109, 0.040081, -0.255983, 0.467728, -0.035757, 0.03985, 0.108072, 0.018535, -0.138871, 0.125592, 0.143479, -0.063877, -0.167211, 0.162028, 0.223101, -0.007703, 0.126928, 0.204104, -0.088013, 0.367148, -0.044102, -0.15345, 0.224648, -0.044004, 0.227728, 0.135123, 0.147889, 0.001147, 0.237806, 0.22173, -0.000475, 0.155046, -0.020683, -0.030158, -0.110854, -0.092038, 0.217294, 0.120645, -0.100475, 0.065707, 0.273385, 0.320025, -0.045633, 0.384442, -0.204577, 0.022412], [0.042674, 0.097781, -0.088022, -0.050636, 0.162165, -0.066104, -0.024601, 0.103168, -0.050074, 0.095286, 0.004484, 0.057679, 0.024107, 0.039239, -0.010962, 0.201362, 0.162315, 0.042155, 0.242977, -0.015374, 0.234541, 0.001294, -0.023164, -0.078254, 0.124959, 0.052754, 0.013986, -0.065524, -0.191462, -0.174715, 0.204551, -0.04335, -0.044885, -0.219396, -0.066399, 0.135575, -0.119357, 0.197266, -0.084991, 0.282154, -0.063929, -0.003628, -0.117634, 0.110622, 0.060276, 0.162157, -0.096575, 0.264965, 0.419654, -0.03846, 0.072676, 0.090651, 0.086241, 0.041097, -0.125436, -0.086946, 0.026723, 0.06622, 0.206324, 0.120829, -0.126085, -0.07574, 0.0777, -0.161831], [0.111702, 0.03346, 0.219266, -0.007121, -0.061222, -0.283798, 0.131083, -0.019607, 0.033964, -0.028605, -0.056821, 0.033888, 0.008293, 0.168956, -0.008937, 0.033324, 0.135344, 0.207756, -0.048585, -0.045264, -0.331321, -0.06476, 0.430979, -0.038476, 0.079943, -0.006376, 0.116094, 0.139278, 0.173186, 0.089138, 0.206048, -0.068932, -0.030018, -0.132756, 0.063433, 0.341171, 0.144565, 0.022462, -0.062524, -0.133901, 0.194539, 0.015324, 0.077078, 0.086043, -0.091039, 0.007956, 0.010265, 0.255539, 0.2121, -0.085487, 0.1284, 0.124673, -0.180057, 0.219066, 0.147573, 0.095792, -0.170047, 0.056781, 0.111247, 0.226932, -0.044798, -0.098702, -0.067017, -0.135701], [0.117132, 0.03179, -0.074027, 0.228122, 0.149298, 0.074485, -0.081545, -0.181102, -0.021204, 0.339028, 0.344173, -0.111673, 0.004589, 0.389372, 0.170541, 0.236404, -0.128919, -0.108014, 0.093001, -0.000156, -0.244488, 0.130213, -0.256461, 0.386895, 0.036083, -0.125377, -0.103329, -0.379743, -0.175361, -0.064971, -0.001639, 0.135659, 0.028957, -0.298716, 0.165759, 0.099618, 0.045683, -0.067953, -0.318369, -0.232753, 0.051525, 0.262637, 0.016883, -0.107904, -0.032663, -0.280847, -0.004728, 0.237935, 0.266091, -0.271432, -0.061038, -0.141786, -0.212249, 0.220255, 0.101028, 0.205112, 0.129992, 0.119095, 0.143435, -0.247749, 0.286867, 0.055323, -0.065946, -0.126779], [-0.274421, 0.355242, -0.28513, 0.136869, 0.113123, -0.067318, 0.110237, -0.148175, -0.112373, -0.081461, -0.06169, 0.115556, -0.066431, -0.177299, -0.103946, -0.035887, -0.067306, 0.015769, -0.235046, 0.119571, -0.013868, 0.151865, -0.024298, 0.084994, -0.114169, 0.178008, 0.033908, 0.630209, -0.260505, -0.165829, 0.194395, -0.225088, 0.183622, 0.492039, 0.061308, -0.105195, -0.189093, 0.233494, 0.087148, 0.477198, -0.027705, 0.013949, 0.162051, -0.008445, 0.027427, 0.297122, -0.254477, -0.210302, -0.019712, 0.089655, 0.113111, 0.00097, -0.532745, -0.140549, -0.061021, -0.097014, -0.124089, -0.22575, -0.108598, 0.316288, -0.157502, 0.404947, 0.091446, 0.062282], [0.111851, 0.170052, 0.087855, 0.288066, -0.079262, 0.117005, 0.068516, -0.049616, 0.235508, 0.001956, 0.117011, -0.074095, 0.044184, -0.068975, 0.136861, -0.058901, 0.324737, 0.318222, 0.14935, -0.116407, -0.025195, 0.190912, -0.077754, 0.133374, 0.227389, -0.112072, -0.110159, -0.148792, -0.099256, -0.053349, 0.100736, 0.225842, -0.295785, -0.152684, 0.218493, 0.250587, -0.046221, 0.195993, 0.123371, 0.027265, 0.006913, -0.019961, 0.013083, -0.135061, -0.167689, -0.20935, -0.00318, 0.133898, -0.143305, -0.03834, -0.155013, 0.095373, 0.146156, 0.071429, 0.227127, 0.259991, -0.055479, 0.004512, 0.085699, 0.106554, 0.057925, 0.086239, 0.050162, 0.060556], [0.059253, -0.079108, 0.430421, -0.126222, -0.120665, 0.026976, 0.023526, -0.071462, -0.022894, 0.00975, -0.205997, 0.142322, 0.056421, 0.184598, 0.366621, 0.138524, 0.181307, 0.073121, 0.403043, 0.269092, 0.098609, 0.151465, 0.347484, -0.173001, 0.14143, -0.187708, -0.128253, 0.267415, -0.066271, 0.268316, 0.013211, 0.205323, -0.025988, -0.031231, 0.00559, 0.917903, 0.113396, -0.176324, 0.235359, 0.277857, -0.060296, 0.140125, 0.106481, 0.284489, 0.418545, 0.108067, -0.163494, 0.341586, -0.072924, 0.053904, 0.016804, 0.259024, -0.19695, 0.179837, 0.787991, -0.033568, -0.129806, 0.347192, 0.233077, 0.69498, -0.177355, -0.021166, 0.039531, -0.184288], [0.231081, -0.226511, 0.135734, -0.281673, -0.077476, 0.01162, -0.238783, 0.023616, 0.221962, -0.135212, -0.007966, 0.082987, -0.17623, -0.362955, -0.024298, -0.035157, 0.297133, 0.033098, 0.101336, -0.007897, -0.249922, 0.084441, -0.002225, -0.077513, -0.19988, -0.097616, -0.104024, -0.013174, 0.198074, -0.078656, -0.012016, 0.061976, -0.262061, -0.127829, -0.030396, 0.127464, 0.059308, -0.350219, 0.183332, 0.094919, 0.110505, -0.05638, -0.311052, -0.121112, 0.029882, -0.175846, 0.122762, 0.024126, -0.229813, 0.104419, -0.161881, -0.049078, 0.093829, 0.080482, 0.160173, 0.053427, -0.342013, -0.053429, -0.079619, -0.120065, 0.034763, -0.191428, 0.069243, -0.086126], [0.130311, -0.086726, -0.085231, 0.187438, -0.003025, 0.003823, -0.310312, -0.104078, 0.058178, 0.167732, 0.143937, 0.020044, -0.138047, -0.109721, -0.246412, -0.003169, -0.3798, -0.018987, -0.143131, -0.331454, -0.241048, -0.112977, 0.037883, 0.100533, -0.037702, 0.086861, -0.157781, -0.572989, -0.042869, -0.025962, -0.019846, -0.025492, -0.088405, -0.284053, 0.123366, -0.198964, -0.160028, 0.033585, 0.01003, 0.092812, -0.025918, 0.177585, -0.15483, -0.166247, -0.134332, 0.008019, 0.130493, -0.2736, 0.213936, -0.002575, -0.203211, -0.203207, -0.000156, -0.140177, -0.265485, -0.031058, -0.077981, -0.175523, -0.108644, -0.727354, 0.108079, 0.142226, -0.098093, -0.055521], [-0.130907, 0.006857, -0.297951, -0.006051, -0.003142, 0.021882, 0.210728, 0.204509, -0.14083, -0.066187, 0.083719, 0.040229, -0.012147, 0.112079, 0.026086, -0.100673, 0.02165, -0.082763, -0.31108, 0.012487, 0.426054, 0.231152, -0.422177, -0.145222, -0.027045, -0.140602, 0.236, -0.226357, 0.260255, -0.147246, -0.119068, -0.235521, -0.006273, -0.171194, 0.033178, -0.225706, -0.313551, -0.092786, -0.114309, -0.126092, 0.207748, -0.200025, 0.014778, -0.097961, -0.062076, -0.185567, 0.145424, -0.043202, 0.10594, 0.037502, 0.318699, 0.168303, 0.075499, -0.067451, -0.445401, -0.02862, 0.214142, -0.360511, -0.063484, 0.311936, 0.031616, 0.187161, 0.141262, 0.043413]], [0.145582, -0.081725, 0.057718, 0.159443, 0.083695, -0.12869, -0.286361, -0.09828, 0.20464, -0.040023, -0.080242, 0.038669, -0.10271, -0.279792, -0.183475, -0.349613, -0.01582, -0.087424, -0.117228, -0.099925, -0.449254, -0.150066, 0.112732, -0.051236, -0.180949, 0.057457, -0.212516, -0.241926, -0.051445, 0.135674, -0.14307, 0.061408, -0.076533, -0.065513, 0.137381, -0.16615, -0.108765, 0.066403, -0.039876, -0.280887, -0.138082, 0.04549, -0.058874, -0.200493, -0.117117, -0.27678, 0.235616, -0.054211, -0.231572, -0.093404, -0.026998, -0.076257, 0.203309, 0.1559, -0.120781, -0.168721, -0.010432, -0.095236, -0.10741, -0.446737, 0.06987, -0.216091, 0.03702, 0.060188]),
    ([[0.155743, -0.01628, -0.005125, 0.211667, -0.062391, -0.337185, 0.075372, -0.131336, 0.026193, -0.079574, 0.0916, -0.095133, -0.163184, 0.204784, -0.283897, -0.273824, 0.050452, -0.070368, -0.028582, 0.081515, -0.075953, -0.006159, 0.016217, 0.117407, 0.147851, -0.208069, 0.083809, 0.063572, -0.102447, 0.140381, -0.081736, 0.082666], [-0.074853, -0.038456, -0.017363, 0.001192, 0.160537, -0.023605, 0.021623, -0.245642, -0.006524, 0.122048, 0.034214, -0.163578, -0.075666, 0.069164, 0.249264, -0.04568, -0.086669, -0.010687, 0.058022, -0.011171, -0.272432, 0.025804, 0.022615, 0.025749, 0.140265, -0.323868, 0.17963, -0.043477, 0.249377, -0.072382, -0.134254, -0.051835], [0.166997, 0.249109, 0.09186, 0.176366, -0.002979, 0.128518, -0.096435, 0.145401, 0.245458, 0.081552, 0.068822, -0.063965, -0.235962, 0.091292, -0.119404, 0.098118, 0.143375, -0.007192, 0.128231, 0.395689, 0.104818, 0.168887, 0.006343, 0.039527, 0.068715, -0.199621, 0.070641, 0.073775, -0.065431, -0.005831, 0.244454, 0.302356], [0.088092, 0.00994, 0.011744, -0.089233, 0.020566, -0.007215, 0.07982, 0.06995, -0.020368, -0.087116, -0.150377, 0.046086, -0.326132, 0.042507, -0.022804, 0.144476, 0.10686, 0.017852, 0.175481, 0.086208, -0.022079, 0.217401, 0.082214, -0.122459, -0.071349, 0.124714, 0.02243, 0.086103, -0.221346, 0.101192, -0.204711, 0.409354], [0.100923, 0.071209, 0.126509, -0.39909, -0.217634, 0.151635, -0.004055, 0.23442, 0.062931, 0.016791, 0.268055, 0.082294, -0.020137, 0.125669, 0.023894, 0.047186, 0.179064, 0.079478, 0.161654, 0.072785, -0.013239, -0.100452, 0.010403, 0.117151, -0.035931, 0.056454, -0.167418, 0.062, -0.700783, 0.055578, -0.077678, -0.013888], [0.223221, -0.074717, -0.033693, -0.137929, 0.015313, 0.240111, -0.271517, 0.032598, 0.033406, -0.094287, 0.011065, -0.086176, 0.236267, 0.027387, -0.010558, 0.181885, 0.117942, -0.041726, 0.078953, -0.135686, 0.101545, -0.016614, -0.012579, -0.202695, -0.002561, 0.005283, -0.124345, -0.077511, -0.018108, 0.065812, 0.146385, 0.059568], [-0.152143, 0.034697, 0.036872, 0.084794, 0.01633, -0.243793, -0.03241, -0.027217, 0.058238, 0.043609, 0.060312, 0.089789, -0.060462, 0.021832, 0.039191, 0.028754, 0.003171, -0.101937, -0.127904, 0.079466, 0.112399, 0.026226, 0.141207, 0.240918, 0.115778, 0.163489, -0.003244, -0.01051, -0.156802, -0.002173, 0.090172, 0.034047], [-0.054214, -0.168205, 0.164846, -0.039062, -0.099494, -0.113992, 0.009105, -0.014423, -0.118888, -0.056709, 0.148566, 0.026327, 0.187883, -0.040294, 0.067611, -0.059221, -0.020088, 0.018493, -0.196544, -0.207501, -0.082895, -0.038367, -0.004095, 0.095781, 0.020092, -0.234393, -0.004894, -0.046547, 0.308467, 0.002603, 0.086776, -0.116008], [0.059751, -0.202088, -0.198082, -0.07012, -0.018564, -0.104853, -0.29696, -0.091212, -0.136381, 0.034735, -0.310957, 0.122944, -0.097534, 0.031694, 0.091051, 0.174129, 0.190366, -0.037663, 0.073409, -0.288557, -0.002848, 0.062527, 0.070804, -0.284893, -0.104295, 0.067852, 0.031709, 0.146173, 0.016336, -0.066593, -0.357822, -0.007306], [0.153937, -0.060511, 0.029936, -0.255041, -0.078673, 0.064618, 0.136176, 0.15256, -0.362591, -0.018086, 0.110904, -0.125917, -0.20931, -0.078081, -0.281991, 0.033009, -0.439458, 0.173461, 0.022854, 0.019181, -0.056764, 0.030591, 0.166826, -0.1192, -0.312671, 0.058414, 0.035257, 0.178895, -0.718335, -0.061277, -0.11492, -0.276873], [0.115856, -0.167496, -0.000438, -0.116313, 0.154097, 0.096387, 0.027963, 0.219211, -0.137834, 0.046955, -0.006765, 0.01259, -0.029902, -0.018264, 0.028477, 0.032714, 0.102135, 0.071151, 0.048297, -0.031062, 0.07533, -0.005149, 0.134718, -0.06532, -0.051165, 0.113957, -0.006987, -0.030769, -0.24694, 0.134587, -0.271014, -0.152709], [0.126094, -0.14921, -0.1108, 0.164103, -0.038433, 0.082079, 0.122338, -0.030835, -0.174778, -0.043063, 0.074055, -0.075729, 0.062195, -0.062134, 0.075398, 0.122657, -0.233549, 0.079081, -0.008725, 0.024631, -0.058084, 0.047145, 0.088586, 0.130719, 0.027807, 0.016081, 0.125347, 0.069212, -0.228505, -0.011957, 0.081823, -0.257998], [-0.11777, -0.056812, 0.16725, -0.168546, -0.004042, -0.09097, 0.007986, -0.106169, 0.026654, 0.024312, 0.169939, 0.034562, 0.10887, -0.035194, 0.060414, 0.026349, -0.080301, 0.076824, -0.01983, 0.038239, -0.019972, -0.014882, -0.035193, 0.064417, 0.070394, 0.132621, -0.060409, 0.110389, -0.278858, 0.010657, 0.091127, 0.05214], [0.121974, 0.16454, -0.076621, -0.062823, -0.195634, 0.161168, 0.033863, -0.013664, -0.114962, -0.092752, 0.100034, 0.00768, 0.031197, -0.119365, -0.07639, -0.006198, -0.167201, 0.044125, 0.178988, 0.017914, 0.012835, -0.2358, 0.114041, 0.090425, -0.116108, -0.015518, 0.071435, -0.217885, 0.226416, -0.295989, 0.033121, 0.091756], [-0.092937, -0.042873, -0.171153, -0.017183, -0.09387, 0.158842, 0.090719, 0.135302, 0.113498, -0.122545, 0.019413, 0.094791, 0.245528, -0.016972, -0.159269, -0.033298, 0.163858, 0.033708, -0.337677, 0.09801, 0.121841, 0.079083, 0.05771, -0.161255, 0.143258, -0.037127, 0.07033, 0.021345, -0.028871, -0.139558, -0.007185, 0.057209], [0.209186, -0.167437, -0.013506, -0.060511, -0.168529, 0.113062, 0.022285, 0.071772, -0.1759, -0.104954, -0.208335, -0.088634, 0.166595, 0.030949, -0.028022, 0.188708, 0.055742, 0.061475, 0.149196, 0.143141, 0.014958, -0.252245, 0.178702, -0.010111, -0.201403, 0.190112, -0.154571, -0.064442, 0.253069, -0.01748, -0.424629, -0.017784], [-0.043426, 0.085351, -0.014479, 0.072636, 0.036235, -0.01927, 0.052096, -0.017869, -0.041659, -0.152533, -0.009605, -0.039766, 0.136932, -0.18998, -0.264686, 0.108533, -0.145435, -0.116795, 0.053967, -0.01152, -0.076635, -0.350367, 0.11282, -0.058113, -0.142181, 0.164661, 0.148906, -0.166746, 0.144427, -0.282071, 0.116955, -0.226725], [0.076813, -0.108396, -0.2458, -0.0225, 0.167744, 0.145745, -0.220271, 0.059152, -0.169684, -0.097162, -0.331041, -0.044859, 0.109725, 0.038284, -0.155096, 0.07167, 0.198193, -0.028373, 0.132057, -0.278748, -0.122114, -0.061777, 0.092139, -0.177955, -0.166524, 0.063363, 0.121383, 0.011149, 0.233707, -0.1443, -0.322819, 0.053961], [0.023187, 0.063244, 0.077164, 0.012562, 0.004623, 0.051488, 0.036944, -0.092175, -0.051535, 0.049717, -0.035007, -0.071721, -0.40132, -0.054128, -0.006093, 0.194827, 0.020629, 0.032058, -0.00556, 0.319097, 0.179762, 0.316164, -0.019588, -0.052101, 0.212331, 0.07503, 0.009147, 0.082125, -0.052748, 0.160516, 0.071598, 0.084781], [0.013878, -0.092867, 0.032451, 0.090241, 0.212823, 0.212307, -0.086435, -0.106456, 0.194732, 0.119999, -0.180845, 0.00779, 0.114756, 0.020055, 0.034767, 0.066562, -0.024571, -0.104436, 0.15919, 0.10282, 0.108866, -0.017964, 0.125037, 0.141908, 0.279032, 0.011764, -0.054251, 0.152163, -0.201143, -0.252235, -0.010401, 0.099992], [0.003301, -0.174756, 0.075674, 0.059814, 0.157357, -0.037965, -0.0596, -0.068126, -0.376718, -0.155757, -0.063343, 0.084548, 0.048679, -0.143174, -0.296922, -0.011357, -0.179301, -0.145397, -0.080621, -0.30727, -0.521342, -0.328296, -0.229452, 0.090213, -0.265181, 0.049401, 0.049561, 0.134639, 0.058038, 0.344934, -0.088606, -0.408469], [0.126503, -0.014709, -0.071622, -0.101185, -0.187837, -0.068333, -0.029005, -0.014002, -0.172711, -0.197583, 0.068768, -0.188248, 0.085691, -0.150981, -0.172406, 0.03594, -0.175511, 0.074207, 0.078109, -0.017397, -0.094155, 0.051812, 0.233162, -0.023809, -0.160121, 0.061259, -0.021258, -0.212394, -0.042943, 0.276284, 0.126029, -0.168659], [-0.11086, 0.064389, 0.147835, 0.004734, 0.177118, -0.071206, 0.033426, -0.016826, 0.016287, 0.087979, 0.075695, 0.226951, -0.075243, 0.080796, 0.170432, -0.025302, 0.056891, -0.077456, 0.001878, 0.003181, -0.037956, 0.297707, 0.015883, 0.02148, 0.205766, -0.017366, 0.17085, 0.293863, -0.209397, -0.103224, 0.244505, 0.16422], [-0.051931, 0.057647, 0.005783, -0.071766, -0.377252, 0.157951, 0.136723, 0.078915, 0.004648, -0.168866, 0.068733, -0.016744, 0.048101, -0.022611, -0.017609, 0.12288, 0.049904, 0.156362, 0.132057, -0.439606, 0.039168, -0.115325, 0.082193, 0.011775, -0.112929, 0.09211, -0.196026, -0.25265, -0.231869, 0.084057, -0.074692, -0.328328], [-0.09383, -0.002619, 0.073672, 0.000875, -0.059513, 0.001617, 6.2e-05, -0.123516, 0.047121, 0.096408, 0.173313, -0.099383, 0.096898, -0.073649, 0.184544, -0.154597, -0.011538, -0.036157, -0.053189, 0.154076, 0.01777, 0.104531, 0.092904, 0.125105, 0.011224, 0.229809, -0.057866, 0.050934, -0.24257, -0.294417, 0.076893, 0.132784], [-0.014414, -0.059452, -0.228348, 0.045224, 0.125209, 0.132659, -0.067383, 0.110141, -0.095258, -0.076671, 0.072562, -0.034834, -0.151017, -0.201959, 0.029193, 0.1598, -0.146523, 0.06205, -0.010869, -0.070638, -0.026507, -0.149716, 0.064373, -0.000218, -0.071942, 0.110767, 0.113027, -0.031783, -0.011576, -0.218181, 0.117814, 0.037289], [-0.217358, -0.229509, -0.163018, -0.246592, 0.106454, 0.076709, -0.104084, 0.013819, -0.14306, 0.009548, -0.194411, -0.243275, 0.067449, -0.008793, -0.530972, -0.110967, -0.208204, -0.065545, 0.05001, -0.212425, -0.037087, -0.129524, -0.227672, -0.343075, -0.115737, 0.014346, 0.004596, -0.014691, -0.145491, 0.174965, 0.020569, -0.160971], [0.037676, -0.009625, 0.133138, 0.206403, -0.076833, -0.003989, 0.088062, 0.222469, 0.353969, 0.033036, -0.02161, -0.127349, 0.13938, 0.153134, 0.013731, 0.182226, 0.374382, 0.148, 0.160031, -0.399039, -0.05738, 0.155411, 0.123355, 0.215525, 0.195054, 0.094582, -0.018091, 0.088081, 0.628742, -0.196224, -0.081212, 0.148062], [0.291307, 0.090311, -0.261671, -0.230076, 0.112148, 0.078282, 0.014691, 0.078495, 0.134326, 0.046143, -0.017033, 0.040625, 0.0787, -0.13177, -0.261931, -0.006444, 0.022797, 0.102019, 0.093746, -0.034156, -0.021381, 0.008011, 0.237542, -0.239089, 0.138789, -0.007169, -0.036996, 0.088546, 0.225538, 0.104093, 0.028047, 0.1298], [0.074622, -0.09665, 0.131293, -0.000893, -0.089994, -0.274704, -0.089433, 0.09709, 0.112456, 0.107268, 0.133593, 0.011021, -0.049419, 0.078515, -0.016941, -0.02516, -0.030827, -0.119628, 0.004935, -0.158282, -0.075093, 0.086948, -0.172219, -0.05552, 0.098732, -0.153055, 0.097701, 0.100681, -0.005182, -0.058209, -0.10828, -0.133834], [0.207632, 0.299896, 0.275972, -0.244988, -0.256065, 0.210908, 0.023637, -0.052582, 0.309917, 0.032322, 0.294236, 0.118621, -0.157735, -0.072462, -0.086628, -0.005989, 0.200314, 0.224131, 0.190731, 0.232836, -0.066054, 0.099914, 0.040437, 0.189926, 0.107764, -0.057139, -0.000251, 0.111815, -0.182627, 0.126666, 0.468486, 0.283063], [-0.154262, 0.152626, 0.050135, -0.003002, 0.072424, 0.147777, 0.094787, 0.014545, 0.113624, 0.032441, -0.047463, 0.087549, -0.138589, 0.045605, -0.110081, 0.065858, -0.075458, 0.125806, -0.016742, 0.027424, 0.094762, 0.155778, -0.162252, -0.108552, 0.03015, -0.029711, 0.105013, 0.165075, -0.133236, -0.090497, -0.032973, 0.020726], [0.10509, 0.118071, -0.055897, -0.147112, -0.126095, 0.288927, -0.007308, 0.091331, 0.0587, -0.071887, 0.101151, 0.111033, 0.061456, 0.116099, 0.221901, 0.077267, 0.056125, 0.312628, 0.212972, 0.112985, -0.05338, 0.1243, 0.028866, -0.130322, 0.011812, 0.021442, 0.01634, -0.09287, -0.156646, 0.009685, 0.077377, -0.043133], [0.146643, 0.02062, -0.232309, -0.075673, -0.157445, 0.304179, -0.041733, 0.294376, 0.198264, -0.043374, -0.144953, 0.000795, 0.106723, -0.045262, -0.039009, 0.198389, 0.15394, 0.368772, 0.323652, 0.193, 0.126417, 0.171307, 0.413789, -0.088872, 0.166158, 0.303618, -0.017463, -0.091307, 0.265856, 0.202361, -0.132471, 0.21838], [-0.011549, 0.015473, 0.059284, 0.046733, -0.029553, -0.339346, 0.024028, -0.221902, -0.010916, -0.015497, -0.070352, 0.048068, -0.048509, 0.025291, -0.127101, 0.060915, 0.045166, -0.380364, -0.133888, -0.016482, -0.002625, -0.086633, 0.021299, -0.055938, -0.003246, -0.070925, -0.106776, -0.057568, 0.083768, 0.035608, 0.0749, 0.14962], [-0.064038, 0.141451, 0.090881, 0.002235, -0.052553, 0.030261, 0.179616, 0.01098, 0.224498, 0.123403, 0.030442, -0.152275, 0.13103, 0.172243, -0.215094, 0.031003, 0.287614, 0.046612, -0.016911, 0.126302, 0.068854, 0.298301, 0.010839, 0.112435, 0.19373, -0.153832, 0.011316, 0.030114, -0.26058, -0.070161, 0.146368, 0.163432], [-0.112976, -0.067977, 0.060268, 0.090481, 0.041116, -0.186063, 0.10791, 0.015366, 0.116936, 0.031427, 0.113334, 0.06057, -0.176436, 0.021237, 0.29957, 0.099743, -0.097364, -0.206188, -0.021584, 0.063831, -0.088886, 0.107981, -0.117269, -0.022656, 0.00023, -0.011279, -0.081099, 0.065692, -0.294159, -0.101344, 0.092104, -0.097013], [0.020146, -0.149655, -0.17935, 0.034812, -0.023775, 0.046759, -0.162962, 0.145595, -0.189823, 0.100672, 0.056311, 0.033879, -0.475778, -0.015228, 0.168497, 0.13993, -0.318922, 0.080616, 0.01691, 0.040385, -0.066033, -0.059692, 0.034056, -0.096972, -0.096042, 0.051013, 0.001279, -0.080502, -0.212275, -0.060106, -0.005465, -0.161429], [-0.037493, 0.033792, 0.00246, 0.123498, 0.094366, -0.152911, 0.117079, -0.177802, -0.004044, -0.053419, -0.13112, -0.040229, -0.11393, 0.021069, -0.048161, 0.021867, 0.007285, -0.20078, -0.471306, -0.182027, 0.047586, 0.06275, -0.222645, 0.014228, -0.069167, 0.17072, -0.060225, -0.060048, 0.073059, -0.023547, -0.025582, 0.030259], [0.221593, -0.135339, 0.040181, 0.130282, 0.013727, 0.182431, -0.258766, 0.145609, -0.070931, -0.08148, -0.15185, -0.108498, -0.0874, -0.1952, 0.326656, 0.107253, -0.171868, 0.044925, 0.112614, -0.101173, -0.103882, -0.335865, 0.072106, 0.148243, -0.245468, 0.053435, -0.03741, -0.16891, -0.039424, -0.011078, 0.329802, 0.061975], [0.105917, 0.042685, -0.039843, -0.117804, 0.071423, -0.559588, -0.055996, -0.120919, 0.107974, 0.008692, -0.024186, -0.107707, -0.059496, -0.105602, -0.32177, -0.109031, -0.073123, -0.209533, -0.533585, 0.111563, 0.113116, 0.021314, -0.05443, 0.148623, -0.051284, 0.120744, 0.024006, 0.128144, -0.570091, 0.15348, 0.033281, 0.142758], [0.167295, 0.076126, 0.04205, 0.249451, -0.104941, -0.183048, 0.055815, -0.273055, -0.089659, -0.101384, -0.047507, -0.145601, -0.272359, 0.019903, -0.415778, -0.163578, -0.009223, -0.189502, -0.008096, -0.007632, 0.089638, 0.001767, -0.116844, -0.044131, -0.087489, -0.319804, 0.066492, 0.124763, 0.081319, 0.07106, -0.071525, 0.014814], [-0.003612, 0.03394, 0.096499, -0.122423, 0.058212, 0.165096, 0.087682, 0.133277, -0.039265, -0.019239, -0.008491, -0.045835, -0.061305, 0.089296, -0.059098, 0.063205, -0.025675, 0.113173, 0.048811, 0.289547, -0.023508, -0.007364, 0.105611, -0.014865, 0.05446, 0.148603, -0.062915, -0.017245, -0.295432, -0.271937, -0.006624, -0.249131], [-0.020093, 0.12452, -0.026885, 0.147399, 0.129505, -0.102054, -0.08208, -0.06537, 0.079243, 0.030988, -0.165515, -0.121328, 0.215383, 0.006448, 0.393263, 0.073974, -0.17468, -0.383367, -0.656426, -0.126978, -0.07916, -0.039559, 0.17284, 0.039024, 0.02637, -0.077207, 0.055063, 0.056488, 0.373585, -0.057776, 0.012976, 0.089763], [0.086433, 0.081179, -0.018151, 0.020757, -0.027001, -0.221052, 0.127279, 0.079349, 0.075849, -0.054843, -0.095527, -0.020247, -0.280757, -0.054802, -0.010989, 0.036028, 0.045431, -0.244592, -0.075307, 0.000999, -0.03397, 0.102575, -0.166232, -0.060888, 0.060705, -0.139289, -0.028142, 0.077542, -0.040484, 0.112021, 0.041132, -0.084375], [0.243947, -0.168873, 0.072434, 0.10751, -0.106413, 0.015862, -0.02879, 0.032351, -0.216895, -0.10399, -0.096608, -0.218511, 0.311228, 0.001478, 0.006822, 0.209473, 0.075825, 0.055598, -0.106797, -0.351364, -0.112433, -0.07085, -0.123429, -0.089433, -0.110735, -0.017164, 0.004475, -0.13584, 0.142225, 0.354171, -0.28879, -0.086083], [-0.115605, -0.007131, 0.022849, -0.049908, -0.072011, 0.075105, 0.112099, 0.112995, -0.07905, 0.070046, -0.036375, 0.072043, 0.069854, -0.011326, -0.370349, 0.008483, -0.240143, -0.00842, -0.090898, -0.274955, -0.382519, 0.005274, -0.003959, -0.012482, -0.231243, 0.017875, -0.059608, 0.084483, -0.161207, -0.008098, -0.040408, -0.577402], [-0.170272, 0.014874, 0.025583, 0.217121, 0.019169, -0.489222, 0.015238, 0.008828, -0.023493, 0.150689, 0.145688, -0.076661, -0.214108, 0.169882, 0.213938, -0.202646, 0.085225, -0.095883, -0.243144, 0.161195, 0.066331, 0.003795, -0.031813, 0.087056, 0.153838, 0.143622, 0.127874, 0.065146, -0.079796, 0.203675, 0.172294, 0.219061], [-0.22299, -0.106002, -0.238144, -0.150162, 0.17358, 0.028958, 0.037907, -0.130209, -0.01719, -0.082962, -0.109386, 0.16203, -0.121777, 0.018063, -0.173157, -0.075357, 0.061622, -0.005591, 0.218699, -0.010151, 0.150664, -0.054687, -0.011093, -0.250693, -0.000367, -0.224723, -0.16142, -0.062884, -0.057172, 0.08051, -0.113516, 0.058395], [0.014772, 0.149157, 0.083595, 0.073089, -0.002962, 0.077362, 0.133544, 0.039574, -0.069636, -0.036204, 0.036808, 0.020991, -0.16547, -0.181317, 0.0372, 0.154824, 0.042448, 0.092995, -0.035056, -0.500616, -0.146621, 0.020553, 0.14214, 0.124811, 0.030964, 0.071932, -0.018699, 0.006046, -0.03445, -0.150795, 0.260752, -0.097842], [-0.065737, -0.078327, 0.009369, -0.282962, -0.125409, 0.188036, -0.011223, 0.264423, -0.038863, -0.135462, -0.102941, -0.18502, 0.106581, -0.032175, -0.404383, 0.013922, -0.231233, 0.161258, 0.100815, 0.088501, -0.308543, -0.084876, -0.063877, -0.066811, 0.039222, 0.121575, -0.087899, 0.009131, 0.168569, -0.06266, -0.148033, -0.089776], [-0.079001, -0.24921, 0.008023, 0.082518, 0.140025, -0.047014, -0.080058, -0.01264, -0.25773, 0.189782, 0.007414, -0.119471, -0.271901, -0.060325, 0.168896, -0.086405, -0.184491, -0.19095, 0.023099, -0.22608, -0.178378, -0.163184, -0.051627, 0.053774, -0.23508, 0.104101, 0.02942, -0.059246, -0.230593, 0.07194, 0.078993, -0.396202], [-0.251731, 0.006868, 0.194992, -0.243897, -0.113006, -0.234392, 0.076504, -0.570883, 0.017414, 0.001907, 0.0777, -0.227193, -0.129882, 0.109434, -0.210588, -0.041449, -0.267924, -0.256081, -0.425925, -0.042992, -0.001817, 0.107828, 0.096649, -7e-06, 0.019172, 0.142984, -0.004768, 0.129809, -0.02433, 0.05811, -0.183748, 0.059241], [0.07318, -0.05833, -0.024393, -0.035865, 0.016248, -0.244907, 0.03726, 0.011003, 0.095577, 0.1359, 0.027547, 0.017031, -0.092383, 0.160485, -0.215848, -0.143524, 0.041098, -0.225543, -0.159791, 0.089179, 0.046269, 0.052653, 0.100254, 0.024607, 0.053684, -0.176199, 0.121796, 0.149999, 0.039442, -0.036632, 0.049821, 0.057186], [0.059617, 0.18046, 0.093238, 0.19523, 0.132693, 0.210378, 0.109333, 0.199841, 0.054701, 0.1132, 0.253744, 0.048413, 0.309146, 0.099246, -0.31732, 0.016532, 0.326099, 0.248829, 0.222597, 0.425592, 0.350099, 0.209017, 0.009354, 0.057895, 0.146811, 0.131774, 0.016543, 0.160213, -0.142355, -0.029985, 0.198379, 0.446023], [0.011214, 0.098558, -0.202444, -0.00961, -0.037741, 0.051486, 0.054143, 0.100562, 0.123672, 0.052604, 0.119931, 0.088437, 0.244583, 0.09491, -0.120078, 0.135663, 0.10574, 0.125514, 0.139735, -0.037472, -0.050252, 0.099168, 0.119449, -0.208827, -0.095597, 0.067873, 0.084147, 0.003607, -0.121479, -0.330495, -0.052071, 0.058065], [-0.280266, -0.027927, -0.213454, -0.079338, 0.119405, -0.510435, 0.081929, -0.212846, -0.056155, 0.115674, 0.046058, 0.119932, -0.03003, -0.10431, -0.376499, -0.23731, 0.12275, -0.37806, -0.228861, 0.050095, 0.093292, -0.078512, -0.160389, 0.137898, 0.054931, -0.017411, 0.03062, 0.121522, -0.139219, 0.174229, -0.286488, 0.162143], [0.090182, 0.172449, 0.033077, 0.254342, 0.117818, 0.061006, 0.159625, 0.003584, 0.069321, -0.005987, 0.181478, 0.081645, 0.117785, 0.11369, -0.127399, 0.014996, 0.10745, 0.014495, -0.079014, 0.028811, 0.137219, 0.119322, 0.027273, 0.025287, 0.061415, 0.157741, -0.003545, 0.079562, -0.113651, -0.149232, 0.224693, 0.111719], [-0.281974, -0.027198, -0.020785, -0.231707, -0.049631, -0.447312, 0.159138, -0.453444, -0.117397, -0.029344, 0.181322, -0.084686, -0.38363, 0.07142, -0.099611, 0.017466, -0.075878, -0.340846, -0.323322, 0.115689, 0.163357, -0.015124, -0.072197, 0.110103, -0.083461, -0.116066, -0.004018, 0.102815, -0.50712, 0.099172, 0.208232, 0.005551], [-0.083703, 0.334531, 0.170215, 0.332582, -0.043309, 0.08402, 0.106677, -0.147118, 0.365636, 0.035778, 0.021385, 0.012305, 0.221495, 0.155564, 0.158134, 0.004427, 0.248369, 0.04982, 0.048321, -0.229209, -0.046795, 0.191002, -0.023105, 0.222961, 0.343958, 0.065575, 0.144494, 0.026743, 0.06851, -0.065013, 0.248775, 0.189499], [-0.044294, 0.020909, -0.00574, -0.118891, 0.035984, -0.528466, -0.029185, -0.430746, 0.008251, -0.040129, 0.042254, 0.038763, -0.40167, -0.033915, -0.354881, -0.137588, 0.023158, -0.187065, -0.017483, 0.02927, 0.101503, 0.070229, -0.111542, -0.007968, 0.016731, 0.146648, 0.028507, 0.135405, -0.32068, 0.004361, -0.000906, -0.086628], [-0.029062, 0.068902, -0.113421, -0.232094, -0.01459, 0.202982, -0.071722, 0.277546, -0.041875, -0.104387, -0.029272, -0.000997, 0.057164, -0.033325, -0.221592, 0.127124, 0.058385, 0.282322, 0.430984, -0.047351, 0.093968, 0.078583, 0.039368, -0.037362, 0.013294, 0.320082, 0.002292, 0.040803, -0.218138, 0.100777, -0.06057, 0.131277], [0.036813, -0.16719, 0.017787, 0.009006, 0.102299, -0.295945, 0.086202, -0.353298, -0.135635, -0.034223, 0.00314, 0.118851, -0.501641, 0.018794, -0.024393, -0.114038, 0.006467, 0.018875, -0.258337, -0.087252, -0.178144, -0.104332, 0.039864, 0.179902, -0.145857, 0.045675, 0.082735, -0.002214, 0.189285, 0.146081, -0.165121, -0.185029], [0.00985, -0.149346, 0.068344, 0.100407, -0.001111, 0.207048, -0.177451, -0.091227, -0.31354, 0.05312, -0.150964, -0.020257, 0.011365, -0.045983, 0.103193, 0.095295, -0.041452, 0.050199, -0.156409, -0.118596, 0.052343, -0.197653, 0.186573, -0.118759, -0.237285, 0.069883, -0.015826, -0.17463, -0.151398, -0.078413, 0.087522, -0.173703]], [0.113535, 0.000611, 0.034523, -0.12776, 0.206671, -0.038148, 0.088098, -0.134493, -0.070723, -0.005184, -0.097199, 0.003003, -0.163131, 0.093437, -0.314687, 0.107476, 0.003127, 0.066736, -0.06172, -0.107799, 0.047835, 0.046139, 0.127412, 0.011851, -0.032332, -0.18181, -0.068554, 0.188361, -0.114208, -0.188831, -0.167382, -0.178191]),
    ([[0.154032], [-0.080624], [-0.17929], [-0.305764], [-0.112293], [0.124533], [-0.06512], [0.292289], [-0.233596], [-0.056795], [-0.109799], [-0.032161], [0.447937], [-0.13448], [-0.187704], [0.131087], [-0.10819], [0.198234], [0.353759], [-0.243505], [-0.238494], [-0.197098], [0.178722], [-0.274882], [-0.107091], [0.082371], [-0.068144], [-0.20518], [-0.276116], [0.198661], [-0.106939], [-0.281412]], [0.14732]),
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
