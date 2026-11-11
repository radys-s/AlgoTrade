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
    ([[-0.183977, 0.195581, 0.207072, -0.000909, -0.282656, 0.079574, -0.032525, 0.784595, -0.039998, -0.196346, 0.159954, -0.168061, -0.309219, 0.018353, -0.011646, -0.125136, -0.062544, 0.007311, 0.285594, -0.021031, 0.000963, -0.085327, -0.019809, 0.066659, 0.101788, 0.057742, 0.039364, 0.058516, 0.057769, 0.237234, -0.017276, -0.150998], [-0.087395, -0.177572, 0.096933, -0.142251, 0.092501, 0.218933, -0.039919, -0.17289, 0.126554, 0.09321, -0.276824, -0.036445, -0.457819, 0.17024, -0.2083, 0.012216, 0.000456, 0.195581, -0.014281, -0.123591, 0.131205, -0.181327, 0.017128, -0.169503, -0.150922, -0.095131, 0.087782, -0.152424, 0.11308, 0.175553, -0.000414, -0.091176], [-0.023827, -0.177049, -0.073544, 0.096806, 0.169128, -0.035538, -0.190252, 0.095771, 0.068212, -0.183469, -0.152699, 0.074306, -0.139255, 0.227058, -0.053421, -0.103685, 0.307753, -0.167131, -0.321001, -0.047745, 0.009524, -0.081591, 0.128776, -0.0746, -0.230742, -0.181576, 0.149865, 0.09527, 0.173558, 0.2066, 0.004046, 0.116235], [-0.188519, -0.193682, -0.056181, 0.142394, 0.352385, -0.062939, 0.066661, -0.066141, -0.369743, 0.161424, -0.019375, 0.070154, 0.344699, 0.232125, 0.06615, 0.074463, -0.004568, -0.064948, -0.464608, 0.243924, -0.116786, 0.080526, -0.006181, 0.138588, -0.022452, 0.215844, -0.188997, 0.124541, -0.246517, -0.184123, -0.074582, 0.335626], [-0.316257, -0.291333, -0.361395, 0.025088, -0.27657, -0.391723, 0.015337, 0.047406, 0.264653, 0.235407, -0.145782, 0.404873, 0.382782, 0.237385, 0.250861, -0.093515, -0.160056, 0.083957, -0.209496, -0.062686, 0.287455, -0.224221, -0.486276, -0.105283, -0.495112, 0.159066, -0.148629, 0.01461, 0.253105, -0.213435, 0.12495, 0.1914], [-0.238994, 0.042102, 0.110318, -0.013641, 0.271594, -0.098333, 0.083096, -0.281775, 0.056252, 0.252193, -0.334648, 0.171519, 0.162836, 0.081481, 0.130939, -0.056633, 0.151633, -0.17892, 0.010158, 0.240771, -0.238753, -0.021692, 0.02783, 0.179135, 0.013795, -0.097245, -0.062087, 0.293521, -0.115602, 0.09497, -0.104842, 0.254334], [0.117543, -0.128522, -0.198699, 0.296959, 0.070936, 0.277498, -0.246145, 0.23564, 0.089088, 0.173631, -0.208246, 0.159133, -0.403842, -0.131726, -0.251867, -0.006374, -0.226879, -0.073457, -0.438369, 0.231004, 0.441813, 0.169898, -0.180698, -0.135573, -0.093824, 0.079576, -0.08757, -0.140133, 0.253271, -0.003766, 0.137889, 0.490278], [0.125303, 0.043883, -0.033969, -0.06363, 0.132271, 0.483334, -0.109769, 0.109238, 0.161544, -0.31221, 0.04656, 0.0547, 0.048741, -0.018128, 0.003106, 0.06606, -0.063687, 0.03788, -0.062469, -0.112174, 0.029627, -0.020387, -0.053674, -0.195872, 0.324776, 0.332393, -0.039343, 0.011144, 0.184489, 0.195516, 0.034232, -0.01415], [0.150175, -0.486017, -0.08283, 0.01977, -0.416775, -0.321959, -0.118162, 0.334088, 0.131065, 0.111698, 0.003849, 0.33584, -0.11972, 0.260824, 0.036148, 0.251966, 0.106376, -0.583677, 0.181072, 0.345511, 0.144543, -0.174128, -0.032314, 0.226996, -0.493615, 0.207445, -0.063006, -0.43738, -0.408312, -0.197056, 0.093825, -0.09313], [0.09663, 0.013626, 0.067416, -0.097392, 0.268475, 0.116663, 0.02392, -0.02202, 0.127682, -0.188323, -0.139514, 0.312071, -0.124621, -0.085354, 0.156338, 0.287234, -0.569058, -0.917522, -0.067081, 0.077674, -0.142406, 0.108149, 0.067096, 0.198843, -0.158204, -0.340439, 0.324622, 0.280907, 0.084267, -0.56607, -0.199184, 0.013401], [-0.14378, 0.006398, 0.091781, -0.005002, 0.320481, 0.029184, -0.013594, -0.033695, -0.049304, 0.292459, 0.076485, -0.031478, -0.010363, -0.106251, -0.210868, -0.041854, -0.120519, -0.01798, 0.085836, -0.059626, 0.119842, 0.109508, 0.122255, -0.018581, -0.049162, 0.037504, -0.123754, -0.207595, 0.079788, 0.040245, 0.07754, -0.226654], [-0.028831, -0.080594, 0.306904, 0.231649, -0.117227, 0.044594, 0.259368, -0.17454, 0.090252, -0.238477, 0.154981, -0.01992, -0.194325, 0.356262, 0.542581, -0.445797, 0.030428, -0.364227, -0.478858, 0.37142, 0.242238, -0.343899, -0.143052, 0.263948, 0.29513, 0.176932, -0.09801, -0.035461, -0.038028, 0.129686, -0.235111, -0.615266], [-0.255355, -0.116866, 0.003741, 0.138035, 0.128593, -0.202481, -0.093191, 0.038583, -0.105469, 0.003003, 0.490552, -0.042324, 0.082857, -0.166259, 0.121643, 0.148132, 0.25307, 0.114756, 0.334412, -0.051676, -0.082359, -0.139747, -0.030605, 0.01377, -0.09891, 0.187515, -0.13872, 0.125641, -0.001335, -0.211728, -0.273789, 0.11404], [-0.031124, -0.28262, 0.124911, 0.028944, 0.001947, -0.100437, -0.03971, -0.133569, -0.159304, -0.16783, 0.262357, -0.127214, -0.097964, 0.102398, 0.025194, 0.119108, -0.108175, 0.022024, 0.116139, 0.133197, -0.149876, -0.445557, -0.053508, 0.198644, 0.049488, 0.010723, -0.03643, 0.110563, 0.08703, 0.017768, -0.158696, 0.176833], [-0.345616, 0.180346, 0.25753, -0.146325, 0.234358, -0.199552, -0.243238, 0.010554, -0.017078, 0.097892, 0.279705, 0.15434, -0.052119, -0.18209, -0.334387, 0.061158, 0.181503, -0.188647, 0.069865, -0.314233, 0.167664, -0.059957, 0.348416, -0.190474, -0.241128, -0.24652, -0.004845, 0.347482, -0.172712, 0.095932, -0.338227, -0.297512], [-0.027399, 0.005357, -0.23529, -0.058546, -0.344419, 0.053789, -0.602958, -0.100429, -0.083655, 0.040844, 0.05719, 0.158657, -0.08845, -0.227205, 0.074026, -0.248596, -0.0652, -0.099444, 0.128448, 0.003564, -0.10745, -0.031282, 0.105121, 0.134404, 0.01809, -0.280054, -0.14447, 0.092056, -0.010309, 0.017149, -0.113881, -0.17289], [-0.130618, -0.317308, 0.195922, -0.218812, 0.178669, 0.269718, 0.314187, 0.021316, -0.085882, -0.04803, -0.297527, 0.333919, 0.018443, 0.383046, 0.272894, -0.213612, -0.164267, 0.503158, 0.261014, 0.210973, -0.520686, -0.003435, 0.102003, -0.436642, 0.331664, 0.491949, -0.029682, -0.285604, 0.973118, -0.35031, -0.029483, -0.375515], [0.174972, -0.061107, 0.149677, -0.217112, -0.072725, 0.021436, 0.043174, 0.258755, -0.131336, -0.605635, -0.174782, -0.264317, 0.205588, 0.112009, -0.162681, 0.185463, 0.528416, -0.267583, 0.648955, 0.162874, -0.037854, 0.220215, 0.244271, 0.036768, -0.231734, 0.362376, 0.040768, -0.385801, -0.186303, -0.035948, 0.295962, 0.244994], [-0.047479, 0.049399, -0.021264, 0.042003, 0.028061, 0.015756, -0.143543, 0.048716, 0.193619, 0.023198, 0.110584, 0.004447, -0.004642, -0.010995, 0.075878, -0.206047, -0.007188, 0.048027, -0.042935, -0.204846, -0.017555, 0.005897, -0.078205, 0.228679, 0.152313, -0.013942, 0.015341, -0.074239, -0.022202, -0.134613, -0.025102, 0.03441], [0.107349, -0.205913, -0.604014, -0.172339, 0.008314, -0.089502, 0.106568, -0.032625, 0.882721, -0.103481, 0.220827, -0.043163, 0.021437, -0.038326, 0.367664, 0.237002, 0.09201, -0.25537, 0.127444, -0.230897, 0.014608, -0.263987, -0.087792, 0.572849, 0.181412, -0.00818, -0.347069, 0.096607, -0.105947, 0.135, 0.111149, 0.100426], [-0.013166, 0.191777, 0.199816, -0.039525, 0.051262, 0.284454, 0.337912, 0.003766, -0.014971, 0.143204, -0.165194, 0.072386, 0.107254, -0.118874, -0.160907, -0.054703, 0.178198, 0.117013, 0.087405, 0.308566, -0.150433, 0.001725, 0.279123, 0.264433, -0.173795, -0.266844, -0.054557, -0.094791, 0.024963, 0.084768, -0.075982, -0.017703], [0.006013, -0.071951, 0.003339, -0.579146, -0.021252, 0.325279, 0.173658, -0.000147, -0.027547, 0.013963, 0.027735, 0.167608, -0.0153, -0.120452, -0.183164, 0.101686, 0.044599, 0.110634, 0.031873, 0.16045, 0.66875, 0.020788, 0.046323, 0.079852, -0.045993, -0.244708, 0.166072, 0.011653, -0.064939, -0.19531, 0.101483, 0.061477]], [0.022143, -0.289545, 0.250986, 0.073018, 0.019093, -0.030878, 0.05372, -0.103474, 0.088949, -0.067865, 0.277231, -0.325786, -0.117688, -0.096544, -0.352898, 0.006973, 0.223334, 0.353205, -0.50724, -0.57201, -0.196814, -0.098365, 0.285937, -0.258894, -0.080782, -0.198968, -0.035498, -0.281459, 0.508386, -0.518977, 0.437144, 0.104638]),
    ([[-0.085478, 0.146413, -0.119915, 0.003778, -0.192033, 0.055807, 0.165322, 0.026839, 0.026123, 0.133138, 0.188421, 0.067776, -0.116116, -0.017485, 0.022526, 0.168299], [0.307415, 0.160202, 0.030897, 0.1109, -0.000815, 0.212205, 0.005317, -0.127825, 0.175152, 0.249033, 0.073157, -0.046294, 0.11356, 0.002135, 0.132577, -0.065687], [-0.335079, 0.108712, -0.005888, 0.133321, 0.134745, 0.278483, -0.099992, 0.198849, -0.003202, -0.140015, 0.139715, -0.262514, 0.270916, 0.140532, 0.184088, -0.162327], [-0.061594, -0.003618, 0.191778, -0.044938, -0.132921, -0.201331, 0.08661, 0.078294, 0.033224, -0.250489, 0.22314, -0.077432, 0.103408, -0.061673, 0.129902, -0.064036], [0.015813, 0.091744, -0.062542, 0.327497, -0.022154, 0.086173, 0.166885, 0.208235, -0.192334, -0.500513, 0.240037, 0.239653, -0.306602, 0.039388, -0.008476, 0.022173], [-0.152919, -0.022713, -0.105721, 0.161227, -0.216688, 0.090263, -0.134804, -0.687291, -0.033536, -0.331676, -0.152197, -0.304995, -0.134706, -0.086794, -0.112938, -0.075973], [-0.400022, 0.082431, 0.02693, -0.039091, 0.105763, -0.43582, -0.171112, -0.319933, 0.03174, 0.050726, -0.043117, -0.381346, 0.213613, 0.002434, 0.038634, -0.173519], [0.150339, 0.344902, -0.257605, 0.118151, -0.015985, 0.218105, 0.30101, -0.107216, -0.028629, 0.255616, 0.730716, 0.217037, -0.286589, 0.359303, -0.041845, 0.576909], [-0.003269, -0.30766, 0.073881, -0.490516, 0.064025, 0.0078, -0.618767, 0.144563, -0.226427, 0.189385, -0.046344, 0.032754, 0.026706, 0.007269, 0.142025, -0.912001], [0.135479, 0.080126, -0.086571, 0.098252, 0.280035, 0.123581, 0.28989, 0.092533, -0.012648, 0.274368, -0.215931, -0.282679, -0.121754, 0.163643, -0.100445, 0.488411], [-0.456105, -0.20285, 0.10316, -0.098409, -0.030141, -0.137579, -0.079021, 0.061564, -0.002107, -0.122756, 0.008655, -0.377278, -0.012769, -0.287446, -0.081802, -0.008907], [0.041219, -0.122831, -0.331705, -0.062918, -0.254374, 0.097647, -0.171008, 0.284369, 0.031139, 0.174414, -0.4184, -0.170442, 0.31483, 0.061953, -0.31592, 0.178168], [0.27528, 0.373163, -0.136019, 0.150615, 0.285637, 0.275142, 0.356818, 0.67754, -0.003495, 0.386881, 0.143639, 0.352361, -0.285861, 0.273358, 0.108647, 0.34039], [0.267317, 0.010084, -0.072104, 0.000348, 0.178073, 0.146296, 0.25886, 0.168568, 0.042377, -0.095311, 0.004858, 0.161664, -0.331386, -0.071256, -0.058218, 0.196656], [0.213655, -0.047505, -0.308972, -0.226485, -0.018648, 0.088859, -0.734387, 0.125088, -0.068207, 0.046449, 0.052577, 0.207257, 0.083977, -0.048422, -0.327264, -0.523301], [-0.21441, -0.149528, -0.048952, 0.116987, 0.121156, -0.464, -0.020386, -0.091582, -0.116824, -0.232677, 0.147523, 0.202767, -0.103541, -0.25936, 0.088095, 0.076968], [0.019209, -0.541305, 0.28332, -0.705011, 0.223743, 0.144616, 0.17752, -0.961361, -0.085838, -0.138373, 0.146487, 0.085913, 0.014991, 0.238035, -0.26355, -0.141884], [-0.176095, -0.246631, 0.140249, -0.124214, -0.211115, -0.645115, 0.066366, 0.228878, 0.235685, -0.222046, 0.150423, -0.081198, -0.127359, -0.646512, -0.136393, 0.118842], [0.318695, 0.052536, 0.061717, 0.234096, 0.097932, 0.395151, 0.090695, 0.11305, -0.104511, 0.455968, 0.161647, 0.027275, 0.294209, 0.3565, 0.301965, 0.088739], [0.137691, -0.156503, 0.089786, -0.035326, 0.577281, 0.107768, 0.475737, 0.135085, 0.119017, 0.212674, -0.086901, 0.578149, -0.228404, 0.161375, 0.182887, 0.39388], [-0.091303, -0.146684, 0.154853, -0.289453, 0.03289, -0.023142, -0.119428, -0.281217, 0.058953, -0.122932, -0.570132, 0.256429, 0.229244, 0.162957, 0.01321, -0.667003], [0.014648, 0.018496, 0.19858, 0.099398, 0.090662, -0.105077, 0.168894, -0.137227, 0.146699, 0.112928, -0.033771, -0.489215, 0.088463, -0.316602, 0.121439, -0.040352], [-0.293396, -0.024137, 0.154278, 0.086618, -0.558262, 0.060518, 0.167593, -0.046826, 0.212106, -0.021616, 0.205655, -0.133044, -0.131744, 0.189763, 0.075085, -0.125867], [-0.012843, -0.067675, 0.15526, -0.359301, 0.044944, 0.036104, -0.031502, -0.515362, -0.033607, -0.140796, -0.122524, 0.041594, -0.192214, -0.059679, 0.301501, -0.67943], [-0.192983, 0.142782, 0.026374, 0.196264, -0.223166, 0.12082, -0.131365, 0.283482, 0.21833, 0.066861, -0.261133, 0.350771, -0.169173, 0.209394, -0.175089, 0.080905], [-0.272857, -0.108701, 0.075833, -0.24059, -0.138539, -0.081946, 0.100877, -0.106846, -0.134471, -0.592236, -0.182405, -0.175161, 0.208352, -0.268322, 0.24306, 0.166459], [0.174888, 0.191509, -0.07159, 0.036785, 0.402624, 0.043841, 0.148967, 0.113967, 0.194235, 0.038989, 0.196183, -0.051741, 0.203074, 0.139079, 0.028654, 0.146392], [0.118535, 0.047711, 0.076176, 0.154153, 0.211785, 0.041679, 0.398926, -0.056328, 0.043197, 0.246976, 0.220771, -0.275925, -0.35101, -0.037503, 0.088306, 0.240659], [0.036282, 0.095509, -0.228251, -0.01779, 0.126334, -0.194349, -0.390653, -0.797954, 0.23237, -0.014962, -0.631781, 0.222485, 0.210166, 0.291431, -0.288159, -0.192518], [-0.209636, -0.214243, -0.248117, 0.005019, 0.134041, 0.334596, -0.08404, -0.251775, -0.122626, -0.095508, -0.119094, 0.141883, -0.388666, 0.187843, -0.205133, -0.216702], [0.30683, 0.115909, 0.056201, 0.208224, -0.026602, -0.214923, 0.061254, -0.186669, 0.128172, -0.04337, 0.021807, -0.015961, 0.157159, 0.093688, 0.024507, 0.148775], [-0.140071, -0.161612, 0.28058, 0.070287, -0.046276, -0.26665, -0.424521, 0.104961, -0.091551, -0.15615, -0.262069, -0.194862, 0.199954, -0.095521, -0.053656, -0.26277]], [0.257632, 0.169007, 0.262092, 0.139217, -0.118461, -0.0445, -0.148468, -0.39751, 0.214065, -0.00676, -0.263833, -0.066441, -0.494193, -0.055495, 0.220676, -0.27004]),
    ([[0.855493], [-0.502948], [-0.267794], [-0.494527], [0.426892], [0.569819], [-0.477829], [-0.97256], [-0.403367], [0.62369], [-0.602266], [0.949811], [0.961032], [0.404121], [-0.393075], [-0.834843]], [0.422994]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.562321557048493e-19, -1.1343482335572739e-18, -5.671741167786369e-19, -1.8905803892621232e-19, 1.1343482335572739e-18, 0.0, 0.0, 0.0, 1.2572534685440513, 0.0, 10.639801906763806, 28.799029822078616], [6.72849538126497, 19.8058956257327, 34.925161910299096, 44.72666958616459, 57.636846319296275, 1.1243487056677264, 0.9903909371630099, 0.9446913427593254, 0.8452682282888074, 0.7575381583058208, 1067.5606541222405, 2053.035725905977, 4046.2991308026644, 0.6179080587708422, 0.3127841039679992, 16.705331990766382, 34.22120473597514], [0.6490227662551226, 0.3600762349252582, -0.1832944003136108, -0.09733275171505404, -0.10066480452612983, 0.8482856190320313, -3.2486297440276823, 6.1619984101404714, -3.540934484781457, 3.1035095898512166, -0.6019915249010418, 0.06557241931064624, -0.1855095648918967, -0.002102755891520747, 0.4105229362630337, 0.0003153876484541001, 0.00020715056621090425], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.568942287397852e-19, -1.1353413431096778e-18, -5.676706715548389e-19, -1.892235571849463e-19, 1.1353413431096778e-18, 0.0, 0.0, 0.0, 1.2574354160321035, 0.0, 10.646654416476506, 28.81409587438229], [6.7313766144182345, 19.814344420832445, 34.939862329400235, 44.74491730719747, 57.65822173996363, 1.124692105470714, 0.9907164467252751, 0.9450073668099886, 0.8455601770714993, 0.7578101560190531, 1067.786637740178, 2053.787560401509, 4047.179122347928, 0.6181145992138308, 0.3129209935553393, 16.710984248831306, 34.23237508220454], [2.9421164343350763, 0.7964360421909548, -1.0744230210945729, -0.9234918162695617, -0.8475723430517554, 1.4775048291769728, -4.635873707445536, 9.213729195881914, -4.907517756528247, 6.23467901404692, -2.386919742393157, 0.4346797676233983, -1.5368709559241014, 0.001829056109893383, -0.22349228908983945, -0.0002742367847445596, -0.00018005766511644536], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.257441, 0.0, 0.0, 5.84623, 5.846549, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987547], [1.124422, 0.990545, 0.944803, 0.844787, 0.75662, 2.57683, 18.120787, 0.790527, 15.924298, 0.695578, 0.651228, 0.54053, 1.677429, 1.677372, 0.999993, 4.403117, 1.087628, 0.310614, 132.397858, 334.626434, 1066.212891, 1.340704], [([[-0.238185, 0.052293, 0.189494, 0.03579, -0.435434, 0.011955, 0.163994, 0.719009, -0.08335, -0.024156, 0.078548, -0.137638, -0.266167, 0.136204, -0.166593, -0.160731, -0.029433, 0.017319, 0.242549, -0.015039, -0.032783, -0.057088, 0.225211, 0.179803, 0.224791, -0.143878, 0.053318, -0.150215, -0.076773, 0.20386, -0.040849, -0.094908], [-0.133257, -0.188863, 0.013296, -0.261241, -0.105976, 0.157159, -0.02324, -0.17371, 0.034629, 0.201033, -0.209434, 0.073905, -0.269319, 0.216479, -0.199918, -0.011933, 0.067028, 0.219, -0.018885, -0.062694, 0.136758, -0.16695, 0.161038, -0.14526, 0.102192, -0.174056, 0.074254, -0.240141, 0.192351, 0.086987, -0.037406, -0.121482], [-0.03541, -0.110437, -0.135369, 0.036666, -0.013907, -0.059181, -0.181754, 0.089956, 0.003281, -0.151169, -0.045444, 0.136091, -0.088073, 0.20067, -0.022146, -0.105527, 0.307392, -0.176358, -0.330839, 0.001008, 0.02511, -0.061349, 0.217815, -0.093773, -0.071837, -0.16863, 0.152369, 0.183027, 0.26037, 0.113455, 0.044213, 0.075282], [-0.09374, -0.042648, 0.012721, 0.224983, 0.510113, 0.090819, 0.161281, -0.1797, -0.407583, 0.036547, 0.134091, -0.012379, 0.06613, 0.041006, 0.035069, 0.120133, -0.188249, -0.052563, -0.470322, 0.240648, -0.244847, 0.160937, 0.028631, 0.074324, -0.038989, 0.339338, -0.16748, 0.428946, -0.390528, -0.204197, -0.03525, 0.379599], [-0.209915, -0.230884, -0.237122, 0.150459, -0.261685, -0.537641, 0.035186, 0.100352, 0.550727, 0.11367, -0.204268, 0.301177, 0.123868, 0.025185, 0.37698, -0.04192, -0.355288, 0.053238, -0.285824, 0.001417, 0.446767, -0.225256, -0.533959, 0.331617, -0.344401, -0.018568, -0.141876, 0.009969, 0.346516, 0.033485, 0.05579, 0.173056], [-0.282235, -0.060366, 0.034313, -0.064134, 0.214005, -0.098891, 0.164883, -0.252142, 0.005226, 0.246242, -0.255541, 0.302588, 0.279546, 0.149923, 0.182551, -0.067856, 0.211587, -0.202155, -0.110294, 0.342171, -0.099208, -0.104449, -0.032561, 0.016376, 0.36039, -0.142775, -0.163072, 0.310374, 0.064868, 0.110291, -0.203735, 0.141687], [0.40205, 0.078597, -0.104594, 0.37757, -0.243781, -0.091113, -0.758081, 0.257717, 0.294728, -0.133293, -0.317439, -0.213858, -0.13771, -0.306106, -0.027561, -0.173036, -0.00687, -0.245728, -0.454244, 0.083741, 0.406741, 0.239755, -0.158625, -0.203803, -0.319181, 0.301281, -0.16469, -0.371114, 0.039642, -0.00255, 0.533334, 0.729947], [0.055405, -0.069106, -0.128119, -0.190884, 0.05613, 0.032943, -0.103566, -0.098479, 0.197751, -0.036849, 0.1607, -0.126327, -0.031454, 0.038558, -0.020068, 0.125327, -0.051974, -0.049682, 0.082259, -0.090364, -0.158498, -0.092122, -0.007629, -0.388608, -0.01294, 0.092293, 0.008969, -0.090117, -0.133078, 0.13798, -0.099408, 0.100298], [0.271365, -0.032943, 0.189711, -0.072127, -0.260246, -0.068888, -0.010257, 0.06543, -0.170894, -0.146404, 0.104147, 0.22334, -0.143663, 0.236365, -0.305272, 0.160232, 0.016564, -0.386804, 0.40065, 0.257499, -0.356792, -0.293241, 0.133764, -0.223166, -0.102412, 0.267304, 0.084274, 0.401294, -0.200231, -0.26722, 0.560381, 0.053419], [-0.210731, 0.085556, 0.104585, -0.202588, 0.263567, -0.045892, 0.027812, 0.096422, 0.131296, -0.983624, -0.247296, -0.063975, -0.019268, -0.263101, 0.204446, 0.23615, -0.242285, -0.874116, 0.065008, -0.124426, -0.057975, 0.16623, 0.009171, 0.027091, -0.180878, -0.740467, 0.258692, 0.266229, 0.178527, -0.321972, -0.406615, -0.059293], [-0.066275, 0.00534, 0.069116, -0.003609, 0.066673, 0.024098, -0.097401, -0.07323, -0.102741, 0.216436, -0.121022, 0.110662, -0.012067, -0.263487, -0.024612, 0.041309, -0.159343, 0.070335, -0.012397, -0.131035, 0.076212, 0.044279, 0.079386, 0.143267, 0.131229, -0.036523, -0.04074, 0.094513, 0.035019, -0.02505, 0.039583, -0.221667], [0.083434, -0.25114, 0.006371, 0.367804, 0.037581, 0.094784, 0.144148, -0.061848, 0.102102, -0.12572, 0.137014, -0.061016, -0.172434, 0.487631, 0.268214, -0.392936, 0.172273, -0.330128, -0.052873, 0.286614, 0.149911, -0.286388, 0.061747, 0.017636, -0.05447, 0.002945, -0.251025, 0.153835, 0.05862, 0.33597, 0.13668, -0.692494], [-0.231663, -0.249263, -0.009954, 0.139692, -0.179103, -0.103471, -0.041312, 0.048645, -0.060172, 0.141571, 0.527497, -0.205793, 0.087593, -0.190187, 0.12841, 0.419556, 0.239755, 0.112453, -0.051189, 0.18042, -0.176801, -0.062427, -0.061079, 0.151636, -0.100892, 0.032241, -0.044329, -0.05719, -0.055531, -0.098718, -0.148952, 0.037075], [0.136804, -0.247053, 0.101344, -0.015384, -0.058437, -0.003979, -0.189485, -0.066254, -0.049306, 0.115158, 0.24845, -0.193962, -0.105023, 0.08371, 0.201794, 0.065786, -0.069958, 0.301496, 0.043389, 0.153492, -0.199985, -0.246235, 0.008777, 0.323181, -0.127345, 0.044439, 0.047381, -0.061089, 0.137712, -0.036073, -0.350493, 0.203137], [-0.201008, 0.060918, 0.416738, 0.000608, 0.071078, -0.177249, -0.111776, 0.03546, 0.03612, 0.045323, 0.036797, 0.217546, 0.102638, -0.06368, -0.144981, 0.155424, 0.064551, -0.119063, -0.221068, -0.194222, 0.321703, -0.184231, 0.295789, -0.180326, 0.29205, -0.132432, 0.237279, -0.06524, 0.034996, -0.066825, -0.225123, -0.215725], [-0.134777, -0.209662, -0.237965, -0.329502, 0.024305, 0.206399, -0.270803, -0.054984, -0.246448, 0.027126, 0.278578, 0.035474, 0.061911, -0.165865, -0.119794, -0.030034, -0.218926, -0.087665, 0.208316, -0.022314, -0.145058, -0.053404, 0.06544, -0.175187, 0.462686, -0.208345, -0.267318, 0.148841, 0.126247, 0.106036, -0.066314, -0.258418], [-0.39912, -0.168665, 0.506429, -0.179753, 0.023119, -0.052036, 0.400715, -0.056349, -0.163477, -0.052835, -0.063914, -0.118492, -0.124553, 0.232293, 0.425903, -0.523122, 0.000166, 0.299222, 0.083645, 0.221131, -0.092847, 0.164209, 0.0261, -0.29879, 0.146384, 0.393521, -0.14772, -0.405434, 0.517771, -0.061474, 0.180701, -0.064154], [-0.022343, -0.255066, 0.034145, -0.047603, -0.048954, -0.109657, 0.191978, 0.046614, 0.015053, -0.246035, -0.372215, -0.271568, -0.056315, -0.036559, -0.066479, 0.105253, 0.279362, -0.373152, -0.000916, 0.137037, -0.003061, 0.184079, 0.027289, 0.138592, -0.046905, -0.041397, 0.183771, -0.183345, -0.136069, 0.421765, 0.368073, -0.181009], [0.0705, 0.031425, -0.089823, -0.040713, 0.008197, -0.024542, -0.18226, 0.009951, 0.063206, -0.065377, 0.05563, -0.048103, 0.044043, -0.049114, -0.045625, -0.154313, 0.003883, 0.01219, -0.040778, -0.044088, 0.062376, -0.058218, -0.00048, -0.016712, -0.059639, 0.04367, 0.095678, -0.019754, -0.028529, 0.004998, -0.05867, -0.056682], [-0.050897, -0.202343, -0.857622, 0.032831, -0.017104, -0.116388, 0.119805, -0.013244, 0.089332, 0.21736, 0.134198, -0.176186, -0.220679, 0.022358, 0.243555, 0.265709, -0.048383, -0.206993, -0.271454, -0.219749, -0.194002, -0.270261, -0.096147, 0.062748, -0.097517, -0.155388, -0.140364, 0.18939, 0.028322, 0.307917, 0.103403, 0.053363], [-0.149638, -0.012432, -0.053765, -0.238264, 0.071836, 0.768926, -0.153266, 0.032254, -0.144795, 0.119407, -0.116206, 0.098301, 0.594783, 0.055924, 0.042191, -0.101106, 0.444116, 0.262254, 0.619189, 0.052816, -0.186999, -0.084536, 0.148272, -0.160853, -0.19857, -0.199648, -0.131186, -0.153691, -0.010262, -0.017555, -0.325027, 0.10075], [-0.021488, 0.036761, -0.152251, -0.509326, -0.026825, -0.001702, 0.369311, 0.02129, 0.003958, 0.198831, 0.093924, -0.052326, 0.037633, 0.033704, 0.066446, -0.082596, 0.1249, 0.224973, -0.06715, 0.078997, 0.366472, 0.014209, 0.037485, 0.065945, -0.038356, -0.023484, 0.263011, -0.113296, -0.195963, -0.200638, 0.088311, -0.003566]], [-0.053736, -0.241734, 0.090757, 0.068371, -0.034052, 0.321178, 0.405801, 0.010668, 0.386576, 0.11784, 0.266263, -0.028316, -0.0441, 0.095868, -0.137939, -0.012656, -0.087616, 0.050136, -0.456816, -0.601457, 0.174212, 0.002744, 0.424413, -0.203412, -0.162847, -0.047849, -0.073106, -0.41314, 0.328521, -0.648173, 0.182669, 0.033197]), ([[-0.047506, 0.118163, -0.101751, 0.062061, -0.001426, 0.077006, 0.272902, 0.115401, 0.057612, -0.095778, 0.208382, 0.157501, -0.157557, 0.136996, 0.048272, 0.152766], [0.171933, 0.197287, -0.027616, 0.161517, 0.142904, 0.033225, 0.063277, -0.225193, 0.140581, 0.119564, -0.12101, 0.163656, 0.016883, -0.003643, 0.083335, 0.061477], [-0.951459, 0.111934, 0.059878, 0.087061, -0.066479, 0.004597, -0.056873, 0.126954, -0.073238, -0.497812, 0.119881, -0.137536, 0.279622, -0.128446, 0.135928, 0.012995], [-0.269688, 0.105051, 0.243895, -0.242941, 0.199648, -0.197391, -0.037963, 0.268273, 0.215226, -0.124895, 0.01097, -0.203656, 0.087463, 0.143386, 0.266697, 0.027829], [0.2158, 0.149919, 0.108455, 0.378993, -0.24326, 0.210054, 0.204078, 0.022478, 0.108591, 0.027505, 0.366415, 0.238727, -0.155762, 0.141524, -0.011846, -0.412928], [-0.052784, -0.076579, 0.183666, 0.419548, -0.546011, 0.166047, -0.10358, 0.212627, -0.166521, -0.622407, -0.110313, -0.263128, 0.025874, -0.234147, 0.001522, -0.240031], [-0.399383, 0.175648, 0.104819, 0.102928, 0.235139, -0.300256, -0.075144, 0.083113, 0.166327, 0.033419, 0.101849, -0.301427, 0.265036, 0.239151, -0.015641, -0.14307], [0.381647, 0.154551, -0.06394, -0.032488, 0.071403, 0.399854, 0.417648, -0.211603, -0.001871, 0.440462, 0.298046, 0.173055, -0.196152, 0.334042, 0.017502, 0.631256], [-0.051941, -0.22941, 0.040782, -0.342475, 0.019862, 0.194838, -0.457865, -0.415496, -0.206046, 0.246841, 0.034192, 0.175738, 0.118067, 0.069933, 0.296659, -1.010469], [0.141215, 0.159731, -0.210922, -0.162473, 0.196478, 0.100183, -0.350501, -0.303798, -0.027898, 0.262166, -0.606246, -0.020886, -0.5266, 0.075221, -0.186414, -0.519651], [0.032723, 0.052939, 0.146421, -0.031365, -0.22836, -0.05069, -0.242982, 0.435001, 0.075997, -0.286272, -0.089695, -0.355288, -0.079183, 0.192182, 0.297115, 0.253311], [-0.002443, -0.157957, -0.287834, -0.038808, -0.136026, 0.124668, -0.397849, -0.328673, 0.117553, 0.043054, -0.053162, -0.035705, -0.304947, 0.088535, -0.206462, 0.470975], [0.265597, 0.285883, 0.091932, 0.172724, 0.231487, 0.262477, 0.333923, 0.087922, 0.015794, 0.274623, 0.215918, 0.228774, -0.443375, 0.016123, -0.021876, 0.245513], [0.276936, 0.089053, -0.255959, 0.156181, 0.24345, 0.161585, 0.300065, 0.009524, 0.02753, 0.059613, -0.094478, 0.088535, -0.308654, -0.085625, -0.044489, -0.072352], [0.048368, -0.157343, 0.229326, -0.072028, 0.045807, -0.195179, -0.146032, -0.118477, -0.069778, 0.254249, -0.601034, -0.110939, -0.280121, 0.060686, -0.038471, -0.421335], [-0.050823, -0.031317, 0.19227, 0.155532, 0.028461, -0.226953, 0.141536, 0.128042, -0.012452, 0.08088, 0.314458, 0.289921, -0.255677, 0.18361, 0.185359, 0.158797], [-0.096005, -0.061413, 0.232745, -0.221727, -0.022341, 0.036255, 0.127011, 0.321503, -0.049319, -0.351623, -0.132738, 0.272837, 0.695551, 0.127216, 0.111569, -0.792631], [-0.465476, -0.095994, 0.088405, -0.264929, 0.015884, -0.311435, 0.05456, -0.152587, 0.04393, -0.251025, 0.180202, -0.668281, -0.042476, -0.221159, -0.121911, 0.201714], [0.226592, -0.098341, -0.069233, 0.094549, -0.471542, 0.395213, 0.096574, -0.529765, -0.182035, 0.116256, -0.015092, -0.248123, 0.254337, 0.237908, 0.159639, 0.073067], [0.069415, -0.191764, 0.052407, -0.474328, 0.315565, -0.021877, 0.180089, -0.191583, -0.348527, 0.226886, -0.664797, 0.324562, -0.293247, -0.014182, 0.343065, 0.432163], [-0.187441, -0.346886, -0.145789, -0.215219, 0.017791, -0.186352, -0.322715, -0.152075, 0.094473, -0.11733, -0.194259, 0.258294, 0.189246, 0.175662, 0.072731, -0.168312], [-0.180201, -0.003144, 0.157444, 0.104418, 0.047242, -0.188824, 0.214527, -0.316953, 0.140584, -0.21101, -0.100425, -0.086527, 0.066934, -0.121101, 0.053498, 0.045205], [-0.27188, 0.089536, 0.109902, 0.048627, -0.394635, 0.106752, 0.497423, 0.101365, 0.189798, -0.333294, 0.313868, 0.244777, 0.018675, 0.139627, 0.057497, 0.148314], [0.058482, 0.086805, 0.055386, -0.037679, -0.084315, -0.194606, -0.372625, 0.146083, -0.028426, -0.208262, -0.578172, -0.017001, 0.023377, 0.070909, 0.224206, 0.018486], [-0.178484, 0.158167, -0.002675, 0.122208, -0.088278, 0.18055, -0.086971, 0.124741, 0.126349, -0.064299, -0.567641, 0.214683, -0.093068, 0.135083, -0.058699, -0.503541], [-0.565435, 0.002913, -0.0382, -0.097742, -0.068865, -0.438001, 0.088476, -0.026582, 0.047221, -0.498733, -0.070898, -0.190016, 0.028512, -0.12434, 0.235972, 0.247604], [0.12082, 0.194849, -0.117257, 0.005664, 0.226945, -0.216296, 0.237312, 0.072423, 0.194486, -0.264497, 0.249568, 0.212042, 0.206801, 0.130391, -0.090977, 0.162266], [0.100499, -0.157382, 0.231151, 0.114864, 0.110791, -0.082545, -0.223893, -0.59773, -0.383005, 0.265473, 0.368537, -0.210948, -0.304319, -0.103445, 0.178663, -0.0564], [-0.038007, -0.019065, -0.173632, -0.123915, 0.338095, 0.166806, -0.248686, -0.305637, 0.055905, 0.133629, -0.253182, 0.096262, 0.025129, 0.362361, -0.29176, -0.054009], [-0.119172, -0.324437, 0.151954, -0.142178, 0.168478, 0.178509, -0.537316, -0.119599, -0.28385, 0.153002, 0.186393, 0.1871, -0.580645, 0.028453, -0.02367, -0.607571], [0.472015, 0.054461, 0.204308, 0.08786, 0.227744, -0.147983, 0.171176, 0.229617, 0.13738, 0.073259, -0.143385, -0.002351, -0.209034, 0.181325, 0.106861, 0.029053], [-0.08342, -0.043164, 0.47534, -0.234399, -0.027492, -0.09125, -0.392128, 0.346257, -0.117713, -0.132761, -0.255606, -0.173258, 0.1376, -0.234842, 0.085331, -0.126679]], [0.138229, 0.219052, 0.094723, 0.18928, 0.159015, -0.041639, -0.0926, -0.043291, 0.179806, 0.273977, -0.009518, -0.071417, -0.447421, -0.113001, 0.104402, -0.292731]), ([[2.112189], [-0.768212], [-0.602485], [-0.980124], [0.725712], [0.908097], [-1.007485], [1.117387], [-0.80268], [1.300427], [-1.429875], [1.275188], [1.962494], [0.475751], [-0.678468], [-1.113134]], [0.655636])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.75191, -9.2e-05, 9.2e-05, 0.0, 0.052846, 0.052846, -0.0, 0.0, 0.0, 1.243668, 5.879185, 5.879185, 0.0]
SEQ_TSD = [1.079295, 0.132411, 0.132411, 0.099558, 1.45424, 1.45424, 0.666687, 0.219428, 0.17921, 0.60273, 1.666237, 1.666237, 0.649942]
SEQ_XM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.257423, 0.0, 0.0, 5.846561, 5.846561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987482]
SEQ_XS = [1.124955, 0.99101, 0.94523, 0.845186, 0.757041, 2.577918, 18.130564, 0.790721, 15.945622, 0.69697, 0.6513, 0.540497, 1.677766, 1.677766, 0.999993, 4.402977, 1.087599, 0.310604, 132.411407, 334.802429, 1066.728027, 1.341266]
SEQ_YSC = 1.939010739326477
SEQ_WI = [[-0.096845, -0.829434, -0.107898, -0.467974, -0.381222, 0.14497, -0.800611, -0.274349, -0.389268, -0.038979, -0.122599, 0.029328, 0.206116, 0.157823, -0.047036, -0.031912, -0.076919, -0.073587, -0.337966, -0.863269, -0.616274, -0.191216, -0.215295, 0.474103, -0.358993, -0.228523, 0.470643, -0.100376, -0.020922, -0.595892, 1.011399, -0.349243, 0.419012, 0.496087, 0.038509, 0.091535, 0.589497, 0.06527, 0.287115, -0.109829, 0.043166, 0.417467, -0.078311, -0.085117, 0.144722, 0.115669, -0.053362, -0.030883], [-0.027401, -0.146192, -0.072932, -0.202239, 0.197195, -0.006623, -0.283735, -0.499554, -0.087327, -0.051059, -0.006442, 0.140465, 0.255112, -0.352978, -0.140672, 0.36669, -0.030271, -0.156444, 0.166739, -0.007936, -0.59538, -0.436368, 0.988001, -0.336479, 0.084002, 0.017793, -0.079667, 0.088734, 0.410812, 0.423281, -0.235728, 0.05501, -0.414948, -0.009137, 0.01798, 0.025652, -0.019992, 0.28593, -0.148918, -0.262812, -0.341957, -0.310673, -0.116173, -0.03513, -0.113561, 0.14654, 0.156494, 0.043794], [0.059137, 0.080677, -0.1622, -0.303595, -0.173196, -0.078944, -0.102948, 0.12596, -0.080073, 0.248752, -0.00192, -0.179498, -0.077641, 0.053633, 0.13648, 0.192982, 0.173017, 0.239493, -0.78292, 0.341406, 0.409213, 0.065498, -0.020364, -0.145103, 0.008457, 0.100771, 0.051532, 0.296784, 0.064331, -0.175662, 0.327942, 0.701583, -0.026544, 0.261512, 0.194755, -0.01832, -0.218506, 0.305425, 0.02077, 0.227596, 0.106909, 0.480207, -0.24818, 0.383426, 0.241013, -0.242632, 0.052755, 0.3075], [0.125125, -0.016511, -0.283455, -0.280965, -0.106362, 0.272869, -0.648902, -0.245734, -0.188635, 0.310182, 0.077352, 0.208862, 0.065432, -0.305237, 0.064526, 0.122466, -0.079075, 0.112342, -0.175575, 0.464988, 0.141376, -0.376074, 0.605418, -0.298275, 0.149922, -0.084348, 0.051551, 0.523982, 0.325267, -0.087046, 0.102545, 0.277371, 0.049704, 0.005167, -0.244663, -0.10391, 0.155015, 0.195984, 0.238801, 0.011884, -0.39505, -0.049664, -0.009135, 0.203437, -0.197036, -0.065031, 0.125882, 0.010936], [-0.255927, 0.395421, -1.19961, 0.157582, 0.318969, 0.006768, 0.636759, 0.008756, 0.813026, -0.250516, -0.112099, -0.204477, -0.166785, 0.327364, -0.263305, 0.424814, 0.449175, 0.326461, -1.221841, 0.346815, 0.06679, -0.936462, -0.215906, 0.109268, -0.278279, -0.138103, -0.217475, 0.133364, -0.597924, -1.653389, -0.290508, 0.130159, -0.820787, 0.109831, -0.864196, 0.02494, 0.180678, 0.299743, 0.236603, -0.097564, 0.179024, 0.270796, -0.432664, 0.13333, 0.062865, 0.817381, -0.089073, -0.288659], [-0.467936, 1.08514, 0.082257, 0.48592, 0.918791, 0.473451, 1.100283, -0.581098, -1.093393, -0.517863, -0.416051, 0.282136, -0.149156, -0.197785, 0.289783, -0.722791, 0.027739, -0.322485, 0.414803, -0.052496, 0.211189, -0.13058, 0.380493, -0.557474, -0.060447, 0.223584, -0.31499, -0.294229, 0.715431, -0.341042, -0.080594, -2.064394, 0.377551, -0.523842, 0.107818, 0.035589, -0.18176, -0.333445, 0.087395, 0.037797, -0.379137, 0.18691, -0.451807, -0.629542, 0.385905, 0.020708, 0.465977, -0.971534], [0.855355, -0.02747, 0.427725, 0.153594, 0.222864, 0.171776, 0.023893, 0.439853, -0.165358, -0.27676, 0.106536, -0.542075, -0.177998, 0.486969, -0.2312, 0.176583, -0.352988, -0.168148, -0.243571, 0.128757, 0.169631, 0.334922, -0.289086, 0.03975, 0.189967, 0.305281, 0.327273, 0.358949, -0.254646, -0.136555, 0.026356, -0.128542, -0.059923, 0.200443, -0.402896, 0.043037, 0.238864, -0.173111, 0.430691, -0.222421, 0.274869, 0.237045, -0.259508, 0.186054, -0.123172, -0.14131, 0.137634, 0.29298], [0.277006, 0.010212, 0.207593, -0.047831, -0.124402, -0.181722, -0.011499, 0.285695, -0.099575, -0.380757, -0.076744, -1.043113, -0.199612, 0.456672, -0.090402, 0.153334, -0.094359, -0.018458, 0.088034, 0.180283, 0.15349, 0.178437, -0.009399, -0.041772, 0.238395, 0.097702, -0.011951, 0.133614, 0.099957, 0.055364, 0.040793, -0.043304, 0.204945, 0.049376, -0.187371, -0.066253, -0.140537, -0.328227, 0.196534, -0.115423, -0.046569, -0.041641, 0.037407, 0.184281, -0.125797, 0.1794, 0.097975, 0.145208], [0.136802, -0.146026, 0.424383, 0.156104, 0.078815, -0.135063, 0.098265, 0.088287, -0.381082, -0.259593, -0.232801, -0.445556, -0.381883, 0.853946, -0.033538, 0.175909, -0.254378, -0.43444, -0.251338, -0.206349, 0.148404, 0.410024, -0.033701, -0.301963, 0.121334, 0.328039, -0.024311, 0.054465, 0.089312, -0.081347, 0.022265, 0.114985, 0.085637, 0.114031, 0.24973, -0.22882, 0.133824, 0.038988, 0.096685, -0.254821, 0.013222, 0.105519, -0.393133, -0.18062, 0.063127, -0.027433, -0.003332, -0.402575], [0.199461, 0.136704, -0.657688, 0.222379, 0.617004, 0.421721, -0.180704, 0.038741, -0.370203, 0.241382, 0.069633, 0.002968, -0.074462, 0.502461, -0.515114, 0.071191, 0.291622, -0.177464, 0.016777, 0.84014, -0.038347, -0.105927, 0.071694, 0.123788, 0.264543, -0.571286, 0.1125, -0.224858, -0.111227, -0.002543, 0.193815, -0.310199, -0.171789, 0.120101, 0.195482, -0.580164, 0.323804, -0.363352, -0.013203, 0.435024, -0.476509, -0.179862, -0.069321, -0.351689, 0.583422, 0.215489, -0.343411, 0.18828], [-0.061472, -0.545001, 0.097164, -0.148622, -0.403721, 0.681324, -0.236496, 0.032959, -0.441683, 0.075886, -0.768594, 0.357233, 0.315312, 0.013095, 0.12477, -0.103041, -0.251021, 0.301232, -0.264603, -0.048235, 0.187347, -0.315802, -0.515249, -0.169646, -0.208017, -0.360256, 0.232114, -0.336039, -0.376431, 0.345115, -0.331174, -0.261709, -0.016991, -0.224685, -0.119617, -0.076946, 0.179956, 0.017569, 0.090909, -0.176083, -0.094659, 0.17537, 0.084328, -0.184041, -0.1566, -0.239284, 0.110299, -0.060964], [0.293755, -0.792504, -0.149742, -0.125478, -0.235851, 0.198933, -0.278648, 0.323385, -0.28329, -0.256687, -0.566862, 0.302278, 0.154172, -0.011144, -0.037373, -0.0249, -0.442576, 0.076822, -0.099783, -0.341711, -0.281036, -0.331319, -0.136031, -0.127986, -0.048341, 0.181513, 0.991789, 0.12623, 0.201399, -0.031808, 0.262606, -0.849125, -0.013037, -0.222368, -0.049284, -0.005807, 0.223716, 0.108584, -0.121113, 0.040457, 0.244362, 0.066337, 0.053165, -0.231332, 0.029491, 0.019139, 0.143099, 0.206507], [-0.357742, 0.152853, -0.030213, -0.43241, -0.184043, 0.759404, 0.435572, -0.586534, 0.220263, 0.404295, 0.169707, -0.102958, -0.005521, 0.296197, 0.24587, 0.033602, -0.010796, 0.570597, 0.054247, 0.164682, 0.015328, -0.017625, 0.371829, 0.320918, 0.23737, 0.057715, -0.267057, -0.131282, -0.060247, -0.28865, -0.243627, 0.375552, -0.048529, 0.134839, 0.189881, 0.113075, -0.074256, -0.212912, 0.476337, 0.250953, -0.101901, 0.324678, 0.151383, -0.24832, -0.065169, 0.150349, -0.240658, 0.064891]]
SEQ_BI = [0.100069, -0.152742, 0.135199, 0.004396, -0.18999, -0.063006, -0.063055, 0.209438, 0.08078, 0.074633, 0.206301, 0.08958, 0.076378, 0.027777, 0.148329, 0.027759, 0.539049, -0.24999, 0.525809, 0.646562, 0.345056, 0.539162, 0.889814, 0.402421, 0.522315, 0.553171, 0.37217, 0.609716, 0.764187, 0.480923, 0.339942, 0.723079, 0.064314, -0.16295, 0.064112, 0.078435, 0.327602, -0.156434, -0.212309, 0.410919, 0.00895, 0.108248, 0.397971, 0.214209, -0.165739, -0.017677, 0.346552, -0.054753]
SEQ_WH = [[-0.062757, 0.596908, -0.026907, -0.37703, 0.472366, -0.000619, 0.514781, -0.522288, 0.049501, 0.203558, -0.66208, -0.282574, 0.143447, -0.198567, 0.030675, -1.129698, 0.375336, -0.195013, 0.152178, 0.416052, -0.391847, -0.003187, -0.027786, -0.153036, 0.190087, 0.095286, -0.373331, 0.586594, 0.058982, 0.536651, 0.606217, -0.007234, 0.632903, 0.212974, 0.259488, 0.218602, -0.265847, 0.299575, -0.073121, 0.549932, 0.185476, 0.236906, -0.527217, -0.180377, -0.192917, -0.093026, 0.190039, -0.396823], [0.050253, -0.349252, 0.005383, -0.429513, 0.240167, -0.13914, -0.567869, -0.030674, -0.188197, 0.28404, -0.464696, 0.166107, -0.107788, 0.302582, -0.569722, -0.009751, -0.704045, -0.135742, -0.843616, -0.043405, -0.670643, -1.136906, -0.602713, 0.129646, -0.214801, -0.540726, -0.4032, -0.844288, -1.067531, 0.134114, -0.253165, -1.081876, -0.328439, 0.460168, -0.298922, 0.303156, -0.092592, -0.212307, -0.108986, 0.009992, 0.113319, 0.09585, -0.156265, 0.10152, -0.090926, -0.133698, -0.179938, -0.193212], [-0.421111, -0.200031, -0.473023, 0.061552, 0.22032, -0.072235, 0.387822, -0.551804, 0.535967, -0.092036, 0.175458, -0.428141, -0.076158, 0.029452, -0.186049, -0.001369, -0.873924, -0.775812, -0.147246, -0.228053, 0.657908, 0.493084, 0.152288, -0.467561, -0.333144, -0.614984, 0.30259, -0.581946, -0.613686, 0.829056, -0.189899, -0.134101, 0.000232, 0.450484, 0.527025, 0.373268, 0.012107, -0.199321, 0.011954, 0.471422, -0.288363, -0.3839, 0.339068, -0.183041, 0.300034, -0.195226, -0.244375, -0.014302], [-0.004853, 0.59423, 0.136709, -0.285359, 0.126438, 0.070549, 0.022077, -0.3733, 0.493312, 0.352472, -0.390112, -0.056865, -0.035848, -0.322403, -0.025675, -0.225976, 0.132523, -0.21975, -0.079198, 0.221588, -0.086266, 0.354895, -0.012236, -0.16837, -0.652186, -0.206838, 0.030287, 0.230855, -0.619968, 0.440591, 0.118572, -0.116922, 0.129674, 0.281639, 0.067055, 0.507729, -0.361063, 0.201005, 0.190367, 0.1243, -0.006271, -0.082315, -0.168548, -0.301316, -0.028316, 0.312776, 0.159471, -0.524448], [-0.508041, -0.257442, -0.561465, 0.224381, 0.139658, -0.224681, -0.196126, 0.099989, -0.357353, 0.011432, -0.213574, -0.112092, -0.183882, -0.038687, 0.336609, -0.566928, 0.450737, 0.100984, 0.60069, 0.330774, 0.216109, 0.05119, 0.344071, -0.246648, 0.353365, -0.25779, 0.397259, 0.523183, 1.270623, 0.463751, 0.638602, -0.099624, 0.07009, -0.245452, 0.00104, -0.236563, 0.663685, -0.480435, -0.353364, -0.013822, -0.643091, 0.293996, -0.077169, -0.014118, 0.177214, 0.526802, 0.19476, -0.405778], [0.848276, 0.368548, 0.237627, 0.129554, -0.160537, 0.098518, 0.115434, -0.148206, -0.035838, 0.056471, -0.194201, 0.517259, 0.338508, 0.093264, -0.040654, 0.018164, 0.924208, 0.268338, 0.319283, 0.267037, -0.064869, -0.432877, -0.35808, 0.216018, -0.092482, -0.034193, 0.453511, 0.264234, 0.503455, -0.067692, 0.272437, 0.68584, 0.044227, -0.096158, -0.384258, -0.13643, -0.256652, 0.726505, 0.253198, 0.020726, 0.469014, -0.026861, -0.236967, -0.228961, -0.03544, 0.36896, 0.244587, -0.002906], [0.172631, -0.168437, -0.131489, -0.48764, 0.224803, 0.101996, 0.16392, -0.371872, 0.040518, 0.006132, -0.507055, 0.615506, 0.052807, -0.246967, -0.099679, 0.280886, -0.304326, -0.342471, -0.605552, -0.345396, -0.058711, -0.570743, -0.487271, 0.029324, -0.107185, -0.474679, -0.303358, -0.005636, -0.541494, -0.009293, -0.416771, -0.342475, -0.088166, 0.410869, 0.170927, 0.079457, -0.133068, 0.291588, 0.356162, -0.460094, 0.469102, -0.097184, -0.063342, -0.232665, -0.027176, 0.121066, 0.01004, 0.190713], [-0.176624, -0.048576, -0.139234, -0.004212, 0.340913, -0.129123, 0.637283, -0.236294, -0.345872, 0.37448, -0.181216, -0.727354, -0.0582, -0.215023, 0.144917, -1.067783, 0.203088, -0.032422, 0.54521, 0.364182, 0.029094, 0.747763, 0.843449, -0.289599, 0.645122, 0.337033, 0.765247, 0.579046, -0.197509, 0.658304, 0.652309, -0.080451, 0.356523, 0.117104, 0.238394, 0.193485, 0.334997, -0.006658, -0.21836, 0.252014, -0.245321, 0.087982, -0.110079, 0.034137, 0.280284, -0.047551, 0.207978, -0.172242], [0.585808, 0.064949, 0.314784, -0.730421, -0.246126, 0.171342, 0.301246, 0.290183, 0.071987, 0.083364, -0.006784, 0.27925, 0.131099, -0.714406, -0.221527, 0.093873, -0.59083, 0.211043, 0.246445, 0.340478, -0.290666, 0.411933, 0.277396, 0.162957, 0.538247, 0.378399, 0.641368, 0.13576, -0.529057, 0.21242, 0.942663, -0.023237, -0.26312, 0.162992, 0.33488, 0.433774, -0.52876, 0.588889, 0.199311, -0.15918, 0.124743, -0.223109, -0.159386, 0.415101, 0.034201, -0.022101, -0.090691, 0.148398], [-0.180547, 0.122939, 0.052737, -0.259899, -0.227978, -0.068767, -0.551743, -0.663149, -0.936553, 0.405275, -0.593568, 0.277541, -0.46758, -0.056311, -0.252733, -0.12321, 0.339507, 0.456348, -0.136218, -0.053034, -0.57065, -0.179491, 0.028984, -0.050055, -0.242401, -0.129962, 0.438476, -0.016639, 0.523588, 0.153624, 0.073819, 0.188997, 0.348628, 0.348907, -0.360363, 0.405467, 0.387453, 0.45776, 0.468421, -0.009062, 0.149368, 0.502722, -0.437323, 0.058906, 0.211234, 0.38087, -0.075674, 0.35582], [-0.186255, -0.012479, -0.099885, 0.149451, -0.018451, 0.059309, -0.031884, 0.122998, 0.33793, 0.233603, 0.198959, -0.120846, 0.077933, 0.026565, 0.253942, 0.357982, 0.164578, -0.278642, 0.041417, 0.932642, 1.323011, 0.951669, 1.032804, 0.147303, 0.707846, 0.723623, 0.303569, 0.361373, 0.647752, 0.65799, 0.449311, -0.063555, 0.162547, -0.510451, 0.272585, -0.013362, 0.14605, -0.077665, 0.176018, 0.072912, -0.123805, 0.071409, 0.828989, 0.129261, -0.042726, -0.174461, 0.299464, 0.18849], [-0.373348, -0.287495, -0.677501, -0.279461, 0.030496, -1.164215, -0.646187, 0.035414, 0.524551, -0.584257, 0.459628, -0.720872, 0.061411, -0.56999, -0.278426, 0.041827, 0.111683, -0.237094, 0.132649, 0.145649, 0.892825, -0.286885, -0.372357, 0.052861, -0.402354, 0.409619, -0.266839, -0.07229, -0.351787, 0.074493, -0.38207, 0.107065, -0.574639, 0.469286, -0.284245, -0.104192, 0.121447, -0.087165, -0.133972, -0.169384, 0.555052, -0.393919, 0.272541, 0.219855, -0.196819, 0.340096, 0.153887, 0.495843], [-0.125524, 0.089643, -0.139444, 0.684918, 0.53245, -0.294419, 0.450821, 0.194286, 0.12557, -0.092283, 0.421995, 0.204412, -0.321545, 0.237776, -0.564171, -0.141103, -0.075124, 0.102728, -0.622288, -0.106143, 0.089841, -0.408272, -0.329749, -0.058829, 0.164829, -0.974027, -0.215332, -0.747256, -0.099623, -0.184318, 0.120612, -0.96427, -0.382595, -0.256952, 0.190826, -0.574621, -0.029627, -0.189144, -0.585539, -0.392762, -0.43285, 0.260827, -0.294975, -0.051345, 0.589058, 0.081395, -0.192232, 0.165077], [-0.177895, 0.215803, -0.460626, 0.128589, 0.21355, -0.128132, 0.28661, -0.168423, -0.421633, 0.121245, 0.444658, -0.120359, -0.73419, 0.065422, 0.328676, -0.063438, 0.811341, 0.452993, 0.485011, 0.358766, -0.021931, 0.160095, 0.747414, 0.390475, 0.161603, -0.301153, 0.066956, 0.382748, 0.644812, -0.111355, 0.529029, 0.015389, 0.166459, -0.421465, -0.392249, -0.189889, 0.47973, -0.193832, -0.32331, -0.160947, -0.285638, 0.492646, -0.122918, 0.151125, 0.072061, 0.905448, 0.289571, 0.014772], [0.137333, -0.045053, 0.029624, -0.173984, 0.16591, 0.034563, 0.173631, 0.010837, 0.147541, -0.126902, -0.028543, -0.065158, 0.209705, -0.384752, 0.445795, -0.091797, 0.577327, -0.130487, 0.554887, 0.704024, 0.450029, 0.10151, 0.149173, 0.06287, 0.913443, 0.850972, 0.096958, 0.541325, 0.355377, 0.408676, 0.580598, 0.51657, -0.001408, -0.291194, 0.202068, -0.130889, 0.088262, 0.379564, 0.237897, 0.155549, -0.04228, 0.280274, 0.136612, -0.043381, 0.012501, 0.033927, 0.52981, -0.051053], [0.371492, -0.439418, -0.472905, 0.303203, 0.184111, -0.629967, 0.046491, 0.090266, 0.495316, -0.643121, -0.013383, 0.130949, -0.070798, -0.143692, -0.238014, 0.533894, 0.183778, -0.262074, -0.401355, 0.817947, 0.223604, -0.333643, -0.455924, 0.601211, -0.011446, -0.618529, -0.218283, -0.156549, -1.014168, 0.046009, -0.178633, -0.411892, -0.179898, 0.171519, -0.054201, -0.190821, -0.123738, 0.041042, 0.510024, -0.338149, 0.00042, -0.26565, 0.661867, 0.408833, 0.457886, 0.064868, -0.473513, 0.48343]]
SEQ_BH = [-0.037325, 0.231197, -0.06545, 0.303721, 0.111861, 0.101955, 0.0974, 0.036454, -0.26108, 0.226435, 0.076561, -0.283823, 0.064093, 0.080455, -0.004294, -0.093892, 0.458521, 0.013529, 0.432742, 0.614034, 0.605419, 0.76906, 0.847488, 0.350393, 0.65197, 0.654116, 0.562806, 0.242514, 0.781909, 0.610012, 0.084274, 0.860178, 0.155889, -0.368195, -0.035358, -0.215894, -0.15861, 0.199366, -0.203307, 0.338327, 0.246687, 0.087302, 0.36932, -0.077323, -0.206973, -0.017805, 0.427157, 0.053942]
SEQ_HEAD = [
    ([[-0.022401, -0.086818, -0.166516, 0.017226, -0.338764, -0.008527, 0.073264, 0.280138, -0.106125, -0.055424, 0.040975, 0.017651, 0.014255, -0.003534, -0.126571, -0.048758, -0.137132, 0.127549, 0.148966, 0.21569, 0.083184, 0.077286, -0.000239, -0.057587, 0.053542, -0.057591, -0.0013, 0.058775, -0.034608, 0.025089, -0.198967, 0.00717, 0.082805, -0.274491, 0.102131, -0.132968, -0.180473, 0.158258, 0.159635, -0.100961, -0.165953, 0.015353, -0.463996, -0.079571, 0.215039, 0.003043, 0.235226, 0.022568, 0.072022, 0.208403, -0.036952, -0.111467, -0.075734, 0.03228, 0.084393, -0.247761, -0.038634, 0.027502, 0.05444, -0.229784, -0.144677, -0.13026, -0.140591, -0.128749], [-0.043947, 0.170433, 0.126789, 0.148737, -0.082517, -0.045117, 0.104748, 0.219447, 0.071935, 0.016401, 0.053475, -0.085557, 0.127086, 0.03322, -0.033096, 0.150827, -0.257697, 0.044952, -0.030441, 0.026593, -0.065161, 0.0402, 0.153432, 0.05378, -0.014323, -0.139202, 0.156149, -0.126512, 0.053566, -0.128336, -0.179831, 0.07976, 0.119902, -0.077807, -0.036144, 0.146399, 0.056623, -0.050791, -0.044896, -0.090925, 0.080181, -0.061773, -0.082253, 0.052572, -0.062827, 0.013867, -0.008888, -0.021776, 0.028601, 0.18941, -0.076659, 0.021127, 0.011187, -0.112261, 0.144561, -0.177075, -0.270706, -0.151383, -0.114878, 0.188542, 0.068932, -0.062081, -0.115396, -0.12019], [0.097276, -0.075059, -0.031881, 0.120946, 0.028066, -0.030585, -0.039372, 0.037343, 0.001169, -0.024302, -0.141081, -0.16441, -0.006304, 0.141434, 0.082253, 0.002422, -0.084196, 0.17101, -0.068464, 0.185218, 0.06956, 0.064034, -0.078411, -0.157665, 0.04465, -0.140253, -0.069328, -0.000815, 0.047882, -0.156492, 0.146933, 0.038805, 0.097657, 0.039682, -0.052364, -0.114262, 0.112601, 0.059888, 0.131882, 0.025252, 0.098352, 0.095905, -0.182152, -0.109476, -0.109379, -0.097754, -0.083949, 0.156733, -0.063842, 0.108902, -0.129378, -0.106393, 0.124804, -0.042636, 0.132379, -0.110076, -0.232974, -0.127894, 0.015216, 0.072291, 0.092922, 0.155054, -0.086689, 0.153105], [-0.045101, -0.221417, 0.179991, -0.174934, 0.162087, 0.030151, -0.194911, -0.10135, 0.015826, -0.176406, -0.112119, 0.048937, -0.146271, 0.116704, -0.070242, -0.090219, 0.21184, 0.079215, -0.015628, -0.089954, 0.180963, 0.04291, 0.055212, -0.068999, -0.266566, 0.041914, -0.116444, -0.033642, 0.02193, 0.056431, 0.032679, -0.058443, -0.079058, 0.321167, -0.060645, -0.049094, -0.161596, -0.030144, 0.077043, -0.180667, 0.096919, 0.12346, 0.047109, -0.064748, -0.310876, 0.102432, -0.177396, -0.020902, -0.127073, -0.047461, 0.248312, -0.019004, -0.088528, -0.159399, -0.250582, -0.122993, 0.20506, 0.133846, 0.054339, -0.000742, -0.107624, 0.18478, 0.108558, 0.035365], [0.240818, 0.152988, 0.05105, -0.200489, 0.006976, -0.168657, 0.07669, -0.175611, -0.120428, 0.151622, 0.008519, 0.344366, -0.380092, 0.270726, 0.095844, -0.016051, 0.349045, 0.302583, 0.069202, 0.135432, 0.173237, -0.229485, 0.117064, 0.393307, -0.040787, -0.259048, -0.243125, -0.056248, 0.150569, -0.133108, 0.287843, 0.147475, -0.018127, 0.112276, -0.171413, 0.243344, -0.213349, 0.003921, 0.073808, -0.213114, -0.18022, 0.261985, 0.23688, -0.080577, -0.139473, -0.035744, 0.144104, -0.289349, -0.213598, -0.000554, 0.071982, 0.263706, -0.160571, -0.220751, -0.182481, 6.3e-05, 0.284526, -0.115586, -0.072563, -0.035662, 0.21463, -0.188951, -0.078499, 0.155401], [-0.157366, 0.055742, 0.088605, -0.023651, 0.024647, -0.092357, 0.082874, 0.147405, 0.005699, 0.07498, -0.108154, -0.143303, -0.161933, -0.047086, 0.146202, 0.080088, 0.03317, -0.108075, -0.091071, 0.131189, -0.032308, -0.106288, -0.009787, 0.11739, -0.128649, 0.061298, 0.051235, 0.048722, -0.087955, -0.154259, 0.093834, -0.060448, -0.085986, 0.070607, 0.035765, -0.050688, -0.084952, 0.086735, -0.143813, 0.114152, 0.123268, -0.078457, 0.126306, -0.154482, -0.179547, -0.050899, -0.192229, -0.002155, -0.102182, 0.004538, 0.014482, -0.052397, -0.020039, 0.111688, 0.066698, -0.088997, -0.279325, 0.020029, 0.05413, 0.169793, 0.056466, 0.145721, -0.085768, -0.128554], [-0.050962, 0.014622, -0.0524, 0.134856, -0.196653, -0.060883, -0.004498, 0.187212, -0.058824, 0.101433, 0.004973, 0.049109, -0.169102, 0.032177, -0.257546, 0.174553, -0.322516, -0.011869, 0.12266, 0.157601, 0.112421, -0.089339, 0.134188, 0.430693, -0.146456, -0.0771, -0.170896, -0.043442, 0.111815, -0.121025, 0.084245, 0.176135, 0.090723, -0.034744, -0.018618, 0.175073, 0.069462, 0.083017, -0.044657, -0.044722, -0.243547, 0.165242, -0.128594, 0.0333, -0.161111, 0.117376, 0.2923, -0.176203, -0.112699, 0.178746, 0.044436, -0.172581, -0.219918, 0.004567, -0.161003, -0.10379, 0.072615, -0.221418, 0.100411, 0.161398, -0.130113, 0.185826, -0.11951, -0.123171], [0.032449, 0.006768, 0.133277, 0.116506, 0.223399, -0.042804, -0.077862, -0.043228, -0.082981, 0.016104, 0.133481, -0.219576, -0.058985, -0.242979, -0.096978, -0.158808, 0.024744, -0.065537, 0.023332, 0.024554, 0.045071, -0.206156, -0.007883, 0.085396, 0.293545, 0.094664, 0.165137, -0.082148, 0.253072, 0.049763, -0.032188, 0.138064, -0.022632, -0.018063, -0.013232, 0.025403, -0.066138, -0.168674, 0.093589, 0.153428, 0.111444, 0.122756, 0.025717, -0.219398, -0.21371, 0.252501, -0.081741, -0.099833, 0.104488, 0.095089, -0.022762, -0.096414, -0.090123, 0.073079, 0.037021, 0.104866, 0.082598, -0.043173, -0.202318, -0.048485, 0.158835, 0.160668, -0.111287, -0.096373], [0.012492, -0.034494, 0.174933, 0.063224, -0.077045, 0.101144, -0.128896, 0.159561, -0.06266, 0.171043, 0.073317, 0.127476, -0.000475, -0.016503, 0.377745, -0.165872, -0.034735, 0.192137, 0.082015, 0.052699, 0.110058, 0.009343, 0.238348, 0.128295, 0.099299, -0.209969, -0.067409, 0.069203, 0.233085, -0.147218, 0.008425, 0.042835, -0.125055, 0.071235, 0.072178, -0.271814, -0.160402, -0.263254, 0.229017, -0.096558, -0.04534, -0.064372, -0.104921, -0.1169, 0.211009, 0.249559, 0.095673, -0.078357, 0.116894, 0.177748, -0.072197, -0.151654, 0.044503, 0.038113, -0.144609, -0.012087, 0.029775, 0.048715, -0.182327, -0.323072, 0.334512, -0.098638, 0.151847, 0.177505], [0.053373, 0.001103, 0.051737, -0.016053, 0.040603, 0.108198, 0.096299, -0.097577, 0.001447, -0.073902, -0.225414, 0.10658, 0.102421, 0.071114, 0.009527, -0.164885, -0.222221, -0.216979, 0.096496, -0.119624, -0.211233, -0.141254, -0.011426, -0.008427, -0.099687, -0.10699, -0.01028, -0.101106, -0.009091, -0.167982, 0.168156, -0.115368, -0.388246, 0.012911, 0.06512, 0.015499, -0.04661, 0.052781, 0.090234, -0.01245, -0.079705, -0.055994, 0.12847, -0.257294, 0.183777, 0.033802, -0.066632, 0.145088, 0.088277, 0.046892, 0.058887, 0.14276, 0.03067, 0.130972, 0.209287, -0.036718, -0.054378, 0.14993, 0.032985, 0.052386, 0.147673, 0.100104, -0.134019, -0.062499], [-0.116873, 0.076397, 0.061106, -0.004036, 0.22004, -0.012575, -0.130515, -0.120633, -0.051884, -0.058528, -0.118152, -0.016042, 0.068959, -0.035239, 0.016749, 0.009207, -0.12902, 0.151726, -0.268755, 0.048642, 0.061039, 0.080471, -0.084989, -0.002792, -0.002764, -0.038312, 0.099628, 0.048696, 0.043345, -0.029916, -0.17486, 0.021537, 0.109119, 0.108962, 0.114728, -0.075918, -0.144078, -0.026182, 0.020974, -0.097311, -0.117215, -0.014417, 0.01264, -0.009085, -0.120581, 0.100829, 0.050854, 0.020197, 0.104511, -0.022364, -0.044867, -0.153536, 0.113939, 0.12918, 0.095807, -0.006411, 0.064135, -0.144264, -0.004397, 0.178037, -0.100352, -0.004939, -0.187338, 0.04676], [0.185439, -0.065495, 0.079407, 0.020497, -0.237672, 0.278556, 0.027624, -0.037842, -0.208114, 0.176567, -0.224462, 0.02394, 0.204528, 0.197884, 0.192116, -0.223093, 0.141505, 0.034971, 0.05576, 0.247077, -0.10421, 0.091255, 0.094847, -0.115165, -0.061671, -0.115634, -0.0683, 0.108814, 0.018378, 0.005586, 0.178872, -0.095272, 0.322463, 0.196056, 0.185831, 0.076547, -0.108314, 0.04878, 0.200016, -0.190382, -0.098012, 0.072077, 0.210933, 0.00741, -0.108761, 0.164403, -0.191682, -0.221971, -0.086769, 0.162848, -0.122808, -0.160718, 0.175399, -0.241315, -0.138518, -0.310064, 0.057345, 0.087775, -0.444385, -0.172133, -0.02195, 0.140704, -0.182928, 0.08251], [0.109161, -0.184956, 0.066749, -0.152345, 0.16087, 0.07196, -0.028127, 0.179376, -0.007096, 0.054802, -0.152029, 0.001261, -0.177351, 0.240991, 0.170865, -0.255003, 0.073952, -0.190598, -0.019841, 0.018337, 0.196639, 0.006725, 0.007569, -0.043581, 0.05875, 0.110609, -0.142957, 0.035513, -0.00653, 0.009587, -0.021112, -0.232394, 0.033943, -0.21064, 0.263068, 0.247533, 0.03198, -0.000656, 0.126573, -0.154447, -0.164472, -0.006976, -0.139327, 0.283971, -0.06543, 0.200043, 0.031406, 0.106607, -0.158397, -0.029442, 0.096208, 0.200048, -0.090473, -0.100008, 0.026353, 0.113253, -0.178273, -0.1527, 0.181689, 0.105718, -0.093473, -0.116254, 0.150515, -0.222046], [0.002734, -0.093501, -0.02929, -0.054129, -0.013894, 0.129953, -0.158307, 0.062407, -0.091789, -0.259164, -0.254174, -0.071958, 0.126505, -0.049184, 0.186164, -0.127255, 0.069214, -0.059573, -0.078968, -0.079102, 0.027635, 0.038319, 0.156899, -0.064181, 0.163055, -0.178944, -0.05154, -0.209693, 0.151181, 0.100645, 0.095684, -0.131693, -0.12391, 0.040478, 0.11129, 0.073581, 0.163612, -0.133276, 0.153519, 0.076187, -0.043404, 0.062022, 0.034755, 0.082066, 0.04988, 0.167258, -0.027623, -0.234817, -0.177026, 0.099031, 0.097646, -0.003618, -0.185935, 0.067545, -0.031957, 0.153597, 0.104537, -0.076978, 0.371392, 0.023244, 0.232178, -0.021961, 0.10685, 0.015706], [-0.163085, 0.149615, 0.11778, 0.034275, 0.09663, -0.199959, -0.057857, 0.003417, -0.072094, 0.201257, -0.033827, -0.217652, -0.077177, -0.322691, 0.011783, 0.016044, 0.027568, -0.012692, -0.109489, -0.050185, -0.083823, 0.050401, 0.008215, -0.003447, -0.244207, -0.061752, 0.059637, -0.146941, -0.037572, 0.073619, -0.101055, 0.010099, -0.131592, 0.201585, 0.080034, 0.174209, 0.073728, -0.075981, 0.043917, -0.03275, -0.121163, -0.272617, -0.069404, -0.072921, 0.034616, -0.143154, -0.15655, 0.076538, -0.000529, 0.023034, -0.002165, 0.133719, -0.163043, 0.028044, 0.099822, -0.116959, 0.150253, 0.199714, 0.373209, -0.045367, 0.076248, 0.076134, 0.001397, -0.165896], [0.054582, 0.184508, -0.259963, -0.016806, 0.118227, -0.170323, 0.238452, 0.08129, -0.075315, -0.000968, -0.107162, 0.003445, -0.206626, -0.109123, -0.073545, -0.177621, -0.01362, -0.171869, 0.048345, 0.06253, 0.121768, 0.10151, -0.133372, 0.048385, 0.05342, 0.080297, 0.11455, -0.098238, 0.108409, -0.062707, 0.130339, -0.000796, -0.181654, -0.165223, 0.102269, -0.165684, -0.042761, -0.083824, -0.031865, 0.169933, 0.104685, -0.194605, 0.021318, -0.175914, 0.204324, -0.111978, -0.089674, 0.127187, -0.036714, -0.040049, -0.326153, 0.018022, -0.233574, -0.063097, -0.196428, 0.094646, -0.192552, -0.138361, -0.029712, 0.080415, 0.006769, -0.206409, 0.03563, 0.036868], [-0.094521, 0.172746, 0.2393, -0.217688, -0.349714, 0.016812, -0.061705, -0.329556, -0.006639, -0.199825, -0.24595, -0.07073, -0.216741, 0.126814, -0.192941, 0.204038, 0.29062, -0.059962, 0.138302, 0.114918, -0.327744, 0.300079, 0.004024, 0.041322, 0.306348, -0.044691, 0.003846, 0.002039, 0.024343, 0.070631, -0.024315, -0.150137, -0.048395, -0.201015, -0.029438, -0.283115, -0.15037, 0.356184, 0.151765, 0.053959, 0.276775, -0.276194, 0.10207, -0.097444, -0.023782, 0.260805, 0.270824, -0.077746, 0.047133, 0.153455, -0.258437, 0.260042, 0.169505, -0.005266, -0.117421, -0.010969, -0.07739, -0.026874, -0.0124, -0.259153, -0.024964, -0.169468, -0.403999, -0.24726], [-0.054334, -0.128319, 0.255671, -0.18805, -0.27256, 0.09809, 0.026198, -0.311352, 0.113627, -0.0419, -0.122133, -0.057416, 0.190838, -0.050476, 0.186498, 0.018307, 0.031088, 0.150489, 0.100253, 0.146465, 0.191617, 0.072888, 0.037935, 0.016182, 0.089582, -0.020572, 0.000571, -0.012484, 0.040505, -0.317947, -0.071655, -0.136195, 0.012234, 0.168898, -0.126074, -0.079047, -0.048958, 0.019009, -0.158077, -0.056125, 0.244028, 0.071188, -0.017995, -0.073827, -0.175905, -0.087718, 0.033096, 0.14645, 0.033808, 0.006599, 0.100921, 0.132703, 0.185581, 0.147442, 0.145002, -0.010134, 0.048183, -0.167482, 0.011758, -0.145139, 0.042807, 0.140813, 0.022856, 0.085871], [0.156094, 0.086468, 0.047472, -0.036771, -0.048631, -0.1553, 0.099299, -0.001254, 0.00958, 0.017682, 0.121038, 0.09687, -0.033999, 0.098436, 0.142107, 0.049144, -0.050253, -0.021698, 0.19188, 0.024466, 0.136642, -0.12553, 0.012382, 0.071166, 0.048652, 0.146217, -0.018272, -0.002987, 0.128016, 0.139985, -0.149857, 0.067696, -0.032581, -0.134323, -0.099276, -0.054556, 0.08286, 0.006646, 0.060263, 0.150913, -0.043978, 0.025898, -0.017552, 0.051987, 0.093856, -9.2e-05, -0.160885, 0.013883, -0.157305, -0.141299, -0.114223, 0.013693, 0.009213, 0.101702, -0.21926, 0.095341, -0.095698, -0.03198, 0.04621, -0.011999, 0.127642, -0.047462, 0.028399, -0.005671], [0.594305, 0.205441, -0.071563, 0.128111, -0.338529, -0.194774, 0.042309, -0.20752, -0.164656, 0.071178, 0.092728, 0.258025, 0.065251, -0.086473, -0.355149, -0.044259, 0.105251, -0.039074, -0.13847, 0.064644, 0.059146, -0.265562, -0.316689, 0.033433, -0.045811, -0.054298, -0.07997, -0.094139, 0.163103, 0.056074, -0.018333, -0.047851, 0.128737, -0.204351, 0.021595, 0.229723, -0.076793, 0.051519, 0.06993, 0.104968, 0.00731, 0.204593, -0.144995, 0.20975, -0.120108, -0.084201, 0.291064, 0.015472, -0.288686, 0.051529, 0.071918, -0.027075, -0.135378, 0.038681, 0.279876, 0.077233, -0.245407, 0.10944, -0.206615, 0.111067, -0.040479, 0.295405, -0.097463, -0.105508], [0.095691, 0.142956, -0.020202, 0.45236, 0.301627, 0.135872, -0.113301, -0.026077, -0.271909, 0.10207, 0.09781, -0.159374, 0.026825, 0.122691, -0.095615, -0.006125, -0.00839, -0.049007, -0.164545, -0.059405, -0.075537, 0.083741, -0.160557, -0.41273, 0.098968, -0.152002, 0.314747, 0.134219, 0.157623, 0.101465, 0.056203, -0.169669, 0.049236, 0.265984, -0.001458, 0.053521, -0.120515, 0.098842, 0.116926, -0.092235, 0.005124, -0.296369, -0.129664, 0.236063, 0.143301, -0.279136, -0.229978, -0.189596, 0.042109, -0.089253, -0.095254, 0.049073, 0.080819, 0.100479, -0.291652, 0.055405, -0.002407, 0.305944, 0.118865, -0.161063, 0.133944, -0.213277, -0.038975, -0.094815], [-0.105496, -0.019887, 0.129204, 0.186655, 0.07165, -0.028359, -0.099896, 0.018274, 0.009001, 0.336257, 0.065173, -2.3e-05, -0.032716, 0.129487, 0.05954, -0.000691, -0.061186, 0.041156, 0.075392, -0.023784, -0.421481, -0.194719, 0.509283, -0.048431, -0.363679, -0.02837, 0.037885, 0.062487, 0.20391, 0.257266, 0.03905, 0.222237, 0.114047, -0.223168, -0.111685, -0.127312, 0.326223, -0.281634, -0.028237, 0.02476, 0.416544, -0.038485, 0.08858, -0.126522, -0.008009, 0.137094, -0.088388, 0.050466, -0.081153, 0.166133, -0.076331, 0.050965, 0.214083, -0.003396, 0.100555, 0.080924, -0.20772, 0.236877, -0.089689, -0.072439, -0.03882, -0.040534, 0.019622, 0.188197], [-0.314557, 0.042736, 0.134488, -0.311648, -0.084501, -0.210928, -0.320652, 0.052152, 0.02395, -0.01473, 0.000635, -0.290147, -0.174438, -0.001889, -0.283405, -0.100012, -0.346597, -0.158467, -0.499468, -0.006221, 0.101448, -0.18455, 0.237285, -0.073839, 0.011034, 0.217321, -0.009845, 0.187953, 0.074251, 0.155578, 0.005689, -0.449383, 0.091161, 0.073976, -0.145835, 0.430722, 0.1619, 0.224534, -0.294555, 0.097564, 0.207339, 0.157644, -0.555003, 0.042925, -0.201867, 0.134219, -0.421737, 0.152549, -0.106107, -0.275477, 0.254385, 0.014754, 0.006411, 0.034276, -0.06295, -0.010146, 0.175732, 0.113205, -0.075224, 0.037992, -0.110647, -0.138634, 0.09572, 0.056816], [-0.126538, 0.208896, 0.113886, -0.03232, -0.114218, 0.007899, -0.10497, -0.059931, 0.137399, 0.244004, 0.018175, 0.147259, 0.226753, 0.182693, 0.382447, 0.068773, 0.009043, -0.014143, 0.213416, 0.1269, -0.140381, -0.034941, -0.159459, 0.035928, -0.018907, 0.15699, 0.100107, -0.083529, 0.004053, 0.129966, 0.080199, 0.081108, -0.032107, 0.07916, 0.144497, 0.201893, -0.001112, 0.003891, 0.125996, 0.010985, 0.072073, 0.203195, 0.025587, 0.093325, 0.036889, -0.117077, -0.101588, 0.010598, 0.175508, -0.009034, 0.052931, -0.019038, 0.225102, -0.034729, 0.22664, 0.149832, 0.060534, 0.019406, 0.009518, -0.06309, -0.040196, 0.011887, 0.144996, -0.00063], [-0.060746, -0.009537, -0.244391, -0.111715, 0.128884, -0.119338, -0.121959, 0.20937, 0.127644, 0.016662, 0.231538, 0.106713, -0.034198, 0.166805, -0.14239, 0.287262, 0.129459, 0.206006, -0.091883, 0.230362, 0.053526, -0.411679, -0.171487, 0.031328, -0.548563, 0.084068, 0.118324, 0.456995, 0.037385, -0.101034, 0.228061, 0.101288, -0.164465, -0.093195, -0.133818, -0.041854, 0.212093, 0.010109, 0.122402, 0.238002, 0.343734, -0.084924, 0.081561, -0.350917, -0.089748, 0.098638, 0.084665, 0.162979, 0.076651, 0.024359, 0.045562, 0.18913, 0.300884, 0.227334, 0.140856, 0.130284, 0.188943, 0.013406, 0.11533, 0.058606, 0.284287, 0.149022, 0.128183, 0.183279], [-0.213314, -0.005026, 0.218181, -0.228019, -0.23812, -0.362806, -0.406018, 0.56218, -0.012071, 0.050082, 0.123532, -0.139233, 0.035581, 0.178718, 0.010132, 0.229183, -0.199114, 0.061397, -0.374196, 0.091367, 0.100561, -0.21557, -0.028639, 0.036102, -0.296948, 0.351641, 0.115602, 0.426163, 0.50041, 0.373952, 0.127743, -0.164861, -0.063314, 0.07006, 0.151234, 0.048897, 0.387618, 0.012479, -0.262908, 0.326201, -0.131975, 0.191384, -0.230432, 0.140712, -0.309441, -0.27757, -0.338698, 0.274594, -0.120441, 0.12548, 0.193444, -0.073802, -0.097654, 0.023888, 0.350893, 0.118733, 0.193991, 0.117286, 0.328958, -0.035992, 0.345603, -0.129807, 0.258748, 0.200967], [0.102088, 0.081957, -0.14874, -0.10587, -0.138308, 0.324206, 0.132827, -0.022395, 0.056922, 0.020887, -0.249325, -0.034995, -0.069059, 0.030116, 0.007047, -0.091653, -0.013521, 0.000507, 0.051462, 0.016983, 0.195332, -0.040001, -0.034743, 0.119863, -0.100355, 0.05642, 0.048038, -0.031997, -0.19874, 0.138208, -0.045277, -0.115083, 0.002694, -0.172252, -0.146529, 0.035165, -0.135363, 0.017201, 0.000731, -0.089721, 0.006105, -0.211602, -0.104071, -0.007646, -0.294363, 0.017498, -0.016103, -0.217785, 0.016478, 0.029944, 0.048825, -0.130119, -0.335511, -0.114625, -0.150333, -0.224003, 0.032317, -0.163572, 0.096397, 0.007268, -0.117673, -0.262092, -0.176372, -0.052631], [0.156047, 0.026287, 0.046857, 0.123388, -0.421085, 0.100507, 0.221179, -0.760614, -0.222923, -0.023685, -0.052058, -0.068757, -0.251414, -0.05938, -0.607522, 0.128761, -0.021002, 0.129749, -0.406203, -0.1922, -0.05963, 0.258776, 0.242636, -0.258287, 0.096807, -0.147732, -0.238101, -0.226683, -0.0362, 0.136587, 0.019297, 0.135626, 0.231689, -0.475818, -0.146535, 0.013473, -0.153736, 0.04833, -0.074182, -0.268109, -0.315325, 0.100831, 0.140891, 0.069869, -0.217198, 0.201817, 0.038039, -0.201892, -0.092677, -0.091086, -0.042797, -0.221355, -0.308732, -0.212573, 0.085651, 0.002152, -0.164498, 0.021269, 0.14585, -0.373018, 0.061186, -0.153245, -0.158319, -0.04125], [0.060485, 0.017913, 0.105837, -0.036728, 0.311169, 0.00895, 0.250343, -0.334976, -0.24016, 0.006677, 0.028555, 0.032848, 0.014148, 0.066047, -0.144457, 0.202701, 0.414119, 0.192728, 0.008837, 0.425632, -0.243201, 0.056237, 0.123909, 0.101558, 0.155773, 0.132272, -0.136339, 0.02098, 0.05958, 0.289923, 0.131004, 0.070516, 0.018531, -0.261784, 0.156606, -0.081988, 0.094253, 0.124061, 0.035647, -0.051337, -0.260442, -0.004787, 0.538248, 0.063167, 0.143342, -0.121885, 0.066053, -0.022049, 0.018281, 0.239412, -0.113102, 0.327546, 0.037695, -0.062194, 0.040527, 0.376191, 0.123721, -0.063408, 0.196608, 0.038875, 0.218387, -0.020686, -0.003685, 0.160356], [-0.019373, 0.242606, -0.151582, -0.263898, -0.06172, -0.214665, -0.136264, -0.079585, -0.051019, 0.128122, -0.039526, -0.030789, 0.010383, -0.027106, 0.025576, -0.080748, -0.102456, -0.179532, -0.257175, -0.052105, -0.039906, -0.377381, 0.075264, 0.237341, -0.018976, 0.169076, 0.135213, 0.225255, -0.068203, -0.184494, 0.035854, -0.065781, -0.321942, 0.115719, 0.073891, -0.028815, 0.009764, 0.137358, -0.037528, -0.066516, -0.254912, -0.107001, -0.040971, 0.003197, -0.05783, -0.118837, 0.059961, -0.148422, 0.018458, -0.284038, 0.238923, -0.165023, -0.057813, 0.068777, -0.12312, -0.305688, 0.115178, -0.166241, 0.10055, 0.014187, -0.176898, -0.057972, -0.196603, 0.137488], [0.37979, 0.207218, 0.149104, -0.308248, -0.104467, -0.26022, 0.265691, -0.092794, -0.306771, 0.046078, 0.008282, 0.098968, 0.045128, 0.061191, -0.231422, 0.128839, 0.185409, 0.036108, 0.101276, 0.180482, -0.269055, -0.13413, 0.219613, 0.17809, 0.49674, 0.031772, -0.025302, -0.001615, -0.058122, 0.240198, 0.195909, -0.143496, 0.023116, -0.322085, -0.119174, -0.244531, 0.002663, 0.425607, 0.101899, 0.007244, -0.214383, -0.026033, 0.292523, 0.073656, 0.317607, -0.229396, 0.531444, -0.311104, -0.104539, 0.277845, -0.200866, 0.120814, -0.07263, 0.047695, 0.112982, -0.175181, -0.199521, -0.157377, 0.075575, -0.360794, 0.040845, -0.207978, 0.077074, -0.057823], [-0.112829, -0.135404, -0.129305, 0.010086, -0.315435, 0.164858, -0.00341, -0.208209, 0.159156, 0.273266, -0.325415, -0.033791, 0.013159, -0.101349, -0.029312, 0.003952, -0.254819, -0.170173, 0.067362, -0.018738, -0.118683, 0.169856, -0.244375, -0.001926, -0.182614, -0.153389, -0.174706, -0.019891, -0.108298, 0.13853, -0.087409, -0.09749, -0.10574, -0.035376, 0.295047, -0.515297, -0.243504, 0.029716, -0.012206, -0.065785, -0.101284, -0.097571, -0.367317, 0.235154, -0.187211, 0.184499, 0.15587, 0.027405, -0.098426, -0.134711, 0.07952, 0.194234, -0.000168, -0.141493, -0.052461, -0.262571, -0.029589, -0.222724, 0.06428, -0.134691, -0.043228, -0.257317, -0.278558, -0.211971], [0.078444, 0.199509, -0.274988, -0.157947, 0.060905, -0.075583, 0.176175, 0.028197, 0.022764, 0.201778, 0.138535, 0.202868, 0.098057, -0.185184, 0.002852, 0.061461, 0.056033, 0.008798, -0.139887, 0.150684, -0.097415, -0.196217, -0.337806, 0.110519, -0.132643, 0.056114, 0.122338, 0.117885, -0.157328, -0.240397, -0.146107, 0.003495, 0.068845, 0.026935, 0.102485, -0.270869, -0.082392, -0.027117, -0.019969, 0.0194, -0.095857, 0.104956, 0.252284, -0.063324, 0.188671, -0.252569, 0.057979, 0.007076, 0.030779, 0.114397, -0.14043, 0.216708, -0.158539, 0.127854, 0.030959, -0.173588, -0.02561, -0.012225, -0.127457, 0.000888, -0.197938, -0.230556, -0.107256, -0.033576], [-0.106944, 0.201152, -0.063464, -0.033031, -0.114287, 0.028123, 0.007813, 0.038494, -0.070497, 0.408353, -0.035211, 0.160172, 0.012792, -0.169446, 0.030455, -0.147424, 0.140587, 0.167197, 0.214819, 0.118798, -0.197972, -0.088202, 0.268002, 0.027449, 0.149913, 0.124457, -0.0587, 0.090073, -0.064514, -0.017319, -0.258627, 0.48674, 0.10016, 0.063077, -0.055622, -0.127015, -0.185392, 0.056153, 0.009701, -0.140942, 0.003908, -0.007493, 0.189516, -0.061446, -0.078516, -0.080221, 0.024225, 0.128644, 0.064019, -0.067898, -0.162279, -0.120678, 0.197284, -0.113768, -0.132084, 0.110553, -0.228014, -0.237576, -0.083121, -0.037862, 0.023298, 0.317221, -0.14527, -0.091096], [0.091676, -0.080186, -0.05731, 0.200008, -0.372784, 0.21804, 0.03358, -0.072314, 0.157773, -0.170592, -0.163557, -0.014944, 0.113125, 0.1058, 0.092857, -0.208374, 0.153966, -0.079422, 0.057864, 0.156103, 0.132109, 0.159831, -0.007465, 0.071816, 0.159079, 0.088963, 0.087141, -0.02693, -0.03331, -0.091038, 0.06993, 0.035706, -0.23316, 0.086769, 0.135243, 0.186388, 0.016294, -0.039499, -0.070623, -0.003221, -0.101156, 0.029669, -0.175793, -0.272019, 0.022761, 0.040385, 0.21446, -0.066964, -0.031636, -0.054845, 0.119229, 0.021388, -0.040692, -0.053519, -0.054134, -0.122563, -0.152477, 0.024735, -0.030526, -0.068377, 0.147117, 0.210499, -0.008551, 0.133595], [0.107738, 0.013687, -0.122666, 0.093127, -0.346786, 0.207435, 0.10286, -0.567588, 0.065422, 0.136137, -0.499851, 0.101193, -0.134328, 0.029113, 0.165003, -0.260383, -0.004696, -0.067281, 0.07323, 0.04148, -0.014209, 0.265597, 0.282523, -0.104595, -0.060025, -0.426086, -0.061422, -0.207923, -0.001183, 0.097004, 0.068479, 0.006687, -0.075402, -0.430264, 0.297621, -0.000862, -0.34969, -0.205081, 0.0463, -0.23804, 0.071703, -0.018467, -0.107372, -0.132699, -0.13143, -0.025165, -0.206752, -0.378892, -0.270046, -0.104669, -0.12117, -0.033734, -0.081763, -0.425631, -0.272693, -0.318951, -0.300322, 0.170281, 0.058156, -0.079668, -0.233234, -0.337319, -0.508863, -0.024227], [-0.179438, 0.034819, -0.098458, -0.280702, -0.012582, -0.314252, -0.057489, -0.146334, 0.070573, 0.18918, 0.022817, 0.047032, -0.029297, -0.220554, -0.333782, 0.119692, 0.028685, -0.173835, -0.208321, -0.145394, 0.023509, -0.183627, -0.177589, 0.007202, -0.086363, -0.119195, 0.020784, 0.127882, 0.038758, -0.042037, -0.108342, 0.026032, 0.110244, 0.05224, -0.093474, -0.367827, -0.03477, 0.15001, 0.051639, -0.14706, -0.080623, -0.009074, -0.145535, -0.023147, 0.052951, -0.054179, -0.042022, -0.210633, -0.123665, -0.044599, 0.285402, -0.013006, -0.18141, -0.134604, -0.067745, -0.238574, -0.04863, -0.239188, 0.152281, 0.120029, -0.152065, -0.135197, -0.174811, 0.150048], [0.052318, 0.231893, 0.071754, -0.295238, -0.097074, -0.035514, 0.034188, 0.085191, 0.125113, 0.132568, 0.055171, 0.05378, 0.1257, -0.057097, 0.234401, -0.012609, 0.707342, 0.467827, 0.162439, 0.703527, -0.284265, -0.155458, 0.110911, -0.214585, 0.239905, -0.074508, -0.043177, -0.081364, -0.162169, -0.218901, -0.028845, 0.190585, 0.191488, -0.010485, 0.202703, 0.107105, 0.148706, 0.521213, 0.435289, -0.063316, -0.172834, -0.293921, 0.453952, -0.163312, -0.234499, -0.178288, 0.460328, -0.144448, -0.069802, 0.297109, 0.240987, 0.083481, -0.156025, -0.006769, 0.198525, 0.034911, -0.006393, -0.070629, 0.072605, -0.117765, 0.068607, 0.021726, 0.108662, 0.291322]], [0.059501, 0.162065, -0.217903, -0.265314, -0.059559, 0.040524, -0.066455, -0.092186, 0.002312, 0.141471, -0.149649, 0.197853, -0.062617, -0.106013, 0.027507, -0.083688, -0.07258, -0.124069, 0.005839, 0.1495, -0.161829, -0.329996, -0.176383, 0.035324, 0.018965, -0.035682, 0.046599, 0.127299, -0.05883, 0.040454, -0.120056, 0.113844, 0.002626, 0.088162, -0.158612, -0.449697, -0.027917, 0.147976, -0.035785, 0.069019, -0.13411, 0.139332, 0.034425, -0.083251, -0.214614, 0.004571, -0.054767, -0.194196, -0.1848, -0.108998, 0.159631, 0.146481, -0.171697, -0.13608, -0.077104, -0.098744, -0.049919, -0.205808, 0.169372, 0.058152, -0.159707, -0.057885, -0.228811, 0.041816]),
    ([[-0.001084, -0.45995, 0.024862, 0.104423, 0.103231, 0.028106, -0.175461, 0.019505, -0.231942, 0.12188, -0.009054, -0.068805, 0.093619, -0.079834, -0.027197, 0.062905, 0.090789, -0.636864, -0.396924, -0.482087, -0.532154, -0.051621, 0.03064, 0.025759, 0.025714, -0.000459, 0.008505, 0.174577, 0.210466, -0.060506, 0.069221, 0.18501], [-0.01659, -0.068467, -0.126497, 0.052286, -0.157102, 0.087271, -0.227893, -0.060986, -0.116513, -0.126048, 0.141849, 0.003898, 0.044779, -0.179151, 0.031498, -0.14083, 0.08989, -0.02042, -0.199768, -0.028805, -0.015585, 0.043312, 0.011409, 0.026553, 0.091312, -0.061825, 0.054565, 0.131078, -0.024599, 0.109481, 0.153702, -0.052602], [0.113514, 0.059143, -0.16248, 0.151995, 0.412882, -0.050099, 0.125688, -0.091676, 0.082434, -0.090645, -0.082743, 0.107178, 0.009822, 0.239145, -0.028349, -0.05803, 0.036282, -0.274331, 0.088316, 0.142771, 0.206935, 0.031094, 0.045292, 0.05717, -0.20049, 0.030761, 0.080955, -0.030827, 0.087928, -0.03659, 0.10112, -0.300308], [-0.012734, -0.247235, 0.200557, -0.022271, 0.220489, 0.176135, -0.393454, -0.084467, -0.29876, -0.057305, 0.02253, 0.182393, 0.046784, -0.047865, 0.144148, 0.019833, 0.06159, -0.434152, -0.035265, -0.410883, -0.214685, -0.072414, -0.026791, 0.150763, 0.097734, 0.077538, 0.023403, -0.202949, 0.044372, -0.202765, 0.090936, 0.081598], [0.046052, 0.079442, 0.399586, 0.000476, 0.178095, 0.218011, 0.170886, 0.080068, 0.16638, 0.307635, 0.17456, 0.026265, 0.250544, 0.011435, 0.253546, 0.322954, 0.148525, 0.279399, 0.263296, 0.220777, 0.055317, -0.062578, 0.154176, 0.131038, 0.052906, 0.159159, 0.363657, 0.0041, -0.005676, 0.094221, 0.21105, 0.126668], [-0.110196, -0.096817, 0.097366, -0.096105, 0.13777, 0.029302, -0.161828, 0.030282, 0.005315, 0.205722, 0.022919, -0.002612, -0.116579, 0.060494, 0.072405, -0.076939, -0.042922, -0.239245, 0.034106, -0.022074, 0.029926, -0.175038, -0.085545, -0.171761, -0.131504, -0.094749, -0.095817, 0.151588, 0.039187, 0.069026, 0.012964, 0.102976], [-0.144454, 0.040467, 0.074428, -0.098142, 0.160859, 0.048834, 0.018048, 0.051756, -0.072091, 0.007428, 0.182661, 0.014469, -0.027656, -0.099393, -0.145556, 0.010646, -0.023542, -0.165019, -0.18298, -0.272269, -0.112206, -0.017964, -0.044685, 0.032079, -0.022262, -0.057808, 0.035379, 0.137167, 0.040617, 0.040793, 0.101172, 0.162791], [0.008214, 0.16629, 0.231317, 0.045816, 0.164479, 0.059493, 0.216291, -0.023756, 0.2361, 0.154199, -0.050765, 0.101303, 0.031289, 0.187992, 0.086692, 0.228047, 0.011917, 0.421839, 0.181811, 0.17913, 0.290812, 0.007815, 0.133424, 0.184499, 0.04767, 0.005215, 0.117391, -0.00161, -0.09075, 0.208414, 0.179535, 0.230112], [-0.232631, 0.146074, -0.090224, -0.363529, 0.172143, -0.153274, 0.059141, 0.091731, -0.022927, -0.160846, 0.040931, 0.007271, -0.053584, 0.173579, 0.021204, -0.231067, -0.017814, 0.052692, -0.126118, 0.091096, -0.093457, 0.110897, 0.088913, 0.055786, -0.029355, 0.095134, -0.059906, 0.109185, -0.228717, -0.033596, -0.082544, 0.0156], [-0.095152, -0.101058, -0.122147, -0.155215, -0.280052, -0.088315, -0.083005, 0.181259, -0.111113, -0.131662, 0.146225, -0.000483, 0.099578, -0.281944, -0.147605, 0.06771, 0.102078, -0.215087, -0.02427, 0.064191, -0.112353, 0.147153, 0.131618, 0.015705, 0.02242, 0.076387, -0.079109, -0.062725, 0.016379, 0.11073, 0.123833, 0.036342], [0.032594, -0.029452, -0.038695, -0.193221, -0.074429, 0.173903, 0.056686, -0.072965, 0.005747, 0.229205, 0.006449, -0.021858, 0.002939, -0.161246, -0.035852, 0.069985, -0.204018, 0.100498, -0.048781, 0.083799, 0.052197, -0.003884, 0.059891, 0.154914, 0.153192, 0.178715, 0.208044, -0.019067, -0.139646, 0.131605, 0.081472, -0.039766], [0.166302, 0.034697, 0.077308, 0.183389, -0.257553, 0.138327, 0.132359, 0.139279, -0.212014, 0.056487, 0.147742, 0.117119, 0.074176, 0.106051, -0.09549, 0.001988, 0.033205, -0.230222, 0.144101, -0.110584, -0.131327, 0.009082, -0.088828, 0.098115, 0.11411, 0.099932, 0.044889, 0.094984, 0.132626, -0.014374, 0.185853, 0.008416], [-0.165368, -0.039262, -0.071744, -0.051194, 0.210953, 0.018859, 0.010784, -0.164174, 0.142388, 0.037771, 0.105626, 0.027525, -0.015407, 0.026865, 0.077183, 0.029923, 0.095745, -0.077668, 0.041951, 0.116825, 0.10626, 0.000987, 0.034055, 0.055455, -0.008317, 0.004784, -0.163709, 0.160522, 0.206453, -0.16162, -0.348618, 0.154304], [0.148573, 0.00415, -0.082549, 0.157251, -0.451027, 0.019277, 0.242691, 0.143707, 0.139563, 0.20403, -0.033011, -0.102963, 0.090347, -0.055908, -0.066694, 0.121998, 0.058027, 0.218377, 0.194871, -0.135455, -0.079948, -0.036561, 0.016793, -0.00298, 0.033604, 0.019706, -0.061202, 0.090018, 0.014004, 0.020384, 0.073632, 0.130727], [0.15416, 0.009331, 0.068375, -0.050749, -0.000266, -0.223754, -0.064587, 0.077297, 0.104337, -0.136919, 0.186997, 0.009846, -0.021721, -0.09515, 0.032345, -0.344728, 0.05733, 0.135051, 0.17445, 0.071766, 0.131553, 0.057263, 0.009092, 0.102312, -0.007959, -0.180871, -0.281649, 0.081947, -0.305799, 0.05311, -0.179604, 0.206999], [-0.134292, -0.077659, -0.091109, 0.001617, -0.004175, -0.002939, 0.132246, -0.043669, 0.014441, -0.226131, 0.122653, -0.041062, 0.029211, -0.008856, 0.120306, -0.248059, -0.038993, -0.018772, 0.081956, -0.079282, -0.098135, 0.095578, 0.019926, -0.075957, 0.140487, 0.010212, -0.035428, -0.211424, -0.417176, 0.096817, -0.081909, -0.071294], [0.249143, 0.046585, 0.257783, 0.11149, -0.30061, 0.155501, 0.260533, 0.045628, 0.034526, 0.189268, 0.107412, 0.043366, -0.104221, 0.100544, 0.159912, 0.243131, 0.007058, 0.316557, -0.049889, -0.032367, -0.072674, 0.227692, -0.151521, 0.191513, -0.103032, 0.198359, 0.077211, 0.117903, -0.024699, -0.060058, -0.010498, 0.063258], [0.140307, 0.057457, 0.130192, 0.124241, 0.103785, -0.039977, 0.214467, 0.112249, 0.161977, 0.075029, 0.176186, -0.04389, -0.102249, 0.184257, -0.080684, 0.158724, -0.026764, 0.141481, 0.109484, 0.018293, 0.136233, -0.063921, -0.011922, 0.203886, -0.256885, 0.092122, 0.075374, -0.080514, -0.129361, -0.134326, 0.12401, -0.064142], [0.021896, 0.125977, 0.314958, 0.209425, 0.058415, 0.026796, -0.140206, 0.152572, -0.053803, -0.002171, 0.121375, 0.118229, -0.167884, 0.101341, 0.230973, 0.16915, 0.244464, -0.342407, 0.095554, -0.061249, 0.038448, 0.037297, -0.024376, 0.170074, -0.210972, -0.043162, -0.003617, 0.169126, -0.032642, -0.044576, -0.042892, -0.463823], [0.191471, -0.06694, 0.167542, 0.123761, 0.029226, 0.069506, 0.020144, 0.055798, 0.08253, 0.007021, 0.201332, 0.15727, 0.01798, 0.002844, -0.076854, -0.003263, 0.160159, -0.107118, 0.110455, 0.078587, 0.025493, 0.180331, -0.149747, 0.053111, -0.005453, 0.130524, 0.096516, -0.103253, -0.166232, -0.09576, 0.122003, 0.143891], [-0.362587, -0.209654, -0.359857, -0.077604, -0.1061, -0.169185, -0.005523, 0.04469, -0.077256, -0.372141, 0.021452, -0.239145, 0.144107, -0.002196, -0.006032, -0.342985, -0.036605, 0.060618, -0.109797, 0.021236, -0.107695, -0.143264, 0.074882, -0.448244, 0.00466, -0.060629, -0.105157, 0.029203, 0.170695, -0.043096, -0.072938, -0.177101], [-0.100128, -0.011457, 0.147406, -0.080986, -0.16613, 0.020963, -0.070596, -0.004414, 0.026563, 0.05971, -0.150458, 0.07871, -0.064898, -0.055658, 0.025494, 0.128627, -0.014559, -0.273355, -0.18598, -0.27612, -0.210035, -0.182671, 0.077203, 0.111861, -0.262377, -0.087464, 0.111561, -0.015627, 0.111455, -0.133229, 0.003951, -0.154931], [0.221081, 0.084371, 0.107195, 0.154243, -0.287069, 0.116662, 0.015769, 0.128134, -0.015624, 0.162864, -0.066053, 0.04256, 0.039168, 0.000746, -0.083129, 0.081065, 0.079184, 0.085216, -0.016131, -0.281557, -0.234499, -0.081916, 0.092995, 0.065511, 0.000316, 0.091535, 0.135587, -0.158375, 0.064745, 0.070855, -0.010009, 0.25785], [-0.162629, -0.095539, 0.005078, 0.071447, -0.12621, -0.052938, 0.040163, -0.093008, -0.329183, -0.014699, 0.117275, 0.056918, -0.530562, 0.345149, -0.231054, -0.048711, 0.117316, -0.147879, -0.329486, -0.024898, -0.122531, -0.074959, -0.10449, -0.183483, -0.020815, 0.185407, 0.216722, 0.089812, 0.214366, -0.141987, -0.032771, -0.061318], [0.074343, -0.097136, 0.183023, 0.136743, -0.115966, 0.053323, -0.192211, -0.064732, 0.006413, 0.10465, 0.025831, 0.012582, -0.412041, 0.005777, -0.023873, 0.201451, 0.074904, -0.530696, -0.344977, -0.047061, -0.158808, -0.025005, -0.080519, -0.074968, -0.05205, 0.093128, 0.091764, 0.11195, -0.141196, 0.111207, 0.023606, -0.231385], [0.058515, -0.020165, -0.101074, 0.206068, -0.044851, -0.032196, 0.056651, -0.096523, 0.115169, -0.193722, -0.077995, -0.070661, 0.070819, 0.051866, 0.10696, 0.087941, -0.148007, -0.029871, 0.050578, 0.175407, 0.160626, -0.122049, -0.077787, -0.187486, 0.022472, 0.030231, 0.015522, -0.050677, -0.172411, 0.144076, 0.13444, 0.047014], [-0.237874, 0.036444, -0.017315, -0.134105, 0.042342, 0.139649, -0.049104, -0.077981, -0.031068, 0.052259, 0.133142, -0.146001, 0.030401, 0.096804, 0.022231, -0.087448, -0.080898, 0.0724, -0.034448, 0.257669, 0.113166, -0.025179, 0.11205, -0.297376, 0.111879, -0.072032, -0.058804, -0.063038, -0.177659, -0.039661, -0.059114, 0.056449], [-0.065252, 0.096318, -0.113211, -0.030256, 0.049203, -0.208605, -0.083481, 0.069728, 0.096998, -0.240275, -0.322039, 0.10961, 0.076241, -0.068114, 0.021275, -0.085666, -0.09813, -0.105965, 0.109584, 0.154253, 0.035845, -0.057263, 0.088906, -0.050239, 0.11621, -0.159492, -0.185102, -0.105251, -0.159833, 0.027677, 0.089396, -0.062724], [0.082278, -0.31333, -0.069945, 0.008738, 0.055205, -0.103187, -0.253158, 0.015345, -0.005537, -0.15745, 0.065662, -0.008558, 0.077866, 0.051183, -0.005807, 0.047807, 0.08388, -0.194267, -0.264775, -0.18761, -0.258431, 0.098379, 0.100966, -0.066498, 0.033155, -0.043571, -0.119499, -0.067234, 0.046769, -0.092743, -0.107788, -0.001421], [-0.108449, 0.059616, -0.161754, -0.188997, -0.092067, -0.132644, -0.070312, -0.123961, 0.101919, -0.166261, -0.05141, -0.09205, -0.028791, -0.161398, -0.119665, -0.01903, 0.160619, 0.140635, 0.087985, -0.001518, -0.052668, 0.025195, 0.103111, -0.176827, 0.030224, -0.282998, -0.300424, 0.051168, 0.038974, 0.134868, -0.160287, -0.161778], [0.030116, 0.072252, -0.08384, 0.045181, -0.480289, 0.150782, 0.122952, 0.162969, 0.180618, -0.09862, 0.023542, -0.013368, -0.10765, 0.014319, -0.083966, 0.156897, 0.144053, 0.099322, 0.102548, -0.022862, -0.00357, -0.042755, -0.129607, 0.110762, 0.0306, 0.130256, 0.149435, 0.135683, -0.02055, -0.25434, 0.104276, 0.004728], [-0.04756, -0.083605, 0.152269, -0.080089, -0.080591, 0.020975, -0.142528, 0.132818, -0.203304, 0.08871, 0.074379, 0.12386, -0.421098, -0.041052, -0.102815, -0.068614, -0.066111, 0.001827, -0.149535, 0.099469, 0.074097, 0.08902, -0.203813, 0.014163, 0.098347, 0.069599, -0.012269, -0.015485, -0.030923, -0.109597, 0.004047, -0.354784], [0.052471, 0.137121, 0.1926, 0.013138, -0.362542, 0.079723, -0.107588, 0.125295, -0.030135, 0.261846, 0.070303, 0.022335, 0.042627, -0.379415, 0.011166, 0.068659, 0.149557, -0.369259, 0.045735, -0.207935, -0.133997, -0.014467, -0.092504, 0.065478, -0.083945, -0.070351, 0.100905, 0.067791, 0.029113, 0.168564, 0.106046, -0.096598], [0.101778, 0.147169, -0.350995, 0.097363, 0.114037, 0.017738, 0.225367, 0.067054, 0.127376, -0.366647, 0.002842, -0.119567, -0.034558, 0.083427, 0.123081, -0.171099, -0.174156, 0.261321, 0.079266, 0.175184, 0.092956, -0.037332, 0.014463, -0.148916, 0.005589, -0.154203, 0.082008, -0.203618, 0.058849, -0.192443, 0.143634, 0.021987], [-0.229411, 0.051694, -0.019041, -0.053903, 0.137207, -0.17004, -0.171274, -0.049021, -0.081668, -0.166353, 0.071496, -0.001296, -0.216133, -0.035305, 0.135364, -0.019606, 0.014216, 0.012416, -0.138919, -0.128162, 0.125536, -0.070945, 0.03158, -0.089411, -0.004194, -0.083725, -0.093827, -0.062456, -0.034989, -0.017748, -0.037237, 0.030098], [-0.440543, -0.048352, -0.206213, -0.448873, 0.162581, -0.125863, -0.031322, -0.119839, -0.188607, -0.473521, 0.031172, -0.213002, 0.250761, 0.051701, 0.026992, -0.2154, 0.128278, -0.108103, -0.091041, -0.128868, -0.201345, -0.252654, -0.042712, -0.37261, 0.01625, -0.123374, -0.042349, -0.034269, 0.393931, -0.127883, -0.060901, 0.177474], [-0.133803, 0.102064, -0.17258, -0.029827, -0.138395, 0.059198, 0.099559, 0.00792, 0.064428, -0.404117, -0.289708, -0.106827, 0.135795, -0.075577, -0.012208, -0.032883, 0.037174, 0.172616, 0.002261, -0.017405, 0.062875, 0.158136, 0.015705, -0.017437, 0.078916, 0.017681, -0.027789, -0.064112, -0.147603, -0.05381, -0.01375, 0.179708], [-0.110232, 0.101254, 0.017231, 0.17936, -0.046311, 0.068935, 0.127224, 0.076878, 0.128105, -0.038703, 0.084167, 0.082608, -0.049491, -0.170032, 0.013308, -0.000795, -0.080841, 0.129298, 0.125959, 0.00742, 0.054283, 0.075001, 0.006509, -0.08766, -0.168323, 0.0914, 0.110226, -0.048625, -0.229639, 0.052524, 0.200756, 0.07441], [0.125533, -0.111751, -0.078984, 0.021251, 0.017527, 0.099266, -0.205963, 0.101752, -0.108803, 0.054117, -0.024931, 0.056351, -0.047897, -0.186026, 0.100501, 0.031719, 0.103607, -0.088895, -0.148448, -0.177021, 0.231926, 0.073277, 0.043936, -0.009159, 0.011293, 0.008668, 0.182881, 0.043343, -0.039945, -0.024433, -0.058178, -0.098913], [0.040576, 0.108895, -0.178255, 0.092831, -0.141007, 0.065345, 0.060753, 0.029702, 0.060706, -0.03332, 0.086816, -0.080604, 0.010599, 0.067525, 0.038434, 0.181683, -0.117489, 0.088172, 0.013393, 0.075534, 0.075021, 0.026597, -0.035189, -0.086371, 0.140682, 0.036344, -0.075797, 0.067636, -0.214192, 0.157547, -0.075778, -0.157521], [-0.031363, 0.105515, -0.399765, -0.048249, 0.173222, 0.106367, -0.006615, -0.166381, 0.107786, 0.170086, -0.182489, -0.133664, 0.053943, -0.037395, 0.222556, -0.004058, -0.164373, -0.004463, 0.22035, 0.148395, 0.24289, -0.022635, 0.099433, -0.103979, 0.006545, -0.146652, 0.039053, -0.156693, 0.075382, -0.007168, -0.128745, -0.176339], [-0.049313, -0.0089, -0.081704, 0.030328, -0.151578, -0.036658, 0.103728, 0.09095, -0.358701, 0.039305, -0.023037, 0.077001, -0.12024, -0.005567, -0.108833, -0.216999, -0.115001, -0.130548, -0.255189, -0.232265, -0.236772, -0.12517, -0.08631, 0.009652, -0.122972, 0.11521, -0.099614, -0.065747, 0.134, -0.081585, -0.108335, -0.032089], [0.353885, 0.169934, 0.229456, 0.132792, -0.043815, 0.031873, 0.059013, 0.067669, 0.177561, 0.098618, 0.153744, 0.123587, 0.048411, -0.191564, 0.066962, 0.194562, 0.186334, 0.140552, 0.189814, 0.174638, 0.12495, 0.191917, -0.025861, 0.29799, 0.006631, 0.213743, 0.154052, 0.094319, 0.042277, 0.066691, 0.117124, 0.050336], [-0.159404, -0.178926, -0.155429, -0.324186, -0.163016, -0.02238, -0.05229, 0.029176, 0.041408, -0.322411, -0.193558, -0.12449, -0.048527, -0.013455, -0.10525, 0.058413, 0.123732, -0.154034, 0.070962, -0.033049, 0.058151, -0.083013, 0.083422, 0.080586, 0.027317, -0.107929, -0.262239, -0.024662, 0.028867, 0.016909, -0.00759, -0.147933], [-0.012176, -0.039391, 0.269622, -0.011802, -0.198101, -0.010665, -0.076838, -0.078513, -0.014699, 0.219964, 0.053616, 0.036629, -0.038182, -0.025682, -0.032763, 0.144197, 0.101233, 0.120041, 0.024368, 0.158987, 0.139903, -0.156532, -0.092039, 0.193901, 0.15849, -0.012232, 0.338618, 0.19059, -0.454065, 0.10254, 0.038557, -0.117942], [0.004669, 0.034726, -0.155557, -0.008634, 0.097552, -0.062819, -0.305035, 0.163515, -0.242724, 0.06652, -0.097813, -0.010355, -0.123905, 0.128834, -0.085856, -0.194603, 0.123722, -0.065184, -0.363716, -0.266935, 0.040685, -0.100821, 0.017689, 0.085429, 0.059123, 0.054581, -0.058823, 0.048858, 0.094007, -0.113619, -0.078303, -0.356687], [0.328732, 0.050415, 0.230266, 0.247917, 0.211067, 0.012787, 0.118242, 0.127237, -0.013587, 0.246815, 0.175106, -0.003429, 0.118872, 0.308818, -0.040057, 0.178299, -0.11526, 0.262593, 0.083148, -0.037405, 0.051955, 0.081083, 0.021095, 0.241923, 0.159512, 0.07729, 0.162193, -0.13022, -0.316765, 0.008486, 0.123329, -0.12862], [-0.168575, 0.041942, -0.13172, 0.10517, 0.236598, 0.123084, -0.024727, -0.145043, -0.006472, -0.222557, -0.173912, 0.047199, 0.073903, 0.110314, 0.106818, 0.160271, 0.072248, -0.088576, -0.02107, 0.046405, 0.179573, -0.101927, 0.004644, -0.067031, 0.054416, -0.029386, -0.093536, 0.041056, 0.135396, 0.030616, -0.077237, -0.07971], [0.020766, -0.069527, 0.231324, -0.48465, -0.039186, 0.093439, 0.014274, 0.075216, 0.123095, 0.129853, -0.183778, 0.123951, -0.035676, 0.063431, -0.045518, 0.0161, 0.004616, 0.052615, 0.04417, 0.039309, 0.052621, -0.206584, 0.104888, 0.082103, 0.036025, 0.083184, -0.077179, 0.176966, -0.075607, -0.124363, -0.053866, 0.087962], [0.121399, 0.009644, 0.135005, 0.043077, 0.011738, 0.066946, 0.071239, 0.124749, 0.125463, 0.198986, -0.054159, 0.143701, -0.04926, -0.143786, 0.084474, 0.057923, 0.092561, 0.046632, -0.065563, -0.189663, -0.032062, -0.037452, -0.141496, 0.102891, 0.125888, -0.079922, 0.170538, -0.072499, 0.126692, 0.05757, 0.085373, -0.077049], [0.115278, 0.143254, -0.273856, 0.084544, 0.113081, -0.082079, 0.076085, 0.007603, -0.156198, -0.064875, 0.062259, -0.233956, -0.138494, 0.138441, 0.076221, -0.138914, -0.100444, -0.191816, -0.144123, 0.005725, 9e-06, 0.093901, 0.054209, -0.279816, 0.016955, 0.02787, -0.11643, -0.219548, -0.193795, 0.125064, -0.035759, -0.091123], [-0.066324, 0.083049, -0.098487, 0.032478, -0.006509, 0.014696, 0.144261, 0.055103, -0.003345, 0.118799, 0.112174, 0.038194, 0.136879, -0.171562, 0.04406, -0.021752, 0.032298, -0.004396, -0.050099, -0.049167, 0.022877, -0.068547, 0.121692, 0.047829, -0.048037, 0.153435, 0.134212, 0.106678, -0.060366, 0.14658, 0.161647, -0.079864], [-0.120676, -0.078336, -0.068168, -0.231515, 0.141633, -0.191637, -0.117984, -0.008228, 0.021583, -0.15593, -0.368836, -0.063817, 0.049068, -0.009458, -0.035954, -0.176296, 0.031498, 0.068913, 0.021405, -0.011305, -0.042605, -0.031056, 0.04711, -0.129087, -0.057045, -0.089961, -0.314066, 0.04856, 0.033308, -0.152862, -0.176171, 0.023245], [-0.200965, -0.02186, -0.068327, 0.18863, 0.105259, -0.061727, 0.145185, 0.061229, -0.026656, 0.260023, -0.006174, -0.11804, 0.160803, 0.118204, -0.073112, 0.0578, -0.002298, 0.104237, -0.042428, 0.139603, 0.14018, -0.034487, 0.04686, -0.086316, 0.10023, 0.035781, -0.057218, 0.162139, 0.132486, 0.114259, -0.062536, 0.058228], [0.129223, 0.127379, -0.153806, 0.166632, 0.100119, 0.126847, 0.059769, 0.037227, 0.098828, -0.120998, 0.074265, -0.187425, -0.001289, 0.112163, 0.17723, 0.203679, -0.286913, 0.153059, 0.18054, 0.091671, 0.078565, 0.14877, 0.083937, -0.062589, -0.08389, 0.250827, -0.015323, -0.250357, 0.07118, -0.032921, 0.118682, -0.134012], [0.10321, 0.156154, 0.041487, 0.005013, -0.015061, -0.186505, 0.172885, -0.080797, -0.060578, 0.25431, 0.128754, -0.036169, 0.001213, -0.033374, 0.118834, -0.047689, 0.001418, 0.200603, 0.003391, 0.229232, 0.161915, -0.202411, -0.018371, -0.162968, 0.057231, -0.243223, -0.093452, 0.202285, -0.052942, 0.041171, -0.101725, -0.008983], [-0.104971, -0.019169, 0.172078, -0.194484, -0.14299, -0.134473, 0.158413, -0.112981, -0.140179, -0.164877, -0.272695, 0.120686, -0.047053, -0.005414, 0.089204, -0.122254, -0.013199, -0.14964, 0.01375, 0.0466, -0.128487, -0.150377, 0.119243, 0.056976, 0.12823, -0.352667, -0.155702, 0.07722, 0.047617, -0.033602, -0.095419, 0.056924], [-0.304599, -0.114089, 0.14229, -0.45893, -0.078344, 0.056977, 0.001271, 0.091306, -0.02009, -0.000785, -0.000915, 0.066068, 0.154591, -0.252961, 0.040803, -0.155297, -0.057867, 0.017007, 0.07785, -0.111345, 0.007232, -0.151494, 0.062265, -0.039005, 0.070627, -0.127724, -0.008849, -0.026395, 0.127323, 0.023695, -0.134307, -0.023534], [-0.100379, 0.039139, -0.098078, -0.26898, -0.09983, -0.046534, -0.079811, 0.01541, -0.087843, -0.288524, -0.302852, -0.110928, -0.119301, 0.071475, 0.02214, -0.188412, 0.078603, -0.09523, -0.008929, -0.151503, 0.057416, 0.000296, -0.021901, -0.290419, -0.026184, 0.019998, -0.111196, -0.170898, -0.031732, 0.136827, -0.09254, -0.081978], [-0.055478, 0.126629, -0.090442, -0.080281, 0.10727, -0.16133, 0.06475, 0.024263, 0.111228, 0.127233, 0.01631, -0.004805, 0.058223, 0.214867, 0.073918, 0.079529, 0.023922, 0.108163, -0.065485, 0.039531, 0.078281, 0.093155, 0.14899, -0.030761, 0.014179, -0.112476, -0.089424, 0.212154, 0.02109, -0.161331, -0.141171, -0.150906], [0.025183, -0.007581, -0.283088, -0.00574, -0.058856, 0.039337, -0.306283, -0.018678, -0.09565, -0.241212, 0.04548, 0.119477, 0.008856, 0.027558, 0.008243, 0.004999, 0.12894, -0.042503, -0.01142, -0.020632, -0.142374, 0.011064, -0.080259, -0.063322, -0.058087, 0.065444, 0.119435, -0.168841, 0.131754, 0.040652, 0.060398, -0.07297], [0.105741, -0.102097, 0.0836, 0.015887, 0.272277, -0.092527, -0.089715, -0.021793, -0.107726, 0.131375, -0.009692, -0.023398, -0.025648, 0.190593, 0.057886, 0.063123, 0.054706, -0.201557, -0.046101, 0.03375, -0.104367, -0.035933, 0.160974, -0.025312, -0.119411, 0.021694, -0.009487, -0.0332, 0.359126, -0.184428, -0.115871, 0.129595], [0.114304, 0.131556, 0.110763, 0.128927, -0.074622, -0.048485, 0.064987, -0.07927, -0.051926, -0.393984, -0.028947, -0.124668, 0.169039, -0.021214, 0.180422, -0.158954, 0.058638, 0.053678, 0.181216, 0.176374, 0.149351, -0.004152, 0.050226, -0.106525, 0.045346, -0.147357, -0.008348, -0.050038, 0.158763, -0.019891, -0.109774, -0.009924], [0.03923, 0.056119, -0.103634, 0.17122, -0.110698, -0.068027, 0.086525, 0.013524, -0.043284, -0.268837, -0.131197, 0.136735, 0.010932, -0.021478, -0.009854, -0.17412, 0.034129, -0.111203, 0.069395, -0.03297, -0.064266, 0.131751, 0.112287, 0.035338, 0.134153, 0.065525, -0.025103, -0.186269, -0.270702, 0.088341, -0.011844, 0.107253]], [-0.186197, 0.089871, -0.414044, -0.070159, -0.063157, 0.092098, -0.107877, 0.062836, 0.066668, -0.173573, 0.075917, -0.177695, -0.014945, -0.108232, 0.073465, -0.020139, 0.029773, -0.134733, 0.016341, 0.094735, 0.008599, 0.128075, 0.082347, -0.208795, 0.078361, 0.107399, 0.130666, -0.00298, -0.024485, 0.11146, 0.166979, -0.104297]),
    ([[0.284949], [-0.093728], [0.391558], [0.296616], [-0.206581], [0.114179], [-0.220453], [0.096073], [-0.171732], [0.400495], [0.158125], [0.076278], [-0.115765], [-0.150722], [-0.091667], [0.336457], [0.149577], [-0.451992], [-0.103802], [-0.189015], [-0.190218], [0.084165], [-0.079835], [0.149669], [-0.112763], [0.116559], [0.166986], [0.187802], [-0.20473], [-0.151996], [0.164766], [-0.064518]], [0.027187]),
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
