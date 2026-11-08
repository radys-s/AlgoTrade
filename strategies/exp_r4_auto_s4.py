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
    ([[0.092816, -0.165425, 0.027805, 0.158767, -0.435251, 0.533125, -0.085711, -0.024919, 0.082759, 0.015153, 0.109145, -0.037518, 0.352506, -0.291004, 0.094419, -0.249011, -0.005812, -0.140965, -0.245148, -0.028046, -0.193748, -0.009864, 0.155613, -0.135419, -0.066935, -0.066094, 0.115048, 0.07137, -0.004073, -0.111684, -0.013872, 0.365325], [0.088515, -0.044303, 0.227851, -0.180314, -0.29163, 0.283278, -0.046535, -0.18077, 0.000778, -0.152871, -0.32472, 0.143637, 0.156609, -0.153241, -0.06282, 0.099302, -0.029074, -0.027852, 0.067562, -0.062263, 0.062422, 0.018116, -0.041374, 0.135473, 0.092515, 0.044814, 0.026798, -0.173229, -0.282827, 0.10199, -0.081473, 0.332459], [-0.138147, 0.148083, 0.167216, -0.098832, 0.080904, -0.020265, 0.142939, -0.289154, -0.090054, 0.179266, -0.293996, 0.129574, 0.165758, -0.031579, 0.045461, 0.294727, -0.175745, 0.32699, -0.189839, 0.179632, -0.031056, 0.071095, -0.19341, 0.225045, -0.140104, 0.078366, 0.021878, 0.00905, 0.073832, 0.080625, -0.213406, -0.11834], [-0.046635, -0.139098, -0.104014, -0.149631, 0.27884, -0.171462, 0.058291, 0.290392, 0.219745, -0.148275, -0.13739, -0.081608, -0.309667, 0.321736, 0.034085, 0.338785, -0.026566, 0.093637, 0.068213, -0.151117, -0.004762, 0.323969, -0.068248, 0.171736, -0.113333, -0.330947, -0.212075, -0.016353, 0.129034, 0.062679, 0.263082, 0.040849], [-0.362651, 0.316728, 0.137564, -0.018976, 0.078282, -0.385131, 0.30768, 0.197707, 0.460901, -0.246078, 0.084967, 0.240956, -0.530104, 0.162391, -0.049421, 0.478407, 0.204324, -0.307887, -0.349231, 0.117654, 0.266779, 0.138989, -0.337474, -0.291727, 0.175287, 0.060186, -0.162601, -0.022314, 0.432376, -0.353031, 0.377607, 0.037944], [-0.384944, 0.162192, 0.166948, -0.283409, 0.210688, -0.048032, 0.10221, -0.193234, 0.055327, 0.294087, -0.134997, 0.022175, 0.087964, 0.168296, 0.108189, 0.189268, -0.180187, 0.324876, -0.195569, 0.089936, 0.245702, -0.000483, 0.127891, -0.120603, 0.12467, 0.172713, -0.284483, 0.169221, 0.081157, -0.25918, 0.155217, 0.10055], [-0.026864, 0.420231, -0.338901, -0.318663, 0.324925, 0.311891, 0.635924, -0.290246, -0.236687, -0.514903, 0.13026, -0.221768, 0.332067, -0.550751, -0.048323, -0.104132, 0.147407, 0.075632, -0.115666, 0.488217, 0.072212, -0.445193, -0.230966, -0.327764, 0.209881, -0.239805, -0.23952, -0.38662, 0.326075, 0.049255, 0.07125, -0.162488], [0.278951, -0.114002, -0.109336, -0.112557, -0.061194, -0.131321, -0.037766, -0.070834, -0.213672, 0.100649, -0.333661, 0.290541, 0.012073, 0.003909, -0.367626, -0.089594, 0.011785, 0.074455, -0.160398, 0.178605, -0.088945, -0.04866, 0.266976, 0.007452, 0.002416, 0.047874, -0.198954, 0.076216, 0.04951, 0.18925, -0.061989, 0.030259], [0.080912, 0.252119, 0.46458, -0.101146, -0.290144, 0.22253, -0.311842, -0.286422, -0.25744, -0.463988, -0.389109, -0.023929, -0.128716, 0.557217, -0.351936, 0.218059, -0.239327, 0.360392, -0.102463, -0.120776, -0.348969, 0.194728, -0.364527, -0.302432, 0.093638, 0.072401, 0.203038, 0.069035, -0.540931, 0.01495, 0.112158, 0.096954], [0.236894, 0.269522, -0.73727, 0.142653, 0.275706, 0.174471, 0.056094, 0.011813, 0.106064, -0.098989, 0.0044, 0.094861, -0.563326, -0.223972, 0.083197, -0.306533, -0.581448, -0.345849, -0.455449, -0.267851, -0.024925, -0.141847, 0.160844, 0.172649, -0.275247, 0.043682, -0.128636, -0.034832, -0.000554, 0.043606, -0.250789, 0.302198], [-0.059647, -0.164232, -0.013543, 0.108108, 0.348027, -0.128368, -0.25326, -0.31839, -0.108407, -0.009524, -0.365668, -0.062003, 0.096012, -0.18409, 0.088542, -0.005037, -0.032713, -0.065483, 0.061867, -0.114947, 0.032356, 0.038933, 0.156568, 0.139075, 0.137732, -0.071121, -0.005176, -0.187382, 0.054664, -0.043694, 0.080954, 0.018801], [-0.24141, 0.440199, 0.274908, 0.359597, -0.104567, 0.379917, -0.366447, -0.200383, 0.164584, -0.099039, 0.303077, -0.062317, 0.206174, 0.126618, -0.115475, 0.219613, -0.496093, -0.140374, -0.070472, 0.195745, -0.01593, 0.023725, -0.304856, -0.434615, 0.158182, -0.349223, 0.32594, 0.203895, 0.071924, 0.206352, 0.164246, -0.197979], [0.098875, 0.226136, 0.028761, -0.249085, -0.044411, -0.028498, 0.059652, 0.083057, -0.089561, -0.141611, -0.167153, -0.236455, -0.013037, -0.054615, -0.002776, -0.143119, 0.113464, 0.092189, 0.107327, 0.213934, -0.103595, 0.103409, -0.175414, 0.007097, 0.019963, 0.004458, 0.403923, 0.188734, -0.112384, -0.000157, -0.031044, -0.102269], [-0.040632, -0.107545, -0.06806, -0.299302, -0.032621, 0.236697, 0.034856, -0.055665, -0.068419, -0.100773, -0.050671, -0.399156, 0.112624, -0.012498, 0.001271, 0.03101, 0.241523, 0.129215, -0.046636, -0.118123, 0.24914, 0.341932, -0.239313, 0.196812, 0.341074, -0.269327, 0.179338, 0.307907, 0.075221, -0.31359, 0.332315, -0.066616], [0.106811, -0.103065, 0.136615, 0.033983, 0.249306, 0.168312, -0.174362, -0.234315, -0.367604, -0.277456, 0.037359, 0.100863, 0.05831, -0.073819, 0.270636, -0.099614, -0.031732, -0.236513, -0.039133, -0.077946, -0.012604, 0.54127, -0.185243, 0.422134, 0.30739, -0.049433, 0.178498, -0.569277, -0.50552, -0.262711, 0.003434, -0.046803], [0.050585, 0.270927, 0.032069, 0.012255, 0.074938, -0.024004, -0.138432, -0.012621, -0.004536, 0.161772, 0.13957, 0.07005, 0.032962, 0.076147, -0.057732, 0.059144, 0.076076, 0.212128, 0.011379, -0.191605, 0.169099, 0.178367, -0.115425, -0.202383, 0.245076, 0.245211, -0.046395, 0.046961, -0.245749, -0.478138, 0.107265, -0.072833], [0.058092, 0.035927, -0.515718, 0.53431, -0.359784, 0.065591, 0.012012, 0.040782, 0.381858, 0.37464, -0.11235, -0.249406, -0.028488, 0.338904, -0.117835, -0.053789, -0.004496, -0.042629, 0.014641, 0.159055, -0.997568, 0.557777, 0.371729, -0.383703, 0.054957, 0.062244, -0.290649, 0.557685, -0.080501, 0.042859, -0.024954, 0.135174], [-0.016787, -0.063789, 0.236588, -0.069785, 0.067342, 0.138437, 0.05908, 0.089488, -0.197646, 0.04892, -0.493719, -0.021931, -0.045571, 0.402867, 0.097961, 0.211066, -0.567934, 0.380852, -0.153881, -0.149748, 0.164678, 0.026079, 0.46862, 0.087867, 0.217219, -0.118144, -0.07776, 0.133936, 0.122919, 0.118915, 0.203364, -0.254784], [-0.111037, -0.004675, -0.191649, 0.049328, -0.078033, -0.001046, -0.061631, -0.025161, 0.054195, 0.056953, 0.001437, 0.222165, 0.086619, -0.01308, 0.093405, 0.051685, 0.324812, 0.081547, 0.124089, -0.090428, 0.105444, -0.016851, 0.104305, -0.068156, 0.261087, 0.155013, 0.083, -0.081572, 0.137198, -0.332348, 0.207277, 0.04368], [-0.42557, 0.651851, 0.16025, -0.201125, -0.152152, -0.012868, -0.049275, -0.067876, 0.11236, -0.042496, 0.005502, 0.273825, -0.008836, 0.008249, -0.771542, -0.096204, -0.196644, -0.144889, -0.269245, -0.243959, 0.172054, -0.188088, 0.062756, 0.130316, 0.351816, 0.502636, 0.098054, 0.130001, 0.128068, -0.0084, 0.055485, -0.017803], [0.140616, 0.044999, -0.08015, 0.063972, 0.344986, -0.290553, -0.090226, 0.173342, -0.087783, 0.104867, -0.077451, -0.178016, 0.192003, -0.24232, 0.016105, -0.119219, -0.056155, 0.269169, -0.23939, -0.061016, -0.101419, -0.409497, -0.035954, -0.055926, -0.312205, 0.11616, 0.132685, -0.03021, -0.508259, 0.023866, 0.277422, -0.265187], [-0.035585, 0.141612, 0.074961, 0.174579, 0.055417, 0.005788, 0.106145, -0.337423, -0.045494, 0.649499, -0.001212, 0.082427, 0.031108, 0.010237, -0.313963, -0.002407, 0.051814, -0.051921, -0.034863, 0.727454, 0.004648, 0.064581, -0.164878, -0.025942, -0.15606, -0.183119, 0.148043, -0.329768, -0.013434, -0.062804, 0.168371, 0.006622]], [-0.13316, 0.007824, 0.15228, -0.134976, 0.136974, -0.087134, 0.110185, -0.120784, -0.219613, -0.048586, -0.161136, 0.256074, -0.564914, -0.130807, 0.243681, -0.28658, -0.056584, 0.009617, 0.119635, -0.352027, -0.359937, -0.122779, -0.049012, -0.412429, 0.067653, -0.046648, 0.57401, 0.283483, -0.07779, 0.035788, -0.012439, -0.07657]),
    ([[-0.063851, 0.171854, -0.387152, -0.277129, 0.223885, 0.0834, 0.064383, 0.377181, -0.183381, 0.107089, -0.067891, -0.252012, 0.201292, 0.109541, 0.177247, 0.150258], [-0.465685, -0.151166, 0.15302, -0.343178, 0.081463, -0.278653, -0.211247, 0.157814, 0.068631, -0.01724, 0.275054, -0.301372, 0.309839, -0.131132, -0.521343, -0.085481], [-0.157517, 0.087602, -0.065172, -0.442791, 0.292382, 0.101768, -0.140649, -0.004286, 0.230424, -0.143612, -0.327465, 0.293676, 0.144396, -0.021503, -0.281786, 0.115738], [-0.16017, -0.378172, 0.057144, -0.14312, 0.098097, 0.086576, -0.259959, 0.030918, 0.042904, -0.151744, -0.037594, -0.199884, -0.14558, 0.027682, 0.164043, 0.046652], [0.106768, 0.351179, 0.224366, 0.145535, -0.248566, 0.231945, 0.25105, 0.015008, 0.249261, -0.015375, 0.000595, 0.176485, 0.045785, -0.018349, -0.017953, -0.310654], [0.34511, 0.188496, 0.356307, 0.155935, 0.222773, 0.113099, 0.296071, -0.087536, 0.153455, 0.095012, -0.422681, 0.030759, 0.120153, 0.112221, 0.239329, -0.085844], [0.201675, 0.152562, 0.07088, -0.197802, 0.044016, -0.131487, -0.283078, -0.205052, -0.080615, -0.179299, -0.396729, -0.008877, -0.07666, -0.324881, -0.435356, -0.146206], [0.102485, 0.130426, -0.354411, 0.131356, -0.190039, 0.289376, -0.044879, -0.041366, 0.134412, 0.146303, -0.072353, 0.101622, -0.403872, -0.126329, 0.260987, 0.021642], [0.081919, -0.062394, 0.037718, -0.145237, 0.030339, -0.172991, -0.900982, 0.344855, -0.024019, 0.018231, 0.279368, 0.202369, -0.313106, 0.129844, -0.039552, -0.075912], [0.297281, -0.220785, -0.549791, 0.089258, -0.075788, 0.109523, 0.108912, -0.121883, -0.316015, -0.149866, 0.353917, -0.329009, -0.122397, 0.047337, -0.190187, 0.147108], [0.227585, 0.018395, 0.357691, 0.108719, 0.129143, 0.082052, 0.147237, -0.093958, 0.34351, -0.081836, -0.019046, 0.241659, -0.055982, -0.152504, -0.167587, 0.145904], [-0.275118, 0.118403, -0.19037, -0.130428, 0.039855, -0.110999, -0.34282, -0.002386, 0.254593, 0.075068, -0.06047, 0.077778, -0.452897, -0.022582, -0.223082, -0.108567], [0.055696, -0.155216, 0.216589, 0.423415, 0.093186, -0.075958, 0.08553, -0.207071, 0.590411, -0.14359, -0.134047, 0.566932, 0.382132, 0.232459, 0.510386, -0.330762], [0.357408, -0.029276, 0.386527, 0.569585, -0.39445, 0.047087, 0.417215, -0.066353, 0.015584, 0.365084, 0.202091, 0.15846, 0.006433, 0.507327, 0.065474, -0.066148], [0.159075, 0.279833, -0.363424, 0.090623, -0.110257, -0.219757, 0.098747, -0.069627, -0.027287, -0.150138, -0.276411, -0.308677, -0.047043, 0.160297, 0.108939, -0.179072], [0.347784, 0.109951, 0.178124, 0.347785, 0.210284, 0.3034, 0.426151, -0.036033, 0.278136, -0.210755, 0.126669, 0.242946, -0.426937, 0.160956, -0.048, -0.304212], [0.105656, 0.033208, 0.208835, -0.200384, -0.098778, 0.053693, 0.330876, 0.19719, -0.221005, 0.124516, 0.174355, -0.21844, -0.209565, -0.081205, -0.449995, 0.041592], [-0.297927, 0.188544, 0.00754, -0.337762, 0.06691, -0.492858, -0.321679, -0.325653, -0.172782, 0.289419, 0.042138, -0.327253, -0.285625, -0.090411, 0.687981, -0.237034], [-0.055223, 0.082803, 0.178157, 0.082514, 0.156394, 0.146107, 0.135024, -0.233832, -0.587796, 0.352719, 0.171057, -0.165572, 0.192987, 0.221873, 0.033499, -0.076445], [-0.245941, -0.029893, 0.08827, -0.371367, 0.374556, -0.164626, -0.162473, 0.099722, 0.093282, -0.302168, 0.078763, 0.192519, 0.037425, -0.425225, -0.48766, 0.085753], [-0.438485, 0.013331, -0.062199, -0.247807, -0.151226, -0.212009, 0.14023, -0.561484, -0.124817, 0.151201, -0.183592, -0.246876, -0.402093, -0.179685, -0.123623, -0.009864], [-0.260109, 0.102119, -0.360651, -0.227963, 0.084271, -0.053075, -0.595814, 0.034749, -0.172925, -0.204409, -0.165087, -0.193791, 0.243609, -0.539744, 0.195285, -0.086628], [0.111698, 0.162484, -0.054736, 0.201966, -0.031963, 0.342225, 0.123827, 0.237077, 0.043794, 0.16313, -0.113776, 0.127913, 0.166362, 0.266936, 0.11671, -0.211783], [0.110548, 0.440395, 0.031634, -0.361218, -0.059876, 0.023804, 0.233739, -0.32216, -0.314463, 0.237049, 0.389424, -0.232644, -0.324919, 0.26443, 0.1072, 0.127479], [-0.536144, -0.152786, 0.046893, -0.02932, -0.027389, -0.28246, -0.236078, 0.085499, -0.015874, -0.003092, -0.626637, 0.072573, -0.378849, -0.142292, -0.782119, 0.160991], [-0.237698, -0.027979, -0.087203, -0.443218, -0.068489, -0.36467, -0.212608, 0.105544, -0.029677, -0.002853, 0.242055, 0.198401, -0.347869, 0.052984, 0.084578, 0.103659], [0.004284, -0.153436, -0.471281, 0.048275, -0.058058, -0.258066, -0.020995, -0.142044, -0.236814, -0.235853, -0.142781, -0.243521, -0.226429, -0.02021, -0.039248, 0.085836], [-0.208928, -0.127831, 0.235527, -0.350678, 0.262216, -0.136071, -0.092383, 0.354161, -0.109827, -0.178189, -0.189249, 0.140944, 0.199167, -0.264534, -0.105228, 0.186478], [0.133365, 0.181488, -0.685866, -0.055734, 0.124261, -0.220237, -0.375464, -0.108689, 0.006165, -0.246004, -0.15078, -0.228384, -0.375169, -0.603651, 0.316423, 0.076902], [0.054397, 0.11097, -0.805636, 0.259137, 0.128878, 0.002551, 0.012216, -0.004296, -0.407237, 0.097012, -0.159235, -0.252001, -0.026322, -0.065087, -0.082566, -0.081995], [0.070974, -0.007157, 0.243648, -0.40277, -0.00502, -0.111197, -0.671959, 0.176218, -0.070489, -0.351459, -0.066155, -0.045519, -0.338005, -0.223576, -0.693842, 0.121535], [0.004521, -0.033416, 0.027048, 0.133071, 0.245904, 0.152155, 0.169147, 0.279904, 0.032014, 0.249867, -0.234671, 0.171789, 0.111767, 0.115554, 0.041034, -0.060412]], [-0.014434, 0.046517, -0.085702, 0.219471, 0.075561, 0.398912, -0.130885, -0.152023, 0.183123, 0.044529, -0.279481, 0.059398, 0.068077, 0.207054, -0.702474, 0.372929]),
    ([[-0.411701], [-0.24342], [1.04472], [-0.50759], [0.181609], [-0.536542], [-0.78386], [0.475964], [0.505554], [-0.326903], [-1.471764], [0.835172], [-0.69974], [-0.514085], [1.901011], [-0.507593]], [0.208802]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.562321557048493e-19, -1.1343482335572739e-18, -5.671741167786369e-19, -1.8905803892621232e-19, 1.1343482335572739e-18, 0.0, 0.0, 0.0, 1.2572534685440513, 0.0, 10.639801906763806, 28.799029822078616], [6.72849538126497, 19.8058956257327, 34.925161910299096, 44.72666958616459, 57.636846319296275, 1.1243487056677264, 0.9903909371630099, 0.9446913427593254, 0.8452682282888074, 0.7575381583058208, 1067.5606541222405, 2053.035725905977, 4046.2991308026644, 0.6179080587708422, 0.3127841039679992, 16.705331990766382, 34.22120473597514], [0.6490227662551226, 0.3600762349252582, -0.1832944003136108, -0.09733275171505404, -0.10066480452612983, 0.8482856190320313, -3.2486297440276823, 6.1619984101404714, -3.540934484781457, 3.1035095898512166, -0.6019915249010418, 0.06557241931064624, -0.1855095648918967, -0.002102755891520747, 0.4105229362630337, 0.0003153876484541001, 0.00020715056621090425], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.568942287397852e-19, -1.1353413431096778e-18, -5.676706715548389e-19, -1.892235571849463e-19, 1.1353413431096778e-18, 0.0, 0.0, 0.0, 1.2574354160321035, 0.0, 10.646654416476506, 28.81409587438229], [6.7313766144182345, 19.814344420832445, 34.939862329400235, 44.74491730719747, 57.65822173996363, 1.124692105470714, 0.9907164467252751, 0.9450073668099886, 0.8455601770714993, 0.7578101560190531, 1067.786637740178, 2053.787560401509, 4047.179122347928, 0.6181145992138308, 0.3129209935553393, 16.710984248831306, 34.23237508220454], [2.9421164343350763, 0.7964360421909548, -1.0744230210945729, -0.9234918162695617, -0.8475723430517554, 1.4775048291769728, -4.635873707445536, 9.213729195881914, -4.907517756528247, 6.23467901404692, -2.386919742393157, 0.4346797676233983, -1.5368709559241014, 0.001829056109893383, -0.22349228908983945, -0.0002742367847445596, -0.00018005766511644536], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.257441, 0.0, 0.0, 5.84623, 5.846549, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987547], [1.124422, 0.990545, 0.944803, 0.844787, 0.75662, 2.57683, 18.120787, 0.790527, 15.924298, 0.695578, 0.651228, 0.54053, 1.677429, 1.677372, 0.999993, 4.403117, 1.087628, 0.310614, 132.397858, 334.626434, 1066.212891, 1.340704], [([[0.105998, -0.103315, -0.0459, 0.171529, -0.690408, 0.769942, -0.014431, -0.073824, 0.281842, 0.175262, 0.402249, -0.349275, 0.214119, -0.170192, 0.105359, -0.249869, 0.152262, 0.080705, -0.158696, 0.031346, -0.278826, 0.22858, 0.232247, -0.275007, 0.116055, -0.078796, 0.013085, 0.044866, 0.017551, -0.052555, 0.063514, 0.09537], [0.101745, 0.027125, 0.184616, -0.088204, -0.499481, 0.119899, -0.073504, -0.119061, 0.125965, -0.215599, -0.223051, 0.007061, 0.071726, -0.134356, -0.100149, 0.116859, -0.009394, -0.124475, 0.088121, -0.130965, 0.03412, -0.014671, -0.088115, 0.020203, 0.247198, 0.082387, 0.02037, -0.269269, -0.222085, -0.036611, -0.067142, 0.248574], [-0.104261, 0.210655, 0.120628, -0.032991, -0.018846, -0.145176, 0.049045, -0.223134, -0.058791, 0.01262, -0.241946, 0.074858, 0.160388, -0.032898, 0.008312, 0.331742, -0.143463, 0.135299, -0.167659, 0.103157, -0.043015, -0.009871, -0.276601, 0.162534, -0.064318, 0.122221, 0.019901, -0.030029, 0.081804, -0.044687, -0.266889, -0.114721], [0.036367, -0.233188, -0.227641, -0.189253, 0.404016, -0.132124, -0.198553, 0.214502, 0.146773, -0.273913, -0.137743, 0.001473, -0.28369, 0.468121, 0.168877, 0.252592, -0.005864, 0.134776, 0.144698, -0.12184, 0.022095, 0.300229, -0.148504, 0.490254, -0.232186, -0.370538, -0.101035, 0.058064, 0.074948, 0.053244, 0.011851, 0.109983], [-0.300455, 0.136387, 0.078898, -0.173497, 0.277653, 0.05057, 0.557975, -0.010465, 0.217469, -0.188637, 0.036151, -0.016929, -0.042514, 0.03277, -0.019818, 0.294775, 0.032497, -0.581812, -0.304832, 0.306756, 0.206865, 0.054288, -0.323084, -0.20193, 0.108383, -0.133606, -0.200598, 0.374162, 0.475577, -0.219858, 0.100148, -0.043148], [-0.262018, 0.130919, 0.178148, -0.165501, 0.343289, 0.049021, -0.037258, -0.063989, 0.131207, 0.04589, -0.251787, -0.05564, 0.125414, 0.249639, -0.032624, 0.300729, -0.292374, 0.07254, -0.271631, 0.114442, 0.191577, 0.015546, -0.055702, -0.175861, 0.23764, 0.19661, -0.423317, 0.019562, -0.014753, -0.336513, 0.12448, 0.143547], [0.125208, 0.288097, -0.280965, -0.276107, 0.597191, 0.171912, 0.331377, -0.221576, -0.26983, -0.463023, 0.141155, -0.356844, 0.411225, 0.1018, -0.013276, -0.018646, 0.047091, -0.627633, -0.096493, 0.32257, 0.220525, -0.439436, -0.30574, -0.407663, 0.167667, -0.224496, -0.558176, -0.138833, 0.507196, 0.031228, -0.151247, 0.202182], [0.170404, 0.04625, -0.005417, -0.164598, -0.016716, -0.214585, -0.175816, -0.150534, -0.263478, -0.259401, -0.191672, 0.0245, 0.001963, 0.083575, -0.250048, -0.020525, -0.125489, 0.054212, -0.295531, -0.012983, 0.009873, -0.138156, 0.200821, -0.082495, 0.253351, 0.074809, -0.098312, 0.083478, -0.066855, -0.001359, -0.051575, -0.105693], [0.180904, -0.154932, 0.201588, -0.081298, -0.111449, 0.067958, 0.028323, -0.140879, -0.389366, -0.628964, 0.08195, -0.276217, -0.215601, 0.013599, -0.159739, -0.219524, 0.049704, 0.117793, 0.243878, -0.58745, -0.394165, -0.01446, -0.025232, 0.133907, 0.009575, 0.220583, 0.030642, -0.08456, -0.665729, 0.390624, -0.364552, 0.022246], [0.189444, 0.270684, -0.562463, -0.122478, 0.107964, 0.096331, 0.180569, -0.70982, 0.103577, -0.173698, -0.16778, -0.036061, 0.062241, 0.189785, -0.013928, -0.072935, -0.575669, 0.002234, -0.297724, -0.25944, -0.026515, -0.054044, 0.186644, 0.088617, -0.229901, -0.009295, -0.1325, -0.180989, -0.019423, -0.212783, -0.331585, 0.27992], [-0.025447, -0.183671, -0.042329, 0.09484, 0.081498, -0.110486, -0.101597, -0.04097, -0.060755, -0.014356, 0.033743, 0.021446, -0.235681, -0.215448, 0.04303, 0.212687, 0.016706, -0.053015, 0.109525, -0.018472, 0.074456, 0.237361, -0.164717, 0.072252, 0.065651, -0.102504, -0.115281, -0.128866, 0.11609, -0.326834, -0.104903, 0.273465], [-0.406919, 0.375384, 0.162952, 0.621415, -0.145089, -0.015444, -0.159572, -0.042294, 0.136233, 0.170143, -0.208978, -0.049295, 0.269897, 0.029764, -0.021075, -0.03417, -0.421592, 0.033447, -0.119838, -0.021553, -0.387845, -0.218768, 0.049185, -0.039383, 0.108436, -0.143336, 0.196975, 0.041958, -0.065254, 0.170546, 0.355856, -0.037739], [0.205144, 0.095345, -0.083149, -0.225354, 0.014472, -0.034618, -0.183121, -0.076067, -0.204657, -0.107444, 0.025235, -0.241148, -0.013143, -0.12791, -0.041088, -0.095155, 0.001041, 0.008876, 0.014527, 0.159409, -0.062684, -0.103379, -0.426409, -0.078089, 0.071699, -0.033862, 0.411884, 0.202137, -0.008582, 0.097077, 0.142814, -0.271224], [-0.166162, 0.039272, 0.001638, -0.333144, -0.052884, -0.051801, -0.210601, -0.129267, 0.050551, 0.009596, -0.164759, -0.248794, 0.194822, -0.270892, -0.110284, -0.158189, 0.355836, -0.002785, -0.053799, 0.282176, 0.129794, 0.094029, -0.16901, -0.154178, 0.262744, -0.091661, 0.058708, -0.071103, -0.03711, -0.173406, 0.387121, -0.427454], [0.095818, -0.189853, 0.099797, 0.277673, -0.152495, 0.082206, 0.035873, 0.030117, -0.253272, -0.135857, 0.035606, -0.038892, 0.004925, -0.085967, 0.397585, -0.088627, -0.060743, -0.157772, -0.070677, -0.129646, 0.125046, 0.537698, -0.012047, -0.005861, 0.345553, -0.328428, 0.218328, -0.204343, -0.351043, -0.36299, 0.192471, -0.064914], [0.036851, 0.053002, 0.255507, 0.155155, -0.026355, -0.161283, -0.11196, -0.153679, 0.073476, 0.015601, -0.160574, 0.228723, 0.26999, 0.287212, -0.059338, 0.092446, 0.074782, 0.198121, -0.068068, -0.41955, 0.297161, -0.011862, -0.026176, -0.113783, 0.337105, 0.327767, 0.047078, 0.044564, -0.696204, -0.039175, 0.204845, -0.049009], [0.234524, 0.161148, -0.640532, 0.241702, -0.131312, 0.108585, 0.007485, 0.534204, 0.571323, 0.544708, -0.031816, -0.136722, -0.441693, -0.057356, -0.244858, -0.352359, 0.010311, -0.028745, 0.180927, 0.287299, -0.819389, 0.377097, 0.19733, -0.073194, 0.010013, 0.312302, -0.06893, 0.316771, 0.016888, 0.178589, 0.169461, -0.223155], [0.005699, -0.060019, 0.308281, -0.218488, -0.006154, -0.068335, 0.131006, 0.230582, -0.253978, 0.175237, -0.324536, -0.285189, -0.097296, 0.090871, 0.161055, 0.075886, -0.602072, -0.014718, -0.120064, -0.353574, -0.058353, -0.075991, 0.204087, 0.026687, -0.020494, -0.187949, -0.191515, 0.138075, 0.004578, 0.383318, 0.263891, -0.303779], [-0.134425, -0.067408, -0.170997, 0.001923, -0.016349, 0.060358, 0.039055, 0.022206, -0.08246, -0.152693, 0.068124, 0.263542, 0.029848, 0.049598, -0.183145, 0.032163, 0.057961, -0.053078, 4.5e-05, -0.186686, 0.188672, -0.044597, 0.029773, -0.023328, -0.021362, 0.020522, 0.010209, -0.058302, -0.012357, -0.106863, 0.166456, -0.092334], [-0.436906, 0.395812, -0.08953, -0.162783, 0.051543, -0.027264, -0.101756, -0.075198, 0.238475, -0.033558, 0.162965, 0.184974, 0.419325, 0.502464, -0.511825, -0.01532, -0.284284, -0.15105, -0.149509, -0.066744, 0.09106, -0.14705, 0.096086, 0.163999, 0.156794, 0.081161, 0.070394, 0.231759, 0.212116, -0.228156, 0.013709, 0.038108], [-0.118346, 0.421409, 0.234046, 0.075659, 0.077547, -0.079496, -0.265106, -0.095647, -0.035622, 0.212483, -0.003438, 0.057641, 0.025478, -0.235555, -0.070059, -0.052883, 0.063274, 0.584849, -0.181161, 0.132826, 0.025292, -0.42368, -0.062576, -0.053807, 0.012295, 0.459077, 0.111001, -0.494441, -0.142857, 0.008742, 0.498513, -0.160638], [-0.012931, -0.170401, 0.017817, 0.241563, -0.002555, 0.011598, -0.007454, -0.003291, 0.255466, -0.02164, 0.125194, 0.439811, -0.217682, 0.149894, -0.399923, -0.086601, 0.09434, -0.002211, 0.056942, 0.443598, 0.253438, 0.080981, -0.064564, -0.149339, 0.06634, -0.256111, -0.073326, -0.064651, -0.048555, 0.107203, -0.061746, -0.19383]], [-0.028014, -0.054069, 0.090217, -0.218037, -0.036931, -0.116598, 0.183952, -0.124976, -0.125814, -0.623538, -0.138606, 0.523604, -0.149643, -0.282485, 0.036602, -0.387555, -0.026117, 0.328904, 0.011356, 0.140368, -0.226675, -0.054635, 0.286361, 0.003111, 0.239219, 0.133907, 0.518125, 0.40133, -0.113409, -0.015193, -0.091854, 0.067847]), ([[-0.121786, 0.165574, -0.49743, -0.018698, 0.036532, 0.021631, 0.261181, 0.022174, -0.082719, 0.269027, 0.073694, -0.039836, 0.077376, 0.031317, 0.154336, 0.220371], [-0.106211, -0.335053, 0.171446, 0.084551, 0.311294, -0.280744, 0.066635, 0.080057, 0.249592, -0.263199, -7e-06, -0.052157, 0.152349, -0.13655, -0.380193, 0.001766], [-0.287124, 0.071051, -0.008619, -0.118326, 0.134208, -0.097776, -0.32694, 0.151781, 0.291061, -0.290814, -0.210155, 0.200903, -0.275512, -0.059385, -0.591093, 0.156258], [-0.258337, -0.161391, -0.176664, -0.165384, 0.169056, -0.181968, -0.036289, -0.166446, 0.108127, -0.25683, -0.024326, -0.283078, -0.079035, -0.016912, 0.165512, -0.004792], [0.423852, 0.130899, 0.363474, -0.097306, -0.050775, 0.331764, 0.336855, -0.105718, 0.329765, 0.171672, -0.213976, 0.288891, -0.086991, 0.164854, -0.016119, -0.314902], [0.079451, 0.167376, 0.303033, 0.163043, 0.218899, 0.319395, 0.059829, 0.001988, 0.313926, 0.645617, -0.081858, 0.140942, -0.061341, -0.374365, 0.249145, 0.054734], [0.251788, -0.046468, -0.085309, -0.373739, 0.434832, -0.11634, -0.365948, 0.216226, 0.304091, -0.256596, -0.087207, 0.100824, 0.066066, -0.455577, 0.277941, -0.310154], [0.285106, 0.198458, 0.424097, 0.204112, -0.053795, 0.285295, 0.111401, 0.039293, -0.138406, 0.121036, -0.131467, -0.071097, -0.213686, 0.083163, -0.004418, -0.046672], [-0.07975, 0.182346, 0.156865, 0.448379, 0.066231, -0.09762, -0.566689, 0.122194, -0.105855, -0.311032, -0.006189, 0.14388, -0.324158, 0.510563, -0.234996, 0.113592], [0.1714, -0.034742, 0.056184, 0.239569, -0.300829, 0.144374, -0.389416, -0.617222, -0.71907, -0.194036, 0.062308, -0.136902, -0.007929, 0.231006, 0.289308, -0.007866], [0.222644, 0.117802, -0.123581, -0.128499, 0.084089, 0.08959, 0.211757, 0.035754, 0.430987, -0.083949, -0.143695, 0.231491, 0.168755, -0.118592, -0.03037, 0.127675], [-0.12968, 0.093029, -0.068434, 0.137047, 0.059174, 0.010427, -0.26551, 0.072245, 0.115259, -0.053238, 0.057464, 0.156921, -0.270937, 0.100478, -0.809451, 0.083231], [0.277083, 0.063384, 0.000847, 0.12799, 0.096725, -0.186147, 0.166424, -0.109428, -0.015089, -0.451187, -0.170678, 0.100174, 0.099386, -0.017403, -0.563626, -0.419517], [0.241631, 0.090796, 0.127551, 0.210332, -0.377405, -0.079317, 0.470508, -0.259311, -0.253725, 0.160524, 0.079733, -0.032723, -0.324739, 0.126605, -0.49552, 0.057427], [0.145266, 0.15236, -0.291837, 0.053798, -0.223112, -0.096845, -0.049667, -0.025041, -0.243693, -0.128011, -0.081389, -0.164631, -0.097608, 0.233762, -0.098736, -0.10393], [0.180999, 0.015329, 0.211384, 0.178746, 0.212702, 0.19725, 0.217718, 0.009911, 0.139109, -0.224795, 0.063858, 0.290761, -0.132884, 0.088044, -0.068448, -0.357878], [0.096262, 0.100519, -0.334067, 0.02983, -0.228346, -0.084685, 0.118701, 0.224075, -0.656994, -0.08501, -0.130175, -0.377351, -0.093023, -0.072071, -0.297576, 0.161989], [0.255087, 0.364081, -0.627572, 0.122575, -0.528905, 0.303222, -0.115259, -0.176428, -0.322082, 0.306749, -0.071981, -0.335739, -0.195576, 0.046488, 0.377594, -0.197153], [0.069336, 0.122632, -0.414238, 0.160119, 0.099057, 0.131639, 0.215077, 0.137609, -0.535423, 0.163879, -0.059874, -0.223916, 0.095039, 0.108622, 0.033037, -0.004246], [-0.031704, -0.260671, 0.307954, -0.439373, 0.143092, -0.102629, -0.116289, -0.222398, -0.030995, -0.200153, 0.066748, -0.150254, 0.030469, -0.141989, 0.022356, -0.086486], [-0.295779, 0.016019, -0.149108, 0.053562, -0.118994, -0.319913, 0.291498, -0.04629, -0.036323, 0.084434, -0.142627, -0.289969, -0.079196, -0.085173, -0.147707, 0.15506], [-0.213248, 0.256951, -0.320755, 0.048865, -0.166267, -0.009597, -0.281815, -0.012312, -0.055762, -0.266199, -0.184484, -0.160847, -0.651931, -0.2633, 0.225956, -0.179686], [0.094882, 0.16911, 0.051487, -0.002994, -0.10341, 0.308805, 0.05023, -0.070132, 0.15529, 0.173108, -0.162697, 0.285904, 0.352841, 0.181612, 0.078513, 0.031604], [0.075595, 0.12479, 0.411484, 0.0289, -0.211207, 0.026984, -0.149111, -0.234819, -0.149681, 0.166332, -0.018129, -0.120243, 0.082485, -0.007537, 0.029636, 0.035476], [-0.438921, -0.223216, -0.054563, -0.047453, 0.079721, -0.307442, -0.256736, 0.297037, 0.013262, -0.362441, -0.198138, 0.13662, -0.633563, -0.162878, -0.386915, 0.053711], [-0.441907, -0.127034, -0.21774, -0.150585, -0.01009, -0.252986, -0.114134, 0.105632, -0.020565, -0.385615, 0.07866, 0.154535, -0.259918, -0.004317, 0.265565, 0.087734], [-0.029578, -0.104976, -0.614013, 0.156166, -0.154015, -0.244365, -0.024549, -0.285275, -0.225973, 0.048798, -0.117113, -0.117106, 0.091781, 0.046951, 0.03402, -0.008486], [-0.06069, -0.096646, 0.351402, -0.451334, 0.005799, -0.178656, 0.185796, -0.046572, 0.053633, -0.098002, -0.082882, 0.033379, -0.504892, -0.522394, -0.162674, 0.237999], [0.168019, 0.207546, -0.471133, -0.262904, -0.031313, -0.397658, -0.164232, -0.014453, -0.322088, -0.406864, -0.216079, -0.365605, -0.07125, -0.215756, 0.341754, 0.145531], [0.089049, 0.183915, -0.215342, 0.187243, 0.063192, -0.001082, 0.017146, -0.153435, -0.913719, -0.046182, -0.134991, 0.183031, -0.146739, -0.154834, -0.106691, -0.078582], [0.301026, -0.045305, 0.262363, -0.319964, 0.198125, 0.02135, -0.513813, 0.26579, -0.046386, 0.220239, 0.056951, -0.04481, -0.077807, -0.257133, 0.215075, 0.184763], [0.039631, 0.030154, 0.035642, 0.15246, 0.133949, 0.286551, 0.087337, 0.321817, 0.039627, 0.302238, -0.142421, 0.206681, -0.171913, 0.007082, -0.13634, -0.131514]], [-0.23218, -0.078435, 0.036299, 0.070969, 0.165887, 0.272402, -0.446815, -0.037148, 0.141766, 0.192126, -0.023014, -0.058416, -0.051726, 0.375119, -0.781338, 0.24885]), ([[-0.980594], [-0.388578], [1.689107], [-0.656268], [0.546541], [-0.914078], [-2.073737], [0.641877], [1.274513], [-1.270913], [-0.478558], [1.171618], [-1.712368], [-0.852627], [3.737945], [-0.812646]], [0.275759])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.75191, -9.2e-05, 9.2e-05, 0.0, 0.052846, 0.052846, -0.0, 0.0, 0.0, 1.243668, 5.879185, 5.879185, 0.0]
SEQ_TSD = [1.079295, 0.132411, 0.132411, 0.099558, 1.45424, 1.45424, 0.666687, 0.219428, 0.17921, 0.60273, 1.666237, 1.666237, 0.649942]
SEQ_XM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.257423, 0.0, 0.0, 5.846561, 5.846561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987482]
SEQ_XS = [1.124955, 0.99101, 0.94523, 0.845186, 0.757041, 2.577918, 18.130564, 0.790721, 15.945622, 0.69697, 0.6513, 0.540497, 1.677766, 1.677766, 0.999993, 4.402977, 1.087599, 0.310604, 132.411407, 334.802429, 1066.728027, 1.341266]
SEQ_YSC = 1.939010739326477
SEQ_WI = [[-0.070723, 0.641992, 0.151581, -0.06173, -0.293843, -0.276257, -0.656172, 0.027625, -0.239344, -0.235494, -0.183657, -0.234974, -0.342108, -0.084699, -0.163604, 0.000495, -0.336846, 0.492282, -0.165879, 0.146184, -0.24549, 0.115349, -0.571242, -0.141601, 0.094685, -0.364175, -0.384133, 0.181337, 0.313127, -0.253937, -0.206113, -0.24206, 0.215715, 0.023049, 0.180858, -0.187986, 0.128494, 0.427522, 0.000753, 0.233631, -0.183145, -0.1169, 0.159493, -0.2748, 0.127654, -0.391265, 0.01121, -0.382073], [0.14534, 0.228249, 0.376502, 0.343937, -0.531317, 0.217278, -0.233975, 0.17619, -0.749926, 0.246498, -0.423143, 0.159494, 0.201964, 0.330062, 0.376864, -0.353392, 0.155861, -0.390389, -0.382869, -0.687103, -0.060599, 0.239572, -0.256318, -0.083233, 0.280983, 0.397001, 0.111105, 0.021686, 0.125573, -0.49908, -0.155528, 0.283092, -0.131455, -0.034294, -0.086677, -0.044649, 0.355605, -0.461287, -0.128023, 0.262764, -0.122562, 0.109483, 0.093908, 0.001051, 0.205877, 0.016979, 0.19344, -0.117434], [-0.312428, -0.053227, -0.611212, 0.386862, -0.059426, -0.362536, 0.510373, 0.083405, 0.299207, 0.254747, -0.164098, -0.202841, 0.038471, 0.341404, -0.105371, 0.099847, -0.233581, -0.104461, 0.166293, 0.076437, 0.037279, 0.183218, -0.025952, 0.417085, -0.140903, -0.030012, -0.126303, 0.132521, 0.248471, 0.063379, 0.1025, -0.090317, -0.64571, 0.154518, -0.013422, -0.091842, 0.170745, -0.217369, -0.250388, 0.269179, -0.126979, -0.149632, -0.204849, -0.186822, 0.426878, -0.376025, -0.229231, 0.445701], [0.028459, 0.265206, -0.494601, 0.601563, 0.292737, -0.084178, 0.04239, 0.241129, -0.466503, 0.047021, -0.19421, -0.008107, -0.027431, 0.10855, 0.007754, -0.023462, -0.291499, -0.0269, -0.115782, -0.157769, 0.32355, 0.590205, -0.233398, -0.045458, 0.165932, 0.216443, -0.041688, -0.128324, 0.22481, -0.574746, -0.188021, 0.134378, -0.378615, -0.018354, 0.179375, 0.271801, -0.152825, 0.075537, -0.167264, -0.099009, -0.031559, 0.036487, 0.146249, -0.082176, 0.078453, 0.130287, 0.186309, 0.058031], [-0.559195, -0.196617, 0.55824, 0.616058, 0.33246, 0.752715, -0.524673, -0.067054, 0.129647, 0.241248, 0.256408, -0.20465, -0.142542, 0.928703, 0.028523, -0.016378, 0.160199, 0.038675, -0.524719, -1.729159, 0.182382, 0.911782, -0.359161, 0.035965, -0.350638, 0.18665, -0.674116, -0.23065, 0.013434, 0.256763, 0.615241, 0.037415, 0.351183, -0.102981, 0.118825, 1.19631, 0.328437, -0.395474, -0.199587, -0.036847, -0.01162, 0.14652, 0.672963, -0.267021, -0.051048, 0.56057, -0.068618, -0.277996], [-0.087166, -0.456993, 0.215153, -0.237072, 0.022919, -0.33122, 0.101158, -0.822287, -0.021385, -0.787911, 0.159653, 0.394757, 0.762524, -0.371026, -0.393064, -0.255969, 0.478964, 0.389736, 0.210468, -0.026942, -0.182133, 0.573355, -0.08358, -0.330476, 0.723976, -0.847531, 0.071494, 0.473477, -1.469327, 0.058566, -0.195186, 0.154137, -0.081996, 0.385341, 0.122025, 0.009621, -0.20466, -0.400883, -0.146039, -0.566883, 0.479545, 0.957039, -0.096426, -0.160262, -1.022717, 0.072868, -0.409744, -0.275228], [0.002659, 0.314688, -0.055577, 0.056775, -0.054432, -0.705574, -0.198954, 0.0434, 0.212602, 0.38208, 0.693969, 0.222788, -0.415491, -0.015559, 0.381258, -0.374553, 0.326197, 0.004311, -0.397149, -0.24576, 0.3288, 0.320553, 0.172807, -0.197013, 0.387445, 0.219178, -0.098308, -0.29427, 0.381612, -0.037732, -0.037692, 0.135886, 0.296599, -0.212429, -0.137781, 0.190535, -0.074114, 0.155051, 0.245813, -0.300417, 0.059626, -0.236479, 0.266998, 0.052918, -0.093821, 0.038714, 0.069984, -0.286555], [-0.106499, 0.572735, 0.334913, 0.147678, -0.096932, -0.26506, -0.09184, 0.166861, 0.172087, 0.651643, 0.548652, 0.119977, -0.258006, -0.009984, -0.013655, -0.166314, 0.159988, -0.130337, -0.153145, -0.224508, -0.076788, 0.03292, 0.291085, -0.190005, -0.055729, 0.376102, 0.030678, 0.215285, 0.227232, 0.109136, 0.224948, -0.225537, 0.104216, 0.080666, 0.165237, 0.127608, -0.153187, -0.146857, 0.209954, -0.0108, 0.130381, -0.045654, -0.23003, -0.038778, 0.327031, 0.053111, 0.405123, 0.055021], [-0.369308, 0.512953, 0.217858, -0.2656, -0.009149, -0.565318, 0.01662, -0.235609, -0.003804, 0.386045, 0.705833, -0.123711, -0.51315, -0.099764, 0.061874, 0.115868, 0.376661, 0.103549, -0.563091, -0.107049, -0.047404, 0.519508, 0.109501, -0.376572, 0.240101, 0.28882, -0.490848, -0.430214, 0.199326, 0.030922, 0.003457, -0.161625, 0.092303, 0.180169, -0.214018, -0.039427, 0.252252, 0.217254, -0.106104, -0.050091, -0.000231, 0.239859, -0.194842, 0.101862, -0.232434, 0.174883, 0.247223, 0.167576], [0.683076, -0.185986, -0.065947, -0.063991, 0.150832, -0.446915, 0.694034, -0.172427, 0.155574, -0.192224, 0.136372, -0.276097, 0.090013, 0.087069, 0.335014, -0.361514, -0.43654, 0.25912, -0.03692, 0.209659, 0.452968, 0.303447, 0.559529, 0.34917, -0.024175, -0.053436, 0.008485, -0.738591, 0.356215, 0.330696, -0.496771, -0.54053, 0.322977, 0.287372, 0.417278, -0.128615, 0.607622, -0.170311, 0.686785, 0.354616, 0.389584, 0.24816, 0.183583, 0.192615, 0.069094, 0.104825, 0.135858, -0.04939], [-0.063094, 0.003643, -0.727339, 0.309471, -0.174407, 0.325419, -0.490068, 0.464592, -0.040874, -0.113998, -0.021576, 0.158801, 0.25391, 0.472607, 0.267686, 0.012122, -0.434147, -0.34602, 0.07698, -0.295733, 0.0362, 0.080702, -0.140732, -0.425342, 0.054838, 0.287714, 0.279445, -0.382892, -0.30273, -0.291982, -1.055595, 0.54626, -0.178946, 0.344886, -0.150785, 0.12621, 0.122629, -0.072488, 0.074626, -0.226018, -0.226088, 0.058334, -0.101526, 0.027001, 0.242918, -0.090234, -0.055672, 0.126561], [0.110758, 0.049766, 0.158799, -0.444468, 0.076079, 0.177247, -0.290174, 0.327914, 0.364216, 0.043025, -0.022326, -0.254958, 0.324435, 0.620402, -0.005799, -0.315606, -0.026772, -0.123768, 0.390298, -0.25065, -0.247568, 0.085769, 0.177951, 0.03001, -0.223529, -0.398014, -0.220159, -0.251469, -0.764396, -0.832062, -0.667957, 0.265858, -0.377719, 0.14673, 0.266898, -0.062597, 0.220876, 0.031614, -0.0981, -0.145459, -0.155018, -0.115464, 0.135739, 0.191808, -0.303697, 0.160921, -0.058478, 0.07952], [-0.136175, 0.084528, 0.06057, 0.416143, -0.071636, 0.016706, 0.145016, 0.048088, -0.222795, -0.044347, -0.150557, 0.040825, -0.167559, -0.220857, -0.158482, 0.120006, 0.22345, -0.042168, -0.434978, -0.416779, 0.170312, -0.071127, 0.089562, -0.227497, -0.469429, 0.253477, 0.226052, -0.175636, 0.554138, 0.459722, -0.488433, 0.252642, -0.005635, 0.246067, 0.08635, -0.042348, -0.076353, 0.218851, -0.052084, -0.030222, -0.066451, 0.308043, 0.091445, 0.007027, 0.433501, 0.186002, 0.177048, 0.093939]]
SEQ_BI = [-0.056483, -0.32115, 0.031369, -0.21786, 0.211721, 0.229992, 0.017663, 0.03984, 0.0059, 0.127821, 0.400164, 0.183122, -0.138261, -0.294102, 0.034948, 0.28807, 0.875892, 0.067306, 0.499214, 0.445585, 0.641934, 0.967945, 0.277309, 0.360181, 0.830414, 0.64021, 0.547707, 0.432761, 0.159842, 0.678589, 0.694308, 1.03375, -0.102949, -0.047796, -0.052595, -0.119549, 0.17032, -0.205365, 0.004772, -0.088124, -0.149606, -0.012832, 0.125628, -0.238142, -0.126909, -0.144469, 0.156929, 0.199875]
SEQ_WH = [[-0.370316, 0.187823, -0.233603, -0.130754, -0.243253, -0.211488, -0.322555, -0.623205, 0.320602, -0.647971, -0.98591, -0.952182, -0.185048, -0.195375, -0.613903, 0.356158, -0.930299, 0.069996, 1.080623, -0.266797, -0.270753, 0.003503, 0.423129, -0.175498, -0.57961, 0.032471, -0.614489, 0.485698, -0.092181, -0.352603, 0.956281, -1.354879, 0.305491, -0.105517, -0.060077, 0.50317, 0.04503, 0.46229, 0.450713, 0.124675, 0.357648, 0.423459, 0.391712, -0.145858, -0.143074, -0.142884, -0.31135, 0.104637], [0.314397, -0.4729, -0.285935, 0.323088, -0.399781, 0.29589, -0.048647, 0.008004, -0.131076, -0.432899, 0.101131, 0.162347, 0.3509, 0.21852, -0.147893, 0.107206, -0.191457, -0.216415, -0.573947, -0.472101, -0.425258, -0.049938, -0.369237, 0.180133, -0.447108, -0.500665, 0.008058, -0.479662, 0.260965, -0.123939, -0.773067, 0.055099, -0.022595, -0.088947, 0.10186, -0.540371, -0.030449, -0.307879, 0.078941, -0.010134, 0.049749, 0.070941, -0.317604, 0.170431, 0.155772, 0.287804, -0.129345, -0.242242], [0.063164, -0.252907, 0.471808, 0.185504, -0.540033, 0.377038, -0.115921, -0.09123, -0.141938, -0.315844, 0.053476, -0.316504, 0.541661, -0.776647, -0.028466, 0.001462, -0.020638, 0.179442, 0.100661, -0.185372, -0.157809, -0.197432, 0.234068, 0.076626, 0.09961, -0.445052, 0.354668, -1.117029, 0.105015, 0.614759, -0.325193, 0.376545, -0.104123, 0.327645, 0.733572, -0.002817, 0.288772, -0.256877, 0.132751, -0.268028, 0.090996, 0.296125, 0.276525, -0.292575, -0.024474, 0.147453, 0.039525, -0.213282], [-0.652576, 0.652909, -0.082963, 0.454052, 0.306979, -0.142751, 0.12088, 0.091608, 0.483458, 0.145104, 0.406921, 0.569122, 0.039071, 0.130367, -0.75903, 0.11748, 0.551657, 0.332638, 0.039875, 0.449931, 0.142139, 0.617858, 0.705456, 0.17384, -0.11606, 0.327101, 0.122368, 0.119493, 0.389399, 0.30647, -0.114417, 0.244079, -0.135955, -0.093162, -0.167412, 0.258521, 0.614232, 0.193217, -0.186193, 0.010933, -0.132771, -0.138455, 0.623504, -0.35, 0.04942, -0.18115, 0.116482, 0.109008], [0.135234, 0.029453, -0.044982, 0.095654, -0.112234, 0.239792, -0.297337, -0.006285, -0.30969, -0.069521, -0.110075, -0.349223, -0.030873, 0.267256, -0.427354, 0.073453, -0.155487, 0.580496, 0.722698, 0.396799, -0.267562, 0.216142, 0.489628, -0.116423, 0.174909, -0.139002, 0.621694, -0.189831, -0.355208, 0.201335, -0.04121, -0.18759, 0.552497, 0.103869, 0.060145, -0.041824, 0.395628, -0.079923, 0.165786, 0.366853, 0.04284, 0.298177, 0.424056, -0.174792, -0.186841, 0.246473, -0.431114, -0.092257], [-0.086845, -0.08208, -0.096806, -0.194354, -0.275771, -0.184799, 0.199675, 0.239225, -0.061944, 0.095925, -0.141065, -0.288083, -0.273101, -0.114632, -0.179971, 0.024936, -0.769726, 0.031359, -0.465087, -0.123553, -0.521805, 0.099415, -0.171461, -0.431428, -1.281016, -0.156988, -0.40603, 0.038962, -0.263526, -0.425618, 0.374471, 0.167238, -0.24641, 0.111011, -0.212593, 0.202096, -0.388712, 0.517056, -0.06105, 0.167422, -0.693363, 0.200514, 0.175027, -0.299974, 0.31846, -0.197132, -0.448059, -0.36988], [0.007179, -0.010842, -0.196372, 0.043416, -0.766767, -0.663584, 0.093232, -0.557007, -0.522771, -0.612617, -0.853757, -0.100771, -0.062558, 0.011963, 0.073635, -0.181354, 0.031789, -0.601151, 0.525236, -0.822035, -0.360896, 0.155781, -0.000706, -0.085351, 0.683847, -0.233673, -0.163378, -0.402291, 0.178112, 0.004251, 0.499888, -0.104624, 0.248616, -0.106507, 0.652807, -0.16288, 0.242748, -0.362625, 0.292057, -0.275537, 0.082223, 0.36348, -0.180098, -0.314624, 0.229669, 0.04944, -0.404608, 0.11056], [-0.0808, 0.270157, 0.132227, 0.189748, -0.176502, -0.394475, -0.19733, -0.279778, -0.309154, 0.25471, 0.358765, -0.594123, 0.079755, 0.3371, -0.078944, -0.238678, -0.270441, 0.477896, 0.095184, -0.871379, -0.396392, 0.327209, 0.013782, -1.558593, -0.684453, -0.181041, -0.337841, 0.060505, -0.597846, 0.030591, -0.363584, -0.498072, -0.28126, -0.134146, 0.114281, 0.232097, 0.485562, 0.080397, 0.040686, 0.28153, 0.45324, -0.09903, 0.021378, 0.325215, 0.233696, 0.007358, -0.183602, 0.108459], [0.294873, -0.381744, 0.278135, 0.328896, 0.326023, -0.216939, 0.274995, -0.362419, -0.107903, 0.08768, 0.161217, 0.197383, 0.192106, 0.166217, 0.080014, 0.091535, 0.279068, -0.197185, -0.323999, -0.224733, 0.329439, 0.41978, 0.052352, -0.265125, -0.014866, 0.24745, -0.245664, 0.607698, -0.160674, 0.145937, 0.546187, -0.502205, -0.330669, 0.147092, -0.309351, -0.082044, -0.08508, -0.71216, 0.536391, 0.248754, 1.102718, 0.490146, -0.22472, -0.010989, -0.060978, 0.427885, -0.107642, 0.617594], [0.113621, -0.008082, -0.291553, 0.304004, -0.436139, -0.015734, -0.188362, -0.021392, 0.270278, -0.435301, 0.024309, 0.958114, 0.334111, -0.144763, 0.308499, -0.129781, 0.184129, -0.168179, -1.006545, -0.22171, -0.097876, -0.164414, 0.264278, 0.302636, -0.040915, 0.246482, -0.354413, -0.420622, 0.943714, -0.073923, -0.428286, -0.158088, -0.067366, 0.275072, 0.271526, -0.140984, 0.036483, -0.048429, 0.275186, -0.368332, 0.285263, 0.434682, 0.027655, -0.306424, -0.455227, -0.240111, 0.179449, -0.169958], [-0.029075, 0.190315, 0.211696, -0.099745, 0.012272, 0.187956, 0.068262, 0.018034, -0.114953, -0.116267, -0.053611, -0.196516, -0.280652, -0.072204, 0.13936, 0.34267, -0.123034, 0.317116, 0.404215, 0.370629, 0.474026, -0.521192, 0.517895, 0.713447, 0.184485, 0.163794, 0.438804, 0.456823, 0.342502, 0.488227, -0.112016, 0.281951, 0.357663, 0.036675, 0.445222, -0.002616, 0.336012, 0.157653, 0.05053, 0.117074, -0.222306, -0.025553, 0.461214, -0.760399, -0.172969, 0.089981, -0.089471, 0.085179], [0.690317, -0.228563, -0.298438, 0.291084, -0.079417, 0.007419, -0.044231, 0.183664, -0.024883, 0.569257, 0.232715, -0.358603, -0.035941, 0.57789, 0.280664, 0.38975, -0.252128, 0.028065, -0.789895, -0.479948, -0.282693, 0.232615, -0.043779, -0.16976, -0.750918, -0.722247, -0.572399, 0.531715, -0.558666, -0.028075, -0.240381, -0.290925, -0.296079, -0.048726, -0.162388, -0.268742, -0.134619, 0.089987, 0.004428, -0.002759, 0.197849, -0.046065, -0.300377, 0.916758, 0.17287, 0.328937, -0.088184, 0.418297], [0.024456, -0.094564, -0.432666, -0.003816, -0.193954, -0.022089, -0.387088, -0.077288, -0.401854, 0.010488, 0.251609, -0.107709, -0.474778, -0.068596, -0.045105, -0.32509, -0.339507, -0.272264, -0.861156, -0.549765, -0.810042, -0.676849, -0.021255, -0.82552, -1.066973, -0.715019, -0.546759, -0.790517, -0.979829, -0.266575, -1.275812, 0.54654, -0.104154, -0.064094, -0.215237, -0.19808, 0.17105, -0.180299, -0.053996, 0.0771, -0.260119, 0.014134, -0.153142, 0.746012, 0.438418, 0.154343, -0.075581, 0.184076], [0.064877, 0.454395, 0.048586, 0.582065, 0.186349, 0.032172, 0.041989, -0.087013, 0.12704, 0.535986, 0.529918, 0.424698, 0.202447, -0.051396, 0.010336, 0.077218, 0.267757, -0.083914, 0.066768, 0.43096, -0.054247, 0.065329, 0.499643, 0.133081, -0.153839, 0.243849, -0.399577, 0.77233, -0.086427, 0.272113, 0.574925, -0.580566, -0.586908, 0.17104, -0.375499, 0.073516, 0.127888, -0.46303, -0.108763, 0.432944, 0.382218, -0.119695, -0.270522, 0.565882, 0.458444, 0.372913, -0.037439, 0.350832], [-0.353404, 0.252138, -0.047183, 0.159483, 0.07333, 0.342573, 0.320772, 0.354452, 0.340277, 0.354726, 0.026445, 0.565242, 0.193103, 0.10394, -0.024638, 0.204428, 1.019749, -0.057536, 0.634371, 0.435326, 0.491344, 1.019075, 0.549937, 0.507473, 1.371022, 0.420513, 0.409361, -0.035, 0.557549, 0.472393, 0.712961, 1.040452, 0.032526, -0.115126, -0.034139, -0.058451, 0.269957, -0.37341, 0.224632, -0.030022, 0.055301, 0.087911, 0.178576, -0.377737, 0.046863, -0.001003, 0.194117, 0.044512], [0.10386, 0.153419, -0.089906, 0.286589, 0.26965, -0.084384, 0.094438, -0.005141, -0.410275, 0.178087, 0.42014, 0.23769, 0.094318, -0.046145, 0.303815, 0.22942, 0.662209, 0.271463, 0.656613, 0.286113, 1.297739, 0.561619, 0.479988, -0.067135, 0.036179, 0.263699, 0.503332, 0.729547, -0.023184, 0.418737, 0.523575, 1.027453, -0.061853, -0.301083, -0.271393, 0.053765, 0.068034, -0.463051, -0.085013, 0.34958, -0.122318, -0.162311, -0.286486, 0.384391, 0.202374, -0.178117, -0.21219, 0.402606]]
SEQ_BH = [0.123677, 0.091761, 0.370369, 0.204269, 0.262002, 0.065131, -0.194434, 0.227873, 0.390985, -0.130939, 0.244818, 0.58848, 0.235908, -0.248416, 0.063532, -0.033455, 0.724399, -0.053698, 0.343475, 0.615577, 0.83064, 0.733036, 0.239235, 0.589216, 1.176059, 0.457463, 0.666284, 0.250879, 0.26301, 0.597748, 0.405775, 0.975061, -0.040296, -0.218922, 0.026303, 0.091949, 0.002549, -0.363885, 0.032721, -0.041966, 0.257855, 0.089524, 0.019181, -0.118932, -0.142731, 0.073263, 0.228245, 0.28269]
SEQ_HEAD = [
    ([[-0.11414, -0.129264, -0.140686, 0.178763, -0.091115, 0.035541, -0.360899, 0.202489, 0.172298, 0.112695, -0.019996, 0.091399, 0.174126, -0.202782, 0.05918, 0.123959, -0.119686, 0.125343, -0.015608, -0.003749, 0.102313, -0.046344, -0.188959, 0.133523, -0.045864, 0.205737, -0.096214, -0.056094, -0.134184, 0.200488, 0.022411, 0.19936, -0.0593, -0.203098, 0.223027, 0.160252, 0.0074, -0.034385, 0.058916, 0.012549, 0.106963, -0.128576, -0.017844, -0.009086, -0.0336, 0.095139, -0.01827, 0.174822, -0.142356, 0.07845, -0.070426, -0.022507, -0.105978, -0.041229, -0.026596, 0.043944, -0.064113, 0.093856, 0.107109, -0.271333, -0.038674, 0.246456, 0.048208, -0.067901], [-0.055683, -0.086958, 0.017467, 0.203279, 0.168216, -0.099246, -0.140055, 0.238839, -0.120801, 0.029694, 0.066567, -0.003784, -0.133608, -0.091116, -0.050348, 0.016671, 0.145095, 0.016064, -0.150168, -0.220446, 0.15613, -0.101357, -0.073711, 0.040805, -0.066263, 0.163106, -0.001614, -0.180946, -0.191986, 0.127158, -0.108206, 0.139814, 0.11081, -0.185685, -0.016083, -0.069306, -0.086748, 0.039689, -0.032545, -0.046599, 0.099969, -0.131942, 0.036713, 0.132181, 0.139172, 0.200081, -0.052205, 0.018594, -0.149241, -0.135024, -0.229669, 0.165056, 0.134166, -0.009864, -0.104582, 0.055937, 0.062216, -0.147148, 0.100743, -0.080986, -0.211654, 0.040027, -0.172003, -0.015184], [0.212308, 0.236189, 0.172397, -0.067549, -0.041482, -0.01006, -0.103821, -0.015584, -0.183699, 0.06965, 0.121625, -0.085118, -0.063308, 0.092165, -0.115866, -0.056123, -0.062733, -0.050945, -0.109297, -0.062699, -0.021857, -0.012629, -0.235927, 0.147277, 0.152695, 0.135322, 0.003138, 0.056998, 0.067072, 0.092764, 0.106796, 0.050224, -0.132549, 0.062729, -0.162588, -0.035995, -0.050945, -0.022253, -0.189951, 0.002095, -0.144081, -0.114614, -0.121293, -0.103871, 0.204112, 0.01513, -0.011532, -0.172233, -0.116506, 0.003715, 0.045925, 0.024178, 0.004441, 0.139711, -0.203129, 0.024635, -0.154233, 0.039936, -0.04628, -0.069836, -0.218808, -0.002884, 0.098213, 0.041515], [0.223021, 0.044889, 0.002703, 0.08635, -0.143617, -0.05595, 0.203516, 0.273279, 0.056065, -0.064822, 0.125572, -0.15358, -0.131878, 0.097923, -0.07625, -0.186154, 0.012894, 0.013607, 0.182368, -0.174347, 0.13123, 0.254373, -0.246318, -6.9e-05, 0.06533, 0.014381, 0.037084, -0.103179, -0.151887, -0.098602, -0.086673, 0.119413, -0.116264, 0.027343, -0.019428, 0.110328, -0.187751, -0.006827, -0.11463, 0.094721, -0.083374, -0.129281, 0.163775, -0.139624, 0.062777, 0.149071, -0.12227, -0.347615, 0.190751, -0.009649, 0.097054, -0.046276, 0.057161, -0.015149, 0.139985, -0.172904, -0.001683, 0.01933, -0.01893, 0.132156, 0.225291, -0.098137, 0.044505, -0.138799], [-0.070323, 0.077614, 0.067641, -0.050244, 0.0154, -0.106771, 0.122131, 0.3369, -0.087208, -0.11646, -0.090396, -0.011327, 0.226782, -0.195009, -0.119563, -0.151773, -0.216224, -0.019504, -0.232833, 0.069173, -0.041279, 0.298548, 0.031713, -0.096021, -0.097984, 0.213517, -0.027338, -0.005078, 0.07296, 0.241776, -0.287228, -0.051368, 0.047437, 0.164774, -0.050125, -0.276547, 0.074569, -0.014832, -0.00971, 0.185907, 0.176631, -0.097412, 0.328896, -0.267657, 0.057057, -0.098965, -0.144633, 0.265125, 0.083013, 0.077108, 0.406117, -0.043815, 0.124637, 0.215968, 0.210747, 0.198562, 0.248086, -0.19883, 0.17671, 0.378089, 0.314344, -0.067378, 0.273375, -0.04308], [0.015789, 0.033793, 0.138603, -0.05619, 0.082803, 0.085525, 0.005101, 0.027947, -0.086238, -0.062539, -0.055078, 0.014957, 0.049235, 0.120109, -0.117424, -0.069486, 0.16684, 0.111352, 0.14691, 0.037087, -0.049332, 0.008015, 0.074571, 0.004507, -0.150162, -0.087026, -0.016271, 0.074404, 0.044216, 0.153892, 0.035117, 0.034131, -0.084524, -0.109133, -0.052951, -0.097882, 0.076637, 0.035324, -0.06066, 0.271659, -0.240227, 0.035951, 0.081779, 0.097673, 0.075087, -0.093467, -0.077908, -0.131535, 0.198131, -0.055034, -0.154099, -0.000867, 0.050327, 0.024688, -0.124634, -0.043498, -0.096593, -0.062394, -0.147507, 0.019842, -0.272535, 0.012936, 0.119244, -0.238687], [0.218545, 0.211796, 0.136245, 0.158788, 0.159697, -0.066889, -0.08964, 0.112343, -0.092984, -0.061027, 0.083766, 0.159919, -0.274857, -0.152817, -0.054333, 0.10488, 0.002138, 0.105759, -0.023527, 0.08609, 0.093057, 0.023397, -0.085756, 0.150285, -0.137019, 0.052837, -0.141886, -0.088341, -0.015662, -0.096134, -0.186821, -0.052334, 0.115447, 0.055693, 0.174427, -0.159805, 0.017396, -0.138386, -0.045378, -0.0614, -0.24075, 0.019857, 0.111977, -0.160391, 0.188379, 0.198016, 0.073389, -0.178937, 0.068063, -0.027467, 0.050632, 0.087899, 0.022537, 0.126298, -0.115412, 0.061158, -0.13094, -0.07605, -0.067377, -0.418319, 0.159904, 0.080868, -0.020579, -0.323299], [-0.086463, 0.092936, -0.256085, 0.110214, 0.170404, -0.274585, 0.144151, -0.018257, 0.167054, 0.007047, -0.156332, 0.11116, -0.069552, 0.037095, 0.077882, 0.038888, -0.11245, -0.130267, 0.14423, -0.110931, 0.043916, 0.005298, -0.151605, -0.012055, -0.065915, 0.029171, 0.06619, -0.093701, 0.044481, -0.096732, -0.347197, -0.132535, 0.042011, 0.066836, -0.050085, -0.040198, -0.036244, -0.066327, 0.084564, -0.204432, 0.156142, 0.084719, 0.001592, -0.110622, -0.073996, -0.002804, 0.200347, 3.3e-05, -0.292018, -0.085027, 0.184624, 0.115114, 0.155469, -0.11054, 0.044809, -0.022419, -0.022629, -0.03522, -0.161541, -0.061473, -0.057531, -0.077524, -0.128631, 0.169979], [0.39609, 0.108595, -0.031587, 0.148099, -0.050475, 0.168948, 0.195978, 0.169586, -0.072433, 0.053994, 0.091278, -0.202036, 0.093074, -0.169445, 0.076437, -0.045279, -0.281207, 0.1425, 0.072241, -0.238934, 0.157788, -0.141558, -0.244708, 0.081043, -0.12726, 0.209869, 0.0631, -0.120244, -0.055347, 0.081376, 0.0169, 0.120949, -0.073825, -0.032896, 0.063423, 0.055406, -0.193648, -0.202406, -0.209544, -0.020906, -0.240181, -0.126834, 0.035739, -0.122951, -0.025258, -0.072424, -0.078895, 0.141783, -0.000119, -0.02525, 0.036916, 0.153941, 0.011992, 0.196181, -0.181614, 0.16613, -0.208981, -0.193192, -0.098999, 0.173505, -0.215146, 0.246166, -0.02095, -0.289518], [0.074473, 0.005116, -0.057127, -0.153721, 0.042628, 0.020166, -0.133216, -0.047801, -0.05628, -0.004305, -0.051162, 0.064327, -0.172497, 0.067812, -0.098522, 0.165383, 0.076812, 0.00771, -0.023216, -0.104106, 0.084227, -0.002464, 0.080771, -0.129872, 0.179457, -0.151645, 0.04717, -0.118707, -0.441631, 0.188791, -0.043471, -0.223126, -0.110244, 0.104805, 0.197303, 0.029982, 0.208551, 0.146083, 0.253587, 0.074418, 0.01432, -0.169834, 0.035629, 0.114766, -0.045976, -0.115657, -0.256229, 0.009582, -0.012982, 0.053426, 0.077391, 0.0779, -0.146553, -0.110877, -0.048563, 0.170728, -0.031733, -0.392576, 0.078114, 0.051194, 0.03714, 0.180741, 0.154197, -0.062883], [0.159863, -0.133029, -0.010444, -0.07264, -0.080377, 0.075779, -0.090248, 0.165675, -0.099792, -0.03557, -0.034307, 0.180277, -0.01378, 0.065706, -0.081398, 0.151268, 0.005499, 0.033208, 0.003681, 0.001245, -0.156654, 0.055816, -0.224154, 0.257404, 0.042515, -0.151064, -0.067584, -0.175102, -0.051729, -0.058903, 0.040922, -0.115821, 0.015321, 0.072017, -0.038052, -0.058888, -0.0382, 0.057753, 0.085934, -0.226399, 0.168832, 0.135419, 0.089921, 0.033408, -0.123746, 0.19687, 0.095958, -0.170145, -0.079019, 0.022178, 0.161305, -0.060293, 0.013146, 0.019736, 0.018769, 0.130343, -0.189757, 0.107453, -0.134387, -0.0407, 0.124124, 0.031937, -0.139973, -0.059266], [0.169547, 0.255844, 0.086221, 0.190302, 0.157982, 0.021853, 0.098725, -0.064887, 0.057877, 0.087541, 0.227955, -0.301584, 0.226042, 0.029354, -0.24047, -0.306479, -0.05862, -0.008389, -0.127865, 0.077732, 0.240938, 0.078172, -0.130265, 0.313672, -0.29735, 0.064177, -0.141649, -0.175929, -0.202077, 0.09435, 0.025932, 0.114822, 0.171553, -0.240023, -0.005459, -0.0729, -0.320136, 0.002239, -0.247756, -0.124815, -0.040727, -0.189574, -0.14231, 0.00065, 0.445592, -0.03398, 0.163887, 0.014177, 0.240354, 0.022903, 0.12522, -0.150269, -0.156556, 0.137972, 0.142369, 0.221249, 0.075486, -0.131839, -0.014348, -0.005732, -0.295008, -0.010065, 0.136947, -0.006126], [-0.110066, -0.067046, 0.234869, 0.067324, 0.003218, -0.175405, -0.182654, 0.088442, -0.005695, 0.045038, 0.063333, -0.096668, 0.129208, -0.075968, -0.0273, -0.007675, -0.019813, -0.005267, -0.031049, 0.170195, 0.035141, -0.016184, 0.032951, 0.038636, -0.0367, -0.136696, -0.074168, 0.054109, -0.038338, 0.16304, 0.141312, 0.009894, 0.190655, 0.075846, -0.071836, 0.050861, -0.054585, -0.083551, 0.600867, 0.13313, 0.214279, -0.060793, -0.099158, 0.088963, 0.260441, -0.035544, 0.019929, 0.014985, -0.032296, -0.073995, -0.189339, -0.003064, 0.111101, -0.059015, 0.182002, 0.027822, 0.136603, -0.056643, -0.032053, -0.068622, 0.06582, -0.142972, 0.000213, 0.057565], [-0.043344, -0.114598, 0.151311, 0.084351, 0.037108, 0.052405, 0.05415, -0.025756, 0.153466, -0.037831, 0.075308, -0.123987, 0.129152, -0.059846, -0.09667, 0.075321, 0.062608, -0.015192, 0.082577, -0.015702, 0.064389, 0.136937, -0.038677, -0.179915, -0.009122, -0.158756, -0.011143, 0.119478, 0.043511, 0.219895, 0.080382, -0.163945, 0.124652, -0.149455, -0.08629, 0.012621, 0.161174, -0.178758, 0.25723, 0.383722, 0.110858, -0.191996, -0.042062, 0.098498, 0.0985, -0.11478, -0.007937, -0.055084, -0.094818, -0.110075, 0.052113, -0.053523, 0.281585, -0.067532, 0.186027, -0.024394, -0.128303, -0.026221, -0.01815, -0.013861, -0.00189, -0.080948, -0.082714, 0.166332], [0.057237, 0.181648, 0.176134, -0.051625, 0.075087, 0.282088, 0.169555, -0.145306, 0.017571, 0.236162, 0.004565, -0.071041, -0.172685, 0.01594, 0.104406, 0.021549, -0.002154, -0.091552, -0.006187, 0.042002, -0.165773, -0.288128, -0.067306, 0.137401, 0.071954, -0.09152, -0.107625, 0.064181, -0.090336, 0.090587, 0.004239, 0.073546, 0.021831, -0.02783, 0.017931, 0.170699, -0.082653, 0.01695, -0.06575, 0.001006, -0.041318, 0.045734, -0.176866, 0.100932, 0.023998, -0.124068, -0.148444, -0.131571, -0.042951, -0.09866, -0.23843, -0.004676, -0.03763, 0.148963, -0.140274, -0.049654, -0.16182, 0.047702, 0.051331, 0.030327, 0.179947, -0.15219, 0.038926, -0.022578], [-0.298521, 0.348007, -0.031409, 0.231621, -0.188827, 0.040665, 0.162015, -0.076003, 0.265624, 0.120126, 0.11431, 0.026183, -0.045469, 0.098672, -0.233916, -0.041126, 0.097423, 0.110305, -0.093872, -0.059874, -0.025157, -0.332794, 0.156131, 0.168055, -0.115788, 0.032583, -0.109197, 0.089459, -0.041274, -0.027903, 0.124228, -0.118568, -0.151769, -0.092869, -0.035112, 0.185957, 0.096753, 0.070313, 0.01744, 0.189283, 0.025671, -0.042968, 0.086636, -0.165728, -0.004273, 0.040394, 0.144454, 0.145116, 0.020881, 0.021753, 0.070125, -0.032919, 0.145352, -0.241527, 0.007634, 0.289827, 0.004864, 0.038439, 0.225105, -0.100945, -0.118096, -0.122494, 0.127276, 0.076382], [-0.268602, 0.075012, -0.361147, -0.167931, 0.483163, 0.416944, -0.132246, 0.245254, 0.298094, 0.094541, -0.020286, -0.048474, 0.360829, 0.050357, -0.051505, -0.061318, 0.120691, -0.12119, -0.016524, -0.121045, 0.351119, -0.047703, -0.183276, 0.047722, -0.178598, -0.000439, -0.274297, 0.057715, -0.333631, 0.374193, -0.28223, 0.202479, 0.310102, -0.049266, -0.008007, -0.026392, 0.182358, -0.049847, -0.066689, -0.066248, 0.3327, 0.007486, 0.020121, -0.185973, 0.08652, 0.210353, 0.012392, -0.132536, -0.206193, 0.09594, -0.094092, -0.252027, -0.452901, -0.236438, 0.157038, 0.092216, 0.287444, 0.173785, 0.056662, 0.200464, -0.054518, 0.064586, -0.058141, 0.341135], [0.192005, 0.198737, -0.116441, 0.282017, -0.217034, 0.021415, 0.314075, -0.06939, 0.203815, -0.085866, 0.110392, -0.098377, 0.013077, 0.150537, -0.026784, 0.11159, 0.017182, -0.020799, -0.119853, -0.127832, -0.218231, -0.084456, -0.264115, 0.040738, 0.169805, 0.003867, -0.143941, -0.162201, 0.215235, 0.150368, 0.052594, -0.006853, -0.257485, -0.016894, -0.018647, 0.070759, -0.156378, -0.087324, -0.090117, 0.119407, -0.070135, 0.016516, -0.125691, 0.088628, 0.126669, -0.010095, -0.05668, -0.061618, -0.201262, 0.103765, -0.144429, 0.034465, 0.050891, 0.05459, 0.192871, -0.158269, 0.113807, -0.121202, -0.051818, -0.275063, -0.037, -0.024734, -0.146417, 0.188427], [-0.141596, 0.049419, 0.000961, -0.04963, -0.027245, -0.063521, -0.051071, -0.055128, 0.101652, 0.0559, 0.009401, -0.079029, -0.005784, 0.026156, -0.004756, -0.051713, -0.130337, -0.035634, -0.050898, 0.193837, -0.02585, 0.017269, -0.014001, -0.059196, -0.039523, 0.092815, -0.027407, 0.126229, 0.067812, 0.032273, 0.086709, 0.036583, -0.113052, -0.057035, 0.011904, -0.02752, 0.160678, -0.175392, 0.213743, -0.016859, -0.155034, -0.012032, 0.1047, -0.055376, -0.017146, 0.237167, 0.075201, 0.088178, 0.146696, -0.177066, -0.164903, 0.006202, 0.167801, -0.056973, -0.004759, 0.196704, 0.090354, 0.016104, 0.052129, 0.060932, -0.093523, 0.052036, -0.039561, 0.003969], [-0.253898, -0.154661, -0.158996, -0.233875, -0.077955, -0.436197, -0.196304, 0.034449, 0.230426, 0.313195, -0.313975, 0.168656, 0.051053, 0.243898, 0.059377, -0.203604, -0.264534, 0.041342, -0.025122, -0.043121, 0.160664, 0.0084, -0.030054, 0.044277, -0.274416, 0.052093, -0.089754, 0.022171, -0.005909, 0.051498, 0.18906, 0.124283, 0.075996, 0.101472, -0.01272, -0.071078, -0.084235, 0.126872, -0.052087, 0.133737, -0.299835, -0.114728, 0.184801, -0.077304, -0.069652, -0.196889, 0.168518, -0.096151, -0.168664, -0.415326, 0.426146, -0.244032, 0.174671, 0.064264, 0.165028, -0.141083, 0.056924, -0.144619, 0.061705, -0.030021, 0.057084, 0.346645, 0.03369, 0.208147], [0.346117, 0.010201, 0.039122, 0.053049, 0.016991, -0.06743, 0.338859, -0.047885, -0.122733, -0.025977, -0.173808, -0.09887, 0.051698, 0.062962, 0.01827, 0.047518, 0.03678, 0.065021, 0.049227, -0.058454, -0.08846, -0.330592, 0.081327, -0.009393, -0.166652, -0.172199, -0.079273, -0.082158, -0.090172, -0.037579, 0.122893, -0.058057, 0.416889, 0.095738, 0.268783, 0.184573, -0.084124, -0.054705, -0.165796, 0.179896, -0.18101, -0.160909, -0.270917, 0.028597, 0.004018, -0.148854, -0.03139, -0.025276, 0.06675, 0.09399, 0.102914, -0.083637, -0.084595, -0.144332, 0.021452, 0.038491, -0.293844, 0.171255, 0.017594, -0.150859, -0.038706, -0.281019, 0.173214, -0.130739], [-0.004582, -0.131154, -0.236005, 0.036816, 0.127329, 0.044078, 0.339639, 0.013286, -0.055859, 0.00915, 0.318113, 0.047345, -0.228593, 0.454368, 0.007009, 0.127897, 0.129799, 0.005401, 0.034707, -0.008412, -0.172467, -0.088629, 0.049445, 0.081263, -0.087888, 0.020285, -0.199384, 0.174212, 0.00439, -0.066535, 0.06, 0.003391, -0.137008, 0.002719, 0.107724, -0.022925, -0.059813, -0.221717, 0.249499, -0.079458, -0.106832, 0.04067, -0.010027, 0.122877, 0.014776, 0.184881, -0.110927, 0.058799, 0.114262, -0.059205, -0.06205, -0.284917, -0.206897, 0.725605, 0.046707, 0.00679, -0.106914, 0.09408, 0.039303, 0.066256, -0.068235, 0.157759, 0.022061, -0.028369], [-0.326288, -0.12344, 0.018419, -0.027261, -0.119327, -0.255277, -0.235686, 0.026713, 0.146733, 0.2785, 0.022415, -0.036169, 0.118969, -0.239895, 0.244583, 0.063177, 0.032524, -0.013718, 0.0204, 0.000302, 0.119473, -0.030256, -0.248691, 0.119649, 0.225657, -0.101404, -0.205678, 0.037145, 0.048874, 0.060609, -0.225452, -0.143353, -0.117212, -0.074554, 0.038441, -0.067163, -0.162284, 0.123947, -0.105166, 0.184703, -0.072604, 0.087529, -0.019297, 0.138714, -0.145223, -0.067411, -0.194825, 0.035989, -0.259956, -0.175021, 0.111105, -0.177322, 0.062951, 0.104296, 0.299752, 0.031899, 0.206571, -0.307036, -0.003745, -0.166026, 0.061433, 0.021611, -0.178918, -0.032595], [0.069052, 0.063456, 0.166517, 0.046988, -0.148156, 0.123532, -0.229469, -0.096141, 0.049825, -0.139095, 0.18086, -0.074389, 0.048022, 0.108138, -0.009576, 0.05166, -0.194554, 0.054768, -0.117041, 0.168436, -0.002971, 0.0158, -0.052166, -0.118739, 0.113794, 0.12167, 0.163979, -0.275418, -0.091287, -0.004437, -0.140868, 0.044849, 0.155181, -0.048199, 0.001284, -0.121461, 0.09689, 0.040057, 0.315867, -0.042493, -0.074811, 0.042445, -0.224132, 0.093711, 0.143983, 0.001927, 0.161623, 0.071275, -0.035332, 0.033127, 0.068809, 0.266019, -0.006606, 0.023609, 0.06256, -0.067647, 0.068334, -0.215501, -0.116933, -0.182558, 0.063337, 0.024199, -0.084024, 0.079445], [0.22704, -0.215234, 0.001914, 0.038811, -0.152054, -0.080369, 0.040444, -0.07801, -0.241684, -0.137733, 0.353184, -0.356792, -0.222904, 0.163601, 0.014883, -0.23318, -0.037959, -0.088908, 0.017189, 0.012829, -0.255632, 0.096514, -0.164337, -0.155353, 0.259029, 0.163183, -0.092607, -0.213139, -0.205116, -0.191078, -0.434074, -0.073703, -0.062755, 0.126707, -0.182632, 0.288233, -0.159699, -0.005121, -0.034386, 0.302067, -0.053406, -0.268965, -0.133846, 0.025414, -0.158723, -0.145374, 0.08482, -0.167221, 0.030408, -0.066306, -0.175177, 0.100874, 0.167295, -0.056171, -0.066758, 0.032075, -0.036797, -0.181927, -0.191269, -0.322191, -0.149891, -0.212889, -0.066092, -0.490956], [-0.139448, 0.051341, -0.087358, 0.213166, 0.169716, 0.200538, -0.106736, -0.131614, -0.197906, 0.060128, -0.011, -0.124419, 0.042112, 0.04436, -0.676558, -0.285608, -0.159054, -0.16641, -0.23718, 0.014066, -0.17991, -0.057186, -0.242885, -0.007275, -0.190153, 0.042298, -0.43434, -0.095426, -0.577991, -0.133356, 0.167566, -0.083384, -0.024794, -0.142842, -0.029229, 0.13125, -0.06272, -0.065749, -0.058898, -0.119785, 0.191981, -0.18613, -0.010826, -0.431586, -0.21215, -0.138891, 0.238718, 0.186754, -0.125639, 0.020647, -0.066042, -0.280296, -0.14895, -0.087189, -0.217502, -0.141344, 0.137991, -0.211971, 0.254227, 0.021396, 0.007868, 0.065795, -0.115079, -0.032609], [-0.414954, -0.270642, -0.604047, -0.062023, 0.333774, 0.181246, -0.065553, -0.087874, -0.058917, -0.003103, 0.024845, -0.189625, 0.120846, -0.041515, -0.184187, -0.226027, -0.357332, -0.168644, -0.42254, -0.412242, -0.02304, 0.104268, -0.004714, 0.052104, -0.087964, -0.109367, -0.333419, -0.072813, -0.180649, -0.010956, 0.066694, -0.004118, -0.147342, 0.220662, 0.136392, 0.016512, 0.11509, -0.114738, -0.067241, 0.011236, 0.126165, -0.299195, 0.163488, -0.066588, -0.21301, -0.238021, 0.259127, 0.473256, -0.082829, -0.058391, -0.150363, -0.433575, -0.068207, -0.26965, 0.047617, 0.428165, 0.197011, -0.212265, 0.341316, -0.083832, -0.348311, -0.358094, -0.133421, -0.054802], [-0.084862, 0.10031, -0.044958, 0.172788, -0.105459, -0.065247, -0.036152, -0.037029, 0.065403, -0.153643, 0.255891, 0.255081, -0.208882, -0.042435, -0.05151, -0.100702, 0.078245, -0.034353, -0.089788, -0.103836, 0.044541, 0.028078, 0.174075, 0.006775, 0.022803, 0.085294, 0.182049, 0.240697, 0.20033, 0.186187, 0.125146, 0.127874, 0.208998, 0.084011, -0.280663, 0.083058, -0.023476, -0.1997, 0.208916, -0.020521, 0.073266, -0.010493, 0.010548, -0.103666, -0.044831, -0.191112, 0.172937, -0.211952, -0.004368, 0.028996, 0.018386, -0.243459, -0.138391, -0.000619, -0.302784, 0.017238, -0.156612, -0.155731, -0.008743, -0.066342, 0.100503, -0.078712, -0.046013, -0.243837], [-0.006269, 0.250753, 0.097917, -0.535832, -0.16932, 0.142444, -0.129224, -0.005651, 0.032287, -0.14399, -0.521921, -0.165555, 0.155803, 0.278891, 0.070168, 0.265686, 0.136062, -0.07139, 0.031627, -0.115607, -0.261293, 0.180728, 0.032892, -0.181229, 0.227607, -0.212639, -0.14408, -0.105141, -0.330645, -0.009779, -0.23346, -0.167631, 0.163059, -0.132172, -0.494617, 0.303142, 0.071611, -0.046492, 0.152747, 0.117982, -0.141623, 0.057646, 0.036168, 0.377289, -0.313645, 0.161194, -0.049676, -0.03569, 0.012108, 0.266772, 0.13097, 0.040633, 0.279263, 0.032232, -0.421926, 0.001964, -0.094468, -0.320521, -0.220678, -0.230808, -0.017662, -0.071643, -0.289175, -0.296118], [0.26824, 0.023888, -0.290752, -0.170301, 0.220795, -0.054624, -0.063821, -0.070951, -0.054045, 0.429333, -0.093633, 0.160279, 0.216257, 0.09875, -0.115556, -0.072165, 0.20504, -0.026572, 0.014579, -0.213544, 0.190258, 0.043767, 0.126987, 0.276633, 0.060331, 0.161066, 0.222235, -0.415133, -0.161962, 0.121549, -0.020004, 0.063719, -0.038485, 0.27933, 0.200812, -0.108224, -0.084369, 0.269096, -0.143464, -0.021668, -0.065674, -0.056884, 0.098743, 0.018986, -0.065724, -0.127648, -0.157311, 0.245426, -0.140528, 0.133109, -0.173383, -0.075594, -0.413226, 0.248102, 0.301055, 0.059145, 0.22917, -0.104858, 0.358012, -0.069146, 0.122925, -0.142563, 0.129655, 0.190593], [0.101419, 0.317314, -0.023883, -0.424593, -0.139511, 0.110262, -0.359235, 0.067063, 0.2232, 0.138044, -0.578061, 0.212029, -0.108449, 0.076909, 0.188497, -0.064289, 0.289273, -0.136251, 0.226099, -0.244512, -0.09688, 0.15505, 0.048382, -0.209777, -0.143412, 0.143027, 0.227353, -0.202352, -0.065596, -0.046501, -0.168215, 0.115792, -0.039141, -0.229601, 0.172336, -0.179536, -0.10211, 0.186566, -0.239288, -0.217977, -0.094573, 0.065783, 0.022399, 0.108146, 0.037622, 0.393874, -0.129538, 0.122551, -0.080425, 0.145103, -0.084511, 0.296083, 0.06912, 0.057887, 0.024503, -0.176693, 0.082473, 0.05298, -0.180733, -0.253619, 0.101592, 0.247035, -0.07331, 0.358576], [0.035655, -0.052491, 0.063434, -0.012999, -0.066402, 0.208188, -0.203024, -0.258207, -0.198676, -0.120696, 0.1368, -0.222012, -0.176887, 0.141196, -0.160878, 0.118412, -0.24964, -0.262489, 0.069243, -0.041559, -0.158639, -0.046081, 0.104857, -0.114951, -0.175345, -0.042951, -0.142256, -0.051563, -0.126462, -0.069903, 0.024449, -0.109778, 0.168621, 0.040947, -0.141013, 0.085424, -0.090129, 0.181673, 0.421524, 0.133982, 0.241531, 0.072648, -0.06176, -0.025849, -0.246019, 0.00514, -0.071167, 0.113828, -0.032172, 0.005358, 0.086648, 0.158798, -0.011327, -0.058363, -0.289742, -0.082137, 0.058217, -0.0299, -0.313596, -0.13699, 0.094751, -0.285747, -0.215941, -0.115196], [-0.492771, -0.187627, -0.30532, -0.121912, -0.131883, -0.064723, -0.120143, 0.011842, -0.281885, 0.064232, 0.073131, -0.005278, -0.029582, -0.069797, -0.190663, -0.250412, -0.088381, -0.205854, -0.111991, -0.020868, 0.081782, 0.010933, -0.241138, 0.011473, -0.116895, -0.149001, -0.201014, 0.065355, -0.255527, -0.168819, 0.053436, -0.19663, 0.195856, 0.038805, -0.262842, 0.004095, -0.05304, -0.199925, -0.047371, 0.142743, 0.099586, -0.070832, 0.025451, -0.147078, -0.182803, -0.150292, 0.049158, 0.040836, -0.107256, -0.151226, 0.015164, -0.419006, -0.005437, -0.14193, 0.192593, 0.131663, 0.119239, -0.004317, 0.150657, 0.048303, -0.148385, -0.090306, -0.070047, -0.000155], [-0.009989, 0.155196, 0.308444, 0.294448, -0.001082, 0.097958, 0.156056, 0.225806, 0.040716, -0.06914, 0.11317, 0.28463, 0.145761, -0.176077, 0.010965, 0.166938, 0.036523, 0.099197, -0.11585, -0.016443, 0.196014, -0.223086, 0.243669, 0.225093, 0.12376, 0.233719, 0.203488, 0.01921, 0.321611, 0.040281, 0.167462, 0.29675, -0.05152, 0.109252, 0.133907, 0.101679, -0.025506, -0.218817, 0.161039, 0.125983, 0.198254, 0.120669, -0.074232, 0.172498, 0.508899, 0.089381, 0.059121, 0.099341, 0.060174, -0.081379, 0.146184, 0.25415, 0.135513, 0.024679, 0.172625, 0.097536, 0.04409, -0.033376, 0.052227, 0.308661, -0.036135, 0.330073, 0.074274, 0.483119], [0.195751, 0.009165, 0.059079, 0.026814, 0.032565, 0.029246, 0.192006, -0.279481, -0.076458, 0.325934, 0.215772, 0.112526, -0.153357, -0.001222, 0.063201, 0.030478, -0.086245, -0.100996, -0.073608, -0.145524, 0.139638, -0.092179, 0.048702, 0.393563, -0.080085, 0.002736, 0.136753, -0.045968, -0.005161, -0.030006, 0.211714, 0.003771, -0.207422, 0.1948, 0.083984, -0.031102, 0.169754, 0.152249, -0.044356, -0.037002, -0.338915, 0.03854, -0.16857, -0.073848, -0.010358, -0.071683, 0.197834, -0.035281, 0.156674, -0.135341, -0.289827, 0.165523, -0.126752, 0.121603, 0.144, 0.158868, 0.033762, 0.08342, -0.085043, 0.072636, -0.130319, -0.126705, 0.153376, -0.010353], [0.273915, -0.00251, 0.009389, -0.046238, 0.01312, 0.454806, -0.0564, 0.147458, -0.019684, -0.014337, -0.052128, 0.192253, -0.144871, 0.265089, -0.18, -0.125345, -0.035394, 0.070241, 0.063043, -0.207349, -0.291907, 0.020369, -0.061719, 0.105765, -0.135268, 0.060223, -0.115402, 0.107713, 0.041794, -0.03735, -0.091987, -0.005653, -0.291832, 0.221415, 0.2532, 0.099743, 0.149499, -0.397651, -0.168585, -0.152389, 0.187081, -0.215709, -0.11979, 0.045574, -0.171991, 0.184243, 0.006343, 0.321836, 0.470786, -0.219254, 0.20391, -0.147636, 0.077198, -0.258496, 0.147339, -0.160347, -0.232819, -0.201833, -0.005657, 0.156132, -0.014303, 0.17458, 0.068665, 0.187335], [0.044391, 0.110029, -0.017518, 0.098712, 0.195, -0.119618, 0.039188, -0.073203, -0.244364, 0.041087, -0.026319, 0.127079, 0.119427, -0.070818, -0.151814, 0.057875, 0.074579, 0.097158, 0.101857, 0.113513, -0.033234, -0.013181, -0.092837, -0.10726, -0.061002, -0.005603, 0.064935, 0.193076, -0.141732, -0.055597, 0.037747, -0.155193, -0.173735, -0.093317, -0.009947, 0.187282, -0.083668, -0.018018, 0.018785, -0.070833, -0.198361, -0.080407, 0.200265, -0.184926, -0.117894, -0.199886, -0.014326, -0.022199, 0.026898, 0.088395, -0.031689, 0.052117, -0.106529, -0.091953, 0.05114, -0.054275, 0.096963, 0.050369, 0.022807, 0.016392, -0.049717, -0.279154, -0.096399, -0.18994], [-0.12867, -0.018833, 0.227194, -0.127413, 0.107904, 0.015194, -0.07234, 0.053566, 0.10964, -0.029171, -0.354686, 0.18086, 0.364555, -0.315143, 0.034636, -0.054754, 0.066065, 0.208173, 0.371762, 0.091213, 0.104281, 0.079888, -0.105335, 0.075339, -0.110935, 0.073581, 0.171298, -0.207037, -0.258749, 0.235074, -0.274032, -0.225441, -0.102626, 0.171202, -0.054219, 0.024363, -0.224281, 0.195331, -0.237853, -0.029139, -0.176474, 0.086883, 0.189552, 0.040722, -0.016323, 0.090553, -0.30165, -0.003426, -0.351161, 0.003933, -0.105883, 0.073309, -0.241428, 0.076248, 0.129594, -0.148664, 0.060367, -0.006633, 0.172682, 0.044697, 0.025357, -0.09117, -0.160663, -0.014104]], [-0.018336, -0.236847, -0.045452, 0.102882, 0.133135, -0.322637, -0.219081, -0.253464, -0.043725, 0.052985, -0.180722, 0.146018, -0.079941, -0.445505, 0.097332, -0.108552, 0.146996, 0.100026, 0.172792, 0.117817, 0.06765, -0.01364, -0.124852, -0.113196, 0.006477, -0.085596, -0.043776, 0.17134, -0.174871, 0.003561, 0.011725, -0.089856, 0.097966, -0.060617, -0.187862, 0.288436, -0.146408, 0.084733, -0.043911, 0.018273, -0.040634, -0.017377, 0.252732, -0.006831, -0.129209, 0.019262, -0.037063, -0.327883, -0.156713, 0.065315, -0.054058, -0.006015, -0.292996, -0.11722, 0.168219, -0.08518, 0.159986, 0.287488, 0.13977, -0.077925, -0.223966, -0.151196, 0.003544, -0.185972]),
    ([[-0.022845, 0.125672, 0.220856, 0.20646, 0.123692, 0.241806, 0.174154, -0.017206, 0.120508, 0.214173, 0.13212, 0.08961, 0.05306, -0.111053, 0.257671, -0.082047, 0.309578, 0.23973, -0.247259, -0.035206, 0.118883, 0.041495, 0.132104, 0.218563, 0.032983, 0.012574, 0.060921, 0.007989, 0.078611, 0.055513, -0.367024, 0.136822], [-0.027743, -0.00193, 0.138154, 0.043772, 0.19528, -0.11837, 0.087617, 0.109772, 0.015429, -0.004723, 0.006694, 0.163735, -0.041749, 0.018682, 0.266269, 0.026142, 0.127158, 0.053019, -0.081628, 0.04993, -0.03974, -0.069492, 0.264734, 0.163374, 0.156709, -0.012858, 0.114225, 0.143948, -0.107581, 0.201186, 0.096844, 0.105094], [-0.081918, 0.214161, 0.082635, 0.011818, -0.212466, 0.074643, 0.204954, -0.14073, 0.222674, -0.351247, -0.095657, 0.084167, 0.199728, -0.259932, -0.037751, 0.019119, 0.461876, 0.050354, -0.252101, -0.001276, 0.037769, 0.187412, 0.307536, 0.08679, -0.053218, -0.069112, -0.258238, 0.08806, -0.317741, 0.253365, -0.299482, 0.149072], [0.067697, -0.320463, 0.114276, -0.074062, 0.128915, -0.222957, -0.291195, -0.047286, -0.002646, 0.127512, 0.023011, -0.211608, -0.071721, -0.056819, -0.216061, -0.082455, -0.379838, -0.119497, 0.027432, 0.028679, -0.076418, -0.007241, -0.201713, -0.433437, -0.21662, 0.090572, 0.136235, -0.064937, -0.051158, -0.232049, 0.016863, 0.035614], [-0.02604, -0.07665, -0.507457, -0.296646, 0.075713, -0.07028, -0.479991, 0.043249, -0.238678, 0.005483, -0.123302, -0.034176, -0.105333, 0.108858, 0.012725, 0.159973, -0.411463, 0.149865, 0.232776, 0.119919, 0.044253, 0.036141, -0.052994, -0.116802, 0.039257, 0.054491, -0.02065, -0.10887, -0.047523, -0.130929, 0.108428, 0.061666], [0.146284, -0.104886, -0.055777, -0.116215, -0.146504, -0.210608, 0.105264, -0.302996, 0.21827, -0.170347, -0.254907, 0.109233, 0.023201, 0.140622, 0.029182, -0.116963, 0.211121, -0.282853, 0.384334, 0.102788, -0.362436, 0.105107, 0.136289, -0.021932, -0.055362, -0.025465, 0.085592, -0.120912, -0.286776, -0.062313, 0.102021, -0.021216], [-0.065193, -0.223617, 0.118209, 0.021535, 0.062449, 0.050977, 0.024367, 0.067037, -0.327135, 0.236604, 0.120483, 0.086299, 0.115718, -0.117249, 0.186894, -0.021139, -0.05584, 0.312809, -0.189938, -0.128305, 0.205555, 0.167438, 0.168194, 0.229104, 0.062546, 0.159671, -0.027704, 0.048632, 0.044189, 0.01483, -0.151045, -0.008663], [-0.022422, 0.169832, 0.018479, 0.127263, -0.012334, 0.061006, 0.243101, 0.028034, 0.010991, 0.121203, 0.166998, 0.113941, 0.017407, -0.071473, -0.044583, 0.132423, 0.131608, -0.07047, 0.071083, 0.015176, 0.091146, -0.075151, -0.113834, 0.096393, 0.138665, 0.086359, 0.167695, 0.079277, 0.049848, 0.119968, -0.180389, 0.013872], [0.009802, -0.115812, 0.117197, 0.084748, -0.101776, -0.049134, 0.19838, 0.253029, 0.167523, -0.077798, 0.41158, -0.094337, -0.052318, -0.11517, 0.113761, 0.113191, 0.172091, -0.051605, 0.001174, 0.20963, 0.12523, 0.000478, -0.110508, 0.03052, -0.023594, 0.15143, 0.024458, -0.03835, 0.169088, 0.075021, 0.126459, 0.08331], [0.049685, -0.118608, -0.202261, -0.148285, -0.136329, -0.064369, -0.421558, 0.202389, -0.181038, 0.052397, -0.097331, 0.089358, -0.09438, -0.129461, -0.005363, -0.072571, -0.400791, -0.06701, 0.048117, 0.185462, 0.1149, 0.033312, -0.055629, -0.224387, -0.063083, -0.072097, -0.021459, -0.085346, 0.141232, -0.152633, 0.155017, -0.047556], [0.144909, 0.011121, -0.024784, -0.140274, 0.044553, -0.095699, -0.283924, 0.116269, -0.018945, 0.118543, 0.138473, -0.228564, -0.026905, 0.012084, -0.318992, -0.036798, -0.532843, 0.050164, -0.200414, 0.109218, -0.011471, 0.057749, -0.058045, -0.469396, -0.211267, 0.10405, -0.064209, -0.037632, -0.014562, -0.136053, 0.016577, -0.166205], [-0.038011, 0.089161, -0.029239, 0.024583, 0.102426, 0.050904, -0.277452, 0.096836, -0.287076, 0.138369, 0.291682, 0.074188, 0.089065, -0.133314, 0.174009, 0.015091, -0.041385, 0.023578, -0.20205, 0.10456, 0.066693, 0.047785, 0.02896, 0.091096, -0.094102, 0.032986, -0.036462, 0.088318, 0.105284, 0.105273, -0.001095, 0.006128], [0.116066, -0.005724, 0.0026, -0.289211, 0.174158, -0.18604, -0.117693, 0.07999, 0.016087, 0.117115, 0.111301, -0.124242, -0.175065, 0.06616, -0.100601, -0.371462, -0.055733, -0.021924, 0.277266, 0.116183, -0.022401, -0.13861, -0.02209, 0.066173, -0.141465, 0.048102, 0.053396, 0.010745, 0.21079, -0.109639, 0.001135, -0.177051], [-0.083272, 0.420551, 0.252462, 0.117375, -0.065475, 0.135551, 0.386422, -0.066138, 0.094116, 0.066187, -0.03223, 0.17226, 0.206498, 0.032542, 0.124773, 0.046337, 0.294606, 0.004047, 0.094884, -0.137004, 0.052517, 0.137416, 0.141768, 0.234917, 0.160639, -0.184102, -0.183276, 0.116059, -0.059866, 0.027238, -0.224959, 0.019286], [-0.084601, 0.148331, 0.010348, -0.01526, -0.051848, 0.02311, -0.140654, -0.071803, -0.255766, 0.095324, 0.246869, 0.049352, 0.075409, -0.244919, -0.03295, 0.055981, -0.017413, -0.22728, 0.056218, 0.15492, -0.063098, 0.023055, 0.115191, 0.040156, 0.144769, -0.023951, -0.153287, 0.115032, -0.073813, 0.011839, -0.041134, 0.000756], [0.019095, 0.260049, 0.114847, -0.006578, 0.047294, 0.167471, -0.009157, -0.149846, -0.143375, -0.091005, -0.207562, -0.122288, 0.108021, 0.175629, 0.014744, 0.108403, 0.178554, -0.036569, 0.229719, -0.092754, -0.126132, 0.072982, 0.135095, 0.085531, 0.016013, -0.226844, 0.013874, 0.027999, -0.118213, 0.160024, -0.317724, 0.004812], [-0.103273, 0.174337, 0.005884, -0.019378, -0.127844, -0.070233, 0.042783, -0.005841, 0.125171, -0.069239, -0.212812, 0.068016, 0.176643, 0.082928, -0.048064, 0.146671, 0.13565, -0.064823, -0.001736, -0.121707, -0.094506, 0.13054, 0.051827, -0.093703, 0.049904, -0.05397, -0.019303, 0.09446, -0.23264, -0.091771, 0.102969, 0.066795], [-0.067053, 0.097592, -0.007012, 0.088506, 0.095821, 0.081991, -0.17978, 0.026995, -0.066139, 0.080674, -0.128225, 0.129979, 0.111775, 8.7e-05, 0.023541, 0.001496, 0.201999, -0.041599, 0.131242, -0.036998, 0.128755, 0.153849, 0.037369, 0.031791, -0.087607, 0.091983, -0.031101, 0.059313, 0.017541, 0.045387, 0.099863, 0.007964], [-0.055342, 0.184006, -0.00238, -0.108605, 0.046305, -0.076237, 0.160666, -0.143926, -0.058005, -0.101639, 0.025337, 0.127051, 0.108415, -0.300066, 0.091553, 0.135321, 0.15109, -0.116859, -0.099497, 0.069937, -0.003126, 0.174349, 0.016605, -0.087136, 0.153273, -0.384602, -0.162838, 0.131995, -0.080932, 0.057676, 0.27407, -0.039996], [0.054317, -0.185405, 0.049995, 0.08054, -0.110813, -0.021301, -0.043913, -0.112976, -0.174372, -0.070252, -0.220142, 0.029599, -0.030537, -0.207819, -0.037335, 0.106371, -0.192711, -0.499215, -0.146843, 0.050276, -0.12921, 0.073241, 0.037057, 0.144136, 0.077246, -0.254723, -0.069256, 0.102731, -0.090848, -0.078243, -0.131509, 0.137102], [-0.047567, 0.018217, -0.025861, 0.008608, 0.052386, 0.126473, -0.22146, 0.07903, -0.035894, 0.084543, -0.045159, 0.050033, -0.108897, 0.073451, -0.012667, -0.015989, -0.185856, 0.02914, -0.011853, -0.036065, 0.035757, 0.00332, -0.000587, 0.153087, 0.038199, 0.17529, 0.168098, -0.13428, -0.001477, -0.032414, -0.059552, -0.097805], [0.058741, -0.180312, -0.127061, 0.14733, -0.129211, -0.336179, 0.345003, -0.222028, -0.085304, -0.228076, -0.175939, 0.05738, 0.104746, 0.148814, 0.002412, -0.053424, -0.346225, -0.071851, -0.002906, -0.066955, -0.044235, -0.028801, -0.46291, -0.485303, 0.031001, 0.041291, 0.071255, -0.089301, 0.029712, -0.160618, -0.003139, 0.057837], [0.088495, -0.150144, 0.086744, 0.062981, 0.026755, 0.222545, 0.028204, 0.285815, -0.203075, 0.007445, 0.211609, 0.094726, 0.125836, -0.08501, 0.068811, 0.089395, -0.0375, 0.169418, 0.173951, 0.065205, 0.019172, 0.052095, 0.007846, 0.183635, 0.051626, 0.400045, 0.098325, 0.08206, 0.016228, -0.031427, -0.050375, 0.089351], [0.123821, -0.2779, -0.133708, -0.104608, -0.007463, 0.103065, -0.257033, 0.1205, 0.033212, 0.014478, -0.147466, 0.081196, -0.115057, -0.127706, 0.24397, 0.03733, -0.01715, -0.013979, 0.071852, 0.179287, 0.092524, 0.095012, 0.097171, 0.123993, -0.067869, 0.035735, 0.100186, 0.031863, 0.003082, -0.061879, 0.046972, -0.069981], [-0.146938, -0.138711, 0.126134, 0.127063, 0.088007, 0.107672, -0.123955, -0.032243, 0.00548, 0.094447, -0.166781, -0.051621, 0.114199, 0.09114, -0.085133, 0.143933, -0.152822, -0.193022, 0.044758, -0.044338, -0.191745, 0.012569, 0.007188, -0.07525, -0.016795, -0.243174, -0.045911, 0.138644, -0.145671, -0.05419, -0.06203, 0.127666], [0.070976, -0.158749, -0.112311, -0.125958, 0.016604, -0.079602, 0.039614, 0.012514, 0.058544, 0.123238, 0.008834, -0.040396, 0.020943, -0.253048, 0.062215, 0.134544, 0.165245, -0.090617, -0.231949, -0.00081, 0.100692, -0.076126, 0.058454, -0.023411, -0.003538, 0.073472, -0.019588, 0.131886, 0.018098, 0.130582, -0.050868, -0.071095], [-0.026971, 0.030823, 0.147665, -0.015089, -0.078229, -0.071973, 0.026902, 0.012968, 0.046422, 0.044145, 0.299421, 0.151413, 0.057407, -0.197917, 0.095465, -0.047539, 0.030534, -0.056892, -0.038712, -0.083592, 0.177005, 0.000144, 0.025651, 0.01398, 0.08431, 0.129044, -0.045409, 0.096536, -0.108707, 0.140194, 0.022789, 0.002257], [-0.110561, 0.023191, -0.054271, -0.013748, -0.089755, -0.046688, -0.072733, -0.248578, -0.287343, -0.126289, -0.187348, 0.062822, -0.019474, -0.165538, -0.00303, 0.049985, 0.11121, -0.084821, -0.145943, -0.225445, 0.041846, 0.172681, -0.12939, 0.016374, 0.058616, -0.270433, -0.212271, 0.133159, -0.161033, 0.083328, -0.02141, 0.059451], [-0.190876, -0.043077, 0.106674, 0.137451, 0.104986, 0.283955, 0.154519, -0.130151, 0.034826, -0.1296, 0.155641, -0.047, 0.120871, -0.084068, 0.110361, 0.108267, 0.181806, -0.006571, -0.196115, -0.209498, 0.215856, 0.046617, 0.152497, 0.135755, -0.01431, 0.044587, 0.057424, 0.131747, -0.189026, -0.055738, -0.252304, 0.047894], [-0.03009, -0.068633, 0.105705, -0.040838, 0.010957, 0.094668, 0.086082, 0.09279, 0.14149, -0.128687, 0.082887, -0.13096, -0.155307, -0.131217, -0.078133, 0.21065, 0.303856, 0.048567, -0.06051, -0.052385, 0.00911, 0.059521, 0.09461, 0.084881, -0.127591, 0.121507, 0.120801, -0.011579, 0.148558, 0.113077, -0.357801, -0.093006], [0.159304, -0.064714, 0.005962, -0.105113, 0.076952, 0.005539, -0.155574, -0.16177, -0.036848, -0.150135, -0.313784, 0.133447, -0.063343, 0.079556, 0.063707, -0.123294, -0.098336, -0.247973, 0.176089, -0.090514, -0.02178, -0.038565, -0.207044, -0.032739, -0.04345, -0.350186, -0.003802, 0.13036, -0.142474, 0.082586, -0.027447, -0.009904], [-0.033485, -0.03624, -0.109042, -0.008206, 0.062996, -0.002009, 0.133247, 0.101762, 0.108366, 0.064781, 0.076, -0.066425, -0.103793, -0.159453, 0.030575, 0.058607, 0.090935, -0.197056, -0.025079, 0.10691, 0.012916, 0.054394, -0.104418, -0.032264, 0.051194, 0.04887, 0.128671, 0.102385, 0.040045, 0.000244, 0.157915, 0.082968], [-0.040293, -0.119302, -0.224389, -0.12144, -0.040977, -0.23111, 0.025372, -0.165934, -0.233947, -0.125589, -0.480308, 0.110264, -0.02256, 0.068691, 0.131048, 0.066451, -0.439738, -0.094944, -0.192103, 0.158146, 0.006476, -0.127792, -0.2797, 0.018107, -0.044304, -0.1918, -0.043135, -0.130547, -0.056397, -0.195327, 0.062931, -0.015923], [0.081414, 0.025653, 0.078379, 0.086394, -0.063693, -0.160873, 0.247567, 0.185381, -0.30663, 0.146779, 0.062837, 0.12943, -0.111642, -0.004656, 0.045356, -0.080321, -0.032041, 0.209496, -0.063221, 0.072568, -0.048154, -0.162242, -0.04566, -0.137383, 0.077439, 0.110034, 0.1201, 0.124794, 0.140678, 0.037803, -0.100915, 0.016545], [0.165154, 0.208971, 0.056437, -0.165031, 0.098067, -0.158253, -0.158714, 0.139745, -0.095133, 0.036201, 0.217381, 0.02786, 0.081862, -0.046626, -0.132277, -0.034753, -0.012147, 0.035649, 0.439802, 0.024344, 0.116085, -0.134395, 0.099677, -0.011057, -0.001623, 0.186043, 0.158919, -0.171599, 0.065466, -0.098469, -0.085701, 0.085062], [0.065577, -0.036616, 0.074327, -0.008655, -0.17601, 0.00815, -0.001025, -0.030563, -0.236407, -0.240885, -0.270273, 0.00283, 0.168919, -0.043608, -0.101255, 0.054559, -0.093921, -0.438302, 0.035803, -0.032604, 0.028531, 0.03974, 0.108905, -0.015128, 0.002704, -0.227406, 0.022589, 0.095718, -0.095694, -0.064451, -0.183963, 0.00123], [-0.041314, -0.04792, -0.064429, -0.033493, 0.02377, 0.206535, -0.290622, 0.095409, -0.087181, -0.100747, 0.053186, -0.09438, 0.122902, 0.140993, 0.039329, 0.064984, -0.086778, 0.099824, 0.092556, 0.09913, -0.050731, 0.065049, -0.087236, 0.063243, 0.050417, 0.252567, 0.152303, -0.046917, 0.134848, -0.046979, -0.052023, -0.064693], [-0.025329, 0.073471, 0.012158, 0.102747, -0.029745, -0.195779, -0.066325, 0.128517, -0.191495, -0.078217, -0.080878, 0.134326, 0.035146, -0.187441, 0.176598, 0.036576, -0.339358, 0.104674, -0.150221, 0.12183, -0.01318, 0.007291, -0.079638, -0.036217, -0.001274, -0.002668, -0.040537, 0.043848, 0.110419, -0.245386, 0.039514, 0.053997], [-0.041695, -0.123953, 0.014589, 0.01456, -0.185004, 0.208313, -0.09819, -0.184659, 0.179637, -0.150973, 0.093329, -0.078757, -0.122708, -0.23012, 0.090935, -0.019171, -0.321674, 0.354048, -0.187622, -0.011765, 0.072037, 0.099666, 0.246515, 0.172925, -0.080223, -0.156978, -0.090252, -0.104918, 0.169937, -0.032702, -0.298501, -0.095981], [-0.043585, -0.175689, -0.288502, 0.035291, -0.088631, -0.15295, -0.237684, -0.019879, 0.154973, -0.19385, -0.13025, -0.157358, 0.115034, -0.234506, -0.181734, -0.086333, -0.182642, 0.185581, -0.540671, -0.137374, 0.071614, 0.142537, -0.100302, -0.125035, 0.074115, -0.047522, -0.106293, 0.08068, -0.000971, -0.102774, -0.147402, 0.006909], [0.089048, -0.086261, -0.125411, 0.103938, 0.026442, 0.193994, -0.12826, -0.243429, 0.052826, 0.056249, 0.101394, -0.156514, 0.032656, 0.014988, -0.124043, -0.089874, -0.177351, -0.456647, 0.034986, -0.217362, -0.102783, -0.051409, -0.25192, -0.013333, -0.11019, -0.103012, -0.195683, -0.111518, -0.274328, -0.101623, 0.051399, 0.026105], [0.041596, -0.042449, -0.009637, 0.120142, -0.03913, 0.051529, -0.113333, -0.012825, -0.058383, -0.061826, -0.04291, 0.009032, -0.033032, -0.150416, 0.070197, -0.066935, 0.11017, -0.320252, -0.08098, -0.024319, -0.147809, -0.034464, -0.099884, 0.160155, 0.14665, -0.268241, -0.127841, 0.083143, 0.008863, 0.081722, 0.023348, 0.093582], [-0.099338, -0.114394, -0.445109, 0.030494, -0.190568, -0.359471, 0.046001, 0.028501, 0.041795, -0.073555, 0.131198, -0.10636, 0.047066, 0.075544, -0.176159, 0.081813, -0.318835, -0.006192, 0.02485, 0.11387, -0.042887, -0.055978, -0.241949, -0.530779, -0.019748, 0.144524, 0.07595, -0.031371, 0.137946, -0.048412, 0.01923, 0.05933], [0.042096, 0.143972, 0.050372, 0.130422, 0.042827, -0.037137, 0.007717, -0.119763, 0.038777, -0.026664, -0.251139, 0.035217, 0.104601, -0.011314, -0.03464, 0.110261, 0.094686, 0.047838, 0.088963, 0.060328, -0.242079, 0.065696, 0.011058, -0.04496, 0.059683, -0.114302, -0.11826, 0.033955, 0.065168, -0.030369, -0.316526, -0.006134], [0.139906, 0.064515, -0.043133, -0.116412, 0.283664, -0.052466, 0.023848, 0.091421, 0.066791, 0.243022, 0.08714, -0.12506, 0.028751, -0.198204, -0.117562, 0.023756, -0.286761, 0.104579, 0.101399, 0.084248, 0.072021, -0.09046, -0.364673, -0.452899, -0.166288, 0.05313, 0.040455, -0.239644, 0.086525, 0.100679, 0.076316, -0.145581], [0.087597, 0.174507, -0.064926, -0.028021, 0.100285, -0.025796, 0.040696, -0.069273, -0.111095, 0.093629, 0.07983, 0.190691, -0.003264, 0.053026, 0.076465, -0.06791, 0.403715, -0.348808, -0.18503, -0.009512, -0.016557, -0.036216, 0.067245, -0.072703, 0.080728, 0.043948, -0.071155, -0.127371, -0.096074, 0.206589, 0.10791, 0.054506], [0.081596, -0.13158, 0.003313, -0.103268, 0.04524, -0.221632, -0.33447, -0.070928, -0.162815, 0.074364, 0.121062, -0.241609, 0.108766, 0.001745, -0.087121, -0.120811, -0.241288, -0.052053, 0.068387, -0.075845, 0.112829, 0.115666, -0.012341, -0.027267, -0.046672, -0.103897, 0.014516, -0.074326, -0.055206, -0.10834, -0.167182, -0.154537], [0.282475, -0.099346, -0.026406, -0.061623, -0.043256, 0.103918, 0.00822, 0.03981, -0.215256, 0.182125, 0.133457, -0.135445, 0.054897, 0.169884, -0.00701, 0.042113, -0.137852, 0.083953, 0.100493, 0.266517, 0.088742, -0.170771, -0.08054, 0.062351, -0.133805, 0.135917, 0.285501, 0.027978, 0.078582, 0.106804, -0.151643, -0.038091], [0.114584, -0.144221, 0.03336, 0.087815, -0.074461, -0.129261, 0.237391, 0.185882, 0.116038, -0.171982, -0.03889, 0.031751, 0.060465, -0.036282, -0.106907, -0.09036, 0.22901, 0.189358, 0.057856, 0.220396, 0.012273, 0.042432, 0.15197, -0.081367, -0.066634, -0.009801, 0.034494, -0.036479, 0.034925, 0.280581, -0.326377, -0.123471], [0.053128, -0.154884, 0.08849, 0.113663, -0.178484, -0.148189, -0.044606, -0.256296, -0.102513, -0.284903, -0.383366, 0.022365, 0.079056, 0.153384, 0.038601, 0.043248, 0.003784, -0.097326, 0.060562, 0.00816, -0.238098, 0.041481, 0.064491, 0.019154, 0.018345, 0.049275, 0.072924, 0.111419, -0.09181, -0.112994, 0.004505, -0.050054], [-0.007833, 0.248683, -0.121954, -0.15162, -0.114729, -0.088004, 0.206485, -0.134237, 0.026145, -0.039432, -0.054603, 0.076295, 0.072999, -0.125478, -0.425772, -0.192101, -0.037278, 0.02212, -0.199027, 0.081956, -0.099393, -0.262157, -0.049061, -0.125494, 0.178139, 0.027317, 0.076477, -0.124299, 0.130968, 0.044159, -0.095766, 0.113407], [-0.05695, 0.23469, 0.177258, 0.048352, 0.022182, 0.125404, 0.163246, -0.113045, -0.019704, 0.094171, -0.23931, 0.040546, -0.049924, 0.223219, 0.121484, 0.100858, 0.144807, 0.087276, -0.104733, -0.079028, 0.032139, 0.045506, 0.068649, 0.040646, 0.160038, -0.086549, -0.056154, 0.029958, -0.206407, 0.248305, -0.011141, -0.055195], [-0.061516, -0.023798, -0.210248, -0.261964, -0.000843, 0.197558, -0.100126, -0.110291, -0.199053, -0.210441, -0.091811, 0.181218, 0.17218, 0.063499, -0.448792, -0.099289, 0.070355, 0.032391, 0.037283, -0.138469, 0.010337, -0.029324, -0.195261, -0.15131, 0.146984, -0.073585, -0.168698, -0.193071, -0.153434, 0.290932, -0.089989, 0.09914], [0.050737, -0.236657, -0.275418, -0.054012, 0.063691, -0.18484, -0.085878, 0.047247, 0.006071, 0.105043, 0.209563, -0.231636, -0.204403, 0.313141, -0.296737, -0.266942, -0.418251, -0.22635, -0.23127, 0.059592, -0.039703, -0.065732, 0.041473, -0.305654, -0.215729, 0.003531, 0.105487, -0.14315, 0.020009, -0.274323, -0.235278, -0.033527], [-0.076981, -0.060797, 0.061684, -0.034642, 0.087385, -0.083348, 0.06991, 0.083499, 0.20312, 0.180405, 0.187598, 0.059454, 0.166635, 0.206423, -0.005394, -0.078694, -0.137468, 0.281488, -0.052911, 0.07961, 0.08477, 0.025004, -0.218773, -0.042912, -0.077268, 0.197554, 0.020415, -0.009613, 0.027668, -0.047286, -0.010034, 0.008705], [-0.067687, -0.046284, -0.033103, -0.193787, -0.05086, 0.001155, -0.129768, 0.046948, -0.018444, -0.156688, 0.100105, -0.164394, -0.079939, 0.014335, -0.069362, -0.141348, -0.107884, 0.04971, 0.002358, 0.134601, 0.112932, -0.023205, 0.125712, -0.182646, -0.205965, 0.135784, 0.13114, -0.087182, 0.045644, 0.045108, -0.177247, -0.201584], [-0.075249, -0.788847, -0.042426, -0.071691, -0.075404, -0.011261, 0.023471, 0.004034, 0.019441, 0.043156, -0.074454, 0.008656, -0.045333, 0.101526, 0.154346, -0.001029, -0.362755, 0.107837, -0.13461, 0.136196, 0.089242, -0.124279, -0.2996, -0.119306, -0.112503, 0.212263, 0.017975, -0.03338, 0.091232, -0.144674, 0.237152, -0.098537], [-0.113126, 0.14535, -0.002869, -0.146361, -0.118477, -0.012612, 0.340105, -0.04679, -0.086213, -0.268176, -0.169263, 0.111347, 0.145707, 0.007931, -0.023659, -0.031655, 0.092973, -0.482197, 0.246555, -0.0144, 0.040672, -0.036439, 0.031185, 0.151082, -0.027756, -0.118326, -0.135085, 0.146554, -0.234955, 0.126945, 0.10588, 0.09591], [-0.123246, -0.14576, -0.063709, -0.14873, -0.021401, 0.115928, -0.101164, 0.105757, 0.093415, 0.051825, 0.104008, -0.183055, -0.116657, 0.139937, -0.159369, -0.100668, -0.062794, -0.052039, 0.057494, 0.09677, 0.065043, 0.063662, 0.034622, -0.069559, -0.172292, -0.013814, 0.039931, -0.084882, 0.173455, -0.02467, -0.226675, -0.121934], [0.113683, -0.053376, 0.092464, 0.204739, 0.071076, 0.160046, 0.313721, 0.070395, 0.256728, 0.087449, 0.071571, 0.050503, 0.157136, 0.034881, -0.090462, -0.110199, 0.26999, 0.207386, 0.051123, 0.013718, -0.027532, 0.052283, -0.002876, 0.064908, 0.032379, 0.301644, 0.118522, -0.027489, 0.111396, -0.018448, -0.102444, 0.141909], [-0.116947, -0.188536, -0.026779, 0.009463, 0.049908, 0.006393, 0.113516, -0.195157, 0.11144, -0.187748, -0.395812, 0.044887, 0.135375, -0.033618, -0.096118, 0.119941, -0.370651, -0.467919, 0.025609, 0.068999, -0.140697, -0.123433, -0.301989, 0.024451, 0.076536, -0.209809, -0.154262, 0.066968, 0.043414, -0.05707, -0.29542, 0.075688], [0.188258, 0.3218, 0.024452, 0.133517, 0.052313, 0.261805, -0.014056, 0.166128, -0.094974, 0.125313, 0.0707, 0.121503, 0.104936, -0.200362, 0.273382, -0.02625, 0.112599, -0.270332, -0.274181, 0.114354, 0.051828, 0.035498, 0.038298, 0.213505, 0.122794, 0.123606, 0.108896, -0.043977, 0.122338, 0.038269, 0.083597, 0.038076], [-0.108638, -0.003626, 0.002119, -0.041037, 0.137719, -0.103396, 0.052359, 0.100524, -0.025417, -0.117306, 0.058504, 0.032876, 0.077642, -0.087324, -0.30546, -0.010359, 0.004792, 0.069102, -0.043778, 0.05401, -0.032522, 0.007423, 0.007849, -0.516566, 0.015811, 0.146133, 0.05013, -0.192448, 0.015748, -0.174773, -0.099869, -0.025419], [0.103302, -0.072477, 0.023745, 0.036229, 0.100358, 0.108631, 0.174414, 0.195342, 0.139755, 0.32807, 0.472581, 0.037965, 0.072757, -0.243529, 0.141117, 0.001722, 0.23126, 0.086797, -0.010402, 0.021943, 0.321277, 0.125158, -0.186007, 0.033388, 0.009962, 0.186606, 0.178443, 0.126417, 0.208532, 0.06396, 0.089693, 0.005086]], [-0.056638, -0.128249, -0.081296, 0.084888, -0.050267, -0.069254, -0.125965, 0.079394, -0.115933, -0.162105, 0.043826, -0.059537, 0.033542, -0.050334, -0.039666, 0.127947, 0.040823, 0.02077, -0.083903, -0.016448, 0.158163, 0.183138, 0.066046, -0.013769, 0.159302, -0.110417, 0.082481, -0.061045, 0.080727, -0.149664, -0.028653, 0.136765]),
    ([[0.061785], [-0.349557], [-0.121882], [-0.10573], [0.260749], [-0.280533], [-0.279766], [0.106153], [-0.24541], [0.354774], [0.163358], [-0.190823], [-0.076402], [0.163411], [-0.201329], [-0.105655], [-0.295778], [0.371307], [0.311341], [0.097462], [0.183654], [-0.11696], [-0.200001], [-0.232525], [-0.134566], [0.257113], [0.161893], [-0.154431], [0.161692], [-0.286809], [-0.281096], [-0.066093]], [0.031314]),
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
