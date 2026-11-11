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
    ([[-0.098564, 0.009319, -0.021231, -0.040489, 0.038515, 0.039861, 0.012133, -0.026507, 0.002144, 0.094243, -0.122411, 0.189279, -0.625649, -0.101071, -0.10765, 0.187664, 0.538934, 0.185205, -0.040637, 0.1027, -0.008224, -0.155116, 0.27173, 0.01962, -0.038199, -0.03145, 0.201982, -0.091824, -0.720838, 0.019917, 0.059685, 0.073377], [-0.288469, 0.194452, 0.031861, 0.109398, -0.139743, 0.265004, -0.03547, -0.20526, -0.200084, -0.01585, 0.155106, 0.091184, 0.048544, 0.157212, -0.031396, 0.212514, 0.207788, 0.30253, -0.196503, -0.09599, -0.020597, 0.109553, -0.040459, -0.26758, -0.007976, -0.04443, 0.122489, -0.084016, -0.214904, -0.241351, 0.218797, 0.260851], [0.108919, 0.249439, 0.074059, 0.185374, -0.136556, -0.090916, -0.280212, -0.045221, -0.22526, -0.027823, -0.080614, -0.184511, 0.13253, -0.022937, 0.158293, -0.069954, -0.129596, 0.203281, -0.045471, 0.051589, -0.261223, -0.035414, 0.096639, -0.208692, -0.187293, 0.258295, -0.203577, 0.177311, 0.027458, -0.05892, 0.127637, 0.130119], [0.162642, -0.169004, -0.034389, -0.20092, 0.105683, 0.103655, 0.173382, -0.169284, -0.06269, -0.048075, 0.195501, -0.179454, 0.084991, 0.043011, 0.101551, -0.195726, -0.290536, 0.104242, -0.023505, -0.326166, 0.202959, 0.374978, -0.077825, -0.026401, 0.119804, -0.217823, 0.025682, 0.011964, 0.103004, 0.081501, 0.05075, 0.027589], [0.42061, -0.248822, 0.238138, 0.428553, 0.162153, -5.3e-05, 0.445588, 0.275332, -0.274932, -0.443198, 0.065321, -0.450344, 0.10942, -0.266662, 0.579378, -0.162101, -0.159185, 0.230646, -0.159218, -0.347602, 0.374301, -0.077828, -0.318927, 0.256187, 0.369035, 0.145031, 0.129532, 0.326427, 0.075455, -0.170477, -0.349091, 0.000871], [-0.133904, -0.115112, -0.234502, 0.010902, 0.051122, 0.203169, -0.181589, 0.237707, -0.288575, 0.007255, 0.211604, -0.16294, 0.319198, 0.074976, 0.014223, 0.166737, -0.154361, 0.327136, 0.339113, 0.239929, 0.102719, -0.045126, 0.077334, -0.050196, 0.063077, 0.472397, -0.052241, -0.146075, -0.06287, -0.099956, -0.005585, 0.097311], [-0.316916, -0.159219, 0.503903, -0.044258, -0.136769, -0.123472, 0.274933, -0.138906, -0.091147, 0.508454, 0.2019, 0.019562, 0.033009, -0.400483, -0.334142, -0.291557, 0.653447, -0.406483, -0.449719, -0.057421, -0.031854, -0.294876, -0.312777, -0.062743, -0.146006, -0.063428, -0.862464, 0.514585, -0.019477, 0.077227, 0.266322, 0.011792], [0.224132, -0.136576, 0.027888, 0.122588, -0.132795, -0.088782, -0.186024, 0.113356, -0.163841, -0.000284, -0.074206, -0.113914, -0.08285, 0.109842, -0.079075, -0.233475, -0.079367, -0.020574, 0.008303, 0.316501, 0.215139, -0.228033, -0.35758, -0.084906, 0.05471, -0.077875, -0.051246, -0.138179, -0.031181, 0.158477, -0.067332, -0.055902], [-0.211579, -0.404956, 0.11634, -0.000884, 0.067101, 0.547014, 0.16453, -0.099568, -0.007676, 0.332728, 0.341174, -0.384723, 0.401669, 0.00417, -0.079702, -0.163164, 0.203383, 0.012222, -0.250169, 0.155389, -0.38992, 0.20712, -0.271505, 0.241919, -0.043188, 0.081657, -0.226193, -0.015158, -0.198373, -0.249172, 0.545472, -0.289543], [0.298213, 0.234587, -0.071308, 0.306411, -0.189197, 0.128996, 0.162053, -0.288775, -0.093878, 0.171971, 0.046832, -0.198495, -0.003538, -0.190164, 0.183577, -0.368222, 0.038785, -0.437574, 0.02639, -0.150156, 0.184871, 0.020216, 0.027018, -0.734123, -0.068795, 0.154299, 0.290087, -0.168344, 0.068756, 0.268436, -0.278654, -0.385757], [0.118468, -0.024101, 0.078421, -0.053448, -0.01752, 0.163671, -0.060682, 0.120801, 0.098745, 0.179084, -0.072677, -0.266377, 0.224455, 0.080287, -0.213935, -0.07208, -0.013035, 0.032401, 0.043632, 0.028351, -0.090307, 0.133536, 0.00666, -0.018805, -0.020333, -0.146107, -0.246396, 0.009207, 0.091535, -0.062424, -0.201253, 0.072599], [0.00807, -0.414377, 0.345407, 0.254586, 0.075974, 0.162464, -0.080175, 0.124994, 0.003669, -0.343326, -0.004169, -0.170646, -0.015917, -0.058008, 0.067035, 0.605165, -0.051401, -0.417936, -0.159818, 0.116786, -0.389893, -0.010755, 0.433862, -0.385263, 0.303176, -0.050208, 0.008666, 0.053456, 0.065957, -0.258745, 0.50452, 0.2228], [-0.18101, -0.009302, -0.102352, -0.171726, -0.263205, 0.247641, 0.086449, 0.269693, -0.041071, 0.180158, 0.094564, 0.054814, -0.005111, 0.010433, -0.118456, 0.104191, 0.110181, -0.227903, -0.02592, 0.152842, 0.060678, -0.45883, 0.258378, 0.175695, 0.064909, -0.394137, -0.211301, 0.058305, -0.176234, -0.120066, -0.455814, -0.022783], [-0.093241, 0.078373, -0.113098, 0.113538, -0.267369, -0.143152, 0.061963, 0.112915, -0.438146, 0.072711, 0.241925, -0.150285, -0.234886, -0.01114, -0.07323, 0.148007, -0.106914, -0.124189, 0.150115, 0.280165, 0.026642, -0.21375, -0.183333, 0.074446, 0.148517, -0.221979, -0.037949, 0.101434, -0.078972, -0.037467, -0.0859, -0.144183], [0.289758, 0.477493, -0.211616, -0.188349, -0.090927, -0.11588, 0.065769, 0.356485, -0.016652, 0.139103, -0.533367, -0.060603, -0.167807, -0.122026, -0.406875, 0.105514, 0.014423, -0.192831, -0.029897, -0.323607, -0.099143, 0.080068, 0.119065, 0.008403, -0.260024, 0.013259, -0.339984, 0.220191, 0.238389, -0.010486, -0.167603, 0.524199], [-0.04514, -0.445046, -0.08703, 0.237478, -0.269098, 0.094667, 0.151984, 0.150614, -0.060832, -0.04488, 0.067357, 0.223122, 0.029364, -0.10238, 0.264042, 0.141577, -0.013007, -0.162591, 0.322021, 0.074054, -0.041949, -0.066059, 0.287902, 0.101834, -0.016435, 0.363924, -0.120669, -0.191533, -0.048301, -0.121699, 0.019274, -0.026904], [0.570883, 0.044846, -0.457967, 0.101473, -0.309815, -0.078258, 0.008352, 0.24824, 0.510698, -0.588933, -0.688561, -0.407495, 0.037608, 0.082333, -0.169585, 0.204352, 0.10687, 0.061789, -0.013706, 0.062381, 0.086813, -0.412236, 0.020811, 0.123034, 0.18654, -0.000477, 0.29705, -0.179044, -0.282709, -0.495672, 0.261493, -0.531239], [-0.11875, 0.202563, 0.130218, -0.042959, -0.289789, 0.088709, -0.089007, 0.303643, 0.176216, 0.230102, 0.261414, -0.309635, 0.191735, 0.519961, -0.184596, 0.011822, 0.426026, 0.097451, 0.016259, 0.187138, 0.208703, 0.175354, -0.013431, 0.151254, -0.239041, -0.14891, 0.198364, -0.141899, -0.367323, 0.129767, 0.168523, -0.293043], [-0.116709, -0.114302, -0.022018, 0.123116, 0.054865, -0.212603, 0.025273, 0.237225, -0.063449, -0.027271, 0.168567, -0.052832, -0.008021, 0.08681, 0.019666, 0.111608, 0.023, 0.073845, -0.01828, -0.177039, -0.049874, 0.016784, -0.015304, 0.146922, -0.010083, 0.255529, 0.044377, -0.049485, -0.074434, 0.056988, 0.051997, -0.07089], [0.153666, -0.437735, 0.045197, -0.03848, 0.226139, -0.105067, -0.01226, 0.150798, -0.618294, -0.160697, 0.241541, -0.01591, -0.081047, 0.340029, -0.020991, -0.013211, -0.046766, 0.084507, 0.156562, 0.16828, -0.183737, -0.027212, -0.225419, -0.161885, 0.131717, 0.348574, 0.289809, -0.587437, 0.032856, -0.05647, -0.037351, 0.042503], [-0.072973, -0.080745, -0.187149, -0.114858, 0.233938, -0.027548, 0.290481, -0.342578, 0.120353, 0.006461, -0.168798, 0.030522, 0.201485, -0.372694, -0.130628, 0.039916, 0.070544, -0.324699, 0.311152, 0.330994, -0.364183, -0.154067, 0.071332, -0.040607, -0.123882, 0.014126, -0.34643, -0.235182, 0.163369, 0.119514, -0.05743, 0.014438], [0.020461, -0.001656, 0.628865, 0.00809, -0.188728, -0.001172, 0.324905, -0.244023, -0.082243, -0.12753, -0.041947, -0.001629, 0.006637, 0.261679, -0.0609, -0.077337, 0.011472, -0.089394, 0.631748, -0.084561, 0.126576, -0.058366, 0.090849, -0.018139, -0.342962, -0.061556, -0.030873, -0.354951, -0.029608, -0.02475, 0.039339, 0.062582]], [-0.388975, -0.460565, -0.393327, -0.080879, -0.14807, -0.238637, -0.300378, -0.017176, -0.206219, -0.298953, -0.032232, -0.346788, -0.062397, -0.510971, -0.201706, -0.378666, -0.048692, -0.160804, -0.427052, -0.730043, -0.034382, -0.16704, 0.421058, 0.107858, 0.177039, 0.129307, 0.116964, -0.269789, -0.215525, 0.071493, -0.102793, 0.063214]),
    ([[-0.010028, -0.297385, -0.454382, 0.064427, -0.824899, -0.075851, -0.133835, 0.057182, 0.232762, 0.305929, 0.061071, 0.113863, 0.257375, 0.146496, 0.227693, 0.012616], [-0.125445, 0.260563, -0.536393, -0.062956, -0.106196, -0.110577, 0.077873, -0.778734, 0.130713, 0.337596, 0.153318, -0.602801, 0.00539, -0.367853, 0.034042, 0.001849], [-0.25126, -0.082049, 0.551472, -0.174132, -0.359233, -0.43695, 0.019742, 0.066342, -0.25196, 0.152826, 0.107681, 0.071066, 0.103442, -0.084412, -0.362206, 0.120429], [-0.062463, 0.322632, 0.191334, 0.127283, 0.101471, 0.187136, -0.029767, 0.165881, -0.240608, 0.103652, -0.095805, -0.002268, 0.279155, -0.000737, -0.200907, -0.014049], [-0.155813, -0.040128, -0.408056, -0.131241, -0.633303, -0.188265, -0.170549, -0.108552, 0.198666, 0.141185, 0.253435, 0.042184, -0.175432, 0.068818, 0.013757, 0.021563], [-0.054033, 0.180048, 0.299871, 0.048656, -0.186469, 0.063342, -0.105311, 0.136435, 0.256228, -0.035518, -0.100205, 0.045435, -0.011315, 0.149815, 0.268139, -0.109965], [-0.035882, -0.229887, -0.049682, 0.029391, 0.206709, -0.277161, 0.012816, 0.257579, -0.005157, 0.001199, -0.046876, 0.202901, 0.113542, -0.00999, -0.550064, 0.138825], [0.036224, -0.224243, -0.256893, 0.081172, -0.349028, -0.073134, -0.178427, 0.077703, -0.225218, -0.63553, 0.060405, 0.262116, -0.197281, 0.058129, -0.447173, -0.061232], [0.143302, 0.295229, -0.666911, 0.11036, -0.196697, 0.21482, -0.056785, -0.248551, 0.051878, -0.643046, -0.194078, -0.151608, 0.113986, -0.359661, -0.185869, 0.103679], [0.181701, -0.059377, -0.015556, -0.159055, -0.062657, 0.334823, -0.142537, -0.333461, 0.014753, 0.177909, -0.090836, -0.285865, -0.000858, 0.05251, 0.166152, 0.018802], [-0.434952, -0.234395, 0.007821, 0.163191, -0.184641, -0.259388, 0.010603, -0.448968, -0.294747, 0.103909, -0.207415, -0.093485, -0.154216, -0.165486, -0.189516, 0.128869], [0.096281, -0.697906, -0.041947, -0.209417, 0.185982, 0.093829, -0.005411, 0.289347, 0.010746, -0.270413, -0.028494, 0.407573, 0.172317, 0.102798, 0.20624, 0.09675], [0.03225, 0.141783, -0.030271, 0.12479, 0.246561, 0.294011, -0.105536, 0.107214, 0.113533, 0.080635, 0.019679, 0.129433, 0.140033, 0.150076, 0.421029, -0.017658], [0.433068, -0.113001, -0.942028, 0.061, -0.442484, 0.175745, -0.068325, 0.31915, -0.090521, -0.067035, 0.020848, -0.368232, -0.361138, 0.5177, 0.211702, -0.062703], [-0.045806, 0.154453, 0.117994, -0.083344, 0.250721, 0.126683, 0.13268, 0.038486, -0.029999, -0.162512, -0.195512, 0.018592, 0.249513, 0.136894, 0.241386, 0.206703], [-0.195596, 0.234366, -0.25803, -0.313091, -0.295132, -0.317518, -0.189725, 0.023987, -0.101282, -0.223273, 0.040197, 0.032877, -0.022768, 0.114385, -0.558157, 0.120513], [0.184198, -0.335048, 0.444951, -0.052529, 0.075401, 0.175469, -0.048762, 0.379396, -0.01912, -0.058304, 0.111368, 0.35836, 0.066577, 0.075416, 0.291953, 0.120271], [0.06155, 0.02928, 0.057115, -0.056676, -0.042201, 0.419459, -0.085895, -0.01245, -0.066151, -0.146452, 0.12887, 0.213514, 0.162086, 0.112814, -0.164376, -0.036599], [-0.094944, 0.295446, -0.170351, 0.19384, -0.878323, 0.302656, -0.108634, -0.244282, 0.223108, -0.027343, -0.111134, -0.197303, -0.450253, 0.068261, 0.250832, -0.260589], [-0.444971, 0.221619, -0.585327, -0.003969, -0.111915, -0.679408, -0.175859, 0.135171, -0.349642, 0.009648, -0.076169, 0.12076, 0.285822, -0.4431, -0.688895, 0.226329], [-0.192878, -0.111962, 0.024255, -0.03938, -0.24969, -0.335938, -0.191048, -0.255271, 0.154149, -0.305847, -0.07963, -0.154096, -0.140482, -0.26598, -0.130497, -0.072615], [0.061649, -0.208511, 0.099914, -0.790026, -0.021063, -0.218819, -0.080816, -0.399966, 0.071622, 0.101728, -0.008987, -0.05045, 0.020193, -0.21365, 0.008893, 0.051227], [-0.117709, 0.063767, -0.333875, -0.055828, -0.063159, -0.089172, -0.188508, -0.348908, 0.167046, 0.075828, -0.343301, -0.344902, 0.155462, 0.057127, -0.141517, -0.005944], [0.102306, -0.193847, -0.674641, -0.113, 0.169349, -0.155627, -0.006896, -0.238146, -0.005554, -0.248735, -0.066649, -0.327385, -0.320897, 0.131842, 0.36909, -0.178268], [-0.009619, 0.300667, 0.055092, -0.23212, -0.216349, -0.267778, -0.146915, -0.079047, 0.058724, -0.176461, -0.030653, -0.086532, 0.096368, 0.065091, -0.152567, 0.194915], [-0.326315, -0.044375, -0.239929, -0.216464, -0.278662, -0.001723, -0.007714, 0.067133, -0.249827, 0.167048, -0.011872, -0.038985, 0.252452, 0.0369, -0.423046, 0.111803], [0.287353, 0.359282, -0.163235, -0.057984, -0.267333, -0.17803, -0.014686, 0.296532, 0.188934, 0.069974, 0.052687, 0.070471, 0.044106, 0.040912, 0.157832, 0.12611], [-0.209097, -0.096177, -0.699714, 0.209704, 0.300893, -0.032135, -0.083873, -0.249233, 0.161861, 0.277261, 0.223124, -0.647543, -0.199431, -0.323381, 0.15857, -0.202316], [0.247584, -0.386016, -0.662217, -0.169566, 0.188193, -0.005982, 0.057536, 0.203337, 0.15232, -0.024291, -0.097615, 0.262214, 0.352717, 0.111205, 0.247982, 0.052342], [0.272469, -0.161249, -0.045021, 0.046704, -0.024872, 0.090407, -0.144674, 0.123367, 0.179671, -0.012771, 0.004764, -0.195599, -0.030747, -0.184563, -0.011647, -0.074056], [0.39014, 0.097733, 0.377333, -0.017362, 0.460308, 0.249773, -0.019441, 0.195783, 0.137383, 0.137351, 0.104553, 0.013444, -0.060441, -0.004915, 0.155029, -0.001155], [0.036846, -0.119483, -0.182522, -0.176039, -0.354359, -0.12192, 0.069181, 0.348741, -0.246542, -0.070548, 0.27214, 0.112679, 0.254668, 0.234072, 0.211463, 0.074579]], [0.088994, -0.210273, -0.478838, 0.366571, -0.056376, 0.000757, -0.172797, -0.213672, -0.02845, -0.392672, 0.197388, -0.039466, -0.184427, 0.290521, -0.401591, 0.20472]),
    ([[-0.554575], [0.789367], [1.369679], [0.494192], [-1.293191], [-1.007916], [-0.027001], [0.81017], [-0.525622], [-1.205507], [0.46492], [0.636434], [0.510379], [0.361957], [-0.824028], [0.197096]], [-0.30095]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.562321557048493e-19, -1.1343482335572739e-18, -5.671741167786369e-19, -1.8905803892621232e-19, 1.1343482335572739e-18, 0.0, 0.0, 0.0, 1.2572534685440513, 0.0, 10.639801906763806, 28.799029822078616], [6.72849538126497, 19.8058956257327, 34.925161910299096, 44.72666958616459, 57.636846319296275, 1.1243487056677264, 0.9903909371630099, 0.9446913427593254, 0.8452682282888074, 0.7575381583058208, 1067.5606541222405, 2053.035725905977, 4046.2991308026644, 0.6179080587708422, 0.3127841039679992, 16.705331990766382, 34.22120473597514], [0.6490227662551226, 0.3600762349252582, -0.1832944003136108, -0.09733275171505404, -0.10066480452612983, 0.8482856190320313, -3.2486297440276823, 6.1619984101404714, -3.540934484781457, 3.1035095898512166, -0.6019915249010418, 0.06557241931064624, -0.1855095648918967, -0.002102755891520747, 0.4105229362630337, 0.0003153876484541001, 0.00020715056621090425], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 7.568942287397852e-19, -1.1353413431096778e-18, -5.676706715548389e-19, -1.892235571849463e-19, 1.1353413431096778e-18, 0.0, 0.0, 0.0, 1.2574354160321035, 0.0, 10.646654416476506, 28.81409587438229], [6.7313766144182345, 19.814344420832445, 34.939862329400235, 44.74491730719747, 57.65822173996363, 1.124692105470714, 0.9907164467252751, 0.9450073668099886, 0.8455601770714993, 0.7578101560190531, 1067.786637740178, 2053.787560401509, 4047.179122347928, 0.6181145992138308, 0.3129209935553393, 16.710984248831306, 34.23237508220454], [2.9421164343350763, 0.7964360421909548, -1.0744230210945729, -0.9234918162695617, -0.8475723430517554, 1.4775048291769728, -4.635873707445536, 9.213729195881914, -4.907517756528247, 6.23467901404692, -2.386919742393157, 0.4346797676233983, -1.5368709559241014, 0.001829056109893383, -0.22349228908983945, -0.0002742367847445596, -0.00018005766511644536], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.257441, 0.0, 0.0, 5.84623, 5.846549, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987547], [1.124422, 0.990545, 0.944803, 0.844787, 0.75662, 2.57683, 18.120787, 0.790527, 15.924298, 0.695578, 0.651228, 0.54053, 1.677429, 1.677372, 0.999993, 4.403117, 1.087628, 0.310614, 132.397858, 334.626434, 1066.212891, 1.340704], [([[0.083691, -0.066933, 0.034012, -0.047874, -0.046178, 0.046892, 0.027156, 0.000954, 0.078329, 0.108704, -0.297333, -0.008781, -0.329832, -0.04742, 0.059635, 0.300794, 0.183879, -0.043792, -0.085566, 0.267475, -0.011261, 0.007552, 0.162077, -0.539039, 0.053327, 0.108903, 0.023464, 0.050461, -0.020575, 0.015161, 0.336468, 0.17935], [-0.210903, 0.063661, -0.004734, 0.084923, -0.139773, 0.168432, 0.056584, -0.250219, -0.210086, -0.052355, 0.004859, 0.032575, 0.202079, 0.040801, 0.02637, 0.39457, 0.098227, 0.104888, -0.123765, -0.100699, -0.044148, 0.129998, -0.121632, -0.362358, 0.074791, 0.010874, 0.152514, -0.062105, 0.030574, -0.256878, 0.153755, 0.201847], [0.106952, 0.159496, -0.013614, 0.144527, -0.121112, -0.214262, -0.163985, -0.093548, -0.224174, -0.067653, -0.185019, -0.161825, 0.197197, -0.143037, 0.186736, -0.001481, -0.132054, 0.040074, 0.033978, 0.032838, -0.275218, -0.054251, 0.095956, -0.139488, -0.145329, 0.209788, -0.160503, 0.167034, 0.09298, -0.110886, -0.022983, 0.042561], [0.033754, -0.240923, -0.246534, -0.235036, 0.134811, 0.003664, 0.133704, -0.370732, 0.002359, 0.02005, 0.230663, 0.063615, 0.182769, 0.021033, 0.077352, -0.410114, -0.313954, 0.094659, 0.099497, -0.333352, 0.132654, 0.410735, 0.138817, 0.114901, -0.027404, -0.43907, -0.150824, -0.039201, -0.100942, -0.007533, 0.112915, -0.034303], [0.293316, -0.127101, 0.335164, 0.597729, -0.356213, 0.222743, 0.486233, 0.172893, -0.17606, -0.38403, 0.131628, -0.22319, -0.022912, -0.342074, 0.4728, -0.334277, 0.218813, 0.459239, -0.263853, -0.200357, 0.606133, -0.475162, -0.672015, 0.425319, 0.286736, 0.25372, 0.096576, 0.155236, -0.6783, -0.263353, -0.155349, 0.015407], [-0.092127, 0.13906, -0.139715, 0.038087, 0.035396, 0.260622, 7.2e-05, 0.178478, -0.306047, -0.071467, 0.010437, -0.285623, 0.332831, -0.112522, 0.107962, 0.274471, -0.06911, 0.291387, 0.324415, 0.265752, -0.046184, -0.240458, -0.012969, 0.339518, 0.122113, 0.366071, 0.042834, -0.305429, -0.214346, -0.235261, -0.074966, 0.070255], [-0.259084, -0.266029, 0.628066, 0.088765, 0.009834, -0.029396, 0.340544, -0.021392, -0.278934, -0.052799, 0.137477, -0.136176, 0.234128, -0.138709, -0.029158, -0.339777, 0.453632, -0.537697, -0.228697, -0.344226, 0.467014, -0.299438, -0.637032, -0.227482, 0.208558, -0.282681, -0.582327, 0.458951, -0.141499, -0.02516, 0.311164, -0.017292], [-0.031019, 0.011773, -0.106968, 0.050751, -0.064897, -0.026979, 0.035396, 0.003838, -0.207885, -0.036971, 0.141796, -0.289091, -0.087186, 0.096288, -0.080105, -0.123699, -0.029461, -0.07631, 0.056465, 0.099274, -0.054143, 0.008211, -0.087627, -0.010177, 0.148708, -0.072681, -0.007703, -0.058144, 0.119885, 0.221356, -0.112493, 0.06584], [-0.272716, -0.243101, -0.032352, -0.120753, 0.371057, 0.403888, -0.363896, -0.254475, 0.444024, 0.483859, 0.12815, -0.209838, 0.355724, -0.016157, 0.070791, -0.112232, -0.212067, -0.410227, 0.178209, 0.058056, -0.285419, -0.055478, 0.111244, -0.198289, -0.309451, -0.174774, -0.640506, -0.060421, 0.121207, -0.260599, -0.055868, -0.241869], [0.331814, 0.110917, 0.081137, 0.194828, -0.016559, -0.021724, 0.066323, 0.136318, 0.015008, 0.221842, -0.031782, -0.248536, 0.137845, -0.613734, -0.218127, -0.059797, 0.051811, -0.15731, -0.061346, -0.11756, 0.033031, 0.115143, 0.107598, -0.171127, -0.289769, 0.118853, 0.25774, -0.162742, 0.348155, 0.058813, -0.783446, -0.917405], [-0.19413, 0.038817, -0.031448, 0.069942, -0.053677, 0.021302, -0.090838, 0.044865, -0.043567, 0.320823, -0.09089, -0.24159, 0.220112, -0.116844, -0.149734, 0.044884, -0.006622, 0.335808, 0.202695, 0.24729, 0.02138, 0.049219, 0.016181, 0.00755, -0.126043, -0.029339, -0.111143, 0.05682, 0.192188, -0.02243, -0.09843, 0.11153], [0.430575, -0.59928, 0.35044, 0.151135, 0.429148, 0.176245, -0.317208, -0.113246, 0.068399, -0.320344, -0.13952, -0.115683, -0.043065, -0.150814, 0.168907, 0.128464, -0.261991, -0.156791, -0.394592, 0.106898, -0.163066, -0.130743, 0.206404, -0.126961, 0.387039, 0.039737, 0.10035, -0.034787, -0.247382, -0.281709, 0.249738, 0.068718], [-0.093691, 0.069875, 0.003809, -0.137954, -0.199014, 0.371799, 0.078662, 0.030284, -0.06985, 0.181542, 0.029093, 0.022784, 0.179603, 0.0883, -0.070134, -0.01311, 0.123435, -0.075218, -0.053928, 0.198574, 0.109201, -0.171639, 0.199373, 0.047747, 0.211193, -0.324415, -0.218674, 0.363287, -0.104487, -0.169855, -0.094857, -0.082735], [0.204197, 0.058722, -0.235286, 0.118344, -0.312946, 0.005157, 0.159569, -0.076305, -0.16585, 0.204394, 0.292841, 0.023002, -0.231827, 0.067474, 0.04435, 0.126412, -0.491197, -0.344189, 0.039848, 0.431646, -0.108506, -0.290249, -0.191405, -0.069628, 0.045578, -0.125285, 0.066035, 0.133357, -0.146397, -0.152431, -0.044618, -0.042809], [0.185594, 0.276352, 0.04588, 0.048216, -0.497134, -0.079718, 0.023186, 0.292003, -0.232592, -0.265861, -0.302859, -0.057325, -0.198825, -0.009996, -0.017108, 0.128485, -0.154962, 0.387945, -0.143937, -0.077279, 0.08671, -0.02967, 0.020237, -0.140468, -0.049817, 0.092731, -0.229067, 0.227389, 0.146033, 0.019125, 0.012, 0.218676], [-0.020711, -0.352823, -0.182761, -0.130328, -0.215273, 0.285966, -0.28759, 0.385763, -0.241606, 0.155238, 0.01912, 0.060738, 0.075917, -0.220927, 0.206309, 0.296134, 0.120797, -0.086442, 0.392036, 0.063102, -0.250879, -0.178673, 0.470963, 0.108195, 0.177845, 0.072553, -0.197236, -0.368962, -0.126934, -0.147734, -0.022775, -0.036339], [0.424718, 0.320003, -0.195852, 0.087856, -0.373862, 0.228359, 0.011056, -0.288259, 0.464885, -0.367686, -0.677121, -0.031732, -0.149169, 0.03594, -0.371741, 0.180374, 0.140854, -0.312879, -0.304889, 0.251554, -0.047202, 0.053815, -0.031256, -0.075805, 0.086034, -0.21902, 0.044142, -0.1347, -0.056476, -0.2859, 0.482506, -0.481022], [-0.057708, 0.148164, -0.020504, -0.089372, 0.036533, 0.14983, -0.036562, -0.103739, -0.027219, 0.212619, 0.225237, -0.386445, 0.223815, 0.138626, -0.140046, 0.018642, 0.08907, 0.039932, -0.012198, 0.181958, 0.117916, 0.190656, 0.047905, 0.241472, 0.100199, -0.090683, 0.123372, -0.109247, -0.178522, -0.071376, 0.139461, -0.445636], [-0.107435, 0.039008, -0.073023, 0.054734, -0.252151, 0.077403, -0.097542, 0.139587, -0.376429, -0.077389, 0.156213, -0.208895, -0.087626, 0.263002, 0.044105, 0.150449, 0.088609, 0.020768, -0.005556, -0.369426, 0.020072, -0.035288, 0.043779, -0.011373, 0.121747, 0.021443, 0.038354, -0.060054, -0.196813, 0.012675, 0.195901, -0.02586], [0.27463, -0.54098, 0.011934, 0.010434, 0.224374, 0.082916, -0.026362, 0.123071, -0.429838, -0.223599, 0.174419, -0.206476, -0.435431, 0.033984, 0.03759, -0.215173, 0.017081, 0.007707, 0.059331, 0.165404, 0.154659, -0.338579, -0.065302, 0.030702, -0.031709, 0.273432, 0.259527, -0.253061, 0.016025, 0.118812, -0.154893, 0.096885], [0.00743, -0.086045, -0.043753, -0.055813, 0.100978, -0.00944, -0.029936, -0.170505, -0.005892, -0.294968, 0.003763, -0.091899, -0.024962, -0.22678, -0.149459, 0.391702, -0.428596, -0.155068, 0.25015, 0.221947, -0.840726, -0.006122, 0.030229, 0.090407, -0.028853, 0.168592, -0.464087, 0.088802, 0.126869, -0.038837, 0.2311, -0.021413], [-0.135267, 0.145032, 0.571108, 0.012061, 0.242854, -0.090965, 0.167138, -0.092969, 0.190153, 0.127308, -0.014146, -0.021721, 0.046025, 0.038387, -0.034405, -0.191667, 0.026561, 0.031527, 0.380076, -0.167082, -0.037611, -0.043762, 0.070931, -0.013548, -0.138786, -0.022301, -0.006544, -0.29668, 0.033655, -0.022371, -0.013556, -0.027353]], [-0.177149, -0.205575, -0.154343, -0.183665, 0.182512, -0.103483, 0.339502, 0.108716, -0.13697, -0.374526, -0.000961, -0.433042, -0.120073, -0.260393, -0.277283, -0.747913, 0.240561, -0.30613, -0.19183, -0.59872, -0.144981, -0.114909, 0.315848, -0.096546, 0.266313, 0.02648, 0.451419, 0.268511, -0.371441, 0.058188, 0.264185, -0.202853]), ([[-0.225662, -0.086178, -0.386103, 0.121249, 0.209776, -0.061454, -0.149259, 0.163284, -0.023092, 0.403859, 0.051743, 0.067287, 0.077575, 0.163286, -0.046814, -0.080028], [-0.117644, 0.479631, 0.094551, -0.735156, 0.11929, 0.073924, 0.059396, -0.284612, 0.154336, -0.102274, 0.127557, 0.19341, 0.027935, -0.052392, -0.052081, -0.101356], [-0.296222, 0.140045, 0.238377, 0.299351, -0.18146, -0.155469, 0.021567, 0.025097, -0.02772, -0.112071, 0.258798, -0.007198, -0.199314, -0.34835, -0.186744, -0.173011], [-0.16143, 0.277801, 0.214181, 0.153817, -0.167561, 0.19026, -0.039169, 0.191701, -0.439415, -0.020046, -0.030746, 0.020404, 0.113949, -0.043627, 0.281476, -0.113561], [-0.132824, -0.004026, -0.062756, -0.354513, -0.49158, -0.288871, -0.160749, -0.037319, 0.168426, 0.12269, 0.164946, -0.108486, -0.451242, -0.056202, -0.17483, 0.089831], [-0.55091, 0.08537, 0.007864, 0.185052, 0.058094, -0.056782, -0.111653, 0.103945, 0.110977, 0.20162, -0.005241, 0.240913, -0.195222, 0.126208, -0.124857, -0.181104], [-0.02544, -0.20971, -0.071989, -0.038171, 0.136508, -0.474861, -0.009714, 0.055581, 0.116678, -0.219118, 0.076938, 0.069903, -0.113421, -0.088772, -0.305348, 0.103763], [-0.038751, -0.038132, 0.169953, -0.065275, -0.003957, -0.12219, -0.17913, 0.046067, 0.276583, -0.44261, 0.083668, 0.090881, 0.056988, 0.171883, -0.0609, -0.050719], [0.205914, 0.362072, -0.035105, 0.123855, -0.184738, 0.176055, -0.056785, -0.162112, 0.065593, -0.049036, -0.132543, 0.153131, 0.219965, -0.196227, -0.096629, 0.145049], [-0.029232, 0.032406, 0.388322, 0.503274, 0.305592, 0.315967, -0.152682, -0.297789, -0.02168, 0.253723, 0.071728, -0.331121, -0.021838, -0.130305, 0.054852, 0.07284], [-0.26408, -0.177267, -0.004843, -0.056074, -0.218035, -0.142481, -0.010138, -0.46836, -0.209977, -0.017058, -0.08239, -0.054489, -0.166818, -0.284903, -0.133818, 0.098842], [0.145107, -0.106007, 0.067011, -0.422763, -0.00573, 0.209393, -0.00904, -0.269953, 0.105744, 0.074641, -0.122534, 0.269519, 0.17582, -0.016163, 0.107855, 0.017656], [-0.089142, 0.16258, 0.194152, 0.227797, -0.002145, 0.27353, -0.125965, 0.083455, 0.123458, 0.015999, 0.050422, -0.032254, 0.082852, 0.141333, 0.510726, -0.12596], [0.157472, -0.079806, -0.10703, -0.134294, -0.571525, 0.339165, -0.056779, 0.163917, 0.019304, -0.278812, -0.057691, -0.107181, -0.126138, 0.445317, 0.172881, -0.195062], [0.231695, 0.157709, 0.008151, 0.098414, 0.311987, -0.031931, 0.128814, 0.151317, -0.117057, -0.22813, -0.120389, 0.127517, 0.064978, 0.081342, -0.307212, 0.098528], [-0.243261, 0.057754, -0.195109, 0.020736, 0.079242, -0.38881, -0.20119, -0.073064, -0.127008, -0.419627, -0.148733, -0.27483, 0.32708, 0.068533, -0.6655, 0.157261], [0.200092, -0.168127, 0.227493, 0.087355, 0.010013, -0.016496, -0.045506, 0.415227, 0.009384, 0.030136, 0.118633, 0.29184, 0.080965, -0.061355, 0.148327, 0.070536], [0.032687, -0.221268, -0.458401, -0.047819, -0.337645, 0.295712, -0.087743, -0.011487, -0.407763, -0.20183, 0.023167, 0.243804, 0.220949, -0.016099, 0.006258, -0.011966], [-0.323449, 0.155056, -0.097491, -0.150413, 0.18099, -0.102588, -0.097636, -0.278317, 0.299979, 0.024221, 0.023145, -0.371108, -0.214634, 0.179198, 0.156379, -0.028628], [-0.358516, -0.121764, -0.134058, 0.129464, -0.451722, -0.055453, -0.17582, -0.047794, -0.215493, 0.012499, -0.171184, -0.328585, 0.289535, -0.039877, -0.73036, 0.325606], [-0.286231, 0.043532, -0.011669, -0.732438, -1.181898, -0.492352, -0.190705, -0.465494, -0.099193, -0.854471, -0.145871, -0.358131, -0.299699, -0.109938, -0.180554, -0.263314], [0.015256, 0.113699, 0.112364, -0.395916, -0.188672, -0.119814, -0.079315, -0.412323, 0.016615, 0.259989, -0.103343, -0.076579, -0.029183, -0.491657, 0.197596, -0.039012], [-0.150797, 0.040162, -0.169439, -0.451773, 0.010386, -0.320982, -0.189484, -0.307507, 0.130767, 0.010022, -0.122681, -0.403639, 0.065785, 0.073207, 0.040883, 0.090741], [0.574343, -0.44572, -0.884286, 0.55692, -0.074033, 0.024237, -0.029916, 0.341162, 0.116813, -0.04682, -0.100667, -0.190129, 0.049777, 0.427767, 0.410352, 0.11696], [-0.064154, 0.168342, 0.059048, 0.041826, -0.386257, -0.247935, -0.154385, -0.086089, 0.024448, -0.246271, 0.075953, -0.048304, 0.066082, 0.037288, -0.561189, 0.140694], [-0.636091, -0.096736, -0.124205, -0.064753, -0.135964, 0.16328, -0.019681, 0.070707, -0.026329, 0.099213, 0.055165, -0.07547, 0.142251, 0.081347, -0.176343, 0.175102], [0.116583, 0.006937, -0.477866, 0.509469, 0.200343, -0.038, -0.023057, 0.392745, 0.19004, 0.129481, 0.180755, 0.029508, 0.089949, 0.094173, 0.06361, 0.098874], [-0.200992, 0.103566, -0.081562, -0.240736, -0.371787, -0.064071, -0.105411, -0.118057, 0.250879, -0.143223, 0.063692, -0.103634, -0.448311, -0.084998, 0.287258, -0.066199], [0.191871, -0.096446, -0.328222, 0.032933, 0.296829, 0.117044, 0.057536, 0.281475, 0.100892, -0.090924, -0.325105, 0.174221, 0.386573, -0.365282, 0.291614, -0.022543], [0.236969, -0.209527, -0.289283, -0.340042, 0.123867, 0.031667, -0.14, 0.140707, 0.106998, 0.056932, 0.061197, -0.121321, -0.184024, -0.251933, -0.101293, -0.080476], [0.367279, 0.124189, 0.240077, 0.043668, 0.043423, 0.109635, -0.026578, -0.207483, 0.273006, 0.062385, 0.124153, -0.187955, -0.146312, 0.168496, 0.357434, 0.08777], [0.436344, -0.195791, -0.14567, 0.08219, 0.319197, 0.082228, 0.062603, 0.39779, 0.019985, -0.271541, 0.104371, 0.087536, 0.314676, 0.271742, -0.232498, 0.011911]], [0.028363, -0.282142, -0.254984, -0.220021, -0.244545, 0.01052, -0.177603, -0.073416, 0.131374, 0.090974, 0.336156, 0.094884, -0.186634, 0.343993, -0.2445, 0.250216]), ([[-1.11461], [0.90315], [1.249313], [1.835465], [-2.022733], [-1.620032], [-0.040441], [1.505434], [-1.032664], [-1.601373], [0.864221], [1.279547], [1.058798], [0.769904], [-1.038208], [0.450484]], [-0.290885])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.75191, -9.2e-05, 9.2e-05, 0.0, 0.052846, 0.052846, -0.0, 0.0, 0.0, 1.243668, 5.879185, 5.879185, 0.0]
SEQ_TSD = [1.079295, 0.132411, 0.132411, 0.099558, 1.45424, 1.45424, 0.666687, 0.219428, 0.17921, 0.60273, 1.666237, 1.666237, 0.649942]
SEQ_XM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.257423, 0.0, 0.0, 5.846561, 5.846561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.987482]
SEQ_XS = [1.124955, 0.99101, 0.94523, 0.845186, 0.757041, 2.577918, 18.130564, 0.790721, 15.945622, 0.69697, 0.6513, 0.540497, 1.677766, 1.677766, 0.999993, 4.402977, 1.087599, 0.310604, 132.411407, 334.802429, 1066.728027, 1.341266]
SEQ_YSC = 1.939010739326477
SEQ_WI = [[-0.574752, -0.150512, 0.094185, 0.358723, -0.38363, 0.21564, 0.633314, -0.212574, -0.512545, -0.23325, -0.700787, -1.343823, -0.123478, -1.25391, 0.567988, 0.282512, -0.308259, 0.760328, -0.061219, -0.331274, -0.270864, 0.443273, 0.311892, -0.195518, -0.05852, -0.185164, -0.242364, 0.117727, 0.334309, -0.240004, -0.05706, -0.381815, -0.103042, 0.415603, 0.006207, -0.12418, -0.015729, 0.086868, 0.15228, -0.443993, -0.146825, -0.190765, -0.618294, 0.118512, 0.288707, -0.067243, -0.1357, -0.442525], [0.02928, -0.528568, -0.04281, 0.079785, -0.289388, 0.042042, 0.061073, -0.670736, -0.576222, -0.066905, -0.203126, -0.095979, -0.044812, -0.239414, -0.45192, -0.050445, -0.490252, -0.247491, 0.138742, 0.209824, -0.755277, 0.540292, -0.145176, 0.422675, 0.05166, 0.142973, -0.101067, -0.190121, -0.204768, 0.015931, -0.123462, -0.150088, 0.462152, -0.023666, 0.077894, -0.149549, -0.413618, -0.575938, 0.469745, 0.037382, -0.282782, -0.029171, 0.114062, -0.003979, -0.079805, -0.297494, -0.196515, -0.261508], [0.008941, -0.368503, -0.591577, 0.114756, -0.625718, 0.679301, -0.082241, 0.143958, 0.090204, 0.333403, -0.331655, -0.030653, 0.238498, 0.306996, -0.282701, 0.138568, 0.357378, 0.538464, -0.407987, -0.437604, -0.135851, -0.293572, -0.19548, -0.732497, 0.342595, -0.319073, 0.35527, 0.398838, -0.585716, 0.019328, 0.46092, 0.331659, 0.142104, 0.344734, 0.042343, 0.174846, -0.022042, -0.017985, 0.204648, -0.302303, 0.46188, 0.126466, -0.278331, -0.380266, -0.164934, -0.097514, 0.47442, 0.193439], [0.147181, -0.159086, -0.120844, 0.013532, -0.57982, 0.483787, 0.020132, -0.242525, -0.131863, 0.150843, -0.152263, -0.014074, 0.355309, -0.052523, -0.311405, 0.00552, 0.194117, 0.144393, 0.183565, -0.174944, -0.472881, 0.161121, -0.160944, -0.263168, 0.527634, -0.295141, 0.346846, -0.322814, -0.12748, -0.235661, 0.356734, -0.000699, -0.104399, -0.038452, -0.044726, -0.031474, -0.004036, 0.11628, -0.0526, -0.048373, 0.257289, -0.277838, -0.123049, -0.036145, 0.101533, 0.132673, 0.128264, -0.129506], [-0.692554, 0.083544, -0.270808, -0.530539, -0.085106, -0.120689, 1.018553, 0.173452, -0.119111, 0.66758, 0.298523, -0.003751, -0.127064, -0.612331, 0.184999, 0.206978, -0.703395, 0.008812, 0.023334, -0.411756, -1.083394, -0.743641, -1.146258, 0.190585, 0.051668, 0.294754, -0.071564, 0.076173, 0.031179, 0.081563, 0.209914, 0.148447, 0.879069, 0.027509, 0.141978, -0.796739, -0.56391, -0.571815, 0.8989, -0.152885, 0.149671, -0.512769, 0.176544, -0.276197, 0.022361, 0.002433, -0.175183, -0.15427], [-0.456537, 0.186525, 0.230979, -0.507978, -0.068274, -0.109299, -0.164184, 0.753749, -0.696371, -0.19495, 0.319466, 0.760594, 0.058036, -0.264542, 0.039542, -0.385775, 0.076416, -0.188728, -0.119312, -0.115969, 0.25483, -0.337357, -0.245843, 0.139393, -1.258818, 0.56506, -0.584384, -0.231286, 0.261865, 0.148354, -0.219293, -0.241888, -0.047699, 0.154281, -0.086978, -0.28497, 0.069299, -0.193743, 0.132797, -0.025892, -1.198574, -0.119683, -0.779935, 0.901243, -0.060529, -0.268139, -0.127111, 0.022298], [0.293038, 0.309556, -0.446441, 0.54768, 0.215841, 0.097045, 0.208634, -0.083176, -0.065188, 0.24407, 0.426663, 0.458823, -0.141432, -0.080562, 0.058358, 0.107705, -0.275477, -0.13242, 0.557054, -0.258971, 0.028429, 0.266549, -0.085807, 0.253968, -0.265135, 0.081944, 0.137608, 0.150627, 0.361873, -0.354376, 0.24609, 0.406017, -0.057534, -0.479683, 0.004533, -0.184305, -0.043119, -0.092512, -0.035708, 0.103744, -0.132925, 0.237306, 0.075686, -0.044952, 0.069488, 0.093911, 0.040933, -0.146332], [0.533588, 0.008337, -0.023851, 0.550212, 0.388059, 0.072336, -0.123345, -0.084916, 0.249763, 0.201303, -0.30085, 0.630474, -0.155362, 0.143091, -0.278841, -0.198872, 0.114133, -0.005805, 0.241463, -0.001293, -0.305627, 0.034153, 0.29071, -0.138166, -0.384453, -0.235007, 0.069813, 0.408137, -0.048058, -0.025907, -0.164722, -0.090613, 0.303375, -0.160052, -0.262964, -0.033196, -0.169849, -0.00271, 0.151942, 0.121352, -0.098665, 0.059622, -0.231646, 0.124251, 0.449041, -0.312749, 0.050538, 0.129227], [0.261264, 0.112367, 0.242351, 0.317435, -0.088244, 0.546521, -0.093358, -0.033375, -0.078381, 0.563257, 0.041612, 0.631979, 0.063834, 0.14382, 0.091421, 0.320388, 0.106136, 0.088025, 0.185659, -0.168454, -0.257301, 0.184535, 0.131938, 0.294295, -0.199853, -0.2248, 0.516492, -0.052156, 0.042746, -0.48008, -0.096239, 0.109163, 0.100811, -0.030184, -0.080974, 0.190151, -0.151845, 0.400954, 0.316308, -0.162318, -0.172029, 0.118151, -0.089878, -0.051, 0.151209, -0.03351, 0.348607, -0.241087], [0.58272, -0.365684, -0.081833, 0.181727, -1.020082, -0.229723, 0.001531, -0.674312, -0.11095, -0.360826, -0.288242, 0.117405, 0.084551, -0.013852, -0.000851, -0.051128, 0.555506, -0.090373, 0.579571, -0.51731, 0.395596, -0.842583, -0.251689, 0.032896, -0.059186, -0.217232, 0.356512, 0.490726, -0.26123, 0.110798, 0.123801, 0.01624, 0.436638, 0.050367, -0.34998, 0.140157, -0.317978, -0.230092, -0.14336, 0.230172, -0.113308, -0.48769, 0.057788, 0.504021, 0.344097, -0.068493, 0.166432, -0.27595], [0.135253, 0.035973, 0.246306, 0.205836, 0.016721, 0.424249, -0.327894, -0.260821, 0.42649, -0.150374, -0.143777, 0.145058, 0.310085, 0.148127, 0.184325, -0.157177, -0.107553, -0.248314, -0.007762, 0.045507, -0.24468, -0.249899, -0.003694, -0.098211, -0.563479, 0.149072, -0.205275, -0.128746, 0.750255, -0.297681, -0.656019, 0.001529, 0.053072, -0.295855, 0.085055, 0.012228, -0.078406, -0.289757, 0.148469, -0.016378, -0.072259, 0.020312, 0.02011, 0.097546, -0.232479, -0.165102, -0.40964, -0.023268], [0.149701, 0.393183, -0.413821, 0.349042, -0.210554, 0.266097, -0.153396, -0.53417, 0.888752, -0.419485, 0.308769, -0.014761, -0.264984, -0.023825, 0.184689, -0.031477, -0.23106, 0.536246, -0.00935, 0.012238, -0.290949, 0.733286, 0.269884, 0.286377, -0.392385, -0.14218, 0.356095, -0.25293, -0.038047, 0.043357, 0.160901, -0.154943, -0.048135, -0.061526, -0.094585, -0.154598, 0.177522, -0.344887, 0.242382, -0.005168, 0.211507, 0.087597, -0.008048, 0.134601, -0.293656, 0.135741, -0.119683, -0.089353], [-0.67485, -0.472819, 0.266478, 0.115706, -0.163008, 0.231141, -0.38294, -0.184831, -0.48539, 0.054448, -0.040483, -0.014928, 0.581851, -0.177432, -0.185961, -0.01834, 0.168167, -0.192577, 0.277192, -0.039553, -0.177378, -0.66557, -0.005854, 0.256819, 0.408978, -0.09701, -0.226979, -0.188096, 0.266892, -0.301404, -0.555062, -0.164756, -0.064886, 0.250439, -0.041791, -0.015376, -0.032563, 0.283761, -0.00956, -0.396169, 0.346853, -0.14154, 0.01809, 0.088797, -0.155437, -0.161088, -0.268503, -0.003691]]
SEQ_BI = [0.040519, 0.188465, 0.150624, 0.043322, 0.06546, 0.020525, -0.148073, 0.218602, 0.081386, 0.290627, -0.063954, 0.228129, 0.098671, 0.225387, 0.236986, 0.132239, 0.764756, -0.187911, 0.230701, 1.152894, 0.441283, -0.058766, 0.344517, 0.546621, 0.307632, 0.670412, 0.847833, 0.80039, 0.068196, 1.036404, 0.102541, 0.825381, -0.190573, -0.394962, 0.204255, -0.211891, -0.042446, 0.061222, -0.004258, 0.261591, 0.244646, -0.112218, 0.089713, -0.060206, -0.326113, -0.055909, 0.370496, -0.053185]
SEQ_WH = [[-0.135869, -0.785515, -0.045826, -0.09881, -0.548525, -0.5185, -0.005697, 0.241422, 0.314739, 0.254685, 0.138707, -0.270169, 0.253356, -0.332395, 0.33931, 0.196264, 0.122746, 0.033921, -0.113167, -0.868194, -0.09105, -0.547692, -0.375984, 0.660343, 0.467687, -0.307216, 0.365494, -0.14426, 0.436708, -0.024295, -0.014302, -0.379843, 0.298518, 0.250991, 0.0149, -0.023749, -0.276281, -0.431085, -0.088066, -0.105062, 0.067208, -0.410851, 0.457952, -0.018622, 0.15985, 0.428842, 0.286648, -0.268048], [-0.02649, -0.30455, 0.210155, -0.208541, 0.21585, 0.051062, -0.21445, 0.04469, 0.205219, 0.134126, -0.249847, 0.006213, 0.004707, -0.308586, 0.168003, -0.106299, -0.71431, -0.290446, -0.458882, -0.69027, -0.408771, -0.380028, -0.644909, -0.452951, -0.259766, -0.906876, -0.474316, -1.202887, -0.35285, -0.737633, 0.138753, -1.001165, 0.15641, 0.616578, -0.357379, 0.134623, 0.106764, -0.030566, -0.239276, 0.007162, -0.063152, 0.163337, -0.208277, 0.009835, 0.354074, 0.016664, -0.174663, 0.345536], [-0.065816, 0.262762, 0.51013, 0.475708, 0.329325, 0.278145, 0.029669, -0.430728, -0.142264, -0.130335, 0.149499, 0.249081, 0.402048, -0.265877, -0.295101, -0.579973, 0.768236, 0.397618, 0.67047, 0.02802, 0.313612, 0.389774, 0.452434, 0.021014, 0.53325, 0.812753, 0.586779, 0.844543, -0.305354, 0.491251, 0.35824, -0.649996, -0.373044, -0.443083, 0.377372, -0.519904, -0.175706, -0.125871, -0.081255, -0.424916, -0.030209, 0.193237, -0.200831, -0.369743, -0.061089, 0.170142, 0.316535, 0.192126], [0.736858, -0.016383, -0.00662, -0.343289, 0.370172, -0.196269, -0.209599, 0.802113, 0.020484, 0.328606, -0.203306, 0.524483, -0.079153, 0.888709, -0.083791, 0.269743, -0.398796, -0.135063, -0.586009, -0.023799, -0.133312, 0.067727, -0.561199, -0.119591, -0.40698, -0.198108, 0.01119, -0.206076, -0.697288, -0.51209, 0.128964, 0.333493, -0.155786, -0.015401, -0.356886, 0.487946, 0.122836, -0.236098, -0.184964, 0.102909, 0.570828, -0.605379, 0.325833, -0.432856, 0.186261, 0.201224, -0.209307, -0.050047], [0.381835, 0.056723, 0.157214, 0.121345, 0.77394, 0.763825, 0.028315, -0.038641, -0.186297, 0.317106, -0.302497, -0.081419, -0.225182, -0.10149, 0.179105, -0.406415, 0.078032, 0.101496, -0.870332, -0.337351, 0.233382, 0.717606, -0.275341, -0.333728, -0.164064, -0.125819, -0.442575, -0.359247, -0.805011, -0.002351, 0.300321, 0.654846, -0.463699, 0.257818, -0.46293, -0.060534, 0.763819, 0.327835, -0.470878, 0.269262, -0.280731, 0.084494, -0.166919, -0.032857, 0.035418, -0.449784, 0.047828, 0.49453], [0.114505, 0.051575, 0.595253, 0.290516, 0.059107, 0.315386, -0.495474, -0.138024, -0.216108, 0.539204, -0.239318, 0.268302, -0.135054, 0.297702, -0.387573, -0.389141, -0.162893, -0.024084, -0.336867, 0.904773, 0.099621, 0.397152, -0.288251, 0.026916, 1.161021, 0.910668, 0.598607, 0.328571, -0.754629, 0.680291, 0.293322, 0.296015, -0.510464, 0.005164, -0.303207, -0.574936, 0.329261, -0.207931, -0.158682, 0.405941, -0.230855, -0.24569, 0.358672, 0.00732, 0.244028, -0.077363, 0.010663, 0.206909], [0.327902, -0.148873, 0.092558, 0.301367, -0.039313, 0.079069, 0.138059, -0.206928, -0.07516, -0.187807, 0.186799, -0.238808, -0.232155, -0.077871, 0.265466, 0.023917, 0.345478, -0.632037, -0.143835, 0.626119, 0.019995, 0.072264, 0.325838, 0.155735, 0.058133, 0.093617, 0.027282, 0.331845, 0.584906, 0.565988, -0.411694, -0.354798, 0.255955, -0.014167, 0.452084, -0.505153, 0.209154, 0.574019, 0.061481, -0.620195, -0.009618, 0.229206, 0.061455, -0.086568, -0.301388, -0.301746, 0.566397, 0.020454], [0.501988, 0.66126, -0.085202, -0.519825, -0.013531, -0.007558, 0.147738, 0.180646, -0.048111, 0.324755, -0.061012, 0.162621, -0.322871, 0.220499, 0.150009, 0.665114, -0.183534, 0.200366, -0.171042, 0.678155, 0.39497, 0.246649, 0.163555, -0.73606, -0.556643, 0.385513, -0.013526, 0.660062, -0.484457, 0.788773, -0.014629, 0.47637, 0.357104, 0.062679, -0.014676, 0.530776, 0.362397, 0.219302, -0.169792, 0.447163, 0.285671, -0.220938, 0.118907, 0.201442, -0.412969, 0.045105, 0.10592, 0.075755], [0.144092, -0.686541, -0.672837, -0.566193, -0.116358, 0.495965, -0.350498, 0.12236, 0.672898, 0.392067, 0.24846, -0.264825, 0.396176, 0.727596, -0.625101, 0.24769, -0.01976, 0.171445, -0.08795, 0.097166, 0.122512, -0.592505, -0.400746, 0.132462, -0.385331, -0.021242, -0.189964, -0.464477, 0.439907, 0.027895, -0.158318, 1.615997, 0.000239, -0.078539, 0.037823, 0.743179, -0.331177, 0.184326, -0.101999, 0.286606, 0.338429, -0.433512, 0.177329, -0.388009, 0.301818, 0.373114, 0.022711, -0.153692], [-0.568774, -0.120899, 0.303737, 0.245293, -0.228922, -0.184987, 0.29012, -0.266691, -0.002206, -0.068464, -0.200041, -0.024276, -0.115999, -0.075579, 0.257015, -0.354032, -0.65002, 0.326053, -0.063145, -1.048787, 0.165313, -0.540972, 0.247391, -0.715164, -0.184853, -0.42045, -0.321952, -0.287949, -0.128033, -0.560245, -0.347248, -0.492645, 0.127738, 0.291251, 0.092757, -0.460163, 0.334377, 0.23879, 0.246407, -0.325246, -0.131685, 0.540482, -0.106588, -0.326365, 0.296307, 0.167992, 0.110386, 0.246744], [0.407732, 0.036441, 0.058587, -0.013086, -0.581068, 0.097543, 0.303064, -0.253022, 0.17359, 0.185769, 0.424746, 0.515157, 0.151624, 0.929744, -0.067666, 0.058583, 0.570614, 0.074833, -0.004928, 0.383759, 0.199628, -0.267251, 0.188526, 0.042089, 0.284056, 0.740219, 0.477748, 0.10557, -0.078137, 0.623825, -0.066273, -0.162469, 0.012573, -0.356622, -0.371157, 0.47494, -0.402506, 0.244975, 0.010165, 0.15944, 0.581448, -0.504525, 0.038738, -0.017932, -0.112942, 0.228145, -0.023994, -0.283019], [0.011902, -0.181798, -0.032887, 0.484438, -0.798544, -0.743414, -0.245413, -0.462353, -0.158256, -0.538934, -0.681034, -0.258714, -0.350387, -0.409686, 0.192251, 0.098632, -0.242379, -0.206704, -0.698276, -0.049791, -0.312438, -0.283288, 0.055515, -0.113461, -0.337926, -0.115164, -0.472218, -0.496021, 0.520434, 0.213035, -0.311766, 0.413459, -0.192356, -0.357037, -0.608359, 0.319383, -0.158263, -0.095339, -0.240993, -0.134704, 0.20681, 0.182445, -0.298468, 0.377284, 0.055687, -0.725876, -0.1708, -0.148871], [0.300559, -0.259922, -0.110014, -0.042505, -0.373712, -0.173669, 0.089357, 0.201002, 0.300702, 0.057155, 0.119385, 0.15838, -0.18584, 0.237377, -0.140744, 0.087505, -0.007211, -0.063725, 0.444088, -0.752024, 0.322011, -0.692161, -0.782903, -0.25325, -0.387741, -1.130525, -0.918748, -0.074817, -0.579543, -0.85027, 0.310598, -0.561778, 0.321287, 0.698409, -0.46423, -0.077769, 0.364199, -0.126373, 0.179214, 0.047381, 0.144613, -0.190337, -0.299467, -0.006553, 0.080549, 0.475508, -0.213724, -0.201337], [-0.498897, -0.671193, -0.415467, -0.267787, -0.217083, 0.01799, 0.112208, 0.269142, 0.715269, 0.373346, -0.009268, 0.142508, -0.276712, 0.009408, -0.298468, -0.11173, -0.284655, 0.931145, -0.381618, -0.210096, -0.403152, 0.278698, -0.043962, -0.408185, -0.100052, -0.743945, 0.076296, -0.702234, -0.566509, -0.440466, -0.222024, -1.140786, -0.072354, 0.718198, 0.129034, -0.130995, -0.22962, -0.16273, -0.245797, 0.158871, 0.115489, 0.32527, -0.060021, -0.145275, 0.043147, 0.427406, 0.101073, 0.109173], [-0.419068, -0.367918, -0.374601, -0.271954, -0.710567, -0.389852, 0.2474, -0.44366, -0.167913, 0.241442, -0.12148, 0.039447, 0.150359, -0.395528, -0.044968, 0.357098, 0.557278, 0.074383, -0.082949, 0.568875, 0.888353, -0.262084, 0.255108, 0.672512, 0.69739, -0.020347, 0.72629, 0.022843, -0.116896, 0.551294, -0.349901, 0.450435, 0.186725, -0.210482, 0.061455, -0.113081, -0.060299, 0.062386, -0.018633, -0.043571, -0.135689, -0.108325, -0.170639, 0.12618, -0.28586, 0.144383, 0.071968, -0.666581], [0.45976, 0.328679, 0.377512, 0.594324, 1.138649, 0.872972, -0.017998, -0.168542, 0.416997, -0.145972, -0.251283, -0.261309, -0.099576, 0.266405, 0.339884, -0.9201, -0.307722, -0.440184, -0.330769, -0.698376, -0.667707, -0.01233, 0.619196, -0.800691, -0.342111, -0.246384, -0.223099, -0.313482, 0.199785, -0.493433, 0.118451, 0.356545, -0.221979, 0.117543, 0.24664, -0.189283, 0.528166, 0.259409, 0.199925, -0.088416, 0.09746, 0.368277, -0.225359, -0.325701, 0.107727, -0.03875, -0.419351, 0.585945]]
SEQ_BH = [0.230822, 0.107216, 0.060762, -0.097437, -0.06855, 0.234381, 0.164245, 0.219174, -0.168933, 0.192573, -0.15056, 0.061971, 0.023622, -0.051797, 0.016143, 0.086553, 0.940286, 0.133527, 0.372644, 0.917065, 0.259859, 0.189074, 0.042464, 0.540473, 0.354592, 0.715826, 0.478451, 0.524396, 0.024992, 0.71866, 0.316238, 0.734366, 0.157711, -0.19383, 0.040137, -0.15722, -0.049159, 0.258502, 0.270862, 0.312188, 0.038899, 0.004676, -0.156535, 0.111806, -0.189649, -0.100551, 0.111995, -0.391141]
SEQ_HEAD = [
    ([[0.071771, 0.21965, 0.208835, 0.022908, -0.036159, 0.083278, 0.095519, 0.169767, 0.20495, 0.016227, -0.053243, 0.019452, -0.081302, 0.002234, -0.085598, 0.02721, 0.044213, 0.025935, 0.068086, 0.056329, 0.078173, -0.002271, 0.176516, -0.090383, 0.068008, -0.131528, -0.079263, -0.171705, -0.06534, 0.141914, 0.046036, -0.05776, 0.305486, 0.078486, 0.153559, -0.249009, -0.176524, 0.078336, 0.061818, 0.232526, -0.069943, 0.035927, 0.097508, 0.0216, -0.199953, -0.075177, -0.079362, -0.330665, 0.075928, 0.096482, -0.118939, -0.25824, -0.118395, -0.051896, -0.209778, 0.029794, -0.019696, -0.159607, -0.014251, -0.23655, 0.182953, -0.080041, 0.058451, 0.029387], [0.029352, 0.018639, -0.035584, -0.166024, 0.169271, 0.032265, 0.106805, -0.059051, 0.113004, -0.09985, 0.038929, -0.062476, 0.160606, 0.060851, 0.064253, 0.185831, 0.000191, -0.038181, 0.152716, -0.07089, -0.030198, 0.006391, 0.120792, 0.096711, 0.033885, -0.171152, -0.072091, 0.069552, -0.079123, 0.021732, -0.01915, 0.061327, 0.111315, -0.042998, 0.028744, -0.182319, -0.014027, -0.038818, -0.094954, -0.026571, -0.085409, -0.115913, 0.098219, 0.042969, 0.156932, -0.041652, -0.002409, -0.050199, 0.07741, -0.05386, -0.096752, -0.17895, 0.120569, -0.061285, 0.108677, -0.012755, -0.025404, 0.060014, 0.126513, -0.183323, 0.049045, -0.185562, -0.139423, 0.062231], [0.047321, -0.044004, -0.069895, -0.091845, 0.145167, -0.080412, 0.058719, -0.109428, -0.06462, 0.030367, 0.077461, -0.152788, -0.073888, -0.199702, 0.036497, 0.235906, -0.003855, -0.170184, 0.187076, 0.073906, -0.150735, 0.178927, 0.040941, 0.113865, 0.092183, -0.090243, -0.062385, 0.110673, 0.05005, 0.06258, -0.122301, -0.073137, 0.004616, 0.106045, 0.047977, -0.028833, 0.013525, 0.110338, -0.032966, 0.111133, 0.005668, 0.004187, -0.082862, -0.06154, -0.081579, 0.06647, 0.037114, -0.021547, 0.151894, -0.061687, -0.143434, -0.087285, -0.004883, -0.020536, 0.182762, -0.107529, -0.011051, -0.065818, 0.179391, -0.027853, -0.176651, 0.063169, -0.040307, 0.029401], [0.099933, 0.001967, -0.159422, 0.073135, -0.039828, 0.121413, -0.029176, -0.186977, 0.199134, -0.190634, 0.055704, -0.15165, -0.085542, -0.223085, -0.099107, 0.078133, -0.178163, -0.053715, 0.172146, 0.063723, -0.000221, -0.065157, -0.059018, 0.004734, -0.070324, -0.038261, -0.142621, -0.061535, 0.063145, -0.222543, -0.098162, 0.143044, -0.247024, -0.085185, -0.041118, 0.179959, -0.066039, 0.05237, 0.13064, -0.096886, -0.051779, 0.079522, 0.105754, 0.000235, 0.145675, -0.14107, 0.079472, 0.071259, 0.179272, -0.077079, 0.083351, 0.11112, 0.179491, -0.015581, 0.080796, 0.142945, -0.146337, 0.215826, 0.151457, 0.119164, -0.158237, 0.039354, 0.202886, -0.282447], [-0.266762, -0.231277, -0.133753, 0.088921, 0.076806, -0.030104, 0.116194, 0.250786, -0.16745, -0.077947, 0.159293, 0.364494, -0.168218, -0.33952, 0.111944, 0.041224, -0.212856, -0.026157, 0.308824, -0.051577, -0.119277, 0.18253, -0.071509, 0.220744, 0.192324, 0.205715, -0.043166, 0.189122, -0.018165, -0.038217, -0.035022, 0.011284, -0.380018, 0.159377, 0.039939, 0.078562, 0.182733, -0.282754, -0.06414, -0.013973, 0.234415, 0.075992, 0.053188, 0.22849, 0.108308, -0.100391, 0.216337, 0.283215, 0.184631, 0.083048, 0.210983, 0.024006, -0.111698, 0.107994, 0.201944, 0.113387, -0.213265, -0.239707, -0.187259, 0.127285, -0.426039, 0.021467, -0.140039, 0.21576], [-0.121192, 0.204125, -0.048487, -0.126971, 0.08172, -0.099703, -0.099483, -0.081848, 0.08174, 0.137343, 0.064812, 0.059017, 0.225828, -0.141322, -0.118749, 0.120678, -0.02508, 0.211851, 0.02623, 0.012706, 0.148375, -0.072396, 0.043604, 0.099874, -0.209794, -0.19027, 0.129232, 0.157673, -0.023498, 0.071325, -0.07233, -0.100541, 0.123233, 0.016309, 0.004384, 0.077532, -0.094157, 0.17108, -0.16055, -0.126121, 0.023649, 0.02262, 0.153603, 0.156573, -0.012277, -0.071695, 0.166367, -0.022182, -0.089401, -0.026267, 0.070425, 0.06993, 0.255708, -0.035866, 0.185051, -0.051564, 0.058275, 0.023743, 0.022697, 0.109954, 0.015544, -0.120066, 0.201389, -0.101062], [-0.157323, -0.228159, -0.074036, 0.121378, -0.123351, 0.089565, 0.178301, 0.213696, 0.085077, 0.03382, 0.074371, -0.028243, -0.246922, 0.058376, -0.229673, 0.014436, -0.093547, -0.19618, 0.190134, 0.185355, -0.158928, -0.012978, 0.049989, 0.012634, -0.10361, -0.247309, -0.012544, -0.039756, -0.019501, -0.215404, 0.016065, 0.00038, 0.299359, 0.082941, -0.100192, -0.122963, 0.059606, -0.268762, -0.221687, 0.152604, 0.138145, 0.15872, 0.023951, -0.276395, -0.010733, 0.019692, -0.074228, -0.349025, 0.164497, -0.186421, -0.104355, -0.062653, 0.012665, 0.1693, 0.148253, 0.317944, 0.143847, 0.003927, -0.047478, -0.154386, -0.118224, -0.042646, -0.051042, 0.099045], [0.172094, -0.179428, 0.170803, -0.075665, 0.07021, -0.047495, -0.111296, 0.030696, 0.031597, 0.222123, 0.102703, 0.147698, -0.014636, 0.128862, 0.027482, 0.109534, 0.137116, -0.067283, 0.003251, 0.044596, 0.087794, -0.115099, 0.070594, 0.092237, 0.031865, 0.048675, -0.172637, 0.12373, 0.044289, 0.059892, -0.037807, 0.018458, -0.065758, 0.022635, -0.072543, -0.007551, -0.023068, 0.000391, -0.27043, -0.01153, -0.092172, -0.125366, 0.018201, -0.121291, 0.288098, -0.031018, -0.284266, 0.022952, -0.075114, 0.006142, -0.012581, 0.010077, 0.10467, 0.034185, -0.053849, -0.143937, -0.018641, -0.040439, -0.054155, -0.009767, -0.119945, -0.093026, 0.101283, 0.245932], [-0.21324, -0.102641, -0.001667, 0.275836, 0.203758, -0.156795, -0.07448, 0.138514, -0.020167, 0.057864, -0.202422, 0.197216, -0.1717, -0.268549, -0.03374, 0.220492, -0.070313, -0.120055, -0.026709, -0.154605, -0.176276, 0.252563, -0.253498, 0.069193, 0.110415, -0.012155, -0.157785, -0.118841, -0.036257, 0.126005, 0.153991, 0.324665, -0.107532, 0.206777, -0.03219, -0.170282, 0.175813, -0.199864, 0.051412, 0.05581, 0.182089, -0.140182, 0.006413, 0.261129, -0.043518, -0.025181, 0.058223, -0.130939, 0.112226, 0.115116, 0.126899, -0.013942, -0.006911, 0.033274, -0.011122, 0.062816, 0.155578, -0.039389, 0.006472, 0.222211, 0.099813, -0.272847, 0.210931, 0.011152], [0.137273, -0.117083, 0.18347, 0.328538, -0.168306, 0.010832, -0.223002, 0.022625, -0.034723, 0.006941, 0.183538, 0.090201, -0.281263, -0.141595, -0.078844, -0.247983, 0.044932, -0.071056, -0.115217, 0.14634, -0.072758, -0.188959, -0.220435, -0.122971, 0.246818, -0.075389, 0.04985, 0.296176, -0.004997, 0.032056, 0.09546, -0.249249, 0.155571, -0.169862, 0.205695, -0.459885, 0.038258, -0.232418, 0.039279, -0.20817, -0.053918, -0.13545, -0.055812, 0.028489, -0.080033, 0.010014, -0.045899, -0.166012, 0.012614, 0.06947, 0.06683, -0.016126, 0.082696, 0.070354, 0.212672, -0.079809, 0.081605, -0.190259, -0.071564, -0.023075, -0.00921, -0.117757, 0.202118, -0.011439], [-0.073171, -0.120781, -0.058592, -0.078941, 0.070213, 0.131801, 0.011798, -0.032847, 0.030214, -0.023719, 0.133234, 0.006818, 0.090195, -0.051439, 0.029044, -0.176314, 0.090912, 0.067406, -0.05856, 0.16542, 0.103657, 0.090374, -0.1613, -0.184347, -0.164796, 0.099257, 0.068151, 0.179753, 0.150966, 0.076256, 0.021087, 0.141825, -0.112145, -0.150159, -0.149663, -0.021013, -0.165948, -0.101816, 0.104343, 0.083249, -0.049959, -0.070321, 0.066742, -0.23533, -0.021434, -0.001066, -0.178116, -0.09377, 0.091538, -0.017496, -0.214914, -0.04036, 0.011083, 0.019877, -0.002902, 0.038599, -0.019417, 0.045856, -0.048791, 0.080612, 0.018972, -0.100858, 0.037565, 0.04405], [-0.277299, -0.392675, -0.173731, -0.096218, 0.2886, -0.194901, 0.071779, -0.088135, 0.394731, -0.248276, -0.230007, 0.25741, -0.081626, -0.033708, -0.07238, 0.017255, -0.162438, -0.217262, 0.218922, -0.04167, 0.001156, -0.19783, 0.205623, -0.020609, 0.160652, -0.007349, -0.168436, 0.054522, -0.29451, 0.26592, -0.252397, 0.308737, 0.000598, 0.11571, -0.031935, 0.053365, -0.068184, 0.112074, 0.059372, 0.208397, 0.064755, -0.327317, 0.284726, -0.003001, -0.209706, -0.011527, 0.194749, 0.236147, 0.049052, 0.109282, -0.064714, -0.247113, 0.067016, 0.179083, 0.039308, 0.252529, -0.21128, -0.09723, 0.220927, -0.113246, -0.171843, 0.010749, 0.240927, 0.066016], [0.078363, 0.122058, -0.034666, 0.068424, -0.094086, 0.014106, 0.00322, -0.070168, -0.179566, -0.021139, 0.070474, -0.081793, -0.25297, 0.083961, -0.081796, 0.010565, 0.237049, 0.072043, 0.015075, 0.023251, -0.081087, -0.128598, -0.001739, -0.185962, -0.065049, -0.09045, 0.080384, -0.039743, 0.122658, 0.021368, 0.163824, 0.027119, -0.062383, -0.154593, -0.107252, -0.113677, -0.003413, 0.196372, -0.18868, -0.10413, 0.185562, -0.052517, 0.22005, 0.049727, -0.228671, -0.20627, 0.000707, -0.19959, -0.042954, -0.082709, -0.091923, 0.05072, 0.055614, -0.041004, 0.074654, -0.016131, 0.133605, -0.107184, -0.070128, -0.072211, -0.032873, -0.006278, -0.105628, 0.01791], [-0.045052, 0.136082, -0.053364, 0.170943, -0.054735, -0.081158, 0.137433, -0.136739, -0.100279, 0.135812, 0.203023, 0.025569, 0.034977, -0.002638, -0.114231, -0.042865, 0.122485, 0.071802, 0.125108, -0.177376, -0.004276, -0.130279, -0.115581, -0.03231, 0.076715, 0.119072, -0.049004, -0.03332, 0.070508, -0.153326, 0.107652, 0.218249, -0.194137, -0.13678, -0.181145, -0.145046, -0.150822, 0.11807, -0.103976, -0.164386, 0.05909, 0.17438, 0.040038, 0.223286, -0.158646, 0.10316, 0.02586, 0.09489, -0.027792, -0.103899, 0.060442, -0.065637, -0.014569, -0.125119, 0.052536, 0.070249, -0.032338, -0.105823, -0.029505, -0.163, -0.117498, 0.07021, -0.084059, -0.165493], [-0.054976, 0.199543, 0.057513, -0.226929, 0.08597, -0.022876, -0.041788, -0.103009, 0.148974, -0.088535, 0.316949, -0.255962, 0.113034, 0.065365, -0.012193, -0.003177, 0.200084, 0.111369, -0.104885, -0.226264, -0.147035, 0.260086, -0.048062, 0.127273, -0.083847, -0.042356, -0.079405, 0.107572, -0.066046, 0.090783, 0.033853, -0.001268, -0.09084, 0.021343, 0.072274, 0.039184, -0.071666, 0.00729, 0.020846, 0.138698, 0.075554, -0.117852, -0.203611, -0.119871, -0.150175, 0.103487, -0.057633, 0.085604, 0.030007, 0.030143, -0.208542, -0.00175, 0.245369, 0.177071, -0.099899, -0.162349, -0.073946, 0.021031, 0.100369, 0.15479, 0.179785, 0.022325, 0.077499, -0.033816], [-0.078457, -0.0646, 0.097862, 0.172274, -0.145841, 0.01223, 0.196356, 0.006704, 0.213777, 0.170058, -0.050353, -0.184821, 0.174346, 0.089242, -0.089766, 0.224737, 0.132169, 0.105663, -0.130209, -0.254193, -0.274113, 0.12444, 0.186824, -0.098975, 0.126198, 0.061489, 0.018303, -0.157384, -0.090996, 0.059416, 0.087468, 0.146284, 0.215149, 0.135392, -0.194907, 0.028488, 0.015149, -0.052512, -0.09532, -0.018657, 0.008579, 0.036664, -0.212731, 0.16678, -0.010082, 0.106484, -0.19178, -0.028553, -0.057951, 0.187557, -0.019521, 0.132465, -0.271444, 0.257406, 0.221534, -0.074549, -0.109393, 0.010625, 0.249207, 0.058568, -0.003563, -0.032126, 0.108911, 0.114769], [0.290366, -0.022665, -0.151211, 0.058349, 0.078115, 0.035685, -0.295663, -0.038857, 0.212708, -0.251462, -0.097148, 0.006499, 0.102059, -0.112421, 0.094334, -0.143154, 0.108345, -0.293072, 0.336629, 0.133506, -0.065912, -0.097544, 0.18515, 0.082524, 0.402881, 0.32697, -0.067129, -0.204439, 0.092197, 0.44116, 0.059838, -0.210792, -0.051119, -0.187509, -0.062485, 0.004434, -0.16208, 0.232852, 0.234262, -0.093461, 0.249111, -0.408914, -0.089154, -0.175578, 0.076446, -0.034352, -0.155539, 0.066135, 0.113951, 0.055535, -0.018988, -0.311195, -0.413405, 0.108488, -0.120767, -0.03099, -0.259811, 0.037377, 0.14832, -0.112816, -0.145174, -0.187603, -0.213811, -0.116981], [0.113616, -0.017648, 0.075701, 0.046982, 0.097071, 0.130418, -0.018291, -0.011949, 0.27876, -0.022024, 0.092955, -0.078905, -0.241191, -0.049301, 0.120444, 0.023041, -0.176771, 0.016687, -0.005148, 0.112144, -0.04047, 0.164712, 0.286427, -0.197581, 0.008484, 0.234845, 0.073955, -0.122085, -0.105806, -0.095173, 0.033571, 0.288638, -0.172162, -0.021921, -0.087819, -0.059062, 0.232906, 0.08223, 0.049976, 0.142004, 0.048916, -0.001575, 0.213363, 0.109276, 0.143257, 0.131926, 0.091113, -0.244446, 0.054699, -0.05652, 0.178667, -0.164032, 0.082904, -0.183236, -0.069086, -0.241946, 0.209187, 0.263021, -0.311681, 0.16678, -0.3003, -0.060851, 0.022455, -0.050983], [-0.058726, -0.222674, 0.099885, -0.004969, -0.032726, -0.006371, 0.161142, 0.008597, -0.104949, 0.243982, 0.085412, -0.027051, 0.184828, -0.029316, -0.091771, -0.197282, 0.051676, 0.121563, 0.010608, -0.064712, -0.051109, 0.153923, 0.000833, 0.169242, 0.114684, 0.100798, -0.151179, 0.078953, 0.095546, 0.036091, 0.07741, 0.048775, 0.018884, 0.134141, 0.06095, -0.066481, 0.009963, -0.108891, -0.012753, -0.060401, 0.153062, 0.026895, -0.098334, 0.004868, -0.038074, -0.163688, -0.044019, 0.048899, -0.265825, 0.104152, 0.091296, 0.010111, -0.044711, 0.050828, 0.022971, -0.075886, 0.073901, -0.019249, 0.037017, -0.079145, -0.026834, 0.104117, -0.082954, 0.090335], [0.279342, -0.074943, 0.085366, -0.184361, -0.01859, -0.219176, -0.206808, -0.00533, 0.15902, -0.156207, -0.369425, -0.037191, -0.194541, -0.088358, 0.03494, -0.094929, 0.23611, 0.088034, 0.098006, 0.173841, -0.353505, 0.133244, 0.043513, -0.006206, 0.11377, 0.346229, -0.238269, -0.063455, 0.226195, -0.155272, -0.046131, 0.094743, -0.010548, -0.115902, 0.091976, -0.018087, -0.02246, 0.105291, -0.525317, 0.016886, 0.089008, 0.067888, -0.28787, 0.116954, 0.135522, -0.086201, 0.289641, -0.10885, 0.037682, 0.220077, 0.033249, -0.199137, -0.042981, 0.04053, 0.268255, 0.026063, -0.058042, 0.090544, -0.022654, -0.070685, -0.144879, -0.019838, 0.070369, 0.290198], [-0.084864, 0.08394, 0.090661, 0.01232, 0.021271, -0.309482, 0.13317, -0.171196, 0.033219, -0.202452, -0.261877, -0.052411, 0.026723, 0.174574, 0.09523, -0.155782, -0.248408, 0.192989, -0.153182, -0.466861, 0.05848, -0.175653, -0.046301, -0.113289, -0.040839, -0.185645, -0.01922, -0.013697, -0.297821, -0.025657, -0.08866, -0.050072, 0.019012, 0.324835, -0.17485, 0.213468, -0.104983, 0.12806, 0.151567, 0.161616, 0.038746, -0.230583, 0.234884, 0.398997, 0.049453, 0.200381, 0.107806, 0.042654, -0.071039, 0.055888, 0.02006, 0.042156, 0.134231, 0.023875, -0.255067, 0.06483, -0.204782, -0.011778, 0.027267, 0.267875, 0.309856, 0.081422, -0.014383, 0.007147], [0.340314, -0.046422, 0.056937, 0.0358, -0.018107, -0.065339, 0.079317, 0.044154, -0.029593, -0.139693, -0.172131, -0.053001, 0.051298, 0.095622, 0.324375, 0.079513, 0.173409, 0.484423, 0.041006, -0.0935, 0.021717, 0.105142, 0.051509, 0.056425, -0.132869, -0.081772, 0.188375, -0.007378, 0.071688, -0.03593, 0.116103, 0.049381, -0.067096, -0.118559, 0.694138, 0.080406, -0.034453, -0.124344, 0.001621, 0.060612, -0.173112, -0.138238, 0.193681, -0.021953, 0.154837, -0.111538, -0.261839, 0.152421, 0.058082, -0.164049, -0.040924, -0.053924, -0.111453, -0.047229, -0.032935, 0.60165, 0.13345, -0.061122, -0.000957, 0.110381, 0.119379, 0.08437, 0.179653, 0.067085], [-0.128881, -0.181811, -0.303066, 0.191224, 0.117424, -0.090854, -0.388212, 0.336256, 0.049969, 0.454267, -0.414668, 0.208352, -0.417369, -0.339865, -0.106821, -0.084866, -0.238113, 0.03993, -0.061968, -0.260515, -0.466787, -0.159547, -0.160479, 0.227144, 0.308582, 0.171728, -0.034741, -0.124399, -0.025371, 0.30631, 0.056339, -0.036883, -0.003993, -0.220157, 0.095471, -0.292615, -0.264168, 0.21596, 0.161363, -0.287905, -0.129424, -0.382428, -0.185396, -0.191633, 0.072069, -0.215348, -0.065727, 0.058699, -0.038886, 0.49483, -0.275386, -0.196817, -0.27634, 0.15679, 0.094104, -0.031997, -0.264629, -0.155778, -0.095284, -0.204423, -0.038381, -0.25647, 0.06386, -0.041407], [-0.065353, 0.032813, -0.099619, 0.091112, 0.113734, 0.226189, 0.138255, -0.072252, 0.264839, 0.130971, 0.058075, -0.063607, -0.09954, 0.176216, 0.077092, 0.018218, 0.040971, 0.033836, 0.120429, 0.269413, -0.088871, -0.067166, -0.140444, 0.230033, -0.049798, 0.252669, 0.098799, 0.025303, 0.062296, -0.050631, 0.13573, 0.060384, -0.084329, 0.096564, 0.087947, -0.117803, 0.060601, 0.207475, 0.223695, 0.002232, 0.061191, -0.016023, 0.016335, 0.064871, 0.061132, -0.098658, 0.209159, 0.135359, 0.073941, -0.019243, 0.006497, 0.017439, -0.037391, -0.035755, 0.123004, 0.147184, 0.030378, -0.01302, 0.123076, -0.011366, 0.074711, 0.115263, -0.013778, -0.055925], [-0.349407, -0.025697, 0.041175, 0.037195, -0.036853, -0.248311, -0.284521, -0.017607, -0.283395, -0.175123, 0.081673, 0.119857, 0.055802, -0.024787, -0.496223, -0.108035, -0.015445, 0.09511, -0.022177, -0.205621, 0.150648, -0.111878, -0.135993, -0.046411, 0.064843, 0.021979, -0.103198, -0.166026, 0.03313, -0.083702, -0.013588, 0.192733, -0.130689, -0.410255, -0.156705, 0.101909, -0.126551, -0.068733, -0.07188, -0.031343, -0.032401, 0.165153, 0.064819, -0.000848, -0.363452, 0.130829, -0.044494, -0.103194, -0.01251, 0.202627, -0.11617, 0.019336, -0.278635, -0.146566, 0.023458, -0.18221, -0.20127, -0.174795, -0.052834, 0.088457, 0.152544, -0.052811, -0.285101, 0.014001], [0.116751, 0.121341, -0.010222, 0.055503, 0.17149, 0.221231, 0.802233, 0.54983, 0.184532, -0.084195, -0.167031, 0.093948, 0.415337, 0.417119, -0.029136, -0.223097, 0.055766, -0.266149, 0.38246, -0.030768, 0.279867, 0.479292, 0.381539, 0.21361, 0.060996, 0.280777, 0.146281, 0.203427, 0.372128, 0.245593, 0.18532, -0.075698, 0.312664, 0.191417, -0.119484, 0.653707, 0.078137, -0.136524, 0.178604, 0.089965, 0.494035, -0.048492, 0.132923, 0.212743, -0.251538, 0.108806, -0.270618, 0.224819, 0.204963, -0.071339, 0.424383, 0.108564, 0.326525, 0.249448, -0.02741, 0.101142, 0.263901, 0.464657, 0.122327, 0.363379, 0.401329, 0.260924, 0.064431, -0.102664], [0.105228, -0.012924, 0.224251, -0.097901, 0.186161, 0.196973, 0.122618, -0.180324, 0.219679, -0.280878, 0.069686, -0.092998, 0.372003, -0.025669, 0.110706, -0.010161, -0.021222, 0.123353, 0.078679, 0.561823, 0.17718, -0.108253, 0.290682, 0.114347, -0.018969, 0.256182, 0.183516, -0.177917, 0.143175, -0.062152, 0.046172, 0.263451, -0.049828, 0.138343, 0.07729, 0.273449, 0.025977, 0.005556, -0.217739, -0.026418, -0.035639, 0.444047, -0.012432, -0.149072, -0.074417, 0.031939, -0.133091, 0.035036, 0.005759, -0.085341, 0.071479, -0.087191, 0.209871, 0.048493, 0.029265, 0.133515, -0.0298, -0.066755, -0.016127, -0.126166, -0.1044, 0.189257, 0.151535, 0.099117], [0.162161, 0.002533, 0.003781, -0.063185, 0.027096, -0.081763, 0.097169, 0.125148, -0.021799, -0.201435, -0.073243, -0.130291, 0.141239, -0.050626, 0.145306, -0.048069, -0.030963, -0.254834, -0.074024, 0.032658, -0.038288, 0.065778, 0.143844, 0.073472, 0.119815, -0.131727, -0.10165, -0.039992, 0.076095, -0.1879, -0.156467, -0.002546, 0.052676, 0.055223, 0.065456, 0.058021, 0.112442, 0.031979, 0.07414, 0.167666, 0.001447, 0.118668, -0.107966, -0.244178, -0.003891, -0.074102, -0.045464, -0.14992, -0.107379, -0.210796, -0.011436, -0.130364, -0.049913, 0.059099, -0.046287, -0.042401, 0.014273, 0.092466, -0.016983, -0.083721, -0.03799, -0.016866, -0.140806, -0.02207], [-0.103317, 0.056215, 0.106459, -0.129625, -0.064302, -0.089695, -0.033666, 0.132466, 0.12051, 0.056025, 0.082167, -0.145362, -0.207397, 0.153384, -0.110789, 0.180089, 0.057759, 0.020944, -0.093023, 0.036956, -0.206976, -0.017557, 0.189387, 0.120403, -0.007333, -0.191656, -0.041439, -0.168413, -0.079422, -0.07236, 0.042516, 0.057833, -0.164694, -0.116261, 0.170027, 0.070642, -0.011325, 0.255688, 0.003095, 0.097002, 0.016217, 0.025748, 0.101251, -0.190159, 0.155729, -0.220371, -0.033006, -0.104098, -0.134393, -0.118391, -0.091564, -0.045715, -0.183246, -0.011587, -0.024086, -0.086045, 0.087118, -0.014274, -0.177113, -0.066427, 0.058445, 0.150257, 0.063237, -0.10313], [0.11771, -0.0659, -0.189773, -0.201971, 0.2446, -0.122361, -0.049762, 0.013092, 0.110357, -0.200802, 0.083395, 0.050224, 0.032418, -0.096061, 0.213058, 0.115767, -0.051556, -0.030687, 0.02418, 0.017104, 0.193306, 0.093608, 0.017031, 0.049987, 0.053073, -0.051796, -0.040933, -0.07945, 0.115731, -0.001079, -0.179077, -0.022138, 0.071783, 0.21734, 0.070311, -0.247946, 0.018801, -0.000899, -0.346705, -0.044854, -0.002415, 0.040512, -0.228438, 0.039225, 0.138261, -0.052559, -0.075933, -0.385849, 0.087704, 0.018116, 0.032772, -0.102325, 0.040029, 0.027302, -0.043682, -0.143665, 0.106135, -0.145385, 0.117507, -0.440924, -0.017661, -0.21067, 0.099562, -0.027271], [0.06811, 0.040619, 0.103983, 0.004619, 0.443538, 0.035256, 0.212055, 0.221363, 0.201157, -0.213745, 0.015594, 0.075868, 0.023426, -0.064672, -0.265633, -0.185948, 0.230362, -0.192169, 0.245962, 0.002496, 0.34524, 0.355294, -0.033513, 0.1795, 0.132793, 0.535848, 0.165175, -0.163072, 0.156016, -0.174324, -0.050823, 0.126609, 0.004773, -0.091887, -0.12043, 0.177546, 0.078911, -0.066685, 0.100791, 0.206944, 0.319667, 0.080612, 0.187317, 0.028464, -0.230491, 0.045999, -0.436473, -0.036866, 0.188817, 0.14072, -0.099324, 0.120926, 0.112417, 0.047182, -0.02196, -0.235987, 0.088751, -0.407827, -0.017003, 0.264676, -0.130671, 0.044215, 0.206119, 0.010675], [0.043807, -0.063492, 0.060354, 0.034698, 0.016428, -0.070578, -0.041481, -0.146409, 0.222897, -0.119461, -0.09358, 0.073785, -0.22766, 0.097561, 0.032499, 0.319871, 0.067817, 0.022712, 0.111801, 0.230771, -0.141842, 0.091049, 0.014697, 0.117534, 0.054884, -0.004531, -0.021075, 0.062995, -0.153148, -0.078167, 0.135369, 0.06144, -0.016018, 0.172685, 0.149906, -0.017031, 0.126571, 0.353775, -0.10703, 0.407506, -0.025873, 0.106104, 0.045384, 0.007112, 0.100265, 0.007845, 0.196174, -0.040699, 0.032918, -0.031927, -0.089695, -0.163608, -0.090224, 0.098988, 0.167495, 0.057422, -0.093212, -0.167736, 0.122064, 0.098913, -0.018721, -0.028307, -0.124817, -0.148391], [0.083869, -0.074038, 0.176473, -0.122279, -0.007121, 0.01011, -0.071181, -0.00196, -0.016428, 0.020858, -0.188509, 0.12208, 0.125714, 0.114563, -0.000938, -0.365209, -0.206051, -0.046603, 0.080884, 0.065049, 0.017439, 0.117329, -0.081721, -0.094971, 0.238019, 0.238974, 0.008146, 0.036363, 0.316811, -0.001062, 0.021606, -0.164594, 0.148426, -0.091061, 0.129433, -0.144616, 0.036542, 0.101259, -0.23997, -0.1654, 0.057668, -0.235137, -0.327931, 0.036098, -0.083149, 0.017503, 0.002227, 0.033859, 0.04591, 0.121752, -0.17901, -0.061432, -0.030603, -0.078577, 0.085134, -0.036509, 0.139355, 0.123672, -0.124041, -0.083464, -0.076159, -0.079092, 0.094656, -0.022104], [0.067235, 0.106826, 0.19335, 0.171716, -0.157925, 0.385796, 0.197625, -0.258055, -0.229121, -0.293456, 0.1643, -0.254815, -0.141053, -0.155199, 0.351038, -0.057534, -0.16471, 0.081763, -0.279402, 0.119204, 0.007496, -0.237384, -0.135923, -0.175542, -0.418007, -0.392827, -0.021009, 0.147304, -0.153629, -0.101459, -0.121432, -0.247687, 0.391882, -0.105209, -0.199277, -0.170048, 0.290614, 0.030765, 0.177424, -0.351937, -0.5012, -0.246567, -0.305968, -0.40778, 0.178324, 0.302621, 0.518013, -0.017352, -0.086413, -0.283174, 0.006278, 0.037882, 0.387468, -0.238261, 0.005938, -0.035683, 0.138407, 0.34085, -0.314755, -0.058811, -0.144146, 0.063563, 0.378999, -0.146911], [0.004012, 0.10881, -0.04284, 0.085481, -0.039375, -0.083939, 0.178546, 0.068182, 0.03529, -0.078305, -0.001744, -0.045782, 0.139161, 0.150015, 0.168769, -0.058792, -0.040051, -0.189691, 0.086976, 0.10697, -0.047192, -0.054114, -0.134538, 0.036642, -0.044709, 0.077549, 0.088181, 0.0179, -0.037358, 0.009816, -0.069304, -0.219425, -0.07475, 0.134274, 0.217271, 0.110121, 0.023565, -0.184892, 0.27349, 0.100995, 0.216775, -0.15743, 0.145415, 0.041843, -0.079112, 0.15994, 0.150747, -0.111146, 0.094844, 0.145904, -0.156693, 0.045599, 0.067336, 0.10336, 0.132317, -0.087519, 0.092635, -0.004083, 0.143434, 0.134159, -0.070495, 0.06442, -0.026327, 0.095659], [0.321113, 0.001435, -0.207627, 0.071162, 0.199374, -0.103016, -0.329694, 0.171297, 0.306965, 0.016617, -0.047791, 0.17916, -0.146411, -0.098494, 0.120623, 0.073695, 0.104432, 0.195044, 0.380711, -0.073779, 0.150225, 0.010009, 0.094204, -0.005007, 0.238635, 0.529582, -0.029422, -0.216867, 0.590081, 0.051053, 0.220901, 0.149362, 0.269325, 0.211972, -0.021898, -0.085444, -0.206766, 0.277314, 0.091434, 0.154656, 0.430738, 0.07076, -0.177099, 0.329375, 0.029777, 0.038335, -4.4e-05, 0.034201, 0.130725, 0.301058, -0.051004, -0.1669, -0.280965, 0.079169, -0.232249, -0.161716, 0.008856, -0.289309, 0.190732, -0.249825, 0.206336, 0.070114, -0.375423, 0.002227], [-0.105117, -0.248577, -0.174382, -0.241689, 0.230167, -0.217242, -0.209394, -0.153915, -0.075247, -0.329718, 0.154824, -0.116595, 0.027265, 0.056259, 0.064691, -0.068232, -0.096351, -0.108866, 0.00442, -0.107895, 0.214711, 0.108527, 0.1466, 0.011285, -0.064437, 0.004795, -0.031102, 0.049553, -0.117449, -0.19817, -0.086916, 0.028319, 0.093418, -0.103841, -0.192988, -0.076979, 0.045354, 0.053534, 0.143442, -0.016702, 0.086832, -0.154873, -0.150956, -0.001527, -0.227767, -0.013823, -0.333237, -0.257071, -0.054242, -0.103115, -0.040414, -0.104404, 0.106025, -0.146252, -0.045363, -0.173002, -0.096852, -0.340715, -0.198936, -0.114077, -0.229486, 0.012771, -0.1548, 0.04139], [-0.098362, 0.081934, 0.142042, -0.135106, 0.086759, 0.162454, 1.232902, -0.213366, 0.242134, -0.013017, 0.394433, -0.05821, 0.05081, 0.515951, -0.018289, 0.324352, -0.034543, 0.023565, 0.149488, 0.351778, 0.176303, -0.165585, 0.139486, 0.113789, -0.113927, 0.117842, 0.465756, -0.176411, -0.104679, -0.328373, 0.103187, 0.142685, 0.302721, 0.155199, 0.126332, 0.227281, -0.011647, 0.207097, -0.070374, 0.00621, 0.2287, 0.533876, 0.227762, -0.013789, -0.057451, 0.063624, 0.008782, 0.117926, 0.15897, 0.214592, -0.04034, 0.440316, 0.217036, -0.049844, 0.079222, -0.025789, 0.026054, 0.067344, -0.009829, 0.333858, 0.397229, 0.423618, 0.128633, -0.12964]], [-0.129332, -0.020739, -0.209351, -0.069153, 0.138222, -0.279769, -0.293473, -0.112514, -0.094445, -0.286198, 0.158469, 0.061514, -0.008806, -0.221169, -0.182767, 0.091654, -0.079747, 0.002825, -0.255774, -0.179302, -0.074016, 0.185188, -0.055275, -0.011643, -0.055851, 0.015162, -0.185279, 0.015187, 0.10739, -0.006286, -0.137005, 0.181481, -0.004922, 0.022133, 0.092214, -0.140877, 0.044049, 0.047839, -0.241068, -0.18565, -0.160963, -0.10711, -0.186229, -0.180219, -0.225757, 0.010662, -0.079321, -0.192449, 0.043566, -0.191857, -0.16653, -0.105351, -0.041225, -0.165096, -0.109161, 0.05319, -0.020612, -0.148544, 0.004544, -0.063894, -0.225297, -0.222966, -0.164671, 0.137897]),
    ([[-0.03159, -0.127749, 0.118366, -0.017261, 0.052195, 0.05788, 0.054399, 0.056934, 0.076145, 0.137598, -0.056705, 0.157647, 0.108447, 0.102338, 0.008065, 0.238826, 0.10537, -0.235089, -0.082198, 0.002646, 0.033343, -0.002262, -0.074897, -0.060274, -0.04348, 0.006309, 0.23886, -0.326626, 0.01458, 0.060286, 0.172882, -0.058712], [-0.166416, -0.024963, 0.046133, -0.05576, 0.121029, -0.051918, -0.092171, -0.175703, -0.10862, 0.174793, -0.087081, 0.028041, -0.082756, -0.027184, 0.113634, 0.061186, 0.045809, -0.154692, -0.173721, 0.133272, 0.177548, 0.057165, 0.071783, -0.134705, -0.14441, 0.020278, 0.026177, 0.168146, -0.25119, -0.014986, -0.025125, 0.085046], [0.055589, 0.020902, 0.142604, -0.334917, 0.008062, 0.132253, -0.159418, -0.164676, 0.16244, 0.167503, -0.012795, -0.168858, 0.059658, -0.00051, -0.138226, 0.27535, 0.097704, -0.130081, -0.282214, -0.006849, 0.11517, -0.008535, 0.014506, -0.052017, 0.081981, 0.042994, -0.090388, -0.033968, 0.064282, 0.05354, -0.064279, -0.086778], [0.009008, 0.021766, 0.066068, 0.139422, 0.068807, -0.023962, -0.000341, -0.006007, -0.003594, -0.176315, -0.130143, 0.079327, 0.114743, 0.173004, 0.093382, -0.093604, 0.121026, 0.170697, 0.207544, -0.134801, -0.035544, -0.157994, 0.081181, -0.041879, 0.141579, -0.013297, 0.153773, 0.091504, 0.034865, -0.047133, 0.127685, -0.020383], [0.116669, 0.134005, 0.100124, -0.001236, -0.021176, 0.088312, 0.064614, 0.001775, 0.036669, -0.209587, -0.059392, 0.11463, 0.112446, 0.101911, 0.117489, 0.01491, -0.101613, 0.0838, -0.002335, 0.029644, -0.111347, -0.117449, -0.062873, 0.01623, 0.161823, -0.084017, 0.02178, 0.022976, 0.138362, -0.089383, 0.043743, -0.072711], [-0.176464, -0.114751, 0.05996, -0.010973, 0.023035, 0.125982, -0.081652, 0.076035, -0.070882, 0.017676, -0.200107, -0.10418, -0.370525, -0.024614, -0.214837, -0.150614, 0.193544, 0.076012, 0.036519, 0.137201, 0.04687, 0.093055, 0.134996, 0.039879, -0.203289, -0.040359, -0.097077, 0.082868, 0.00982, -0.019672, 0.058046, 0.093599], [-0.027387, -0.014611, 0.262048, -0.062389, -0.449126, 0.100599, 0.077472, -0.062022, 0.096899, 0.611093, -0.137777, 0.174455, -0.015236, 0.292435, 0.107119, 0.030953, 0.33699, -0.240136, 0.05956, 0.102194, 0.283788, 0.245366, 0.755556, -0.015203, 0.034318, 0.437551, 0.230086, -0.685341, 0.12344, 0.023306, 0.451036, -0.418351], [-0.073092, 0.158532, -0.144332, 0.017118, 0.003178, -0.147952, 0.077639, 0.111776, 0.071842, -0.307608, -0.379702, 0.038297, 0.072509, -0.088931, -0.03472, 0.081026, -0.153798, 0.026368, 0.053714, -0.142261, -0.033739, -0.030424, -0.060824, 0.112484, 0.107714, -0.009821, 0.145142, 0.117044, 0.125778, 0.029555, -0.023084, 0.01372], [0.148319, 0.091868, 0.037291, -0.012926, 0.24951, 0.108586, 0.069039, -0.024723, -0.091427, -0.029205, -0.123878, 0.154928, 0.105565, 0.005164, 0.037273, 0.023677, 0.068083, -0.037405, -0.112854, 0.066898, -0.061837, 0.148747, 0.246278, -0.225392, 0.013478, 0.094658, 0.131738, 0.177412, 0.047542, 0.075496, 0.128269, 0.202505], [0.06532, 0.043921, -0.031805, 0.153558, -0.104262, 0.06074, -0.031085, 0.008808, -0.159508, -0.074668, 0.043949, -0.212519, -0.067847, -0.124301, -0.086599, -0.149244, -0.139819, 0.131707, 0.033627, -0.111894, -0.043705, -0.078292, -0.449199, -0.036644, -0.059824, -0.254605, 0.000328, 0.045413, -0.035265, -0.102083, -0.295999, 0.248399], [-0.15178, -0.139652, -0.009886, -0.005991, -0.133575, 0.026113, 0.054876, 0.063831, -0.165217, -0.00484, -0.020895, 0.007063, -0.066476, 0.147509, 0.122432, -0.076641, 0.043969, -0.111181, -0.053051, 0.046217, 0.11102, 0.032498, 0.038375, 0.052965, -0.255596, -0.065301, -0.575461, 0.062061, 0.09664, 0.15816, 0.097088, -0.024041], [0.056883, 0.166228, -0.149778, -0.289752, 0.105353, -0.18307, 0.085188, 0.061502, 0.069523, -0.061583, 0.026403, -0.110277, 0.121537, -0.062634, 0.069905, 0.093603, 0.055549, -0.025049, 0.070122, -0.030864, -0.025158, -0.409915, -0.113775, -0.035649, 0.118676, -0.08264, -0.017363, -0.062573, -0.035155, -0.118632, -0.093658, 0.077997], [-0.074203, -0.086795, 0.017364, 0.009041, 0.006104, 0.147543, -0.072451, -0.110747, 0.063891, 0.008342, -0.20856, -0.124639, -0.222665, -0.05409, -0.118619, -0.178965, 0.073104, 0.015201, -0.177184, 0.000822, 0.119616, 0.135976, 0.179434, -0.065099, -0.292762, 0.016704, -0.76021, -0.017374, 0.166831, 0.104237, 0.099307, -0.015696], [0.192268, 0.001935, 0.085261, -0.332147, 0.18493, 0.091929, -0.073933, -0.113407, -0.101342, 0.211417, -0.476644, -0.256387, -0.455352, 0.096207, -0.334067, 0.15426, 0.112218, 0.081215, -0.164911, -0.061949, -0.009325, 0.161426, 0.126833, -0.28048, 0.041097, 0.22634, 0.173689, 0.036517, -0.228243, -0.342275, 0.13535, -0.039104], [-0.100976, -0.221974, 0.100441, 0.019819, 0.120933, -0.02445, 0.074054, -0.139009, 0.021789, -0.137096, -0.14078, -0.046656, 0.026184, 0.105255, -0.062789, -0.090867, 0.144597, -0.235443, 0.002947, 0.103158, 0.033589, -0.066134, -0.031906, 0.196618, -0.013296, 0.068358, -0.505449, -0.081252, 0.017275, -0.129881, -0.063784, -0.07321], [0.112014, 0.056082, 0.082399, 0.004002, -0.267204, -0.098598, -0.012936, 0.0585, 0.038082, -0.06731, 0.11752, 0.0322, 0.143344, -0.032441, -0.022527, -0.095373, 0.008402, -0.007486, 0.042018, 0.117186, -0.096334, -0.26814, -0.383116, -0.082065, 0.036241, -0.345041, -0.094885, -0.238203, -0.106523, 0.047949, -0.189743, -0.058595], [-0.014748, -0.033041, -0.099032, -0.081299, -0.033876, 0.029738, -0.051131, -0.02063, -0.015419, 0.216085, -0.048325, -0.106811, -0.132726, -0.051988, -0.026327, 0.179661, 0.117948, -0.244598, -0.35741, 0.097964, 0.165937, -0.104602, 0.08228, -0.007164, -0.108677, 0.03208, -0.164565, -0.045815, 0.063583, 0.18149, 0.016247, 0.001384], [-0.223327, 0.075915, 0.129833, 0.209037, -0.137544, 0.142454, -0.056213, 0.081581, 0.024801, 0.044009, 0.059818, -0.207118, -0.244302, -0.085528, 0.162314, -0.22472, 0.006101, -0.098435, -0.012807, 0.025585, 0.1748, 0.126702, 0.134939, -0.016929, -0.082837, 0.014356, -0.254597, 0.150827, 0.092307, 0.143136, 0.02361, -0.127144], [0.102633, 0.016624, 0.104039, 0.11469, 0.08334, -0.143884, 0.023085, 0.146949, 0.045545, 0.083477, -0.403595, 0.003744, -0.028319, -0.086331, 0.031905, 0.090444, 0.010911, 0.220336, 0.165818, -0.065786, -0.077268, -0.293012, -0.081854, -0.045493, 0.155401, -0.178601, 0.211323, -0.269862, 0.088123, 0.197654, -0.07512, -0.443939], [-0.196621, -0.119458, 0.060846, 0.096816, -0.30278, -0.022922, 0.027628, 0.070308, -0.07552, -0.137465, 0.208241, 0.098591, -0.289258, -0.084352, -0.098084, -0.242733, 0.020939, -0.134687, -0.145317, 0.185491, 0.101807, 0.17862, 0.119595, 0.263182, -0.045959, 0.058202, -0.313998, 0.091278, -0.025073, 0.078224, 0.001269, -0.07982], [-0.053459, -0.028695, -0.015263, -0.059798, -0.028118, 0.031081, -0.128616, 0.019334, 0.050527, 0.12667, -0.130764, 0.041452, -0.510984, 0.182615, -0.225721, -0.09586, -0.019713, -0.1155, -0.173596, 0.032565, 0.182778, -0.136897, 0.097902, 0.006105, -0.37707, -0.035549, -0.513053, 0.099883, 0.06451, 0.20595, 0.118018, -0.017436], [-0.013252, 0.047676, -0.005916, -0.067722, 0.040599, 0.006288, 0.050387, -0.040421, 0.137447, -0.348261, 0.128523, -0.025502, 0.155539, -0.216904, 0.164406, -0.093202, -0.056137, -0.138825, 0.129257, 0.105075, 0.048152, -0.00773, -0.245954, 0.127595, 0.137249, -0.158786, -0.069522, -0.134674, -0.061741, -0.010674, -0.152935, -0.207532], [0.039001, 0.106754, 0.159341, 0.023812, -0.010627, 0.085924, -0.041464, -0.096694, -0.071311, -0.115134, -0.04933, -0.005861, 0.113405, 0.056559, 0.127405, 0.03516, -0.015933, 0.032529, -0.13462, 0.085735, 0.028433, 0.189467, 0.281254, -0.210799, 0.055156, 0.121944, -0.027397, 0.045793, 0.061167, -0.052213, 0.144862, 0.082327], [0.109286, 0.029405, 0.18963, -0.210363, -0.268465, -0.110859, -0.093304, -0.050097, -0.015053, 0.083215, -0.025412, 0.041126, 0.099968, -0.173829, -0.060117, -0.02891, -0.156163, 0.071289, -0.028835, 0.116337, -0.107843, -0.170919, -0.22508, -0.184122, 0.022196, -0.153314, 0.105466, -0.02329, 0.105896, -0.015498, 0.035596, -0.055921], [-0.041218, 0.097104, 0.025721, -0.182657, 0.301065, -0.052522, 0.129607, 0.073629, 0.150688, -0.047354, 0.021058, -0.131833, -0.033617, 0.040449, 0.114649, 0.189668, -0.085039, 0.116152, 0.188047, 0.057988, -0.146548, -0.108011, -0.053899, -0.099334, 0.075062, -0.030892, -0.067369, 0.148965, 0.169571, 0.035944, -0.010439, -0.11374], [0.205808, 0.046584, -0.07614, -0.21656, -0.147218, 0.015368, -0.026681, -0.03237, 0.070604, 0.185142, 0.138825, 0.181428, 0.147566, -0.062229, 0.174082, 0.274761, 0.126963, 0.09574, 0.058683, 0.040117, 0.168768, 0.108507, 0.041592, -0.26925, 0.25394, -0.018654, 0.468159, -0.108854, 0.208747, 0.007648, -0.019238, -0.050527], [0.00834, 0.025273, 0.053004, 0.062575, 0.081156, 0.073385, -0.05515, 0.033407, 0.082465, 0.084217, 0.094113, 0.0427, -0.010808, 0.109439, 0.083799, -0.112254, 0.123405, 0.123318, -0.087223, 0.116635, 0.039567, -0.051231, 0.051035, 0.145396, -0.1636, 0.088297, -0.086069, 0.132179, 0.083771, 0.033362, -0.056979, 0.063865], [-0.038149, 0.006877, -0.077964, 0.037511, -0.786549, -0.177453, 0.106166, 0.020806, 0.038693, 0.084568, -0.323187, -0.036062, -0.030002, 0.042362, -0.060775, -0.147122, 0.045173, 0.133936, 0.092541, -0.031708, -0.034918, 0.056981, 0.014393, -0.026709, 0.033547, 0.110284, -0.115983, -0.216496, -0.047842, -0.187723, -0.056607, -0.155652], [-0.05201, 0.020883, -0.098152, 0.086622, -0.058749, 0.186828, -0.054295, 0.160478, 0.090694, -0.027508, 0.11955, 0.183355, 0.037845, 0.043464, 0.131988, 0.046632, 0.085751, 0.039044, 0.1486, 0.072168, 0.138992, 0.07732, -0.081909, 0.033432, -0.057032, -0.012397, 0.168049, -0.185492, 0.06697, 0.222099, 0.133139, -0.125982], [-0.018065, 0.192585, -0.09857, 0.175861, 0.140568, 0.126813, 0.080781, -0.025997, 0.036826, -0.200498, 0.113857, -0.061774, 0.107797, -0.022815, -0.055444, -0.181653, -0.149064, -0.087197, 0.047363, -0.199612, -0.159517, -0.006078, -0.030155, 0.14473, 0.029807, 0.03216, -0.266741, 0.17877, 0.177125, -0.102944, 0.152933, 0.198652], [0.006018, 0.044916, 0.023605, 0.160653, 0.175039, 0.015657, 0.015278, -0.019086, -0.005161, 0.059622, -0.04858, -0.091945, -0.076352, 0.118724, -0.116522, 0.149828, 0.125971, 0.082388, 0.10572, -0.071959, 0.108964, 0.108332, 0.137826, 0.099228, -0.080748, 0.085412, -0.008416, -0.013721, 0.005442, -0.095663, -0.010171, 0.172513], [-0.020335, 0.040805, -0.240932, 0.047095, -0.154161, 0.082219, 0.044056, -0.05006, -0.014051, -0.27825, 0.131716, 0.101028, -0.078136, -0.077133, 0.071739, -0.176658, -0.232212, 0.08298, -0.041062, -0.184162, 0.052512, -0.164427, -0.253618, 0.033808, 0.019901, -0.268392, -0.158, -0.018065, 0.066764, 0.078366, -0.121143, 0.218124], [0.302092, 0.137061, 0.024081, -0.024958, 0.041762, 0.079921, 0.07212, 0.012085, 0.047693, 0.21663, -0.608138, -0.00863, 0.416201, 0.0797, 0.15572, 0.156979, -0.061051, -0.028854, -0.076592, -0.03247, 0.119325, -0.07882, 0.013048, -0.22539, 0.129506, 0.223599, 0.233316, -0.024915, -0.014673, -0.140449, 0.054694, -0.077279], [-0.002515, -0.044255, 0.031463, 0.196964, -0.13443, -0.114619, 0.137218, -0.125285, 0.092318, 0.021047, 0.120303, -0.039455, 0.114041, 0.007358, 0.069238, -0.193081, 0.005535, -0.0023, 0.105711, -0.126112, -0.124795, -0.184512, -0.266414, 0.23087, -0.08831, -0.407825, -0.085273, -0.102539, 0.061186, -0.041093, 0.002018, -0.028263], [0.006714, -0.043441, -0.01857, 0.07014, -0.702329, 0.110402, 0.010507, 0.089827, -0.008251, 0.064923, 0.08549, 0.092806, 0.000529, -0.010514, 0.014756, -0.259234, 0.08379, 0.019353, 0.115118, -0.068663, 0.075041, 0.068324, 0.053827, 0.213163, 0.084419, 0.007158, 0.115089, -0.016922, 0.070546, -0.032222, -0.082618, -0.395509], [0.039084, 0.137662, 0.084447, -0.365235, 0.079706, 0.097948, 0.01531, 0.119058, 0.208297, -0.088897, -0.013437, 0.270095, 0.074145, -0.039567, 0.08942, 0.152755, 0.139424, 0.061152, -0.036241, 0.121024, 0.026191, 0.20585, -0.039823, 0.021272, 0.041464, 0.20904, 0.311841, -0.27305, -0.045092, -0.345784, 0.270972, -0.088134], [-0.135623, -0.028022, 0.013907, -0.009087, 0.069617, 0.085386, 0.081636, 0.092956, -0.062726, -0.14062, -0.008664, 0.061371, 0.158028, 0.122779, -0.003821, 0.070514, -0.076234, 0.066824, 0.084651, 0.015983, 0.153116, 0.031915, 0.008928, 0.138283, 0.094998, -0.000558, -0.377484, -0.137553, -0.13715, -0.139399, 0.066839, -0.097601], [0.197427, 0.169949, -0.345513, -0.213665, 0.152653, -0.085489, -0.145933, 0.029736, -0.017502, -0.440407, 0.146379, 0.094858, -0.039541, 0.070421, 0.097501, 0.098701, -0.1169, 0.089907, -0.125663, -0.198741, 0.093282, -0.148558, -0.073862, -0.012505, -0.013318, -0.25233, -0.066443, -0.048698, 0.086355, 0.130057, -0.085113, -0.025298], [-0.003041, 0.019156, -0.083524, 0.188188, 0.020446, -0.117781, -0.025301, 0.024262, -0.049466, 0.06316, -0.005992, -0.287007, -0.301063, 0.034079, -0.221329, -0.256891, 0.061634, 0.095817, 0.043103, 0.096334, 0.027009, -0.189912, -0.04758, -0.014582, -0.144889, -0.02908, -0.127843, 0.158859, -0.067595, 0.080171, 0.100804, 0.181571], [0.129146, 0.192151, -0.08952, 0.085774, -0.155593, -0.040035, -0.004419, 0.02276, 0.035345, -0.037565, 0.126804, -0.01084, -0.009593, -0.109107, 0.161641, -0.090387, 0.08277, 0.004482, -0.001771, -0.250999, -0.071063, -0.062135, 0.00414, -0.036503, 0.202299, 0.176768, 0.024403, 0.025729, 0.134067, 0.144265, 0.308956, 0.107838], [0.130873, 0.051126, -0.105595, 0.020115, 0.031275, -0.149413, 0.119276, 0.073848, -0.063929, -0.00806, 0.094908, 0.091099, 0.143285, -0.080415, 0.034798, 0.046333, -0.041366, 0.092125, 0.143367, 0.089272, 0.024394, -0.162906, -0.236929, -0.154577, -0.040194, -0.249917, 0.05363, -0.117793, -0.014965, 0.041366, -0.137614, -0.191785], [-0.363637, -0.048668, 0.182044, -0.037207, -0.211112, 0.210466, -0.002682, -0.091874, -0.114746, -0.144677, -0.006015, -0.076711, -0.034509, 0.04355, 0.170826, -0.168729, 0.067557, -0.156058, -0.258608, 0.1665, 0.186146, 0.181676, 0.107017, 0.110515, -0.076963, 0.000705, -0.138448, 0.050065, 0.062325, 0.055085, 0.057881, -0.114993], [0.072712, 0.137249, -0.316956, 0.144384, -0.391019, -0.250211, 0.079388, 0.129602, 0.094723, -0.217671, 0.123666, -0.013635, 0.053362, -0.080403, 0.052928, -0.00692, -0.090267, 0.036453, 0.096623, -0.099225, 0.101868, -0.425335, -0.187424, -0.234427, 0.044646, 0.054073, 0.095769, -0.162825, 0.090555, 0.061364, -0.015436, 0.049651], [-0.072135, 0.128929, 0.064391, 0.029023, -0.102644, -0.02694, -0.025115, -0.113266, 0.034013, -0.299147, 0.096113, -0.035893, -0.038014, -0.440775, 0.10347, -0.05632, -0.131354, 0.054255, 0.048475, -0.095328, -0.017114, -0.006433, -0.1538, 0.084165, -0.105271, -0.395055, -0.093531, -0.024845, 0.114186, -0.107312, -0.163983, -0.088332], [-0.053133, -0.116849, 0.191426, -0.093347, -0.05287, 0.034835, -0.191049, -0.173613, 0.136836, 0.346793, -0.072789, 0.00682, 0.040257, 0.008371, -0.135575, 0.181751, -0.068075, -0.158819, -0.056779, 0.246536, 0.091676, 0.204047, 0.075436, -0.033377, -0.105167, 0.057579, 0.082555, -0.136692, 0.110931, 0.144208, -0.119279, 0.022039], [0.015931, -0.10074, -0.03515, -0.031033, 0.189859, 0.120934, -0.148435, -0.100858, -0.127709, 0.017654, -0.044747, 0.07717, -0.064428, 0.142913, 0.07735, 0.103952, 0.097353, -0.075445, -0.164288, -0.11833, 0.032267, -0.300273, 0.022326, -0.005867, -0.171915, 0.099332, -0.279605, 0.067849, -0.078212, -0.040981, 0.014988, 0.149101], [-0.375663, -0.03637, 0.047514, -0.051828, -0.116521, 0.039727, -0.10208, -0.067974, -0.090035, 0.14516, -0.150739, -0.27614, -0.446701, 0.035818, -0.119041, -0.473764, -0.028365, -0.086287, -0.047465, -0.040513, 0.149972, 0.030206, -0.226031, 0.144509, -0.022556, 0.017361, -0.467804, -0.00686, -0.094807, 0.016381, -0.136806, 0.043457], [0.095797, 0.14453, 0.024987, -0.004906, -0.134652, -0.030954, 0.090866, -0.01678, 0.100551, 0.225481, -0.131092, 0.012672, 0.167055, 0.029667, 0.11864, -0.09746, 0.118171, 0.047568, 0.089044, 0.112777, 0.146952, 0.118066, 0.262817, -0.092406, 0.043461, 0.104465, 0.309422, 0.058991, 0.166856, -0.201555, 0.069475, 0.078509], [0.102546, 0.033791, 0.094429, -0.097326, -0.205119, -0.075377, 0.07249, 0.078406, 0.045497, 0.108868, -0.094517, 0.082285, 0.009152, 0.106751, 0.129448, 0.134897, 0.091816, -0.080931, -0.110591, 0.072523, -0.028328, 0.040102, 0.056508, 0.120344, 0.083166, 0.187959, 0.131882, -0.135149, -0.050467, 0.078652, 0.062523, -0.008967], [0.102942, 0.081211, -0.009979, -0.11336, 0.051013, -0.027868, -0.025283, 0.098099, -0.044435, -0.105124, 0.165214, -0.140152, 0.039301, -0.040904, 0.034704, 0.074053, -0.331469, -0.018542, -0.076273, -0.126104, 0.040137, 0.051367, -0.301327, 0.048807, 0.169032, -0.119403, -0.002851, 0.095184, 0.103846, 0.015025, 0.050263, -0.002876], [-0.045286, 0.087059, 0.131448, -0.053697, -0.040425, -0.068403, -0.110451, -0.000604, -0.152702, -0.204043, -0.185551, 0.030709, 0.177103, -0.042659, -0.084218, 0.042459, 0.110035, 0.03122, 0.083569, -0.075969, 0.138796, 0.129381, -0.066448, 0.089837, -0.052563, 0.148318, -0.287468, -0.000165, -0.08381, -0.093319, 0.086533, -0.06368], [0.275407, 0.107834, -0.017926, 0.081877, -0.02035, -0.007444, 0.028184, -0.110505, -0.008674, 0.082245, -0.268246, -0.006916, 0.177704, 0.169856, 0.009865, -0.29362, 0.142076, -0.020487, -0.018598, 0.014413, 0.041846, -0.000651, 0.188072, -0.15672, -0.032442, 0.077965, 0.378518, 0.124413, -0.113539, -0.063617, 0.126393, 0.045235], [-0.041494, -0.079313, 0.145453, 0.03402, -0.329921, -0.016772, -0.121748, -0.116276, -0.083411, 0.365532, -0.28907, 0.216177, -0.03268, 0.195041, 0.120491, -0.194033, 0.086086, -0.039307, -0.09665, 0.217676, 0.208807, 0.095767, 0.242275, 0.054608, 0.02983, 0.184053, 0.038353, -0.06343, -0.081433, -0.31751, 0.009945, -0.193138], [-0.078435, 0.15338, 0.080313, 0.012735, -0.223046, -0.040179, -0.022842, 0.070255, 0.050625, 0.222252, -0.050712, -0.028749, 0.06097, -0.154647, 0.132666, -0.079008, -0.176528, 0.026587, 0.127758, -0.025294, -0.118264, 0.16977, 0.033862, -0.155749, 0.086706, 0.039905, 0.129694, -0.008409, 0.065078, -0.060962, -0.03721, -0.089721], [0.069722, -0.051497, 0.177711, -0.100908, -0.451223, -0.030836, 0.116602, -0.053962, 0.112106, 0.196365, -0.139855, -0.053114, 0.039552, -0.107601, 0.064936, -0.09491, -0.092187, 0.010085, -0.004074, 0.078677, -0.033247, 0.22596, 0.013393, -0.145455, 0.105428, -0.12332, 0.006667, -0.154362, 0.018989, -0.083722, -0.311431, -0.388483], [-0.031904, 0.135994, -0.194817, -0.031369, 0.009106, -0.034213, -0.036977, 0.135987, 0.159189, -0.131005, 0.173732, 0.024097, 0.081953, 0.087621, 0.063902, 0.013283, -0.002372, 0.111651, -0.090427, -0.062027, 0.127587, -0.232131, -0.353896, -0.074995, 0.019021, -0.069276, 0.37377, 0.039644, 0.063198, 0.167381, -0.011924, 0.070357], [-0.012065, 0.062887, 0.067949, -0.101572, 0.015701, 0.078475, 0.063255, 0.021888, -0.018028, 0.121313, -0.390221, 0.066962, -0.095254, 0.103914, -0.00761, -0.028952, 0.097235, 0.100862, -0.058268, 0.047601, 0.002296, 0.026514, -0.005853, 0.165587, -0.089668, 0.055462, -0.692464, -0.110511, 0.038042, -0.103704, 0.057001, 0.036513], [-0.067963, -0.046922, -0.024892, -0.156476, 0.183515, 0.112529, -0.18297, -0.083508, 0.059043, -0.002679, -0.159168, -0.205276, -0.160746, 0.051698, -0.083202, 0.146005, 0.082484, -0.068829, -0.254449, 0.027564, -0.040587, 0.192886, 0.136178, -0.160874, -0.182847, -0.01579, -0.115231, -0.057425, -0.314296, -0.225126, -0.086654, -0.08287], [0.152153, 0.048259, 0.027181, 0.13, 0.089491, 0.114257, -0.006377, -0.028405, -0.066614, 0.012561, -0.019957, 0.132189, -0.068444, -0.096683, 0.006925, -0.043569, 0.093679, 0.114615, -0.168653, 0.0769, -0.129309, 0.216124, 0.098468, -0.267029, 0.081254, -0.009115, -0.068217, 0.086713, -0.011324, 0.106408, 0.19909, 0.257105], [0.114512, 0.128921, 0.014345, -0.120962, -0.532193, 0.036834, -0.015536, -0.0461, 0.039535, 0.272194, 0.054124, 0.095373, 0.116354, 0.067547, 0.141037, 0.032742, 0.001106, 0.118938, 0.056755, -0.023093, -0.032092, 0.084294, 0.311793, 0.02825, 0.151803, 0.271642, 0.188155, -0.229935, 0.087189, -0.040166, 0.210093, -0.11833], [0.373219, -0.001867, -0.047787, 0.151554, 0.193057, 0.042113, 0.071991, -0.010933, -0.102712, 0.201175, -0.103909, -0.121828, 0.125845, 0.003983, 0.108681, -0.056175, 0.157118, 0.190608, 0.233534, 0.142625, 0.021814, 0.034144, 0.198654, -0.243207, 0.070922, -0.005735, 0.292267, 0.119132, -0.062043, -0.363081, 0.110712, 0.161578], [-0.014133, -0.164699, 0.0527, -0.012725, 0.112599, 0.004041, -0.11121, 0.033114, -0.033348, 0.118191, -0.212356, -0.008657, -0.037617, 0.099181, 0.050956, -0.139398, 0.051568, 0.053188, -0.03526, 0.094898, 0.132388, 0.043599, -0.051141, -0.000986, -0.083461, 0.038981, -0.234219, 0.069459, 0.043308, -0.050787, -0.016836, -0.010444], [0.096923, 0.084609, 0.151011, 0.268405, -0.19646, 0.239979, 0.049628, 0.024591, -0.014704, 0.061106, -0.26187, -0.160166, -0.033988, 0.0341, -0.052796, -0.315854, -0.011006, 0.055042, 0.071789, 0.120587, -0.025508, 0.274578, 0.256877, 0.008995, 0.021888, 0.055692, -0.073277, -0.18626, 0.044866, -0.221144, 0.054484, -0.028667], [-0.099432, 0.124437, -0.052104, 0.000694, -0.271749, -0.006993, -0.087183, -0.082622, 0.122905, 0.017451, 0.074356, 0.02879, -0.117275, -0.139633, -0.025672, 0.08837, -0.042751, -0.095412, 0.066475, -0.138744, 0.041998, 0.028587, -0.592245, 0.037039, 0.06762, -0.094691, -0.066076, 0.053742, -0.011514, -0.024373, -0.150252, -0.038766]], [-0.234387, 0.085519, -0.156411, 0.028373, -0.305486, 0.072049, 0.08707, 0.043132, 0.181615, -0.284501, 0.023807, -0.100428, -0.140678, 0.023729, 0.005021, -0.189595, 0.147266, -0.167986, -0.08336, -0.133335, 0.079904, -0.196433, -0.061823, 0.187612, 0.064104, 0.119239, -0.184025, -0.162332, -0.009122, 0.104304, 0.136532, -0.130058]),
    ([[0.165096], [0.142139], [-0.102466], [0.074436], [0.385396], [-0.166237], [0.064366], [0.0193], [0.082165], [-0.28714], [-0.334468], [0.177541], [0.157014], [-0.167912], [0.169009], [0.392561], [-0.148346], [0.228366], [0.129908], [-0.085974], [-0.177952], [-0.217341], [-0.153376], [0.2742], [0.217958], [-0.228435], [0.45331], [0.332926], [0.052194], [0.256642], [-0.197964], [0.180744]], [-0.093088]),
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
