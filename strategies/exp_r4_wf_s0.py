"""exp_r4_auto: mm_r3_gru3 with the learned tables (ridge,mlp,mq,slow,seq) refitted by research/retrain.py on round4p75.csv + mirror (first 75 %).

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
    # rfit 19 (auto: h200 on round4p75 + mirror, first 75 %, research/retrain.py); params rfit=19
    (-0.0, (0.50233, -1.9759984, 4.7772194, -2.6840441, 1.6987565, -0.4531857, 0.007839, 0.0315912, 0.0003458, -0.0, -0.1546411, 0.1785022, 0.0448765, -0.0448765, -0.0454307, -0.0054872, 0.1748529, -0.3495935, 0.0004214, 0.0003719, -4.6e-05, -0.0)),
)
# exp_r3_mlp: 22-32-16-1 ReLU MLP on the same 22 features, target = 200 ms (robust mid at t+5ms+200ms - mid now), fitted on the
# first 75 % of round3 + mirror (research/diag/mlp_r3.py); holdout corr 0.49 vs 0.463 for the ridge. Inputs clipped at +-8 sd.
MLP_MEAN = [-0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.235948, 0.0, 0.0, 5.908366, 5.908316, 0.0, 7e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009057]
MLP_SCALE = [1.133423, 0.999352, 0.952236, 0.8477, 0.755997, 2.569228, 20.906504, 0.789135, 18.420658, 0.707402, 0.649149, 0.542064, 1.664617, 1.664603, 0.999991, 4.403231, 1.165521, 0.303697, 133.782867, 335.202209, 1064.944946, 1.340472]
MLP_LAYERS = [
    ([[0.038653, -0.075661, -0.265962, -0.03258, -0.064205, -0.186241, -0.069602, -0.007402, 0.137598, -0.12091, 0.03118, -0.093596, 0.149921, -0.115085, 0.126047, -0.050412, -0.185311, -0.2962, 0.244865, 0.13018, -0.326857, -0.015429, 0.12474, 0.008386, 0.008227, -0.246412, 0.044911, 0.047034, 0.56665, -0.009656, 0.090828, 0.028933], [0.177976, 0.279584, -0.190195, -0.009627, -0.112359, 0.019943, 0.06425, -0.275086, 0.050218, -0.016135, 0.117941, 0.011577, -0.06302, -0.02983, 0.014319, -0.149567, 0.117363, -0.122707, -0.079047, 0.319767, -0.371873, -0.244333, 0.229106, 0.0246, -0.042022, -0.101078, -0.104078, 0.064627, 0.159709, 0.157348, 0.031805, -0.02742], [-0.137783, 0.136851, 0.089348, 0.004511, 0.153528, -0.227239, 0.211109, 0.060702, 0.252596, -0.003527, -0.016039, 0.171182, -0.054546, -0.145344, -0.158682, 0.190671, 0.166341, 0.237104, -0.185386, 0.118053, -0.187856, 0.054666, 0.183461, -0.082928, 0.226351, 0.339065, 0.008768, -0.210446, 0.173641, -0.091111, 0.149535, 0.061964], [-0.146977, 0.25799, 0.220041, -0.014434, 0.222677, -0.038706, -0.049392, -0.122365, -0.183896, 0.173425, -0.004771, -0.096741, 0.117338, -0.006248, -0.184248, 0.03219, 0.094053, 0.308214, 0.087155, -0.128427, 0.215667, 0.336093, 0.065154, 0.210528, 0.223251, -0.103582, 0.311409, 0.11254, -0.221242, -0.112353, -0.417311, 0.091435], [-0.517551, 0.041589, -0.311955, -0.427326, 0.35373, 0.085436, 0.090936, -0.067544, -0.145463, -0.281878, -0.331177, 0.261592, 0.207329, -0.116606, 0.089267, -0.108394, -0.103342, 0.538417, 0.252672, 0.068786, -0.121962, 0.212146, 0.071305, 0.067861, -0.074239, -0.193066, -0.223384, 0.50242, -0.349674, 0.21639, -0.058561, -0.00402], [0.045904, -0.003556, -0.081442, -0.264906, 0.055457, -0.244439, 0.095168, -0.171064, -0.033339, -0.042055, 0.072655, -0.031247, 0.139607, -0.029811, 0.014061, -0.031687, 0.171624, 0.242483, -0.100557, 0.18276, 0.001679, -0.226344, -0.048276, 0.032627, -0.135747, 0.078989, -0.275558, 0.172771, 0.036708, -0.030797, 0.123107, -0.224563], [-0.223679, -0.071896, 0.238229, 0.050835, 0.145163, -0.122491, 0.054903, -0.112573, 0.168083, -0.254901, -0.347914, 0.351914, -0.153833, -0.013107, -0.019793, 0.417342, 0.148413, 0.220619, -0.270995, 0.05747, -0.290477, -0.018525, 0.056193, 0.181425, -0.254247, 0.296257, 0.228083, -0.192852, 0.099033, 0.313354, 0.249056, 0.484004], [-0.065376, 0.102449, 0.386553, 0.078518, 0.18188, 0.06032, 0.033526, 0.062277, 0.166411, 0.415539, -0.2073, -0.228538, -0.092917, -0.093668, 0.025724, 0.141747, 0.097656, -0.055123, -0.008014, -0.047371, 0.141723, -0.010442, 0.031694, -0.204641, 0.022385, 0.108527, 0.040512, -0.037243, -0.175011, 0.304019, 0.334306, 0.161182], [-0.23113, 0.154836, 0.014663, 0.479161, -0.080929, -0.359115, 0.398921, -0.370676, -0.096674, 0.036137, -0.153562, 0.344429, -0.488193, -0.311465, -0.237963, 0.534774, -0.205514, 0.15013, 0.335187, 0.160109, -0.183584, 0.065681, 0.324423, 0.123269, -0.484284, -0.203423, 0.281834, 0.037322, 0.168312, 0.424957, 0.071886, 0.399173], [0.21555, 0.177904, 0.246069, 0.077122, 0.065952, 0.007547, 0.083499, -0.398194, 0.265074, 0.106868, -0.005604, 0.05847, 0.025661, -0.19126, 0.080168, 0.352539, -0.596935, 0.164266, 0.015135, -0.433643, -0.214772, 0.230195, -0.19359, -0.161288, -0.017395, 0.023043, 0.06414, 0.220628, -0.017127, 0.240331, 0.181295, -0.217143], [-0.014279, 0.028623, -0.002853, -0.042258, -0.175438, -0.192216, 0.121514, -0.084126, -0.030135, -0.137486, 0.031603, 0.207499, -0.178857, -0.046373, 0.023856, -0.121209, -0.08089, -0.076674, 0.334829, 0.017252, 0.041405, 0.176097, 0.175058, -0.018048, -0.310646, -0.152201, -0.117382, -0.055356, 0.026214, -0.268462, -0.237596, 0.159293], [-0.090904, -0.342846, -0.28807, -0.268682, -0.633336, -0.301622, 0.198853, -0.245598, -0.448155, 0.078403, 0.18608, -0.175066, -0.070046, -0.239758, 0.152378, 0.261753, 0.103806, 0.093633, 0.15576, 0.146628, -0.009774, -0.371152, -0.006169, -0.025157, 0.120929, -0.167826, -0.000285, -0.314224, 0.183968, 0.141413, 0.232177, 0.299361], [-0.10061, -0.188915, -0.000314, 0.164683, 0.160024, -0.119988, -0.224234, -0.209083, 0.16825, -0.00169, 0.236097, 0.097395, 0.114644, 0.145068, -0.015195, 0.203997, 0.156601, -0.118154, 0.14654, -0.029664, 0.181696, 0.255309, -0.103118, 0.240108, -0.507132, 0.018492, -0.02577, -0.208228, -0.104259, 0.036422, -0.070732, -0.192603], [-0.123931, -0.265969, -0.059021, -0.076886, 0.331355, -0.21377, -0.252548, -0.030837, 0.21433, -0.145695, 0.240861, -0.216841, -0.057457, -0.093041, 0.151759, 0.042344, 0.049209, 0.00902, 0.076358, 0.094534, 0.148504, -0.04178, -0.099428, -0.075665, -0.077342, 0.188819, -0.203516, -0.361907, 0.066164, -0.194671, -0.154863, -0.368923], [0.037613, -0.291434, -0.334877, -0.319046, -0.211215, -0.07353, 0.166218, -0.255991, 0.240778, 0.071929, 0.117595, -0.029414, -0.067506, -0.094995, -0.374051, -0.370835, 0.018515, -0.137221, -0.143057, 0.120058, -0.097073, 0.080229, 0.180801, 0.030327, -0.011666, -0.009295, -0.102492, -0.1268, -0.024172, 0.04015, -0.180804, 0.20863], [-0.05508, 0.073227, -0.352023, 0.241855, -0.118355, 0.167631, -0.178198, -0.076521, -0.203847, 0.097195, -0.050812, 0.093098, 0.001622, -0.1701, 0.081931, 0.17177, -0.256512, 0.019196, -0.140784, -0.082679, -0.005094, -0.137922, 0.159654, -0.108196, -0.077421, 0.412132, -0.146314, -0.114751, -0.015277, -0.094935, 0.478826, -0.195283], [0.067703, 0.250277, 0.052818, 0.289109, -0.095405, -0.511172, -0.322192, 0.416593, -0.292737, 0.276536, -0.023413, -0.03115, 0.201037, 0.055654, 0.476451, 0.426893, 0.05046, 0.13888, 0.527534, -0.055435, -0.256501, -0.231654, 0.011074, -0.117849, 0.347615, -0.442477, 0.48496, 0.034349, 0.119976, 0.144104, 0.298629, 0.062357], [0.112469, 0.189945, 0.097085, -0.314349, 0.100153, -0.240451, -0.169246, -0.231017, -0.001776, 0.248617, 0.020312, -0.163366, -0.325209, -0.086781, 0.004063, 0.146086, -0.003453, -0.099913, 0.040806, 0.136612, 0.134423, 0.214233, 0.407824, -0.228327, 0.384475, -0.078966, 0.007613, 0.011927, 0.377041, -0.174513, -0.129415, 0.230703], [0.009168, -0.095962, -0.074537, 0.233197, 0.016, 0.04094, -0.163785, 0.137352, 0.036197, 0.015329, 0.077309, -0.192963, -0.007182, 0.104228, 0.015103, -0.106403, -0.058851, 0.024678, -0.153191, 0.05571, -0.156866, -0.013026, -0.022884, 0.068177, -0.045809, 0.129479, -0.164784, 0.222228, -0.000629, 0.036939, 0.148058, 0.057739], [-0.148709, -0.217844, 0.252739, -0.375359, -0.108929, -0.344452, 0.002955, -0.056844, -0.090812, 0.299599, -0.271826, -0.293928, 0.155427, -0.140224, 0.115692, 0.368643, 0.45242, -0.039139, -0.227867, 0.020999, -0.132878, -0.409874, 0.388018, -0.193648, 0.068868, 0.152356, -0.535038, 0.003911, 0.024137, 0.30265, 0.031295, 0.099204], [0.205279, 0.264414, -0.0196, 0.402338, -0.285258, 0.00791, 0.029194, 0.019533, -0.196957, -0.041043, 0.022286, 0.388018, -0.090571, -0.069999, 0.066678, 0.145387, -0.043892, -0.16398, -0.171256, 0.047236, 0.012703, -0.227874, 0.053958, -0.05219, -0.261008, 0.072942, 0.005305, -0.085634, -0.175094, -0.195253, 0.009874, -0.330898], [-0.048316, 0.04361, -0.083, -0.067177, 0.046814, 0.082337, -0.049749, -0.039737, 0.146987, 0.15887, 0.063784, 0.01192, 0.022899, 0.207845, -0.435744, 0.172416, -0.088114, -0.005445, -0.003837, -0.036959, -0.360776, 0.012982, -0.101135, -0.546598, 0.024915, 0.063612, 0.005324, 0.044002, -0.032513, 0.101425, 0.009047, 0.317743]], [-0.299636, -0.411892, -0.009426, -0.287485, 0.075385, 0.054427, 0.157069, -0.113547, -0.489737, -0.621888, 0.764921, -0.186161, 0.115277, 0.09802, 0.089545, 0.257961, 0.513729, -0.200332, -0.537441, -0.449226, -0.081038, -0.132077, -0.006251, -0.160277, 0.055805, -0.573139, 0.009229, 0.092957, -0.087619, -0.14227, -0.125663, -0.229625]),
    ([[-0.252338, -0.260099, -0.090547, -0.277361, 0.277373, -0.060079, 0.092303, -0.408316, -0.088337, 0.011832, 0.078543, 0.178125, -0.078286, 0.29119, -0.030967, -0.313654], [-0.297348, 0.007402, -0.3176, -0.255162, -0.246215, 0.039139, 0.154627, 0.017122, -0.097043, -0.221354, 0.201436, -0.013456, -0.18923, -0.515331, 0.061905, 0.13112], [-0.088452, 0.143476, 0.182834, -0.362677, -0.376776, -0.12359, 0.331693, 0.162151, 0.230859, -0.025486, 0.311774, -0.124425, -0.116953, 0.165286, 0.135468, -0.101387], [-0.290187, -0.329535, 0.020681, 0.186064, -0.112828, 0.184591, -0.245402, -0.140087, -0.062221, -0.014315, -0.127209, 0.159829, -0.026965, -0.224324, 0.41814, 0.189612], [-0.292312, 0.097933, -0.181817, -0.078958, -0.091911, -0.19539, -0.486403, -0.26104, -0.162155, -0.214958, 0.034393, -0.54579, -0.175305, 0.030332, 0.147845, 0.132954], [0.128766, 0.013983, 0.228457, 0.111781, 0.123375, 0.370025, 0.066351, 0.279204, -0.007467, 0.146366, 0.127282, 0.056044, -0.067067, -0.386816, -0.293694, -0.000599], [0.081918, 0.116442, 0.179324, 0.066977, 0.108105, 0.099273, 0.276383, 0.146933, 0.135825, 0.137686, 0.070281, -0.030793, -0.149026, -0.384936, -0.107071, -0.246987], [0.002359, -0.120541, -0.02625, -0.084081, -0.195833, 0.057961, 0.163917, -0.084893, 0.057254, 0.185867, 0.245686, -0.008441, -0.135443, 0.041258, -0.039317, 0.050645], [0.277533, 0.026215, 0.026333, 0.042844, 0.137866, 0.290016, 0.413112, 0.016474, 0.042782, 0.333332, 0.148757, 0.55369, 0.044515, -0.292741, -0.121292, -0.194158], [-0.030468, -0.561673, 0.035504, 0.00688, -0.432512, 0.007627, -0.064892, 0.119603, 0.208606, 0.169341, 0.095081, 0.122244, -0.134805, 0.140293, 0.295024, 0.179217], [0.054308, 0.272188, -0.297227, -0.100134, -0.031872, -0.451722, 0.258275, -0.022192, -0.111376, -0.110615, -0.062118, -0.133868, -0.172899, -0.15508, -0.087022, -0.539691], [0.048222, -0.285096, 0.061193, 0.213879, 0.112727, 0.194292, -0.084201, 0.107037, 0.073657, 0.300552, -0.319829, 0.084521, 0.059789, 0.30584, -0.106358, -0.220288], [-0.102353, -0.453999, 0.031131, 0.079939, 0.156522, 0.009578, 0.21295, 0.146353, 0.175581, -0.241208, 0.035329, 0.194093, 0.032745, -0.222518, 0.062965, -0.054762], [0.059317, 0.119478, -0.070008, -0.219112, 0.012568, -0.21888, -0.098203, 0.051445, 0.155092, 0.128164, 0.239171, -0.034999, 0.115761, -0.480327, -0.120384, -0.343133], [-0.083839, 0.072738, 0.097364, 0.130238, 0.076551, 0.154343, 0.002768, 0.065402, -0.142714, -0.43951, 0.003766, 0.20669, -0.046816, 0.195073, 0.05669, 0.246455], [0.035986, 0.257132, 0.064436, 0.178622, 0.012103, -0.068064, 0.238609, 0.082319, -0.213952, -0.772693, 0.198343, 0.117475, -0.084604, -0.049031, -0.156608, 0.276775], [0.26648, -0.290544, -0.105266, -0.053405, -0.085946, 0.178138, 0.000355, 0.11144, -0.476574, -0.183066, -0.115847, -0.104581, -0.016209, 0.041721, 0.051399, -0.761732], [0.159061, -0.466224, 0.308972, 0.157269, -0.007072, 0.118878, -0.312751, 0.286468, 0.067172, 0.479757, -0.120911, -0.585362, 0.000886, 0.430203, 0.090623, 0.017399], [-0.172728, 0.017239, 0.05694, -0.002901, -0.255993, 0.148453, -0.472937, -0.222734, -0.313792, -0.331064, 0.283867, -0.369821, 0.093427, -0.142473, 0.125392, 0.569083], [0.042015, 0.183102, 0.089642, 0.083504, 0.193612, 0.03566, 0.040911, 0.083389, 0.22533, 0.030441, -0.18631, 0.09681, -0.174481, 0.47267, -0.316175, -0.671885], [0.010377, -0.199031, 0.109443, -0.096817, 0.015315, -0.489962, -0.063933, 0.038314, 0.180149, 0.184605, -0.112863, -0.01157, -0.057106, -0.246211, -0.099882, 0.293937], [0.294997, 0.06857, -0.09123, -0.067941, 0.35449, -0.464662, -0.031233, -0.237817, 0.065547, 0.137549, 0.153253, 0.307486, 0.030446, -0.194029, -0.152757, -0.035514], [0.042185, -0.047178, 0.173269, 0.126717, 0.146096, 0.121348, -0.107397, 0.073486, -0.004381, -0.171049, 0.151305, 0.128749, -0.096452, -0.173424, -0.118143, 0.127411], [-0.257838, 0.043804, -0.12154, -0.052069, -0.304896, -0.186148, -0.247252, -0.034587, 0.099148, 0.29762, 0.059607, -0.105422, -0.129147, -0.333697, 0.037819, -0.215692], [-0.088845, 0.253657, -0.045241, -0.022958, -0.008831, 0.155683, 0.045141, 0.087934, 0.323956, 0.030337, 0.157955, 0.059156, 0.07071, 0.108865, 0.017605, -0.266672], [0.06227, 0.033497, -0.413227, -0.156502, -0.266141, -0.413019, -0.535756, -0.371025, 0.105275, 0.133864, 0.199673, 0.207901, -0.167834, 0.182603, 0.152072, -0.02436], [-0.250112, 0.170499, -0.437513, 0.06247, -0.197843, -0.514944, 0.010255, -0.221056, -0.061335, -0.120853, 0.075316, 0.012772, -0.009039, -0.231486, 0.207264, -0.051176], [0.070505, -0.059999, -0.002487, -0.11045, 0.263874, 0.137732, -0.023531, 0.013779, 0.170485, -0.236495, -0.213688, -0.391113, -0.174649, -0.284301, 0.109315, 0.051818], [0.229328, -0.3622, 0.171268, 0.226068, 0.153583, 0.262756, -0.098873, -0.010234, 0.241352, 0.389403, -0.036456, 0.175068, 0.076933, 0.187392, 0.04575, -0.11751], [-0.226471, 0.011959, -0.034117, 0.140463, 0.064445, -0.196061, -0.238456, 0.098343, -0.467399, 0.126849, 0.127079, -0.067966, -0.149777, -0.092426, -0.057878, -0.187328], [-0.129738, 0.090312, 0.087067, 0.010538, -0.000157, 0.058667, 0.076962, -0.016138, -0.356973, -0.761717, 0.128124, -0.337534, 0.107679, 0.248426, -0.049451, -0.08035], [0.268712, 0.060363, -0.014754, -0.155156, 0.018119, -0.117376, 0.051405, 0.156363, 0.059804, -0.384721, 0.113256, 0.124977, -0.197812, -0.44871, -0.623945, -0.111512]], [0.250912, 0.241678, 0.138379, 0.198739, -0.047919, -0.134062, 0.292555, 0.246881, 0.173358, -0.251822, 0.01943, 0.127819, -0.203631, -0.077761, 0.235435, -0.194678]),
    ([[0.391266], [0.664829], [0.354467], [0.574379], [0.705436], [0.802152], [-0.466911], [0.53547], [-0.795513], [-0.993672], [-0.231995], [-0.635174], [0.414765], [1.403551], [0.603721], [1.143111]], [-0.452727]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 5.068810841538566e-19, 0.0, 0.0, 0.0, -2.534405420769283e-19, 0.0, 0.0, 0.0, 1.2354840004101888, 0.0, 10.235373650724295, 29.045780629283016], [6.894880047762485, 19.932973433822138, 35.828844117459674, 46.28838112885247, 59.58345850504529, 1.1288535466346525, 0.9972382401475413, 0.9509100251726608, 0.8477614371106685, 0.7566240757362533, 1065.974830301333, 2039.56663015695, 4012.5353057956445, 0.5930296144511749, 0.3020581511144633, 17.10440165990182, 36.041044035504086], [0.566097143562707, 0.259264995938383, 0.003998079933218471, -0.18461439386994238, -0.09909113913895669, 0.8100942713400355, -2.814201129099548, 5.70728211279055, -3.713167888183222, 3.3848996757781586, -0.545115637040028, 0.02632223148335277, -0.18305262194727148, 0.001881757544255653, 0.25602824974821586, -0.0002618872582300937, -0.00019672322469201564], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 5.074727552069564e-19, 0.0, 0.0, 0.0, -2.537363776034782e-19, 0.0, 0.0, 0.0, 1.2357253973320002, 0.0, 10.24382157388333, 29.06687743136384], [6.898175212270779, 19.94432111131209, 35.84821729762494, 46.31374990156119, 59.61458245938529, 1.1292479320432283, 0.997614069404236, 0.9512769517801772, 0.8481040200541816, 0.756949523396557, 1064.2421889182478, 2038.6600009884921, 4007.1636750665784, 0.5932910613542522, 0.3022343925906883, 17.112570355001356, 36.056623057820026], [2.6498341383747235, 0.8365014691593934, -0.25579914674007964, -1.4326632881404089, -0.7070860011656996, 1.5376983088542304, -4.666766920955899, 9.850415060556683, -6.38626239553004, 7.444746721364866, -2.2326835524909727, -0.07363618221570306, -1.293430464591002, 0.010452110942003077, -1.0561616959005966, -0.0014540827961403151, -0.0010915738303765145], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.235968, 0.0, 0.0, 5.908278, 5.908217, 0.0, 7e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009079], [1.133464, 0.999389, 0.952272, 0.847731, 0.756024, 2.569325, 20.907391, 0.789165, 18.42144, 0.707427, 0.649163, 0.542068, 1.664649, 1.664601, 0.999991, 4.40323, 1.165561, 0.30371, 133.77359, 335.203674, 1064.987427, 1.340487], [([[0.071936, -0.037121, -0.154114, -0.084987, -0.065119, 0.155045, -0.108048, 0.110805, 0.039366, 0.141581, -0.041152, 0.045776, 0.183344, 0.08416, 0.032361, -0.193603, -0.072431, -0.235393, 0.242962, 0.026338, -0.13632, -0.002054, 0.107636, -0.239207, -0.099621, -0.169517, 0.068316, -0.249694, 0.595656, -0.076047, 0.039354, 0.17823], [0.143711, 0.212373, -0.179101, -0.070034, -0.21227, 0.34517, 0.103646, -0.304091, -0.048782, 0.047662, 0.102792, 0.037405, -0.07636, 0.029484, 0.075069, -0.166557, 0.130817, -0.108321, -0.031079, 0.157472, -0.316193, -0.18955, 0.24105, -0.018053, -0.177325, -0.170444, -0.100531, -0.063158, 0.136543, 0.143397, -0.03817, -0.002571], [-0.174634, 0.054518, 0.068904, 0.083045, 0.032284, -0.067261, 0.258481, 0.002808, 0.193881, -0.003473, -0.015009, 0.145965, -0.060489, -0.141771, -0.114869, 0.232067, 0.114876, 0.264447, -0.196085, -0.005113, -0.191832, 0.098398, 0.188449, -0.080067, 0.167821, 0.26111, -0.008179, -0.285328, 0.17878, -0.08632, 0.099797, 0.040544], [-0.121066, 0.191184, 0.174631, 0.50075, 0.234989, -0.357897, -0.135192, -0.170084, -0.236432, 0.062539, 0.031318, -0.246542, 0.090482, -0.058231, -0.289319, 0.114195, 0.114061, 0.267053, 0.059628, -0.165901, 0.09545, 0.4305, -0.047486, 0.258506, 0.319187, -0.046137, 0.095909, -0.027868, -0.153997, -0.044781, -0.422971, -0.088082], [-0.597658, -0.040773, -0.22789, -0.624874, 0.467621, 0.063167, 0.211634, -0.255726, -0.09816, -0.537483, -0.404531, 0.36246, 0.342404, -0.283422, -0.016394, -0.107249, 0.026602, 0.426718, 0.404479, 0.134058, -0.387814, -0.147852, 0.093913, 0.051178, -0.026643, -0.130653, -0.474337, 0.450431, -0.249712, 0.492532, 0.190043, 0.021213], [-0.067394, -0.147425, -0.181758, -0.044039, -0.099997, -0.229491, 0.05401, -0.246959, -0.089528, -0.099344, 0.082931, -0.155453, 0.067755, -0.105262, 0.037263, 0.188383, 0.195179, 0.255631, -0.082259, 0.070546, 0.004623, -0.057948, -0.026345, 0.088006, -0.256602, 0.066377, -0.166005, 0.175053, 0.106275, 0.056969, 0.170922, -0.227015], [-0.318694, -0.108462, 0.049903, 0.034137, -0.082634, -0.03125, 0.093059, -0.351551, 0.113497, -0.132162, -0.518192, 0.19411, -0.167605, -0.147481, 0.064839, 0.167563, -0.21325, 0.459094, -0.34718, 0.283035, -0.412416, -0.289023, -0.084931, 0.220886, -0.307761, 0.164687, -0.057628, -0.211158, 0.129955, 0.086733, 0.146786, 0.331092], [-0.040411, -0.166431, 0.224753, 0.137737, 0.010545, -0.015188, 0.046151, 0.056528, -0.073786, 0.212852, -0.142921, 0.028276, -0.154232, -0.139205, 0.103173, 0.100792, 0.081974, -0.099785, -0.338653, -0.187466, -0.067238, -0.113045, 0.054507, -0.228338, -0.186152, 0.233636, 0.089171, -0.109735, -0.196492, 0.218561, 0.231288, 0.1207], [0.329595, 0.127395, -0.124029, 0.283661, -0.093166, 0.004204, 0.199516, 0.054754, -0.053552, 0.156137, -0.529806, -0.311327, -0.271201, -0.226658, -0.120775, 0.015938, -0.448889, 0.038164, -0.000399, 0.304412, 0.004226, 0.045911, -0.057197, -0.053233, -0.385034, 0.129796, 0.006803, -0.284244, 0.097825, -0.341561, 0.247603, 0.465275], [0.130308, 0.057878, 0.078349, 0.226606, 0.058323, 0.272728, -0.024836, -0.259659, -0.05634, 0.162205, -0.114226, 0.097124, 0.188131, -0.335359, 0.082292, 0.366937, -0.710315, 0.146733, 0.088942, -0.236564, -0.321343, 0.334372, -0.164059, -0.194284, -0.598916, 0.071507, -0.148866, 0.162786, -0.244007, 0.116231, 0.277835, -0.190559], [-0.065914, -0.107481, 0.032769, 0.008014, -0.013654, -0.029863, 0.180525, -0.046576, -0.038825, -0.122533, -0.174649, 0.022806, 0.060619, 0.109443, 0.056932, 0.084081, -0.018099, -0.095287, 0.292118, -0.071025, 0.217916, -0.059373, 0.066881, 0.082568, -0.001318, -0.025566, -0.107989, -0.031927, -0.016569, -0.070327, -0.196212, 0.044665], [-0.038468, -0.439036, -0.279111, -0.023839, -0.193574, 0.03804, 0.203579, -0.262927, -0.254585, 0.00566, 0.282002, 0.059142, -0.219662, -0.047325, 0.09307, 0.037711, 0.008221, 0.208996, 0.075969, 0.344007, -0.024611, 0.074391, 0.166333, -0.217984, 0.280524, -0.299936, 0.56317, -0.25299, 0.19412, -0.049032, 0.150181, 0.373242], [-0.024414, -0.362334, -0.048348, -0.076618, 0.05819, -0.161679, -0.19566, -0.246971, -0.036491, 0.181742, 0.314563, -0.324285, 0.310086, 0.271924, -0.017421, 0.253553, 0.15465, -0.040328, -0.108428, 0.034544, 0.090988, -0.153482, -0.177712, 0.279117, -0.343354, 0.097009, -0.053998, -0.048437, -0.148636, 0.092897, -0.198716, -0.013004], [-0.114456, -0.278746, -0.127324, -0.128528, 0.23862, 0.127185, -0.330968, -0.036125, 0.084123, 0.142459, 0.235839, -0.145525, -0.232954, -0.022524, 0.190385, 0.094663, 0.185555, 0.135869, 0.120777, 0.218187, -0.07519, -0.18851, -0.081534, 0.005564, -0.025165, 0.050745, -0.251024, -0.346559, 0.032199, -0.009001, -0.01279, -0.177804], [0.011228, -0.168244, -0.185702, -0.167667, 0.008368, -0.069621, 0.302903, -0.262601, 0.008687, 0.133196, 0.153169, -0.384472, 0.11833, 0.068075, -0.337088, -0.133424, 0.048268, -0.137657, 0.297518, 0.051935, 0.030303, -0.028709, 0.176377, 0.005286, 0.157673, -0.214966, -0.007173, -0.151279, 0.046362, 0.116079, 0.12091, 0.289481], [0.068778, 0.191216, -0.200199, 0.056174, -0.033507, 0.030664, -0.01472, 0.032174, -0.138908, 0.05804, 0.091235, 0.332864, 0.161248, -0.09424, 0.196414, 0.499621, 0.04633, 0.010655, 0.071084, 0.062093, 0.115382, 0.023333, 0.227401, -0.032743, 0.098886, 0.368052, 0.28593, -0.033662, -0.091322, -0.149721, 0.244185, -0.196129], [-0.017551, 0.020672, -0.037224, 0.191896, 0.096284, 0.069577, -0.376096, 0.416316, -0.338576, 0.109816, -0.136992, -0.452083, 0.132515, -0.316246, 0.23003, 0.078959, -0.024681, 0.057491, 0.416835, -0.101159, -0.229493, -0.179114, -0.235536, -0.194543, 0.236283, -0.061935, 0.092442, 0.16367, 0.135277, -0.299706, 0.110153, -0.009276], [-0.019393, 0.177722, 0.083837, -0.243151, -0.012605, -0.342064, -0.198942, -0.002825, -0.051847, -0.032397, -0.261767, -0.085298, -0.122323, -0.051624, 0.139582, 0.157535, -0.167433, -0.020851, 0.026005, 0.0254, 0.225578, 0.191145, 0.324349, -0.108808, -0.082383, -0.080593, -0.120438, -0.123181, -0.035527, -0.125472, -0.110929, 0.268434], [-0.02663, 0.027566, 0.013178, 0.03098, 0.013275, 0.131554, -0.12258, 0.050743, 0.005525, -0.133489, 0.05378, 0.076478, 0.016534, 0.015555, 0.034534, -0.289303, -0.165838, 0.022417, -0.125288, -0.003132, -0.266431, -0.085075, 0.095502, -0.002191, -0.004796, 0.093153, -0.046588, 0.037232, 0.073206, -0.007104, 0.039454, -0.043533], [-0.132907, -0.252786, -0.1096, -0.065691, 0.033724, -0.104305, -0.102119, -0.027217, -0.338285, -0.391181, 0.073644, 0.527579, 0.157817, -0.079851, 0.192167, 0.202014, 0.168885, 0.112064, -0.304967, 0.08253, -0.074057, 0.098004, 0.222676, -0.261854, -0.092352, 0.234403, -0.028506, 0.059049, 0.071892, 0.200571, 0.104841, -0.110194], [0.226597, 0.115009, -0.203793, 0.173755, -0.642334, -0.347842, 0.113137, 0.155138, 0.11562, 0.219826, 0.093567, -0.028565, -0.011555, -0.164879, 0.031426, 0.140549, -0.060476, -0.27823, -0.304064, 0.210862, 0.067162, -0.026279, 0.050604, 0.02853, 0.058335, 0.208369, 0.124599, -0.163542, -0.319937, -0.215594, 0.349692, -0.352126], [0.020951, 0.272394, -0.103139, -0.14434, -0.073585, 0.08828, -0.051031, 0.013711, 0.143271, 0.025009, -0.042669, -0.084894, 0.049169, -0.015509, -0.468127, 0.459487, -0.098499, 0.023265, -0.039247, -0.093943, -0.151463, 0.096833, -0.043133, -0.246663, -0.04394, 0.295103, -0.039837, 0.028215, -0.059183, 0.000292, -0.012871, 0.176395]], [-0.453757, -0.197537, 0.175202, -0.015049, -0.473625, 0.286154, -0.178364, -0.270126, -0.066601, -0.605945, 0.390027, -0.471994, -0.003148, 0.006604, 0.101203, 0.044526, 0.500153, -0.339133, -0.294896, -0.310276, -0.235771, -0.146408, 0.037121, -0.459538, 0.109424, -0.401985, -0.47962, 0.341915, -0.241922, -0.338193, -0.161562, 0.038357]), ([[-0.234595, -0.560154, 0.138416, 0.022011, 0.23849, 0.138383, 0.188155, -0.421652, 0.062623, 0.15357, 0.205205, 0.066384, 0.103861, -0.096743, -0.141827, 0.376649], [-0.043166, 0.02152, -0.185021, -0.105523, -0.067099, -0.003714, 0.172564, 0.087401, -0.068811, -0.40603, 0.056298, -0.051051, -0.175901, -0.18313, -0.077655, 0.111663], [0.066586, 0.006366, -0.079111, -0.121655, -0.175886, -0.079461, 0.257613, 0.16331, 0.109237, -0.038619, 0.236035, 0.118249, -0.28696, -0.220886, 0.098582, -0.406231], [-0.326315, -0.195933, -0.299927, 0.106423, -0.356793, -0.028465, -0.144171, -0.206174, 0.239919, 0.008984, -0.124995, 0.216755, 0.038925, -0.471392, 0.220304, 0.148827], [-0.307123, 0.195654, -0.327554, 0.1256, -0.310989, -0.1096, -0.133298, -0.459009, -0.000403, 0.008662, -0.026531, -0.063053, -0.364202, -0.327128, -0.120158, -0.358671], [0.130185, 0.161473, 0.33015, 0.07147, 0.055421, 0.258627, 0.125574, 0.198539, 0.00585, 0.154703, 0.097278, 0.137171, -0.023618, 0.336088, -0.351621, -0.126584], [0.202684, 0.220336, 0.255129, 0.014127, 0.047696, 0.206115, 0.243705, 0.151276, 0.098062, 0.282411, -0.026778, -0.012179, 0.006185, -0.172602, -0.034535, -0.371647], [-0.022831, -0.200613, -0.298006, 0.044494, -0.12163, 0.234465, 0.159344, -0.115419, 0.034512, 0.238854, 0.197222, -0.000419, -0.29919, -0.485451, 0.198099, 0.09002], [-0.008766, 0.239891, 0.168392, -0.163413, 0.076731, -0.072332, 0.19025, 0.052223, -0.049895, 0.090987, -0.01069, 0.044356, 0.013338, -0.586398, -0.010873, 0.098212], [-0.126941, -0.202726, -0.356881, -0.12407, -0.59761, -0.124394, 0.169567, -0.156212, -0.196265, -0.086629, 0.121749, 0.23307, -0.111892, -0.704071, 0.417163, 0.125186], [-0.091645, 0.276755, -0.308474, 0.028654, -0.038964, -0.247421, 0.177416, 0.11545, -0.017866, -0.210079, 0.072889, 0.068183, -0.054664, -0.464537, -0.1061, -0.096331], [0.008387, -0.33002, 0.048027, 0.00587, -0.028161, 0.087851, -0.687766, 0.005054, -0.419732, 0.376022, -0.344606, -0.064359, 0.003526, -0.065799, -0.147005, -0.123246], [0.010259, -0.507925, 0.015037, 0.103543, 0.276165, 0.025628, 0.18978, 0.19582, 0.306466, -0.148258, 0.084805, 0.160732, 0.056204, 0.062544, 0.257864, -0.145944], [-0.076235, 0.078593, -0.350192, -0.241675, 0.204295, -0.451099, -0.018648, 0.042259, 0.058796, 0.136336, 0.230682, 0.102835, 0.047065, -0.935146, -0.086656, 0.012909], [-0.138942, 0.178729, 0.03359, 0.056169, -0.085547, 0.103195, -0.026772, 0.102165, -0.342882, -0.227942, 0.10977, 0.304308, -0.120263, 0.252273, 0.200669, 0.161273], [0.036584, 0.238295, 0.009416, 0.159133, -0.025081, -0.081981, 0.186622, 0.096346, -0.451755, -0.015647, 0.102708, 0.398103, -0.043281, 0.095969, 0.101329, 0.116299], [0.132231, -0.168945, -0.153656, 0.132054, -0.022087, 0.025842, 0.14175, 0.112741, -0.437927, -0.419907, 0.193468, -0.190107, 0.003837, -0.049682, 0.082216, -0.023227], [0.227401, -0.547602, 0.227913, 0.047933, -0.063698, 0.161207, -0.212563, 0.157701, 0.024695, 0.366759, -0.300615, -0.275388, 0.005754, 0.258276, 0.237513, 0.055885], [0.058494, -0.058847, -0.348843, -0.185228, -0.30267, 0.223933, -0.461501, -0.58527, -0.209255, -0.134438, -0.308035, -0.492882, 0.18186, -0.320807, 0.139084, 0.30929], [0.084776, -0.27229, 0.152067, 0.063065, 0.229255, 0.105991, -0.004648, 0.047788, 0.356047, 0.007993, -0.214601, -0.208928, -0.03438, 0.224965, -0.14248, -0.27917], [-0.148794, -0.062228, 0.069545, 0.011373, 0.17442, -0.502323, 0.140474, -0.044401, 0.101036, 0.26136, -0.102981, 0.04575, -0.070015, -0.531, -0.031547, 0.163822], [0.131502, -0.014729, -0.038591, -0.179014, 0.085686, -0.224236, 0.083099, -0.207904, 0.057508, 0.012301, 0.21966, 0.363934, -0.0542, 0.073118, -0.350835, -0.140651], [0.035076, -0.078441, 0.090186, 0.128596, 0.169738, 0.096342, -0.085746, 0.037523, -0.232406, 0.114113, 0.133782, -0.010214, -0.232723, -0.097081, 0.038612, 0.173269], [-0.262098, 0.027235, -0.150357, -0.048756, -0.190703, -0.108386, 0.216923, -0.118892, 0.220896, 0.35322, 0.276624, -0.387684, -0.157484, -0.061242, -0.080079, -0.223907], [-0.022461, -0.021063, -0.066026, -0.09222, 0.074731, 0.026972, 0.045474, 0.101659, 0.348756, -0.043143, 0.191342, 0.102264, 0.207169, 0.187807, -0.094276, -0.577808], [-0.165922, 0.185379, -0.333205, -0.181554, -0.246289, -0.233357, -0.190695, -0.195543, 0.041508, 0.049326, -0.012127, 0.198701, -0.262634, -0.161945, 0.176134, 0.215118], [-0.219028, 0.158014, -0.095706, -0.122631, -0.197265, -0.401772, 0.020801, -0.034413, -0.270495, -0.327028, -0.035941, -0.178408, 0.002518, 0.008298, 0.01059, 0.408972], [0.293874, -0.163148, 0.136421, -0.004744, 0.250767, 0.204546, -0.054644, 0.099171, 0.163132, -0.109775, -0.250065, -0.239903, -0.252242, 0.285226, -0.210559, -0.197331], [0.136743, -0.123517, 0.184118, 0.250178, 0.069051, 0.088253, 0.330269, 0.01229, 0.545502, 0.10348, 0.008512, -0.012821, 0.138171, 0.017971, 0.22143, 0.050037], [-0.083658, 0.04731, 0.129687, 0.162713, 0.161622, -0.102368, -0.313851, 0.07408, -0.391633, 0.318208, -0.077838, -0.317159, -0.189108, -0.093462, -0.234323, -0.079317], [-0.113844, 0.239987, 0.217243, 0.074217, 0.02971, 0.202227, 0.105859, 0.115508, -0.583346, -0.305801, 0.139542, -0.061267, 0.131292, 0.161006, -0.105789, 0.093445], [0.313455, 0.051323, 0.060968, -0.104609, 0.089864, -0.058338, 0.137325, 0.217604, -0.109544, -0.148276, -0.045586, 0.123294, -0.102457, 0.041283, -0.216772, -0.561999]], [0.204871, 0.223309, 0.062279, 0.190859, -0.050276, -0.175765, 0.176322, 0.246331, 0.047851, -0.374271, 0.181238, 0.33116, -0.229081, -0.047342, 0.257801, -0.485864]), ([[0.568411], [1.579788], [0.864936], [0.814892], [1.17313], [1.129058], [-0.661822], [1.092318], [-1.590705], [-2.110234], [-0.473004], [-1.120833], [0.982589], [2.389995], [1.15028], [2.74338]], [-0.786465])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.748641, -0.000111, 0.000111, 0.0, 0.053308, 0.053308, 0.0, 0.0, 0.0, 1.25666, 5.846188, 5.846188, 0.0]
SEQ_TSD = [1.078778, 0.135841, 0.135841, 0.102663, 1.444015, 1.444015, 0.669029, 0.222177, 0.183079, 0.616848, 1.678203, 1.678203, 0.651227]
SEQ_XM = [0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 1.235948, -0.0, -0.0, 5.908182, 5.908182, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009119]
SEQ_XS = [1.133826, 0.999664, 0.952553, 0.84802, 0.756213, 2.570136, 20.914425, 0.789277, 18.437494, 0.704278, 0.649224, 0.542084, 1.664759, 1.664759, 0.999991, 4.40289, 1.165521, 0.303697, 133.786133, 335.341339, 1065.236694, 1.340655]
SEQ_YSC = 1.9801905155181885
SEQ_WI = [[-0.436789, -0.298285, 0.259961, 0.134887, -0.526838, -0.329, 0.112731, 0.021516, -0.102757, 0.100807, -0.096792, 0.144346, 0.002183, -0.25868, 0.043191, -0.213889, 0.0248, 0.063759, -0.551466, -0.468945, -0.072131, 0.222312, -0.404532, 0.187232, 0.034501, 0.006336, -0.261401, 0.277897, 0.408691, 0.5022, 0.232969, -0.008734, 0.105206, -0.170352, 0.105333, 0.308116, -0.108443, -0.119291, 0.441489, -0.159496, 0.624551, -0.158476, -0.021348, 0.09292, -0.144258, -0.192, 0.132743, -0.326269], [0.172524, -0.243803, -0.784155, 0.116793, -0.172044, -0.124295, 0.043491, 0.581501, 0.409673, -0.21558, 0.208012, -0.691627, -0.046033, 0.253138, -0.251664, 0.02298, -0.384166, 0.022396, -0.628577, -0.045403, -0.290212, 0.147707, 0.489896, -0.218334, -0.099766, -0.440363, 0.092807, 0.181401, 0.066973, 0.551569, -0.575788, 0.383814, -0.028745, -0.083837, 0.645276, 0.365335, -0.224622, -0.118489, 0.487969, -0.115336, 0.004892, -0.659815, -0.133702, 0.079242, 0.13253, 0.10473, -0.189183, 0.365709], [-0.24765, 0.030204, -0.14671, 0.280207, 0.148685, 0.107472, -0.303904, 0.316045, -0.44605, 0.087929, -0.298784, -0.053659, -0.07257, -0.118952, 1.041025, 0.095588, 0.131373, 0.175966, 0.355901, 0.035777, 0.583969, -0.138024, 0.051123, -0.110204, -0.35863, 0.087525, -0.351157, 0.185799, 0.551791, -0.684639, -0.379777, -0.06862, -0.108593, 0.078406, -0.266638, -0.227342, 0.054229, 0.291747, -0.243771, 0.078097, -0.170634, 0.039985, 0.250538, -0.412001, 0.238767, -0.396237, 0.020501, -0.100285], [-0.266239, -0.043324, -0.107609, 0.238281, -0.378078, -0.088685, 0.204243, 0.05658, -0.063299, 0.139442, 0.044171, -0.03752, -0.049007, 0.25786, 0.295421, -0.159535, -0.148449, -0.087639, -0.315618, -0.184459, 0.333256, 0.417844, 0.265792, -0.297941, 0.083516, -0.340447, -0.294964, 0.269375, 0.037508, -0.213568, -0.600317, 0.243985, 0.099107, 0.218476, 0.169287, 0.012388, 0.05734, 0.069109, 0.095784, -0.14463, -0.332373, -0.093304, 0.281396, -0.16175, 0.49875, 0.102003, -0.331416, 0.047953], [-0.152924, 0.139874, -0.215992, -0.281209, 0.066285, 0.072026, 0.087593, -0.348037, -0.666013, -0.981552, 0.743511, -0.210247, -0.058205, -0.365915, 0.036552, 0.310041, -0.107298, 0.779358, 0.285626, -0.159852, 0.44467, 0.263754, -0.128169, -0.016004, 0.377612, -1.336636, 0.305529, 0.309618, 0.323041, -0.187764, -0.565591, -0.086794, 0.182155, 0.024796, -0.10274, 0.115269, 0.310056, 0.002482, 0.195836, 0.187123, -0.118032, -1.111974, -0.658904, 0.368019, 0.00499, 0.234965, -0.260976, 0.025986], [0.139213, -0.112878, 0.340684, -0.038359, -0.159706, 0.4276, -0.735315, 0.321965, 0.205531, -0.283354, -0.164875, -0.152519, -0.47611, -0.23328, 0.400072, -0.060604, -0.333584, -1.226924, 0.523895, 0.78925, -0.357482, 0.031005, 0.49827, 0.460245, 0.633554, 0.164303, 0.281124, 0.205397, -0.192106, 0.231704, 0.05152, -0.051738, 0.13082, -0.782234, 0.172104, -0.177354, -0.164179, -0.073352, 0.443522, -0.226032, -0.581114, -0.082894, -0.35601, -0.109435, -0.777267, 0.272099, 0.144294, 0.387968], [-0.037569, -0.104121, -0.397391, 0.470938, -0.439294, -0.116518, 0.430726, 0.232235, -0.56989, 0.291813, 0.023595, -0.042956, -0.368943, -0.05168, -0.336322, 0.143657, 0.042227, -0.108756, 0.308276, -0.097455, -0.054073, -0.02552, -0.201642, -0.050112, 0.270764, -0.003203, 0.155871, 0.313369, -0.1321, 0.652037, -1.7e-05, -0.134957, 0.085756, -0.07401, -0.240924, -0.250572, -0.048025, 0.2244, -0.018339, -0.006983, 0.193752, 0.147044, 0.136981, -0.007548, -0.100817, -0.344686, -0.045708, 0.071711], [0.149418, 0.005999, -0.401308, 0.613633, -0.48367, 0.211941, 0.259958, -0.144791, -0.309991, -0.111429, -0.182369, -0.219115, -0.387448, -0.117262, -0.117036, -0.219138, -0.085474, -0.136664, 0.170201, -0.084603, -0.050784, 0.029239, -0.157111, 0.046854, 0.015795, 0.380871, 0.202087, -0.175178, 0.241851, 0.182375, -0.157328, -0.285306, 0.184899, 0.150138, -0.305364, 0.048412, -0.339649, -0.038961, 0.276566, 0.025708, -0.107768, -0.015521, -0.111762, -0.067063, -0.018371, -0.19297, 0.164836, 0.331945], [0.042432, 0.334813, -0.397162, 0.500684, -0.114491, 0.321407, 0.181025, -0.255713, -0.606252, -0.244342, -0.078999, -0.081338, 0.1758, -0.18098, -0.07606, -0.464906, -0.264272, -0.074176, 0.058721, -0.157285, 0.176376, -0.162841, -0.353727, 0.190059, 0.145987, 0.430731, 0.154007, -0.014206, -0.407409, 0.430604, -0.146859, -0.163487, 0.272335, -0.052105, 0.061644, -0.23772, -0.450812, 0.01673, 0.211938, -0.035005, -0.123824, 0.143695, 0.046044, -0.219257, -0.115696, 0.523208, 0.17318, 0.142601], [-0.060221, -0.443069, 0.052611, 0.037008, 0.321444, 0.358191, 0.266529, -0.03838, 0.391242, -0.116398, 0.045846, 0.124047, 0.013087, -0.347541, -0.16371, -0.06908, 0.372132, -0.251732, 0.116849, -0.17025, -0.224391, -0.144345, -0.459604, -0.398169, 0.126511, -0.057147, 0.049083, -0.369205, 0.064154, 0.105991, 0.490687, 0.355563, -0.613546, 0.081404, -0.00067, 0.222511, 0.183753, -0.005959, 0.086759, -0.228162, -0.145304, 0.000255, -0.077959, 0.254303, -0.166011, 0.202768, -0.255778, -0.125576], [-0.243339, 0.25226, 0.037912, 0.002618, -0.183854, 0.148949, -0.514025, 0.255882, 0.317973, 0.591629, 0.18022, 0.228874, -0.187744, -0.292816, -0.093497, -0.09569, 0.088702, 0.572897, -0.246039, -0.33109, -0.563589, 0.286522, 0.092766, -0.014988, -0.476888, -0.424907, -0.597841, -0.45901, -0.140731, 0.070411, -0.038762, -0.322256, -0.02419, -0.240454, 0.392135, 0.003511, -0.063944, 0.261161, 0.041057, -0.131182, 0.127966, -0.002371, 0.035178, 0.119439, -0.089644, -0.123026, -0.12227, 0.163629], [0.100596, 0.325022, -0.372758, -0.121958, 0.0492, 0.094058, -0.506241, -0.046504, -0.129422, 0.195804, -0.019426, 0.260231, -0.02973, 0.099123, 0.229856, -0.599315, -0.13026, 0.310463, -0.617476, -0.189129, 0.187217, -0.215613, -0.419791, -0.142958, -0.276799, 0.114749, 0.075281, -0.228527, -0.528124, -0.390422, -0.351613, 0.193521, -0.185349, -0.275161, 0.190041, -0.005008, 0.123357, 0.360154, -0.16664, 0.006455, 0.06244, 0.086651, -0.07633, 0.086634, 0.088531, -0.150866, -0.067265, -0.068367], [-0.410873, 0.019133, 0.618279, 0.811816, -0.000394, -0.364686, -0.149484, 0.007061, 0.009581, -0.14004, 0.054875, -0.448455, -0.165103, -0.063908, -0.345076, 0.595686, -0.07427, 0.274741, 0.23565, -0.195721, -1.027839, 0.34177, 0.187998, 0.185301, 0.376555, -0.077405, -0.598826, -0.084738, 0.778908, 0.154081, -0.230772, -0.743589, -0.114603, -0.33445, -0.140519, -0.268643, -0.043738, -0.05131, 0.055033, 0.00944, -0.092483, 0.072127, 0.19071, 0.115068, 0.218926, 0.08871, -0.186193, -0.201307]]
SEQ_BI = [0.222414, 0.421276, 0.327159, -0.030458, 0.140194, 0.214866, 0.293138, 0.144903, 0.148267, -0.190889, -0.067146, 0.033132, 0.100272, 0.22289, 0.366408, -0.011414, 0.566122, 0.509907, 0.448106, 0.500677, 0.702412, 1.103417, 0.79392, 0.535979, 0.723233, 0.498008, 0.666517, 0.293577, 0.786952, 0.158576, 0.299352, 0.180309, -0.306705, 0.076377, 0.178554, 0.159975, -0.172899, -0.289236, -0.314618, 0.110892, 0.206989, -0.0125, -0.163326, -0.139958, 0.059701, -0.171551, -0.264094, 0.255297]
SEQ_WH = [[0.121747, 0.26583, -0.476402, 0.178816, 0.18587, -0.075012, 0.160314, 0.437798, -0.135436, -0.139386, -0.448822, 0.188088, 0.228666, 0.271539, 0.103248, 0.413264, 0.313151, -0.234672, -0.100305, -0.008462, 0.319599, -0.370142, 0.265744, 0.249386, -0.77319, -0.024614, -0.245089, -0.072336, 0.042169, -0.538004, -0.116744, -0.506003, 0.476163, -0.233087, 0.078779, -0.406538, 0.264642, 0.191888, -0.499333, 0.135183, 0.246344, -0.199033, 0.362394, -0.030334, 0.043404, -0.296544, 0.521527, -0.259447], [0.174983, -0.15513, 0.677347, 0.110034, 0.060986, 0.078212, 0.379035, -0.411435, -0.13643, 0.053138, 0.204288, 0.016655, 0.534697, -0.37869, -0.009845, 0.098873, 0.268815, -0.334701, 0.474399, 0.604526, 0.330652, 0.191722, 0.323059, -0.024709, -0.545205, -0.234094, 0.791848, 0.005673, -0.228437, -0.322093, 0.379582, -0.074948, -0.090494, 0.363913, -0.444689, -0.385772, 0.417358, 0.198532, -0.377066, -0.242221, -0.373935, 0.35271, -0.032631, -0.133937, -0.010929, -0.263941, 0.139841, 0.243333], [0.490729, 0.37931, -0.092938, 0.322584, 0.461025, 0.20772, 0.079218, 0.132163, 0.23024, 0.465088, 0.388612, 0.599608, -0.386533, -0.239381, 1.35063, 0.207801, -0.598469, -0.099938, 0.903975, 0.561018, -0.683025, -0.041149, 0.234088, 0.360354, 0.976363, 0.247562, 0.448398, 0.411037, 0.267467, -0.282816, 0.169993, 0.45799, 0.037103, 0.614402, 0.641843, -0.203938, 0.494328, -0.408927, 0.412745, 0.139059, 0.079455, 0.40559, 0.30396, 0.482783, -0.582098, 0.56458, -0.221789, -0.074328], [-0.653406, -0.53663, -0.344943, -0.362188, -0.781335, -0.26075, -0.084509, -0.765798, 0.567641, -0.457627, -0.513855, -0.592477, -0.174529, 0.169342, 0.247539, -0.301174, 0.21709, 0.195328, -0.010702, -0.10354, -0.421842, 0.018664, -0.043649, 0.31788, 0.56241, 0.200079, -0.740179, -0.158903, -0.361143, 0.178517, -0.173649, 0.392073, -0.520095, -0.516055, 0.322038, 0.716605, -0.210018, -0.543158, -0.144554, 0.535709, 0.392842, 0.016366, -0.435666, 0.15474, -0.313188, 0.006173, -0.035595, -0.093904], [0.112837, -0.293793, 0.01628, 0.299938, 0.463322, 0.336463, 0.310746, 0.058421, -0.46395, 0.286584, -0.174199, 0.25052, 0.680614, 0.28136, -0.122613, 0.250558, 0.17044, 0.012611, -0.3621, 0.26872, 0.121207, -0.084608, -0.911817, 0.308981, -0.763162, -0.792532, -0.081061, -0.571421, -0.902763, -0.023378, 0.833604, -0.582176, -0.006926, 0.05232, -0.347269, 0.034707, 0.149766, 0.581244, -0.057565, -0.334615, -0.126848, 0.225308, -0.152669, -0.363985, 0.452085, -0.122031, -0.204626, 0.012062], [0.240633, 0.310903, -0.015866, 0.12082, 0.277768, 0.213894, -0.007287, 0.26922, -0.14764, 0.280233, -0.110894, 0.454509, 0.072824, -0.293485, -0.073116, 0.01179, -0.348191, -0.09771, -0.751222, -0.08288, -0.09036, -0.191825, -0.764104, -0.220077, -0.982866, -0.534352, -0.742811, -0.057465, -1.043661, -0.00576, -0.408049, -0.570442, 0.280305, 0.129206, -0.058913, -0.357009, -0.020537, 0.567308, -0.120244, -0.418584, 0.085148, -0.098135, 0.057789, -0.220352, 0.09204, 0.222516, 0.202451, -0.273917], [0.119132, -0.04749, 0.617775, 0.360158, 0.451782, -0.024385, 0.220566, -0.136325, -0.040946, -0.027085, 0.37989, 0.177871, 0.063559, -0.723037, -0.528412, 0.069708, -0.164357, 0.043035, -0.212112, -0.598342, -1.288844, -0.381684, -0.250517, -0.550648, 0.026765, -0.587087, -0.045768, 0.125341, -0.166269, 0.25609, -0.256878, 0.154161, -0.417883, -0.068044, -0.460725, -0.039722, 0.061771, -0.184613, 0.853865, -0.020735, -0.338463, -0.174576, -0.124715, 0.147289, -0.107516, -0.235193, -0.367951, -0.003482], [-0.154164, 0.140486, -0.67105, -0.337191, -0.625718, -0.41982, -0.45329, -0.154061, 0.152752, -0.164936, -0.403638, -0.800366, -0.417455, 0.033743, 0.01052, -0.123717, -0.023019, 0.62433, -0.061554, 0.092106, 0.053153, -0.072734, 0.433477, 0.258015, 0.548434, 0.799536, 0.317398, 0.173996, 0.757851, 0.169309, 0.182165, -0.080725, 0.009864, -0.146793, 0.184052, 0.314616, -0.146381, -0.33452, -0.006942, 0.504596, 0.684981, 0.071343, 0.292515, -0.021841, -0.247358, 0.551352, -0.02914, 0.046289], [-0.40705, -0.152227, -0.182356, -0.078612, -0.171283, -0.285812, 0.357188, 0.439275, 0.106573, -0.175906, -0.629446, -0.693127, -0.235009, 0.129205, 0.181136, -0.320604, 0.411898, 1.087189, -0.286882, 0.249145, 0.110902, 0.667073, 0.12529, 0.826569, 0.479136, -0.020865, -0.125196, -0.464146, -0.253538, 0.695708, 0.155758, 0.189327, 0.28064, -0.436668, 0.116855, 0.407819, -0.086798, -0.076363, -0.251885, 0.336397, 0.150143, -0.307373, 0.164622, -0.393205, 0.211486, 0.347953, -0.220027, 0.149502], [0.204967, 0.217452, 0.283819, 0.557703, -0.308467, -0.313973, 0.035416, 0.182421, 0.171628, 0.357534, 0.239178, 0.369376, 0.332052, 0.386343, -0.103441, 0.06282, 0.564395, -0.53303, -0.400839, -0.093744, -0.566755, -0.052223, 0.103548, -0.597087, 0.54117, -0.521053, 0.007993, 0.359471, 0.71082, -0.855948, 0.004587, -0.35794, -0.163014, 0.245768, 0.272812, -0.157827, -0.024912, 0.215674, 0.021935, -0.41843, -0.794234, 0.413629, 0.415687, 0.589209, 0.063878, 0.17249, 0.464345, 0.425521], [0.552199, 0.573509, -0.007182, 0.22697, 0.066125, 0.320513, 0.338055, 0.380558, -0.068375, 0.629563, -0.255032, 0.413884, 0.189566, 0.205286, 0.204418, 0.304516, -0.252205, -0.046757, 0.370471, -0.107699, 0.473098, 0.238415, -0.23374, 0.129692, 0.936796, -0.491859, 0.635622, 0.417281, 0.604817, -0.452298, -0.39869, -0.266157, 0.407435, -0.011742, -0.067859, 0.144739, -0.61008, -0.36805, 0.092409, 0.07692, 0.235721, 0.299225, 0.424988, 0.119949, 0.315935, -0.372932, -0.042423, -0.055238], [-0.142432, -0.252445, 0.536784, 0.005856, 0.179335, -0.215093, 0.157318, -0.510669, 0.415265, 0.242211, 0.552129, 0.236861, -0.111794, -0.196834, 0.095981, 0.129966, 0.025776, -1.010051, 0.384372, -0.466247, -0.856181, -0.401281, 0.384114, -0.426647, 0.205683, 0.210258, -0.525564, 0.114751, 0.589584, 0.037183, -0.458709, 0.031161, -0.165538, 0.126248, 0.025547, -0.190559, -0.002206, 0.059491, -0.238851, 0.109825, -0.448718, 0.452466, -0.294993, -0.014326, 0.035011, 0.236591, 0.086438, 0.070237], [-0.059819, 0.394652, -0.126459, 0.185339, -0.097106, 0.041219, -0.350617, 0.261047, -0.050544, -0.46344, -0.214568, 0.100627, 0.235681, 0.310228, 0.020433, 0.092471, 0.376877, 0.386834, 0.137618, 0.328966, 0.650675, 1.018322, -0.306309, 0.292011, -0.209422, -0.367783, 0.451011, -0.423725, -0.325152, 0.178825, 0.118791, -0.511331, 0.282853, 0.280914, -0.228556, 0.029061, 0.253157, 0.355632, -0.692199, -0.385356, 0.482134, -0.232871, -0.20018, -0.381838, 0.078458, -0.252604, -0.25859, -0.556727], [-0.331348, -0.45543, 0.251075, 0.105849, -0.044897, -0.216257, 0.030213, -0.43856, -0.059923, 0.143571, -0.102223, -0.352534, -0.263193, -0.584738, 0.520799, 0.312018, -0.097477, -0.272122, 0.143007, -1.283065, -1.726461, 0.158832, 0.05097, 0.130373, 0.602383, 0.204853, 0.177486, 0.347736, 0.250445, -0.318542, -0.016424, 0.68284, -0.18388, -0.077629, 0.004927, -0.004922, -0.465973, -0.313052, 0.25064, -0.021761, -0.016552, -0.069822, -0.280037, 0.234224, -0.055028, -0.077878, -0.182816, 0.440546], [-0.13643, -0.334099, -0.653407, -0.199646, -0.429491, -0.075861, -0.272082, -0.244981, -0.167409, 0.451161, -0.236197, 0.090258, 0.105878, 0.095356, -0.103068, -0.17117, -0.295218, -0.445261, -0.315225, -0.488242, 0.06187, -0.832369, 0.207675, -0.509785, -0.713981, -0.48146, -1.043212, 0.526236, 0.420072, -0.054718, -0.338488, -0.291856, 0.333354, -0.019785, 0.514845, 0.046115, -0.384744, 0.057849, -0.10712, 0.245242, 0.058429, -0.030568, 0.443301, 0.338488, -0.244756, 0.113099, 0.50969, 0.121338], [0.448997, 0.123949, -0.620297, -0.270596, 0.152932, 0.484923, 0.185942, -0.009115, 0.285315, 0.03822, 0.020949, -0.047885, -0.308837, 0.152587, 0.134341, -0.179342, 0.372595, 0.804961, -0.804061, -0.128925, 0.688792, 0.486063, 0.59803, -1.161007, 0.048977, 0.512978, 0.634405, 0.368698, 0.264153, 0.838827, -0.291812, 0.341999, -0.119514, 0.102389, -0.056857, -0.199594, -0.468216, 0.106736, -0.106976, -0.120555, -0.347576, -0.147788, -0.043163, -0.126472, 0.03787, 0.045896, 0.231353, 0.534533]]
SEQ_BH = [-0.038336, 0.158413, 0.063595, -0.087395, 0.023413, 0.394341, 0.067091, 0.156679, 0.125517, -0.266559, 0.33635, 0.140501, 0.212179, 0.119489, 0.327407, -0.140732, 0.416131, 0.596411, 0.308577, 0.662115, 0.414568, 0.856814, 0.505749, 0.290944, 0.535275, 0.535777, 0.957767, 0.382158, 1.018803, 0.270443, 0.422224, 0.345384, 0.182381, 0.320687, -0.153356, -0.152326, -0.243328, -0.186908, 0.136285, 0.194014, 0.284884, 0.268919, -0.033833, -0.196821, 0.307272, 0.255533, 0.074023, -0.105571]
SEQ_HEAD = [
    ([[0.096345, 0.048322, 0.106593, 0.099609, 0.165356, -0.145806, 0.200549, -0.077803, 0.02981, -0.238045, 0.202254, 0.059918, -0.045975, 0.03997, -0.155013, -0.107747, 0.03474, -0.067094, 0.168457, 0.113112, 0.042961, -0.162045, -0.170343, 0.315838, -0.003608, -0.15767, 0.116412, -0.135623, -0.022829, -0.054582, -0.043351, -0.05177, -0.121528, -0.126726, -0.099944, 0.056051, 0.007834, -0.037971, 0.120574, 0.127283, 0.223659, 0.150458, -0.082844, -0.015115, 0.073826, -0.003977, 0.118856, -0.446616, -0.036345, -0.016298, 0.242417, -0.006808, -0.182948, 0.015154, -0.117675, 0.1769, -0.23369, -0.100531, 0.035747, 0.223633, 0.171959, -0.07147, 0.006797, 0.052445], [0.024361, -0.058737, -0.046409, -0.014151, -0.165328, -0.017587, 0.112354, -0.048587, 0.04365, -0.192155, 0.124057, 0.018651, 0.071694, -0.04824, -0.1057, 0.050004, 0.056923, 0.054849, 0.065618, -0.116879, -0.061306, 0.002346, 0.024424, -0.075879, -0.050744, -0.037784, -0.070054, -0.047192, -0.001524, -0.035834, 0.044895, -0.066292, -0.013177, -0.183296, 0.120789, -0.064918, -0.082776, 0.117634, 0.008777, 0.044516, -0.011856, 0.041586, 0.03482, -0.118905, -0.007999, -0.129481, 0.146492, -0.234891, 0.135323, 0.122206, 0.178169, 0.152274, -0.133309, -0.166195, 0.233656, 0.041856, 0.057799, -0.110301, -0.039641, 0.164226, -0.117336, 0.102802, -0.002943, -0.056165], [0.123175, -0.097053, -0.004092, -0.003249, -0.165145, -0.088471, 0.174865, -0.060112, 0.066276, -0.14181, 0.169969, -0.127795, 0.053237, 0.009345, -0.005233, 0.051502, -0.107431, 0.106615, 0.015619, 0.049656, 0.035228, -0.139824, 0.061113, -0.100329, -0.089807, 0.036956, -0.137033, 0.037611, 0.097183, -0.140837, -0.019075, -0.027397, 0.005262, 0.033469, -0.054749, -0.155919, 0.123063, 0.009777, -0.018014, -0.21714, 0.104787, -0.125242, 0.005991, 0.098407, -0.049172, -0.092198, 0.092487, -0.109851, 0.010835, -0.061615, -0.082801, 0.026441, -0.109249, -0.227442, 0.055792, -0.15729, 0.095925, 0.125332, -0.145335, -0.041731, -0.121087, 0.074749, -0.065499, -0.168567], [0.057218, -0.17694, -0.20042, -0.093265, -0.23906, 0.151319, -0.110333, 0.023558, 0.041524, -0.054903, -0.046657, 0.125898, 0.153462, 0.016078, -0.094725, 0.036196, 0.171339, -0.19705, 0.121587, -0.189069, 0.1764, -0.031109, 0.141089, -0.001472, 0.211477, 0.072357, 0.064371, -0.096197, 0.012131, -0.008322, -0.053508, 0.048994, 0.218289, -0.03569, 0.052188, 0.024601, 0.198142, -0.067066, -0.089815, -0.108379, -0.057455, -0.27299, -0.065645, -0.156943, 0.023889, -0.099357, -0.069338, 0.103069, -0.133031, 0.027644, -0.079176, -0.101077, -0.151155, 0.004112, 0.011771, 0.103858, 0.324879, -0.003488, 0.150287, -0.053008, -0.128898, 0.130332, 0.136429, 0.031804], [0.085024, 0.10043, -0.327434, -0.063614, 0.245625, -0.176305, 0.172718, -0.060075, 0.326654, 0.01844, -0.073856, 0.064128, 0.155628, 0.267019, -0.020141, -0.11546, -0.202528, 0.114138, 0.1256, 0.197073, 0.198986, 0.067334, 0.016968, -0.063551, -0.02824, -0.008498, -0.169385, 0.033905, -0.406079, -0.235895, -0.29113, -0.000689, 0.277164, -0.010253, -0.032024, -0.093182, -0.059694, 0.271671, 0.096389, -0.145421, -0.126968, -0.239357, 0.013713, -0.022132, -0.260226, -0.124859, -0.005151, 0.068846, -0.107539, -0.041017, -0.184435, -0.212738, -0.11936, 0.022328, -0.24667, -0.155907, 0.199666, 0.0944, -0.069986, -0.074935, -0.268146, 0.079265, 0.00792, 0.075649], [-0.155066, -0.126114, 0.092027, -0.032453, -0.066736, -0.086216, -0.046872, 0.090247, 0.05592, -0.148909, 0.071837, -0.098326, -0.098715, -0.267067, 0.010156, 0.033506, 0.021992, 0.038861, -0.000397, 0.042711, 0.156827, -0.041658, 0.049648, -0.037681, 0.040426, -0.109672, -0.007211, -0.070527, -0.045175, 0.09182, 0.039463, -0.028603, 0.122891, -0.067891, -0.091258, 0.015622, 0.174848, 0.055733, 0.134925, -0.129251, 0.075039, -0.024334, 0.121212, -0.040952, 0.06086, -0.156351, -0.097528, 0.088754, 0.131992, 0.098352, -0.011395, 0.147209, 0.009215, 0.01912, 0.178476, -0.049994, 0.09249, 0.065, -0.072811, 0.049363, -0.155944, -0.024838, 0.025588, -0.059679], [-0.036631, 0.043559, -0.316565, -0.283978, -0.061641, -0.063096, -0.040853, 0.109141, 0.019991, 0.086645, 0.035235, 0.106914, 0.004486, 0.092355, 0.085852, 0.069442, -0.097452, -0.115941, 0.01027, 0.213273, -0.011366, 0.15153, 0.137921, 0.289834, -0.013412, -0.141149, -0.115348, -0.015487, -0.053433, 0.112075, -0.026911, -0.007881, 0.039754, -0.011691, 0.109366, -0.026402, 0.079794, 0.122489, 0.2273, 0.069693, -0.075492, -0.069902, -0.19099, 0.287507, 0.030875, -0.206002, 0.033188, -0.040619, 0.02587, -0.011942, 0.17786, 0.040176, -0.170441, -0.07264, -0.096524, -0.121184, 0.035882, -0.101848, -0.017888, 0.109826, 0.017676, 0.027897, 0.016712, 0.235258], [0.045334, -0.043363, -0.048469, 0.023881, -0.010095, -0.086198, -0.005129, 0.250034, -0.09035, 0.158811, -0.115002, -0.159992, -0.037203, 0.216324, -0.123347, 0.209861, 0.010915, 0.192175, -0.071941, -0.02344, -0.067224, 0.095653, -0.101238, -0.040029, -0.22946, 0.127625, -0.091529, 0.16011, 0.036746, 0.130472, 0.195655, -0.008829, -0.052351, 0.07953, 0.293104, -0.108969, -0.0344, -0.063446, -0.013591, -0.08545, 0.03771, 0.193182, -0.007042, -0.07227, -0.129815, -0.122585, 0.085079, 0.254204, -0.327934, 0.000768, -0.059956, 0.065797, -0.003267, -0.007664, 0.00188, -0.290126, -0.201493, -0.017355, -0.119619, -0.010026, 0.093161, -0.020925, -0.08331, 0.008036], [0.09306, -0.021626, 0.050066, -0.131016, -0.139196, -0.17707, 0.063866, 0.046818, 0.068523, -0.084167, 0.061209, 0.236415, 0.210018, -0.204418, -0.133191, -0.058552, -0.117481, -0.11738, 0.156074, -0.070626, 0.158658, -0.163905, -0.151855, -0.036898, -0.232368, -0.063339, -0.249397, -0.010138, -0.276694, -0.024083, -0.034384, -0.273384, -0.234066, 0.014764, -0.021216, -0.194599, 0.279465, 0.135731, -0.021815, 0.019993, 0.147951, 0.077121, 0.126287, -0.107741, -0.282603, -0.201131, 0.027738, -0.098652, -0.005909, 0.000755, 0.114713, 0.223022, -0.151369, -0.151691, -0.017969, -0.111201, 0.086676, -0.06979, -0.129713, 0.205573, -0.232675, 0.014845, 0.096249, 0.13423], [-0.183377, -0.273376, 0.004613, 0.15515, 0.001202, 0.149063, 0.050982, 0.009161, -0.077414, 0.167866, -0.121894, 0.120727, -0.072859, -0.010878, 0.001914, -0.031684, 0.130318, -0.197437, -0.002001, -0.033637, 0.113388, -0.00888, -0.360521, -0.007346, -0.046723, -0.158338, -0.098367, -0.237862, -0.197419, 0.111114, 0.161505, 0.063694, 0.014889, -0.161346, -0.016761, -0.241267, -0.085365, 0.143017, 0.057758, -0.031978, -0.094741, -0.202753, -0.031083, -0.08052, 0.082221, -0.138078, -0.155764, -0.015984, -0.095424, -0.114144, 0.019928, 0.1516, -0.1783, -0.024801, 0.075073, -0.07153, -0.285188, -0.003863, -0.150985, 0.048566, -0.117305, -0.055885, 0.109845, 0.101063], [0.14558, -0.085998, -0.008823, -0.095606, 0.055859, -0.109618, 0.075601, 0.215934, -0.032795, -0.097431, -0.208157, 0.052988, -0.137454, 0.098279, -0.139418, -0.029702, 0.056564, -0.04185, 0.136975, 0.048251, -0.062948, -0.008409, 0.012521, 0.119779, -0.092127, -0.117207, 0.071025, 0.062998, 0.086076, 0.195243, -0.010443, 0.13044, -0.125327, 0.013642, 0.000338, -0.076369, 0.006002, 0.019082, 0.114247, 0.080318, 0.026842, 0.096132, -0.14826, 0.034225, -0.036891, 0.000412, -0.121378, 0.07284, 0.110801, -0.158492, -0.035785, -0.179003, -0.06866, 0.003961, 0.21921, 0.116576, -0.040702, -0.050135, -0.23273, 0.113779, 0.051785, -0.004532, 0.118811, -0.100734], [0.142326, 0.213757, -0.028437, 0.114617, -0.010243, 0.012166, 0.081526, 0.03079, 0.020605, -0.196685, 0.025078, 0.420408, -0.00406, 0.037673, 0.062783, -0.2522, -0.126468, 0.128118, 0.026461, -0.069341, 0.321251, 0.129753, 0.24418, -0.135417, 0.208814, -0.230064, -0.002419, -0.311761, -0.058717, -0.036561, -0.184133, -0.077965, -0.050043, -0.135692, -0.01232, -0.15069, -0.077252, 0.055815, 0.267804, 0.320217, 0.435817, 0.113899, -0.094541, -0.157841, -0.2391, 0.111424, 0.207093, -0.118551, 0.254097, -0.257284, 0.16359, -0.067866, -0.157805, -0.186219, 0.031013, -0.060555, -0.217976, 0.199785, -0.176995, 0.175732, -0.021525, -0.282093, 0.221315, 0.09722], [-0.025501, 0.121777, 0.02529, -0.092709, -0.130062, 0.064798, -0.014523, 0.038196, -0.08194, -0.005866, 0.086933, 0.29242, 0.116195, -0.083363, -0.09046, 0.089014, 0.177966, 0.143029, -0.052594, -0.156898, -0.062972, 0.211347, -0.04169, -0.060792, -0.120933, 0.09217, -0.150461, 0.256894, -0.121717, -0.136049, -0.132486, -0.016547, 0.013111, -0.072698, -0.029443, -0.063654, 0.083849, -0.01933, -0.04682, -0.080469, 0.016932, 0.128644, -0.141729, 0.186619, -0.025918, 0.192816, -0.036383, 0.103396, 0.125485, -0.05272, 0.023612, -0.234896, 0.107898, -0.048739, -0.006854, 0.088339, -0.0411, 0.081393, 0.096232, 0.096228, -0.035268, 0.083077, 0.03542, -0.068153], [-0.274379, 0.079653, -0.132879, 0.015321, -0.29254, -0.006007, -0.089172, 0.089351, 0.040469, -0.064208, -0.068296, 0.300922, -0.03176, 0.108906, 0.017531, 0.075187, 0.108173, 0.029042, 0.047662, -0.001496, -0.205779, -0.055731, 0.125095, -0.027502, -0.080028, 0.103754, -0.073383, -0.075006, -0.044877, -0.213432, -0.106248, 0.05319, -0.040933, 0.050969, -0.007969, -0.003512, 0.03365, -0.182182, -0.093934, -0.196421, 0.082146, 0.106616, -0.025975, 0.220245, -0.09978, 0.168491, 0.130064, -0.195727, -0.024913, 0.00918, -0.075311, -0.143319, 0.03244, -0.088889, -0.039229, -0.019506, 0.065186, 0.255675, 0.083229, 0.130048, 0.069602, -0.077619, -0.105004, -0.062283], [-0.039302, 0.095539, 0.126896, 0.245364, 0.080568, 0.17547, -0.069428, 0.069909, -0.160745, -0.139076, -0.108582, -0.178621, -0.162361, -0.120773, -0.040444, -0.123806, 0.152761, -0.275914, -0.014993, -0.108544, 0.051966, -0.129385, -0.037198, -0.10935, 0.068001, -0.037474, -0.192737, -0.023285, -0.123994, 0.035062, 0.057608, -0.11424, -0.336758, -0.104469, 0.002959, 0.170821, -0.022596, -0.079625, 0.153251, -0.160674, -0.018072, 0.040244, -0.116878, 0.04804, 0.207683, -0.075725, 0.098256, 0.141954, 0.207753, -0.068859, -0.170863, -0.067593, 0.031256, -0.218632, -0.000119, 0.299133, -0.06412, -0.197474, 0.217805, -0.1407, 0.063876, 0.001355, 0.070381, -0.144956], [-0.170762, -0.005381, 0.104348, -0.027217, 0.246696, -0.087552, 0.10754, -0.087736, -0.084367, -0.10901, -0.134839, -0.183065, 0.094882, -0.17038, 0.068931, 0.093921, -0.039854, -0.004193, 0.036876, -0.221164, 0.04707, -0.167208, -0.153759, -0.091265, -0.324405, -0.061422, 0.07825, 0.0159, -0.065128, -0.14212, -0.098563, -0.100621, -0.440533, 0.005919, 0.211524, 0.167511, -0.187964, -0.04377, -0.011803, 0.068907, -0.019609, 0.010407, 0.095388, -0.073725, -0.161232, -0.127558, -0.00915, -0.087576, -0.015269, -0.052561, -0.126281, 0.23926, 0.093738, 0.100082, -0.06278, 0.075811, -0.135431, -0.075332, 0.178084, -0.111941, 0.214062, 0.089869, 0.057134, 0.126546], [-0.192489, 0.262641, -0.043432, -0.251712, 0.070908, 0.02674, 0.265742, -0.190895, -0.038635, -0.116481, 0.118718, 0.027187, 0.254277, -0.120809, -0.082431, 0.126744, -0.197499, 0.462537, 0.250177, 0.178669, 0.131268, 0.191635, 0.218657, 0.04923, -0.00237, -0.098329, -0.012833, -0.08104, 0.201265, 0.008903, 0.017516, 0.003714, -0.15609, -0.366239, -0.195139, 0.144762, -0.392347, -0.042652, -0.309762, 0.019819, 0.133991, 0.103597, -0.109742, -0.24618, 0.027055, -0.125093, 9.5e-05, -0.006583, -0.38998, -0.297973, 0.103252, 0.045366, -0.167635, 0.422396, 0.220959, 0.108164, 0.162212, 0.260912, -0.399789, 0.246547, -0.283589, 0.121191, 0.147094, -0.124269], [0.13691, 0.15855, 0.052309, -0.041552, -0.211171, 0.136077, 0.286586, -0.045792, 0.133578, -0.172437, -0.123032, 0.110793, 0.050812, 0.19783, -0.096377, -0.063473, 0.146726, -0.013907, -0.122717, -0.11724, -0.196388, -0.137018, -0.045556, 0.239828, 0.10124, -0.19355, 0.330132, -0.029989, -0.174089, -0.023377, 0.13776, 0.1373, -0.038307, -0.097223, 0.044446, -0.209761, -0.098577, -0.124767, 0.016445, -0.184224, -0.051267, -0.024255, -0.142818, 0.009552, 0.102205, -0.138553, -0.239341, 0.208101, 0.045051, -0.224784, 0.041092, -0.04522, -0.166639, 0.025321, 0.012012, 0.058, 0.115407, 0.035641, -0.016416, 0.07414, -0.111593, 0.092265, -0.11673, -0.151262], [-0.051866, 0.03618, -0.061822, 0.081917, 0.050464, -0.007215, 0.028909, 0.129391, -0.113834, 0.100003, 0.103471, -0.046859, -0.133342, 0.059422, 0.044375, 0.048192, -0.051369, 0.041311, 0.005286, 0.024934, -0.001009, -0.105125, 0.06715, -0.060609, -0.252564, -0.015037, -0.016842, 0.032014, 0.108474, -0.118022, -0.012232, -0.211104, 0.094441, 0.058873, 0.086479, -0.006528, -0.105497, 0.109781, -0.12188, 0.066089, 0.058524, -0.154529, -0.151789, -0.010375, 0.082225, 0.184237, 0.094915, -0.05798, 0.005041, -0.063361, -0.136484, 0.233737, 0.176745, 0.172782, -0.080264, -0.136328, 0.036033, -0.023984, -0.007857, -0.080369, 0.039778, -0.053989, -0.160435, -0.061273], [0.069846, 0.236254, 0.018786, -0.140772, 0.032987, 0.09782, 0.026147, 0.223301, 0.012039, -0.009077, -0.080372, 0.095703, 0.171638, -0.121792, -0.072974, -0.099245, -0.191179, -0.04324, 0.069883, -0.178709, 0.143981, -0.146818, 0.121398, 0.208955, -0.041657, -0.040602, 0.093842, 0.158013, 0.003381, -0.21341, 0.050173, -0.086891, 0.252725, -0.130404, -0.027147, 0.003981, -0.25056, -0.168919, -0.160868, 0.063996, -0.091065, 0.036627, -0.076732, 0.180127, -0.274873, -0.130661, 0.091535, -0.051332, 0.233421, -0.11034, 0.154743, 0.373355, -0.2739, 0.080899, -0.228706, -0.436924, -0.077764, 0.085651, 0.122035, 0.092389, -0.138025, -0.082094, -0.02343, 0.062133], [-0.13212, -0.08173, 0.247523, -0.234561, -0.216658, -0.276095, 0.024002, -0.205603, 0.047467, 0.136377, -0.011854, 0.029588, -0.092392, -0.017691, -0.055835, 0.004448, -0.148211, 0.026429, -0.105781, -0.166165, 0.004975, 0.142351, -0.067265, -0.302273, -0.000638, -0.03542, 0.075594, 0.24805, 0.152785, -0.001003, 0.037838, -0.225309, -0.1969, 0.054773, -0.249117, 0.021373, 0.302517, 0.027296, 0.071677, 0.258739, 0.037406, 0.291339, 0.145179, 0.14447, 0.003616, 0.010239, -0.067857, 0.302641, -0.014259, 0.071616, -0.218244, 0.066291, 0.022958, -0.031983, 0.031897, -0.071962, -0.25073, -0.13452, 0.163924, 0.103002, 0.108129, -0.218333, 0.068304, 0.223888], [0.248676, -0.083834, 0.027645, 0.36906, -0.062351, 0.011657, 0.005711, 0.013867, 0.221535, -0.035305, 0.047804, 0.114409, -0.135036, 0.005188, 0.057421, 0.026672, -0.208427, -0.317709, -0.006895, 0.2234, -0.094756, -0.023665, 0.353957, 0.081966, -0.073491, 0.069879, 0.392325, -0.122854, 0.196983, 0.026407, 0.08018, -0.078806, -0.055097, -0.050027, -0.183095, 0.07097, -0.044844, 0.112437, -0.096841, -0.042979, 0.032893, -0.162693, 0.159263, 0.094059, -0.007444, 0.145485, -0.087184, 0.095948, -0.192971, -0.306396, -0.128388, -0.101773, -0.059067, -0.157951, -0.030119, -0.142981, 0.095354, -0.0058, -0.176679, 0.262852, 0.120421, 0.134658, 0.309935, 0.308857], [0.210653, 0.153473, -0.281312, 0.29096, -0.106575, 0.199056, 0.261917, -0.263352, 0.439713, 0.219317, 0.080596, -0.006938, 0.054536, -0.026757, 0.28263, -0.068071, 0.131725, 0.008365, 0.179808, -0.068327, 0.175891, 0.4076, 0.127653, 0.094171, -0.312235, 0.139358, -0.214901, 0.02082, 0.092905, -0.250372, -0.066744, 0.095263, -0.237269, 0.099811, -0.146439, 0.289163, 0.011271, -0.05934, 0.273248, -0.056138, 0.291479, 0.016789, -0.000308, -0.001851, -0.116285, 0.103066, -0.144719, -0.014117, -0.240123, -0.027931, 0.223467, -0.349849, 0.094702, 0.087574, -0.112723, -0.060933, -0.263519, 0.077728, 0.130895, -0.124922, 0.327625, -0.158078, 0.249547, -0.107562], [0.146118, 0.164724, 0.212427, 0.07676, 0.267192, -0.146497, 0.176402, -0.05909, 0.081676, 0.059217, 0.085008, -0.18853, -0.070535, -0.148063, 0.137523, -0.078065, -0.003832, -0.173978, -0.027698, -0.20093, 0.161828, -0.120161, -0.121644, 0.116744, 0.085057, -0.077777, 0.035179, -0.202638, 0.106711, 0.099634, 0.154999, -0.047616, -0.12238, 0.052561, -0.155245, -0.251592, -0.003727, 0.199487, 0.328121, -0.006269, -0.029131, -0.054496, 0.07662, -0.578765, 0.045526, 4.9e-05, 0.063422, -0.100129, -0.106093, -0.078843, 0.02454, 0.142007, -0.128306, 0.143484, 0.297722, -0.193261, -0.237192, -0.01093, 0.024689, -0.004858, -0.142817, 0.083791, -0.077275, 0.026732], [-0.109718, -0.039239, 0.164063, 0.138426, 0.003739, 0.296767, -0.226747, 0.092919, 0.132506, 0.172712, 0.280519, -0.188802, -0.284801, 0.06271, -0.062533, -0.301623, 0.02377, 0.107943, 0.005372, 0.168747, 0.066085, 0.124265, 0.284977, -0.129405, 0.268873, 0.23749, 0.064193, 0.049098, -0.03431, -0.044718, 0.253718, 0.118403, -0.157731, 0.120619, 0.106759, -0.088366, 0.092422, -0.253663, -0.173413, 0.063797, 0.03652, -0.006813, 0.242569, -0.033673, 0.050522, 0.163731, 0.097547, 0.102053, 0.281401, 0.008691, -0.116333, 0.265472, -0.389325, -0.082023, -0.136097, 0.009329, -0.082628, 0.044548, 0.014279, 0.184647, 0.00282, -0.177724, 0.032507, 0.095405], [-0.286835, -0.320935, -0.023259, -0.178839, -0.107176, -0.205695, 0.026978, 0.105649, -0.024125, 0.036936, -0.20589, -0.214303, -0.092256, -0.003928, 0.063702, 0.212542, -0.014327, -0.003411, -0.287194, 0.07433, 0.021971, 0.109381, -0.224893, 0.009203, -0.087096, -0.215408, 0.056818, -0.095304, 0.049046, -0.076756, -0.10965, -0.11594, 0.067062, 0.020946, 0.09223, -0.042542, -0.187172, -0.137171, -0.128664, -0.127759, -0.164495, 0.009256, 0.043906, -0.062271, -0.034177, -0.00743, -0.293992, -0.031145, -0.183983, -0.051055, 0.194703, -0.058907, -0.078501, 0.005579, -0.039823, -0.0147, -0.043198, -0.058453, -0.041754, 0.16654, -0.033226, 0.092536, -0.093979, -0.045143], [-0.003577, 0.530964, -0.105148, 0.368189, 0.049416, 0.05276, 0.26434, 0.173513, -0.011189, 0.204005, -0.071343, -0.218573, -0.058066, -0.08649, 0.06552, -0.098045, -0.049664, 0.139451, 0.113083, 0.182629, 0.05732, -0.081861, 0.131622, 0.093676, -0.249437, 0.100991, -0.120767, -0.068457, -0.079873, -0.086402, 0.064673, -0.05264, -0.169916, 0.045476, 0.034976, -0.256544, 0.250984, 0.189006, 0.187738, 0.040131, -0.059881, 0.088224, 0.098349, -0.208942, -0.006939, -0.237635, 0.002181, 0.196506, -0.046064, -0.061384, -0.01464, 0.163176, -0.059193, 0.344772, 0.09033, 0.34842, -0.244071, -0.166824, -0.189441, 0.308951, -0.012382, -0.162966, 0.13962, -0.053583], [-0.062342, 0.519813, -0.147312, 0.49417, 0.086822, 0.018858, 0.157562, -0.073413, -0.140981, 0.105076, 0.20125, 0.132268, 0.14646, 0.059426, -0.028835, -0.009188, 0.016576, 0.148849, 0.107394, 0.156299, 0.111018, 0.260856, 0.191267, 0.217917, -0.170546, 0.122899, 0.169887, -0.117794, -0.339258, -0.277901, -0.098503, 0.071957, -0.02066, 0.00168, -0.03949, -0.111854, 0.153808, 0.013854, -0.06818, -0.119322, 0.365753, 0.324909, -0.019698, -0.119063, 0.007739, -0.112153, 0.083514, -0.137312, -0.124504, 0.158636, 0.275953, 0.153841, 0.19243, -0.04618, -0.135479, 0.14352, 0.185142, 0.019497, 0.132939, 0.132196, 0.169978, 0.152301, 0.201301, 0.14546], [-0.198334, -0.089978, -0.043276, -0.196743, 0.073268, 0.019169, -0.110951, 0.150066, -0.16634, -0.087838, -0.065458, 0.274953, -0.033388, 0.106385, -0.132887, -0.031001, -0.119269, -0.210581, 0.215036, 0.352374, -0.016251, -0.046269, 0.034192, -0.469792, 0.284853, 0.002517, -0.034936, -0.138243, -0.116582, 0.297486, 0.074471, 0.07859, 0.099857, -0.159722, 0.058939, -0.099678, 0.322344, 0.087581, 0.206878, 0.19608, 0.017998, -0.012732, -0.081917, 0.051829, -0.046752, 0.017004, 0.142524, 0.152078, 0.122264, -0.123515, -0.333097, -0.013406, -0.021003, -0.113604, -0.098701, 0.242904, 0.244623, -0.224311, 0.163127, 0.347847, -0.02294, -0.159418, 0.050864, -0.06264], [-0.138927, -0.302714, 0.07252, -0.26699, -0.177971, -0.226184, -0.044119, -0.112459, 0.124996, 0.016781, -0.00669, 0.02563, -0.107365, 0.161816, -0.013747, -0.16691, 0.021593, 0.144058, -0.236703, 0.001552, -0.116117, -0.070814, -0.314394, -0.142283, -0.000964, -0.069361, -0.145771, 0.04637, 0.148165, 0.221931, -0.082894, 0.016686, 0.078457, -0.04863, 0.051073, 0.160801, -0.114885, -0.149787, -0.269996, -0.141928, -0.123311, -0.061059, -0.03634, 0.257881, -0.163225, 0.097199, 0.036168, -0.190048, -0.099526, -0.061483, -0.09926, 0.01402, -0.148233, -0.250858, -0.300824, -0.064308, -0.154372, 0.030397, -0.014909, 0.084728, -0.004444, -0.069118, 0.074858, -0.042041], [-0.077264, -0.076642, 0.269801, 0.089039, -0.103857, -0.179718, -0.03052, -0.040788, 0.116359, -0.160906, -0.103722, 0.264708, 0.042243, 0.043794, -0.174819, -0.179894, 0.04446, 0.242294, -0.024553, -0.280013, 0.034146, 0.244267, -0.083975, 0.139263, -0.122169, -0.271218, -0.092725, -0.137512, -0.029464, -0.118905, -0.024502, 0.041883, -0.130004, -0.251744, -0.029448, 0.014323, -0.668918, -0.014397, -0.246837, -0.092352, 0.054324, -0.003011, -0.028094, -0.122727, -0.020218, 0.139237, -0.203831, -0.460274, -0.31818, -0.115627, 0.03398, -0.201807, -0.280236, -0.044543, -0.301391, -0.167128, -0.14084, 0.384625, -0.002726, -0.283125, -0.277655, 0.114855, -0.212148, -0.052875], [-0.104755, -0.017953, -0.018173, -0.063745, 0.022938, 0.071635, 0.012674, -0.33657, -0.129947, 0.286065, 0.206068, 0.100505, 0.032373, 0.04886, 0.286822, 0.125764, 0.355827, -0.031577, 0.074105, -0.139898, 0.095011, -0.208875, -0.030972, 0.186378, 0.202719, 0.257628, 0.138307, -0.157894, 0.115935, 0.132046, 0.47513, 0.099574, -0.039675, 0.449451, -0.099758, 0.057774, 0.552424, -0.085505, 0.134191, -0.268698, -0.017943, -0.428632, -0.27061, -0.311813, -0.052015, 0.207103, 0.002785, 0.3173, -0.016384, 0.596356, 0.257434, -0.28027, 0.19989, 0.459275, 0.200457, -0.301194, 0.30755, -0.000991, -0.101179, -0.12302, 0.497785, -0.064387, 0.158532, -0.148438], [0.314316, -0.231479, -0.036988, -0.54518, -0.148474, 0.163319, -0.173848, 0.163503, 0.173778, 0.171413, 0.271697, -0.098695, 0.227261, 0.041919, 0.046158, -0.362265, 0.167674, 0.004171, -0.078259, -0.277764, 0.018586, -0.023713, 0.202944, -0.375451, 0.209805, 0.178338, -0.288845, 0.052027, -0.046391, 0.244962, -0.108344, 0.180629, -0.112803, 0.189602, 0.128994, -0.150994, 0.37542, -0.088095, 0.401176, -0.158904, -0.073678, 0.042801, 0.015764, -0.00947, 0.041447, 0.202652, -0.117932, 0.318956, 0.220127, 0.24281, -0.227251, -0.139029, -0.056785, -0.004251, -0.076176, -0.205844, 0.119199, 0.196501, 0.089419, -0.127863, 0.070757, 0.071076, 0.317958, -0.104348], [-0.108535, -0.304289, -0.040655, -0.198683, -0.124513, 0.088052, -0.208193, 0.023979, -0.025629, 0.177207, 0.167175, -0.080341, 0.046268, -0.113251, -0.106283, -0.16235, -0.195833, 0.032765, 0.161593, -0.097458, -0.095428, -0.129072, -0.046361, -0.002834, 0.165832, -0.0384, 0.122963, 0.06546, -0.001455, 0.143987, 0.138628, 0.055154, 0.074596, 0.157606, 0.172566, 0.070144, 0.015695, 0.071704, 0.186403, -0.007313, -0.049666, -0.085255, -0.021152, 0.012849, 0.077352, 0.028934, -0.050873, -0.110692, 0.138648, -0.017747, 0.073253, 0.025365, 0.24259, -0.197521, 0.164505, -0.147598, 0.271528, -0.058003, -0.196508, 0.274154, 0.28164, 0.006533, -0.198514, 0.166059], [0.239063, 0.214075, 0.093252, 0.109033, 0.188365, -0.034336, 0.186734, -0.259844, 0.088894, -0.039287, -0.157351, -0.34048, 0.177215, -0.077917, 0.062227, 0.264001, 0.017638, 0.078697, -0.17344, -0.261712, 0.114346, -0.103841, 0.264659, 0.104794, -0.186265, -0.018948, -0.062423, -0.1244, 0.19833, -0.337524, -0.029176, 0.157457, -0.092257, 0.195612, 0.015079, -0.191401, 0.037665, 0.274125, 0.098489, 0.123496, 0.029988, 0.205942, 0.075542, -0.011082, 0.06991, 0.081846, 0.054256, 0.424377, -0.136179, 0.029445, 0.034804, -0.231018, -0.022757, -0.005401, -0.188767, -0.018615, -0.055414, 0.08947, 0.001799, -0.276814, -0.134099, -0.101789, 0.152337, -0.2211], [-0.148879, -0.122066, -0.081905, 0.148171, -0.199324, -0.056855, 0.14961, 0.175283, 0.341804, -0.193861, 0.132987, 0.123795, -0.059195, 0.016799, -0.020752, -0.089603, -0.016323, 0.068574, -0.052521, -0.102774, -0.004263, 0.173833, 0.129275, 0.150189, 0.041133, -0.072422, -0.001958, 0.086604, 0.088448, 0.001993, 0.065461, -0.0716, 0.00453, -0.001566, -0.069986, -0.038576, 0.004371, -0.011622, 0.066423, -0.28833, -0.04686, -0.132991, -0.09228, 0.130388, 0.09787, 0.039381, -0.127864, 0.071625, 0.066335, -0.019409, -0.005583, -0.10418, 0.048864, -0.064891, -0.116231, 0.161444, -0.24997, -0.024627, 0.053389, 0.060699, 0.034307, -0.200394, -0.025739, -0.137601], [0.086942, -0.089698, -0.111778, 0.108005, 0.0131, 0.212757, 0.106916, 0.001994, -0.035648, -0.017032, 0.211849, -0.013305, -0.099084, -0.252745, 0.210538, -0.067894, -0.149166, -0.110818, -0.076924, -0.181213, -0.051966, 0.364325, -0.066005, -0.261248, 0.218524, 0.038999, 0.203326, -0.157252, -0.19525, 0.109898, 0.055313, 0.192189, 0.195498, 0.052117, -0.188524, 0.057372, 0.124173, -0.056539, 0.148655, -0.046499, 0.118634, 0.105075, 0.255886, -0.148478, -0.05466, 0.045152, -0.060317, -0.245361, -0.113775, 0.076811, 0.208929, -0.142825, 0.107411, -0.140592, 0.289378, -0.231482, 0.341415, 0.048921, 0.166126, 0.042527, -0.00944, 0.092126, -0.037502, -0.124534], [0.090763, -0.137615, 0.001139, -0.078288, -0.020503, -0.117644, 0.029987, 0.145042, 0.177466, -0.120394, -0.092033, 0.171788, -0.036476, -0.260912, 0.153088, -0.022188, 0.05958, -0.035728, -0.022722, -0.238037, -0.251911, -0.031308, -0.152518, -0.140252, 0.192405, -0.082647, -0.114119, 0.169783, 0.110775, -0.057342, 0.144836, -0.179975, 0.108024, 0.023505, -0.198391, 0.065913, 0.052295, -0.227177, 0.100996, 0.072642, -0.008771, 0.120122, -0.416249, -0.066872, 0.065386, -0.0471, -0.162735, 0.106439, 0.207756, 0.163009, 0.051936, -0.290091, -0.026492, -0.263315, -0.077467, 0.050162, 0.026307, 0.077713, 0.035079, -0.16212, 0.019634, -0.035198, -0.087186, -0.100556]], [-0.134754, -0.060658, 0.215691, -0.180305, 0.234109, -0.1635, 0.034443, -0.04523, 0.08129, 0.11321, 0.07969, -0.146156, 0.044026, 0.043679, -0.002164, -0.133232, 0.053852, 0.208048, -0.196507, -0.386112, -0.099338, -0.079932, -0.072803, 0.055159, 0.165366, -0.075406, -0.166173, -0.019657, 0.150647, 0.038968, 0.120536, 0.015036, -0.25503, 0.152889, 0.02318, -0.098052, -0.058121, 0.094024, -0.052943, -0.148312, -0.267667, -0.159455, -0.093286, -0.042815, -0.0445, 0.14263, 0.039253, -0.139794, -0.046849, 0.123152, -0.034956, -0.125907, -0.189196, -0.08987, 0.164416, -0.156092, -0.037889, -0.061515, -0.105627, -0.355699, -0.164469, 0.094656, -0.197106, -0.073862]),
    ([[-0.026039, -0.031018, -0.211664, 0.031319, -0.207929, 0.064248, 0.006692, 0.075892, 0.012499, -0.188288, -0.075339, -0.064187, 0.016094, 0.037408, -0.087213, 0.043218, 0.008209, 0.072507, 0.105058, -0.039963, -0.120191, -0.209732, -0.022399, -0.099402, 0.108814, 0.12451, -0.13397, 0.09482, -0.20261, 0.030519, -0.179546, 0.178969], [0.274752, -0.1676, 0.114255, 0.108512, 0.08009, 0.091817, 0.262656, 0.069813, 0.357568, 0.171128, -0.0148, -0.159423, -0.125362, 0.254497, -0.109947, 0.302505, 0.284105, 0.224046, 0.190841, 0.088711, 0.069854, 0.05375, 0.109019, -0.026689, 0.316695, 0.228406, -0.105147, 0.377555, -0.11859, 0.307561, 0.003985, 0.081183], [-0.468467, 0.026268, 0.086401, 0.000606, -0.005401, -0.025483, -0.325127, 0.013532, -0.000864, 0.108332, -0.03743, 0.123212, -0.010165, 0.078005, -0.043664, 0.14379, 0.089909, -0.288355, 0.093317, 0.133663, 0.012326, -0.08058, 0.090421, 0.06252, -0.197323, -0.0166, 0.095014, -0.240481, -0.024762, -0.345019, -0.069698, 0.050638], [0.285491, -0.038107, 0.143363, 0.077079, 0.226409, -0.101349, 0.277216, -0.052159, 0.315979, 0.133777, 0.292288, 0.139652, 0.075953, -0.020075, 0.223446, 0.014639, 0.141896, 0.113432, 0.192361, 0.247737, 0.071243, 0.13402, 0.16176, 0.335235, 0.121712, 0.038095, 0.000558, 0.082778, 0.140592, 0.199051, 0.052649, 0.080936], [-0.043522, 0.011128, -0.065193, -0.154179, -0.021712, 0.155065, 0.272939, 0.015056, 0.16376, -0.217735, -0.139358, 0.068492, -0.032615, 0.012095, 0.139059, 0.166742, 0.107275, 0.01941, -0.00126, 0.065888, -0.078231, -0.116507, 0.114241, 0.08756, 0.144095, 0.067346, -0.062249, -0.015819, 0.043546, 0.113897, 0.03111, 0.155786], [-0.141794, 0.004291, 0.10942, 0.019908, 0.051661, -0.128492, -0.199749, -0.057811, -0.001576, -0.038487, 0.174227, -0.029171, 0.074187, 0.063939, 0.097586, -0.104567, -0.144505, -0.079089, 0.04513, 0.089414, 0.109598, -0.083606, -0.019405, 0.147997, -0.008056, 0.002361, 0.07381, 0.126454, -0.075168, -0.155212, -0.060453, 0.103473], [0.030782, 0.096247, -0.029894, -0.096129, 0.019249, 0.045506, -0.01431, 0.03779, 0.063433, 0.036268, 0.149161, -0.038094, 0.042396, 0.095368, -0.240936, 0.108993, 0.160499, 0.123224, 0.145525, 0.141226, -0.032237, -0.030689, 0.035855, -0.003341, 0.080773, 0.09572, 0.084667, -0.087682, -0.072565, 0.157797, -0.148237, 0.053507], [-0.109928, -0.029896, -0.005679, 0.004444, -0.008136, 0.117889, -0.310259, 0.089245, -0.096401, 0.098165, 0.045693, -0.036092, 0.006569, -0.075575, -0.17961, -0.026831, -0.110707, -0.088738, -0.120412, -0.009298, -0.137101, -0.053349, -0.162819, -0.077624, -0.298167, 0.030395, -0.065113, 0.002221, -0.077438, 0.177281, -0.06581, -0.118419], [0.0108, -0.072235, 0.067314, -0.14546, 0.00131, -0.075206, -0.017565, 0.056239, -0.035644, -0.222155, 0.054805, -0.184744, -0.045773, -0.068023, -0.358493, 0.068754, -0.118097, 0.172982, 0.035008, -0.056132, -0.206559, 0.079725, -0.055721, 0.090504, -0.026683, 0.039886, 0.118157, 0.049887, -0.210141, 0.118864, -0.179287, -0.264112], [-0.00135, 0.015399, 0.105388, -0.060319, 0.120861, 0.052651, 0.17355, 0.080772, -0.151168, -0.13931, 0.067765, -0.230283, 0.118181, 0.059333, 0.152265, 0.137771, 0.03183, 0.017987, -0.075916, -0.025715, 0.144932, 0.167676, -0.093915, -0.041413, 0.029343, 0.158927, 0.110981, -0.122703, 0.052102, 0.168486, -0.082353, 0.03651], [0.087967, 0.025007, 0.045116, 0.173394, 0.174192, 0.081124, 0.074032, -0.001084, 0.077004, 0.052838, -0.059277, -0.081708, 0.213315, 0.072733, 0.273969, 0.004557, 0.089451, 0.086368, 0.060336, 0.07152, 0.115306, 0.051429, -0.04516, -0.148381, -0.126153, 0.086292, 0.051562, -0.043924, 0.274349, 0.073887, 0.048995, 0.111543], [0.254072, 0.012758, 0.15995, -0.147042, -0.134077, 0.108355, 0.360832, 0.000136, -0.251246, -0.007123, -0.103544, -0.139939, -0.116559, -0.101816, -0.470106, 0.023535, -0.102319, -0.103226, -0.187188, 0.001189, -0.074387, -0.003908, 0.013412, 0.058928, 0.01313, 0.023953, -0.130238, -0.075791, -0.179937, -0.022476, -0.170507, -0.067475], [0.098931, -0.027548, -0.005492, 0.128181, -0.228956, 0.051705, 0.054497, 0.057611, 0.132175, 0.082815, -0.047914, 0.067359, -0.070605, -0.099907, -0.389123, -0.044817, 0.107926, 0.050608, 0.109596, -0.136662, 0.02022, -0.092619, -0.072626, -0.069371, -0.116283, 0.076205, -0.014341, -0.09382, -0.05676, 0.013896, -0.047146, -0.327255], [0.009003, 0.109125, 0.111135, -0.027115, -0.046975, 0.072754, 0.031907, 0.084496, -0.168023, -0.110226, 0.02603, -0.226148, -0.095047, -0.123104, -0.340469, -0.059373, -0.079164, -0.090308, -0.063668, 0.091317, 0.015364, 0.100346, -0.068946, -0.144312, 0.08377, 0.142203, 0.118546, 0.015987, -0.224331, 0.15026, 0.052974, 0.143516], [-0.012949, 0.100337, -0.097457, 0.10624, -0.032766, -0.11559, 0.056115, -0.163763, 0.061577, 0.073617, -0.00114, -0.033094, 0.096893, -0.081029, 0.004141, 0.124701, -0.070413, -0.096198, -0.002613, 0.042035, 0.070125, 0.057112, 0.056461, 0.182413, 0.049526, -0.072997, 0.018836, 0.095228, 0.146054, -0.216435, 0.14155, -0.011486], [0.06241, 0.089048, 0.048699, 0.017267, 0.006703, 0.093368, -0.226409, -0.014795, 0.062668, -0.007712, 0.081333, 0.17228, 0.116048, -0.005684, 0.025936, -0.118498, -0.062485, -0.140858, -0.174079, -0.007411, 0.117882, -0.027433, -0.220229, 0.015873, -0.163668, 0.044778, 0.140009, -0.150684, -0.062744, -0.130494, 0.120656, 0.025971], [0.008056, 0.021752, -0.014903, -0.070371, 0.122261, -0.042306, 0.055052, -0.117359, -0.069625, 0.023666, 0.119905, -0.116861, -0.058556, -0.07259, 0.092829, -0.057186, -0.054723, 0.006667, 0.132949, 0.001297, 0.020048, 0.09917, 0.036062, -0.107103, -0.088628, -0.069786, -0.030781, -0.024189, -0.114513, -0.300004, 0.121301, -0.016455], [0.0064, 0.043999, -0.1252, -0.086196, -0.177662, 0.106204, -0.08199, -0.061138, -0.110903, -0.324404, 0.015808, 0.185107, -0.138206, -0.122298, -0.271648, 0.141167, -0.056138, 0.077442, 0.178156, 0.01745, -0.00568, 0.052209, 0.041948, 0.100154, 0.010066, -0.044302, -0.170005, -0.039523, 0.024454, 0.109594, -0.139359, -0.096483], [0.108635, 0.003102, -0.061079, -0.019189, 0.001644, 0.005777, -0.115295, 0.02083, 0.141725, 0.185905, 0.046737, -0.082749, 0.075217, -0.00867, 0.18944, -0.092316, 0.026159, 0.017577, 0.103306, -0.307489, -0.163553, -0.003677, 0.110704, 0.01632, 0.113543, -0.078962, 0.166029, 0.106797, 0.104787, 0.094075, 0.137748, -0.253291], [-0.048076, -0.210354, -0.083761, -0.194831, -0.118113, -0.006439, 0.094157, 0.126304, 0.314887, -0.261437, -0.032452, -0.20664, -0.317494, 0.266458, -0.196199, -0.014682, 0.071687, 0.079192, 0.141686, 0.077307, -0.188092, 0.011771, 0.075315, 0.265325, 0.272785, -0.012069, -0.333658, 0.280137, -0.19129, 0.347722, -0.178149, -0.014109], [0.151415, -0.013086, -0.211791, -0.099958, -0.447699, 0.074506, 0.03302, 0.017586, -0.029423, 0.087913, -0.066559, -0.057651, 0.012696, -0.087818, 0.018266, 0.143645, 0.120768, 0.134331, 0.035735, -0.142086, -0.060119, -0.3173, 0.110317, 0.069749, -0.047598, -0.020357, -0.002391, -0.034392, -0.037491, 0.11131, -0.111499, 0.173715], [0.058751, -0.084161, -0.072684, -0.031531, -0.1785, 0.116495, -0.253957, -0.117859, 0.023513, -0.162965, -0.101427, 0.066491, 0.041409, -0.130877, -0.031647, -0.247755, -0.037207, 0.086524, 0.008625, 0.068782, -0.043909, -0.083843, -0.058298, 0.094008, -0.034203, -0.043704, 0.060649, -0.134168, -0.033266, 0.069273, 0.036782, -0.027021], [0.177651, -0.135887, 0.153704, -0.040112, -0.017386, 0.095616, 0.252212, 0.020158, 0.237377, -0.211953, -0.115442, -0.141346, -0.089498, 0.223284, -0.130253, 0.039977, 0.036616, 0.178022, 0.034133, 0.014752, -0.067333, 0.068847, 0.06664, 0.040836, 0.291809, 0.134865, -0.052379, 0.355667, 0.034016, 0.215985, -0.073591, -0.486596], [0.046033, 0.06685, 0.136699, -0.121387, 0.144483, -0.000529, 0.11184, 0.101704, -0.006009, 0.028622, 0.10418, -0.149547, -0.011907, 0.165234, 0.175548, 0.204035, 0.050341, 0.188203, 0.045623, 0.032373, -0.002891, 0.149344, 0.157287, -0.057893, 0.096428, 0.276535, -0.032187, 0.132089, 0.103634, 0.171841, 0.073251, -0.013446], [-0.00448, 0.172535, -0.012281, 0.165257, -0.083309, 0.03474, -0.049182, -0.072097, -0.050351, 0.002372, -0.032929, 0.035126, 0.133219, -0.252413, -0.122694, -0.160617, -0.221164, -0.144285, 0.016531, -0.196994, 0.087514, 0.005171, -0.173946, 0.029967, 0.058541, 0.042635, 0.167877, -0.064823, 0.055719, -0.229353, 0.135679, 0.111409], [0.120282, -0.002733, 0.062746, -0.00164, 0.090365, 0.0492, 0.192265, 0.064711, -0.049651, -0.008523, 0.160358, -0.127065, 0.114354, -0.168071, 0.066859, 0.081188, 0.083775, 0.079183, -0.040908, -0.040972, 0.017665, 0.110605, 0.056576, 0.109165, -0.033926, -0.025566, 0.129748, 0.292066, 0.023248, 0.266675, 0.004165, 0.075845], [-0.140761, 0.058209, 0.084648, -0.010391, -0.076095, 0.007641, 0.220008, -0.165546, 0.096755, 0.046331, 0.116039, -0.04399, 0.150716, -0.106761, -0.046454, 0.136847, -0.05791, 0.135398, 0.035427, 0.15989, 0.143806, 0.050108, -0.071934, 0.171354, 0.122551, -0.153598, -0.073186, 0.008043, 0.025717, 0.021083, 0.158348, 0.205675], [-0.051721, -0.099718, 0.021253, 0.020664, 0.024107, 0.089525, -0.100252, 0.071944, 0.144881, 0.091748, 0.004214, -0.171127, 0.024787, 0.09929, 0.203438, -0.205768, -0.080371, 0.035332, 0.009121, -0.197097, -0.019517, 0.032229, 0.078576, -0.738094, -0.074286, -0.029404, -0.121671, -0.061807, 0.077776, 0.007327, 0.106244, -0.065314], [-0.018496, 0.064519, 0.063161, -0.067488, 0.012082, -0.125768, -0.299688, -0.05863, -0.206818, -0.139448, 0.005298, 0.047959, 0.100946, -0.138972, 0.149698, 0.006939, 0.006717, -0.213624, -0.323005, 0.033318, 0.044967, 0.044692, 0.00882, -0.141615, -0.296219, -0.05545, -0.062592, -0.12951, 0.020833, -0.267564, 0.017868, -0.031902], [0.015381, 0.099347, 0.045651, 0.038538, -0.041537, -0.006043, -0.156797, 0.053375, 0.090599, 0.075113, -0.15732, 0.148718, 0.065304, -0.06017, 0.023738, -0.219883, -0.19017, -0.22991, -0.102525, 0.00535, 0.140796, 0.104969, -0.10396, -0.024524, -0.054522, -0.081625, -0.113714, 0.029299, 0.035939, -0.03699, 0.150458, -0.051996], [-0.135568, 0.083893, 0.100089, 0.114097, -0.00871, 0.054905, -0.082711, 0.036356, -0.116344, 0.170306, 0.034517, -0.103077, 0.13956, 0.111155, -0.047536, 0.091478, -0.076494, 0.077312, 0.005786, 0.075155, 0.181721, 0.0881, -0.087136, -0.025059, 0.017798, 0.025964, -0.019208, -0.159067, -0.12725, 0.202923, 0.125855, 0.021614], [0.028478, 0.120239, -0.060896, -0.078128, 0.036041, 0.100959, 0.021809, -0.082072, 0.017457, 0.027005, -0.057575, -0.049588, -0.05809, -0.137331, -0.104513, -0.05701, 0.058007, -0.096538, 0.092583, 0.0019, 0.126947, 0.134401, -0.004404, 0.252503, -0.184833, -0.013525, 0.09643, -0.017725, -0.029448, -0.230809, 0.001868, 0.023364], [-0.126677, -0.050031, -0.001028, 0.092143, 0.006962, -0.066896, -0.163756, -0.06091, -0.197036, -0.091036, -0.242783, -0.032803, 0.05839, -0.205695, -0.429121, -0.05393, -0.251399, -0.123268, -0.200377, -0.096191, -0.086854, -0.004099, -0.315958, 0.093827, -0.260099, -0.141189, 0.031519, 0.017376, -0.044101, -0.107875, 0.13451, 0.168574], [-0.055119, 0.025239, 0.127275, 0.071002, 0.129807, 0.144222, -0.037681, 0.12368, 0.031814, 0.022873, -0.000999, -0.25481, 0.078102, -0.119058, 0.021523, 0.097703, -0.027734, 0.094702, -0.1689, -0.075783, -0.032665, -0.03331, 0.015086, 0.048597, -0.046174, 0.123906, 0.134156, 0.028527, 0.112182, 0.089442, 0.139329, -0.064486], [0.006328, 0.005893, -0.09509, -0.150264, -0.155077, -0.018829, -0.005229, 0.073627, -0.15062, -0.020293, -0.057897, 0.065054, 0.002398, -0.031654, -0.509528, -0.140677, 0.021529, -0.05911, -0.114149, -0.088717, -0.072158, 0.139758, -0.012484, -0.118287, -0.11833, -0.053252, 0.043338, -0.0737, -0.274881, -0.10169, -0.000518, -0.221632], [-0.133922, 0.159483, 0.048121, -0.058715, -0.124017, -0.159942, -0.098102, -0.234001, -0.242123, -0.0151, 0.094913, -0.274833, 0.053819, 0.035001, 0.011718, -0.132443, -0.105982, -0.288339, -0.220622, 0.03986, -0.017082, 0.03404, -0.061591, -0.096302, -0.076512, 0.024055, 0.015511, 0.015967, 0.057204, -0.327885, 0.128284, -0.043652], [0.078618, 0.170558, 0.203238, 0.134093, 0.018177, 0.13492, 0.187326, 0.115916, -0.294096, 0.000885, -0.225206, 0.050061, 0.169123, 0.14095, 0.36924, 0.104096, 0.14067, 0.017424, 0.096637, -0.13569, 0.208943, 0.193852, 0.051535, -0.146956, 0.179359, -0.026325, 0.101473, 0.07907, 0.366934, 0.054547, 0.208303, -0.257779], [-0.012856, 0.112092, -0.285143, -0.045226, -0.086211, 0.123243, 0.139054, 0.200994, -0.038597, -0.092822, 0.079118, -0.020861, -0.056322, 0.10474, 0.143231, 0.15838, -0.079756, 0.139861, 0.032546, 0.125057, 0.037198, -0.036338, 0.078381, -0.142952, 0.070016, 0.122609, 0.096096, 0.078574, 0.197045, -0.047002, 0.004153, -0.004082], [0.096743, 0.161245, -0.135277, 0.094528, 0.219095, 0.091747, -0.017617, 0.009645, -0.051528, -0.066152, -0.161899, -0.069594, 0.035106, 0.045631, 0.336425, -0.082847, 0.097433, 0.096696, 0.028987, 0.042424, -0.031956, 0.036718, 0.114227, -0.196157, 0.042427, 0.066564, 0.098256, -0.184299, 0.09482, -0.12517, 0.111699, 0.149287], [-0.179562, -0.071123, -0.005444, -0.050424, -0.162224, -0.02241, 0.018764, -0.07466, -0.114637, -0.156508, -0.249999, -0.119834, 0.010979, -0.054789, -0.114947, 0.046355, 0.028407, -0.190982, -0.125293, -0.104985, -0.109551, -0.209209, -0.148536, 0.222029, 0.129293, -0.071447, -0.187616, 0.022651, -0.19322, -0.015576, -0.14831, 0.205015], [0.096269, -0.103913, -0.124452, -0.12801, -0.26914, -0.031078, 0.044806, 0.077256, 0.098319, 0.077876, -0.03993, 0.100318, -0.140614, 0.047084, -0.035093, 0.2179, 0.004099, 0.044487, 0.048955, -0.021831, -0.008068, -0.213281, 0.147385, 0.223634, -0.048583, 0.12429, 0.005704, 0.058807, -0.107175, 0.021455, -0.017386, -0.250342], [0.081956, -0.249326, -0.151705, -0.096935, -0.175701, -0.045115, -0.115218, -0.011443, -0.190167, -0.297323, -0.057375, 0.153752, -0.297051, 0.232112, -0.407002, 0.002863, 0.069611, -0.010179, -0.013386, -0.326489, -0.100045, 0.087379, -0.071812, 0.184317, 0.030492, -0.031439, -0.023998, 0.122648, -0.127232, 0.097028, -0.197336, -0.351327], [-0.167857, -0.001258, -0.036807, 0.122157, -0.056495, 0.084796, 0.123614, 0.10238, -0.063002, 0.028651, 0.141696, -0.139536, -0.054974, -0.056986, 0.056586, 0.044907, -0.123643, 0.048684, -0.137605, 0.019827, 0.015839, -0.032504, 0.009339, 0.049436, 0.067051, 0.079926, 0.156438, 0.044347, -0.078381, -0.023632, -0.060134, -0.003689], [-0.145665, -0.108709, 0.091592, 0.02058, -0.015736, -0.152438, -0.085606, -0.221633, -0.158051, -0.028157, -0.032795, 0.014119, -0.033967, -0.230047, 0.038372, -0.189207, 0.053912, -0.2683, -0.127569, -0.013777, -0.110787, 0.143698, 0.000572, 0.012836, -0.222754, 0.123601, -0.032833, -0.28959, -0.039429, -0.226076, -0.074534, 0.092185], [-0.141917, 0.125691, -0.006252, -0.066685, 0.061401, -0.228111, -0.152393, -0.050398, 0.239083, 0.122456, 0.066647, -0.006437, -0.110856, -0.038692, 0.028612, -0.032879, -0.131077, -0.14871, 0.103934, 0.128055, -0.028121, -0.048885, -0.043652, 0.15219, 0.011725, -0.31486, -0.065499, -0.191309, -0.021641, -0.219724, 0.162291, -0.119947], [-0.232739, 0.084696, 0.039191, 0.098865, -0.028994, -0.037278, -0.360229, -0.022283, -0.216449, 0.138597, -0.024792, -0.221385, 0.03757, -0.026088, 0.060716, -0.224003, -0.168628, -0.020051, -0.035444, -0.045084, 0.057142, 0.011256, -0.063199, 0.142647, -0.038265, 0.105588, 0.069295, -0.047751, 0.087112, -0.207901, -0.046883, 0.065032], [-0.025254, 0.030504, -0.062795, -0.12569, -0.130357, 0.065325, -0.106867, 0.004437, 0.053098, -0.04063, -0.118059, 0.039829, 0.016795, 0.137404, -0.209701, 0.058519, 0.123451, 0.038914, 0.027262, -0.278967, -0.086721, 0.118024, -0.085373, 0.227514, 0.006725, 0.081291, -0.113623, 0.069797, -0.248849, 0.030927, 0.063339, -0.191862], [0.161877, 0.008353, 0.108905, 0.135988, 0.102273, 0.073489, 0.236508, 0.163578, 0.125497, 0.180773, 0.029637, 0.260337, 0.029799, 0.059555, 0.039897, -0.01367, 0.186883, 0.192696, 0.084623, 0.012834, 0.030637, 0.217569, 0.152972, -0.647825, -0.034642, 0.102744, 0.220761, 0.100995, 0.202221, 0.253251, 0.107461, 0.111372], [-0.162451, 0.030238, -0.129577, 0.006837, -0.17219, -0.041704, 0.05772, -0.100696, -0.087043, 0.217418, -0.048257, -0.074286, 0.1095, -0.224255, 0.012352, 0.086927, -0.009765, -0.205489, -0.078599, -0.402047, -0.229559, 0.100832, -0.042787, -0.026449, 0.068495, 0.159216, 0.022276, -0.324462, -0.053952, -0.580426, 0.002593, -0.236696], [-0.212779, 0.033965, 0.112131, 0.141467, 0.141038, 0.103736, 0.203451, -0.18119, -0.196448, 0.174991, -0.144331, -0.149623, 0.070742, -0.241821, 0.188392, 0.050095, 0.105016, -0.050476, -0.122338, -0.008113, 0.110197, 0.191057, -0.123725, 0.043683, 0.129674, 0.069777, 0.118874, -0.047717, 0.151643, 0.022513, 0.004709, -0.206267], [0.127467, 0.004263, 0.057162, 0.125145, 0.165533, 0.151606, -0.047253, -0.129078, -0.131895, 0.126171, 0.07765, -0.067746, 0.075812, -0.082463, 0.323446, 0.118067, 0.211581, 0.064743, 0.094269, 0.070288, -0.041126, -0.017992, 0.209517, -0.106499, 0.0282, 0.118052, 0.030002, -0.063331, 0.262057, -0.081544, -0.041119, 0.219178], [-0.10655, 0.040494, -0.076515, 0.120337, -0.156075, 0.031177, -0.082108, -0.047037, -0.120412, 0.163227, 0.100539, 0.026377, -0.124064, 0.055409, -0.604589, 0.005871, 0.138255, 0.132613, -0.009212, -0.036374, -0.117572, -0.145444, -0.082328, 0.082305, 0.031616, 0.037688, 0.141583, 0.049101, -0.247257, 0.019052, -0.017183, 0.112214], [0.001796, 0.073575, 0.120549, -0.05532, 0.071205, -0.097331, -0.229285, -0.03798, 0.036758, 0.10399, 0.169808, -0.065865, -0.073956, -0.016532, 0.092852, -0.276707, 0.003577, -0.133689, 0.062879, 0.03342, 0.069061, -0.010938, 0.066699, -0.044682, -0.088196, -0.180033, -0.001401, 0.07758, 0.013658, -0.085976, 0.12962, -0.023876], [0.229575, -0.050401, 0.068553, -0.014716, 0.05004, 0.018517, -0.05481, 0.090429, 0.013837, -0.170878, -0.061823, 0.054914, -0.042954, 0.090986, 0.081899, 0.084825, 0.191986, 0.040732, 0.285282, -0.021784, 0.156384, 0.084461, -0.084564, -0.281763, 0.113616, 0.025294, -0.088258, 0.048763, 0.139803, 0.211269, -0.018448, -0.142752], [-0.024858, 0.095504, 0.011346, -0.010204, 0.058824, -0.14919, -0.038858, -0.052535, 0.092247, 0.007127, 0.006551, 0.087412, 0.068343, 0.043869, 0.171593, -0.080175, -0.003108, 0.012527, -0.008271, 0.045301, 0.167059, 0.041776, 0.050859, -0.046149, -0.104387, -0.104423, -0.02699, 0.03225, 0.049902, -0.083697, 0.136474, -0.014797], [-0.14454, -0.106083, -0.079105, -0.07979, -0.061854, -0.223626, -0.31992, -0.102846, 0.338745, 0.020332, 0.003588, 0.138578, -0.15504, 0.114702, -0.059107, -0.283113, -0.024336, -0.374597, 0.24503, -0.206993, 0.020872, 0.053011, 0.027183, 0.214782, 0.099092, -0.328285, 0.022009, -0.038841, -0.034314, -0.346984, -0.090606, -0.259254], [-0.247971, 0.271028, 0.076354, 0.23658, -0.134405, -0.190794, -0.116657, -0.07431, -0.058664, -0.057113, -0.182916, -0.200319, 0.309201, 0.069927, -0.028226, 0.045285, -0.09238, -0.163597, -0.048715, -0.161344, 0.026644, 0.151314, -0.062077, 0.032357, -0.182487, -0.100248, 0.065241, 0.019808, 0.11766, -0.092496, 0.193656, 0.06048], [0.10879, 0.073504, -0.032821, -0.094314, 0.093446, 0.079461, 0.104525, -0.058505, -0.215335, -0.271392, -0.171217, -0.038945, -0.140115, 0.088878, -0.120608, 0.08154, 0.025799, 0.167895, -0.122366, -0.030733, -0.045134, 0.064357, 0.018421, 0.009422, -0.075757, 0.170978, -0.149942, 0.158948, -0.071036, 0.004063, -0.076389, -0.079942], [-0.290579, 0.021011, -0.048998, -0.149718, 0.05732, 0.029433, -0.19857, -0.03342, -0.18436, 0.075572, -0.118411, -0.223825, -0.111632, -0.040708, -0.022146, -0.148256, -0.03945, -0.125694, -0.045656, -0.236536, 0.036013, 0.101402, -0.037874, 0.201228, 0.037227, -0.099511, 0.002818, -0.007971, -0.115496, -0.213868, -0.060124, -0.120234], [0.162041, 0.131402, 0.085972, 0.036437, 0.26917, 0.012786, 0.142106, -0.06105, -0.241431, 0.239704, 0.017273, 0.026404, -0.064722, 0.031346, -0.017438, 0.032287, 0.026481, 0.083129, -0.107626, 0.056479, 0.021285, 0.114029, -0.080275, -0.473287, 0.019319, 0.059323, -0.062194, 0.109982, -0.094009, 0.141578, 0.092882, 0.076372], [0.008, -0.055321, 0.135705, -0.071858, 0.100877, 0.080494, -0.049493, 0.084807, -0.168478, 0.160873, 0.012107, -0.12023, 0.152916, 0.020706, 0.181071, -0.166042, 0.03058, 0.142959, -0.012854, 0.179989, 0.025403, 0.070478, -0.087587, 0.059532, 0.138196, -0.216094, 0.021288, -0.091308, 0.106757, 0.108381, 0.011144, 0.098117], [-0.092809, -0.03078, 0.107915, 0.085127, -0.078616, -0.047318, 0.018218, 0.090834, -0.091939, -0.017363, -0.034245, -0.040423, -0.006152, -0.110896, -0.32458, 0.014688, -0.09055, -0.025867, -0.019237, -0.123966, 0.040339, 0.090896, 0.064336, -0.054158, -0.098856, 0.102217, -0.040926, 0.070088, -0.032345, -0.043561, -0.0167, -0.087754], [0.167345, -0.110039, -0.06296, 0.069225, -0.151076, -0.002425, 0.158773, 0.005571, 0.030177, -0.061365, -0.027317, -0.060716, 0.080865, -0.005465, 0.244567, -0.023715, -0.012383, 0.106648, -0.078138, 0.126736, -0.154411, -0.131392, 0.153526, -0.252624, 0.143081, 0.089007, 0.074761, 0.051078, 0.022587, 0.148729, -0.018469, 0.113671], [-0.021095, -0.104356, -0.057081, -0.051176, -0.060369, 0.049584, 0.163237, 0.052993, -0.134286, 0.116471, 0.077323, -0.050479, -0.00058, -0.008205, -0.157939, -0.11783, 0.100017, -0.033436, -0.057976, 0.13006, -0.08454, -0.120407, -0.095522, -0.006183, 0.104973, 0.10281, 0.01525, -0.09249, -0.153073, -0.101945, 0.011725, 0.032486]], [-0.114045, 0.137853, 0.109728, 0.097623, 0.025072, 0.083754, 0.06007, 0.08029, -0.032797, -0.139059, -0.011901, -0.122669, 0.034196, 0.038636, -0.040222, 0.173209, 0.112167, 0.033982, -0.150751, 0.001901, 0.147205, 0.092193, 0.064276, -0.07801, -0.192092, 0.152386, 0.053122, 0.034046, -0.069861, -0.119843, 0.119231, 0.015518]),
    ([[0.288109], [-0.149277], [-0.111151], [-0.032841], [-0.157199], [0.121927], [0.353174], [0.186097], [0.202194], [-0.300989], [-0.152309], [0.071518], [-0.102213], [0.143652], [-0.275341], [0.263899], [0.218614], [0.258674], [0.10595], [-0.222187], [-0.087537], [-0.148314], [0.109677], [0.127173], [0.241665], [0.090364], [-0.075646], [0.143296], [-0.144941], [0.246674], [-0.169445], [-0.097758]], [0.10115]),
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
