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
    ([[0.09132, -0.190108, 0.076983, 0.265125, -0.320825, 0.635826, -0.022921, -0.185406, -0.06138, 0.033235, 0.191594, 0.006146, 0.418388, -0.15123, 0.013711, -0.27957, -0.042871, -0.028432, -0.119462, 0.00315, -0.117081, 0.179929, 0.202461, -0.126169, 0.05771, 0.008235, 0.06223, 0.074499, 0.063745, -0.083601, 0.051062, 0.249052], [0.078251, -0.059677, 0.312914, -0.039793, -0.270618, 0.090112, -0.078534, -0.195279, -0.145093, -0.208712, -0.313175, 0.036804, 0.115159, 0.021842, -0.106944, 0.110484, -0.062664, 0.04494, 0.071262, -0.055085, 0.051254, 0.02442, -0.083392, 0.022862, 0.096475, 0.047176, 0.030601, -0.227837, -0.295698, 0.080948, -0.031945, 0.297263], [-0.133112, 0.143875, 0.212871, 0.038063, 0.074182, -0.129153, 0.070882, -0.251344, -0.170713, 0.151637, -0.315725, 0.036403, 0.143626, 0.102596, 0.032443, 0.298444, -0.18609, 0.379238, -0.180632, 0.157388, -0.07804, 0.028457, -0.252912, 0.150618, -0.196674, 0.067529, 0.039211, -0.016475, 0.00442, 0.052301, -0.182756, -0.13993], [-0.088203, -0.093663, -0.178457, -0.00497, 0.215337, 0.055925, 0.043313, 0.234918, 0.308803, -0.100087, -0.1742, 0.037136, -0.231369, 0.324175, 0.045833, 0.288049, -0.018233, 0.137356, 0.133665, -0.160925, -0.099798, 0.22908, -0.135741, 0.257938, -0.118457, -0.349361, -0.181961, 0.080843, -0.008134, 0.009133, 0.237741, 0.028531], [-0.420055, 0.294551, -0.03097, 0.165814, 0.061899, -0.051678, 0.495502, -0.031608, 0.380831, -0.132146, 0.172899, 0.258921, -0.304602, 0.047987, 0.047323, 0.423214, 0.147096, -0.311536, -0.250344, 0.214797, 0.059719, 0.040469, -0.246553, -0.225061, 0.313765, -0.110068, -0.135466, 0.210861, 0.19901, -0.145581, 0.32272, -0.127741], [-0.253087, 0.044063, 0.260835, -0.111024, 0.20392, 0.070607, -0.002624, -0.107549, -0.034356, 0.194501, -0.195182, -0.069755, 0.043236, 0.339251, 0.078873, 0.213624, -0.2277, 0.288277, -0.233014, 0.063833, 0.248821, -0.027729, 0.037527, -0.197087, 0.030784, 0.039582, -0.249575, 0.07748, -0.061591, -0.140856, 0.177506, 0.086391], [0.206953, 0.097515, -0.25877, -0.199673, 0.567999, 0.191362, 0.252022, -0.192979, -0.023289, -0.559679, -0.108658, -0.185791, 0.27094, 0.323637, 0.143717, -0.01444, 0.132706, -0.143641, -0.053254, 0.258134, -0.033423, -0.292786, -0.542665, 0.096325, 0.287624, -0.19292, -0.072291, -0.36944, 0.138962, -0.105523, -0.247225, 0.004998], [0.227217, 0.041646, -0.011797, -0.441908, -0.183345, -0.107835, 0.01494, 0.011277, -0.032789, -0.01919, -0.065818, 0.253332, -0.025027, 0.268786, -0.016199, -0.01956, -0.071422, 0.128077, -0.023267, 0.09166, 0.104989, -0.09605, 0.296938, -0.097281, 0.166388, 0.017275, -0.209862, 0.145244, -0.024004, 0.025239, -0.12708, 0.055771], [-0.063647, 0.403863, 0.427496, 0.155948, -0.004533, 0.192454, -0.140171, -0.210512, 0.232271, -0.530785, -0.146914, 0.249927, 0.182612, 0.184692, -0.110327, 0.087665, -0.339197, 0.29635, 0.007019, -0.010857, -0.479249, 0.374076, -0.208494, -0.095576, 0.490954, 0.178661, -0.061082, 0.234962, -0.132238, -0.002476, 0.199535, -0.055526], [0.205093, 0.227666, -0.516365, 0.238689, 0.278326, 0.184444, 0.072484, -0.063463, 0.17198, -0.031523, 0.04996, 0.145544, -0.031636, 0.15285, 0.143684, -0.393461, -0.274219, -0.234295, -0.446387, -0.344065, -0.01967, -0.033124, 0.134248, 0.153663, -0.189088, 0.089881, 0.027419, -0.208048, 0.202244, -0.288373, -0.173806, 0.226917], [-0.044503, -0.142231, 0.062598, 0.126914, 0.226346, 0.058444, -0.200517, -0.225113, -0.255806, -0.105036, -0.147183, 0.002089, -0.183364, -0.129518, 0.081186, 0.041501, -0.146901, -0.032409, 0.086143, -0.083929, 0.060556, 0.290857, -0.034964, 0.081129, 0.079713, -0.108271, -0.014693, -0.006857, 0.035617, -0.103028, -0.129191, 0.014641], [-0.384284, 0.411454, 0.366135, 0.48038, -0.390362, -0.435094, -0.372971, -0.106318, 0.071962, -0.086807, -0.031014, -0.12733, 0.260864, 0.290418, 0.279877, 0.101769, -0.709916, -0.323849, 0.078151, 0.083082, -0.196899, -0.209513, -0.102015, -0.26167, 0.044313, -0.289483, 0.30592, 0.301544, 0.185012, 0.127472, 0.35545, 0.034805], [0.173756, 0.018373, -0.040265, -0.153246, 0.177977, 0.131461, 0.097906, 0.159327, -0.13373, -0.244325, -0.211729, -0.262426, -0.130937, -0.072415, 0.083916, -0.065216, 0.047608, 0.174281, 0.075327, 0.235128, 0.028608, -0.010975, -0.331709, -0.005228, 0.089079, 0.017464, 0.364177, 0.301947, 0.138741, 0.031343, -0.062753, -0.075167], [-0.228858, -0.133797, -0.14632, -0.146342, -0.048047, -0.072028, 0.170326, -0.002499, -0.11061, -0.085031, -0.039891, -0.354068, 0.35971, -0.347871, -0.081731, -0.016731, 0.157114, 0.194942, -0.058908, 0.028938, 0.330291, 0.155785, -0.213574, 0.050982, 0.208204, -0.029346, 0.144009, 0.21284, 0.148451, -0.05992, 0.295429, -0.044651], [0.018813, -0.219547, 0.205998, 0.151217, 0.236881, 0.232038, -0.221434, -0.29984, -0.250577, -0.21502, 0.225575, -0.056102, -0.080211, -0.01431, 0.117077, -0.058768, 0.078513, -0.155247, 0.054605, -0.16172, 0.064859, 0.407734, -0.130209, 0.270839, 0.228948, -0.205454, 0.263167, -0.263891, -0.37193, -0.326504, 0.056808, 0.051139], [0.024225, 0.072376, 0.018802, 0.218094, 0.026361, -0.272164, -0.168329, -0.175963, -0.043163, 0.104823, 0.178498, 0.158169, -0.020771, -0.085669, -0.136444, -0.063259, 0.244891, 0.157754, 0.004533, -0.141237, 0.164511, 0.100877, -0.086903, -0.183742, 0.254297, 0.392058, -0.158721, -0.002937, -0.412453, -0.366539, 0.242215, -0.068163], [-0.031836, 0.080441, -0.408008, 0.035291, -0.208255, 0.091291, -0.079845, 0.179999, 0.524824, 0.421226, -0.222064, -0.366079, -0.172751, -0.172425, -0.452637, 0.013134, -0.169185, -0.106984, 0.01982, 0.302169, -0.839945, 0.70433, 0.508504, -0.417803, -0.276232, 0.164621, -0.04286, 0.323177, 0.287191, 0.001037, 0.125189, 0.092635], [0.078951, -0.25395, 0.255084, -0.167673, 0.073565, 0.110987, 0.116053, -0.065076, -0.194427, 0.032659, -0.384076, 0.00996, -0.078041, 0.193785, 0.23329, 0.188319, -0.203678, 0.234177, -0.390175, -0.455871, -0.069874, -0.105293, 0.464852, 0.132902, 0.149676, -0.110912, -0.030486, -0.021809, 0.325363, 0.447424, 0.13605, -0.368449], [-0.202225, -0.018613, -0.113704, 0.083933, -0.181928, 0.015675, 0.099298, 0.233474, 0.000433, -0.001038, -0.042884, 0.287535, -0.019838, 0.057516, 0.001714, 0.016267, 0.135863, -0.051517, 0.073733, -0.045302, 0.279599, -0.145701, 0.137424, -0.021401, 0.056013, 0.053556, 0.007068, 0.024598, -0.028171, -0.001626, 0.136713, -0.010802], [-0.314821, 0.46955, 0.042493, -0.328312, -0.158075, 0.065943, 0.013318, -0.03138, 0.169661, -0.069204, -0.126699, 0.115986, 0.028155, 0.37954, -0.432807, -0.081113, -0.129772, -0.17435, -0.36271, -0.080215, 0.164711, -0.181882, 0.075897, 0.035572, 0.285859, 0.077944, 0.070161, 0.19463, -0.119485, -0.013206, -0.121628, 0.017022], [0.166722, 0.086694, 0.072931, 0.152838, 0.473214, -0.059451, -0.287099, 0.155988, -0.182706, 0.184005, -0.111746, -0.339245, -0.196252, -0.095945, -0.174364, -0.161539, -0.193828, 0.301155, -0.165384, 0.008734, -0.053386, -0.342866, -0.025027, -0.081592, -0.089763, 0.480497, 0.333591, -0.180102, -0.186236, -0.100679, 0.169625, -0.198867], [-0.093415, -0.007403, 0.13326, 0.068921, -0.168086, 0.098968, 0.002512, -0.127034, 0.107531, 0.536442, 0.309171, 0.110693, -0.122819, 0.126303, -0.335568, -0.096091, -0.066779, -0.088221, -0.058159, 0.554177, 0.497729, 0.013548, 0.034477, -0.111138, 0.146302, -0.248386, 0.002734, -0.416166, 0.076033, -0.007908, 0.239325, -0.091518]], [-0.121827, -0.025671, 0.136468, -0.424383, 0.128293, 0.046588, 0.049429, -0.137377, -0.079082, -0.192166, -0.069462, 0.233591, -0.337955, -0.505603, 0.00207, -0.197113, -0.204985, 0.089292, 0.139886, 0.090104, -0.181624, 0.030389, 0.031614, -0.003125, 0.173934, 0.0815, 0.48797, 0.304785, -0.132288, 0.220848, -0.00815, -0.037922]),
    ([[-0.009973, 0.175224, -0.394581, -0.263136, 0.171071, 0.154546, 0.077956, 0.309497, -0.237586, 0.210205, -0.23518, -0.0675, -0.022476, 0.079254, 0.215822, 0.210222], [0.042264, -0.358642, 0.143185, 0.118595, 0.176404, -0.323061, -0.145244, 0.137117, 0.201619, -0.02394, 0.082421, -0.198344, -0.350983, 0.016296, -0.013811, -0.072887], [-0.169073, -0.048098, -0.08212, -0.344501, 0.344238, 0.043594, -0.240977, 0.007446, 0.225504, -0.173465, -0.346553, 0.215952, -0.025625, -0.06293, 0.270694, 0.06939], [-0.604387, -0.099296, -0.026639, -0.026123, 0.151392, 0.036575, -0.251243, 0.222031, 0.104988, -0.309346, 0.184914, -0.10849, -0.014453, -0.01162, 0.231386, 0.172282], [0.020175, 0.313811, 0.234103, 0.356755, -0.164403, 0.269896, 0.31768, -0.005475, 0.213665, 0.01005, -0.208206, 0.26208, -0.07075, -0.076181, -0.392676, -0.225598], [0.129371, 0.278901, 0.189863, 0.187554, 0.154382, 0.257786, 0.170271, 0.031379, 0.147766, 0.194459, -0.279497, 0.15374, 0.374822, -0.030281, -0.011057, -0.056735], [0.051551, -0.012317, -0.370673, -0.062677, -0.00683, -0.152058, -0.323355, -0.064554, 0.019349, -0.483401, 0.387245, -0.042805, 0.092619, -0.576359, -0.273302, -0.119537], [0.065026, 0.108598, -0.132346, 0.214925, -0.174203, 0.281005, 0.011941, 0.17563, -0.146863, 0.164825, -0.19465, 0.098051, -0.193532, -0.107777, 0.158622, 0.042755], [0.229763, 0.079589, 0.211431, 0.348835, 0.016771, 0.059389, -0.228258, 0.290296, 0.035512, 0.343236, 0.120013, 0.276746, -0.223112, 0.25311, -0.308005, -0.12441], [0.158123, -0.264933, -0.53381, 0.078529, -0.044662, 0.089587, 0.160062, -0.256818, -0.325047, -0.108191, -0.380939, -0.070669, 0.201326, 0.319758, -0.122615, 0.065025], [0.151907, 0.102687, 0.36696, -0.0859, 0.197998, 0.046685, 0.09327, -0.14207, 0.315751, -0.114125, -0.005426, 0.15096, 0.141418, -0.162296, -0.175888, 0.065729], [0.004117, -0.020923, -0.15841, -0.124502, -0.007736, -0.083875, -0.376247, -0.058276, 0.324315, -0.068501, 0.105742, 0.063206, -0.505452, -0.13735, -0.182723, 0.06738], [0.052358, 0.081275, 0.281196, 0.247337, -0.02001, 0.004895, 0.148955, -0.104238, 0.172622, 0.009311, -0.372183, 0.321315, 0.265533, 0.140579, 0.231373, -0.079161], [0.167289, -0.507739, 0.154117, 0.012621, -0.193538, -0.152713, 0.497728, -0.17702, -0.053723, 0.187064, 0.163742, 0.019125, 0.118204, 0.473921, -0.027621, -0.148521], [0.109108, 0.132097, -0.326851, 0.151793, -0.110976, -0.047301, 0.147426, 0.088628, -0.017168, -0.116193, 0.274582, -0.183607, 0.197826, 0.113736, -0.119677, -0.11444], [0.004654, 0.142727, 0.146888, 0.269603, 0.186193, 0.36817, 0.292308, -0.154144, 0.227774, -0.17764, 0.145715, 0.26514, 0.20001, -0.076979, 0.094148, -0.136059], [0.236743, -0.035191, 0.262815, -0.048912, -0.118063, 0.092606, 0.270309, 0.068629, -0.199241, -0.00092, 0.405798, -0.255816, -0.087568, -0.133952, -0.259453, 0.120016], [0.135526, 0.046335, -0.106441, -0.511872, 0.102263, -0.410152, -0.502916, -0.209118, -0.296256, -0.237487, -0.331776, -0.298577, -0.42, -0.12845, 0.139808, -0.28984], [-0.058999, 0.016913, -0.163579, 0.123169, 0.162172, 0.136871, 0.113159, -0.21289, -0.657045, 0.285739, -0.459432, -0.591718, 0.099704, 0.073778, 0.111351, -0.087905], [-0.037079, 0.174365, 0.15542, -0.439624, 0.100683, -0.083104, -0.27049, -0.087391, -0.139091, -0.478602, 0.160192, -0.0084, -0.306769, -0.304737, 0.003887, -0.068311], [0.036189, -0.213848, -0.074437, -0.292096, -0.381271, -0.21925, 0.3763, -0.040947, -0.029136, 0.263335, -0.224898, -0.269273, -0.152204, 0.288935, -0.024066, 0.119946], [-0.214122, 0.164666, -0.35343, -0.048592, 0.123361, -0.235416, -0.438806, 0.05892, -0.179778, -0.127242, -0.103551, -0.137817, -0.283705, -0.226053, -0.081945, -0.129538], [-0.103644, 0.084981, 0.111362, 0.154906, 0.037823, 0.343801, 0.04271, 0.259969, 0.020744, 0.09793, -0.577625, 0.276419, 0.26474, 0.116283, 0.18244, -0.122928], [0.145291, 0.149248, -0.183187, 0.034833, -0.093658, 0.035423, -0.010636, -0.380578, -0.07901, 0.130595, -0.24163, -0.28052, -0.167602, 0.178017, -0.035901, 0.081511], [0.005212, -0.254771, -0.094432, -0.324679, -0.072253, -0.47865, -0.282242, 0.114474, -0.012481, -0.175249, 0.030215, 0.04568, -0.40884, -0.395907, 0.016584, 0.11299], [0.187364, -0.164073, -0.075472, -0.366398, -0.186644, -0.227231, -0.132208, 0.14704, -0.152702, -0.191441, -0.308701, 0.150854, -0.558146, 0.080961, 0.258701, 0.05467], [-0.232456, -0.130441, -0.508207, 0.068653, -0.042084, -0.147996, -0.016374, -0.201654, -0.034398, -0.141149, -0.128783, -0.224236, 0.014307, 0.157309, 0.074234, 0.027429], [-0.187943, 0.079705, 0.335098, -0.374063, 0.195131, -0.246702, -0.174548, 0.251002, -0.057054, -0.09272, 0.115685, 0.100117, -0.371796, -0.224603, 0.005262, 0.165625], [0.116075, 0.167876, 0.469727, -0.21276, 0.02166, -0.161155, -0.21266, 0.110999, 0.088589, -0.045642, -0.371772, -0.355215, -0.157119, -0.370808, 0.220209, 0.019343], [-0.05488, 0.093783, -0.768168, 0.224855, 0.131865, -0.001547, 0.06012, -0.193468, -0.485171, 0.037401, -0.405669, 0.023081, -0.139337, -0.039831, -0.132189, -0.079367], [-0.286616, 0.130188, 0.225672, -0.332477, 0.103288, 0.120935, -0.340164, 0.221505, -0.054157, -0.379164, 0.023021, -0.054696, 0.427422, -0.173472, -0.019345, 0.087344], [0.046177, 0.030096, 0.038619, 0.204566, 0.171499, 0.211106, 0.106061, 0.197521, 0.001063, 0.208136, 0.051359, 0.136748, -0.105426, 0.09571, -0.005504, -0.056168]], [-0.313566, 0.212543, -0.051863, -0.019212, 0.045801, 0.280792, -0.214392, -0.256103, 0.170938, 0.120342, -0.479641, -0.03305, 0.046433, 0.260632, -0.489349, 0.324577]),
    ([[-0.449594], [-0.28416], [0.901836], [-0.615054], [0.183308], [-0.507772], [-0.832273], [0.574579], [0.57184], [-0.52993], [-1.464501], [0.731683], [-0.988186], [-0.63745], [1.037009], [-0.414476]], [0.16672]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 5.068810841538566e-19, 0.0, 0.0, 0.0, -2.534405420769283e-19, 0.0, 0.0, 0.0, 1.2354840004101888, 0.0, 10.235373650724295, 29.045780629283016], [6.894880047762485, 19.932973433822138, 35.828844117459674, 46.28838112885247, 59.58345850504529, 1.1288535466346525, 0.9972382401475413, 0.9509100251726608, 0.8477614371106685, 0.7566240757362533, 1065.974830301333, 2039.56663015695, 4012.5353057956445, 0.5930296144511749, 0.3020581511144633, 17.10440165990182, 36.041044035504086], [0.566097143562707, 0.259264995938383, 0.003998079933218471, -0.18461439386994238, -0.09909113913895669, 0.8100942713400355, -2.814201129099548, 5.70728211279055, -3.713167888183222, 3.3848996757781586, -0.545115637040028, 0.02632223148335277, -0.18305262194727148, 0.001881757544255653, 0.25602824974821586, -0.0002618872582300937, -0.00019672322469201564], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 5.074727552069564e-19, 0.0, 0.0, 0.0, -2.537363776034782e-19, 0.0, 0.0, 0.0, 1.2357253973320002, 0.0, 10.24382157388333, 29.06687743136384], [6.898175212270779, 19.94432111131209, 35.84821729762494, 46.31374990156119, 59.61458245938529, 1.1292479320432283, 0.997614069404236, 0.9512769517801772, 0.8481040200541816, 0.756949523396557, 1064.2421889182478, 2038.6600009884921, 4007.1636750665784, 0.5932910613542522, 0.3022343925906883, 17.112570355001356, 36.056623057820026], [2.6498341383747235, 0.8365014691593934, -0.25579914674007964, -1.4326632881404089, -0.7070860011656996, 1.5376983088542304, -4.666766920955899, 9.850415060556683, -6.38626239553004, 7.444746721364866, -2.2326835524909727, -0.07363618221570306, -1.293430464591002, 0.010452110942003077, -1.0561616959005966, -0.0014540827961403151, -0.0010915738303765145], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.235968, 0.0, 0.0, 5.908278, 5.908217, 0.0, 7e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009079], [1.133464, 0.999389, 0.952272, 0.847731, 0.756024, 2.569325, 20.907391, 0.789165, 18.42144, 0.707427, 0.649163, 0.542068, 1.664649, 1.664601, 0.999991, 4.40323, 1.165561, 0.30371, 133.77359, 335.203674, 1064.987427, 1.340487], [([[0.038162, 0.006707, 0.11989, 0.236987, -0.649475, 0.42753, -0.029032, -0.138089, 0.059373, 0.270187, 0.430266, -0.249858, 0.311759, -0.187665, 0.078249, -0.041072, -0.000435, 0.055687, -0.058814, -0.020494, -0.088421, 0.186157, 0.204982, -0.171409, 0.120732, -0.040417, 0.060514, -0.010344, 0.001585, -0.164247, 0.088223, 0.132331], [0.003363, 0.024639, 0.269113, -0.052162, -0.409094, 0.097496, -0.034, -0.25482, -0.002582, -0.230246, -0.175168, 0.131536, -0.047456, -0.08267, -0.110237, 0.229617, -0.084128, -0.130424, 0.127429, -0.097178, 0.067471, -0.013524, -0.135567, -0.010869, 0.189976, 0.068856, -0.012723, -0.312722, -0.213597, 0.061059, -0.016317, 0.275109], [-0.187778, 0.179429, 0.141717, -0.009439, 0.048501, -0.163173, 0.067523, -0.313196, -0.084915, 0.042854, -0.171877, 0.161061, 0.028544, 0.020524, 0.022637, 0.371086, -0.186028, 0.176332, -0.103631, 0.104124, -0.033992, 0.011903, -0.237821, 0.159075, -0.132432, 0.099054, 0.007902, -0.094996, 0.083294, 0.074958, -0.196115, -0.15962], [-0.057595, -0.193367, -0.256834, -0.15092, 0.374503, -0.163115, -0.192613, 0.237684, 0.257036, -0.269565, -0.108827, 0.067511, -0.300464, 0.43362, 0.071442, 0.178991, 0.008085, 0.164841, 0.220515, -0.16923, 0.008904, 0.357049, 0.127647, 0.400262, -0.194615, -0.33912, -0.177821, 0.008933, 0.040744, 0.112316, 0.092537, -0.039629], [-0.365681, 0.134845, 0.01458, -0.154444, 0.253121, 0.107791, 0.540112, -0.079887, 0.253976, -0.247543, 0.075964, -0.039452, -0.119488, 0.276195, -0.121829, 0.26035, 0.167937, -0.563253, -0.34724, 0.477052, 0.095855, 0.100655, -0.376108, -0.3222, 0.093737, -0.08131, -0.34269, 0.444414, 0.476581, -0.105494, 0.156802, 0.049728], [-0.300037, 0.114051, 0.251614, -0.148896, 0.278949, -0.024941, -0.062933, -0.144576, 0.108601, 0.007579, -0.113576, -0.021389, 0.086774, 0.329687, 0.06364, 0.367444, -0.307072, 0.130046, -0.225378, 0.086458, 0.206267, -0.039657, 0.139775, -0.193847, 0.189857, 0.15898, -0.253542, 0.023925, -0.065054, -0.1813, 0.205146, 0.145278], [0.040342, 0.257703, -0.208273, -0.208095, 0.499629, 0.145577, 0.231111, -0.393782, -0.199321, -0.35761, 0.106776, -0.49949, -0.089566, -0.063401, 0.152796, 0.06003, -0.016809, -0.771453, -0.194395, 0.33689, -0.517565, -0.40921, -0.241406, -0.399536, 0.312036, -0.182273, -0.317752, -0.181418, 0.457396, -0.177686, -0.062055, 0.347385], [0.187951, 0.039169, 0.032688, -0.212517, -0.06049, -0.00317, -0.137253, -0.104887, -0.091371, -0.051069, -0.006421, -0.035477, 0.016653, 0.328894, -0.325428, -0.152345, -0.151106, 0.037485, -0.091806, 0.078326, 0.053738, -0.040436, 0.188651, -0.151882, 0.289021, 0.084167, -0.077618, -0.041707, -0.009735, -0.145643, -0.009614, -0.124398], [0.168075, 0.14915, 0.16574, -0.093816, -0.054902, -0.092762, 0.081587, -0.257433, -0.321902, -0.514099, -0.33578, -0.4519, -0.204045, 0.12572, -0.212238, -0.221079, -0.065663, 0.285799, 0.052932, -0.361027, -0.109121, 0.060427, -0.076322, 0.009259, 0.18084, 0.130675, 0.058967, -0.102517, -0.497866, 0.175731, -0.222675, 0.079702], [0.198423, 0.20941, -0.431395, 0.044572, 0.180465, 0.136397, 0.150489, -0.187059, 0.276909, -0.209507, -0.277691, 0.136973, 0.421741, -0.030352, 0.057785, -0.046422, -0.905783, 0.001276, -0.204054, -0.227955, 0.112938, -0.0056, 0.312562, 0.082704, -0.188657, -0.046812, 0.108023, 0.003679, 0.041069, -0.124326, -0.235106, 0.165628], [-0.057228, -0.107449, -0.042672, 0.042325, 0.215329, -4.2e-05, -0.087538, -0.184061, 0.025173, -0.079124, 0.177395, 0.088827, -0.131904, -0.350503, 0.064847, -0.007212, 0.08631, -0.032878, -0.032355, 0.012892, 0.047772, 0.258494, 0.06051, 0.086991, 0.082763, -0.103571, -0.14441, 0.005835, 0.120556, -0.1166, 0.05073, 0.058483], [-0.302978, 0.25327, 0.115399, 0.657617, -0.135124, 0.057018, -0.196416, -0.133963, 0.011036, 0.179966, -0.366469, 0.154438, 0.350902, 0.297831, -0.139179, 0.086313, -0.231839, -0.02456, -0.171866, -0.183095, -0.360172, -0.338721, -0.041233, -0.283759, 0.159172, -0.229999, 0.410577, -0.011435, -0.006966, -0.047269, 0.259588, 0.220579], [0.032571, 0.177282, 0.085751, -0.29787, -0.02382, -0.05236, -0.164816, -0.231664, -0.030542, -0.108647, 0.063248, -0.418152, -0.048427, -0.069455, 0.321081, -0.133651, -0.005893, -0.016486, 0.118151, 0.000579, -0.016817, -0.126499, -0.114567, -0.270664, 0.033157, -0.107652, 0.377561, 0.005113, -0.006661, 0.010417, 0.14137, -0.359706], [-0.324373, -0.078727, 8.2e-05, -0.267077, -0.065011, 0.03704, -0.156904, -0.067496, -0.196337, 0.028114, -0.309582, -0.016687, 0.144306, -0.257364, 0.066179, -0.095816, 0.155672, -0.044667, 0.014571, 0.267541, 0.170948, 0.06162, -0.162246, -0.202245, 0.184628, -0.16175, 0.035637, 0.043866, -0.041866, -0.194338, 0.197918, -0.231201], [0.021999, -0.31346, 0.1125, 0.178032, -0.12118, -0.071192, -0.070255, -0.07654, -0.219515, -0.039213, 0.081357, -0.015592, 0.152761, -0.14888, 0.160783, -0.167012, 0.043718, -0.146171, -0.236791, -0.210863, 0.107617, 0.467839, -0.088861, -0.023998, 0.374945, -0.299625, 0.267924, -0.180441, -0.450323, -0.253451, -0.079428, -0.047639], [-0.047943, -0.080556, 0.131714, 0.178307, -0.119954, -0.256669, -0.062982, -0.106301, 0.091808, 0.031219, -0.066791, -0.001757, 0.219838, 0.162587, 0.001917, -0.040602, 0.109847, 0.000929, -0.201298, -0.258967, 0.145328, -0.054255, -0.129256, -0.050987, 0.30386, 0.283101, 0.110275, -0.08045, -0.584208, -0.17916, -0.032703, 0.057912], [0.287245, 0.382292, -0.105366, 0.348592, -0.271804, 0.224156, 0.162553, 0.383894, 0.194448, 0.406152, -0.227908, -0.149077, -0.219464, -0.177486, -0.296766, -0.048781, 0.162332, 0.044682, 0.183507, 0.208579, -0.427916, 0.51678, 0.229216, -0.02045, -0.08099, 0.142904, -0.276796, 0.30654, -0.073917, 0.216037, 0.049636, -0.268367], [-0.047502, -0.093809, 0.317596, -0.011144, 0.011104, 0.086752, 0.217829, 0.212539, -0.103124, 0.289701, -0.225678, -0.359813, 0.070787, -0.054177, 0.092778, -0.0577, -0.390892, 0.045393, -0.21604, -0.371451, -0.08513, 0.090589, 0.351551, 0.076607, 0.254101, -0.112232, -0.099887, 0.025744, 0.108918, 0.344442, 0.266419, -0.216588], [-0.074424, 0.010497, -0.370827, -0.033342, -0.091665, 0.063385, 0.050207, 0.03568, 0.068768, -0.098618, -0.06205, 0.238288, 0.111782, 0.097873, 0.170874, 0.081456, 0.12714, -0.113938, 0.030396, -0.1299, 0.295378, -0.015051, 0.055608, -0.005704, 0.05534, 0.074379, 0.003808, -0.03914, 0.0356, -0.169536, 0.082874, -0.054294], [-0.30726, 0.275821, -0.108239, -0.069262, 0.080651, 0.251791, -0.093032, -0.002512, 0.256838, -0.270606, 0.056608, -0.014491, 0.105385, 0.425761, -0.367996, 0.019623, -0.202043, -0.115839, -0.277436, -0.055299, 0.3719, -0.160474, 0.259118, 0.033872, 0.132737, 0.21697, 0.112638, 0.256218, 0.23073, -0.02402, 0.065433, 0.026866], [0.031287, 0.258029, 0.294867, -0.018815, 0.038883, -0.560887, -0.228976, 0.063839, -0.021311, 0.303699, -0.082679, 0.209402, 0.249536, -0.11606, 0.009979, -0.235684, 0.014433, 0.553942, -0.058087, 0.095014, 0.098072, -0.464333, 0.175818, -0.030529, -0.02475, 0.45737, 0.047179, -0.339618, -0.242247, -0.162935, 0.599203, -0.105924], [-0.051601, -0.102131, 0.242341, 0.16334, 0.00146, 0.005181, 0.015693, -0.079758, 0.191719, 0.016957, 0.020844, 0.306842, -0.105127, 0.037666, -0.39241, -0.065251, 0.010122, 0.027902, 0.019933, 0.208262, 0.641601, 0.071932, -0.094237, -0.088965, -0.062714, -0.308577, 0.018304, 0.017896, 0.005124, -0.11724, -0.01484, -0.132588]], [-0.052632, 0.123794, 0.323968, -0.205524, -0.127012, 0.086314, 0.134538, -0.11118, -0.072702, -0.493178, 0.14125, 0.446338, -0.307668, -0.572704, 0.029136, -0.46304, -0.013474, 0.243485, 0.221294, 0.309877, -0.352782, -0.027892, 0.220777, 0.037195, 0.291596, 0.088726, 0.42084, 0.417676, -0.028505, 0.049188, 0.03966, -0.019805]), ([[-0.035589, 0.130034, -0.825811, 0.025949, 0.18865, 0.147721, 0.131516, 0.259859, 0.151069, 0.251712, 0.044842, -0.158251, 0.02135, 0.152928, 0.179831, 0.201872], [0.024245, -0.307007, 0.156407, 0.104816, 0.208897, -0.285935, 0.030064, 0.137001, 0.228838, -0.179649, -0.031639, -0.024047, -0.172673, -0.034469, -0.332735, 0.046382], [-0.242139, 0.076791, 0.044605, -0.216732, 0.14159, 0.064758, -0.558657, 0.036028, 0.076873, -0.152599, -0.158024, 0.194438, -0.136931, 0.018591, -0.37243, -0.026564], [-0.209456, -0.304296, -0.431713, -0.163971, 0.088926, -0.067533, -0.205521, 0.009342, -0.109341, -0.052761, -0.037739, -0.134031, -0.179202, -0.080548, 0.162198, 0.119217], [0.355191, 0.16797, 0.172645, 0.088558, -0.136279, 0.313898, 0.298749, 0.048477, 0.416779, 0.054835, -0.266066, 0.257756, -0.360423, -0.047262, -0.209633, -0.342651], [-0.039524, 0.322972, 0.236469, -0.205574, 0.098826, 0.099135, 0.117171, 0.150838, 0.240408, 0.310537, 0.003244, 0.264176, 0.118096, -0.206037, -0.303119, 0.084195], [0.075507, 0.047592, -0.026404, -0.344157, 0.385709, -0.140114, -0.095142, -0.096331, 0.337946, -0.384676, 0.128769, 0.016055, 0.152342, -0.550854, 0.114177, -0.377594], [0.197878, 0.045075, 0.044528, 0.273136, -0.110668, 0.24878, 0.026389, 0.160812, -0.30308, 0.251329, -0.418952, 0.251929, -0.274953, -0.059554, 0.109909, 0.059948], [0.220124, 0.087022, 0.149412, 0.359619, -0.077686, 0.055479, -0.116617, 0.158958, -0.032428, 0.182809, 0.072012, 0.166923, 0.135263, 0.233337, -0.521494, -0.027005], [0.201444, -0.099229, -0.000888, 0.178568, -0.440651, 0.134939, -0.38831, -0.280876, -0.30007, 0.0856, -0.083186, -0.279401, 0.251268, 0.162655, 0.307659, 0.113004], [0.15521, 0.018281, 0.320782, -0.167583, 0.242227, -0.002794, 0.162483, -0.14071, 0.401358, -0.095977, -0.249511, 0.102121, 0.288701, -0.00076, -0.04292, 0.090121], [-0.130378, 0.037976, -0.089524, 0.180402, 0.15652, 0.045628, -0.322945, 0.138633, 0.270223, -0.116855, 0.244219, 0.240431, -0.141929, 0.138349, -0.374482, 0.023041], [-0.387054, -0.10178, 0.16555, 0.148437, 0.084318, -0.234147, 0.152166, -0.094821, -0.001848, -0.121525, -0.355825, 0.307946, 0.309509, 0.041605, 0.077203, -0.082232], [0.155772, 0.059897, 0.246004, 0.205121, -0.352177, -0.128603, 0.591262, -0.171264, -0.133657, 0.137266, 0.142853, 0.052409, -0.229404, 0.129558, -0.211542, -0.100808], [0.15176, 0.214477, -0.400788, -0.009865, -0.163486, -0.08284, 0.013391, -0.000952, -0.029231, -0.128818, 0.110742, -0.210783, -0.204364, 0.184028, -0.127639, -0.191068], [-0.089433, 0.131791, 0.261913, 0.11148, 0.210735, 0.242865, 0.183163, -0.051516, 0.133042, 0.007139, 0.193583, 0.240528, 0.351272, -0.084654, -0.033894, -0.320394], [0.272152, -0.009495, -0.0081, -0.008485, -0.188471, 0.054725, 0.180105, -0.15819, -0.601403, 0.065034, -0.901848, -0.06527, -0.230713, -0.216719, -0.33993, 0.136459], [0.175201, 0.191, -0.447803, 0.233176, -0.250723, 0.095531, -0.059676, -0.312332, -0.675196, 0.363487, -0.195142, -0.176774, -0.317109, 0.133492, 0.572487, -0.079315], [0.040696, 0.050362, -0.707556, 0.121299, 0.077654, 0.127231, 0.098155, -0.103299, -0.567779, 0.237565, 0.131306, -0.370135, 0.22786, 0.088742, 0.049689, -0.011257], [0.12858, -0.000977, 0.140491, -0.344112, -0.133978, -0.246676, 0.035664, -0.044749, -0.056347, -0.606177, 0.109584, -0.185118, -0.031107, -0.252598, 0.171359, -0.118207], [-0.201896, 0.059729, -0.233292, -0.064163, -0.294247, 0.010176, 0.131834, -0.140501, -0.044197, 0.416281, -0.645167, -0.1867, -0.294649, 0.036478, -1.154875, 0.086524], [-0.180275, 0.121044, -0.306786, 0.026528, -0.096913, -0.035534, -0.145318, 0.072395, -0.102864, -0.297495, -0.192627, -0.336561, 0.139674, -0.191625, 0.476344, 0.041293], [-0.080246, 0.085999, -0.254005, 0.249981, -0.068245, 0.350255, 0.083991, 0.12083, -0.036432, 0.086575, -0.282499, 0.112457, 0.233779, 0.248285, 0.14697, -0.170566], [0.102474, 0.264615, 0.478027, 0.163178, -0.166168, 0.092685, -0.089727, -0.417976, -0.160577, 0.120726, -0.28899, -0.148201, 0.124514, 0.177261, 0.04645, 0.112129], [-0.357444, -0.155139, 0.002103, 0.015405, -0.053746, -0.260242, -0.271116, 0.182959, -0.02551, -0.160934, -0.667735, 0.171876, -0.349371, -0.098207, -0.795117, 0.09216], [-0.240065, -0.110157, -0.083085, -0.2637, 0.085784, -0.375751, -0.217147, 0.058972, -0.004249, -0.371072, 0.087966, 0.175842, -0.331265, 0.084791, 0.078711, 0.096641], [-0.017605, -0.100488, -0.574508, 0.044667, -0.256567, -0.24294, 0.152783, -0.139728, -0.355953, 0.014084, -0.165497, -0.045002, 0.111745, 0.090252, 0.028152, -0.079367], [0.039461, 0.06724, 0.377083, -0.300722, 0.05682, -0.13149, 0.036359, 0.115001, 0.115897, -0.296529, -0.125067, -0.047651, -0.391216, -0.341481, -0.100459, 0.223458], [0.041801, 0.291951, -0.499285, -0.228896, -0.061189, -0.225001, -0.162502, -0.249771, -0.209196, -0.329755, -0.052105, -0.42395, -0.585931, -0.150721, 0.380873, -0.029731], [0.040528, 0.09679, -0.059953, 0.138569, 0.054284, 0.031295, 0.025278, -0.050516, -0.44674, 0.068908, -0.67598, 0.04523, 0.06974, -0.19502, -0.161143, -0.06296], [0.074697, -0.199099, 0.255174, -0.273987, 0.17747, -0.02417, -0.368543, 0.161265, -0.076233, -0.059239, 0.020097, 0.047276, -0.236069, -0.042045, -0.241687, 0.109255], [-0.04724, 0.11582, 0.015491, 0.11206, 0.18061, 0.266055, 0.118635, 0.298364, -0.041952, 0.131605, 0.097699, 0.170478, 0.087926, -0.118666, -0.190394, -0.115451]], [-0.194495, -0.006413, 0.013192, 0.089417, 0.038733, 0.257999, -0.535274, -0.064047, 0.197732, 0.138827, -0.129228, -0.031899, 0.069091, 0.385016, -0.564226, 0.191574]), ([[-0.662115], [-0.381915], [1.763333], [-0.791083], [0.566434], [-0.882403], [-2.460289], [0.612066], [1.446638], [-1.160545], [-2.412985], [1.178131], [-1.960822], [-0.671346], [3.026239], [-0.770355]], [0.299988])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.748641, -0.000111, 0.000111, 0.0, 0.053308, 0.053308, 0.0, 0.0, 0.0, 1.25666, 5.846188, 5.846188, 0.0]
SEQ_TSD = [1.078778, 0.135841, 0.135841, 0.102663, 1.444015, 1.444015, 0.669029, 0.222177, 0.183079, 0.616848, 1.678203, 1.678203, 0.651227]
SEQ_XM = [0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 1.235948, -0.0, -0.0, 5.908182, 5.908182, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009119]
SEQ_XS = [1.133826, 0.999664, 0.952553, 0.84802, 0.756213, 2.570136, 20.914425, 0.789277, 18.437494, 0.704278, 0.649224, 0.542084, 1.664759, 1.664759, 0.999991, 4.40289, 1.165521, 0.303697, 133.786133, 335.341339, 1065.236694, 1.340655]
SEQ_YSC = 1.9801905155181885
SEQ_WI = [[-0.016346, 0.288055, -0.22221, 0.332104, -0.503843, -0.019517, -0.894682, 0.154314, 0.101178, -0.200373, -0.160753, -0.32522, -0.261867, -0.194158, -0.136082, 0.181262, -0.041605, -0.089235, -0.116916, 0.303556, -0.647778, 0.186708, -0.659287, -0.112209, -0.115885, -0.606882, -0.443752, -0.236371, 0.234662, -0.291613, 0.049455, -0.268167, -0.140996, -0.164524, 0.047686, 0.033543, 0.014067, 0.216292, -0.239202, -0.068741, -0.06103, -0.099191, -0.079749, -0.298615, 0.195431, -0.276882, 0.207428, -0.183976], [0.356529, 0.277448, 0.529299, 0.492315, -0.739352, 0.217591, -0.155617, 0.051698, -0.173938, 0.430884, 0.593022, 0.303822, 0.147327, 0.223561, 0.130294, -0.17037, 0.245728, 0.117889, -0.508002, -0.808377, -0.009473, -0.063548, -0.340038, -0.043953, 0.131247, 0.13316, 0.07098, 0.03356, -0.101707, -0.767006, -0.34675, 0.130071, -0.242762, 0.19494, -0.075649, 0.004097, 0.186619, -0.365316, -0.099521, 0.167231, -0.138424, 0.056613, 0.139826, -0.376562, 0.100636, 0.050992, 0.435406, -0.15923], [0.08012, 0.247624, -0.106353, -0.040789, -0.25802, -0.339393, 0.742596, -0.384773, 0.012931, 0.409409, 0.023367, -0.269917, 0.060323, 0.290953, -0.443934, 0.045236, -0.241965, 0.267456, 0.170545, 0.110414, 0.2152, 0.131827, -0.19513, 0.412304, -0.025727, 0.08778, -0.206506, 0.14257, 0.606282, 0.242509, -0.357881, -0.09774, -0.413861, 0.133255, -0.114032, 0.059849, 0.137164, 0.05136, -0.07776, 0.224119, -0.282515, -0.287701, -0.245304, -0.279752, 0.444535, -0.165147, -0.07582, 0.599586], [0.186926, 0.495571, 0.032371, 0.471241, 0.295578, -0.149517, 0.073333, -0.282135, -0.269086, 0.329211, 0.044232, -0.19302, -0.11591, -0.068531, -0.382223, 0.049686, -0.106182, 0.554792, -0.250352, -0.216126, 0.450639, 0.105607, -0.309027, 0.094295, 0.23097, 0.407987, -0.126597, -0.19648, 0.457784, -0.496811, -0.486795, 0.14144, -0.396814, 0.170089, 0.215951, 0.405988, -0.235139, 0.14626, -0.108082, -0.029894, 0.003376, -0.00457, 0.056171, -0.175986, -0.030222, 0.097882, 0.192142, 0.039388], [-0.144558, 0.074924, 0.501573, 0.38634, 0.831399, 0.664572, -0.727447, -0.170386, 0.022136, -0.078665, 0.125476, 0.444677, 0.066048, 0.947658, 0.133947, -0.123147, 0.072284, -0.123759, -0.726523, -0.958824, 0.36959, 0.503609, -0.164851, 0.232547, -0.436608, -0.124568, -1.019579, -0.131967, -0.055683, 0.441387, 0.802386, -0.118615, 0.0285, 0.165017, 0.027341, 0.864682, 0.032962, -0.344558, -0.192435, 0.187212, -0.014175, -0.052743, 0.789763, -0.219111, 0.04677, 0.564899, 0.101492, -0.13884], [0.125919, -0.506825, 0.487097, -0.205492, 0.33863, 0.303952, 0.862087, -0.601236, -0.290219, -1.004414, 0.211976, 0.211952, 0.735101, -0.231553, -0.059783, -0.039204, 0.306132, 0.31622, 0.373809, -0.131496, -0.086082, 0.538874, 0.036428, -0.308251, 0.467032, -0.537572, 0.508211, 0.385927, -1.0736, 0.20077, -0.019314, -0.106089, -0.131938, 0.448808, 0.12548, 0.180463, -0.200943, -0.574686, -0.048126, -0.725543, 0.286014, 0.778638, 0.005239, -0.157415, -0.949663, 0.247124, -0.144072, -0.106947], [0.146238, -0.031456, -0.033362, 0.103745, -0.48259, -0.409474, 0.004548, 0.080725, -0.0705, 0.14022, 0.671498, 0.072355, -0.445969, -0.069964, 0.437233, -0.177101, 0.46964, -0.080123, -0.386678, -0.228473, -0.108892, 0.251015, 0.090235, -0.173817, 0.033905, 0.278665, -0.303801, -0.325706, 0.530018, -0.038007, 0.00583, 0.119633, 0.17827, -0.35832, -0.272697, 0.074371, -0.1196, 0.229526, 0.218226, -0.338438, 0.036954, -0.230397, 0.385105, 0.257687, -0.077648, -0.051782, 0.058196, -0.261993], [0.255691, 0.316427, 0.358623, -0.108919, -0.396398, -0.393234, 0.174873, 0.398296, -0.255984, 0.384223, 0.372701, -0.101911, -0.119603, -0.146548, 0.175176, -0.026523, 0.21977, 0.053268, -0.021576, -0.024386, -0.359145, -0.254556, 0.279467, -0.178296, -0.229079, 0.333073, -0.156389, 0.13165, 0.301828, 0.06375, -0.045788, -0.243168, 0.006158, -0.076245, 0.048187, 0.091305, -0.19257, -0.055091, 0.276767, -0.093292, 0.064722, 0.023733, -0.132388, -0.015779, 0.135536, -0.00588, 0.46433, 0.046485], [-0.238229, 0.28208, 0.305099, -0.311093, -0.540227, -0.365046, 0.157274, -0.062757, -0.322515, 0.227453, 0.626948, -7.8e-05, -0.324356, 0.020751, 0.262141, 0.096877, 0.450975, 0.250223, -0.369439, -0.000872, -0.334006, 0.170686, 0.235658, -0.365806, 0.073926, 0.211324, -0.45491, -0.424345, 0.409829, 0.095869, -0.171488, -0.064507, 0.157961, -0.025104, -0.178244, -0.112701, 0.268898, 0.274184, -0.107746, -0.050764, -0.076635, 0.221176, -0.128446, 0.040546, -0.193205, 0.047858, 0.310178, 0.057], [0.549896, 0.198698, 0.051705, -0.183462, -0.100511, -0.341146, 0.478215, -0.143266, 0.05949, -0.502614, 0.161723, -0.21522, 0.164245, -0.065827, 0.059874, -0.469407, -0.304938, -0.042846, 0.160828, 0.115829, 0.66933, 0.149705, 0.771891, 0.232959, -0.223463, -0.020773, -0.082417, -0.964242, 0.180491, 0.288254, -0.094959, -0.531944, 0.240905, 0.271037, 0.382192, -0.109366, 0.479089, -0.152626, 0.591946, 0.242517, 0.29514, 0.201509, 0.1154, 0.086942, 0.068051, 0.157043, 0.095324, -0.016008], [-0.527717, 0.278056, -0.344197, 0.346403, -0.233394, 0.422738, -0.17215, 0.256173, 0.306739, 0.024703, -0.315827, 0.07511, 0.157636, 0.556996, 0.226607, 0.0035, -0.54023, -0.247136, 0.125722, -0.295312, -0.102339, 0.233968, -0.171444, -0.393668, 0.264146, 0.355127, -0.312184, -0.314949, 0.016642, -0.269907, -1.397939, 0.253469, -0.228669, 0.236419, -0.119785, 0.215433, 0.165903, -0.071894, 0.069334, -0.100866, -0.207731, 0.029258, -0.105744, 0.219283, 0.162052, -0.146524, -0.069113, 0.142064], [0.048333, 0.209454, 0.226019, -0.480933, -0.125201, 0.1692, -0.386056, 0.262375, 0.395582, -0.106591, 0.128897, -0.991305, 0.251738, 0.677971, -0.166572, -0.188841, -0.048328, -0.355616, 0.369544, -0.231037, -0.081529, -0.087377, 0.069486, -0.438364, -0.013739, -0.295175, 0.041991, -0.397453, -0.296979, -0.334144, 0.035485, 0.09894, -0.466311, 0.163738, 0.131215, -0.043695, 0.21399, -0.016943, -0.008376, -0.0716, -0.02727, -0.090102, 0.142075, 0.097033, -0.30117, 0.146268, -0.078216, 0.036081], [-0.316307, 0.134698, 0.038727, 0.415327, 0.208888, -7.7e-05, 0.254175, -0.068098, -0.019167, 0.040218, -0.438865, 0.209521, -0.192843, -0.205782, 0.118995, -0.138304, 0.12839, 0.446372, -0.464546, -0.345359, -0.090958, 0.115717, 0.179559, -0.019693, -0.327586, 0.17719, 0.057078, 0.010824, 0.47482, 0.259648, -1.118931, 0.145895, 0.006368, 0.136348, 0.24714, -0.037531, -0.097085, 0.091012, -0.151678, -0.015236, 0.039233, 0.307768, 0.002989, 0.023256, 0.396533, 0.262786, 0.453628, 0.157023]]
SEQ_BI = [-0.051056, -0.1905, -0.003273, -0.237324, 0.15492, 0.10284, 0.105873, 0.023256, -0.080821, 0.256947, 0.284948, -0.031427, -0.110888, -0.225821, -0.03724, 0.252917, 0.773931, -0.047822, 0.808529, 0.394132, 0.540016, 1.294996, 0.344858, 0.427144, 0.799737, 0.88933, 0.753404, 0.642564, 0.390066, 0.683127, 0.96856, 1.205936, -0.152756, -0.068501, -0.02563, -0.096812, 0.12387, -0.182072, 0.005446, -0.124605, -0.228872, -0.016497, 0.08803, -0.186604, -0.21429, -0.18706, 0.117081, 0.209677]
SEQ_WH = [[-0.311917, -0.685243, -0.442332, -0.12094, -0.018747, -0.881702, -0.41669, -0.296937, -0.034027, -0.519056, -0.249185, -0.610794, 0.28672, -0.605562, -0.521866, -0.199778, -0.67342, -0.149447, 0.875742, -0.261061, 0.01973, -0.664421, 0.346243, -0.54202, 0.085335, -0.070619, -0.481701, 0.393983, -0.350378, -0.435443, 0.261293, -0.543949, 0.05431, -0.119604, -0.14936, 0.490595, -0.025238, 0.172304, 0.394335, -0.001439, 0.124799, 0.419946, 0.125548, -0.400728, 0.303277, -0.194522, -0.038989, 0.027635], [-0.06365, -0.053727, 0.273689, 0.2659, -0.31012, 0.335401, 0.067297, 0.00935, 0.199799, -0.477645, 0.019716, 0.298831, 0.268269, 0.109919, 0.03219, -0.03741, 0.071941, -0.083359, -0.881929, -0.187091, 0.118709, 0.015449, -0.196392, 0.070943, -0.474566, -0.473781, -0.333385, -0.39831, 0.172756, -0.122091, -1.203129, 0.228819, 0.164656, -0.215545, 0.192751, -0.441529, 0.031242, -0.32369, 0.222964, 0.084364, 0.157809, 0.056422, -0.319065, 0.062603, 0.256937, 0.001716, 0.168575, -0.2082], [0.180905, -0.143446, 0.32397, 0.372043, -0.164148, -0.065425, -0.192811, -0.188912, -0.558607, -0.135822, -0.001938, 0.562897, 0.688145, -0.810766, 0.282792, -0.05472, 0.363002, 0.741198, -0.04714, -0.017917, -0.013159, 0.294206, 0.463521, 0.316287, -0.006573, -0.205615, 0.559104, -0.515117, 0.404476, 0.611781, -0.53123, -0.205129, 0.492909, 0.056298, 0.551605, 0.172606, 0.106464, 0.142532, 0.170284, -0.286893, 0.014875, 0.241639, -0.04086, -0.472469, 0.008812, -0.018514, 0.204976, -0.070043], [-0.119403, 0.481452, 0.126207, 0.507529, 0.288936, -0.222206, 0.301393, 0.090036, 0.606734, 0.078771, 0.084826, 0.259828, 0.087832, 0.029191, -0.775958, -0.024564, 0.354986, -0.209856, -0.017124, 0.473395, 0.282754, -0.107022, 0.448809, 0.202551, -0.181973, 0.152555, 0.199855, 0.315983, 0.104371, 0.361812, 0.413485, 0.241033, -0.133646, 0.032143, -0.071452, 0.183064, 0.494312, 0.210362, -0.200762, 0.024229, -0.187257, -0.171817, 0.446947, -0.375833, -0.018886, -0.246427, -0.356135, 0.017204], [0.290432, 0.056123, 0.297388, 0.287833, -0.003868, 0.07646, -0.459679, 0.106933, -0.186173, -0.121566, -0.190577, -0.007219, 0.094029, 0.125178, -0.33315, -0.123528, -0.246658, -0.025836, 0.402812, 0.426609, -0.426076, 0.290177, 0.230001, 0.431941, 0.572075, -0.393578, 0.240171, -0.23318, -0.885494, 0.388342, -0.004171, 0.327735, 0.379316, 0.300501, 0.092577, -0.077666, 0.397369, -0.126531, 0.284227, 0.451248, 0.028421, 0.17597, 0.254842, -0.241654, 0.014894, 0.27321, -0.116071, -0.1527], [-0.061383, -0.233861, -0.54389, -0.218684, -0.527652, -0.247723, -0.098511, 0.128215, -0.474583, 0.109878, -0.189056, -0.553788, -0.170615, -0.201275, -0.061408, -0.083633, -0.361426, -0.116272, -0.446052, 0.110009, -0.44365, -0.205709, -0.038744, 0.030758, -0.676149, -0.431586, -0.170768, 0.016445, -0.50174, -0.319168, -1.170647, -0.055229, 0.001659, -0.113441, -0.283752, 0.173422, -0.275004, 0.473329, 0.043978, 0.03128, -0.544493, 0.129634, 0.482584, -0.444216, 0.279231, -0.455278, -0.51348, -0.398131], [-0.304855, -0.078876, -0.392707, -0.004574, -0.624812, -0.520948, 0.14226, -0.27597, -0.590341, -0.439842, -0.535919, 0.112713, 0.151043, -0.068632, -0.067664, -0.28765, 0.519523, 0.141889, 0.500221, -0.901458, -0.120592, 0.067489, -0.211593, 0.365829, 0.517385, -0.008252, -0.294822, -0.620998, -0.099392, 0.014324, -0.304304, -0.575204, 0.212883, 0.178656, 0.608055, -0.216198, 0.320725, -0.386296, 0.426933, -0.287019, -0.096737, 0.318039, -0.294392, 0.047596, 0.248647, -0.064776, 0.041691, 0.097392], [0.123876, 0.027976, 0.163222, 0.203806, -0.040692, -0.215421, -0.511793, -0.07545, -0.034428, 0.127055, 0.631094, 0.147796, -0.035921, 0.120063, -0.044323, -0.208597, -0.183993, 0.311541, 0.125415, -0.681757, 0.390679, 0.102132, -0.005778, -1.173143, -0.846689, -0.456561, -0.410487, 0.334247, -1.194201, -0.021504, 0.396551, -0.157935, -0.065282, -0.087683, 0.091023, 0.205929, 0.390401, -0.012071, -0.138767, 0.186746, 0.383882, -0.33688, -0.268959, 0.037217, 0.123937, 0.303434, -0.075592, 0.154348], [0.134497, -0.334125, 0.336486, 0.228893, 0.302496, 0.084538, 0.400272, 0.093119, 0.091504, 0.266263, 0.03449, 0.450908, -0.043808, 0.062598, 0.369005, 0.055358, 0.287454, 0.000895, -0.440273, -0.338049, -0.050572, 0.23393, 0.264008, -0.687889, -0.172479, -0.118509, -0.339837, 0.798589, -0.460464, 0.046392, 1.148886, -0.278861, -0.07367, 0.275617, -0.272593, -0.04058, -0.184773, -0.527514, 0.496815, 0.241769, 0.781232, 0.444097, -0.265685, -0.101566, -0.032112, 0.420388, 0.129482, 0.409331], [-0.323246, -0.087338, 0.066069, -0.004014, -0.092098, -0.070499, 0.183677, -0.023918, -0.075342, -0.467977, -0.288521, 0.604072, 0.274839, -0.101684, 0.08084, -0.245936, 0.231626, 0.40064, -0.6295, -0.39712, -0.234105, -0.009804, 0.182574, 0.083322, -0.906863, -0.108482, -0.351974, -0.73505, 1.061067, -0.224973, -0.524066, -0.209089, -0.038794, 0.317021, 0.369225, -0.125157, -0.080015, -0.209052, 0.480286, -0.241443, 0.038254, 0.534757, 0.011154, -0.17068, -0.05323, -0.29453, 0.089818, -0.195286], [0.330764, 0.339137, 0.22606, 0.112678, 0.010997, 0.097475, 0.256934, 0.149382, 0.091733, -0.102118, -0.222467, -0.335667, -0.289304, -0.155965, 0.034314, 0.071145, 0.160553, 0.033761, 0.494559, 0.99109, 0.853892, -0.391423, 0.338632, 0.605153, 0.895576, 0.231205, -0.035841, -0.136356, -0.081, 0.647156, 0.243096, 0.410844, 0.086639, 0.163562, 0.49591, 0.205439, 0.511735, 0.242498, 0.149761, 0.103065, -0.328491, -0.054264, 0.452202, -0.493989, -0.166966, -0.063042, -0.079405, 0.028023], [0.095429, 0.069286, 0.094499, 0.231773, -0.157846, 0.612004, 0.201212, 0.466536, 0.629335, 0.535329, 0.405341, -0.293758, -0.073334, 0.773091, 0.181553, 0.786129, -0.170766, -0.136939, -0.548011, -0.356535, -0.238491, 0.000458, 0.134533, -0.000347, -0.139348, 0.111441, -0.334731, -0.080941, -0.23653, -0.072731, -0.040529, -0.094866, -0.235845, -0.20192, -0.059083, -0.347264, -0.139522, -0.037531, 0.046006, 0.053254, 0.383297, -0.136729, -0.234479, 0.629396, -0.052794, 0.420858, 0.168467, 0.503359], [-0.347552, -0.23887, -0.35055, -0.059189, -0.113029, -0.258245, -0.54286, -0.169296, -0.29856, -0.171702, 0.262854, -0.058734, -0.525635, -0.315476, 0.142264, -0.321486, -0.099157, 0.398482, -0.970647, -0.369391, -0.225486, -0.670977, 0.159345, -0.744431, -1.126485, -0.876636, -0.496346, -0.566054, -0.805221, -0.10778, -1.715732, 0.118432, 0.14682, -0.215416, -0.162357, -0.287663, 0.195069, -0.182915, -0.016743, 0.066273, -0.183129, 0.099594, -0.097941, 0.349055, 0.533721, 0.172227, 0.211159, 0.125808], [0.38879, 0.043983, 0.395868, 0.52884, 0.441443, 0.068068, 0.38531, 0.04872, 0.343109, 0.360459, 0.368213, 0.860362, 0.07156, -0.330509, 0.158338, -0.110632, 0.453112, 0.16409, -0.144366, 0.507525, -0.290024, -0.306766, 0.162202, -0.06688, -0.286384, 0.013607, -0.609179, 1.03205, -0.422988, 0.287145, 0.731144, -0.496776, -0.433223, 0.169059, -0.286481, 0.096205, 0.005644, -0.410338, -0.178112, 0.472939, 0.426323, -0.255535, -0.298138, 0.066661, 0.24701, 0.380963, 0.154523, 0.243301], [-0.573182, 0.199238, 0.161345, 0.230511, 0.067927, -0.22535, 0.169511, 0.413369, -0.093683, 0.476623, -0.172299, 0.206528, 0.316038, 0.38296, -0.074714, 0.125974, 0.648798, 0.187008, 0.547281, 0.437551, 0.475132, 1.150051, 0.817769, 0.213673, 0.574808, 0.297475, 0.367857, 0.266347, 0.586073, 0.591008, 0.864652, 0.458589, -0.039013, -0.148862, 0.097679, 0.049197, 0.331955, -0.089917, 0.262986, -0.009924, 0.034175, 0.180364, 0.265171, -0.223465, 0.12522, -0.062566, 0.227235, 0.017315], [-0.015533, 0.262035, 0.518788, 0.072218, 0.03541, -0.082313, 0.356517, 0.14349, -0.080672, 0.150041, 0.1894, 0.226568, 0.011369, 0.034933, 0.360256, 0.459504, 0.435579, 0.082545, 0.627159, -0.045403, 0.864913, 0.802719, 0.799464, -0.102216, -0.051394, 0.052427, 0.484543, 0.464226, 0.288081, 0.630863, 0.799206, 0.958796, -0.091466, -0.075397, -0.096121, -0.073599, -0.048526, -0.472872, -0.10252, 0.334393, -0.129277, -0.207135, -0.445567, 0.353848, 0.077327, 0.177586, 0.058148, 0.337637]]
SEQ_BH = [0.129349, 0.222972, 0.336184, 0.185376, 0.205271, -0.062245, -0.106512, 0.211544, 0.304786, -0.002164, 0.129393, 0.374482, 0.263788, -0.180073, -0.008617, -0.069043, 0.622234, -0.168991, 0.652581, 0.564352, 0.728982, 1.059763, 0.306733, 0.65649, 1.145857, 0.706331, 0.872145, 0.460438, 0.493373, 0.602179, 0.679641, 1.147169, -0.122659, -0.230845, 0.038505, 0.108622, -0.042546, -0.316169, 0.040494, -0.061808, 0.193277, 0.0838, -0.001678, -0.000355, -0.204188, 0.023439, 0.180578, 0.30233]
SEQ_HEAD = [
    ([[-0.108677, -0.172742, -0.166889, 0.13701, -0.055418, 0.075358, -0.430309, 0.14723, 0.208306, 0.146498, -0.001273, 0.040637, 0.181017, -0.163806, 0.040713, 0.145974, -0.082929, 0.072858, -0.023223, -0.033336, 0.073164, 0.055604, -0.194179, 0.075348, -0.109108, 0.174639, -0.08003, -0.115579, -0.118918, 0.171087, 0.039448, 0.193297, -0.131894, -0.14125, 0.187332, 0.145823, -0.041548, -0.078218, 0.098325, -0.04594, 0.007048, -0.167744, -0.047357, -0.067633, -0.045363, 0.063351, 0.018616, 0.101273, -0.157125, 0.096719, 0.029182, -0.031061, -0.079393, -0.068716, 0.015074, 0.021396, -0.040331, 0.101115, 0.082465, -0.288435, -0.076536, 0.258826, 0.029074, 0.007717], [-0.015823, -0.079155, 0.003922, 0.165578, 0.174611, -0.079777, -0.157092, 0.187025, -0.09993, 0.029653, 0.021748, -0.042959, -0.122978, -0.066651, -0.006186, 0.003971, 0.190105, 0.008307, -0.138571, -0.209683, 0.137649, -0.084386, -0.033705, 0.041913, -0.06649, 0.150188, 0.007289, -0.167368, -0.154176, 0.096352, -0.138392, 0.146338, 0.069261, -0.160292, -0.012616, -0.087905, -0.108878, 0.043491, 0.01375, -0.09873, 0.121693, -0.147538, 0.049286, 0.139676, 0.120062, 0.217284, -0.067554, -0.040158, -0.12613, -0.131897, -0.124226, 0.131397, 0.149713, -0.015547, -0.097621, 0.044146, 0.093365, -0.139312, 0.086081, -0.07195, -0.171078, 0.051179, -0.150518, 0.020423], [0.273346, 0.246333, 0.164807, -0.092737, -0.04329, -0.003333, -0.130244, -0.067654, -0.175341, 0.061601, 0.07714, -0.115344, -0.050811, 0.101877, -0.056003, -0.066983, -0.023756, -0.055377, -0.10822, -0.040755, -0.036977, -0.069321, -0.180269, 0.160176, 0.152162, 0.123039, 0.016243, 0.096851, 0.104205, 0.068129, 0.07553, 0.057382, -0.153675, 0.07292, -0.153294, -0.046176, -0.064012, -0.01467, -0.140515, -0.028024, -0.08702, -0.118668, -0.101079, -0.085806, 0.191316, 0.048046, -0.025962, -0.213478, -0.060715, -0.006169, 0.103545, 0.010921, 0.005361, 0.138097, -0.216929, 0.026507, -0.125338, 0.053751, -0.02767, -0.073286, -0.19757, 0.008134, 0.122835, 0.057214], [0.28397, 0.046898, 0.01742, 0.121456, -0.164775, -0.091865, 0.147056, 0.237675, 0.04509, -0.07778, 0.142688, -0.147765, -0.121542, 0.091628, -0.021308, -0.175902, -0.012878, -0.005474, 0.155824, -0.14845, 0.135667, 0.112905, -0.184922, 0.001286, 0.047018, 0.007161, 0.065715, -0.052713, -0.153817, -0.115523, -0.070485, 0.125363, -0.078532, 0.010607, -0.01274, 0.115525, -0.182337, -0.00505, -0.023945, 0.113406, -0.001651, -0.118367, 0.190124, -0.141519, 0.075896, 0.203826, -0.107099, -0.304433, 0.314898, -0.069417, 0.023164, 0.020355, 0.036785, -0.024512, 0.105731, -0.13918, 0.009289, 0.019, 0.076333, 0.080866, 0.134029, -0.069982, 0.061748, -0.1411], [-0.126222, 0.116655, 0.120453, 0.056723, -0.040867, -0.154832, 0.060502, 0.323787, -0.110876, -0.143291, 0.068752, 0.023514, 0.226446, -0.172343, -0.089593, -0.139734, -0.279311, -0.015704, -0.197192, 0.057785, -0.013436, 0.35827, 0.057857, -0.067801, -0.046667, 0.242967, 0.000463, -0.012449, -0.011659, 0.192564, -0.189076, -0.01914, 0.137557, 0.140224, -0.032724, -0.249107, 0.102253, 0.028229, 0.020708, 0.252364, 0.271942, -0.104697, 0.352596, -0.280365, 0.110691, -0.066222, -0.084754, 0.238984, 0.223581, 0.019522, 0.253549, 0.042009, 0.067319, 0.198281, 0.173968, 0.211786, 0.223913, -0.24062, 0.230942, 0.265426, 0.163631, -0.035836, 0.303847, -0.046156], [0.074037, 0.078978, 0.098963, -0.070328, 0.095945, 0.124765, -0.031934, 0.023608, -0.053829, -0.072807, -0.030486, -0.018148, 0.066536, 0.118176, -0.079607, -0.063785, 0.177549, 0.118747, 0.12126, 0.050384, -0.026732, -0.052989, 0.099273, 0.062894, -0.129205, -0.070249, -0.022208, 0.103886, 0.064419, 0.165256, 0.018845, 0.078361, -0.067697, -0.102818, -0.046394, -0.110985, 0.056614, 0.022925, -0.017994, 0.172685, -0.162437, 0.021922, 0.073388, 0.107355, 0.079676, -0.071805, -0.097869, -0.122182, 0.236696, -0.059206, -0.095145, -0.005048, 0.030076, 0.064294, -0.151168, -0.039801, -0.049903, -0.072285, -0.123675, 0.021901, -0.197129, 0.106272, 0.150402, -0.227834], [-0.094745, 0.249776, 0.079857, 0.181211, 0.197764, -0.029681, -0.093981, 0.004919, -0.018084, -0.099774, 0.039814, 0.057016, -0.295659, -0.091418, -0.019861, 0.136041, 0.122689, 0.129499, -0.091171, 0.216623, 0.11458, -0.069942, -0.035826, 0.173888, -0.074012, 0.050615, -0.115661, -0.019074, -0.077441, -0.041166, -0.150493, -0.039169, 0.112012, 0.063515, 0.123445, -0.139011, 0.009845, -0.096425, -0.073293, -0.009257, -0.025831, 0.006144, 0.097236, -0.085038, 0.166412, 0.173085, 0.099417, -0.22006, 0.043581, 0.041018, 0.003528, 0.017742, -0.04639, 0.062433, -0.10219, 0.062725, -0.046615, -0.116356, 0.055493, -0.327482, 0.051419, 0.00487, -0.018898, -0.297349], [-0.111176, 0.234497, -0.221309, 0.095927, 0.168264, -0.232884, 0.342414, 0.038006, 0.08994, 0.060739, -0.004938, 0.103634, -0.061943, 0.00751, 0.08993, 0.00956, -0.090222, -0.021711, 0.012271, -0.167646, 0.085714, -0.043655, -0.123858, 0.110139, -0.085838, 0.101828, 0.058466, -0.022633, -0.001536, -0.230951, -0.215757, -0.119213, 0.032513, 0.014837, 0.002291, -0.027891, -0.024871, -0.071332, 0.186703, -0.209085, 0.32235, 0.049176, -0.053703, -0.206572, -0.03725, 0.045631, 0.100787, 0.013516, -0.148287, -0.169003, -0.006793, 0.131587, 0.188683, -0.149598, -0.059571, -0.12712, 0.052431, -0.068648, -0.134293, -0.019569, -0.027539, -0.064216, -0.120102, 0.107976], [0.277476, 0.129986, -0.05407, 0.175502, 0.172926, 0.025279, -0.125911, 0.06105, -0.127354, 0.10269, 0.124619, -0.185173, 0.074956, -0.268198, 0.050041, -0.098378, -0.29415, 0.159711, 0.021384, -0.180522, 0.157045, 0.103764, -0.165098, 0.065699, -0.276103, 0.255671, 0.032185, -0.144155, -0.037142, 0.010693, -0.056441, 0.100093, -0.084204, 0.050099, 0.147777, -0.112982, -0.184935, -0.095876, -0.343258, 0.097249, -0.053541, -0.154836, 0.254738, -0.144331, -0.066014, 0.010418, -0.147686, 0.106261, -0.046939, -0.195788, 0.123762, 0.176643, 0.072381, 0.194464, -0.012908, 0.123198, -0.033347, -0.218374, 0.067264, 0.283188, -0.185015, 0.204149, 0.025727, -0.103008], [0.06953, -0.091194, -0.146017, -0.051006, 0.02211, 0.059656, -0.00679, -0.146128, -0.026616, 0.007984, -0.239853, -0.103343, -0.188831, 0.086364, -0.094145, 0.166589, 0.095804, -0.067431, -0.09218, -0.067431, 0.018426, -0.06785, 0.051351, -0.152289, 0.12293, -0.073958, 0.012809, -0.094491, -0.315233, 0.018149, -0.066327, -0.196734, -0.076852, 0.167598, 0.144826, -0.037805, 0.199281, 0.146637, 0.269064, 0.06465, -0.039429, -0.053477, 0.058713, 0.130138, -0.035884, -0.029937, -0.274522, -0.022637, -0.111402, 0.025427, 0.113034, 0.032321, -0.076718, -0.125593, -0.053914, 0.174952, -0.074577, -0.313573, -0.060927, -0.072537, 0.029333, 0.062607, 0.128158, -0.11812], [0.12191, -0.181511, -0.026788, -0.10831, -0.114562, 0.075571, -0.046596, 0.153931, -0.09928, -0.065271, -0.053872, 0.116758, 0.028592, -0.019718, -0.019388, 0.154576, 0.037158, -0.005734, -0.046915, 0.031915, -0.172867, 0.072794, -0.192939, 0.192286, 0.115979, -0.165643, -0.067787, -0.237132, -0.065439, -0.010316, 0.125265, -0.07859, 0.008636, 0.013334, -0.096028, -0.034343, -0.002286, 0.02117, 0.1089, -0.147263, 0.132863, 0.021972, 0.020399, 0.019193, -0.072982, 0.040701, 0.0581, -0.176885, -0.130706, 0.022435, 0.107699, -0.121066, 0.162222, 0.050724, 0.103513, 0.137133, -0.223311, 0.096878, -0.068366, 0.111641, 0.183249, -0.01683, -0.104631, -0.037327], [0.10451, 0.247187, -0.014869, 0.070121, 0.163978, 0.062125, 0.134109, -0.047959, 0.071243, 0.224168, 0.190372, -0.274529, 0.285719, 0.049313, -0.231383, -0.237469, -0.109387, -0.069802, -0.110184, 0.059841, 0.268036, 0.066943, -0.132767, 0.201193, -0.29492, -0.04, -0.158274, -0.171713, -0.20006, 0.214744, 0.065733, 0.102633, 0.218958, -0.18469, -0.131944, -0.06224, -0.172819, 0.060342, -0.0604, -0.222566, -0.078814, -0.165833, -0.254694, -0.049134, 0.365062, -0.15003, -0.030941, 0.145732, 0.134523, 0.101558, 0.25992, -0.277187, -0.287749, 0.198719, 0.11221, 0.258675, 0.183038, -0.083783, -0.016216, -0.028079, -0.165641, 0.074483, 0.143568, -0.022774], [-0.072093, -0.035794, 0.263323, 0.06866, -0.018315, -0.095802, -0.092572, 0.07109, 0.012019, 0.073366, -0.044841, -0.111079, 0.079007, -0.095799, 0.017151, 0.022404, -0.00575, -0.098758, -0.009201, 0.115085, 0.012826, -0.053724, -0.001549, -0.040652, 0.025095, -0.119361, -0.09045, -0.033668, -0.057257, 0.197759, 0.273811, 0.083305, 0.152859, 0.020734, -0.109928, 0.032449, 0.029471, -0.078759, 0.617749, 0.077432, 0.208853, -0.103708, -0.138491, 0.088291, 0.236638, -0.0973, -0.1742, 0.006657, 0.044481, -0.084869, -0.164143, -0.014749, 0.175091, 0.008672, 0.180538, -0.123437, 0.118075, -0.09056, 0.06628, 0.01704, 0.138359, -0.24706, -0.028489, 0.015198], [-0.004691, -0.139506, 0.157616, 0.09547, 0.027911, 0.073125, 0.106098, -0.027813, 0.183432, 0.121642, 0.006328, -0.084637, 0.048045, -0.073689, -0.083841, 0.095149, 0.081274, -0.095257, 0.110097, -0.047202, 0.094578, 0.055547, -0.069409, -0.113142, 0.006486, -0.06916, -0.02529, 0.070033, 0.099027, 0.207486, 0.051941, -0.112013, 0.109638, -0.098447, -0.123627, -0.018556, 0.20813, -0.136791, 0.104159, 0.207538, 0.223384, -0.075788, -0.083856, 0.159322, 0.091385, -0.010367, -0.052255, -0.059164, -0.018093, -0.16262, 0.097218, 0.002477, 0.143436, -0.062012, 0.090962, -0.072772, -0.069695, -0.00508, 0.012334, 0.008767, 0.006162, -0.111741, -0.11194, 0.124583], [0.084688, -0.148134, 0.194744, -0.147304, 0.161574, 0.277519, 0.022846, -0.186373, 0.074403, 0.133888, -0.040743, -0.048482, -0.193014, 9.6e-05, 0.074302, -0.013292, 0.05378, -0.071563, -0.056072, 0.006726, -0.121384, -0.075583, -0.051078, 0.12134, 0.100013, -0.075756, -0.152115, 0.005216, -0.082259, 0.044797, 0.033378, 0.150306, -0.146508, -0.014499, 0.052335, 0.134287, -0.094255, 0.000824, 0.058074, -0.089128, -0.149701, 0.050498, -0.231887, 0.072469, -0.008249, -0.04485, -0.179684, -0.064845, -0.08036, -0.16177, -0.183635, -0.100442, 0.080214, 0.103515, -0.110886, -0.088708, -0.224766, 0.040937, 0.154288, 0.120478, 0.141946, -0.164349, 0.072468, 0.049762], [-0.308668, 0.094598, 0.079833, 0.109753, -0.123261, 0.024903, 0.044956, -0.058183, 0.187801, 0.032964, -0.084967, -0.036079, -0.031362, 0.05066, -0.168125, -0.035784, 0.090526, 0.17288, -0.062504, -0.097264, -0.017465, -0.258163, 0.116235, 0.159435, 0.028568, 0.057882, -0.115151, 0.06969, -0.068324, -0.102742, 0.015348, -0.056503, -0.203508, -0.075086, 0.024248, 0.128726, 0.05995, 0.071574, 0.04048, 0.112179, 0.023445, -0.072208, 0.012627, -0.122237, -0.063407, -0.018018, 0.129736, 0.233686, -0.201403, 0.000418, 0.182666, -0.080522, 0.240127, -0.217712, 0.077567, 0.246181, -0.085695, 0.053017, 0.251417, -0.080574, -0.047763, -0.210848, 0.143212, 0.016458], [-0.145267, 0.069488, -0.317785, -0.208633, 0.401829, 0.225045, 0.033422, 0.336733, 0.229978, 0.275202, -0.191711, 0.058772, 0.330175, -0.008566, 0.05838, -0.096419, 0.283163, -0.046372, -0.000466, -0.026584, 0.383334, 0.175349, -0.172698, 0.20197, -0.125191, 0.014714, -0.233062, 0.040713, -0.331863, 0.10324, -0.270978, 0.212979, 0.137454, -0.064347, 0.008951, 0.005175, 0.115499, 0.027143, 0.034071, -0.099289, 0.534043, -0.147077, 0.091, -0.149711, -0.017898, 0.263654, 0.071273, -0.168762, -0.255515, 0.028547, -0.075525, -0.183049, -0.360924, -0.232826, 0.200527, 0.022906, 0.218178, 0.176831, 0.020868, 0.135462, -0.24194, 0.014517, -0.039625, 0.274657], [0.193042, 0.338745, -0.130234, 0.193394, -0.18197, 0.019356, 0.199051, -0.099114, 0.130652, -0.082205, 0.218311, -0.180186, -0.029415, 0.130597, -0.155766, 0.089705, 0.142858, 0.106793, -0.096569, -0.134892, -0.179819, -0.137761, -0.229471, 0.096083, 0.057078, 0.149409, -0.180691, -0.100953, 0.123821, 0.071372, 0.046696, -0.050992, -0.10915, 0.019442, -0.004621, 0.055386, -0.13582, -0.055281, -0.153265, 0.160355, -0.03713, -0.128468, -0.075538, 0.065756, 0.157149, 0.085843, 0.139795, -0.143307, -0.069431, 0.084909, -0.206654, 0.071798, 0.065424, 0.15446, 0.238667, -0.131564, 0.027607, -0.104362, -0.301065, -0.114103, 0.067805, -0.119332, -0.131491, 0.174575], [-0.014384, 0.056358, 0.216413, -0.118468, -0.062766, -0.114129, 0.02233, 0.013898, 0.10366, 0.06741, -0.02218, -0.024561, 0.013815, 0.030408, 0.154748, 0.005717, -0.033936, 0.096677, 0.001028, 0.08595, -0.039032, -0.078538, -0.076117, -0.026372, 0.043317, 0.134857, -0.036481, 0.110179, 0.02246, 0.005916, 0.082271, 0.057334, -0.025814, -0.049049, 0.021816, -0.106433, 0.188121, -0.148981, 0.097094, 0.012806, -0.058828, -0.086072, 0.062591, -0.073104, -0.090057, -0.010029, 0.096342, 0.086238, -0.087743, -0.090833, -0.124909, -0.000117, 0.21544, -0.04804, -0.021094, 0.171291, 0.120326, 0.016105, -0.025816, 0.064945, -0.108336, -0.101961, -0.094799, 0.023871], [-0.203354, 0.004448, -0.207975, -0.223807, -0.081874, -0.368251, -0.089234, 0.098214, 0.16037, 0.236035, -0.135096, 0.152284, 0.110148, 0.195266, 0.082666, -0.166968, -0.208318, 0.07867, -0.103959, -0.046868, 0.166578, -0.139404, -0.082596, -0.031027, -0.227347, -0.020407, -0.116923, 0.05551, -0.016941, 0.039274, 0.227928, 0.096018, 0.17392, 0.200958, 0.027132, 0.003217, -0.089464, 0.133558, -0.06529, -0.004979, -0.207278, -0.141464, 0.07076, -0.109845, -0.076736, -0.071273, 0.1368, -0.162986, -0.059902, -0.323753, 0.345769, -0.168624, 0.149936, 0.069314, 0.207186, -0.194657, 0.030262, -0.080242, 0.110651, -0.002765, 0.033496, 0.086243, -0.010226, 0.171256], [0.328435, -0.105595, -0.013212, 0.283887, -0.040224, -0.142552, 0.210807, -0.123985, -0.19883, -0.050458, -0.196235, -0.131567, -0.008561, 0.008225, -0.025307, 0.100225, 0.004449, 0.039747, 0.059404, -0.193088, -0.076902, -0.107941, 0.101222, 0.036505, -0.161476, -0.031714, -0.132989, 0.000999, -0.026457, 0.051515, 0.038217, -0.017234, 0.221903, 0.002458, 0.23479, 0.199185, -0.101345, -0.13738, -0.024147, 0.108175, -0.140648, -0.138385, -0.31971, 0.036148, 0.029758, -0.06138, -0.159089, 0.003669, -0.157551, -0.019764, -0.111447, -0.05222, -0.036997, -0.171538, -0.135872, 0.186384, -0.119265, 0.197059, -0.10152, 0.160431, 0.000321, -0.187789, 0.09994, -0.085696], [-0.001416, 0.034249, -0.099004, 0.13785, 0.167341, 0.022051, 0.118883, 0.027006, 0.05822, 0.03065, 0.126466, 0.039765, -0.254153, 0.343087, 0.108628, 0.182634, 0.046605, 0.005364, 0.097203, -0.102458, -0.087481, -0.15445, 0.080026, 0.07156, -0.033696, -0.010471, -0.16501, 0.152255, 0.075491, 0.047558, 0.222301, 0.072389, -0.09133, 0.132928, 0.016208, -0.059043, 0.00336, -0.193413, 0.20275, -0.015355, -0.01023, 0.053896, -0.00592, 0.071348, 0.130775, 0.134898, -0.282545, 0.105997, -0.154772, -0.108767, 0.113338, -0.34473, -0.058479, 0.619636, 0.042535, -0.05547, -0.061589, 0.115446, 0.067534, 0.134274, -0.053957, 0.100154, 0.050787, -0.043392], [-0.17393, -0.095438, 0.015667, -0.06629, -0.13823, -0.090482, -0.035861, -0.033769, 0.258734, 0.24358, -0.154733, -0.238939, 0.084783, -0.219438, 0.318769, 0.088152, 0.020633, 0.195449, 0.083675, 0.085259, 0.009815, 0.108426, -0.298758, 0.182641, 0.200247, 0.049381, -0.051219, -0.066927, 0.280613, -0.157935, -0.173097, -0.137066, 0.160826, -0.047026, 0.005799, -0.113817, -0.156163, 0.087994, 0.029756, 0.069417, -0.031814, 0.169453, -0.00323, 0.115136, -0.178698, -0.017971, 0.11886, -0.220091, 0.029903, -0.010087, 0.051948, -0.226872, 0.037567, -0.023756, 0.207168, 0.052158, -0.016837, -0.121952, -0.164645, -0.489046, -0.084689, -0.103712, -0.173762, 0.030714], [0.043923, -0.070806, -0.010545, 0.138633, -0.18777, 0.111002, -0.018719, -0.166446, 0.010239, -0.062638, 0.00741, -0.120611, 0.061745, -0.004183, 0.035592, 0.03237, -0.128432, 0.032194, -0.008187, 0.126891, 0.048323, 0.026294, -0.071843, -0.094623, 0.011059, 0.163351, 0.18329, -0.190458, -0.104885, 0.040662, 0.077959, 0.144227, 0.153006, -0.020831, -0.043484, -0.160927, 0.140979, 0.091759, 0.314055, -0.171851, 0.019506, 0.082442, -0.173528, 0.060566, 0.163462, 0.045516, 0.065021, -0.041263, -0.095106, -0.094609, 0.1993, 0.19195, -0.054115, 0.057721, 0.046639, -0.096883, 0.06549, -0.221061, -0.068653, -0.157802, 0.11249, 0.051209, -0.070908, 0.033786], [0.229702, -0.182875, -0.087856, -0.050409, -0.158105, -0.135351, -0.127572, -0.047577, -0.209972, -0.127777, -0.069664, -0.46497, -0.132835, 0.15115, -0.028763, -0.296126, -0.002223, -0.106466, -0.091029, 0.066907, -0.163384, -0.018633, -0.215951, -0.008213, 0.084448, -0.043611, -0.215502, -0.059604, -0.264745, -0.182539, -0.08487, 0.03403, -0.095677, 0.129508, -0.198551, 0.174187, -0.113631, 0.037469, -0.241099, 0.10306, -0.023235, -0.226054, -0.003667, 0.074922, -0.179348, 0.162615, -0.066496, -0.230567, 0.050563, -0.087327, -0.110226, 0.141527, 0.119967, -0.007935, -0.111484, -0.069752, -0.023183, -0.265606, -0.173435, -0.475852, 0.122774, -0.168446, -0.103059, -0.548374], [-0.072498, 0.073685, -0.181388, 0.177036, 0.167381, 0.25311, 0.004881, -0.159018, -0.305779, 0.055729, 0.159519, -0.084544, 0.039057, 0.106166, -0.457706, -0.224814, 0.067824, -0.173282, -0.51396, -0.071269, -0.131053, 0.219804, -0.009556, 0.097178, -0.053789, 0.034422, -0.339069, -0.084364, -0.165943, -0.233446, 0.141833, -0.152388, -0.177373, -0.065895, 0.028585, 0.136985, -0.138554, 0.14574, -0.050636, -0.298339, -0.019209, -0.317223, -0.092389, -0.339921, -0.201988, 0.079166, 0.064752, 0.145127, -0.368235, 0.132367, -0.172646, -0.134195, -0.153395, -0.135626, -0.125796, -0.137222, 0.147786, -0.222872, 0.04639, 0.081922, -0.179693, 0.030258, -0.047099, -0.07151], [-0.193421, -0.09627, -0.505491, -0.077839, 0.314053, 0.26885, 0.047984, -0.131921, 0.006966, 0.001515, -0.15763, -0.169737, 0.00286, 0.12324, -0.257062, -0.291922, -0.298479, -0.183688, -0.319543, -0.272009, -0.067719, 0.182949, -0.09252, 0.088869, -0.070765, -0.224462, -0.259011, -0.043723, -0.119469, -0.181308, 0.128274, 0.046159, -0.120624, 0.266756, 0.235909, 0.023683, 0.214183, -0.024628, -0.135179, -0.231749, 0.068796, -0.181103, 0.151802, -0.027452, -0.336727, 0.063874, 0.012878, 0.45246, -0.196756, 0.087067, -0.209432, -0.201081, -0.120645, -0.180108, -0.041964, 0.256109, 0.158899, -0.126673, 0.360352, -0.190195, -0.10674, -0.308577, -0.178734, -0.017274], [-0.303037, -0.031772, -0.193293, 0.187611, -0.137795, 0.153657, 0.143802, -0.091069, 0.157514, -0.033672, 0.100099, 0.11482, -0.011178, -0.068135, -0.042407, -0.15323, -0.017079, 0.02427, -0.102345, -0.066078, -0.051793, 0.054319, -0.028253, 0.270999, 0.107802, 0.170117, 0.112265, 0.16226, 0.109779, 0.089655, -0.039067, 0.055077, 0.213052, 0.092544, -0.26888, 0.022801, -0.05791, -0.192986, 0.047938, 0.349345, 0.179612, -0.043266, -0.059648, -0.190949, 0.024888, -0.056783, 0.084339, -0.222642, -0.219885, -0.053007, -0.077269, -0.32969, -0.045918, -0.013954, -0.247929, 0.043255, -0.070511, -0.228296, -0.09746, -0.23152, 0.102133, -0.353605, -0.156121, -0.28949], [0.312254, 0.001056, -0.008402, -0.387731, -0.046031, -0.038096, -0.245364, 0.011292, 0.131117, -0.073595, -0.551403, -0.364778, 0.178439, 0.307756, -0.08459, 0.129069, 0.181671, -0.09103, 0.098621, 0.069273, -0.197043, 0.035657, -0.021633, -0.246886, -0.011663, -0.365424, -0.097267, -0.082247, -0.226919, -0.169896, -0.174681, -0.095446, 0.410021, -0.007408, -0.305138, 0.126361, 0.10353, 0.018835, 0.124259, 0.111805, 0.012708, 0.102236, 0.077257, 0.302041, -0.294966, 0.111947, -0.205895, -0.051752, 0.105963, 0.306071, 0.213307, -0.062338, 0.13462, 0.031034, -0.377361, -0.077381, -0.031996, -0.399328, -0.010019, -0.438699, 0.164145, 0.136484, -0.108498, -0.073004], [0.236621, 0.034837, -0.12525, -0.103675, 0.246097, -0.015223, 0.076857, -0.006222, 0.015306, 0.381195, 0.008515, 0.186766, 0.322102, 0.112006, -0.158539, -0.066627, 0.071856, 0.17915, -0.078697, -0.233569, 0.304482, -0.057984, 0.157533, 0.418199, 0.120646, 0.285626, 0.166487, -0.294078, -0.082876, 0.11014, -0.216683, 0.139878, 0.101568, 0.240644, 0.297185, -0.278651, 0.099235, 0.063913, 0.085953, -0.107699, -0.008687, -0.103682, 0.085582, 0.002556, -0.022802, -0.113384, 0.076231, 0.203372, -0.299365, 0.097789, 0.044693, -0.075479, -0.294113, 0.110505, 0.491668, 0.037199, 0.100692, -0.070388, 0.172514, 0.082617, 0.159725, -0.045083, 0.123998, 0.359075], [0.173084, 0.123083, -0.021648, -0.230391, -0.075086, 0.032428, -0.188971, 0.197002, 0.256841, 0.122797, -0.358526, 0.106267, -0.155611, -0.007153, 0.295263, -0.113205, 0.270711, -0.08514, 0.206924, -0.176017, 0.056795, 0.029632, 0.192627, -0.277073, -0.097342, 0.146121, 0.135431, -0.104727, 0.02564, 0.027467, -0.172976, 0.121344, 0.204229, -0.26865, 0.208084, -0.155258, 0.092154, 0.178455, -0.074735, -0.145922, -0.146929, 0.098663, -0.010683, 0.155953, 0.182141, 0.07035, 0.013675, 0.060908, 0.062819, 0.161872, 0.001072, 0.050316, 0.094844, 0.103878, 0.208542, -0.061299, -0.000422, 0.05654, -0.26432, 0.014688, -0.155971, 0.515044, 0.186162, 0.509825], [0.148279, -0.127267, 0.040175, -0.005084, -0.188487, 0.216026, -0.134475, -0.185566, -0.27528, -0.168236, 0.061913, -0.296698, -0.120292, 0.147364, -0.117477, 0.033542, -0.137172, -0.197673, 0.04047, -0.001576, -0.146607, -0.098535, 0.166727, -0.111977, -0.279524, -0.19118, -0.15169, 0.015742, -0.189818, -0.026135, 0.073115, -0.019767, 0.226608, 0.145041, -0.022123, 0.095878, -0.058182, 0.090501, 0.336861, 0.102005, 0.117238, 0.034777, -0.087514, -0.045357, -0.088241, 0.010319, -0.044334, 0.161555, -0.055977, 0.017621, 0.207213, 0.110925, -0.027094, -0.053065, -0.25021, 0.025299, 0.048037, -0.229597, -0.159767, -0.00046, 0.085448, -0.178255, -0.10587, -0.067302], [-0.451404, 0.031299, -0.261868, -0.149539, -0.106585, 0.10044, -0.243121, 0.020575, -0.13222, 0.105127, 0.078906, -0.160804, 0.009232, 0.066725, -0.224252, -0.322436, -0.182892, -0.160317, -0.313023, -0.136279, 0.068399, 0.020659, -0.402987, 0.09574, -0.215352, -0.08212, -0.29774, -0.107023, -0.279895, -0.240592, 0.076303, -0.19289, 0.20788, -0.015021, -0.189795, -0.030178, 0.015527, -0.23334, -0.04297, 0.08991, 0.058598, -0.103887, 0.0814, -0.081756, -0.216563, -0.025531, 0.077037, -0.047687, -0.241647, -0.100144, 0.049581, -0.377014, 0.062034, 0.026818, 0.054878, 0.0302, 0.258979, -0.188841, -0.006725, -0.228854, -0.143009, -0.219516, -0.059163, 0.090698], [-0.063237, -0.068253, 0.305681, 0.157784, 0.031078, 0.054425, 0.116314, 0.193332, 0.093761, -0.097446, -0.07, 0.190356, 0.111262, -0.060728, -0.032365, 0.169696, -0.020338, -0.087575, -0.196896, -0.043387, 0.094672, -0.196682, 0.186084, -0.033724, 0.062715, 0.201674, 0.097033, -0.117083, 0.186984, 0.182367, 0.16606, 0.297153, 0.140088, 0.033895, -0.030205, 0.052617, -0.008566, -0.1612, 0.34617, 0.110436, 0.269332, 0.032918, -0.046995, 0.108825, 0.391665, -0.099166, -0.046167, 0.061657, 0.249107, -0.028958, 0.196508, 0.285823, 0.083971, -0.09569, 0.099858, -0.017441, -0.009343, 0.01385, 0.17535, 0.342362, -0.103143, 0.220737, -0.023317, 0.552409], [0.190987, 0.073126, -0.001371, 0.081381, -0.024074, 0.111869, 0.307265, -0.055561, -0.18783, 0.177178, 0.20803, 0.084608, -0.070661, 0.086408, 0.077207, 0.008652, -0.072075, 0.06409, -0.014662, -0.137579, 0.211847, 0.094401, -0.020995, 0.564156, 0.05454, 0.054107, 0.160673, -0.05733, -0.094089, 0.137486, 0.019058, 0.217643, -0.097927, 0.116296, 0.052457, -0.084612, 0.032654, 0.138488, -0.131466, -0.012154, -0.221541, -0.002069, -0.22161, -0.05028, 0.196938, -0.135896, 0.167902, -0.180438, 0.092647, -0.031986, -0.116596, 0.160443, -0.050841, 0.08626, 0.25354, 0.049012, -0.084286, 0.096966, -0.123603, 0.111287, -0.018798, -0.207264, 0.083073, -0.092882], [0.351276, 0.102422, 0.075722, 0.018559, 0.143748, 0.424665, 0.20003, 0.184736, 0.060952, -0.139983, 0.046587, 0.229794, -0.127921, 0.481082, -0.159758, -0.295768, -0.063599, 0.059545, -0.22092, -0.206876, -0.103852, 0.384868, 0.042159, 0.160078, -0.040922, 0.057667, -0.218981, 0.137952, 0.165102, 0.117076, -0.124268, 0.065767, -0.169675, 0.0341, 0.323108, 0.079443, 0.145184, -0.210828, 0.084036, -0.220424, 0.022984, -0.203897, -0.019184, 0.171414, -0.013262, 0.224892, 0.122954, 0.402145, 0.07376, 0.030909, 0.00911, -0.172739, 0.092597, -0.087457, 0.22956, -0.142004, -0.094843, -0.157458, -0.05032, 0.553785, -0.31635, 0.286303, 0.171685, 0.243492], [0.070498, 0.115178, -0.247459, 0.044428, 0.191654, 0.129066, -0.036798, -0.013348, -0.152346, 0.102954, -0.006354, -0.002457, 0.042033, -0.140666, -0.173745, -0.008236, 0.082324, 0.170939, -0.012431, 0.066579, -0.022657, 0.019944, -0.083915, -0.029292, -0.095848, 0.078508, 0.041188, 0.139908, -0.458413, -0.136535, -0.125189, -0.089783, -0.243167, 0.003124, 0.1649, 0.170708, 0.048878, 0.022906, -0.117047, 0.035413, -0.18618, -0.152764, 0.161792, -0.168798, -0.132479, -0.14437, -0.004476, 0.186078, -0.131838, 0.102001, 0.033158, -0.095537, -0.1171, -0.082379, 0.087304, -0.015629, 0.076202, 0.004412, 0.018682, -0.108, 0.145273, -0.21375, 0.034608, -0.063797], [-0.083253, -0.021884, 0.142906, -0.043746, 0.06997, -0.036066, 0.029283, 0.089816, 0.220085, 0.096255, -0.032316, 0.222628, 0.387958, -0.187373, -0.004555, 0.032637, 0.004699, 0.204256, 0.242006, 0.024122, 0.166352, -0.041903, 0.089359, 0.03764, -0.110675, 0.153841, 0.128984, -0.107566, -0.172648, 0.242793, -0.301713, -0.115462, -0.130593, 0.054801, -0.01237, 0.096607, -0.134928, 0.142548, -0.037487, 0.084745, -0.318693, 0.017425, 0.136885, 0.042092, -0.046528, -0.195335, -0.126061, 0.017843, -0.099998, 0.08828, 0.02267, 0.079085, -0.218947, -0.058777, 0.201733, -0.065961, -0.026882, 0.113973, 0.113916, 0.256857, 0.12832, -0.018009, -0.050858, 0.12831]], [0.012519, -0.192561, -0.2426, 0.046011, 0.018563, -0.239061, -0.229209, -0.261331, 0.086859, 0.113633, -0.060413, 0.093928, -0.037923, -0.417058, 0.085263, -0.123357, 0.056338, 0.088054, 0.13655, 0.114712, 0.060067, -0.150522, -0.111153, -0.060959, -0.053599, -0.039524, -0.05522, 0.146446, -0.213786, -0.049751, -0.020177, -0.082997, 0.086831, -0.058201, -0.14488, 0.281375, -0.122545, 0.091926, -0.097751, 0.084152, -0.067918, -0.018053, 0.212662, 0.030422, -0.159926, -0.039797, 0.03595, -0.283277, -0.073132, 0.061321, 0.037922, -0.034247, -0.259326, -0.103132, 0.160544, -0.078534, 0.20932, 0.239028, 0.14449, -0.124023, -0.055803, 0.009249, 0.072778, -0.097683]),
    ([[-0.007414, 0.204532, 0.16985, 0.222305, 0.167077, 0.253976, 0.196669, -0.127697, 0.084187, 0.069356, 0.005519, 0.107407, 0.093181, -0.324803, 0.265267, 0.020708, 0.303836, -0.014708, -0.205127, -0.023622, 0.079301, 0.010498, 0.125542, 0.136582, 0.075677, -0.10136, 0.076042, 0.018536, 0.067206, -0.058613, -0.012227, 0.168366], [-0.035629, 0.016395, 0.162516, -0.013272, 0.131775, -0.165987, 0.042382, 0.140097, -0.09989, 0.028539, 0.071424, 0.235188, -0.046756, -0.168376, 0.292526, 0.071794, -0.14276, 0.078488, -0.132681, 0.099623, -0.00542, -0.066798, 0.248236, -0.069397, 0.120738, 0.013194, 0.117729, 0.22446, -0.060331, 0.165182, -0.059471, 0.139822], [-0.055623, 0.472928, 0.040064, 0.000995, -0.107176, 0.037736, 0.359123, -0.146065, 0.200533, -0.368353, -0.228607, 0.130512, 0.176416, -0.199085, -0.000406, 0.045624, 0.363613, 0.113649, 0.008343, 0.004631, -0.095295, 0.103571, 0.169315, -0.015815, -0.087414, -0.098136, -0.234405, 0.108315, -0.465337, 0.140686, -0.200402, 0.140846], [0.076119, -0.303364, 0.229028, 0.007339, 0.079874, -0.162745, -0.283729, -0.056288, -0.02294, 0.114499, 0.0009, -0.104951, -0.033893, -0.419321, -0.069526, -0.079283, -0.30975, -0.155035, -0.011428, -0.009829, -0.115052, 0.01137, -0.045539, -0.279973, -0.085171, 0.04929, 0.13472, -0.150233, -0.027891, -0.189931, -0.0461, 0.155527], [-0.011454, -0.100164, -0.291138, -0.211934, 0.056452, -0.141793, -0.407226, 0.081755, -0.211032, 0.072735, -0.085973, 0.007556, -0.098799, 0.178423, 0.016009, 0.136459, -0.527331, 0.12778, 0.0184, 0.130248, 0.063693, 0.044951, 0.187337, -0.105071, 0.077695, 0.110432, 0.009003, -0.224022, -0.046264, -0.154969, 0.108093, 0.090909], [0.103708, -0.074199, -0.04775, -0.101426, -0.134696, -0.191551, 0.030132, -0.187565, 0.1278, -0.13395, -0.340774, -0.036338, 0.071465, 0.142276, -0.010893, -0.09274, 0.08618, -0.385812, 0.268747, 0.014935, -0.303194, 0.097648, 0.083278, -0.001167, -0.126664, -0.018641, 0.030979, -0.10344, -0.14659, -0.147954, -0.003253, -0.046714], [0.040988, -0.113454, 0.028558, -0.017476, 0.030832, -0.063335, 0.22479, 0.081317, -0.161558, 0.230695, 0.251016, 0.051408, 0.011047, -0.17782, 0.123438, -0.014295, 0.057577, 0.305747, -0.091208, -0.057629, 0.228002, 0.034709, -0.011673, 0.100419, 0.047053, 0.118954, -0.061595, 0.025879, 0.107835, 0.02369, -0.021471, -0.03326], [-0.037247, 0.167973, -0.02533, 0.097873, -0.01873, 0.040245, 0.252998, 0.017102, -0.007667, 0.142906, 0.145388, 0.081814, 0.017915, -0.04612, -0.226651, 0.097294, -0.033963, -0.073574, -0.146034, 0.013117, 0.075769, -0.090816, -0.264979, 0.021064, 0.078565, 0.060538, 0.156772, -0.033781, 0.014283, 0.049256, -0.116006, 0.019807], [-0.00123, -0.100103, 0.145398, 0.101059, -0.034318, -0.003093, 0.041882, 0.206623, 0.106853, 0.0494, 0.3617, -0.061623, -0.023918, -0.106121, 0.18873, 0.133303, -0.002688, -0.098521, 0.027823, 0.154592, 0.062907, 0.055128, -0.035608, 0.111074, 0.027227, 0.230817, 0.004691, -0.008017, 0.09075, 0.072492, 0.059726, 0.114135], [0.06329, -0.135072, -0.131856, -0.17001, -0.171938, -0.110073, -0.399001, 0.195692, -0.11357, 0.095867, -0.088037, 0.124355, -0.122225, 0.123984, 0.01053, -0.066945, -0.159074, -0.051991, 0.212054, 0.170866, 0.115927, 0.023493, 0.020659, 0.011882, -0.071632, 0.005464, -0.006141, 0.013599, 0.086648, -0.127897, 0.231828, 0.001749], [0.055632, -0.02368, -0.08892, -0.074178, -0.07529, 0.024926, -0.062092, 0.124357, -0.067828, 0.033325, 0.041072, -0.034341, 0.026817, -0.104695, -0.098591, -0.078269, -0.189601, 0.034289, -0.294124, 0.047353, -0.069748, 0.073827, 0.028071, -0.366363, -0.14943, 0.080525, -0.073267, -0.144273, -0.041046, -0.021676, -0.114409, -0.027013], [-0.044226, 0.052061, -0.001412, 0.05998, 0.168031, 0.039433, -0.206646, 0.163703, 0.025852, 0.159282, 0.32208, 0.050682, 0.108802, -0.077566, 0.135573, 0.004098, 0.022388, -0.013514, -0.125531, 0.028629, 0.161752, 0.009935, 0.091684, 0.122844, -0.056468, 0.118955, 0.036078, 0.108934, 0.127219, 0.116049, -0.045934, -0.004149], [0.078582, -0.023617, 0.0158, -0.261737, 0.107391, -0.145601, -0.193403, 0.069688, 0.033605, 0.138185, 0.014472, -0.217155, -0.2477, 0.013102, -0.150908, -0.334727, -0.095505, 0.041657, 0.079153, 0.142953, 0.004754, -0.07091, -0.000593, -0.024005, -0.233475, -0.038521, 0.06599, -0.008745, 0.150456, -0.038577, -0.231154, -0.242053], [-0.103833, 0.269018, 0.208491, 0.101752, -0.089023, 0.307015, 0.388371, 0.001021, 0.064765, -0.010816, 0.058477, 0.200222, 0.177836, 0.091116, 0.198655, 0.039453, 0.303971, 0.105692, 0.077502, -0.085638, -0.027602, 0.046153, 0.05705, 0.091484, 0.126363, -0.12983, -0.239304, 0.184365, -0.020914, 0.013785, -0.13395, 0.032728], [-0.076464, 0.134146, 0.002393, 0.007055, -0.058614, 0.005313, -0.064426, -0.049027, 0.015505, 0.105616, 0.185628, 0.031558, 0.070474, -0.105056, -0.03426, 0.043238, -0.053071, -0.148192, 0.102945, 0.016577, 0.005958, 0.006841, 0.096234, 0.049873, 0.134079, -0.050599, -0.09886, 0.106311, -0.092423, -0.020894, 0.03409, -0.002248], [-0.035491, 0.207074, 0.1394, 0.013717, 0.022484, 0.198422, 0.03932, -0.122045, -0.043041, -0.03762, -0.223264, -0.090149, 0.139924, 0.166224, 0.026838, 0.142428, 0.191736, 0.025354, 0.114806, -0.156273, -0.059374, 0.090618, 0.143153, 0.113652, 0.037452, -0.146301, 0.077852, 0.048261, -0.220481, 0.156371, -0.102704, 0.026719], [-0.098083, 0.062385, 0.016129, -0.016234, -0.091841, 0.024581, 0.059853, -0.014332, 0.10809, -0.017703, -0.092118, -0.01772, 0.171471, 0.184388, -0.095639, 0.159354, -0.031729, -0.090027, 0.110999, -0.22577, -0.117586, 0.103704, 0.049617, -0.040551, 0.04428, -0.097519, -0.076161, 0.059176, -0.299672, -0.110924, 0.134323, 0.055879], [-0.096577, 0.031405, -0.100083, 0.039713, 0.040611, 0.054306, -0.199088, 0.049268, -0.0474, 0.065715, -0.088331, 0.097484, 0.059383, 0.017417, -0.070994, -0.052264, 0.062097, -0.015064, 0.156395, 0.001321, 0.154281, 0.080135, 0.036839, -0.17146, -0.168024, 0.109822, -0.006718, 0.064753, -0.000637, 0.021075, -0.129849, -0.008918], [0.006776, 0.113819, 0.065996, -0.04199, 0.000744, 0.028376, 0.29405, -0.222222, 0.171209, 0.015299, -0.027556, 0.076153, 0.130262, -0.105795, 0.074629, 0.162639, 0.167351, -0.242839, 0.021139, -0.079833, -0.024008, 0.187263, -0.027961, -0.008277, 0.17957, -0.498337, -0.06005, 0.118439, -0.12754, 0.105925, 0.185124, -0.022012], [0.055586, -0.201149, 0.055803, 0.071056, -0.122902, -0.047606, -0.146219, -0.120753, 0.000848, -0.064578, -0.264599, 0.014796, -0.028848, -0.154163, -0.064188, 0.065208, -0.2522, -0.130145, -0.140696, 0.0668, -0.249926, 0.09183, 0.039045, 0.123704, 0.02836, -0.246171, -0.078743, 0.063561, -0.149033, -0.065815, -0.076874, 0.139034], [-0.0542, 0.053871, -0.053099, 0.068585, 0.018166, 0.086891, -0.15911, 0.077594, -0.00358, 0.116103, -0.088753, 0.081089, -0.087456, 0.052314, 0.052509, -0.015038, -0.101538, -0.019939, -0.046189, -0.040783, 0.033567, -0.002196, 0.045493, 0.108856, 0.027625, 0.148316, 0.158482, -0.127889, -0.028234, -0.089225, 0.136165, -0.073376], [0.106353, 0.078732, -0.063501, 0.075248, -0.044459, -0.312088, 0.255203, -0.157194, -0.045871, -0.011566, -0.128424, 0.142859, 0.07784, 0.268542, 0.020731, -0.116363, -0.028854, 0.017754, 0.131645, -0.162415, 0.103602, 0.004293, -0.312877, -0.474016, -0.127484, 0.065236, 0.048911, -0.222204, 0.002003, -0.122707, -0.191281, -0.00886], [0.004427, -0.01031, 0.149928, 0.061901, -0.004107, 0.178293, 0.12545, 0.089255, -0.055232, -0.083404, 0.135588, 0.052226, 0.125589, 0.043396, 0.067966, 0.055447, 0.004359, 0.175921, 0.178863, 0.089178, -0.030227, 0.036755, -0.021562, 0.178411, 0.049663, 0.248865, 0.042948, 0.077421, -0.057061, -0.021894, -0.187077, 0.072777], [0.109084, -0.134572, -0.126523, -0.040603, -0.070454, 0.070557, -0.148213, 0.128979, 0.039137, 0.023834, -0.126551, 0.078433, -0.069905, 0.020342, 0.041031, -0.001241, -0.089687, -0.046685, 0.180838, 0.192381, 0.092903, 0.094356, 0.130015, -0.054903, -0.051442, 0.058037, 0.120707, -0.037383, -0.006101, -0.077638, -0.254376, -0.011361], [-0.118534, -0.138384, 0.115391, 0.096342, 0.075497, 0.113934, -0.194344, 0.019325, -0.008479, 0.087694, -0.107711, -0.054923, 0.086213, 0.083417, -0.108231, 0.12333, -0.155031, -0.18224, 0.016116, 0.004201, -0.126034, -0.007097, 0.02431, -0.032586, -0.036026, -0.103906, -0.028767, 0.124167, -0.131982, -0.044896, -0.121108, 0.10334], [0.043093, -0.17837, -0.121974, -0.220277, -0.002627, -0.099171, -0.05164, 0.019279, 0.050407, 0.125485, 0.011225, -0.020533, 0.015838, -0.103292, 0.036447, 0.093208, -0.063722, -0.067088, -0.16866, 0.01713, 0.101122, -0.065953, 0.004721, -0.190986, -0.045863, 0.075916, -0.006349, 0.037604, 0.006288, 0.096536, -0.445057, -0.047136], [-0.073502, 0.067153, 0.13504, 7.3e-05, -0.072055, -0.081764, -0.064141, -0.020647, 0.060731, 0.047884, 0.121062, 0.144369, 0.076861, -0.135777, 0.111711, -0.034839, 0.112385, 0.0097, -0.030275, -0.064329, 0.105492, 0.028165, 0.043996, 0.082334, 0.101662, 0.026681, -0.10401, 0.093326, -0.103393, 0.130327, 0.093088, 0.017963], [-0.163301, -0.055089, 0.002261, 0.035013, -0.080876, -0.031862, 0.032602, -0.149098, -0.219953, -0.036981, -0.125083, 0.005032, -0.052242, -0.215605, -0.019269, 0.042076, -0.006134, -0.015111, -0.10747, -0.162654, 0.032566, 0.112346, -0.168813, 0.015513, 0.085852, -0.196844, -0.134648, 0.083833, -0.095323, 0.091668, -0.122434, 0.028823], [-0.114465, 0.12182, 0.095793, 0.128641, 0.071017, 0.211605, 0.129004, -0.155293, 0.074846, -0.136884, -0.187727, -0.020246, 0.104823, -0.050672, 0.147279, 0.079441, 0.276126, -0.304004, -0.187693, -0.18983, 0.072804, 0.006513, 0.070982, 0.151352, -0.045744, -0.344471, 0.052006, 0.095223, -0.169298, -0.073067, -0.290041, 0.032239], [-0.001558, -0.002033, 0.09288, -0.00855, 0.031355, 0.096624, 0.140916, 0.093671, 0.098406, -0.026356, 0.045341, -0.085434, -0.068996, -0.172017, -0.062661, 0.191885, 0.326036, 0.087045, -0.110475, -0.01155, 0.025312, 0.134109, 0.042784, -0.176709, -0.130994, 0.118375, 0.130158, 0.035317, 0.145702, 0.060247, -0.224579, 0.007586], [0.112999, 0.004836, -0.003505, -0.109271, 0.041294, 0.004676, -0.086746, -0.107171, -0.025818, -0.117582, -0.248476, 0.138952, -0.088469, 0.027551, 0.049546, -0.125523, 0.02997, -0.170794, 0.033365, -0.141335, -0.077835, -0.084677, -0.21231, -0.098537, -0.062063, -0.316432, 0.033901, 0.106142, -0.146567, 0.067321, 0.002057, 0.010302], [-0.072066, 0.076256, -0.068297, -0.026122, 0.037642, -0.001286, 0.211361, 0.106237, 0.116797, 0.034991, 0.057119, -0.069236, -0.121523, 0.108382, 0.035756, 0.005384, 0.183966, -0.110404, 0.053021, 0.112844, 0.015527, 0.047233, -0.164773, -0.052003, 0.012577, 0.056168, 0.135434, 0.085596, 0.026768, -0.020977, 0.012611, 0.101728], [-0.060752, -0.005273, -0.140412, -0.12654, -0.054916, -0.290224, 0.060412, -0.137536, -0.096941, -0.198277, -0.579811, 0.022848, 0.003145, 0.0065, 0.095242, 0.061958, -0.214485, 0.019775, -0.041204, 0.068198, -0.040414, -0.13102, -0.081122, -0.106528, -0.076288, -0.240028, -0.018316, -0.119264, -0.1198, -0.152506, 0.011136, 0.059806], [0.062207, 0.032069, 0.059764, 0.132884, 0.010258, -0.007005, 0.183993, 0.178224, -0.300949, 0.083277, 0.02388, 0.087577, -0.065391, -0.102653, 0.09956, -0.042019, 0.082607, 0.294945, -0.125455, 0.118153, -0.011914, -0.138163, -0.08727, -0.081187, 0.11851, 0.01815, 0.157611, 0.131884, 0.134637, 0.065642, 0.021161, 0.016237], [0.078391, 0.12213, 0.029541, -0.098824, 0.143318, 0.076297, -0.118545, 0.131001, -0.066913, 0.068575, 0.232615, 0.063955, 0.118353, -0.172817, -0.111481, 0.101183, 0.04657, 0.083487, 0.183148, -0.009454, 0.110666, -0.104546, 0.051541, -0.286486, 0.092721, 0.153501, 0.136634, -0.263689, 0.021348, -0.041573, -0.020788, 0.125365], [0.030285, -0.109908, 0.097037, -0.063307, -0.289203, 0.078717, 0.098068, -0.057846, -0.112448, -0.176011, -0.317099, -0.071054, 0.129783, 0.046963, -0.106218, 0.075473, -0.125934, -0.33686, 0.080152, -0.105822, -0.037075, 0.009071, 0.032095, -0.008924, -0.007339, -0.132487, 0.049725, 0.055945, -0.177539, -0.001261, -0.046328, -0.033424], [-0.061936, -0.046646, -0.045309, -0.039356, 0.026904, 0.137612, -0.257936, 0.017653, -0.151342, -0.116345, -0.020097, -0.162605, 0.104608, 0.128309, 0.018912, 0.036997, -0.121319, 0.095536, 0.062851, 0.143375, -0.086109, 0.053775, -0.097485, 0.106515, 0.057052, 0.179912, 0.050835, -0.076978, 0.110129, -0.012219, -0.04921, -0.069133], [-0.0258, 0.037896, -0.023431, 0.164785, 0.00591, -0.14202, -0.033073, 0.046734, -0.09941, -0.032099, -0.023992, 0.106296, 0.094183, -0.081742, 0.086943, 0.085254, -0.199836, 0.158549, -0.056389, 0.120196, 0.013304, 0.063012, -0.037069, -0.028208, 0.032781, 0.011362, -0.012481, 0.097151, 0.141274, -0.184067, 0.133479, 0.077703], [-0.008953, -0.161582, 0.010654, 0.027135, -0.230505, 0.218164, -0.251129, -0.080383, -0.017772, -0.172983, 0.03258, 0.117723, -0.058165, -0.352865, -0.007851, 0.029044, -0.190904, 0.187459, -0.107201, -0.006595, 0.157299, 0.126667, 0.25709, 0.077549, 6.7e-05, -0.178376, -0.078475, -0.015145, 0.064977, -0.115563, 0.163116, -0.03099], [-0.032046, -0.276751, -0.206767, 0.002992, -0.035472, -0.26521, -0.170771, 0.013618, 0.054838, -0.155909, -0.073696, -0.32387, 0.064439, -0.192688, -0.123502, -0.066194, -0.274292, 0.234243, -0.450218, 0.010107, 0.047135, 0.106764, -0.1528, -0.102231, 0.046336, -0.082273, -0.070222, -0.119223, 0.002133, 0.018167, -0.012253, -0.041866], [-0.005099, -0.18882, -0.20627, 0.022502, -0.098433, -0.041243, -0.281348, -0.184641, -0.035059, -0.03143, -0.080565, -0.170913, 0.089635, 0.061361, -0.169156, -0.088406, -0.471942, -0.502713, 0.054559, -0.20428, -0.128616, -0.131168, -0.131091, -0.060901, -0.214866, -0.088264, -0.188962, -0.109581, -0.142967, 0.036076, -0.269331, 0.027621], [0.037421, -0.000111, -0.00301, 0.124737, -0.044234, 0.03446, -0.121254, -0.007496, -0.035264, -0.057189, -0.061931, 0.005306, -0.041912, -0.094823, 0.058176, -0.07753, 0.10702, -0.12768, -0.105575, -0.077737, -0.096995, -0.042187, -0.066428, 0.185393, 0.145082, -0.129371, -0.070134, 0.075778, 0.031937, 0.084254, 0.00745, 0.091365], [-0.070528, -0.159066, -0.465891, 0.057411, -0.145847, -0.437284, 0.052677, 0.050125, -0.033327, -0.039283, 0.16869, -0.074094, 0.079857, 0.084318, -0.379655, 0.142943, -0.346801, 0.084429, -0.043371, 0.092908, -0.028525, -0.056825, -0.291814, -0.517054, 0.020534, 0.154229, 0.078364, -0.135323, 0.151645, 0.056894, 0.018451, 0.091089], [0.011785, 0.08683, 0.024055, 0.130254, 0.040666, -0.011464, 0.015228, -0.115669, 0.028665, -0.090428, -0.36087, 0.020353, 0.095917, 0.141036, -0.05909, 0.106725, 0.144022, 0.011103, 0.145636, 0.046828, -0.291825, 0.04377, -0.027218, -0.052931, 0.070788, 0.017645, -0.081948, 0.059664, 0.00146, -0.033527, -0.07838, 0.00268], [0.131427, 0.089804, 0.036255, -0.118492, 0.213458, 0.011039, 0.051294, 0.099408, 0.032982, 0.217566, 0.145489, -0.042293, 0.011021, -0.278019, -0.001965, -0.042, -0.105607, 0.108601, 0.006114, 0.081099, 0.090035, -0.058366, -0.369613, -0.178896, -0.196694, 0.087911, 0.043136, -0.051047, 0.135241, 0.090982, 0.0857, -0.107283], [0.036028, 0.077838, -0.046106, -0.055591, -0.019734, -0.022467, 0.005917, -0.040146, -0.090513, 0.013198, 0.134208, 0.139579, -0.025941, 0.214761, 0.022023, 0.004945, 0.0889, -0.1428, -0.149787, -0.052969, -0.080802, -0.003852, 0.00689, -0.12507, 0.105646, 0.090677, -0.089956, -0.147893, -0.079637, 0.112015, 0.136193, 0.071419], [0.023491, -0.097778, -0.01145, -0.091868, 0.050572, -0.18909, -0.449889, -0.060924, -0.227819, 0.086038, 0.074069, -0.065774, 0.15079, 0.012684, 0.025352, -0.084979, -0.157327, 0.02859, 0.02338, -0.146273, 0.213407, 0.128148, 0.137734, 0.043287, 0.020595, -0.072816, -0.029577, -0.012582, -0.15555, -0.009715, -0.005472, -0.090754], [0.216658, -0.114202, -0.059449, -0.085142, -0.012308, 0.098262, 0.079793, 0.120972, -0.048275, 0.050796, 0.208379, -0.16301, 0.052512, 0.199762, -0.084088, 0.009245, -0.014697, 0.057233, -0.022645, 0.209927, 0.12004, -0.193116, -0.101883, 0.058463, -0.066493, 0.34942, 0.265548, 0.109155, 0.09922, 0.134109, -0.161735, -0.042185], [0.013628, 0.041921, 0.004925, 0.133526, -0.002688, -0.021769, 0.309031, 0.072712, 0.004556, -0.277141, -0.207265, 0.081818, 0.174417, -0.094811, -0.113096, -0.068979, 0.216721, -0.042086, 0.009206, 0.143174, -0.135178, 0.086028, -0.047789, -0.138169, -0.025159, -0.122362, -0.126735, -0.075739, -0.073881, 0.142844, -0.19384, -0.046656], [0.050239, -0.103751, 0.093858, 0.102183, -0.15628, -0.051577, -0.098299, -0.392323, -0.024795, -0.296435, -0.364961, -0.030278, 0.096944, 0.156945, 0.003784, 0.045755, -0.10504, -0.170582, 0.074816, -0.037192, -0.209194, 0.062886, 0.056595, 0.044798, -0.011716, 0.089498, 0.083503, 0.101253, -0.009376, -0.092729, 0.042428, -0.084685], [0.030468, 0.201284, -0.156695, -0.150583, -0.054346, -0.019167, 0.006575, -0.078503, 0.032146, -0.065914, -0.036941, 0.060923, 0.028677, -0.101827, -0.342998, -0.133806, 0.132621, -0.062389, -0.14889, 0.081211, -0.054509, -0.219379, -0.057099, -0.121253, 0.053622, -0.013309, 0.10513, -0.084977, 0.158567, 0.034944, 0.209895, 0.075337], [-0.135759, 0.164879, 0.172129, 0.024197, -0.029231, 0.096312, 0.180197, -0.136287, -0.0134, 0.012062, -0.22093, 0.018243, -0.022955, 0.105514, 0.11235, 0.135256, 0.187165, 0.082415, -0.098337, -0.019102, -0.026902, 0.068654, 0.080644, 0.053882, 0.173923, -0.034267, -0.118339, -0.037529, -0.081161, 0.169675, -0.123963, -0.024747], [-0.015593, -0.132146, 0.018674, -0.185571, -0.065287, 0.173485, 0.017828, -0.06565, -0.154879, -0.183941, -0.044729, 0.142201, 0.118398, -0.019563, -0.176043, -0.034656, 0.042851, -0.138808, 0.11437, -0.114268, 0.069073, -0.161862, -0.273723, -0.112412, 0.118763, -0.099363, -0.115362, -0.180204, -0.157061, 0.250162, -0.127214, 0.042035], [0.071889, -0.144444, -0.325986, 0.026156, 0.093611, -0.162416, -0.004164, 0.086806, 0.032919, 0.097668, 0.19438, -0.199092, -0.09133, 0.154632, -0.259067, -0.272277, -0.142431, 0.05554, -0.295512, 0.018284, -0.037775, -0.108403, 0.022518, -0.435836, -0.119803, 0.010227, 0.131437, -0.060212, 0.047534, -0.196697, -0.096422, 0.040891], [-0.113146, -0.058058, 0.125218, 0.013844, 0.163791, -0.084759, -0.064987, 0.156442, -0.164702, 0.173961, 0.252808, 0.040325, 0.154968, 0.019294, 0.165537, -0.075245, 0.059978, 0.359157, -0.117341, 0.218309, 0.145196, 0.06346, 0.02929, -0.018349, -0.093894, 0.198349, 0.047907, 0.085467, 0.070488, 0.10532, 0.020906, 0.013704], [0.032316, -0.036028, 0.043625, -0.093201, -0.056674, -0.005394, -0.013722, 0.077583, 0.048974, -0.100277, 0.120728, -0.140726, 0.006274, -0.188627, -0.148179, -0.171939, -0.097214, 0.034054, -0.112644, 0.176903, 0.137608, 0.012812, 0.21328, 0.125383, -0.167393, 0.135894, 0.125353, 0.106412, 0.079433, 0.096579, -0.145232, -0.169948], [-0.140358, -0.431965, -0.106917, -0.035705, -0.191689, -0.239804, 0.078457, -0.112113, -0.054805, -0.083792, -0.148954, -0.005261, -0.019725, 0.159816, 0.028336, 0.008528, -0.538424, 0.036092, -0.157136, 0.119226, 0.098456, -0.047655, -0.140795, -0.176965, -0.056949, 0.143124, -0.061504, -0.030403, 0.019377, -0.029869, 0.151156, -0.062068], [-0.067585, 0.199789, 0.117017, -0.085012, -0.185168, 0.035679, 0.267551, 0.011472, -0.073391, -0.14645, -0.206465, 0.052683, 0.095604, 0.108102, -0.029368, -0.039234, 0.066074, -0.332457, 0.284261, -0.216123, 0.030238, -0.062416, 0.097829, 0.210296, -0.042849, -0.061017, -0.12389, 0.112833, -0.308376, 0.089201, 0.213703, 0.074245], [-0.102618, -0.130633, 0.02328, -0.101008, -0.028543, 0.063962, 0.160037, 0.061986, 0.115617, 0.025471, 0.083518, -0.21119, -0.199176, 0.146744, -0.176292, -0.174479, 0.052254, 0.048123, -0.093459, 0.046631, 0.031685, 0.043764, 0.081009, -0.386155, -0.181151, -0.00703, 0.000447, -0.299792, 0.151815, 0.084778, -0.301913, -0.17315], [0.107987, 0.132901, 0.238721, 0.201429, 0.120965, 0.249872, 0.298502, 0.081732, 0.086063, 0.1244, 0.105538, -0.100801, 0.142484, -0.067658, 0.037565, -0.132277, 0.401086, 0.287465, -0.115599, 0.063688, -0.001955, 0.104441, 0.126661, 0.177399, -0.017208, 0.258084, 0.129825, -0.027871, 0.146173, -0.014144, -0.089923, 0.091101], [-0.068586, -0.151027, 0.069409, -0.036566, 0.057734, -0.165822, 0.146349, -0.078531, 0.064969, -0.101941, -0.165535, 0.047309, 0.125722, 0.023498, -0.109186, 0.113815, -0.289468, -0.054096, -0.030585, 0.098176, -0.064508, -0.080646, -0.255164, -0.018893, 0.054148, -0.172574, -0.030659, 0.046738, 0.12763, -0.002891, -0.474063, 0.077349], [0.179616, 0.352108, -0.092307, 0.193955, 0.033794, 0.324006, -0.044602, 0.112505, 0.039862, 0.06344, -0.017405, 0.088647, 0.048287, -0.035677, 0.191475, 0.022947, 0.390771, -0.11165, -0.301963, 0.16383, 0.058315, 0.065539, 0.155175, 0.093941, 0.025712, 0.135866, 0.176927, -0.039873, 0.141093, -0.14153, 0.179859, -0.010389], [-0.087216, 0.052162, 0.007901, -0.009209, 0.165961, -0.084357, 0.122697, 0.105563, -0.048692, -0.09057, 0.080791, 0.039361, 0.099515, -0.156017, -0.289097, -0.046924, 0.020395, 0.009391, -0.2852, 0.0756, -0.031538, -0.01246, -0.018761, -0.311248, -0.016765, 0.139178, 0.068376, -0.209185, 0.025923, -0.162313, -0.059039, -0.021588], [0.07176, 0.072823, -0.017937, -0.034084, 0.14432, 0.089423, 0.096403, 0.218251, -0.003765, 0.309348, 0.459256, 0.034026, 0.03411, -0.106665, 0.200535, -0.06282, 0.312892, 0.12646, 0.007588, 0.047695, 0.280431, 0.069506, 0.014984, 0.051189, 0.045475, 0.231834, 0.141958, 0.1448, 0.225512, 0.057144, -0.11665, -0.018556]], [-0.053793, -0.197068, -0.091146, 0.080524, -0.082114, -0.032102, -0.074633, 0.070846, -0.082551, -0.145511, 0.031158, -0.099342, 0.031519, -0.057337, -0.091019, 0.11965, -0.051309, 0.06506, -0.015237, -0.014644, 0.142869, 0.164036, 0.050239, -0.003224, 0.165876, -0.079542, 0.046195, -0.105767, 0.045443, -0.107678, 0.001341, 0.133015]),
    ([[0.034074], [-0.263169], [-0.138678], [-0.108307], [0.286416], [-0.30414], [-0.345304], [0.101464], [-0.134563], [0.291564], [0.172446], [-0.214555], [-0.078005], [0.174842], [-0.248383], [-0.115467], [-0.260555], [0.288568], [0.256746], [0.11886], [0.214492], [-0.113875], [-0.112986], [-0.244953], [-0.14661], [0.242861], [0.164286], [-0.190325], [0.180586], [-0.154701], [-0.332398], [-0.060354]], [0.032055]),
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
