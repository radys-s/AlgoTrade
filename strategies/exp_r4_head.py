"""mm_r4_mixv (home, R4 candidate): mm_r4_mix (R3 GRU + non-GRU tables fitted on round3 + round4 0-75 %) with the volatility-scaled taker signal and thresholds (pred_scale 1.15, gate_thr 24, slw_thr 3; R4 has 1.4-1.9x round3 volatility): slice A holdout 222,726 / 222,767 (gru3 189,536 / 203,350); walk-forward twin on slice B 241,953 / 249,133 (gru3 240,675 / 241,518).

mm_r4_mix (home, R4 candidate): mm_r3_gru3 (R3 GRU kept) with ridge rfit 19, slow ridges, taker MLP and 500 ms quote MLP refitted on round3 (all) + round4 0-75 % (+ mirrors; research/retrain_multi.py): slice A holdout 198,187 / 210,039 (gru3 189,536 / 203,350); its walk-forward twin exp_r4_mix_wf: slice B 240,040 / 249,204 (gru3 240,675 / 241,518).

exp_r4_mix: mm_r3_gru3 with ridge/slow/MLP/MLPQ refit on round3+round3_mirror (all) + round4+round4_mirror (first 75 %).

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
    # rfit 19 (multi: round3+round3_mirror+round4+round4_mirror, research/retrain_multi.py); params rfit=19
    (0.0169861, (0.3814539, -1.9124198, 4.0507163, -1.9727742, 1.1506434, -0.3190874, 0.010884, 0.1038778, 0.0046135, -0.0092874, -0.1531461, 0.2081766, 0.0400311, -0.0405483, -0.0537366, 0.0001202, 0.1560331, -0.4305973, 0.0004778, 0.0004052, -3.69e-05, -0.0005951)),
)
# exp_r3_mlp: 22-32-16-1 ReLU MLP on the same 22 features, target = 200 ms (robust mid at t+5ms+200ms - mid now), fitted on the
# first 75 % of round3 + mirror (research/diag/mlp_r3.py); holdout corr 0.49 vs 0.463 for the ridge. Inputs clipped at +-8 sd.
MLP_MEAN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.2432, 0.0, -0.0, 5.881423, 5.881551, 0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, -0.000355, 4.023237]
MLP_SCALE = [1.059764, 0.922894, 0.876378, 0.774188, 0.683643, 2.404117, 14.718479, 0.720885, 12.674707, 0.703674, 0.651477, 0.542767, 1.661212, 1.661459, 0.999997, 4.402524, 1.042989, 0.351861, 129.844147, 328.828094, 995.444214, 1.340413]
MLP_LAYERS = [
    ([[0.167737, 0.187968, -0.137388, 0.134786, -0.045501, 0.124601, -0.402818, 0.191184, 0.08205, -0.895144, 0.019977, -0.159408, 0.012598, -0.246979, 0.128134, 0.17084, -0.126864, -0.298794, 0.09293, 0.179686, -0.266784, 0.0867, 0.128622, 0.028522, 0.077595, -0.296583, 0.171259, -0.086596, 0.422049, -0.005077, 0.022484, 0.078917], [0.223727, 0.125992, -0.200133, -0.053732, -0.021447, 0.144178, 0.002697, -0.274733, -0.047937, -0.056509, 0.103894, 0.088339, 0.00714, -0.02412, -0.014676, -0.135893, 0.109601, -0.212796, 0.016091, 0.310103, -0.320552, -0.042949, 0.232096, 0.012034, 0.027313, -0.060835, -0.001957, 0.080005, 0.275295, -0.021941, 0.055134, 0.096902], [-0.286894, -0.158192, -0.044781, 0.027184, 0.233046, -0.238448, 0.170214, -0.066778, 0.077083, 0.235269, -0.078913, 0.25867, 0.158444, -0.140389, -0.076824, 0.267265, 0.130717, 0.300969, 0.001788, 0.14779, -0.28313, 0.067359, 0.301308, -0.225554, 0.223485, 0.354737, 0.031875, -0.059307, 0.351919, -0.268079, 0.277877, 0.225271], [-0.205486, 0.355362, 0.257852, 0.220462, 0.337585, 0.05289, -0.009061, -0.062303, -0.138065, 0.417642, 0.037646, -0.160119, -0.052266, 0.112601, -0.253011, 0.067211, -0.116219, 0.408682, 0.064772, -0.209434, 0.16194, 0.168483, -0.006436, 0.210506, 0.097104, 0.082558, 0.249359, 0.179293, -0.263654, -0.237996, -0.44527, -0.004382], [-0.588895, -0.094681, -0.185865, -0.057078, -0.0832, 0.087271, -0.139836, 0.06551, 0.013875, -0.233207, 0.124288, 0.009331, 0.419, -0.052362, 0.115878, 0.005438, -0.155965, 0.713827, 0.129833, 0.182989, 0.037378, -0.119728, 0.095126, 0.034896, -0.278876, -0.029736, 0.097816, 0.241113, -0.501853, 0.315688, 0.072694, -0.182511], [0.14954, -0.388543, -0.008837, -0.23438, 0.413165, -0.140481, 0.322025, -0.014161, -0.008468, 0.284929, -0.061019, -0.135125, 0.241875, 0.151066, 0.009351, -0.160342, 0.239139, 0.173936, -0.021583, 0.09033, 0.19451, -0.056423, -0.184497, -0.046197, -0.124555, 0.275172, -0.372085, 0.191886, -0.038661, -0.101997, 0.296123, -0.317826], [-0.143828, -0.4378, 0.133894, 0.675126, 0.597495, 0.168509, -0.580594, 0.180364, 0.177329, -0.305333, -0.536568, 0.858188, -0.346884, 0.049936, -0.30348, 0.069209, -0.141229, 0.284537, -0.201625, 0.01443, -0.225471, 0.505668, -0.103025, 0.632152, -0.039504, 0.492465, 0.240614, 0.273858, 0.364527, 0.163308, -0.144025, 0.337032], [0.049601, -0.045581, 0.034438, -0.088092, -0.076497, -0.16104, 0.170714, -0.048016, 0.169244, 0.021446, -0.163429, 0.210974, -0.131506, -0.102431, 0.050432, 0.092231, 0.224199, 0.018182, -0.08137, 0.085771, 0.100208, 0.385577, 0.111765, -0.048118, -0.009352, -0.00251, 0.213842, -0.08766, 0.024004, 0.408504, 0.238074, -0.153006], [-0.088436, -0.109392, -0.418074, -0.327742, -0.324618, -0.734408, -0.216597, -0.63453, -0.513865, -0.386962, 0.061837, -0.046901, -0.106374, -0.467263, -0.270429, 0.079684, -0.611138, -0.132174, 0.133154, 0.152344, -0.599398, -0.056381, 0.368596, 0.085478, -0.183194, -0.213947, -0.502669, -0.057804, 0.502943, -0.263842, 0.090223, 0.281855], [0.290202, 0.079045, 0.116823, -0.129744, -0.182139, 0.005205, 0.176174, -0.781784, 0.213461, -0.152264, 0.008316, 0.125531, -0.006854, -0.349191, 0.234614, 0.131106, -0.410424, 0.195165, -0.074582, -0.822895, -0.216891, 0.234729, -0.136134, -0.004295, 0.140571, -0.080694, -0.012735, 0.135504, -0.123424, 0.248097, 0.18866, -0.568377], [0.009675, 0.055763, -0.154284, 0.142879, 0.031863, -0.032371, 0.146688, 0.025003, 0.186769, -0.109686, 0.00957, -0.15052, -0.050894, -0.025394, -0.020109, -0.132641, -0.152346, -0.050047, -0.078104, -0.028983, -0.058118, -0.045159, 0.058031, -0.061965, -0.178661, 0.020965, -0.000277, 0.008046, 0.024799, 0.106992, -0.179467, -0.065945], [-0.012211, -0.024465, -0.375052, -0.55697, -0.451197, -0.277888, -0.118554, -0.151675, -0.75211, 0.084656, 0.189457, -0.042457, 0.28504, -0.35131, 0.103716, 0.253698, -0.001659, 0.020496, 0.871329, 0.246459, -0.058868, 0.119596, 0.230259, 0.069616, 0.438707, -0.147872, -0.34721, -0.142067, 0.161009, -0.419308, 0.090886, 0.417671], [-0.090496, -0.183459, 0.041298, -0.067951, -0.081229, 0.06989, -0.123544, -0.27019, -0.052236, 0.086084, 0.367271, -0.086811, -0.210137, 0.162085, 0.015762, 0.489837, 0.210928, -0.033389, 0.135099, -0.157999, 0.106368, -0.112738, -0.101558, 0.246944, -0.200326, -0.076242, 0.057999, -0.138295, -0.10916, 0.064592, -0.297804, 0.016123], [-0.052887, -0.05072, -0.30413, -0.172321, 0.17841, 0.041997, -0.196681, -0.038522, 0.249196, -0.125436, 0.318026, -0.343398, -0.446813, -0.108333, 0.10543, 0.257452, 0.120008, 0.023009, 0.05966, -0.032263, 0.261663, -0.505288, -0.139167, -0.027428, -0.057096, -0.002374, -0.100388, -0.128112, 0.04754, -0.191978, -0.012841, -0.318078], [0.067444, -0.099249, -0.66192, -0.560911, -0.35049, -0.424088, 0.70201, -0.414331, 0.275078, -0.14894, 0.136532, -0.306775, 0.134327, -0.691978, -0.657714, -0.162446, -0.006833, -0.134834, -0.37608, 0.338216, -0.114399, -0.05505, 0.130381, 0.164685, -0.071152, -0.218986, -0.325576, -0.299694, 0.153614, 0.396434, -0.1804, 0.335843], [-0.002731, 0.123591, 0.105368, 0.287228, 0.045613, 0.133539, -0.216216, 0.051392, -0.063407, 0.17179, -0.224123, 0.144818, 0.136653, -0.007915, 0.221795, 0.160093, 0.080865, -0.053171, 0.281739, 0.181242, 0.101319, -0.02234, -0.150872, -0.104537, 0.140201, 0.29673, -0.024131, 0.258796, -0.175275, -0.396897, 0.206523, -0.308989], [-0.012253, 0.161414, 0.60046, 0.287586, -0.685566, -0.728524, -0.10674, 0.877053, -0.38239, 0.070558, 0.059856, 0.279842, -0.034744, 0.184467, 0.764707, 0.082535, -0.201695, -0.089445, 0.607131, -0.280961, -0.043402, -0.251791, -0.335526, -0.251452, 0.470467, -0.233193, 0.829516, -0.159901, -0.086339, -0.156788, 0.39536, 0.478699], [0.241762, 0.120645, 0.372967, -0.70362, 0.166897, -0.228166, -0.331561, -0.318888, 0.141117, 0.074461, 0.261362, -0.203933, -0.283429, -0.177152, -0.069184, 0.401367, -0.565647, -0.20813, 0.010511, -0.194778, 0.146011, 0.180047, 0.595036, -0.152062, 0.374834, -0.02455, -0.07564, 0.119928, 0.583717, -0.127012, -0.234129, -0.518469], [-0.051511, -0.023471, 0.125186, 0.122722, 0.046992, 0.075995, -0.196196, 0.032661, 0.103953, -0.04841, 0.083926, 0.067757, -0.02135, 0.235931, 0.044153, -0.314894, -0.042521, 0.054416, -0.054051, -0.108749, 0.078198, -0.127643, 0.03834, -0.050536, 0.13035, 0.057141, -0.064681, 0.295549, 0.020935, 0.03474, 0.090572, 0.042831], [-0.282291, -1.280338, -0.01163, 0.078474, -0.033576, -0.123178, -0.102453, -0.317946, -0.409666, 0.007488, -0.136496, -0.024172, -0.11998, -0.175467, 0.17267, 0.39104, 0.133294, 0.00287, -0.302082, 0.192873, -0.273834, -0.169636, 0.375012, -0.103551, 0.031855, 0.492742, -0.66037, 0.119837, -0.053744, 0.398227, 0.058159, 0.188947], [0.151893, -0.051194, 0.087648, -0.060848, 0.0038, -0.105491, 0.200101, 0.028842, -0.076815, -0.011946, -0.167222, 0.044388, -0.152289, -0.131595, -0.017088, -0.341746, 0.16258, -0.183387, 0.0044, 0.075448, -0.216063, -0.093083, 0.066127, 0.012657, -0.5358, 0.040718, -0.132966, -0.283521, -0.046183, -0.229461, 0.189822, -0.079454], [0.039713, -0.134561, 0.063451, -0.247611, 0.057374, -0.020836, 0.057474, 0.000993, 0.003274, 0.027542, 0.112282, 0.852423, 0.011043, 0.260849, -0.555098, 0.163296, -0.498937, -0.110884, 0.182288, -0.176176, -0.09715, -0.0497, -0.346507, -0.825403, -0.000506, -0.103086, 0.012117, 0.108964, 0.074235, -0.19602, -0.012748, 0.49503]], [-0.319614, -0.158095, -0.03122, 0.188811, 0.040244, -0.009514, 0.198845, -0.306181, -0.646037, -0.125284, 0.6019, -0.206988, 0.02426, -0.173494, 0.110818, -0.054744, 0.3418, -0.219746, -0.696359, -0.395212, -0.180229, -0.525071, -0.141888, 0.072551, -0.230566, -0.728083, -0.026204, 0.777414, -0.081704, -0.022131, 0.042696, 0.057651]),
    ([[0.106602, -0.286832, 0.100834, -0.411439, 0.518966, 0.013067, -0.037714, -0.430896, 0.051929, 0.069895, -0.155523, 0.152984, -0.093256, -0.061116, -0.2918, -0.142227], [-0.119435, -0.090059, -1.259927, -0.568687, -0.259586, -0.964679, -0.018294, -0.212066, 0.135732, -0.015001, -0.134676, -0.020712, -0.166096, 0.199159, 0.009033, 0.061565], [0.185787, 0.096409, 0.080779, -0.000287, -0.45946, 0.364404, 0.405319, 0.045683, 0.274875, 0.012352, 0.178597, 0.18109, -0.169464, 0.247676, -0.090051, -0.148379], [-0.446847, -0.263101, 0.142408, 0.210128, -0.204448, 0.208913, 0.019631, 0.08472, 0.139165, 0.188216, 0.103477, 0.314771, -0.053994, -0.129772, 0.512334, 0.197692], [-0.240875, 0.311409, -0.369803, 0.035565, -0.118732, -0.241278, -0.155941, -0.265428, -0.225564, -0.004799, 0.042571, -0.392449, -0.171976, -0.191392, 0.06208, 0.368451], [0.082979, -0.172901, 0.341608, 0.101058, 0.17377, 0.33505, -0.040051, 0.224005, -0.006976, 0.164601, 0.231607, 0.208257, -0.034152, -0.010938, -0.367431, -0.513344], [0.084386, 0.11728, 0.101788, 0.084339, 0.118992, 0.034862, 0.530901, 0.209418, 0.079989, 0.223759, 0.110805, 0.280122, -0.120886, -0.460781, 0.180694, -0.597076], [-0.122174, -0.34584, -1.027533, -0.162344, 0.131991, 0.253831, 0.183372, -0.152877, 0.137549, 0.32572, 0.300194, -0.055636, -0.129052, 0.183803, 0.124502, 0.249929], [-0.047979, 0.278663, -0.268435, 0.033193, -0.5568, -0.450568, 0.079616, -0.158518, -0.355971, 0.087955, 0.120216, 0.4397, 0.002411, 0.013284, -0.272862, -0.309192], [-0.263991, -0.20083, 0.504441, 0.541107, -0.091448, 0.426495, -0.016144, 0.105167, 0.253861, 0.237973, -0.52854, 0.02921, -0.132174, 0.581109, -0.460537, -0.109538], [0.050366, 0.314193, -0.248904, -0.073001, -0.027912, -0.192936, 0.127598, 0.045687, -0.266473, 0.02128, -0.003061, 0.065037, -0.130101, -0.363347, -0.158573, -0.249831], [0.106088, -0.064971, -0.406235, -0.163241, 0.021555, -0.13851, -0.043786, 0.081094, -0.288831, -0.048672, -0.080694, 0.095374, 0.097963, -0.958411, -0.07869, 0.303275], [-0.193246, -0.047756, 0.020739, 0.099117, 0.222117, 0.012559, 0.239598, 0.188321, -0.091215, -0.248625, -0.186739, -0.69512, 0.032105, -0.194903, 0.310529, -0.163781], [0.413002, 0.025582, -0.171429, -0.123999, -0.074598, -0.313635, -0.096016, -0.058856, -0.012609, 0.293049, 0.095623, 0.06898, 0.146406, -0.537725, 0.105854, -1.242292], [-0.217511, 0.232331, 0.054952, 0.222225, 0.051362, -0.009079, -0.138373, 0.176921, -0.222924, -0.457316, 0.302266, 0.228906, -0.101501, 0.397442, -0.107827, 0.534408], [0.120544, -0.202424, -0.095044, 0.214286, -0.012249, 0.01396, -0.013232, 0.030865, -0.210774, -0.155356, 0.52361, -0.047573, -0.138381, 0.067415, 0.311215, 0.320212], [0.347272, 0.227949, -0.117549, -0.085301, -0.060507, 0.187239, 0.21247, 0.043253, -0.515059, -0.360215, 0.115984, -0.151864, -0.015887, -0.090166, -0.144903, -0.430222], [0.132442, -0.925802, 0.371942, 0.157157, 0.030731, 0.072252, -0.356912, 0.349055, 0.300339, 0.343646, -0.548985, -0.572697, 0.000869, 0.211764, -0.121343, 0.009316], [0.224772, 0.184278, -0.104883, -0.102594, -0.091712, 0.421735, -0.607486, -0.15487, -0.532567, 0.034393, 0.190384, -0.845032, 0.121418, 0.115897, -0.380617, 0.015782], [0.186083, -0.540784, 0.08067, 0.181834, 0.159576, 0.097511, -0.155034, 0.151672, 0.005542, -0.469394, -0.373453, -0.429388, -0.171071, 0.429485, -0.345736, -0.446399], [-0.064563, -0.037591, 0.096003, -0.070986, -0.010088, -0.781513, 0.04956, -0.314835, -0.05422, 0.032341, -0.183041, 0.036449, -0.022245, -0.395753, 0.149732, -0.869338], [0.053822, 0.188773, -0.15618, -0.009805, -0.053572, -0.279277, -0.218749, -0.024686, -0.169501, 0.135578, 0.083385, 0.202335, 0.020143, -0.401388, -0.507919, 0.285397], [0.027474, -0.06916, 0.197428, 0.208148, 0.26487, 0.150313, 0.034954, 0.182916, 0.03257, 0.244455, 0.135057, 0.24823, -0.094567, -0.217599, 0.059605, 0.005982], [-0.035487, 0.069867, -0.16562, -0.218982, -0.159905, -0.254453, 0.236883, 0.02452, 0.329523, 0.369426, -0.043937, -0.116793, -0.08728, -0.162384, 0.091846, -0.033453], [-0.252758, 0.262241, 0.167348, -0.161605, -0.027447, 0.116994, -0.080836, 0.044309, 0.201639, 0.079097, 0.291829, 0.049651, 0.018616, 0.151874, 0.122183, -0.466072], [0.168751, 0.245102, -0.091662, -0.356182, -0.312924, -0.053741, -0.771395, -0.276586, -0.147301, 0.569786, 0.409548, 0.512965, -0.164575, -0.173053, -0.400227, 0.062886], [-0.050703, 0.428238, -0.165027, -0.299126, -0.341997, -1.014854, -0.047002, -0.185258, -0.171224, -0.107592, -0.081459, -0.283446, 0.009471, -0.339218, -0.042036, 0.094896], [0.09001, -0.050602, -0.115939, -0.182723, 0.183156, 0.056999, 0.135814, 0.054567, 0.051336, -0.468332, -0.214647, -0.583703, -0.13293, -0.08298, -0.13682, 0.02136], [0.2874, -0.263244, 0.323982, 0.185909, 0.20717, 0.253573, -0.00632, 0.057213, 0.443971, 0.286647, -0.049734, 0.290883, 0.029736, 0.070768, -0.08591, 0.037799], [-0.314658, -0.487491, -0.043216, 0.144961, 0.225052, 0.013762, -0.322395, 0.361777, 0.149739, 0.380119, 0.033895, -0.298843, -0.202414, 0.017905, -0.211345, -0.148079], [-0.213032, 0.168955, 0.135823, 0.129177, 0.038303, -0.213462, 0.182501, 0.00431, -0.562407, -0.10154, 0.073568, -0.360132, 0.049961, 0.188143, 0.116007, -0.18722], [0.45572, -0.03268, -0.138385, -0.103306, 0.042991, -0.072638, 0.399731, 0.375594, -0.225707, -0.495638, 0.201068, -0.118547, -0.172599, -0.222437, -0.162601, 0.166984]], [0.134368, 0.257774, 0.014342, 0.186677, -0.065266, -0.316538, 0.364659, 0.309315, 0.278653, -0.426298, 0.177742, -0.03967, -0.223774, -0.058486, 0.509364, -0.241451]),
    ([[0.248405], [0.43418], [0.538626], [0.330013], [0.325395], [0.673142], [-0.36096], [0.378976], [-0.577311], [-0.735859], [-0.312503], [-0.50694], [0.266693], [0.741452], [0.511984], [1.192671]], [-0.34408]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 3.235551291250151e-19, -4.853326936875226e-19, -2.426663468437613e-19, -8.088878228125378e-20, 4.853326936875226e-19, 0.0, 0.0, 0.0, 1.2426752312925702, 0.0, 7.937122146241367, 20.493922322125467], [5.615629838862812, 15.496747018120706, 26.746166564746794, 34.38704311432976, 45.02951410728161, 1.0619958314291302, 0.9254505191878091, 0.878579481743969, 0.7763521174650135, 0.6856217269834721, 997.7126395327831, 1843.6519936918294, 3471.520342042865, 0.5904620416703459, 0.35469850161893585, 13.309818187259422, 27.61282097144313], [0.30001558606461376, 0.36697157409702663, -0.11477796762850129, -0.10685908463199831, -0.05279911810478326, 0.7629809894672391, -3.558942541001653, 6.348834100309926, -3.741642497920976, 2.4767125849278577, -0.21428611432845168, -0.0423323650967772, -0.19513322043255904, -0.004700709949453424, 0.4634823130081002, 0.0007223818556955333, 0.0003723681100327979], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 3.2380072050901566e-19, -4.857010807635235e-19, -2.4285054038176173e-19, -8.095018012725392e-20, 4.857010807635235e-19, 0.0, 0.0, 0.0, 1.242832373852272, 0.0, 7.940945453768678, 20.50454284910692], [5.617590421166445, 15.50237754368255, 26.75590670122211, 34.3993347429808, 45.0431531895883, 1.062235608435713, 0.9256629506811449, 0.8787796907040197, 0.7765231987188638, 0.685773488800987, 997.6489392772352, 1843.935419321767, 3471.5521787165553, 0.5906352374569319, 0.35483309131351326, 13.314093840978767, 27.620245352079028], [2.0565010218512936, 0.9506262372416432, -0.8315183984978434, -0.7473486128688933, -0.4314968636167152, 0.9466726688595271, -2.460659837994783, 5.17103440609462, -2.093185053845402, 2.283257210618055, -1.13368724269057, -0.07001180030011026, -1.2088337107942146, -0.003092841857178375, -0.12455466613503349, 0.00047515215503538406, 0.00024475095809617213], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.243216, 0.0, -0.0, 5.881366, 5.881495, 0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, -0.000321, 4.023269], [1.059803, 0.922929, 0.87641, 0.774215, 0.683666, 2.404207, 14.719129, 0.720911, 12.675267, 0.703701, 0.651475, 0.542766, 1.661134, 1.661457, 0.999997, 4.402571, 1.043015, 0.351876, 129.849274, 328.842102, 995.482361, 1.340429], [([[-0.11541, 0.042624, -0.046881, -0.072875, -0.079329, -0.031198, -0.044351, 0.147388, 0.267682, -0.578288, -0.120422, 0.080703, 0.18212, -0.154018, 0.15077, -0.208501, -0.032094, -0.351757, 0.121585, 0.087075, 0.002907, 0.238302, 0.046203, -0.060105, 0.046668, -0.0858, 0.244803, 0.067367, 0.929257, 0.052138, 0.078895, 0.055364], [0.228532, 0.248096, -0.161029, -0.007166, -0.629777, 0.229989, 0.110369, -0.220215, -0.067411, 0.000395, 0.227248, -0.002087, -0.216033, 0.027483, 0.074711, -0.110676, 0.075241, -0.062173, 0.053095, 0.332582, -0.126807, -0.198581, 0.193284, -0.023037, -0.086421, -0.071854, -0.048861, 0.044568, -0.016288, 0.188364, -0.022214, -0.059585], [-0.13132, -0.017988, 0.062113, 0.001434, -0.234778, 0.036762, 0.336419, -0.038283, 0.179105, 0.122147, 0.031076, 0.08531, -0.220218, -0.112449, -0.114179, 0.364686, 0.033549, 0.41875, -0.054878, 0.184397, -0.132105, 0.008278, 0.252719, -0.134225, 0.129999, 0.331236, -0.005454, -0.210657, -0.061964, -0.014227, 0.210317, -0.01811], [0.009758, 0.365169, 0.342561, 0.400722, 0.315736, 0.172009, -0.015335, -0.103162, -0.024283, 0.195387, -0.101302, -0.107082, 0.00704, 0.133217, -0.274732, -0.208374, -0.077029, 0.261638, -0.017724, -0.288828, 0.192489, 0.357229, -0.165742, 0.371189, 0.389993, 0.156115, 0.2388, -0.084024, -0.282235, -0.385664, -0.432885, -0.044316], [-0.684216, -0.012542, -0.400801, -0.694909, 0.51735, -0.122428, 0.06859, -0.318549, -0.061553, -0.097093, 0.12091, -0.00996, 0.623277, -0.254818, 0.375717, 0.066279, 0.215558, 0.639178, 0.11162, 0.095233, -0.541058, -0.123832, -0.018401, -0.055039, -0.123815, -0.048973, 0.095898, 0.388891, 0.213566, 0.649968, 0.351456, 0.254802], [0.15674, 0.091349, -0.197634, -0.088785, 0.044219, -0.338233, 0.030482, -0.242724, -0.358424, 0.29279, 0.126492, -0.19526, 0.065471, -0.063003, 0.06849, -0.083191, 0.27995, 0.302467, 0.211532, 0.134877, 0.091203, -0.158658, -0.08921, -0.104219, -0.227699, 0.119279, -0.190366, 0.17641, -0.356902, -0.192294, 0.306565, -0.226498], [-0.016493, -0.215309, 0.517588, -0.450819, -0.193282, 0.504532, -0.225606, -0.214953, -0.212115, -0.814936, -0.23538, 0.539247, -0.121537, -0.070209, -0.482243, -0.070521, -0.992824, 0.30128, -0.299743, 0.141994, -0.55982, 0.139869, -0.100286, 0.330651, 0.112102, 0.22918, -0.636813, 0.117103, -0.006462, 0.071417, -0.119163, -0.076516], [0.034911, -0.03125, 0.015338, 0.006454, -0.070426, 0.014589, 0.012318, 0.14366, -0.048995, 0.133109, -0.272628, 0.089196, -0.151429, -0.200495, -0.027233, -0.04351, 0.090847, -0.050091, 0.108365, -0.047888, -0.083299, 0.207051, 0.10833, -0.034691, -0.218409, 0.03222, -0.074859, -0.250806, -0.176055, -0.005509, -0.083008, 0.002303], [0.240183, -0.643501, -0.056929, 0.29418, 0.096668, -0.32486, 0.479809, -0.526416, -0.326839, -0.367984, 0.41659, -0.33369, -0.046289, -0.239198, -0.342949, 0.227261, -0.627322, 0.259769, -0.033086, 0.374068, -0.056987, 0.082826, 0.063759, -0.005443, -0.075893, -0.043857, -0.337783, -0.0184, 0.034749, -0.395939, -0.032635, 0.56911], [0.13514, 0.266398, 0.170947, 0.274175, -0.41493, -0.215812, 0.127356, -0.781198, 0.256524, -0.056629, -0.082874, 0.257235, 0.277324, -0.187757, 0.136272, 0.477055, -0.489645, -0.107915, -0.032123, -0.82357, -0.066932, 0.300174, -0.061988, -0.135248, -0.335392, -0.075575, 0.124303, 0.312288, -0.201509, 0.418487, 0.095982, -0.635265], [-0.093989, 0.039931, -0.002336, -0.037394, -0.14268, 0.229657, -0.159553, -0.238543, 0.086296, 0.043208, -0.274391, -0.122568, -0.100016, 0.107903, -0.082971, -0.12625, -0.136533, -0.003672, 0.305174, 0.020348, 0.26871, -0.027198, -0.003234, -0.070626, -0.049617, -0.10321, 0.085028, -0.170668, 0.008892, -0.109894, -0.282493, -0.176221], [0.078482, -0.015957, -0.231851, -0.077104, -0.005429, -0.203571, 0.701384, -0.100273, -0.253765, 0.096687, 0.532374, 0.103794, -0.168888, -0.362169, 0.269581, 0.0002, 0.063701, 0.116334, 0.258803, 0.153039, -0.167443, -0.05132, 0.277134, 0.089854, 0.167269, -0.610884, -0.247992, 0.013795, 0.030759, 0.024839, -0.185373, 0.395114], [-0.06422, -0.350497, -0.050344, -0.113418, 0.140301, -0.061023, 0.001333, -0.089136, -0.090621, -0.048377, 0.166066, -0.13079, 0.03658, 0.068093, 0.072699, 0.075159, 0.228096, -0.084453, 0.017159, -0.103483, 0.10429, 0.233942, -0.381758, 0.12999, -0.174962, -0.181692, -0.092741, -0.170303, -0.058386, -0.129045, -0.158159, 0.00317], [-0.049318, -0.170201, -0.122579, -0.158467, -0.079583, 0.032631, 0.200291, -0.091348, 0.103456, 0.102195, 0.153931, -0.082068, -0.161796, 0.296211, 0.082704, -0.229822, 0.269889, 0.068697, 0.157431, -0.215272, -0.124603, -0.032351, -0.162756, 0.102813, -0.149643, -0.038351, 0.001581, -0.031004, 0.038747, -0.194037, -0.003131, -0.307152], [0.171069, -0.090175, -0.144398, -0.362172, -0.265756, -0.486476, 0.374396, -0.027137, 0.231641, -0.057981, 0.112169, -0.374597, 0.168259, -0.276078, -0.383006, -0.248072, 0.00776, -0.130403, 0.092115, 0.15993, 0.036326, 0.253533, 0.083839, 0.196743, -0.268838, -0.620032, 0.364793, -0.039396, 0.147095, 0.26821, -0.041938, 0.142261], [-0.127068, 0.262361, -0.389452, 0.323218, 0.099652, 0.062952, 0.130716, -0.179436, 0.105324, 0.419365, -0.31581, 0.32673, 0.073423, -0.179683, 0.254885, 0.173807, 0.078427, -0.130402, -0.352134, 0.027662, -0.013619, 0.009265, 0.158647, 0.030961, -0.33609, 0.291058, 0.092776, -0.115557, -0.089875, -0.191053, 0.084861, -0.18271], [0.071805, -0.310221, 0.251568, 0.224205, 0.393944, 0.156661, -0.529237, 0.385359, -0.071672, -0.113218, 0.177309, 0.007709, 0.094947, 0.247442, 0.451186, 0.374969, -0.096275, -0.126599, 0.699698, -0.206498, -0.171058, -0.064535, -0.036841, -0.2567, 0.672446, -0.238907, 0.412707, 0.098482, 0.034731, -0.053586, 0.35549, 0.178366], [0.134135, -0.008081, 0.00864, -0.288657, 0.231482, -0.661534, -0.314487, 0.093299, 0.222562, 0.144892, -0.12855, -0.558898, 0.013921, -0.070821, 0.038629, 0.296729, -0.430303, -0.02088, 0.114896, 0.014302, 0.088464, 0.399377, 0.50424, -0.150196, 0.254732, -0.146933, 0.023694, 0.160136, -0.095552, -0.233103, -0.282141, 0.447409], [-0.027848, -0.062961, -0.180237, 0.032011, -0.00317, 0.157638, -0.197812, 0.044707, 0.21111, -0.135887, -0.073761, -0.121794, -0.060629, -0.016343, 0.043478, -0.092252, -0.083407, 0.006143, -0.063349, -0.080291, -0.281282, -0.007602, -0.087036, -0.048031, -0.03247, -0.019634, -0.137273, 0.017267, 0.074423, -0.029791, -0.039455, 0.005495], [-0.450925, -0.17276, -0.07954, -0.12422, 0.04273, -0.040519, 0.124061, -0.103427, -0.447802, -0.11251, 0.150579, 0.299196, 0.253762, -0.244738, -0.079274, 0.236591, 0.057456, 0.064058, -0.413469, 0.162661, -0.383137, -0.647043, 0.568028, -0.401078, 0.162114, 0.145424, -0.922983, -0.22312, -0.044911, 0.14399, 0.023423, 0.108758], [0.407513, -0.031865, 0.267356, 0.167154, -0.026859, -0.226368, -0.103705, 0.022854, -0.519704, 0.368978, 0.081292, 0.167492, -0.190978, -0.102556, -0.014401, -0.208638, 0.017049, -0.24717, 0.032475, 0.179884, -0.000303, -0.212431, 0.03903, 0.020093, -0.049859, 0.290353, -0.262606, -0.045171, -0.005934, -0.226047, 0.032793, 0.023305], [-0.014963, -0.098072, -0.197891, -0.087328, 0.024411, 0.194212, -0.037688, -0.017632, -0.02241, 0.041896, 0.129069, 0.549051, 0.026503, 0.267901, -0.478893, 0.074029, -0.304492, -0.151001, 0.149175, -0.085988, 0.017632, -0.025413, -0.081037, -0.709829, -0.134661, 0.26845, 0.202567, 0.495253, -0.010511, -0.01882, -0.267339, 0.471209]], [-0.513653, -0.702019, 0.154476, -0.185145, -0.113073, -0.006612, -0.65513, -0.48647, -0.714846, -0.819588, 0.554615, -0.48984, 0.111735, 0.12882, -0.135365, 0.109589, 0.499748, -0.293753, -0.747006, -0.2579, -0.247388, -0.424623, -0.044759, -0.242767, -0.291809, -0.356535, -0.10714, 0.309436, -0.174395, -0.233903, 0.083277, 0.029756]), ([[0.105808, -0.414375, -0.24155, 0.152402, 0.523111, -0.272792, 0.021613, -0.199358, 0.136534, 0.194749, 0.063203, 0.202599, -0.095116, 0.747097, -0.051443, 0.366513], [-0.045556, 0.130903, -0.642861, 0.053668, -0.486882, 0.0491, 0.32973, 0.089226, -0.083347, -0.215378, -0.016724, -0.010743, -0.202268, 0.069002, -0.141399, -0.061044], [0.116345, 0.183234, -0.000204, -0.26024, -0.225183, 0.120034, 0.323137, 0.114909, 0.129014, -0.07042, 0.110883, 0.261006, -0.205046, 0.13971, -0.258269, -0.512226], [-0.845135, -0.150092, 0.038772, 0.224027, -0.354642, 0.213332, -0.122686, -0.329492, 0.295428, 0.066137, 0.024045, 0.337517, -0.075052, 0.10007, 0.130729, -0.076519], [-0.309175, -0.209885, 0.276775, 0.359158, 0.120566, 0.209621, -0.232368, -0.238472, 0.500899, 0.407314, -0.167081, -0.407525, -0.201566, 0.122039, 0.299316, -0.062971], [0.272954, -0.062819, 0.344348, 0.270305, 0.231874, 0.300957, 0.400534, 0.121243, -0.097362, -0.033435, 0.297757, -0.040548, -0.038705, -0.034508, 0.134902, -0.524158], [0.084974, -0.033663, 0.500087, 0.217987, 0.23401, 0.102004, 0.115206, 0.263908, 0.043343, 0.289199, 0.100315, 0.628272, -0.122557, -0.419249, 0.081721, 0.105596], [-0.220254, -0.206455, -0.107038, -0.213277, -0.014417, 0.414374, 0.104923, -0.05383, 0.17637, 0.104162, 0.134562, 0.018685, -0.133023, 0.070114, -0.034481, 0.413725], [-0.010807, 0.441236, 0.069813, -0.309708, -0.490967, -0.245938, 0.244748, -0.243722, -0.048837, 0.118941, 0.042342, 0.326316, -0.005201, -0.472393, 0.122758, 0.187027], [0.427327, -0.099284, 0.293417, 0.449167, -0.142944, 0.262495, 0.609297, 0.547813, -0.111484, 0.348848, 0.786202, 0.257029, -0.132174, 0.143393, 0.439696, -0.276569], [0.019612, 0.370147, -0.027187, -0.107256, 0.098894, -0.123985, 0.190306, -0.022039, -0.120544, -0.524844, 0.177958, -0.006771, -0.206062, 0.308735, -0.107037, -0.34343], [-0.028477, -0.111016, -0.382037, -0.069827, 0.01841, 0.197778, -0.193113, 0.025195, -0.588775, -0.00128, 0.122742, 0.288954, 0.097963, 0.048818, -0.312559, -0.20127], [-0.043508, -0.656235, 0.071962, 0.11706, 0.160731, -0.03194, 0.052554, 0.175611, 0.151602, -0.216846, 0.258116, 0.308616, 0.032105, -0.124581, 0.331562, 0.272133], [0.023908, 0.064661, -0.10752, -0.762613, 0.046837, -0.3535, 0.013984, 0.067904, 0.050654, 0.256353, 0.212771, -0.078485, 0.137171, -0.2024, -0.233139, -0.121938], [-0.124443, 0.059647, -0.204347, 0.109237, -0.500995, 0.158471, -0.269433, -0.099687, -0.286613, -0.305106, -0.064847, -0.007202, -0.045901, 0.432074, 0.169446, 0.24053], [-0.055246, 0.26652, 0.053227, -0.003244, -0.113488, 0.266273, -0.031818, 0.146374, -0.322961, 0.259874, 0.109404, 0.131406, -0.141606, -0.356142, 0.000524, 0.17716], [0.212976, 0.113167, -0.184222, -0.233121, -0.120852, 0.093891, 0.375207, 0.255789, -0.589819, -0.129882, 0.128657, -0.364436, -0.015887, -0.499126, 0.069359, -0.337621], [0.110393, -0.844047, 0.238918, 0.209696, -0.007851, 0.246279, -0.448136, 0.128442, 0.376493, 0.444908, -0.550366, 0.112913, 0.000869, 0.086232, 0.13098, -0.001225], [0.130674, -0.316438, 0.013018, -0.122749, 0.041484, -0.080089, -0.724556, -0.368686, 0.024353, -0.092314, 0.119756, -0.631391, 0.086596, 0.451442, 0.004651, 0.262645], [0.124311, -0.167845, 0.218799, 0.113892, 0.203438, 0.098176, 0.041922, 0.279035, -0.051699, -0.418912, -0.493598, -0.037892, -0.18104, -0.345448, -0.077049, -0.115876], [-0.101271, -0.081932, -0.251427, -0.166421, 0.180094, -0.271289, 0.073525, -0.061546, 0.044861, 0.226885, -0.137487, 0.063043, -0.038983, 0.11918, -0.569184, 0.345166], [0.482844, -0.057961, 0.229088, -0.145635, 0.276075, 0.131651, 0.353846, 0.052346, -0.265852, 0.033279, 0.29645, 0.143179, 0.001531, -0.630995, -0.216132, -0.451039], [0.053531, -0.086558, 0.235725, 0.104997, 0.104585, 0.06408, 0.016849, 0.167716, -0.48835, 0.239159, 0.052619, 0.181148, -0.094567, -0.339607, 0.062014, 0.050614], [-0.241585, 0.039517, -0.089838, -0.023412, -0.662066, -0.324893, 0.110337, -0.079014, 0.323984, 0.170282, 0.043131, -0.111486, -0.149613, -0.280427, -0.155624, 0.088721], [-0.071648, 0.214069, -0.446754, -0.500823, -0.096849, -0.222106, -0.110967, 0.011574, 0.112152, -0.282337, 0.226522, -0.013561, 0.011321, -0.192194, -0.172775, -0.199188], [-0.10045, 0.163903, -0.361662, -0.326847, -0.036055, -0.318929, -0.282826, -0.278405, -0.17799, 0.191255, 0.210087, -0.296804, -0.164575, 0.114177, 0.080109, 0.10551], [-0.079828, 0.353049, -0.027054, -0.049603, 0.051893, -1.00526, -0.103092, -0.350895, -0.061205, -0.162798, 0.160733, -0.312416, -0.038623, -0.102965, 0.059551, 0.0044], [0.365354, -0.144495, -0.145209, -0.028567, 0.270301, 0.157341, 0.089668, -0.021855, -0.017634, -0.3209, 0.054306, -0.155973, -0.182974, -0.016684, -0.124478, 0.116843], [0.036122, -0.373261, 0.248902, 0.087888, 0.178318, 0.334694, 0.067843, 0.09854, 0.785346, 0.100745, 0.257163, -0.17939, 0.026672, 0.144607, 0.289176, 0.141633], [0.162362, -0.10951, -0.018315, 0.393643, 0.172685, -0.194763, -0.416879, 0.255716, 0.252828, 0.34097, -0.322864, -0.168465, -0.146855, -0.17133, -0.182178, -0.123853], [-0.027919, 0.265817, 0.095154, 0.120332, -0.172115, 0.202745, 0.208892, 0.157566, -0.154056, -0.311558, 0.130127, -0.188776, 0.04691, 0.198702, 0.120365, 0.108765], [0.187736, 0.071625, -0.233296, -0.099747, 0.090704, -0.139826, 0.26441, 0.193424, -0.352862, -0.187372, 0.175954, -0.086525, -0.204355, -0.52316, -0.326403, -0.537695]], [0.245942, 0.255591, 0.043167, 0.202775, -0.106738, -0.311385, 0.189967, 0.276025, 0.131119, -0.553269, 0.245762, 0.207298, -0.236216, -0.298434, 0.425395, -0.629562]), ([[0.446602], [0.920164], [0.742451], [0.768723], [0.748547], [1.27626], [-0.645524], [0.631514], [-1.344479], [-1.597193], [-0.497254], [-0.756138], [0.591977], [1.695626], [0.535391], [2.09858]], [-0.45827])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.73969, -1.6e-05, 1.6e-05, 0.0, 0.047376, 0.047376, 0.0, 0.0, 0.0, 1.231694, 5.907079, 5.907079, 0.0]
SEQ_TSD = [1.080247, 0.118222, 0.118222, 0.089617, 1.451691, 1.451691, 0.632855, 0.196772, 0.156216, 0.572964, 1.651027, 1.651027, 0.652184]
SEQ_XM = [-0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.233382, 0.0, -0.0, 5.906633, 5.906633, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.05788]
SEQ_XS = [1.020368, 0.879681, 0.830847, 0.723417, 0.628551, 2.281788, 12.978411, 0.674084, 10.704477, 0.755279, 0.651221, 0.545623, 1.651871, 1.651871, 0.999999, 4.399956, 1.05635, 0.427403, 127.463814, 323.861115, 938.76239, 1.34419]
SEQ_YSC = 1.490180253982544
SEQ_WI = [[0.229102, -0.40139, -0.528024, -0.330295, -0.366072, -0.521249, -0.162857, -0.526011, -0.623727, -0.537097, 0.076265, -0.251674, -0.169837, -0.263135, -0.097052, -0.21004, -0.183888, -0.057175, -0.111486, -0.173709, -0.27799, 0.371826, -0.219055, -0.042791, -0.20599, -0.173884, -0.189076, -0.118241, 0.085325, -0.003733, 0.02171, -0.167547, 0.021253, -0.06059, -0.160962, 0.114273, 0.05224, -0.1299, 0.001044, -0.106125, -0.151806, 0.065807, 0.021898, 2.5e-05, -0.130033, -0.022992, -0.25497, -6e-06], [-0.120357, 0.163209, 0.014291, -0.084067, 0.215118, -0.00412, 0.292222, 0.093146, -0.054728, 0.270186, -0.120017, 0.151667, -0.118864, -0.272867, -0.102464, 0.08224, 0.229617, -0.024143, -0.104143, -0.100867, 0.213904, -0.222547, 0.605152, -0.529798, -0.35376, -0.209398, -0.147999, 0.047944, -0.084492, -0.300755, 0.215656, 0.054365, 0.178046, 0.08393, 0.008386, 0.009949, 0.001503, 0.057073, -0.002392, 0.116808, -0.445775, -0.09849, 0.293669, -0.044832, -0.148245, 0.154914, 0.107693, 0.171752], [-0.197059, -0.085762, -0.159657, 0.123214, 0.071298, 0.184471, -0.140158, -0.036246, -0.190779, -0.136279, 0.131868, -0.126716, -0.08627, -0.0163, -0.156197, 0.139585, -0.367331, -0.459447, -0.354157, 0.104795, -0.425264, 0.274844, -0.066143, -0.021038, 0.17842, -0.034517, 0.137059, 0.112439, -0.552564, -0.049234, 0.254372, 0.031615, -0.144886, -0.215927, 0.11465, -0.098419, -0.0174, 0.030461, 0.362342, -0.122314, 0.138684, -0.321507, -0.099045, -0.211409, 0.077378, -0.009029, 0.410879, -0.09625], [0.058106, 0.127148, 0.03709, 0.123989, 0.278669, -0.12041, -0.038189, -0.146065, -0.324755, 0.205175, 0.115978, 0.11819, -0.303442, -0.198367, 0.159174, 0.232194, -0.224631, -0.190303, 0.000774, 0.266145, -0.134085, -0.102182, 0.058412, -0.12884, -0.306878, -0.065407, -0.194902, 0.165846, -0.304569, -0.258105, 0.028734, 0.221322, -0.011146, -0.019764, 0.094145, 0.22014, 0.026938, 0.210776, -0.133132, -0.125761, -0.048051, 0.045592, 0.009513, -0.085579, -0.060649, 0.320549, 0.003225, -0.103007], [-0.522245, 0.121013, -0.293836, -0.109994, 0.102355, 0.407511, -0.27909, 0.141521, 0.033621, 0.262275, -0.094713, -0.290751, -0.086302, -0.414082, 0.107421, -0.057036, -0.429157, -0.475561, 0.151911, 0.140959, -0.014049, -0.056747, -0.162072, -0.14468, -0.16876, 0.275552, -0.357762, 0.441312, 0.105367, -0.527128, 0.380701, 0.104254, -0.58126, 0.1158, -0.062427, -0.090359, 0.153838, -0.346968, -0.195041, 0.056019, -0.163649, -0.110766, 0.084558, -0.208565, -0.10653, 0.495258, 0.175766, 0.260163], [0.145879, 0.296968, 0.597929, -0.029928, -0.573815, 0.395683, 0.112871, -0.049482, 0.012181, -0.444898, 0.170041, 0.206143, -0.085015, 0.111929, -0.023617, 0.147129, 0.342582, -0.299453, 0.323605, -0.006738, -0.398414, -0.349693, -0.144026, -0.466554, -0.080044, -0.561031, 0.115395, 0.301334, 0.241202, 0.325084, -0.227408, 0.405896, 0.134782, 0.070304, 0.016274, -0.097398, 0.240112, -0.41944, -0.061512, -0.15, -0.041202, -0.633855, -0.199955, 0.005124, -0.000492, 0.094824, -0.102681, -0.002461], [-0.047475, -0.071425, -0.216282, 0.302595, -0.089898, 0.046074, -0.553033, -0.282011, -0.184008, -0.344442, 0.03231, -0.062297, -0.067477, -0.115351, -0.064201, -0.286953, 0.268692, 0.570535, 0.036506, -0.177719, 0.132087, 0.037678, 0.249666, -0.151281, 0.439513, 0.247822, -0.298792, -0.192223, -0.21023, 0.2225, -0.119548, 0.008794, 0.020793, -0.250043, -0.152116, -0.321315, 0.195864, -0.09312, -0.002162, 0.177035, 0.049869, 0.31609, 0.236254, -0.122939, 0.194693, -0.030974, -0.024717, 0.144314], [0.006433, -0.248603, -0.449655, -0.036608, -0.193793, 0.095991, -0.81439, -0.245997, -0.101984, 0.083959, -0.078489, -0.125357, -0.003672, -0.168635, -0.054407, -0.042764, 0.146655, 0.314931, 0.021866, -0.150356, -0.098764, -0.088633, 0.165, -0.23683, 0.229348, 0.321827, -0.245096, -0.022416, 0.008871, 0.114012, -0.010247, 0.118389, 0.173049, 0.076843, -0.16604, -0.02766, 0.235443, -0.112895, 0.016696, 0.182383, 0.008004, 0.272475, -0.048195, -0.086742, 0.064778, -0.093501, -0.131332, 0.246849], [0.077361, -0.28025, -0.520509, 0.239421, -0.337946, -0.068581, -0.639934, -0.248012, 0.053045, -0.107412, -0.243008, 0.196074, -0.223366, -0.253246, 0.056355, -0.300751, 0.120627, 0.348356, 0.178229, -0.271188, 0.140977, -0.042571, 0.198849, -0.010497, 0.164892, -0.021881, -0.230191, -0.211945, 0.086287, 0.520863, -0.106241, 0.190145, 0.076757, -0.060238, 0.128493, 0.067946, -0.154441, -0.098756, -0.099413, -0.08541, -0.103559, -0.045043, 0.019576, 0.014792, 0.212813, -0.190096, 0.118938, 0.188957], [-0.133395, 0.333227, -0.142773, -0.175365, 0.267442, 0.47306, 0.197375, 0.594507, -0.304927, 0.10003, -0.335891, 0.219707, 0.173693, -0.13615, -0.26721, 0.410579, -0.285262, 0.106748, -0.280714, 0.165273, -0.499347, -0.069756, -0.584022, 0.344966, -0.074853, 0.012917, 0.233901, -0.339199, 0.220398, 0.100825, -0.082701, 0.301527, 0.091325, 0.196625, 0.083313, -0.168427, -0.038553, -0.02233, 0.041335, -0.427371, 0.0134, -0.122955, 0.188466, 0.018168, -0.060598, -0.047959, 0.230439, -0.105634], [0.059592, 0.081357, 0.072735, 0.008144, -0.141259, -0.16786, -0.031309, -0.004392, 0.069575, -0.079681, -0.261614, 0.080037, -0.029491, 0.140233, -0.133718, -0.089141, -0.251575, -0.67986, -0.389732, -0.126622, 0.355716, 0.316947, -0.39815, 0.103324, -0.375597, -0.090752, -0.402526, -0.007809, 0.022757, -0.213504, -0.161528, -0.533411, 0.060245, 0.132406, 0.327521, 0.018137, -0.007169, 0.041427, 0.012994, -0.15696, 0.095944, -0.00482, 0.11616, -0.007645, 0.004863, 0.20195, -0.142601, -0.01748], [-0.098847, 0.10678, -0.272739, -0.317044, -0.212534, -0.079103, 0.153022, -0.246806, 0.342318, 0.044445, -0.173691, -0.519484, 0.054165, 0.066094, 0.00815, -0.172803, 0.17591, 0.383608, -0.150461, -0.097678, -0.271461, 0.215012, -0.221201, -0.332687, -0.201771, -0.276977, -0.3295, 0.115017, -0.143357, -0.269351, -0.087788, 0.208339, -0.072912, -0.093167, 0.070529, -0.104165, -0.304428, 0.048559, -0.046728, -0.073403, 0.102011, -0.075361, -0.065913, 0.130478, 0.244398, 0.123329, 0.021299, 0.120885], [0.023732, 0.036401, 0.116701, 0.032638, -0.133175, -0.036381, 0.096534, 0.300745, 0.091197, 0.113786, -0.162143, 0.095276, -0.151084, -0.164487, 0.257288, 0.056142, -0.251666, -0.544558, -0.48125, -0.179908, 0.420663, -0.058845, -0.205714, 0.147753, 0.054538, 0.058403, 0.086911, 0.141737, -0.009492, -0.086649, 0.130208, -0.183422, 0.031237, 0.190964, 0.24582, -0.138173, 0.120738, -0.040916, -0.08575, -0.048421, 0.049852, 0.122805, -0.147887, -0.019726, 0.039633, -0.081348, 0.022273, 0.088315]]
SEQ_BI = [0.043717, 0.095844, -0.190564, 0.178191, -0.015243, -0.042721, 0.148482, -0.009885, 0.06159, 0.155928, 0.191479, -0.042982, -0.05835, -0.013821, 0.297563, 0.100608, 0.51885, 0.577467, 0.694201, 0.619818, 0.365445, 0.262166, 0.581244, 0.380357, 0.536134, 0.765383, 0.770967, 0.567199, 0.574948, 0.335591, 0.616829, 0.340402, 0.034908, 0.146871, -0.037021, 0.26549, 0.111044, 0.307196, -0.072654, -0.060311, 0.059353, 0.018916, 0.237076, 0.001554, 0.23037, -0.152795, 0.095833, -0.264387]
SEQ_WH = [[0.185203, -0.039865, 0.368894, -0.051423, 0.003325, 0.153879, 0.080748, 0.026468, -0.049912, 0.103086, -0.165285, 0.096717, -0.064096, 0.063865, -0.543287, 0.090954, 0.107799, -0.416741, 0.53312, 0.512945, 0.000842, -0.387646, 0.785436, -0.049102, 0.32874, 0.169858, 0.034971, -0.264978, 0.410645, 0.059197, -0.135228, 0.388423, 0.377452, -0.09874, -0.158481, 0.250777, 0.083068, 0.30542, 0.352065, -0.192317, 0.392744, -0.225038, -0.309983, -0.043867, 0.00469, -0.340056, 0.108467, 0.117788], [0.186237, 0.053429, 0.000108, -0.157502, -0.018302, -0.06767, 0.14722, 0.000122, -0.113026, 0.05091, -0.186708, 0.23282, 0.559868, -0.288292, 0.263661, 0.375529, -0.162172, -0.045516, 0.032051, 0.014103, 0.319026, -0.098332, 0.264497, -0.131356, 0.109779, 0.043231, 0.407756, 0.111673, 0.487945, 0.466802, 0.705232, -0.057533, 0.065595, -0.010507, -0.217714, 0.176693, 0.269971, -0.105856, -0.184901, -0.118709, -0.166012, -0.110351, 0.082675, 0.284611, -0.232421, -0.047078, 0.157258, -0.14417], [-0.046022, 0.703642, 0.375707, 0.485421, 0.452616, 0.385374, 0.204341, 0.587759, 0.296667, 0.254602, 0.069019, 0.806515, -0.077377, 0.315173, -0.21999, 0.314014, 0.066207, -0.385451, 0.543105, -0.704236, 0.181319, -0.654181, 0.290342, -0.121446, 0.381441, 0.437658, -0.370511, -0.223328, -0.073081, 0.305794, 0.523339, 0.221829, -0.067153, -0.086385, 0.376847, 0.084281, 0.033898, 0.22479, 0.284998, 0.208524, 0.221257, -0.157314, 0.115826, 0.073172, 0.114522, -0.186913, -0.092525, -0.016432], [0.355889, 0.01888, 0.160055, 0.245022, -0.54021, -0.033688, -0.011172, -0.248334, 0.170435, -0.348791, 0.117975, 0.329658, -0.107708, 0.014929, -0.190581, -0.160514, 0.197274, -0.054933, 0.415763, -0.064198, 0.306439, -0.300006, 0.436991, 0.813956, 0.786799, 0.825771, 0.42564, 0.488727, -0.328557, 0.452079, 0.413033, -0.138441, 0.225946, -0.021465, -0.155051, 0.364393, 0.56703, -0.022682, -0.189944, -0.082486, 0.089789, -0.609469, -0.353612, 0.06363, 0.022968, -0.132607, -0.018688, 0.079147], [-0.090857, 0.005666, -0.030178, -0.003486, -0.024384, 0.335144, -0.330657, -0.08099, -0.17448, -0.472136, 0.145752, 0.142435, 0.153741, 0.052497, -0.086424, 0.084779, -0.036539, -0.204562, -0.221555, -0.284656, -0.371445, 0.155062, 0.420982, 0.06498, 0.722494, 0.310323, 0.043947, 0.120277, -0.413829, 0.209596, 0.180155, 0.067455, -0.218751, 0.35426, -0.243573, 0.202916, 0.411445, -0.280248, -0.073945, -0.086893, -0.121822, -0.286194, -0.126309, 0.135477, -0.199989, -0.090666, 0.311663, 0.013164], [0.190795, 0.098966, 0.32693, 0.056893, 0.193404, 0.070814, 0.319656, -0.186517, 0.107435, 0.510992, 0.166944, 0.005301, -0.12508, -0.101104, -0.313968, 0.323225, 0.753879, 0.443332, 0.45785, 0.630975, 0.376045, 0.318798, 0.434329, -0.19793, 0.079052, 0.253551, 0.865245, 0.128211, 0.379101, -0.1174, 0.793393, 0.21278, 0.276859, -0.006464, 0.054343, 0.150834, -0.214888, 0.307214, -0.186358, -0.054548, 0.110673, 0.102054, 0.082569, 0.004461, 0.005703, -0.15009, 0.044298, -0.233999], [0.202207, -0.017603, 0.393288, 0.391507, -0.090429, 0.171084, -0.120119, 0.122785, -0.121935, 0.428726, -0.049214, 0.16026, -0.660325, 0.066864, -0.405633, -0.086646, -0.133711, -0.471242, 0.333765, 0.259915, 0.146369, -0.696999, -0.109877, -0.576319, -0.287155, -0.508845, 0.182909, -0.938341, 0.167697, -0.117747, -0.668266, -0.062134, 0.361104, -0.254827, 0.236275, 0.320895, -0.117318, 0.187855, 0.401976, 0.199408, 0.148597, 0.021861, -0.116539, -0.250901, -0.119222, -0.023938, 0.258156, 0.189168], [0.126208, 0.156742, -0.10853, 0.22919, -0.039234, 0.103648, 0.0649, 0.127362, 0.211526, 0.059378, 0.29707, 0.297765, -0.137374, 0.24797, 0.243233, -0.205126, 0.15497, 0.31015, 0.097752, 0.165035, 0.009202, 0.350817, -0.199542, 0.244035, -0.157963, 0.313941, 0.072222, 0.398529, -0.546123, -0.140305, 0.219406, 0.392222, -0.043844, -0.183619, 0.234796, 0.090095, -0.013821, 0.454468, -0.086628, 0.431827, 0.20406, 0.283076, 0.015377, -0.316148, 0.110728, -0.187698, 0.15962, 0.22633], [0.069567, -0.04554, 0.738913, 0.382345, -0.162707, 0.373861, 0.0051, 0.020629, -0.084691, 0.05756, 0.00698, 0.002878, -0.459639, 0.187907, -0.340067, -0.041427, -0.347209, -0.309927, 0.333345, 0.216029, -0.004246, -0.306317, -0.20738, -0.117943, -0.293822, 0.068479, -0.057236, -0.49902, 0.182628, -0.456255, -0.385584, -0.41233, 0.197336, -0.278384, 0.299747, -0.050545, -0.221214, 0.373463, 0.393082, 0.148543, 0.328025, 0.121747, -0.085256, -0.436793, 0.126097, -0.271973, 0.340912, 0.236746], [-0.239535, -0.399732, -0.056543, 0.102965, 0.113261, -0.125747, -0.114222, 0.348467, -0.268067, 0.070339, 0.275003, -0.221314, -0.231986, -0.301782, 0.028209, 0.00889, 0.465063, -0.090534, 0.075684, -0.08591, 0.007329, 0.015976, 0.092231, 0.074351, -0.228274, 0.11678, 0.324478, 0.507547, -0.361802, -0.244355, 0.39961, 0.180533, -0.339975, 0.010842, 0.212149, -0.405063, -0.269161, 0.07819, -0.274307, 0.323375, 0.158645, 0.503665, 0.459066, -0.091835, -0.305071, 0.004824, 0.145094, -0.206798], [-0.046934, -0.141639, -0.015316, -0.065448, 0.545576, -0.00401, 0.345876, 0.254612, 0.2631, 0.259234, 0.372782, 0.132433, 0.144015, -0.022568, 0.081012, 0.236818, 0.935611, 0.505616, 0.459441, 0.255526, 0.248898, 0.579007, 0.662744, 0.187274, 0.359695, 0.668434, 0.44817, 0.787551, 0.532688, 0.226871, 0.625874, 0.361031, -0.141444, 0.211079, -0.068866, 0.022483, 0.017298, 0.075359, -0.036775, 0.211153, -0.168255, 0.331639, 0.214535, 0.007375, -0.237614, 0.134083, -0.036938, -0.342389], [-0.151054, 0.041215, -0.295579, -0.546647, -0.296893, -0.412726, 0.241987, -0.18067, -0.042712, -0.632113, -0.217937, -0.200292, 0.387411, -0.055886, -0.108711, -0.320138, -0.209059, -0.295862, -0.372663, -0.444655, -0.098922, 0.176201, -0.081907, -0.069893, -0.17014, 0.077461, -0.779189, -0.155364, 0.148396, 0.333853, -0.167783, -0.007297, 0.01617, -0.160838, -0.197693, 0.236233, 0.224434, -0.395437, -0.207941, -0.139826, -0.141471, -0.173277, 0.003866, 0.344313, -0.035803, 0.023522, -0.207784, 0.264901], [0.088788, 0.06881, -0.296965, -0.017627, -0.224942, -0.351783, 0.448594, -0.146076, 0.301326, 0.210473, 0.168363, 0.059889, -0.069343, 0.39861, 0.302269, -0.229369, 0.640806, 0.050149, 0.239179, 0.229132, 0.396838, 0.370131, -0.522854, 0.150869, -0.432166, 0.05332, 0.605971, 0.358376, 0.60536, 0.200934, 0.012102, -0.026572, 0.13281, -0.105984, -0.131605, 0.240381, -0.546517, 0.356872, -0.115578, -0.06068, -0.189573, 0.421737, 0.04081, -0.186574, 0.235319, -0.236211, -0.275741, 0.116294], [-0.137505, 0.061506, -0.315444, -0.406217, -0.050682, 0.045748, -0.256232, 0.083621, 0.125981, -0.034453, -0.233045, 0.059244, 0.306349, -0.187467, 0.141762, -0.120137, -0.164656, -0.050908, -0.585877, -0.401971, 0.040154, 0.312078, -0.522849, -0.118127, -0.572243, -0.57116, -0.425959, 0.098222, -0.38535, 0.200645, -0.410133, 0.056561, -0.242097, 0.144049, 0.146308, -0.218144, -0.072606, -0.375775, -0.150222, -0.028268, -0.125868, 0.004603, 0.20052, 0.200162, 0.040745, 0.294844, 0.085158, -0.048826], [-0.381971, -0.117985, 0.133032, 0.134638, 0.382539, 0.586142, -0.332008, 0.387007, -0.536319, 0.23146, -0.150312, -0.063804, -0.022247, -0.374485, -0.543365, 0.136631, -0.324103, 0.038288, -0.110409, 0.45949, -0.456035, -0.61475, -0.026116, -0.175718, -0.422846, -0.146288, 0.304111, -0.224202, -0.129557, -0.820456, -0.335384, 0.001184, -0.008018, -0.136815, 0.011087, -0.023949, 0.053638, 0.050987, 0.47108, -0.000274, 0.402213, 0.062119, 0.000136, -0.322198, -0.568002, -0.05596, 0.730043, -0.087874], [-0.200635, 0.049341, 0.22596, -0.137985, -0.234855, -0.117142, 0.070655, 0.05864, 0.143766, -0.190711, -0.34049, 0.030949, -0.031073, 0.032879, 0.090483, -0.428953, -0.555019, -0.195357, -0.326494, -0.207958, -0.329151, -0.511159, -0.432517, -0.484539, -0.491874, -0.667492, -0.692973, -0.536952, -0.498459, -0.202002, -0.490541, 0.053052, -0.033141, -0.021604, 0.12637, -0.039945, 0.128686, -0.009328, 0.212263, -0.130448, -0.057189, 0.209947, -0.244135, -0.212696, 0.103114, 0.096325, 0.097199, 0.107993]]
SEQ_BH = [0.237409, -0.043742, 0.08805, -0.077712, -0.033592, 0.140632, 0.108754, 0.052592, -0.020799, 0.184222, 0.277619, 0.188461, 0.18309, 0.088225, 0.060128, 0.187383, 0.776402, 0.283991, 0.561597, 0.303973, 0.583236, 0.421555, 0.825268, 0.633835, 0.373069, 0.598055, 0.628285, 0.72975, 0.455046, 0.325947, 0.498994, 0.275445, 0.184381, 0.014692, 0.099162, -0.044338, 0.025463, -0.023374, -0.279813, 0.045009, -0.150538, 0.149506, 0.203731, -0.172639, 0.016241, -0.088306, -0.115054, -0.273958]
SEQ_HEAD = [
    ([[-0.005447, 0.044048, -0.195417, -0.080351, 0.02355, -0.150678, 0.004532, 0.059081, -0.044093, -0.001767, 0.054182, -0.084463, -0.022318, 0.018983, 0.030449, 0.056326, 0.080324, 0.060073, 0.012171, -0.022336, -0.091714, -0.025909, 0.00312, 0.07884, -0.066889, 0.015257, -0.040629, -0.182955, -0.088372, -0.127731, 0.14297, -0.016432, -0.043989, 0.078674, -0.060935, -0.140232, 0.062601, 0.124077, -0.112812, -0.039063, 0.080894, -0.118419, 0.056292, 0.148127, -0.10532, 0.090682, 0.000568, 0.065401, 0.068834, -0.088523, 0.013317, 0.048184, -0.046382, -0.156783, -0.123159, -0.067309, 0.010782, -0.146849, -0.026966, -0.038723, 0.215488, -0.058354, -0.086811, 0.002322], [0.074575, 0.07739, -0.057451, 0.12666, 0.118737, -0.045616, -0.059734, -0.127118, 0.10578, -0.049319, -0.127373, 0.038833, -0.009381, -0.103261, -0.056554, 0.059295, 0.036004, 0.012472, -0.060464, -0.006196, -0.086327, -0.008063, -0.01821, -0.004085, 0.089262, -0.008008, -0.049509, -0.091458, 0.038306, 0.054767, -0.108692, 0.053476, 0.114522, 0.020745, 0.018104, 0.03728, 0.072195, -0.010156, -0.051735, 0.037976, 0.13671, 0.006872, -0.10592, 0.069175, -0.086005, 0.095963, 0.088829, 0.116462, -0.008541, -0.052671, -0.07496, -0.142144, -0.090569, 0.06288, 0.091751, -0.043675, -0.127712, -0.003425, 0.099864, -0.002774, -0.090102, -0.028877, -0.132624, -0.055576], [0.054706, 0.104121, -0.014047, 0.148795, 0.100069, -0.061955, -0.02514, -0.160778, 0.010675, -0.029174, -0.107198, -0.080354, -0.021907, -0.09086, -0.091007, -0.118937, -0.047253, -0.024886, 0.040512, 0.134516, 0.060386, 0.039798, 0.062909, -0.004789, 0.094558, -0.074314, -0.053345, 0.070483, 0.018972, 0.102062, 0.069823, 0.038141, -0.02434, -0.132643, -0.040936, 0.08768, 0.088324, -0.026425, 0.04429, 0.167838, -0.111501, 0.057646, 0.053715, 0.008075, 0.037731, -0.035191, 0.028802, 0.107895, 0.041648, 0.04751, 0.049693, -0.11455, -0.143691, 0.052, 0.034969, -0.052623, 0.050028, -0.112842, 0.132746, -0.059528, 0.110588, -0.021307, -0.068897, 0.044539], [0.070305, -0.064044, -0.020526, 0.015129, -0.063295, -0.007114, 0.075748, -0.052684, 0.078755, 0.127503, -0.111182, 0.057617, -0.14332, -0.144034, 0.128901, -0.082267, -0.019647, -0.118348, -0.002512, 0.156791, 0.050481, -0.116819, -0.038257, -0.042714, 0.073747, 0.114484, -0.113476, -0.031011, 0.078148, -0.119203, 0.05103, -0.077202, 0.058256, -0.031891, 0.033268, -0.007023, 0.152893, 0.017549, 0.061203, -0.010324, -0.221345, -0.018152, 0.015844, -0.24738, -0.022152, 0.065298, -0.092466, -0.089898, 0.079372, -0.145117, 0.165584, -0.084367, -0.121093, 0.143136, 0.078685, 0.086602, 0.06253, 0.103459, 0.13991, -0.086478, -0.041678, 0.180032, -0.073292, 0.070996], [-0.014517, 0.051085, 0.154224, 0.189279, -0.147302, 0.016546, 0.082197, -0.180441, 0.116006, -0.188877, -0.108728, -0.10208, 0.016528, 0.193029, -0.024761, -0.174169, -0.086646, 0.189084, 0.048668, 0.188243, 0.031121, -0.033762, -0.02751, -0.072028, 0.019866, -0.113716, -0.113824, 0.061683, 0.002454, -0.352555, 0.018758, 0.16286, 0.164383, 0.085227, 0.082083, 0.066008, -0.010505, 0.135896, -0.003275, -0.0141, -0.279939, -0.06387, -0.070344, -0.286225, -0.103325, -0.016594, 0.090177, -0.039816, -0.153557, -0.042729, 0.086614, 0.034652, 0.135776, 0.014488, -0.100911, -0.048391, 0.211741, 0.242802, 0.201617, -0.127436, 0.109111, -0.025043, 0.02915, 0.158577], [0.051267, -0.03627, -0.0527, 0.116777, 0.016096, 0.130408, -0.027128, 0.077971, -0.003107, 0.004234, -0.011418, 0.062221, -0.055727, 0.114512, -0.108631, 0.111624, 0.081608, -0.113868, 0.050064, -0.095767, -0.004553, -0.036408, 0.146175, 0.060146, -0.056101, 0.076708, 0.159655, 0.093332, -0.064993, 0.021351, -0.00878, 0.080582, 0.160478, 0.052182, 0.000942, 0.092281, 0.117723, -0.071776, 0.066506, 0.022305, -0.022452, 0.033662, -0.117551, -0.049859, -0.093148, 0.030595, 0.051451, 0.046793, 0.158986, -0.063413, 0.080827, -0.108749, -0.048245, -0.040415, -0.04029, -0.026544, -0.143063, -0.00637, -0.127786, 0.142011, 0.003041, 0.159861, -0.00651, -0.045716], [0.040498, 0.046197, -0.117853, 0.08937, -0.196278, -0.172673, 0.137314, 0.057785, 0.217836, -0.043507, -0.017519, -0.023419, -0.152084, 0.322287, -0.178439, -0.107988, 0.005729, -0.000599, 0.021489, 0.168832, 0.009557, -0.092873, 0.008218, -0.099302, 0.027619, 0.082557, 0.003825, -0.169823, 0.196385, -0.068924, 0.110109, 0.06507, 0.062659, 0.08312, -0.179674, 0.066517, -0.073589, 0.121704, 0.13931, 0.096478, 0.043469, 0.046091, -0.018287, -0.184566, 0.126385, -0.251575, 0.096772, 0.111209, -0.209224, -0.080093, -0.041393, -0.019979, -0.056068, 0.081623, -0.184893, -0.066299, 0.005136, 0.017248, 0.085918, -0.089169, 0.030511, -0.005456, -0.030558, 0.083169], [-0.011389, -0.10517, -0.060864, 0.056861, 0.047506, 0.027323, -0.137058, -0.098883, -0.004747, 0.018434, 0.072061, 0.045728, -0.051474, 0.05579, -0.083218, -0.04447, 0.131609, -0.047186, 0.157896, 0.133417, 0.069087, -0.002955, -0.089285, -0.133688, 0.049156, 0.147927, 0.088571, 0.071639, 0.028771, 0.040861, -0.099841, -0.153714, 0.020315, 0.046012, -0.054979, -0.090197, -0.024606, -0.080077, -0.066194, -0.069629, 0.016977, -0.088533, 0.038281, 0.069552, -0.025988, 0.072994, 0.025033, 0.04903, -0.010212, 0.078354, -0.022126, -0.010966, -0.048544, 0.160666, 0.127688, 0.001267, -0.048102, 0.138128, 0.036088, 0.080223, 0.050191, 9.7e-05, -0.133814, -0.019464], [0.114648, 0.117059, 0.052168, 0.30869, -0.019259, -0.161553, 0.061861, -0.131997, 0.174211, -0.083202, 0.063359, -0.229807, -0.125422, 0.02791, -0.079648, 0.04548, -0.06138, 0.185869, -0.078535, -0.020274, -0.060408, 0.114368, -0.082799, -0.178817, 0.011326, 0.0653, -0.085755, -0.092886, 0.04917, 0.162266, 0.006139, 0.038081, -0.178542, 0.032, -0.06547, 0.001541, -0.062286, 0.034986, 0.105152, 0.005961, 0.065651, 0.195934, -0.015073, 0.187289, -0.104364, -0.090829, 0.060694, 0.216069, 0.00373, 0.019657, -0.054001, -0.15589, 0.087573, -0.121825, -0.104657, -0.065494, -0.075575, -0.100168, 0.201193, 0.107351, 0.040535, -0.118006, -0.051611, 0.212998], [-0.118726, 0.052452, -0.072386, -0.041789, 0.045437, -0.013123, 0.080621, 0.226382, 0.064155, 0.011566, 0.086326, -0.027079, 0.087765, 0.072381, 0.090944, 0.306554, 0.072502, 0.066919, -0.101497, 0.240904, -0.198194, 0.112457, 0.015184, 0.106048, -0.080439, -0.132424, -0.058427, 0.074606, 0.03576, -0.070853, 0.039477, -0.005867, -0.212443, -0.191631, -0.012572, 0.010753, -0.116502, 0.212974, -0.028624, 0.07933, -0.185611, 0.110996, -0.110792, -0.090637, 0.068744, 0.083689, 0.163346, -0.170349, -0.091905, -0.120783, -0.224071, -0.009943, 0.007878, -0.016164, 0.221705, 0.034643, 0.062566, 0.033738, 0.163233, 0.001829, -0.089579, -0.056465, -0.244113, 0.001218], [0.015606, 0.019057, 0.011203, -0.058964, -0.077143, 0.101775, -0.049574, -0.00458, -0.039854, 0.024887, 0.176494, -0.094506, -0.030981, 0.077431, 0.004111, -0.010747, 0.010677, -0.130557, 0.030688, -0.091624, 0.055813, -0.123819, -0.060828, 0.137137, 0.114964, 0.046416, -0.00593, -0.01979, -0.002064, 0.022691, -0.024207, 0.188825, 0.060758, -0.129072, -0.004534, -0.059424, -0.043372, -0.032806, 0.033453, -0.017085, 0.008193, 0.065707, 0.072451, -0.032461, 0.087838, -0.052697, -0.086, 0.132335, -0.047549, 0.038535, -0.059468, 0.108261, -0.082817, 0.095464, 0.209739, 0.11359, -0.007963, -0.022357, 0.081174, -0.005951, -0.078249, -0.067153, -0.149494, -0.099018], [0.112704, 0.106721, 0.114657, 0.062702, 0.034594, -0.166678, 0.070223, -0.26945, 0.213649, -0.066763, -0.087595, -0.171622, -0.163404, -0.086242, 0.085923, -0.198249, 0.024443, 0.317244, 0.130758, 0.034854, -0.190415, 0.043359, 0.026629, 0.029647, -0.068272, -0.275411, -0.148832, -0.085136, 0.038715, 0.030736, -0.058834, 0.021366, -0.182698, -0.220256, -0.168862, -0.076069, -0.132918, -0.384639, 0.160748, -0.03297, -0.115947, 0.240979, 0.113614, 0.124167, -0.144201, 0.137648, 0.183705, 0.327199, -0.044664, -0.001036, -0.019182, -0.260042, 0.034062, -0.072283, -0.05567, -0.007686, -0.100146, 0.087049, 0.122807, 0.130754, 0.24534, -0.125959, -0.076765, -0.062571], [-0.058266, 0.08674, 0.031949, 0.053012, 0.055527, -0.077283, 0.084944, -0.110617, -0.067749, -0.001728, 0.12477, 0.003719, -0.032545, -0.103346, -0.121831, 0.112566, 0.033465, 0.123671, -0.089574, 0.012634, -0.169742, 0.030319, 0.078521, -0.051955, 0.02149, 0.014291, 0.017476, -0.063882, -0.061699, 0.052718, -0.109029, -0.080694, 0.07289, -0.095861, -0.159201, -0.04167, 0.060192, 0.125794, -0.053128, 0.000374, 0.017453, 0.019739, 0.335436, 0.020248, -0.030444, -0.120198, -0.072637, 0.109536, 0.021749, -0.017514, -0.103794, -0.028324, -0.04053, 0.027252, 0.088139, 0.023671, -0.004873, -0.122442, 0.01032, -0.014248, -0.000183, -0.146693, -0.181364, -0.115072], [-0.129282, 0.030065, 0.008175, 0.003296, 0.002478, 0.020814, -0.049678, -0.07543, 0.05423, -0.114771, 0.093453, -0.044412, 0.035417, 0.106918, -0.123921, -0.02741, -0.03319, 0.276476, -0.183099, 0.00719, -0.10756, 0.004293, 0.021761, -0.045414, 0.082423, 0.012248, 0.046685, -0.060597, -0.060112, 0.032678, -0.019996, 0.00489, 0.13771, 0.073786, 0.0834, 0.030596, -0.003706, 0.279677, -0.050753, -0.111674, -0.060281, -0.171035, -0.102499, 0.016398, -0.079667, -0.021563, -0.054438, -0.071334, -0.044031, 0.076495, 0.131955, 0.049677, -0.078633, 0.062463, -0.136873, 0.153424, 0.0389, -0.068162, -0.039834, -0.177868, -0.06138, 0.082432, 0.09099, 0.072264], [0.034723, 0.011495, 0.140962, 0.039423, 0.062849, -0.065526, -0.091373, -0.060198, -0.108042, -0.08021, -0.055522, 0.022886, 0.123449, -0.135583, 0.179559, -0.188154, 0.082844, -0.066378, -0.022329, 0.069087, -0.129168, -0.077296, 0.141604, 0.111403, -0.001541, -0.087263, -0.057329, 0.008558, -0.242393, 0.056075, -0.182179, 0.038647, -0.204384, -0.046131, -0.081164, -0.060165, 0.103725, 0.099026, -0.117543, 0.111489, 0.024672, 0.071289, -0.146607, 0.018606, -0.153396, -0.074165, -0.114086, 0.062127, -0.053808, 0.046748, -0.056382, -0.07634, 0.100593, 0.119957, 0.139643, -0.00824, 0.044262, -0.210517, -0.030778, 0.049252, -0.060265, 0.17323, 0.048969, -0.06471], [-0.070882, 0.048576, 0.051956, 0.038468, -0.0064, -0.22071, 0.147693, 0.072288, 0.017072, 0.028454, -0.04078, -0.065719, 0.072681, 0.057256, 0.007051, 0.052467, -0.061161, 0.163926, -0.068477, -0.122682, -0.033995, -0.028097, -0.037813, 0.07653, -0.07827, -0.011516, -0.03505, 0.037633, -0.092917, -0.081773, 0.022381, 0.073603, -0.003744, 0.036749, -0.058892, -0.048453, -0.045646, -0.16262, -0.057964, -0.048208, 0.067587, 0.01337, 0.085163, -0.024439, 0.031291, -0.026018, -0.020744, 0.054244, 0.149703, -0.073377, 0.056912, -9.3e-05, -0.025468, -0.180469, 0.065535, -0.006124, 0.035277, -0.198517, 0.076486, -0.07754, 0.076231, 0.082345, -0.021285, 0.016722], [0.131552, 0.115834, 0.180928, -0.082288, 0.051804, -0.238016, 0.12445, 0.231405, 0.063147, -0.226977, 0.005816, -0.010365, -0.022516, -0.039889, -0.141876, -0.235585, 0.085923, 0.087464, -0.022551, 0.123456, 0.096473, 0.199842, -0.254595, 0.086263, -0.104937, -0.230521, -0.077744, -0.044534, -0.043246, -0.296724, 0.023738, 0.181905, 0.075854, 0.144339, 0.083234, -0.148173, -0.080024, -0.054203, -0.322663, -0.07317, -0.070014, -0.207066, -0.041822, 0.155116, -0.228035, -0.058572, 0.076643, -0.13761, -0.003484, 0.129884, 0.05147, -0.101452, -0.088229, -0.207298, -0.054332, 0.08039, 0.227506, -0.081041, 0.187513, 0.017911, 0.1715, 0.065436, 0.097559, 0.221312], [0.204331, -0.099669, -0.021821, 0.001906, -0.130462, -0.386161, -0.119068, 0.135769, -0.116707, -0.009846, 0.070107, -0.031209, 0.129824, -0.088635, 0.090381, 0.265674, -0.116623, 0.103611, -0.102079, 0.014097, 0.364858, -0.130032, -0.064625, 0.086405, 0.044795, 0.411195, -0.009467, 0.06916, -0.07766, 0.009885, -0.136201, -0.12025, 0.054735, -0.11641, 0.070896, -0.20905, -0.070386, -0.100516, -0.185184, 0.079674, -0.137908, -0.014404, -0.038236, -0.354164, 0.121534, 0.073008, -0.115306, 0.102717, -0.042353, 0.205667, -0.294715, -0.051268, -0.262202, -0.061088, 0.085566, -0.128755, 0.026242, 0.023685, -0.138616, 0.058257, 0.226062, 0.053847, -0.194084, -0.106128], [-0.001784, -0.04336, 0.043473, -0.005351, -0.041561, -0.040229, 0.094875, 0.021091, 0.022307, -0.101231, -0.176994, 6.1e-05, -0.073049, 0.149979, 0.013752, -0.007313, 0.014073, 0.026925, 0.059289, 0.060693, 0.025688, 0.086527, -0.006449, -0.065381, -0.017993, 0.029106, 0.104121, -0.04777, 0.094967, -0.077463, 0.146929, 0.074566, 0.054733, 0.085365, -0.063188, 0.053065, -0.150214, 0.131365, 0.021597, 0.01264, -0.057148, -0.072345, -0.142689, 0.059749, -0.096859, -0.053863, -0.05233, -0.093519, -0.123749, 0.002966, -0.100789, 0.085111, -0.054313, -0.017479, 0.032165, -0.042572, -0.009065, 0.023841, 0.092444, -0.086504, -0.062007, -0.03721, 0.141546, 0.08739], [0.16475, 0.123166, -0.050643, 0.049625, 0.009824, -0.098954, 0.013051, 0.047422, 0.124984, 0.000777, -0.169879, -0.136398, -0.225972, 0.260848, -0.365646, 0.009144, -0.091922, 0.001812, 0.015749, -0.081936, 0.057987, -0.081608, -0.006615, -0.338108, 0.201316, -0.01049, 0.283833, -0.000406, 0.537597, -0.141295, 0.092189, -0.032687, 0.103814, -0.118121, -0.231185, -0.124698, 0.024502, 0.035404, -0.054135, -0.207879, 0.032076, -0.038402, 0.073337, -0.135338, 0.098812, 0.030346, 0.210397, 0.037173, -0.069947, 0.206599, -0.025506, 0.105014, -0.057335, -0.092629, 0.062603, -0.005604, -0.367898, 0.032293, 0.171543, -0.178366, 0.127542, -0.179708, -0.077429, 0.0834], [0.071547, -0.130738, 0.143576, 0.026387, -0.110932, -0.13926, 0.017684, -0.014689, 0.012231, 0.160611, -0.166317, 0.011362, 0.063868, 0.052939, -0.084102, 0.044371, 0.138119, 0.024465, -0.147351, -0.03852, -0.167705, 0.043111, 0.013432, -0.031012, 0.037668, -0.050468, 0.153578, 0.180857, 0.026869, 0.199535, -0.209874, 0.006761, -0.016556, 0.039141, 0.014159, 0.143136, -0.116377, 0.029074, 0.126739, -0.027807, 0.002019, 0.045514, -0.082394, 0.029176, -0.146015, 0.145825, 0.033221, 0.034876, 0.053288, -0.052395, -0.05925, -0.029858, -0.018941, -0.04977, 0.131086, -0.016412, -0.30659, 0.089361, -0.036549, 0.073789, -0.019061, -0.153359, -0.044283, -0.019194], [0.008048, -0.084771, 0.029653, 0.03537, 0.138758, 0.044649, -0.174072, -0.032356, -0.149541, -0.034474, -0.082556, 0.152875, 0.069602, -0.016423, -0.093928, -0.070494, -0.025793, -0.151233, 0.011859, -0.024424, -0.035767, -0.179842, -0.325497, 0.007616, -0.103269, -0.084386, 0.171147, 0.071608, 0.082255, 0.039759, -0.027653, -0.073856, 0.135658, -0.144256, 0.023968, -0.147863, 0.009916, 0.000377, 0.308188, -0.014896, -0.07522, 0.151402, -0.01648, -0.011699, 0.046872, 0.235012, -0.053122, -0.04299, 0.071737, -0.103425, 0.02254, 0.017433, 0.085276, -0.182362, 0.014742, -0.342586, 0.094235, -0.151196, 0.087349, 0.029477, 0.022135, 0.077316, -0.100357, -0.03261], [-0.002202, 0.138502, 0.044401, -0.029189, 0.220187, -0.101968, -0.052874, 0.224189, 0.090477, 0.145605, 0.16046, 0.204804, 0.096454, 0.040524, 0.046498, -0.120585, 0.117756, 0.084747, -0.126172, -0.036433, 0.069223, -0.148056, 0.199268, 0.12958, -0.081698, -0.006626, -0.282877, 0.030354, 0.105565, -0.011213, -0.020999, 0.128386, 0.32909, 0.270019, 0.139215, 0.041838, 0.108623, 0.0085, -0.139481, 0.040284, -0.201664, 0.0134, 0.001705, 0.018776, -0.02334, -0.016401, -0.014059, 0.046503, 0.092393, -0.054649, -0.172859, -0.185429, -0.105666, 0.131407, 0.010732, -0.171322, -0.025795, -0.031952, 0.155213, 0.11279, 0.020862, 0.21953, 0.120316, 0.06439], [0.059148, 0.08031, -0.084514, -0.144597, -0.080233, -0.10708, -0.151119, -0.037762, -0.073506, 0.083297, -0.079268, -0.057147, 0.029753, 0.034893, -0.039737, 0.028156, -0.16516, 0.070687, -0.132946, 0.12178, -0.079725, -0.214145, 0.09227, -0.141142, 0.029696, -0.26406, 0.035448, 0.008575, -0.052634, 0.026097, -0.135343, -0.025355, -0.151662, -0.227592, -0.165511, 0.04022, -0.04261, -0.095031, -0.122073, 0.014904, 0.081308, -0.118944, -0.06973, 0.001235, -0.182731, 0.002053, 0.065355, -0.096081, -0.077797, 0.007412, 0.091829, -0.100863, -0.029605, -0.175061, 0.023291, -0.12687, 0.014687, -0.080627, -0.061875, 0.015628, -0.140887, -0.127482, -0.00531, -0.097679], [-0.07984, -0.158157, 0.168114, 0.075483, 0.169521, 0.248416, -0.037432, 0.063229, -0.102444, -0.03085, -0.055966, -0.174351, 0.11553, -0.082019, 0.040862, 0.05602, 0.102158, 0.081509, -0.049399, -0.052302, 0.039148, -0.176146, 0.181258, 0.04727, -0.250364, 0.014669, -0.135808, 0.181115, 0.124105, 0.063286, -0.227851, -0.054238, 0.043192, -0.077813, -0.024871, 0.064909, -0.08285, 0.301959, 0.107841, 0.069846, 0.188766, 0.056108, 0.104731, -0.104401, 0.143673, -0.227038, 0.0083, 0.03599, -0.190867, -0.018501, 0.141608, 0.176482, -0.09003, 0.220693, 0.028715, 0.085863, -0.27721, -0.040369, -0.125505, -0.022068, 0.007085, -0.107568, 0.045871, -0.131756], [0.141696, -0.068265, 0.04398, -0.089118, -0.046205, -0.070226, -0.215073, -0.019251, -0.025979, 0.144837, 0.107851, 0.180916, 0.077677, 0.025187, 0.001645, 0.017929, 0.036674, -0.123143, 0.010444, -0.064419, 0.156535, -0.151731, 0.059116, -0.094777, -0.086125, -0.021012, -0.069223, -0.035517, 0.041372, -0.148573, -0.516429, -0.207053, 0.09327, 0.124797, 0.087958, -0.077645, -0.050753, -0.091926, 0.013189, -0.146005, -0.152581, -0.064288, 0.111719, -0.253092, -0.123146, 0.015927, -0.128262, 0.09399, 0.182519, -0.082975, 0.082077, -0.191307, -0.251928, 0.035919, 0.123385, -0.133745, -0.164291, 0.107781, -0.043875, 0.11038, -0.246093, -0.070707, 0.018033, -0.152728], [0.124182, -0.087249, 0.178832, -0.094891, -0.127417, -0.103317, -0.22962, 0.14696, 0.098514, 0.079587, -0.041357, -0.041744, -0.046578, -0.045744, -0.089771, -0.04041, -0.038982, -0.232404, 0.131206, 0.029912, 0.182854, -0.094527, -0.036615, 0.004401, 0.116406, 0.067267, 0.089571, 0.021262, -0.069039, 0.086537, -0.081513, 0.130511, 0.084887, 0.049307, -0.026566, -0.106027, -0.053925, -0.191281, 0.038957, 0.056917, 0.080319, 0.067443, -0.171787, 0.148154, 0.076853, 0.181552, 0.00989, -0.105643, 0.047604, -0.179827, -0.124018, -0.077715, 0.024944, -0.020141, 0.128004, -0.156509, 0.144835, 0.056273, -0.033947, 0.159297, -0.123412, 0.09273, -0.125109, -0.080004], [0.155685, 0.091706, 0.018154, -6.3e-05, -0.003255, 0.009556, 0.219044, 0.026561, 0.132091, -0.005967, 0.054176, 0.214603, -0.022883, 0.141108, 0.008419, 0.083255, 0.217198, 0.147411, 0.048927, -0.207811, -0.133357, -0.111391, 0.154918, -0.058477, -0.258461, -0.001854, -0.30524, -0.028337, 0.104606, -0.030069, -0.026275, -0.080627, 0.107932, 0.080487, 0.164905, 0.094323, -0.048017, 0.072676, 0.067133, 0.147523, -0.124539, -0.091929, -0.027444, -0.09684, 0.022329, -0.059406, -0.000697, 0.162246, -0.087999, 0.057556, 0.113308, 0.018916, -0.187124, 0.15749, -0.057737, -0.14517, -0.254861, 0.035943, 0.001246, -0.071084, -0.01665, 0.056962, 0.098825, 0.169102], [0.134107, 0.001242, 0.11078, 0.192037, 0.254932, 0.432933, -0.138837, 0.020079, -0.095944, 0.260852, -0.05345, 0.213628, 0.108697, 0.277199, 0.006394, 0.189292, 0.218192, -0.037944, -0.07942, -0.048983, 0.144013, 0.082004, 0.089453, 0.182167, -0.036655, -0.037612, -0.272171, 0.178935, 0.214392, 0.027863, 0.096196, 0.12722, 0.163018, 0.151875, 0.19397, 0.034772, 0.309031, 0.018805, -0.073248, 0.149692, 0.040889, -0.004121, -0.049074, 0.049036, -0.005262, 0.155918, 0.130423, -0.069568, 0.077653, -0.13709, -0.131675, 0.038239, -0.032983, 0.305442, 0.096154, -0.056713, -0.174085, 0.112511, 0.010381, -0.020162, 0.135346, 0.010752, -0.05873, 0.210692], [0.38005, 0.056145, 0.004735, 0.069029, 0.157812, -0.037077, 0.204157, 0.130488, 0.24212, -0.050483, -0.066667, 0.154831, 0.086586, 0.05945, 0.086229, -0.138982, 0.014657, 0.160044, 0.064013, -0.108554, -0.053136, 0.033132, 0.027793, -0.210216, -0.263232, -0.151988, -0.301845, -0.22192, -0.036759, 0.104696, -0.032886, 0.23827, -0.041436, -0.235885, -0.208263, 0.02252, 0.061971, -0.211673, 0.076951, -0.004088, 0.127755, 0.321616, 0.104138, 0.0489, 0.166453, -0.196701, -0.053134, 0.364069, -0.046583, 0.089649, 0.221564, 0.189344, -0.054152, -0.036389, -0.149008, 0.170797, -0.283485, 0.029389, -0.031958, 0.104973, 0.286362, -0.154722, 0.176823, 0.373639], [0.010177, 0.124757, 0.065961, 0.058389, 0.027556, 0.117592, 0.024358, 0.20239, -0.033335, 0.048417, 0.035756, 0.253778, 0.029659, 0.209239, 0.131007, 0.129108, 0.228101, 0.18981, -0.00722, 0.032929, -0.021586, -0.027682, 0.21003, -0.077298, 0.033925, 0.04125, -0.244548, 0.228016, 0.260736, 0.219438, 0.168932, 0.125871, 0.173498, -0.011536, 0.039921, 0.091386, -0.055685, 0.264245, -0.22837, 0.177717, 0.048629, -0.031056, 0.007908, 0.01913, 0.031319, 0.059993, -0.043943, 0.033763, 0.137175, 0.384524, -0.0407, -0.010667, -0.019001, 0.401725, -0.047672, -0.045465, 0.029667, 0.012155, 0.037881, 0.163967, 0.088869, 0.09579, -0.015169, 0.132769], [0.152279, 0.228693, 0.040441, 0.121042, 0.02249, 0.08201, 0.375594, 0.010707, 0.281459, -0.052806, 0.121248, 0.09602, 0.038221, 0.015878, 0.049685, -0.035353, -0.027034, 0.235752, 0.079537, 0.08873, -0.190686, -0.003292, -0.034991, -0.090008, -0.21112, 0.032374, -0.139958, -0.230259, -0.10275, 0.086929, 0.175573, 0.199606, -0.087367, 0.040297, -0.078163, 0.025707, -0.072901, 0.245342, 0.025504, 0.116158, 0.018961, 0.237095, -0.126846, 0.23497, 0.25901, -0.17348, 0.038588, 0.115474, -0.083466, 0.263559, -0.052808, 0.016979, 0.122802, 0.011097, -0.183994, -0.116835, -0.072554, 0.000136, 0.142373, 0.015912, 0.168696, -0.226042, 0.089099, 0.213224], [0.06968, 0.13369, 0.051992, -0.240608, -0.332252, -0.123143, 0.139773, 0.037356, 0.08483, -0.268382, -0.114722, -0.026145, -0.167943, 0.12532, -0.040218, -0.134829, -0.083733, -0.023946, 0.22987, -0.144451, -0.090002, 0.026727, 0.04043, -0.078785, -0.132979, -0.26472, -0.189046, -0.191547, 0.221927, 0.110578, 0.143653, -0.038249, -0.234285, -0.31931, -0.148412, -0.063286, -0.014104, 0.117143, 0.077762, 0.028387, -0.271583, 0.018757, -0.236504, 0.041511, 0.185127, -0.246041, 0.14395, -0.02879, -0.17799, 0.178661, 0.154115, -0.367135, -0.003013, -0.221213, -0.058736, -0.106477, -0.229347, 0.069319, 0.048131, 0.011473, -0.103487, -0.09286, -0.106693, 0.122045], [-0.165335, -0.175339, 0.044189, 0.08761, -0.128252, -0.024661, -0.090557, -0.01639, -0.06431, -0.197414, 0.12762, -0.19688, 0.049949, -0.169422, -0.125461, -0.092933, -0.277789, 0.000717, -0.136878, 0.030226, 0.126547, 0.13093, -0.006246, -0.089354, 0.10041, -0.004553, 0.209025, 0.105058, -0.038737, -0.055182, 0.04308, -0.065464, 0.078528, 0.241949, -0.035667, -0.022579, -0.019272, 0.077828, 0.001876, 0.019393, -0.047321, -0.101596, -0.050251, -0.090633, -0.104726, 0.107661, -0.098717, -0.126943, -0.073021, -0.250456, -0.001531, 0.0901, 0.020578, -0.115469, -0.05339, -0.126603, 0.153944, -0.072017, -0.10605, -0.157549, -0.133944, -0.11725, 0.128473, -0.132083], [-0.172568, -0.056528, -0.023982, -0.049706, 0.022166, 0.12176, 0.061356, 0.143775, 0.020387, 0.028604, 0.234087, -0.079046, -0.033845, -0.038602, 0.124853, -0.01786, -0.052662, 0.104533, -0.217766, -0.016362, -0.175203, -0.063517, 0.100839, 0.147506, -0.040476, -0.046025, -0.014083, 0.084944, -0.122671, -0.193962, 0.048422, -0.017989, 0.053954, 0.047833, 0.098239, -0.014943, -0.114408, 0.228674, -0.276269, 0.048675, -0.011516, -0.253215, 0.096689, -0.046602, -0.143951, 0.005238, -0.043061, -0.051711, -0.017802, -0.076796, 0.038724, 0.00462, -0.170274, -0.10661, 0.032554, 0.04408, -0.084409, 0.01671, -0.080695, -0.089652, 0.106862, -0.110377, -0.023115, 0.057384], [-0.126722, -0.098358, -0.090974, 0.122224, -0.072955, -0.013803, -0.07857, -0.175877, -0.023172, -0.082529, -0.008807, -0.270218, -0.085826, -0.212331, -0.251769, -0.039435, -0.117063, -0.192338, -0.109191, -0.058543, -0.103778, 0.196125, -0.125187, -0.012619, 0.04014, -0.003063, 0.166769, -0.133447, -0.027742, 0.1719, 0.017418, -0.178201, -0.22901, -0.116951, -0.222549, -0.16268, -0.185207, 0.014604, 0.052341, 0.085812, 0.166898, 0.04666, 0.101676, 0.157008, 0.044034, 0.024353, -0.11133, 0.082183, -0.063564, -0.158456, 0.080905, 0.147711, 0.185237, -0.253736, -0.085232, 0.187655, 0.057384, -0.18039, -0.061856, -0.146364, 0.022775, 0.046708, 0.045486, -0.227427], [0.119842, 0.209061, 0.154335, -0.214123, 0.251829, 0.005032, 0.029106, 0.008938, 0.126066, 0.109456, -0.143788, -0.017994, -0.125135, 0.192433, 0.014046, 0.021352, 0.247233, 0.110811, 0.27328, -0.018743, -0.331981, 0.053294, -0.124723, -0.035461, -0.280929, -0.076528, -0.366476, 0.172307, 0.26641, 0.116031, 0.1942, 0.224268, -0.047347, -0.119763, 0.036186, 0.014395, -0.013899, -0.058452, -0.152281, -0.116996, 0.058431, -0.025309, -0.148139, 0.163727, 0.100847, 0.285414, 0.017672, -0.078813, 0.211727, 0.268677, 0.025444, -0.100941, 0.188398, 0.073085, 0.003779, -0.133371, 0.059605, 0.138298, 0.221659, -0.056702, -0.001377, 0.137602, -0.177999, 0.290247], [0.02734, 0.046411, -0.006803, 0.251066, 0.264694, 0.094088, 0.000115, 0.071769, -0.114342, -0.029607, 0.097265, -0.003364, 0.12311, -0.156113, 0.011906, 0.017397, 0.019145, 0.055341, 0.004486, 0.113472, 0.101448, 0.218161, 0.077669, 0.0579, 0.270895, 0.112729, 0.22406, 0.009105, 0.007149, 0.021532, 0.161954, -0.083299, 0.194069, 0.123586, 0.005083, 0.016844, 0.162847, -0.061341, 0.019306, 0.088731, 0.193729, -0.017113, 0.097771, -0.113899, 0.096117, 0.003443, -0.153251, 0.021486, 0.108844, -0.062364, -0.032295, 0.317122, -0.10646, 0.04568, -0.03041, 0.067588, -0.058182, 0.029555, -0.100395, 0.053622, 0.028292, -0.086624, 0.199165, -0.095184]], [0.129682, -0.108296, -0.015425, -0.194292, -0.184276, -0.200567, 0.052241, -0.05146, -0.093759, 0.088199, 0.067741, 0.20097, -0.148926, 0.06189, 0.115419, -0.080506, 0.035953, -0.063082, 0.009507, -0.147847, -0.046142, -0.309072, 0.040627, -0.095403, -0.204589, -0.141247, 0.006627, -0.131427, 0.079984, -0.083496, -0.264407, 0.079375, -0.040463, -0.114705, 0.16561, -0.123956, 0.042783, 0.065527, -0.104726, -0.145964, -0.079074, -0.032983, 0.068665, -0.010791, -0.025157, 0.031554, 0.001094, -0.157597, -0.070389, -0.066716, 0.088631, -0.301412, -0.190611, -0.188415, -0.043639, 0.056589, -0.143143, 0.027345, -0.0128, 0.120092, -0.193075, 0.024479, 0.02599, -0.109028]),
    ([[0.082478, -0.057386, 0.017164, 0.062862, -0.011994, -0.193863, 0.02976, -0.05915, -0.027453, 0.034614, -0.058151, -0.043637, 0.001939, -0.007795, 0.12509, -0.022755, 0.043624, -0.07842, 0.151287, 0.090118, 0.026965, -0.122956, 0.034601, 0.07606, 0.083487, 0.096219, 0.086084, -0.033528, 0.063312, -0.112961, -0.082726, 0.090341], [-0.025248, 0.110355, -0.013796, 0.016194, -0.084679, 0.059453, 0.030837, 0.153293, -0.08392, 0.094771, 0.042243, -0.020262, 0.0158, -0.105034, -0.0058, 0.12993, 0.07687, -0.105787, 0.103214, 0.096284, -0.118594, 0.132148, -0.092959, 0.006473, 0.08068, -0.010348, 0.149259, 0.006142, -0.03134, -0.010271, -0.127052, -0.110361], [0.070342, -0.041663, 0.154863, 0.031872, -0.120649, -0.038199, 0.074445, 0.113848, 0.050125, 0.048487, 0.073648, 0.046361, -0.012488, -0.005083, 0.078899, 0.018531, 0.113765, 0.062324, 0.140205, -0.037603, -0.133555, 0.12209, 0.08434, 0.195777, -0.096787, 0.035715, 0.026107, -0.005425, 0.03321, 0.10686, 0.124207, -0.016613], [-0.003578, 0.175783, 0.153951, 0.043394, 0.035717, 0.143749, -0.163587, 0.089416, -0.229118, 0.127726, 0.054185, 0.019635, 0.019995, -0.044092, 0.051969, 0.102111, -0.206243, 0.098645, 0.042402, -0.035461, -0.015233, 0.07943, -0.00632, 0.007133, -0.132707, 0.002584, 0.087564, -0.042205, -0.10014, -0.045741, -0.024842, 0.086941], [-0.027457, 0.177141, -0.125624, 0.0278, -0.18493, 0.108284, 0.157056, -0.207173, 0.15637, -0.177284, 0.197728, 0.084872, -0.037327, -0.047238, -0.120829, -0.078446, 0.011261, 0.096161, -0.039625, 0.135388, 0.031463, -0.17202, 0.147805, -0.201057, 0.085491, -0.12091, -0.218833, -0.037186, 0.162916, 0.129498, 0.182118, 0.059082], [-0.001015, 0.13784, -0.121658, 0.014105, -0.168928, 0.062395, 0.185715, -0.036446, 0.028778, 0.005974, 0.160353, 0.03283, 0.111846, -0.096617, 0.074566, -0.05208, 0.13765, 0.052396, -0.032079, 0.038698, -0.102099, 0.116829, 0.060709, 0.285609, 0.218387, 0.151435, 0.073978, 0.084096, -0.018435, 0.236006, 0.153455, -0.084191], [0.077804, -0.071282, 0.037251, 0.082094, 0.01678, 0.059061, 0.024883, 0.093714, -0.12316, -0.04375, -0.129661, -0.124125, 0.005974, -0.00868, 0.013789, 0.08571, 0.069172, -0.121782, 0.017742, -0.177086, 0.017057, -0.034191, -0.082169, 0.140582, -0.180425, 0.113275, 0.040343, -0.147729, 0.01723, -0.012709, -0.106904, 0.0841], [0.032999, 0.116103, 0.050783, 0.117176, 0.030008, -0.013989, 0.156798, 0.099324, 0.127803, 0.118151, 0.100845, 0.082093, 0.097435, -0.034231, -0.0402, 0.068722, 0.081696, 0.044382, 0.147163, 0.114725, -0.094733, 0.058856, 0.091917, 0.290824, 0.211615, 0.21805, 0.11495, -0.005538, 0.104164, 0.136106, 0.110899, -0.134491], [-0.004165, -0.002538, 0.052211, 0.088231, 0.021404, -0.025287, 0.017631, -0.023234, -0.046274, 0.112018, -0.016128, -0.048853, 0.024986, -0.039173, 0.074765, 0.012509, 0.097242, 0.06546, 0.030017, 0.00218, -0.143365, 0.025805, 0.013813, 0.082357, -0.126535, 0.097018, 0.146286, -0.032971, -0.072946, -0.111856, 0.034277, -0.026573], [-0.027821, 0.005615, -0.055511, -0.021745, -0.079368, 0.144723, 0.064401, -0.008061, 0.104778, -0.02743, 0.095909, 0.123587, -0.011942, 0.028366, -0.000403, 0.067583, 0.079695, 0.153654, 0.096994, 0.054458, 0.05639, -0.04473, 0.107852, -0.172093, 0.035923, -0.090437, -0.060786, -0.042144, 0.062709, 0.073669, -0.007774, -0.014331], [-0.012645, 0.140353, 0.104824, -0.039974, 0.0201, -0.198157, 0.00866, -0.092666, 0.145375, -0.037288, 0.00416, 0.090318, -0.125806, -0.082556, -0.057972, -0.205999, 0.082657, 0.10879, -0.04794, 0.103587, 0.025491, -0.14666, -0.033715, -0.045331, 0.079709, -0.039393, -0.073359, -0.052384, 0.100325, -0.057499, -0.044914, 0.062175], [-0.19164, -0.049457, -0.004162, -0.032368, -0.098121, 0.043037, -0.003373, -0.033534, 0.111005, -0.20621, 0.088473, 0.093611, 0.043516, 0.00176, -0.017842, -0.133059, 0.031372, -0.008654, -0.055111, 0.008397, -0.092888, -0.098236, -0.016493, -0.233452, -0.012659, 0.023716, -0.069484, -0.054589, 0.063093, 0.032453, 0.119428, -0.106391], [0.071186, 0.122539, 0.162185, -0.077622, 0.029345, 0.003453, 0.025239, -0.072973, 0.119153, 0.05734, 0.001426, 0.038526, 0.038645, 0.078688, -0.003186, -0.14598, 0.118589, -0.060624, -0.064759, -0.016398, -0.077693, -0.126528, -0.019752, -0.044285, 0.091188, 0.036688, -0.021163, 0.03943, 0.101765, 0.101451, -0.023821, 0.046594], [0.043282, 0.050397, -0.23328, 0.007203, -0.023207, -0.080471, -0.238784, 0.108892, -0.107648, 0.005348, 0.090628, 0.038477, 0.092323, -0.189854, -0.036892, -0.027313, -0.152651, 0.024598, 0.124421, -0.112525, -0.048633, -0.013198, -0.076329, 0.052846, -0.187021, -0.017532, 0.116793, -0.073932, 0.064871, -0.239901, 0.090743, -0.117165], [-0.103935, 0.094133, 0.120535, -0.067046, -0.073297, 0.113572, 0.101988, -0.068273, 0.135586, -0.174488, 0.056084, 0.068618, 0.063029, 0.089865, 0.023978, -0.154617, 0.009587, -0.08954, -0.090067, -0.008806, -0.081771, 0.025442, 0.052886, -0.271886, 0.018177, -0.165014, -0.13164, -0.011511, 0.116068, 0.077926, -0.041787, 0.022697], [0.099229, 0.096922, 0.00011, 0.009562, 0.065078, -0.068183, 0.046316, 0.109735, 0.12507, 0.074147, 0.148146, -0.003845, 0.063269, 0.133964, 0.016937, -0.004157, -0.010112, 0.071067, -0.013225, 0.032582, -0.108759, -0.010394, 0.135856, 0.205663, 0.079635, 0.120711, 0.097587, -0.099249, 0.168387, 0.123105, 0.036013, -0.128513], [-0.08828, 0.004058, -0.131159, 0.040729, -0.130095, -0.023231, 0.038147, -0.166528, 0.068825, -0.035543, 0.056355, 0.036732, 0.007222, -0.0178, -0.078777, 0.017117, 0.111016, -0.082533, -0.098958, -0.038143, -0.039379, -0.004126, -0.041168, -0.115551, -0.038212, -0.079197, 0.040855, -0.0084, 0.171655, 0.11712, 0.033833, -0.040046], [0.061206, -0.007456, 0.180728, -0.045238, -0.065231, -0.164219, 0.167103, 0.178065, -0.179924, 0.020383, -0.077861, -0.056259, 0.048757, 0.178725, 0.063149, -0.049771, 0.06909, -0.176737, 0.115794, -0.182375, -0.135264, 0.178503, -0.114063, 0.131963, -0.154405, -0.036145, 0.000682, 0.092043, 0.010783, 0.01564, -0.004726, -0.032841], [-0.132196, -0.13979, -0.000697, -0.009524, -0.174007, -0.153277, 0.01128, 0.040963, -0.006398, 0.009571, 0.02856, -0.078393, 0.147691, -0.088479, 0.109125, 0.159842, 0.0398, -0.010221, 0.101663, 0.022976, 0.022873, 0.098791, 0.099167, -0.022711, -0.107433, -0.00359, 0.121029, 0.148858, -0.010131, -0.116687, -0.079103, -0.079346], [0.083933, -0.003614, 0.13106, 0.114453, 0.102518, 0.155346, 0.089407, -0.046197, -0.091976, 0.011053, 0.125, -0.021706, 0.116493, 0.071613, 0.101956, 0.043962, -0.010975, 0.104814, 0.055177, 0.08007, 0.03881, 0.066953, -0.019783, 0.089639, 0.196306, -0.019158, 0.125973, -0.088256, -0.148889, 0.050051, -0.055762, 0.171217], [-0.138406, 0.050228, -0.237085, -0.211536, -0.011195, 0.149567, 0.021942, -0.116586, 0.069101, -0.083028, 0.094892, 0.100709, 0.105527, -0.229215, -0.081959, -0.046079, -0.03862, 0.180378, -0.109063, -0.000508, 0.135396, -0.175353, 0.138245, -0.081993, 0.130897, -0.020526, -0.018113, -0.093114, 0.032783, 0.148915, 0.077625, -0.07401], [0.053356, -0.113547, 0.1096, 0.178534, -0.095755, 0.049809, 0.008685, 0.184089, -0.162973, 0.174012, -0.169718, -0.099799, 0.078416, -0.018309, 0.06881, 0.114593, -0.032056, -0.005105, 0.023371, -0.046512, 0.102447, 0.072974, -0.002422, 0.191524, -0.089878, 0.044534, 0.007499, 0.202849, -0.051104, -0.022886, -0.120927, -0.041578], [-0.211236, 0.060553, -0.082934, -0.1165, -0.028457, 0.021165, 0.103908, -0.06734, -0.029693, -0.011934, 0.100251, -0.036764, -0.167505, -0.019959, -0.089816, 0.047853, 0.052662, -0.105926, -0.160955, 0.096879, 0.107582, -0.134752, 0.091954, -0.165464, 0.023898, -0.057531, -0.065063, -0.16879, 0.128471, -0.013833, 0.039066, -0.060839], [-0.094058, 0.114503, 0.075471, 0.085881, 0.014776, 0.18983, 0.028821, 0.061136, 0.120351, -0.055964, 0.049495, -0.004434, -0.059505, 0.125955, 0.0184, -0.067239, 0.07951, 0.02441, 0.02308, 0.042992, 0.043516, -0.016211, 0.064323, -0.151957, -0.016618, -0.178475, -0.270045, 0.137249, 0.060397, 0.015953, -0.006897, -0.063628], [-0.02821, 0.119503, -0.025768, -0.023373, 0.046873, 0.096663, -0.019309, -0.228248, -0.125446, -0.157424, -0.11149, 0.032218, -0.073594, 0.015973, -0.137619, -0.122702, 0.065896, 0.088462, -0.064698, -0.049693, 0.055033, -0.228806, 0.067283, -0.193758, -0.284208, -0.148423, -0.066388, 0.135063, 0.01069, -0.106873, 0.072435, 0.135146], [-0.057551, 0.004545, 0.083916, -0.161919, -0.012975, 0.029121, 0.143408, -0.180902, -0.001467, 0.03667, -0.004782, 0.113696, 0.007516, 0.038682, -0.114504, -0.111187, 0.12726, 0.164579, -0.133939, 0.120674, 0.120756, -0.082224, 0.02202, -0.251804, 0.102059, -0.05474, -0.12572, 0.037097, 0.132674, 0.14317, 0.145434, 0.041335], [-0.030594, 0.094886, -0.094543, -0.14782, 0.108001, -0.004392, -0.056924, -0.079085, -0.096407, -0.169517, -0.069873, 0.099964, -0.076757, -0.103792, -0.061833, -0.275618, 0.113873, -0.011438, -0.042506, -0.085118, 0.18494, -0.147132, 0.045047, -0.368272, 0.092621, -0.370115, -0.010157, -0.154937, 0.114918, -0.025505, 0.095549, 0.105866], [0.009582, 0.074777, -0.270863, -0.007137, 0.050312, 0.098255, 0.026862, -0.016347, 0.104892, 0.011735, 0.058039, 0.06821, -0.094347, 0.030299, -0.076491, 0.137194, 0.06138, -0.009301, -0.003759, 0.14325, -0.022137, 0.04514, 0.095825, 0.061224, 0.056831, 0.245513, 0.023503, 0.010087, 0.082539, 0.136682, -0.010949, 0.010573], [-0.087126, 0.020365, -0.110927, -0.051037, 0.124059, -0.125827, -0.262508, 0.093116, -0.05482, 0.078717, 0.112491, 0.069764, 0.000829, -0.067421, -0.036006, -0.060502, -0.218226, 0.084858, 0.060286, 0.069653, 0.057624, -0.024852, -0.096962, -0.001808, -0.269775, 0.103168, -0.021615, 0.00864, -0.026354, -0.384679, -0.100817, -0.068991], [0.274155, 0.059645, 0.008881, 0.138326, 0.072225, 0.199992, 0.107851, 0.167403, 0.01882, 0.235469, 0.075154, 0.044962, -0.130699, 0.029243, 0.149295, 0.04696, 0.111502, 0.130054, 0.067385, 0.116644, -0.050826, 0.02333, 0.044988, -0.092872, -0.012073, 0.111157, 0.089012, -0.066773, -0.023347, 0.03427, -0.00089, 0.129088], [0.125873, -0.123042, 0.194105, 0.239308, 0.062627, -0.183489, 0.050411, 0.040607, -0.132476, 0.185397, -0.188096, -0.059316, 0.09547, -0.146295, 0.146809, 0.027009, -0.044553, -0.015245, 0.0142, 0.066898, -0.131984, 0.136941, -0.086784, 0.230564, 0.009937, 0.185983, 0.024801, 0.225288, 0.002215, -0.159607, -0.061442, -0.093599], [0.037141, -0.087992, 0.001485, -0.017114, 0.025488, 0.023611, 0.032326, -0.023998, -0.035912, 0.031296, -0.073768, -0.11615, 0.109243, -0.151058, 0.119687, 0.123589, 0.017598, -0.139032, 0.016484, -0.039816, -0.034331, -0.025381, -0.065429, -0.006131, -0.16133, 0.002311, 0.131979, -0.079058, -0.036281, -0.01623, -0.008105, -0.107575], [-0.138223, 0.080412, -0.289174, -0.122214, 0.079001, 0.068128, -0.030274, -0.244067, 0.035112, -0.074184, 0.068675, 0.078555, -0.036698, -0.157933, -0.006701, -0.064388, -0.038666, -0.018716, -0.021704, -0.074659, -0.098207, -0.163189, -0.054425, -0.191808, -0.002717, -0.096994, -0.045461, 0.0251, -0.052163, 0.006418, 0.074513, -0.225531], [-0.014403, 0.145137, -0.237463, -0.03353, -0.184765, -0.061626, -0.033594, 0.04494, 0.171748, -0.12891, 0.083176, 0.101798, 0.00633, -0.082203, -0.017256, -0.069066, 0.056936, 0.08633, -0.07952, -0.081064, 0.045553, 0.120075, -0.076552, 0.092357, 0.019931, 0.159435, 0.055053, -0.243648, 0.121057, 0.084229, 0.119043, -0.091133], [-0.247434, -0.028606, -0.055457, -0.071758, -0.051813, -0.011275, 0.075345, -0.083178, 0.096495, -0.13568, 0.131521, 0.122578, -0.022669, 0.011947, -0.130795, -0.19691, -0.054447, 0.115271, -0.030715, -0.047654, -0.056488, -0.032322, 0.033078, -0.058739, -0.081411, -0.121281, -0.057227, -0.09187, 0.141771, 0.064389, 0.100387, -0.091141], [-0.03459, 0.108534, -0.176536, -0.104513, -0.133751, 0.017448, 0.033255, -0.038357, 0.048574, 0.06321, 0.078668, -0.020082, -0.04324, 0.015323, -0.099821, -0.026315, -0.023341, -0.023807, -0.059321, -0.022863, 0.002247, -0.028164, 0.109845, 0.121592, -0.032886, 0.003827, 0.076402, 0.172009, -4e-06, 0.143221, 0.070134, -0.086564], [-0.09812, 0.097414, -0.002547, 0.109779, -0.016423, 0.152469, 0.045915, -0.06205, 0.056162, -0.024608, 0.061311, 0.025091, 0.046436, -0.009108, 0.060242, 0.044867, -0.040558, 0.056181, 0.052354, -0.079385, -0.114272, -0.067309, 0.023142, 0.066527, 0.008221, 0.102347, 0.06549, 0.091149, 0.085708, 0.119479, 0.079448, 0.117706], [0.051071, 0.100946, -0.242977, -0.141059, -0.296709, -0.03294, 0.002605, 0.145511, -0.026565, -0.120639, 0.13555, -0.050477, 0.065998, 0.046768, -0.086686, 0.11244, 0.099096, 0.013396, -0.006085, 0.078511, -0.128355, 0.031301, -0.030023, -0.107457, 0.237414, 0.007301, 0.08495, 0.052436, 0.120893, 0.153485, -0.003582, 0.049467], [0.040919, -0.017305, 0.131108, 0.141389, -0.035835, -0.050067, -0.007356, 0.121019, 0.030766, 0.116372, -0.075946, 0.113533, -0.029188, 0.075538, 0.207679, 0.048916, 0.030651, -0.012417, 0.117769, 0.047899, 0.017009, -0.00953, 0.059308, 0.096174, -0.379096, 0.155649, 0.015453, -0.003475, -0.069585, -0.121209, 0.023491, -0.059609], [-0.078865, 0.122601, 0.06122, 0.00094, 0.06239, 0.169809, 0.143137, -0.077375, 0.093191, -0.065158, 0.046915, 0.037503, 0.046982, 0.09138, -0.010628, -0.069811, 0.063345, 0.052334, 0.055101, 0.015027, -0.020675, 0.028698, 0.019399, 0.092181, 0.002311, -0.091062, 0.042342, -0.004751, 0.027999, 0.159363, 0.006374, 0.044543], [0.262567, 0.071767, -0.007112, 0.117532, -0.358886, -0.059798, 0.111343, 0.020227, 0.009406, 0.151147, 0.030191, 0.095104, -0.0505, 0.011672, 0.056986, -0.081213, -0.012015, 0.072173, -0.061101, 0.079579, 0.083573, -0.077554, 0.138557, -0.124481, 0.21207, -0.120911, 0.064471, 0.084273, 0.097494, 0.087414, -0.00545, 0.158829], [0.050715, -0.130518, 0.051769, 0.067785, -0.044557, -0.052892, 0.017931, 0.115594, 0.14941, -0.064497, -0.041434, -0.128685, -0.057329, -0.059971, 0.152609, 0.014941, 0.132622, -0.047201, 0.081363, -0.007748, -0.054775, 0.023383, -0.038757, 0.079744, 0.125133, -0.023718, -0.021473, -0.082634, 0.087891, 0.092095, -0.042965, 0.026021], [-0.077246, -0.092496, -0.025606, -0.082453, 0.117309, -0.174042, -0.054024, 0.065603, -0.128473, 0.014712, -0.080307, -0.014277, 0.058039, -0.14392, 0.062073, 0.008442, 0.053778, -0.128759, -0.064619, -0.12337, 0.219073, -0.105675, -0.065279, 0.076439, -0.006327, 0.120711, -0.167156, -0.094112, -0.063393, -0.017429, -0.073843, -0.011059], [0.272841, -0.031251, 0.063943, 0.017847, -0.103351, 0.115076, -0.024952, 0.135732, -0.045653, 0.164374, 0.111068, 0.126532, -0.094545, 0.037772, 0.096666, 0.011547, -0.002164, 0.106583, 0.09382, 0.116364, 0.029458, -0.050557, 0.110977, 0.215976, 0.113265, 0.001055, 0.136941, 0.023518, -0.002816, 0.070705, 0.125993, 0.085325], [-0.025284, 0.011553, -0.018184, 0.044258, -0.272019, -0.132839, 0.078864, 0.0595, 0.02791, 0.062664, -0.023687, -0.050184, 0.078876, 0.009548, 0.134898, 0.040623, 0.009728, -0.001302, 0.168958, 0.063378, -0.014355, 0.023212, 0.037238, 0.015366, 0.161908, 0.029458, 0.128434, -0.011619, 0.072072, -0.046146, 0.09452, -0.056095], [-0.211046, -0.031726, -0.004871, -0.012366, 0.102773, 0.040721, -0.107837, -0.167661, 0.122086, -0.009277, 0.006355, 0.009902, 0.088227, 0.036561, -0.024932, 0.048185, -0.118834, 0.056838, 0.005624, -0.010931, -0.02536, 0.061937, -0.054575, -0.039417, -0.001054, -0.006019, -0.059002, 0.036085, 0.073833, -0.069701, -0.032601, 0.02136], [0.054387, 0.000812, -0.076302, -0.038302, 0.087649, -0.066009, 0.004572, 0.093764, 0.18527, 0.082704, 0.106333, 0.032665, -0.027672, -0.138335, 0.067475, 0.003788, -0.066457, 0.012879, 0.10238, 0.150969, 0.076222, 0.101333, 0.087816, 0.068368, 0.017554, 0.063179, 0.034834, 0.050647, -0.009846, -0.10598, -0.097405, 0.059329], [0.128975, -0.050429, 0.266894, 0.026534, -0.038313, -0.127132, 0.059304, 0.006269, -0.032831, 0.148704, -0.05001, 0.033864, 0.041464, 0.112878, 0.053232, 0.047128, 0.09282, -0.175577, 0.007803, -0.097256, 0.034489, -0.047793, -0.019797, 0.067316, -0.115909, 0.062953, 0.081646, -0.017657, 0.088474, -0.139868, -0.035025, 0.120579], [-0.025107, 0.110229, -0.108366, -0.123779, -0.070559, 0.099691, -0.065288, -0.207682, 0.087021, -0.024813, 0.04695, 0.13598, 0.008283, -0.026333, 0.101396, -0.048965, 0.131179, 0.043588, -0.099916, 0.061011, -0.056602, -0.085195, 0.003324, -0.07345, 0.149552, -0.032184, 0.035889, -0.108134, 0.127316, 0.074598, 0.08735, 0.058218], [0.09859, -0.037882, -0.004963, 0.071709, -0.006554, -0.056768, -0.094261, 0.100292, -0.004387, 0.116434, -0.090579, -0.061141, 0.085853, -0.134469, 0.059443, 0.017782, 0.011187, -0.108746, 0.079242, 0.044148, 0.157597, 0.071991, -0.051382, 0.139159, -0.096679, 0.04214, 0.03107, 0.027626, -0.04357, -0.033518, 0.075685, 0.055161], [-0.028149, -0.067144, -0.014611, -0.064969, 0.02952, 0.05703, -0.114518, -0.052855, -0.198205, 0.12265, -0.17206, -0.068931, 0.072602, -0.123896, -0.01432, 0.021814, 0.010308, -0.215405, -0.047916, -0.284734, -0.106226, 0.008117, -0.17633, -0.035493, -0.372851, 0.102265, 0.060843, 0.08565, 0.040204, -0.146639, 0.108929, -0.002293], [0.035369, 0.205836, 0.177393, 0.043873, -0.273529, 0.015764, 0.141363, 0.009258, 0.194641, 0.104873, 0.124298, 0.136545, -0.166582, 0.137855, -0.000543, -0.013406, 0.179315, 0.048342, 0.017126, 0.111202, -0.24133, -0.072553, 0.121764, -0.252365, 0.255053, -0.185106, -0.12648, -0.002475, 0.187598, 0.143787, 0.143772, 0.121943], [0.11111, 0.025129, 0.08289, 0.152735, -0.206912, -0.108732, -0.003798, -0.016264, 0.088288, 0.127993, 0.008518, -0.019515, -0.032902, 0.097144, 0.087078, 0.132, 0.049931, -0.060283, 0.121687, 0.007648, 0.085304, 0.159767, -0.065204, 0.170961, 0.070259, -0.056736, 0.048956, 0.191145, -0.080233, 0.038561, 0.059057, 0.082029], [-0.116392, 0.190899, -0.089055, -0.121739, 0.074628, 0.235506, 0.092784, -0.0169, 0.037396, -0.102116, 0.097348, 0.041218, 0.017466, -0.070099, -0.219227, -0.006613, 0.091511, 0.199818, -0.041358, 0.009838, 0.073854, -0.136912, 0.183724, -0.132398, -0.062487, 0.091722, -0.021051, -0.125223, -0.001095, 0.188192, 0.0594, -0.095944], [0.021913, -0.024119, -0.246778, 0.024408, 0.034756, 0.067472, 0.024205, -0.058467, 0.150352, -0.053773, -0.011937, -0.09258, 0.009271, -0.10558, 0.111922, 0.113745, 0.024377, 0.135917, 0.117836, 0.110079, 0.017739, -0.048558, 0.133633, -0.00423, 0.114347, 0.00658, 0.068393, -0.013107, 0.092071, 0.047169, -0.081119, 0.130282], [-0.04957, 0.070633, -0.114899, -0.039803, 0.046641, -0.144483, -0.124968, -0.037284, -0.097856, -0.118127, -0.125262, 0.053004, 0.074497, -0.035309, -0.051683, -0.02514, -0.016059, -0.156967, -0.028898, -0.006838, 0.065504, 0.050219, -0.07746, -0.283581, -0.090805, -0.072557, 0.012989, 0.133056, -0.002216, -0.198808, -0.053501, 0.040796], [-0.141099, -0.010855, 0.258043, 0.17517, 0.199124, -0.082493, -0.013801, -0.006598, 0.010884, -0.106428, -0.029965, -0.159456, 0.125315, -0.0601, -0.203927, -0.100238, -0.210666, -0.105962, 0.028399, -0.216394, 0.081852, 0.104091, -0.298355, -0.304934, -0.073482, -0.356804, -0.192501, 0.150547, -0.037229, -0.162386, -0.136649, -0.087476], [0.038798, 0.003875, -0.068727, 0.064501, 0.050377, 0.078214, -0.042248, 0.020393, 0.111473, 0.156487, 0.078118, 0.132712, 0.101698, -0.093792, -0.011202, 0.06775, -0.130691, 0.11604, 0.121475, -0.006416, 0.054661, -0.043527, 0.134253, 0.089973, -0.300243, 0.114237, 0.154347, -0.08621, -0.093442, 0.069648, 0.069634, 0.081231], [0.144547, 0.038267, 0.029185, 0.013176, 0.033249, 0.009955, -0.11764, 0.106695, 0.089571, 0.014572, -0.108126, -0.097892, 0.073697, -0.074043, 0.037211, 0.08261, -0.07537, 0.084927, 0.098867, -0.009119, -0.111248, 0.120708, -0.085823, 0.027984, 0.082552, 0.098483, 0.004973, 0.060235, -0.101502, 0.023852, -0.047277, 0.002956], [-0.027557, 0.052436, 0.075074, 0.028591, -0.085651, -0.038922, 0.002433, -0.025207, 0.0702, -0.012676, -0.054633, 0.078314, -0.096782, 0.064002, -0.057036, 0.068966, 0.08224, 0.090462, 0.06707, 0.09585, -0.002822, 0.005267, 0.081846, -0.377097, -0.062079, -0.249142, -0.087126, -0.034762, 0.11473, -0.057306, 0.108883, 0.050983], [0.242613, -0.025949, 0.194099, 0.042314, -0.033931, 0.010031, -0.079904, 0.113567, -0.042991, 0.197401, -0.059036, -0.049372, 0.088109, -0.060013, 0.102302, 0.002256, -0.12721, -0.058641, 0.003703, 0.003599, 0.100447, 0.178701, 0.069413, 0.112088, 0.17802, 0.057185, 0.164287, -0.001085, -0.086281, 0.157149, 0.05088, 0.029045], [-0.20482, -0.004297, 0.032298, 0.072366, -0.052856, 0.083796, 0.077882, -0.175535, -0.059399, -0.179975, -0.082047, 0.049186, 0.033051, -0.013269, 0.042374, -0.010965, -0.046183, 0.06418, -0.040257, -0.024752, -0.104505, -0.06514, -0.084492, -0.124484, 0.066528, -0.150436, -0.037076, 0.118329, 0.02305, -0.050704, -0.039928, 0.000827], [0.024076, 0.088784, -0.028678, -0.033378, -0.096817, -0.075351, 0.136988, -0.083003, 0.061675, -0.09713, 0.102763, 0.130466, -0.052233, 0.057185, 0.062507, -0.041369, 0.125963, -0.009576, -0.031984, 0.129394, 0.094954, -0.11063, 0.037805, -0.258573, -0.002396, -0.060975, -0.060772, -0.004838, 0.007889, 0.108535, 0.083236, 0.018718], [0.00635, 0.049549, 0.01271, -0.050276, -0.075719, -0.028143, -0.155099, 0.042527, -0.113606, 0.117004, 0.041747, 0.032953, 0.039845, -0.110299, 0.026691, 0.078071, 0.011104, -0.118068, 0.062023, -0.168765, -0.075264, 0.072793, -0.310282, 0.122194, -0.288466, -0.062719, 0.065049, -0.194805, 0.026215, -0.222009, -0.050734, -0.117336]], [-0.21866, 0.064715, -0.217994, -0.056718, 0.015031, -0.037728, 0.114059, -0.081731, 0.060069, -0.108472, 0.026271, 0.077378, 0.078699, -0.193917, -0.017182, 0.096823, 0.076703, -0.151, 0.079405, 0.060603, -0.027497, -0.067791, -0.099105, -0.160052, -0.163071, -0.005303, 0.010511, 0.009215, 0.156657, -0.083314, 0.079362, 0.056393]),
    ([[0.251127], [-0.107817], [0.343058], [0.061999], [-0.2275], [-0.239245], [-0.134913], [0.188172], [-0.136084], [0.177929], [-0.062901], [-0.059583], [0.081976], [0.270249], [0.110778], [0.093021], [-0.065286], [-0.101159], [0.126903], [-0.083405], [0.169073], [0.178107], [-0.2276], [0.197622], [-0.2155], [0.164656], [0.170088], [0.086753], [-0.123834], [-0.163657], [-0.107913], [0.188534]], [0.062196]),
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
        'gate_thr': 24.0,
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
        'pred_scale': 1.15,
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
        'slw_thr': 3.0,
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
