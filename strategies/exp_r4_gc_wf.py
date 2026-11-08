"""exp_r4_gc_wf (home, R4 walk-forward twin): exp_r4_mix_wf + ht_* + cs_*; pass the base params by --set.

exp_r4_comp_wf: cs_gru (prediction inside the taker signal), cs_fair (fair_sig inside the taker signal) and cs_q (pred_raw inside the quote signal) split the signal scaling; all 1.0 = identical to the base.

exp_r4_mix_wf: mm_r3_gru3 with ridge/slow/MLP/MLPQ refit on round3+round3_mirror (all) + round4p75+round4p75_mirror (first 75 %).

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
    # rfit 19 (multi: round3+round3_mirror+round4p75+round4p75_mirror, research/retrain_multi.py); params rfit=19
    (0.0212472, (0.3513042, -1.6783239, 4.0246674, -1.9210756, 1.0979778, -0.398875, 0.0102183, 0.0968577, 0.0071829, -0.0118946, -0.1526338, 0.2036759, 0.0412353, -0.0418429, -0.0612165, 0.002647, 0.1503187, -0.4787358, 0.0004501, 0.0003679, -2.89e-05, -0.0007412)),
)
# exp_r3_mlp: 22-32-16-1 ReLU MLP on the same 22 features, target = 200 ms (robust mid at t+5ms+200ms - mid now), fitted on the
# first 75 % of round3 + mirror (research/diag/mlp_r3.py); holdout corr 0.49 vs 0.463 for the ridge. Inputs clipped at +-8 sd.
MLP_MEAN = [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.233775, 0.0, 0.0, 5.907416, 5.907558, 0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, -0.000398, 4.035515]
MLP_SCALE = [1.05536, 0.918208, 0.870829, 0.76647, 0.674128, 2.379676, 15.556883, 0.711496, 13.418242, 0.710788, 0.650832, 0.543642, 1.654303, 1.654388, 0.999996, 4.40365, 1.067292, 0.354426, 130.050751, 328.422974, 986.285767, 1.340564]
MLP_LAYERS = [
    ([[0.075683, 0.074795, -0.227473, -0.086523, -0.017725, -0.027789, -0.194203, 0.145311, 0.078562, -0.633915, 0.080782, -0.269909, 0.187394, -0.200121, 0.170571, 0.026247, -0.084647, -0.325396, 0.063111, -0.011416, 0.001265, 0.2094, -0.05235, -0.009533, -0.00486, -0.273607, 0.201519, -0.119638, 0.823696, -0.048841, 0.202683, 0.021632], [0.111856, 0.095862, -0.340618, 0.027021, -0.017469, 0.130538, 0.138749, -0.273431, 0.088162, -0.215127, 0.056187, 0.070035, -0.106594, 0.076734, -0.046007, -0.128656, 0.100604, -0.163906, 0.059392, 0.395674, -0.136016, -0.10811, 0.262195, 0.069502, 0.095441, -0.111492, -0.022697, 0.014787, 0.099527, 0.134348, -0.025443, -0.010442], [-0.300925, -0.15056, -0.141804, 0.118648, 0.230049, -0.217948, 0.357067, -0.010077, 0.15646, 0.109526, -0.222928, 0.274551, -0.054344, -0.050076, -0.139315, 0.237803, 0.174901, 0.300263, -0.003827, 0.222558, -0.111873, 0.009261, 0.324098, -0.1338, 0.241577, 0.319872, 0.029774, -0.206615, 0.10863, -0.10605, 0.19976, 0.158664], [0.000192, 0.334658, 0.265172, 0.171809, 0.330379, 0.142137, -0.149455, 0.042665, -0.234543, 0.535934, -0.001115, -0.100444, -0.004036, 0.049055, -0.210754, -0.11105, 0.08549, 0.319627, 0.016094, -0.253372, 0.205472, 0.109421, -0.015525, 0.196675, -0.001678, 0.042977, 0.26502, 0.282349, -0.224825, -0.304888, -0.329848, 0.00662], [-0.499627, -0.119108, 0.127551, -0.164143, 0.107839, 0.054356, 0.152941, 0.100169, 0.054863, 0.089925, -0.121659, 0.063607, 0.484774, -0.273628, 0.135119, -0.05337, -0.052627, 0.599112, 0.130208, 0.006436, -0.263923, 0.020988, -0.017579, -0.112287, -0.716247, -0.069042, 0.097564, 0.436437, 0.016953, 0.124096, 0.157659, 0.052212], [0.013825, -0.250362, -0.019028, -0.078264, 0.370495, -0.218759, 0.030124, -0.001401, -0.006636, 0.15979, -0.001575, 0.032762, 0.007824, 0.237712, -0.050158, -0.068646, 0.227387, 0.233884, 0.182086, 0.099467, 0.157328, -0.125512, -0.082266, -0.027957, -0.146087, 0.115777, -0.311449, 0.054341, -0.137805, -0.020543, 0.329304, -0.356427], [-0.330586, -0.69569, -0.310808, 0.622811, 0.257328, -0.304728, 0.260837, -0.095289, 0.075523, -0.654826, -0.628017, 0.786608, -0.323401, -0.00209, -0.196628, 0.425474, 0.031432, 0.33189, -0.397387, 0.332116, 0.016289, 0.16029, -0.225047, 0.396944, -0.360602, 0.077913, 0.029114, 0.20861, 0.243386, 1.155258, -0.167677, 0.437229], [0.012561, -0.02264, 0.192884, -0.048162, -0.138079, -0.062756, 0.027356, -0.078823, 0.044559, -0.028469, -0.088423, 0.125152, -0.388669, -0.131078, 0.090791, 0.053913, 0.075317, 0.084514, -0.128746, -0.058532, 0.215933, 0.067407, -0.038793, -0.088024, 0.257034, 0.112354, 0.250029, -0.166336, -0.03006, 0.290378, -0.02062, -0.257743], [-0.245816, 0.017608, -0.609828, -0.321535, -0.594186, -0.60884, 0.48884, -0.593136, -0.489634, 0.223724, 0.016361, 0.209197, -0.20875, -0.531847, -0.459911, 0.500473, -0.543365, -0.094138, 0.250466, 0.249824, 0.110754, -0.523117, 0.500593, 0.017029, -0.03782, 0.209925, -0.808018, 1.028506, 0.395661, 0.030574, 0.270837, 0.434902], [0.263084, 0.131446, 0.139007, 0.05843, -0.302933, 0.068078, 0.168481, -0.598553, 0.003825, -0.576409, -0.046204, 0.043396, -0.209585, -0.440762, 0.124081, 0.306258, -0.789853, 0.168164, -0.028061, -0.679271, -0.081958, 0.420998, -0.580633, 0.000634, 0.133472, 0.07571, 0.007597, 0.097897, -0.161263, 0.168431, 0.254743, -0.165557], [-0.075055, 0.052692, 0.162172, 0.096054, 0.106524, 0.042039, 0.023403, -0.065254, 0.080293, 0.012591, -0.090603, -0.101639, 0.107044, 0.095511, -0.024341, -0.073761, -0.032306, -0.022463, -0.085643, 0.069062, 0.028568, 0.241063, 0.009818, -0.130612, -0.130221, 0.044295, 0.000426, 0.134134, 0.03182, -0.107021, -0.338486, 0.072078], [-0.244343, -0.024328, -0.52687, -0.414594, -0.517841, -0.21209, 0.341545, -0.197471, -0.669954, 0.000406, 0.219969, -0.132852, 0.098258, -0.470526, 0.101868, 0.190779, 0.054612, 0.137146, 0.657835, 0.171414, 0.013619, -0.183018, 0.138397, 0.051787, 0.458986, -0.10562, -0.231879, -0.139067, -0.062, -0.168454, 0.279597, 0.300276], [-0.17663, -0.238656, 0.197878, -0.13689, 0.014033, 0.11002, -0.246895, -0.12002, 0.004516, -0.041193, 0.176037, -0.25264, -0.196636, 0.166735, -0.008482, 0.227106, 0.04912, -0.005743, 0.137158, 0.02443, 0.147193, -0.020373, -0.140737, 0.266061, -0.243536, 0.026253, 0.015565, -0.207819, -0.063042, -0.140904, -0.311693, -0.188562], [-0.264299, -0.164233, -0.288073, -0.084604, 0.241027, 0.089939, -0.199353, -0.060877, 0.18923, -0.108485, 0.171664, -0.38137, -0.262244, -0.267518, 0.043262, 0.022709, 0.111131, 0.129667, -0.056772, -0.039347, -0.055447, -0.021056, 0.026907, -0.061836, 0.017492, 0.14653, -0.053284, 0.042175, -0.023171, -0.292748, -0.041273, -0.490937], [0.053423, -0.168493, -0.007093, -0.74357, -0.373087, -0.285105, 0.357806, -0.648586, 0.293502, -0.114554, 0.121988, -0.340183, 0.208448, -0.427509, -0.544521, -0.513086, 0.081128, -0.214402, -0.430635, 0.044322, -0.144415, 0.179659, -0.108284, 0.346582, 0.131974, -0.249861, -0.29107, -0.212177, 0.075327, 0.268209, -0.245074, 0.34787], [-0.104563, 0.105019, 0.044832, 0.324376, 0.044461, 0.114826, -0.115811, 0.062148, -0.13399, 0.065845, 0.014669, 0.095153, 0.267212, -0.140944, 0.109234, -0.090716, 0.033763, 0.034871, 0.07469, 0.251418, -0.564404, -0.07787, 0.021861, -0.120144, -0.120219, 0.407552, 0.210756, -0.165472, -0.191395, -0.27292, 0.581425, -0.225083], [-0.086611, 0.187019, -0.143583, 0.223286, -0.72818, -0.600216, -0.283828, 1.041698, -0.31311, 0.1348, -0.047148, 2.6e-05, 0.162926, 0.123248, 0.633698, 0.724786, -0.156768, -0.149534, 0.657108, -0.226893, 0.141877, -0.126174, -0.277029, -0.28346, 0.302135, -0.266615, 0.520841, -0.204112, -0.015263, -0.344249, 0.063642, 0.035624], [0.21348, 0.098439, 0.393624, -0.658589, 0.12942, -0.361483, -0.319596, -0.239574, -0.014637, 0.324638, -0.022768, -0.24847, -0.363909, 0.184763, -0.04402, 0.360958, -0.871341, -0.224791, 0.049767, 0.138808, -0.291806, 0.348155, 0.266448, -0.195822, 0.302996, -0.078048, -0.006136, 0.081448, 0.643719, -0.294218, -0.102346, 0.070947], [0.027575, -0.088119, -0.065624, 0.122668, -0.049396, -0.112722, -0.226075, 0.026849, 0.032074, 0.006887, 0.085675, -0.129087, 0.021114, 0.112581, 0.028834, -0.014751, -0.114302, 0.014802, -0.011836, -0.013027, -0.349793, 0.006154, 0.113274, 0.010148, 0.013522, 0.118581, -0.029806, 0.125543, 0.014409, -0.099629, 0.165448, -0.080339], [-0.38273, -0.902265, 0.290908, 0.066293, -0.062189, -0.292755, -0.090298, -0.185425, -0.110625, 0.003217, -0.317454, 0.045243, 0.051021, -0.352572, 0.140705, 0.129737, 0.345458, 0.061704, -0.308228, 0.084975, -0.049494, -0.595959, 0.241822, -0.259504, 0.1214, 0.527904, -0.726591, -0.078186, 0.049313, 0.345762, 0.047233, 0.169584], [0.168967, 0.024423, -0.055361, -0.102382, -0.032007, -0.021625, 0.041787, 0.048035, -0.083783, 0.013587, -0.018757, 0.115234, -0.063085, 0.184998, 0.074672, -0.035, 0.1585, -0.201018, 0.039357, -0.017361, 0.081517, -0.165226, 0.213405, 0.028643, -0.139204, 0.075855, -0.138205, -0.17496, -0.080946, -0.112622, 0.018732, -0.147173], [0.002638, -0.051117, 0.071015, -0.238943, 0.011195, 0.037016, -0.058293, 0.051393, 0.018005, -0.016983, 0.1275, 0.846202, -0.021169, 0.039996, -0.536029, 0.08579, -0.095376, -0.064175, 0.171334, -0.125788, -0.198834, 0.217446, -0.131492, -0.739765, -0.014677, -0.060614, 0.013755, 0.01375, -0.005561, 0.010506, -0.010354, 0.380033]], [-0.076851, -0.3202, -0.052532, -0.061751, -0.091892, -0.1391, 0.001288, -0.109363, -0.872272, -0.43713, 0.674744, -0.221472, -0.038087, -0.285853, -0.011812, 0.227513, 0.226172, -0.217668, -0.75577, -0.706568, -0.320094, -0.255348, -0.050356, -0.140845, -0.237027, -0.569651, -0.048226, 0.071566, -0.156, -0.028402, 0.05375, -0.092506]),
    ([[0.030733, -0.161501, 0.189318, -0.019626, 0.024331, -0.212023, 0.187958, -0.242039, 0.025778, 0.041608, 0.174665, 0.338322, 0.342381, -0.114679, -0.095973, -0.216048], [-0.443943, -0.080832, -0.569555, 0.176453, -0.153935, -0.689125, 0.129274, -0.247111, 0.043405, -0.046343, -0.200157, 0.047808, 0.154091, -0.256085, -0.043364, 0.212163], [0.191603, 0.039572, 0.102649, -0.074802, -0.072014, 0.308154, 0.322344, 0.316775, 0.391497, 0.051946, 0.400608, 0.025018, 0.098928, 0.219042, -0.066735, 0.050617], [-0.120938, -0.233692, 0.072433, 0.00625, -0.111305, 0.210997, -0.158657, -0.010867, 0.06172, 0.042411, 0.136799, 0.291891, -0.576429, -0.077735, 0.377114, 0.307216], [-0.434239, 0.390646, -0.530729, -0.035819, -0.118176, -0.165199, -0.604074, -0.127598, -0.116914, -0.132649, 0.023859, -0.590514, -0.523929, -0.166796, -0.067349, 0.149557], [0.044891, -0.150522, 0.487652, 0.036326, 0.272578, 0.317794, 0.125403, 0.044514, 0.020759, 0.072322, 0.165706, 0.159308, -0.177432, 0.066164, -0.324495, -0.496984], [0.126168, 0.144924, 0.177064, 0.091118, 0.093429, 0.143254, 0.317293, 0.176756, 0.146394, 0.214082, -0.084704, 0.121973, -0.602494, -0.487748, 0.174565, -0.004022], [0.200859, -0.036713, -0.324702, -0.118166, 0.049477, -0.451697, 0.472046, -0.095023, 0.34843, 0.210252, 0.446831, 0.114026, -0.008604, 0.103999, 0.130709, 0.201464], [0.183533, -0.09082, -0.518523, -0.008791, -0.525699, -0.510662, -0.046103, -0.262565, -0.180984, 0.48735, 0.06058, 0.428534, 0.176018, -0.111423, -0.398917, -0.027573], [0.125143, -0.514786, 0.305959, 0.198842, 0.166361, 0.438431, 0.033217, 0.041707, 0.233685, 0.716361, -0.457069, -0.057658, -0.43992, 0.068007, -0.525379, -0.342604], [-0.062074, 0.351175, -0.243187, -0.230994, 0.014587, -0.223742, 0.015352, -0.12466, -0.085003, 0.09671, 0.012103, 0.024435, -0.100468, -0.288059, -0.168811, -0.293797], [-0.093638, 0.072542, -0.28183, -0.137433, 0.057408, -0.211615, -0.162964, 0.050998, -0.295846, -0.025272, 0.192996, -0.246017, -0.317679, -1.02885, 0.01943, 0.075516], [-0.256929, -0.448358, 0.105072, -0.024581, 0.190575, -0.096557, 0.015917, 0.016484, 0.175754, -0.64909, -0.340827, -0.038096, 0.288269, 0.059468, 0.44334, 0.051188], [0.165591, 0.053402, -0.350067, -0.303904, 0.295562, -0.026125, -0.130658, -0.141596, 0.092961, 0.212294, 0.191292, 0.157071, 0.405371, -0.662964, 0.308881, -1.2055], [-0.268456, -0.019149, 0.027956, 0.147156, 0.183066, 0.137229, -0.085604, 0.218996, -0.220073, -0.518875, -0.055724, 0.097858, 0.004846, 0.372107, -0.017897, 0.241436], [-0.114744, 0.294579, -0.200828, -0.106536, -0.171355, -0.086278, -0.106011, 0.023397, -0.462413, -0.493261, 0.247334, 0.025624, -0.164814, 0.01641, 0.142075, 0.196002], [0.341958, -0.169024, -0.142703, -0.009222, -0.116072, 0.087864, 0.076244, 0.135722, -0.671102, -0.588491, 0.067084, -0.707576, 0.391467, 0.104683, -0.118833, -1.310535], [0.165451, -0.706429, 0.248832, 0.229202, -0.054061, 0.291652, -0.108842, 0.312887, 0.2002, 0.39779, -0.465104, -0.746911, -0.311901, 0.169682, -0.030687, 0.01643], [0.378402, 0.337334, 0.31746, -0.116726, -0.135502, 0.119334, -0.602721, -0.436748, -0.179108, -0.148259, 0.309713, -0.333852, 0.346799, 0.02738, -0.485724, 0.407742], [0.147529, 0.483445, 0.051344, 0.164802, 0.038798, -0.06406, -0.138541, 0.126916, 0.164624, 0.127547, -0.556318, -0.357619, 0.153477, 0.361638, -0.850123, -0.870227], [-0.250166, 0.048769, 0.173177, -0.345488, -0.152426, -0.461413, -0.112819, -0.219765, -0.067719, 0.255043, -0.208631, -0.046672, 0.244893, 0.073982, -0.071593, 0.042265], [0.387332, -0.364732, -0.143643, 0.167753, 0.383435, -0.214226, -0.298545, -0.236514, -0.004978, 0.06275, -0.050244, 0.077792, -0.46877, -0.26423, -0.464383, 0.011756], [-0.074299, 0.130857, 0.118468, 0.183257, 0.170337, 0.078303, 0.059107, 0.085497, -0.202908, 0.134752, 0.049007, 0.630064, 0.013052, -0.306406, -0.049647, -0.26087], [0.051769, 0.047675, -0.155278, -0.206177, -0.784463, -0.128762, -0.025949, -0.032092, 0.232828, 0.255064, 0.324845, -0.077637, 0.010685, -0.216395, 0.109071, -0.220796], [-0.101511, 0.230135, 0.173616, -0.032233, -0.331099, 0.2933, 0.033641, 0.104961, 0.410071, -0.014528, 0.171907, 0.040882, 0.207385, 0.151978, 0.004161, -0.418287], [0.126215, 0.051654, 0.035176, -0.421939, -0.192916, -0.172429, -0.62282, -0.219312, -0.466893, 0.537413, 0.189147, 0.022686, -0.318892, -0.160101, -0.165722, -0.001212], [-0.165315, 0.363472, -0.14601, 0.069912, -0.183596, -0.546916, -0.155729, -0.382603, -0.205695, -0.211499, -0.006807, -0.35087, -0.16966, -0.329071, 0.083358, -0.338402], [0.026393, 0.001362, -0.271452, -0.178655, 0.049808, 0.020113, 0.335004, -0.046437, 0.253419, -0.219912, -0.065091, -0.42214, -0.375829, -0.323787, -0.002365, 0.18435], [0.299741, -0.490124, 0.123893, 0.193206, 0.062399, 0.417122, 0.134286, 0.119415, 0.404867, 0.088912, 0.064031, 0.653035, 0.118441, 0.094792, 0.165317, -0.029951], [-0.042765, -0.317757, -0.153358, 0.369839, 0.279617, -0.108771, 0.05255, 0.327606, -0.094404, 0.310769, 0.224164, -0.34523, -0.181907, -0.218083, -0.109321, -0.277676], [-0.088281, 0.174686, 0.240136, -0.076975, -0.021679, -0.009785, -0.003617, 0.065314, -0.231152, -0.507668, 0.181313, -0.707491, 0.096095, 0.285246, -0.092756, 0.019853], [0.299539, 0.027341, -0.121581, 0.081799, 0.02206, 0.00756, 0.256407, 0.285107, -0.185979, -0.327488, -0.081372, -0.102014, -0.262928, -0.266402, -0.378391, -0.43797]], [0.282679, 0.211005, 0.125216, 0.276136, -0.049373, -0.133051, 0.289761, 0.333305, -0.010996, -0.342619, 0.152775, -0.108044, -0.709434, -0.188807, 0.322213, -0.353687]),
    ([[0.321501], [0.371814], [0.452617], [0.343787], [0.322364], [0.578703], [-0.311471], [0.35999], [-0.621716], [-0.69642], [-0.300668], [-0.626301], [1.482639], [0.849598], [0.447848], [0.999554]], [-0.281355]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 1.8150514678811415e-19, 0.0, 0.0, 0.0, -9.075257339405708e-20, 0.0, 0.0, 0.0, 1.23310226133381, 0.0, 7.462731657396113, 19.56953274176347], [5.53821291192578, 14.946806225323003, 26.014301364590626, 33.66922113068142, 44.17611268580645, 1.0558626719575712, 0.919869864907879, 0.8725961764501532, 0.7685091479780537, 0.6759626415937752, 988.2449633620847, 1811.0274726344712, 3380.3988044149432, 0.5776143913640635, 0.3561757459778461, 12.950469201809328, 27.39798970393506], [0.18853986897446542, 0.2937399245041398, 0.018047934228479216, -0.1643386174592359, -0.04410700601748327, 0.7335853255644459, -3.4410924081382075, 6.204829841015924, -3.8458471122896896, 2.5214944434775717, -0.1351561589482947, -0.06584420838215777, -0.20390047970224467, -0.0037289615940253864, 0.38046675727018436, 0.0005558869652611978, 0.0002792148092801856], 0.0)
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 1.8165929541207803e-19, 0.0, 0.0, 0.0, -9.082964770603902e-20, 0.0, 0.0, 0.0, 1.2332786587933198, 0.0, 7.466528955002561, 19.58090127503797], [5.540083088437315, 14.952865954265596, 26.024495688550573, 33.68268786260075, 44.19157755072277, 1.0561016096905944, 0.9200796121454458, 0.8727935155335413, 0.7686765477435015, 0.6761117957842069, 987.37957371303, 1810.5555136276132, 3377.5876417715444, 0.5778027820882502, 0.35632696028178795, 12.955274817946957, 27.406418352393846], [1.7072639650356551, 0.90202529589946, -0.25482662634616554, -1.02422950794453, -0.3018395669828723, 0.927489177174886, -2.58870486180548, 5.420564846193685, -2.6507958673310714, 2.4532030933951985, -0.8866983084808405, -0.3280069603995014, -1.1103906803452683, -0.0008746350997088204, -0.569169306831935, 0.00013033994696102662, 6.541173017788018e-05], 0.0)
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.233793, -0.0, 0.0, 5.907374, 5.907519, 0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, -0.000361, 4.035521], [1.055403, 0.918246, 0.870864, 0.7665, 0.674154, 2.379774, 15.55764, 0.711525, 13.418896, 0.710817, 0.650826, 0.543624, 1.654265, 1.654359, 0.999996, 4.403696, 1.067322, 0.354443, 130.051437, 328.43396, 986.32782, 1.340595], [([[0.153197, 0.024577, -0.077695, -0.042458, 0.183153, 0.033375, -0.183001, 0.431035, 0.174539, -0.771803, 0.000581, -0.223927, 0.231905, -0.158765, 0.0637, -0.053104, -0.064703, -0.242732, 0.155888, 0.170749, -0.00178, 0.198982, 0.004137, -0.082656, 0.007355, -0.141179, 0.157015, 0.052691, 0.755914, 0.039558, 0.035489, 0.065459], [0.102675, 0.261775, -0.200635, 0.033781, -0.407756, 0.01466, 0.037306, -0.326957, -0.046157, -0.009883, 0.150295, 0.013609, -0.092343, -0.028485, 0.11439, -0.11951, 0.123484, -0.03903, 0.012906, 0.268499, -0.15435, -0.146797, 0.259013, -0.008512, -0.069906, -0.101711, -0.093966, 0.060321, 0.194702, 0.317467, -0.009227, 0.001897], [-0.317064, 0.032628, 0.051791, 0.042142, -0.102303, -0.225224, 0.275575, -0.159115, 0.234967, 0.20622, -0.007119, 0.177643, -0.153792, -0.133794, -0.061215, 0.311491, 0.147983, 0.394979, -0.105948, 0.096865, -0.14736, 0.124323, 0.328405, -0.084553, 0.173758, 0.329751, -0.005517, -0.203922, 0.174947, 0.031779, 0.234435, 0.078569], [0.178329, 0.420396, 0.251227, 0.351366, 0.080308, 0.101474, -0.076216, -0.08672, -0.028399, 0.42312, 0.039276, -0.338184, -0.099153, 0.164248, -0.351092, 0.037758, -0.113815, 0.207469, -0.003079, -0.285505, 0.212557, 0.399068, -0.127889, 0.425473, 0.45332, 0.107415, 0.311214, -0.066407, -0.324274, -0.434712, -0.38568, -0.063299], [-0.498998, 0.076952, -0.282936, -0.759055, 0.857676, 0.119873, -0.002694, -0.102762, -0.011567, -0.270931, -0.035063, -0.133083, 0.492353, -0.189855, 0.209607, -0.372128, -0.041207, 0.600679, 0.167561, 0.177195, -0.505798, -0.267498, 0.076603, -0.078434, -0.206291, -0.050616, 0.141974, 0.232432, -0.1775, 0.342611, 0.288847, 0.181954], [-0.115628, 0.080309, -0.266723, -0.038177, -0.128796, -0.330457, 0.149417, -0.293241, -0.141403, 0.302404, -0.030144, -0.156012, 0.115509, -0.087978, 0.187603, 0.044658, 0.278071, 0.255107, 0.08466, 0.068216, 0.072338, -0.177941, -0.009384, -0.039865, -0.280117, 0.133105, -0.152917, 0.177425, -0.045837, -0.23604, 0.271248, -0.180588], [-0.543355, -0.219383, 0.234997, -0.406037, 0.033918, 0.713737, 0.12966, -0.284399, 0.093113, -0.393535, -0.529214, 0.624837, -0.015551, 0.051565, -0.173152, 0.656445, -1.10299, 0.477219, -0.251365, 0.108242, -0.27063, -0.201846, -0.024273, 0.336189, 0.115855, 0.059476, -0.203152, 0.256858, 0.443244, 0.112228, -0.198333, -0.124929], [-0.087285, 0.108927, 0.182654, 0.135994, -0.107092, -0.060755, 0.150829, -0.096684, 0.081041, 0.0346, -0.087234, 0.357508, -0.201449, -0.061775, 0.032085, 0.394495, -0.067978, 0.070671, 0.071872, 0.004495, -0.034252, 0.082396, 0.10809, -0.132996, -0.093927, 0.10886, -0.103052, -0.246819, -0.190068, 0.048803, -0.018236, -0.032448], [0.305367, -0.37374, -0.350215, 0.126293, 0.331703, 0.099485, -0.448579, -0.640842, -0.429006, -0.384398, -0.115183, 0.682187, -0.307028, -0.079762, -0.45164, 0.334049, -0.364011, 0.098552, -0.143026, 0.180238, -0.142695, -0.487428, 0.035966, -0.173191, -0.252283, -0.095691, -0.498344, -0.251611, -0.075478, -0.141431, -0.11466, 0.367877], [0.204844, 0.000594, 0.164573, 0.31285, -0.053859, -0.026354, 0.451541, -0.6788, -0.210448, -0.366521, -0.018118, 0.026736, 0.293086, -0.374867, 0.270365, 0.250286, -0.427419, 0.154856, -0.012471, -0.888707, -0.110576, 0.329946, -0.146161, -0.089999, -0.322994, -0.069511, 0.127691, 0.18332, -0.399623, 0.346717, 0.094764, -0.576847], [-0.136013, -0.03249, 0.065195, 0.040428, -0.257706, 0.017679, -0.032692, -0.104728, 0.068466, -0.002911, -0.120766, -0.113272, 0.116348, 0.040663, -0.0431, -0.266716, -0.133378, 0.022501, 0.195111, 0.012384, 0.030652, 0.188896, -0.016849, -0.105367, -0.134462, -0.110977, 0.141315, -0.138009, -0.033799, -0.025462, -0.319363, -0.083885], [-0.077819, -0.255754, -0.484369, -0.074363, 0.069263, -0.230806, 0.066377, -0.149523, -0.575072, -0.032555, -0.147522, 0.214393, -0.355095, -0.326098, 0.281635, 0.078999, 0.139504, 0.131196, 0.487924, 0.069352, -0.021441, -0.027699, 0.222513, 0.019203, -0.030299, -0.322494, -0.392409, 0.02861, 0.221188, -0.112868, -0.17403, 0.555147], [-0.00568, -0.239815, -0.094346, -0.027472, -0.000629, -0.152971, -0.077725, -0.05692, 0.068214, -0.04444, 0.254207, 0.106743, 0.224424, 0.132859, 0.022814, -0.064122, 0.148428, -0.109747, 0.367651, -0.210544, 0.133706, 0.129336, -0.122166, 0.106592, -0.181046, -0.187227, -0.064836, -0.11912, -0.182072, -0.112651, -0.332116, -0.082096], [-0.182069, -0.333674, -0.143594, -0.165184, -0.134924, 0.024485, -0.158498, -0.033223, 0.683683, -0.068515, 0.130087, -0.008253, -0.043862, 0.153615, 0.077127, -0.194793, 0.267795, 0.073457, 0.179671, -0.075927, 0.096428, 0.054619, -0.06405, 0.061599, -0.12383, -0.047951, 0.081937, 0.033938, 0.043788, -0.222036, -0.043074, -0.333533], [0.02216, -0.115529, -0.284619, -0.172733, -0.082809, -0.469192, 0.730418, -0.035583, -0.375587, -0.065769, -0.093035, -0.161026, 0.102211, -0.568075, -0.341648, -0.381354, 0.061511, -0.117499, 0.110424, 0.090008, 0.246924, 0.029787, 0.045749, 0.064839, -0.284718, -0.294751, 0.171735, -0.349573, -0.024616, 0.27965, -0.243442, 0.225781], [-0.007703, 0.156276, -0.103365, 0.104321, -0.027035, 0.01244, -0.015133, -0.168764, 0.047076, 0.074926, -0.182144, -0.106171, 0.250255, -0.092611, 0.091646, 0.167897, 0.091391, -0.030864, -0.226257, 0.130919, -0.184238, -0.031055, 0.162714, 0.070001, -0.337809, 0.471379, 0.234163, -0.110319, 0.054628, -0.25829, 0.306209, 0.047241], [0.152187, -0.091287, -0.029986, 0.14916, 0.147238, 0.229532, -0.127381, 0.48653, -0.598024, 0.184357, 0.364873, -0.707481, 0.090455, 0.255808, 0.721639, 0.717893, -0.139369, 0.029626, 0.64696, -0.251894, -0.227181, -0.106405, -0.231776, -0.209393, 0.702237, -0.280244, 0.468131, 0.052442, 0.046664, -0.078807, 0.324028, 0.099381], [0.203734, 0.134745, 0.244684, -0.325014, 0.069284, -0.386981, -0.478708, 0.211177, 0.026804, 0.031014, -0.490083, -0.09791, -0.106429, -0.196236, 0.040481, 0.305194, -0.219009, 0.147975, 0.180713, -0.191203, 0.187528, 0.171948, 0.417616, -0.038796, 0.122157, -0.219644, 0.026542, -0.129487, -0.056112, 0.029308, -0.308132, 0.314089], [-0.050477, 0.003732, -0.059611, 0.03619, -0.012626, 0.12418, -0.148018, 0.024063, 0.04794, -0.030432, -0.113505, 0.047458, -0.077885, 0.124031, 0.013072, -0.134328, 0.005535, 0.028046, -0.013133, -0.053896, -0.274842, 0.066894, 0.011685, 0.03186, -0.032215, 0.012656, -0.037328, 0.036254, 0.03386, 0.074229, -0.018404, 0.053641], [-0.093454, -0.369741, -0.197401, -0.074991, 0.129859, -0.026204, -0.004813, 0.057313, -0.081953, -0.026339, 0.027884, 0.335735, 0.064505, -0.25104, 0.033606, 0.184907, 0.237217, 0.053006, -0.316062, 0.247319, -0.579651, -0.629592, -0.043527, -0.332268, 0.113528, 0.293557, -0.873066, -0.136805, -0.047142, 0.408069, 0.020694, -0.024922], [0.143836, 0.131464, -0.042272, 0.142351, -0.162709, 0.182132, 0.141804, -0.027122, 0.13933, 0.055706, 0.176013, 0.195401, -0.048265, -0.001212, -0.048686, 0.059553, -0.016503, -0.209521, 0.032892, 0.209076, 0.051194, -0.134646, 0.039565, 0.050484, 0.090743, 0.122122, -0.391513, 0.053354, -0.075246, -0.341358, 0.068839, -0.208927], [0.015069, 0.219462, -0.124061, -0.049574, -0.095289, 0.000879, 0.075429, -0.053082, -0.259569, 0.030274, 0.307549, 0.222152, 0.123774, 0.354428, -0.299833, 0.159181, -0.378777, 0.068473, -0.018109, -0.060119, -0.122198, 0.086814, 0.047089, -0.695561, -0.014071, 0.409534, 0.176322, 0.737021, 0.034895, 0.068202, -0.146577, 0.509206]], [-0.440461, -0.550555, 0.047347, -0.103002, -0.308939, -0.190948, -0.609648, -0.480606, -0.246894, -0.305806, 0.500369, -0.34854, 0.201881, -0.003285, 0.079054, -0.501237, 0.55604, -0.533208, -0.397538, -0.179502, 0.013007, -0.282805, -0.041289, -0.239612, -0.109751, -0.76172, -0.122256, -0.028098, -0.299941, -0.070311, -0.00904, 0.273804]), ([[-0.192731, -0.288018, -0.200865, 0.072698, 0.213151, -0.156488, 0.075512, -0.130264, 0.002181, 0.110345, 0.123366, 0.095383, -0.096943, -0.026117, -0.966604, 0.218544], [-0.043603, 0.098592, -0.494712, 0.028447, -0.3704, -0.140534, -0.055987, 0.049449, -0.08385, -0.275402, 0.075227, 0.047696, -0.21752, -0.42392, 0.030811, 0.149693], [0.280112, 0.138083, -0.037187, -0.398387, -0.193154, 0.110175, 0.331463, 0.167955, 0.106999, -0.06755, 0.229824, 0.122156, -0.184026, 0.118737, -0.178001, -0.326951], [-0.682274, -0.194102, -0.054706, 0.153123, -0.20905, 0.308504, -0.157819, -0.174175, 0.229561, 0.207746, 0.084891, 0.247294, -0.08187, -0.213452, 0.132633, 0.226378], [-0.204608, -0.716649, 0.210195, 0.182897, 0.062946, 0.263995, -0.565567, -0.328864, 0.43938, 0.424401, -0.639509, -0.030894, -0.17291, -0.055461, 0.257198, 0.017358], [0.02747, -0.26532, 0.083184, -0.021634, 0.194452, 0.337494, 0.267383, -0.008126, 0.079927, 0.027627, 0.198969, -0.301667, -0.049173, -0.165855, 0.428718, -0.305937], [0.214745, 0.01722, 0.400657, 0.20604, 0.178974, 0.120708, 0.399973, 0.256826, -0.110194, 0.208908, 0.463659, 0.123066, -0.123658, -0.706522, 0.109868, -0.333158], [-0.043216, -0.298761, -0.241977, 0.000379, -0.037039, 0.330773, 0.141975, -0.073281, 0.191632, 0.240016, 0.121261, -0.221884, -0.135821, 0.186725, 0.05971, 0.388994], [0.038785, 0.365735, -0.127909, -0.274377, -0.250444, -0.014196, 0.269105, -0.083485, -0.18067, 0.185738, 0.0099, -0.303506, 0.035608, -0.029318, -0.167551, -0.655726], [0.150303, -0.512964, 0.24344, 0.291419, 0.260971, 0.750589, 0.238169, 0.312686, 0.378075, 0.28617, 0.269576, 0.362095, -0.193511, -0.23913, 0.297755, 0.082758], [0.082541, 0.407266, -0.124581, -0.292287, 0.040883, -0.217731, 0.186972, -0.069796, -0.136538, 0.055649, 0.18407, -0.104767, -0.240751, 0.076392, -0.012183, -0.121754], [-0.205055, -0.314154, -0.061091, 0.027886, -0.075584, -0.077565, -0.756002, -0.056962, -0.076444, -0.052091, -0.276592, -0.263365, 0.098261, -0.243379, -0.302331, 0.417473], [0.105119, -0.38071, 0.123285, 0.060056, 0.268194, 0.025469, 0.065901, 0.164146, 0.159219, -0.363336, 0.32325, 0.182163, -0.020825, 0.128674, 0.199241, 0.371211], [-0.014129, -0.084275, -0.320824, 0.140749, 0.302865, -0.753426, 0.05102, -0.053208, 0.091424, 0.254485, 0.225893, -0.006908, 0.070642, -0.256229, 0.147942, -0.151931], [-0.269417, 0.381439, 0.044604, 0.071832, -0.412778, 0.154437, -0.076454, -0.099734, -0.51474, -0.261366, 0.055507, 0.083404, -0.044427, 0.374791, 0.192959, 0.071437], [0.04243, 0.300393, -0.060563, 0.026295, 0.032241, 0.108226, 0.016795, 0.083314, -1.642045, 0.216534, 0.091496, 0.395927, -0.055879, 0.221914, -0.144076, 0.0769], [0.362132, -0.020314, -0.410901, -0.170759, -0.058096, 0.214896, 0.281623, 0.206113, -0.291179, -0.387527, 0.296842, -0.247347, -0.019998, -0.169915, 0.297052, -0.449157], [0.296857, -0.73784, 0.277109, 0.090811, 0.028836, 0.221621, -0.286846, 0.175173, 0.08582, 0.308482, -0.491621, 0.474772, -0.009042, 0.150099, 0.183827, 0.129224], [0.138022, -0.071107, -0.181828, 0.180948, 0.089914, -0.031428, -0.34708, -0.236684, 0.118669, -0.025634, 0.248614, -0.482836, 0.061938, -1.156468, -0.044362, 0.329652], [0.153287, -0.43053, 0.16304, 0.220797, 0.239029, 0.095872, -0.01769, 0.221716, -0.081417, -0.355243, -0.215938, 0.034151, -0.21673, 0.231422, -0.001894, -0.698763], [0.000635, -0.139851, -0.140209, 0.237282, -0.053305, -0.678829, 0.016271, -0.11507, 0.034391, 0.207524, -0.169905, 0.108093, -0.09384, -0.071846, -0.212002, 0.342114], [0.460428, -0.038617, 0.047083, -0.267224, 0.169117, 0.067223, 0.047269, -0.296962, -0.180597, 0.253426, 0.140382, 0.127801, 0.008057, -0.095194, -0.279302, -0.321146], [-0.013559, 0.204256, 0.120637, 0.212891, 0.00668, 0.031722, -0.064985, 0.169374, -0.188291, 0.127604, 0.099111, 0.058696, -0.148115, -0.115104, 0.144294, 0.087274], [-0.303619, -0.065699, 0.033278, -0.104831, -0.532146, -0.26679, -0.096836, -0.119843, 0.357261, 0.046895, 0.087526, 0.048784, -0.13137, -0.090438, 0.166353, -0.264975], [0.134858, 0.081496, -0.241535, -0.449684, -0.196284, -0.268981, -0.24234, -0.075273, 0.115861, -0.210722, -0.103645, 0.269919, 0.070881, -0.492155, -0.109457, -0.065274], [-0.069264, 0.230447, -0.437695, -0.284002, -0.311095, -0.168577, 0.063521, -0.31337, 0.062647, 0.532106, 0.027757, -0.357786, -0.16509, -0.106249, 0.034166, 0.173159], [-0.357555, 0.256647, -0.222813, 0.031553, -0.130492, -0.896248, -0.071241, -0.278268, 0.003305, -0.240929, -0.095362, -0.313431, -0.013029, -0.225357, 0.091514, -0.024662], [0.22442, -0.036946, -0.075867, -0.035841, 0.094476, 0.162429, 0.064908, -0.120239, 0.033286, -0.327486, -0.173266, -0.114981, -0.234717, -0.046766, -0.201719, 0.035626], [0.128597, -0.369404, 0.120354, 0.163819, 0.187095, 0.249917, 0.107048, 0.036289, 0.65105, -0.037763, 0.132972, 0.223558, 0.056078, -0.024743, 0.306706, 0.141797], [0.056951, -0.1237, 0.227699, 0.169564, 0.191599, -0.059657, -0.274399, 0.320225, -0.024891, 0.414484, 0.071095, -0.366572, -0.145792, -0.012217, -0.386224, 0.014802], [-0.20415, 0.035317, 0.242766, 0.103443, -0.071659, -0.007667, 0.253338, 0.167846, -0.353052, -0.350555, 0.008275, -0.041264, 0.035146, 0.260423, 0.170768, -0.100182], [0.121151, 0.094633, -0.056589, 0.075555, 0.177768, -0.197945, 0.361886, 0.243976, -0.267425, -0.161537, 0.209876, -0.12215, -0.246621, -0.128847, -0.197605, -0.730285]], [0.205713, 0.292396, 0.225158, 0.132544, -0.020543, -0.267123, 0.218774, 0.252529, 0.042492, -0.575022, 0.272832, 0.176409, -0.274537, -0.135948, 0.292249, -0.951168]), ([[0.532692], [0.922244], [0.675769], [0.67014], [0.544294], [1.382717], [-0.595589], [0.586845], [-1.35484], [-1.795383], [-0.527949], [-0.760141], [0.429179], [1.449347], [0.685301], [2.770788]], [-0.533779])])
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
    ([[-0.031685, 0.021227, -0.157469, -0.06404, 0.000452, -0.134694, 0.032433, 0.050157, -0.043047, 0.015412, 0.081953, -0.081309, -0.020971, -0.005637, 0.035934, 0.043799, 0.09602, 0.074511, -0.01373, -0.016966, -0.084658, -0.053895, 0.021423, 0.09112, -0.045765, 0.025197, -0.055564, -0.142186, -0.08867, -0.156614, 0.142236, -0.017799, -0.025828, 0.084504, -0.03319, -0.130043, 0.069381, 0.104968, -0.130095, -0.02809, 0.045513, -0.113662, 0.053594, 0.063966, -0.100343, 0.103357, 0.001415, 0.050665, 0.082025, -0.071881, -0.006505, 0.03643, -0.052353, -0.13748, -0.091721, -0.089884, 0.006565, -0.108645, -0.017069, -0.038219, 0.201643, -0.04131, -0.068862, 0.00776], [0.059849, 0.07083, -0.036867, 0.141176, 0.096293, -0.041642, -0.035489, -0.131343, 0.112762, -0.054128, -0.106542, 0.043089, -0.014241, -0.111303, -0.047797, 0.046812, 0.031874, 0.020947, -0.062139, -0.002657, -0.088731, -0.015794, -0.013137, -0.003787, 0.085368, -0.008955, -0.064647, -0.088665, 0.045962, 0.043833, -0.093295, 0.057834, 0.113005, 0.030936, 0.030391, 0.04341, 0.101732, -0.014768, -0.048828, 0.018655, 0.120964, 0.01666, -0.112534, 0.0202, -0.069837, 0.078476, 0.092596, 0.113315, -0.01583, -0.032188, -0.09777, -0.152462, -0.07171, 0.072808, 0.089801, -0.044097, -0.110824, 0.020823, 0.113314, -0.017272, -0.087613, -0.025975, -0.119892, -0.043157], [0.038589, 0.098334, -0.014048, 0.151084, 0.086038, -0.058531, -0.010893, -0.159124, 0.012141, -0.036349, -0.086021, -0.073116, -0.022865, -0.092697, -0.080722, -0.123963, -0.049937, -0.022751, 0.041296, 0.130315, 0.060237, 0.033433, 0.068355, -0.007009, 0.085412, -0.07007, -0.059271, 0.063453, 0.027393, 0.100299, 0.085113, 0.035683, -0.028076, -0.12316, -0.033684, 0.093487, 0.117797, -0.025545, 0.047625, 0.142769, -0.1099, 0.063581, 0.044343, -0.021153, 0.056281, -0.051476, 0.026153, 0.102083, 0.03772, 0.061058, 0.020653, -0.119136, -0.128511, 0.06232, 0.028856, -0.049008, 0.069721, -0.107439, 0.139694, -0.074515, 0.11104, -0.020098, -0.054265, 0.052562], [0.057184, -0.070864, -0.05352, -0.009446, -0.059952, -0.003285, 0.07383, -0.047064, 0.068797, 0.118015, -0.104543, 0.061104, -0.139544, -0.118127, 0.122719, -0.070249, -0.018501, -0.122596, 0.002969, 0.140659, 0.053968, -0.130258, -0.032255, -0.05566, 0.062328, 0.126102, -0.101818, -0.052021, 0.098412, -0.098457, 0.06184, -0.090808, 0.057221, -0.030443, 0.023292, -0.008811, 0.166374, 0.01833, 0.064983, -0.034503, -0.185612, -0.019683, 0.016624, -0.23117, -0.007045, 0.066143, -0.108038, -0.100081, 0.084948, -0.142628, 0.147157, -0.083108, -0.128191, 0.160005, 0.063957, 0.097899, 0.081005, 0.073279, 0.136473, -0.119739, -0.049422, 0.18165, -0.063801, 0.070856], [-0.044662, 0.022934, 0.104073, 0.134661, -0.13455, 0.03039, 0.062553, -0.155458, 0.075801, -0.165505, -0.102209, -0.084563, 0.043328, 0.210293, -0.01574, -0.132884, -0.065562, 0.179321, 0.018582, 0.14892, 0.054439, -0.056019, -0.003289, -0.054822, 0.022857, -0.099457, -0.078514, 0.060757, 0.004966, -0.313042, 0.016657, 0.10538, 0.174201, 0.095945, 0.085367, 0.076834, -0.023959, 0.127935, -0.017862, -0.016548, -0.20988, -0.078979, -0.075522, -0.23139, -0.099722, 0.016187, 0.054099, -0.060614, -0.120637, -0.059978, 0.069877, 0.045286, 0.091884, 0.042278, -0.093862, -0.04565, 0.197396, 0.17315, 0.169829, -0.131297, 0.087745, -0.007272, 0.052406, 0.12699], [0.051095, -0.028366, -0.048393, 0.137135, -0.009853, 0.120958, -0.00018, 0.063106, 0.012453, -0.01909, 0.006158, 0.058927, -0.064638, 0.121376, -0.099011, 0.099014, 0.066428, -0.104803, 0.066649, -0.084555, -0.012903, -0.04806, 0.13501, 0.065224, -0.074266, 0.079155, 0.148208, 0.071559, -0.056175, 0.006384, 0.009384, 0.092137, 0.143836, 0.048577, 0.004133, 0.079353, 0.16524, -0.072626, 0.086029, -0.006796, -0.034648, 0.050902, -0.114372, -0.085727, -0.087462, 0.0043, 0.058638, 0.059994, 0.138236, -0.033986, 0.077367, -0.125451, -0.036721, -0.042105, -0.062169, -0.012535, -0.106325, 0.006082, -0.105204, 0.109536, 0.020482, 0.156864, -0.006131, -0.027234], [0.066773, 0.054246, -0.115106, 0.086174, -0.184568, -0.180128, 0.167244, 0.03431, 0.235099, -0.056045, -0.038726, -0.038145, -0.169543, 0.303093, -0.193897, -0.103093, -0.001867, 0.019546, 0.044557, 0.162254, -0.028743, -0.053489, -0.020029, -0.095466, -0.015609, 0.070674, -0.054111, -0.173109, 0.188892, -0.026932, 0.11464, 0.075534, 0.030624, 0.067166, -0.201038, 0.046984, -0.054548, 0.124389, 0.144125, 0.034729, 0.043451, 0.048749, 0.02461, -0.146757, 0.100706, -0.263584, 0.114075, 0.126867, -0.22103, -0.027086, -0.029948, -0.052871, -0.02295, 0.032356, -0.181802, -0.029113, 0.014992, 0.049748, 0.09877, -0.09292, 0.034369, -0.007608, -0.044502, 0.080764], [-0.029332, -0.105672, -0.105507, 0.04409, 0.072233, 0.030286, -0.12847, -0.080798, -0.004696, 0.00506, 0.070036, 0.075683, -0.033916, 0.071532, -0.074469, -0.059568, 0.091529, -0.009065, 0.112302, 0.114251, 0.076549, -0.021181, -0.102481, -0.134628, 0.058813, 0.117827, 0.094589, 0.050376, 0.074208, -0.005481, -0.117813, -0.147945, 0.038968, 0.080223, -0.045841, -0.10195, -0.014665, -0.104477, -0.059109, -0.080576, -0.00366, -0.103425, 0.000626, 0.05361, -0.014798, 0.071205, 0.026788, 0.030348, 0.004502, 0.093257, 0.011857, -0.022169, -0.08562, 0.154866, 0.150932, -0.00348, -0.019239, 0.149301, 0.036199, 0.021079, 0.041256, 0.018883, -0.129109, 0.03259], [0.1185, 0.141755, 0.044304, 0.325857, -0.005665, -0.152363, 0.105203, -0.135099, 0.217249, -0.08627, 0.038555, -0.219321, -0.161329, 0.06005, -0.135624, 0.027716, -0.083883, 0.207429, -0.052967, -0.007725, -0.068047, 0.144226, -0.104692, -0.218453, 0.025399, 0.045265, -0.105528, -0.115234, 0.089878, 0.177135, 0.041013, 0.083663, -0.193685, 0.027866, -0.07008, -0.012568, -0.031576, -0.002364, 0.106227, -0.045617, 0.078445, 0.194337, 0.025298, 0.203674, -0.130454, -0.100697, 0.083056, 0.222983, -0.007878, 0.040874, -0.037777, -0.176578, 0.096552, -0.123631, -0.114893, -0.037999, -0.076867, -0.032524, 0.238788, 0.070557, 0.058881, -0.129522, -0.034547, 0.231403], [-0.10434, 0.079894, -0.02077, -0.030928, 0.043059, -0.032013, 0.079921, 0.215408, 0.098468, 0.054513, 0.114255, 0.02548, 0.070066, 0.062576, 0.077202, 0.2727, 0.080026, 0.071176, -0.09735, 0.204537, -0.197011, 0.116813, 0.072256, 0.096388, -0.028297, -0.138025, -0.014603, 0.083659, 0.029398, -0.059589, 0.071897, 0.006758, -0.204207, -0.181453, -0.003359, 0.040691, -0.08321, 0.191727, -0.091504, 0.084751, -0.242624, 0.109278, -0.135191, -0.079264, 0.066911, 0.075852, 0.15751, -0.136919, -0.068781, -0.067398, -0.198659, 0.005617, 0.003417, 0.031574, 0.224379, 0.005018, 0.078418, 0.045212, 0.162667, 0.017868, -0.057919, -0.07131, -0.27206, 0.028967], [0.02821, -0.029638, 0.006478, -0.05979, -0.128643, 0.097038, -0.041501, 0.003577, -0.046744, 0.021363, 0.166063, -0.099936, -0.034163, 0.079551, 0.003761, 0.006532, 0.023321, -0.162354, 0.044311, -0.09772, 0.052316, -0.104764, -0.070826, 0.137227, 0.094865, 0.050181, 0.022863, 0.013058, 0.012721, 0.049078, -0.044087, 0.169949, -0.018688, -0.127389, -0.009999, -0.050397, -0.078723, -0.039862, 0.034366, -0.025483, -0.029026, 0.083694, 0.034033, -0.005078, 0.069512, -0.049201, -0.083536, 0.112656, -0.04908, 0.064304, -0.100057, 0.1314, -0.068902, 0.103528, 0.200103, 0.113428, -0.017043, -0.054541, 0.0678, -0.00733, -0.088295, -0.043348, -0.166139, -0.068998], [0.105417, 0.089162, 0.116684, 0.063358, 0.053807, -0.180593, 0.075021, -0.285453, 0.202223, -0.101038, -0.128347, -0.171785, -0.147646, -0.06441, 0.08286, -0.225492, 0.046631, 0.285512, 0.14555, 0.047423, -0.202106, 0.08033, 0.006158, 0.003628, -0.095289, -0.247808, -0.187247, -0.072479, 0.003099, 0.03055, -0.081997, 0.033272, -0.204195, -0.22535, -0.162476, -0.114666, -0.114609, -0.331978, 0.140379, -0.075868, -0.144221, 0.204937, 0.097734, 0.120764, -0.192557, 0.116446, 0.194081, 0.304155, -0.046738, 0.03219, -0.00171, -0.204437, 0.075016, -0.017123, -0.07795, 0.00798, -0.039274, 0.099114, 0.133817, 0.085062, 0.23533, -0.124078, -0.076702, -0.061699], [-0.071355, 0.047254, 0.030485, 0.053246, 0.041665, -0.080879, 0.058646, -0.093995, -0.093321, 0.00617, 0.095569, 0.038854, -0.053056, -0.107141, -0.111589, 0.129445, 0.066032, 0.118266, -0.093395, 0.006642, -0.173778, 0.079328, 0.087916, -0.057864, -0.000122, 0.004624, 0.01194, -0.0392, -0.058531, 0.074023, -0.126316, -0.082346, -0.01804, -0.107388, -0.160911, -0.031508, 0.034098, 0.08785, -0.062554, 0.059412, 0.007753, 0.048362, 0.282389, 0.041108, -0.076572, -0.129339, -0.08585, 0.072426, 0.008158, -0.016968, -0.150884, -0.036549, -0.016013, 0.026023, 0.122062, 0.029817, 0.002206, -0.153512, 0.009524, 0.01772, 0.012903, -0.128848, -0.186159, -0.124766], [-0.13438, 0.048022, 0.000218, 0.008743, 0.03357, 0.035125, -0.065476, -0.060903, 0.048827, -0.108548, 0.086129, -0.021501, 0.011591, 0.095879, -0.141803, -0.0367, -0.025777, 0.325039, -0.178827, 0.007311, -0.120929, 0.015755, 0.032024, -0.060159, 0.139555, -0.01545, 0.060603, -0.079945, -0.070674, 0.020357, -0.010481, 0.023416, 0.152229, 0.036731, 0.090814, 0.037151, -0.005235, 0.237783, -0.076266, -0.067351, -0.027836, -0.172794, -0.073266, -0.022787, -0.093691, -0.010541, -0.062556, -0.074661, -0.07133, 0.059946, 0.127018, 0.017023, -0.070223, 0.060028, -0.100764, 0.167003, 0.049429, -0.059708, -0.032057, -0.162291, -0.054277, 0.077277, 0.070862, 0.044641], [0.031065, 0.029733, 0.11461, 0.021035, 0.095193, -0.063191, -0.059979, -0.034473, -0.109923, -0.060249, -0.056167, 0.014858, 0.138267, -0.126111, 0.188544, -0.172915, 0.046432, -0.031034, -0.034204, 0.02986, -0.114533, -0.096542, 0.107132, 0.113506, 0.049309, -0.067097, -0.040731, 0.019685, -0.221369, 0.062516, -0.176324, 0.039305, -0.163069, -0.016684, -0.075934, -0.06275, 0.060413, 0.066181, -0.142104, 0.082841, 0.049543, 0.046906, -0.164748, 0.024726, -0.11998, -0.056403, -0.086899, 0.05964, -0.013614, 0.041731, -0.066193, -0.040918, 0.043077, 0.116394, 0.160429, -0.028219, 0.039116, -0.210043, -0.045193, 0.058165, -0.077243, 0.203835, 0.052941, -0.034307], [-0.038972, 0.054036, 0.063759, 0.070254, 0.001078, -0.231484, 0.176156, 0.051211, 0.032311, 0.036799, -0.062225, -0.101106, 0.081918, 0.029498, 0.025894, 0.035464, -0.090166, 0.157247, -0.055195, -0.107492, -0.048379, -0.039344, -0.062176, 0.114273, -0.067815, 0.012936, -0.018591, 0.056291, -0.085082, -0.038718, -0.018024, 0.082185, -0.039504, 0.019505, -0.082719, -0.03057, -0.033308, -0.172679, -0.043628, -0.019743, 0.049889, 0.032198, 0.121189, -0.056622, 0.026517, -0.032164, 0.000549, 0.099122, 0.149379, -0.066863, 0.056714, -0.00612, -0.060235, -0.162506, 0.075213, -0.000739, 0.045688, -0.190409, 0.071723, -0.094298, 0.082151, 0.083758, -0.046608, 0.031813], [0.133791, 0.09645, 0.146294, -0.047519, 0.025893, -0.210816, 0.106685, 0.147146, 0.086216, -0.216223, -0.048752, 0.030868, -0.051504, -0.023072, -0.142814, -0.255783, 0.104161, 0.079664, -0.030764, 0.164631, 0.056994, 0.235761, -0.24338, 0.086417, -0.104062, -0.217261, -0.048146, -0.041148, -0.008996, -0.291209, 0.067935, 0.179878, 0.068781, 0.134088, 0.078985, -0.0899, -0.031401, -0.07445, -0.279059, -0.049849, -0.163451, -0.171594, -0.082521, 0.118547, -0.233185, -0.067994, 0.081671, -0.075341, 0.012413, 0.062546, 0.065332, -0.157993, -0.068669, -0.156482, -0.067756, 0.023541, 0.236377, -0.023355, 0.201414, 0.047238, 0.153543, 0.106298, 0.063693, 0.238332], [0.201271, -0.082875, -0.053811, -0.039925, -0.124329, -0.36523, -0.123425, 0.122662, -0.099814, 0.001644, 0.034685, 0.018673, 0.11813, -0.092492, 0.050924, 0.245587, -0.108321, 0.096615, -0.099394, 0.006589, 0.365883, -0.113972, -0.031771, 0.059504, 0.078697, 0.416977, -0.021867, 0.038276, -0.059885, 0.050346, -0.114536, -0.105814, 0.11133, -0.051097, 0.086055, -0.174041, -0.056289, -0.097306, -0.160355, 0.039885, -0.060609, 0.014186, -0.050305, -0.365646, 0.145685, 0.048884, -0.124185, 0.079159, -0.035783, 0.191709, -0.273104, -0.02169, -0.254755, -0.019205, 0.077757, -0.114923, 0.021091, 0.044687, -0.121157, 0.057735, 0.222806, 0.102224, -0.153108, -0.115122], [-0.016011, -0.035549, 0.04978, -0.030447, -0.046906, -0.04392, 0.091602, 0.023738, 0.025979, -0.086123, -0.190481, -0.045961, -0.108788, 0.124057, 0.023836, 0.003618, 0.006599, -0.004382, 0.073533, 0.04818, 0.027386, 0.077625, -0.005374, -0.078185, 0.007556, 0.030143, 0.111766, -0.036061, 0.076941, -0.047973, 0.159448, 0.088059, 0.043712, 0.105099, -0.07255, 0.061355, -0.134838, 0.111959, 0.014668, 0.018115, -0.03983, -0.058259, -0.128917, 0.110365, -0.09603, -0.11694, -0.026086, -0.107735, -0.119773, 0.015297, -0.131683, 0.072222, -0.039077, -0.014072, 0.038651, -0.040399, 0.007727, 0.018117, 0.094859, -0.079274, -0.058287, -0.058093, 0.103143, 0.065649], [0.14936, 0.127447, -0.097433, 0.047407, -0.05065, -0.104363, 0.040956, 0.066689, 0.131211, -0.008646, -0.228825, -0.150012, -0.239402, 0.227899, -0.358128, 0.02144, -0.10978, 0.051986, 0.016969, -0.114568, 0.047977, -0.005717, -0.060366, -0.291395, 0.183931, -0.062697, 0.295244, -0.019309, 0.483146, -0.049182, 0.109436, -0.012784, 0.108388, -0.109563, -0.244849, -0.1116, -0.001972, -0.027732, -0.089136, -0.19291, 0.059806, -0.033034, 0.105043, -0.067552, 0.12693, 0.026147, 0.19883, 0.025066, -0.05182, 0.21878, 0.008799, 0.046669, 0.005881, -0.140062, 0.01222, 0.038913, -0.365579, -0.007378, 0.176527, -0.191783, 0.123379, -0.189432, -0.06822, 0.102081], [0.071515, -0.127245, 0.113136, 0.041864, -0.096917, -0.151655, 0.044605, -0.020206, 0.034512, 0.17199, -0.167272, -0.072454, 0.046314, 0.104611, -0.136037, 0.044415, 0.129617, 0.055524, -0.105597, 0.033476, -0.145982, -0.014775, 0.015467, -0.02147, 0.068067, -0.060466, 0.189454, 0.162387, 0.027066, 0.124525, -0.230901, 0.004364, -0.03363, 0.032316, 0.028232, 0.125365, -0.048713, 0.065641, 0.13313, -0.000309, -0.031287, 0.050519, -0.049251, 0.023321, -0.105213, 0.142375, 0.053173, 0.041796, 0.079632, -0.023601, -0.026905, -0.054546, -0.074348, -0.048519, 0.17486, 0.012545, -0.323038, 0.104899, -0.012183, 0.005406, 0.00062, -0.179198, -0.064366, -0.031586], [0.00561, -0.080422, 0.042778, 0.02516, 0.073917, 0.058217, -0.190522, -0.047535, -0.142896, -0.021823, -0.047039, 0.128519, 0.025133, -0.01512, -0.093218, -0.098488, -0.008958, -0.137087, 0.004798, -0.048227, -0.033752, -0.160495, -0.300567, 0.027847, -0.083185, -0.122785, 0.129827, -0.008862, 0.080904, 0.023772, -0.036125, -0.066919, 0.145793, -0.125153, 0.039773, -0.162746, -0.00491, -0.043482, 0.241375, -0.064597, -0.04275, 0.11657, -0.027508, -0.014227, 0.02587, 0.194843, -0.050374, -0.086982, 0.03066, -0.122327, 0.038435, 0.006965, 0.041777, -0.187561, -0.007978, -0.320548, 0.119251, -0.132292, 0.088276, 0.051008, 0.033449, 0.021507, -0.078336, -0.021187], [-0.015386, 0.123325, 0.051316, -0.053622, 0.26311, -0.075999, -0.022275, 0.195073, 0.094544, 0.170124, 0.10786, 0.163857, 0.110567, 0.019647, 0.024968, -0.087371, 0.126525, 0.07113, -0.148594, -0.049521, 0.075887, -0.108266, 0.175168, 0.194614, -0.130307, 0.009204, -0.283896, 0.066889, 0.065786, 0.026985, -0.054476, 0.120648, 0.253401, 0.228143, 0.102986, 0.073756, 0.133541, 0.000563, -0.178458, 0.087639, -0.178792, 0.018549, 0.003839, 0.054156, -0.024918, -0.010273, -0.01501, 0.013379, 0.119963, -0.036871, -0.166234, -0.171835, -0.12246, 0.163719, 0.016817, -0.148677, -0.025721, 0.0062, 0.123133, 0.146514, 0.011868, 0.210336, 0.094983, 0.045757], [0.055775, 0.059929, -0.093116, -0.128623, -0.112495, -0.156681, -0.148443, -0.06997, -0.079197, 0.074708, -0.097069, -0.07892, 0.016587, 0.069969, -0.04741, 0.012378, -0.134611, 0.036026, -0.120953, 0.11805, -0.093712, -0.210853, 0.093806, -0.162581, 0.050989, -0.245569, 0.061737, -0.004026, -0.018651, 0.003716, -0.132367, -0.045627, -0.156123, -0.224357, -0.186716, 0.027984, -0.031733, -0.114882, -0.11506, 0.008023, 0.037228, -0.11257, -0.07855, -0.041725, -0.191334, 0.009631, 0.058391, -0.093596, -0.070769, 0.015036, 0.067431, -0.106894, -0.017115, -0.180848, 0.009358, -0.108942, 0.040037, -0.098863, -0.087945, -0.020651, -0.144559, -0.107037, -0.057847, -0.057116], [-0.043829, -0.135852, 0.128497, 0.096492, 0.156569, 0.269973, -0.032509, 0.108003, -0.064141, 0.001467, -0.025645, -0.126664, 0.111326, -0.097624, 0.089364, 0.113021, 0.106368, 0.107237, -0.017595, -0.051504, 0.081467, -0.216409, 0.194374, 0.081284, -0.238711, -0.003892, -0.158994, 0.20606, 0.092989, 0.034994, -0.192074, -0.029334, -0.014621, -0.103641, 0.026519, 0.087893, -0.039517, 0.286393, 0.070044, 0.105298, 0.169296, 0.082282, 0.05439, -0.081596, 0.119182, -0.226022, 0.049372, 0.01452, -0.217138, 0.039139, 0.112059, 0.175218, -0.106527, 0.28489, 0.050836, 0.037839, -0.245669, -0.057906, -0.069406, 0.005489, 0.069936, -0.095624, 0.044942, -0.11112], [0.08752, -0.092096, 0.016903, -0.054409, -0.044802, -0.059631, -0.215432, 0.002753, -0.061902, 0.145342, 0.0934, 0.133598, 0.100471, 0.007828, 0.020347, 0.037668, 0.023144, -0.14185, -0.017176, -0.036729, 0.194826, -0.081092, 0.004266, -0.030982, -0.097752, 0.024634, -0.104429, -0.030772, 0.016386, -0.151473, -0.43995, -0.261614, 0.114729, 0.135142, 0.073353, -0.072452, -0.076068, -0.057375, 0.027219, -0.140599, -0.125449, -0.084667, 0.098929, -0.243643, -0.07764, -0.018444, -0.136385, 0.075023, 0.181076, -0.154758, 0.049873, -0.183773, -0.211051, 0.061712, 0.113461, -0.139269, -0.193302, 0.122495, -0.068809, 0.119136, -0.218873, -0.06453, 0.031136, -0.203134], [0.103309, -0.089632, 0.131068, -0.088026, -0.173589, -0.136716, -0.215953, 0.103046, 0.080303, 0.075245, -0.048572, -0.104593, -0.051011, -0.055264, -0.119029, -0.059557, -0.004779, -0.288142, 0.117075, 0.044664, 0.153966, -0.069857, -0.021094, 0.014836, 0.154747, 0.069299, 0.120082, 0.029344, -0.036169, 0.061106, -0.047922, 0.096908, 0.102394, 0.042443, -0.05057, -0.11535, -0.03412, -0.222574, 0.045508, 0.024353, 0.049693, 0.049352, -0.147457, 0.178721, 0.087588, 0.182608, -0.005959, -0.108292, 0.066079, -0.153266, -0.150265, -0.067964, 0.036837, -0.002851, 0.103885, -0.155015, 0.167203, 0.029834, -0.058385, 0.113859, -0.116904, 0.117466, -0.161896, -0.070104], [0.120414, 0.066898, 0.003256, 0.036297, 0.060026, 0.024761, 0.205189, 0.011388, 0.107486, -0.050087, 0.045224, 0.194795, -0.051977, 0.135642, 0.030496, 0.111168, 0.132796, 0.132289, 0.03586, -0.209418, -0.084515, -0.105236, 0.102789, -0.033626, -0.264148, 0.020196, -0.312688, -0.048181, 0.09457, -0.016007, -0.051595, -0.099847, 0.118378, 0.059444, 0.156379, 0.05756, -0.058848, 0.093686, 0.041704, 0.153996, -0.120003, -0.092194, -0.036555, -0.107228, 0.047037, -0.060085, 0.009274, 0.138817, -0.12313, 0.033932, 0.104894, 0.01921, -0.190797, 0.177661, -0.078336, -0.11932, -0.238944, 0.043842, -0.020213, -0.070361, -0.0114, 0.082587, 0.081248, 0.164337], [0.091764, 0.024933, 0.122666, 0.186115, 0.287051, 0.409569, -0.103501, 0.003912, -0.105675, 0.319629, -0.085445, 0.247147, 0.120617, 0.220483, 0.042914, 0.215181, 0.264624, -0.036194, -0.134302, -0.077682, 0.152192, 0.09605, 0.102959, 0.224047, -0.050965, -0.008854, -0.300753, 0.237392, 0.132135, 0.092822, 0.024717, 0.104749, 0.126015, 0.140725, 0.239759, 0.069412, 0.344076, 0.08911, -0.101181, 0.173741, -0.001506, -0.009305, -0.009167, 0.119785, -0.025281, 0.171521, 0.128691, -0.064074, 0.158225, -0.035491, -0.130529, 0.049413, -0.068445, 0.347087, 0.10392, -0.028873, -0.158064, 0.104429, 0.000633, 0.028416, 0.142464, 0.030914, -0.022615, 0.141067], [0.355773, 0.055875, -0.044867, 0.080992, 0.12688, 0.010562, 0.181441, 0.157831, 0.20483, -0.080586, -0.024349, 0.125252, 0.093735, 0.034597, 0.157085, -0.071856, -0.043918, 0.125948, 0.014724, -0.106353, 0.020111, 0.021321, 0.031643, -0.186081, -0.222824, -0.139765, -0.279557, -0.231652, -0.032575, 0.116905, -0.060049, 0.215337, 0.009261, -0.203761, -0.161162, 0.050762, 0.086664, -0.179429, 0.06668, 0.020396, 0.093808, 0.294971, 0.05293, 0.036337, 0.161695, -0.143613, -0.054068, 0.346617, -0.089495, 0.049894, 0.208898, 0.176012, -0.06748, -0.014251, -0.161717, 0.169954, -0.287207, -0.031933, -0.044891, 0.086885, 0.275016, -0.143167, 0.181912, 0.352755], [0.014056, 0.143281, 0.072842, 0.067714, 0.065368, 0.140079, 0.047649, 0.222971, -0.009626, 0.025922, -0.031369, 0.267646, 0.02601, 0.219156, 0.092105, 0.1556, 0.189556, 0.192348, -0.0074, 0.045915, 0.001381, -0.015939, 0.15761, -0.035516, -0.021811, 0.03588, -0.254298, 0.275249, 0.218342, 0.284369, 0.138326, 0.148166, 0.068749, -0.067695, 0.024317, 0.074041, -0.029609, 0.256732, -0.215881, 0.229164, 0.048853, 0.032642, -0.001745, 0.073574, 0.075143, 0.034538, -0.017028, 0.030187, 0.112876, 0.410351, -0.026438, 0.006636, -0.008196, 0.398895, -0.034753, -0.035016, 0.020652, 0.028995, 0.06072, 0.15577, 0.102745, 0.094711, -0.005006, 0.103514], [0.190988, 0.225876, 0.023129, 0.137681, -0.002362, 0.088457, 0.368301, 0.0199, 0.325081, -0.109163, 0.11997, 0.059418, 0.036675, 0.018907, 0.058492, -0.036002, -0.080535, 0.206464, 0.131004, 0.125078, -0.183818, -0.052762, -0.097073, -0.088673, -0.222267, 0.030082, -0.123201, -0.239596, -0.09138, 0.155654, 0.144349, 0.250482, -0.142957, 0.046604, -0.097805, -0.004119, -0.043041, 0.223053, 0.075155, 0.103565, 0.00563, 0.296112, -0.128341, 0.241428, 0.252901, -0.201694, 0.091064, 0.138172, -0.14514, 0.232376, 0.031193, 0.006796, 0.11126, -0.020422, -0.181913, -0.105588, -0.053506, 0.025744, 0.189058, -0.039284, 0.167421, -0.235887, 0.079511, 0.247374], [0.063409, 0.122155, 0.045186, -0.214719, -0.256851, -0.123536, 0.135216, 0.016413, 0.085015, -0.288697, -0.079694, -0.045541, -0.163705, 0.147933, -0.050996, -0.156747, -0.107916, -0.023336, 0.23264, -0.142707, -0.072538, -0.050542, 0.050985, -0.079311, -0.08243, -0.26136, -0.155366, -0.206786, 0.230067, 0.092011, 0.126305, -0.047854, -0.195526, -0.262026, -0.167056, -0.112268, 0.019318, 0.099664, 0.067398, -0.010063, -0.221074, 0.012442, -0.200575, 0.004734, 0.198369, -0.244754, 0.138205, -0.02779, -0.158491, 0.176236, 0.142517, -0.297983, -0.030487, -0.231822, -0.069125, -0.087311, -0.197824, 0.07061, 0.029136, -0.041558, -0.104667, -0.078962, -0.091625, 0.174532], [-0.161063, -0.198605, 0.003581, 0.00818, -0.150074, -0.050686, -0.100972, -0.068042, -0.087299, -0.150767, 0.10833, -0.189059, 0.04707, -0.167819, -0.143452, -0.139639, -0.196354, 0.048255, -0.16766, -0.031251, 0.060765, 0.169056, 0.019418, -0.10545, 0.140529, -0.042527, 0.211431, 0.104633, -0.067228, -0.122339, 0.091726, -0.084251, 0.139919, 0.206796, -0.081801, -0.002763, -0.063379, 0.024329, -0.041806, 0.010405, -0.013014, -0.139158, -0.023241, -0.108313, -0.173972, 0.120689, -0.167859, -0.14096, -0.049126, -0.255251, 0.033308, 0.076987, -0.024278, -0.100736, -0.055993, -0.096293, 0.182777, -0.118889, -0.142295, -0.108898, -0.148058, -0.100557, 0.08107, -0.121296], [-0.154477, -0.100766, -0.011946, -0.069527, 0.060754, 0.104064, 0.026739, 0.09767, -0.010027, 0.003371, 0.198078, -0.052921, -0.039337, -0.021796, 0.139507, -0.03973, -0.07823, 0.088364, -0.164203, -0.060024, -0.18046, -0.100363, 0.119423, 0.091546, -0.082649, -0.004232, -0.026429, 0.070908, -0.097222, -0.244094, 0.001381, -0.011006, 0.022509, 0.020512, 0.074233, -0.052334, -0.226897, 0.246732, -0.270306, 0.040094, 0.003104, -0.23453, 0.14328, -0.129231, -0.142942, 0.00512, -0.08226, -0.048443, -0.075978, -0.123847, -0.019005, 0.002929, -0.142864, -0.172547, 0.010162, 0.100807, -0.084515, -0.038533, -0.093821, -0.04684, 0.066437, -0.061506, -0.041346, 0.026521], [-0.115453, -0.065699, -0.077426, 0.127798, -0.104053, -0.007145, -0.072576, -0.132485, -0.010818, -0.038024, 0.017464, -0.21804, -0.091805, -0.208185, -0.223656, -0.031525, -0.10557, -0.141656, -0.12266, -0.040555, -0.088293, 0.133736, -0.079275, -0.087146, 0.090714, -0.045262, 0.173949, -0.151306, -0.015974, 0.173128, 0.045384, -0.162615, -0.203556, -0.074852, -0.18125, -0.141366, -0.174412, 0.007791, 0.021931, 0.060794, 0.184408, 0.04001, 0.093899, 0.149252, 0.010232, 0.005562, -0.10276, 0.074342, -0.04366, -0.140708, 0.064208, 0.136719, 0.160394, -0.251914, -0.102365, 0.160128, 0.05145, -0.19969, -0.004088, -0.130246, 0.052406, -0.004101, 0.074633, -0.207565], [0.022129, 0.230968, 0.117251, -0.148016, 0.271032, -0.022027, 0.070698, -0.00136, 0.144874, 0.105052, -0.173771, -0.039317, -0.132599, 0.151769, -0.004804, 0.007186, 0.2174, 0.07785, 0.207396, 0.018035, -0.291427, 0.059253, -0.12501, -0.012139, -0.178172, -0.087871, -0.337289, 0.179813, 0.237808, 0.210843, 0.220111, 0.186122, -0.05955, -0.114896, 0.058684, -0.002078, 0.078035, -0.082209, -0.157354, -0.108627, 0.034105, -0.03008, -0.156869, 0.240464, 0.121256, 0.283331, 0.048887, -0.097761, 0.21786, 0.293166, 0.086352, -0.078481, 0.19078, 0.111343, 0.041954, -0.104543, 0.080513, 0.14961, 0.215889, -0.09288, 0.007327, 0.106757, -0.14399, 0.261226], [0.037979, 0.049581, -0.007039, 0.21884, 0.219592, 0.095722, -0.003119, 0.07322, -0.120522, -0.006174, 0.088754, 0.028906, 0.109141, -0.15953, 0.031792, 0.028896, 0.04119, 0.057768, 1.2e-05, 0.107816, 0.091144, 0.215769, 0.093007, 0.042826, 0.23605, 0.100942, 0.205168, 0.026188, -0.005946, 0.017948, 0.128231, -0.059014, 0.164248, 0.07682, 0.034627, 0.04382, 0.139758, -0.040211, 0.016481, 0.114596, 0.168715, -0.015232, 0.081964, -0.09234, 0.056259, 0.014867, -0.156395, 0.014517, 0.092146, -0.060735, -0.02247, 0.291753, -0.116149, 0.03355, -0.010469, 0.06611, -0.056264, 0.009508, -0.082305, 0.094499, 0.018572, -0.092495, 0.186505, -0.106119]], [0.107508, -0.110564, -0.010966, -0.162276, -0.140788, -0.187936, 0.05751, -0.041428, -0.098358, 0.090421, 0.07477, 0.157739, -0.117256, 0.047302, 0.10988, -0.068345, 0.026868, -0.078468, -0.005226, -0.136069, -0.037818, -0.256489, 0.043688, -0.054132, -0.175778, -0.117774, 0.01425, -0.105267, 0.057897, -0.066209, -0.21452, 0.038547, -0.010696, -0.104979, 0.136203, -0.102127, 0.051373, 0.066812, -0.116601, -0.122747, -0.04451, -0.035252, 0.074005, -0.021555, 0.008939, 0.023724, 0.002212, -0.160312, -0.052336, -0.068605, 0.062035, -0.276644, -0.164706, -0.144092, -0.034555, 0.057268, -0.157183, 0.030793, -0.035615, 0.100525, -0.185708, 0.030753, 0.018088, -0.112724]),
    ([[0.086004, -0.026513, 0.010593, 0.069014, -0.008799, -0.136986, 0.046195, -0.061395, -0.045075, 0.040014, -0.045507, -0.027353, -0.010383, 0.003415, 0.116538, -0.038872, 0.010211, -0.044279, 0.137632, 0.107141, 0.017347, -0.110276, 0.051092, 0.082694, 0.033991, 0.094844, 0.066681, -0.027098, 0.057027, -0.130939, -0.071935, 0.089399], [0.011127, 0.100934, 0.024014, 0.009584, -0.076731, 0.063526, 0.0158, 0.158261, -0.062782, 0.103892, 0.013337, 0.012421, 0.017902, -0.079194, -0.006125, 0.122844, 0.085819, -0.162481, 0.10559, 0.088788, -0.170974, 0.129976, -0.103412, 0.021938, 0.011588, -0.001876, 0.13602, -0.009015, -0.020395, -0.04801, -0.119951, -0.083609], [0.069063, -0.065064, 0.098902, -0.024163, -0.106756, -0.00015, 0.051224, 0.094142, 0.049572, 0.037635, 0.051344, 0.033727, -0.038394, 0.001083, 0.04778, -0.038608, 0.097207, 0.064384, 0.121642, -0.034941, -0.07891, 0.068196, 0.094793, 0.106887, -0.08652, 0.01334, -0.006268, -0.050564, 0.030776, 0.081536, 0.104012, -0.009039], [-0.007391, 0.14795, 0.133776, 0.046867, 0.00676, 0.122057, -0.120165, 0.081875, -0.205149, 0.118182, 0.042148, 0.012042, 0.008178, -0.040472, 0.040165, 0.091536, -0.167041, 0.089809, 0.030487, -0.006891, -0.023605, 0.051635, -0.001365, -0.00893, -0.093401, -0.012905, 0.075898, -0.046207, -0.075358, 0.001762, -0.022498, 0.075472], [0.03444, 0.149068, -0.079632, 0.068301, -0.206371, 0.107099, 0.1436, -0.161068, 0.141816, -0.138041, 0.164733, 0.069595, 0.013166, -0.039256, -0.090325, -0.02944, 0.013174, 0.065779, -0.022246, 0.10686, -0.113452, -0.101372, 0.124872, -0.123576, 0.038554, -0.091266, -0.195653, 0.022932, 0.13676, 0.103293, 0.162807, 0.014619], [-0.016412, 0.139021, -0.114137, 0.003164, -0.081567, 0.03245, 0.186011, -0.032474, 0.033738, 0.002804, 0.161936, 0.0408, 0.107467, -0.117919, 0.055418, -0.060575, 0.148525, 0.060996, -0.0682, 0.037906, -0.080218, 0.100427, 0.066734, 0.262836, 0.226025, 0.134381, 0.048517, 0.073409, -0.01424, 0.236214, 0.1598, -0.067501], [0.076319, -0.072569, 0.055603, 0.078515, 0.027056, 0.036018, 0.046998, 0.083835, -0.103611, -0.038897, -0.158382, -0.090263, -0.001615, 0.022072, 0.024508, 0.088289, 0.063978, -0.096213, 0.022425, -0.134276, 0.00503, -0.043968, -0.055794, 0.136703, -0.178636, 0.113034, 0.035812, -0.137637, 0.028725, -0.029314, -0.109437, 0.097822], [0.044229, 0.098104, 0.058708, 0.11808, -0.028361, -0.007832, 0.14411, 0.111024, 0.103517, 0.112777, 0.076777, 0.073785, 0.090646, -0.024884, -0.027508, 0.075628, 0.084697, 0.021376, 0.159543, 0.090592, -0.085239, 0.062818, 0.084886, 0.238108, 0.186066, 0.201378, 0.117063, 0.00392, 0.085947, 0.114879, 0.102018, -0.11656], [-0.001498, 0.01296, 0.045113, 0.102637, 0.023001, -0.003409, 0.025756, -0.020506, -0.06391, 0.119092, -0.02577, -0.035228, 0.024653, -0.063834, 0.078465, 0.014949, 0.07117, 0.053904, 0.029305, 0.011581, -0.173402, 0.03246, 0.002072, 0.088873, -0.126191, 0.100926, 0.145226, -0.000248, -0.080565, -0.094678, 0.044811, -0.014945], [0.019438, 0.005052, -0.033487, -0.027395, -0.06193, 0.155654, 0.069831, 0.038869, 0.107538, -0.012634, 0.090958, 0.124592, -0.030723, 0.040911, -0.012278, 0.078382, 0.089355, 0.156366, 0.083552, 0.056386, 0.079964, -0.077707, 0.114303, -0.181826, 0.050205, -0.080737, -0.074353, -0.067479, 0.059641, 0.061988, -0.01285, -0.032331], [-0.040579, 0.134669, 0.116926, -0.046573, 0.055023, -0.192101, 0.007194, -0.090592, 0.139519, -0.046125, 0.001741, 0.088028, -0.132671, -0.078307, -0.053274, -0.168161, 0.087465, 0.110861, -0.046367, 0.097207, -0.015767, -0.105548, -0.045279, -0.021944, 0.079915, -0.053912, -0.068183, -0.027336, 0.093929, -0.046488, -0.037835, 0.062069], [-0.144444, -0.056437, 0.00628, -0.028255, -0.075282, 0.038395, 0.003183, 0.006712, 0.107093, -0.175584, 0.081727, 0.093532, 0.031907, 0.014465, -0.026971, -0.108108, 0.039637, -0.005168, -0.067357, -9.3e-05, -0.08238, -0.057048, -0.020231, -0.203413, -0.005222, 0.028627, -0.082392, -0.030324, 0.054698, 0.019459, 0.110703, -0.081339], [0.044531, 0.13114, 0.154438, -0.101649, 0.033212, 0.029087, 0.03303, -0.082195, 0.12968, 0.037595, 0.014605, 0.049043, 0.02261, 0.041404, 0.008675, -0.143959, 0.126562, -0.040098, -0.099566, -0.002767, -0.096641, -0.100087, -0.015299, -0.023531, 0.103531, 0.018046, 0.00344, 0.019432, 0.11071, 0.110548, -0.013321, 0.046954], [0.051805, 0.051229, -0.212527, 0.007225, -0.00519, -0.04943, -0.217581, 0.121475, -0.106305, -0.004351, 0.056328, 0.023487, 0.099753, -0.109656, -0.036105, -0.02893, -0.15679, 0.023235, 0.133206, -0.096264, -0.020878, -0.020094, -0.044921, 0.051328, -0.129338, -0.023114, 0.117499, -0.059987, 0.065308, -0.221238, 0.080669, -0.141481], [-0.103769, 0.100288, 0.131873, -0.08658, -0.031225, 0.109198, 0.104706, -0.06148, 0.136261, -0.153899, 0.061485, 0.080465, 0.044812, 0.062228, 0.014804, -0.157315, 0.025185, -0.068341, -0.11148, -0.007181, -0.106056, -0.001893, 0.049887, -0.340327, 0.029308, -0.150889, -0.137957, -0.018701, 0.127347, 0.086565, -0.032711, 0.057075], [0.093044, 0.091821, 0.014175, 0.014994, 0.067447, -0.072275, 0.026168, 0.10635, 0.111818, 0.072129, 0.136653, 0.000833, 0.052559, 0.110663, 0.031681, 0.0135, -0.017727, 0.063841, -0.022602, 0.019771, -0.083284, -0.027864, 0.139028, 0.174554, 0.060587, 0.098598, 0.100254, -0.08938, 0.160667, 0.110663, 0.037649, -0.131354], [0.011822, 0.011508, -0.135416, 0.076223, -0.12356, 0.064286, 0.072579, -0.084624, 0.108362, 0.000304, 0.082995, 0.097037, 0.04215, -0.011957, -0.039806, 0.043535, 0.126428, -0.027093, -0.078969, -0.055067, -0.036512, 0.054231, 0.009319, -0.040002, -0.030377, -0.046109, 0.049298, -0.019996, 0.135279, 0.130338, 0.068673, 0.000898], [0.079419, -0.019623, 0.172558, -0.028475, -0.076543, -0.121237, 0.130095, 0.196341, -0.191495, 0.031126, -0.117459, -0.010583, 0.054515, 0.172367, 0.088991, -0.007881, 0.039425, -0.132614, 0.137771, -0.17575, -0.062935, 0.186589, -0.104818, 0.152219, -0.191739, -0.016607, 0.030739, 0.071311, 0.001445, 0.004368, 0.004815, 0.012734], [-0.113162, -0.118232, -0.023511, -0.016738, -0.144046, -0.094397, 0.013215, 0.051155, -0.038309, 0.019474, 0.021308, -0.075006, 0.135533, -0.058704, 0.092176, 0.152388, 0.019989, -0.009489, 0.091249, 0.008044, 0.031248, 0.105243, 0.08626, -0.025094, -0.118297, 0.006787, 0.103956, 0.138392, -0.021823, -0.088337, -0.10418, -0.077326], [0.073704, -0.009724, 0.095664, 0.118484, 0.073686, 0.148012, 0.053213, -0.060102, -0.09886, -0.014428, 0.100283, 0.00734, 0.108885, -0.023914, 0.089423, 0.033815, 0.022413, 0.081325, 0.046924, 0.047878, -0.005747, 0.059755, -0.029584, 0.079829, 0.109882, -0.033976, 0.114161, -0.06646, -0.144497, 0.05735, -0.031806, 0.155368], [-0.111285, 0.051426, -0.22396, -0.154678, 0.005254, 0.146676, 0.030522, -0.083193, 0.081772, -0.085454, 0.107314, 0.114346, 0.073403, -0.173355, -0.097449, -0.073079, -0.021254, 0.186201, -0.108887, 0.013157, 0.088652, -0.144788, 0.145768, -0.113652, 0.129981, -0.059953, -0.047015, -0.046216, 0.048883, 0.152222, 0.084523, -0.073588], [0.049409, -0.121086, 0.100057, 0.16652, -0.025706, 0.086502, 0.020807, 0.190404, -0.157144, 0.162998, -0.17271, -0.092073, 0.112341, -0.006216, 0.082224, 0.118805, -0.024303, 3.7e-05, 0.038979, -0.055657, 0.046838, 0.098699, 0.013673, 0.191524, -0.083385, 0.064735, -0.002443, 0.180264, -0.03814, -0.003058, -0.117815, -0.056313], [-0.202487, 0.031018, -0.06288, -0.126155, -0.016192, 0.012707, 0.093572, -0.049896, -0.033828, 0.003919, 0.067782, -0.039828, -0.139278, -0.006325, -0.090583, 0.067627, 0.050944, -0.104064, -0.140465, 0.072025, 0.131693, -0.140452, 0.074016, -0.156052, 0.036181, -0.072437, -0.052208, -0.158065, 0.102861, -0.027007, 0.014844, -0.058507], [-0.076464, 0.09888, 0.093026, 0.060652, -0.049204, 0.181658, 0.012175, 0.048287, 0.110953, -0.05807, 0.041219, 0.000437, -0.046774, 0.07871, 0.029431, -0.080839, 0.093919, 0.025873, 0.009497, 0.02664, 0.033862, -0.01442, 0.065675, -0.163015, -0.05973, -0.160747, -0.258254, 0.122104, 0.051411, 0.011634, -0.003409, -0.032291], [-0.021876, 0.094448, 0.045103, 0.017861, 0.042887, 0.028245, -0.055319, -0.184578, -0.0823, -0.120163, -0.125732, -0.007916, -0.049648, 0.038808, -0.126249, -0.093673, 0.030676, 0.091999, -0.064034, -0.023116, 0.069609, -0.219121, 0.069804, -0.146823, -0.219576, -0.127983, -0.026917, 0.119715, 0.036812, -0.038697, 0.056561, 0.111356], [-0.103972, -0.013293, 0.086896, -0.098017, 0.001362, -0.010416, 0.13041, -0.162622, -0.01754, 0.036677, -0.024184, 0.104636, 0.010469, -0.027114, -0.082777, -0.091779, 0.118571, 0.142453, -0.109158, 0.109256, 0.071097, -0.075386, -0.00836, -0.271716, 0.081157, -0.053413, -0.102287, 0.096352, 0.131975, 0.117786, 0.14061, 0.04296], [-0.107938, 0.110887, -0.068306, -0.178821, 0.083402, -0.027159, -0.052636, -0.115186, -0.071039, -0.195808, -0.005304, 0.073764, -0.082624, -0.063048, -0.053389, -0.264538, 0.081271, 0.029406, -0.03615, -0.059237, 0.166175, -0.195198, 0.040202, -0.435428, 0.088226, -0.370924, 0.019178, -0.190337, 0.116127, 0.026298, 0.087311, 0.0739], [0.009981, 0.068211, -0.239215, -0.000199, 0.043913, 0.106726, -0.008511, -0.008012, 0.087845, -0.031653, 0.046142, 0.069618, -0.08829, 0.046888, -0.08496, 0.112724, 0.042906, -0.006477, -0.005755, 0.127662, -0.017265, 0.019328, 0.117529, 0.024763, 0.081381, 0.211231, 0.007029, 0.008757, 0.070088, 0.133398, -0.012884, -0.051518], [-0.0642, 0.026943, -0.04821, -0.03792, 0.113085, -0.105245, -0.209076, 0.103651, -0.058172, 0.077553, 0.06994, 0.051493, 0.000662, -0.009048, -0.039981, -0.049385, -0.165789, 0.053539, 0.056755, 0.074402, 0.035476, -0.015168, -0.07007, 0.011423, -0.224768, 0.108303, -0.028128, 0.035952, -0.008199, -0.306694, -0.080596, -0.097875], [0.219604, 0.036915, 0.009673, 0.098761, 0.149985, 0.153104, 0.078875, 0.152357, 0.00103, 0.201883, 0.046767, 0.021327, -0.110051, -0.016247, 0.122448, 0.071323, 0.075177, 0.114047, 0.040024, 0.100282, -0.037026, 0.027857, 0.028244, -0.082058, -0.020489, 0.124544, 0.054902, -0.072991, -0.041755, 0.000718, -0.03418, 0.103794], [0.106861, -0.128923, 0.199084, 0.212854, 0.073117, -0.200505, 0.038081, 0.042897, -0.140065, 0.175036, -0.211335, -0.059037, 0.10136, -0.143561, 0.127085, 0.015017, 0.003322, -0.054, 0.019458, 0.051429, -0.148163, 0.109848, -0.070357, 0.199145, -0.038844, 0.167573, 0.014511, 0.220436, 0.01072, -0.140057, -0.079829, -0.048084], [0.040875, -0.067612, -0.001209, -0.027533, 0.057937, 0.061439, 0.054046, -0.03883, -0.022732, 0.045612, -0.034991, -0.049915, 0.094655, -0.131658, 0.114048, 0.103088, 0.019539, -0.087558, 0.002402, -0.018846, -0.112222, -0.034911, -0.022944, 0.002826, -0.127879, 0.014091, 0.112555, -0.082623, -0.033079, -0.008104, -0.011936, -0.088948], [-0.115363, 0.04994, -0.265824, -0.071579, 0.007357, 0.028508, -0.058929, -0.212886, 0.012029, -0.050216, 0.039075, 0.058768, 0.008629, -0.143998, 0.014063, -0.048765, -0.063424, -0.048225, 0.005529, -0.116594, -0.105526, -0.14799, -0.045956, -0.151003, -0.00174, -0.077694, -0.034475, 0.004573, -0.052292, -0.030997, 0.046166, -0.190403], [-0.0229, 0.137846, -0.197211, -0.025565, -0.135511, -0.035324, -0.034573, 0.03364, 0.16549, -0.143709, 0.081567, 0.095671, 8e-05, -0.049985, -0.015472, -0.098159, 0.042633, 0.089766, -0.067165, -0.05963, 0.037273, 0.109274, -0.063319, 0.078543, -0.001044, 0.114191, 0.048642, -0.197532, 0.108904, 0.065086, 0.097566, -0.097591], [-0.233071, -0.031001, -0.067302, -0.084422, -0.02951, -0.01422, 0.08146, -0.045643, 0.101707, -0.137465, 0.136835, 0.129308, -0.026976, 0.00145, -0.156931, -0.182579, -0.039251, 0.128018, -0.043808, -0.04587, -0.056154, -0.005489, 0.045605, -0.072484, -0.064084, -0.144454, -0.067346, -0.074085, 0.142866, 0.065207, 0.103146, -0.095006], [-0.031265, 0.103299, -0.126469, -0.055592, -0.154093, -0.004101, 0.007213, 0.008983, 0.056037, 0.087113, 0.082579, 0.00194, -0.013071, 0.019644, -0.074439, -0.017162, -0.03228, -0.01136, -0.044221, -0.034819, 0.007393, -0.045797, 0.112046, 0.145761, -0.000558, 0.007281, 0.078376, 0.17404, -0.003218, 0.120058, 0.072066, -0.051297], [-0.080885, 0.10152, -0.022697, 0.113029, -0.003439, 0.166777, 0.050253, -0.038642, 0.066773, -0.018944, 0.068816, 0.044935, 0.045881, -0.059331, 0.053089, 0.035341, -0.055483, 0.07022, 0.043441, -0.074244, -0.140539, -0.069838, 0.026343, 0.088872, 0.018099, 0.0998, 0.05441, 0.072796, 0.075266, 0.106597, 0.07207, 0.111237], [0.041782, 0.077706, -0.208761, -0.124083, -0.261056, -0.039223, -0.007629, 0.130184, -0.026538, -0.11037, 0.124118, -0.05375, 0.083735, 0.054816, -0.072596, 0.109563, 0.069558, 0.048666, -0.003374, 0.097906, -0.069872, 0.024819, 0.020273, -0.07763, 0.220079, -0.004383, 0.089291, 0.041508, 0.109318, 0.135663, -0.00469, 0.021562], [0.037375, -0.00638, 0.113904, 0.162621, -0.045426, -0.071009, -0.013734, 0.123254, 0.002102, 0.131565, -0.083281, 0.106762, -0.032696, 0.056123, 0.197523, 0.045822, 0.036398, -0.001991, 0.119156, 0.035465, 0.032882, -0.02727, 0.046004, 0.052007, -0.334588, 0.155689, 0.018443, -0.003815, -0.06472, -0.116085, 0.032306, -0.057356], [-0.090501, 0.095692, 0.057711, -0.018364, 0.045071, 0.139334, 0.107119, -0.0916, 0.081456, -0.047479, 0.034086, 0.029744, 0.071669, 0.018632, 0.021514, -0.035457, 0.049683, 0.061673, 0.058323, -0.015909, 0.015726, 0.030165, -0.021944, 0.10822, -0.013613, -0.055829, 0.07355, -0.044483, 0.010918, 0.149222, -0.007417, 0.064342], [0.209884, 0.041835, -0.006134, 0.084116, -0.251499, -0.062268, 0.076172, 0.023379, -0.023943, 0.124314, -0.000362, 0.06164, -0.027352, -0.002711, 0.057105, -0.037684, -0.047498, 0.041273, -0.063935, 0.040341, 0.05996, -0.02915, 0.091828, -0.082006, 0.180683, -0.105732, 0.028546, 0.061245, 0.063457, 0.055695, -0.038785, 0.129498], [0.070621, -0.114451, 0.062401, 0.092511, -0.0255, -0.050104, 0.01644, 0.125938, 0.127192, -0.020182, -0.030354, -0.104645, -0.067634, -0.082243, 0.157725, 0.012054, 0.122524, -0.011471, 0.072361, 0.013577, -0.063159, 0.038227, -0.014973, 0.061838, 0.109856, -0.002539, -0.014762, -0.050191, 0.084868, 0.090258, -0.022367, 0.036601], [-0.046652, -0.101064, -0.03866, -0.088304, 0.094026, -0.162929, -0.057343, 0.049702, -0.101058, 0.025198, -0.054727, -0.00309, 0.044961, -0.093768, 0.063801, 0.016367, 0.050179, -0.10698, -0.060974, -0.088139, 0.214163, -0.063462, -0.040275, 0.034427, -0.006172, 0.082247, -0.115187, -0.084738, -0.070244, -0.004736, -0.062679, 0.001833], [0.255113, -0.063215, 0.069613, -0.019295, -0.085368, 0.120959, -0.050473, 0.128812, -0.066868, 0.137708, 0.07947, 0.096133, -0.053262, 0.025804, 0.085362, 0.005302, -0.016005, 0.099351, 0.091259, 0.082293, 0.021403, -0.036092, 0.087482, 0.225753, 0.094041, 0.007708, 0.093785, -0.009847, -0.023403, 0.05158, 0.091171, 0.058983], [-0.011139, 0.008231, -0.015037, 0.068912, -0.204759, -0.133376, 0.080987, 0.100328, 0.025336, 0.107085, -0.029841, -0.053706, 0.052207, -0.002056, 0.155812, 0.108948, 0.013909, 0.003455, 0.158674, 0.058985, -0.003662, 0.01975, 0.03414, -0.014794, 0.162134, 0.065629, 0.155137, -0.003448, 0.067498, -0.036587, 0.082626, -0.038966], [-0.192607, 0.002867, -0.032217, -0.014929, 0.074116, 0.094961, -0.095219, -0.148092, 0.115069, -0.024705, 0.018694, 0.024049, 0.078758, 0.016594, -0.050561, 0.038386, -0.076428, 0.066461, -0.002994, -0.028402, -0.035345, 0.066378, -0.016102, -0.042007, -0.01392, -0.015504, -0.074848, 0.028015, 0.060381, -0.059509, -0.014306, -0.01003], [0.082741, 0.012824, -0.056855, -0.024028, 0.063865, -0.045878, -0.016879, 0.103819, 0.158581, 0.094046, 0.078712, 0.035378, -0.015241, -0.074812, 0.068598, 0.011352, -0.064756, -0.024696, 0.113559, 0.126624, 0.066335, 0.11623, 0.08611, 0.082961, -0.027036, 0.080575, 0.038231, 0.072678, -0.03159, -0.094375, -0.087795, 0.065936], [0.131173, -0.079589, 0.241185, 0.022955, -0.050616, -0.12157, 0.070279, 0.015704, -0.043063, 0.150769, -0.056036, 0.022327, 0.050411, 0.074654, 0.059266, 0.057161, 0.073148, -0.138526, 0.015436, -0.059393, 0.057178, -0.035617, -0.002607, 0.060012, -0.081565, 0.05597, 0.087193, -0.024943, 0.061543, -0.089208, -0.051397, 0.122678], [0.008876, 0.142444, -0.084136, -0.111111, -0.050361, 0.153593, -0.030466, -0.177325, 0.086145, 0.005763, 0.074866, 0.166455, 0.032184, -0.012725, 0.104916, -0.044281, 0.136831, 0.092692, -0.107726, 0.089603, -0.036295, -0.053846, 0.052611, -0.015347, 0.123377, 0.003896, 0.031696, -0.099351, 0.09065, 0.053801, 0.089419, 0.063244], [0.088751, -0.022557, -0.042396, 0.048385, 0.009664, 0.004067, -0.07053, 0.086776, 0.007861, 0.104814, -0.109626, -0.055094, 0.087875, -0.131555, 0.056966, 0.031407, 0.027158, -0.082177, 0.072136, 0.04584, 0.108801, 0.081106, -0.050769, 0.129245, -0.031165, 0.040097, 0.032217, -0.01386, -0.031681, 0.034361, 0.081808, 0.029474], [-0.040242, -0.065472, -0.02172, -0.054897, 0.018073, 0.039803, -0.072552, -0.066191, -0.195949, 0.122423, -0.132942, -0.048412, 0.06807, -0.04895, -0.02578, 0.001948, -0.019311, -0.185375, -0.056835, -0.258146, -0.108127, -0.006933, -0.166222, -0.039694, -0.385638, 0.079878, 0.068522, 0.076252, 0.037116, -0.107936, 0.105538, -0.009509], [0.021029, 0.190702, 0.148132, 0.023288, -0.14838, 0.039586, 0.124942, 0.018289, 0.175101, 0.101592, 0.107845, 0.121739, -0.162333, 0.109851, 0.003489, -0.027056, 0.163941, 0.027398, 0.01031, 0.087503, -0.249468, -0.017094, 0.106283, -0.197692, 0.232152, -0.147727, -0.145958, -0.075852, 0.172113, 0.12358, 0.128502, 0.118471], [0.112123, 0.031202, 0.089697, 0.126056, -0.150212, -0.107755, 0.00677, -0.008237, 0.103711, 0.113556, 0.016301, -0.005567, -0.049507, 0.072875, 0.071372, 0.103431, 0.064644, -0.040657, 0.093302, 0.017604, 0.066494, 0.12215, -0.033674, 0.142137, 0.077866, -0.072736, 0.023575, 0.151844, -0.061777, 0.055003, 0.065123, 0.05702], [-0.128774, 0.212762, -0.131165, -0.11974, 0.023152, 0.251452, 0.119882, -0.023068, 0.076966, -0.096362, 0.123453, 0.072113, -0.003764, -0.103734, -0.223999, -0.018158, 0.121401, 0.212079, -0.070628, 0.017945, 0.090876, -0.149502, 0.196413, -0.134056, -0.019688, 0.084371, -0.057812, -0.134395, 0.035833, 0.208863, 0.083605, -0.090009], [0.040549, -0.014861, -0.222078, 0.018822, 0.028634, 0.088472, 0.033274, -0.075782, 0.156617, -0.069301, -0.019812, -0.082574, 0.01228, -0.135473, 0.093149, 0.103532, 0.040265, 0.151682, 0.108876, 0.132569, 0.026365, -0.038645, 0.171083, 0.005127, 0.120322, 0.010727, 0.072504, -0.033556, 0.074935, 0.070615, -0.073924, 0.095762], [-0.051899, 0.083238, -0.103369, -0.067879, 0.036544, -0.120903, -0.090581, -0.024046, -0.089301, -0.094037, -0.11608, 0.073166, 0.049897, -0.012713, 0.000363, 0.010302, 0.002768, -0.148465, -0.016089, 0.018362, 0.062389, 0.060375, -0.079091, -0.270565, -0.067108, -0.063116, 0.024908, 0.085668, 0.015366, -0.176005, -0.058108, 0.023252], [-0.131987, -0.041126, 0.253633, 0.179932, 0.191051, -0.097721, -0.03057, 0.012786, -0.000559, -0.106917, -0.061369, -0.155894, 0.159026, -0.111993, -0.213537, -0.081899, -0.178578, -0.117004, 0.049039, -0.246543, 0.024993, 0.128086, -0.292137, -0.268738, -0.099675, -0.325402, -0.166131, 0.165782, -0.038727, -0.137441, -0.128893, -0.089834], [0.019478, 0.003082, -0.108574, 0.075247, 0.041937, 0.091083, -0.074054, 0.020837, 0.077184, 0.132053, 0.054959, 0.116643, 0.107742, -0.070799, -0.048315, 0.051795, -0.145379, 0.086479, 0.112805, -0.044898, 0.068481, -0.06599, 0.11052, 0.048384, -0.257542, 0.102762, 0.132572, -0.043491, -0.097384, 0.045957, 0.045374, 0.053772], [0.148042, 0.042705, 0.022861, 0.012708, 0.016569, 0.012657, -0.111829, 0.101106, 0.08475, 0.011599, -0.128436, -0.080475, 0.067497, -0.071544, 0.024818, 0.07449, -0.070185, 0.05954, 0.092254, -0.001046, -0.12043, 0.121954, -0.062237, 0.031242, 0.032535, 0.100322, -0.005213, 0.057838, -0.100461, 0.042778, -0.019015, -0.007141], [-0.003786, 0.041285, 0.089951, 0.032271, -0.087289, -0.036711, 0.004593, -0.017578, 0.057743, 0.02232, -0.062973, 0.07324, -0.090243, 0.070512, -0.06165, 0.083769, 0.085457, 0.097163, 0.070753, 0.081167, -0.027248, 0.061558, 0.076496, -0.325954, -0.073416, -0.160393, -0.092233, -0.033741, 0.098156, -0.071041, 0.100725, 0.061391], [0.216839, -0.031484, 0.179481, 0.040351, -0.066281, 0.014258, -0.066736, 0.097873, -0.04813, 0.175448, -0.079942, -0.064727, 0.073765, -0.067777, 0.086291, -0.011033, -0.108609, -0.088596, -0.011296, 0.022522, 0.033387, 0.169991, 0.064049, 0.093426, 0.113674, 0.038746, 0.147674, -0.017069, -0.072986, 0.12287, 0.048837, 0.015684], [-0.231461, -0.000485, 0.008317, 0.027041, -0.039692, 0.100804, 0.09056, -0.185147, -0.050734, -0.200024, -0.084101, 0.073325, 0.022452, -0.033906, 0.00845, -0.0387, -0.03068, 0.071208, -0.060459, -0.022427, -0.053849, -0.091574, -0.07008, -0.143782, 0.074034, -0.170267, -0.051239, 0.104305, 0.011125, -0.034757, -0.048688, 0.007656], [0.04342, 0.080537, -0.017669, -0.025367, -0.107339, -0.076513, 0.136547, -0.043917, 0.052355, -0.072423, 0.103169, 0.126408, -0.079424, 0.065512, 0.074181, -0.026529, 0.124712, -0.011794, -0.018982, 0.11833, 0.07983, -0.068474, 0.02716, -0.227603, -0.011409, -0.051119, -0.054066, -0.001901, -0.000252, 0.09111, 0.075177, 0.023711], [0.004639, 0.068315, 0.014465, -0.043673, -0.073457, 0.009424, -0.126665, 0.034661, -0.108228, 0.114492, 0.02923, 0.048688, 0.038216, -0.061587, 0.03028, 0.072531, -0.000697, -0.118131, 0.06275, -0.167622, -0.061941, 0.074176, -0.269748, 0.120063, -0.336087, -0.064362, 0.059649, -0.160882, 0.031521, -0.247463, -0.049617, -0.091714]], [-0.178231, 0.070373, -0.201895, -0.061309, 0.023016, -0.00621, 0.127397, -0.068742, 0.057256, -0.089357, 0.02212, 0.080281, 0.056584, -0.131822, -0.038231, 0.078998, 0.076534, -0.136099, 0.076774, 0.064615, -0.01667, -0.071563, -0.095591, -0.151519, -0.141545, -0.017118, 0.002431, -0.003789, 0.141431, -0.069155, 0.06807, 0.047294]),
    ([[0.195203], [-0.102915], [0.284452], [0.055022], [-0.132014], [-0.207791], [-0.122255], [0.174276], [-0.126385], [0.165357], [-0.0507], [-0.051491], [0.075384], [0.125821], [0.107202], [0.080982], [-0.051544], [-0.08176], [0.127555], [-0.062354], [0.130269], [0.176456], [-0.21218], [0.167154], [-0.177637], [0.145142], [0.160593], [0.04298], [-0.110151], [-0.13233], [-0.095831], [0.149445]], [0.060887]),
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
        # exp_r4_gap: two-tier take. ht_on 1 = a weaker taker signal (take threshold lowered by ht_d ticks) sends an IOC
        # only back to inventory ht_tgt (0 = flat) when we hold the opposite side; the full flip still needs the normal
        # threshold. Only at spread <= ht_sp; ht_pull 1 pulls the same-side quote like a normal take. 0 = off (base).
        'ht_on': 0, 'ht_d': 0.2, 'ht_tgt': 0, 'ht_min': 10, 'ht_sp': 1, 'ht_pull': 1,
        'cs_gru': 1.0,
        'cs_fair': 1.0,
        'cs_q': 1.0,
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
        self.ht_on = int(params['ht_on'])
        self.ht_d = float(params['ht_d'])
        self.ht_tgt = int(params['ht_tgt'])
        self.ht_min = int(params['ht_min'])
        self.ht_sp = int(params['ht_sp'])
        self.ht_pull = int(params['ht_pull'])
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
        self.cs_gru = float(params['cs_gru'])
        self.cs_fair = float(params['cs_fair'])
        self.cs_q = float(params['cs_q'])
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
            p = self.pred_blend * self.cs_gru * p + (1.0 - self.pred_blend) * self.cs_fair * self.fair_sig
        else:
            p = self.cs_gru * p
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
            sig = self.mq_b * self.cs_q * self.pred_raw + (1.0 - self.mq_b) * self.fair_sig
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
        if self.ht_on and take > 0 and tk > 0.0 and not take_buy and not take_sell and ba - bb <= self.ht_sp:
            tgt = self.ht_tgt
            if self.inv < tgt and floor(ft - tkb + self.ht_d) >= ba:
                if self.ht_pull:
                    pull_bid = True
                cred = self._cxl_credit(0) if (self.zd_take and self.ht_pull) else 0
                q = min(take, tgt - self.inv - self.out[0] + cred)
                if q >= self.ht_min and self._take(0, 1, _BUY, ba, q, cred) is not None:
                    self.n_takes += 1
            elif self.inv > -tgt and -floor(-(ft + tks - self.ht_d)) <= bb:
                if self.ht_pull:
                    pull_ask = True
                cred = self._cxl_credit(1) if (self.zd_take and self.ht_pull) else 0
                q = min(take, self.inv + tgt - self.out[1] + cred)
                if q >= self.ht_min and self._take(1, -1, _SELL, bb, q, cred) is not None:
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
