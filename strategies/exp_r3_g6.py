"""exp_r3_g6 (home, R3): exp_r3_g5 (mm_r3_gru4 + take veto) + the near-front keep rule from exp_r3_qcap (qc_ahead; 0 = off).

exp_r3_g5 (home, R3): mm_r3_gru4 + the learned take veto from exp_r3_veto (veto_on; 0 = mm_r3_gru4).

mm_r3_gru4 (home, R3 upload candidate): mm_r3_gru3 (leaderboard 984k) + the 500 ms quote head on the taker GRU state blended 50/50 with MLPQ[500] (qg_on 1, qg_set 0, qg_mix 0.5) + direction memory hp_ms 300: round3 653,817 / mirror 640,477 / round2 443,678; holdout 159,531 / 159,874 / 113,023 (gru3 653,327 / 639,714 / 447,310; holdout 157,925 / 158,799 / 112,106).

exp_r3_g4 (home, R3): mm_r3_gru3 (leaderboard 984k) + the GRU-state quote head from exp_r3_qgru (qg_on; 0 = mm_r3_gru3).

mm_r3_gru3 (home, R3 upload candidate): mm_r3_gru2 (gate_thr 14) + rt_ms 150, fq_k_sp2 1.5 and a 50 ms hysteresis after passive flips (hy_only 1, hy_ms 50, hy_k 0.2; exp_r3_hyst): round3 653,327 / mirror 639,714 / round2 447,310; holdout 157,925 / 158,799 / 112,106 (gru2 638,024 / 625,673 / 447,035; holdout 157,152 / 158,439 / 112,310).

mm_r3_gru2 (home, R3 upload candidate): mm_r3_gru (leaderboard 976,148) with gate_thr 14 (from 11): round3 638,024 / mirror 625,673 / round2 447,035; holdout 157,152 / 158,439 / 112,310 (gru 634,519 / 624,778 / 437,941; holdout 156,075 / 157,441 / 109,156); gate 17 / 22 on the same plateau / lower.

mm_r3_gru (home, R3 upload candidate): mm_r3_msq + the streaming GRU taker predictor (seq_on 1): round3 634,519 / mirror 624,778 / round2 437,941; holdout 156,075 / 157,441 / 109,156 (msq 571,085 / 595,553 / 439,016; holdout 144,549 / 143,063 / 109,960); check PASS 6.7 min.

exp_r3_veto (home, R3 experiment): mm_r3_gru2 + a learned logistic veto on the fq IOC take (veto_on 1: a take whose
P(bad take, 2 s markout < 0) > veto_thr is skipped this event; weights fitted by research/diag/veto_r3.py on the first 75 %
of round3 + mirror decisions). veto_on 0 = identical to mm_r3_gru2.

exp_r3_qcap (home): mm_r3_gru2 + qc_ahead/qc_rules: a touch quote near the front of its queue is exempt from chosen pull rules; qc_ahead 0 = identical.

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
    # rfit 19 (h200_r3m_75); params rfit=19
    (0.005415, (0.3303941, -1.9259269, 4.2404875, -1.4825975, 0.5955986, -0.4271199, 0.0057687, 0.0997469, 0.0100156, -0.0028265, -0.1671455, 0.2063794, 0.0448473, -0.0450412, -0.0610664, 0.0078129, 0.11599, -0.4641638, 0.0004312, 0.0003986, -1.65e-05, -0.0001859)),
)
# exp_r3_mlp: 22-32-16-1 ReLU MLP on the same 22 features, target = 200 ms (robust mid at t+5ms+200ms - mid now), fitted on the
# first 75 % of round3 + mirror (research/diag/mlp_r3.py); holdout corr 0.49 vs 0.463 for the ridge. Inputs clipped at +-8 sd.
MLP_MEAN = [0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233382, -0.0, -0.0, 5.906055, 5.906348, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057909]
MLP_SCALE = [1.019957, 0.879338, 0.830501, 0.723114, 0.628298, 2.28085, 12.9744, 0.673953, 10.69429, 0.757186, 0.651131, 0.545643, 1.651632, 1.651633, 0.999999, 4.399308, 1.05635, 0.427403, 127.451202, 323.701813, 938.297913, 1.343849]
MLP_LAYERS = [
    ([[0.066943, 0.120272, -0.358368, -0.115884, 0.010945, -0.175679, -0.261411, 0.112239, 0.1838, -0.114866, -0.031451, -0.188784, 0.022319, -0.149227, 0.267015, -0.132292, -0.076388, -0.41682, 0.115046, 0.201321, -0.458292, 0.162697, -0.017254, 0.008661, 0.536648, -0.250936, 0.146941, 0.062188, 0.481761, -0.221041, 0.14004, -0.049273], [0.049527, 0.165229, -0.239083, -0.042354, -0.066629, 0.171265, 0.111111, -0.316688, 0.045589, 0.025842, 0.140061, 0.121268, -0.035111, 0.166081, -0.061191, -0.202608, 0.138511, -0.094053, -0.030576, 0.335219, -0.271021, -0.113142, 0.220937, 0.135911, -0.409814, -0.085793, -0.027404, 0.105903, 0.136293, 0.185961, -0.016854, 0.015552], [-0.291672, -0.119207, 0.030961, 0.122247, 0.162517, -0.094942, 0.253666, 0.014278, 0.172065, 0.15009, 0.000265, 0.284292, 0.071912, 0.026779, -0.134713, 0.259852, 0.100663, 0.353949, -0.17241, 0.146077, -0.115363, 0.022841, 0.222134, -0.084045, -0.273249, 0.304072, 0.020471, -0.134488, 0.218031, 0.028553, 0.187728, 0.151106], [-0.000635, 0.089779, 0.184897, 0.234758, 0.233159, 0.197303, -0.01192, -0.023077, -0.234368, 0.319151, 0.03369, -0.048972, 0.095157, 0.048649, -0.08427, 0.080902, 0.048699, 0.44223, 0.101248, -0.178398, 0.191833, 0.172732, -0.034094, 0.12217, 0.079339, -0.030245, 0.243988, 0.179318, -0.185257, -0.080207, -0.311141, -0.000833], [0.02545, 0.052024, 0.083945, -0.174566, 0.154057, 0.081128, -0.086149, 0.056167, -0.055505, -0.062265, -0.273824, 0.004291, 0.398352, -0.115718, 0.13195, -0.158535, 0.013204, 0.368176, 0.465274, 0.018586, -0.167041, 0.070921, -0.116818, -0.138021, 0.283977, 0.015354, -0.037041, 0.221298, -0.432361, 0.116947, 0.09942, -0.205754], [-0.046082, -0.213961, 0.017157, 0.016043, 0.291605, -0.152497, -0.017401, -0.098684, 0.067304, -0.117707, 0.091451, 0.021718, 0.179773, 0.047944, 0.019562, -0.060467, 0.116437, 0.25667, 0.103636, 0.148801, 0.20462, -0.072123, -0.020666, 0.04013, -0.551843, 0.124648, -0.116716, 0.190634, 0.145153, 0.111042, 0.234161, -0.266948], [-0.028178, -0.366293, -0.235355, -0.112522, 0.095321, 0.514204, -0.46427, -0.196664, -0.001375, -0.428219, -0.426284, 0.696893, -0.087232, 0.609089, -0.240243, 0.362304, 0.199522, 0.29372, -0.169118, 0.154267, 0.083371, -0.019951, -0.06784, 0.19711, 0.025124, 0.024469, 0.509556, 0.141625, 0.075955, 0.434384, 0.241584, 0.398111], [0.026493, -0.025249, 0.182736, 0.092406, 0.042804, -0.024718, 0.143291, -0.101047, 0.120779, 0.138127, -0.100367, 0.053303, -0.2141, -0.118666, 0.025511, 0.11286, 0.062464, 0.058981, 0.015699, -0.046808, 0.060583, 0.044302, -0.017981, -0.077249, 0.103769, 0.222212, 0.117276, -0.101792, -0.034874, 0.181032, 0.020078, -0.134994], [-0.242958, -0.416172, -0.276257, -0.101694, -0.209015, -0.275512, 0.229838, -0.588181, -0.427902, 0.175556, -0.204039, 0.022584, 0.000452, -0.222633, -0.082669, 0.574998, -0.678333, 0.558933, -0.408714, 0.075776, -0.285633, -0.319221, 0.195898, 0.081872, -0.206058, 0.023249, -0.312369, 0.244347, -0.012093, 0.129399, 0.266651, 0.445851], [0.32444, 0.071472, -0.14634, 0.261524, -0.62179, 0.131166, 0.251585, -0.65535, 0.188753, -0.094319, 0.018025, 0.12533, 0.178684, -0.396574, 0.226798, 0.242977, -0.631209, -0.172422, -0.085186, -0.600798, -0.357081, 0.342211, -0.288159, 0.10646, 0.065198, 0.145692, -0.162816, -0.045557, -0.073317, 0.379005, 0.102916, -0.077891], [0.088014, -0.072944, 0.081135, 0.048031, -0.118287, 0.210904, -0.004548, 0.086928, 0.057418, 0.023458, -0.145608, 0.039913, -0.104034, 0.049363, 0.019398, -0.040156, -0.028357, -0.068551, 0.277338, -0.088502, -0.00418, 0.124905, 0.18883, 0.01386, -0.152409, 0.036733, -0.261399, 0.008652, 0.185204, -0.166706, -0.176721, 0.10537], [-0.322693, -0.158956, -0.401976, -0.129965, -0.402451, -0.190516, -0.011472, -0.330192, -0.49554, 0.31314, -0.022773, -0.121285, 0.069603, -0.418982, 0.01482, 0.209771, -0.027682, 0.177813, -0.067498, 0.172307, -0.042609, -0.143668, -0.101161, 0.026613, 0.120705, -0.196704, 0.024583, -0.191531, -0.073524, -0.132547, 0.279965, 0.400684], [-0.085093, -0.081297, 0.325137, -0.176173, -0.026384, -0.227667, -0.047251, -0.191921, 0.022097, 0.163785, 0.240984, -0.286312, -0.007035, -0.027705, -0.092092, 0.233661, 0.311685, -0.036284, 0.032067, -0.163136, 0.05885, 0.011786, -0.204477, 0.228237, -0.106065, -0.136709, -0.223084, -0.16489, -0.075028, -0.084014, -0.247336, 0.010806], [-0.282997, -0.12814, -0.19534, -0.334562, 0.428732, 0.038923, -0.040778, -0.04184, 0.344938, 0.069975, -0.066574, -0.385734, -0.176927, 0.086314, 0.157292, 0.128472, 0.268988, 0.019416, 0.08695, -0.041762, 0.06346, -0.092785, -0.189507, -0.047271, 0.010201, 0.186727, -0.085047, -0.106782, 0.023611, -0.128958, -0.026165, -0.363271], [-0.065195, -0.153132, -0.080385, -0.688909, -0.027196, -0.292153, 0.712369, -0.328982, 0.042757, 0.01859, 0.055972, -0.251503, -0.05227, -0.105975, -0.615423, -0.423421, 0.12091, -0.256202, -0.377161, 0.237194, -0.183856, 0.154538, -0.161818, 0.24888, -0.041748, -0.151389, -0.304998, -0.231785, 0.105836, 0.203381, -0.116649, 0.179649], [-0.18632, 0.241947, 0.097802, 0.473644, 0.038817, -0.20453, -0.191199, -0.076638, -0.058033, 0.084998, -0.076714, -0.071364, 0.235598, -0.028088, 0.16679, 0.246116, -0.152069, -0.090617, -0.089045, 0.143986, 0.1488, 0.2621, 0.13354, -0.263948, -0.116339, 0.078125, -0.075966, -0.080283, -0.200206, -0.238513, 0.248191, -0.104852], [-0.024095, 0.113596, -0.123797, 0.358121, -0.100889, -0.125164, -0.090714, 0.520833, -0.13413, 0.195057, -0.063276, 0.018434, 0.120175, -0.694796, 0.261976, 0.487608, -0.038586, -0.010077, 0.567651, -0.202001, -0.142432, -0.605223, -0.20975, -0.192642, 0.034903, -0.203975, 0.590032, -0.255833, 0.035756, -0.125958, 0.440037, -0.094081], [0.154172, 0.114625, 0.569564, -0.111379, -0.610456, -0.613659, -0.478289, -0.190994, -0.009435, -0.055415, -0.04209, -0.210513, -0.2225, 0.161215, 0.048415, 0.026395, 0.460373, -0.18949, -0.184556, 0.010055, 0.06885, 0.219711, 0.23177, 0.127199, 0.34185, -0.062599, 0.240196, -0.073394, 0.475098, -0.403108, -0.285911, -0.105996], [-0.108529, 0.096269, -0.025211, -0.055042, -0.139981, 0.050161, -0.15435, -0.015445, 0.053185, -0.043506, -0.127909, -0.077368, -0.061149, -0.022121, 0.016729, -0.0872, -0.043714, 0.044695, -0.109019, -0.057356, 0.032643, 0.050282, -0.035206, -0.008852, -0.027579, 0.044184, -0.00963, 0.270621, 0.025517, -0.068444, 0.08516, -0.008622], [-0.222509, -0.92691, 0.094995, -0.03933, -0.112769, -0.193575, -0.030115, 0.017401, -0.058151, -0.002767, -0.061206, 0.270805, 0.052331, -0.086804, 0.185191, 0.168124, 0.161172, 0.018984, -0.25976, 0.007301, -0.03831, -0.233665, 0.329697, -0.082291, 0.070424, 0.328462, -0.496405, 0.101706, 0.021723, 0.304964, 0.181378, 0.141366], [0.164095, -0.051133, 0.195924, -0.078633, -0.266317, -0.194948, 0.154941, -0.10852, -0.117282, 0.199036, 0.254971, 0.090201, -0.030095, 0.055339, 0.049931, -0.062985, 0.086792, -0.181168, -0.202497, 0.045323, -0.071063, -0.103218, 0.138162, 0.082975, 0.078019, 0.421609, -0.160835, -0.178847, -0.043429, -0.057394, -0.038515, 0.038923], [0.070086, -0.139566, -0.061042, -0.292309, 0.041117, 0.032878, -0.005407, -0.031323, 0.168689, -0.1224, 0.498416, 0.393255, 0.079576, -0.096018, -0.479617, 0.28723, -0.266583, 0.032518, 0.008717, -0.130674, -0.003462, -0.048226, -0.435315, -0.629183, -0.011066, -0.059616, 0.111921, 0.405622, -0.054787, -0.085482, -0.095232, 0.261683]], [-0.278376, -0.099034, -0.204593, -0.053975, -0.317732, -0.091683, -0.047354, -0.174534, -0.609226, -0.704569, 0.499123, -0.472611, 0.247864, 0.067568, 0.046678, 0.155329, 0.341443, -0.088468, -0.009622, -0.350855, -0.295885, -0.24397, 0.113148, -0.169626, -0.001047, -0.296195, 0.034764, 0.312772, -0.120348, 0.103988, -0.091402, 0.082136]),
    ([[0.168585, -0.252286, 0.039887, -0.109907, 0.12596, -0.167051, 0.246393, -0.141371, -0.030026, 0.061707, 0.194228, 0.128337, 0.154953, -0.144466, -0.231496, -0.158408], [-0.11185, -0.001954, -0.65748, -0.182059, -0.130215, -0.272908, 0.144975, -0.485061, 0.083155, -0.127845, 0.105343, 0.033351, 0.012089, 0.038348, 0.233256, 0.046223], [0.024713, -0.003143, 0.275231, -0.060463, -0.017949, 0.334513, 0.250913, 0.346796, 0.339438, 0.160482, 0.332899, 0.175774, -0.150747, 0.211427, -0.002518, -0.014795], [-0.639437, -0.2416, -0.044399, 0.032352, -0.251727, 0.198805, -0.183777, -0.061502, 0.036179, 0.057994, 0.031361, 0.127759, -0.055203, -0.106212, 0.319384, 0.326432], [-0.454435, 0.080872, -0.268573, -0.007907, -0.122726, -0.313976, -0.476468, -0.122549, 0.012155, -0.238251, -0.023788, -0.505511, -0.714493, 0.285875, 0.009196, 0.221654], [-0.002674, 0.189375, 0.278, 0.258679, 0.018169, 0.326685, 0.267519, -0.120731, 0.02031, 0.181177, 0.200836, 0.302716, 0.040863, -0.609574, -0.25271, 0.319428], [0.108734, 0.15706, 0.250209, 0.043576, 0.129074, 0.156547, 0.323675, 0.143479, 0.353787, 0.107248, 0.185543, 0.321341, -0.235009, -0.170051, -0.024423, -0.551516], [-0.113804, -0.3113, -0.11749, -0.104168, -0.010707, 0.30636, 0.158643, -0.111952, 0.148734, 0.194378, 0.208802, 0.004262, -0.460504, 0.139871, -0.000776, 0.065859], [0.181902, 0.068307, 0.026174, -0.216895, -0.245778, -0.13641, 0.246353, 0.124476, 0.327407, 0.336649, 0.338545, 0.100827, 0.251603, -0.167972, 0.373931, -0.116822], [0.239059, -0.197664, 0.080324, 0.126201, 0.162274, -0.011076, -0.010038, 0.100613, -0.45117, 0.229271, 0.00641, -0.816061, 0.598643, 0.146778, -0.073125, 0.00545], [0.097529, -0.057735, -0.1484, -0.040611, -0.176287, -0.487833, -0.097353, -0.108813, -0.227546, 0.116581, -0.055772, -0.125936, -0.076689, 0.064261, -0.137101, -0.478306], [-0.033589, -0.106713, -0.262553, -0.04954, -0.170773, 0.019786, -0.323806, -0.068054, -0.264491, 0.194995, 0.033952, -0.326182, 0.3498, -0.495059, -0.40789, -0.048604], [-0.127255, 0.1263, 0.132122, -0.012348, 0.143074, 0.180594, 0.115362, 0.134678, 0.115873, -0.409901, 0.17746, -0.176669, -0.013763, -0.117479, 0.22609, 0.187761], [0.054243, 0.351503, -0.186892, 0.074319, 0.077268, -0.150439, -0.117804, 0.128678, 0.090459, 0.180649, 0.251682, 0.042095, 0.266992, -0.689368, -0.441449, -0.421321], [-0.215694, -0.024014, 0.145645, 0.06306, -0.108096, 0.161536, 0.006547, 0.066546, -0.054053, -0.477606, 0.027263, 0.170556, 0.112535, 0.397616, 0.059122, 0.322156], [0.181385, 0.199542, -0.161469, 0.035065, 0.032752, -0.142462, 0.157537, 0.000695, -0.627874, -0.368862, 0.120877, 0.054108, 0.010247, 0.032214, -0.118763, 0.310466], [0.261721, -0.075117, -0.31792, -0.005442, -0.132271, 0.012447, 0.160078, 0.033721, -0.324435, -0.236227, 0.047451, -0.516395, -0.122214, -0.12113, -0.126514, -0.469225], [0.093794, -0.562656, 0.307276, 0.140741, 0.155311, 0.063097, 0.028136, 0.28139, 0.275088, 0.54642, -0.204701, 0.636974, 0.217433, 0.157309, 0.022828, 0.015885], [0.127768, 0.132281, -0.386891, -0.072621, -0.230695, 0.143519, -0.469635, -0.453128, -0.023589, -0.187015, 0.205327, -0.430509, 0.124684, -0.09954, -0.001156, -0.33405], [0.137309, -0.147078, 0.070003, 0.216889, 0.261085, 0.064633, -0.012219, 0.134411, 0.194118, -0.022313, -0.344139, 0.290876, 0.173418, 0.321086, -0.363999, -0.223875], [0.042691, -0.089243, 0.230801, -0.101536, 0.249435, -0.040305, 0.027876, -0.104094, 0.170332, 0.218709, -0.07791, -0.036683, -0.110513, -0.443043, 0.016472, -0.409957], [0.205812, 0.130191, -0.097385, -0.106529, 0.005118, -0.424658, -0.164644, -0.260955, 0.046359, 0.126803, 0.047614, 0.107673, 0.280215, -0.384282, -0.192837, 0.22982], [-0.038826, -0.047256, 0.058717, 0.141466, 0.181242, 0.115117, -0.036607, 0.136449, -0.184992, -0.005515, 0.12189, 0.000646, -0.535637, -0.156596, -0.084076, -0.170541], [-0.223482, 0.15001, -0.052376, -0.110291, -0.267031, -0.330386, -0.166686, -0.13269, 0.153282, 0.403131, 0.116884, 0.03204, -0.180861, -0.192244, 0.187456, -0.175297], [0.003585, -4e-06, 0.07922, -0.163196, 0.074003, 0.028042, 0.013547, 0.335226, 0.371282, -0.229818, 0.179316, 0.030298, 0.013467, 0.249659, -0.010651, -0.152682], [0.032085, 0.102724, -0.165203, -0.223382, -0.114059, -0.094106, -0.541972, -0.280941, -0.170173, 0.000357, 0.062926, -0.835212, -0.487093, -0.053907, 0.132584, 0.08954], [-0.023315, 0.119127, -0.102696, 0.290801, -0.093167, -0.744975, 0.123413, -0.277015, 0.035289, 0.101998, 0.089249, -0.114926, 0.166078, -0.347571, -0.090007, 0.022492], [0.044123, 0.089042, -0.075957, -0.12383, 0.073883, 0.070634, 0.114846, -0.029639, -0.029193, -0.432353, -0.11187, -0.296778, -0.113306, -0.199852, -0.015916, -0.101034], [0.166041, -0.346702, 0.277411, 0.239204, 0.185581, 0.175612, 0.292734, 0.182268, 0.295251, -0.003075, 0.007127, 0.308381, -0.029198, 0.024698, -0.014589, 0.107169], [0.078143, -0.537069, 0.121968, 0.149884, 0.210873, -0.207563, -0.346699, 0.206404, -0.13718, 0.171427, 0.155044, -0.267504, -0.131033, 0.001403, -0.066493, -0.118575], [-0.050594, 0.259558, 0.143369, -0.021736, 0.006612, 0.091827, 0.185095, 0.029657, -0.236907, -0.35728, 0.191691, -0.451339, -0.041276, 0.285251, -0.123816, 0.025568], [0.402898, -0.17066, -0.077135, 0.040564, 0.174057, -0.153129, 0.122233, 0.173263, -0.085006, -0.159128, -0.147667, -0.08058, 0.127474, 0.091702, -0.177335, -0.331555]], [0.043458, 0.149123, 0.110197, 0.342115, -0.086261, -0.190143, 0.149109, 0.146057, -0.040206, -0.240387, 0.123344, 0.219386, -0.231043, -0.214568, 0.421145, -0.260841]),
    ([[0.298042], [0.399064], [0.435127], [0.414081], [0.317696], [0.666317], [-0.287826], [0.446264], [-0.610079], [-0.767302], [-0.174593], [-0.626661], [0.825489], [0.968076], [0.43161], [0.946813]], [-0.338221]),
]
# exp_r3_slow: slow-direction ridges (research/diag/slow_r3.py): targets rm(t+1s)/rm(t+5s) - rm(t); (mean, scale, w on clipped z, intercept)
SLW_NAMES = ['d1', 'd5', 'd15', 'd30', 'd60', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fl1', 'fl3', 'fl10', 'spread', 'hole', 'd5_abs', 'd30_abs']
SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, -7.546108291001272e-19, 7.546108291001272e-19, 0.0, 0.0, 1.6978743654752861e-18, 0.0, 0.0, 0.0, 1.232240613980502, 0.0, 5.966187927815806, 14.89545073636175], [4.84636859198212, 11.75951390089026, 19.444828455429125, 25.35692689478354, 34.69348778951227, 1.023785965479704, 0.8835995739934038, 0.8341675367341286, 0.7258298308787036, 0.6303098602816982, 939.3928403585505, 1665.0692101965788, 2975.0770665754285, 0.5677248883233125, 0.43221930100946554, 10.133645385319308, 20.520703908685007], [-0.30162275415038114, 0.02823086719450577, -0.06996426376957159, -0.03961008184054012, 0.003341342517940403, 0.6996990466473364, -3.8994252920921424, 6.68434148334, -4.354653838744496, 2.4396694005261232, 0.231590644527058, -0.09355699340554056, -0.18760311208474853, -0.008353986196611391, 0.38980241135999855, 0.0015446629860642585, 0.0004657713673523611], (0.0))
SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, -7.552361978829417e-19, 7.552361978829417e-19, 0.0, 0.0, 1.6992814452366187e-18, 0.0, 0.0, 0.0, 1.2323782729860528, 0.0, 5.969883379890588, 14.899700394933951], [4.848318109281616, 11.764292345260863, 19.451727900064906, 25.36585477477367, 34.7066439494089, 1.0240868561669956, 0.8838581897273222, 0.8344085493754315, 0.7260306565079572, 0.6304745466287232, 939.6747578125994, 1665.547042902018, 2974.5372827629403, 0.5678948771639639, 0.4323983605128832, 10.137014689510544, 20.528651114869255], [-0.15786874499637382, 0.2747852896221565, -0.7038221822907381, -0.2257156563518991, 0.0037097989813588183, 0.3887871050456339, -0.30905971967311574, 1.6286919405225677, -0.8860508726649116, 0.6063206771629822, 0.3573265603031138, -0.255600756384128, -1.057809738347582, -0.012269982157182408, -0.5156724413798738, 0.0022674487048791395, 0.0006840598087049105], (0.0))
# exp_r3_mlpq: quote-signal MLPs, 22-32-16-1 ReLU, targets 500 / 1000 ms (research/diag/mlpq_r3.py), same features and clip.
MLPQ = {}
MLPQ[500] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233398, -0.0, -0.0, 5.906047, 5.906341, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057889], [1.019985, 0.879362, 0.830525, 0.723134, 0.628317, 2.280914, 12.974841, 0.673973, 10.694654, 0.757209, 0.651131, 0.54564, 1.651657, 1.65166, 0.999999, 4.399385, 1.056378, 0.427418, 127.455505, 323.712799, 938.328491, 1.343882], [([[0.115109, 0.08131, -0.481813, -0.033026, -0.035607, -0.06865, -0.508995, 0.227342, 0.021407, -0.22998, -0.287102, -0.178263, 0.107989, -0.090605, 0.199303, -0.138347, -0.026529, -0.294793, 0.001721, 0.071763, -0.266043, -0.003354, -0.07766, -0.008722, 0.004604, -0.116805, 0.005479, 0.108453, 0.29898, -0.15498, 0.072663, -0.063772], [0.08769, 0.149117, -0.250442, 0.010093, -0.035912, 0.175083, 0.064414, -0.295316, 0.036123, 0.1519, 0.087482, 0.041627, -0.194121, -0.045303, -0.006856, -0.146292, 0.030236, -0.230171, -0.009391, 0.232014, -0.258071, -0.080572, 0.208026, 0.039167, -0.132178, -0.049618, -0.068801, 0.046952, 0.127808, 0.194819, -0.044145, -0.003987], [-0.266126, -0.093176, 0.026431, 0.055216, 0.222501, -0.126468, 0.415956, 0.002549, 0.266536, 0.238942, 0.034284, 0.195732, -0.167629, -0.170927, -0.196969, 0.297339, -0.036176, 0.148772, -0.11925, 0.053945, -0.09908, 0.159437, 0.236204, -0.098253, 0.185261, 0.326333, 0.073272, -0.222619, 0.236005, 0.018491, 0.114234, 0.101216], [-0.030971, 0.196108, 0.222997, 0.214366, 0.173242, 0.188922, 0.334656, -0.130267, -0.138726, 0.281685, 0.059938, -0.131253, 0.067455, 0.105312, -0.319866, 0.068162, -0.066349, 0.21627, 0.022132, -0.242165, 0.276485, 0.324566, 0.063787, 0.25966, 0.3882, -0.101718, 0.389858, 0.070626, -0.230057, -0.21201, -0.437887, 0.128943], [-0.104698, 0.093788, -0.18043, -0.195878, 0.148768, -0.054328, -0.027779, -0.042905, -0.215658, -0.315049, -0.272297, 0.161117, 0.441851, -0.033491, 0.244433, -0.104716, 0.017434, 0.50229, 0.382223, 0.121333, -0.184043, -0.271823, 0.064372, -0.201825, 0.069882, -0.14777, -0.10573, 0.236696, -0.592918, 0.233478, 0.311388, -0.034302], [-0.102519, -0.31588, -0.191132, -0.036516, 0.205428, -0.292731, 0.140924, -0.270616, 0.074158, -0.062893, -0.03157, -0.177386, 0.037616, -0.113725, 0.063291, 0.062435, 0.016837, 0.144181, 0.245639, 0.149041, 0.048, -0.030262, -0.005333, -0.023866, -0.124105, -0.103331, -0.013642, 0.20814, 0.249844, 0.140061, 0.212924, -0.022641], [-0.145737, 0.123118, 0.266918, -0.154178, 0.525502, 0.082521, 0.357779, -0.127305, 0.129955, -0.210581, -0.274216, 0.646637, -0.608116, 0.256869, -0.309918, 0.166673, -0.089959, 0.245354, -0.296633, -0.043799, -0.033266, -0.365897, -0.029972, 0.261553, 0.161181, 0.32231, -0.142665, -0.056436, 0.11609, 0.503542, 0.073304, 0.337991], [0.010023, -0.031729, 0.221711, -0.097427, -0.09831, -0.020864, 0.196795, -0.025679, 0.013509, 0.214285, -0.104273, 0.002678, 0.044222, 0.015359, 0.035346, 0.146186, 0.089557, 0.028707, 0.165488, -0.016088, 0.122614, 0.116689, -0.065433, -0.094468, 0.071323, 0.256305, 0.050983, -0.30004, 0.008667, 0.175388, 0.003072, 0.076042], [-0.044712, -0.335733, -0.35422, -0.179145, -0.308059, -0.040498, 0.816857, -0.631996, -0.436549, 0.057177, 0.392975, 0.109243, 0.202032, -0.098387, -0.1613, 0.592864, 0.069968, 0.351654, -0.149132, -0.198207, -0.107817, -0.257237, 0.224818, 0.086145, -0.310643, 0.137517, -0.413977, 0.022462, 0.568893, 0.089471, 0.304746, 0.073402], [0.307617, 0.198516, -0.011837, 0.31397, -0.011644, -0.114495, -0.229954, -0.69023, 0.259967, -0.004998, 0.28715, 0.345677, 0.039612, -0.353215, 0.162263, 0.201599, -0.191179, 0.279943, -0.035721, -0.873456, -0.195453, 0.440526, -0.206786, -0.200491, 0.107341, -0.011388, -0.03055, 0.275617, -0.013686, 0.41657, 0.191921, -0.011356], [-0.007512, -0.100997, 0.158744, -0.062291, -0.005208, 0.014861, 0.103861, 0.010875, -0.107833, 0.089424, -0.075439, 0.07113, -0.112056, -0.051424, 0.15266, -0.225319, -0.186682, -0.253783, 0.068688, 0.074682, 0.115944, 0.292323, 0.130906, -0.103793, -0.210985, -0.271156, -0.146921, -0.033059, 0.068966, -0.173479, -0.326958, 0.044835], [-0.293766, -0.014759, -0.239696, -0.120804, -0.256313, -0.110954, 0.052427, -0.14453, -0.207265, 0.276011, 0.099352, 0.066597, -0.075492, -0.375897, 0.118436, 0.229526, 0.446694, 0.2202, 0.059842, 0.172016, -0.189967, -0.057838, 0.058855, 0.097717, 0.057487, -0.160291, -0.211831, 0.27925, 0.148212, 0.039902, 0.17616, 0.484504], [-0.087051, -0.209873, 0.017324, 0.127079, 0.084773, 0.140061, -0.036246, -0.203272, -0.101121, -0.042156, -0.023212, -0.15853, 0.101139, 0.069891, -0.086772, -0.063974, 0.463618, -0.096305, 0.197799, -0.18594, 0.172228, 0.122812, -0.109458, 0.1599, -0.518722, -0.102582, -0.204714, 0.205109, -0.147882, -0.149087, -0.209181, -0.244766], [-0.254743, -0.126187, -0.076998, -0.252473, 0.406508, 0.206606, -0.068291, 0.055321, 0.477239, -0.016189, 0.085828, -0.224513, -0.308259, -0.266456, -0.003084, 0.004895, 0.130246, -0.040233, 0.086394, -0.035288, 0.107073, -0.069314, -0.063121, -0.048023, -0.092764, -0.09863, -0.068292, -0.136035, 0.171861, -0.226231, 0.02435, -0.22302], [-0.052619, -0.127311, 0.005488, -0.504799, -0.43054, -0.440891, 0.158251, -0.075181, 0.345987, 0.236556, 0.147496, -0.34722, -0.0452, -0.036106, -0.386265, -0.265135, 0.078254, -0.197558, -0.207254, 0.101348, -0.265949, 0.255541, -0.003083, 0.192721, 0.200683, -0.195185, -0.341508, -0.375652, 0.152378, 0.086826, -0.059544, 0.057144], [-0.13775, 0.242909, -0.005907, 0.44844, -0.415135, -0.372315, -0.216782, -0.008496, -0.061756, 0.063716, -0.076524, -0.071264, 0.309635, -0.212554, 0.064809, 0.337121, -0.0081, 0.07664, -0.049354, 0.11408, -0.218488, -0.092531, 0.465593, -0.082356, -0.15722, -0.052206, 0.056863, 0.039156, -0.157993, -0.150267, 0.112872, -0.201406], [0.141609, -0.127156, -0.08208, 0.373559, -0.053354, -0.006918, -0.231946, 0.517365, -0.027706, 0.201558, 0.287923, -0.059449, 0.007901, 0.039504, 0.599348, 0.433473, -0.206942, -0.074132, 0.540262, -0.135382, -0.051343, -0.135425, -0.170685, -0.116639, -0.230237, -0.268847, 0.613923, 0.087198, -0.057924, 0.150436, 0.452446, 0.147452], [-0.064263, 0.278875, 0.170945, -0.812123, 0.129643, -0.345258, -0.081733, -0.335755, -0.159335, -0.038192, -0.653121, -0.149873, 0.018238, 0.115026, -0.020372, 0.013565, -0.047387, 0.08097, 0.373992, -0.084767, -0.007531, 0.098308, 0.164801, -0.264834, 0.126127, -0.16459, 0.097245, -0.003784, 0.080837, -0.172237, -0.012318, 0.34499], [-0.082329, 0.028847, -0.037552, 0.031313, -0.131903, -0.02273, 0.001399, 0.086937, 0.026009, -0.082261, -0.132608, -0.05406, -0.094803, 0.110767, -0.06258, -0.081498, -0.1238, -0.028089, -0.055431, -0.103229, -0.238426, 0.011909, 0.1937, -0.045414, -0.106124, -0.014605, -0.090673, 0.012243, -0.002725, 0.096515, 0.056717, 0.087911], [-0.327925, -0.354286, 0.033337, -0.200396, 0.078204, -0.000625, -0.036924, 0.000585, -0.066808, 0.001351, 0.137458, 0.293542, 0.228481, -0.205807, 0.170008, 0.1919, 0.588821, 0.034542, -0.296234, 0.072862, -0.111831, -0.256547, 0.24892, -0.239631, 0.005729, 0.12444, -0.503377, -0.075771, 0.013446, -0.003297, 0.049894, -0.004127], [0.27244, 0.029346, 0.203111, 0.028286, -0.187616, -0.117, -0.163532, -0.009975, -0.038063, 0.393999, 0.187416, 0.162314, -0.014199, 0.013896, 0.212023, -0.206297, -0.097191, -0.041453, -0.10745, 0.257097, -0.329072, 0.171151, 0.109605, 0.086465, -0.467505, 0.557462, -0.314047, -0.2525, 0.008148, -0.171068, -0.023366, -0.157404], [0.090283, -0.186987, -0.035271, -0.256407, 0.045629, 0.198763, -0.048447, -0.115413, 0.208447, -0.24229, 0.276619, 0.341334, 0.135446, 0.335644, -0.144817, 0.349127, -0.054681, -0.015183, 0.062379, -0.333045, -0.006188, 0.087574, 0.056237, -0.520558, 0.180935, 0.044444, 0.112337, 0.588327, -0.066598, -0.001226, -0.061122, 0.396077]], [-0.372347, -0.499918, -0.188837, -0.056223, -0.181156, 0.003637, -0.163869, -0.32384, -0.831185, -0.574919, 0.175337, -0.136583, -0.26222, -0.074937, 0.039918, 0.112298, 0.443901, -0.05605, 0.032003, -0.113589, -0.120146, -0.04158, 0.275286, -0.160633, 0.167644, -0.305702, 0.043595, -0.125528, -0.117764, -0.182941, 0.022267, 0.125932]), ([[-0.027571, -0.192883, -0.347745, 0.003811, 0.393574, 0.315324, 0.172431, -0.050025, -0.015271, 0.029441, 0.228554, 0.039888, 0.05789, -0.322095, -0.283432, -0.287397], [-0.516558, -0.180118, -0.22989, -0.280263, -0.220198, 0.124947, 0.335766, -0.137457, 0.064612, 0.091685, 0.184085, 0.021986, 0.046635, -0.230553, -0.03998, -0.040155], [0.264683, 0.099662, 0.184927, -0.1882, 0.036348, 0.244889, 0.26155, 0.257839, 0.252649, -0.077098, 0.186338, 0.15112, -0.24361, -0.037689, -0.150445, 0.043713], [-0.432379, -0.529689, 0.037091, 0.112402, -0.174819, 0.273874, -0.120713, -0.085909, 0.104011, -0.12416, 0.09996, 0.111862, -0.024285, -0.21911, 0.396906, -0.38468], [-0.217629, 0.204877, -0.120021, 0.020856, -0.11155, -0.012376, -0.180681, -0.17877, 0.106232, -0.298243, -0.264039, -0.360213, -0.180375, 0.015746, 0.11138, -0.470938], [0.063312, 0.113532, 0.17034, 0.070951, -0.04261, 0.080077, 0.221567, 0.018596, 0.045228, 0.307195, 0.073176, 0.188384, -0.232729, -0.229723, -0.299516, -0.21771], [0.056707, 0.003541, 0.146726, 0.098616, 0.113662, 0.12565, 0.257192, 0.227652, 0.137563, 0.532293, 0.056374, 0.411372, -0.154828, -0.669552, -0.116143, -0.017964], [-0.101517, -0.207592, -0.144319, -0.054479, -0.066972, 0.298298, 0.173618, -0.150759, 0.082006, 0.136435, 0.142844, -0.023351, -0.691815, 0.094354, 0.178209, 0.097141], [-0.044074, 0.190889, 0.288308, -0.311091, -0.75981, -0.048301, 0.511061, -0.025749, 0.358775, 0.269606, 0.287984, 0.069557, -0.159675, -0.374379, 0.28878, -0.199024], [0.249384, -0.72588, 0.001731, 0.007052, 0.137179, -0.276184, -0.367004, 0.089363, -0.254962, 0.021374, -0.040686, -0.078688, -0.419414, -0.437615, -0.175822, 0.094268], [0.176473, 0.028815, 0.026196, 0.033194, -0.064246, -0.38243, 0.139448, -0.033864, -0.135825, -0.066798, -0.020727, -0.053612, -0.124619, -0.471133, -0.0338, -0.765846], [-0.197948, -0.126046, -0.23739, 0.003467, -0.115913, 0.112551, -0.273635, -0.081429, -0.220967, 0.092315, -0.00708, -0.433149, -0.318495, -0.10326, -0.208953, 0.040635], [-0.239361, -0.622824, 0.041045, 0.082334, 0.022787, 0.174476, 0.180614, 0.123924, 0.204871, 0.09193, 0.233018, 0.308447, -0.668923, 0.264364, 0.185073, 0.241751], [-0.107393, 0.153394, -0.357966, 0.000194, 0.230583, -0.30471, 0.053541, 0.105979, 0.227124, 0.094895, 0.092704, 0.044728, 0.132688, -0.229915, -0.235589, -0.759849], [-0.244567, 0.081194, 0.07362, 0.171076, -0.228859, 0.228338, -0.074355, -0.040903, -0.059604, -0.189602, -0.039561, 0.362199, -0.118674, 0.421706, 0.33095, 0.137971], [-0.009388, 0.151339, 0.023937, 0.1112, -0.005057, 0.025711, 0.104544, -0.017875, -0.714073, -0.228809, 0.05757, -0.001946, 0.161567, 0.094676, -0.047479, 0.199499], [0.234889, -0.180448, -0.033876, 0.069956, -0.12887, 0.089671, -0.009197, -0.00356, -0.359505, -0.30836, -0.030578, -0.54559, -0.626562, -0.064614, -0.176315, -0.077614], [0.121374, -0.12841, 0.294677, 0.02278, 0.036726, 0.218473, -0.071181, 0.318749, 0.216688, 0.321945, -0.202713, 0.183378, -0.028863, 0.070938, 0.123052, 0.276189], [0.295418, -0.202506, -0.228601, 0.028921, -0.017109, -0.102226, -0.232673, -0.3637, 0.009982, -0.12841, -0.024512, -0.746574, 0.209217, -0.260951, -0.128635, 0.084713], [0.061995, 0.057549, 0.014073, 0.200728, 0.315521, 0.215353, -0.300167, 0.085389, 0.10956, -0.393146, -0.27488, -0.060099, -0.047562, 0.257816, -0.330261, -0.016686], [-0.03017, 0.007836, -0.102176, -0.13593, 0.030219, -0.551862, 0.090825, -0.057643, 0.01743, 0.219743, -0.193572, 0.077315, -0.003199, -0.178152, 0.002861, 0.080977], [0.129695, 0.140087, 0.130868, -0.235848, 0.004614, -0.368368, 0.332013, -0.025784, 0.109194, 0.097701, 0.172828, 0.454377, 0.251621, 0.088554, -0.159965, -0.021441], [-0.052951, 0.0147, 0.032005, 0.090394, 0.095634, 0.014506, -0.068332, -0.000865, -0.113952, 0.035284, 0.174354, -0.845791, -0.82915, -0.130916, -0.113619, 0.163424], [-0.066457, -0.041319, -0.222609, -0.145888, -0.054722, -0.344231, 0.034595, -0.150036, 0.175916, 0.238161, 0.036528, 0.122612, -0.069512, -0.474215, 0.293481, 0.085179], [-0.013744, 0.267296, -0.212125, -0.126317, 0.033841, 0.041463, 0.117248, 0.163385, 0.231587, -0.159418, 0.231955, 0.145253, 0.175698, -0.169891, 0.022792, -0.574972], [-0.046273, -0.030519, -0.204473, -0.19274, 0.030095, -0.25686, -0.449842, -0.306331, -0.152954, 0.226623, 0.009407, -0.500178, -0.186803, -0.286045, 0.104134, 0.011833], [-0.125458, 0.113711, -0.139693, 0.086296, -0.382783, -0.214655, 0.313744, -0.627059, -0.141977, -0.001737, 0.096826, -0.247034, 0.218764, -0.27585, 0.058044, -0.399138], [0.313673, -0.099728, -0.027795, -0.119856, 0.263856, 0.125975, 0.261776, 0.037098, -0.000654, -0.326278, -0.238669, -0.167382, 0.177572, -0.5667, -0.224387, 0.137225], [0.028506, -0.250241, 0.36016, 0.263846, 0.132327, 0.231349, 0.310536, 0.212473, 0.081309, 0.072361, 0.026478, 0.633416, -0.094806, 0.002471, 0.148398, 0.203716], [-0.126785, 0.042597, 0.026755, 0.155432, 0.364866, -0.127499, -0.162217, 0.287633, -0.286054, 0.161216, 0.122574, -0.304419, -0.009432, 0.007609, 0.102193, -0.152798], [-0.026782, 0.180205, 0.246158, 0.012003, -0.056198, 0.137258, 0.151755, 0.080839, -0.325863, -0.381937, 0.14843, -0.184154, -0.080751, 0.300176, 0.023724, -0.025867], [0.233108, 0.079817, -0.284902, -0.001156, 0.056095, -0.255498, -0.054593, 0.028927, 0.243284, -0.131359, 0.002102, -0.116436, -0.005287, -0.216927, -0.07909, -0.160026]], [0.307555, 0.241415, 0.201411, 0.218637, -0.061865, -0.307291, 0.064863, 0.08104, 0.035137, -0.532488, 0.024755, 0.111849, -0.04148, -0.077175, 0.278864, -0.409452]), ([[0.533238], [0.73617], [0.542852], [0.532674], [0.773582], [1.47966], [-0.544173], [0.713273], [-0.867546], [-1.493959], [-0.233072], [-1.243846], [1.697017], [2.181837], [0.79342], [1.35674]], [-0.625446])])
MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
# exp_r3_seq: GRU + head weights (research/diag/seq_r3.py export); holdout corr streaming 0.505 vs 0.494 for the MLP.
SEQ_LITE = True
SEQ_TM = [2.73969, -1.6e-05, 1.6e-05, 0.0, 0.047376, 0.047376, 0.0, 0.0, 0.0, 1.231694, 5.907079, 5.907079, 0.0]
SEQ_TSD = [1.080247, 0.118222, 0.118222, 0.089617, 1.451691, 1.451691, 0.632855, 0.196772, 0.156216, 0.572964, 1.651027, 1.651027, 0.652184]
SEQ_XM = [-0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.233382, 0.0, -0.0, 5.906633, 5.906633, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.05788]
SEQ_XS = [1.020368, 0.879681, 0.830847, 0.723417, 0.628551, 2.281788, 12.978411, 0.674084, 10.704477, 0.755279, 0.651221, 0.545623, 1.651871, 1.651871, 0.999999, 4.399956, 1.05635, 0.427403, 127.463814, 323.861115, 938.76239, 1.34419]
VETO_W = (-0.6034678381827514, -0.02134485592014866, 0.28409870574515245, 0.1280916261120317, -0.0642351206352549, -0.01117088601994815, -0.03164938808484751, -0.1066196251657105, -0.0016446962327560496, 0.00787443459112447, 0.00872608419060543, -0.2561832522240634, 0.07814509372705362, -0.08169749820538913, -0.023513655592619788, 0.13389273764182133, 0.09026371899329524, -0.09997630552383359, 0.11277140646855736, 0.2224159724217037, 0.3068233731159911, 0.12668103292412772, -0.08422022713110422, -0.0014684456953883364, -0.008076593320109914, 0.18235104982255612, 0.2730387016585246, 0.143345900793056)
VETO_B = -0.4166766047150409
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

# exp_r3_qgru: same head trained on the mean move at t+5ms+{300,500,700,900} ms (qg_set 1).
SEQL_XM = [0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233398, -0.0, -0.0, 5.906047, 5.906341, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057889]
SEQL_XS = [1.019985, 0.879362, 0.830525, 0.723134, 0.628317, 2.280914, 12.974841, 0.673973, 10.694654, 0.757209, 0.651131, 0.54564, 1.651657, 1.65166, 0.999999, 4.399385, 1.056378, 0.427418, 127.455505, 323.712799, 938.328491, 1.343882]
SEQL_YSC = 2.6552202701568604
SEQL_HEAD = [
    ([[0.062218, -0.174878, 0.037461, -0.020523, 0.043446, -0.200942, 0.060817, -0.068237, 0.025641, -0.180907, 0.094438, 0.073901, -0.062632, -0.073746, -0.067966, -0.088145, 0.048333, -0.142451, 0.083615, 0.144055, 0.009432, -0.177903, -0.098997, 0.374912, 0.129099, -0.04468, 0.142977, -0.107797, -0.082448, 0.033986, -0.037295, 0.034128, -0.124717, -0.062826, -0.06883, 0.026447, -0.257979, -0.187208, 0.027684, 0.076346, 0.208135, 0.143293, -0.055476, 0.009761, 0.101568, 0.047319, -0.004787, -0.077487, 0.029795, -0.123376, 0.06444, -0.167534, -0.111567, -0.018417, 0.073287, 0.086584, -0.041914, -0.075564, 0.115761, 0.177881, 0.176165, -0.031945, -0.008336, -0.029121], [-0.006274, -0.03038, -0.09584, -0.112596, -0.163168, -0.067809, 0.058072, -0.019638, 0.04855, -0.145966, 0.100918, 0.010142, 0.030238, -0.072567, -0.086162, 0.0402, 0.083374, 0.070544, 0.045238, -0.107796, -0.079445, -0.00252, -0.004819, -0.052411, -0.034642, 0.005665, -0.077999, -0.030336, -0.019843, -0.009807, 0.032823, -0.036679, -0.025998, -0.150678, 0.130762, -0.062779, -0.122994, 0.096156, -0.032395, 0.058716, 0.013539, 0.022846, 0.022071, -0.096619, -0.033391, -0.079676, 0.079348, -0.095973, 0.20364, 0.095666, 0.177586, 0.172877, -0.09846, -0.108429, 0.219012, 0.012317, 0.15614, -0.147657, -0.019758, 0.150265, -0.085987, 0.094364, -0.000999, -0.080085], [0.088418, -0.00372, -0.024696, -0.072118, -0.129348, -0.124733, 0.147863, -0.04132, 0.050455, -0.072387, 0.178753, -0.12395, 0.008847, 0.044577, -0.008335, 0.042459, -0.074485, 0.15167, 0.019471, 0.041401, 0.009907, -0.141016, 0.027096, -0.138195, -0.127464, 0.052983, -0.164588, 0.03793, 0.100561, -0.159967, -0.056388, -0.021115, -0.006933, 0.04921, -0.05116, -0.136957, 0.144226, 0.03086, -0.040987, -0.189026, 0.142741, -0.100135, -0.030353, 0.120965, -0.081082, -0.055705, 0.047195, -0.056774, 0.051751, -0.032883, -0.035763, 0.098714, -0.083243, -0.137178, -0.007473, -0.15713, 0.124971, 0.058606, -0.175331, -0.065107, -0.077115, 0.056411, -0.068559, -0.149423], [0.01099, -0.09438, -0.125805, -0.026451, -0.199463, 0.205097, -0.045095, 0.070921, -0.090082, 0.084298, -0.063292, 0.112342, 0.129123, 0.11071, -0.136653, 0.105423, 0.230128, -0.13178, 0.147168, -0.314423, 0.111359, 0.087955, 0.170206, -0.210674, 0.098419, -0.022753, -0.013502, -0.143281, 0.08961, -0.112374, -0.164963, 0.066428, 0.204018, -0.071728, 0.111249, 0.075448, 0.336207, -0.023842, -0.021016, -0.038442, -0.064572, -0.080029, -0.181646, -0.117056, 0.074533, -0.060992, -0.113167, 0.012896, -0.146918, 0.122161, -0.126683, 0.015686, -0.106459, 0.068207, -0.041962, 0.240134, 0.234473, -0.115089, -0.014622, -0.04447, -0.06815, 0.089637, 0.109022, 0.157066], [0.091444, 0.18316, -0.268748, 0.026045, 0.351427, -0.089609, 0.38249, 0.060424, 0.070579, 0.137213, -0.128501, 0.116767, 0.257816, 0.090943, -0.181034, -0.09522, -0.247092, 0.072691, 0.19641, 0.037274, 0.26762, 0.126077, -0.062791, -0.216157, -0.02018, -0.031649, -0.28047, -0.140612, -0.249699, -0.327849, -0.447748, -0.017583, 0.066016, -0.052904, -0.065674, -0.029012, 0.087845, 0.310694, -0.076647, -0.196351, -0.201611, -0.092056, 0.019564, -0.180364, -0.179351, -0.065522, 0.030191, 0.014981, -0.307193, -0.063547, -0.186618, -0.01124, -0.055168, -0.010026, -0.291836, -0.071319, -0.119243, 0.155248, -0.008791, -0.096448, -0.26225, 0.067332, 0.037228, 0.302723], [-0.137923, -0.000866, 0.052821, -0.081962, 0.016937, -0.096031, -0.070452, 0.098728, 0.08936, -0.101178, 0.090685, -0.111609, -0.097241, -0.233138, 0.002433, -0.020406, 0.043951, 0.115092, 0.03428, 0.083721, 0.176246, -0.042942, 0.092476, -0.017651, 0.009357, -0.064138, -0.036939, -0.073661, -0.053283, 0.077183, -0.01922, -0.048227, 0.061259, -0.051764, -0.083574, 0.005728, 0.09144, 0.050064, 0.058158, -0.094472, 0.118686, 0.015357, 0.065243, -0.087805, 0.077407, -0.132024, -0.089316, 0.038919, 0.085926, 0.116029, 0.098856, 0.192198, 0.040177, 0.105068, 0.138076, -0.027471, 0.023467, 0.033512, -0.118771, 0.061539, -0.117748, -0.019773, 0.04695, -0.016178], [-0.120014, 0.106671, -0.278894, -0.244848, -0.00693, -0.157749, -0.043013, 0.072937, 0.099344, 0.143419, 0.059983, 0.448819, 0.008424, 0.121485, 0.086158, 0.006178, 0.141936, 0.068685, 0.044233, -0.02877, 0.036698, -0.17622, 0.100226, -0.02719, -0.087116, -0.084407, -0.08403, -0.078763, 0.049943, 0.114371, -0.009521, -0.018309, 0.097356, 0.144188, 0.115002, -0.015759, -0.002806, 0.156916, -0.240472, 0.126501, 0.00374, -0.098131, 0.00659, 0.382773, -0.144038, -0.226062, 0.095162, -0.10107, 0.227259, -0.18159, 0.310655, -0.02646, -0.239079, -0.024421, -0.227295, -0.492617, -0.127122, -0.08734, -0.062376, -0.393972, 0.062262, -0.273668, 0.059452, 0.200806], [-0.04628, -0.073755, -0.101009, 0.031373, -0.103535, -0.060924, 0.049755, 0.135566, -0.014833, 0.072484, -0.055742, -0.055503, 0.013168, 0.08548, -0.092498, 0.077619, -0.016046, 0.061386, -0.026216, -0.07115, -0.033855, 0.051503, -0.201777, -0.10326, -0.210411, 0.093907, -0.022425, 0.104709, -0.060346, 0.136279, 0.123294, 0.030914, 0.076874, 0.090083, 0.033801, -0.112287, 0.128676, 0.107983, 0.099949, 0.09605, 0.045443, 0.06428, 0.168238, 0.013231, -0.063424, 0.033791, 0.028131, 0.093867, -0.052818, -0.035762, -0.045527, 0.154701, 0.006343, -0.016169, 0.061419, -0.031797, -0.019018, 0.068965, -0.190114, 0.091299, 0.043889, -0.170306, -0.052313, 0.093234], [-0.176961, 0.029633, -0.037855, -0.208679, 0.158925, -0.070076, -0.042314, -0.001506, -0.063699, -0.032594, 0.007737, 0.190706, 0.132801, -0.133456, -0.170821, -0.316651, 0.00977, -0.141015, 0.13914, -0.075066, 0.117335, -0.048906, -0.194591, -0.177392, -0.127564, 0.065118, -0.226814, 0.012392, 0.181399, -0.088151, -0.124656, -0.303073, -0.217859, -0.00708, 0.004352, -0.153495, 0.14652, 0.17172, -0.305837, 0.087845, 0.143072, 0.130207, 0.187404, -0.050893, -0.283348, -0.141631, 0.101409, 0.050747, -0.119857, -0.068562, 0.172611, 0.112149, -0.119498, -0.04803, -0.134311, -0.140599, -0.05243, -0.070205, 0.202176, -0.229135, -0.219283, 0.127986, 0.076242, 0.009832], [-0.227917, -0.247151, 0.070002, -0.048627, 0.070756, 0.081873, 0.086378, -0.029558, -0.308379, 0.08908, -0.141786, 0.20271, 0.045892, -0.175724, 0.155568, -0.094966, 0.113598, -0.267471, -0.067176, 0.082706, 0.024082, -0.133485, -0.19649, -0.066849, -0.042382, -0.191924, -0.246587, -0.114126, -0.39679, 0.0977, 0.018027, 0.084953, 0.020121, -0.009004, -0.066313, -0.247665, -0.133658, 0.076871, 0.117961, 0.048329, -0.05653, -0.012372, -0.046833, 0.070589, 0.158346, -0.133262, 0.029084, 0.223648, -0.036615, 0.033112, 0.058766, 0.331882, -0.079437, 0.068839, -0.019508, -0.270515, -0.155901, 0.095924, -0.223432, 0.015651, 0.022222, 0.01699, 0.030338, 0.150971], [0.250673, 0.076489, 0.007699, -0.117622, 0.018912, -0.1089, 0.117603, 0.297565, -0.165032, -0.124727, -0.254863, -0.024377, -0.192926, 0.136702, 0.042135, -0.118991, 0.049671, 0.025641, -0.033449, 0.014894, -0.03277, -0.032464, 0.04261, 0.222699, -0.070214, 0.038154, 0.074885, 0.137783, 0.120553, 0.179432, 0.007655, 0.265944, -0.071934, -0.077277, 0.116979, -0.045707, -0.139152, -0.033075, 0.164792, -0.019784, -0.01925, 0.110692, -0.106257, 0.199942, 0.065436, 0.082014, -0.055501, -0.17177, -0.083896, -0.064473, 0.100632, 0.057159, -0.036289, -0.134165, 0.176583, 0.208586, -0.005655, -0.122634, -0.061033, 0.085668, 0.06774, 0.180022, 0.038161, -0.03678], [0.13902, 0.055165, -0.145357, -0.132865, -0.100619, 0.023025, -0.26137, -0.047682, 0.239006, -0.29971, 0.219694, 0.045829, -0.07558, -0.205034, 0.04308, -0.240305, -0.158736, -0.071778, 0.048393, -0.00311, 0.223733, -0.031175, 0.059, -0.095549, 0.111535, 0.088093, -0.111788, -0.188367, -0.120228, -0.062985, -0.27717, 0.195426, -0.087798, -0.055591, 0.102662, -0.28701, 0.161571, 0.043436, -0.019605, 0.27442, 0.284001, -0.029647, 0.231644, -0.229674, -0.056354, 0.032245, 0.250654, -0.307626, 0.153502, -0.157033, -0.07141, -0.14221, -0.269099, -0.106787, 0.05886, -0.280773, 0.080282, 0.311321, -0.064754, -0.161942, -0.209234, -0.148042, 0.131437, -0.034488], [-0.002027, 0.094767, 0.050745, -0.095884, -0.082585, 0.13216, -0.029622, -0.085032, -0.104951, -0.023783, 0.115953, 0.396302, 0.125058, 0.03496, -0.072098, 0.066071, 0.080753, 0.245673, -0.091681, -0.071239, -0.007614, 0.001103, 0.139046, -0.034045, -0.027338, 0.162416, -0.061934, 0.160464, -0.06291, 0.03551, 0.033661, -0.045075, -0.052461, -0.074854, -0.141807, -0.006634, 0.019101, -0.076565, 0.080802, -0.104969, 0.002439, -0.024311, 0.022718, 0.050238, 0.065391, 0.130352, 0.033861, 0.114974, -0.233791, 0.108677, -0.058259, -0.138476, 0.08888, -0.181198, 0.104321, 0.311531, -0.155366, 0.362416, 0.24991, 0.186742, -0.037966, -0.015729, 0.019781, -0.025685], [-0.318563, 0.039657, -0.153477, 0.055632, 0.026259, -0.020136, -0.133429, 0.084332, 0.223303, 0.015881, -0.126145, 0.217239, -0.054586, 0.05755, -0.1429, -0.007788, -0.009448, -0.065885, 0.039561, -0.114505, -0.145663, -0.044972, 0.228472, 0.068001, -0.177043, 0.070637, -0.051965, -0.114949, -0.12875, -0.144699, -0.020162, -0.005734, -0.157489, 0.328319, -0.213424, -0.096337, -0.026929, -0.045527, -0.075123, -0.193164, 0.068304, 0.053272, 0.126642, 0.276678, -0.240985, 0.162319, 0.074143, 0.140725, 0.162579, 0.061908, -0.230117, -0.05964, -0.020568, -0.041046, 0.089913, -0.053613, -0.061259, 0.152633, 0.221592, 0.282988, 0.037593, -0.041191, -0.09836, -0.071926], [-0.078107, 0.092544, 0.038369, 0.157296, -0.201853, 0.001306, -0.157698, 0.006056, -0.154129, -0.106121, -0.170751, -0.031585, -0.133447, 0.165281, -0.000768, -0.201398, 0.004135, -0.307944, -0.083621, 0.003231, 0.064045, -0.039494, -0.403233, 0.019279, -0.061115, -0.11997, -0.224927, 0.087367, -0.014308, 0.131446, -0.076605, -0.173158, -0.266806, -0.201144, 0.00012, 0.113524, 0.022591, -0.023879, 0.149518, 0.033058, 0.086529, 0.278143, -0.2232, 0.053631, 0.072584, -0.270707, -0.038805, 0.03016, 0.068427, -0.048203, -0.197941, -0.008958, 0.006805, -0.094813, 0.032078, -0.034016, 0.135698, -0.052492, 0.048931, -0.090449, -0.067236, 0.158673, 0.089407, -0.190156], [0.028864, -0.184955, -0.048543, -0.124372, 0.144021, -0.302887, 0.370093, 0.061299, 0.061401, -0.064171, -0.124189, -0.158558, 0.172122, 0.176857, 0.042113, 0.022973, -0.148258, 0.042008, 0.198005, -0.047659, 0.034032, -0.112104, -0.154182, 0.095724, -0.071933, 0.147122, -0.094067, 0.01874, 0.001604, -0.021005, -0.223731, -0.19412, -0.387682, -0.010412, 0.229789, 0.139209, -0.191791, -0.079381, 0.200581, 0.041183, 0.055518, 0.002187, 0.092634, 0.033237, -0.219559, -0.22168, -0.09333, -0.073173, -0.238752, -0.001794, -0.147187, 0.086269, 0.029698, 0.040337, 0.078606, -0.113848, -0.004566, -0.037268, 0.350017, -0.095489, -0.061153, 0.217647, 0.190645, 0.247463], [-0.118227, -0.050234, -0.033166, 0.011109, -0.091454, 0.145326, 0.07133, 0.067746, -0.096386, 0.140762, -0.184351, 0.002373, 0.196371, -0.181943, 0.027506, 0.054324, -0.259343, 0.049937, 0.0573, 0.276963, 0.12368, -0.196311, 0.015842, 0.288902, -0.196913, 0.385828, -0.140648, -0.047636, 0.074808, -0.040499, 0.0602, -0.004117, -0.155705, -0.113641, -0.236111, -0.084611, -0.422608, -0.124489, -0.14024, -0.348966, 0.106185, -0.117191, -0.093228, -0.08896, 0.132799, -0.380268, -0.08243, 0.236264, -0.477374, -0.201373, -0.031388, -0.064338, -0.011765, 0.188564, 0.14584, -0.044893, 0.067371, 0.035849, -0.19891, -0.054274, 0.043701, 0.154996, 0.150271, 0.009889], [0.109805, 0.019949, -0.048149, -0.136993, -0.147594, 0.126348, -0.055608, -0.214915, 0.139045, -0.103544, 0.093871, 0.092198, -0.137702, 0.268882, -0.151776, -0.049049, 0.227569, 0.110965, -0.134254, -0.30438, -0.13626, 0.311928, -0.12833, -0.053992, 0.034374, 0.054688, 0.093257, -0.001145, 0.087025, 0.000859, 0.050989, 0.125312, 0.25186, -0.100979, 0.350767, -0.142397, -0.001164, -0.047315, 0.148228, -0.137127, -0.150686, -0.072588, -0.073944, -0.050951, 0.055464, -0.171855, -0.237217, -0.058309, -0.198439, -0.374803, 0.135062, -0.098231, -0.021898, -0.140156, -0.100086, -0.35501, -0.024502, 0.116857, 0.093826, -0.315433, 0.086588, 0.312787, 0.022044, -0.092915], [-0.114691, -0.099019, -0.061156, 0.081851, -0.025864, -0.000372, 0.03532, 0.055871, -0.167101, 0.237949, 0.003842, 0.140397, -0.108661, -0.02587, -0.07166, -0.026316, -0.046263, 0.081513, -0.056676, 0.229298, 0.084457, -0.010347, -0.002136, 0.017754, -0.137682, -0.000924, -0.057599, -0.051088, 0.102824, -0.034501, 0.050501, -0.142439, -0.078847, 0.045321, 0.129514, -0.061802, -0.111178, 0.049364, 0.014948, 0.047984, 0.095599, -0.113986, 0.064414, -0.138388, 0.136698, 0.083132, 0.093025, 0.155953, 0.026768, -0.110339, 0.053989, 0.075312, 0.060275, 0.027217, -0.089262, 0.018803, 0.002387, 0.008609, -0.046739, 0.162905, 0.11487, 0.060661, -0.297063, 0.040676], [-0.032019, -0.007261, -0.006873, -0.171992, 0.185104, 0.009216, 0.093857, 0.121506, 0.068009, -0.093039, -0.065154, 0.309002, 0.195258, -0.164192, 0.068533, -0.273198, -0.095946, 0.059983, 0.088604, -0.016956, 0.176488, -0.05058, 0.105925, 0.270356, 0.162945, 0.178172, -0.005679, 0.181832, 0.160115, -0.166549, -0.043477, -0.0259, 0.117149, -0.09123, -0.015967, 0.166994, -0.362341, -0.299182, -0.158946, 0.199839, -0.182911, 0.000445, 0.233112, 0.216292, -0.20869, 0.10747, 0.188968, 0.027764, 0.086511, -0.009528, 0.128155, 0.130542, -0.203162, 0.083504, -0.126242, -0.04761, -0.055824, 0.137768, 0.283415, -0.194617, -0.13979, -0.1826, 0.091671, 0.027579], [0.061067, 0.295044, 0.16061, -0.150556, -0.177888, -0.131778, -0.048035, 0.089448, 0.027719, 0.046892, -0.093431, -0.064502, -0.084033, -0.005158, 0.140351, -0.042925, -0.017795, 0.120362, 0.135501, -0.042853, -0.148174, 0.225129, 0.127762, -0.125898, 0.204446, 0.254611, 0.100887, 0.296058, -0.107189, -0.207613, 0.114517, -0.11429, 0.299816, 0.273025, -0.233006, -0.088623, 0.022403, 0.10132, 0.371135, 0.347607, 0.138287, 0.217906, 0.353334, 0.113304, 0.204464, 0.019157, 0.014863, 0.07002, -0.069845, 0.007293, -0.072798, -0.278998, 0.004462, -0.238107, -0.054223, -0.147911, -0.102242, -0.234847, -0.019634, 0.172628, -0.004993, -0.483242, 0.140911, 0.227742], [-0.074648, -0.094373, -0.035619, 0.087342, -0.17072, -0.088507, 0.067788, 0.107029, -0.037069, -0.201651, 0.334162, 0.170984, -0.113307, -0.035696, 0.286829, 0.024417, -0.231691, 0.033352, -0.023025, -0.348976, -0.051606, -0.083079, -0.097408, 0.030111, 0.031624, -0.096321, -0.010467, -0.073554, 0.050037, 0.146383, -0.060631, -0.132357, -0.042964, -0.000494, -0.272345, -0.05322, 0.007061, 0.164678, 0.104683, -0.022075, 0.04823, -0.068273, -0.028553, -0.015342, 0.093737, 0.102366, -0.150706, 0.222934, -0.280802, 0.122474, 0.090158, 0.05506, -0.075305, 0.064257, -0.122334, -0.073044, -0.107056, 0.026527, -0.081599, 0.327425, 0.002005, 0.097326, 0.293249, 0.185422], [-0.14941, -0.027868, -0.113226, 0.146109, -0.159661, -0.11237, 0.077576, -0.076952, -0.42752, 0.241868, -0.11576, 0.152319, -0.331591, 0.474407, 0.120402, 0.246637, 0.19509, 0.536524, -0.082601, -0.29106, 0.139883, 0.222018, 0.540249, -0.032863, -0.195752, -0.24865, -0.200851, -0.251492, 0.019348, -0.070335, -0.013043, 0.102663, -0.032154, 0.25228, 0.145887, 0.096682, 0.098942, 0.061583, 0.280316, -0.22076, -0.082895, 0.232234, -0.160688, -0.217293, -0.101725, -0.118367, -0.217056, -0.064271, -0.111903, -0.140631, 0.045211, -0.343891, -0.183784, -0.260773, -0.048385, -0.024803, 0.10947, -0.031078, -0.101472, 0.251278, -0.007797, -0.183769, 0.096341, 0.016095], [-0.141433, -0.101476, 0.06634, -0.116149, 0.054804, -0.13921, -0.051892, -0.194687, 0.066447, -0.282205, -0.135535, 0.049895, -0.088701, -0.217952, 0.099157, -0.182425, -0.173113, -0.223869, -0.110282, -0.287721, 0.064165, -0.050007, -0.179881, -0.071955, -0.112921, -0.072648, -0.016481, 0.134124, 0.161179, 0.051335, -0.052148, -0.12356, -0.306525, -0.206969, -0.040612, -0.227914, -0.283702, 0.040108, 0.191714, -0.16234, -0.104893, -0.010933, 0.131916, 0.020494, 0.155958, 0.069187, -0.105744, -0.007942, -0.140719, -0.224416, -0.016375, -0.193583, -0.313144, -0.261498, 0.174189, 0.07127, -0.170676, 0.118492, 0.083106, -0.213973, -0.217162, 0.120413, 0.010984, -0.044189], [-0.09694, -0.01746, 0.398259, 0.339595, -0.307415, 0.287863, -0.297858, -0.166248, 0.137683, -0.021336, -0.088225, -0.03393, -0.15915, 0.005507, 0.178655, -0.245134, -0.000883, -0.03839, 0.08702, -0.18389, -0.054432, 0.283926, 0.028859, -0.126813, -0.135385, 0.244126, 0.251019, 0.1555, -0.139006, -0.043972, 0.33702, 0.300255, -0.349726, 0.274629, -0.006395, -0.09877, 0.163756, -0.149031, 0.123821, 0.108164, -0.078799, 0.046061, -0.069409, -0.198291, 0.269777, 0.448718, 0.046729, 0.07612, 0.500204, 0.17608, -0.16991, -0.275138, 0.150502, -0.06399, 0.099234, 0.125934, 0.04695, 0.188109, 0.090627, -0.181802, 0.368451, 0.108772, 0.046092, 0.35446], [-0.221772, 0.103464, 0.055245, 0.136739, 0.058077, -0.132223, -0.147486, -0.098422, 0.086624, 0.141613, -0.161325, -0.227987, -0.200412, -0.006023, 0.001239, 0.266764, 0.100876, 0.058828, -0.273346, -0.285283, 0.077559, 0.268186, -0.017697, -0.197206, -0.085, -0.193642, -0.00316, -0.259244, 0.079726, -0.138766, -0.139053, 0.060516, -0.119779, 0.133448, -0.059477, 0.034601, -0.023763, -0.123995, 0.021674, -0.156855, -0.362765, 0.162636, 0.158561, -0.079163, 0.16977, -0.006013, -0.09766, 0.006436, -0.350972, 0.19267, -0.259263, -0.768335, 0.036899, 0.067732, 0.217749, 0.06628, 0.229357, -0.095292, 0.18331, -0.097395, 0.060547, -0.023146, -0.082114, -0.2876], [0.054876, 0.307085, 0.208252, -0.039161, -0.209785, -0.107087, -0.084783, 0.153086, -0.105115, 0.208057, -0.365016, -0.101947, -0.218065, -0.264959, 0.061219, 0.058684, -0.019702, -0.007421, -0.114641, 0.145165, 0.073597, -0.109871, 0.215378, 0.004217, -0.053583, 0.062803, -0.099597, -0.175808, 0.024977, -0.017936, 0.060197, -0.030534, -0.126446, -0.025398, -0.084133, 0.016278, -0.212428, 0.215161, 0.047339, 0.101961, -0.100751, 0.111846, -0.197651, -0.120122, -0.07374, -0.107934, -0.063368, -0.019966, -0.182892, -0.116685, 0.022686, -0.227052, -0.097495, 0.018875, 0.053016, 0.030791, -0.072913, 0.030276, -0.136489, -0.230271, -0.111976, -0.070656, 0.138136, -0.031719], [-0.035834, 0.204636, 0.116134, 0.219322, 0.145623, -0.077745, -0.117337, -0.439244, -0.077032, 0.153178, -0.232148, 0.01186, -0.0642, 0.252524, 0.056661, 0.115273, 0.002041, 0.287053, -0.196928, -0.288734, 0.111381, 0.121441, 0.191912, 0.054726, -0.081907, 0.098505, 0.129339, -0.505231, -0.227654, -0.107094, 0.112815, 0.043117, -0.474656, 0.170049, -0.110748, -0.120652, 0.037506, -0.138297, -0.128528, -0.209763, -0.038529, 0.160891, 0.041744, -0.420084, 0.064865, -0.064656, -0.1634, -0.265362, -0.000705, 0.253063, 0.015756, -0.238901, 0.07473, -0.378294, -0.097034, 0.17542, 0.262143, -0.279498, 0.001113, 0.15967, 0.008989, 0.025815, 0.188078, -0.008711], [-0.278075, 0.164446, 0.181226, 0.059301, 0.015287, 0.141033, 0.120752, 0.200389, -0.272414, 0.06153, -0.024069, 0.446646, 0.026113, -0.019256, 0.150965, -0.01408, -0.089198, 0.016502, 0.175211, 0.045136, -0.096195, 0.664454, 0.42428, -0.111562, 0.122249, -0.090649, 0.058824, -0.19212, -0.189819, 0.367072, 0.109036, 0.302425, 0.290803, 0.230227, 0.306956, -0.250441, 0.352265, 0.376912, 0.58401, -0.030962, -0.132682, -0.111392, -0.143923, -0.349752, -0.07703, 0.207626, 0.163665, 0.009181, 0.079597, 0.062793, 0.170778, 0.070519, 0.031121, -0.078439, 0.064977, 0.170155, 0.557018, -0.20467, 0.249714, 0.077989, 0.159199, -0.182951, 0.139615, 0.27897], [0.245302, 0.150135, -0.048291, 0.070989, 0.025101, -0.088817, -0.073258, -0.462607, 0.423082, 0.34135, 0.000682, -0.226633, -0.026376, 0.021303, -0.209759, -0.00442, 0.073914, 0.004976, 0.066167, -0.105648, -0.054853, -0.261548, -0.212067, 0.025823, 0.057481, 0.228371, -0.001652, -0.27196, 0.000904, -0.041073, 0.020132, 0.067812, -0.394639, 0.195657, -0.121005, 0.201958, 0.186835, -0.107957, -0.28142, -0.307639, -0.120392, -0.149039, 0.247495, -0.261562, -0.11349, 0.035826, 0.130779, -0.237962, -0.370379, -0.075918, 0.062503, -0.04677, -0.11156, -0.115626, -0.1872, 0.070613, -0.179802, -0.276227, 0.245035, -0.032178, 0.178707, -0.079778, 0.309728, 0.159574], [-0.117796, 0.169293, 0.181407, 0.149589, 0.325796, -0.063148, -0.07283, 0.044025, -0.086994, 0.022001, 0.184826, 0.549846, -0.037741, 0.085382, -0.049002, -0.050188, 0.032708, 0.098342, 0.064681, 0.162391, 0.16822, 0.405438, 0.171106, -0.033606, 0.33494, -0.136036, 0.117362, -0.262173, -0.287506, -0.033533, 0.188586, 0.153594, -0.17187, 0.134624, -0.203676, -0.094916, 0.209892, 0.048417, 0.123897, -0.26216, 0.021632, 0.145872, 0.261432, -0.355358, -0.055672, 0.182546, -0.082479, 0.150193, 0.075296, 0.430505, -0.135199, -0.005791, -0.021857, 0.042373, 0.094133, -0.221573, 0.173235, 0.219248, 0.170938, 0.233583, -0.132685, 0.110673, -0.131548, 0.100277], [0.15624, -0.087851, -0.089233, -0.038597, -0.040547, 0.147373, -0.082924, -0.310166, 0.021278, 0.21471, 0.289177, -0.036443, -0.011386, 0.109038, 0.053584, -0.039171, 0.074987, -0.029191, 0.385303, 0.196006, 0.145773, -0.095698, 0.050288, 0.40682, 0.090175, 0.67815, 0.209713, -0.176463, -0.257242, -0.088162, 0.219776, -0.067874, -0.415808, 0.08141, -0.097868, -0.126425, 0.401235, -0.120327, -0.243158, -0.192333, 0.330724, -0.136412, 0.082477, -0.25716, -0.197476, 0.102144, -0.039197, -0.046914, -0.04249, 0.200883, 0.219188, 0.170543, -0.150907, 0.142744, -0.064044, -0.198791, -0.273386, -0.030672, -0.034654, 0.297067, 0.212938, -0.072759, 0.335342, 0.093837], [0.127834, -0.108777, -0.398505, -0.403847, -0.059408, 0.165068, -0.298889, -0.457402, -0.153116, -0.023698, 0.031167, -0.037759, 0.039504, -0.017828, -0.118496, -0.210028, -0.14738, 0.037054, -0.345449, -0.163658, 0.094628, -0.284685, 0.19006, -0.002544, -0.162691, 0.086082, -0.02275, -0.088974, -0.134641, -0.006589, -0.044568, -0.366057, -0.47557, -0.223196, -0.490799, -0.162358, 0.047768, -0.217104, -0.047532, -0.151158, -0.166677, -0.01524, 0.194059, 0.003773, 0.003569, 0.004743, -0.094417, -0.130891, -0.215954, 0.290356, 0.050964, -0.129351, -0.34629, -0.001212, -0.107257, -0.068036, -0.128562, 0.175717, -0.265947, 0.203222, -1.2e-05, -0.308319, 0.130711, -0.113951], [-0.244157, -0.315677, -0.138647, -0.382642, 0.054523, -0.051292, -0.084496, -0.181591, -0.053174, 0.210627, -0.086774, 0.130756, 0.087694, -0.029527, -0.096214, -0.067628, 0.090183, 0.034993, 0.197406, 0.199338, -0.02958, -0.37644, -0.014969, -0.328176, 0.094723, -0.330367, -0.025565, 0.066794, 0.422908, 0.090939, -0.191378, -0.147787, -0.013313, 0.048965, 0.003837, 0.11686, -0.275629, 0.198578, -0.006234, 0.40797, -0.18272, -0.017299, -0.175429, 0.057744, 0.184548, -0.115939, -0.083632, 0.08631, 0.046455, 0.01688, -0.691212, -0.275346, -0.021613, 0.166523, 0.32425, -0.172003, -0.071647, 0.340884, -0.40821, 0.222327, -0.17881, -0.240535, -0.2056, 0.142328], [-0.111347, -0.111105, -0.003381, 0.228269, 0.286241, -0.221009, -0.11134, -0.254623, 0.285593, 0.112563, -0.096721, 0.112572, 0.005367, 0.186217, -0.188881, -0.007619, 0.163204, 0.071933, -0.005973, -0.477362, -0.068067, -0.359108, -0.143587, 0.141826, -0.173018, -0.067152, -0.213393, -0.034722, 0.267145, -0.053112, 0.12621, 0.201723, -0.055844, 0.102619, 0.010996, -0.25725, -0.203715, -0.020571, 0.043565, -0.326987, -0.096355, 0.100761, 0.024376, 0.30973, 0.250714, 0.081021, -0.205758, 0.212488, 0.149993, -0.032083, 0.113364, -0.160406, -0.108783, -0.230675, -0.262433, 0.023015, -0.170114, -0.052947, 0.108431, 0.220938, -0.022661, -0.169673, 0.052472, 0.060688], [0.071345, 0.04295, -0.154205, 0.037229, -0.116717, -0.014103, 0.277448, 0.310156, 0.094386, -0.366292, 0.130812, -0.096935, -0.06395, 0.034941, 0.037878, -0.410882, -0.172483, -0.084325, 0.050624, 0.344397, -0.198078, -0.036936, -0.150802, -0.06923, -0.010813, 0.032438, 0.092668, 0.1166, 0.133878, -0.204112, 0.020116, -0.139348, 0.114742, -0.014724, -0.017194, -0.073251, -0.171031, -0.133311, -0.151811, 0.200274, -0.031467, -0.048483, -0.158225, 0.074275, 0.096072, -0.044635, -0.08366, 0.067258, -0.072977, -0.278135, -0.112398, 0.065106, 0.016054, 0.023931, -0.188798, -0.149597, -0.241077, 0.272756, 0.145457, -0.260338, -0.082279, 0.022817, -0.247137, -0.086653], [-0.137985, 0.105931, -0.083422, -0.021442, 0.494803, -0.051209, 0.25387, 0.117216, -0.557288, 0.011998, 0.008626, -0.007532, -0.12628, -0.143246, 0.227425, 0.058227, -0.47159, -0.097913, -0.207206, 0.050151, 0.12336, 0.31244, 0.074024, 0.263121, 0.366925, -0.182122, 0.173274, -0.126269, -0.31969, 0.23376, 0.059631, 0.01267, -0.163039, 0.079486, -0.156398, -0.111324, 0.073388, 0.005613, 0.337843, 0.007643, -0.190793, 0.106256, 0.205141, -0.413862, -0.303135, 0.006812, 0.009718, 0.05161, -0.122448, 0.045425, 0.543085, 0.017559, -0.038579, 0.077275, 0.453254, -0.037484, 0.488102, 0.031584, 0.24655, 0.11701, -0.207948, 0.029949, -0.065057, -0.152107], [0.14032, -0.022601, 0.159975, 0.201817, -0.09043, 0.101044, 0.087054, 0.463483, 0.086273, 0.185556, 0.047526, 0.022419, -0.003132, -0.007539, 0.202548, -0.114188, 0.164159, -0.145482, 0.115636, 0.17576, -0.138451, 0.006142, -0.129466, -0.133863, -0.001124, -0.023925, 0.087798, 0.116482, 0.26443, -0.142218, 0.258969, 0.002698, 0.039875, 0.276783, 0.075188, 0.138584, 0.156189, 0.003929, 0.047234, 0.076701, 0.058242, 0.111055, -0.286037, -0.124289, 0.048275, 0.011659, -0.137874, 0.286155, 0.119881, -0.050757, 0.065567, 0.033985, 0.084208, -0.058524, -0.183519, -0.066892, -0.02918, -0.140435, 0.166273, 0.070501, 0.236222, 0.246117, 0.107393, 0.33025]], [-0.272609, -0.158619, -0.030815, -0.239183, 0.185318, -0.252381, -0.17945, -0.282783, -0.069886, 0.06002, -0.10192, -0.262473, -0.114739, 0.075848, -0.145496, -0.070933, -0.045438, 0.258379, -0.490246, -0.578544, -0.237008, -0.068711, 0.161412, -0.198142, 0.056012, -0.211943, -0.25178, -0.07668, -0.187349, -0.071111, -0.120714, -0.105791, -0.243468, 0.043301, -0.003144, -0.183598, -0.133706, -0.11168, -0.152413, -0.130554, -0.350507, 0.199713, 0.204122, 0.039944, -0.060379, -0.016521, 0.050301, -0.247477, -0.209724, 0.089978, -0.112398, -0.288363, -0.311578, -0.260528, 0.25916, 0.005903, 0.224152, -0.136045, -0.134596, -0.171829, -0.279957, 0.101125, -0.298636, -0.201686]),
    ([[-0.087747, -0.019001, -0.158563, -0.012486, -0.128532, 0.105776, -0.007688, 0.049188, -0.665582, -0.092064, -0.071464, 0.017143, 0.045114, -0.268634, 0.043122, 0.07083, 0.08071, 0.020798, 0.180362, -0.014404, -0.052827, -0.061757, -0.02059, 0.189634, 0.061196, 0.192411, -0.099437, 0.058356, -0.054874, 0.069735, -0.021107, 0.064835], [-0.18906, -0.117787, 0.039758, 0.029988, 0.070601, 0.040928, 0.018663, -0.074424, 0.068724, 0.179051, 0.055368, -0.115968, -0.054865, -0.113546, -0.142023, 0.15825, 0.097928, 0.16446, 0.082453, 0.070797, -0.014056, -0.050471, -0.118489, -0.064569, 0.163625, 0.167232, -0.132512, 0.116078, -0.138953, -0.011335, 0.036158, 0.07589], [-0.264563, 0.083782, 0.104466, 0.044405, 0.060405, -0.053851, -0.257895, 0.018601, 0.180115, 0.169293, -0.042554, 0.247468, 0.050162, 0.098451, -0.159007, 0.137584, -0.035438, -0.28724, 0.128242, 0.182904, 0.031309, -0.081542, 0.029316, -0.003931, -0.033321, -0.068004, 0.114731, 0.045245, 0.080821, -0.114507, -0.052706, 0.164437], [-0.356582, -0.026453, 0.108364, 0.044186, 0.172892, -0.100397, 0.232703, 0.056871, -0.230458, 0.107381, 0.119205, 0.252312, 0.024446, -0.113827, -0.101507, -0.096355, 0.162293, -0.172017, -0.049131, 0.090489, 0.00678, 0.074735, -0.024277, 0.009417, -0.264951, -0.103516, 0.011888, -0.017581, 0.006027, 0.12147, 0.024418, 0.005481], [0.143111, 0.064635, 0.001119, -0.106623, 0.097714, 0.163142, 0.264744, 0.038948, -0.116149, -0.228456, -0.119859, -0.206707, -0.051688, -0.12607, 0.033925, 0.038624, 0.098851, 0.101646, 0.191216, 0.146015, -0.062839, -0.053241, 0.097706, 0.084887, 0.059009, 0.022448, 0.056208, -0.056241, 0.011434, 0.101572, 0.100914, 0.031013], [0.081271, 0.002883, 0.097886, 0.047964, 0.053834, -0.1814, -0.177391, -0.000902, 0.008507, 0.01203, 0.207646, -0.193024, 0.10929, 0.136091, 0.027243, -0.026488, -0.086538, -0.017208, 0.1721, 0.073634, 0.104737, -0.076712, -0.02613, -0.024817, 0.100479, -0.136147, 0.096735, 0.159643, -0.032088, -0.058289, -0.047629, 0.087204], [-0.019699, 0.178793, -0.099273, -0.044236, -0.093071, 0.031385, 0.021249, 0.016414, -0.211878, 0.209642, 0.230502, 0.054941, 0.205171, -0.006894, -0.09397, -0.012752, 0.087786, 0.084119, 0.236129, 0.219384, 0.161536, 0.116105, 0.083435, -0.296158, 0.101085, 0.06565, -0.135062, -0.10397, 0.098532, 0.095669, 0.062186, 0.229395], [-0.477171, 0.152348, -0.004371, 0.140313, 0.101224, 0.011674, -0.198185, -0.000808, 0.000707, 0.235129, 0.331384, -0.379806, 0.155583, 0.016226, 0.055526, -0.17059, -0.207996, -0.0007, -0.158879, 0.126297, -0.069849, 0.004961, -0.170546, 0.001824, -0.097814, -0.246155, 0.049227, -0.146333, 0.157542, -0.128521, 0.07909, 0.12132], [-0.000501, -0.271196, 0.128548, -0.16488, -0.121609, 0.059065, -0.088809, 0.073546, -0.36653, -0.064373, -0.048325, 0.013293, -0.299332, -0.087008, -0.126311, 0.06281, -0.147988, 0.099407, 0.140165, -0.071596, -0.220578, 0.123626, -0.092784, 0.176321, 0.130625, 0.079937, -0.032764, 0.106704, -0.301598, 0.180563, -0.151904, -0.202767], [0.188215, 0.017099, 0.128914, -0.054334, 0.188995, 0.017358, 0.118361, 0.066572, -0.149314, -0.11344, 0.108227, -0.310472, 0.09922, -0.083301, 0.083669, 0.128016, -0.033506, 0.0386, -0.047684, 0.025488, 0.136779, 0.150786, -0.086124, -0.23728, 0.068062, 0.064581, 0.166444, -0.114147, 0.056574, 0.080412, -0.08399, 0.034147], [0.134737, -0.163595, -0.08332, 0.052981, 0.08575, 0.080062, 0.205015, 0.015446, 0.171841, -0.150828, -0.114402, -0.355742, 0.038416, 0.307853, -0.16256, -0.03789, 0.137229, 0.072961, 0.055511, 0.004128, -0.000151, -0.107417, -0.037493, -0.140477, -0.055421, 0.114563, -0.044663, -0.09234, -0.057696, 0.221512, -0.04158, -0.071969], [0.209456, -0.024217, 0.251482, -0.10604, -0.111846, 0.206146, 0.114933, -0.003386, -0.180303, -0.235024, -0.379383, 0.104742, -0.314319, 0.044519, -0.075323, -0.077208, 0.008443, -0.019565, 0.019598, -0.143438, -0.065451, -0.037333, 0.157132, -0.186473, -0.015023, 0.097939, -0.296427, -0.125039, -0.337375, 0.07804, -0.17086, -0.354735], [0.098293, -0.024109, -0.14061, 0.12515, -0.311197, 0.024882, 0.0849, 0.078466, 0.194356, 0.148021, 0.012361, 0.358375, 0.024314, -0.210022, -0.170847, -0.047943, 0.047986, -0.020478, 0.115459, -0.04558, 0.039065, -0.06251, -0.072541, 0.056812, -0.078133, 0.078945, -0.212694, -0.076506, 0.104839, -0.001958, 0.022308, -0.132713], [-0.159277, 0.063849, 0.127145, -0.068699, -0.080078, 0.015047, 0.076177, -0.018469, -0.025653, -0.234487, 0.043102, 0.052463, -0.112121, -0.281, 0.024229, -0.035196, -0.042572, -0.239698, -0.135839, 0.194379, 0.119268, 0.106362, 0.022858, -0.060087, -0.066338, 0.002966, 0.104708, 0.091185, 0.000488, -0.055969, 0.071501, -0.009152], [0.044818, 0.084423, -0.126233, 0.075617, -0.04232, -0.21355, -0.147139, -0.165552, 0.235551, 0.112128, 0.113641, 0.198011, 0.14313, 0.00817, -0.114957, 0.198538, -0.170035, -0.192823, -0.006433, 0.048463, 0.032858, 0.037746, 0.076788, 0.118983, 0.118209, -0.258217, 0.05785, 0.233324, 0.151432, -0.031497, 0.119336, 0.11491], [0.169485, 0.131035, 0.066518, 0.016149, 0.102424, 0.118569, -0.059582, -0.005138, 0.033324, -0.057273, -0.064008, 0.287531, 0.117176, -0.392168, 0.050261, -0.075516, -0.097991, -0.128296, -0.398942, 0.031969, 0.143541, -0.013566, -0.259594, -0.100662, -0.077558, -0.052039, 0.189247, -0.091188, 0.060659, 0.168577, 0.185131, -0.030387], [-0.01945, 0.091265, 0.063335, -0.038858, 0.249032, 0.006851, 0.105465, -0.112788, -0.111653, 0.037027, 0.066512, 0.145284, 0.049689, -0.066052, 0.040341, -0.049611, 0.040972, -0.033045, 0.013772, 0.013029, 0.079834, 0.143237, -0.036926, -0.136107, -0.04258, -0.167086, 0.048834, 0.009322, -0.020407, -0.11126, 0.217864, 0.108774], [-0.099448, -0.006972, -0.08715, 0.01223, 0.017094, 0.0409, -0.087202, -0.069165, -0.336045, -0.417316, -0.133879, 0.369356, 0.054722, -0.359449, -0.10799, 0.061599, -0.121906, 0.009233, 0.054444, 0.094407, 0.101333, -0.002828, 0.031445, 0.046239, 0.116247, -0.054752, -0.128843, 0.03092, 0.173044, -0.046677, 0.059849, -0.161545], [0.233813, -0.169392, -0.033474, -0.146371, 0.051374, 0.127942, -0.025472, 0.088013, 0.105062, 0.32821, 0.123064, 0.145781, -0.025462, -0.082578, -0.056747, -0.003626, 0.090309, 0.113328, 0.260818, -0.127887, -0.244696, -0.073283, 0.196722, -0.00278, 0.258525, 0.001743, -0.030185, 0.152006, -0.11842, 0.219581, 0.113754, -0.228943], [-0.156796, -0.241116, -0.203771, -0.136788, -0.100214, 0.045115, -0.124404, -0.027474, 0.371407, -0.370093, -0.077891, -0.129141, -0.258525, 0.282555, -0.035622, 0.174716, -0.203467, -0.008444, 0.262553, -0.20467, -0.202529, -0.072767, -0.05114, -0.069328, 0.399796, -0.136607, -0.207192, 0.153732, -0.103545, 0.079365, -0.139592, -0.097986], [0.110351, -0.014616, -0.154544, -0.115858, -0.387576, 0.0928, 0.040204, 0.017888, 0.022972, 0.108066, 0.072733, 0.109053, 0.050193, 0.029741, 0.024345, 0.137043, 0.104063, 0.132536, 0.141872, -0.078087, -0.011804, -0.151968, 0.090178, 0.203684, 0.032655, -0.009017, -0.221056, 0.005518, 0.020285, 0.106228, -0.032423, 0.086336], [0.033527, -0.084127, -0.029806, 0.034533, -0.057119, 0.056589, -0.290746, -0.201757, -0.01715, 0.16069, 0.045218, -0.066588, 0.084329, -0.317221, -0.055883, -0.080395, -0.059911, 0.182468, -0.447312, 0.178192, 0.011681, -0.03761, -0.073311, 0.146982, 0.085836, 0.057816, 0.133753, -0.162754, 0.018951, -0.218682, 0.100155, 0.180168], [-0.442463, -0.122126, 0.09501, -0.073671, 0.052402, 0.130367, -0.100345, -0.103165, -0.017425, -0.167465, -0.080644, -0.566035, -0.034673, -0.19385, 0.061042, -0.06328, -0.231871, 0.136764, -0.303934, 0.110807, 0.055417, 0.027092, -0.035029, 0.243106, 0.126695, 0.118992, 0.083712, -0.073574, 0.104375, -0.059684, 0.098284, -0.268951], [-0.026735, 0.149096, 0.103818, -0.087088, 0.153987, -0.150386, 0.098317, 0.208331, 0.124606, 0.138525, -0.040893, -0.429104, 0.093845, 0.052509, -0.038394, 0.093975, -0.027068, 0.168087, 0.311074, -0.082541, -0.017538, 0.185351, 0.246874, -0.308993, -0.016987, 0.107007, 0.104141, 0.195254, 0.172378, 0.069115, 0.125228, -0.01916], [0.087016, 0.160337, 0.004538, 0.087182, -0.079536, -0.082419, -0.217865, -0.092108, 0.07515, -0.040333, 0.037761, 0.002785, -0.006115, -0.063612, -0.175632, 0.002808, -0.172029, -0.045245, 0.19639, -0.072647, 0.070239, -0.035323, -0.188565, -0.12769, 0.206143, 0.044439, 0.194325, 0.019825, -0.020348, 0.109693, 0.105039, 0.074032], [0.171488, -0.278985, 0.038771, -0.070424, -0.29028, -0.162899, 0.032117, 0.177493, 0.035362, -0.158129, -0.119414, 0.210069, -0.047362, -0.123782, -0.124896, 0.136303, 0.066324, -0.019423, 0.032724, -0.153278, -0.173903, 0.061166, 0.064692, -0.282972, -0.096585, 0.017734, 0.178592, 0.202433, -0.123695, 0.105717, -0.179719, -0.392114], [-0.044917, 0.022611, 0.068752, -0.068141, -0.115517, 0.06556, 0.016996, -0.14329, 0.295849, 0.111557, 0.040616, 0.043477, 0.142036, 0.088478, -0.087073, 0.166019, -0.289701, 0.0738, 0.170263, 0.078016, 0.111959, 0.052016, -0.219831, 0.05732, 0.203314, -0.166132, -0.067088, 0.183202, 0.056496, -0.149322, 0.1243, 0.122777], [-0.330746, -0.072153, 0.030954, 0.014625, -0.025457, 0.091826, -0.167654, 0.04593, -0.107787, 0.130565, 0.012872, 0.003098, -0.04523, -0.089111, 0.014786, -0.270601, -0.134681, 0.086816, -0.092217, -0.055939, 0.017229, -0.052443, 0.094157, -0.144317, 0.015983, -0.078971, -0.223546, -0.173262, -0.137822, -0.271645, 0.034914, 0.041316], [-0.067554, -0.040954, 0.073966, -0.02808, -0.006405, -0.139119, -0.278034, 0.008442, -0.142023, 0.143867, 0.063797, -0.463533, 0.069737, -0.003099, -0.035358, -0.041589, -0.142185, -0.15303, -0.216531, 0.051875, 0.0159, 0.082791, 0.067689, -0.073932, -0.138668, -0.233822, -0.021301, -0.159444, 0.030911, -0.40437, -0.084726, 0.012463], [0.369461, 0.05563, 0.018376, -0.016187, 0.01945, -0.005804, -0.202058, 0.0503, 0.205346, 0.122473, -0.095734, 0.230037, 0.018731, 0.076819, -0.034443, -0.003244, -0.160065, -0.133086, -0.180583, -0.035917, 0.077177, 0.086296, -0.04367, 0.013528, 0.080836, -0.153356, -0.071719, 0.086288, 0.087654, 0.175487, 0.099052, -0.0996], [-0.013332, 0.030275, 0.082627, 0.075129, 0.04996, 0.1926, 0.065311, 0.115202, 0.077809, 0.265697, 0.000836, -0.447121, 0.128308, 0.130856, -0.111212, 0.216818, -0.052014, 0.074253, 0.099165, 0.074038, 0.14112, 0.049367, -0.146448, 0.052612, 0.249295, 0.180627, -0.03659, 0.1109, -0.036964, 0.178604, 0.085167, 0.051975], [0.110962, 0.103563, -0.071264, -0.035071, 0.083891, 0.046721, -0.082176, -0.090968, -0.065454, 0.102799, -0.062047, 0.096108, -0.03265, -0.204774, -0.071902, -0.138846, 0.162678, -0.165032, -0.127914, -0.029083, 0.128588, 0.134857, 0.032551, 0.128874, -0.123635, -0.136824, 0.097167, -0.092666, -0.016931, -0.211137, 0.024888, 0.026728], [0.025753, -0.060417, -0.013432, -0.007068, -0.100503, -0.41147, -0.338731, -0.166895, 0.240621, -0.188351, -0.184008, -0.490153, -0.048842, 0.159292, -0.047109, -0.058839, -0.274792, -0.075936, -0.046988, 0.067629, -0.157136, 0.002199, -0.282668, -0.220373, -0.133328, -0.213666, -0.018363, -0.131568, 0.018769, 0.129741, 0.054936, 0.033892], [0.053963, -0.027464, 0.085556, 0.073567, 0.122731, 0.094142, 0.018167, 0.16332, 0.167362, 0.014892, -0.020957, -0.076559, 0.032196, -0.11855, -0.105723, 0.187378, -0.064449, 0.193349, -0.23075, -0.093086, -0.085736, -0.089014, -0.01934, 0.003626, -0.035229, 0.150017, 0.113288, 0.170923, 0.047005, 0.017359, 0.093889, -0.080775], [-0.099146, 0.007935, -0.073933, -0.080903, -0.201616, -0.141232, -0.002317, 0.051611, -0.3964, -0.011112, -0.084585, 0.20384, 0.008291, -0.199998, -0.147849, -0.207375, 0.075864, -0.166689, -0.211557, 0.028614, -0.006902, 0.172858, -0.000543, -0.057762, -0.151018, -0.073302, -0.099469, -0.02757, 0.031984, -0.401432, 0.068624, -0.100944], [-0.365062, 0.183429, 0.147528, -0.011574, 0.050813, -0.206641, 0.095169, -0.129525, -0.097386, 0.064397, 0.006508, 0.173949, 0.042884, 0.175303, -0.179979, -0.008256, 0.173216, -0.198432, 0.148326, 0.154251, 0.011181, 0.0414, 0.07945, -0.213267, -0.076142, -0.239215, 0.132279, -0.077154, 0.092231, 0.040447, 0.105305, 0.049434], [0.015998, 0.162695, 0.332322, 0.032531, 0.142669, 0.060423, 0.191601, 0.10446, -0.338525, 0.233352, -0.033751, 0.077673, 0.199005, -0.104101, -0.138116, 0.034759, 0.225449, 0.006658, 0.109956, 0.075958, 0.166598, 0.19554, -0.003732, -0.308995, 0.075606, -0.093373, 0.27115, -0.042532, 0.149765, 0.103011, 0.236346, -0.058884], [0.081181, 0.138335, -0.141637, -0.047316, -0.286278, 0.075301, 0.117655, 0.128373, 0.156426, 0.19478, 0.201186, 0.024202, 0.072253, 0.255805, -0.120507, 0.094523, -0.108857, 0.172388, 0.245908, 0.112906, 0.03677, 0.077931, 0.082864, -0.19024, 0.043324, 0.044698, -0.030532, -0.006728, 0.084357, 0.061395, 0.054276, -0.017367], [0.18645, 0.197983, 0.021975, 0.074428, 0.282139, 0.076714, 0.005419, 0.071932, -0.438647, 0.17994, -0.016503, 0.020241, 0.149327, -0.034183, -0.038335, -0.105333, -0.012262, 0.17582, 0.054302, 0.113167, 0.058177, 0.055713, 0.081134, -0.142226, -0.009823, 0.095068, 0.199382, -0.096057, 0.000917, -0.003827, 0.20395, 0.320608], [-0.098272, 0.095835, 0.127048, -0.035819, -0.132084, -0.217105, -0.228417, -0.089202, 0.164555, -0.299976, -0.010857, -0.082935, -0.100986, 0.073642, -0.020411, 0.095195, -0.199244, -0.168275, -0.250845, -0.015917, -0.041641, -0.127775, -0.163345, 0.100352, 0.101678, -0.254748, -0.169885, 0.166431, -0.210238, 0.049165, -0.016902, 0.02374], [0.055464, -0.212193, -0.064027, -0.102915, -0.271145, -0.081625, 0.049193, 0.098029, 0.02632, 0.24537, -0.046585, -0.2274, -0.184743, 0.223532, 0.03174, 0.222879, -0.037302, 0.108938, 0.057211, -0.025769, -0.113298, -0.185094, 0.128775, -0.003044, -0.002338, 0.109243, -0.21015, 0.1138, -0.214174, 0.017207, -0.08798, -0.397155], [-0.032263, -0.092865, 0.085086, 0.047686, 0.141603, -0.126274, -0.028057, -0.013232, -0.195352, -0.185398, -0.075286, -0.020877, -0.052712, 0.068042, -0.187364, -0.114391, 0.003116, 0.111098, -0.176623, -0.081247, 0.097978, 0.09913, -0.123464, -0.08339, -0.114856, 0.02423, -0.000657, 0.154607, 0.02699, -0.292465, 0.000228, 0.03381], [-0.144227, -0.0589, 0.02084, 0.04681, -0.273264, 0.069499, 0.087927, 0.152799, -0.111145, -0.397275, -0.069884, -0.116569, -0.032381, -0.029691, -0.017487, 0.067579, -0.070872, 0.054361, -0.051944, 0.022302, 0.021788, -0.011995, 0.046676, 0.15362, -0.035534, 0.171283, 0.157209, 0.132361, 0.000878, -0.154706, -0.027703, -0.099495], [-0.07663, -0.116309, 0.12163, 0.077398, -0.03021, -0.107062, -0.138175, -0.133056, -0.04053, 0.048172, -0.04397, -0.111076, -0.066895, -0.30854, -0.023551, -0.191336, 0.031281, -0.312289, -0.181759, 0.041957, -0.042342, 0.133538, 0.086354, -0.197008, -0.307711, 0.124187, -0.053284, -0.265444, -0.019119, -0.105356, -0.097414, 0.071454], [0.126993, 0.056839, -0.025103, -0.095894, 0.080543, -0.175082, -0.497614, -0.089485, 0.061491, 0.099272, 0.000146, 0.140611, -0.031512, -0.16123, 0.049832, 0.004704, -0.150019, -0.083263, -0.207035, 0.129329, -0.050504, -0.086379, -0.153168, -0.096783, -0.049387, -0.312092, -0.085726, -0.131873, 0.022932, 0.013136, 0.075189, -0.034172], [-0.224297, 0.084706, 0.058205, 0.094499, -0.01834, -0.021599, -0.312692, -0.042392, -0.104659, 0.162684, 0.057336, -0.088735, 0.009466, -0.021045, -0.018274, -0.113193, -0.222477, 0.015999, 0.068937, -0.044737, 0.071234, 0.009044, -0.04961, 0.288319, 0.088181, 0.113287, 0.070821, 0.034033, 0.01176, -0.298337, -0.024741, 0.046493], [-0.132932, 0.018074, 0.09851, -0.112277, -0.063465, 0.045985, -0.103779, 0.03375, 0.00299, -0.109368, -0.016133, -0.125347, 0.027155, 0.084844, 0.01067, 0.092277, 0.053101, 0.056696, 0.017459, -0.019009, -0.034963, 0.095028, -0.113208, 0.221544, -0.010858, 0.127449, -0.137575, 0.056595, -0.066182, 0.002517, 0.116412, -0.264138], [-0.071725, -0.087423, -0.034317, -0.057876, 0.040272, 0.095063, 0.124512, 0.134716, 0.249228, 0.170047, 0.148278, -0.056538, 0.088813, -0.184268, -0.154693, -0.001596, 0.085066, 0.023184, 0.000592, -0.01452, -0.066557, 0.070945, 0.027276, -0.29951, -0.06475, 0.221114, 0.192005, 0.01377, 0.140633, -0.028145, -0.039699, -0.003028], [-0.495099, 0.127521, -0.001916, 0.031092, -0.008284, -0.198393, -0.107806, -0.253375, -0.263572, 0.398135, 0.166442, 0.215747, 0.101039, 0.066417, -0.055489, 0.077254, -0.049223, -0.046958, -0.207953, -0.129705, -0.117207, 0.166351, -0.040469, 0.1187, 0.057767, 0.134183, 0.044901, -0.110583, -0.181758, -0.509467, 0.096892, 0.054422], [-0.163432, -0.010727, 0.085897, 0.083282, 0.052107, 0.105164, 0.227341, -0.074589, -0.112847, 0.051273, -0.114755, -0.05907, 0.148208, -0.131179, 0.044534, 0.055666, 0.21996, -0.099006, -0.12618, 0.123219, 0.100052, 0.181907, -0.140068, 0.097002, 0.101204, 0.115908, 0.100832, 0.083507, 0.12858, -0.038126, -0.000518, -0.070131], [0.187201, -0.08391, 0.069861, 0.071412, 0.128158, 0.195518, 0.120531, -0.059934, -0.104405, 0.039984, -0.235277, -0.306603, -0.02487, -0.28569, 0.029965, 0.100314, 0.209029, 0.182313, 0.285369, -0.152757, -0.033783, -0.017367, 0.148428, -0.009001, 0.19148, 0.139771, 0.088736, -0.107111, 0.110236, 0.071607, -0.065871, -0.119106], [-0.188861, 0.067019, -0.196617, 0.117849, -0.417552, 0.135138, 0.033373, -0.090948, 0.024283, 0.293814, 0.22535, -0.067678, -0.026139, 0.049796, -0.146175, -0.000809, 0.1067, 0.14684, -0.008632, -0.063604, -0.065354, -0.069484, -0.054499, 0.136004, 0.186498, 0.095101, -0.126727, 0.121338, -0.006445, -0.007323, 0.017937, 0.298795], [0.093359, 0.043831, 0.085036, -0.056942, 0.040619, -0.08652, -0.042008, 0.086548, 0.106192, 0.094243, 0.108928, 0.305182, -0.080897, 0.153594, -0.070158, -0.062954, 0.136742, 0.020077, 0.229368, -0.015128, 0.03495, -0.049713, 0.023093, -0.204961, -0.047492, -0.147765, -0.014768, 0.198861, -0.022148, 0.036681, 0.110383, -0.006766], [-0.199169, -0.066053, 0.003685, 0.007048, 0.043753, -0.033871, 0.005957, 0.017986, 0.121419, 0.118442, 0.073997, -0.240424, -0.074982, 0.055049, -0.031705, -0.027874, 0.21378, -0.190062, 0.228446, -0.027619, 0.100672, 0.057271, -0.100199, -0.29083, -0.02949, -0.008891, -0.108985, -0.14932, 0.010359, 0.10763, -0.013485, -0.025353], [-0.024936, 0.120303, -0.003953, -0.082867, 0.121473, -0.196711, -0.067658, -0.002326, 0.093466, -0.179304, -0.098608, -0.098332, 0.070093, 0.123028, 0.034917, -0.028729, 0.033352, 0.082384, -0.228089, 0.102564, 0.172782, 0.009486, 0.017225, -0.050835, -0.238928, -0.020272, -0.012773, 0.014493, 0.022267, -0.130556, 0.108415, -0.020207], [-0.192154, -0.044905, -0.088998, -0.035322, 0.004319, -0.114462, -0.332189, -0.200969, -0.0972, 0.030094, 0.07044, 0.054545, 0.041435, 0.03739, -0.036486, -0.147431, -0.081132, -0.33697, -0.012737, -0.024556, 0.119431, 0.049423, -0.008805, 0.371719, 0.04286, -0.174062, 0.053676, -0.019819, 0.144836, -0.220366, -0.030973, -0.065271], [-0.266703, 0.150659, -0.045977, 0.128118, -0.026405, -0.098407, 0.062766, 0.017912, -0.332902, 0.029776, 0.105898, -0.54707, 0.228976, 0.2236, -0.051327, 0.112254, -0.023229, -0.167573, 0.022948, 0.043923, -0.021222, 0.056707, -0.255388, 0.082762, -0.135061, -0.120297, 0.005717, -0.187221, 0.094713, 0.167087, 0.078631, 0.315891], [-0.013245, 0.081454, -0.010784, -0.031812, 0.101562, 0.047168, -0.017815, 0.123799, -0.158725, -0.314594, -0.136307, 0.059184, -0.379991, 0.134074, -0.097424, -0.009052, -0.040443, 0.134345, 0.075468, -0.14157, 0.058475, 0.047486, -0.150509, -0.069616, -0.048811, 0.001401, -0.254812, 0.075054, -0.351249, 0.170879, -0.020141, -0.363251], [0.294425, 0.037993, -0.007827, -0.12115, 0.014764, 0.146973, 0.233628, 0.167848, 0.120617, 0.11239, -0.030794, -0.234536, -0.118898, 0.138686, -0.062307, 0.105392, 0.298188, 0.115416, 0.183518, -0.085833, 0.024433, 0.059946, 0.045616, -0.141853, 0.126456, 0.01231, 0.022494, 0.172535, -0.096026, 0.128831, -0.050919, 0.03367], [0.046507, 0.097285, -0.049811, -0.020567, -0.044264, -0.060353, -0.051268, -0.12142, -0.224337, 0.180614, 0.021707, 0.270782, 0.05053, -0.189825, 0.025036, -0.199456, -0.095584, -0.090324, -0.223081, 0.069323, 0.089883, 0.110019, -0.112084, -0.149257, -0.220058, 0.052035, 0.028486, -0.050587, 0.124027, -0.067311, 0.141553, 0.215842], [-0.027935, -0.072542, 0.114388, -0.069398, 0.038637, 0.075274, 0.015089, 0.145377, -0.053067, 0.214694, -0.086926, 0.012043, 0.123968, 0.165153, -0.106361, 0.02546, 0.080757, 0.100521, 0.102529, 0.103446, -0.011686, 0.034303, -0.144008, 0.00764, 0.098321, -0.068408, 0.025435, 0.050857, 0.017055, 0.03137, -0.011778, 0.103896], [-0.013924, -0.010406, 0.002742, 0.071498, -0.281178, -0.128919, 0.039512, 0.091624, 0.120469, -0.098973, 0.14054, -0.142386, -0.089268, -0.194218, -0.080345, 0.049303, -0.143014, -0.249543, -0.269072, -0.070587, 0.054668, 0.112078, -0.07818, -0.224431, -0.2174, -0.007061, -0.192804, 0.064681, 0.007915, 0.098629, -0.056544, 0.063696], [0.20324, -0.145347, -0.003, -0.012523, -0.165009, 0.006727, 0.170306, 0.014878, -0.45503, -0.058019, -0.003138, -0.07587, 0.095686, -0.506523, 0.085452, -0.049853, 0.005076, 0.067178, -0.021862, 0.13432, -0.05521, -0.036203, 0.169776, -0.480389, 0.089783, 0.108094, 0.009921, -0.061088, 0.038851, 0.13162, -0.001747, 0.083358], [0.10484, -0.14478, 0.132088, -0.063981, -0.252125, 0.109149, 0.176481, 0.069243, -0.204515, 0.28598, 0.045325, 0.204102, 0.005826, -0.122969, 0.07963, -0.102972, 0.128624, 0.040922, 0.156591, 0.129248, -0.085273, 0.018152, -0.031528, -0.341596, 0.089619, 0.126273, -0.087081, -0.144294, 0.034049, -0.030158, 0.142015, -0.442306]], [-0.160225, 0.104853, 0.084222, 0.086696, 0.054901, -0.002333, 0.086008, 0.077079, 0.047044, -0.334917, -0.036152, -0.254796, 0.015861, 0.079248, -0.108412, 0.149909, 0.118116, 0.00623, -0.336238, 0.041357, 0.109284, 0.05244, 0.001958, 0.021495, -0.142032, 0.056946, 0.050077, 0.030501, -0.066819, -0.204363, 0.090976, -0.110849]),
    ([[0.443856], [-0.164521], [-0.097964], [0.001923], [-0.18933], [0.210491], [0.285968], [0.165306], [0.270545], [-0.471038], [-0.135908], [0.388973], [-0.088613], [0.342851], [-0.034097], [0.17324], [0.252053], [0.273725], [0.311408], [-0.116402], [-0.057782], [-0.086015], [0.108639], [0.146488], [0.213835], [0.123912], [-0.128472], [0.11066], [-0.073726], [0.201869], [-0.114903], [-0.13586]], [0.091589]),
]

# exp_r3_qgru: 500 ms quote head on the taker GRU state (research/diag/seqq_r3.py head, export_head); holdout corr 0.490 vs 0.477 for MLPQ[500].
SEQQ_XM = [0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233398, -0.0, -0.0, 5.906047, 5.906341, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.057889]
SEQQ_XS = [1.019985, 0.879362, 0.830525, 0.723134, 0.628317, 2.280914, 12.974841, 0.673973, 10.694654, 0.757209, 0.651131, 0.54564, 1.651657, 1.65166, 0.999999, 4.399385, 1.056378, 0.427418, 127.455505, 323.712799, 938.328491, 1.343882]
SEQQ_YSC = 2.618162155151367
SEQQ_HEAD = [
    ([[0.059518, -0.24212, 0.055314, 0.019285, 0.020806, -0.149459, 0.070098, -0.165548, 0.02589, -0.275875, 0.046002, 0.043426, -0.072471, -0.04046, -0.118933, -0.062362, 0.073456, -0.174803, 0.144345, 0.032109, 0.008313, -0.099883, -0.095792, 0.332169, 0.086387, -0.089097, 0.188294, -0.071253, -0.060531, 0.063431, -0.045916, 0.06205, -0.142972, -0.014195, -0.113461, -0.004988, -0.130699, -0.161773, 0.022458, 0.080237, 0.247737, 0.164528, -0.076497, 0.043459, 0.114146, 0.053891, 0.036144, -0.102244, 0.00045, -0.166534, 0.062104, -0.136316, -0.053066, 0.083278, 0.041703, 0.146831, -0.00022, -0.079298, 0.101497, 0.208455, 0.163254, -0.090132, 0.028595, 0.002928], [0.011201, -0.046875, -0.104546, -0.051054, -0.1547, -0.122968, -0.038168, -0.063214, 0.059417, -0.205467, 0.103985, -0.026581, 0.028214, -0.030317, -0.081378, 0.009633, 0.075171, 0.055597, 0.066618, -0.083832, -0.066034, 0.010681, 0.006037, -0.080033, -0.055163, 0.020199, -0.111404, -0.006331, 0.025355, -0.020377, 0.066837, -0.042684, 0.022219, -0.151245, 0.123037, -0.096775, -0.100968, 0.100983, -0.017975, 0.082052, 0.050618, 0.030921, 0.033695, -0.132952, -0.055547, -0.086435, 0.110092, -0.091581, 0.195735, 0.147462, 0.184656, 0.122269, -0.101936, -0.046459, 0.207465, -0.028358, 0.132289, -0.154421, 0.023672, 0.190145, -0.096093, 0.106574, 0.034418, -0.097878], [0.103305, -0.001733, -0.045951, -0.005944, -0.115392, -0.196087, 0.02766, -0.060512, 0.064571, -0.10246, 0.186344, -0.156841, 0.020157, 0.07053, 0.009066, 0.010087, -0.074141, 0.159499, 0.026707, 0.085076, 0.027164, -0.15567, 0.011164, -0.166856, -0.152153, 0.072032, -0.205122, 0.03731, 0.152802, -0.160709, -0.008561, -0.023244, 0.052731, 0.023085, -0.039843, -0.174996, 0.158455, 0.036676, -0.000216, -0.193019, 0.149667, -0.10831, -0.015514, 0.085878, -0.103263, -0.054308, 0.05035, -0.032783, 0.068603, 0.006469, -0.013325, 0.031325, -0.109718, -0.116654, -0.029298, -0.203448, 0.089251, 0.054339, -0.133461, -0.054158, -0.106153, 0.109553, -0.044617, -0.175159], [-0.024089, -0.083472, -0.118297, 0.042164, -0.217375, 0.214602, -0.149942, 0.079635, -0.112588, 0.1228, -0.051458, 0.094145, 0.1152, 0.088281, -0.094146, 0.141789, 0.269386, -0.094553, 0.097616, -0.282791, 0.100702, 0.057259, 0.063197, -0.221961, 0.006805, -0.065526, 0.066423, -0.195573, 0.114403, 0.030001, -0.094547, 0.12033, 0.278083, -0.103825, 0.174865, 0.068134, 0.381897, -0.039944, 0.041042, -0.070719, -0.188973, -0.094817, -0.212767, -0.07857, 0.122431, -0.015165, -0.175274, 0.063205, -0.084879, 0.035338, -0.126719, -0.04994, -0.102821, -0.036968, -0.072063, 0.288727, 0.254419, -0.160422, -0.033686, -0.045718, -0.112277, 0.175338, 0.064514, 0.095926], [0.083339, 0.257073, -0.251668, 0.031709, 0.322508, -0.097349, 0.20672, 0.059992, 0.098429, 0.180367, -0.053243, 0.160526, 0.259393, 0.013779, -0.027139, -0.115968, -0.346414, 0.232262, 0.131427, 0.215279, 0.268095, 0.077555, -0.25124, -0.172142, 0.004151, 0.00079, -0.19573, -0.237903, -0.419644, -0.231263, -0.440462, 0.007473, 0.152732, 0.058974, -0.090331, -0.000126, 0.016703, 0.422448, -0.177125, -0.182754, -0.296009, -0.116467, 0.096264, -0.090941, -0.161335, -0.012294, 0.006255, 0.035446, -0.357559, -0.14021, -0.162782, -0.020494, 0.005021, -0.011069, -0.26137, -0.051654, -0.08639, 0.192483, -0.002135, -0.04951, -0.297026, 0.197273, 0.023565, 0.191067], [-0.133121, -0.000192, 0.037029, 0.016334, 0.015741, -0.138578, -0.121821, 0.101998, 0.085685, -0.118605, 0.084902, -0.171654, -0.084249, -0.240193, 0.027492, -0.065711, 0.049374, 0.089008, 0.030673, 0.106304, 0.191583, -0.056889, -0.012206, -0.033812, 0.006786, -0.04815, -0.07955, -0.073938, -0.024605, 0.101417, 0.007766, -0.044809, 0.128727, -0.076706, -0.062123, -0.008297, 0.054639, 0.117103, 0.15824, -0.109408, 0.129733, -0.003445, 0.030877, -0.111863, 0.0642, -0.139037, -0.081846, 0.100461, 0.10603, 0.082318, 0.058158, 0.194753, 0.054391, 0.149506, 0.130186, -0.073528, 0.002343, 0.055489, -0.023701, 0.106518, -0.119618, 0.018329, 0.053694, -0.004354], [-0.074897, -0.04157, -0.348697, -0.366058, 0.081596, -0.286468, -0.192546, 0.134914, 0.123263, 0.110252, 0.010612, 0.420203, 0.00632, 0.141898, 0.046085, -0.069189, 0.040288, 0.161266, 0.002478, -0.082932, 0.023531, -0.232773, 0.043624, -0.049814, -0.073202, -0.151288, -0.106414, 0.101053, 0.131126, 0.00757, -0.046681, 0.00384, 0.037646, 0.150226, 0.142635, 0.198091, -0.051667, 0.112915, -0.258019, 0.182699, -0.07156, -0.009783, 0.059636, 0.144391, -0.260575, -0.173318, 0.057512, -0.156719, 0.1615, -0.100381, 0.414425, -0.023776, -0.344434, -0.072803, -0.119039, -0.48309, -0.067759, -0.067176, -0.143259, -0.267017, -0.02255, -0.01075, 0.060777, 0.128595], [0.001512, -0.047988, -0.071983, 0.049261, -0.138334, -0.032888, 0.017408, 0.190265, -0.050433, 0.12194, -0.056149, 0.005265, 0.062759, 0.141596, 0.001343, 0.053966, -0.010768, -0.103966, -0.03271, -0.057731, -0.014325, -0.038772, -0.03969, -0.091107, -0.287491, 0.074976, -0.174325, 0.118608, 0.04572, 0.07947, 0.17025, -0.003215, -0.049847, 0.007471, 0.017865, -0.045035, 0.094898, 0.200084, 0.180672, 0.121987, 0.020038, 0.078896, 0.104477, -0.133298, -0.10083, 0.092526, 0.086417, 0.003955, -0.061811, 0.055375, -0.102733, 0.097189, 0.013094, -0.108016, 0.090956, -0.048614, -0.055174, 0.08109, -0.230325, 0.114157, 0.020944, 0.060189, -0.002117, 0.035229], [-0.206407, 0.020364, 0.012264, -0.218943, 0.028379, -0.108329, 0.030696, 0.071645, 0.042469, -0.158039, 0.041886, 0.195318, 0.187676, -0.160284, -0.067884, -0.272431, -0.070267, -0.104348, 0.156674, -0.215796, 0.129204, -0.165531, -0.337833, -0.111734, -0.156645, 0.062551, -0.407915, 0.259491, 0.197261, -0.091617, -0.104755, -0.321491, -0.294443, 0.061401, 0.014506, 0.002824, 0.062086, 0.220026, -0.258713, 0.085952, 0.183835, 0.167345, 0.300319, -0.131474, -0.251213, -0.117788, 0.117899, -0.063856, -0.071469, 0.006887, 0.22364, -0.054878, -0.0823, -0.18416, -0.102497, -0.051266, -0.005435, -0.12568, 0.239087, -0.017249, -0.145531, 0.13032, 0.054824, -0.077735], [-0.390603, -0.203959, 0.147153, 0.02889, 0.022396, 0.206581, 0.143404, -0.111094, -0.151102, 0.124945, -0.113582, 0.120413, 0.011465, 0.020002, 0.142247, -0.134874, 0.101133, -0.297079, -0.039567, 0.032197, -0.004888, -0.18971, -0.183949, -0.04843, -0.263534, -0.159466, -0.029959, -0.022411, -0.254059, 0.214078, 0.117261, 0.080953, -0.073424, 0.194515, -0.021077, -0.385567, -0.076501, 0.184272, 0.164995, 0.147372, -0.08377, 0.011334, 0.059757, 0.022446, 0.215125, -0.144147, 0.029789, 0.178961, -0.02884, -0.072768, 0.037126, 0.335378, -0.078501, -0.021367, -0.030912, -0.285108, -0.137971, 0.084117, -0.059788, -0.158325, 0.096725, 0.046147, 0.007193, -0.022069], [0.294879, 0.148368, -0.086975, -0.159569, 0.118804, -0.116599, 0.150519, 0.31723, -0.172322, -0.050166, -0.275638, -0.065253, -0.225084, 0.022847, 0.051381, -0.195648, 0.166203, 0.025135, -0.018145, 0.058167, -0.083635, 0.024564, -0.010997, 0.085851, -0.119724, 0.082357, 0.107526, 0.037213, 0.137972, 0.245533, 0.003029, 0.223411, -0.023541, -0.256024, 0.067884, -0.034306, -0.146282, -0.105467, 0.124811, -0.069528, 0.02117, 0.100483, -0.194897, 0.183508, 0.15005, 0.241429, -0.032006, -0.164756, -0.07216, 0.042687, 0.109014, 0.072908, 0.020432, 0.050722, 0.227326, 0.005806, -0.00313, -0.146985, -0.196466, 0.135391, 0.036988, 0.13642, 0.058898, -0.12319], [0.147697, 0.021249, -0.148754, 0.236721, -0.059855, -0.025928, -0.300147, -0.17821, 0.206617, -0.293022, 0.225898, 0.185103, -0.083183, -0.442954, 0.090751, -0.333614, -0.091648, -0.142476, -0.028514, -0.225196, 0.276321, 0.021707, 0.013998, 0.152958, 0.410723, 0.091888, -0.153832, -0.121045, -0.0139, 0.014448, -0.313707, -0.087492, -0.099741, -0.088304, 0.089529, -0.263703, 0.097555, 0.017027, -0.06537, 0.32717, 0.304383, -0.085649, 0.145681, -0.245042, -0.116204, -0.066099, 0.22854, -0.207438, 0.19389, -0.207492, -0.075962, -0.312686, -0.300571, -0.187689, 0.091783, -0.172619, 0.091621, 0.216725, -0.256412, -0.334525, -0.284445, -0.204486, 0.084167, 0.109518], [0.060182, 0.102123, 0.060827, -0.125765, -0.063489, 0.036493, 0.013663, 0.021343, -0.117415, -0.02797, 0.070531, 0.64666, 0.104566, -0.032679, -0.07606, 0.04879, 0.103883, 0.173296, -0.076875, -0.156705, 0.010791, 0.036988, 0.136766, -0.110144, 0.282281, 0.191214, -0.109252, 0.168372, -0.018105, 0.062522, 0.03235, 0.049784, -0.013831, -0.080363, -0.209033, -0.068426, 0.068566, -0.049277, 0.148461, -0.131075, -0.040223, -0.002307, 0.038659, 0.001259, 0.018476, 0.223494, 0.059299, 0.336892, -0.310077, -0.182782, -0.069718, 0.00274, 0.052346, -0.040961, 0.036853, 0.145391, -0.164147, 0.117601, 0.147641, 0.221429, -0.059455, -0.055944, 0.018786, -0.060946], [-0.352706, 0.031643, -0.081909, 0.006415, -0.111189, 0.03507, -0.198852, 0.146111, 0.118333, -0.10573, -0.077717, 0.168302, -0.081854, 0.145951, -0.006589, -0.008291, 0.063722, 0.012, -0.016046, -0.018479, -0.133448, -0.019658, 0.254229, 0.154266, 0.036784, 0.07287, -0.135278, -0.127372, -0.173569, -0.24079, -0.051545, 0.205782, -0.199903, 0.314978, -0.129981, -0.019483, -0.191368, -0.205361, -0.03302, -0.157171, 0.159724, 0.047403, 0.23186, 0.345083, -0.269867, 0.018651, 0.032118, 0.337686, 0.245072, -0.105689, -0.247894, -0.052192, -0.113191, 0.015479, 0.115181, -0.018492, -0.070784, 0.026299, 0.265412, 0.201411, 0.082653, -0.139121, -0.115582, -0.065053], [-0.019007, 0.099423, 0.095915, 0.036823, -0.217981, 0.064396, -0.022413, -0.031011, -0.254427, -0.01083, -0.176173, -0.07467, -0.048935, -0.082468, 0.031693, -0.17393, 0.048844, -0.293745, -0.033277, -0.084693, -0.026918, 0.109322, -0.535184, 0.110319, 0.134503, -0.214423, -0.206396, 0.025945, -0.022906, -0.027248, -0.086898, -0.252675, -0.357364, -0.211666, -0.002289, -0.002748, 0.000686, -0.042218, 0.156404, -0.012165, -0.050499, 0.308362, -0.181195, 0.027021, 0.139509, -0.137779, -0.024656, 0.012056, 0.237117, -0.022547, -0.209675, -0.124274, 0.016049, -0.32874, 0.022644, 0.139028, 0.073318, -0.013152, 0.082088, -0.118925, 0.084119, 0.066831, 0.058057, -0.135574], [0.020529, 0.010655, -0.044804, -0.256409, 0.149936, -0.346821, 0.486005, 0.018989, 0.015261, 0.029072, -0.140201, -0.133044, 0.215608, 0.170753, -0.151349, 0.099951, -0.088804, 0.070252, 0.291065, -0.302799, -0.001585, -0.098275, -0.413966, -8e-06, 0.00348, -0.053141, -0.071729, -0.046454, -0.058877, 0.021766, -0.332668, -0.212039, -0.326133, 0.04548, 0.21128, 0.040145, -0.231315, -0.008431, 0.224016, -0.028043, -0.017217, 0.033176, 0.096643, -0.054343, -0.294217, -0.303049, -0.07168, -0.074641, -0.187438, -0.171085, -0.171297, 0.224138, 0.038581, 0.235558, 0.061095, 0.081281, -0.031757, -0.072706, 0.420479, -0.200074, -0.011499, 0.252491, 0.129855, 0.354764], [-0.127539, -0.082868, -0.026265, -0.381722, 0.018012, 0.138294, 0.316183, -0.085865, -0.01526, 0.210727, -0.156088, -0.131937, 0.290064, -0.003928, 0.005352, 0.059137, -0.139628, -0.116641, 0.162819, 0.278812, 0.070947, 0.048868, 0.219221, 0.314891, -0.323243, 0.40233, -0.083477, 0.195998, 0.246134, -0.044919, 0.161817, -0.0767, 0.12896, -0.193336, -0.291351, -0.069919, -0.347415, -0.220583, -0.203323, -0.200656, 0.047312, -0.294944, 0.091588, -0.139937, -0.014117, -0.269012, -0.053551, 0.262626, -0.393173, -0.175129, -0.031549, -0.122631, -0.183214, 0.365675, 0.191989, -0.120909, 0.056698, -0.033334, -0.25535, 0.142085, -0.055191, 0.320631, 0.165359, 0.119148], [0.017931, -0.081803, -0.051804, -0.174239, -0.124471, 0.262622, -0.037265, 0.008466, 0.209047, -0.13329, -0.222518, 0.08977, -0.19444, 0.23466, 0.137547, -0.103408, 0.192109, 0.160302, -0.064458, -0.290495, -0.147535, -0.249357, -0.369808, -0.026924, -0.177694, 0.051445, 0.029391, -0.213345, 0.157054, -0.018077, 0.116574, 0.181547, 0.292871, -0.032892, 0.382364, 0.027814, 0.178158, -0.208347, 0.079093, -0.23762, -0.179534, -0.170217, -0.133408, -0.119581, 0.081076, -0.195338, -0.177015, 0.13661, 0.040081, -0.276519, 0.206444, -0.121722, 0.051037, -0.183554, -0.104194, -0.216676, -0.005072, 0.129962, 0.124908, -0.267241, -0.126762, 0.31076, 0.052631, 0.094238], [-0.004784, 0.062804, -0.086608, -0.012035, -0.060903, -0.059208, 0.0916, -0.074191, -0.159227, 0.163932, 0.020838, 0.088289, -0.025681, 0.02516, 0.041663, 0.005297, -0.006092, -0.014026, -0.065895, 0.114887, 0.075914, -0.118088, -0.054914, 0.055259, -0.127793, -0.066204, -0.027925, -0.043479, 0.030168, -0.114175, 0.165646, -0.171759, -0.083603, 0.019503, 0.007727, -0.050739, -0.013832, 0.078368, -0.050003, 0.071081, 0.172104, -0.120504, 0.060193, -0.09429, 0.20764, 0.120611, 0.060296, 0.127847, 0.020105, -0.085739, 0.027848, 0.039883, 0.073925, 0.079835, -0.038858, 0.084537, -0.005063, 0.040833, -0.114785, 0.056307, 0.043512, 0.045345, -0.282082, 0.182122], [0.059991, 0.242514, -0.076994, -0.253838, 0.204263, -0.04971, 0.039506, 0.234695, 0.138318, -0.090937, -0.125397, 0.365755, 0.135364, -0.039423, -0.087075, -0.256544, -0.205324, -0.00982, 0.071012, -0.058253, 0.16597, -0.118402, 0.072927, 0.214262, 0.035058, 0.119451, 0.075582, 0.064136, 0.213919, -0.12948, -0.012894, 0.03073, -0.007203, -0.249523, -0.040017, 0.308951, -0.412123, -0.085867, -0.043886, 0.197872, -0.208683, -0.051911, 0.322465, 0.341424, -0.152107, -0.025472, 0.209777, 0.090031, 0.003328, 0.003397, 0.127102, 0.056965, -0.230236, 0.420271, -0.027138, -0.05135, -0.050638, 0.080171, 0.31, -0.164768, -0.12634, -0.16528, 0.050351, 0.081722], [0.043029, 0.299633, 0.08228, -0.113691, -0.089219, -0.03686, 0.099044, -0.048258, 0.003945, 0.080126, -0.280858, -0.187091, -0.029175, 0.143715, 0.025117, -0.014738, 0.123838, 0.081368, 0.09337, 0.248862, -0.084404, 0.268898, 0.245755, -0.162087, 0.286228, 0.256833, 0.318829, 0.463751, 0.091596, -0.175553, 0.038274, -0.094913, 0.167787, 0.163969, -0.318892, -0.194677, 0.145886, -0.108924, 0.328243, 0.430781, 0.222086, 0.219916, 0.254287, 0.009005, 0.178846, 0.008365, -0.058063, 0.099857, 0.04494, -0.092611, -0.128509, -0.175449, 0.070738, -0.054155, -0.124184, -0.187867, -0.000712, -0.239535, -0.146055, 0.046863, -0.007957, -0.488701, 0.264163, 0.120353], [-0.04719, -0.028357, 0.04925, 0.10842, -0.129946, 0.01684, 0.321228, -0.146866, 0.061037, -0.217636, 0.045609, 0.205138, -0.259384, 0.025178, 0.396452, -0.128286, -0.207808, -0.051697, 0.116909, -0.305105, -0.097482, 0.08441, -0.052709, 0.012764, -0.095687, -0.220391, 0.117999, -0.071193, 0.01417, -0.052955, -0.046431, -0.12579, 0.025514, -0.17326, -0.323182, -0.048126, 0.01986, 0.003739, 0.045581, 0.011903, -0.037358, -0.09856, -0.116317, 0.050354, 0.144848, 0.052394, -0.171529, 0.134781, -0.144588, -0.088263, 0.060193, 0.109076, -0.068418, -0.138906, -0.040609, -0.161185, -0.153137, -0.062423, -0.03811, 0.370846, -0.002699, 0.395978, 0.229113, 0.296808], [-0.193596, 0.073755, -0.099451, 0.115153, 0.03188, -0.017724, 0.111728, 0.036101, -0.2759, 0.266136, -0.138756, 0.114279, -0.436744, 0.178983, 0.085906, 0.203212, 0.200149, 0.408648, -0.07501, -0.149595, 0.108529, 0.254451, 0.345495, -0.051574, 0.018031, -0.237671, -0.262192, -0.212491, 0.077778, -0.039141, -0.013026, 0.014465, 0.003883, 0.118927, -0.032802, 0.009447, 0.154112, -0.081056, 0.250676, -0.287614, -0.010962, 0.206867, -0.204237, -0.304624, -0.072263, -0.016854, -0.191579, -0.025519, -0.037978, -0.42072, 0.10682, -0.382225, -0.207813, -0.128604, -0.149005, -0.21848, 0.134975, -0.106912, -0.535506, 0.221158, -0.003138, -0.02984, 0.030716, -0.053208], [-0.157751, 0.021762, -0.043547, -0.183344, 0.211001, -0.27138, -0.06285, -0.17554, 0.123608, -0.156008, -0.026087, 0.099584, 0.007747, -0.072215, 0.062639, -0.254343, -0.13654, -0.137425, 0.001597, -0.334469, 0.055187, -0.014696, -0.206216, 0.072188, -0.001595, -0.019852, 0.191771, 0.095155, 0.219903, 0.157986, -0.106926, -0.097556, -0.27011, -0.387146, -0.018086, -0.449321, -0.290685, 0.051315, 0.250191, -0.111865, -0.03506, -0.016078, 0.054296, 0.14169, 0.194408, 0.21003, -0.164364, -0.031296, -0.244748, -0.310017, 0.051577, -0.19936, -0.182663, 0.296128, 0.055902, 0.235041, -0.149289, 0.022507, -0.114978, -0.224806, -0.241364, 0.068602, 0.001008, -0.070165], [-0.159312, -0.038132, 0.397691, 0.343517, -0.121487, 0.183837, -0.1789, -0.191895, 0.108244, 0.045138, -0.140584, 0.229998, -0.258113, -0.278245, 0.224865, -0.175934, -0.024257, -0.15289, 0.222317, -0.341143, -0.06986, 0.552927, 0.111888, -0.05335, 0.043652, 0.321143, 0.106722, 0.321103, -0.062706, -0.137534, 0.371884, 0.403323, -0.48216, 0.38293, -0.036505, -0.102544, 0.251221, -0.318942, 0.240183, -0.035195, 0.018295, 0.084568, 0.008342, -0.142501, 0.325062, 0.391624, 0.090604, 0.090632, 0.527721, -0.059831, -0.075866, -0.073222, 0.177934, -0.164727, 0.192259, 0.25875, 0.054293, 0.158548, 0.025105, -0.243215, 0.421182, -0.033602, 0.051268, 0.312769], [0.003055, 0.149054, 0.055457, 0.031456, -0.050494, -0.052052, -0.064905, 0.111429, 0.085304, 0.227032, -0.06597, -0.305244, -0.18828, -0.14065, -0.095243, 0.182725, 0.040825, 0.141122, -0.396331, -0.243861, 0.054195, 0.23952, -0.018028, -0.175541, 0.047081, -0.149787, 0.055845, -0.210608, 0.150212, -0.017316, -0.204893, 0.077116, -0.10269, 0.054416, -0.063048, 0.015078, -0.046896, -0.19144, 0.003408, -0.252436, -0.381931, 0.220185, 0.070576, -0.018857, 0.13218, 0.150271, -0.21668, 0.096279, -0.415687, 0.083421, -0.298492, -0.52491, 0.034925, -0.138188, 0.169997, 0.07705, 0.16655, -0.151112, 0.066927, -0.004935, 0.018971, 0.027851, -0.010731, -0.257676], [0.05628, 0.403914, 0.08505, -0.05437, -0.017619, -0.077362, 0.073761, 0.183383, -0.110609, 0.318685, -0.368731, -0.007507, -0.033915, -0.09931, 0.002563, 0.01669, 0.011108, 0.044616, -0.111758, 0.216648, 0.018563, -0.226867, 0.097972, 0.198033, -0.131089, 0.090378, 0.075702, -0.204524, 0.081039, -0.031389, 0.130511, 0.014071, -0.101752, -0.319666, -0.038432, -0.020332, -0.145075, -0.009174, -0.060758, 0.076809, -0.059317, 0.11634, -0.049121, -0.073454, -0.091683, -0.112708, -0.072732, -0.157226, -0.32726, 0.020406, 0.059315, -0.190425, -0.049004, -0.043494, 0.090894, 0.182539, -0.064689, 0.106519, -0.32187, -0.201751, -0.223362, -0.084015, 0.184921, -0.104656], [-0.027826, 0.13091, 0.117371, 0.039052, -0.007711, 0.079675, -0.160182, -0.193122, -0.099603, 0.198442, -0.112974, -0.014744, -0.238901, -0.108888, -0.090577, 0.131734, 0.061062, 0.20556, -0.165157, -0.126383, 0.061942, 0.077751, 0.262548, -0.06633, -0.003931, 0.193496, 0.096962, -0.556464, -0.169544, -0.028729, 0.076184, 0.035101, -0.365198, 0.116102, -0.081071, -0.127971, -0.04413, 0.034774, -0.224734, -0.150733, -0.078691, 0.264588, -0.015587, -0.479901, 0.121997, 0.038714, -0.119909, -0.273201, 0.093586, 0.156559, 0.081424, -0.028045, 0.043703, -0.115467, -0.071078, 0.02948, 0.255766, -0.30756, -0.008883, 0.204861, 0.023837, -0.03294, 0.157156, -0.002621], [-0.236292, 0.295226, 0.211195, 0.352533, 0.425431, 0.069006, 0.237455, 0.061437, -0.353903, 0.061447, -0.138239, 0.471477, 0.043254, -0.015995, -0.06027, -0.037181, -0.130621, -0.195898, 0.162191, 0.158045, -0.069676, 0.974285, 0.731305, 0.004026, 0.051521, -0.193835, 0.066639, -0.157288, -0.20858, 0.343024, 0.177568, 0.192984, 0.252126, 0.308969, 0.406033, -0.376465, 0.574531, 0.09296, 0.367234, -0.003768, -0.062785, -0.091722, -0.072649, -0.224373, -0.132474, 0.320434, 0.171536, 0.034712, 0.053611, -0.082351, 0.147853, 0.128068, -0.050692, -0.145332, 0.076601, 0.182509, 0.642386, -0.219767, 0.103991, 0.000398, 0.118447, 0.00377, 0.086534, 0.154696], [0.243752, -0.058375, -0.187566, -0.081138, 0.025446, -0.218803, -0.030277, -0.561955, 0.394138, 0.220752, 0.068309, -0.180184, 0.030317, -0.091979, -0.161018, -0.014902, 0.059839, 0.049276, 0.14006, 0.121865, -0.030227, -0.212777, -0.228718, 0.01229, 0.093482, 0.317796, -0.128738, -0.29079, 0.070036, 0.023469, 0.113383, -0.002263, -0.603902, 0.052871, -0.264235, 0.104902, 0.129058, -0.049868, -0.316558, -0.259778, -0.158753, -0.161918, 0.220034, -0.118578, 0.055206, 0.01143, 0.100451, -0.343419, -0.426366, 0.147268, 0.088515, 0.044654, -0.216699, 0.032049, -0.108247, 0.14857, -0.265808, -0.167383, 0.164149, 0.039903, 0.011465, -0.380697, 0.316331, 0.122981], [-0.128713, 0.176699, 0.169017, 0.195925, 0.234172, 0.090642, -0.030091, 0.117827, -0.024637, 0.014985, 0.123562, 0.381169, -0.094483, -0.130504, 0.039736, 0.001135, 0.035906, 0.118487, 0.016642, 0.21747, 0.148197, 0.499105, 0.194682, -0.114116, 0.534679, -0.107822, -0.017221, -0.237081, -0.152198, 0.083675, 0.068893, 0.179034, -0.173253, 0.159231, -0.259992, -0.03794, 0.237964, 0.142125, 0.184722, -0.332567, -0.046928, 0.128774, 0.306246, -0.404374, -0.079969, 0.154798, -0.118997, 0.29282, 0.098576, 0.300628, -0.151604, 0.01532, -0.015146, 0.013316, 0.150388, -0.178912, 0.229006, 0.14906, 0.146736, 0.222048, -0.114095, 0.148118, -0.234431, 0.056051], [0.129861, -0.198835, -0.151513, -0.005966, -0.071555, 0.088347, -0.038047, -0.346829, 0.107589, 0.175248, 0.373198, -0.043735, -0.139251, 0.031062, 0.071938, 0.028042, 0.176098, -0.112938, 0.511359, -0.039906, 0.17322, -0.042859, 0.376912, 0.252919, -0.016828, 0.60985, 0.04461, -0.266418, -0.248231, -0.180313, 0.150481, -0.148946, -0.287346, 0.067353, -0.014792, -0.087796, 0.41414, 0.022338, -0.248879, -0.373706, 0.430409, -0.057946, 0.08432, -0.25189, -0.230723, 0.089728, 0.108626, -0.153886, 0.019764, 0.301545, 0.206268, 0.222864, -0.219022, 0.314896, 0.019403, -0.211532, -0.225394, -0.026336, 0.004521, 0.331458, 0.183963, -0.353078, 0.368291, 0.170504], [0.097556, -0.175837, -0.328126, -0.584331, -0.134157, -0.020993, -0.363444, -0.427172, -0.144828, -0.01571, 0.096302, 0.147743, 0.024076, 0.033129, -0.145244, -0.204319, -0.075116, -0.140473, -0.258568, -0.252659, 0.092446, -0.374099, 0.320911, 0.056048, -0.166074, 0.184903, -0.105173, -0.133125, -0.29608, -0.052735, 0.079647, -0.254002, -0.527444, -0.197332, -0.506076, -0.322891, 0.070902, -0.06507, 0.106738, -0.25682, -0.070568, -0.008763, 0.111351, -0.032887, -0.064948, -0.022851, -0.155119, -0.136843, -0.206701, 0.054189, 0.101623, -0.13948, -0.476915, -0.164649, -0.149797, 0.02152, -0.127309, 0.361637, -0.160836, 0.050403, 0.05955, -0.619827, 0.14788, -0.088128], [-0.259388, -0.285187, -0.100947, -0.307658, -0.101859, -0.094402, -0.305205, -0.24332, -0.004663, 0.380895, 0.115285, 0.181911, 0.086643, 0.151449, -0.045195, -0.090521, 0.160707, 0.146336, 0.254415, 0.124218, -0.020161, -0.345566, -0.001586, -0.366413, -0.01901, -0.557542, -0.013515, 0.285475, 0.133037, 0.07551, -0.238633, 0.037341, -0.13083, 0.114811, -0.126835, 0.136321, -0.229123, 0.440116, 0.113356, 0.428876, -0.277397, -0.050728, -0.050501, 0.18516, 0.033178, -0.231535, -0.158085, 0.169026, 0.07928, -0.165037, -0.625127, -0.249275, -0.017691, -0.114578, 0.342763, -0.347363, -0.164815, 0.182486, -0.441026, 0.264863, -0.145623, -0.16077, -0.20128, 0.275374], [-0.094862, -0.223797, 0.051377, 0.028294, 0.129212, 0.057456, -0.118542, -0.14019, 0.358212, 0.093825, -0.082467, -0.087863, -0.005682, -0.039313, -0.078907, -0.152346, 0.205668, 0.127834, 0.004179, -0.199931, -0.153044, -0.386974, -0.181103, -0.196743, -0.208284, -0.0136, -0.204089, -0.17383, 0.229582, 0.113653, 0.233508, 0.318912, -0.131391, 0.152135, -0.14117, -0.022065, -0.416604, -0.062117, 0.140885, -0.237835, -0.212428, 0.060046, 0.02524, 0.097875, 0.257184, 0.154871, -0.276037, 0.285215, 0.374202, -0.054121, 0.306001, -0.179167, -0.20491, -0.152097, -0.510172, -0.010017, -0.157668, -0.359542, 0.185861, 0.354659, -0.16025, 0.001619, 0.063369, 0.152684], [0.100107, 0.028219, -0.086865, 0.109409, -0.068545, -0.068943, 0.112193, 0.261404, 0.092214, -0.381483, 0.028428, 0.003894, -0.049994, 0.068865, 0.054656, -0.357342, -0.22302, -0.118933, 0.072503, 0.249783, -0.18266, -0.067017, -0.268731, 0.128856, 0.220284, -0.088479, -0.040671, 0.14172, -0.017423, -0.28557, 0.038394, -0.114804, 0.202904, 0.026725, -0.102168, -0.02184, -0.19013, -0.103842, -0.141632, 0.21983, -0.064823, 0.008557, -0.201968, 0.161011, 0.07041, -0.117819, 0.002808, 0.036805, -0.040403, -0.381734, -0.119771, 0.039268, 0.088451, -0.131711, -0.087908, -0.06881, -0.292049, 0.299118, -0.005181, -0.236557, 0.022217, -0.115244, -0.167244, -0.138911], [-0.143404, 0.268263, -0.148811, 0.187512, 0.455588, 0.050018, 0.282505, 0.171129, -0.512532, -0.055458, 0.037219, 0.155388, -0.095294, -0.016077, 0.21896, 0.192356, -0.551772, -0.243254, -0.237105, 0.191384, 0.131339, 0.446835, 0.278293, 0.492039, 0.429325, -0.121976, 0.228526, -0.269613, -0.177424, 0.308741, 0.012755, -0.035002, -0.344188, 0.007804, -0.196132, -0.428955, -0.065532, 0.196179, 0.252288, 0.08843, 0.043804, 0.116781, 0.270683, -0.46188, -0.305111, 0.068228, 0.014575, 0.022196, -0.363015, 0.074691, 0.541624, 0.061508, 0.027093, 0.030213, 0.565884, 0.201174, 0.483397, 0.207467, 0.383038, 0.015396, -0.116672, -0.007947, -0.138617, -0.215927], [0.109584, -0.018323, 0.173863, 0.282457, 0.080382, 0.130863, 0.143549, 0.423316, 0.0529, 0.185297, -0.033479, -0.044143, -0.023709, -0.002445, 0.248345, -0.140834, 0.15098, -0.133039, 0.061149, 0.273042, -0.092081, 0.085881, -0.143851, -0.204822, 0.093073, -0.088593, 0.005998, 0.057951, 0.224158, -0.203054, 0.305491, -0.059829, 0.078509, 0.254119, 0.093798, 0.197274, 0.148123, -0.10686, 0.021942, 0.07983, 0.003943, 0.06122, -0.256889, -0.108325, 0.031565, -0.088431, -0.068082, 0.336795, 0.145186, 0.028332, 0.078904, 0.019509, 0.101554, 0.093991, -0.182155, -0.136105, -0.037439, -0.201765, 0.075512, 0.144808, 0.192211, 0.292043, 0.085585, 0.339618]], [-0.256584, -0.224205, -0.048525, -0.373917, 0.046846, -0.251827, -0.295671, -0.25233, -0.029227, 0.040196, -0.017192, -0.203606, -0.072071, 0.022908, -0.171927, -0.109118, -0.079935, 0.24595, -0.43956, -0.408869, -0.263576, -0.158374, 0.089709, -0.138874, 0.087536, -0.039915, -0.194816, -0.06171, -0.175844, -0.001879, -0.123021, -0.054829, -0.312346, 0.046863, -0.055234, -0.225473, -0.171648, -0.062106, -0.15676, -0.159657, -0.306119, 0.20265, 0.203955, 0.106929, -0.078121, 0.056033, -0.022445, -0.308138, -0.311416, -0.019179, -0.100376, -0.291963, -0.330219, -0.448367, 0.231622, 0.095362, 0.151945, -0.118597, -0.114453, -0.136793, -0.278368, 0.067021, -0.293577, -0.193003]),
    ([[-0.101106, -0.029604, -0.19857, -0.056259, -0.090703, 0.081925, 0.001969, 0.006993, -0.943141, -0.030808, -0.167824, -0.167994, 0.139664, -0.544933, -0.054384, 0.092792, 0.073199, 0.086202, 0.159774, -0.000509, -0.048782, -0.082605, 0.008205, 0.050211, 0.083271, 0.173611, -0.2091, 0.056144, -0.00373, 0.040307, -0.064809, 0.08491], [-0.08248, -0.143062, -0.001266, 0.066509, 0.006403, 0.017969, -0.068849, -0.13915, -0.232631, 0.215407, 0.243564, -0.34069, 0.086606, -0.108337, -0.101167, 0.105071, 0.002092, 0.228297, 0.149546, 0.169159, -0.062666, -0.069236, 0.096418, -0.182683, 0.182299, 0.032813, -0.16443, 0.083136, -0.085879, 0.092945, 0.10024, 0.196954], [-0.160099, 0.065512, 0.060466, 0.047959, 0.04355, -0.048482, -0.187248, 0.120059, 0.168404, 0.149264, 0.03438, 0.129667, 0.050135, 0.276318, -0.042172, 0.09009, 0.005143, -0.203126, 0.220236, 0.1933, 0.005596, -0.072132, 0.045855, -0.031378, -0.057932, 0.103573, 0.15249, -0.017617, 0.033628, -0.140162, -0.065437, 0.175924], [-0.162176, 0.04146, 0.16895, 0.086466, 0.276221, -0.212299, 0.230917, -0.120615, -0.492658, 0.103817, 0.111521, 0.235375, 0.086354, -0.467305, -0.091173, -0.225269, 0.074714, -0.269342, -0.309856, 0.138137, 0.066257, 0.134324, 0.022813, 0.198159, -0.34219, -0.138788, 0.063573, -0.150916, 0.048733, 0.203502, 0.084688, -0.027132], [0.07726, 0.087759, -0.008067, -0.094245, 0.056976, 0.15319, 0.219825, 0.027427, -0.080779, -0.154808, -0.084897, -0.06924, -0.007967, -0.047396, 0.063175, -0.024102, 0.107028, 0.109786, 0.057117, 0.168113, -0.09304, -0.040834, 0.12162, -0.014861, 0.127373, -0.03008, 0.042837, -0.041312, 0.050146, 0.301711, 0.111156, 0.120659], [0.041865, 0.037329, 0.12462, 0.06137, 0.092547, -0.19493, -0.134753, 0.028065, 0.077051, 0.043083, 0.205095, 0.130558, 0.132748, 0.145952, 0.091173, 0.024007, 0.008407, -0.190794, 0.153516, 0.135125, 0.119517, -0.038009, -0.022763, -0.12618, 0.061071, -0.034716, 0.17071, 0.178245, -0.019029, -0.076083, -0.020545, 0.159175], [-0.039912, 0.146714, -0.129101, -0.026599, -0.104133, 0.106655, 0.066929, 0.114314, -0.139925, 0.178104, 0.303232, -0.027704, 0.134166, -0.264497, -0.081285, 0.023148, 0.16622, 0.213829, 0.114469, 0.231598, 0.070679, 0.094961, 0.087631, -0.254639, 0.015388, 0.085471, 0.037075, -0.0503, -0.032491, 0.248801, 0.017632, 0.221458], [-0.277011, 0.07248, -0.014149, 0.136247, -0.28464, -0.214289, -0.47646, -0.020988, -0.018483, -0.425388, 0.24325, -0.094069, -0.099484, 0.148101, -0.022618, -0.214875, -0.37574, -0.246195, -0.344613, 0.133202, -0.100191, -0.177353, -0.159626, 0.171651, -0.451919, -0.381947, 0.015653, -0.167568, 0.246641, -0.439799, 0.03172, 0.307777], [0.000629, -0.223381, 0.180804, -0.164645, -0.161521, 0.033053, -0.062587, 0.004352, -0.073223, -0.331761, -0.122386, 0.05697, -0.198039, -0.111077, -0.067658, 0.059102, -0.078923, 0.140988, 0.160547, 0.026646, -0.101051, 0.106553, 0.015627, 0.274082, 0.031309, 0.048336, 0.048971, 0.220393, -0.197702, 0.078972, -0.101918, -0.267957], [0.142902, 0.01037, 0.145485, -0.051861, 0.199611, 0.01924, 0.17589, 0.117216, -0.099508, -0.151994, 0.073183, -0.14689, 0.126884, 0.032218, 0.075635, 0.117031, 0.016078, 0.013447, -0.233854, 0.020535, 0.131619, 0.185665, -0.107973, -0.080042, 0.067746, 0.108293, 0.205618, -0.147445, 0.067396, 0.158134, -0.07512, 0.007922], [0.030339, -0.128266, -0.049765, 0.012405, 0.202309, 0.131396, 0.134532, 0.02106, 0.039826, 0.121817, 0.072459, -0.222264, 0.226326, 0.101497, -0.162389, -0.078275, 0.147115, 0.021661, 0.158607, 0.07732, 0.031414, -0.058223, -0.070552, -0.015942, -0.175447, 0.115222, 0.000523, -0.017222, 0.030732, 0.108707, -0.027325, 0.031066], [0.329111, 0.065026, 0.265623, -0.09912, -0.029375, 0.193475, -0.048026, -0.01428, -0.020168, -0.255252, -0.437185, 0.020041, -0.291545, -0.213201, -0.108923, -0.024368, 0.042811, -0.061883, 0.125228, -0.359382, 0.091409, -0.07787, 0.005829, -0.175338, 0.132369, 0.083481, -0.093301, -0.081209, -0.374028, 0.323849, -0.098231, -0.445749], [0.004558, -0.051353, 0.028649, 0.065831, -0.320056, 0.07973, 0.123789, 0.036942, 0.158721, -0.192924, -0.186582, -0.250001, -0.139299, -0.195907, -0.161521, -0.098784, 0.051477, -0.120474, 0.269291, -0.059965, 2.7e-05, -0.178648, -0.156696, -0.157778, -0.118226, 0.065536, -0.124941, -0.117835, -0.021468, -0.055359, 0.018743, -0.098825], [-0.168534, -0.029961, 0.069209, -0.118121, -0.106142, 0.0453, 0.076288, 0.026644, 0.080039, -0.162035, 0.030179, -0.64799, -0.138778, -0.243841, 0.002391, -0.027708, -0.040146, -0.07484, -0.046245, 0.094419, 0.021458, -0.001392, 0.054081, 0.111702, -0.006225, 0.02007, 0.149059, 0.061005, -0.041798, 0.0511, -0.010204, 0.053958], [0.071634, 0.082294, -0.132063, 0.100894, -0.049842, -0.216614, 0.102258, -0.154338, 0.198206, 0.101861, 0.068965, 0.33372, 0.087026, -0.06436, -0.072768, 0.197203, 0.002469, -0.144395, 0.120303, 0.053667, 0.036905, 0.045803, -0.07588, 0.099898, 0.08134, -0.155555, 0.021612, 0.138494, 0.079299, 0.006319, 0.116053, 0.068606], [0.039822, 0.164538, 0.071784, -0.027886, 0.144038, 0.05678, -0.099975, -0.036343, 0.15099, -0.02595, -0.073755, 0.088165, 0.155652, -0.334319, 0.018104, -0.04591, -0.033156, -0.251354, -0.321445, 0.019563, 0.114525, -0.008719, -0.260217, 0.030831, -0.02187, 0.101919, 0.177416, -0.027747, 0.05656, 0.104434, 0.176993, -0.056773], [-0.064714, 0.011112, 0.082767, -0.091657, 0.261612, 0.023736, 0.083931, -0.090363, -0.249103, 0.06306, 0.223131, 0.132168, 0.084266, -0.106022, 0.07328, -0.113115, -0.03985, -0.080418, -0.059489, -0.021855, 0.089976, 0.161594, 0.026377, -0.192064, -0.194333, -0.018759, 0.02769, 0.008451, -0.09333, -0.202162, 0.2121, 0.175587], [-0.02912, 0.077813, 0.017505, 0.014704, 0.040382, 0.062629, -0.098763, -0.193505, -0.387176, -0.405858, -0.082591, 0.032827, -0.074547, -0.431431, -0.101769, 0.075931, -0.136243, 0.186413, 0.204405, 0.053104, 0.121686, 0.046534, 0.027822, -0.01912, 0.11977, -0.065792, -0.042991, 0.043147, 0.14543, 0.048021, 0.103406, -0.027192], [0.25365, -0.064757, 0.076828, -0.105011, -0.120373, 0.115384, -0.05434, 0.098639, 0.183846, 0.025831, 0.165653, -0.115931, -0.120438, -0.314524, -0.045559, -0.023705, 0.16426, 0.076213, 0.209137, -0.099671, -0.251466, 0.044618, 0.147855, -0.028408, 0.180158, -0.007264, 0.023655, 0.108669, -0.027678, 0.1434, 0.079378, -0.172145], [0.009806, -0.214432, -0.171939, -0.122507, -0.067663, 0.010707, -0.157727, -0.057412, 0.470138, 0.027718, -0.182396, 0.00762, -0.237789, 0.270795, -0.056724, 0.005076, -0.151197, 0.107743, 0.165647, -0.065842, -0.242228, -0.0748, -0.199332, 0.072649, 0.357011, -0.105763, -0.158067, -0.010937, -0.178357, 0.118246, -0.042973, -0.228512], [0.129071, 0.056799, -0.123982, -0.133714, -0.461729, 0.090632, 0.037492, 0.003982, 0.017294, -0.086096, 0.077018, -0.244343, 0.004647, 0.111548, 0.005583, 0.100611, 0.108347, 0.094464, 0.138829, -0.052993, -0.007933, -0.22117, 0.060764, -0.003178, 0.019757, -0.036459, -0.139229, -0.063149, 0.081739, 0.076266, -0.018206, 0.153196], [-0.079388, -0.035908, -0.01478, 0.07382, 0.017751, 0.020426, -0.264808, -0.241932, 0.070193, 0.214939, 0.100439, -0.304065, 0.149951, -0.200624, -0.025524, -0.093891, -0.11856, 0.181469, -0.57092, 0.180092, -0.034473, 0.015083, -0.017514, 0.299247, 0.097078, -0.053243, 0.166779, -0.188344, 0.028309, -0.75872, 0.111174, 0.121995], [-0.48635, -0.041037, 0.076411, -0.067743, 0.027999, 0.167992, -0.199556, -0.183335, 0.125995, -0.101271, 0.088928, -0.874344, -0.087239, -0.125594, -0.039068, -0.079259, -0.443592, 0.149491, -0.538784, 0.134808, 0.04231, -0.017789, 0.085608, 0.234484, 0.148903, 0.03021, 0.083326, -0.045381, 0.224574, -0.230582, 0.11512, -0.21886], [-0.345206, 0.125227, 0.077274, -0.053976, 0.194457, -0.208792, 0.089812, 0.234145, 0.142258, 0.074703, -0.098639, 0.143649, 0.143384, 0.132092, -0.015715, 0.055525, -0.023501, 0.131244, 0.168127, -0.032743, -0.021175, 0.255702, 0.067086, -0.147661, 0.011727, -0.006923, 0.157866, 0.192201, 0.084058, 0.281934, 0.098295, -0.077929], [-0.196904, 0.114678, -0.099092, 0.099717, -0.176474, -0.085475, -0.106738, 0.011015, 0.220664, -0.066007, -0.100821, 0.300345, 0.011058, 0.025832, -0.091957, 0.102572, -0.16285, 0.109238, 0.222287, -0.132878, 0.110817, -0.094058, -0.102674, -0.058962, 0.232428, 0.198434, 0.193206, 0.09134, -0.0146, 0.329238, 0.098797, 0.00586], [0.159624, -0.062608, 0.036579, -0.040788, -0.233657, -0.189388, 0.044229, 0.163226, -0.012937, -0.38346, -0.244019, -0.153791, -0.208236, -0.205634, -0.127897, 0.166009, 0.208849, 0.019659, 0.020565, -0.111984, -0.196561, -0.124719, 0.109232, -0.051058, -0.252933, -0.017238, 0.15665, 0.232708, -0.008202, 0.107889, -0.163645, -0.344112], [0.099308, 0.102937, 0.120328, -0.11135, -0.04955, 0.023629, -0.154995, -0.205938, 0.298839, 0.07564, 0.136729, 0.139063, 0.13655, -0.075187, -0.191369, 0.038778, -0.335579, -0.028953, 0.149401, 0.10532, 0.141805, 0.052335, -0.215265, 0.018529, 0.159426, -0.271295, -0.062089, 0.029684, 0.033977, -0.44383, 0.181777, 0.100699], [-0.19825, -0.112645, 0.101557, 0.00399, -0.106808, -0.016851, -0.260844, 0.111869, -0.022795, 0.120462, -0.038084, -0.353424, -0.042434, 0.149982, 0.005422, -0.281483, -0.230427, 0.12921, -0.089751, -0.148076, 0.004363, -0.037277, 0.044169, -0.019975, -0.20181, -0.309732, -0.238459, -0.2191, -0.199002, -0.233023, 0.199612, -0.036787], [-0.179606, -0.047628, -0.005507, -0.027074, 0.010398, -0.046395, -0.084897, 0.109767, -0.055692, 0.141598, -0.047441, -0.301444, 0.277726, 0.031809, -0.06333, -0.001698, -0.020263, -0.158393, -0.455963, 0.122873, -0.044037, 0.012662, 0.013579, -0.231034, -0.260776, -0.235961, -0.113786, -0.094714, 0.032807, -0.367475, 0.019966, -0.061644], [0.293975, 0.09446, 0.041863, -0.029123, 0.065436, 0.068542, -0.030893, 0.033209, 0.142454, 0.094473, -0.058938, 0.294332, -0.007735, -0.062232, -0.015647, -0.077244, -0.164324, -0.147497, -0.449138, 0.020667, 0.105769, 0.154519, -0.081085, -0.037969, 0.079311, -0.232024, -0.076987, 0.056047, 0.097633, 0.13068, 0.148495, -0.047743], [-0.242021, 0.041242, 0.10952, 0.038745, 0.033081, 0.205963, 0.127206, 0.181191, 0.025188, 0.277258, 0.07347, -0.349698, 0.140794, 0.214077, -0.105838, 0.173915, -0.070337, -0.079846, -0.059823, 0.061374, 0.13669, 0.053772, -0.069808, 0.040227, 0.221731, 0.219329, 0.006816, 0.035753, -0.120269, 0.04885, 0.109394, 0.068037], [0.060391, 0.074561, -0.052611, -0.043358, 0.050307, 0.096222, -0.038207, -0.131777, -0.151683, 0.003091, -0.066376, -0.072976, -0.043711, -0.200833, -0.128482, -0.181341, 0.066489, -0.18713, -0.030541, -0.048315, 0.137377, 0.137513, 0.216656, 0.079765, -0.206787, -0.045054, 0.137019, -0.027605, -0.000334, -0.372378, 0.044665, 0.066745], [-0.207051, -0.081051, -0.047185, 0.03685, -0.070901, -0.441295, -0.348027, -0.196789, 0.190632, -0.357434, -0.093184, -0.101321, -0.147531, 0.167088, -0.114449, -0.043245, -0.255158, -0.163379, -0.110496, 0.074916, -0.182272, -0.0233, -0.404604, -0.014493, -0.140618, -0.251648, 0.003312, -0.15585, 0.005859, -0.137107, 0.027806, 0.132034], [0.224594, -0.05226, 0.083937, 0.086753, 0.150572, 0.079151, -0.057, 0.151615, -0.05633, -0.037133, -0.202264, 0.093194, 0.065462, 0.021297, -0.117551, 0.181623, -0.060527, 0.205829, -0.269735, -0.124575, -0.094738, -0.061267, 0.184297, 0.104328, 0.044621, 0.237239, 0.093951, 0.078043, -0.002686, -0.171659, 0.092708, -0.087471], [-0.302307, -0.030632, 0.022252, -0.098881, -0.111829, -0.06675, -0.016404, 0.012664, -0.435106, -0.114591, 0.069981, -0.163985, -0.107762, -0.027792, -0.169049, -0.232434, 0.08701, -0.069537, -0.361893, 0.020459, -0.018175, 0.187526, 0.095116, -0.098493, -0.205937, -0.171059, -0.003763, 0.04513, -0.021908, -0.243061, 0.033983, -0.011578], [-0.370896, 0.248104, 0.128517, -0.048231, 0.049976, -0.142147, 0.026772, -0.080007, -0.201819, 0.306437, 0.004536, 0.132363, 0.092649, 0.171261, -0.119197, -0.050768, 0.12742, -0.251107, -0.113985, 0.166117, -0.016101, 0.007013, -0.194612, -0.3469, 0.03217, -0.255553, 0.142837, -0.113229, 0.055803, 0.271301, 0.083493, 0.189207], [0.053148, 0.24753, 0.343784, 0.03596, 0.157238, -0.09149, 0.111684, 0.111095, -0.343109, 0.257486, -0.119856, 0.277289, -0.025453, -0.092736, -0.084343, 0.186856, 0.192065, 0.001034, 0.086244, 0.042954, 0.16329, 0.18701, 0.001435, -0.630588, 0.027832, -0.162805, 0.278411, -0.015288, 0.136626, 0.168252, 0.234092, -0.100876], [0.136081, 0.206396, -0.16932, -0.09287, -0.226654, 0.148258, 0.15881, 0.13814, 0.117339, -0.176278, 0.266726, -0.032719, -0.125021, 0.346945, -0.161123, 0.12797, -0.068366, 0.068954, 0.202624, 0.165599, 0.060379, -0.019949, -3.6e-05, -0.347671, 0.013981, 0.122343, -0.005904, -0.047422, 0.102629, -0.023838, 0.092657, 0.04473], [0.219169, 0.278885, -0.010473, 0.08228, 0.191031, 0.12779, -0.02064, 0.041049, -0.450609, 0.258118, 0.020296, -0.003039, 0.202924, -0.13067, 0.053698, -0.173095, 0.035252, 0.089992, 0.01393, 0.06788, 0.076852, 0.026972, 0.136394, -0.109382, 0.017054, 0.046667, 0.174878, -0.220845, 0.04189, -0.000199, 0.251301, 0.191629], [0.020274, 0.126677, 0.036095, -0.000511, -0.241209, -0.42958, -0.259726, -0.141312, 0.247014, -0.303201, 0.162961, -0.116015, -0.011711, 0.098294, -0.005612, 0.029009, -0.201896, -0.110702, -0.096457, -0.062489, -0.000553, -0.196422, -0.210447, 0.255089, 0.015368, -0.219193, -0.285468, 0.089434, -0.202734, -0.079455, 0.018791, 0.111618], [0.111629, -0.219091, -0.097674, -0.066779, -0.450352, -0.157192, 0.04821, 0.171206, -0.421302, -0.286925, -0.160774, 0.096278, -0.287829, 0.224174, 0.037548, 0.23373, 0.017734, 0.15773, -0.100699, -0.111106, -0.141947, -0.207426, 0.17898, 0.199797, -0.07842, 0.097423, -0.153939, 0.071292, -0.202958, -4.1e-05, -0.207211, -0.511175], [0.007766, -0.079371, 0.115475, 0.042962, 0.09244, -0.149029, -0.115282, 0.019092, -0.474332, -0.228724, -0.005891, 0.028382, -0.0339, 0.240288, -0.138262, -0.184746, -0.032471, 0.176326, -0.191122, -0.07062, 0.084104, 0.108431, -0.009727, -0.126555, -0.155753, 0.004739, 0.047532, 0.029504, 0.077897, -0.085545, 0.070464, -0.043764], [-0.171143, 0.048953, -0.013504, 0.039673, -0.167661, 0.022609, 0.052301, 0.079589, -0.393206, -0.352151, -0.12779, -0.535033, -0.15539, -0.063991, 0.000943, 0.055844, -0.09927, 0.096563, 0.004171, 0.030252, -0.010925, -0.170961, 0.157463, 0.272612, -0.090568, 0.130049, 0.078216, 0.10344, 0.047402, -0.140577, -0.029117, -0.09404], [-0.01421, -0.151289, 0.074568, 0.059463, -0.045113, -0.202544, -0.199513, -0.169808, -0.035417, 0.072869, -0.076438, -0.164433, -0.078295, -0.277359, -0.069886, -0.113053, -0.024363, -0.306764, -0.208231, -0.010762, -0.020096, 0.078235, 0.044445, -0.262442, -0.355884, 0.013132, -0.108873, -0.094043, -0.024839, -0.39657, -0.04191, 0.124777], [0.003146, 0.000911, -0.051255, -0.072008, 0.057793, -0.327536, -0.162217, -0.071785, 0.08058, 0.189783, 0.003542, 0.2582, -0.042294, -0.301585, 0.050697, 0.044467, -0.103739, -0.10086, -0.274731, 0.0251, -0.093973, -0.088364, -0.192135, -0.282354, 0.026929, -0.407373, -0.04446, -0.02225, -0.059338, -0.44842, 0.096517, -0.161254], [-0.427269, 0.063589, 0.010679, 0.072804, -0.088513, 0.000912, -0.160274, -0.026824, -0.134405, 0.112985, 0.006869, -0.10552, -0.027266, -0.016475, -0.039835, -0.126569, -0.185904, 0.049858, 0.210839, -0.136814, 0.022306, -0.066691, 0.025041, 0.260465, 0.008195, -0.009101, 0.116593, 0.043685, 0.06001, -0.404267, -0.064719, 0.082428], [-0.074724, 0.07667, -0.042406, -0.109713, -0.169918, 0.01576, -0.083045, 0.033583, -0.040134, -0.408833, 0.027661, -0.200022, -0.038608, 0.125267, 0.023573, 0.066865, 0.043242, 0.074708, 0.04809, -0.029437, -0.08953, 0.023298, -0.022863, 0.260979, -0.077936, 0.097088, -0.293002, 0.078579, 0.056287, -0.060646, 0.087147, -0.052046], [-0.061276, -0.057771, -0.013614, -0.032707, 0.137862, 0.080886, 0.187906, 0.155024, -0.311246, 0.176146, -0.090656, 0.012592, 0.144749, -0.125253, -0.210342, 0.074197, 0.153093, -0.046604, -0.046308, -0.067999, -0.072396, 0.018617, 0.213131, -0.400956, -0.117952, 0.203971, 0.192737, 0.084122, -0.011048, -0.024927, -0.068012, -0.118791], [-0.436014, 0.024899, 0.015233, 0.018308, -0.023516, -0.149075, -0.185154, -0.388077, -0.519011, 0.433877, 0.117693, -0.011788, 0.200294, -0.018923, 0.046054, -0.027757, -0.097586, -0.098213, -0.160452, -0.108823, -0.056741, 0.155027, 0.158144, -0.36672, -0.030379, 0.100118, 0.156161, -0.047765, -0.220919, -0.416406, 0.059817, 0.11307], [0.038865, 0.052863, 0.045154, 0.023692, 0.149428, 0.083939, 0.120485, -0.14999, 0.001948, 0.071322, -0.23869, -0.109883, 0.036177, -0.251709, 0.057016, 0.118622, 0.228213, -0.063272, 0.139638, 0.065446, 0.014442, 0.071634, -0.006539, 0.183147, 0.243059, 0.124419, 0.109653, 0.118267, 0.101198, -0.112363, -0.067406, -0.128029], [0.179897, -0.094102, 0.075478, 0.10328, 0.169901, 0.172783, 0.157521, 0.02247, 0.004181, 0.090907, -0.229607, -0.306237, 0.019655, -0.317413, 0.030482, 0.08499, 0.210372, 0.179458, 0.066901, -0.176941, -0.084503, -0.018674, 0.177599, 0.085054, 0.210398, 0.145986, -0.020748, 0.031649, 0.017297, 0.189796, -0.082855, -0.282659], [-0.063669, 0.102934, -0.20394, 0.111321, -0.313953, 0.10682, -0.055413, -0.095046, -0.009511, -0.038563, 0.395357, -0.237156, -0.129479, 0.043025, -0.151188, -0.036052, 0.143325, 0.159033, 0.026205, 0.017599, -0.025007, -0.002384, -0.077147, 0.142085, 0.095506, 0.078565, -0.036032, 0.02749, -0.153748, 0.003011, 0.047548, 0.335707], [0.108803, 0.065778, 0.098208, -0.048251, 0.073945, -0.192903, -0.166184, 0.035557, 0.114531, 0.050488, -0.009168, 0.320894, -0.058853, 0.199516, -0.032064, -0.091038, 0.110824, -0.107277, 0.234951, 0.011256, 0.038903, -0.016453, -0.065377, -0.07714, -0.074026, -0.180255, 0.043253, 0.090218, -0.027131, -0.187163, 0.12422, 0.044225], [0.041359, -0.337476, -0.070033, -0.003676, -0.093513, 0.050721, -0.051456, 0.087625, 0.215486, -0.111532, -0.163632, 0.01026, -0.055053, -0.144413, -0.04457, 0.056547, 0.097512, 0.112441, 0.030264, -0.258121, 0.000886, 0.012524, -0.060907, 0.044808, 0.173175, -0.029585, -0.254592, 0.095944, -0.189703, 0.220036, -0.188119, -0.350582], [-0.002968, 0.124865, -0.035215, -0.088055, 0.152032, -0.19949, -0.004523, 0.054928, 0.07457, -0.198114, -0.068043, 0.106333, -0.032057, 0.168863, 0.024543, -0.005966, -0.008793, 0.097617, -0.36399, 0.047906, 0.159638, 0.03454, 0.060067, 0.093597, -0.156748, 0.004563, -0.09341, 0.027557, 0.004413, -0.166013, 0.098408, -0.042027], [-0.408778, -0.008428, -0.115176, -0.038605, 0.008597, -0.186547, -0.247378, -0.239569, -0.120982, 0.002301, 0.018978, 0.046971, -0.028398, -0.084726, -0.098801, -0.18235, -0.042783, -0.140629, -0.014856, -0.031548, 0.105122, 0.047537, 0.015638, 0.06333, -0.034047, -0.134257, 0.015138, -0.024396, 0.139406, -0.035334, -0.011271, -0.063686], [-0.148644, 0.212415, -0.040727, 0.115564, -0.083974, -0.189468, 0.062738, 0.005183, -0.562276, 0.148904, -0.050512, -0.059167, 0.203639, 0.161958, -0.016317, 0.050459, -0.0723, -0.112399, -0.121684, 0.10462, -0.045473, 0.027193, -0.073788, -0.0252, -0.142599, -0.055464, -0.045787, -0.170406, 0.080236, 0.105223, 0.122423, 0.21938], [0.043976, 0.060991, 0.003953, -0.00037, 0.153008, 0.055412, 0.058213, 0.04807, -0.018483, -0.181294, -0.291143, -0.002964, -0.41689, 0.225112, -0.097411, 0.093154, -0.184239, 0.027526, -0.012821, -0.309087, 0.064472, 0.035196, -0.052439, -0.219314, 0.0076, 0.088128, -0.280155, 0.075547, -0.213442, -0.138915, 0.004973, -0.251574], [0.238214, -0.009497, 0.024882, -0.134244, 0.10257, 0.131622, 0.009777, 0.204266, -0.23362, 0.158032, -0.101985, 0.422852, 0.082392, 0.052596, -0.071726, 0.099355, 0.21932, 0.074774, 0.312624, -0.059307, 0.050134, 0.194324, 0.017852, -0.485744, 0.225869, 0.044498, 0.079041, 0.081636, -0.150128, 0.233596, -0.124061, 0.041896], [0.05274, 0.209359, -0.066277, -0.08252, -0.041409, -0.059392, -0.002456, -0.079437, -0.266498, 0.134848, -0.163688, -0.038331, -0.059974, -0.106595, -0.024835, -0.164582, -0.055423, -0.182875, -0.257968, 0.013215, 0.010656, -0.017421, -0.254263, -0.639752, 0.015859, -0.138668, -0.004595, -0.001588, 0.039831, 0.058246, 0.08501, 0.140264], [-0.103849, -0.094433, 0.087203, -0.066858, 0.021884, 0.064331, -0.13628, 0.198393, -0.060342, 0.206356, -0.093456, 0.180281, 0.113522, 0.274257, -0.091404, 0.001435, 0.052921, 0.079599, 0.069319, 0.089004, -0.036511, 0.020599, -0.07034, 0.158228, 0.18815, 0.045586, 0.050413, -0.016552, -0.0146, -0.088629, -0.031419, 0.134928], [-0.109881, -0.10809, -0.062412, 0.127772, -0.312971, 0.042833, 0.142485, 0.044323, -0.127927, -0.410159, 0.343047, 0.113915, -0.29926, -0.29119, -0.184652, 0.147403, -0.052242, -0.314622, -0.069902, -0.005142, 0.1139, 0.133318, -0.099448, 0.153235, 0.150969, 0.07718, -0.120078, 0.080915, -0.02584, 0.010023, 0.029497, 0.311478], [0.170313, -0.097967, 0.022345, 0.003175, -0.072581, -0.050312, 0.141026, 0.041121, -0.280991, -0.065879, -0.155198, -0.203045, -0.004679, -0.387535, 0.077787, -0.020763, 0.064003, 0.082714, -0.109684, 0.104753, -0.155112, -0.18626, 0.120932, -0.344185, 0.097046, 0.110108, -0.081621, -0.047794, 0.043936, 0.142126, -0.046934, -0.055474], [0.118399, -0.159697, 0.132746, -0.051525, -0.448967, 0.113164, 0.123859, 0.081133, -0.26496, -0.034351, 0.07291, 0.243957, -0.022601, -0.292679, 0.081489, -0.060492, 0.141397, 0.030104, 0.114648, 0.168996, -0.101636, -0.026696, 0.02281, -0.043343, 0.132331, 0.071468, 0.125307, -0.065998, 0.14132, -0.092425, 0.051949, -0.12135]], [-0.244145, 0.093396, 0.105233, 0.114766, 0.088177, -0.042141, 0.096329, 0.047244, -0.031752, -0.310099, -0.083182, -0.176677, -0.0148, 0.038309, -0.109532, 0.156127, 0.05513, 0.018392, -0.322484, 0.022789, 0.113585, 0.044851, 0.03078, -0.071561, -0.139474, 0.093726, 0.031252, 0.046168, -0.069932, -0.279453, 0.121217, -0.152231]),
    ([[0.393673], [-0.160599], [-0.079556], [0.02469], [-0.184254], [0.264898], [0.209074], [0.169656], [0.373848], [-0.447444], [-0.300417], [0.517695], [-0.139147], [0.424508], [0.006301], [0.155089], [0.196889], [0.23751], [0.343084], [-0.133087], [-0.048463], [-0.107087], [0.076503], [0.257021], [0.241093], [0.117841], [-0.12196], [0.059083], [-0.067342], [0.312343], [-0.108669], [-0.143775]], [0.095367]),
]

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
        'hy_ms': 50.0, 'hy_k': 0.2, 'hy_kq': 0.0, 'hp_ms': 300.0, 'ad_k': 0.0, 'hy_only': 1,
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
        # exp_r3_qgru: 1 = pred_raw is the 500 ms head on the taker GRU state (SEQQ_*) instead of MLPQ[500]; 0 = identical to mm_r3_gru.
        # qg_gate 1 = only when the ridge passes mlp_gate (else the ridge, as before), 0 = every event.
        'qg_on': 1,
        'qg_gate': 1,
        # blend weight of the head vs MLPQ[500] in pred_raw (1 = head only)
        'qg_mix': 0.5,
        # 0 = head on the 500 ms target (SEQQ_*), 1 = head on the 300..900 ms mean target (SEQL_*)
        'qg_set': 0,
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
        'veto_on': 0,
        'veto_thr': 0.5,
        'slw_thr': 2.0,
        'slw_h': 5,
        'slw_every_us': 20000,
        # exp_r3_qcap: keep a touch quote with < qc_ahead market lots ahead when only the rules in qc_rules
        # (bits: 1 rt, 2 slw, 4 gate, 8 / 16 quote signal behind the touch at spread 1 / >= 2) want it pulled; 0 = off (identical).
        'qc_ahead': 0,
        'qc_rules': 8,
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
        self.veto_on = int(params['veto_on'])
        self.veto_thr = float(params['veto_thr'])
        self.veto_w = np.array(VETO_W)
        self._vsnap = None
        self._vslow = 0.0
        self._vlast = -10**12
        self._vnow = 0
        self.n_veto = 0
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
        self.qg_on = int(params['qg_on']); self.qg_gate = int(params['qg_gate']); self.qg_mix = float(params['qg_mix'])
        if self.qg_on:
            if int(params['qg_set']):
                self.qg_xm = np.array(SEQL_XM); self.qg_xs = np.array(SEQL_XS); self.qg_ysc = SEQL_YSC
                self.qg_head = [(np.array(a), np.array(b)) for a, b in SEQL_HEAD]
            else:
                self.qg_xm = np.array(SEQQ_XM); self.qg_xs = np.array(SEQQ_XS); self.qg_ysc = SEQQ_YSC
                self.qg_head = [(np.array(a), np.array(b)) for a, b in SEQQ_HEAD]
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
        self.qc_ahead = int(params['qc_ahead'])
        self.slw_sp_last = 0.0
        self.gate_dr_last = 0.0
        self.qc_rules = int(params['qc_rules'])
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
        self._vsnap = snapshot
        self._vnow = snapshot.timestamp_us
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
            self._vslow = sp
            self.slw_sp_last = sp
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
            self.gate_dr_last = drift
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
        qc_keep = (False, False)
        if self.qc_ahead > 0:
            qc_pre = (pull_bid, pull_ask)
            qc_bits = [0, 0]
            for k, sg in ((0, 1), (1, -1)):
                if not qc_pre[k]:
                    continue
                b = 0
                if self.rt_us > 0 and now - self.rt_t[k] < self.rt_us:
                    b |= 1
                if self.slw_on and self.slw_sp_last * sg <= -self.slw_thr:
                    b |= 2
                if self.gate_thr > 0.0 and self.slw_on != 2 and self.gate_dr_last * sg <= -self.gate_thr:
                    b |= 4
                qc_bits[k] = b if b else 64
        qc_nt = self.n_takes
        if self.fq_on:
            bid, ask, pull_bid, pull_ask = self._fq(now, bb, ba, fair_shift, pull_bid, pull_ask)
        if self.qc_ahead > 0 and self.n_takes == qc_nt:
            kp = [False, False]
            for k, pulled, touch, behind in ((0, pull_bid, bb, bb - bid), (1, pull_ask, ba, ask - ba)):
                if not pulled:
                    continue
                b = qc_bits[k]
                if not qc_pre[k]:
                    b = (8 if ba - bb < 2 else 16) if behind > self.fq_max_behind else 64  # 64: a take needs the room, never kept
                if b & ~self.qc_rules:
                    continue
                act = self.active[k]
                if not act or self.quote_px[k] != touch:
                    continue
                if all(self._ahead(self.orders[o]) < self.qc_ahead for o in act):
                    kp[k] = True
            qc_keep = (kp[0], kp[1])
            if kp[0]:
                pull_bid = False
            if kp[1]:
                pull_ask = False
        lay = self.layer_size
        off = self.layer_offset
        ladder = self.ladder_size > 0
        if ladder and not self.ladder_reserve:
            self._ladder(0, 1, _BUY, bb, ask, pull_bid)
            self._ladder(1, -1, _SELL, ba, bid, pull_ask)
        if pull_bid:
            self._pull(0)
            self._pull(2)
        elif qc_keep[0]:
            pass
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
        elif qc_keep[1]:
            pass
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
                if self.qg_on:
                    pq = self.qg_mix * self._qg(h0) + (1.0 - self.qg_mix) * pq
            self.pred_raw = pq if self.mq_h else p
        elif self.qg_on and not self.qg_gate:
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
            self.pred_raw = self._qg(h0)
        else:
            self.pred_raw = p
        if self.pred_blend != 1.0:
            p = self.pred_blend * p + (1.0 - self.pred_blend) * self.fair_sig
        self.pred_sig = p

    def _qg(self, h0) -> float:
        """500 ms quote prediction: head on the 22 quote features and the taker GRU state (ticks)."""
        h = np.clip((h0 - self.qg_xm) / self.qg_xs, -8.0, 8.0)
        ls = self.qg_head
        h = np.maximum(np.concatenate((h, self.sq_h)) @ ls[0][0] + ls[0][1], 0.0)
        h = np.maximum(h @ ls[1][0] + ls[1][1], 0.0)
        return float((h @ ls[2][0] + ls[2][1])[0]) * self.qg_ysc

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
        if self.veto_on and self._vsnap is not None:
            s = self._vsnap
            touch = px - sign * self.fq_slip
            fq = 0
            for v in (s.asks if sign > 0 else s.bids):
                if v.owner == 'market' and v.price == touch:
                    fq += v.quantity
            bb = ba = -1
            for v in s.bids:
                if v.owner == 'market':
                    bb = v.price
                    break
            for v in s.asks:
                if v.owner == 'market':
                    ba = v.price
                    break
            pts = self.gate_pts
            dr = (self.gate_mid2 - pts[0][1]) / 2.0 if pts else 0.0
            now = self._vnow
            x = np.concatenate((np.array((sign * self.pred_sig, sign * self.pred_raw, sign * self.fair_sig, ba - bb, log1p(fq),
                                          self.hole[k], log1p((now - self._vlast) / 1000.0), sign * self._vslow, sign * dr,
                                          sign * self.inv, q, sign * (self.fair_prev_mid2 / 2.0 - touch))), self.sq_h))
            z = float(x @ self.veto_w) + VETO_B
            if 1.0 / (1.0 + exp(-z)) > self.veto_thr:
                self.n_veto += 1
                return None
            self._vlast = now
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
