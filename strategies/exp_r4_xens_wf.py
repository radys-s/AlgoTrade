"""exp_r4_xens_wf: mm_r3_gru3 plus a second, R4-trained table set (from exp_r4_wf_s2.py); taker/quote/slow predictions blended by xw / xwq / xws (0 = gru3)."""
"""mm_r3_gru3 (home, R3 upload candidate): mm_r3_gru2 (gate_thr 14) + rt_ms 150, fq_k_sp2 1.5 and a 50 ms hysteresis after passive flips (hy_only 1, hy_ms 50, hy_k 0.2; exp_r3_hyst): round3 653,327 / mirror 639,714 / round2 447,310; holdout 157,925 / 158,799 / 112,106 (gru2 638,024 / 625,673 / 447,035; holdout 157,152 / 158,439 / 112,310).

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
# ---- exp_r4_xens: the second (R4-trained) table set, copied verbatim from exp_r4_wf_s2.py
R4_PRED = (-0.0, (0.50233, -1.9759984, 4.7772194, -2.6840441, 1.6987565, -0.4531857, 0.007839, 0.0315912, 0.0003458, -0.0, -0.1546411, 0.1785022, 0.0448765, -0.0448765, -0.0454307, -0.0054872, 0.1748529, -0.3495935, 0.0004214, 0.0003719, -4.6e-05, -0.0))
R4_MLP_MEAN = [-0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.235948, 0.0, 0.0, 5.908366, 5.908316, 0.0, 7e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009057]
R4_MLP_SCALE = [1.133423, 0.999352, 0.952236, 0.8477, 0.755997, 2.569228, 20.906504, 0.789135, 18.420658, 0.707402, 0.649149, 0.542064, 1.664617, 1.664603, 0.999991, 4.403231, 1.165521, 0.303697, 133.782867, 335.202209, 1064.944946, 1.340472]
R4_MLP_LAYERS = [
    ([[0.038815, 0.082213, 0.002476, -0.051215, -0.033662, 0.172645, 0.224291, 0.042557, 0.052052, -0.115846, 0.142528, -0.030408, 0.051506, -0.000848, 0.218164, -0.147607, -0.089976, -0.43767, -0.041171, -0.020761, -0.094035, -0.428793, 0.048806, 0.085113, -0.116123, -0.324268, 0.080224, 0.203035, -0.232865, 0.331888, 0.161819, 0.053209], [-0.12702, 0.266464, 0.001945, -0.117987, -0.181996, -0.075443, -0.119959, -0.22503, 0.037096, 0.178363, -0.246008, -0.179414, 0.143705, -0.089633, 0.329226, -0.041092, -0.047844, -0.094047, 0.116404, -0.182224, -0.05625, -0.178459, 0.238823, -0.063594, -0.272023, -0.024293, -0.15012, 0.08048, -0.277131, 0.015943, 0.071035, 0.054355], [-0.047379, 0.108723, -0.008029, 0.055618, -0.103323, -0.041723, 0.021227, -0.179573, 0.116155, -0.088687, 0.042598, 0.163924, -0.167263, 0.112263, 0.206929, 0.015178, 0.099921, -0.188582, 0.066428, 0.072339, 0.071888, 0.008285, 0.280366, -0.000915, -0.002284, 0.177135, -0.015911, 0.212548, -0.109189, -0.209607, -0.28653, 0.13849], [0.0091, 0.040882, -0.068534, 0.204628, -0.146295, -0.046119, -0.162132, 0.050768, 0.084269, -0.000806, 0.134811, 0.093368, -0.342394, -0.200093, -0.372762, 0.03513, 0.100357, 0.076607, 0.162938, -0.100714, 0.131835, -0.067721, 0.118689, 0.132527, 0.144089, 0.288212, 0.03786, 0.127275, 0.046638, -0.036398, -0.09354, -0.018096], [0.323072, -0.097922, 0.161048, -0.328138, 0.176664, -0.576757, -0.226953, 0.384696, -0.424386, -0.196861, -0.169762, -0.248319, -0.129862, -0.221372, -0.373001, -0.379122, -0.02091, 0.150623, 0.185473, 0.370534, -0.376137, 0.114682, 0.063876, 0.149429, 0.185774, -0.185723, 0.225333, 0.226667, -0.165236, -0.390522, -0.272878, 0.157028], [0.015103, -0.046386, 0.018604, 0.244219, -0.075184, -0.076081, -0.271469, 0.038235, -0.082025, -0.004972, -0.189476, 0.116789, 0.129716, -0.086453, 0.015738, -0.045727, -0.254627, 0.282413, -0.097656, -0.309781, 0.126794, -0.144935, 0.308188, -0.064915, -0.123273, -0.03205, -0.064387, 0.33284, -0.045275, -0.219385, -0.275614, -0.147371], [-0.088414, -0.093804, 0.220465, -0.032377, 0.010531, 0.05224, -0.026328, -0.038937, -0.43366, -0.376473, 0.029597, -0.378849, 0.108328, -0.267795, -0.327883, 0.207994, 0.312401, 0.363079, -0.130542, 0.220013, -0.438669, 0.035295, -0.026974, -0.308877, 0.473222, 0.246745, 0.434716, -0.07075, -0.010169, 0.135286, -0.154034, 0.186524], [-0.462159, 0.03359, 0.081198, 0.322306, 0.155765, 0.383131, 0.021095, 0.067416, -0.073096, 0.222666, 0.074551, -0.028481, 0.203971, -0.116091, 0.009442, -0.079302, 0.004335, -0.102361, 0.309159, -0.216072, -0.050892, 0.052873, 0.01149, 0.018191, 0.375668, 0.044541, -0.041683, -0.010596, 0.197582, -0.027913, 0.078293, -0.227768], [-0.185389, -0.299097, 0.146024, 0.111316, -0.291825, -0.243276, -0.195734, 0.593329, -0.139688, -0.214168, 0.10227, -0.834435, 0.190856, 0.218455, -0.279067, -0.064759, -0.184551, -0.144001, -0.177387, -0.09486, -0.051017, -0.306248, 0.343564, 0.178651, 0.23701, -0.126262, 0.481296, -0.005616, -0.294868, 0.092487, 0.149165, 0.604205], [-0.192219, -0.631789, 0.280533, -0.090798, -0.114125, 0.256285, -0.128802, -0.161407, 0.03811, -0.020263, -0.261758, -0.165236, -0.052521, -0.288464, -0.090732, 0.02618, -0.286652, -0.06974, -0.088296, 0.023597, 0.02281, 0.06749, -0.20918, -0.19211, -0.015812, 0.213181, 0.015933, 0.311395, -0.159771, 0.299894, -0.505524, -0.149357], [0.053065, -0.051498, -0.179431, -0.053231, -0.162536, 0.066364, 0.262942, 0.013343, 0.022161, -0.209452, 0.02805, 0.044909, -0.089422, 0.097788, 0.006678, 0.332208, -0.087869, 0.191738, 0.081422, 0.176859, -0.096943, 0.02664, 0.081754, -0.089915, -0.02556, 0.026751, -0.103046, 0.127239, -0.114688, 0.144356, -0.036748, 0.01042], [0.071889, 0.035562, 0.317751, -0.260362, -0.224528, -0.297939, 0.542474, 0.117213, 0.040254, 0.141074, 0.009474, -0.432771, -0.064732, 0.149092, 0.361487, -0.352873, -0.296804, -0.184435, -0.375911, -0.175671, 0.021188, -0.285508, -0.268999, 0.452193, -0.133883, -0.21886, 0.174042, 0.126347, 0.047213, -0.001196, -0.131428, 0.366771], [0.037182, -0.016725, -0.165839, 0.049191, -0.220551, -0.155955, -0.129706, -0.00614, 0.279477, -0.323512, 0.031065, -0.085931, 0.08693, 0.378484, -0.154903, 0.049756, 0.088821, 0.19164, 0.258493, -0.064197, 0.022128, -0.031085, 0.161186, 0.344225, -0.108947, 0.062756, 0.050294, -0.228743, 0.074548, -0.010199, 0.115623, -0.286719], [0.311207, -0.052439, 0.114838, -0.078135, 0.016869, -0.025663, -0.125742, 0.092837, -0.008405, -0.187213, -0.08703, 0.005052, 0.13685, 0.233203, 0.116225, -0.097266, 0.073485, -0.187725, 0.25764, -0.016205, 0.000667, -0.135465, -0.206951, 0.208859, -0.006477, -0.196073, -0.239802, -0.112244, 0.149751, 0.051622, 0.005937, -0.124729], [0.24321, 0.100568, -0.144268, 0.249385, -0.300842, 0.183571, 0.078455, -0.204709, 0.413279, -0.141435, -0.293907, -0.104747, -0.254341, -0.032754, 0.038235, 0.262127, 0.127878, 0.136956, -0.228613, 0.201637, -0.181441, -0.265268, -0.261918, -0.302164, -0.460914, -0.049111, 0.138742, 0.075222, 0.218856, 0.208641, 0.051571, -0.095056], [-0.07568, -0.089033, 0.160882, 0.253109, 0.110341, -0.339738, 0.30382, 0.026454, -0.005432, -0.153367, -0.252035, 0.051888, 0.297128, -0.134303, 0.052793, 0.038459, -0.062948, 0.100346, -0.260612, -0.158043, 0.039712, 0.071429, -0.188986, -0.026365, 0.002194, -0.167233, -0.071086, 0.071352, 0.116423, -0.045927, 0.178513, 0.158149], [-0.274849, -0.255329, 0.572293, -0.325371, 0.497683, -0.069532, 0.126294, 0.274418, 0.004612, 0.116217, 0.130238, 0.550673, 0.301354, -0.15723, -0.178927, -0.19436, -0.448617, -0.302135, 0.722526, 0.043169, -0.166746, 0.058201, 0.343769, 0.47947, -0.238349, -0.01748, -0.263104, -0.113465, -0.23078, -0.44442, -0.024699, 0.107243], [-0.163171, -0.205734, -0.319245, 0.036385, 0.038278, 0.184019, -0.143965, 0.45026, 0.006756, 0.179324, -0.158754, 0.053235, 0.032803, -0.299603, -0.471258, -0.259901, -0.073647, 0.44531, 0.023661, 0.166153, 0.020436, -0.000769, 0.305317, 0.062531, -0.065378, 0.102669, -0.073252, -0.247423, -0.170361, 0.187508, 0.051744, -0.099436], [0.068508, 0.000153, 0.003137, 0.238933, 0.154383, -0.066562, -0.02437, 0.064238, -0.054636, 0.065456, -0.133383, 0.009374, 0.010442, -0.039445, 0.01794, -0.040101, -0.035245, -0.045607, -0.14503, 0.024574, -0.130649, 0.163701, 0.052982, -0.00372, 0.139239, -0.02508, -0.01892, -0.083849, 0.134069, -0.037628, -0.010736, 0.117334], [0.255813, 0.133699, 0.219203, -0.172441, 0.225368, -0.042345, -0.050527, 0.302908, -0.268245, -0.046398, -0.259593, -0.144423, -0.221698, 0.09831, -0.015908, 0.144433, -0.230192, -0.113057, -0.17458, -0.047985, 0.480202, 0.018973, 0.156803, 0.415672, 0.634968, 0.020347, -0.041985, 0.07833, -0.174757, 0.076297, -0.166942, 0.067395], [-0.435844, 0.2434, 0.271643, 0.182931, -0.234359, 0.084778, 0.071235, -0.356933, 0.128393, 0.182128, 0.000398, 0.205785, 0.348466, 0.510552, -0.048978, -0.027198, 0.224708, 0.474226, -0.035852, -0.088464, 0.188074, 0.133347, -0.261699, -0.320837, -0.130366, -0.139843, -0.178094, 0.070276, 0.136807, 0.070218, 0.179941, 0.01649], [0.013724, 0.010975, 0.049618, 0.043793, 0.048169, -0.057061, 0.084716, 0.007217, -0.008319, 0.521521, 0.10758, 0.434718, -0.218815, -0.126056, -0.013166, -0.06741, -0.36642, -0.004359, 0.061363, -0.203742, 0.235402, 0.139064, -0.025839, -0.226232, -0.037556, -0.025594, 0.670481, 0.003133, -0.051812, -0.001479, -0.066593, -0.0106]], [-0.182695, 0.035304, -0.235249, 0.023398, 0.035018, -0.14191, -0.225015, 0.056366, 0.314307, -0.12775, 0.394256, 0.180903, -0.361756, 0.274522, -0.101282, -0.116085, 0.045192, -0.006512, 0.149349, -0.108784, -0.066308, 0.095253, -0.206116, -0.218492, 0.059648, -0.252508, -0.123427, -0.206501, -0.034878, 0.046563, -0.204632, 0.220581]),
    ([[-0.096198, -0.042049, -0.269048, -0.685741, -0.096326, 0.098447, 0.120643, -0.401895, -0.403851, -0.420759, -0.203985, 0.156098, -0.097318, -0.13395, -0.145091, -0.702788], [-0.210883, 0.160916, 0.034258, 0.196168, -0.092905, -0.150291, 0.086462, 0.39829, -0.056743, 0.155911, 0.238015, -0.102974, 0.020424, -0.575222, 0.118216, -0.338491], [-0.425994, 0.04527, -0.153845, 0.210414, -0.018498, -0.028586, 0.036945, -0.086234, 0.236304, 0.079387, 0.082938, 0.031681, -0.074538, -0.695311, 0.220047, 0.303491], [-0.230922, -0.049177, -0.250288, -0.169596, -0.718167, 0.15886, 0.11306, -0.232124, -0.29375, -0.049439, 0.031612, -0.255652, -0.441102, -0.203452, -0.278561, -0.718327], [-0.129473, 0.131323, -0.060386, -0.05975, 0.108635, 0.02767, 0.13576, 0.196748, 0.325414, 0.198359, 0.144009, 0.166895, 0.174224, 0.116528, 0.365754, -0.21794], [0.116326, -0.020001, -0.164886, -0.17636, -0.146477, -0.187813, -0.138205, 0.1057, -0.387485, -0.037984, 0.047805, 0.212333, 0.236917, 0.105649, -0.417024, -0.180106], [-0.385836, -0.093058, 0.078885, -0.447825, -0.117714, -0.146801, 0.050119, 0.126155, -0.125398, -0.233745, 0.06175, -0.162422, -0.24679, -0.075391, -0.400427, -0.089052], [-0.241745, 0.093228, 0.472175, 0.372155, 0.411612, -0.01334, -0.151447, -0.089177, -0.032639, -0.017653, 0.029541, -0.272358, 0.047327, 0.402251, 0.240101, -0.329609], [0.089238, -0.161304, 0.185933, -0.558748, -0.113487, 0.100941, -0.059441, -0.000861, -0.386189, -0.166682, -0.243089, -0.082553, 0.013869, -0.06168, -0.366755, 0.329176], [0.095864, 0.253143, 0.169796, 0.104439, 0.068939, -0.357188, -0.002538, -0.125099, -0.324184, 0.286219, -0.027536, 0.103407, 0.079504, 0.063249, -0.812822, 0.240949], [0.059294, 0.043518, 0.100743, -0.105259, 0.100519, 0.033779, -0.133747, -0.4, -0.425177, -0.639166, 0.109859, 0.079832, 0.128668, 0.143403, -0.394961, -0.028054], [0.083313, -0.016489, -0.133353, -0.326814, -0.005322, -0.001827, 0.261406, -0.166413, -0.167197, -0.26668, -0.523315, 0.066133, -0.129355, -0.113562, -0.575907, 0.124385], [-0.455081, -0.013348, -0.045595, -0.016915, -0.325185, 0.159289, -0.102664, 0.581496, 0.183028, 0.020466, -0.08441, -0.123239, 0.062417, -0.260125, 0.037539, -0.382958], [-0.192699, -0.152436, -0.114283, -0.329967, -0.088678, 0.218895, -0.076383, 0.078251, -0.265243, -0.105904, -0.211071, -0.257479, 0.011588, -0.248812, -0.205677, -0.266577], [0.080254, 0.125459, 0.034126, 0.02899, 0.369042, -0.080245, -0.030352, 0.261176, 0.121269, 0.316982, 0.257521, 0.140454, 0.250099, 0.196671, 0.320772, -0.028324], [0.023136, 0.02448, -0.073649, 0.372745, 0.045547, 0.068906, -0.071341, -0.19141, 0.206294, 0.115319, -0.05328, 0.117331, 0.097418, 0.24321, 0.404937, -0.06443], [0.073976, 0.045391, -0.030292, 0.324158, 0.200906, 0.188922, -0.305318, -0.463166, -0.180318, -0.169441, -0.195865, 0.217062, -0.063431, 0.029499, -0.272129, -0.005159], [0.341618, 0.126107, 0.019035, 0.242404, 0.189479, -0.03415, -0.145136, 0.040621, 0.372978, 0.204003, 0.279473, 0.211418, 0.08266, 0.380293, 0.256689, 0.177123], [-0.19847, 0.000708, -0.150501, -0.086431, -0.171494, 0.150592, 0.032795, -0.077246, 0.020801, -0.187316, -0.416374, -0.116646, -0.38055, -0.196095, -0.539222, -0.341595], [-0.156427, 0.042701, -0.240951, -0.486748, 0.020995, 0.040159, 0.084, -0.128756, 0.047284, -0.010778, 0.00038, 0.037005, -0.206648, 0.054217, 0.080833, -0.07183], [0.231606, 0.133162, -0.047461, -0.193119, -0.174911, -0.258099, 0.426256, 0.007726, -0.028124, -0.396186, -0.255037, -0.165657, -0.294125, 0.060907, -0.100493, 0.250534], [-0.199919, -0.002002, 0.008114, 0.403914, 0.077318, -0.055565, 0.104422, 0.202378, 0.164282, 0.262387, 0.079306, 0.200775, 0.149934, 0.1163, -0.127956, 0.04883], [-0.038387, -0.185268, -0.0852, 0.116492, 0.30419, 0.123003, -0.397168, 0.129662, 0.333071, 0.121348, -0.123992, -0.003854, 0.278543, 0.090715, 0.023886, -0.133772], [-0.192422, -0.220606, 0.111772, 0.161771, -0.308815, -0.097045, -0.161832, 0.152531, 0.008747, -0.151173, 0.249494, -0.204343, -0.23794, -0.386366, 0.019749, -0.235737], [-0.177355, -0.368535, -0.101581, -0.161229, -0.385199, 0.453928, 0.043165, 0.112859, -0.253384, 0.067509, -0.113411, -0.251772, -0.432847, 0.056592, -0.10979, 0.147826], [0.141684, -0.083731, 0.001025, -0.342417, 0.228107, 0.15254, 0.040225, 0.244122, -0.306002, -0.169254, 0.014226, 0.175355, 0.048431, 0.03956, -0.248891, -0.18248], [-0.084344, -0.027589, -0.152247, 0.094778, -0.370334, 0.106061, -0.043708, 0.046389, 0.03021, 0.074479, 0.22978, -0.118425, -0.246398, -0.255713, 0.11965, -0.380683], [0.362647, -0.080055, -0.387139, 0.136783, -0.521524, -0.203267, 0.105263, -0.066815, 0.140841, 0.039514, 0.206678, 0.12653, -0.009522, -0.114336, -0.013287, 0.173804], [0.095196, 0.037445, 0.026966, -0.15085, 0.129089, -0.106308, 0.011458, -0.159905, -0.061119, -0.558993, -0.038598, -0.019491, 0.106834, 0.159734, 0.204847, 0.220348], [0.020045, 0.142767, 0.053362, -0.064638, 0.001453, 0.163414, -0.055425, 0.364926, 0.293485, 0.172302, -0.059005, 0.345728, 0.210391, 0.051618, 0.014723, -0.001809], [0.231532, 0.100112, 0.093821, 0.361541, 0.237378, -0.071783, 0.013542, 0.230097, 0.319335, 0.191205, -0.286356, 0.034932, 0.304132, 0.170835, 0.197851, 0.161505], [-0.075199, 0.011121, 0.187721, -0.074877, 0.209604, -0.11347, 0.011403, 0.003546, -0.06381, 0.145631, 0.1322, 0.097132, 0.206159, 0.234852, 0.17234, -0.162289]], [-0.039631, 0.501755, 0.284939, -0.026697, -0.189119, 0.141774, 0.133675, -0.375185, 0.309076, 0.226762, 0.076376, 0.048267, -0.097495, -0.100448, -0.098417, -0.135132]),
    ([[-1.058728], [-0.252424], [-0.438026], [0.728344], [-0.876234], [-0.389521], [-0.333616], [0.767832], [0.774839], [0.621087], [0.38254], [-0.644937], [-0.517826], [-0.431747], [0.793], [-0.917306]], [0.412062]),
]
R4_SLW_STD1 = ([0.0, 0.0, 0.0, 0.0, 0.0, 5.068810841538566e-19, 0.0, 0.0, 0.0, -2.534405420769283e-19, 0.0, 0.0, 0.0, 1.2354840004101888, 0.0, 10.235373650724295, 29.045780629283016], [6.894880047762485, 19.932973433822138, 35.828844117459674, 46.28838112885247, 59.58345850504529, 1.1288535466346525, 0.9972382401475413, 0.9509100251726608, 0.8477614371106685, 0.7566240757362533, 1065.974830301333, 2039.56663015695, 4012.5353057956445, 0.5930296144511749, 0.3020581511144633, 17.10440165990182, 36.041044035504086], [0.566097143562707, 0.259264995938383, 0.003998079933218471, -0.18461439386994238, -0.09909113913895669, 0.8100942713400355, -2.814201129099548, 5.70728211279055, -3.713167888183222, 3.3848996757781586, -0.545115637040028, 0.02632223148335277, -0.18305262194727148, 0.001881757544255653, 0.25602824974821586, -0.0002618872582300937, -0.00019672322469201564], 0.0)
R4_SLW_STD5 = ([0.0, 0.0, 0.0, 0.0, 0.0, 5.074727552069564e-19, 0.0, 0.0, 0.0, -2.537363776034782e-19, 0.0, 0.0, 0.0, 1.2357253973320002, 0.0, 10.24382157388333, 29.06687743136384], [6.898175212270779, 19.94432111131209, 35.84821729762494, 46.31374990156119, 59.61458245938529, 1.1292479320432283, 0.997614069404236, 0.9512769517801772, 0.8481040200541816, 0.756949523396557, 1064.2421889182478, 2038.6600009884921, 4007.1636750665784, 0.5932910613542522, 0.3022343925906883, 17.112570355001356, 36.056623057820026], [2.6498341383747235, 0.8365014691593934, -0.25579914674007964, -1.4326632881404089, -0.7070860011656996, 1.5376983088542304, -4.666766920955899, 9.850415060556683, -6.38626239553004, 7.444746721364866, -2.2326835524909727, -0.07363618221570306, -1.293430464591002, 0.010452110942003077, -1.0561616959005966, -0.0014540827961403151, -0.0010915738303765145], 0.0)
R4_MLPQ = {}
R4_MLPQ[500] = ([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.235968, 0.0, 0.0, 5.908278, 5.908217, 0.0, 7e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009079], [1.133464, 0.999389, 0.952272, 0.847731, 0.756024, 2.569325, 20.907391, 0.789165, 18.42144, 0.707427, 0.649163, 0.542068, 1.664649, 1.664601, 0.999991, 4.40323, 1.165561, 0.30371, 133.77359, 335.203674, 1064.987427, 1.340487], [([[-0.006898, -0.01135, 0.023763, 0.00422, -0.121094, 0.281272, 0.162643, 0.121966, 0.040366, 0.284437, 0.118828, 0.019081, 0.142668, 0.035105, 0.277402, -0.182055, -0.333358, -0.277113, -0.119759, -0.118229, -0.118237, -0.584428, -0.135276, -0.194045, -0.018787, -0.011941, 0.04542, 0.237056, -0.152817, 0.37892, 0.184053, 0.028213], [-0.19216, 0.185446, 0.045485, -0.157048, -0.23047, -0.17158, -0.097488, -0.071917, -0.056814, 0.224483, -0.225684, -0.162839, 0.176643, -0.073183, 0.187533, -0.086867, -0.119958, -0.051308, 0.124766, -0.10187, -0.100346, -0.135032, 0.163344, -0.069368, -0.232314, -0.012, -0.132082, 0.098594, -0.21317, -0.035276, 0.106095, 0.14148], [-0.073575, 0.016494, 0.000558, 0.029055, -0.11908, -0.116075, 0.058391, -0.10146, 0.064645, -0.14532, 0.118166, 0.196451, -0.161724, 0.14794, 0.144192, 0.028852, 0.139131, -0.177523, 0.076667, 0.12402, 0.038249, 0.11678, 0.266969, 0.019684, -0.027093, 0.148427, -0.012169, 0.207924, -0.035819, -0.195949, -0.245871, 0.198429], [0.084376, 0.005838, -0.186514, 0.296011, -0.107062, 0.004419, -0.018206, -0.180846, 0.226352, -0.145447, 0.335331, 0.163681, -0.398008, 0.044852, -0.121074, 0.261861, 0.249109, 0.222223, 0.155623, -0.224604, 0.226016, 0.073241, 0.221338, 0.115421, -0.068593, 0.311138, -0.052642, 0.050222, 0.111638, -0.006959, -0.046352, -0.174727], [0.313246, 0.085109, 0.215469, -0.357993, 0.051103, -0.351169, 0.010729, 0.345831, -0.428949, -0.333909, -0.052555, -0.267996, -0.096091, -0.565981, -0.215013, -0.285006, 0.111831, -0.03821, 0.366907, 0.446065, -0.516754, 0.174385, 0.359433, 0.184559, 0.328809, -0.288926, 0.326073, 0.029682, -0.127399, -0.008015, -0.39913, 0.386952], [-0.050478, 0.002344, 0.075878, 0.212791, -0.138468, -0.271574, -0.202051, 0.224094, -0.13804, -0.255865, -0.210132, 0.080615, 0.16818, 0.077524, 0.084695, -0.049613, -0.229106, 0.185471, -0.00412, -0.138381, 0.10435, -0.030105, 0.313871, -0.010455, 0.005263, -0.087209, 0.163473, 0.330715, -0.04081, -0.337886, -0.24269, -0.040832], [-0.026106, -0.019762, 0.212713, -0.061415, 0.320519, 0.16519, -0.000731, 0.172961, -0.430274, -0.205562, -0.1414, -0.611023, -0.132209, -0.650771, -0.199458, 0.140001, 0.41597, 0.494502, 0.241988, 0.026491, -0.423876, -0.099293, 0.183647, 0.234882, 0.313317, 0.051822, 0.171644, 0.041886, 0.21061, 0.600977, -0.098473, 0.090976], [-0.171186, -0.102776, 0.026194, 0.311426, 0.143325, 0.092721, -0.170055, 0.197535, -0.039608, 0.389112, -0.099099, -0.008645, 0.136485, 0.096509, 0.014442, -0.123506, 0.031573, -0.013746, 0.132481, -0.486881, 0.030576, -0.006972, -0.007266, -0.074486, 0.207229, 0.006291, 0.103296, 0.181736, -0.022626, 0.107454, 0.152435, -0.18088], [0.018713, -0.25715, 0.178259, 0.047878, -0.316199, -0.253691, -0.127901, 0.168686, -0.313395, -0.192564, 0.094403, -0.496802, 0.29826, 0.267212, -0.189285, 0.048421, 0.084048, 0.116841, -0.047256, -0.17322, 0.437266, -0.282503, -0.227367, -0.121848, 0.217275, -0.060691, -0.09137, 0.014816, 0.195566, 0.264693, 0.060315, 0.225959], [-0.263945, -0.616252, 0.153861, 0.04645, -0.47349, 0.154867, 0.004199, -0.205967, -0.242834, 0.112186, -0.216442, -0.147351, -0.07057, -0.024872, -0.354377, -0.055931, -0.203845, 0.248437, -0.149916, -0.023228, 0.058174, 0.227698, -0.183432, -0.342861, 0.042122, 0.174896, -0.114335, 0.153736, -0.75606, 0.064254, -0.043697, 0.098863], [0.144444, -0.043757, -0.09477, -0.003398, 0.013548, -0.039858, 0.241353, 0.150404, -0.166422, -0.132916, -0.00585, 0.026981, -0.084983, 0.029639, 0.140148, 0.156305, -0.091563, -0.101286, 0.1018, 0.199799, -0.122873, 0.180022, -0.176574, 0.011979, -0.036225, 0.084105, -0.09844, 0.113286, -0.127586, 0.147001, 0.044238, -0.041594], [-0.008117, -0.000147, 0.256386, -0.182231, -0.200492, -0.002739, 0.527542, 0.142895, -5.8e-05, -0.162227, 0.070446, -0.516588, 0.156881, 0.05252, -0.075635, -0.321833, -0.242182, 0.234297, -0.462533, -0.087481, 0.106849, -0.183458, -0.09691, 0.501397, -0.042743, -0.296873, 0.001929, 0.04231, 0.130556, -0.171737, -0.07739, 0.287154], [-0.030051, 0.222674, -0.202416, -0.064768, -0.031444, -0.130211, 0.088657, 0.07205, 0.361255, -0.197175, -0.053942, -0.011053, 0.09661, -0.078195, -0.336144, -0.014969, 0.359515, -0.166665, 0.193911, -0.017726, 0.122395, 0.02655, 0.135455, 0.362645, -0.147985, 0.106178, 0.13086, -0.094937, 0.120213, -0.057488, 0.100787, -0.327265], [0.130871, 0.079747, 0.114667, -0.162623, 0.01924, 0.339211, 0.034226, -0.066934, 0.039131, 0.037212, -0.057116, 0.062974, 0.19566, 0.030203, -0.269248, -0.01087, 0.146219, -0.128679, 0.540731, 0.075411, -0.046026, -0.144164, 0.094243, 0.327432, 0.030279, -0.098156, -0.13984, -0.223931, 0.134422, 0.033634, 0.002298, -0.086449], [-0.216546, 0.06408, 0.119761, 0.203923, -0.415774, 0.20263, 0.30452, -2.9e-05, 0.062407, -0.092559, -0.108555, -0.146691, -0.103535, 0.009197, -0.047654, 0.053794, 0.116927, 0.061723, 0.028935, 0.232684, -0.119637, -0.10923, -0.111825, -0.11846, -0.351082, -0.20075, 0.249308, -0.054846, 0.154802, 0.168554, 0.09437, 0.141053], [0.009178, 0.017393, 0.24361, 0.155788, -0.052795, -0.046852, 0.468795, 0.050969, -0.120539, -0.123005, -0.176512, 0.2031, 0.171381, 0.004791, 0.154077, -0.101286, -0.139808, -0.07082, -0.27635, -0.379216, 0.276464, 0.12562, -0.033592, -0.127245, 0.03936, 0.132347, -0.141299, 0.039202, -0.08117, 0.011888, 0.101656, 0.14826], [0.360621, -0.155404, 0.085285, -0.276365, 0.602372, -0.178076, 0.017834, -0.133767, -0.034886, 0.041083, -0.156273, 0.45447, 0.404397, -0.047987, 0.227432, -0.350659, -0.250066, 0.201995, 0.098966, -0.138291, -0.117145, 0.113446, -0.029541, 0.127909, -0.26119, -0.043831, -0.126261, -0.176706, 0.066208, -0.053764, -0.27873, -0.195609], [-0.232079, 0.069298, -0.220877, -0.02048, 0.000477, 0.193968, -0.037439, 0.197131, 0.189638, 0.051414, -0.076742, -0.057794, 0.075522, -0.040285, -0.322153, 0.022428, -0.178843, 0.333276, 0.090385, 0.289811, -0.028476, -0.052418, 0.406096, -0.044208, -0.071558, 0.137772, -0.215248, -0.032996, 0.004342, 0.049052, -0.157909, 0.157515], [0.006106, -0.004514, -0.046276, 0.091515, 0.05511, 0.033964, 0.004532, 0.201101, 0.035425, 0.013195, 0.010512, -0.01509, 0.014018, -0.005671, 0.114539, -0.0581, -0.128566, -0.067834, -0.011917, 0.088951, 0.016092, 0.002017, 0.14889, -0.217306, 0.02362, 0.025891, 0.024339, -0.029398, 0.093532, 0.045723, -0.00767, 0.09168], [0.314796, -0.026867, 0.124367, -0.21005, 0.14819, -0.32235, -0.107522, 0.021091, -0.025919, 0.060676, -0.142728, -0.095351, -0.0071, -0.136904, 0.059281, 0.039862, -0.113973, 0.158781, -0.143148, -0.287219, 0.189647, -0.013473, 0.141675, 0.330872, 0.914156, -0.149708, -0.021428, 0.129655, -0.282061, -0.069155, -0.204261, -0.190986], [-0.433609, 0.546334, 0.450915, 0.108662, -0.12187, -0.345567, -0.229713, -0.137053, 0.140186, 0.099232, -0.155886, 0.122799, 0.028638, 0.602638, -0.071989, -0.062527, -0.115532, 0.228139, -0.278803, -0.058807, 0.214605, 0.168014, -0.358086, -0.192242, -0.074498, -0.379009, -0.522843, -0.082536, -0.085408, -0.198139, 0.221474, -0.013], [0.040793, -0.081267, 0.044455, 0.021963, 0.060856, -0.169532, -0.005918, -0.111436, -0.086775, 0.190351, 0.073679, 0.472825, -0.343589, 0.01261, 0.025121, -0.000978, -0.248237, -0.128065, 0.014146, -0.011311, 0.358983, 0.107017, -0.021649, 0.029036, -0.023125, -0.040711, 0.200929, 0.11506, -0.116181, 0.412247, -0.059014, 0.07051]], [-0.529106, -0.026835, -0.03812, -0.136223, 0.099815, -0.176419, -0.069994, 0.097455, 0.401246, -0.174535, 0.35516, 0.067691, -0.465978, 0.286237, 0.271582, 0.200758, -0.349607, 0.359241, 0.27694, -0.068791, -0.00292, 0.056427, -0.371305, -0.024854, 0.224947, -0.227092, 0.296372, -0.247784, -0.104242, 0.172701, -0.187908, -0.059534]), ([[0.105818, -0.039392, 0.269491, -0.045322, 0.104673, -0.189062, 0.003265, -0.632449, -0.291946, -0.456618, 0.192385, -0.039907, 0.15017, 0.327226, 0.779006, 0.151817], [-0.461137, 0.092986, -0.044881, 0.336627, -0.312451, 0.039755, 0.071838, 0.07846, 0.059145, -0.085598, 0.367953, -0.019712, 0.139535, -0.017306, 0.101039, 0.072008], [-0.117618, 0.047773, -0.135694, 0.17998, 0.103735, -0.166524, 0.092991, 0.128891, 0.199979, -0.051531, 0.049713, 0.084808, -0.322245, -0.42609, 0.164884, 0.243598], [-0.105339, 0.083584, -0.292104, -0.28412, -0.056083, -0.079528, 0.072045, -0.521667, -0.298925, -0.147913, 0.075467, 0.044271, -0.505219, -0.0919, -0.250064, -0.230066], [0.051886, 0.132959, -0.056796, 0.389099, -0.004818, 0.176506, 0.038433, 0.198672, -0.534998, -0.578075, 0.19523, 0.186845, 0.079311, 0.11682, -0.260001, -0.070105], [0.177279, -0.140016, -0.050969, -0.22228, -0.23131, -0.137285, -0.063596, 0.229839, -0.253257, 0.154934, 0.134876, 0.111361, 0.166549, 0.139852, -0.137811, -0.064601], [-0.167864, -0.048646, 0.141755, -0.26682, -0.134046, 0.012457, 0.077886, -0.141843, -0.335875, -0.266701, 0.130628, 0.032767, -0.163409, -0.456437, -0.134977, -0.063378], [-0.273063, 0.002844, 0.264457, 0.171437, 0.070377, 0.041016, -0.017098, -0.071227, 0.064676, 0.114766, 0.132427, -0.155979, -0.194763, 0.16971, 0.059694, -0.69835], [-0.050996, -0.238996, -0.019611, -0.509112, -0.042025, 0.037416, -0.063726, -0.2184, -0.997655, -0.567865, -0.182058, -0.026208, 0.037524, -0.075561, -0.040468, -0.04441], [-0.001692, 0.092422, 0.264503, 0.465454, 0.073836, -0.166616, -0.02942, -0.307613, -0.175457, 0.224646, -0.352837, 0.167372, 0.115977, 0.094056, 0.294871, 0.246723], [0.201116, 0.059109, 0.232642, -0.2052, -0.076594, -0.106914, -0.092193, 0.003249, -0.161675, -0.335314, 0.118311, 0.117684, 0.022917, 0.059411, -0.38828, 0.216588], [-0.021434, -0.129498, 0.01763, -0.023513, 0.268847, -0.13102, 0.139084, -0.06369, -0.278653, -0.150517, -0.197323, -0.130668, -0.231541, 0.218495, -0.31093, 0.020221], [-0.300761, -0.025509, -0.182639, 0.322787, -0.651116, 0.209111, -0.246977, 0.081301, 0.022745, -0.09382, -0.06282, -0.119593, -0.263349, -0.30208, 0.551946, -0.338365], [-0.033417, -0.062497, -0.279349, -0.23723, 0.228669, 0.254371, -0.019341, -0.741734, -0.254246, -0.559587, -0.348438, -0.195757, -0.054603, 0.03078, 0.068495, -0.084432], [0.129973, 0.217688, -0.012273, -0.139733, 0.104019, -0.098374, -0.246775, 0.059752, 0.101691, 0.147468, 0.144703, 0.08522, 0.327378, -0.502815, 0.249538, -0.005424], [0.170606, 0.051109, 0.091249, 0.050336, -0.038535, 0.001965, 0.015141, -0.117813, -0.071075, 0.03036, 0.168102, 0.069063, 0.032405, 0.194933, 0.121214, 0.102225], [-0.262074, -0.070142, -0.015288, -0.271409, 0.390561, 0.144796, -0.197684, -0.022206, -0.190613, -0.289386, -0.212909, 0.268905, -0.112804, 0.184409, -0.586041, 0.159184], [0.343903, 0.077054, -0.223161, -0.070631, -0.253582, 0.043412, -0.074271, 0.046338, 0.216128, 0.031717, -0.115212, 0.027119, 0.191208, 0.209512, 0.060595, 0.19303], [-0.426762, 0.10498, -0.177212, -0.176187, -0.122336, -0.109094, 0.15861, -0.377207, 0.079678, -0.208182, -0.232647, -0.055372, -0.154223, -0.276275, 0.046168, -0.455803], [-0.216927, 0.096936, -0.138926, -0.838378, 0.07659, 0.116774, 0.000139, 0.145443, -0.121372, 0.120616, -0.054825, 0.025484, -0.213397, 0.131386, 0.117818, 0.125746], [0.085794, -0.063971, -0.018205, -0.183952, -0.07871, -0.757435, 0.236559, -0.181792, -0.344167, -0.406139, -0.303116, -0.301565, -0.328299, 0.076361, 0.225004, 0.151819], [0.132014, 0.023899, 0.149493, 0.486329, 0.115626, -0.161445, 0.029821, 0.209866, 0.245169, 0.160334, -0.01529, 0.111432, 0.217931, 0.091802, -0.01254, 0.148415], [-0.113953, -0.300316, -0.378861, 0.182403, 0.421576, 0.235989, -0.241633, 0.174004, 0.270814, 0.026879, 0.010497, 0.01037, 0.205528, 0.132496, 0.171895, -0.444922], [-0.259426, -0.079669, 0.112377, 0.235916, -0.034372, -0.021334, -0.079323, 0.058541, -0.00764, 0.063882, 0.10043, -0.087252, -0.032724, -0.126014, 0.187226, -0.253974], [-0.16702, -0.247545, 0.073633, -0.114268, 0.036023, 0.116441, 0.104719, 0.060235, -0.124395, 0.091433, -0.154404, -0.522724, -0.746876, 0.193732, 0.087265, -0.102142], [0.397027, -0.022031, 0.094076, -0.325465, 0.067549, 0.116112, 0.137973, 0.160351, -0.269663, -0.031018, -0.049755, 0.125549, 0.24874, -0.172399, 0.04613, -0.008276], [-0.262835, -0.003646, -0.11384, 0.064583, -0.068169, 0.167557, -0.090502, -0.330916, 0.150662, 0.032056, 0.087571, 0.026272, -0.25694, -0.226, -0.078353, -0.615739], [0.29974, -0.093737, -0.205914, 0.148184, -0.153929, -0.061588, 0.053555, 0.035029, 0.238927, 0.019246, 0.135, 0.044182, -0.055393, 0.348993, 0.05172, 0.074659], [0.192741, 0.038102, 0.103149, -0.255371, 0.247157, 0.035024, -0.124386, 0.042722, 0.051178, -0.330868, 0.190495, -0.006817, 0.08671, 0.18624, -0.422336, 0.308702], [0.048537, -0.064125, 0.137132, -0.098935, -0.138922, -0.025733, 0.053855, 0.424568, 0.166963, 0.342601, 0.240311, 0.125357, 0.086281, -0.008305, -0.227193, 0.133202], [0.22348, 0.079484, 0.115378, 0.191075, 0.022425, 0.187548, 0.129486, -0.316117, 0.294171, 0.041372, -0.552257, 0.008626, 0.34545, 0.029569, 0.477363, 0.142885], [-0.190942, -0.105598, -0.025565, 0.052899, 0.18741, -0.049662, -0.16738, 0.129765, 0.026168, 0.086871, 0.153666, 0.060738, 0.283803, 0.223318, 0.276028, -0.504658]], [0.074595, 0.379878, 0.237562, -0.097735, -0.242652, 0.176373, 0.315203, 0.143906, 0.278298, 0.340427, 0.087237, 0.190805, -0.045697, -0.339541, -0.130864, -0.033558]), ([[-1.891328], [-0.255113], [-0.715978], [1.732278], [-1.709194], [-0.577428], [-0.47028], [1.622205], [1.247224], [1.295154], [0.726284], [-0.98295], [-1.400439], [-1.312037], [0.678966], [-0.778168]], [0.476753])])
R4_MLPQ[1000] = ([0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.233418, -0.0, -0.0, 5.906039, 5.906329, 0.0, -2e-06, 0.0, 0.0, 0.0, 0.0, -0.001637, 4.05786], [1.020024, 0.879398, 0.830559, 0.723165, 0.628344, 2.281009, 12.975401, 0.674001, 10.695116, 0.757238, 0.651143, 0.545656, 1.65171, 1.651716, 0.999999, 4.399457, 1.056413, 0.427436, 127.460831, 323.726532, 938.366638, 1.343881], [([[0.02628, -0.363432, -0.300187, -0.082597, 0.038718, 0.020264, -0.320384, 0.216757, 0.081505, -0.130916, -0.089847, -0.075299, 0.165005, -0.04158, -0.040825, -0.004421, -0.081376, -0.287049, 0.275215, 0.191288, -0.253308, 0.15291, -0.02153, -0.226823, -0.036074, -0.21859, 0.000498, -0.047929, 0.197176, -0.245651, 0.152467, -0.21688], [0.05845, 0.146292, -0.177079, 0.063993, -0.218867, 0.027406, 0.085844, -0.240855, -0.073286, 0.015702, 0.117258, -0.000693, -0.153014, -0.003328, -0.121513, -0.167089, 0.003922, -0.092678, -0.024726, 0.270396, -0.239695, -0.018534, 0.080929, 0.016386, -0.079618, -0.184348, 0.021911, 0.130295, 0.078742, 0.184818, 0.025427, -0.043058], [-0.30063, 0.034765, 0.05708, 0.148024, 0.021363, -0.271922, 0.369315, 0.037669, 0.136034, 0.099037, -0.042406, 0.143921, -0.123686, -0.17832, -0.221682, 0.200641, -0.008565, 0.27138, -0.142181, 0.011812, -0.135575, 0.166644, 0.057843, -0.0359, 0.27496, 0.253812, 0.16198, -0.120128, 0.217159, -0.045631, 0.123004, 0.082698], [-0.32552, 0.354015, 0.131981, 0.138686, 0.219961, 0.196572, 0.172469, -0.242936, -0.218583, 0.31725, -0.095255, -0.150574, 0.081545, -0.111023, -0.082453, 0.063401, 0.060792, 0.306757, 0.090175, -0.314043, 0.023655, 0.247365, 0.008721, 0.27986, 0.26359, -0.014763, 0.269334, 0.103991, -0.141654, -0.155781, -0.474363, 0.160407], [-0.243658, 0.183422, -0.316769, -0.327856, 0.492256, 0.443811, -0.008897, 0.009047, -0.002257, -0.078222, -0.359639, -0.009449, 0.449115, -0.070318, 0.235567, -0.091651, 0.0662, 0.290091, 0.046117, -0.085242, -0.309345, -0.190035, 0.003758, -0.149202, -0.188949, -0.053089, -0.232694, 0.010492, -0.55096, 0.127428, 0.192042, 0.305204], [-0.095277, -0.180898, -0.324454, 0.061605, -0.008912, -0.449429, 0.088346, -0.194214, -0.079898, -0.103524, 0.054892, -0.251713, -0.014354, -0.014501, -0.050379, 0.067322, 0.00466, 0.216744, -0.114633, 0.130761, 0.076533, -0.123817, -0.17226, 0.046684, -0.108438, -0.063985, 0.004123, 0.256789, 0.284225, 0.119876, 0.277011, 0.01993], [-0.095473, -0.315475, 0.463227, -0.244939, -0.023855, -0.181832, 0.024547, 0.111438, -0.09819, -0.279002, -0.551772, 0.297573, -0.161755, 0.142754, -0.195436, 0.31479, -0.026181, 0.285049, -0.257257, -0.019012, -0.030502, -0.203907, 0.148703, 0.258732, 0.290958, 0.405735, -0.19158, 0.256921, 0.31023, -0.075333, 0.16206, 0.503936], [0.00272, 0.043862, 0.144085, -0.148631, -0.054564, 0.015429, 0.160624, -0.038062, 0.131628, 0.150817, -0.116925, 0.028317, -0.138027, -0.152113, -0.128935, 0.125654, 0.149176, -0.071992, 0.098724, -0.120242, 0.165609, -0.015071, -0.00996, 0.100052, -0.07292, -0.185856, 0.224972, -0.208831, 0.043069, 0.15682, 0.126995, -0.03858], [-0.29343, -0.223058, -0.151768, -0.40794, 0.623311, 0.030578, 0.342411, -0.382211, -0.336424, 0.238355, 0.348005, 0.333598, 0.146131, -0.254101, 0.066573, 0.178002, -0.286081, 0.688201, -0.000765, 0.159772, -0.202533, -0.219824, 0.09108, 0.061968, -0.70461, 0.131763, -0.232212, 0.279717, -0.194941, -0.354014, 0.441058, -0.181559], [0.045629, 0.009733, 0.114764, 0.327012, 0.122201, 0.081272, 0.251553, -0.517828, 0.17387, -0.218944, 0.088404, 0.159648, -0.054696, -0.270814, 0.208802, 0.200893, -0.317052, 0.05717, -0.14567, -0.677809, -0.166985, 0.443095, -0.222697, -0.196397, 0.118669, -0.137116, 0.074644, 0.25702, -0.186534, 0.420972, 0.394483, 0.016425], [-0.052515, -0.162131, -0.143092, -0.020105, -0.115492, 0.049815, 0.001105, -0.078825, 0.000356, -0.134416, 0.165267, -0.106918, -0.158789, 0.18022, -0.070934, -0.13067, -0.198709, -0.119138, 0.234674, 0.061093, 0.259348, 0.207043, -0.001345, -0.174668, -0.273214, -0.248124, -0.265936, -0.268493, 0.047131, -0.18379, -0.196665, 0.115494], [-0.01605, -0.044181, -0.032653, -0.166552, -0.172159, 0.069027, 0.196365, -0.21078, -0.315097, 0.073504, 0.237534, 0.098486, 0.135164, -0.26724, 0.231416, -0.052619, 0.281307, 0.229526, 0.283751, 0.169034, 0.176718, -0.134988, 0.236663, 0.102997, 0.173543, 0.073607, -0.017507, 0.366619, 0.188716, 0.067564, 0.28873, 0.324112], [-0.011199, -0.178059, -0.122834, -0.033347, 0.06706, -0.090017, -0.273225, -0.104933, -0.110172, 0.115721, 0.30494, -0.319958, -0.14688, 0.219057, -0.170353, -0.122106, 0.145902, -0.024618, 0.6517, -0.143798, 0.18282, 0.063265, -0.111269, 0.128013, -0.13456, 0.096144, -0.009701, 0.227374, -0.123798, -0.017959, 0.10917, -0.121667], [-0.134427, -0.372124, -0.17819, -0.328052, 0.006634, 0.062286, -0.105225, 0.087762, 0.515324, 0.053265, 0.09412, -0.227193, -0.227417, 0.124036, 0.033106, 0.047096, 0.416823, 0.022654, 0.105586, 0.001457, -0.087033, -0.061514, 0.004436, 0.083336, 0.108088, 0.387937, 0.014007, -0.032625, 0.19962, -0.201453, 0.003172, -0.213997], [-0.00646, -0.087321, 0.010951, -0.500518, 0.148553, 0.085019, 0.36274, -0.059055, 0.080976, 0.284897, 0.271682, -0.154593, -0.043661, -0.243697, -0.202668, -0.123819, 0.095993, 0.053428, -0.098835, 0.174665, -0.103639, -0.167226, -0.143671, 0.045884, 0.003199, -0.07096, -0.037515, -0.424465, 0.094738, 0.225586, -0.000246, 0.143615], [-0.211876, 0.155558, -0.060225, 0.360384, 0.233871, -0.085613, 0.122478, -0.036609, -0.004193, 0.020714, -0.043183, 0.229626, 0.179323, -0.124512, 0.129707, 0.44697, 0.136207, 0.158334, 0.134428, 0.258819, -0.13384, 0.306995, 0.267681, -0.171126, -0.362791, 0.226214, 0.247905, -0.347478, -0.121306, -0.143562, 0.1485, -0.075593], [0.016195, 0.065335, -0.090405, 0.593343, -0.160357, 0.245916, -0.294861, 0.45475, 0.274847, -0.003179, 0.124144, -0.522358, 0.101318, -0.574684, 0.295705, 0.114986, -0.227724, -0.131908, 0.466738, -0.063662, -0.417295, -0.095587, -0.175758, -0.305539, -0.064856, -0.526029, 0.532316, 0.03199, -0.023737, -0.00111, 0.340184, -0.026499], [0.156406, 0.186682, 0.223339, -0.447671, 0.271473, -0.244169, -0.033076, -0.352023, -0.349853, 0.367781, -0.257416, -0.265984, 0.086283, -0.309158, 0.11687, 0.339581, -0.292134, -0.232568, 0.11837, 0.073447, 0.083649, 0.095229, -0.131998, -0.038081, 0.041189, -0.082056, -0.145258, 0.246849, 0.350962, -0.309929, -0.00408, 0.288811], [-0.036384, -0.017552, -0.009032, 0.113676, -0.062519, 0.163813, -0.033, 0.104698, -0.030389, 0.077511, -0.042789, -0.029519, -0.070973, -0.022979, 0.193832, -0.308145, -0.144572, -0.074973, -0.199587, -0.101701, -0.011183, -0.276206, 0.053081, 0.064541, -0.093552, 0.10659, -0.092737, 0.081513, 0.006635, 0.079461, 0.047269, 0.030704], [-0.021999, -0.117211, 0.075734, -0.143001, -0.250168, -0.061696, -0.110454, 0.019545, -0.119176, -0.19909, -0.003257, 0.174502, 0.195423, 0.023776, 0.293199, 0.265261, 0.301452, -0.029184, -0.069606, -0.010445, 0.002943, 0.029006, 0.722559, -0.327656, -0.128888, 0.261586, -0.231636, 0.108865, -0.010676, 0.219569, 0.118775, 0.199565], [0.165229, -0.007146, 0.300859, -0.155253, -0.031882, -0.031798, -0.126467, -0.119002, 0.030983, 0.357428, 0.032603, 0.522197, 0.048694, -0.02594, 0.014005, -0.040425, -0.00238, -0.097633, -0.104104, 0.321051, -0.161289, -0.014594, 0.114032, 0.136863, -0.532133, 0.081785, -0.255012, -0.075671, -0.088848, -0.436953, 0.061618, -0.33933], [0.064095, -0.097559, 0.084855, -0.23899, -0.041873, 0.399672, 0.279778, -0.180052, 0.356163, -0.090146, -0.031446, 0.318742, 0.026262, 0.191605, -0.217917, 0.300327, -0.117108, -0.048977, 0.088357, -0.186315, -0.273792, 0.159073, -0.171125, -0.343617, 0.013256, -0.119513, 0.216895, 0.572569, -0.017171, 0.18691, -0.079317, 0.24992]], [-0.585839, -0.296393, 0.313879, -0.339998, -0.486788, -0.349877, -0.391199, -0.394468, -0.534858, -0.674263, 0.540821, -0.253944, -0.029608, -0.377191, 0.245742, -0.022026, 0.546175, -0.364443, -0.505994, -0.111708, -0.453764, -0.221011, 0.106119, -0.492546, 0.030037, -0.30642, -0.335051, -0.144727, -0.291141, -0.101174, -0.077033, -0.022093]), ([[0.036764, -0.212588, 0.33421, 0.150965, 0.327239, 0.247855, 0.044371, -0.014708, 0.032804, 0.22044, 0.020667, 0.155674, -0.074105, -0.266989, -0.184741, -0.16641], [-0.066874, -0.205631, 0.058292, 0.069414, 0.053751, -0.098852, 0.376369, 0.041605, 0.132577, 0.056221, 0.087704, 0.288081, -0.180692, -0.502556, 0.078443, -0.127009], [0.130702, 0.041429, 0.093183, -0.294784, 0.019126, 0.135138, 0.231576, 0.23535, 0.208976, -0.070235, 0.170349, 0.045112, -0.140339, -0.244974, -0.10764, -0.860864], [-0.374669, -0.292811, -0.271849, 0.136223, -0.376925, 0.003759, -0.037764, -0.526862, -0.119882, -0.068803, -0.11129, 0.132884, -0.054411, -0.134551, 0.434609, -0.059965], [-0.452771, -0.085834, -0.167462, 0.005745, -0.163734, 0.052674, 0.100416, -0.186056, 0.199775, -0.259994, 0.066491, -0.755548, -0.180281, -0.506632, 0.246027, 0.16012], [0.169697, 0.139546, 0.130209, -0.089849, 0.038466, 0.165281, 0.069553, 0.25521, 0.091932, 0.401475, 0.115936, 0.424523, -0.052211, -0.101391, -0.132082, 0.678177], [0.219061, 0.219222, 0.351909, -0.126841, 0.098229, 0.200847, -0.103395, 0.288064, -0.092483, 0.297051, 0.140585, -0.160253, -0.112741, -0.301792, -0.152733, -0.027045], [-0.013269, -0.057563, 0.046583, 0.033437, -0.15017, 0.235841, 0.221814, -0.040013, 0.198485, 0.201846, 0.137033, 0.048061, -0.15325, 0.091519, 0.012713, 0.062032], [-0.030841, 0.043326, -0.091367, -0.210784, -0.227645, 0.03162, 0.355916, 0.025473, -0.049254, 0.238999, 0.118476, 0.034902, 0.034616, -0.724555, 0.253003, 0.057721], [0.218237, -0.537941, -0.056729, 0.030664, 0.073024, 0.082938, -0.308521, 0.196572, -0.448029, 0.288469, -0.013295, -0.621985, -0.134376, -0.841865, 0.067682, 0.316227], [-0.069601, 0.182058, -0.098603, -0.008835, -0.126364, -0.153715, 0.122679, -0.050911, -0.20615, -0.122768, -0.13996, -0.006608, -0.162946, -0.375224, -0.227126, -0.72095], [-0.199034, -0.311205, -0.188406, 0.015625, -0.157938, -0.067654, -0.14799, -0.064001, -0.506469, -0.129391, -0.057507, -0.290307, 0.052254, -0.312923, -0.12971, -0.031783], [-0.108599, -0.430997, -0.036019, 0.047741, 0.075573, -0.054612, 0.161773, 0.093555, 0.061704, 0.071224, 0.083151, -0.272564, 0.032641, 0.276581, 0.399791, 0.110323], [-0.103858, -0.01875, -0.28765, 0.068354, 0.236288, -0.557907, -0.016616, 0.110863, 0.3035, 0.122349, 0.151577, 0.030463, 0.105633, -0.953675, -0.089754, 0.215647], [-0.049331, -0.017039, 0.107893, 0.026788, -0.179851, 0.142361, -0.020642, -0.153094, -0.120289, -0.421441, -0.033311, 0.138253, -0.10228, 0.272307, 0.127256, 0.215484], [-0.115883, 0.1967, -0.011788, 0.059559, -0.043884, 0.046866, -0.016413, -0.014349, -0.281206, 0.269577, 0.111903, 0.094833, -0.139882, 0.105811, -0.223867, 0.110947], [0.08841, -0.235082, -0.10917, -0.043386, -0.094869, 0.030826, -0.17331, 0.191267, -0.124512, -0.509072, -0.013978, -0.472551, -0.046619, -0.069204, -0.005555, 0.025329], [0.124704, -0.274891, 0.37471, 0.09729, -0.122615, 0.270383, 0.033642, 0.284455, 0.122754, 0.26027, -0.05285, 0.467184, 0.000883, -0.012845, 0.078201, 0.197613], [0.045995, -0.182722, -0.075179, 0.080047, -0.149448, 0.356729, -0.047822, -0.13933, -0.136683, -0.440895, 0.184407, -0.226607, 0.102503, 2e-05, 0.183928, 0.453027], [0.063049, 0.065521, -0.018931, 0.159142, 0.21883, 0.08149, -0.062077, 0.118043, 0.24741, -0.465916, -0.192391, -0.701116, -0.173933, 0.281051, -0.265217, 0.095724], [0.054392, -0.060139, 0.198904, 0.056542, 0.13978, -0.496936, 0.145473, -0.114953, 0.055997, 0.284707, -0.072491, 0.116736, -0.066633, -0.882334, -0.044762, 0.218948], [0.033172, -0.045775, -0.086774, -0.099595, 0.127174, -0.282667, 0.238378, -0.249217, -0.030052, 0.257042, 0.207339, 0.285553, 0.003455, 0.111205, -0.186458, 0.154332], [0.015627, -0.014452, -0.008727, 0.116446, 0.151199, 0.093959, -0.562416, 0.064408, -0.216267, 0.179346, 0.20204, -0.436767, -0.151577, -0.092848, -0.038969, 0.138939], [0.164677, -0.052011, -0.489875, -0.17383, -0.16021, -0.111893, 0.065835, -0.083829, 0.2228, 0.349566, 0.107926, 0.18789, -0.165448, -0.323831, 0.182491, -0.45449], [0.162926, 0.321291, 0.051916, -0.143797, -0.04958, 0.206369, 0.049461, -0.00382, 0.279971, -0.077421, 0.134601, -0.116602, 0.018243, 0.258182, 0.092394, -0.143757], [0.041085, 0.209953, -0.254687, -0.138735, 0.033899, -0.183419, -0.400152, -0.068028, 0.030543, 0.50036, -0.167342, -0.200332, -0.214527, -0.304678, -0.068111, -0.136758], [-0.241392, 0.100559, -0.229525, 0.135294, -0.254659, -0.232794, 0.305228, -0.330371, -0.022364, -0.175392, 0.088188, -0.147032, 0.007939, -0.133429, 0.157195, -0.137957], [0.248971, -0.073778, -0.019129, -0.136735, 0.371954, 0.141088, 0.291193, 0.042992, -0.017694, -0.353813, -0.138234, -0.474829, -0.135164, -0.158875, -0.143001, 0.067201], [0.044584, -0.262756, -0.101939, 0.247755, 0.147373, 0.214269, 0.240178, 0.180164, 0.215382, 0.108315, 0.018966, 0.760469, 0.03118, 0.011611, -0.048123, 0.16609], [-0.251322, -0.323019, 0.147069, 0.167467, 0.242037, -0.018986, -0.349595, 0.220131, 0.281066, 0.232048, -0.064186, -0.335505, -0.250834, -0.127645, 0.16123, 0.031918], [0.145292, 0.272564, 0.149639, 0.128554, -0.251594, -0.006316, 0.041444, 0.189865, -0.353045, -0.238899, 0.007344, -0.032442, 0.051707, 0.291039, -0.156524, -0.1002], [0.292529, 0.111147, -0.150659, -0.000599, 0.181839, -0.134804, -0.042586, -0.07375, 0.110309, 0.01023, -0.012953, -0.627039, -0.16528, -0.095937, -0.205564, -0.085954]], [0.141248, 0.263534, 0.079964, 0.266926, -0.035797, -0.418868, 0.040177, 0.103939, 0.140668, -0.456693, -0.007948, 0.215908, -0.239994, -0.089222, 0.367511, -0.489804]), ([[0.589896], [1.24656], [0.795202], [0.84801], [1.374813], [1.569468], [-0.73401], [1.00288], [-1.523917], [-3.0239], [-0.123909], [-1.73551], [0.416893], [2.927867], [1.39967], [2.149822]], [-0.974535])])
R4_SEQ_TM = [2.748641, -0.000111, 0.000111, 0.0, 0.053308, 0.053308, 0.0, 0.0, 0.0, 1.25666, 5.846188, 5.846188, 0.0]
R4_SEQ_TSD = [1.078778, 0.135841, 0.135841, 0.102663, 1.444015, 1.444015, 0.669029, 0.222177, 0.183079, 0.616848, 1.678203, 1.678203, 0.651227]
R4_SEQ_XM = [0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 1.235948, -0.0, -0.0, 5.908182, 5.908182, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.009119]
R4_SEQ_XS = [1.133826, 0.999664, 0.952553, 0.84802, 0.756213, 2.570136, 20.914425, 0.789277, 18.437494, 0.704278, 0.649224, 0.542084, 1.664759, 1.664759, 0.999991, 4.40289, 1.165521, 0.303697, 133.786133, 335.341339, 1065.236694, 1.340655]
R4_SEQ_YSC = 1.9801905155181885
R4_SEQ_WI = [[-0.169195, -0.132419, -0.142336, -0.428097, -0.440597, -0.429641, -0.520144, -0.461962, -0.31828, 0.008659, -0.345212, -0.410421, -0.391295, -0.417314, 0.020247, -0.438753, -0.202643, -0.323748, 0.008674, 0.016355, -0.072377, 0.306202, -0.367941, 0.133205, -0.450873, -0.709349, 0.146085, -0.150736, 0.087064, -0.669954, 0.299197, 0.087421, 0.006438, 0.48471, 0.058269, 4.4e-05, 0.46423, 0.042561, -0.114798, -0.071156, -0.265512, -0.037033, 0.444319, -0.325478, 0.502876, 0.040446, 0.165029, 0.239209], [0.382948, 0.034564, -0.003513, -0.081056, 0.300186, 0.053905, -0.273815, 0.348162, -0.25403, -0.24527, -0.122419, 0.365307, 0.232344, -0.179303, 0.113157, 0.068269, -0.475344, -0.107903, 0.133006, -0.756081, 0.038669, -0.248246, 0.380676, 0.641334, 0.059424, -0.255391, -0.316562, -0.70393, 0.056989, 0.169868, 0.341469, -0.366813, 0.351478, -0.253492, -0.212106, 0.324496, 0.204707, 0.142744, -0.446539, 0.071269, -0.093701, -0.102588, 0.32999, 0.063131, 0.041138, 0.166995, 0.091635, 0.175031], [-0.224144, 0.28954, -0.11131, -0.028156, -0.04252, 0.035627, -0.061294, -0.115687, 0.123399, -0.067932, -0.325259, -0.414896, 0.404247, 0.022767, 0.141942, 0.072709, -0.303867, 0.414231, -0.019525, 0.022404, 0.215641, 0.115172, 0.157416, -0.172553, 0.082431, -0.132781, 0.167536, 1.275342, -0.287875, 0.575271, -0.182181, 0.317924, -0.098419, 0.339848, -0.101572, 0.000441, -0.215609, 0.34358, 0.256924, 0.530265, 0.081645, 0.399656, -0.004448, -0.056651, -0.021282, -0.402407, 0.163969, 0.34048], [0.060003, 0.244644, 0.074944, 0.228176, 0.057769, -0.041817, -0.244955, 0.223312, 0.114456, -0.066926, -0.548028, -0.440366, 0.468411, 0.101744, 0.27223, 0.074518, -0.383679, -0.227394, -0.107015, -0.356206, 0.292411, 0.254516, -0.016451, 0.337813, 0.468395, -0.205376, -0.37056, 0.260808, -0.479217, 0.397673, 0.147606, -0.145702, 0.013635, 0.197955, -0.054547, 0.122776, 0.150278, -0.055583, -0.040988, 0.445627, -0.207568, -0.072618, 0.243776, 0.219837, 0.030659, -0.125508, 0.202731, 0.030813], [-0.155813, 0.960929, -0.369576, -0.041903, -0.799994, 0.167058, 0.132406, -0.114651, -0.159425, -0.084227, 0.443455, 0.968866, -0.118919, -0.409901, -0.492758, 0.069174, -1.061688, 0.445951, -0.496106, -0.19455, -0.352564, 0.056197, -0.250158, 0.065062, -0.107345, -0.335186, 0.579736, 0.28474, 0.162722, -0.412392, 0.859212, -0.575938, 0.955911, -0.589082, -0.035276, 0.319375, 0.288336, 0.119448, -0.257125, -0.042845, -0.218063, 0.130806, 0.066301, -0.048481, 0.248738, -0.140214, -0.222413, 0.376879], [-0.304754, -0.048247, -0.409132, 0.672324, -0.931505, -0.007898, 1.09774, -1.048276, 0.103861, -0.33164, -0.16669, 0.225724, 0.01572, 0.321977, 0.583113, 0.615788, 0.134497, -0.452058, 0.221292, -0.015617, 0.652405, 0.293749, -0.232109, 0.221328, -0.287898, -0.763982, 0.07078, 0.145258, 0.034063, -0.554309, -0.536947, 0.282117, 0.006083, 0.334222, -0.074799, 0.405814, -0.11413, 0.004994, -0.896796, 0.699048, 0.165972, 0.166476, 0.21256, -0.101486, 0.378734, 0.259717, 0.339516, -0.612874], [0.013966, 0.231075, 0.111269, 0.115065, 0.556597, 0.433921, 0.056857, 0.024683, 0.048735, 0.181921, -0.379706, 0.07852, 0.140247, -0.216401, 0.300687, -0.386723, -0.246354, 0.24899, -0.166003, -0.159559, -0.201133, 0.083743, 0.117313, 0.244844, 0.377342, -0.042378, 0.343219, -0.471289, 0.017696, -0.101821, -0.033368, 0.380064, 0.313767, 0.175103, -0.395308, 0.078355, -0.17185, 0.229616, 0.599666, -0.136619, -0.467228, -0.159612, 0.235288, 0.119539, -0.352927, 0.060192, 0.245474, -0.175766], [-0.179453, -0.428533, 0.155056, 0.237302, 0.769893, 0.048732, -0.099947, -0.017921, -0.405917, 0.053057, -0.667472, 0.364036, -0.269715, -0.170391, 0.110335, -0.201087, -0.005273, 0.075472, 0.107464, -0.082394, -0.216122, 0.166896, -0.021653, -0.063986, 0.209059, 0.042005, -0.164305, -0.007643, -0.043499, 0.042984, 0.172087, 0.018809, -0.168021, 0.144637, -0.091735, -0.134928, -0.1407, -0.090954, 0.160557, -0.00268, -0.22998, 0.04165, 0.19991, -0.003381, 0.085075, -0.083175, 0.046109, -0.071123], [0.133214, -0.246498, 0.001704, 0.160336, 0.561576, 0.30253, 0.06112, 0.110011, -0.256601, -0.169039, -0.224365, 0.177672, 0.023158, 0.0738, 0.026511, -0.158541, -0.053016, -0.189065, -0.286981, 0.290491, 0.155501, -0.127304, 0.210851, -0.433577, 0.116099, 0.05029, -0.036518, -0.274002, 0.257533, 0.181963, 0.231965, 0.335628, -0.062093, -0.187755, 0.081351, 0.039409, 0.027536, 0.372707, 0.164378, -0.157401, 0.073408, 0.295268, -0.068117, 0.047623, 0.112639, 0.374752, 0.153275, -0.14148], [-0.477401, 0.41397, -0.13621, -0.129892, -0.442187, 0.135917, 0.221869, -0.271776, -0.190579, 0.103136, -0.131877, -0.502775, -0.015713, 0.299681, -0.417769, 0.259153, -0.153835, -0.195451, -0.116892, 0.27128, -0.299523, 0.297139, -0.370924, 0.157727, -0.156444, 0.259318, -0.025736, 0.792444, 0.390148, 0.156677, 0.587054, -0.071566, -0.143708, -0.332663, 0.276779, -0.183708, -0.024598, -0.269102, -0.14823, 0.292798, -0.45991, -0.715135, 0.114178, -0.267518, 0.084421, 0.336085, -0.080364, -0.371262], [0.102571, 0.07506, 0.361089, -0.252904, 0.34259, 0.071349, -0.346109, 0.311666, -0.233458, -0.241696, 0.179349, -0.426861, -0.117047, -0.54023, -0.077853, 0.252582, -0.390957, -0.200428, -0.151827, 0.060552, 0.155281, -0.270188, 0.368772, 0.244334, -0.083445, 0.09637, -0.56761, 0.007824, 0.018175, 0.023635, -0.429452, 0.358646, 0.065802, 0.032025, -0.196784, -0.117987, 0.117382, 0.089657, -0.039875, 0.283377, 0.053934, -0.177128, -0.041592, -0.082722, -0.141787, 0.036564, -0.259845, -0.115437], [-0.333491, -0.01208, -0.055375, 0.117226, 0.474171, -0.23881, -0.730656, -0.073678, -0.263656, -0.289869, -0.08159, -0.047448, -0.057548, -0.390445, 0.045811, 0.125523, -0.012989, 0.072068, -0.70301, 0.169926, -0.126921, 0.099659, -0.10895, -0.376565, -0.252272, -0.138872, 0.066089, -0.10252, 0.137839, 0.059152, 0.161771, -0.071306, 0.043703, -0.142491, -0.205838, -0.103094, -0.001244, -0.244574, 0.290882, 0.157096, 0.193574, -0.07581, -0.057158, -0.262073, 0.075192, 0.165605, 0.010512, 0.096854], [-0.011502, 0.00694, -0.022786, 0.089407, 0.005373, -0.076275, 0.222754, 0.283758, -0.042847, -0.488315, 0.31333, -0.408434, -0.056944, -0.051498, 0.005418, 0.019592, -0.0354, -0.260676, 0.538284, -0.170252, 0.063044, 0.055341, 0.173896, 0.285204, 0.126823, 0.073978, -0.41604, 0.059264, 0.082034, 0.006349, -0.255177, 0.324166, -0.009012, 0.120783, 0.111646, -0.006926, -0.198434, -0.1223, -0.076027, 0.187068, -0.054244, -0.013024, 0.161776, -0.019835, 0.037243, -0.018545, -0.168237, -0.081269]]
R4_SEQ_BI = [0.058372, 0.049873, 0.150153, 0.197917, 0.580026, 0.21393, -0.112724, -0.203039, -0.261806, 0.375519, -0.34043, 0.1299, 0.158485, 0.353304, 0.210162, 0.052733, 0.646503, 0.610235, 0.378962, 0.271637, 0.657345, -0.007259, 0.93576, 0.411659, 0.514101, 0.601052, 1.09133, 0.387587, 0.259448, 0.763073, 0.420511, 0.820559, 0.109955, 0.153978, 0.183238, -0.02982, -0.025899, -0.218617, 0.178718, -0.26668, -0.189621, -0.162662, -0.299809, -0.068163, -0.343113, 0.188106, 0.332609, 0.166712]
R4_SEQ_WH = [[-0.421166, -0.29339, -0.294689, -0.067572, -0.101207, -0.192788, 0.038165, 0.101917, -0.439946, -0.142406, -0.03666, 0.225194, -0.171208, -0.675972, -0.059385, -0.404992, 1.003411, 0.898858, -0.285002, 0.444605, 1.43665, -0.007348, 0.383252, -0.076349, -0.597169, -0.321462, 0.516974, -0.065892, -0.072681, -0.374294, 0.169441, -0.44882, 0.353187, -0.247723, 0.358562, 0.275364, -0.006782, 0.001474, -0.247201, -0.418601, -0.495491, 0.044351, 0.194429, -0.548648, -0.614909, -0.119802, 0.242401, 0.54832], [-0.5983, -0.236871, 0.002342, 0.186222, -0.306169, -0.242901, -0.022751, -0.256547, 0.231342, -0.164342, -0.171718, 0.126123, -0.190047, -0.09452, 0.398796, -0.394446, 0.144057, -0.322397, -0.330348, -0.03711, -0.044241, 0.293986, -0.032203, -0.491145, 0.207078, -0.202335, -0.343401, -0.360278, -0.036295, -0.681966, -0.426498, 0.071264, -0.085083, 0.615397, 0.311138, 0.130101, 0.946168, -0.122568, 0.181024, 0.36147, 0.004463, 0.658809, -0.050829, -0.120333, -0.09349, -0.456134, -0.044972, 0.056332], [-0.894787, -0.266543, -0.71857, 0.12757, -0.117641, -0.000171, 0.394062, -1.008661, -0.569152, 0.28902, -0.393102, -0.426411, -0.981969, 0.488975, -0.385389, 0.080932, 0.357054, -0.358642, -0.507844, 1.019376, 0.515648, -0.077718, 0.259334, -0.495053, 0.135478, 0.062623, 0.076356, 0.41178, 1.091867, 0.118534, -0.699176, 0.293781, 0.371889, -0.318989, 0.306166, 0.556706, 0.159965, -0.049119, -0.421589, 0.305686, -0.298815, -0.117272, -0.024696, 0.062385, 0.025565, 0.291001, 0.368935, -0.084956], [0.069316, 0.053006, 0.088397, 0.24352, -0.278779, 0.330925, 0.263367, -0.340691, -0.32494, 0.520949, -0.123081, -0.06856, -0.27838, 0.243655, -0.330782, 0.162454, 0.187142, 0.226553, 0.302066, 0.174315, 0.63936, -0.119614, 0.346339, -0.063366, 0.076659, 0.461277, 0.429392, 0.805504, 0.117014, -0.084127, 0.07507, 0.042295, -0.080756, -0.2184, 0.150502, -0.1392, 0.247979, -0.321692, -0.106574, 0.043133, 0.0729, -0.187516, 0.254934, -0.013795, 0.079434, -0.116636, 0.378742, -0.253609], [-0.086275, -0.271085, 0.190619, -0.408581, 0.075276, -0.396326, -0.058131, -0.140357, -0.216176, 0.015636, 0.3819, -0.550041, 0.158892, -0.336532, -0.164423, -0.031055, -0.102379, -0.42519, -0.550655, -0.038098, 0.584982, 0.020822, -0.382513, 0.289026, -0.17634, -0.574277, -0.479349, -0.427055, -0.527082, -0.550832, -0.707964, 0.241982, -0.165579, 0.49909, 0.069807, -0.186605, 0.585459, -0.490315, -0.170612, -0.059115, -0.444674, 0.416814, -0.280336, -0.341769, 0.278639, -0.328561, 0.156395, 0.401405], [0.168173, -0.393209, 0.141353, 0.014732, -0.126927, -0.565537, 0.205327, 0.23767, 0.146377, -0.722141, -0.005927, 0.048097, 0.250187, -0.295467, 0.192863, -0.468871, -0.021922, -0.583494, -0.603529, -0.578312, -0.397035, 0.225857, -0.529911, -0.14113, -0.693922, -0.248675, -0.567945, -0.19936, -0.881075, -0.297858, -0.273366, -0.635828, 0.182502, 0.187506, 0.189344, -0.084631, -0.242937, 0.247404, 0.471727, 0.150917, 0.018554, -0.058248, -0.029368, 0.015814, 0.035994, -0.494357, -0.146538, 0.316989], [0.149608, -0.110923, -0.087795, 0.080837, 0.187075, -0.508034, -0.600371, -0.240872, 0.047644, -0.174144, -0.199221, 0.298421, 0.003649, -0.284461, 0.030503, -0.200859, 0.651533, 0.14837, -0.079439, -0.028884, -0.237892, 0.374802, 0.344241, -0.074435, -0.511525, -0.476551, 0.289721, 0.361773, -0.987555, 0.220264, -0.414683, -0.411486, -0.226044, -0.446741, -0.062902, -0.250425, -0.302031, 0.53075, 0.081075, -0.150438, -0.03624, 0.20812, -0.017584, 0.641315, 0.122938, -0.009168, -0.076485, 0.218173], [0.050123, 0.076277, 0.126266, -0.232562, -0.440142, -0.105806, -0.023615, -0.14078, -0.017167, -0.197859, 0.441513, -0.047812, 0.134571, 0.02616, -0.328624, 0.174875, -0.61943, -0.475051, -0.752857, -0.325484, -1.521935, -0.117978, -0.849618, -0.090429, -0.305481, -0.10335, -0.725209, -0.238641, -0.413962, -0.77988, -0.270273, -0.707455, -0.168736, 0.393847, 0.035402, 0.178922, 0.304907, -0.01963, 0.178871, 0.266301, -0.000936, -0.160562, -0.271592, -0.116848, -0.054421, -0.42679, -0.208689, -0.182733], [0.182116, 0.036423, 0.048252, 0.520045, 0.685257, 0.099407, -0.021756, 0.194397, 0.342245, -0.242909, -0.081672, 0.678099, 0.488446, -0.24565, 0.192089, 0.345996, -0.385408, -0.339906, 0.369143, -0.081902, -0.061229, -0.405199, 0.281057, -0.016602, 0.119989, -0.557314, -0.132679, -1.105358, 0.363186, 0.403273, -0.465337, 0.345927, -0.485747, 0.243355, -0.137492, -0.024968, 0.024125, 0.127158, 0.046978, 0.199609, 0.425723, 0.478557, -0.209082, -0.226594, 0.311565, 0.1974, -0.354221, -0.07282], [0.188875, -0.128706, -0.283807, 0.226056, 0.249171, -0.601277, -0.226035, -0.020189, 0.456475, -0.045028, -0.203005, 0.269164, 0.273435, 0.117902, 0.63676, 0.154517, -0.205537, -0.127407, -0.167184, -0.05318, -0.310749, -0.295332, 0.059735, -0.629918, 0.184722, 0.35773, -0.427854, -0.216489, -0.079835, -0.05352, -0.278523, -0.130814, -0.245611, 0.332964, -0.012003, -0.088869, 0.118601, 0.106172, -0.095711, 0.204103, 0.288622, 0.699177, 0.089832, 0.404988, -0.265319, -0.491979, 0.006963, 0.122334], [-0.042481, 0.030096, -0.023717, -0.490548, -0.385607, 0.120708, 0.483876, -0.022401, -0.117097, 0.01968, 0.228403, 0.544981, 0.288317, 0.078098, -0.220796, -0.013278, -0.189575, 0.161028, -0.101916, -0.393344, 0.240036, 0.318961, -0.862202, 0.342379, -0.629845, -0.6349, -0.076398, -0.216471, -0.159995, -0.205347, -0.099249, -0.445052, 0.085282, -0.051052, -0.058904, -0.155858, 0.052194, 0.293116, 0.091954, 0.057976, 0.092214, -0.210084, 0.349199, -0.331359, 0.105471, -0.226578, -0.103276, -0.095818], [0.141228, 0.264655, 0.016004, 0.100252, -0.301969, 0.275895, 0.01019, 0.246586, 0.635663, -0.313344, -0.502179, 0.123016, 0.241384, 0.005021, 0.122914, -0.002468, -0.483551, 0.426586, -0.761813, 0.244967, -0.393896, 0.389682, 0.292289, -0.53303, 1.181957, 0.448787, 0.311755, 0.017585, 0.136862, 0.007165, 0.56178, -0.375342, -0.272726, 0.162599, 0.083526, -0.415366, -0.145752, 0.245184, 0.427385, 0.185296, -0.10219, -0.002933, -0.296558, 0.461417, 0.121002, -0.010513, -0.128536, -0.512883], [0.61121, -0.157814, 0.709128, 0.09103, -0.517315, -0.525426, -0.030662, -0.339352, 0.622871, 0.299772, 0.299553, -0.366347, 0.018865, 0.530852, -0.313632, -0.00551, -0.355135, -0.724703, 0.338525, -0.188784, -1.040733, -0.110041, -0.52107, -0.277258, -0.225179, 0.138837, -1.255736, -0.879806, -0.449491, -0.71779, -0.05917, -0.831853, -0.111802, -0.096792, -0.025792, 0.06578, -0.027813, -0.207508, 0.056521, -0.138337, 0.111351, 0.31433, 0.298138, -0.099034, 0.868571, 0.22113, -0.495679, -0.36451], [-0.123445, 0.210746, -0.159008, 0.233839, 0.203895, 0.829066, 0.157112, -0.197742, -0.250865, 0.720041, -0.054025, -0.320734, 0.003862, 0.678847, -0.26069, 0.371461, 0.217055, 0.711333, 0.51968, 0.430149, 0.324644, 0.043493, 1.033089, -0.095149, 0.438312, -0.223264, 0.505723, 0.316645, 0.356188, 0.385278, 0.549718, 0.960233, 0.157726, -0.346427, -0.264676, 0.194006, -0.053129, -0.658034, -0.479801, -0.339906, -0.129548, -0.02791, -0.225119, -0.086886, -0.009127, 0.288497, -0.072187, -0.453142], [-0.634016, -0.274617, -0.600246, 0.569016, -0.173618, 0.163972, 0.166427, 0.003026, -0.090336, 0.324422, -0.526296, 0.224288, -0.351709, 0.399137, 0.157362, -0.02057, 0.807025, 0.473299, -0.718074, 0.524704, 0.655396, 0.739409, 0.469371, -0.562492, -0.166811, 0.158421, 0.819014, 0.266567, 0.155347, 0.35227, -0.084593, 0.564948, 0.399329, -0.070407, 0.34312, 0.053805, 0.219511, -0.3965, -0.156934, -0.073732, -0.112178, 0.170579, 0.159867, 0.202753, -0.020633, -0.133941, -0.181912, 0.022853], [-0.696904, -0.129084, -0.117088, -0.261686, 0.128966, -0.466472, 0.026969, -0.249246, -0.195815, -0.493643, -0.301839, 0.501055, -0.024266, -0.723561, 0.041282, -0.285971, 0.572715, 0.642489, -0.136043, -0.208435, 0.825302, -0.262096, 0.122893, 0.224667, -0.430965, 0.176129, 0.529802, 0.262092, -0.391125, -0.233002, -0.362349, -0.189159, 0.199828, 0.363888, 0.49768, 0.045061, 0.482033, 0.358743, 0.065548, -0.292072, -0.187193, 0.097872, -0.076832, -0.096458, -0.458633, -0.693298, 0.252936, 0.450667]]
R4_SEQ_BH = [-0.197806, 0.296876, 0.017487, -0.004855, 0.363256, 0.496878, -0.092919, -0.094779, -0.035687, 0.394359, 0.014358, 0.131048, 0.231876, 0.229817, 0.362533, 0.036228, 0.815608, 0.529334, 0.61662, 0.518632, 0.746695, -0.12232, 0.484024, 0.162608, 0.357614, 0.607639, 0.931062, 0.552862, 0.279198, 0.383948, 0.358702, 0.699408, -0.166779, 0.009942, -0.22951, 0.054884, 0.041162, -0.416618, 0.172655, -0.083947, 0.20578, -0.121712, -0.064866, -0.1635, 0.002319, 0.128414, 0.189741, -0.077191]
R4_SEQ_HEAD = [
    ([[0.039455, -0.077297, -0.389902, -0.058263, -0.346051, 0.043926, -0.135992, -0.065134, 0.218063, 0.034783, 0.005605, -0.080655, 0.004053, 0.057323, -0.112435, 0.240665, 0.016597, 0.083047, 0.111305, -0.001439, 0.078346, 0.051299, -0.088766, 0.068191, -0.171547, -0.073403, -0.00411, 0.217176, 0.182027, -0.139162, -0.170753, -0.000787, -0.026707, 0.246962, -0.025874, 0.123361, -0.117441, 0.029101, 0.129608, 0.212581, -0.113425, 0.046902, 0.098224, 0.008162, 0.166522, -0.016092, -0.017295, 0.117633, 0.090812, 0.05164, 0.093294, -0.166782, 0.063103, -0.042944, 0.173608, -0.036734, 0.108081, 0.081923, 0.064142, -0.079124, -0.029491, 0.036334, 0.083739, -0.000331], [-0.146865, 0.06184, 0.015446, 0.141263, -0.239919, -0.01746, -0.137976, 0.066102, -0.07937, -0.185128, 0.139712, 0.113718, -0.148342, 0.128565, -0.044489, 0.105307, 0.170814, -0.020772, -0.086577, 0.10956, 0.000931, 0.061582, -0.059131, 0.090831, -0.028799, 0.03127, 0.175903, -0.121883, 0.180837, 0.040771, -0.039669, 0.0368, -0.189204, -0.137798, -0.073384, -0.054664, 0.012601, -0.12236, -0.018998, 0.090069, 0.006888, -0.107977, 0.136522, 0.042388, -0.109868, -0.05277, -0.009043, 0.005149, 0.05789, -0.088412, -0.029915, 0.090519, -0.084818, -0.060175, 0.220801, 0.005445, 0.025828, 0.044114, -0.042943, -0.054477, -0.1824, -0.125434, -0.069267, 0.145714], [0.16706, -0.066657, 0.033924, 0.023475, -0.007281, -0.018569, -0.039288, 0.010115, 0.07528, -0.010966, 0.139296, -0.002148, -0.117233, 0.128739, 0.008867, -0.133188, 0.195495, -0.088982, 0.053696, -0.083806, -0.14649, 0.213656, 0.197771, -0.105033, -0.145374, 0.033216, -0.108195, 0.089686, 0.090307, 0.08402, 0.058284, -0.149167, -0.093073, -0.074867, -0.042374, 0.075765, 0.122807, 0.0634, -0.07294, -0.006226, 0.001465, -0.003703, 0.170531, 0.027871, -0.02482, -0.006677, 0.108693, 0.072668, 0.006308, -0.001018, 0.102781, 0.130601, 0.129624, -0.123175, 0.10874, 0.09325, -0.115563, -0.138372, -0.004366, 0.084421, -0.044083, -0.162206, 0.106149, -0.079297], [0.244397, 0.175216, 0.327704, 0.103993, 0.205596, -0.041552, 0.019202, 0.198191, -0.090402, 0.099293, 0.056197, 0.19649, -0.006626, 0.150713, -0.002097, 0.060768, 0.003128, 0.031324, -0.124771, 0.124545, 0.046999, 0.028515, 0.041156, -0.046493, -0.057814, -0.017496, 0.039703, 0.128676, -0.152601, -0.029152, 0.213066, -0.033735, -0.227726, -0.320012, -0.115418, 0.023916, 0.194439, 0.079032, -0.131919, -0.171899, -0.141777, -0.217776, 0.142687, -0.071859, -0.042083, -0.175957, 0.19095, -0.21541, -0.062498, 0.105203, -0.154861, -0.099043, -0.01999, -0.1106, -0.094143, 0.112422, -0.066515, -0.206705, 0.214362, 0.282394, 0.121667, -0.164252, -0.186293, -0.169015], [-0.065649, -0.065798, 0.036776, 0.104692, 0.197191, 0.103881, -0.074155, -0.050257, -0.047373, -0.009065, 0.024338, 0.241011, -0.08521, 0.037584, -0.108973, -0.103812, 0.150212, -0.105625, -0.294961, -0.15968, -0.063941, -0.113583, -0.269611, 0.226759, 0.123211, 0.194598, -0.243613, -0.096898, -0.132865, -0.03078, 0.161004, -0.209628, 0.032802, 0.111761, -0.206607, 0.092147, 0.038135, 0.011002, -0.347791, -0.026817, -0.126467, 0.037519, 0.222761, 0.129794, -0.011677, 0.174945, 0.346983, -0.474377, 0.307146, 0.238474, 0.104689, -0.078135, -0.226148, -0.208426, -0.202615, -0.069708, -0.250617, -0.186192, 0.261465, 0.009155, -0.167536, -0.005146, -0.394735, 0.258568], [0.030952, -0.039952, 0.172633, 0.145749, 0.138413, -0.082161, 0.021291, 0.008549, -0.170599, 0.096623, -0.083496, 0.102295, 0.020709, 0.137415, -0.040774, 0.169366, 0.145078, 0.024353, 0.003223, 0.132916, 0.133294, -0.122839, 0.130995, -0.113626, -0.08721, 0.110287, -0.097148, -0.086992, -0.144853, -0.017535, 0.115958, 0.057295, -0.17426, -0.135968, 0.19796, -0.094696, 0.004797, -0.027277, -0.049022, -0.167945, 0.066289, 0.046371, 0.056947, -0.093929, -0.152572, 0.049207, 0.01766, 0.027109, -0.149075, -0.103162, -0.092193, 0.17405, 0.097155, 0.09571, 0.203098, -0.105237, 0.051232, -0.125272, -0.051403, -0.041916, -0.062596, -0.157808, 0.076757, 0.026826], [-0.034626, 0.066697, 0.034723, 0.093977, -0.324464, 0.252674, -0.003738, 0.238958, 0.171013, 0.162047, -0.020378, 0.090926, 0.103477, 0.04756, -0.098188, 0.09772, 0.098634, 0.199708, 0.076267, 0.096542, -0.105763, 0.128498, -0.097225, 0.060704, -0.012935, 0.091484, 0.039375, 0.05726, 0.004677, -0.069768, 0.136232, -0.177736, -0.037925, -0.175137, 0.066351, 0.106509, -0.120794, -0.036609, -0.021744, -0.070639, -0.012068, 0.112989, 0.050733, 0.014174, -0.095087, -0.047845, 0.202159, -0.093817, -0.026379, -0.008508, -0.053137, 0.084953, 0.069851, -0.095318, 0.122482, -0.004932, -0.209498, -0.010166, -0.055163, 0.000269, 0.044688, 0.098317, -0.223876, 0.059033], [0.160232, -0.154931, -0.064949, -0.112697, 0.062966, 0.045726, -0.04384, 0.097996, -0.122944, -0.112544, -0.13265, -0.106311, 0.019047, 0.144402, -0.069014, -0.015861, 0.031091, -0.027275, 0.336089, -0.084774, -0.219669, 0.084752, 0.188122, 0.023522, 0.189353, -0.030564, 0.145012, -0.020384, 0.084264, -0.041992, -0.020699, -0.099336, -0.011514, 0.050555, 0.044361, -0.0308, -0.102508, 0.085158, 0.001012, -0.085355, -0.000465, 0.045098, -0.13185, 0.291591, 0.015441, 0.017925, -0.074214, 0.006246, 0.135123, 0.199146, -0.271878, 0.175056, -0.157613, 0.166569, -0.164067, -0.125926, 0.086275, -0.152008, -0.196697, 0.121099, 0.034491, -0.058102, -0.124053, 0.254169], [-0.019492, -0.174363, -0.141992, 0.034947, -0.17037, 0.218847, -0.165491, -0.001477, -0.136613, 0.063168, 0.186426, 0.08568, -0.195201, 0.049924, -0.283516, 0.160501, -0.010902, 0.280535, -0.069797, -0.107216, 0.002816, 0.302281, 0.127798, 0.153342, -0.088784, -0.007235, 0.034536, 0.143446, -0.261624, 0.121177, 0.107334, -0.079028, -0.232003, 0.075886, 0.059298, 0.133274, 0.105264, -0.139891, -0.008773, -0.23307, -0.151756, -0.242718, -0.03163, 0.074596, 0.096234, 0.000893, 0.103252, -0.232781, -0.018545, 0.209429, 0.09201, -0.050982, 0.012085, -0.11605, 0.221359, -0.246602, -0.260319, -0.197837, -0.050888, 0.168152, -0.232505, 0.021438, 0.094, 0.286917], [0.006779, 0.141626, -0.104571, 0.107294, -0.137326, -0.084205, -0.117416, -0.074584, -0.257672, -0.124062, 0.026274, 0.090117, 0.053159, -0.114259, -0.005109, -0.049172, -0.220336, -0.158988, -0.22697, 0.116945, -0.13514, -0.091785, -0.019612, -0.024315, -0.09416, 0.193524, -0.258636, -0.135557, -0.238362, 0.053477, -0.238531, 0.085274, 0.154575, 0.037219, 0.013104, 0.003481, -0.17009, -0.072005, 0.065169, -0.01454, -0.359002, 0.047641, -0.106486, 0.072985, 0.029691, 0.14083, 0.156054, 0.091627, 0.007766, 0.094537, 0.121829, 0.031285, -0.039733, 0.06943, -0.097133, -0.189226, -0.088991, -0.258648, 0.060628, 0.08286, 0.035865, 0.064957, -0.024119, 0.125352], [-0.033841, 0.075351, 0.180998, 0.094969, -0.099203, 0.11877, 0.000498, -0.105179, 0.15962, -0.109227, 0.026426, -0.113633, -0.044341, 0.13167, -0.047372, -0.039556, 0.092222, 0.193668, -0.007168, 0.089129, 0.007737, 0.134586, 0.0153, -0.05462, 0.023045, -0.05406, 0.102311, 0.007599, -0.151125, -0.104279, -0.038344, 0.004365, -0.04692, -0.09426, -0.070431, 0.030226, -0.038845, 0.038084, 0.168205, 0.123167, 0.055316, 0.104099, 0.02177, -0.089585, -0.081486, 0.002945, -0.098399, -0.029009, -0.180719, -0.251062, 0.008515, 0.003704, -0.008816, -0.032758, -0.055312, 0.014363, -0.145988, 0.074807, 0.00991, 0.066605, 0.050146, -0.050814, 0.103906, -0.10629], [-0.132004, -0.143644, 0.033762, 0.084493, -0.074695, -0.046429, -0.216565, -0.107285, 0.188544, -0.183764, 0.103767, 0.031037, -0.084276, 0.034791, -0.035952, 0.36947, 0.312004, 0.212616, -0.113819, 0.089535, -0.134019, 0.169435, 0.105672, 0.271991, -0.219656, -0.063227, 0.287273, 0.311658, 0.153026, -0.054602, -0.348721, -0.215594, -0.210508, -0.010967, 0.076524, 0.099471, 0.131541, 0.153243, 0.342716, 0.082279, -0.254981, -0.045655, 0.219109, 0.162786, -0.263316, 0.083555, -0.146192, 0.048464, -0.146323, -0.001704, 0.188902, 0.110097, -0.058493, -0.229286, -0.154158, -0.339288, -0.1704, -0.211245, 0.05334, 0.261928, 0.136461, -0.048471, -0.109873, 0.244294], [0.131701, 0.099636, -0.095832, -0.05602, -0.066417, -0.005757, -0.013367, -0.057361, -0.091478, 0.005815, 0.111675, -0.157403, -0.140732, -0.070887, -0.121412, 0.110522, 0.205304, 0.127653, -0.027999, 0.062595, -0.107732, -0.129301, -0.089887, 0.129344, 0.060938, 0.135816, 0.168692, 0.140675, -0.144665, 0.025124, 0.04596, -0.061202, 0.029717, 0.028436, 0.240428, 0.120286, -0.041315, 0.079638, 0.0349, -0.115078, -0.103822, 0.433384, -0.063737, 0.150718, -0.143991, 0.082183, 0.076504, -0.048218, 0.033333, -0.188454, 0.172076, 0.103043, 0.197062, -0.186422, 0.096936, -0.131555, 0.031424, -0.001079, 0.101388, 0.041614, 0.148509, -0.135194, 0.043573, -0.026756], [-0.013576, -0.079509, 0.004145, 0.083624, -0.024948, -0.210678, -0.082216, 0.193878, -0.29437, 0.121666, -0.084897, -0.109458, -0.154933, -0.035379, -0.060532, 0.171717, 0.161819, -0.038082, 0.001883, -0.043743, 0.143002, 0.065327, -0.137167, 0.073908, 0.082701, 0.147143, 0.256551, 0.030813, 0.129983, -0.057807, -0.136282, -0.183592, 0.123986, 0.014082, 0.004776, -0.075802, -0.059135, 0.080134, -0.072215, 0.050962, 0.033642, 0.170368, -0.121161, -0.107511, -0.007473, -0.073421, 0.118361, -0.045051, 0.199143, 0.044818, 0.090117, -0.022348, 0.160696, 0.062499, -0.062235, -0.144401, 0.076129, 0.035212, 0.005527, -0.068486, 0.087166, -0.151758, 0.062383, 0.007132], [0.130157, 0.240246, -0.002785, 0.035215, 0.272556, 0.180914, -0.070314, -0.013208, 0.156062, 0.41166, 0.129158, 0.045467, 0.13773, 0.045711, 0.075673, -0.198216, 0.016171, 0.049875, 0.02824, 0.118091, -0.076032, -0.089599, -0.139785, -0.216697, 0.072213, 0.001909, -0.040607, -0.035794, 0.059717, 0.048461, -0.148706, 0.024283, -0.102693, -0.13539, 0.042233, -0.209565, -0.113727, 0.178364, 0.08156, 0.130853, 0.107558, 0.125165, 0.085517, -0.141774, -0.145262, -0.202754, -0.215634, 0.109984, -0.14458, -0.119875, -0.098036, 0.190268, 0.229418, 0.051913, -0.003777, -0.123729, 0.171988, 0.095319, 0.086297, 0.004779, -0.07454, 0.19551, 0.314986, 0.026579], [-0.166161, -0.063157, -0.008116, 0.033573, -0.075814, 0.015715, -0.04961, 0.048643, 0.034219, 0.239948, 0.012712, 0.196922, -0.015419, -0.131112, -0.050972, 0.046262, 0.210829, -0.040271, -0.001755, 0.061828, -0.004479, 0.129027, 0.095712, 0.033552, -0.050351, 0.163513, -0.063281, 0.09226, 0.159365, -0.079163, 0.282121, -0.1909, 0.255041, 0.168789, 0.051833, 0.075503, -0.196205, 0.155199, 0.025452, 0.124764, 0.046551, 0.076107, 0.009899, -0.02492, -0.104254, 0.089205, -0.078559, 0.075512, -0.03014, -0.080523, -0.171269, 0.133193, -0.072665, 0.001525, 0.066308, 0.026854, 0.110773, 0.10989, 0.220464, 0.080726, -0.034506, 0.059187, 0.349811, 0.142922], [0.142308, -0.164892, -0.189434, 0.350022, -0.066451, -0.288141, 0.119039, -0.01418, 0.147811, 0.228535, 0.132155, -0.035781, -0.225667, 0.207453, 0.090814, 0.198425, -0.146497, 0.170334, 0.106874, -0.004271, 0.522851, -0.24454, 0.101138, 0.108436, -0.241887, 0.097429, 0.394783, 0.09923, 0.173917, -0.161387, -0.016766, 0.088265, 0.072447, 0.214252, -0.261711, 0.181422, -0.18785, 0.296662, -0.008689, -0.598344, -0.176806, 0.140209, -0.1278, -0.073081, -0.05829, 0.071584, 0.026107, -0.099943, 0.256579, 0.062328, 0.219023, -0.621753, -0.322337, -0.040875, -0.199927, -0.150285, 0.129171, -0.229787, -0.003251, 0.40099, 0.099876, 0.23306, 0.014075, -0.056557], [-0.063169, 0.068477, 0.205265, 0.258999, -0.108521, 0.122363, 0.247543, 0.031414, -0.019246, 0.038034, 0.03485, -0.226569, 0.02143, -0.017702, -0.206348, 0.04024, 0.032154, 0.040759, 0.126569, 0.026609, 0.069729, 0.037836, 0.063469, 0.16353, -0.006434, 0.01756, 0.057691, 0.210212, 0.053476, 0.173026, -0.075114, 0.12417, -0.159755, -0.171689, -0.055577, 0.302788, -0.083701, -0.014344, 0.107272, -0.022382, 0.117092, -0.100973, -0.224846, 0.032339, 0.253519, -0.131808, 0.239511, 0.10936, -0.165455, -0.041161, -0.128968, 0.085024, -0.024788, 0.146166, -0.177222, -0.209423, -0.300585, -0.03084, 0.069388, 0.186611, -0.172594, -0.099409, 0.005293, -0.120982], [-0.179012, -0.102887, -0.008934, 0.03925, 0.213039, -0.076837, 0.075601, 0.121975, 0.011057, 0.033153, 0.050384, -0.23943, -0.125722, -0.006826, 0.081711, -0.085887, -0.107705, -0.163803, -0.042, -0.050072, -0.077553, -0.204456, -0.110224, 0.011428, -0.145225, 0.027634, 0.013749, 0.050061, 0.001891, -0.129389, 0.068346, -0.008963, 0.189078, 0.023363, -0.12784, -0.12512, -0.035624, 0.013623, 0.009082, -0.058144, 0.00953, 0.170985, -0.023074, -0.014345, -0.007572, 0.228753, -0.014999, -0.063941, -0.083378, 0.067164, -0.13209, 0.170256, -0.010974, -0.115566, -0.036523, 0.079274, 0.108603, 0.063364, 0.033968, -0.043461, 0.05952, -0.000639, -0.025765, 0.133786], [-0.242506, 0.231558, -0.117172, 0.100477, -0.317246, -0.083465, 0.0756, -0.107663, 0.07815, 0.137166, 0.229789, -0.104436, -0.229512, 0.096775, 0.143598, -0.146231, -0.172862, 0.21124, -0.000849, 0.082547, -0.2128, 0.086991, 0.04889, 0.200245, -0.041904, 0.201193, -0.139776, 0.377218, 0.137686, -0.006635, 0.068988, -0.12371, -0.016746, 0.139198, 0.017142, 0.066995, -0.149889, 0.169493, -0.073065, 0.254153, -0.089445, -0.19033, -0.107905, 0.18762, -0.077346, -0.211679, 0.001551, -0.072937, -0.298573, 0.309111, -0.106997, 0.363089, -0.028982, -0.09002, -0.249093, 0.109222, 0.196891, -0.108514, -0.269486, 0.140978, -0.083203, -0.165591, -0.039435, 0.157706], [-0.047867, -0.28455, 0.142216, 0.179472, 0.315864, -0.229522, -0.015906, -0.304223, 0.008184, 0.011399, 0.02588, 0.217909, 0.167947, -0.110943, -0.01272, 0.228416, -0.124887, -0.151339, 0.21965, 0.253123, -0.052456, 0.010453, 0.157414, -0.045336, 0.034752, -0.091803, -0.071354, -0.165574, -0.480682, -0.066212, -0.284447, 0.080866, -0.068429, -0.173161, -0.018526, -0.039881, -0.043314, 0.07903, 0.082701, 0.258155, 0.03902, -0.167844, -0.004671, -0.123934, -0.082931, -0.102821, -0.135841, 0.087291, 0.272117, 0.167698, 0.02834, -0.206438, 0.173064, -0.090577, -0.023856, 0.09786, 0.204994, 0.160844, -0.195609, -0.142718, -0.081604, -0.030304, 0.263911, 0.061206], [-0.20289, -0.042383, 0.049637, -0.025488, 0.112736, 0.368976, 0.375026, 0.092316, -0.075137, 0.05487, 0.14417, 0.077761, -0.116169, 0.072565, 0.332056, 0.17624, 0.160435, 0.267291, -0.029085, 0.133581, -0.072902, 0.028657, 0.123479, -0.146696, 0.121339, -0.155403, -0.18724, 0.09939, 0.096586, -0.170487, 0.027698, 0.060101, -0.070934, -0.064663, 0.282663, 0.44792, 0.003754, -0.061844, 0.121369, -0.074661, 0.115739, 0.082171, 0.020324, -0.069421, 0.138502, -0.115732, 0.309725, 0.040596, 0.107369, -0.041246, 0.256187, -0.017165, -0.22223, 0.136355, 0.160862, 0.079513, 0.183133, 0.049769, -0.052308, 0.094392, -0.038808, 0.1818, -0.028521, 0.032166], [-0.148, 0.163202, -0.271747, -0.150086, -0.358886, -0.037179, -0.122657, 0.005805, -0.099429, -0.179192, -0.321976, 0.061575, 0.239933, -0.261472, -0.329256, -0.058747, -0.047893, -0.065038, 0.172197, 0.199621, 0.334229, -0.113405, -0.181419, -0.25728, -0.223466, 0.141777, 0.220698, 0.086848, -0.053392, -0.462067, -0.116549, -0.111064, -0.325909, 0.086997, 0.210228, -0.128931, -0.433039, -0.08998, 0.243405, 0.101774, -0.324643, 0.095624, 0.098228, 0.174987, -0.274821, 0.061792, 0.231877, -0.066336, 0.117013, 0.009744, 0.157433, -0.073341, -0.0761, -0.170823, -0.572944, -0.175116, 0.212266, -0.393547, -0.128502, 0.256737, -0.186972, 0.066277, 0.151625, 0.127886], [0.05115, 0.207658, 0.198999, -0.197963, 0.003394, -0.142271, -0.10863, -0.178804, -0.023314, 0.057373, 0.000464, -0.264019, -0.130463, -0.043041, -0.117022, 0.007488, 0.015622, -0.141889, -0.113761, -0.103629, -0.194767, 0.12455, -0.065136, -0.258745, 0.13121, -0.06091, 0.067426, -0.361208, -0.330618, 0.226214, 0.029699, 0.00748, -0.081876, -0.167097, -0.029086, -0.132033, 0.068326, -0.156137, 0.111302, -0.131409, 0.135285, -0.055922, -0.009145, -0.107085, 0.11102, 0.045527, 0.195879, -0.154571, 0.011341, -0.264776, -0.004143, 0.210874, 0.230091, -0.111541, 0.000218, -0.01467, 0.299173, 0.068445, -0.028451, -0.209231, 0.142174, -0.165743, 0.025656, -0.174338], [-0.230883, -0.528042, -0.163107, -0.035094, 0.033902, 0.112214, -0.161886, 0.091094, 0.099966, 0.112919, 0.075193, 0.023698, -0.307776, -0.050257, 0.189019, -0.090976, -0.087123, 0.140945, -0.414558, -0.337549, -0.055302, 0.133656, -0.00104, -0.010229, -0.11605, 0.212606, -0.134303, -0.356037, -0.169153, 0.322843, -0.530397, 0.018234, -0.098058, -0.07916, 0.132577, 0.094327, -0.161557, -0.121869, 0.049159, -0.152348, -0.122819, -0.085025, 0.145143, -0.221982, 0.129675, -0.24155, 0.140984, -0.098173, -0.221636, -0.075984, -0.027026, -0.256322, 0.123436, -0.005101, -0.392971, 0.095846, 0.10013, -0.099178, -0.231334, -0.294593, -0.020838, -0.273169, 0.171665, 0.11971], [-0.012216, -0.018163, -0.117758, -0.113669, -0.024645, 0.141153, -0.008509, -0.130815, -0.042168, -0.219988, -0.209035, 0.148435, -0.083618, -0.302199, -0.257231, -0.349037, -0.249667, -0.127255, -0.055537, 0.069301, 0.17727, -0.171985, 0.194137, -0.048289, -0.180772, -0.15668, -0.129583, -0.214371, -0.123437, 0.052281, -0.331497, 0.063265, -0.171746, -0.169153, 0.068305, -0.153508, -0.088895, -0.175215, 0.088768, -0.069255, -0.169224, -0.158026, -0.015268, 0.228099, 0.072896, -0.086286, 0.117646, 0.186925, -0.125314, 0.156392, -0.100475, 0.046771, 0.044077, -0.068803, -0.145288, -0.162517, -0.205495, -0.234638, -0.176501, 0.158552, -0.097594, -0.270064, 0.146787, -0.252245], [-0.131715, -0.112179, 0.173164, -0.305698, -0.059994, 0.00194, -0.126641, 0.124813, -0.238716, -0.408051, -0.535132, -0.030352, -0.078974, -0.131733, 0.050585, -0.011021, -0.217194, -0.357363, -0.19306, -0.268387, -0.020573, -0.032876, -0.002002, -0.346238, 0.195496, -0.115015, 0.202564, -0.475899, -0.298361, -0.017692, 0.22413, 0.061015, 0.091581, -0.511996, 0.142717, 0.142071, -0.191101, -0.245368, -0.199059, -0.023283, 0.032114, -0.061182, -0.039598, 0.183533, 0.151829, 0.20885, 0.311883, -0.022184, 0.182969, -0.31586, 0.047279, 0.349521, 0.189654, -0.103096, -0.084283, -0.166778, 0.259369, 0.104541, -0.118525, -0.066596, -0.167489, -0.084243, -0.138698, 0.112522], [-0.094066, 0.045912, 0.044343, -0.110118, 0.076061, 0.193412, 0.234972, 0.032006, -0.003739, 0.224536, 0.208463, 0.046749, 0.010732, 0.211346, 0.1952, 0.177355, 0.101991, 0.017385, 0.25151, 0.033391, 0.06025, 0.165789, -0.013316, 0.102717, 0.287754, 0.070838, 0.078289, 0.32062, -0.040292, -0.169254, -0.024552, 0.028783, 0.178898, 0.315277, 0.117905, 0.167529, 0.143404, 0.148607, -0.185592, 0.326348, 0.324581, 0.16917, 0.05279, -0.08761, -0.037881, -0.046386, 0.08775, -0.02925, -0.044933, 0.120207, -0.039237, -0.083144, -0.115316, -0.191227, 0.117006, 0.128106, 0.230254, 0.162783, 0.132568, 0.327748, -0.019985, 0.315939, 0.125631, 0.110587], [0.031411, -0.067826, -0.114317, 0.210109, -0.0865, -0.006951, 0.188264, -0.238582, -0.093167, 0.249952, 0.03974, -0.041095, 0.08889, 0.208277, 0.34895, 0.094586, 0.081999, -0.022714, 0.114464, 0.033528, -0.197281, 0.3321, -0.062491, -0.018638, 0.078578, -0.101416, 0.00764, 0.143117, 0.220707, -0.046484, -0.100201, 0.144475, 0.116476, 0.259943, 0.119846, -0.03516, 0.005794, 0.365639, -0.010876, -0.16542, 0.303591, -0.057617, 0.097908, 0.11738, 0.183983, 0.051261, 0.159559, 0.079064, 0.070511, 0.16372, -0.009045, 0.007428, 0.109421, -0.04519, 0.13006, 0.098131, 0.117504, 0.111398, 0.303156, 0.165626, 0.022456, 0.031531, -0.060164, 0.005837], [0.030877, 0.25845, -0.060668, 0.00458, 0.128765, -0.076909, -0.031967, 0.074556, -0.1956, 0.095781, 0.071132, 0.039058, -0.358188, 0.062303, 0.073972, 0.181337, 0.221543, -0.079211, 0.135009, 0.098087, 0.088456, 0.149088, 0.052757, 0.012002, 0.150323, -0.001881, 0.068812, 0.084044, -0.03651, -0.163004, -0.143227, -0.060549, 0.030867, -0.269261, -0.072358, 0.243123, -0.090598, 0.207828, 0.04503, 0.009105, 0.032648, 0.060016, -0.02664, 0.017266, -0.149191, 0.087709, 0.033709, 0.152841, 0.084181, -0.032693, 0.067058, 0.035006, 0.107837, 0.01476, -0.184173, 0.085944, 0.014466, 0.040772, 0.261193, 0.225698, -0.149628, -0.081945, 0.047517, -0.026668], [0.099879, 0.206881, 0.100151, -0.063814, -0.027128, -0.285016, 0.258717, -0.077957, -0.086844, -0.155521, -0.032253, 0.048791, 0.00147, 0.109568, -0.007297, 0.142078, 0.145611, 0.204016, -0.033619, -0.034401, -0.127677, -0.156913, 0.300861, -0.061039, 0.0241, -0.004761, 0.059405, -0.011383, -0.00649, 0.059013, 0.226444, -0.083823, -0.207637, 0.030686, -0.053991, -0.02213, 0.118535, 0.002417, -0.100673, -0.137568, 0.200391, -0.043663, 0.020657, 0.213937, 0.049366, -0.057548, 0.105658, 0.068397, 0.170379, -0.137608, 0.311845, 0.021273, -0.149291, 0.134422, 0.102527, 0.047998, -0.083975, 0.179526, 0.139847, 0.15846, -0.069425, -0.108623, -0.132632, -0.294344], [0.075458, 0.214609, 0.007634, 0.055708, -0.168933, 0.010839, 0.081113, -0.260464, 0.126272, 0.17522, 0.204872, -0.331297, -0.049872, 0.285114, 0.196564, 0.159692, -0.231912, 0.073175, 0.089039, 0.011247, -0.192332, 0.143972, -0.176034, 0.089274, -0.012206, -0.131309, -0.10156, -0.158376, -0.084852, -0.046984, 0.105911, -0.100037, -0.006105, -0.151557, 0.097509, 0.180761, 0.019837, -0.092088, -0.234107, -0.289565, 0.049776, 0.223337, -0.11059, -0.061966, 0.102915, -0.220673, 0.143279, -0.092534, 0.207387, -0.26904, 0.094378, 0.048482, -0.11009, 0.156206, 0.249153, 0.200907, 0.267139, 0.112914, 0.230461, -0.06151, 0.323719, 0.093755, -0.125018, -0.215998], [-0.250057, 0.373229, -0.202802, 0.155263, -0.085343, 0.337287, 0.22502, -0.069733, -0.08449, -0.223597, -0.299742, 0.253977, -0.085688, -0.291766, -0.148448, -0.173538, 0.029349, -0.180013, 0.023178, 0.118096, 0.077539, 0.166001, 0.01037, -0.127973, 0.030461, -0.08489, 0.122052, 0.34047, -0.193306, -0.037976, 0.202724, -0.159162, -0.12082, 0.408819, -0.015086, -0.050339, -0.34029, 0.061629, 0.088208, 0.356007, -0.014401, -0.046317, 0.291321, 0.044431, 0.056969, 0.330267, -0.183653, -0.143452, 0.060112, 0.091417, 0.225506, 0.177184, -0.470985, -0.066546, -0.179675, -0.104697, -0.084303, -0.28117, -0.10336, 0.449278, -0.15806, 0.419706, 0.110017, 0.191951], [0.290129, 0.197031, 0.009058, 0.358793, -0.058555, 0.132255, 0.074835, -0.168742, 0.355594, 0.142082, 0.360625, -0.301825, -0.15333, 0.210039, 0.20044, -0.056823, 0.212603, 0.312608, -0.120854, -0.104602, -0.127211, 0.362408, -0.200925, 0.424831, 0.181731, 0.121948, -0.008381, -0.231417, 0.196535, -0.122088, -0.23506, 0.170756, -0.44371, -0.176854, 0.139799, 0.095755, 0.00601, 0.444048, 0.042417, -0.092821, 0.164891, -0.099242, 0.008026, -0.270306, -0.071222, -0.269541, 0.150276, -0.089423, -0.231759, -0.068845, -0.175214, 0.067965, 0.307993, 0.015995, 0.312091, 0.209882, -0.114777, 0.115964, 0.18072, -0.095984, 0.174405, -0.113866, -0.058479, 0.046451], [0.028765, 0.608078, 0.413624, -0.217517, 0.205628, 0.088245, 0.474858, -0.010158, -0.060632, -0.074743, -0.240792, 0.337277, 0.019412, 0.037216, 0.25518, 0.084998, 0.128736, -0.052078, -0.002907, 0.559879, -0.117811, 0.122116, 0.339175, -0.249188, 0.283312, -0.221266, -0.179439, 0.248639, 0.015372, 0.022167, 0.154858, 0.184557, 0.074174, -0.030602, -0.010037, 0.35125, -0.088842, -0.194083, 0.071325, 0.423579, 0.382104, 0.002447, 0.082987, 0.272131, 0.128463, -0.067591, -0.033316, 0.137427, 0.038145, 0.050215, -0.041241, 0.30869, -0.310578, 0.03636, 0.583942, -0.006657, -0.159893, 0.307304, 0.134363, 0.617471, -0.206379, 0.233454, 0.003011, -0.073925], [0.209214, -0.076684, 0.050143, -0.227005, -0.103687, -0.076037, -0.183667, -0.026541, 0.127046, 0.063309, -0.141647, 0.044858, -0.01615, -0.316611, -0.006297, 0.00929, 0.010331, -0.026235, -0.122459, 0.056431, -0.055543, 0.056582, 0.038246, 0.035346, -0.25834, -0.14179, -0.133403, 0.109104, 0.248476, -0.186213, 0.197687, 0.087021, -0.198695, -0.111625, -0.018878, -0.016742, 0.050162, -0.240326, 0.237606, -0.051088, 0.083946, -0.101178, -0.269116, 0.095708, -0.06407, -0.172349, 0.024532, 0.070814, 0.025847, 0.213735, -0.091525, -0.137145, 0.033662, 0.086611, 0.117288, 0.08721, -0.306669, -0.073391, 0.034455, 0.022525, 0.059848, -0.292724, -0.104019, -0.080237], [0.101667, -0.245683, -0.038072, 0.089025, 0.081504, -0.057456, -0.295095, -0.07127, 0.050898, 0.185353, 0.004928, -0.106206, -0.070163, -0.01962, -0.245567, -0.048334, -0.336197, 0.034063, -0.280695, -0.153023, -0.127349, -0.103042, 0.187023, 0.017101, -0.101676, 0.116079, -0.075958, -0.346296, -0.169706, 0.013495, -0.018207, -0.018298, -0.350022, -0.387888, 0.029063, -0.142827, -0.116054, -0.020709, 0.04233, 0.167114, -0.141003, -0.104423, -0.098826, 0.026301, -0.054835, -0.139444, 0.068443, -0.087934, 0.082976, 0.055621, -0.139372, -0.054681, -0.128839, -0.066149, -0.281694, -0.061275, -0.076176, -0.208106, -0.180017, -0.092936, 0.045041, -0.22796, -0.10033, -0.114228], [-0.237492, -0.321768, -0.38718, 0.172786, -0.255394, -0.067664, -0.032957, 0.040028, 0.167617, 0.163276, 0.132402, 0.001086, -0.160166, 0.316392, -0.096534, -0.01277, 0.085484, 0.032581, 0.004015, -0.163081, 0.396563, 0.182915, -0.388461, 0.154847, -0.073754, 0.025274, 0.430558, -0.261152, -0.119518, -0.074022, -0.373216, -0.260348, -0.055726, -0.064388, 0.158626, -0.037174, -0.163446, 0.055582, -0.009253, -0.161312, -0.018216, -0.102625, 0.109113, -0.202242, -0.081151, -0.004506, 0.171617, -0.136024, -0.011146, 0.084629, 0.318609, -0.128182, 0.059526, -0.099601, -0.458442, -0.122497, 0.195632, -0.361253, -0.371946, 0.002935, -0.149716, 0.185956, 0.063045, 0.073909]], [0.151865, -0.144926, 0.005675, 0.202679, 0.022947, -0.124043, -0.295215, -0.080347, 0.170991, -0.008591, -0.149078, -0.069797, -0.04378, -0.19757, -0.136062, -0.320518, -0.066681, -0.080031, -0.244603, -0.141803, -0.217911, -0.071843, 0.1071, 0.021015, -0.204259, 0.114162, -0.185276, -0.233757, -0.098459, 0.139848, -0.181041, 0.0834, -0.074084, -0.048517, 0.146199, -0.161373, -0.088323, 0.063725, -0.021617, -0.296346, -0.224178, 0.083745, -0.019519, -0.151343, -0.107672, -0.20723, 0.139633, 0.001713, -0.20333, -0.06043, -0.039927, -0.110847, 0.205663, 0.183601, -0.07696, -0.168211, 0.035238, -0.091375, -0.045953, -0.30526, 0.127442, -0.298424, 0.004741, -0.03145]),
    ([[0.117739, -0.050706, -0.055344, 0.231997, -0.078199, -0.194545, 0.065628, 0.006694, 0.020231, -0.102582, 0.099264, -0.156986, -0.180522, 0.221902, -0.082872, -0.214669, 0.076081, -0.149755, -0.01881, 0.182153, -0.160948, -0.020773, -0.09246, 0.170015, 0.133021, -0.132163, 0.062787, 0.011991, -0.243871, 0.16223, -0.102003, 0.108512], [-0.127864, 0.1001, 0.030331, 0.147361, 0.233851, -0.156616, 0.065521, -0.190273, 0.064019, 0.222258, 0.066398, 0.041054, -0.134398, 0.063839, 0.339451, -0.097891, 0.060256, -0.246333, -0.09963, 0.000625, -0.016814, 0.002281, -0.158317, 0.173421, 0.132188, -0.492096, 0.214603, -0.00094, 0.068582, 0.118019, 0.059474, 0.070407], [0.220345, 0.169711, 0.097831, 0.053125, -0.022506, 0.171402, -0.105041, 0.157922, 0.158746, 0.107576, 0.021301, -0.119313, -0.169189, 0.00521, -0.11157, 0.135325, 0.098507, 0.095972, 0.178447, 0.475587, -0.013888, 0.127828, -0.013658, 0.112134, 0.035242, -0.093432, -0.003422, 0.00703, -0.229312, -0.079072, 0.05977, 0.119733], [0.139342, 0.014331, 0.04471, -0.08408, -0.043478, 0.031401, 0.042139, 0.117935, 0.009141, 0.028963, -0.218621, 0.096752, -0.205972, -0.075489, -0.024466, 0.152822, -0.076901, 0.001967, 0.15985, 0.058697, -0.009068, 0.101189, 0.082732, -0.107594, 0.035598, 0.119804, -0.023123, 0.043188, -0.412422, 0.243876, -0.188886, -0.268623], [0.122851, 0.110924, 0.074668, -0.101803, -0.074331, 0.346312, 0.000993, 0.162205, 0.099443, 0.203337, 0.313032, 0.087867, 0.099161, 0.173043, -0.176779, 0.007906, 0.367814, 0.187113, 0.182422, 0.061598, 0.09775, 0.018805, 0.101309, 0.144799, 0.098936, -0.068273, -0.083186, 0.120742, -0.271965, -0.065795, 0.05835, 0.167443], [0.23991, -0.126028, 0.005557, 0.017999, -0.013002, 0.201578, -0.205303, 0.020352, -0.032884, -0.118664, -0.044177, -0.136867, 0.23925, 0.119521, 0.061317, 0.171129, 0.166104, 0.029923, 0.139515, -0.2239, 0.081598, 0.054991, -0.059111, -0.120188, -0.118557, 0.012641, -0.138461, -0.061625, 0.169792, 0.139027, -0.158327, -0.084007], [-0.287632, 0.066969, -0.0492, 0.106595, 0.056062, -0.153107, -0.031944, 0.038004, 0.074844, 0.100943, 0.069094, 0.137948, -0.050679, 0.034625, -0.085278, 0.017883, 0.048643, -0.204522, 0.004829, -0.009387, 0.135188, 0.028911, 0.017145, 0.257198, 0.123371, 0.123493, 0.00723, 8.2e-05, -0.148852, -0.14138, 0.101414, 0.008686], [-0.026939, -0.119487, 0.111633, -0.021194, -0.069205, -0.051018, -0.00405, 0.000673, -0.145024, -0.226587, 0.053011, 0.037502, 0.307145, -0.025795, -0.01448, -0.037543, -0.019204, 0.057622, -0.185356, -0.264668, -0.176829, 0.007336, -0.001309, 0.08167, 0.038455, -0.20291, 0.01287, -0.051368, 0.087944, 0.037188, 0.003512, -0.039625], [0.113921, -0.078468, -0.117688, -0.214827, -0.055002, -0.149667, -0.18331, -0.064482, -0.064472, 0.04988, -0.109556, 0.090426, -0.048426, 0.076613, -0.074023, 0.113389, 0.210941, -0.016121, 0.093682, -0.182405, 0.072816, 0.000862, 0.047418, -0.233182, 0.048227, 0.047419, 0.001766, 0.161613, -0.346364, -0.053142, -0.22256, 0.091807], [0.109874, -0.1557, -0.071334, -0.068689, -0.064598, 0.014343, 0.07958, 0.10123, -0.432336, -0.217687, -0.0282, 0.008432, -0.12294, -0.090171, -0.240249, -0.129492, -0.228514, 0.087806, -0.019361, -0.186874, -0.06816, 0.0838, 0.093174, -0.055361, -0.343358, -0.006497, 0.08302, 0.109086, -0.211702, -0.441553, -0.04291, -0.209506], [0.130898, -0.173052, -0.030539, -0.116095, 0.071672, 0.085065, 0.022221, 0.202368, -0.115792, -0.006749, 0.029439, 0.053154, -0.161797, -0.00952, 0.052636, 0.011176, 0.148383, 0.056938, 0.03514, 0.134952, 0.129167, -0.045445, 0.130388, -0.040894, 0.102788, 0.083568, -0.019368, -0.020956, -0.114749, 0.155549, -0.188688, -0.105102], [0.131068, -0.214445, -0.108132, 0.164429, 0.001627, 0.102438, 0.126649, -0.033784, -0.221457, -0.034774, 0.072536, -0.02191, 0.201395, -0.037555, 0.066705, 0.113522, -0.147691, 0.095102, -0.005891, 0.078237, -0.070858, 0.114355, 0.089199, 0.147344, 0.004786, 0.032573, 0.093149, 0.116415, -0.078253, 0.07214, 0.061784, -0.371208], [-0.024091, -0.052269, 0.071654, -0.061167, -0.031586, -0.017784, 0.004886, -0.228502, 0.011525, 0.035054, 0.163396, 0.03191, 0.153408, -0.034584, -0.009865, 0.034395, -0.089924, 0.174305, -0.23049, -0.007842, -0.05339, -0.027647, 0.073972, 0.066565, 0.009893, 0.115015, -0.049483, 0.112134, 0.2718, 0.088587, -0.020044, 0.03086], [0.123102, 0.094242, -0.109478, -0.132746, -0.17736, 0.141648, 0.044553, 0.000857, -0.056719, -0.082957, 0.049161, -0.063584, 0.091513, -0.155552, -0.211734, -0.021943, -0.083879, 0.006413, 0.173333, -0.046191, 0.009704, -0.058192, 0.106932, 0.014895, -0.111204, -0.032676, 0.08483, -0.210279, 0.367999, 0.180828, -0.01375, 0.097622], [-0.061101, -0.050371, -0.104498, -0.020663, -0.060319, 0.072237, 0.098809, 0.114244, 0.103014, -0.076022, 0.035846, 0.11132, 0.060142, -0.017663, -0.131485, -0.006649, 0.179672, -0.104009, -0.017242, 0.057596, 0.151866, 0.065221, 0.051056, -0.069069, 0.12811, -0.100947, 0.08564, 0.026525, -0.045684, -0.174084, -0.0257, 0.086087], [0.254669, 0.00019, -0.073477, -0.111899, -0.09107, 0.14034, -0.012405, 0.133542, -0.125927, -0.02176, -0.134574, -0.06409, 0.271889, 0.029383, 0.107203, 0.218747, 0.060443, 0.103005, 0.137067, 0.184728, 0.077602, -0.262015, 0.224363, 0.009421, -0.130093, 0.243521, -0.133779, -0.040594, 0.125159, 0.208362, -0.324905, -0.164797], [-0.011605, 0.073082, 0.010732, 0.05873, 0.011941, -0.037955, 0.024657, 0.013313, 0.04014, -0.041547, 0.132238, -0.012712, 0.072126, -0.170508, -0.339702, 0.164761, -0.097251, -0.066572, 0.048929, 0.220265, -0.042514, -0.074899, 0.179506, -0.03795, -0.02591, 0.18917, 0.192839, -0.087892, 0.220515, -0.436626, 0.095296, 0.099692], [0.127901, -0.067776, -0.162785, -0.090501, 0.11631, 0.086253, -0.184511, 0.004625, -0.170954, -0.091943, -0.178872, -0.044686, 0.08721, 0.074187, -0.189695, 0.060468, 0.138161, -0.053003, 0.120776, -0.233522, -0.043057, -0.062526, 0.090492, -0.143346, -0.01501, 0.023724, 0.099755, -0.01144, 0.012086, -0.240657, 0.094911, 0.062055], [-0.032054, -0.165795, 0.106599, -0.090774, -0.105493, -0.04758, 0.063181, 0.014059, -0.287969, -0.1449, -0.131875, -0.137802, -0.076438, -0.138486, -0.029399, 0.139899, -0.026677, 0.052212, 0.0058, 0.133563, 0.087412, -0.094397, -0.043939, -0.153122, -0.034025, 0.175357, 0.013172, 0.057774, -0.151398, 0.375284, -0.37496, -0.303583], [0.00372, -0.131014, 0.046055, 0.125431, 0.274934, 0.185685, -0.127764, -0.085693, 0.078306, 0.13778, -0.318278, 0.067185, 0.150927, 0.01109, 0.234862, 0.101141, -0.02889, -0.060836, 0.150122, 0.029175, 0.079453, -0.161625, 0.081872, 0.129673, 0.056399, -0.000121, -0.060118, 0.205633, -0.069185, 0.03348, 0.060314, -0.146752], [-0.026322, -0.074859, -0.036733, -0.035218, 0.137395, 0.000954, -0.110225, -0.047257, -0.263194, -0.126138, -0.166019, 0.082626, -0.034881, -0.205928, -0.084526, -0.076324, -0.069969, -0.194158, -0.147417, -0.277998, -0.08217, -0.091041, -0.070589, -0.101822, -0.181138, -0.055481, 0.040324, 0.112905, 0.056762, 0.259944, 0.087371, -0.211614], [0.052587, -0.040915, -0.102583, -0.202043, -0.188155, -0.061377, -0.050536, -0.081961, -0.155536, -0.202875, 0.055225, -0.041894, 0.060629, -0.048361, -0.237434, -0.059578, 0.0787, 0.046543, 0.006156, -0.206848, -0.101346, 0.031159, 0.155706, -0.082336, 0.097719, -0.019861, 0.014414, -0.178408, 0.017421, -0.177897, -0.061795, -0.089536], [-0.130308, 0.050058, 0.129064, -0.013042, 0.192199, -0.029135, 0.064997, -0.062028, 0.00437, -0.065923, -0.058111, 0.260826, 0.048535, 0.043702, 0.248805, -0.124229, -0.046565, -0.030312, -0.026378, -0.180642, 0.026747, 0.19922, 0.015316, 0.101385, 0.203202, 0.038711, 0.132216, 0.2419, -0.126693, -0.147583, 0.243264, 0.010561], [-0.017194, -0.027663, -0.037541, -0.013444, -0.233614, 0.113986, 0.048266, 0.132421, 0.000654, -0.039133, 0.018998, -0.066649, -0.058359, 0.026554, 0.032796, 0.113265, 0.027297, 0.108551, 0.150332, -0.062748, 0.075272, -0.200981, 0.03195, -0.039757, 0.012095, 0.061875, -0.184752, -0.104223, -0.257798, -0.212658, -0.078785, -0.485927], [-0.085584, 0.04125, 0.174849, 0.198858, -0.046308, -0.059625, -0.019494, -0.113116, 0.078622, 0.118378, 0.190306, -0.08309, -0.007404, -0.064722, 0.162006, -0.082776, 0.005536, -0.130018, -0.093852, 0.095677, 0.048075, 0.10124, 0.08751, 0.144917, 0.027785, 0.17298, -0.051073, 0.059359, 0.056738, -0.095734, 0.167151, 0.178572], [0.028676, -0.174727, -0.188515, 0.032282, 0.133345, 0.134915, -0.019634, 0.159292, -0.083139, -0.103896, 0.023609, -0.135695, -0.148442, -0.160024, -0.092313, 0.157838, -0.403324, 0.033916, -0.015534, -0.122361, -0.04508, -0.127847, 0.080505, 0.026855, -0.168873, 0.060955, 0.102653, -0.056791, 0.116975, -0.360462, 0.10579, -0.288373], [-0.011368, -0.232942, -0.164418, -0.077914, 0.053748, 0.039602, -0.200847, 0.159891, -0.250002, -0.150717, -0.266368, -0.247197, -0.386995, -0.012718, -0.189796, -0.106491, -0.116514, -0.063485, 0.090154, -0.302955, -0.197596, -0.189771, -0.0509, -0.369188, -0.253243, 0.128209, -0.044636, -0.048248, 0.033094, 0.092571, -0.286786, -0.304039], [0.04028, 0.006207, 0.092044, 0.212056, -0.026249, 0.010521, 0.05398, 0.220389, 0.280035, 0.183878, 0.027192, -0.084261, 0.093706, 0.149243, 0.061706, 0.194661, 0.134663, 0.123284, 0.100272, -0.129037, 0.088157, 0.053532, 0.082082, 0.18442, 0.099669, 0.042798, -0.063086, 0.024075, 0.143686, -0.117285, -0.007765, 0.253811], [0.250309, 0.072368, -0.178133, -0.312662, 0.094544, 0.154895, -0.021759, 0.198381, 0.143204, 0.123952, -0.01165, 0.081208, -0.071488, -0.053778, -0.4844, -0.107969, 0.04486, 0.169999, 0.256695, 0.084886, 0.012798, 0.010753, 0.200569, -0.061685, 0.199763, -0.027166, -0.037801, 0.060125, -0.186053, -0.082197, 0.141382, 0.230657], [0.113797, -0.0584, 0.038436, -0.015817, -0.163481, 0.058543, -0.075492, 0.155792, 0.11666, 0.058661, 0.126433, -0.006776, 0.02237, 0.089879, -0.027978, 0.037508, 0.06212, 0.13569, 0.165446, -0.159367, -0.097196, 0.086892, -0.088475, -0.068405, 0.052856, -0.076765, 0.084897, 0.082109, 0.124629, -0.060437, -0.16732, -0.097236], [-0.004166, 0.078174, 0.214335, 0.02062, -0.092074, -0.173149, -0.026009, -0.155692, 0.290798, 0.247187, 0.13151, 0.063835, -0.043701, -0.083448, 0.033419, -0.230932, 0.106567, -0.086976, -0.234513, 0.277113, -0.024542, 0.121463, -0.26668, 0.148149, 0.128341, -0.267366, 0.078042, 0.035567, -0.409998, 0.0243, 0.524573, 0.133761], [-0.161238, 0.155135, 0.03821, 0.140945, 0.051833, -0.013436, 0.101147, -0.124602, 0.129645, 0.028197, -0.054405, 0.095203, -0.058212, 0.038683, -0.103696, 0.049099, -0.069274, 0.088024, -0.059612, -0.031516, 0.076356, 0.146751, -0.131295, -0.013282, 0.057186, -0.089262, 0.126462, 0.161882, -0.088704, 0.053765, 0.010061, 0.024631], [0.05471, 0.133464, -0.025552, -0.400199, -0.128748, 0.179087, 0.017507, 0.03179, 0.06557, -0.060394, 0.157729, 0.138763, 0.025308, 0.138585, -0.066298, 0.068771, 0.107032, 0.129365, 0.291525, 0.029891, -0.088779, 0.158241, 0.064186, -0.104236, -0.035218, -0.015894, 0.02293, -0.080416, -0.121305, 0.012689, 0.068356, -0.022478], [0.06127, -0.023265, -0.109813, -0.122086, -0.105589, 0.240515, 0.031597, 0.291302, 0.157294, 0.119663, -0.097148, 0.045988, 0.059577, -0.048684, -0.031396, 0.155653, -0.023545, 0.323271, 0.287605, 0.178274, 0.152678, 0.116479, 0.384908, -0.062413, 0.009313, 0.273629, -0.030844, -0.083889, -0.378215, 0.175144, 0.001041, 0.227803], [0.032127, -0.014589, -0.024797, 0.068362, -0.055341, -0.274704, 0.007698, -0.215059, -0.025194, -0.001718, -0.064971, 0.00968, -0.134413, 0.028983, 0.121814, 0.070755, -0.026817, -0.239417, -0.053085, -0.080907, -0.01999, -0.114839, -0.06665, -0.055801, -0.058204, -0.004518, -0.085565, -0.081506, -0.009744, -0.195463, 0.026776, 0.163157], [0.00447, 0.03639, 0.007881, -0.0324, -0.052765, 0.238931, 0.089061, 0.126288, 0.074498, 0.064139, -0.113213, -0.084882, 0.109371, 0.144416, -0.132751, 0.066684, 0.125324, 0.138097, 0.089668, -0.179462, -0.104234, 0.151458, 0.018713, 0.0469, 0.06726, -0.010094, -0.052364, -0.065663, -0.021654, -0.064018, -0.02727, -0.404703], [-0.071679, -0.052094, 0.055411, 0.200708, 0.083119, -0.17541, 0.109103, 0.010722, 0.110917, 0.033308, 0.064915, 0.023689, -0.085351, 0.026021, 0.226994, 0.110162, -0.055208, -0.217421, -0.248818, 0.092432, -0.091049, 0.121382, -0.11576, 0.02739, 0.004375, -0.084406, -0.071844, 0.078659, -0.25792, -0.032481, 0.065395, -0.074766], [0.032575, -0.114282, -0.12019, 0.057896, -0.082781, 0.054487, -0.137452, 0.177709, -0.013677, 0.010968, 0.063253, 0.004973, -0.379544, 0.003954, 0.074162, 0.116048, -0.209415, 0.104891, 0.007274, -0.058128, -0.093698, -0.083584, 0.035347, -0.051156, -0.066652, 0.024661, 0.002857, -0.068786, -0.137431, 0.0931, 0.030732, -0.151323], [0.064633, 0.015106, -0.02373, 0.120925, 0.098489, -0.260869, 0.102057, -0.197597, 0.021477, -0.011186, -0.134821, -0.041045, 0.054804, 0.047952, -0.066201, 0.035607, -0.005908, -0.067583, -0.431257, -0.231206, 0.024523, 0.034836, -0.107123, -0.023118, -0.094242, 0.204867, -0.089282, -0.065399, 0.117496, -0.008557, -0.109745, 0.024522], [0.173843, -0.286527, 0.088991, 0.122207, 0.036307, 0.152636, -0.306006, 0.019869, 0.065554, -0.175088, -0.154257, -0.223971, -0.029311, -0.223474, 0.254379, 0.100012, -0.346845, 0.141801, 0.16398, -0.150932, -0.054705, -0.52906, 0.097041, 0.166591, -0.215405, 0.08603, 0.003255, -0.137712, -0.001266, 0.044954, 0.313641, -0.022941], [-0.035093, 0.070343, 0.014289, 0.125664, 0.107512, -0.221698, -0.062738, 0.023508, 0.132283, 0.086021, -0.056329, -0.05724, 0.198225, -0.089817, 0.040153, -0.06124, -0.059146, -0.06344, 0.010064, 0.147501, 0.152963, 0.06286, -0.088092, 0.151303, -0.041797, 0.158353, 0.04891, 0.115058, -0.255617, -0.029249, 0.105164, 0.225823], [0.096503, 0.112997, 0.095074, 0.099847, -0.028226, -0.200244, 0.049018, 0.025294, -0.053166, 0.037201, -0.013159, -0.110487, -0.30826, 0.04634, -0.114995, -0.302816, 0.10713, -0.226805, -0.029982, 0.021302, 0.20034, 0.063398, -0.172047, -0.116742, -0.102294, -0.437993, 0.097666, 0.124791, -0.029232, -0.037607, -0.154104, 0.080193], [0.024312, -0.08393, 0.066802, -0.109087, 0.027492, 0.146168, 0.069775, 0.163292, -0.133529, 0.07748, -0.017652, 0.004692, -0.083904, 0.038067, -0.104503, 0.072714, 0.052284, 0.082806, 0.050265, 0.158396, -0.021755, -0.089465, 0.104692, -0.018574, 0.003738, 0.136139, -0.071506, -0.031652, -0.171197, -0.226054, -0.063686, -0.243707], [0.084253, 0.022505, -0.014043, 0.140971, 0.058095, -0.019403, -0.139748, -0.065882, 0.090266, -0.044104, -0.314723, -0.180471, 0.120163, 0.051197, 0.313395, 0.107574, -0.270066, -0.064055, -0.350659, -0.185156, -0.309848, -0.208529, 0.149818, 0.071241, -0.072471, 0.07706, -0.030439, 0.053199, 0.188728, -0.038837, -0.109772, -0.170668], [0.128316, 0.104649, -0.051605, -0.073708, -0.044917, -0.002815, 0.136637, 0.062716, 0.09235, -0.067197, -0.06871, -0.00304, -0.058073, -0.036501, -0.029646, 0.092467, 0.079511, -0.026672, 0.073361, -0.041193, -0.041558, 0.119423, -0.077953, -0.048456, 0.015318, -0.139737, -0.011054, 0.085831, -0.144507, 0.171, 0.123792, -0.05854], [0.2053, 0.005422, 0.081912, 0.02389, -0.072867, 0.183871, 0.016711, 0.034217, -0.085485, 0.026153, 0.052595, -0.250824, 0.255787, 0.05699, -0.018871, 0.182389, 0.201012, 0.159352, 0.013226, -0.057931, -0.044893, 0.057419, 0.022397, -0.178164, -0.15507, -0.008347, 0.06438, -0.131584, 0.496723, 0.27586, -0.36179, 0.078162], [-0.175714, 0.051524, -0.021372, 0.058637, -0.008003, 0.103624, 0.089895, 0.148511, -0.061136, -0.103164, -0.021471, 0.050589, 0.180177, 0.009703, -0.052278, 0.003077, -0.139083, -0.125327, -0.034292, -0.186331, -0.407102, 0.012043, -0.057304, -0.090332, -0.178531, -0.039706, -0.030869, 0.087111, 0.000795, -0.312755, 0.114956, -0.48494], [-0.200958, -0.070753, -0.01235, 0.156133, -0.057275, -0.274353, 0.000326, 0.067991, -0.048389, 0.157873, 0.139724, -0.066422, -0.237968, 0.163361, 0.109738, -0.11472, -0.016362, 0.109051, -0.206132, 0.041035, -0.023022, -0.027778, 0.013161, 0.10774, 0.059586, 0.25916, 0.113193, 0.044018, 0.080174, 0.148534, 0.067944, 0.148869], [-0.118398, -0.045155, -0.028521, -0.323519, 0.058256, 0.0129, 0.106246, -0.185628, -0.037515, -0.229254, -0.081164, 0.12348, 0.070179, 0.034883, -0.05643, -0.106964, 0.008924, 0.086158, -0.002586, -0.240152, 0.0819, -0.049463, 0.020784, -0.308677, -0.136937, -0.140838, -0.189406, -0.071386, -0.072577, 0.126165, 0.081155, -0.134928], [0.019268, 0.101826, 0.097783, 0.112263, 0.065297, 0.113077, 0.105296, 0.0044, -0.027222, -0.168776, 0.094309, 0.073251, -0.132631, -0.090586, 0.096506, 0.125819, -0.013392, 0.12395, -0.05251, -0.550107, -0.256254, 0.010449, 0.152967, 0.115959, 0.104209, 0.070013, -0.055696, 0.026003, -0.377699, -0.136648, 0.205784, -0.297808], [-0.049155, -0.060675, 0.064939, -0.070822, -0.061633, 0.12771, 0.043485, 0.159675, -0.089223, -0.002346, -0.066135, -0.197242, -0.004028, -0.041074, -0.395718, 0.020478, -0.146751, -0.028466, 0.117732, 0.184276, -0.238828, -0.098137, -0.082273, -0.0862, -0.032464, 0.133147, 0.003246, 0.001263, 0.114644, -0.052817, -0.244385, -0.114537], [-0.118711, -0.416351, -0.031699, 0.101834, 0.194116, -0.06505, -0.043306, -0.003201, 0.086067, 0.022292, -0.117867, -0.179005, -0.129537, 0.00132, 0.157414, -0.143238, -0.354784, -0.196748, 0.014394, -0.212457, -0.109748, -0.448896, -0.081413, 0.047032, -0.166007, -0.023528, 0.049563, -0.101788, 0.062717, -0.094502, 0.141402, 0.052132], [-0.254251, 0.065471, 0.122378, -0.044396, -0.152849, -0.410794, -0.022974, -0.431292, 0.092631, 0.047366, 0.11982, -0.204462, -0.342046, 0.08807, -0.262792, -0.065799, -0.076978, -0.484641, -0.267842, -0.05867, 0.017098, 0.132281, -0.018145, 0.090781, 0.081708, -0.001065, -0.001835, 0.076923, 0.060631, -0.287911, 0.011309, 0.188143], [0.009779, -0.05804, -0.168472, -0.178747, -0.027844, -0.166454, 0.012057, 0.001479, 0.100721, 0.106219, 0.008591, 0.018115, -0.050994, 0.14133, -0.095886, -0.101058, 0.093488, -0.088769, -0.12331, 0.023056, 0.015972, 0.053384, 0.121005, 0.045412, -0.009581, -0.141408, 0.116401, 0.131879, -0.244886, -0.00894, 0.10436, 0.089494], [0.084439, 0.114946, 0.001871, 0.173107, 0.135327, 0.140458, 0.048676, 0.172013, 0.04834, 0.262094, 0.212775, 0.046087, 0.198751, 0.029058, -0.303999, 0.098822, 0.142891, 0.220475, 0.183512, 0.432699, 0.155088, 0.176387, -0.043254, 0.044642, 0.072862, -0.0063, 0.024461, 0.165712, 0.047124, 0.125169, 0.087031, 0.373344], [0.011294, 0.087285, -0.090258, -0.090092, -0.018853, 0.025044, 0.026042, 0.022825, 0.108854, 0.091779, 0.050396, 0.066802, 0.160659, 0.091077, 0.018102, 0.114976, 0.120184, 0.032684, 0.142752, -0.094774, -0.025167, 0.067995, 0.121225, -0.154617, -0.040925, 0.016257, 0.087178, -0.00309, -0.217431, -0.217435, 0.075687, 0.08219], [-0.147385, -0.056743, -0.28422, -0.086014, -0.002713, -0.357214, 0.129872, -0.247251, -0.12279, -0.054379, -0.057356, 0.063296, -0.009079, -0.10357, 0.082524, -0.328865, 0.031554, -0.333041, -0.357206, -0.056286, 0.137259, -0.07677, -0.166356, 0.05982, -0.07235, -0.100269, 0.031801, 0.078074, 0.04586, -0.004741, -0.203024, 0.235722], [-0.106693, 0.154122, 0.058859, 0.250889, 0.110139, -0.032097, 0.158874, -0.04087, 0.071071, 0.036423, 0.161888, 0.124929, 0.112593, 0.113296, 0.132717, 0.03677, 0.113791, -0.189933, -0.142139, 0.004826, 0.140419, 0.112044, -0.101882, 0.022626, 0.045778, 0.106975, -0.002197, 0.076978, -0.376805, -0.158227, 0.189647, 0.14406], [-0.165818, -0.029611, -0.055457, 0.003808, 0.081425, -0.379065, 0.149352, -0.261571, -0.1157, 0.055449, 0.144496, -0.052335, -0.212696, 0.048349, -0.177356, -0.014983, -0.053756, -0.20859, -0.224192, 0.216548, 0.124043, 0.018651, -0.040301, 0.118091, -0.104682, -0.10599, 0.03219, 0.097352, 0.056067, 0.220947, 0.13583, -0.038528], [-0.005474, 0.197717, 0.108718, 0.287275, -0.039137, 0.132311, 0.131601, -0.038812, 0.299249, 0.128073, -0.038266, 0.051973, 0.140164, 0.06887, 0.158762, 0.056564, 0.018665, 0.080799, 0.036475, -0.299036, -0.030438, 0.093204, 0.009066, 0.167621, 0.203344, 0.104504, 0.126421, -0.03224, -0.251113, -0.015607, 0.265357, -0.022987], [-0.010701, 0.041017, -0.009355, -0.132852, 0.026557, -0.427887, -0.023595, -0.311665, 0.01745, -0.060045, 0.131579, 0.060338, -0.330521, -0.03264, -0.16545, -0.131896, 0.134746, -0.314436, -0.13644, 0.038275, 0.055801, 0.062399, -0.248822, 0.014555, -0.010258, 0.128571, 0.041624, 0.121491, -0.06776, -0.181443, 0.00243, -0.039988], [-0.024636, 0.082793, -0.176963, -0.286473, 0.013873, 0.094839, -0.081978, 0.07736, -0.062336, -0.085668, -0.040439, -0.029051, 0.197847, -0.048568, -0.198294, 0.131467, 0.05294, 0.146396, 0.043818, -0.135687, 0.117622, 0.069151, 0.008601, -0.088697, -0.00619, 0.247343, -0.017471, 0.030441, -0.116917, 0.22012, 0.017288, 0.127713], [0.045497, -0.30345, -0.104293, -0.041788, 0.023505, -0.339418, 0.012941, -0.288823, -0.258117, -0.279676, 0.001839, 0.099158, -0.236476, -0.010414, -0.282063, -0.190754, -0.069622, -0.094791, -0.23405, -0.199395, -0.160961, -0.047323, -0.021173, 0.155799, -0.289372, 0.056291, 0.05909, -0.050311, 0.157302, 0.224518, -0.127793, -0.271878], [0.002916, -0.193035, 0.060543, 0.098667, -0.000535, 0.188804, -0.117386, -0.08597, -0.214649, -0.020712, -0.195775, -0.099075, 0.030709, -0.082629, -0.006526, 0.077774, -0.137049, 0.070197, -0.154354, -0.1315, -0.009992, -0.196368, 0.198989, -0.141757, -0.214739, 0.039619, -0.026701, -0.213002, -0.16221, -0.209018, 0.014656, -0.041104]], [0.11289, 0.055956, -0.010228, -0.118373, 0.182886, -0.019918, 0.088589, -0.077526, -0.0533, -0.099509, -0.049883, 0.007715, -0.116894, 0.134515, -0.22339, 0.088667, 0.131947, 0.075171, -0.035012, -0.098662, 0.079937, 0.065134, 0.124256, 0.055086, 0.003467, -0.217216, -0.044476, 0.177907, 0.067975, -0.351005, -0.124049, -0.054866]),
    ([[0.152242], [-0.074794], [-0.144717], [-0.286318], [-0.101388], [0.118989], [-0.05699], [0.259481], [-0.230173], [-0.160375], [-0.11883], [-0.031271], [0.313857], [-0.14965], [-0.19172], [0.146565], [-0.077508], [0.193014], [0.317582], [-0.339685], [-0.258994], [-0.200792], [0.174317], [-0.284901], [-0.080518], [0.094208], [-0.073028], [-0.199491], [-0.111615], [0.448889], [-0.07717], [-0.314018]], [0.13587]),
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
        # exp_r4_xens: blend weight of the R4-trained table set (0 = mm_r3_gru3, 1 = R4 tables only); xwq / xws < 0 = same as xw
        'xw': 0.3,
        'xwq': -1.0,
        'xws': -1.0,
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
        self.xw = float(params['xw'])
        self.xwq = self.xw if float(params['xwq']) < 0 else float(params['xwq'])
        self.xws = self.xw if float(params['xws']) < 0 else float(params['xws'])
        self.xb = 1 if max(self.xw, self.xwq, self.xws) > 0.0 else 0
        self.b_c0, self.b_w = R4_PRED
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
            sets = [(SEQ_TM, SEQ_TSD, SEQ_WI, SEQ_BI, SEQ_WH, SEQ_BH)]
            if self.xb:
                sets.append((R4_SEQ_TM, R4_SEQ_TSD, R4_SEQ_WI, R4_SEQ_BI, R4_SEQ_WH, R4_SEQ_BH))
            k = self.sq_k = len(sets)
            n = self.sq_n = len(SEQ_BH) // 3
            nt = self.sq_nt = len(SEQ_TM)
            kn = k * n
            # block-diagonal fusion: one matmul steps every GRU. Rows [tokens of set 0..k-1 | hidden of set 0..k-1],
            # columns [r (k*n) | z (k*n) | n input part (k*n) | n hidden part (k*n)], each with set 0 first.
            W = np.zeros((k * nt + kn, 4 * kn))
            B = np.zeros(4 * kn)
            lo = []; hi = []
            for s, (tm_, tsd_, wi_, bi_, wh_, bh_) in enumerate(sets):
                tm_ = np.array(tm_); tsd_ = np.array(tsd_); wi_ = np.array(wi_); bi_ = np.array(bi_); wh_ = np.array(wh_); bh_ = np.array(bh_)
                assert len(bh_) // 3 == n and len(tm_) == nt
                wn = wi_ / tsd_[:, None]
                off = (tm_ / tsd_) @ wi_
                rt = slice(s * nt, (s + 1) * nt); rh = slice(k * nt + s * n, k * nt + (s + 1) * n)
                W[rt, s * n:(s + 1) * n] = 0.5 * wn[:, :n]
                W[rh, s * n:(s + 1) * n] = 0.5 * wh_[:, :n]
                W[rt, kn + s * n:kn + (s + 1) * n] = 0.5 * wn[:, n:2 * n]
                W[rh, kn + s * n:kn + (s + 1) * n] = 0.5 * wh_[:, n:2 * n]
                W[rt, 2 * kn + s * n:2 * kn + (s + 1) * n] = wn[:, 2 * n:]
                W[rh, 3 * kn + s * n:3 * kn + (s + 1) * n] = wh_[:, 2 * n:]
                bz = 0.5 * (bi_[:2 * n] - off[:2 * n] + bh_[:2 * n])
                B[s * n:(s + 1) * n] = bz[:n]; B[kn + s * n:kn + (s + 1) * n] = bz[n:]
                B[2 * kn + s * n:2 * kn + (s + 1) * n] = bi_[2 * n:] - off[2 * n:]
                B[3 * kn + s * n:3 * kn + (s + 1) * n] = bh_[2 * n:]
                lo.append(tm_ - 8.0 * tsd_); hi.append(tm_ + 8.0 * tsd_)
            self.sq_w = W; self.sq_b = B
            self.sq_lo = np.concatenate(lo); self.sq_hi = np.concatenate(hi)
            self.sq_buf = np.zeros(k * nt + kn)
            self.sq_tv = self.sq_buf[:k * nt]
            self.sq_h = self.sq_buf[k * nt:]  # the GRU states live in the input buffer
            self.sq_t = -1
            self.sq_bb = self.sq_ba = -1
            self.sq_qb = self.sq_qa = 0
            self.sq_ltp = 0
        def _arr(ls):
            return [(np.array(w), np.array(b)) for w, b in ls]
        self.stk = [(np.array(MLP_MEAN), np.array(MLP_SCALE), _arr(MLP_LAYERS), np.array(SEQ_XM), np.array(SEQ_XS), _arr(SEQ_HEAD),
                     SEQ_YSC, self.mq_mean if self.mq_h else None, self.mq_scale if self.mq_h else None, self.mq_layers if self.mq_h else None)]
        if self.xb:
            assert self.mlp_on and self.seq_on and self.mq_h
            q4 = R4_MLPQ[self.mq_h]
            self.stk.append((np.array(R4_MLP_MEAN), np.array(R4_MLP_SCALE), _arr(R4_MLP_LAYERS), np.array(R4_SEQ_XM), np.array(R4_SEQ_XS),
                             _arr(R4_SEQ_HEAD), R4_SEQ_YSC, np.array(q4[0]), np.array(q4[1]), _arr(q4[2])))
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
        tv_[:self.sq_nt] = (log1p(dt / 100.0), min(max(bb - pbb, -5), 5), min(max(ba - pba, -5), 5), min(max(mid - pmid, -5.0), 5.0),
                        (log1p(db) if db > 0 else -log1p(-db)), (log1p(da) if da > 0 else -log1p(-da)),
                        (log1p(tv) if tv > 0 else -log1p(-tv)), sw, min(max(ltp - pmid, -10.0), 10.0) if tc else 0.0,
                        min(ba - bb, 20), log1p(qb), log1p(qa), (qb - qa) / max(qb + qa, 1))
        if self.sq_k > 1:
            tv_[self.sq_nt:] = tv_[:self.sq_nt]
        np.maximum(tv_, self.sq_lo, out=tv_)
        np.minimum(tv_, self.sq_hi, out=tv_)
        n = self.sq_n * self.sq_k
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

    def _ridge(self, c0, w, now, pg, f70, af, fw, s100, ltv, ltp, spread, mid2, rm2, qb, qa, nb, na) -> float:
        p = (c0 + w[0] * pg[0] + w[1] * pg[1] + w[2] * f70 + w[3] * pg[3] + w[4] * pg[4]
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
        return p

    def _stack(self, s: int, h0, pr: float):
        """Learned stack of one table set on the 22 features: (taker value m, quote value). pr = that set's ridge."""
        mean, scale, mls, xm, xs, head, ysc, qmean, qscale, qls = self.stk[s]
        h = (h0 - mean) / scale
        sq = self.sq_h[s * self.sq_n:(s + 1) * self.sq_n]
        if self.seq_mix < 1.0:
            hm0 = np.clip(h, -8.0, 8.0)
            hm0 = np.maximum(hm0 @ mls[0][0] + mls[0][1], 0.0)
            hm0 = np.maximum(hm0 @ mls[1][0] + mls[1][1], 0.0)
            m0 = float((hm0 @ mls[2][0] + mls[2][1])[0])
        h = np.clip((h * scale + mean - xm) / xs, -8.0, 8.0)
        h = np.maximum(np.concatenate((h, sq)) @ head[0][0] + head[0][1], 0.0)
        h = np.maximum(h @ head[1][0] + head[1][1], 0.0)
        m = float((h @ head[2][0] + head[2][1])[0]) * ysc
        if self.seq_mix < 1.0:
            m = self.seq_mix * m + (1.0 - self.seq_mix) * m0
        if self.mq_h:
            g = np.clip((h0 - qmean) / qscale, -8.0, 8.0)
            g = np.maximum(g @ qls[0][0] + qls[0][1], 0.0)
            g = np.maximum(g @ qls[1][0] + qls[1][1], 0.0)
            return m, float((g @ qls[2][0] + qls[2][1])[0])
        return m, (1.0 - self.mlp_mix) * pr + self.mlp_mix * m

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
        p = self._ridge(self.pred_c0, self.pred_w, now, pg, f70, af, fw, s100, ltv, ltp, spread, mid2, rm2, qb, qa, nb, na)
        pR = p
        if self.xb:
            pB = self._ridge(self.b_c0, self.b_w, now, pg, f70, af, fw, s100, ltv, ltp, spread, mid2, rm2, qb, qa, nb, na)
            pA = p
            p = (1.0 - self.xw) * pA + self.xw * pB
            pR = (1.0 - self.xwq) * pA + self.xwq * pB
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
            if self.xb:
                m3, q3 = self._stack(0, h0, pA)
                mB, qB = self._stack(1, h0, pB)
                p3 = (1.0 - self.mlp_mix) * pA + self.mlp_mix * m3
                p4 = (1.0 - self.mlp_mix) * pB + self.mlp_mix * mB
                p = (1.0 - self.xw) * p3 + self.xw * p4
                pq = (1.0 - self.xwq) * q3 + self.xwq * qB
            else:
                m3, pq = self._stack(0, h0, p)
                p = (1.0 - self.mlp_mix) * p + self.mlp_mix * m3
            self.pred_raw = pq if self.mq_h else p
        else:
            self.pred_raw = pR
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

    def _slw_lin(self, t3, t4, xa) -> float:
        mn, sc, w, c = t3
        p = float(np.clip((xa - mn) / sc, -8.0, 8.0) @ w) + c
        if self.xws > 0.0:
            mn, sc, w, c = t4
            p = (1.0 - self.xws) * p + self.xws * (float(np.clip((xa - mn) / sc, -8.0, 8.0) @ w) + c)
        return p

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
            p += self._slw_lin(SLW_STD1, R4_SLW_STD1, xa)
        if self.slw_h in (5, 6):
            p += self._slw_lin(SLW_STD5, R4_SLW_STD5, xa)
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
