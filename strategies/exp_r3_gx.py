"""mm_r3_gru2 (home, R3 upload candidate): mm_r3_gru (leaderboard 976,148) with gate_thr 14 (from 11): round3 638,024 / mirror 625,673 / round2 447,035; holdout 157,152 / 158,439 / 112,310 (gru 634,519 / 624,778 / 437,941; holdout 156,075 / 157,441 / 109,156); gate 17 / 22 on the same plateau / lower.

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
# exp_r3_gx: GRU + head weights trained on round3 + mirror + round2 + mirror first 75 % (research/diag/mlp2x_r3.py, seed 3, export by seq_r3.py).
SEQ_LITE = True
SEQ_TM = [2.785425, -3.6e-05, 3.6e-05, 0.0, 0.05362, 0.05362, 0.0, 0.0, 0.0, 1.215438, 6.052741, 6.052741, 0.0]
SEQ_TSD = [1.084381, 0.11764, 0.11764, 0.089737, 1.514138, 1.514138, 0.644491, 0.194564, 0.152923, 0.568171, 1.62469, 1.62469, 0.643755]
SEQ_XM = [0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 1.220976, -0.0, 0.0, 6.04486, 6.045293, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.000858, 4.078255]
SEQ_XS = [0.861619, 0.738671, 0.69536, 0.597753, 0.510978, 1.88118, 9.619223, 0.563317, 8.326649, 0.712977, 0.643663, 0.539282, 1.627786, 1.627759, 0.999999, 4.385919, 0.963627, 0.357081, 128.598709, 307.96048, 894.555725, 1.345891]
SEQ_YSC = 1.3769046068191528
SEQ_WI = [[-0.278667, -0.070721, -0.886666, -0.272579, -0.166848, 0.65566, -0.077416, -0.229041, -0.525288, -0.07702, -0.09747, 0.038377, -0.069472, -0.156059, -0.110966, -0.742511, -0.134672, 0.001394, -0.069795, 0.035039, -0.324192, -0.185162, -0.122867, 0.137179, -0.027384, 0.074076, 0.555826, -0.175223, 0.295878, -0.006533, 0.050633, -0.124573, -0.294945, 0.194117, 0.017016, -0.044371, 0.153577, -0.198946, 0.486308, 0.029522, -0.113967, 0.127787, -0.049294, -0.349499, 0.151395, 0.175167, -0.419039, -0.099539], [0.472768, 0.288257, 0.083415, -0.301317, 0.147999, 0.027097, 0.293884, -0.188736, -0.126302, 0.122368, 0.246284, -0.078949, 0.2559, -0.254095, 0.432966, 0.076657, -0.01299, -0.119542, -0.205103, -0.15746, -0.381063, -0.127592, 0.421955, -0.364408, 0.038677, 0.295042, -0.211071, 0.00781, -0.092453, 0.501992, -0.029725, 0.239867, -0.304383, -0.31921, 0.077197, 0.278974, 0.128707, 0.257454, -0.386305, -0.09358, -0.283933, -0.277735, 0.339313, -0.397702, 0.100134, 0.227341, 0.416021, 0.04502], [0.083853, -0.342474, 0.462378, -0.049868, 0.181435, -0.090037, -0.410199, -0.026861, -0.574282, -0.340091, -0.132863, -0.583647, 0.256805, -0.065295, 0.037561, 0.501914, 0.264591, 0.494101, -0.147671, 0.648247, -0.025977, -0.32664, 0.298238, 0.143836, 0.023996, 0.125753, -0.16597, 0.415123, 0.317779, -0.043967, 0.011324, -0.086434, -0.071963, 0.643553, 0.019994, 0.002635, -0.130657, 0.129103, 0.308464, 0.087564, 0.064753, 0.168265, -0.235263, 0.237664, 0.108554, -0.238046, 0.021611, -0.01904], [0.119433, 0.139884, 0.007349, 0.013362, 0.124039, 0.232108, -0.233281, -0.167561, -0.178783, -0.405803, -0.297999, -0.426683, 0.443782, -0.671345, 0.058092, 0.104586, -0.015534, -0.012108, -0.322473, 0.41621, -0.076955, 0.003102, 0.489438, -0.219471, 0.093029, 0.100136, -0.509056, 0.496517, 0.281389, -0.112443, 0.10691, -0.035323, 0.004976, -0.112821, -0.08814, -0.113965, 0.234465, 0.135615, 0.154745, 0.003266, -0.252782, -0.116522, 0.10233, -0.062499, -0.341701, -0.287685, 0.190246, -0.072234], [-0.291661, 0.855982, -0.774142, 0.302063, 0.139256, -0.187092, -0.79332, 0.18081, -0.118721, -0.077224, -0.447489, -0.461509, 0.601722, 0.05688, -0.253744, 0.156456, 0.477743, -0.255725, -0.34067, -0.434947, -0.035834, -0.539227, -0.617733, 0.103158, -0.28157, 0.021089, -0.345728, 0.454517, 0.166297, -1.138608, -0.219479, 0.138748, -0.223279, -0.328816, -0.647243, -0.317259, 0.429082, 0.425669, 0.502412, 0.045214, 0.646376, 0.370793, -0.236882, -0.02757, 0.137137, 0.85986, -0.058242, -0.185975], [-0.122455, 0.628171, 0.020373, 0.349755, 0.34466, 0.320317, -0.243748, -0.407605, -0.720177, 0.308073, 0.591297, 0.074543, 0.020727, 0.402231, -0.088542, -1.030208, -0.162674, -0.245791, 0.290393, 0.209996, -0.437408, 0.334561, -0.021134, -0.111606, 0.056268, 0.333231, 0.038669, -0.382293, -0.058608, -0.122231, -0.174889, -1.294057, 0.277249, -0.267827, 0.004813, 0.248136, -0.203696, -0.166784, 0.202522, 0.153938, -0.070325, -0.004924, 0.387685, -0.661073, 0.466475, 0.122607, -0.221785, -0.750809], [0.247025, 0.216928, 0.4153, -0.012166, 0.478503, 0.347994, 0.270683, 0.605079, -0.060952, -0.230563, -0.140975, 0.419927, 0.049141, 0.292403, -0.442679, -0.228774, -0.275605, 0.390188, -0.265848, -0.090421, 0.245412, 0.023435, -0.059778, -0.444576, 0.25545, 0.311962, -0.230855, -0.368197, -0.069346, -0.221641, 0.121421, 0.156685, -0.098013, 0.158914, -0.404562, 0.305103, 0.120209, 0.006949, -0.000213, -0.004469, 0.29132, 0.082276, -0.241314, -0.147575, 0.127231, -0.003139, 0.134849, 0.274274], [0.159501, 0.041297, 0.336887, -0.089071, 0.361652, -0.065147, -0.155388, 0.274316, -0.013967, -0.252137, -0.003075, -0.221837, 0.081206, -0.101097, -0.259621, -0.1662, -0.168702, 0.54927, -0.008704, 0.074017, 0.254349, -0.09046, 0.299789, -0.447519, 0.482692, 0.117693, -0.452096, -0.304086, 0.19865, -0.267084, 0.044469, 0.271339, 0.199311, 0.049034, -0.164649, 0.0409, -0.103204, -0.227782, 0.046697, -0.000922, 0.034175, -0.13048, 0.296787, 0.127239, 0.108837, 0.242724, 0.238789, 0.244597], [0.015226, -0.144261, 0.340933, 0.137316, 0.360895, -0.00385, 0.096329, 0.228093, -0.212858, -0.414871, -0.35021, 0.198088, -0.228219, 0.233833, -0.127463, -0.161472, -0.214496, 0.265711, -0.11766, -0.317081, 0.192911, -0.013779, 0.518696, -0.419459, 0.449892, 0.259439, -0.261332, -0.330649, 0.387893, -0.256374, 0.056226, 0.362769, 0.002775, -0.055239, 0.156698, -0.179607, -0.084029, 0.066714, -0.126742, -0.092414, 0.110203, 0.001132, -0.062691, -0.091598, 0.131192, 0.078637, 0.017613, -0.178732], [0.009493, -0.006176, 0.05735, -0.171072, 0.212456, 0.105291, -0.025374, -0.070294, -0.300866, -0.133806, 0.134312, -0.046826, -0.110901, 0.215797, 0.087456, 0.220957, 0.229485, 0.002251, 0.227406, 0.620712, -0.064123, -0.030662, 0.390628, 0.278278, 0.385021, 0.221516, -0.038, -0.195318, -0.164063, -0.129502, -0.109708, -0.271068, -0.188349, 0.219416, 0.301085, -0.534475, 0.336994, -0.246608, 0.036502, 0.46284, -0.158487, 0.010078, -0.074533, -0.164958, 0.42954, 0.026034, -0.136368, 0.136873], [0.0723, -0.292481, -0.116724, 0.197363, -0.421191, 0.172628, -0.884467, -0.271868, 0.111685, 0.053126, 0.054203, 0.024352, -0.124264, -0.282586, -0.097576, -0.41889, 0.105995, 0.09354, -0.084485, 0.094804, 0.01377, -0.018657, -0.744833, -0.403473, -0.241004, 0.02002, -0.157001, -0.029045, 0.005663, 0.003595, 0.064751, 0.0942, 0.000868, -0.173475, -0.115102, -0.06044, -0.077786, 0.047031, 0.188727, -0.114666, -0.211059, 0.146314, -0.18368, -0.066794, -0.07815, -0.181362, -0.021987, -0.164561], [0.051834, -0.217263, -0.562239, -0.118627, -0.424446, 0.163881, -0.276547, 0.086859, -0.592683, -0.401313, 0.059909, -0.119088, -0.315452, 0.116071, -0.16893, -0.589752, -0.032341, 0.054757, 0.088368, 0.009579, 0.041764, 0.082374, 0.025428, 0.07322, -0.015382, 0.012242, 0.271338, -0.080985, 0.25046, -0.0687, -0.179536, -0.212965, -0.035213, -0.097529, -0.059849, 0.167767, 0.129608, 0.141116, -0.071652, 0.158285, 0.012433, 0.010233, -0.172274, -0.086079, -0.001837, -0.001483, 0.003273, 0.146386], [0.040386, -0.217432, 0.031471, -0.026543, -0.123927, 0.065456, -0.315349, -0.347774, 0.743193, 0.28665, 0.323821, 0.015305, 0.008444, -0.009503, 0.236602, -0.122487, 0.323978, 0.183121, 0.026864, 0.093671, -0.337518, -0.166921, -0.049569, -0.102961, 0.207668, 0.1793, -0.357138, 0.163981, -0.257363, -0.245811, 0.424644, 0.165245, -0.111923, 0.180066, 0.149531, -0.009822, -0.357226, -0.146629, 0.321833, 0.147612, -0.044377, 0.24729, 0.384985, -0.188541, -0.078641, 0.188865, -0.215669, 0.02737]]
SEQ_BI = [0.160491, -0.155765, 0.219454, -0.109179, -0.066887, 0.076343, 0.071042, 0.238866, 0.074691, 0.035103, 0.443215, 0.239753, 0.253554, 0.200023, 0.352325, -0.008752, 0.927922, 0.094761, 0.751976, 0.575078, 0.492674, 0.694216, 0.678815, 0.608987, 0.420007, 0.520129, 0.110059, 0.503246, -0.010921, 0.202663, 0.485897, 0.441596, 0.097212, -0.170031, 0.074736, 0.065713, 0.348511, -0.093029, -0.225177, 0.33471, 0.035572, 0.166264, 0.460497, 0.06344, -0.356592, 0.006117, 0.405135, -0.065748]
SEQ_WH = [[0.250821, 0.278989, -0.054297, -0.185412, 0.542731, 0.441115, 0.074134, -0.324849, 0.002199, 0.636427, -0.280077, -0.154893, 0.444044, 0.012992, -0.199779, -0.634903, -0.158341, 0.004641, -0.07124, 0.65382, 0.38131, -0.005634, 0.175495, -0.161891, 0.457503, -0.019665, 0.162687, -0.050595, 0.320317, 0.507216, 0.261495, -0.00375, 0.620673, -0.084306, 0.166854, 0.010197, -0.040025, 0.714456, 0.120791, 0.43829, 0.299679, 0.39603, -0.325995, -0.040254, -0.103834, 0.148009, 0.042295, -0.108713], [0.071911, -0.384547, 0.205031, -0.29678, 0.11792, -0.208216, 0.177188, -0.185863, -0.213176, -0.048948, 0.089738, -0.133856, -0.128951, -0.003966, -0.423744, 0.076149, -0.497793, -0.163566, -0.043271, 0.141289, -0.573045, -1.062598, -0.143751, -0.260687, 0.356591, -0.389376, -0.831004, -0.456638, 0.285093, -0.486354, -0.528206, -0.191259, -0.138183, 0.151469, -0.166262, 0.163581, -0.11523, -0.041938, 0.12195, 0.025584, -0.077602, -0.144258, 0.209773, 0.149114, -0.089564, -0.281561, 0.15456, -0.150926], [-0.761535, 0.43966, -0.304465, 0.094118, 0.309253, -0.396212, 0.35981, -0.111574, -0.021568, -0.068892, 0.066425, -0.01284, -0.361853, 0.175616, 0.099835, 0.365602, 0.047576, 0.28845, -0.232514, 0.080658, 0.212832, -0.246024, 0.673686, 0.106229, -0.482694, -0.114362, 0.008683, -0.158406, 0.427534, 0.237681, -0.019494, -0.120015, 0.005017, -0.437479, 0.59677, 0.247181, 0.218619, -0.662072, -0.522064, 0.497996, -0.219148, -0.198939, 0.369369, -0.520111, 0.246439, -0.105751, -0.057485, -0.564859], [0.14538, 0.262287, 0.03232, 0.037984, 0.18217, -0.140542, -0.242226, -0.014633, 0.197467, 0.036942, -0.552303, 0.057976, -0.298293, -0.049016, -0.339019, 0.285697, 0.090582, -0.351532, -0.413372, 0.006371, 0.054354, 0.20545, -0.106476, -0.183654, -0.132039, -0.334268, 0.096575, -0.408288, 0.226059, -0.098205, 0.051593, -0.622622, 0.022067, 0.185417, -0.023193, 0.198883, -0.530471, -0.04741, 0.108546, -0.009263, -0.121224, 0.094953, -0.301249, -0.043241, -0.038805, -0.025116, -0.039626, -0.252328], [-0.258354, -0.095656, -0.339758, 0.150294, 0.187336, -0.153382, -0.03262, 0.113683, -0.23663, 0.125477, -0.08718, 0.048754, 0.029047, 0.242339, 0.59817, -0.371925, 0.462863, 0.620443, 0.450796, 0.125816, 0.364585, 0.669268, 0.098887, 0.354513, 0.548075, 0.420475, -0.126787, 0.485223, 0.021236, 0.823447, 0.463177, 0.397826, -0.058493, -0.102176, 0.012331, -0.329062, 0.607525, -0.399769, -0.39586, -0.104515, -0.22949, 0.445734, -0.002535, -0.005082, 0.141988, 0.548697, 0.475518, -0.129561], [0.839875, 0.315412, 0.126229, 0.101846, -0.228781, 0.710131, -0.394903, -0.235683, 0.13032, 0.134765, -0.185098, 0.070633, 0.416751, -0.056671, 0.035618, -0.785022, 0.052588, 0.296262, 0.149907, -0.364974, 0.386165, -0.401549, -0.325343, -0.144435, -0.162452, -0.123193, 0.539948, 0.260096, 0.14368, 0.1866, -0.45725, 0.23923, 0.525424, -0.08829, -0.337239, -0.191249, 0.120851, 0.873307, 0.323031, 0.169284, 0.404673, 0.140068, -0.020217, -0.334888, -0.168654, 0.373775, 0.302461, 0.187146], [0.224055, -0.466121, 0.019206, -0.151644, -0.047061, 0.293224, -0.431369, -0.49581, -0.295137, -0.022656, 0.128927, 0.195217, -0.230538, -0.240529, -0.214731, -0.02162, -0.2547, -0.305417, -0.191797, -0.019185, -0.251521, -0.91722, -0.452201, -0.330884, -0.057264, -0.627542, -0.21052, -0.065192, -0.30425, 0.051985, -0.384709, -0.09159, 0.132561, 0.441005, 0.151535, 0.088114, -0.141509, 0.352926, -0.021875, -0.210623, 0.081138, -0.390328, 0.032541, -0.288127, 0.132255, -0.117505, 0.069914, -0.162712], [-0.08457, -0.077333, 0.02229, 0.030156, 0.292062, -0.020773, 0.200864, -0.102945, -0.017832, 0.204783, 0.368278, -0.104972, 0.245307, 0.00334, 0.165378, -0.310625, 0.049463, 0.148923, 0.276246, 0.566549, 0.244906, 0.353455, 0.437054, 0.296139, 0.737578, 0.464736, 0.110384, 0.29108, 0.049247, 0.319495, 0.473912, -0.028155, 0.092711, -0.039676, 0.271289, 0.193981, 0.256744, -0.090578, -0.202342, 0.138268, -0.30005, 0.106381, 0.207081, -0.016097, 0.114336, -0.09911, -0.005103, -0.14237], [0.07775, 0.172586, 0.177324, -0.298583, -0.17535, 0.196581, -0.142726, 0.124774, -0.14833, 0.009161, -0.194217, -0.452377, -0.295252, -0.31355, 0.051986, -0.297168, -0.032714, 0.113412, 0.438302, -0.043502, 0.168561, 0.340278, -0.044111, -0.154493, 0.212121, 0.164058, 0.189921, -0.405518, -0.26354, 0.399486, -0.227639, -0.08249, 0.156734, 0.055076, 0.062948, 0.136512, -0.251598, 0.331132, 0.457791, -0.093672, 0.18965, 0.089922, -0.324256, 0.138558, 0.240942, 0.27426, 0.09357, 0.2005], [0.012752, 0.010715, -0.35376, -0.014107, 0.026435, 0.104039, -0.2652, -0.167164, -0.222989, 0.423257, 0.158975, -0.141626, 0.084702, -0.030307, 0.132431, -0.273743, 0.207169, 0.250078, 0.69399, 0.325712, -0.097571, 0.26644, 0.029708, 0.119979, 0.430837, -0.038152, -0.165536, 0.295231, -0.346201, 0.374373, 0.070013, -0.030789, 0.064772, 0.023291, -0.388865, 0.065466, 0.704014, 0.155149, 0.149654, 0.035932, -0.201927, 0.669613, 0.096248, 0.05685, -0.032764, 0.283792, 0.25807, 0.412732], [-0.251683, 0.061054, 0.222738, 0.197677, 0.149568, 0.107685, 0.183402, 0.060739, 0.091568, 0.028366, 0.575608, 0.026328, -0.091896, 0.297565, 0.322255, 0.036162, 1.185154, 0.779274, 0.255662, 0.620073, 0.826643, 0.728629, 0.619421, 0.248386, 0.581413, 0.624994, 0.558604, 0.303644, 0.365685, 0.455531, 0.481775, 0.296302, 0.263417, -0.402717, 0.356358, 0.000352, 0.177292, -0.24789, -0.138316, 0.090686, -0.103559, -0.011504, 0.441505, -0.227634, -0.126268, -0.105216, 0.344256, -0.047284], [-0.554745, -0.13631, 0.206034, -0.086868, -0.073771, -0.708547, 0.280809, 0.671889, 0.732135, -0.370898, 0.298328, -0.395335, -0.114365, -0.15077, 0.146557, 0.174769, 0.65244, 0.4583, 0.252414, -0.239355, 0.506204, 0.38852, -0.620372, 0.142942, 0.291371, 0.346651, -0.125617, 0.654986, -0.02443, 0.196251, -0.5715, -0.170707, -0.514802, 0.377887, -0.053199, -0.139928, 0.226782, -0.480319, 0.044338, -0.463381, 0.579383, -0.305304, 0.714743, 0.316524, -0.437176, 0.109696, 0.435121, 0.461475], [-0.465073, 0.304627, -0.231472, 0.022088, 0.081874, -0.458924, 0.303463, 0.05956, -0.257584, -0.106543, 0.089324, 0.105796, -0.384435, -0.061381, -0.312104, -0.073388, -0.592095, 0.005919, -0.563795, -0.143629, -0.779412, -0.489553, -0.288585, -0.268827, -0.523389, -0.293974, -0.251021, -0.480677, -0.062291, -0.16388, 0.441246, -0.848734, -0.516511, -0.31152, 0.2154, -0.082985, -0.347598, -0.055501, -0.176029, -0.237003, -0.068117, 0.356178, -0.383879, -0.111113, 0.725524, 0.120546, -0.000659, -0.009202], [-0.475736, -0.019726, -0.499263, 0.346723, -0.161105, -0.036964, -0.351033, -0.10545, -0.314475, 0.444826, -0.1481, -0.016291, -0.41105, 0.28527, 0.105596, 0.035689, 0.387864, 0.795638, 0.525549, -0.091691, 0.522515, -0.454172, 0.355953, 0.269163, 0.221642, -0.022227, 0.47765, 1.041002, -0.186685, -0.004062, -0.033483, 0.142617, -0.000693, -0.246591, -0.211544, -0.336481, 0.275819, -0.275627, -0.086817, -0.182392, 0.147312, 0.066911, -0.184222, 0.05143, 0.26871, 0.543042, -0.272243, 0.248796], [0.164369, -0.120321, 0.363147, -0.225521, 0.434752, 0.11715, 0.12465, 0.312049, 0.324921, -0.045735, 0.245646, 0.167832, 0.226355, -0.19978, 0.753386, -0.08181, 0.341331, 0.418514, 0.572144, 0.568229, 0.791738, 0.576809, 0.210962, 0.413969, 0.726517, 0.520479, -0.105628, 0.612351, -0.280463, 1.044456, 0.180356, 0.341495, 0.130152, -0.143824, 0.276726, -0.172755, 0.501668, 0.068872, 0.166455, 0.068074, -0.034645, 0.352208, 0.399189, -0.134831, -0.178314, 0.052042, 0.811379, 0.026429], [-0.448999, 0.089141, -0.213067, 0.237829, -0.393913, -0.630376, 0.320297, -0.020398, 0.223749, -0.704375, -0.097689, -0.455556, -0.430111, -0.235862, -0.018855, 0.117756, 0.401892, 0.008939, 0.558266, -0.051464, -0.331871, 0.950991, -0.611866, 0.065132, -0.308369, -0.395479, 0.483031, 0.244236, -0.652813, 0.256418, 0.370775, -0.368756, -0.142322, 0.086542, -0.108767, 0.047511, -0.023001, -0.207604, 0.190217, -0.203479, 0.017412, 0.1144, 0.185017, 0.278657, 0.639762, 0.315619, -0.173622, 0.328794]]
SEQ_BH = [0.023063, 0.228264, 0.018757, 0.190219, 0.235037, 0.241342, 0.231536, 0.06584, -0.267252, 0.186941, 0.313446, -0.133739, 0.241268, 0.252708, 0.199666, -0.130431, 0.847369, 0.358344, 0.658885, 0.542539, 0.753104, 0.924166, 0.636472, 0.556941, 0.549688, 0.621105, 0.300743, 0.135967, 0.006806, 0.331779, 0.230175, 0.578735, 0.127118, -0.349277, -0.001117, -0.192835, -0.107401, 0.214987, -0.281844, 0.256818, 0.279548, 0.131822, 0.479537, -0.186838, -0.416683, 0.049188, 0.437556, 0.029971]
SEQ_HEAD = [
    ([[0.009977, -0.166405, -0.079154, 0.070164, -0.221024, 0.042473, -0.039259, 0.296094, -0.04969, -0.090223, 0.037416, -0.093834, 0.074344, -0.216307, -0.283711, -0.121138, -0.28584, 0.102459, 0.074229, 0.179008, 0.240427, -0.075792, -0.000948, -0.0729, 0.046357, -0.016885, -0.028891, 0.134448, -0.117236, -0.015844, -0.252157, 0.003861, 0.026184, -0.129485, -0.037217, -0.102331, -0.149524, 0.108126, 0.284039, -0.106136, -0.079765, 0.005892, -0.236083, -0.110722, 0.260034, -0.032179, 0.246342, 0.05958, 0.127753, 0.147584, 0.076972, -0.098854, -0.079361, 0.016707, 0.113793, -0.185642, 0.051566, 0.073294, 0.106391, -0.2276, -0.153718, 0.009546, -0.190015, -0.169302], [-0.000492, 0.120457, 0.216388, 0.143316, -0.005151, -0.072338, 0.050781, 0.224994, 0.091633, 0.004199, -0.038727, -0.140741, 0.10677, -0.068075, -0.057679, 0.047061, -0.164856, 0.089641, -0.043602, -6e-06, -0.08986, -0.081099, 0.131981, 0.026589, -0.096743, -0.14086, 0.207612, -0.084544, 0.024844, -0.19609, -0.120083, 0.074638, 0.10272, 0.041439, -0.046569, 0.148972, 0.061828, -0.064854, -0.0524, -0.106011, 0.082373, 0.026799, -0.018419, 0.132979, -0.203992, 0.040322, -0.041571, -0.123406, 0.012701, 0.189425, 0.051446, 0.035937, -0.027886, -0.160244, 0.188266, -0.13441, -0.07939, -0.078667, -0.104259, 0.201628, 0.062867, -0.065012, -0.111344, -0.140244], [0.163606, -0.07554, 0.050305, 0.125976, 0.108946, -0.051586, 0.020422, 0.034581, -0.023131, 0.008875, -0.245149, -0.149478, -0.05542, 0.108993, 0.077375, -0.10797, 0.049762, 0.22833, -0.023101, 0.177603, -0.016632, -0.029015, -0.08612, -0.163525, 0.018716, -0.159736, -0.02009, -0.001962, 0.043768, -0.203035, 0.259409, 0.085385, 0.084688, 0.058916, -0.046212, -0.134626, 0.100093, 0.083381, 0.101662, 0.006063, 0.052832, 0.174486, -0.158932, -0.01616, -0.227812, -0.069301, -0.12012, 0.00174, -0.100973, 0.1342, -0.099631, -0.099308, 0.063509, -0.105304, 0.130749, -0.060384, -0.109386, -0.104563, -0.022113, 0.014363, 0.098351, 0.099059, -0.072088, 0.128921], [-0.038485, -0.189106, 0.194128, -0.090472, 0.208204, 0.044611, 0.000785, -0.102946, -0.022868, -0.057353, -0.153572, 0.107823, -0.240518, 0.138874, -0.06427, -0.165593, 0.314776, 0.074572, 0.034102, -0.103677, 0.063608, 0.037628, 0.024207, -0.10136, -0.137968, 0.044634, -0.045538, -0.068389, 0.025274, 0.106978, 0.151023, 0.035933, -0.115277, 0.163113, -0.033811, -0.118218, -0.195767, 0.044515, 0.021705, -0.172279, -0.028308, 0.130583, 0.02284, -0.046565, -0.295218, 0.142146, -0.170228, -0.133437, -0.178233, -0.045843, 0.083675, -0.013802, -0.172692, -0.204642, -0.295922, -0.027203, 0.092953, 0.047812, -0.010391, -0.189553, -0.136423, 0.073965, 0.135587, 0.04737], [0.001961, 0.045021, -0.032677, -0.033181, 0.076225, -0.106083, 0.028037, -0.050209, -0.004551, 0.136147, 0.050973, 0.170346, -0.227714, 0.21396, 0.064327, -0.007673, 0.228718, 0.184241, -0.038239, 0.017804, 0.065578, -0.11552, -0.006368, 0.061972, 0.149298, -0.199305, 0.017037, -0.025094, 0.098893, -0.004214, 0.258382, 0.032281, -0.086943, 0.106946, -0.056633, 0.110717, -0.230673, 0.033579, -0.066315, -0.128021, -0.102135, 0.095471, 0.036483, -0.208743, 0.014845, 0.023842, 0.016889, -0.186103, -0.198864, -0.091994, 0.024609, 0.131172, -0.151158, -0.18005, -0.142916, 0.125144, 0.141393, -0.239182, 0.022696, -0.14015, 0.119595, -0.144192, -0.03472, 0.253846], [-0.177656, -0.002996, 0.065046, -0.067417, -0.016913, -0.107354, 0.07811, 0.021764, -0.001742, 0.204387, -0.134329, -0.059076, -0.268297, -0.064207, 0.219644, -0.009278, 0.254004, -0.125294, -0.054523, 0.061462, -0.180829, -0.297629, -0.063612, 0.134869, -0.105104, 0.080819, 0.094983, -0.004956, -0.111528, -0.172917, 0.114993, 0.064085, -0.156047, 0.01486, 0.048724, -0.032104, -0.056056, 0.108467, -0.239966, 0.159446, 0.020682, 0.081044, 0.203997, -0.04697, -0.164835, 0.031984, -0.046936, -0.051747, -0.021221, -0.031356, -0.001507, 0.012124, -0.041946, 0.124422, 0.034118, 0.043855, -0.137494, -0.013228, -0.014153, 0.071661, 0.026716, 0.098132, -0.035119, -0.134696], [-0.099822, 0.125448, -0.138982, -0.020651, 0.200711, -0.166649, 0.139487, -0.116353, -0.149025, -0.058425, -0.080373, 0.109851, 0.021264, -0.196917, -0.262838, 0.126704, 0.049096, -0.035248, 0.24919, 0.167555, -0.212927, -0.212177, 0.195805, 0.368299, -0.164549, -0.086704, -0.073287, -0.133293, -0.019883, -0.155981, 0.131487, 0.088941, -0.157054, -0.01474, -0.172586, 0.160635, 0.003385, 0.175342, -0.116288, -0.075853, -0.020246, 0.142071, -0.044347, 0.239348, -0.142998, 0.103463, 0.590683, -0.069677, -0.065779, 0.161205, 0.032912, -0.213495, -0.295123, 0.044859, -0.239841, -0.038187, -0.109816, -0.178027, 0.066972, -0.138818, -0.171048, -0.016881, -0.044371, 0.022786], [-0.009554, -0.124076, 0.078577, 0.009054, 0.036172, 0.010826, -0.027517, -0.09998, -0.093788, 0.0075, 0.006397, -0.17407, -0.151728, -0.123094, -0.033685, -0.029088, 0.137007, -0.079022, 0.09162, -0.109009, -0.122795, -0.122543, -0.098066, 0.005614, 0.104663, 0.10496, 0.086786, -0.113624, 0.048946, -0.00168, 0.06731, -0.04483, -0.001954, 0.004043, -0.006693, -0.060147, -0.102494, -0.151253, 0.118604, -0.006938, 0.198265, -0.000451, 0.059806, -0.066112, -0.107087, 0.16089, -0.12676, -0.038118, 0.008562, 0.082451, -0.063207, 0.008874, -0.069926, -0.01494, 0.124995, 0.047885, -0.039606, 0.009992, -0.072781, 0.033209, 0.158955, -0.135457, -0.13061, -0.04043], [0.151933, 0.042539, -0.014298, 0.084558, 0.244421, 0.130191, 0.157604, 0.109976, 0.00403, 0.183161, -0.187028, -0.063415, 0.210328, 0.062993, 0.032853, -0.106532, 0.01708, 0.179838, 0.07304, 0.030552, -0.003305, 0.079404, 0.357417, 0.096632, -0.011271, -0.243012, 0.044425, 0.001694, 0.079697, -0.304864, 0.054917, -0.167502, -0.049764, 0.010844, 0.170378, -0.04519, -0.071521, -0.133857, 0.100907, -0.149293, 0.039328, -0.18299, -0.355062, -0.039785, 0.078041, -0.042461, 0.061008, 0.015812, -0.091418, 0.203578, -0.09761, -0.179544, -0.031321, -0.059481, -0.120659, -0.031813, 0.113575, 0.136497, -0.078564, -0.121032, 0.249467, 0.078075, 0.001468, -0.088258], [-0.047969, -0.06601, 0.256296, -0.013525, 0.038384, -0.094124, 0.129514, -0.104073, -0.014308, 0.005488, -0.213375, 0.079582, -0.154168, -0.011589, 0.007857, -0.249874, -0.267786, -0.41858, 0.111062, -0.185336, -0.150942, -0.28086, -0.092419, 0.007459, 0.080103, 0.032358, -0.070843, 0.169422, -0.10102, -0.224287, 0.080685, 0.003861, -0.383832, 0.017699, -0.016118, 0.043594, -0.104436, 0.09317, 0.023214, -0.052938, -0.197908, -0.207464, 0.086165, -0.089713, 0.202215, -0.00788, -0.136593, 0.073763, -0.04261, -0.00416, 0.087099, 0.043936, 0.023593, 0.096974, 0.173142, -0.185386, 0.079763, 0.180701, -0.14667, 0.217364, 0.155916, 0.026238, -0.055901, -0.16859], [-0.123741, 0.10979, -0.0186, 0.168741, 0.101151, -0.023351, -0.179643, -0.025361, -0.087986, -0.096028, -0.062362, -0.084282, 0.188195, -0.107287, -0.166062, 0.129599, -0.087849, 0.093023, -0.086353, -0.016739, 0.095734, 0.004333, -0.162181, 0.069211, -0.024743, 0.108016, 0.018341, 0.086436, 0.153066, -0.023001, -0.17051, -0.068149, 0.148606, 0.135067, 0.072922, -0.162638, -0.105235, -0.129589, 0.083378, 0.02366, -0.164569, 0.020271, -0.009717, 0.024347, -0.014324, -0.004363, 0.099651, -0.045789, 0.177363, 0.015623, -0.121487, -0.083853, -0.043526, 0.223566, 0.028119, 0.051, 0.208698, 0.11165, -0.095928, 0.286042, 0.067328, 0.022714, -0.129077, -0.006991], [0.207458, 0.098308, 0.138732, 0.098906, 0.024246, 0.270766, 0.315145, -0.203387, -0.108437, -0.001663, -0.288653, -0.039197, 0.135671, 0.088649, 0.161081, -0.345648, -0.282947, 0.207205, 0.081834, 0.42113, 0.100558, 0.203389, 0.074568, 0.0652, -0.035537, -0.123157, -0.107426, -0.018639, 0.210091, -0.104934, 0.297502, 0.039345, 0.323347, 0.103793, 0.132962, 0.167775, -0.232101, 0.010596, 0.087171, -0.258279, 0.271696, -0.127126, 0.162715, -0.032335, -0.347275, 0.145885, -0.348254, -0.196812, -0.089752, 0.243419, -0.107313, -0.320383, 0.094083, -0.355404, 0.056782, -0.344917, -0.030597, -0.095578, -0.227197, -0.078112, -0.081884, 0.129958, -0.224512, -0.020346], [0.035599, -0.164166, 0.043198, -0.157075, -0.089543, 0.054206, -0.161143, 0.014938, -0.032083, 0.198014, -0.0948, -0.204141, -0.023297, 0.080115, 0.218049, -0.12206, 0.131255, -0.231625, -0.002047, 0.06508, 0.275543, 0.292036, 0.057053, 0.1265, -0.059198, 0.213463, -0.157776, 0.009383, -0.119419, -0.13526, 0.022943, -0.326256, 0.204629, -0.062593, 0.200421, 0.131917, 0.061484, -0.008731, 0.078708, -0.02561, -0.11378, 0.187777, -0.059201, 0.213861, -0.081355, -0.08428, -0.090965, 0.139889, -0.1771, 0.007976, -0.085878, 0.078645, 0.088996, -0.04962, -0.149445, 0.083539, -0.22294, -0.111629, 0.055257, 0.380875, -0.066961, -0.20069, 0.053513, -0.221533], [0.031807, -0.109444, -0.019279, -0.035518, -0.056395, 0.046563, -0.120038, -0.012541, -0.131862, -0.151121, -0.150141, 0.005073, 0.107666, -0.133488, 0.131012, -0.057094, -0.073072, -0.120293, -0.086215, -0.020603, -0.176668, 0.228512, 0.087384, 0.11332, 0.120686, -0.036647, 0.04272, -0.27789, -0.143075, 0.066464, 0.047453, -0.083209, -0.088851, 0.036264, 0.025161, -0.086789, 0.179317, -0.060372, 0.107485, 0.112629, -0.057056, 0.242534, -0.004396, -0.118203, 0.07629, 0.036412, 0.0167, -0.172948, -0.232676, 0.12274, -0.122973, -0.201709, 0.0174, 0.110758, -0.088513, 0.005294, -0.01802, -0.172414, 0.365628, 0.001074, 0.230336, -0.165655, -0.02151, 0.123486], [-0.112589, 0.031077, 0.089601, -0.040702, -0.159329, -0.055335, -0.20572, -0.040247, 0.068955, 0.145235, 0.039728, -0.177328, -0.040905, -0.062487, -0.054902, 0.02483, 0.06744, -0.02879, -0.060532, -0.101962, -0.088297, 0.085383, 0.009494, -0.176156, -0.240154, -0.097979, 0.01758, -0.040869, -0.120248, 0.026701, -0.106702, -0.229585, -0.005263, 0.255576, 0.008606, 0.245718, 0.119411, -0.094874, 0.012936, -0.107026, 0.048266, -0.277463, -0.007345, -0.106353, -0.073803, -0.129783, -0.025734, 0.080008, -0.141685, 0.083543, 0.082214, 0.164675, 0.001382, 0.004654, 0.240767, -0.144721, 0.233409, 0.212688, 0.207936, 0.04432, 0.066309, 0.114748, -0.026184, -0.143575], [-0.026105, 0.055675, -0.11868, 0.012527, -0.002666, -0.071878, 0.14308, -0.011449, 0.040042, 0.009032, -0.07928, 0.233841, -0.054572, 0.104662, -0.246659, -0.170057, -0.118237, -0.037455, 0.095955, 0.083011, 0.173271, 0.045327, 0.079533, -0.081453, 0.099208, -0.01012, -0.000924, 0.066238, -0.040388, -0.141838, 0.148332, -0.03364, -0.063611, -0.141774, -0.046823, -0.219502, -0.138449, -0.013515, 0.067523, 0.135676, 0.140201, -0.178903, 0.056652, -0.317465, 0.248316, 0.089177, 0.001352, 0.102425, -0.139147, 0.072524, -0.153553, 0.084343, -0.021831, -0.129588, -0.255887, 0.044814, -0.037595, -0.139336, -0.067127, -0.063687, 0.038769, -0.002196, 0.06242, 0.109751], [-0.054618, 0.193672, 0.154384, -0.005982, 0.011572, 0.002095, 0.07207, -0.154723, -0.142012, -0.187278, 0.066489, -0.001584, -0.087459, 0.270574, -0.194466, 0.180408, 0.20329, -0.143635, 0.163616, 0.096768, -0.316356, 0.066393, -0.100232, -0.019256, 0.30085, -0.196339, 0.055158, -0.015432, 0.211911, 0.082511, -0.029518, -0.175381, -0.057324, -0.163257, -0.156763, -0.188749, 0.074685, 0.243761, 0.140211, -0.096913, 0.057194, -0.145197, 0.129933, 0.337557, 0.11478, 0.279431, 0.103456, 0.036309, -0.146205, 0.03336, -0.193992, 0.207521, 0.216031, 0.013137, -0.141482, 0.015226, -0.147834, -0.155877, -0.245099, -0.245678, 0.150572, -0.173975, -0.185989, -0.140224], [-0.086475, -0.168805, 0.301266, -0.171639, -0.374781, 0.059991, -0.086679, -0.282818, 0.077776, -0.119716, -0.083256, -0.111072, 0.122805, -0.149739, -0.053953, -0.126831, -0.091796, 0.037531, -0.051422, 0.202325, -0.055874, 0.196633, 0.171521, -0.013747, 0.116291, -0.101356, 0.049511, 0.102343, -0.015306, -0.144716, 0.08234, -0.220582, 0.031669, 0.099347, -0.051741, -0.027004, 0.007312, -0.098413, -0.119693, -0.122094, 0.208071, -0.149383, -0.069978, -0.081193, -0.06585, 0.037498, -0.251354, 0.327056, -0.077229, 0.106907, 0.091238, 0.146053, -0.129619, 0.292098, 0.181207, -0.022422, 0.096108, -0.010981, -0.172296, -0.101166, 0.02285, -0.034072, -0.007543, 0.163595], [0.115184, 0.047857, 0.080223, -0.140296, 0.032093, -0.135063, 0.175347, -0.065633, -0.02443, 0.080746, -0.022972, 0.126066, -0.035708, 0.153901, -0.016117, 0.152268, -0.032842, -0.077215, 0.018801, 0.064597, 0.057158, -0.164215, -0.098333, 0.044232, 0.044068, 0.049422, 0.035752, 0.061197, 0.069115, 0.0982, -0.101486, 0.049869, -0.000943, -0.122953, -0.153831, 0.011243, -0.009119, -0.073041, 0.054267, 0.095616, 0.13782, 0.140834, 0.069529, -0.01678, 0.005806, 0.019248, -0.117464, 0.001282, -0.076598, -0.131406, -0.092933, 0.054175, -0.111762, 0.032138, -0.21158, 0.096481, -0.071515, 0.01339, -0.056814, -0.006074, -0.003392, 0.118458, -0.037504, 0.001851], [0.408704, 0.2863, -0.090783, 0.100697, -0.071523, -0.228506, 0.249325, 0.029302, -0.338896, 0.32164, -0.02273, 0.330123, 0.226693, 0.001971, -0.239711, -0.038954, -0.232771, 0.073431, 0.001239, 0.094236, -0.080307, -0.045955, -0.38957, 0.40122, -0.193149, -0.147702, -0.090482, -0.264105, 0.2695, 0.175329, -0.120014, -0.031802, 0.047868, -0.346014, 0.020377, -0.129903, -0.152018, 0.193134, 0.223267, 0.042532, -0.050064, 0.175159, -0.256069, 0.411557, 0.106365, -0.062392, 0.028063, 0.010041, -0.235241, 0.083, -0.047989, -0.083245, -0.032738, -0.037945, -0.124709, -0.156099, -0.160117, 0.172177, -0.363618, 0.013745, 0.068174, 0.354302, -0.102557, -0.057068], [0.172752, 0.089972, -0.003049, 0.340757, -0.003567, 0.112544, -0.011873, -0.082053, -0.170175, 0.339446, 0.055868, -0.060527, 0.064552, 0.24353, 0.018584, -0.075514, 0.03749, -0.009596, -0.164205, -0.031066, -0.068841, -0.104716, -0.222681, -0.186636, -0.047415, -0.153974, 0.151279, -0.087003, 0.025101, -0.019048, 0.037525, -0.194172, 0.118925, 0.088522, 0.124635, 0.087304, -0.176168, 0.100331, -0.097714, -0.104912, 0.024776, 0.025941, -0.053067, 0.062511, 0.120528, -0.172507, -0.179559, -0.020166, 0.015341, -0.026061, -0.020173, 0.227755, 0.140506, 0.004976, -0.096313, 0.165958, 0.111799, 0.216326, 0.087434, 0.114733, 0.06254, -0.053942, -0.03651, -0.148311], [-0.243787, 0.081754, -0.027332, 0.152367, 0.230808, -0.125016, -0.121872, -0.088227, -0.074823, 0.237588, -0.02202, -0.110228, -0.03206, 0.074952, 0.036425, 0.068739, -0.011004, 0.023537, -0.041357, -0.003238, -0.298018, -0.052907, 0.144416, 0.101328, -0.378242, -0.17734, 0.11094, 0.01019, -0.041993, 0.149518, 0.05803, 0.094522, 0.049091, -0.309277, 0.00902, -0.093392, 0.218702, -0.157005, -0.018062, 0.007211, 0.075138, -0.106694, 0.216612, -0.1022, -0.026943, 0.135421, -0.154757, 0.053956, 0.03881, 0.103087, -0.169684, 0.312247, 0.277039, -0.058052, 0.154815, 0.198782, -0.152141, 0.012048, -0.073683, -0.05321, -0.177544, 0.048556, -0.040644, 0.102958], [-0.404741, -0.161341, -0.07144, -0.335886, -0.016361, -0.008507, -0.103886, -0.256888, -0.08037, -0.02174, 0.09421, -0.456119, -0.505195, -0.304855, -0.286298, -0.043645, -0.167459, -0.031073, -0.40711, -0.100676, 0.034509, -0.021246, 0.213127, -0.339981, 0.251401, 0.291967, 0.223042, 0.294166, -0.174524, 0.283636, 0.085763, -0.405323, 0.009446, -0.270135, -0.279945, -0.035624, -0.226213, -0.174027, -0.402007, -0.075736, -0.153579, 0.428319, -0.37622, 0.28455, -0.255376, 0.270898, 0.136298, 0.13184, -0.312873, -0.380024, 0.287524, -0.156295, -0.121059, -0.056433, -0.076296, 0.018, -0.105671, 0.151861, 0.146494, -0.281389, -0.14651, -0.226943, 0.011704, 0.062852], [-0.057354, 0.287138, 0.408768, 0.187972, -0.073035, -0.00346, 0.097147, 0.156414, 0.132895, 0.324434, -0.021704, 0.248463, 0.151109, 0.346636, 0.338759, 0.071998, 0.021715, 0.114971, 0.344099, 0.31509, 0.159565, 0.261829, -0.413053, 0.088841, 0.202758, 0.170941, -0.033148, 0.13216, 0.19746, -0.076046, 0.279179, -0.017424, 0.031119, 0.185459, 0.300222, 0.359019, 0.006425, 0.137506, 0.34414, 0.049299, 0.064119, 0.130074, 0.04437, 0.059995, 0.029469, -0.182468, -0.118585, 0.032916, 0.149001, 0.198836, 0.093949, -0.072607, 0.145549, 0.00469, 0.314818, -0.057442, 0.112974, -0.138671, -0.107629, 0.221045, 0.125733, -0.007574, 0.142451, -0.037728], [-0.040744, -0.221378, -0.49616, 0.133306, -0.010043, -0.015291, -0.157187, -0.021749, 0.103428, 0.153583, 0.009471, -0.056459, 0.09455, -0.108582, -0.105389, 0.115593, 0.062282, -0.050187, -0.235194, -0.126557, 0.088046, -0.278728, -0.203283, 0.061695, -0.19831, -0.074178, 0.453663, 0.252668, 0.103027, -0.060313, -0.111355, 0.018894, -0.114439, -0.017838, 0.06355, 0.036111, -0.028591, -0.237918, -0.250726, 0.003724, 0.205396, 0.10113, 0.000104, 0.037356, -0.117277, 0.387638, 0.02337, 0.129587, 0.212165, -0.392146, 0.264478, 0.073076, 0.01523, 0.099314, 0.249666, 0.044884, 0.274463, 0.459944, 0.008732, 0.024041, -0.132239, 0.003708, 0.054259, -0.035157], [-0.103305, -0.145504, -0.062209, -0.117807, -0.036733, -0.226538, -0.126274, 0.194508, -0.027169, -0.114678, 0.248246, -0.228867, -0.054221, 0.146221, 0.198124, 0.346037, -0.020923, 0.142594, -0.084206, 0.249071, -0.000496, -0.297977, -0.269234, -0.055154, -0.143584, 0.282223, 0.132386, 0.107107, 0.230043, 0.24243, 0.262562, -0.043882, -0.041115, 0.057924, 0.208421, 0.361681, 0.119401, -0.055223, 0.020359, 0.284556, 0.046071, 0.410863, 0.107635, 0.072515, 0.062117, -0.069076, -0.031681, 0.284208, -0.240406, 0.313188, 0.24513, 0.158456, -0.572554, 0.11523, 0.338889, 0.417509, 0.17883, 0.040738, 0.174787, -0.088887, 0.093761, -0.047709, 0.270623, 0.195472], [0.087681, 0.095124, -0.069354, -0.185239, -0.150433, 0.351905, 0.172576, -0.135154, -0.063088, 0.067271, -0.470459, 0.037386, 0.164782, -0.022896, -0.061648, -0.320792, -0.097177, -0.056552, 0.088252, -0.147001, 0.183425, 0.082889, 0.060941, 0.062538, -0.171159, -0.076797, 0.025115, -0.076211, -0.069782, 0.015874, -0.24292, -0.178285, 0.082161, -0.330717, -0.040291, -0.097693, -0.275748, -0.025575, -0.219554, -0.335899, 0.005713, -0.309423, 0.00669, -0.153321, -0.178474, -0.138485, -0.314341, -0.200534, -0.176512, -0.076354, -0.180292, -0.505495, -0.045874, -0.308715, -0.290826, -0.374546, -0.248822, -0.30699, 0.19221, -0.02818, -0.068557, -0.206354, -0.347728, -0.233284], [0.021034, -0.03697, 0.108353, -0.049462, -0.084068, 0.195658, 0.027618, -0.439779, -0.168636, -0.080541, 0.073445, -0.048239, -0.644204, -0.013587, -0.395628, 0.216932, -0.27138, 0.087398, -0.485559, -0.186296, 0.012751, 0.137644, 0.165755, -0.376906, 0.171801, -0.016831, -0.147017, -0.060342, -0.144361, 0.448733, 0.259411, 0.201658, 0.079309, -0.404741, -0.345492, -0.182824, -0.253189, 0.04813, -0.065533, -0.003655, -0.546933, 0.367029, -0.179963, 0.38823, -0.295622, 0.144787, 0.436331, -0.22941, -0.154587, -0.089173, 0.024846, -0.101639, -0.005343, -0.068258, -0.23289, -0.102538, -0.174282, 0.245298, -0.167588, -0.299769, 0.145438, -0.148702, -0.029732, -0.090002], [-0.026554, -0.070543, -0.012223, 0.080233, 0.026176, -0.14768, 0.025873, -0.108559, -0.369427, 0.252809, -0.077162, -0.063216, 0.021161, 0.171319, 0.000616, 0.051153, -0.085376, -0.106, -0.216283, 0.280427, 0.135448, 0.112273, -0.198168, 0.206023, 0.111966, 0.19319, -0.187598, -0.047291, 0.091582, 0.325419, 0.152922, -0.003642, 0.100572, -0.184274, 0.327251, 0.185169, 0.155847, 0.066575, -0.083752, -0.074588, -0.38767, 0.17118, -0.016931, 0.098378, -0.026438, -0.305482, -0.032508, 0.04071, -0.036148, 0.137419, -0.084347, -0.045287, -0.007536, -0.013718, -0.023667, 0.105564, -0.009508, 0.085528, 0.067414, 0.22348, 0.11317, 0.153855, 0.008523, -0.007029], [-0.04175, 0.024806, -0.126959, -0.23666, -0.23064, -0.332976, -0.149252, -0.054212, 0.031043, 0.089158, 0.0206, -0.261478, -0.178499, -0.217023, 0.111754, -0.096733, -0.029359, -0.21082, -0.185639, -0.263321, -0.232628, -0.476154, -0.297145, 0.036635, -0.039584, 0.378783, 0.381957, 0.33849, -0.126972, -0.18291, -0.30132, -0.136622, -0.179926, 0.034753, 0.056776, 0.171622, -0.078028, -0.024249, -0.271048, -0.143434, -0.226507, -0.095472, 0.014882, 0.108124, -0.095187, -0.098628, -0.078687, 0.049517, -0.047602, -0.411407, 0.325829, -0.283251, -0.261413, 0.072744, 0.078247, -0.182771, 0.112387, 0.204159, 0.213941, -0.008663, -0.079291, -0.109608, -0.113577, 0.146857], [0.191323, -0.042859, 0.058342, -0.298577, 0.051068, 0.214112, 0.22667, -0.104189, -0.061753, 0.145202, -0.032245, 0.037434, 0.043192, 0.150868, -0.098794, 0.184355, -0.085601, 0.082076, 0.004222, 0.049482, 0.006431, 0.018042, -0.181817, -0.02724, 0.384377, 0.02442, -0.083358, -0.14558, -0.074296, 0.339343, 0.084561, -0.046491, 0.023687, -0.164946, -0.094714, -0.01508, -0.092651, 0.111445, 0.012356, 0.070057, -0.246322, -0.146223, -0.035806, 0.296379, 0.096431, -0.025193, 0.267994, -0.142898, -0.06882, 0.122424, -0.20284, -0.240223, 0.071047, 0.166728, 0.177284, -0.146812, -0.240221, -0.089157, 0.174044, -0.304759, 0.180377, -0.16442, 0.182503, -0.041977], [-0.036544, 0.00406, -0.00486, 0.141376, -0.127026, 0.143752, 0.158133, 0.064015, 0.043564, 0.144836, -0.216064, -0.073253, 0.089742, -0.030558, -0.131337, -0.017106, -0.159321, 0.038182, 0.112225, 0.062745, -0.020066, 0.028423, -0.242562, 0.035142, -0.120332, -0.106158, -0.232498, -0.18742, 0.164239, 0.006871, -0.09423, -0.101323, -0.047081, -0.273198, -0.000379, -0.436247, -0.21225, -0.041445, -0.170366, -0.04495, -0.258467, -0.25104, -0.591052, -0.081023, -0.170411, -0.00529, -0.035138, -0.035741, -0.35263, 0.043695, -0.17894, -0.267094, 0.017461, -0.122839, -0.338691, -0.307365, -0.360564, -0.147308, 0.133546, -0.198895, 0.086266, 0.062128, -0.378079, -0.288871], [0.197555, 0.131331, -0.22213, 0.045553, -0.167577, 0.0561, 0.088494, 0.038717, 0.197682, 0.074065, 0.012462, -0.018389, 0.191065, -0.292692, -0.047023, -0.036125, 0.09511, 0.010399, -0.036625, 0.056915, -0.164017, -0.091141, -0.169993, 0.120567, -0.258606, 0.105678, -0.000373, -0.000398, 0.064822, -0.055528, -0.226654, -0.197682, 0.169152, -0.125817, 0.162604, 0.175977, -0.118826, -0.129805, -0.169054, -0.146964, -0.016886, 0.019544, 0.078657, 0.069578, 0.101462, -0.17378, -0.140967, 0.091026, -0.015275, 0.098003, -0.057305, 0.065309, -0.259832, -0.063496, -0.063576, -0.188712, 0.036118, 0.014432, -0.161202, -0.071832, -0.024103, -0.138174, -0.132487, -0.136949], [0.114377, 0.457057, 0.097457, -0.084812, -0.1144, 0.147737, 0.271664, 0.163454, 0.031266, 0.093793, -0.078325, 0.378287, 0.365783, -0.077736, 0.037221, -0.03515, -0.010321, 0.458968, 0.258401, 0.047479, 0.131282, 0.131047, -0.058566, -0.227905, -0.24656, 0.109803, -0.252316, 0.054078, 0.08054, -0.072706, -0.084128, 0.322947, 0.073406, 0.012779, 0.033198, 0.07802, -0.067938, 0.196967, 0.278411, -0.042341, 0.055384, -0.26388, 0.025266, 0.08843, 0.037665, -0.232932, -0.042945, 0.182967, -0.046553, 0.223782, 0.015816, -0.106939, 0.164785, -0.069218, 0.061719, -0.124833, -0.017555, -0.218043, -0.039886, -0.04464, 0.230459, 0.169058, -0.111113, -0.030536], [-0.041861, -0.247211, 0.096064, 0.233098, -0.247124, 0.289727, -0.01709, -0.055987, 0.006879, 0.072883, -0.104819, 0.156211, -0.059063, -0.080402, -0.043465, -0.111673, 0.01269, -0.091668, -0.101791, 0.169275, 0.024187, 0.047547, 0.087985, 0.08044, 0.230205, 0.04315, 0.094028, -0.025319, -0.090886, 0.004349, 0.053881, 0.301973, -0.219732, 0.172948, 0.030332, -0.005119, 0.175754, -0.013875, -0.119409, 0.101844, -0.277934, -0.019888, 0.033841, 0.138268, -0.160323, 0.318062, 0.158818, -0.208101, 0.183886, -0.048465, -0.08065, 0.172099, 0.288132, -0.02158, -0.085571, -0.082014, -0.173507, 0.169541, 0.030093, -0.057168, -0.058507, 0.022761, 0.083142, 0.173918], [0.088259, 0.072139, -0.049688, -0.060037, -0.069852, 0.138461, 0.110366, 0.198977, 0.019305, -0.017034, -0.3511, 0.242028, -0.001512, 0.054555, -0.079788, -0.240426, -0.056222, -0.170303, 0.121931, -0.052797, -0.00043, 0.231483, 0.298224, -0.197554, -0.063119, -0.364224, -0.175834, -0.191748, -0.0212, 0.138296, -0.092372, 0.048234, 0.022618, -0.400826, 0.09886, -0.277488, -0.12967, -0.186068, -0.034957, -0.114168, -0.251321, -0.013172, -0.191558, -0.138343, 0.059626, -0.073671, -0.324246, -0.342445, -0.131845, -0.088154, -0.315033, 0.032286, 0.094513, -0.267427, -0.231676, -0.23085, -0.166728, 0.01244, 0.030533, -0.129929, -0.059422, -0.256819, -0.261709, -0.0905], [-0.019477, 0.025968, -0.027256, -0.199854, -0.121252, -0.060209, 0.062835, -0.033604, 0.179794, 0.128988, -0.074077, -0.093154, 0.038923, -0.09135, 0.046119, -0.027356, 0.011734, -0.086927, 0.136901, -0.184314, -0.016046, 0.089571, -0.259094, -0.23489, -0.192498, -0.036248, -0.073842, 0.014471, 0.06665, 0.023966, -0.049516, -0.173467, 0.202203, -0.107494, -0.03776, 0.044439, -0.163606, 0.039244, 0.041435, -0.296359, -0.071611, -0.092195, -0.295223, -0.047037, 0.110885, -0.145245, -0.095708, -0.175252, -0.400852, 0.008345, 0.260741, -0.249166, -0.084619, -0.208791, -0.097455, -0.330701, -0.210002, -0.239268, 0.16757, -0.152909, 0.141402, -0.027283, -0.173547, 0.167771], [-0.043768, 0.092955, 0.169305, 0.014358, -0.219814, 0.288787, 0.02093, -0.004771, 0.161174, 0.088619, 0.049165, 0.160603, -0.068519, 0.007532, -0.013046, 0.067914, 0.000679, 0.168888, 0.274277, 0.175857, -0.133824, 0.108421, 0.11974, -0.319131, 0.043633, -0.04693, -0.116061, -0.084414, 0.026085, -0.067135, 0.163239, 0.159794, -0.082268, -0.061563, -0.038651, 0.099208, 0.161514, 0.223052, 0.182389, -0.031749, -0.119209, -0.38567, 0.034295, 0.067257, 0.023063, -0.145625, 0.032775, -0.143245, -0.01093, 0.037502, 0.026804, -0.116953, 0.011773, -0.021691, -0.02953, -0.143777, -0.167506, 0.220799, -0.255216, -0.050054, 0.160376, -0.119872, 0.002405, 0.192441]], [0.134869, 0.055041, -0.140291, -0.168775, -0.11565, 0.081218, -0.071786, -0.077865, 0.111393, 0.021036, -0.169929, 0.063543, -0.066233, -0.160082, 0.082291, -0.118583, -0.074497, -0.102605, 0.083964, 0.051417, -0.137071, -0.351036, -0.200069, -0.151392, -0.138262, 0.04791, -0.057759, 0.119901, -0.032956, 0.157266, -0.184609, 0.067514, 0.004429, -0.063376, -0.102013, -0.103858, -0.026514, 0.035499, -0.181092, 0.001204, 0.012248, -0.001222, 0.048501, -0.071964, -0.14889, -0.112044, -0.204301, -0.102386, -0.202418, -0.136962, 0.183464, 0.017101, -0.195872, -0.171812, -0.043236, 0.016941, 0.011742, -0.17356, 0.190566, 0.002952, -0.045538, 0.15107, -0.16302, 0.03941]),
    ([[0.080449, -0.21873, -0.131811, 0.111323, -0.047018, 0.013207, -0.139009, -0.059426, -0.111119, 0.176469, -0.081073, -0.113774, 0.017226, -0.079901, -0.080821, 0.061372, 0.034515, -0.161814, -0.14112, -0.235509, -0.251887, -0.045328, 0.0965, 0.066356, -0.015679, -0.000159, -0.035472, 0.178826, 0.247598, -0.041521, 0.027804, 0.065959], [0.055404, -0.039162, -0.028122, 0.095719, -0.000435, 0.085119, -0.137758, -0.06171, -0.005144, -0.020771, 0.139831, 0.043407, 0.121336, -0.227642, 0.031865, -0.07302, 0.051163, -0.2935, -0.025297, 0.012108, -0.043099, 0.024085, 0.020334, 0.135726, 0.041066, -0.032639, 0.038306, 0.083705, -0.015929, 0.060899, 0.132722, -0.222155], [0.169393, 0.085271, -0.11174, 0.085289, 0.036594, 0.054304, 0.26286, -0.036517, 0.135713, 0.109792, 0.037605, 0.062409, -0.053406, 0.06883, 0.033698, -0.076334, 0.112928, 0.020558, 0.126686, 0.084807, 0.257086, 0.111849, -0.006243, 0.049385, -0.067637, 0.099233, 0.122422, 0.164708, -0.071257, -0.095913, 0.152067, 0.031926], [0.062929, -0.012589, 0.190835, 0.061752, -0.131657, 0.088376, 0.158779, -0.104265, -0.204567, 0.08887, -0.07255, 0.179101, 0.213753, 0.088456, -0.118447, 0.004273, 0.095183, 0.002577, 0.092158, -0.30369, -0.118092, -0.109337, 0.070197, 0.133394, 0.132425, 0.060572, 0.038617, -0.086823, 0.203733, -0.075304, -0.033053, 0.1881], [-0.223774, -0.070053, 0.135017, 0.120358, -0.105754, 0.117614, 0.052654, 0.002395, -0.014378, 0.134744, 0.139786, -0.07284, -0.022666, 0.038827, 0.157339, 0.057446, 0.08231, 0.138489, 0.039628, 0.063945, 0.032849, -0.08691, 0.047747, 0.004282, -0.09224, 0.117261, 0.15928, -0.046402, -0.013737, 0.023673, 0.033658, 0.017337], [-0.057165, -0.143695, -0.114056, -0.392363, -0.162717, 0.038289, -0.139463, 0.065982, 0.001185, 0.06913, -0.006858, -0.063584, -0.107995, 0.067163, 0.053399, -0.011622, -0.044783, -0.136293, 0.043128, 0.022266, 0.041881, -0.053705, -0.117739, -0.006547, -0.125249, -0.077232, 0.079138, 0.124692, -0.02968, 0.024772, 0.023468, 0.052296], [-0.031121, 0.085708, -0.029022, -0.033454, 0.037955, 0.0478, -0.010222, 0.066075, -0.108163, 0.094825, 0.175895, 0.014961, -0.040829, -0.191868, -0.122911, 0.065822, -0.025352, -0.001579, -0.136518, -0.150326, -0.000584, -0.026879, 0.046361, 0.078784, -0.040942, -0.044601, -0.000541, 0.203183, -0.066988, -0.043503, 0.106665, 0.164994], [-0.068957, 0.164083, 0.013249, -0.052794, 0.019845, -0.006901, -0.024158, -0.09516, 0.019216, 0.192457, -0.089187, 0.140637, -0.206344, 0.229079, 0.096667, 0.137363, 0.047279, 0.126476, -0.041708, 0.060562, 0.069906, 0.015562, 0.155127, 0.156067, 0.001323, 0.030574, 0.05228, 0.073883, -0.120738, 0.051363, 0.069978, 0.151456], [-0.172115, 0.123888, -0.128502, -0.446375, 0.022067, -0.163455, 0.040747, 0.129852, 0.00202, -0.119463, 0.014049, 0.038679, -0.085907, 0.163096, 0.015845, -0.049652, 0.053624, 0.037886, -0.111935, 0.131894, -0.081618, 0.080546, 0.088532, 0.173227, -0.061951, 0.061869, -0.010564, 0.031434, -0.122936, 0.0139, -0.01117, -0.044022], [-0.069375, -0.277096, -0.219894, -0.051856, -0.08647, -0.062669, -0.099473, 0.138157, -0.17219, -0.19795, 0.115149, -0.013164, 0.033374, -0.045363, -0.273177, 0.074309, 0.100474, -0.024149, -0.03308, -0.139239, -0.366515, 0.154301, 0.088743, 0.016384, 0.046922, 0.059793, -0.063371, -0.02324, 0.059394, 0.008176, 0.124314, 0.099001], [-0.003936, -0.012998, -0.104714, -0.102862, -0.003596, 0.090832, 0.067284, -0.102744, 0.043073, 0.019244, 0.022849, -0.102054, 0.060839, -0.002945, 0.02663, -0.03129, -0.104024, 0.118658, -0.056326, 0.155708, 0.112716, 0.034415, 0.087253, 0.112022, 0.168684, 0.057985, 0.051558, 0.0019, -0.05312, 0.09314, -0.000416, 0.030415], [0.042237, 0.129458, -0.022573, 0.228723, -0.12085, 0.12295, 0.008159, 0.14593, -0.14076, 0.168717, 0.013916, 0.121269, 0.194183, 0.040296, -0.13446, 0.042046, -0.048044, -0.17526, 0.114484, -0.076561, -0.072354, -0.033996, -0.070842, 0.065241, 0.140916, 0.133292, -0.031087, 0.147109, 0.098342, 0.12727, 0.144652, 0.067588], [-0.057522, -0.0225, -0.293219, -0.07543, 0.070703, 0.134523, -0.020997, -0.076756, 0.133165, 0.179539, 0.171853, 0.027778, -0.037313, -0.011154, 0.074188, 0.113207, 0.173617, -0.11943, -0.0389, 0.125701, 0.112012, 0.067213, -0.045943, 0.24682, -0.053057, -0.030527, 0.090093, 0.189886, -0.464926, -0.170535, -0.187276, -0.038457], [0.098671, 0.060935, -0.088294, 0.24074, -0.01862, 0.153495, 0.175573, 0.134144, -0.010546, 0.051972, 0.064634, -0.028136, 0.07518, 0.099646, 0.003509, 0.118797, 0.048971, 0.056342, 0.104267, -9.6e-05, 0.002653, 0.05699, 0.038046, -0.030182, 0.005556, 0.109703, 0.01345, 0.102198, 0.145524, 0.109806, 0.123405, -0.018583], [0.007717, 0.034039, -0.188438, 0.081788, -0.132468, -0.02039, 0.017161, -0.073946, 0.088145, 0.160107, 0.121364, -0.093131, -0.064946, -0.002166, 0.123172, -0.090289, -0.031681, 0.149234, 0.236546, 0.12117, 0.147821, -0.021895, 0.045484, -0.075192, 0.032015, -0.152524, -0.239316, -0.001512, -0.101942, -0.019758, -0.146907, 0.149038], [-0.152148, -0.101344, -0.090338, -0.042509, -0.089572, -0.064391, 0.183076, -0.216352, 0.00999, -0.257602, 0.164052, -0.137031, 0.008203, 0.177494, 0.160558, -0.154229, -0.003925, 0.060054, -0.015991, -0.024578, -0.073407, 0.10164, -0.005105, -0.149551, 0.112482, -0.19784, -0.0238, -0.084756, -0.124235, 0.013128, -0.124122, -0.048629], [0.12414, 0.006351, -0.155985, 0.007889, -0.168421, 0.126708, 0.108694, -0.00155, 0.073913, -0.10226, 0.089815, -0.079969, -0.000533, 0.079902, 0.288972, 0.127531, -0.074626, 0.341025, -0.059597, 0.116738, 0.069873, 0.113919, -0.132986, 0.107523, 0.034664, 0.087546, 0.034747, 0.117982, -0.190747, 0.180689, -0.082009, 0.033302], [0.156959, -0.090963, 0.089453, 0.112048, 0.065847, -0.056935, 0.003326, 0.133584, -0.089061, -0.006144, 0.178578, -0.036852, -0.141211, 0.150374, 0.004513, 0.139384, -0.010017, 0.068022, -0.147257, 0.083399, -0.063049, -0.129985, 0.044713, 0.168941, -0.290039, 0.119439, 0.022222, 0.061357, -0.102921, -0.129942, 0.12439, -0.114233], [-0.078628, 0.080076, 0.046218, 0.157949, -0.112305, 0.069158, -0.00498, 0.073229, -0.125547, -0.014504, -0.003152, 0.029797, -0.245431, -0.078236, 0.243999, 0.20203, 0.17893, -0.070097, 0.206277, 0.188127, -0.030084, 0.10351, -0.119718, 0.175219, -0.199486, -0.012884, -0.071538, 0.155749, -0.346302, -0.136692, -0.010753, -0.268044], [0.200501, -0.092629, 0.24329, 0.090126, -0.073079, 0.077706, 0.02361, 0.04556, -0.004631, 0.009444, 0.18144, 0.126441, -0.030019, 0.09928, -0.039353, -0.020771, 0.14902, -0.042887, 0.008364, 0.132417, -0.038411, 0.091946, -0.113638, 0.063661, 0.01491, 0.079054, 0.061217, -0.011813, -0.232395, -0.14033, 0.086333, 0.135527], [-0.180448, -0.178479, -0.145365, -0.205098, -0.060819, -0.13671, -0.159358, 0.103263, -0.193196, -0.12796, -0.109677, -0.072812, -0.074956, 0.014182, 0.011233, -0.025133, -0.068901, -0.175948, -0.224457, -0.020476, -0.02607, -0.24499, 0.098449, -0.085115, -0.047168, -0.050541, -0.10403, 0.077671, 0.122914, -0.181379, 0.019879, -0.070782], [0.063084, -0.053904, 0.033751, -0.048904, -0.044103, -0.053003, -0.137453, -0.098068, -0.01624, -0.067641, -0.060429, -0.035578, -0.089125, -0.081303, -0.013055, 0.089844, -0.000743, -0.34127, -0.159593, -0.193766, -0.127646, -0.156472, 0.079511, 0.138246, -0.20483, -0.188045, 0.166518, -0.020676, 0.102101, -0.217418, -0.033516, -0.116973], [0.162045, -0.064224, 0.22221, 0.119813, -0.114538, 0.119361, -0.143317, 0.107341, -0.289784, 0.135799, 0.09364, 0.237524, -0.029868, -0.142168, -0.125223, -0.064586, 0.120465, -0.106604, -0.209793, -0.166781, -0.203578, -0.074475, 0.079677, 0.072796, -0.081187, 0.131966, 0.0924, -0.137362, 0.176353, 0.127867, 0.230342, 0.03172], [-0.12656, -0.235377, 0.144108, 0.130681, -0.028561, -0.0423, -0.424687, -0.060176, 0.072507, 0.120236, 0.009797, 0.109277, -0.171308, -0.01236, -0.147358, 0.006197, 0.120807, -0.059759, -0.14928, -0.112012, -0.068059, -0.120272, -0.129431, 0.006681, 0.02302, 0.18278, 0.221042, -0.056916, -0.02801, -0.039116, -0.089338, -0.186985], [-0.178007, -0.190065, 0.00019, 0.09524, -0.061858, 0.125745, -0.327074, 0.076479, -0.145634, 0.078226, -0.122161, -0.020659, -0.294426, 0.001487, 0.000227, 0.175763, -0.089504, -0.376822, -0.416097, -0.036215, -0.133016, 0.019774, -0.158905, -0.133569, -0.125875, -0.007678, -0.085356, 0.137912, -0.038572, -0.014627, 0.058986, -0.093947], [0.085534, -0.022441, -0.043334, 0.040939, -0.061014, -0.020909, 0.068238, -0.077594, 0.178686, -0.021583, 0.020144, -0.065982, 0.121287, 0.069793, 0.175824, 0.038126, -0.112382, -0.038323, 0.103471, 0.206915, 0.172733, -0.293225, -0.083797, -0.116265, 0.024935, -0.05238, 0.003595, -0.090441, -0.065765, 0.179636, 0.145611, 0.045213], [-0.099548, 0.010271, 0.01791, -0.068302, 0.069011, 0.087226, -0.115845, -0.106878, -0.095323, 0.00111, 0.050852, -0.145023, -0.099236, 0.033375, -0.044611, -0.109581, -0.065122, -0.09869, -0.064836, 0.116382, 0.105863, 0.040251, 0.128106, -0.131074, 0.122578, -0.165832, -0.106089, -0.073195, 0.033573, -0.032328, -0.217079, -0.082273], [-0.092822, 0.095754, 0.036473, -0.013218, -0.033951, -0.109289, -0.067289, 0.102435, 0.124113, -0.182215, -0.250618, 0.153717, 0.08304, -0.030908, 0.051408, 0.10803, -0.052027, -0.103676, 0.131674, 0.163471, 0.066415, 0.083858, 0.088051, -0.023015, 0.069525, -0.073535, -0.153514, -0.179557, 0.056596, -0.022079, 0.050016, -0.05984], [0.076112, -0.186271, -0.107578, 0.040089, 0.006127, -0.069707, -0.037879, 0.064726, 0.046517, -0.136646, 0.095902, -0.019643, 0.072154, -0.164945, 0.019015, 0.084652, 0.126413, 0.054175, -0.134129, -0.072961, -0.133843, 0.172979, 0.021043, -0.025181, -0.033116, 0.030823, -0.089628, -0.038301, -0.030365, -0.15182, -0.083299, -0.120422], [-0.304345, 0.052349, -0.032749, -0.375276, -0.061653, -0.254123, -0.014347, -0.22131, 0.112243, -0.399586, -0.104283, -0.223813, 0.055216, -0.116088, -0.094643, -0.082784, 0.071735, 0.138067, 0.071366, 0.018128, -0.070067, -0.114001, 0.110195, -0.319364, 0.07009, -0.373226, -0.084366, 0.022337, -0.007054, 0.116452, -0.104678, -0.027182], [0.02715, 0.01818, -0.042814, 0.023757, -0.046516, 0.27227, 0.115635, 0.2463, -0.036532, -0.088813, 0.071018, 0.041435, -0.158945, 0.063506, -0.055696, 0.138054, 0.20766, 0.278688, -0.03361, 0.071779, 0.038537, 0.006197, -0.309442, 0.278673, -0.079992, 0.239257, 0.18315, 0.220279, -0.177715, -0.134157, 0.200134, -0.017672], [-0.058678, -0.074209, -0.049169, 0.025622, -0.122125, 0.012703, -0.097444, 0.157133, -0.157639, 0.159894, 0.138128, 0.131524, -0.264657, -0.123221, -0.165217, -0.073631, -0.0526, -0.019664, 0.01527, 0.065703, -0.014703, 0.039421, -0.152149, 0.071241, 0.067778, 0.085515, 0.019414, 0.090667, -0.226864, -0.144344, 0.04113, -0.154333], [0.072328, -0.217764, -0.108613, -0.037041, -0.160605, 0.015488, -0.165793, 0.082206, -0.112976, 0.164214, 0.003247, 0.086632, 0.009683, -0.152188, -0.056543, 0.048254, 0.131563, -0.389476, -0.071418, -0.16358, -0.261837, -0.155892, -0.014208, 0.015598, -0.103159, -0.096293, 0.144078, 0.032833, 0.052763, 0.084484, 0.082313, -0.123273], [0.084495, 0.10213, -0.175015, -0.127976, -0.098745, -0.055966, 0.133476, 0.043821, 0.138252, -0.159559, -0.058265, -0.249157, -0.124066, 0.131374, 0.187008, -0.162273, -0.271025, 0.229414, 0.177128, -0.010661, 0.162848, -0.118444, 0.001327, -0.187271, 0.009741, -0.186252, -0.061467, -0.171157, 0.145631, -0.122964, 0.073115, 0.15681], [-0.176316, -0.028424, 0.109699, -0.047536, -0.066553, -0.068624, -0.055146, -0.095761, -0.079915, -0.121882, -0.05152, 0.019626, -0.122523, -0.061027, 0.095562, 0.088262, 0.073159, 0.183444, -0.022343, -0.134696, 0.0562, -0.104582, 0.03142, -0.03302, 0.079039, -0.077216, -0.009411, -0.110149, -0.065098, -0.055968, -0.133814, 0.14856], [-0.264545, 0.086944, 0.085424, -0.065633, 0.080526, -0.120119, 0.035941, -0.274366, -0.019838, -0.206657, -0.012353, -0.13048, 0.041343, 0.128866, 0.117525, 0.029601, 0.016927, 0.117223, 0.157271, -0.015459, -0.049569, -0.31921, -0.027725, -0.263688, 0.068635, -0.115687, -0.004087, -0.25989, -0.020033, -0.035723, -0.177371, 0.052306], [-0.190305, 0.114709, -0.126052, -0.218669, -0.067519, 0.11154, 0.080559, 0.000569, 0.051897, 0.006106, -0.166952, -0.046, 0.137001, -0.051331, -0.036514, 0.080683, 0.107142, 0.188696, -0.021247, -0.039659, 0.019159, 0.226073, -0.008293, 0.154579, 0.063992, 0.111189, 0.218953, -0.045236, 0.05565, -0.086831, -0.041855, 0.121055], [-0.078371, -0.003661, 0.044042, 0.160741, -0.148265, 0.064229, 0.060208, 0.079917, -0.045529, 0.017682, 0.050512, 0.072046, -0.054696, 0.003102, -0.057139, 0.041091, -0.09513, 0.027678, -0.065856, -0.091466, -0.105463, 0.072978, 0.023089, -0.067625, -0.143716, 0.065191, 0.044777, 0.036629, -0.025861, -0.048658, 0.165442, 0.105177], [0.260765, -0.089774, -0.070864, 0.108564, 0.011921, 0.196963, -0.005822, 0.152783, -0.077056, 0.224812, 0.036342, 0.143099, -0.093101, -0.149751, -0.155045, 0.110645, 0.164611, -0.082799, -0.124199, -0.046781, 0.158971, 0.185983, 7.6e-05, 0.130418, -0.124079, 0.058286, 0.21387, 0.192114, -0.112945, 0.091114, -0.036563, -0.123447], [-0.107416, 0.113479, 0.05573, 0.033839, -0.112688, 0.056241, 0.04236, 0.081485, 0.037763, 0.056707, 0.112999, 0.037828, 0.089206, 0.067849, 0.038126, 0.159053, -0.043369, 0.100201, -0.001303, 0.038897, 0.031316, -0.035781, -0.085883, 0.075995, 0.088027, 0.087268, -0.027138, 0.111589, -0.002755, 0.158102, -0.107869, -0.092577], [0.018619, 0.114381, -0.028183, -0.026107, 0.057373, 0.14595, 0.018965, -0.203148, 0.073501, 0.111306, -0.093152, -0.130306, -0.232356, -0.179916, 0.203154, 0.016285, -0.014966, 0.027336, 0.191841, 0.123418, 0.151614, 0.013277, -0.005546, 0.035769, -0.05073, -0.078911, 0.097619, -0.095694, -0.381912, -0.134332, -0.083583, -0.29457], [-0.204486, -0.073652, 0.007624, -0.360597, -0.02229, -0.050698, 0.118293, 0.061612, -0.186621, -0.043616, 0.048482, -0.140334, -0.016336, -0.084868, -0.071199, -0.10464, -0.165947, 0.171246, -0.207595, -0.01519, -0.078784, -0.302895, -0.181513, -0.000574, -0.087375, 0.057305, -0.237806, 0.003772, 0.122426, -0.123308, 0.021496, -0.01393], [-0.139959, 0.143621, 0.194534, 0.014429, -0.086206, -0.05065, 0.025118, 0.028003, 0.125743, -0.035607, -0.012975, 0.154269, -0.044113, -0.162704, 0.101057, 0.055186, -0.018202, 0.122255, 0.152625, 0.195927, 0.157357, 0.091751, -0.040932, 0.027454, -0.038357, 0.124616, -0.045188, -0.104351, -0.144846, 0.129606, 0.139195, 0.043825], [-0.183582, -0.112599, -0.25667, -0.516297, -0.006137, 0.096284, -0.011991, -0.000494, 0.118204, -0.307825, -0.11782, -0.261114, 0.045136, 0.114563, -0.048117, 0.132183, 0.029273, -0.136669, -0.033026, 0.013495, 0.082312, -0.164346, 0.121812, -0.046065, 0.114232, -0.25197, -0.036221, 0.006896, -0.15709, 0.000119, 0.041933, -0.032919], [0.109438, -0.031455, 0.027697, 0.212819, -0.099679, 0.021694, -0.073836, -0.039726, 0.007114, 0.274659, 0.006642, -0.03014, 0.023225, -0.05809, -0.008286, 0.128592, 0.029683, 0.115856, -0.015763, 0.175268, 0.136417, 0.025906, -0.145034, 0.193026, 0.12604, 0.034705, 0.146979, 0.062483, -0.096497, 0.137948, 0.036209, -0.158072], [-0.192653, -0.092484, -0.024052, -0.149005, 0.054705, -0.228931, -0.001036, 0.148944, -0.18803, -0.07381, -0.17719, 0.057308, 0.006022, -0.167261, -0.064213, -0.134027, 0.09676, -0.032187, -0.008647, -0.027161, 0.042597, 0.029945, 0.065364, 0.022446, 0.043311, 0.093588, -0.015203, -0.138971, 0.058143, -0.159605, -0.092756, -0.197862], [0.170811, 0.019657, 0.150667, 0.328506, 0.077549, -0.104347, 0.061515, 0.165853, -0.053845, 0.116372, 0.143261, 0.014922, 0.234732, 0.215771, -0.041836, -0.121964, -0.231703, 0.103696, -0.080597, -0.059014, 0.077135, -0.010358, 0.040951, -0.02913, 0.10431, -0.03975, -0.081059, -0.165119, -0.135609, 0.002684, 0.111388, -0.108912], [-0.198518, 0.055764, -0.126277, 0.202238, -0.017631, 0.098833, -0.00734, -0.123917, 0.044762, -0.013679, 0.01415, 0.028611, 0.088508, 0.038692, 0.118988, 0.095882, 0.091303, -0.080714, -0.001063, 0.03118, 0.149706, -0.016477, 0.003426, -0.029515, 0.069011, -0.103385, 0.064476, -0.058326, -0.048883, 0.009492, -0.039848, -0.065455], [0.082202, -0.104002, 0.208238, -0.087469, -0.100422, 0.055234, 0.093528, 0.175147, 0.144844, 0.218188, -0.065714, 0.154396, -0.011186, -0.000376, -0.03838, 0.044176, 0.016872, 0.100001, 0.085881, 0.051392, 0.067842, -0.07931, 0.091458, 0.132538, 0.031602, 0.254323, 0.020754, 0.158401, -0.320263, -0.16825, 0.104542, 0.016638], [0.227019, -0.026848, 0.159758, 0.038078, -0.038183, 0.104413, 0.047308, 0.142516, 0.067924, 0.20882, -0.0444, 0.199803, 0.055118, -0.190694, 0.072964, 0.039857, 0.110958, 0.010005, -0.050889, -0.158647, -0.057969, -0.070846, -0.074334, 0.157959, 0.069295, -0.051074, 0.162855, 0.0339, 0.011045, 0.050241, 0.092695, -0.012938], [0.024896, 0.158064, -0.228212, -0.142183, -0.034611, -0.061316, 0.032072, -0.030812, -0.071282, 0.052668, 0.132269, -0.22071, -0.156795, 0.202721, 0.105166, -0.091796, -0.09162, -0.142289, -0.113762, 0.042972, 0.068561, -0.011039, 0.062913, -0.192436, 0.039526, -0.073839, -0.178779, -0.098692, 0.075418, 0.128476, -0.00521, -0.058833], [-0.081562, 0.152897, -0.080771, 0.057943, -0.078589, -0.070413, 0.137803, 0.000493, 0.038256, 0.009307, 0.039415, 0.05927, -0.021383, -0.130065, 0.112414, -0.057619, -0.114785, 0.128896, 0.04356, 0.077269, 0.113351, -0.096271, 0.200761, -0.093096, 0.105621, 0.017592, -0.034778, -0.031486, -0.005581, 0.120872, 0.070174, -0.049386], [-0.110615, -0.138651, 0.084391, 0.12495, -0.058384, -0.072974, -0.131238, -0.018742, -0.014661, 0.027168, -0.151054, 0.013179, 0.022218, -0.274877, -0.084115, -0.011699, -0.01829, 0.035349, 0.110119, -0.031001, -0.087146, 0.031833, 0.071162, -0.116537, -0.018741, 0.093805, 0.014952, -0.074273, 0.166901, -0.216595, -0.184484, -0.036362], [-0.24822, -0.013569, -0.118984, 0.331792, 0.05696, -0.034987, 0.178355, 0.07594, 0.00337, 0.368336, -0.080475, -0.1299, 0.198175, 0.173484, -0.035812, 0.056438, 0.038304, 0.134219, -0.068489, 0.153861, 0.171569, 0.095015, 0.050872, 0.008329, 0.114867, 0.040624, 0.02132, 0.145234, 0.044249, 0.104817, -0.016797, 0.04722], [0.16466, 0.211483, 0.190561, 0.021376, -0.068614, 0.088248, 0.246664, -0.005537, 0.193547, -0.069294, 0.104904, -0.052537, 0.077984, 0.179955, 0.181735, 0.063549, -0.055884, 0.145598, 0.263834, 0.157844, 0.17685, 0.010079, 0.128702, 0.026535, 0.064351, 0.139696, 0.026675, -0.076845, -0.033503, 0.055073, 0.083087, 0.057069], [0.097485, 0.11783, 0.058654, -0.080534, 0.070397, -0.064456, 0.114784, 0.023875, 0.047217, -0.001678, 0.083715, -0.067951, 0.042956, -0.052213, 0.153716, -0.113319, -0.05795, 0.307059, 0.016248, 0.219718, 0.181671, -0.220418, -0.004077, -0.156744, 0.066022, -0.132749, -0.094335, 0.079359, -0.090106, 0.020525, -0.003567, 0.043536], [0.059664, -0.028253, 0.16393, -0.080644, -0.047331, -0.154633, 0.162457, -0.073681, -0.034327, -0.109374, -0.137913, 0.109122, 0.015907, -0.027021, 0.120478, -0.072862, -0.118001, -0.053551, 0.084399, 0.037991, -0.045569, -0.049003, 0.142884, -0.046349, 0.108458, -0.089513, -0.184025, 0.007592, 0.101108, -0.075972, -0.100592, 0.058304], [-0.1027, -0.021369, 0.276633, -0.175336, 0.098475, -0.028185, 0.147212, 0.051898, -0.025491, -0.131857, -0.110907, 0.026476, 0.159112, -0.099698, 0.028567, -0.098934, -0.154859, -0.008734, 0.098909, -0.100456, 0.032663, -0.331214, 0.109227, -0.070688, 0.124061, -0.134001, 0.031967, -0.042205, 0.245077, -0.06272, -0.217181, -0.022306], [-0.022163, 0.065449, -0.115531, -0.367217, -0.005919, -0.026496, -0.067256, -0.101871, -0.147734, -0.015418, -0.328246, -0.15458, -0.124062, 0.147357, 0.000785, -0.216051, 0.077955, -0.035933, -0.061607, -0.064888, 0.059981, -0.02227, -0.013909, -0.149399, -0.022078, -0.083065, -0.275053, -0.069968, -0.06472, 0.116265, 0.007564, -0.100912], [-0.169499, 0.1249, 0.171816, -0.10702, -0.107006, -0.09868, 0.207036, -0.06066, 0.186817, 0.102638, -0.15662, 0.059271, -0.012359, 0.273066, 0.076005, 0.10082, 0.035778, 0.215023, 0.012098, -0.021255, 0.056004, -0.023277, 0.163689, 0.058016, 0.050439, -0.075418, -0.132434, -0.024807, 0.106259, -0.111069, -0.167706, 0.011982], [0.000119, -0.010444, -0.164041, -0.063989, -0.049161, 0.069471, -0.19639, -0.007326, -0.241023, -0.077123, 0.061828, 0.069781, 0.026235, -0.000256, -0.03632, 0.041076, 0.107559, -0.004102, -0.02412, -0.016155, -0.221055, 0.076357, -0.150301, -0.035514, -0.091234, 0.088688, 0.106974, -0.058286, 0.193838, 0.056239, 0.048009, -0.030534], [0.157689, -0.03101, 0.022205, 0.07725, -0.012866, -0.105989, -0.095978, -0.0315, -0.1399, 0.067174, -0.010882, -0.028047, -0.104581, -0.042075, -0.062023, 0.003952, 0.09339, -0.217649, -0.063582, 0.035264, -0.109793, -0.163886, 0.198794, -0.123379, -0.098577, 0.005771, 0.050332, -0.049695, 0.159093, -0.132091, -0.143091, 0.029276], [0.034743, 0.136928, 0.19724, -0.031805, -0.071832, -0.011809, -0.007736, -0.052806, -0.070353, 0.040439, 0.06499, -0.128017, 0.184728, 0.002512, 0.165534, -0.109646, 0.003464, 0.098843, 0.077754, 0.137993, 0.109562, -0.010119, 0.034745, -0.068671, 0.020501, -0.10773, -0.071801, -0.021291, -0.023435, -0.04642, -0.1163, -0.118324], [-0.178756, 0.053727, 0.023728, -0.334529, -0.089601, -0.06111, 0.074246, 0.003911, -0.03257, -0.648536, 0.025228, 0.111025, 0.051512, 0.044715, -0.007084, -0.107921, 0.027566, -0.079712, 0.037445, -0.02745, -0.057429, 0.103723, 0.100445, 0.001952, 0.117647, -0.036236, -0.010152, -0.124113, -0.200477, 0.084837, 0.026084, 0.015043]], [-0.141976, 0.039617, -0.163922, -0.187089, -0.178855, 0.034393, -0.157148, -0.001587, 0.11247, -0.104551, -0.008352, -0.206397, -0.024011, -0.104717, 0.118141, 0.07994, 0.041426, -0.144076, 0.027512, 0.167847, 0.02782, 0.108145, 0.063167, -0.107537, 0.075259, 0.033332, 0.001064, -0.01435, 0.016486, 0.03075, 0.119651, -0.126744]),
    ([[0.239729], [-0.10196], [0.226641], [0.397365], [-0.003571], [0.119322], [-0.201139], [0.133121], [-0.113547], [0.197798], [0.122912], [0.135728], [-0.100617], [-0.128843], [-0.138154], [0.11711], [0.138485], [-0.31405], [-0.102125], [-0.135354], [-0.154779], [0.12119], [-0.111524], [0.081646], [-0.104385], [0.094887], [0.13299], [0.108644], [-0.211142], [-0.145666], [0.130892], [-0.039661]], [0.023759]),
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
        'rt_ms': 130.0,
        'ri_ms': 0.0,
        'qa_max': 0,
        'fq_k_sp2': 2.0,
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
        dbid = floor(f - kb)
        dask = -floor(-(f + ka))
        tk = self.fq_take_k
        if tk > 0.0:
            ft = f
            if self.pred_on and self.pred_mode:
                ft = mid + (self.pred_scale * self.pred_sig if self.pred_mode == 1 else self.fq_scale * self.fair_sig) + fshift
            take_buy = floor(ft - tk) >= ba
            take_sell = -floor(-(ft + tk)) <= bb
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
