"""exp_r3_ensb (home, R3): mm_r3_gru2 + an ensemble of the taker GRU with two more trained GRUs (ens_n; 0 = mm_r3_gru2):
the members share the token / feature normalisation and run as ONE block-diagonal GRU and head (one matmul per layer),
the prediction is the mean of the member heads. Members: the exported GRU, seq_r3 seed 0 (logs/seqt_y200.pt), lr 1e-3 x 10
epochs (logs/sweep_h16lr1.pt).

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
# exp_r3_ensb: extra GRU members (same SEQ_TM / TSD / XM / XS / YSC)
SEQ2_WI = [[-0.063672, -0.25791, -0.514315, -0.360631, -0.547316, -0.592458, -0.516213, -0.562651, -0.37759, -0.068542, -0.831015, -0.790268, -0.48112, -0.890628, -0.216625, -0.331622, 0.540606, -0.28143, -0.039745, -0.056916, -0.356588, -0.146426, 0.442499, -0.228928, -0.132944, -0.321226, -0.212894, -0.568153, -0.410465, 0.326723, -0.046889, 0.094519, -0.002737, -0.061397, -0.241799, 0.297741, -0.253589, -0.136411, 0.219342, 0.216864, 0.221509, -0.023243, -0.128762, 0.033807, -0.123511, 0.355129, 0.233694, -0.414913], [0.006224, -0.134661, -0.425047, 0.572864, 0.359771, 0.463249, -0.265426, 0.412036, -0.328978, -0.071944, -0.013751, -0.20735, 0.115186, 0.238729, -0.021339, -0.320621, -0.121781, 0.022823, -0.568014, -0.034603, 0.192208, -0.070949, 0.369053, -0.010141, 0.170818, -0.285707, 0.132043, -0.291841, 0.162712, 0.536923, -0.138395, 0.061627, -0.063597, -0.587077, 0.592772, 0.00586, 0.168347, -0.374395, 0.124168, 0.166315, 0.067165, -0.773557, -0.131824, -0.334535, -0.117845, 0.252964, -0.117867, 0.019755], [0.153219, 0.232041, -0.234232, -0.337705, 0.513845, -0.389489, -0.118561, 0.07636, -0.617485, -0.225896, 0.035093, -0.482726, 0.301143, -0.530021, -0.196016, 0.223054, 0.063042, 0.216843, -0.164767, -0.177893, 0.611343, -0.256263, -0.353494, -0.519862, -0.28825, 0.314482, -0.591019, 0.514152, 0.763723, -0.051458, -0.338221, -0.032346, 0.202419, 0.168341, -0.552118, -0.053129, 0.058666, 0.316639, -0.224092, -0.026529, 0.29137, 0.157961, 0.149976, -0.204095, 0.221082, -0.292693, 0.16873, -0.199449], [-0.320907, 0.055062, -0.202472, 0.209716, 0.230712, 0.184962, 0.569057, -0.04141, -0.789052, -0.213233, 0.044998, -0.25928, 0.029294, -0.053838, -0.053803, -0.016479, -0.045592, -0.097219, -0.640911, -0.192546, 0.464068, 0.330523, 0.055823, -0.071308, 0.047918, -0.133151, -0.166299, 0.039418, 0.011932, 0.187461, -0.323452, 0.360485, 0.192044, 0.162265, 0.029293, -0.232091, 0.462207, -0.073739, 0.169525, 0.039343, -0.233629, 0.018506, 0.436437, -0.20553, 0.334189, 0.200052, -0.335814, -0.016575], [-0.178744, 0.207778, 0.150889, -0.68076, 0.23617, 0.483356, -0.042754, 0.201705, -0.370955, -1.168121, -0.413086, 0.091422, 0.28931, 0.737936, 0.282414, 0.181165, -0.127599, 0.402004, -0.138279, 0.256455, -0.095942, 0.315855, -0.471182, -0.051644, 0.615891, -0.992221, 0.280511, 0.329508, 0.30129, -0.184783, -0.324388, 0.210409, 0.034057, 0.167407, -0.482107, 0.052334, -0.097679, 0.107892, 0.090255, 0.191599, -0.074688, -0.827665, -0.217972, 0.364669, 0.216146, 0.068106, -0.35026, 0.053259], [0.181438, -0.188658, -0.364514, -0.631256, -0.48935, -0.654386, 0.398538, 0.0788, -0.27278, -0.282457, 0.431688, -0.455667, -0.455701, 0.052239, 0.798088, -0.509646, -0.080218, -0.994397, 0.481035, 0.154483, -0.570768, 0.267661, 0.331701, 0.227495, 0.398978, -0.255482, 0.53568, 0.109212, -0.173211, -0.338867, 0.04764, -0.370934, -0.264809, -1.154929, 0.37074, 0.008044, 0.005886, 0.000428, 0.480796, 0.107909, -0.211513, -0.08416, -0.553512, -0.027387, -0.501645, -0.109696, 0.118335, 0.430564], [0.14895, 0.092904, -0.534278, 0.489348, 0.193821, 0.080739, -0.094977, 0.303447, -0.18597, 0.192544, 0.202808, 0.145392, 0.552374, 0.267402, -0.380841, 0.319356, 0.002112, -0.099182, 0.255639, -0.061903, -0.386021, 0.024836, -0.176092, 0.052265, 0.069249, 0.119538, 0.110137, 0.243696, -0.351582, 0.634823, -0.295187, 0.052809, 0.257477, -0.099402, 0.030893, -0.073023, 0.14669, 0.532316, -0.190914, -0.009661, 0.196981, 0.259232, -0.025478, -0.249401, 0.000264, -0.391238, -0.376806, -0.010007], [0.475296, 0.274703, -0.523477, 0.714445, 0.224376, 0.274094, -0.208391, -0.024609, 0.104578, -0.209213, 0.164277, 0.031993, 0.505082, 0.241787, -0.133816, -0.177461, 0.245513, -0.226302, 0.446375, -0.040503, -0.235314, 0.010941, 0.005167, 0.052499, -0.21253, 0.405609, 0.280623, -0.095877, 0.010099, 0.15468, -0.399213, -0.325704, 0.069948, 0.027662, -0.254429, 0.174562, -0.196908, 0.212963, 0.181034, 0.137092, -0.124806, -0.077478, -0.198203, -0.442545, 0.074494, -0.357877, -0.347834, 0.209368], [0.2017, 0.296916, -0.647647, 0.656524, 0.379506, 0.416344, -0.262744, -0.227967, 0.025111, -0.096349, 0.078212, 0.014524, 0.544212, -0.053127, -0.049152, -0.47086, -0.14673, -0.147464, 0.202903, -0.072676, -0.057238, -0.178764, -0.198555, 0.104543, 0.26017, 0.286792, 0.175267, 0.134032, -0.266324, 0.493595, -0.49871, -0.316656, 0.177853, -0.225592, 0.24988, -0.315075, -0.091052, 0.022753, 0.127384, -0.010851, -0.061672, 0.189835, 0.023992, -0.356182, -0.146887, 0.164171, -0.011024, -0.011513], [0.194479, -0.643113, -0.368612, -0.042366, 0.03895, -0.083954, 0.019552, -0.041605, -0.077208, 0.196624, -0.056471, -0.281979, 0.28387, 0.288079, 0.683795, -0.356908, -0.035653, -0.056504, -0.288508, 0.137835, 0.000623, -0.334101, -0.104089, -0.663584, -0.062685, 0.102605, -0.119579, -0.511736, -0.323237, 0.08166, 0.469239, 0.388078, -0.655274, 0.229843, 0.159968, 0.196238, 0.141753, 0.198038, 0.170402, -0.156963, -0.248489, -0.090911, -0.189746, 0.319414, -0.239676, 0.231616, -0.33394, -0.167326], [-0.461688, 0.103655, -0.38701, 0.163323, -0.048147, 0.082727, 0.269658, -0.385318, -0.361424, -0.037358, 0.239713, -0.237137, -0.102448, -0.092947, 0.123982, -0.221088, 0.319678, -0.284659, 0.133882, -0.266465, -0.703753, -0.068282, 0.023999, 0.090024, -0.955163, -0.395579, -0.050259, -0.109512, -0.408222, 0.350494, -0.273019, -0.105678, -0.048187, -0.120645, 0.211872, -0.080239, 0.013878, 0.160757, 0.295742, -0.141636, 0.402703, -0.036472, -0.103896, -0.074979, -0.131945, -0.075958, -0.106208, 0.20833], [-0.155312, -0.244739, 0.083517, -0.090946, 0.177891, -0.454639, -0.312972, -0.654708, 0.084801, 0.005477, -0.120123, 0.151806, 0.784792, 0.065475, 0.376589, -0.469982, -0.349376, -0.216134, -0.516303, -0.485907, -0.259902, 0.118921, -0.497738, 0.002537, -0.568232, -0.11003, -0.107836, -0.31648, -0.559233, -0.208856, 0.153663, -0.449763, -0.262334, -0.168734, 0.066447, -0.049687, 0.100479, 0.068513, -0.118114, 0.024082, 0.254192, 0.032102, -0.099585, 0.108906, -0.001234, -0.222729, -0.041283, 0.068026], [-0.541077, 0.13487, -0.047329, 0.531358, 0.022665, 0.039334, 0.356943, -0.155588, -0.387365, -0.356557, 0.227415, -0.330125, -0.850598, 0.159989, -0.043975, 0.037558, 0.002999, 0.297964, 0.149947, -0.307643, -0.729209, 0.558643, 0.101021, -0.401679, 0.060511, 0.081895, 0.365885, 0.573691, 0.284303, 0.440335, -0.081076, -0.122741, -0.092549, -0.137968, -0.19939, -0.288746, -0.212662, -0.14494, 0.082523, -0.036617, -0.14703, 0.112453, 0.08077, 0.085287, 0.003151, 0.106287, -0.058473, -0.279818]]
SEQ2_BI = [0.02743, 0.440348, 0.481677, -0.041872, 0.254574, 0.109349, 0.302931, 0.189871, -0.121573, -0.206353, -0.027563, 0.024091, 0.112224, 0.288008, 0.057126, -0.065663, 0.263339, 0.559711, 0.59846, 0.377206, 0.620957, 1.001435, 1.145357, 0.718324, 1.234877, 0.338472, 0.3679, 0.485474, 0.570638, 0.288179, 0.412057, 0.385537, -0.287597, 0.10706, 0.187713, 0.151614, -0.242915, -0.34383, -0.151574, 0.127066, 0.305487, -0.089034, -0.148083, -0.228485, -0.0004, -0.369442, -0.231589, 0.336363]
SEQ2_WH = [[-0.085105, 0.371297, -0.101456, 0.088684, 0.014511, -0.198044, -0.008425, 0.158212, 0.385623, -0.26304, -0.081505, 0.13491, -0.049853, -0.049345, -0.741293, 0.411361, 0.019276, -0.845054, 0.14039, 0.387264, -0.875292, -0.265394, -0.333366, -0.072376, -0.696693, -0.407477, -0.369424, 0.051049, 0.706938, -0.395274, -0.275479, -0.263132, 0.40499, -0.383208, 0.117523, -0.203643, 0.512445, 0.354144, -0.475209, 0.049595, 0.258206, -0.168336, 0.236893, 0.047025, 0.258431, 0.066448, 0.218902, -0.263293], [-0.257007, -0.359314, 0.442398, -0.129047, 0.029848, 0.074037, -0.574512, -0.428699, -0.568998, -0.059036, 0.201723, 0.214267, 0.283087, -0.203199, -0.129047, 0.289674, -0.333692, -0.219751, -0.150305, 0.45108, -0.052674, -0.139071, 0.419814, -0.326038, 0.588165, -0.097735, 0.567738, 0.053708, -0.045573, -0.650689, -0.300751, 0.266955, 0.03329, 0.60043, -0.018377, -0.462138, 0.331156, 0.097483, 0.016203, -0.079547, -0.345681, -0.006977, -0.185859, -0.390665, -0.198189, -0.733761, 0.157543, 0.057513], [-0.491008, 0.089239, -0.777616, 0.113723, 0.474734, -0.118139, -0.18127, -0.042223, -0.052715, 0.669709, 0.215357, 0.017999, 0.487876, -0.024301, 0.228283, -0.327862, -0.17529, 0.346297, 0.837078, 0.046751, 0.103126, 0.552467, -0.275244, 0.66981, 1.560628, -0.009858, 0.623082, 0.617206, -0.186782, 0.530227, -0.077303, -0.149024, -0.030838, 0.449002, 0.305012, -0.221365, 0.014546, -0.234532, 0.461918, 0.269, 0.564044, 0.592637, 0.684149, 0.119608, -0.300897, 0.113966, -0.007712, 0.114181], [-0.137931, 0.051719, -0.611991, -0.02373, -0.691776, -0.141086, -0.287083, -0.245475, 0.609692, -0.11475, -0.996425, -0.593037, -0.197002, -0.304296, -0.127827, -0.496979, 0.271586, 0.568124, 0.208491, -0.00183, 0.00652, -0.060788, -0.226951, -0.10966, -0.39988, 0.469283, 0.207204, -0.657875, -0.34122, 0.53573, 0.351892, 0.103468, -0.259835, -0.28964, 0.141576, 0.330273, -0.408823, -0.481219, -0.32473, 0.526036, 0.227009, 0.125655, -0.304808, 0.031764, -0.540903, -0.170782, 0.017905, -0.014769], [-0.293804, -0.286043, -0.993868, 0.066127, 0.101052, -0.116905, -0.265539, -0.291041, 0.119464, 0.341656, -0.110351, 0.222684, 0.600888, -0.241437, -0.414294, 0.199656, -0.419289, -0.373267, -0.562634, 0.251924, -0.688364, -0.626693, -0.573115, -0.477692, -0.371883, -0.771891, -0.157413, -0.643875, -0.680537, -0.088467, 0.228236, -0.491388, 0.089033, -0.161072, -0.178215, 0.102223, 0.215153, 0.463289, 0.122249, -0.064115, -0.013758, 0.207081, -0.071, -0.200928, 0.37413, 0.152808, -0.084645, -0.071667], [-0.089042, -0.038781, -0.474532, -0.051761, 0.219049, -0.159447, -0.306226, -0.089956, 0.32222, 0.183591, 0.079967, 0.301343, 0.319891, -0.105235, 0.175207, 0.009104, -0.252886, -0.769373, -0.797263, 0.202016, -0.183531, 0.01891, -0.244112, -0.550143, -0.8287, -0.32256, -0.730195, -0.216631, -0.707863, -0.292616, -0.598478, -0.366372, 0.214342, 0.20664, -0.290913, -0.380244, 0.05339, 0.469663, -0.236912, -0.409917, 0.389525, -0.100187, -0.064391, -0.103558, 0.553386, 0.345541, 0.120405, -0.348914], [0.159482, -0.659949, 0.427014, 0.188255, 0.101188, -0.124681, 0.638508, -0.034734, -0.365876, 0.139565, -0.103847, -0.077542, -0.435941, -0.270442, 0.121772, -0.1966, -0.011613, 0.199122, -0.1892, -0.415403, -0.574512, -0.871047, -0.011482, -0.526641, 0.208517, -0.06648, 0.488442, 0.233672, 0.002226, 0.076468, -0.764402, -0.361497, -0.172768, -0.207776, -0.26886, 0.052259, -0.281029, 0.029505, 0.813447, 0.267335, -0.220999, -0.053382, 0.172238, 0.437403, -0.445399, -0.227889, 0.074304, -0.171101], [0.277315, 0.619903, 0.292193, -0.274046, -0.567021, -0.30164, 0.163067, 0.039815, 0.404254, -0.71619, -0.341211, -0.865665, -0.442964, 0.119712, -0.564446, -0.116782, -0.147287, 0.388889, 0.525077, 0.288525, -0.018208, 0.497269, 0.047291, 0.019517, 0.489098, 0.860275, 1.089644, -0.25367, 0.435672, 0.422118, 0.609036, -0.073528, 0.012147, -0.018851, -0.303381, 0.518811, 0.241279, -0.44279, -0.217639, 0.399252, 0.469986, -0.125779, 0.179947, -0.037192, -0.285525, 0.507069, -0.155743, -0.101311], [-0.189589, 0.102354, -0.378382, -0.04241, -0.218885, -0.576426, -0.117002, 0.21606, 0.288898, -0.179733, -0.19903, -0.654257, -0.31446, 0.19761, -0.208573, -0.873625, 0.771493, 0.579898, 0.004741, 0.172476, -0.376273, 0.296912, 0.159166, 0.383207, 0.293806, -0.085767, -0.072886, -0.745916, 0.387065, 0.521616, 1.089366, 0.147232, 0.001021, -0.458323, 0.284943, 0.374401, -0.371613, -0.290023, -0.545555, 0.329856, 0.3195, -0.266214, 0.141938, -0.105461, 0.108935, 0.649946, -0.365161, 0.081846], [0.145536, -0.184017, 0.034779, 0.336114, 0.111346, -0.23467, -0.671252, -0.136213, 0.035012, 0.432642, 0.814537, 0.642999, 0.302423, -0.262612, 0.206915, 0.081285, -0.173492, 0.04, -0.304597, -0.24434, 0.204548, -0.02378, -0.223023, 0.431266, 0.457456, -0.683045, -0.074437, -0.486539, -0.331569, -0.493469, 0.266612, -0.395754, -0.273496, 0.137888, 0.321893, -0.287134, -0.258147, 0.176473, 0.02911, -0.542766, -0.248553, 0.580372, 0.526177, 0.373546, 0.103177, -0.198579, 0.646401, 0.389266], [0.12505, 0.509121, 0.597418, 0.3945, 0.483959, 0.953826, 0.081032, 0.361194, -0.143467, 0.56859, 0.372033, -0.328796, 0.109137, -0.341159, 0.097671, -0.331033, -0.159142, -0.494718, 0.232517, -0.171231, -0.656716, -0.45703, -0.282482, 0.342298, 0.254304, 0.088334, 0.324764, 0.311347, 0.045304, 0.202325, -0.445151, 0.283082, -0.029504, -0.204516, 0.512195, 0.08793, -0.471269, -0.309021, 0.595954, 0.055129, -0.05529, 0.556759, 0.595052, 0.143327, -0.042524, -0.010993, 0.05279, 0.331428], [0.070512, -0.230699, 0.222699, 0.105835, 0.212554, -0.133604, 0.120832, -0.221948, 0.366661, 0.447648, -0.065834, 0.027526, -0.026374, -0.186045, 0.435864, 0.028413, 0.228776, -0.380824, -0.429211, -0.44981, -0.178749, -0.290864, -0.538383, -0.241022, -0.060926, -0.54143, -0.284717, -0.274053, -1.070223, -0.120987, -0.780451, -0.180781, -0.337774, 0.069503, -0.236881, -0.252566, -0.191768, 0.259914, 0.045303, 0.233781, -0.28338, 0.626519, -0.052787, 0.187901, 0.070529, 0.228854, 0.249485, 0.11717], [-0.285545, 0.290201, -0.073843, 0.253534, -0.056598, 0.092673, -0.336244, 0.0929, 0.342467, -0.402253, 0.059541, 0.173902, 0.345129, 0.288817, 0.130652, 0.281285, 0.169294, -0.053774, 0.267913, 0.621728, -0.385262, 0.715966, 0.078203, 0.206662, -0.133563, -0.474508, -0.074418, 0.168496, 0.238267, -0.341012, 0.330806, 0.139567, 0.365734, 0.218655, 0.374722, 0.12707, 0.160678, 0.277776, -0.589399, -0.520546, 0.425558, -0.220288, -0.312654, -0.405477, 0.224866, 0.051392, -0.200607, -0.499511], [-0.108743, -0.082747, -0.567492, 0.229117, 0.190196, 0.126456, 0.705454, -0.26249, 0.278838, 0.644436, -0.505677, -0.579552, -0.505753, -0.013833, 0.490996, 0.084375, 0.54181, 0.230277, -0.622493, -0.587885, -0.503493, 0.842019, 0.155153, 0.665992, -0.456289, -0.693662, 0.434189, -0.126287, -0.527634, 0.216529, -0.19241, -0.070734, -0.327744, -0.375757, -0.144215, 0.059315, -0.426885, -0.099066, 0.045303, 0.08216, 0.376952, 0.259608, -0.227759, 0.271461, 0.083927, 0.301038, -0.214266, -0.05207], [-0.019828, -0.288202, -0.417203, -0.027927, -0.629975, -0.037873, -0.586854, -0.146101, -0.176209, 0.161174, 0.078502, 0.227174, -0.071186, -0.416552, -0.123689, 0.029541, -0.139094, -0.303436, 0.181532, -0.420825, 0.207969, -0.649702, -0.632246, -0.922756, -0.553966, -0.456005, -0.575313, -0.116762, 0.028056, -0.462826, -0.546873, -0.541345, 0.241699, -0.037006, 0.457691, 0.074282, -0.177898, 0.180116, -0.146268, 0.175106, 0.068884, -0.111705, 0.468838, 0.364176, -0.078292, -0.054915, 0.301103, 0.08553], [0.088726, -0.1922, 0.820167, -0.139657, 0.192402, 0.12731, 0.24915, 0.23478, -0.042131, 0.059229, -0.047547, -0.112433, 0.431716, 0.246146, 0.244857, -0.534239, 0.211312, 0.508024, -0.074718, -0.608069, 0.216629, -0.418199, -0.300257, 0.55377, 0.625806, 0.672728, 0.075238, 0.363687, 0.095735, 0.327617, -0.003029, 0.481824, -0.42786, 0.208165, -0.223478, -0.181289, -0.343575, 0.152889, -0.023607, -0.245263, 0.242749, -0.138618, 0.275694, 0.106559, 0.376106, 0.044494, 0.152977, 1.255427]]
SEQ2_BH = [-0.232965, 0.177843, 0.218474, -0.098731, 0.137955, 0.288582, 0.077193, 0.20163, -0.144289, -0.281921, 0.375381, 0.131314, 0.22398, 0.184749, 0.018177, -0.194805, 0.113556, 0.646095, 0.459123, 0.538415, 0.333513, 0.755185, 0.857592, 0.473622, 1.047179, 0.376183, 0.658755, 0.573938, 0.802198, 0.399893, 0.534758, 0.550386, 0.201075, 0.32262, -0.152389, -0.1824, -0.314794, -0.210673, 0.278922, 0.193893, 0.297255, 0.233263, 0.000877, -0.321048, 0.185686, 0.081464, 0.093779, -0.034781]
SEQ2_HEAD = [
    ([[0.04971, -0.190509, 0.295287, 0.107863, 0.14797, -0.171776, 0.191818, -0.014388, 0.022122, -0.09817, 0.114334, 0.036859, -0.065103, 0.017696, -0.173765, -0.077878, 0.051944, -0.043275, 0.122132, 0.087001, -0.036141, -0.280836, -0.218244, 0.266737, 0.074198, -0.168017, 0.225473, -0.009649, -0.052968, 0.058609, 0.024315, 0.032066, -0.183547, -0.137123, -0.122477, -0.032404, -0.017621, -0.293395, 0.0443, 0.052082, 0.163595, 0.09682, -0.120697, -0.137863, 0.066313, -0.001695, 0.024798, -0.05207, -0.021532, -0.064457, 0.076284, -0.077871, -0.148199, 0.059159, -0.028698, 0.101475, -0.109002, -0.036486, 0.007486, 0.126412, 0.467096, -0.00173, -0.067757, 0.076167], [0.017258, -0.180853, 0.013534, 0.022194, -0.174864, -0.084896, 0.064378, 0.00055, 0.075976, -0.133448, 0.112482, 0.024226, 0.01785, 0.023528, -0.087314, 0.069419, 0.049091, 0.046722, 0.036351, -0.146583, -0.09359, 0.069786, -0.008959, -0.150115, -0.013886, -0.020192, -0.106838, 0.042722, 0.023945, 0.009732, 0.01591, -0.009258, 0.050201, -0.187925, 0.113216, -0.057035, -0.133528, 0.099395, -0.029559, 0.083928, -0.004555, -0.001126, 0.048503, -0.150252, -0.05147, -0.116868, 0.076757, -0.118614, 0.167075, 0.080343, 0.134113, 0.161285, -0.110442, -0.1026, 0.206439, 0.018523, 0.070981, -0.154136, 0.015215, 0.157998, 0.005212, 0.053181, -0.030651, -0.070726], [0.133126, -0.181698, -0.026693, 0.016495, -0.139547, -0.164751, 0.142021, -0.061604, 0.111617, -0.095059, 0.174174, -0.087345, 0.037701, 0.070572, 0.022231, 0.05689, -0.150764, 0.12672, 0.009928, 0.044588, 0.042774, -0.064309, 0.062429, -0.174704, -0.112626, 0.047398, -0.240642, 0.080751, 0.141651, -0.178143, -0.121397, -0.009834, 0.032595, 0.023855, -0.039091, -0.114568, 0.033294, 0.089659, -0.070371, -0.134312, 0.15266, -0.120694, 0.0502, 0.098958, -0.107052, -0.076191, 0.055632, -0.04567, 0.003593, -0.109138, -0.065395, 0.071576, -0.112368, -0.111962, -0.023151, -0.175621, 0.022855, 0.071974, -0.113376, -0.021695, -0.096685, -0.010275, -0.067921, -0.183056], [0.04858, -0.184257, -0.281209, -0.101619, -0.156981, 0.1952, -0.145807, -0.020143, 0.022768, 0.029131, -0.087346, 0.14586, 0.188232, 0.003593, -0.015016, 0.092667, 0.144666, -0.186765, 0.163784, -0.190465, 0.20228, 0.002036, 0.156265, -0.087301, 0.117154, 0.102811, -0.041628, -0.189637, 0.107861, -0.105067, -0.229177, 0.031274, 0.153144, -0.048018, 0.074817, 0.144099, 0.154339, 0.108568, -0.117569, -0.040622, 0.018011, -0.177159, -0.092426, -0.078451, 0.074611, -0.047088, -0.106279, 0.035141, -0.127811, 0.034501, -0.064245, -0.03491, -0.134024, 0.091893, -0.032601, 0.183993, 0.137694, -0.051229, 0.136127, -0.084814, -0.262197, 0.056709, 0.121512, 0.085587], [0.06587, 0.189741, -0.192737, -0.088531, 0.171122, 0.052823, 0.118718, -0.03426, 0.125871, 0.130586, -0.167419, -0.052581, 0.313612, 0.153774, 0.090994, -0.011326, -0.177687, 0.058506, 0.231674, 0.113579, 0.178133, -0.019459, -0.015825, -0.044084, -0.141231, 0.007799, -0.150581, -0.115374, -0.279559, -0.219066, -0.326379, 0.007692, 0.136823, 0.02584, 0.041452, 0.113277, -0.006552, 0.297897, 0.060577, -0.207384, -0.088929, -0.069456, -0.085353, 0.037417, -0.034631, -0.046303, -0.018816, -0.096467, -0.083056, 0.058902, -0.220082, -0.031993, -0.012683, 0.035342, -0.159453, 0.031936, -0.023169, 0.095921, -0.207674, -0.158073, -0.383068, 0.056084, -0.071646, 0.065232], [-0.156531, -0.103761, 0.069622, 0.048193, -0.048651, -0.06344, -0.140263, 0.214004, 0.072046, -0.084117, 0.052749, -0.108417, -0.132611, -0.089328, 0.0317, 0.093185, -0.009256, -0.005455, -0.056019, 0.015273, 0.125195, 0.037786, 0.011434, -0.039534, -0.005241, -0.067871, -0.048087, -0.032855, -0.006271, 0.099424, 0.001479, 0.005634, 0.188679, -0.057637, -0.076685, 0.06948, 0.028196, 0.031836, 0.006888, -0.051279, 0.086027, -0.032269, 0.135555, -0.016874, 0.051757, -0.122433, -0.133318, 0.020969, 0.098884, 0.075034, 0.025001, 0.160153, 0.094927, 0.140827, 0.130541, -0.147173, -0.038313, 0.012943, -0.028717, -0.038278, -0.091105, -0.060513, -0.015418, -0.024033], [-0.059496, 0.173387, -0.189208, -0.202472, -0.235533, -0.165167, 0.153534, -0.160226, 0.010096, 0.073434, 0.162745, 0.403997, -0.025938, 0.115955, 0.154547, 0.181299, -0.165399, -0.141929, 0.116246, 0.162865, 0.031771, 0.250313, 0.268038, 0.003445, -0.114056, -0.296226, -0.194519, -0.025434, 0.070969, -0.067519, -0.037721, 0.203721, -0.14258, 0.150409, -0.060927, 0.292831, -0.084163, 0.214329, 0.270707, 0.169584, -0.062637, -0.208899, 0.024053, 0.144619, -0.119542, -0.213075, 0.14312, -0.071322, 0.079403, -0.159708, 0.149973, 0.274833, -0.184626, -0.221691, -0.173261, -0.165992, -0.044295, 0.037324, -0.303928, 0.243672, -0.040339, -0.249542, -0.053476, -0.207848], [-0.051574, 0.010215, -0.090505, 0.094075, -0.028634, -0.104226, -0.034881, 0.088905, 0.0047, 0.154191, -0.118885, -0.048911, -0.101484, 0.216737, -0.102382, 0.065407, -0.004616, 0.151612, -0.052147, -0.107121, -0.013042, -0.079337, -0.115373, -0.158867, -0.043107, 0.063255, 0.004794, 0.034224, 0.046311, -0.049346, 0.085918, 0.036616, 0.006732, 0.058218, 0.127597, -0.087041, 0.078177, 0.021005, 0.051692, 0.011597, 0.121667, -0.060085, 0.055417, -0.014383, -0.065809, -0.016846, 0.057216, 0.046605, -0.059822, 0.156482, -0.070254, 0.059976, -0.001509, -0.112872, 0.023081, -0.139634, -0.09414, -0.000362, -0.073181, 0.042435, -0.147982, -0.200748, -0.098381, 0.101106], [0.015968, 0.04706, 0.148829, -0.197939, -0.138716, -0.092995, -0.04725, -0.2066, 0.154169, 0.021348, 0.033266, 0.217441, 0.213072, -0.186944, -0.046246, 0.020206, 0.005234, -0.031457, 0.007053, -0.087679, 0.187232, 0.13137, 0.027608, -0.084465, -0.126299, -0.014983, -0.172195, 0.171363, -0.018043, -0.113856, -0.000211, -0.221106, -0.428727, 0.04316, 0.043427, 0.054537, -0.005225, 0.23513, 0.014669, 0.231736, 0.291663, -0.000959, 0.299314, -0.241499, -0.285, -0.091454, 0.132664, 0.006405, 0.137556, -0.166444, 0.163387, -0.040886, 0.008393, -0.093977, -0.087862, -0.185235, -0.02709, -0.067745, 0.024071, 0.226695, 0.070135, -0.182902, 0.130041, -0.117735], [-0.356528, -0.16518, 0.02947, 0.057694, 0.02169, 0.010632, 0.008786, 0.055169, -0.010272, 0.113738, -0.06629, 0.065323, 0.030823, -0.360682, 0.066096, -0.360306, 0.102335, -0.175988, -0.136953, -0.026004, 0.147483, -0.117306, -0.071902, -0.076256, 0.022106, 0.041641, -0.212761, -0.234698, -0.014977, 0.16737, 0.115083, 0.10014, -0.029155, -0.028209, -0.194368, -0.281214, 0.029774, -0.045998, 0.159282, 0.121911, -0.032302, -0.265256, 0.113578, -0.103385, 0.106151, -0.224633, -0.068319, 0.250224, -0.040328, 0.054837, 0.046927, 0.037342, -0.343592, 0.107149, 0.121305, 0.019764, -0.004481, 0.183066, -0.07045, -0.112586, -0.192344, 0.038258, 0.116572, -0.080489], [0.214331, -0.045245, 0.070196, -0.115076, 0.170053, -0.080549, 0.087095, 0.175703, -0.008946, 0.025442, -0.207637, -0.045963, -0.139746, 0.106684, -0.052205, -0.253158, 0.167693, -0.054762, 0.080586, 0.03608, -0.126123, -0.045172, 0.036932, 0.130713, -0.154626, -0.160161, 0.025884, 0.076022, 0.034305, 0.192693, 0.074302, 0.126567, -0.173967, 0.033686, -0.0441, -0.045448, 0.013794, 0.099522, 0.001649, 0.091103, 0.069782, 0.128154, -0.088502, 0.062567, 0.017664, 0.127385, -0.125821, -0.084504, 0.085428, -0.136845, 0.029062, 0.19854, 0.025015, 0.009437, 0.118545, 0.110515, -0.049811, -0.102223, -0.112865, 0.04535, 0.139444, -0.109807, 0.049179, 0.013037], [0.127759, 0.02778, -0.137099, 0.075433, 0.079735, -0.069812, -0.011711, 0.007018, 0.203815, -0.264528, 0.141249, 0.129634, 0.08733, -0.059905, 0.153125, -0.211349, -0.009773, 0.099067, 0.036829, -0.117205, 0.319319, 0.134736, 0.111816, 0.05395, 0.140539, -0.017578, -0.159191, -0.07924, -0.0049, -0.059486, -0.258035, -0.119322, -0.061539, -0.25849, -0.048631, -0.192539, -0.181233, -0.072351, 0.288125, 0.059159, 0.434217, 0.225731, 0.177317, -0.294125, -0.141548, 0.062109, 0.186769, -0.066522, -0.032674, -0.287185, 0.233175, -0.044848, -0.455098, -0.098689, -0.090028, -0.006177, -0.116935, 0.04873, -0.072993, 0.442468, -0.186206, -0.233945, 0.22268, -0.065109], [0.093835, 0.029579, 0.048883, -0.23705, -0.035572, 0.107903, 0.041576, -0.049115, -0.043601, -0.018129, 0.017288, 0.041685, 0.146451, 0.07672, -0.076947, -0.071908, 0.121539, 0.081566, -0.103611, -0.176297, 0.006732, 0.0911, -0.100455, -0.171753, 0.025961, 0.107649, -0.062233, 0.180146, -0.004007, -0.025927, 0.081616, -0.056903, -0.195998, -0.115615, -0.087672, -0.09414, -0.013553, 0.092454, 0.024727, 0.018452, -0.002336, 0.248504, -0.115975, 0.096098, 0.003058, 0.175536, -0.067004, 0.254859, 0.160729, -0.162474, -0.018548, -0.16234, 0.068083, -0.052079, 0.079733, 0.083479, -0.096614, 0.013125, 0.082903, 0.186789, 0.070116, 0.027323, -0.012341, -0.040003], [-0.251288, -0.088134, -0.102915, -0.042166, -0.257805, 0.016963, 0.002813, 0.135868, 0.047567, -0.035778, -0.06868, -0.054998, -0.071937, 0.09914, -0.049186, 0.127122, 0.044692, -0.077526, 0.102819, -0.082967, -0.019138, -0.214552, -0.097543, -0.015212, -0.036667, 0.139457, -0.025227, -0.107395, 0.09435, -0.273042, -0.00064, 0.049643, -0.083991, 0.135667, -0.136874, 0.029109, -0.082322, -0.122924, -0.081115, -0.223605, 0.044012, 0.148495, -0.042138, 0.395117, -0.165623, 0.00993, 0.080733, 0.050127, 0.020258, -0.050926, -0.216414, -0.102879, 0.008116, -0.022077, 0.053938, -0.083727, 0.056188, 0.16746, -0.094264, 0.200444, 0.113987, -0.039334, -0.096763, -0.179033], [0.082514, 0.095476, 0.189083, -0.004456, 0.040317, 0.017628, -0.002226, 0.128011, -0.097164, -0.216631, -0.095795, -0.206218, -0.20211, 0.006291, 0.035507, -0.160208, 0.110623, -0.401378, 0.023168, -0.149877, 0.156555, -0.077668, -0.073707, 0.000379, 0.190041, 0.003245, -0.034287, -0.064083, -0.344167, 0.075836, -0.0032, -0.151782, -0.007185, -0.14873, -0.019562, -0.03347, 0.089309, -0.037891, 0.310025, -0.282846, 0.045731, 0.054104, -0.224328, 0.086731, 0.090364, -0.068137, -0.07944, -0.112099, 0.165453, 0.05339, -0.158025, 0.045039, -0.009755, -0.439054, -0.017989, 0.19583, 0.167295, -0.205683, 0.248848, -0.009872, 0.025811, 0.152853, 0.087134, -0.126787], [0.047495, -0.204639, 0.107249, -0.146973, 0.158951, -0.369345, 0.310541, -0.133803, 0.002494, -0.266226, -0.265325, -0.124177, 0.065675, -0.04798, 0.035734, 0.023163, -0.141765, -0.043718, 0.149309, -0.23973, 0.148594, -0.081617, -0.145943, 0.039657, -0.091048, -0.182584, 0.054378, -0.053943, -0.375576, 0.047054, -0.105949, -0.199426, -0.015349, -0.058756, 0.042206, 0.042676, -0.201584, -0.242907, -0.023429, -0.020362, -0.069069, -0.013548, 0.009251, -0.08144, -0.190225, -0.076351, -0.086839, -0.091136, -0.218367, -0.0494, -0.123843, 0.189347, 0.012311, 0.212261, -0.034318, 0.010761, 0.104423, 0.03868, 0.03453, -0.006097, 0.242262, 0.173729, 0.136692, 0.280467], [0.041767, 0.126465, 0.020984, -0.261062, 0.024581, 0.018131, 0.23272, -0.195364, -0.128658, -0.110572, -0.083744, 0.052601, 0.18037, 0.033702, 0.019397, 0.078992, -0.114387, 0.4516, -0.026516, 0.158729, 0.381114, -0.222155, -0.249603, 0.252356, -0.137172, 0.034974, 0.22438, -0.045361, 0.267961, -0.11367, 0.186701, -0.259319, -0.078698, -0.089018, -0.097752, 0.136285, -0.10653, -0.280183, -0.247002, 0.000745, 0.060185, 0.026475, -0.133761, -0.007672, -0.021772, -0.029113, -0.033421, 0.310235, -0.292617, -0.13268, 0.110301, 0.000378, -0.040279, 0.296152, -0.008248, 0.0884, 0.206052, 0.074087, -0.462151, 0.071362, -0.022829, -0.02288, 0.278862, 0.234538], [0.211886, 0.35262, 0.037848, -0.135449, 0.028232, 0.024415, 0.359744, -0.027195, 0.217586, -0.030777, -0.238448, -0.057912, -0.116803, 0.28951, -0.211435, 0.089326, 0.217015, 0.169219, 0.020636, -0.143014, -0.197577, 0.000517, -0.125624, 0.062107, 0.056131, -0.369573, 0.497695, 0.102538, -0.084073, -0.041293, 0.030758, 0.141916, -0.007236, -0.13428, -0.17859, 0.083912, 0.266619, -0.064105, 0.052455, -0.146654, 0.050376, 0.040598, -0.136924, -0.046213, 0.068882, -0.009363, -0.153739, -0.03301, -0.136817, 0.048244, -0.025988, -0.161236, -0.170246, -0.027124, -0.046104, 0.113967, 0.044868, -0.073322, 0.146064, 0.12577, -0.431214, 0.135806, -0.015838, -0.02372], [0.03435, 0.022816, -0.041784, 0.226101, -0.067769, 0.03963, 0.101621, -0.019056, -0.038261, 0.145374, 0.018334, -0.087109, -0.086527, 0.067616, -0.011004, -0.009991, 0.007426, -0.011278, -0.088122, 0.032547, 0.026605, 0.024729, -0.074445, 0.084603, -0.132728, -0.040286, -0.091139, -0.093271, 0.156679, -0.115366, 0.0428, -0.166233, -0.047733, 0.071316, 0.062671, 0.060769, 0.013337, -0.115319, -0.095235, 0.065567, 0.053645, -0.153726, 0.041221, -0.053397, 0.003732, 0.150204, 0.001787, 0.051826, -0.000362, -0.074576, -0.037475, 0.000524, 0.073866, 0.157294, -0.077337, -0.154254, 0.003112, -0.135124, -0.005469, -0.231102, 0.007199, 0.061012, -0.136912, 0.098599], [0.074362, -0.011834, 0.015856, -0.206593, -0.121851, -0.042866, 0.077131, 0.100198, 0.062682, 0.137955, -0.055511, 0.256553, 0.200381, 0.010369, -0.028443, -0.164511, -0.134673, 0.027418, 0.125064, -0.096823, 0.151823, -0.025861, 0.296703, 0.15413, 0.072756, -0.092893, 0.030863, 0.143591, 0.234407, -0.2808, -0.016606, -0.021021, 0.160527, 0.074332, -0.015601, 0.120579, -0.281114, -0.007506, -0.104896, 0.303615, -0.092177, -0.170967, 0.436765, 0.151323, -0.33599, 0.011758, 0.25887, 0.088557, 0.227543, -0.092218, 0.232502, 0.184761, -0.091247, 0.277784, -0.317215, -0.418326, -0.451917, 0.201329, 0.010735, -0.071569, 0.023962, -0.336693, 0.075415, 0.059883], [0.073041, 0.045402, -0.011842, -0.228379, 0.161173, -0.076054, 0.065188, 0.034564, 0.064195, 0.165739, -0.296804, -0.025571, -0.070158, -0.169104, 0.055611, -0.187259, -0.164998, 0.023728, -0.069842, -0.132863, -0.010194, -0.007443, 0.011436, -0.175036, 0.07171, 0.217977, 0.069701, 0.169585, 0.26432, -0.076935, 0.048332, -0.222658, -0.023509, -0.003553, -0.185438, -0.094113, 0.062434, 0.124559, 0.340984, 0.152718, -0.014575, 0.169055, 0.241071, 0.15687, 0.037485, 0.016626, -0.013336, 0.010573, 0.121667, 0.066985, 0.018523, -0.29857, 0.052484, -0.181912, -0.013225, -0.41987, -0.259065, -0.2739, 0.046342, 0.022307, 0.153594, -0.118701, 0.053169, 0.262768], [-0.048598, -0.046301, -0.113277, 0.380904, -0.064232, 0.131794, -0.003708, 0.381157, 0.266619, -0.061565, 0.013787, 0.46283, -0.241786, -0.005136, -0.120451, -0.142143, -0.228168, -0.236932, 0.041434, -0.024588, -0.040064, -0.043505, 0.410819, 0.045177, -0.198116, 0.072901, 0.116768, -0.058665, 0.112095, 0.079494, 0.100189, -0.099537, 0.022504, -0.113184, -0.193806, 0.0273, -0.218858, -0.069053, -0.113184, 0.047984, 0.035935, -0.135623, -0.117639, 0.05473, 0.081193, 0.311007, -0.200253, 0.000899, -0.376819, -0.048565, 0.019736, -0.043713, 0.101448, -0.23871, -0.163167, -0.17575, -0.062327, 0.202583, -0.285112, 0.133424, 0.074749, 0.228742, 0.124081, 0.494722], [0.099304, -0.011913, -0.292982, 0.191841, 0.227385, 0.249626, 0.178659, -0.101554, 0.123094, 0.076388, -0.018376, 0.076681, -0.087939, -0.072035, 0.108871, -0.031399, 0.187324, -0.055034, 0.214367, 0.109253, 0.089711, 0.20533, 0.156484, 0.134767, -0.116578, -0.00139, -0.144892, 0.093913, -0.061484, -0.170865, 0.055204, 0.257783, -0.328013, 0.06009, 0.14033, -0.083533, 0.06683, -0.032745, 0.228921, -0.030359, 0.134661, 0.21398, -0.121974, -0.149244, -0.045113, 0.272301, -0.038376, -0.198194, -0.2141, -0.23265, 0.149737, -0.085504, 0.084819, 0.092091, -0.112908, -0.340439, -0.21943, -0.002371, 0.057223, 0.081784, 0.277641, 0.04409, 0.240571, -0.010307], [0.096225, 0.184485, 0.468153, 0.129508, -0.095332, -0.133454, 0.178855, 0.093999, 0.183712, 0.225218, 0.296349, 0.194601, -0.017792, -0.413526, 0.182404, -0.000138, -0.16777, -0.155472, -0.077547, -0.018358, 0.083104, 0.099563, 0.041309, 0.173628, 0.068902, 0.152649, 0.028617, 0.098598, -0.139606, 0.080911, 0.106848, -0.01126, -0.285493, -0.064849, 0.028886, -0.245923, 0.163357, 0.249696, 0.110161, -0.105106, -0.044132, -0.021668, 0.396834, -0.42079, 0.096758, 0.088944, 0.172095, -0.094698, -0.212385, 0.069746, 0.181646, 0.348818, -0.419564, 0.255371, 0.234663, 0.118321, -0.068668, 0.188186, 0.065178, -0.176976, -0.312663, 0.229078, -0.0098, 0.065578], [0.013993, -0.168615, 0.002395, 0.077001, 0.126668, 0.091575, -0.05142, 0.017388, 0.199662, 0.086761, 0.083362, -0.05138, -0.082146, -0.061484, 0.052549, -0.061058, 0.150143, -0.128343, -0.079474, -0.373526, 0.13302, 0.004653, 0.093609, -0.349315, 0.049861, 0.105506, 0.298461, -0.067167, 0.054098, 0.085807, 0.10182, 0.218206, -0.495031, -0.053294, -0.451103, -0.018388, -0.046245, -0.368626, -0.078084, 0.085725, 0.151773, 0.225569, 0.036279, -0.199406, -0.018271, 0.356401, 0.0642, -0.28054, 0.055037, -0.089859, 0.001363, -0.397668, -0.065792, -0.089517, -0.187154, -0.178811, -0.029083, 0.12332, -0.148351, 0.013529, -0.097579, -0.046753, 0.285548, -0.264805], [-0.190633, 0.019955, -0.106679, 0.002875, 0.0239, -0.246891, 0.018411, -0.02136, -0.026392, -0.019226, 0.091797, -0.291491, -0.066234, 0.166103, -0.009066, 0.236684, 0.055299, 0.048503, -0.104513, 0.135155, 0.199102, 0.123642, -0.174958, -0.074206, -0.052988, -0.148241, 0.050716, 0.04512, 0.129423, -0.009665, -0.177557, 0.045964, 0.06794, -0.018419, 0.23911, 0.097886, -0.309959, -0.061252, -0.019186, -0.186141, -0.033806, 0.065911, -0.087059, 0.111479, 0.036994, -0.297035, -0.167511, 0.207503, -0.243543, 0.069304, 0.171621, -0.159182, -0.107188, 0.140181, -0.008426, -0.138631, 0.035024, 0.089758, 0.00738, 0.090529, 0.051952, -0.169002, 0.025713, -0.2854], [-0.152348, 0.085279, -0.039783, 0.34647, -0.106673, 0.082139, 0.222756, 0.405631, 0.159247, 0.124051, -0.022303, 0.03217, -0.09617, -0.036044, 0.135806, -0.099025, -0.201477, 0.119425, 0.290105, 0.302078, 0.065076, -0.047432, 0.005587, 0.199388, -0.155153, 0.163469, -0.094118, -0.182406, 0.055983, -0.097389, 0.152766, -0.037692, -0.021174, 0.069269, 0.20802, -0.176983, 0.127091, 0.137929, 0.512832, 0.04399, 0.014161, 0.192744, -0.128108, -0.088107, -0.074302, -0.06527, -0.052114, -0.030235, 0.122432, 0.008348, -0.14107, -0.036976, 0.103552, 0.405815, -0.010196, 0.176062, -0.20894, 0.214953, -0.088823, 0.062064, -0.055756, 0.117804, 0.149353, 0.297086], [-0.057682, -0.167199, 0.00118, 0.338103, 0.133249, -0.017414, 0.386839, -0.06723, -0.001198, 0.013939, 0.076299, 0.077286, 0.117167, 0.130368, -0.003531, -0.065709, -0.082192, 0.0254, 0.225726, 0.119938, 0.213849, 0.29945, 0.092905, 0.148354, -0.069354, 0.151098, 0.179995, -0.134026, -0.146269, -0.314495, 0.077746, 0.066748, -0.029717, 0.072168, 0.069439, -0.129836, 0.083375, -0.000114, 0.116223, -0.129739, 0.269179, 0.258638, -0.120194, -0.240327, -0.047323, -0.033067, -0.044116, 0.177805, -0.115356, 0.086345, 0.059462, 0.140929, 0.23692, 0.083247, -0.14968, 0.216438, 0.07818, 0.000257, -0.17508, 0.365707, 0.183915, 0.274808, 0.179944, 0.256398], [0.133974, 0.09167, -0.126116, 0.044918, 0.170484, 0.012868, -0.240948, 0.238053, -0.076319, 0.059047, -0.053108, 0.415671, -0.06744, -0.04987, -0.176765, -0.139166, 0.014345, -0.225854, -0.167074, 0.143465, -0.084142, -0.073343, 0.304779, -0.310168, -0.099613, 0.267308, -0.173817, -0.172999, 0.010919, 0.078312, -0.074089, -0.053926, 0.298958, -0.02389, -0.039208, -0.373708, 0.190249, 0.068618, 0.229232, 0.283546, -0.053695, -0.285257, 0.021939, -0.133958, 0.004686, -0.12463, 0.069, -0.009139, 0.315021, -0.025011, 0.105147, -0.164732, -0.151286, -0.328616, -0.081678, 0.224262, 0.174857, -0.126597, 0.099182, 0.126061, -0.127789, -0.199822, 0.189116, 0.065883], [-0.052976, 0.107029, -0.038961, -0.281586, 0.09702, -0.325967, -0.241199, -0.140714, -0.238997, 0.017687, 0.033638, -0.136467, -0.145693, 0.117295, -0.178854, 0.107537, 0.058462, 0.026825, -0.219068, 0.046663, -0.215356, -0.075121, -0.306236, 0.010064, -0.112328, -0.233958, -0.18088, -0.045977, 0.081386, 0.204754, -0.227539, -0.018133, 0.133196, -0.18642, -0.038327, 0.150532, -0.130785, -0.268723, -0.034775, -0.271737, -0.224603, -0.031758, -0.242664, 0.107714, -0.087524, -0.1481, -0.08803, -0.07821, -0.228766, -0.110193, -0.156315, -0.08813, -0.251006, -0.336724, -0.183472, -0.162687, -0.174607, -0.051671, 0.133511, -0.103216, 0.141047, -0.352514, 0.022846, -0.015168], [0.065864, -0.376269, -0.150442, -0.051602, 0.203957, -0.059921, -0.06801, -0.08317, 0.163589, -0.340069, 0.107056, -0.379154, 0.064118, -0.063662, -0.288572, -0.30095, 0.124314, 0.018116, 0.012284, -0.291269, 0.121444, 0.055762, -0.187682, -0.2307, -0.325329, -0.547203, -0.021115, 0.102069, 0.092901, -0.209564, -0.073191, 0.238707, -0.360392, -0.344022, -0.146938, -0.102049, -0.619557, -0.230165, -0.082608, -0.258969, 0.133279, 0.482528, 0.117323, 0.166985, -0.17016, 0.221891, -0.191952, 0.126016, -0.043147, -0.14707, -0.150167, -0.324447, -0.065935, 0.014372, -0.431909, -0.271468, -0.13615, 0.081212, -0.008239, 0.217117, -0.06107, -0.278286, -0.063397, -0.151302], [-0.02369, 0.033933, 0.220877, 0.35539, 0.103073, 0.229727, 0.081335, -0.162622, 0.070612, 0.640818, 0.123571, -0.108873, 0.062524, 0.132608, 0.419908, 0.22014, 0.513469, 0.113941, 0.121446, -0.629936, 0.155919, 0.332264, -0.11675, 0.214191, 0.425373, 0.519706, 0.474162, -0.128785, 0.043819, 0.130271, 0.678388, 0.371858, -0.271941, 0.489328, -0.268789, 0.166817, 0.343559, 0.142852, -0.078072, -0.215154, -0.022344, -0.303881, -0.067279, -0.312361, 0.118916, 0.168625, -0.085324, 0.221292, 0.163001, 0.565657, 0.044514, 0.120656, 0.275022, 0.303555, 0.309903, -0.119913, 0.262157, -0.158479, -0.162854, 0.060832, 0.303289, -0.261893, 0.203094, -0.075968], [0.083305, 0.051764, -0.060863, -0.294965, -0.199195, 0.145804, -0.283578, 0.303653, -0.374212, 0.172096, -0.105299, -0.280165, -0.11441, 0.292038, -0.033978, -0.184244, 0.311537, -0.121565, -0.354395, -0.032174, -0.094393, -0.201498, 0.001465, -0.067983, 0.055796, 0.12877, -0.065373, 0.230854, 0.069507, 0.14622, -0.178922, 0.155333, -0.032605, -0.054437, -0.392658, 0.01116, 0.428006, 0.025764, 0.037443, -0.073819, -0.057575, 0.032742, -0.058389, -0.441276, 0.303729, 0.097793, -0.245676, -0.228795, 0.172891, 0.139745, -0.212781, 0.015056, -0.055433, 0.050436, 0.08077, -0.22641, 0.025451, -0.189403, -0.102324, -0.032482, -0.013434, -0.392122, 0.092764, -0.178666], [0.032603, -0.221037, -0.099264, 0.116301, -0.088796, -0.03416, -0.059727, 0.011661, -0.154221, 0.12852, 0.072281, 0.040151, 0.194262, 0.229313, -0.002741, -0.169248, -0.167355, -0.120292, 0.145606, -0.112687, 0.021287, 0.23023, -0.047477, -0.167387, 0.071935, 0.21804, 0.04149, 0.141787, 0.225305, 0.023772, 0.362427, 0.120197, 0.067155, 0.176797, -0.014024, 0.164272, -0.193944, 0.088508, 0.03869, 0.170442, -0.029477, -0.309884, -0.102356, 0.068441, 0.151556, -0.050503, -0.02436, 0.307127, -0.009351, -0.023263, -0.132332, -0.111562, 0.210922, -0.076878, 0.095641, -0.110257, 0.066681, 0.20801, -0.129946, 0.119264, 0.19053, -0.036463, -0.102892, 0.245291], [0.209017, 0.074857, 0.011909, 0.068764, 0.109888, 0.123427, 0.216772, -0.449287, 0.396011, -0.1237, 0.056729, -0.24815, 0.119836, -0.274257, -0.07362, -0.018015, -0.095947, 0.2717, 0.109938, 0.061194, 0.147323, -0.197982, 0.256586, 0.296272, 0.069852, -0.256104, 0.071122, 0.014772, -0.162818, -0.08548, 0.103092, 0.050663, -0.16189, 0.17295, 0.37854, -0.331696, -0.033877, 0.252802, 0.142806, 0.081677, 0.243076, 0.34319, 0.344381, 0.136918, -0.034723, 0.370205, 0.121645, 0.084197, -0.078501, -0.090852, 0.217625, 0.139118, 0.090326, -0.003955, -0.161334, 0.152405, 0.033126, -0.088487, -0.043232, 0.175014, 0.078696, -0.202291, 0.165088, -0.245716], [-0.193589, -0.043912, -0.258219, 0.128404, 0.236051, -0.123706, 0.091567, -0.024496, 0.009189, -0.428117, -0.005936, -0.174507, -0.271548, 0.255936, 0.033021, -0.276603, 0.052323, 0.035331, 0.131532, -0.062535, -0.022061, 0.041301, 0.319757, 0.155454, -0.024108, -0.007485, 0.022483, 0.224373, 0.116377, -0.0076, 0.023339, -0.105688, 0.241615, -0.001817, -0.118716, -0.046564, -0.047146, 0.16973, 0.162753, -0.168913, 0.013757, 0.154073, -0.093593, 0.251684, 0.232455, -0.102008, -0.023817, 0.354778, 0.258022, -0.029178, -0.061354, -0.0399, 0.141593, 0.111909, -0.133018, -0.014926, -0.251615, 0.05188, 0.208386, 0.224392, 0.22645, -0.343028, 0.045465, -0.090893], [0.237045, -0.129779, -0.111544, 0.592223, 0.234217, 0.065226, 0.140744, 0.245097, -0.196835, 0.12488, -0.034439, 0.149613, -0.146403, -0.03179, 0.122018, 0.182831, 0.003379, -0.25185, -0.02341, -0.157653, 0.00208, 0.205576, 0.063651, 0.016287, 0.253571, 0.233832, 0.143755, 0.139028, 0.043177, -0.02236, 0.190863, 0.172132, 0.314495, 0.192147, -0.070178, 0.160164, -0.165471, 0.155226, 0.052076, 0.133197, 0.092011, 0.122055, 0.043649, -0.125535, 0.047873, 0.148528, -0.032197, -0.1697, -0.109122, 0.072319, -0.003569, 0.09106, 0.190083, -0.204306, 0.085305, -0.174508, 0.095192, -0.106372, 0.174658, -0.024766, 0.080715, -0.016027, 0.03019, 0.141922], [0.05524, -0.101432, -0.021667, -0.122601, -0.318877, -0.039361, -0.137118, 0.240619, 0.015091, -0.097486, -0.366563, -0.145382, -0.216856, 0.082755, 0.131812, 0.046991, 0.044674, -0.133488, -0.444416, -0.197891, -0.322061, -0.038241, -0.661258, -0.371395, 0.031196, -0.125518, -0.112361, 0.07928, 0.228013, -0.090222, 0.012408, -0.187608, 0.273566, -0.119705, -0.348644, 0.065676, -0.048155, -0.435333, -0.281409, 0.125692, -0.298266, 0.19313, -0.341811, -0.133194, -0.032283, -0.281253, -0.487701, -0.008404, 0.215951, -0.067477, -0.208598, -0.404525, -0.130837, -0.266192, -0.104171, 0.006658, 0.032829, -0.341415, -0.197788, 0.069876, 0.001779, -0.006147, -0.17436, 0.000351]], [-0.152694, 0.044947, 0.2312, -0.247483, -0.010912, -0.191683, -0.060154, -0.114133, 0.033286, 0.180157, -0.065063, -0.18581, 0.136502, -0.103647, -0.035758, -0.10748, 0.021269, 0.251483, -0.306061, -0.50336, -0.167175, -0.083529, -0.051774, -0.13931, 0.121284, -0.093233, -0.2574, -0.091145, -0.052236, 0.051431, -0.092718, -0.005921, -0.433957, 0.072137, -0.048796, -0.19775, 0.044904, 0.006238, -0.323289, -0.123512, -0.326922, -0.00712, 0.154305, -0.08317, -0.072065, 0.119249, 0.071879, -0.220135, -0.110279, 0.106771, 0.110207, -0.081294, -0.336168, -0.162594, 0.14498, -0.156887, 0.147984, -0.072604, -0.111311, -0.29968, -0.275416, 0.184755, -0.212353, -0.168719]),
    ([[0.014999, -0.139066, -0.141143, 0.047187, -0.156661, 0.08814, 0.067517, 0.070875, -0.138591, -0.085334, -0.082611, -0.035534, 0.013668, -0.027359, -0.035458, 0.041817, 0.047124, 0.026227, 0.108789, -0.024052, -0.209314, -0.143578, 0.011734, -0.130804, -0.011684, 0.100408, -0.279287, 0.062267, -0.226622, 0.024132, -0.09629, 0.072224], [-0.009002, -0.065068, 0.080521, 0.03365, 0.11134, -0.031677, 0.097463, 0.053232, 0.254361, 0.085262, 0.034232, 0.006937, 0.051904, 0.177287, -0.112551, 0.194504, 0.259289, 0.121657, 0.086343, 0.213036, 0.003675, 0.015598, 0.320698, -0.180889, 0.074814, 0.154569, -0.094099, 0.166993, -0.061986, -0.086769, 0.111149, 0.120098], [-0.281618, 0.085545, 0.110973, -0.000519, 0.054514, 0.037841, -0.479111, 0.095274, -0.009556, 0.129392, -0.101762, -0.063602, 0.074967, 0.136313, -0.022388, 0.211822, 0.062807, 0.082525, 0.223399, 0.145451, 0.031179, -0.074051, 0.118409, -0.461456, -0.089898, 0.124485, 0.138686, 0.079445, 0.094097, -0.256049, -0.010247, 0.20306], [-0.077798, 0.054874, 0.180019, 0.069808, 0.285002, -0.139926, 0.069021, -0.046704, -0.065548, 0.180461, 0.26344, 0.265578, 0.119251, -0.10264, 0.138705, -0.135716, 0.1207, -0.088718, 0.07611, 0.202345, 0.10398, 0.153526, 0.184462, 0.487883, 0.017336, -0.124967, 0.094993, -0.143667, 0.149676, 0.073038, 0.125571, 0.146887], [-0.030071, -0.011847, -0.061698, -0.057226, 0.034336, 0.046663, 0.153568, -0.104162, -0.017528, -0.16832, -0.281292, -0.01012, -0.049252, -0.096423, 0.152625, -0.078506, 0.066709, -0.095632, -0.036928, 0.145611, -0.165577, -0.155403, -0.03506, 0.018283, -0.081598, -0.07329, -0.024945, -0.110538, -0.092581, 0.035407, 0.071889, 0.118485], [-0.133699, 0.018335, 0.124923, 0.041526, 0.071816, -0.077005, -0.24511, -0.020911, -0.057616, -0.024063, 0.16474, -0.240864, 0.099345, 0.113053, 0.05302, -0.169342, -0.054666, 0.065406, 0.019458, 0.114047, 0.12388, -0.051625, 0.045811, -0.172937, 0.085031, -0.000462, 0.094267, 0.164462, -0.049979, -0.054335, -0.036568, -0.013097], [0.078036, 0.048993, -0.023707, -0.067315, 0.122896, 0.041105, -0.01882, 0.050447, 0.135636, 0.143699, 0.174866, 0.198606, 0.088372, 0.14651, -0.272919, 0.098236, 0.124281, 0.138238, 0.132705, 0.101723, 0.067817, 0.04125, -0.01813, -0.116263, 0.158156, 0.080808, -0.050671, -0.167599, 0.089133, 0.136447, -0.050594, 0.152928], [-0.007736, -0.076135, -0.010312, 0.063365, -0.059239, 0.035014, -0.236393, 0.075085, 0.030011, 0.079746, 0.219468, -0.181439, 0.047771, -0.008724, 0.13803, -0.104077, -0.140798, -0.103772, -0.065872, -0.045792, -0.092128, -0.030454, 0.068871, -0.078716, -0.31289, -0.158102, 0.007636, -0.067124, 0.058636, 0.023858, -0.080531, -0.10119], [0.064151, -0.157694, 0.081302, -0.068793, -0.018202, -0.069592, -0.07119, 0.072671, 0.042504, -0.079898, 0.054082, -0.040623, -0.069332, -0.076764, 0.043193, 0.032621, -0.048129, 0.183544, 0.048122, -0.017157, -0.135615, 0.176455, 0.026401, 0.163558, -0.000531, 0.063899, 0.015681, -0.071729, -0.126234, 0.152557, -0.140478, -0.065761], [0.123482, 0.044853, 0.117084, -0.04263, 0.131454, 0.084908, 0.100866, 0.042213, -0.17064, -0.099098, 0.119775, -0.228963, 0.168568, 0.020934, 0.015689, 0.176989, 0.034739, -0.005022, -0.07927, 0.038147, 0.131836, 0.151817, -0.158157, -0.287574, 0.046168, 0.180441, 0.136985, -0.188461, 0.049158, 0.1332, -0.07397, 0.092543], [0.158726, 0.003233, -0.029805, 0.04967, 0.18616, 0.110328, 0.042966, 0.057043, 0.160157, 0.017302, -0.065736, 0.008616, 0.166512, 0.055629, -0.177677, 0.023323, 0.120121, 0.110928, -0.008339, 0.149959, 0.151884, 0.032099, -0.006471, 0.243832, -0.09836, 0.104912, -0.027026, -0.15159, 0.18612, 0.112866, 0.100943, 0.137044], [0.315519, 0.069906, 0.117222, -0.047871, -0.05032, 0.074767, -0.028342, -0.025464, 0.1816, -0.191322, -0.280134, -0.009053, -0.035409, 0.096798, 0.20641, -0.074879, -0.013596, -0.12114, 0.02522, -0.146872, -0.039049, -0.020029, 0.137296, 0.25807, 0.191056, -0.112849, 0.028514, -0.179999, -0.251664, 0.14628, -0.160075, -0.074816], [0.085009, -0.075614, -0.159973, 0.074614, -0.285225, 0.009625, 0.197566, 0.049781, 0.154987, 0.159443, 0.068149, 0.071658, -0.028583, -0.108238, -0.241176, 0.004243, 0.052856, 0.049426, 0.105596, -0.210954, 0.052995, -0.162343, -0.110202, -0.096865, -0.0777, 0.057675, -0.149909, -0.083641, -0.079234, 0.033187, -0.100246, -0.060849], [-0.169602, 0.059632, 0.027118, -0.116882, -0.241304, -0.023276, 0.068906, -0.109211, -0.0864, 0.009749, 0.236623, -0.753089, -0.160526, -0.324304, -0.272429, -0.172393, -0.176309, -0.112004, -0.044361, 0.037904, 0.035214, -0.007929, -0.015813, 0.121434, -0.054244, -0.033383, -0.038894, -0.110792, -0.039747, 0.108707, -0.032082, 0.027393], [0.128628, 0.110645, -0.123126, 0.12095, -0.029877, -0.105115, 0.060096, -0.179411, 0.145832, 0.091964, -0.113413, -0.141573, 0.119863, 0.007981, -0.122167, 0.081915, -0.010048, -0.019471, -0.025363, 0.104885, 0.061866, 0.046358, 0.007154, -0.147725, 0.11574, -0.041096, 0.025276, 0.088779, 0.127957, 0.040344, 0.140983, -0.009851], [0.003915, 0.145781, 0.062015, -0.021263, 0.017136, 0.059219, -0.465766, 0.086804, 0.161212, -0.063881, -0.004634, -0.150119, 0.18809, 0.002219, -0.108145, -0.018678, -0.061326, -0.187797, -0.353719, 0.003394, 0.178983, -0.017527, -0.250373, 0.226686, -0.123119, 0.059847, 0.168829, -0.081206, 0.081066, 0.071077, 0.187066, 0.039017], [-0.188037, 0.028443, -0.008736, -0.075276, 0.152438, -0.09116, -0.04436, -0.093041, -0.247035, -0.00256, 0.049377, -0.015411, -0.03016, -0.086753, 0.138487, -0.175887, -0.076247, 0.023105, 0.058275, -0.021866, 0.010507, 0.087528, -0.03867, 0.133146, -0.028211, -0.11876, -0.001756, -0.00815, -0.079176, -0.234932, 0.171723, -0.03303], [-0.065106, -0.002673, -0.051767, 0.004275, -0.175474, 0.130308, -0.061933, -0.084423, 0.068239, -0.244422, -0.055052, -0.086425, 0.033844, -0.214379, -0.400363, 0.11105, -0.057772, 0.043103, 0.070679, 0.009197, 0.088289, -0.024709, 0.114721, 0.044429, 0.062557, -0.069463, -0.212258, 0.077442, -0.024605, 0.025062, -0.028921, -0.083207], [0.103813, -0.10161, -0.114267, -0.101777, -0.034335, 0.08734, -0.111323, 0.060308, 0.129831, 0.343089, 0.212774, -0.161403, -0.00971, 0.057359, -0.346099, -0.03832, 0.043178, 0.067048, 0.129181, -0.115278, -0.110833, -0.008575, 0.141174, 0.124845, 0.153018, -0.028114, -0.035912, -0.038879, -0.010904, 0.040071, 0.128021, 0.128333], [0.117026, -0.280019, -0.246884, -0.216187, -0.167716, 0.027391, 0.246119, 0.300947, 0.214615, -0.10424, -0.122047, -0.003484, -0.25905, 0.239108, -0.150581, 0.04107, 0.003218, -0.085696, 0.075233, -0.105274, -0.225203, -0.073873, -0.16069, 0.06988, 0.302461, -0.146815, -0.233613, 0.214135, -0.097466, 0.197187, -0.030094, -0.058179], [0.197026, -0.055859, -0.171849, -0.095778, -0.255235, 0.07636, 0.100925, 0.030705, -0.066234, 0.165206, -0.009794, -0.017315, -0.059048, -0.148546, -0.041473, 0.168895, 0.113452, 0.143146, 0.021127, -0.293293, 0.002313, -0.284504, 0.112827, 0.138157, 0.029131, -0.022719, -0.193802, -0.085517, -0.036117, 0.131847, -0.112499, 0.104753], [-0.021189, -0.066441, -0.052692, 0.007496, -0.084908, 0.155943, -0.183436, -0.157919, -0.071714, 0.032945, -0.056919, 0.000313, 0.054516, -0.182015, -0.217364, -0.191031, -0.027759, 0.084773, -0.102112, 0.129155, 0.005686, -0.052051, -0.049986, 0.196522, -0.021191, 0.025301, 0.091827, -0.168203, -0.003918, -0.120689, 0.075536, -0.108009], [0.153333, -0.20015, -0.030904, 0.004847, -0.114639, 0.291951, 0.157145, 0.161517, 0.202775, -0.207774, 0.059774, -0.266424, -0.04466, 0.212403, -0.048628, -0.04565, -0.009304, 0.181463, 0.151923, 0.008414, -0.036962, 0.028254, 0.190605, 0.148818, 0.275042, 0.104736, -0.033218, -0.100046, 0.023136, 0.2452, 0.022493, -0.413899], [0.080778, 0.050865, 0.111561, -0.100976, 0.209431, -0.149456, -0.166372, 0.17176, 0.097638, -0.010983, 0.062514, -0.002912, 0.101448, 0.155974, -0.153455, -0.034923, -0.0162, 0.15938, 0.165142, 0.035675, -0.003215, 0.186801, 0.095695, -0.25584, 0.068938, 0.021269, 0.090627, 0.107629, 0.126893, 0.288823, 0.105761, -0.024677], [0.06291, 0.158873, -0.060842, 0.133313, -0.01705, 0.043463, -0.068455, -0.068938, -0.101187, -0.043377, -0.02506, -0.06843, 0.089804, -0.135461, -0.076032, -0.028372, -0.078628, 0.008501, 0.009451, -0.019111, 0.058101, -0.020501, -0.148962, -0.055785, 0.111917, 0.159089, 0.149289, -0.030953, 0.062472, -0.018969, 0.111323, 0.123023], [0.250623, 0.101177, 0.139043, 0.064698, 0.134302, -0.064553, -0.027584, -0.056902, 0.13326, 0.065289, 0.213697, -0.27214, 0.214027, -0.306636, 0.124635, 0.071718, -0.007949, -0.157406, -0.252464, 0.082236, 0.063095, 0.204937, -0.224094, -0.439095, -0.171125, -0.000141, 0.228232, 0.116835, 0.07887, 0.076962, 0.100736, 0.083321], [0.029213, 0.090497, 0.125168, -0.032123, -0.004873, -0.06944, 0.268403, -0.126975, -0.048845, 0.124609, 0.169386, -0.183604, 0.209634, -0.08491, 0.10165, 0.088256, -0.098016, 0.140993, 0.106937, 0.193338, 0.176124, 0.099116, -0.265971, -0.126782, 0.159968, -0.136762, -0.061757, 0.001035, 0.11898, -0.149104, 0.214769, 0.180519], [-0.16374, -0.098392, -0.033716, 0.014974, 0.064029, 0.1496, -0.010887, -0.002726, 0.004474, 0.166808, -0.056954, -0.285222, 0.05222, 0.1517, -0.012093, 0.012399, 0.001687, 0.127765, -0.097668, 0.012098, -0.044509, -0.089104, -0.01079, -0.028866, -0.019088, -0.007057, -0.164984, -0.106911, -0.009483, -0.195285, 0.066344, 0.043123], [-0.364206, 0.056685, 0.104944, -0.08491, -0.032153, -0.122658, -0.14823, -0.196706, -0.105947, -0.106013, 0.023398, -0.832389, 0.079962, -0.091355, -0.018256, -0.00714, -0.020478, -0.041555, -0.429883, 0.00802, 0.050071, 0.02806, -0.249987, -0.089464, -0.34294, -0.164542, -0.028562, -0.087923, -0.134943, -0.517374, -0.097415, -0.173267], [0.016987, 0.050542, 0.019823, 0.010118, 0.023337, -0.002827, -0.62151, 0.095906, 0.097094, 0.071114, -0.111456, 0.214279, 0.008101, 0.054438, 0.038237, -0.222165, -0.165729, -0.18573, -0.140268, 0.027549, 0.11536, 0.082261, -0.08609, 0.038697, -0.163712, -0.08421, -0.061141, 0.117746, 0.037546, -0.160598, 0.106201, 0.020917], [-0.065724, 0.035122, 0.094775, 0.060617, 0.021797, 0.104125, -0.006278, 0.006131, -0.243498, 0.177354, 0.049232, -0.538131, 0.119241, 0.109287, -0.079457, 0.158997, -0.123028, 0.128751, -0.017067, 0.106647, 0.147421, 0.056705, -0.166894, -0.370192, 0.149799, 0.118819, -0.00823, 0.006498, -0.056243, 0.068262, 0.099536, 0.033612], [0.030008, 0.156055, -0.044868, -0.045051, 0.115491, 0.102878, -0.045797, -0.001275, -0.026871, 0.086433, -0.048211, -0.012529, -0.023753, -0.161087, -0.378422, 0.048051, 0.078574, -0.062614, 0.018442, 0.022393, 0.164455, 0.165917, 0.053962, 0.295903, -0.06788, -0.04725, 0.129795, -0.003273, 0.038295, -0.109191, 0.067983, -0.075691], [-0.212501, -0.090884, 0.014495, 0.036987, -0.06357, -0.173982, -0.122128, -0.058898, -0.17488, -0.013972, -0.010339, 0.318402, -0.08987, -0.054572, 0.249907, -0.028778, -0.174357, -0.197199, -0.049718, 0.137621, -0.096371, -0.081725, -0.222013, 0.191227, -0.270177, -0.168364, -0.105762, 0.043972, -0.080675, 0.08745, 0.068297, 0.320634], [0.072115, 0.010254, 0.119776, 0.074897, 0.097896, 0.11725, -0.10931, 0.075532, 0.040197, -0.018061, -0.018484, -0.127142, 0.070173, -0.146572, -0.290691, 0.169092, -0.048586, 0.046809, -0.175991, -0.060166, -0.078575, -0.049654, -0.077022, -0.068527, -0.037388, 0.103515, 0.134927, -0.022339, 0.079359, -0.043786, 0.120558, -0.12588], [0.17388, -0.104914, -0.11778, -0.137882, -0.098921, 0.010076, 0.060639, 0.140233, 0.0721, 0.056607, -0.006704, 0.055142, -0.025608, -0.092374, -0.104116, -0.092047, 0.111678, -0.012902, -0.063041, -0.269151, -0.104896, 0.096918, 0.055208, 0.120201, -0.061486, -0.046082, 0.099927, 0.138076, -0.12451, 0.034204, -0.107921, -0.266511], [-0.342879, 0.15694, 0.059155, -0.083189, -0.084435, -0.228536, -0.298459, -0.142432, -0.058348, 0.062323, 0.05888, -0.480494, -0.037841, 0.004227, -0.127256, 0.006939, -0.059603, -0.362056, -0.364746, 0.071156, 0.048696, 0.021471, -0.212231, 0.030029, -0.223414, -0.068441, -0.096761, -0.212228, 0.008706, -0.10789, 0.028625, -0.023974], [-0.15482, 0.202599, 0.272274, 0.01131, 0.013462, 0.017472, 0.232378, 0.025118, -0.426743, 0.119956, 0.00963, 0.116779, 0.188506, -0.232515, -0.145134, 0.005739, 0.04561, -0.11819, -0.05395, 0.097879, 0.182964, 0.157901, 0.003058, -0.042837, -0.04734, -0.068824, 0.075376, 0.026059, 0.082926, -0.029203, 0.173345, 0.119405], [-0.013706, 0.13502, -0.169467, -0.046046, -0.054234, 0.048853, 0.229289, 0.141965, -0.113632, 0.15818, 0.263323, -0.055091, 0.041136, 0.080016, -0.026908, 0.144104, -0.080694, 0.1415, 0.101882, 0.190281, 0.072696, 0.186105, 0.108201, -0.382911, 0.061219, 0.062058, 0.097976, 0.219822, 0.073223, -0.035941, 0.026058, -0.009816], [0.084171, 0.157508, 0.018339, 0.085053, 0.295664, 0.039071, 0.026352, 0.034765, 0.011166, 0.06379, -0.007081, -0.293422, 0.109241, -0.003674, 0.018476, -0.076611, 0.069977, 0.045097, -0.019, 0.190878, 0.088354, 0.183611, 0.116082, -0.122931, -0.002261, 0.075264, 0.090658, -0.21127, 0.091537, -0.053783, 0.202198, 0.250502], [-0.023971, 0.012541, 0.055183, -0.028246, -0.102154, 0.042893, -0.333861, -0.159722, -0.086958, -0.295289, -0.236307, -0.425208, -0.004514, -0.012785, 0.235303, -0.046091, -0.01572, -0.113006, -0.226196, -0.141767, -0.121508, -0.107609, -0.360073, -0.331391, 0.035179, -0.039574, -0.076392, -0.010095, -0.259798, -0.027128, -0.133587, 0.068911], [0.079789, -0.159906, 0.033837, -0.123254, -0.068076, -0.016403, -0.049476, 0.079924, 0.053467, 0.113297, -0.077529, 0.122335, -0.131846, -0.053073, -0.011181, 0.213277, -0.003514, 0.074229, 0.005462, 0.035875, 0.039027, -0.112495, 0.188953, 0.301611, -0.099809, 0.098488, -0.092395, 0.180405, -0.079427, -0.049983, -0.027129, -0.168677], [0.101098, -0.115594, -0.004965, 0.024064, -0.022644, -0.099947, -0.208635, 0.012782, 0.104161, -0.420888, -0.344317, -0.048115, -0.159217, 0.259821, -0.201206, -0.089342, -0.078711, 0.083495, -0.033408, -0.234914, 0.058571, 0.126916, -0.047999, -0.079768, 0.11833, -0.148626, -0.099637, 0.332512, -0.168536, -0.074056, -0.05703, -0.417867], [-0.134113, -0.103902, -0.042181, 0.046835, -0.328634, 0.089047, -0.01077, 0.092399, -0.05664, -0.225379, -0.024655, -0.237671, -0.110125, 0.027903, 0.235135, 0.041555, -0.033826, 0.09902, -0.132289, -0.098253, -0.213582, -0.016162, 0.042345, 0.042576, 0.008306, 0.112558, 0.110142, 0.167411, -0.164013, -0.017936, -0.273906, -0.226959], [-0.120164, -0.044774, 0.099584, 0.071836, -0.021961, -0.226245, -0.020953, -0.306766, -0.350728, 0.047141, -0.072193, -0.19483, -0.047735, -0.261369, -0.090556, -0.12036, -0.048128, -0.155494, 0.014092, 0.002866, -0.007066, 0.158361, -0.019676, -0.14892, -0.265269, -0.01337, -0.001424, -0.310816, 0.045942, -0.150193, -0.125428, -0.069477], [-0.084251, 0.096097, -0.069609, -0.056378, 0.054143, -0.053938, -0.110785, 0.077516, 0.013555, 0.160102, 0.069532, -0.013737, -0.099178, 0.00454, -0.000386, -0.088677, -0.182206, -0.110612, -0.082405, 0.134896, -0.022433, -0.063584, -0.093375, 0.011059, -0.033644, -0.219391, -0.054455, -0.07362, 0.065445, -0.096747, 0.164539, -0.037527], [-0.418894, 0.043791, 0.026647, 0.084252, -0.033233, -0.051061, -0.312771, -0.00308, -0.17992, 0.089415, -0.006566, -0.056468, 0.015791, 0.04814, -0.039477, -0.259819, -0.161656, 0.097439, 0.019178, -0.068012, 0.012755, -0.014029, -0.059526, 0.270722, -0.015865, 0.071789, 0.016949, 0.038802, 0.025993, -0.400751, -0.06619, -0.034627], [0.054301, -0.019319, 0.075314, -0.194061, -0.052222, 0.062768, -0.096564, 0.010174, 0.090027, -0.099021, -0.170165, 0.182753, -0.000884, 0.157705, 0.030802, 0.070232, 0.127652, 0.058447, 0.082372, -0.22077, -0.192313, 0.115276, -0.027409, -0.104032, 0.003469, 0.105809, -0.055923, 0.025167, -0.131537, 0.022122, -0.073272, -0.282047], [-0.033373, -0.116293, -0.05078, -0.053534, 0.122316, 0.012789, -0.056635, 0.003046, 0.049298, 0.149291, -0.051786, 0.254818, -0.103745, -0.058754, 0.205609, 0.160591, 0.0179, 0.102097, 0.091002, -0.158615, -0.152565, -0.036682, -0.099362, -0.077372, -0.011649, 0.094444, 0.229174, -0.079941, -0.111178, -0.039858, -0.102594, -0.246508], [-0.237457, 0.090932, -0.103455, 0.016901, -0.041434, -0.136659, -0.07541, -0.148332, -0.391505, 0.149947, 0.027874, -0.031206, 0.050617, -0.167018, 0.211225, 0.100852, 0.046642, -0.079494, -0.065774, -0.108469, -0.185404, 0.146394, -0.050348, -0.006592, 0.016811, 0.165066, 0.124831, -0.239418, -0.204323, -0.427927, -0.027436, -0.142104], [-0.11335, 0.08973, 0.141534, 0.112663, 0.020717, 0.146538, 0.111254, -0.065577, -0.012755, 0.07138, -0.144874, 0.058262, 0.131938, -0.276851, -0.053308, 0.18068, 0.18781, -0.077145, -0.114497, 0.124053, 0.120259, 0.198468, -0.001907, 0.113099, 0.082598, 0.117879, 0.160768, 0.059412, 0.172037, 0.139891, 0.023499, -0.041284], [0.118733, -0.08346, 0.016254, 0.1209, 0.095398, 0.176051, -0.024289, -0.129508, -0.117983, -0.143635, -0.49308, -0.187452, 0.072549, -0.111448, 0.157966, -0.022573, 0.175713, 0.059816, 0.043522, -0.061294, -0.024293, -0.073064, 0.11292, -0.251233, 0.032277, 0.02809, 0.02639, -0.109982, 0.037077, -0.077046, -0.06326, -0.015442], [-0.001851, -0.005849, -0.025145, 0.067922, -0.28931, 0.077904, -0.06018, 0.000909, 0.055153, 0.143878, 0.103761, -0.171343, -0.137165, 0.098979, -0.030832, 0.015843, 0.155887, 0.163869, 0.03198, -0.143165, -0.162614, -0.162237, -0.033312, 0.020087, 0.101585, 0.070488, 0.055269, 0.129895, -0.110696, -0.049338, -0.03849, 0.114456], [0.039382, 0.10829, 0.15585, -0.028194, 0.12863, -0.038767, -0.079657, 0.010158, 0.052367, 0.161111, 0.234661, 0.070313, -0.015843, 0.131352, 0.275169, -0.28344, 0.101117, 0.121115, 0.195768, 0.087569, 0.10634, 0.029676, 0.037659, -0.154662, 0.0566, -0.217544, 0.03458, 0.169814, 0.054036, 0.009891, 0.182851, -0.020075], [0.301516, -0.150727, -0.037787, -0.091141, -0.102283, 0.102268, -0.194074, 0.146982, 0.242254, -0.186077, -0.101568, 0.187436, -0.153332, 0.178486, -0.203417, 0.050731, 0.184424, 0.053963, 0.273966, -0.173779, 0.013605, -0.052456, -0.158477, 0.055261, 0.126652, 0.033456, -0.239172, 0.04812, -0.077789, 0.396605, -0.097712, -0.349861], [0.01381, 0.062772, -0.046722, -0.063256, 0.05601, -0.055211, -0.100832, -0.03179, 0.115327, -0.030993, -0.009283, 0.029226, 0.064101, 0.057138, 0.035336, -0.022987, 0.01637, 0.054021, -0.15113, 0.124175, 0.16111, -0.002532, 0.099407, -0.141055, -0.098628, 0.061106, -0.021689, 0.171741, 0.016107, -0.158733, 0.072395, 0.061105], [-0.405826, -0.017392, -0.06792, -0.028121, -0.06942, -0.396404, -0.265332, -0.111507, -0.035377, 0.063078, 0.019754, 0.233768, -0.060596, -0.001048, 0.056646, -0.548017, -0.138967, -0.252096, 0.187265, -0.057108, 0.102284, 0.024415, 0.081796, 0.136434, -0.037438, -0.292919, 0.139437, 0.054217, 0.041174, -0.175229, -0.075196, -0.17352], [-0.240005, 0.169842, -0.024323, 0.065301, -0.194136, -0.283356, 0.032637, -0.083807, -0.021168, -0.085463, -0.038954, -0.000283, 0.109163, 0.083364, 0.123331, 0.010275, -0.167846, -0.242963, 0.045716, 0.039047, -0.078828, 0.002267, -0.146646, 0.030376, -0.144007, -0.109951, -0.036043, -0.076025, 0.006286, 0.031889, 0.099646, 0.129447], [0.088239, 0.051312, -0.091212, -0.045, 0.362579, 0.021264, 0.151016, 0.052283, 0.063221, -0.076685, -0.208746, 0.009763, -0.077091, 0.137534, -0.092049, 0.036206, -0.010619, 0.169965, 0.234265, -0.156216, 0.019372, -0.074862, 0.113182, -0.26195, 0.096011, 0.01567, -0.011927, -0.006163, -0.009712, 0.024991, -0.103902, -0.150409], [-0.183345, 0.077168, -0.047501, -0.120316, 0.035759, 0.052328, 0.025784, -0.042313, -0.15577, 0.101744, -0.058546, -0.091438, -0.101137, 0.161982, 0.17163, -0.271647, 0.038679, -0.057612, -0.048218, -0.025117, 0.001519, 0.123551, -0.04701, -0.053003, 0.111962, -0.072571, 0.106361, 0.054043, -0.138094, -0.272555, -0.063836, 0.039523], [0.220485, 0.041176, -0.024683, -1.6e-05, -0.05581, 0.196013, 0.004172, 0.027169, -0.052287, 0.141655, -0.09996, -0.430812, -0.065156, 0.055046, -0.148946, 0.082973, 0.11691, 0.105807, -0.135184, -0.081282, 0.073521, 0.076863, -0.060881, 0.064912, 0.232673, 0.013253, -0.054356, 0.169745, -0.036176, 0.135666, 0.00547, 0.036473], [0.262129, -0.106674, 0.116577, -0.089522, 0.153362, 0.103044, -0.137692, 0.153249, 0.154632, 0.164809, 0.099698, 0.217449, 0.154636, 0.189025, -0.401223, -0.109458, 0.098314, 0.305036, 0.154998, 0.192502, -0.003295, 0.047705, -0.001226, -0.161678, 0.268624, -0.065827, 0.021959, 0.205546, 0.084642, -0.045988, 0.031809, 0.079299], [-0.29381, -0.05001, 0.021632, 0.038528, -0.217106, -0.123448, -2.6e-05, 0.085026, -0.334032, 0.012673, 0.011975, 0.027499, -0.086456, -0.158925, -0.005552, 0.009851, -0.150869, -0.129293, -0.067031, -0.108837, 0.012188, 0.03255, 0.099723, 0.011093, -0.121433, 0.103566, -0.129813, 0.094522, 0.001096, -0.241152, -0.114396, -0.022028], [0.166077, -0.100893, 0.089759, 0.063815, -0.240205, -0.032354, 0.161282, -0.004894, -0.032884, -0.014425, -0.033501, -0.244705, 0.090582, -0.103013, 0.090345, -0.02242, -0.039296, 0.085799, -0.125034, 0.100676, -0.141416, -0.08864, 0.12338, -0.169665, 0.123952, 0.026097, -0.006031, -0.162155, 0.050997, 0.090228, 0.005274, 0.136181], [0.162162, 0.009519, 0.152835, 0.029935, 0.086949, 0.080303, 0.21596, 0.078093, 0.07859, 0.224662, 0.157377, 0.139856, -0.039408, 0.029333, 0.192322, -0.05264, 0.148968, -0.043011, 0.127574, 0.248626, -0.049596, 0.087233, -0.003195, -0.089919, 0.222012, 0.087364, 0.011276, -0.308983, 0.047941, 0.089749, 0.140691, 0.105858]], [-0.088882, 0.123563, 0.080655, 0.109122, 0.050122, 0.06006, 0.033225, 0.086767, -0.052283, -0.239468, -0.129399, -0.146761, 0.047567, -0.016211, -0.01783, 0.181639, 0.124545, 0.046701, -0.145332, -0.022393, 0.138729, 0.058349, 0.154006, -0.088758, -0.17783, 0.095622, 0.072058, 0.03934, -0.107525, -0.141332, 0.071798, -0.06465]),
    ([[0.347932], [-0.147188], [-0.093466], [-0.005907], [-0.186618], [0.15218], [0.417094], [0.177668], [0.183366], [-0.324353], [-0.272949], [0.239712], [-0.084076], [0.190046], [-0.217289], [0.250396], [0.165998], [0.241982], [0.165498], [-0.215202], [-0.083476], [-0.12136], [0.141479], [0.245653], [0.245069], [0.069803], [-0.094263], [0.126866], [-0.077331], [0.233032], [-0.156342], [-0.093046]], [0.114574]),
]
SEQ3_WI = [[-0.25763, -0.23943, -0.000325, -0.195049, -0.402302, -0.426178, -0.249342, -0.297825, -0.530796, -0.001394, -0.36672, -0.464989, -0.302642, -0.227521, 0.019347, -0.190058, 0.333633, -0.027207, -0.149488, -0.146106, -0.24286, -0.021526, 0.251579, -0.278339, -0.066686, 0.006664, -0.18131, -0.28697, -0.209427, 0.077272, 0.06796, 0.147999, -0.046244, 0.187983, -0.286699, 0.039466, -0.310986, -0.171105, 0.013629, 0.088791, 0.235057, -0.07574, -0.188432, 0.008195, -0.025096, 0.180983, 0.159517, -0.245639], [0.202341, -0.176005, -0.191779, 0.342137, 0.227591, 0.173818, 0.009921, 0.493734, 0.154836, -0.238466, -0.078283, -0.212337, 0.279305, -0.038833, -0.072097, -0.239257, -0.01951, 0.126294, -0.469742, -0.065015, -0.12163, -0.113097, -0.02046, 0.168089, 0.129477, -0.262391, 0.089934, -0.099473, 0.106034, 0.173649, -0.350472, 0.150321, -0.086091, -0.45176, 0.489407, 0.115381, 0.146289, -0.233115, -0.004696, -0.010991, 0.043518, -0.359513, -0.12458, -0.294652, -0.108851, 0.138772, -0.067181, 0.07628], [0.025268, 0.107515, -0.273343, -0.146657, 0.285796, -0.193162, -0.150861, -0.082927, -0.637771, -0.070919, -0.004218, -0.280771, 0.288012, -0.356352, -0.11751, 0.183381, 0.137822, 0.347375, -0.203461, 0.085147, 0.181454, -0.252411, -0.225841, -0.293531, -0.278978, 0.246393, -0.443531, 0.307215, 0.496203, 0.496109, -0.179835, -0.116071, 0.138774, 0.146304, -0.396514, 0.098415, 0.222134, 0.129038, -0.155258, 0.01838, 0.263411, 0.092804, 0.035658, -0.212534, 0.116151, -0.018015, 0.085507, -0.084907], [-0.050775, 0.004051, -0.192612, 0.271047, 0.035539, 0.142532, 0.411361, -0.174501, -0.413111, -0.042772, 0.050731, -0.035, 0.388974, -0.046443, -0.0863, -0.080456, 0.009572, -0.041549, -0.481812, -0.096553, 0.176855, 0.29794, -0.31491, -0.074231, -0.003102, -0.071726, -0.24965, 0.025265, -0.103636, 0.47912, -0.302266, 0.276335, 0.203376, 0.209905, 0.130302, -0.045558, 0.413767, -0.057013, 0.071263, -0.069019, -0.211055, 0.00247, 0.33097, -0.211484, 0.190954, 0.31445, -0.325769, 0.103872], [-0.171305, -0.016311, 0.207419, -0.244137, 0.189891, 0.544148, 0.065094, 0.192046, -0.18069, -0.780652, -0.15369, 0.040542, 0.364022, 0.408819, -0.086448, 0.141267, -0.174611, 0.513528, -0.105456, 0.075263, 0.16594, 0.282445, -0.367259, -0.069487, 0.558216, -0.854538, 0.203712, 0.060676, 0.21948, -0.207602, -0.241356, 0.158361, -0.020163, 0.033385, -0.249525, -0.028732, 0.155244, -0.000923, 0.211942, 0.131575, -0.087453, -0.843335, -0.387007, 0.085488, -0.025801, 0.024796, -0.167716, 0.060484], [0.470443, 0.217835, -0.138051, -0.343003, -0.46432, -0.488113, 0.273141, 0.108518, -0.066924, -0.19044, 0.168441, -0.167302, -0.211664, -0.078253, 0.666856, -0.503145, -0.177053, -0.687099, 0.125475, 0.219466, -0.540419, 0.122569, 0.168406, 0.189942, 0.673589, -0.06098, 0.275876, 0.260456, -0.213207, 0.126285, 0.115991, -0.196094, -0.190868, -0.53787, 0.13119, -0.019272, -0.021244, 0.030837, 0.393604, 0.023234, -0.033083, -0.092539, -0.281301, -0.055364, -0.461366, -0.017363, 0.136618, 0.343962], [0.187702, 0.023108, -0.431879, 0.233228, -0.023802, 0.107202, -0.193721, 0.314353, -0.275051, 0.050419, 0.059439, -0.154827, -0.181479, -0.112435, -0.305467, 0.383272, 0.017933, -0.133753, 0.187565, -0.089922, 0.035633, 0.044574, -0.059673, -0.09959, 0.094537, -0.295941, 0.036323, 0.253861, -0.250281, 0.389736, -0.01536, 0.036671, 0.128648, -0.129149, -0.059996, -0.119385, 0.073031, 0.270608, -0.263872, 0.041968, 0.214533, -0.173231, 0.104673, -0.083145, -0.011528, -0.332645, -0.258104, 0.03867], [0.461518, 0.239316, -0.474909, 0.51204, -0.014526, 0.250709, -0.216362, 0.057912, 0.051692, -0.232766, 0.058771, -0.226224, -0.121888, -0.255103, 0.022296, -0.043766, 0.1377, -0.221097, 0.224541, -0.058546, 0.225154, 0.062396, -0.019026, 0.037495, -0.138107, 0.155909, 0.281558, -0.057609, 0.07671, 0.105922, -0.172004, -0.263383, 0.098673, 0.028119, -0.313967, 0.085962, -0.10653, 0.040459, 0.183007, 0.16669, -0.065253, -0.25328, -0.166519, -0.299696, 0.095582, -0.299175, -0.290001, 0.235264], [0.249119, 0.351755, -0.531105, 0.460903, 0.295711, 0.359076, -0.247234, -0.026263, -0.097323, -0.19626, 0.0022, -0.217293, -0.013083, -0.347656, 0.039789, -0.095216, -0.157583, -0.384811, 0.104697, -0.053019, 0.42383, -0.182196, -0.142635, 0.029672, 0.16423, 0.103302, 0.213507, 0.265866, -0.271534, 0.473302, -0.258119, -0.190337, 0.166742, -0.214742, 0.193912, -0.316377, -0.085652, -0.044946, 0.118249, 0.142505, -0.042057, -0.039113, 0.045015, -0.326235, -0.160111, 0.189361, 0.000249, 0.030881], [0.145793, -0.617472, -0.427367, 0.132146, 0.043408, 0.242211, -0.194678, 0.048813, -0.048319, -0.019336, 0.044214, 0.034194, 0.124313, -0.025135, 0.269086, -0.308124, -0.102581, -0.361637, 0.096269, -0.068616, 0.171441, -0.259347, 0.054313, -0.263149, 0.084454, -0.022863, -0.078467, -0.207471, -0.151594, 0.458066, 0.284512, 0.279054, -0.509864, 0.165957, 0.058567, 0.108075, 0.097071, 0.135093, 0.156352, -0.119886, -0.117874, -0.071708, -0.136535, 0.164326, -0.16245, 0.22298, -0.231583, -0.162984], [-0.292271, 0.221456, -0.107807, -0.125212, -0.274129, 0.029259, 0.103948, -0.11701, 0.026254, 0.132548, 0.201806, -0.13078, -0.100963, 0.041228, -0.038913, -0.06228, 0.396969, -0.207832, 0.15028, -0.088088, -0.526897, 0.037153, 0.073661, 0.117882, -0.551765, -0.427321, -0.018653, -0.196948, -0.220605, 0.042673, -0.096932, -0.291898, -0.009847, -0.151422, 0.174838, -0.013511, 0.042475, 0.155004, 0.237723, -0.070401, 0.279188, -0.025792, -0.045941, 0.005336, -0.166838, 0.020696, -0.086836, 0.27152], [-0.083011, -0.216332, -0.034448, 0.023987, -0.000309, -0.155677, -0.29592, -0.57705, 0.352979, -0.002457, -0.029132, 0.017783, 0.05294, 0.07683, -0.035932, -0.118328, -0.534266, 0.191651, -0.432699, 0.038195, -0.383005, 0.099888, -0.216681, -0.070167, -0.398658, -0.060974, -0.176457, -0.439395, -0.407514, -0.113534, 0.044694, -0.236523, -0.212742, -0.22051, 0.132717, -0.047474, 0.09651, 0.197984, -0.083194, 0.051965, 0.223014, 0.065273, -0.073924, 0.212011, 0.013221, -0.201066, -0.008732, 0.006573], [-0.322793, 0.137329, 0.246408, 0.218806, -0.027999, -0.25067, 0.285594, -0.011264, -0.329961, -0.220305, 0.105879, -0.335804, -0.246959, 0.084123, -0.029104, 0.035944, 0.22692, 0.036049, 0.276164, -0.345549, -0.527827, 0.367884, 0.04397, -0.187749, 0.18473, 0.011393, 0.405745, 0.381912, 0.352588, 0.241181, -0.225946, -0.317483, -0.052566, -0.233657, -0.299302, -0.224526, -0.080547, -0.115454, -0.010005, -0.051079, -0.097707, 0.134871, 0.130652, 0.044382, 0.051831, 0.221642, -0.066845, -0.160615]]
SEQ3_BI = [0.210347, 0.253297, 0.304285, 0.007616, 0.21611, 0.131461, 0.275613, 0.211779, -0.040059, -0.137341, 0.031999, 0.130993, 0.237258, 0.040292, 0.297077, 1.8e-05, 0.262898, 0.504228, 0.446796, 0.337631, 0.633438, 0.74533, 0.698204, 0.534787, 1.031403, 0.260249, 0.251986, 0.397892, 0.514279, 0.424936, 0.226839, 0.249488, -0.292719, 0.023913, 0.215827, 0.117871, -0.216788, -0.302195, -0.167557, 0.05557, 0.188116, -0.109644, -0.134643, -0.221451, 0.000576, -0.325364, -0.235833, 0.30678]
SEQ3_WH = [[-0.081322, 0.131727, -0.099796, 0.018397, 0.109501, -0.245385, -0.079427, 0.151506, 0.182603, -0.269075, -0.150278, 0.126376, 0.042697, 0.117158, -0.264277, 0.378539, -0.179567, -0.532798, -0.145683, 0.448372, -0.621554, -0.166396, -0.145234, -0.31339, -0.577562, -0.238799, -0.518062, 0.007833, 0.381454, -0.433821, -0.325492, -0.027011, 0.363766, -0.050501, 0.107779, -0.346414, 0.351244, 0.283422, -0.406563, 0.106037, 0.223959, -0.201243, 0.217314, -0.019092, 0.140289, -0.13453, 0.274122, -0.15919], [-0.126393, -0.309218, 0.228615, 0.067837, 0.009504, 0.046392, -0.34808, -0.328266, -0.516638, 0.114362, 0.024844, 0.135226, 0.316069, -0.446765, -0.075287, 0.027173, -0.467625, -0.316471, -0.101472, 0.342474, 0.190946, 0.205847, 0.423308, -0.266089, 0.096089, -0.365617, 0.603389, -0.088793, -0.247038, -0.298098, 0.130238, -0.071948, -0.050401, 0.209906, -0.003106, -0.449844, 0.281948, 0.117619, -0.058508, -0.132619, -0.521755, 0.19068, -0.110431, -0.116224, -0.14098, -0.311997, 0.22175, -0.020552], [0.303285, 0.01071, -0.266806, 0.143282, 0.414157, 0.19837, 0.177445, -0.048416, 0.357756, 0.472331, -0.025874, 0.319575, 0.276538, -0.341671, 0.15255, -0.063292, -0.329147, 0.443732, 0.663031, 0.470411, 0.229358, 0.634103, -0.224035, 0.486821, 0.907797, 0.148683, 0.490727, 0.387971, 0.217556, -0.001939, 0.12589, 0.201362, 0.093885, 0.211026, 0.321118, -0.045194, 0.000233, -0.169391, 0.138747, 0.132433, 0.040923, 0.322419, 0.378051, 0.041484, -0.26354, 0.22648, -0.16536, -0.040404], [-0.265238, 0.036177, 0.248734, 0.082078, -0.445813, 0.020021, -0.059574, -0.288751, 0.364911, -0.278808, -0.706381, -0.577859, -0.263847, -0.078227, 0.207975, -0.257043, 0.261976, 0.251558, 0.253677, -0.034018, 0.012094, -0.313376, 0.063075, 0.000602, 0.023361, 0.582428, 0.35089, -0.170967, -0.250947, 0.168617, 0.290993, -0.134434, -0.290103, -0.272141, 0.148291, 0.369281, -0.316023, -0.253532, -0.205362, 0.507183, 0.33228, -0.047204, -0.162435, 0.114417, -0.464549, -0.07725, 0.03963, 0.1252], [-0.200145, -0.337912, -0.567171, 0.196823, 0.033174, -0.011862, -0.431582, -0.158125, -0.105538, 0.127998, -0.184425, 0.056398, 0.281864, 0.10112, -0.473173, 0.183602, -0.385893, -0.026253, -0.496767, -0.328138, -0.354086, -0.586211, -0.455966, -0.406043, -0.228207, -0.346236, -0.066372, -0.516972, -0.683583, -0.438552, 0.235712, -0.463389, -0.02679, -0.050807, -0.174926, 0.038815, 0.14731, 0.463964, 0.078791, 0.04579, 0.01681, 0.039675, -0.131823, -0.406883, 0.268784, 0.074691, -0.12332, -0.050016], [0.008767, 0.159373, -0.462301, 0.017161, 0.184249, -0.059464, -0.27181, -0.114572, 0.156161, 0.093995, -0.085433, 0.248315, 0.069957, 0.027759, -0.249775, 0.0629, -0.247122, -0.322078, -0.594427, 0.18421, -0.801905, 0.04875, -0.562763, -0.552983, -0.883561, -0.400429, -0.624829, -0.252162, -0.482275, -0.101137, -0.401978, -0.249825, 0.2741, 0.22846, -0.213664, -0.379215, 0.070861, 0.377817, -0.099121, -0.383927, 0.279927, -0.12815, -0.020606, -0.19827, 0.238194, 0.280216, 0.176067, -0.277683], [0.191345, -0.409004, 0.327339, 0.190857, 0.056447, -0.057139, 0.682048, 0.033447, -0.311186, 0.209553, 0.253075, 0.196663, 0.086056, -0.23291, 0.283629, -0.092963, 0.028908, 0.010433, 0.346357, -0.186939, -0.270426, -0.725208, 0.098961, 0.11657, 0.254618, 0.23228, 0.556767, 0.239328, -0.125402, -0.076994, -0.424578, -0.14247, -0.372314, -0.123935, -0.212969, 0.063645, -0.176153, 0.014149, 0.780592, 0.144935, -0.251756, -0.029234, 0.053012, 0.226694, -0.184882, -0.130262, -0.025866, -0.070549], [-0.094671, 0.282292, 0.096553, -0.223638, -0.431124, -0.184935, 0.002507, 0.052037, 0.259839, -0.409641, -0.186764, -0.633221, -0.383919, 0.095201, -0.078654, -0.099351, -0.153167, 0.44507, 0.312271, 0.175696, -0.109791, -0.000511, 0.019378, 0.367091, 0.524514, 0.913676, 0.25907, 0.00127, 0.232213, 0.386026, 0.293193, -0.072242, -0.191556, -0.068959, -0.120929, 0.516263, 0.018985, -0.391124, -0.203857, 0.361076, 0.561634, -0.024976, 0.227757, 0.082336, -0.303114, 0.479577, -0.133254, 0.066663], [0.017174, 0.033283, 0.092917, 0.227684, -0.092268, -0.359492, 0.029645, 0.265732, 0.397777, 0.036427, -0.282616, -0.351961, -0.243945, 0.13842, -0.092494, -0.248364, 0.471455, 0.710572, -0.106655, 0.322344, -0.244551, 0.232994, 0.048744, 0.253068, 0.298732, 0.144265, -0.249925, -0.272219, 0.225885, 0.121247, 0.561048, 0.048692, 0.095158, -0.410902, 0.129128, 0.287668, -0.066014, -0.007168, -0.506416, 0.27882, 0.22069, -0.243164, 0.113723, -0.049366, 0.140439, 0.461799, -0.288805, 0.208093], [0.271173, 0.175818, 0.186301, -0.238199, 0.196334, -0.30752, 0.036644, 0.000948, 0.132198, 0.360864, 0.479787, 0.52326, 0.23542, -0.546339, -0.041417, -0.0702, 0.133671, -0.442983, -0.372623, -0.07451, 0.139009, 0.028471, -0.284717, 0.074159, 0.553431, -0.340698, 0.109567, -0.307678, -0.540322, -0.423407, 0.131053, -0.412737, -0.230184, 0.181414, 0.241536, 0.043389, -0.197396, 0.068787, 0.198498, -0.422828, -0.230495, 0.345302, 0.407421, 0.420899, -0.107798, -0.021918, 0.392477, 0.351328], [0.161914, 0.459821, 0.407541, -0.146503, 0.561724, 0.330729, 0.467784, 0.255885, -0.125457, 0.723159, 0.227128, -0.014519, 0.276222, -0.389682, 0.298961, -0.277766, 0.004738, -0.384168, 0.103867, 0.177322, -0.279487, -0.271813, -0.330862, 0.138739, 0.134167, -0.138434, 0.215841, 0.071166, -0.003715, -0.190648, -0.420718, -0.368777, -0.036711, -0.207581, 0.292298, 0.186359, -0.367805, -0.251114, 0.421878, 0.289432, -0.066247, 0.497553, 0.457, 0.235016, -0.103962, 0.169085, 0.001106, 0.208639], [-0.0806, -0.153632, 0.14561, -0.128541, 0.115639, -0.161732, 0.277382, -0.39545, 0.401183, 0.213156, -0.195615, 0.069093, -0.205121, -0.110066, -0.11383, -0.138025, 0.166756, -0.381704, -0.085662, -0.332631, 0.109569, -0.30239, -0.247738, -0.3267, -0.032642, -0.669449, -0.215556, -0.346805, -0.61881, -0.070048, -0.622674, -0.047481, -0.264959, 0.127953, -0.209123, -0.111074, -0.221729, 0.126924, -0.087959, 0.205905, -0.174116, 0.55372, -0.113897, 0.002432, -0.03568, 0.20236, 0.249321, 0.040165], [-0.055744, 0.272345, 0.016152, 0.207575, -0.069859, 0.092533, -0.368697, 0.120486, 0.093, -0.314204, -0.028025, 0.107288, 0.364695, 0.385204, -0.024098, 0.403636, 0.120316, -0.053865, 0.203023, 0.418509, -0.190928, 0.789827, -0.052652, 0.157079, -0.176032, -0.216932, -0.080937, 0.040946, 0.38975, -0.167227, 0.17498, 0.14209, 0.314938, 0.225679, 0.238572, 0.00963, 0.182036, 0.285534, -0.518476, -0.384677, 0.313588, -0.179722, -0.289934, -0.4261, 0.099236, -0.172838, -0.252008, -0.36633], [-0.266901, -0.159259, 0.25107, 0.208385, 0.132308, 0.100081, 0.370729, -0.275226, -0.10998, 0.550941, -0.507733, -0.373573, 0.044656, 0.230025, 0.406439, 0.050082, 0.54033, -0.159085, -0.312748, -0.518456, -0.124976, 0.365817, 0.122921, 0.054959, -0.444478, -0.43007, 0.321894, 0.044561, 0.007673, -0.022273, 0.070264, 0.022663, -0.269098, -0.356797, -0.138391, 0.037863, -0.289131, -0.168695, -0.135521, 0.043956, 0.125065, 0.197452, -0.067558, 0.085539, 0.121454, 0.133834, -0.150898, 0.119139], [-0.237628, -0.130765, -0.335289, -0.361569, -0.199642, -0.097621, -0.278959, -0.350455, -0.014982, 0.067912, 0.026453, 0.173845, -0.075943, -0.216167, -0.063032, -0.205388, -0.099269, -0.431998, -0.202947, -0.253346, 0.041458, -0.468822, -0.348527, -0.555586, -0.462294, -0.5356, -0.511631, -0.171009, -0.059263, -0.412376, -0.323391, -0.233348, 0.233103, 0.085597, 0.286978, 0.124981, -0.248828, 0.11077, -0.171618, 0.21018, 0.162448, -0.056934, 0.432474, 0.35273, -0.214033, 0.014366, 0.276628, 0.084854], [0.050968, 0.147412, 0.041363, -0.409243, 0.230877, 0.16372, 0.448034, -0.096785, 0.318073, 0.091746, -0.09301, -0.119012, 0.145985, 0.243282, 0.112419, -0.313414, 0.116739, 0.635383, 0.16968, -0.083964, -0.172812, -0.090344, -0.156705, 0.62659, 0.684064, 0.472323, -0.130414, 0.282567, -0.16542, 0.535414, 0.008517, 0.043942, -0.358086, 0.043425, -0.200761, -0.050453, -0.319328, 0.279468, -0.074713, -0.274744, 0.105574, -0.085245, 0.089035, 0.119917, 0.206957, 0.174701, 0.117259, 0.699318]]
SEQ3_BH = [-0.050664, -0.009825, 0.040459, -0.049377, 0.099213, 0.311123, 0.049348, 0.223569, -0.062828, -0.213088, 0.435905, 0.238464, 0.349284, -0.063211, 0.258035, -0.129428, 0.112767, 0.590817, 0.307129, 0.499211, 0.345326, 0.498447, 0.409744, 0.289518, 0.843306, 0.298067, 0.543532, 0.486559, 0.746411, 0.536905, 0.349845, 0.414729, 0.184252, 0.241719, -0.090846, -0.195449, -0.292676, -0.18185, 0.257377, 0.135748, 0.223651, 0.216944, -0.01391, -0.235592, 0.272148, 0.117725, 0.086091, -0.105285]
SEQ3_HEAD = [
    ([[0.083695, -0.270672, 0.200319, 0.084372, 0.174351, -0.164448, 0.146627, -0.019648, 0.046092, -0.160519, 0.1075, 0.02963, -0.059093, 0.043969, -0.122021, -0.072153, 0.023636, -0.047426, 0.100451, 0.052194, -0.063425, -0.252483, -0.182344, 0.238687, 0.135321, -0.168193, 0.232913, -0.109508, -0.077533, 0.071166, -0.004349, 0.043966, -0.089247, -0.119831, -0.120264, -0.015851, 0.034313, -0.186739, 0.096998, 0.061774, 0.125688, 0.09149, -0.090873, -0.0883, 0.086175, 0.008321, 0.004165, -0.059984, -0.007118, -0.028178, 0.050771, -0.091035, -0.154362, -0.011061, -0.064371, 0.144718, -0.139215, -0.07456, -0.054137, 0.121943, 0.347221, -0.040585, -0.027307, 0.103235], [0.006357, -0.124807, -0.062926, -0.00449, -0.129794, -0.075125, 0.048112, -0.0383, 0.069306, -0.159008, 0.109665, 0.022497, 0.048324, 0.024733, -0.108278, 0.069045, 0.033291, 0.06151, 0.002676, -0.107359, -0.126354, 0.054938, 0.01482, -0.100104, -0.006295, -0.029665, -0.087949, -0.007737, -0.010843, 0.005089, 0.029187, -0.023896, 0.055219, -0.180334, 0.116157, -0.046658, -0.088675, 0.100343, -0.029714, 0.061428, -0.045765, 0.005524, 0.071133, -0.119656, -0.059619, -0.089682, 0.092792, -0.127234, 0.180817, 0.10056, 0.120142, 0.183435, -0.145573, -0.098013, 0.141994, -0.007828, 0.088418, -0.12477, -0.023068, 0.14308, -0.056225, 0.08323, 0.00528, -0.039314], [0.107199, -0.082643, -0.039447, 0.008371, -0.116753, -0.151338, 0.127886, -0.085085, 0.08577, -0.112291, 0.18227, -0.100544, 0.05019, 0.062579, -0.016914, 0.053483, -0.154635, 0.118055, -0.028212, 0.089491, 0.003783, -0.082019, 0.056885, -0.099688, -0.121956, 0.04972, -0.202571, 0.051586, 0.115536, -0.162667, -0.064598, -0.026192, 0.032203, 0.0408, -0.04621, -0.104085, 0.086433, 0.054148, -0.08424, -0.150602, 0.109499, -0.111806, 0.056745, 0.118635, -0.115228, -0.060235, 0.074259, -0.066034, 0.038092, -0.09263, -0.07931, 0.107802, -0.133327, -0.108337, -0.083177, -0.206971, 0.048136, 0.10504, -0.124489, -0.041185, -0.116555, 0.040557, -0.038477, -0.154601], [0.035773, -0.083067, -0.17926, -0.067262, -0.148046, 0.18374, -0.136397, -0.005686, -0.007286, -0.02095, -0.070125, 0.119302, 0.166891, -0.014538, -0.082872, 0.076341, 0.131014, -0.200086, 0.128278, -0.17049, 0.182328, -0.019034, 0.119023, -0.046141, 0.082833, 0.102991, 0.002196, -0.163314, 0.117582, -0.080801, -0.11383, 0.016514, 0.151054, -0.015999, 0.073173, 0.127201, 0.161277, 0.035758, -0.115882, -0.006894, -0.008137, -0.147612, -0.064695, -0.081658, 0.051658, -0.0785, -0.072473, 0.031387, -0.095745, 0.02962, -0.048736, 0.036196, -0.138198, 0.095508, -0.06698, 0.136604, 0.147476, -0.032133, 0.189818, -0.047822, -0.230083, 0.080077, 0.171074, 0.100977], [0.054243, 0.182564, -0.090828, -0.042182, 0.155706, 0.010108, 0.144334, 0.003366, 0.120598, 0.03972, -0.172454, -0.053105, 0.188714, 0.126943, 0.043725, 0.000763, -0.18606, 0.105979, 0.213664, 0.106027, 0.170648, -0.016325, -0.03712, -0.119187, -0.102598, -0.00301, -0.11896, -0.087353, -0.180412, -0.205431, -0.251641, 0.008638, 0.103083, 0.059745, 0.006353, 0.069969, 0.008555, 0.241214, 0.114935, -0.068387, -0.088668, -0.021849, -0.077867, 0.068935, -0.025534, -0.081784, -0.001284, -0.056883, -0.05317, 0.05375, -0.176164, -0.017764, -0.032852, -0.003647, -0.131252, 0.048113, -0.012269, 0.106014, -0.046109, -0.006203, -0.36337, 0.03301, 0.020332, 0.057099], [-0.179144, -0.044143, 0.056968, 0.036308, 0.057309, -0.067657, -0.129371, 0.135843, 0.054564, -0.119821, 0.045158, -0.110966, -0.132966, -0.11489, -0.000852, 0.085369, -0.00641, 0.035075, -0.071351, 0.064262, 0.090623, 0.036295, 0.017165, 0.00559, 0.012255, -0.085233, -0.031691, -0.041973, -0.024658, 0.09925, 0.016972, -0.017139, 0.17712, -0.042276, -0.089559, 0.081081, 0.07411, 0.045894, 0.060358, -0.077691, 0.054057, -0.000124, 0.152782, -0.015264, 0.042074, -0.122996, -0.105042, -0.022875, 0.12609, 0.081496, 0.026276, 0.212039, 0.028347, 0.125808, 0.075409, -0.131415, 0.003521, 0.068992, -0.066207, -0.002216, -0.128431, -0.046822, 0.020024, -0.003772], [-0.056149, 0.147275, -0.189834, -0.176747, -0.116733, -0.091142, 0.105543, -0.113302, 0.088677, 0.144545, 0.115879, 0.271355, 0.06688, 0.045677, 0.098298, 0.076131, -0.147673, -0.133421, 0.028307, 0.220206, 0.045251, 0.03237, 0.202752, 0.041503, -0.118722, -0.223082, -0.11862, -0.048524, 0.055779, -0.032077, 0.017209, 0.097475, -0.062342, 0.068374, -0.004419, 0.136058, -0.163106, 0.157915, 0.158437, 0.115782, -0.007947, -0.139875, 0.020836, 0.092471, -0.067715, -0.146581, 0.176145, -0.005292, 0.11101, -0.115816, 0.220079, 0.022897, -0.174657, -0.061488, -0.185466, -0.266639, -0.120314, -0.118452, -0.26107, 0.044665, 0.058604, -0.188878, 0.064664, 0.158519], [0.025173, 0.092353, -0.122169, 0.086621, -0.013553, -0.113418, -0.012325, 0.023039, -0.028088, 0.147465, -0.113762, -0.036656, -0.010222, 0.193169, -0.117812, 0.098369, -0.006407, 0.120303, -0.032447, -0.119734, -0.008441, -0.018223, -0.08227, -0.037721, -0.05467, 0.088404, -0.049507, 0.081377, 0.067488, -0.029471, 0.142831, 0.059539, 0.029638, 0.106453, 0.073011, -0.062793, 0.01031, 0.029249, 0.091084, -0.064123, 0.127169, -0.051174, 0.131086, 0.05843, -0.094479, -0.028182, 0.058097, 0.083383, -0.14104, 0.1106, -0.096687, 0.127008, -0.019568, -0.036135, -0.043766, -0.107627, -0.128751, 0.025844, -0.081884, 0.054932, -0.093568, -0.095805, 0.004331, 0.187928], [-0.038082, -0.014423, 0.039317, -0.202973, -0.095145, -0.101701, -0.052165, -0.163582, 0.098961, 0.047168, 0.07538, 0.278312, 0.194767, -0.273309, -0.126401, -0.046268, -0.027478, -0.085762, 0.004034, -0.056848, 0.144921, 0.167586, 0.074828, -0.059811, -0.130768, -0.009923, -0.193662, -0.031242, 0.021449, -0.078592, 0.040255, -0.180671, -0.152984, 0.03373, -0.100439, -0.044752, 0.020779, 0.080799, -0.034381, 0.092241, 0.250052, 0.111368, 0.173211, -0.220622, -0.289327, -0.07896, 0.117133, 0.027847, 0.155039, -0.158157, 0.070438, -0.043044, -0.064942, -0.0355, -0.074652, -0.175436, -0.023397, -0.170101, -0.032358, 0.121878, 0.054603, -0.134907, 0.126192, -0.062753], [-0.315615, -0.136154, 0.048664, 0.087311, -0.2058, 0.035099, 0.028522, 0.02927, -0.028615, 0.116259, -0.077003, 0.127673, 0.025873, -0.163063, 0.105334, -0.266151, 0.131711, -0.135002, -0.022198, -0.016424, 0.073809, -0.140786, -0.18756, -0.042272, 0.005147, 0.066849, -0.150358, -0.156374, -0.032919, 0.136332, 0.184738, 0.121288, 0.101261, 0.006445, -0.090411, -0.219475, 0.039012, 0.031093, 0.101558, 0.112317, -0.046664, -0.185077, 0.091949, -0.068666, 0.132018, -0.079782, -0.107559, 0.147628, -0.038713, 0.034836, -0.018596, 0.057873, -0.178805, 0.047415, 0.147823, -0.001595, -0.014792, 0.180525, -0.051646, 0.050033, -0.035104, -0.020932, 0.163947, -0.000653], [0.161136, -0.021618, -0.013866, -0.074186, 0.123593, -0.088625, 0.052847, 0.077346, -0.144227, -0.105539, -0.086981, -0.080338, -0.168777, 0.1252, -0.008939, -0.163753, 0.05955, -0.081036, 0.115831, 0.034495, -0.14153, 0.000883, 0.017125, 0.089747, -0.099164, -0.059676, 0.077646, 0.075815, 0.056444, 0.210246, 0.071874, 0.133473, -0.234778, 2.7e-05, 0.005579, -0.037364, 0.022017, -0.002896, 0.049035, 0.087753, 0.020118, 0.038627, -0.053976, -0.015479, -0.031764, 0.062996, -0.120377, -0.183873, 0.063884, -0.111334, 0.015189, 0.143669, 0.056666, -0.035169, 0.129618, 0.06536, -0.12003, -0.12395, -0.103119, -0.080148, 0.130467, -0.102586, 0.052871, -0.002112], [0.083597, -0.049747, -0.068834, 0.028954, 0.090643, 0.000154, -0.04089, -0.160568, 0.134915, -0.270967, 0.181086, 0.120826, -0.087002, -0.06387, 0.089267, -0.164062, -0.070006, 0.0437, -0.029052, -0.108245, 0.23847, 0.046639, 0.138659, -0.040964, 0.092552, 0.012647, -0.141704, -0.097993, -0.030681, -0.033826, -0.227962, -0.041426, -0.171252, -0.133626, -0.009535, -0.176916, -0.092647, -0.115212, 0.304583, 0.062326, 0.369085, 0.178349, 0.032735, -0.181681, -0.128478, 0.073319, 0.127692, -0.23655, 0.036132, -0.221465, 0.214682, -0.09815, -0.273176, -0.101539, 0.051942, 0.035622, -0.105906, -0.018962, -0.122916, 0.325414, -0.004608, -0.155873, 0.153895, 0.020078], [0.067497, 0.059855, 0.003373, -0.157161, -0.160671, 0.105525, 0.030626, -0.009104, -0.125932, -0.071492, 0.090055, 0.049473, 0.092123, -0.018228, -0.049037, -0.00891, 0.141521, 0.052278, -0.090439, -0.137031, 0.024877, 0.132931, -0.082222, -0.073904, 0.046065, 0.132322, -0.064886, 0.112596, 0.006811, -0.053735, 0.026651, 0.007037, -0.115397, -0.174607, -0.067559, -0.081523, -0.069504, -0.005211, -0.01929, -0.03959, -0.027166, 0.034461, -0.094053, 0.028315, 0.000717, 0.152069, -0.061878, 0.19958, 0.065243, -0.189344, 0.006311, -0.125998, 0.037406, -0.022217, -0.006974, 0.121113, -0.190405, 0.025728, 0.078198, 0.103904, 0.018696, 0.04823, 0.081072, -0.038408], [-0.225249, -0.07378, -0.120949, 0.034885, -0.150942, -0.021505, -0.053282, 0.331897, 0.083757, -0.02358, -0.119769, 0.078771, -0.033517, 0.150112, 0.012424, 0.119392, 0.129809, -0.034237, 0.072905, -0.019411, -0.066258, -0.108856, -0.007974, 0.142317, -0.040592, 0.082055, -0.026983, -0.115521, 0.064505, -0.230588, -0.017569, 0.136779, 0.04742, 0.11335, -0.155589, -0.004562, -0.064461, -0.080173, -0.099793, -0.232853, 0.011908, 0.127304, -0.082333, 0.258553, -0.148713, 0.085235, 0.058701, 0.132872, 0.028868, -0.138377, -0.212911, -0.114449, -0.056238, -0.024383, -0.015596, -0.031362, 0.007093, 0.141768, -0.069483, 0.175167, 0.070918, -0.120498, -0.067016, -0.061628], [0.085948, 0.11519, 0.102805, 0.036881, 0.130162, 0.034963, -0.001995, 0.074742, -0.101288, -0.166564, -0.072685, -0.100596, -0.095684, 0.044857, -0.018115, -0.216488, 0.059775, -0.252455, -0.014896, -0.095989, 0.130547, -0.114992, 0.002786, 0.044345, 0.093999, 0.015664, -0.148099, 0.073163, -0.314593, -0.028409, -0.066335, -0.154863, 0.002122, -0.112757, -0.05348, 0.024711, 0.048229, -0.014769, 0.188971, -0.261496, 0.049109, 0.076391, -0.162083, 0.052995, 0.065123, -0.117367, -0.01358, 0.014999, 0.1409, -0.000807, -0.123819, 0.128452, -0.012885, -0.302994, -0.078929, 0.088775, 0.06593, -0.167101, 0.184519, -0.007449, 0.094897, 0.088703, 0.121951, -0.07623], [-0.006682, -0.168089, 0.145781, -0.041775, 0.171843, -0.243406, 0.249428, -0.074102, -0.007636, -0.230704, -0.19456, -0.098014, 0.122865, -0.08256, 0.069209, 0.032036, -0.064241, -0.024574, 0.110918, -0.168368, 0.076441, -0.098881, -0.117437, -0.022947, -0.084451, -0.126617, 0.027404, 0.023403, -0.19427, 0.041532, -0.155265, -0.132606, -0.030034, 0.03801, 0.176163, 0.065998, -0.097451, -0.174004, 0.045273, -0.052569, -0.013224, -0.046562, 0.0431, -0.06918, -0.13262, -0.089491, -0.074719, -0.126176, -0.095246, -0.013328, -0.10263, 0.194177, 0.049934, 0.06557, 0.040688, 0.043159, 0.061851, 0.038801, 0.089147, -0.021722, 0.273017, 0.120344, 0.120005, 0.133975], [-0.001082, 0.012249, -0.025036, -0.220343, 0.01258, 0.00661, 0.172385, -0.029743, -0.090198, 0.045421, -0.040526, -0.033433, 0.171752, -0.039345, 0.030476, 0.079227, -0.182138, 0.238954, 0.104887, 0.129973, 0.209772, -0.202935, -0.137996, 0.258229, -0.161157, 0.039021, 0.149526, 0.045175, 0.19583, -0.033511, -0.030045, -0.110381, 0.052746, 0.039659, -0.23394, 0.088292, -0.165346, -0.135085, -0.20591, 0.045458, 0.053898, 0.093586, -0.204803, -0.049672, 0.000526, -0.076346, -0.107007, 0.222257, -0.259223, -0.07501, 0.033185, -0.019344, -0.052798, 0.153786, 0.108403, 0.050584, 0.192855, 0.056885, -0.298298, 0.011474, 0.038299, 0.054927, 0.084704, -0.028105], [0.010725, 0.260414, 0.116509, -0.103292, 0.055691, 0.231931, 0.298847, -0.098302, 0.208238, 0.008761, -0.17339, 0.046119, 0.032434, 0.26088, -0.073608, 0.065639, 0.137946, 0.153798, -0.139329, -0.084725, -0.189864, -0.087834, -0.039935, 0.14866, -0.006023, -0.272437, 0.399455, -0.047595, -0.016206, -0.020539, 0.067632, 0.169007, 0.072102, -0.128079, 0.120455, 0.139591, 0.119006, -0.079876, 0.086139, -0.112862, 0.029013, -0.036154, -0.106373, -0.02775, 0.102006, -0.110941, -0.08981, 0.122771, -0.12131, -0.001695, -0.003052, -0.17909, -0.147199, -0.040436, -0.019709, 0.065308, 0.086812, -0.037952, 0.15658, 0.154847, -0.364187, 0.12606, -0.118817, 0.037537], [-0.018198, -0.033842, -0.080915, 0.08381, 0.054193, 0.015276, 0.090954, 0.03426, -0.047902, 0.124614, 0.02944, 0.068627, -0.060248, 0.139297, 0.095969, 0.073856, 0.043133, 0.058135, -0.012966, 0.079626, -0.046678, -0.055057, -0.00482, 0.073727, -0.075759, -0.051646, -0.019555, -0.026947, 0.146151, -0.075115, -0.019234, -0.151937, -0.055993, 0.120962, 0.103507, -0.010621, -0.025047, -0.0565, -0.086774, 0.005305, 0.087784, -0.099778, 0.010357, -0.062119, 0.076419, 0.164275, 0.066616, 0.035215, 0.030289, -0.018342, -0.010096, 0.04247, 0.096913, 0.138525, -0.089375, -0.10722, -0.033318, -0.112857, -0.030274, -0.142731, -0.001618, 0.035498, -0.055133, 0.051117], [0.072399, 0.078296, 0.049264, -0.163021, 0.07072, 0.010752, 0.028959, -0.044572, 0.104417, 0.002521, -0.093238, 0.269235, 0.160404, 0.007675, -0.090173, -0.128886, -0.024985, 0.035015, 0.090746, -0.136521, 0.128438, -0.133279, 0.134756, 0.214642, 0.078114, -0.101299, 0.002557, 0.121752, 0.194997, -0.152993, 0.0464, -0.015077, 0.126364, -0.040633, -0.177039, 0.051327, -0.249591, -0.066618, -0.022623, 0.21483, -0.031438, -0.062047, 0.293675, 0.171691, -0.263767, -0.017357, 0.212556, 0.074934, 0.131208, -0.043638, 0.235883, 0.139292, -0.215017, 0.225862, -0.257553, -0.246025, -0.228244, 0.09664, 0.086913, -0.084361, -0.102168, -0.267138, 0.112673, 0.091778], [0.039632, -0.010694, 0.013618, -0.215625, -0.16543, -0.101679, 0.048491, 0.066576, 0.113244, 0.055825, -0.224206, -0.002084, 0.001132, -0.072672, 0.006606, -0.100328, -0.138417, 0.095964, -0.090155, -0.032989, 0.016368, 0.020343, -0.035477, -0.118226, 0.062918, 0.097016, 0.07718, 0.157685, 0.163437, -0.043685, 0.056826, -0.251707, -0.021008, 0.089549, -0.074414, -0.084196, 0.108689, 0.089663, 0.156957, 0.175716, 0.033388, 0.201912, 0.266596, 0.181437, 0.02853, -0.014829, -0.013852, 0.042411, 0.053457, 0.050608, 0.00827, -0.268575, 0.098099, -0.10657, 0.080488, -0.24374, -0.122354, -0.245985, 0.017518, 0.102313, 0.125264, -0.193465, 0.028076, 0.096866], [-0.002033, -0.032035, -0.069346, 0.207637, 0.014816, 0.059049, -0.010649, 0.243532, 0.148695, -0.113492, -0.027705, 0.139347, -0.199996, -0.005116, -0.105367, -0.114387, -0.214906, -0.254457, -0.161333, -0.054017, -0.086906, -0.119541, 0.344962, -0.024054, -0.188749, 0.007551, 0.06575, -0.156318, 0.125596, 0.014634, -0.054265, -0.101084, -0.037458, -0.107266, -0.101131, -0.059762, -0.17397, -0.015815, -0.107229, -0.08614, -0.008875, -0.180405, -0.098403, 0.046234, 0.098668, 0.160568, -0.184429, 0.146764, -0.312335, -0.10845, -0.062154, 0.051119, -0.053419, -0.20965, -0.115199, -0.200403, -0.040969, 0.058941, -0.270397, 0.034316, 0.105535, 0.122864, 0.123398, 0.16665], [0.124293, -0.03464, -0.160196, 0.265204, 0.132353, 0.228255, 0.290013, -0.145575, 0.194157, 0.186676, -0.008429, -0.083111, -0.045642, 0.017173, 0.183364, -0.069676, 0.207796, 0.003874, 0.18384, 0.058658, 0.129038, 0.095134, 0.108052, 0.072066, -0.145155, 0.033166, -0.060792, -0.082837, -0.064708, -0.174608, 0.055837, 0.156211, -0.269923, 0.066654, 0.093165, -0.034649, 0.085799, -0.166747, 0.250014, -0.001132, 0.210954, 0.145285, -0.038243, -0.044031, -0.0428, -0.068197, -0.013099, -0.108633, -0.218311, -0.099829, 0.230431, -0.082354, 0.054153, 0.007015, -0.080137, -0.240489, -0.206248, 0.021264, -0.005087, 0.035481, 0.265023, -0.168256, 0.16704, -0.052973], [0.004775, 0.101132, 0.24573, 0.164754, 0.144176, -0.093308, 0.110542, 0.010813, 0.100407, 0.190904, 0.058325, 0.014683, -0.081426, -0.164472, 0.142164, -0.015786, -0.080113, -0.127587, 0.071579, -0.02754, -0.019809, -0.006736, -0.056503, 0.121374, 0.053654, 0.137021, -0.009462, 0.028874, -0.036203, 0.099534, 0.204596, 0.012899, -0.117223, 0.106197, -0.295578, -0.233768, 0.094163, 0.233788, 0.199689, -0.009148, -0.008661, -0.080319, 0.257236, -0.265723, 0.103193, 0.062042, 0.065193, -0.16748, 0.003753, 0.002815, 0.122849, 0.232733, -0.205642, 0.085728, 0.250453, -0.059497, -0.034956, 0.104043, 0.08898, -0.018489, -0.112453, 0.010578, 0.004512, 0.089453], [0.056106, -0.145199, 0.068226, 0.139474, -0.041034, -0.020395, -0.150945, 0.025846, 0.11019, 0.064734, 0.095189, -0.177935, -0.128808, -0.116899, -0.005797, -0.126195, -0.048175, -0.034973, -0.133277, -0.270843, 0.066318, 0.000945, 0.136887, -0.18641, 0.08546, -0.041635, -0.011647, 0.032993, -0.084547, -0.066392, 0.028236, 0.142129, -0.185522, -0.054512, 0.046998, -0.06646, -0.150376, -0.213076, -0.066952, 0.072516, 0.082732, 0.195556, 0.059473, -0.151977, -0.066798, 0.089176, 0.058692, -0.270297, 0.08284, 0.069888, -0.110737, -0.298677, -0.279975, -0.17828, -0.04203, -0.026825, -0.017012, 0.084029, -0.069113, 0.138641, 0.006045, 0.06495, 0.107969, 0.105428], [-0.148865, -0.09311, -0.058111, -0.01194, 0.105978, -0.240019, 0.03823, 0.067991, -0.033347, -0.087986, 0.097808, -0.17542, -0.099527, 0.12346, -0.049453, 0.136194, 0.054493, -0.136487, -0.302412, 0.15079, 0.088452, 0.1107, -0.085213, -0.130434, -0.099886, -0.313954, -0.001166, -0.04646, 0.191629, -0.087514, -0.284242, -0.029687, 0.270588, -0.00573, 0.127966, 0.093656, -0.194503, -0.073195, -0.193725, -0.207303, -0.098256, 0.088299, -0.067849, 0.160263, 0.072817, -0.056605, -0.190928, 0.183128, -0.13062, 0.045767, 0.049743, -0.205203, -0.186692, 0.029268, -0.003168, 0.048361, 0.093693, 0.039972, 0.176265, 0.271937, 0.049012, 0.000457, 0.038192, -0.078948], [-0.104556, 0.087611, 0.06696, 0.255173, -0.108117, -0.021163, 0.126531, 0.29987, 0.108159, 0.150208, -0.089516, 0.05136, -0.084166, -0.061053, 0.08352, -0.085793, -0.133045, 0.186557, 0.116989, 0.334265, 0.02992, -0.138857, -0.028234, 0.083546, -0.235962, 0.085294, -0.149836, -0.12624, 0.032214, -0.127809, 0.022286, -0.044284, 0.040212, 0.072015, 0.130599, -0.15663, 0.137505, 0.109337, 0.216054, 0.077277, -0.010954, 0.136124, -0.059695, -0.129062, -0.084567, -0.258947, 0.021986, 0.056885, 0.0595, -0.095554, -0.106198, -0.039753, 0.007149, 0.190986, -0.068995, 0.156448, -0.147957, 0.127568, -0.125441, 0.138132, -0.082837, -0.078942, 0.117392, 0.063988], [-0.10635, -0.132493, 0.037813, 0.216779, 0.005783, -0.051545, 0.272259, 0.120125, 0.024525, 0.094687, 0.052998, 0.040013, 0.066416, 0.134824, -0.020658, -0.080615, 0.001633, -0.024709, 0.058812, 0.148022, 0.219583, 0.148214, -0.043244, 0.065187, -0.150332, 0.074273, 0.190958, -0.135471, -0.125054, -0.254562, -0.077997, 0.060321, -0.00976, 0.008163, 0.016966, -0.111816, 0.026914, -0.073783, -0.079487, -0.128263, 0.237125, 0.188117, -0.13576, -0.103751, -0.035763, -0.11044, -0.059978, 0.153709, -0.105546, 0.034068, 0.062138, 0.104093, 0.197333, -0.012541, -0.154286, 0.106933, 0.041816, -0.047195, -0.057825, 0.15426, 0.115178, 0.157622, 0.089597, 0.101544], [0.013871, 0.037742, -0.116973, -0.115975, 0.08177, -0.00833, -0.121853, -0.05895, -0.116942, 0.023183, -0.107702, 0.253554, -0.009009, -0.15668, -0.160472, -0.117674, -0.095346, -0.095575, 0.072895, 0.179268, 0.028222, -0.003152, 0.210075, -0.06081, 0.044628, 0.188241, -0.124438, -0.084397, -0.101691, 0.159789, 0.019495, -0.028864, 0.234747, -0.144447, 0.066532, -0.277821, 0.080612, 0.236835, 0.17822, 0.089728, -0.01607, -0.233842, -0.024282, -0.051601, -0.034723, 0.066783, 0.14365, -0.167662, 0.223176, -0.087761, 0.091729, -0.002756, -0.169776, -0.163465, -0.006219, 0.174891, 0.143372, -0.01648, 0.149962, -0.040531, -0.070037, -0.095886, 0.192234, 0.142414], [0.020938, -0.1606, -0.125894, -0.129572, 0.163317, -0.268226, -0.151894, -0.064059, -0.029915, -0.087563, 0.08639, -0.072231, -0.129976, 0.140661, -0.107179, 0.04982, 0.068613, -0.009905, -0.200603, 0.095229, -0.174308, -0.101389, -0.19036, -0.04117, -0.050598, -0.269957, -0.16128, -0.089799, 0.212384, 0.176174, -0.313246, -0.036422, 0.06302, -0.07681, -0.031431, 0.167976, -0.161142, -0.273558, -0.177904, -0.211833, -0.170179, 0.005834, -0.177081, 0.179725, -0.106837, -0.008049, -0.048926, -0.116102, -0.15169, -0.028181, -0.192338, 0.021653, -0.177812, -0.275951, -0.205421, 0.057174, -0.14879, -0.119284, 0.123881, 0.09746, 0.088002, -0.161114, 0.119345, 0.048088], [0.023833, -0.321065, 0.038976, 0.128946, -0.007207, -0.023013, -0.010824, 0.188243, 0.154716, -0.307905, 0.258404, -0.13192, 0.118395, -0.068355, -0.164776, -0.240304, 0.193263, -0.08882, -0.028022, -0.142133, 0.103205, 0.067531, -0.058849, -0.150916, -0.128076, -0.405852, -0.02708, -0.080196, 0.096864, -0.15904, -0.143541, 0.261756, -0.123031, -0.239541, 0.060976, -0.02504, -0.450497, -0.300963, -0.252721, -0.133414, 0.066633, 0.353303, 0.076714, -0.114491, -0.078972, 0.241706, -0.174779, 0.059883, -0.192212, -0.057799, -0.226871, -0.132227, -0.065467, -0.059866, -0.35584, -0.178843, -0.108845, 0.13524, 0.024295, 0.278464, -0.129249, -0.057088, -0.071728, -0.149537], [-0.03027, 0.04582, 0.193837, 0.102922, 0.077969, 0.182163, 0.051941, -0.176195, 0.015869, 0.528463, -0.02694, -0.099025, -0.062924, 0.026405, 0.200072, 0.064227, 0.247944, 0.066618, 0.131932, -0.298028, 0.084229, 0.16528, -0.111857, 0.105735, 0.147395, 0.374969, 0.233414, -0.185548, -0.003256, 0.04729, 0.447145, 0.108749, -0.077477, 0.364277, -0.144461, -0.058612, 0.276047, 0.201286, 0.008423, -0.033955, -0.025407, -0.183532, -0.022202, -0.314171, 0.005869, 0.147065, -0.082398, 0.183467, 0.066619, 0.346028, 0.087356, 0.036751, 0.097899, 0.186859, 0.191556, -0.123078, 0.101142, -0.069455, -0.02024, -0.074833, 0.222267, -0.224751, 0.097791, -0.052007], [0.081594, -0.050309, -0.070027, -0.277586, 0.056042, 0.151809, -0.253137, -0.235742, -0.245011, 0.126718, 0.040884, -0.189228, -0.001177, 0.152696, 0.050687, -0.202064, 0.221444, -0.030105, -0.190744, -0.157258, -0.098005, 0.043697, 0.031388, -0.102401, 0.067441, 0.195864, -0.04526, 0.17689, -0.029424, 0.099691, -0.203238, 0.140088, -0.048592, 0.014124, -0.199095, -0.0604, 0.233889, 0.099494, 0.158225, -0.121185, -0.058405, 0.189483, -0.05648, -0.067247, 0.1218, 0.195661, -0.295568, -0.044489, 0.13237, 0.187577, -0.09234, -0.010713, -0.027945, -0.074459, -0.009016, -0.022885, 0.03545, -0.051682, 0.075683, -0.130258, 0.022911, -0.1445, 0.081538, 0.070973], [-0.020097, -0.189264, 0.020019, -0.124893, 0.001111, -0.035734, -0.077785, -0.097736, -0.07768, 0.230468, 0.076094, -0.178438, 0.024255, 0.00853, -0.0326, -0.16799, -0.106858, -0.105386, 0.117893, -0.142527, 0.027043, 0.032118, -0.085131, -0.172313, 0.062359, 0.150198, 0.17903, 0.178886, 0.146506, 0.065868, 0.235235, 0.10052, 0.070172, 0.126318, 0.100603, 0.13048, -0.093277, 0.196614, 0.160654, 0.124074, -0.014126, -0.140821, -0.154561, -0.007851, 0.189834, 0.031789, -0.047405, 0.107126, 0.145851, 0.008363, -0.152876, -0.109601, 0.138175, -0.188697, 0.23167, -0.047958, 0.209978, 0.144476, -0.025591, 0.027305, 0.199235, -0.014991, -0.141532, 0.296825], [0.149443, 0.095859, 0.091123, 0.090595, -0.041747, 0.060007, 0.050347, -0.113662, 0.162466, -0.053656, -0.06007, -0.174634, 0.167309, -0.0567, -0.004059, 0.115463, -0.065247, 0.186661, -0.080562, -0.069043, -0.018273, -0.039362, 0.160719, 0.255812, 0.111697, -0.210091, -0.0222, -0.102186, -0.00767, -0.11168, 0.115713, 0.089637, -0.071801, 0.192073, 0.090333, -0.227993, 0.065004, 0.158484, 0.012844, 0.091977, 0.017001, 0.183334, 0.275985, 0.052277, 0.011044, 0.119081, 0.036361, 0.134993, -0.07494, -0.071109, 0.186475, 0.001931, 0.044498, -0.00112, -0.206145, -0.04852, -0.016981, -0.04281, -0.093535, 0.179126, -0.029553, -0.168906, 0.002871, -0.28755], [-0.016672, -0.153406, -0.093173, 0.093784, -0.090231, -0.183968, 0.140685, 0.169447, 0.143755, -0.3331, 0.057039, 0.03859, -0.058916, 0.035085, 0.020731, -0.265199, 0.022092, 0.134544, 0.000721, -0.019962, 0.089072, 0.025885, 0.201705, 0.191969, 0.076514, 0.044196, 0.023417, 0.081101, 0.137297, -0.068672, -0.107432, 0.021859, 0.053294, -0.044151, 0.04447, -0.0989, 0.009133, 0.021514, 0.004481, -0.193788, 0.072934, 0.087578, -0.11815, 0.021572, 0.173989, -0.074005, -0.067725, 0.236713, 0.187098, -0.013781, -0.118419, -0.120091, 0.004003, 0.034673, -0.082557, 0.112782, -0.204399, -0.105728, 0.194027, 0.207276, -0.048904, -0.187914, -0.019079, -0.19821], [0.137864, -0.023346, -0.119387, 0.315592, 0.265424, 0.155818, 0.199523, 0.039227, -0.107883, 0.222351, -0.065717, -0.132021, -0.176253, -0.095413, 0.171556, 0.1564, -0.045282, -0.210715, -0.03834, -0.116603, 0.031508, 0.199002, -0.010294, -0.143531, 0.157375, 0.102297, 0.16679, -0.002258, 0.00579, 0.000186, 0.125494, 0.028415, 0.199318, 0.117406, -0.132293, 0.086878, -0.064196, 0.097498, 0.108067, 0.065245, 0.096586, 0.03358, 0.069536, -0.091202, 0.016797, -0.019285, -0.025023, -0.072888, -0.117381, 0.142946, 0.134556, -0.009712, 0.116132, -0.137269, 0.163516, -0.146693, 0.070239, -0.054918, 0.141111, -0.127558, -0.04378, -0.085462, -0.085755, 0.048952], [0.070564, -0.080369, 0.055269, -0.014732, -0.026945, 0.0396, -0.102832, 0.141349, 0.07101, 0.023271, -0.203209, 0.023889, -0.152461, 0.074279, 0.174621, 0.068077, 0.018388, -0.151106, -0.248303, -0.133019, -0.15701, -0.033755, -0.214684, 0.007337, 0.089471, -0.012441, -0.023746, 0.072896, 0.152766, -0.10136, 0.127968, -0.159941, 0.088668, 0.054464, -0.236251, 0.038122, 0.065125, -0.238689, 0.036064, 0.078587, -0.157043, 0.210081, -0.191311, -0.152197, -0.032089, -0.040616, -0.306651, 0.224554, 0.203404, 0.097635, -0.018883, -0.158877, -0.041512, -0.214268, -0.123289, -0.041058, 0.045867, -0.225433, -0.114393, 0.04555, 0.091374, 0.106907, 0.01443, 0.099158]], [-0.15415, 0.053086, 0.116126, -0.104483, 0.141081, -0.12217, -0.061995, 0.02439, -0.015053, 0.108167, 0.06546, -0.078174, 0.105444, 0.017855, -0.033105, -0.060716, 0.02765, 0.23623, -0.217384, -0.334551, -0.20915, 0.012492, 0.023484, -0.050606, 0.20594, -0.03975, -0.197805, -0.040455, -0.037007, 0.018829, 0.148543, 0.0354, -0.266447, 0.146306, 0.038305, -0.121953, 0.076244, 0.064213, -0.183286, -0.087958, -0.284297, 0.065353, 0.12662, -0.005867, -0.0793, 0.142526, 0.050639, -0.241594, -0.072489, 0.161178, 0.066051, -0.086827, -0.234717, -0.140361, 0.146439, -0.125641, 0.158464, -0.067547, -0.125299, -0.180346, -0.182559, 0.125556, -0.144085, -0.07959]),
    ([[-0.00471, -0.020907, -0.117607, 0.012902, 0.005626, 0.063049, -0.016621, 0.085791, -0.119403, -0.079303, -0.011787, 0.097851, 0.006873, -0.025998, 0.064516, 0.052641, 0.04463, 0.030964, 0.128012, 0.021889, -0.052874, -0.090776, 0.000862, -0.042642, 0.037498, 0.13079, -0.133651, 0.044253, -0.096093, 0.032037, -0.069834, 0.127542], [0.102584, -0.077168, 0.126828, 0.097846, 0.063983, 0.043449, 0.081068, 0.052137, 0.077746, 0.109875, 0.064679, 0.062231, 0.043823, 0.083942, -0.197281, 0.201213, 0.191104, 0.153606, 0.07897, 0.151316, 0.03248, 0.030301, 0.128225, -0.328825, 0.173449, 0.189119, -0.13261, 0.092335, -0.062061, 0.025309, 0.107702, 0.124469], [-0.036226, 0.037331, 0.090167, 0.040746, 0.048527, 0.049523, -0.021861, 0.088788, 0.090067, 0.159202, -0.026031, 0.048364, 0.055313, 0.099421, -0.129526, 0.189072, 0.02964, 0.002225, 0.184589, 0.139976, 0.028782, -0.074634, 0.170826, -0.33311, -0.030753, 0.026446, 0.116456, 0.054161, 0.050997, -0.06856, -0.067352, 0.057441], [-0.181587, -0.026348, 0.119392, 0.078654, 0.226179, -0.085125, 0.112265, -0.027695, -0.101305, 0.101277, 0.138952, 0.221263, 0.060487, -0.094032, 0.067692, -0.109475, 0.116456, -0.154477, 0.057795, 0.116529, 0.047219, 0.089499, 0.046894, 0.399771, -0.170809, -0.15134, 0.035427, -0.04127, 0.061673, 0.165519, -0.008687, 0.006955], [-0.127005, 0.051243, -0.056822, -0.051643, 0.041078, 0.043589, -0.180729, -0.048716, 0.048852, -0.111285, -0.120516, 0.002357, -0.04692, -0.11628, 0.168599, -0.014972, 0.040769, -0.054739, -0.011023, 0.188275, -0.103632, -0.127491, 0.027272, -0.071846, 0.028181, 0.015145, 0.00378, -0.061062, -0.02771, 0.137578, 0.033924, 0.154938], [-0.028566, 0.022179, 0.143857, 0.073178, 0.0604, -0.084101, -0.227745, -0.013046, -0.008114, -0.028174, 0.121397, 0.022441, 0.101304, 0.102262, -0.014694, -0.109065, -0.02808, 0.063845, 0.072752, 0.107503, 0.132558, -0.055926, 0.01275, -0.207574, 0.064516, 0.020568, 0.085866, 0.095776, -0.034508, 0.016696, -0.033155, 0.092089], [0.078898, 0.099111, -0.087938, -0.10932, -0.010139, 0.039771, -0.020867, 0.058394, 0.1499, 0.065027, 0.118656, -0.079607, 0.078815, 0.163176, -0.107819, 0.091901, 0.13991, 0.108903, 0.144377, 0.11795, 0.04429, 0.073344, 0.008063, -0.046434, 0.099478, 0.096863, -0.004738, -0.070212, 0.036266, 0.189653, -0.119966, -0.008771], [-0.146259, -0.026369, 0.010176, 0.099393, 0.074957, 0.092926, -0.158009, 0.065951, -0.121274, 0.090854, 0.099451, -0.456478, 0.025398, -0.140391, 0.206099, -0.045722, -0.16856, -0.052286, -0.053121, 0.03243, -0.021955, -0.026224, -0.073249, 0.001907, -0.17875, -0.089255, -0.001587, -0.221314, 0.038855, -0.096096, -0.05527, 0.028717], [0.021876, -0.068825, 0.060399, -0.086817, -0.041694, -0.0721, 0.000558, 0.094127, 0.10655, -0.075874, 0.057061, -0.012876, -0.054489, -0.067836, 0.00361, 0.037556, -0.042315, 0.140792, 0.046597, 0.042783, -0.067258, 0.137422, -0.054561, 0.116881, 0.001749, 0.079086, 0.087728, -0.011467, -0.068025, 0.130521, -0.148855, -0.073594], [0.129612, 0.009001, 0.115709, -0.02677, 0.136209, 0.033103, 0.01313, 0.0769, -0.166483, -0.103834, 0.1282, -0.120718, 0.128734, 0.036557, 0.031248, 0.176105, 0.030459, 0.004951, -0.054581, 0.026371, 0.135572, 0.139817, -0.126295, -0.204915, 0.068479, 0.157623, 0.119703, -0.164435, 0.061733, 0.084137, -0.086317, 0.077367], [0.132176, -0.051934, -0.016689, 0.082806, 0.108146, 0.090464, 0.125534, 0.027186, 0.118333, 0.05293, -0.058854, -0.187906, 0.129417, 0.058656, -0.24405, -0.031672, 0.139467, 0.068734, 0.02673, 0.071661, 0.087165, 0.005001, -0.037227, 0.135833, -0.111454, 0.154921, 0.011467, -0.126933, 0.167523, 0.175998, 0.023177, 0.007078], [0.162481, 0.063232, 0.169811, -0.087895, -0.069269, 0.107178, 0.017521, 0.01697, 0.052764, -0.191029, -0.214021, -0.044809, -0.104253, 0.037961, 0.004401, -0.014776, -0.010147, -0.069076, -0.068318, -0.070169, -0.058888, -0.02355, 0.144334, -0.001691, 0.083951, -0.04458, -0.10496, -0.046645, -0.044163, -0.03593, -0.167462, -0.011862], [0.097032, -0.069459, -0.060359, 0.098506, -0.075003, 0.055383, 0.095154, 0.087621, 0.081364, 0.123468, 0.030871, -0.045655, -0.12892, -0.026348, -0.157559, -0.023418, 0.076889, 0.056705, 0.116737, -0.160322, 0.040156, -0.109104, -0.063367, -0.127403, -0.071408, 0.083241, -0.105686, -0.141731, -0.047969, 0.049211, -0.057316, -0.162159], [-0.06721, 0.08744, 0.05097, -0.030073, -0.216485, 0.066477, 0.04029, 0.01848, 0.065279, 0.03528, 0.160305, -0.447501, -0.128601, -0.144728, -0.015588, -0.105875, -0.046946, -0.10032, -0.073028, 0.097782, 0.047435, 0.041334, -0.001751, 0.141786, 0.027277, 0.103555, 0.060648, 0.088977, -0.007028, 0.087676, 0.093266, 0.117325], [-0.002308, 0.082076, -0.110591, 0.120891, -0.029195, -0.066904, -0.010086, -0.144935, 0.13943, 0.075986, -0.032133, 0.185235, 0.086891, -0.018876, -0.134312, 0.167406, 0.025344, -0.009703, 0.084386, 0.075619, 0.047191, 0.03318, 0.072613, -0.017548, 0.070707, -0.04309, 0.025249, 0.13334, 0.130409, 0.061231, 0.126596, 0.045183], [-0.036813, 0.102335, 0.06894, 0.071682, -0.077487, 0.010271, -0.176715, -0.062427, 0.168983, 0.048786, 0.033111, -0.077748, 0.176291, -0.128298, -0.010598, -0.105463, -0.057072, -0.097466, -0.338966, 0.013377, 0.181526, -0.070947, -0.347662, 0.123954, -0.134697, 0.092069, 0.149917, -0.080093, 0.138036, 0.062252, 0.157284, 0.076293], [0.025794, 0.0191, -0.025594, -0.031662, 0.140271, -0.040722, -0.025536, -0.102255, -0.109672, 0.002971, 0.085615, -0.030916, -0.050418, -0.036607, 0.220827, -0.132993, -0.024352, -0.007346, 0.117004, -0.04823, 0.024376, 0.086907, -0.063943, 0.064252, -0.0382, -0.089265, -0.01133, -0.059232, -0.074711, -0.172153, 0.135184, 0.029143], [-0.093405, 0.008659, -0.099843, 0.001449, -0.178581, 0.137444, -0.073068, -0.020587, -0.113741, -0.259991, -0.052621, 0.121761, -0.023371, -0.121054, -0.136986, 0.10643, -0.042187, 0.037956, 0.065892, -0.050392, 0.103145, -0.020548, 0.098207, 0.119891, 0.137102, -0.027509, -0.201854, -0.0178, 0.066477, 0.027501, -0.107855, -0.077971], [0.142758, -0.055575, -0.168076, -0.069544, 0.050091, 0.056035, -0.109229, 0.075337, 0.185681, 0.279792, 0.169809, -0.11825, 0.031337, 0.047078, -0.080657, -0.057813, 0.047173, 0.024886, 0.12812, -0.114777, -0.180041, -0.061312, 0.094983, 0.047077, 0.124634, -0.024586, 0.028022, 0.053216, 0.034945, 0.126119, 0.176572, -0.083922], [0.176561, -0.268087, -0.165337, -0.213245, -0.094195, 0.075113, 0.05688, 0.249577, 0.255972, -0.047131, -0.015521, 0.098188, -0.247916, 0.335929, -0.133955, -0.042935, 0.046648, 0.131778, 0.078201, -0.165277, -0.215755, -0.121193, 0.164241, 0.154037, 0.182464, 0.065426, -0.220423, 0.091087, -0.1253, 0.258681, -0.164289, -0.153453], [0.21824, -0.025547, -0.164681, -0.114656, -0.235652, 0.093913, 0.06158, 0.062187, -0.045263, 0.148358, -0.01718, -0.051685, -0.046373, -0.02766, 0.069021, 0.163896, 0.139331, 0.135214, 0.0561, -0.182968, -0.07739, -0.193419, 0.114442, 0.097365, -0.020261, -0.008816, -0.105467, -0.068821, -0.021955, 0.157802, -0.088036, 0.062615], [0.047674, -0.054499, -0.015205, 0.043787, -0.112766, 0.115232, -0.177679, -0.121398, 0.072269, -0.069827, -0.025657, 0.047025, 0.038627, -0.143306, -0.123594, -0.120496, -0.001463, 0.117129, 0.034294, 0.156261, 0.020356, -0.04656, -0.0318, 0.18163, -0.01291, 0.022613, 0.081118, -0.162065, 0.035608, -0.011344, 0.091226, 0.00162], [0.119078, -0.092469, 0.165267, -0.020191, -0.057058, 0.180959, 0.079928, 0.070053, 0.145294, -0.260826, -0.020657, -0.051071, -0.020626, 0.135429, 0.05057, -0.008553, -0.0856, 0.14432, 0.066045, 0.048728, 0.057845, -0.00354, 0.120926, 0.066543, 0.23088, 0.114493, 0.014699, -0.070233, 0.070671, 0.088259, 0.039874, -0.12696], [-0.070893, 0.00508, 0.022685, -0.142863, 0.063853, -0.102873, -0.146351, 0.145707, 0.067387, -0.026671, -0.049572, 0.121172, 0.005696, 0.171907, -0.08869, 0.017993, -0.056822, 0.170997, 0.126208, -0.110228, -0.046087, 0.153515, 0.048253, -0.201436, 0.088849, 0.102483, -0.037683, 0.190229, 0.005975, 0.181048, 0.012203, -0.18434], [-0.047127, 0.127618, -0.057151, 0.141624, -0.048704, 0.00692, -0.105983, -0.031142, -0.044448, -0.074904, 0.070078, 0.060314, 0.069618, -0.148036, -0.055594, 0.016482, -0.048602, -0.019311, 0.039759, -0.025458, 0.047926, -0.041421, -0.078844, -0.089615, 0.085885, 0.090702, 0.168985, -0.014853, 0.072981, -0.070731, 0.100599, 0.11661], [0.138064, 0.090371, 0.130879, 0.12025, 0.1311, -0.077299, -0.129858, -0.044096, -0.045073, 0.082363, 0.27053, -0.268152, 0.216108, -0.267807, -0.096033, 0.081946, 0.095458, -0.109165, -0.018781, 0.065464, 0.099646, 0.206804, -0.174781, -0.183378, -0.104643, -0.077456, 0.241817, 0.022998, 0.086475, 0.152022, 0.09823, 0.213082], [0.117315, 0.096572, 0.131356, 0.029738, -0.012046, -0.020712, 0.067749, -0.152602, 0.084643, 0.107538, 0.142246, 0.063827, 0.202691, -0.048805, -0.001964, 0.011248, -0.162521, 0.043987, 0.025607, 0.173399, 0.17906, 0.102188, -0.19155, -0.134813, 0.063806, -0.211817, -0.027503, 0.059503, 0.143225, -0.129092, 0.196711, 0.220817], [-0.09736, -0.097273, -0.005064, 0.000281, 0.022869, 0.137619, -0.138598, 0.036363, 0.01333, 0.143419, -0.007264, -0.157213, 0.011172, 0.041936, 0.127593, -0.086477, -0.029638, 0.017941, -0.039583, -0.052461, 0.020304, -0.075888, -0.006784, 0.049426, -0.044141, 0.022636, -0.180855, -0.160615, -0.095029, -0.245083, 0.052123, -0.041283], [-0.057723, 0.071036, 0.073439, -0.022573, -0.084224, -0.19158, -0.25154, -0.005747, -0.121896, -0.136442, -0.029144, 0.028051, 0.004493, -0.139978, 0.113094, 0.010003, -0.05122, -0.170686, -0.348127, 0.003853, 0.028195, 0.008425, -0.184917, -0.101479, -0.141401, -0.106603, -0.084838, -0.068315, 0.043236, -0.374084, -0.09108, -0.000943], [0.028312, 0.072337, 0.035646, 0.071822, -0.022985, -0.052613, -0.25228, 0.015145, 0.105054, 0.086968, -0.106692, 0.167372, -0.030244, 0.006096, -0.028438, -0.183785, -0.158948, -0.119825, -0.151213, 0.03737, 0.085402, 0.089577, -0.053291, -0.026809, -0.01715, -0.030344, -0.10006, 0.090528, 0.118629, -0.035644, 0.098092, -0.027069], [0.04637, 0.042065, 0.101358, 0.111277, 0.000257, 0.111465, -0.197581, 0.015603, -0.219826, 0.153917, 0.047928, -0.270899, 0.113369, 0.070025, 0.047553, 0.136076, -0.081243, 0.078148, -0.05869, 0.077384, 0.151825, 0.075128, -0.181947, -0.367979, 0.1811, 0.068957, -0.03487, -0.085678, -0.072842, -0.006053, 0.076854, 0.072815], [0.013102, 0.133858, -0.042729, -0.002585, 0.121817, 0.078767, -0.143349, -0.063879, -0.112563, 0.06475, -0.026327, -0.015974, -0.044687, -0.087959, -0.262559, 0.02203, 0.132063, -0.051362, 0.04553, 0.027582, 0.156721, 0.150852, 0.011341, 0.255385, -0.108406, -0.095042, 0.121403, -0.038479, 0.042824, -0.094168, 0.036315, 0.026729], [-0.097058, -0.06157, 0.009274, 0.077378, 0.022272, -0.11851, -0.050613, -0.033571, -0.016946, -0.004176, -0.009404, -0.014285, -0.046017, -0.081015, 0.164592, -0.077194, -0.11825, -0.098929, -0.059633, 0.111832, -0.138179, -0.037739, -0.178878, 0.1412, -0.076696, -0.154645, -0.080464, 0.010569, -0.033713, 0.065594, 0.087604, 0.221142], [0.107728, 0.004915, 0.119153, 0.09293, 0.100923, 0.088194, -0.131969, 0.101662, 0.089166, 0.010969, 0.010424, -0.038939, 0.097986, -0.113593, -0.092067, 0.158844, -0.047344, 0.093474, -0.050971, -0.055436, -0.024718, -0.043455, 0.003361, -0.000686, -0.031072, 0.082033, 0.145863, 0.051804, 0.103354, 0.021962, 0.12797, -0.02159], [0.120573, -0.041605, -0.124479, -0.097183, -0.108417, -0.064741, 0.113262, 0.071965, -0.221295, 0.011337, -0.102199, 0.055004, -0.044398, -0.028678, -0.094752, -0.151622, 0.077164, -0.036498, -0.099503, -0.108903, -0.037962, 0.117068, 0.122799, 0.027886, -0.150115, -0.084136, 0.022713, 0.038476, -0.077538, -0.078052, -0.011946, -0.106406], [-0.401698, 0.153658, 0.060627, 0.015515, -0.055995, -0.238465, -0.161067, -0.163213, -0.265737, 0.045338, 0.077158, -0.225134, 0.004015, -0.129957, -0.237663, -0.032712, -0.020583, -0.335259, -0.367808, 0.086514, 0.024839, -0.017481, -0.233438, -0.075368, -0.110285, 0.051386, -0.05015, -0.165892, 0.049902, 0.070691, 0.048613, 0.104559], [-0.103566, 0.147747, 0.191762, 0.023318, -0.035953, 0.032314, 0.186067, 0.059689, -0.368162, 0.120636, 0.05735, 0.11263, 0.112234, 0.068786, -0.215206, 0.065145, 0.054606, -0.097651, 0.091711, 0.025711, 0.14858, 0.149763, 0.029755, 0.068176, 0.010155, -0.114202, 0.069065, -0.001617, 0.099237, 0.013016, 0.170879, 0.009554], [0.037084, 0.153164, -0.121822, -0.031472, -0.06104, 0.053597, 0.161297, 0.140379, 0.011226, 0.046403, 0.191347, 0.047272, 0.038477, 0.185315, -0.106308, 0.110942, -0.136181, 0.164041, 0.131124, 0.108397, 0.05733, 0.117777, 0.125623, -0.248487, 0.04277, 0.0569, 0.086481, 0.056705, 0.159486, 0.005564, 0.065991, 0.01005], [0.082297, 0.192901, 0.066499, 0.095891, 0.319247, 0.041136, 0.000456, -0.00659, -0.030169, 0.03063, -0.057203, -0.09042, 0.109835, 0.044728, 0.089105, -0.086447, 0.059641, 0.064413, 0.03027, 0.159308, 0.020556, 0.1466, 0.073637, -0.038826, -0.050501, 0.06559, 0.126062, -0.140937, 0.037477, -0.019762, 0.212327, 0.198189], [0.000134, -0.037372, 0.038617, -0.09206, -0.135843, 0.022454, -0.186927, -0.052889, 0.090046, -0.2167, -0.291934, -0.214387, -0.006874, 0.038018, 0.215699, 0.059474, 0.035895, -0.015999, -0.027999, -0.083951, -0.12529, -0.187302, -0.200174, -0.069901, 0.102424, -0.002453, -0.21117, 0.164841, -0.193794, 0.095446, -0.13586, -0.009348], [0.10691, -0.080632, -0.041035, -0.149487, -0.291945, -0.0492, 0.095783, 0.062064, 0.067336, 0.07319, -0.080295, -0.030724, -0.090552, 0.055669, 0.09652, 0.175909, -0.035442, 0.042758, -0.021537, -0.002417, -0.019897, -0.093824, 0.113559, 0.235525, -0.038536, 0.087873, -0.043891, 0.075213, -0.134153, 0.05482, -0.046022, -0.230565], [0.105652, -0.148443, -0.041511, 0.018358, -0.127686, -0.102419, -0.11347, -0.033334, 0.159931, -0.315055, -0.200537, 0.218324, -0.144983, 0.199534, -0.184399, -0.13388, -0.053507, 0.057749, 0.029058, -0.166726, 0.149261, 0.077812, -0.113854, -0.116506, 0.001447, -0.075704, -0.058132, 0.246609, -0.024408, -0.084656, -0.139979, -0.087239], [-0.152625, -0.011017, 0.002955, 0.084162, -0.309094, 0.093036, 0.084083, 0.125435, -0.036517, -0.173564, -0.01643, -0.323883, -0.029831, -0.000186, 0.223419, 0.074056, -0.050027, 0.112862, -0.091474, -0.017886, 0.010127, -0.003304, 0.009415, 0.105546, 0.064148, 0.11558, 0.102406, 0.125615, -0.085557, -0.004261, -0.088847, -0.021099], [0.017934, -0.094649, 0.068431, 0.114721, -0.01972, -0.095486, 0.015429, -0.109412, -0.201472, 0.011349, -0.010438, -0.267647, 0.010067, -0.183086, -0.027975, -0.056217, 0.034446, -0.127816, -0.054936, 0.008, 0.03613, 0.138181, 0.107767, 0.045934, -0.126703, 0.096597, -0.029436, -0.315226, 0.040607, -0.233096, -0.066136, 0.193542], [-0.176455, 0.123618, -0.026485, -0.021205, 0.081918, -0.126917, -0.09037, -0.02879, 0.017253, 0.162092, 0.063088, 0.11411, -0.082669, -0.044385, -0.035838, -0.116183, -0.202775, -0.079129, -0.05662, 0.163032, -0.033301, -0.057245, 0.037925, 0.018596, 0.003715, -0.191366, -0.057724, -0.124602, 0.127067, -0.150179, 0.137574, -0.044866], [-0.073972, 0.074585, 0.045475, 0.116398, -0.028515, -0.034627, -0.209003, 0.00861, -0.076988, 0.082554, -0.047131, -0.079309, 0.013457, 0.021375, -0.282818, -0.146295, -0.087273, 0.088598, 0.094134, -0.053661, 0.056904, 0.011724, -0.078057, 0.185838, 0.058835, 0.104814, 0.040377, -0.055611, 0.07908, -0.250251, -0.054295, 0.070779], [0.059587, 0.016687, 0.08762, -0.121932, -0.087018, 0.07562, -0.056792, 0.023604, 0.153136, -0.06791, -0.075195, 0.131515, 0.071093, 0.170166, 0.138753, 0.089916, 0.139617, 0.093836, 0.107544, -0.118587, -0.044413, 0.122198, -0.033859, 0.029413, 0.060501, 0.104114, -0.10334, 0.076398, -0.186819, 0.047473, 0.029916, -0.150857], [0.128273, -0.114933, 0.011052, -0.058279, 0.074559, 0.088522, 0.155395, 0.171754, 0.044467, 0.151429, 0.00562, 0.144184, -0.041869, 0.013243, -0.289209, 0.034911, 0.103427, 0.095815, 0.119125, -0.031826, -0.075742, 0.069633, -0.102959, -0.004975, 0.006106, 0.139922, 0.163571, -0.029069, -0.018881, 0.063476, -0.042315, -0.101296], [-0.109083, 0.121772, -0.093902, 0.034155, -0.095676, -0.039272, 0.0641, -0.095093, -0.164208, 0.181546, 0.142632, -0.196712, 0.071605, -0.116096, 0.229531, 0.092536, 0.08428, -0.089869, -0.004003, -0.121252, -0.147555, 0.125024, -0.007888, -0.15686, 0.100875, 0.163062, 0.087815, -0.107013, -0.20381, -0.185559, 0.064753, -0.131571], [-0.2662, 0.032381, 0.101614, 0.140302, 0.046763, 0.025918, 0.128935, -0.115134, -0.02107, 0.15889, -0.043377, 0.034971, 0.128592, -0.148572, 0.009301, 0.075747, 0.127392, -0.104157, -0.093358, 0.091597, 0.104125, 0.162215, -0.092348, 0.088979, 0.12902, 0.032878, 0.138381, 0.085196, 0.139473, 0.074408, -0.003339, -0.042378], [0.135051, -0.059112, 0.034706, 0.118115, 0.055706, 0.172854, -0.029203, -0.042595, -0.104841, -0.044561, -0.381336, -0.234718, 0.123032, -0.057049, 0.161428, 0.019582, 0.182943, 0.160374, 0.075647, -0.076702, -0.008705, -0.03617, 0.13427, -0.185045, 0.135026, 0.105231, 0.013501, -0.081107, 0.069954, -0.000634, -0.050804, -0.032663], [-0.066646, 0.073725, -0.101248, 0.128246, -0.232721, 0.050702, -0.076875, -0.038847, -0.010899, 0.152417, 0.067136, -0.030434, -0.125561, 0.078376, -0.016684, -0.006467, 0.128273, 0.151728, -0.027601, -0.04586, -0.215487, -0.166713, -0.082083, 0.063728, 0.080063, 0.027646, 0.083705, 0.142283, -0.095248, -0.04059, 0.017893, 0.125591], [0.134876, 0.080918, 0.139471, -0.001928, 0.089186, -0.049801, -0.083488, -0.028388, 0.074323, 0.141621, 0.130319, 0.13582, -0.042293, 0.097129, -0.060429, -0.262964, 0.067892, 0.021196, 0.182205, 0.041047, 0.086261, 0.005189, 0.053376, -0.106, -0.052429, -0.171995, -0.002212, 0.120831, 0.036511, -0.025003, 0.149537, 0.057817], [0.156157, -0.108007, 0.014078, -0.055357, 0.021221, 0.075022, -0.195178, 0.077697, 0.128899, -0.157281, -0.078763, 0.081833, -0.114335, 0.095914, 0.030453, 0.040252, 0.185606, -0.008165, 0.21528, -0.142642, 0.061126, 0.010211, -0.140288, -0.033585, 0.115925, 0.040569, -0.147673, 0.047623, -0.050301, 0.278644, -0.049489, -0.23755], [-0.042813, 0.070642, -0.01373, -0.03469, 0.031257, -0.124888, -0.021034, -0.078705, 0.147317, -0.005886, 0.056686, 0.103752, 0.037767, 0.055963, -0.019765, -0.055906, -0.007064, 0.030967, -0.082124, 0.138723, 0.128873, 0.006788, 0.057275, -0.040005, -0.076527, 0.008997, -0.026282, 0.08811, 0.074459, -0.124976, 0.081978, 0.079208], [-0.209218, -0.045424, -0.028337, 0.025809, -0.015503, -0.246125, -0.388308, -0.103075, 0.022031, 0.002155, -0.070155, 0.211586, -0.109528, 0.013503, -0.053703, -0.257773, -0.007852, -0.169399, 0.152554, -0.090875, 0.093996, 0.032881, 0.111791, 0.258494, -0.038543, -0.158581, 0.10074, 0.057473, 0.118586, -0.096106, -0.052639, -0.100311], [-0.241684, 0.144334, -0.016887, 0.12124, -0.192654, -0.163349, 0.137126, 0.012976, -0.072159, -0.054531, 0.008229, -0.007439, 0.138014, 0.008276, -0.101836, 0.09173, -0.100953, -0.151267, 0.016851, -0.001644, -0.076983, 0.013378, -0.082137, 0.015158, -0.168733, 0.009447, -0.061598, -0.065145, 0.057991, 0.158697, 0.090404, 0.13173], [-0.00717, 0.049879, -0.107959, 0.014076, 0.241032, 0.097043, 0.067774, 0.028748, 0.131039, -0.110775, -0.129026, 0.018874, -0.166499, 0.217114, -0.147122, 0.090938, 0.070058, 0.173953, 0.116253, -0.041466, 0.012284, -0.010067, 0.139064, -0.076164, 0.137596, 0.110858, -0.104287, 0.183528, -0.005144, 0.151073, -0.094905, -0.026462], [-0.182599, 0.087118, -0.010192, -0.091103, 0.035277, 0.101738, -0.004205, -0.057268, -0.07819, 0.115222, 0.009214, -0.044033, -0.099125, 0.027721, 0.255155, -0.113473, 0.020701, -0.083612, -0.05958, -0.039457, 0.079016, 0.110846, -0.001837, -0.068018, 0.064162, -0.098028, 0.063423, -0.026323, -0.148938, -0.329904, -0.010772, 0.134784], [0.105625, 0.104177, -0.114484, -0.015727, 0.193266, 0.057037, 0.113174, 0.023572, -0.077173, 0.098988, -0.088911, 0.022919, -0.138014, 0.047854, -0.06255, 0.088547, 0.071222, 0.026912, -0.156626, 0.004303, 0.068873, 0.131077, -0.146337, -0.013743, 0.130224, 0.003509, -0.040385, 0.142961, 0.043121, 0.105356, 0.080123, 0.03631], [0.2629, -0.084868, 0.127231, -0.065461, 0.075564, 0.095463, -0.116619, 0.124803, 0.039063, 0.204334, -0.011358, 0.030901, 0.193626, 0.128491, -0.247288, -0.049122, 0.032873, 0.229885, 0.138976, 0.147246, 0.024118, 0.063547, -0.052319, -0.023265, 0.194353, -0.120315, 0.035683, 0.113755, 0.046909, 0.147033, 0.016564, 0.109562], [-0.140519, -0.043781, 0.050415, 0.072569, -0.115877, -0.080976, 0.052795, 0.101188, -0.127667, 0.020131, -0.037772, 0.084432, -0.06698, -0.13909, -0.102706, 0.000216, -0.131684, -0.038937, -0.042564, -0.104262, -0.014274, 0.073069, 0.055473, -0.047374, -0.007337, 0.120812, -0.08345, 0.064122, 0.079858, -0.066959, -0.061599, 0.088495], [0.169481, -0.062989, 0.017574, 0.049324, 0.007523, -0.017846, 0.121767, 0.010254, -0.060211, -0.012363, -0.023485, 0.00394, 0.03131, -0.03845, 0.136885, -0.029982, -0.039876, 0.083455, -0.104462, 0.114379, -0.155593, -0.105137, 0.102735, -0.166032, 0.050725, 0.054657, 0.060719, -0.115179, 0.081649, 0.096599, 0.002092, 0.160946], [0.02188, -0.012361, 0.112188, -0.016777, 0.068466, 0.051632, 0.142959, 0.025766, -0.020934, 0.155528, 0.108425, -0.100514, 0.040402, 0.072151, 0.171956, -0.1244, 0.13137, 0.000995, 0.075231, 0.12873, -0.066212, -0.023129, -0.045751, -0.041016, 0.070333, 0.07249, 0.00274, -0.161938, 0.075241, -0.056554, 0.084834, 0.000836]], [-0.161138, 0.124248, 0.060747, 0.137806, 0.00369, 0.021413, 0.025528, 0.081482, 0.004889, -0.182872, -0.021627, -0.105774, 0.013249, 0.023879, -0.024032, 0.141941, 0.089338, 0.028978, -0.144827, 0.010351, 0.105068, 0.048618, 0.085342, -0.052408, -0.030415, 0.115973, 0.055861, 0.030838, -0.036065, -0.139889, 0.074737, 0.081968]),
    ([[0.426921], [-0.181949], [-0.129349], [-0.026739], [-0.207084], [0.170953], [0.288678], [0.178611], [0.213613], [-0.355895], [-0.208785], [0.259766], [-0.117545], [0.188687], [-0.37404], [0.313265], [0.212559], [0.258485], [0.20207], [-0.228618], [-0.089999], [-0.157498], [0.196472], [0.235827], [0.161224], [0.11408], [-0.117405], [0.177544], [-0.113358], [0.23195], [-0.173738], [-0.069933]], [0.1083]),
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
        # exp_r3_ensb: number of extra GRU members averaged with the exported one (0 = off, 1 or 2)
        'ens_n': 0,
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
            n1 = len(SEQ_BH) // 3
            nt = len(SEQ_TM)
            mem = [(SEQ_WI, SEQ_BI, SEQ_WH, SEQ_BH, SEQ_HEAD), (SEQ2_WI, SEQ2_BI, SEQ2_WH, SEQ2_BH, SEQ2_HEAD),
                   (SEQ3_WI, SEQ3_BI, SEQ3_WH, SEQ3_BH, SEQ3_HEAD)][:1 + max(0, min(2, int(params['ens_n'])))]
            M = len(mem)
            n = self.sq_n = M * n1
            # gate layout over all members: [r (n) | z (n) | input part of the candidate (n) | hidden part (n)], member m
            # owning columns m*n1..(m+1)*n1 of each block and hidden rows nt + m*n1..; sigmoid = 0.5 + 0.5 tanh(x/2)
            W = np.zeros((nt + n, 4 * n))
            bias = np.zeros(4 * n)
            for m, (wi, bi, wh, bh, _hd) in enumerate(mem):
                wi = np.array(wi); bi = np.array(bi); wh = np.array(wh); bh = np.array(bh)
                wn = wi / self.sq_tsd[:, None]
                off = (self.sq_tm / self.sq_tsd) @ wi
                hr = slice(nt + m * n1, nt + (m + 1) * n1)
                for blk, src, half in ((0, 0, True), (1, 1, True)):
                    c = slice(blk * n + m * n1, blk * n + (m + 1) * n1); g = slice(src * n1, (src + 1) * n1)
                    W[:nt, c] = 0.5 * wn[:, g]; W[hr, c] = 0.5 * wh[:, g]
                    bias[c] = 0.5 * (bi[g] - off[g] + bh[g])
                g = slice(2 * n1, 3 * n1)
                c = slice(2 * n + m * n1, 2 * n + (m + 1) * n1); W[:nt, c] = wn[:, g]; bias[c] = bi[g] - off[g]
                c = slice(3 * n + m * n1, 3 * n + (m + 1) * n1); W[hr, c] = wh[:, g]; bias[c] = bh[g]
            self.sq_w = W
            self.sq_b = bias
            if M > 1:
                # heads: [x (22), h_1..h_M] -> M*64 -> M*32 -> 1 (the last layer averages the members)
                hd = [[(np.array(w), np.array(b)) for w, b in mm[4]] for mm in mem]
                nx = hd[0][0][0].shape[0] - n1
                a1 = hd[0][0][0].shape[1]; a2 = hd[0][1][0].shape[1]
                W1 = np.zeros((nx + n, M * a1)); B1 = np.zeros(M * a1)
                W2 = np.zeros((M * a1, M * a2)); B2 = np.zeros(M * a2)
                W3 = np.zeros((M * a2, 1)); B3 = np.zeros(1)
                for m, ((w1, b1), (w2, b2), (w3, b3)) in enumerate(hd):
                    W1[:nx, m * a1:(m + 1) * a1] = w1[:nx]; W1[nx + m * n1:nx + (m + 1) * n1, m * a1:(m + 1) * a1] = w1[nx:]
                    B1[m * a1:(m + 1) * a1] = b1
                    W2[m * a1:(m + 1) * a1, m * a2:(m + 1) * a2] = w2; B2[m * a2:(m + 1) * a2] = b2
                    W3[m * a2:(m + 1) * a2] = w3 / M; B3 += b3 / M
                self.sq_head = [(W1, B1), (W2, B2), (W3, B3)]
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
