# R4 tasks for the home brain (from radys, 2026-11-08 08:20, updated 08:40)

Read `origin/radys` for the files. Judge every task the home way: holdout sums tape + mirror on slice A (last 25 %) AND
slice B (walk-forward 56-75 %), base = `mm_r4_mix` (x1.15 if that is the current pick). Stop a task as soon as it is flat.

## T1 sweep riding - DEAD in the sim (radys 08:35), skip

Sim on round4 (rd_r4_sw, base rd_r4_q 1,035,606 / hold 198,036): sw_k 1 988,562 (188,953), sw_k 2 956,507,
sw_k 4 937,549. The GRU already trades these sweeps; the shift only adds late takes. `fq_deep` (IOC limit past the
touch at the forecast price) = no change at all (1,035,582). Offline evidence kept for the record:

Offline on round4_public (`research/diag/tape_ideas.py`, radys): a market IOC that trades at >= 3 price levels is
followed by continuation, measured at OUR real entry (far touch 5 ms later) and exit at mid:

| sweep levels | n / tape | +100 ms | +500 ms | +1 s | +3 s | win @500 ms |
|---|---:|---:|---:|---:|---:|---:|
| >= 2 | 9,458 | -0.56 | +0.73 | +1.71 | +3.88 | 0.46 |
| >= 3 | 1,580 | -0.02 | **+2.69** | **+4.71** | +9.98 | **0.74** |
| >= 4 | 527 | +0.17 | +3.24 | +5.59 | +12.78 | 0.77 |

(ticks per lot, after paying the spread). Unknown: how much of it the GRU already trades.
Mechanic in `strategies/rd_r4_sw.py` (radys, = s14 + R4 quote MLP): params `sw_k` (taker fair shift in ticks, in the
sweep direction, for `sw_ms`), `sw_lv` (min distinct trade prices of one market taker), `sw_q` (same shift on the
passive fair). All 0 = parity. Port the ~15 lines (search `rd_r4_sw`) onto mm_r4_mix.
Grid: `sw_k {1, 2, 4}` x `sw_lv {3, 4}` x `sw_ms {500, 1500}`, then `sw_q {0, 0.5, 1}` on the best. Runtime cost ~0.

## T2: taker ceiling gap

Your ceiling: perfect 205 ms taker 1.486M vs 1.027M. R4 moves are 1.4-1.9x larger and persistent (holds >= 10 s
earn 178k from 61 flips). Besides pred_scale, try a HOLD bias: after a take, raise the opposite-take threshold by
`k * EWMA(|dmid|, 10 s) / median` (vol-scaled hysteresis; static hysteresis was a dead end in R3, the vol-scaled
one was not tested).

## T3: queue priority (your plan's biggest lever)

Perfect 1 s quotes +164k on R4. If any R3-end queue agent (exp_r3_res / qcap / flive) worked, port it first.

## Dead on R4 (radys, do not repeat)

- Insert-price micro-EWMA (30 ms "fair leak"): raw corr 0.36 but +0.0003 holdout corr added to the MLP (fair_sig has it).
- 50 ms cancel imbalance: +0.0025 corr. Post-sweep REVERSION: the opposite happens (see T1).
- R4 MLP taker instead of the R3 GRU (seq_on 0): 1,029k / hold 202k vs 1,036k / 198k, same.
- Staged / graduated flips: our IOC is limited at the touch price, it never walks 3 levels; target-inventory policies
  were dead in R3.

## Update 08:55 (radys, after a Gemini idea round; ranked)

### T4 (new, do before T2): 10 s drift lockout on reverse flips
When at +100 (-100), block a take to the other side unless mid(t) - mid(t - 10 s) <= -D (>= +D); entries from flat /
small inventory are unchanged. Params `lk_s {5, 10, 20}` x `lk_d {0, 2, 4}` ticks, 0 = off. Offline check first (fill
probe / npz): split the reverse-flip takes of mm_r4_mix by the sign of the 10 s drift vs the flip; if flips AGAINST the
drift have a 1-5 s mark-out clearly below flips WITH it, implement; else drop. Targets the "flip back too early" gap
(holds >= 10 s = 178k from 61 flips).

### T2 addition: vol-elastic threshold
fq_take_k_eff = fq_take_k * clip(1 + g * (sig1s / median - 1), 0.7, 1.5), sig1s = 30 s EWMA of |1 s mid change|;
g {0.5, 1}. Merge with the vol-scaled hysteresis grid, same EWMA.

### T5 (low): trailing lock
After the unrealized move since entry peaks >= P ticks, exit to 0 by IOC when it gives back G. Offline first: mean
(peak - exit) per flip; implement only if give-back > 3 ticks/lot on flips with peak >= 6. P {6, 10} x G {3, 5}.

Rejected from that round: wall-delay of exits (250 ms delay loses the 200 ms edge), depth-capped IOC (our IOC is
already limited at the touch and re-takes on the next event).

## Update 09:10 (radys): T4 DEAD, skip it. Server: mm_r4_mix x1.15 = 1,720,970 (s14 1,676,168), uploaded as final.

- Local baseline mm_r4_mix x1.15 (pred_scale 1.15): round4 1,060,265 (hold 214,620), mirror 1,058,610 (215,873);
  server / local = 1.62.
- Fill probe (research/diag/flip_drift.py, radys): reverse flips AGAINST a >= 8 tick 20 s move are the best trades
  (83k lots, +5.7 ticks/lot @5 s, +15.9 @30 s); flips CHASING it look bad (63k lots, -0.1 @5 s, -7.9 @30 s).
- But blocking the chase flips in the sim (rd_r4_lk, lk_s 20): lk_d 8 855,821 / 863,748, lk_d 12 878,359 / 885,437
  (-17..-19 %). A blocked flip holds the wrong side through the move. Do not test drift lockouts or trailing exits
  built on 30 s mark-outs: our flips pay at 1 s, the 30 s reversion is not tradeable by holding.
- Remaining from radys: T2 (vol-elastic threshold / hysteresis) and T3 (queue priority) only.

## Update 09:35 (radys): flip completion / IOC past the touch DEAD on R4 too
- Flip anatomy (mm_r4_mix x1.15, round4 fill probe): 2,537 flips, 128 lots each, 87 % of the lots fill in the FIRST IOC
  at the touch (112 lots), 1.2 IOCs per flip, the later 13 % fill +2.3 ticks worse.
- IOC limit 1 tick past the touch from the first take (rd_r4_mixdeep, deep_max 1): deep_m -1 998,536 / 1,000,519
  (-5.7 %), deep_m -0.5 1,059,683 / 1,059,469 (flat). Execution is not the gap; the predictor and the queue are.

## Update 09:55 (radys): queue-priority ideas (Gemini round 2), base = mm_r4_mixv (agreed upload pick)

Ranked; radys runs the offline kill-check for Q1 now (`research/diag/cancel_probe.py`: every cancel of a resting quote
with market qty ahead + pred; counterfactual fill within 1 s and its 1 s / 5 s mark-out by queue-ahead bucket).
Result will be appended here. Only start Q1 / Q2 if that check shows ahead <= 15 cancels marking out >= +0.4 tick/lot.

- **Q1 (T3 concrete): survival keep.** Do not cancel a touch quote on a weak adverse prediction when the market qty
  ahead is expected to be gone within 100 ms: E[ahead after 100 ms] = sum over orders ahead of V_i * P(lifetime >
  age_i + 0.1 s | > age_i), lognormal lifetime (mu -1.2174, sigma 1.0720 in seconds, fitted on R4 inserts). Keep if
  E < 15 lots AND direction-signed pred >= -0.10. Order ids/ages: we already see market ids (fair_sig tracks them).
- **Q2: rank-dependent cancel threshold.** cancel_thr(ahead) = 0.20 * (ahead / ahead_at_insert)^1.5 - 0.15 (front of
  queue tolerates a mildly adverse prediction). Same check as Q1; can be one mechanic with Q1.
- Weak / rejected: L2 pre-aging + modify_order size-up (assumes a size increase keeps priority: verify in
  simulator/ before anything; pre-placing was dead in R3); order-id acceleration (unsigned event rate cannot predict a
  signed move; a vol-burst threshold is covered by exp_r4_vol); lower reverse-flip threshold at the cap (= lower
  fq_take_k, lost in every grid); IOC depth cap (our IOC is limited at the touch, it never walks).

### 10:05 result: Q1 / Q2 KILLED, do not implement
cancel_probe on mm_r4_mixv round4 (1,090,202; 22,429 cancels of resting quotes). Counterfactual "kept" quote, fill
proxy = our price becomes the touch within 1 s (optimistic), mark-out per lot:
ahead 0: n 1,993, mo1 -0.21, mo5 +0.10 | ahead 1-15: n 1,260, mo1 -0.56, mo5 -0.50 | 16-50: +0.05 | >200: -0.20.
Ahead <= 15 by direction-signed pred: <= -0.3: -3.09; -0.3..-0.1: -0.46; -0.1..0.1 (the proposed keep zone): -0.90;
0.1..0.3: +0.53; > 0.3: +2.24 (those are re-pricings, not losses). Kill line was +0.4: our cancels are right; queue
keeping by survival / rank does not pay on R4.

## Update 10:20 (radys)
- 500 ms quote MLP blended into the TAKER prediction (rd_r4_tkq on mm_r4_mixv, `tk_q`): 0.3 round4 1,093,706 (223,302)
  / mirror 1,094,720 (223,950) vs mixv 1,090,202 (222,726) / 1,087,684 (222,767): +0.3 / +0.6 %, noise; 0.6 -4 %. Dead.
- **T6 (biggest remaining idea, home compute): volatility-normalized models.** R4 moves are 1.4-1.9x R3; the R3 GRU
  and the tables see out-of-range inputs, and pred_scale / gate / slw scaling (mixv, exp_r4_vol) only patches the
  output. Retrain GRU + MLPs with every price-difference feature and the target divided by a causal EWMA vol (e.g.
  hl 60 s of |1 s mid change|), multiply the prediction back by the same vol at run time. Train on round3 + round2 +
  round4 0-56 % (+ mirrors), judge on both slices. This makes one model valid in all regimes, which is what the private
  seed needs. If short on time: do it for the MLP taker (seq_on 0) only, which radys found equal to the GRU on R4.

## Update 10:45 (radys): Gemini "structural" ideas checked against simulator code. None are worth a run.

- **Over-cap penalty arbitrage.** The penalty really is flat (`simulator.py:1290`, 10k $/s whenever |inv| > 100,
  no matter how far over), and orders up to |inv| 1000 are accepted (`MAX_POTENTIAL_POSITION`). But there is no
  liquidity to take. Our flip IOCs ask for up to 200 lots and fill on average 112 at the touch, so the touch is already
  emptied. Extra lots would come from deeper levels, and IOC past the touch was already measured at -5.7 %.
  Over-cap quoting lost -133k..-316k in R1 (`exp_overcap.py`). Skip.
- **Replay front-running / liquidity hollowing** (take L1, rest at L1+1, so a replayed IOC "walks" into us).
  `_execute_taker` stops at the IOC's own limit price. An IOC at the touch whose liquidity we consumed fills nothing,
  because it is a zombie. It does not walk. Only IOCs whose limit price was already deep reach L1+1, and there we
  sit behind the historical L2 queue. That is the sweep case, which the GRU already trades (sweep riding was -4.5 %). Skip.
- **Path-length micro-cycling, trailing lock** = graduated / partial flips. Dead in R3 and R4 (lockout -17 %). Skip.
- **Order-age queue promotion**: the cancel probe already measured it. Near-front cancels mark out +0.10 / -0.50
  and the keep zone -0.90. Skip.
- **Inside-spread quoting on wide spreads** = `fq_inside`, dead in R3. Skip.

Still open on your side: T6 (vol-normalized GRU/MLP retrain) and the vol-adaptive variants. Upload only if both
slices (A and B) beat mm_r4_mixv by >= 3 % (server 1,763,582 = current final).

## T7 (radys 11:05): taker predictor upgrade, the only lever with measured headroom

Why: the perfect 205 ms taker makes 1.486M local vs mixv 1.090M. The board 2M needs ~1.23M local, about 1/3 of that gap.
All execution mechanics we tried were flat or negative. The predictor itself has not changed since R3 (16-hidden GRU,
and your own R3 note says the exported GRU is partly a lucky seed, +-20k between retrains).

Steps, cheapest first. Stop at the first flat step:
1. **Offline first (minutes).** On the round4 + mirror dumps, measure the holdout corr of the current GRU taker output vs
   the 205 ms target. That is the baseline number.
2. **Seed ensemble.** Train 4-5 GRUs (same arch, `seq_r3.py`, trained on round3 + round4 + both mirrors, 0-75 %)
   and average their outputs offline. Seed averaging removes the lucky-seed noise and normally adds corr.
   Also try hidden 24/32 for a single model. Gate: holdout corr >= baseline + 0.01, otherwise stop.
3. **Runtime.** The GRU costs ~45 us/event (check was 6.7 min). Three GRUs fit inside 15 min. Five probably do not, so
   check `research.check` runtime before any grid.
4. Bake the ensemble into mixv. Upload only if slice A **and** slice B (tape + mirror) beat mixv by >= 3 %.

## T8 (radys 11:40): add multi-level order flow (MLOFI) and multi-horizon targets to the T7 retrain

Source: Xu-Gould-Howison 2018 (multi-level OFI) and Kolm-Turiel-Westray 2023 ("Deep order flow imbalance": models on
order flow beat models on raw book states; multi-horizon "alpha term structure" outputs).

Measured (radys, `research/diag/mlofi_probe.py` on origin/radys, round4_public, OLS on our 26 npz features, fit 0-75 %,
holdout corr 75-100 %):

| target | own | own + MLOFI | MLOFI only |
|---|---:|---:|---:|
| 200 ms | 0.5502 | **0.5561** (+0.006) | 0.395 |
| 1 s | 0.5942 | **0.6024** (+0.008) | 0.422 |

MLOFI = signed flow per depth bucket 0..4 ticks from the same-side best (bid add +v, bid cancel/consumed -v, ask add -v,
ask cancel/consumed +v), summed over 50 ms / 200 ms / 1 s windows, signed log1p. 15 cols, and the per-event update is
O(1) in the strategy.
It beats every earlier add-on (fs30 +0.0003, ci50 +0.0025) but is below a standalone upload bar. Use it only inside T7:
- Add the 15 MLOFI cols to the GRU head inputs (or the 5 per-event bucket flows to the GRU tokens).
- Train with multi-horizon outputs (100 / 200 / 500 / 1000 ms). Use the 200 ms output for the taker and optionally the
  500 ms output for quotes. Multi-task targets usually regularise the main head.
- Same gate as T7: holdout corr +0.01 offline, then both slices +3 % vs mixv.

## Update 12:30 (radys): T8 MLOFI dead in a nonlinear model; T7 ensemble is small. Both measured here.

MLP 32-16, h200, trained on round4 + mirror 0-75 %, holdout corr 75-100 % (`research/diag/mlp_fit.py`, radys):

| model | round4 | mirror |
|---|---:|---:|
| own 26 feats, seed 0 / 1 | 0.6570 / 0.6574 | 0.6590 / 0.6576 |
| own + 15 MLOFI, seed 0 / 1 | 0.6580 / 0.6557 | 0.6585 / 0.6564 |
| avg of 2 seeds | 0.6609 | 0.6618 |
| avg of 4 seeds | **0.6627** | **0.6639** |

- **MLOFI: dead.** The OLS gain (+0.006) disappears in the MLP. The nonlinear model already gets that information. Drop it from T8.
- **Seed ensemble: +0.005 corr** with 4 seeds, consistent on both tapes. By R3 scaling (~+0.05 corr = +10 %) that is about +1 % pnl,
  below the 3 % upload bar. Run the GRU ensemble only if the home PC is idle. The GRU had larger seed variance (+-20k), so
  its gain may be larger. Check the offline corr of 3 GRUs first, then sim only if it is >= +0.01.
- Multi-horizon heads (rest of T8) are untested. Same offline gate.

## T9 (radys, URGENT, top priority): full-depth GRU. The only lever left that adds information

Measured here: a bigger MLP does NOT help (64-32: holdout corr 0.6555 / 0.6547 vs 32-16: 0.6570 / 0.6590). The model is
limited by information, not capacity. The file has ~800 kB free, so size is no constraint.
mixv runs `SEQ_LITE = True` (touch-only tokens). `seq_r3.py` already supports the full top-5 tokens (LITE False) and larger GRUs.

1. Train on round3 + round4 + both mirrors, 0-75 %: (a) LITE False, hidden 16; (b) LITE False, hidden 32;
   (c) LITE True, hidden 32. Report holdout corr vs the current mixv GRU on round4 + mirror.
2. Gate +0.01 corr. Then bake the best one into mixv. The strategy must build the level 2-5 tokens online, so check
   `research.check` runtime (< 12 min).
3. Average 2-3 seeds of the winner (+0.005 more, measured on the MLP).
4. Upload only if slice A and B beat mixv by >= 3 %.

## Server: mm_r4_mixv3 = 1,810,997 (10:23, +47k / +2.7 % vs mixv 1,763,582) -> FINAL. Ratio 1.645 to round4 local.

Penalty 902 $, |pos| quantiles 100/100/100/100, 18,148 fills, SE 54k. Two-tier take works on the server, so extend it:
- radys is testing a third tier now (`strategies/rd_r4_t3.py` on origin/radys: `ht2_d` weaker signal -> partial target
  `ht2_tgt` -50, checked after the ht_d tier). Result follows.
- Please grid on mixv3: `ht_tgt {0, 30, 50}` (partial cross instead of flat) and `ht_d {0.08, 0.16}`. Report slice A/B.

## Update 10:40 (radys): two-tier extensions dead; please switch to T9 now

On mixv3, round4 full (mixv3 1,101,157 / hold 224,861):
- 3rd tier (`ht2_d` 0.2 / 0.3 -> partial target -50): 1,100,930 (225,691) / 1,097,826 (225,038). Flat.
- `ht_sp 2` (two-tier also at a 2-tick spread), `ht_min` 10 / 5: 1,099,210 / 1,099,174 (224,658). Flat.
- mixv4 (gate 48) is +0.3 %, within server noise (SE 54k). Do not spend an upload on it alone.

Gap to 2M is +10 %, and only a better predictor can close it. Your own dissection says 85-90 % of the oracle gap is
predictor latency. Please run **T9 (full top-5 tokens GRU, hidden 32)** next, before any more small grids, and add one
variant aimed at latency: an early-move target (mid move t -> t+1 s, sampled only when |move over the last 100 ms| <= 1 tick),
so the model learns the start of a move rather than its continuation. Gate: offline stream corr +0.01, then slice A/B +3 %.

## Update 11:00 (radys): early-move target is dead offline; skip it, keep T9 depth tokens

`research/diag/early_probe.py` (radys): on QUIET rows (|mid move, last 100 ms| <= 0.5 tick, 77 % of rows), holdout corr
with the 1 s move: h200 MLP 0.4565 / 0.4532, MLP trained on quiet rows with the 1 s target 0.4573 / 0.4537, all rows 1 s
target 0.4593 / 0.4563 (round4 / mirror). The 26 features carry no earlier information. Lateness can only shrink with
NEW inputs: T9 (top-5 tokens) is the remaining test.

## T10 (radys, after T9): quote signal from the GRU state

The perfect 1 s quotes ceiling is +164k (+15 %), and cs_q alone gave +13k, so the quote signal is under-used. MLPQ500 is
a 26-feature MLP. Train a 500 ms head on the GRU hidden state + features (the same streaming GRU as the taker, so zero
extra recurrent cost; only a second small head). Gate: offline corr vs MLPQ500 +0.01 at 500 ms, then slice A/B.

## Note 10:48 (radys): your running "multi-tier policy" = my rd_r4_t3 (flat). Stop it if it is the same; keep the quote upgrade.
3rd tier ht2_d 0.2 / 0.3 -> -50 on mixv3: round4 1,100,930 / 1,097,826 vs 1,101,157 (hold 225,691 / 225,038 vs 224,861).
