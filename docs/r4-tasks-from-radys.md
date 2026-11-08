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
