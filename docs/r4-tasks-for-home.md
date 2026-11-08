# R4 tasks for the home brain (from radys, 2026-11-08 08:20)

Read `origin/radys` for the files. Judge every task the home way: holdout sums tape + mirror on slice A (last 25 %) AND
slice B (walk-forward 56-75 %), base = `mm_r4_mix` (x1.15 if that is the current pick). Stop a task as soon as it is flat.

## T1 (new alpha, highest priority): sweep riding

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
