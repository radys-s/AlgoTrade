# Round 2: final candidates (home, 2026-11-07 ~23:00)

The leaderboard tape is a different, ~2x longer realization of the round-2 generator (mark-out price 10,482); local
round2 gains scale ~x2.1, but parameter tweaks worth < ~3-4 % locally behave like +-10k noise there.

## Leaderboard results so far

| file | local round2 | leaderboard | note |
|---|---:|---:|---|
| strategies/mm_r2_fq2.py | 349,077 | 740,067 | aggressive hidden-fair IOC taker + F-side touch quotes + 30 s gate |
| **strategies/mm_r2_pred.py** | 371,961 | **797,904** | + ridge predictor for the taker (blend 0.5) — **current final** |
| strategies/mm_r2_pred2.py | 378,507 | 787,414 | gate 20 s / thr 10 — worse, do not use |
| strategies/mm_r2_pred_g6.py | 373,833 | 786,205 | gate 6 / 30 s — worse, do not use |

## Next candidate

**strategies/mm_r2_zd.py** = mm_r2_pred + zero-delay take (the flip IOC goes out in the same callback as the cancel of
our own blocking passive quote; a guard IOC trims any |I| > 100 at once; ~1k penalty per tape is inside the pnl) +
zero-delay requote toward the market. Better than mm_r2_pred on all four local tapes:

| tape | mm_r2_pred | mm_r2_zd |
|---|---:|---:|
| round2 | 371,961 | 392,436 |
| round2_mirror | 369,664 | 390,836 |
| round1 | 441,626 | 457,289 |
| sample_v2 | 265,868 | 267,405 |

check_submission OK, research.check PASS (1.5 min / 8.8M rows). Upload it; if the leaderboard shows more than
797,904, mark it final, otherwise keep mm_r2_pred as final.

## Not recommended

mm_r2_zd100 / mm_r2_pred_qs (quote_size 100): +2.5..11k on round2 but -29..-50k on other tapes (regime-fragile).

## Where the remaining money is (diagnostics, local round2)

- Front-of-queue oracle on the current structure: 622k vs 383k real (+239k): being early in thick queues. An
  F-guarded early-placed ladder + taker is being tested (strategies/exp_r2_ladderF.py).
- A perfect 150 ms predictor would give 935k, but every realistic estimator (Kalman, Tobit, CUSUM, multi-tau ridge,
  cancel flow) is at the information limit; the ridge in mm_r2_pred is the only one that helped (+23k).
- Over-cap, small-order grids without the taker, inside-spread quotes, sticky quotes, capacity release by IOC and the
  ML shrink all lose (docs/progress.md has every number).
