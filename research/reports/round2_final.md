# Round 2: final candidates (home, 2026-11-07 ~23:25)

The leaderboard tape is a different, ~2x longer realization of the round-2 generator (mark-out price 10,482). Taker
improvements transferred ~x2.5 (mm_r2_pred +58k, mm_r2_zd +52k); gate tweaks worth +2..7k locally lost ~11k there.
Final scoring is on private data from the same market, so pick the best EXPECTED strategy, not the luckiest upload.

## Leaderboard results so far

| file | local round2 | leaderboard | note |
|---|---:|---:|---|
| strategies/mm_r2_fq2.py | 349,077 | 740,067 | aggressive hidden-fair IOC taker + F-side touch quotes + 30 s gate |
| strategies/mm_r2_pred.py | 371,961 | 797,904 | + ridge predictor for the taker (blend 0.5) |
| strategies/mm_r2_pred2.py | 378,507 | 787,414 | gate 20 s / thr 10 - worse |
| strategies/mm_r2_pred_g6.py | 373,833 | 786,205 | gate 6 / 30 s - worse |
| **strategies/mm_r2_zd.py** | **392,436** | **849,747** | + zero-delay flip take and requote - **current final** |

Public chart at 22:53: 75th percentile ~860k (+-10k), median 288k; the top edge of the band moved with our own uploads.

## Rejected after mm_r2_zd (round2 + mirror sum, base 783,271)

| idea | result |
|---|---|
| F-guarded small-order ladder (queue age) | +80 / -730 (pred_qs), -336 / -190 (zd) |
| order-statistic F (max bid / min ask insert) in taker or quotes | 529-755k |
| deeper taker IOC when F_hat is far (sk) | never triggers (0) |
| taker needs ridge AND fair_sig | 727-737k |
| over-cap trade on hidden-F jumps | 768-781k (jump inserts are > 20 ticks away = invisible; the book collapses in 20-50 ms) |
| follow / fade big sweeps with the taker | -390k..682k / -99k..663k |
| taker slip 1 / 2 ticks | 661k / 615k |
| thin-touch pull | 706-744k |
| flow pull after big sweeps (300 lots, 200-300 ms) | 789-795k (+6..11k) but round1 -45k, sample_v2 -18k: noise-level, not used |

## Where the remaining money is

- A perfect 150 ms predictor would make 935k locally: precision at the taker's threshold crossings is the lever.
  Bulk correlation gains (order statistics, multi-tau, pred2) do not carry over; only the ridge did (+23k / +58k LB).
- Queue priority (+239k oracle) cannot be reached with resting orders: deep rungs only fill on informed sweeps.
