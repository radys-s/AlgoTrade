# round1 MLP 100 ms forecast vs ridge (mm_ml)

Split: time order, train first 55 %, valid next 20 % (epoch pick), test last 25 % (2.20M rows, stride 2, 23 OnlineFeatures
in OnlineFeatures order). Ridge refit on train+valid (alpha 10, winsorised 0.1 %); MLPs get the same clipping bounds.
Scripts: logs/mlp_r1/run.py, gen.py, bench.py.

| model | test IC | top1% / bot1% edge (ticks) | top10% / bot10% | fc std | frac abs>0.15 |
|---|---|---|---|---|---|
| ridge | 0.375 | +0.36 / -0.41 | +0.22 / -0.23 | 0.111 | 19.0 % |
| MLP 16 | 0.4468 | +0.50 / -0.49 | +0.24 / -0.25 | 0.130 | 15.4 % |
| MLP 32 | 0.4467 | +0.50 / -0.52 | +0.24 / -0.25 | 0.133 | 16.4 % |
| MLP 64 | 0.4452 | +0.51 / -0.50 | +0.24 / -0.25 | 0.134 | 17.4 % |
| MLP 16-16 | 0.4464 | +0.50 / -0.52 | +0.24 / -0.25 | 0.132 | 16.6 % |
| MLP 32-16 | 0.4472 | +0.51 / -0.52 | +0.24 / -0.25 | 0.134 | 16.8 % |
| MLP 32-32 | 0.4501 | +0.50 / -0.51 | +0.24 / -0.25 | 0.132 | 16.4 % |

Chosen: 23-16-1 (within 0.01 of best). Pure-Python forward ~10 us/event (23 clips + 368 MACs).
exp_ml_mlp.py: parity ridge 490,460 exact. check (mlp) PASS, 34.6 us/event overhead, projected 341 s (5.7 min) for 8.8M rows.

## Sweep (tag r1-ml-mlp, round1, ml_model mlp)
| ml_thr | pnl | hold | fills |
|---|---|---|---|
| 0.10 | 469,064 | 119,499 | 49,049 |
| 0.125 | 466,972 | 119,687 | 47,181 |
| 0.15 | 464,022 | 120,813 | 46,135 |
| 0.20 | 461,267 | 119,812 | 44,771 |
| 0.25 | 458,567 | 119,145 | 43,807 |
ridge @0.15: 490,460 (hold 126,291).

## Compare (mm_ml vs mlp thr 0.1): REJECT
round1 -21,396 (0/8 chunks, holdout -6,792); sample_v2 254,061 -> 198,530 (-55,531, 0/8 chunks).

## Interpretation
- The MLP is clearly the better forecaster (IC +0.07, ~40 % more edge in the tails), but that does not turn into PnL.
- The shrink rule only gains from large, rare, confident calls; the MLP's extra IC is in the bulk, and PnL falls monotonically as thr rises, so shrink fires "too rarely" at 0.15 yet more firing (0.10) still stays below ridge.
- Ridge (winsorised, linear in flow/imbalance) probably flags exactly the run-over events that matter (big flow, depth imbalance); the MLP may redistribute forecast mass to small-move regimes that do not hurt a resting quote.
- Big loss on sample_v2 suggests the net, trained only on round1, does not transfer across tapes; retrain on both before any further try, or use a different decision rule (e.g. loss-weighted label) instead of mean-move MSE.
