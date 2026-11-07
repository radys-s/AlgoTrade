# H19: shorter forecast horizon for the ML shrink (50 ms, 20 ms) - REJECTED

Tool: research.refit_ml on data/round1.csv, template strategies/mm_ml.py, label = mid(t+5ms+h) - mid(t+5ms), ml_thr at the 0.81 quantile of |pred|.
Files: strategies/exp_ml_h50.py, strategies/exp_ml_h20.py (train-frac 1.0); honest copies logs/ml_short/h{50,20}_tf75.py. Both pass `research.check --no-run`; diff vs mm_ml touches only LINEAR_MODEL and ml_thr.

## IC and thresholds (round1)
| h | IC fit (1.0) | IC fit (0.75) | IC holdout (last 25 %) | forecast std | label std | |pred| q70/80/90/95 | ml_thr auto (share) |
|---|---|---|---|---|---|---|---|
| 100 (champion) | n/a here | | | 0.11 | | | 0.15 |
| 50 ms | +0.2933 | +0.2947 | +0.2887 | 0.0593 | 0.2022 | 0.065/0.078/0.095/0.108 | 0.0798 (19.0 %) |
| 20 ms | +0.2073 | +0.2082 | +0.2045 | 0.0258 | 0.1244 | 0.027/0.033/0.041/0.047 | 0.0336 (19.1 %) |
(0.75 fits: thr 0.0805 / 0.0339.) IC is stable fit-vs-holdout, so no overfit; the IC is on a shorter label so it is not comparable to the 100 ms IC as a PnL predictor.

## Sweep (ledger tag r1-ml-short, ml_stub 5; champion mm_ml: round1 490,460 / sample_v2 254,061)
| h | ml_thr | dataset | pnl | train_pnl | hold_pnl | chunk_wins | chunk_min | fills | breach | inv_q997 | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 0.0638 | round1 | 487,443 | 362,233 | 125,210 | 8 | 40,963 | 49451 | 0 | 100 | 116.8 |
| 50 | 0.0798 | round1 | 487,319 | 362,136 | 125,183 | 8 | 42,256 | 47086 | 0 | 100 | 118.9 |
| 50 | 0.0997 | round1 | 469,001 | 347,508 | 121,493 | 8 | 40,392 | 44428 | 0 | 100 | 116.0 |
| 50 | 0.0638 | sample_v2 | 231,945 | 175,697 | 56,248 | 8 | 20,519 | 12394 | 0 | 100 | 72.3 |
| 50 | 0.0798 | sample_v2 | 244,176 | 184,562 | 59,614 | 8 | 21,779 | 11960 | 0 | 100 | 74.8 |
| 50 | 0.0997 | sample_v2 | 250,467 | 190,912 | 59,555 | 8 | 22,839 | 11011 | 0 | 100 | 76.3 |
| 20 | 0.0269 | round1 | 468,348 | 347,573 | 120,775 | 8 | 39,130 | 50085 | 0 | 100 | 125.2 |
| 20 | 0.0336 | round1 | 473,969 | 351,449 | 122,520 | 8 | 41,051 | 48145 | 0 | 100 | 126.1 |
| 20 | 0.042 | round1 | 461,417 | 341,751 | 119,666 | 8 | 40,047 | 45525 | 0 | 100 | 121.0 |
| 20 | 0.0269 | sample_v2 | 205,459 | 154,826 | 50,633 | 8 | 17,457 | 12354 | 0 | 100 | 74.9 |
| 20 | 0.0336 | sample_v2 | 222,124 | 167,710 | 54,414 | 8 | 19,864 | 11801 | 0 | 100 | 77.5 |
| 20 | 0.042 | sample_v2 | 233,711 | 177,303 | 56,408 | 8 | 21,202 | 11305 | 0 | 100 | 76.6 |
(chunk_wins = positive chunks, 8/8 everywhere.)

## Compare: mm_ml vs h50 @ ml_thr 0.0798, stub 5
- round1: pnl A 490,460 vs B 487,319 (-3,141); holdout 126,291 vs 125,183 (-1,108); per-chunk B-A [-838, 215, -194, -293, -415, -52, -9, -447], B wins 1/8.
- sample_v2: A 254,061 vs B 244,176 (-9,885); holdout 60,928 vs 59,614 (-1,314); chunks all negative, B wins 0/8.
- VERDICT: REJECT.

## Interpretation
1. Neither short horizon beats the 100 ms model on either tape; 50 ms is roughly equal on round1 (-0.6 %) and clearly worse on sample_v2 (-3.9 %), 20 ms is worse on both (-3 %, -9 %). H19 is not supported: the shrink timing is not improved by a sharper horizon.
2. Higher thresholds help on sample_v2 (h50 0.0997: 250k, h20 0.042: 234k, still rising at the grid edge) but hurt round1, so the quantile rule does not transfer between tapes for short horizons; the shorter the horizon, the less the forecast covers the quote-refresh time after a shrink.
3. Stay with mm_ml (100 ms); the horizon axis looks flat-to-negative on both sides (1000 ms lost 6k, 50 ms loses 3-10k). Not worth a further finer grid.
