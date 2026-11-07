# Round 2: final summary (home, 2026-11-08 00:05)

The leaderboard tape is a different, ~2x longer realization of the round-2 generator (mark-out price 10,482). Final
scoring is on private data from the same market. Mark as final the upload with the highest leaderboard pnl; the safe
fallback is `strategies/mm_r2_rth.py` (970,101).

## Leaderboard progression

| file | local round2 | local mirror | leaderboard | what it added |
|---|---:|---:|---:|---|
| strategies/mm_r2_fq2.py | 349,077 | - | 740,067 | hidden-fair IOC taker + F-side touch quotes + 30 s gate |
| strategies/mm_r2_pred.py | 371,961 | 369,664 | 797,904 | ridge predictor for the taker (blend 0.5) |
| strategies/mm_r2_pred2.py | 378,507 | 374,093 | 787,414 | gate 20 s / thr 10 - worse |
| strategies/mm_r2_pred_g6.py | 373,833 | - | 786,205 | gate 6 / 30 s - worse |
| strategies/mm_r2_zd.py | 392,436 | 390,836 | 849,747 | zero-delay flip take and requote |
| strategies/mm_r2_rtq.py | 406,182 | 404,769 | 891,735 | no quote 20 ms after our touch retreats; 100-lot quote on a weak lean |
| strategies/mm_r2_rts.py | 420,715 | 419,434 | 957,133 | at spread >= 2 quote only on a 0.7-tick lean |
| **strategies/mm_r2_rth.py** | **430,781** | **429,296** | **970,101** | stricter quote over a >= 3-tick hole behind the touch |
| strategies/mm_r2_rth3.py | 436,496 | 439,487 | 954,077 | + pred_blend 0.6 (taker) - worse, gain was in the ridge fit window |
| strategies/mm_r2_final.py | 434,477 | 432,324 | (last upload) | rth + sp2 0.75, sp3 0.15, hk_k 0.6, qs_k 0.3; taker unchanged |

## What worked

1. Trade with the hidden fair price F (limit inserts land at F -+ D): aggressive IOC taker when F_hat is past the far
   touch, passive touch quotes only on the side F favours, inventory held at +-100.
2. Execution latency: the flip IOC in the same callback as the cancel of our blocking quote (zd).
3. Passive-side state filters, where pulling costs no queue place: a just-retreated touch, a wide spread, a hole behind
   the touch. They transferred x1.3-4.5 to the leaderboard; taker-side tweaks transferred x2.5-3 when real, and the
   in-sample-only ones (pred_blend 0.6) lost.

## What did not (all logged in docs/progress.md)

Queue-age ladders, order-statistic F, over-cap trades on F jumps (jump inserts are outside the 20-tick snapshot),
following or fading sweeps, taker slip, AND-taker, thin-touch / shallow-depth / few-orders / imbalance / own-fill /
trade-at-touch pulls, new-only filter variants, mid-jump filter, refitted ridges with extra features, gate retunes.
