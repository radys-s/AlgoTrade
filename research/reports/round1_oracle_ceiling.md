# Round1 oracle ceiling of "shrink on toxicity signal" (diag_oracle_ml.py, lookahead, never submit)

Oracle pred = true mid(t+5ms+h) - mid(t+5ms) in ticks from data/features/round1.npz. Same shrink rule as mm_ml
(side about to be run over -> ml_stub lots). Parity oracle=0: 490,460 (exact). Mid moves in half ticks, so ml_thr 0.05..0.25 are identical.

| oracle h | ml_thr | ml_stub | pnl | holdout | fills |
|---|---|---|---|---|---|
| ridge (champion) | 0.15 | 5 | 490,460 | 126,291 | 46,527 |
| none (mm_park) | - | - | 452,651 | - | - |
| 100 ms | 0.05-0.25 | 5 | 539,181 | 140,810 | 48,852 |
| 100 ms | 0.5 | 5 | 601,310 | 152,441 | 45,869 |
| 100 ms | 0.15 | 0 | 545,760 | 142,692 | 31,853 |
| 100 ms | 0.15 | 20 | 520,324 | 135,730 | 48,230 |
| 1000 ms | 0.05-0.25 | 5 | 808,123 | 200,080 | 48,075 |
| 1000 ms | 0.5 (only truly toxic) | 5 | 886,570 | 217,686 | 46,974 |
| 1000 ms | 0.15 | 0 | 846,217 | 209,177 | 15,855 |
| 1000 ms | 0.15 | 20 | 693,086 | 172,776 | 50,407 |

All cells: 8/8 chunks, no breach.

## Ceiling
- Perfect 100 ms foresight is worth only +49k to +111k over mm_park (539k-601k). The ridge (IC 0.38) already captures 38k of that
  (490k); the remaining headroom at the 100 ms horizon is ~50-110k, reachable only with a near-perfect signal.
- Perfect 1 s foresight reaches 808k-887k, but a 1 s mid move is not predictable at IC 0.38-level from this tape; that number is an upper bound, not a target.
- Stub size: for perfect knowledge a smaller stub is better at 1 s (0: 846k, 5: 808k, 20: 693k); at 100 ms the full pull has more pnl (546k vs 539k) but halves fills.
- Avoiding only truly toxic fills (thr 0.5) beats shrinking on any move, for both horizons (+62k at 100 ms, +78k at 1 s).

## Verdict
1. A 100 ms signal inside this maker is capped near 540-600k even with perfect foresight; 650k is out of reach for a 100 ms-horizon shrink signal.
2. Reaching 650k+ needs foresight of roughly the 1 s mid move (oracle 808k+), i.e. a much longer-horizon predictor than ours.
3. So a competitor at ~650k either predicts the multi-hundred-ms to 1 s move far better than IC 0.38 at 100 ms, or has an edge outside toxicity filtering (queue/sizing/other layers), or overfit public data.
4. Search next: longer-horizon (300 ms-1 s) targets and the "avoid only large adverse moves" objective (thr 0.5 gain), rather than refining the 100 ms ridge.
5. Task item 4 is covered by the thr 0.5 h 1000 cell (886,570): the ceiling of avoiding only truly toxic fills.
