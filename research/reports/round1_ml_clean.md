# round1: clean (non-leaky) ML model and runtime of ml_every

## 1. Leakage
Clean ridge: same 23-feature `LINEAR_MODEL` format, h = 100 ms, alpha 10, fitted on round1 rows with
time <= 75 % of the span (train rows 1,651,965, label purge 1 s; holdout rows 551,615) via `logs/ml_taker/fit.py`.
Model in `strategies/exp_ml_clean.py` (copy of mm_ml, only LINEAR_MODEL replaced; scratch in `logs/ml_clean/`).

| set | IC | top10% edge up/down (ticks) | top1% edge up/down |
|---|---|---|---|
| train | +0.3833 | +0.228 / -0.233 | +0.417 / -0.431 |
| holdout | +0.3752 | +0.219 / -0.228 | +0.361 / -0.406 |

IC barely drops out of sample (0.383 -> 0.375): the ridge is not overfitted.

| strategy | dataset | pnl | train_pnl | hold_pnl |
|---|---|---|---|---|
| mm_park | round1 | 452,651 | 334,765 | 117,886 |
| mm_ml (leaky weights) | round1 | 490,460 | 364,169 | 126,291 |
| exp_ml_clean | round1 | 490,905 | 364,777 | 126,128 |
| mm_park | sample_v2 | 251,236 | 192,143 | 59,093 |
| mm_ml (leaky) | sample_v2 | 254,061 | 193,133 | 60,928 |
| exp_ml_clean | sample_v2 | 253,400 | 192,646 | 60,754 |

Holdout gain over mm_park: clean +8,242 on round1 (leaky +8,405), +1,661 on sample_v2 (leaky +1,835).
Paired chunks (mm_park vs exp_ml_clean): round1 B wins 8/8 (diffs +3.3k..+4.4k per chunk), sample_v2 4/8
(+2,164 total, chunk diffs mixed, sample_v2 is not a chunk-level win). **VERDICT: ACCEPT.**

## 2. Runtime (research.check, round1, 4.41M rows, 4 runs in parallel so absolute us are slightly pessimistic)
Cost split: the state update of `_FullFeatures` (deques) must run every event; only the dict build
(depth loops) and the ridge dot product are skipped on non-evaluated events.

| setting | overhead us/event | projected min / 8.8M rows | round1 pnl |
|---|---|---|---|
| ml_every 1 | 24.4 | 4.2 | 490,905 |
| ml_every 2 | 17.1 | 3.1 | 489,238 |
| ml_every 4 | 12.9 | 2.5 | 487,276 |
| ml_every 1 + touch_change | 18.4 | 3.3 | 490,809 |
| ml_every 2 + touch_change | 16.7 | 3.1 | 488,813 |
| ml_every 4 + touch_change | 13.6 | 2.7 | 487,062 |

(Projection was 4.2 min even for ml_every 1, below the 7 min estimated earlier.)

## 3. Interpretation
- The ML gain is real out of sample: clean holdout IC 0.375 vs train 0.383, clean holdout gain +8.2k vs +8.4k leaky; 8/8 chunk wins on round1.
- sample_v2 gain is small (+2.2k, 4/8 chunks): the stub-shrink is a round1-sized effect, do not oversell it.
- ml_every costs pnl monotonically (-0.3 % at 2, -0.7 % at 4); `ml_on_touch_change=1` is within 100 pnl of ml_every 1 (490,809 vs 490,905) and saves 6 us/event.
- Bake ml_every 1 with ml_on_touch_change 1 (3.3 min projected, far below the 7 min target); use ml_every 2 only if more runtime is needed.
