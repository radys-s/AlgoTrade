# H21 big-move ridge scores (round1)

Code: `research/fit_bigmove.py` (fit + precision), `strategies/exp_ml_bigmove.py` (PARAMS `ml_target` adds D1, D4; default B = parity 503,148 / 258,042), scratch `logs/ml_bigmove/`.
Same 23 features, winsorise, standardise, alpha 10; fit on first 75 % of round1 time (1.2 s purge), tested on the last 25 %. thr = 0.5 tick.
Share of events followed by a >= 0.5 tick 100 ms move: 13.7 % (holdout 13.5 %). D1 nonzero 13.7 %, D2 29.6 %, D3 11.5 %, D4 15.4 %.
Precision = share of the top q of |score| whose mid(t+105ms)-mid(t+5ms) moved >= 0.5 tick in the score's direction (holdout). B and base are refit on the same train split.

| score | IC own target | IC plain 100 ms | IC vs D1 | prec top 5 % | top 10 % | top 20 % |
|---|---:|---:|---:|---:|---:|---:|
| base (plain ridge) | +0.3753 | +0.3753 | +0.3662 | 0.411 | 0.360 | 0.283 |
| B | +0.3015 | +0.3693 | +0.3580 | 0.415 | 0.362 | 0.278 |
| D1 | +0.3672 | +0.3744 | +0.3672 | 0.418 | 0.367 | 0.287 |
| D2 (300 ms) | +0.4836 | +0.3739 | +0.3648 | 0.403 | 0.355 | 0.282 |
| D3 | +0.2924 | +0.3663 | +0.3553 | 0.420 | 0.367 | 0.278 |
| D4 (max excursion) | +0.3709 | +0.3725 | +0.3650 | 0.422 | 0.368 | 0.286 |

Chosen for the strategy: D1 and D4 (best precision at 10 % and 20 %, top-5 within 0.004 of the best).

## Thresholds (|pred| at firing rate 10 / 19 / 28 % of events, full-round1 fit)
B 0.143 / 0.122 / 0.105 (current mm_ml_b 0.105); D1 0.223 / 0.187 / 0.158; D4 0.237 / 0.199 / 0.170.

## Sweep (ledger tag r1-ml-bigmove; reference mm_ml_b: round1 503,148, sample_v2 258,042)

| dataset | set | pnl | train_pnl | hold_pnl | chunk_wins | chunk_min | fills | breach | inv_q997 | runtime_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round1 | D1 28 % (0.158) | 495,748 | 367,516 | 128,232 | 8 | 41,308 | 48,816 | 0 | 100 | 132 |
| round1 | D1 19 % (0.187) | 490,487 | 365,078 | 125,409 | 8 | 42,612 | 46,893 | 0 | 100 | 130 |
| round1 | D1 10 % (0.223) | 475,682 | 352,588 | 123,094 | 8 | 41,377 | 44,678 | 0 | 100 | 131 |
| sample_v2 | D1 28 % | 238,850 | 180,556 | 58,294 | 8 | 21,186 | 12,221 | 0 | 100 | 77 |
| sample_v2 | D1 19 % | 248,406 | 189,028 | 59,378 | 8 | 22,537 | 11,773 | 0 | 100 | 78 |
| sample_v2 | D1 10 % | 254,164 | 193,967 | 60,197 | 8 | 23,171 | 11,200 | 0 | 100 | 82 |
| round1 | D4 28 % (0.170) | 496,600 | 368,717 | 127,883 | 8 | 41,189 | 49,228 | 0 | 100 | 128 |
| round1 | D4 19 % (0.199) | 492,291 | 366,676 | 125,615 | 8 | 42,457 | 47,020 | 0 | 100 | 128 |
| round1 | D4 10 % (0.237) | 475,730 | 352,366 | 123,364 | 8 | 40,948 | 44,753 | 0 | 100 | 123 |
| sample_v2 | D4 28 % | 237,812 | 179,380 | 58,432 | 8 | 21,266 | 12,281 | 0 | 100 | 77 |
| sample_v2 | D4 19 % | 246,549 | 186,420 | 60,129 | 8 | 21,920 | 11,815 | 0 | 100 | 79 |
| sample_v2 | D4 10 % | 252,788 | 192,173 | 60,615 | 8 | 23,112 | 11,174 | 0 | 100 | 78 |

## research.compare: mm_ml_b (A) vs exp_ml_bigmove D4 thr 0.17 (best round1 row)
```
round1 (primary)   pnl A=503,148 B=496,600 diff=-6,548   hold A=129,688 B=127,883 diff=-1,805
  chunks B-A [-2351.0, -1024.1, -450.9, -1852.1, 692.1, 1306.0, -72.5, -990.8] -> B wins 2/8   fills 48770 / 49228
sample_v2          pnl A=258,042 B=237,812 diff=-20,230   hold A=62,614 B=58,432 diff=-4,182
  chunks B-A [-2188.1, -1898.0, -1038.9, -2202.4, -2486.3, -1619.2, -2210.4, -2404.7] -> B wins 0/8   fills 12292 / 12281
VERDICT: REJECT (round1 wins only 2/8; holdout not better)
```
Parity (defaults) 503,148 / 258,042 exact; `research.check --no-run strategies/exp_ml_bigmove.py` PASS.

## Interpretation
1. Big-move targets rank slightly better but trivially: top-5/10/20 % precision D1 0.418/0.367/0.287, D4 0.422/0.368/0.286 versus B 0.415/0.362/0.278 and base 0.411/0.360/0.283 (<= +0.009; base rate 0.135). D2 (300 ms) is worse (0.403 top-5) despite the highest own IC.
2. Linear 23-feature space again limits it: all scores correlate ~0.36-0.37 with the plain 100 ms move and the mean signed move in the top 5 % is 0.34-0.36 tick for every target.
3. The tiny precision gain does not turn into pnl: at matched rates D1/D4 lose 2-12k on round1 and 0-20k on sample_v2 versus B/mm_ml_b; D4 at 28 % loses 0/8 sample_v2 chunks. Hit-conditioned B (which also zeroes no-hit events) ranks fills better than events-based indicators.
4. H21 rejected; no change to mm_ml_b. Remaining headroom needs nonlinear/queue-state information (selectivity for large moves via a better model, not a different linear target).
