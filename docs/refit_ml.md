# Re-fitting the ML champion on a new round (research/refit_ml.py)

The ridge forecast scale (and so the best `ml_thr`) moves with the market: sample_v2 forecast std is
0.064 ticks vs 0.11 on round1. Re-fit at the start of each round, minutes not hours.

What it does: builds `data/features/<round>.npz` if missing/stale (research.features), fits the same
ridge as `research.signals --export-linear` (23 OnlineFeatures-order features, winsorise 0.1 %/99.9 %,
standardise, alpha 10, max-rows 3M, 6 significant digits) on the first `--train-frac` of the tape's time
(label-overlap purge; 1.0 = all rows), then copies the template with only the `LINEAR_MODEL` dict
literal (brace-matched in the text) and the `'ml_thr'` value in PARAMS replaced. `ml_thr` = the
`--thr-quantile` (0.81) quantile of |pred| over the tape, i.e. the same ~19 % of events the champion's
0.15 hits on round1. `ml_stub` is untouched. Prints rows, IC (fit / holdout), forecast std, |pred|
quantiles, ml_thr, and runs `research.check --no-run`. About 10 s on round1.

```
# file to upload (fit on everything)
uv run python -X utf8 -m research.refit_ml data/round2.csv --template strategies/mm_ml.py --out strategies/mm_ml_round2.py
# leak-free evaluation variant (fit on the first 75 % of time, IC reported on the rest)
uv run python -X utf8 -m research.refit_ml data/round2.csv --out strategies/mm_ml_round2_tf75.py --train-frac 0.75
# follow-up grid around the chosen value T (printed), then compare with the champion
uv run python -X utf8 -m research.run --strategy strategies/mm_ml_round2.py --data data/round2.csv data/round1.csv \
  --grid '{"ml_thr": [0.7T, 0.85T, T, 1.2T, 1.5T], "ml_stub": [3, 5]}'
uv run python -X utf8 -m research.compare ...   # vs strategies/mm_ml.py on the same data (see run-experiment skill)
```

Validation: refitting round1 reproduces `strategies/mm_ml.py` byte-for-byte (ml_thr 0.15 included).
Fragility: the template must keep `LINEAR_MODEL ... = {` as one brace-balanced literal (no `}` in
comments inside it) and a numeric `'ml_thr': <number>,` in PARAMS; otherwise the tool exits with an error.
