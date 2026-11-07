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

## Target B (strategies/mm_ml_b.py): `--target B`

`--target base` (default) is everything above, byte-identical to before. `--target B` builds the loss-aligned
label exactly as `research.fit_target` does (`build_targets`: mid move over 100 ms after the FIRST taker hit on
the side within 100 ms after the 5 ms latency, x [hit], 0 when no hit; rows with an unknown 1 s outcome dropped),
fits the same ridge (23 features, winsorise, standardise, alpha 10, 6 digits), and replaces ONLY the `'B': {...}`
entry inside the template's `LINEAR_MODELS` (base `LINEAR_MODEL` and the A and C entries stay). `ml_thr` is set by firing
rate: the share of events with |pred| > the template's `ml_thr` under the template's own B model on `--ref-data`
(default data/round1.csv; 27.95 % for mm_ml_b) is printed, and the new `ml_thr` is the matching |pred| quantile on the
new tape. `--thr-share 0.28` overrides the share. `ml_stub` and `ml_target` are untouched; the template's line endings are kept.

```
# R2 kickoff: B refit baked into a copy of mm_ml_b (fit on the whole tape)
uv run python -X utf8 -m research.refit_ml data/round2.csv --template strategies/mm_ml_b.py --target B --out strategies/mm_ml_b_round2.py
# plain model of the champion, as before
uv run python -X utf8 -m research.refit_ml data/round2.csv --template strategies/mm_ml.py --out strategies/mm_ml_round2.py
# follow-up grid for B around the printed threshold T (plateau on round1 was 0.9T-1.1T)
uv run python -X utf8 -m research.run --strategy strategies/mm_ml_b_round2.py --data data/round2.csv data/round1.csv   --grid '{"ml_thr": [0.7T, 0.85T, T, 1.2T, 1.5T], "ml_stub": [3, 5]}'
```

Validation (round1 and sample_v2, local): refitting round1 with template mm_ml_b reproduces the file byte-for-byte
(pnl 503,148; share 27.95 %, ml_thr 0.105); `--target base` on round1 still reproduces mm_ml.py byte-for-byte.
`--target B` on sample_v2 -> strategies/mm_ml_b_v2fit.py (ml_thr 0.0372, forecast std 0.036): sample_v2 252,091, round1 207,829
(mm_ml_b: 258,042 / 503,148). Taking the v2 scale blindly is worse than the round1 model on both tapes, so on R2 check
the fit on its own tape AND on round1 before choosing it over mm_ml_b as is.
Fragility: the template needs `LINEAR_MODELS = {... 'B': {...} ...}` brace-balanced with a literal `'B'` key and the
same feature order as research.features; B mode needs data/features/round1.npz (or `--ref-data`) for the share and ~1 GB RAM extra.
