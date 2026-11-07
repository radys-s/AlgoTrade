---
name: signal-research
description: Find and validate short-horizon price signals and small ML models on an Algo Trade Arena tape with this repo's pipeline (research.features, research.profile, research.signals, research.train_mlp), then move a validated model into a strategy cheaply. Use for any question about predictive signals, IC, feature importance, ridge/GBM/MLP models or Colab/GPU training in this repo.
---

# Signal research pipeline

A forecast is only worth something if it is stable across time chunks, survives the 5 ms
latency, and its edge is large against the spread and cheap enough to compute ~9M times.

## Commands

```bash
uv run python -m research.features data/<round>.csv          # -> data/features/<round>.npz (one sim pass)
uv run python -m research.profile  data/<round>.csv          # data card + simple-signal IC per chunk
uv run python -m research.signals  data/features/<round>.npz --horizon-ms 1000   # walk-forward ridge + HGB
uv run python -m research.signals  data/<round>.csv --horizon-ms 100 --export-linear   # paste-ready LINEAR_MODEL
uv run python -m research.signals  data/features/<round>.npz --horizon-ms 1000 --export-dataset data/features/<round>_ml.npz
uv run --with torch python -m research.train_mlp data/features/<round>_ml.npz --hidden 32 --epochs 30   # or on Colab / home GPU
uv run python -m research.signals  data/<round>.csv --parity --no-hgb   # OnlineFeatures == offline features
```
Options: `--features a,b,c`, `--chunks`, `--train-chunks` (rolling window), `--no-hgb`, `--max-rows`.

## Alignment (already enforced by the code, keep it that way)
- npz row i = state right after tape row i; features use rows <= i only.
- Label = mid(t+5 ms+h) - mid(t+5 ms): the forecast starts where our order could first land.
- Walk-forward never shuffles; training rows whose label window reaches the test chunk are dropped.
- Rows are per event, so bursts weigh more and labels overlap: trust across-chunk t-stats, not
  per-row significance (statsmodels HAC errors if you need a regression t-stat, offline only).

## Reading results
- Report IC per fold/chunk and the sign consistency (e.g. 7/7), never just the mean.
- Compare the predicted move (edge10 / edge1 columns: mean label in the top 10 % / 1 % of
  forecasts) with the half-spread: taking with IOC needs edge > half-spread; for quoting, a
  smaller edge still helps as a skew or as a reason to pull one side.
- HGB adding nothing over ridge means: ship the linear model.
- Labels are mid changes, not PnL: queue position and fill probability decide the real value,
  so every signal ends in `research.run` / `research.compare` on the strategy, not in IC.

## Into a strategy
- Paste `OnlineFeatures` (pure Python, O(1) per event) and the exported `LINEAR_MODEL` dict, or
  `mlp_compile` / `mlp_predict` from `train_mlp.py` output, into the strategy file. No file
  loading, no torch, no pickle in the submission (sandbox rules, 1 MiB limit).
- Cost: OnlineFeatures is ~10-25 us per event; at ~9M events that is minutes of the 15 min
  sandbox budget. Keep only features that carry weight; evaluate a model every N events or on
  book changes at the touch, not on every row. Check with `research.check --rows 0`.
