---
name: run-experiment
description: Protocol for testing one trading-strategy idea in this repo - copy the champion, add the mechanic behind a PARAMS flag, prove flag-off parity, sweep with research.run (halving for broad grids), judge with research.compare against the champion on chunks, holdout and earlier datasets, and log the result even if it failed. Use for any strategy change or parameter tuning.
---

# Run an experiment

One hypothesis per experiment, one file per experiment, every run in the ledger.

## 1. Write the hypothesis first
One sentence on the market behaviour exploited and why it should hold on *unseen* data
(the arena scores on private data). If you cannot say why, it is curve fitting.

## 2. Implement behind a flag
- `cp strategies/champion.py strategies/exp_<slug>.py`; you own only that file.
- Every tunable lives in the class dict `PARAMS` and is read via `self.PARAMS[...]`.
  The new mechanic is off by default so the file reproduces the champion.
- Keep per-event work tiny (up to ~9M callbacks, 15 min sandbox on 1 CPU): no numpy on every
  event unless needed, stop scanning the book after a few levels, update features incrementally
  (EMAs, running sums) instead of recomputing windows.
- Keep the position invariant: |inventory| + all live and in-flight same-side quantity
  <= max_position <= 100. A breach costs 10,000 per second and wipes out a round.
- Sandbox rules (checked by `research.check`): stdlib + numpy/pandas/scipy/sklearn +
  `simulator.models` + `simulator.strategy.BaseStrategy` only; no os/sys/io/pathlib/...,
  no open/eval/exec/getattr/type, no dunders except `__init__`, no `str.format`.

## 3. Parity and static check
```bash
uv run python -m research.run --strategy strategies/exp_<slug>.py --data data/<round>.csv --no-ledger
uv run python -m research.check strategies/exp_<slug>.py --no-run
```
The first pnl must equal the champion's exactly. If not, the flag leaks.

## 4. Sweep
```bash
# broad grid: successive halving (good set, not necessarily the best)
uv run python -m research.run --strategy strategies/exp_<slug>.py --data data/<round>.csv <earlier...> \
    --grid '{"flag": [true], "k": [0.5, 1, 2, 4]}' --halving 0.1,0.3 --workers 3 --tag <round>-<slug>
# then a narrow full grid around the survivors (no --halving)
```
Pick the middle of a plateau (neighbours also good), not the single best cell. Look at
`chunk_wins`, `chunk_min` and `hold_pnl`, not just `pnl`. Never tune on the holdout.

## 5. Judge
```bash
uv run python -m research.compare --a strategies/champion.py --b strategies/exp_<slug>.py \
    --b-set '<best overrides json>' --data data/<round>.csv <earlier...>
```
ACCEPT requires: no breach, >= 6/8 paired training chunks won, better holdout, and still
positive on every earlier dataset where the champion is positive. Then `research.check` with
the runtime run. Promotion to `strategies/champion.py` and submissions are a human decision.

## 6. Log, also when it failed
Append to `docs/progress.md` with the template at its end (hypothesis, change, data, result
with numbers before/after, conclusion, next step). Failed experiments are research too; the
ledger already holds the raw numbers, so cite the tag.
