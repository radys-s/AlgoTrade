# submission/ — the ONE file we upload

**`submission/best_strategy.py` is the only file that gets uploaded to the platform
(arena.qminers.com → Dashboard → Submit strategy).** Nothing else in the repo is ever submitted.

## Rules

1. This folder always holds **exactly one** strategy: `best_strategy.py`, our current best.
   The file name never changes; only its content is replaced.
2. It gets replaced only when a new strategy is **better on the current round's public data**
   (higher local pnl, no penalty) **and** robust (good in all time quarters from `research/grid.py`,
   not one lucky segment).
3. Develop in `strategies/<name>.py` (keep every version there, with flags/parameters, never delete).
   When one wins, **copy** it over `submission/best_strategy.py`. Don't edit this file directly.
4. Before replacing:
   ```bash
   uv run python -X utf8 research/check_submission.py strategies/<name>.py   # must print OK
   uv run exchange-simulator --strategy strategies/<name>.py --backtest data/<round>.csv --log-dir logs/<run> --no-logs
   ```
5. Commit the replacement with the source file and local pnl in the message, update the table below,
   then merge into `main`. Local pnl on public data = dashboard pnl (verified 2026-11-07), so the
   dashboard only confirms the code runs in the sandbox.

## Current best

| Updated | Round / data | Source | Local pnl | Server pnl |
|---|---|---|---|---|
| 2026-11-07 11:40 | Playground / `sample.csv` | `strategies/momentum_hold.py` (QTY 50, HOLD 5 s, SLIP 3) | 50,965 | 50,965 (final) |

Absolute path for upload (Tomáš's PC):
`C:\Users\tomas\OneDrive\Desktop\Projekty\Ostatni\AlgoTradeArena\submission\best_strategy.py`

## Before each new round

`best_strategy.py` was tuned on the previous round's data. On new data, run
`research/analyze_tape.py` first. Also check `FLATTEN_AT_US` (end-of-data time, hard-coded to
3,590 s for the 1 h sample) before uploading.
