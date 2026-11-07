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
| 2026-11-07 11:50 | Playground / `sample.csv` | `strategies/momentum_hold_v2.py` (same params, end-of-data flatten off and relative to the first event) | 51,225 | not uploaded yet |
| 2026-11-07 12:15 | Playground v2 / `sample_v2.csv` | `strategies/mm_v2.py` (Honza's `mm.py` core: quote 100, max_pos 100, skew 0) | 237,306 | pending |

Absolute path for upload (Tomáš's PC):
`C:\Users\tomas\OneDrive\Desktop\Projekty\Ostatni\AlgoTradeArena\submission\best_strategy.py`

## Before each new round

`best_strategy.py` was tuned on the previous round's data. On new data, run
`research/analyze_tape.py` first. End of data is no longer hard-coded: v2's `FLATTEN_AFTER_US` is
off by default and, when set, counts from the first event. v1's absolute `FLATTEN_AT_US = 3,590 s`
stopped trading after one hour and would never trade on a tape not starting at 0 (see progress log 11:50).
