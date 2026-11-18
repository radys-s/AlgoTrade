---
name: pre-submit
description: Checklist before a strategy file is uploaded to arena.qminers.com - sandbox static check, runtime projection, champion comparison, baked-in PARAMS, commit and merge to main, and the reminder about which submission is final. Use whenever the team is about to submit or asks if a strategy is ready.
---

# Pre-submit checklist

Only humans can log in and submit. Your job is to make the file safe and say exactly what to
upload. The arena allows 10 submissions per round; the *final* one defaults to the last
successful upload, so a careless experimental upload silently becomes the final.

1. **One file, self-contained.** Tuned values are written into `PARAMS` in the file itself
   (the runner's overrides do not exist on the arena). One `BaseStrategy` subclass, no
   constructor arguments, no helper modules, under 1 MiB.
2. **Static + smoke + runtime:**
   `uv run python -m research.check <file> --data data/<round>.csv --rows 0`
   Must PASS. Projected time for the full file must stay well under 15 min (aim < 7 min
   locally; the sandbox has 1 CPU and may be slower). Fix every ERROR; read every WARN.
3. **Better than what is live:** `research.compare` against the current `submission/best_strategy.py`
   on the round data and earlier datasets says ACCEPT, or the humans accept a written reason.
4. **No breach anywhere:** breach_penalty 0 on every dataset in the ledger for this file.
5. **Record it:** copy the winning `strategies/<name>.py` over `submission/best_strategy.py` (the
   only file ever uploaded; rules in `submission/README.md`) and update its "Current best" table.
   Commit (`git add` the strategy, ledger files and `docs/progress.md`), merge the personal branch
   into `main` (merge, never rebase or force-push), push. `main` must equal what is on the arena.
   Check end-of-data constants (e.g. `FLATTEN_AT_US` in `momentum_hold`) against the round length.
6. **Tell the human** the exact file path to upload, the expected public PnL (the server reruns
   the public data, so its number should match the local run), and to verify on the dashboard
   which submission is marked final.
7. **After upload:** update `STATUS.md` (round, final submission, local vs server PnL).
