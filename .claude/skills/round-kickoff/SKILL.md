---
name: round-kickoff
description: First 20 minutes of an Algo Trade Arena round - new public backtest CSV arrives in data/, profile it, run the champion and baselines on it and on every earlier dataset, recommend the safety submission, then start research waves. Use when a new round's data file appears or the team says a round started.
---

# Round kickoff (Algo Trade Arena)

Goal: within ~20 minutes of the round start, know what this market looks like, have a safe
submission recommendation, and have the first research wave running. Read `docs/workflow.md`
for the full round plan and the acceptance protocol.

## Steps

1. **Data in place.** The round file is downloaded by a human from arena.qminers.com into
   `data/`. Unzip: `uv run python -m gzip -d data/<round>.csv.gz` (the simulator reads only
   plain CSV). List earlier datasets in `data/` (playground, previous rounds): they are the
   robustness set for every comparison this round.
2. **3-second first look** (Tomáš's tool): `uv run python -X utf8 research/analyze_tape.py data/<round>.csv`
   tells momentum vs reversion, spread regime and whether imbalance carries signal.
   **Features + data card** (one simulator pass, a few minutes on ~9M rows):
   `uv run python -m research.features data/<round>.csv` then
   `uv run python -m research.profile data/<round>.csv` -> `research/reports/<round>.md`.
   Read the report; note spread regime (1 tick vs wider), event rate, touch depth, mid-change
   autocorrelation (momentum vs mean reversion) and which signals have stable IC across chunks.
3. **Baselines on everything** (in parallel with step 2, `--workers` so total stays <= 12):
   `uv run python -m research.run --strategy submission/best_strategy.py --data data/<round>.csv <earlier...> --tag <round>-baseline`
   and the same for `strategies/mm.py` defaults. Check breach_penalty is 0 and the runtime
   (scale to the full file: sandbox is 15 min on 1 CPU; keep under ~7 min locally).
4. **Safety submission.** If the champion is positive on the new public data with no breach
   and passes `uv run python -m research.check submission/best_strategy.py --data data/<round>.csv`,
   tell the humans to submit it now as insurance (only humans can log in to the arena). If it
   is negative, recommend the best baseline that is positive, or the champion with smaller
   size. Say explicitly which file and that they must check which submission is *final*.
5. **Start research.** Launch a wave with the workflow script `research/workflows/research_wave.js`
   (Workflow tool, `scriptPath`), args `{data, prior, champion, wave: 1, recon: false}`, or
   run experiments by hand with the `run-experiment` skill.
6. **Log it.** Append the kickoff (data card highlights, baseline table, what was submitted) to
   `docs/progress.md` using its template, and update `STATUS.md` (round, final submission).

## Do not

- Tune anything on the new data before the baseline is recorded in the ledger.
- Submit without `research.check` passing; the arena allows only 10 submissions per round.
- Delete or rewrite ledger files in `research/results/` or old entries in `docs/progress.md`.
