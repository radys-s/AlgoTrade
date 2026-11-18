---
name: round-kickoff
description: First 20 minutes of an Algo Trade Arena round - new public backtest CSV arrives, run the one-command kickoff (research.round) that unzips, shares the data with the home PC, runs baselines, the sandbox check, the data card, signals, taker scan and market-making edge, then read the kickoff report, recommend the insurance submission and start research waves. Use when a new round's data file appears or the team says a round started.
---

# Round kickoff (Algo Trade Arena)

Goal: within ~20 minutes of the round start, know what this market looks like, have a safe
submission recommendation, and have the first research wave running. Read `docs/workflow.md`
for the full round plan and the acceptance protocol.

## Steps

1. **Data in place.** A human downloads the round file from arena.qminers.com (only humans can
   log in). Move it into `data/` if it landed elsewhere (`data/` is git-ignored; never commit data).
2. **One command.**
   - Laptop (has the arena file): `uv run python -m research.round data/<file>.csv[.gz] --publish`
     (also uploads it to the private release `data-<name>` for the home PC).
   - Home PC: `uv run python -m research.round --fetch <name>`.
   Earlier datasets default to every other `data/*.csv` (`--prior` to choose). It runs two lanes in
   parallel: baselines (champion `submission/best_strategy.py`, `strategies/mm.py`,
   `strategies/trend_follower.py`) + `research.check` of the champion; and analyze_tape, features,
   data card, signals (100 ms, 1 s), taker scan, market-making edge. Logs: `logs/round/<name>/`.
3. **Read `research/reports/<name>_kickoff.md`.** It states the insurance verdict (champion positive
   on the new public data, no breach, check PASS, projected runtime under 7 min). Relay it to the
   humans with the exact file to upload and a reminder to verify which submission is *final*.
4. **Decide the regime** from the quick look and the data card: momentum or mean reversion,
   spread regime, depth, which signals are stable across chunks, whether makers at the touch make
   money (mm_edge). That picks the strategy family for wave 1.
5. **Start research.** Launch a wave (Workflow tool, `scriptPath: research/workflows/research_wave.js`)
   with the args printed in the kickoff report, or run experiments by hand with `run-experiment`.
   Confirm the best taker-scan configs in the real simulator (commands in the taker-scan report).
6. **Log it.** Append the kickoff (data card highlights, baseline table, what was submitted) to
   `docs/progress.md` using its template; humans update `STATUS.md`.

## Do not

- Tune anything on the new data before the baseline is recorded in the ledger.
- Submit without `research.check` passing; the arena allows only 10 submissions per round.
- Delete or rewrite ledger files in `research/results/` or old entries in `docs/progress.md`.
