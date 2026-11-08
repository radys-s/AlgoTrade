# Round 4 playbook (radys, written after R3, 2026-11-08)

Champion going in: `submission/best_strategy.py` = s14 `mm_r3_gru3` (R3 server 984,495, final). Ideas and dead ends:
`docs/ideas-radys-for-honza.md`. Read both before the round.

## First 20 min (insurance)

1. Human downloads the tape into `data/` (only humans can log in). Run the `round-kickoff` skill:
   `uv run python -m research.round data/round4_public.csv --publish`.
2. Run s14 as-is on the new tape (+ `research.check`). If positive, no breach, check PASS -> upload as insurance.
3. `uv run python -m research.diag.mirror_tape data/round4_public.csv data/round4_mirror.csv`.

## Retrain the learned parts on the new tape (first hour, biggest expected gain)

The R3 jump came from predictors fitted on the R3 tape (ridge refit, MLP, GRU), not from grids. On a new tape:
1. `research.refit_ml` for the ridge (pred model) on round4 + mirror, first 75 %.
2. `research.diag.mlp_dump` with the new champion -> `research.diag.mlp_fit` (22-32-16-1, h200) -> bake into a copy.
   Check holdout corr vs the old baked MLP on the new tape; if old >= new - 0.01, keep the old one (generator unchanged).
3. GRU head: Honza's `research/diag/seq_r3.py` (on herzic / his home node) with the same split.
4. `research.diag.fill_probe` on the result: taker share, time at cap, maker fills against the prediction.
   This tells which family of change can still pay before any grid.

## Judging (lessons from R3)

- Judge on new public + its mirror + round3 + round2; holdout and the non-public tapes count more than the full public pnl.
- Server / local fell from 1.63 to 1.51 over R3 and local gains under ~2 % did not show on the server
  (s14 + slw_thr 2.5: +0.3 % r3 local, +1.2 % round2, server 983k vs 984k). Spend uploads only on >= 3 % structural gains.
- Server PnL standard error ~55k. Board differences under ~3 % are noise.

## Compute and process

- Laptop = 12 threads; one GRU-era full run ~4-6 min alone, ~12 min under full load. A 100-set grid is not feasible
  in a round; keep grids to <= 12 sets, single-axis first, then combine.
- Each window works in its own checkout: radys uses the worktree `..\ata-radys` (branch `radys`); Copilot / others the
  main checkout. Do not share a working tree with a running sweep.
- **Kill orphans:** stopping an agent or a runner can leave `multiprocessing.spawn` workers running (R3: 5 orphans ate
  half the CPU for 40 min). Check with
  `powershell "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | select ProcessId,CommandLine"`.
- Copilot ran out of monthly credits at 05:15 on R3 night (resets 2026-12-01). Do not plan around it for R4.
- PowerShell strips JSON quotes in `--grid`; run sweeps from Git Bash.

## Ranked plan for R4 work (if the tape looks like R3)

1. Retrain predictors on the new tape (above). 2. Learned combiner of GRU / MLP / ridge / slow signals with regime
features (ideas doc, item 1). 3. Vol-adaptive thresholds. 4. Only then small grids around the new optimum.
