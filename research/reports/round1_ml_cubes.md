# mm_ml with cubed-imbalance forecast models (exp_ml_cubes)

Files: `strategies/exp_ml_cubes.py` (PARAMS `ml_model` base|cubes|cubes_avg; models LINEAR_MODEL_CUBES / _CUBES_AVG fitted on all round1 rows),
scratch `logs/ml_cubes/` (fit.py, model_*_{train,all}.json, sweep.log, compare.log, exp_honest_*.py). Ledger tag r1-ml-cubes.

## 1. Forecast quality (h=100 ms, round1, ridge alpha 10, same winsorising/standardising as champion; train = first 75 % with 1 s purge)
Edge = half the (top-x % mean label minus bottom-x % mean label), ticks.
| model | IC train | IC holdout | e10 train/hold | e1 train/hold | pred std (hold) | firing at 0.15 (hold) | |pred| q81 (hold) |
|---|---|---|---|---|---|---|---|
| base | .3833 | .3752 | .231/.224 | .424/.384 | .111 | 19.0 % | .150 |
| cubes | .4137 | .4064 | .238/.229 | .502/.476 | .119 | 20.4 % | .157 |
| cubes_avg | .4141 | .4069 | .238/.229 | .495/.473 | .119 | 20.2 % | .157 |
Matched firing rate (19 %) needs t_match ~ 0.16 (q81 .157-.161). Sweep thresholds 0.13 / 0.16 / 0.19, ml_stub 5.

## 2. Parity / check / runtime
- ml_model base: round1 pnl 490,460 (exactly), hold 126,291. check --no-run PASS.
- cubes, whole round1 (4.41M rows) : PASS, 24.1 us/event overhead, projected 247 s (4.1 min) per 8.8M rows (under load, 2 procs). Crash-free.

## 3. Sweep (full-sample weights, includes in-sample fit)
| dataset | ml_model | thr | pnl | train_pnl | hold_pnl | chunk_wins | chunk_min | fills | breach | inv_q997 | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| round1 | cubes | .13 | 488,036 | 363,291 | 124,745 | 8 | 42,388 | 50,523 | 0 | 100 | 124 |
| round1 | cubes_avg | .13 | 487,292 | 362,465 | 124,827 | 8 | 42,185 | 50,469 | 0 | 100 | 124 |
| round1 | cubes | .16 | 486,164 | 361,745 | 124,419 | 8 | 41,666 | 48,263 | 0 | 100 | 122 |
| round1 | cubes_avg | .16 | 485,667 | 361,018 | 124,649 | 8 | 41,956 | 48,327 | 0 | 100 | 123 |
| round1 | cubes | .19 | 478,683 | 354,952 | 123,731 | 8 | 41,046 | 46,159 | 0 | 100 | 121 |
| round1 | cubes_avg | .19 | 477,007 | 353,551 | 123,456 | 8 | 40,574 | 46,255 | 0 | 100 | 123 |
| sample_v2 | cubes | .13 | 253,364 | 191,727 | 61,637 | 8 | 22,690 | 12,761 | 0 | 100 | 75 |
| sample_v2 | cubes_avg | .13 | 253,940 | 192,347 | 61,593 | 8 | 22,839 | 12,726 | 0 | 100 | 79 |
| sample_v2 | cubes | .16 | 257,761 | 196,487 | 61,274 | 8 | 23,912 | 12,343 | 0 | 100 | 77 |
| sample_v2 | cubes_avg | .16 | 257,719 | 196,337 | 61,382 | 8 | 23,705 | 12,230 | 0 | 100 | 77 |
| sample_v2 | cubes | .19 | 258,324 | 197,173 | 61,151 | 8 | 23,962 | 11,959 | 0 | 100 | 78 |
| sample_v2 | cubes_avg | .19 | 257,827 | 196,736 | 61,091 | 8 | 23,810 | 11,788 | 0 | 100 | 78 |
Champion for reference: round1 490,460 (hold 126,291), sample_v2 254,061 (hold 60,928). Best two-dataset sum: cubes .16 (743,925).

## 4. Compare mm_ml vs cubes, thr .16, stub 5
- round1: pnl 490,460 -> 486,164 (-4,296); hold 126,291 -> 124,419 (-1,872); chunk diffs B-A [-1428, -207, -106, -376, -498, +789, +89, -687] -> B wins 2/8; fills 46,527 -> 48,263; inv_q997 100 both.
- sample_v2: 254,061 -> 257,761 (+3,700); hold 60,928 -> 61,274 (+346); chunk diffs [-2, +181, +179, +747, +61, +540, +348, +1300] -> B wins 7/8; fills 11,723 -> 12,343.
- VERDICT: REJECT (round1 wins only 2/8 chunks, holdout not better).

## 5. Honest holdout (train-only weights, cubes thr .16, round1)
pnl 486,057, train_pnl 361,561, **hold 124,496** vs mm_ml leak-free holdout 126,128 (-1,632). Same story as the full-sample fit (hold 124,419), so no leakage artefact: cubes lose ~1.6k on the round1 holdout.

## 6. Interpretation
- IC +0.031 and e1 +0.09 to +0.09 ticks (+24 % holdout) hold out cleanly, but they do not turn into round1 pnl: every cubes cell is 2-12k below the champion, falling with a higher threshold; the shrink fires slightly more often at equal thr (20.4 vs 19 %) and extra fills come with adverse selection.
- On sample_v2 cubes give +3.7k (7/8 chunks), in line with the IC gain being robust cross-market; on round1 the shrink already captures the confident tail, so a sharper tail adds nothing, same lesson as the MLP.
- avgsz_imb adds nothing (IC +.0005; pnl equal or slightly worse); threshold choice around t_match is flat on sample_v2 and monotone-worse above it on round1.
- Cost is fine (24 us/event, 4.1 min projected), so the model could be kept as an option if the final tape looks like sample_v2, but on round1 it is rejected. Possible next try: use cubes only to set the side-selective stub size (graded shrink) rather than the same threshold.
