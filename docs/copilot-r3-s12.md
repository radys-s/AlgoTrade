# Copilot brief: R3 push from s12 to >= 1M server (written 2026-11-08 by radys)

## Goal and numbers

- Champion: `strategies/honza_r3_s12.py` (mm_r3_mlp2, taker ridge + MLP). Server **864,770**; local round3_public 540,341,
  round3_mirror 560,237, round2 427,786; holdouts 138,223 / 137,230 / 108,183.
- Server ~= 1.6 x local. **Target: >= 1.0M server = >= 625k local on round3_public (+16 %)**. Nothing below s12 gets submitted.
- Round closes ~05:55 Prague. Report a candidate every ~45 min; do not sit on results.

## Working rules (do not break)

1. Work in the main checkout `C:\Users\tomas\OneDrive\Desktop\Projekty\Ostatni\AlgoTradeArena`. First finish or stop the
   pred-mode sweep, commit its ledger to `copilot/r3-s11-tune`, then `git fetch && git checkout -b copilot/r3-s12 origin/radys`.
   Claude (radys) works in a separate worktree `..\ata-radys` on the MLP retraining. Never touch that folder.
2. Never edit `strategies/honza_*.py`. Copy s12 to `strategies/cp_s12_<idea>.py` and change the copy.
3. `--workers 5` max (Claude uses the other 5 of 12 threads). Python: `~/.local/bin/uv.exe run python ...`.
   PowerShell strips JSON quotes. Run from Git Bash, or put the grid JSON in a file and pass `--grid "$(cat g.json)"`.
4. Every new mechanic sits behind a PARAMS flag whose default reproduces s12 **exactly** (flag-off parity). Prove it with
   one run on round3_public = 540,341 before any sweep.
5. Datasets: `data/round3_public.csv`, `data/round3_mirror.csv` (already created), `data/round2.csv`.
   Broad grids use `--halving` on round3_public only, then the finalists run on all 3.
6. A winner must: beat s12 on round3_public AND on its holdout, not lose more than 2 % on mirror or round2, 6/8 chunk wins,
   breach penalty < 3k, then pass `uv run python -m research.check <file>` (projected runtime < 7 min).
7. Commit ledgers + candidate files to `copilot/r3-s12` and push. No AI co-author trailer. Append results to the end of this file
   (section "Results") with numbers.

## Gotchas found already

- `pms_on=1` rewrites `pred_mode` to 1 inside `__init__`. To test `pred_mode` 0/2 set `pms_on 0, pred_on 1` explicitly
  and keep the same model (check `pred_model` / rfit in PARAMS of s12).
- The MLP only runs when `|ridge| >= mlp_gate`. Any change that makes the passive side use `pred_sig` must check whether
  the gated MLP value or the plain ridge reaches it.

## Tasks (in this order)

### T1. MLP runtime knobs (no code, grid only)
`mlp_gate {0, 0.15, 0.3, 0.5}` x `mlp_mix {0.7, 0.85, 1.0}` x `pred_blend {0.6, 0.7, 0.8, 0.9}` x `pred_scale {0.8, 1.0, 1.2}`,
halving 0.1,0.3, tag `cp-s12-mlpknobs`. Then around the best: `fq_take_k` +-0.05/0.1.

### T2. Passive re-tune on top of the MLP (grid only)
The MLP changed the taker, so the passive optimum may have moved. Grid on the best of T1:
`fq_k_sp2 {1.5, 2.0, 2.5}` x `fq_k_sp3 {0.5, 1.0, 1.5, 2.0}` x `rt_ms {100, 130, 170}` x `gate_thr {9, 11, 13}` x `qs_k {0.45, 0.6, 0.75}`,
halving, tag `cp-s12-passive`.

### T3. Prediction-driven quote pull (code, the biggest expected gain)
Adverse-selection avoidance: new PARAMS `pull_k` (0 = off). On each update, if `pred_sig < -pull_k` do not rest / cancel the bid
at the touch, if `pred_sig > pull_k` the same for the ask. Re-quote as soon as the prediction falls back inside the band.
Variant `pull_mode 1`: step the quote one tick back instead of cancelling. Make sure the MLP output is what drives it (set
`mlp_gate 0` for this test or compute the MLP when needed). Grid `pull_k {0.4, 0.6, 0.8, 1.0, 1.3}` x `pull_mode {0, 1}`.
Tag `cp-s12-pull`.

### T4. pred_mode on s12
`pms_on 0, pred_on 1`, `pred_mode {0, 1, 2}` x `pred_scale {0.6, 1.0}` x `fq_scale` (if present) `{0.7, 1.0}`. Tag `cp-s12-predmode`.

### T5. Exit by prediction (code)
`exit_k` (0 = off): when long and `pred_sig < -exit_k` (short and `> exit_k`), IOC at the touch for `min(|inv|, exit_q)` lots.
Grid `exit_k {0.8, 1.2, 1.6}` x `exit_q {10, 30, 100}`. Tag `cp-s12-exit`. Check first whether the existing `fq_take`
already covers the closing direction; if yes, report and skip.

### T6. Volatility-adaptive thresholds (code)
`vol_adapt` (0 = off): EWMA of |mid change| over ~5 s divided by its running median, clipped [0.7, 1.5], multiplies
`fq_take_k`, `fq_k_sp2`, `gate_thr`. Grid `vol_adapt {0, 1}` x clip range. Tag `cp-s12-vol`.

### T7. Stack
Combine every individually positive change from T1-T6 into `strategies/cp_s12_stack.py`, run all 3 datasets,
`research.check`, report.

## Report format (append under Results)

`| task | file | params | r3 | r3 hold | mirror | round2 | chunks | breach | verdict |` plus one line on what you learned.

## radys findings (03:20, read before T3/T5)

- MLP retraining is saturated: 32-16 / 64-32 / 128-64, +level-2 depth features, horizons 150/300 ms, +round2 in training
  all give holdout corr 0.50 (gated 0.58), same as s12's MLP +-0.003. Do not spend runs on new MLP weights.
- Fill probe of s12 on round3_public (`research/diag/fill_probe.py`): taker IOC fills = 448k lots at +1.05 tick/lot
  (30 s mark-out) = 470k of the 540k; maker fills only 96k. |inv| >= 95 for 71 % of the time: the cap binds,
  the strategy is a position flipper. Taker edge by horizon: 1 s 1.08, 5 s 1.37, 30 s 1.05 (peaks ~5 s).
- T5 (IOC exit on a faded signal) is unlikely to pay: crossing costs >= 0.5 tick, the decay it saves is ~0.3. Run it last,
  only the `exit_k >= 0.6` cells.
- Maker fills when the prediction is against the fill (pred in fill direction < -0.3) lose: -3.2 tick/lot at 30 s on 16k lots
  (-52k). That is exactly what T3 (quote pull) removes, so T3 has priority over T2. Start T3 with pull_k 0.3-0.6.

## Results
