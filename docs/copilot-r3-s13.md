# Copilot brief v2: R3 from s13 (976,148 server) to >= 1M (radys, 2026-11-08 04:40)

Supersedes `docs/copilot-r3-s12.md` (its findings still hold; read its "radys findings/results" sections, skip its tasks).

## Numbers

- Champion `strategies/honza_r3_s13.py` (mm_r3_gru: s12 + MLP in the passive quote signal (mq_on, mq_h 500) + 5 s slow
  direction pull (slw_on) + streaming GRU taker predictor (seq_on)). Server **976,148**. Local round3_public 634,519,
  round3_mirror 624,778, round2 437,941; holdouts 156,075 / 157,441 / 109,156. Server / local = 1.54.
- **1M server = about 650k local on round3_public = +2.5 %.** Every +1 % local is about +10k server. Small wins count now.
- Runtime: s13 runs 4 m 40 s on the server, `research.check` projects 6.7 min. Anything that adds per-event compute must
  keep the projection under ~8 min (sandbox limit 15 min). Prefer changes with no extra compute.
- Round closes ~05:55 Prague. Report a candidate as soon as one beats s13 on all 3 datasets; do not wait for the full list.

## Rules (same as v1)

Main checkout, new branch `copilot/r3-s13` from `origin/radys` (`git fetch` first). Never edit `strategies/honza_*.py`;
copy to `strategies/cp_s13_<idea>.py`. `--workers 5` (radys uses the rest). Run from Git Bash (PowerShell strips JSON quotes).
Flag-off parity first (= 634,519 on round3_public). Winner = beats s13 on round3_public AND its holdout, loses < 1 % on
mirror and round2, 6/8 chunks, `research.check` PASS. Commit + push ledgers and candidates, no AI co-author trailer,
append results to the end of this file.

## Tasks (in order)

### T1. Taker / quote knobs on s13 (grid, halving 0.1,0.3, tag `cp-s13-knobs`)
Read PARAMS of s13 first and use its current values as the centre. Grid: `pred_scale {0.9, 1.0, 1.1}` x
`fq_take_k {centre-0.05, centre, centre+0.05, centre+0.1}` x `mq_b {centre-0.15, centre, centre+0.15}` x
`slw_thr {centre-0.5, centre, centre+0.5}`. Finalists on all 3 datasets.

### T2. Passive re-tune on s13 (grid, halving, tag `cp-s13-passive`)
`fq_k_sp2 {1.5, 2.0, 2.5}` x `fq_k_sp3 {1.0, 1.5, 2.0}` x `rt_ms {100, 130, 170}` x `gate_thr {9, 11, 13}` x `qs_k {0.45, 0.6, 0.75}`.

### T3. Quote pull follow-up (only if radys reports pull > 0 helps on s13; see Results)
radys built `strategies/rd_s13_pull.py` (`pull_k`, 0 = s13 parity). If pull_k 0.35 or 0.5 wins, fine-grid
`pull_k {0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6}` and add `pull_mode 1` = step the quote one tick back instead of cancelling.

### T4. Volatility-adaptive thresholds (code, tag `cp-s13-vol`)
`vol_adapt` (0 = off): EWMA of |mid change| over ~5 s / its running median, clipped [0.7, 1.5], multiplies `fq_take_k`,
`fq_k_sp2`, `gate_thr`. Grid on/off x clip range {[0.8,1.25], [0.7,1.5]}. Cheap per event (one EWMA).

### T5. Stack
All individually positive changes into `strategies/cp_s13_stack.py`, all 3 datasets, `research.check`, report path + numbers.

## Report

`| task | file | params | r3 | r3 hold | mirror | round2 | chunks | breach | check | verdict |`

## Results
