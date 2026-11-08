# radys -> Honza: what R3 taught us and what to try next (2026-11-08 05:05)

Short version: the predictor family is saturated around corr 0.50; the next real jump is a **learned combiner** of the
signals we already have and **regime-scaled thresholds**, not another model or more grid tuning.

## 1. Facts measured on s12 / s13 (reuse them, do not re-measure)

- Server / local: 1.63 (s10), 1.61 (s11), 1.60 (s12), 1.54 (s13). Use about x1.55 for GRU-era strategies.
- Fill probe of s12 (`research/diag/fill_probe.py`): taker IOC fills = **87 % of PnL** (+1.05 tick/lot at 30 s);
  |inv| >= 95 for **71 % of the time**. The strategy is a cap-bound position flipper; quote-side tweaks are worth 1-4 %.
- Realized move after a signal, by |pred| (s12 MLP, round3 holdout, spread 1):
  0.4-0.5 -> 200 ms +0.44 / 3 s +0.74 / 15 s +0.38;  0.7-0.9 -> +0.77 / +1.66 / +0.74;  > 1.2 -> +1.72 / +5.47 / +1.39.
  The edge peaks at 1-5 s and fades by 15 s. Despite that, lower taker thresholds lose in the sim (latency + flip cost).
- **Transfer is decaying (05:15):** s13 -> s14 = +3.0 % local (634,519 -> 653,327) but only +0.85 % server
  (976,148 -> 984,495); ratio 1.54 -> 1.51. Recent local gains are partly public-tape fit. Weight the holdout and
  mirror / round2 more than the full round3 number from here on (s14 holdout gain was only +1.2 %).
- Server PnL standard error is ~55k: differences under ~3 % on the board are noise.

## 2. Dead ends (do not repeat)

- MLP size (32-16 / 64-32 / 128-64), level-2 depth features, +round2 data, 150 / 300 ms targets: holdout corr 0.50 +- 0.003.
- 1 s / 2 s target MLPs for the taker: +24 % offline tail edge on r3, but sim r3 hold 130k vs 138k and round2 -3 %.
- Quote pull on top of the passive MLP (`pull_k`): +1.9 % on s12, **zero on s13** (mq already removes those fills).
- s11 quote-side mechanics (layer_size, ri_ms, tr_rest, park_size): plateau, +0.3 %.

## 3. Ideas ranked (for the next round or a long session)

1. **Learned combiner (stacking).** Today GRU, MLP-200, MLP-500, ridge, slow ridge and fair_sig are blended with fixed
   weights (pred_blend, mq_b, seq_mix, slw_thr). Dump all of them per event plus regime features (spread, EWMA vol,
   time since last trade, inventory sign) with `research/diag/mlp_dump.py` (extend the column list), fit a small linear
   or GBM model on the 1-3 s target, gated by regime. Expected: a few % IC, which is what moved s12 -> s13 (+17 %).
2. **Vol-adaptive thresholds.** Multiply fq_take_k / fq_k_sp2 / gate_thr by EWMA(|dmid|, 5 s) / its median, clipped.
   One EWMA per event, no runtime risk. (Copilot T4 if it got there.)
3. **Flip economics.** A flip costs >= 0.5 tick x 200 lots; partial flips (to 0 on a medium signal, to the far cap on a
   strong one) can raise edge per lot. Implement as target position = f(|pred|) with hysteresis, then grid f.
4. **Round kickoff with these tools.** On a new tape: `mirror_tape` -> `mlp_dump` -> `mlp_fit` (retrain MLP / GRU heads
   on the new tape in minutes) -> `fill_probe` to see where the money is, before any grid.

## 4. Tools added by radys (all in `research/diag/`)

| Tool | What |
|---|---|
| `mirror_tape.py` | B<->A, price 20000-p copy of a tape (overfit check) |
| `mlp_dump.py` | exact in-strategy features per event (22 + 4 extras) + mid path -> npz |
| `mlp_fit.py` | sklearn MLP on the dump, time split, corr + tail-edge report, bakes weights into a strategy copy |
| `fill_probe.py` | per-fill mark-out by prediction / inventory / type, time at the cap |
