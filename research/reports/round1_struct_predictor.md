# Structural depletion predictor (closed form from the generator rules) vs the learned 23-feature ridge

Code: `research/features_struct.py` (`uv run python -X utf8 -m research.features_struct eval data/features/round1.npz data/features/sample_v2.npz`). Raw log: `logs/struct/eval.txt`.
Setup: every 2nd event (2.2 M rows round1, 1.0 M sample_v2), 8 chunks, expanding walk-forward (7 test folds), ridge alpha 10, HGB as in research.signals. Constants are the round1 ones for both tapes (no refit for v2).

## Formula (per side, q = touch lots, n = touch orders, h in ms)
* P_big = 1 - Phi((ln q - 4.106)/1.395)
* P_trade(h) = (1 - exp(-4.0 h/1000)) * P_big
* P_cxl(h) = F_R(h)^n, F_R = stationary residual-life CDF of the shifted lognormal lifetime (shift -30, scale 353, sigma 0.85): F_R(100) = 0.2000, F_R(1000) = 0.8621
* P_dep = 1 - (1-P_trade)(1-P_cxl);  E_dm(100) = 0.5 (P_dep_ask - P_dep_bid);  E_dm(1000) = 0.5 * 1.7 * (P_dep_ask - P_dep_bid)
* Extra features: pbig_b/a/diff, pdep_b/a/diff and log-odds difference at both horizons (13 columns total).

## Results (IC = Pearson vs future mid change in ticks, mean over 7 folds; positive folds 7/7 everywhere)
| tape / h | E[dm] alone (no fit) | ridge 23 | ridge 23 + E[dm] | ridge 23 + 13 struct | HGB 23 | HGB 23 + 13 struct |
|---|---|---|---|---|---|---|
| round1 100 ms | +0.351 | +0.380 | +0.412 | +0.418 | +0.4515 | +0.4523 |
| round1 1000 ms | +0.308 | +0.397 | +0.3975 | +0.400 | +0.4083 | +0.4083 |
| sample_v2 100 ms | +0.309 | +0.363 | +0.380 | +0.384 | +0.3842 | +0.3854 |
| sample_v2 1000 ms | +0.285 | +0.408 | +0.408 | +0.4175 | +0.4199 | +0.4195 |

Top-1 % edge (ticks, ridge): round1 100 ms 0.448 -> 0.534 (+struct), 1000 ms 0.953 -> 1.061; top-10 % edge 0.2985 -> 0.316 and 0.769 -> 0.772. sample_v2: 100 ms e10 0.147 -> 0.160, 1000 ms e1 0.603 -> 0.680.
Cross-fit round1 -> sample_v2 (fit on round1): ridge23 IC 0.278 (100 ms) / 0.215 (1000 ms); ridge23+struct 0.268 / 0.188 (the ridge coefficients overfit round1 queue sizes); **E[dm] unfitted gets 0.310 / 0.286 on v2, better than the fitted ridge**.

## Calibration of P_dep(100 ms) vs realised "touch level gone within 100 ms" (both sides pooled)
round1: pred 0.002 / 0.007 / 0.014 / 0.032 / 0.071 / 0.142 / 0.245 / 0.397 -> realised 0.004 / 0.011 / 0.020 / 0.041 / 0.081 / 0.139 / 0.242 / 0.325. Calibrated from 10 % up (slightly high at the 0.3-0.5 bin), under-predicts by ~0.5-1 pt below 5 % (partial-fill/cancel-coupling not in the closed form; thin-queue cancels hazard 0.27 vs 0.21). A monotone transform would only change the tails.
sample_v2 (round1 constants): over-predicts 2-10x in the low bins (0.0033 -> 0.0003, 0.14 -> 0.081) because v2 IOCs are smaller (mu 3.71) and limit sizes uniform 20-40; refit MU there and it would calibrate. The ranking IC is unaffected.

## Verdict
1. Yes, it adds information the ridge lacks at 100 ms: ridge23 -> ridge23+E[dm] raises IC by +0.032 (round1) and +0.017 (v2), 7/7 folds, e1 edge +0.045 ticks; HGB on the 23 features already captures it (+0.001 only), so the gain is a nonlinearity (1-Phi of ln q, F_R^n) the linear ridge cannot form, not new raw information.
2. At 1000 ms there is no gain from E[dm] (+0.0002 round1; +0.009 v2 with all 13 columns); the structural idea is a 100 ms signal.
3. E[dm] alone (no fitting, 0 parameters) gets 0.351 vs the fitted 23-feature ridge 0.380 at 100 ms; it is 92 % of the ridge IC and transfers across tapes better than the ridge (v2: 0.31 vs 0.28 cross-fit).
4. Threshold 0.02 IC is met on round1 100 ms only (+0.032/+0.038 with all 13); cheapest port: add ONE feature, `edm_100`, to the ridge.
5. Port to `strategies/mm_ml_b.py`: per event, for b and a: `big = 0.5*erfc((ln(max(q,1)) - 4.106)/(1.395*sqrt2))`; `pt = 0.18127*big` (1-exp(-0.4)); `pc = exp(n*ln 0.2000)` = `0.2**max(n,1)`; `pdep = 1-(1-pt)*(1-pc)`; `edm_100 = 0.5*(pdep_a - pdep_b)`. Cost: 2 erfc, 2 log, 2 pow, ~25 flops; no state, no history, no deque. For 1000 ms add `pdep` with `(1-exp(-4))=0.98168` and `0.8621**n`, and multiply by 0.85.
6. Caveat: the constants are generator parameters of round1; if a later round's generator differs (as v2 did), refit MU / the lifetime shape or keep it as a ranking feature inside a refit ridge (the ridge then absorbs the scale).
