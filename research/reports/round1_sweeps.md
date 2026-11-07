# Round 1: can a big taker sweep be anticipated from the public tape? (research/sweep_predict.py)

## Verdict

1. **No.** Big IOCs (>= 300 lots, 12,067; >= 1000 lots, 2,095) arrive as a memoryless Poisson process on each side: gap cv 0.99-1.02, size autocorrelation lag 1..10 within +-0.01, no same-side or opposite-side clustering (P(big within 100/500/2000 ms of another big, same side) = 0.0463/0.222/0.630 vs unconditional 0.0496/0.223/0.633, lift 0.93-1.00), no repeated exact sizes (P(same as previous) 0.0013 vs iid 0.0016), no round-number preference beyond uniform, ids strictly sequential with no bait/limit-then-IOC pattern (prev insert is a same-side limit 47.9 % vs 47.8 % opposite).
2. Precursor features (touch cancels, limit inserts behind/at the touch, small probing IOCs on the same side, executed volume on the side, L1 imbalance, touch depletion, time since last mid change, earlier bigs) at 50/100/500 ms: every flag has lift 0.9-1.1 (best single cell: hit volume in 500 ms top 1 %, lift 1.26, recall 1.3 %). Max single-feature AUC 0.506.
3. Walk-forward (7 test chunks) logistic score from the 5 best precursors: AUC 0.499 for >= 300 and 0.502 for >= 1000; all 20 features: 0.497. Top 1/2/5/10 % of rows flag 0.9/1.8/4.9/10 % of the big sweeps (equal to the flag rate: zero lift), precision 4.3-5.0 % vs 4.9 % base.
4. The existing ridge trigger (|pred| > 0.15 against the side) is on 9.6 % of the time and flags 9.7 % of big sweeps (also lift ~1.0): the ridge does not anticipate sweeps either, it predicts the drift that the ordinary flow carries. The sweep losses are therefore not avoidable by timing; only by size/queue position (a sweep exceeds the touch quantity in 39 % of cases, median size/touch 0.67).
5. Incremental information: sweep score vs ridge correlation -0.10 (5-feat) / +0.05 (all-feat); incremental OOS IC for the 100 ms mid move: 0.3802 -> 0.3812 (+score, 5/7 folds) -> 0.3844 (+score + raw side-difference precursors, 7/7 folds). +0.004 IC is a flow-information effect on the mid move, not sweep anticipation; not worth the code.
6. sample_v2 (synthetic) shows the same: Poisson arrivals, no clustering for >= 300, all lifts 0.9-1.1 (only >= 1000 same-side lift 1.6-1.75 on 361 events, i.e. noise-level counts; not seen in round1).
7. Recommendation: do NOT add a sweep trigger. No formula to ship; cost per flagged event is pure loss (a score flag at 2 % of the time costs the same fill rate as unflagged time and avoids no more sweeps than a random 2 % of time would).
8. Caveat on section 5: fill price is approximated by the row's last trade price (worst level of a multi-level sweep), so its sweep-row markouts (about 0 to +0.1/lot) are biased versus the -0.37/lot from real fills; ignore the sweep rows there. The only usable observation: non-sweep fills inside ridge-flagged windows earn +0.18/lot vs +0.32 unflagged, consistent with the ridge doing its job on drift.

Reproduce: `uv run python -X utf8 -m research.sweep_predict round1 --out logs/sweeps/round1.md` (4 min); `... sample_v2 --skip-model`. Label = a big IOC attacking the side in (t+5 ms, t+105 ms]; side-relative features pooled over both sides.

---

# Sweep anticipation: `round1` (4,411,336 rows, 12000 s)

## 1. IOC arrivals

IOC inserts: 96,088; size quantiles q50/q90/q95/q99/q99.9 = 61 / 360 / 603 / 1543 / 4630; >= 300: 12,067; >= 1000: 2,095

| set | side | n | median gap ms | mean gap ms | cv of gap (1 = Poisson) |
|---|---|---|---|---|---|
| all IOC | bid hit | 48020 | 173.9 | 249.9 | 0.99 |
| all IOC | ask hit | 48068 | 172.9 | 249.6 | 1.01 |
| >= 300 | bid hit | 5992 | 1414.2 | 2002.6 | 0.99 |
| >= 300 | ask hit | 6075 | 1329.8 | 1974.0 | 1.02 |
| >= 1000 | bid hit | 1056 | 7749.8 | 11347.0 | 1.00 |
| >= 1000 | ask hit | 1039 | 7970.3 | 11549.8 | 1.02 |

Autocorrelation of log(size), lag 1..10:
| lag | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| all IOC | +0.005 | +0.000 | +0.004 | -0.006 | -0.000 | -0.004 | -0.001 | +0.009 | +0.003 | +0.005 |
| same side | +0.003 | +0.002 | +0.004 | +0.000 | -0.003 | -0.002 | +0.000 | +0.002 | -0.004 | +0.002 |

P(big IOC within window after another big IOC) vs unconditional window probability:
| size | next on | window | P(cond) | P(uncond) | lift |
|---|---|---|---|---|---|
| >= 300 | same side | 100 ms | 0.0463 | 0.0496 | 0.93 |
| >= 300 | same side | 500 ms | 0.2218 | 0.2229 | 1.00 |
| >= 300 | same side | 2000 ms | 0.6301 | 0.6328 | 1.00 |
| >= 300 | opposite side | 100 ms | 0.0471 | 0.0496 | 0.95 |
| >= 300 | opposite side | 500 ms | 0.2169 | 0.2229 | 0.97 |
| >= 300 | opposite side | 2000 ms | 0.6329 | 0.6328 | 1.00 |
| >= 1000 | same side | 100 ms | 0.0086 | 0.0087 | 0.99 |
| >= 1000 | same side | 500 ms | 0.0377 | 0.0428 | 0.88 |
| >= 1000 | same side | 2000 ms | 0.1585 | 0.1606 | 0.99 |
| >= 1000 | opposite side | 100 ms | 0.0076 | 0.0087 | 0.88 |
| >= 1000 | opposite side | 500 ms | 0.0434 | 0.0428 | 1.02 |
| >= 1000 | opposite side | 2000 ms | 0.1599 | 0.1606 | 1.00 |

Same exact size as the previous same-side IOC (observed vs iid expectation sum p^2), and equal to any of the previous 10:
| sizes | n | P(repeat prev) | iid expect | P(in last 10) |
|---|---|---|---|---|
| >= 1 | 96088 | 0.0051 | 0.0054 | 0.0523 |
| >= 100 | 34915 | 0.0026 | 0.0029 | 0.0280 |
| >= 300 | 12067 | 0.0013 | 0.0016 | 0.0134 |

Round numbers among >= 300 IOCs: %50==0 0.0237 (iid 0.02), %100==0 0.0124 (0.01), %10==0 0.1039 (0.10), non-integer 0.0000
Most frequent big sizes: 314x53, 316x53, 332x53, 300x52, 313x52, 310x51, 308x51, 307x48

## 3. Order-id patterns

Insert ids strictly sequential (+1) for 1.0000 of consecutive inserts.
Gap (in inserts) between consecutive big IOCs: median 131, mean 187 (geometric/iid expectation mean 187); P(gap<=3) 0.0143 vs iid 0.0160
Baseline P(insert is limit) = 0.9574; baseline P(IOC) 0.0426
| immediately preceding insert of a big IOC | share |
|---|---|
| prev insert is IOC | 0.0425 |
| prev insert is limit | 0.9575 |
| prev limit same side as sweep (bait?) | 0.4792 |
| prev limit opposite side (will be hit) | 0.4783 |
| prev insert is big IOC | 0.0046 |

| previous tape row of a big IOC | share | base share |
|---|---|---|
| limit insert | 0.4887 | 0.4891 |
| IOC | 0.0224 | 0.0218 |
| cancel | 0.4889 | 0.4891 |

Big IOC size equals the size of one of the previous 20 tape rows: 0.0017 (small IOCs of 20-299 lots: 0.0585; sizes are rarer so a lower rate is expected)
Big IOC size / opposite touch qty before it: q25/q50/q75 = 0.31 / 0.67 / 1.94; share that exceed the touch: 0.389

## 2. Precursors of a big IOC in the next 100 ms (pooled sides)

Rows with a two-sided book: 4,410,659 per side. Base rate P(>= 300 IOC hits the side within (5, 105] ms) = 0.0492; for >= 1000: 0.0087. Lift = P(label | flag) / base; recall = share of positive rows flagged.

| feature | flag: x > | flag share | lift>=300 | recall>=300 | lift>=1000 | recall>=1000 |
|---|---|---|---|---|---|---|
| cancel_touch_vol_50 | 46 | 0.497 | 1.00 | 0.499 | 1.03 | 0.514 |
| cancel_touch_vol_50 | 423 | 0.100 | 1.00 | 0.100 | 1.08 | 0.107 |
| cancel_touch_vol_50 | 716 | 0.050 | 1.03 | 0.051 | 1.10 | 0.055 |
| cancel_touch_vol_50 | 1.93e+03 | 0.010 | 0.99 | 0.010 | 1.06 | 0.011 |
| ins_behind_vol_50 | 29 | 0.499 | 1.01 | 0.502 | 1.03 | 0.513 |
| ins_behind_vol_50 | 356 | 0.100 | 1.03 | 0.103 | 1.05 | 0.105 |
| ins_behind_vol_50 | 615 | 0.050 | 1.04 | 0.052 | 1.09 | 0.054 |
| ins_behind_vol_50 | 1.68e+03 | 0.010 | 1.02 | 0.010 | 1.04 | 0.010 |
| ins_touch_vol_50 | 59 | 0.497 | 1.00 | 0.498 | 0.98 | 0.489 |
| ins_touch_vol_50 | 459 | 0.100 | 1.01 | 0.101 | 1.04 | 0.103 |
| ins_touch_vol_50 | 759 | 0.050 | 1.02 | 0.051 | 1.12 | 0.056 |
| ins_touch_vol_50 | 2e+03 | 0.010 | 1.00 | 0.010 | 1.05 | 0.011 |
| probe_n_50 | 0 | 0.172 | 1.00 | 0.172 | 1.05 | 0.181 |
| probe_n_50 | 1 | 0.016 | 1.00 | 0.016 | 1.08 | 0.017 |
| probe_vol_50 | 0 | 0.172 | 1.00 | 0.172 | 1.05 | 0.181 |
| probe_vol_50 | 42 | 0.100 | 1.02 | 0.101 | 1.09 | 0.108 |
| probe_vol_50 | 105 | 0.050 | 1.03 | 0.052 | 1.12 | 0.056 |
| probe_vol_50 | 240 | 0.010 | 1.05 | 0.010 | 1.32 | 0.013 |
| hit_vol_50 | 0 | 0.193 | 1.00 | 0.193 | 1.05 | 0.202 |
| hit_vol_50 | 66 | 0.100 | 1.02 | 0.102 | 1.06 | 0.106 |
| hit_vol_50 | 177 | 0.050 | 1.03 | 0.051 | 1.12 | 0.056 |
| hit_vol_50 | 648 | 0.010 | 1.10 | 0.011 | 1.03 | 0.010 |
| opp_cancel_touch_50 (neg) | 46 | 0.499 | 1.01 | 0.505 | 1.05 | 0.523 |
| cancel_touch_vol_100 | 147 | 0.500 | 1.00 | 0.499 | 1.01 | 0.505 |
| cancel_touch_vol_100 | 809 | 0.100 | 1.01 | 0.101 | 1.09 | 0.109 |
| cancel_touch_vol_100 | 1.25e+03 | 0.050 | 1.02 | 0.051 | 1.08 | 0.054 |
| cancel_touch_vol_100 | 2.96e+03 | 0.010 | 1.08 | 0.011 | 1.22 | 0.012 |
| ins_behind_vol_100 | 104 | 0.498 | 1.01 | 0.502 | 1.02 | 0.509 |
| ins_behind_vol_100 | 697 | 0.100 | 1.03 | 0.103 | 1.07 | 0.107 |
| ins_behind_vol_100 | 1.1e+03 | 0.050 | 1.02 | 0.051 | 1.04 | 0.052 |
| ins_behind_vol_100 | 2.62e+03 | 0.010 | 0.94 | 0.009 | 1.13 | 0.011 |
| ins_touch_vol_100 (neg) | 180 | 0.499 | 1.00 | 0.499 | 0.99 | 0.497 |
| probe_n_100 | 0 | 0.307 | 1.00 | 0.308 | 1.04 | 0.319 |
| probe_n_100 | 1 | 0.054 | 0.97 | 0.053 | 1.07 | 0.058 |
| probe_n_100 | 2 | 0.006 | 1.13 | 0.007 | 1.12 | 0.007 |
| probe_vol_100 | 0 | 0.307 | 1.00 | 0.308 | 1.04 | 0.319 |
| probe_vol_100 | 104 | 0.100 | 1.02 | 0.102 | 1.08 | 0.108 |
| probe_vol_100 | 176 | 0.049 | 1.04 | 0.051 | 1.12 | 0.055 |
| probe_vol_100 | 284 | 0.010 | 1.06 | 0.011 | 1.25 | 0.012 |
| hit_vol_100 | 0 | 0.341 | 1.00 | 0.341 | 1.04 | 0.356 |
| hit_vol_100 | 175 | 0.100 | 1.03 | 0.102 | 1.11 | 0.110 |
| hit_vol_100 | 345 | 0.050 | 1.06 | 0.053 | 1.13 | 0.057 |
| hit_vol_100 | 960 | 0.010 | 1.14 | 0.011 | 1.24 | 0.012 |
| opp_cancel_touch_100 (neg) | 147 | 0.499 | 1.01 | 0.502 | 1.04 | 0.517 |
| opp_cancel_touch_100 (neg) | 4 | 0.095 | 1.02 | 0.097 | 0.97 | 0.092 |
| cancel_touch_vol_500 | 1.25e+03 | 0.500 | 1.00 | 0.500 | 1.01 | 0.504 |
| cancel_touch_vol_500 | 3.31e+03 | 0.100 | 1.08 | 0.108 | 1.09 | 0.109 |
| cancel_touch_vol_500 | 4.35e+03 | 0.050 | 1.13 | 0.056 | 1.30 | 0.065 |
| cancel_touch_vol_500 | 7.85e+03 | 0.010 | 0.99 | 0.010 | 1.37 | 0.014 |
| ins_behind_vol_500 | 968 | 0.500 | 1.01 | 0.506 | 1.02 | 0.509 |
| ins_behind_vol_500 | 2.99e+03 | 0.100 | 1.01 | 0.100 | 1.01 | 0.101 |
| ins_behind_vol_500 | 3.98e+03 | 0.050 | 0.95 | 0.048 | 1.00 | 0.050 |
| ins_behind_vol_500 | 7.23e+03 | 0.010 | 0.90 | 0.009 | 1.15 | 0.011 |
| ins_touch_vol_500 | 1.45e+03 | 0.500 | 1.00 | 0.502 | 0.97 | 0.486 |
| ins_touch_vol_500 | 3.59e+03 | 0.100 | 1.05 | 0.105 | 1.00 | 0.100 |
| ins_touch_vol_500 | 4.69e+03 | 0.050 | 1.04 | 0.052 | 0.99 | 0.050 |
| ins_touch_vol_500 | 8.34e+03 | 0.010 | 1.05 | 0.010 | 1.03 | 0.010 |
| probe_n_500 (neg) | 2 | 0.473 | 1.01 | 0.476 | 0.98 | 0.462 |
| probe_vol_500 | 90 | 0.498 | 1.01 | 0.504 | 1.04 | 0.521 |
| probe_vol_500 | 319 | 0.100 | 1.02 | 0.101 | 1.10 | 0.109 |
| probe_vol_500 | 401 | 0.050 | 1.01 | 0.050 | 1.15 | 0.057 |
| probe_vol_500 | 570 | 0.010 | 0.99 | 0.010 | 1.34 | 0.013 |
| hit_vol_500 | 143 | 0.499 | 1.00 | 0.500 | 1.03 | 0.512 |
| hit_vol_500 | 747 | 0.100 | 1.03 | 0.103 | 1.03 | 0.103 |
| hit_vol_500 | 1.08e+03 | 0.050 | 1.07 | 0.053 | 1.00 | 0.050 |
| hit_vol_500 | 1.91e+03 | 0.010 | 1.26 | 0.013 | 1.34 | 0.013 |
| opp_cancel_touch_500 (neg) | 1.25e+03 | 0.500 | 1.01 | 0.503 | 1.03 | 0.515 |
| opp_cancel_touch_500 (neg) | 362 | 0.100 | 1.03 | 0.103 | 1.10 | 0.109 |
| opp_cancel_touch_500 (neg) | 231 | 0.050 | 1.04 | 0.052 | 1.11 | 0.055 |
| opp_cancel_touch_500 (neg) | 79 | 0.010 | 1.09 | 0.011 | 1.03 | 0.010 |
| bigprev_500 | 0 | 0.228 | 1.00 | 0.228 | 1.04 | 0.236 |
| bigprev_500 | 1 | 0.027 | 1.02 | 0.028 | 1.02 | 0.028 |
| bigprev_500 | 2 | 0.002 | 1.14 | 0.002 | 0.23 | 0.001 |
| bigprev_2000 | 1 | 0.266 | 1.02 | 0.272 | 0.96 | 0.256 |
| bigprev_2000 | 2 | 0.082 | 1.02 | 0.084 | 0.94 | 0.077 |
| bigprev_2000 | 3 | 0.020 | 1.11 | 0.022 | 1.09 | 0.021 |
| bigprev_2000 | 4 | 0.004 | 1.02 | 0.004 | 1.09 | 0.004 |
| bigprev_100 | 0 | 0.056 | 1.01 | 0.057 | 1.07 | 0.060 |
| imb_risk (neg) | 0 | 0.500 | 1.01 | 0.505 | 1.01 | 0.507 |
| imb_risk (neg) | -0.884 | 0.100 | 0.98 | 0.098 | 1.02 | 0.102 |
| imb_risk (neg) | -0.959 | 0.050 | 0.99 | 0.050 | 0.96 | 0.048 |
| imb_risk (neg) | -0.995 | 0.010 | 0.96 | 0.010 | 0.73 | 0.007 |
| log_own | 6.81 | 0.500 | 1.00 | 0.502 | 1.00 | 0.498 |
| log_own | 7.97 | 0.100 | 1.03 | 0.103 | 1.07 | 0.107 |
| log_own | 8.26 | 0.050 | 1.05 | 0.053 | 1.08 | 0.054 |
| log_own | 8.88 | 0.010 | 1.09 | 0.011 | 0.96 | 0.010 |
| log_opp (neg) | 6.81 | 0.500 | 1.01 | 0.506 | 1.02 | 0.512 |
| log_opp (neg) | 4.56 | 0.100 | 0.97 | 0.097 | 0.99 | 0.099 |
| log_opp (neg) | 3.58 | 0.050 | 1.00 | 0.050 | 0.98 | 0.049 |
| log_opp (neg) | 1.61 | 0.008 | 0.95 | 0.008 | 0.74 | 0.006 |
| cnt_own | 16 | 0.493 | 1.00 | 0.494 | 1.00 | 0.495 |
| cnt_own | 33 | 0.091 | 1.02 | 0.092 | 0.95 | 0.086 |
| cnt_own | 37 | 0.046 | 1.05 | 0.048 | 1.03 | 0.047 |
| cnt_own | 44 | 0.010 | 0.92 | 0.009 | 0.67 | 0.006 |
| log_dt_mid (neg) | 6.28 | 0.500 | 1.00 | 0.499 | 1.04 | 0.518 |
| log_dt_mid (neg) | 3.96 | 0.100 | 1.00 | 0.100 | 1.05 | 0.105 |
| log_dt_mid (neg) | 3.07 | 0.050 | 0.99 | 0.049 | 1.04 | 0.052 |
| log_dt_mid (neg) | 0.773 | 0.010 | 0.98 | 0.010 | 1.05 | 0.010 |

AUC per feature for the >= 300 label (best first): ins_behind_vol_500 0.506, opp_cancel_touch_50 0.495, opp_cancel_touch_100 0.496, log_opp 0.497, opp_cancel_touch_500 0.497, hit_vol_500 0.503, imb_risk 0.497, ins_behind_vol_50 0.503, ins_behind_vol_100 0.502, probe_vol_500 0.502, ins_touch_vol_500 0.502, bigprev_2000 0.502, cancel_touch_vol_50 0.502, probe_n_500 0.498

## 4. Sweep-risk score, walk-forward (8 chunks; train on chunks < k with 1 s embargo, test chunk k = 1..7)

Greedy forward selection on chunks 0-2 (cumulative AUC): opp_cancel_touch_100 (0.510) > probe_vol_500 (0.511) > ins_behind_vol_500 (0.512) > imb_risk (0.513) > bigprev_2000 (0.513)

Mean standardized logistic coefficients of the 5-feature score: opp_cancel_touch_100 -0.02, probe_vol_500 +0.01, ins_behind_vol_500 +0.01, imb_risk -0.01, bigprev_2000 +0.01

Pooled OOS AUC (chunks 1-7):
| model | AUC >= 300 in 100 ms | AUC >= 1000 |
|---|---|---|
| 5-feature score | 0.499 | 0.502 |
| all-feature score | 0.497 | 0.499 |

Event-level recall: share of big sweeps whose pre-arrival state (last row <= t - 5 ms) is flagged (score cut from OOS pooled score quantile). Ridge flag = existing |pred| > 0.15 ticks against that side.
| score cut | sweeps | n | recall: score | recall: ridge | recall: score OR ridge |
|---|---|---|---|---|---|
| top 1% of rows | >= 300 | 10587 | 0.009 | 0.097 | 0.105 |
| top 1% of rows | >= 1000 | 1816 | 0.007 | 0.097 | 0.104 |
| top 2% of rows | >= 300 | 10587 | 0.018 | 0.097 | 0.114 |
| top 2% of rows | >= 1000 | 1816 | 0.018 | 0.097 | 0.114 |
| top 5% of rows | >= 300 | 10587 | 0.049 | 0.097 | 0.143 |
| top 5% of rows | >= 1000 | 1816 | 0.053 | 0.097 | 0.147 |
| top 10% of rows | >= 300 | 10587 | 0.100 | 0.097 | 0.191 |
| top 10% of rows | >= 1000 | 1816 | 0.100 | 0.097 | 0.192 |

Share of TIME with the side flagged by the score: top 1% of rows = 0.010, top 2% of rows = 0.020, top 5% of rows = 0.050, top 10% of rows = 0.100; by the ridge rule: 0.096.

Row-level precision / lift / recall of "a >= 300 IOC hits this side in (5,105] ms" (and >= 1000), 5-feature score:
| flag | precision | lift | recall | prec>=1000 | lift | recall |
|---|---|---|---|---|---|---|
| top 1% | 0.043 | 0.9 | 0.009 | 0.005 | 0.6 | 0.006 |
| top 2% | 0.045 | 0.9 | 0.018 | 0.007 | 0.8 | 0.016 |
| top 5% | 0.049 | 1.0 | 0.050 | 0.009 | 1.0 | 0.051 |
| top 10% | 0.050 | 1.0 | 0.101 | 0.009 | 1.0 | 0.100 |

Signed score (score_ask_hit - score_bid_hit) vs the ridge forecast and the 100 ms mid move y:
| model | corr(score, ridge) | IC(score, y) | IC(ridge, y) | corr(score, y - fit(ridge)) |
|---|---|---|---|---|
| 5-feature score | -0.103 | -0.027 | +0.380 | +0.013 |
| all-feature score | +0.047 | +0.067 | +0.380 | +0.053 |

  fold 1: IC base +0.3818  + score +0.3813  + score + raw precursors +0.3869
  fold 2: IC base +0.3815  + score +0.3811  + score + raw precursors +0.3863
  fold 3: IC base +0.3864  + score +0.3869  + score + raw precursors +0.3884
  fold 4: IC base +0.3761  + score +0.3771  + score + raw precursors +0.3795
  fold 5: IC base +0.3848  + score +0.3856  + score + raw precursors +0.3903
  fold 6: IC base +0.3717  + score +0.3756  + score + raw precursors +0.3772
  fold 7: IC base +0.3792  + score +0.3809  + score + raw precursors +0.3819

| model for the 100 ms mid move | mean OOS IC (7 folds) | folds improved vs base |
|---|---|---|
| ridge, 23 features | +0.3802 | - |
| + sweep score (signed, ask, bid) | +0.3812 | 5/7 |
| + score + raw side-difference precursors | +0.3844 | 7/7 |

## 5. Economics of a flag (maker view, 1 s markout, per lot, ticks)

Fill price approximated by the row's last trade price (exact for single-level fills). Maker markout = s * (mid(t+1s) - price) for the side hit.

| side | fills of | state | lots hit per flagged row (5-105 ms) | markout per lot | PnL per row |
|---|---|---|---|---|---|
| 1 | sweep rows | score top 2% | 26.98 | +0.114 | +3.070 |
| 1 | sweep rows | not flagged | 30.89 | +0.034 | +1.036 |
| 1 | sweep rows | ridge flag | 31.28 | +0.035 | +1.108 |
| 1 | other rows | score top 2% | 25.04 | +0.276 | +6.924 |
| 1 | other rows | not flagged | 25.38 | +0.317 | +8.058 |
| 1 | other rows | ridge flag | 26.26 | +0.178 | +4.666 |
| -1 | sweep rows | score top 2% | 29.73 | -0.048 | -1.426 |
| -1 | sweep rows | not flagged | 30.60 | +0.059 | +1.814 |
| -1 | sweep rows | ridge flag | 30.40 | +0.132 | +4.019 |
| -1 | other rows | score top 2% | 26.40 | +0.238 | +6.294 |
| -1 | other rows | not flagged | 25.24 | +0.312 | +7.877 |
| -1 | other rows | ridge flag | 25.92 | +0.173 | +4.480 |
| 1 | sweep rows | score top 5% | 30.81 | +0.048 | +1.471 |
| 1 | sweep rows | not flagged | 30.81 | +0.034 | +1.057 |
| 1 | sweep rows | ridge flag | 31.28 | +0.035 | +1.108 |
| 1 | other rows | score top 5% | 25.15 | +0.271 | +6.812 |
| 1 | other rows | not flagged | 25.39 | +0.319 | +8.100 |
| 1 | other rows | ridge flag | 26.26 | +0.178 | +4.666 |
| -1 | sweep rows | score top 5% | 32.73 | -0.004 | -0.146 |
| -1 | sweep rows | not flagged | 30.47 | +0.061 | +1.849 |
| -1 | sweep rows | ridge flag | 30.40 | +0.132 | +4.019 |
| -1 | other rows | score top 5% | 25.92 | +0.253 | +6.559 |
| -1 | other rows | not flagged | 25.23 | +0.314 | +7.913 |
| -1 | other rows | ridge flag | 25.92 | +0.173 | +4.480 |

