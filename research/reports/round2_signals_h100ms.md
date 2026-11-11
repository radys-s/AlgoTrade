# Signals - `data\round2.csv`, horizon 100 ms

Label: mid change from t + 5 ms to t + 5 ms + 100 ms (ticks). 2,188,070 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 274,005 | +0.3142 | +0.3350 | +0.07051 | +0.308 | +0.353 |
| 1 | hgb | 274,005 | +0.3078 | +0.3503 | +0.01732 | +0.344 | +0.284 |
| 2 | ridge | 273,993 | +0.2648 | +0.3492 | +0.06510 | +0.610 | +1.013 |
| 2 | hgb | 273,993 | +0.2880 | +0.3606 | +0.08172 | +0.608 | +1.257 |
| 3 | ridge | 274,014 | +0.2555 | +0.3515 | +0.06500 | +0.512 | +1.130 |
| 3 | hgb | 274,014 | +0.3128 | +0.3757 | +0.09654 | +0.623 | +1.281 |
| 4 | ridge | 273,995 | +0.2584 | +0.3479 | +0.06635 | +0.641 | +0.937 |
| 4 | hgb | 273,995 | +0.2958 | +0.3567 | +0.08561 | +0.587 | +1.555 |
| 5 | ridge | 273,898 | +0.3421 | +0.3951 | +0.10322 | +0.882 | +1.399 |
| 5 | hgb | 273,898 | +0.4127 | +0.4265 | +0.16857 | +0.867 | +1.976 |
| 6 | ridge | 274,086 | +0.2685 | +0.3376 | +0.05575 | +0.434 | +0.869 |
| 6 | hgb | 274,086 | +0.3601 | +0.3649 | +0.12823 | +0.627 | +2.252 |
| 7 | ridge | 273,817 | +0.1888 | +0.3736 | +0.03535 | +0.692 | +1.273 |
| 7 | hgb | 273,817 | +0.2385 | +0.4014 | +0.05677 | +0.816 | +1.777 |
| **mean** | ridge | | +0.2703 | +0.3557 | +0.06590 | +0.583 | +0.996 |
| **mean** | hgb | | +0.3165 | +0.3766 | +0.09068 | +0.639 | +1.483 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | -0.0093 | -0.0143 | +0.0052 | +0.0045 | +0.0040 | -0.0037 | -0.0006 | +0.0069 |
| imb1 | -0.0224 | -0.0075 | +0.0182 | +0.0146 | +0.0036 | +0.0023 | -0.0011 | +0.0207 |
| imb2 | +0.0426 | +0.0438 | +0.0468 | +0.0398 | +0.0324 | +0.0313 | +0.0344 | +0.0355 |
| imb3 | -0.0732 | -0.0659 | -0.0016 | +0.0200 | +0.0105 | -0.0093 | -0.0113 | -0.0031 |
| imb5 | +0.0283 | +0.0165 | -0.0492 | -0.0588 | -0.0399 | -0.0192 | -0.0214 | -0.0346 |
| micro | +0.0100 | +0.0135 | -0.0189 | -0.0177 | -0.0101 | -0.0168 | -0.0102 | -0.0414 |
| cnt_imb1 | +0.1354 | +0.1120 | +0.1206 | +0.1238 | +0.1299 | +0.1444 | +0.1390 | +0.1454 |
| depth1_log | +0.0202 | +0.0164 | +0.0029 | +0.0054 | +0.0033 | +0.0081 | +0.0064 | +0.0058 |
| flow_ev1 | +0.0066 | +0.0042 | +0.0044 | +0.0040 | +0.0040 | +0.0047 | +0.0042 | +0.0047 |
| flow_ev10 | +0.0198 | +0.0123 | +0.0045 | +0.0047 | +0.0047 | +0.0050 | +0.0041 | +0.0052 |
| flow_ev100 | +0.0296 | +0.0315 | +0.0431 | +0.0392 | +0.0380 | +0.0411 | +0.0403 | +0.0378 |
| flow_ms10 | -0.0020 | +0.0001 | +0.0058 | +0.0044 | +0.0078 | +0.0103 | +0.0096 | +0.0099 |
| flow_ms100 | +0.0118 | -0.0004 | +0.0237 | +0.0198 | +0.0255 | +0.0450 | +0.0384 | +0.0456 |
| flow_ms1000 | -0.0083 | +0.0036 | -0.0108 | -0.0079 | -0.0068 | -0.0179 | -0.0145 | -0.0128 |
| flow_ms10000 | -0.0028 | -0.0014 | -0.0010 | -0.0018 | -0.0022 | -0.0034 | -0.0035 | -0.0010 |
| ntr_ms1000 | -0.0032 | -0.0013 | +0.0049 | +0.0072 | +0.0074 | +0.0105 | +0.0086 | +0.0036 |
| nev_ms1000 | -0.0067 | -0.0000 | -0.0012 | +0.0061 | +0.0086 | +0.0103 | +0.0057 | -0.0061 |
| ret_ms100 | +0.1078 | +0.0779 | +0.0930 | +0.0973 | +0.0997 | +0.1195 | +0.1164 | +0.1197 |
| ret_ms1000 | +0.0157 | +0.0090 | +0.0256 | +0.0137 | +0.0026 | +0.0095 | +0.0081 | +0.0070 |
| ret_ms10000 | -0.0044 | -0.0038 | +0.0027 | +0.0017 | +0.0035 | -0.0001 | +0.0015 | -0.0062 |
| last_dmid | +0.0327 | +0.0253 | +0.0484 | +0.0461 | +0.0520 | +0.0645 | +0.0587 | +0.0647 |
| dt_mid_log | +0.0051 | -0.0009 | +0.0011 | +0.0039 | +0.0066 | +0.0081 | +0.0077 | +0.0053 |
| ltp_dev | +0.0440 | +0.0291 | +0.0462 | +0.0433 | +0.0391 | +0.0490 | +0.0461 | +0.0410 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| cnt_imb1 | +0.04689 |
| flow_ms100 | +0.03742 |
| flow_ev100 | +0.03463 |
| ret_ms100 | +0.02525 |
| last_dmid | +0.02404 |
| ret_ms1000 | +0.01204 |
| ret_ms10000 | +0.01059 |
| imb5 | +0.00445 |
| depth1_log | +0.00418 |
| dt_mid_log | +0.00412 |
| flow_ms1000 | +0.00385 |
| flow_ev10 | +0.00207 |
| imb2 | +0.00170 |
| imb3 | +0.00085 |
| imb1 | +0.00074 |
| flow_ms10 | +0.00049 |
| flow_ev1 | +0.00007 |
| flow_ms10000 | -0.00031 |
| ltp_dev | -0.00068 |
| nev_ms1000 | -0.00216 |
| ntr_ms1000 | -0.00276 |
| micro | -0.00344 |
| spread | -0.00833 |

## Feature definitions

- `spread`: best ask - best bid (ticks)
- `imb1`: (b1 - a1) / (b1 + a1), b1/a1 = quantity at the best bid/ask price
- `imb2`: same with the quantity of the best 2 prices per side
- `imb3`: same with the best 3 prices per side
- `imb5`: same with the best 5 prices per side
- `micro`: microprice - mid = spread * imb1 / 2 (ticks)
- `cnt_imb1`: (nb - na) / (nb + na), nb/na = number of orders at the best bid/ask
- `depth1_log`: log1p(b1 + a1)
- `flow_ev{N}`: signed traded volume (+ buyer-initiated) of the last N events incl. this one
- `flow_ms{T}`: signed traded volume of events with ts in (t - T ms, t]
- `ntr_ms1000`: number of fills in (t - 1 s, t]
- `nev_ms1000`: number of tape events in (t - 1 s, t]
- `ret_ms{T}`: mid now - mid after the last event with ts <= t - T ms (ticks; one-sided books keep the last mid)
- `last_dmid`: size of the most recent non-zero mid change (ticks, signed; 0 before the first)
- `dt_mid_log`: log1p(ms since the most recent mid change, or since the first event)
- `ltp_dev`: last trade price - mid (ticks; 0 before the first trade)
