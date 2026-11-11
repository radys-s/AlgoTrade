# Signals - `..\AlgoTradeArena\data\round4_public.csv`, horizon 100 ms

Label: mid change from t + 5 ms to t + 5 ms + 100 ms (ticks). 2,393,895 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 299,764 | +0.3306 | +0.4871 | +0.10826 | +1.074 | +1.584 |
| 1 | hgb | 299,764 | +0.1513 | +0.5163 | -0.28417 | +1.029 | -0.049 |
| 2 | ridge | 299,667 | +0.3153 | +0.5312 | +0.09978 | +1.126 | +1.276 |
| 2 | hgb | 299,667 | +0.3575 | +0.5428 | +0.12733 | +1.171 | +2.482 |
| 3 | ridge | 299,715 | +0.2964 | +0.4278 | +0.08512 | +1.202 | +1.018 |
| 3 | hgb | 299,715 | +0.3527 | +0.4631 | +0.12346 | +1.224 | +2.429 |
| 4 | ridge | 299,649 | +0.4253 | +0.5061 | +0.18043 | +1.237 | +1.421 |
| 4 | hgb | 299,649 | +0.4599 | +0.5348 | +0.21151 | +1.268 | +1.771 |
| 5 | ridge | 299,720 | +0.4509 | +0.4812 | +0.20385 | +1.186 | +1.674 |
| 5 | hgb | 299,720 | +0.4752 | +0.5101 | +0.22069 | +1.176 | +1.355 |
| 6 | ridge | 299,764 | +0.3248 | +0.4712 | +0.10550 | +1.200 | +1.826 |
| 6 | hgb | 299,764 | +0.3503 | +0.5000 | +0.12188 | +1.221 | +1.840 |
| 7 | ridge | 299,554 | +0.4204 | +0.4920 | +0.17677 | +1.206 | +1.347 |
| 7 | hgb | 299,554 | +0.4774 | +0.5216 | +0.22780 | +1.256 | +2.108 |
| **mean** | ridge | | +0.3662 | +0.4852 | +0.13710 | +1.176 | +1.450 |
| **mean** | hgb | | +0.3749 | +0.5127 | +0.10693 | +1.192 | +1.705 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | -0.0441 | -0.0204 | -0.0148 | -0.0209 | -0.0187 | -0.0164 | -0.0158 | -0.0123 |
| imb1 | -0.1089 | -0.0844 | -0.0562 | -0.0572 | -0.0547 | -0.0451 | -0.0381 | -0.0415 |
| imb2 | +0.0462 | +0.0488 | +0.0357 | +0.0344 | +0.0318 | +0.0331 | +0.0345 | +0.0361 |
| imb3 | +0.0485 | +0.0466 | +0.0643 | +0.0571 | +0.0478 | +0.0490 | +0.0518 | +0.0504 |
| imb5 | -0.0163 | -0.0232 | -0.0118 | -0.0107 | +0.0039 | -0.0018 | -0.0094 | -0.0082 |
| micro | +0.0593 | +0.0322 | +0.0008 | +0.0043 | -0.0026 | -0.0124 | -0.0146 | -0.0106 |
| cnt_imb1 | +0.2564 | +0.2399 | +0.2447 | +0.2342 | +0.2386 | +0.2338 | +0.2276 | +0.2270 |
| depth1_log | +0.0357 | +0.0032 | -0.0083 | -0.0034 | +0.0014 | -0.0001 | -0.0008 | -0.0057 |
| flow_ev1 | +0.0044 | +0.0070 | +0.0076 | +0.0068 | +0.0069 | +0.0068 | +0.0062 | +0.0059 |
| flow_ev10 | +0.0176 | +0.0222 | +0.0225 | +0.0203 | +0.0201 | +0.0224 | +0.0205 | +0.0212 |
| flow_ev100 | +0.0194 | +0.0302 | +0.0383 | +0.0481 | +0.0511 | +0.0549 | +0.0528 | +0.0504 |
| flow_ms10 | +0.0227 | +0.0127 | +0.0126 | +0.0124 | +0.0128 | +0.0124 | +0.0131 | +0.0129 |
| flow_ms100 | -0.0334 | -0.0357 | -0.0083 | -0.0057 | +0.0050 | +0.0057 | +0.0044 | +0.0020 |
| flow_ms1000 | +0.0071 | +0.0074 | -0.0147 | -0.0470 | -0.0465 | -0.0547 | -0.0481 | -0.0452 |
| flow_ms10000 | -0.0256 | -0.0197 | -0.0257 | -0.0173 | -0.0221 | -0.0195 | -0.0184 | -0.0190 |
| ntr_ms1000 | -0.0095 | +0.0014 | -0.0095 | -0.0030 | -0.0022 | +0.0011 | -0.0020 | -0.0072 |
| nev_ms1000 | +0.1197 | +0.0225 | -0.0071 | +0.0020 | +0.0068 | -0.0005 | +0.0014 | -0.0022 |
| ret_ms100 | +0.2401 | +0.2039 | +0.1814 | +0.1660 | +0.1589 | +0.1596 | +0.1613 | +0.1656 |
| ret_ms1000 | +0.0205 | +0.1029 | +0.1180 | +0.1462 | +0.1532 | +0.1594 | +0.1535 | +0.1459 |
| ret_ms10000 | +0.0286 | +0.0086 | +0.0283 | +0.0142 | +0.0178 | +0.0148 | +0.0150 | +0.0174 |
| last_dmid | +0.1265 | +0.1164 | +0.1394 | +0.1199 | +0.1207 | +0.1153 | +0.1073 | +0.1100 |
| dt_mid_log | +0.0174 | -0.0084 | -0.0105 | -0.0059 | -0.0050 | -0.0043 | -0.0058 | -0.0066 |
| ltp_dev | +0.0089 | +0.0374 | +0.0457 | +0.0531 | +0.0575 | +0.0575 | +0.0565 | +0.0577 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| cnt_imb1 | +0.10571 |
| ret_ms1000 | +0.04131 |
| ret_ms100 | +0.02754 |
| flow_ev100 | +0.02276 |
| last_dmid | +0.02227 |
| ntr_ms1000 | +0.01787 |
| depth1_log | +0.01137 |
| flow_ms100 | +0.01045 |
| ltp_dev | +0.01030 |
| nev_ms1000 | +0.00865 |
| imb3 | +0.00609 |
| dt_mid_log | +0.00609 |
| micro | +0.00296 |
| flow_ev10 | +0.00190 |
| imb2 | +0.00189 |
| ret_ms10000 | +0.00102 |
| flow_ms10000 | +0.00098 |
| spread | +0.00077 |
| flow_ms10 | +0.00036 |
| flow_ms1000 | +0.00024 |
| imb5 | +0.00023 |
| flow_ev1 | +0.00000 |
| imb1 | -0.00008 |

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
