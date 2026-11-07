# Signals - `data\sample.csv`, horizon 1000 ms

Label: mid change from t + 5 ms to t + 5 ms + 1000 ms (ticks). 13,866 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 1,734 | +0.7687 | +0.7822 | +0.58705 | +5.069 | nan |
| 2 | ridge | 1,736 | +0.7436 | +0.7421 | +0.55191 | +5.110 | nan |
| 3 | ridge | 1,737 | +0.8127 | +0.7935 | +0.66235 | +5.402 | +6.583 |
| 4 | ridge | 1,733 | +0.8207 | +0.8260 | +0.67243 | +5.196 | +5.727 |
| 5 | ridge | 1,739 | +0.8240 | +0.7989 | +0.67789 | +6.140 | +10.513 |
| 6 | ridge | 1,742 | +0.7624 | +0.7644 | +0.56147 | +3.959 | +5.333 |
| 7 | ridge | 1,728 | +0.8115 | +0.8011 | +0.66053 | +5.455 | +6.800 |
| **mean** | ridge | | +0.7919 | +0.7869 | +0.62480 | +5.190 | +6.991 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | +0.0000 | -0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | -0.0000 |
| imb1 | +0.0178 | +0.0159 | +0.0108 | +0.0131 | +0.0129 | +0.0113 | +0.0106 | +0.0125 |
| imb2 | +0.0178 | +0.0159 | +0.0108 | +0.0131 | +0.0129 | +0.0113 | +0.0106 | +0.0125 |
| imb3 | +0.0178 | +0.0159 | +0.0108 | +0.0131 | +0.0129 | +0.0113 | +0.0106 | +0.0125 |
| imb5 | +0.0178 | +0.0159 | +0.0108 | +0.0131 | +0.0129 | +0.0113 | +0.0106 | +0.0125 |
| micro | +0.0178 | +0.0159 | +0.0108 | +0.0131 | +0.0129 | +0.0113 | +0.0106 | +0.0125 |
| cnt_imb1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| depth1_log | -0.0012 | +0.0092 | -0.0104 | -0.0038 | -0.0105 | -0.0185 | -0.0317 | -0.0297 |
| flow_ev1 | -0.2097 | -0.1038 | -0.0263 | -0.0091 | +0.0669 | +0.0702 | +0.0462 | +0.0433 |
| flow_ev10 | +0.1056 | +0.1451 | +0.0895 | +0.1420 | +0.1582 | +0.1450 | +0.1370 | +0.1330 |
| flow_ev100 | +0.6783 | +0.5725 | +0.5686 | +0.6282 | +0.5913 | +0.6245 | +0.6554 | +0.6622 |
| flow_ms10 | +0.1484 | +0.1050 | -0.0173 | +0.0008 | -0.1098 | -0.1381 | -0.1247 | -0.1213 |
| flow_ms100 | +0.0662 | -0.0394 | +0.0173 | -0.0382 | +0.0019 | +0.0384 | +0.0400 | +0.0443 |
| flow_ms1000 | +1.3545 | +1.1758 | +1.1015 | +1.0337 | +1.0151 | +1.1115 | +1.1205 | +1.1004 |
| flow_ms10000 | +0.0316 | +0.0676 | +0.1592 | +0.1470 | +0.1808 | +0.1487 | +0.0577 | +0.0641 |
| ntr_ms1000 | -0.3256 | -0.2058 | -0.1504 | -0.1311 | -0.1339 | -0.1255 | -0.0971 | -0.0616 |
| nev_ms1000 | -0.1360 | -0.0246 | +0.1060 | +0.0353 | +0.0053 | +0.0323 | +0.0716 | +0.0738 |
| ret_ms100 | +0.1028 | +0.0420 | +0.0397 | +0.0126 | -0.0396 | -0.0584 | -0.0826 | -0.0733 |
| ret_ms1000 | +1.4811 | +1.6872 | +1.6983 | +1.7393 | +1.8053 | +1.8227 | +1.7809 | +1.8327 |
| ret_ms10000 | -0.2863 | -0.3684 | -0.4234 | -0.3640 | -0.3370 | -0.3521 | -0.3053 | -0.2932 |
| last_dmid | -0.1863 | -0.2106 | -0.2110 | -0.2169 | -0.1909 | -0.1691 | -0.1622 | -0.1857 |
| dt_mid_log | +0.0052 | +0.0297 | +0.0784 | +0.0531 | +0.0263 | +0.0275 | +0.0309 | +0.0267 |
| ltp_dev | +0.1719 | +0.1577 | +0.1974 | +0.2499 | +0.2471 | +0.1898 | +0.1759 | +0.2042 |

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
