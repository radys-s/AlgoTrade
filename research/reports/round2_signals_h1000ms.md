# Signals - `data\round2.csv`, horizon 1000 ms

Label: mid change from t + 5 ms to t + 5 ms + 1000 ms (ticks). 2,187,776 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 274,005 | +0.2815 | +0.2924 | +0.05659 | +0.819 | +1.012 |
| 1 | hgb | 274,005 | +0.1872 | +0.2680 | -0.06638 | +0.406 | -0.462 |
| 2 | ridge | 273,993 | +0.1300 | +0.2315 | +0.01183 | +0.698 | +0.774 |
| 2 | hgb | 273,993 | +0.0637 | +0.1862 | -0.05684 | +0.285 | +0.477 |
| 3 | ridge | 274,014 | +0.1412 | +0.2751 | +0.01211 | +0.750 | +1.113 |
| 3 | hgb | 274,014 | +0.1634 | +0.2587 | -0.01720 | +0.866 | +2.273 |
| 4 | ridge | 273,995 | +0.1569 | +0.2635 | +0.02353 | +1.018 | +0.775 |
| 4 | hgb | 273,995 | +0.1182 | +0.2553 | -0.03781 | +0.983 | +0.227 |
| 5 | ridge | 273,898 | +0.1445 | +0.2005 | +0.02319 | +1.473 | +2.215 |
| 5 | hgb | 273,898 | +0.0810 | +0.1581 | -0.03581 | +0.611 | +1.363 |
| 6 | ridge | 274,086 | +0.1527 | +0.2742 | +0.00526 | +0.482 | +1.073 |
| 6 | hgb | 274,086 | +0.0487 | +0.2768 | -0.09311 | +0.332 | -3.892 |
| 7 | ridge | 273,523 | +0.0882 | +0.2607 | +0.00617 | +0.708 | +0.324 |
| 7 | hgb | 273,523 | +0.0621 | +0.2369 | -0.01788 | +0.709 | -0.406 |
| **mean** | ridge | | +0.1564 | +0.2568 | +0.01981 | +0.850 | +1.041 |
| **mean** | hgb | | +0.1035 | +0.2343 | -0.04643 | +0.599 | -0.060 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | -0.0563 | -0.1185 | +0.0149 | +0.0132 | +0.0050 | -0.0974 | -0.0637 | -0.0016 |
| imb1 | -0.0339 | +0.0039 | -0.0992 | -0.1397 | -0.1129 | -0.0917 | -0.1008 | -0.0851 |
| imb2 | +0.2418 | +0.1673 | +0.0138 | -0.0009 | -0.0561 | -0.0813 | -0.0297 | -0.0289 |
| imb3 | -0.3434 | -0.2855 | +0.1595 | +0.0748 | +0.1061 | +0.0682 | +0.0382 | +0.0677 |
| imb5 | -0.0518 | +0.0246 | -0.2630 | -0.1293 | -0.0956 | -0.0356 | -0.0570 | -0.0963 |
| micro | -0.1308 | -0.1512 | -0.0541 | -0.0035 | -0.0465 | -0.0493 | -0.0500 | -0.0631 |
| cnt_imb1 | +0.3502 | +0.3470 | +0.3922 | +0.3692 | +0.3765 | +0.3609 | +0.3619 | +0.3671 |
| depth1_log | +0.1356 | +0.1259 | -0.0182 | +0.0145 | +0.0198 | +0.0732 | +0.0568 | +0.0599 |
| flow_ev1 | +0.0076 | +0.0037 | +0.0005 | +0.0020 | +0.0041 | +0.0076 | +0.0068 | +0.0067 |
| flow_ev10 | +0.0266 | +0.0134 | +0.0068 | +0.0100 | +0.0157 | +0.0224 | +0.0203 | +0.0210 |
| flow_ev100 | +0.1456 | +0.1476 | +0.1480 | +0.1387 | +0.1404 | +0.1431 | +0.1439 | +0.1346 |
| flow_ms10 | +0.0093 | +0.0136 | +0.0279 | +0.0216 | +0.0219 | +0.0232 | +0.0221 | +0.0244 |
| flow_ms100 | +0.1019 | +0.0797 | +0.0974 | +0.0929 | +0.1036 | +0.1309 | +0.1189 | +0.1313 |
| flow_ms1000 | +0.0689 | +0.0411 | +0.0221 | +0.0594 | +0.0250 | -0.0067 | +0.0040 | -0.0072 |
| flow_ms10000 | -0.0749 | -0.0552 | -0.0516 | -0.0439 | -0.0317 | -0.0102 | -0.0057 | +0.0457 |
| ntr_ms1000 | +0.0458 | +0.0587 | +0.0409 | +0.0663 | +0.0639 | +0.0863 | +0.0766 | +0.0309 |
| nev_ms1000 | -0.0241 | +0.0135 | -0.0022 | +0.0739 | +0.0655 | +0.1025 | +0.0587 | -0.0179 |
| ret_ms100 | +0.1221 | +0.0820 | +0.1776 | +0.1432 | +0.1215 | +0.1221 | +0.1314 | +0.1583 |
| ret_ms1000 | +0.0872 | +0.0445 | -0.0374 | -0.1195 | -0.1214 | -0.0170 | -0.0276 | -0.0304 |
| ret_ms10000 | +0.1117 | +0.0537 | +0.1644 | +0.1258 | +0.1228 | +0.0678 | +0.0735 | -0.0102 |
| last_dmid | +0.0762 | +0.0817 | +0.1402 | +0.1255 | +0.1308 | +0.1267 | +0.1138 | +0.1106 |
| dt_mid_log | +0.1014 | +0.0172 | +0.0232 | +0.0232 | +0.0377 | +0.0425 | +0.0397 | +0.0592 |
| ltp_dev | +0.1098 | +0.0894 | +0.1398 | +0.1156 | +0.1011 | +0.1092 | +0.1063 | +0.1027 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| flow_ev100 | +0.04067 |
| flow_ms10000 | +0.03939 |
| flow_ms100 | +0.03707 |
| ret_ms10000 | +0.02940 |
| cnt_imb1 | +0.01883 |
| micro | +0.01634 |
| flow_ms1000 | +0.01627 |
| ret_ms100 | +0.00285 |
| ntr_ms1000 | +0.00237 |
| flow_ev10 | +0.00217 |
| flow_ms10 | +0.00055 |
| imb2 | +0.00016 |
| flow_ev1 | +0.00003 |
| imb1 | +0.00003 |
| imb3 | -0.00004 |
| dt_mid_log | -0.00067 |
| ltp_dev | -0.00088 |
| ret_ms1000 | -0.00217 |
| last_dmid | -0.00324 |
| imb5 | -0.00908 |
| spread | -0.01544 |
| nev_ms1000 | -0.01940 |
| depth1_log | -0.02559 |

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
