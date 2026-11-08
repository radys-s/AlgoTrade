# Signals - `data\round3_public.csv`, horizon 1000 ms

Label: mid change from t + 5 ms to t + 5 ms + 1000 ms (ticks). 2,407,824 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 301,511 | +0.3124 | +0.3136 | +0.09668 | +2.842 | +4.494 |
| 1 | hgb | 301,511 | +0.2671 | +0.2484 | +0.03316 | +2.204 | +19.036 |
| 2 | ridge | 301,524 | +0.3063 | +0.2894 | +0.07862 | +2.316 | +4.933 |
| 2 | hgb | 301,524 | +0.2084 | +0.2787 | -0.20531 | +1.432 | +2.776 |
| 3 | ridge | 301,563 | +0.2989 | +0.3489 | +0.08958 | +3.000 | +3.323 |
| 3 | hgb | 301,563 | +0.2350 | +0.3223 | +0.02406 | +2.813 | +1.234 |
| 4 | ridge | 301,539 | +0.2587 | +0.2842 | +0.05900 | +2.517 | +2.366 |
| 4 | hgb | 301,539 | +0.3108 | +0.2791 | +0.07996 | +2.901 | +5.095 |
| 5 | ridge | 301,483 | +0.2482 | +0.2995 | +0.06106 | +3.312 | +5.001 |
| 5 | hgb | 301,483 | +0.2458 | +0.2700 | +0.05489 | +3.064 | +4.741 |
| 6 | ridge | 301,504 | +0.2622 | +0.3142 | +0.05886 | +2.364 | +3.285 |
| 6 | hgb | 301,504 | +0.2081 | +0.3010 | +0.00100 | +2.161 | +0.733 |
| 7 | ridge | 301,138 | +0.3655 | +0.3550 | +0.13139 | +3.129 | +5.039 |
| 7 | hgb | 301,138 | +0.3759 | +0.3368 | +0.13902 | +3.278 | +5.537 |
| **mean** | ridge | | +0.2932 | +0.3150 | +0.08217 | +2.783 | +4.063 |
| **mean** | hgb | | +0.2644 | +0.2909 | +0.01811 | +2.551 | +5.593 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | -0.0110 | +0.1482 | +0.0518 | -0.0064 | +0.0402 | +0.0614 | +0.0403 | +0.0229 |
| imb1 | -0.4436 | -0.4558 | -0.3071 | -0.2393 | -0.2231 | -0.2570 | -0.1871 | -0.2135 |
| imb2 | +0.2007 | +0.1721 | +0.1149 | +0.1021 | +0.0916 | +0.0689 | +0.0902 | +0.0958 |
| imb3 | +0.2203 | +0.0582 | +0.2135 | +0.1913 | +0.2387 | +0.2004 | +0.2273 | +0.2073 |
| imb5 | -0.1611 | -0.0265 | -0.1811 | -0.1816 | -0.1830 | -0.1538 | -0.2307 | -0.1960 |
| micro | +0.1740 | +0.1514 | +0.0066 | -0.0995 | -0.1258 | -0.0898 | -0.1391 | -0.1096 |
| cnt_imb1 | +0.6870 | +0.7273 | +0.6651 | +0.7211 | +0.7181 | +0.7158 | +0.6767 | +0.6890 |
| depth1_log | -0.1017 | -0.2168 | -0.0957 | -0.0132 | -0.0420 | -0.0628 | -0.0386 | -0.0426 |
| flow_ev1 | -0.0062 | +0.0042 | +0.0046 | +0.0061 | +0.0059 | +0.0059 | +0.0057 | +0.0054 |
| flow_ev10 | +0.0118 | +0.0426 | +0.0337 | +0.0367 | +0.0331 | +0.0235 | +0.0197 | +0.0195 |
| flow_ev100 | -0.0166 | +0.0097 | +0.0608 | +0.0845 | +0.0828 | +0.0914 | +0.0970 | +0.1051 |
| flow_ms10 | +0.0254 | +0.0059 | +0.0240 | +0.0318 | +0.0343 | +0.0358 | +0.0369 | +0.0388 |
| flow_ms100 | +0.0945 | +0.1267 | +0.0988 | +0.1318 | +0.1498 | +0.2088 | +0.1930 | +0.1912 |
| flow_ms1000 | +0.1690 | +0.1422 | +0.1545 | +0.1957 | +0.1859 | +0.1629 | +0.1405 | +0.1571 |
| flow_ms10000 | -0.3113 | -0.2873 | -0.2566 | -0.2878 | -0.2659 | -0.2557 | -0.2460 | -0.2489 |
| ntr_ms1000 | +0.0638 | +0.0259 | -0.0301 | +0.0866 | +0.0700 | +0.1029 | +0.1135 | +0.1027 |
| nev_ms1000 | -0.1110 | -0.3326 | -0.2571 | -0.0723 | -0.1230 | -0.1601 | -0.1221 | -0.1172 |
| ret_ms100 | +0.4741 | +0.4286 | +0.4565 | +0.4705 | +0.4413 | +0.3330 | +0.3374 | +0.3632 |
| ret_ms1000 | +0.1706 | +0.1361 | +0.1946 | +0.2165 | +0.1357 | +0.2470 | +0.2377 | +0.2090 |
| ret_ms10000 | -0.0237 | -0.1268 | -0.0830 | -0.0293 | -0.0201 | -0.0262 | -0.0136 | -0.0314 |
| last_dmid | +0.3065 | +0.3839 | +0.2561 | +0.2825 | +0.2998 | +0.3212 | +0.3156 | +0.3378 |
| dt_mid_log | -0.0645 | -0.0801 | -0.0380 | -0.0013 | -0.0111 | -0.0131 | -0.0114 | -0.0169 |
| ltp_dev | +0.2919 | +0.1841 | +0.1633 | +0.1396 | +0.1136 | +0.1332 | +0.1300 | +0.1396 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| cnt_imb1 | +0.04901 |
| flow_ev100 | +0.03988 |
| flow_ms1000 | +0.03660 |
| last_dmid | +0.02982 |
| flow_ms10000 | +0.02618 |
| ret_ms1000 | +0.02276 |
| flow_ms100 | +0.02092 |
| nev_ms1000 | +0.01561 |
| ltp_dev | +0.01002 |
| imb3 | +0.00857 |
| ret_ms100 | +0.00758 |
| imb2 | +0.00331 |
| ntr_ms1000 | +0.00315 |
| imb5 | +0.00227 |
| ret_ms10000 | +0.00182 |
| flow_ev10 | +0.00107 |
| flow_ms10 | +0.00034 |
| depth1_log | +0.00031 |
| imb1 | +0.00027 |
| flow_ev1 | +0.00000 |
| dt_mid_log | -0.00052 |
| micro | -0.00064 |
| spread | -0.00198 |

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
