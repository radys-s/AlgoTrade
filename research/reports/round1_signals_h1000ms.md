# Signals - `data\round1.csv`, horizon 1000 ms

Label: mid change from t + 5 ms to t + 5 ms + 1000 ms (ticks). 2,203,446 rows (tape events with stride 2). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 275,685 | +0.4039 | +0.4058 | +0.16258 | +0.776 | +0.982 |
| 1 | hgb | 275,685 | +0.4006 | +0.3977 | +0.15706 | +0.854 | +1.334 |
| 2 | ridge | 275,674 | +0.4079 | +0.4014 | +0.16632 | +0.824 | +1.014 |
| 2 | hgb | 275,674 | +0.4154 | +0.4003 | +0.17046 | +0.870 | +1.243 |
| 3 | ridge | 275,674 | +0.4023 | +0.3989 | +0.16190 | +0.775 | +0.997 |
| 3 | hgb | 275,674 | +0.4031 | +0.3848 | +0.16081 | +0.833 | +1.422 |
| 4 | ridge | 275,661 | +0.3868 | +0.3801 | +0.14892 | +0.785 | +1.014 |
| 4 | hgb | 275,661 | +0.3980 | +0.3799 | +0.15650 | +0.832 | +1.281 |
| 5 | ridge | 275,670 | +0.3932 | +0.3932 | +0.15351 | +0.722 | +0.953 |
| 5 | hgb | 275,670 | +0.4060 | +0.3972 | +0.16313 | +0.809 | +1.219 |
| 6 | ridge | 275,670 | +0.3907 | +0.3916 | +0.15187 | +0.704 | +0.857 |
| 6 | hgb | 275,670 | +0.4154 | +0.4036 | +0.17179 | +0.816 | +1.354 |
| 7 | ridge | 275,476 | +0.3965 | +0.3964 | +0.15719 | +0.793 | +0.851 |
| 7 | hgb | 275,476 | +0.4198 | +0.4105 | +0.17617 | +0.858 | +1.134 |
| **mean** | ridge | | +0.3973 | +0.3954 | +0.15747 | +0.769 | +0.953 |
| **mean** | hgb | | +0.4083 | +0.3963 | +0.16513 | +0.839 | +1.284 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | +0.0025 | -0.0046 | -0.0032 | +0.0018 | -0.0014 | -0.0031 | -0.0029 | -0.0021 |
| imb1 | -0.0247 | +0.0022 | -0.0029 | +0.0047 | -0.0067 | -0.0087 | -0.0091 | +0.0034 |
| imb2 | +0.1439 | +0.1355 | +0.1031 | +0.1027 | +0.1072 | +0.1085 | +0.1100 | +0.1066 |
| imb3 | -0.1045 | -0.0816 | -0.0443 | -0.0477 | -0.0151 | -0.0224 | -0.0087 | -0.0263 |
| imb5 | -0.0387 | -0.0548 | -0.0574 | -0.0594 | -0.0973 | -0.0923 | -0.1049 | -0.0870 |
| micro | -0.0631 | -0.0742 | -0.0563 | -0.0549 | -0.0473 | -0.0452 | -0.0431 | -0.0511 |
| cnt_imb1 | +0.3387 | +0.3259 | +0.3201 | +0.3155 | +0.3109 | +0.3103 | +0.3026 | +0.3024 |
| depth1_log | +0.0027 | +0.0087 | +0.0057 | +0.0011 | +0.0047 | +0.0054 | +0.0042 | +0.0040 |
| flow_ev1 | +0.0019 | +0.0026 | +0.0027 | +0.0019 | +0.0018 | +0.0019 | +0.0020 | +0.0018 |
| flow_ev10 | +0.0127 | +0.0101 | +0.0097 | +0.0094 | +0.0093 | +0.0091 | +0.0087 | +0.0088 |
| flow_ev100 | +0.1057 | +0.0970 | +0.0926 | +0.0889 | +0.0914 | +0.0909 | +0.0892 | +0.0902 |
| flow_ms10 | +0.0004 | -0.0004 | -0.0001 | -0.0005 | -0.0007 | -0.0005 | -0.0003 | -0.0002 |
| flow_ms100 | +0.0395 | +0.0425 | +0.0397 | +0.0391 | +0.0411 | +0.0409 | +0.0398 | +0.0393 |
| flow_ms1000 | +0.1033 | +0.1198 | +0.1133 | +0.1068 | +0.1046 | +0.0994 | +0.1025 | +0.1014 |
| flow_ms10000 | -0.0224 | -0.0125 | -0.0195 | -0.0065 | +0.0035 | +0.0114 | +0.0142 | +0.0064 |
| ntr_ms1000 | -0.0006 | +0.0025 | -0.0021 | +0.0007 | +0.0058 | +0.0120 | +0.0124 | +0.0097 |
| nev_ms1000 | +0.0026 | +0.0019 | +0.0007 | +0.0066 | +0.0040 | +0.0034 | +0.0008 | +0.0019 |
| ret_ms100 | +0.0177 | +0.0123 | +0.0167 | +0.0132 | +0.0086 | +0.0083 | +0.0071 | +0.0090 |
| ret_ms1000 | -0.0453 | -0.0736 | -0.0683 | -0.0550 | -0.0504 | -0.0533 | -0.0549 | -0.0561 |
| ret_ms10000 | -0.0041 | -0.0079 | +0.0016 | -0.0153 | -0.0186 | -0.0252 | -0.0224 | -0.0138 |
| last_dmid | +0.0464 | +0.0576 | +0.0512 | +0.0538 | +0.0503 | +0.0519 | +0.0441 | +0.0463 |
| dt_mid_log | +0.0076 | +0.0101 | +0.0110 | +0.0129 | +0.0043 | +0.0083 | +0.0080 | +0.0084 |
| ltp_dev | +0.0711 | +0.0714 | +0.0784 | +0.0812 | +0.0808 | +0.0803 | +0.0813 | +0.0801 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| cnt_imb1 | +0.22902 |
| flow_ev100 | +0.04787 |
| flow_ms1000 | +0.03390 |
| flow_ms100 | +0.01112 |
| ltp_dev | +0.00842 |
| dt_mid_log | +0.00721 |
| ret_ms1000 | +0.00681 |
| ntr_ms1000 | +0.00553 |
| imb5 | +0.00216 |
| last_dmid | +0.00158 |
| nev_ms1000 | +0.00105 |
| flow_ev10 | +0.00072 |
| imb2 | +0.00068 |
| ret_ms100 | +0.00035 |
| flow_ms10 | +0.00010 |
| imb3 | +0.00008 |
| spread | +0.00003 |
| flow_ev1 | +0.00000 |
| depth1_log | -0.00002 |
| imb1 | -0.00042 |
| micro | -0.00050 |
| ret_ms10000 | -0.00071 |
| flow_ms10000 | -0.00109 |

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
