# Signals - `data\sample.csv`, horizon 100 ms

Label: mid change from t + 5 ms to t + 5 ms + 100 ms (ticks). 13,870 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 1,734 | +0.3085 | +0.2678 | +0.09294 | +0.448 | +0.000 |
| 1 | hgb | 1,734 | +0.2093 | +0.1804 | +0.03466 | +0.183 | +0.727 |
| 2 | ridge | 1,736 | +0.2983 | +0.2639 | +0.08861 | +0.384 | +1.000 |
| 2 | hgb | 1,736 | +0.2663 | +0.2283 | +0.07074 | +0.444 | +0.571 |
| 3 | ridge | 1,737 | +0.3715 | +0.3411 | +0.13522 | +0.447 | +1.222 |
| 3 | hgb | 1,737 | +0.3680 | +0.3279 | +0.13034 | +0.466 | +0.737 |
| 4 | ridge | 1,733 | +0.3639 | +0.3225 | +0.13218 | +0.466 | +0.700 |
| 4 | hgb | 1,733 | +0.3410 | +0.3006 | +0.11570 | +0.481 | +0.833 |
| 5 | ridge | 1,739 | +0.3937 | +0.3428 | +0.15558 | +0.536 | +0.828 |
| 5 | hgb | 1,739 | +0.4145 | +0.3717 | +0.16966 | +0.527 | +0.667 |
| 6 | ridge | 1,742 | +0.3594 | +0.3374 | +0.12700 | +0.500 | +0.600 |
| 6 | hgb | 1,742 | +0.3691 | +0.3528 | +0.13570 | +0.461 | +0.700 |
| 7 | ridge | 1,732 | +0.3252 | +0.2966 | +0.10617 | +0.446 | +0.846 |
| 7 | hgb | 1,732 | +0.3526 | +0.3036 | +0.12668 | +0.472 | +0.929 |
| **mean** | ridge | | +0.3458 | +0.3103 | +0.11967 | +0.461 | +0.742 |
| **mean** | hgb | | +0.3315 | +0.2950 | +0.11193 | +0.433 | +0.738 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | +0.0000 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | -0.0000 | -0.0000 | +0.0000 |
| imb1 | -0.0003 | -0.0010 | -0.0002 | +0.0003 | +0.0006 | +0.0002 | -0.0002 | -0.0005 |
| imb2 | -0.0003 | -0.0010 | -0.0002 | +0.0003 | +0.0006 | +0.0002 | -0.0002 | -0.0005 |
| imb3 | -0.0003 | -0.0010 | -0.0002 | +0.0003 | +0.0006 | +0.0002 | -0.0002 | -0.0005 |
| imb5 | -0.0003 | -0.0010 | -0.0002 | +0.0003 | +0.0006 | +0.0002 | -0.0002 | -0.0005 |
| micro | -0.0003 | -0.0010 | -0.0002 | +0.0003 | +0.0006 | +0.0002 | -0.0002 | -0.0005 |
| cnt_imb1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| depth1_log | +0.0028 | +0.0064 | -0.0003 | -0.0007 | -0.0025 | -0.0027 | -0.0056 | -0.0046 |
| flow_ev1 | -0.0216 | -0.0087 | +0.0014 | +0.0038 | +0.0016 | +0.0033 | +0.0023 | +0.0047 |
| flow_ev10 | +0.0140 | +0.0082 | +0.0004 | +0.0166 | +0.0210 | +0.0165 | +0.0141 | +0.0132 |
| flow_ev100 | +0.0570 | +0.0618 | +0.0542 | +0.0657 | +0.0640 | +0.0589 | +0.0655 | +0.0650 |
| flow_ms10 | +0.0440 | +0.0159 | -0.0028 | -0.0076 | -0.0104 | -0.0177 | -0.0126 | -0.0122 |
| flow_ms100 | -0.0750 | -0.0644 | -0.0542 | -0.0554 | -0.0534 | -0.0426 | -0.0472 | -0.0449 |
| flow_ms1000 | +0.0976 | +0.0895 | +0.0835 | +0.0656 | +0.0627 | +0.0659 | +0.0713 | +0.0661 |
| flow_ms10000 | +0.0215 | +0.0008 | +0.0152 | +0.0142 | +0.0128 | +0.0148 | +0.0062 | +0.0072 |
| ntr_ms1000 | -0.0105 | +0.0075 | +0.0035 | +0.0015 | -0.0014 | +0.0004 | -0.0003 | +0.0025 |
| nev_ms1000 | -0.0337 | -0.0241 | -0.0090 | -0.0116 | -0.0201 | -0.0120 | -0.0060 | -0.0024 |
| ret_ms100 | -0.1671 | -0.1517 | -0.1581 | -0.1660 | -0.1737 | -0.1735 | -0.1827 | -0.1771 |
| ret_ms1000 | +0.1563 | +0.1718 | +0.1730 | +0.1788 | +0.1817 | +0.1982 | +0.1970 | +0.1974 |
| ret_ms10000 | -0.0345 | -0.0255 | -0.0292 | -0.0283 | -0.0323 | -0.0350 | -0.0298 | -0.0311 |
| last_dmid | +0.0224 | -0.0067 | -0.0047 | -0.0073 | +0.0018 | -0.0047 | -0.0057 | -0.0074 |
| dt_mid_log | +0.0016 | +0.0036 | +0.0050 | +0.0021 | -0.0049 | -0.0011 | -0.0008 | +0.0024 |
| ltp_dev | +0.0058 | +0.0131 | +0.0175 | +0.0162 | +0.0165 | +0.0121 | +0.0098 | +0.0101 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| ret_ms1000 | +0.23235 |
| ret_ms100 | +0.11043 |
| flow_ev100 | +0.10158 |
| nev_ms1000 | +0.06281 |
| flow_ms1000 | +0.04424 |
| dt_mid_log | +0.02523 |
| ret_ms10000 | +0.00950 |
| last_dmid | +0.00416 |
| ntr_ms1000 | +0.00271 |
| flow_ev10 | +0.00204 |
| flow_ms10000 | +0.00106 |
| imb1 | +0.00012 |
| spread | +0.00000 |
| imb2 | +0.00000 |
| imb3 | +0.00000 |
| imb5 | +0.00000 |
| micro | +0.00000 |
| cnt_imb1 | +0.00000 |
| flow_ev1 | +0.00000 |
| flow_ms10 | +0.00000 |
| flow_ms100 | +0.00000 |
| ltp_dev | -0.00108 |
| depth1_log | -0.00462 |

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
