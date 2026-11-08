# Signals - `..\AlgoTradeArena\data\round4_public.csv`, horizon 1000 ms

Label: mid change from t + 5 ms to t + 5 ms + 1000 ms (ticks). 2,393,456 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 299,764 | +0.4829 | +0.4670 | +0.23271 | +7.707 | +7.875 |
| 1 | hgb | 299,764 | +0.3614 | +0.4194 | -0.02911 | +7.579 | -2.508 |
| 2 | ridge | 299,667 | +0.4476 | +0.4959 | +0.20199 | +7.224 | +5.699 |
| 2 | hgb | 299,667 | +0.4716 | +0.4760 | +0.19423 | +7.000 | +10.761 |
| 3 | ridge | 299,715 | +0.3630 | +0.3387 | +0.09688 | +6.578 | +6.800 |
| 3 | hgb | 299,715 | +0.4523 | +0.3594 | +0.19362 | +8.611 | +18.678 |
| 4 | ridge | 299,649 | +0.4986 | +0.4775 | +0.24853 | +7.396 | +6.979 |
| 4 | hgb | 299,649 | +0.5410 | +0.4581 | +0.28681 | +7.874 | +14.862 |
| 5 | ridge | 299,720 | +0.5149 | +0.4143 | +0.26679 | +7.212 | +6.634 |
| 5 | hgb | 299,720 | +0.5540 | +0.4275 | +0.29775 | +7.804 | +8.930 |
| 6 | ridge | 299,764 | +0.4743 | +0.3964 | +0.22455 | +8.679 | +6.407 |
| 6 | hgb | 299,764 | +0.4744 | +0.4378 | +0.21166 | +8.046 | +7.923 |
| 7 | ridge | 299,115 | +0.4913 | +0.4463 | +0.23910 | +7.178 | +3.429 |
| 7 | hgb | 299,115 | +0.5663 | +0.4765 | +0.31838 | +7.841 | +12.119 |
| **mean** | ridge | | +0.4675 | +0.4337 | +0.21579 | +7.425 | +6.260 |
| **mean** | hgb | | +0.4887 | +0.4364 | +0.21048 | +7.822 | +10.109 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | -0.6658 | -0.1499 | -0.0141 | -0.0273 | -0.0467 | -0.0288 | -0.0496 | +0.0023 |
| imb1 | -1.2615 | -0.7023 | -0.3413 | -0.3689 | -0.4822 | -0.4406 | -0.4643 | -0.5196 |
| imb2 | +0.1514 | +0.1816 | +0.2616 | +0.1864 | +0.1614 | +0.1279 | +0.1626 | +0.1540 |
| imb3 | +0.5214 | +0.4442 | +0.5390 | +0.4970 | +0.4607 | +0.4616 | +0.4636 | +0.4269 |
| imb5 | -0.1818 | -0.0910 | -0.2683 | -0.2153 | -0.1388 | -0.1336 | -0.1767 | -0.1271 |
| micro | +0.6112 | -0.0160 | -0.4125 | -0.3174 | -0.2162 | -0.2038 | -0.1772 | -0.1273 |
| cnt_imb1 | +1.5440 | +1.4718 | +1.4051 | +1.2851 | +1.3017 | +1.2352 | +1.2333 | +1.2332 |
| depth1_log | +0.2877 | +0.1005 | +0.0325 | +0.0167 | +0.0538 | +0.0487 | +0.0452 | +0.0150 |
| flow_ev1 | +0.0401 | +0.0296 | +0.0242 | +0.0201 | +0.0231 | +0.0222 | +0.0212 | +0.0207 |
| flow_ev10 | +0.1850 | +0.1504 | +0.1463 | +0.1228 | +0.1293 | +0.1276 | +0.1251 | +0.1200 |
| flow_ev100 | -0.0635 | +0.0262 | +0.0732 | +0.1515 | +0.1750 | +0.2006 | +0.1705 | +0.1601 |
| flow_ms10 | -0.0106 | +0.0006 | +0.0016 | +0.0089 | +0.0055 | +0.0030 | +0.0101 | +0.0128 |
| flow_ms100 | -0.1714 | -0.1410 | -0.0749 | -0.0099 | +0.0250 | +0.0563 | +0.0370 | +0.0270 |
| flow_ms1000 | +0.0135 | -0.1679 | -0.3198 | -0.6223 | -0.6884 | -0.7872 | -0.6826 | -0.6147 |
| flow_ms10000 | -0.3199 | -0.2458 | -0.3184 | -0.2040 | -0.2053 | -0.1586 | -0.1503 | -0.1750 |
| ntr_ms1000 | -0.0297 | +0.0693 | -0.1371 | -0.0250 | -0.0141 | +0.0171 | +0.0055 | -0.0610 |
| nev_ms1000 | +0.8995 | +0.1887 | -0.0167 | -0.0113 | +0.0221 | -0.0468 | +0.0012 | +0.0117 |
| ret_ms100 | +1.5596 | +1.2393 | +1.1239 | +0.9274 | +0.8508 | +0.7606 | +0.8211 | +0.8230 |
| ret_ms1000 | +0.7865 | +1.5889 | +1.5828 | +1.8130 | +1.8863 | +2.0113 | +1.8800 | +1.7698 |
| ret_ms10000 | +0.1753 | -0.0777 | +0.2217 | +0.0570 | +0.0486 | -0.0120 | +0.0103 | +0.0766 |
| last_dmid | +0.7156 | +0.7205 | +0.8845 | +0.7162 | +0.7471 | +0.6984 | +0.6577 | +0.6638 |
| dt_mid_log | -0.1830 | -0.0679 | -0.0021 | +0.0130 | +0.0082 | +0.0119 | -0.0024 | +0.0010 |
| ltp_dev | -0.1133 | +0.0306 | +0.0824 | +0.1144 | +0.0926 | +0.1076 | +0.0937 | +0.1061 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| ret_ms1000 | +0.26931 |
| cnt_imb1 | +0.04571 |
| ret_ms100 | +0.04531 |
| last_dmid | +0.02846 |
| ntr_ms1000 | +0.02022 |
| nev_ms1000 | +0.01211 |
| flow_ms100 | +0.00983 |
| imb3 | +0.00975 |
| flow_ms1000 | +0.00852 |
| ret_ms10000 | +0.00616 |
| flow_ev100 | +0.00525 |
| dt_mid_log | +0.00498 |
| flow_ev10 | +0.00218 |
| ltp_dev | +0.00204 |
| micro | +0.00158 |
| imb1 | +0.00153 |
| imb5 | +0.00111 |
| imb2 | +0.00046 |
| flow_ms10 | +0.00039 |
| flow_ev1 | +0.00000 |
| depth1_log | -0.00162 |
| spread | -0.00298 |
| flow_ms10000 | -0.00877 |

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
