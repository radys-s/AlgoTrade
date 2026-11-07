# Signals - `data\sample_v2.csv`, horizon 1000 ms

Label: mid change from t + 5 ms to t + 5 ms + 1000 ms (ticks). 1,996,283 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 250,000 | +0.3949 | +0.3940 | +0.15344 | +0.363 | +0.481 |
| 1 | hgb | 250,000 | +0.3455 | +0.3456 | +0.08993 | +0.349 | +0.485 |
| 2 | ridge | 250,000 | +0.4252 | +0.4132 | +0.17885 | +0.446 | +0.557 |
| 2 | hgb | 250,000 | +0.4181 | +0.4045 | +0.16670 | +0.454 | +0.763 |
| 3 | ridge | 249,998 | +0.4169 | +0.3960 | +0.17390 | +0.434 | +0.579 |
| 3 | hgb | 249,998 | +0.4149 | +0.3850 | +0.16641 | +0.441 | +0.814 |
| 4 | ridge | 249,996 | +0.4094 | +0.3926 | +0.16781 | +0.449 | +0.633 |
| 4 | hgb | 249,996 | +0.4255 | +0.4153 | +0.17995 | +0.438 | +0.715 |
| 5 | ridge | 249,996 | +0.4048 | +0.3849 | +0.16299 | +0.425 | +0.671 |
| 5 | hgb | 249,996 | +0.4440 | +0.4209 | +0.19731 | +0.483 | +0.796 |
| 6 | ridge | 250,000 | +0.3949 | +0.3882 | +0.15556 | +0.380 | +0.561 |
| 6 | hgb | 250,000 | +0.4326 | +0.4206 | +0.18612 | +0.437 | +0.794 |
| 7 | ridge | 249,646 | +0.4076 | +0.3847 | +0.16640 | +0.455 | +0.743 |
| 7 | hgb | 249,646 | +0.4530 | +0.4200 | +0.20531 | +0.489 | +0.932 |
| **mean** | ridge | | +0.4077 | +0.3934 | +0.16556 | +0.422 | +0.604 |
| **mean** | hgb | | +0.4191 | +0.4017 | +0.17025 | +0.442 | +0.757 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | +0.0094 | -0.0110 | -0.0115 | -0.0093 | -0.0089 | -0.0108 | -0.0080 | -0.0017 |
| imb1 | +0.2148 | +0.1589 | +0.1469 | +0.1157 | +0.1020 | +0.0525 | +0.0270 | -0.0011 |
| imb2 | +0.0600 | +0.0219 | +0.0425 | +0.0706 | +0.0678 | +0.0787 | +0.0768 | +0.0843 |
| imb3 | +0.3097 | -0.0522 | -0.1345 | -0.0206 | -0.0128 | -0.0383 | -0.0300 | -0.0313 |
| imb5 | -0.3451 | +0.0760 | +0.1381 | -0.0108 | -0.0169 | -0.0087 | -0.0164 | -0.0258 |
| micro | +0.1631 | +0.1542 | +0.1727 | +0.1743 | +0.1821 | +0.1866 | +0.1931 | +0.2003 |
| cnt_imb1 | -0.1849 | -0.1168 | -0.1220 | -0.0946 | -0.0854 | -0.0397 | -0.0213 | -0.0007 |
| depth1_log | +0.0048 | +0.0065 | +0.0142 | +0.0125 | +0.0096 | +0.0130 | +0.0084 | +0.0022 |
| flow_ev1 | -0.0011 | -0.0010 | -0.0006 | -0.0005 | -0.0004 | -0.0004 | -0.0003 | -0.0003 |
| flow_ev10 | -0.0050 | -0.0045 | -0.0055 | -0.0054 | -0.0056 | -0.0054 | -0.0048 | -0.0049 |
| flow_ev100 | -0.0250 | -0.0297 | -0.0262 | -0.0252 | -0.0256 | -0.0256 | -0.0253 | -0.0240 |
| flow_ms10 | -0.0005 | -0.0006 | -0.0004 | -0.0005 | -0.0002 | +0.0002 | -0.0000 | -0.0001 |
| flow_ms100 | -0.0151 | -0.0221 | -0.0226 | -0.0225 | -0.0216 | -0.0220 | -0.0219 | -0.0219 |
| flow_ms1000 | -0.0425 | -0.0276 | -0.0370 | -0.0378 | -0.0396 | -0.0383 | -0.0352 | -0.0350 |
| flow_ms10000 | -0.0215 | -0.0210 | -0.0133 | -0.0088 | -0.0042 | -0.0071 | -0.0060 | -0.0048 |
| ntr_ms1000 | -0.0165 | -0.0055 | -0.0127 | -0.0095 | -0.0106 | -0.0071 | -0.0095 | -0.0101 |
| nev_ms1000 | -0.0120 | -0.0178 | -0.0194 | -0.0131 | -0.0085 | -0.0048 | -0.0062 | -0.0057 |
| ret_ms100 | +0.0412 | +0.0429 | +0.0431 | +0.0449 | +0.0459 | +0.0471 | +0.0463 | +0.0464 |
| ret_ms1000 | +0.0516 | +0.0273 | +0.0241 | +0.0272 | +0.0318 | +0.0350 | +0.0343 | +0.0343 |
| ret_ms10000 | -0.0159 | -0.0158 | -0.0100 | -0.0032 | -0.0028 | -0.0047 | -0.0012 | -0.0013 |
| last_dmid | +0.0447 | +0.0421 | +0.0485 | +0.0562 | +0.0526 | +0.0536 | +0.0543 | +0.0582 |
| dt_mid_log | +0.0117 | +0.0045 | -0.0015 | -0.0063 | -0.0059 | -0.0020 | -0.0028 | -0.0001 |
| ltp_dev | -0.0068 | -0.0055 | -0.0081 | -0.0072 | -0.0051 | -0.0047 | -0.0066 | -0.0063 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| micro | +0.18551 |
| last_dmid | +0.06412 |
| dt_mid_log | +0.03337 |
| cnt_imb1 | +0.03123 |
| spread | +0.02133 |
| imb2 | +0.02031 |
| depth1_log | +0.01565 |
| flow_ms1000 | +0.01088 |
| imb1 | +0.00687 |
| flow_ev100 | +0.00332 |
| ltp_dev | +0.00301 |
| flow_ms10000 | +0.00282 |
| ret_ms100 | +0.00141 |
| flow_ev10 | +0.00137 |
| flow_ms100 | +0.00086 |
| ntr_ms1000 | +0.00084 |
| imb5 | +0.00051 |
| ret_ms10000 | +0.00039 |
| flow_ev1 | +0.00000 |
| flow_ms10 | +0.00000 |
| nev_ms1000 | -0.00018 |
| imb3 | -0.00023 |
| ret_ms1000 | -0.00051 |

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
