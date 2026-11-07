# Signals - `data\sample_v2.csv`, horizon 100 ms

Label: mid change from t + 5 ms to t + 5 ms + 100 ms (ticks). 1,996,592 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 250,000 | +0.3632 | +0.3521 | +0.13179 | +0.140 | +0.188 |
| 1 | hgb | 250,000 | +0.3693 | +0.3451 | +0.11840 | +0.138 | +0.259 |
| 2 | ridge | 250,000 | +0.3649 | +0.3515 | +0.13312 | +0.150 | +0.201 |
| 2 | hgb | 250,000 | +0.3713 | +0.3568 | +0.12879 | +0.145 | +0.224 |
| 3 | ridge | 249,998 | +0.3541 | +0.3387 | +0.12479 | +0.143 | +0.224 |
| 3 | hgb | 249,998 | +0.3648 | +0.3445 | +0.12126 | +0.137 | +0.198 |
| 4 | ridge | 249,996 | +0.3760 | +0.3597 | +0.14075 | +0.161 | +0.290 |
| 4 | hgb | 249,996 | +0.4038 | +0.3673 | +0.16295 | +0.167 | +0.311 |
| 5 | ridge | 249,996 | +0.3623 | +0.3481 | +0.13116 | +0.148 | +0.275 |
| 5 | hgb | 249,996 | +0.3969 | +0.3558 | +0.15745 | +0.164 | +0.317 |
| 6 | ridge | 250,000 | +0.3588 | +0.3477 | +0.12877 | +0.142 | +0.226 |
| 6 | hgb | 250,000 | +0.3922 | +0.3626 | +0.15342 | +0.163 | +0.270 |
| 7 | ridge | 249,955 | +0.3610 | +0.3429 | +0.13011 | +0.149 | +0.285 |
| 7 | hgb | 249,955 | +0.3925 | +0.3538 | +0.15323 | +0.163 | +0.256 |
| **mean** | ridge | | +0.3629 | +0.3487 | +0.13150 | +0.147 | +0.241 |
| **mean** | hgb | | +0.3844 | +0.3551 | +0.14221 | +0.154 | +0.262 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | -0.0039 | -0.0038 | -0.0016 | -0.0005 | -0.0004 | -0.0020 | -0.0012 | -0.0000 |
| imb1 | +0.0551 | +0.0541 | +0.0418 | +0.0265 | +0.0266 | +0.0246 | +0.0258 | +0.0187 |
| imb2 | +0.0328 | +0.0180 | +0.0220 | +0.0215 | +0.0215 | +0.0236 | +0.0242 | +0.0248 |
| imb3 | +0.0302 | +0.0249 | -0.0217 | -0.0069 | -0.0118 | -0.0126 | -0.0098 | -0.0199 |
| imb5 | -0.0607 | -0.0405 | +0.0036 | -0.0104 | -0.0065 | -0.0080 | -0.0111 | -0.0017 |
| micro | +0.0412 | +0.0378 | +0.0381 | +0.0386 | +0.0403 | +0.0401 | +0.0411 | +0.0409 |
| cnt_imb1 | -0.0343 | -0.0276 | -0.0160 | -0.0023 | -0.0022 | -0.0005 | -0.0030 | +0.0042 |
| depth1_log | +0.0046 | +0.0033 | +0.0021 | +0.0004 | +0.0001 | +0.0021 | +0.0011 | +0.0001 |
| flow_ev1 | -0.0004 | -0.0002 | -0.0003 | -0.0003 | -0.0004 | -0.0003 | -0.0004 | -0.0004 |
| flow_ev10 | -0.0027 | -0.0025 | -0.0026 | -0.0032 | -0.0034 | -0.0033 | -0.0031 | -0.0031 |
| flow_ev100 | -0.0069 | -0.0070 | -0.0079 | -0.0075 | -0.0080 | -0.0081 | -0.0079 | -0.0079 |
| flow_ms10 | -0.0017 | -0.0012 | -0.0008 | -0.0006 | -0.0008 | -0.0008 | -0.0008 | -0.0008 |
| flow_ms100 | -0.0057 | -0.0065 | -0.0065 | -0.0062 | -0.0053 | -0.0048 | -0.0047 | -0.0047 |
| flow_ms1000 | -0.0076 | -0.0072 | -0.0084 | -0.0083 | -0.0085 | -0.0087 | -0.0086 | -0.0085 |
| flow_ms10000 | -0.0017 | -0.0016 | -0.0008 | -0.0013 | -0.0003 | -0.0004 | -0.0005 | -0.0004 |
| ntr_ms1000 | +0.0002 | +0.0010 | -0.0013 | -0.0011 | -0.0009 | -0.0006 | -0.0007 | -0.0003 |
| nev_ms1000 | +0.0008 | -0.0007 | -0.0006 | -0.0009 | -0.0006 | -0.0004 | -0.0002 | +0.0002 |
| ret_ms100 | +0.0072 | +0.0078 | +0.0086 | +0.0083 | +0.0086 | +0.0087 | +0.0088 | +0.0083 |
| ret_ms1000 | +0.0091 | +0.0083 | +0.0084 | +0.0082 | +0.0095 | +0.0101 | +0.0097 | +0.0097 |
| ret_ms10000 | +0.0013 | +0.0010 | +0.0002 | +0.0006 | +0.0010 | +0.0008 | +0.0010 | +0.0007 |
| last_dmid | +0.0089 | +0.0079 | +0.0073 | +0.0078 | +0.0081 | +0.0077 | +0.0076 | +0.0078 |
| dt_mid_log | -0.0002 | -0.0017 | -0.0018 | -0.0022 | -0.0020 | -0.0018 | -0.0015 | -0.0011 |
| ltp_dev | -0.0003 | +0.0003 | +0.0004 | +0.0010 | +0.0016 | +0.0018 | +0.0017 | +0.0017 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| micro | +0.21693 |
| cnt_imb1 | +0.02273 |
| imb1 | +0.01818 |
| dt_mid_log | +0.01685 |
| ltp_dev | +0.00593 |
| last_dmid | +0.00559 |
| depth1_log | +0.00452 |
| flow_ev10 | +0.00418 |
| flow_ev100 | +0.00315 |
| flow_ms1000 | +0.00279 |
| flow_ms100 | +0.00276 |
| ret_ms1000 | +0.00114 |
| spread | +0.00108 |
| nev_ms1000 | +0.00088 |
| imb2 | +0.00076 |
| flow_ms10000 | +0.00035 |
| flow_ms10 | +0.00032 |
| imb5 | +0.00025 |
| ret_ms100 | +0.00013 |
| flow_ev1 | +0.00000 |
| ret_ms10000 | -0.00012 |
| ntr_ms1000 | -0.00056 |
| imb3 | -0.00065 |

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
