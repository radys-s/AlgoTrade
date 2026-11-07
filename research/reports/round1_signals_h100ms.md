# Signals - `data\round1.csv`, horizon 100 ms

Label: mid change from t + 5 ms to t + 5 ms + 100 ms (ticks). 2,203,598 rows (tape events with stride 2). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 275,685 | +0.3816 | +0.3561 | +0.14563 | +0.286 | +0.461 |
| 2 | ridge | 275,674 | +0.3817 | +0.3541 | +0.14540 | +0.301 | +0.420 |
| 3 | ridge | 275,674 | +0.3867 | +0.3585 | +0.14931 | +0.318 | +0.511 |
| 4 | ridge | 275,661 | +0.3759 | +0.3479 | +0.14121 | +0.288 | +0.401 |
| 5 | ridge | 275,670 | +0.3846 | +0.3575 | +0.14778 | +0.304 | +0.466 |
| 6 | ridge | 275,670 | +0.3717 | +0.3485 | +0.13769 | +0.295 | +0.423 |
| 7 | ridge | 275,628 | +0.3789 | +0.3519 | +0.14359 | +0.298 | +0.457 |
| **mean** | ridge | | +0.3802 | +0.3535 | +0.14437 | +0.299 | +0.448 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | +0.0044 | +0.0038 | +0.0034 | +0.0025 | +0.0020 | +0.0018 | +0.0013 | +0.0010 |
| imb1 | +0.0204 | +0.0234 | +0.0208 | +0.0214 | +0.0249 | +0.0263 | +0.0263 | +0.0256 |
| imb2 | +0.0414 | +0.0445 | +0.0411 | +0.0408 | +0.0382 | +0.0378 | +0.0380 | +0.0367 |
| imb3 | +0.0036 | -0.0147 | -0.0119 | -0.0038 | +0.0037 | +0.0013 | +0.0065 | +0.0079 |
| imb5 | -0.0422 | -0.0284 | -0.0278 | -0.0367 | -0.0426 | -0.0405 | -0.0458 | -0.0458 |
| micro | +0.0014 | -0.0025 | -0.0002 | +0.0010 | -0.0013 | -0.0023 | -0.0029 | -0.0021 |
| cnt_imb1 | +0.0721 | +0.0742 | +0.0738 | +0.0729 | +0.0715 | +0.0710 | +0.0705 | +0.0706 |
| depth1_log | -0.0024 | -0.0007 | -0.0004 | -0.0003 | +0.0002 | +0.0007 | +0.0007 | +0.0004 |
| flow_ev1 | +0.0020 | +0.0020 | +0.0020 | +0.0019 | +0.0018 | +0.0016 | +0.0016 | +0.0015 |
| flow_ev10 | +0.0050 | +0.0050 | +0.0036 | +0.0038 | +0.0040 | +0.0042 | +0.0042 | +0.0041 |
| flow_ev100 | +0.0214 | +0.0197 | +0.0182 | +0.0184 | +0.0168 | +0.0176 | +0.0173 | +0.0179 |
| flow_ms10 | -0.0007 | +0.0006 | +0.0011 | +0.0014 | +0.0015 | +0.0014 | +0.0015 | +0.0014 |
| flow_ms100 | -0.0022 | -0.0056 | -0.0036 | -0.0023 | -0.0019 | -0.0014 | -0.0015 | -0.0013 |
| flow_ms1000 | +0.0249 | +0.0282 | +0.0275 | +0.0281 | +0.0305 | +0.0301 | +0.0294 | +0.0282 |
| flow_ms10000 | +0.0042 | +0.0043 | +0.0019 | +0.0027 | +0.0021 | +0.0031 | +0.0032 | +0.0035 |
| ntr_ms1000 | -0.0045 | -0.0024 | -0.0022 | -0.0018 | -0.0008 | -0.0006 | -0.0007 | -0.0007 |
| nev_ms1000 | +0.0005 | +0.0005 | +0.0017 | +0.0014 | +0.0002 | +0.0003 | +0.0002 | +0.0002 |
| ret_ms100 | +0.0142 | +0.0165 | +0.0150 | +0.0150 | +0.0144 | +0.0139 | +0.0142 | +0.0141 |
| ret_ms1000 | -0.0046 | -0.0102 | -0.0098 | -0.0113 | -0.0132 | -0.0134 | -0.0128 | -0.0117 |
| ret_ms10000 | -0.0042 | -0.0052 | -0.0023 | -0.0034 | -0.0028 | -0.0039 | -0.0037 | -0.0036 |
| last_dmid | +0.0126 | +0.0146 | +0.0139 | +0.0148 | +0.0151 | +0.0147 | +0.0135 | +0.0128 |
| dt_mid_log | -0.0046 | -0.0004 | +0.0000 | -0.0004 | -0.0007 | +0.0000 | -0.0000 | -0.0001 |
| ltp_dev | +0.0188 | +0.0205 | +0.0205 | +0.0201 | +0.0199 | +0.0202 | +0.0202 | +0.0202 |

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
