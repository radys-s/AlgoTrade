# Signals - `data\round3_public.csv`, horizon 100 ms

Label: mid change from t + 5 ms to t + 5 ms + 100 ms (ticks). 2,408,176 rows (tape events with stride 1). Walk-forward over equal-event chunks: train on the chunks before, test on the next, training rows whose label reaches into the test chunk purged.
ic/rank = Pearson/Spearman of prediction vs label on the test chunk; r2 vs a zero forecast; edge10/edge1 = mean sign(pred)*label (ticks) when |pred| is in the top 10% / 1% (train thresholds).

| fold | model | n_test | ic | rank_ic | r2 | edge10 | edge1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | 301,511 | +0.3248 | +0.3945 | +0.10499 | +0.701 | +1.162 |
| 1 | hgb | 301,511 | +0.3094 | +0.3983 | +0.07585 | +0.652 | +1.071 |
| 2 | ridge | 301,524 | +0.3820 | +0.4053 | +0.14400 | +0.634 | +0.924 |
| 2 | hgb | 301,524 | +0.3504 | +0.4147 | +0.06595 | +0.643 | +0.668 |
| 3 | ridge | 301,563 | +0.3192 | +0.4512 | +0.10028 | +0.786 | +1.102 |
| 3 | hgb | 301,563 | +0.3429 | +0.4712 | +0.11721 | +0.802 | +1.362 |
| 4 | ridge | 301,539 | +0.3013 | +0.3992 | +0.09000 | +0.722 | +0.944 |
| 4 | hgb | 301,539 | +0.3475 | +0.4277 | +0.11925 | +0.796 | +1.113 |
| 5 | ridge | 301,483 | +0.2647 | +0.4383 | +0.06795 | +0.968 | +1.358 |
| 5 | hgb | 301,483 | +0.3071 | +0.4722 | +0.09285 | +0.990 | +1.781 |
| 6 | ridge | 301,504 | +0.3008 | +0.3891 | +0.08673 | +0.705 | +0.891 |
| 6 | hgb | 301,504 | +0.3401 | +0.4064 | +0.11232 | +0.832 | +1.099 |
| 7 | ridge | 301,490 | +0.3985 | +0.4426 | +0.15871 | +0.732 | +1.090 |
| 7 | hgb | 301,490 | +0.4361 | +0.4718 | +0.18894 | +0.789 | +1.207 |
| **mean** | ridge | | +0.3273 | +0.4172 | +0.10752 | +0.750 | +1.067 |
| **mean** | hgb | | +0.3476 | +0.4375 | +0.11034 | +0.786 | +1.186 |

## Ridge coefficients (standardized) per fold

| feature | 1 | 2 | 3 | 4 | 5 | 6 | 7 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | -0.0272 | -0.0123 | -0.0106 | -0.0111 | +0.0020 | +0.0066 | +0.0055 | +0.0024 |
| imb1 | -0.0373 | -0.0290 | -0.0225 | -0.0117 | -0.0109 | -0.0067 | -0.0061 | -0.0089 |
| imb2 | +0.0414 | +0.0376 | +0.0368 | +0.0320 | +0.0303 | +0.0228 | +0.0242 | +0.0257 |
| imb3 | +0.0010 | -0.0056 | +0.0088 | +0.0278 | +0.0308 | +0.0334 | +0.0335 | +0.0299 |
| imb5 | +0.0252 | +0.0300 | +0.0104 | -0.0055 | -0.0069 | -0.0024 | -0.0093 | -0.0051 |
| micro | +0.0154 | -0.0109 | -0.0183 | -0.0345 | -0.0298 | -0.0354 | -0.0296 | -0.0268 |
| cnt_imb1 | +0.1672 | +0.1859 | +0.1791 | +0.1894 | +0.1847 | +0.1899 | +0.1836 | +0.1852 |
| depth1_log | -0.0045 | -0.0175 | -0.0052 | +0.0082 | -0.0003 | -0.0005 | -0.0003 | -0.0006 |
| flow_ev1 | +0.0044 | +0.0061 | +0.0052 | +0.0053 | +0.0055 | +0.0065 | +0.0066 | +0.0067 |
| flow_ev10 | +0.0161 | +0.0204 | +0.0157 | +0.0148 | +0.0140 | +0.0129 | +0.0140 | +0.0158 |
| flow_ev100 | +0.0333 | +0.0209 | +0.0279 | +0.0357 | +0.0382 | +0.0404 | +0.0375 | +0.0388 |
| flow_ms10 | +0.0048 | +0.0035 | +0.0052 | +0.0069 | +0.0071 | +0.0106 | +0.0102 | +0.0102 |
| flow_ms100 | -0.0127 | +0.0073 | +0.0051 | +0.0182 | +0.0193 | +0.0287 | +0.0226 | +0.0200 |
| flow_ms1000 | -0.0032 | -0.0035 | -0.0020 | -0.0036 | +0.0010 | -0.0075 | -0.0077 | -0.0053 |
| flow_ms10000 | -0.0170 | -0.0171 | -0.0167 | -0.0175 | -0.0173 | -0.0164 | -0.0149 | -0.0149 |
| ntr_ms1000 | +0.0094 | +0.0078 | +0.0031 | +0.0102 | +0.0083 | +0.0140 | +0.0143 | +0.0120 |
| nev_ms1000 | -0.0042 | -0.0272 | -0.0230 | -0.0020 | -0.0149 | -0.0133 | -0.0064 | -0.0049 |
| ret_ms100 | +0.1441 | +0.1377 | +0.1292 | +0.1278 | +0.1227 | +0.1209 | +0.1261 | +0.1246 |
| ret_ms1000 | -0.0037 | +0.0113 | +0.0183 | +0.0275 | +0.0186 | +0.0261 | +0.0225 | +0.0262 |
| ret_ms10000 | +0.0058 | -0.0021 | -0.0007 | +0.0004 | +0.0015 | +0.0004 | +0.0018 | +0.0001 |
| last_dmid | +0.0592 | +0.0743 | +0.0661 | +0.0778 | +0.0773 | +0.0883 | +0.0853 | +0.0869 |
| dt_mid_log | -0.0065 | -0.0132 | -0.0096 | -0.0039 | -0.0012 | +0.0036 | +0.0053 | +0.0044 |
| ltp_dev | +0.0443 | +0.0440 | +0.0480 | +0.0462 | +0.0444 | +0.0513 | +0.0504 | +0.0499 |

## HGB permutation importance (drop in IC, last fold)

| feature | importance |
|---|---:|
| cnt_imb1 | +0.11847 |
| flow_ev100 | +0.04878 |
| last_dmid | +0.02344 |
| flow_ms100 | +0.01653 |
| ltp_dev | +0.01380 |
| ret_ms100 | +0.00924 |
| nev_ms1000 | +0.00851 |
| depth1_log | +0.00841 |
| imb2 | +0.00719 |
| flow_ms1000 | +0.00640 |
| dt_mid_log | +0.00554 |
| ntr_ms1000 | +0.00544 |
| ret_ms1000 | +0.00455 |
| imb3 | +0.00210 |
| imb1 | +0.00115 |
| flow_ev10 | +0.00101 |
| spread | +0.00092 |
| flow_ms10000 | +0.00083 |
| micro | +0.00079 |
| imb5 | +0.00044 |
| flow_ms10 | +0.00016 |
| flow_ev1 | +0.00001 |
| ret_ms10000 | -0.00054 |

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
