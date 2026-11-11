# Data card - `..\AlgoTradeArena\data\round4_public.csv`

Generated 2026-11-08 07:06 by `research.profile` from `data\features\round4_public.npz` (features built in 84.3 s). Prices in ticks.

## Observations

- Spread is 1 tick only 81.0% of the time (median 1 ticks) -> room to quote inside the spread (inserts improving the touch: 1.7%).
- Mid changes trend at the tick level (P(reversal) = 0.29): a move tends to be followed by another move the same way.
- Variance ratio 60 s vs 1 s = 1.28 > 1: prices trend over a minute; inventory against the trend is dangerous, momentum may pay.
- Most reliable simple signal: `imb_l1` at 10 ms, mean rank IC +0.109 (t +71.4), same sign in 8/8 chunks; the whole move it predicts has std 0.38 ticks at 10 ms, vs. 0.5 tick(s) half-spread paid by an IOC.
- 1 s mid-change std is 6.26 ticks vs a typical spread of 1 tick(s): moves are large relative to the spread.

## Tape

| metric | value |
|---|---:|
| rows | 2,398,520 |
| duration | 6,000.0 s (1.67 h), Time 0.006 .. 5999.999 |
| events / s (mean) | 400 |
| rows sharing the previous row's timestamp | 0.0% |
| insert limit (G) / insert IOC (F) / cancel | 48.8% / 2.4% / 48.8% |
| cancels of unknown ids | 0 |
| limit inserts bid / ask (count) | 50.0% / 50.0% |
| IOC buy / sell (count) | 29,125 / 29,320 |
| IOC buy / sell (volume) | 4,744,608 / 4,606,002 |
| traded volume buyer- / seller-initiated | 3,863,672 / 3,843,622 |
| insert price range (ticks) | 9,146 .. 10,039 |

**Order sizes (lots)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| limit (G) | 1,170,078 | 74.3 | 1 | 5 | 18 | 54 | 277 | 884 | 74,020 |
| IOC (F) | 58,445 | 160 | 6 | 24 | 63 | 160 | 594 | 1,577 | 14,395 |

**Where limit orders are placed** (vs. the book just before the insert)

| placement | share |
|---|---:|
| crosses the spread (marketable) | 0.0% |
| own side empty | 0.0% |
| inside the spread (improves) | 1.7% |
| at the touch | 43.9% |
| 1-2 ticks behind | 41.0% |
| 3-5 ticks behind | 10.1% |
| 6-19 ticks behind | 3.3% |
| 20+ ticks behind (invisible) | 0.0% |

IOC orders: 96.2% trade anything, 21.7% move the mid, 3.59 fills per IOC on average.

**Order lifetime, insert -> cancel (ms)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cancelled orders | 1,169,997 | 535 | 47.7 | 151 | 310 | 603 | 1,507 | 3,764 | 50,597 |

100.0% of limit orders are eventually cancelled (the rest fill, or live to the end); 0.0% of cancels come within 5 ms of the insert (faster than our latency).

**Activity by 5-minute bucket**

| minute | events | events/s | IOC | fills | traded vol | mid low | mid high | mid end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 120,280 | 400.9 | 2,931 | 10,087 | 374,757 | 9,759 | 10,024 | 9,778 |
| 5 | 119,397 | 398.0 | 2,792 | 10,343 | 385,790 | 9,750 | 9,908 | 9,804 |
| 10 | 115,122 | 383.7 | 2,776 | 9,520 | 356,320 | 9,774 | 9,944 | 9,774 |
| 15 | 120,353 | 401.2 | 2,928 | 10,232 | 368,542 | 9,698 | 9,876 | 9,760 |
| 20 | 120,427 | 401.4 | 2,701 | 9,698 | 363,421 | 9,648 | 9,886 | 9,840 |
| 25 | 121,182 | 403.9 | 3,168 | 11,670 | 428,365 | 9,760 | 9,942 | 9,782 |
| 30 | 128,866 | 429.6 | 3,376 | 12,171 | 433,960 | 9,438 | 9,856 | 9,478 |
| 35 | 121,093 | 403.6 | 3,123 | 10,962 | 401,084 | 9,472 | 9,674 | 9,605 |
| 40 | 114,372 | 381.2 | 2,529 | 9,228 | 345,223 | 9,588 | 9,696 | 9,694 |
| 45 | 117,637 | 392.1 | 2,723 | 9,754 | 357,118 | 9,522 | 9,704 | 9,586 |
| 50 | 118,733 | 395.8 | 2,806 | 10,625 | 390,136 | 9,478 | 9,656 | 9,490 |
| 55 | 122,127 | 407.1 | 3,129 | 11,293 | 395,329 | 9,411 | 9,580 | 9,526 |
| 60 | 124,523 | 415.1 | 3,240 | 11,588 | 425,209 | 9,329 | 9,540 | 9,334 |
| 65 | 124,253 | 414.2 | 3,124 | 11,236 | 408,487 | 9,245 | 9,392 | 9,298 |
| 70 | 115,987 | 386.6 | 2,783 | 9,817 | 359,750 | 9,186 | 9,318 | 9,222 |
| 75 | 117,117 | 390.4 | 2,578 | 9,547 | 354,069 | 9,212 | 9,350 | 9,276 |
| 80 | 115,631 | 385.4 | 2,717 | 9,994 | 369,948 | 9,268 | 9,432 | 9,400 |
| 85 | 122,771 | 409.2 | 3,095 | 10,818 | 403,397 | 9,287 | 9,571 | 9,534 |
| 90 | 120,612 | 402.0 | 3,130 | 11,076 | 407,610 | 9,482 | 9,640 | 9,632 |
| 95 | 118,037 | 393.5 | 2,796 | 9,996 | 378,779 | 9,442 | 9,666 | 9,555 |

## Book

Two-sided book after 100.0% of events.

**Spread (ticks)**

| spread | share of events | share of time |
|---|---:|---:|
| 1 | 81.2% | 81.0% |
| 2 | 14.8% | 14.5% |
| 3 | 3.1% | 3.1% |
| 4 | 0.5% | 0.8% |
| 5+ | 0.3% | 0.5% |

**Touch size and depth (lots / orders, two-sided events)**

| quantity | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best bid qty | 2,397,597 | 916 | 13 | 125 | 494 | 1,232 | 3,049 | 5,920 | 57,451 |
| best ask qty | 2,397,597 | 949 | 13 | 129 | 501 | 1,246 | 3,196 | 6,332 | 75,502 |
| orders at best bid | 2,397,597 | 12 | 1 | 3 | 9 | 19 | 33 | 41 | 60 |
| orders at best ask | 2,397,597 | 12.3 | 1 | 3 | 9 | 19 | 33 | 41 | 65 |
| top-5 bid qty | 2,397,597 | 3,801 | 835 | 2,334 | 3,675 | 4,806 | 7,317 | 11,114 | 61,008 |
| top-5 ask qty | 2,397,597 | 3,855 | 838 | 2,335 | 3,665 | 4,862 | 7,643 | 11,567 | 77,151 |

**Mid changes**: after 1.7% of events, 6.65 per second; size 0.5 tick 82.3%, 1 tick 12.4%, 1.5-2 ticks 3.9%, more 1.4%.

**Mid-change volatility by horizon** (regular time grid; VR = variance ratio vs. 1 s, <1 mean reversion, >1 trending)

| horizon | samples | std (ticks) | mean abs | P(no change) | std / sqrt(s) | VR vs 1 s |
|---|---:|---:|---:|---:|---:|---:|
| 10 ms | 599,997 | 0.422 | 0.043 | 94.4% | 4.217 | 0.45 |
| 100 ms | 59,999 | 1.532 | 0.363 | 69.9% | 4.844 | 0.60 |
| 1,000 ms | 5,999 | 6.263 | 2.745 | 25.1% | 6.263 | 1.00 |
| 10,000 ms | 599 | 27.593 | 15.263 | 4.3% | 8.726 | 1.94 |
| 60,000 ms | 99 | 54.889 | 38.909 | 1.0% | 7.086 | 1.28 |

**Successive mid changes** (39,870): autocorrelation lag 1..5 = +0.322, +0.081, +0.081, +0.079, +0.048; P(next change reverses) = 0.288 (0.5 = random walk, >0.5 bounce / mean reversion).

**Trade-sign autocorrelation** (56,234 trading events): lag 1 +0.131, lag 2 +0.114, lag 5 +0.074, lag 10 +0.052, lag 50 -0.001; buyer-initiated share 49.7%.

## Signals vs. future mid change

Label: mid(t + 5 ms + h) - mid(t + 5 ms), mid = last two-sided mid of the book after all events before that time (what an order sent at t would meet). 2,393,122 events with a two-sided book and a known label, cut into 8 equal-event chunks. Signals use rows <= t only.

Label std (ticks): 10 ms 0.380 (6.0% non-zero), 100 ms 1.253 (31.5% non-zero), 1000 ms 5.445 (75.9% non-zero), 10000 ms 27.956 (95.2% non-zero).

Signals: `imb_l1` = (b1-a1)/(b1+a1); `imb_top5` same over 5 price levels; `micro_minus_mid` = microprice - mid (ticks); `flow_Nev` = signed traded volume (+ = buyer-initiated) of the last N events; `flow_100ms` / `flow_1s` = over the last T.

**Mean Spearman IC over chunks** (t-stat across chunks, chunks agreeing in sign / chunks)

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.109 (t +71.4, 8/8) | +0.212 (t +26.1, 8/8) | +0.109 (t +12.1, 8/8) | +0.033 (t +7.6, 8/8) |
| `imb_top5` | +0.082 (t +26.9, 8/8) | +0.173 (t +21.2, 8/8) | +0.137 (t +13.2, 8/8) | +0.054 (t +2.7, 7/8) |
| `micro_minus_mid` | +0.105 (t +53.1, 8/8) | +0.195 (t +21.5, 8/8) | +0.098 (t +10.7, 8/8) | +0.030 (t +7.6, 8/8) |
| `flow_10ev` | +0.097 (t +26.6, 8/8) | +0.156 (t +29.1, 8/8) | +0.137 (t +38.8, 8/8) | +0.041 (t +12.0, 8/8) |
| `flow_100ev` | +0.151 (t +29.3, 8/8) | +0.343 (t +35.0, 8/8) | +0.326 (t +40.1, 8/8) | +0.092 (t +9.1, 8/8) |
| `flow_100ms` | +0.138 (t +24.3, 8/8) | +0.300 (t +24.9, 8/8) | +0.288 (t +37.1, 8/8) | +0.084 (t +11.0, 8/8) |
| `flow_1s` | +0.139 (t +28.7, 8/8) | +0.321 (t +28.9, 8/8) | +0.322 (t +18.3, 8/8) | +0.088 (t +4.9, 8/8) |

**Pooled-chunk mean Pearson IC**

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.059 | +0.117 | +0.082 | +0.026 |
| `imb_top5` | +0.068 | +0.146 | +0.174 | +0.076 |
| `micro_minus_mid` | +0.039 | +0.089 | +0.061 | +0.021 |
| `flow_10ev` | +0.065 | +0.088 | +0.087 | +0.021 |
| `flow_100ev` | +0.087 | +0.203 | +0.221 | +0.054 |
| `flow_100ms` | +0.079 | +0.177 | +0.197 | +0.049 |
| `flow_1s` | +0.087 | +0.216 | +0.282 | +0.062 |

**Spearman IC per chunk, h = 10 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.109 | +0.112 | +0.113 | +0.109 | +0.105 | +0.102 | +0.114 | +0.105 |
| `imb_top5` | +0.073 | +0.088 | +0.095 | +0.072 | +0.086 | +0.088 | +0.074 | +0.080 |
| `micro_minus_mid` | +0.106 | +0.113 | +0.105 | +0.107 | +0.101 | +0.096 | +0.111 | +0.102 |
| `flow_10ev` | +0.094 | +0.097 | +0.113 | +0.094 | +0.113 | +0.094 | +0.085 | +0.090 |
| `flow_100ev` | +0.144 | +0.155 | +0.172 | +0.134 | +0.171 | +0.153 | +0.137 | +0.143 |
| `flow_100ms` | +0.133 | +0.137 | +0.163 | +0.119 | +0.159 | +0.140 | +0.121 | +0.131 |
| `flow_1s` | +0.145 | +0.159 | +0.144 | +0.112 | +0.148 | +0.137 | +0.134 | +0.133 |

**Spearman IC per chunk, h = 100 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.233 | +0.233 | +0.189 | +0.220 | +0.186 | +0.182 | +0.236 | +0.219 |
| `imb_top5` | +0.146 | +0.194 | +0.204 | +0.142 | +0.177 | +0.190 | +0.154 | +0.176 |
| `micro_minus_mid` | +0.217 | +0.219 | +0.165 | +0.205 | +0.169 | +0.161 | +0.220 | +0.205 |
| `flow_10ev` | +0.151 | +0.150 | +0.181 | +0.140 | +0.173 | +0.165 | +0.138 | +0.152 |
| `flow_100ev` | +0.330 | +0.348 | +0.387 | +0.303 | +0.374 | +0.351 | +0.319 | +0.335 |
| `flow_100ms` | +0.283 | +0.296 | +0.360 | +0.252 | +0.332 | +0.311 | +0.272 | +0.293 |
| `flow_1s` | +0.321 | +0.368 | +0.339 | +0.256 | +0.325 | +0.317 | +0.319 | +0.325 |

**Spearman IC per chunk, h = 1000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.132 | +0.156 | +0.108 | +0.089 | +0.079 | +0.102 | +0.119 | +0.089 |
| `imb_top5` | +0.108 | +0.180 | +0.165 | +0.101 | +0.131 | +0.159 | +0.110 | +0.140 |
| `micro_minus_mid` | +0.125 | +0.144 | +0.092 | +0.075 | +0.069 | +0.094 | +0.108 | +0.078 |
| `flow_10ev` | +0.139 | +0.153 | +0.148 | +0.121 | +0.135 | +0.132 | +0.131 | +0.139 |
| `flow_100ev` | +0.341 | +0.368 | +0.331 | +0.289 | +0.319 | +0.313 | +0.322 | +0.322 |
| `flow_100ms` | +0.292 | +0.320 | +0.310 | +0.247 | +0.287 | +0.278 | +0.279 | +0.291 |
| `flow_1s` | +0.338 | +0.419 | +0.315 | +0.249 | +0.318 | +0.278 | +0.315 | +0.341 |

**Spearman IC per chunk, h = 10000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.029 | +0.043 | +0.033 | +0.049 | +0.034 | +0.017 | +0.044 | +0.015 |
| `imb_top5` | -0.012 | +0.049 | +0.059 | +0.024 | +0.010 | +0.042 | +0.173 | +0.083 |
| `micro_minus_mid` | +0.024 | +0.037 | +0.030 | +0.045 | +0.027 | +0.017 | +0.041 | +0.015 |
| `flow_10ev` | +0.033 | +0.042 | +0.048 | +0.032 | +0.025 | +0.045 | +0.050 | +0.050 |
| `flow_100ev` | +0.078 | +0.099 | +0.087 | +0.056 | +0.053 | +0.117 | +0.128 | +0.120 |
| `flow_100ms` | +0.070 | +0.080 | +0.098 | +0.056 | +0.056 | +0.103 | +0.103 | +0.109 |
| `flow_1s` | +0.038 | +0.101 | +0.071 | +0.058 | +0.013 | +0.128 | +0.137 | +0.154 |
