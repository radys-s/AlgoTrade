# Data card - `data\round3_public.csv`

Generated 2026-11-08 01:03 by `research.profile` from `data\features\round3_public.npz` (features built in 35.5 s). Prices in ticks.

## Observations

- Spread is 1 tick only 81.6% of the time (median 1 ticks) -> room to quote inside the spread (inserts improving the touch: 1.5%).
- Mid changes trend at the tick level (P(reversal) = 0.33): a move tends to be followed by another move the same way.
- Most reliable simple signal: `imb_l1` at 10 ms, mean rank IC +0.100 (t +102.2), same sign in 8/8 chunks; the whole move it predicts has std 0.33 ticks at 10 ms, vs. 0.5 tick(s) half-spread paid by an IOC.
- 1 s mid-change std is 4.49 ticks vs a typical spread of 1 tick(s): moves are large relative to the spread.

## Tape

| metric | value |
|---|---:|
| rows | 2,412,805 |
| duration | 6,000.0 s (1.67 h), Time 0.001 .. 5999.994 |
| events / s (mean) | 402 |
| rows sharing the previous row's timestamp | 0.0% |
| insert limit (G) / insert IOC (F) / cancel | 48.8% / 2.3% / 48.8% |
| cancels of unknown ids | 0 |
| limit inserts bid / ask (count) | 50.0% / 50.0% |
| IOC buy / sell (count) | 28,161 / 28,113 |
| IOC buy / sell (volume) | 4,456,113 / 4,514,316 |
| traded volume buyer- / seller-initiated | 3,519,762 / 3,644,851 |
| insert price range (ticks) | 9,732 .. 10,160 |

**Order sizes (lots)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| limit (G) | 1,178,322 | 72.8 | 1 | 5 | 18 | 53 | 264 | 878 | 111,687 |
| IOC (F) | 56,274 | 159 | 6 | 23 | 60 | 154 | 590 | 1,528 | 50,114 |

**Where limit orders are placed** (vs. the book just before the insert)

| placement | share |
|---|---:|
| crosses the spread (marketable) | 0.0% |
| own side empty | 0.0% |
| inside the spread (improves) | 1.5% |
| at the touch | 43.6% |
| 1-2 ticks behind | 41.9% |
| 3-5 ticks behind | 9.9% |
| 6-19 ticks behind | 3.0% |
| 20+ ticks behind (invisible) | 0.0% |

IOC orders: 94.1% trade anything, 19.4% move the mid, 3.63 fills per IOC on average.

**Order lifetime, insert -> cancel (ms)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cancelled orders | 1,178,209 | 567 | 48.2 | 155 | 317 | 616 | 1,581 | 4,495 | 66,776 |

100.0% of limit orders are eventually cancelled (the rest fill, or live to the end); 0.0% of cancels come within 5 ms of the insert (faster than our latency).

**Activity by 5-minute bucket**

| minute | events | events/s | IOC | fills | traded vol | mid low | mid high | mid end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 115,777 | 385.9 | 2,497 | 9,195 | 342,283 | 9,956 | 10,004 | 9,974 |
| 5 | 115,270 | 384.2 | 2,545 | 9,285 | 340,197 | 9,896 | 9,986 | 9,904 |
| 10 | 127,015 | 423.4 | 3,160 | 11,326 | 408,831 | 9,904 | 10,028 | 9,960 |
| 15 | 118,294 | 394.3 | 2,846 | 10,136 | 345,534 | 9,958 | 10,034 | 10,034 |
| 20 | 117,861 | 392.9 | 2,633 | 9,665 | 329,733 | 10,030 | 10,114 | 10,104 |
| 25 | 116,377 | 387.9 | 2,650 | 9,431 | 342,901 | 10,078 | 10,132 | 10,098 |
| 30 | 121,469 | 404.9 | 2,805 | 10,048 | 341,642 | 9,945 | 10,100 | 9,974 |
| 35 | 124,160 | 413.9 | 3,059 | 10,694 | 377,628 | 9,866 | 9,986 | 9,866 |
| 40 | 123,879 | 412.9 | 2,909 | 10,158 | 354,774 | 9,767 | 9,920 | 9,778 |
| 45 | 118,563 | 395.2 | 2,641 | 9,792 | 354,025 | 9,766 | 9,828 | 9,784 |
| 50 | 118,571 | 395.2 | 2,675 | 9,768 | 337,725 | 9,774 | 9,843 | 9,816 |
| 55 | 122,602 | 408.7 | 2,879 | 10,253 | 356,994 | 9,799 | 9,892 | 9,875 |
| 60 | 125,825 | 419.4 | 2,975 | 11,039 | 380,333 | 9,869 | 9,934 | 9,880 |
| 65 | 123,412 | 411.4 | 3,128 | 10,881 | 379,749 | 9,832 | 10,103 | 10,082 |
| 70 | 119,154 | 397.2 | 2,578 | 9,755 | 326,399 | 9,930 | 10,082 | 9,976 |
| 75 | 118,514 | 395.0 | 2,663 | 9,790 | 343,489 | 9,904 | 9,980 | 9,904 |
| 80 | 118,053 | 393.5 | 2,804 | 10,561 | 371,649 | 9,882 | 9,984 | 9,940 |
| 85 | 116,863 | 389.5 | 2,600 | 9,690 | 345,541 | 9,868 | 9,956 | 9,882 |
| 90 | 124,208 | 414.0 | 3,048 | 10,708 | 380,398 | 9,794 | 9,903 | 9,882 |
| 95 | 126,938 | 423.1 | 3,179 | 11,949 | 404,788 | 9,802 | 9,898 | 9,872 |

## Book

Two-sided book after 100.0% of events.

**Spread (ticks)**

| spread | share of events | share of time |
|---|---:|---:|
| 1 | 81.8% | 81.6% |
| 2 | 14.3% | 14.0% |
| 3 | 3.2% | 3.2% |
| 4 | 0.5% | 0.8% |
| 5+ | 0.2% | 0.3% |

**Touch size and depth (lots / orders, two-sided events)**

| quantity | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best bid qty | 2,412,166 | 950 | 14 | 141 | 528 | 1,256 | 3,128 | 5,969 | 54,983 |
| best ask qty | 2,412,166 | 930 | 14 | 132 | 500 | 1,215 | 3,093 | 6,049 | 117,447 |
| orders at best bid | 2,412,166 | 12.5 | 1 | 3 | 10 | 19 | 33 | 42 | 62 |
| orders at best ask | 2,412,166 | 12.2 | 1 | 3 | 9 | 19 | 33 | 41 | 67 |
| top-5 bid qty | 2,412,166 | 3,842 | 916 | 2,417 | 3,642 | 4,812 | 7,377 | 11,465 | 86,653 |
| top-5 ask qty | 2,412,166 | 3,841 | 934 | 2,422 | 3,624 | 4,782 | 7,501 | 11,421 | 120,149 |

**Mid changes**: after 1.4% of events, 5.77 per second; size 0.5 tick 83.5%, 1 tick 12.0%, 1.5-2 ticks 3.5%, more 0.9%.

**Mid-change volatility by horizon** (regular time grid; VR = variance ratio vs. 1 s, <1 mean reversion, >1 trending)

| horizon | samples | std (ticks) | mean abs | P(no change) | std / sqrt(s) | VR vs 1 s |
|---|---:|---:|---:|---:|---:|---:|
| 10 ms | 599,997 | 0.398 | 0.036 | 95.1% | 3.984 | 0.79 |
| 100 ms | 59,999 | 1.219 | 0.294 | 72.9% | 3.856 | 0.74 |
| 1,000 ms | 5,999 | 4.488 | 1.987 | 28.0% | 4.488 | 1.00 |
| 10,000 ms | 599 | 14.623 | 7.906 | 6.5% | 4.624 | 1.06 |
| 60,000 ms | 99 | 34.060 | 20.692 | 2.0% | 4.397 | 0.96 |

**Successive mid changes** (34,637): autocorrelation lag 1..5 = +0.058, +0.099, +0.110, -0.070, +0.029; P(next change reverses) = 0.327 (0.5 = random walk, >0.5 bounce / mean reversion).

**Trade-sign autocorrelation** (52,961 trading events): lag 1 +0.052, lag 2 +0.047, lag 5 +0.023, lag 10 +0.013, lag 50 +0.002; buyer-initiated share 49.8%.

## Signals vs. future mid change

Label: mid(t + 5 ms + h) - mid(t + 5 ms), mid = last two-sided mid of the book after all events before that time (what an order sent at t would meet). 2,408,150 events with a two-sided book and a known label, cut into 8 equal-event chunks. Signals use rows <= t only.

Label std (ticks): 10 ms 0.332 (5.3% non-zero), 100 ms 0.982 (28.4% non-zero), 1000 ms 3.803 (73.3% non-zero), 10000 ms 14.803 (93.2% non-zero).

Signals: `imb_l1` = (b1-a1)/(b1+a1); `imb_top5` same over 5 price levels; `micro_minus_mid` = microprice - mid (ticks); `flow_Nev` = signed traded volume (+ = buyer-initiated) of the last N events; `flow_100ms` / `flow_1s` = over the last T.

**Mean Spearman IC over chunks** (t-stat across chunks, chunks agreeing in sign / chunks)

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.100 (t +102.2, 8/8) | +0.202 (t +36.2, 8/8) | +0.098 (t +11.4, 8/8) | +0.022 (t +3.1, 7/8) |
| `imb_top5` | +0.059 (t +24.5, 8/8) | +0.125 (t +23.6, 8/8) | +0.059 (t +8.4, 8/8) | -0.010 (t -1.2, 6/8) |
| `micro_minus_mid` | +0.096 (t +100.8, 8/8) | +0.185 (t +28.6, 8/8) | +0.087 (t +9.3, 8/8) | +0.019 (t +2.7, 7/8) |
| `flow_10ev` | +0.081 (t +22.7, 8/8) | +0.120 (t +24.7, 8/8) | +0.102 (t +32.1, 8/8) | +0.019 (t +9.5, 8/8) |
| `flow_100ev` | +0.132 (t +20.8, 8/8) | +0.300 (t +25.4, 8/8) | +0.260 (t +31.3, 8/8) | +0.037 (t +7.2, 8/8) |
| `flow_100ms` | +0.114 (t +17.4, 8/8) | +0.251 (t +19.8, 8/8) | +0.221 (t +31.7, 8/8) | +0.039 (t +9.7, 8/8) |
| `flow_1s` | +0.107 (t +22.9, 8/8) | +0.248 (t +23.8, 8/8) | +0.221 (t +19.4, 8/8) | -0.019 (t -2.0, 6/8) |

**Pooled-chunk mean Pearson IC**

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.058 | +0.125 | +0.066 | +0.011 |
| `imb_top5` | +0.053 | +0.121 | +0.078 | -0.007 |
| `micro_minus_mid` | +0.023 | +0.077 | +0.037 | +0.005 |
| `flow_10ev` | +0.065 | +0.084 | +0.078 | +0.013 |
| `flow_100ev` | +0.079 | +0.195 | +0.193 | +0.025 |
| `flow_100ms` | +0.072 | +0.172 | +0.179 | +0.028 |
| `flow_1s` | +0.071 | +0.182 | +0.198 | -0.012 |

**Spearman IC per chunk, h = 10 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.099 | +0.096 | +0.100 | +0.104 | +0.104 | +0.099 | +0.100 | +0.101 |
| `imb_top5` | +0.052 | +0.053 | +0.056 | +0.063 | +0.061 | +0.069 | +0.051 | +0.067 |
| `micro_minus_mid` | +0.096 | +0.092 | +0.097 | +0.096 | +0.100 | +0.093 | +0.098 | +0.095 |
| `flow_10ev` | +0.070 | +0.082 | +0.074 | +0.087 | +0.078 | +0.098 | +0.069 | +0.090 |
| `flow_100ev` | +0.119 | +0.122 | +0.117 | +0.151 | +0.125 | +0.159 | +0.113 | +0.148 |
| `flow_100ms` | +0.100 | +0.106 | +0.100 | +0.137 | +0.106 | +0.139 | +0.094 | +0.132 |
| `flow_1s` | +0.100 | +0.095 | +0.106 | +0.129 | +0.098 | +0.116 | +0.092 | +0.121 |

**Spearman IC per chunk, h = 100 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.218 | +0.190 | +0.218 | +0.192 | +0.207 | +0.177 | +0.221 | +0.196 |
| `imb_top5` | +0.108 | +0.101 | +0.141 | +0.124 | +0.131 | +0.130 | +0.122 | +0.145 |
| `micro_minus_mid` | +0.202 | +0.175 | +0.202 | +0.172 | +0.189 | +0.155 | +0.208 | +0.178 |
| `flow_10ev` | +0.112 | +0.115 | +0.110 | +0.144 | +0.114 | +0.136 | +0.105 | +0.126 |
| `flow_100ev` | +0.275 | +0.285 | +0.279 | +0.343 | +0.288 | +0.344 | +0.260 | +0.330 |
| `flow_100ms` | +0.227 | +0.237 | +0.227 | +0.303 | +0.237 | +0.294 | +0.205 | +0.277 |
| `flow_1s` | +0.236 | +0.228 | +0.254 | +0.292 | +0.221 | +0.254 | +0.210 | +0.285 |

**Spearman IC per chunk, h = 1000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.149 | +0.079 | +0.094 | +0.089 | +0.113 | +0.072 | +0.102 | +0.087 |
| `imb_top5` | +0.071 | +0.039 | +0.047 | +0.055 | +0.078 | +0.060 | +0.031 | +0.089 |
| `micro_minus_mid` | +0.144 | +0.067 | +0.080 | +0.075 | +0.099 | +0.057 | +0.091 | +0.081 |
| `flow_10ev` | +0.111 | +0.099 | +0.109 | +0.111 | +0.090 | +0.099 | +0.089 | +0.104 |
| `flow_100ev` | +0.277 | +0.258 | +0.286 | +0.274 | +0.232 | +0.242 | +0.227 | +0.283 |
| `flow_100ms` | +0.229 | +0.216 | +0.234 | +0.241 | +0.200 | +0.214 | +0.188 | +0.243 |
| `flow_1s` | +0.225 | +0.213 | +0.237 | +0.280 | +0.178 | +0.209 | +0.186 | +0.237 |

**Spearman IC per chunk, h = 10000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.044 | +0.007 | +0.036 | -0.012 | +0.017 | +0.016 | +0.046 | +0.017 |
| `imb_top5` | +0.010 | -0.035 | -0.008 | -0.043 | -0.004 | -0.004 | -0.024 | +0.028 |
| `micro_minus_mid` | +0.046 | +0.005 | +0.033 | -0.012 | +0.014 | +0.008 | +0.040 | +0.015 |
| `flow_10ev` | +0.022 | +0.025 | +0.020 | +0.025 | +0.011 | +0.018 | +0.010 | +0.021 |
| `flow_100ev` | +0.037 | +0.048 | +0.038 | +0.061 | +0.018 | +0.039 | +0.017 | +0.037 |
| `flow_100ms` | +0.041 | +0.050 | +0.046 | +0.053 | +0.027 | +0.037 | +0.020 | +0.034 |
| `flow_1s` | -0.051 | -0.019 | -0.007 | +0.021 | -0.030 | +0.004 | -0.059 | -0.013 |
