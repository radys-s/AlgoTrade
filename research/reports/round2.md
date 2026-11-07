# Data card - `data\round2.csv`

Generated 2026-11-07 19:17 by `research.profile` from `data\features\round2.npz` (features built in 17.5 s). Prices in ticks.

## Observations

- Spread is 1 tick only 83.9% of the time (median 1 ticks) -> room to quote inside the spread (inserts improving the touch: 1.4%).
- Mid changes trend at the tick level (P(reversal) = 0.38): a move tends to be followed by another move the same way.
- Variance ratio 60 s vs 1 s = 1.27 > 1: prices trend over a minute; inventory against the trend is dangerous, momentum may pay.
- Most reliable simple signal: `flow_1s` at 100 ms, mean rank IC +0.179 (t +68.1), same sign in 8/8 chunks; the whole move it predicts has std 0.92 ticks at 100 ms, vs. 0.5 tick(s) half-spread paid by an IOC.
- 1 s mid-change std is 3.79 ticks vs a typical spread of 1 tick(s): moves are large relative to the spread.

## Tape

| metric | value |
|---|---:|
| rows | 2,193,064 |
| duration | 6,000.0 s (1.67 h), Time 0.000 .. 5999.996 |
| events / s (mean) | 366 |
| rows sharing the previous row's timestamp | 0.0% |
| insert limit (G) / insert IOC (F) / cancel | 48.8% / 2.4% / 48.8% |
| cancels of unknown ids | 0 |
| limit inserts bid / ask (count) | 50.0% / 50.0% |
| IOC buy / sell (count) | 26,643 / 26,780 |
| IOC buy / sell (volume) | 4,297,629 / 4,349,817 |
| traded volume buyer- / seller-initiated | 3,479,444 / 3,520,448 |
| insert price range (ticks) | 9,444 .. 10,017 |

**Order sizes (lots)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| limit (G) | 1,069,859 | 72.4 | 1 | 4 | 14 | 48 | 276 | 940 | 72,482 |
| IOC (F) | 53,423 | 162 | 6 | 23 | 61 | 156 | 614 | 1,543 | 26,005 |

**Where limit orders are placed** (vs. the book just before the insert)

| placement | share |
|---|---:|
| crosses the spread (marketable) | 0.0% |
| own side empty | 0.0% |
| inside the spread (improves) | 1.4% |
| at the touch | 51.7% |
| 1-2 ticks behind | 43.8% |
| 3-5 ticks behind | 2.9% |
| 6-19 ticks behind | 0.2% |
| 20+ ticks behind (invisible) | 0.0% |

IOC orders: 96.1% trade anything, 16.7% move the mid, 3.70 fills per IOC on average.

**Order lifetime, insert -> cancel (ms)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cancelled orders | 1,069,782 | 442 | 48.1 | 154 | 312 | 583 | 1,280 | 2,061 | 7,149 |

100.0% of limit orders are eventually cancelled (the rest fill, or live to the end); 0.0% of cancels come within 5 ms of the insert (faster than our latency).

**Activity by 5-minute bucket**

| minute | events | events/s | IOC | fills | traded vol | mid low | mid high | mid end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 109,245 | 364.1 | 2,445 | 8,825 | 326,680 | 9,948 | 10,002 | 9,972 |
| 5 | 110,058 | 366.9 | 2,732 | 10,027 | 349,966 | 9,898 | 9,978 | 9,902 |
| 10 | 108,848 | 362.8 | 2,521 | 9,477 | 345,126 | 9,858 | 9,906 | 9,868 |
| 15 | 109,534 | 365.1 | 2,423 | 9,117 | 327,504 | 9,862 | 9,882 | 9,874 |
| 20 | 109,807 | 366.0 | 2,432 | 9,798 | 347,066 | 9,848 | 9,876 | 9,848 |
| 25 | 108,252 | 360.8 | 2,650 | 9,685 | 346,516 | 9,842 | 9,956 | 9,912 |
| 30 | 109,295 | 364.3 | 2,712 | 10,020 | 372,853 | 9,842 | 9,944 | 9,932 |
| 35 | 110,067 | 366.9 | 2,672 | 9,681 | 339,044 | 9,887 | 9,949 | 9,900 |
| 40 | 109,854 | 366.2 | 2,673 | 10,053 | 343,503 | 9,898 | 10,004 | 9,968 |
| 45 | 109,939 | 366.5 | 2,539 | 9,496 | 335,592 | 9,904 | 9,968 | 9,924 |
| 50 | 109,310 | 364.4 | 2,501 | 9,650 | 354,242 | 9,918 | 10,008 | 10,000 |
| 55 | 109,511 | 365.0 | 2,543 | 9,523 | 340,468 | 9,932 | 10,008 | 9,946 |
| 60 | 110,003 | 366.7 | 3,415 | 12,208 | 425,816 | 9,722 | 9,950 | 9,726 |
| 65 | 110,346 | 367.8 | 3,069 | 10,889 | 376,722 | 9,580 | 9,766 | 9,582 |
| 70 | 109,895 | 366.3 | 2,953 | 10,391 | 357,131 | 9,455 | 9,584 | 9,478 |
| 75 | 110,444 | 368.1 | 2,428 | 9,345 | 323,981 | 9,470 | 9,491 | 9,480 |
| 80 | 109,894 | 366.3 | 2,589 | 9,540 | 342,664 | 9,476 | 9,582 | 9,570 |
| 85 | 109,258 | 364.2 | 2,511 | 9,281 | 333,138 | 9,557 | 9,684 | 9,682 |
| 90 | 109,275 | 364.2 | 3,058 | 10,996 | 376,510 | 9,645 | 9,824 | 9,770 |
| 95 | 110,229 | 367.4 | 2,557 | 9,586 | 335,370 | 9,672 | 9,770 | 9,690 |

## Book

Two-sided book after 99.9% of events.

**Spread (ticks)**

| spread | share of events | share of time |
|---|---:|---:|
| 1 | 85.1% | 83.9% |
| 2 | 11.4% | 11.6% |
| 3 | 2.8% | 3.1% |
| 4 | 0.5% | 0.9% |
| 5+ | 0.3% | 0.4% |

**Touch size and depth (lots / orders, two-sided events)**

| quantity | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best bid qty | 2,191,857 | 1,122 | 19 | 226 | 694 | 1,475 | 3,513 | 6,700 | 68,446 |
| best ask qty | 2,191,857 | 1,142 | 19 | 235 | 723 | 1,518 | 3,538 | 6,657 | 67,200 |
| orders at best bid | 2,191,857 | 14.4 | 1 | 5 | 12 | 22 | 35 | 43 | 63 |
| orders at best ask | 2,191,857 | 14.8 | 1 | 5 | 13 | 23 | 35 | 43 | 63 |
| top-5 bid qty | 2,191,857 | 2,660 | 601 | 1,404 | 2,191 | 3,272 | 6,133 | 10,573 | 68,494 |
| top-5 ask qty | 2,191,857 | 2,672 | 595 | 1,425 | 2,197 | 3,284 | 6,121 | 10,424 | 74,990 |

**Mid changes**: after 1.3% of events, 4.85 per second; size 0.5 tick 83.2%, 1 tick 11.2%, 1.5-2 ticks 4.3%, more 1.3%.

**Mid-change volatility by horizon** (regular time grid; VR = variance ratio vs. 1 s, <1 mean reversion, >1 trending)

| horizon | samples | std (ticks) | mean abs | P(no change) | std / sqrt(s) | VR vs 1 s |
|---|---:|---:|---:|---:|---:|---:|
| 10 ms | 599,997 | 0.313 | 0.030 | 95.9% | 3.126 | 0.68 |
| 100 ms | 59,999 | 1.094 | 0.240 | 78.1% | 3.458 | 0.83 |
| 1,000 ms | 5,999 | 3.786 | 1.479 | 35.4% | 3.786 | 1.00 |
| 10,000 ms | 599 | 12.046 | 6.178 | 7.2% | 3.809 | 1.01 |
| 60,000 ms | 99 | 33.022 | 19.631 | 0.0% | 4.263 | 1.27 |

**Successive mid changes** (29,107): autocorrelation lag 1..5 = +0.184, +0.003, +0.055, +0.046, +0.006; P(next change reverses) = 0.381 (0.5 = random walk, >0.5 bounce / mean reversion).

**Trade-sign autocorrelation** (51,336 trading events): lag 1 +0.030, lag 2 +0.020, lag 5 +0.010, lag 10 +0.010, lag 50 +0.005; buyer-initiated share 49.7%.

## Signals vs. future mid change

Label: mid(t + 5 ms + h) - mid(t + 5 ms), mid = last two-sided mid of the book after all events before that time (what an order sent at t would meet). 2,188,131 events with a two-sided book and a known label, cut into 8 equal-event chunks. Signals use rows <= t only.

Label std (ticks): 10 ms 0.308 (4.1% non-zero), 100 ms 0.917 (21.8% non-zero), 1000 ms 3.060 (64.7% non-zero), 10000 ms 11.564 (92.5% non-zero).

Signals: `imb_l1` = (b1-a1)/(b1+a1); `imb_top5` same over 5 price levels; `micro_minus_mid` = microprice - mid (ticks); `flow_Nev` = signed traded volume (+ = buyer-initiated) of the last N events; `flow_100ms` / `flow_1s` = over the last T.

**Mean Spearman IC over chunks** (t-stat across chunks, chunks agreeing in sign / chunks)

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.095 (t +26.3, 8/8) | +0.204 (t +11.1, 8/8) | +0.092 (t +11.4, 8/8) | +0.014 (t +1.7, 6/8) |
| `imb_top5` | +0.055 (t +26.5, 8/8) | +0.115 (t +24.2, 8/8) | +0.049 (t +8.8, 8/8) | +0.014 (t +1.5, 6/8) |
| `micro_minus_mid` | +0.094 (t +20.0, 8/8) | +0.190 (t +9.3, 8/8) | +0.083 (t +8.9, 8/8) | +0.011 (t +1.3, 4/8) |
| `flow_10ev` | +0.073 (t +11.8, 8/8) | +0.109 (t +11.8, 8/8) | +0.091 (t +57.5, 8/8) | +0.027 (t +11.6, 8/8) |
| `flow_100ev` | +0.108 (t +12.4, 8/8) | +0.244 (t +17.3, 8/8) | +0.222 (t +36.0, 8/8) | +0.060 (t +10.5, 8/8) |
| `flow_100ms` | +0.094 (t +9.3, 8/8) | +0.195 (t +10.6, 8/8) | +0.183 (t +62.9, 8/8) | +0.053 (t +13.8, 8/8) |
| `flow_1s` | +0.077 (t +45.0, 8/8) | +0.179 (t +68.1, 8/8) | +0.153 (t +14.9, 8/8) | +0.046 (t +4.9, 8/8) |

**Pooled-chunk mean Pearson IC**

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.054 | +0.115 | +0.041 | -0.001 |
| `imb_top5` | +0.038 | +0.085 | +0.027 | +0.013 |
| `micro_minus_mid` | +0.045 | +0.103 | +0.031 | -0.002 |
| `flow_10ev` | +0.048 | +0.084 | +0.068 | +0.020 |
| `flow_100ev` | +0.071 | +0.185 | +0.140 | +0.043 |
| `flow_100ms` | +0.060 | +0.164 | +0.128 | +0.039 |
| `flow_1s` | +0.047 | +0.124 | +0.084 | +0.033 |

**Spearman IC per chunk, h = 10 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.095 | +0.110 | +0.094 | +0.094 | +0.092 | +0.083 | +0.109 | +0.083 |
| `imb_top5` | +0.050 | +0.046 | +0.064 | +0.051 | +0.055 | +0.061 | +0.058 | +0.054 |
| `micro_minus_mid` | +0.097 | +0.113 | +0.091 | +0.091 | +0.092 | +0.075 | +0.111 | +0.080 |
| `flow_10ev` | +0.070 | +0.045 | +0.083 | +0.064 | +0.074 | +0.101 | +0.059 | +0.087 |
| `flow_100ev` | +0.101 | +0.074 | +0.122 | +0.092 | +0.110 | +0.154 | +0.091 | +0.123 |
| `flow_100ms` | +0.089 | +0.053 | +0.110 | +0.076 | +0.098 | +0.143 | +0.068 | +0.111 |
| `flow_1s` | +0.074 | +0.073 | +0.081 | +0.070 | +0.075 | +0.083 | +0.082 | +0.077 |

**Spearman IC per chunk, h = 100 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.215 | +0.273 | +0.177 | +0.239 | +0.193 | +0.122 | +0.258 | +0.157 |
| `imb_top5` | +0.109 | +0.103 | +0.129 | +0.125 | +0.108 | +0.120 | +0.133 | +0.095 |
| `micro_minus_mid` | +0.205 | +0.268 | +0.157 | +0.224 | +0.178 | +0.098 | +0.252 | +0.141 |
| `flow_10ev` | +0.105 | +0.074 | +0.120 | +0.092 | +0.111 | +0.157 | +0.086 | +0.124 |
| `flow_100ev` | +0.225 | +0.186 | +0.264 | +0.223 | +0.244 | +0.318 | +0.224 | +0.271 |
| `flow_100ms` | +0.176 | +0.124 | +0.223 | +0.161 | +0.204 | +0.289 | +0.156 | +0.227 |
| `flow_1s` | +0.174 | +0.178 | +0.185 | +0.173 | +0.168 | +0.180 | +0.190 | +0.186 |

**Spearman IC per chunk, h = 1000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.095 | +0.126 | +0.081 | +0.110 | +0.077 | +0.058 | +0.113 | +0.077 |
| `imb_top5` | +0.032 | +0.051 | +0.050 | +0.065 | +0.036 | +0.077 | +0.049 | +0.034 |
| `micro_minus_mid` | +0.088 | +0.122 | +0.072 | +0.100 | +0.063 | +0.046 | +0.108 | +0.061 |
| `flow_10ev` | +0.089 | +0.094 | +0.095 | +0.090 | +0.088 | +0.087 | +0.098 | +0.086 |
| `flow_100ev` | +0.225 | +0.240 | +0.219 | +0.242 | +0.209 | +0.198 | +0.240 | +0.204 |
| `flow_100ms` | +0.173 | +0.190 | +0.188 | +0.183 | +0.180 | +0.175 | +0.195 | +0.174 |
| `flow_1s` | +0.198 | +0.174 | +0.138 | +0.170 | +0.116 | +0.153 | +0.159 | +0.114 |

**Spearman IC per chunk, h = 10000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.043 | +0.028 | +0.003 | +0.002 | +0.002 | -0.010 | -0.006 | +0.048 |
| `imb_top5` | +0.061 | +0.002 | +0.016 | -0.011 | -0.019 | +0.036 | +0.019 | +0.006 |
| `micro_minus_mid` | +0.043 | +0.027 | +0.008 | -0.001 | -0.006 | -0.013 | -0.012 | +0.041 |
| `flow_10ev` | +0.035 | +0.024 | +0.029 | +0.023 | +0.022 | +0.024 | +0.037 | +0.019 |
| `flow_100ev` | +0.070 | +0.054 | +0.065 | +0.032 | +0.061 | +0.054 | +0.088 | +0.054 |
| `flow_100ms` | +0.061 | +0.046 | +0.056 | +0.044 | +0.046 | +0.047 | +0.075 | +0.046 |
| `flow_1s` | +0.066 | +0.021 | +0.036 | +0.014 | +0.059 | +0.061 | +0.090 | +0.023 |
