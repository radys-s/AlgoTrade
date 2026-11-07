# Data card - `data\sample.csv`

Generated 2026-11-07 11:21 by `research.profile` from `data\features\sample.npz` (features built in 0.7 s). Prices in ticks.

## Observations

- The book is one-sided or empty after 74.2% of events -> guard every use of best bid/ask, and expect the mid (last two-sided) to jump when a side reappears.
- Spread is 1 tick only 0.0% of the time (median 2 ticks) -> room to quote inside the spread (inserts improving the touch: 0.0%).
- Mid changes trend at the tick level (P(reversal) = 0.35): a move tends to be followed by another move the same way.
- Variance ratio 60 s vs 1 s = 1.85 > 1: prices trend over a minute; inventory against the trend is dangerous, momentum may pay.
- Most reliable simple signal: `flow_100ev` at 1000 ms, mean rank IC +0.683 (t +55.4), same sign in 8/8 chunks; the whole move it predicts has std 3.28 ticks at 1000 ms, vs. 1 tick(s) half-spread paid by an IOC.

## Tape

| metric | value |
|---|---:|
| rows | 53,758 |
| duration | 3,599.0 s (1.00 h), Time 0.000 .. 3599.000 |
| events / s (mean) | 14.9 |
| rows sharing the previous row's timestamp | 0.0% |
| insert limit (G) / insert IOC (F) / cancel | 49.4% / 1.1% / 49.4% |
| cancels of unknown ids | 0 |
| limit inserts bid / ask (count) | 50.0% / 50.0% |
| IOC buy / sell (count) | 307 / 297 |
| IOC buy / sell (volume) | 3,868 / 3,838 |
| traded volume buyer- / seller-initiated | 3,868 / 3,838 |
| insert price range (ticks) | 9,866 .. 10,074 |

**Order sizes (lots)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| limit (G) | 26,578 | 25 | 11 | 17 | 25 | 33 | 39 | 40 | 40 |
| IOC (F) | 604 | 12.8 | 5 | 8 | 12 | 17 | 22 | 26 | 28 |

**Where limit orders are placed** (vs. the book just before the insert)

| placement | share |
|---|---:|
| crosses the spread (marketable) | 0.0% |
| own side empty | 100.0% |
| inside the spread (improves) | 0.0% |
| at the touch | 0.0% |
| 1-2 ticks behind | 0.0% |
| 3-5 ticks behind | 0.0% |
| 6-19 ticks behind | 0.0% |
| 20+ ticks behind (invisible) | 0.0% |

IOC orders: 100.0% trade anything, 0.0% move the mid, 1.00 fills per IOC on average.

**Order lifetime, insert -> cancel (ms)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cancelled orders | 26,576 | 271 | 14 | 83 | 190 | 334 | 1,000 | 1,000 | 1,000 |

100.0% of limit orders are eventually cancelled (the rest fill, or live to the end); 2.2% of cancels come within 5 ms of the insert (faster than our latency).

**Activity by 5-minute bucket**

| minute | events | events/s | IOC | fills | traded vol | mid low | mid high | mid end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4,541 | 15.1 | 55 | 55 | 740 | 9,919 | 10,013 | 9,940 |
| 5 | 4,185 | 13.9 | 45 | 45 | 533 | 9,867 | 9,954 | 9,867 |
| 10 | 4,244 | 14.1 | 44 | 44 | 574 | 9,867 | 9,988 | 9,985 |
| 15 | 4,413 | 14.7 | 49 | 49 | 673 | 9,925 | 9,987 | 9,930 |
| 20 | 4,820 | 16.1 | 56 | 56 | 709 | 9,930 | 9,998 | 9,985 |
| 25 | 4,319 | 14.4 | 47 | 47 | 575 | 9,902 | 9,988 | 9,915 |
| 30 | 4,552 | 15.2 | 52 | 52 | 607 | 9,915 | 10,012 | 9,976 |
| 35 | 4,149 | 13.8 | 45 | 45 | 600 | 9,921 | 10,024 | 9,980 |
| 40 | 4,879 | 16.3 | 55 | 55 | 690 | 9,953 | 10,005 | 9,956 |
| 45 | 4,622 | 15.4 | 54 | 54 | 741 | 9,904 | 9,977 | 9,946 |
| 50 | 4,667 | 15.6 | 55 | 55 | 704 | 9,939 | 10,073 | 10,031 |
| 55 | 4,367 | 14.6 | 47 | 47 | 560 | 9,994 | 10,056 | 10,037 |

## Book

Two-sided book after 25.8% of events.

**Spread (ticks)**

| spread | share of events | share of time |
|---|---:|---:|
| 1 | 0.0% | 0.0% |
| 2 | 100.0% | 100.0% |
| 3 | 0.0% | 0.0% |
| 4 | 0.0% | 0.0% |
| 5+ | 0.0% | 0.0% |

**Touch size and depth (lots / orders, two-sided events)**

| quantity | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best bid qty | 13,893 | 24.7 | 11 | 17 | 25 | 33 | 39 | 40 | 40 |
| best ask qty | 13,893 | 24.7 | 11 | 17 | 25 | 33 | 39 | 40 | 40 |
| orders at best bid | 13,893 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| orders at best ask | 13,893 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| top-5 bid qty | 13,893 | 24.7 | 11 | 17 | 25 | 33 | 39 | 40 | 40 |
| top-5 ask qty | 13,893 | 24.7 | 11 | 17 | 25 | 33 | 39 | 40 | 40 |

**Mid changes**: after 18.0% of events, 2.69 per second; size 0.5 tick 0.0%, 1 tick 100.0%, 1.5-2 ticks 0.0%, more 0.0%.

**Mid-change volatility by horizon** (regular time grid; VR = variance ratio vs. 1 s, <1 mean reversion, >1 trending)

| horizon | samples | std (ticks) | mean abs | P(no change) | std / sqrt(s) | VR vs 1 s |
|---|---:|---:|---:|---:|---:|---:|
| 10 ms | 359,900 | 0.164 | 0.027 | 97.4% | 1.643 | 0.41 |
| 100 ms | 35,990 | 0.518 | 0.238 | 77.7% | 1.637 | 0.41 |
| 1,000 ms | 3,599 | 2.565 | 1.775 | 27.1% | 2.565 | 1.00 |
| 10,000 ms | 359 | 12.618 | 9.944 | 3.1% | 3.990 | 2.42 |
| 60,000 ms | 59 | 27.026 | 20.712 | 3.4% | 3.489 | 1.85 |

**Successive mid changes** (9,689): autocorrelation lag 1..5 = +0.295, +0.347, +0.305, +0.280, +0.268; P(next change reverses) = 0.352 (0.5 = random walk, >0.5 bounce / mean reversion).

**Trade-sign autocorrelation** (604 trading events): lag 1 -0.015, lag 2 +0.069, lag 5 -0.039, lag 10 -0.034, lag 50 -0.001; buyer-initiated share 50.8%.

## Signals vs. future mid change

Label: mid(t + 5 ms + h) - mid(t + 5 ms), mid = last two-sided mid of the book after all events before that time (what an order sent at t would meet). 13,854 events with a two-sided book and a known label, cut into 8 equal-event chunks. Signals use rows <= t only.

Label std (ticks): 10 ms 0.159 (2.5% non-zero), 100 ms 0.515 (22.0% non-zero), 1000 ms 3.280 (81.6% non-zero), 10000 ms 13.239 (96.6% non-zero).

Signals: `imb_l1` = (b1-a1)/(b1+a1); `imb_top5` same over 5 price levels; `micro_minus_mid` = microprice - mid (ticks); `flow_Nev` = signed traded volume (+ = buyer-initiated) of the last N events; `flow_100ms` / `flow_1s` = over the last T.

**Mean Spearman IC over chunks** (t-stat across chunks, chunks agreeing in sign / chunks)

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.010 (t +2.1, 7/8) | -0.007 (t -1.2, 5/8) | +0.045 (t +6.8, 8/8) | +0.052 (t +9.2, 8/8) |
| `imb_top5` | +0.010 (t +2.1, 7/8) | -0.007 (t -1.2, 5/8) | +0.045 (t +6.8, 8/8) | +0.052 (t +9.2, 8/8) |
| `micro_minus_mid` | +0.010 (t +2.1, 7/8) | -0.007 (t -1.2, 5/8) | +0.045 (t +6.8, 8/8) | +0.052 (t +9.2, 8/8) |
| `flow_10ev` | -0.001 (t -0.1, 4/8) | +0.058 (t +5.1, 8/8) | +0.366 (t +54.2, 8/8) | +0.234 (t +28.0, 8/8) |
| `flow_100ev` | +0.071 (t +8.1, 8/8) | +0.226 (t +18.3, 8/8) | +0.683 (t +55.4, 8/8) | +0.218 (t +7.2, 8/8) |
| `flow_100ms` | +0.002 (t +0.1, 3/8) | -0.003 (t -0.3, 4/8) | +0.233 (t +35.4, 8/8) | +0.164 (t +23.0, 8/8) |
| `flow_1s` | +0.042 (t +2.9, 6/8) | +0.151 (t +12.7, 8/8) | +0.590 (t +43.1, 8/8) | +0.319 (t +22.4, 8/8) |

**Pooled-chunk mean Pearson IC**

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.008 | -0.005 | +0.052 | +0.054 |
| `imb_top5` | +0.008 | -0.005 | +0.052 | +0.054 |
| `micro_minus_mid` | +0.008 | -0.005 | +0.052 | +0.054 |
| `flow_10ev` | +0.002 | +0.061 | +0.334 | +0.220 |
| `flow_100ev` | +0.076 | +0.237 | +0.684 | +0.219 |
| `flow_100ms` | -0.001 | +0.005 | +0.219 | +0.154 |
| `flow_1s` | +0.046 | +0.152 | +0.570 | +0.301 |

**Spearman IC per chunk, h = 10 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.028 | +0.003 | +0.026 | +0.006 | +0.000 | +0.021 | -0.010 | +0.006 |
| `imb_top5` | +0.028 | +0.003 | +0.026 | +0.006 | +0.000 | +0.021 | -0.010 | +0.006 |
| `micro_minus_mid` | +0.028 | +0.003 | +0.026 | +0.006 | +0.000 | +0.021 | -0.010 | +0.006 |
| `flow_10ev` | +0.051 | -0.035 | -0.029 | +0.001 | +0.025 | -0.020 | -0.010 | +0.010 |
| `flow_100ev` | +0.087 | +0.061 | +0.074 | +0.050 | +0.059 | +0.099 | +0.033 | +0.103 |
| `flow_100ms` | +0.059 | -0.001 | -0.029 | -0.017 | -0.019 | +0.014 | -0.025 | +0.031 |
| `flow_1s` | +0.087 | -0.005 | +0.027 | -0.015 | +0.074 | +0.064 | +0.017 | +0.089 |

**Spearman IC per chunk, h = 100 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.008 | -0.016 | +0.011 | -0.003 | +0.015 | -0.017 | -0.025 | -0.026 |
| `imb_top5` | +0.008 | -0.016 | +0.011 | -0.003 | +0.015 | -0.017 | -0.025 | -0.026 |
| `micro_minus_mid` | +0.008 | -0.016 | +0.011 | -0.003 | +0.015 | -0.017 | -0.025 | -0.026 |
| `flow_10ev` | +0.086 | +0.022 | +0.012 | +0.098 | +0.087 | +0.050 | +0.043 | +0.062 |
| `flow_100ev` | +0.270 | +0.206 | +0.167 | +0.247 | +0.219 | +0.259 | +0.197 | +0.240 |
| `flow_100ms` | -0.001 | -0.024 | -0.036 | +0.010 | +0.005 | +0.019 | -0.029 | +0.033 |
| `flow_1s` | +0.206 | +0.143 | +0.107 | +0.137 | +0.144 | +0.192 | +0.161 | +0.122 |

**Spearman IC per chunk, h = 1000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.063 | +0.053 | +0.019 | +0.041 | +0.076 | +0.047 | +0.031 | +0.030 |
| `imb_top5` | +0.063 | +0.053 | +0.019 | +0.041 | +0.076 | +0.047 | +0.031 | +0.030 |
| `micro_minus_mid` | +0.063 | +0.053 | +0.019 | +0.041 | +0.076 | +0.047 | +0.031 | +0.030 |
| `flow_10ev` | +0.385 | +0.361 | +0.331 | +0.380 | +0.374 | +0.384 | +0.347 | +0.364 |
| `flow_100ev` | +0.677 | +0.676 | +0.613 | +0.705 | +0.727 | +0.702 | +0.662 | +0.698 |
| `flow_100ms` | +0.244 | +0.241 | +0.202 | +0.232 | +0.256 | +0.253 | +0.217 | +0.223 |
| `flow_1s` | +0.639 | +0.589 | +0.529 | +0.589 | +0.583 | +0.649 | +0.571 | +0.569 |

**Spearman IC per chunk, h = 10000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.059 | +0.078 | +0.064 | +0.028 | +0.048 | +0.047 | +0.056 | +0.037 |
| `imb_top5` | +0.059 | +0.078 | +0.064 | +0.028 | +0.048 | +0.047 | +0.056 | +0.037 |
| `micro_minus_mid` | +0.059 | +0.078 | +0.064 | +0.028 | +0.048 | +0.047 | +0.056 | +0.037 |
| `flow_10ev` | +0.210 | +0.259 | +0.226 | +0.213 | +0.277 | +0.243 | +0.222 | +0.222 |
| `flow_100ev` | +0.201 | +0.168 | +0.156 | +0.216 | +0.413 | +0.208 | +0.141 | +0.245 |
| `flow_100ms` | +0.146 | +0.187 | +0.152 | +0.156 | +0.196 | +0.160 | +0.175 | +0.140 |
| `flow_1s` | +0.297 | +0.351 | +0.297 | +0.292 | +0.400 | +0.332 | +0.276 | +0.309 |
