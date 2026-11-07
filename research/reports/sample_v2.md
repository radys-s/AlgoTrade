# Data card - `data\sample_v2.csv`

Generated 2026-11-07 12:09 by `research.profile` from `data\features\sample_v2.npz` (features built in 14.7 s). Prices in ticks.

## Observations

- Spread is 1 tick only 41.4% of the time (median 2 ticks) -> room to quote inside the spread (inserts improving the touch: 0.5%).
- Mid changes mean-revert at the tick level (P(next change reverses) = 0.63): fade single-tick moves / expect bid-ask bounce rather than chase them.
- Most reliable simple signal: `micro_minus_mid` at 100 ms, mean rank IC +0.333 (t +129.9), same sign in 8/8 chunks; the whole move it predicts has std 0.17 ticks at 100 ms, vs. 1 tick(s) half-spread paid by an IOC.
- 1 s mid-change std is 0.50 ticks vs a typical spread of 2 tick(s): crossing the spread to take short-horizon moves is expensive.

## Tape

| metric | value |
|---|---:|
| rows | 2,000,000 |
| duration | 5,749.9 s (1.60 h), Time 0.027 .. 5749.896 |
| events / s (mean) | 348 |
| rows sharing the previous row's timestamp | 0.0% |
| insert limit (G) / insert IOC (F) / cancel | 51.7% / 1.7% / 46.6% |
| cancels of unknown ids | 0 |
| limit inserts bid / ask (count) | 50.0% / 50.0% |
| IOC buy / sell (count) | 17,281 / 17,222 |
| IOC buy / sell (volume) | 1,859,420 / 1,891,607 |
| traded volume buyer- / seller-initiated | 1,582,394 / 1,595,195 |
| insert price range (ticks) | 9,963 .. 10,026 |

**Order sizes (lots)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| limit (G) | 1,033,029 | 30 | 21 | 25 | 30 | 35 | 39 | 40 | 40 |
| IOC (F) | 34,503 | 109 | 4 | 16 | 41 | 106 | 409 | 1,017 | 21,367 |

**Where limit orders are placed** (vs. the book just before the insert)

| placement | share |
|---|---:|
| crosses the spread (marketable) | 0.0% |
| own side empty | 0.0% |
| inside the spread (improves) | 0.5% |
| at the touch | 52.0% |
| 1-2 ticks behind | 47.4% |
| 3-5 ticks behind | 0.1% |
| 6-19 ticks behind | 0.0% |
| 20+ ticks behind (invisible) | 0.0% |

IOC orders: 97.7% trade anything, 9.5% move the mid, 3.82 fills per IOC on average.

**Order lifetime, insert -> cancel (ms)**

| orders | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cancelled orders | 932,468 | 703 | 64.1 | 236 | 500 | 932 | 2,037 | 3,357 | 11,974 |

90.3% of limit orders are eventually cancelled (the rest fill, or live to the end); 0.0% of cancels come within 5 ms of the insert (faster than our latency).

**Activity by 5-minute bucket**

| minute | events | events/s | IOC | fills | traded vol | mid low | mid high | mid end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 103,732 | 345.8 | 1,807 | 7,102 | 171,433 | 9,995 | 10,008 | 10,008 |
| 5 | 104,566 | 348.6 | 1,794 | 6,939 | 167,532 | 9,988 | 10,009 | 9,990 |
| 10 | 104,108 | 347.0 | 1,764 | 6,602 | 158,441 | 9,984 | 9,998 | 9,997 |
| 15 | 103,845 | 346.1 | 1,867 | 7,039 | 169,117 | 9,992 | 10,007 | 10,003 |
| 20 | 105,325 | 351.1 | 1,762 | 6,849 | 166,734 | 9,998 | 10,015 | 10,013 |
| 25 | 104,924 | 349.7 | 1,782 | 6,540 | 156,030 | 9,996 | 10,013 | 10,000 |
| 30 | 104,024 | 346.7 | 1,798 | 7,163 | 173,255 | 9,984 | 10,001 | 9,990 |
| 35 | 104,271 | 347.6 | 1,773 | 6,769 | 163,857 | 9,980 | 9,992 | 9,982 |
| 40 | 104,013 | 346.7 | 1,773 | 6,803 | 165,343 | 9,970 | 9,984 | 9,974 |
| 45 | 105,051 | 350.2 | 1,744 | 6,428 | 153,406 | 9,969 | 9,984 | 9,976 |
| 50 | 104,034 | 346.8 | 1,806 | 7,273 | 178,640 | 9,966 | 9,979 | 9,968 |
| 55 | 104,992 | 350.0 | 1,900 | 6,856 | 161,254 | 9,967 | 9,982 | 9,978 |
| 60 | 104,511 | 348.4 | 1,843 | 6,806 | 162,042 | 9,978 | 9,987 | 9,982 |
| 65 | 103,819 | 346.1 | 1,773 | 6,661 | 160,776 | 9,976 | 9,995 | 9,994 |
| 70 | 103,422 | 344.7 | 1,853 | 7,351 | 178,936 | 9,990 | 9,997 | 9,997 |
| 75 | 104,338 | 347.8 | 1,786 | 7,063 | 170,146 | 9,994 | 10,004 | 9,994 |
| 80 | 104,579 | 348.6 | 1,754 | 6,657 | 161,166 | 9,992 | 10,012 | 10,011 |
| 85 | 105,175 | 350.6 | 1,765 | 6,681 | 161,981 | 10,010 | 10,023 | 10,017 |
| 90 | 104,289 | 347.6 | 1,841 | 6,960 | 167,337 | 10,000 | 10,017 | 10,008 |
| 95 | 16,982 | 56.6 | 318 | 1,236 | 30,163 | 10,002 | 10,008 | 10,004 |

## Book

Two-sided book after 100.0% of events.

**Spread (ticks)**

| spread | share of events | share of time |
|---|---:|---:|
| 1 | 42.3% | 41.4% |
| 2 | 57.6% | 58.4% |
| 3 | 0.2% | 0.2% |
| 4 | 0.0% | 0.0% |
| 5+ | 0.0% | 0.0% |

**Touch size and depth (lots / orders, two-sided events)**

| quantity | n | mean | q5 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best bid qty | 1,999,989 | 821 | 36 | 263 | 768 | 1,302 | 1,836 | 2,125 | 2,899 |
| best ask qty | 1,999,989 | 818 | 36 | 269 | 766 | 1,291 | 1,832 | 2,124 | 2,999 |
| orders at best bid | 1,999,989 | 27.6 | 1 | 9 | 26 | 44 | 61 | 71 | 96 |
| orders at best ask | 1,999,989 | 27.6 | 1 | 9 | 26 | 43 | 61 | 71 | 100 |
| top-5 bid qty | 1,999,989 | 1,949 | 1,234 | 1,661 | 1,958 | 2,250 | 2,627 | 2,880 | 3,652 |
| top-5 ask qty | 1,999,989 | 1,943 | 1,243 | 1,659 | 1,955 | 2,236 | 2,619 | 2,873 | 3,534 |

**Mid changes**: after 0.5% of events, 1.62 per second; size 0.5 tick 99.5%, 1 tick 0.5%, 1.5-2 ticks 0.0%, more 0.0%.

**Mid-change volatility by horizon** (regular time grid; VR = variance ratio vs. 1 s, <1 mean reversion, >1 trending)

| horizon | samples | std (ticks) | mean abs | P(no change) | std / sqrt(s) | VR vs 1 s |
|---|---:|---:|---:|---:|---:|---:|
| 10 ms | 574,986 | 0.062 | 0.008 | 98.5% | 0.621 | 1.56 |
| 100 ms | 57,498 | 0.175 | 0.059 | 88.6% | 0.553 | 1.24 |
| 1,000 ms | 5,749 | 0.497 | 0.354 | 41.9% | 0.497 | 1.00 |
| 10,000 ms | 574 | 1.636 | 1.289 | 11.8% | 0.517 | 1.08 |
| 60,000 ms | 95 | 4.301 | 3.621 | 4.2% | 0.555 | 1.25 |

**Successive mid changes** (9,320): autocorrelation lag 1..5 = -0.250, +0.097, -0.031, +0.014, -0.004; P(next change reverses) = 0.626 (0.5 = random walk, >0.5 bounce / mean reversion).

**Trade-sign autocorrelation** (33,718 trading events): lag 1 +0.000, lag 2 +0.004, lag 5 -0.007, lag 10 -0.010, lag 50 +0.001; buyer-initiated share 50.0%.

## Signals vs. future mid change

Label: mid(t + 5 ms + h) - mid(t + 5 ms), mid = last two-sided mid of the book after all events before that time (what an order sent at t would meet). 1,996,376 events with a two-sided book and a known label, cut into 8 equal-event chunks. Signals use rows <= t only.

Label std (ticks): 10 ms 0.062 (1.5% non-zero), 100 ms 0.175 (11.5% non-zero), 1000 ms 0.502 (58.6% non-zero), 10000 ms 1.653 (87.4% non-zero).

Signals: `imb_l1` = (b1-a1)/(b1+a1); `imb_top5` same over 5 price levels; `micro_minus_mid` = microprice - mid (ticks); `flow_Nev` = signed traded volume (+ = buyer-initiated) of the last N events; `flow_100ms` / `flow_1s` = over the last T.

**Mean Spearman IC over chunks** (t-stat across chunks, chunks agreeing in sign / chunks)

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.121 (t +91.8, 8/8) | +0.331 (t +129.6, 8/8) | +0.327 (t +124.0, 8/8) | +0.101 (t +15.4, 8/8) |
| `imb_top5` | +0.027 (t +30.9, 8/8) | +0.098 (t +87.4, 8/8) | +0.168 (t +20.7, 8/8) | +0.046 (t +2.2, 7/8) |
| `micro_minus_mid` | +0.123 (t +84.6, 8/8) | +0.333 (t +129.9, 8/8) | +0.334 (t +111.8, 8/8) | +0.104 (t +14.2, 8/8) |
| `flow_10ev` | -0.016 (t -12.9, 8/8) | -0.024 (t -12.9, 8/8) | -0.023 (t -8.4, 8/8) | -0.008 (t -3.6, 7/8) |
| `flow_100ev` | -0.021 (t -25.7, 8/8) | -0.049 (t -28.5, 8/8) | -0.059 (t -13.0, 8/8) | -0.026 (t -4.5, 8/8) |
| `flow_100ms` | -0.019 (t -11.1, 8/8) | -0.035 (t -8.7, 8/8) | -0.040 (t -8.2, 8/8) | -0.015 (t -4.4, 8/8) |
| `flow_1s` | -0.017 (t -15.4, 8/8) | -0.043 (t -10.8, 8/8) | -0.055 (t -7.7, 8/8) | -0.028 (t -2.3, 6/8) |

**Pooled-chunk mean Pearson IC**

| signal | 10 ms | 100 ms | 1000 ms | 10000 ms |
|---|---:|---:|---:|---:|
| `imb_l1` | +0.123 | +0.337 | +0.331 | +0.102 |
| `imb_top5` | +0.025 | +0.093 | +0.166 | +0.047 |
| `micro_minus_mid` | +0.124 | +0.331 | +0.336 | +0.105 |
| `flow_10ev` | -0.029 | -0.040 | -0.031 | -0.011 |
| `flow_100ev` | -0.023 | -0.051 | -0.054 | -0.022 |
| `flow_100ms` | -0.026 | -0.045 | -0.046 | -0.017 |
| `flow_1s` | -0.017 | -0.040 | -0.050 | -0.024 |

**Spearman IC per chunk, h = 10 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.125 | +0.117 | +0.119 | +0.114 | +0.123 | +0.121 | +0.123 | +0.122 |
| `imb_top5` | +0.030 | +0.026 | +0.031 | +0.028 | +0.024 | +0.025 | +0.028 | +0.028 |
| `micro_minus_mid` | +0.128 | +0.118 | +0.122 | +0.116 | +0.124 | +0.124 | +0.128 | +0.124 |
| `flow_10ev` | -0.018 | -0.015 | -0.015 | -0.020 | -0.019 | -0.012 | -0.020 | -0.011 |
| `flow_100ev` | -0.022 | -0.025 | -0.018 | -0.022 | -0.022 | -0.019 | -0.019 | -0.019 |
| `flow_100ms` | -0.025 | -0.021 | -0.014 | -0.021 | -0.021 | -0.015 | -0.025 | -0.013 |
| `flow_1s` | -0.019 | -0.024 | -0.017 | -0.015 | -0.014 | -0.016 | -0.017 | -0.016 |

**Spearman IC per chunk, h = 100 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.334 | +0.335 | +0.330 | +0.316 | +0.342 | +0.328 | +0.331 | +0.331 |
| `imb_top5` | +0.098 | +0.097 | +0.101 | +0.101 | +0.093 | +0.093 | +0.100 | +0.097 |
| `micro_minus_mid` | +0.337 | +0.335 | +0.332 | +0.319 | +0.344 | +0.331 | +0.337 | +0.332 |
| `flow_10ev` | -0.027 | -0.025 | -0.020 | -0.029 | -0.024 | -0.020 | -0.033 | -0.017 |
| `flow_100ev` | -0.048 | -0.059 | -0.047 | -0.045 | -0.049 | -0.052 | -0.043 | -0.051 |
| `flow_100ms` | -0.046 | -0.046 | -0.022 | -0.043 | -0.020 | -0.025 | -0.045 | -0.035 |
| `flow_1s` | -0.045 | -0.063 | -0.048 | -0.042 | -0.027 | -0.052 | -0.034 | -0.036 |

**Spearman IC per chunk, h = 1000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.325 | +0.337 | +0.335 | +0.323 | +0.330 | +0.329 | +0.319 | +0.315 |
| `imb_top5` | +0.172 | +0.195 | +0.165 | +0.172 | +0.166 | +0.120 | +0.191 | +0.161 |
| `micro_minus_mid` | +0.330 | +0.342 | +0.345 | +0.322 | +0.341 | +0.334 | +0.331 | +0.324 |
| `flow_10ev` | -0.027 | -0.031 | -0.020 | -0.024 | -0.006 | -0.024 | -0.030 | -0.022 |
| `flow_100ev` | -0.065 | -0.073 | -0.057 | -0.061 | -0.034 | -0.074 | -0.055 | -0.053 |
| `flow_100ms` | -0.044 | -0.057 | -0.028 | -0.039 | -0.014 | -0.045 | -0.052 | -0.041 |
| `flow_1s` | -0.071 | -0.052 | -0.074 | -0.071 | -0.036 | -0.076 | -0.027 | -0.034 |

**Spearman IC per chunk, h = 10000 ms**

| signal | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imb_l1` | +0.087 | +0.096 | +0.106 | +0.113 | +0.137 | +0.088 | +0.103 | +0.077 |
| `imb_top5` | -0.034 | +0.014 | +0.038 | +0.142 | +0.044 | +0.001 | +0.121 | +0.043 |
| `micro_minus_mid` | +0.077 | +0.102 | +0.109 | +0.117 | +0.143 | +0.090 | +0.110 | +0.087 |
| `flow_10ev` | -0.014 | -0.001 | -0.003 | -0.012 | -0.006 | -0.012 | -0.013 | +0.001 |
| `flow_100ev` | -0.048 | -0.009 | -0.026 | -0.024 | -0.017 | -0.052 | -0.016 | -0.012 |
| `flow_100ms` | -0.026 | -0.005 | -0.003 | -0.016 | -0.012 | -0.023 | -0.027 | -0.007 |
| `flow_1s` | -0.071 | -0.031 | -0.040 | +0.001 | -0.013 | -0.078 | +0.013 | -0.003 |
