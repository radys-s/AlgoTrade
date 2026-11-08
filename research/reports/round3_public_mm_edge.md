# Market-making edge - `data\round3_public.csv`

Generated 2026-11-08 01:07 by `research.mm_edge` from `data\features\round3_public.npz`. Prices in ticks. 2,412,805 events over 6,000 s; 52,955 taker fills of the touch (529.6/min). Conditions from the book 5000 us before each fill.

## Summary

- **VIABLE ONLY WITH FILTER: flow toward maker, 100 ms < 0 -> +0.397 ticks/fill (t +13.6, 8/8 chunks > 0) vs -0.222 unfiltered.**
- front-of-queue touch maker edge: 100 ms +0.265, 1000 ms -0.222, 5000 ms -0.185, 30000 ms +0.073 ticks/fill
- back-of-queue joiner (filled only on depletion): 100 ms -0.518, 1000 ms -1.226, 5000 ms -1.027, 30000 ms -0.528
- unfiltered edge at 1000 ms positive in 0/8 chunks
- queue position, edge at 1000 ms with Q lots ahead: Q=0 -0.222, Q=5 -0.239, Q=20 -0.308, Q=50 -0.385, Q=100 -0.518, Q=200 -0.713, Q=500 -1.132, Q=1000 -1.630
- quote a side only when **flow toward maker, 100 ms < 0**: edge +0.397 ticks/fill on 28.4% of fills (t +13.6, 8/8 chunks > 0) vs -0.222 unfiltered.
- quote a side only when **flow toward maker, 1 s < 0**: edge +0.214 ticks/fill on 47.9% of fills (t +8.8, 7/8 chunks > 0) vs -0.222 unfiltered.
- quote a side only when **own touch queue (lots) >= 682**: edge +0.118 ticks/fill on 40.0% of fills (t +7.2, 7/8 chunks > 0) vs -0.222 unfiltered.
- quote a side only when **events in last 1 s < 390**: edge +0.100 ticks/fill on 39.3% of fills (t +3.9, 7/8 chunks > 0) vs -0.222 unfiltered.

## Touch-maker markout: effective = realized + impact

Per fill event, in ticks. effective = half-spread captured vs the pre-fill mid; realized = net edge per lot (markout); impact = adverse mid move (taker-signed). vw = weighted by taker volume.

| maker | h | n | effective half | realized (edge) | impact | t | edge vw | edge > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front of queue: every touch fill | 100 ms | 52,955 | +0.641 | +0.265 | +0.377 | +42.0 | -0.277 | 74.4% |
|  | 1000 ms | 52,950 | +0.641 | -0.222 | +0.864 | -12.6 | -1.234 | 52.2% |
|  | 5000 ms | 52,921 | +0.641 | -0.185 | +0.826 | -4.1 | -1.189 | 50.2% |
|  | 30000 ms | 52,702 | +0.642 | +0.073 | +0.569 | +0.7 | -0.550 | 50.9% |
| back of queue: touch depleted | 100 ms | 10,958 | +0.677 | -0.518 | +1.195 | -23.5 | -1.176 | 25.3% |
|  | 1000 ms | 10,957 | +0.677 | -1.226 | +1.903 | -23.5 | -2.453 | 36.6% |
|  | 5000 ms | 10,952 | +0.677 | -1.027 | +1.704 | -8.3 | -2.415 | 45.4% |
|  | 30000 ms | 10,922 | +0.677 | -0.528 | +1.206 | -2.0 | -1.398 | 48.5% |

Queue position (cancels ahead ignored): a maker with Q market lots ahead at the touch is filled only when the taker trades more than Q there.

| lots ahead Q | fills | of all | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t 1000 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 52,955 | 100.0% | +0.265 | -0.222 | -0.185 | +0.073 | -12.6 |
| 5 | 49,563 | 93.6% | +0.262 | -0.239 | -0.208 | +0.022 | -13.3 |
| 20 | 38,418 | 72.5% | +0.237 | -0.308 | -0.269 | -0.018 | -15.6 |
| 50 | 25,073 | 47.3% | +0.201 | -0.385 | -0.361 | -0.055 | -16.2 |
| 100 | 14,772 | 27.9% | +0.170 | -0.518 | -0.403 | -0.055 | -18.7 |
| 200 | 6,889 | 13.0% | +0.110 | -0.713 | -0.717 | -0.202 | -19.5 |
| 500 | 1,609 | 3.0% | -0.008 | -1.132 | -0.979 | -0.742 | -16.3 |
| 1000 | 354 | 0.7% | -0.199 | -1.630 | -1.833 | -0.742 | -9.8 |


## Filter rules (auto-generated, h = 1000 ms)

Each pre-fill condition: the one-sided threshold (keep buckets at/above or at/below b) that maximises the total edge of the fills it keeps; then the best second condition on top of the best rule. In-sample selection: expect some shrinkage out of sample.

| rule: quote a side only when | fills kept | edge | t | total ticks | chunks > 0 |
|---|---:|---:|---:|---:|---:|
| no filter (quote always) | 100.0% | -0.222 | -12.6 | -11,780 | 0/8 |
| flow toward maker, 100 ms < 0 | 28.4% | +0.397 | +13.6 | +5,966 | 8/8 |
| flow toward maker, 1 s < 0 | 47.9% | +0.214 | +8.8 | +5,417 | 7/8 |
| own touch queue (lots) >= 682 | 40.0% | +0.118 | +7.2 | +2,494 | 7/8 |
| events in last 1 s < 390 | 39.3% | +0.100 | +3.9 | +2,078 | 7/8 |
| L1 imbalance, maker view >= 0.7 | 19.9% | -0.131 | -3.2 | -1,382 | 3/8 |
| spread (ticks) >= 2 | 21.8% | -0.259 | -4.3 | -2,981 | 2/8 |

Rules:

- quote a side only when **flow toward maker, 100 ms < 0**: edge +0.397 ticks/fill on 28.4% of fills (t +13.6, 8/8 chunks > 0) vs -0.222 unfiltered.
- quote a side only when **flow toward maker, 1 s < 0**: edge +0.214 ticks/fill on 47.9% of fills (t +8.8, 7/8 chunks > 0) vs -0.222 unfiltered.
- quote a side only when **own touch queue (lots) >= 682**: edge +0.118 ticks/fill on 40.0% of fills (t +7.2, 7/8 chunks > 0) vs -0.222 unfiltered.
- quote a side only when **events in last 1 s < 390**: edge +0.100 ticks/fill on 39.3% of fills (t +3.9, 7/8 chunks > 0) vs -0.222 unfiltered.

**Verdict:** VIABLE ONLY WITH FILTER: flow toward maker, 100 ms < 0 -> +0.397 ticks/fill (t +13.6, 8/8 chunks > 0) vs -0.222 unfiltered.

## Net edge conditioned on what is visible before the fill (h = 1000 ms)

Edge columns = mean markout in ticks at each horizon; t and stability at 1000 ms. Rows marked (post) are not usable as filters.

### spread (ticks)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41,427 | 78.2% | +0.232 | -0.212 | -0.194 | +0.049 | -14.0 | 8/8 |
| 2 | 9,096 | 17.2% | +0.376 | -0.283 | -0.198 | +0.138 | -5.5 | 6/8 |
| 3 | 2,093 | 4.0% | +0.611 | -0.095 | +0.144 | -0.020 | -0.7 | 5/8 |
| 4 | 167 | 0.3% | -0.380 | -0.434 | +1.165 | +3.991 | -0.4 | 5/8 |
| >= 5 | 164 | 0.3% | -1.470 | -0.851 | -2.418 | +0.137 | -0.4 | 4/7 |

### L1 imbalance, maker view

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < -0.73 | 10,503 | 19.8% | -0.091 | -0.584 | -0.429 | -0.270 | -13.3 | 8/8 |
| [-0.73, -0.28) | 10,588 | 20.0% | +0.259 | -0.180 | -0.141 | -0.155 | -4.8 | 6/8 |
| [-0.28, 0.22) | 10,627 | 20.1% | +0.356 | -0.061 | -0.127 | +0.393 | -1.7 | 5/8 |
| [0.22, 0.7) | 10,676 | 20.2% | +0.374 | -0.160 | -0.102 | +0.513 | -4.3 | 6/8 |
| >= 0.7 | 10,561 | 19.9% | +0.421 | -0.131 | -0.128 | -0.122 | -3.2 | 5/8 |

### own touch queue (lots)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 85 | 10,514 | 19.9% | -0.201 | -0.895 | -0.619 | -0.124 | -16.3 | 8/8 |
| [85, 298) | 10,654 | 20.1% | +0.146 | -0.374 | -0.470 | -0.453 | -7.5 | 8/8 |
| [298, 682) | 10,581 | 20.0% | +0.373 | -0.083 | -0.116 | +0.020 | -2.5 | 5/8 |
| [682, 1399) | 10,607 | 20.0% | +0.480 | +0.096 | +0.113 | +0.637 | +3.7 | 7/8 |
| >= 1399 | 10,599 | 20.0% | +0.522 | +0.139 | +0.166 | +0.289 | +6.8 | 6/8 |

### flow toward maker, 100 ms

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 15,040 | 28.4% | +0.472 | +0.397 | +0.376 | +0.389 | +13.6 | 8/8 |
| none (0) | 21,112 | 39.9% | +0.264 | -0.216 | -0.215 | +0.097 | -7.8 | 7/8 |
| toward maker (> 0) | 16,803 | 31.7% | +0.080 | -0.784 | -0.649 | -0.239 | -23.2 | 8/8 |

### flow toward maker, 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 25,345 | 47.9% | +0.429 | +0.214 | -0.183 | -0.971 | +8.8 | 7/8 |
| none (0) | 78 | 0.1% | -0.526 | -0.929 | -1.346 | -2.724 | -0.6 | 5/8 |
| toward maker (> 0) | 27,532 | 52.0% | +0.115 | -0.622 | -0.184 | +1.042 | -25.1 | 8/8 |

### events in last 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 368 | 10,538 | 19.9% | +0.309 | +0.079 | +0.191 | +0.632 | +1.7 | 5/8 |
| [368, 390) | 10,283 | 19.4% | +0.380 | +0.121 | +0.063 | +0.172 | +6.2 | 6/8 |
| [390, 414) | 10,809 | 20.4% | +0.348 | -0.003 | +0.158 | +0.356 | -0.1 | 4/8 |
| [414, 450) | 10,592 | 20.0% | +0.251 | -0.381 | -0.374 | -0.370 | -9.4 | 7/8 |
| >= 450 | 10,733 | 20.3% | +0.041 | -0.913 | -0.948 | -0.418 | -17.8 | 8/8 |

### maker side

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bid (maker buys) | 26,570 | 50.2% | +0.261 | -0.183 | -0.138 | -0.309 | -7.3 | 5/8 |
| ask (maker sells) | 26,385 | 49.8% | +0.268 | -0.262 | -0.232 | +0.458 | -10.6 | 7/8 |

### taker size (lots, post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 18 | 10,465 | 19.8% | +0.517 | +0.239 | +0.259 | +0.320 | +5.6 | 7/8 |
| [18, 41) | 10,692 | 20.2% | +0.466 | +0.108 | +0.025 | +0.166 | +2.7 | 7/8 |
| [41, 82) | 10,546 | 19.9% | +0.388 | +0.044 | +0.136 | +0.181 | +1.2 | 6/8 |
| [82, 183) | 10,622 | 20.1% | +0.230 | -0.318 | -0.215 | -0.023 | -8.7 | 7/8 |
| >= 183 | 10,630 | 20.1% | -0.273 | -1.179 | -1.122 | -0.277 | -31.0 | 8/8 |

### taker vs touch (post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| partial (touch remains) | 41,997 | 79.3% | +0.469 | +0.039 | +0.035 | +0.230 | +2.3 | 5/8 |
| depleted exactly | 1,916 | 3.6% | +0.574 | +0.650 | +0.561 | -0.584 | +4.4 | 7/8 |
| swept beyond | 9,042 | 17.1% | -0.749 | -1.623 | -1.363 | -0.516 | -29.9 | 8/8 |

### Strongest buckets (|t|)

| condition | bucket | n | edge | t | same sign |
|---|---:|---:|---:|---:|---:|
| taker size (lots, post-fill) (post) | >= 183 | 10,630 | -1.179 | -31.0 | 8/8 |
| taker vs touch (post-fill) (post) | swept beyond | 9,042 | -1.623 | -29.9 | 8/8 |
| flow toward maker, 1 s | toward maker (> 0) | 27,532 | -0.622 | -25.1 | 8/8 |
| flow toward maker, 100 ms | toward maker (> 0) | 16,803 | -0.784 | -23.2 | 8/8 |
| events in last 1 s | >= 450 | 10,733 | -0.913 | -17.8 | 8/8 |
| own touch queue (lots) | < 85 | 10,514 | -0.895 | -16.3 | 8/8 |
| spread (ticks) | 1 | 41,427 | -0.212 | -14.0 | 8/8 |
| flow toward maker, 100 ms | against (< 0) | 15,040 | +0.397 | +13.6 | 8/8 |
| L1 imbalance, maker view | < -0.73 | 10,503 | -0.584 | -13.3 | 8/8 |
| maker side | ask (maker sells) | 26,385 | -0.262 | -10.6 | 7/8 |

## Simulator specifics: queues at the touch

| side | single-order touch (events) | single-order touch (time) | median orders at touch | median lots at touch |
|---|---:|---:|---:|---:|
| bid | 10.4% | 9.9% | 10 | 528 |
| ask | 10.5% | 9.8% | 9 | 500 |

Queue ahead of a newly joined order (market limit inserts that join the existing touch price):

| side | joins | lots ahead p25/p50/p75/p90 | orders ahead p25/p50/p75/p90 |
|---|---:|---:|---:|
| bid | 257,419 | 368 / 885 / 1677 / 2736 | 8 / 17 / 25 / 32 |
| ask | 256,761 | 340 / 847 / 1621 / 2674 | 8 / 16 / 25 / 32 |

Touch depletion (a joiner at the back of the queue is filled only when the taker empties the level):

| side | taker fills | fills that deplete the touch | depletions / min | touch retreats | retreat by trade | retreat by cancel |
|---|---:|---:|---:|---:|---:|---:|
| bid | 26,570 | 20.9% | 55.6 | 8,815 | 63.1% | 36.9% |
| ask | 26,385 | 20.5% | 54.2 | 8,548 | 63.4% | 36.6% |

At taker fills the touch held a single order before the fill in 11.6% of cases.

Method: see the `research/mm_edge.py` docstring. t-stats assume independent fills (optimistic in bursts).
