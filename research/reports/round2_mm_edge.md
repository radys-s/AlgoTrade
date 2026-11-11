# Market-making edge - `data\round2.csv`

Generated 2026-11-07 19:20 by `research.mm_edge` from `data\features\round2.npz`. Prices in ticks. 2,193,064 events over 6,000 s; 51,317 taker fills of the touch (513.2/min). Conditions from the book 5000 us before each fill.

## Summary

- **VIABLE unconditionally: a front-of-queue touch maker earns +0.069 ticks/fill at 1000 ms (t +4.4, 6/8 chunks > 0).**
- front-of-queue touch maker edge: 100 ms +0.313, 1000 ms +0.069, 5000 ms +0.054, 30000 ms +0.064 ticks/fill
- back-of-queue joiner (filled only on depletion): 100 ms -0.368, 1000 ms -0.570, 5000 ms -0.523, 30000 ms -0.260
- unfiltered edge at 1000 ms positive in 6/8 chunks
- queue position, edge at 1000 ms with Q lots ahead: Q=0 +0.069, Q=5 +0.050, Q=20 -0.024, Q=50 -0.116, Q=100 -0.239, Q=200 -0.445, Q=500 -0.693, Q=1000 -1.160
- quote a side only when **flow toward maker, 100 ms <= 0**: edge +0.250 ticks/fill on 69.8% of fills (t +13.8, 8/8 chunks > 0) vs +0.069 unfiltered.
- quote a side only when **own touch queue (lots) >= 124**: edge +0.131 ticks/fill on 80.1% of fills (t +9.4, 7/8 chunks > 0) vs +0.069 unfiltered.
- quote a side only when **flow toward maker, 1 s < 0**: edge +0.192 ticks/fill on 48.9% of fills (t +8.9, 7/8 chunks > 0) vs +0.069 unfiltered.
- quote a side only when **spread (ticks) <= 1**: edge +0.109 ticks/fill on 80.8% of fills (t +9.6, 7/8 chunks > 0) vs +0.069 unfiltered.

## Touch-maker markout: effective = realized + impact

Per fill event, in ticks. effective = half-spread captured vs the pre-fill mid; realized = net edge per lot (markout); impact = adverse mid move (taker-signed). vw = weighted by taker volume.

| maker | h | n | effective half | realized (edge) | impact | t | edge vw | edge > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front of queue: every touch fill | 100 ms | 51,317 | +0.632 | +0.313 | +0.319 | +63.0 | -0.107 | 79.3% |
|  | 1000 ms | 51,310 | +0.632 | +0.069 | +0.563 | +4.4 | -0.625 | 57.7% |
|  | 5000 ms | 51,293 | +0.632 | +0.054 | +0.578 | +1.4 | -0.669 | 52.2% |
|  | 30000 ms | 51,100 | +0.633 | +0.064 | +0.569 | +0.6 | -0.689 | 50.7% |
| back of queue: touch depleted | 100 ms | 9,110 | +0.705 | -0.368 | +1.074 | -20.2 | -0.959 | 29.9% |
|  | 1000 ms | 9,109 | +0.705 | -0.570 | +1.275 | -10.5 | -1.453 | 40.6% |
|  | 5000 ms | 9,108 | +0.705 | -0.523 | +1.228 | -4.1 | -1.337 | 46.4% |
|  | 30000 ms | 9,085 | +0.706 | -0.260 | +0.966 | -0.9 | -1.320 | 48.9% |

Queue position (cancels ahead ignored): a maker with Q market lots ahead at the touch is filled only when the taker trades more than Q there.

| lots ahead Q | fills | of all | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t 1000 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 51,317 | 100.0% | +0.313 | +0.069 | +0.054 | +0.064 | +4.4 |
| 5 | 48,108 | 93.7% | +0.311 | +0.050 | +0.031 | +0.042 | +3.2 |
| 20 | 37,630 | 73.3% | +0.288 | -0.024 | -0.049 | -0.077 | -1.5 |
| 50 | 25,128 | 49.0% | +0.248 | -0.116 | -0.184 | -0.216 | -6.0 |
| 100 | 15,184 | 29.6% | +0.198 | -0.239 | -0.296 | -0.418 | -10.0 |
| 200 | 7,423 | 14.5% | +0.120 | -0.445 | -0.495 | -0.487 | -13.0 |
| 500 | 1,936 | 3.8% | +0.007 | -0.693 | -1.065 | -1.651 | -8.8 |
| 1000 | 428 | 0.8% | -0.083 | -1.160 | -2.141 | -0.659 | -11.1 |


## Filter rules (auto-generated, h = 1000 ms)

Each pre-fill condition: the one-sided threshold (keep buckets at/above or at/below b) that maximises the total edge of the fills it keeps; then the best second condition on top of the best rule. In-sample selection: expect some shrinkage out of sample.

| rule: quote a side only when | fills kept | edge | t | total ticks | chunks > 0 |
|---|---:|---:|---:|---:|---:|
| no filter (quote always) | 100.0% | +0.069 | +4.4 | +3,562 | 6/8 |
| flow toward maker, 100 ms <= 0 | 69.8% | +0.250 | +13.8 | +8,950 | 8/8 |
| own touch queue (lots) >= 124 | 80.1% | +0.131 | +9.4 | +5,394 | 7/8 |
| flow toward maker, 1 s < 0 | 48.9% | +0.192 | +8.9 | +4,824 | 7/8 |
| spread (ticks) <= 1 | 80.8% | +0.109 | +9.6 | +4,528 | 7/8 |
| events in last 1 s < 391 | 79.9% | +0.095 | +5.4 | +3,895 | 6/8 |
| L1 imbalance, maker view < 0.68 | 79.8% | +0.093 | +5.4 | +3,810 | 6/8 |

Rules:

- quote a side only when **flow toward maker, 100 ms <= 0**: edge +0.250 ticks/fill on 69.8% of fills (t +13.8, 8/8 chunks > 0) vs +0.069 unfiltered.
- quote a side only when **own touch queue (lots) >= 124**: edge +0.131 ticks/fill on 80.1% of fills (t +9.4, 7/8 chunks > 0) vs +0.069 unfiltered.
- quote a side only when **flow toward maker, 1 s < 0**: edge +0.192 ticks/fill on 48.9% of fills (t +8.9, 7/8 chunks > 0) vs +0.069 unfiltered.
- quote a side only when **spread (ticks) <= 1**: edge +0.109 ticks/fill on 80.8% of fills (t +9.6, 7/8 chunks > 0) vs +0.069 unfiltered.

**Verdict:** VIABLE unconditionally: a front-of-queue touch maker earns +0.069 ticks/fill at 1000 ms (t +4.4, 6/8 chunks > 0).

## Net edge conditioned on what is visible before the fill (h = 1000 ms)

Edge columns = mean markout in ticks at each horizon; t and stability at 1000 ms. Rows marked (post) are not usable as filters.

### spread (ticks)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41,481 | 80.8% | +0.307 | +0.109 | +0.084 | +0.041 | +9.6 | 7/8 |
| 2 | 7,221 | 14.1% | +0.319 | -0.165 | -0.234 | +0.306 | -2.8 | 5/8 |
| 3 | 2,117 | 4.1% | +0.432 | +0.122 | +0.037 | -0.630 | +0.8 | 7/8 |
| 4 | 258 | 0.5% | +0.008 | -0.089 | +1.122 | +0.326 | -0.1 | 2/7 |
| >= 5 | 222 | 0.4% | +0.507 | -0.063 | +2.401 | +3.351 | -0.0 | 5/6 |

### L1 imbalance, maker view

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < -0.71 | 10,203 | 19.9% | +0.090 | +0.046 | +0.074 | +0.469 | +1.2 | 5/8 |
| [-0.71, -0.26) | 10,238 | 20.0% | +0.340 | +0.191 | +0.209 | +0.177 | +5.9 | 8/8 |
| [-0.26, 0.21) | 10,364 | 20.2% | +0.395 | +0.087 | +0.061 | +0.248 | +2.5 | 6/8 |
| [0.21, 0.68) | 10,155 | 19.8% | +0.382 | +0.048 | -0.086 | -0.159 | +1.5 | 4/8 |
| >= 0.68 | 10,357 | 20.2% | +0.359 | -0.024 | +0.011 | -0.413 | -0.6 | 5/8 |

### own touch queue (lots)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 124 | 10,228 | 19.9% | -0.027 | -0.179 | -0.139 | +0.070 | -3.2 | 6/8 |
| [124, 416) | 10,283 | 20.0% | +0.208 | +0.004 | +0.063 | +0.059 | +0.1 | 4/8 |
| [416, 869) | 10,279 | 20.0% | +0.394 | +0.118 | +0.259 | +0.512 | +4.1 | 7/8 |
| [869, 1633) | 10,259 | 20.0% | +0.474 | +0.191 | +0.036 | -0.279 | +9.2 | 7/8 |
| >= 1633 | 10,268 | 20.0% | +0.517 | +0.212 | +0.048 | -0.044 | +12.4 | 8/8 |

### flow toward maker, 100 ms

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 14,542 | 28.3% | +0.462 | +0.464 | +0.526 | +0.547 | +16.0 | 8/8 |
| none (0) | 21,274 | 41.5% | +0.358 | +0.104 | -0.011 | +0.092 | +4.5 | 7/8 |
| toward maker (> 0) | 15,501 | 30.2% | +0.114 | -0.348 | -0.300 | -0.428 | -11.1 | 8/8 |

### flow toward maker, 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 25,075 | 48.9% | +0.404 | +0.192 | +0.321 | +0.361 | +8.9 | 7/8 |
| none (0) | 76 | 0.1% | -0.822 | -3.257 | -3.164 | +1.546 | -2.0 | 6/8 |
| toward maker (> 0) | 26,166 | 51.0% | +0.230 | -0.039 | -0.193 | -0.225 | -1.7 | 4/8 |

### events in last 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 345 | 9,902 | 19.3% | +0.313 | +0.133 | +0.128 | +0.275 | +2.7 | 7/8 |
| [345, 361) | 10,132 | 19.7% | +0.346 | +0.094 | -0.016 | -0.069 | +3.3 | 7/8 |
| [361, 375) | 10,559 | 20.6% | +0.333 | +0.084 | +0.114 | +0.099 | +2.9 | 5/8 |
| [375, 391) | 10,418 | 20.3% | +0.314 | +0.071 | +0.096 | -0.067 | +2.3 | 6/8 |
| >= 391 | 10,306 | 20.1% | +0.260 | -0.032 | -0.053 | +0.089 | -0.9 | 4/8 |

### maker side

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bid (maker buys) | 25,790 | 50.3% | +0.299 | -0.032 | -0.385 | -2.384 | -1.4 | 4/8 |
| ask (maker sells) | 25,527 | 49.7% | +0.328 | +0.172 | +0.497 | +2.538 | +7.8 | 7/8 |

### taker size (lots, post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 18 | 10,063 | 19.6% | +0.545 | +0.578 | +0.608 | +0.653 | +12.9 | 8/8 |
| [18, 41) | 10,315 | 20.1% | +0.471 | +0.284 | +0.298 | +0.245 | +8.4 | 8/8 |
| [41, 82) | 10,253 | 20.0% | +0.398 | +0.162 | +0.123 | +0.430 | +5.0 | 7/8 |
| [82, 184) | 10,420 | 20.3% | +0.273 | -0.003 | -0.057 | -0.315 | -0.1 | 4/8 |
| >= 184 | 10,266 | 20.0% | -0.115 | -0.662 | -0.692 | -0.676 | -20.0 | 8/8 |

### taker vs touch (post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| partial (touch remains) | 42,207 | 82.2% | +0.461 | +0.207 | +0.178 | +0.134 | +13.6 | 8/8 |
| depleted exactly | 1,944 | 3.8% | +0.613 | +0.988 | +1.067 | +1.614 | +6.9 | 8/8 |
| swept beyond | 7,166 | 14.0% | -0.635 | -0.993 | -0.955 | -0.769 | -17.7 | 8/8 |

### Strongest buckets (|t|)

| condition | bucket | n | edge | t | same sign |
|---|---:|---:|---:|---:|---:|
| taker size (lots, post-fill) (post) | >= 184 | 10,266 | -0.662 | -20.0 | 8/8 |
| taker vs touch (post-fill) (post) | swept beyond | 7,166 | -0.993 | -17.7 | 8/8 |
| flow toward maker, 100 ms | against (< 0) | 14,542 | +0.464 | +16.0 | 8/8 |
| taker vs touch (post-fill) (post) | partial (touch remains) | 42,207 | +0.207 | +13.6 | 8/8 |
| taker size (lots, post-fill) (post) | < 18 | 10,063 | +0.578 | +12.9 | 8/8 |
| own touch queue (lots) | >= 1633 | 10,268 | +0.212 | +12.4 | 8/8 |
| flow toward maker, 100 ms | toward maker (> 0) | 15,501 | -0.348 | -11.1 | 8/8 |
| spread (ticks) | 1 | 41,481 | +0.109 | +9.6 | 7/8 |
| own touch queue (lots) | [869, 1633) | 10,259 | +0.191 | +9.2 | 7/8 |
| flow toward maker, 1 s | against (< 0) | 25,075 | +0.192 | +8.9 | 7/8 |

## Simulator specifics: queues at the touch

| side | single-order touch (events) | single-order touch (time) | median orders at touch | median lots at touch |
|---|---:|---:|---:|---:|
| bid | 7.7% | 7.9% | 12 | 694 |
| ask | 7.3% | 7.5% | 13 | 723 |

Queue ahead of a newly joined order (market limit inserts that join the existing touch price):

| side | joins | lots ahead p25/p50/p75/p90 | orders ahead p25/p50/p75/p90 |
|---|---:|---:|---:|
| bid | 273,092 | 432 / 986 / 1794 / 2937 | 10 / 18 / 26 / 33 |
| ask | 279,570 | 441 / 1009 / 1810 / 2932 | 10 / 18 / 27 / 34 |

Touch depletion (a joiner at the back of the queue is filled only when the taker empties the level):

| side | taker fills | fills that deplete the touch | depletions / min | touch retreats | retreat by trade | retreat by cancel |
|---|---:|---:|---:|---:|---:|---:|
| bid | 25,790 | 17.9% | 46.3 | 7,189 | 64.4% | 35.6% |
| ask | 25,527 | 17.6% | 45.1 | 6,957 | 64.9% | 35.1% |

At taker fills the touch held a single order before the fill in 9.2% of cases.

Method: see the `research/mm_edge.py` docstring. t-stats assume independent fills (optimistic in bursts).
