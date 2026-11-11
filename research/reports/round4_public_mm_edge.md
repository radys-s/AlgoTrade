# Market-making edge - `..\AlgoTradeArena\data\round4_public.csv`

Generated 2026-11-08 07:19 by `research.mm_edge` from `data\features\round4_public.npz`. Prices in ticks. 2,398,520 events over 6,000 s; 56,213 taker fills of the touch (562.1/min). Conditions from the book 5000 us before each fill.

## Summary

- **VIABLE ONLY WITH FILTER: flow toward maker, 1 s < 0 AND flow toward maker, 100 ms <= 0 -> +0.510 ticks/fill (t +14.1, 8/8 chunks > 0) vs -0.937 unfiltered.**
- front-of-queue touch maker edge: 100 ms +0.150, 1000 ms -0.937, 5000 ms -1.920, 30000 ms -0.471 ticks/fill
- back-of-queue joiner (filled only on depletion): 100 ms -0.707, 1000 ms -2.566, 5000 ms -4.662, 30000 ms -1.769
- unfiltered edge at 1000 ms positive in 0/8 chunks
- queue position, edge at 1000 ms with Q lots ahead: Q=0 -0.937, Q=5 -0.946, Q=20 -0.984, Q=50 -0.971, Q=100 -0.889, Q=200 -0.910, Q=500 -1.238, Q=1000 -1.630
- quote a side only when **flow toward maker, 1 s < 0**: edge +0.281 ticks/fill on 42.8% of fills (t +8.9, 8/8 chunks > 0) vs -0.937 unfiltered.
- quote a side only when **flow toward maker, 100 ms < 0**: edge +0.406 ticks/fill on 26.1% of fills (t +10.1, 8/8 chunks > 0) vs -0.937 unfiltered.
- quote a side only when **own touch queue (lots) >= 628**: edge +0.017 ticks/fill on 40.0% of fills (t +0.9, 5/8 chunks > 0) vs -0.937 unfiltered.

## Touch-maker markout: effective = realized + impact

Per fill event, in ticks. effective = half-spread captured vs the pre-fill mid; realized = net edge per lot (markout); impact = adverse mid move (taker-signed). vw = weighted by taker volume.

| maker | h | n | effective half | realized (edge) | impact | t | edge vw | edge > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front of queue: every touch fill | 100 ms | 56,212 | +0.646 | +0.150 | +0.496 | +18.8 | -0.458 | 70.1% |
|  | 1000 ms | 56,198 | +0.646 | -0.937 | +1.583 | -38.5 | -2.024 | 48.4% |
|  | 5000 ms | 56,148 | +0.646 | -1.920 | +2.566 | -24.3 | -3.196 | 47.4% |
|  | 30000 ms | 55,860 | +0.645 | -0.471 | +1.117 | -2.5 | -1.458 | 49.8% |
| back of queue: touch depleted | 100 ms | 12,784 | +0.677 | -0.707 | +1.383 | -29.6 | -1.456 | 23.9% |
|  | 1000 ms | 12,778 | +0.677 | -2.566 | +3.243 | -36.9 | -3.756 | 33.6% |
|  | 5000 ms | 12,761 | +0.677 | -4.662 | +5.339 | -22.3 | -5.971 | 41.7% |
|  | 30000 ms | 12,650 | +0.677 | -1.769 | +2.446 | -3.9 | -2.818 | 47.4% |

Queue position (cancels ahead ignored): a maker with Q market lots ahead at the touch is filled only when the taker trades more than Q there.

| lots ahead Q | fills | of all | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t 1000 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 56,213 | 100.0% | +0.150 | -0.937 | -1.920 | -0.471 | -38.5 |
| 5 | 52,569 | 93.5% | +0.148 | -0.946 | -1.910 | -0.488 | -38.5 |
| 20 | 40,877 | 72.7% | +0.128 | -0.984 | -1.946 | -0.672 | -37.2 |
| 50 | 26,705 | 47.5% | +0.116 | -0.971 | -1.715 | -0.864 | -31.9 |
| 100 | 15,658 | 27.9% | +0.118 | -0.889 | -1.352 | -1.281 | -25.5 |
| 200 | 6,726 | 12.0% | +0.060 | -0.910 | -1.049 | -0.970 | -19.8 |
| 500 | 1,620 | 2.9% | -0.041 | -1.238 | -1.543 | -0.468 | -15.0 |
| 1000 | 357 | 0.6% | -0.220 | -1.630 | -1.718 | +1.216 | -10.2 |


## Filter rules (auto-generated, h = 1000 ms)

Each pre-fill condition: the one-sided threshold (keep buckets at/above or at/below b) that maximises the total edge of the fills it keeps; then the best second condition on top of the best rule. In-sample selection: expect some shrinkage out of sample.

| rule: quote a side only when | fills kept | edge | t | total ticks | chunks > 0 |
|---|---:|---:|---:|---:|---:|
| no filter (quote always) | 100.0% | -0.937 | -38.5 | -52,672 | 0/8 |
| flow toward maker, 1 s < 0 | 42.8% | +0.281 | +8.9 | +6,755 | 8/8 |
| flow toward maker, 100 ms < 0 | 26.1% | +0.406 | +10.1 | +5,966 | 8/8 |
| own touch queue (lots) >= 628 | 40.0% | +0.017 | +0.9 | +386 | 5/8 |
| events in last 1 s < 366 | 19.4% | -0.064 | -1.0 | -693 | 4/8 |
| L1 imbalance, maker view >= 0.67 | 20.2% | -0.666 | -12.6 | -7,568 | 0/8 |
| spread (ticks) >= 2 | 22.3% | -0.803 | -11.0 | -10,086 | 0/8 |
| flow toward maker, 1 s < 0 AND flow toward maker, 100 ms <= 0 | 32.2% | +0.510 | +14.1 | +9,219 | 8/8 |

Rules:

- quote a side only when **flow toward maker, 1 s < 0**: edge +0.281 ticks/fill on 42.8% of fills (t +8.9, 8/8 chunks > 0) vs -0.937 unfiltered.
- quote a side only when **flow toward maker, 100 ms < 0**: edge +0.406 ticks/fill on 26.1% of fills (t +10.1, 8/8 chunks > 0) vs -0.937 unfiltered.
- quote a side only when **own touch queue (lots) >= 628**: edge +0.017 ticks/fill on 40.0% of fills (t +0.9, 5/8 chunks > 0) vs -0.937 unfiltered.

**Verdict:** VIABLE ONLY WITH FILTER: flow toward maker, 1 s < 0 AND flow toward maker, 100 ms <= 0 -> +0.510 ticks/fill (t +14.1, 8/8 chunks > 0) vs -0.937 unfiltered.

## Net edge conditioned on what is visible before the fill (h = 1000 ms)

Edge columns = mean markout in ticks at each horizon; t and stability at 1000 ms. Rows marked (post) are not usable as filters.

### spread (ticks)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43,636 | 77.6% | +0.089 | -0.975 | -2.083 | -0.693 | -41.9 | 8/8 |
| 2 | 9,759 | 17.4% | +0.340 | -0.681 | -1.665 | +0.028 | -11.0 | 8/8 |
| 3 | 2,202 | 3.9% | +0.412 | -1.314 | -0.448 | +0.978 | -6.8 | 8/8 |
| 4 | 269 | 0.5% | +0.414 | -1.184 | -0.454 | +1.391 | -1.4 | 4/8 |
| >= 5 | 336 | 0.6% | +0.628 | -0.716 | +1.216 | +3.924 | -0.5 | 6/8 |

### L1 imbalance, maker view

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < -0.75 | 11,362 | 20.2% | -0.259 | -1.738 | -3.729 | -1.706 | -26.4 | 8/8 |
| [-0.75, -0.33) | 11,061 | 19.7% | +0.102 | -1.088 | -2.527 | +0.147 | -20.2 | 8/8 |
| [-0.33, 0.18) | 11,350 | 20.2% | +0.263 | -0.551 | -1.089 | +0.275 | -11.6 | 8/8 |
| [0.18, 0.67) | 11,076 | 19.7% | +0.310 | -0.639 | -1.296 | -0.668 | -12.9 | 8/8 |
| >= 0.67 | 11,364 | 20.2% | +0.335 | -0.666 | -0.958 | -0.395 | -12.6 | 8/8 |

### own touch queue (lots)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 73 | 11,204 | 19.9% | -0.460 | -2.464 | -5.033 | -1.298 | -30.0 | 8/8 |
| [73, 246) | 11,271 | 20.1% | -0.052 | -1.860 | -3.855 | -1.190 | -29.1 | 8/8 |
| [246, 628) | 11,242 | 20.0% | +0.293 | -0.400 | -0.645 | -0.193 | -9.1 | 8/8 |
| [628, 1344) | 11,245 | 20.0% | +0.456 | +0.022 | -0.052 | +0.127 | +0.7 | 4/8 |
| >= 1344 | 11,251 | 20.0% | +0.510 | +0.012 | -0.028 | +0.184 | +0.5 | 4/8 |

### flow toward maker, 100 ms

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 14,701 | 26.2% | +0.483 | +0.406 | +0.225 | +0.252 | +10.1 | 8/8 |
| none (0) | 21,433 | 38.1% | +0.183 | -0.726 | -1.530 | -0.650 | -18.9 | 8/8 |
| toward maker (> 0) | 20,079 | 35.7% | -0.129 | -2.146 | -3.906 | -0.811 | -48.5 | 8/8 |

### flow toward maker, 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 24,086 | 42.8% | +0.440 | +0.281 | +0.364 | +0.193 | +8.9 | 8/8 |
| none (0) | 52 | 0.1% | -6.135 | -7.933 | -7.221 | -11.654 | -1.6 | 4/6 |
| toward maker (> 0) | 32,075 | 57.1% | -0.058 | -1.840 | -3.625 | -0.952 | -54.4 | 8/8 |

### events in last 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 366 | 10,902 | 19.4% | +0.297 | -0.064 | +0.015 | -0.469 | -1.0 | 4/8 |
| [366, 388) | 11,183 | 19.9% | +0.348 | -0.006 | -0.098 | +0.148 | -0.2 | 4/8 |
| [388, 411) | 11,302 | 20.1% | +0.315 | -0.213 | -0.584 | -0.234 | -6.5 | 8/8 |
| [411, 447) | 11,426 | 20.3% | +0.143 | -0.964 | -1.879 | +0.004 | -18.9 | 8/8 |
| >= 447 | 11,400 | 20.3% | -0.342 | -3.381 | -6.938 | -1.812 | -45.1 | 8/8 |

### maker side

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bid (maker buys) | 28,269 | 50.3% | +0.137 | -1.079 | -2.534 | -3.706 | -32.3 | 8/8 |
| ask (maker sells) | 27,944 | 49.7% | +0.162 | -0.794 | -1.298 | +2.805 | -22.3 | 8/8 |

### taker size (lots, post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 18 | 10,772 | 19.2% | +0.435 | -0.445 | -1.112 | +0.567 | -6.9 | 8/8 |
| [18, 42) | 11,593 | 20.6% | +0.346 | -0.540 | -1.568 | -0.376 | -10.2 | 8/8 |
| [42, 85) | 11,168 | 19.9% | +0.266 | -0.695 | -1.785 | -0.452 | -12.8 | 8/8 |
| [85, 178) | 11,364 | 20.2% | +0.163 | -0.965 | -2.012 | -0.413 | -19.6 | 8/8 |
| >= 178 | 11,316 | 20.1% | -0.450 | -2.024 | -3.089 | -1.634 | -40.3 | 8/8 |

### taker vs touch (post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| partial (touch remains) | 43,429 | 77.3% | +0.402 | -0.458 | -1.113 | -0.091 | -19.5 | 8/8 |
| depleted exactly | 1,864 | 3.3% | +0.600 | +1.234 | +3.452 | -0.052 | +4.9 | 6/8 |
| swept beyond | 10,920 | 19.4% | -0.930 | -3.216 | -6.049 | -2.065 | -47.6 | 8/8 |

### Strongest buckets (|t|)

| condition | bucket | n | edge | t | same sign |
|---|---:|---:|---:|---:|---:|
| flow toward maker, 1 s | toward maker (> 0) | 32,075 | -1.840 | -54.4 | 8/8 |
| flow toward maker, 100 ms | toward maker (> 0) | 20,079 | -2.146 | -48.5 | 8/8 |
| taker vs touch (post-fill) (post) | swept beyond | 10,920 | -3.216 | -47.6 | 8/8 |
| events in last 1 s | >= 447 | 11,400 | -3.381 | -45.1 | 8/8 |
| spread (ticks) | 1 | 43,636 | -0.975 | -41.9 | 8/8 |
| taker size (lots, post-fill) (post) | >= 178 | 11,316 | -2.024 | -40.3 | 8/8 |
| maker side | bid (maker buys) | 28,269 | -1.079 | -32.3 | 8/8 |
| own touch queue (lots) | < 73 | 11,204 | -2.464 | -30.0 | 8/8 |
| own touch queue (lots) | [73, 246) | 11,271 | -1.860 | -29.1 | 8/8 |
| L1 imbalance, maker view | < -0.75 | 11,362 | -1.738 | -26.4 | 8/8 |

## Simulator specifics: queues at the touch

| side | single-order touch (events) | single-order touch (time) | median orders at touch | median lots at touch |
|---|---:|---:|---:|---:|
| bid | 11.4% | 10.7% | 9 | 494 |
| ask | 10.9% | 10.2% | 9 | 501 |

Queue ahead of a newly joined order (market limit inserts that join the existing touch price):

| side | joins | lots ahead p25/p50/p75/p90 | orders ahead p25/p50/p75/p90 |
|---|---:|---:|---:|
| bid | 251,513 | 341 / 871 / 1658 / 2673 | 8 / 16 / 25 / 32 |
| ask | 262,046 | 331 / 859 / 1652 / 2760 | 8 / 16 / 25 / 32 |

Touch depletion (a joiner at the back of the queue is filled only when the taker empties the level):

| side | taker fills | fills that deplete the touch | depletions / min | touch retreats | retreat by trade | retreat by cancel |
|---|---:|---:|---:|---:|---:|---:|
| bid | 28,269 | 22.5% | 63.7 | 10,049 | 63.4% | 36.6% |
| ask | 27,944 | 23.0% | 64.5 | 9,780 | 66.0% | 34.0% |

At taker fills the touch held a single order before the fill in 14.1% of cases.

Method: see the `research/mm_edge.py` docstring. t-stats assume independent fills (optimistic in bursts).
