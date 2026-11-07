# Market-making edge - `data\sample.csv`

Generated 2026-11-07 12:07 by `research.mm_edge` from `data\features\sample.npz`. Prices in ticks. 53,758 events over 3,599 s; 604 taker fills of the touch (10.1/min). Conditions from the book 5000 us before each fill.

## Summary

- **NOT VIABLE at the touch: unfiltered edge -2.056 ticks/fill at 1000 ms, best filter (flow toward maker, 1 s < 0) +1.500 on 6.3% of fills; no rule is positive with t >= 2 in >= 3/4 of chunks (each chunk >= 5 fills).**
- front-of-queue touch maker edge: 100 ms +1.008, 1000 ms -2.056, 5000 ms -9.053, 30000 ms -5.502 ticks/fill
- back-of-queue joiner: no taker depleted the touch on this tape (a joiner at the back is never filled)
- unfiltered edge at 1000 ms positive in 0/8 chunks
- queue position, edge at 1000 ms with Q lots ahead: Q=0 -2.056, Q=5 -2.052, Q=20 -2.227
- quote a side only when **flow toward maker, 1 s < 0**: edge +1.500 ticks/fill on 6.3% of fills (t +4.0, 4/4 chunks > 0) vs -2.056 unfiltered.
- strategy own maker fills, edge/lot: 100 ms +0.326, 1000 ms -1.592, 5000 ms -5.895, 30000 ms -4.516

## Touch-maker markout: effective = realized + impact

Per fill event, in ticks. effective = half-spread captured vs the pre-fill mid; realized = net edge per lot (markout); impact = adverse mid move (taker-signed). vw = weighted by taker volume.

| maker | h | n | effective half | realized (edge) | impact | t | edge vw | edge > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front of queue: every touch fill | 100 ms | 604 | +1.000 | +1.008 | -0.008 | +51.6 | +0.999 | 90.2% |
|  | 1000 ms | 604 | +1.000 | -2.056 | +3.056 | -18.8 | -2.059 | 13.9% |
|  | 5000 ms | 604 | +1.000 | -9.053 | +10.053 | -24.6 | -9.113 | 12.6% |
|  | 30000 ms | 598 | +1.000 | -5.502 | +6.502 | -7.2 | -5.528 | 38.3% |
| back of queue: touch depleted | 100 ms | 0 | - | - | - | - | - | - |
|  | 1000 ms | 0 | - | - | - | - | - | - |
|  | 5000 ms | 0 | - | - | - | - | - | - |
|  | 30000 ms | 0 | - | - | - | - | - | - |

Queue position (cancels ahead ignored): a maker with Q market lots ahead at the touch is filled only when the taker trades more than Q there.

| lots ahead Q | fills | of all | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t 1000 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 604 | 100.0% | +1.008 | -2.056 | -9.053 | -5.502 | -18.8 |
| 5 | 556 | 92.1% | +1.005 | -2.052 | -9.126 | -5.250 | -18.4 |
| 20 | 66 | 10.9% | +0.939 | -2.227 | -9.106 | -3.561 | -6.5 |


## Filter rules (auto-generated, h = 1000 ms)

Each pre-fill condition: the one-sided threshold (keep buckets at/above or at/below b) that maximises the total edge of the fills it keeps; then the best second condition on top of the best rule. In-sample selection: expect some shrinkage out of sample.

| rule: quote a side only when | fills kept | edge | t | total ticks | chunks > 0 |
|---|---:|---:|---:|---:|---:|
| no filter (quote always) | 100.0% | -2.056 | -18.8 | -1,242 | 0/8 |
| flow toward maker, 1 s < 0 | 6.3% | +1.500 | +4.0 | +57 | 4/4 |
| own touch queue (lots) < 16 | 18.2% | -2.191 | -7.5 | -241 | 0/8 |
| events in last 1 s < 8 | 19.2% | -2.112 | -14.1 | -245 | 0/8 |
| L1 imbalance, maker view < -0.24 | 20.4% | -2.089 | -8.1 | -257 | 0/8 |
| flow toward maker, 100 ms <= 0 | 99.3% | -2.040 | -18.6 | -1,224 | 0/8 |
| spread (ticks) >= 2 | 100.0% | -2.056 | -18.8 | -1,242 | 0/8 |
| flow toward maker, 1 s < 0 AND L1 imbalance, maker view < 0.26 | 5.3% | +1.875 | +5.3 | +60 | 2/2 |

Rules:

- quote a side only when **flow toward maker, 1 s < 0**: edge +1.500 ticks/fill on 6.3% of fills (t +4.0, 4/4 chunks > 0) vs -2.056 unfiltered.

**Verdict:** NOT VIABLE at the touch: unfiltered edge -2.056 ticks/fill at 1000 ms, best filter (flow toward maker, 1 s < 0) +1.500 on 6.3% of fills; no rule is positive with t >= 2 in >= 3/4 of chunks (each chunk >= 5 fills).

## Net edge conditioned on what is visible before the fill (h = 1000 ms)

Edge columns = mean markout in ticks at each horizon; t and stability at 1000 ms. Rows marked (post) are not usable as filters.

### spread (ticks)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 604 | 100.0% | +1.008 | -2.056 | -9.053 | -5.502 | -18.8 | 8/8 |

### L1 imbalance, maker view

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < -0.24 | 123 | 20.4% | +1.049 | -2.089 | -8.772 | -4.421 | -8.1 | 8/8 |
| [-0.24, -0.07) | 117 | 19.4% | +1.043 | -1.684 | -8.761 | -6.586 | -6.4 | 8/8 |
| [-0.07, 0.06) | 124 | 20.5% | +0.919 | -2.242 | -9.395 | -5.415 | -9.1 | 8/8 |
| [0.06, 0.26) | 121 | 20.0% | +1.025 | -2.033 | -8.132 | -5.983 | -8.5 | 8/8 |
| >= 0.26 | 119 | 19.7% | +1.008 | -2.218 | -10.210 | -5.137 | -10.4 | 8/8 |

### own touch queue (lots)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 16 | 110 | 18.2% | +1.064 | -2.191 | -9.900 | -6.841 | -7.5 | 8/8 |
| [16, 22) | 122 | 20.2% | +0.967 | -2.000 | -7.664 | -3.727 | -8.0 | 8/8 |
| [22, 28) | 121 | 20.0% | +1.050 | -1.669 | -9.107 | -6.248 | -7.8 | 7/8 |
| [28, 34) | 105 | 17.4% | +1.000 | -2.238 | -9.419 | -3.933 | -8.5 | 8/8 |
| >= 34 | 146 | 24.2% | +0.973 | -2.192 | -9.267 | -6.497 | -10.1 | 8/8 |

### flow toward maker, 100 ms

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 2 | 0.3% | +1.000 | +1.500 | +0.500 | +1.000 | - | 0/0 |
| none (0) | 598 | 99.0% | +1.010 | -2.052 | -9.002 | -5.512 | -18.7 | 8/8 |
| toward maker (> 0) | 4 | 0.7% | +0.750 | -4.500 | -21.500 | -7.250 | -7.0 | 0/0 |

### flow toward maker, 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 38 | 6.3% | +1.184 | +1.500 | -0.921 | +0.158 | +4.0 | 4/4 |
| none (0) | 520 | 86.1% | +1.019 | -2.031 | -8.771 | -5.272 | -19.9 | 8/8 |
| toward maker (> 0) | 46 | 7.6% | +0.739 | -5.283 | -18.957 | -12.911 | -11.9 | 6/6 |

### events in last 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 8 | 116 | 19.2% | +1.009 | -2.112 | -9.112 | -4.591 | -14.1 | 8/8 |
| [8, 16) | 237 | 39.2% | +1.042 | -1.819 | -8.338 | -6.744 | -14.4 | 8/8 |
| [16, 24) | 128 | 21.2% | +0.992 | -1.969 | -9.633 | -3.746 | -7.5 | 8/8 |
| >= 24 | 123 | 20.4% | +0.959 | -2.553 | -9.772 | -5.789 | -6.9 | 8/8 |

### maker side

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bid (maker buys) | 297 | 49.2% | +1.034 | -2.013 | -8.734 | -4.364 | -13.3 | 8/8 |
| ask (maker sells) | 307 | 50.8% | +0.984 | -2.098 | -9.362 | -6.602 | -13.2 | 8/8 |

### taker size (lots, post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 7 | 86 | 14.2% | +1.023 | -2.081 | -8.616 | -5.289 | -6.6 | 8/8 |
| [7, 11) | 153 | 25.3% | +1.026 | -2.118 | -9.490 | -5.303 | -10.3 | 8/8 |
| [11, 14) | 120 | 19.9% | +1.042 | -1.825 | -8.125 | -4.375 | -7.1 | 8/8 |
| [14, 18) | 110 | 18.2% | +0.936 | -2.018 | -8.500 | -4.727 | -8.4 | 8/8 |
| >= 18 | 135 | 22.4% | +1.007 | -2.207 | -10.111 | -7.519 | -9.4 | 8/8 |

### taker vs touch (post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| partial (touch remains) | 604 | 100.0% | +1.008 | -2.056 | -9.053 | -5.502 | -18.8 | 8/8 |

### Strongest buckets (|t|)

| condition | bucket | n | edge | t | same sign |
|---|---:|---:|---:|---:|---:|
| flow toward maker, 1 s | none (0) | 520 | -2.031 | -19.9 | 8/8 |
| spread (ticks) | 2 | 604 | -2.056 | -18.8 | 8/8 |
| taker vs touch (post-fill) (post) | partial (touch remains) | 604 | -2.056 | -18.8 | 8/8 |
| flow toward maker, 100 ms | none (0) | 598 | -2.052 | -18.7 | 8/8 |
| events in last 1 s | [8, 16) | 237 | -1.819 | -14.4 | 8/8 |
| events in last 1 s | < 8 | 116 | -2.112 | -14.1 | 8/8 |
| maker side | bid (maker buys) | 297 | -2.013 | -13.3 | 8/8 |
| maker side | ask (maker sells) | 307 | -2.098 | -13.2 | 8/8 |
| flow toward maker, 1 s | toward maker (> 0) | 46 | -5.283 | -11.9 | 6/6 |
| L1 imbalance, maker view | >= 0.26 | 119 | -2.218 | -10.4 | 8/8 |

## Simulator specifics: queues at the touch

| side | single-order touch (events) | single-order touch (time) | median orders at touch | median lots at touch |
|---|---:|---:|---:|---:|
| bid | 100.0% | 100.0% | 1 | 25 |
| ask | 100.0% | 100.0% | 1 | 25 |

Queue ahead of a newly joined order (market limit inserts that join the existing touch price):

| side | joins | lots ahead p25/p50/p75/p90 | orders ahead p25/p50/p75/p90 |
|---|---:|---:|---:|

Touch depletion (a joiner at the back of the queue is filled only when the taker empties the level):

| side | taker fills | fills that deplete the touch | depletions / min | touch retreats | retreat by trade | retreat by cancel |
|---|---:|---:|---:|---:|---:|---:|
| bid | 297 | 0.0% | 0.0 | 13,288 | 0.0% | 100.0% |
| ask | 307 | 0.0% | 0.0 | 13,288 | 0.0% | 100.0% |

At taker fills the touch held a single order before the fill in 100.0% of cases.

## Strategy check: own maker fills from `logs\mmedge-check\000_sample`

422 maker fills (2,097 lots), 2 taker fills. Markout vs the features (market-only) mid; compare with the touch-maker columns.

| h | fills | effective half | edge / fill | edge / lot | impact | tool: front | tool: back (depleted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 ms | 422 | +0.590 | +0.329 | +0.326 | +0.261 | +1.008 | - |
| 1000 ms | 422 | +0.590 | -1.592 | -1.592 | +2.182 | -2.056 | - |
| 5000 ms | 422 | +0.590 | -5.903 | -5.895 | +6.493 | -9.053 | - |
| 30000 ms | 420 | +0.590 | -4.555 | -4.516 | +5.143 | -5.502 | - |

Method: see the `research/mm_edge.py` docstring. t-stats assume independent fills (optimistic in bursts).
