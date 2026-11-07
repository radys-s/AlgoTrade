# Market-making edge - `data\round1.csv`

Generated 2026-11-07 13:06 by `research.mm_edge` from `data\features\round1.npz`. Prices in ticks. 4,411,336 events over 12,000 s; 94,691 taker fills of the touch (473.5/min). Conditions from the book 5000 us before each fill.

## Summary

- **VIABLE unconditionally: a front-of-queue touch maker earns +0.254 ticks/fill at 1000 ms (t +83.3, 8/8 chunks > 0).**
- front-of-queue touch maker edge: 100 ms +0.412, 1000 ms +0.254, 5000 ms +0.252, 30000 ms +0.268 ticks/fill
- back-of-queue joiner (filled only on depletion): 100 ms -0.194, 1000 ms -0.219, 5000 ms -0.188, 30000 ms -0.170
- unfiltered edge at 1000 ms positive in 8/8 chunks
- queue position, edge at 1000 ms with Q lots ahead: Q=0 +0.254, Q=5 +0.249, Q=20 +0.217, Q=50 +0.164, Q=100 +0.098, Q=200 +0.001, Q=500 -0.213, Q=1000 -0.432
- quote a side only when **L1 imbalance, maker view >= -0.69**: edge +0.334 ticks/fill on 79.9% of fills (t +99.6, 8/8 chunks > 0) vs +0.254 unfiltered.
- quote a side only when **own touch queue (lots) >= 253**: edge +0.329 ticks/fill on 80.0% of fills (t +97.6, 8/8 chunks > 0) vs +0.254 unfiltered.

## Touch-maker markout: effective = realized + impact

Per fill event, in ticks. effective = half-spread captured vs the pre-fill mid; realized = net edge per lot (markout); impact = adverse mid move (taker-signed). vw = weighted by taker volume.

| maker | h | n | effective half | realized (edge) | impact | t | edge vw | edge > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front of queue: every touch fill | 100 ms | 94,690 | +0.534 | +0.412 | +0.122 | +336.5 | +0.203 | 86.3% |
|  | 1000 ms | 94,682 | +0.534 | +0.254 | +0.280 | +83.3 | -0.077 | 61.8% |
|  | 5000 ms | 94,662 | +0.534 | +0.252 | +0.281 | +37.5 | -0.078 | 54.2% |
|  | 30000 ms | 94,457 | +0.534 | +0.268 | +0.266 | +16.6 | -0.056 | 51.9% |
| back of queue: touch depleted | 100 ms | 11,156 | +0.511 | -0.194 | +0.705 | -42.8 | -0.408 | 23.5% |
|  | 1000 ms | 11,156 | +0.511 | -0.219 | +0.730 | -25.1 | -0.506 | 37.7% |
|  | 5000 ms | 11,155 | +0.511 | -0.188 | +0.699 | -9.7 | -0.472 | 45.5% |
|  | 30000 ms | 11,132 | +0.511 | -0.170 | +0.682 | -3.6 | -0.445 | 48.5% |

Queue position (cancels ahead ignored): a maker with Q market lots ahead at the touch is filled only when the taker trades more than Q there.

| lots ahead Q | fills | of all | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t 1000 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 94,691 | 100.0% | +0.412 | +0.254 | +0.252 | +0.268 | +83.3 |
| 5 | 89,540 | 94.6% | +0.413 | +0.249 | +0.248 | +0.258 | +79.4 |
| 20 | 71,459 | 75.5% | +0.406 | +0.217 | +0.217 | +0.222 | +62.0 |
| 50 | 48,952 | 51.7% | +0.389 | +0.164 | +0.165 | +0.159 | +38.8 |
| 100 | 30,594 | 32.3% | +0.369 | +0.098 | +0.098 | +0.104 | +18.5 |
| 200 | 15,533 | 16.4% | +0.330 | +0.001 | +0.006 | +0.035 | +0.1 |
| 500 | 4,156 | 4.4% | +0.227 | -0.213 | -0.222 | -0.163 | -15.1 |
| 1000 | 931 | 1.0% | +0.095 | -0.432 | -0.402 | -0.427 | -14.5 |


## Filter rules (auto-generated, h = 1000 ms)

Each pre-fill condition: the one-sided threshold (keep buckets at/above or at/below b) that maximises the total edge of the fills it keeps; then the best second condition on top of the best rule. In-sample selection: expect some shrinkage out of sample.

| rule: quote a side only when | fills kept | edge | t | total ticks | chunks > 0 |
|---|---:|---:|---:|---:|---:|
| no filter (quote always) | 100.0% | +0.254 | +83.3 | +24,048 | 8/8 |
| L1 imbalance, maker view >= -0.69 | 79.9% | +0.334 | +99.6 | +25,296 | 8/8 |
| own touch queue (lots) >= 253 | 80.0% | +0.329 | +97.6 | +24,903 | 8/8 |
| spread (ticks) <= 3 | 100.0% | +0.254 | +83.3 | +24,047 | 8/8 |
| flow toward maker, 100 ms <= 0 | 72.3% | +0.321 | +90.3 | +21,958 | 8/8 |
| events in last 1 s >= 346 | 80.4% | +0.256 | +75.5 | +19,442 | 8/8 |
| flow toward maker, 1 s <= 0 | 49.1% | +0.403 | +95.1 | +18,750 | 8/8 |

Rules:

- quote a side only when **L1 imbalance, maker view >= -0.69**: edge +0.334 ticks/fill on 79.9% of fills (t +99.6, 8/8 chunks > 0) vs +0.254 unfiltered.
- quote a side only when **own touch queue (lots) >= 253**: edge +0.329 ticks/fill on 80.0% of fills (t +97.6, 8/8 chunks > 0) vs +0.254 unfiltered.

**Verdict:** VIABLE unconditionally: a front-of-queue touch maker earns +0.254 ticks/fill at 1000 ms (t +83.3, 8/8 chunks > 0).

## Net edge conditioned on what is visible before the fill (h = 1000 ms)

Edge columns = mean markout in ticks at each horizon; t and stability at 1000 ms. Rows marked (post) are not usable as filters.

### spread (ticks)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 88,265 | 93.2% | +0.393 | +0.224 | +0.224 | +0.234 | +71.0 | 8/8 |
| 2 | 6,372 | 6.7% | +0.667 | +0.671 | +0.651 | +0.741 | +61.5 | 8/8 |
| 3 | 50 | 0.1% | +0.670 | +0.780 | +0.620 | +1.040 | +7.0 | 6/6 |

### L1 imbalance, maker view

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < -0.69 | 19,018 | 20.1% | +0.120 | -0.066 | -0.033 | -0.030 | -9.7 | 8/8 |
| [-0.69, -0.24) | 18,936 | 20.0% | +0.403 | +0.166 | +0.164 | +0.169 | +24.6 | 8/8 |
| [-0.24, 0.21) | 18,741 | 19.8% | +0.477 | +0.279 | +0.259 | +0.240 | +41.8 | 8/8 |
| [0.21, 0.67) | 19,010 | 20.1% | +0.504 | +0.365 | +0.381 | +0.435 | +54.8 | 8/8 |
| >= 0.67 | 18,986 | 20.1% | +0.556 | +0.527 | +0.492 | +0.527 | +80.6 | 8/8 |

### own touch queue (lots)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 253 | 18,920 | 20.0% | +0.107 | -0.045 | -0.019 | -0.038 | -6.7 | 8/8 |
| [253, 656) | 18,928 | 20.0% | +0.402 | +0.180 | +0.199 | +0.238 | +26.8 | 8/8 |
| [656, 1155) | 18,960 | 20.0% | +0.481 | +0.302 | +0.307 | +0.277 | +45.7 | 8/8 |
| [1155, 1956) | 18,935 | 20.0% | +0.522 | +0.402 | +0.369 | +0.355 | +60.0 | 8/8 |
| >= 1956 | 18,948 | 20.0% | +0.546 | +0.430 | +0.405 | +0.509 | +63.8 | 8/8 |

### flow toward maker, 100 ms

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 25,479 | 26.9% | +0.461 | +0.449 | +0.465 | +0.513 | +78.4 | 8/8 |
| none (0) | 42,963 | 45.4% | +0.401 | +0.245 | +0.240 | +0.235 | +54.6 | 8/8 |
| toward maker (> 0) | 26,249 | 27.7% | +0.381 | +0.080 | +0.067 | +0.086 | +13.7 | 8/8 |

### flow toward maker, 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 46,369 | 49.0% | +0.461 | +0.404 | +0.388 | +0.382 | +95.3 | 8/8 |
| none (0) | 117 | 0.1% | +0.470 | +0.051 | +0.060 | -0.026 | +0.5 | 4/8 |
| toward maker (> 0) | 48,205 | 50.9% | +0.364 | +0.110 | +0.123 | +0.160 | +25.7 | 8/8 |

### events in last 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 346 | 18,604 | 19.6% | +0.414 | +0.248 | +0.230 | +0.282 | +35.3 | 8/8 |
| [346, 361) | 18,973 | 20.0% | +0.420 | +0.264 | +0.272 | +0.316 | +39.1 | 8/8 |
| [361, 374) | 19,120 | 20.2% | +0.414 | +0.260 | +0.268 | +0.277 | +38.9 | 8/8 |
| [374, 389) | 18,549 | 19.6% | +0.412 | +0.259 | +0.277 | +0.233 | +38.2 | 8/8 |
| >= 389 | 19,445 | 20.5% | +0.399 | +0.239 | +0.215 | +0.233 | +35.0 | 8/8 |

### maker side

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bid (maker buys) | 47,353 | 50.0% | +0.413 | +0.263 | +0.278 | +0.329 | +60.8 | 8/8 |
| ask (maker sells) | 47,338 | 50.0% | +0.411 | +0.245 | +0.227 | +0.208 | +57.0 | 8/8 |

### taker size (lots, post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 18 | 17,880 | 18.9% | +0.504 | +0.447 | +0.444 | +0.493 | +65.5 | 8/8 |
| [18, 42) | 19,686 | 20.8% | +0.481 | +0.371 | +0.354 | +0.418 | +56.6 | 8/8 |
| [42, 85) | 19,071 | 20.1% | +0.447 | +0.315 | +0.325 | +0.282 | +47.3 | 8/8 |
| [85, 192) | 19,066 | 20.1% | +0.404 | +0.206 | +0.203 | +0.189 | +31.1 | 8/8 |
| >= 192 | 18,988 | 20.1% | +0.225 | -0.062 | -0.056 | -0.033 | -9.1 | 8/8 |

### taker vs touch (post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| partial (touch remains) | 83,535 | 88.2% | +0.493 | +0.317 | +0.311 | +0.327 | +99.4 | 8/8 |
| depleted exactly | 2,383 | 2.5% | +0.313 | +0.335 | +0.465 | +0.373 | +18.9 | 8/8 |
| swept beyond | 8,773 | 9.3% | -0.332 | -0.370 | -0.366 | -0.318 | -39.4 | 8/8 |

### Strongest buckets (|t|)

| condition | bucket | n | edge | t | same sign |
|---|---:|---:|---:|---:|---:|
| taker vs touch (post-fill) (post) | partial (touch remains) | 83,535 | +0.317 | +99.4 | 8/8 |
| flow toward maker, 1 s | against (< 0) | 46,369 | +0.404 | +95.3 | 8/8 |
| L1 imbalance, maker view | >= 0.67 | 18,986 | +0.527 | +80.6 | 8/8 |
| flow toward maker, 100 ms | against (< 0) | 25,479 | +0.449 | +78.4 | 8/8 |
| spread (ticks) | 1 | 88,265 | +0.224 | +71.0 | 8/8 |
| taker size (lots, post-fill) (post) | < 18 | 17,880 | +0.447 | +65.5 | 8/8 |
| own touch queue (lots) | >= 1956 | 18,948 | +0.430 | +63.8 | 8/8 |
| spread (ticks) | 2 | 6,372 | +0.671 | +61.5 | 8/8 |
| maker side | bid (maker buys) | 47,353 | +0.263 | +60.8 | 8/8 |
| own touch queue (lots) | [1155, 1956) | 18,935 | +0.402 | +60.0 | 8/8 |

## Simulator specifics: queues at the touch

| side | single-order touch (events) | single-order touch (time) | median orders at touch | median lots at touch |
|---|---:|---:|---:|---:|
| bid | 4.8% | 4.8% | 16 | 904 |
| ask | 4.9% | 4.9% | 16 | 902 |

Queue ahead of a newly joined order (market limit inserts that join the existing touch price):

| side | joins | lots ahead p25/p50/p75/p90 | orders ahead p25/p50/p75/p90 |
|---|---:|---:|---:|
| bid | 606,025 | 592 / 1168 / 2008 / 3212 | 13 / 21 / 29 / 35 |
| ask | 602,721 | 596 / 1171 / 2022 / 3210 | 13 / 21 / 29 / 35 |

Touch depletion (a joiner at the back of the queue is filled only when the taker empties the level):

| side | taker fills | fills that deplete the touch | depletions / min | touch retreats | retreat by trade | retreat by cancel |
|---|---:|---:|---:|---:|---:|---:|
| bid | 47,353 | 11.8% | 28.0 | 7,555 | 74.1% | 25.9% |
| ask | 47,338 | 11.8% | 27.9 | 7,577 | 73.7% | 26.3% |

At taker fills the touch held a single order before the fill in 5.0% of cases.

Method: see the `research/mm_edge.py` docstring. t-stats assume independent fills (optimistic in bursts).
