# Market-making edge - `data\sample_v2.csv`

Generated 2026-11-07 12:05 by `research.mm_edge` from `data\features\sample_v2.npz`. Prices in ticks. 2,000,000 events over 5,750 s; 33,718 taker fills of the touch (351.8/min). Conditions from the book 5000 us before each fill.

## Summary

- **VIABLE unconditionally: a front-of-queue touch maker earns +0.773 ticks/fill at 1000 ms (t +258.1, 8/8 chunks > 0).**
- front-of-queue touch maker edge: 100 ms +0.754, 1000 ms +0.773, 5000 ms +0.770, 30000 ms +0.764 ticks/fill
- back-of-queue joiner (filled only on depletion): 100 ms +0.250, 1000 ms +0.391, 5000 ms +0.405, 30000 ms +0.342
- unfiltered edge at 1000 ms positive in 8/8 chunks
- no single pre-fill threshold keeps a positive total edge that beats quoting always.

## Touch-maker markout: effective = realized + impact

Per fill event, in ticks. effective = half-spread captured vs the pre-fill mid; realized = net edge per lot (markout); impact = adverse mid move (taker-signed). vw = weighted by taker volume.

| maker | h | n | effective half | realized (edge) | impact | t | edge vw | edge > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front of queue: every touch fill | 100 ms | 33,716 | +0.789 | +0.754 | +0.034 | +415.5 | +0.730 | 92.1% |
|  | 1000 ms | 33,707 | +0.789 | +0.773 | +0.015 | +258.1 | +0.775 | 81.7% |
|  | 5000 ms | 33,689 | +0.789 | +0.770 | +0.019 | +121.5 | +0.776 | 66.3% |
|  | 30000 ms | 33,540 | +0.788 | +0.764 | +0.024 | +47.7 | +0.756 | 55.7% |
| back of queue: touch depleted | 100 ms | 3,294 | +0.557 | +0.250 | +0.307 | +40.7 | +0.346 | 40.6% |
|  | 1000 ms | 3,294 | +0.557 | +0.391 | +0.165 | +41.2 | +0.476 | 55.8% |
|  | 5000 ms | 3,292 | +0.557 | +0.405 | +0.152 | +20.1 | +0.503 | 54.1% |
|  | 30000 ms | 3,281 | +0.557 | +0.342 | +0.215 | +6.6 | +0.504 | 50.0% |

## Filter rules (auto-generated, h = 1000 ms)

Each pre-fill condition: the one-sided threshold (keep buckets at/above or at/below b) that maximises the total edge of the fills it keeps; then the best second condition on top of the best rule. In-sample selection: expect some shrinkage out of sample.

| rule: quote a side only when | fills kept | edge | t | total ticks | chunks > 0 |
|---|---:|---:|---:|---:|---:|
| no filter (quote always) | 100.0% | +0.773 | +258.1 | +26,060 | 8/8 |
| spread (ticks) in 3 or below | 100.0% | +0.773 | +258.1 | +26,060 | 8/8 |
| own touch queue (lots) in [186, 537) or above | 80.1% | +0.876 | +284.2 | +23,638 | 8/8 |
| L1 imbalance, maker view in [-0.47, -0.14) or above | 80.0% | +0.859 | +272.3 | +23,171 | 8/8 |
| events in last 1 s in [355, 370) or below | 79.7% | +0.783 | +235.6 | +21,041 | 8/8 |
| flow toward maker, 100 ms in none (0) or above | 77.7% | +0.774 | +227.7 | +20,273 | 8/8 |
| flow toward maker, 1 s in none (0) or above | 50.1% | +0.793 | +188.8 | +13,398 | 8/8 |

Rules:

- no single pre-fill threshold keeps a positive total edge that beats quoting always.

**Verdict:** VIABLE unconditionally: a front-of-queue touch maker earns +0.773 ticks/fill at 1000 ms (t +258.1, 8/8 chunks > 0).

## Net edge conditioned on what is visible before the fill (h = 1000 ms)

Edge columns = mean markout in ticks at each horizon; t and stability at 1000 ms. Rows marked (post) are not usable as filters.

### spread (ticks)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14,250 | 42.3% | +0.442 | +0.495 | +0.492 | +0.477 | +109.6 | 8/8 |
| 2 | 19,430 | 57.6% | +0.983 | +0.976 | +0.973 | +0.974 | +295.0 | 8/8 |
| 3 | 38 | 0.1% | +1.132 | +1.289 | +1.303 | +1.447 | +15.8 | 4/4 |

### L1 imbalance, maker view

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < -0.47 | 6,736 | 20.0% | +0.475 | +0.429 | +0.431 | +0.405 | +65.3 | 8/8 |
| [-0.47, -0.14) | 6,687 | 19.8% | +0.818 | +0.743 | +0.705 | +0.753 | +121.1 | 8/8 |
| [-0.14, 0.12) | 6,718 | 19.9% | +0.895 | +0.891 | +0.882 | +0.889 | +150.2 | 8/8 |
| [0.12, 0.46) | 6,844 | 20.3% | +0.885 | +0.967 | +1.018 | +0.974 | +157.1 | 8/8 |
| >= 0.46 | 6,733 | 20.0% | +0.697 | +0.833 | +0.809 | +0.798 | +124.9 | 8/8 |

### own touch queue (lots)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 186 | 6,725 | 19.9% | +0.355 | +0.360 | +0.372 | +0.344 | +56.5 | 8/8 |
| [186, 537) | 6,760 | 20.0% | +0.601 | +0.598 | +0.589 | +0.594 | +95.4 | 8/8 |
| [537, 965) | 6,733 | 20.0% | +0.830 | +0.825 | +0.787 | +0.767 | +142.1 | 8/8 |
| [965, 1379) | 6,752 | 20.0% | +0.969 | +0.980 | +0.967 | +0.994 | +179.2 | 8/8 |
| >= 1379 | 6,748 | 20.0% | +1.016 | +1.102 | +1.134 | +1.123 | +207.7 | 8/8 |

### flow toward maker, 100 ms

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 7,512 | 22.3% | +0.768 | +0.771 | +0.796 | +0.851 | +121.6 | 8/8 |
| none (0) | 18,760 | 55.6% | +0.736 | +0.753 | +0.748 | +0.750 | +186.4 | 8/8 |
| toward maker (> 0) | 7,446 | 22.1% | +0.787 | +0.826 | +0.800 | +0.712 | +132.4 | 8/8 |

### flow toward maker, 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| against (< 0) | 16,811 | 49.9% | +0.746 | +0.753 | +0.759 | +0.765 | +176.5 | 8/8 |
| none (0) | 149 | 0.4% | +0.698 | +0.772 | +0.805 | +1.107 | +15.4 | 8/8 |
| toward maker (> 0) | 16,758 | 49.7% | +0.764 | +0.793 | +0.781 | +0.761 | +188.3 | 8/8 |

### events in last 1 s

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 326 | 6,464 | 19.2% | +0.789 | +0.808 | +0.821 | +0.757 | +121.4 | 8/8 |
| [326, 341) | 6,553 | 19.4% | +0.770 | +0.786 | +0.775 | +0.792 | +119.1 | 8/8 |
| [341, 355) | 7,200 | 21.4% | +0.757 | +0.776 | +0.778 | +0.786 | +119.9 | 8/8 |
| [355, 370) | 6,650 | 19.7% | +0.745 | +0.764 | +0.751 | +0.749 | +111.5 | 8/8 |
| >= 370 | 6,851 | 20.3% | +0.714 | +0.733 | +0.726 | +0.736 | +106.9 | 8/8 |

### maker side

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bid (maker buys) | 16,847 | 50.0% | +0.755 | +0.773 | +0.764 | +0.774 | +182.6 | 8/8 |
| ask (maker sells) | 16,871 | 50.0% | +0.754 | +0.773 | +0.776 | +0.754 | +182.5 | 8/8 |

### taker size (lots, post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 13 | 6,743 | 20.0% | +0.776 | +0.768 | +0.750 | +0.795 | +115.8 | 8/8 |
| [13, 28) | 6,394 | 19.0% | +0.768 | +0.779 | +0.773 | +0.789 | +113.4 | 8/8 |
| [28, 57) | 7,029 | 20.8% | +0.754 | +0.775 | +0.781 | +0.794 | +118.8 | 8/8 |
| [57, 125) | 6,772 | 20.1% | +0.743 | +0.768 | +0.771 | +0.701 | +115.5 | 8/8 |
| >= 125 | 6,780 | 20.1% | +0.731 | +0.777 | +0.774 | +0.743 | +113.8 | 8/8 |

### taker vs touch (post-fill) (post)

| bucket | n | share | edge 100 ms | edge 1000 ms | edge 5000 ms | edge 30000 ms | t | stable /8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| partial (touch remains) | 30,424 | 90.2% | +0.809 | +0.814 | +0.809 | +0.810 | +265.8 | 8/8 |
| depleted exactly | 1,402 | 4.2% | +0.533 | +0.689 | +0.684 | +0.632 | +52.7 | 8/8 |
| swept beyond | 1,892 | 5.6% | +0.040 | +0.170 | +0.199 | +0.127 | +15.7 | 8/8 |

### Strongest buckets (|t|)

| condition | bucket | n | edge | t | same sign |
|---|---:|---:|---:|---:|---:|
| spread (ticks) | 2 | 19,430 | +0.976 | +295.0 | 8/8 |
| taker vs touch (post-fill) (post) | partial (touch remains) | 30,424 | +0.814 | +265.8 | 8/8 |
| own touch queue (lots) | >= 1379 | 6,748 | +1.102 | +207.7 | 8/8 |
| flow toward maker, 1 s | toward maker (> 0) | 16,758 | +0.793 | +188.3 | 8/8 |
| flow toward maker, 100 ms | none (0) | 18,760 | +0.753 | +186.4 | 8/8 |
| maker side | bid (maker buys) | 16,847 | +0.773 | +182.6 | 8/8 |
| maker side | ask (maker sells) | 16,871 | +0.773 | +182.5 | 8/8 |
| own touch queue (lots) | [965, 1379) | 6,752 | +0.980 | +179.2 | 8/8 |
| flow toward maker, 1 s | against (< 0) | 16,811 | +0.753 | +176.5 | 8/8 |
| L1 imbalance, maker view | [0.12, 0.46) | 6,844 | +0.967 | +157.1 | 8/8 |

## Simulator specifics: queues at the touch

| side | single-order touch (events) | single-order touch (time) | median orders at touch | median lots at touch |
|---|---:|---:|---:|---:|
| bid | 5.7% | 5.7% | 26 | 768 |
| ask | 5.4% | 5.4% | 26 | 766 |

Queue ahead of a newly joined order (market limit inserts that join the existing touch price):

| side | joins | lots ahead p25/p50/p75/p90 | orders ahead p25/p50/p75/p90 |
|---|---:|---:|---:|
| bid | 268,454 | 710 / 1153 / 1510 / 1789 | 24 / 39 / 51 / 60 |
| ask | 268,290 | 705 / 1137 / 1498 / 1783 | 24 / 38 / 50 / 60 |

Touch depletion (a joiner at the back of the queue is filled only when the taker empties the level):

| side | taker fills | fills that deplete the touch | depletions / min | touch retreats | retreat by trade | retreat by cancel |
|---|---:|---:|---:|---:|---:|---:|
| bid | 16,847 | 9.9% | 17.4 | 2,339 | 71.3% | 28.7% |
| ask | 16,871 | 9.6% | 17.0 | 2,300 | 70.7% | 29.3% |

At taker fills the touch held a single order before the fill in 5.6% of cases.

Method: see the `research/mm_edge.py` docstring. t-stats assume independent fills (optimistic in bursts).
