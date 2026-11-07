# Order-age features (cancel clock known at insertion)

Code: `research/features_age.py` (numba order-level replay of the tape, ~6 s per tape; run with `uv run --with numba python -X utf8 -m research.features_age build|mech|mechn|eval`). Raw outputs `logs/age/` (`eval.txt`, `*_age.npz`). The replay reproduces the simulator's touch quantity on 100 % of rows (round1 and sample_v2).

## Verdict

**Round1 ridge IC gain at 100 ms is +0.007 (0.3828 -> 0.3900), below the 0.02 bar. Do not port.** At 1000 ms +0.011. HGB gain is smaller still (+0.002 / +0.003). The mechanism is real but almost everything it says is already carried by queue size and count. Only on sample_v2 (uniform 20-40 sizes, different regime) is the gain large (+0.022 at 100 ms, +0.075 at 1000 ms ridge), and the cross-fit round1 -> sample_v2 improves (+0.054 / +0.132) because the age features are scale-free (fractions, not lots).

## Features (touch = best level; age = t - insert time; S = empirical lifetime survival on a 10 ms grid, fitted on the first 1/8 of the tape only)

g200(a) = 1 - S(a+0.2)/S(a), g500 likewise, h(a) = -d ln S/da per bin.
For side s with touch orders i (size r_i, age a_i), Q = sum r_i:

| name | formula (positive = bullish) |
|---|---|
| exp200 (a) | E_s = sum r_i g200(a_i) |
| ag_expd_n (e) | (E_ask - E_bid) / (Q_bid + Q_ask) |
| ag_frac_d (b) | E_ask/Q_ask - E_bid/Q_bid ; ag_frac_s = sum of the two |
| ag_imb_after | ((Q-E)_bid - (Q-E)_ask) / sum : imbalance expected after 200 ms; ag_imb_after12 adds level 2 |
| ag_haz_d (c) | sum_i h(a_i) ask minus bid |
| ag_young_d (d) | share of Q younger than 150 ms, bid minus ask |
| ag_old_d (d) | share of Q older than 650 ms, ask minus bid |
| ag_oldest_d (f) | log1p(oldest age ms) bid minus ask |
| ag_l2_expd_n (g) | same as ag_expd_n on the second level |
| ag_exp500_d_n | as ag_expd_n with g500 |

## Mechanism check: does (a) predict the realised cancel volume of the touch orders in the next 200 ms?

Realised = sum of sizes of today's touch orders whose scheduled cancel time falls in (t, t+200 ms] (includes those hit first by an IOC, whose cancel is then a no-op). Both sides pooled, round1.

| target | R2 queue size only | + count | (a) alone | q+count+(a) | corr((a), real) |
|---|---:|---:|---:|---:|---:|
| lots, 200 ms | 0.461 | 0.467 | 0.465 | 0.474 | 0.68 |
| lots, 500 ms | 0.775 | 0.777 | 0.776 | 0.780 | 0.88 |
| fraction of queue, 200 ms (frac mean 0.41) | count+log q: 0.010 | | 0.018 | 0.035 | 0.13 |
| fraction of queue, 500 ms (mean 0.73) | 0.009 | | 0.002 | 0.013 | 0.05 |

sample_v2, fraction 200 ms: 0.013 / 0.044 / 0.069. Reading: in lots, (a) is just 0.4 x Q (R2 0.46 from size alone). The age profile explains only 2-4 % of the variance of the cancelled FRACTION, because orders are i.i.d. and one order's age changes its P(cancel) only moderately (hazard 0.6 -> 2.1/s), while a touch of many orders averages it out. The cancel timing noise (one lifetime draw per order) dominates.

## Walk-forward ridge IC (expanding, 7 folds, all positive), baseline = 23 features

| set | round1 h100 | e10 | e1 | round1 h1000 | e10 | e1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline23 | 0.3828 | 0.3010 | 0.4520 | 0.3999 | 0.7729 | 0.9712 |
| +ag_young_d (best single) | 0.3850 | 0.3088 | 0.4728 | 0.4069 | 0.8000 | 1.0475 |
| +ag_frac_d | 0.3841 | 0.3062 | 0.4661 | 0.4065 | 0.7943 | 1.0409 |
| +ag_oldest_d | 0.3837 | 0.3067 | 0.4685 | 0.4043 | 0.7905 | 1.0263 |
| +AGE_all (11) | **0.3900** | 0.3110 | 0.4896 | **0.4104** | 0.8089 | 1.0991 |
| v2_all (23+12 cand.) | 0.3892 | 0.2956 | 0.4631 | 0.4042 | 0.7751 | 1.0353 |
| v2_all + AGE | 0.3976 | 0.3050 | 0.5008 | 0.4119 | 0.8087 | 1.1175 |
| HGB baseline23 | 0.4533 | 0.3421 | 0.5625 | 0.4097 | 0.8398 | 1.2657 |
| HGB +AGE_all | 0.4551 | 0.3429 | 0.5658 | 0.4130 | 0.8485 | 1.3036 |
| HGB v2_all+AGE | 0.4573 | 0.3451 | 0.5707 | 0.4156 | 0.8510 | 1.3252 |

(e10/e1 = top-10 % / top-1 % mean edge, ticks.) Gain at 100 ms: IC +0.0072 ridge, edge10 +0.010, edge1 +0.038. The other single features (expd_n, imb_after, frac_s, haz_d, l2, 500 ms) are +0.000-0.0003 at 100 ms. The signal is in young share, oldest age and the fraction difference, not in the headline expected cancel volume (a): (a) is nearly collinear with queue imbalance already in the ridge.

sample_v2 (same walk-forward): h100 ridge 0.3634 -> 0.3854 (+0.022), HGB 0.3834 -> 0.4208 (+0.037); h1000 ridge 0.4080 -> 0.4837 (+0.076), HGB 0.4198 -> 0.4667.

## Cross-fit (fit on all round1, test on sample_v2)

| set | h100 IC | e10 | e1 | h1000 IC | e10 | e1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline23 | 0.2829 | 0.0960 | 0.0185 | 0.2213 | 0.0935 | -0.0705 |
| +AGE_all | 0.3369 | 0.1016 | 0.1464 | 0.3533 | 0.3815 | -0.0656 |
| v2_all | 0.2732 | 0.0947 | 0.0318 | 0.2403 | 0.0989 | -0.0841 |
| v2_all + AGE | 0.3298 | 0.1165 | 0.1863 | 0.3601 | 0.3870 | -0.0238 |

The age features transfer better than the lot-denominated base features (they are fractions), so they are a robustness help if the scored tape resembles v2 more than round1; on round1 itself the gain is small.

## Porting (only if wanted despite the small gain)

Tracking: keep `ins_t[order_id]` from the first time an id appears in the snapshot (or the insert row; ids are 1000001 + insert index). Per event, the touch level of each side is a list of (id, size); the plain feature is an O(touch orders) loop (mean ~20-40 orders) with two table lookups per order (g200 on a 10 ms age bin, table of 800 floats from the first 1/8 of round1 lifetimes: S(b) = P(lifetime >= 10 ms * b)). An O(1)-per-event variant: keep per side 80 age buckets of 100 ms holding summed size; advance the ring on the clock, decrement on cancel/fill (needs id -> bucket), read E = sum bucket_size * g200(bucket centre). The cheaper winning pieces (young share < 150 ms, share > 650 ms, oldest age) need just three bucket sums. Expected benefit on round1: ~+0.007 IC at 100 ms, so porting is not recommended.
