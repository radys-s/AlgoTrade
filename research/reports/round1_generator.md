# Round 1 tape: recording or synthetic model? (generator fingerprint)

Script: `research/generator_fingerprint.py` (parts 1-7; `uv run python -X utf8 research/generator_fingerprint.py round1 [part...]`).
Raw outputs in `logs/generator/` (p1..p7 = round1, sv2.txt = sample_v2 parts 1,3,5). Pre-state = book after the previous event; rates are events per second of time spent in that state.

## Verdict

**Synthetic.** A stochastic limit-order-book model with Poisson arrivals, queue-reactive price placement, i.i.d. order lifetimes (cancel scheduled at insertion) and a lognormal taker size. Not a recording: sample_v2 comes from the same generator with different parameters (limit size uniform 20-40 instead of heavy tailed, IOC lognormal mu 3.71 instead of 4.11, 348 vs 368 events/s) and shows the same curves.

Signatures:

| signature | measured | reading |
|---|---|---|
| G inserts per side and IOCs: counts per 1 s / 100 ms | Fano 0.99-1.00 (G, per side, IOC) | exactly Poisson, constant rate, no Hawkes |
| total G insert rate per side | 89.9/s regardless of queue size, spread, opposite queue (88.6 at q<74, 88.5 at q>3100; 89.9 at spread 2) | arrival TIMES are state-free; only the PLACEMENT is state-dependent |
| IOC rate | 8.0/s (4.0 per side), P(next type) independent of previous type, side coin-flip (P same side 0.500) | zero-intelligence takers |
| order ids | insert id == 1000001 + insert index (100%), cancel ids unique | counter-based generator |
| timestamps | us resolution, last digit uniform (0.100 each), us-in-ms chi2 p 0.54, no periodic grid | continuous clock rounded to us |
| cancels | Fano 1.35 (not Poisson), corr(per-second G inserts, cancels) = 0.51; 66 of 2,157,657 G never cancelled | cancels are a delayed copy of inserts plus state coupling |
| filled orders still get a "cancel" | 3.4 % of cancels hit an order already gone from the book (price better than best at cancel time) | generator does not track fills for scheduling, cancel is just a no-op in our engine |
| seed | row 1 = bid 9999, two-sided from row 8, stationary rate from t=0 (367 ev/s in every 1000 s bucket) | book empty at t=0, fills in ~3 s |

## 1. Insert intensity vs state (G inserts per second, per side)

Bid side by own touch quantity (ask is the mirror image to within noise):

| own touch qty | at touch /s | 1 behind /s | 2 behind /s | share at touch |
|---|---:|---:|---:|---:|
| 1-74 | 21.6 | 56.6 | 10.4 | 0.24 |
| 202-356 | 37.3 | 45.1 | 6.6 | 0.41 |
| 522-705 | 49.3 | 35.1 | 4.5 | 0.55 |
| 904-1128 | 56.9 | 28.4 | 3.3 | 0.63 |
| 1728-2216 | 63.3 | 22.3 | 2.8 | 0.71 |
| >3132 | 64.5 | 21.5 | 2.7 | 0.72 |

Clean, monotone, saturating curve (queue-reactive signature); sum of columns is constant. Same with the number of orders at the touch (bid at-touch rate 16.5/s at <=3 orders, 75/s at >35). Dependence on the OPPOSITE queue: bid at-touch rate 70.3/s when ask touch qty <130, 63/s (130-352), 51 (611-902), 41.7 (1250-1726), 36.9/s when >2588. By spread: spread 1 -> at touch 49.2, 1 behind 35.1, 2 behind 4.9 (/s/side); spread 2 -> inside 8.9, touch 68.2, 1 behind 11.5; spread 3 -> inside 42.9, touch 37.1. Placement shares (all): inside 0.7 %, touch 56.0 %, 1 behind 37.4 %, 2 behind 5.2 %, 3 behind 0.6 %.

Surprise: the BIGGER the own queue, the MORE new orders join the touch (not fewer). Zero-intelligence would be flat; it is not. sample_v2 has the same shape, stronger (share at touch 0.09 -> 0.87).

## 2. Cancel hazard

* Lifetime: mean 467 ms, median 337, std 443 (CV 0.95), q1 % 20 ms, q99 % 2.13 s, max 7.20 s. KS vs exponential 0.056 (rejected, no memorylessness). Best fits: shifted lognormal KS 0.020 (sigma 0.85, shift -30 ms, scale 353), gamma KS 0.022 (shape 1.30), Weibull 0.026.
* Hazard by age (per second): 0.60 (0-25 ms), 1.45 (25-50), 1.81 (50-100), 2.03 (100-150), 2.13 (150-200), 2.11 (200-300), 2.17 (300-500), 2.04 (500-650), 1.83 (800-1000), 1.60 (1000-1300), 1.26 (1700-2200), ~0.2 (>3 s). Rises to a plateau at ~150-650 ms then decays.
* Lifetime does not depend on size (mean 466-468 ms for all size bands), on offset from the touch at insertion (462-469), on touch qty at insertion (466-468), on time of day. So the cancel clock is set at insertion, i.i.d.
* Mild state coupling of the cancel clock: among orders alive at 300 ms, P(cancel in next 100 ms) = 0.230 at the touch, 0.195 one behind, 0.190 two behind; at the touch it is 0.270 if touch qty <752 vs 0.208 if >2225; independent of the opposite qty (0.225-0.232). (Orders already consumed show 0.46: pure clock.)

## 3. Sizes

| | n | mean | median | log mean / sd | log skew / kurt | fit |
|---|---:|---:|---:|---:|---:|---|
| limit round1 | 2,157,657 | 71.9 | 14 | 2.71 / 1.70 | 0.28 / -0.36 | smooth heavy tail (Hill alpha 1.5), pmf(1)=0.106, pmf(2)=0.061, pmf(14)=0.016, pmf(45)=0.004; lognormal rejected (KS 0.023); NOT uniform 1-40 |
| IOC round1 | 96,088 | 160.7 | 61 | 4.106 / 1.395 | -0.002 / -0.02 | lognormal, KS 0.0020 (p 0.85) |
| limit sample_v2 | 1,033,029 | 30.0 | 30 | - | - | uniform integers 20..40 (pmf 0.0475 each) |
| IOC sample_v2 | 34,503 | 108.7 | 41 | 3.712 / 1.397 | 0.006 / -0.06 | lognormal, KS 0.004 |

No state dependence: IOC size vs hit-queue Spearman 0.009 (median 60-64 for every queue band); limit size vs own touch qty -0.0003; limit size by offset identical (median 14); lag-1 size autocorrelation -0.001. Takers do NOT size to the queue: P(IOC size >= hit-side touch qty) = 0.68 (qty<94), 0.20 (94-349), 0.053 (349-898), 0.017 (898-1711), 0.005 (1711-2857), 0.001 (>2857), overall 11.7 %. IOC size >= queue empties the level in 97-99 % of cases; size < queue never does (0.000): fully mechanical.

## 4. Refill of widened spread

* Spread-1 -> 2 episodes: opened by IOC 72 % / cancel 28 %, bid-side and ask-side 50/50; duration median 22 ms (p10 2.9 ms, p25 8.2, p75 58, p90 144, p99 467); ended by a G inside insert 98.7 % (IOC 1.2 %), about 20 events in between.
* WHO refills is imbalance driven: inside-insert rate per side vs imbalance quintile (spread 2): bid 5.9, 7.1, 7.9, 9.1, 14.8 /s as bid-ask imbalance rises; ask mirror 14.5 -> 6.6 /s. By opposite qty: bid inside rate 14.8/s when the ask touch qty is in its lowest quintile, 6.2/s in its highest. The inside insert goes onto the side with the bigger queue, i.e. AWAY from the side that retreated: the mid continues 70 % (9,922 of 14,106 episodes: refill moves the mid 0.5 tick the same way as the opening jump, 30 % back). This is the source of the 0.46 reversal probability.
* After a bid-touch retreat the next bid G insert comes after median 7.8 ms (p10 1.2 ms, p90 25.9 ms), 62 % at the new bid, 23 % one tick inside, size median 14.

## 5. Mid dynamics

2.56 mid changes/s. Cause shares: new inside G insert 51.1 %, IOC depletion 36.0 %, cancel depletion 12.9 %. Every IOC mid move is toward the IOC side (100 %); inside inserts always move the mid toward their own side. Depletion by IOC happens only when size >= queue (see 3). Depletion by cancel: removed level had median 31 lots and 1 order (IOC: median 65 lots, 3 orders). Cancel depletions up/down 50.4/49.6 %.

## 6. Predictive tables (touch-level depletion)

P(bid level gone with a lower bid, within W):

| bid touch qty | 50 ms | 200 ms |
|---|---:|---:|
| 1-35 | 0.182 | 0.395 |
| 35-96 | 0.108 | 0.301 |
| 96-262 | 0.055 | 0.190 |
| 262-667 | 0.019 | 0.089 |
| 667-1175 | 0.007 | 0.041 |
| 1175-1991 | 0.003 | 0.020 |
| >1991 | 0.002 | 0.014 |

By orders at the touch: 1 order 0.218 (50 ms) / 0.455 (200 ms); 2: 0.106 / 0.337; 3: 0.058 / 0.236; 4: 0.032 / 0.138; 5: 0.017 / 0.082; >=6: 0.004 / 0.021. By imbalance (bid-ask)/(bid+ask) in 200 ms: imb<-0.88 -> bid falls 0.350, ask rises 0.013; imb>0.89 -> bid falls 0.013, ask rises 0.343 (same shape at 50 ms: 0.142 vs 0.0025).

## 7. Three most exploitable regularities for a passive maker

1. **Thin-queue depletion is a computable hazard, and the arrival process does not refill it in time.** A touch with <96 lots or <=2 orders loses its level within 200 ms with 30-45 % (vs 1.4-2 % when >2000 lots / >=6 orders); with the opposite queue thick (imb beyond +-0.89) the thin side moves 34 % vs 1.3 % against it. Quote only the thick side; pull the thin side (a bid sitting last in a <100 lot queue is hit by P(IOC>=position): size lognormal(4.106, 1.395), 4 IOCs/s per side, i.e. P(fill) per IOC for queue-ahead q is 1-Phi((ln q - 4.106)/1.395)).
2. **Takers never look at the book, so fills are arithmetic.** IOC size ~ lognormal(4.106,1.395) independent of the queue, 4.0/s per side Poisson, no autocorrelation (lag 1 +0.015). Expected time to be hit for position q: lambda_sweep(q) = 4.0/s * (1-Phi((ln q-4.106)/1.395)); level removal when IOC size >= whole queue is 97-99 % mechanical. Combined with the cancel clock (i.i.d. lifetime, median 337 ms, hazard 2.1/s, max 7.2 s; ~27 % of orders at a small touch cancel per 100 ms vs 21 % at a big one), the time at which queue-ahead drains is predictable; orders ahead older than 2.2 s have hazard <=1.0/s.
3. **Spread widening resolves in ~22 ms and continues the move 70 % of the time.** After a 1->2 spread the refill is an inside insert (98.7 %), median 22 ms, placed on the side with the larger queue (inside rate 14.8/s vs 6.2/s). So after the touch of one side vanishes expect the mid to follow through (0.5 tick, P=0.70) rather than revert: do not leave a quote on the side that just retreated's opposite (the thick side side is the one that inserts inside, stepping ahead of a resting quote there); by contrast the retreated side's queue is rebuilt at the touch within median 7.8 ms, p90 26 ms, 62 % exactly at the new best.

Caveats: tests were on the whole tape, no walk-forward split because they describe generator parameters (the same curves appear in sample_v2 and in every 1000 s bucket). Queue-position-level hazards (front vs back) were not measured (no per-order queue reconstruction). The Fano 1.35 of cancels and 0.51 insert/cancel per-second correlation are not fully explained by the independent-lifetime model and hint at a state-dependent cancel component.
