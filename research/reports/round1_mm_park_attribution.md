# mm_park fill attribution (round1 and sample_v2): what the parked stubs do

Script: `research/fill_attribution.py`, new modes `park` (stub economics + tape events) and `compare` (two runs by fill category); existing modes untouched.
Logs: `logs/r1-park-inspect`, `logs/v2-park-inspect` (strategies/mm_park.py), `logs/r1-fullcap-inspect`, `logs/v2-fullcap-inspect` (strategies/exp_full_capacity.py; the v2 one was generated for this report).
Commands: `uv run python -X utf8 -m research.fill_attribution <log>/000_<set> data/features/<set>.npz park` and `... <log A> <npz> compare <log B>`.

Method
- Stub = our order with a `modify` event (the only use of modify_order in mm_park is parking, one modify per order). A fill is a **stub fill** when it happens at or after the order's modify time; fills of the same order before the park count as ordinary touch fills.
- Fill = rows grouped by (timestamp, taker order, our order), so row counts (round1 60,610 / 50,748) differ from the runner's fill counts (41,934 / 30,548); lots are comparable.
- Markout = sign*(tape mid(t+h) - fill price), lot-weighted, ticks; $ = lots*markout (proxy; sum of mo1s vs real PnL: park r1 452.1k vs 452.7k, fullcap 430.9k vs 425.7k).
- Front = no market lots of the same taker order were matched at our price before our row (nothing ahead of us at the fill).
- Tape events come from the market-only book in data/features: improvement = the side's best price moves to a better price, retreat = moves away, flicker = the previous best is the touch again within 500 ms.
- Caveat: "live bid/ask share" printed by the standard report is meaningless for mm_park (stubs never "end", a side always looks live: 1.000). Time at |I|=100 is valid.

## 1. Standard attribution, mm_park vs exp_full_capacity

| | round1 park | round1 fullcap | sample_v2 park | sample_v2 fullcap |
|---|---|---|---|---|
| lots filled | 2,335,875 | 2,647,809 | 479,314 | 523,666 |
| markout 100 ms / 1 s / 5 s / 30 s (ticks/lot) | 0.330 / 0.194 / 0.195 / 0.194 | 0.306 / 0.163 / 0.161 / 0.162 | 0.470 / 0.510 / 0.512 / 0.526 | 0.448 / 0.489 / 0.493 / 0.507 |
| mo1s $ | 452,080 | 430,932 | 244,559 | 256,286 |
| time at \|I\| = 100 | 0.261 | 0.504 | 0.217 | 0.533 |

round1 park by cut (mo1s per lot): spread 1 tick 0.18 / 2+ ticks 0.35; resting <50 ms 0.09 (30 s: -0.18), 50-500 ms 0.11, 0.5-5 s 0.28, >5 s 0.43 (3.4k lots); queue_ahead 0: 0.20, 1-50: 0.24, 51-200: 0.20, 201+: 0.16; inventory <50: 0.13, 50-89: 0.20, >=90: 0.28; reduces position 0.23, adds 0.10; imbalance against (<-0.5): +0.01 (fullcap -0.05; 766k lots vs 847k), book with (>0.5): 0.43.
Imbalance x queue_ahead, "book against" row (mo1s per lot): 1-50 ahead +0.08, 51-200 -0.01, 201+ -0.06 (fullcap +0.04 / -0.04 / -0.17). Parking turned the deep-queue against-the-book fills from -46.3k to -17.9k at 1 s and the whole against bucket from -39.7k to +6.6k.
sample_v2 park: spread 2+ ticks is 49 % of lots (0.64 per lot) vs 1 tick 0.39; resting 0.5-5 s 0.50; imbalance-against still +0.33. No losing bucket anywhere on v2 (all cuts 0.3-0.7).

## 2. Stub economics

| | round1 | sample_v2 |
|---|---|---|
| stubs created (modify events) | 19,922 | 2,779 |
| tape improvements (bid + ask) | 15,724 | 4,681 |
| filled in full | 7,936 (39.8 %) | 1,174 (42.2 %) |
| part-filled, then cancelled | 174 | 16 |
| never filled (cancelled; 3 left at end in r1) | 11,812 (59.3 %) | 1,589 (57.2 %) |
| stub lots filled (stub-fill rows) | 160,533 (9,339) | 23,637 (1,431) |
| lots filled per stub created (of 20) | 8.1 | 8.5 |
| lifetime to full fill, mean / median / p90 | 1.37 s / 1.13 s / 2.76 s | 3.81 s / 3.02 s / 7.80 s |
| lifetime to cancel (unfilled), mean / median / p90 | 0.95 s / 0.65 s / 2.33 s | 2.20 s / 1.61 s / 5.31 s |
| share of stub lots filled at the FRONT (0 market lots ahead) | 59.6 % | 27.7 % |
| stub markout/lot 100 ms / 1 s / 5 s / 30 s | 0.492 / 0.393 / 0.419 / 0.422 | 0.748 / 0.780 / 0.794 / 0.817 |
| touch (non-stub) markout/lot 1 s | 0.179 | 0.496 |
| stub $ at 1 s / 30 s | 63.2k / 67.8k | 18.4k / 19.3k |
| stub edge at fill (ticks/lot, vs touch) | 0.93 (0.51) | 0.87 (0.54) |
| stub share of lots / of mo1s $ | 6.9 % / 14.0 % | 4.9 % / 7.5 % |

Round1 stub fills at the front: 95.6k lots, mo1s 0.414, 30 s 0.525 (+39.6k at 1 s); behind market lots: 64.9k lots, 0.363, 30 s 0.270 (+23.6k). By time since park: <50 ms 3.4k lots (0.60), 50-500 ms 27.7k (0.46), 0.5-5 s 127.5k (0.37), >5 s 2.0k. 80 % of stub lots add to inventory (0.35 per lot); the reducing ones earn 0.56. On v2 stub fills behind market lots (17.1k lots) are the better ones, 0.83 (front 0.65).
Reading: a stub sits at the old, worse price (edge at fill 0.9 tick vs 0.5 for touch fills) and the market moves back through it, so the fill comes with the mid already beyond it. No adverse selection in either dataset: stub markout is 2.2x the touch markout on round1 and 1.6x on v2.

## 3. Where the +26.9k (+6.3 %) on round1 comes from (mo1s proxy: +21.1k; actual +26,943)

| | fullcap | park | change |
|---|---|---|---|
| touch (non-stub) lots | 2,647,809 | 2,175,342 | -472,467 (-17.8 %) |
| touch fills (rows) | 50,748 | 51,271 | +523 (+1 %) |
| touch markout/lot 1 s (30 s) | 0.163 (0.162) | 0.179 (0.177) | +0.016 |
| touch $ at 1 s | 430,932 | 388,924 | -42,008 |
| stub lots / $ at 1 s | 0 | 160,533 / 63,156 | +63,156 |
| total mo1s $ | 430,932 | 452,080 | +21,148 |
| time at \|I\|=100 | 50.4 % | 26.1 % | halved |

Touch change: volume -472k lots x 0.163 = -77.0k; quality +0.016 x 2.175M lots = +33.9k (interaction ~+1k). Stubs +63.2k. Net +21.1k; the remaining 5.8k to the true +26.9k is proxy error (mark vs tape mid without zombies; the fullcap proxy is itself 5.2k above its true PnL).
Read: touch fill count is unchanged (+1 %) but the fills are 18 % smaller in lots, because capacity is shared with the stubs and the bulk is re-sent smaller after a park. The remaining touch lots are better (+0.016: far less time stuck at the cap, fewer against-the-book deep-queue fills). Stubs pay because each stub lot is worth 0.39 against 0.18 for the touch lot it displaces (about +0.21 per lot moved, 160k lots, +34k net of displacement) plus the quality gain. Buy/sell touch are symmetric (1 s: buy 0.183, sell 0.174; the 30 s drift stays worse on sells, 0.098 vs 0.257).

## 4. Regime contrast: sample_v2 (mm_park -4.7 %, proxy -11.7k, actual -12,382)

| | fullcap | park | change |
|---|---|---|---|
| touch lots | 523,666 | 455,677 | -67,989 (-13.0 %) |
| touch markout/lot 1 s | 0.489 | 0.496 | +0.007 |
| touch $ at 1 s | 256,286 | 226,122 | -30,164 |
| stub lots / markout / $ at 1 s | 0 | 23,637 / 0.780 / 18,436 | +18,436 |
| total mo1s $ | 256,286 | 244,559 | -11,727 |
| time at \|I\|=100 | 53.3 % | 21.7 % | |

Stub fills on v2 are not adverse (0.78 vs 0.50 for touch fills). The cost is the reduced bulk: volume -68.0k lots x 0.489 = -33.2k, quality +0.007 x 456k = +3.2k, stubs +18.4k. Stubs fill only 23.6k lots (4.9 % of the total) while the touch loses 68k lots; each lot is worth about 0.5 on v2 (wide spreads, no losing bucket), so the lots displaced are expensive, while on round1 they were worth 0.16. (Inference: three 20-lot stubs per side hold up to 60 of the 100 free lots, and 57 % of v2 stubs never fill, living 2-4 s.)

## 5. Events (tape-level, market-only book)

| | round1 bid / ask | sample_v2 bid / ask |
|---|---|---|
| improvements of the touch (park trigger) | 7,878 / 7,846 | 2,368 / 2,313 |
| per second | 0.66 / 0.65 | 0.41 / 0.40 |
| retreats (touch moves away) | 7,544 / 7,557 | 2,339 / 2,300 |
| improvement undone within 500 ms (old touch is the touch again) | 2,955 (37.5 %) / 2,882 (36.7 %) | 855 (36.1 %) / 839 (36.3 %) |

Our actions: round1 19,922 modifies (11,797 distinct timestamp-side groups, 1.7 stubs per park event), 20,583 cancels, 186 cancel_failed, 26 modify_failed; v2 2,779 modifies (1,716 groups), 4,554 cancels, 51 cancel_failed. About 37 % of improvements are flicker in both datasets.
