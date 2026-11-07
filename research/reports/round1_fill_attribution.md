# Round 1 fill attribution (market-making candidate and champion)

Script: `research/fill_attribution.py` (`uv run python -X utf8 -m research.fill_attribution logs/<tag>/000_round1 data/features/round1.npz <label>`).
Logs: `logs/r1-fullcap-inspect` (strategies/exp_full_capacity.py, pnl 425,708) and `logs/r1-champ-inspect` (submission/best_strategy.py, pnl 321,024).

Method notes
- CORRECTION (3rd round): inventory is now rebuilt from ALL our trades, including the 131 rows where we are the taker (earlier versions missed them and drifted, which had hidden the cap-time).
- Fill = rows of trades.csv with maker_owner=user, grouped by (timestamp, taker order, our order): 50,748 for the candidate (the runner's 30,548 counts differently; lots match, 2.648M).
- Mid path = data/features/round1.npz (market-only tape book, forward-filled when a side is empty). Markout = sign*(mid(t+h) - fill price), lot-weighted, ticks ($1 per lot per tick). Spread and imbalance are read from the feature state just before the fill row.
- Fresh/back proxies: (i) resting time = fill time minus the applied "add" time of our order; (ii) queue_ahead = market quantity resting at our price when the order was applied (0 = improver / new level). No ">5 s" bucket exists: the strategy never leaves an order resting that long.
- Imbalance is signed so + means the L1 book leans the way we trade (bid-heavy when we buy).
- edge = sign*(mid before - price). mo1s / mo30s $ = sum of qty*markout. Sum of mo1s (431k) is close to the real pnl (425.7k), so the 1 s markout is a good PnL proxy; edge minus markout (about 0.34/lot) is mid drift against us after fills.
- Caveat: the feature book is the tape without our orders (no zombies), so mid(t+h) differs slightly from the simulated book.

## Summary (candidate unless stated)

Lose money (per lot, markout at 1 s / 30 s):
1. Fills when the L1 book leans against the fill side (imbalance below -0.5, e.g. we buy into an ask-heavy book): 847k lots (32%), -39.7k at 1 s, -68.4k at 30 s (-0.05 / -0.08 per lot). Champion: -49.9k / -40.0k. Best filter candidate.
2. Fills that add to existing inventory: 517k lots, -6.7k at 1 s, +9.9k at 30 s (about 0 per lot) vs reducing fills +437k / +420k (+0.21 per lot). Almost all PnL comes from fills that cut inventory.
3. Orders that rest under 50 ms before filling (just joined / improved, 153k lots): +8.3k at 1 s but -30.2k at 30 s (-0.20 per lot); champion -18.0k at 30 s. Flat-inventory fills (8k lots) also -7.4k at 30 s. Sell fills drift worse than buys at 30 s (+92k vs +338k).

Earn money:
- Book leaning with the fill (imbalance above 0.5): 651k lots, +272.9k at 1 s (+0.42 per lot), 42% of the 1 s total on 25% of lots; 0.1-0.5 adds +98.8k (+0.22).
- Orders resting 0.5-5 s: 1.23M lots, +305k at 1 s (+0.25 per lot), 71% of 1 s PnL; 50-500 ms ones only +0.09 per lot.
- Small-to-mid takers (under 100 lots): 1.15M lots, +279.6k (+0.24 per lot) vs sweeps (taker 100+ lots) 1.50M lots, +151.3k (+0.10 per lot, still positive). Our fills of 11-99 lots earn 0.20-0.24 per lot, 100+ lot fills 0.10.
- Spread 2+ ticks is better per lot (0.27 vs 0.16) but only 6% of lots (+40k). Inventory is not a loser: |I| >= 90 earns the most per lot (0.24, 1.13M lots, +265k at 1 s), |I| < 50 the least (0.09). Time at |I| = 100: 50.4% (champion 28.3%); quotes live ~74% of time per side (champion 85%).

## Candidate: exp_full_capacity

fills 50748, lots 2647809, tape span 12000s. $ columns: edge = qty*sign*(mid_before-price); mo1s/mo30s = qty*markout vs mid 1s/30s after the fill (1 tick = $1 per lot). Sums are proxies, not the exact score.

### (a) by spread at fill

| spread_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| 1 tick | 47381 | 2499910 | 1237654.0 | 390762.0 | 382345.0 | 0.16 | 0.15 |
| 2+ ticks | 3367 | 147899 | 98614.0 | 40170.5 | 47497.5 | 0.27 | 0.32 |

### (b) by resting time of the order

| rest_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| <50ms | 2113 | 152913 | 76743.0 | 8290.5 | -30194.5 | 0.05 | -0.2 |
| 50-500ms | 20976 | 1264485 | 631566.5 | 117163.0 | 111551.5 | 0.09 | 0.09 |
| 0.5-5s | 27659 | 1230411 | 627958.5 | 305479.0 | 348485.5 | 0.25 | 0.28 |

### (b2) fresh vs back proxy: market qty ahead of us when the order was applied

| qa_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| 0 (front/improver) | 333 | 11494 | 4692.0 | 1387.5 | 2551.5 | 0.12 | 0.22 |
| 1-50 | 14332 | 917424 | 461552.5 | 189783.0 | 133983.5 | 0.21 | 0.15 |
| 51-200 | 9466 | 574219 | 290429.0 | 99284.5 | 76751.0 | 0.17 | 0.13 |
| 201+ | 26616 | 1144580 | 579548.5 | 140339.5 | 216510.5 | 0.12 | 0.19 |
| far/off-book | 1 | 92 | 46.0 | 138.0 | 46.0 | 1.5 | 0.5 |

### (c) by inventory before fill

| inv_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| |I|<50 | 18580 | 844620 | 408268.0 | 72556.0 | 63231.0 | 0.09 | 0.07 |
| 50-89 | 15909 | 677794 | 334678.0 | 93150.0 | 141921.5 | 0.14 | 0.21 |
| >=90 | 16259 | 1125395 | 593322.0 | 265226.5 | 224690.0 | 0.24 | 0.2 |

by side

| side | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| buy | 25434 | 1323970 | 667347.5 | 221461.0 | 337588.0 | 0.17 | 0.25 |
| sell | 25314 | 1323839 | 668920.5 | 209471.5 | 92254.5 | 0.16 | 0.07 |

by direction vs inventory

| inv_dir | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| adds to position | 18400 | 517305 | 235669.0 | -6742.0 | 9862.5 | -0.01 | 0.02 |
| flat | 158 | 6963 | 3213.0 | 353.0 | 312.5 | 0.05 | 0.04 |
| reduces position | 32190 | 2123541 | 1097386.0 | 437321.5 | 419667.5 | 0.21 | 0.2 |

### (d) by our fill size, and by total taker size

| fill_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| 1-10 | 10181 | 55882 | 25360.0 | 13665.0 | 12451.0 | 0.24 | 0.22 |
| 11-50 | 22484 | 603283 | 295218.5 | 142127.0 | 160138.5 | 0.24 | 0.27 |
| 51-99 | 9572 | 681396 | 341421.0 | 138846.0 | 147372.5 | 0.2 | 0.22 |
| 100+ | 8511 | 1307248 | 674268.5 | 136294.5 | 109880.5 | 0.1 | 0.08 |

| size_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| taker <100 lots | 37288 | 1145349 | 560683.0 | 279599.0 | 268335.0 | 0.24 | 0.23 |
| taker >=100 lots (sweep) | 13460 | 1502460 | 775585.0 | 151333.5 | 161507.5 | 0.1 | 0.11 |

### L1 imbalance before fill (signed: + = book leans the way we trade, i.e. we are on the heavy side)

| imb_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| <-.5 (book against) | 15909 | 847192 | 435325.5 | -39735.0 | -68422.0 | -0.05 | -0.08 |
| -.5..-.1 | 8960 | 479027 | 246392.0 | 62596.5 | 60924.5 | 0.13 | 0.13 |
| flat | 4380 | 226138 | 117304.0 | 36357.5 | 34129.0 | 0.16 | 0.15 |
| .1..5 | 8519 | 444621 | 228636.0 | 98829.0 | 147737.5 | 0.22 | 0.33 |
| >.5 (book with) | 12980 | 650831 | 308610.5 | 272884.5 | 255473.5 | 0.42 | 0.39 |

### (e) lot-weighted mean markout (ticks)

| h | 100ms | 1s | 5s | 30s | edge at fill |
|---|---|---|---|---|---|
| markout | 0.306 | 0.163 | 0.161 | 0.162 | 0.505 |

### (f) time
- fraction of tape time with a live bid: 0.739; live ask: 0.743
- fraction of time at |I| = 100: 0.504


## Champion: quote_size 100

fills 59484, lots 2199374, tape span 12000s. $ columns: edge = qty*sign*(mid_before-price); mo1s/mo30s = qty*markout vs mid 1s/30s after the fill (1 tick = $1 per lot). Sums are proxies, not the exact score.

### (a) by spread at fill

| spread_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| 1 tick | 55347 | 2068488 | 1019769.0 | 301078.0 | 286915.5 | 0.15 | 0.14 |
| 2+ ticks | 4137 | 130886 | 75662.5 | 22697.5 | 32166.0 | 0.17 | 0.25 |

### (b) by resting time of the order

| rest_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| <50ms | 2777 | 138517 | 64328.5 | 3398.5 | -17959.0 | 0.02 | -0.13 |
| 50-500ms | 26168 | 1089542 | 532445.5 | 73550.5 | 80646.5 | 0.07 | 0.07 |
| 0.5-5s | 30539 | 971315 | 498657.5 | 246826.5 | 256394.0 | 0.25 | 0.26 |

### (b2) fresh vs back proxy: market qty ahead of us when the order was applied

| qa_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| 0 (front/improver) | 1577 | 51758 | 13696.5 | 1725.0 | 1179.5 | 0.03 | 0.02 |
| 1-50 | 13983 | 639790 | 324594.0 | 156337.5 | 137180.0 | 0.24 | 0.21 |
| 51-200 | 9353 | 389624 | 195632.5 | 68710.0 | 39577.0 | 0.18 | 0.1 |
| 201+ | 34568 | 1118121 | 561468.0 | 96868.0 | 141131.5 | 0.09 | 0.13 |
| far/off-book | 3 | 81 | 40.5 | 135.0 | 13.5 | 1.67 | 0.17 |

### (c) by inventory before fill

| inv_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| |I|<50 | 31994 | 1118356 | 540751.0 | 46011.0 | 10192.5 | 0.04 | 0.01 |
| 50-89 | 15532 | 501466 | 245956.0 | 87195.0 | 132413.0 | 0.17 | 0.26 |
| >=90 | 11958 | 579552 | 308724.5 | 190569.5 | 176476.0 | 0.33 | 0.3 |

by side

| side | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| buy | 29703 | 1099777 | 547161.0 | 168584.5 | 231347.0 | 0.15 | 0.21 |
| sell | 29781 | 1099597 | 548270.5 | 155191.0 | 87734.5 | 0.14 | 0.08 |

by direction vs inventory

| inv_dir | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| adds to position | 20854 | 552369 | 246396.0 | -57683.5 | -22240.0 | -0.1 | -0.04 |
| flat | 9294 | 407419 | 197922.0 | 1023.5 | -52585.0 | 0.0 | -0.13 |
| reduces position | 29336 | 1239586 | 651113.5 | 380435.5 | 393906.5 | 0.31 | 0.32 |

### (d) by our fill size, and by total taker size

| fill_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| 1-10 | 13878 | 75276 | 34413.5 | 14607.0 | 14266.0 | 0.19 | 0.19 |
| 11-50 | 28667 | 764421 | 372696.5 | 140894.0 | 179370.0 | 0.18 | 0.23 |
| 51-99 | 11475 | 813277 | 407621.5 | 119874.5 | 86445.5 | 0.15 | 0.11 |
| 100+ | 5464 | 546400 | 280700.0 | 48400.0 | 39000.0 | 0.09 | 0.07 |

| size_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| taker <100 lots | 46215 | 1347474 | 661381.5 | 284225.5 | 270181.5 | 0.21 | 0.2 |
| taker >=100 lots (sweep) | 13269 | 851900 | 434050.0 | 39550.0 | 48900.0 | 0.05 | 0.06 |

### L1 imbalance before fill (signed: + = book leans the way we trade, i.e. we are on the heavy side)

| imb_b | fills | lots | edge_usd | mo1s_usd | mo30s_usd | mo1s_per_lot | mo30s_per_lot |
|---|---|---|---|---|---|---|---|
| <-.5 (book against) | 19751 | 754185 | 385916.5 | -49926.5 | -39972.5 | -0.07 | -0.05 |
| -.5..-.1 | 10700 | 396735 | 201054.5 | 53515.5 | 44105.5 | 0.13 | 0.11 |
| flat | 5026 | 183011 | 93378.0 | 30979.5 | 27518.5 | 0.17 | 0.15 |
| .1..5 | 9781 | 359821 | 181704.0 | 79786.0 | 91788.5 | 0.22 | 0.26 |
| >.5 (book with) | 14226 | 505622 | 233378.5 | 209421.0 | 195641.5 | 0.41 | 0.39 |

### (e) lot-weighted mean markout (ticks)

| h | 100ms | 1s | 5s | 30s | edge at fill |
|---|---|---|---|---|---|
| markout | 0.285 | 0.147 | 0.148 | 0.145 | 0.498 |

### (f) time
- fraction of tape time with a live bid: 0.850; live ask: 0.854
- fraction of time at |I| = 100: 0.283


## Follow-up (candidate)

### 1. Imbalance at fill x queue_ahead (market qty at our price when the order was applied, 5 ms after send)

Cells: lots / markout per lot at 1 s / total $ at 1 s. Imbalance signed (+ = book leans the way we trade).

| imbalance x queue_ahead | 0 | 1-50 | 51-200 | 201+ | all |
|---|---|---|---|---|---|
| <-0.5 | 3,978 / -0.08 / -322 | 373,674 / 0.04 / 14,732 | 193,160 / -0.04 / -7,802 | 276,380 / -0.17 / -46,342 | 847,192 / -0.05 / -39,735 |
| -0.5..0.1 | 3,343 / 0.17 / 581 | 225,697 / 0.20 / 45,262 | 157,643 / 0.14 / 22,801 | 318,482 / 0.10 / 30,310 | 705,165 / 0.14 / 98,954 |
| 0.1..0.5 | 1,101 / -0.03 / -34 | 139,289 / 0.28 / 39,436 | 91,876 / 0.31 / 28,650 | 212,355 / 0.14 / 30,778 | 444,621 / 0.22 / 98,829 |
| >0.5 | 3,072 / 0.38 / 1,163 | 178,764 / 0.51 / 90,354 | 131,540 / 0.42 / 55,636 | 337,455 / 0.37 / 125,732 | 650,831 / 0.42 / 272,884 |

### 2. Why a side has no live quote (share of tape time, 12,000 s)

| side | live | (a) at cap (|I| = 100 on that side) | (b) not capped, send in flight (<5 ms after a send on that side) | (c) other | gaps (no live, not capped) | of which >50 ms | mean gap ms | share of time in >50 ms gaps |
|---|---|---|---|---|---|---|---|---|
| bid | 0.739 | 0.254 | 0.006 | 0.002 | 13663 | 0 | 6.5 | 0.000 |
| ask | 0.743 | 0.250 | 0.006 | 0.002 | 13586 | 0 | 6.5 | 0.000 |

(b) counts only time while a send is in flight; (c) is the remainder. Gaps merge consecutive not-capped no-quote stretches.


Findings (corrected inventory): the no-quote time is the position cap. Per side 25% of the time the side is blocked at the cap (free capacity 0), 0.6% a send is in flight, 0.2% other; no not-capped stretch is longer than 50 ms. An earlier version of this section (c = 10-14%) was a bug: inventory ignored our own taker trades and drifted by about 32 lots. Imbalance x queue_ahead table is unaffected by the correction.

- Imbalance-against fills (<-0.5) that lose are the deep-queue ones: queue_ahead 201+ = 276k lots at -0.17 per lot (-46.3k at 1 s); 51-200 ahead -0.04 (-7.8k); 1-50 ahead +0.04 (+14.7k); fresh-level (0 ahead) only 4k lots. The deep 201+ queue also holds the biggest winners when the book leans with us (+125.7k), so an imbalance pull cuts valuable fills too. Narrower candidates: imbalance < -0.5 and queue_ahead > 200 (-46k on 276k lots), or > 50 (-54k on 470k lots).

### 3. Raw traces of two gaps (inventory now includes our own taker trades)

Bid gap:
Trace 3670.084-3670.679 s (gap 3670.384-3670.629 s). market bid/ask = tape book after the last tape row at or before the time.
```
 3670.097536 bid=10057 ask=10058  add_sent      u000017280 A px=10058.0 qty=78.0
 3670.097536 bid=10057 ask=10058  FILL         u000017277 B px=10057 qty=78 taker=1689611  inventory_after=-22
 3670.102536 bid=10057 ask=10058  add           u000017280 A px=10058.0 qty=78.0
 3670.383635 bid=10056 ask=10058  add_sent      u000017281 A px=10058.0 qty=122.0
 3670.383635 bid=10056 ask=10058  FILL         u000017277 B px=10057 qty=35 taker=1689663  inventory_after=13
 3670.383635 bid=10056 ask=10058  FILL         u000017278 B px=10057 qty=12 taker=1689663  inventory_after=25
 3670.383635 bid=10056 ask=10058  FILL         u000017279 B px=10057 qty=75 taker=1689663  inventory_after=100
 3670.383856 bid=10056 ask=10057  cancel_sent   u000017280 - px=- qty=-
 3670.383856 bid=10056 ask=10057  cancel_sent   u000017281 - px=- qty=-
 3670.388635 bid=10056 ask=10057  add           u000017281 A px=10058.0 qty=122.0
 3670.388856 bid=10056 ask=10057  cancel        u000017280 A px=10058.0 qty=78.0
 3670.388856 bid=10056 ask=10057  cancel        u000017281 A px=10058.0 qty=122.0
 3670.392361 bid=10056 ask=10057  add_sent      u000017282 A px=10057.0 qty=200.0
 3670.397361 bid=10056 ask=10057  add           u000017282 A px=10057.0 qty=200.0
 3670.624288 bid=10056 ask=10057  add_sent      u000017283 B px=10056.0 qty=9.0
 3670.624288 bid=10056 ask=10057  FILL         u000017282 A px=10057 qty=9 taker=1689712  inventory_after=91
 3670.629288 bid=10056 ask=10057  add           u000017283 B px=10056.0 qty=9.0
tape rows inside the gap: 97
```

Ask gap:
Trace 11090.209-11091.147 s (gap 11090.509-11091.097 s). market bid/ask = tape book after the last tape row at or before the time.
```
11090.418845 bid=10014 ask=10015  add_sent      u000052393 B px=10014.0 qty=58.0
11090.418845 bid=10014 ask=10015  FILL         u000052392 A px=10015 qty=58 taker=3082825  inventory_after=42
11090.423845 bid=10014 ask=10015  add           u000052393 B px=10014.0 qty=58.0
11090.447285 bid=10014 ask=10016  add_sent      u000052394 B px=10014.0 qty=18.0
11090.447285 bid=10014 ask=10016  FILL         u000052392 A px=10015 qty=18 taker=3082833  inventory_after=24
11090.452285 bid=10014 ask=10016  add           u000052394 B px=10014.0 qty=18.0
11090.460411 bid=10015 ask=10016  add_sent      u000052395 B px=10014.0 qty=1.0
11090.460411 bid=10015 ask=10016  FILL         u000052392 A px=10015 qty=1 taker=3082837  inventory_after=23
11090.465411 bid=10015 ask=10016  add           u000052395 B px=10014.0 qty=1.0
11090.469746 bid=10015 ask=10016  add_sent      u000052396 B px=10014.0 qty=21.0
11090.469746 bid=10015 ask=10016  FILL         u000052392 A px=10015 qty=21 taker=3082841  inventory_after=2
11090.474746 bid=10015 ask=10016  add           u000052396 B px=10014.0 qty=21.0
11090.508727 bid=10015 ask=10016  cancel_sent   u000052393 - px=- qty=-
11090.508727 bid=10015 ask=10016  cancel_sent   u000052394 - px=- qty=-
11090.508727 bid=10015 ask=10016  cancel_sent   u000052395 - px=- qty=-
11090.508727 bid=10015 ask=10016  cancel_sent   u000052396 - px=- qty=-
11090.508727 bid=10015 ask=10016  FILL         u000052392 A px=10015 qty=102 taker=3082848  inventory_after=-100
11090.513727 bid=10015 ask=10016  cancel        u000052393 B px=10014.0 qty=58.0
11090.513727 bid=10015 ask=10016  cancel        u000052394 B px=10014.0 qty=18.0
11090.513727 bid=10015 ask=10016  cancel        u000052395 B px=10014.0 qty=1.0
11090.513727 bid=10015 ask=10016  cancel        u000052396 B px=10014.0 qty=21.0
11090.514126 bid=10015 ask=10016  add_sent      u000052397 B px=10015.0 qty=200.0
11090.519126 bid=10015 ask=10016  add           u000052397 B px=10015.0 qty=200.0
11091.091598 bid=10015 ask=10016  add_sent      u000052398 A px=10016.0 qty=59.0
11091.091598 bid=10015 ask=10016  FILL         u000052397 B px=10015 qty=59 taker=3082976  inventory_after=-41
11091.096598 bid=10015 ask=10016  add           u000052398 A px=10016.0 qty=59.0
tape rows inside the gap: 231
```


Reading: nothing is wrong with the re-quote logic. In the bid gap the sweep at 3670.383635 fills three bids and takes inventory to +100 (bid free capacity 0); the bid returns only when the ask fill at 3670.624 lowers inventory to 91 and exactly 9 lots are sent. In the ask gap inventory goes to -100 at 11090.508727 and the 59-lot ask is sent only after the bid fill brings it to -41. The replacement quantity equals the freed capacity.
