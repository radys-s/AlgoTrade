# Round 1 front-of-queue oracle (diagnostic, not shippable)

Patch: `research/diag/sim_frontqueue.py` monkey-patches `OrderBook.insert` so a user order rests ahead of all market orders at its price (behind our own earlier orders). Everything else unchanged. Patch disabled reproduces mm_r1_final2 = 514,093 exactly. data/round1.csv, full tape.

| run | pnl | fills | lots | pnl/lot | breach | inv q95 / q997 |
|---|---|---|---|---|---|---|
| mm_r1_final2 normal | 514,093 | 40,601 | 1,968,282 | 0.261 | 0 | 100 / 100 |
| mm_r1_final2 FRONT | 1,395,933 | 72,196 | 3,839,706 | 0.364 | 0 | 100 / 100 |
| exp_full_capacity normal (ledger) | 425,708 | | | | | |
| exp_full_capacity FRONT | 1,301,334 | 57,568 | 4,486,350 | 0.290 | 0 | 100 / 100 |
| mm_park normal (ledger) | 452,651 | | | | | |
| mm_park FRONT | 1,301,703 | 68,788 | 4,067,898 | 0.320 | 0 | 100 / 100 |
| mm_r1_final2 ml_thr 9, normal | 463,252 | 51,365 | 2,278,096 | 0.203 | 0 | 100 / 100 |
| mm_r1_final2 ml_thr 9, FRONT | 1,311,712 | 80,276 | 4,037,426 | 0.325 | 0 | 100 / 100 |
| mm_r1_final2 FRESH-only front (<100 market lots at level at placement) | 546,305 | 42,040 | 2,040,762 | 0.268 | 0 | 100 / 100 |

FRESH variant: 12,840 of 68,921 placements got the front.
Chunk pnls (6 train chunks) are steady in all runs (FRONT final2: 177k-188k each), so not one lucky stretch.
Caveat: front priority also creates more zombie liquidity (market orders we jump stay in the book), which flatters markout somewhat; it is an upper bound, not a prediction.

## Verdict
1. Front-of-queue ceiling is 1.30-1.40M, roughly 2.5-3x our normal 514k and about 2x the best competitor (650-790k): queue priority alone can explain their whole gap.
2. It is not the ML filter: with the filter off, front gives 1.31M vs 1.40M with it; the filter adds ~85k under front vs ~51k normally.
3. Priority value comes mostly from deep queues: fresh-level-only front gives just +32k (546k); ~850k of the ceiling is from jumping existing queues.
4. Fills and lots roughly double (40.6k to 72.2k fills; 1.97M to 3.84M lots) and pnl/lot rises 0.26 to 0.36, so both volume and edge improve.
5. Round 2 effort should go to queue position (join early, quote on new/improved levels before the queue forms, never sit behind big queues, re-place when depth builds ahead), not to more signal/filter work.
