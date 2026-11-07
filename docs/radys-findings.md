# Findings from the radys window (for Honza's Claude)

Last update: 2026-11-07 14:15, Round 1 (`data/round1.csv`, release `data-round1`). Read this before
designing the next market-making experiment. Every number is from the real simulator on the full round1 tape
unless marked otherwise; ledgers are in `research/results/*radys-s*`.

## TL;DR

- Champion = `strategies/exp_full_capacity.py` (Honza: mm_v2 + quote_size 200): **round1 pnl 425,708**
  (mm_v2 there: 321,024). The +33 % came from Honza's quote_size 200, not from the radys experiments.
- **Every mechanic that cancels or withholds a touch quote lost pnl on round1** (imbalance pull, flow pull,
  thin-level pull, requote tolerance, skew, unwind, second level). 60+ runs, best +0.2 % = noise.
- Why: **queue position is the edge**, and every cancel sends us to the back of a ~900-lot queue.
  The next gains must come from keeping or gaining queue priority, not from filtering fills.

## 1. Where the edge is (round1, champion fills)

`research.mm_edge` (hypothetical touch maker, h = 1 s): front-of-queue +0.254 tick/fill, back-of-queue
(filled only when the level is depleted) -0.219. By lots ahead of us: Q=0 +0.254, 50 +0.164, 200 +0.001,
500 -0.213, 1000 -0.432.

Champion's own fills (50,754 fills, 2.65M lots; script measured our live queue position from the snapshot,
where `owner == 'user'` marks our orders):

| market lots AHEAD of our order | share of our lots | 1 s edge $/lot |
|---|---|---|
| 0 | 52 % | +0.207 |
| 1-50 | 29 % | +0.168 |
| 50-200 | 13 % | +0.063 |
| 200-500 | 4 % | -0.041 |
| 500+ | 1 % | -0.22 .. -0.52 |

| market lots at our level (thickness) | share | 1 s edge |
|---|---|---|
| < 100 (about to be depleted) | 13 % | -0.12 .. -0.49 |
| 100-300 | 13 % | -0.01 |
| 300-1000 | 33 % | +0.12 .. +0.19 |
| > 1000 | 40 % | +0.32 |

Other splits: spread-1 fills +0.15, spread-2 fills +0.53. Fills that ADD to |I| -0.015 (517k lots),
fills that REDUCE |I| +0.20 (2.13M lots). Markout 10 ms 0.40, 1 s 0.16, 60 s 0.16 (no further decay).
Market G orders live only ~337 ms (median), so a quote that simply WAITS climbs to the front by itself:
that is why 52 % of our lots fill with nothing ahead.

## 2. Tested on round1 and rejected (do not retest blindly)

| mechanic (flag, file) | best setting | pnl | vs 425,708 |
|---|---|---|---|
| L1 imbalance pull (`imb_pull`, exp_imb_pull/exp_layers) | 0.9 | 377,309 | -11 % (0.5: -70 %) |
| requote tolerance, keep quote behind touch (`requote_tolerance`) | 1 | 310,838 | -27 % |
| pull after a big aggressive trade (`flow_pull_qty/us`, exp_flow_pull) | 1000 lots / 50 ms | 426,506 | +0.2 % (noise) |
| pull a side when its level is thin (`thin_pull`, exp_queue) | 30 lots | 391,601 | -8 % (300: -65 %) |
| second resting level (`layer_size/offset`, exp_layers) | 30 lots, 1 tick | 426,034 | +0.1 % (noise) |
| one-sided +1 tick at spread 2 (`jump_imb`) | 0.6 | ~-3 % | worse |
| quote_size 300 | - | = 200 | saturated (M + abs(I)) |
| max_position 90 / 80 | - | -10 % / -9 % | worse |
| inventory skew (`skew_per_lot` 0.01, sample_v2) | - | -33 % | worse |
| IOC unwind at abs(I) >= 90 (sample_v2) | - | -10 % | worse |
| taker family (`research.taker_scan` on round1, 1152 configs) | ltp_dev | +206 | nothing |

All flags default to off (= champion, parity checked: off reproduces 425,708 / 237,306 exactly).

## 3. Ideas not yet tried (ranked by expected value)

1. **Shrink instead of cancel.** `modify_order(order_id, new_qty)` reduces size and (per the simulator,
   check `simulator.py` modify path) keeps the order in place. When the quote has to move (touch improved
   past us), keep a small remainder at the old level with its queue priority and use the freed room at
   the new touch. Same for any "pull": shrink to a stub instead of cancelling, so we never lose the slot.
2. **Pull only when we are at the BACK.** The snapshot shows our queue position. Cancelling an order with
   > 200 lots ahead costs nothing (its fills are negative anyway); cancelling one at the front is what
   killed every filter above. Combine: pull only if lots ahead > X AND the level is thin / imbalance against us.
3. **Be first on new levels.** Fills with 0 ahead are +0.21; when the touch moves, the new level is short.
   Join it as fast as possible (we already do), and never let a front position go.
4. **Fill-quality model (phase 2).** Linear/ridge model of 1 s markout from (lots ahead, level thickness,
   L1 imbalance, flow, spread), weights embedded as literals (file limit 1 MiB, ours ~18 KB). Only useful
   together with 1-2: the decision must not cost queue priority.

## Tools from this window

- `research/analyze_tape.py` — 1-minute tape overview (spread, momentum vs reversion, signal correlations).
- Queue/fill diagnostics: subclass the strategy, record per fill the market lots ahead/behind our order from
  the snapshot (`owner != 'market'` rows are ours), join with mid 1 s later. Script:
  `uv run python -X utf8 research/diag/queue_fills.py data/round1.csv strategies/exp_full_capacity.py` (~2 min).
- Quick tests: `research.run ... --halving 0.25 --keep 0.25` (screen on the first 25 %, full run for the best).
