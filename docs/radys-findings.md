# Findings from the radys window (for Honza's Claude)

Last update: 2026-11-07 14:15, Round 1 (`data/round1.csv`, release `data-round1`). Read this before
designing the next market-making experiment. Every number is from the real simulator on the full round1 tape
unless marked otherwise; ledgers are in `research/results/*radys-s*`.

## TL;DR

- **Origin of each strategy:** `submission/best_strategy.py` = **our model (radys window, Tomáš's PC)**
  `strategies/mm_park.py` = Honza's `exp_full_capacity` core + parked 20-lot stubs: round1 **452,651**.
  Previous champion = **Honza's model** `exp_full_capacity.py` (425,708). Not uploaded yet (Tomáš waits for a bigger jump).
- `research.compare` mm_park vs exp_full_capacity: round1 +26,943, holdout +7,605, 8/8 chunks -> ACCEPT;
  **sample_v2 -12,382 (-4.7 %, 0/8 chunks)** -> the park mechanic helps on momentum tapes (round1) and costs a
  little on mean-reverting ones (v2). Watch this on later rounds.
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

## 4. Results of the ideas above (radys, 14:40)

| idea | file / flag | screen (25 % halving, then full round1) | status |
|---|---|---|---|
| #1 park a stub instead of cancelling | `strategies/exp_shrink.py`, baked as `strategies/mm_park.py` (park_size 20, park_depth 1, park_max 3) | **452,651 full (+6.3 %)**, holdout 117,886 (+6.9 %), 8/8 chunks; plateau: park 20 = 447-452k at depth 1/2/4, park 10 = 443k, park 50 = 440k | **in submission** (compare ACCEPT: round1 8/8, sample_v2 -4.7 %) |
| #2 pull only at the back of the queue | `strategies/exp_backpull.py` (`back_ahead`, `back_imb`) | every setting < off (screen 25 %) | rejected |
| leaders #1 standing ladder | `strategies/exp_ladder.py` (`ladder_size/levels/far/first`) | best 452,244 (= off); ladder placed before the touch quote -15..-60 % | rejected |
| ML shrink (ridge, 100 ms) | `strategies/exp_ml_shrink.py` (`ml_thr`, `ml_stub`) | running | - |

**Overfit check of mm_park (blocked k-fold, `research/diag/kfold.py`, ledger `r1-park-kfold`, 12 full runs):**
for each of 9 time blocks (8 chunks + holdout) the park setting is chosen on the other 8 blocks only.
Out-of-fold total 452,550 vs baseline 425,708 (+6.3 %), in-sample best 453,321 (optimism gap 0.2 %), the
out-of-fold pick beats the baseline in 8/9 blocks (the loss: -150), picks are park 30 (8x) / park 20 (1x).
-> not overfit on round1. Remaining risk = market type (sample_v2 -4.7 %).
**ML signal (`research.signals`, ridge, 100 ms horizon):** walk-forward IC +0.372..+0.387 in all 7 folds, top-10 %
edge 0.30 tick; coefficients stable across folds -> real, not overfit. Weights embedded in exp_ml_shrink.py.

## 5. Heavy experiments for Honza's compute (not run in the radys window)

These need large grids or long training; the radys laptop does only quick 25 % screens. Each starts from the
best baked file (now `mm_park.py`) and keeps the new mechanic behind a flag that is off by default.

1. **Park-stub fine grid on all datasets.** park_size 10-40 step 5 x park_depth 1-4 x park_max 1-6, plus
   quote_size 150/200/250, on round1 + sample_v2 (+ every later round). Pick the plateau centre with
   `research.ledger --robust`. ~150 full runs.
2. **Park + back-pull combined.** back_ahead {100, 200, 300, 500} x back_imb {0.3, 0.5, 0.7} on top of the
   park winner. Interactions matter: parking keeps front positions, back-pull only drops back positions.
3. **Fill-quality model (walk-forward).** Label = 1 s markout of each touch fill (`research.mm_edge` data);
   features = lots ahead of us, level thickness, L1 imbalance, microprice, flow 100 ms / 1 s, spread, abs(I).
   Train ridge then a small MLP (`research.train_mlp`) on chunks 1..k, test on k+1 (walk-forward, 8 folds).
   Use it only to decide when to SHRINK (modify) a quote, never to cancel a front order. Export weights as
   literals (file limit 1 MiB; inference must stay < ~50 us/event: 4.4M events in a 15 min, 1 CPU sandbox).
4. **Queue-aware sizing.** Size the touch quote by level thickness (> 1000 lots: +0.32 $/lot, < 100: negative):
   size = f(level qty) instead of constant 200. Grid over 3-4 breakpoints.
5. **Regime switch per round.** v2 (mean-reverting, spread 2 for 58 %) and round1 (momentum after trades,
   spread 1 for 93 %) want different settings. Detect the regime online (spread share, trade-follow markout
   over the last N s) and switch parameter sets; validate by training on one dataset, testing on the other.
6. **Stress tests for the final pick.** The same strategy on: round1 shifted in time, round1 reversed in
   price, round1 + sample_v2 concatenated, a 2x longer tape. Reject anything that is not positive on all.

## 6. What the leading teams probably do (hypotheses to test)

Evidence: on the playground leaderboard the top-2 curves had the same shape as ours (smooth, no blow-ups),
just ~20 % steeper -> they market-make too, with more edge per second. On a replayed tape the edge beyond
plain quoting comes from queue priority and from the replay artefacts (`.claude/skills/arena-market-mechanics`).

1. **Standing ladder (pre-positioned queue priority).** Keep small orders permanently on touch-1 .. touch-4
   and never move them; market orders live ~337 ms, so a ladder order that waits is at the FRONT when the
   price arrives (Q = 0 fills: +0.21 $/lot). Our `layer` test lost because the layer was re-quoted with the
   touch (lost priority every tick); a ladder that stays at fixed prices is a different mechanic. Cap the
   total by the invariant (ladder size counts in OUT_k).
2. **Zombie liquidity.** When our order is ahead of a market order, takers fill us and the market order
   stays in the simulated book with no tape cancel. The simulated book becomes thicker where we quote early,
   which protects our stubs (thick level behind us = +0.32 $/lot). Measure: book qty at our level vs the
   historical qty; quote early on levels where we create zombies.
3. **Repeated participants by order id / lifespan / size.** The tape has order ids and lifespans; informed
   flow (big IOC sweeps, max Insert 124,248 lots in round1) may come from recognisable patterns (size
   buckets, specific Lifespan values, Insert-then-IOC sequences). Classify the next taker's size from the
   recent pattern and shrink the side it will hit just before it lands (5 ms latency is the constraint).
4. **Sweep anticipation through queue depletion.** Cancels and small takers eating the touch queue ahead of
   us precede the sweep that moves the price. Rate of depletion of the touch (lots/ms) as a feature for the
   shrink decision (not cancel).
5. **Inventory turnover instead of skew.** Fills that reduce abs(I) earn +0.20 $/lot, adding fills -0.015.
   Leaders likely keep the reducing side at full size (we do, quote 200) AND shrink the adding side when
   abs(I) is high, without moving its price (moving price lost every test).
6. **Per-round regime detection.** The round data differs a lot (v2: mean-reverting, spread 2 for 58 %;
   round1: momentum, spread 1 for 93 %); a team that adapts settings online beats one tuned on public data.

## Tools from this window

- `research/analyze_tape.py` — 1-minute tape overview (spread, momentum vs reversion, signal correlations).
- Queue/fill diagnostics: subclass the strategy, record per fill the market lots ahead/behind our order from
  the snapshot (`owner != 'market'` rows are ours), join with mid 1 s later. Script:
  `uv run python -X utf8 research/diag/queue_fills.py data/round1.csv strategies/exp_full_capacity.py` (~2 min).
- Quick tests: `research.run ... --halving 0.25 --keep 0.25` (screen on the first 25 %, full run for the best).
