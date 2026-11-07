# Prompt for Honza's Claude (Fable): round 2 workflow — goal 1st place

Paste everything below the line into Honza's Claude session on the home PC.

---

We finished round 1 12th (511k) while the leader made 793k (+55 %). Our models were incremental filters on one
architecture. For round 2 the goal is **1st place**. Change the workflow as follows and keep to it.

## Facts from round 1 you must build on (details: `docs/radys-findings.md` sections 7-9, `docs/progress.md`)

1. Simulator: replayed non-reactive tape, 5 ms latency, price-time priority, no fees. `modify_order` down keeps queue
   priority. The exchange rejects only POTENTIAL |position| > 1000; |I| > 100 costs a flat 10,000 $/s (prorated per
   µs), regardless of size. Score = private tape (different from the public one); local sim = server exactly.
2. No directional edge: the round-1 tape spans 105 ticks, a 2 s round trip averages -1.07 ticks/lot, big sweeps are
   Poisson and unpredictable (your sweep study, AUC 0.50). Profit = passive spread capture.
3. Champion fills (mm_ml_r1tuned): 0 lots ahead +0.336 $/lot (43 % of volume), 500+ ahead -0.26..-0.48; nobody behind
   us -0.15, 500+ behind +0.32; inventory-adding fills only 22 % of volume yet +0.18..+0.24 $/lot; |I| median 60.
4. **Every rule that reduced resting size lost** (pulls, thin, back-pull, ahead_stub -150k, fresh-size gate -212k,
   inventory lockout). **Every change that added resting size won** (quote 100->200 +33 %, parking +6 %, ML stub via
   modify +8 %). The edge is capacity + queue age, not filtering.
5. Capacity-value diagnostic (`research/diag/nolimit.py`, penalty switched off): see the table appended at the
   bottom of this file. It shows how much the 100-lot limit costs us; that gap is the prize.

## Workflow (strict order)

**Phase 0 (first 20 min of the round):** `research.round` kickoff, run the current champion + the stack
(`exp_overcap.py`, ml_thr/ml_stub refit with `research.refit_ml`, ml_tight_only=1, park_max 10, park_size 15),
upload the better one as insurance. Write the score into `submission/README.md`.

**Phase 1 (structure, ~70 % of compute, before any filter tuning):** each item is a flag in ONE experiment file,
default off = champion parity, screened with `--halving 0.25` on 20 workers, best on full tape, judged by
`research.compare` (chunks + holdout + sample_v2).
- S1 **Breach economics.** Over-cap with instrumentation: breach episodes, seconds over 100, penalty $, cut cost $,
  extra fill pnl, per max_position 120/150/200 and cut style (IOC through the touch / modify-down of the adding side /
  passive only). Find the configuration where extra queue capacity earns more than the breaches cost. The round-1
  over-cap screen (-133k on 25 %) was never diagnosed; the ledger was not written. Do not reject it without the numbers.
- S2 **Inventory turnover.** Keep I near 0 so BOTH sides can hold 100 lots: e.g. only the reducing side may jump into a
  2-tick gap, small IOC unwind only while the reducing side's queue ahead is long, ML stub disabled on the reducing
  side when |I| > X. Metric: share of time |I| < 30 and pnl.
- S3 **Budget allocation across queue spots and levels.** The 100 lots per side are split between the touch bulk,
  parked stubs and orders one level behind the touch (they gain queue age before becoming the touch). Only modify DOWN,
  never cancel an aged order. Grid: share of budget at L2 0/20/40 %, park_max up to 10.
- S4 **Data-driven sizing table, up to ~0.9 MB in the file.** Build a fill-EV lookup table from fills of several
  configs (`research/diag/queue_fills2.py` saves `logs/fills_*.npz`): bins of lots ahead, lots behind, level qty,
  ML pred, inventory sign, spread. Use it ONLY to decide how many NEW lots to add per side (never to shrink aged
  orders). Validate on pnl, not IC (your MLP with better IC lost).

**Phase 2 (filters, only after Phase 1 has a winner):** ML threshold refit per round, stub size, park params, stack.

**Rules**
- Leaderboard check every 30 min. If the leader is > 20 % ahead after 2 h, stop tuning and move to the next
  Phase-1 lever.
- Never declare a ceiling unless it is a hard simulator fact, with the number.
- File budget: < 1 MB (aim < 0.95 MB). Runtime: < 7 min projected for 8.8 M rows (`research.check`).
- Robustness: the private tape differs; accept only changes that win >= 6/8 chunks AND do not lose > 5 % on
  sample_v2 / the previous round's tape. No timestamp lookups of the public tape.
- Coordination: you own `strategies/exp_*` files you create on `home`; the radys window reviews and runs
  diagnostics on its PC. Put winners into `submission/best_strategy.py` with the pnl in the README table, log every
  experiment in `docs/progress.md` (English), push after each result.

## Capacity-value diagnostic (round 1, mm_ml_r1tuned params, penalty off)

| max_position (penalty OFF) | pnl | breach episodes | seconds over 100 |
|---|---|---|---|
| 100 (champion) | 497,292 | 0 | 0 |
| 150 | 598,403 (+101k) | 7,945 | 4,628 |
| 200 | 634,727 (+137k) | 7,986 | 5,657 |
| 300 | 677,830 (+181k) | 7,689 | 6,695 |

**Reading:** extra capacity is worth +100..+180k on round 1, which is the gap to the leaders. Buying it by breaching is
impossible (7,900 episodes x latency penalty + cut cost >> gain). It has to come from **inventory turnover**: keep
|I| near 0 so the adding side holds ~100 lots instead of ~40 (|I| median is 60). This makes S2 the top priority, S1
second. The old IOC unwind never fired because the reducing side is always full (I + OUT <= M blocks it);
`exp_overcap.py` now has `unwind_slack` (extra potential room for the unwind IOC only). radys is sweeping
unwind_threshold x unwind_size with slack 40 (ledger `*r1-unwind.csv`). Value of 1 lot of capacity is roughly
0.17 $/lot/s, an IOC unwind costs ~0.5-0.6 $/lot: unwinding pays if the freed room lasts > ~4 s.
