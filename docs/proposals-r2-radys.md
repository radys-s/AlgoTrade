# R2 strategy proposals for Honza's Claude (radys, 2026-11-07 ~21:45)

Goal: R2 public tape 101,933 (mm_r2_fair) / Honza ~200k -> **500k+**. Leaderboard 20:31: median 55k, 75th pct ~400k.
Round closes ~00:15 on 2026-11-08. Every idea below has: why, exact spec, grid, acceptance. Ordered by expected value.
Read with: `docs/radys-findings.md` §10-§14, `docs/prompt-fable-r2-fair.md`.

**Acceptance for any upload:** R2 pnl > current best, >= 6/8 chunks positive, penalty 0 (unless the idea is an
explicit over-cap one and the penalty is counted), R1 not negative, sandbox check OK, single self-contained file
(no imports of other strategy files — bake parent classes in before upload).

---

## 0. The one diagnosis that drives everything

`mm_r2_fair` has |I| = 100 at p50/p90/p99. Consequences:
- The growing side has **zero capacity** (I + OUT_buy <= 100), so only the unload side quotes. Half of the book
  presence is gone. Rungs/ladder add nothing (ladder 15 = +1k) because rungs also need capacity.
- 4.8k fills vs the R1 winner's 426k. The pnl is mostly "inventory rides the 30 s trend", not spread capture.
- Honza's front-of-queue oracle: gate maker 84k -> 232k. Queue position is worth +150k, but it needs free capacity.
- Fill markout by fair signal (home `research/diag/fair_fills.py`): fills WITH the signal +106k at 1 s, AGAINST
  -305k. The filter removes the loss side; the win side is capacity-limited.

=> **Capacity, not signal, is the binding constraint.** Every top idea below frees capacity or uses it better.

---

## A. Free capacity: turn inventory over (highest priority)

### A1. Fair-signal inventory unload / taker overlay — `strategies/exp_r2_take.py` (radys, written, not yet swept)
- Why: when long 100 and fair is below mid (sig < -x), the price will fall through our position; selling at the
  bid now (pay ~0.5 tick half-spread) beats waiting passively, and frees 100 lots of buy capacity at lower prices.
- Spec (already in the file, flag-off = identical to mm_r2_fair):
  `take_exit_thr` (reducing IOC when sig against inventory), `take_entry_thr` (growing IOC with sig, room only),
  `take_size`, `take_cool_us`, `take_max_spread`. If our own quote holds the room, the quote is pulled first.
- Grid: take_exit_thr {0.8, 1.2, 1.6, 2.0, 3.0} x take_entry_thr {0, 2.0, 3.0} x take_size {20, 50, 100}
  x take_cool_us {20000, 100000}. Use `--halving 0.1,0.3`.
- Offline numbers supporting it (tau 50, event deduped 1 s, crossing both ways, gap-free mid):
  |sig| > 2: +1.9 ticks/lot at 300 ms; > 3: +3.9; > 5: +11 (31 events). |sig| > 1.0 non-overlapping was slightly
  negative as a pure taker — but an UNLOAD trade avoids the adverse move on 100 lots we already hold, so its
  break-even threshold is lower than a fresh entry. That's the thing to measure.
- Before upload: bake mm_r2_fair into the file (it imports it via importlib now).

### A2. Lower inventory target without losing the trend
- `max_position` stays 100, but split the cap: **trend inventory** (what the gate wants, e.g. 60) + **MM room**
  (40 lots both ways). Implement as: the gate pulls the against-trend side only while sign*I >= trend_target;
  below that, quote both sides with the fair filter.
- Grid: trend_target {40, 60, 80, 100(=current)} x quote_size {50, 100, 200}.
- Why: keeps most of the trend pnl and restores two-sided spread capture + front-queue rungs.

### A3. Inventory skew toward zero (cheap check, running on radys PC now)
- `skew_per_lot` {0, 0.02, 0.05} x gate_thr {0, 4} x fair_thr {0.3, 0.6, 1.0} x quote_size {50, 200}.
  Results will be appended below (§Results). If gate 0 + fair filter + skew is positive, the "R1 champion loses on
  R2" problem was trend exposure, and the fair filter fixed it.

### A4. Passive unload at a better price when pinned
- When |I| = 100, put the unload quote ONE TICK INSIDE the spread when spread >= 2 (we are first in queue, fill
  sooner), and at the touch when spread 1. Today it waits at the back of a 1000-lot touch queue.
- Param `unload_improve` {0, 1}. Cheap.

---

## B. Use queue position (front-of-queue oracle = +150k)

### B1. Rungs only on the side with capacity + fair-gated
- Rungs at touch-1..touch-3 placed >= 1 s before the level becomes touch are at the front 84 % of the time (2 s: 97 %).
- Spec: ladder_size {10, 20, 40}, ladder_depth {2, 3, 5}, ladder_reserve {0, 1}, **with A1 or A2 active** (otherwise
  capacity is 0 and rungs never exist — the reason ladder 15 gave only +1k).
- Cancel a rung when fair says the price will run THROUGH it by more than 1 tick (sig < -(depth+1) on bids).

### B2. Keep the touch quote instead of re-joining at the back
- `park_size` stubs were worth +27k on R1 (mm_park). R2 gate86 has park_size 0. Re-grid park_size {0, 10, 20, 40}
  x park_depth {1, 2} on top of mm_r2_fair + A1/A2. Queue age is the edge; never cancel an old order if you can
  shrink it with modify_order (keeps priority).

### B3. Fair filter as SHRINK not CANCEL
- Today the fair filter cancels the whole side when sig < -0.3, losing queue place. Instead modify_order down to a
  1-5 lot stub (keeps the place), grow back with a NEW order when sig recovers (new order goes to back — so only
  if it's worth it). Grid stub {0 (=cancel), 1, 5}. On R1 the 5-lot stub ML version gave +8.4 %.

---

## C. Better signal (multiplies everything above)

### C1. Better F estimator
- Current: time-decayed mean of (insert price - side offset - mid), tau 50 ms, top 3 levels.
- Try: tau {10, 20, 30, 80}; median of the last N {5, 10, 20} inserts; weight by insert size; include deeper levels
  (fair_levels {3, 5, 10, 20}); separate bid/ask offsets estimated in a rolling window (offsets drift with regime);
  combine two taus (fast for filter, slow for gate). Target: corr with mid move t+5 ms -> 100 ms > 0.41, -> 1 s > 0.21.
- Cancels: a cancelled order far from mid may have been placed when F was there -> try "price of cancelled
  orders" as a second, lagged F reading.

### C2. Asymmetric thresholds by inventory
- Pull the GROWING side at a lower |sig| (0.2) and the UNLOAD side only at a higher |sig| (0.8): unloading is good
  even with mildly adverse fair. Grid fair_thr_grow {0.2, 0.3, 0.5} x fair_thr_unload {0.3, 0.8, 1.5, inf}.

### C3. Gate on F drift, not mid drift, with hysteresis
- drift of the slow F (tau 1-5 s) over windows {5, 10, 15, 30} s; enter at gate_thr {3, 4, 6}, exit at half.
  Mid drift lags the generator's F; gate flips late -> the 4-tick threshold eats the first part of every trend.

### C4. Jump handling
- R2 has vertical jumps (~0:27, 0:57, 1:30) and ~1,207 rows with an empty side. Mid is garbage there (63 % of mid
  variance is these gaps). Every component must use gap-free mid (forward-fill only while spread <= 3).
- During a one-sided gap: do NOT quote the side that is empty on the market (we'd be the only liquidity, adverse).
  Measure: our fills during spread > 3 and their 1 s markout.

---

## D. Over-cap (flat penalty — use with care)

- Fact (home): penalty is a FLAT 10,000 $/s while |I| > 100, regardless of how far. Exchange rejects only
  potential |pos| > 1000.
- radys check (offline, tau 20/50, events deduped 1 s): |sig| > 8 happens ~20 times per run, mean +15..17 ticks/lot
  at 300 ms, but median only +2..3.5 and depth over 5 levels is only ~60-90 lots. 20 x 16 ticks x 100 lots ≈ 32k
  gross, minus 10k per second over cap. **Not the 400k.** Only worth it as: over-cap for <= 200 ms on |sig| > 8
  with size = available depth. Low priority.
- Over-cap MM (max_position 150-300 with penalty) — R1 no-limit test: +100..+180k capacity value but ~4,600 s over
  100 -> millions of penalty. Rejected unless over-cap time can be kept to a few seconds total.

---

## E. Things already rejected (don't redo)

- Deep resting orders to catch IOC sweeps (radys §14): adverse at the median, sweeps continue.
- Pure fair taker with low threshold (Copilot, non-overlapping): thr 1.0 slightly negative.
- Gemini rule "quote bid only when sig > s" — inverted, wrong.
- IOC unwind by threshold (R1): crossing cost > freed room. (A1 differs: unload is triggered by the fair signal.)
- R1 champions on R2: -148k..-230k (trend exposure).
- Taker on flow / returns: break-even (home).

---

## F. Process for the remaining ~2.5 h

1. Each idea = copy the champion, add a flag (off = parity, check the pnl equals the champion exactly), sweep with
   `research.run --halving 0.1,0.3 --workers 11`, judge with chunks.
2. Upload every new best immediately (bake params + parent classes into one file, `research/check_submission.py`).
3. Order: A1 + A3 (running) -> A2 -> B1/B2 on the winner -> C2 -> C1/C3 -> D.
4. Log each result (even failed) in `docs/progress.md`.

## Results (appended as they arrive)

- (pending) A3 turnover grid on radys PC: 36 runs, gate {0,4} x fair_thr {0.3,0.6,1.0} x skew {0,0.02,0.05} x
  quote_size {50,200}.
