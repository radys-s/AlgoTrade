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

- **A3 turnover grid — REJECTED** (36 runs, ledger `research/results/20261107-211334_radys-s_r2-turnover-grid.csv`):
  best is still the champion (gate 4, fair 0.3, q 200, skew 0) 101,933. skew 0.02 -> 20k, 0.05 -> 4k; gate off ->
  +50k at best (q 50, fair 0.3), down to -500k. fair_thr 0.6 / 1.0 worse (92.7k / 85k). Lesson: **the trend
  inventory IS the edge; blind turnover destroys it.** Free capacity only at signal-timed moments (A1), not by skew.
  => A2 (split cap) is lower priority than thought; try trend_target 80-100 only.
- **A1 fair-signal unload — WORKS**: exp_r2_take take_exit_thr 1.5 (size 50, cool 20 ms, entry off) = **113,491**
  (+11.6k), 8/8 chunks, holdout 43.6k -> 47.9k, penalty 0. Flag-off parity exact (101,933). Baked single file:
  `strategies/mm_r2_take.py`.
- **A1 full sweep (132 runs, `research/results/20261107-212138_radys-s_r2-take-sweep.csv`) — bigger than 1.5**:
  exit_thr 1.0 + size 100 + cool 100 ms + entry off = **131,984** (+30,051 vs mm_r2_fair), 8/8 chunks,
  chunk_min +2,513, holdout 48,714, penalty 0, 3,987 fills. exit 1.0/size 100/cool 20 ms = 128,847.
  Shape of the surface: **cross the whole pinned side in one go, rarely** (size 100 > 50 everywhere, cool 100 ms >
  20 ms, exit 1.0 > 1.2 > 1.5 > 2.0). `take_entry_thr` adds nothing (3.0 = 132,086 vs entry off = 131,984, +0.1 %,
  noise) — confirming there is no capacity for entries; unload only. `take_max_spread` 1 is best (a wide or one-sided
  book has a garbage touch to cross into).
  => Baked as `strategies/mm_r2_take.py`, R2 131,984; this is the champion to beat.
- **B1/B2 queue position on top of the unload — REJECTED** (18 runs, `20261107-213132_radys-s_r2-park-ladder.csv`):
  park_size 20 costs -16.4k (115,564) and park_depth 2 -23.5k (108,480). Standing rungs at touch-1..-3 are flat:
  ladder 15 = 131,773 (-211), ladder 40 = 131,912 (-72, holdout +1,198 = noise). Best run is the champion itself.
  Why: with the unload in place the pinned side is already crossed out, so a parked stub only re-enters at a worse
  price and a rung only ties up capacity the unload wants; the queue-age edge of R1 does not survive on R2 because
  the R2 touch queue is 1,000+ lots (R1 was ~100). **Do not spend more time on park/ladder on R2.**

## Results 22:25 — fq2 base (server 740,067 / local 349,077): what is dead

- The local x2.12 = server scaling means local ranking is what matters; all numbers below are local round2.
- **Dead on the fq2 base**: asymmetric take margins (-1.1 %), hardcoded generator offsets (-3.6 %), fast taker tau
  (screen), taker slippage 1-2 ticks (-18 % / -25 %), fair_levels 10/30 (inert, identical pnl), gate_thr 12 with
  10 ms cooldown (+1.5 % total but holdout -16.8k => overfit).
- **Dead on the maker-only base** (mm_r2_take 131,984): park stubs (-16.4k), ladders (-0.2k), inside-spread unload
  (-1.5k), fair-stub instead of cancel (-2.3k). Asymmetric MAKER thresholds +2.3k were the only positive.
- **Still untested and worth a home-PC run**: (a) trend-conditional take margin — crossing margin reduced by
  `take_gate_bonus` in the 30 s trend direction and increased by `take_counter_penalty` against it; (b) thin-book
  deep sweep — when the opposite touch holds < N lots, sweep 1 tick deeper (the touch is thin exactly when a sweep
  pays); (c) gate window 15 s vs 45 s; (d) F from un-clamped inserts only (46 % of inserts are clamped at the
  opposite touch, so half the devs are censored — drop them and see if the signal sharpens).
- **Radys special finding**: the oracle says a perfect 150 ms predictor is worth 935k on the fq2 skeleton, and every
  realistic predictor tried (ridge blends, tau mixes, F velocity) lands at 244-273k. So the remaining value is not in
  a better estimator but in better USE of the same estimator at the decision points: trend-conditional margins and
  sweep depth on thin touches are the two untested uses.

## Results 22:45 — the gate window is the only lever left on the fq2 skeleton

- `gate_window_us` 15 s (was 30 s): **358,764 (+2.8 %)**, holdout +476. 5 s/10 s/45 s are all worse.
- `gate_thr` 9 at 15 s: **363,943 (+4.25 %)**, holdout +817, 8/8 chunks, chunk_min +17,021, penalty 0.
  thr 7 = 351.5k, thr 10 = 357.5k (holdout -6.5k), thr 11 = 356.4k (holdout -9.2k), thr 12 = 354.4k (holdout -16.8k):
  above 9 everything is train-pnl buying. The gate must be FAST (15 s) and MEDIUM (9 ticks), not slow and loose.
- `take_k_spread` 0.1 (bigger crossing margin when spread >= 2): +2.0k, holdout +911. On a one-sided/wide book the
  touch is stale, so crossing needs more edge — this is the "one-sided book" fact (D6) finally paying.
- Dead this wave: dropping clamped inserts (-26 %; the censored devs still tell you the level), take persistence
  (-1.5 %), F-drift gate (neutral), taker slippage (-18/-25 %), fq_take_k 0.18/0.22.
- **For Honza's Claude (home PC, needs >15 min)**: (1) re-run the whole fq2 sweep with `gate_window_us` 15000000
  *inside* every grid (the 15 s window may move the optimum of fq_take_k / fq_take / gate_max_spread; my sweeps
  fixed 30 s for all of them); (2) the gate and the fair signal are redundant in principle — the gate is a slow
  proxy of F drift — so try removing the gate entirely at the 15 s horizon with a *lower* fair threshold instead;
  (3) the crossing margin should probably also scale with the book's one-sidedness (spread 2-3 vs 10-80 ticks are
  different regimes) — `take_k_spread` only handles spread >= 2.
- **A1 sweep** (90 sets, halving; ledger `research/results/*r2-take-sweep*`): best take_exit_thr 1.0, take_size 100,
  cool 100 ms = **132,086** (8/8, holdout 48.7k); entry taker adds ~0. Superseded by Honza's fair taker line
  (mm_r2_fq2 -> mm_r2_pred -> mm_r2_zd, leaderboard 740k -> 798k -> ~850k public), which flips the full position
  with an IOC whenever F_hat crosses the far touch, i.e. A1 taken to its limit.
- 23:15: grid on mm_r2_zd (copy `strategies/zd_radys_grid.py`): fq_take_k {0.1,0.2,0.3} x pred_blend {0.5,0.75}
  x zd_min_ahead {0,100}. Results below if they land before the close.

## Last simple ideas for Honza (radys, 2026-11-07 23:45, untested)
Base: strategy (9) = mm_r2_rts, 957k public. The final grid was stopped, so none of these are measured.
- `fq_k_sp2`: try 0.9 and 1.2. The jump 0.7 over (7) came from this knob, so the optimum may be higher.
- `qs_k` 0.7 (rtq grid on (7) gave +0.4%, noise level). Low priority.
- End position was -100 with penalty about 1.8k. A small end-of-day flatten could save about 1-2k.
- Turn on `ri_ms` (20) or `qa_max` (500) one at a time; both exist but are off.
- Judge each change on round2 and round2_mirror, 6/8 chunks or better.
