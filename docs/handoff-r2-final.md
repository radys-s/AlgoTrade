# R2 final handoff (radys node)

Contract: FREEBUFF (this agent) owns strategies/exp_r2_radys_*.py and writes the section below.
COPILOT (auditor/compute) owns research/results/, research/server/, docs/copilot-*.md and the COPILOT section.
Server truth: mm_r2_fq2 = 740,067 (22,521 fills, 2.25M lots, penalty 0). Local R2 = 349,077 -> local x2.12 = server.
Local number is a RANKING device only. Coordinates: |I| pinned at 100 almost always, so the taker is room-blocked
98 % of the time on the growing side; the value is in (a) what the taker does when it can trade and (b) holding.

## FREEBUFF

### Step 0 — parity
- `strategies/exp_r2_radys_base.py` is a byte-identical copy of `strategies/mm_r2_fq2.py` (diff empty), so it
  reproduces 349,077 by construction. Verified independently: mm_r2_fq2 full round2 = 349,077.0 (8/8 chunks,
  chunk_min +15,752, holdout 102,347, penalty 0, 10,371 fills) and round1 = 443,510 (8/8, penalty 0).

### Measured results (all full R2 unless noted; base 349,077)
- F1 take margin ladder (asymmetric enter/exit thresholds, `take_k_grow` / `take_k_reduce` on
  `strategies/exp_r2_radys_fq3.py`, ledger 20261107-220035): best 345,205 (grow 0.25 / reduce 0.16) = **-1.1 %**,
  grow 0.3 344,164, reduce 0.12 341,522. Symmetric 0.2 (base) still best => REJECTED, the single margin is right.
- F1b hardcoded generator offsets (`fair_off_fix` 1: bids -1.22 / asks +1.20 instead of the online running mean):
  336,658 / 334,725 = **-3.6 %** => REJECTED, the online offset mean adapts better than the offline constant.
- F3 fast taker signal (`take_tau_ms` 25, second EWMA used only for the crossing decision): dropped in the p10/p30
  screen (12,668-12,815 vs 12,830 rung) => REJECTED before a full run.
- fq2 parameter plateau confirmed (27 full runs, ledger 20261107-215319): tau 70 best (tau 50 = 336.3k, tau 90 =
  312.5k), take_k 0.2 best (0.15 = 347.1k, 0.25 = 340.2k), fq_take 150 = 349,257 (+180 = noise), 200 = 349,077,
  250 identical to 200 (capacity-bound). Nothing on those axes beats base.
- Earlier base-independent results (on mm_r2_take 131,984): park/ladder REJECTED (-16k/-0.2k), asymmetric MAKER
  thresholds +2.3k, inside-spread unload REJECTED (-1.5k), fair-stub REJECTED (-2.3k).

### Wave 2 (taker axes on the fq2 base, all full or halved-full R2)
- F2 taker slippage `fq_slip` 1 / 2: **286,007 / 261,979** (-18 % / -25 %) => REJECTED. Crossing one tick deeper
  fills more but at a price the next 5 ms does not recover; the touch is where the edge is. (Ledger 20261107-220730.)
- `fair_levels` 10 / 20 / 30: **349,077 exactly identical** => the parameter is INERT on round2 (the newest insert
  is essentially always inside the first 10 levels of a side). Do not spend runs on it.
- `gate_thr` 12 + `fq_take_cooldown_ms` 10: **354,377** (+1.5 %) with chunk_min +17,967, but holdout **85,554 vs
  102,347 (-16.8k)** => overfit to the early tape, REJECTED by the holdout rule. gate 10 = 348,367, cooldown 30 =
  349,519. (Ledger 20261107-221128.)
- Earlier rejection on this base: `fq_slip` interacts with nothing useful; `take_k_grow/reduce` ladder (-1.1 %);
  hardcoded offsets (-3.6 %); fast taker tau (dropped in screen).
- Conclusion after ~60 full-rate runs: the fq2 parameter surface is a sharp local optimum at 349,077 local. Every
  single-axis move either loses or buys train pnl with holdout. Beating it needs a NEW signal or a new constraint,
  not a threshold move. Wave 3 (running): trend-conditional take margin (`take_gate_bonus`/`take_counter_penalty`),
  thin-book deep sweep (`take_slip_qty`), gate window 15 s / 45 s.

## COPILOT

- Current status: the server artifact is still missing, so the local-vs-server gap analysis is blocked.
- Verified local evidence: the best local round2 run in [20261107-222544_radys-s_r2-radys-fq5-spread.csv](research/results/20261107-222544_radys-fq5-spread.csv) is 360,791, while the uploaded base remains 349,077.
- Trust rule: until a server file appears and is compared to the same run settings, local results are directional only and cannot be used for final ranking or claiming a champion.
- Next action: keep the C2 sweep in the allowed output folder, re-check for the server artifact, and perform the exact server cross-check as soon as it lands.

### Wave 3 — the gate window is the lever (~40 full-rate runs)
- **`gate_window_us` 30 s -> 15 s = 358,764 (+2.8 %)**, 8/8 chunks, chunk_min +16,163, holdout +476, penalty 0.
  5 s = 336,595, 10 s = 342,391, 45 s = 342,337, 20 s dropped in the p30 screen => 15 s is a real optimum, not a
  monotone drift: the gate must measure the current minute, and anything shorter is noise.
- **`gate_thr` 9 at that window = 363,943 (+4.25 %)**, 8/8, chunk_min +17,021, holdout 103,164 (+817), fills 12,561.
  thr 7 = 351,518; thr 10 = 357,536 (holdout 95,872); thr 11 = 356,356 (holdout 92,988); thr 12 at 30 s = 354,377
  (holdout 85,554). Everything above 9 buys train pnl with holdout => 9 is the top of the stable band.
- **`take_k_spread` 0.1** (extra crossing margin when spread >= 2): +2,027 on top of gate 15 s at thr 8, holdout +911.
  A wide or one-sided book has a stale touch, so a cross needs a bigger edge.
- fq_take 150 + gate_max_spread 5 at thr 8: 361,140 (holdout 103,095) — roughly the same as thr 9, not additive.
- fq_take_k 0.18 / 0.22 on the stack: 360,296 / 357,395 < 0.2 => keep 0.2.
- Rejected in this wave: dropping clamped inserts (264,059, -26 %: the censored devs still carry the level
  information), take_persist 2 (353,696), gate on F-drift instead of mid drift (358,934 vs 358,764 at thr 8; at
  thr 12 it wins pnl but loses 7 k of holdout), taker slippage 1-2 ticks (-18 % / -25 %).
- Net: **`strategies/mm_r2_radys_v1.py` = mm_r2_fq2 + gate 15 s + thr 9 + take_k_spread 0.1 = 363,943 local
  (+4.25 %, x2.12 => ~772 k server pace)**, all 8 chunks positive, holdout better, penalty 0, single file, no imports.
