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
