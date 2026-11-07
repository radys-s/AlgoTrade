# Round 3 playbook: break-it agent swarm (home brain, 2026-11-08 00:58)

Plan: `C:\Users\honza\.claude\plans\serene-mixing-alpaca.md` (approved). Round 3 runs Sun 01:00-06:00 (weight 1.0),
round 4 07:00-12:00 (weight 1.2, no live leaderboard). We are 13th of 61 after R1 + R2; delfini made 1,875,116 in R2
(2x us, near our own perfect-predictor ceiling), so R3 is about finding a structural break, not tuning.

Agents: `model: opus`, `effort: medium`, background. Brain = the home session (coordinates, integrates, bakes files,
hands uploads to the humans). Fill `<DEADLINE>` and the data file names at launch.

## Common header (paste at the top of every brief)

```
Team p<0.05, Qminers Algo Trade Arena, round 3. HARD DEADLINE: final report by <DEADLINE> local time (check `date`;
stop 5 min before and report what you have). Work in C:\Users\honza\QuantHack, branch `home`. Do NOT commit, push,
switch branches, edit existing files, submit anything or log in anywhere; create only the files named in this brief.
Python: `.venv/Scripts/python.exe -X utf8` (Bash tool, Git Bash). `research.run --workers` at most 5 (shared 20-thread
CPU). `-m research.check <file>` takes the file as a positional argument and must PASS for any strategy you recommend.
Data: data/round3.csv (public R3 tape), data/round3_mirror.csv (price' = 20000 - price, side swapped: a symmetry check,
not independent data), prior round: data/round2.csv. Features: `-m research.features <csv>` -> data/features/<stem>.npz.
Base strategy: strategies/exp_r3_base.py (= our R2 final mm_r2_final, leaderboard 974,031 on R2): hidden-fair signal
from new limit-order prices (fair_sig, EWMA of insert price - side offset - mid), an aggressive IOC taker when
F_hat = mid + signal is past the far touch (fq_take_k), passive touch quotes only on the side F favours (fq_k, 200-lot
flip quotes, inventory sits at +-100), zero-delay flip (zd_take / zd_quote), a 30 s trend gate, and passive filters
(rt_ms: no quote 20 ms after our touch retreats; fq_k_sp2: stricter at spread >= 2; hk_gap / hk_k: stricter over a
>= 3-tick hole behind the touch; qs_k: 100-lot quote on a weak lean). Its ridge PRED_MODELS were fitted on round 2.
Simulator facts: replayed non-reactive tape; our orders/cancels apply 5 ms later (before the first row >= t+5 ms);
price-time FIFO; our IOCs skip our own resting orders; |I| > 100 costs 10,000 per second; an order is rejected if
|I + resting same side + new| > 1000; end markout liquidates I into resting market orders. Snapshots list individual
orders (id, price, qty, owner) within 20 ticks of each side's best; trades carry taker/maker order ids; insert OrderIds
are strictly consecutive across limit and IOC rows, so id gaps count the inserts we never saw.
Acceptance: prove parity with the base first (flag off = identical pnl); a change counts only if it gains on BOTH
round3 and round3_mirror, its neighbours agree, and holdout pnl is not worse. Offline correlation is NOT evidence: in
R2 every higher-IC signal lowered pnl; judge the taker at its threshold crossings (trigger count x edge per trigger).
Pulling a resting quote loses its queue place (shrink with modify_order instead). New mechanics go behind a PARAMS flag
that is off by default. Never hard-code anything from the public tape (no timestamp lookups).
Report: a short table (config, round3, mirror, delta, holdout), a one-line verdict, the file paths.
```

## Wave 1 briefs (45-min box)

1. **Leak scanner.** Files: `research/diag/leak_scan.py`, `strategies/exp_r3_idgap.py`. On round3: rate of unseen
   inserts (id gaps not explained by snapshot ids or trade taker ids), split into non-trading IOCs vs limit orders
   placed >= 20 ticks away; their rate in the 10 / 20 / 40 ms after big hidden-F jumps vs random windows (R2: 0.35/s
   overall, 1.13 per 20 ms after jumps vs 0.004 random); how many ms after the jump the book is still intact. Then add
   `idg_on` to the base: track the highest id seen (snapshot + trades); on >= idg_n unseen inserts within idg_ms plus a
   side signature (new inserts on one side landing at the clamp next to the far touch), flip the position that way at
   once (IOC). Grid idg_n 1/2, idg_ms 10/20. Kill if < +2 % per tape.
2. **Generator card.** File: `research/diag/gen_r3.py`. Measure on round3 and diff vs round2: insert / IOC / cancel
   rates, sizes, lifetimes, placement relative to touch and to the hidden fair price (D distribution, clamp share), IOC
   limit model, F process (step sizes, jumps, drift, mean reversion), spread / hole statistics. Output a one-page card
   (markdown in the report) that agents 3, 4 and 8 can build on. No strategy work.
3. **Ceilings.** Files: copies with a csv argument (`research/diag/oracle_fair_r3.py` etc.). On exp_r3_base: replace
   the signal by the true future robust-mid move at 50 / 150 / 500 / 1000 ms; centered fair estimate (+-50 / 100 ms);
   front-of-queue (`research/diag/sim_frontqueue.py --mode front`). Output: which lever is worth the most on R3.
4. **Exact-likelihood fair filter.** File: `strategies/exp_r3_filt.py`. A grid filter over F (bids at floor(F - D),
   asks at ceil(F + D), clamp at the far touch; unseen inserts as evidence) replacing or blending with fair_sig in the
   taker test. Judge at threshold crossings, then pnl on both tapes. Kill if it does not rise on both.
5. **Insurance tuner.** Files: `research/diag/pred_r3.py` (copy of pred_r2.py with --csv), `strategies/exp_r3_tune.py`.
   Refit the ridge on round3 (first 75 %, `--export 150 --target z`) as a new PRED_MODELS entry; then grid one family
   at a time with `research.run --halving 0.3`: fq_take_k, fair_tau_ms, fq_k_sp2, rt_ms, hk_gap / hk_k, qs_k, gate_thr.
   Deliver an upload candidate by the deadline (both tapes up, check PASS).
6. **Engine artefacts.** File: `strategies/exp_r3_eng.py`. Look for a structural pnl source in the mechanics: zombie /
   missing liquidity (does our resting order ahead of a market order leave a stale market order we can later trade
   against?), short over-cap bursts vs the 10,000/s penalty, the 1000 reject rule (resting qty at activation), the end
   markout, self-trade skip, modify semantics. Quantify each in the simulator; report anything worth >= +2 %.
7. **Big-move capture.** File: `strategies/exp_r3_jump.py`. Only with a detector that fires before the book collapses
   (use agent 1's leak signal or clamp signatures): hold beyond 100 (up to 1000) for 50-200 ms into the still-stale
   side; measure penalty vs gain. In R2 the jump inserts were 22-109 ticks away (invisible) and the book collapsed in
   20-50 ms; a detector on visible 15-20-tick inserts lost money.
8. **Decision-point learner.** File: `strategies/exp_r3_learn.py`. Log the base taker's threshold crossings with the
   outcome (robust mid at +150 ms / +1 s vs entry); fit a tiny model on online features (signal components, spread,
   queues, unseen-insert counts, time since last insert) on the first 75 %; use it to veto or force takes. Evaluate as
   trigger count x edge, then pnl on both tapes. Kill if holdout pnl does not rise.

## Brain checklist
- 01:00 data in `data/`; `-m research.round data/<file> --publish --prior data/round2.csv` (its insurance verdict checks
  the wrong champion, ignore it); `-m research.diag.mirror data/round3.csv`; features for both tapes.
- Insurance: exp_r3_base with pred_on 0 / 1 on round3 + mirror + round2, `research.check`, hand the file to upload.
- Launch the 8 briefs (opus, effort medium, background) with the deadline filled in.
- Triage 02:00: rank by strategy pnl delta on both tapes + holdout; wave 2 on the top 2 leads; integrate into
  `strategies/mm_r3_champ.py`; uploads ~02:30 and ~03:15; log in `docs/progress.md`; push to `home`.
