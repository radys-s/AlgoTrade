# Round 4 plan (home brain, written 2026-11-08 04:55 during round 3)

Round 4: Sun 07:00-12:00, weight 1.2, NO live leaderboard. Pick the final by local robustness: both public tapes
(tape + mirror), holdout (last 25 %), round2 / round3 as regime checks, never by in-sample full-tape pnl alone.

## What R3 taught us (details in docs/progress.md, 01:00-05:00 entries)

| step | R3 leaderboard | lesson |
|---|---:|---|
| R2 final as is (pms) | 675,488 | the R2 stack transfers, but every learned table must be refitted |
| ridge refit + passive retune (g2) | 772,540 | refit on the new tape first (+22 %), then the passive filters (rt_ms, sp2, gate, qs_k) |
| MLP taker (mlp2) | 864,770 | 22-feature MLP on the 200 ms target, round3 + mirror first 75 % |
| MLP in quotes + slow pull (ms) | 925,809 | 500 ms quote MLP blended 50/50 with fair_sig; learned 5 s direction pull |
| streaming GRU taker (gru) | 976,148 | 16-hidden GRU over 13 LITE event tokens + the 22 features; x0.8 transfer of the local gain |

Dead ends in R3 (do not repeat): exact fair-price reading (generator agent: present F loses money, the GRU already has
it), price-aware sweeps, over-cap bursts, passive veto / quote blend / size by the GRU, zombie undo, GRU ensembles and
bigger GRUs (offline corr up, pnl flat), averaged / longer targets, order-flow and order-age tokens, hysteresis,
target-inventory policies, two-sided MM, queue tricks that keep stale prices (fq_tol, park, pre-place, layer, inside).

## The one big lever left: queue priority

Front-of-queue simulation on mm_r3_gru2 (research/diag/frontq_r3.py --mode front): round3 +196k (holdout +48.6k),
mirror +215k, round2 +256k, almost all from passive fills (small market takers that the replay sends to the orders
queued ahead of us). Our quotes are cancelled before they climb the queue (24k cancels for 28k inserts). R3-end agents
(exp_r3_res.py, exp_r3_qcap.py, exp_r3_flive.py) test reservations, pull exemptions near the front, longer-lived quotes:
read their reports first. If one of them works, it is the first thing to port to R4.

## R4 timeline

| when | step |
|---|---|
| 07:00-07:15 | data in `data/`; mirror (`-m research.diag.mirror`); features; run mm_r3_gru2 and mm_r3_ms on the new tape + mirror + round3 (regime check) |
| 07:15 | insurance pick: the better of mm_r3_ms / mm_r3_gru2 / mm_r2_final on tape + mirror holdout; humans upload |
| 07:15-08:30 | refit in this order, each judged on holdout: ridge (rfit), taker MLP, 500 ms quote MLP, slow ridge, GRU (seq_r3.py T128 h16 LITE, 3-6 epochs, keep the best of 2-3 seeds by tape + mirror holdout) |
| 08:30-10:00 | passive filter retune (rt_ms, fq_k_sp2 / sp3, gate_thr, qs_k) and the queue-priority mechanism if one worked in R3 |
| 10:00-11:30 | robustness: chunks, holdout, round3 cross-check; bake, research.check (< 7 min per 8.8M rows), parity |
| 11:30 | final upload; R4 has no leaderboard, so the final is the most robust local candidate |

## Insurance predictor for a new regime
strategies/exp_r3_gx.py carries a GRU trained on round3 + round2 (+ mirrors), research/diag/mlp2x_r3.py: on round3 it
loses 3-5k holdout vs the round3-only GRU, but on round2 it gains +5..7k holdout (+14 % taker edge). Before the R4 refit
is done, compare it with mm_r3_gru2 on the new tape: the multi-regime GRU should transfer better to an unseen regime.

## Tooling to have ready
- research/diag/seq_r3.py (+ seqt_r3.py `--npz` swap), mlp2_r3.py feature builder, mlpq_r3.py, slow_r3.py, pred_rfit_r3.py:
  all hard-code round3 / round3_mirror names; copy to *_r4.py with the new names at 07:05.
- Swap new GRU weights into a strategy: replace the `SEQ_LITE = ` .. `SEQ_HEAD = [ ... ]` block with `seq_r3.py export` output.
- research/diag/oracle_pred_r3.py (taker / quote oracles) and research/diag/frontq_r3.py (queue ceiling) take any csv.
