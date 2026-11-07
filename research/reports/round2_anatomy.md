# Round 2 anatomy: engine + data (home, 2026-11-07 20:00-20:20)

Scripts: `research/diag/r2_anatomy.py` (raw output `research/reports/r2_anatomy.txt`), `research/diag/queue_age.py`,
`research/diag/fill_decomp.py`, front-of-queue oracle `research/diag/sim_frontqueue.py` (logs in `logs/r2/`).

## 1. Same generator as round 1, different placement parameters

| | round1 | round2 |
|---|---:|---:|
| limit inserts / s / side | 89.9 | 89.2 |
| IOC / s / side, size median | 4.0, 61 | 4.45, 61 |
| placement at touch / 1 / 2 / 3 / 4+ behind | 56 / 37 / 5.2 / 0.6 / 0.1 % | 52 / 36 / 7.7 / 2.0 / 1.4 % |
| IOC limit 2+ ticks through the touch | 1.2 % | 16 % (0.9 % >= 10 ticks) |
| touch qty median | 904 | 694 |
| spread 1 tick | 93 % | 85 % |
| robust 1 s mid std (spread <= 3 only) | 0.92 | 3.90 |

Arrival rates, sizes and lifetimes are unchanged; orders are placed deeper and IOCs reach deeper, so the book
behind the touch is thin and full of gaps.

## 2. The book has holes: 34 jumps carry 63 % of the event-level mid variance

When the touch empties, the next level is often 10-80 ticks away (a stale deep order); the mid jumps and snaps
back within milliseconds when the next insert arrives. 1,207 one-sided events, 257 events with spread >= 10.
Any mid-based signal (trend gates, volatility, ML features) must ignore states with a wide spread:
the gate drift sampled only while spread <= 3 is +2.3k better on round2 (84,025 -> 86,333).

## 3. Price process: random walk with weak minute-scale momentum

Robust variance ratios vs 1 s: 10 s 0.98, 30 s 1.16, 60 s 1.20, 120 s 1.40, 300 s 1.51 (SE 0.08-0.26); round1
0.99-1.14 and < 1 at 300 s. Lag-1 autocorrelation of 30 s / 60 s returns is not stable across chunks. Flow,
inside-insert and placement imbalances do not predict the next 10-120 s consistently (signs flip across 5
chunks). There is no hidden fair value to read from the inserts; the net drift of the public tape (-310
ticks) is about 1 sd of a random walk.

## 4. Queue age: an order 1 s old is at the front

`queue_age.py`: when a touch retreats onto an existing level, an order parked at that level A seconds earlier has
0 lots ahead with probability 16 % (0.1 s), 39 % (0.25 s), 62 % (0.5 s), 84 % (1 s), 97 % (2 s) on round2
(round1: 4 / 14 / 35 / 68 / 95 %). Market orders live ~0.3 s, so queues turn over in about a second.

## 5. What queue priority and the gate are worth

Front-of-queue oracle (our orders inserted at the front of their level): trend gate 84,025 -> 231,667,
insurance 20,151 -> 69,443. Queue priority is again the biggest lever, as in round 1 (GammaStrike, round 1
winner, made 426k fills on the private tape, 12x the rest of the top 5).

Standing ladder / parking on top of the gate (`exp_r2_ladder.py`, ledger r2-ladder-s1..s3): parking HURTS
(-20k: stubs are run over), a 15-lot ladder 1-3 ticks behind adds only +1k.

## 6. Where the gate maker's 86k comes from (`fill_decomp.py`)

9,263 fills, 518k lots, |I| >= 90 for 83 % of the time (it is a passive trend follower: the against-trend side is
pulled, so the position sits at +-100 in the direction of the 30 s drift).

| per lot | ticks |
|---|---:|
| spread captured vs robust mid at the fill | +0.368 |
| markout 100 ms | +0.251 |
| markout 1 s | +0.053 |
| markout 10 s | +0.079 |
| markout 60 s / 300 s | +0.149 / +0.149 |

Immediate adverse selection (-0.32/lot within 1 s = -163k) eats almost all of the spread; the minute-scale trend
then gives back +0.10/lot (+50k). Levers, in order: (a) cut the first-second adverse selection (flow pull,
refitted ML shrink, queue position), (b) a better trend estimate (robust drift, threshold, hysteresis),
(c) queue position.
