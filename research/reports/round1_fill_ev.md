# H20: fill-EV model with private features (negative result)

Files: `research/fill_ev.py` (dataset, walk-forward, deciles, fit), `strategies/exp_fill_ev.py` (mm_ml + `ev_thr`/`ev_mode`), data under `logs/fill_ev/`.
Fills: `logs/r1-mlml-inspect`, `logs/v2-mlml-inspect` (mm_ml full logs; pnl 490,460 / 254,061 reproduced).

## Dataset
One row per fill (attribution grouping), lot-weighted. round1: 63,428 fills, 2,197,192 lots, mo1s 0.224/lot. sample_v2: 15,679 fills, 424,077 lots, 0.590/lot.
Features at the last tape event before the fill row: pred (recomputed ridge, signed), spread, spread>=2, L1 imbalance (maker view), log lots AHEAD (market lots matched before us in the same taker order at our price = queue ahead just before the sweep), front flag, log level thickness (npz market book at the fill price), log behind, offset from touch, |I|, reduces-|I|, signed inventory, signed flow 100 ms / 1 s, signed ret 100 ms, dt since mid change, log order age (dataset only; not in the live model). Label 1 s markout in ticks (100 ms, 5 s, 30 s stored).
Caveat: `ahead` is read from the sweep that filled us, so conditional on a fill it also encodes the taker size. Live, the strategy feeds the current queue ahead, which is not conditioned on a fill: a train/live mismatch that partly explains the backtest result.

## Walk-forward (8 chunks, train on earlier chunks, purge 6 s), out-of-fold on chunks 1-7
| dataset | fills | lots | model | features | R2 (1 s, OOF) | IC lot-w | rank IC | chunks IC>0 (of 7) | min chunk IC |
|---|---|---|---|---|---|---|---|---|---|
| round1 | 63428 | 2,197,192 | ridge | pred only | 0.0785 | 0.280 | 0.273 | 7 | 0.261 |
| round1 | 63428 | 2,197,192 | ridge | market feats (no private) | 0.0938 | 0.306 | 0.307 | 7 | 0.291 |
| round1 | 63428 | 2,197,192 | ridge | full | 0.1301 | 0.361 | 0.385 | 7 | 0.373 |
| round1 | 63428 | 2,197,192 | hgb | full | 0.1293 | 0.360 | 0.381 | 7 | 0.354 |
| sample_v2 | 15679 | 424,077 | ridge | pred only | 0.0149 | 0.128 | 0.103 | 7 | 0.085 |
| sample_v2 | 15679 | 424,077 | ridge | market feats (no private) | 0.1024 | 0.327 | 0.311 | 7 | 0.280 |
| sample_v2 | 15679 | 424,077 | ridge | full | 0.3852 | 0.622 | 0.659 | 7 | 0.637 |
| sample_v2 | 15679 | 424,077 | hgb | full | 0.3800 | 0.618 | 0.661 | 7 | 0.640 |


**round1: deciles by out-of-fold full-ridge EV (chunks 1-7)**

| decile | mean EV | realised mo/lot | lots | $ at 1 s |
|---|---|---|---|---|
| 1 | -0.375 | -0.346 | 193,521 | -66,871 |
| 2 | -0.140 | -0.147 | 193,640 | -28,384 |
| 3 | -0.027 | -0.031 | 193,653 | -5,940 |
| 4 | 0.072 | 0.079 | 193,623 | 15,213 |
| 5 | 0.162 | 0.159 | 193,527 | 30,768 |
| 6 | 0.254 | 0.284 | 193,645 | 55,088 |
| 7 | 0.350 | 0.397 | 193,669 | 76,951 |
| 8 | 0.453 | 0.466 | 193,598 | 90,268 |
| 9 | 0.581 | 0.611 | 193,620 | 118,246 |
| 10 | 0.789 | 0.771 | 193,645 | 149,298 |

**round1: deciles by pred_s alone (OOF ridge on pred)**

| decile | mean EV | realised mo/lot | lots | $ at 1 s |
|---|---|---|---|---|
| 1 | -0.226 | -0.161 | 193,601 | -31,170 |
| 2 | -0.068 | -0.068 | 193,620 | -13,213 |
| 3 | 0.017 | 0.016 | 193,607 | 3,074 |
| 4 | 0.092 | 0.111 | 193,613 | 21,456 |
| 5 | 0.169 | 0.185 | 193,491 | 35,852 |
| 6 | 0.245 | 0.213 | 193,732 | 41,316 |
| 7 | 0.325 | 0.286 | 193,553 | 55,373 |
| 8 | 0.412 | 0.385 | 193,644 | 74,466 |
| 9 | 0.512 | 0.544 | 193,633 | 105,354 |
| 10 | 0.646 | 0.734 | 193,647 | 142,130 |

**round1: deciles by out-of-fold HGB EV**

| decile | mean EV | realised mo/lot | lots | $ at 1 s |
|---|---|---|---|---|
| 1 | -0.406 | -0.362 | 193,608 | -70,172 |
| 2 | -0.138 | -0.116 | 193,611 | -22,420 |
| 3 | -0.019 | -0.033 | 193,610 | -6,330 |
| 4 | 0.077 | 0.102 | 193,593 | 19,826 |
| 5 | 0.168 | 0.185 | 193,637 | 35,798 |
| 6 | 0.264 | 0.250 | 193,624 | 48,465 |
| 7 | 0.363 | 0.349 | 193,613 | 67,612 |
| 8 | 0.476 | 0.507 | 193,595 | 98,191 |
| 9 | 0.610 | 0.596 | 193,629 | 115,324 |
| 10 | 0.790 | 0.766 | 193,621 | 148,341 |

**sample_v2: deciles by out-of-fold full-ridge EV (chunks 1-7)**

| decile | mean EV | realised mo/lot | lots | $ at 1 s |
|---|---|---|---|---|
| 1 | -0.186 | -0.118 | 37,069 | -4,390 |
| 2 | 0.118 | 0.171 | 37,140 | 6,354 |
| 3 | 0.248 | 0.253 | 37,058 | 9,392 |
| 4 | 0.428 | 0.456 | 37,171 | 16,956 |
| 5 | 0.604 | 0.583 | 37,101 | 21,620 |
| 6 | 0.753 | 0.719 | 37,040 | 26,638 |
| 7 | 0.848 | 0.824 | 37,194 | 30,640 |
| 8 | 0.925 | 0.858 | 37,117 | 31,850 |
| 9 | 1.008 | 0.982 | 37,103 | 36,436 |
| 10 | 1.165 | 1.110 | 37,131 | 41,207 |

**sample_v2: deciles by pred_s alone (OOF ridge on pred)**

| decile | mean EV | realised mo/lot | lots | $ at 1 s |
|---|---|---|---|---|
| 1 | 0.457 | 0.344 | 37,084 | 12,773 |
| 2 | 0.503 | 0.473 | 37,126 | 17,548 |
| 3 | 0.535 | 0.539 | 37,033 | 19,958 |
| 4 | 0.564 | 0.641 | 37,181 | 23,822 |
| 5 | 0.591 | 0.669 | 37,112 | 24,824 |
| 6 | 0.616 | 0.585 | 37,133 | 21,720 |
| 7 | 0.641 | 0.616 | 37,102 | 22,840 |
| 8 | 0.668 | 0.660 | 37,092 | 24,474 |
| 9 | 0.702 | 0.711 | 37,126 | 26,392 |
| 10 | 0.758 | 0.602 | 37,135 | 22,352 |

**sample_v2: deciles by out-of-fold HGB EV**

| decile | mean EV | realised mo/lot | lots | $ at 1 s |
|---|---|---|---|---|
| 1 | -0.193 | -0.106 | 37,038 | -3,916 |
| 2 | 0.132 | 0.159 | 37,156 | 5,899 |
| 3 | 0.277 | 0.278 | 37,129 | 10,316 |
| 4 | 0.424 | 0.412 | 37,086 | 15,266 |
| 5 | 0.580 | 0.597 | 37,126 | 22,158 |
| 6 | 0.711 | 0.707 | 37,078 | 26,224 |
| 7 | 0.822 | 0.820 | 37,145 | 30,460 |
| 8 | 0.922 | 0.888 | 37,084 | 32,926 |
| 9 | 1.029 | 1.001 | 37,169 | 37,220 |
| 10 | 1.183 | 1.082 | 37,113 | 40,150 |

**Cross-fit** (fit on all fills of one dataset, score the other; ridge, full features)

| train -> test | R2 | IC lot-w | rank IC |
|---|---|---|---|
| round1 -> sample_v2 | -0.2605 | 0.446 | 0.497 |
| sample_v2 -> round1 | -0.1200 | 0.159 | 0.212 |

**sample_v2 fills by the round1 model's EV (cross-fit deciles)**

| decile | mean EV | realised mo/lot | lots | $ at 1 s |
|---|---|---|---|---|
| 1 | -0.458 | 0.107 | 42,353 | 4,516 |
| 2 | -0.221 | 0.236 | 42,451 | 10,025 |
| 3 | -0.092 | 0.316 | 42,414 | 13,412 |
| 4 | 0.029 | 0.479 | 42,410 | 20,324 |
| 5 | 0.143 | 0.626 | 42,381 | 26,528 |
| 6 | 0.255 | 0.721 | 42,433 | 30,594 |
| 7 | 0.360 | 0.781 | 42,408 | 33,124 |
| 8 | 0.466 | 0.823 | 42,392 | 34,884 |
| 9 | 0.583 | 0.840 | 42,405 | 35,618 |
| 10 | 0.803 | 0.972 | 42,430 | 41,223 |

## Strategy (`strategies/exp_fill_ev.py`)
`ev_thr` None = parity: `research.run` round1 prints 490460 (train 364169, hold 126291, 46527 fills), identical to mm_ml. `research.check --no-run`: PASS.
Live EV per side (ridge, 16 literal coefficients, queue ahead of our first touch order from the snapshot, thickness = market lots at the touch, off = 0). Shrink to ml_stub (5 lots) when EV < ev_thr; `ev_mode` replace | and | or with the |pred| rule.
Runtime (solo, round1): parity 121 s vs mm_ml 128 s (noise); with EV on: 162-176 s in 5-worker sweeps vs ~121-128 s solo, so about +15-25 us/event, +25-35 % (not measured solo; the sweep is under 5-way contention). The scan of the whole touch level is the larger part.

## Sweep (ledger tag r1-fill-ev, 5 workers). mm_ml reference: round1 490,460 / sample_v2 254,061.
| dataset | ev_thr | ev_mode | pnl | train_pnl | hold_pnl | chunk_wins* | chunk_min | fills | breach | inv_q997 | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| round1 | -0.5 | or | 488,065 | 362,037 | 126,028 | 8 | 42,603 | 47349 | 0 | 100 | 173 |
| round1 | -0.5 | replace | 467,223 | 347,064 | 120,159 | 8 | 40,251 | 45075 | 0 | 100 | 168 |
| round1 | -0.35 | or | 476,134 | 354,839 | 121,295 | 8 | 40,699 | 48485 | 0 | 100 | 176 |
| round1 | -0.35 | replace | 467,643 | 348,760 | 118,883 | 8 | 40,506 | 47482 | 0 | 100 | 176 |
| round1 | -0.199 | or | 437,432 | 327,010 | 110,422 | 8 | 35,925 | 49588 | 0 | 100 | 162 |
| round1 | -0.199 | replace | 437,343 | 327,385 | 109,958 | 8 | 35,943 | 49513 | 0 | 100 | 163 |
| round1 | -0.0705 | or | 378,240 | 283,828 | 94,412 | 8 | 31,650 | 50421 | 0 | 100 | 163 |
| round1 | -0.0705 | replace | 377,811 | 283,508 | 94,303 | 8 | 31,660 | 50402 | 0 | 100 | 164 |
| round1 | 0.034 | or | 313,184 | 234,887 | 78,297 | 8 | 25,760 | 50424 | 0 | 100 | 164 |
| round1 | 0.034 | replace | 313,079 | 235,054 | 78,025 | 8 | 25,794 | 50413 | 0 | 100 | 162 |
| sample_v2 | -0.5 | or | 253,189 | 192,672 | 60,517 | 8 | 23,035 | 11814 | 0 | 100 | 100 |
| sample_v2 | -0.5 | replace | 253,249 | 193,443 | 59,806 | 8 | 23,130 | 11046 | 0 | 100 | 97 |
| sample_v2 | -0.35 | or | 248,777 | 187,743 | 61,034 | 8 | 22,135 | 12094 | 0 | 100 | 104 |
| sample_v2 | -0.35 | replace | 248,179 | 188,474 | 59,705 | 8 | 22,573 | 11904 | 0 | 100 | 103 |
| sample_v2 | -0.199 | or | 233,756 | 175,725 | 58,031 | 8 | 20,285 | 12941 | 0 | 100 | 95 |
| sample_v2 | -0.199 | replace | 235,124 | 176,413 | 58,711 | 8 | 20,581 | 12822 | 0 | 100 | 96 |
| sample_v2 | -0.0705 | or | 209,465 | 157,611 | 51,854 | 8 | 18,125 | 13740 | 0 | 100 | 91 |
| sample_v2 | -0.0705 | replace | 210,128 | 157,715 | 52,413 | 8 | 18,102 | 13779 | 0 | 100 | 94 |
| sample_v2 | 0.034 | or | 182,310 | 138,826 | 43,484 | 8 | 16,160 | 14011 | 0 | 100 | 93 |
| sample_v2 | 0.034 | replace | 182,283 | 138,585 | 43,698 | 8 | 15,832 | 14055 | 0 | 100 | 91 |
*chunk_wins as printed by research.run (positive training chunks, 8/8 everywhere); the paired comparison vs mm_ml is below. EV thresholds -0.199 / -0.0705 / 0.034 are the lot-weighted 10/20/30 % quantiles of the fill-level EV; -0.35 and -0.5 were added after the first six cells all lost.

## research.compare: mm_ml (A) vs exp_fill_ev ev_thr -0.5 mode or (B, best cell)
```
A = strategies/mm_ml.py {}
B = strategies/exp_fill_ev.py {"ev_thr": -0.5, "ev_mode": "or"}

round1 (primary)
  pnl    A=     490,460  B=     488,065  diff=      -2,395
  hold   A=     126,291  B=     126,028  diff=        -263
  chunks B-A [-40.0, -117.0, -447.0, 165.0, -375.0, -988.0, 295.0, -625.3]  -> B wins 2/8
  fills  A=46527  B=47349   inv_q997 A=100 B=100
sample_v2 (other)
  pnl    A=     254,061  B=     253,189  diff=        -872
  hold   A=      60,928  B=      60,517  diff=        -411
  chunks B-A [-475.3, -135.7, 205.0, -130.0, 307.5, -34.2, -499.2, 300.9]  -> B wins 3/8
  fills  A=11723  B=11814   inv_q997 A=100 B=100

VERDICT: REJECT
  - round1: wins only 2/8 paired chunks (< 6)
  - round1: holdout not better (126,028 <= 126,291)
```

## Interpretation
1. As a model the idea is sound: private features lift out-of-fold IC of the 1 s fill markout from 0.28 (pred alone) to 0.36-0.39 on round1 (R2 0.079 -> 0.130) and from 0.13 to 0.62 on sample_v2 (R2 0.015 -> 0.385); 7/7 chunks positive, HGB adds nothing over ridge. The bottom three EV deciles hold -101k of 1 s markout on round1 (pred alone: bottom two -44k).
2. As a shrink rule it loses everywhere: PnL falls monotonically as ev_thr rises (-0.5: 488k, -0.35: 476k, -0.2: 437k, -0.07: 378k, +0.03: 313k on round1; v2 253k ... 182k). The best cell is a wash against mm_ml (-2.4k round1, -0.9k v2; 2/8 and 3/8 paired chunk wins, REJECT).
3. Why: a lot at the tail is negative at 1 s but the shrink also removes lots whose cost is mostly spread capture; mm_ml already removes the pred-driven tail, and the fill-conditional EV that includes the swept `ahead` mismatches the live feature (not conditioned on a fill). The cut that matters is probably volume lost versus the 5-lot stub, not the EV ranking.
4. Cross-fit: the round1 model keeps rank IC 0.50 on sample_v2 (but R2 < 0 from the level shift), the reverse 0.21: ranking transfers, levels do not. v2 has no loss decile for the model to isolate (decile 1 = -0.12).
5. Not tried (time): training the live EV on state sampled at every event with a hazard-weighted fill probability (removes the conditioning mismatch), and an EV-scaled partial shrink (size between 5 and 20) rather than all-or-nothing.
