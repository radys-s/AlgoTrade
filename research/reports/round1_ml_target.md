# H18 loss-aligned target for the ML shrink (round1)

Code: `research/fit_target.py` (fit + IC), `strategies/exp_ml_target.py` (PARAMS `ml_target` base|A|B|C), scratch `logs/ml_target/`.
Same 23 features, winsorise, standardise, alpha 10 as the champion. Hit = a taker trades on the side within 100 ms after the 5 ms latency; move after hit = mid(th+100ms) - mid(th), th = hit time.
Candidates: A = weight 1+10*[any hit] on the plain 100 ms move (A3 = k 3 also fitted), B = move after the first hit x [hit] (0 when no hit), C = signed adverse-fill score (+1 ask hit then mid up >= 0.5 tick within 1 s, -1 bid hit then mid down >= 0.5 tick, 0 else; linear probability score).
54.6 % of events are followed by a hit; adverse bid 12.8 %, adverse ask 12.9 %.
Fit on first 75 % of time with 1.2 s purge (2.2M rows after stride 2), tested on the last 25 % (holdout).

## IC table (Pearson)

| cand | set | IC own target | IC plain 100 ms | IC plain, hit events only | R2 hit events (IC^2) | IC vs B target | IC vs C target |
|---|---|---:|---:|---:|---:|---:|---:|
| base | train | +0.3833 | +0.3833 | +0.4367 | 0.191 | +0.2995 | +0.1894 |
| base | hold | +0.3753 | +0.3753 | +0.4294 | 0.184 | +0.2955 | +0.1889 |
| A3 | hold | +0.3744 | +0.3744 | +0.4313 | 0.186 | +0.2986 | +0.1874 |
| A10 | hold | +0.3739 | +0.3739 | +0.4315 | 0.186 | +0.2992 | +0.1869 |
| B | hold | +0.3015 | +0.3693 | +0.4302 | 0.185 | +0.3015 | +0.1842 |
| C | hold | +0.1950 | +0.3572 | +0.4039 | 0.163 | +0.2773 | +0.1950 |

Each candidate wins only on its own target and by at most ~0.005 IC; the 23-feature linear space leaves no room for a different target to find different structure.

## Sweep (ledger tag r1-ml-target; thresholds matched to the champion's 19 % firing rate = |pred| 81st percentile of the full-round1 fit)

Matched thresholds: base 0.15 (0.1496), A 0.204, B 0.122, C 0.133 (the middle row of each block).

| dataset | set | pnl | train_pnl | hold_pnl | chunk_wins | chunk_min | fills | breach | inv_q997 | runtime_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round1 | base 0.15 (parity) | 490,460 | 364,169 | 126,291 | 8 | 42,720 | 46,527 | 0 | 100 | 117.6 |
| sample_v2 | base 0.15 (parity) | 254,061 | 193,133 | 60,928 | 8 | 23,165 | 11,723 | 0 | 100 | 76.0 |
| round1 | A 0.164 | 499,538 | 369,822 | 129,716 | 8 | 41,010 | 49,313 | 0 | 100 | 121.3 |
| round1 | A 0.204 | 492,656 | 365,463 | 127,193 | 8 | 43,036 | 46,393 | 0 | 100 | 117.5 |
| round1 | A 0.245 | 473,160 | 350,356 | 122,804 | 8 | 40,792 | 44,105 | 0 | 100 | 119.2 |
| sample_v2 | A 0.164 | 252,405 | 191,014 | 61,391 | 8 | 22,729 | 12,474 | 0 | 100 | 75.1 |
| sample_v2 | A 0.204 | 256,200 | 195,180 | 61,020 | 8 | 23,479 | 11,649 | 0 | 100 | 78.7 |
| sample_v2 | A 0.245 | 257,954 | 196,999 | 60,955 | 8 | 23,539 | 10,949 | 0 | 100 | 79.3 |
| round1 | B 0.097 | 499,397 | 370,478 | 128,919 | 8 | 41,529 | 49,816 | 0 | 100 | 115.8 |
| round1 | B 0.122 | 494,990 | 367,171 | 127,819 | 8 | 42,740 | 46,315 | 0 | 100 | 117.9 |
| round1 | B 0.146 | 469,724 | 347,824 | 121,900 | 8 | 40,122 | 43,784 | 0 | 100 | 118.3 |
| sample_v2 | B 0.097 | 256,572 | 194,652 | 61,920 | 8 | 23,386 | 12,366 | 0 | 100 | 75.6 |
| sample_v2 | B 0.122 | 258,239 | 196,547 | 61,692 | 8 | 23,919 | 11,620 | 0 | 100 | 76.8 |
| sample_v2 | B 0.146 | 255,267 | 194,974 | 60,293 | 8 | 23,520 | 10,820 | 0 | 100 | 77.3 |
| round1 | C 0.107 | 491,604 | 366,436 | 125,168 | 8 | 42,513 | 48,676 | 0 | 100 | 118.8 |
| round1 | C 0.133 | 482,630 | 357,717 | 124,913 | 8 | 41,798 | 46,008 | 0 | 100 | 117.7 |
| round1 | C 0.160 | 468,993 | 347,746 | 121,247 | 8 | 39,779 | 43,957 | 0 | 100 | 116.0 |
| sample_v2 | C 0.107 | 244,071 | 184,884 | 59,187 | 8 | 22,413 | 12,269 | 0 | 100 | 74.0 |
| sample_v2 | C 0.133 | 253,076 | 191,962 | 61,114 | 8 | 22,725 | 11,639 | 0 | 100 | 76.0 |
| sample_v2 | C 0.160 | 253,025 | 193,913 | 59,112 | 8 | 23,256 | 11,020 | 0 | 100 | 76.8 |

## research.compare, mm_ml (A) vs exp_ml_target B (ml_target=B, ml_thr=0.122)

```
round1 (primary)   pnl A=490,460 B=494,990 diff=+4,530   hold A=126,291 B=127,819 diff=+1,528
  chunks B-A [182.0, 20.0, 978.0, -207.0, 16.0, 684.8, 1496.2, -168.0] -> B wins 6/8   fills 46527 / 46315
sample_v2          pnl A=254,061 B=258,239 diff=+4,178   hold A=60,928 B=61,692 diff=+764
  chunks B-A [13.3, 48.6, 127.0, 776.0, 834.2, -41.9, 358.6, 1298.1] -> B wins 7/8   fills 11723 / 11620
VERDICT: ACCEPT (inv_q997 100 both, no breach)
```

## Interpretation
1. A loss-aligned target barely changes the linear forecast: hold IC vs the plain move drops 0.375 -> 0.369 (B), 0.374 (A), 0.357 (C); on hit events the fit is unchanged (R2 0.184-0.186, C 0.163). The 23 linear features carry the same signal whichever event subset is weighted.
2. Pnl effect is small but same-signed on both datasets for B and A at the matched threshold: B +4.5k round1 / +4.2k sample_v2 (+0.9 % / +1.6 %), A +2.2k / +2.1k, C -8k / -1k (the adverse-fill score is a worse ranker). Gains are 1-2 % and within chunk noise (B chunk diffs range -207..+1,496); round1 base was already the fit-on-all-data value, so this is not a fit-vs-holdout artefact but the improvement is not strongly proven.
3. B (move x hit, zeroing no-hit events) is the only cheap variant worth keeping: it ranks hit events as well as base and stops spending fit on quiet events; 6/8 and 7/8 chunks, holdout up on both datasets. Pnl rises with a lower threshold on round1 (more shrinking, 0.097: 499k) but falls on sample_v2, so 0.122 (matched rate) is the plateau middle.
4. The 100 ms MSE objective was not the bottleneck: further gains need a nonlinear model or different information (hit/queue state), not a different linear target. B is a drop-in dict swap (no runtime cost); promotion is a human decision (not committed; `research.check --no-run` PASS, parity 490,460 / 254,061 exact).
