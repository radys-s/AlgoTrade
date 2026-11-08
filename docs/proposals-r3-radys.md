# Round 3 proposals from the radys window (for Honza's Claude)

Written 2026-11-08 by radys (single-thinker pass, no subagents). Read this after `docs/progress.md`.

## 1. Calibration (verified)

| Strategy | Local round3_public | Server | Ratio |
|---|---|---|---|
| mm_r3_pms (s10) | 413,750 | 675,488 | 1.63 |
| mm_r3_g2 (s11) | 478,572 | 772,540 | 1.61 |

Local tracks the server at about x1.6, so **1.2M on the server = about 750k local**. We need +57 % local over s11.
R3 board at 01:49 Prague: median best 503k, 75th percentile about 800k. s11 is already around the 75th percentile.
Local verification of s11 reproduced Honza's 478,572 exactly (holdout 117,658, 8/8 chunks, breach 1,157).

## 2. What radys measured on R3 (transferable)

- The round-2 ridge is stale on R3; `pred_blend 0` beat 0.5 by about 8 % with it. With the R3-refitted ridge (rfit 19) the blend 0.7 is right again, so do not carry `pred_blend 0` over.
- On the old mm_r2_rts base, the best neighbourhood was fq_k_sp2 about 1.3, gate_thr 6-8, ri_ms 20-80, qs_k 0.45. s11 already sits next to that (1.5, 10, 0, 0.6).
- Public signals: ridge/hgb IC about 0.33/0.35 at 100 ms, imb_l1 IC +0.10. mm_edge filter "flow toward the maker over 100 ms < 0" gives +0.397 ticks per fill. See `research/reports/round3_public_*.md`.
- hgb beats ridge by only 0.02 IC, so a nonlinear model is not the lever.

## 3. Where the next +57 % can come from (ranked by expected value, using the trading skills)

PnL is roughly (fills per second) x (edge per fill), and inventory is capped at 100. So the two levers are **edge per fill** and **turnover of the 100 lots**.

1. **Spread-split models** (signal-classification, regime-detection). Fit one ridge for spread == 1 and one for spread >= 2 (and maybe one for thin touch). The feature-to-return map differs sharply by spread state; a single ridge averages them. Cheap: `research.refit_ml` with a mask, bake as two PRED_MODELS entries, pick by `spread` in `_pred_update`. Expected IC gain +0.03 to +0.06.
2. **Volatility-normalised thresholds** (volatility-modeling). `fq_take_k`, `fq_k_sp2`, `gate_thr` are fixed in ticks. In calm minutes the same tick threshold is too loose, in fast minutes too tight. Scale each by EWMA of |mid change| over 5 s divided by its tape median, clipped to [0.7, 1.5]. One new param `vol_adapt`, flag-off parity required.
3. **Predictor drives the passive side too** (market-microstructure). s11 uses `pred_mode 1` (prediction only in the IOC test). Test `pred_mode 0` and a half-way option: pull or step back only the quote the prediction says will be picked off (`pred < -thr` on the bid side). This is classic adverse-selection avoidance; queue priority is lost only when the quote is about to be hit by informed flow. Needs a grid on `pred_mode` and a pull threshold.
4. **Exit by prediction, not by inventory** (exit-strategies). When inventory is long and `pred` turns negative beyond a threshold, IOC out at the touch even if inventory is small. This frees the cap for the next entry (turnover). Check whether `fq_take` already does it for the closing direction; if not, add `exit_k`.
5. **Time-decay of a stale quote.** A resting quote older than X ms whose edge was set from a prediction that has since decayed is cancelled and re-placed (ri_ms is close to this, but tied to the touch, not to the prediction age). Test ri_ms 20-80 on s11; wave 2 on the old base favoured it.
6. **Do not spend effort on the penalty band** (risk-management). Breach cost is 1.2k local, 2.2k server, 0.3 % of PnL. Even a perfect guard buys under 0.5 %.

## 4. Honest limits

- No single parameter change will give +57 %. Items 1 and 3 are the only ones I would expect to move IC or adverse selection materially; 2, 4 and 5 are 1 to 5 % each.
- Anything tuned on public must hold on `round2`, `round1` and the mirror dataset before submitting (walk-forward-validation: 8/8 chunk wins and holdout positive are the minimum).
- The server has about x1.6 local, not x2, so part of the server tape's edge is not reproduced locally (higher penalty, different queue behaviour). Expect gains to transfer at about 0.8 of the local gain.

## 5. Structural grid on s11 (radys, running)

Grid: layer_size {0,40} x tr_rest {0,1} x pred_scale {1.0,1.4} x park_size {0,20} x ri_ms {0,40}, halving 0.1,0.3, ledger tag `r3-s11-struct`. 
Result (full runs of the survivors, round3_public): base 478,572; ri_ms 40 479,794; layer_size 40 479,821; both 480,026 (holdout 120,388, 8/8). All within +0.3 %, i.e. noise. `tr_rest 1`, `pred_scale 1.4` and `park_size 20` were dropped at the 10 % rung (about 35 % to 0 % below base at 30 % rung: 65-68k vs about 103k). **Conclusion: s11 is on a plateau for quote-side mechanics; do not spend more runs there. The gain has to come from the predictor (items 1, 3 above) or a new information source.**

## 6. Tasks handed to Copilot (light compute)

See the "Copilot" section at the end of this file once filled; the prompt was given to Tomas in chat.
