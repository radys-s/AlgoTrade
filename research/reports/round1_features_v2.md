# Features v2: candidate microstructure features vs the 23-feature ridge

Code: `research/features_v2.py` (candidates derived from the existing npz columns, no replay; `eval` runs the table).
Logs: `logs/features_v2/{eval,nl,rec}.log`, scripts `nl.py`, `rec.py`. Walk-forward = signals.py (8 chunks, 7 folds,
expanding, purged), ridge alpha 10, stride so <=3M rows (rec/nl: ~1.1M rows). e10/e1 = mean label (ticks) in top 10 % / 1 % |pred|.

## 1. Single candidates, ridge, IC mean (7/7 folds positive in EVERY row below)
| config | r1 h100 | r1 h1000 | s2 h100 | s2 h1000 |
|---|---|---|---|---|
| baseline23 | .3802 | .3973 | .3629 | .4077 |
| +dep_net_100 | .3803 | .3978 | .3744 | .4265 |
| +dep_net_500 | .3804 | .3973 | .3663 | .4157 |
| +cxl_net_200 | .3806 | .3978 | .3767 | .4298 |
| +ins_net_200 | .3802 | .3985 | .3823 | .4651 |
| +cnt_tot_log | .3802 | .3972 | .3628 | .4076 |
| +avgsz_imb | **.3845** | .3974 | .3690 | .4082 |
| +imb_l2 | .3803 | .3974 | .3644 | .4243 |
| +nmid_1s | .3802 | .3971 | .3628 | .4075 |
| +last_ioc_slog | .3817 | .3978 | .3640 | .4079 |
| +last_tr_share | .3802 | .3993 | .3633 | .4100 |
| +spread_dur_log | .3801 | .3973 | .3629 | .4077 |
| +flow_rel_1s | .3802 | .3984 | .3631 | .4076 |
| **+ALL 12** | .3864 | .4021 | .3970 | .4831 |
| HGB baseline23 | **.4515** | .4083 | .3844 | .4191 |
| HGB +ALL | .4546 | .4136 | .4127 | .4638 |

Top-1 % / top-10 % edge (ticks), baseline -> +ALL (ridge): r1 h100 e10 .2985->.2940, e1 .4482->.4553;
r1 h1000 e10 .769->.772, e1 .953->1.029; s2 h100 e10 .147->.159, e1 .241->.248; s2 h1000 e10 .422->.485, e1 .604->.744.
(per-candidate e10/e1 in eval.log.)

Cross-market (ridge fitted on all round1, tested on sample_v2): h100 baseline IC .2776 vs +ALL .2668 (worse);
h1000 .2148 vs .2350. i.e. the new flow/cancel features are partly market-specific (strong on the synthetic
sample, nearly nil on round1).

## 2. The big finding: nonlinearity, not new information
On round1 the 12 candidates add only +0.006 IC at 100 ms (all of them together). But HGB on the SAME 23 features
gets .4515 vs ridge .3802 (+0.07). Probing which transform ridge misses (nl.log, round1 h100, 1.1M rows):
base .3801; +imb1^3 **.3932** (+0.013); +imb1*dt_mid .3812; +slog(flow_ev100) .3810; products with spread/depth/one_tick ~0.
Cubing the book imbalances (ridge sees the extremes of imbalance as much stronger signal than the middle):

| config (rec.log) | r1 h100 | r1 h1000 | s2 h100 | s2 h1000 | cross h100 | cross h1000 |
|---|---|---|---|---|---|---|
| baseline | .3801 | .3974 | .3628 | .4078 | .2776 | .2147 |
| +cubes(imb1,imb2,micro,cnt_imb1) | **.4104** | .3991 | .3789 | .4110 | **.2927** | .2233 |
| +cubes +top5 cands | .4123 | .4016 | .4053 | .4825 | .2856 | .2388 |
| +cubes +top8 cands | .4124 | .4048 | .4061 | .4849 | .2858 | .2443 |
| +top5 cands only | .3864 | .3998 | .3962 | .4817 | .2665 | .2302 |
All 7/7. Edges (round1 h100): e10 .299->.324, e1 .449->.543 with cubes (the quantity that matters for "this side is about to be run over").
Cubes help cross-market too (.2776->.2927), so they are robust; the sample-only gains of the candidates are not.

## 3. Winners (ranked by robust gain), formulas for OnlineFeatures (all O(1))
1. **Cubes of existing features: imb1^3, imb2^3, micro^3 (micro = spread*imb1/2), cnt_imb1^3.** +0.030 IC r1 h100,
   +0.015 cross-market, +0.016 s2. Cost: 4 multiplies (~0.3 us). Just append `x*x*x` columns (same winsorize/standardize in the ridge).
2. **avgsz_imb** = (b1/nb - a1/na)/(b1/nb + a1/na) with b1,a1 = lots and nb,na = order counts at the touch (0 if a count is 0 or sum 0).
   +0.0043 r1 h100, +0.006 s2 h100. Cost ~0.5 us (from quantities OnlineFeatures already has).
3. **ins_net_200** = lots inserted by LIMIT rows at/inside the best bid minus lots inserted at/inside the best ask, window (t-200 ms, t].
   Per event: if row is a limit insert with side +1 and px >= previous best bid: +qty; side -1 and px <= previous best ask: -qty
   (previous = state before this row; ignore when that side was empty). Deque of (ts, signed qty) + running sum. Needs the tape row (op,
   side, qty, price) which the strategy only sees indirectly (book diff); in the snapshot callback it must be inferred by diffing the touch
   quantity/count - not free. Gain: +0.0 r1, +0.019 s2 h100, +0.058 s2 h1000.
4. **cxl_net_200** = lots cancelled at the ask touch minus at the bid touch, (t-200 ms, t] (cancel row with px == previous best of that side;
   qty = removed lots). Same inference caveat. +0.014 s2 h100, +0.022 s2 h1000, ~0 on round1. (dep_net_100 below is the cheap proxy.)
5. **dep_net_100** = lots removed from the ask touch minus from the bid touch in (t-100 ms, t]; removal = max(prev touch lots - touch lots, 0)
   when the best price is unchanged, or the whole previous touch when the best moved away (bid fell / ask rose or emptied). Computable purely from
   the snapshot diff; bundles trades and cancels. +0.011 s2 h100, +0.019 s2 h1000, ~0 r1.
Honorable: last_ioc_slog (side*log1p(volume of the latest IOC)) +0.0015 r1, last_tr_share +0.002 r1 h1000, flow_rel_1s (+1s signed
flow / (touch lots+1)) raises e1 at h1000 on r1 (.953->.996). Useless: cnt_tot_log, nmid_1s, spread_dur_log, imb_l2 (alone), dep_net_500.
Not implemented: age of oldest touch order (needs per-order timestamps, not O(1) from aggregates).

## 4. Cost estimate and recommendation
Per event: cubes ~0.3 us, avgsz_imb ~0.5 us, each deque window (ins/cxl/dep) ~1.5-2 us (append, pop-left, sum) => whole top-5 bundle ~5-7 us
on top of OnlineFeatures' ~10-25 us. Cubes alone are essentially free.
Recommendation for the champion (round1 market): add the four cubes first (+0.03 IC at 100 ms, edge e1 +21 %, robust cross-market),
then avgsz_imb; the dep/ins/cxl windows are worth it only if the final tape resembles sample_v2 (they were nil on round1). If model size
is no issue, HGB-like nonlinearity gives .4515 at 100 ms on round1 (another +0.04 over cubes): a small piecewise/2-layer model on the
same 23+4 inputs is the next lever (note HGB is not exportable cheaply; an MLP via train_mlp.py is).
Caveats: labels overlap, stride 1-2 rows, IC not PnL; research.run / compare on mm_ml is the real test.
