"""Offline: pred_r2's lean ridge (H 150 ms, target z) plus a few nonlinear taker-oriented features.

    .venv/Scripts/python.exe -X utf8 -m research.diag.pred_r2b
New features (computed exactly as exp_r2_predb._pred_flow/_pred_update do online):
  sw150      sign(flow of this update) if |flow| >= 150 lots else 0 (market trades only)
  sw50       sign(signed market flow over the last 50 ms) if |it| >= 300 else 0
  fs70_x_lqs fs70 * log1p(market qty at the touch the signal points to: ask if fs70 > 0, bid if < 0)
  fs70_c1    fs70 clipped to [-1, 1]
  fs70_c3    fs70 clipped to [-3, 3]
Fits on the first 75 % of round2 time, reports holdout corr of the old lean set vs lean + new, the tail edge,
and prints the folded raw weights (c0, w in feature order) for the strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.diag.pred_r2 import CSV, LEAN, NPZ, fair_multi
from research.features import mid_ticks
from research.signals import LATENCY_US, build_matrix, fit_ridge, value_before

TAIL = 0.0
NEW = ['sw150', 'sw50', 'fs70_x_lqs', 'fs70_c1', 'fs70_c3']
H = 150


def main() -> None:
    global TAIL
    import sys
    TAIL = float(sys.argv[sys.argv.index('--tail') + 1]) if '--tail' in sys.argv else 0.0
    z = np.load(NPZ)
    d = {k: z[k] for k in z.files}
    ts = d['ts_us'].astype(np.int64)
    n = len(ts)
    X0, names0 = build_matrix(d)
    cols = {nm: X0[:, i].astype(np.float64) for i, nm in enumerate(names0)}
    del X0
    cols.update(fair_multi(CSV, d))
    b = d['bid_px'].astype(float); a = d['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    rm = np.where(ok, (b + a) / 2, np.nan)
    rm = pd.Series(rm).ffill().bfill().values
    mid_ff = mid_ticks(d, ffill=True)
    qb = d['bid_qty'][:, 0].astype(float); qa = d['ask_qty'][:, 0].astype(float)
    cols['lq_b'] = np.log1p(qb)
    cols['lq_a'] = np.log1p(qa)
    tr = d['trade_cnt'] > 0
    last_tr = np.maximum.accumulate(np.where(tr, np.arange(n), -1))
    cols['dt_tr_log'] = np.where(last_tr >= 0, np.log1p((ts - ts[np.maximum(last_tr, 0)]) / 1000), 10.0)
    tv = d['trade_vol'].astype(float)
    ltv = np.where(last_tr >= 0, tv[np.maximum(last_tr, 0)], 0.0)
    cols['ltr_sign'] = np.sign(ltv)
    cols['ltr_lsz'] = np.sign(ltv) * np.log1p(np.abs(ltv))
    cols['hole'] = mid_ff - rm
    f70 = cols['fs70']
    cols['fs70_x_spread'] = f70 * cols['spread']
    cols['sgnflow100_x_absfs'] = np.sign(cols['flow_ms100']) * np.abs(f70)
    cols['fs70_cube'] = f70 * np.abs(f70)
    cols['fs70_x_lw'] = f70 * np.log1p(cols['fw70'])
    # --- new
    cols['sw150'] = np.where(np.abs(tv) >= 150, np.sign(tv), 0.0)
    cs = np.cumsum(tv)
    j = np.searchsorted(ts, ts - 50000, side='right')  # first row with ts > now - 50 ms
    s50 = cs - np.where(j > 0, cs[np.maximum(j - 1, 0)], 0.0)
    cols['sw50'] = np.where(np.abs(s50) >= 300, np.sign(s50), 0.0)
    cols['fs70_x_lqs'] = np.where(f70 > 0, f70 * np.log1p(qa), np.where(f70 < 0, f70 * np.log1p(qb), 0.0))
    cols['fs70_c1'] = np.clip(f70, -1, 1)
    cols['fs70_c3'] = np.clip(f70, -3, 3)
    print('fs70 |q| 50/90/99/99.9:', np.round(np.quantile(np.abs(f70), [0.5, 0.9, 0.99, 0.999]), 3),
          ' sw150 rate', round(float((cols['sw150'] != 0).mean()), 5), ' sw50 rate', round(float((cols['sw50'] != 0).mean()), 5),
          flush=True)
    t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
    e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
    tgt = e - mid_ff
    tgt[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
    allf = LEAN + NEW
    valid = np.isfinite(tgt) & np.isfinite(mid_ff)
    for nm in allf:
        valid &= np.isfinite(cols[nm])
    tri = np.where(valid & (ts + LATENCY_US + H * 1000 <= t_end))[0][::3]
    if TAIL > 0:  # fit only where the taker can fire: |fs70| above its TAIL quantile (training rows)
        af = np.abs(f70[tri])
        tri = tri[af >= np.quantile(af, TAIL)]
        print('tail fit: rows', len(tri), flush=True)
    hoi = np.where(valid & (ts > t_end))[0]
    yh = tgt[hoi]
    spr = cols['spread'][hoi]

    def fit(feats):
        Xtr = np.column_stack([cols[f][tri] for f in feats]).astype(np.float32)
        m = fit_ridge(Xtr, tgt[tri], feats, alpha=10.0, clip_q=0.0)
        w = m.coef / m.scale
        c0 = m.intercept - float(np.sum(m.coef * m.mean / m.scale))
        p = c0 + np.column_stack([cols[f][hoi] for f in feats]) @ w  # raw (unclipped), as online
        return c0, w, p

    fsh = cols['fair_sig'][hoi] if 'fair_sig' in cols else f70[hoi]
    variants = (('old lean', LEAN), ('lean+new', allf), ('lean+sw+lqs', LEAN + ['sw150', 'sw50', 'fs70_x_lqs']))
    for label, feats in variants + tuple((f'lean+{f}', LEAN + [f]) for f in NEW):
        c0, w, p = fit(feats)
        c = np.corrcoef(p, yh)[0, 1]
        out = f'{label:16s} holdout corr {c:.4f}'
        for bl in (0.5, 0.75):
            ft = bl * p + (1 - bl) * fsh
            # the taker fires when floor(mid + ft - 0.2) >= ask, roughly |ft| >= spread/2 + 0.2
            hit = np.abs(ft) >= spr / 2 + 0.2
            out += f' | b{bl}: n {hit.mean():.5f} edge {(np.sign(ft[hit]) * yh[hit]).mean() - spr[hit].mean() / 2:+.4f}'
        print(out, flush=True)
        if label in ('old lean', 'lean+new', 'lean+fs70_c1', 'lean+sw+lqs', 'lean+fs70_x_lqs'):
            full = [w[feats.index(f)] if f in feats else 0.0 for f in allf]  # 27 slots, PRED_NAMES + NEW order
            print(f'PRED {label} c0', round(c0, 7))
            print(f'PRED {label} w', ', '.join(f'{x:.7g}' for x in full))


if __name__ == '__main__':
    main()
