"""Ridge refit variants for exp_r3_rfit (lean 22 features, alpha 10, train rows strided by 3).
    .venv/Scripts/python.exe -X utf8 -m research.diag.pred_rfit_r3 > out.txt
Variants: sets of (dataset, time fraction used) + target H + alpha. Prints raw folded (c0, weights) per variant."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals import LATENCY_US, build_matrix, fit_ridge, value_before
from research.features import mid_ticks
from research.diag import pred_r3


def cols_for(name):
    z = np.load(f'data/features/{name}.npz'); d = {k: z[k] for k in z.files}
    ts = d['ts_us'].astype(np.int64); n = len(ts)
    X0, names0 = build_matrix(d)
    cols = {nm: X0[:, i].astype(np.float64) for i, nm in enumerate(names0)}
    del X0
    cols.update(pred_r3.fair_multi(f'data/{name}.csv', d))
    b = d['bid_px'].astype(float); a = d['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    rm = pd.Series(np.where(ok, (b + a) / 2, np.nan)).ffill().bfill().values
    mid_ff = mid_ticks(d, ffill=True)
    cols['lq_b'] = np.log1p(d['bid_qty'][:, 0].astype(float)); cols['lq_a'] = np.log1p(d['ask_qty'][:, 0].astype(float))
    tr = d['trade_cnt'] > 0
    last_tr = np.maximum.accumulate(np.where(tr, np.arange(n), -1))
    cols['dt_tr_log'] = np.where(last_tr >= 0, np.log1p((ts - ts[np.maximum(last_tr, 0)]) / 1000), 10.0)
    tv = d['trade_vol'].astype(float)
    ltv = np.where(last_tr >= 0, tv[np.maximum(last_tr, 0)], 0.0)
    cols['ltr_sign'] = np.sign(ltv); cols['ltr_lsz'] = np.sign(ltv) * np.log1p(np.abs(ltv))
    cols['hole'] = mid_ff - rm
    f70 = cols['fs70']
    cols['fs70_x_spread'] = f70 * cols['spread']
    cols['sgnflow100_x_absfs'] = np.sign(cols['flow_ms100']) * np.abs(f70)
    cols['fs70_cube'] = f70 * np.abs(f70)
    cols['fs70_x_lw'] = f70 * np.log1p(cols['fw70'])
    return ts, cols, rm, mid_ff


def rows(data, H, frac, feats):
    ts, cols, rm, mid_ff = data
    e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
    tgt = e - mid_ff
    tgt[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
    valid = np.isfinite(tgt)
    for nm in feats:
        valid &= np.isfinite(cols[nm])
    t_end = ts[0] + frac * (ts[-1] - ts[0])
    tri = np.where(valid & (ts + LATENCY_US + H * 1000 <= t_end))[0][::3]
    return np.column_stack([cols[c][tri] for c in feats]).astype(np.float32), tgt[tri]


VARIANTS = {  # name: (datasets with fraction, H, alpha)
    'parity_r3_75': ([('round3', .75)], 150, 10.0),
    'r3_100': ([('round3', 1.0)], 150, 10.0),
    'r3m_75': ([('round3', .75), ('round3_mirror', .75)], 150, 10.0),
    'r3m_100': ([('round3', 1.0), ('round3_mirror', 1.0)], 150, 10.0),
    'r3r2_75': ([('round3', .75), ('round2', .75)], 150, 10.0),
    'r3r2_100': ([('round3', 1.0), ('round2', 1.0)], 150, 10.0),
    'h100': ([('round3', .75)], 100, 10.0),
    'h200': ([('round3', .75)], 200, 10.0),
    'a30': ([('round3', .75)], 150, 30.0),
    'a3': ([('round3', .75)], 150, 3.3),
    'h250': ([('round3', .75)], 250, 10.0),
    'h300': ([('round3', .75)], 300, 10.0),
    'h400': ([('round3', .75)], 400, 10.0),
    'h200_r3r2_75': ([('round3', .75), ('round2', .75)], 200, 10.0),
    'h200_r3m_75': ([('round3', .75), ('round3_mirror', .75)], 200, 10.0),
}

if __name__ == '__main__':
    feats = list(pred_r3.LEAN)
    data = {n: cols_for(n) for n in ('round3', 'round3_mirror', 'round2')}
    print('# order', feats, flush=True)
    import sys
    for vn, (sets, H, alpha) in VARIANTS.items():
        if len(sys.argv) > 1 and vn not in sys.argv[1:]:
            continue
        parts = [rows(data[n], H, fr, feats) for n, fr in sets]
        X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        m = fit_ridge(X, y, feats, alpha=alpha, clip_q=0.0)
        wr = m.coef / m.scale
        c0 = float(m.intercept - (wr * m.mean).sum())
        print(f'{vn} = ({round(c0, 7)}, {tuple(round(float(x), 7) for x in wr)})', flush=True)
