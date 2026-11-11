"""Training rows for research/retrain.py, built once per tape (copies of the round-3 scripts with the tape paths as arguments).

Per tape (the public tape and its mirror) this computes the lean 22 columns exactly like research/diag/pred_rfit_r3.cols_for,
the extras of research/diag/mlp2_r3.py and the slow columns of research/diag/slow_r3.py, sharing one fair_multi pass, and
returns the row sets every component trains on:
  r200 (ridge rfit + taker MLP): H 200, valid lean, train = first 75 % strided by 3, holdout = after 75 %;
  r500 (quote MLP): same with H 500;
  m2 (GRU, = mlp2_r3.npz): y200 and y150 valid + extras valid, X = lean 22 + extras, idx kept;
  slow (5 s ridge): X 17 slow columns, targets 1 s / 5 s, ts, t_end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals import LATENCY_US, build_matrix, value_before
from research.features import mid_ticks
from research.diag import pred_r3, mlp2_r3, slow_r3

LEAN = list(pred_r3.LEAN)


def lean_cols(csv, d, fm):
    """pred_rfit_r3.cols_for with explicit inputs."""
    ts = d['ts_us'].astype(np.int64); n = len(ts)
    X0, names0 = build_matrix(d)
    cols = {nm: X0[:, i].astype(np.float64) for i, nm in enumerate(names0)}
    del X0
    cols.update(fm)
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


def slow_X(d, fm):
    """slow_r3.cols_for + build with explicit inputs."""
    ts = d['ts_us'].astype(np.int64)
    b = d['bid_px'].astype(float); a = d['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    rm = pd.Series(np.where(ok, (b + a) / 2, np.nan)).ffill().bfill().values
    bad = (b <= 0) | (a <= 0) | (a <= b)
    mid_ff = (pd.Series(np.where(bad, np.nan, b)).ffill().bfill().values + pd.Series(np.where(bad, np.nan, a)).ffill().bfill().values) / 2
    cols = {}
    for w in slow_r3.WINS:
        cols[f'd{w}'] = rm - value_before(ts, rm, ts - w * 1_000_000)
        cols[f'd{w}'][np.isnan(cols[f'd{w}'])] = 0.0
    for t in (20, 50, 70, 150, 300):
        cols[f'fs{t}'] = fm[f'fs{t}']
    cf = np.cumsum(d['trade_vol'].astype(np.float64))
    for w in slow_r3.FW:
        prev = value_before(ts, cf, ts - w * 1_000_000 + 1)
        prev[np.isnan(prev)] = 0.0
        cols[f'fl{w}'] = cf - prev
    cols['spread'] = (a - b).clip(0, 10)
    cols['hole'] = mid_ff - rm
    cols['d5_abs'] = np.abs(cols['d5']); cols['d30_abs'] = np.abs(cols['d30'])
    X = np.column_stack([cols[n] for n in slow_r3.NAMES]).astype(np.float32)
    ys = {}
    for h in slow_r3.HS:
        e = value_before(ts, rm, ts + LATENCY_US + h * 1_000_000)
        y = e - rm
        y[ts + LATENCY_US + h * 1_000_000 > ts[-1]] = np.nan
        ys[h] = y
    return X, ys


def tape_rows(csv, npz):
    z = np.load(npz); d = {k: z[k] for k in z.files if k != 'source'}
    fm = pred_r3.fair_multi(csv, d)
    ts, cols, rm, mid_ff = lean_cols(csv, d, fm)
    t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
    out = {}
    tg = {}
    for H in (150, 200, 500):
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        t = e - mid_ff
        t[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        tg[H] = t
    # r200 / r500: pred_rfit_r3.rows (frac .75) == mlp_r3.build == mlpq_r3.build rows
    for H in (200, 500):
        valid = np.isfinite(tg[H])
        for nm in LEAN:
            valid &= np.isfinite(cols[nm])
        tri = np.where(valid & (ts + LATENCY_US + H * 1000 <= t_end))[0][::3]
        hoi = np.where(valid & (ts > t_end))[0]
        for tag, idx in (('tr', tri), ('ho', hoi)):
            out[f'r{H}_{tag}_X'] = np.column_stack([cols[c][idx] for c in LEAN]).astype(np.float32)
            out[f'r{H}_{tag}_y'] = tg[H][idx]
            out[f'r{H}_{tag}_spread'] = cols['spread'][idx].astype(np.float32)
    # m2: mlp2_r3.build rows (used by seq_r3)
    cols.update(mlp2_r3.extras(d, ts, cols['spread']))
    valid = np.isfinite(tg[200]) & np.isfinite(tg[150])
    for nm in LEAN + mlp2_r3.EXTRA:
        valid &= np.isfinite(cols[nm])
    tri = np.where(valid & (ts + LATENCY_US + 200 * 1000 <= t_end))[0][::3]
    hoi = np.where(valid & (ts > t_end))[0]
    for tag, idx in (('tr', tri), ('ho', hoi)):
        out[f'm2_{tag}_X'] = np.column_stack([cols[c][idx] for c in LEAN]).astype(np.float32)
        out[f'm2_{tag}_idx'] = idx
        out[f'm2_{tag}_y200'] = tg[200][idx].astype(np.float32)
        out[f'm2_{tag}_spread'] = cols['spread'][idx].astype(np.float32)
    del cols
    sX, sy = slow_X(d, fm)
    out['slow_X'] = sX
    for h, y in sy.items():
        out[f'slow_y{h}'] = y
    out['ts'] = ts
    out['t_end'] = np.array(t_end)
    return out
