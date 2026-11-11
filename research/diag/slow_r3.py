"""Learned slow direction model (exp_r3_slow). Ridge on slow features -> robust-mid move over 1 s / 5 s.
    .venv/Scripts/python.exe -X utf8 -m research.diag.slow_r3 > logs/slow_r3.txt
Train = first 75 % of round3 + mirror (rows strided by 3), holdout = last 25 %. Prints baked (mean, scale, c0, w) per horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals import LATENCY_US, value_before
from research.diag import pred_r3

WINS = (1, 5, 15, 30, 60)          # drift windows, s
FW = (1, 3, 10)                    # flow windows, s
NAMES = ([f'd{w}' for w in WINS] + ['fs20', 'fs50', 'fs70', 'fs150', 'fs300'] + [f'fl{w}' for w in FW]
         + ['spread', 'hole', 'd5_abs', 'd30_abs'])
HS = (1, 5)


def cols_for(name):
    z = np.load(f'data/features/{name}.npz'); d = {k: z[k] for k in z.files}
    ts = d['ts_us'].astype(np.int64)
    b = d['bid_px'].astype(float); a = d['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    rm = pd.Series(np.where(ok, (b + a) / 2, np.nan)).ffill().bfill().values
    bad = (b <= 0) | (a <= 0) | (a <= b)
    mid_ff = (pd.Series(np.where(bad, np.nan, b)).ffill().bfill().values + pd.Series(np.where(bad, np.nan, a)).ffill().bfill().values) / 2
    fm = pred_r3.fair_multi(f'data/{name}.csv', d)
    cols = {}
    for w in WINS:
        cols[f'd{w}'] = rm - value_before(ts, rm, ts - w * 1_000_000)
        cols[f'd{w}'][np.isnan(cols[f'd{w}'])] = 0.0
    for t in (20, 50, 70, 150, 300):
        cols[f'fs{t}'] = fm[f'fs{t}']
    cf = np.cumsum(d['trade_vol'].astype(np.float64))
    for w in FW:
        # decayed in-window flow: cf now minus cf at t - w (events with ts in (t-w, t])
        prev = value_before(ts, cf, ts - w * 1_000_000 + 1)
        prev[np.isnan(prev)] = 0.0
        cols[f'fl{w}'] = cf - prev
    cols['spread'] = (a - b).clip(0, 10)
    cols['hole'] = mid_ff - rm
    cols['d5_abs'] = np.abs(cols['d5']); cols['d30_abs'] = np.abs(cols['d30'])
    return ts, cols, rm


def build(name):
    ts, cols, rm = cols_for(name)
    X = np.column_stack([cols[n] for n in NAMES]).astype(np.float32)
    ys = {}
    for h in HS:
        e = value_before(ts, rm, ts + LATENCY_US + h * 1_000_000)
        y = e - rm
        y[ts + LATENCY_US + h * 1_000_000 > ts[-1]] = np.nan
        ys[h] = y
    t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
    return ts, X, ys, t_end


def main():
    sets = ('round3', 'round3_mirror')
    B = {s: build(s) for s in sets}
    for h in HS:
        Xtr, ytr = [], []
        for s in sets:
            ts, X, ys, t_end = B[s]
            y = ys[h]
            m = np.isfinite(y) & (ts + LATENCY_US + h * 1_000_000 <= t_end)
            idx = np.where(m)[0][::3]
            Xtr.append(X[idx]); ytr.append(y[idx])
        Xtr = np.vstack(Xtr).astype(np.float64); ytr = np.concatenate(ytr)
        mean = Xtr.mean(0); sc = Xtr.std(0); sc[sc == 0] = 1
        Z = (Xtr - mean) / sc; Z = np.clip(Z, -8, 8)
        A = Z.T @ Z + 10.0 * np.eye(Z.shape[1]); ym = ytr.mean()
        w = np.linalg.solve(A, Z.T @ (ytr - ym))
        wraw = w / sc; c0 = ym - (mean * wraw).sum()
        print(f'## H={h}s n={len(ytr)} ystd={ytr.std():.3f}')
        for s in sets:
            ts, X, ys, t_end = B[s]
            y = ys[h]; ho = np.where(np.isfinite(y) & (ts > t_end))[0][::2]
            p = np.clip((X[ho] - mean) / sc, -8, 8) @ w + ym
            cor = np.corrcoef(p, y[ho])[0, 1]
            g = np.corrcoef(X[ho][:, NAMES.index('d30')], y[ho])[0, 1]
            print(f'  {s}: holdout corr {cor:.4f}  (hand gate drift30 corr {g:.4f}) pstd {p.std():.3f}')
            for thr in (0.5, 1, 1.5, 2, 3):
                hit = np.abs(p) >= thr
                print(f'    |p|>={thr}: frac {hit.mean():.4f} signed move {np.mean(np.sign(p[hit]) * y[ho][hit]) if hit.any() else np.nan:.3f}')
        print('NAMES =', NAMES)
        print(f'W{h} = ({c0!r}, {np.round(wraw, 8).tolist()!r})  # raw-feature weights, clip not baked')
        print(f'W{h}_STD = ({mean.tolist()!r}, {sc.tolist()!r}, {w.tolist()!r}, {ym!r})')


if __name__ == '__main__':
    main()
