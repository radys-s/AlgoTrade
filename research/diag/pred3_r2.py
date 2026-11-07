"""Offline: ridge predictors aimed at the fair taker's decision region (round2, lean features of mm_r2_pred).

    uv run python -X utf8 -m research.diag.pred3_r2 [--export name,name]
Variants (all on the first 75 % of round2 time, evaluated on the last 25 %):
  * region fits: rows with |fs70| > g only (the taker can only act where |signal| > spread/2 + take_k);
    online the model is used only where |fair_sig| > g (else fair_sig) -> evaluated the same way;
  * horizon 300 / 500 ms targets; separate models per spread (1 vs >= 2) and per side (fs70 > 0 / < 0).
Metrics: mean signed move (ticks) of the top 1/3/6 % of |signal| for the raw model and the 50/50 blend with fs70, and
the realized cross edge (move - spread/2) summed over rows where the blend crosses (|s| >= spread/2 + 0.2).
Weights are printed folded to raw features in PRED_NAMES order (c0, w) for strategies/exp_r2_pred3.py.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from research.diag.pred_r2 import CSV, LEAN, NPZ, fair_multi
from research.features import mid_ticks
from research.signals import LATENCY_US, build_matrix, value_before

CACHE = os.environ.get('PRED3_CACHE', '')


def build_cols() -> tuple[dict, np.ndarray, np.ndarray]:
    if CACHE and os.path.exists(CACHE):
        z = np.load(CACHE)
        cols = {k: z[k] for k in z.files if k not in ('_ts', '_rm', '_mid')}
        return cols, z['_ts'], (z['_rm'], z['_mid'])
    z = np.load(NPZ)
    d = {k: z[k] for k in z.files}
    ts = d['ts_us'].astype(np.int64)
    n = len(ts)
    X0, names0 = build_matrix(d)
    cols = {nm: X0[:, i].astype(np.float64) for i, nm in enumerate(names0) if nm in LEAN}
    del X0
    cols.update(fair_multi(CSV, d))
    b = d['bid_px'].astype(float); a = d['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    rm = pd.Series(np.where(ok, (b + a) / 2, np.nan)).ffill().bfill().values
    mid_ff = mid_ticks(d, ffill=True)
    cols['lq_b'] = np.log1p(d['bid_qty'][:, 0].astype(float))
    cols['lq_a'] = np.log1p(d['ask_qty'][:, 0].astype(float))
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
    cols = {k: cols[k] for k in LEAN}
    if CACHE:
        np.savez(CACHE, _ts=ts, _rm=rm, _mid=mid_ff, **cols)
    return cols, ts, (rm, mid_ff)


def ridge(X: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> tuple[float, np.ndarray]:
    """Standardized closed-form ridge, folded back to raw-feature weights: y ~ c0 + X @ w."""
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(X.shape[1]), Z.T @ (y - ym))
    w = coef / sd
    return float(ym - mu @ w), w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default='', help='comma list of variant names to print weights for')
    args = ap.parse_args()
    cols, ts, (rm, mid) = build_cols()
    X = np.column_stack([cols[f] for f in LEAN])
    fs = cols['fs70']; spr = cols['spread']
    t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
    tgt = {}
    for H in (150, 300, 500):
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        zt = e - mid
        zt[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        tgt[H] = zt
    valid = np.isfinite(X).all(1) & np.isfinite(mid) & np.isfinite(tgt[500])
    tr_all = valid & (ts + LATENCY_US + 500_000 <= t_end)
    ho = np.where(valid & (ts > t_end))[0]
    y_ho = tgt[150][ho]; X_ho = X[ho]; fs_ho = fs[ho]; spr_ho = spr[ho]
    st = 3

    def fit(H, mask):
        idx = np.where(tr_all & mask)[0][::st]
        return ridge(X[idx], tgt[H][idx])

    variants = {}
    variants['m0_all_150'] = ('plain', fit(150, np.ones_like(valid)), 0.0)
    for g in (0.3, 0.5, 0.7):
        variants[f'a{g}_150'] = ('plain', fit(150, np.abs(fs) > g), g)
    variants['all_300'] = ('plain', fit(300, np.ones_like(valid)), 0.0)
    variants['all_500'] = ('plain', fit(500, np.ones_like(valid)), 0.0)
    variants['a0.5_300'] = ('plain', fit(300, np.abs(fs) > 0.5), 0.5)
    variants['a0.5_500'] = ('plain', fit(500, np.abs(fs) > 0.5), 0.5)
    variants['spr_150'] = ('spread', (fit(150, spr <= 1), fit(150, spr >= 2)), 0.0)
    variants['spr_a0.5_150'] = ('spread', (fit(150, (spr <= 1) & (np.abs(fs) > 0.5)), fit(150, (spr >= 2) & (np.abs(fs) > 0.5))), 0.5)
    variants['side_a0.3_150'] = ('side', (fit(150, fs > 0.3), fit(150, fs < -0.3)), 0.3)
    variants['side_a0.5_150'] = ('side', (fit(150, fs > 0.5), fit(150, fs < -0.5)), 0.5)

    def predict(kind, m, Xh, fsh, sprh, g):
        if kind == 'plain':
            p = m[0] + Xh @ m[1]
        elif kind == 'spread':
            p = np.where(sprh <= 1, m[0][0] + Xh @ m[0][1], m[1][0] + Xh @ m[1][1])
        else:
            p = np.where(fsh > 0, m[0][0] + Xh @ m[0][1], m[1][0] + Xh @ m[1][1])
        if g > 0:
            p = np.where(np.abs(fsh) > g, p, fsh)
        return p

    def tails(s):
        out = []
        for q in (0.01, 0.03, 0.06):
            t = np.abs(s) >= np.quantile(np.abs(s), 1 - q)
            out.append((np.sign(s[t]) * y_ho[t]).mean())
        return out

    def cross(s, tk=0.2):
        hit = np.abs(s) >= spr_ho / 2 + tk
        edge = np.sign(s[hit]) * y_ho[hit] - spr_ho[hit] / 2
        return hit.mean() * 100, edge.mean(), edge.sum() / 1000

    print(f'holdout rows {len(ho)}; corr(fs70, y150) {np.corrcoef(fs_ho, y_ho)[0, 1]:.4f}')
    t = tails(fs_ho); c = cross(fs_ho)
    print(f"{'fair_sig fs70':16s} raw  top1/3/6 {t[0]:.3f}/{t[1]:.3f}/{t[2]:.3f}  cross% {c[0]:.2f} edge {c[1]:.3f} sum {c[2]:.1f}k")
    for name, (kind, m, g) in variants.items():
        p = predict(kind, m, X_ho, fs_ho, spr_ho, g)
        cr = np.corrcoef(p, y_ho)[0, 1]
        t1 = tails(p); bl = 0.5 * p + 0.5 * fs_ho; t2 = tails(bl)
        c1 = cross(p); c2 = cross(bl)
        print(f'{name:16s} corr {cr:.3f} raw {t1[0]:.3f}/{t1[1]:.3f}/{t1[2]:.3f} x% {c1[0]:.2f} e {c1[1]:.3f} sum {c1[2]:.1f}k'
              f' | blend {t2[0]:.3f}/{t2[1]:.3f}/{t2[2]:.3f} x% {c2[0]:.2f} e {c2[1]:.3f} sum {c2[2]:.1f}k', flush=True)
    for name in [s for s in args.export.split(',') if s]:
        kind, m, g = variants[name]
        ms = [m] if kind == 'plain' else list(m)
        for c0, w in ms:
            print(name, kind, g, f'({c0:.7g}, ({", ".join(f"{x:.7g}" for x in w)})),')


if __name__ == '__main__':
    main()
