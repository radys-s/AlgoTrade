"""Offline: lean ridge (pred_r3 LEAN) + momentum / trade-sweep features for the 150 and 500 ms robust-mid move.

    .venv/Scripts/python.exe -X utf8 -m research.diag.pred_ms_r3 [--csv data/round3.csv --npz data/features/round3.npz]
Fit on the first 75 % of the tape's time (z target = robust mid at t+5ms+H minus mid now), report holdout corr and the
decision-point edge: mean signed move where the taker would fire (|signal| >= spread/2 + 0.2), with the prediction
scaled so it fires as often as the base's fair_sig (fs70) does. Prints the weights for strategies/exp_r3_pms.py.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from research.signals import LATENCY_US, build_matrix, fit_ridge, value_before
from research.features import mid_ticks
from research.diag import pred_r3

NEW = ['mom50', 'mom200', 'mom1000', 'dfs50', 'dfs200', 'swp100', 'swp_n']


def ewm_events(ts, x, has, tau_us):
    """Time-decayed mean of x over events where has, evaluated at every event (step value at the last update)."""
    idx = np.where(has)[0]
    out = np.full(len(ts), np.nan)
    s = w = 0.0
    tl = ts[idx[0]] if len(idx) else 0
    vals = np.empty(len(idx)); ws = np.empty(len(idx))
    for j, i in enumerate(idx):
        a = np.exp(-(ts[i] - tl) / tau_us); tl = ts[i]
        s = s * a + x[i]; w = w * a + 1.0
        vals[j] = s / w; ws[j] = w
    out[idx] = vals
    wo = np.full(len(ts), np.nan); wo[idx] = ws
    return pd.Series(out).ffill().fillna(0).values, pd.Series(wo).ffill().fillna(0).values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/round3.csv')
    ap.add_argument('--npz', default='data/features/round3.npz')
    ap.add_argument('--hs', default='150,500')
    args = ap.parse_args()
    z = np.load(args.npz)
    d = {k: z[k] for k in z.files}
    ts = d['ts_us'].astype(np.int64)
    n = len(ts)
    X0, names0 = build_matrix(d)
    cols = {nm: X0[:, i].astype(np.float64) for i, nm in enumerate(names0)}
    del X0
    cols.update(pred_r3.fair_multi(args.csv, d))
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
    # momentum of the robust mid and of fair_sig's level (mid + fs70), causal: value strictly before t - L
    for L in (50, 200, 1000):
        cols[f'mom{L}'] = rm - value_before(ts, rm, ts - L * 1000)
    lvl = rm + f70
    for L in (50, 200):
        cols[f'dfs{L}'] = lvl - value_before(ts, lvl, ts - L * 1000)
    # trade-implied: a market sweep's deepest price minus the pre-trade robust mid (last_trade_px = deepest fill)
    pm = np.r_[rm[0], rm[:-1]]
    ltp = d['last_trade_px'].astype(float)
    sw = np.where(tr & (ltp > 0), ltp - pm, 0.0)
    cols['swp100'], wsw = ewm_events(ts, sw, tr & (ltp > 0), 100e3)
    cols['swp_n'] = cols['swp100'] * np.log1p(wsw)
    lean = list(pred_r3.LEAN)
    t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
    spr = cols['spread']
    for H in [int(h) for h in args.hs.split(',')]:
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        tgt = e - mid_ff
        tgt[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        valid = np.isfinite(tgt)
        for nm in lean + NEW:
            valid &= np.isfinite(cols[nm])
        tri = np.where(valid & (ts + LATENCY_US + H * 1000 <= t_end))[0][::3]
        hoi = np.where(valid & (ts > t_end))[0]
        yh = tgt[hoi]
        f = f70[hoi]; sp = spr[hoi]
        thr = sp / 2 + 0.2
        hitf = np.abs(f) >= thr
        netf = (np.sign(f[hitf]) * yh[hitf] - sp[hitf] / 2).sum()
        print(f'H={H} net/lot {netf:.0f} fs70 corr {np.corrcoef(f, yh)[0, 1]:.4f} fires {hitf.sum()} edge {(np.sign(f[hitf]) * yh[hitf]).mean():.3f}')
        for tag, feats in (('lean', lean), ('lean+new', lean + NEW)):
            Xtr = np.column_stack([cols[c][tri] for c in feats]).astype(np.float32)
            m = fit_ridge(Xtr, tgt[tri], feats, alpha=10.0, clip_q=0.0)
            p = m.predict(np.column_stack([cols[c][hoi] for c in feats]).astype(np.float32))
            out = f'  {tag:9s} corr {np.corrcoef(p, yh)[0, 1]:.4f}'
            for bl in (0.5, 1.0):
                s = bl * p + (1 - bl) * f
                # scale so the trigger count matches fair_sig's
                lo, hi = 0.1, 10.0
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if (np.abs(mid * s) >= thr).sum() > hitf.sum():
                        hi = mid
                    else:
                        lo = mid
                    sc = mid
                hit = np.abs(sc * s) >= thr
                hit1 = np.abs(s) >= thr
                net1 = (np.sign(s[hit1]) * yh[hit1] - sp[hit1] / 2).sum()
                out += (f' | b{bl} sc{sc:.2f} edge {(np.sign(s[hit]) * yh[hit]).mean():.3f}'
                        f' (sc1 fires {hit1.sum()} edge {(np.sign(s[hit1]) * yh[hit1]).mean():.3f} net {net1:.0f})')
            print(out, flush=True)
            if tag == 'lean':
                wr = m.coef / m.scale
                print('  RAW c0', round(float(m.intercept - (wr * m.mean).sum()), 7), 'w', tuple(round(float(x), 7) for x in wr))
            if tag == 'lean+new':
                print('  c0', round(float(m.intercept), 6))
                print('  W', {c: round(float(w), 6) for c, w in zip(feats, m.coef)})


if __name__ == '__main__':
    main()
