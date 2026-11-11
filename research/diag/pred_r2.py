"""Offline: can a ridge on book/flow features + multi-tau fair_sig predict the robust-mid move better than fair_sig?

    uv run python -X utf8 -m research.diag.pred_r2 [--full] [--export H] [--features a,b,c]
Target: robust mid (mid of the last state with spread <= 3) at t+5ms+H minus at t+5ms (y_H) and minus the
current mid at any spread (z_H, what _fq adds fair_sig to). Fit on the first 75 % of round2 time, report holdout corr.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from research.signals import LATENCY_US, build_matrix, export_linear, fit_ridge, value_before
from research.features import mid_ticks

CSV = 'data/round2.csv'
NPZ = 'data/features/round2.npz'
TAUS = (20, 50, 70, 150, 300)
LEAN = ['fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'fs70_x_lw', 'fs70_cube', 'sgnflow100_x_absfs', 'fs70_x_spread',
        'spread', 'imb1', 'cnt_imb1', 'lq_b', 'lq_a', 'ltr_sign', 'ltr_lsz', 'ltp_dev', 'hole', 'flow_ev10',
        'flow_ms100', 'flow_ms1000', 'dt_tr_log']


def fair_multi(csv: str, d: dict, taus=TAUS):
    """fair_fills.fair_signal for several taus in one pass; also the decayed insert count w (tau 70)."""
    df = pd.read_csv(csv)
    ts = d['ts_us'].astype(np.int64)
    bb = d['bid_px'].astype(float); ba = d['ask_px'].astype(float)
    bad = (bb <= 0) | (ba <= 0) | (ba <= bb)
    bb = pd.Series(np.where(bad, np.nan, bb)).ffill().bfill().values
    ba = pd.Series(np.where(bad, np.nan, ba)).ffill().bfill().values
    mid = (bb + ba) / 2
    pm = np.r_[mid[0], mid[:-1]]
    px = df.Price.values.astype(float)
    sd = np.where(df.Side.values == 'B', 1, -1)
    isG = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
    off_b = np.mean((px - pm)[isG & (sd > 0)]); off_a = np.mean((px - pm)[isG & (sd < 0)])
    dev = np.where(isG, px - np.where(sd > 0, off_b, off_a) - pm, np.nan)
    idx = np.where(isG)[0]
    dt = np.diff(ts[idx], prepend=ts[idx[0]]).astype(float)
    x = dev[idx]
    out = {}
    for tau in taus:
        dec = np.exp(-dt / (tau * 1e3))
        s = w = 0.0
        o = np.empty(len(idx)); ow = np.empty(len(idx))
        for i in range(len(idx)):
            a = dec[i]
            s = s * a + x[i]; w = w * a + 1.0; o[i] = s / w; ow[i] = w
        sig = np.full(len(df), np.nan); sig[idx] = o
        out[f'fs{tau}'] = pd.Series(sig).ffill().fillna(0).values
        if tau == 70:
            # w decays between inserts too; approximate with the value at the last insert, decayed to now
            wl = np.full(len(df), np.nan); wl[idx] = ow
            tl = np.full(len(df), np.nan); tl[idx] = ts[idx]
            wl = pd.Series(wl).ffill().fillna(0).values; tl = pd.Series(tl).ffill().fillna(ts[0]).values
            out['fw70'] = wl * np.exp(-(ts - tl) / 70e3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', type=int, default=0, help='horizon ms to export (full-feature ridge)')
    ap.add_argument('--full', action='store_true', help='export the model fitted on 100 %% of the tape')
    ap.add_argument('--features', default='', help='comma list for the exported model (default: all)')
    ap.add_argument('--target', default='z', help='y (rm - rm(t+5ms)) or z (rm - mid now) for the export')
    ap.add_argument('--hs', default='100,150,300')
    ap.add_argument('--tg', default='y,z')
    ap.add_argument('--clip-y', type=float, default=0.0, help='clip the TRAINING target at +-this (0 = off)')
    ap.add_argument('--lean', default='', help='comma list replacing the lean set')
    args = ap.parse_args()
    global HS, TG, LEAN, CLIPY
    CLIPY = args.clip_y
    if args.lean:
        LEAN = args.lean.split(',')
    HS = [int(h) for h in args.hs.split(',')]
    TG = args.tg.split(',')
    z = np.load(NPZ)
    d = {k: z[k] for k in z.files}
    ts = d['ts_us'].astype(np.int64)
    n = len(ts)
    X0, names0 = build_matrix(d)
    cols = {nm: X0[:, i].astype(np.float64) for i, nm in enumerate(names0)}
    del X0
    fm = fair_multi(CSV, d)
    cols.update(fm)
    b = d['bid_px'].astype(float); a = d['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    rm = np.where(ok, (b + a) / 2, np.nan)
    rm = pd.Series(rm).ffill().bfill().values
    mid_ff = mid_ticks(d, ffill=True)
    # cheap extras
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
    cols['fs70_x_imb1'] = f70 * cols['imb1']
    names = list(cols)
    t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
    res = []
    for H in HS:
        s0 = value_before(ts, rm, ts + LATENCY_US)
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        y = e - s0
        zt = e - mid_ff
        y[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        zt[np.isnan(y)] = np.nan
        for tname, tgt in (('y', y), ('z', zt)):
            if tname not in TG:
                continue
            valid = np.isfinite(tgt) & np.isfinite(mid_ff)
            for nm in names:
                valid &= np.isfinite(cols[nm])
            tr_m = valid & (ts + LATENCY_US + H * 1000 <= t_end)
            ho_m = valid & (ts > t_end)
            st = 3
            tri = np.where(tr_m)[0][::st]; hoi = np.where(ho_m)[0]

            def fit_eval(feats, alpha=10.0):
                Xtr = np.column_stack([cols[f][tri] for f in feats]).astype(np.float32)
                yt = np.clip(tgt[tri], -CLIPY, CLIPY) if CLIPY > 0 else tgt[tri]
                m = fit_ridge(Xtr, yt, feats, alpha=alpha, clip_q=0.0)
                Xho = np.column_stack([cols[f][hoi] for f in feats]).astype(np.float32)
                p = m.predict(Xho)
                return m, p
            yh = tgt[hoi]
            c_f70 = np.corrcoef(cols['fs70'][hoi], yh)[0, 1]
            fsets = {
                'fs_taus': [f'fs{t}' for t in TAUS] + ['fw70', 'fs70_x_lw'],
                'base': names0,
                'base+fs': names0 + [f'fs{t}' for t in TAUS],
                'all': names,
                'lean': LEAN,
            }
            row = {'H': H, 'tgt': tname, 'fs70': c_f70}
            for k, feats in fsets.items():
                m, p = fit_eval(feats)
                row[k] = np.corrcoef(p, yh)[0, 1]
                pr = p.copy() if k == 'lean' else None
                if k == 'lean':
                    # mean signed move in the top q of |pred| vs the top q of |fs70| (equal trigger rate)
                    f = cols['fs70'][hoi]
                    for q in (0.01, 0.03, 0.06):
                        tp = np.abs(pr) >= np.quantile(np.abs(pr), 1 - q)
                        tf = np.abs(f) >= np.quantile(np.abs(f), 1 - q)
                        row[f'edge{q}'] = f'{(np.sign(pr[tp]) * yh[tp]).mean():.3f}/{(np.sign(f[tf]) * yh[tf]).mean():.3f}'
                if k == 'all':
                    row['std_p'] = p.std(); row['std_fs70'] = cols['fs70'][hoi].std()
                    beta = np.polyfit(cols['fs70'][hoi], yh, 1)[0]
                    row['beta_fs70'] = beta
                    # taker-trigger quality: mean target where scaled pred beyond spread/2+0.2
                    spr = cols['spread'][hoi]
                    for sc in (1.0, 1.5, 2.0):
                        thr = spr / 2 + 0.2
                        hit = np.abs(sc * p) >= thr
                        row[f'n{sc}'] = hit.mean(); row[f'e{sc}'] = (np.sign(p[hit]) * yh[hit]).mean() if hit.any() else np.nan
                    hitf = np.abs(cols['fs70'][hoi]) >= spr / 2 + 0.2
                    row['nF'] = hitf.mean(); row['eF'] = (np.sign(cols['fs70'][hoi][hitf]) * yh[hitf]).mean()
                    coefs = sorted(zip(m.coef, feats), key=lambda t: -abs(t[0]))[:10]
                    row['top'] = ' '.join(f'{f}:{c:+.3f}' for c, f in coefs)
            res.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if args.export == H and args.target == tname:
                feats = args.features.split(',') if args.features else names
                idx_fit = np.where(valid)[0][::st] if args.full else tri
                Xf = np.column_stack([cols[f][idx_fit] for f in feats]).astype(np.float32)
                yf = np.clip(tgt[idx_fit], -CLIPY, CLIPY) if CLIPY > 0 else tgt[idx_fit]
                m = fit_ridge(Xf, yf, feats, alpha=10.0, clip_q=0.0)
                m.horizon_ms = H
                Xho = np.column_stack([cols[f][hoi] for f in feats]).astype(np.float32)
                print('export holdout corr', np.corrcoef(m.predict(Xho), yh)[0, 1], 'full' if args.full else '75%')
                export_linear(m)


if __name__ == '__main__':
    main()
