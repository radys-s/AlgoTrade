"""Structural (closed-form) depletion features from the round1 generator rules, plus walk-forward evaluation.

    uv run python -X utf8 -m research.features_struct eval data/features/round1.npz data/features/sample_v2.npz

Per side s with touch lots q and touch order count n (state after the row):
  P_big(q)   = 1 - Phi((ln q - MU)/SIG)                   an IOC at least as big as the queue
  P_trade(h) = (1 - exp(-LAM h)) * P_big(q)                one big IOC within h (4.0/s per side)
  P_cxl(h)   = F_R(h) ** n                                 all n orders cancel within h, F_R = stationary residual-life CDF
  P_dep(h)   = 1 - (1 - P_trade)(1 - P_cxl)
  E_dm(h)    = 0.5 (P_dep_ask - P_dep_bid) * (1 + 0.7 if h >= 1000 else 1)    ticks, + = bullish
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from research.features import load_features, mid_ticks
from research.signals import (
    build_matrix, fit_ridge, future_mid_change, oos_stats, value_at_or_before, walk_forward,
)

MU, SIG, LAM = 4.106, 1.395, 4.0          # IOC size lognormal, IOC rate per side per second
L_SHIFT, L_SCALE, L_SIG = -30.0, 353.0, 0.85  # cancel lifetime ms: shift + scale * lognormal(0, sigma)
CONT = 0.7                                # refill continuation probability
HS = (100, 1000)


def _residual_cdf(h_ms: float) -> float:
    """F_R(h) = (1/E[L]) * int_0^h S(u) du with S the shifted-lognormal survival (lifetimes clipped at 0)."""
    u = np.linspace(0.0, 20000.0, 200001)
    z = np.log(np.maximum(u - L_SHIFT, 1e-9) / L_SCALE) / L_SIG
    s = 1.0 - ndtr(z)
    cum = np.concatenate([[0.0], np.cumsum((s[1:] + s[:-1]) / 2 * np.diff(u))])
    return float(np.interp(h_ms, u, cum) / cum[-1])


LOG_FR = {h: math.log(_residual_cdf(h)) for h in HS}  # log F_R(h)


def p_dep(q, n, h_ms):
    """P(touch level is gone within h): big IOC or all n orders cancelled."""
    q = np.maximum(q.astype(np.float64), 1.0)
    big = 1.0 - ndtr((np.log(q) - MU) / SIG)
    trade = (1.0 - math.exp(-LAM * h_ms / 1000.0)) * big
    cxl = np.exp(np.maximum(n.astype(np.float64), 1.0) * LOG_FR[h_ms])
    return 1.0 - (1.0 - trade) * (1.0 - cxl), big


NAMES = (['pbig_b', 'pbig_a', 'pbig_diff']
         + [f'{k}_{h}' for h in HS for k in ('pdep_b', 'pdep_a', 'pdep_diff', 'pdep_lo', 'edm')])


def struct_features(d: dict[str, np.ndarray]) -> np.ndarray:
    """(n_events, len(NAMES)) float32; NaN on one-sided rows. O(1) per event: 2 ndtr, 4 exp."""
    bq, aq = d['bid_qty'][:, 0], d['ask_qty'][:, 0]
    bn, an = d['bid_cnt1'], d['ask_cnt1']
    cols = []
    _, bb = p_dep(bq, bn, 100)
    _, ba = p_dep(aq, an, 100)
    cols += [bb, ba, ba - bb]
    for h in HS:
        pb, _ = p_dep(bq, bn, h)
        pa, _ = p_dep(aq, an, h)
        eps = 1e-6
        lo = np.log((pa + eps) / (1 - pa + eps)) - np.log((pb + eps) / (1 - pb + eps))
        edm = 0.5 * (pa - pb) * (1 + CONT if h >= 1000 else 1.0)
        cols += [pb, pa, pa - pb, lo, edm]
    X = np.column_stack(cols).astype(np.float32)
    X[np.isnan(mid_ticks(d))] = np.nan
    return X


def build(d, rows, horizons=HS, n_chunks=8):
    ts = d['ts_us']
    n = len(ts)
    X0, names0 = build_matrix(d, rows=rows)
    Xs = struct_features(d)[rows]
    X = np.hstack([X0, Xs])
    mid_ff = mid_ticks(d, ffill=True)
    Y = np.column_stack([future_mid_change(ts, mid_ff, h * 1000)[rows] for h in horizons])
    # realised "touch level gone within 100 ms": future best price on the side is worse than now
    t100 = ts[rows] + 100_000
    gb = (value_at_or_before(ts, d['bid_px'].astype(np.float64), t100) < d['bid_px'][rows]).astype(np.float32)
    ga = (value_at_or_before(ts, d['ask_px'].astype(np.float64), t100) > d['ask_px'][rows]).astype(np.float32)
    keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return {'X': X[keep], 'Y': Y[keep].astype(np.float64), 'ts_us': ts[rows][keep], 'gone': np.column_stack([gb, ga])[keep],
            'chunk': (rows[keep] * n_chunks // n).astype(np.int8), 'names': names0 + NAMES}


def summ(folds, model='ridge'):
    ics = np.array([f.stats[model]['ic'] for f in folds])
    return ics, np.nanmean([f.stats[model]['edge10'] for f in folds]), np.nanmean([f.stats[model]['edge1'] for f in folds])


def fold_ic_raw(x, y, chunk):
    """IC of an unfitted signal per test chunk 1.. (same chunks as the walk-forward) + edge on its own top deciles."""
    ics, e10, e1 = [], [], []
    for k in range(1, int(chunk.max()) + 1):
        m = chunk == k
        ics.append(np.corrcoef(x[m], y[m])[0, 1])
        for q, lst in ((0.9, e10), (0.99, e1)):
            s = np.abs(x[m]) >= np.quantile(np.abs(x[chunk < k]), q)
            lst.append(float((np.sign(x[m][s]) * y[m][s]).mean()) if s.any() else np.nan)
    return np.array(ics), np.nanmean(e10), np.nanmean(e1)


def run_eval(paths, hgb=True, stride=2):
    sets = {}
    for p in paths:
        d = load_features(Path(p))
        t = time.time()
        rows = np.arange(0, len(d['ts_us']), stride)
        sets[Path(p).stem] = build(d, rows)
        print(p, len(sets[Path(p).stem]['ts_us']), f'rows {time.time() - t:.0f}s', flush=True)
        del d
    names = next(iter(sets.values()))['names']
    ix = {n: i for i, n in enumerate(names)}
    base = list(range(23))
    for stem, ds in sets.items():
        X, G = ds['X'], ds['gone']
        pd_ = np.concatenate([X[:, ix['pdep_b_100']], X[:, ix['pdep_a_100']]])
        gn = np.concatenate([G[:, 0], G[:, 1]])
        edges = np.array([0, .005, .01, .02, .05, .1, .2, .3, .5, .7, 1.0001])
        print(f'\n[{stem}] calibration P_dep(100ms): bin n mean_pred realised')
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pd_ >= lo) & (pd_ < hi)
            if m.sum() > 100:
                print(f'  [{lo:.3f},{hi:.3f}) {m.sum():8d} {pd_[m].mean():.4f} {gn[m].mean():.4f}', flush=True)
        for hi_, h in enumerate(HS):
            y = ds['Y'][:, hi_]
            e = X[:, ix[f'edm_{h}']]
            ics, e10, e1 = fold_ic_raw(e, y, ds['chunk'])
            print(f'[{stem}] h{h} E[dm] alone: ic {ics.mean():+.4f} pos {(ics > 0).sum()}/{len(ics)} e10 {e10:.4f} e1 {e1:.4f}', flush=True)
            allS = base + [ix[n] for n in NAMES]
            cfg = {'baseline23': base, '+edm': base + [ix[f'edm_{h}']], '+struct_all': allS}
            for cn, idx in cfg.items():
                folds, _ = walk_forward(X[:, idx], y, ds['ts_us'], ds['chunk'], [names[i] for i in idx], h * 1000, use_hgb=False, verbose=False)
                ics, e10, e1 = summ(folds)
                print(f'[{stem}] h{h} ridge {cn:12s} ic {ics.mean():+.4f} pos {(ics > 0).sum()}/{len(ics)} e10 {e10:.4f} e1 {e1:.4f}  folds {np.round(ics, 3).tolist()}', flush=True)
            if hgb:
                for cn in ('baseline23', '+struct_all'):
                    idx = cfg[cn]
                    folds, _ = walk_forward(X[:, idx], y, ds['ts_us'], ds['chunk'], [names[i] for i in idx], h * 1000, use_hgb=True, verbose=False)
                    ics, e10, e1 = summ(folds, 'hgb')
                    print(f'[{stem}] h{h} HGB   {cn:12s} ic {ics.mean():+.4f} pos {(ics > 0).sum()}/{len(ics)} e10 {e10:.4f} e1 {e1:.4f}', flush=True)
    if len(sets) == 2:
        (s1, a), (s2, b) = list(sets.items())
        for hi_, h in enumerate(HS):
            e = b['X'][:, ix[f'edm_{h}']]
            print(f'cross {s2} h{h} E[dm] alone (no fit): ic {np.corrcoef(e, b["Y"][:, hi_])[0, 1]:+.4f}')
            for cn, idx in {'baseline23': base, '+struct_all': base + [ix[n] for n in NAMES]}.items():
                m = fit_ridge(a['X'][:, idx], a['Y'][:, hi_], [names[i] for i in idx])
                st = oos_stats(m.predict(b['X'][:, idx]), b['Y'][:, hi_], m.predict(a['X'][::10, idx]))
                print(f'cross {s1}->{s2} h{h} ridge {cn}: ic {st["ic"]:+.4f} e10 {st["edge10"]:.4f} e1 {st["edge1"]:.4f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['eval'])
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--no-hgb', action='store_true')
    a = ap.parse_args()
    print('F_R(100)=', math.exp(LOG_FR[100]), 'F_R(1000)=', math.exp(LOG_FR[1000]))
    run_eval(a.paths, hgb=not a.no_hgb)
