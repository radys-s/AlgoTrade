"""H21: ridge scores trained on LARGE-move (classification-like) targets, same 23 features as the champion.

    uv run python -X utf8 -m research.fit_bigmove [--out logs/ml_bigmove]

Targets (order sent at t lands at t+5 ms; thr = 0.5 tick):
  D1: +1 / -1 / 0 = mid(t+105ms) - mid(t+5ms) >= thr / <= -thr / else     D2: same, 300 ms horizon
  D3: sign of the move after the first taker hit within 100 ms when |move| >= thr, else 0
  D4: D1 as max excursion inside the 100 ms window (touch of +-thr; larger excursion wins if both)
Reference scores: base (plain 100 ms ridge) and B (move after first hit), both fitted on the same train split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.features import load_features, mid_ticks
from research.fit_target import ALPHA, build_targets, hit_targets
from research.signals import LATENCY_US, build_dataset, export_linear, fit_ridge, pearson, value_before

THR = 0.5
H100 = 100_000


def excursion(ts: np.ndarray, mid: np.ndarray, rows: np.ndarray, h_us: int) -> tuple[np.ndarray, np.ndarray]:
    """Max up / max down excursion of the mid from mid(t+5ms) inside the window, via reduceat."""
    start = ts[rows] + LATENCY_US
    m0 = value_before(ts, mid, start)
    first = np.searchsorted(ts, start, side='left')
    lim = np.searchsorted(ts, start + h_us, side='left')
    lim = np.minimum(lim, len(ts) - 1)
    first = np.minimum(first, len(ts) - 1)
    idx = np.empty(2 * len(rows), dtype=np.int64)
    idx[0::2], idx[1::2] = first, np.maximum(lim, first + 0)
    ok = lim > first
    midf = mid.astype(np.float64)
    mx = np.maximum.reduceat(midf, idx)[0::2]
    mn = np.minimum.reduceat(midf, idx)[0::2]
    up = np.where(ok, np.maximum(mx - m0, 0.0), 0.0)
    dn = np.where(ok, np.maximum(m0 - mn, 0.0), 0.0)
    return up, dn


def big_targets(d: dict, rows: np.ndarray, y100: np.ndarray) -> dict[str, np.ndarray]:
    ts = d['ts_us']
    mid = mid_ticks(d, ffill=True)
    start = ts[rows] + LATENCY_US
    m0 = value_before(ts, mid, start)
    out: dict[str, np.ndarray] = {}
    out['D1'] = np.where(y100 >= THR, 1.0, np.where(y100 <= -THR, -1.0, 0.0))
    y300 = value_before(ts, mid, start + 300_000) - m0
    y300[start + 300_000 > ts[-1]] = np.nan
    out['y300'] = y300
    out['D2'] = np.where(y300 >= THR, 1.0, np.where(y300 <= -THR, -1.0, 0.0))
    out['D2'][np.isnan(y300)] = np.nan
    T = build_targets(d, rows, y100)
    yB = T['B']
    out['B'] = yB
    out['D3'] = np.where(yB >= THR, 1.0, np.where(yB <= -THR, -1.0, 0.0))
    out['D3'][np.isnan(yB)] = np.nan
    up, dn = excursion(ts, mid, rows, H100)
    out['D4'] = np.where((up >= THR) & (up >= dn), 1.0, np.where((dn >= THR) & (dn > up), -1.0, 0.0))
    return out


def precision(score: np.ndarray, move: np.ndarray, q: float) -> tuple[float, float]:
    """Share of the top-q |score| events whose move >= THR in the score's direction; and the mean signed move."""
    a = np.abs(score)
    cut = np.quantile(a, 1 - q)
    sel = a >= cut
    sgn = np.sign(score[sel])
    mv = move[sel] * sgn
    return float((mv >= THR).mean()), float(mv.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('logs/ml_bigmove'))
    ap.add_argument('--data', default='data/features/round1.npz')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = load_features(args.data)
    data = build_dataset(d, (100,), max_rows=3_000_000)
    names = [str(x) for x in data['feature_names']]
    X, y, ts, rows = data['X'], data['Y'][:, 0].astype(np.float64), data['ts_us'], data['row']
    T = big_targets(d, rows, y)
    ok = np.ones(len(y), bool)
    for k in ('D1', 'D2', 'D3', 'D4', 'B'):
        ok &= np.isfinite(T[k])
    cut = d['ts_us'][0] + 0.75 * (d['ts_us'][-1] - d['ts_us'][0])
    tr = ok & (ts <= cut - 1_200_000)
    te = ok & (ts > cut)
    big = np.abs(y) >= THR
    print('rows', len(y), 'ok', ok.sum(), 'share |y100|>=0.5: %.3f' % big[ok].mean(),
          'D1 nonzero %.3f D2 %.3f D3 %.3f D4 %.3f' % tuple((T[k][ok] != 0).mean() for k in ('D1', 'D2', 'D3', 'D4')))
    cands = {'base': y, 'B': T['B'], 'D1': T['D1'], 'D2': T['D2'], 'D3': T['D3'], 'D4': T['D4']}
    rows_out = []
    print('%-5s %8s %8s %8s | prec top5/10/20 (100ms move >= .5 in score dir) | mean signed move top5/10/20' % ('cand', 'ic_own', 'ic_y100', 'ic_D1'))
    for k, tgt in cands.items():
        m = fit_ridge(X[tr], tgt[tr], names, ALPHA)
        p = m.predict(X[te])
        pr = [precision(p, y[te], q) for q in (0.05, 0.10, 0.20)]
        ic = (pearson(p, tgt[te]), pearson(p, y[te]), pearson(p, T['D1'][te]))
        print('%-5s %+8.4f %+8.4f %+8.4f | %.3f %.3f %.3f | %.3f %.3f %.3f' % (k, *ic, *[a for a, _ in pr], *[b for _, b in pr]))
        rows_out.append({'cand': k, 'ic': ic, 'prec': [a for a, _ in pr], 'mean': [b for _, b in pr]})
        mf = fit_ridge(X[ok], tgt[ok], names, ALPHA)
        mf.horizon_ms = 100
        mf.meta = {'source': 'data\round1.csv', 'rows': int(ok.sum())}
        pf = mf.predict(X)
        np.save(args.out / f'pred_{k}.npy', pf.astype(np.float32))
        qs = {str(q): float(np.quantile(np.abs(pf[ok]), 1 - q)) for q in (0.10, 0.19, 0.28)}
        with open(args.out / f'model_{k}.json', 'w') as f:
            json.dump({'model': export_linear(mf, print_it=False), 'thr_by_rate': qs, 'pred_std': float(pf.std())}, f)
        print('   full-fit |pred| thresholds at firing rate 10/19/28 %%:', qs)
    # base rate of an event followed by a >= 0.5 tick move either way, and one-sided
    print('base rate |y100|>=0.5 on holdout: %.3f' % (np.abs(y[te]) >= THR).mean())
    with open(args.out / 'table.json', 'w') as f:
        json.dump(rows_out, f)


if __name__ == '__main__':
    main()
