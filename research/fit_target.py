"""H18: ridge on the champion's 23 features with a target aligned to the shrink decision.

    uv run python -X utf8 -m research.fit_target [--out logs/ml_target]

Per event i (order sent at t lands at t+5 ms): hit_b / hit_a = a seller / buyer taker trades within
the next 100 ms; the move after a hit = mid(th + h) - mid(th) with th the hit event's time (mid before
the hit is applied: the book our resting order faced). Candidates:
  base: plain 100 ms mid move         A_k: same target, weight 1 + k*[any hit]
  B:    move after the first hit x [hit]      C: +1 ask / -1 bid adverse-fill indicator (>= 0.5 tick in 1 s)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.features import load_features, mid_ticks
from research.signals import LATENCY_US, build_dataset, export_linear, fit_ridge, pearson, value_before

H_US = 100_000
ADV_US = 1_000_000
ADV_TICKS = 0.5
ALPHA = 10.0


def hit_targets(d: dict, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Per tape row in `rows`: hit flags in the next 100 ms (after latency) and the post-hit moves."""
    ts = d['ts_us']
    mid = mid_ticks(d, ffill=True)
    tv = d['trade_vol']
    start = ts[rows] + LATENCY_US
    first = np.searchsorted(ts, start, side='left')
    lim = np.searchsorted(ts, start + H_US, side='left')  # events with ts < start + 100 ms
    out: dict[str, np.ndarray] = {}
    for name, idx in (('b', np.flatnonzero(tv < 0)), ('a', np.flatnonzero(tv > 0))):  # b: seller hits our bid
        k = np.searchsorted(idx, first, side='left')
        j = idx[np.minimum(k, len(idx) - 1)]
        hit = (k < len(idx)) & (j < lim)
        th = ts[j]
        m0 = value_before(ts, mid, th)
        mv = value_before(ts, mid, th + H_US) - m0
        mv[th + H_US > ts[-1]] = np.nan
        adv = value_before(ts, mid, th + ADV_US) - m0
        adv[th + ADV_US > ts[-1]] = np.nan
        out['hit_' + name] = hit
        out['t_' + name] = np.where(hit, th, np.iinfo(np.int64).max)
        out['mv_' + name] = np.where(hit, mv, 0.0)
        out['adv_' + name] = np.where(hit, adv, 0.0)
    return out


def build_targets(d: dict, rows: np.ndarray, y_plain: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-candidate target arrays (NaN = unknown, drop) and the any-hit flag."""
    h = hit_targets(d, rows)
    anyhit = h['hit_b'] | h['hit_a']
    first_is_b = h['hit_b'] & (~h['hit_a'] | (h['t_b'] <= h['t_a']))
    mv_first = np.where(first_is_b, h['mv_b'], h['mv_a'])
    yB = np.where(anyhit, mv_first, 0.0)
    cb = (h['hit_b'] & (h['adv_b'] <= -ADV_TICKS)).astype(float)
    ca = (h['hit_a'] & (h['adv_a'] >= ADV_TICKS)).astype(float)
    yC = ca - cb
    unknown = np.isnan(yB) | np.isnan(h['adv_a']) | np.isnan(h['adv_b'])
    yB[unknown] = np.nan
    yC[unknown] = np.nan
    return {'plain': y_plain, 'B': yB, 'C': yC, 'anyhit': anyhit}


def fit_weighted(X: np.ndarray, y: np.ndarray, w: np.ndarray, names: list[str], alpha: float = ALPHA):  # noqa: ANN201
    """Ridge with row weights; clip bounds / standardisation as fit_ridge, alpha scaled by the mean weight."""
    step = max(1, len(X) // 500_000)
    base = fit_ridge(X[::step], y[::step], names, alpha)
    lo, hi, mean, scale = base.lo, base.hi, base.mean, base.scale
    p = X.shape[1]
    sw = 0.0
    szz = np.zeros((p, p))
    sz = np.zeros(p)
    szy = np.zeros(p)
    sy = 0.0
    for s in range(0, len(X), 500_000):
        z = (np.clip(X[s : s + 500_000].astype(np.float64), lo, hi) - mean) / scale
        wb, yb = w[s : s + 500_000], y[s : s + 500_000]
        zw = z * wb[:, None]
        szz += zw.T @ z
        sz += zw.sum(0)
        szy += zw.T @ yb
        sy += float((wb * yb).sum())
        sw += float(wb.sum())
    zm, ym = sz / sw, sy / sw
    cov = szz - sw * np.outer(zm, zm)
    coef = np.linalg.solve(cov + alpha * (sw / len(X)) * np.eye(p), szy - sw * zm * ym)
    base.coef = coef
    base.intercept = float(ym - zm @ coef)
    return base


def main() -> None:
    """Fit all candidates on the first 75 % of round1 and print the IC table."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('logs/ml_target'))
    ap.add_argument('--data', default='data/features/round1.npz')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = load_features(args.data)
    data = build_dataset(d, (100,), max_rows=3_000_000)
    names = [str(x) for x in data['feature_names']]
    X, y, ts, rows = data['X'], data['Y'][:, 0].astype(np.float64), data['ts_us'], data['row']
    T = build_targets(d, rows, y)
    anyhit = T['anyhit']
    ok = np.isfinite(T['B']) & np.isfinite(T['C'])
    print('rows', len(y), 'ok', ok.sum(), 'hit share %.3f' % anyhit[ok].mean(),
          'adverse bid %.4f ask %.4f' % ((T['C'][ok] < 0).mean(), (T['C'][ok] > 0).mean()))
    cut = d['ts_us'][0] + 0.75 * (d['ts_us'][-1] - d['ts_us'][0])
    tr = ok & (ts <= cut - 1_200_000)
    te = ok & (ts > cut)
    cands = {'base': (y, None), 'A3': (y, 1 + 3 * anyhit), 'A10': (y, 1 + 10 * anyhit), 'B': (T['B'], None), 'C': (T['C'], None)}
    ev = {}
    for k, (tgt, w) in cands.items():
        m = fit_ridge(X[tr], tgt[tr], names, ALPHA) if w is None else fit_weighted(X[tr], tgt[tr], w[tr].astype(np.float64), names)
        for nm, sel in (('train', tr), ('hold', te)):
            p = m.predict(X[sel])
            hs = anyhit[sel]
            r = {
                'ic_own': pearson(p, tgt[sel]),
                'ic_plain': pearson(p, y[sel]),
                'ic_plain_hit': pearson(p[hs], y[sel][hs]),
                'ic_B': pearson(p, T['B'][sel]),
                'ic_C': pearson(p, T['C'][sel]),
            }
            r['r2_hit'] = r['ic_plain_hit'] ** 2
            ev[(k, nm)] = r
        mf = fit_ridge(X[ok], tgt[ok], names, ALPHA) if w is None else fit_weighted(
            X[ok], tgt[ok], w[ok].astype(np.float64), names)
        mf.horizon_ms = 100
        mf.meta = {'source': 'data\\round1.csv', 'rows': int(ok.sum())}
        pf = mf.predict(X)
        np.save(args.out / f'pred_{k}.npy', pf.astype(np.float32))
        with open(args.out / f'model_{k}.json', 'w') as f:
            json.dump({'model': export_linear(mf, print_it=False), 'abs_q81': float(np.quantile(np.abs(pf), 0.81)),
                       'pred_std': float(pf.std())}, f)
    print('%-5s %-5s %8s %8s %12s %7s %7s %7s' % ('cand', 'set', 'ic_own', 'ic_plain', 'ic_plain|hit', 'r2|hit', 'ic_B', 'ic_C'))
    for (k, nm), r in ev.items():
        print('%-5s %-5s %+8.4f %+8.4f %+12.4f %7.4f %+7.4f %+7.4f' % (k, nm, r['ic_own'], r['ic_plain'], r['ic_plain_hit'], r['r2_hit'], r['ic_B'], r['ic_C']))
    for k in cands:
        with open(args.out / f'model_{k}.json') as f:
            print(k, 'abs_q81 (full fit)', json.load(f)['abs_q81'])


if __name__ == '__main__':
    main()
