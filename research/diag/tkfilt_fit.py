"""Diagnostic: fit a filter for the fair taker's entries from tkfilt_record.py output.

    uv run python -X utf8 -m research.diag.tkfilt_fit rec.npz data/features/round2.npz [train_frac]
Labels: robust mid (mid of the last state with spread <= 3) at t + 150 ms / 1 s minus the entry touch price,
signed by the trade direction. Ridge on a few transformed features (the same transform as exp_r2_tkfilt.py),
fitted on the first train_frac of time, scored on the rest: correlation, and the markout of the FILLED lots
kept / skipped at several thresholds.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

FEATS = ['edge', 'sig20', 'sig150', 'spread', 'lq_take', 'lq_opp', 'dt_ins', 'n_ins50', 'tr_sign', 'tr_dt',
         'inv', 'drift', 'cond_age', 'room']


def transform(d: dict) -> np.ndarray:
    """Same transform as the strategy's online gate."""
    return np.column_stack([
        d['edge'], d['sig20'] * d['side'], d['sig150'] * d['side'],
        np.minimum(d['spread'], 4.0), np.log1p(d['q_take']), np.log1p(d['q_opp']),
        np.minimum(d['dt_ins'], 200_000.0) / 1000.0, d['n_ins50'], d['tr_sign'],
        np.minimum(d['tr_dt'], 1_000_000.0) / 1000.0, d['inv'] / 100.0, d['drift'] / 8.0,
        np.minimum(d['cond_age'], 200_000.0) / 1000.0, d['room'] / 100.0])


def robust_mid(npz: Path):
    z = np.load(npz)
    ts = z['ts_us']; b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    idx = np.where(ok)[0]
    rm = (b + a) / 2

    def at(t):
        k = np.searchsorted(ts, t, side='right') - 1
        j = np.searchsorted(idx, np.maximum(k, 0), side='right') - 1
        return rm[idx[np.maximum(j, 0)]]
    return at, ts


def load(rec: Path, npz: Path):
    z = np.load(rec)
    cols = list(z['cols']); data = z['data']
    d = {c: data[:, i] for i, c in enumerate(cols)}
    at, ts = robust_mid(npz)
    for h, name in ((150_000, 'y150'), (1_000_000, 'y1s')):
        fut = at(d['t'] + h)
        d[name] = (fut - d['touch']) * d['side']
        d[name + 'f'] = np.where(d['fill_qty'] > 0, (fut - d['fill_px']) * d['side'], 0.0)
    return d, ts


def ridge(X, y, lam=1.0):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return w / sd, y.mean() - (mu / sd) @ w


def report(d, tr, ho, X, w, b, yname):
    p = X @ w + b
    for part, m in (('train', tr), ('hold', ho)):
        y = d[yname][m]; yf = d[yname + 'f'][m] * d['fill_qty'][m]
        c = np.corrcoef(p[m], y)[0, 1]
        line = f'  {part}: n={m.sum()} corr={c:+.3f} lots={d["fill_qty"][m].sum():.0f} markout={yf.sum():+.0f}'
        print(line)
        qs = np.quantile(p[tr], [0.05, 0.1, 0.2, 0.3, 0.4])
        for q, thr in zip((5, 10, 20, 30, 40), qs):
            skip = p[m] < thr
            print(f'    thr=q{q:02d} ({thr:+.3f}): skip {skip.mean():.0%} of entries, {d["fill_qty"][m][skip].sum():.0f} lots, '
                  f'skipped markout {yf[skip].sum():+.0f} (mean label {y[skip].mean() if skip.any() else 0:+.3f} vs '
                  f'kept {y[~skip].mean():+.3f})')
    return p


def main() -> None:
    rec, npz = Path(sys.argv[1]), Path(sys.argv[2])
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
    d, ts = load(rec, npz)
    t0, t1 = ts[0], ts[-1]
    cut = t0 + frac * (t1 - t0)
    tr = d['t'] < cut; ho = ~tr
    X = transform(d)
    print(f'records {len(X)} (train {tr.sum()}, hold {ho.sum()}); filled {np.mean(d["fill_qty"] > 0):.0%}; '
          f'mean y150 {d["y150"].mean():+.3f}, y1s {d["y1s"].mean():+.3f}')
    for i, f in enumerate(FEATS):
        print(f'  {f:9s} corr y150 {np.corrcoef(X[:, i], d["y150"])[0, 1]:+.3f}  y1s {np.corrcoef(X[:, i], d["y1s"])[0, 1]:+.3f}'
              f'  mean {X[:, i].mean():+.3f}')
    for yname in ('y150', 'y1s'):
        w, b = ridge(X[tr], d[yname][tr], lam=10.0)
        print(f'ridge on {yname} (fit first {frac:.0%}): w={np.round(w, 8).tolist()} b={b:+.8f}')
        report(d, tr, ho, X, w, b, yname)
        wa, ba_ = ridge(X, d[yname], lam=10.0)
        print(f'  fit-100%: w={np.round(wa, 8).tolist()} b={ba_:+.8f}')


if __name__ == '__main__':
    main()
