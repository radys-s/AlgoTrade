"""Train MLP taker predictors on mlp_dump npz files and bake the best into a copy of s12.

    uv run python -m research.diag.mlp_fit --npz a.npz b.npz --train-frac 0.75 --hidden 64,32 --horizon 200 \
        [--extra] [--bake strategies/x.py] [--eval-only]

Target: robust mid at t+5ms+horizon minus mid now (ticks), clipped at +-6. Split by time inside every file (first
train-frac trains, the rest is holdout); any npz given with --test-npz is pure out-of-sample. Reported: Pearson
corr on the holdout, on all rows and on the gated rows (|own ridge| >= 0.3, a proxy for the s12 MLP gate),
for the new model and for the MLP baked in s12.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

S12 = Path('strategies/honza_r3_s12.py')


def load(path: str, horizon_ms: int, nf: int):
    d = np.load(path)
    X, t, mid2, rm2 = d['X'][:, :nf], d['t'], d['mid2'], d['rm2']
    j = np.searchsorted(t, t + 5000 + horizon_ms * 1000, side='right') - 1
    y = np.clip(rm2[j] / 2.0 - mid2 / 2.0, -6.0, 6.0).astype(np.float32)
    ok = (j < len(t) - 1) & np.isfinite(X).all(1)
    return X[ok], y[ok]


def split(X, y, frac):
    n = int(len(y) * frac)
    return (X[:n], y[:n]), (X[n:], y[n:])


def s12_mlp():
    ns: dict = {}
    src = S12.read_text(encoding='utf-8')
    block = src[src.index('MLP_MEAN ='):src.index(']\n', src.index('MLP_LAYERS =')) + 2]
    exec(block, ns)
    mean, scale = np.array(ns['MLP_MEAN']), np.array(ns['MLP_SCALE'])
    layers = [(np.array(w), np.array(b)) for w, b in ns['MLP_LAYERS']]
    return mean, scale, layers


def forward(X, mean, scale, layers):
    h = np.clip((X - mean) / scale, -8.0, 8.0)
    for i, (w, b) in enumerate(layers):
        h = h @ w + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h[:, 0]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else float('nan')


def tail(pred, y):
    # mean signed realized move in the top 1 % / 3 % / 10 % of |pred| (what triggers takes)
    out = []
    ap = np.abs(pred)
    for q in (0.99, 0.97, 0.90):
        m = ap >= np.quantile(ap, q)
        out.append(f'{float(np.mean(np.sign(pred[m]) * y[m])):.3f}')
    return '/'.join(out)


def literal(mean, scale, layers) -> str:
    r = lambda v: round(float(v), 6)
    out = [f'MLP_MEAN = {[r(v) for v in mean]}', f'MLP_SCALE = {[r(v) for v in scale]}', 'MLP_LAYERS = [']
    for w, b in layers:
        out.append(f'    ({[[r(v) for v in row] for row in w]}, {[r(v) for v in b]}),')
    out.append(']')
    return '\n'.join(out) + '\n'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--npz', nargs='+', required=True)
    p.add_argument('--test-npz', nargs='*', default=[])
    p.add_argument('--train-frac', type=float, default=0.75)
    p.add_argument('--hidden', default='32,16')
    p.add_argument('--horizon', type=int, default=200)
    p.add_argument('--extra', action='store_true', help='use all 26 columns (needs a strategy that computes them)')
    p.add_argument('--stride', type=int, default=1, help='subsample training rows')
    p.add_argument('--alpha', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bake', default='')
    p.add_argument('--eval-horizons', default='', help='also report corr / tail edge vs these target horizons (ms)')
    p.add_argument('--save', default='', help='save mean/scale/layers to this npz-like pickle')
    a = p.parse_args()
    nf = 26 if a.extra else 22
    tr, ho, te = [], [], []
    for f in a.npz:
        X, y = load(f, a.horizon, nf)
        (x1, y1), (x2, y2) = split(X, y, a.train_frac)
        tr.append((x1, y1))
        ho.append((Path(f).stem, x2, y2))
    for f in a.test_npz:
        X, y = load(f, a.horizon, nf)
        te.append((Path(f).stem + '*', X, y))
    Xtr = np.concatenate([x for x, _ in tr])[:: a.stride]
    ytr = np.concatenate([y for _, y in tr])[:: a.stride]
    mean = Xtr.mean(0)
    scale = Xtr.std(0) + 1e-6
    Z = np.clip((Xtr - mean) / scale, -8, 8)
    # own ridge as the gate proxy
    A = np.c_[Z, np.ones(len(Z))]
    wr = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ ytr)
    t0 = time.time()
    hidden = tuple(int(v) for v in a.hidden.split(','))
    m = MLPRegressor(hidden_layer_sizes=hidden, alpha=a.alpha, batch_size=4096, learning_rate_init=1e-3,
                     max_iter=a.epochs, early_stopping=True, validation_fraction=0.1, n_iter_no_change=4,
                     random_state=a.seed)
    m.fit(Z, ytr)
    layers = list(zip(m.coefs_, m.intercepts_))
    fit_s = time.time() - t0
    base = s12_mlp() if nf == 22 else None
    print(f'h{a.horizon} hidden {hidden} extra {a.extra} rows {len(ytr):,} fit {fit_s:.0f}s epochs {m.n_iter_}')
    for name, X, y in ho + te:
        Zh = np.clip((X - mean) / scale, -8, 8)
        g = np.abs(np.c_[Zh, np.ones(len(Zh))] @ wr) >= 0.3
        pn = forward(X, mean, scale, layers)
        line = f'  {name:16s} n {len(y):>9,} gated {g.mean():.2f} | new all {corr(pn, y):.4f} gated {corr(pn[g], y[g]):.4f}'
        if base is not None:
            pb = forward(X, *base)
            line += f' | s12 all {corr(pb, y):.4f} gated {corr(pb[g], y[g]):.4f}'
        print(line, flush=True)
    for eh in [int(v) for v in a.eval_horizons.split(',') if v]:
        for f in a.npz + a.test_npz:
            X, y = load(f, eh, nf)
            if f in a.npz:
                X, y = X[int(len(y) * a.train_frac):], y[int(len(y) * a.train_frac):]
            pn = forward(X, mean, scale, layers)
            line = f'  eval h{eh:<5d} {Path(f).stem:14s} new corr {corr(pn, y):.4f} tail {tail(pn, y)}'
            if base is not None:
                pb = forward(X, *base)
                line += f' | s12 corr {corr(pb, y):.4f} tail {tail(pb, y)}'
            print(line, flush=True)
    if a.save:
        import pickle
        Path(a.save).write_bytes(pickle.dumps((mean, scale, layers)))
    if a.bake:
        src = S12.read_text(encoding='utf-8')
        s = src.index('MLP_MEAN =')
        e = src.index(']\n', src.index('MLP_LAYERS =')) + 2
        Path(a.bake).write_text(src[:s] + literal(mean, scale, layers) + src[e:], encoding='utf-8')
        print('baked ->', a.bake)


if __name__ == '__main__':
    main()
