"""Early-move probe: on QUIET rows (|mid move over the last 100 ms| <= 0.5 tick) compare the holdout corr with the
future move (200 ms and 1 s) of (a) the standard h200 MLP trained on all rows vs (b) an MLP trained on quiet rows
with the 1 s target. usage: python research/diag/early_probe.py NPZ [NPZ...]"""
import sys
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def rows(path):
    d = np.load(path); X, t, mid2, rm2 = d['X'][:, :26], d['t'], d['mid2'], d['rm2']
    m = rm2 / 2.0
    jp = np.searchsorted(t, t - 100000, side='right') - 1
    f = lambda h: np.clip(m[np.searchsorted(t, t + 5000 + h * 1000, side='right') - 1] - mid2 / 2.0, -6, 6)
    past = mid2 / 2.0 - mid2[np.maximum(jp, 0)] / 2.0
    ok = np.isfinite(X).all(1) & (t + 1_100_000 < t[-1])
    return X[ok], f(200)[ok], f(1000)[ok], (np.abs(past) <= 0.5)[ok]
parts = [rows(p) for p in sys.argv[1:]]
def fit(Xs, ys, seed=0):
    sc = StandardScaler().fit(Xs); m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-4, max_iter=15, random_state=seed, early_stopping=False)
    m.fit(np.clip(sc.transform(Xs), -8, 8), ys); return lambda X: m.predict(np.clip(sc.transform(X), -8, 8))
tr = [(X[:int(.75*len(X)):2], y2[:int(.75*len(X)):2], y1[:int(.75*len(X)):2], q[:int(.75*len(X)):2]) for X, y2, y1, q in parts]
Xa = np.vstack([a[0] for a in tr]); y2a = np.concatenate([a[1] for a in tr]); y1a = np.concatenate([a[2] for a in tr]); qa = np.concatenate([a[3] for a in tr])
print('quiet share train', qa.mean().round(3))
A = fit(Xa, y2a)
B = fit(Xa[qa], y1a[qa])
C = fit(Xa, y1a)
for p, (X, y2, y1, q) in zip(sys.argv[1:], parts):
    c = int(.75 * len(X)); Xh, y2h, y1h, qh = X[c:][q[c:]], y2[c:][q[c:]], y1[c:][q[c:]], q[c:]
    pa, pb, pc = A(Xh), B(Xh), C(Xh)
    r = lambda a, b: np.corrcoef(a, b)[0, 1]
    print(p.split('/')[-1], f'quiet rows {len(Xh):,} | vs 200ms: A {r(pa,y2h):.4f} B {r(pb,y2h):.4f} C {r(pc,y2h):.4f} | vs 1s: A {r(pa,y1h):.4f} B {r(pb,y1h):.4f} C {r(pc,y1h):.4f} | A+B {r(pa/pa.std()+pb/pb.std(),y2h):.4f}')
