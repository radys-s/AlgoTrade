"""Quote-specific markout model for mm_r3_g2 passive fills (exp_r3_qm).

collect: replay exp_r3_qm with qm_rec=1 (flag off otherwise = champion), keep each passive fill's quote-time features
and the markout vs the normal mid 1 s / 5 s later (signed for us, ticks, vs fill price).
fit: ridge on the first 75 % of fills in time (standardised, weights folded back to raw features); prints test corr
and prints qm_c0 / qm_w as JSON.

    python -X utf8 -m research.diag.qmodel_r3 data/round3.csv [alpha] [horizon_ms]
"""
import json, sys
from pathlib import Path
import numpy as np
from research.run import load_strategy_class, with_params, simulation_config
from simulator.simulator import ExchangeSimulator

NAMES = ['edge', 'fs20', 'fs50', 'fs70', 'fs150', 'fs300', 'spread', 'lq_own', 'lq_opp', 'ln_own', 'ln_opp', 'hole',
         'lt_touch', 'drift', 's100', 's1000', 'ev10', 'inv', 'pred']


def collect(data, cache):
    cache = Path(cache)
    if cache.exists():
        return dict(np.load(cache))
    cls = with_params(load_strategy_class('strategies/exp_r3_qm.py'), {'qm_rec': 1})
    st = cls()
    sim = ExchangeSimulator(strategy=st, config=simulation_config(None))
    res = sim.run(Path(data))
    print('pnl', res.pnl, 'fills', len(st.qm_fills))
    mids = np.array(st.qm_mids, dtype=np.float64)
    f = st.qm_fills
    out = dict(t=np.array([x[0] for x in f], dtype=np.float64), k=np.array([x[1] for x in f]),
               px=np.array([x[2] for x in f], dtype=np.float64), q=np.array([x[3] for x in f], dtype=np.float64),
               X=np.array([x[4] for x in f], dtype=np.float64), mt=mids[:, 0], mm=mids[:, 1] / 2.0)
    np.savez(cache, **out)
    return out


def labels(d, h_ms):
    idx = np.searchsorted(d['mt'], d['t'] + h_ms * 1000.0, side='right') - 1
    idx = np.clip(idx, 0, len(d['mm']) - 1)
    s = np.where(d['k'] == 0, 1.0, -1.0)
    return s * (d['mm'][idx] - d['px'])


def fit(d, y, alpha, frac=0.75):
    n = len(y)
    order = np.argsort(d['t'])
    ntr = int(n * frac)
    tr, te = order[:ntr], order[ntr:]
    X = d['X']; w = d['q']
    mu = np.average(X[tr], axis=0, weights=w[tr]); sd = X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    ym = np.average(y[tr], weights=w[tr])
    A = Z[tr].T @ (Z[tr] * w[tr, None]) + alpha * w[tr].sum() * np.eye(X.shape[1]) * 1e-3
    b = Z[tr].T @ (w[tr] * (y[tr] - ym))
    beta = np.linalg.solve(A, b)
    pred = Z @ beta + ym
    raw = beta / sd
    c0 = ym - (mu * raw).sum()
    return pred, tr, te, c0, raw


if __name__ == '__main__':
    data = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    h = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
    d = collect(data, Path(__file__).parent / f'qm_{Path(data).stem}.npz')
    y = labels(d, h)
    pred, tr, te, c0, raw = fit(d, y, alpha)
    print('fills', len(y), 'mean markout', np.average(y, weights=d['q']), 'by side', [y[d['k'] == k].mean() for k in (0, 1)])
    for nm, idx in (('train', tr), ('test', te)):
        print(nm, 'corr', np.corrcoef(pred[idx], y[idx])[0, 1], 'mean y', np.average(y[idx], weights=d['q'][idx]))
    qs = np.quantile(pred[te], [0.1, 0.2, 0.3, 0.5])
    for qv in qs:
        m = pred[te] < qv
        print(f'test pred<{qv:.2f}: share {m.mean():.2f} qty-share {d["q"][te][m].sum()/d["q"][te].sum():.2f} mean y low {np.average(y[te][m], weights=d["q"][te][m]):.3f} rest {np.average(y[te][~m], weights=d["q"][te][~m]):.3f}')
    for nm, v in zip(NAMES, raw): print(f'  {nm:9s} {v:+.4f}')
    print(json.dumps({'qm_c0': float(c0), 'qm_w': [float(v) for v in raw]}))
