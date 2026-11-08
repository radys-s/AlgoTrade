"""Ensemble / spread-split ridge fits for exp_r3_ens (same recipe as pred_rfit_r3: lean 22 feats, alpha 10, stride 3, r3+mirror first 75 %).
    .venv/Scripts/python.exe -X utf8 -m research.diag.pred_ens_r3 > out.txt
Prints folded (c0, weights): h150/h200/h250 fits, their mean (avg3), the 150+250 mean (avg2), and spread==1 / spread>=2 fits at 200 ms."""
from __future__ import annotations
import numpy as np
from research.signals import fit_ridge
from research.diag import pred_r3
from research.diag.pred_rfit_r3 import cols_for, rows


def fold(m):
    wr = m.coef / m.scale
    return float(m.intercept - (wr * m.mean).sum()), wr


def fmt(name, c0, w):
    return f'{name} = ({round(c0, 7)}, {tuple(round(float(x), 7) for x in w)})'


if __name__ == '__main__':
    feats = list(pred_r3.LEAN)
    data = {n: cols_for(n) for n in ('round3', 'round3_mirror')}
    si = feats.index('spread')
    fits = {}
    for H in (150, 200, 250):
        parts = [rows(data[n], H, .75, feats) for n in data]
        X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        fits[H] = fold(fit_ridge(X, y, feats, alpha=10.0, clip_q=0.0))
        print(fmt(f'h{H}', *fits[H]), flush=True)
        if H == 200:
            for nm, sel in (('sp1', X[:, si] <= 1.5), ('sp2p', X[:, si] > 1.5)):
                print('#', nm, int(sel.sum()), flush=True)
                print(fmt(f'h200_{nm}', *fold(fit_ridge(X[sel], y[sel], feats, alpha=10.0, clip_q=0.0))), flush=True)
    for nm, hs in (('avg3', (150, 200, 250)), ('avg2', (150, 250))):
        c0 = float(np.mean([fits[h][0] for h in hs])); w = np.mean([fits[h][1] for h in hs], axis=0)
        print(fmt(nm, c0, w), flush=True)
