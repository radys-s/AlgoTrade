"""T4 offline check: are reverse-flip takes against the recent drift worse than those with it?
usage: python -m research.diag.flip_drift NPZ (reads NPZ.replace('.npz','_fills.csv') from fill_probe)"""
import sys
import numpy as np, pandas as pd

npz = sys.argv[1]
z = np.load(npz); t = z['t']; mid = z['mid2'] / 2.0
f = pd.read_csv(npz.replace('.npz', '_fills.csv'))
f = f[f.ioc == 1].copy()


def mid_at(tt):
    return mid[np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 1)]


m0 = mid_at(f.t.values)
for h, us in (('1s', 1e6), ('5s', 5e6), ('30s', 30e6)):
    f['mo' + h] = f.sign * (mid_at(f.t.values + us) - f.px)
f['rev'] = (np.sign(f.inv0) == -f.sign) & (f.inv0.abs() >= 50)
tot = (f.qty * f.mo5s).sum()
print(f'taker fills {len(f):,}, lots {f.qty.sum():,}, reverse-flip lots {f[f.rev].qty.sum():,}; taker 5 s markout pnl {tot:,.0f}')
for w in (5, 10, 20):
    f['al'] = f.sign * (m0 - mid_at(f.t.values - w * 1e6))  # >0: flip goes WITH the last w s drift
    g = f[f.rev]
    b = pd.cut(g.al, [-1e9, -8, -4, -2, 0, 2, 4, 8, 1e9])
    r = g.groupby(b, observed=True).apply(lambda x: pd.Series({'lots': x.qty.sum(), 'mo1s': np.average(x.mo1s, weights=x.qty),
        'mo5s': np.average(x.mo5s, weights=x.qty), 'mo30s': np.average(x.mo30s, weights=x.qty), 'pnl5s': (x.qty * x.mo5s).sum()}))
    print(f'\nreverse flips by drift over {w} s (ticks, flip direction):'); print(r.round(2).to_string())
