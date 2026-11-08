"""Quick generator fit for the R3 filter: depth histogram of unclamped G inserts vs a non-causal F proxy.

    python -X utf8 -m research.diag.gen_fit_r3 data/round3.csv
F proxy = mid 200 ms later (non-causal). The geometric tail ratio of the depth histogram is robust to proxy noise.
Also IOC limit offset vs proxy and F-proxy step variance per ms.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

csv = Path(sys.argv[1])
df = pd.read_csv(csv, usecols=['Operation', 'Lifespan', 'Side', 'Price'])
z = np.load(Path('data/features') / (csv.stem + '.npz'))
ts = z['ts_us']; b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
pb = np.r_[0, b[:-1]]; pa = np.r_[0, a[:-1]]
ins = df.Operation.values == 'Insert'
G = ins & (df.Lifespan.values == 'G'); F = ins & (df.Lifespan.values == 'F')
side = np.where(df.Side.values == 'B', 1, -1); px = df.Price.values.astype(float)
two = (pb > 0) & (pa > 0) & (b > 0) & (a > 0)
mid = (b + a) / 2
for h in (0, 100, 200, 500):
    k = np.searchsorted(ts, ts + h * 1000, side='right') - 1
    fp = mid[k]
    cl = np.where(side == 1, px >= pa - 1, px <= pb + 1)
    m = G & two & ~cl
    d = np.where(side == 1, np.floor(fp) - px, px - np.ceil(fp))[m]
    vals, cnt = np.unique(d.astype(int), return_counts=True)
    sel = (vals >= 1) & (vals <= 12)
    print(f'h={h}ms unclamped n={m.sum()} clamped frac={(G & two & cl).sum() / (G & two).sum():.3f}')
    print('  d:', dict(zip(vals[sel].tolist(), cnt[sel].tolist())))
    c = dict(zip(vals.tolist(), cnt.tolist()))
    print('  ratios:', [round(c.get(i + 1, 0) / max(c.get(i, 1), 1), 3) for i in range(1, 10)])
    io = (side * (px - fp))[F & two]
    print(f'  IOC limit offset side*(px-F): mean {io.mean():.2f} q10/50/90 {np.percentile(io, [10, 50, 90])}')
# F step variance: variance of mid change over horizon (robust, trimmed) per ms
for h in (50, 200, 1000):
    t0 = np.arange(ts[0], ts[-1] - h * 1000, 20000)
    i0 = np.searchsorted(ts, t0, side='right') - 1; i1 = np.searchsorted(ts, t0 + h * 1000, side='right') - 1
    dm = mid[i1] - mid[i0]; ok = two[i0] & two[i1]
    dm = dm[ok]; q = np.abs(dm) < 10
    print(f'mid var over {h} ms: {dm[q].var():.3f} -> per ms {dm[q].var() / h:.5f}; jump frac |dm|>=10 {(~q).mean():.5f}')
