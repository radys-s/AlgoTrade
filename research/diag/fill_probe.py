"""Where does s12 make and lose money? Record every own fill with the predictor state, then mark it out.

    uv run python -m research.diag.fill_probe strategies/honza_r3_s12.py data/round3_public.csv ../mlpdata/round3_public.npz

Per fill: time, sign (+1 buy), qty, price, ioc (taker) flag, pred_sig and fair_sig at fill time, inventory before,
spread. Mark-out per lot = sign * (mid at t+h - price) for h in 100 ms, 1 s, 5 s, 30 s (mid from the mlp_dump npz).
Prints the mark-out by (maker/taker) x (prediction in the fill direction) buckets and by inventory buckets,
plus the share of time spent at |inv| >= 95.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.run import load_strategy_class, simulation_config
from simulator.simulator import ExchangeSimulator

H = {'100ms': 100_000, '1s': 1_000_000, '5s': 5_000_000, '30s': 30_000_000}


def make_probe(cls):
    class Probe(cls):
        def __init__(self) -> None:
            super().__init__()
            self._fills = []
            self._inv = []

        def on_private_order_update(self, event):
            order = self.orders.get(event.order_id)
            if order is not None and event.filled_quantity and event.fill_price is not None:
                self._fills.append((event.timestamp_us, order.sign, event.filled_quantity, event.fill_price,
                                    int(bool(order.ioc)), self.pred_sig, self.fair_sig, self.inv, order.k))
            super().on_private_order_update(event)
            self._inv.append((event.timestamp_us, self.inv))

    return Probe


def main(strategy: str, data: str, npz: str) -> None:
    st = make_probe(load_strategy_class(strategy))()
    res = ExchangeSimulator(strategy=st, config=simulation_config()).run(Path(data))
    f = pd.DataFrame(st._fills, columns=['t', 'sign', 'qty', 'px', 'ioc', 'pred', 'fair', 'inv0', 'k'])
    f.to_csv(npz.replace('.npz', '_fills.csv'), index=False)
    d = np.load(npz)
    t, mid = d['t'], d['mid2'] / 2.0
    for name, h in H.items():
        j = np.searchsorted(t, f.t.to_numpy() + h, side='right') - 1
        f[name] = f.sign * (mid[j] - f.px)
    f['pdir'] = f.sign * f.pred  # prediction in the fill direction: > 0 = the model agreed with the trade
    f['fdir'] = f.sign * f.fair
    out = Path(npz).with_name(Path(data).stem + '_fills.csv')
    f.to_csv(out, index=False)
    print(f'{data}: pnl {res.pnl:,.0f}  fills {len(f):,}  lots {f.qty.sum():,}  -> {out}')
    w = lambda g: pd.Series({'fills': len(g), 'lots': g.qty.sum(),
                             **{h: np.average(g[h], weights=g.qty) for h in H},
                             'pnl30s': (g['30s'] * g.qty).sum()})
    print('\nby type:')
    print(f.groupby('ioc').apply(w).round(3).to_string())
    f['pq'] = pd.cut(f.pdir, [-99, -1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0, 99])
    print('\nmaker fills by prediction in fill direction:')
    print(f[f.ioc == 0].groupby('pq', observed=True).apply(w).round(3).to_string())
    print('\ntaker fills by prediction in fill direction:')
    print(f[f.ioc == 1].groupby('pq', observed=True).apply(w).round(3).to_string())
    f['iq'] = pd.cut(f.sign * f.inv0, [-101, -90, -50, -10, 10, 50, 90, 101])
    print('\nmaker fills by inventory in fill direction (+ = adds to position):')
    print(f[f.ioc == 0].groupby('iq', observed=True).apply(w).round(3).to_string())
    print('\nmaker fills by layer k:')
    print(f[f.ioc == 0].groupby('k').apply(w).round(3).to_string())
    iv = np.array(st._inv)
    tt, ii = iv[:, 0], np.abs(iv[:, 1])
    dt = np.diff(np.r_[tt, tt[-1]])
    print(f'\ntime share |inv|>=95: {dt[ii >= 95].sum() / dt.sum():.3f}  >=50: {dt[ii >= 50].sum() / dt.sum():.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
