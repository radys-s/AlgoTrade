"""Queue-keep check (Gemini Q1 #1/#6): every cancel of a resting passive quote is logged with the market quantity
ahead of it, the prediction and the price. Counterfactual: had we kept it, would it fill (market trades through its
price within w ms) and what would that fill mark out at 1 s / 5 s?
usage: python -m research.diag.cancel_probe STRATEGY TAPE NPZ"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from research.run import load_strategy_class, simulation_config
from simulator.simulator import ExchangeSimulator


def make(cls):
    class P(cls):
        def __init__(self) -> None:
            super().__init__(); self._c = []; self._now = 0

        def on_order_book_change(self, snapshot, trades):
            self._now = snapshot.timestamp_us
            return super().on_order_book_change(snapshot, trades)

        def cancel_order(self, order_id):
            o = self.orders.get(order_id)
            if o is not None and not o.ioc:
                self._c.append((self._now, o.sign, o.price, o.remaining, self._ahead(o), self.pred_sig, self.inv))
            return super().cancel_order(order_id)
    return P


def main(strategy, tape, npz):
    from pathlib import Path
    st = make(load_strategy_class(strategy))()
    res = ExchangeSimulator(strategy=st, config=simulation_config()).run(Path(tape))
    c = pd.DataFrame(st._c, columns=['t', 'sign', 'px', 'qty', 'ahead', 'pred', 'inv'])
    c.to_csv(npz.replace('.npz', '_cancels.csv'), index=False)
    z = np.load(npz); t = z['t']; mid = z['mid2'] / 2.0
    n = len(t); i0 = np.searchsorted(t, c.t.values, side='right')
    W = 1_000_000
    rows = []
    for (tt, sg, px, q, ah, pr, inv), a in zip(c.itertuples(index=False), i0):
        b = np.searchsorted(t, tt + W, side='right')
        seg = mid[a:b]
        # touch fill proxy: mid reaches px + 0.5 (buy) i.e. our price is the touch with a 1-tick spread
        hit = np.nonzero(sg * (seg - px) <= 0.5)[0]
        thr = np.nonzero(sg * (seg - px) <= -0.5)[0]   # traded through: certain fill
        def mo(k, h):
            j = np.searchsorted(t, t[a + k] + h, side='right') - 1
            return sg * (mid[min(j, n - 1)] - px)
        rows.append((ah, sg * pr, q, len(hit) > 0, mo(hit[0], 1_000_000) if len(hit) else np.nan, mo(hit[0], 5_000_000) if len(hit) else np.nan,
                     len(thr) > 0, mo(thr[0], 5_000_000) if len(thr) else np.nan))
    r = pd.DataFrame(rows, columns=['ahead', 'pred_dir', 'qty', 'hit', 'mo1', 'mo5', 'thru', 'mo5t'])
    print(f'pnl {res.pnl if hasattr(res, "pnl") else ""}  cancels of resting quotes {len(r):,}')
    r['ab'] = pd.cut(r.ahead, [-1, 0, 15, 50, 200, 1 << 31], labels=['0', '1-15', '16-50', '51-200', '>200'])
    r['pb'] = pd.cut(r.pred_dir, [-99, -0.3, -0.1, 0.1, 0.3, 99])
    agg = lambda x: pd.Series({'n': len(x), 'hit%': x.hit.mean(), 'mo1': x.mo1.mean(), 'mo5': x.mo5.mean(), 'thru%': x.thru.mean(), 'mo5_thru': x.mo5t.mean()})
    print('by queue ahead at cancel (counterfactual fill within 1 s, mark-out ticks/lot):')
    print(r.groupby('ab', observed=True).apply(agg).round(3).to_string())
    print('ahead <= 15, by prediction in the quote direction:')
    print(r[r.ahead <= 15].groupby('pb', observed=True).apply(agg).round(3).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:4])
