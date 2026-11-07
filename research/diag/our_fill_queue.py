"""Diagnostic: markout of OUR maker fills by the lots the same taker order filled ahead of us at our price
(= our queue position when the sweep reached us) and by how the touch level was born.

    uv run python -X utf8 -m research.diag.our_fill_queue logs/<tag>/<run>/trades.csv data/features/round2.npz
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from research.diag.fill_decomp import robust_mid_fn


def main() -> None:
    tr = pd.read_csv(sys.argv[1], dtype={'taker_order_id': str, 'maker_order_id': str}, low_memory=False)
    at, _, _ = robust_mid_fn(sys.argv[2])
    tr['ahead'] = 0
    tr['key'] = tr.taker_order_id + '@' + tr.price.astype(str)
    tr['cum'] = tr.groupby('key').quantity.cumsum() - tr.quantity
    ours = tr[tr.maker_owner == 'user'].copy()
    sign = np.where(ours.taker_side == 'B', -1, 1)  # taker bought -> we sold
    ts = ours.timestamp_us.values; px = ours.price.values.astype(float); q = ours.quantity.values
    ahead = ours.cum.values
    # did this taker order go deeper than our price (depleted our level)?
    last_px = tr.groupby('taker_order_id').price.last()
    deeper = (last_px.reindex(ours.taker_order_id).values != px)
    print(f'our maker fills {len(ours):,}, lots {q.sum():,}')
    for h in (0.1, 1.0, 10.0):
        mk = sign * (at(ts + int(h * 1e6)) - px)
        print(f'== markout {h:.1f} s (ticks/lot, lot-weighted)')
        for lo, hi in ((0, 1), (1, 20), (20, 100), (100, 300), (300, 1000), (1000, 1e9)):
            s = (ahead >= lo) & (ahead < hi)
            if s.sum() > 30:
                print(f'  ahead {lo:>5}-{hi:<6}: fills {s.sum():6,} lots {q[s].sum():8,} markout {(mk[s] * q[s]).sum() / q[s].sum():+.3f}  pnl {(mk[s] * q[s]).sum():+10,.0f}')
        for lab, s in (('sweep went deeper', deeper), ('sweep stopped at our level', ~deeper)):
            print(f'  {lab:28s}: lots {q[s].sum():8,} markout {(mk[s] * q[s]).sum() / q[s].sum():+.3f}  pnl {(mk[s] * q[s]).sum():+10,.0f}')


if __name__ == '__main__':
    main()
