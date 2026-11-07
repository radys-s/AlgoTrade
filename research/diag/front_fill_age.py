"""Diagnostic: markout of the FRONT-of-queue fill at the touch, by how long that price has been the touch and how
the level became the touch (retreat onto an existing level vs a fresh improving level).

    uv run python -X utf8 -m research.diag.front_fill_age logs/<run>/trades.csv data/features/round2.npz
(trades.csv from any --full-logs run: it lists every market-vs-market trade with taker/maker ids.)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd


def main() -> None:
    tr = pd.read_csv(sys.argv[1], dtype={'taker_order_id': str, 'maker_order_id': str}, low_memory=False)
    tr = tr[(tr.taker_owner == 'market') & (tr.maker_owner == 'market')]
    first = tr.groupby('taker_order_id', sort=False).head(1)
    d = np.load(sys.argv[2])
    t = d['ts_us']; b = d['bid_px'].astype(np.int64); a = d['ask_px'].astype(np.int64)
    ok = (b > 0) & (a > 0) & ((a - b) <= 3); idx = np.where(ok)[0]; rm = (b + a) / 2

    def rmid(T):
        k = np.clip(np.searchsorted(t, T, side='right') - 1, 0, len(t) - 1)
        j = np.clip(np.searchsorted(idx, k, side='right') - 1, 0, len(idx) - 1)
        return rm[idx[j]]
    # touch-change bookkeeping per side: time the current best price became best, and how (retreat/improve)
    out = []
    for side_name, px_arr, better in (('bid', b, 1), ('ask', a, -1)):
        chg = np.r_[True, px_arr[1:] != px_arr[:-1]]
        ci = np.where(chg)[0]
        born_t = t[ci]
        prev = np.r_[0, px_arr[ci[:-1]]]
        how = np.where((px_arr[ci] - prev) * better > 0, 1, -1)  # 1 = improved (fresh), -1 = retreated onto existing
        # taker sold -> hits bids (taker_side 'A'); taker bought -> hits asks
        sel = first[first.taker_side == ('A' if side_name == 'bid' else 'B')]
        ts = sel.timestamp_us.values; px = sel.price.values
        # index of the touch state just before the trade
        k = np.searchsorted(t, ts, side='left') - 1
        j = np.searchsorted(ci, k, side='right') - 1
        valid = (j >= 0) & (px_arr[np.clip(k, 0, None)] == px)
        age = (ts - born_t[np.clip(j, 0, None)]) / 1e6
        hw = how[np.clip(j, 0, None)]
        maker_sign = 1 if side_name == 'bid' else -1  # maker bought on bid
        for h in (1.0, 10.0):
            mk = maker_sign * (rmid(ts + int(h * 1e6)) - px)
            out.append((side_name, h, valid, age, hw, mk, sel.quantity.values))
    for h in (1.0, 10.0):
        rows = [o for o in out if o[1] == h]
        valid = np.concatenate([o[2] for o in rows]); age = np.concatenate([o[3] for o in rows])
        hw = np.concatenate([o[4] for o in rows]); mk = np.concatenate([o[5] for o in rows])
        print(f'== markout {h:.0f} s of the front-of-queue fill (ticks/lot, maker view), n={valid.sum():,}')
        for lab, m in (('improved (fresh level)', hw == 1), ('retreated onto existing', hw == -1)):
            for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1), (1, 3), (3, 1e9)):
                s = valid & m & (age >= lo) & (age < hi)
                if s.sum() > 50:
                    print(f'  {lab:26s} touch age {lo:>4}-{hi:<5}s: n {s.sum():6,} markout {mk[s].mean():+.3f} (se {mk[s].std() / np.sqrt(s.sum()):.3f})')


if __name__ == '__main__':
    main()
