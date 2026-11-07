"""Diagnostic: markout of our maker fills by the hidden-fair signal (radys, findings s.11) at the fill time.

    uv run python -X utf8 -m research.diag.fair_fills logs/<tag>/<run>/trades.csv data/round2.csv [tau_ms]
sig = time-decayed mean (tau) over new LIMIT inserts of (price - side offset - mid before the insert).
For a maker, a fill is 'against' when it is a buy with sig < 0 (fair below) or a sell with sig > 0.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from research.diag.fill_decomp import robust_mid_fn


def fair_signal(csv: str, npz: str, tau_ms: float):
    df = pd.read_csv(csv)
    z = np.load(npz)
    ts = z['ts_us'].astype(np.int64)
    bb = z['bid_px'].astype(float); ba = z['ask_px'].astype(float)
    bad = (bb <= 0) | (ba <= 0) | (ba <= bb)
    bb = pd.Series(np.where(bad, np.nan, bb)).ffill().bfill().values
    ba = pd.Series(np.where(bad, np.nan, ba)).ffill().bfill().values
    mid = (bb + ba) / 2
    pm = np.r_[mid[0], mid[:-1]]
    px = df.Price.values.astype(float)
    sd = np.where(df.Side.values == 'B', 1, -1)
    isG = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
    off_b = np.mean((px - pm)[isG & (sd > 0)]); off_a = np.mean((px - pm)[isG & (sd < 0)])
    dev = np.where(isG, px - np.where(sd > 0, off_b, off_a) - pm, np.nan)
    idx = np.where(isG)[0]
    dec = np.exp(-np.diff(ts[idx], prepend=ts[idx[0]]) / (tau_ms * 1e3))
    s = w = 0.0
    out = np.empty(len(idx))
    for i, (a, x) in enumerate(zip(dec, dev[idx])):
        s = s * a + x; w = w * a + 1.0; out[i] = s / w
    sig = np.full(len(df), np.nan); sig[idx] = out
    sig = pd.Series(sig).ffill().fillna(0).values
    return ts, sig


def main() -> None:
    tr = pd.read_csv(sys.argv[1], dtype={'taker_order_id': str, 'maker_order_id': str}, low_memory=False)
    csv = sys.argv[2]
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    npz = csv.replace('data/', 'data/features/').replace('.csv', '.npz')
    ts, sig = fair_signal(csv, npz, tau)
    at, _, _ = robust_mid_fn(npz)
    ours = tr[tr.maker_owner == 'user']
    sign = np.where(ours.taker_side == 'B', -1, 1)
    t = ours.timestamp_us.values; px = ours.price.values.astype(float); q = ours.quantity.values
    # signal as known strictly before the fill's event (the order rested earlier; use the state before the row)
    k = np.clip(np.searchsorted(ts, t, side='left') - 1, 0, len(ts) - 1)
    s_for_us = sign * sig[k]  # > 0: fair on our side (good), < 0: fair moving away (bad)
    print(f'fills {len(ours):,} lots {q.sum():,}; tau {tau} ms')
    for h in (0.1, 1.0, 10.0):
        mk = sign * (at(t + int(h * 1e6)) - px)
        rows = []
        for lo, hi in ((-99, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 99)):
            m = (s_for_us >= lo) & (s_for_us < hi)
            if m.sum():
                rows.append(f'[{lo},{hi}): lots {q[m].sum():7,} mk {(mk[m] * q[m]).sum() / q[m].sum():+.2f} pnl {(mk[m] * q[m]).sum():+9,.0f}')
        print(f'== markout {h} s')
        print('   ' + '\n   '.join(rows))


if __name__ == '__main__':
    main()
