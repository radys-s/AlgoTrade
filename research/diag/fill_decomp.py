"""Diagnostic: split a run's PnL into spread capture at the fill and inventory (trend) PnL, plus fill markouts.

    uv run python -X utf8 -m research.diag.fill_decomp logs/<tag>/<run>/trades.csv data/features/round2.npz
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd


def robust_mid_fn(npz: str):
    d = np.load(npz)
    b = d['bid_px'].astype(float); a = d['ask_px'].astype(float); t = d['ts_us']
    ok = (b > 0) & (a > 0) & ((a - b) <= 3)
    idx = np.where(ok)[0]; rm = (b + a) / 2

    def at(T):
        k = np.searchsorted(t, T, side='right') - 1
        j = np.clip(np.searchsorted(idx, k, side='right') - 1, 0, len(idx) - 1)
        return rm[idx[j]]
    return at, t[0], t[-1]


def main() -> None:
    tr = pd.read_csv(sys.argv[1])
    at, t0, t1 = robust_mid_fn(sys.argv[2])
    mk = tr[(tr.maker_owner == 'user') | (tr.taker_owner == 'user')].copy()
    # our side: maker on bid side = we bought. taker_side = side of the taker ('A' = taker sold? check)
    us_maker = mk.maker_owner == 'user'
    # taker_side 'B' = taker bought (lifted the ask) -> maker sold
    taker_buys = mk.taker_side == 'B'
    sign = np.where(us_maker, np.where(taker_buys, -1, 1), np.where(taker_buys, 1, -1))
    mk['sign'] = sign
    ts = mk.timestamp_us.values; px = mk.price.values.astype(float); q = mk.quantity.values
    m0 = at(ts)
    print(f'fills {len(mk):,} lots {q.sum():,}  maker share {us_maker.mean():.3f}')
    spread_pnl = (sign * (m0 - px) * q).sum()
    print(f'spread capture vs robust mid at fill: {spread_pnl:+,.0f}  ({spread_pnl / q.sum():+.3f}/lot)')
    for h in [0.1, 1, 10, 60, 300]:
        mh = at(ts + int(h * 1e6))
        e = sign * (mh - px)
        print(f'  markout {h:>5}s: {(e * q).sum():+,.0f}  ({(e * q).sum() / q.sum():+.3f}/lot)')
    # inventory path on a 1 s grid
    grid = np.arange(t0, t1, 1_000_000)
    inv_changes = pd.Series(sign * q, index=ts).groupby(level=0).sum()
    cum = inv_changes.cumsum()
    inv_grid = np.r_[0, cum.values][np.searchsorted(cum.index.values, grid, side='right')]
    mg = at(grid)
    inv_pnl = (inv_grid[:-1] * np.diff(mg)).sum()
    print(f'inventory pnl on 1 s grid: {inv_pnl:+,.0f};  final inv {inv_grid[-1]}  mean|I| {np.abs(inv_grid).mean():.1f}  share |I|>=90 {np.mean(np.abs(inv_grid) >= 90):.3f}')
    print(f'  check: spread + inventory = {spread_pnl + inv_pnl:+,.0f}')
    # trend alignment: sign(I) vs past 30 s and next 30 s return
    r_next = np.r_[mg[30:] - mg[:-30], np.zeros(30)]
    print(f'  corr(I, next 30 s ret) {np.corrcoef(inv_grid, r_next)[0, 1]:+.3f}')
    for c in range(8):
        sl = slice(c * len(grid) // 8, (c + 1) * len(grid) // 8)
        ip = (inv_grid[sl][:-1] * np.diff(mg[sl])).sum()
        msk = (ts >= grid[sl][0]) & (ts < grid[sl][-1])
        sp = (sign[msk] * (m0[msk] - px[msk]) * q[msk]).sum()
        print(f'  chunk {c}: inventory {ip:+9,.0f}  spread {sp:+8,.0f}  price move {mg[sl][-1] - mg[sl][0]:+6.0f}')


if __name__ == '__main__':
    main()
