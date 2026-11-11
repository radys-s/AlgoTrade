"""Diagnostic: is a pure IOC taker viable? Gross move vs crossing cost by signal decile, several horizons.

    uv run python -X utf8 -m research.diag.taker_econ data/features/round2.npz

Decision at every event with a trade (one sample per trading event). Entry = cross the touch of the book
as of the first row at/after t + 5 ms (buy at ask / sell at bid, signal sign); exit = cross back at the
first row at/after t + 5 ms + h (taker exit) or the robust mid there (passive exit, no adverse cost).
Prints mean pnl per lot per decile of each signal (long when signal > 0 is assumed; the top decile is the
interesting row, bottom decile shown mirrored).
"""
from __future__ import annotations
import sys
import numpy as np


def main() -> None:
    d = np.load(sys.argv[1])
    t = d['ts_us']; b = d['bid_px'].astype(float); a = d['ask_px'].astype(float); tv = d['trade_vol'].astype(float)
    two = (b > 0) & (a > 0)
    ok = two & ((a - b) <= 3)
    idx = np.where(ok)[0]
    rm = (b + a) / 2

    def row_at(T):
        k = np.clip(np.searchsorted(t, T, side='left'), 0, len(t) - 1)
        j = np.clip(np.searchsorted(idx, k, side='left'), 0, len(idx) - 1)
        return idx[j]
    ev = np.where(tv != 0)[0]
    ev = ev[(ev > 200) & (t[ev] < t[-1] - 70e6)]
    # signals
    cs = np.cumsum(tv)
    flow100 = cs[ev] - cs[ev - 100]
    flow20 = cs[ev] - cs[ev - 20]
    def rmid_back(T):
        k = np.clip(np.searchsorted(t, T, side='right') - 1, 0, len(t) - 1)
        j = np.clip(np.searchsorted(idx, k, side='right') - 1, 0, len(idx) - 1)
        return rm[idx[j]]
    now_mid = rmid_back(t[ev])
    ret1s = now_mid - rmid_back(t[ev] - 1_000_000)
    ret10s = now_mid - rmid_back(t[ev] - 10_000_000)
    ret30s = now_mid - rmid_back(t[ev] - 30_000_000)
    e0 = row_at(t[ev] + 5000)
    for name, s in [('flow100ev', flow100), ('flow20ev', flow20), ('ret1s', ret1s), ('ret10s', ret10s), ('ret30s', ret30s)]:
        print(f'== {name}  (n={len(ev):,})')
        q = np.quantile(s, [0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
        for h in [1, 5, 15, 60]:
            e1 = row_at(t[ev] + 5000 + h * 1_000_000)
            buy_cross = b[e1] - a[e0]
            sell_cross = b[e0] - a[e1]
            buy_pass = rm[e1] - a[e0]
            sell_pass = b[e0] - rm[e1]
            out = []
            for lab, m_long, m_short in [('1%', s >= q[5], s <= q[0]), ('5%', s >= q[4], s <= q[1]), ('10%', s >= q[3], s <= q[2])]:
                pc = np.r_[buy_cross[m_long], sell_cross[m_short]]
                pp = np.r_[buy_pass[m_long], sell_pass[m_short]]
                out.append(f'top{lab}: cross {pc.mean():+.2f} pass-exit {pp.mean():+.2f} (n {len(pc)})')
            print(f'  h={h:2d}s  ' + ' | '.join(out))


if __name__ == '__main__':
    main()
