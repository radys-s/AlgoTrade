"""Diagnostic: a pure hidden-fair TAKER, non-overlapping trades, full 100-lot size, realistic crossing.

    uv run python -X utf8 -m research.diag.fair_taker data/round2.csv
Entry at the first row >= t + 5 ms (buy at its ask / sell at its bid, the touch price; size <= touch qty),
exit by crossing back at the first row >= entry + hold, or earlier when the signal flips sign (exit_on_flip).
One position at a time.
"""
from __future__ import annotations
import sys
import numpy as np
from research.diag.fair_fills import fair_signal


def main() -> None:
    csv = sys.argv[1]
    npz = csv.replace('data/', 'data/features/').replace('.csv', '.npz')
    z = np.load(npz)
    ts = z['ts_us'].astype(np.int64)
    b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
    bq = z['bid_qty'][:, 0].astype(float); aq = z['ask_qty'][:, 0].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    for tau in (20.0, 50.0):
        _, sig = fair_signal(csv, npz, tau)
        n = len(ts)
        for thr in (1.0, 1.5, 2.0, 3.0):
            for hold_ms in (100, 300, 1000, 3000):
                i = 0
                pnl = 0.0; lots = 0; trades = 0
                cand = np.where(ok & (np.abs(sig) > thr))[0]
                ptr = 0
                chunk_pnl = np.zeros(8)
                while ptr < len(cand):
                    i = cand[ptr]
                    s = 1 if sig[i] > 0 else -1
                    e = np.searchsorted(ts, ts[i] + 5000)
                    if e >= n:
                        break
                    if not ok[e]:
                        ptr += 1
                        continue
                    q = min(100.0, aq[e] if s > 0 else bq[e])
                    if q <= 0:
                        ptr += 1
                        continue
                    px_in = a[e] if s > 0 else b[e]
                    x = np.searchsorted(ts, ts[e] + hold_ms * 1000)
                    # walk to the first ok row at/after x
                    while x < n and not ok[x]:
                        x += 1
                    if x >= n:
                        break
                    px_out = b[x] if s > 0 else a[x]
                    r = s * (px_out - px_in) * q
                    pnl += r; lots += q; trades += 1
                    chunk_pnl[min(7, int(8 * (ts[i] - ts[0]) / (ts[-1] - ts[0])))] += r
                    # next candidate after exit (+5 ms for the exit order)
                    ptr = np.searchsorted(cand, x + 1)
                print(f'tau {tau:3.0f} thr {thr:3.1f} hold {hold_ms:5d} ms: trades {trades:6,} lots {lots:9,.0f} pnl {pnl:+10,.0f} '
                      f'({pnl / max(lots, 1):+.2f}/lot)  chunks+ {(chunk_pnl > 0).sum()}/8 min {chunk_pnl.min():+9,.0f}')


if __name__ == '__main__':
    main()
