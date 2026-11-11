"""Diagnostic: hidden-fair JUMPS. A new limit insert whose implied fair (bid + 1.4 / ask - 1.4) is far from the
current mid reveals a jump of F; the old book on that side is stale until the generator's IOCs sweep it.

    uv run python -X utf8 -m research.diag.jumps_f data/round2.csv [min_jump_ticks]
For each detected jump: size, stale lots on the wrong side priced worse than new F by >= 2 ticks (top 5 levels
of the book just before), and how many of them are still there 5 / 50 / 200 / 1000 ms later.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd


def main() -> None:
    csv = sys.argv[1]
    min_jump = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    df = pd.read_csv(csv)
    z = np.load(csv.replace('data/', 'data/features/').replace('.csv', '.npz'))
    ts = z['ts_us']; b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
    bq = z['bid_qty'].astype(float); aq = z['ask_qty'].astype(float)
    n = len(ts)
    pb = np.r_[0, b[:-1]]; pa = np.r_[0, a[:-1]]
    G = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
    side = np.where(df.Side.values == 'B', 1, -1); px = df.Price.values.astype(float)
    ok = (pb > 0) & (pa > 0)
    mid = (pb + pa) / 2
    implied = np.where(side == 1, px + 1.4, px - 1.4)
    jump = implied - mid
    cand = np.where(G & ok & (np.abs(jump) >= min_jump))[0]
    # keep the first insert of each jump episode (no other candidate within the previous 500 ms)
    first = []
    last_t = -10**12
    for i in cand:
        if ts[i] - last_t > 500_000:
            first.append(i)
        last_t = ts[i]
    first = np.array(first)
    print(f'{csv}: candidate inserts {len(cand):,}, jump episodes (>= {min_jump} ticks) {len(first)}')
    sizes = jump[first]
    print('  jump size quantiles (abs):', np.percentile(np.abs(sizes), [10, 50, 90, 99]).round(1), ' up', (sizes > 0).sum(), 'down', (sizes < 0).sum())
    tot_edge = 0.0; tot_lots = 0.0
    for lab, lim in (('all', 1e9),):
        pass
    rows = []
    for i in first:
        s = 1 if jump[i] > 0 else -1  # F jumped up: stale ASKS (cheap) -> buy; down: stale bids -> sell
        F = implied[i]
        # stale liquidity in the top-5 levels before the insert: asks below F-2 (up) / bids above F+2 (down)
        if s > 0:
            lv_px = pa[i] + np.arange(5)  # approx consecutive ticks (features store qty per distinct level)
            q = aq[i - 1]
            stale = q[lv_px <= F - 2].sum()
            edge = ((F - lv_px) * q)[lv_px <= F - 2].sum()
        else:
            lv_px = pb[i] - np.arange(5)
            q = bq[i - 1]
            stale = q[lv_px >= F + 2].sum()
            edge = ((lv_px - F) * q)[lv_px >= F + 2].sum()
        later = []
        for dt in (5_000, 50_000, 200_000, 1_000_000):
            j = min(n - 1, np.searchsorted(ts, ts[i] + dt))
            if s > 0:
                later.append(aq[j][(a[j] + np.arange(5)) <= F - 2].sum() if a[j] > 0 else 0)
            else:
                later.append(bq[j][(b[j] - np.arange(5)) >= F + 2].sum() if b[j] > 0 else 0)
        rows.append((abs(jump[i]), stale, edge, *later))
        tot_edge += edge; tot_lots += stale
    r = np.array(rows)
    print(f'  stale lots in top-5 levels at the jump: median {np.median(r[:, 1]):.0f}, total {r[:, 1].sum():,.0f}; '
          f'edge sum (ticks*lots, if all were taken at once) {r[:, 2].sum():,.0f}')
    print(f'  stale lots left after 5 / 50 / 200 / 1000 ms (median): {np.median(r[:, 3]):.0f} / {np.median(r[:, 4]):.0f} / '
          f'{np.median(r[:, 5]):.0f} / {np.median(r[:, 6]):.0f}')
    for lo, hi in ((5, 10), (10, 20), (20, 50), (50, 1000)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum():
            print(f'  jumps {lo}-{hi} ticks: n {m.sum():4d}  stale lots median {np.median(r[m, 1]):6.0f}  edge sum {r[m, 2].sum():12,.0f}  '
                  f'edge capped at 200 lots/jump ~{(np.minimum(r[m, 1], 200) * r[m, 0] / 2).sum():10,.0f}')


if __name__ == '__main__':
    main()
