"""Diagnostic: the order-id leak. Insert OrderIds are consecutive across limit (G) and IOC (F) rows, so a gap in
the ids we see counts inserts we never saw: IOCs that found nothing to trade and limit orders placed >= 20 ticks
behind their side's best (outside the public snapshot). Do those unseen inserts cluster right after big hidden-F
jumps, and do they show up before the visible evidence of the jump?

    .venv/Scripts/python.exe -X utf8 -m research.diag.leak_scan data/round3.csv [min_jump_ticks=20]

Needs data/features/<stem>.npz (research.features), row-aligned with the CSV (book after each row).
Jump = a new G insert whose implied fair (bid + 1.4 / ask - 1.4) is >= min_jump ticks from the mid before it;
episodes are the first such insert after 500 ms without one.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

DEPTH = 20  # snapshot depth (simulator.visible_views: abs(price - best) >= depth is hidden)


def main() -> None:
    csv = sys.argv[1]
    min_jump = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    df = pd.read_csv(csv, usecols=['Time', 'Operation', 'Side', 'Price', 'Lifespan'])
    z = np.load(csv.replace('data/', 'data/features/').replace('.csv', '.npz'))
    ts = z['ts_us'].astype(np.int64)
    b = z['bid_px'].astype(float)
    a = z['ask_px'].astype(float)
    n = len(ts)
    assert n == len(df), (n, len(df))
    pb = np.r_[0.0, b[:-1]]
    pa = np.r_[0.0, a[:-1]]
    ok = (pb > 0) & (pa > 0)
    ins = df.Operation.values == 'Insert'
    lif = df.Lifespan.values
    G = ins & (lif == 'G')
    F = ins & (lif == 'F')
    side = np.where(df.Side.values == 'B', 1, -1)
    px = np.nan_to_num(df.Price.values.astype(float))
    # unseen inserts
    f_cross = np.where(side == 1, px >= pa, px <= pb)
    f_unseen = F & ok & ~f_cross
    g_deep = G & ok & np.where(side == 1, px <= pb - DEPTH, px >= pa + DEPTH)
    unseen = f_unseen | g_deep
    T = (ts[-1] - ts[0]) / 1e6
    print(f'{csv}: {n:,} rows, {T:.0f} s, inserts {ins.sum():,}')
    print(f'  unseen inserts: {unseen.sum():,} = {unseen.sum() / T:.3f}/s  (non-trading IOC {f_unseen.sum():,} '
          f'{f_unseen.sum() / T:.3f}/s, deep limit {g_deep.sum():,} {g_deep.sum() / T:.3f}/s)')
    ins_idx = np.where(ins)[0]
    vis_ins = ins & ~unseen
    # jumps
    mid = (pb + pa) / 2
    implied = np.where(side == 1, px + 1.4, px - 1.4)
    jump = np.where(G & ok, implied - mid, 0.0)
    cand = np.where(np.abs(jump) >= min_jump)[0]
    first = []
    last_t = -10**12
    for i in cand:
        if ts[i] - last_t > 500_000:
            first.append(i)
        last_t = ts[i]
    first = np.array(first, dtype=np.int64)
    sz = jump[first]
    print(f'  jump episodes >= {min_jump:.0f} ticks: {len(first)} ({len(first) / T * 60:.2f}/min), '
          f'up {(sz > 0).sum()} down {(sz < 0).sum()}, |size| p10/50/90 {np.percentile(np.abs(sz), [10, 50, 90]).round(0)}')
    print(f'  trigger insert itself unseen (deep limit): {unseen[first].mean() * 100:.0f} %')
    u_ts = ts[unseen]
    rng = np.random.default_rng(0)
    rand_t = rng.uniform(ts[0], ts[-1], 20000).astype(np.int64)
    print('  unseen inserts per window (excluding the trigger row):')
    for w in (10_000, 20_000, 40_000):
        after = np.searchsorted(u_ts, ts[first] + w, 'right') - np.searchsorted(u_ts, ts[first], 'right')
        before = np.searchsorted(u_ts, ts[first], 'left') - np.searchsorted(u_ts, ts[first] - w, 'left')
        rnd = np.searchsorted(u_ts, rand_t + w, 'right') - np.searchsorted(u_ts, rand_t, 'right')
        print(f'    {w // 1000:3d} ms: after jump {after.mean():.2f} (P>=1 {np.mean(after >= 1) * 100:.0f} %, >=2 '
              f'{np.mean(after >= 2) * 100:.0f} %), before jump {before.mean():.2f}, random {rnd.mean():.4f} '
              f'(P>=1 {np.mean(rnd >= 1) * 100:.1f} %, >=2 {np.mean(rnd >= 2) * 100:.2f} %)')
    # Book intact: the stale side's best has not moved half the jump toward the new fair yet.
    # Lead: the first VISIBLE evidence (a visible insert, same direction, implied >= 10 ticks past the mid at
    # the trigger, or a crossing IOC in the jump's direction) vs the first id-gap reveal (the first visible
    # insert after an unseen one in the episode).
    intact = []
    lead_vis = []
    lead_gap = []
    unseen_first = 0
    gap_side_far = []
    for i in first:
        s = 1 if jump[i] > 0 else -1
        t0 = ts[i]
        m0 = mid[i]
        Fi = implied[i]
        stop = np.searchsorted(ts, t0 + 200_000)
        j = i
        if s > 0:
            target = pa[i] + 0.5 * (Fi - pa[i])
            while j < stop and a[j] < target:
                j += 1
        else:
            target = pb[i] - 0.5 * (pb[i] - Fi)
            while j < stop and b[j] > target:
                j += 1
        intact.append((ts[min(j, n - 1)] - t0) / 1000)
        # visible evidence
        tv = None
        tg = None
        saw_unseen = False
        for k in range(i, stop):
            if not ins[k]:
                continue
            if unseen[k]:
                saw_unseen = True
                continue
            if tg is None and saw_unseen:
                tg = ts[k] - t0
            if G[k] and (implied[k] - m0) * s >= 10:
                tv = ts[k] - t0
            elif F[k] and side[k] == s and f_cross[k]:
                tv = ts[k] - t0
            if tv is not None:
                break
        lead_vis.append(tv / 1000 if tv is not None else np.nan)
        lead_gap.append(tg / 1000 if tg is not None else np.nan)
        if tg is not None and (tv is None or tg < tv):
            unseen_first += 1
        # side signature: for an up jump the hidden inserts are asks behind the ask (deep) or sells that found
        # nothing; record which side the first 2 unseen inserts were on
        ks = [k for k in range(i, min(stop, i + 400)) if unseen[k]][:2]
        gap_side_far.append([side[k] * s for k in ks])
    intact = np.array(intact)
    lv = np.array(lead_vis)
    lg = np.array(lead_gap)
    print(f'  book intact after trigger (ms) p10/25/50/75/90: {np.percentile(intact, [10, 25, 50, 75, 90]).round(1)}'
          f'  share intact at +5 ms {np.mean(intact > 5) * 100:.0f} %, +20 ms {np.mean(intact > 20) * 100:.0f} %, '
          f'+50 ms {np.mean(intact > 50) * 100:.0f} %')
    print(f'  first visible evidence (ms) p25/50/75: {np.nanpercentile(lv, [25, 50, 75]).round(2)}, none in 200 ms '
          f'{np.isnan(lv).mean() * 100:.0f} %')
    print(f'  first id-gap reveal (ms) p25/50/75: {np.nanpercentile(lg, [25, 50, 75]).round(2)}, gap reveal BEFORE '
          f'visible evidence in {unseen_first} / {len(first)} episodes')
    flat = [v for l in gap_side_far for v in l]
    if flat:
        print(f'  side of the first unseen inserts relative to the jump: same-side (buy on up) {np.mean(np.array(flat) > 0) * 100:.0f} %, '
              f'opposite {np.mean(np.array(flat) < 0) * 100:.0f} %  (n={len(flat)})')
    del ins_idx, vis_ins


if __name__ == '__main__' and sys.argv[1] != 'label':
    main()


def label_triggers(csv: str, trig_file: str, min_jump: float = 20.0, tol_us: int = 100_000) -> None:
    """Compare strategy trigger log (ts_us dir unseen) with labelled jump episodes: a trigger is true if it falls within
    [jump, jump+tol] with the jump's direction (wrong_dir if opposite sign), else false."""
    df = pd.read_csv(csv, usecols=['Time', 'Operation', 'Side', 'Price', 'Lifespan'])
    z = np.load(csv.replace('data/', 'data/features/').replace('.csv', '.npz'))
    ts = z['ts_us'].astype(np.int64)
    pb = np.r_[0.0, z['bid_px'][:-1].astype(float)]
    pa = np.r_[0.0, z['ask_px'][:-1].astype(float)]
    ok = (pb > 0) & (pa > 0)
    G = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
    side = np.where(df.Side.values == 'B', 1, -1)
    px = np.nan_to_num(df.Price.values.astype(float))
    jump = np.where(G & ok, np.where(side == 1, px + 1.4, px - 1.4) - (pb + pa) / 2, 0.0)
    cand = np.where(np.abs(jump) >= min_jump)[0]
    first, last_t = [], -10**12
    for i in cand:
        if ts[i] - last_t > 500_000:
            first.append(i)
        last_t = ts[i]
    jt = np.array([ts[i] for i in first])
    jd = np.array([1 if jump[i] > 0 else -1 for i in first])
    tr = np.loadtxt(trig_file, ndmin=2) if __import__('os').path.exists(trig_file) else np.zeros((0, 3))
    t0 = ts[0]
    tt = tr[:, 0].astype(np.int64) + 0  # simulator time base may differ; align by offset below
    print(f'{trig_file}: {len(tr)} triggers, {len(jt)} jumps; trig ts range {tt.min() if len(tt) else None}..{tt.max() if len(tt) else None}, jump ts range {jt.min()}..{jt.max()}')
    good = wrong = false = 0
    hit = set()
    for t, d in zip(tt, tr[:, 1]):
        k = np.where((t >= jt) & (t <= jt + tol_us))[0]
        if len(k):
            if d == jd[k[0]]:
                good += 1
                hit.add(k[0])
            else:
                wrong += 1
        else:
            false += 1
    print(f'  true (right dir) {good}, wrong-dir {wrong}, false {false}; jumps caught {len(hit)}/{len(jt)}')


if __name__ == '__main__' and len(sys.argv) > 3 and sys.argv[1] == 'label':
    label_triggers(sys.argv[2], sys.argv[3])
