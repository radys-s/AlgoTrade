"""Round-2 anatomy: what the generator does differently from round 1 and where the price trend comes from.

    uv run python -X utf8 -m research.diag.r2_anatomy data/features/round2.npz data/features/round1.npz

Reads the per-event feature npz files (research.features) and prints, per dataset:
  A. generator parameters (rates, sizes, placement, lifetimes, touch depth, spread, volatility)
  B. which tape events move the mid (IOC sweep / cancel depletion / insert improve) and how much of the
     net drift each type carries
  C. persistence and predictive power of flow imbalances (IOC, touch inserts, touch cancels) over
     1 s / 10 s / 60 s windows, against the NEXT window's mid change
  D. long-horizon return predictability on non-overlapping windows (variance ratios, lag-1 autocorrelation)
"""
from __future__ import annotations

import sys

import numpy as np


def load(path: str) -> dict:
    d = np.load(path)
    out = {k: d[k] for k in ['ts_us', 'bid_px', 'ask_px', 'ev_type', 'ev_side', 'ev_qty', 'ev_px', 'trade_vol']}
    out['bq1'] = d['bid_qty'][:, 0]
    out['aq1'] = d['ask_qty'][:, 0]
    return out


def mid_of(d: dict) -> np.ndarray:
    b, a = d['bid_px'].astype(float), d['ask_px'].astype(float)
    m = np.where((b > 0) & (a > 0), (b + a) / 2, np.nan)
    # forward fill one-sided moments
    idx = np.where(np.isnan(m), 0, np.arange(len(m)))
    np.maximum.accumulate(idx, out=idx)
    m = m[idx]
    m[np.isnan(m)] = np.nanmean(m[:1000])
    return m


def q(x, ps=(5, 25, 50, 75, 95, 99)):
    return ' '.join(f'q{p}={np.percentile(x, p):.0f}' for p in ps)


def section_a(d: dict, mid: np.ndarray) -> None:
    t = d['ts_us'] / 1e6
    T = t[-1] - t[0]
    et, es, eq = d['ev_type'], d['ev_side'], d['ev_qty']
    print(f'duration {T:.0f} s, events {len(t):,} ({len(t) / T:.0f}/s)')
    for name, m in [('limit G', et == 0), ('IOC', et == 1), ('cancel', et == 2)]:
        print(f'  {name:8s} rate {m.sum() / T:7.2f}/s  bid {((m) & (es == 1)).sum() / T:6.2f}/s  ask {((m) & (es == -1)).sum() / T:6.2f}/s'
              f'  size mean {eq[m].mean():7.1f} {q(eq[m])}')
    # placement of limit inserts vs book before
    gi = np.where(et == 0)[0]
    gi = gi[gi > 0]
    prevb, preva = d['bid_px'][gi - 1], d['ask_px'][gi - 1]
    ok = (prevb > 0) & (preva > 0)
    gi, prevb, preva = gi[ok], prevb[ok], preva[ok]
    side = es[gi]
    px = d['ev_px'][gi]
    behind = np.where(side == 1, prevb - px, px - preva)  # >0 behind touch, 0 at touch, <0 improves
    vals, cnts = np.unique(np.clip(behind, -3, 6), return_counts=True)
    print('  placement (ticks behind own touch; neg = improves): ' + ' '.join(f'{v}:{c / cnts.sum():.3f}' for v, c in zip(vals, cnts)))
    # size by placement
    for v in [-1, 0, 1, 2]:
        mm = behind == v if v >= 0 else behind < 0
        if mm.sum():
            print(f'    size at {v:+d}: mean {d["ev_qty"][gi][mm].mean():.1f} median {np.median(d["ev_qty"][gi][mm]):.0f}  n={mm.sum():,}')
    spread = (d['ask_px'] - d['bid_px'])[(d['bid_px'] > 0) & (d['ask_px'] > 0)]
    vals, cnts = np.unique(np.clip(spread, 1, 5), return_counts=True)
    print('  spread (events): ' + ' '.join(f'{v}:{c / cnts.sum():.3f}' for v, c in zip(vals, cnts)))
    print(f'  touch qty bid {q(d["bq1"], (5, 25, 50, 75, 95))}')
    grid = np.arange(t[0] + 1, t[-1], 1.0)
    gm = mid[np.searchsorted(t, grid) - 1]
    r1 = np.diff(gm)
    print(f'  1s mid std {r1.std():.3f} ticks; mid start {gm[0]:.1f} end {gm[-1]:.1f} min {gm.min():.1f} max {gm.max():.1f}')


def section_b(d: dict, mid: np.ndarray) -> None:
    dm = np.diff(mid)
    et = d['ev_type'][1:]
    es = d['ev_side'][1:]
    tot_abs = np.abs(dm).sum()
    print(f'  total |dmid| {tot_abs:,.0f} ticks, net {dm.sum():+,.1f}')
    for name, m in [('insert G', et == 0), ('IOC', et == 1), ('cancel', et == 2)]:
        x = dm[m]
        nz = x != 0
        print(f'  {name:8s}: moves {nz.sum():7,} |dmid| {np.abs(x).sum():9,.0f} ({np.abs(x).sum() / tot_abs:.1%})  net {x.sum():+9,.1f}'
              f'  up {(x > 0).sum():6,} down {(x < 0).sum():6,}')
    # IOC by side: buy IOC should move up
    for s, nm in [(1, 'buy'), (-1, 'sell')]:
        m = (et == 1) & (es == s)
        x = dm[m]
        print(f'    IOC {nm}: n {m.sum():,}, mean dmid {x.mean():+.3f}, share moving {np.mean(x != 0):.3f}, net {x.sum():+,.0f}')
    for s, nm in [(1, 'bid'), (-1, 'ask')]:
        m = (et == 2) & (es == s)
        x = dm[m]
        print(f'    cancel {nm}: n {m.sum():,}, moving {np.sum(x != 0):,}, net {x.sum():+,.0f}')
        m = (et == 0) & (es == s)
        x = dm[m]
        print(f'    insert {nm}: n {m.sum():,}, moving {np.sum(x != 0):,}, net {x.sum():+,.0f}')


def window_series(d: dict, mid: np.ndarray, w: float):
    t = d['ts_us'] / 1e6
    nb = int((t[-1] - t[0]) // w)
    b = np.minimum(((t - t[0]) // w).astype(int), nb - 1)
    et, es, eq = d['ev_type'], d['ev_side'], d['ev_qty'].astype(float)
    ioc = np.bincount(b, weights=np.where(et == 1, es * eq, 0), minlength=nb)
    iocn = np.bincount(b, weights=np.where(et == 1, es, 0), minlength=nb)
    # inserts at or better than own touch (book before)
    prevb = np.r_[0, d['bid_px'][:-1]]
    preva = np.r_[0, d['ask_px'][:-1]]
    at_touch = np.where(es == 1, d['ev_px'] >= prevb, d['ev_px'] <= preva) & (prevb > 0) & (preva > 0)
    ins = np.bincount(b, weights=np.where((et == 0) & at_touch, es * eq, 0), minlength=nb)
    improve = np.where(es == 1, d['ev_px'] > prevb, d['ev_px'] < preva) & (prevb > 0) & (preva > 0)
    imp = np.bincount(b, weights=np.where((et == 0) & improve, es, 0), minlength=nb)
    # cancels at the touch (side of cancelled order, price == own touch before)
    can_touch = np.where(es == 1, d['ev_px'] == prevb, d['ev_px'] == preva)
    can = np.bincount(b, weights=np.where((et == 2) & can_touch, es * eq, 0), minlength=nb)
    edges = t[0] + np.arange(nb + 1) * w
    me = mid[np.clip(np.searchsorted(t, edges) - 1, 0, len(mid) - 1)]
    ret = np.diff(me)
    return {'ioc_vol': ioc, 'ioc_cnt': iocn, 'ins_touch': ins, 'improve': imp, 'cancel_touch': -can, 'ret': ret}


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def section_c(d: dict, mid: np.ndarray) -> None:
    for w in [1.0, 10.0, 60.0]:
        s = window_series(d, mid, w)
        n = len(s['ret'])
        se = 1 / np.sqrt(n)
        print(f'  window {w:.0f} s (n={n}, SE~{se:.3f}):')
        r = s['ret']
        print(f'    ret: lag1 autocorr {corr(r[:-1], r[1:]):+.3f}')
        for k in ['ioc_vol', 'ioc_cnt', 'ins_touch', 'improve', 'cancel_touch']:
            x = s[k]
            print(f'    {k:12s}: same-window corr w/ ret {corr(x, r):+.3f}  -> NEXT ret {corr(x[:-1], r[1:]):+.3f}'
                  f'  self lag1 {corr(x[:-1], x[1:]):+.3f}  lag5 {corr(x[:-5], x[5:]):+.3f}')


def section_d(d: dict, mid: np.ndarray) -> None:
    t = d['ts_us'] / 1e6
    grid = np.arange(t[0], t[-1], 1.0)
    gm = mid[np.clip(np.searchsorted(t, grid) - 1, 0, len(mid) - 1)]
    r1 = np.diff(gm)
    v1 = r1.var()
    for h in [5, 10, 30, 60, 120, 300]:
        rh = gm[::h]
        r = np.diff(rh)
        vr = r.var() / (h * v1)
        ac = corr(r[:-1], r[1:])
        print(f'  h={h:4d}s n={len(r):5d}: VR {vr:.2f}  lag1 ac {ac:+.3f} (SE {1 / np.sqrt(len(r)):.3f})')


def main() -> None:
    for path in sys.argv[1:]:
        print(f'\n===== {path}')
        d = load(path)
        mid = mid_of(d)
        print('A. generator')
        section_a(d, mid)
        print('B. who moves the mid')
        section_b(d, mid)
        print('C. flow imbalances vs next-window return')
        section_c(d, mid)
        print('D. long-horizon predictability (non-overlapping)')
        section_d(d, mid)


if __name__ == '__main__':
    main()
