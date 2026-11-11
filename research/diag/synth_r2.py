"""Synthetic round-2 tape from the reverse-engineered generator fingerprint.

    uv run python -X utf8 -m research.diag.synth_r2 --seed 1 --dur 1500 --out data/synth/r2s_1500_s1.csv

Own book (price levels with FIFO order lists), so limit-order placement reacts to the current touch:
  * G inserts: Poisson 89.15/s/side, offset from own touch ~ empirical round2 distribution given own touch-qty
    bucket and spread, sizes and lifetimes resampled from round2, a Cancel row at insert+lifetime always.
  * F inserts: Poisson 4.45/s/side, size lognormal(4.105, 1.40), limit = touch + offset | hit-side qty quintile.
Side convention in the npz: ev_side +1 = bid (B), -1 = ask (A).
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
G_RATE, F_RATE = 89.15, 4.45
X_H = [2.4, 1.9, 1.5, 1.2, 0.9, 0.6, 0.4, 0.2]  # extra per-order cancel hazard (1/s) at a touch with 1..8 orders
QEDGES = np.array([1, 40, 116, 226, 359, 514, 694, 906, 1153, 1475, 1932, 2813, 68447])
SPREAD_CLASSES = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10**9)]


def _spread_class(sp: int) -> int:
    for i, (lo, hi) in enumerate(SPREAD_CLASSES):
        if lo <= sp <= hi:
            return i
    return 0


CEDGES = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33])  # own touch order count buckets
QT = np.array([300, 1000])  # own touch qty terciles (secondary)
OC = np.array([6, 14])  # opposite touch order count


def cell(c, cnt, q, ocnt):
    """Placement cell. The real generator is driven by the ORDER COUNT at the touch (share at touch 0.1 -> 0.78
    from 1-2 to 25+ orders at every qty tercile), qty is secondary."""
    cb = int(np.searchsorted(CEDGES, cnt, side='right'))
    if c == 0:
        return (0, cb, int(np.searchsorted(QT, q, side='right')), int(np.searchsorted(OC, ocnt, side='right')))
    if c <= 2:
        return (c, min(cb, 9), 0, 0)
    return (c, 0, 0, 0)


def load_tables():
    cache = ROOT / 'data/synth/r2_tables_v3.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z['place'].item(), z['ioc'].item(), z['ioc_edges'], z['gsize'], z['life']
    import pandas as pd
    d = np.load(ROOT / 'data/features/round2.npz')
    et, es, px = d['ev_type'], d['ev_side'], d['ev_px']
    qty = d['ev_qty']
    pb = np.r_[0, d['bid_px'][:-1]]
    pa = np.r_[0, d['ask_px'][:-1]]
    bq = np.r_[0, d['bid_qty'][:-1, 0]]
    aq = np.r_[0, d['ask_qty'][:-1, 0]]
    two = (pb > 0) & (pa > 0)
    sp = pa - pb
    own_qty = np.where(es == 1, bq, aq)
    behind = np.where(es == 1, pb - px, px - pa)
    g = (et == 0) & two
    qb = np.clip(np.searchsorted(QEDGES[1:-1], own_qty, side='right'), 0, 11)
    own_cnt = np.where(es == 1, np.r_[0, d['bid_cnt1'][:-1]], np.r_[0, d['ask_cnt1'][:-1]])
    opp_cnt = np.where(es == 1, np.r_[0, d['ask_cnt1'][:-1]], np.r_[0, d['bid_cnt1'][:-1]])
    cls = np.zeros(len(sp), int)
    for c, (lo, hi) in enumerate(SPREAD_CLASSES):
        cls[(sp >= lo) & (sp <= hi)] = c
    cb = np.searchsorted(CEDGES, own_cnt, side='right')
    qt = np.searchsorted(QT, own_qty, side='right')
    oc = np.searchsorted(OC, opp_cnt, side='right')
    gi = np.where(g)[0]
    keys = {}
    key = np.where(cls[gi] == 0, ((cb[gi] * 3 + qt[gi]) * 3 + oc[gi]), np.where(cls[gi] <= 2, np.minimum(cb[gi], 9), 0))
    place = {}
    for c in range(len(SPREAD_CLASSES)):
        mc = cls[gi] == c
        for k in np.unique(key[mc]):
            m = mc & (key == k)
            arr = behind[gi][m].astype(np.int16)
            if c == 0:
                kk = int(k)
                c_, rest = kk // 9, kk % 9
                place[(0, c_, rest // 3, rest % 3)] = arr
            else:
                place[(c, int(k), 0, 0)] = arr
        # fallback pool for sparse cells
        place[(c, 'all')] = behind[gi][mc].astype(np.int16)
    for c_ in range(len(CEDGES) + 1):
        place[(0, c_, 'all')] = behind[gi][(cls[gi] == 0) & (cb[gi] == c_)].astype(np.int16)
    # IOC offset through the touch given hit-side qty quintile
    i = (et == 1) & two
    hit_q = np.where(es == 1, aq, bq)
    off = np.where(es == 1, px - pa, pb - px)
    edges = np.percentile(hit_q[i], [20, 40, 60, 80])
    qq = np.searchsorted(edges, hit_q, side='right')
    ioc = {k: off[i & (qq == k)].astype(np.int16) for k in range(5)}
    gsize = qty[et == 0].astype(np.int32)
    df = pd.read_csv(ROOT / 'data/round2.csv', usecols=['Time', 'Operation', 'OrderId'])
    ins = df[df.Operation == 'Insert'].set_index('OrderId')['Time']
    can = df[df.Operation == 'Cancel'].set_index('OrderId')['Time']
    life = (can - ins.reindex(can.index)).dropna().to_numpy()
    life = life[life > 0]
    np.savez(cache, place=np.array(place, dtype=object), ioc=np.array(ioc, dtype=object), ioc_edges=edges,
             gsize=gsize, life=life)
    return place, ioc, edges, gsize, life


class Side:
    def __init__(self, sign: int):
        self.sign = sign  # +1 bid, -1 ask; heap key = -px for bids so the min is the best
        self.levels: dict[int, dict] = {}
        self.qty: dict[int, int] = {}
        self.heap: list[int] = []

    def best(self):
        h, lv = self.heap, self.levels
        while h:
            p = -h[0] if self.sign == 1 else h[0]
            if p in lv:
                return p
            heapq.heappop(h)
        return None

    def add(self, oid, px, v):
        lv = self.levels.get(px)
        if lv is None:
            lv = self.levels[px] = {}
            self.qty[px] = 0
            heapq.heappush(self.heap, -px if self.sign == 1 else px)
        lv[oid] = v
        self.qty[px] += v

    def remove(self, oid, px):
        lv = self.levels[px]
        self.qty[px] -= lv.pop(oid)
        if not lv:
            del self.levels[px], self.qty[px]


def generate(seed: int, dur: float, out: Path) -> None:
    place, ioc_tab, ioc_edges, gsize, life = load_tables()
    rng = np.random.default_rng(seed)
    total = 2 * (G_RATE + F_RATE)
    N = int(dur * total * 1.05) + 1000
    arr_t = np.cumsum(rng.exponential(1 / total, N))
    u_type = rng.random(N)
    u_side = rng.random(N) < 0.5
    u_off = rng.random(N)
    gsz = gsize[rng.integers(0, len(gsize), N)]
    lf = life[rng.integers(0, len(life), N)]
    isz = np.maximum(1, np.rint(np.exp(rng.normal(4.105, 1.40, N)))).astype(np.int64)
    p_f = F_RATE / (G_RATE + F_RATE)
    bid, ask = Side(1), Side(-1)
    sides = {1: bid, -1: ask}
    live: dict[int, tuple] = {}
    cancels: list[tuple] = []
    rows: list[str] = []
    ref = 10000
    last_us = 0
    oid = 1000000

    def stamp(t):
        nonlocal last_us
        us = max(int(round(t * 1e6)), last_us + 1)
        last_us = us
        return f'{us // 1000000}.{us % 1000000:06d}'

    i = 0
    early: set[int] = set()  # orders already cancelled by the touch-pull hazard
    tprev = 0.0
    xu = rng.random(N * 2)
    xi = 0
    while True:
        ta = arr_t[i]
        tn = min(ta, cancels[0][0]) if cancels else ta
        # extra cancel hazard of orders sitting at a THIN touch (the real touch empties faster than i.i.d. lifetimes
        # give; fitted on cancels/s by touch order count): per-order hazard X_H[cnt] / s
        dtx = tn - tprev
        tprev = tn
        for sd in (bid, ask):
            bp = sd.best()
            if bp is None:
                continue
            lv = sd.levels[bp]
            n = len(lv)
            if n <= len(X_H):
                xi += 1
                if xu[xi % len(xu)] < n * X_H[n - 1] * dtx:
                    cid = list(lv)[int(xu[(xi + 7) % len(xu)] * n) % n]
                    rows.append(f'{stamp(tn)},Cancel,{cid},,,,')
                    early.add(cid)
                    live.pop(cid, None)
                    sd.remove(cid, bp)
        if cancels and cancels[0][0] <= ta:
            tc, cid = heapq.heappop(cancels)
            if tc > dur:
                break
            if cid in early:
                early.discard(cid)
                continue
            rows.append(f'{stamp(tc)},Cancel,{cid},,,,')
            o = live.pop(cid, None)
            if o is not None:
                sides[o[0]].remove(cid, o[1])
            continue
        if ta > dur:
            break
        s = 1 if u_side[i] else -1
        own, opp = sides[s], sides[-s]
        bb, ba = bid.best(), ask.best()
        if bb is not None:
            ref = bb if ba is None else (bb + ba) // 2
        elif ba is not None:
            ref = ba
        oid += 1
        if u_type[i] >= p_f:  # limit order
            ob = opp.best()
            ownb = own.best()
            if ownb is None:
                if ob is not None:
                    ownb = ob - 1 if s == 1 else ob + 1
                else:
                    ownb = ref - 1 if s == 1 else ref + 1
                q = 0
            else:
                q = own.qty[ownb]
            sp = abs(ob - ownb) if ob is not None else 1
            sp = max(sp, 1)
            c = _spread_class(sp)
            cnt = len(own.levels[own.best()]) if own.best() is not None else 0
            ocnt = len(opp.levels[ob]) if ob is not None else 0
            k = cell(c, cnt, q, ocnt)
            arr = place.get(k)
            if arr is None or len(arr) < 200:
                arr = place.get((0, k[1], 'all')) if c == 0 else place[(c, 'all')]
                if arr is None or len(arr) < 200:
                    arr = place[(c, 'all')]
            beh = int(arr[int(u_off[i] * len(arr))])
            if beh < -(sp - 1):
                beh = -(sp - 1)
            px = ownb - s * beh
            v = int(gsz[i])
            own.add(oid, px, v)
            live[oid] = (s, px)
            heapq.heappush(cancels, (ta + float(lf[i]), oid))
            rows.append(f'{stamp(ta)},Insert,{oid},{"B" if s == 1 else "A"},{v},{px},G')
        else:  # IOC; s=+1 buys against the ask
            hit = opp
            hb = hit.best()
            hq = hit.qty[hb] if hb is not None else 0
            k = int(np.searchsorted(ioc_edges, hq, side='right'))
            arr = ioc_tab[k]
            off = int(arr[int(u_off[i] * len(arr))])
            base = hb if hb is not None else ref
            px = base + s * off
            v = int(isz[i])
            rows.append(f'{stamp(ta)},Insert,{oid},{"B" if s == 1 else "A"},{v},{px},F')
            rem = v
            while rem > 0:
                hb = hit.best()
                if hb is None or (s == 1 and hb > px) or (s == -1 and hb < px):
                    break
                lv = hit.levels[hb]
                for mo in list(lv):
                    take = min(rem, lv[mo])
                    rem -= take
                    lv[mo] -= take
                    hit.qty[hb] -= take
                    if lv[mo] == 0:
                        del lv[mo]
                        live.pop(mo, None)
                    if rem == 0:
                        break
                if not lv:
                    del hit.levels[hb], hit.qty[hb]
        i += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='\n') as f:
        f.write('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n')
        f.write('\n'.join(rows))
        f.write('\n')
    print(f'{out}: {len(rows):,} rows, {len(rows) / dur:.0f}/s')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dur', type=float, default=1500.0)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    generate(a.seed, a.dur, a.out)


if __name__ == '__main__':
    main()
