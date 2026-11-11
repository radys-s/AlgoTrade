"""Hidden-fair placement model (round2): how the generator draws a limit price given F, and estimator tests.

    uv run python -X utf8 -m research.diag.fair_model [explore|est]
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

CSV = 'data/round2.csv'
NPZ = 'data/features/round2.npz'
TR = 'logs/r2-plain-logs/000_round2/trades.csv'


def load(csv=CSV, npz=NPZ):
    df = pd.read_csv(csv)
    z = np.load(npz)
    ts = z['ts_us'].astype(np.int64)
    bb = z['bid_px'].astype(float)
    ba = z['ask_px'].astype(float)
    bad = (bb <= 0) | (ba <= 0) | (ba <= bb)
    bb = pd.Series(np.where(bad, np.nan, bb)).ffill().bfill().values
    ba = pd.Series(np.where(bad, np.nan, ba)).ffill().bfill().values
    bq = z['bid_qty'][:, 0].astype(float)
    aq = z['ask_qty'][:, 0].astype(float)
    return df, ts, bb, ba, bq, aq


def explore():
    df, ts, bb, ba, bq, aq = load()
    N = len(df)
    pbb = np.r_[bb[0], bb[:-1]]; pba = np.r_[ba[0], ba[:-1]]
    pbq = np.r_[bq[0], bq[:-1]]; paq = np.r_[aq[0], aq[:-1]]
    pm = (pbb + pba) / 2
    px = df.Price.values.astype(float)
    sd = np.where(df.Side.values == 'B', 1, -1)
    ins = df.Operation.values == 'Insert'
    isG = ins & (df.Lifespan.values == 'G')
    isF = ins & (df.Lifespan.values == 'F')
    G = np.where(isG)[0]
    sp = (pba - pbb)[G]
    s = sd[G]
    # distance from own touch (ticks behind; <0 improves), and from the opposite touch
    dist = np.where(s > 0, pbb[G] - px[G], px[G] - pba[G])
    cross = np.where(s > 0, px[G] - pba[G], pbb[G] - px[G])  # >= 0 means marketable
    print('G inserts', len(G), 'IOC', isF.sum())
    print('marketable G share', (cross >= 0).mean(), ' at opp touch -1 (inside, adjacent to opp):', (cross == -1).mean())
    print('spread dist at G inserts', pd.Series(sp).value_counts(normalize=True).sort_index().head(8).round(4).to_dict())
    print('dist from own touch (all):', pd.Series(np.clip(dist, -4, 8)).value_counts(normalize=True).sort_index().round(4).to_dict())
    for k in (1, 2, 3):
        m = sp == k
        print(f' spread {k}: dist', pd.Series(np.clip(dist[m], -4, 8)).value_counts(normalize=True).sort_index().round(3).to_dict())
    # F proxies: centred insert smoother (excluding self) and future mids
    off_b = np.mean((px - pm)[G][s > 0]); off_a = np.mean((px - pm)[G][s < 0])
    print('offsets vs mid before: bid %.3f ask %.3f' % (off_b, off_a))
    lvl = px[G] - np.where(s > 0, off_b, off_a)
    tG = ts[G]
    # centred smoother: mean of lvl over inserts within +-W us excluding self
    cs = np.r_[0.0, np.cumsum(lvl)]
    for W in (50_000, 150_000):
        lo = np.searchsorted(tG, tG - W, side='left'); hi = np.searchsorted(tG, tG + W, side='right')
        n = hi - lo - 1
        Fc = (cs[hi] - cs[lo] - lvl) / np.maximum(n, 1)
        r = px[G] - Fc
        print(f'== centred smoother +-{W/1e3:.0f} ms (mean n {n.mean():.1f})')
        for side in (1, -1):
            m = s == side
            e = r[m] - (off_b if side > 0 else off_a)
            print('  side', side, 'resid mean %.3f sd %.3f' % (e.mean(), e.std()),
                  'quantiles', np.round(np.quantile(r[m], [.01, .05, .1, .25, .5, .75, .9, .95, .99]), 2))
        # placement relative to F (round to ticks): d = Fc - px for bids, px - Fc for asks
        d = np.where(s > 0, Fc - px[G], px[G] - Fc)
        Fd = np.where(s > 0, Fc - pbb[G], pba[G] - Fc)  # how far F sits inside (+) or behind (-) the own touch
        Fo = np.where(s > 0, pba[G] - Fc, Fc - pbb[G])  # distance from F to the opposite touch (+ = opp touch beyond F)
        tab = pd.DataFrame({'d': np.round(d), 'Fo': np.clip(np.round(Fo), -2, 4), 'Fd': np.clip(np.round(Fd), -3, 4), 'dist': np.clip(dist, -3, 5)})
        print('  E[d] by F vs opposite touch:', tab.groupby('Fo').d.agg(['mean', 'count']).round(2).to_dict('index'))
        print('  dist from own touch by F vs own touch (rows Fd):')
        print(pd.crosstab(tab.Fd, tab.dist, normalize='index').round(3).to_string())
    # sizes vs distance
    vol = df.Volume.values.astype(float)[G]
    print('size median by dist', pd.Series(vol).groupby(np.clip(dist, -3, 5)).median().to_dict())
    print('size mean log by dist', pd.Series(np.log(vol)).groupby(np.clip(dist, -3, 5)).mean().round(2).to_dict())



def _ewma(t, x, tau_us):
    out = np.empty(len(x)); s = w = 0.0; tp = t[0]
    e = np.exp
    dec = np.exp(-np.diff(t, prepend=t[0]) / tau_us)
    for i in range(len(x)):
        a = dec[i]; s = s * a + x[i]; w = w * a + 1.0; out[i] = s / w
    return out


def _bracket(t, side, px, W):
    """max of bid inserts and min of ask inserts in the last W us (after each insert)."""
    from collections import deque
    lo = np.empty(len(px)); hi = np.empty(len(px))
    qb = deque(); qa = deque()  # (t, px) monotonic
    lb = np.nan; la = np.nan
    for i in range(len(px)):
        ti = t[i]; p = px[i]
        if side[i] > 0:
            while qb and qb[-1][1] <= p:
                qb.pop()
            qb.append((ti, p))
        else:
            while qa and qa[-1][1] >= p:
                qa.pop()
            qa.append((ti, p))
        h = ti - W
        while qb and qb[0][0] < h:
            qb.popleft()
        while qa and qa[0][0] < h:
            qa.popleft()
        lo[i] = qb[0][1] if qb else np.nan
        hi[i] = qa[0][1] if qa else np.nan
    return lo, hi


def _kalman(t, y, side, cens, c, q, r, sk=None):
    """F random walk (q ticks^2/s), obs y = px - off_side ~ N(F, r); censored obs (clamped at the opposite touch):
    bids only tell F + e >= c, asks F + e <= c (c in the y scale). Gaussian (ADF) update."""
    from math import erf, exp, sqrt
    out = np.empty(len(y)); m = y[0]; P = 1.0; tp = t[0]
    SQ2 = sqrt(2.0); ISQ2PI = 1.0 / sqrt(2.0 * np.pi)
    for i in range(len(y)):
        P += q * (t[i] - tp) * 1e-6; tp = t[i]
        if cens[i]:
            s = sqrt(P + r)
            sg = side[i]
            z = sg * (m - c[i]) / s  # bids: P(F+e >= c) = Phi((m-c)/s); asks mirrored
            if z > -8:
                Phi = 0.5 * (1.0 + erf(z / SQ2))
                lam = exp(-0.5 * z * z) * ISQ2PI / max(Phi, 1e-12)
                m += sg * P / s * lam
                P -= P * P / (s * s) * lam * (z + lam)
            else:
                m = c[i] - sg * 0.5
            if P < 1e-4:
                P = 1e-4
        else:
            K = P / (P + r)
            m += K * (y[i] - m)
            P *= (1 - K)
        out[i] = m
    return out


def est():
    from research.diag.fill_decomp import robust_mid_fn
    df, ts, bb, ba, bq, aq = load()
    N = len(df)
    pbb = np.r_[bb[0], bb[:-1]]; pba = np.r_[ba[0], ba[:-1]]
    ok = (ba - bb) <= 3
    rmid = pd.Series(np.where(ok, (bb + ba) / 2, np.nan)).ffill().bfill().values  # robust mid after each row
    pm = np.r_[rmid[0], rmid[:-1]]  # before
    px = df.Price.values.astype(float)
    sd = np.where(df.Side.values == 'B', 1, -1)
    isG = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
    G = np.where(isG & ((pba - pbb) <= 3))[0]
    s = sd[G]; pG = px[G]; tG = ts[G]
    off = {1: np.mean((px - pm)[G][s > 0]), -1: np.mean((px - pm)[G][s < 0])}
    offv = np.where(s > 0, off[1], off[-1])
    at, _, _ = robust_mid_fn(NPZ)
    y100 = at(tG + 100_000) - at(tG + 5_000)
    y1s = at(tG + 1_000_000) - at(tG + 5_000)
    tr = pd.read_csv(TR, dtype={'taker_order_id': str, 'maker_order_id': str}, low_memory=False)
    ours = tr[tr.maker_owner == 'user']
    fsign = np.where(ours.taker_side == 'B', -1, 1)
    ft = ours.timestamp_us.values; fq = ours.quantity.values; fpx = ours.price.values.astype(float)
    mk1 = fsign * (at(ft + 1_000_000) - fpx)
    kf = np.clip(np.searchsorted(ts, ft, side='left') - 1, 0, N - 1)
    print(f'G {len(G):,} offsets {off}; fills lots {fq.sum()/1e3:.0f}k pnl1s {(mk1*fq).sum()/1e3:+.0f}k', flush=True)
    res = {}

    def report(name, sigG, level=False):
        """sigG: value after each G insert. level=True: it is an F level, sig = F - robust mid now."""
        full = np.full(N, np.nan); full[G] = sigG
        full = pd.Series(full).ffill().values
        if level:
            sigfull = full - rmid
        else:
            sigfull = full
        sigfull = np.nan_to_num(sigfull)
        sg = sigfull[G]
        c100 = np.corrcoef(sg, y100)[0, 1]; c1 = np.corrcoef(sg, y1s)[0, 1]
        sf = fsign * sigfull[kf]
        o = np.argsort(sf, kind='stable'); cum = np.cumsum(fq[o])
        row = [f'{name:34s} c100 {c100:+.3f} c1s {c1:+.3f} sd {sg.std():.2f}']
        for target in (100e3, 200e3, 300e3):
            j = np.searchsorted(cum, target); ag = o[:j]
            row.append(f'{target/1e3:.0f}k:{(mk1[ag]*fq[ag]).sum()/1e3:+5.0f}k')
        for thr in (0.3,):
            ag = sf < -thr
            row.append(f'thr{thr} ag {fq[ag].sum()/1e3:.0f}k {(mk1[ag]*fq[ag]).sum()/1e3:+.0f}k')
        print(' | '.join(row), flush=True)
        res[name] = (c100, c1)
        return sigfull

    dev = pG - offv - pm[G]
    lvl = pG - offv
    for tau in (10, 20, 50, 100, 200):
        report(f'E0 dev ewma tau {tau}', _ewma(tG, dev, tau * 1e3))
    for tau in (20, 50, 100):
        report(f'E1 level ewma tau {tau}', _ewma(tG, lvl, tau * 1e3), level=True)
    # censoring: bid at the opposite touch - 1 (clamped), ask at bb + 1
    cens = np.where(s > 0, pG >= pba[G] - 1, pG <= pbb[G] + 1)
    print('censored share', cens.mean())
    c = np.where(s > 0, pba[G] - 1.5, pbb[G] + 1.5) - offv
    for q in (0.4, 1.5, 6.0):
        for r in (0.5,):
            report(f'E2 kalman q {q} r {r} (no cens)', _kalman(tG, lvl, s, np.zeros(len(G), bool), c, q, r), level=True)
            report(f'E2 kalman q {q} r {r} tobit', _kalman(tG, lvl, s, cens, c, q, r), level=True)
    for W in (30, 60, 120):
        lo, hi = _bracket(tG, s, pG, W * 1e3)
        mid_b = np.where(np.isnan(lo) | np.isnan(hi), np.nan, (lo + hi) / 2)
        f = report(f'E3 bracket W {W}', pd.Series(mid_b).ffill().bfill().values, level=True)
        res[f'br{W}'] = f
    e0 = report('E0 tau 50 (again)', _ewma(tG, dev, 50e3))
    for W in (60, 120):
        comb = 0.5 * e0 + 0.5 * res[f'br{W}']
        report(f'E4 0.5 E0 + 0.5 bracket {W}', comb[G])




def est2():
    """Dev-space variants: Tobit Kalman on dev, momentum hybrid, dev bracket."""
    from research.diag.fill_decomp import robust_mid_fn
    df, ts, bb, ba, bq, aq = load()
    N = len(df)
    pbb = np.r_[bb[0], bb[:-1]]; pba = np.r_[ba[0], ba[:-1]]
    ok = (ba - bb) <= 3
    rmid = pd.Series(np.where(ok, (bb + ba) / 2, np.nan)).ffill().bfill().values
    pm = np.r_[rmid[0], rmid[:-1]]
    px = df.Price.values.astype(float)
    sd = np.where(df.Side.values == 'B', 1, -1)
    isG = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
    G = np.where(isG & ((pba - pbb) <= 3))[0]
    s = sd[G]; pG = px[G]; tG = ts[G]
    off = {1: np.mean((px - pm)[G][s > 0]), -1: np.mean((px - pm)[G][s < 0])}
    offv = np.where(s > 0, off[1], off[-1])
    at, _, _ = robust_mid_fn(NPZ)
    y100 = at(tG + 100_000) - at(tG + 5_000)
    y1s = at(tG + 1_000_000) - at(tG + 5_000)
    tr = pd.read_csv(TR, dtype={'taker_order_id': str, 'maker_order_id': str}, low_memory=False)
    ours = tr[tr.maker_owner == 'user']
    fsign = np.where(ours.taker_side == 'B', -1, 1)
    ft = ours.timestamp_us.values; fq = ours.quantity.values; fpx = ours.price.values.astype(float)
    mk1 = fsign * (at(ft + 1_000_000) - fpx)
    kf = np.clip(np.searchsorted(ts, ft, side='left') - 1, 0, N - 1)

    def report(name, sigfull):
        sigfull = np.nan_to_num(sigfull)
        sg = sigfull[G]
        c100 = np.corrcoef(sg, y100)[0, 1]; c1 = np.corrcoef(sg, y1s)[0, 1]
        sf = fsign * sigfull[kf]
        o = np.argsort(sf, kind='stable'); cum = np.cumsum(fq[o])
        row = [f'{name:36s} c100 {c100:+.3f} c1s {c1:+.3f} sd {sg.std():.2f}']
        for target in (100e3, 200e3, 300e3):
            j = np.searchsorted(cum, target); ag = o[:j]
            row.append(f'{target/1e3:.0f}k:{(mk1[ag]*fq[ag]).sum()/1e3:+5.0f}k')
        ag = sf < -0.3
        row.append(f'thr0.3 ag {fq[ag].sum()/1e3:.0f}k {(mk1[ag]*fq[ag]).sum()/1e3:+.0f}k')
        print(' | '.join(row), flush=True)

    def ff(vG):
        full = np.full(N, np.nan); full[G] = vG
        return pd.Series(full).ffill().values

    dev = pG - offv - pm[G]
    cens = np.where(s > 0, pG >= pba[G] - 1, pG <= pbb[G] + 1)
    c = np.where(s > 0, pba[G] - 1.5, pbb[G] + 1.5) - offv - pm[G]  # censor bound in dev space
    for tau in (20, 50):
        e = _ewma(tG, dev, tau * 1e3)
        mbar = _ewma(tG, pm[G], tau * 1e3)  # decayed mean of the mid at the inserts
        base = ff(e); mb = ff(mbar)
        for g in (-0.5, -0.25, 0.0, 0.25):
            report(f'H dev tau {tau} + {g} (mbar - mid)', base + g * (mb - rmid))
    for q in (2.0, 5.0, 12.0, 30.0):
        report(f'K dev kalman q {q} (no cens)', ff(_kalman(tG, dev, s, np.zeros(len(G), bool), c, q, 0.5)))
        report(f'K dev kalman q {q} tobit', ff(_kalman(tG, dev, s, cens, c, q, 0.5)))
    for W in (30, 60):
        lo, hi = _bracket(tG, s, pG - pm[G], W * 1e3)
        report(f'B dev bracket W {W}', ff(np.where(np.isnan(lo) | np.isnan(hi), np.nan, (lo + hi) / 2)))
    # only uncensored inserts
    u = ~cens
    e = np.full(len(G), np.nan); e[u] = _ewma(tG[u], dev[u], 50e3)
    report('U dev ewma tau 50 uncensored only', ff(e))




def _kalman2(t, y, r, side, cens, c, q):
    """Like _kalman with a per-observation noise r[i]."""
    from math import erf, exp, sqrt
    out = np.empty(len(y)); m = 0.0; P = 1.0; tp = t[0]
    for i in range(len(y)):
        P += q * (t[i] - tp) * 1e-6; tp = t[i]
        ri = r[i]
        if cens[i]:
            s = sqrt(P + ri); sg = side[i]
            z = sg * (m - c[i]) / s
            if z > -8:
                Phi = 0.5 * (1.0 + erf(z * 0.7071067811865476))
                lam = exp(-0.5 * z * z) * 0.3989422804014327 / max(Phi, 1e-12)
                m += sg * P / s * lam
                P -= P * P / (s * s) * lam * (z + lam)
            else:
                m = c[i] - sg * 0.5
            if P < 1e-4:
                P = 1e-4
        else:
            K = P / (P + ri); m += K * (y[i] - m); P *= (1 - K)
        out[i] = m
    return out


def est3():
    """Visible IOC information (deepest trade price of each IOC that traded) merged into the dev Kalman."""
    from research.diag.fill_decomp import robust_mid_fn
    df, ts, bb, ba, bq, aq = load()
    z = np.load(NPZ)
    N = len(df)
    pbb = np.r_[bb[0], bb[:-1]]; pba = np.r_[ba[0], ba[:-1]]
    ok = (ba - bb) <= 3
    rmid = pd.Series(np.where(ok, (bb + ba) / 2, np.nan)).ffill().bfill().values
    pm = np.r_[rmid[0], rmid[:-1]]
    px = df.Price.values.astype(float)
    sd = np.where(df.Side.values == 'B', 1, -1)
    ins = df.Operation.values == 'Insert'
    isG = ins & (df.Lifespan.values == 'G')
    isF = ins & (df.Lifespan.values == 'F')
    good = (pba - pbb) <= 3
    G = np.where(isG & good)[0]
    at, _, _ = robust_mid_fn(NPZ)
    tr = pd.read_csv(TR, dtype={'taker_order_id': str, 'maker_order_id': str}, low_memory=False)
    ours = tr[tr.maker_owner == 'user']
    fsign = np.where(ours.taker_side == 'B', -1, 1)
    ft = ours.timestamp_us.values; fq = ours.quantity.values; fpx = ours.price.values.astype(float)
    mk1 = fsign * (at(ft + 1_000_000) - fpx)
    kf = np.clip(np.searchsorted(ts, ft, side='left') - 1, 0, N - 1)
    tG = ts[G]
    y100 = at(tG + 100_000) - at(tG + 5_000)
    y1s = at(tG + 1_000_000) - at(tG + 5_000)

    def report(name, sigfull):
        sigfull = np.nan_to_num(sigfull)
        sg = sigfull[G]
        c100 = np.corrcoef(sg, y100)[0, 1]; c1 = np.corrcoef(sg, y1s)[0, 1]
        sf = fsign * sigfull[kf]
        o = np.argsort(sf, kind='stable'); cum = np.cumsum(fq[o])
        row = [f'{name:40s} c100 {c100:+.3f} c1s {c1:+.3f} sd {sg.std():.2f}']
        for target in (100e3, 200e3, 300e3):
            j = np.searchsorted(cum, target); ag = o[:j]
            row.append(f'{target/1e3:.0f}k:{(mk1[ag]*fq[ag]).sum()/1e3:+5.0f}k')
        print(' | '.join(row), flush=True)

    # IOC rows: true limit (tape only) and the visible deepest trade
    Fi = np.where(isF & good)[0]
    tc = z['trade_cnt'][Fi]; ltp = z['last_trade_px'][Fi].astype(float)
    vis = tc > 0
    sF = sd[Fi]
    dl = px[Fi] - pm[Fi]
    dv = ltp - pm[Fi]
    print(f'IOC {len(Fi):,} visible {vis.mean():.3f}')
    for side in (1, -1):
        m = sF == side
        print(f' side {side}: limit-mid mean {dl[m].mean():+.2f} sd {dl[m].std():.2f}; visible deepest-mid mean {dv[m & vis].mean():+.2f} sd {dv[m & vis].std():.2f}')
    yF100 = at(ts[Fi] + 100_000) - at(ts[Fi] + 5_000)
    print(' corr(limit dev, mid move 100ms) all', np.corrcoef(dl - np.where(sF > 0, dl[sF > 0].mean(), dl[sF < 0].mean()), yF100)[0, 1])
    vo = {1: dv[(sF > 0) & vis].mean(), -1: dv[(sF < 0) & vis].mean()}
    dvc = dv - np.where(sF > 0, vo[1], vo[-1])
    print(' corr(visible deepest dev, mid move 100ms)', np.corrcoef(dvc[vis], yF100[vis])[0, 1], ' invisible share by side', [(~vis[sF == k]).mean() for k in (1, -1)])
    # merged stream: G (dev, r .5) + visible IOC (dev, r_ioc)
    s = sd[G]; pG = px[G]
    off = {1: np.mean((px - pm)[G][s > 0]), -1: np.mean((px - pm)[G][s < 0])}
    dev = pG - np.where(s > 0, off[1], off[-1]) - pm[G]
    V = Fi[vis]
    for rio in (None, 2.0, 1.0, 0.5):
        if rio is None:
            idx = G; yy = dev; rr = np.full(len(G), 0.5)
        else:
            idx = np.r_[G, V]; yy = np.r_[dev, dvc[vis]]; rr = np.r_[np.full(len(G), 0.5), np.full(len(V), rio)]
            o = np.argsort(idx, kind='stable'); idx = idx[o]; yy = yy[o]; rr = rr[o]
        out = _kalman2(ts[idx], yy, rr, np.ones(len(idx)), np.zeros(len(idx), bool), yy, 5.0)
        full = np.full(N, np.nan); full[idx] = out
        report(f'K q5 + visible IOC r {rio}', pd.Series(full).ffill().values)


if __name__ == '__main__':
    {'explore': explore, 'est': est, 'est2': est2, 'est3': est3}[sys.argv[1] if len(sys.argv) > 1 else 'explore']()
