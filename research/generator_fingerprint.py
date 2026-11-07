"""Fingerprint of the round tape generator: state-dependent intensities, hazards, sizes, timing.
Usage: uv run python -X utf8 research/generator_fingerprint.py [round1|sample_v2] [part...]"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

name = sys.argv[1] if len(sys.argv) > 1 else 'round1'
parts = sys.argv[2:] or ['1', '2', '3', '4', '5', '6']
z = np.load(f'data/features/{name}.npz')
d = pd.read_csv(f'data/{name}.csv')
n = len(d)
t = z['ts_us'].astype(np.int64)
et = z['ev_type']
es = z['ev_side']
eq = z['ev_qty']
ep = z['ev_px']
bp = z['bid_px'].astype(np.int64)
ap = z['ask_px'].astype(np.int64)
bq = z['bid_qty']
aq = z['ask_qty']
oid = d.OrderId.values
ins_idx = np.flatnonzero(et != 2)
canc_idx = np.flatnonzero(et == 2)
cid = np.full(n, -1, np.int64)  # for cancels: row of the insert
cid[canc_idx] = pd.Series(ins_idx, index=oid[ins_idx]).reindex(oid[canc_idx]).values


def prev(a):
    return np.concatenate([a[:1], a[:-1]])


pb, pa = prev(bp), prev(ap)
pbq, paq = prev(bq[:, 0]), prev(aq[:, 0])
dt = np.diff(t, prepend=t[0]) / 1e6  # seconds the pre-state lasted
T = (t[-1] - t[0]) / 1e6
two = (pb > 0) & (pa > 0)
spr = pa - pb


def hdr(s):
    print('\n=== ' + s)


def pct(x, q):
    return np.percentile(x, q)


if '1' in parts:
    hdr('1. insert intensity vs state (events/s; exposure = time in state)')
    G = (et == 0) & two
    own_px = np.where(es == 1, pb, pa)
    off = np.where(es == 1, own_px - ep, ep - own_px)  # 0 at touch, k behind, -1 inside
    qs = np.unique(np.percentile(pbq[two], np.linspace(0, 100, 13)).astype(int))
    for side, lab in ((1, 'bid'), (-1, 'ask')):
        q = pbq if side == 1 else paq
        b = np.digitize(q, qs[1:-1])
        expo = np.bincount(b[two], weights=dt[two], minlength=len(qs))
        print(f'-- {lab} side, own touch-qty buckets', list(qs))
        rows = []
        for k in range(len(qs) - 1):
            m = G & (es == side) & (b == k)
            e = expo[k]

            def r(mm):
                return mm.sum() / e if e > 0 else np.nan
            rows.append((f'{qs[k]}-{qs[k+1]}', e, r(m & (off == 0)), r(m & (off == 1)), r(m & (off == 2)),
                         r(m & (off < 0)), (m & (off == 0)).sum() / max(m.sum(), 1)))
        print(pd.DataFrame(rows, columns=['qty', 'exposure_s', 'at_touch/s', 'b1/s', 'b2/s', 'inside/s', 'share_at_touch']).round(3).to_string(index=False))
    print('-- bid at-touch insert rate vs OPPOSITE (ask) touch qty')
    qo = np.unique(np.percentile(paq[two], np.linspace(0, 100, 9)).astype(int))
    b = np.digitize(paq, qo[1:-1])
    expo = np.bincount(b[two], weights=dt[two], minlength=len(qo))
    rows = []
    for k in range(len(qo) - 1):
        m = G & (es == 1) & (b == k) & (off == 0)
        rows.append((f'{qo[k]}-{qo[k+1]}', expo[k], m.sum() / expo[k]))
    print(pd.DataFrame(rows, columns=['ask_qty', 'exposure_s', 'bid_at_touch/s']).round(3).to_string(index=False))
    print('-- by spread (rates by side and offset)')
    rows = []
    for s in (1, 2, 3):
        mm = two & (spr == s)
        e = dt[mm].sum()
        row = [s, e]
        for sd in (1, -1):
            for o in (-1, 0, 1, 2):
                row.append((G & mm & (es == sd) & (off == o)).sum() / e)
        rows.append(row)
    print(pd.DataFrame(rows, columns=['spr', 'exp_s'] + [f'{s}{o}' for s in 'BA' for o in ('in', '0', '1', '2')]).round(3).to_string(index=False))
    print('-- overall mean G insert rate/s per side', (et == 0).sum() / T / 2)
    cb = prev(z['bid_cnt1'])
    bc = np.digitize(cb, [3, 6, 9, 12, 15, 18, 21, 25, 30, 35])
    expo = np.bincount(bc[two], weights=dt[two], minlength=11)
    rows = []
    for k in range(11):
        m = G & (es == 1) & (bc == k)
        rows.append((k, expo[k], (m & (off == 0)).sum() / max(expo[k], 1e-9), (m & (off == 1)).sum() / max(expo[k], 1e-9)))
    print('-- bid insert rate vs number of orders at bid touch (bucket edges 3,6,9,..35)')
    print(pd.DataFrame(rows, columns=['cntbucket', 'exp_s', 'bid_touch/s', 'bid_b1/s']).round(3).to_string(index=False))

if '2' in parts:
    hdr('2. cancel hazard')
    ci = canc_idx[cid[canc_idx] >= 0]
    ii = cid[ci]
    life = (t[ci] - t[ii]) / 1e3  # ms
    print('lifetime ms mean/std/median', life.mean(), life.std(), np.median(life), 'cv', life.std() / life.mean())
    edges = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1300, 1700, 2200, 3000, 8000])
    cnt = np.histogram(life, edges)[0]
    surv = np.array([(life >= e).sum() for e in edges[:-1]])
    haz = cnt / surv / (np.diff(edges) / 1e3)
    print(pd.DataFrame({'age_lo_ms': edges[:-1], 'n': cnt, 'haz_per_s': haz}).round(3).to_string(index=False))
    print('KS vs exp(mean)', stats.kstest(life, 'expon', args=(0, life.mean())))
    print('life quantiles 0.1,1,5,50,95,99,99.9', pct(life, [0.1, 1, 5, 50, 95, 99, 99.9]).round(1))
    sz = eq[ii]
    for lo, hi in ((1, 5), (5, 14), (14, 50), (50, 200), (200, 10**7)):
        m = (sz >= lo) & (sz < hi)
        print(f'size[{lo},{hi}) n={m.sum()} life mean {life[m].mean():.1f} med {np.median(life[m]):.1f}')
    bid_s = es[ii] == 1
    own_q = np.where(bid_s, pbq[ii], paq[ii])
    own_px = np.where(bid_s, pb[ii], pa[ii])
    offi = np.where(bid_s, own_px - ep[ii], ep[ii] - own_px)
    for o in (-1, 0, 1, 2):
        m = offi == o
        print(f'offset-at-insert {o}: n={m.sum()} mean life {life[m].mean():.1f} med {np.median(life[m]):.1f}')
    qb = np.percentile(own_q, [0, 20, 40, 60, 80, 100])
    for k in range(5):
        m = (own_q >= qb[k]) & (own_q <= qb[k + 1])
        print(f'touch qty@insert {qb[k]:.0f}-{qb[k+1]:.0f}: mean life {life[m].mean():.1f}')
    for k in range(4):
        m = (t[ii] >= k * 3e9) & (t[ii] < (k + 1) * 3e9)
        print('quarter', k, 'mean life', life[m].mean())
    ref_ins = np.where(bid_s, bp[ii], ap[ii])
    ref_c = np.where(bid_s, bp[ci - 1], ap[ci - 1])
    moved = ref_c != ref_ins
    print('touch moved since insert: share', moved.mean(), 'mean life moved / not', life[moved].mean(), life[~moved].mean())
    dist_c = np.where(bid_s, bp[ci - 1] - ep[ii], ep[ii] - ap[ci - 1])
    vc = pd.Series(dist_c).value_counts().sort_index()
    print('ticks behind own touch at cancel time (counts):')
    print(vc[(vc.index > -3) & (vc.index < 8)].to_string())
    print('cancel events per s', len(ci) / T)
    # hazard conditional on distance behind touch at cancel time vs exposure is unavailable; use life by distance
    for dd in (-1, 0, 1, 2):
        m = dist_c == dd
        print(f'cancelled while {dd} behind touch: n={m.sum()} mean life {life[m].mean():.1f}')

if '3' in parts:
    hdr('3. sizes')
    gs = eq[et == 0].astype(float)
    fs = eq[et == 1].astype(float)
    rng = np.random.default_rng(0)
    for nm, x in (('limit', gs), ('IOC', fs)):
        print(nm, 'n', len(x), 'mean', x.mean(), 'median', np.median(x), 'std', x.std(), 'min', x.min(), 'max', x.max())
        lx = np.log(x)
        print('  log mean/std', lx.mean(), lx.std(), 'skew of log', stats.skew(lx), 'kurt', stats.kurtosis(lx))
        sub = rng.choice(x, min(200000, len(x)), replace=False)
        print('  KS lognormal(fit) subsample (jittered):', stats.kstest(np.log(sub + rng.uniform(-.5, .5, len(sub))), lambda v: stats.norm.cdf(v, lx.mean(), lx.std())))
    vc = pd.Series(gs.astype(int)).value_counts().sort_index()
    print('limit size pmf first 45:', (vc.iloc[:45] / len(gs)).round(4).to_dict())
    for nm, x in (('limit', gs), ('IOC', fs)):
        xs = np.sort(x)[::-1]
        k = int(len(x) * 0.02)
        print(nm, 'Hill alpha (top 2%)', 1 / np.mean(np.log(xs[:k] / xs[k])))
    print('limit mult5', (gs % 5 == 0).mean(), 'mult10', (gs % 10 == 0).mean(), 'IOC mult5', (fs % 5 == 0).mean(), 'mult10', (fs % 10 == 0).mean())
    Fm = (et == 1) & two
    hq = np.where(es == 1, paq, pbq).astype(float)[Fm]
    fsz = eq[Fm].astype(float)
    print('corr(log ioc size, log hit-queue)', np.corrcoef(np.log(fsz), np.log(hq + 1))[0, 1], 'Spearman', stats.spearmanr(fsz, hq)[0])
    qb = np.percentile(hq, [0, 10, 25, 50, 75, 90, 100])
    for k in range(6):
        m = (hq >= qb[k]) & (hq <= qb[k + 1])
        print(f'hit-queue {qb[k]:.0f}-{qb[k+1]:.0f}: ioc size median {np.median(fsz[m]):.0f} mean {fsz[m].mean():.0f}; share size>=queue {(fsz[m] >= hq[m]).mean():.3f}')
    print('share IOC size >= touch qty', (fsz >= hq).mean())
    Gm = (et == 0) & two
    gq = np.where(es == 1, pbq, paq).astype(float)[Gm]
    gz = eq[Gm].astype(float)
    print('Spearman(limit size, own touch qty)', stats.spearmanr(gz, gq)[0])
    own_px = np.where(es == 1, pb, pa)
    off = np.where(es == 1, own_px - ep, ep - own_px)[Gm]
    for o in (-1, 0, 1, 2):
        m = off == o
        print('limit size by offset', o, 'median', np.median(gz[m]), 'mean', gz[m].mean())
    for s in (1, 2):
        m = spr[Fm] == s
        print('IOC size by spread', s, 'median', np.median(fsz[m]), 'n', m.sum())
    # size autocorrelation across successive inserts
    print('lag-1 corr of log limit size', np.corrcoef(np.log(gs[1:]), np.log(gs[:-1]))[0, 1])

if '4' in parts:
    hdr('4. price placement and refills')
    G = (et == 0) & two
    own_px = np.where(es == 1, pb, pa)
    off = np.where(es == 1, own_px - ep, ep - own_px)
    own_q = np.where(es == 1, pbq, paq)
    print('offset shares overall', pd.Series(off[G]).value_counts(normalize=True).sort_index().round(4).to_dict())
    for s in (1, 2, 3):
        m = G & (spr == s)
        print('spread', s, pd.Series(off[m]).value_counts(normalize=True).sort_index().round(4).to_dict())
    qb = np.percentile(own_q[G], [0, 20, 40, 60, 80, 100])
    for k in range(5):
        m = G & (own_q >= qb[k]) & (own_q <= qb[k + 1])
        print(f'own touch qty {qb[k]:.0f}-{qb[k+1]:.0f}', pd.Series(off[m]).value_counts(normalize=True).sort_index().round(4).to_dict())
    qa = np.where(es == 1, paq, pbq)
    qb2 = np.percentile(qa[G], [0, 20, 40, 60, 80, 100])
    for k in range(5):
        m = G & (qa >= qb2[k]) & (qa <= qb2[k + 1])
        print(f'opp touch qty {qb2[k]:.0f}-{qb2[k+1]:.0f}', pd.Series(off[m]).value_counts(normalize=True).sort_index().round(4).to_dict())
    sp = np.where((bp > 0) & (ap > 0), ap - bp, 0)
    st = np.flatnonzero((sp[1:] == 2) & (sp[:-1] == 1)) + 1
    rows = []
    for i in st[:60000]:
        j = i
        while j + 1 < n and sp[j] == 2:
            j += 1
        rows.append((t[j] - t[i], et[j], es[j], bp[i] - bp[i - 1], ap[i] - ap[i - 1], et[i], es[i], j - i))
    r = pd.DataFrame(rows, columns=['dur_us', 'end_type', 'end_side', 'dbid', 'dask', 'start_type', 'start_side', 'nev'])
    print('spread 1->2 episodes (first 60k)', len(r))
    print('duration us pct', r.dur_us.describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(0).to_dict())
    print('start event type counts', r.start_type.value_counts().to_dict(), '(0=G ins,1=IOC,2=cancel)')
    print('ending event (type, side) counts', r.groupby(['end_type', 'end_side']).size().to_dict())
    print('start: bid fell share', (r.dbid < 0).mean(), 'ask rose', (r.dask > 0).mean())
    print('refill side == retreated side share:', ((r.end_side == 1) & (r.dbid < 0) | (r.end_side == -1) & (r.dask > 0)).mean())
    print('ender is G insert share', (r.end_type == 0).mean(), ' mean events in episode', r.nev.mean())

if '5' in parts:
    hdr('5. timing')
    dts = np.diff(t)
    print('inter-event us: mean', dts.mean(), 'std', dts.std(), 'cv', dts.std() / dts.mean(), 'min', dts.min(), 'median', np.median(dts), 'zeros', (dts == 0).sum())
    rng = np.random.default_rng(0)
    print('KS vs exp', stats.kstest(dts[:500000] + rng.uniform(-.5, .5, 500000), 'expon', args=(0, dts.mean())))
    ts = (d.Time.values * 1e6).round().astype(np.int64)
    print('mod 10 us digit dist', (np.bincount(ts % 10, minlength=10) / n).round(4))
    print('us-within-ms chi2 p', stats.chisquare(np.bincount(ts % 1000, minlength=1000))[1])
    print('us mod 100 chi2 p', stats.chisquare(np.bincount(ts % 100, minlength=100))[1])
    for w, lab in ((1_000_000, '1s'), (100_000, '100ms'), (10_000, '10ms')):
        c = np.bincount((t // w).astype(int))
        print(f'per-{lab} count Fano', c.var() / c.mean())
    x = dts[:1000000] - dts.mean()
    print('acf dt lag1..3', [round(float((x[:-k] * x[k:]).mean() / x.var()), 4) for k in (1, 2, 3)])
    pt = np.concatenate([[0], et[:-1]])
    print('P(next type | prev type)'); print(pd.crosstab(pt, et, normalize='index').round(4))
    print('insert id==1000001+insert index share', (oid[ins_idx] == 1000001 + np.arange(len(ins_idx))).mean(), ' cancel ids share unique', len(np.unique(oid[canc_idx])) / len(canc_idx))
    print('first 1000 rows: (type,side) counts', pd.Series(list(zip(et[:1000], es[:1000]))).value_counts().to_dict())
    print('first two-sided row', np.flatnonzero((bp > 0) & (ap > 0))[0], ' t of row 1000 (s)', t[999] / 1e6)
    print('events/s by 1000s bucket', (np.bincount((t // 1_000_000_000).astype(int)) / 1000).round(1))
    sg = es[et == 0]
    print('P(same side as previous G insert)', (sg[1:] == sg[:-1]).mean())
    sg = es[et != 2]
    print('P(same side as previous insert)', (sg[1:] == sg[:-1]).mean())
    # cancel order: oldest-first? rank of cancelled order's age among live orders is expensive; use lag of id
    ci = canc_idx[cid[canc_idx] >= 0]
    print('P(cancel side == previous event side)', (es[ci] == es[ci - 1]).mean())
    print('(side of cancelled order == side of previous cancel)', (es[cid[ci]][1:] == es[cid[ci]][:-1]).mean())

if '6' in parts:
    hdr('6. mid dynamics')
    valid = (bp > 0) & (ap > 0)
    cur = np.where(valid, bp + ap, 0)
    pm = prev(cur)
    ch = (cur != pm) & (cur > 0) & (pm > 0)
    print('mid change events', ch.sum(), 'per s', ch.sum() / T)
    for ty, lab in ((0, 'G insert'), (1, 'IOC'), (2, 'cancel')):
        m = ch & (et == ty)
        print(f'{lab}: n={m.sum()} share={m.sum()/ch.sum():.3f} mean |d(2*mid)|={np.abs((cur-pm)[m]).mean():.2f}')
    m = ch & (et == 1)
    print('IOC: mid move sign == IOC side', (np.sign((cur - pm)[m]) == es[m]).mean())
    m = ch & (et == 2)
    print('cancel mid moves: up share', ((cur - pm)[m] > 0).mean())
    m = ch & (et == 0)
    offi = np.where(es == 1, ep - pb, pa - ep)
    print('G insert mid moves: inside-insert share', (offi[m] > 0).mean(), ' of which toward own side sign ok', (np.sign((cur - pm)[m]) == es[m]).mean())
    cb = prev(z['bid_cnt1'])
    bid_drop = (bp < pb) & (pb > 0)
    ask_rise = (ap > pa) & (pa > 0)
    print('level depletions by cancel', ((et == 2) & (bid_drop | ask_rise)).sum(), 'by IOC', ((et == 1) & (bid_drop | ask_rise)).sum())
    for ty, lab in ((1, 'IOC'), (2, 'cancel')):
        m = (et == ty) & bid_drop
        print(lab, 'bid-touch depletion: pre-qty median', np.median(pbq[m]), ' pre-cnt median', np.median(cb[m]), 'mean qty', pbq[m].mean())
    m = (et == 1) & two
    hit = np.where(es == 1, paq, pbq)[m]
    sz = eq[m]
    dep = (np.where(es == 1, ap > pa, bp < pb))[m]
    r = sz / np.maximum(hit, 1)
    for lo, hi in ((0, .05), (.05, .2), (.2, .5), (.5, .99), (.99, 1.5), (1.5, 100)):
        mm = (r >= lo) & (r < hi)
        print(f'ioc size/queue [{lo},{hi}) n={mm.sum()} P(deplete)={dep[mm].mean():.3f}')
    # refill after touch retreat: time and size of next G insert on that side
    ev = np.flatnonzero(bid_drop & (et != 0))[:30000]
    gi = np.flatnonzero(et == 0)
    lag, szs, off_ = [], [], []
    for i in ev:
        k = gi[np.searchsorted(gi, i + 1):][:400]
        k = k[es[k] == 1]
        if len(k):
            lag.append(t[k[0]] - t[i]); szs.append(eq[k[0]]); off_.append(ep[k[0]] - bp[i])
    print('after bid retreat: next bid G insert lag us pct(10,50,90)', pct(lag, [10, 50, 90]), 'size median', np.median(szs), 'offset vs new bid', pd.Series(off_).value_counts(normalize=True).sort_index().round(3).head(6).to_dict())

if '7' in parts:
    hdr('7. extras: arrival Fano by type, lifetime fits, conditional hazard, depletion odds')
    for ty, lab in ((0, 'G'), (1, 'IOC'), (2, 'cancel')):
        for w, wl in ((1_000_000, '1s'), (100_000, '100ms')):
            c = np.bincount((t[et == ty] // w).astype(int), minlength=int(t[-1] // w) + 1)
            print(f'{lab} per-{wl} Fano', round(c.var() / c.mean(), 3))
    for sd in (1, -1):
        c = np.bincount((t[(et == 0) & (es == sd)] // 1_000_000).astype(int), minlength=12000)
        print('G side', sd, 'per-1s Fano', round(c.var() / c.mean(), 3))
    cb_ = np.bincount((t[et == 0] // 1_000_000).astype(int), minlength=12000)
    cc_ = np.bincount((t[et == 2] // 1_000_000).astype(int), minlength=12000)
    print('corr per-second G inserts vs cancels', np.corrcoef(cb_, cc_)[0, 1], ' G lag-1 autocorr', np.corrcoef(cb_[1:], cb_[:-1])[0, 1])
    ci = canc_idx[cid[canc_idx] >= 0]
    ii = cid[ci]
    life = (t[ci] - t[ii]) / 1e3
    sub = np.random.default_rng(1).choice(life, 300000, replace=False)
    for nm, dist in (('gamma', stats.gamma), ('lognorm', stats.lognorm), ('weibull', stats.weibull_min), ('invgauss', stats.invgauss), ('expon', stats.expon), ('norm', stats.norm)):
        par = dist.fit(sub[:60000])
        print(nm, 'params', np.round(par, 3), 'KS', round(stats.kstest(sub, dist.cdf, args=par).statistic, 4))
    print('life quantiles (ms)', pct(life, [1, 10, 25, 50, 75, 90, 99, 99.99]).round(1), 'max', life.max())
    lt = np.log(life)
    print('log life mean/sd/skew/kurt', lt.mean(), lt.std(), stats.skew(lt), stats.kurtosis(lt))
    # conditional hazard: alive at 300 ms; cancel in next 100 ms vs state at 300 ms
    tc = t[ii] + 300_000
    alive = (t[ci] > tc)
    k = np.searchsorted(t, tc[alive]) - 1
    ci_a, ii_a = ci[alive], ii[alive]
    bid_s = es[ii_a] == 1
    px = ep[ii_a]
    best = np.where(bid_s, bp[k], ap[k])
    at_touch = px == best
    dist = np.where(bid_s, best - px, px - best)
    cancelled_soon = (t[ci_a] - tc[alive]) < 100_000
    for dd in (-1, 0, 1, 2, 3):
        m = dist == dd
        print(f'alive@300ms dist-behind-touch {dd}: n={m.sum()} P(cancel in next 100ms)={cancelled_soon[m].mean():.4f}')
    tq = np.where(bid_s, bq[k, 0], aq[k, 0])
    m0 = dist == 0
    qb = np.percentile(tq[m0], [0, 25, 50, 75, 100])
    for j in range(4):
        m = m0 & (tq >= qb[j]) & (tq <= qb[j + 1])
        print(f'at-touch, touch qty {qb[j]:.0f}-{qb[j+1]:.0f}: P(cancel next 100ms)={cancelled_soon[m].mean():.4f}')
    oq = np.where(bid_s, aq[k, 0], bq[k, 0])
    qb = np.percentile(oq[m0], [0, 25, 50, 75, 100])
    for j in range(4):
        m = m0 & (oq >= qb[j]) & (oq <= qb[j + 1])
        print(f'at-touch, OPPOSITE qty {qb[j]:.0f}-{qb[j+1]:.0f}: P(cancel next 100ms)={cancelled_soon[m].mean():.4f}')
    # touch depletion odds: P(own touch level removed within W | qty, cnt)
    for W in (50_000, 200_000):
        idx = np.arange(0, n - 1, 7)
        idx = idx[(bp[idx] > 0) & (ap[idx] > 0)]
        j = np.searchsorted(t, t[idx] + W) - 1
        j = np.minimum(j, n - 1)
        # bid depleted if bid price fell at any time: use min of bp in window approx via bp[j] < bp[idx]
        dep = bp[j] < bp[idx]
        q = bq[idx, 0]
        cnt = z['bid_cnt1'][idx]
        qb = np.percentile(q, [0, 5, 10, 20, 40, 60, 80, 100])
        rows = []
        for a in range(7):
            m = (q >= qb[a]) & (q <= qb[a + 1])
            rows.append((f'{qb[a]:.0f}-{qb[a+1]:.0f}', m.sum(), dep[m].mean()))
        print(f'P(bid touch level gone, bid px lower, after {W/1000:.0f} ms) by bid qty'); print(pd.DataFrame(rows, columns=['bidqty', 'n', 'P']).round(4).to_string(index=False))
        imb = (bq[idx, 0] - aq[idx, 0]) / (bq[idx, 0] + aq[idx, 0])
        ib = np.percentile(imb, [0, 10, 25, 50, 75, 90, 100])
        rows = []
        for a in range(6):
            m = (imb >= ib[a]) & (imb <= ib[a + 1])
            up = ap[j][m] > ap[idx][m]
            rows.append((f'{ib[a]:.2f}..{ib[a+1]:.2f}', dep[m].mean(), up.mean()))
        print('P(bid down) / P(ask up) by imbalance'); print(pd.DataFrame(rows, columns=['imb', 'P_bid_down', 'P_ask_up']).round(4).to_string(index=False))
        cb2 = np.digitize(cnt, [1, 2, 3, 5, 8, 12])
        rows = [(a, (cb2 == a).sum(), dep[cb2 == a].mean()) for a in range(7)]
        print('P(bid down) by order count at bid (bucket edges 1,2,3,5,8,12)'); print(pd.DataFrame(rows, columns=['cntb', 'n', 'P']).round(4).to_string(index=False))
