"""Diagnostic only: where does a strategy's pnl come from on round4 vs round3 (flip anatomy, regime card, 10-min blocks).

    python -X utf8 -m research.diag.anat_r4 [strategy] [--cache DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.diag import flips_r3 as F  # noqa: E402
from research.features import load_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STRAT = 'strategies/mm_r3_gru3.py'
TAPES = ['round4', 'round3']


def get_run(strat, tape, cache):
    p = Path(cache) / f'anat_{Path(strat).stem}_{tape}.npz'
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    d = F.record(ROOT / strat, ROOT / f'data/{tape}.csv', {})
    np.savez(p, **d)
    return d


def mid_series(f):
    bb = f['bid_px'].astype(float)
    ba = f['ask_px'].astype(float)
    ok = (bb > 0) & (ba > 0) & (ba - bb <= 3)
    m = F.ffill(np.where(ok, (bb + ba) / 2, np.nan))
    first = np.argmax(ok)
    m[:first] = m[first]
    return m, ok


def pnl_series(d):
    """Row-aligned running pnl marked at the clean mid (fill edge + inventory * dmid)."""
    ts = d['ts']
    inv = d['inv']
    bb = d['bb'].astype(float)
    ba = d['ba'].astype(float)
    ok = (bb > 0) & (ba > 0) & (ba - bb <= 3)
    first = np.argmax(ok)
    rm = np.nan_to_num(F.ffill(np.where(ok, (bb + ba) / 2, np.nan)), nan=0.0)
    rm[:first] = rm[first]
    fi = np.maximum(np.searchsorted(ts, d['f_ts'], side='right') - 1, 0)
    edge = (rm[fi] - d['f_px']) * d['f_sign'] * d['f_qty']
    e_row = np.bincount(fi, weights=edge, minlength=len(ts))
    drm = np.diff(rm, append=rm[-1])
    return ts, np.cumsum(inv * drm + e_row), rm


def regime(tape):
    f = load_features(ROOT / f'data/features/{tape}.npz')
    ts = f['ts_us']
    T = (ts[-1] - ts[0]) / 1e6
    m, ok = mid_series(f)
    print(f'--- {tape}: {len(ts)} rows, {T:.0f} s')
    et = f['ev_type']
    print(f'event rate/s: limit {np.sum(et == 0) / T:.1f}  ioc {np.sum(et == 1) / T:.1f}  cancel {np.sum(et == 2) / T:.1f}  trading rows {np.sum(f["trade_vol"] != 0) / T:.2f}')
    sp = (f['ask_px'] - f['bid_px'])[ok]
    print('spread dist (ticks) 1/2/3: ' + ' '.join(f'{np.mean(sp == k):.3f}' for k in (1, 2, 3)) + f'  mean {sp.mean():.2f}; two-sided&<=3 share {ok.mean():.3f}')
    for lag in (1, 10, 60):
        g = np.arange(ts[0], ts[-1], lag * 1_000_000)
        mm = m[np.searchsorted(ts, g, side='right') - 1]
        dm = np.diff(mm)
        print(f'mid vol {lag:>2}s: std {dm.std():.3f} ticks  mean|d| {np.abs(dm).mean():.3f}')
    bq = f['bid_qty'][:, 0][ok]
    aq = f['ask_qty'][:, 0][ok]
    q = np.concatenate([bq, aq])
    print(f'touch qty: mean {q.mean():.1f} p10/50/90 {np.percentile(q, 10):.0f}/{np.percentile(q, 50):.0f}/{np.percentile(q, 90):.0f}')
    tv = np.abs(f['trade_vol'][f['trade_vol'] != 0])
    iq = f['ev_qty'][et == 1]
    print(f'trade size per event: mean {tv.mean():.1f} p50 {np.percentile(tv, 50):.0f} p90 {np.percentile(tv, 90):.0f} p99 {np.percentile(tv, 99):.0f};  IOC qty mean {iq.mean():.1f} p90 {np.percentile(iq, 90):.0f}')
    g = np.arange(ts[0], ts[-1], 600_000_000)
    mm = m[np.searchsorted(ts, g, side='right') - 1]
    print('drift per 10 min (ticks): ' + ' '.join(f'{x:+.0f}' for x in np.diff(mm)))


def deciles(tape, d, R):
    ts, cum, rm = pnl_series(d)
    t0 = ts[0]
    T = ts[-1] - t0
    print(f'--- {tape}: pnl by time decile (mark-to-mid {cum[-1]:,.0f}, sim {d["pnl"]:,.0f}); flips segpnl too')
    print(f'{"dec":>3}{"pnl":>10}{"flips":>7}{"seg":>10}{"mid drift":>10}')
    ed = np.linspace(0, T, 11) + t0
    ix = np.searchsorted(ts, ed, side='right') - 1
    ix[0] = 0
    for k in range(10):
        a, b = ix[k], ix[k + 1]
        m = (R['t'] >= ed[k]) & (R['t'] < ed[k + 1] + (1 if k == 9 else 0))
        print(f'{k:>3}{cum[b] - cum[a]:>10,.0f}{m.sum():>7}{R["seg_pnl"][m].sum():>10,.0f}{rm[b] - rm[a]:>+10.0f}')


def blocks(tape, d, R):
    ts, cum, rm = pnl_series(d)
    t0 = ts[0]
    nb = int((ts[-1] - t0) // 600_000_000) + 1
    ed = t0 + np.arange(nb + 1) * 600_000_000
    ix = np.clip(np.searchsorted(ts, ed, side='right') - 1, 0, len(ts) - 1)
    rows = []
    for k in range(nb):
        a, b = ix[k], ix[k + 1]
        if b <= a:
            continue
        seg = rm[a:b + 1]
        g = seg[::max((b - a) // 600, 1)]
        vol = np.std(np.diff(g))
        pnl = cum[b] - cum[a]
        m = (R['t'] >= ed[k]) & (R['t'] < ed[k + 1])
        ws = m & (R['seg_dur'] < 0.5)
        rows.append((k, pnl, rm[b] - rm[a], vol, np.abs(np.diff(seg)).sum(), m.sum(), R['seg_pnl'][ws].sum(), ws.sum()))
    print(f'--- {tape}: per 10-min block')
    print(f'{"blk":>3}{"pnl":>10}{"drift":>8}{"vol(600pt)":>11}{"path":>8}{"flips":>7}{"whipsawN":>9}{"whipsawPnl":>11}')
    for r in rows:
        print(f'{r[0]:>3}{r[1]:>10,.0f}{r[2]:>+8.0f}{r[3]:>11.3f}{r[4]:>8.0f}{r[5]:>7}{r[7]:>9}{r[6]:>11,.0f}')
    A = np.array(rows)
    for nm, j in (('|drift|', 2), ('vol', 3), ('path', 4), ('flips', 5)):
        x = np.abs(A[:, j]) if j == 2 else A[:, j]
        print(f'  corr(pnl, {nm}) = {np.corrcoef(A[:, 1], x)[0, 1]:+.2f}', end='')
    print()
    return A


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    strat = args[0] if args else STRAT
    cache = Path(__file__).parent
    if '--cache' in sys.argv:
        cache = Path(sys.argv[sys.argv.index('--cache') + 1])
    for tape in TAPES:
        regime(tape)
    for tape in TAPES:
        d = get_run(strat, tape, cache)
        R, pnl = F.analyse(d)
        N = len(R['t'])
        print(f'\n===== {tape}: pnl {pnl:,.0f} flips {N} fills {len(d["f_ts"])} lots {d["f_qty"].sum()} taker-lot share {(d["f_qty"] * d["f_ioc"]).sum() / d["f_qty"].sum():.2f}')
        tk = R['tfrac'] >= 0.8
        ps_ = R['tfrac'] <= 0.05
        mx = ~tk & ~ps_
        sd = R['seg_dur']
        pd_ = R['ps_dir']
        spr = R['spread']
        cats = [('all', np.ones(N, bool)), ('taker', tk), ('passive', ps_), ('mixed', mx),
                ('whipsaw seg<0.5s', sd < 0.5), ('seg 0.5-2s', (sd >= 0.5) & (sd < 2)), ('seg 2-10s', (sd >= 2) & (sd < 10)), ('seg>=10s', sd >= 10),
                ('taker whipsaw', tk & (sd < 0.5)), ('taker non-whipsaw', tk & (sd >= 0.5)),
                ('spread<=1', spr <= 1), ('spread==2', spr == 2), ('spread>=3', spr >= 3),
                ('taker ps<0.5', tk & (pd_ < 0.5)), ('taker ps 0.5-1', tk & (pd_ >= 0.5) & (pd_ < 1)), ('taker ps>=1', tk & (pd_ >= 1)),
                ('taker drift against>=5', tk & (R['drift_dir'] <= -5)), ('taker drift with>=5', tk & (R['drift_dir'] >= 5)),
                ('long flips (new dir +)', R['dirn'] > 0), ('short flips (new dir -)', R['dirn'] < 0)]
        F.table(R, cats)
        deciles(tape, d, R)
        blocks(tape, d, R)


if __name__ == '__main__':
    main()
