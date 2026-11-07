"""Order-age features: the cancel clock of every resting order is set at insertion (i.i.d. lifetime), so the
age profile of the touch queue predicts how much of it cancels soon.

    uv run --with numba python -X utf8 -m research.features_age build round1     # -> logs/age/round1_age.npz
    uv run --with numba python -X utf8 -m research.features_age mech round1

Per-event, per-side raw columns (touch = best level, L2 = next non-empty level within 20 ticks), see COLS.
All of them are a plain loop over the touch level; O(1) variants with age buckets are in the report.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

OUT = Path('logs/age')
BIN_US = 10_000          # age bins of 10 ms
NB = 800                 # 8 s
COLS = ['exp200', 'q', 'haz', 'young', 'old', 'oldest_ms', 'exp200_l2', 'q_l2', 'real200', 'cnt', 'exp500', 'real500']
P = 20000


def load_tape(name: str):
    df = pd.read_csv(f'data/{name}.csv')
    op = np.where(df.Operation.values == 'Cancel', 2, np.where(df.Lifespan.values == 'F', 1, 0)).astype(np.int8)
    oid = df.OrderId.values.astype(np.int64)
    side = np.where(df.Side.values == 'B', 1, np.where(df.Side.values == 'A', -1, 0)).astype(np.int8)
    vol = df.Volume.fillna(0).values.astype(np.int32)
    px = df.Price.fillna(0).values.astype(np.int32)
    ts = np.rint(df.Time.values * 1e6).astype(np.int64)
    return ts, op, oid, side, vol, px


def lifetimes(ts, op, oid):
    base = oid.min()
    n_ids = int(oid.max() - base + 1)
    ins_t = np.full(n_ids, -1, np.int64)
    ins = op != 2
    ins_t[oid[ins] - base] = ts[ins]
    can_t = np.full(n_ids, -1, np.int64)
    c = op == 2
    can_t[oid[c] - base] = ts[c]
    return base, ins_t, can_t


def survival_tables(ins_t, can_t, ts_limit):
    """Empirical survival of lifetimes on a 10 ms grid, fitted only on cancels before ts_limit."""
    m = (ins_t >= 0) & (can_t >= 0) & (can_t < ts_limit)
    life = (can_t[m] - ins_t[m]).astype(np.int64)
    cnt = np.bincount(np.minimum(life // BIN_US, NB + 299), minlength=NB + 300)
    S = 1.0 - np.concatenate([[0], np.cumsum(cnt)[:-1]]) / len(life)   # S[b] = P(life >= b*10ms)
    S = np.maximum(S, 1e-5)
    b = np.arange(NB)
    g200 = 1 - S[b + 20] / S[b]
    g500 = 1 - S[b + 50] / S[b]
    haz = -np.log(S[b + 1] / S[b]) / 0.01
    return g200, g500, haz, S


@njit(cache=True)
def _side_feats(si, best, t, head, nxt, rem, tins, can_t, g200, g500, haz, out, row):
    sgn = 1 if si == 0 else -1
    lv = best
    for k in range(1, 21):
        px2 = best - sgn * k
        if px2 <= 0 or px2 >= 20000:
            break
        if head[si, px2] >= 0:
            lv = px2
            break
    for pass_ in range(2):
        if pass_ == 0:
            px_ = best
        else:
            px_ = lv
            if lv == best:
                break
        o = head[si, px_]
        e2 = 0.0
        e5 = 0.0
        q = 0.0
        hz = 0.0
        yg = 0.0
        od = 0.0
        oldest = 0.0
        cnt = 0.0
        r2 = 0.0
        r5 = 0.0
        while o >= 0:
            a = t - tins[o]
            b = a // 10000
            if b >= 800:
                b = 799
            s = rem[o]
            q += s
            e2 += s * g200[b]
            e5 += s * g500[b]
            hz += haz[b]
            if a < 150000:
                yg += s
            if a > 650000:
                od += s
            if a > oldest:
                oldest = a
            cnt += 1
            ct = can_t[o]
            if ct > t:
                if ct - t <= 200000:
                    r2 += s
                if ct - t <= 500000:
                    r5 += s
            o = nxt[o]
        if pass_ == 0:
            out[row, si, 0] = e2
            out[row, si, 1] = q
            out[row, si, 2] = hz
            out[row, si, 3] = yg / q if q > 0 else 0.0
            out[row, si, 4] = od / q if q > 0 else 0.0
            out[row, si, 5] = oldest / 1000.0
            out[row, si, 8] = r2
            out[row, si, 9] = cnt
            out[row, si, 10] = e5
            out[row, si, 11] = r5
        else:
            out[row, si, 6] = e2
            out[row, si, 7] = q


@njit(cache=True)
def replay(ts, op, idx, side, vol, px, can_t, n_ids, g200, g500, haz):
    n = len(ts)
    head = np.full((2, P), -1, np.int32)
    tail = np.full((2, P), -1, np.int32)
    lvn = np.zeros((2, P), np.int32)
    nxt = np.full(n_ids, -1, np.int32)
    prv = np.full(n_ids, -1, np.int32)
    rem = np.zeros(n_ids, np.int32)
    tins = np.zeros(n_ids, np.int64)
    opx = np.zeros(n_ids, np.int32)
    osd = np.zeros(n_ids, np.int8)
    alive = np.zeros(n_ids, np.bool_)
    out = np.full((n, 2, 12), np.nan, np.float32)
    bb = 0
    ba = 0
    for i in range(n):
        t = ts[i]
        o = op[i]
        if o == 0:
            k = idx[i]
            si = 0 if side[i] == 1 else 1
            p = px[i]
            rem[k] = vol[i]
            tins[k] = t
            opx[k] = p
            osd[k] = side[i]
            alive[k] = True
            nxt[k] = -1
            prv[k] = tail[si, p]
            if tail[si, p] >= 0:
                nxt[tail[si, p]] = k
            else:
                head[si, p] = k
            tail[si, p] = k
            lvn[si, p] += 1
            if si == 0:
                if bb == 0 or p > bb:
                    bb = p
            else:
                if ba == 0 or p < ba:
                    ba = p
        elif o == 1:
            left = vol[i]
            if side[i] == 1:   # buy sweeps asks
                while left > 0 and ba > 0 and ba <= px[i]:
                    m = head[1, ba]
                    while m >= 0 and left > 0:
                        f = min(left, rem[m])
                        left -= f
                        rem[m] -= f
                        nm = nxt[m]
                        if rem[m] == 0:
                            alive[m] = False
                            lvn[1, ba] -= 1
                            head[1, ba] = nm
                            if nm >= 0:
                                prv[nm] = -1
                            else:
                                tail[1, ba] = -1
                        m = nm
                    if lvn[1, ba] == 0:
                        while ba < P and lvn[1, ba] == 0:
                            ba += 1
                        if ba >= P:
                            ba = 0
            else:
                while left > 0 and bb > 0 and bb >= px[i]:
                    m = head[0, bb]
                    while m >= 0 and left > 0:
                        f = min(left, rem[m])
                        left -= f
                        rem[m] -= f
                        nm = nxt[m]
                        if rem[m] == 0:
                            alive[m] = False
                            lvn[0, bb] -= 1
                            head[0, bb] = nm
                            if nm >= 0:
                                prv[nm] = -1
                            else:
                                tail[0, bb] = -1
                        m = nm
                    if lvn[0, bb] == 0:
                        while bb > 0 and lvn[0, bb] == 0:
                            bb -= 1
        else:
            k = idx[i]
            if k >= 0 and k < n_ids and alive[k]:
                si = 0 if osd[k] == 1 else 1
                p = opx[k]
                a = prv[k]
                b = nxt[k]
                if a >= 0:
                    nxt[a] = b
                else:
                    head[si, p] = b
                if b >= 0:
                    prv[b] = a
                else:
                    tail[si, p] = a
                lvn[si, p] -= 1
                alive[k] = False
                if lvn[si, p] == 0:
                    if si == 0 and p == bb:
                        while bb > 0 and lvn[0, bb] == 0:
                            bb -= 1
                    elif si == 1 and p == ba:
                        while ba < P and lvn[1, ba] == 0:
                            ba += 1
                        if ba >= P:
                            ba = 0
        if bb > 0 and ba > 0:
            _side_feats(0, bb, t, head, nxt, rem, tins, can_t, g200, g500, haz, out, i)
            _side_feats(1, ba, t, head, nxt, rem, tins, can_t, g200, g500, haz, out, i)
    return out


def build(name: str):
    t0 = time.time()
    ts, op, oid, side, vol, px = load_tape(name)
    base, ins_t, can_t = lifetimes(ts, op, oid)
    n_ids = len(ins_t)
    g200, g500, haz, S = survival_tables(ins_t, can_t, ts[len(ts) // 8])
    idx = (oid - base).astype(np.int32)
    out = replay(ts, op, idx, side, vol, px, can_t, n_ids, g200, g500, haz)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'{name}_age.npz', raw=out, g200=g200, g500=g500, haz=haz, S=S)
    d = np.load(f'data/features/{name}.npz')
    ok = np.isfinite(out[:, 0, 1]) & (d['bid_px'] > 0) & (d['ask_px'] > 0)
    pb = (out[ok, 0, 1] == d['bid_qty'][ok, 0]).mean()
    pa = (out[ok, 1, 1] == d['ask_qty'][ok, 0]).mean()
    print(name, 'rows', len(ts), 'npz rows', len(d['ts_us']), 'touch qty parity bid/ask', pb, pa, f'{time.time() - t0:.0f}s', flush=True)


AGE_FEATS = ['ag_expd_n', 'ag_frac_d', 'ag_frac_s', 'ag_imb_after', 'ag_haz_d', 'ag_young_d', 'ag_old_d', 'ag_oldest_d',
             'ag_l2_expd_n', 'ag_imb_after12', 'ag_exp500_d_n']


def age_features(name: str) -> np.ndarray:
    """(n, 11) float32 from the raw columns. Positive = bullish (bid side safer than ask side)."""
    r = np.load(OUT / f'{name}_age.npz')['raw'].astype(np.float64)
    b, a = r[:, 0, :], r[:, 1, :]
    eb, ea, qb, qa = b[:, 0], a[:, 0], b[:, 1], a[:, 1]
    with np.errstate(all='ignore'):
        f = {}
        f['ag_expd_n'] = (ea - eb) / (qb + qa)
        fb, fa = eb / qb, ea / qa
        f['ag_frac_d'] = fa - fb
        f['ag_frac_s'] = fa + fb
        rb, ra = qb - eb, qa - ea
        f['ag_imb_after'] = (rb - ra) / (rb + ra)
        f['ag_haz_d'] = a[:, 2] - b[:, 2]
        f['ag_young_d'] = b[:, 3] - a[:, 3]
        f['ag_old_d'] = a[:, 4] - b[:, 4]
        f['ag_oldest_d'] = np.log1p(b[:, 5]) - np.log1p(a[:, 5])
        f['ag_l2_expd_n'] = (a[:, 6] - b[:, 6]) / (b[:, 7] + a[:, 7] + 1)
        rb2, ra2 = rb + (b[:, 7] - b[:, 6]), ra + (a[:, 7] - a[:, 6])
        f['ag_imb_after12'] = (rb2 - ra2) / (rb2 + ra2)
        f['ag_exp500_d_n'] = (a[:, 10] - b[:, 10]) / (qb + qa)
    X = np.column_stack([f[k] for k in AGE_FEATS])
    X[~np.isfinite(X)] = np.nan
    return X.astype(np.float32)


def mechanism(name: str):
    r = np.load(OUT / f'{name}_age.npz')['raw'].astype(np.float64)
    res = {}
    ok = np.isfinite(r[:, 0, 0]) & np.isfinite(r[:, 1, 0])
    for h, ec, rc in (('200ms', 0, 8), ('500ms', 10, 11)):
        y = np.concatenate([r[ok, 0, rc], r[ok, 1, rc]])
        e = np.concatenate([r[ok, 0, ec], r[ok, 1, ec]])
        q = np.concatenate([r[ok, 0, 1], r[ok, 1, 1]])
        c = np.concatenate([r[ok, 0, 9], r[ok, 1, 9]])

        def r2(cols):
            A = np.column_stack(cols + [np.ones(len(y))])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(1 - ((y - A @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        res[h] = {'mean_real': float(y.mean()), 'R2 q only': r2([q]), 'R2 q+count': r2([q, c]), 'R2 exp alone': r2([e]),
                  'R2 q+cnt+exp': r2([q, c, e]), 'corr exp,real': float(np.corrcoef(e, y)[0, 1])}
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build':
        for nm in sys.argv[2:]:
            build(nm)
    elif cmd == 'mech':
        for nm in sys.argv[2:]:
            print(nm, mechanism(nm))


def mech_norm(name: str):
    r = np.load(OUT / f'{name}_age.npz')['raw'].astype(np.float64)
    ok = np.isfinite(r[:, 0, 0]) & np.isfinite(r[:, 1, 0]) & (r[:, 0, 1] > 0) & (r[:, 1, 1] > 0)
    out = {}
    for h, ec, rc in (('200ms', 0, 8), ('500ms', 10, 11)):
        q = np.concatenate([r[ok, 0, 1], r[ok, 1, 1]])
        y = np.concatenate([r[ok, 0, rc], r[ok, 1, rc]]) / q
        e = np.concatenate([r[ok, 0, ec], r[ok, 1, ec]]) / q
        c = np.concatenate([r[ok, 0, 9], r[ok, 1, 9]])
        def r2(cols):
            A = np.column_stack(cols + [np.ones(len(y))])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(1 - ((y - A @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out[h] = {'frac mean': float(y.mean()), 'R2 count only': r2([1 / c, np.log(q)]), 'R2 exp_frac alone': r2([e]),
                  'R2 cnt+logq+exp_frac': r2([1 / c, np.log(q), e]), 'corr': float(np.corrcoef(e, y)[0, 1])}
    return out


def run_eval(names=('round1', 'sample_v2'), horizons=(100, 1000)):
    import pickle
    from research.features import load_features, mid_ticks
    from research.features_v2 import candidates, CAND, summarize
    from research.signals import build_matrix, fit_ridge, future_mid_change, oos_stats, walk_forward
    sets = {}
    for nm in names:
        d = load_features(Path(f'data/features/{nm}.npz'))
        ts = d['ts_us']
        n = len(ts)
        rows = np.arange(0, n, max(1, -(-n // 3_000_000)))
        X0, n0 = build_matrix(d, rows=rows)
        c = candidates(d)
        Xc = np.column_stack([c[k][rows] for k in CAND])
        Xa = age_features(nm)[rows]
        X = np.hstack([X0, Xc, Xa])
        mid_ff = mid_ticks(d, ffill=True)
        Y = np.column_stack([future_mid_change(ts, mid_ff, h * 1000)[rows] for h in horizons])
        keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        sets[nm] = {'X': X[keep], 'Y': Y[keep].astype(np.float64), 'ts_us': ts[rows][keep],
                    'chunk': (rows[keep] * 8 // n).astype(np.int8), 'names': n0 + CAND + AGE_FEATS}
        print(nm, keep.sum(), flush=True)
    names_all = sets[names[0]]['names']
    nb, nc = 23, len(CAND)
    base = list(range(nb))
    v2 = list(range(nb + nc))
    ag = list(range(nb + nc, nb + nc + len(AGE_FEATS)))
    cfg = {'baseline23': base}
    for k, a in enumerate(AGE_FEATS):
        cfg['+' + a] = base + [ag[k]]
    cfg['+AGE_all'] = base + ag
    cfg['+AGE_core(expd,imb_after,frac_s,haz_d)'] = base + [ag[0], ag[3], ag[2], ag[4]]
    cfg['v2_all'] = v2
    cfg['v2_all+AGE'] = v2 + ag
    res = {}
    for stem, ds in sets.items():
        for hi, h in enumerate(horizons):
            y = ds['Y'][:, hi]
            for cn, idx in cfg.items():
                folds, _ = walk_forward(ds['X'][:, idx], y, ds['ts_us'], ds['chunk'], [names_all[i] for i in idx], h * 1000, use_hgb=False, verbose=False)
                res[(stem, h, cn)] = summarize(folds)
                ics, m, pos, e10, e1 = res[(stem, h, cn)]
                print(f'{stem} h{h} {cn:40s} ic {m:+.4f} pos {pos}/{len(ics)} e10 {e10:.4f} e1 {e1:.4f}', flush=True)
            for cn in ('baseline23', '+AGE_all', 'v2_all', 'v2_all+AGE'):
                idx = cfg[cn]
                folds, _ = walk_forward(ds['X'][:, idx], y, ds['ts_us'], ds['chunk'], [names_all[i] for i in idx], h * 1000, use_hgb=True, verbose=False)
                res[(stem, h, 'HGB ' + cn)] = summarize(folds, 'hgb')
                ics, m, pos, e10, e1 = res[(stem, h, 'HGB ' + cn)]
                print(f'{stem} h{h} HGB {cn:36s} ic {m:+.4f} pos {pos}/{len(ics)} e10 {e10:.4f} e1 {e1:.4f}', flush=True)
    a, b = sets[names[0]], sets[names[1]]
    for hi, h in enumerate(horizons):
        for cn in ('baseline23', '+AGE_all', 'v2_all', 'v2_all+AGE'):
            idx = cfg[cn]
            m = fit_ridge(a['X'][:, idx], a['Y'][:, hi], [names_all[i] for i in idx])
            st = oos_stats(m.predict(b['X'][:, idx]), b['Y'][:, hi], m.predict(a['X'][::10, idx]))
            res[('cross', h, cn)] = st
            print(f'cross h{h} {cn}: ic {st["ic"]:+.4f} e10 {st["edge10"]:.4f} e1 {st["edge1"]:.4f}', flush=True)
    pickle.dump(res, open(OUT / 'results.pkl', 'wb'))


if __name__ == '__main__' and sys.argv[1] == 'eval':
    run_eval()
if __name__ == '__main__' and sys.argv[1] == 'mechn':
    for nm in sys.argv[2:]:
        print(nm, mech_norm(nm))
