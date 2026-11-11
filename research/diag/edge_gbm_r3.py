"""Does a wide, raw event-sequence feature set carry information the lean 22-feature 200 ms ridge misses?

    .venv/Scripts/python.exe -X utf8 -m research.diag.edge_gbm_r3 [round3] > out.txt
Rows sampled every 5th event. Target z = robust mid at t+5ms+200ms minus the current mid (as pred_rfit_r3).
Split by time: train 60 %, valid 15 %, test 25 %. Models: lean ridge, GBM on lean, GBM on wide, ridge on lean +
top new features. Edge = mean sign(pred) * target in the top q of |pred| (equal trigger rate) and at |pred| >= spread/2+0.2.
All features are causal: built from events <= i (windows via searchsorted on ts <= t).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from research.signals import LATENCY_US, fit_ridge, value_before
from research.diag import pred_r3, pred_rfit_r3

H = 200
STEP = 5


def win_count(ts, mask, w_us):
    """Number of events with mask in (t - w, t] for every row (causal)."""
    c = np.cumsum(mask.astype(np.float64))
    j = np.searchsorted(ts, ts - w_us, side='right') - 1
    return c - np.where(j >= 0, c[np.maximum(j, 0)], 0.0)


def win_sum(ts, x, w_us):
    c = np.cumsum(x)
    j = np.searchsorted(ts, ts - w_us, side='right') - 1
    return c - np.where(j >= 0, c[np.maximum(j, 0)], 0.0)


def since(ts, mask, cap_ms=5000.0):
    n = len(ts)
    last = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
    v = np.where(last >= 0, (ts - ts[np.maximum(last, 0)]) / 1000.0, cap_ms)
    return np.log1p(np.minimum(v, cap_ms))


def lag_val(ts, x, lag_us):
    """x at the last row with ts <= t - lag."""
    j = np.searchsorted(ts, ts - lag_us, side='right') - 1
    return x[np.maximum(j, 0)]


def wide(name, d, cols, rm, mid_ff):
    df = pd.read_csv(f'data/{name}.csv')
    ts = d['ts_us'].astype(np.int64)
    n = len(ts)
    op = df.Operation.values
    # cancels carry only the OrderId: take side / price / volume from the order's insert
    insm = df.Operation.values == 'Insert'
    ref = df.loc[insm, ['OrderId', 'Side', 'Price', 'Volume']].drop_duplicates('OrderId').set_index('OrderId')
    cm = ~insm
    for c in ('Side', 'Price', 'Volume'):
        df.loc[cm, c] = df.loc[cm, 'OrderId'].map(ref[c]).values
    sd = np.where(df.Side.values == 'B', 1.0, -1.0)
    vol = np.nan_to_num(df.Volume.values.astype(float))  # cancels carry no volume
    px = df.Price.values.astype(float)
    ins = op == 'Insert'
    can = op == 'Cancel'
    ioc = ins & (df.Lifespan.values != 'G')
    gins = ins & ~ioc
    bb = d['bid_px'].astype(float); ba = d['ask_px'].astype(float)
    pbb = np.r_[bb[0], bb[:-1]]; pba = np.r_[ba[0], ba[:-1]]
    pm = np.r_[mid_ff[0], mid_ff[:-1]]
    W = {}
    # last k events
    lv = np.log1p(vol)
    off = np.clip(sd * (px - pm), -10, 10)  # aggressiveness: + = toward the other side
    for k in range(3):
        sh = lambda x: np.r_[np.zeros(k), x[:n - k]] if k else x
        W[f'e{k}_gins'] = sh(sd * gins)
        W[f'e{k}_can'] = sh(sd * can)
        W[f'e{k}_ioc_lsz'] = sh(sd * ioc * lv)
        W[f'e{k}_off'] = sh(np.where(ins, off, 0.0))
    # inter-arrival per type / side
    for nm, m in (('gins', gins), ('can', can), ('ioc', ioc)):
        W[f'since_{nm}_b'] = since(ts, m & (sd > 0))
        W[f'since_{nm}_a'] = since(ts, m & (sd < 0))
        W[f'since_{nm}_d'] = W[f'since_{nm}_b'] - W[f'since_{nm}_a']
    # insert size vs rolling median
    medv = pd.Series(np.where(gins, vol, np.nan)).rolling(5000, min_periods=50).median().ffill().fillna(10).values
    rel = np.where(gins, np.log(vol / np.maximum(medv, 1)), 0.0)
    W['ins_relsz_signed'] = pd.Series(sd * rel).where(gins).ffill().fillna(0).values
    W['bigins_50'] = win_sum(ts, sd * gins * (rel > 1.0), 50_000)
    # counts in windows per side, depth
    at_touch_b = (sd > 0) & (px >= pbb); at_touch_a = (sd < 0) & (px <= pba)
    touch = at_touch_b | at_touch_a
    for w in (20, 50, 200):
        wu = w * 1000
        W[f'can_imb{w}'] = win_sum(ts, sd * can, wu)
        W[f'cantouch_imb{w}'] = win_sum(ts, sd * (can & touch), wu)
        W[f'gins_imb{w}'] = win_sum(ts, sd * gins, wu)
        W[f'ginstouch_imb{w}'] = win_sum(ts, sd * (gins & touch), wu)
        W[f'ioc_vol{w}'] = win_sum(ts, sd * ioc * vol, wu)
        W[f'nev{w}'] = np.log1p(win_count(ts, np.ones(n, bool), wu))
    # inside-spread inserts (improving) and deep cancels
    impr = gins & (((sd > 0) & (px > pbb)) | ((sd < 0) & (px < pba)))
    W['impr_imb50'] = win_sum(ts, sd * impr, 50_000)
    deep = can & ~touch
    W['candeep_imb50'] = win_sum(ts, sd * deep, 50_000)
    # id gaps (unseen inserts)
    oid = df.OrderId.values.astype(np.int64)
    ins_id = np.where(ins, oid, -1)
    prev_max = np.maximum.accumulate(np.r_[ins_id[0], ins_id[:-1]])
    gap = np.where(ins, np.clip(oid - prev_max - 1, 0, 50), 0).astype(float)
    W['idgap50'] = win_sum(ts, gap, 50_000)
    W['idgap200'] = win_sum(ts, gap, 200_000)
    # IOC sweep depth and size (beyond the touch)
    sweep = np.where(ioc, np.clip(np.where(sd > 0, px - pba, pbb - px), -5, 10), 0.0)
    W['ioc_sweep_last'] = pd.Series(np.where(ioc, sd * (sweep + 1), np.nan)).ffill().fillna(0).values
    W['ioc_sweep50'] = win_sum(ts, sd * ioc * np.maximum(sweep + 1, 0), 50_000)
    W['since_ioc'] = since(ts, ioc)
    # queue depletion at the touch
    lqb = np.log1p(d['bid_qty'][:, 0].astype(float)); lqa = np.log1p(d['ask_qty'][:, 0].astype(float))
    for w in (10, 50):
        dqb = lqb - lag_val(ts, lqb, w * 1000); dqa = lqa - lag_val(ts, lqa, w * 1000)
        same_b = bb == lag_val(ts, bb, w * 1000); same_a = ba == lag_val(ts, ba, w * 1000)
        W[f'dq_imb{w}'] = np.where(same_b, dqb, 0) - np.where(same_a, dqa, 0)
    W['since_bchg'] = since(ts, bb != pbb); W['since_achg'] = since(ts, ba != pba)
    W['since_chg_d'] = W['since_bchg'] - W['since_achg']
    sp = cols['spread']
    W['spread_lag50'] = lag_val(ts, sp, 50_000)
    W['spread_max100'] = pd.Series(sp).rolling(200, min_periods=1).max().values  # ~events, causal
    W['hole_lag50'] = lag_val(ts, cols['hole'], 50_000)
    W['hole_abs'] = np.abs(cols['hole'])
    # order size per order at the touch, deeper imbalance
    bq = d['bid_qty'].astype(float); aq = d['ask_qty'].astype(float)
    W['avgord_b'] = np.log1p(bq[:, 0] / np.maximum(d['bid_cnt1'], 1)); W['avgord_a'] = np.log1p(aq[:, 0] / np.maximum(d['ask_cnt1'], 1))
    W['avgord_d'] = W['avgord_b'] - W['avgord_a']
    for L in (2, 5):
        sb = bq[:, :L].sum(1); sa = aq[:, :L].sum(1)
        W[f'imb_l{L}'] = (sb - sa) / np.maximum(sb + sa, 1)
    W['imb_l2only'] = (bq[:, 1] - aq[:, 1]) / np.maximum(bq[:, 1] + aq[:, 1], 1)
    # momentum
    for lag in (20, 50, 100, 300, 1000):
        W[f'mom{lag}'] = rm - lag_val(ts, rm, lag * 1000)
    for lag in (20, 50, 200):
        W[f'fsmom{lag}'] = cols['fs70'] - lag_val(ts, cols['fs70'], lag * 1000)
    W['fs20_m_fs300'] = cols['fs20'] - cols['fs300']
    W['microprice'] = (bb * aq[:, 0] + ba * bq[:, 0]) / np.maximum(aq[:, 0] + bq[:, 0], 1) - mid_ff
    for k in W:
        bad = ~np.isfinite(W[k])
        if bad.mean() > 0.01:
            print(f'# {k} non-finite {bad.mean():.3f}', flush=True)
        W[k] = np.where(bad, 0.0, W[k]).astype(np.float64)
    return W


def edges(p, y, sp):
    out = {}
    for q in (0.01, 0.03, 0.06):
        t = np.abs(p) >= np.quantile(np.abs(p), 1 - q)
        out[f'top{q}'] = (np.sign(p[t]) * y[t]).mean()
    hit = np.abs(p) >= sp / 2 + 0.2
    out['n_thr'] = hit.mean()
    out['e_thr'] = (np.sign(p[hit]) * y[hit]).mean() if hit.any() else np.nan
    out['sum_thr'] = (np.sign(p[hit]) * y[hit] - sp[hit] / 2).sum() if hit.any() else np.nan
    return out


def fmt(dct):
    return ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in dct.items())


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'round3'
    t0 = time.time()
    ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
    z = np.load(f'data/features/{name}.npz'); d = {k: z[k] for k in z.files}
    W = wide(name, d, cols, rm, mid_ff)
    print(f'# features built {time.time() - t0:.0f}s, {len(W)} new', flush=True)
    e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
    y = e - mid_ff
    y[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
    lean = list(pred_r3.LEAN)
    new = list(W)
    allc = {**{k: cols[k] for k in lean}, **W}
    idx = np.arange(0, len(ts), STEP)
    ok = np.isfinite(y[idx])
    for k in allc:
        ok &= np.isfinite(allc[k][idx])
    idx = idx[ok]
    for k in lean:
        print(f'# lean {k} finite {np.isfinite(allc[k]).mean():.3f}', flush=True) if np.isfinite(allc[k]).mean() < 0.99 else None
    span = ts[-1] - ts[0]
    t60 = ts[0] + 0.60 * span; t75 = ts[0] + 0.75 * span
    hz = (H * 1000 + LATENCY_US)
    tr = idx[ts[idx] + hz <= t60]; va = idx[(ts[idx] > t60) & (ts[idx] + hz <= t75)]; te = idx[ts[idx] > t75]
    print(f'# rows train {len(tr)} valid {len(va)} test {len(te)}', flush=True)
    M = lambda feats, ii: np.column_stack([allc[f][ii] for f in feats]).astype(np.float32)
    yt, yv, yte = y[tr], y[va], y[te]
    sp_te = cols['spread'][te]; sp_va = cols['spread'][va]

    def ridge(feats, ii=tr, yy=None):
        return fit_ridge(M(feats, ii), y[ii] if yy is None else yy, feats, alpha=10.0, clip_q=0.0)

    rl = ridge(lean)
    p_rl = rl.predict(M(lean, te)); p_rl_v = rl.predict(M(lean, va))
    print(f'ridge_lean   valid corr {np.corrcoef(p_rl_v, yv)[0,1]:.4f} test corr {np.corrcoef(p_rl, yte)[0,1]:.4f} | {fmt(edges(p_rl, yte, sp_te))}', flush=True)
    feats_all = lean + new
    res = {}
    for tag, feats in (('gbm_lean', lean), ('gbm_wide', feats_all)):
        g = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=None,
                                          n_iter_no_change=30, scoring='neg_mean_squared_error', random_state=0)
        # sklearn early stopping needs its own split: pass valid via a manual loop instead
        g.set_params(early_stopping=False, max_iter=50, warm_start=True)
        best = (-1, 0); Xtr = M(feats, tr); Xva = M(feats, va)
        for it in range(50, 801, 50):
            g.set_params(max_iter=it); g.fit(Xtr, yt)
            c = np.corrcoef(g.predict(Xva), yv)[0, 1]
            if c > best[0]:
                best = (c, it)
            elif it - best[1] >= 150:
                break
        g.set_params(max_iter=best[1], warm_start=False); g.fit(Xtr, yt)
        p = g.predict(M(feats, te))
        res[tag] = (g, feats, p)
        print(f'{tag:12s} iters {best[1]} valid corr {best[0]:.4f} test corr {np.corrcoef(p, yte)[0,1]:.4f} | {fmt(edges(p, yte, sp_te))}', flush=True)
        # ridge-residual information: corr of gbm pred with the ridge residual
        print(f'   corr(gbm, y - ridge) test {np.corrcoef(p, yte - p_rl)[0,1]:.4f}; blend 0.5 corr {np.corrcoef(0.5*p+0.5*p_rl, yte)[0,1]:.4f}', flush=True)
    g, feats, p = res['gbm_wide']
    sub = te[:: max(1, len(te) // 60000)]
    pi = permutation_importance(g, M(feats, sub), y[sub], n_repeats=2, random_state=0, scoring='r2')
    order = np.argsort(-pi.importances_mean)
    print('# permutation importance (gbm_wide, test, r2 drop), top 40:', flush=True)
    for j in order[:40]:
        print(f'  {feats[j]:22s} {pi.importances_mean[j]*1e4:8.1f}e-4 {"NEW" if feats[j] in new else ""}', flush=True)
    # each new feature: corr with the ridge residual on train (linear info), single-feature add to the ridge (valid)
    resid_tr = yt - rl.predict(M(lean, tr))
    rows = []
    for f in new:
        x = allc[f][tr]
        c = np.corrcoef(x, resid_tr)[0, 1] if x.std() > 0 else 0.0
        rows.append((abs(c), f, c))
    rows.sort(reverse=True)
    print('# new features by |corr with lean-ridge residual| (train):', flush=True)
    for a, f, c in rows[:20]:
        print(f'  {f:22s} {c:+.4f}', flush=True)
    newrank = [feats[j] for j in order if feats[j] in new][:6]
    linrank = [f for _, f, _ in rows[:6]]
    for tag, add in (('perm_top3', newrank[:3]), ('perm_top6', newrank), ('lin_top3', linrank[:3]), ('lin_top6', linrank)):
        fl = lean + add
        m = ridge(fl)
        pp = m.predict(M(fl, te)); pv = m.predict(M(fl, va))
        print(f'ridge+{tag:9s} {add} valid {np.corrcoef(pv, yv)[0,1]:.4f} test {np.corrcoef(pp, yte)[0,1]:.4f} | {fmt(edges(pp, yte, sp_te))}', flush=True)
    # tree count / depth for a pure-Python runtime estimate
    nt = len(g._predictors); nodes = sum(len(pr[0].nodes) for pr in g._predictors)
    print(f'# gbm_wide trees {nt}, total nodes {nodes}, avg depth ~{np.log2(max(nodes / max(nt, 1), 2)):.1f}', flush=True)
    print(f'# done {time.time() - t0:.0f}s', flush=True)


GBX = ['mom50', 'mom100', 'mom300', 'microprice', 'imb_l2only', 'since_chg_d']


def fit_gbx(extras):
    """Ridge on lean + extras, round3 + mirror first 75 %, H 200, stride 3 (as rfit 19); holdout = last 25 % of round3."""
    lean = list(pred_r3.LEAN)
    feats = lean + extras
    Xs, ys, ho = [], [], None
    for name in ('round3', 'round3_mirror'):
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
        z = np.load(f'data/features/{name}.npz'); d = {k: z[k] for k in z.files}
        W = wide(name, d, cols, rm, mid_ff)
        allc = {**{k: cols[k] for k in lean}, **{k: W[k] for k in extras}}
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        y = e - mid_ff
        y[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        ok = np.isfinite(y)
        for k in feats:
            ok &= np.isfinite(allc[k])
        t75 = ts[0] + 0.75 * (ts[-1] - ts[0])
        tri = np.where(ok & (ts + LATENCY_US + H * 1000 <= t75))[0][::3]
        Xs.append(np.column_stack([allc[f][tri] for f in feats])); ys.append(y[tri])
        if name == 'round3':
            hoi = np.where(ok & (ts > t75))[0][::5]
            ho = (np.column_stack([allc[f][hoi] for f in feats]), y[hoi], cols['spread'][hoi])
    X = np.vstack(Xs).astype(np.float32); y = np.concatenate(ys)
    for tag, fl in (('lean', lean), ('lean+gbx', feats)):
        ii = [feats.index(f) for f in fl]
        m = fit_ridge(X[:, ii], y, fl, alpha=10.0, clip_q=0.0)
        p = m.predict(ho[0][:, ii].astype(np.float32))
        print(f'{tag:9s} holdout corr {np.corrcoef(p, ho[1])[0,1]:.4f} | {fmt(edges(p, ho[1], ho[2]))}', flush=True)
        wr = m.coef / m.scale
        c0 = float(m.intercept - (wr * m.mean).sum())
        print(f'  c0={round(c0, 7)} w={tuple(round(float(x), 7) for x in wr)}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fit':
        fit_gbx(sys.argv[2].split(',') if len(sys.argv) > 2 else GBX)
    else:
        main()
