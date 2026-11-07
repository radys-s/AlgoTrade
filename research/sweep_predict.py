"""Is a big taker sweep predictable from the public tape, 5 ms ahead?

    uv run python -X utf8 -m research.sweep_predict round1 --out logs/sweeps/round1.md
    uv run python -X utf8 -m research.sweep_predict sample_v2 --out logs/sweeps/sample_v2.md

A "sweep" is an IOC insert with >= BIG lots. Side s = +1 means OUR BID is at risk (the IOC is a sell),
s = -1 our ask. All per-side features are expressed from the point of view of the side at risk, so
both sides are pooled. Label for row i: a big IOC attacking side s arrives in (t_i + 5 ms, t_i + 105 ms],
i.e. the window an order sent at row i can still influence. Nothing here looks ahead of row i except the label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.features import load_features, mid_ticks
from research.signals import (
    LATENCY_US,
    build_matrix,
    fit_ridge,
    future_mid_change,
    pearson,
    sum_last_us,
    value_at_or_before,
)

BIG = 300
HUGE = 1000
WIN_US = 100_000
OUT: list[str] = []


def say(line: str = '') -> None:
    """Print and collect a markdown line."""
    print(line, flush=True)
    OUT.append(line)


def table(header: list[str], rows: list[list[object]]) -> None:
    """Emit a markdown table."""
    say('| ' + ' | '.join(str(h) for h in header) + ' |')
    say('|' + '|'.join('---' for _ in header) + '|')
    for r in rows:
        say('| ' + ' | '.join(f'{x:.4g}' if isinstance(x, float) else str(x) for x in r) + ' |')
    say()


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via ranks (ties averaged)."""
    from scipy.stats import rankdata

    pos = y > 0
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ part 1-3: tape statistics


def tape_stats(t: pd.DataFrame, d: dict[str, np.ndarray]) -> None:
    """Sizes, clustering, repeats, round numbers, id patterns."""
    ts = d['ts_us'] / 1e6
    ty, sd, q = d['ev_type'], d['ev_side'], d['ev_qty'].astype(np.float64)
    ioc = ty == 1
    say('## 1. IOC arrivals')
    say()
    say(f'IOC inserts: {ioc.sum():,}; size quantiles q50/q90/q95/q99/q99.9 = '
        + ' / '.join(f'{np.quantile(q[ioc], p):.0f}' for p in (.5, .9, .95, .99, .999))
        + f'; >= {BIG}: {(ioc & (q >= BIG)).sum():,}; >= {HUGE}: {(ioc & (q >= HUGE)).sum():,}')
    say()
    rows = []
    for name, m in (('all IOC', ioc), (f'>= {BIG}', ioc & (q >= BIG)), (f'>= {HUGE}', ioc & (q >= HUGE))):
        for s in (+1, -1):
            tt = ts[m & (sd == -s)]  # attacker side -s hits side s
            dt = np.diff(tt)
            rows.append([name, 'bid hit' if s == 1 else 'ask hit', len(tt), f'{np.median(dt) * 1000:.1f}',
                         f'{dt.mean() * 1000:.1f}', f'{np.std(dt) / dt.mean():.2f}'])
    table(['set', 'side', 'n', 'median gap ms', 'mean gap ms', 'cv of gap (1 = Poisson)'], rows)

    # size autocorrelation (log size), per attacking side and over all IOCs
    ls_all = np.log(q[ioc])
    ac_all = [pearson(ls_all[:-k], ls_all[k:]) for k in range(1, 11)]
    ac_side = []
    for k in range(1, 11):
        v = []
        for s in (+1, -1):
            x = np.log(q[ioc & (sd == s)])
            v.append(pearson(x[:-k], x[k:]))
        ac_side.append(np.mean(v))
    say('Autocorrelation of log(size), lag 1..10:')
    table(['lag'] + list(range(1, 11)), [['all IOC'] + [f'{a:+.3f}' for a in ac_all], ['same side'] + [f'{a:+.3f}' for a in ac_side]])

    # clustering of big IOCs
    rng = np.random.default_rng(0)
    u = rng.uniform(ts[0], ts[-1] - 3, 300_000)
    rows = []
    for thr in (BIG, HUGE):
        for same in (True, False):
            for w in (0.1, 0.5, 2.0):
                num = den = 0
                unc = []
                for a in (+1, -1):
                    ta = ts[ioc & (sd == a) & (q >= thr)]
                    tb = ta if same else ts[ioc & (sd == -a) & (q >= thr)]
                    i0 = np.searchsorted(tb, ta, 'right')
                    i1 = np.searchsorted(tb, ta + w, 'right')
                    num += int((i1 > i0).sum())
                    den += len(ta)
                    j0 = np.searchsorted(tb, u, 'right')
                    j1 = np.searchsorted(tb, u + w, 'right')
                    unc.append((j1 > j0).mean())
                p, p0 = num / den, float(np.mean(unc))
                rows.append([f'>= {thr}', 'same side' if same else 'opposite side', f'{w * 1000:.0f} ms', f'{p:.4f}', f'{p0:.4f}', f'{p / p0:.2f}'])
    say('P(big IOC within window after another big IOC) vs unconditional window probability:')
    table(['size', 'next on', 'window', 'P(cond)', 'P(uncond)', 'lift'], rows)

    # same-size repeats
    rows = []
    for lo in (1, 100, BIG):
        m = ioc & (q >= lo)
        sizes = q[m]
        sides = sd[m]
        rep1 = np.mean([np.mean(q[m & (sd == s)][1:] == q[m & (sd == s)][:-1]) for s in (1, -1)])
        _, cnt = np.unique(sizes, return_counts=True)
        exp = float(((cnt / cnt.sum()) ** 2).sum())
        # any of last 10 same-side sizes
        hit = tot = 0
        for s in (1, -1):
            x = q[m & (sd == s)]
            ok = np.zeros(len(x) - 10, bool)
            for k in range(1, 11):
                ok |= x[10:] == x[10 - k:len(x) - k]
            hit += int(ok.sum())
            tot += len(ok)
        rows.append([f'>= {lo}', len(sizes), f'{rep1:.4f}', f'{exp:.4f}', f'{hit / tot:.4f}'])
        del sides
    say('Same exact size as the previous same-side IOC (observed vs iid expectation sum p^2), and equal to any of the previous 10:')
    table(['sizes', 'n', 'P(repeat prev)', 'iid expect', 'P(in last 10)'], rows)
    big = ioc & (q >= BIG)
    say(f'Round numbers among >= {BIG} IOCs: %50==0 {np.mean(q[big] % 50 == 0):.4f} (iid 0.02), %100==0 {np.mean(q[big] % 100 == 0):.4f} (0.01), '
        f'%10==0 {np.mean(q[big] % 10 == 0):.4f} (0.10), non-integer {np.mean(q[big] != np.round(q[big])):.4f}')
    vc = pd.Series(q[big]).value_counts()
    say('Most frequent big sizes: ' + ', '.join(f'{k:.0f}x{v}' for k, v in vc.head(8).items()))
    say()

    # ids
    say('## 3. Order-id patterns')
    say()
    ids = t.OrderId.to_numpy()
    ins = ty != 2
    ins_ids = ids[ins]
    seq = np.mean(np.diff(ins_ids) == 1)
    say(f'Insert ids strictly sequential (+1) for {seq:.4f} of consecutive inserts.')
    ins_idx = np.flatnonzero(ins)
    pos_in_ins = np.cumsum(ins) - 1  # insert ordinal of each row
    bigrows = np.flatnonzero(big)
    bo = pos_in_ins[bigrows]
    gaps = np.diff(bo)
    base_gap = len(ins_idx) / len(bigrows)
    say(f'Gap (in inserts) between consecutive big IOCs: median {np.median(gaps):.0f}, mean {gaps.mean():.0f} '
        f'(geometric/iid expectation mean {base_gap:.0f}); P(gap<=3) {np.mean(gaps <= 3):.4f} vs iid {1 - (1 - 1 / base_gap) ** 3:.4f}')
    prev = ins_idx[np.maximum(bo - 1, 0)]
    pt, ps, pq = ty[prev], sd[prev], q[prev]
    atk = sd[bigrows]
    rows = []
    for name, m in (('prev insert is IOC', pt == 1), ('prev insert is limit', pt == 0),
                    ('prev limit same side as sweep (bait?)', (pt == 0) & (ps == atk)),
                    ('prev limit opposite side (will be hit)', (pt == 0) & (ps == -atk)),
                    ('prev insert is big IOC', (pt == 1) & (pq >= BIG))):
        rows.append([name, f'{m.mean():.4f}'])
    base_lim = np.mean(ty[ins] == 0)
    say(f'Baseline P(insert is limit) = {base_lim:.4f}; baseline P(IOC) {1 - base_lim:.4f}')
    table(['immediately preceding insert of a big IOC', 'share'], rows)
    # previous TAPE row type
    pr = bigrows - 1
    rows = [[n, f'{np.mean(ty[pr] == k):.4f}', f'{np.mean(ty == k):.4f}'] for n, k in (('limit insert', 0), ('IOC', 1), ('cancel', 2))]
    table(['previous tape row of a big IOC', 'share', 'base share'], rows)
    # does sweep size match a recent cancel / limit size
    m20 = 0
    for r in bigrows[:20000]:
        w = slice(max(0, r - 20), r)
        if (q[w] == q[r]).any():
            m20 += 1
    rnd = rng.choice(np.flatnonzero(ioc & (q >= 20) & (q < BIG)), 20000, replace=False)
    r20 = sum(1 for r in rnd if (q[max(0, r - 20):r] == q[r]).any())
    say(f'Big IOC size equals the size of one of the previous 20 tape rows: {m20 / min(len(bigrows), 20000):.4f} '
        f'(small IOCs of 20-299 lots: {r20 / 20000:.4f}; sizes are rarer so a lower rate is expected)')
    # touch vs size
    ob = np.where(sd[bigrows] == 1, d['ask_qty'][pr, 0], d['bid_qty'][pr, 0]).astype(np.float64)
    say(f'Big IOC size / opposite touch qty before it: q25/q50/q75 = '
        + ' / '.join(f'{np.quantile(q[bigrows] / np.maximum(ob, 1), p):.2f}' for p in (.25, .5, .75))
        + f'; share that exceed the touch: {np.mean(q[bigrows] > ob):.3f}')
    say()


# ------------------------------------------------------------------ part 2/4: per-row precursors


def side_features(d: dict[str, np.ndarray], s: int) -> dict[str, np.ndarray]:
    """Precursor features for the side s at risk (+1 bid, -1 ask), state after each row."""
    ts = d['ts_us']
    ty, sd, q, px = d['ev_type'], d['ev_side'], d['ev_qty'].astype(np.float64), d['ev_px']
    own_px = d['bid_px'] if s == 1 else d['ask_px']
    prev_best = np.roll(own_px, 1)
    mine = sd == s
    behind = (px < prev_best) if s == 1 else (px > prev_best)
    tc = (ty == 2) & mine & (px == prev_best)
    tc_opp = (ty == 2) & (sd == -s) & (px == np.roll(d['ask_px'] if s == 1 else d['bid_px'], 1))
    ins_behind = (ty == 0) & mine & behind
    ins_touch = (ty == 0) & mine & ~behind
    probe = (ty == 1) & (sd == -s) & (q < BIG)
    bigp = (ty == 1) & (sd == -s) & (q >= BIG)
    hit = np.clip(-s * d['trade_vol'].astype(np.float64), 0, None)
    f: dict[str, np.ndarray] = {}
    for w in (50, 100, 500):
        f[f'cancel_touch_vol_{w}'] = sum_last_us(ts, q * tc, w * 1000)
        f[f'ins_behind_vol_{w}'] = sum_last_us(ts, q * ins_behind, w * 1000)
        f[f'ins_touch_vol_{w}'] = sum_last_us(ts, q * ins_touch, w * 1000)
        f[f'probe_n_{w}'] = sum_last_us(ts, probe.astype(np.float64), w * 1000)
        f[f'probe_vol_{w}'] = sum_last_us(ts, q * probe, w * 1000)
        f[f'hit_vol_{w}'] = sum_last_us(ts, hit, w * 1000)
        f[f'opp_cancel_touch_{w}'] = sum_last_us(ts, q * tc_opp, w * 1000)
    f['bigprev_500'] = sum_last_us(ts, bigp.astype(np.float64), 500_000)
    f['bigprev_2000'] = sum_last_us(ts, bigp.astype(np.float64), 2_000_000)
    f['bigprev_100'] = sum_last_us(ts, bigp.astype(np.float64), 100_000)
    own = (d['bid_qty'] if s == 1 else d['ask_qty'])[:, 0].astype(np.float64)
    opp = (d['ask_qty'] if s == 1 else d['bid_qty'])[:, 0].astype(np.float64)
    f['imb_risk'] = np.divide(opp - own, opp + own, out=np.zeros_like(own), where=(opp + own) > 0)
    f['log_own'] = np.log1p(own)
    f['log_opp'] = np.log1p(opp)
    f['cnt_own'] = (d['bid_cnt1'] if s == 1 else d['ask_cnt1']).astype(np.float64)
    f['spread'] = (d['ask_px'] - d['bid_px']).astype(np.float64)
    f['deplet_100'] = np.log1p(own) - np.log1p(value_at_or_before(ts, own, ts - 100_000).clip(0))
    mid = mid_ticks(d, ffill=True)
    dm = np.diff(mid, prepend=np.nan)
    ch = np.isfinite(dm) & (dm != 0)
    last = np.maximum.accumulate(np.where(ch, np.arange(len(ts)), -1))
    f['log_dt_mid'] = np.log1p((ts - np.where(last >= 0, ts[np.maximum(last, 0)], ts[0])) / 1000)
    f['recent_move_vs_side'] = -s * (mid - value_at_or_before(ts, mid, ts - 100_000))  # >0: mid moved toward our side's danger
    return f


def labels(d: dict[str, np.ndarray], s: int, thr: int) -> np.ndarray:
    """1 where a >= thr IOC attacking side s arrives in (t + 5 ms, t + 105 ms]."""
    ts = d['ts_us']
    m = (d['ev_type'] == 1) & (d['ev_side'] == -s) & (d['ev_qty'] >= thr)
    tb = ts[m]
    i = np.searchsorted(tb, ts + LATENCY_US, 'right')
    nxt = np.where(i < len(tb), tb[np.minimum(i, len(tb) - 1)], np.iinfo(np.int64).max)
    return (nxt <= ts + LATENCY_US + WIN_US).astype(np.int8)


def flag_table(name: str, x: np.ndarray, ys: dict[str, np.ndarray], valid: np.ndarray) -> list[list[object]]:
    """Lift / recall of 'x > thr' for a few thresholds (best direction by correlation sign)."""
    xv = x[valid]
    sign = 1.0 if pearson(xv[::7], ys['y300'][valid][::7].astype(float)) >= 0 else -1.0
    xv = sign * xv
    qs = np.unique(np.quantile(xv, [0.5, 0.9, 0.95, 0.99]))
    thrs = [-np.inf] if False else list(qs)
    if (xv > xv.min()).mean() < 0.3:  # sparse feature: also test "> 0"
        thrs = sorted(set([xv.min()] + thrs))
    rows = []
    for thr in thrs:
        fl = xv > thr
        if fl.mean() < 0.002 or fl.mean() > 0.6:
            continue
        r = [name + ('' if sign > 0 else ' (neg)'), f'{thr * sign:.3g}', f'{fl.mean():.3f}']
        for k in ('y300', 'y1000'):
            y = ys[k][valid]
            base = y.mean()
            p = y[fl].mean()
            r += [f'{p / base:.2f}', f'{y[fl].sum() / max(y.sum(), 1):.3f}']
        rows.append(r)
    return rows


def main() -> None:
    """Run the whole study for one dataset."""
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--skip-model', action='store_true')
    a = ap.parse_args()
    d = load_features(Path(f'data/features/{a.name}.npz'))
    t = pd.read_csv(f'data/{a.name}.csv')
    n = len(t)
    assert n == len(d['ts_us'])
    ts = d['ts_us']
    say(f'# Sweep anticipation: `{a.name}` ({n:,} rows, {ts[-1] / 1e6:.0f} s)')
    say()
    tape_stats(t, d)

    mid = mid_ticks(d)
    valid = np.isfinite(mid)
    chunk = (np.arange(n) * 8 // n).astype(np.int8)
    feats = {s: side_features(d, s) for s in (1, -1)}
    ys = {s: {'y300': labels(d, s, BIG), 'y1000': labels(d, s, HUGE)} for s in (1, -1)}
    # pool both sides
    names = list(feats[1])
    P = {k: np.concatenate([feats[1][k], feats[-1][k]]) for k in names}
    Y = {k: np.concatenate([ys[1][k], ys[-1][k]]) for k in ('y300', 'y1000')}
    V = np.concatenate([valid, valid])
    CH = np.concatenate([chunk, chunk])
    say('## 2. Precursors of a big IOC in the next 100 ms (pooled sides)')
    say()
    say(f'Rows with a two-sided book: {valid.sum():,} per side. Base rate P(>= {BIG} IOC hits the side within (5, 105] ms) = '
        f'{Y["y300"][V].mean():.4f}; for >= {HUGE}: {Y["y1000"][V].mean():.4f}. Lift = P(label | flag) / base; recall = share of positive rows flagged.')
    say()
    rows: list[list[object]] = []
    for k in names:
        rows += flag_table(k, P[k], Y, V)
    table(['feature', 'flag: x >', 'flag share', f'lift>={BIG}', f'recall>={BIG}', f'lift>={HUGE}', f'recall>={HUGE}'], rows)
    # AUC per feature
    sub = np.flatnonzero(V)[::5]
    aucs = sorted(((abs(auc(P[k][sub], Y['y300'][sub]) - 0.5) + 0.5, k, auc(P[k][sub], Y['y300'][sub])) for k in names), reverse=True)
    say('AUC per feature for the >= 300 label (best first): ' + ', '.join(f'{k} {v:.3f}' for _, k, v in aucs[:14]))
    say()

    if a.skip_model:
        finish(a)
        return
    model_part(d, t, feats, ys, valid, chunk, names)
    finish(a)


def finish(a: argparse.Namespace) -> None:
    """Write the markdown."""
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text('\n'.join(OUT) + '\n', encoding='utf-8')


# ------------------------------------------------------------------ part 4: score + comparison with the ridge


def prep(F: np.ndarray) -> np.ndarray:
    """Heavy-tailed volumes -> log1p; others untouched (columns are chosen by the caller)."""
    return F


def model_part(d, t, feats, ys, valid, chunk, names) -> None:  # noqa: ANN001
    """Walk-forward logistic score, event-level recall, ridge comparison and incremental IC."""
    from sklearn.linear_model import LogisticRegression

    n = len(d['ts_us'])
    ts = d['ts_us']
    say('## 4. Sweep-risk score, walk-forward (8 chunks; train on chunks < k with 1 s embargo, test chunk k = 1..7)')
    say()
    vol_like = [k for k in names if any(k.startswith(p) for p in ('cancel_touch', 'ins_behind', 'ins_touch', 'probe', 'hit_vol', 'opp_cancel'))]
    cnt_like = ['bigprev_100', 'bigprev_500', 'bigprev_2000']

    def design(s: int, cols: list[str]) -> np.ndarray:
        cols_arr = []
        for k in cols:
            x = feats[s][k]
            cols_arr.append(np.log1p(x) if (k in vol_like or k in cnt_like) else x)
        return np.nan_to_num(np.column_stack(cols_arr)).astype(np.float32)

    def fit_pred(cols: list[str], train_chunks: np.ndarray, test_mask_by_side: dict[int, np.ndarray], max_rows: int = 600_000):  # noqa: ANN202
        """Fit pooled logistic on train rows, return the per-side decision scores for the rows in test_mask."""
        Xs, ysx = [], []
        for s in (1, -1):
            idx = np.flatnonzero(valid & train_chunks)
            idx = idx[:: max(1, len(idx) // (max_rows // 2))]
            Xs.append(design(s, cols)[idx])
            ysx.append(ys[s]['y300'][idx])
        X = np.vstack(Xs)
        yv = np.concatenate(ysx)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=300)
        lr.fit((X - mu) / sd, yv)
        out = {s: ((design(s, cols) - mu) / sd) @ lr.coef_[0] + lr.intercept_[0] for s in (1, -1)}
        return out, lr.coef_[0], mu, sd

    def folds(cols: list[str]):  # noqa: ANN202
        """Yield (k, scores per side over all rows, train mask) for k = 1..7."""
        for k in range(1, 8):
            start = ts[np.searchsorted(chunk, k)]
            train = (chunk < k) & (ts + 1_000_000 < start)
            sc, coef, mu, sd = fit_pred(cols, train, {})
            yield k, sc, coef, train

    # greedy forward selection on chunks 0..2 (a 4th chunk is never used for choosing)
    sel_rows = {s: np.flatnonzero(valid & (chunk <= 2))[::6] for s in (1, -1)}
    cand = [k for k in names if k not in ('spread', 'log_opp', 'cnt_own')]

    def quick_auc(cols: list[str]) -> float:
        """AUC of a logistic fit on chunks 0-2, in-sample (selection only)."""
        X = np.vstack([design(s, cols)[sel_rows[s]] for s in (1, -1)])
        yv = np.concatenate([ys[s]['y300'][sel_rows[s]] for s in (1, -1)])
        mu, sdv = X.mean(0), X.std(0) + 1e-9
        lr = LogisticRegression(max_iter=200).fit((X - mu) / sdv, yv)
        return auc(lr.decision_function((X - mu) / sdv), yv)

    chosen: list[str] = []
    trace = []
    for _ in range(5):
        best = max(((quick_auc(chosen + [c]), c) for c in cand if c not in chosen))
        chosen.append(best[1])
        trace.append(f'{best[1]} ({best[0]:.3f})')
    say('Greedy forward selection on chunks 0-2 (cumulative AUC): ' + ' > '.join(trace))
    say()

    ridge_pred = ridge_walk(d, chunk)
    mid_ff = mid_ticks(d, ffill=True)
    y100 = future_mid_change(ts, mid_ff, 100_000)

    results = {}
    for label, cols in (('5-feature score', chosen), ('all-feature score', names)):
        sc_test = {s: np.full(n, np.nan) for s in (1, -1)}
        coefs = []
        for k, sc, coef, train in folds(cols):
            m = chunk == k
            for s in (1, -1):
                sc_test[s][m] = sc[s][m]
            coefs.append(coef)
        results[label] = (sc_test, np.mean(coefs, axis=0), cols)

    test = valid & (chunk >= 1)
    base_tbl = []
    for label, (sc, coef, cols) in results.items():
        pooled_s = np.concatenate([sc[1][test], sc[-1][test]])
        pooled_y = np.concatenate([ys[1]['y300'][test], ys[-1]['y300'][test]])
        pooled_y1k = np.concatenate([ys[1]['y1000'][test], ys[-1]['y1000'][test]])
        base_tbl.append([label, f'{auc(pooled_s, pooled_y):.3f}', f'{auc(pooled_s, pooled_y1k):.3f}'])
        if label.startswith('5'):
            say('Mean standardized logistic coefficients of the 5-feature score: ' + ', '.join(f'{c} {v:+.2f}' for c, v in zip(cols, coef)))
            say()
    say('Pooled OOS AUC (chunks 1-7):')
    table(['model', f'AUC >= {BIG} in 100 ms', f'AUC >= {HUGE}'], base_tbl)

    # operating points
    sc5 = results['5-feature score'][0]
    dts = np.diff(ts, append=ts[-1]).astype(np.float64)  # time each state persists
    # decision row for each big sweep: last row with ts <= t_big - 5 ms
    ioc = d['ev_type'] == 1
    ridge_flag = {1: ridge_pred < -0.15, -1: ridge_pred > 0.15}
    rows = []
    for thr in (BIG, HUGE):
        for s in (1, -1):
            pass
    pooled_sc = np.concatenate([sc5[1][test], sc5[-1][test]])
    for q_ in (0.01, 0.02, 0.05, 0.10):
        cut = np.quantile(pooled_sc, 1 - q_)
        for thr in (BIG, HUGE):
            tp = fn = 0
            tp_r = tp_u = 0
            for s in (1, -1):
                ev = np.flatnonzero(ioc & (d['ev_side'] == -s) & (d['ev_qty'] >= thr) & (chunk >= 1))
                j = np.searchsorted(ts, ts[ev] - LATENCY_US, 'right') - 1
                ok = (j >= 0) & valid[np.maximum(j, 0)] & (chunk[np.maximum(j, 0)] >= 1)
                j = j[ok]
                fl = sc5[s][j] > cut
                rf = ridge_flag[s][j]
                tp += int(fl.sum())
                fn += int((~fl).sum())
                tp_r += int(rf.sum())
                tp_u += int((fl | rf).sum())
            tot = tp + fn
            rows.append([f'top {q_ * 100:.0f}% of rows', f'>= {thr}', tot, f'{tp / tot:.3f}', f'{tp_r / tot:.3f}', f'{tp_u / tot:.3f}'])
    # time share flagged
    cut_by_q = {q_: np.quantile(pooled_sc, 1 - q_) for q_ in (0.01, 0.02, 0.05, 0.10)}
    share = []
    for q_, cut in cut_by_q.items():
        tw = 0.0
        tsum = 0.0
        for s in (1, -1):
            m = test
            tw += float(dts[m & (sc5[s] > cut)].sum())
            tsum += float(dts[m].sum())
        share.append([f'top {q_ * 100:.0f}% of rows', f'{tw / tsum:.3f}'])
    rf_t = sum(float(dts[test & ridge_flag[s]].sum()) for s in (1, -1)) / (2 * float(dts[test].sum()))
    say('Event-level recall: share of big sweeps whose pre-arrival state (last row <= t - 5 ms) is flagged (score cut from OOS pooled score quantile). '
        'Ridge flag = existing |pred| > 0.15 ticks against that side.')
    table(['score cut', 'sweeps', 'n', 'recall: score', 'recall: ridge', 'recall: score OR ridge'], rows)
    say(f'Share of TIME with the side flagged by the score: ' + ', '.join(f'{a} = {b}' for a, b in share) + f'; by the ridge rule: {rf_t:.3f}.')
    say()

    # precision (row level) at top-q
    rows = []
    for q_, cut in cut_by_q.items():
        yy = np.concatenate([ys[1]['y300'][test & (sc5[1] > cut)], ys[-1]['y300'][test & (sc5[-1] > cut)]])
        yall = np.concatenate([ys[1]['y300'][test], ys[-1]['y300'][test]])
        yy1 = np.concatenate([ys[1]['y1000'][test & (sc5[1] > cut)], ys[-1]['y1000'][test & (sc5[-1] > cut)]])
        yall1 = np.concatenate([ys[1]['y1000'][test], ys[-1]['y1000'][test]])
        rows.append([f'top {q_ * 100:.0f}%', f'{yy.mean():.3f}', f'{yy.mean() / yall.mean():.1f}', f'{yy.sum() / yall.sum():.3f}',
                     f'{yy1.mean():.3f}', f'{yy1.mean() / yall1.mean():.1f}', f'{yy1.sum() / yall1.sum():.3f}'])
    say(f'Row-level precision / lift / recall of "a >= {BIG} IOC hits this side in (5,105] ms" (and >= {HUGE}), 5-feature score:')
    table(['flag', 'precision', 'lift', 'recall', f'prec>={HUGE}', 'lift', 'recall'], rows)

    # relation with the ridge
    r_rows = []
    for label, (sc, coef, cols) in results.items():
        sig = (sc[-1] - sc[1])  # ask hit -> price up ; bid hit -> price down
        m = test & np.isfinite(sig) & np.isfinite(ridge_pred) & np.isfinite(y100)
        r_rows.append([label, f'{pearson(sig[m], ridge_pred[m]):+.3f}', f'{pearson(sig[m], y100[m]):+.3f}', f'{pearson(ridge_pred[m], y100[m]):+.3f}'])
        # both sides' marginal: partial correlation of sig with y after ridge
        res = y100[m] - np.polyval(np.polyfit(ridge_pred[m], y100[m], 1), ridge_pred[m])
        r_rows[-1].append(f'{pearson(sig[m], res):+.3f}')
    say('Signed score (score_ask_hit - score_bid_hit) vs the ridge forecast and the 100 ms mid move y:')
    table(['model', 'corr(score, ridge)', 'IC(score, y)', 'IC(ridge, y)', 'corr(score, y - fit(ridge))'], r_rows)

    # incremental IC: ridge on 23 features vs plus score features (walk-forward)
    X, fn = build_matrix(d)
    ok_rows = valid & np.isfinite(y100) & np.isfinite(X).all(1)
    sig5 = {}
    # in-sample scores for training chunks come from the fold's own fit: refit per fold
    ic_base, ic_aug, ic_aug_raw = [], [], []
    cols = chosen
    for k in range(1, 8):
        start = ts[np.searchsorted(chunk, k)]
        train = (chunk < k) & (ts + 1_000_000 + 105_000 < start) & ok_rows
        tm = (chunk == k) & ok_rows
        sc, _, _, _ = fit_pred(cols, (chunk < k) & (ts + 1_000_000 < start), {})
        extra = np.column_stack([sc[-1] - sc[1], sc[-1], sc[1]]).astype(np.float32)
        raw = np.column_stack([np.log1p(feats[-1][c]) - np.log1p(feats[1][c]) if c in vol_like or c in cnt_like else feats[-1][c] - feats[1][c] for c in cols]).astype(np.float32)
        ytr, yte = y100[train], y100[tm]
        m0 = fit_ridge(X[train], ytr, fn)
        ic_base.append(pearson(m0.predict(X[tm]), yte))
        Xa = np.hstack([X, extra])
        m1 = fit_ridge(Xa[train], ytr, fn + ['sig', 'sig_ask', 'sig_bid'])
        ic_aug.append(pearson(m1.predict(Xa[tm]), yte))
        Xr = np.hstack([X, extra, raw])
        m2 = fit_ridge(Xr[train], ytr, fn + ['sig', 'sig_ask', 'sig_bid'] + [f'raw_{c}' for c in cols])
        ic_aug_raw.append(pearson(m2.predict(Xr[tm]), yte))
        say(f'  fold {k}: IC base {ic_base[-1]:+.4f}  + score {ic_aug[-1]:+.4f}  + score + raw precursors {ic_aug_raw[-1]:+.4f}')
        del Xa, Xr
    say()
    table(['model for the 100 ms mid move', 'mean OOS IC (7 folds)', 'folds improved vs base'],
          [['ridge, 23 features', f'{np.mean(ic_base):+.4f}', '-'],
           ['+ sweep score (signed, ask, bid)', f'{np.mean(ic_aug):+.4f}', f'{sum(b > a for a, b in zip(ic_base, ic_aug))}/7'],
           ['+ score + raw side-difference precursors', f'{np.mean(ic_aug_raw):+.4f}', f'{sum(b > a for a, b in zip(ic_base, ic_aug_raw))}/7']])

    cost_part(d, valid, chunk, results['5-feature score'][0], ridge_flag, y100)


def ridge_walk(d: dict[str, np.ndarray], chunk: np.ndarray) -> np.ndarray:
    """Walk-forward OOS ridge forecast of the 100 ms mid move (same recipe as the strategy's model)."""
    ts = d['ts_us']
    X, fn = build_matrix(d)
    mid_ff = mid_ticks(d, ffill=True)
    y = future_mid_change(ts, mid_ff, 100_000)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    out = np.full(len(ts), np.nan)
    for k in range(1, 8):
        start = ts[np.searchsorted(chunk, k)]
        train = (chunk < k) & (ts + 105_000 < start) & ok
        m = fit_ridge(X[train], y[train], fn)
        te = (chunk == k) & np.isfinite(X).all(1)
        out[te] = m.predict(X[te])
    return out


def cost_part(d, valid, chunk, sc5, ridge_flag, y100) -> None:  # noqa: ANN001
    """What a flagged window costs/saves: maker 1 s markout of non-sweep vs sweep fills inside flagged windows."""
    ts = d['ts_us']
    mid = mid_ticks(d, ffill=True)
    mk = value_at_or_before(ts, mid, ts + 1_000_000) - np.where(np.isfinite(d['last_trade_px']), d['last_trade_px'], np.nan)
    tv = d['trade_vol'].astype(np.float64)
    is_big = (d['ev_type'] == 1) & (d['ev_qty'] >= BIG)
    say('## 5. Economics of a flag (maker view, 1 s markout, per lot, ticks)')
    say()
    say('Fill price approximated by the row\'s last trade price (exact for single-level fills). Maker markout = s * (mid(t+1s) - price) for the side hit.')
    say()
    test = valid & (chunk >= 1)
    pooled = np.concatenate([sc5[1][test], sc5[-1][test]])
    rows = []
    for q_ in (0.02, 0.05):
        cut = np.quantile(pooled, 1 - q_)
        for s in (1, -1):
            hitrow = (-s * tv > 0) & np.isfinite(mk)
            lots = np.where(hitrow, np.abs(tv), 0.0)
            pnl_pl = np.where(hitrow, s * mk, 0.0)
            for nm, mask in (('sweep rows', is_big), ('other rows', ~is_big)):
                l = lots * mask
                c_l = np.concatenate([[0], np.cumsum(l)])
                c_p = np.concatenate([[0], np.cumsum(l * pnl_pl)])
                i0 = np.searchsorted(ts, ts + LATENCY_US, 'right')
                i1 = np.searchsorted(ts, ts + LATENCY_US + WIN_US, 'right')
                wl = c_l[i1] - c_l[i0]
                wp = c_p[i1] - c_p[i0]
                for fnm, fm in ((f'score top {q_ * 100:.0f}%', test & (sc5[s] > cut)), ('not flagged', test & (sc5[s] <= cut)),
                                ('ridge flag', test & ridge_flag[s])):
                    if fm.sum() == 0:
                        continue
                    rows.append([s, nm, fnm, f'{wl[fm].mean():.2f}', f'{wp[fm].sum() / max(wl[fm].sum(), 1):+.3f}', f'{wp[fm].mean():+.3f}'])
    table(['side', 'fills of', 'state', 'lots hit per flagged row (5-105 ms)', 'markout per lot', 'PnL per row'], rows)


if __name__ == '__main__':
    main()
    sys.exit(0)
