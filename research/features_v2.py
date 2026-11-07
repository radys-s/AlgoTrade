"""Candidate features on top of research.signals' 23 (v2), plus the walk-forward comparison.

    uv run python -X utf8 -m research.features_v2 build data/features/round1.npz
    uv run python -X utf8 -m research.features_v2 eval  data/features/round1.npz data/features/sample_v2.npz

Everything is derived from the existing per-event npz columns (no new replay needed): row i = state
after tape row i, only rows <= i are used. Each candidate is O(1) per event online (a deque window sum
or a "last value" register). "bullish positive" means a positive value predicts a mid rise.

CANDIDATES (touch = best price of the side; prev = state after the previous row):
  dep_net_100 / dep_net_500  lots removed from the ask touch minus from the bid touch in the last
        100 / 500 ms (trades AND cancels; removal = drop of touch lots at an unchanged best price, or
        the whole previous touch when the best price moved away from the market).
  cxl_net_200  lots cancelled at the ask touch minus at the bid touch, last 200 ms
        (cancel row with ev_px == previous best price of that side).
  ins_net_200  lots inserted (limit) at/inside the bid touch minus at/inside the ask touch, 200 ms.
  cnt_tot_log  log1p(orders at the bid touch + orders at the ask touch).
  avgsz_imb    imbalance of the mean order size at the touch: (b1/nb - a1/na)/(b1/nb + a1/na).
  imb_l2       imbalance of level-2 lots only: (b2 - a2)/(b2 + a2).
  nmid_1s      number of mid changes in (t - 1 s, t].
  last_ioc_slog  side * log1p(volume) of the most recent IOC row (+1 buy), carried forward.
  last_tr_share  sign * min(|trade vol| / touch lots of the hit side before it, 3) of the latest
        trade event (+ buyer-initiated), carried forward.
  spread_dur_log log1p(ms since the spread last changed).
  flow_rel_1s  signed traded volume of (t - 1 s, t] / (bid touch + ask touch lots + 1).
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np

from research.features import load_features, mid_ticks
from research.signals import (
    _imbalance, build_matrix, fit_ridge, future_mid_change, oos_stats, sum_last_us, walk_forward,
)

OUT_DIR = Path('logs/features_v2')
CAND = ['dep_net_100', 'dep_net_500', 'cxl_net_200', 'ins_net_200', 'cnt_tot_log', 'avgsz_imb', 'imb_l2',
        'nmid_1s', 'last_ioc_slog', 'last_tr_share', 'spread_dur_log', 'flow_rel_1s']


def _prev(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = 0
    out[1:] = x[:-1]
    return out


def _ffill(mask: np.ndarray, vals: np.ndarray, default: float = 0.0) -> np.ndarray:
    """vals at the last row where mask is true (default before the first)."""
    n = len(mask)
    last = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
    return np.where(last >= 0, vals[np.maximum(last, 0)], default)


def candidates(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ts = d['ts_us']
    n = len(ts)
    bpx, apx = d['bid_px'].astype(np.int64), d['ask_px'].astype(np.int64)
    bq1, aq1 = d['bid_qty'][:, 0].astype(np.float64), d['ask_qty'][:, 0].astype(np.float64)
    bc, ac = d['bid_cnt1'].astype(np.float64), d['ask_cnt1'].astype(np.float64)
    bpx_p, apx_p, bq_p, aq_p = _prev(bpx), _prev(apx), _prev(bq1), _prev(aq1)
    et, es, eq, ep = d['ev_type'], d['ev_side'], d['ev_qty'].astype(np.float64), d['ev_px'].astype(np.int64)
    flow = d['trade_vol'].astype(np.float64)
    out: dict[str, np.ndarray] = {}

    rem_b = np.where((bpx == bpx_p) & (bpx > 0), np.maximum(bq_p - bq1, 0), np.where((bpx < bpx_p) & (bpx_p > 0), bq_p, 0.0))
    rem_a = np.where((apx == apx_p) & (apx > 0), np.maximum(aq_p - aq1, 0), np.where(((apx > apx_p) | (apx == 0)) & (apx_p > 0), aq_p, 0.0))
    dep = rem_a - rem_b
    out['dep_net_100'] = sum_last_us(ts, dep, 100_000)
    out['dep_net_500'] = sum_last_us(ts, dep, 500_000)

    cxl = np.where((et == 2) & (es == -1) & (ep == apx_p) & (apx_p > 0), eq, 0.0) - np.where(
        (et == 2) & (es == 1) & (ep == bpx_p) & (bpx_p > 0), eq, 0.0)
    out['cxl_net_200'] = sum_last_us(ts, cxl, 200_000)
    ins = np.where((et == 0) & (es == 1) & (ep >= bpx_p) & (bpx_p > 0), eq, 0.0) - np.where(
        (et == 0) & (es == -1) & (ep <= apx_p) & (apx_p > 0) & (ep > 0), eq, 0.0)
    out['ins_net_200'] = sum_last_us(ts, ins, 200_000)

    out['cnt_tot_log'] = np.log1p(bc + ac)
    sb = np.divide(bq1, bc, out=np.zeros(n), where=bc > 0)
    sa = np.divide(aq1, ac, out=np.zeros(n), where=ac > 0)
    out['avgsz_imb'] = _imbalance(sb, sa)
    out['imb_l2'] = _imbalance(d['bid_qty'][:, 1], d['ask_qty'][:, 1])

    mid_ff = mid_ticks(d, ffill=True)
    dm = np.diff(mid_ff, prepend=np.nan)
    out['nmid_1s'] = sum_last_us(ts, (np.isfinite(dm) & (dm != 0)).astype(np.float64), 1_000_000)

    ioc = et == 1
    out['last_ioc_slog'] = _ffill(ioc, es * np.log1p(eq))
    tr = flow != 0
    touch_before = np.where(flow > 0, aq_p, bq_p)
    share = np.sign(flow) * np.minimum(np.abs(flow) / np.maximum(touch_before, 1.0), 3.0)
    out['last_tr_share'] = _ffill(tr, share)

    spr = (apx - bpx).astype(np.float64)
    sp_changed = np.concatenate([[True], spr[1:] != spr[:-1]])
    out['spread_dur_log'] = np.log1p((ts - _ffill(sp_changed, ts.astype(np.float64), float(ts[0]))) / 1000)
    out['flow_rel_1s'] = sum_last_us(ts, flow, 1_000_000) / (bq1 + aq1 + 1)

    invalid = np.isnan(mid_ticks(d))
    return {k: np.where(invalid, np.nan, out[k]).astype(np.float32) for k in CAND}


def build_dataset_v2(d, horizons_ms, max_rows=3_000_000, n_chunks=8):
    ts = d['ts_us']
    n = len(ts)
    stride = max(1, math.ceil(n / max_rows))
    rows = np.arange(0, n, stride)
    X0, names0 = build_matrix(d, rows=rows)
    c = candidates(d)
    Xc = np.column_stack([c[k][rows] for k in CAND])
    X = np.hstack([X0, Xc])
    mid_ff = mid_ticks(d, ffill=True)
    Y = np.column_stack([future_mid_change(ts, mid_ff, h * 1000)[rows] for h in horizons_ms])
    keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return {'X': X[keep], 'Y': Y[keep].astype(np.float64), 'ts_us': ts[rows][keep],
            'chunk': (rows[keep] * n_chunks // n).astype(np.int8), 'names': names0 + CAND}


def summarize(folds, model='ridge'):
    ics = np.array([f.stats[model]['ic'] for f in folds])
    e10 = np.nanmean([f.stats[model]['edge10'] for f in folds])
    e1 = np.nanmean([f.stats[model]['edge1'] for f in folds])
    return ics, float(ics.mean()), int((ics > 0).sum()), float(e10), float(e1)


def run_eval(paths, horizons=(100, 1000), hgb=True):
    base_n = 23
    sets = {}
    for p in paths:
        d = load_features(Path(p))
        t = time.time()
        ds = build_dataset_v2(d, horizons)
        print(p, len(ds['ts_us']), 'rows', f'{time.time() - t:.0f}s', flush=True)
        sets[Path(p).stem] = ds
        del d
    names = next(iter(sets.values()))['names']
    base_idx = list(range(base_n))
    configs = {'baseline23': base_idx}
    for k, c in enumerate(CAND):
        configs['+' + c] = base_idx + [base_n + k]
    configs['+ALL'] = list(range(len(names)))
    res = {}
    for stem, ds in sets.items():
        for hi, h in enumerate(horizons):
            y = ds['Y'][:, hi]
            for cname, idx in configs.items():
                folds, _ = walk_forward(ds['X'][:, idx], y, ds['ts_us'], ds['chunk'], [names[i] for i in idx], h * 1000,
                                        use_hgb=False, verbose=False)
                res[(stem, h, cname)] = summarize(folds)
                ics, m, pos, e10, e1 = res[(stem, h, cname)]
                print(f'{stem} h{h} {cname:18s} ic {m:+.4f} pos {pos}/{len(ics)} e10 {e10:.4f} e1 {e1:.4f}', flush=True)
            if hgb:
                for cname in ('baseline23', '+ALL'):
                    idx = configs[cname]
                    folds, _ = walk_forward(ds['X'][:, idx], y, ds['ts_us'], ds['chunk'], [names[i] for i in idx], h * 1000,
                                            use_hgb=True, verbose=False)
                    res[(stem, h, 'HGB ' + cname)] = summarize(folds, 'hgb')
                    ics, m, pos, e10, e1 = res[(stem, h, 'HGB ' + cname)]
                    print(f'{stem} h{h} HGB {cname:14s} ic {m:+.4f} pos {pos}/{len(ics)} e10 {e10:.4f} e1 {e1:.4f}', flush=True)
    if len(sets) == 2:
        (s1, a), (s2, b) = list(sets.items())
        for hi, h in enumerate(horizons):
            for cname in ('baseline23', '+ALL'):
                idx = configs[cname]
                m = fit_ridge(a['X'][:, idx], a['Y'][:, hi], [names[i] for i in idx])
                st = oos_stats(m.predict(b['X'][:, idx]), b['Y'][:, hi], m.predict(a['X'][::10, idx]))
                res[('cross', h, cname)] = st
                print(f'cross {s1}->{s2} h{h} {cname}: ic {st["ic"]:+.4f} e10 {st["edge10"]:.4f} e1 {st["edge1"]:.4f}', flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'results.pkl', 'wb') as f:
        pickle.dump(res, f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'eval'])
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--no-hgb', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'build':
        for p in a.paths:
            d = load_features(Path(p))
            c = candidates(d)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(OUT_DIR / f'{Path(p).stem}_cand.npz', **c)
            print(p, {k: float(np.nanstd(v)) for k, v in c.items()})
    else:
        run_eval(a.paths, hgb=not a.no_hgb)


if __name__ == '__main__':
    main()
