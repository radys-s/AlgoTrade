"""Seed ensembles / extra inputs / 2-horizon for the taker MLP (exp_r3_mlp2). Two steps (torch lives in a separate env):
    .venv/Scripts/python.exe -X utf8 -m research.diag.mlp2_r3 build        # -> data/features/mlp2_r3.npz
    %USERPROFILE%/torch-env/Scripts/python -X utf8 research/diag/mlp2_r3.py train --feats base|A|B --seeds 4 --hz 200 [150] --out logs/x.py
Same rows/targets as mlp_r3.py (train = first 75 % of round3 + mirror strided by 3, holdout = last 25 %), plus extra
input columns and the 150 ms target at the same rows. Extra inputs (all causal, computed from events <= i):
  ioc_sweep50 = 50 ms window sum of sign * clip(levels traded through, 0, 11) per market-taker event, where levels =
      last trade price - previous best ask + 1 (buy) / previous best bid - last trade price + 1 (sell);
  spread_max100 = max touch spread over the last 200 events; nev200 = log1p(events in the last 200 ms)."""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

NPZ = 'data/features/mlp2_r3.npz'
EXTRA = ['ioc_sweep50', 'spread_max100', 'nev200']
SETS = {'base': [], 'A': ['ioc_sweep50'], 'B': ['ioc_sweep50', 'spread_max100', 'nev200'], 'C': ['spread_max100'], 'D': ['nev200'], 'E': ['spread_max100', 'nev200']}


def sweep_event(d):
    """Per-event signed levels traded through (0 where no market trade)."""
    tv = d['trade_vol'].astype(np.int64); tc = d['trade_cnt']; ltp = d['last_trade_px'].astype(np.int64)
    pbb = np.r_[d['bid_px'][0], d['bid_px'][:-1]].astype(np.int64)
    pba = np.r_[d['ask_px'][0], d['ask_px'][:-1]].astype(np.int64)
    buy = (tc > 0) & (tv > 0); sell = (tc > 0) & (tv < 0)
    x = np.zeros(len(tv))
    x[buy] = np.clip(ltp[buy] - pba[buy] + 1, 0, 11)
    x[sell] = -np.clip(pbb[sell] - ltp[sell] + 1, 0, 11)
    return x


def win_sum(ts, x, w_us):
    c = np.cumsum(x)
    j = np.searchsorted(ts, ts - w_us, side='right') - 1
    return c - np.where(j >= 0, c[np.maximum(j, 0)], 0.0)


def spread_max(d, spread):
    """Max touch spread over the last 200 events with a two-sided book (the strategy never sees empty-side updates)."""
    import pandas as pd
    ok = (d['bid_px'] > 0) & (d['ask_px'] > 0) & (spread >= 1) & (spread <= 100)
    v = pd.Series(np.where(ok, spread, np.nan))
    cleaned = v.dropna()
    m = cleaned.rolling(200, min_periods=1).max()
    return m.reindex(v.index).ffill().bfill().values


def extras(d, ts, spread):
    import pandas as pd
    n = len(ts)
    return {'ioc_sweep50': win_sum(ts, sweep_event(d), 50_000),
            'spread_max100': spread_max(d, spread),
            'nev200': np.log1p(win_sum(ts, np.ones(n), 200_000))}


def build():
    sys.path.insert(0, '.')
    from research.diag import pred_rfit_r3, pred_r3
    from research.signals import LATENCY_US, value_before
    feats = list(pred_r3.LEAN)
    out = {}
    for name in ('round3', 'round3_mirror'):
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
        d = np.load(f'data/features/{name}.npz')
        ex = extras({k: d[k] for k in d.files if k != 'source'}, ts, cols['spread'])
        cols.update(ex)
        y = {}
        for H in (150, 200):
            e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
            t = e - mid_ff
            t[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
            y[H] = t
        valid = np.isfinite(y[200]) & np.isfinite(y[150])
        for nm in feats + EXTRA:
            valid &= np.isfinite(cols[nm])
        t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
        tri = np.where(valid & (ts + LATENCY_US + 200 * 1000 <= t_end))[0][::3]
        hoi = np.where(valid & (ts > t_end))[0]
        for tag, idx in (('tr', tri), ('ho', hoi)):
            X = np.column_stack([cols[c][idx] for c in feats + EXTRA]).astype(np.float32)
            out[f'{name}_{tag}_X'] = X; out[f'{name}_{tag}_idx'] = idx
            out[f'{name}_{tag}_y200'] = y[200][idx].astype(np.float32); out[f'{name}_{tag}_y150'] = y[150][idx].astype(np.float32)
            out[f'{name}_{tag}_spread'] = cols['spread'][idx].astype(np.float32)
        print(name, len(tri), len(hoi), flush=True)
    np.savez(NPZ, names=np.array(feats + EXTRA), **out)


def edge(p, y, spr, thr_extra=0.2):
    hit = np.abs(p) >= spr / 2 + thr_extra
    return hit.mean(), (np.sign(p[hit]) * y[hit]).mean() if hit.any() else float('nan')


def train(a):
    import torch
    from torch import nn
    z = np.load(NPZ); names = [str(x) for x in z['names']]
    cols = list(range(22)) + [names.index(c) for c in SETS[a.feats]]
    sets = ('round3', 'round3_mirror')
    G = lambda s, tag: z[f'{s}_{tag}_X'][:, cols]
    Xtr = np.vstack([G(s, 'tr') for s in sets])
    mean = Xtr.mean(0); scale = Xtr.std(0); scale[scale == 0] = 1
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    f = lambda X: torch.tensor(np.clip((X - mean) / scale, -8, 8), dtype=torch.float32, device=dev)
    tx = f(Xtr)
    nets = []  # (net, ysc, H)
    t0 = time.time()
    for H in a.hz:
        ytr = np.concatenate([z[f'{s}_tr_y{H}'] for s in sets]); ysc = float(ytr.std())
        ty = torch.tensor(np.clip(ytr / ysc, -a.clip, a.clip), dtype=torch.float32, device=dev)
        for seed in range(a.seed0, a.seed0 + a.seeds):
            torch.manual_seed(seed)
            sizes = [Xtr.shape[1], *a.hidden, 1]; L = []
            for i in range(len(sizes) - 1):
                L.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    L.append(nn.ReLU())
            net = nn.Sequential(*L).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
            for ep in range(a.epochs):
                net.train(); order = torch.randperm(len(ty), device=dev)
                for s in range(0, len(ty), a.batch):
                    b = order[s:s + a.batch]
                    loss = nn.functional.mse_loss(net(tx[b]).squeeze(1), ty[b])
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            nets.append((net, ysc, H)); print(f'  trained H{H} seed{seed} {time.time() - t0:.0f}s', flush=True)

    def pred1(k, X):
        net, ysc, _ = nets[k]; net.eval()
        with torch.no_grad():
            return net(f(X)).squeeze(1).cpu().numpy() * ysc
    print(f'--- {a.feats} hz {a.hz} seeds {a.seeds}: holdout (corr vs y200 / y150; edge at spread/2+0.2: take-rate, signed ticks)')
    for s in sets:
        X = G(s, 'ho'); y = z[f'{s}_ho_y200']; y1 = z[f'{s}_ho_y150']; spr = z[f'{s}_ho_spread']
        ps = [pred1(k, X) for k in range(len(nets))]
        for nm, p in (('net0', ps[0]), ('ens', np.mean(ps, 0))):
            n, e = edge(p, y, spr)
            print(f'{s:14s} {nm:6s} corr200 {np.corrcoef(p, y)[0, 1]:.4f} corr150 {np.corrcoef(p, y1)[0, 1]:.4f} n={n:.4f} e={e:.3f}')
    lst = lambda x: repr(np.round(x, 6).tolist())
    with open(a.out, 'w') as fh:
        fh.write(f'MLP2_COLS = {[names[c] for c in cols]!r}\nMLP2_MEAN = {lst(mean.astype(np.float64))}\nMLP2_SCALE = {lst(scale.astype(np.float64))}\nMLP2_NETS = [\n')
        for net, ysc, H in nets:
            lin = [m for m in net if isinstance(m, nn.Linear)]
            W = [(l.weight.detach().cpu().numpy().T.astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64)) for l in lin]
            W[-1] = (W[-1][0] * ysc, W[-1][1] * ysc)
            fh.write('    [' + ', '.join(f'({lst(w)}, {lst(b)})' for w, b in W) + '],\n')
        fh.write(']\n')
    print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('--hidden', type=int, nargs='+', default=[32, 16])
    ap.add_argument('--feats', default='base'); ap.add_argument('--hz', type=int, nargs='+', default=[200])
    ap.add_argument('--seeds', type=int, default=4); ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=2048); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-3); ap.add_argument('--clip', type=float, default=6.0)
    ap.add_argument('--out', default='logs/mlp2_r3.py')
    a = ap.parse_args()
    build() if a.mode == 'build' else train(a)
