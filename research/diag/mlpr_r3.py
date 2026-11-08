"""Raw-state MLP for the taker (exp_r3_mlpr). Same rows/target as mlp_r3 (22 hand features, 200 ms robust-mid move), PLUS raw book state:
log1p level qty x5 per side (now, 20 ms, 100 ms ago), touch order counts (now, 100 ms ago), last 8 trades (side, log qty, price offset from mid),
log time since last insert per side, spread now / 50 ms ago.  (Level PRICES are not in the npz: only quantities of the best 5 distinct prices.)
    .venv/Scripts/python.exe -X utf8 research/diag/mlpr_r3.py build
    %USERPROFILE%/torch-env/Scripts/python -X utf8 research/diag/mlpr_r3.py train --hidden 64 32 --out logs/mlpr_a.py"""
from __future__ import annotations
import argparse, sys, time
import numpy as np

NPZ = 'data/features/mlpr_r3.npz'
H = 200
NT = 8


def raw_feats(d, idx):
    import pandas as pd
    ts = d['ts_us'].astype(np.int64)
    n = len(ts)
    bp = d['bid_px'].astype(np.float64); ap = d['ask_px'].astype(np.float64)
    two = (bp > 0) & (ap > 0)
    mid = pd.Series(np.where(two, (bp + ap) / 2, np.nan)).ffill().bfill().values
    spr = pd.Series(np.clip(np.where(two, ap - bp, np.nan), 0, 10)).ffill().bfill().values
    lb = np.log1p(d['bid_qty'].astype(np.float64)); la = np.log1p(d['ask_qty'].astype(np.float64))
    cb = np.log1p(d['bid_cnt1'].astype(np.float64)); ca = np.log1p(d['ask_cnt1'].astype(np.float64))

    def ago(ms):  # last row with ts <= t - ms
        j = np.searchsorted(ts, ts[idx] - ms * 1000, side='right') - 1
        return np.maximum(j, 0)
    cols = []
    for j in (idx, ago(20), ago(100)):
        cols += [lb[j], la[j]]
    for j in (idx, ago(100)):
        cols += [cb[j][:, None], ca[j][:, None]]
    cols += [spr[idx][:, None], spr[ago(50)][:, None]]
    tr_rows = np.where(d['trade_cnt'] > 0)[0]
    tv = d['trade_vol'][tr_rows].astype(np.float64); tp = d['last_trade_px'][tr_rows].astype(np.float64)
    k = np.searchsorted(tr_rows, idx, side='right')  # number of trades at rows <= idx
    T = np.zeros((len(idx), NT, 3))
    for m in range(NT):
        t = k - 1 - m
        v = t >= 0
        tt = np.maximum(t, 0)
        T[:, m, 0] = np.where(v, np.sign(tv[tt]), 0)
        T[:, m, 1] = np.where(v, np.log1p(np.abs(tv[tt])), 0)
        T[:, m, 2] = np.where(v, tp[tt] - mid[idx], 0)
    cols.append(T.reshape(len(idx), -1))
    for sd in (1, -1):
        ins = (d['ev_type'] == 0) & (d['ev_side'] == sd)
        last = np.maximum.accumulate(np.where(ins, np.arange(n), -1))[idx]
        cols.append(np.where(last >= 0, np.log1p((ts[idx] - ts[np.maximum(last, 0)]) / 1000.0), 12.0)[:, None])
    return np.hstack(cols).astype(np.float32)


def build():
    sys.path.insert(0, '.')
    from research.diag import pred_r3, pred_rfit_r3
    from research.signals import LATENCY_US, value_before
    feats = list(pred_r3.LEAN)
    out = {}
    for name in ('round3', 'round3_mirror'):
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
        d = dict(np.load(f'data/features/{name}.npz'))
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        tgt = e - mid_ff
        tgt[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        valid = np.isfinite(tgt)
        for nm in feats:
            valid &= np.isfinite(cols[nm])
        t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
        tri = np.where(valid & (ts + LATENCY_US + H * 1000 <= t_end))[0][::3]
        hoi = np.where(valid & (ts > t_end))[0]
        for tag, idx in (('tr', tri), ('ho', hoi)):
            X = np.column_stack([cols[c][idx] for c in feats]).astype(np.float32)
            out[f'{name}_{tag}_X'] = X; out[f'{name}_{tag}_R'] = raw_feats(d, idx); out[f'{name}_{tag}_y'] = tgt[idx].astype(np.float32)
            out[f'{name}_{tag}_idx'] = idx
            out[f'{name}_{tag}_spread'] = cols['spread'][idx].astype(np.float32)
        print(name, len(tri), len(hoi), flush=True)
    np.savez(NPZ, names=np.array(feats), **out)


def edge(p, y, spr, thr_extra=0.2):
    hit = np.abs(p) >= spr / 2 + thr_extra
    return hit.mean(), (np.sign(p[hit]) * y[hit]).mean() if hit.any() else float('nan')


def train(a):
    import torch
    from torch import nn
    z = np.load(NPZ)
    sets = ('round3', 'round3_mirror')
    get = lambda s, t: z[f'{s}_{t}_X'] if a.base else np.hstack([z[f'{s}_{t}_X'], z[f'{s}_{t}_R']])
    Xtr = np.vstack([get(s, 'tr') for s in sets]); ytr = np.concatenate([z[f'{s}_tr_y'] for s in sets])
    mean = Xtr.mean(0); scale = Xtr.std(0); scale[scale == 0] = 1
    ysc = float(ytr.std())
    dev = 'cuda'
    torch.manual_seed(a.seed)
    sizes = [Xtr.shape[1], *a.hidden, 1]
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.ReLU())
            if a.drop > 0:
                L.append(nn.Dropout(a.drop))
    net = nn.Sequential(*L).to(dev)
    f = lambda X: torch.tensor(np.clip((X - mean) / scale, -8, 8), dtype=torch.float32, device=dev)
    tx = f(Xtr); ty = torch.tensor(np.clip(ytr / ysc, -a.clip, a.clip), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    def pred(X):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(f(X[i:i + 200000])).squeeze(1) for i in range(0, len(X), 200000)]).cpu().numpy() * ysc
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(0, len(ty), a.batch):
            b = order[s:s + a.batch]
            loss = nn.functional.mse_loss(net(tx[b]).squeeze(1), ty[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % max(1, a.epochs // 4) == 0 or ep == a.epochs - 1:
            r = [np.corrcoef(pred(get(s, 'ho')), z[f'{s}_ho_y'])[0, 1] for s in sets]
            print(f'ep {ep + 1} loss {loss.item():.4f} ho corr r3 {r[0]:.4f} mirror {r[1]:.4f}  {time.time() - t0:.0f}s', flush=True)
    for s in sets:
        y = z[f'{s}_ho_y']; spr = z[f'{s}_ho_spread']; p = pred(get(s, 'ho'))
        n, e = edge(p, y, spr)
        print(f'{s:14s} corr {np.corrcoef(p, y)[0, 1]:.4f} take {n:.4f} edge {e:.3f}')
    lin = [m for m in net if isinstance(m, nn.Linear)]
    W = [(l.weight.detach().cpu().numpy().T.astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64)) for l in lin]
    W[-1] = (W[-1][0] * ysc, W[-1][1] * ysc)
    lst = lambda x: repr(np.round(x, 6).tolist())
    with open(a.out, 'w') as fh:
        fh.write(f'MLPR_MEAN = {lst(mean)}\nMLPR_SCALE = {lst(scale)}\nMLPR_LAYERS = [\n')
        for w, b in W:
            fh.write(f'    ({lst(w)}, {lst(b)}),\n')
        fh.write(']\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('--hidden', type=int, nargs='+', default=[64, 32])
    ap.add_argument('--epochs', type=int, default=16); ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--clip', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--base', action='store_true')
    ap.add_argument('--out', default='logs/mlpr.py')
    a = ap.parse_args()
    build() if a.mode == 'build' else train(a)
