"""MLP vs ridge for the taker predictor (exp_r3_mlp). Two steps (torch lives in a separate env):
    .venv/Scripts/python.exe -X utf8 -m research.diag.mlp_r3 build          # -> data/features/mlp_r3.npz
    %USERPROFILE%/torch-env/Scripts/python -X utf8 research/diag/mlp_r3.py train --hidden 32 16 --act tanh --out logs/mlp_r3_a.py
Target = robust mid at t+5ms+200ms minus the current mid (ticks), train = first 75 % of round3 + mirror (rows strided by 3,
like the ridge), holdout = last 25 % of each tape. Column 'ridge' = the baked rfit-19 ridge on the same rows."""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

H = int(sys.argv[2]); NPZ = f'data/features/mlpq_r3_{H}.npz'


def build():
    sys.path.insert(0, '.')
    from research.diag import pred_rfit_r3, pred_r3
    from research.signals import LATENCY_US, value_before
    import re
    src = open('strategies/mm_r3_g2.py', encoding='utf8').read()
    m = re.search(r'# rfit 19 .*?\n\s*\((-?[\d.e-]+), \(([^)]*)\)\)', src)
    c0 = float(m.group(1)); w = np.array([float(x) for x in m.group(2).split(',')])
    feats = list(pred_r3.LEAN)
    out = {}
    for name in ('round3', 'round3_mirror'):
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
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
            out[f'{name}_{tag}_X'] = X; out[f'{name}_{tag}_y'] = tgt[idx].astype(np.float32)
            out[f'{name}_{tag}_ridge'] = (c0 + X.astype(np.float64) @ w).astype(np.float32)
            out[f'{name}_{tag}_spread'] = cols['spread'][idx].astype(np.float32)
            out[f'{name}_{tag}_fs70'] = cols['fs70'][idx].astype(np.float32)
        print(name, len(tri), len(hoi), flush=True)
    np.savez(NPZ, names=np.array(feats), **out)


def edge(p, y, spr, thr_extra=0.2, scale=1.0):
    hit = np.abs(scale * p) >= spr / 2 + thr_extra
    return hit.mean(), (np.sign(p[hit]) * y[hit]).mean() if hit.any() else float('nan')


def train(a):
    import torch
    from torch import nn
    z = np.load(NPZ); names = [str(x) for x in z['names']]
    sets = ('round3', 'round3_mirror')
    Xtr = np.vstack([z[f'{s}_tr_X'] for s in sets]); ytr = np.concatenate([z[f'{s}_tr_y'] for s in sets])
    mean = Xtr.mean(0); scale = Xtr.std(0); scale[scale == 0] = 1
    ysc = float(ytr.std())
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    sizes = [Xtr.shape[1], *a.hidden, 1]
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.Tanh() if a.act == 'tanh' else nn.ReLU())
    net = nn.Sequential(*L).to(dev)
    # inputs clipped at +-8 sd (fat tails); the same clip goes into the strategy
    f = lambda X: torch.tensor(np.clip((X - mean) / scale, -8, 8), dtype=torch.float32, device=dev)
    tx = f(Xtr); ty = torch.tensor(np.clip(ytr / ysc, -a.clip, a.clip), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    def pred(X):
        net.eval()
        with torch.no_grad():
            return net(f(X)).squeeze(1).cpu().numpy() * ysc
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(0, len(ty), a.batch):
            b = order[s:s + a.batch]
            loss = nn.functional.mse_loss(net(tx[b]).squeeze(1), ty[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % max(1, a.epochs // 5) == 0 or ep == a.epochs - 1:
            r = []
            for s in sets:
                p = pred(z[f'{s}_ho_X']); r.append(np.corrcoef(p, z[f'{s}_ho_y'])[0, 1])
            print(f'ep {ep + 1} loss {loss.item():.4f} ho corr r3 {r[0]:.4f} mirror {r[1]:.4f}  {time.time() - t0:.0f}s', flush=True)
    print('--- holdout summary (corr; take-rate/edge at spread/2+0.2, edge = signed mean move in ticks)')
    for s in sets:
        y = z[f'{s}_ho_y']; spr = z[f'{s}_ho_spread']; pm = pred(z[f'{s}_ho_X']); pr = z[f'{s}_ho_ridge']; ps = z[f'{s}_ho_fs70']
        for nm, p in (('ridge', pr), ('mlp', pm), ('blend.5', .5 * pr + .5 * pm), ('fs70', ps)):
            c = np.corrcoef(p, y)[0, 1]
            es = [edge(p, y, spr, 0.2, sc) for sc in (1.0, 1.5)]
            print(f'{s:14s} {nm:8s} corr {c:.4f}  ' + '  '.join(f'n={n:.4f} e={e:.3f}' for n, e in es))
    # export in train_mlp format
    lin = [m for m in net if isinstance(m, nn.Linear)]
    W = [(l.weight.detach().cpu().numpy().T.astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64)) for l in lin]
    W[-1] = (W[-1][0] * ysc, W[-1][1] * ysc)
    lst = lambda x: repr(np.round(x, 6).tolist())
    with open(a.out, 'w') as fh:
        fh.write(f'MLP_ACT = {a.act!r}\nMLP_MEAN = {lst(mean.astype(np.float64))}\nMLP_SCALE = {lst(scale.astype(np.float64))}\nMLP_LAYERS = [\n')
        for w, b in W:
            fh.write(f'    ({lst(w)}, {lst(b)}),\n')
        fh.write(']\n')
    print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('h'); ap.add_argument('--hidden', type=int, nargs='+', default=[32, 16])
    ap.add_argument('--act', default='tanh'); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=2048); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-3); ap.add_argument('--clip', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='logs/mlp_r3.py')
    a = ap.parse_args()
    build() if a.mode == 'build' else train(a)
