"""Wide-feature MLP study (exp_r3_mlpw). Build with the repo venv, train with the torch env:
    .venv/Scripts/python.exe -X utf8 -m research.diag.mlpw_r3 build            # -> data/features/mlpw_r3.npz
    %USERPROFILE%/torch-env/Scripts/python -X utf8 research/diag/mlpw_r3.py train --extra all|none|a,b,c --out logs/x.py
Target as mlp_r3 (robust mid at t+5ms+200ms minus the current mid), train = first 75 % of round3 + mirror (stride 3),
holdout = last 25 % of each tape (stride 2). Wide features are those of edge_gbm_r3.wide."""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

NPZ = 'data/features/mlpw_r3.npz'
H = 200


def build():
    sys.path.insert(0, '.')
    from research.diag import pred_rfit_r3, pred_r3, edge_gbm_r3
    from research.signals import LATENCY_US, value_before
    lean = list(pred_r3.LEAN)
    out = {}
    wnames = None
    for name in ('round3', 'round3_mirror'):
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
        z = np.load(f'data/features/{name}.npz'); d = {k: z[k] for k in z.files}
        W = edge_gbm_r3.wide(name, d, cols, rm, mid_ff)
        wnames = list(W)
        allc = {**{k: cols[k] for k in lean}, **W}
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        tgt = e - mid_ff
        tgt[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        valid = np.isfinite(tgt)
        for nm in allc:
            valid &= np.isfinite(allc[nm])
        t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
        tri = np.where(valid & (ts + LATENCY_US + H * 1000 <= t_end))[0][::3]
        hoi = np.where(valid & (ts > t_end))[0][::2]
        for tag, idx in (('tr', tri), ('ho', hoi)):
            out[f'{name}_{tag}_X'] = np.column_stack([allc[c][idx] for c in lean + wnames]).astype(np.float32)
            out[f'{name}_{tag}_y'] = tgt[idx].astype(np.float32)
            out[f'{name}_{tag}_spread'] = cols['spread'][idx].astype(np.float32)
        print(name, len(tri), len(hoi), flush=True)
    np.savez(NPZ, names=np.array(lean + wnames), nlean=len(lean), **out)


def edge(p, y, spr, thr_extra=0.2):
    hit = np.abs(p) >= spr / 2 + thr_extra
    return hit.mean(), (np.sign(p[hit]) * y[hit]).mean() if hit.any() else float('nan')


def edge_rate(p, y, spr, rate):
    """Edge at a fixed take rate (equal trigger rate comparisons)."""
    t = np.abs(p) >= np.quantile(np.abs(p), 1 - rate)
    return (np.sign(p[t]) * y[t]).mean()


def train(a):
    import torch
    from torch import nn
    z = np.load(NPZ); names = [str(x) for x in z['names']]; nl = int(z['nlean'])
    if a.extra == 'all':
        sel = list(range(len(names)))
    elif a.extra == 'none':
        sel = list(range(nl))
    else:
        want = a.extra.split(',')
        sel = list(range(nl)) + [names.index(w) for w in want]
    sets = ('round3', 'round3_mirror')
    Xtr = np.vstack([z[f'{s}_tr_X'] for s in sets])[:, sel]; ytr = np.concatenate([z[f'{s}_tr_y'] for s in sets])
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
    f = lambda X: torch.tensor(np.clip((X - mean) / scale, -8, 8), dtype=torch.float32, device=dev)
    tx = f(Xtr); ty = torch.tensor(np.clip(ytr / ysc, -a.clip, a.clip), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    def pred(X):
        net.eval()
        with torch.no_grad():
            return net(f(X[:, sel])).squeeze(1).cpu().numpy() * ysc
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(0, len(ty), a.batch):
            b = order[s:s + a.batch]
            loss = nn.functional.mse_loss(net(tx[b]).squeeze(1), ty[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % max(1, a.epochs // 4) == 0 or ep == a.epochs - 1:
            r = [np.corrcoef(pred(z[f'{s}_ho_X']), z[f'{s}_ho_y'])[0, 1] for s in sets]
            print(f'ep {ep + 1} loss {loss.item():.4f} ho corr r3 {r[0]:.4f} mirror {r[1]:.4f} {time.time() - t0:.0f}s', flush=True)
    res = []
    for s in sets:
        y = z[f'{s}_ho_y']; spr = z[f'{s}_ho_spread']; p = pred(z[f'{s}_ho_X'])
        c = np.corrcoef(p, y)[0, 1]
        n, e = edge(p, y, spr)
        e2 = [edge_rate(p, y, spr, q) for q in (0.01, 0.03, 0.06)]
        res.append(c)
        print(f'RESULT {a.extra[:40]} {s:14s} corr {c:.4f} thr n={n:.4f} e={e:.3f} | top1/3/6% edge ' + ' '.join(f'{x:.3f}' for x in e2), flush=True)
    np.save(a.out + '.pred.npy', np.concatenate([pred(z[f'{s}_ho_X']) for s in sets]))
    if a.export:
        lin = [m for m in net if isinstance(m, nn.Linear)]
        W = [(l.weight.detach().cpu().numpy().T.astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64)) for l in lin]
        W[-1] = (W[-1][0] * ysc, W[-1][1] * ysc)
        lst = lambda x: repr(np.round(x, 6).tolist())
        with open(a.out, 'w') as fh:
            fh.write(f'MLPW_NAMES = {[names[i] for i in sel]!r}\nMLPW_ACT = {a.act!r}\nMLPW_MEAN = {lst(mean.astype(np.float64))}\nMLPW_SCALE = {lst(scale.astype(np.float64))}\nMLPW_LAYERS = [\n')
            for w, b in W:
                fh.write(f'    ({lst(w)}, {lst(b)}),\n')
            fh.write(']\n')
        print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('--hidden', type=int, nargs='+', default=[64, 32])
    ap.add_argument('--act', default='relu'); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=2048); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-3); ap.add_argument('--clip', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='logs/mlpw_r3.py')
    ap.add_argument('--extra', default='all'); ap.add_argument('--export', type=int, default=0)
    a = ap.parse_args()
    build() if a.mode == 'build' else train(a)
