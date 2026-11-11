"""GRU taker predictor (research/diag/seq_r3.py model, LITE tokens) trained on more tapes: round3 + mirror + round2 + mirror.
    .venv/Scripts/python.exe -X utf8 -m research.diag.mlp2x_r3 build      # -> data/features/mlp2x_r3.npz (4 tapes)
    $USERPROFILE/torch-env/Scripts/python -X utf8 research/diag/mlp2x_r3.py train --train round3 round3_mirror round2 round2_mirror --save logs/gx_all_s0.pt
    $USERPROFILE/torch-env/Scripts/python -X utf8 research/diag/seq_r3.py export logs/gx_all_s0.pt logs/gx_all_s0_weights.py
Rows, 22 hand features, 200 ms target and the split (first 75 % train strided by 3 / last 25 % holdout, per tape) are
mlp2_r3.py's build logic applied to round2 and its mirror; round3 rows are copied from mlp2_r3.npz unchanged.
Normalisation (tokens and hand features) is recomputed on the union of the TRAINING tapes, as seq_r3.main does.
Training is seq_r3.main with the training tapes separated from the evaluation tapes (STREAM holdout corr on all four)."""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, '.')
SRC = 'data/features/mlp2_r3.npz'
NPZ = 'data/features/mlp2x_r3.npz'
ALL = ('round3', 'round3_mirror', 'round2', 'round2_mirror')


def build():
    from research.diag import pred_rfit_r3, pred_r3
    from research.diag.mlp2_r3 import EXTRA, extras
    from research.signals import LATENCY_US, value_before
    feats = list(pred_r3.LEAN)
    z = np.load(SRC); out = {k: z[k] for k in z.files}
    assert [str(x) for x in z['names']] == feats + EXTRA
    for name in ('round2', 'round2_mirror'):
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
        d = np.load(f'data/features/{name}.npz')
        cols.update(extras({k: d[k] for k in d.files if k != 'source'}, ts, cols['spread']))
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
            out[f'{name}_{tag}_X'] = np.column_stack([cols[c][idx] for c in feats + EXTRA]).astype(np.float32)
            out[f'{name}_{tag}_idx'] = idx
            out[f'{name}_{tag}_y200'] = y[200][idx].astype(np.float32); out[f'{name}_{tag}_y150'] = y[150][idx].astype(np.float32)
            out[f'{name}_{tag}_spread'] = cols['spread'][idx].astype(np.float32)
        print(name, len(tri), len(hoi), flush=True)
    np.savez(NPZ, **out)
    print('wrote', NPZ)


def train(a):
    import torch
    from torch import nn
    from research.diag import seq_r3
    seq_r3.LITE = True
    dev = 'cuda'
    z = np.load(NPZ)
    TR = tuple(a.train); EV = ALL
    tok = {s: seq_r3.tokens(s) for s in EV}
    allT = np.vstack([tok[s][::7] for s in TR])
    tm = allT.mean(0); tsd = allT.std(0); tsd[tsd == 0] = 1
    Xtr = np.vstack([z[f'{s}_tr_X'][:, :22] for s in TR])
    xm = Xtr.mean(0); xs = Xtr.std(0); xs[xs == 0] = 1
    ytr = np.concatenate([z[f'{s}_tr_y200'] for s in TR]); ysc = float(ytr.std())
    gtok = {s: torch.tensor(np.vstack([np.zeros((a.T, tok[s].shape[1]), np.float32), np.clip((tok[s] - tm) / tsd, -8, 8)]), device=dev) for s in EV}
    nx = lambda X: torch.tensor(np.clip((X - xm) / xs, -8, 8), dtype=torch.float32, device=dev)
    X = torch.cat([nx(z[f'{s}_tr_X'][:, :22]) for s in TR])
    tape = torch.cat([torch.full((len(z[f'{s}_tr_idx']),), j, device=dev) for j, s in enumerate(TR)])
    idx = torch.cat([torch.tensor(z[f'{s}_tr_idx'].astype(np.int64), device=dev) for s in TR])
    ar = torch.arange(a.T, device=dev); nf = gtok[TR[0]].shape[1]

    def window(tp, ix):
        out = torch.empty((len(ix), a.T, nf), device=dev)
        for j, s in enumerate(TR):
            m = tp == j
            if m.any():
                out[m] = gtok[s][ix[m][:, None] + 1 + ar[None, :]]
        return out

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.gru = nn.GRU(nf, a.hidden, batch_first=True)
            s.head = nn.Sequential(nn.Linear(22 + a.hidden, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(s, X, W):
            _, h = s.gru(W)
            return s.head(torch.cat([X, h[0]], 1)).squeeze(1)
    torch.manual_seed(a.seed)
    net = Net().to(dev)
    ty = torch.tensor(np.clip(ytr / ysc, -6, 6), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    nb = (len(ty) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(nb):
            b = order[s * a.batch:(s + 1) * a.batch]
            loss = nn.functional.mse_loss(net(X[b], window(tape[b], idx[b])), ty[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        print(f'ep {ep + 1} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
    net.eval(); res = []
    with torch.no_grad():
        for s in EV:
            seq = gtok[s][a.T:]; h = None; H = []
            for c in range(0, len(seq), 4096):
                o, h = net.gru(seq[c:c + 4096][None].contiguous(), h); H.append(o[0])
            H = torch.cat(H); i = torch.tensor(z[f'{s}_ho_idx'].astype(np.int64), device=dev)
            p = net.head(torch.cat([nx(z[f'{s}_ho_X'][:, :22]), H[i]], 1)).squeeze(1).cpu().numpy() * ysc
            y = z[f'{s}_ho_y200']; spr = z[f'{s}_ho_spread']
            hit = np.abs(p) >= spr / 2 + 0.5
            res.append(f'{s} {np.corrcoef(p, y)[0, 1]:.4f} edge0.5 {(np.sign(p[hit]) * y[hit]).sum():.0f}')
    print('STREAM ho corr | ' + ' | '.join(res), flush=True)
    if a.save:
        torch.save({'state': net.state_dict(), 'tm': tm, 'tsd': tsd, 'xm': xm, 'xs': xs, 'ysc': ysc, 'T': a.T, 'lite': True}, a.save)
        print('saved', a.save)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--train', nargs='+', default=list(ALL))
    ap.add_argument('--T', type=int, default=128); ap.add_argument('--hidden', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--save', default='')
    train(ap.parse_args())
