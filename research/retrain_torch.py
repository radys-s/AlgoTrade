"""Torch half of research/retrain.py (runs in the torch env). Same training as the round-3 scripts, with paths as arguments:
    mlp  = research/diag/mlp_r3.py train --act relu (22-32-16-1, 20 epochs, seed 0) on the r200 rows
    mq   = research/diag/mlpq_r3.py train 500 --act relu on the r500 rows
    seq  = research/diag/seq_r3.py --T 128 --hidden 16 --lite 1 (6 epochs, seed 0) on the m2 rows, then its export
    %USERPROFILE%/torch-env/Scripts/python -X utf8 research/retrain_torch.py mlp --rows A.npz B.npz --out logs/x.py
--rows: one rows npz per tape (research/retrain_feats.py output), --feat: the matching feature npz (seq tokens only)."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np



def lst(x):
    return repr(np.round(np.asarray(x, np.float64), 6).tolist())


def train_mlp(a, key):
    import torch
    from torch import nn
    zs = [np.load(p) for p in a.rows]
    Xtr = np.vstack([z[f'{key}_tr_X'] for z in zs]); ytr = np.concatenate([z[f'{key}_tr_y'] for z in zs])
    mean = Xtr.mean(0); scale = Xtr.std(0); scale[scale == 0] = 1
    ysc = float(ytr.std())
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    sizes = [Xtr.shape[1], 32, 16, 1]
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.ReLU())
    net = nn.Sequential(*L).to(dev)
    f = lambda X: torch.tensor(np.clip((X - mean) / scale, -8, 8), dtype=torch.float32, device=dev)
    tx = f(Xtr); ty = torch.tensor(np.clip(ytr / ysc, -6.0, 6.0), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    epochs, batch = 20, 2048
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def pred(X):
        net.eval()
        with torch.no_grad():
            return net(f(X)).squeeze(1).cpu().numpy() * ysc
    t0 = time.time()
    for ep in range(epochs):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(0, len(ty), batch):
            b = order[s:s + batch]
            loss = nn.functional.mse_loss(net(tx[b]).squeeze(1), ty[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 5 == 0:
            r = [np.corrcoef(pred(z[f'{key}_ho_X']), z[f'{key}_ho_y'])[0, 1] for z in zs]
            print(f'{key} ep {ep + 1} loss {loss.item():.4f} ho corr ' + ' '.join(f'{c:.4f}' for c in r) + f'  {time.time() - t0:.0f}s', flush=True)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    W = [(l.weight.detach().cpu().numpy().T.astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64)) for l in lin]
    W[-1] = (W[-1][0] * ysc, W[-1][1] * ysc)
    return mean.astype(np.float64), scale.astype(np.float64), W


def write_mlp(a, key):
    mean, scale, W = train_mlp(a, key)
    with open(a.out, 'w') as fh:
        fh.write(f'MLP_ACT = {"relu"!r}\nMLP_MEAN = {lst(mean)}\nMLP_SCALE = {lst(scale)}\nMLP_LAYERS = [\n')
        for w, b in W:
            fh.write(f'    ({lst(w)}, {lst(b)}),\n')
        fh.write(']\n')
    print('wrote', a.out)


LITE_COLS = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16]


def tokens(npz):
    """research/diag/seq_r3.tokens with the npz path as argument (lite columns)."""
    d = np.load(npz)
    ts = d['ts_us'].astype(np.int64)
    bp = d['bid_px'].astype(np.float64); ap = d['ask_px'].astype(np.float64)
    ok = (bp > 0) & (ap > 0) & (ap > bp)
    k = np.where(ok, np.arange(len(bp)), 0); k = np.maximum.accumulate(k)
    bp = bp[k]; ap = ap[k]
    bq = d['bid_qty'].astype(np.float64)[k]; aq = d['ask_qty'].astype(np.float64)[k]
    mid = (bp + ap) / 2
    prev = lambda x: np.r_[x[:1], x[:-1]]
    slog = lambda x: np.sign(x) * np.log1p(np.abs(x))
    tv = d['trade_vol'].astype(np.float64); tc = d['trade_cnt'].astype(np.float64)
    ltp = d['last_trade_px'].astype(np.float64)
    pbb = prev(bp); pba = prev(ap)
    buy = (tc > 0) & (tv > 0); sell = (tc > 0) & (tv < 0)
    sw = np.zeros(len(tv)); sw[buy] = np.clip(ltp[buy] - pba[buy] + 1, 0, 11); sw[sell] = -np.clip(pbb[sell] - ltp[sell] + 1, 0, 11)
    same_b = bp == pbb; same_a = ap == pba
    d_b1 = np.where(same_b, bq[:, 0] - prev(bq[:, 0]), 0.0); d_a1 = np.where(same_a, aq[:, 0] - prev(aq[:, 0]), 0.0)
    d_b2 = np.where(same_b, bq[:, 1] - prev(bq[:, 1]), 0.0); d_a2 = np.where(same_a, aq[:, 1] - prev(aq[:, 1]), 0.0)
    tot_b = bq.sum(1); tot_a = aq.sum(1)
    d_tb = tot_b - prev(tot_b); d_ta = tot_a - prev(tot_a)
    dt = np.r_[0, np.diff(ts)].astype(np.float64)
    spread = ap - bp
    imb1 = (bq[:, 0] - aq[:, 0]) / np.maximum(bq[:, 0] + aq[:, 0], 1)
    imb5 = (tot_b - tot_a) / np.maximum(tot_b + tot_a, 1)
    T = np.column_stack([
        np.log1p(dt / 100.0),
        np.clip(bp - pbb, -5, 5), np.clip(ap - pba, -5, 5), np.clip(mid - prev(mid), -5, 5),
        slog(d_b1), slog(d_a1), slog(d_b2), slog(d_a2), slog(d_tb), slog(d_ta),
        slog(tv), sw, np.where(tc > 0, np.clip(ltp - prev(mid), -10, 10), 0.0),
        np.clip(spread, 0, 20), np.log1p(bq[:, 0]), np.log1p(aq[:, 0]), imb1, imb5,
    ]).astype(np.float32)
    return T[:, LITE_COLS]


def train_seq(a):
    """research/diag/seq_r3.main (T 128, hidden 16, lite, 6 epochs, batch 2048, lr 3e-3, wd 1e-3, seed 0) + export."""
    import torch
    from torch import nn
    T_, HID, EPOCHS, BATCH, LR, WD = 128, 16, a.seq_epochs, 2048, 3e-3, 1e-3
    dev = 'cuda'
    zs = [np.load(p) for p in a.rows]
    S = range(len(zs))
    hand = list(range(22))
    tok = [tokens(p) for p in a.feat]
    allT = np.vstack([t[::7] for t in tok])
    tm = allT.mean(0); tsd = allT.std(0); tsd[tsd == 0] = 1
    Xtr = np.vstack([z['m2_tr_X'][:, hand] for z in zs])
    xm = Xtr.mean(0); xs = Xtr.std(0); xs[xs == 0] = 1
    ytr = np.concatenate([z['m2_tr_y200'] for z in zs]); ysc = float(ytr.std())
    gtok = [torch.tensor(np.vstack([np.zeros((T_, t.shape[1]), np.float32), np.clip((t - tm) / tsd, -8, 8)]), device=dev) for t in tok]
    nx = lambda X: torch.tensor(np.clip((X - xm) / xs, -8, 8), dtype=torch.float32, device=dev)

    def pack(tag):
        X = torch.cat([nx(zs[j][f'm2_{tag}_X'][:, hand]) for j in S])
        tape = torch.cat([torch.full((len(zs[j][f'm2_{tag}_idx']),), j, device=dev) for j in S])
        idx = torch.cat([torch.tensor(zs[j][f'm2_{tag}_idx'].astype(np.int64), device=dev) for j in S])
        y = np.concatenate([zs[j][f'm2_{tag}_y200'] for j in S])
        return X, tape, idx, y
    tr = pack('tr'); ho = pack('ho')
    ar = torch.arange(T_, device=dev)

    def window(tape, idx):
        out = torch.empty((len(idx), T_, gtok[0].shape[1]), device=dev)
        for j in S:
            m = tape == j
            if m.any():
                p = idx[m][:, None] + 1 + ar[None, :]
                out[m] = gtok[j][p]
        return out
    nf = gtok[0].shape[1]

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.gru = nn.GRU(nf, HID, batch_first=True)
            s.head = nn.Sequential(nn.Linear(22 + HID, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(s, X, W):
            _, h = s.gru(W); X = torch.cat([X, h[0]], 1)
            return s.head(X).squeeze(1)
    torch.manual_seed(a.seed)
    net = Net().to(dev)
    ty = torch.tensor(np.clip(tr[3] / ysc, -6, 6), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    nb = (len(ty) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb)
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(nb):
            b = order[s * BATCH:(s + 1) * BATCH]
            loss = nn.functional.mse_loss(net(tr[0][b], window(tr[1][b], tr[2][b])), ty[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        print(f'seq ep {ep + 1} loss {loss.item():.4f}  {time.time() - t0:.0f}s', flush=True)
    # streaming holdout corr (what the strategy runs)
    net.eval()
    with torch.no_grad():
        for j in S:
            seq = gtok[j][T_:]; h = None; Hs = []
            for c in range(0, len(seq), 4096):
                o, h = net.gru(seq[c:c + 4096][None].contiguous(), h); Hs.append(o[0])
            Hs = torch.cat(Hs); i = torch.tensor(zs[j]['m2_ho_idx'].astype(np.int64), device=dev)
            p = net.head(torch.cat([nx(zs[j]['m2_ho_X'][:, hand]), Hs[i]], 1)).squeeze(1).cpu().numpy() * ysc
            print(f'seq STREAM ho corr tape{j} {np.corrcoef(p, zs[j]["m2_ho_y200"])[0, 1]:.4f}', flush=True)
    st = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    nl = '\n'
    with open(a.out, 'w') as fh:
        fh.write(f"SEQ_LITE = True{nl}SEQ_TM = {lst(tm)}{nl}SEQ_TSD = {lst(tsd)}{nl}")
        fh.write(f"SEQ_XM = {lst(xm)}{nl}SEQ_XS = {lst(xs)}{nl}SEQ_YSC = {float(ysc)!r}{nl}")
        fh.write(f"SEQ_WI = {lst(st['gru.weight_ih_l0'].T)}{nl}SEQ_BI = {lst(st['gru.bias_ih_l0'])}{nl}")
        fh.write(f"SEQ_WH = {lst(st['gru.weight_hh_l0'].T)}{nl}SEQ_BH = {lst(st['gru.bias_hh_l0'])}{nl}SEQ_HEAD = [{nl}")
        for i in (0, 2, 4):
            fh.write(f"    ({lst(st[f'head.{i}.weight'].T)}, {lst(st[f'head.{i}.bias'])}),{nl}")
        fh.write(']' + nl)
    print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('mlp', 'mq', 'seq'))
    ap.add_argument('--rows', nargs='+', required=True)
    ap.add_argument('--feat', nargs='+', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seq-epochs', type=int, default=6)
    a = ap.parse_args()
    if a.mode == 'seq':
        train_seq(a)
    else:
        write_mlp(a, 'r200' if a.mode == 'mlp' else 'r500')
