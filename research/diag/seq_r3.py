"""Sequence model (GRU over the last T event tokens + the 22 hand features) vs the taker MLP (exp_r3_seq).
    %USERPROFILE%/torch-env/Scripts/python -X utf8 research/diag/seq_r3.py --T 64 --hidden 48 --epochs 6
Rows/targets/split come from data/features/mlp2_r3.npz (train = first 75 % of round3 + mirror strided by 3, holdout =
last 25 %, target = robust mid at t+5ms+200ms minus the current mid). Tokens are built only from what the strategy sees
online: the per-event change of the top-5 snapshot and the trade fields of data/features/<tape>.npz (no tape op types).
--seq 0 trains the same head without the GRU (the MLP baseline under identical training)."""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

NPZ = 'data/features/mlp2_r3.npz'
SETS = ('round3', 'round3_mirror')
LITE = False
# touch-only tokens (no level 2..5 scan in the strategy): dt, dbid, dask, dmid, dq_b1, dq_a1, trade vol, sweep, trade px, spread, lq_b1, lq_a1, imb1
LITE_COLS = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16]


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def tokens(name):
    d = np.load(f'data/features/{name}.npz')
    ts = d['ts_us'].astype(np.int64)
    bp = d['bid_px'].astype(np.float64); ap = d['ask_px'].astype(np.float64)
    ok = (bp > 0) & (ap > 0) & (ap > bp)
    # forward-fill the touch over one-sided snapshots (the strategy keeps its last two-sided view)
    k = np.where(ok, np.arange(len(bp)), 0); k = np.maximum.accumulate(k)
    bp = bp[k]; ap = ap[k]
    bq = d['bid_qty'].astype(np.float64)[k]; aq = d['ask_qty'].astype(np.float64)[k]
    mid = (bp + ap) / 2
    prev = lambda x: np.r_[x[:1], x[:-1]]
    tv = d['trade_vol'].astype(np.float64); tc = d['trade_cnt'].astype(np.float64)
    ltp = d['last_trade_px'].astype(np.float64)
    pbb = prev(bp); pba = prev(ap)
    buy = (tc > 0) & (tv > 0); sell = (tc > 0) & (tv < 0)
    sw = np.zeros(len(tv)); sw[buy] = np.clip(ltp[buy] - pba[buy] + 1, 0, 11); sw[sell] = -np.clip(pbb[sell] - ltp[sell] + 1, 0, 11)
    # queue changes at the same price level: compare the level-1 qty only when the touch price did not move
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
    if LITE:
        T = T[:, LITE_COLS]
    return T


def main(a):
    import torch
    from torch import nn
    dev = 'cuda'
    z = np.load(NPZ)
    hand = list(range(22))
    tok = {};
    for s in SETS:
        t0 = time.time(); tok[s] = tokens(s); print(s, 'tokens', tok[s].shape, f'{time.time() - t0:.0f}s', flush=True)
    allT = np.vstack([tok[s][::7] for s in SETS])
    tm = allT.mean(0); tsd = allT.std(0); tsd[tsd == 0] = 1
    Xtr = np.vstack([z[f'{s}_tr_X'][:, hand] for s in SETS])
    xm = Xtr.mean(0); xs = Xtr.std(0); xs[xs == 0] = 1
    ytr = np.concatenate([z[f'{s}_tr_y200'] for s in SETS]); ysc = float(ytr.std())
    # one GPU token tensor per tape with T zero-pad rows in front, sample = (tape, idx)
    gtok = [torch.tensor(np.vstack([np.zeros((a.T, tok[s].shape[1]), np.float32), np.clip((tok[s] - tm) / tsd, -8, 8)]), device=dev) for s in SETS]
    nx = lambda X: torch.tensor(np.clip((X - xm) / xs, -8, 8), dtype=torch.float32, device=dev)
    def pack(tag):
        X = torch.cat([nx(z[f'{s}_{tag}_X'][:, hand]) for s in SETS])
        tape = torch.cat([torch.full((len(z[f'{s}_{tag}_idx']),), j, device=dev) for j, s in enumerate(SETS)])
        idx = torch.cat([torch.tensor(z[f'{s}_{tag}_idx'].astype(np.int64), device=dev) for s in SETS])
        y = np.concatenate([z[f'{s}_{tag}_y200'] for s in SETS])
        spr = np.concatenate([z[f'{s}_{tag}_spread'] for s in SETS])
        return X, tape, idx, y, spr
    tr = pack('tr'); ho = pack('ho')
    ar = torch.arange(a.T, device=dev)
    def window(tape, idx):
        # rows idx-T+1..idx of the tape (padded index = idx + T)
        out = torch.empty((len(idx), a.T, gtok[0].shape[1]), device=dev)
        for j in range(len(SETS)):
            m = tape == j
            if m.any():
                p = idx[m][:, None] + 1 + ar[None, :]
                out[m] = gtok[j][p]
        return out
    nf = gtok[0].shape[1]

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.gru = nn.GRU(nf, a.hidden, batch_first=True) if a.seq else None
            nin = 22 + (a.hidden if a.seq else 0)
            s.head = nn.Sequential(nn.Linear(nin, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(s, X, W):
            if s.gru is not None:
                _, h = s.gru(W); X = torch.cat([X, h[0]], 1)
            return s.head(X).squeeze(1)
    torch.manual_seed(a.seed)
    net = Net().to(dev)
    ty = torch.tensor(np.clip(tr[3] / ysc, -6, 6), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    nb = (len(ty) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb)
    def predict(P):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(P[0]), 16384):
                sl = slice(s, s + 16384)
                W = window(P[1][sl], P[2][sl]) if a.seq else None
                out.append(net(P[0][sl], W).cpu().numpy())
        return np.concatenate(out) * ysc
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(nb):
            b = order[s * a.batch:(s + 1) * a.batch]
            W = window(tr[1][b], tr[2][b]) if a.seq else None
            loss = nn.functional.mse_loss(net(tr[0][b], W), ty[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        p = predict(ho); y = ho[3]; n0 = len(z['round3_ho_idx'])
        c = [np.corrcoef(p[:n0], y[:n0])[0, 1], np.corrcoef(p[n0:], y[n0:])[0, 1]]
        print(f'ep {ep + 1} loss {loss.item():.4f} ho corr r3 {c[0]:.4f} mirror {c[1]:.4f}  {time.time() - t0:.0f}s', flush=True)
    p = predict(ho); y = ho[3]; spr = ho[4]
    for thr in (0.2, 0.5):
        hit = np.abs(p) >= spr / 2 + thr
        print(f'edge thr spread/2+{thr}: take-rate {hit.mean():.4f} signed ticks {(np.sign(p[hit]) * y[hit]).mean():.3f} sum {(np.sign(p[hit]) * y[hit]).sum():.0f}')
    np.save(a.out, p.astype(np.float32))
    if a.seq:
        # streaming mode (what the strategy can afford): one GRU state carried over the whole tape, read at the rows
        net.eval(); ps = []
        with torch.no_grad():
            for j, s in enumerate(SETS):
                seq = gtok[j][a.T:]; h = None; H = []
                for c in range(0, len(seq), 4096):
                    o, h = net.gru(seq[c:c + 4096][None].contiguous(), h); H.append(o[0])
                H = torch.cat(H); i = torch.tensor(z[f'{s}_ho_idx'].astype(np.int64), device=dev)
                X = nx(z[f'{s}_ho_X'][:, hand])
                ps.append(net.head(torch.cat([X, H[i]], 1)).squeeze(1).cpu().numpy() * ysc)
        n0 = len(z['round3_ho_idx']); pst = np.concatenate(ps)
        print(f'STREAM ho corr r3 {np.corrcoef(pst[:n0], y[:n0])[0, 1]:.4f} mirror {np.corrcoef(pst[n0:], y[n0:])[0, 1]:.4f}')
        for thr in (0.2, 0.5, 1.0):
            hit = np.abs(pst) >= spr / 2 + thr
            print(f'STREAM edge spread/2+{thr}: take-rate {hit.mean():.4f} sum {(np.sign(pst[hit]) * y[hit]).sum():.0f}')
        np.save(a.out.replace('.npy', '_stream.npy'), pst.astype(np.float32))
    if a.save:
        torch.save({'state': net.state_dict(), 'tm': tm, 'tsd': tsd, 'xm': xm, 'xs': xs, 'ysc': ysc, 'T': a.T, 'lite': LITE}, a.save)
    print('saved', a.out)


def export(pt, out):
    """Write a saved checkpoint as python literals for the strategy (GRU weights in torch r,z,n gate order, transposed)."""
    import torch
    c = torch.load(pt, weights_only=False); st = {k: v.cpu().numpy().astype(np.float64) for k, v in c['state'].items()}
    lst = lambda x: repr(np.round(np.asarray(x, np.float64), 6).tolist())
    with open(out, 'w') as fh:
        nl = chr(10)
        fh.write(f"SEQ_LITE = {bool(c.get('lite', True))}{nl}SEQ_TM = {lst(c['tm'])}{nl}SEQ_TSD = {lst(c['tsd'])}{nl}")
        fh.write(f"SEQ_XM = {lst(c['xm'])}{nl}SEQ_XS = {lst(c['xs'])}{nl}SEQ_YSC = {float(c['ysc'])!r}{nl}")
        fh.write(f"SEQ_WI = {lst(st['gru.weight_ih_l0'].T)}{nl}SEQ_BI = {lst(st['gru.bias_ih_l0'])}{nl}")
        fh.write(f"SEQ_WH = {lst(st['gru.weight_hh_l0'].T)}{nl}SEQ_BH = {lst(st['gru.bias_hh_l0'])}{nl}SEQ_HEAD = [{nl}")
        for i in (0, 2, 4):
            fh.write(f"    ({lst(st[f'head.{i}.weight'].T)}, {lst(st[f'head.{i}.bias'])}),{nl}")
        fh.write(']' + nl)
    print('wrote', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        export(sys.argv[2], sys.argv[3]); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=64); ap.add_argument('--hidden', type=int, default=48)
    ap.add_argument('--seq', type=int, default=1); ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=2048); ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--wd', type=float, default=1e-3); ap.add_argument('--out', default='logs/seq_r3_ho.npy')
    ap.add_argument('--save', default='')
    ap.add_argument('--lite', type=int, default=0)
    a = ap.parse_args(); LITE = bool(a.lite)
    main(a)
