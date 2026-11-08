"""Protocol check + ensemble + 100 % fit for the R3 taker MLP (exp_r3_mlpf). Modes:
  build  (.venv)      adds the 100 % training rows (stride 3, round3 + mirror) -> data/features/mlp_full_r3.npz
  sweep  (torch-env)  75 % fit, epochs x wd x 3 seeds, holdout corr / decision-point edge, single vs ensemble
  final  (torch-env)  --epochs E --wd W --frac 75|100 : 3 seeds -> logs/mlpf_<frac>.py (MLP_ENS_<frac>)
  bake   (.venv)      strategies/mm_r3_mlp2.py + logs/mlpf_75.py/mlpf_100.py -> strategies/exp_r3_mlpf.py
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np

NPZ = 'data/features/mlp_r3.npz'
NPZF = 'data/features/mlp_full_r3.npz'
SETS = ('round3', 'round3_mirror')
H = 200


def build():
    sys.path.insert(0, '.')
    from research.diag import pred_rfit_r3, pred_r3
    from research.signals import LATENCY_US, value_before
    feats = list(pred_r3.LEAN)
    out = {}
    for name in SETS:
        ts, cols, rm, mid_ff = pred_rfit_r3.cols_for(name)
        e = value_before(ts, rm, ts + LATENCY_US + H * 1000)
        tgt = e - mid_ff
        tgt[ts + LATENCY_US + H * 1000 > ts[-1]] = np.nan
        valid = np.isfinite(tgt)
        for nm in feats:
            valid &= np.isfinite(cols[nm])
        idx = np.where(valid)[0][::3]
        out[f'{name}_X'] = np.column_stack([cols[c][idx] for c in feats]).astype(np.float32)
        out[f'{name}_y'] = tgt[idx].astype(np.float32)
        print(name, len(idx), flush=True)
    np.savez(NPZF, names=np.array(feats), **out)


def edge(p, y, spr, thr_extra=0.2):
    hit = np.abs(p) >= spr / 2 + thr_extra
    return hit.mean(), (np.sign(p[hit]) * y[hit]).mean() if hit.any() else float('nan')


def fit(Xtr, ytr, epochs, wd, seed, hidden=(32, 16), lr=2e-3, batch=2048, clip=6.0):
    import torch
    from torch import nn
    dev = 'cuda'
    mean = Xtr.mean(0); scale = Xtr.std(0); scale[scale == 0] = 1
    ysc = float(ytr.std())
    torch.manual_seed(seed)
    sizes = [Xtr.shape[1], *hidden, 1]
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.ReLU())
    net = nn.Sequential(*L).to(dev)
    f = lambda X: torch.tensor(np.clip((X - mean) / scale, -8, 8), dtype=torch.float32, device=dev)
    tx = f(Xtr); ty = torch.tensor(np.clip(ytr / ysc, -clip, clip), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train(); order = torch.randperm(len(ty), device=dev)
        for s in range(0, len(ty), batch):
            b = order[s:s + batch]
            loss = nn.functional.mse_loss(net(tx[b]).squeeze(1), ty[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    def pred(X):
        net.eval()
        with torch.no_grad():
            return net(f(X)).squeeze(1).cpu().numpy() * ysc
    lin = [m for m in net if isinstance(m, nn.Linear)]
    W = [(l.weight.detach().cpu().numpy().T.astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64)) for l in lin]
    W[-1] = (W[-1][0] * ysc, W[-1][1] * ysc)
    return pred, mean, scale, W


def sweep(a):
    z = np.load(NPZ)
    Xtr = np.vstack([z[f'{s}_tr_X'] for s in SETS]); ytr = np.concatenate([z[f'{s}_tr_y'] for s in SETS])
    ho = {s: (z[f'{s}_ho_X'], z[f'{s}_ho_y'], z[f'{s}_ho_spread']) for s in SETS}
    print('train rows', len(ytr), flush=True)
    def score(preds):
        cs, es, ns = [], [], []
        for s in SETS:
            X, y, spr = ho[s]; p = preds[s]
            n, e = edge(p, y, spr)
            cs.append(np.corrcoef(p, y)[0, 1]); es.append(e); ns.append(n)
        return np.mean(cs), np.mean(es), np.mean(ns)
    for ep in a.epochs_list:
        for wd in a.wd_list:
            t0 = time.time(); allp = []; line = []
            for sd in range(3):
                pred = fit(Xtr, ytr, ep, wd, sd)[0]
                pp = {s: pred(ho[s][0]) for s in SETS}; allp.append(pp)
                c, e, n = score(pp); line.append(f'{c:.4f}/{e:.3f}')
            ens = {s: np.mean([p[s] for p in allp], 0) for s in SETS}
            c, e, n = score(ens)
            sc = [score(p) for p in allp]
            print(f'ep {ep:3d} wd {wd:g}  seeds corr/edge {" ".join(line)}  single-mean corr {np.mean([x[0] for x in sc]):.4f} edge {np.mean([x[1] for x in sc]):.3f}'
                  f'  ENS corr {c:.4f} edge {e:.3f} take {n:.4f}  {time.time() - t0:.0f}s', flush=True)


def final(a):
    z = np.load(NPZ if a.frac == 75 else NPZF)
    if a.frac == 75:
        Xtr = np.vstack([z[f'{s}_tr_X'] for s in SETS]); ytr = np.concatenate([z[f'{s}_tr_y'] for s in SETS])
    else:
        Xtr = np.vstack([z[f'{s}_X'] for s in SETS]); ytr = np.concatenate([z[f'{s}_y'] for s in SETS])
    print('train rows', len(ytr), flush=True)
    lst = lambda x: repr(np.round(x, 6).tolist())
    nets = []
    for sd in range(3):
        pred, mean, scale, W = fit(Xtr, ytr, a.epochs, a.wd, sd)
        nets.append((mean, scale, W)); print('seed', sd, 'done', flush=True)
    with open(a.out, 'w') as fh:
        fh.write(f'MLP_ENS_{a.frac} = [\n')
        for mean, scale, W in nets:
            fh.write(f'    ({lst(mean.astype(np.float64))}, {lst(scale.astype(np.float64))}, [\n')
            for w, b in W:
                fh.write(f'        ({lst(w)}, {lst(b)}),\n')
            fh.write('    ]),\n')
        fh.write(']\n')
    print('wrote', a.out)


def bake():
    src = open('strategies/mm_r3_mlp2.py', encoding='utf8').read()
    e75 = open('logs/mlpf_75.py').read(); e100 = open('logs/mlpf_100.py').read()
    marker = 'PRED_TAUS_US = '
    assert src.count(marker) == 1
    src = src.replace(marker, '# exp_r3_mlpf: 3-seed ReLU ensembles (outputs averaged); _75 = first 75 % of round3 + mirror, _100 = all of it.\n'
                      + e75 + e100 + marker)
    old = ("        if self.mlp_on:\n"
           "            self.mlp_mean = np.array(MLP_MEAN)\n"
           "            self.mlp_scale = np.array(MLP_SCALE)\n"
           "            self.mlp_layers = [(np.array(w), np.array(b)) for w, b in MLP_LAYERS]\n")
    new = ("        self.mlp_w = int(params['mlp_w'])\n"
           "        if self.mlp_on:\n"
           "            # mlp_w 0 = the single 75 % net of mm_r3_mlp2 (parity), 75 / 100 = the 3-seed ensembles\n"
           "            ens = {0: [(MLP_MEAN, MLP_SCALE, MLP_LAYERS)], 75: MLP_ENS_75, 100: MLP_ENS_100}[self.mlp_w]\n"
           "            self.mlp_nets = [(np.array(mu), np.array(sc), [(np.array(w), np.array(b)) for w, b in ly]) for mu, sc, ly in ens]\n")
    assert old in src; src = src.replace(old, new)
    i0 = src.index("            h = (np.array((pg[0], pg[1], f70"); i1 = src.index("            p = (1.0 - self.mlp_mix) * p + self.mlp_mix * m")
    new = ("            h0 = np.array((pg[0], pg[1], f70, pg[3], pg[4], f70 * log1p(fw), f70 * af, sg100 * af, f70 * spread, spread,\n"
           "                           imb1, cnt1, log1p(qb), log1p(qa), ltr_sign, ltr_lsz, (ltp - mid2 * 0.5) if ltp > 0 else 0.0,\n"
           "                           (mid2 - rm2) * 0.5, self.pr_ev10, s100, self.pr_s1000,\n"
           "                           log1p((now - self.pr_ttr) / 1000.0) if self.pr_ttr >= 0 else 10.0))\n"
           "            m = 0.0\n"
           "            for mu, sc, ls in self.mlp_nets:\n"
           "                h = np.clip((h0 - mu) / sc, -8.0, 8.0)\n"
           "                h = np.maximum(h @ ls[0][0] + ls[0][1], 0.0)\n"
           "                h = np.maximum(h @ ls[1][0] + ls[1][1], 0.0)\n"
           "                m += float((h @ ls[2][0] + ls[2][1])[0])\n"
           "            m /= len(self.mlp_nets)\n")
    src = src[:i0] + new + src[i1:]
    src = src.replace("        'mlp_on': 1,\n", "        'mlp_on': 1,\n        'mlp_w': 100,\n", 1)
    assert "'mlp_w': 100" in src
    open('strategies/exp_r3_mlpf.py', 'w', encoding='utf8').write(src)
    print('baked')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--frac', type=int, default=75); ap.add_argument('--out', default=None)
    ap.add_argument('--epochs_list', type=int, nargs='+', default=[10, 20, 40])
    ap.add_argument('--wd_list', type=float, nargs='+', default=[1e-3, 1e-2, 1e-1])
    a = ap.parse_args()
    if a.out is None:
        a.out = f'logs/mlpf_{a.frac}.py'
    {'build': build, 'sweep': lambda: sweep(a), 'final': lambda: final(a), 'bake': bake}[a.mode]()
