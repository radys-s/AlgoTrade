"""Decision-focused taker filter (exp_r3_dpc). Three steps:
    python -X utf8 -m research.diag.dpc_r3 record round3 out.npz       # strategies/exp_r3_dpc.py with dpc_on 0, logs triggers + near-triggers
    python -X utf8 -m research.diag.dpc_r3 fit rec_round3.npz rec_round3_mirror.npz   # label, ridge on first 75 %, print baked params
Rows: kind 1 = an IOC was sent, kind 0 = near trigger (ft within 0.5 ticks below the threshold, >= 10 ms apart per side).
Label y1s = (robust mid at t+1 s - touch price) * side (ticks/lot, includes the half spread paid); yfill uses the real fill price.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.simulator import ExchangeSimulator
from simulator.models import OrderStatus
from research.run import load_strategy_class, simulation_config

FN = ['edge', 'pred', 'fair', 'pf20', 'pf70', 'pf300', 'spread', 'lq_take', 'lq_opp', 'dn', 'hole', 'since',
      'inv', 'drift', 'lt', 'vol']
COLS = ['t', 'kind', 'side', 'touch', 'fill_qty', 'fill_px'] + FN


def make_recorder(cls):
    class Rec(cls):
        def __init__(self) -> None:
            super().__init__()
            self.rec = []
            self.rec_ids = {}
            self.r_pending = -1
            self.r_near = [-10**12, -10**12]

        def _fq(self, now, bb, ba, fshift, pull_bid, pull_ask):
            ref2 = self.fair_prev_mid2
            mid = ref2 / 2.0 if ref2 > 0 else (bb + ba) / 2.0
            ft = mid + self.pred_scale * self.pred_sig + fshift
            tk = self.fq_take_k
            self._dp_track(now, bb, ba)
            rows = []
            for k, s, x, touch in ((0, 1, ft - tk - ba, ba), (1, -1, bb - ft - tk, bb)):
                if x >= -0.5:
                    rows.append((k, s, x, touch, self._dp_feats(now, bb, ba, s, x)))
            pre = self.n_takes
            self.r_pending = -1
            out = super()._fq(now, bb, ba, fshift, pull_bid, pull_ask)
            if rows:
                sent = self.n_takes > pre
                for k, s, x, touch, f in rows:
                    if x >= 0 and sent and self.dp_last_send == now and (self.r_side == s):
                        self.rec.append([now, 1, s, touch, 0.0, 0.0] + f)
                        self.r_pending = len(self.rec) - 1
                    elif x < 0 and now - self.r_near[k] >= 10_000:
                        self.r_near[k] = now
                        self.rec.append([now, 0, s, touch, 0.0, 0.0] + f)
            return out

        def _send(self, k, sign, side, px, qty, ioc, j, *a, **kw):
            self.r_side = sign
            oid = super()._send(k, sign, side, px, qty, ioc, j, *a, **kw)
            if oid is not None and ioc and self.r_pending_next:
                pass
            self.r_last_oid = oid
            return oid

        r_pending_next = False
        r_side = 0
        r_last_oid = None

        def on_private_order_update(self, event) -> None:
            i = self.rec_ids.get(event.order_id)
            if i is not None and event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED) and event.filled_quantity:
                r = self.rec[i]
                q = event.filled_quantity
                r[5] = (r[5] * r[4] + q * event.fill_price) / (r[4] + q)
                r[4] += q
            super().on_private_order_update(event)
    return Rec


def record(name: str, out: str) -> None:
    t0 = time.time()
    cls = make_recorder(load_strategy_class('strategies/exp_r3_dpc.py'))
    s = cls()
    # map each sent IOC to the row appended in the same callback
    orig = s._send
    def send(k, sign, side, px, qty, ioc, j, *a, **kw):
        oid = orig(k, sign, side, px, qty, ioc, j, *a, **kw)
        if oid is not None and ioc:
            s.pending_oid = oid
        return oid
    s._send = send
    base_fq = s._fq
    def fq(now, bb, ba, fshift, pb, pa):
        s.pending_oid = None
        n0 = len(s.rec)
        r = base_fq(now, bb, ba, fshift, pb, pa)
        if s.pending_oid is not None:
            for i in range(len(s.rec) - 1, n0 - 1, -1):
                if s.rec[i][1] == 1:
                    s.rec_ids[s.pending_oid] = i
                    break
        return r
    s._fq = fq
    res = ExchangeSimulator(strategy=s, config=simulation_config()).run(Path(f'data/{name}.csv'))
    arr = np.array(s.rec, dtype=float)
    np.savez(out, data=arr, cols=np.array(COLS))
    print(f'pnl={res.pnl:.0f} fills={res.fills} rows={len(arr)} trig={int((arr[:,1]==1).sum())} filled={int((arr[:,4]>0).sum())} {time.time()-t0:.0f}s')


if __name__ == '__main__':
    if sys.argv[1] == 'record':
        record(sys.argv[2], sys.argv[3])


def robust_mid(name):
    z = np.load(f'data/features/{name}.npz')
    ts = z['ts_us']; b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    idx = np.where(ok)[0]
    rm = (b + a) / 2

    def at(t):
        k = np.searchsorted(ts, t, side='right') - 1
        j = np.searchsorted(idx, np.maximum(k, 0), side='right') - 1
        return rm[idx[np.maximum(j, 0)]]
    return at, ts


def load(name):
    z = np.load(f'logs/dpc_rec_{name}.npz')
    d = {c: z['data'][:, i] for i, c in enumerate(COLS)}
    at, ts = robust_mid(name)
    d['y1s'] = (at(d['t'] + 1_000_000) - d['touch']) * d['side']
    d['y200'] = (at(d['t'] + 205_000) - d['touch']) * d['side']
    t_end = ts[0] + 0.75 * (ts[-1] - ts[0])
    d['train'] = d['t'] <= t_end - 1_000_000
    d['hold'] = d['t'] > t_end
    return d


def fit(target='y1s', lam=30.0):
    ds = [load(n) for n in ('round3', 'round3_mirror')]
    X = np.vstack([np.column_stack([d[f] for f in FN]) for d in ds])
    y = np.concatenate([d[target] for d in ds])
    kind = np.concatenate([d['kind'] for d in ds])
    tr = np.concatenate([d['train'] for d in ds]); ho = np.concatenate([d['hold'] for d in ds])
    X = np.clip(X, -1e4, 1e4)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z[tr]; yy = y[tr]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (yy - yy.mean()))
    b = yy.mean()
    p = Z @ w + b
    print(f'target {target}: rows tr {tr.sum()} ho {ho.sum()}  trig tr {(tr&(kind==1)).sum()} ho {(ho&(kind==1)).sum()}')
    for nm, m in (('train', tr), ('hold', ho)):
        for kn in (1, 0):
            mm = m & (kind == kn)
            print(f'  {nm} kind {kn}: n={mm.sum()} mean y={y[mm].mean():+.3f} corr={np.corrcoef(p[mm], y[mm])[0,1]:+.3f} '
                  f'pred q10/50/90={np.quantile(p[mm],[.1,.5,.9]).round(2)}')
    mm = ho & (kind == 1)
    for q in (0.1, 0.2, 0.3):
        thr = np.quantile(p[tr & (kind == 1)], q)
        sk = mm & (p < thr)
        print(f'  hold triggers with pred < q{int(q*100)} ({thr:+.2f}): n={sk.sum()} mean y={y[sk].mean():+.3f} (rest {y[mm & ~sk].mean():+.3f})')
    for hi in (0.5, 1.0, 1.5):
        mn = ho & (kind == 0) & (p > hi)
        print(f'  hold near with pred > {hi}: n={mn.sum()} mean y={y[mn].mean() if mn.any() else 0:+.3f}')
    print('weights', dict(zip(FN, w.round(3))))
    print('DPC', 'dpc_w', list(map(float, w)), 'dpc_mu', list(map(float, mu)), 'dpc_sd', list(map(float, sd)), 'dpc_b', float(b))
    return w, mu, sd, b


if __name__ == '__main__' and sys.argv[1] == 'fit':
    fit(sys.argv[2] if len(sys.argv) > 2 else 'y1s', float(sys.argv[3]) if len(sys.argv) > 3 else 30.0)
