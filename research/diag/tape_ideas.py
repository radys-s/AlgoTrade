"""Offline IC of outside ideas on a tape: insert-price micro-EWMA (fs30), 50 ms cancel imbalance (ci50), post-sweep move.
Rebuilds the book (price-time, F = IOC), target = mid(t+h) - mid(t). Incremental test: corr with the residual of the
target after OLS on the strategy's own 26 features (mlp_dump npz).
usage: python -m research.diag.tape_ideas TAPE.csv NPZ"""
import sys, math
import numpy as np, pandas as pd

tape, npz = sys.argv[1], sys.argv[2]
d = pd.read_csv(tape)
T = d.Time.values; OP = (d.Operation.values == 'Insert'); OID = d.OrderId.values
SD = d.Side.values; V = d.Volume.fillna(0).values.astype(int); P = d.Price.fillna(0).values.astype(int)
IOC = (d.Lifespan.values == 'F')
bids, asks, orders = {}, {}, {}
n = len(d)
mid = np.zeros(n); fs = np.zeros(n); ci = np.zeros(n); near = np.zeros(n)
ew, ewt, ewinit = 0.0, 0.0, False
cb = ca = 0.0; ct = 0.0
sweeps = []  # (idx, dir, levels)
bb, ba = 0, 10**9
def best():
    return (max(bids) if bids else 0), (min(asks) if asks else 10**9)
for i in range(n):
    t = T[i]
    # decay cancel accumulators (50 ms) and ewma clock
    dec = math.exp(-(t - ct) / 0.05); cb *= dec; ca *= dec; ct = t
    if OP[i]:
        s, v, p = SD[i], V[i], P[i]
        if s == 'B':
            lv = 0
            while v > 0 and asks and min(asks) <= p:
                a = min(asks); q = asks[a]; take = min(v, q); v -= take; lv += 1
                if take == q: del asks[a]
                else: asks[a] = q - take
            if IOC[i]:
                if lv >= 3: sweeps.append((i, 1, lv))
            elif v > 0:
                bids[p] = bids.get(p, 0) + v; orders[OID[i]] = ('B', p, v)
                x = p + 0.5
                if ewinit:
                    w = math.exp(-(t - ewt) / 0.03); ew = w * ew + (1 - w) * x
                else: ew, ewinit = x, True
                ewt = t
        else:
            lv = 0
            while v > 0 and bids and max(bids) >= p:
                b = max(bids); q = bids[b]; take = min(v, q); v -= take; lv += 1
                if take == q: del bids[b]
                else: bids[b] = q - take
            if IOC[i]:
                if lv >= 3: sweeps.append((i, -1, lv))
            elif v > 0:
                asks[p] = asks.get(p, 0) + v; orders[OID[i]] = ('A', p, v)
                x = p - 0.5
                if ewinit:
                    w = math.exp(-(t - ewt) / 0.03); ew = w * ew + (1 - w) * x
                else: ew, ewinit = x, True
                ewt = t
    else:
        o = orders.pop(OID[i], None)
        if o:
            s, p, v = o
            book = bids if s == 'B' else asks
            if p in book:
                rem = min(v, book[p]); book[p] -= rem
                if book[p] <= 0: del book[p]
                b_, a_ = best()
                if (s == 'B' and p >= b_ - 2) or (s == 'A' and p <= a_ + 2):
                    if s == 'B': cb += rem
                    else: ca += rem
    b_, a_ = best()
    m = (b_ + a_) / 2 if bids and asks else (mid[i - 1] if i else 0)
    mid[i] = m; fs[i] = ew - m; ci[i] = (ca - cb) / (ca + cb + 1)
# targets
def fwd(h):
    j = np.searchsorted(T, T + h, side='right') - 1
    return mid[j] - mid
z = np.load(npz); X = z['X']; tz = z['t']
k = np.searchsorted(T, tz / 1e6, side='right') - 1
print(f'events {n:,}, npz rows {len(tz):,}, sweeps>=3 levels {len(sweeps):,}')
for h in (0.1, 0.2, 1.0):
    y = fwd(h)[k]
    ok = np.isfinite(X).all(1)
    A = np.c_[X[ok], np.ones(ok.sum())]; beta = np.linalg.lstsq(A, y[ok], rcond=None)[0]; res = y[ok] - A @ beta
    r2 = 1 - res.var() / y[ok].var()
    for nm, s in (('fs30', fs[k]), ('ci50', ci[k])):
        s = s[ok]
        print(f'h{int(h*1000):5d} {nm}: corr {np.corrcoef(s, y[ok])[0,1]:+.4f}  resid-corr {np.corrcoef(s, res)[0,1]:+.4f}  (own feats R2 {r2:.3f})')
for h in (0.1, 1.0):
    f = fwd(h)
    sw = np.array([dr * f[i] for i, dr, _ in sweeps])
    print(f'post-sweep (>=3 lv) signed mid move h{int(h*1000)}: mean {sw.mean():+.3f} ticks (neg = reversion), n {len(sw)}')
if len(sys.argv) > 3:  # append fs30 + ci50 as columns 26, 27 -> new npz
    np.savez(sys.argv[3], X=np.c_[X, fs[k], ci[k]].astype(np.float32), t=tz, mid2=z['mid2'], rm2=z['rm2'])
    print('saved', sys.argv[3])
