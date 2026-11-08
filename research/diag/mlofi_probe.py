"""Multi-level order-flow imbalance (Xu-Gould-Howison / Kolm deep-OFI idea) vs our own features.
Rebuilds the book, accumulates signed flow per depth bucket (0..4 ticks from the same-side best):
bid add +v, bid removal -v, ask add -v, ask removal +v (cancels and taker consumption both count).
Features = flow over 50 ms / 200 ms / 1 s windows (15 cols). Test: OLS on own npz features vs own + MLOFI,
fit on the first 75 % of rows, holdout corr on the last 25 %.
usage: python -m research.diag.mlofi_probe TAPE.csv NPZ"""
import sys
import numpy as np, pandas as pd

tape, npz = sys.argv[1], sys.argv[2]
d = pd.read_csv(tape)
T = d.Time.values; OP = (d.Operation.values == 'Insert'); OID = d.OrderId.values
SD = d.Side.values; V = d.Volume.fillna(0).values.astype(int); P = d.Price.fillna(0).values.astype(int)
IOC = (d.Lifespan.values == 'F')
n = len(d); K = 5
bids, asks, orders = {}, {}, {}
mid = np.zeros(n); C = np.zeros((n, K)); acc = np.zeros(K)
bb, ba = 0, 10**9
for i in range(n):
    bb = max(bids) if bids else 0; ba = min(asks) if asks else 10**9
    if OP[i]:
        s, v, p = SD[i], V[i], P[i]
        if s == 'B':
            while v > 0 and asks and min(asks) <= p:
                a = min(asks); q = asks[a]; take = min(v, q); v -= take
                acc[min(max(a - ba, 0), K - 1)] += take
                if take == q: del asks[a]
                else: asks[a] = q - take
            if v > 0 and not IOC[i]:
                bids[p] = bids.get(p, 0) + v; orders[OID[i]] = ('B', p, v)
                acc[min(max(bb - p, 0), K - 1)] += v
        else:
            while v > 0 and bids and max(bids) >= p:
                b = max(bids); q = bids[b]; take = min(v, q); v -= take
                acc[min(max(bb - b, 0), K - 1)] -= take
                if take == q: del bids[b]
                else: bids[b] = q - take
            if v > 0 and not IOC[i]:
                asks[p] = asks.get(p, 0) + v; orders[OID[i]] = ('A', p, v)
                acc[min(max(p - ba, 0), K - 1)] -= v
    else:
        o = orders.pop(OID[i], None)
        if o:
            s, p, v = o
            book = bids if s == 'B' else asks
            if p in book:
                rem = min(v, book[p]); book[p] -= rem
                if book[p] <= 0: del book[p]
                if s == 'B': acc[min(max(bb - p, 0), K - 1)] -= rem
                else: acc[min(max(p - ba, 0), K - 1)] += rem
    C[i] = acc
    b_ = max(bids) if bids else 0; a_ = min(asks) if asks else 10**9
    mid[i] = (b_ + a_) / 2 if bids and asks else (mid[i - 1] if i else 0)

z = np.load(npz); X = z['X']; tz = z['t'] / 1e6
k = np.searchsorted(T, tz, side='right') - 1
F = []
for w in (0.05, 0.2, 1.0):
    j = np.searchsorted(T, tz - w, side='right') - 1
    F.append(C[k] - C[np.maximum(j, 0)])
F = np.hstack(F)
F = np.sign(F) * np.log1p(np.abs(F))
ok = np.isfinite(X).all(1)
X, F, k = X[ok], F[ok], k[ok]
cut = int(len(k) * 0.75)
def hold(A, y):
    A = np.c_[A, np.ones(len(A))]
    beta = np.linalg.lstsq(A[:cut], y[:cut], rcond=None)[0]
    return np.corrcoef(A[cut:] @ beta, y[cut:])[0, 1]
print(f'events {n:,} rows {len(k):,}')
for h in (0.2, 1.0):
    jf = np.searchsorted(T, T[k] + h, side='right') - 1
    y = mid[jf] - mid[k]
    y = np.clip(y, -20, 20)
    print(f'h{int(h*1000)} own {hold(X, y):.4f}  own+MLOFI {hold(np.c_[X, F], y):.4f}  MLOFI-only {hold(F, y):.4f}')
