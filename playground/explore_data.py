"""Explore a tape: look for patterns / flaws in how the generator makes the data.

    uv run python -m research.features data/round2.csv          # once: builds data/features/round2.npz (~20 s)
    uv run python -X utf8 playground/explore_data.py data/round2.csv
    # or open in VS Code and run cell by cell (each "# %%" is a Jupyter-style cell: "Run Cell")

Data you get (all numpy arrays, one row per tape event, state AFTER the event):
    df      the raw CSV: Time, Operation (Insert/Cancel), OrderId, Side (B/A), Volume, Price, Lifespan (G limit / F IOC)
    ts      event time in microseconds
    bb, ba  best bid / best ask after the event (empty-side rows forward-filled)
    mid     (bb + ba) / 2
    at(t)   index of the last event at or before time t (us) -> use mid[at(ts + 1_000_000)] for "mid 1 s later"

MAIN FINDING (2026-11-07, radys): the generator places new limit orders around a HIDDEN FAIR PRICE, and the
visible book lags it. A time-decayed average of (new limit order price - side offset - mid) predicts the mid
move over the next 100 ms with corr ~ +0.4 (R2) and the next 1 s with ~ +0.38 (R1). Section 3 below.
Ideas to look at yourself are marked "TRY".
"""

# %% 0. load
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.csv') else 'data/round2.csv'
NPZ = CSV.replace('data/', 'data/features/').replace('.csv', '.npz')
df = pd.read_csv(CSV)
z = np.load(NPZ)
ts = z['ts_us'].astype(np.int64)
bb = z['bid_px'].astype(float); ba = z['ask_px'].astype(float)
bad = (bb <= 0) | (ba <= 0) | (ba <= bb)           # one side of the book empty (happens ~1,200 times on R2)
bb = pd.Series(np.where(bad, np.nan, bb)).ffill().bfill().values
ba = pd.Series(np.where(bad, np.nan, ba)).ffill().bfill().values
mid = (bb + ba) / 2
pm = np.r_[mid[0], mid[:-1]]                        # mid just BEFORE each event
px = df.Price.values.astype(float)
sd = np.where(df.Side.values == 'B', 1, -1)
isG = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
isF = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'F')
def at(t): return np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1)
print(f'{CSV}: {len(df):,} rows, {ts[-1]/1e6:,.0f} s, empty-side rows {bad.sum()}')

# %% 1. basic facts (already checked: nothing exploitable here, but verify on new rounds)
# - OrderIds strictly increasing, no gaps -> every new market order is "id > last seen id"
# - IOC arrival times uniform (no clock schedule), next IOC same side P = 0.50 (no persistence)
# - limit order lifetime ~ exponential, mean ~440 ms -> the whole queue in front of you refreshes in ~1 s
ins = df[isG]; can = df[df.Operation == 'Cancel']
life = can[['OrderId', 'Time']].merge(ins[['OrderId', 'Time']], on='OrderId', suffixes=('_c', '_i'))
life = (life.Time_c - life.Time_i) * 1e3
print('limit lifetime ms quantiles 10/50/90/99:', np.percentile(life, [10, 50, 90, 99]).round(0))
print('limit sizes most common:', ins.Volume.value_counts().head(6).to_dict())
# TRY: does lifetime depend on size / distance from touch / time of day? (if big orders live longer you know
#      which queue fronts will vanish)

# %% 2. where new limit orders are placed vs the touch (0 = at touch, 1 = one tick behind, -1 = improves)
rel = np.where(sd > 0, np.r_[bb[0], bb[:-1]] - px, px - np.r_[ba[0], ba[:-1]])
v, c = np.unique(rel[isG], return_counts=True); m = (v > -4) & (v < 6)
print('limit placement vs own touch:', dict(zip(v[m].astype(int), c[m])))

# %% 3. HIDDEN FAIR PRICE signal (the main finding)
off_b = np.mean((px - pm)[isG & (sd > 0)]); off_a = np.mean((px - pm)[isG & (sd < 0)])
print(f'avg limit insert price - mid: bids {off_b:+.2f}, asks {off_a:+.2f}')
dev = np.where(isG, px - np.where(sd > 0, off_b, off_a) - pm, np.nan)   # each insert = noisy look at fair - mid
TAU_MS = 50                                                            # TRY 10..300
idx = np.where(isG)[0]; dec = np.exp(-np.diff(ts[idx], prepend=ts[idx[0]]) / (TAU_MS * 1e3))
s = w = 0.0; out = np.empty(len(idx))
for i, (a, x) in enumerate(zip(dec, dev[idx])):
    s = s * a + x; w = w * a + 1.0; out[i] = s / w
sig = np.full(len(df), np.nan); sig[idx] = out; sig = pd.Series(sig).ffill().fillna(0).values
smp = np.arange(1000, len(df) - 1000, 5); k5 = at(ts[smp] + 5000)          # we can act 5 ms later
for h in (100_000, 1_000_000, 5_000_000):
    print(f'corr(signal, mid move from t+5ms to t+{h/1e3:.0f}ms) = '
          f'{np.corrcoef(sig[smp], mid[at(ts[smp] + h)] - mid[k5])[0, 1]:+.3f}')
# taker check: buy at ask(t+5ms) if sig > thr, sell at bid(t+h) (and mirror). Overlapping samples -> optimistic n.
for thr in (1.0, 1.5, 2.0):
    kh = at(ts[smp] + 500_000); b = sig[smp] > thr; se = sig[smp] < -thr
    rt = np.r_[bb[kh][b] - ba[k5][b], bb[k5][se] - ba[kh][se]]
    print(f'thr {thr}: samples {len(rt):,}  IOC round trip after 500 ms {rt.mean():+.2f} ticks/lot')
# TRY: better fair estimate (median of recent inserts, weight inserts at touch more, include cancels:
#      orders far from fair get cancelled?), and whether IOC prices (not visible to us!) also centre on fair.

# %% 4. IOC limit price vs touch -> future move (IOC prices are NOT visible to a strategy, only their trades)
iF = np.where(isF)[0]
reach = np.where(sd[iF] > 0, px[iF] - ba[iF - 1], bb[iF - 1] - px[iF])
for r in (-1, 0, 1, 2, 3):
    m = reach == r; k = at(ts[iF[m]] + 1_000_000)
    print(f'IOC priced {r:+d} beyond touch: n {m.sum():6d}  signed mid move 1 s {np.mean(sd[iF][m]*(mid[k]-mid[iF][m])):+.2f}')
# TRY: an IOC that did not fully fill (trade volume < IOC volume) reveals its price limit = fair-ish. Visible!

# %% 5. TRY yourself
# - mid move after the book side becomes EMPTY (bad rows): who refills, at what price?
# - spread 2+ ticks: where do new orders land relative to fair? (quote inside the spread when fair is on our side)
# - per 5-minute bucket: does fair drift (R2 crash) show up in the insert prices before the mid?
