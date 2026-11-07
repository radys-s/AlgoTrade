import numpy as np, pandas as pd
z=np.load('data/features/round2.npz'); d=pd.read_csv('data/round2.csv',usecols=['Side','Lifespan'])
ts=z['ts_us'].astype(np.int64); bid=z['bid_px'].astype(float); ask=z['ask_px'].astype(float)
mid=pd.Series(np.where(np.isfinite(ask-bid)&(ask-bid<=3),(bid+ask)/2,np.nan)).ffill().values; ltp=z['last_trade_px'].astype(float); tv=z['trade_vol']
pb=np.r_[np.nan,bid[:-1]]; pa=np.r_[np.nan,ask[:-1]]
f=(d.Lifespan=='F').values; buy=(d.Side=='B').values
idx=np.where(f&(tv!=0)&np.isfinite(pa)&np.isfinite(pb)&(pa-pb<=3))[0]
def fut(h):
    j=np.searchsorted(ts,ts[idx]+h); j=np.minimum(j,len(ts)-1); return mid[j]
H={'1s':1_000_000,'5s':5_000_000,'30s':30_000_000}
F={k:fut(v) for k,v in H.items()}
b=buy[idx]
reach=np.where(b,ltp[idx]-pa[idx],pb[idx]-ltp[idx])  # ticks past touch the sweep executed
print('reach dist',np.bincount(np.clip(reach,0,12).astype(int)))
print('k  events  ' + '  '.join(f'mk{h}' for h in H))
for k in range(0,11):
    m=reach>=k
    p=np.where(b,pa[idx]+k,pb[idx]-k)  # our resting price at depth k, filled as maker
    row=[]
    for h in H:
        mk=np.where(b,p-F[h],F[h]-p)[m]   # we sold to buyer / bought from seller
        mk=mk[np.isfinite(mk)]; row.append(f'{mk.mean():+.2f}/{np.median(mk):+.1f}')
    print(k,m.sum(),'  '.join(row))
