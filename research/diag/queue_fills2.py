import sys
import numpy as np
from pathlib import Path
from research.run import load_strategy_class
from simulator import main as sm
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.models import OrderStatus
Base = load_strategy_class(sys.argv[2])
class Rec(Base):
    def __init__(self):
        super().__init__(); self.ts=[]; self.mid=[]; self.f=[]; self.ahead={}; self.behind={}; self.lvl={}
    def on_order_book_change(self, snap, trades):
        if snap.bids and snap.asks:
            self.ts.append(snap.timestamp_us); self.mid.append((snap.bids[0].price+snap.asks[0].price)/2)
            for side in (snap.bids, snap.asks):
                px=None; acc=0; mine=[]
                for o in side:
                    if o.price!=px:
                        if mine:
                            for oid,a in mine: self.behind[oid]=acc-a; self.lvl[oid]=acc
                        px=o.price; acc=0; mine=[]
                    if o.owner!='market': mine.append((o.order_id, acc)); self.ahead[o.order_id]=acc
                    else: acc+=o.quantity
                for oid,a in mine: self.behind[oid]=acc-a; self.lvl[oid]=acc
        super().on_order_book_change(snap, trades)
    def on_private_order_update(self, ev):
        o=self.orders.get(ev.order_id)
        if o is not None and ev.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self.f.append((ev.timestamp_us, o.sign, ev.filled_quantity, ev.fill_price, self.ahead.get(ev.order_id,-1), self.behind.get(ev.order_id,-1), self.lvl.get(ev.order_id,-1), self.inv, o.quantity if hasattr(o,'quantity') else -1, 1 if getattr(o,'parked',False) else 0))
        super().on_private_order_update(ev)
st=Rec()
cfg=SimulationConfig(max_position=sm.MAX_POSITION, max_potential_position=sm.MAX_POTENTIAL_POSITION, position_breach_penalty_per_second=sm.POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=sm.ORDER_ACCEPT_DELAY_MS)
res=ExchangeSimulator(strategy=st, config=cfg).run(Path(sys.argv[1]))
print('pnl', res.pnl)
ts=np.array(st.ts); mid=np.array(st.mid); F=np.array(st.f,dtype=float); t,s,q,p,ah,bh,lv,inv,oq,pk=F.T
i=np.clip(np.searchsorted(ts,t+1_000_000,side='right')-1,0,len(mid)-1); e=s*(mid[i]-p)
def rep(name, x, edges):
    for lo,hi in zip(edges[:-1],edges[1:]):
        k=(x>=lo)&(x<hi); print(f'{name} [{lo},{hi}) lots={q[k].sum():>9.0f} share={q[k].sum()/q.sum():.2f} 1s edge={((e[k]*q[k]).sum()/max(1,q[k].sum())):+.3f}')
rep('ahead (market lots before us, last snapshot)', ah, [-1,0,1,50,200,500,1000,1e9])
rep('behind (market lots after us)', bh, [-1,0,1,50,200,500,1e9])
rep('level market qty', lv, [-1,0,1,100,300,600,1000,1e9])

np.savez('logs/fills_'+Path(sys.argv[2]).stem+'.npz', F=F, ts=ts, mid=mid)
rep('inventory before fill * side (+ = adds |I|)', s*inv, [-101,-60,-20,0.5,20,60,101])
rep('parked stub', pk, [0,0.5,2])
for H in (0.1,5):
    j=np.clip(np.searchsorted(ts,t+H*1_000_000,side='right')-1,0,len(mid)-1); eh=s*(mid[j]-p)
    print(f'all fills {H}s edge {(eh*q).sum()/q.sum():+.3f}, total $ {(eh*q).sum():,.0f}')
print('total lots', q.sum(), 'fills', len(q), '1s $', f'{(e*q).sum():,.0f}')
