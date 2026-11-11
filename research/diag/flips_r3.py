"""Diagnostic only: flip anatomy of a strategy (inventory sign changes) -- how, cost, signal, spread, pnl until the next flip.

    python -X utf8 -m research.diag.flips_r3 strategies/mm_r3_ms.py data/round3.csv [json_overrides]
"""
from __future__ import annotations

import json
import sys
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.simulator import ExchangeSimulator  # noqa: E402
from research.run import load_strategy_class, simulation_config  # noqa: E402


def record(strat, csv, overrides):
    cls = load_strategy_class(Path(strat))
    params = dict(cls.PARAMS)
    params.update(overrides)
    U = {k: array('q') for k in ('ts', 'inv', 'mid2', 'bb', 'ba', 'h0', 'h1')}
    UF = {k: array('d') for k in ('fs', 'ps', 'dr')}
    F = {k: array('q') for k in ('ts', 'sign', 'qty', 'px', 'ioc')}

    class Rec(cls):
        PARAMS = params

        def __init__(self):
            super().__init__()
            self._now = 0

        def on_order_book_change(self, snapshot, trades):
            self._now = snapshot.timestamp_us
            super().on_order_book_change(snapshot, trades)
            bb = ba = -1
            for v in snapshot.bids:
                if v.owner == 'market':
                    bb = v.price
                    break
            for v in snapshot.asks:
                if v.owner == 'market':
                    ba = v.price
                    break
            U['ts'].append(self._now); U['inv'].append(self.inv); U['mid2'].append(self.fair_prev_mid2)
            U['bb'].append(bb); U['ba'].append(ba); U['h0'].append(self.hole[0]); U['h1'].append(self.hole[1])
            pts = self.gate_pts
            UF['dr'].append((self.gate_mid2 - pts[0][1]) / 2.0 if pts else 0.0)
            UF['fs'].append(self.fair_sig); UF['ps'].append(self.pred_sig)

        def on_private_order_update(self, event):
            o = self.orders.get(event.order_id)
            st = event.status.name if hasattr(event.status, 'name') else str(event.status)
            if o is not None and st in ('FILLED', 'PARTIALLY_FILLED'):
                F['ts'].append(self._now); F['sign'].append(o.sign); F['qty'].append(event.filled_quantity)
                F['px'].append(event.fill_price); F['ioc'].append(1 if o.ioc else 0)
            super().on_private_order_update(event)

    res = ExchangeSimulator(strategy=Rec(), config=simulation_config()).run(Path(csv))
    out = {k: np.frombuffer(v, dtype=np.int64) for k, v in U.items()}
    out.update({k: np.frombuffer(v, dtype=np.float64) for k, v in UF.items()})
    out.update({'f_' + k: np.frombuffer(v, dtype=np.int64) for k, v in F.items()})
    out['pnl'] = float(res.pnl)
    return out


def ffill(a):
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def analyse(d):
    ts = d['ts']; inv = d['inv']; bb = d['bb'].astype(float); ba = d['ba'].astype(float)
    n = len(ts)
    ok = (bb > 0) & (ba > 0) & (ba - bb <= 3)
    rm = ffill(np.where(ok, (bb + ba) / 2, np.nan))
    rm = np.nan_to_num(rm, nan=0.0)
    sp = ba - bb
    sgn = np.sign(inv)
    side = 0
    ends = []
    for i in range(n):
        s = sgn[i]
        if s != 0 and s != side:
            if side != 0:
                ends.append(i)
            side = s
    ends = np.array(ends)
    f_ts = d['f_ts']; f_sign = d['f_sign']; f_qty = d['f_qty']; f_px = d['f_px']; f_ioc = d['f_ioc']
    fi = np.maximum(np.searchsorted(ts, f_ts, side='right') - 1, 0)
    edge = (rm[fi] - f_px) * f_sign * f_qty
    drm = np.diff(rm, append=rm[-1])
    hold_cum = np.concatenate([[0], np.cumsum(inv * drm)])
    rows = []
    prev_end = 0
    for j, e in enumerate(ends):
        new = int(sgn[e])
        o = e
        # walk back while still moving away from the old side's extreme (|inv| < 90 on the way)
        while o > 0 and abs(inv[o - 1]) < 90 and ts[e] - ts[o - 1] < 3_000_000:
            o -= 1
        o = max(o - 1, 0)
        in_win = (f_ts >= ts[o]) & (f_ts <= ts[e]) & (f_sign == new)
        q = f_qty[in_win].sum()
        if q == 0:
            continue
        tq = f_qty[in_win & (f_ioc == 1)].sum()
        avgpx = (f_px[in_win] * f_qty[in_win]).sum() / q
        nxt = ends[j + 1] if j + 1 < len(ends) else n - 1
        seg = (f_ts >= ts[e]) & (f_ts < ts[nxt])
        seg_pnl = hold_cum[nxt] - hold_cum[e] + edge[seg].sum()
        win_edge = edge[in_win].sum()
        t1 = min(np.searchsorted(ts, ts[e] + 1_000_000), n - 1)
        mk1 = new * (rm[t1] - rm[e])
        mk1_px = new * (rm[t1] - avgpx)
        ps = d['ps'][o]
        rows.append(dict(t=ts[e], dur_prev=(ts[e] - ts[prev_end]) / 1e6 if prev_end else 1e9, seg_dur=(ts[nxt] - ts[e]) / 1e6,
                         dirn=new, lots=q, tfrac=tq / q, spread=sp[o], hole=(d['h0'] if new > 0 else d['h1'])[o],
                         ps_dir=new * ps, fs_dir=new * d['fs'][o], drift_dir=new * d['dr'][o], entry_edge=win_edge / q,
                         seg_pnl=seg_pnl, mk1=mk1, mk1_px=mk1_px, win_pnl=win_edge))
        prev_end = e
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return R, d['pnl']


def table(R, cats):
    print(f"{'category':<28}{'n':>6}{'segpnl':>11}{'mean':>8}{'entry/lot':>10}{'mk1':>7}{'mk1_px':>8}{'lots':>6}")
    for name, m in cats:
        if m.sum() == 0:
            continue
        print(f"{name:<28}{m.sum():>6}{R['seg_pnl'][m].sum():>11,.0f}{R['seg_pnl'][m].mean():>8,.0f}{R['entry_edge'][m].mean():>10.2f}"
              f"{R['mk1'][m].mean():>7.2f}{R['mk1_px'][m].mean():>8.2f}{R['lots'][m].mean():>6.0f}")


def main():
    ov = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    d = record(sys.argv[1], sys.argv[2], ov)
    R, pnl = analyse(d)
    N = len(R['t'])
    print(f"pnl {pnl:,.0f}  flips {N}  sum segpnl {R['seg_pnl'].sum():,.0f}  fills {len(d['f_ts'])}")
    tk = R['tfrac'] >= 0.8; ps_ = R['tfrac'] <= 0.05; mx = ~tk & ~ps_
    sd = R['seg_dur']; pv = R['dur_prev']; pd_ = R['ps_dir']; spr = R['spread']
    cats = [('all', np.ones(N, bool)), ('taker (>=80% lots)', tk), ('passive (<=5% taker)', ps_), ('mixed', mx),
            ('seg<0.5s (whipsaw)', sd < 0.5), ('seg 0.5-2s', (sd >= 0.5) & (sd < 2)), ('seg 2-10s', (sd >= 2) & (sd < 10)), ('seg>=10s', sd >= 10),
            ('taker spread<=1', tk & (spr <= 1)), ('taker spread==2', tk & (spr == 2)), ('taker spread>=3', tk & (spr >= 3)),
            ('passive spread<=1', ps_ & (spr <= 1)), ('passive spread>=2', ps_ & (spr >= 2)),
            ('taker ps<0.5', tk & (pd_ < 0.5)), ('taker ps 0.5-1', tk & (pd_ >= 0.5) & (pd_ < 1)), ('taker ps>=1', tk & (pd_ >= 1)),
            ('passive ps<0.3', ps_ & (pd_ < 0.3)), ('passive ps 0.3-0.6', ps_ & (pd_ >= 0.3) & (pd_ < 0.6)), ('passive ps>=0.6', ps_ & (pd_ >= 0.6)),
            ('mixed ps<0.5', mx & (pd_ < 0.5)), ('mixed ps>=0.5', mx & (pd_ >= 0.5)),
            ('taker hole>=3', tk & (R['hole'] >= 3)), ('taker hole<3', tk & (R['hole'] < 3)),
            ('taker prev<0.5s', tk & (pv < 0.5)), ('taker prev 0.5-2s', tk & (pv >= 0.5) & (pv < 2)), ('taker prev>=2s', tk & (pv >= 2)),
            ('taker drift against>=5', tk & (R['drift_dir'] <= -5)), ('taker drift with>=5', tk & (R['drift_dir'] >= 5)),
            ('any drift against>=5', R['drift_dir'] <= -5)]
    table(R, cats)
    np.savez(Path(__file__).with_name('_flips_tmp.npz'), **R)


if __name__ == '__main__':
    main()
