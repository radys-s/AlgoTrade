"""Diagnostic only, never shipped: how fast and how cheaply the inventory follows the taker signal (F_hat).

    uv run python -X utf8 -m research.diag.flip_exec strategies/mm_r2_pred2.py data/round2.csv [json_overrides] [out.npz]

Records per book update: inventory, outstanding per side, fair_sig, the taker signal (pred_sig), mid, touch,
IOC sends and every fill, then reports (a) time spent with the inventory against the signal and the holding pnl
there, (b) latency from a take condition against the inventory to the inventory flip, (c) cost per flip:
crossing, IOC partial fills and room blocked by our own resting quotes / in-flight cancels.
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


def record(strat: Path, csv: Path, overrides: dict) -> dict:
    cls = load_strategy_class(strat)
    params = dict(cls.PARAMS)
    params.update(overrides)
    U = {k: array('q') for k in ('ts', 'inv', 'o0', 'o1', 'mid2', 'bb', 'ba', 'tk', 'bl', 'a0', 'a1')}
    UF = {k: array('d') for k in ('fs', 'ps')}
    F = {k: array('q') for k in ('ts', 'sign', 'qty', 'px', 'ioc')}
    S = {k: array('q') for k in ('ts', 'sign', 'qty', 'px', 'oid')}

    class Rec(cls):
        PARAMS = params

        def __init__(self):
            super().__init__()
            self._now = 0
            self._ioc_req = {}

        def on_order_book_change(self, snapshot, trades):
            self._now = snapshot.timestamp_us
            t0 = self.n_takes
            b0 = self.n_blocked
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
            U['ts'].append(self._now); U['inv'].append(self.inv); U['o0'].append(self.out[0]); U['o1'].append(self.out[1])
            U['mid2'].append(self.fair_prev_mid2); U['bb'].append(bb); U['ba'].append(ba)
            U['tk'].append(self.n_takes - t0); U['bl'].append(self.n_blocked - b0)
            U['a0'].append(self.act_qty[0]); U['a1'].append(self.act_qty[1])
            UF['fs'].append(self.fair_sig); UF['ps'].append(self.pred_sig)

        def _send(self, k, sign, side, px, qty, ioc, j):
            oid = super()._send(k, sign, side, px, qty, ioc, j)
            if oid is not None and ioc:
                S['ts'].append(self._now); S['sign'].append(sign); S['qty'].append(qty); S['px'].append(px)
                S['oid'].append(len(self._ioc_req))
                self._ioc_req[oid] = len(self._ioc_req)
            return oid

        def on_private_order_update(self, event):
            o = self.orders.get(event.order_id)
            st = event.status.name if hasattr(event.status, 'name') else str(event.status)
            if o is not None and st in ('FILLED', 'PARTIALLY_FILLED'):
                F['ts'].append(self._now); F['sign'].append(o.sign); F['qty'].append(event.filled_quantity)
                F['px'].append(event.fill_price); F['ioc'].append(self._ioc_req.get(event.order_id, -1))
            super().on_private_order_update(event)

    sim = ExchangeSimulator(strategy=Rec(), config=simulation_config())
    res = sim.run(csv)
    out = {k: np.frombuffer(v, dtype=np.int64) for k, v in U.items()}
    out.update({k: np.frombuffer(v, dtype=np.float64) for k, v in UF.items()})
    out.update({'f_' + k: np.frombuffer(v, dtype=np.int64) for k, v in F.items()})
    out.update({'s_' + k: np.frombuffer(v, dtype=np.int64) for k, v in S.items()})
    out['pnl'] = np.array([res.pnl])
    out['take_k'] = np.array([float(params['fq_take_k'])])
    return out


def analyse(d: dict) -> None:
    ts = d['ts']; inv = d['inv']; bb = d['bb'].astype(float); ba = d['ba'].astype(float)
    ok = (bb > 0) & (ba > 0)
    rm = np.where(ok & (ba - bb <= 3), (bb + ba) / 2, np.nan)
    rm = _ffill(rm)
    mid = np.where(d['mid2'] > 0, d['mid2'] / 2.0, (bb + ba) / 2)
    ps = d['ps']; tk = float(d['take_k'][0])
    ft = mid + ps
    take_b = (np.floor(ft - tk) >= ba) & ok
    take_s = (-np.floor(-(ft + tk)) <= bb) & ok
    dt = np.diff(ts, append=ts[-1]).astype(float) / 1e6
    T = dt.sum()
    drm = np.diff(rm, append=rm[-1])
    drm[np.isnan(drm)] = 0.0
    hold = inv * drm
    print(f"pnl {d['pnl'][0]:,.0f}  updates {len(ts):,}  takes {int(d['tk'].sum()):,}  blocked sends {int(d['bl'].sum()):,}")
    print(f"time |I|>=90: {dt[np.abs(inv) >= 90].sum() / T:.1%}   holding pnl total {hold.sum():,.0f}")
    # (a) state vs the taker signal sign, by |signal| bucket
    print('\n(a) inventory vs taker signal ps (= F_hat - mid used by the IOC test); holding pnl = I x d(robust mid)')
    print(f"{'|ps| bucket':>14} {'time%':>7} {'agree%':>7} {'wrong%':>7} {'hold agree':>12} {'hold wrong':>12} {'hold flat':>11}")
    edges = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 1e9]
    sg = np.sign(ps) * np.sign(inv)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (np.abs(ps) >= lo) & (np.abs(ps) < hi)
        tm = dt[m].sum()
        ag = m & (sg > 0) & (np.abs(inv) >= 50)
        wr = m & (sg < 0) & (np.abs(inv) >= 50)
        fl = m & (np.abs(inv) < 50)
        print(f"{lo:>6.2f}-{hi:<7.2f} {tm / T:>7.1%} {dt[ag].sum() / max(tm, 1e-9):>7.1%} {dt[wr].sum() / max(tm, 1e-9):>7.1%}"
              f" {hold[ag].sum():>12,.0f} {hold[wr].sum():>12,.0f} {hold[fl].sum():>11,.0f}")
    # take condition while the inventory points the other way
    wrong_take = (take_b & (inv < 0)) | (take_s & (inv > 0))
    print(f"\ntime with a take condition and inventory against it: {dt[wrong_take].sum():.1f} s ({dt[wrong_take].sum() / T:.2%}),"
          f" holding pnl there {hold[wrong_take].sum():,.0f}; updates {wrong_take.sum():,}")
    room0 = np.where(take_b, 100 - inv - d['o0'], np.where(take_s, 100 + inv - d['o1'], 0))
    tb = take_b | take_s
    print(f"updates with take condition: {tb.sum():,}; of these room<=0: {(room0[tb] <= 0).mean():.1%};"
          f" with |I|<100 (could still trade) and room<=0: {((room0 <= 0) & tb & (np.abs(inv) < 100)).sum():,}")
    # (b) flip episodes: start = first update where a take condition opposes an inventory with |I| >= 50
    print('\n(b) flips: start = first take condition opposing |I| >= 50')
    starts = np.where(((take_b & (inv <= -50)) | (take_s & (inv >= 50))))[0]
    eps = []
    last_end = -1
    for s in starts:
        if s <= last_end:
            continue
        dirn = 1 if take_b[s] else -1
        # find when inventory crosses zero and reaches +-90 in dirn
        j = s
        n = len(ts)
        t0 = ts[s]
        t_zero = t_90 = None
        first_ioc = None
        k = s
        while k < n and ts[k] - t0 < 5_000_000:
            if first_ioc is None and d['tk'][k] > 0:
                first_ioc = ts[k] - t0
            if t_zero is None and inv[k] * dirn >= 0:
                t_zero = ts[k] - t0
            if inv[k] * dirn >= 90:
                t_90 = ts[k] - t0
                break
            k += 1
        last_end = k
        rm0 = rm[s]
        rmk = rm[min(k, n - 1)]
        eps.append((t0, dirn, inv[s], first_ioc, t_zero, t_90, (rmk - rm0) * dirn, rm[min(np.searchsorted(ts, t0 + 1_000_000), n - 1)] - rm0))
    E = np.array([(e[0], e[1], e[2], -1 if e[3] is None else e[3], -1 if e[4] is None else e[4], -1 if e[5] is None else e[5], e[6], e[7] * e[1]) for e in eps], dtype=float)
    print(f"episodes {len(E)}")
    for name, col in (('first IOC sent', 3), ('inventory crosses 0', 4), ('inventory reaches 90 the new way', 5)):
        v = E[:, col]
        vv = v[v >= 0] / 1000
        print(f"  {name:>34}: reached {len(vv) / len(E):.0%}, ms median {np.median(vv):.1f} p75 {np.percentile(vv, 75):.1f} p90 {np.percentile(vv, 90):.1f} mean {vv.mean():.1f}")
    print(f"  robust mid move in the signal direction: until reaching 90 {np.nanmean(E[:, 6]):+.2f} ticks; over 1 s {np.nanmean(E[:, 7]):+.2f}")
    # (c) IOC economics
    print('\n(c) IOC sends and fills')
    s_ts = d['s_ts']; s_sign = d['s_sign']; s_qty = d['s_qty']; s_px = d['s_px']
    f_ioc = d['f_ioc']; f_qty = d['f_qty']; f_px = d['f_px']; f_sign = d['f_sign']; f_ts = d['f_ts']
    filled = np.zeros(len(s_ts))
    np.add.at(filled, f_ioc[f_ioc >= 0], f_qty[f_ioc >= 0])
    idx = np.searchsorted(ts, s_ts, side='right') - 1
    inv_at = inv[idx]
    flip_side = inv_at * s_sign < 0
    for name, m in (('all', np.ones(len(s_ts), bool)), ('reducing (inv against)', flip_side), ('adding (inv 0/with)', ~flip_side)):
        print(f"  {name:>24}: sends {m.sum():,} req {s_qty[m].sum():,} filled {filled[m].sum():,.0f} ({filled[m].sum() / max(s_qty[m].sum(), 1):.0%});"
              f" zero-fill sends {(filled[m] == 0).mean():.0%}")
    # cost: fill price vs robust mid at the send, and markouts
    fm = f_ioc >= 0
    fi = np.searchsorted(ts, f_ts, side='right') - 1
    send_rm = rm[idx]
    for name, mask in (('taker', fm), ('maker', ~fm)):
        q = f_qty[mask]; sgn = f_sign[mask]; px = f_px[mask]; ii = fi[mask]
        edge0 = ((rm[ii] - px) * sgn * q).sum()
        out = [f"edge@fill(rm) {edge0 / q.sum():+.2f}/lot"]
        for h in (0.1, 1.0, 10.0):
            j = np.minimum(np.searchsorted(ts, f_ts[mask] + int(h * 1e6)), len(ts) - 1)
            out.append(f"mk{h:g}s {((rm[j] - px) * sgn * q).sum() / q.sum():+.2f}")
        print(f"  {name}: lots {q.sum():,}  " + '  '.join(out))
    ioc_cost = []
    for i in range(len(s_ts)):
        pass
    # time from the take condition to the IOC send and to the first fill, per take run
    print('\n(d) take condition runs (consecutive updates with the same take side)')
    tbs = np.where(take_b, 1, np.where(take_s, -1, 0))
    change = np.flatnonzero(np.diff(tbs, prepend=0) != 0)
    runs = []
    for a in change:
        if tbs[a] == 0:
            continue
        b = a
        while b + 1 < len(tbs) and tbs[b + 1] == tbs[a]:
            b += 1
        dur = ts[b] - ts[a] + (ts[b + 1] - ts[b] if b + 1 < len(ts) else 0)
        sent = d['tk'][a:b + 1].sum()
        runs.append((tbs[a], inv[a], dur, sent, d['o0'][a] if tbs[a] > 0 else d['o1'][a], inv[min(b + 1, len(inv) - 1)]))
    R = np.array(runs, dtype=float)
    wrong = R[:, 0] * R[:, 1] < 0
    print(f"  runs {len(R):,}; with inventory against {wrong.sum():,}: median duration {np.median(R[wrong, 2]) / 1000:.1f} ms,"
          f" no IOC sent in the run {(R[wrong, 3] == 0).mean():.0%}, own same-side OUT at start median {np.median(R[wrong, 4]):.0f}")
    dI = (R[wrong, 5] - R[wrong, 1]) * R[wrong, 0]
    print(f"  inventory change in the take direction during a wrong-way run: mean {dI.mean():.0f} lots, 0 for {(dI <= 0).mean():.0%}")
    with_ = (R[:, 0] * R[:, 1] > 0)
    print(f"  runs with inventory already with it {with_.sum():,}, flat {(R[:, 1] == 0).sum():,}")


def _ffill(x: np.ndarray) -> np.ndarray:
    m = np.isnan(x)
    idx = np.where(~m, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    return out


def main() -> None:
    strat, csv = Path(sys.argv[1]), Path(sys.argv[2])
    ov = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    if out is not None and out.exists():
        d = dict(np.load(out))
    else:
        d = record(strat, csv, ov)
        if out is not None:
            np.savez(out, **d)
    analyse(d)


if __name__ == '__main__':
    main()
