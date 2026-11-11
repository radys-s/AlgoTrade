"""Diagnostic: record the fair taker's entry decisions of a fq strategy (features at every update where it would cross).

    uv run python -X utf8 -m research.diag.tkfilt_record strategies/mm_r2_fq2.py data/round2.csv out.npz
At every update where take_buy / take_sell holds and the room is > 0 (an IOC is about to be sent) it logs a feature
vector; the IOC's actual fills are attached afterwards (fill_qty, fill_px). Labels come from the tape offline
(research/diag/tkfilt_fit.py). The strategy's behaviour is unchanged (pure observation).
"""
from __future__ import annotations
import sys
import time
from collections import deque
from math import exp, floor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.simulator import ExchangeSimulator
from simulator.models import OrderStatus
from research.run import load_strategy_class, simulation_config

COLS = ['t', 'side', 'touch', 'bb', 'ba', 'sig70', 'sig20', 'sig150', 'edge', 'spread', 'q_take', 'q_opp',
        'n_take', 'dt_ins', 'n_ins50', 'tr_sign', 'tr_qty', 'tr_dt', 'inv', 'drift', 'cond_age', 'room',
        'fill_qty', 'fill_px']


def make_recorder(cls):
    class Rec(cls):
        def __init__(self) -> None:
            super().__init__()
            self.rec: list[list[float]] = []
            self.rec_ids: dict[str, int] = {}
            self.r_s = [0.0, 0.0]
            self.r_w = [0.0, 0.0]
            self.r_tau = [20_000.0, 150_000.0]
            self.r_ins: deque = deque()
            self.r_tr = (0, 0, -(1 << 62))
            self.r_since = [-1, -1]
            self.r_drift = 0.0
            self.r_snap = None
            self.r_pending = -1

        def on_order_book_change(self, snapshot, trades) -> None:
            self.r_snap = snapshot
            for tr in trades:
                if tr.taker_owner == 'market':
                    self.r_tr = (1 if tr.book_side == 'ask' else -1, tr.quantity, tr.timestamp_us)
            super().on_order_book_change(snapshot, trades)

        def _drift(self, now, bb, ba):
            d = super()._drift(now, bb, ba)
            self.r_drift = d
            return d

        def _fair_update(self, now, bids, asks, bb, ba):
            s0, t0, n0 = self.fair_s, self.fair_t, self.fair_n[0] + self.fair_n[1]
            super()._fair_update(now, bids, asks, bb, ba)
            if self.fair_n[0] + self.fair_n[1] == n0:
                return
            dev = self.fair_s - s0 * exp(-(now - t0) / self.fair_tau_us)
            for j in (0, 1):
                a = exp(-(now - t0) / self.r_tau[j])
                self.r_s[j] = self.r_s[j] * a + dev
                self.r_w[j] = self.r_w[j] * a + 1.0
            ins = self.r_ins
            ins.append(now)
            while ins and ins[0] < now - 50_000:
                ins.popleft()

        def _fq(self, now, bb, ba, fshift, pull_bid, pull_ask):
            ref2 = self.fair_prev_mid2
            mid = ref2 / 2.0 if ref2 > 0 else (bb + ba) / 2.0
            f = mid + self.fq_scale * self.fair_sig + fshift
            tk = self.fq_take_k
            tb = floor(f - tk) >= ba
            ts_ = (not tb) and -floor(-(f + tk)) <= bb
            for k, c in ((0, tb), (1, ts_)):
                if c:
                    if self.r_since[k] < 0:
                        self.r_since[k] = now
                else:
                    self.r_since[k] = -1
            self.r_pending = -1
            if (tb or ts_) and self.fq_take > 0:
                k = 0 if tb else 1
                sd = 1 if tb else -1
                room = (self.max_pos - self.inv - self.out[0]) if tb else (self.max_pos + self.inv - self.out[1])
                if room > 0 and now >= self.fq_next_take[k]:
                    snap = self.r_snap
                    qb = nb = qa = na = 0
                    for v in snap.bids:
                        if v.owner != 'market':
                            continue
                        if v.price != bb:
                            break
                        qb += v.quantity
                        nb += 1
                    for v in snap.asks:
                        if v.owner != 'market':
                            continue
                        if v.price != ba:
                            break
                        qa += v.quantity
                        na += 1
                    ins = self.r_ins
                    while ins and ins[0] < now - 50_000:
                        ins.popleft()
                    sg, sq, st = self.r_tr
                    sig20 = self.r_s[0] / self.r_w[0] if self.r_w[0] > 0 else 0.0
                    sig150 = self.r_s[1] / self.r_w[1] if self.r_w[1] > 0 else 0.0
                    self.rec.append([
                        now, sd, ba if tb else bb, bb, ba, self.fair_sig, sig20, sig150,
                        (f - ba) if tb else (bb - f), ba - bb, qa if tb else qb, qb if tb else qa,
                        na if tb else nb, now - self.fair_t, len(ins), sg * sd, sq, now - st,
                        self.inv * sd, self.r_drift * sd, now - self.r_since[k], room, 0.0, 0.0])
                    self.r_pending = len(self.rec) - 1
            out = super()._fq(now, bb, ba, fshift, pull_bid, pull_ask)
            self.r_pending = -1
            return out

        def _send(self, k, sign, side, px, qty, ioc, j):
            oid = super()._send(k, sign, side, px, qty, ioc, j)
            if oid is not None and ioc and self.r_pending >= 0:
                self.rec_ids[oid] = self.r_pending
            return oid

        def on_private_order_update(self, event) -> None:
            i = self.rec_ids.get(event.order_id)
            if i is not None and event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED) and event.filled_quantity:
                r = self.rec[i]
                q = event.filled_quantity
                r[23] = (r[23] * r[22] + q * event.fill_price) / (r[22] + q)
                r[22] += q
            super().on_private_order_update(event)
    return Rec


def main() -> None:
    strat, csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    t0 = time.time()
    rec_cls = make_recorder(load_strategy_class(strat))
    s = rec_cls()
    res = ExchangeSimulator(strategy=s, config=simulation_config()).run(csv)
    arr = np.array(s.rec, dtype=float)
    np.savez(out, data=arr, cols=np.array(COLS))
    print(f'pnl={res.pnl:.0f} fills={res.fills} records={len(arr)} filled={int((arr[:, 22] > 0).sum())} '
          f'runtime={time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
