"""Diagnostic: how old must an order be to stand at the front when its level becomes the touch?

    uv run python -X utf8 -m research.diag.queue_age data/round2.csv

Replays the tape with a non-trading recorder. Whenever the best bid drops to a lower price (or the best
ask rises) the new touch is a level that already existed: an order of ours parked there earlier would
now be the touch. For that level we list the resting market orders with their age (time since their
tape insert) and print, for A = 0.1 .. 3 s, the lots of orders OLDER than A (= lots ahead of an order
we had parked there A seconds earlier) and how often that is zero.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.simulator import ExchangeSimulator
from simulator.strategy import BaseStrategy
from research.run import simulation_config

AGES = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]


class Rec(BaseStrategy):
    def __init__(self, ins_ts: dict) -> None:
        super().__init__()
        self.ins = ins_ts
        self.pb = None
        self.pa = None
        self.rows = []  # (side, total, [older-than-A lots...], spread_after)

    def on_order_book_change(self, snapshot, trades) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            self.pb = self.pa = None
            return
        now = snapshot.timestamp_us
        bb, ba = bids[0].price, asks[0].price
        for side, cur, prev, lst, worse in ((1, bb, self.pb, bids, lambda c, p: c < p), (-1, ba, self.pa, asks, lambda c, p: c > p)):
            if prev is not None and worse(cur, prev):
                tot = 0
                older = [0] * len(AGES)
                for v in lst:
                    if v.price != cur:
                        break
                    tot += v.quantity
                    age = (now - self.ins.get(v.order_id, now)) / 1e6
                    for j, a in enumerate(AGES):
                        if age > a:
                            older[j] += v.quantity
                self.rows.append((side, cur - prev, tot, *older, ba - bb))
        self.pb, self.pa = bb, ba


def main() -> None:
    path = Path(sys.argv[1])
    tape = pd.read_csv(path, usecols=['Time', 'Operation', 'OrderId'])
    ins = tape[tape.Operation == 'Insert']
    ins_ts = dict(zip(ins.OrderId.astype(str), (ins.Time * 1e6).round().astype(np.int64)))
    rec = Rec(ins_ts)
    sim = ExchangeSimulator(strategy=rec, config=simulation_config())
    sim.run(path)
    r = np.array(rec.rows, dtype=float)
    print(f'{path}: touch retreats onto an existing level: {len(r):,}')
    jump = np.abs(r[:, 1])
    for lab, m in [('all', np.ones(len(r), bool)), ('1 tick', jump == 1), ('2+ ticks', jump >= 2)]:
        rr = r[m]
        print(f'  [{lab}] n={len(rr):,} new-touch lots: median {np.median(rr[:, 2]):.0f} mean {rr[:, 2].mean():.0f}')
        for j, a in enumerate(AGES):
            x = rr[:, 3 + j]
            print(f'    parked {a:4.2f}s before: lots ahead median {np.median(x):6.0f} mean {x.mean():7.1f}  P(0 ahead) {np.mean(x == 0):.3f}  P(<=20) {np.mean(x <= 20):.3f}')


if __name__ == '__main__':
    main()
