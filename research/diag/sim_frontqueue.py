"""Diagnostic only, never shipped: backtest with user orders inserted at the FRONT of their level's queue.

    uv run python -X utf8 -m research.diag.sim_frontqueue --strategy strategies/mm_r1_final2.py \
        --data data/round1.csv --mode off|front|fresh [--fresh-lots 100] [--over '{"ml_thr": 9}']
Monkey-patches OrderBook.insert in this process only; simulator/ is untouched.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import deque
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import simulator.simulator as S
from research.run import load_strategy_class, with_params, simulation_config, chunk_metrics

MODE = {'m': 'off', 'fresh': 100}
STATS = {'front': 0, 'back': 0}
_orig_insert = S.OrderBook.insert


def insert(self, order):
    mode = MODE['m']
    if mode == 'off' or order.owner != 'user':
        return _orig_insert(self, order)
    if order.order_id in self._orders:
        raise ValueError(f'Order {order.order_id} already exists')
    levels = self._bids if order.side == S._BUY else self._asks
    level = levels.get(order.price)
    if level is not None and mode == 'fresh':
        mq = sum(o.remaining_quantity for o in level.queue if o.owner != 'user')
        if mq >= MODE['fresh']:
            STATS['back'] += 1
            return _orig_insert(self, order)
    if level is None:
        STATS['front'] += 1
        return _orig_insert(self, order)
    # front of market orders, behind our own earlier orders (keep our FIFO)
    k = 0
    kv = 0
    for o in level.queue:
        if o.owner != 'user':
            break
        k += 1
        if o.remaining_quantity > 0:
            kv += 1
    self._orders[order.order_id] = order
    level.queue.insert(k, order)
    if level.views is not None and order.remaining_quantity > 0:
        level.views.insert(kv, S._view_of(order))
    self._user_quantity[order.side] += order.remaining_quantity
    self._bump_version(order.side)
    STATS['front'] += 1


S.OrderBook.insert = insert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--data', default='data/round1.csv')
    ap.add_argument('--mode', default='off')
    ap.add_argument('--fresh-lots', type=int, default=100)
    ap.add_argument('--over', default='{}')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    MODE['m'] = a.mode
    MODE['fresh'] = a.fresh_lots
    cls = with_params(load_strategy_class(a.strategy), json.loads(a.over))
    t = time.perf_counter()
    sim = S.ExchangeSimulator(strategy=cls(), config=simulation_config(None))
    res = sim.run(Path(a.data))
    curve = sim.pnl_curve(res.pnl)
    cm = chunk_metrics(curve, 6, 0.2)
    out = dict(strategy=Path(a.strategy).stem, mode=a.mode, over=a.over, pnl=round(res.pnl, 1), fills=res.fills,
               lots=res.traded_volume, breach=round(res.breach_penalty, 1), inv=res.inventory,
               q95=res.abs_inventory_q95, q997=res.abs_inventory_q997, train=cm['train_pnl'], hold=cm['hold_pnl'],
               chunks=cm['chunk_pnls'], placed=dict(STATS), secs=round(time.perf_counter() - t))
    out['pnl_per_lot'] = round(res.pnl / max(1, res.traded_volume), 3)
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out))


if __name__ == '__main__':
    main()
