"""Diagnostic only, never shipped: the taker prediction of a pred_sig strategy replaced by an ORACLE (look-ahead).

    .venv/Scripts/python.exe -X utf8 -m research.diag.oracle_pred_r3 strategies/mm_r3_gru.py data/round3.csv [lead_ms] [blend] [qlead_ms]
lead_ms 0 leaves the taker alone; qlead_ms > 0 replaces the passive quote prediction (pred_raw) by the oracle at that lead.
pred_sig (the taker signal) becomes blend * (robust mid at t + lead_ms - mid now) + (1 - blend) * pred_sig; the passive
quote signal is untouched. Tells how much a better taker predictor could still be worth on this tape.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.simulator import ExchangeSimulator
from research.run import load_strategy_class, simulation_config


def main() -> None:
    strat, csv = Path(sys.argv[1]), Path(sys.argv[2])
    lead_us = int(float(sys.argv[3]) * 1000) if len(sys.argv) > 3 else 205_000
    blend = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    qlead_us = int(float(sys.argv[5]) * 1000) if len(sys.argv) > 5 else 0  # > 0: the passive quote signal gets an oracle too
    tlead = lead_us > 0
    z = np.load(Path('data/features') / (csv.stem + '.npz'))
    ts = z['ts_us']; b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    idx = np.where(ok)[0]; rm = (b + a) / 2
    cls = load_strategy_class(strat)

    class Oracle(cls):
        def _pred_update(self, now, bids, asks, bb, ba):
            super()._pred_update(now, bids, asks, bb, ba)
            ref2 = self.fair_prev_mid2
            mid = ref2 / 2.0 if ref2 > 0 else (bb + ba) / 2.0
            if tlead:
                k = np.searchsorted(ts, now + lead_us, side='right') - 1
                j = np.searchsorted(idx, max(k, 0), side='right') - 1
                self.pred_sig = blend * (rm[idx[max(j, 0)]] - mid) + (1.0 - blend) * self.pred_sig
            if qlead_us > 0:
                k = np.searchsorted(ts, now + qlead_us, side='right') - 1
                j = np.searchsorted(idx, max(k, 0), side='right') - 1
                self.pred_raw = rm[idx[max(j, 0)]] - mid

    sim = ExchangeSimulator(strategy=Oracle(), config=simulation_config())
    res = sim.run(csv)
    print(json.dumps({'strategy': strat.name, 'csv': csv.name, 'lead_ms': lead_us / 1000, 'blend': blend, 'qlead_ms': qlead_us / 1000, 'pnl': res.pnl,
                      'fills': res.fills, 'breach': res.breach_penalty}))


if __name__ == '__main__':
    main()
