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
    leads = [int(float(x) * 1000) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [205_000]  # several = their mean
    lead_us = leads[0]
    blend = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    qleads = [int(float(x) * 1000) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else [0]  # > 0: quote oracle (mean)
    qlead_us = qleads[0]
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
                o = 0.0
                for L in leads:
                    k = np.searchsorted(ts, now + L, side='right') - 1
                    j = np.searchsorted(idx, max(k, 0), side='right') - 1
                    o += rm[idx[max(j, 0)]] - mid
                self.pred_sig = blend * o / len(leads) + (1.0 - blend) * self.pred_sig
            if qlead_us > 0:
                o = 0.0
                for L in qleads:
                    k = np.searchsorted(ts, now + L, side='right') - 1
                    j = np.searchsorted(idx, max(k, 0), side='right') - 1
                    o += rm[idx[max(j, 0)]] - mid
                self.pred_raw = o / len(qleads)

    sim = ExchangeSimulator(strategy=Oracle(), config=simulation_config())
    res = sim.run(csv)
    print(json.dumps({'strategy': strat.name, 'csv': csv.name, 'lead_ms': [L / 1000 for L in leads], 'blend': blend, 'qlead_ms': [L / 1000 for L in qleads], 'pnl': res.pnl,
                      'fills': res.fills, 'breach': res.breach_penalty}))


if __name__ == '__main__':
    main()
