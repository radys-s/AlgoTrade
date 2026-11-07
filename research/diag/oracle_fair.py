"""Diagnostic only, never shipped: run a fair-signal strategy with an ORACLE fair price (look-ahead).

    uv run python -X utf8 -m research.diag.oracle_fair strategies/mm_r2_fq2.py data/round2.csv [lead_ms] [blend]
fair_sig is replaced by (robust mid at t + lead_ms) - mid now (blend in [0,1] mixes it with the real signal:
sig = blend * oracle + (1 - blend) * real). Tells how much better the strategy would be with a better F estimate.
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
    lead_us = int(float(sys.argv[3]) * 1000) if len(sys.argv) > 3 else 150_000
    blend = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    clip = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
    z = np.load(Path('data/features') / (csv.stem + '.npz'))
    ts = z['ts_us']; b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
    ok = (b > 0) & (a > 0) & (a - b <= 3)
    idx = np.where(ok)[0]; rm = (b + a) / 2
    cls = load_strategy_class(strat)

    class Oracle(cls):
        def _fair_update(self, now, bids, asks, bb, ba):
            super()._fair_update(now, bids, asks, bb, ba)
            k = np.searchsorted(ts, now + lead_us, side='right') - 1
            j = np.searchsorted(idx, max(k, 0), side='right') - 1
            fut = rm[idx[max(j, 0)]]
            ref2 = self.fair_prev_mid2
            mid = ref2 / 2.0 if ref2 > 0 else (bb + ba) / 2.0
            o = max(-clip, min(clip, fut - mid))
            self.fair_sig = blend * o + (1.0 - blend) * self.fair_sig

    sim = ExchangeSimulator(strategy=Oracle(), config=simulation_config())
    res = sim.run(csv)
    print(json.dumps({'strategy': strat.name, 'lead_ms': lead_us / 1000, 'blend': blend, 'clip': clip, 'pnl': res.pnl,
                      'fills': res.fills, 'breach': res.breach_penalty}))


if __name__ == '__main__':
    main()
