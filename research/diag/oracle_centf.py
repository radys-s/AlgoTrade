"""Diagnostic only, never shipped: run a fair-signal strategy with a CENTERED fair estimate (small look-ahead).

    uv run python -X utf8 -m research.diag.oracle_centf strategies/mm_r2_fq2.py data/round2.csv [half_window_ms] [unclamped_only]
F_c(t) = mean implied fair (bid + 1.42 / ask - 1.42) of the limit inserts in [t - W, t + W]; fair_sig := F_c - mid.
Tells how much a near-perfect estimate of the hidden fair price NOW (not the future path) is worth.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.simulator import ExchangeSimulator
from research.run import load_strategy_class, simulation_config


def main() -> None:
    strat, csv = Path(sys.argv[1]), Path(sys.argv[2])
    w_us = int(float(sys.argv[3]) * 1000) if len(sys.argv) > 3 else 50_000
    uncl = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    df = pd.read_csv(csv)
    z = np.load(Path('data/features') / (csv.stem + '.npz'))
    ts = z['ts_us']; b = z['bid_px'].astype(float); a = z['ask_px'].astype(float)
    pb = np.r_[0, b[:-1]]; pa = np.r_[0, a[:-1]]
    G = (df.Operation.values == 'Insert') & (df.Lifespan.values == 'G')
    side = np.where(df.Side.values == 'B', 1, -1); px = df.Price.values.astype(float)
    two = (pb > 0) & (pa > 0)
    cl = np.where(side == 1, px >= pa - 1, px <= pb + 1)
    m = G & two & (~cl if uncl else True)
    t_i = ts[m]; f_i = np.where(side == 1, px + 1.42, px - 1.42)[m]
    cs = np.r_[0.0, np.cumsum(f_i)]
    cls = load_strategy_class(strat)

    class CentF(cls):
        def _fair_update(self, now, bids, asks, bb, ba):
            super()._fair_update(now, bids, asks, bb, ba)
            lo = np.searchsorted(t_i, now - w_us); hi = np.searchsorted(t_i, now + w_us, side='right')
            if hi > lo:
                fc = (cs[hi] - cs[lo]) / (hi - lo)
                ref2 = self.fair_prev_mid2
                mid = ref2 / 2.0 if ref2 > 0 else (bb + ba) / 2.0
                self.fair_sig = fc - mid

    sim = ExchangeSimulator(strategy=CentF(), config=simulation_config())
    res = sim.run(csv)
    print(json.dumps({'strategy': strat.name, 'half_window_ms': w_us / 1000, 'unclamped_only': uncl, 'pnl': res.pnl,
                      'fills': res.fills, 'breach': res.breach_penalty}))


if __name__ == '__main__':
    main()
