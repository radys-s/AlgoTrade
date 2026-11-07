"""Grid parametrů strategie paralelně přes procesy; výsledky do research/results/<name>.csv.

Použití:
    uv run python research/grid.py strategies/momentum_hold.py data/sample.csv QTY=1,10,20,50 HOLD_US=1000000,3000000

Konfigurace (limity, pokuta, latence) je převzatá ze simulator.main, tj. stejná jako u hodnocení.
Segmenty: PnL se navíc rozpadne na čtvrtiny času podle pnl křivky (robustnost, ne jen součet).
"""

from __future__ import annotations

import csv
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

from simulator import main as sim_main
from simulator.simulator import ExchangeSimulator, SimulationConfig


def run_one(args: tuple[str, str, dict]) -> dict:
    strat_path, data_path, params = args
    strategy = sim_main.load_strategy_from_file(strat_path)
    for k, v in params.items():
        setattr(strategy, k, v)
    config = SimulationConfig(
        max_position=sim_main.MAX_POSITION,
        max_potential_position=sim_main.MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=sim_main.POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=sim_main.ORDER_ACCEPT_DELAY_MS,
    )
    sim = ExchangeSimulator(strategy=strategy, config=config)
    res = sim.run(Path(data_path))
    curve = [p for _, p in sim.pnl_curve(res.pnl)]
    n = len(curve)
    cuts = [curve[min(n - 1, n * i // 4)] if i else 0.0 for i in range(5)]
    cuts[4] = curve[-1]
    quarters = [round(cuts[i + 1] - cuts[i]) for i in range(4)]
    return {**params, 'pnl': res.pnl, 'pnl_se': round(res.pnl_se or 0), 'fills': res.fills,
            'penalty': res.breach_penalty, 'q997_absI': res.abs_inventory_q997, 'quarters': quarters}


def main() -> None:
    strat, data, *specs = sys.argv[1:]
    grid = {}
    for s in specs:
        k, vals = s.split('=')
        grid[k] = [int(v) for v in vals.split(',')]
    combos = [dict(zip(grid, vs)) for vs in itertools.product(*grid.values())] or [{}]
    with Pool() as pool:
        rows = pool.map(run_one, [(strat, data, c) for c in combos])
    rows.sort(key=lambda r: -r['pnl'])
    out = Path('research/results') / f'{Path(strat).stem}__{Path(data).stem}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows[:12]:
        print(r)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
