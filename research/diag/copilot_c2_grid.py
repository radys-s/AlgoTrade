from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from research.run import load_strategy_class, with_params, chunk_metrics
from simulator import main as sim_main
from simulator.simulator import ExchangeSimulator, SimulationConfig

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / 'research' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sim_config():
    return SimulationConfig(
        public_depth=20,
        max_position=sim_main.MAX_POSITION,
        max_potential_position=sim_main.MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=sim_main.POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=sim_main.ORDER_ACCEPT_DELAY_MS,
    )


def run_single(strategy_path: str | Path, data_path: str | Path, overrides: dict):
    strategy_cls = with_params(load_strategy_class(strategy_path), overrides)
    strategy = strategy_cls()
    sim = ExchangeSimulator(strategy=strategy, config=sim_config())
    result = sim.run(Path(data_path))
    curve = sim.pnl_curve(result.pnl)
    metrics = chunk_metrics(curve, 8, 0.25)
    return {
        'overrides': overrides,
        'pnl': round(float(result.pnl), 2),
        'train_pnl': metrics['train_pnl'],
        'hold_pnl': metrics['hold_pnl'],
        'chunk_wins': metrics['chunk_wins'],
        'chunks': metrics['chunks'],
        'chunk_min': metrics['chunk_min'],
        'chunk_t': metrics['chunk_t'],
        'fills': int(result.fills),
        'traded_volume': int(result.traded_volume),
        'breach_penalty': round(float(result.breach_penalty), 2),
        'inv_q95': round(float(result.abs_inventory_q95), 2),
        'inv_q997': round(float(result.abs_inventory_q997), 2),
        'pnl_se': round(float(result.pnl_se), 2) if result.pnl_se is not None else None,
        'runtime_s': None,
        'error': '',
        'chunk_pnls': metrics['chunk_pnls'],
    }


def grid_rows(title: str, grid: dict, fixed: dict | None = None):
    fixed = fixed or {}
    keys = list(grid)
    combos = list(product(*(grid[k] for k in keys)))
    rows = []
    for values in combos:
        overrides = {**fixed, **dict(zip(keys, values, strict=True))}
        rows.append((overrides, title))
    return rows


def write_csv(path: Path, rows: list[dict]):
    fieldnames = ['label', 'overrides', 'pnl', 'train_pnl', 'hold_pnl', 'chunk_wins', 'chunk_min', 'chunk_t', 'fills', 'traded_volume', 'breach_penalty', 'inv_q95', 'inv_q997', 'r1_pnl', 'error']
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def eval_r1_pnl(overrides: dict):
    strategy_path = ROOT / 'strategies' / 'mm_r2_fq2.py'
    strategy_cls = with_params(load_strategy_class(strategy_path), overrides)
    strategy = strategy_cls()
    sim = ExchangeSimulator(strategy=strategy, config=sim_config())
    result = sim.run(ROOT / 'data' / 'round1.csv')
    return round(float(result.pnl), 2)


def run_grid(label: str, grid: dict, fixed: dict | None = None, refine: list[tuple] | None = None):
    all_rows = []
    configs = grid_rows(label, grid, fixed)
    if refine:
        configs = [((dict(overrides)), label) for overrides in refine]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = []
        for overrides, _ in configs:
            futures.append(pool.submit(run_single, ROOT / 'strategies' / 'mm_r2_fq2.py', ROOT / 'data' / 'round2.csv', overrides))
        for future in as_completed(futures):
            result = future.result()
            all_rows.append({
                'label': label,
                'overrides': json.dumps(result['overrides'], sort_keys=True),
                'pnl': result['pnl'],
                'train_pnl': result['train_pnl'],
                'hold_pnl': result['hold_pnl'],
                'chunk_wins': result['chunk_wins'],
                'chunk_min': result['chunk_min'],
                'chunk_t': result['chunk_t'],
                'fills': result['fills'],
                'traded_volume': result['traded_volume'],
                'breach_penalty': result['breach_penalty'],
                'inv_q95': result['inv_q95'],
                'inv_q997': result['inv_q997'],
                'r1_pnl': eval_r1_pnl(result['overrides']),
                'error': result['error'],
            })
    all_rows.sort(key=lambda r: r['pnl'], reverse=True)
    path = RESULTS_DIR / f'{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}_{label}.csv'
    write_csv(path, all_rows)
    print(f'GRID {label}: best={all_rows[0]["pnl"]} rows={len(all_rows)} path={path}')
    for row in all_rows[:5]:
        print(row)
    return all_rows


def refine_neighborhood(best_overrides: dict, params: dict[str, list], scale: float = 0.15):
    refined = []
    for key, values in params.items():
        base = best_overrides.get(key, values[0])
        if isinstance(base, (int, float)):
            subvals = []
            for val in values:
                if val == 0:
                    subvals.append(0)
                else:
                    subvals.extend([base * (1 - scale), base, base * (1 + scale)])
            refined.append(subvals)
    return [
        {**best_overrides, **dict(zip(params.keys(), values, strict=True))}
        for values in product(*[sorted(set(vals)) for vals in refined])
    ]


def main():
    grid_a = {
        'fq_take_k': [0.05, 0.1, 0.2, 0.3],
        'fair_tau_ms': [50, 70, 100],
        'fq_scale': [1.0, 1.15, 1.3],
    }
    run_grid('gridA', grid_a)

    grid_b = {
        'gate_thr': [4, 6, 8, 10, 0],
        'fq_k': [0.05, 0.3],
        'fair_levels': [10, 20, 30],
    }
    run_grid('gridB', grid_b)

    grid_c = {
        'fq_take_cooldown_ms': [0, 5, 20, 50],
        'fq_slip': [0, 1, 2],
    }
    run_grid('gridC', grid_c)


if __name__ == '__main__':
    main()
