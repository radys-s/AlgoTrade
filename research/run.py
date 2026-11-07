"""Parallel backtest runner: one strategy x a parameter grid x datasets, scored per time chunk, written to the ledger.

    uv run python -m research.run --strategy strategies/mm.py --data data/round1.csv \
        --grid '{"quote_size": [1, 5], "skew_per_lot": [0, 0.1]}' --workers 6 --tag mm-size-skew

The strategy keeps every tunable in a class attribute `PARAMS` (a dict) and reads it only via
`self.PARAMS[...]`; a run overrides it by subclassing, so the file itself stays submittable.

With `--halving 0.1,0.3` a big grid gets cheaper: every set first replays the first 10 % of the
data, the best third the first 30 %, and only the best third of those the whole file (about a
third of the CPU of a full sweep). Prefix replays are exact, since the simulator starts from row
one anyway; their rows go to the ledger as `<dataset>@p10` and so on.

Each run's PnL curve (sampled every second) is cut in time: the last `--holdout` fraction is the
holdout, never looked at while tuning, and the rest is split into `--chunks` equal windows. A
change is worth keeping when it wins most chunks, not when it wins the total by one lucky minute.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from simulator.main import (
    MAX_POSITION,
    MAX_POTENTIAL_POSITION,
    ORDER_ACCEPT_DELAY_MS,
    POSITION_BREACH_PENALTY_PER_SECOND,
)
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

RESULTS_DIR = Path(__file__).parent / 'results'
SNAPSHOT_DIR = RESULTS_DIR / 'strategies'
LEDGER_COLUMNS = [
    'time', 'user', 'branch', 'commit', 'dirty', 'tag', 'strategy', 'strategy_class', 'strategy_sha',
    'dataset', 'overrides', 'params', 'pnl', 'train_pnl', 'hold_pnl', 'chunk_wins', 'chunks', 'chunk_min',
    'chunk_t', 'fills', 'traded_volume', 'breach_penalty', 'inventory', 'inv_q95', 'inv_q997', 'pnl_se',
    'runtime_s', 'events_per_s', 'chunk_pnls', 'error',
]  # fmt: skip

_class_cache: dict[str, type[BaseStrategy]] = {}


def load_strategy_class(file_path: str | Path) -> type[BaseStrategy]:
    """Import a strategy file and return its `BaseStrategy` subclass (alphabetically first, as the simulator does)."""
    path = Path(file_path).resolve()
    key = str(path)
    if key in _class_cache:
        return _class_cache[key]
    module_name = f'strategy_{hashlib.sha1(key.encode()).hexdigest()[:12]}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load module spec from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in dir(module):
        candidate = getattr(module, name)
        if isinstance(candidate, type) and issubclass(candidate, BaseStrategy) and candidate is not BaseStrategy:
            _class_cache[key] = candidate
            return candidate
    raise ValueError(f'No BaseStrategy subclass found in {path}')


def with_params(cls: type[BaseStrategy], overrides: dict[str, Any]) -> type[BaseStrategy]:
    """Return `cls` itself, or a subclass whose `PARAMS` has `overrides` applied."""
    if not overrides:
        return cls
    base = getattr(cls, 'PARAMS', None)
    if not isinstance(base, dict):
        raise ValueError(f'{cls.__name__} has no PARAMS dict, so it cannot take overrides')
    unknown = sorted(set(overrides) - set(base))
    if unknown:
        raise ValueError(f'Unknown parameter(s) for {cls.__name__}: {", ".join(unknown)}')
    return type(cls.__name__, (cls,), {'PARAMS': {**base, **overrides}})


def simulation_config(log_dir: Path | None = None) -> SimulationConfig:
    """The scoring configuration from `simulator/main.py`, with full logs only when `log_dir` is given."""
    return SimulationConfig(
        public_depth=20,
        run_log_path=log_dir / 'run.jsonl' if log_dir else None,
        book_delta_log_path=log_dir / 'book_deltas.csv' if log_dir else None,
        event_log_path=log_dir / 'events.csv' if log_dir else None,
        trade_log_path=log_dir / 'trades.csv' if log_dir else None,
        max_position=MAX_POSITION,
        max_potential_position=MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS,
    )


def chunk_metrics(curve: list[tuple[int, float]], chunks: int, holdout: float) -> dict[str, Any]:
    """Split a (timestamp_us, pnl) curve into a holdout tail and `chunks` equal training windows."""
    times = np.array([point[0] for point in curve], dtype=np.float64)
    pnls = np.array([point[1] for point in curve], dtype=np.float64)
    start, end = times[0], times[-1]
    hold_start = start + (1.0 - holdout) * (end - start)
    edges = np.linspace(start, hold_start, chunks + 1)
    at_edges = np.interp(edges, times, pnls)
    per_chunk = np.diff(at_edges)
    train_pnl = float(at_edges[-1] - at_edges[0])
    hold_pnl = float(pnls[-1] - at_edges[-1])
    spread = float(per_chunk.std(ddof=1)) if chunks > 1 else 0.0
    chunk_t = float(per_chunk.mean() / spread * math.sqrt(chunks)) if spread > 0 else 0.0
    return {
        'train_pnl': round(train_pnl, 2),
        'hold_pnl': round(hold_pnl, 2),
        'chunk_wins': int((per_chunk > 0).sum()),
        'chunks': chunks,
        'chunk_min': round(float(per_chunk.min()), 2),
        'chunk_t': round(chunk_t, 2),
        'chunk_pnls': [round(float(value), 2) for value in per_chunk],
    }


def write_full_logs(log_dir: Path, simulator: ExchangeSimulator, result: Any, data: Path, class_name: str) -> None:
    """Write metadata.json, result.json and pnl.csv next to the CSV logs, as `exchange-simulator` does."""
    metadata = {
        'trader': getpass.getuser(),
        'trader_class': class_name,
        'data_file': str(data.resolve()),
        'latency': ORDER_ACCEPT_DELAY_MS / 1000,
        'position_limit': MAX_POSITION,
        'max_potential_position': MAX_POTENTIAL_POSITION,
        'position_breach_penalty_per_second': POSITION_BREACH_PENALTY_PER_SECOND,
        'run_timestamp': datetime.now().isoformat(),
    }
    (log_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (log_dir / 'result.json').write_text(json.dumps(result.model_dump(mode='json'), indent=2))
    points = simulator.pnl_curve(result.pnl)
    with (log_dir / 'pnl.csv').open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['elapsed_s', 'pnl'])
        for timestamp_us, pnl in points:
            writer.writerow([(timestamp_us - points[0][0]) / 1_000_000, pnl])


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one (strategy, overrides, dataset) backtest; errors are returned, not raised, so a sweep survives them."""
    label = job.get('label') or Path(job['data']).stem
    row: dict[str, Any] = {'dataset': label, 'overrides': job['overrides'], 'index': job['index']}
    started = time.perf_counter()
    try:
        cls = with_params(load_strategy_class(job['strategy']), job['overrides'])
        log_dir = Path(job['log_dir']) if job['log_dir'] else None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
        strategy = cls()
        simulator = ExchangeSimulator(strategy=strategy, config=simulation_config(log_dir))
        result = simulator.run(Path(job['data']))
        runtime = time.perf_counter() - started
        if log_dir:
            write_full_logs(log_dir, simulator, result, Path(job['data']), cls.__name__)
        curve = simulator.pnl_curve(result.pnl)
        row.update(
            strategy_class=cls.__name__,
            params=dict(getattr(cls, 'PARAMS', {}) or {}),
            pnl=round(result.pnl, 2),
            fills=result.fills,
            traded_volume=result.traded_volume,
            breach_penalty=round(result.breach_penalty, 2),
            inventory=result.inventory,
            inv_q95=result.abs_inventory_q95,
            inv_q997=result.abs_inventory_q997,
            pnl_se=None if result.pnl_se is None else round(result.pnl_se, 2),
            runtime_s=round(runtime, 1),
            events_per_s=round(job['rows'] / runtime) if runtime > 0 else None,
            error='',
            **chunk_metrics(curve, job['chunks'], job['holdout']),
        )
    except Exception:  # noqa: BLE001 - one broken config must not kill the sweep
        row.update(error=traceback.format_exc(limit=6), runtime_s=round(time.perf_counter() - started, 1))
    return row


def git_info() -> dict[str, str]:
    """Branch, short commit and whether the tree has uncommitted changes, for the ledger."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(['git', *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''

    return {
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'commit': git('rev-parse', '--short', 'HEAD'),
        'dirty': 'yes' if git('status', '--porcelain', '--untracked-files=no') else 'no',
        'user': git('config', 'user.name') or getpass.getuser(),
    }


def snapshot_strategy(path: Path) -> str:
    """Copy the strategy file into the results folder under its content hash and return the hash."""
    content = path.read_bytes()
    sha = hashlib.sha256(content).hexdigest()[:10]
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    target = SNAPSHOT_DIR / f'{path.stem}_{sha}.py'
    if not target.exists():
        target.write_bytes(content)
    return sha


def count_rows(path: Path) -> int:
    """Data rows in a backtest CSV."""
    return ExchangeSimulator.count_backtest_rows(path)


def prefix_csv(path: Path, fraction: float, rows: int) -> tuple[Path, int]:
    """The first `fraction` of a backtest's rows as its own CSV under logs/prefix/, written once and reused.

    Replaying a prefix is exact: the simulator always starts from the first row, so a run on the
    prefix is identical to the start of the full run.
    """
    keep = max(1, int(rows * fraction))
    target = Path('logs') / 'prefix' / f'{path.stem}__p{round(fraction * 100)}.csv'
    if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return target, keep
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.tmp')
    with path.open('rb') as source, partial.open('wb') as sink:
        for line_number, line in enumerate(source):
            if line_number > keep:
                break
            sink.write(line)
    partial.replace(target)
    return target, keep


def expand_grid(grid: dict[str, list[Any]], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of `grid`, each combined with the `fixed` overrides."""
    keys = list(grid)
    combos = itertools.product(*(grid[key] if isinstance(grid[key], list) else [grid[key]] for key in keys))
    return [{**fixed, **dict(zip(keys, values, strict=True))} for values in combos] or [dict(fixed)]


def print_table(rows: list[dict[str, Any]], sort: str) -> None:
    """Print the finished runs, best first, and a per-parameter-set summary when there are several datasets."""
    import pandas as pd

    ok = [row for row in rows if not row.get('error')]
    for row in rows:
        if row.get('error'):
            print(f'\nERROR in run {row["index"]} {row["dataset"]} {json.dumps(row["overrides"])}:\n{row["error"]}')
    if not ok:
        return
    frame = pd.DataFrame(ok)
    frame['set'] = frame['overrides'].map(lambda value: json.dumps(value, sort_keys=True) if value else '(defaults)')
    columns = ['dataset', 'set', 'pnl', 'train_pnl', 'hold_pnl', 'chunk_wins', 'chunk_min', 'chunk_t', 'fills',
               'breach_penalty', 'inv_q997', 'runtime_s']  # fmt: skip
    with pd.option_context('display.width', 250, 'display.max_colwidth', 70, 'display.max_rows', 500):
        print()
        print(frame.sort_values(sort, ascending=False)[columns].to_string(index=False))
        if frame['dataset'].nunique() > 1:
            summary = frame.groupby('set').agg(
                datasets=('dataset', 'nunique'),
                positive=('pnl', lambda values: int((values > 0).sum())),
                pnl_sum=('pnl', 'sum'),
                pnl_min=('pnl', 'min'),
                train_sum=('train_pnl', 'sum'),
                hold_sum=('hold_pnl', 'sum'),
                breach=('breach_penalty', 'sum'),
            )
            print('\nAcross datasets (judge by positive and pnl_min, not only pnl_sum):')
            print(summary.sort_values(['positive', 'pnl_min'], ascending=False).to_string())


def write_ledger(rows: list[dict[str, Any]], meta: dict[str, str]) -> Path:
    """Append this sweep's runs to a new ledger CSV under research/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    safe_tag = ''.join(ch if ch.isalnum() or ch in '-_' else '-' for ch in meta['tag'])[:40] or 'run'
    path = RESULTS_DIR / f'{stamp}_{meta["user"]}_{safe_tag}.csv'
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=LEDGER_COLUMNS, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item['index']):
            record = {**meta, **row}
            record['overrides'] = json.dumps(row.get('overrides', {}), sort_keys=True)
            record['params'] = json.dumps(row.get('params', {}), sort_keys=True)
            record['chunk_pnls'] = json.dumps(row.get('chunk_pnls', []))
            record['error'] = (row.get('error') or '').strip().splitlines()[-1] if row.get('error') else ''
            writer.writerow(record)
    return path


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Parallel backtests over a parameter grid, written to the ledger')
    parser.add_argument('--strategy', type=Path, required=True, help='Strategy file (tunables in a PARAMS dict)')
    parser.add_argument('--data', type=Path, nargs='+', required=True, help='One or more backtest CSVs')
    parser.add_argument('--grid', default='{}', help='JSON dict of parameter -> list of values (cartesian product)')
    parser.add_argument('--set', default='{}', help='JSON dict of fixed parameter overrides for every run')
    parser.add_argument('--workers', type=int, default=4, help='Parallel processes (machine has 14 threads)')
    parser.add_argument('--chunks', type=int, default=8, help='Equal time windows in the training part')
    parser.add_argument('--holdout', type=float, default=0.25, help='Fraction of time at the end kept as holdout')
    parser.add_argument('--tag', default='', help='Short label for the ledger file name')
    parser.add_argument('--sort', default='train_pnl', help='Column to sort the table by')
    parser.add_argument(
        '--halving',
        default='',
        help='Successive halving: comma-separated data fractions, e.g. 0.1,0.3. Every set runs on the first 10%%, '
        'the best --keep of them on the first 30%%, the best of those on everything',
    )
    parser.add_argument('--keep', type=float, default=0.34, help='Fraction of sets kept at each halving rung')
    parser.add_argument('--min-keep', type=int, default=3, help='Never keep fewer sets than this at a rung')
    parser.add_argument('--full-logs', action='store_true', help='Write explorer logs to logs/<tag>/<run>/')
    parser.add_argument('--no-ledger', action='store_true', help='Do not write a ledger file')
    return parser.parse_args()


def main() -> None:
    """Expand the grid, run every (parameter set, dataset) in parallel, print and record the results."""
    args = parse_args()
    grid = json.loads(args.grid)
    fixed = json.loads(args.set)
    strategy_path: Path = args.strategy
    cls = load_strategy_class(strategy_path)
    param_sets = expand_grid(grid, fixed)
    for overrides in param_sets:
        with_params(cls, overrides)  # fail fast on typos, before any worker starts
    rows_by_data = {str(path): count_rows(path) for path in args.data}

    tag = args.tag or strategy_path.stem
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    survivors = param_sets
    rungs = sorted(float(value) for value in args.halving.split(',') if value.strip())
    for fraction in rungs:
        if len(survivors) <= args.min_keep:
            break
        prefixes = {str(path): prefix_csv(path, fraction, rows_by_data[str(path)]) for path in args.data}
        label = f'@p{round(fraction * 100)}'
        jobs = [
            {
                'index': len(all_rows) + index,
                'strategy': str(strategy_path),
                'overrides': overrides,
                'data': str(prefixes[str(data)][0]),
                'label': data.stem + label,
                'rows': prefixes[str(data)][1],
                'chunks': args.chunks,
                'holdout': 0.0,
                'log_dir': None,
            }
            for index, (overrides, data) in enumerate(itertools.product(survivors, args.data))
        ]
        print(f'rung {label}: {len(survivors)} sets on the first {fraction:.0%} of the data')
        rung_rows = run_jobs(jobs, args.workers)
        all_rows.extend(rung_rows)
        survivors = best_sets(survivors, rung_rows, args.keep, args.min_keep)

    jobs = []
    for overrides, data in itertools.product(survivors, args.data):
        index = len(all_rows) + len(jobs)
        log_dir = Path('logs') / tag / f'{index:03d}_{data.stem}' if args.full_logs else None
        jobs.append(
            {
                'index': index,
                'strategy': str(strategy_path),
                'overrides': overrides,
                'data': str(data),
                'rows': rows_by_data[str(data)],
                'chunks': args.chunks,
                'holdout': args.holdout,
                'log_dir': str(log_dir) if log_dir else None,
            }
        )
    print(f'full: {len(jobs)} runs ({len(survivors)} parameter sets x {len(args.data)} datasets)')
    rows = run_jobs(jobs, args.workers)
    all_rows.extend(rows)
    print(f'done in {time.perf_counter() - started:.0f}s')

    print_table(rows, args.sort)
    if not args.no_ledger:
        meta = {**git_info(), 'time': datetime.now().isoformat(timespec='seconds'), 'tag': tag}
        meta.update(strategy=strategy_path.as_posix(), strategy_sha=snapshot_strategy(strategy_path))
        print(f'\nledger: {write_ledger(all_rows, meta).as_posix()}')


def run_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Run jobs on a process pool, printing one line per finished run."""
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            status = 'ERROR' if row.get('error') else f'pnl={row["pnl"]:>12,.0f} hold={row["hold_pnl"]:>10,.0f}'
            print(f'[{done}/{len(jobs)}] {row["dataset"]} {json.dumps(row["overrides"])} {status} ({row["runtime_s"]}s)')
    return rows


def best_sets(
    param_sets: list[dict[str, Any]], rows: list[dict[str, Any]], keep: float, min_keep: int
) -> list[dict[str, Any]]:
    """The parameter sets with the highest PnL summed over datasets at this rung (errors rank last)."""
    totals = {json.dumps(overrides, sort_keys=True): 0.0 for overrides in param_sets}
    for row in rows:
        key = json.dumps(row['overrides'], sort_keys=True)
        totals[key] += -math.inf if row.get('error') else row['pnl']
    count = min(len(param_sets), max(min_keep, math.ceil(len(param_sets) * keep)))
    ranked = sorted(param_sets, key=lambda overrides: totals[json.dumps(overrides, sort_keys=True)], reverse=True)
    for overrides in ranked[count:]:
        print(f'  dropped {json.dumps(overrides)} (rung pnl {totals[json.dumps(overrides, sort_keys=True)]:,.0f})')
    return ranked[:count]


if __name__ == '__main__':
    main()
