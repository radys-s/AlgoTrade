"""Champion vs challenger, paired by time chunk on every dataset, with the team's acceptance verdict.

    uv run python -m research.compare --a submission/best_strategy.py --b strategies/exp_skew.py \
        --data data/round1.csv data/playground.csv [--b-set '{"skew_per_lot": 0.1}'] [--workers 6]

The first dataset is the current round (primary). The challenger is ACCEPTED only if all hold:
  1. no breach penalty anywhere;
  2. on the primary dataset it beats the champion in at least `--min-wins` of the paired training chunks;
  3. on the primary dataset its holdout PnL (the untouched last 25 %) is higher;
  4. on every other dataset where the champion is positive, the challenger is positive too.
Anything else is REJECTED with the failed rules listed; a human may still override with a written reason.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from research.run import count_rows, load_strategy_class, run_job, with_params


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Paired champion/challenger comparison with a verdict')
    parser.add_argument('--a', type=Path, required=True, help='Champion strategy file')
    parser.add_argument('--b', type=Path, required=True, help='Challenger strategy file')
    parser.add_argument('--a-set', default='{}', help='JSON overrides for the champion')
    parser.add_argument('--b-set', default='{}', help='JSON overrides for the challenger')
    parser.add_argument('--data', type=Path, nargs='+', required=True, help='Primary (current round) first')
    parser.add_argument('--chunks', type=int, default=8)
    parser.add_argument('--holdout', type=float, default=0.25)
    parser.add_argument('--min-wins', type=int, default=6, help='Paired training chunks the challenger must win')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run both strategies on every dataset and print the paired comparison and verdict."""
    args = parse_args()
    sides = {'A': (args.a, json.loads(args.a_set)), 'B': (args.b, json.loads(args.b_set))}
    for path, overrides in sides.values():
        with_params(load_strategy_class(path), overrides)
    rows = {str(path): count_rows(path) for path in args.data}
    jobs = [
        {
            'index': index,
            'side': side,
            'strategy': str(path),
            'overrides': overrides,
            'data': str(data),
            'rows': rows[str(data)],
            'chunks': args.chunks,
            'holdout': args.holdout,
            'log_dir': None,
        }
        for index, (side, (path, overrides), data) in enumerate(
            (side, spec, data) for side, spec in sides.items() for data in args.data
        )
    ]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        results = list(pool.map(run_job, jobs))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for job, result in zip(jobs, results, strict=True):
        if result.get('error'):
            print(f'ERROR {job["side"]} on {Path(job["data"]).stem}:\n{result["error"]}')
            raise SystemExit(2)
        by_key[(job['side'], Path(job['data']).stem)] = result

    failures: list[str] = []
    print(f'A = {args.a.as_posix()} {args.a_set}\nB = {args.b.as_posix()} {args.b_set}\n')
    for position, data in enumerate(args.data):
        name = data.stem
        a, b = by_key[('A', name)], by_key[('B', name)]
        diffs = [round(cb - ca, 1) for ca, cb in zip(a['chunk_pnls'], b['chunk_pnls'], strict=True)]
        wins = sum(diff > 0 for diff in diffs)
        role = 'primary' if position == 0 else 'other'
        print(f'{name} ({role})')
        print(f'  pnl    A={a["pnl"]:>12,.0f}  B={b["pnl"]:>12,.0f}  diff={b["pnl"] - a["pnl"]:>+12,.0f}')
        print(f'  hold   A={a["hold_pnl"]:>12,.0f}  B={b["hold_pnl"]:>12,.0f}  diff={b["hold_pnl"] - a["hold_pnl"]:>+12,.0f}')
        print(f'  chunks B-A {diffs}  -> B wins {wins}/{len(diffs)}')
        print(f'  fills  A={a["fills"]}  B={b["fills"]}   inv_q997 A={a["inv_q997"]} B={b["inv_q997"]}')
        if b['breach_penalty'] > 0:
            failures.append(f'{name}: challenger breach penalty {b["breach_penalty"]:,.0f}')
        if position == 0:
            if wins < args.min_wins:
                failures.append(f'{name}: wins only {wins}/{len(diffs)} paired chunks (< {args.min_wins})')
            if b['hold_pnl'] <= a['hold_pnl']:
                failures.append(f'{name}: holdout not better ({b["hold_pnl"]:,.0f} <= {a["hold_pnl"]:,.0f})')
        elif a['pnl'] > 0 and b['pnl'] <= 0:
            failures.append(f'{name}: champion positive but challenger not ({b["pnl"]:,.0f})')
    print()
    if failures:
        print('VERDICT: REJECT')
        for failure in failures:
            print(f'  - {failure}')
    else:
        print('VERDICT: ACCEPT (still run research.check before submitting)')


if __name__ == '__main__':
    main()
