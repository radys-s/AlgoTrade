"""Read every ledger CSV under research/results/ (both teammates' runs) and show the best ones.

    uv run python -m research.ledger                          # top runs per dataset
    uv run python -m research.ledger --data round1 --strategy mm --top 30
    uv run python -m research.ledger --robust --data round1 playground   # parameter sets ranked across datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent / 'results'


def load_ledger() -> pd.DataFrame:
    """All ledger rows, newest last; empty frame when nothing has been run yet."""
    files = sorted(RESULTS_DIR.glob('*.csv'))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(file, dtype={'commit': str, 'strategy_sha': str}) for file in files])
    return frame.sort_values('time').reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Best runs from the research ledger')
    parser.add_argument('--data', nargs='*', default=[], help='Dataset names (CSV stems) to keep')
    parser.add_argument('--strategy', default='', help='Substring of the strategy path to keep')
    parser.add_argument('--tag', default='', help='Substring of the tag to keep')
    parser.add_argument('--user', default='', help='Only this git user')
    parser.add_argument('--top', type=int, default=15, help='Rows to show per dataset')
    parser.add_argument('--sort', default='train_pnl', help='Column to rank by')
    parser.add_argument('--robust', action='store_true', help='Rank parameter sets across all selected datasets')
    return parser.parse_args()


def main() -> None:
    """Filter the ledger and print the leaders."""
    args = parse_args()
    frame = load_ledger()
    if frame.empty:
        print('Ledger is empty: run research.run first.')
        return
    frame = frame[frame['error'].isna() | (frame['error'] == '')]
    if args.data:
        frame = frame[frame['dataset'].isin(args.data)]
    if args.strategy:
        frame = frame[frame['strategy'].str.contains(args.strategy, regex=False)]
    if args.tag:
        frame = frame[frame['tag'].str.contains(args.tag, regex=False)]
    if args.user:
        frame = frame[frame['user'] == args.user]
    if frame.empty:
        print('No runs match.')
        return
    frame = frame.assign(run=frame['strategy'].map(lambda path: Path(path).stem) + '@' + frame['strategy_sha'])
    columns = ['time', 'user', 'run', 'overrides', 'pnl', 'train_pnl', 'hold_pnl', 'chunk_wins', 'chunk_t',
               'fills', 'breach_penalty', 'inv_q997', 'runtime_s']  # fmt: skip
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60, 'display.max_rows', 500):
        if args.robust:
            grouped = frame.groupby(['run', 'overrides']).agg(
                datasets=('dataset', 'nunique'),
                positive=('pnl', lambda values: int((values > 0).sum())),
                pnl_min=('pnl', 'min'),
                pnl_sum=('pnl', 'sum'),
                hold_sum=('hold_pnl', 'sum'),
                breach=('breach_penalty', 'sum'),
            )
            ranked = grouped.sort_values(['datasets', 'positive', 'pnl_min'], ascending=False)
            print(ranked.head(args.top).to_string())
            return
        for dataset, group in frame.groupby('dataset'):
            print(f'\n=== {dataset}: {len(group)} runs')
            print(group.sort_values(args.sort, ascending=False)[columns].head(args.top).to_string(index=False))


if __name__ == '__main__':
    main()
