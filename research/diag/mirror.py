"""Mirror a tape: price' = C - price, side B <-> A, cancels unchanged (a symmetry check, not independent data).

    uv run python -X utf8 -m research.diag.mirror data/round3.csv [data/round3_mirror.csv] [--c 20000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst', nargs='?')
    ap.add_argument('--c', type=int, default=20000)
    a = ap.parse_args()
    src = Path(a.src)
    dst = Path(a.dst) if a.dst else src.with_name(src.stem + '_mirror.csv')
    df = pd.read_csv(src, dtype={'Side': 'string', 'Lifespan': 'string'})
    ins = df['Operation'] == 'Insert'
    df.loc[ins, 'Price'] = a.c - df.loc[ins, 'Price']
    df.loc[ins, 'Side'] = df.loc[ins, 'Side'].map({'B': 'A', 'A': 'B'})
    if pd.api.types.is_float_dtype(df['Price']):
        df['Price'] = df['Price'].astype('Int64')
    for c in ('Volume',):
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype('Int64')
    df.to_csv(dst, index=False, float_format='%.6f')
    print(f'{src} -> {dst}: {len(df):,} rows, {int(ins.sum()):,} inserts mirrored around {a.c}')


if __name__ == '__main__':
    main()
