"""Re-fit the champion's ridge on a new round's tape and bake it into a copy of the strategy.

Usage:
    uv run python -X utf8 -m research.refit_ml data/round2.csv --template strategies/mm_ml.py \
        --out strategies/mm_ml_round2.py [--horizon-ms 100] [--train-frac 1.0] [--thr-quantile 0.81]

Same fit as `research.signals --export-linear` (build_dataset + fit_ridge, alpha 10, 6 significant
digits). Only the LINEAR_MODEL dict literal and the `'ml_thr'` value in PARAMS change in the copy.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from research.features import ensure_features, load_features
from research.signals import build_dataset, export_linear, fit_ridge, pearson

ALPHA = 10.0
MAX_ROWS = 3_000_000  # same default as research.signals, so round1 reproduces the champion's stride


def dict_span(text: str, marker: str = 'LINEAR_MODEL') -> tuple[int, int]:
    """Return [start, end) of the dict literal assigned to `marker` (brace matching, strings skipped)."""
    m = re.search(rf'\b{marker}\b[^=\n]*=\s*\{{', text)
    if m is None:
        raise SystemExit(f'template has no `{marker} = {{...}}` literal')
    start = m.end() - 1
    depth, i, quote = 0, start, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == chr(92):
                i += 1
            elif c == quote:
                quote = None
        elif c in '\'"':
            quote = c
        elif c == '#':
            i = text.index('\n', i)
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    raise SystemExit('unbalanced braces in the LINEAR_MODEL literal')


def format_dict(model: dict, indent: str = '    ') -> str:
    """Render the model dict in the template's layout (first key on the brace line, nested dicts hanging)."""
    lines = []
    for key, value in model.items():
        if isinstance(value, dict):
            items = [f'{k!r}: {v!r}' for k, v in value.items()]
            pad = indent + ' ' * (len(repr(key)) + 4)
            lines.append(f'{indent} {key!r}: {{' + f',\n{pad}'.join(items) + '}')
        else:
            lines.append(f'{indent} {key!r}: {value!r}')
    return '{' + ',\n'.join(lines).lstrip(' ') + '}'


def main(argv: list[str] | None = None) -> None:
    """Fit, substitute, report."""
    p = argparse.ArgumentParser(description='Refit the ridge on a tape and bake it into a strategy copy')
    p.add_argument('data', type=Path)
    p.add_argument('--template', type=Path, default=Path('strategies/mm_ml.py'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--horizon-ms', type=int, default=100)
    p.add_argument('--train-frac', type=float, default=1.0)
    p.add_argument('--thr-quantile', type=float, default=0.81)
    p.add_argument('--max-rows', type=int, default=MAX_ROWS)
    args = p.parse_args(argv)

    h = args.horizon_ms
    npz = ensure_features(args.data)  # builds data/features/<round>.npz when missing or stale
    d = load_features(npz)
    data = build_dataset(d, (h,), max_rows=args.max_rows)
    names = [str(n) for n in data['feature_names']]
    X, y, ts = data['X'], data['Y'][:, 0].astype(np.float64), data['ts_us']

    if args.train_frac >= 1.0:
        train = np.ones(len(y), bool)
        test = np.zeros(len(y), bool)
    else:
        t0, t1 = d['ts_us'][0], d['ts_us'][-1]
        cut = t0 + args.train_frac * (t1 - t0)
        train = ts <= cut - h * 1000 - 10000  # purge: labels must not reach the holdout
        test = ts > cut
    model = fit_ridge(X[train], y[train], names, ALPHA)
    model.horizon_ms = h
    model.meta = {'source': str(args.data), 'rows': int(train.sum())}
    exported = export_linear(model, print_it=False)

    # Threshold from the exported (rounded) model's own forecasts over all rows of the tape.
    pred = model.predict(X)
    absp = np.abs(pred)
    thr = float(f'{np.quantile(absp, args.thr_quantile):.3g}')

    text = args.template.read_text(encoding='utf-8')
    s, e = dict_span(text)
    out = text[:s] + format_dict(exported) + text[e:]
    out, n = re.subn(r"('ml_thr':\s*)[0-9.eE+-]+", lambda m: f'{m.group(1)}{thr}', out, count=1)
    if n != 1:
        raise SystemExit("template PARAMS has no numeric 'ml_thr'")
    args.out.write_text(out, encoding='utf-8')

    print(f'rows {len(y):,} (stride {int(data["stride"])}), fitted on {int(train.sum()):,}, tape {args.data}')
    print(f'IC fitted data {pearson(pred[train], y[train]):+.4f}', end='')
    if test.any():
        print(f', holdout {pearson(pred[test], y[test]):+.4f} ({int(test.sum()):,} rows)', end='')
    print(f'\nforecast std {pred.std():.4f} ticks, label std {y.std():.4f}')
    qs = {q: float(np.quantile(absp, q / 100)) for q in (70, 80, 90, 95)}
    print('|pred| quantiles ' + ', '.join(f'{q}%: {v:.3f}' for q, v in qs.items()))
    print(f"chosen ml_thr {thr} (|pred| quantile {args.thr_quantile}, share above {np.mean(absp > thr):.1%}), wrote {args.out}")
    print(f'next grid: ml_thr around {thr} (x0.7, x0.85, x1, x1.2, x1.5)')
    rc = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'research.check', str(args.out), '--no-run']).returncode
    if rc:
        raise SystemExit(f'research.check failed ({rc})')


if __name__ == '__main__':
    main()
