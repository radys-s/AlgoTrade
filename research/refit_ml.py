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
    return brace_span(text, m.end() - 1)


def brace_span(text: str, start: int) -> tuple[int, int]:
    """Return [start, end) of the brace-balanced literal whose opening brace sits at `start`."""
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
    raise SystemExit('unbalanced braces in a model dict literal')


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


def target_b_entry_span(text: str) -> tuple[int, int]:
    """Span of the `'B': {...}` value inside the template's LINEAR_MODELS literal."""
    s0, e0 = dict_span(text, 'LINEAR_MODELS')
    m = re.search(r"""['"]B['"]\s*:\s*\{""", text[s0:e0])
    if m is None:
        raise SystemExit("template LINEAR_MODELS has no 'B' entry")
    return brace_span(text, s0 + m.end() - 1)


def predict_dict(model: dict, X: np.ndarray, names: list[str]) -> np.ndarray:
    """Evaluate an exported model dict on feature rows (columns in `names` order)."""
    if list(model['coef']) != names:
        raise SystemExit('model dict feature order differs from the dataset feature order')
    c = lambda k: np.array([model[k][n] for n in names])  # noqa: E731
    z = (np.clip(X, c('lo'), c('hi')) - c('mean')) / c('scale')
    return z @ c('coef') + model['intercept']


def firing_share(template_text: str, thr: float | None, ref_npz: Path, h: int, max_rows: int) -> tuple[float, float]:
    """Share of events with |pred_B| > the template's ml_thr, B model of the template, on the reference tape."""
    import ast

    s0, e0 = target_b_entry_span(template_text)
    model = ast.literal_eval(template_text[s0:e0])
    if thr is None:
        m = re.search(r"'ml_thr':\s*([0-9.eE+-]+)", template_text)
        thr = float(m.group(1))
    ref = build_dataset(load_features(ref_npz), (h,), max_rows=max_rows)
    names = [str(n) for n in ref['feature_names']]
    absp = np.abs(predict_dict(model, ref['X'].astype(np.float64), names))
    return float(np.mean(absp > thr)), thr


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
    p.add_argument('--target', choices=('base', 'B'), default='base',
                   help="B: loss-aligned target (research.fit_target); replaces the template's LINEAR_MODELS['B'] only")
    p.add_argument('--thr-share', type=float, default=None,
                   help='target B: share of events with |pred| > ml_thr (default: the template B model at its own ml_thr on --ref-data)')
    p.add_argument('--ref-data', type=Path, default=Path('data/round1.csv'), help='target B: tape for the firing-share reference')
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
    if args.target == 'B':
        from research.fit_target import build_targets  # same label construction as the H18 study

        T = build_targets(d, data['row'], y)
        ok = np.isfinite(T['B']) & np.isfinite(T['C'])  # fit_target's `ok`: rows whose 1 s outcome is known
        train &= ok
        y = np.where(ok, T['B'], 0.0)
        test &= ok
    model = fit_ridge(X[train], y[train], names, ALPHA)
    model.horizon_ms = h
    model.meta = {'source': str(args.data), 'rows': int(train.sum())}
    exported = export_linear(model, print_it=False)

    # Threshold from the exported (rounded) model's own forecasts over all rows of the tape.
    pred = model.predict(X)
    absp = np.abs(pred)
    with open(args.template, encoding='utf-8', newline='') as f:  # keep the template's line endings
        text = f.read()
    nl = chr(13) + chr(10) if chr(13) in text else chr(10)
    if args.target == 'B':
        share = args.thr_share
        if share is None:
            ref_npz = ensure_features(args.ref_data)
            share, ref_thr = firing_share(text, None, ref_npz, h, args.max_rows)
            print(f'reference firing share {share:.2%} (template B model, |pred| > {ref_thr} on {args.ref_data})')
        thr_q = 1.0 - share
        s, e = target_b_entry_span(text)
        exported = {k: v for k, v in exported.items() if k not in ('source', 'rows')}  # the template's B entry has no meta
        out = text[:s] + repr(exported) + text[e:]
    else:
        thr_q = args.thr_quantile
        s, e = dict_span(text)
        out = text[:s] + format_dict(exported).replace(chr(10), nl) + text[e:]
    thr = float(f'{np.quantile(absp, thr_q):.3g}')
    out, n = re.subn(r"('ml_thr':\s*)[0-9.eE+-]+", lambda m: f'{m.group(1)}{thr}', out, count=1)
    if n != 1:
        raise SystemExit("template PARAMS has no numeric 'ml_thr'")
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        f.write(out)

    print(f'rows {len(y):,} (stride {int(data["stride"])}), fitted on {int(train.sum()):,}, tape {args.data}')
    print(f'IC fitted data {pearson(pred[train], y[train]):+.4f}', end='')
    if test.any():
        print(f', holdout {pearson(pred[test], y[test]):+.4f} ({int(test.sum()):,} rows)', end='')
    print(f'\nforecast std {pred.std():.4f} ticks, label std {y.std():.4f}')
    qs = {q: float(np.quantile(absp, q / 100)) for q in (70, 80, 90, 95)}
    print('|pred| quantiles ' + ', '.join(f'{q}%: {v:.3f}' for q, v in qs.items()))
    print(f"chosen ml_thr {thr} (|pred| quantile {thr_q:.4g}, share above {np.mean(absp > thr):.1%}), wrote {args.out}")
    print(f'next grid: ml_thr around {thr} (x0.7, x0.85, x1, x1.2, x1.5)')
    rc = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'research.check', str(args.out), '--no-run']).returncode
    if rc:
        raise SystemExit(f'research.check failed ({rc})')


if __name__ == '__main__':
    main()
