"""research/retrain.py variant: refit the non-GRU tables (ridge rfit, slow ridges, taker MLP, quote MLP 500) on several tapes.

    .venv/Scripts/python.exe -X utf8 -m research.retrain_multi --base strategies/mm_r3_gru3.py --out strategies/exp_r4_mix.py \
        --full round3 round3_mirror --part round4 round4_mirror
--full: rows stems whose tr + ho (strided by 3) are all used (a tape that is entirely training data).
--part: rows stems whose tr (first 75 % of that tape) is used. GRU (SEQ_*) tables stay as in the base."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research.retrain import fit_ridge_rfit, fit_slow, replace_block, TORCH_PY, PY  # noqa: E402


def load(stem):
    return dict(np.load(ROOT / 'data' / 'features' / f'{stem}_auto_rows.npz'))


def full_rows(r):
    """Whole-tape version: ridge/MLP rows tr + every 3rd ho row; slow rows without the 75 % cut."""
    o = dict(r)
    for H in (200, 500):
        for k in ('X', 'y', 'spread'):
            o[f'r{H}_tr_{k}'] = np.concatenate([r[f'r{H}_tr_{k}'], r[f'r{H}_ho_{k}'][::3]])
    o['t_end'] = np.array(np.inf)
    return o


def merged_npz(rows, path):
    o = {}
    for H in (200, 500):
        for k in ('X', 'y', 'spread'):
            o[f'r{H}_tr_{k}'] = np.concatenate([r[f'r{H}_tr_{k}'] for r in rows])
            o[f'r{H}_ho_{k}'] = rows[-1][f'r{H}_ho_{k}']
    np.savez(path, **o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=Path('strategies/mm_r3_gru3.py'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--full', nargs='*', default=[])
    ap.add_argument('--part', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    rows = [full_rows(load(s)) for s in a.full] + [load(s) for s in a.part]
    tag = a.out.stem
    mp = Path('logs') / f'{tag}_rows.npz'
    merged_npz(rows, mp)
    procs = {}
    for s in ('mlp', 'mq'):
        out = Path('logs') / f'{tag}_{s}.py'
        log = Path('logs') / f'{tag}_{s}.log'
        cmd = [TORCH_PY, '-X', 'utf8', '-m', 'research.retrain_torch', s, '--rows', str(mp), '--out', str(out), '--seed', str(a.seed)]
        procs[s] = (subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, 'w'), stderr=subprocess.STDOUT), out, log)
    src = a.base.read_text(encoding='utf8')
    c0, w = fit_ridge_rfit(rows)
    k = int(re.search(r"'rfit':\s*(\d+)", src).group(1))
    pat = re.compile(r'(    # rfit %d \()[^\n]*\n    \([^\n]*\),\n' % k)
    assert pat.search(src)
    new = f'    # rfit {k} (multi: {"+".join(a.full + a.part)}, research/retrain_multi.py); params rfit={k}\n    ({c0}, {w}),\n'
    src = pat.sub(lambda _: new, src, count=1)
    sl = fit_slow(rows)
    src = replace_block(src, r'SLW_STD1 = ', r'SLW_STD1 = ', f'SLW_STD1 = {sl[1]!r}', 'slow')
    src = replace_block(src, r'SLW_STD5 = ', r'SLW_STD5 = ', f'SLW_STD5 = {sl[5]!r}', 'slow')
    for s, (p, out, log) in procs.items():
        if p.wait():
            raise SystemExit(f'{s} failed, see {log}')
        g = {}
        exec(out.read_text(encoding='utf8'), g)
        if s == 'mlp':
            new = f"MLP_MEAN = {g['MLP_MEAN']!r}\nMLP_SCALE = {g['MLP_SCALE']!r}\nMLP_LAYERS = [\n"
            new += ''.join(f'    ({w!r}, {b!r}),\n' for w, b in g['MLP_LAYERS']) + ']'
            src = replace_block(src, r'MLP_MEAN = ', r'\]\s*$', new, 'mlp')
        else:
            src = replace_block(src, r'MLPQ\[500\] = ', r'MLPQ\[500\] = ',
                                f"MLPQ[500] = ({g['MLP_MEAN']!r}, {g['MLP_SCALE']!r}, {[tuple(l) for l in g['MLP_LAYERS']]!r})", 'mq')
    src = f'"""{a.out.stem}: {a.base.stem} with ridge/slow/MLP/MLPQ refit on {"+".join(a.full)} (all) + {"+".join(a.part)} (first 75 %).\n\n' + src[3:]
    a.out.write_text(src, encoding='utf8')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
