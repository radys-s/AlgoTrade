"""One-command refit of every learned table of the round-3 stack (mm_r3_gru) on a new public tape.

    .venv/Scripts/python.exe -X utf8 -m research.retrain --csv data/<new>.csv --base strategies/mm_r3_gru.py \
        --out strategies/exp_r4_auto.py [--steps ridge,mlp,mq,slow,seq] [--force]

Pipeline (each component fitted on the first 75 % of the tape + its mirror, exactly like the round-3 scripts):
  mirror   data/<stem>_mirror.csv via research.diag.mirror (reused if present)
  features data/features/<stem>_auto.npz and <stem>_mirror_auto.npz via research.features (reused if newer than the csv)
  rows     research/retrain_feats.py -> data/features/<stem>[_mirror]_auto_rows.npz (lean 22, extras, slow columns)
  ridge    pred_rfit_r3 'h200_r3m_75' -> the PRED_MODELS entry used by PARAMS rfit
  slow     slow_r3 ridges (1 s, 5 s) -> SLW_STD1 / SLW_STD5
  mlp      mlp_r3 train --act relu -> MLP_MEAN / MLP_SCALE / MLP_LAYERS          (torch env, research/retrain_torch.py)
  mq       mlpq_r3 train 500 --act relu -> MLPQ[500]                              (torch env)
  seq      seq_r3 --T 128 --hidden 16 --lite 1 + export -> SEQ_*                   (torch env)
Steps left out keep the base file's tables. Non-learned constants and PARAMS stay as they are.
Then research/check_submission.py runs on the output and a timing per step is printed."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TORCH_PY = os.path.join(os.environ.get('USERPROFILE', ''), 'torch-env', 'Scripts', 'python.exe')
ALL_STEPS = ('ridge', 'mlp', 'mq', 'slow', 'seq')
T = {}


def tick(name, t0):
    T[name] = time.time() - t0
    print(f'[retrain] {name}: {T[name]:.1f}s', flush=True)


def run(cmd, log=None):
    print('[retrain] $', ' '.join(str(c) for c in cmd), flush=True)
    if log:
        with open(log, 'w', encoding='utf8') as fh:
            r = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f'failed ({r.returncode}): {cmd}' + (f' (log {log})' if log else ''))


def stale(out: Path, src: Path, force: bool) -> bool:
    return force or not out.exists() or out.stat().st_mtime < src.stat().st_mtime


def build_rows(csv, npz, out):
    sys.path.insert(0, str(ROOT))
    from research import retrain_feats
    t0 = time.time()
    rows = retrain_feats.tape_rows(str(csv), str(npz))
    np.savez(out, **rows)
    return f'{out} {time.time() - t0:.0f}s'


def fit_ridge_rfit(rows):
    """pred_rfit_r3 variant h200_r3m_75: lean 22, H 200, alpha 10, first 75 % of tape + mirror strided by 3."""
    from research.signals import fit_ridge
    from research.retrain_feats import LEAN
    X = np.vstack([r['r200_tr_X'] for r in rows]); y = np.concatenate([r['r200_tr_y'] for r in rows]).astype(np.float64)
    m = fit_ridge(X, y, LEAN, alpha=10.0, clip_q=0.0)
    wr = m.coef / m.scale
    c0 = float(m.intercept - (wr * m.mean).sum())
    return round(c0, 7), tuple(round(float(x), 7) for x in wr)


def fit_slow(rows):
    """slow_r3.main: per horizon (1 s, 5 s) ridge on clipped z, alpha 10 -> (mean, scale, w, intercept)."""
    out = {}
    for h in (1, 5):
        Xtr, ytr = [], []
        for r in rows:
            ts = r['ts']; y = r[f'slow_y{h}']; t_end = float(r['t_end'])
            from research.signals import LATENCY_US
            m = np.isfinite(y) & (ts + LATENCY_US + h * 1_000_000 <= t_end)
            idx = np.where(m)[0][::3]
            Xtr.append(r['slow_X'][idx]); ytr.append(y[idx])
        Xtr = np.vstack(Xtr).astype(np.float64); ytr = np.concatenate(ytr)
        mean = Xtr.mean(0); sc = Xtr.std(0); sc[sc == 0] = 1
        Z = np.clip((Xtr - mean) / sc, -8, 8)
        A = Z.T @ Z + 10.0 * np.eye(Z.shape[1]); ym = ytr.mean()
        w = np.linalg.solve(A, Z.T @ (ytr - ym))
        out[h] = (mean.tolist(), sc.tolist(), w.tolist(), float(ym))
        print(f'[retrain] slow H={h}s n={len(ytr)} ystd={ytr.std():.3f}', flush=True)
    return out


def replace_block(src, start_pat, end_pat, new, what):
    """Replace from the line matching start_pat through the first following line matching end_pat (inclusive)."""
    lines = src.split('\n')
    i = next((k for k, l in enumerate(lines) if re.match(start_pat, l)), None)
    if i is None:
        raise SystemExit(f'{what}: start {start_pat!r} not found in base')
    j = next((k for k in range(i, len(lines)) if re.match(end_pat, lines[k])), None)
    if j is None:
        raise SystemExit(f'{what}: end {end_pat!r} not found in base')
    return '\n'.join(lines[:i] + new.rstrip('\n').split('\n') + lines[j + 1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--base', type=Path, default=Path('strategies/mm_r3_gru.py'))
    ap.add_argument('--out', type=Path, default=Path('strategies/exp_r4_auto.py'))
    ap.add_argument('--steps', default=','.join(ALL_STEPS))
    ap.add_argument('--mirror', type=Path, default=None, help='mirror csv (default data/<stem>_mirror.csv, built if missing)')
    ap.add_argument('--force', action='store_true', help='rebuild features / rows even if present')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seq-epochs', type=int, default=6)
    a = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    steps = [s for s in a.steps.split(',') if s]
    bad = set(steps) - set(ALL_STEPS)
    if bad:
        raise SystemExit(f'unknown steps {bad}')
    t_all = time.time()
    csv = a.csv
    stem = csv.stem
    mir = a.mirror or csv.with_name(stem + '_mirror.csv')
    tag = f'{stem}_auto'
    logd = Path('logs'); logd.mkdir(exist_ok=True)

    t0 = time.time()
    if not mir.exists():
        run([PY, '-X', 'utf8', '-m', 'research.diag.mirror', csv, mir])
    tick('mirror', t0)

    t0 = time.time()
    feats = {}
    procs = []
    for c in (csv, mir):
        f = Path('data/features') / f'{c.stem}_auto.npz'
        feats[c] = f
        if stale(f, c, a.force):
            log = logd / f'retrain_feat_{c.stem}.txt'
            procs.append((subprocess.Popen([PY, '-X', 'utf8', '-m', 'research.features', str(c), '--out', str(f), '--no-compress'],
                                           cwd=ROOT, stdout=open(log, 'w'), stderr=subprocess.STDOUT), log))
    for p, log in procs:
        if p.wait():
            raise SystemExit(f'features failed, see {log}')
    tick('features', t0)

    t0 = time.time()
    rows_p = {c: Path('data/features') / f'{c.stem}_auto_rows.npz' for c in (csv, mir)}
    todo = [c for c in (csv, mir) if stale(rows_p[c], feats[c], a.force)]
    if todo:
        with ProcessPoolExecutor(len(todo)) as ex:
            for msg in ex.map(build_rows, todo, [feats[c] for c in todo], [rows_p[c] for c in todo]):
                print('[retrain] rows', msg, flush=True)
    tick('rows', t0)

    # torch steps in the background while the numpy fits run
    tpro = {}
    rows_args = [str(rows_p[csv]), str(rows_p[mir])]
    for s in ('mlp', 'mq', 'seq'):
        if s in steps:
            out = logd / f'retrain_{tag}_{s}.py'
            cmd = [TORCH_PY, '-X', 'utf8', '-m', 'research.retrain_torch', s, '--rows', *rows_args, '--out', str(out),
                   '--seed', str(a.seed), '--seq-epochs', str(a.seq_epochs)]
            if s == 'seq':
                cmd += ['--feat', str(feats[csv]), str(feats[mir])]
            log = logd / f'retrain_{tag}_{s}.log'
            print('[retrain] $', ' '.join(cmd), f'> {log}', flush=True)
            tpro[s] = (subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, 'w'), stderr=subprocess.STDOUT), out, log, time.time())

    src = a.base.read_text(encoding='utf8')
    if 'ridge' in steps or 'slow' in steps:
        rows = [dict(np.load(p)) for p in rows_args]
    if 'ridge' in steps:
        t0 = time.time()
        c0, w = fit_ridge_rfit(rows)
        m = re.search(r"'rfit':\s*(\d+)", src)
        k = int(m.group(1))
        pat = re.compile(r'(    # rfit %d \()[^\n]*\n    \([^\n]*\),\n' % k)
        if not pat.search(src):
            raise SystemExit(f'PRED_MODELS entry for rfit {k} not found in base')
        new = f'    # rfit {k} (auto: h200 on {stem} + mirror, first 75 %, research/retrain.py); params rfit={k}\n    ({c0}, {w}),\n'
        src = pat.sub(lambda _: new, src, count=1)
        print(f'[retrain] ridge rfit {k}: c0 {c0} w[:4] {w[:4]}', flush=True)
        tick('ridge', t0)
    if 'slow' in steps:
        t0 = time.time()
        sl = fit_slow(rows)
        src = replace_block(src, r'SLW_STD1 = ', r'SLW_STD1 = ', f'SLW_STD1 = {sl[1]!r}', 'slow')
        src = replace_block(src, r'SLW_STD5 = ', r'SLW_STD5 = ', f'SLW_STD5 = {sl[5]!r}', 'slow')
        tick('slow', t0)
    if 'ridge' in steps or 'slow' in steps:
        del rows

    for s, (p, out, log, ts0) in tpro.items():
        if p.wait():
            raise SystemExit(f'{s} failed, see {log}')
        T[s] = time.time() - ts0
        print(f'[retrain] {s}: {T[s]:.1f}s (torch, parallel; log {log})', flush=True)
        g = {}
        txt = out.read_text(encoding='utf8')
        if s == 'mlp':
            exec(txt, g)
            new = f"MLP_MEAN = {g['MLP_MEAN']!r}\nMLP_SCALE = {g['MLP_SCALE']!r}\nMLP_LAYERS = [\n"
            new += ''.join(f'    ({w!r}, {b!r}),\n' for w, b in g['MLP_LAYERS']) + ']'
            src = replace_block(src, r'MLP_MEAN = ', r'\]\s*$', new, 'mlp')
        elif s == 'mq':
            exec(txt, g)
            src = replace_block(src, r'MLPQ\[500\] = ', r'MLPQ\[500\] = ',
                                f"MLPQ[500] = ({g['MLP_MEAN']!r}, {g['MLP_SCALE']!r}, {[tuple(l) for l in g['MLP_LAYERS']]!r})", 'mq')
        elif s == 'seq':
            src = replace_block(src, r'SEQ_LITE = ', r'\]\s*$', txt, 'seq')

    head = (f'"""exp_r4_auto: {a.base.stem} with the learned tables ({",".join(steps)}) refitted by research/retrain.py on '
            f'{csv.name} + mirror (first 75 %).\n\n')
    assert src.startswith('"""')
    src = head + src[3:]
    a.out.write_text(src, encoding='utf8')
    print(f'[retrain] wrote {a.out}', flush=True)

    t0 = time.time()
    r = subprocess.run([PY, '-X', 'utf8', 'research/check_submission.py', str(a.out)], cwd=ROOT)
    tick('check', t0)
    print('[retrain] timings: ' + ', '.join(f'{k} {v:.0f}s' for k, v in T.items()) + f'; total {time.time() - t_all:.0f}s', flush=True)
    if r.returncode:
        raise SystemExit('check_submission FAILED')


if __name__ == '__main__':
    main()
