"""Bayesian (TPE) tuning of a strategy's PARAMS with Optuna, batched over research.run processes.

Each batch asks Optuna for N parameter sets, runs them in parallel (one research.run --set per set,
--no-ledger for speed is NOT used: every run still lands in the ledger), reads train/holdout pnl from
the output and tells Optuna the TRAIN pnl (the holdout is never used for the search). At the end the
best trials by train pnl are printed with their holdout, so the human picks a plateau point.

Usage:
  uv run --with optuna python -X utf8 -m research.optuna_tune strategies/mm_r1_final2.py data/round1.csv \
      --batches 2 --batch-size 12 --tag r1-optuna
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

import optuna

SPACE = {
    'ml_thr': ('float', 0.08, 0.14),
    'ml_thr_lo': ('float', 0.03, 0.10),
    'ml_frac_lo': ('float', 0.2, 0.8),
    'ml_stub': ('cat', [0, 1, 3]),
    'park_size': ('int', 10, 30),
    'park_max': ('int', 3, 8),
    'park_depth': ('cat', [1, 2]),
    'quote_size': ('cat', [160, 180, 200]),
    'ml_tight_only': ('cat', [0, 1]),
}

RE_PNL = re.compile(r'pnl=\s*([-\d,]+)\s+hold=\s*([-\d,]+)')


def suggest(trial: optuna.Trial) -> dict:
    params = {}
    for name, spec in SPACE.items():
        if spec[0] == 'float':
            params[name] = round(trial.suggest_float(name, spec[1], spec[2]), 4)
        elif spec[0] == 'int':
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def run_batch(strategy: str, data: str, sets: list[dict], tag: str) -> list[tuple[float, float] | None]:
    procs = []
    for i, s in enumerate(sets):
        cmd = [sys.executable, '-X', 'utf8', '-m', 'research.run', '--strategy', strategy, '--data', data,
               '--set', json.dumps(s), '--workers', '1', '--tag', f'{tag}-{i}']
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    out = []
    for p in procs:
        text = p.communicate()[0]
        m = RE_PNL.search(text)
        if m:
            out.append((float(m.group(1).replace(',', '')), float(m.group(2).replace(',', ''))))
        else:
            out.append(None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy')
    ap.add_argument('data')
    ap.add_argument('--batches', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=12)
    ap.add_argument('--tag', default='optuna')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=a.batch_size))
    results = []
    for b in range(a.batches):
        trials = [study.ask() for _ in range(a.batch_size)]
        sets = [suggest(t) for t in trials]
        res = run_batch(a.strategy, a.data, sets, f'{a.tag}-b{b}')
        for t, s, r in zip(trials, sets, res):
            if r is None:
                study.tell(t, state=optuna.trial.TrialState.FAIL)
                continue
            pnl, hold = r
            train = pnl - hold
            study.tell(t, train)
            results.append((train, hold, pnl, s))
            print(f'[b{b}] train={train:,.0f} hold={hold:,.0f} pnl={pnl:,.0f} {json.dumps(s)}', flush=True)
    results.sort(key=lambda x: -x[0])
    print('\nTop by TRAIN pnl (holdout shown, never used for the search):')
    for train, hold, pnl, s in results[:10]:
        print(f'train={train:,.0f} hold={hold:,.0f} pnl={pnl:,.0f} {json.dumps(s)}')


if __name__ == '__main__':
    main()
