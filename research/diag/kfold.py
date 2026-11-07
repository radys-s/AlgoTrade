"""Blocked k-fold check of parameter selection: is the chosen setting overfit to the tape?

    uv run python -X utf8 research/diag/kfold.py --data round1 --base '{"park_size": 0}' research/results/<ledger>.csv [...]

Blocks = the 8 training chunks + the holdout of every full run (ledger column `chunk_pnls` + `hold_pnl`).
For each block i: pick the config with the highest pnl summed over the OTHER blocks (the honest choice a
researcher would make without seeing block i), then score it on block i against the baseline config.
Not overfit = the out-of-fold pick beats the baseline in most blocks and its out-of-fold total is close to
the in-sample best total. Blocks are contiguous in time (no shuffling: the tape is a time series).
"""

from __future__ import annotations

import argparse
import csv
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('ledgers', nargs='+')
    ap.add_argument('--data', required=True, help='dataset name in the ledger, e.g. round1 (full runs only)')
    ap.add_argument('--base', required=True, help='JSON subset of overrides that identifies the baseline row')
    a = ap.parse_args()
    base_sel = json.loads(a.base)
    rows: dict[str, list[float]] = {}
    base_key = None
    for path in a.ledgers:
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['dataset'] != a.data or r.get('error'):
                    continue
                ov = json.loads(r['overrides'])
                key = json.dumps(ov, sort_keys=True)
                blocks = json.loads(r['chunk_pnls']) + [float(r['hold_pnl'])]
                rows[key] = blocks
                if all(ov.get(k) == v for k, v in base_sel.items()):
                    base_key = key
    if base_key is None:
        raise SystemExit('baseline row not found; pass --base matching one config')
    n = len(rows[base_key])
    keys = list(rows)
    print(f'{len(keys)} configs, {n} time blocks, baseline = {base_key}')
    wins = 0
    oof_total = 0.0
    base_total = 0.0
    picks = []
    for i in range(n):
        best = max(keys, key=lambda k: sum(v for j, v in enumerate(rows[k]) if j != i))
        got = rows[best][i]
        ref = rows[base_key][i]
        wins += got > ref
        oof_total += got
        base_total += ref
        picks.append(best)
        print(f'block {i}: pick {best}  oof {got:,.0f}  baseline {ref:,.0f}  diff {got - ref:+,.0f}')
    in_best = max(keys, key=lambda k: sum(rows[k]))
    print(f'out-of-fold total {oof_total:,.0f} vs baseline {base_total:,.0f} ({oof_total / base_total - 1:+.1%}); '
          f'in-sample best {sum(rows[in_best]):,.0f}; pick wins {wins}/{n} blocks; '
          f'distinct picks {len(set(picks))}')


if __name__ == '__main__':
    main()
