"""Compact top-N of a ledger (full-data rows only): pnl, holdout, chunk wins, min chunk, params that vary.

    uv run python -X utf8 research/diag/top.py research/results/<ledger>.csv [--n 8] [--data round1]
"""

from __future__ import annotations

import argparse
import csv
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--data', default=None)
    a = ap.parse_args()
    with open(a.ledger, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if not r.get('error') and '@' not in r['dataset']]
    if a.data:
        rows = [r for r in rows if r['dataset'] == a.data]
    if not rows:
        print('no full-data rows')
        return
    ovs = [json.loads(r['overrides']) for r in rows]
    keys = sorted({k for o in ovs for k in o})
    vary = [k for k in keys if len({json.dumps(o.get(k)) for o in ovs}) > 1]
    order = sorted(range(len(rows)), key=lambda i: -float(rows[i]['pnl']))
    print(f"{len(rows)} full runs; varying: {', '.join(vary) or '-'}")
    for i in order[: a.n]:
        r = rows[i]
        p = ' '.join(f'{k}={ovs[i].get(k)}' for k in vary)
        print(f"{float(r['pnl']):>10,.0f} hold {float(r['hold_pnl']):>9,.0f} wins {r['chunk_wins']}/{r['chunks']} "
              f"min {float(r['chunk_min']):>8,.0f} pen {float(r['breach_penalty']):.0f} {r['dataset']} | {p}")


if __name__ == '__main__':
    main()
