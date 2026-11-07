#!/usr/bin/env bash
# radys: screen the inventory/fill-freeze/spread ideas on top of mm_ml (exp_overcap, max_position 100 = no over-cap).
set -e
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
uv run python -X utf8 -m research.run --strategy strategies/exp_overcap.py --data data/round1.csv \
  --grid '{"fill_freeze_us":[0,20000,100000],"ml_tight_only":[0,1],"inv_lock":[0,60],"ml_inv_k":[0,0.1]}' \
  --halving 0.25 --keep 0.25 --min-keep 4 --workers 12 --tag r1-ideas2 > logs/r1_ideas2.log 2>&1
uv run python -X utf8 research/diag/top.py "$(ls -t research/results/*r1-ideas2.csv | head -1)" --n 8
