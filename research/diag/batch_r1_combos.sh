#!/usr/bin/env bash
# Round 1 combination batch (radys): factorial of the mechanics that survived single tests, screened on
# the first 25 % of round1, best quarter on the full tape. Prints only compact tops.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
BASE='{"book_scan_orders":1000,"skew_per_lot":0,"quote_size":200,"max_position":100,"park_depth":1,"park_max":3,"flow_pull_us":50000}'
uv run python -X utf8 -m research.run --strategy strategies/exp_ml_shrink.py --data data/round1.csv \
  --grid '{"park_size":[20,30],"ml_thr":[0,0.15,0.3],"ml_stub":[10],"thin_cap":[0,150,300],"thin_cap_size":[20],"flow_pull_qty":[0,1000]}' \
  --set "$BASE" --halving 0.25 --keep 0.25 --min-keep 4 --workers 12 --tag r1-combo1 > logs/r1_combo1.log 2>&1
L=$(ls -t research/results/*r1-combo1.csv | head -1)
uv run python -X utf8 research/diag/top.py "$L" --n 10
