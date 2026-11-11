#!/usr/bin/env bash
# Round-4 kickoff (home brain): one command after the public R4 tape lands in data/.
#   bash research/r4_kickoff.sh data/round4.csv
# Runs in parallel: the one-command refit with 3 GRU seeds (base mm_r3_gru3), the baselines on the new tape + mirror,
# and the ceilings (taker / quote oracles). Logs go to logs/r4_*.log; read them in this order: baselines, ceilings, autos.
set -u
CSV=${1:-data/round4.csv}
STEM=$(basename "$CSV" .csv)
PY=.venv/Scripts/python.exe
cd "$(dirname "$0")/.."
mkdir -p logs
echo "[r4] $(date +%H:%M:%S) tape $CSV"

# 1. mirror + features first (the refit and the oracles need them); the refit reuses them
$PY -X utf8 -m research.diag.mirror "$CSV" > logs/r4_mirror.log 2>&1
MIR=data/${STEM}_mirror.csv
$PY -X utf8 -m research.features "$CSV" > logs/r4_feat.log 2>&1 &
$PY -X utf8 -m research.features "$MIR" > logs/r4_feat_mirror.log 2>&1 &
wait
echo "[r4] $(date +%H:%M:%S) mirror + features done"

# 2. refit with three seeds: seed 0 first (it builds the shared feature / row files), then seeds 1 and 2 together
(
  $PY -X utf8 -m research.retrain --csv "$CSV" --mirror "$MIR" --base strategies/mm_r3_gru3.py \
      --out strategies/exp_r4_auto_s0.py --seed 0 > logs/r4_retrain_s0.log 2>&1
  for s in 1 2; do
    $PY -X utf8 -m research.retrain --csv "$CSV" --mirror "$MIR" --base strategies/mm_r3_gru3.py \
        --out strategies/exp_r4_auto_s$s.py --seed $s > logs/r4_retrain_s$s.log 2>&1 &
  done
  wait
) &

# 3. baselines on the new tape + mirror (+ round3 as the regime reference)
$PY -X utf8 -m research.run --strategy strategies/mm_r3_gru3.py --data "$CSV" "$MIR" data/round3.csv --workers 3 --tag r4-base-gru3 > logs/r4_base_gru3.log 2>&1 &
$PY -X utf8 -m research.run --strategy strategies/mm_r3_ms.py --data "$CSV" "$MIR" --workers 2 --tag r4-base-ms > logs/r4_base_ms.log 2>&1 &
$PY -X utf8 -m research.run --strategy strategies/mm_r2_final.py --data "$CSV" "$MIR" --workers 2 --tag r4-base-r2f > logs/r4_base_r2f.log 2>&1 &
$PY -X utf8 -m research.run --strategy strategies/exp_r3_gx.py --data "$CSV" "$MIR" --workers 2 --tag r4-base-gx > logs/r4_base_gx.log 2>&1 &

# 4. ceilings on the R3 final: perfect 205 ms taker, perfect 1 s quotes
$PY -X utf8 -m research.diag.oracle_pred_r3 strategies/mm_r3_gru3.py "$CSV" 205 1.0 > logs/r4_oracle_taker.log 2>&1 &
$PY -X utf8 -m research.diag.oracle_pred_r3 strategies/mm_r3_gru3.py "$CSV" 0 1.0 1005 > logs/r4_oracle_quote.log 2>&1 &
wait
echo "[r4] $(date +%H:%M:%S) refits, baselines and ceilings done"

# 5. the three refits on the new tape + mirror + round3
for s in 0 1 2; do
  if [ -f strategies/exp_r4_auto_s$s.py ]; then
    $PY -X utf8 -m research.run --strategy strategies/exp_r4_auto_s$s.py --data "$CSV" "$MIR" data/round3.csv --workers 3 --tag r4-auto-s$s > logs/r4_auto_s$s.log 2>&1 &
  fi
done
wait
echo "[r4] $(date +%H:%M:%S) all done"
grep -h "^\[" logs/r4_base_*.log logs/r4_auto_s*.log
grep -h '"pnl"' logs/r4_oracle_*.log
