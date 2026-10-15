#!/bin/bash
# Steps 1-2 of the two-model plan: fast path validation + predictions, then the hybrid evaluation.
#   bash evaluations/fastpath/launch.sh                 # from the repository root, on the login node
# Env: PARTITION (default cpu_sapphire), N_VAL / N_PRED tasks (8 / 8), F_RUN / ESP_RUN for hybrid.py.
# Outputs: evaluations/fastpath/results/report_validate.md, results/fastpath_pred.json,
#          evaluations/esp/results/report_hybrid_fast.md
set -euo pipefail
P=${PARTITION:-cpu_sapphire}
N_VAL=${N_VAL:-8}; N_PRED=${N_PRED:-8}
mkdir -p evaluations/fastpath/slurm
ids=()
for i in $(seq 0 $((N_VAL - 1))); do
  ids+=("$(sbatch --parsable -p "$P" --job-name=fp_val_$i evaluations/fastpath/slurm/task.sh validate "$i" "$N_VAL")")
done
for i in $(seq 0 $((N_PRED - 1))); do
  ids+=("$(sbatch --parsable -p "$P" --job-name=fp_pred_$i evaluations/fastpath/slurm/task.sh predict "$i" "$N_PRED")")
done
dep=$(IFS=:; echo "${ids[*]}")
fin=$(sbatch --parsable -p "$P" --dependency=afterany:"$dep" --export=ALL evaluations/fastpath/slurm/finish.sh)
echo "tasks: ${ids[*]} on $P; finish (report + merge + hybrid): $fin"
