#!/bin/bash
#SBATCH --job-name=compile_check
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

#SBATCH --output=/home/atudisco/pred-distr-tool/evaluations/compile_check/slurm/%x_%j.log
# Submit once per task (the cluster has no job arrays):
#   for t in $(seq 0 15); do TASK_ID=$t sbatch compile_check.sh --out <dir>; done
# Recompile a stratified qaoa sample with several seeds / levels and check validity +
# equivalence of every result (see compile_check.py).  Extra args are passed through,
# e.g. --layouts layouts.json --out results/model_layouts
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV:-pred-distr}"
cd "$HOME/compileCircuits"
python /home/atudisco/pred-distr-tool/evaluations/compile_check/compile_check.py \
  --task-id "$TASK_ID" --n-tasks 16 "$@"
