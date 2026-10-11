#!/bin/bash
#SBATCH --job-name=cc_consist
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=/home/atudisco/pred-distr-tool/evaluations/compile_check/slurm/%x_%j.log
# Compiler self-consistency (see compiler_consistency.py):  sbatch --export=ALL,CMD=run,TASK_ID=t,NT=24 …  or CMD=report
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
cd "$HOME/compileCircuits"
python /home/atudisco/pred-distr-tool/evaluations/compile_check/compiler_consistency.py "${CMD:-run}" \
  --task-id "${TASK_ID:-0}" --n-tasks "${NT:-1}" --out /home/atudisco/pred-distr-tool/evaluations/compile_check/results/consistency
