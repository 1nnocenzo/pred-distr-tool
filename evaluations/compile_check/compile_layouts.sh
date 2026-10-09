#!/bin/bash
#SBATCH --job-name=compile_layouts
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/home/atudisco/pred-distr-tool/evaluations/compile_check/slurm/%x_%j.log
# Compiler layouts of the whole dataset (v6 targets).  One job per task (no arrays here):
#   for t in $(seq 0 15); do sbatch --export=ALL,TASK_ID=$t,NT=16 compile_layouts.sh --exclude grover --save-qpy --out <dir>; done
#   for t in $(seq 0 31); do sbatch --export=ALL,TASK_ID=$t,NT=32 compile_layouts.sh --families grover --save-qpy --out <dir>; done
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV:-pred-distr}"
cd "$HOME/compileCircuits"
python /home/atudisco/pred-distr-tool/evaluations/compile_check/compile_layouts.py run \
  --task-id "$TASK_ID" --n-tasks "$NT" "$@"
