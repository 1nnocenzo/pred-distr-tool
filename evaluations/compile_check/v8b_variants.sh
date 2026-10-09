#!/bin/bash
#SBATCH --job-name=v8b_compile
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/atudisco/pred-distr-tool/evaluations/compile_check/slurm/%x_%j.log
# v8b: dataset circuits compiled on 8 calibration variants per device (see v8b_variants.py).
#   for t in $(seq 0 47); do sbatch --export=ALL,TASK_ID=$t,NT=48 v8b_variants.sh --k 3 --out <dir>; done
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV:-pred-distr}"
cd "$HOME/compileCircuits"
python /home/atudisco/pred-distr-tool/evaluations/compile_check/v8b_variants.py run \
  --task-id "$TASK_ID" --n-tasks "$NT" "$@"
