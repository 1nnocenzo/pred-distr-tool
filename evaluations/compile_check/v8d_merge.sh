#!/bin/bash
#SBATCH --job-name=v8d_merge
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/atudisco/pred-distr-tool/evaluations/compile_check/slurm/%x_%j.log
# Merge the v8b variant compilations into v8b_labels.pt (run after all v8b_compile jobs).
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
cd "$HOME/compileCircuits"
python /home/atudisco/pred-distr-tool/evaluations/compile_check/v8b_variants.py merge \
  --n-aug 100 --out /home/atudisco/pred-distr-tool/evaluations/compile_check/results/v8d
