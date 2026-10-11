#!/bin/bash
#SBATCH --job-name=v8u_merge
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/atudisco/pred-distr-tool/evaluations/compile_check/slurm/%x_%j.log
# v11 pilot: usage maps on the original devices + the 300 v8d calibrations -> results/v8u/usage_labels.pt
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
cd /home/atudisco/pred-distr-tool/evaluations/compile_check
export PYTHONPATH=/home/atudisco/pred-distr-tool/src
python3 merge_usage.py --orig-dir results/v8u_orig --var-dir results/v8u --n-aug 100
