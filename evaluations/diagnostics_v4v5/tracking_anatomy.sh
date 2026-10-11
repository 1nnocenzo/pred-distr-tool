#!/bin/bash
#SBATCH --job-name=trk_anatomy
#SBATCH --output=evaluations/diagnostics_v4v5/results/tracking_anatomy_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
# Where and why the tracking fails (see tracking_anatomy.py); args passed through.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
export PYTHONPATH="$PWD/src"
cd evaluations/diagnostics_v4v5
python3 -u tracking_anatomy.py "$@"
