#!/bin/bash
#SBATCH --job-name=trk_oracle
#SBATCH --output=evaluations/diagnostics_v4v5/results/tracking_oracle_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# Oracle substitutions for the local tracking (see tracking_oracle.py); args passed through.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
export PYTHONPATH="$PWD/src"
cd evaluations/diagnostics_v4v5
python3 -u tracking_oracle.py "$@"
