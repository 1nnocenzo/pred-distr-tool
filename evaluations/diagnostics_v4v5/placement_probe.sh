#!/bin/bash
#SBATCH --job-name=place_probe
#SBATCH --output=evaluations/diagnostics_v4v5/results/placement_probe_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# Mechanistic look at the Sinkhorn placement (see placement_probe.py); args passed through.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
export PYTHONPATH="$PWD/src"
cd evaluations/diagnostics_v4v5
python3 -u placement_probe.py "$@"
