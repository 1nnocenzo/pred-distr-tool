#!/bin/bash
#SBATCH --job-name=diag_v4v5
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=evaluations/diagnostics_v4v5/results/%x_%j.log
#     sbatch evaluations/diagnostics_v4v5/diagnose.sh [all|channels|uniform|parts|attention]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
python3 -u evaluations/diagnostics_v4v5/diagnose.py "${1:-all}"
