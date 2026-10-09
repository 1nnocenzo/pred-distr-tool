#!/bin/bash
#SBATCH --job-name=timing_scaling
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=evaluations/timing_v8/%x_%j.log
# Exhaustive compilation vs v8 predictor for 3-27 devices (see timing_scaling.py).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
python3 -u evaluations/timing_v8/timing_scaling.py --per-family 10 --threads 1 "$@"
