#!/bin/bash
#SBATCH --job-name=v12_phase1
#SBATCH --output=evaluations/v12/results/phase1_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# v12 phase 1 on one GPU:  sbatch -p gpu_h200 --gres=gpu:2 evaluations/v12/phase1.sh [--steps N]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src"
source evaluations/generalization_v3/slurm/pick_gpu.sh
python3 -u evaluations/v12/phase1_joint.py "$@"
