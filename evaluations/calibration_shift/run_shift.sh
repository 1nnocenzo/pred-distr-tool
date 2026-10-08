#!/bin/bash
# Full calibration-shift study: ground truth on 18 device variants, predictions, report.
#     sbatch evaluations/calibration_shift/run_shift.sh
#SBATCH --job-name=calib_shift
#SBATCH --output=evaluations/calibration_shift/results/run_%j.log
#SBATCH --error=evaluations/calibration_shift/results/run_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
cd evaluations/calibration_shift
python3 -u run_shift.py truth --n-per-family "${N_PER_FAMILY:-40}" --workers "${SLURM_CPUS_PER_TASK:-32}"
python3 -u run_shift.py predict
python3 -u run_shift.py report
