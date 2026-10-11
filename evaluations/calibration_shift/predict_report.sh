#!/bin/bash
# Re-run predictions + report of the calibration-shift study (ground truth reused).
#     sbatch evaluations/calibration_shift/predict_report.sh
#SBATCH --job-name=calib_predict
#SBATCH --output=evaluations/calibration_shift/results/predict_%j.log
#SBATCH --error=evaluations/calibration_shift/results/predict_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
cd evaluations/calibration_shift
python3 -u run_shift.py predict
python3 -u run_shift.py report
