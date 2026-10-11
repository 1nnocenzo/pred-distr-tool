#!/bin/bash
#SBATCH --job-name=calib_k
#SBATCH --output=evaluations/calibration_shift/results_k/%x_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Low-noise calibration shift (see shift_k.py):  sbatch evaluations/calibration_shift/shift_k.sh [truth|predict_report]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
cd evaluations/calibration_shift
if [ "${1:-truth}" = truth ]; then
  python3 -u shift_k.py truth --k 5 --workers 32
else
  export OMP_NUM_THREADS=8
  python3 -u shift_k.py predict
  python3 -u shift_k.py report
fi
