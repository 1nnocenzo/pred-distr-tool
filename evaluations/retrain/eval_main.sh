#!/bin/bash
# Early evaluation of the retrained main F model (v8kdc10) and its base (v8k): calibration test of every model
# available so far, comparison with the original models.  The full best-model evaluation (eval_best.sh) follows.
#SBATCH --job-name=eval_main_v10
#SBATCH --output=evaluations/retrain/logs/%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
export GS_RESULTS_ROOT=generalization_v10 GS_CALIB_DIR=results_k_v10 GS_TAG=_v10
(cd evaluations/calibration_shift && python3 -u shift_k.py predict && python3 -u shift_k.py report)
python3 -u evaluations/retrain/compare_v10.py --runs v1,v4_phys,v8k,v8kdc10
cp evaluations/retrain/results/compare_v10.md evaluations/retrain/results/compare_v10_main.md
