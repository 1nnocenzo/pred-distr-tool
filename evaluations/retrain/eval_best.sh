#!/bin/bash
# Standard evaluation of the retrained best models (generalization_v10): calibration test, ESP / duration, comparison.
#SBATCH --job-name=eval_best_v10
#SBATCH --output=evaluations/retrain/logs/%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
export GS_RESULTS_ROOT=generalization_v10 GS_CALIB_DIR=results_k_v10 GS_TAG=_v10
(cd evaluations/calibration_shift && python3 -u shift_k.py predict && python3 -u shift_k.py report)
python3 -u evaluations/esp/esp_eval.py --run v8kdg10 --kind sinkhorn_rf_route_czc_esps --root evaluations/generalization_v10/results
python3 -u evaluations/esp/esp_eval.py --run v8kdi10 --kind sinkhorn_rf_route_czc_espi --root evaluations/generalization_v10/results
(cd evaluations/esp && python3 -u esp_duration.py --runs v8kdi10,v8kdg10 --tag _v10)
python3 -u evaluations/retrain/compare_v10.py --runs v8k,v8kdc10,v8kdg10,v8kdi10
