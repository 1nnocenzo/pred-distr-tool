#!/bin/bash
# Full evaluation of the retrained priority-1 models (generalization_v10): calibration test of every available model,
# ESP of the ESP models, hybrid rule, decision policies, and the old-vs-new comparison of the paper storyline.
#SBATCH --job-name=eval_story_v10
#SBATCH --output=evaluations/retrain/logs/%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
export GS_RESULTS_ROOT=generalization_v10 GS_CALIB_DIR=results_k_v10 GS_TAG=_v10
(cd evaluations/calibration_shift && python3 -u shift_k.py predict && python3 -u shift_k.py report)
for rk in v8kde10:sinkhorn_rf_route_czc_esp v8kda10:sinkhorn_rf_route_czc_espa v8kds10:sinkhorn_rf_route_czc_espa; do
  python3 -u evaluations/esp/esp_eval.py --run ${rk%%:*} --kind ${rk#*:} --root evaluations/generalization_v10/results
done
(cd evaluations/esp && python3 -u hybrid.py --esp-run v8kdg10)
(cd evaluations/calibration_shift && python3 -u ensemble_tracking.py --models v8kdc10 && \
  python3 -u decision_policies.py --models v8kdc10,v8kdg10,v8kdi10,v8kdc10_ens --tag _v10)
python3 -u evaluations/retrain/compare_v10.py --runs v1,v4_phys,v5_sinkhorn,v8a_c001,v8k,v8kd40,v8kdl40,v8kdc10,v8kde10,v8kda10,v8kdi10,v8kdg10
cp evaluations/retrain/results/compare_v10.md evaluations/retrain/results/compare_v10_storyline.md
