#!/bin/bash
# esp_eval.py for one run:  RUN=v8kda10 KIND=sinkhorn_rf_route_czc_espa sbatch evaluations/esp/eval.sh
#SBATCH --job-name=esp_eval
#SBATCH --output=evaluations/esp/slurm/%x_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=01:00:00
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
python -u evaluations/esp/esp_eval.py --run "${RUN:?}" --kind "${KIND:-sinkhorn_rf_route_czc_esp}" "$@"
