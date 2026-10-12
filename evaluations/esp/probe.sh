#!/bin/bash
# ESP probes of one or more fidelity-model versions (esp_probe.py run), one after the other.
#   RUNS="v8kdc10 v8kd40" SEEDS=5,6,7 sbatch -p cpu_sapphire evaluations/esp/probe.sh
#SBATCH --job-name=esp_probe
#SBATCH --output=evaluations/esp/slurm/%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=04:00:00
set -euo pipefail
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate pred-distr
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" CUDA_VISIBLE_DEVICES=""
for r in ${RUNS:?set RUNS}; do
  python evaluations/esp/esp_probe.py run --run "$r" --seeds "${SEEDS:-5,6,7}" --epochs "${EPOCHS:-40}" ${PROBE_ARGS:-}
done
