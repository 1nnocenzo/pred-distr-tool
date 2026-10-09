#!/bin/bash
#SBATCH --job-name=v9f_decide
#SBATCH --output=evaluations/generalization_v9/results/v9f_decide_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
# Runs decide_v9f.py once v9 / v8r_cz / v8s and their calibration tests are done (submit with
# --dependency=afterany:<jobs>); it launches v9f if the add-ons help.  sbatch works from compute nodes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src"
python3 -u evaluations/generalization_v9/decide_v9f.py
