#!/bin/bash
#SBATCH --job-name=counts_cz
#SBATCH --output=evaluations/diagnostics_v4v5/results/report_counts_check_%x.txt
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# counts_check.py on the v9-root runs (v8r_cz, v8s) and v8r for comparison
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate pred-distr
export PYTHONPATH="$PWD/src"
cd evaluations/diagnostics_v4v5
python3 -u counts_check.py --model v8r --kind sinkhorn_rf --root generalization_v8
python3 -u counts_check.py --model v8r_cz --kind sinkhorn_rf --root generalization_v9
python3 -u counts_check.py --model v8s --kind sinkhorn_rf_route --root generalization_v9
