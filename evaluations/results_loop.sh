#!/bin/bash
# Rewrite evaluations/RESULTS.md and evaluations/figures/ every 15 minutes from the result files.
#     sbatch evaluations/results_loop.sh      (5-day limit on cpu_sapphire_ext; resubmit to extend)
#SBATCH --job-name=results_md
#SBATCH --output=evaluations/generalization_v3/results/slurm/results_md_%j.log
#SBATCH --partition=cpu_sapphire_ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5-00:00:00
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
while true; do
  python3 evaluations/results_report.py || echo "$(date) results_report.py failed"
  sleep 900
done
