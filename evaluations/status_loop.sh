#!/bin/bash
# Rewrite evaluations/STATUS.md every 5 minutes (reads logs/metrics only).
#     sbatch evaluations/status_loop.sh   (6h limit: short jobs start at once; resubmit to extend)
#SBATCH --job-name=status_md
#SBATCH --output=evaluations/generalization_v3/results/slurm/status_%j.log
#SBATCH --error=evaluations/generalization_v3/results/slurm/status_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
while true; do
  python3 evaluations/status.py --out evaluations/STATUS.md || echo "$(date) status.py failed"
  sleep 300
done
