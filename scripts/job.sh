#!/bin/bash
# One pipeline step as a SLURM job:  sbatch [options] scripts/job.sh <module> [args...]
#   e.g. sbatch -p cpu_sapphire --mem=2G scripts/job.sh labels run --task-id 3 --n-tasks 64
#SBATCH --job-name=esptime
#SBATCH --output=work/slurm/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
for c in "$HOME/miniforge3/bin/conda" "$HOME/miniconda/bin/conda" "$HOME/miniconda3/bin/conda"; do
  [ -x "$c" ] && eval "$("$c" shell.bash hook)" && break
done
conda activate "${CONDA_ENV:-pred-distr}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mod="$1"; shift
python -m "esptime.$mod" "$@"
