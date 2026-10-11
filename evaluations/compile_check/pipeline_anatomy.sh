#!/bin/bash
#SBATCH --job-name=anatomy
#SBATCH --output=evaluations/compile_check/results/pipeline_anatomy/slurm_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:30:00
# for i in $(seq 0 7); do sbatch -p cpu_sapphire_ext evaluations/compile_check/pipeline_anatomy.sh $i 8; done
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src"
python3 -u evaluations/compile_check/pipeline_anatomy.py run --n-circuits 200 --task-id "$1" --n-tasks "$2" --out "${OUT:-evaluations/compile_check/results/pipeline_anatomy}"
