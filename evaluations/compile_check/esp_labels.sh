#!/bin/bash
#SBATCH --job-name=esp_labels
#SBATCH --output=evaluations/compile_check/results/esp_labels/slurm_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
# for i in $(seq 0 15); do sbatch -p cpu_sapphire_ext evaluations/compile_check/esp_labels.sh $i 16; done
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src"
python3 -u evaluations/compile_check/esp_labels.py run --task-id "$1" --n-tasks "$2"
