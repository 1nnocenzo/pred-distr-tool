#!/bin/bash
#SBATCH --job-name=tvar_train
#SBATCH --output=evaluations/compile_check/results/esp_tvar_train/slurm_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
# for i in $(seq 0 95); do sbatch -p cpu_sapphire evaluations/compile_check/esp_tvar_train.sh run $i 96; done
# sbatch --dependency=afterany:<ids> --mem=8G evaluations/compile_check/esp_tvar_train.sh merge
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src"
if [ "$1" = merge ]; then python3 -u evaluations/compile_check/esp_tvar_train.py merge
else python3 -u evaluations/compile_check/esp_tvar_train.py run --task-id "$2" --n-tasks "$3"; fi
