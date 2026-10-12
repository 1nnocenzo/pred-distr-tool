#!/bin/bash
#SBATCH --job-name=esp_tvar
#SBATCH --output=evaluations/compile_check/results/esp_tvariants/slurm_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:45:00
# for i in $(seq 0 15); do sbatch -p cpu_sapphire evaluations/compile_check/esp_tvariants.sh run $i 16; done
# sbatch --dependency=afterany:<ids> evaluations/compile_check/esp_tvariants.sh report
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src"
if [ "$1" = report ]; then python3 -u evaluations/compile_check/esp_tvariants.py report
else python3 -u evaluations/compile_check/esp_tvariants.py run --task-id "$2" --n-tasks "$3"; fi
