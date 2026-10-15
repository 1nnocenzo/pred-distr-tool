#!/bin/bash
# One task of fastpath/run_fastpath.py (validate | predict); submitted by fastpath/launch.sh.
#SBATCH --job-name=fastpath
#SBATCH --output=evaluations/fastpath/slurm/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1
python3 -u evaluations/fastpath/run_fastpath.py "$1" --task-id "$2" --n-tasks "$3" ${EXTRA_ARGS:-}
