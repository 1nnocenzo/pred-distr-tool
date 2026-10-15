#!/bin/bash
# After the tasks: validation report, merged predictions, hybrid evaluation (esp/hybrid.py --fast).
#SBATCH --job-name=fastpath_finish
#SBATCH --output=evaluations/fastpath/slurm/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
python3 -u evaluations/fastpath/run_fastpath.py report
python3 -u evaluations/fastpath/run_fastpath.py merge
bash evaluations/fastpath/hybrid_step.sh
