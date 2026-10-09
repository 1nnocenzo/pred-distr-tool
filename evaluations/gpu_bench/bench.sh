#!/bin/bash
#SBATCH --job-name=gpu_bench
#SBATCH --output=evaluations/gpu_bench/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# GPU throughput check: 3 epochs of v7 (lambda 0.1) on LOFO qaoa, same settings as the
# CPU runs (~77 s/epoch on 8 CPU cores).  sbatch -p <partition> --gres=gpu:N bench.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$PWD/src" LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source evaluations/generalization_v3/slurm/pick_gpu.sh
pick_working_gpu
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
python3 -u evaluations/generalization_v3/scripts/run_experiment.py lofo --model sinkhorn \
  --balance-alpha 0 --lr 3e-4 --grad-clip 1.0 --seed 5 --families qaoa --loss mixed --val-mode random \
  --epochs 3 --layout-lambda 0.1 --layout-loss region_dist \
  --layouts-path evaluations/compile_check/results/compiler_layouts_v6pilot.pt \
  --results-dir "$HOME/claude_scratch/gpu_bench_$SLURM_JOB_ID" --no-save-model --overwrite
