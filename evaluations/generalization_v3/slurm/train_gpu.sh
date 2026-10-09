#!/bin/bash
# One v3 configuration on one GPU (default A100; e.g. sbatch -p gpu_a40 for an A40; same options as train_cpu.sh). Log names keep
# the cpu_ prefix so that evaluations/status.py shows their progress.
# same RUN: thanks to the per-split lock every job trains a different unfinished
# split (finished splits are skipped, a split being trained elsewhere is skipped),
# so N jobs train N splits in parallel.  GPU jobs on the same RUN coexist safely.
#
#     for i in 1 2 3; do
#       RUN=xattn_a00 MODEL=xattn ALPHA=0 sbatch --partition=cpu_sapphire \
#         evaluations/generalization_v3/slurm/train_cpu.sh
#     done
#
# Overrides: RUN MODEL ALPHA SEED LR CLIP EXPERIMENTS FAMILIES SPLIT_GROUPS CONDA_ENV
#            RESULTS_ROOT (default evaluations/generalization_v3/results)
#            EXTRA_ARGS (passed verbatim to run_experiment.py, e.g. "--loss mixed --val-mode random")
# (FAMILIES / SPLIT_GROUPS restrict lofo / logo to the given comma-separated splits.)

#SBATCH --job-name=v3_gpu
#SBATCH --output=evaluations/generalization_v3/results/slurm/cpu_%x_%j.log
#SBATCH --error=evaluations/generalization_v3/results/slurm/cpu_%x_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "$HOME/miniforge3/bin/conda" ]; then
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
else
  eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
fi
conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
source evaluations/generalization_v3/slurm/pick_gpu.sh
pick_working_gpu || exit 3
cd "$REPO_DIR"

RUN="${RUN:?set RUN=<name>}"
MODEL="${MODEL:-xattn}"
ALPHA="${ALPHA:-0.5}"
LR="${LR:-3e-4}"
CLIP="${CLIP:-1.0}"
SEED="${SEED:-5}"
EXPERIMENTS="${EXPERIMENTS:-lofo size}"
FAMILIES="${FAMILIES:-large}"
SPLIT_GROUPS="${SPLIT_GROUPS:-vqe,fourier,variational}"
RESULTS_ROOT="${RESULTS_ROOT:-evaluations/generalization_v3/results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== v3 $RUN on GPU $(hostname) (${OMP_NUM_THREADS} threads): model=$MODEL alpha=$ALPHA seed=$SEED ==="
for exp in $EXPERIMENTS; do
  python3 -u evaluations/generalization_v3/scripts/run_experiment.py "$exp" \
    --model "$MODEL" --balance-alpha "$ALPHA" --lr "$LR" --grad-clip "$CLIP" \
    --seed "$SEED" --families "$FAMILIES" --groups "$SPLIT_GROUPS" $EXTRA_ARGS \
    --results-dir "$RESULTS_ROOT/$RUN"
done
echo "=== Done: $RUN ==="
