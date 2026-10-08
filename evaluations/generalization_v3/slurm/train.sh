#!/bin/bash
# One v3 configuration (control + lofo + size) on one GPU.  Submit from the
# repository root, one job per configuration:
#
#     RUN=pooled_stab_a05 MODEL=pooled ALPHA=0.5 sbatch evaluations/generalization_v3/slurm/train.sh
#     RUN=xattn_a05       MODEL=xattn  ALPHA=0.5 sbatch evaluations/generalization_v3/slurm/train.sh
#
# Results: evaluations/generalization_v3/results/$RUN/.  Idempotent like v2
# (finished splits skipped, interrupted ones resumed, locked ones skipped).
#
# Overrides: RUN MODEL ALPHA LR CLIP FAMILIES EXPERIMENTS MAX_TRAIN_QUBITS CONDA_ENV

#SBATCH --job-name=gen_v3
#SBATCH --output=evaluations/generalization_v3/results/slurm/train_%x_%j.log
#SBATCH --error=evaluations/generalization_v3/results/slurm/train_%x_%j.err
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

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
cd "$REPO_DIR"

RUN="${RUN:?set RUN=<name>}"
MODEL="${MODEL:-xattn}"
ALPHA="${ALPHA:-0.5}"
LR="${LR:-3e-4}"
CLIP="${CLIP:-1.0}"
FAMILIES="${FAMILIES:-large}"
EXPERIMENTS="${EXPERIMENTS:-control lofo size}"
MAX_TRAIN_QUBITS="${MAX_TRAIN_QUBITS:-14}"
RESULTS="evaluations/generalization_v3/results/$RUN"

echo "=== v3 $RUN on $(hostname): model=$MODEL alpha=$ALPHA lr=$LR clip=$CLIP ==="
# No silent CPU fallback (v2's job ended up on the CPU because it requested no GPU),
# and skip a broken GPU if more than one was requested (see pick_gpu.sh).
source evaluations/generalization_v3/slurm/pick_gpu.sh
pick_working_gpu || exit 3

for exp in $EXPERIMENTS; do
  echo "=== experiment: $exp ==="
  python3 -u evaluations/generalization_v3/scripts/run_experiment.py "$exp" \
    --model "$MODEL" --balance-alpha "$ALPHA" --lr "$LR" --grad-clip "$CLIP" \
    --families "$FAMILIES" --max-train-qubits "$MAX_TRAIN_QUBITS" \
    --results-dir "$RESULTS"
done
echo "=== Done: $RESULTS ==="
