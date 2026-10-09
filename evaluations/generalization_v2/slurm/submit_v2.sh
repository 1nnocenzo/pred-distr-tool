#!/bin/bash
# Generalization study v2 (device-graph encoder + log fidelity + family-balanced
# sampling + out-of-distribution validation) on one node (GPU optional).
#
# Submit from the repository root:
#
#     sbatch evaluations/generalization_v2/slurm/submit_v2.sh
#
# No GPU is requested or required: the job runs on the gpu_h200 partition and
# uses CUDA only if a device happens to be visible. It is idempotent: finished splits are skipped, interrupted ones resume
# from their per-epoch checkpoint, and a split that another job is training is
# skipped (per-split lock), so resubmitting or chaining with afterany is safe:
#
#     first=$(sbatch --parsable evaluations/generalization_v2/slurm/submit_v2.sh)
#     sbatch --dependency=afterany:$first evaluations/generalization_v2/slurm/submit_v2.sh
#
# Overrides (export before sbatch):
#   FAMILIES=qaoa,vqe_su2      subset of leave-one-family-out folds (default: large)
#   MAX_TRAIN_QUBITS=14        size-extrapolation cut-offs
#   EXPERIMENTS="control lofo size"
#   WEIGHTS=0.0:1.0:0.1        scheduling weight sweep
#   SMOKE=1                    2-epoch dry run into results_smoke/
#   COMPILE_CIRCUITS_DIR=/path dataset + createDevice.py location
#   CONDA_ENV=pred-distr

#SBATCH --job-name=gen_v2
#SBATCH --output=evaluations/generalization_v2/results/slurm/gen_v2_%j.log
#SBATCH --error=evaluations/generalization_v2/results/slurm/gen_v2_%j.err
#SBATCH --partition=gpu_h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
STUDY_DIR="$REPO_DIR/evaluations/generalization_v2"
SCRIPTS="$STUDY_DIR/scripts"
V1_SCRIPTS="$REPO_DIR/evaluations/generalization/scripts"
if [ "${SMOKE:-0}" = "1" ]; then
  RESULTS="${RESULTS_DIR:-$STUDY_DIR/results_smoke}"
else
  RESULTS="${RESULTS_DIR:-$STUDY_DIR/results}"
fi
mkdir -p "$RESULTS/slurm"

# --- Conda -----------------------------------------------------------------
if [ -x "$HOME/miniforge3/bin/conda" ]; then
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
elif [ -x "$HOME/miniconda/bin/conda" ]; then
  eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Error: conda executable not found." >&2
  exit 1
fi
conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO_DIR"

# --- Configuration ---------------------------------------------------------
FAMILIES="${FAMILIES:-large}"
MAX_TRAIN_QUBITS="${MAX_TRAIN_QUBITS:-14}"
EXPERIMENTS="${EXPERIMENTS:-control lofo size}"
WEIGHTS="${WEIGHTS:-0.0:1.0:0.1}"
TRAIN_FLAGS=(--results-dir "$RESULTS")
if [ "${SMOKE:-0}" = "1" ]; then
  echo "SMOKE mode: 2 epochs per split."
  TRAIN_FLAGS+=(--epochs 2 --patience 1 --no-save-model --overwrite)
fi

echo "=== Generalization study v2 on $(hostname) ==="
echo "results:     $RESULTS"
echo "experiments: $EXPERIMENTS"
echo "families:    $FAMILIES"

# --- 1. Device graphs ------------------------------------------------------
echo "=== [1/4] device graphs ==="
python3 -u "$SCRIPTS/build_device_graphs.py" --output "$RESULTS/device_graphs.pt"

# --- 2. Training / evaluation ---------------------------------------------
for exp in $EXPERIMENTS; do
  echo "=== [2/4] experiment: $exp ==="
  python3 -u "$SCRIPTS/run_experiment.py" "$exp" \
    --device-graphs "$RESULTS/device_graphs.pt" \
    --families "$FAMILIES" \
    --max-train-qubits "$MAX_TRAIN_QUBITS" \
    "${TRAIN_FLAGS[@]}"
done

# --- 3. Scheduling replay (v1 driver, v2 predictions) ---------------------
LOFO_PRED="$RESULTS/leave_one_family_out/heldout_predictions.json"
if [ -f "$LOFO_PRED" ]; then
  echo "=== [3/4] scheduling replay (leave-one-family-out) ==="
  python3 -u "$V1_SCRIPTS/run_scheduling_eval.py" \
    --results-dir "$RESULTS" --predictions "$LOFO_PRED" --tag leave_one_family_out \
    --fidelity-weights "$WEIGHTS" --per-family
fi
for n in ${MAX_TRAIN_QUBITS//,/ }; do
  pred="$RESULTS/size_extrapolation/qmax_${n}/predictions.json"
  if [ -f "$pred" ]; then
    echo "=== [3/4] scheduling replay (size extrapolation, qmax_${n}) ==="
    python3 -u "$V1_SCRIPTS/run_scheduling_eval.py" \
      --results-dir "$RESULTS" --predictions "$pred" \
      --tag "size_extrapolation_qmax_${n}" --fidelity-weights "$WEIGHTS"
  fi
done

# --- 4. Comparison report --------------------------------------------------
echo "=== [4/4] report (v1 vs v2) ==="
python3 -u "$SCRIPTS/make_report.py" --results-dir "$RESULTS"

echo "=== Done. Artefacts in $RESULTS ==="
