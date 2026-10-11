#!/bin/bash
# Full generalization study on one GPU node.
#
# Submit from the repository root:
#
#     sbatch evaluations/generalization/slurm/submit_generalization.sh
#
# The job is idempotent: every training step skips splits whose results already
# exist, so if it hits the partition time limit it can simply be resubmitted and
# will continue with the next fold. Nothing under evaluations/pipeline/ is
# written to; all output goes to evaluations/generalization/results/.
#
# Useful overrides (export before sbatch, or use --export):
#   FAMILIES=qaoa,qft            subset of leave-one-family-out folds
#   MAX_TRAIN_QUBITS=12,14,16    size-extrapolation cut-offs
#   WEIGHTS=0.0:1.0:0.1          scheduling weight sweep
#   RUN_CONTROL=0                skip the random-split control
#   SMOKE=1                      2-epoch dry run of the whole pipeline
#   COMPILE_CIRCUITS_DIR=/path   location of graph_dataset_expected_fidelity.pt
#   CONDA_ENV=pred-distr         conda environment to activate

#SBATCH --job-name=gen_study
#SBATCH --output=evaluations/generalization/results/slurm/gen_study_%j.log
#SBATCH --error=evaluations/generalization/results/slurm/gen_study_%j.err

#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
STUDY_DIR="$REPO_DIR/evaluations/generalization"
SCRIPTS="$STUDY_DIR/scripts"
# A smoke run trains for 2 epochs; it must never land in the real results tree,
# where the resume logic would later mistake those 2-epoch folds for finished work.
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
# torch pulls in the system libstdc++ first; prepending the environment's lib
# keeps the newer CXXABI symbols that sqlite3/ICU need visible.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO_DIR"

# --- Configuration ---------------------------------------------------------
FAMILIES="${FAMILIES:-large}"
MAX_TRAIN_QUBITS="${MAX_TRAIN_QUBITS:-14}"
WEIGHTS="${WEIGHTS:-0.0:1.0:0.1}"
RUN_CONTROL="${RUN_CONTROL:-1}"
MIN_FAMILY_SIZE="${MIN_FAMILY_SIZE:-50}"

TRAIN_FLAGS=(--results-dir "$RESULTS")
if [ "${SMOKE:-0}" = "1" ]; then
  echo "SMOKE mode: 2 epochs per split, no checkpoints."
  TRAIN_FLAGS+=(--epochs 2 --patience 1 --no-save-model --overwrite)
fi

echo "=== Generalization study ==="
echo "repo:            $REPO_DIR"
echo "results:         $RESULTS"
echo "conda env:       ${CONDA_ENV:-pred-distr}"
echo "dataset dir:     ${COMPILE_CIRCUITS_DIR:-<repository-relative search>}"
echo "families:        $FAMILIES"
echo "qubit cut-offs:  $MAX_TRAIN_QUBITS"
echo "weight sweep:    $WEIGHTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 1. Random-split control (reference row of every table) ----------------
if [ "$RUN_CONTROL" = "1" ]; then
  echo "=== [1/5] random-split control ==="
  python3 -u "$SCRIPTS/run_random_split_control.py" "${TRAIN_FLAGS[@]}"
else
  echo "=== [1/5] random-split control — skipped (RUN_CONTROL=0) ==="
fi

# --- 2. Leave-one-family-out ----------------------------------------------
echo "=== [2/5] leave-one-family-out ==="
python3 -u "$SCRIPTS/run_leave_one_family_out.py" \
  --families "$FAMILIES" \
  --min-family-size "$MIN_FAMILY_SIZE" \
  "${TRAIN_FLAGS[@]}"

# --- 3. Size extrapolation ------------------------------------------------
echo "=== [3/5] size extrapolation ==="
python3 -u "$SCRIPTS/run_size_extrapolation.py" \
  --max-train-qubits "$MAX_TRAIN_QUBITS" \
  "${TRAIN_FLAGS[@]}"

# --- 4. Scheduling replay on held-out predictions -------------------------
echo "=== [4/5] scheduling replay (leave-one-family-out predictions) ==="
python3 -u "$SCRIPTS/run_scheduling_eval.py" \
  --results-dir "$RESULTS" \
  --fidelity-weights "$WEIGHTS" \
  --per-family \
  --min-family-size "$MIN_FAMILY_SIZE"

for n in ${MAX_TRAIN_QUBITS//,/ }; do
  pred="$RESULTS/size_extrapolation/qmax_${n}/predictions.json"
  if [ -f "$pred" ]; then
    echo "=== [4/5] scheduling replay (size extrapolation, qmax_${n}) ==="
    python3 -u "$SCRIPTS/run_scheduling_eval.py" \
      --results-dir "$RESULTS" \
      --predictions "$pred" \
      --tag "size_extrapolation_qmax_${n}" \
      --fidelity-weights "$WEIGHTS"
  fi
done

# --- 5. Tables and figures ------------------------------------------------
echo "=== [5/5] report ==="
python3 -u "$SCRIPTS/make_report.py" --results-dir "$RESULTS"

echo "=== Done. Artefacts in $RESULTS (report/ holds the LaTeX tables and figures). ==="
