#!/bin/bash
# One training of the retraining plan (retrain_all.py): environment of train_gpu.sh, then the given command.
# Used inside run_packed.sh, which sets CUDA_VISIBLE_DEVICES per line.
set -uo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO_DIR"
echo "=== retrain $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-none}: $*"
"$@"; rc=$?
echo "=== done ($rc)"
exit $rc
