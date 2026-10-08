#!/bin/bash
# Several v3 configurations in parallel on ONE working GPU (the models are small:
# < 1M parameters, < 4 GB peak activations, so they share an H200 easily).
# Requests 2 GPUs so that a broken one assigned by SLURM can be skipped.
#
#     RUNS="pooled_stab_a05:pooled:0.5 xattn_a00:xattn:0" \
#       sbatch evaluations/generalization_v3/slurm/train_multi.sh
#
# Each entry is RUN:MODEL:ALPHA.  Results go to results/<RUN>/, logs to
# results/slurm/multi_<jobid>_<RUN>.log.  Idempotent like train.sh.

#SBATCH --job-name=gen_v3_multi
#SBATCH --output=evaluations/generalization_v3/results/slurm/multi_%j.log
#SBATCH --error=evaluations/generalization_v3/results/slurm/multi_%j.err
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=4:00:00

set -uo pipefail
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

source evaluations/generalization_v3/slurm/pick_gpu.sh
pick_working_gpu || exit 3

RUNS="${RUNS:?set RUNS='run:model:alpha ...'}"
LR="${LR:-3e-4}"
CLIP="${CLIP:-1.0}"
EXPERIMENTS="${EXPERIMENTS:-control lofo size}"
LOGDIR="evaluations/generalization_v3/results/slurm"
n_runs=$(wc -w <<<"$RUNS")
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-24} / n_runs ))

pids=()
for entry in $RUNS; do
  IFS=: read -r run model alpha <<<"$entry"
  log="$LOGDIR/multi_${SLURM_JOB_ID:-local}_${run}.log"
  echo "=== $run: model=$model alpha=$alpha -> $log"
  (
    for exp in $EXPERIMENTS; do
      python3 -u evaluations/generalization_v3/scripts/run_experiment.py "$exp" \
        --model "$model" --balance-alpha "$alpha" --lr "$LR" --grad-clip "$CLIP" \
        --results-dir "evaluations/generalization_v3/results/$run" || exit 1
    done
  ) >"$log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "=== Done (status $status) ==="
exit $status
