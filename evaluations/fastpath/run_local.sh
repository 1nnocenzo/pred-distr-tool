#!/usr/bin/env bash
# Fast-path study on a machine without SLURM, with the pred-distr env active (qiskit 2.3.1, rustworkx, mqt.predictor):
#   bash evaluations/fastpath/run_local.sh            # validate + predict in parallel, then report, merge, hybrid
# Env: JOBS (parallel processes, 8), DATA_DIR (QASM circuits; default ~/compileCircuits/benchmark_dataset_30k,
#      else data/benchmark_dataset_30k, unpacked from data/circuits.zip if missing), N_PER_SPLIT (validation circuits per split, 60).
# Without the models' predictions.json the circuits come from esp_labels.pt, split by family (run_fastpath.py --circuits).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
JOBS=${JOBS:-8}
ARGS=(--n-per-split "${N_PER_SPLIT:-60}")
[ -n "${DATA_DIR:-}" ] && ARGS+=(--data-dir "$DATA_DIR")
if [ -z "${DATA_DIR:-}" ] && [ ! -d "$HOME/compileCircuits/benchmark_dataset_30k" ] && [ ! -d data/benchmark_dataset_30k ]; then
  echo "unpacking data/circuits.zip -> data/benchmark_dataset_30k"
  python3 -c "import zipfile; zipfile.ZipFile('data/circuits.zip').extractall('data')"
fi
LOGS=evaluations/fastpath/local_logs
mkdir -p "$LOGS"
for mode in validate predict; do
  pids=()
  for i in $(seq 0 $((JOBS - 1))); do
    python3 -u evaluations/fastpath/run_fastpath.py $mode --task-id "$i" --n-tasks "$JOBS" "${ARGS[@]}" \
      > "$LOGS/${mode}_$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "$mode done (failures: $fail; logs in $LOGS)"
done
python3 -u evaluations/fastpath/run_fastpath.py report > "$LOGS/report.log" 2>&1 && echo "report: evaluations/fastpath/results/report_validate.md"
python3 -u evaluations/fastpath/run_fastpath.py merge 2>&1 | tee "$LOGS/merge.log"
bash evaluations/fastpath/hybrid_step.sh 2>&1 | tee "$LOGS/hybrid.log"
