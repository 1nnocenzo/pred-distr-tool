#!/bin/bash
# Pipeline evaluation (switch -> exact compile or light-Sabre compile) on any machine, CPU only.
#   JOBS=8 N=2000 bash run_local.sh            # eval tasks + report -> results/report.md
#   DATA_DIR=... (default ../../data/benchmark_dataset_30k, unpacked from data/circuits.zip when missing)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=../../src RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1      # one thread per task: times comparable
JOBS=${JOBS:-4}; N=${N:-400}; DATA_DIR=${DATA_DIR:-../../data/benchmark_dataset_30k}
[ -d "$DATA_DIR" ] || (cd ../../data && unzip -q circuits.zip)
mkdir -p local_logs
pids=()
for i in $(seq 0 $((JOBS - 1))); do
    python3 -u light_compile.py eval --task-id $i --n-tasks $JOBS --n $N --data-dir "$DATA_DIR" > local_logs/eval_$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python3 -u light_compile.py report | tee local_logs/report.log
