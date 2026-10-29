#!/usr/bin/env bash
# The whole pipeline on one machine, without SLURM.  Every step can be re-run: finished work is skipped.
#
#   bash scripts/run_local.sh test       # unit test (~15 s): must print "all checks passed"
#   bash scripts/run_local.sh smoke      # whole pipeline on ~90 circuits in work_smoke/ (~10 min)
#   bash scripts/run_local.sh labels     # synthetic circuits + one compilation per (circuit, calibration, device)
#   bash scripts/run_local.sh dataset    # merge labels -> work/labels.pt, devices.json -> work/dataset.pt
#   bash scripts/run_local.sh train      # SPLITS x SEEDS trainings, each with its evaluation and timing
#   bash scripts/run_local.sh report     # table of every work/runs/*/metrics.json -> work/runs/summary.md
#   bash scripts/run_local.sh all        # labels, dataset, train, report
#
# Settings (environment variables):
#   JOBS=16            parallel label / dataset processes (default: number of cores)
#   CALIBS=orig,aug0-19  calibrations to compile (variants train the change loss; orig alone is ~20x cheaper)
#   SPLITS="control lofo:qaoa lofo:qnn logo:variational"   SEEDS="5 6 7"
#   TRAIN_JOBS=2       trainings in parallel (each holds the dataset in memory)
#   DEVICE=cuda GPUS="0 1"   GPUs, assigned round robin to the trainings (default: CPU)
#   THREADS=8          torch threads per training      EPOCHS=40
#   TRAIN_ARGS="--calib-every 4"   extra arguments of esptime.train (e.g. "--esp-busy counts", "--max-nodes 100000")
#   ESPTIME_WORK=/path  where generated data goes (default: work/ in the repository)
# Logs: $ESPTIME_WORK/logs/
set -euo pipefail
cd "$(dirname "$0")/.."
export ESPTIME_WORK=${ESPTIME_WORK:-$PWD/work}
PY=${PY:-python}
JOBS=${JOBS:-$(nproc)}
CALIBS=${CALIBS:-orig,aug0-19}
SPLITS=${SPLITS:-"control lofo:qaoa lofo:qnn logo:variational"}
SEEDS=${SEEDS:-"5 6 7"}
TRAIN_JOBS=${TRAIN_JOBS:-2}
DEVICE=${DEVICE:-cpu}
GPUS=(${GPUS:-0})
THREADS=${THREADS:-8}
EPOCHS=${EPOCHS:-40}
TRAIN_ARGS=${TRAIN_ARGS:---calib-every 4}
LOG=$ESPTIME_WORK/logs
mkdir -p "$LOG"

labels() {
  [ -n "$(ls "$ESPTIME_WORK"/circuits/synth/*.qasm 2>/dev/null | head -1)" ] || $PY -m esptime.synth > "$LOG/synth.log" 2>&1
  for src in mqt synth; do
    echo "labels $src: $JOBS processes, calibrations $CALIBS (logs $LOG/labels_${src}_*.log)"
    for t in $(seq 0 $((JOBS - 1))); do
      OMP_NUM_THREADS=1 $PY -m esptime.labels run --source $src --calibs "$CALIBS" --task-id $t --n-tasks $JOBS \
        > "$LOG/labels_${src}_$t.log" 2>&1 &
    done
    wait
  done
  { grep -h "error" "$ESPTIME_WORK"/labels/*.jsonl 2>/dev/null || true; } | wc -l | xargs echo "circuits with an error:"   # no match = exit 1 under pipefail
}

dataset() {
  $PY -m esptime.labels merge 2>&1 | tee "$LOG/merge.log"
  $PY -m esptime.dataset --jobs "$JOBS" 2>&1 | tee "$LOG/dataset.log"
}

train() {
  local i=0
  for split in $SPLITS; do
    for seed in $SEEDS; do
      local name="${split//:/_}_s$seed"
      if [ -f "$ESPTIME_WORK/runs/$name/metrics.json" ]; then echo "skip $name (done)"; continue; fi
      local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      echo "train $name on $DEVICE$([ "$DEVICE" = cuda ] && echo " GPU $gpu") (log $LOG/train_$name.log)"
      CUDA_VISIBLE_DEVICES=$([ "$DEVICE" = cuda ] && echo "$gpu" || echo "") \
        $PY -m esptime.train --split "$split" --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" --threads "$THREADS" \
        $TRAIN_ARGS --out "$ESPTIME_WORK/runs/$name" > "$LOG/train_$name.log" 2>&1 &
      i=$((i + 1))
      while [ "$(jobs -rp | wc -l)" -ge "$TRAIN_JOBS" ]; do sleep 30; done
    done
  done
  wait
  grep -l Traceback "$LOG"/train_*.log 2>/dev/null && echo "^ trainings that failed" || echo "all trainings finished"
}

report() {
  $PY - <<'EOF'
import json, os, glob
rows = []
for f in sorted(glob.glob(os.path.join(os.environ["ESPTIME_WORK"], "runs", "*", "metrics.json"))):
    m = json.load(open(f)); d = m.get("duration (dur vs label, relative error)", {}); e = m.get("ESP", {}); t = m.get("timing", {}).get("by_2q_ms", {})
    g = lambda x, k: x.get(k, {}).get("median", float("nan"))
    rows.append(f"| {os.path.basename(os.path.dirname(f))} | {g(d, 'routed'):.3f} | {g(d, 'routing-free'):.3f} | "
                f"{e.get('abs_dlog_median_esp_gt_1e-3') or float('nan'):.3f} | {e.get('abs_dlog_median_esp_gt_1e-3_routed') or float('nan'):.3f} | "
                f"{e.get('abs_dlog_median_esp_gt_1e-3_routing_free') or float('nan'):.3f} | {m.get('F', {}).get('R2', float('nan')):.3f} | "
                f"{m.get('device_by_ESP', {}).get('right', float('nan')):.2f} | {m.get('device_by_ESP', {}).get('regret', float('nan')):.4f} | "
                + " / ".join(f"{v['total_ms']:.0f}" for v in t.values()) + " |")
head = ["| run | duration routed | duration routing-free | ESP all | ESP routed | ESP routing-free | R2 F | device right | regret | ms per circuit by 2q bin |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
out = "\n".join(head + rows)
open(os.path.join(os.environ["ESPTIME_WORK"], "runs", "summary.md"), "w").write(out + "\n")
print(out)
EOF
}

case "${1:-}" in
  test)    $PY tests/test_model.py ;;
  smoke)   PY="$PY" bash scripts/smoke.sh ;;
  labels)  labels ;;
  dataset) dataset ;;
  train)   train ;;
  report)  report ;;
  all)     labels; dataset; train; report ;;
  *)       sed -n '2,24p' "$0"; exit 1 ;;
esac
