#!/usr/bin/env bash
# The whole pipeline on a few circuits, on CPU, in its own work directory (~10-15 min):
#   labels (2 circuits per family + 30 synthetic, original calibration + 2 variants) -> dataset -> train 2 epochs -> eval
# Usage: bash scripts/smoke.sh        (WORK_DIR=... to change where it writes; default work_smoke/)
set -euo pipefail
cd "$(dirname "$0")/.."
export ESPTIME_WORK=${WORK_DIR:-$PWD/work_smoke}
PY=${PY:-python}
rm -rf "$ESPTIME_WORK"
$PY -m esptime.synth --n-generic 20 --n-layered 10
$PY -m esptime.labels run --source mqt --per-family 2 --calibs orig,aug0-1
$PY -m esptime.labels run --source synth --calibs orig,aug0-1
$PY -m esptime.labels merge
$PY -m esptime.dataset --jobs 4
$PY -m esptime.train --split control --epochs 2 --calib-every 2 --mc 8 --time-n 10 --threads 4
echo "smoke OK: $ESPTIME_WORK/runs/control_s5/metrics.json"
