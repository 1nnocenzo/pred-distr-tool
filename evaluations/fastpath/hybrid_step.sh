#!/bin/bash
# esp/hybrid.py --fast, only when its inputs exist (the models' predictions and checkpoints are not in git).
# Run from the repository root.  Env: F_RUN (v8kdc10), ESP_RUN (v8kdi10), GS_RESULTS_ROOT (generalization_v9).
F_RUN=${F_RUN:-v8kdc10}; ESP_RUN=${ESP_RUN:-v8kdi10}
R=evaluations/${GS_RESULTS_ROOT:-generalization_v9}/results
miss=()
ls $R/${F_RUN}*/*/*/predictions.json >/dev/null 2>&1 || miss+=("$R/${F_RUN}*/*/*/predictions.json")
ls $R/${ESP_RUN}*/*/*/model.pth >/dev/null 2>&1 || miss+=("$R/${ESP_RUN}*/*/*/model.pth")
for f in evaluations/compile_check/results/esp_labels.pt evaluations/compile_check/results/struct_labels_v2.pt \
         evaluations/fastpath/results/fastpath_pred.json; do
  [ -f "$f" ] || miss+=("$f")
done
if [ ${#miss[@]} -gt 0 ]; then
  echo "hybrid.py skipped, missing: ${miss[*]}"
  echo "copy them from the cluster (or run this step there): bash evaluations/fastpath/hybrid_step.sh"
  exit 0
fi
GS_TAG=_fast python3 -u evaluations/esp/hybrid.py --f-run "$F_RUN" --esp-run "$ESP_RUN" \
  --fast evaluations/fastpath/results/fastpath_pred.json
