#!/usr/bin/env python3
"""End-to-end ESP models side by side: results/esp_eval_<run>.json (esp_eval.py) + each source version's own F
(results/report_esp_versions.json, "F only").
    python esp_e2e_report.py      # -> results/report_esp_e2e.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPLITS = ["LOFO qaoa", "LOFO qnn", "LOGO variational", "control"]
# run: (label, source version for the F comparison, how it was trained)
RUNS = {
    "v5_espa": ("v5 + ESP", "v5_sinkhorn", "v5 checkpoint, 40 ep"),
    "v8a_espa": ("v8a + ESP", "v8a_c001", "v8a checkpoint, 40 ep"),
    "v8kd40_espa": ("v8kd40 + ESP", "v8kd40", "v8kd40 checkpoint, 40 ep"),
    "v8kdl40_espa": ("v8kdl40 + ESP", "v8kdl40", "v8kdl40 checkpoint, 40 ep"),
    "v8kdc10_espa": ("v8kdc10 + ESP", "v8kdc10", "v8kdc10 checkpoint, 40 ep"),
    "v8kde10": ("v8kde10 (v13 head)", "v8kdc10", "v8k checkpoint, v8kdc10 recipe"),
    "v8kda10": ("v8kda10 (analytic)", "v8kdc10", "v8k checkpoint, v8kdc10 recipe"),
    "v8kds10": ("v8kds10 (+ structure loss)", "v8kdc10", "v8k checkpoint, v8kdc10 recipe"),
}
KEYS = ["R² F", "R² log depth", "depth error (median)", "R² log duration", "duration error (median)", "R² ESP",
        "right device by ESP, model", "ESP regret, model", "ESP regret, baseline F"]


def main():
    probe = json.loads((HERE / "results/report_esp_versions.json").read_text())
    src_f = {}
    for r in probe:
        src_f.setdefault((r["run"], r["split"]), []).append(r["F only"]["R² F"])
    res = {}
    for run in RUNS:
        p = HERE / f"results/esp_eval_{run}.json"
        if p.is_file():
            res[run] = json.loads(p.read_text())["res"]
    L = ["# End-to-end ESP models", "", "Mean ± std over seeds (3 each). Error = median |pred/true − 1|. "
         "R² F of the source version = the same version before the ESP fine-tuning.", ""]
    for k in KEYS + ["R² F, source version"]:
        L += [f"## {k}", "", "| model | trained from | " + " | ".join(SPLITS) + " |", "| --- | --- |" + " --- |" * len(SPLITS)]
        for run, (lab, src, how) in RUNS.items():
            if run not in res:
                continue
            cells = []
            for s in SPLITS:
                if k == "R² F, source version":
                    v = np.array(src_f.get((src, s), []))
                else:
                    v = np.array([m[k] for m in res[run].get(s, []) if k in m])
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else (f"{v.mean():.3f}" if len(v) else ""))
            L.append(f"| {lab} | {how} | " + " | ".join(cells) + " |")
        L.append("")
    (HERE / "results/report_esp_e2e.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
