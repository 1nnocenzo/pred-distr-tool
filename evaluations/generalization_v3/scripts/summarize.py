#!/usr/bin/env python3
"""Seed-aggregated comparison of every model on every held-out split.

Scans the v1 study (seed 5 only), the v3 results and the v4 results.  A run named
``<config>_s<seed>`` is a seed replicate of ``<config>``; a run without the
suffix is seed 5 (``v1_s5`` is the v1 model run through the v3 runner).  For each
(experiment, split) and config it reports the mean, the standard deviation and the
number of seeds, for R², MAE and device-choice regret, and for LOGO also per family
inside the group.

    python summarize.py [PLAN ...]     (plan arguments are accepted and ignored)

Writes ``generalization_v4/results/summary_<metric>.csv`` and prints the tables.
Only reads JSON files: safe to run on the login node.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # evaluations/
SOURCES = [ROOT / "generalization_v3" / "results", ROOT / "generalization_v4" / "results",
           ROOT / "generalization_v5" / "results"]
V1_RESULTS = ROOT / "generalization" / "results"
EXP_DIRS = {"random_control": "control", "leave_one_family_out": "lofo",
            "size_extrapolation": "size", "leave_one_group_out": "logo"}
METRICS = ("r2", "mae", "fidelity_regret_mean")
SKIP_RUNS = {"xattn_a05_logsel"}
ORDER = ["v1", "pooled_stab_a00", "pooled_stab_a05", "xattn_a00", "xattn_a05",
         "v4_xattn_mixed", "v4_phys", "v5_sinkhorn", "v5b_sinkhorn"]


def config_and_seed(run: str) -> tuple[str, int]:
    m = re.fullmatch(r"(.+)_s(\d+)", run)
    return (m.group(1), int(m.group(2))) if m else (run, 5)


def collect():
    # values[(exp, split, family|None)][config][seed] = {metric: value}
    values = defaultdict(lambda: defaultdict(dict))

    def add(exp, split, config, seed, metrics_json):
        values[(exp, split, None)][config][seed] = metrics_json["test"]
        for fam, m in metrics_json.get("test_by_family", {}).items():
            if exp == "logo":
                values[(exp, split, fam)][config][seed] = m

    for exp_dir, exp in EXP_DIRS.items():
        for m in (V1_RESULTS / exp_dir).glob("*/metrics.json"):
            add(exp, m.parent.name, "v1", 5, json.loads(m.read_text()))
    for root in SOURCES:
        if not root.is_dir():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if run_dir.name in SKIP_RUNS or run_dir.name.startswith(("_", "slurm")):
                continue
            config, seed = config_and_seed(run_dir.name)
            for exp_dir, exp in EXP_DIRS.items():
                for m in (run_dir / exp_dir).glob("*/metrics.json"):
                    add(exp, m.parent.name, config, seed, json.loads(m.read_text()))
    return values


def fmt(vals: list[float]) -> str:
    if not vals:
        return "-"
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{statistics.stdev(vals):.3f}({len(vals)})"


def main() -> int:
    values = collect()
    configs = sorted({c for per in values.values() for c in per},
                     key=lambda c: (ORDER.index(c) if c in ORDER else len(ORDER), c))
    out_dir = ROOT / "generalization_v4" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_order = ["control", "lofo", "logo", "size"]
    keys = sorted(values, key=lambda k: (exp_order.index(k[0]), k[1], k[2] or ""))
    for metric in METRICS:
        print(f"\n=== test {metric}: mean±std(n seeds) ===")
        print(f"{'exp':6}{'split':16}{'family':16}" + "".join(f"{c:>24}" for c in configs))
        rows = []
        for key in keys:
            exp, split, fam = key
            cells = []
            row = {"experiment": exp, "split": split, "family": fam or ""}
            for c in configs:
                vals = [v[metric] for v in values[key].get(c, {}).values()
                        if v.get(metric) is not None]
                cells.append(fmt(vals))
                row[c] = statistics.fmean(vals) if vals else ""
                row[f"{c}_std"] = statistics.stdev(vals) if len(vals) > 1 else ""
                row[f"{c}_n"] = len(vals)
            print(f"{exp:6}{split:16}{(fam or ''):16}" + "".join(f"{x:>24}" for x in cells))
            rows.append(row)
        with open(out_dir / f"summary_{metric}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nwrote {out_dir}/summary_<metric>.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
