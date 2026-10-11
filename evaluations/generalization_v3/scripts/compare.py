#!/usr/bin/env python3
"""Compare v1, v2 and every v3 run split by split (R², MAE, regret), from metrics.json files.

Prints a table and writes ``results/comparison.csv``.  Works on partial results:
splits not finished yet are shown as ``-``.

    python evaluations/generalization_v3/scripts/compare.py [--metric r2]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from gsv3 import paths

EXP_DIRS = ("random_control", "leave_one_family_out", "size_extrapolation")


def collect(root: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for exp in EXP_DIRS:
        for m in sorted((root / exp).glob("*/metrics.json")):
            out[(exp, m.parent.name)] = json.loads(m.read_text())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--metric", default="r2",
                   help="test metric to tabulate: r2, mae, fidelity_regret_mean, ...")
    args = p.parse_args()

    sources = {"v1": paths.V1_STUDY_DIR / "results", "v2": paths.V2_RESULTS_DIR}
    for run in sorted(d for d in paths.RESULTS_DIR.iterdir()
                      if d.is_dir() and d.name != "slurm"):
        sources[run.name] = run
    tables = {name: collect(root) for name, root in sources.items()}
    keys = sorted({k for t in tables.values() for k in t},
                  key=lambda k: (EXP_DIRS.index(k[0]), k[1]))

    names = list(tables)
    width = max(12, *(len(n) for n in names))
    print(f"test {args.metric}")
    print(f"{'split':24}" + "".join(f"{n:>{width + 1}}" for n in names))
    rows = []
    for exp, split in keys:
        vals = []
        for n in names:
            m = tables[n].get((exp, split))
            vals.append(m["test"].get(args.metric) if m else None)
        print(f"{split:24}" + "".join(
            f"{v:>{width + 1}.4f}" if v is not None else f"{'-':>{width + 1}}" for v in vals))
        rows.append({"experiment": exp, "split": split, **dict(zip(names, vals))})

    out = paths.RESULTS_DIR / f"comparison_{args.metric}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "split", *names])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
