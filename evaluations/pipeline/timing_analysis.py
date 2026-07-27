#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Join the outputs of testComp-compilation-time.py (brute-force baseline:
per-backend compile + fidelity-evaluation wall-clock) and
testComp-inference-time.py (GNN path: DAG encoding + forward pass) into the
timing numbers the reviewers asked for:

  - per-circuit cost of exhaustive device selection (compile + score on all
    D backends) vs. the GNN path (encode + inference + compile only on the
    predicted-best backend), overall and binned by circuit width and depth;
  - training-label generation cost (exhaustive cost summed over the training
    split) and the break-even number of dispatched circuits after which that
    one-off cost is amortized by the per-circuit saving.

Stdlib only. Prints a report and writes a CSV of the binned tables.

Run:
  python evaluations/pipeline/timing_analysis.py \
      --comp compilation_time_results_benchmark_30k.json \
      --inf  inference_time_results_benchmark_30k.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

# Model output order (matches gnn_device_policy.DEVICE_NAMES; hardcoded here
# to keep this script free of the torch import chain).
DEVICE_NAMES = ["EQE1_Top", "EQE1_Bottom", "QExa20"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = Path(__file__).resolve().parent / "timing_results"
DEFAULT_TEST_NAMES = _REPO_ROOT / "src" / "model" / "test_circuit_names.json"
DEFAULT_COMP = _RESULTS_DIR / "compilation_time_results_benchmark_30k.json"
DEFAULT_INF = _RESULTS_DIR / "inference_time_results_benchmark_30k.json"


# -------------------------------
# Core computation
# -------------------------------

def exhaustive_cost(per_backend: dict) -> float:
    """Brute-force cost of one circuit: compile + fidelity eval on all backends."""
    return sum(d["compilation_time"] + d["fidelity_time"] for d in per_backend.values())


def build_rows(comp: dict, inf: dict) -> tuple[list[dict], int]:
    """Per-circuit joined cost rows; circuits missing a backend are dropped
    (their exhaustive cost would be understated). Returns (rows, n_dropped)."""
    rows, dropped = [], 0
    for tag, entry in inf.items():
        c = comp.get(tag)
        if c is None:
            continue
        per_backend = c[0]
        if len(per_backend) < len(DEVICE_NAMES):
            dropped += 1
            continue
        pred = entry["predicted"]
        pred_best = DEVICE_NAMES[max(range(len(pred)), key=pred.__getitem__)]
        inference = entry["encode_time"] + entry["inference_time"]
        gnn_total = inference + per_backend[pred_best]["compilation_time"]
        exhaustive = exhaustive_cost(per_backend)
        rows.append({
            "tag": tag,
            "qubits": entry["num_qubits"],
            "depth_hl": next(iter(per_backend.values()))["depth_hl"],
            "exhaustive": exhaustive,
            "inference": inference,
            "gnn_total": gnn_total,
            "speedup": exhaustive / gnn_total,
        })
    return rows, dropped


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "exhaustive_mean": mean(r["exhaustive"] for r in rows),
        "exhaustive_median": median(r["exhaustive"] for r in rows),
        "inference_mean": mean(r["inference"] for r in rows),
        "inference_median": median(r["inference"] for r in rows),
        "gnn_mean": mean(r["gnn_total"] for r in rows),
        "gnn_median": median(r["gnn_total"] for r in rows),
        "speedup_mean": mean(r["speedup"] for r in rows),
        "speedup_median": median(r["speedup"] for r in rows),
    }


def depth_bin_edges(rows: list[dict]) -> list[int]:
    """Quartile edges over high-level depth."""
    depths = sorted(r["depth_hl"] for r in rows)
    return [depths[len(depths) * q // 4] for q in (1, 2, 3)]


def bin_rows(rows: list[dict]) -> list[tuple[str, str, dict]]:
    """(dimension, bin label, summary) for width and depth-quartile bins."""
    out = []
    for q in sorted({r["qubits"] for r in rows}):
        out.append(("width", f"q={q}", summarize([r for r in rows if r["qubits"] == q])))
    e1, e2, e3 = depth_bin_edges(rows)
    depth_bins = [
        (f"depth<={e1}", lambda r: r["depth_hl"] <= e1),
        (f"{e1}<depth<={e2}", lambda r: e1 < r["depth_hl"] <= e2),
        (f"{e2}<depth<={e3}", lambda r: e2 < r["depth_hl"] <= e3),
        (f"depth>{e3}", lambda r: r["depth_hl"] > e3),
    ]
    for label, pred in depth_bins:
        sel = [r for r in rows if pred(r)]
        if sel:
            out.append(("depth", label, summarize(sel)))
    return out


# -------------------------------
# Reporting
# -------------------------------

def _fmt(s: dict) -> str:
    return (
        f"n={s['n']:>6}  exhaustive={s['exhaustive_median'] * 1e3:8.2f}ms  "
        f"inference={s['inference_median'] * 1e3:7.2f}ms  "
        f"gnn+compile={s['gnn_median'] * 1e3:8.2f}ms  "
        f"speedup(med)={s['speedup_median']:6.2f}x  speedup(mean)={s['speedup_mean']:6.2f}x"
    )


def report(comp: dict, inf: dict, test_names: set[str], csv_path: Path) -> None:
    rows, dropped = build_rows(comp, inf)
    print(f"[JOIN] comp={len(comp)} inf={len(inf)} joined={len(rows)} dropped(incomplete)={dropped}")

    test_rows = [r for r in rows if r["tag"] in test_names]
    eval_rows = test_rows if test_rows else rows
    split = "test split" if test_rows else "all joined circuits"

    print(f"\n=== Per-circuit cost ({split}, medians unless noted) ===")
    print(_fmt(summarize(eval_rows)))

    binned = bin_rows(eval_rows)
    print("\n=== By circuit width ===")
    for dim, label, s in binned:
        if dim == "width":
            print(f"{label:>6}  {_fmt(s)}")
    print("\n=== By high-level depth (quartiles) ===")
    for dim, label, s in binned:
        if dim == "depth":
            print(f"{label:>20}  {_fmt(s)}")

    # Amortization: one-off label-generation cost vs per-circuit saving.
    # Uses the comp JSON alone for the training side (no inference needed to
    # generate labels); GNN training time itself is not included here.
    train_costs = [
        exhaustive_cost(c[0]) for tag, c in comp.items()
        if tag not in test_names and len(c[0]) == len(DEVICE_NAMES)
    ]
    label_cost = sum(train_costs)
    saving = mean(r["exhaustive"] - r["gnn_total"] for r in eval_rows)
    print("\n=== Amortization ===")
    print(f"training circuits: {len(train_costs)}  label-generation cost: {label_cost:.1f}s "
          f"({label_cost / 3600:.2f}h)")
    print(f"mean saving per dispatched circuit ({split}): {saving * 1e3:.2f}ms")
    if saving > 0:
        print(f"break-even: {label_cost / saving:,.0f} dispatched circuits")
    else:
        print("break-even: never (GNN path is not cheaper on this split)")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        header = ["dimension", "bin", "n",
                  "exhaustive_mean_s", "exhaustive_median_s",
                  "inference_mean_s", "inference_median_s",
                  "gnn_mean_s", "gnn_median_s",
                  "speedup_mean", "speedup_median"]
        wr.writerow(header)
        for dim, label, s in [("overall", split, summarize(eval_rows))] + binned:
            wr.writerow([dim, label, s["n"],
                         s["exhaustive_mean"], s["exhaustive_median"],
                         s["inference_mean"], s["inference_median"],
                         s["gnn_mean"], s["gnn_median"],
                         s["speedup_mean"], s["speedup_median"]])
    print(f"\n[DONE] Wrote '{csv_path}'")


# -------------------------------
# Self-check
# -------------------------------

def self_test() -> None:
    be = lambda ct, ft: {"compilation_time": ct, "fidelity_time": ft, "depth_hl": 10}
    comp = {
        "fam/a": [{"EQE1_Top": be(0.10, 0.01), "EQE1_Bottom": be(0.20, 0.01),
                   "QExa20": be(0.30, 0.01)}, "EQE1_Top", 0.9],
        "fam/train": [{"EQE1_Top": be(1.0, 0.0), "EQE1_Bottom": be(1.0, 0.0),
                       "QExa20": be(1.0, 0.0)}, "EQE1_Top", 0.9],
        "fam/partial": [{"EQE1_Top": be(0.1, 0.0)}, "EQE1_Top", 0.9],
    }
    inf = {
        "fam/a": {"predicted": [0.1, 0.9, 0.2], "num_qubits": 5,
                  "encode_time": 0.005, "inference_time": 0.005},
        "fam/partial": {"predicted": [0.9, 0.1, 0.1], "num_qubits": 5,
                        "encode_time": 0.005, "inference_time": 0.005},
    }
    rows, dropped = build_rows(comp, inf)
    assert dropped == 1 and len(rows) == 1
    r = rows[0]
    assert abs(r["exhaustive"] - 0.63) < 1e-12          # sum over 3 backends
    assert abs(r["gnn_total"] - (0.01 + 0.20)) < 1e-12  # predicted best = EQE1_Bottom
    assert abs(r["speedup"] - 0.63 / 0.21) < 1e-9
    label = sum(exhaustive_cost(c[0]) for t, c in comp.items()
                if t not in {"fam/a"} and len(c[0]) == 3)
    assert abs(label - 3.0) < 1e-12                     # only fam/train counts
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--comp", default=str(DEFAULT_COMP),
                    help="JSON from testComp-compilation-time.py")
    ap.add_argument("--inf", default=str(DEFAULT_INF),
                    help="JSON from testComp-inference-time.py")
    ap.add_argument("--test-names", default=str(DEFAULT_TEST_NAMES),
                    help="JSON list of test-split circuit tags")
    ap.add_argument("--csv", default="timing_analysis_summary.csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    with open(args.comp, encoding="utf-8") as f:
        comp = json.load(f)
    with open(args.inf, encoding="utf-8") as f:
        inf = json.load(f)
    test_names: set[str] = set()
    if Path(args.test_names).exists():
        with open(args.test_names, encoding="utf-8") as f:
            test_names = set(json.load(f))
    else:
        print(f"[WARN] test-names file not found ({args.test_names}); "
              "reporting on all circuits, amortization uses full comp set as training.")

    report(comp, inf, test_names, Path(args.csv))


if __name__ == "__main__":
    main()
