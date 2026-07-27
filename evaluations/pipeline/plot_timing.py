#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the brute-force compilation-cost measurements from
testComp-compilation-time.py (the reviewer-facing "what does exhaustive
device selection actually cost" evidence).

Works on a partial run: everything is computed from whatever circuits are
present in the results JSON.

Figures written to <out-dir>:
  cost_vs_width.pdf   per-circuit brute-force cost vs circuit width, by family
  cost_breakdown.pdf  compile vs fidelity-eval share, and per-backend spread
  blowup.pdf          compiled/high-level gate-count blow-up, by family

Run:
  python evaluations/pipeline/plot_timing.py \
      --results evaluations/pipeline/timing_results/compilation_time_results_benchmark_30k.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _SCRIPT_DIR / "timing_results" / "compilation_time_results_benchmark_30k.json"

DEVICES = ["EQE1_Top", "EQE1_Bottom", "QExa20"]
DEVICE_COLORS = ["#4c72b0", "#dd8452", "#55a868"]

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "figure.titlesize": 17,
})

_QRE = re.compile(r"_q(\d+)_")


def load_rows(path: Path) -> list[dict]:
    """One row per circuit with complete per-backend data."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for tag, (per_backend, argmax_be, _argmax_fid) in raw.items():
        if len(per_backend) < len(DEVICES):
            continue
        m = _QRE.search(tag)
        any_e = next(iter(per_backend.values()))
        rows.append({
            "tag": tag,
            "family": tag.split("/")[0],
            "qubits": int(m.group(1)) if m else 0,
            "size_hl": any_e["size_hl"],
            "compile": {b: e["compilation_time"] for b, e in per_backend.items()},
            "fidelity_eval": {b: e["fidelity_time"] for b, e in per_backend.items()},
            "size": {b: e["size"] for b, e in per_backend.items()},
            "brute_force": sum(e["compilation_time"] + e["fidelity_time"] for e in per_backend.values()),
            "selected_compile": per_backend[argmax_be]["compilation_time"],
        })
    return rows


def _median_by(rows, keyf, valf):
    acc = defaultdict(list)
    for r in rows:
        acc[keyf(r)].append(valf(r))
    return {k: median(v) for k, v in sorted(acc.items())}, {k: len(v) for k, v in acc.items()}


def plot_cost_vs_width(rows: list[dict], out_dir: Path) -> None:
    """Brute-force cost vs width; one line per family (log y — costs span 1e-3..1e3 s)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    families = sorted({r["family"] for r in rows})
    cmap = plt.get_cmap("tab20")
    for i, fam in enumerate(families):
        sel = [r for r in rows if r["family"] == fam]
        med, _ = _median_by(sel, lambda r: r["qubits"], lambda r: r["brute_force"])
        if len(med) < 2:
            ax.scatter(list(med), list(med.values()), s=28, color=cmap(i % 20), label=fam, zorder=3)
        else:
            ax.plot(list(med), list(med.values()), marker="o", ms=4,
                    color=cmap(i % 20), label=fam)
    ax.set_yscale("log")
    ax.set_xlabel("circuit width (qubits)")
    ax.set_ylabel("brute-force cost / circuit [s]")
    ax.set_title(r"Exhaustive device selection: compile + score on all $D{=}3$ backends")
    ax.legend(ncol=2, fontsize=10, framealpha=0.9)
    out = out_dir / "cost_vs_width.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_breakdown(rows: list[dict], out_dir: Path) -> None:
    """Left: compile vs fidelity-eval share by width. Right: per-backend compile spread."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    widths = sorted({r["qubits"] for r in rows})
    comp = [median([sum(r["compile"].values()) for r in rows if r["qubits"] == w]) for w in widths]
    fid = [median([sum(r["fidelity_eval"].values()) for r in rows if r["qubits"] == w]) for w in widths]
    ax.bar(widths, comp, color="#4c72b0", label="transpilation")
    ax.bar(widths, fid, bottom=comp, color="#dd8452", label="fidelity evaluation")
    ax.set_yscale("log")
    ax.set_xlabel("circuit width (qubits)")
    ax.set_ylabel("median cost [s]")
    ax.set_title("Cost breakdown (all backends)")
    ax.legend()

    ax = axes[1]
    data = [[r["compile"][b] for r in rows] for b in DEVICES]
    bp = ax.boxplot(data, labels=DEVICES, showfliers=False, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], DEVICE_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_yscale("log")
    ax.set_ylabel("compilation time [s]")
    ax.set_title("Per-backend compilation time")
    ax.tick_params(axis="x", labelsize=12)

    out = out_dir / "cost_breakdown.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_blowup(rows: list[dict], out_dir: Path) -> None:
    """Compiled/high-level gate-count ratio per family — explains the cost spread."""
    fams, vals = [], []
    for fam in sorted({r["family"] for r in rows}):
        sel = [r for r in rows if r["family"] == fam]
        fams.append(f"{fam} ({len(sel)})")
        vals.append(median([median(list(r["size"].values())) / max(r["size_hl"], 1) for r in sel]))
    order = np.argsort(vals)
    fams = [fams[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.38 * len(fams))))
    ax.barh(fams, vals, color="#55a868")
    ax.set_xscale("log")
    ax.set_xlabel("compiled / high-level gate count")
    ax.set_title("Transpilation blow-up by benchmark family")
    ax.tick_params(axis="y", labelsize=11)
    out = out_dir / "blowup.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results", default=str(_DEFAULT_RESULTS))
    ap.add_argument("--out-dir", default=None, help="default: <results dir>/plots")
    args = ap.parse_args()

    results = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results)
    if not rows:
        raise SystemExit(f"No complete entries in {results}")
    print(f"[PLOT] {len(rows)} circuits, {len({r['family'] for r in rows})} families "
          f"-> {out_dir}")
    plot_cost_vs_width(rows, out_dir)
    plot_breakdown(rows, out_dir)
    plot_blowup(rows, out_dir)


if __name__ == "__main__":
    main()
