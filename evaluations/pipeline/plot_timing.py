#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the timing measurements behind the reviewers' question: what does
exhaustive device selection actually cost, and does GNN-predicted selection
beat it?

  exhaustive path = compile + evaluate fidelity on all D backends
  GNN path        = encode + GNN inference + compile on the predicted-best backend

Works on a partial run: everything is computed from whatever circuits are
present in the results JSONs.

Figures written to <out-dir>:
  cost_vs_width.pdf        brute-force cost vs circuit width, by family
  cost_breakdown.pdf       compile vs fidelity-eval share, and per-backend spread
  blowup.pdf               compiled/high-level gate-count blow-up, by family
and, when --inf is given (the head-to-head the reviewers asked for):
  compile_vs_inference.pdf median cost of both paths vs width, with IQR bands
  speedup_vs_width.pdf     exhaustive/GNN speedup vs width, with the 1x crossover
  cost_composition.pdf     where the time goes in each path, per width

Run:
  python evaluations/pipeline/plot_timing.py
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
_RESULTS_DIR = _SCRIPT_DIR / "timing_results"
_DEFAULT_RESULTS = _RESULTS_DIR / "compilation_time_results_benchmark_30k.json"
_DEFAULT_INF = _RESULTS_DIR / "inference_time_results_benchmark_30k.json"

DEVICES = ["EQE1_Top", "EQE1_Bottom", "QExa20"]
DEVICE_COLORS = ["#4c72b0", "#dd8452", "#55a868"]

# Head-to-head palette. The repo's default blue/orange/green triple fails
# CVD separation (orange vs green: dE 6.6 deuteranopia, 5.2 protanopia, both
# under the dE>=8 floor), so the third slot is teal instead; blue/orange/teal
# clears every check with the widest margin. Series are also distinguished by
# line style and direct labels, so identity never rests on hue alone.
C_EXHAUSTIVE = "#4c72b0"   # blue
C_GNN = "#dd8452"          # orange
C_OVERHEAD = "#17becf"     # teal
C_EXHAUSTIVE_L = "#a8c0e0" # light blue  (fidelity-eval share)
C_GNN_L = "#f2c9ae"        # light orange (selected-device compile share)

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


def load_joined(comp_path: Path, inf_path: Path,
                keep: set[str] | None = None) -> list[dict]:
    """One row per circuit measured on BOTH sides of the comparison.

    Circuits present in only one file are dropped: the head-to-head is
    meaningless without both halves.  If `keep` is given, only those tags are
    retained, so the figures cover the same population as timing_analysis.py.
    """
    with open(comp_path, encoding="utf-8") as f:
        comp = json.load(f)
    with open(inf_path, encoding="utf-8") as f:
        inf = json.load(f)

    rows = []
    for tag, entry in inf.items():
        if keep is not None and tag not in keep:
            continue
        c = comp.get(tag)
        if c is None:
            continue
        per_backend = c[0]
        if len(per_backend) < len(DEVICES):
            continue
        pred = entry["predicted"]
        best = DEVICES[max(range(len(pred)), key=pred.__getitem__)]
        overhead = entry["encode_time"] + entry["inference_time"]
        selected_compile = per_backend[best]["compilation_time"]
        transpile = sum(e["compilation_time"] for e in per_backend.values())
        fid_eval = sum(e["fidelity_time"] for e in per_backend.values())
        exhaustive = transpile + fid_eval
        gnn_total = overhead + selected_compile
        rows.append({
            "tag": tag,
            "family": tag.split("/")[0],
            "qubits": entry["num_qubits"],
            "encode": entry["encode_time"],
            "inference": entry["inference_time"],
            "overhead": overhead,
            "selected_compile": selected_compile,
            "transpile": transpile,
            "fid_eval": fid_eval,
            "exhaustive": exhaustive,
            "gnn_total": gnn_total,
            "speedup": exhaustive / gnn_total,
        })
    return rows


def _band(rows, key, widths, scale=1e3):
    """Median and inter-quartile band of `key` per width (scale=1e3 -> ms)."""
    med, lo, hi = [], [], []
    for w in widths:
        v = np.array([r[key] for r in rows if r["qubits"] == w]) * scale
        med.append(np.median(v))
        lo.append(np.percentile(v, 25))
        hi.append(np.percentile(v, 75))
    return np.array(med), np.array(lo), np.array(hi)


def plot_compile_vs_inference(rows: list[dict], out_dir: Path) -> None:
    """Median cost of both selection paths vs width (one y-axis, log scale)."""
    widths = sorted({r["qubits"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, color, style, label in (
        ("exhaustive", C_EXHAUSTIVE, "-", f"Exhaustive (compile+score on {len(DEVICES)} QPUs)"),
        ("gnn_total", C_GNN, "--", "GNN (encode+infer+1 compile)"),
        ("overhead", C_OVERHEAD, ":", "GNN overhead alone"),
    ):
        med, lo, hi = _band(rows, key, widths)
        ax.plot(widths, med, style, color=color, lw=2, marker="o", ms=5, label=label)
        ax.fill_between(widths, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.annotate(f"{med[-1]:.0f} ms", (widths[-1], med[-1]), textcoords="offset points",
                    xytext=(8, 0), color=color, fontsize=12, va="center")

    ax.set_yscale("log")
    ax.set_xticks(widths[::2])
    ax.set_xlabel("circuit width (qubits)")
    ax.set_ylabel("per-circuit cost [ms]")
    ax.set_title("Device-selection cost: exhaustive vs GNN-predicted")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.margins(x=0.10)
    out = out_dir / "compile_vs_inference.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_speedup(rows: list[dict], out_dir: Path) -> None:
    """Speedup vs width, and the share of circuits the GNN path actually wins.

    Two panels rather than one chart with two y-scales: a speedup ratio and a
    percentage do not share an axis.
    """
    widths = sorted({r["qubits"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    med, lo, hi = _band(rows, "speedup", widths, scale=1.0)   # ratio, unitless
    ax.plot(widths, med, "-", color=C_GNN, lw=2, marker="o", ms=5)
    ax.fill_between(widths, lo, hi, color=C_GNN, alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="#555555", lw=1.2, ls="--")
    ax.annotate("break-even", (widths[0], 1.0), textcoords="offset points",
                xytext=(2, 5), fontsize=12, color="#555555")
    ax.set_xticks(widths[::2])
    ax.set_xlabel("circuit width (qubits)")
    ax.set_ylabel(r"speedup  (exhaustive / GNN)")
    ax.set_title("Speedup vs width (median, IQR band)")

    ax = axes[1]
    frac = [100 * np.mean([r["speedup"] > 1 for r in rows if r["qubits"] == w]) for w in widths]
    ax.bar(widths, frac, color=C_GNN, width=0.7)
    ax.axhline(50, color="#555555", lw=1.2, ls="--")
    ax.set_xticks(widths[::2])
    ax.set_ylim(0, 100)
    ax.set_xlabel("circuit width (qubits)")
    ax.set_ylabel("circuits faster with GNN [%]")
    ax.set_title("Share of circuits where the GNN path wins")

    out = out_dir / "speedup_vs_width.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_cost_composition(rows: list[dict], out_dir: Path) -> None:
    """Where the time goes in each path, per width (grouped stacked bars)."""
    widths = sorted({r["qubits"] for r in rows})
    x = np.arange(len(widths), dtype=float)
    w = 0.38

    def med(key):
        return np.array([np.median([r[key] for r in rows if r["qubits"] == q]) * 1e3
                         for q in widths])

    transpile, fid_eval = med("transpile"), med("fid_eval")
    overhead, sel = med("overhead"), med("selected_compile")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, transpile, w, color=C_EXHAUSTIVE, label="exhaustive: transpilation ×3")
    ax.bar(x - w / 2, fid_eval, w, bottom=transpile, color=C_EXHAUSTIVE_L,
           label="exhaustive: fidelity eval ×3")
    ax.bar(x + w / 2, overhead, w, color=C_GNN, label="GNN: encode + inference")
    ax.bar(x + w / 2, sel, w, bottom=overhead, color=C_GNN_L,
           label="GNN: transpilation ×1")

    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(q) for q in widths[::2]])
    ax.set_xlabel("circuit width (qubits)")
    ax.set_ylabel("median per-circuit cost [ms]")
    ax.set_title("Cost composition of the two selection paths")
    ax.legend(ncol=2, fontsize=12, framealpha=0.9)
    out = out_dir / "cost_composition.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


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
    # tick_labels=, not labels=: the latter was removed in matplotlib 3.9
    # (this project pins 3.10).
    bp = ax.boxplot(data, tick_labels=DEVICES, showfliers=False, patch_artist=True, widths=0.55)
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
    ap.add_argument("--inf", default=str(_DEFAULT_INF),
                    help="JSON from testComp-inference-time.py; head-to-head "
                         "figures are skipped if it is absent")
    ap.add_argument("--out-dir", default=None, help="default: <results dir>/plots")
    ap.add_argument("--test-names", default=None,
                    help="JSON list of test-split circuit tags; restricts the "
                         "head-to-head figures to that split so they match the "
                         "numbers timing_analysis.py reports")
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

    inf_path = Path(args.inf)
    if not inf_path.exists():
        print(f"[PLOT] {inf_path} not found; skipping head-to-head figures.")
        return
    keep = None
    if args.test_names:
        with open(args.test_names, encoding="utf-8") as f:
            keep = set(json.load(f))
    joined = load_joined(results, inf_path, keep)
    if not joined:
        print("[PLOT] no circuits measured on both sides; skipping head-to-head figures.")
        return
    scope = f"test split of {len(keep)}" if keep else "all circuits"
    print(f"[PLOT] head-to-head on {len(joined)} circuits measured on both sides ({scope})")
    plot_compile_vs_inference(joined, out_dir)
    plot_speedup(joined, out_dir)
    plot_cost_composition(joined, out_dir)


if __name__ == "__main__":
    main()
