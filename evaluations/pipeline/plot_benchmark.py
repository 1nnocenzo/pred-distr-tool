#!/usr/bin/env python3
"""Plot benchmark / dispatch results for GNN device selection.

Supports two input formats:

1. **Benchmark results** (from ``run_gnn_dispatch.py --dry-run``):
   ``{weight: {policy: {overall, per_device, load_balance_cv}}}``
   Compares GNN, GT-Weighted, Round-Robin, Oracle, FIFO side-by-side.

2. **Dispatch results** (from ``run_gnn_dispatch.py``):
   ``{weight: {overall, per_device, load_balance_cv}}``
   GNN-only — plots are single-policy.

The format is auto-detected.  Figures produced:

  1. sweep.pdf                        — Mean fidelity vs fidelity weight (line chart)
  2. fidelity_bars.pdf                — Per-device mean fidelity at each weight (grouped bars + stdev)
  3. load_balance.pdf                 — Circuit count per device per policy (grouped bars)
  4. summary_bands.pdf                — Min/mean/max fidelity range across weights
  5. confusion_matrix.pdf             — Device assignment confusion matrix (GNN vs Oracle)
  6. regret.pdf                       — Fidelity regret distribution (GNN vs Oracle)
  7. confusion_matrix_gt_weighted.pdf — Confusion matrix (GNN vs GT-Weighted, when --gt-path used)
  8. regret_gt_weighted.pdf           — Regret distribution (GNN vs GT-Weighted, when --gt-path used)

Usage::

    python evaluations/pipeline/plot_benchmark.py
    python evaluations/pipeline/plot_benchmark.py --results evaluations/pipeline/results/dispatch_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _SCRIPT_DIR / "results" / "dispatch_results.json"

ALL_POLICIES = ["oracle", "gt_weighted", "gnn", "fifo", "round_robin"]
POLICY_LABELS = {"oracle": "Oracle", "gt_weighted": "GT-Weighted", "gnn": "GNN", "fifo": "FIFO", "round_robin": "Round-Robin"}
POLICY_COLORS = {"oracle": "#2ca02c", "gt_weighted": "#9467bd", "gnn": "#1f77b4", "fifo": "#ff7f0e", "round_robin": "#d62728"}
POLICY_STYLES = {"oracle": "--", "gt_weighted": "--", "gnn": "-", "fifo": ":", "round_robin": "-."}

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

# plt.rcParams.update({
#     "figure.dpi": 130,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "axes.grid": True,
#     "grid.alpha": 0.35,
#     "font.size": 14, 
#     "axes.titlesize": 16,
#     "axes.labelsize": 15,
#     "xtick.labelsize": 13,
#     "ytick.labelsize": 13,
#     "legend.fontsize": 13,
#     "figure.titlesize": 17,
# })


# ---------------------------------------------------------------------------
# Loading + format normalisation
# ---------------------------------------------------------------------------

def _load_and_normalise(results_path: Path) -> tuple[dict, list[str]]:
    """Load results JSON and normalise to multi-policy format.

    Returns:
        data: {weight_str: {policy: {overall, per_device, load_balance_cv}}}
        policies: ordered list of policy names present in the data.
    """
    with open(results_path) as f:
        raw = json.load(f)

    if not raw:
        raise ValueError(f"Empty results file: {results_path}")

    # Detect format by inspecting the first weight entry
    first_entry = next(iter(raw.values()))

    if "overall" in first_entry:
        # Dispatch format (single-policy): wrap each entry as {"gnn": entry}
        data = {w: {"gnn": v} for w, v in raw.items()}
        policies = ["gnn"]
        print(f"Detected dispatch results (GNN-only) from {results_path}")
    else:
        # Benchmark format (multi-policy): already correct
        data = raw
        # Discover which policies are present across all weights
        policy_set: set[str] = set()
        for entry in data.values():
            policy_set.update(entry.keys())
        policies = [p for p in ALL_POLICIES if p in policy_set]
        print(f"Detected benchmark results ({', '.join(policies)}) from {results_path}")

    return data, policies


def _weights(data: dict) -> list[float]:
    return sorted(float(w) for w in data)


# ---------------------------------------------------------------------------
# Figure 1: Sweep — mean fidelity vs fidelity weight
# ---------------------------------------------------------------------------

def plot_sweep(data: dict, policies: list[str], out_dir: Path) -> None:
    weights = _weights(data)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pol in policies:
        means = []
        for w in weights:
            entry = data[f"{w:.2f}"]
            if pol in entry:
                means.append(entry[pol]["overall"]["mean"])
            else:
                means.append(None)

        valid = [(w, m) for w, m in zip(weights, means) if m is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(
            xs, ys,
            label=POLICY_LABELS.get(pol, pol),
            color=POLICY_COLORS.get(pol, "#333333"),
            linestyle=POLICY_STYLES.get(pol, "-"),
            linewidth=2,
            marker="o",
            markersize=5,
        )

    ax.set_xlabel("Fidelity weight")
    ax.set_ylabel("Mean predicted fidelity")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))

    # Auto-zoom y-axis to data range with 15% padding so small differences are visible
    all_ys = [
        entry[pol]["overall"]["mean"]
        for entry in data.values()
        for pol in policies
        if pol in entry
    ]
    if all_ys:
        y_min, y_max = min(all_ys), max(all_ys)
        margin = max((y_max - y_min) * 0.15, 0.001)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.legend(framealpha=0.7)
    fig.tight_layout()
    out = out_dir / "sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Per-device mean fidelity with error bars, one sub-plot per weight
# ---------------------------------------------------------------------------

def plot_fidelity_bars(data: dict, policies: list[str], out_dir: Path) -> None:
    weights = _weights(data)
    n_weights = len(weights)

    fig, axes = plt.subplots(
        1, n_weights,
        figsize=(5 * n_weights, 4),
        sharey=True,
        squeeze=False,
    )

    x = np.arange(len(DEVICES))
    n_pol = len(policies)
    width = min(0.18, 0.7 / max(n_pol, 1))
    offsets = np.linspace(-(n_pol - 1) / 2, (n_pol - 1) / 2, n_pol) * width

    for idx, w in enumerate(weights):
        ax = axes[0][idx]
        entry = data[f"{w:.2f}"]

        for pidx, pol in enumerate(policies):
            if pol not in entry:
                continue
            means = [entry[pol]["per_device"][d]["mean"] for d in DEVICES]
            stdevs = [entry[pol]["per_device"][d]["stdev"] for d in DEVICES]
            ax.bar(
                x + offsets[pidx], means,
                width=width,
                yerr=stdevs,
                label=POLICY_LABELS.get(pol, pol),
                color=POLICY_COLORS.get(pol, "#333333"),
                capsize=3,
                alpha=0.85,
                error_kw={"elinewidth": 1},
            )

        ax.set_title(f"weight = {w:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(DEVICES, fontsize=12)
        if idx == 0:
            ax.set_ylabel("Mean fidelity")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_pol, framealpha=0.7)
    fig.tight_layout()
    out = out_dir / "fidelity_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Load balance — circuits per device
# ---------------------------------------------------------------------------

def plot_load_balance(data: dict, policies: list[str], out_dir: Path) -> None:
    weights = _weights(data)

    fig, axes = plt.subplots(1, len(weights), figsize=(4 * len(weights), 4), sharey=True, squeeze=False)

    x = np.arange(len(DEVICES))
    n_pol = len(policies)
    width = min(0.18, 0.7 / max(n_pol, 1))
    offsets = np.linspace(-(n_pol - 1) / 2, (n_pol - 1) / 2, n_pol) * width

    for widx, w in enumerate(weights):
        ax = axes[0][widx]
        entry = data[f"{w:.2f}"]
        for pidx, pol in enumerate(policies):
            if pol not in entry:
                continue
            counts = [entry[pol]["per_device"][d]["count"] for d in DEVICES]
            ax.bar(
                x + offsets[pidx], counts,
                width=width,
                label=POLICY_LABELS.get(pol, pol),
                color=POLICY_COLORS.get(pol, "#333333"),
                alpha=0.85,
            )
        ax.set_title(f"weight = {w:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(DEVICES, fontsize=12)
        if widx == 0:
            ax.set_ylabel("Circuits assigned")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_pol, framealpha=0.7,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "load_balance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Summary — min / mean / max fidelity bands per policy, all weights
# ---------------------------------------------------------------------------

def plot_summary_bands(data: dict, policies: list[str], out_dir: Path) -> None:
    weights = _weights(data)
    n_pol = len(policies)

    fig, axes = plt.subplots(1, n_pol, figsize=(4.5 * n_pol, 4), sharey=True, squeeze=False)

    for pidx, pol in enumerate(policies):
        ax = axes[0][pidx]
        means, mins_, maxs = [], [], []
        for w in weights:
            entry = data[f"{w:.2f}"]
            if pol not in entry:
                means.append(None); mins_.append(None); maxs.append(None)
                continue
            o = entry[pol]["overall"]
            means.append(o["mean"])
            mins_.append(o["min"])
            maxs.append(o["max"])

        valid = [(w, mn, mi, mx) for w, mn, mi, mx in zip(weights, means, mins_, maxs)
                 if mn is not None]
        if not valid:
            continue
        xs, ys, lo, hi = zip(*valid)

        color = POLICY_COLORS.get(pol, "#333333")
        ax.fill_between(xs, lo, hi, alpha=0.15, color=color, label="min-max range")
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label="mean")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(POLICY_LABELS.get(pol, pol))
        ax.set_xlabel("Fidelity weight")
        if pidx == 0:
            ax.set_ylabel("Predicted fidelity")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.legend(fontsize=12, framealpha=0.7)

    fig.tight_layout()
    out = out_dir / "summary_bands.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Confusion matrix — GNN device assignment vs Oracle
# ---------------------------------------------------------------------------

def _plot_confusion_for_key(
    data: dict, cp_key: str, baseline_label: str, out_dir: Path, filename: str,
) -> None:
    """Plot confusion matrices for a given cross-policy key."""
    weights = _weights(data)
    cp_weights = [w for w in weights if cp_key in data[f"{w:.2f}"]]
    if not cp_weights:
        return

    n_weights = len(cp_weights)

    fig, axes = plt.subplots(
        1, n_weights,
        figsize=(5.5 * n_weights, 5.5),
        sharey=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.08},
    )

    for idx, w in enumerate(cp_weights):
        ax = axes[0][idx]
        cp = data[f"{w:.2f}"][cp_key]
        matrix = np.array(cp["confusion_matrix"]["matrix"])
        labels = cp["confusion_matrix"]["labels"]
        n = len(labels)

        ax.imshow(matrix, cmap="Blues", vmin=0, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=12, rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=12)
        if idx == 0:
            ax.set_ylabel(f"{baseline_label} device")
        ax.set_title(f"weight = {w:.2f}")

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

    fig.supxlabel("GNN device", fontsize=15, y=-0.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    out = out_dir / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_confusion_matrix(data: dict, out_dir: Path) -> None:
    _plot_confusion_for_key(data, "cross_policy", "Oracle", out_dir, "confusion_matrix.pdf")
    _plot_confusion_for_key(data, "cross_policy_gt_weighted", "GT-Weighted", out_dir, "confusion_matrix_gt_weighted.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Regret distribution — fidelity loss from non-oracle assignment
# ---------------------------------------------------------------------------

def _plot_regret_for_key(
    data: dict, cp_key: str, baseline_label: str, out_dir: Path, filename: str,
) -> None:
    """Plot regret distributions for a given cross-policy key."""
    weights = _weights(data)
    cp_weights = [w for w in weights if cp_key in data[f"{w:.2f}"]]
    if not cp_weights:
        return

    n_weights = len(cp_weights)

    fig, axes = plt.subplots(
        1, n_weights,
        figsize=(5 * n_weights, 3.5),
        sharey=True,
        squeeze=False,
    )

    for idx, w in enumerate(cp_weights):
        ax = axes[0][idx]
        cp = data[f"{w:.2f}"][cp_key]
        regret_vals = cp["regret"]["values"]

        if not regret_vals:
            ax.set_visible(False)
            continue

        ax.hist(regret_vals, bins=30, color="#1f77b4", edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.axvline(cp["regret"]["mean"], color="#d62728", linewidth=1.5,
                   linestyle="--", label=f"mean = {cp['regret']['mean']:.4f}")
        ax.axvline(cp["regret"]["median"], color="#ff7f0e", linewidth=1.5,
                   linestyle=":", label=f"median = {cp['regret']['median']:.4f}")
        if idx == 0:
            ax.set_ylabel("Circuit count")
        ax.set_title(f"weight = {w:.2f}")
        ax.legend(fontsize=12, framealpha=0.7)

    fig.supxlabel(f"Fidelity regret ({baseline_label} - GNN)", fontsize=15)
    fig.tight_layout()
    out = out_dir / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_regret(data: dict, out_dir: Path) -> None:
    _plot_regret_for_key(data, "cross_policy", "Oracle", out_dir, "regret.pdf")
    _plot_regret_for_key(data, "cross_policy_gt_weighted", "GT-Weighted", out_dir, "regret_gt_weighted.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark / dispatch results for GNN device selection.",
    )
    parser.add_argument(
        "--results", type=Path, default=_DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {_DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to save plots (default: same directory as results file)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        print("Run run_gnn_dispatch.py --dry-run first.")
        raise SystemExit(1)

    out_dir = args.output_dir or (args.results.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    data, policies = _load_and_normalise(args.results)
    print(f"Loaded results for {len(data)} fidelity weight(s), policies: {policies}")

    plot_sweep(data, policies, out_dir)
    plot_fidelity_bars(data, policies, out_dir)
    plot_load_balance(data, policies, out_dir)
    plot_summary_bands(data, policies, out_dir)

    # Cross-policy plots (only if data contains cross_policy metrics)
    has_cross = any(
        "cross_policy" in entry or "cross_policy_gt_weighted" in entry
        for entry in data.values()
    )
    if has_cross:
        plot_confusion_matrix(data, out_dir)
        plot_regret(data, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
