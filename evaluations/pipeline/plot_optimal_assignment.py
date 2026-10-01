"""Plot greedy vs. optimal assignment from ``optimal_assignment.csv``.

Left: share of the Round-Robin -> Oracle gap recovered by each policy as a
function of the fidelity weight.  Right: the same gain against the load
coefficient of variation (the trade-off curve).  Greedy and optimum are
shown for the shuffled queue of the main sweep (``--shuffle-seed 0``), and
greedy also for the name-sorted queue, which groups circuits by family.  Each
queue is normalised by its own Round-Robin.  Colours follow the warm
Approach-4 thesis palette.

Usage::

    python evaluations/pipeline/plot_optimal_assignment.py \
        --shuffled-dir evaluations/pipeline/results_v4_shuffled \
        --sorted-dir evaluations/pipeline/results_v3_dense
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

GT_COLOR = "#912534"
GNN_COLOR = "#D9402A"
ORACLE_COLOR = "#F4A464"
REF_COLOR = "#3D1B28"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
})

# (queue, gain column, cv column, label, colour, linestyle, marker)
SERIES = [
    ("shuffled", "optimal_gt_norm", "cv_gt", "Optimal-GT", GT_COLOR, "--", "s"),
    ("shuffled", "greedy_gt_norm", "cv_gt", "Greedy-GT (GT-Weighted)", GT_COLOR, "-", "o"),
    ("sorted", "greedy_gt_norm", "cv_gt", "Greedy-GT, sorted queue", GT_COLOR, ":", "^"),
    ("shuffled", "optimal_gnn_norm", "cv_gnn", "Optimal-GNN", GNN_COLOR, "--", "s"),
    ("shuffled", "greedy_gnn_norm", "cv_gnn", "Greedy-GNN", GNN_COLOR, "-", "o"),
    ("sorted", "greedy_gnn_norm", "cv_gnn", "Greedy-GNN, sorted queue", GNN_COLOR, ":", "^"),
]


def _load(results_dir: Path) -> list[dict[str, float]]:
    with (results_dir / "optimal_assignment.csv").open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shuffled-dir", type=Path,
                        default=Path("evaluations/pipeline/results_v4_shuffled"))
    parser.add_argument("--sorted-dir", type=Path,
                        default=Path("evaluations/pipeline/results_v3_dense"))
    args = parser.parse_args()

    data = {"shuffled": _load(args.shuffled_dir), "sorted": _load(args.sorted_dir)}
    ws = [r["w"] for r in data["shuffled"]]

    fig, (ax_w, ax_cv) = plt.subplots(1, 2, figsize=(11, 4.6))
    for queue, key, cv_key, label, color, ls, marker in SERIES:
        rows = data[queue]
        ys = [100 * r[key] for r in rows]
        cvs = [r[cv_key] for r in rows]
        kw = dict(color=color, linestyle=ls, marker=marker, markersize=4.5, linewidth=2)
        ax_w.plot(ws, ys, label=label, **kw)
        ax_cv.plot(cvs, ys, **kw)

    for ax in (ax_w, ax_cv):
        ax.axhline(100, color=ORACLE_COLOR, linewidth=1.5, linestyle="--")
        ax.axhline(0, color=REF_COLOR, linewidth=1, linestyle="-.")
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax_w.text(0.0, 101.5, "Oracle", color=REF_COLOR, fontsize=10)
    ax_w.text(0.12, 1.5, "Round-Robin", color=REF_COLOR, fontsize=10)

    ax_w.set_xlabel("Fidelity weight $w$")
    ax_w.set_ylabel("Gain over Round-Robin\n(% of Oracle gap)")
    ax_w.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_w.set_title("(a) Gain vs. fidelity weight", fontsize=12)
    ax_cv.set_xlabel("Load coefficient of variation")
    ax_cv.set_xscale("symlog", linthresh=0.01)
    ax_cv.set_xlim(left=0)
    ax_cv.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax_cv.set_title("(b) Gain vs. load imbalance", fontsize=12)
    ax_w.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), framealpha=0.85, fontsize=9.5)

    fig.tight_layout()
    out = args.shuffled_dir / "plots" / "optimal_vs_greedy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
