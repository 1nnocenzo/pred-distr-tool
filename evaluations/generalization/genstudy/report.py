"""Publication-ready reporting: CSV tables, LaTeX tables and figures.

The study writes raw per-split JSON; this module turns it into the artefacts a
paper needs, so no number is ever re-typed by hand:

``summary_<experiment>.csv``
    One row per held-out split and device (plus an ``ALL`` row), ready for any
    plotting tool.

``table_leave_one_family_out.tex`` / ``table_size_extrapolation.tex`` / ``table_scheduling.tex``
    ``booktabs`` tables; drop them into the manuscript with ``\\input{...}``.

``fig_*.pdf`` / ``fig_*.png``
    R^2 and MAE per held-out family against the random-split control, and mean
    fidelity versus the scheduler's fidelity weight ``w``.

LaTeX output uses ``\\sisetup``-free plain numbers so it compiles without
``siunitx``; only ``booktabs`` is required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import DEVICE_NAMES
from .metrics import baseline_regrets
from .io import load_json, save_csv, save_text

logger = logging.getLogger(__name__)

#: Human-readable device names for tables.
DEVICE_LABELS = {
    "EQE1_Top": r"EQE1\_Top",
    "EQE1_Bottom": r"EQE1\_Bottom",
    "QExa20": r"QExa20",
}


def _escape(text: str) -> str:
    """Escape the LaTeX-significant characters that occur in family names."""
    return text.replace("_", r"\_")


# ---------------------------------------------------------------------------
# Collecting raw results
# ---------------------------------------------------------------------------

def collect_split_metrics(experiment_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*/metrics.json`` under an experiment directory, sorted by split."""
    if not experiment_dir.is_dir():
        return []
    results = []
    for metrics_path in sorted(experiment_dir.glob("*/metrics.json")):
        try:
            results.append(load_json(metrics_path))
        except Exception:
            logger.exception("Could not read %s", metrics_path)
    logger.info("Collected %d splits from %s", len(results), experiment_dir)
    return results


def add_baseline_regrets(
    results: Sequence[dict[str, Any]], experiment_dir: Path, ground_truth_path: Path
) -> None:
    """Fill in the baseline regrets for splits whose ``metrics.json`` predates them.

    The test targets are rebuilt from the split's ``predictions.json`` (which
    names exactly the test circuits) and the ground-truth benchmark file, so
    finished splits gain the columns without retraining.  ``results`` is
    updated in place; nothing on disk is modified.
    """
    missing = [r for r in results if "regret_random_device" not in r["test"]]
    if not missing:
        return
    ground_truth = load_json(ground_truth_path)
    for result in missing:
        predictions_path = experiment_dir / str(result.get("split", "")) / "predictions.json"
        if not predictions_path.is_file():
            logger.warning("No %s; baseline regrets left empty", predictions_path)
            continue
        names = list(load_json(predictions_path))
        absent = [n for n in names if n not in ground_truth]
        if absent:
            logger.warning("%d test circuits of %s missing from %s; baseline regrets left empty",
                           len(absent), result.get("split"), ground_truth_path)
            continue
        targets = np.array(
            [[ground_truth[n][0][d]["fidelity"] for d in DEVICE_NAMES] for n in names]
        )
        result["test"].update(baseline_regrets(targets))
        logger.info("Back-filled baseline regrets for %s", result.get("split"))


def metrics_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-split metrics into one row per (split, device) plus ``ALL``."""
    rows: list[dict[str, Any]] = []
    for result in results:
        test = result["test"]
        base = {
            "experiment": result.get("experiment", ""),
            "split": result.get("split", ""),
            "n_train": result.get("n_train", ""),
            "n_test": result.get("n_test", ""),
        }
        rows.append(
            base
            | {
                "device": "ALL",
                "mae": test["mae"],
                "rmse": test["rmse"],
                "r2": test["r2"],
                "target_std": test["target_std"],
                "device_choice_accuracy": test["device_choice_accuracy"],
                "fidelity_regret_mean": test["fidelity_regret_mean"],
                "regret_best_fixed_device": test.get("regret_best_fixed_device", ""),
                "regret_random_device": test.get("regret_random_device", ""),
            }
        )
        for device in DEVICE_NAMES:
            per_device = test["per_device"][device]
            rows.append(
                base
                | {
                    "device": device,
                    "mae": per_device["mae"],
                    "rmse": per_device["rmse"],
                    "r2": per_device["r2"],
                    "target_std": per_device["target_std"],
                    "device_choice_accuracy": "",
                    "fidelity_regret_mean": "",
                    "regret_best_fixed_device": "",
                    "regret_random_device": "",
                }
            )
    return rows


def qubit_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per (split, test qubit count) for the size-extrapolation breakdown."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for qubits, block in result.get("test_by_qubits", {}).items():
            rows.append(
                {
                    "split": result.get("split", ""),
                    "qubits": int(qubits),
                    "n_circuits": block["n_circuits"],
                    "mae": block["mae"],
                    "rmse": block["rmse"],
                    "r2": block["r2"],
                    "device_choice_accuracy": block["device_choice_accuracy"],
                }
            )
    rows.sort(key=lambda r: (r["split"], r["qubits"]))
    return rows


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _tabular(header: Sequence[str], body: Sequence[Sequence[str]], column_spec: str) -> str:
    lines = [
        r"\begin{tabular}{" + column_spec + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _table(
    body: str, caption: str, label: str, *, note: str | None = None
) -> str:
    parts = [
        r"% Generated by evaluations/generalization/scripts/make_report.py — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        body,
    ]
    if note:
        parts.append(r"\par\smallskip\footnotesize " + note)
    parts.append(r"\end{table}")
    return "\n".join(parts) + "\n"


def latex_generalization_table(
    results: Sequence[Mapping[str, Any]],
    *,
    caption: str,
    label: str,
    split_header: str,
    control: Mapping[str, Any] | None = None,
    note: str | None = None,
) -> str:
    """Per-split table: overall MAE/RMSE/R^2 and per-device MAE/R^2.

    ``control`` (the random-split result) is appended as a final reference row so
    the generalization gap is visible in a single table.
    """
    header = [
        split_header,
        r"$n_\text{test}$",
        "MAE",
        "RMSE",
        r"$R^2$",
    ]
    for device in DEVICE_NAMES:
        header += [f"MAE$_{{\\text{{{DEVICE_LABELS[device]}}}}}$",
                   f"$R^2_{{\\text{{{DEVICE_LABELS[device]}}}}}$"]
    header += [r"Dev.\ acc.", "Regret", r"Regret$_\text{fixed}$", r"Regret$_\text{rand}$"]

    def _row(result: Mapping[str, Any], name: str) -> list[str]:
        test = result["test"]
        cells = [
            name,
            f"{test['n_circuits']:d}",
            f"{test['mae']:.4f}",
            f"{test['rmse']:.4f}",
            f"{test['r2']:.3f}",
        ]
        for device in DEVICE_NAMES:
            per_device = test["per_device"][device]
            cells += [f"{per_device['mae']:.4f}", f"{per_device['r2']:.3f}"]
        cells.append(f"{test['device_choice_accuracy']:.3f}")
        cells.append(f"{test['fidelity_regret_mean']:.4f}")
        for key in ("regret_best_fixed_device", "regret_random_device"):
            cells.append(f"{test[key]:.4f}" if key in test else "--")
        return cells

    body_rows = [_row(r, _escape(str(r.get("split", "")))) for r in results]

    body = _tabular(header, body_rows, "l" + "r" * (len(header) - 1))
    if control is not None:
        control_row = " & ".join(_row(control, r"\textit{random split (control)}"))
        body = body.replace(
            r"\bottomrule",
            r"\midrule" + "\n" + control_row + r" \\" + "\n" + r"\bottomrule",
        )
    return _table(body, caption, label, note=note)


def latex_qubit_table(
    rows: Sequence[Mapping[str, Any]], *, caption: str, label: str, note: str | None = None
) -> str:
    """Size-extrapolation breakdown: metrics per test qubit count."""
    header = ["Qubits", r"$n_\text{test}$", "MAE", "RMSE", r"$R^2$", r"Dev.\ acc."]
    body_rows = [
        [
            f"{row['qubits']:d}",
            f"{row['n_circuits']:d}",
            f"{row['mae']:.4f}",
            f"{row['rmse']:.4f}",
            f"{row['r2']:.3f}",
            f"{row['device_choice_accuracy']:.3f}",
        ]
        for row in rows
    ]
    return _table(
        _tabular(header, body_rows, "l" + "r" * (len(header) - 1)), caption, label, note=note
    )


def latex_scheduling_table(
    sweep_rows: Sequence[Mapping[str, Any]],
    *,
    caption: str,
    label: str,
    note: str | None = None,
) -> str:
    """Scheduling re-run: mean fidelity, agreement and regret versus ``w``."""
    header = [
        "$w$",
        "Oracle",
        "GT-Weighted",
        "Predictor",
        "Round-Robin",
        r"Agree.\ vs GT-W.",
        r"Regret vs GT-W.",
    ]
    body_rows = [
        [
            f"{row['fidelity_weight']:.2f}",
            f"{row['oracle_mean_fidelity']:.4f}",
            f"{row['gt_weighted_mean_fidelity']:.4f}",
            f"{row['gnn_mean_fidelity']:.4f}",
            f"{row['round_robin_mean_fidelity']:.4f}",
            f"{row['agreement_vs_gt_weighted']*100:.1f}\\%",
            f"{row['regret_vs_gt_weighted']:.4f}",
        ]
        for row in sweep_rows
    ]
    return _table(
        _tabular(header, body_rows, "l" + "r" * (len(header) - 1)), caption, label, note=note
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        logger.info("Wrote %s", path)


def plot_family_generalization(
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    control: Mapping[str, Any] | None = None,
    stem: str = "fig_leave_one_family_out",
) -> None:
    """Grouped bars: MAE and R^2 per held-out family, with the control as a line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figure '%s'", stem)
        return
    if not results:
        return

    names = [str(r.get("split", "")) for r in results]
    mae = [r["test"]["mae"] for r in results]
    r2 = [r["test"]["r2"] for r in results]
    positions = range(len(names))

    fig, (ax_mae, ax_r2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)

    ax_mae.bar(positions, mae, color="#4C72B0")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("Leave-one-family-out generalization")

    ax_r2.bar(positions, r2, color="#55A868")
    ax_r2.set_ylabel(r"$R^2$")
    ax_r2.axhline(0.0, color="black", linewidth=0.8)

    if control is not None:
        ax_mae.axhline(
            control["test"]["mae"], color="#C44E52", linestyle="--",
            label=f"random split (control): {control['test']['mae']:.4f}",
        )
        ax_r2.axhline(
            control["test"]["r2"], color="#C44E52", linestyle="--",
            label=f"random split (control): {control['test']['r2']:.3f}",
        )
        ax_mae.legend(fontsize="small")
        ax_r2.legend(fontsize="small")

    ax_r2.set_xticks(list(positions))
    ax_r2.set_xticklabels(names, rotation=35, ha="right")
    for axis in (ax_mae, ax_r2):
        axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_size_extrapolation(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stem: str = "fig_size_extrapolation",
) -> None:
    """MAE and R^2 as a function of the test circuit size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figure '%s'", stem)
        return
    if not rows:
        return

    fig, ax_mae = plt.subplots(figsize=(6.4, 3.6))
    ax_r2 = ax_mae.twinx()
    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), []).append(row)

    for split_name, split_rows in sorted(by_split.items()):
        qubits = [r["qubits"] for r in split_rows]
        ax_mae.plot(qubits, [r["mae"] for r in split_rows], "o-", label=f"MAE ({split_name})")
        ax_r2.plot(qubits, [r["r2"] for r in split_rows], "s--", color="#55A868",
                   label=f"$R^2$ ({split_name})")

    ax_mae.set_xlabel("Test circuit size (qubits)")
    ax_mae.set_ylabel("MAE")
    ax_r2.set_ylabel(r"$R^2$")
    ax_mae.grid(alpha=0.3)
    handles = ax_mae.get_legend_handles_labels()[0] + ax_r2.get_legend_handles_labels()[0]
    labels = ax_mae.get_legend_handles_labels()[1] + ax_r2.get_legend_handles_labels()[1]
    ax_mae.legend(handles, labels, fontsize="small")
    fig.tight_layout()
    _save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_scheduling_sweep(
    sweep_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stem: str = "fig_scheduling_heldout",
) -> None:
    """Mean fidelity versus ``w`` for every policy, on held-out predictions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figure '%s'", stem)
        return
    if not sweep_rows:
        return

    weights = [row["fidelity_weight"] for row in sweep_rows]
    series = [
        ("Oracle", "oracle_mean_fidelity", "#937860", ":"),
        ("GT-Weighted", "gt_weighted_mean_fidelity", "#C44E52", "--"),
        ("Predictor (held-out)", "gnn_mean_fidelity", "#4C72B0", "-"),
        ("Round-Robin", "round_robin_mean_fidelity", "#8172B2", "-."),
    ]

    fig, (ax_fid, ax_agree) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    for label, key, color, style in series:
        ax_fid.plot(weights, [row[key] for row in sweep_rows], style, color=color, label=label)
    ax_fid.set_xlabel("Fidelity weight $w$")
    ax_fid.set_ylabel("Mean achieved fidelity")
    ax_fid.grid(alpha=0.3)
    ax_fid.legend(fontsize="small")

    ax_agree.plot(weights, [row["agreement_vs_gt_weighted"] for row in sweep_rows],
                  "o-", color="#4C72B0", label="Agreement vs GT-Weighted")
    ax_regret = ax_agree.twinx()
    ax_regret.plot(weights, [row["regret_vs_gt_weighted"] for row in sweep_rows],
                   "s--", color="#C44E52", label="Mean regret vs GT-Weighted")
    ax_agree.set_xlabel("Fidelity weight $w$")
    ax_agree.set_ylabel("Device agreement rate")
    ax_regret.set_ylabel("Mean fidelity regret")
    ax_agree.grid(alpha=0.3)
    handles = ax_agree.get_legend_handles_labels()[0] + ax_regret.get_legend_handles_labels()[0]
    labels = ax_agree.get_legend_handles_labels()[1] + ax_regret.get_legend_handles_labels()[1]
    ax_agree.legend(handles, labels, fontsize="small", loc="center right")

    fig.tight_layout()
    _save_figure(fig, output_dir, stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrappers used by make_report.py
# ---------------------------------------------------------------------------

def write_experiment_summary(
    results: Sequence[Mapping[str, Any]], output_dir: Path, experiment: str
) -> None:
    """Write ``summary_<experiment>.csv`` for the given per-split results."""
    save_csv(metrics_rows(results), output_dir / f"summary_{experiment}.csv")


def write_latex(text: str, output_dir: Path, stem: str) -> None:
    save_text(text, output_dir / f"{stem}.tex")
