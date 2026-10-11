#!/usr/bin/env python3
"""Turn the study's raw JSON into publication-ready tables and figures.

Reads whatever experiments are present under ``--results-dir`` and writes, into
``--results-dir/report/``:

``summary_leave_one_family_out.csv``, ``summary_size_extrapolation.csv``, ``summary_random_control.csv``
    One row per (split, device) plus an ``ALL`` row.
``summary_size_extrapolation_by_qubits.csv``
    Metrics per test circuit size.
``table_leave_one_family_out.tex``, ``table_size_extrapolation.tex``, ``table_size_extrapolation_by_qubits.tex``, ``table_scheduling.tex``
    ``booktabs`` tables for direct ``\\input{}`` into the manuscript.
``fig_leave_one_family_out.*``, ``fig_size_extrapolation.*``, ``fig_scheduling_heldout.*``
    PDF + PNG figures.
``report_summary.json``
    The headline numbers (control vs held-out), convenient for the text.

Safe to run repeatedly and on a partial study: missing experiments are reported
and skipped.

Example::

    python evaluations/generalization/scripts/make_report.py
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from genstudy import paths, report
from genstudy.io import load_json, save_csv, save_json, setup_logging

logger = logging.getLogger("make_report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=paths.RESULTS_DIR)
    parser.add_argument(
        "--scheduling-tag", default="leave_one_family_out",
        help="Which results/scheduling/<tag> run to tabulate.",
    )
    parser.add_argument(
        "--gt-path", type=Path,
        default=paths.MODEL_DIR / "expected_fidelity_results_benchmark.json",
        help="Ground-truth fidelities, used to back-fill the baseline regrets of "
             "splits run before they were recorded.",
    )
    parser.add_argument(
        "--label-prefix", default="tab:ml4qc",
        help="Prefix for the generated LaTeX labels.",
    )
    return parser


#: Table note shared by the per-split tables.
REGRET_NOTE = (
    r"Dev.\ acc.\ is the fraction of circuits for which the predictor's best device "
    r"equals the ground-truth best device. Regret is the mean fidelity lost by "
    r"following the predictor's best device instead of the true one; "
    r"Regret$_\text{fixed}$ is the same for always choosing the single device with "
    r"the highest mean fidelity on the test set (chosen a posteriori), and "
    r"Regret$_\text{rand}$ for a uniformly random device."
)


def _collect(experiment_dir: Path, gt_path: Path) -> list[dict[str, Any]]:
    """Per-split results with the baseline regrets filled in."""
    results = report.collect_split_metrics(experiment_dir)
    report.add_baseline_regrets(results, experiment_dir, gt_path)
    return results


def _control_result(results_dir: Path, gt_path: Path) -> dict[str, Any] | None:
    """The random-split control, if it has been run."""
    control = _collect(results_dir / "random_control", gt_path)
    if not control:
        logger.info("No random-split control found — tables omit the reference row.")
        return None
    return control[0]


def _regret_headline(test: dict[str, Any]) -> dict[str, Any]:
    return {
        key: test[key]
        for key in ("fidelity_regret_mean", "regret_best_fixed_device", "regret_random_device")
        if key in test
    }


def main() -> int:
    args = build_parser().parse_args()

    report_dir = args.results_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(report_dir / "make_report.log")
    logger.info("Command line: %s", " ".join(sys.argv))

    control = _control_result(args.results_dir, args.gt_path)
    headline: dict[str, Any] = {}

    if control is not None:
        report.write_experiment_summary([control], report_dir, "random_control")
        headline["random_control"] = {
            "n_test": control["test"]["n_circuits"],
            "mae": control["test"]["mae"],
            "rmse": control["test"]["rmse"],
            "r2": control["test"]["r2"],
            "device_choice_accuracy": control["test"]["device_choice_accuracy"],
            **_regret_headline(control["test"]),
        }

    # --- Leave-one-family-out ------------------------------------------------
    lofo = _collect(args.results_dir / "leave_one_family_out", args.gt_path)
    if lofo:
        lofo.sort(key=lambda r: -r["test"]["n_circuits"])
        report.write_experiment_summary(lofo, report_dir, "leave_one_family_out")
        report.write_latex(
            report.latex_generalization_table(
                lofo,
                caption=(
                    "Leave-one-family-out generalization of the fidelity predictor. "
                    "For every row the model is trained on all other benchmark families "
                    "with the tuned hyper-parameters of "
                    r"Table~\ref{tab:ml4qc:scheduling-hparams} (no re-tuning) and tested "
                    "on the held-out family. The last row repeats the random, "
                    "fidelity-stratified split, where near-duplicate circuits of the same "
                    "family occur on both sides."
                ),
                label=f"{args.label_prefix}:lofo",
                split_header="Held-out family",
                control=control,
                note=REGRET_NOTE,
            ),
            report_dir, "table_leave_one_family_out",
        )
        report.plot_family_generalization(lofo, report_dir, control=control)

        headline["leave_one_family_out"] = {
            "n_folds": len(lofo),
            "mae_mean": statistics.fmean(r["test"]["mae"] for r in lofo),
            "rmse_mean": statistics.fmean(r["test"]["rmse"] for r in lofo),
            "r2_mean": statistics.fmean(r["test"]["r2"] for r in lofo),
            "r2_min": min(r["test"]["r2"] for r in lofo),
            "r2_max": max(r["test"]["r2"] for r in lofo),
            "device_choice_accuracy_mean": statistics.fmean(
                r["test"]["device_choice_accuracy"] for r in lofo
            ),
            "worst_family": min(lofo, key=lambda r: r["test"]["r2"])["split"],
            "per_family_regret": {r["split"]: _regret_headline(r["test"]) for r in lofo},
        }
    else:
        logger.warning("No leave-one-family-out results under %s", args.results_dir)

    # --- Size extrapolation --------------------------------------------------
    size = _collect(args.results_dir / "size_extrapolation", args.gt_path)
    if size:
        report.write_experiment_summary(size, report_dir, "size_extrapolation")
        report.write_latex(
            report.latex_generalization_table(
                size,
                caption=(
                    "Size extrapolation of the fidelity predictor. Row $q_{\\max}=n$ is "
                    "trained only on circuits with at most $n$ qubits and tested on all "
                    "larger circuits of the benchmark set, with the hyper-parameters of "
                    r"Table~\ref{tab:ml4qc:scheduling-hparams} left unchanged."
                ),
                label=f"{args.label_prefix}:size-extrapolation",
                split_header="Training cut-off",
                control=control,
                note=REGRET_NOTE,
            ),
            report_dir, "table_size_extrapolation",
        )

        qubit_rows = report.qubit_rows(size)
        if qubit_rows:
            save_csv(qubit_rows, report_dir / "summary_size_extrapolation_by_qubits.csv")
            report.write_latex(
                report.latex_qubit_table(
                    qubit_rows,
                    caption=(
                        "Size extrapolation broken down by test circuit size, for the "
                        "model trained on the smaller circuits only."
                    ),
                    label=f"{args.label_prefix}:size-extrapolation-by-qubits",
                ),
                report_dir, "table_size_extrapolation_by_qubits",
            )
            report.plot_size_extrapolation(qubit_rows, report_dir)

        headline["size_extrapolation"] = [
            {
                "split": r["split"],
                "n_test": r["test"]["n_circuits"],
                "mae": r["test"]["mae"],
                "rmse": r["test"]["rmse"],
                "r2": r["test"]["r2"],
                "device_choice_accuracy": r["test"]["device_choice_accuracy"],
                **_regret_headline(r["test"]),
            }
            for r in size
        ]
    else:
        logger.warning("No size-extrapolation results under %s", args.results_dir)

    # --- Scheduling re-run ---------------------------------------------------
    sweep_path = args.results_dir / "scheduling" / args.scheduling_tag / "sweep_summary.csv"
    summary_path = args.results_dir / "scheduling" / args.scheduling_tag / "scheduling_summary.json"
    if summary_path.is_file():
        scheduling = load_json(summary_path)
        # Prefer the JSON (typed) over re-parsing the CSV.
        rows = scheduling.get("sweep") or _rows_from_csv(sweep_path)
        if rows:
            report.write_latex(
                report.latex_scheduling_table(
                    rows,
                    caption=(
                        "Scheduling evaluation driven by held-out predictions "
                        f"({args.scheduling_tag.replace('_', ' ')}): mean achieved fidelity "
                        "versus the policy weight $w$, together with the device agreement "
                        "rate and the mean fidelity regret of the predictor-driven policy "
                        "against GT-Weighted, i.e.\\ the same scoring rule supplied with "
                        "ground-truth fidelities."
                    ),
                    label=f"{args.label_prefix}:lofo-scheduling",
                ),
                report_dir, "table_scheduling",
            )
            report.plot_scheduling_sweep(rows, report_dir)

            best = max(rows, key=lambda r: r["gnn_mean_fidelity"])
            headline["scheduling"] = {
                "tag": args.scheduling_tag,
                "n_circuits": best.get("n_circuits"),
                "best_weight": best["fidelity_weight"],
                "gnn_mean_fidelity_at_best_w": best["gnn_mean_fidelity"],
                "gt_weighted_mean_fidelity_at_best_w": best["gt_weighted_mean_fidelity"],
                "gap_to_gt_weighted_pct": best["gnn_vs_gt_weighted_pct"],
                "agreement_vs_gt_weighted_at_best_w": best["agreement_vs_gt_weighted"],
                "regret_vs_gt_weighted_at_best_w": best["regret_vs_gt_weighted"],
            }
    else:
        logger.warning("No scheduling results for tag '%s'", args.scheduling_tag)

    save_json(headline, report_dir / "report_summary.json")
    logger.info("Report written to %s", report_dir)
    return 0


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    """Read ``sweep_summary.csv`` back, coercing numeric columns to floats."""
    import csv

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
