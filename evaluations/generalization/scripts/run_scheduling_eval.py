#!/usr/bin/env python3
"""Replay the scheduling benchmark on held-out predictions.

The question this answers: when the fidelities the scheduler consumes come from a
model that has never seen the circuit's family (or its size), does the policy
still track GT-Weighted — the same scoring rule supplied with perfect fidelity
knowledge?

The scheduling code itself is **not** re-implemented.  This script imports
``evaluations/pipeline/run_gnn_dispatch.py`` and feeds it the held-out
predictions in place of the in-distribution ones, so Oracle, GT-Weighted,
Round-Robin, the mean-fidelity sweep over the weight ``w``, the device agreement
rate and the regret are computed by exactly the code that produced the published
numbers.  Nothing under ``evaluations/pipeline/`` is modified or overwritten.

Outputs (under ``--results-dir/scheduling/<tag>/``)::

    dispatch_results.json       full per-weight metrics (pipeline format)
    dispatch_summary.csv        per-policy, per-device breakdown
    cross_policy_summary.csv    agreement / regret versus Oracle and GT-Weighted
    sweep_summary.csv           tidy one-row-per-weight table used by make_report.py
    by_family/<family>/...      same artefacts restricted to one held-out family

Examples::

    # Pooled over all leave-one-family-out folds, full weight sweep
    python evaluations/generalization/scripts/run_scheduling_eval.py \\
        --fidelity-weights 0.0:1.0:0.1 --per-family

    # Size-extrapolation predictions instead
    python evaluations/generalization/scripts/run_scheduling_eval.py \\
        --predictions evaluations/generalization/results/size_extrapolation/qmax_14/predictions.json \\
        --tag size_extrapolation_qmax_14
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from genstudy import paths
from genstudy.data import family_of
from genstudy.io import load_json, save_csv, save_json, setup_logging

paths.ensure_pipeline_imports()

# The pipeline driver registers the policy and owns every scheduling metric.
import run_gnn_dispatch as pipeline  # noqa: E402

EXPERIMENT = "scheduling"

logger = logging.getLogger("run_scheduling_eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=paths.RESULTS_DIR / "leave_one_family_out" / "heldout_predictions.json",
        help="Held-out predictions: {'family/name': [fid_EQE1_Top, fid_EQE1_Bottom, fid_QExa20]}.",
    )
    parser.add_argument(
        "--gt-path", type=Path,
        default=paths.MODEL_DIR / "expected_fidelity_results_benchmark.json",
        help="Ground-truth fidelity JSON used by the oracle and all metrics.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Sub-directory name under results/scheduling (default: the predictions' "
             "parent directory name).",
    )
    parser.add_argument(
        "--fidelity-weights", default="0.0:1.0:0.1",
        help="Comma-separated list or start:stop:step for the policy weight w.",
    )
    parser.add_argument(
        "--min-best-fidelity", type=float, default=None, metavar="F",
        help="Drop circuits whose best ground-truth fidelity is below F.",
    )
    parser.add_argument(
        "--per-family", action="store_true",
        help="Additionally run the sweep separately for each held-out family.",
    )
    parser.add_argument(
        "--min-family-size", type=int, default=50,
        help="With --per-family, skip families with fewer circuits than this.",
    )
    parser.add_argument("--results-dir", type=Path, default=paths.RESULTS_DIR)
    return parser


def _sweep_rows(all_results: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the pipeline's nested results into one row per weight."""
    rows = []
    for weight in sorted(all_results):
        result = all_results[weight]
        oracle = result["oracle"]["overall"]["mean"]
        gt_weighted = result["gt_weighted"]["overall"]["mean"]
        predictor = result["gnn"]["overall"]["mean"]
        round_robin = result["round_robin"]["overall"]["mean"]
        vs_oracle = result["cross_policy"]
        vs_gt = result["cross_policy_gt_weighted"]
        rows.append(
            {
                "fidelity_weight": float(weight),
                "n_circuits": result["gnn"]["overall"]["total_circuits"],
                "oracle_mean_fidelity": oracle,
                "gt_weighted_mean_fidelity": gt_weighted,
                "gnn_mean_fidelity": predictor,
                "round_robin_mean_fidelity": round_robin,
                "gnn_vs_gt_weighted_pct": (
                    (predictor - gt_weighted) / gt_weighted * 100 if gt_weighted else 0.0
                ),
                "gnn_vs_round_robin_pct": (
                    (predictor - round_robin) / round_robin * 100 if round_robin else 0.0
                ),
                "agreement_vs_gt_weighted": vs_gt["device_agreement_rate"],
                "regret_vs_gt_weighted": vs_gt["regret"]["mean"],
                "agreement_vs_oracle": vs_oracle["device_agreement_rate"],
                "regret_vs_oracle": vs_oracle["regret"]["mean"],
                "load_balance_cv": result["gnn"]["load_balance_cv"],
            }
        )
    return rows


def _run_sweep(
    ground_truth: dict[str, list[float]],
    predictions: dict[str, list[float]],
    weights: list[float],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run the pipeline sweep for one circuit set and persist its artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Both dicts must carry the same keys in the same (sorted) order: the policy
    # walks a single unified queue, so a mismatch would desynchronise the
    # round-robin baseline from the ground truth it is scored against.
    common = sorted(set(ground_truth) & set(predictions))
    ground_truth = {k: ground_truth[k] for k in common}
    predictions = {k: predictions[k] for k in common}
    logger.info("Scheduling on %d circuits -> %s", len(common), output_dir)

    all_results = pipeline.run_sweep(ground_truth, weights, predictions)
    pipeline.print_report(all_results)
    pipeline.save_results(all_results, ground_truth, output_dir, predictions)

    rows = _sweep_rows(all_results)
    save_csv(rows, output_dir / "sweep_summary.csv")
    return rows


def main() -> int:
    args = build_parser().parse_args()

    tag = args.tag or args.predictions.parent.name
    output_dir = args.results_dir / EXPERIMENT / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "run.log")
    logger.info("Command line: %s", " ".join(sys.argv))

    if not args.predictions.is_file():
        logger.error(
            "Predictions not found: %s — run the training experiments first.",
            args.predictions,
        )
        return 1

    predictions: dict[str, list[float]] = {
        name: [float(v) for v in fids]
        for name, fids in load_json(args.predictions).items()
    }
    logger.info("Loaded %d held-out predictions from %s", len(predictions), args.predictions)

    ground_truth = pipeline.load_ground_truth_fidelities(
        args.gt_path, names_path=None, min_best_fidelity=args.min_best_fidelity,
    )
    weights = pipeline._parse_weight_list(args.fidelity_weights)
    logger.info("Weight sweep: %s", weights)

    missing = sorted(set(predictions) - set(ground_truth))
    if missing:
        logger.warning(
            "%d predicted circuits have no ground-truth entry (e.g. %s) and are excluded.",
            len(missing), ", ".join(missing[:3]),
        )

    pooled_rows = _run_sweep(ground_truth, predictions, weights, output_dir)

    summary: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "tag": tag,
        "predictions_path": str(args.predictions),
        "ground_truth_path": str(args.gt_path),
        "fidelity_weights": weights,
        "min_best_fidelity": args.min_best_fidelity,
        "n_circuits": pooled_rows[0]["n_circuits"] if pooled_rows else 0,
        "sweep": pooled_rows,
        "per_family": {},
    }

    if args.per_family:
        by_family: dict[str, list[str]] = {}
        for name in predictions:
            by_family.setdefault(family_of(name), []).append(name)

        for family, names in sorted(by_family.items()):
            if len(names) < args.min_family_size:
                logger.info("Skipping family %s (%d circuits < --min-family-size)",
                            family, len(names))
                continue
            logger.info("=== per-family scheduling: %s (%d circuits) ===", family, len(names))
            family_rows = _run_sweep(
                {n: ground_truth[n] for n in names if n in ground_truth},
                {n: predictions[n] for n in names},
                weights,
                output_dir / "by_family" / family,
            )
            summary["per_family"][family] = family_rows

        # One tidy table across families for plotting/report generation.
        flat = [
            {"family": family, **row}
            for family, rows in summary["per_family"].items()
            for row in rows
        ]
        save_csv(flat, output_dir / "sweep_summary_by_family.csv")

    save_json(summary, output_dir / "scheduling_summary.json")
    logger.info("Scheduling evaluation finished: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
