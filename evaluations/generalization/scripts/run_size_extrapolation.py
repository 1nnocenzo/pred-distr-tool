#!/usr/bin/env python3
"""Size-extrapolation evaluation of the fidelity predictor.

The predictor is trained only on circuits with at most *n* qubits and tested on
the larger ones (up to the dataset maximum, 20 qubits for the 30k benchmark set),
which measures whether the learned fidelity model transfers to circuit sizes it
has never seen.  Hyper-parameters are the tuned ones, unchanged.

Outputs (under ``--results-dir/size_extrapolation/``)::

    qmax_<n>/metrics.json       metrics on all test circuits, plus a per-size
                                breakdown in "test_by_qubits"
    qmax_<n>/predictions.json   held-out predictions for the larger circuits
    qmax_<n>/model.pth          weights of the selected epoch
    thresholds.json             index of completed thresholds

Examples::

    # The configuration reported in the paper: train on <= 14 qubits
    python evaluations/generalization/scripts/run_size_extrapolation.py

    # Several cut-offs in one job
    python evaluations/generalization/scripts/run_size_extrapolation.py \\
        --max-train-qubits 12,14,16
"""

from __future__ import annotations

import logging
import sys

import _bootstrap  # noqa: F401

from genstudy import paths
from genstudy.cli import common_parser, training_kwargs
from genstudy.data import GraphDataset
from genstudy.experiment import run_split, split_is_complete
from genstudy.io import save_json, setup_logging
from genstudy.model import load_hparams, resolve_device

EXPERIMENT = "size_extrapolation"

logger = logging.getLogger("run_size_extrapolation")


def build_parser():
    parser = common_parser(__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--max-train-qubits", default="14",
        help="Comma-separated qubit cut-offs n: train on circuits with <= n qubits, "
             "test on circuits with > n qubits.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    experiment_dir = args.results_dir / EXPERIMENT
    experiment_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(experiment_dir / "run.log")
    logger.info("Command line: %s", " ".join(sys.argv))

    dataset_dir = paths.find_dataset_dir(args.dataset_dir, args.figure_of_merit)
    logger.info("Dataset directory: %s", dataset_dir)

    params = load_hparams(args.params_path)
    device = resolve_device(args.device)
    logger.info("Compute device: %s", device)

    dataset = GraphDataset.load(dataset_dir, args.figure_of_merit)
    save_json(dataset.summary(), experiment_dir / "dataset_summary.json")

    lo, hi = dataset.qubit_range()
    logger.info("Dataset covers %d-%d qubits", lo, hi)

    thresholds = [int(v.strip()) for v in args.max_train_qubits.split(",") if v.strip()]
    completed: list[int] = []

    for position, max_qubits in enumerate(thresholds, start=1):
        if not lo <= max_qubits < hi:
            logger.warning(
                "Cut-off n=%d leaves an empty train or test side for a %d-%d qubit "
                "dataset — skipping.", max_qubits, lo, hi,
            )
            continue

        split = dataset.size_holdout_split(max_qubits)
        split_dir = experiment_dir / split.name
        if not args.overwrite and split_is_complete(split_dir):
            logger.info("[%d/%d] %s already done — skipping (use --overwrite to redo)",
                        position, len(thresholds), split.name)
            completed.append(max_qubits)
            continue

        logger.info("[%d/%d] training on <= %d qubits, testing on %d-%d qubits",
                    position, len(thresholds), max_qubits, max_qubits + 1, hi)
        run_split(
            dataset,
            split,
            params,
            split_dir,
            dataset_dir=dataset_dir,
            device=device,
            group_by="qubits",
            **training_kwargs(args),
        )
        completed.append(max_qubits)

    save_json(
        {
            "experiment": EXPERIMENT,
            "max_train_qubits": completed,
            "dataset_qubit_range": [lo, hi],
            "dataset_dir": str(dataset_dir),
        },
        experiment_dir / "thresholds.json",
    )
    logger.info("Size extrapolation finished for cut-offs: %s", completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
