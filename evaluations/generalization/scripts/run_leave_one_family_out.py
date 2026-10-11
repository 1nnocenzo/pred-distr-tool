#!/usr/bin/env python3
"""Leave-one-family-out evaluation of the fidelity predictor.

For each selected benchmark family, the predictor is trained on *all other*
families and tested on the held-out one, so no near-duplicate of a test circuit
can appear in training.  The tuned hyper-parameters are used unchanged for every
fold — nothing is re-tuned on the held-out family.

Outputs (under ``--results-dir/leave_one_family_out/``)::

    <family>/metrics.json        MAE, RMSE, R^2 overall and per device
    <family>/predictions.json    held-out per-device fidelity predictions
    <family>/model.pth           weights of the selected epoch
    folds.json                   index of completed folds + dataset summary
    heldout_predictions.json     union over folds, one prediction per circuit

``heldout_predictions.json`` is what ``run_scheduling_eval.py`` replays through
the scheduling benchmark.

Examples::

    # All ten large families (default), resuming an interrupted job
    python evaluations/generalization/scripts/run_leave_one_family_out.py

    # One fold only — useful to spread folds over several jobs
    python evaluations/generalization/scripts/run_leave_one_family_out.py --families qaoa

    # Every family in the dataset that has at least 100 circuits
    python evaluations/generalization/scripts/run_leave_one_family_out.py \\
        --families all --min-family-size 100

    # Fast smoke test of the whole code path
    python evaluations/generalization/scripts/run_leave_one_family_out.py \\
        --families qft --epochs 2 --patience 1 --no-save-model
"""

from __future__ import annotations

import logging
import sys

import _bootstrap  # noqa: F401  (puts the study package on sys.path)

from genstudy import paths
from genstudy.cli import common_parser, training_kwargs
from genstudy.data import GraphDataset, LARGE_FAMILIES
from genstudy.experiment import merge_predictions, run_split, split_is_complete
from genstudy.io import save_json, setup_logging
from genstudy.model import load_hparams, resolve_device

EXPERIMENT = "leave_one_family_out"

logger = logging.getLogger("run_leave_one_family_out")


def build_parser():
    parser = common_parser(__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--families", default="large",
        help=(
            "Comma-separated families to hold out, or 'large' for "
            f"[{', '.join(LARGE_FAMILIES)}], or 'all' for every family present."
        ),
    )
    parser.add_argument(
        "--min-family-size", type=int, default=50,
        help="Skip families with fewer circuits than this (R^2 is meaningless on "
             "a handful of near-constant targets).",
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

    requested = None if args.families in ("", "large") else tuple(
        f.strip() for f in args.families.split(",") if f.strip()
    )
    families = dataset.select_families(requested, min_size=args.min_family_size)

    completed: list[str] = []
    for position, family in enumerate(families, start=1):
        split_dir = experiment_dir / family
        if not args.overwrite and split_is_complete(split_dir):
            logger.info("[%d/%d] %s already done — skipping (use --overwrite to redo)",
                        position, len(families), family)
            completed.append(family)
            continue

        logger.info("[%d/%d] holding out family '%s'", position, len(families), family)
        run_split(
            dataset,
            dataset.family_holdout_split(family),
            params,
            split_dir,
            dataset_dir=dataset_dir,
            device=device,
            group_by="qubits",
            **training_kwargs(args),
        )
        completed.append(family)

    save_json(
        {
            "experiment": EXPERIMENT,
            "families": completed,
            "min_family_size": args.min_family_size,
            "dataset_dir": str(dataset_dir),
        },
        experiment_dir / "folds.json",
    )

    # Union of the folds: every circuit predicted by a model blind to its family.
    merged = merge_predictions([experiment_dir / family for family in completed])
    save_json(merged, experiment_dir / "heldout_predictions.json")

    logger.info("Leave-one-family-out finished: %d folds, %d held-out predictions",
                len(completed), len(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
