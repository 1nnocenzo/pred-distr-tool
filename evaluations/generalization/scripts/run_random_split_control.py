#!/usr/bin/env python3
"""Random-split control for the generalization study.

Reproduces the original split — random over circuits, stratified by fidelity —
and trains it with *this* code, the same frozen protocol and the same
hyper-parameters as the leave-one-family-out and size-extrapolation runs.

Its only purpose is to make the comparison fair: the published in-distribution
score was produced by a different driver, so quoting it next to the held-out
numbers would confound the split with the training code.  This control removes
that confound and provides the reference row of the generalization tables.

Outputs (under ``--results-dir/random_control/``)::

    random_seed<seed>/metrics.json      metrics, with a per-family breakdown in
                                        "test_by_family"
    random_seed<seed>/predictions.json  predictions on the random test split
    random_seed<seed>/model.pth         weights of the selected epoch

Example::

    python evaluations/generalization/scripts/run_random_split_control.py
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

EXPERIMENT = "random_control"

logger = logging.getLogger("run_random_split_control")


def build_parser():
    parser = common_parser(__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--test-fraction", type=float, default=0.3,
        help="Test fraction of the random split (the published split used 0.3).",
    )
    parser.add_argument(
        "--stratify-bins", type=int, default=5,
        help="Number of fidelity quantile bins used for stratification.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    experiment_dir = args.results_dir / EXPERIMENT
    experiment_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(experiment_dir / "run.log")
    logger.info("Command line: %s", " ".join(sys.argv))

    dataset_dir = paths.find_dataset_dir(args.dataset_dir, args.figure_of_merit)
    params = load_hparams(args.params_path)
    device = resolve_device(args.device)
    logger.info("Dataset directory: %s | device: %s", dataset_dir, device)

    dataset = GraphDataset.load(dataset_dir, args.figure_of_merit)
    split = dataset.random_split(
        test_fraction=args.test_fraction,
        seed=args.seed,
        stratify_bins=args.stratify_bins,
    )

    split_dir = experiment_dir / split.name
    if not args.overwrite and split_is_complete(split_dir):
        logger.info("%s already done — nothing to do (use --overwrite to redo)", split.name)
        return 0

    run_split(
        dataset,
        split,
        params,
        split_dir,
        dataset_dir=dataset_dir,
        device=device,
        group_by="family",
        **training_kwargs(args),
    )
    save_json(
        {
            "experiment": EXPERIMENT,
            "split": split.name,
            "test_fraction": args.test_fraction,
            "dataset_dir": str(dataset_dir),
        },
        experiment_dir / "control.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
