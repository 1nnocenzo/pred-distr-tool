"""Shared command-line arguments for the experiment drivers.

Every driver accepts the same training/IO flags so that a run can be reproduced
from the command line recorded in its log, and so a smoke test (``--epochs 2``)
exercises exactly the code path of the full run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import paths
from .data import FIGURE_OF_MERIT
from .model import TRAIN_PROTOCOL


def common_parser(description: str) -> argparse.ArgumentParser:
    """Build a parser with the dataset, training and output flags."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("dataset")
    data_group.add_argument(
        "--dataset-dir", type=Path, default=None,
        help=(
            "Directory holding graph_dataset_<fom>.pt and names_list_<fom>.npy "
            f"(the compileCircuits folder). Defaults to ${paths.DATASET_DIR_ENV} "
            "or a repository-relative search."
        ),
    )
    data_group.add_argument(
        "--figure-of-merit", default=FIGURE_OF_MERIT,
        help="Dataset flavour, i.e. the graph_dataset_<fom>.pt suffix.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--params-path", type=Path, default=None,
        help="Tuned hyper-parameters JSON (default: src/model/best_params.json). "
             "Never re-tuned on held-out data.",
    )
    model_group.add_argument(
        "--device", default=None, help="Torch device (default: cuda when available)."
    )

    train_group = parser.add_argument_group("training protocol")
    train_group.add_argument("--batch-size", type=int, default=TRAIN_PROTOCOL["batch_size"])
    train_group.add_argument("--epochs", type=int, default=TRAIN_PROTOCOL["num_epochs"])
    train_group.add_argument("--patience", type=int, default=TRAIN_PROTOCOL["patience"])
    train_group.add_argument("--val-fraction", type=float, default=TRAIN_PROTOCOL["val_fraction"])
    train_group.add_argument("--seed", type=int, default=TRAIN_PROTOCOL["seed"])

    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--results-dir", type=Path, default=paths.RESULTS_DIR,
        help="Root for this study's results; the pipeline results are never touched.",
    )
    out_group.add_argument(
        "--overwrite", action="store_true",
        help="Re-run splits whose results already exist (default: skip them, so an "
             "interrupted job can be resubmitted unchanged).",
    )
    out_group.add_argument(
        "--no-save-model", action="store_true",
        help="Do not write model.pth for each split (saves disk space).",
    )
    return parser


def training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the training flags in the form :func:`experiment.run_split` expects."""
    return {
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "patience": args.patience,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "save_model": not args.no_save_model,
        # --overwrite means "redo this split from scratch", so an interrupted
        # training state from a previous attempt must not be picked up.
        "resume": not args.overwrite,
    }
