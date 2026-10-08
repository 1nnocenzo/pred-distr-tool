#!/usr/bin/env python3
"""Train and evaluate the v2 predictor on the v1 study's splits.

Experiments (same splits and seed as ``evaluations/generalization``):

``control``  random split stratified by fidelity (validation: random)
``lofo``     leave-one-family-out over the ten large families (validation: whole families)
``size``     train on <= n qubits, test on larger circuits (validation: largest training sizes)

Uses the GPU when one is usable, the CPU otherwise.  Splits whose
``metrics.json`` exists are skipped, interrupted splits resume from their
per-epoch checkpoint, and a split being trained by another job is skipped.

Examples::

    python evaluations/generalization_v2/scripts/run_experiment.py control
    python evaluations/generalization_v2/scripts/run_experiment.py lofo --families qaoa,vqe_su2
    python evaluations/generalization_v2/scripts/run_experiment.py size --max-train-qubits 14
    # 2-epoch smoke test, into a separate results tree
    python evaluations/generalization_v2/scripts/run_experiment.py lofo --families qft \\
        --epochs 2 --patience 1 --results-dir evaluations/generalization_v2/results_smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from gsv2 import paths
from gsv2.devices import load_device_batch
from gsv2.experiment import run_split, split_is_complete
from gsv2.gpu import select_device
from gsv2.model import load_hparams
from gsv2.training import TRAIN_PROTOCOL

paths.ensure_v1_imports()

from genstudy import paths as v1_paths  # noqa: E402
from genstudy.data import DEVICE_NAMES, FIGURE_OF_MERIT, GraphDataset  # noqa: E402
from genstudy.experiment import merge_predictions  # noqa: E402
from genstudy.io import save_json, setup_logging  # noqa: E402

logger = logging.getLogger("run_experiment")

#: Directory name and default validation mode per experiment.
EXPERIMENTS = {
    "control": ("random_control", "random"),
    "lofo": ("leave_one_family_out", "family"),
    "size": ("size_extrapolation", "size"),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--families", default="large",
                   help="lofo: comma-separated families, 'large' or 'all'.")
    p.add_argument("--min-family-size", type=int, default=50)
    p.add_argument("--max-train-qubits", default="14", help="size: comma-separated cut-offs.")
    p.add_argument("--test-fraction", type=float, default=0.3, help="control: test fraction.")
    p.add_argument("--val-mode", choices=["random", "family", "size"], default=None,
                   help="Override the experiment's default validation mode.")

    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--figure-of-merit", default=FIGURE_OF_MERIT)
    p.add_argument("--params-path", type=Path, default=None)
    p.add_argument("--device-graphs", type=Path, default=paths.DEVICE_GRAPHS_PATH)

    for key in ("batch_size", "patience", "seed"):
        p.add_argument(f"--{key.replace('_', '-')}", type=int, default=TRAIN_PROTOCOL[key])
    p.add_argument("--epochs", type=int, default=TRAIN_PROTOCOL["num_epochs"])
    for key in ("val_fraction", "balance_alpha", "log_eps"):
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=TRAIN_PROTOCOL[key])

    p.add_argument("--results-dir", type=Path, default=paths.RESULTS_DIR)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-save-model", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    exp_name, default_val_mode = EXPERIMENTS[args.experiment]
    val_mode = args.val_mode or default_val_mode

    experiment_dir = args.results_dir / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(experiment_dir / "run.log")
    logger.info("Command line: %s", " ".join(sys.argv))

    device = select_device()

    protocol = {**TRAIN_PROTOCOL, "batch_size": args.batch_size, "num_epochs": args.epochs,
                "patience": args.patience, "seed": args.seed, "val_fraction": args.val_fraction,
                "balance_alpha": args.balance_alpha, "log_eps": args.log_eps}
    params = load_hparams(args.params_path)
    devices = load_device_batch(args.device_graphs, DEVICE_NAMES)

    dataset_dir = v1_paths.find_dataset_dir(args.dataset_dir, args.figure_of_merit)
    dataset = GraphDataset.load(dataset_dir, args.figure_of_merit)
    save_json(dataset.summary(), experiment_dir / "dataset_summary.json")

    if args.experiment == "control":
        splits = [dataset.random_split(test_fraction=args.test_fraction, seed=args.seed)]
        group_by = "family"
    elif args.experiment == "lofo":
        requested = None if args.families in ("", "large") else tuple(
            f.strip() for f in args.families.split(",") if f.strip())
        families = dataset.select_families(requested, min_size=args.min_family_size)
        splits = [dataset.family_holdout_split(f) for f in families]
        group_by = "qubits"
    else:
        lo, hi = dataset.qubit_range()
        cutoffs = [int(v) for v in args.max_train_qubits.split(",") if v.strip()]
        splits = [dataset.size_holdout_split(n) for n in cutoffs if lo <= n < hi]
        group_by = "qubits"

    completed, skipped = [], []
    for position, split in enumerate(splits, start=1):
        split_dir = experiment_dir / split.name
        if not args.overwrite and split_is_complete(split_dir):
            logger.info("[%d/%d] %s already done — skipping", position, len(splits), split.name)
            completed.append(split.name)
            continue
        logger.info("[%d/%d] %s", position, len(splits), split.name)
        result = run_split(dataset, split, devices, params, protocol, split_dir,
                           device=device, val_mode=val_mode, group_by=group_by,
                           save_model=not args.no_save_model, resume=not args.overwrite)
        (completed if result is not None else skipped).append(split.name)

    if skipped:
        logger.warning("Splits held by another job (not run here): %s", ", ".join(skipped))
    if args.experiment == "lofo" and not skipped:
        save_json({"experiment": exp_name, "families": completed, "val_mode": val_mode},
                  experiment_dir / "folds.json")
        save_json(merge_predictions([experiment_dir / f for f in completed]),
                  experiment_dir / "heldout_predictions.json")
    logger.info("%s finished: %d complete, %d skipped", exp_name, len(completed), len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
