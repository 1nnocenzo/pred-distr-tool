#!/usr/bin/env python3
"""Train and evaluate a v3 predictor on the v1/v2 splits.

Experiments (same splits, seed and validation modes as v2):

``control``  random split stratified by fidelity (validation: random)
``lofo``     leave-one-family-out over the large families (validation: whole families)
``size``     train on <= n qubits, test on larger circuits (validation: largest sizes)

Each run writes to ``--results-dir`` (one directory per configuration).  Finished
splits are skipped, interrupted ones resume, and splits locked by another job
are skipped.

Examples::

    # v2 model, stabilised training
    python evaluations/generalization_v3/scripts/run_experiment.py lofo --model pooled \\
        --results-dir evaluations/generalization_v3/results/pooled_stab_a05
    # qubit cross-attention model (needs the qubit-annotated dataset)
    python evaluations/generalization_v3/scripts/run_experiment.py lofo --model xattn \\
        --results-dir evaluations/generalization_v3/results/xattn_a05
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from gsv3 import paths
from gsv3.devices import load_devices
from gsv3.experiment import run_split
from gsv3.groups import GROUPS, group_holdout_split
from gsv3.model import MODEL_KINDS, load_hparams
from gsv3.training import TRAIN_PROTOCOL

paths.ensure_imports()

from genstudy import paths as v1_paths  # noqa: E402
from genstudy.data import DEVICE_NAMES, FIGURE_OF_MERIT, GraphDataset  # noqa: E402
from genstudy.experiment import merge_predictions, split_is_complete  # noqa: E402
from genstudy.io import save_json, setup_logging  # noqa: E402
from gsv2.experiment import split_lock  # noqa: E402
from gsv2.gpu import select_device  # noqa: E402

logger = logging.getLogger("run_experiment")

EXPERIMENTS = {
    "control": ("random_control", "random"),
    "lofo": ("leave_one_family_out", "family"),
    "size": ("size_extrapolation", "size"),
    "logo": ("leave_one_group_out", "family"),
}

#: ``v1`` = the original v1 predictor (genstudy: linear target, 3 fixed outputs,
#: random validation split), run unchanged through ``genstudy.experiment.run_split``.
ALL_MODELS = (*MODEL_KINDS, "v1")


def attach_layouts(dataset, path: Path) -> None:
    """v6: ``d.layout`` = compiler layout ``(n_qubits, n_devices)``, ``-1`` where missing."""
    import torch
    layouts = torch.load(path, weights_only=False)
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        n = int(d.n_qubits)
        lay = layouts.get(name)
        if lay is not None and lay.shape == (n, len(DEVICE_NAMES)):
            d.layout, found = lay.clone(), found + 1
        else:
            d.layout = torch.full((n, len(DEVICE_NAMES)), -1, dtype=torch.long)
    logger.info("Compiler layouts (%s): %d/%d circuits", path, found, len(dataset.data))


def attach_counts(dataset, path: Path) -> None:
    """v8: ``d.counts`` = true (r, measure, cz) per device ``(1, n_devices, 3)``, ``-1`` if missing."""
    import torch
    counts = torch.load(path, weights_only=False)
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        c = counts.get(name)
        if c is not None and c.shape == (len(DEVICE_NAMES), 3):
            d.counts, found = c.unsqueeze(0).float(), found + 1
        else:
            d.counts = torch.full((1, len(DEVICE_NAMES), 3), -1.0)
    logger.info("Compiler counts (%s): %d/%d circuits", path, found, len(dataset.data))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--model", choices=ALL_MODELS, default=TRAIN_PROTOCOL["model"])
    p.add_argument("--families", default="large")
    p.add_argument("--groups", default="vqe,fourier,variational",
                   help=f"logo: comma-separated groups among {sorted(GROUPS)}.")
    p.add_argument("--min-family-size", type=int, default=50)
    p.add_argument("--max-train-qubits", default="14")
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--val-mode", choices=["random", "family", "size"], default=None)

    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Default: the v3 qubit dataset for xattn, the original one for pooled.")
    p.add_argument("--params-path", type=Path, default=None)
    p.add_argument("--device-graphs", type=Path, default=paths.V2_DEVICE_GRAPHS)

    for key in ("batch_size", "patience", "seed"):
        p.add_argument(f"--{key.replace('_', '-')}", type=int, default=TRAIN_PROTOCOL[key])
    p.add_argument("--epochs", type=int, default=TRAIN_PROTOCOL["num_epochs"])
    for key in ("val_fraction", "balance_alpha", "log_eps", "lr", "grad_clip"):
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=TRAIN_PROTOCOL[key])

    p.add_argument("--loss", choices=["log", "mixed"], default=TRAIN_PROTOCOL["loss_kind"],
                   help="log: MSE on log F (v2/v3); mixed: MSE on F + mix_lambda * MSE on log F.")
    p.add_argument("--mix-lambda", type=float, default=TRAIN_PROTOCOL["mix_lambda"])
    p.add_argument("--epoch-budget", type=int, default=None,
                   help="v5b: fixed epochs + cosine LR + SWA final model (no early stopping).")
    p.add_argument("--swa-frac", type=float, default=None,
                   help="Fraction of the budget averaged by SWA (default 0.25).")
    p.add_argument("--tau-start", type=float, default=None,
                   help="Sinkhorn temperature at epoch 1, annealed to sinkhorn_tau over 60%% of the budget.")
    p.add_argument("--layout-lambda", type=float, default=None,
                   help="v6 (sinkhorn only): weight of the cross-entropy between the placement "
                        "and the compiler's initial layout (needs --layouts-path).")
    p.add_argument("--layout-loss", choices=["assign", "region_dist"], default=None,
                   help="v6 'assign' (CE on the exact compiler layout, default) or v7 'region_dist' "
                        "(which physical qubits are used + operand distances).")
    p.add_argument("--count-lambda", type=float, default=None,
                   help="v8: weight of the loss on the predicted native-operation counts (needs --counts-path).")
    p.add_argument("--counts-path", type=Path, default=None,
                   help="compiler_counts.pt from evaluations/compile_check/compile_layouts.py counts.")
    p.add_argument("--layouts-path", type=Path, default=None,
                   help="compiler_layouts.pt from evaluations/compile_check/compile_layouts.py merge.")
    p.add_argument("--select-metric", choices=["mse", "log_mse"],
                   default=TRAIN_PROTOCOL["select_metric"],
                   help="Validation metric for early stopping / model selection.")
    p.add_argument("--results-dir", type=Path, required=True)
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
                "balance_alpha": args.balance_alpha, "log_eps": args.log_eps, "lr": args.lr,
                "grad_clip": args.grad_clip, "model": args.model,
                "select_metric": args.select_metric, "loss_kind": args.loss,
                "mix_lambda": args.mix_lambda}
    # Opt-in v5b keys: only present when requested, so older configurations (and
    # their checkpoints) are unchanged.
    for key in ("epoch_budget", "swa_frac", "tau_start", "layout_lambda", "layout_loss", "count_lambda"):
        if getattr(args, key) is not None:
            protocol[key] = getattr(args, key)
    if args.model == "v1":
        from genstudy import experiment as v1_experiment
        from genstudy import model as v1_model
        params = v1_model.load_hparams(args.params_path)
        devices = None
    else:
        params = load_hparams(args.params_path)
        qubit_aware = args.model in ("xattn", "phys", "sinkhorn", "phys_uniform")
        devices = load_devices(args.device_graphs, DEVICE_NAMES,
                               lap_pe=params["lap_pe"] if qubit_aware else 0,
                               physical_errors=args.model in ("phys", "sinkhorn", "phys_uniform"))

    if args.dataset_dir is None and args.model in ("xattn", "phys", "sinkhorn", "phys_uniform"):
        args.dataset_dir = paths.DATA_DIR
    dataset_dir = v1_paths.find_dataset_dir(args.dataset_dir, FIGURE_OF_MERIT)
    dataset = GraphDataset.load(dataset_dir, FIGURE_OF_MERIT)
    if args.layout_lambda:
        if args.model != "sinkhorn" or args.layouts_path is None:
            raise SystemExit("--layout-lambda needs --model sinkhorn and --layouts-path")
        attach_layouts(dataset, args.layouts_path)
    if args.count_lambda:
        if args.model not in ("sinkhorn", "phys", "phys_uniform") or args.counts_path is None:
            raise SystemExit("--count-lambda needs a physics-head model and --counts-path")
        attach_counts(dataset, args.counts_path)
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
    elif args.experiment == "logo":
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        splits = [group_holdout_split(dataset, g) for g in groups]
        group_by = "family"
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
        if args.model == "v1":
            with split_lock(split_dir) as acquired:
                result = v1_experiment.run_split(
                    dataset, split, params, split_dir, dataset_dir=dataset_dir,
                    device=device, seed=args.seed, group_by=group_by,
                    batch_size=args.batch_size, num_epochs=args.epochs,
                    patience=args.patience,
                    save_model=not args.no_save_model, resume=not args.overwrite,
                ) if acquired else None
            if result is None:
                logger.warning("[%s] locked by another job — skipping", split.name)
        else:
            result = run_split(dataset, split, devices, params, protocol, split_dir,
                               device=device, val_mode=val_mode, group_by=group_by,
                               save_model=not args.no_save_model,
                               resume=not args.overwrite)
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
