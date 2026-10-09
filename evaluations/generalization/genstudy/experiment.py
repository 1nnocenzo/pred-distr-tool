"""Run one split end-to-end: train, evaluate, persist.

A single function, :func:`run_split`, is shared by all three experiment drivers
(leave-one-family-out, size extrapolation, random-split control) so that the
three conditions differ *only* in how the split is constructed.

Each split writes a self-contained directory::

    <results>/<experiment>/<split-name>/
        metrics.json       metrics + split description + provenance
        predictions.json   {"family/name": [fid_EQE1_Top, fid_EQE1_Bottom, fid_QExa20]}
        model.pth          weights of the selected epoch
        history.json       per-epoch train/validation losses

``predictions.json`` is the input of ``run_scheduling_eval.py``: because each
prediction comes from a model that never saw the circuit's family (or its size),
replaying the scheduler on those files measures the policy on genuinely unseen
workloads.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import metrics as metrics_mod
from . import model as model_mod
from .data import GraphDataset, Split
from .io import save_json

logger = logging.getLogger(__name__)

#: Files that must exist for a split directory to count as complete.
_REQUIRED_OUTPUTS = ("metrics.json", "predictions.json")


def split_is_complete(split_dir: Path) -> bool:
    """True if this split has already been run (used by ``--resume``)."""
    return all((split_dir / name).is_file() for name in _REQUIRED_OUTPUTS)


def _provenance(dataset_dir: Path, params: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir),
        "hyperparameters": params,
        "train_protocol": model_mod.TRAIN_PROTOCOL,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "host": platform.node(),
    }


def run_split(
    dataset: GraphDataset,
    split: Split,
    params: dict[str, Any],
    output_dir: Path,
    *,
    dataset_dir: Path,
    device: torch.device,
    batch_size: int = model_mod.TRAIN_PROTOCOL["batch_size"],
    num_epochs: int = model_mod.TRAIN_PROTOCOL["num_epochs"],
    patience: int = model_mod.TRAIN_PROTOCOL["patience"],
    val_fraction: float = model_mod.TRAIN_PROTOCOL["val_fraction"],
    seed: int = model_mod.TRAIN_PROTOCOL["seed"],
    group_by: str | None = None,
    save_model: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Train on ``split.train``, evaluate on ``split.test``, write the artefacts.

    Args:
        dataset: The full graph dataset.
        split: Train/test partition to run.
        params: Tuned hyper-parameters; used as-is, never re-tuned.
        output_dir: Directory for this split's artefacts.
        dataset_dir: Recorded for provenance.
        device: Compute device.
        group_by: Optional secondary breakdown of the test metrics —
            ``"qubits"`` (per test qubit count) or ``"family"`` (per test family).
        save_model: Persist the selected weights.
        resume: Continue from ``checkpoint.pt`` if this split was interrupted
            mid-training (set ``False`` for ``--overwrite``).

    Returns:
        The metrics dictionary that was written to ``metrics.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s / %s ===", split.kind, split.name)
    logger.info("%s", split.description)

    train_idx, val_idx = GraphDataset.train_val_indices(split.train, val_fraction, seed)
    logger.info(
        "train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(split.test)
    )

    trained, history = model_mod.train_model(
        dataset.subset(train_idx),
        dataset.subset(val_idx),
        params,
        device=device,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=patience,
        seed=seed,
        tag=f"{split.kind}:{split.name}",
        checkpoint_path=output_dir / "checkpoint.pt",
        resume=resume,
    )

    test_graphs = dataset.subset(split.test)
    test_names = dataset.names_for(split.test)
    predictions, targets = model_mod.predict(
        trained, test_graphs, batch_size=batch_size, device=device
    )

    result: dict[str, Any] = {
        "experiment": split.kind,
        "split": split.name,
        "description": split.description,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(split.test),
        "training": {k: v for k, v in history.items() if k != "history"},
        "test": metrics_mod.regression_metrics(predictions, targets),
        "provenance": _provenance(dataset_dir, params, device),
    }

    if group_by == "qubits":
        groups = [dataset.qubits[i] for i in split.test]
        result["test_by_qubits"] = metrics_mod.metrics_by_group(predictions, targets, groups)
    elif group_by == "family":
        groups = [dataset.families[i] for i in split.test]
        result["test_by_family"] = metrics_mod.metrics_by_group(predictions, targets, groups)

    test_metrics = result["test"]
    logger.info(
        "[%s] TEST  n=%d  MAE=%.4f  RMSE=%.4f  R2=%.4f  device-acc=%.3f  regret=%.4f",
        split.name, test_metrics["n_circuits"], test_metrics["mae"],
        test_metrics["rmse"], test_metrics["r2"],
        test_metrics["device_choice_accuracy"], test_metrics["fidelity_regret_mean"],
    )

    save_json(result, output_dir / "metrics.json")
    save_json(
        {name: [float(v) for v in row] for name, row in zip(test_names, predictions)},
        output_dir / "predictions.json",
    )
    save_json({"split": asdict(split) | {"train": None, "test": None}, **history},
              output_dir / "history.json")
    if save_model:
        torch.save(trained.state_dict(), output_dir / "model.pth")
        logger.info("Wrote %s", output_dir / "model.pth")

    # The split is complete and its artefacts are on disk, so the mid-training
    # checkpoint is no longer needed (it is the largest file this split writes).
    checkpoint = output_dir / "checkpoint.pt"
    if checkpoint.is_file():
        checkpoint.unlink()
        logger.info("Removed %s (split finished)", checkpoint)

    return result


def merge_predictions(split_dirs: list[Path]) -> dict[str, list[float]]:
    """Concatenate the held-out predictions of several splits.

    Every circuit is predicted by exactly one model — the one that never saw its
    family — so the union is a leakage-free prediction for the whole dataset.
    Duplicate circuits (possible only if splits overlap) are reported and the
    first occurrence wins.
    """
    from .io import load_json

    merged: dict[str, list[float]] = {}
    duplicates = 0
    for split_dir in split_dirs:
        path = split_dir / "predictions.json"
        if not path.is_file():
            logger.warning("No predictions in %s; skipped", split_dir)
            continue
        for name, fids in load_json(path).items():
            if name in merged:
                duplicates += 1
                continue
            merged[name] = [float(v) for v in fids]
    if duplicates:
        logger.warning("%d circuits appeared in more than one split; kept first", duplicates)
    logger.info("Merged held-out predictions for %d circuits", len(merged))
    return merged


def predictions_array(
    predictions: dict[str, list[float]], names: list[str]
) -> np.ndarray:
    """Stack a prediction dict into an ``(n, 3)`` array following ``names``."""
    return np.asarray([predictions[name] for name in names], dtype=np.float64)
