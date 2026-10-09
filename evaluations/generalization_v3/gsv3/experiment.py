"""Run one split end-to-end with a v3 predictor.

Same artefacts and per-split ``flock`` as v2 (``metrics.json``, ``predictions.json``,
``history.json``, ``model.pth``), so v1's ``merge_predictions`` and report code
read v3 results unchanged.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from . import paths
from .training import predict, train_model, validation_split

paths.ensure_imports()

from genstudy import metrics as metrics_mod  # noqa: E402
from genstudy.data import GraphDataset, Split  # noqa: E402
from genstudy.io import save_json  # noqa: E402
from gsv2.experiment import split_lock  # noqa: E402

logger = logging.getLogger(__name__)


def run_split(dataset: GraphDataset, split: Split, devices, params: dict[str, Any],
              protocol: dict[str, Any], output_dir: Path, *, device: torch.device,
              val_mode: str, group_by: str | None = None, save_model: bool = True,
              resume: bool = True) -> dict[str, Any] | None:
    """Train on ``split.train``, evaluate on ``split.test``; ``None`` if locked elsewhere."""
    with split_lock(output_dir) as acquired:
        if not acquired:
            logger.warning("[%s] locked by another job — skipping", split.name)
            return None

        logger.info("=== %s / %s ===", split.kind, split.name)
        logger.info("%s", split.description)
        train_idx, val_idx, val_info = validation_split(
            split.train, dataset.families, dataset.qubits,
            mode=val_mode, fraction=protocol["val_fraction"], seed=protocol["seed"])
        logger.info("train=%d  val=%d (%s)  test=%d", len(train_idx), len(val_idx), val_info,
                    len(split.test))

        devices = devices.to(device)
        model, history = train_model(
            dataset.subset(train_idx), [dataset.families[i] for i in train_idx],
            dataset.subset(val_idx), devices, params, device=device, protocol=protocol,
            tag=f"{split.kind}:{split.name}", checkpoint_path=output_dir / "checkpoint.pt",
            resume=resume)

        predictions, targets, test_log_mse = predict(
            model, dataset.subset(split.test), devices,
            batch_size=protocol["batch_size"], device=device, eps=protocol["log_eps"])
        test = metrics_mod.regression_metrics(predictions, targets)
        test["log_mse"] = test_log_mse

        result: dict[str, Any] = {
            "experiment": split.kind, "split": split.name, "description": split.description,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(split.test),
            "validation": val_info,
            "training": {k: v for k, v in history.items() if k != "history"},
            "test": test,
            "provenance": {
                "model": f"gsv3:{protocol['model']}", "hyperparameters": params,
                "train_protocol": protocol, "torch_version": torch.__version__,
                "cuda_device": (torch.cuda.get_device_name(device)
                                if device.type == "cuda" else None),
                "python": platform.python_version(), "host": platform.node(),
            },
        }
        if group_by:
            source = dataset.qubits if group_by == "qubits" else dataset.families
            groups = [source[i] for i in split.test]
            result[f"test_by_{group_by if group_by == 'qubits' else 'family'}"] = \
                metrics_mod.metrics_by_group(predictions, targets, groups)

        logger.info("[%s] TEST  n=%d  MAE=%.4f  R2=%.4f  device-acc=%.3f  regret=%.4f",
                    split.name, test["n_circuits"], test["mae"], test["r2"],
                    test["device_choice_accuracy"], test["fidelity_regret_mean"])

        save_json(result, output_dir / "metrics.json")
        save_json({name: [float(v) for v in row]
                   for name, row in zip(dataset.names_for(split.test), predictions)},
                  output_dir / "predictions.json")
        save_json({"split": asdict(split) | {"train": None, "test": None}, **history},
                  output_dir / "history.json")
        if save_model:
            torch.save(model.state_dict(), output_dir / "model.pth")
        checkpoint = output_dir / "checkpoint.pt"
        if checkpoint.is_file():
            checkpoint.unlink()
        return result
