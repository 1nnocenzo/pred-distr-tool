"""Run one split end-to-end with the v2 predictor.

Writes the same artefacts as v1 (``metrics.json``, ``predictions.json``,
``history.json``, ``model.pth``), so v1's ``merge_predictions``, scheduling replay
and report code work on v2 results unchanged.

Concurrency: a split is trained only while holding an exclusive ``flock`` on
``<split>/.lock``.  A second job that reaches the same split skips it instead of
training it in parallel and overwriting the first job's files.  The lock is
released by the kernel if the process dies, so it never goes stale.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch

from .paths import ensure_v1_imports
from .training import predict, train_model, validation_split

ensure_v1_imports()

from genstudy import metrics as metrics_mod  # noqa: E402
from genstudy.data import GraphDataset, Split  # noqa: E402
from genstudy.experiment import split_is_complete  # noqa: E402,F401  (re-exported)
from genstudy.io import save_json  # noqa: E402

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def split_lock(split_dir: Path) -> Iterator[bool]:
    """Yield ``True`` if this process holds the split's lock, ``False`` if another does."""
    split_dir.mkdir(parents=True, exist_ok=True)
    handle = open(split_dir / ".lock", "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def run_split(
    dataset: GraphDataset,
    split: Split,
    devices,
    params: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    *,
    device: torch.device,
    val_mode: str,
    group_by: str | None = None,
    save_model: bool = True,
    resume: bool = True,
) -> dict[str, Any] | None:
    """Train on ``split.train``, evaluate on ``split.test``; ``None`` if locked elsewhere."""
    with split_lock(output_dir) as acquired:
        if not acquired:
            logger.warning("[%s] locked by another job — skipping", split.name)
            return None
        return _run_locked(dataset, split, devices, params, protocol, output_dir,
                           device=device, val_mode=val_mode, group_by=group_by,
                           save_model=save_model, resume=resume)


def _run_locked(dataset, split, devices, params, protocol, output_dir, *, device,
                val_mode, group_by, save_model, resume):
    logger.info("=== %s / %s ===", split.kind, split.name)
    logger.info("%s", split.description)

    train_idx, val_idx, val_info = validation_split(
        split.train, dataset.families, dataset.qubits,
        mode=val_mode, fraction=protocol["val_fraction"], seed=protocol["seed"],
    )
    logger.info("train=%d  val=%d (%s)  test=%d", len(train_idx), len(val_idx), val_info,
                len(split.test))

    devices = devices.to(device)
    model, history = train_model(
        dataset.subset(train_idx),
        [dataset.families[i] for i in train_idx],
        dataset.subset(val_idx),
        devices,
        params,
        device=device,
        protocol=protocol,
        tag=f"{split.kind}:{split.name}",
        checkpoint_path=output_dir / "checkpoint.pt",
        resume=resume,
    )

    predictions, targets, test_log_mse = predict(
        model, dataset.subset(split.test), devices,
        batch_size=protocol["batch_size"], device=device, eps=protocol["log_eps"],
    )
    test = metrics_mod.regression_metrics(predictions, targets)
    test["log_mse"] = test_log_mse

    result: dict[str, Any] = {
        "experiment": split.kind,
        "split": split.name,
        "description": split.description,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(split.test),
        "validation": val_info,
        "training": {k: v for k, v in history.items() if k != "history"},
        "test": test,
        "provenance": {
            "model": "DeviceAwarePredictor (gsv2)",
            "hyperparameters": params,
            "train_protocol": protocol,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "host": platform.node(),
        },
    }
    if group_by == "qubits":
        groups = [dataset.qubits[i] for i in split.test]
        result["test_by_qubits"] = metrics_mod.metrics_by_group(predictions, targets, groups)
    elif group_by == "family":
        groups = [dataset.families[i] for i in split.test]
        result["test_by_family"] = metrics_mod.metrics_by_group(predictions, targets, groups)

    logger.info(
        "[%s] TEST  n=%d  MAE=%.4f  R2=%.4f  device-acc=%.3f  regret=%.4f "
        "(fixed %.4f, random %.4f)",
        split.name, test["n_circuits"], test["mae"], test["r2"],
        test["device_choice_accuracy"], test["fidelity_regret_mean"],
        test["regret_best_fixed_device"], test["regret_random_device"],
    )

    save_json(result, output_dir / "metrics.json")
    save_json(
        {name: [float(v) for v in row]
         for name, row in zip(dataset.names_for(split.test), predictions)},
        output_dir / "predictions.json",
    )
    save_json({"split": asdict(split) | {"train": None, "test": None}, **history},
              output_dir / "history.json")
    if save_model:
        torch.save(model.state_dict(), output_dir / "model.pth")

    checkpoint = output_dir / "checkpoint.pt"
    if checkpoint.is_file():
        checkpoint.unlink()
    return result
