"""Regression metrics reported by the generalization study.

Every experiment reports the same quantities so that leave-one-family-out, size
extrapolation and the random-split control are directly comparable:

* ``mae``, ``rmse`` — over all circuit/device pairs of the test set,
* ``r2``            — coefficient of determination, averaged uniformly over the
  three device outputs,
* ``per_device``    — the same three metrics for each device separately,
* ``device_choice_accuracy`` — how often the predictor's arg-max device equals
  the ground-truth best device, i.e. the quantity the scheduler ultimately acts
  on,
* ``fidelity_regret_mean`` — mean fidelity lost by following the predictor's
  arg-max device instead of the true best device,
* ``regret_best_fixed_device`` / ``regret_random_device`` — the same regret for
  two predictor-free baselines (see :func:`baseline_regrets`).  The devices
  differ little in fidelity, so every regret is small in absolute terms and the
  predictor's regret is only meaningful next to these two.

``r2`` is computed per output and then averaged; for a test set whose fidelities
are nearly constant (which happens for some small families) the variance in the
denominator is tiny and R^2 becomes unstable, so ``target_std`` is reported
alongside it to make such cases visible rather than silently misleading.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .data import DEVICE_NAMES


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Compute the full metric block for one test set.

    Args:
        predictions: ``(n, 3)`` predicted fidelities, device order ``DEVICE_NAMES``.
        targets: ``(n, 3)`` ground-truth fidelities.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(f"shape mismatch: {predictions.shape} vs {targets.shape}")
    if predictions.ndim != 2 or predictions.shape[1] != len(DEVICE_NAMES):
        raise ValueError(f"expected (n, {len(DEVICE_NAMES)}) arrays, got {predictions.shape}")

    flat_t, flat_p = targets.reshape(-1), predictions.reshape(-1)
    best_true = targets.argmax(axis=1)
    best_pred = predictions.argmax(axis=1)
    rows = np.arange(len(targets))

    metrics: dict[str, Any] = {
        "n_circuits": int(targets.shape[0]),
        "mae": float(mean_absolute_error(flat_t, flat_p)),
        "rmse": float(np.sqrt(mean_squared_error(flat_t, flat_p))),
        "r2": float(r2_score(targets, predictions, multioutput="uniform_average")),
        "target_std": float(flat_t.std()),
        "device_choice_accuracy": float(accuracy_score(best_true, best_pred)),
        "fidelity_regret_mean": float(
            np.mean(targets[rows, best_true] - targets[rows, best_pred])
        ),
        "mean_true_best_fidelity": float(targets[rows, best_true].mean()),
        **baseline_regrets(targets),
        "per_device": {},
    }

    for i, device in enumerate(DEVICE_NAMES):
        metrics["per_device"][device] = {
            "mae": float(mean_absolute_error(targets[:, i], predictions[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))),
            "r2": float(r2_score(targets[:, i], predictions[:, i])),
            "target_std": float(targets[:, i].std()),
        }
    return metrics


def baseline_regrets(targets: np.ndarray) -> dict[str, Any]:
    """Regret of two predictor-free device-choice baselines on one test set.

    * ``regret_best_fixed_device`` — always send every circuit to the single
      device with the highest mean fidelity *on this test set*.  The device is
      chosen a posteriori, so this is a deliberately strong baseline.
    * ``regret_random_device`` — expected regret of a uniformly random device
      choice, i.e. best fidelity minus the mean over the devices.
    """
    targets = np.asarray(targets, dtype=np.float64)
    best = targets.max(axis=1)
    fixed = int(targets.mean(axis=0).argmax())
    return {
        "best_fixed_device": DEVICE_NAMES[fixed],
        "regret_best_fixed_device": float(np.mean(best - targets[:, fixed])),
        "regret_random_device": float(np.mean(best - targets.mean(axis=1))),
    }


def metrics_by_group(
    predictions: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[Any],
    *,
    min_size: int = 2,
) -> dict[str, dict[str, Any]]:
    """Per-group metric blocks, e.g. one per test qubit count or per family.

    Groups with fewer than ``min_size`` circuits are skipped: R^2 is undefined
    for a single sample.  Keys are stringified group labels, sorted so that
    integer labels (qubit counts) come out in numeric order.
    """
    labels = list(groups)
    if len(labels) != len(targets):
        raise ValueError(f"{len(labels)} group labels for {len(targets)} rows")

    unique = sorted(set(labels), key=lambda g: (isinstance(g, str), g))
    out: dict[str, dict[str, Any]] = {}
    for group in unique:
        mask = np.array([g == group for g in labels], dtype=bool)
        if mask.sum() < min_size:
            continue
        out[str(group)] = regression_metrics(predictions[mask], targets[mask])
    return out
