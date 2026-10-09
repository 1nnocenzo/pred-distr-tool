"""Training protocol of the v2 predictor.

Differences from the v1 protocol (``genstudy.model.TRAIN_PROTOCOL``):

* **Log target.**  The loss is the MSE between the predicted ``log F`` and
  ``log(max(F, eps))``.  Fidelities below ``eps`` (default 1e-3) are equivalent
  for scheduling and are clipped so they do not dominate the loss.
* **Family-balanced sampling.**  Each training circuit is drawn with weight
  ``n_family ** -alpha`` (default ``alpha = 0.5``), so the three VQE families
  (about half of the data) no longer dictate the learned device preferences,
  while the tiny families are not repeated hundreds of times per epoch.
* **Out-of-distribution validation.**  Early stopping is driven by a validation
  set that mimics the test condition: whole medium/small families
  (``val_mode="family"``; families above half the validation budget are never
  used, so the large families stay in training),
  the largest training circuits (``"size"``), or a random subset (``"random"``,
  used only by the in-distribution control).

Optimiser, batch size, epoch budget and patience are unchanged from v1.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from .model import build_model

logger = logging.getLogger(__name__)

TRAIN_PROTOCOL: dict[str, Any] = {
    "optimizer": "adam",
    "loss": "mse_log_fidelity",
    "log_eps": 1e-3,
    "batch_size": 32,
    "num_epochs": 1000,
    "patience": 30,
    "val_fraction": 0.2,
    "balance_alpha": 0.5,
    "seed": 5,
}


# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------

def validation_split(
    train: Sequence[int],
    families: Sequence[str],
    qubits: Sequence[int | None],
    *,
    mode: str,
    fraction: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Split training positions into (train, val) according to ``mode``.

    Returns the two index lists and a description recorded in ``metrics.json``.
    """
    train = list(train)
    target = fraction * len(train)
    rng = np.random.default_rng(seed)

    if mode == "random":
        perm = np.array(train)
        rng.shuffle(perm)
        n_val = max(1, int(round(target)))
        return perm[n_val:].tolist(), perm[:n_val].tolist(), {"mode": "random"}

    if mode == "family":
        sizes = Counter(families[i] for i in train)
        # Families larger than half the validation budget stay in training: they
        # would make validation a single family, and removing one of the large
        # VQE siblings would change the training data far more than v1's split.
        order = sorted(f for f, n in sizes.items() if n <= 0.5 * target)
        rng.shuffle(order)
        chosen, total = [], 0
        for fam in order:
            if total >= target:
                break
            # Skip a family that would overshoot the target by more than half.
            if chosen and total + sizes[fam] > 1.5 * target:
                continue
            chosen.append(fam)
            total += sizes[fam]
        val_set = set(chosen)
        val = [i for i in train if families[i] in val_set]
        rest = [i for i in train if families[i] not in val_set]
        return rest, val, {"mode": "family", "val_families": sorted(chosen)}

    if mode == "size":
        sizes = Counter(qubits[i] for i in train if qubits[i] is not None)
        chosen, total = [], 0
        for q in sorted(sizes, reverse=True):
            if total >= target:
                break
            chosen.append(q)
            total += sizes[q]
        val_set = set(chosen)
        val = [i for i in train if qubits[i] in val_set]
        rest = [i for i in train if qubits[i] not in val_set]
        return rest, val, {"mode": "size", "val_qubits": sorted(chosen)}

    raise ValueError(f"unknown validation mode {mode!r}")


def balanced_sampler(
    families: Sequence[str], alpha: float, seed: int
) -> WeightedRandomSampler | None:
    """Sampler with per-circuit weight ``n_family ** -alpha`` (``None`` if ``alpha == 0``)."""
    if alpha == 0:
        return None
    counts = Counter(families)
    weights = torch.tensor([counts[f] ** -alpha for f in families], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(families), replacement=True,
                                 generator=generator)


# ---------------------------------------------------------------------------
# Loss and inference
# ---------------------------------------------------------------------------

def log_target(y: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log(torch.clamp(y, min=eps))


@torch.no_grad()
def predict(model, graphs, devices, *, batch_size: int, device: torch.device,
            eps: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(fidelity predictions, targets, log-space MSE)``.

    Predictions are ``exp(min(log F, 0))``, i.e. fidelities in ``(0, 1]``.
    """
    model.eval()
    loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False)
    preds, targets, sq, n = [], [], 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        log_pred = torch.clamp(model(batch, devices), max=0.0)
        y = batch.y.float().view(log_pred.shape)
        sq += float(((log_pred - log_target(y, eps)) ** 2).sum())
        n += y.numel()
        preds.append(torch.exp(log_pred).cpu())
        targets.append(y.cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), sq / max(1, n)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write; the temp name is per-process so concurrent writers cannot collide."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path, model, optimizer, device, config: dict[str, Any]):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("config") != config:
            logger.warning("Checkpoint %s was written with a different configuration; "
                           "training from scratch", path)
            return None
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        return payload
    except Exception:
        logger.exception("Ignoring unusable checkpoint %s; training from scratch", path)
        return None


def train_model(
    train_graphs: Sequence[Any],
    train_families: Sequence[str],
    val_graphs: Sequence[Any],
    devices,
    params: dict[str, Any],
    *,
    device: torch.device,
    protocol: dict[str, Any],
    tag: str,
    checkpoint_path: Path | None,
    resume: bool = True,
):
    """Train under ``protocol``; restore the best-validation weights."""
    seed = protocol["seed"]
    eps = protocol["log_eps"]
    torch.manual_seed(seed)
    model = build_model(params, devices, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    sampler = balanced_sampler(train_families, protocol["balance_alpha"], seed)
    loader = DataLoader(list(train_graphs), batch_size=protocol["batch_size"],
                        shuffle=sampler is None, sampler=sampler)

    # Identifies the run a checkpoint belongs to; the epoch budget is excluded so a
    # resubmission with a larger --epochs still resumes.
    config = {"params": params,
              "protocol": {k: v for k, v in protocol.items() if k != "num_epochs"},
              "n_train": len(train_graphs), "n_val": len(val_graphs)}
    best_val, best_epoch, best_state, no_improve = float("inf"), 0, None, 0
    history: list[dict[str, float]] = []
    start_epoch = 1

    if checkpoint_path is not None and resume and checkpoint_path.is_file():
        state = _load_checkpoint(checkpoint_path, model, optimizer, device, config)
        if state is not None:
            start_epoch = state["epoch"] + 1
            best_val, best_epoch = state["best_val"], state["best_epoch"]
            best_state, no_improve = state["best_state"], state["no_improve"]
            history = state["history"]
            logger.info("[%s] resumed at epoch %d (best %d, val=%.5f, patience %d/%d)",
                        tag, state["epoch"], best_epoch, best_val, no_improve,
                        protocol["patience"])

    logger.info("[%s] train=%d val=%d, epochs %d-%d, patience %d, balance alpha %.2f",
                tag, len(train_graphs), len(val_graphs), start_epoch,
                protocol["num_epochs"], protocol["patience"], protocol["balance_alpha"])

    epoch = start_epoch - 1
    for epoch in range(start_epoch, protocol["num_epochs"] + 1):
        if no_improve >= protocol["patience"]:
            break
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            log_pred = model(batch, devices)
            target = log_target(batch.y.float().view(log_pred.shape), eps)
            loss = torch.mean((log_pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * log_pred.size(0)
            seen += log_pred.size(0)
        train_loss = running / max(1, seen)

        val_pred, val_true, val_loss = predict(
            model, val_graphs, devices, batch_size=protocol["batch_size"], device=device, eps=eps
        )
        val_mse_lin = float(np.mean((val_pred - val_true) ** 2))
        history.append({"epoch": epoch, "train_log_mse": train_loss,
                        "val_log_mse": val_loss, "val_mse": val_mse_lin})

        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        logger.info("[%s] epoch %04d  train_log_mse=%.5f  val_log_mse=%.5f  val_mse=%.6f  "
                    "patience=%d/%d", tag, epoch, train_loss, val_loss, val_mse_lin,
                    no_improve, protocol["patience"])

        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, {
                "config": config, "epoch": epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "best_val": best_val, "best_epoch": best_epoch, "best_state": best_state,
                "no_improve": no_improve, "history": history,
            })

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_log_mse": best_val,
        "resumed": start_epoch > 1,
        "history": history,
    }
