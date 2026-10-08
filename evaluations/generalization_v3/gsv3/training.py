"""Training protocol of v3: the v2 protocol made numerically stable.

In v2 the training loss exploded in 4 of the 5 finished leave-one-family-out
splits (train log-MSE from ~0.05 to 10-9860 between epochs 24 and 46), and
early stopping kept a model from before the blow-up.  v3 changes only:

* **gradient clipping** (global norm, default 1.0);
* **learning rate** as a protocol entry (default 3e-4 instead of the tuned 1e-3);
* **model kind** (``pooled`` = v2 model, ``xattn`` = qubit cross-attention model);
* **model selection on the linear validation MSE** (``select_metric="mse"``).
  The log-MSE of the out-of-distribution validation families is dominated by
  circuits with ``F`` near ``1e-3`` (log errors of several units on values that
  are irrelevant to R²): in the first xattn run its minimum was at epoch 2 while
  the linear MSE kept improving.  The linear MSE is what R² measures.

Loss (MSE on ``log max(F, 1e-3)``), validation split, family-balanced sampler,
batch size, epoch budget and patience are v2's, imported unchanged.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from . import paths
from .model import build_model

paths.ensure_imports()

from gsv2.training import TRAIN_PROTOCOL as V2_PROTOCOL  # noqa: E402
from gsv2.training import balanced_sampler, log_target, validation_split  # noqa: E402,F401

logger = logging.getLogger(__name__)

TRAIN_PROTOCOL: dict[str, Any] = {
    **V2_PROTOCOL,
    "lr": 3e-4,
    "grad_clip": 1.0,
    "model": "xattn",
    "select_metric": "mse",   # "mse" (linear, = what R² measures) or "log_mse" (v2)
    # v4: "mixed" = MSE on F + mix_lambda * MSE on log F.  The linear term is what
    # R² measures (v1 trained on it and was best on the QFT family); the log term
    # keeps the additive structure informative for low-fidelity circuits.
    "loss_kind": "log",
    "mix_lambda": 0.1,
}


@torch.no_grad()
def predict(model, graphs, devices, *, batch_size: int, device: torch.device,
            eps: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``(fidelity predictions, targets, log-space MSE)``; predictions are ``exp(min(log F, 0))``."""
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


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


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
    seed, eps = protocol["seed"], protocol["log_eps"]
    torch.manual_seed(seed)
    model = build_model(protocol["model"], params, devices, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"])

    sampler = balanced_sampler(train_families, protocol["balance_alpha"], seed)
    loader = DataLoader(list(train_graphs), batch_size=protocol["batch_size"],
                        shuffle=sampler is None, sampler=sampler)

    config = {"params": params,
              "protocol": {k: v for k, v in protocol.items() if k != "num_epochs"},
              "n_train": len(train_graphs), "n_val": len(val_graphs)}
    best_val, best_epoch, best_state, no_improve = float("inf"), 0, None, 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    # v5b "budget" mode (opt-in, absent from the defaults so older configs are
    # unchanged): fixed number of epochs, cosine learning-rate decay, SWA (uniform
    # average of the end-of-epoch weights over the last ``swa_frac`` of the budget,
    # returned as the final model, no early stopping) and, for Sinkhorn models, a
    # temperature annealed from ``tau_start`` to the model's ``sinkhorn_tau``.
    budget = int(protocol.get("epoch_budget", 0) or 0)
    swa_state, swa_n = None, 0
    tau_end = getattr(model, "tau", None)

    if checkpoint_path is not None and resume and checkpoint_path.is_file():
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if state.get("config") == config:
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                start_epoch = state["epoch"] + 1
                best_val, best_epoch = state["best_val"], state["best_epoch"]
                best_state, no_improve = state["best_state"], state["no_improve"]
                history = state["history"]
                swa_state, swa_n = state.get("swa_state"), state.get("swa_n", 0)
                logger.info("[%s] resumed at epoch %d (best %d, val=%.5f)",
                            tag, state["epoch"], best_epoch, best_val)
            else:
                logger.warning("Checkpoint %s has a different configuration; starting over",
                               checkpoint_path)
        except Exception:
            logger.exception("Ignoring unusable checkpoint %s", checkpoint_path)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[%s] model=%s (%d params) train=%d val=%d lr=%g clip=%s alpha=%.2f",
                tag, protocol["model"], n_params, len(train_graphs), len(val_graphs),
                protocol["lr"], protocol["grad_clip"], protocol["balance_alpha"])

    last_epoch = budget if budget else protocol["num_epochs"]
    for epoch in range(start_epoch, last_epoch + 1):
        if not budget and no_improve >= protocol["patience"]:
            break
        if budget:
            frac = (epoch - 1) / max(1, budget - 1)
            for group in optimizer.param_groups:
                group["lr"] = protocol["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * frac)))
            if tau_end is not None and protocol.get("tau_start"):
                anneal = min(1.0, frac / 0.6)
                model.tau = float(protocol["tau_start"] * (tau_end / protocol["tau_start"]) ** anneal)
        model.train()
        running, seen, max_grad = 0.0, 0, 0.0
        for batch in loader:
            batch = batch.to(device)
            log_pred = model(batch, devices)
            y = batch.y.float().view(log_pred.shape)
            loss = torch.mean((log_pred - log_target(y, eps)) ** 2)
            if protocol.get("loss_kind", "log") == "mixed":
                fid = torch.exp(torch.clamp(log_pred, max=0.0))
                loss = torch.mean((fid - y) ** 2) + protocol["mix_lambda"] * loss
            optimizer.zero_grad()
            loss.backward()
            if protocol["grad_clip"]:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["grad_clip"])
                max_grad = max(max_grad, float(norm))
            optimizer.step()
            running += loss.item() * log_pred.size(0)
            seen += log_pred.size(0)
        train_loss = running / max(1, seen)

        val_pred, val_true, val_loss = predict(
            model, val_graphs, devices, batch_size=protocol["batch_size"], device=device, eps=eps)
        val_mse = float(np.mean((val_pred - val_true) ** 2))
        history.append({"epoch": epoch, "train_log_mse": train_loss, "val_log_mse": val_loss,
                        "val_mse": val_mse, "max_grad_norm": max_grad})

        if budget:
            history[-1]["lr"] = optimizer.param_groups[0]["lr"]
            if tau_end is not None:
                history[-1]["tau"] = float(model.tau)
            if epoch > budget * (1 - protocol.get("swa_frac", 0.25)):
                w = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state, swa_n = w, 1
                else:
                    swa_n += 1
                    for k in swa_state:
                        swa_state[k] += (w[k] - swa_state[k]) / swa_n
        score = val_mse if protocol["select_metric"] == "mse" else val_loss
        if score < best_val:
            best_val, best_epoch, no_improve = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        logger.info("[%s] epoch %04d  train_log_mse=%.5f  val_log_mse=%.5f  val_mse=%.6f  "
                    "max_grad=%.2f  patience=%d/%d", tag, epoch, train_loss, val_loss, val_mse,
                    max_grad, no_improve, protocol["patience"])

        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, {
                "config": config, "epoch": epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "best_val": best_val, "best_epoch": best_epoch, "best_state": best_state,
                "no_improve": no_improve, "history": history,
                "swa_state": swa_state, "swa_n": swa_n,
            })

    if budget and swa_state is not None:
        ref = model.state_dict()
        model.load_state_dict({k: v.to(ref[k].dtype) for k, v in swa_state.items()})
        model.to(device)
        best_epoch = budget
        _, _, swa_val = predict(model, val_graphs, devices, batch_size=protocol["batch_size"],
                                device=device, eps=eps)
        logger.info("[%s] final model = SWA of the last %d epochs (val_log_mse=%.5f)",
                    tag, swa_n, swa_val)
    elif best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model, {"best_epoch": best_epoch, "epochs_run": len(history),
                   "best_val_score": best_val, "select_metric": protocol["select_metric"], "resumed": start_epoch > 1,
                   "history": history}
