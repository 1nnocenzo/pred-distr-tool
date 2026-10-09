"""Model construction, training and inference under a frozen configuration.

The architecture is :class:`gnn.GNN` from ``src/model/gnn.py`` and the
hyper-parameters are read from ``src/model/best_params.json`` — the tuned
configuration of Table ``tab:ml4qc:scheduling-hparams``.  **No hyper-parameter is
re-tuned on a held-out family or on the larger circuits**: every split in this
study trains the very same configuration with the very same optimisation
protocol, so any change in the reported metrics is attributable to the split.

The frozen optimisation protocol (:data:`TRAIN_PROTOCOL`) matches the one used to
produce the published model: Adam, MSE loss, batch size 32, up to 1000 epochs,
early stopping on the validation MSE with patience 30, best weights restored.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch_geometric.loader import DataLoader

from . import paths

paths.ensure_model_imports()

from gnn import GNN  # noqa: E402  (src/model/gnn.py)
from encoding import get_gnn_input_features  # noqa: E402  (src/model/encoding.py)

logger = logging.getLogger(__name__)

#: Number of device outputs of the predictor.
NUM_OUTPUTS = 3

#: Frozen optimisation protocol, identical for every split of the study.
TRAIN_PROTOCOL: dict[str, Any] = {
    "optimizer": "adam",
    "loss": "mse",
    "batch_size": 32,
    "num_epochs": 1000,
    "patience": 30,
    "val_fraction": 0.2,
    "seed": 5,
}


def load_hparams(params_path: Path | str | None = None) -> dict[str, Any]:
    """Load the tuned hyper-parameters (Table ``tab:ml4qc:scheduling-hparams``)."""
    params_path = Path(params_path or paths.MODEL_DIR / "best_params.json")
    with open(params_path) as handle:
        params = json.load(handle)
    logger.info("Hyper-parameters from %s: %s", params_path, params)
    return params


def resolve_device(requested: str | None = None) -> torch.device:
    """Return the compute device, defaulting to CUDA when available."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(params: dict[str, Any], device: torch.device) -> GNN:
    """Instantiate the predictor from the tuned hyper-parameters.

    Architecture knobs that the tuning study fixed rather than searched
    (``sag_pool``, ``combine``, ``dropedge_p``) are stated explicitly, with the
    JSON file allowed to override them, so the configuration can never drift
    silently with ``gnn.py`` defaults.
    """
    return GNN(
        in_feats=get_gnn_input_features(),
        hidden_dim=params["hidden_dim"],
        num_conv_wo_resnet=params["num_conv_wo_resnet"],
        num_resnet_layers=params["num_resnet_layers"],
        mlp_units=params["mlp"],
        output_dim=NUM_OUTPUTS,
        dropout_p=params["dropout"],
        dropedge_p=params.get("dropedge_p", 0),
        bidirectional=params["bidirectional"],
        combine=params.get("combine", "sum"),
        use_sag_pool=params.get("sag_pool", False),
        sag_ratio=params.get("sag_ratio", 0.9),
        num_features_manual=0,
        readout=params.get("readout", "meanmaxsum"),
        conv_activation=torch.nn.functional.leaky_relu,
        mlp_activation=torch.nn.functional.leaky_relu,
    ).to(device)


@torch.no_grad()
def predict(
    model: nn.Module,
    graphs: Sequence[Any],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return ``(predictions, targets)`` as ``(n, 3)`` arrays.

    Predictions are clamped to ``[0, 1]``, exactly as in the deployed inference
    path (``evaluations/pipeline/gnn_device_policy.py``), so the metrics describe
    the predictor as the scheduler actually consumes it.
    """
    model.eval()
    loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False)
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = torch.clamp(model(batch), 0.0, 1.0)
        target = batch.y.float()
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if out.shape != target.shape:
            raise ValueError(f"shape mismatch: preds {out.shape} vs targets {target.shape}")
        preds.append(out.cpu())
        targets.append(target.cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def train_model(
    train_graphs: Sequence[Any],
    val_graphs: Sequence[Any],
    params: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int = TRAIN_PROTOCOL["batch_size"],
    num_epochs: int = TRAIN_PROTOCOL["num_epochs"],
    patience: int = TRAIN_PROTOCOL["patience"],
    seed: int = TRAIN_PROTOCOL["seed"],
    tag: str = "split",
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
) -> tuple[GNN, dict[str, Any]]:
    """Train one model under the frozen protocol and restore the best weights.

    Args:
        checkpoint_path: When given, the full training state (weights, optimiser,
            best-so-far weights, patience counter, history) is written there after
            every epoch and reloaded on start, so a job killed by a cluster time
            limit resumes mid-split instead of restarting the split from epoch 1.
        resume: Set ``False`` to ignore an existing checkpoint and train from
            scratch (used by ``--overwrite``).

    Returns:
        ``(model, history)`` where ``history`` records the per-epoch losses, the
        selected epoch and the best validation MSE.
    """
    torch.manual_seed(seed)
    model = build_model(params, device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loader = DataLoader(list(train_graphs), batch_size=batch_size, shuffle=True)

    best_val, best_epoch, best_state, no_improve = float("inf"), 0, None, 0
    history: list[dict[str, float]] = []
    start_epoch = 1

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path is not None and resume and checkpoint_path.is_file():
        state = _load_checkpoint(checkpoint_path, model, optimizer, device)
        if state is not None:
            start_epoch = state["epoch"] + 1
            best_val = state["best_val"]
            best_epoch = state["best_epoch"]
            best_state = state["best_state"]
            no_improve = state["no_improve"]
            history = state["history"]
            logger.info(
                "[%s] resumed from %s at epoch %d (best epoch %d, best val_mse=%.6f, "
                "patience %d/%d)",
                tag, checkpoint_path, state["epoch"], best_epoch, best_val,
                no_improve, patience,
            )
            if no_improve >= patience or start_epoch > num_epochs:
                logger.info("[%s] checkpoint already satisfies the stopping rule", tag)
                if best_state is not None:
                    model.load_state_dict(best_state)
                    model.to(device)
                return model, {
                    "best_epoch": best_epoch,
                    "epochs_run": len(history),
                    "best_val_mse": best_val,
                    "resumed": True,
                    "history": history,
                }

    logger.info(
        "[%s] training on %d graphs, validating on %d, epochs %d-%d (patience %d)",
        tag, len(train_graphs), len(val_graphs), start_epoch, num_epochs, patience,
    )

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            targets = batch.y.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * targets.size(0)
            seen += targets.size(0)
        train_mse = running / max(1, seen)

        val_preds, val_targets = predict(
            model, val_graphs, batch_size=batch_size, device=device
        )
        val_mse = float(np.mean((val_preds - val_targets) ** 2))
        val_r2 = float(r2_score(val_targets, val_preds, multioutput="uniform_average"))
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "val_r2": val_r2})

        if val_mse < best_val:
            best_val, best_epoch, no_improve = val_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        logger.info(
            "[%s] epoch %04d/%d  train_mse=%.6f  val_mse=%.6f  val_r2=%.4f  patience=%d/%d",
            tag, epoch, num_epochs, train_mse, val_mse, val_r2, no_improve, patience,
        )

        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val=best_val,
                best_epoch=best_epoch,
                best_state=best_state,
                no_improve=no_improve,
                history=history,
            )

        if no_improve >= patience:
            logger.info("[%s] early stop at epoch %d; best epoch %d", tag, epoch, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_mse": best_val,
        "resumed": start_epoch > 1,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Mid-split checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, **state: Any) -> None:
    """Atomically persist the training state after one epoch.

    Written to a temporary file and renamed, so a job killed during the write
    leaves the previous checkpoint intact rather than a truncated one.
    """
    model = state.pop("model")
    optimizer = state.pop("optimizer")
    payload = {
        **state,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> dict[str, Any] | None:
    """Restore a checkpoint into ``model``/``optimizer``; ``None`` if unusable.

    A corrupt or stale checkpoint (for instance one written by a different
    architecture) must not abort a multi-fold job: it is reported and the split
    simply trains from scratch.
    """
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
    except Exception:
        logger.exception("Ignoring unusable checkpoint %s; training from scratch", path)
        return None
    return {
        "epoch": int(payload["epoch"]),
        "best_val": float(payload["best_val"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_state": payload.get("best_state"),
        "no_improve": int(payload["no_improve"]),
        "history": list(payload.get("history", [])),
    }
