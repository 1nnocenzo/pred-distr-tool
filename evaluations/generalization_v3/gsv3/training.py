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


def count_loss(model, batch, kind_weights=None) -> tuple[torch.Tensor, float]:
    """v8: predicted native-operation counts vs the compiled circuits' true ones.

    ``batch.counts`` is ``(n_circuits, n_devices, 3)`` = true (1q ``r``, ``measure``,
    ``cz``) per circuit and device (``-1`` where unknown); ``model.last_counts`` is the
    sum over the gates of the count head's ``n_k``.  MSE on ``log1p`` over the known
    entries; also returns the mean absolute relative error of the 2q count.
    """
    pred, true = model.last_counts, batch.counts.view_as(model.last_counts)
    known = true >= 0
    err = (torch.log1p(pred) - torch.log1p(true.clamp(min=0))) ** 2
    if kind_weights is not None:      # per-kind weights (r, measure, cz)
        err = err * torch.as_tensor(kind_weights, dtype=err.dtype, device=err.device)
    loss = (err * known).sum() / known.sum().clamp(min=1)
    k2 = known[..., 2] & (true[..., 2] > 0)
    rel2 = ((pred[..., 2] - true[..., 2]).abs() / true[..., 2].clamp(min=1))[k2].mean() if k2.any() else pred.new_tensor(0.0)
    return loss, rel2.item()


def layout_loss(model, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """v6: cross-entropy of the Sinkhorn placement vs the compiler's initial layout.

    ``batch.layout`` is ``(n_qubits, n_devices)`` per circuit: the device-graph node
    the compiler put each logical qubit on, ``-1`` where unknown (those qubits are
    left out).  Returns ``(mean -log A[q, layout[q]], fraction of qubits whose argmax
    is the compiler's node)``, both over the known (qubit, device) pairs.
    """
    from torch_geometric.utils import to_dense_batch

    assign, q_mask = model.last_assign, model.last_q_mask      # (B, D, Qm, P), (B, Qm)
    n_q = batch.n_qubits.view(-1).long()
    q_batch = torch.repeat_interleave(torch.arange(n_q.numel(), device=n_q.device), n_q)
    target, _ = to_dense_batch(batch.layout, q_batch, fill_value=-1,
                               max_num_nodes=assign.size(2))    # (B, Qm, D)
    target = target.permute(0, 2, 1)                            # (B, D, Qm)
    known = (target >= 0) & q_mask.unsqueeze(1)
    picked = assign.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    ce = -torch.log(picked.clamp_min(1e-9))
    n = known.sum().clamp(min=1)
    acc = ((assign.argmax(-1) == target) & known).sum() / n
    return (ce * known).sum() / n, acc


def region_distance_loss(model, batch) -> tuple[torch.Tensor, dict[str, float]]:
    """v7: supervise the parts of the compiler's layout that do not depend on the family.

    * **region**: occupancy of each physical qubit, ``u_p = Σ_q A[q, p]`` (in [0, 1]),
      vs whether the compiler put any logical qubit on it (binary cross-entropy over
      the device's qubits);
    * **distance**: the model's expected chip distance between the operands of every
      multi-qubit gate (``model.last_dist``, the input of its count head) vs the hop
      distance between the qubits the compiler put them on (MSE on ``log1p``).

    Circuits without a known layout (``batch.layout`` = -1) are left out.  Returns
    ``(region_bce + distance_mse, {"region_bce", "dist_mse", "region_overlap"})``.
    """
    from torch_geometric.utils import to_dense_batch

    assign, q_mask = model.last_assign, model.last_q_mask         # (B, D, Qm, P), (B, Qm)
    col_mask = model.last_col_mask                                # (D, P)
    n_dev, p_max = col_mask.shape
    n_q = batch.n_qubits.view(-1).long()
    n_circ = n_q.numel()
    q_batch = torch.repeat_interleave(torch.arange(n_circ, device=n_q.device), n_q)
    target, _ = to_dense_batch(batch.layout, q_batch, fill_value=-1,
                               max_num_nodes=assign.size(2))      # (B, Qm, D)
    known_q = (target >= 0) & q_mask.unsqueeze(-1)
    known_c = (known_q | ~q_mask.unsqueeze(-1)).all(1)            # (B, D): every qubit known

    # Region: which physical qubits the compiler used.
    used = torch.zeros(n_circ, n_dev, p_max + 1, device=assign.device)
    idx = torch.where(known_q, target, torch.full_like(target, p_max)).permute(0, 2, 1)  # (B, D, Qm)
    used.scatter_(2, idx, 1.0)
    used = used[..., :p_max]
    occ = assign.sum(2).clamp(1e-6, 1 - 1e-6)                     # (B, D, P)
    cell = (known_c.unsqueeze(-1) & col_mask.unsqueeze(0)).float()
    bce = -(used * occ.log() + (1 - used) * (1 - occ).log())
    region = (bce * cell).sum() / cell.sum().clamp(min=1)
    overlap = (torch.minimum(occ, used) * cell).sum() / (used * cell).sum().clamp(min=1)

    # Distance between interacting qubits under the compiler's layout.
    offsets = torch.cumsum(n_q, 0) - n_q
    gq = batch.gate_qubits
    valid = gq >= 0
    glob = torch.where(valid, gq + offsets[batch.batch].unsqueeze(1), torch.zeros_like(gq))
    hops = model.hops[:, :p_max, :p_max]
    dev = torch.arange(n_dev, device=gq.device).view(1, -1)
    true_d = torch.zeros(gq.size(0), n_dev, device=assign.device)
    n_pairs = torch.zeros(gq.size(0), n_dev, device=assign.device)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        la, lb = batch.layout[glob[:, a]], batch.layout[glob[:, b]]      # (G, D)
        ok = (valid[:, a] & valid[:, b]).unsqueeze(1) & (la >= 0) & (lb >= 0)
        h = hops[dev, la.clamp(min=0), lb.clamp(min=0)]
        true_d = true_d + ok * h
        n_pairs = n_pairs + ok
    gate_ok = n_pairs > 0
    true_d = true_d / n_pairs.clamp(min=1)
    err = (torch.log1p(model.last_dist) - torch.log1p(true_d)) ** 2
    dist = (err * gate_ok).sum() / gate_ok.sum().clamp(min=1)
    return region + dist, {"region_bce": region.item(), "dist_mse": dist.item(),
                           "region_overlap": overlap.item()}


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
    if protocol.get("init_from"):     # fine-tuning (e.g. the change loss on a trained v8a)
        model.load_state_dict(torch.load(protocol["init_from"], map_location=device))
        logger.info("[%s] initialised from %s", tag, protocol["init_from"])
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
    # v6 (opt-in): weight of the layout cross-entropy; needs ``layout`` on the graphs.
    layout_lambda = float(protocol.get("layout_lambda", 0) or 0)
    # v7 (opt-in): "region_dist" = region + distance loss instead of v6's exact-layout CE.
    layout_kind = protocol.get("layout_loss", "assign")
    # v8 (opt-in): weight of the count loss; needs ``counts`` on the graphs.
    count_lambda = float(protocol.get("count_lambda", 0) or 0)
    # optional per-kind weights of the count loss, "r,measure,cz" (e.g. "0.1,0.1,1": cz-heavy)
    count_w = ([float(v) for v in str(protocol["count_weights"]).split(",")]
               if protocol.get("count_weights") else None)
    # v8b (opt-in): calibration augmentation.  Each training batch uses the original
    # calibration with probability ``aug_p_orig``, otherwise one of the variants in
    # ``calib_aug`` (device graphs + per-variant labels / layouts / counts on the graphs).
    aug_devices = None
    if protocol.get("calib_aug"):
        import json as _json
        from .devices import variant_batch
        raw = _json.loads((Path(protocol["calib_aug"]) / "variants_raw.json").read_text())
        names = ("EQE1_Top", "EQE1_Bottom", "QExa20")   # = genstudy.data.DEVICE_NAMES
        n_aug = len(raw) // len(names)
        aug_devices = [variant_batch(raw, [f"{d}/aug{j}" for d in names], params["lap_pe"]).to(device)
                       for j in range(n_aug)]
        # optional subset of the variant sets (e.g. to test how many calibrations are needed)
        aug_ids = [int(j) for j in str(protocol.get("aug_variants", "")).split(",") if j.strip()] or list(range(n_aug))
        aug_rng = np.random.default_rng(seed + 1)
        p_orig = float(protocol.get("aug_p_orig", 0.25))
    # Change loss (opt-in, needs calib_aug and ``y_ob`` on the graphs): every batch is run on
    # the original devices (absolute loss as usual) and on one random variant, and
    # (log F^_var - log F^_orig) is fitted to (log F_var - log F_orig), both best-of-K labels.
    delta_lambda = float(protocol.get("delta_lambda", 0) or 0)
    if delta_lambda and aug_devices is None:
        raise ValueError("delta_lambda needs calib_aug")
        logger.info("[%s] calibration augmentation: variant sets %s of %d, p(orig) = %.2f",
                    tag, aug_ids, n_aug, p_orig)
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
        lay_ce, lay_acc, lay_n = 0.0, 0.0, 0
        v7_stats: dict[str, float] = {}
        cnt_loss, cnt_rel2, cnt_n = 0.0, 0.0, 0
        dl_sum, dl_n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            batch_devices = devices
            if aug_devices is not None and not delta_lambda and aug_rng.random() >= p_orig:
                j = aug_ids[int(aug_rng.integers(len(aug_ids)))]
                batch_devices = aug_devices[j]
                batch.y = batch.y_aug[:, j]
                batch.layout = batch.layout_aug[:, j]
                batch.counts = batch.counts_aug[:, j]
            log_pred = model(batch, batch_devices)
            y = batch.y.float().view(log_pred.shape)
            w = (y >= 0).float()            # v8b: circuits without a label on this variant
            n_w = w.sum().clamp(min=1)
            y = y.clamp(min=0)
            loss = (w * (log_pred - log_target(y, eps)) ** 2).sum() / n_w
            if protocol.get("loss_kind", "log") == "mixed":
                fid = torch.exp(torch.clamp(log_pred, max=0.0))
                loss = (w * (fid - y) ** 2).sum() / n_w + protocol["mix_lambda"] * loss
            fid_loss = loss.item()        # logged as train_log_mse (without the layout term)
            if layout_lambda and layout_kind == "region_dist":
                aux, stats = region_distance_loss(model, batch)
                loss = loss + layout_lambda * aux
                for k, v in stats.items():
                    v7_stats[k] = v7_stats.get(k, 0.0) + v
                lay_n += 1
            elif layout_lambda:
                ce, acc = layout_loss(model, batch)
                loss = loss + layout_lambda * ce
                lay_ce, lay_acc, lay_n = lay_ce + float(ce), lay_acc + float(acc), lay_n + 1
            if count_lambda:
                closs, rel2 = count_loss(model, batch, count_w)
                loss = loss + count_lambda * closs
                cnt_loss, cnt_rel2, cnt_n = cnt_loss + closs.item(), cnt_rel2 + rel2, cnt_n + 1
            if delta_lambda:          # second pass on a random calibration variant
                j = aug_ids[int(aug_rng.integers(len(aug_ids)))]
                log_var = model(batch, aug_devices[j])
                y_v = batch.y_aug[:, j].float().view(log_var.shape)
                y_o = batch.y_ob.float().view(log_var.shape)
                ok = ((y_v > 0.01) & (y_o > 0.01)).float()
                d_true = torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))
                d_pred = torch.clamp(log_var, max=0.0) - torch.clamp(log_pred, max=0.0)
                dloss = (ok * (d_pred - d_true) ** 2).sum() / ok.sum().clamp(min=1)
                loss = loss + delta_lambda * dloss
                dl_sum, dl_n = dl_sum + dloss.item(), dl_n + 1
            optimizer.zero_grad()
            loss.backward()
            if protocol["grad_clip"]:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["grad_clip"])
                max_grad = max(max_grad, float(norm))
            optimizer.step()
            running += fid_loss * log_pred.size(0)
            seen += log_pred.size(0)
        train_loss = running / max(1, seen)

        val_pred, val_true, val_loss = predict(
            model, val_graphs, devices, batch_size=protocol["batch_size"], device=device, eps=eps)
        val_mse = float(np.mean((val_pred - val_true) ** 2))
        history.append({"epoch": epoch, "train_log_mse": train_loss, "val_log_mse": val_loss,
                        "val_mse": val_mse, "max_grad_norm": max_grad})
        if delta_lambda:
            history[-1]["train_delta_mse"] = dl_sum / max(1, dl_n)
            logger.info("[%s] epoch %04d  delta_mse=%.4f", tag, epoch, history[-1]["train_delta_mse"])
        if count_lambda:
            history[-1]["train_count_mse"] = cnt_loss / max(1, cnt_n)
            history[-1]["train_count_rel_err_2q"] = cnt_rel2 / max(1, cnt_n)
            logger.info("[%s] epoch %04d  count_mse=%.4f  count_rel_err_2q=%.3f", tag, epoch,
                        history[-1]["train_count_mse"], history[-1]["train_count_rel_err_2q"])
        if layout_lambda and layout_kind == "region_dist":
            for k, v in v7_stats.items():
                history[-1][f"train_{k}"] = v / max(1, lay_n)
            logger.info("[%s] epoch %04d  region_bce=%.4f  dist_mse=%.4f  region_overlap=%.3f", tag,
                        epoch, history[-1]["train_region_bce"], history[-1]["train_dist_mse"],
                        history[-1]["train_region_overlap"])
        elif layout_lambda:
            history[-1]["train_layout_ce"] = lay_ce / max(1, lay_n)
            history[-1]["train_layout_acc"] = lay_acc / max(1, lay_n)
            logger.info("[%s] epoch %04d  layout_ce=%.4f  layout_acc=%.3f", tag, epoch,
                        history[-1]["train_layout_ce"], history[-1]["train_layout_acc"])

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
    elif best_state is not None and not protocol.get("keep_last"):
        model.load_state_dict(best_state)
        model.to(device)
    return model, {"best_epoch": best_epoch, "epochs_run": len(history),
                   "best_val_score": best_val, "select_metric": protocol["select_metric"], "resumed": start_epoch > 1,
                   "history": history}
