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


def esp_loss(model, batch, weights=None) -> tuple[torch.Tensor, list[float]]:
    """v13: predicted (ESP decay, log duration, log depth) vs the stored compilation's (``batch.esp``,
    NaN = unknown).  The decay (≤ 0) is compared as log1p(−decay); MSE per target, weighted; also
    returns the three MSEs."""
    pred = torch.stack(model.last_esp, -1)                       # (B, D, 3)
    true = batch.esp.view_as(pred)
    known = torch.isfinite(true)
    p = torch.cat([torch.log1p(-pred[..., :1].clamp(max=0)), pred[..., 1:]], -1)
    t = torch.cat([torch.log1p(-true[..., :1].clamp(max=0)), true[..., 1:]], -1)
    err = torch.where(known, (p - torch.nan_to_num(t)) ** 2, torch.zeros_like(p))
    mse = err.sum((0, 1)) / known.sum((0, 1)).clamp(min=1)       # (3,)
    w = mse.new_tensor(weights if weights is not None else (1.0, 1.0, 1.0))
    return (w * mse).sum(), mse.tolist()


def struct_loss(model, batch) -> tuple[torch.Tensor, list[float]]:
    """v15: the analytic ESP head's per-qubit structure (busiest qubit's gate time in µs, measurements, operations;
    ``model.last_esp_feats``) vs the stored compilation's (``batch.struct``, NaN = unknown), MSE of log1p each;
    also returns the three MSEs.  The gradient reaches the count head, which the structure is built from."""
    f = model.last_esp_feats
    pred = torch.stack([f["tmax"], f["mmax"], f["omax"]], -1)              # (B, D, 3)
    true = batch.struct.view(pred.shape[0], pred.shape[1], -1)[..., :3]
    known = torch.isfinite(true)
    err = torch.where(known, (torch.log1p(pred) - torch.log1p(torch.nan_to_num(true))) ** 2, torch.zeros_like(pred))
    mse = err.sum((0, 1)) / known.sum((0, 1)).clamp(min=1)
    return mse.sum(), mse.tolist()


def usage_loss(model, batch, true_maps=None) -> tuple[torch.Tensor, float]:
    """v11: predicted usage maps vs the compiled circuit's (``batch.usage_r/m/cz``, -1 = no maps),
    or vs ``true_maps`` = (r, m, cz) given explicitly (calibration variants).
    MSE on log1p over the real qubits / couplers of each device, the three maps weighted alike;
    also returns the mean correlation between predicted and true CZ maps (diagnostic)."""
    u_r, u_m, u_cz, qmask, emask = model.last_usage               # (B, D, P|E), (D, P|E)
    t_r, t_m, t_cz = true_maps if true_maps is not None else (batch.usage_r, batch.usage_m, batch.usage_cz)
    loss, n = u_r.new_tensor(0.0), 0
    for pred, true, m in ((u_r, t_r, qmask), (u_m, t_m, qmask), (u_cz, t_cz, emask)):
        true = true.view(pred.size(0), pred.size(1), -1)[..., : pred.size(2)]
        known = (true >= 0) & m.unsqueeze(0)
        if known.any():
            err = (torch.log1p(pred) - torch.log1p(true.clamp(min=0))) ** 2
            loss = loss + (err * known).sum() / known.sum()
            n += 1
    tc = t_cz.view(u_cz.size(0), u_cz.size(1), -1)[..., : u_cz.size(2)]
    ok = (tc >= 0).all(-1) & (tc.sum(-1) > 0)
    corr = 0.0
    if ok.any():
        a, b = u_cz[ok].detach(), tc[ok]
        a = a - a.mean(-1, keepdim=True); b = b - b.mean(-1, keepdim=True)
        corr = float(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-9)).mean())
    return loss / max(n, 1), corr


def _occupancy(layout, n_qubits, p_max):
    """compiler layout (sum n, D) node indices -> 0/1 occupancy (B, D, P) and a (B, D) mask of
    the circuits whose every logical qubit has a known place."""
    from torch_geometric.utils import to_dense_batch
    n_q = n_qubits.view(-1).long()
    q_batch = torch.repeat_interleave(torch.arange(n_q.numel(), device=n_q.device), n_q)
    lay, qm = to_dense_batch(layout, q_batch, fill_value=-1)                  # (B, Qm, D)
    known = ((lay >= 0) | ~qm.unsqueeze(-1)).all(1)                            # (B, D)
    onehot = torch.nn.functional.one_hot((lay + 1).clamp(min=0), p_max + 1)[..., 1:].float()
    return onehot.sum(1).clamp(max=1.0), known                                 # (B, D, P)


def reaction_loss(assign_o, assign_v, col_mask, batch, j) -> tuple[torch.Tensor, float]:
    """How the placement moves when the calibration changes, vs how the compiler moves:
    MSE between Δu_model = Σ_q A_variant − Σ_q A_original and Δu_compiler (0/1 occupancy of the
    compiled initial layouts), over the physical qubits of each device; also returns their mean
    correlation over the cases where the compiler moves (diagnostic)."""
    p_max = assign_o.size(-1)
    occ_o, k_o = _occupancy(batch.layout, batch.n_qubits, p_max)
    occ_v, k_v = _occupancy(batch.layout_aug[:, j], batch.n_qubits, p_max)
    du_m = assign_v.sum(2) - assign_o.sum(2)                                   # (B, D, P)
    du_c = occ_v - occ_o
    m = (k_o & k_v).unsqueeze(-1) & col_mask.unsqueeze(0)                     # (B, D, P)
    if not m.any():
        return du_m.new_tensor(0.0), float("nan")
    loss = (((du_m - du_c) ** 2) * m).sum() / m.sum()
    with torch.no_grad():
        mv = m & (du_c.abs().sum(-1, keepdim=True) > 0)
        a = (du_m * mv); b = (du_c * mv)
        n = mv.float().sum(-1).clamp(min=1)
        a = a - a.sum(-1, keepdim=True) / n.unsqueeze(-1); b = b - b.sum(-1, keepdim=True) / n.unsqueeze(-1)
        c = ((a * b * mv).sum(-1) / ((a * a * mv).sum(-1).sqrt() * (b * b * mv).sum(-1).sqrt()).clamp(min=1e-9))
        ok = mv.any(-1)
        corr = float(c[ok].mean()) if ok.any() else float("nan")
    return loss, corr


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


def esp_place_loss(model, batch) -> tuple[torch.Tensor, float]:
    """v20: cross-entropy of the ESP head's own placement (row softmax) on the compiler's exact qubit; returns the
    loss and the share of known (qubit, device) pairs whose argmax is the compiler's qubit."""
    from torch_geometric.utils import to_dense_batch

    logits, q_mask = model.last_esp_logits, model.last_q_mask              # (B, D, Qm, P), (B, Qm)
    n_q = batch.n_qubits.view(-1).long()
    q_batch = torch.repeat_interleave(torch.arange(n_q.numel(), device=n_q.device), n_q)
    target, _ = to_dense_batch(batch.layout, q_batch, fill_value=-1, max_num_nodes=logits.size(2))
    target = target.permute(0, 2, 1)                                        # (B, D, Qm)
    known = (target >= 0) & q_mask.unsqueeze(1)
    picked = logits.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    n = known.sum().clamp(min=1)
    acc = ((logits.argmax(-1) == target) & known).sum() / n
    return -(picked * known).sum() / n, float(acc)


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
        tau_wanted = getattr(model, "tau", None)
        sd = torch.load(protocol["init_from"], map_location=device)
        own = model.state_dict()
        reshaped = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        if reshaped:      # e.g. another ESP head's MLP: start it fresh
            logger.info("[%s] init_from: shapes differ, left at init: %s", tag, reshaped)
            sd = {k: v for k, v in sd.items() if k not in reshaped}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if tau_wanted is not None:          # the fine-tuning's temperature, not the checkpoint's
            model.tau = float(tau_wanted); model.tau_saved.fill_(float(tau_wanted))
        if unexpected:
            raise RuntimeError(f"init_from has unexpected keys: {unexpected}")
        logger.info("[%s] initialised from %s%s", tag, protocol["init_from"],
                    f" (new parameters, not in the checkpoint: {missing})" if missing else "")
    # optional higher learning rate for the placement (fine-tuning that must move it):
    # projections w_q / w_p, dummy score, low-error preference err_w, MLP scorer sc_*
    plr = float(protocol.get("placement_lr_mult", 1) or 1)
    if plr != 1:
        place_names = ("w_q.", "w_p.", "dummy_score", "err_w.", "vf2_log_beta", "sc_a.", "sc_b.", "sc_ab", "sc_out.")
        place = [p_ for n_, p_ in model.named_parameters() if n_.startswith(place_names)]
        rest = [p_ for n_, p_ in model.named_parameters() if not n_.startswith(place_names)]
        optimizer = torch.optim.Adam([{"params": rest, "lr": protocol["lr"]},
                                      {"params": place, "lr": protocol["lr"] * plr}])
        logger.info("[%s] placement learning rate x%g on %d tensors", tag, plr, len(place))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"])
    if getattr(model, "esp_place", False):     # v20: the new ESP placement starts from scratch: lr x10
        ep = [p_ for n_, p_ in model.named_parameters() if n_.startswith(("ep_q.", "ep_p."))]
        ids = {id(p_) for p_ in ep}
        groups = [{**g_, "params": [p_ for p_ in g_["params"] if id(p_) not in ids]} for g_ in optimizer.param_groups]
        optimizer = torch.optim.Adam(groups + [{"params": ep, "lr": protocol["lr"] * 10}])
        logger.info("[%s] ESP placement: %d tensors at lr x10", tag, len(ep))

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
    # v13 (opt-in): ESP decay / duration / depth of the stored compilation; needs ``esp`` on the graphs
    esp_lambda = float(protocol.get("esp_lambda", 0) or 0)
    esp_place_lambda = float(protocol.get("esp_place_lambda", 0) or 0)
    # v21 (opt-in): keep the formula head's duration on the count-based schedule (learned correction -> 0)
    dur_corr_lambda = float(protocol.get("dur_corr_lambda", 0) or 0)
    ep_acc = [0.0, 0.0, 0]
    esp_w = [float(x) for x in str(protocol.get("esp_weights") or "1,1,1").split(",")]
    # v15 (opt-in): per-qubit structure of the stored compilation; needs ``struct`` on the graphs and an *_espa model
    struct_lambda = float(protocol.get("struct_lambda", 0) or 0)
    usage_lambda = float(protocol.get("usage_lambda", 0) or 0)
    # usage maps on the calibration variants (change-loss pass), indexed by ``batch.usage_id``
    usage_aug_lambda = float(protocol.get("usage_aug_lambda", 0) or 0)
    # layout (v7) and count losses also on the calibration variant of the change-loss pass,
    # against the compiler's layout and counts on that variant (v8d labels, every circuit)
    aug_aux = bool(protocol.get("aug_aux"))
    react_lambda = float(protocol.get("react_lambda", 0) or 0)   # placement reaction (needs calib_aug)
    if usage_aug_lambda:
        lab = torch.load(protocol["usage_path_aug"], weights_only=False)
        names = sorted(n for n, e in lab.items() if "r_aug" in e)
        usage_aug = tuple(torch.stack([lab[n][k] for n in names]).to(device) for k in ("r_aug", "m_aug", "cz_aug"))
        del lab
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
    # correlation change loss: 1 − Pearson(Δ predicted, Δ true) over the batch, per device (each batch
    # uses one calibration variant, so this is the local-tracking metric itself); added to the MSE
    corr_lambda = float(protocol.get("delta_corr_lambda", 0) or 0)
    dc_sum, dc_n = 0.0, 0
    # v18 (opt-in): T1 / T2 / gate-duration variants (``tvar_path``: device graphs with per-qubit durations and
    # labels ``tdec`` / ``tdur`` on the graphs).  Every batch is also run on one random variant: the change of the
    # ESP decay (as log1p(−decay), like esp_loss) and of log duration is fitted to the rescheduled compilation's, and
    # log F must not move (the compiler's layout and counts do not change with T1 / T2 / durations).
    tvar_lambda = float(protocol.get("tvar_lambda", 0) or 0)
    tvar_f_lambda = float(protocol.get("tvar_f_lambda", tvar_lambda) or 0)
    tv_acc, tv_n = [0.0, 0.0, 0.0], 0
    if tvar_lambda:
        import json as _json
        from .devices import variant_batch
        raw_t = _json.loads((Path(protocol["tvar_path"]) / "tvar_raw.json").read_text())
        names_t = ("EQE1_Top", "EQE1_Bottom", "QExa20")
        n_tv = sum(1 for k in raw_t if k.startswith(names_t[0] + "/t"))
        tvar_devices = [variant_batch(raw_t, [f"{d}/t{j}" for d in names_t], params["lap_pe"]).to(device)
                        for j in range(n_tv)]
        tv_rng = np.random.default_rng(seed + 7)
        logger.info("[%s] T / duration variants: %d sets, lambda %.3g (F invariance %.3g)", tag, n_tv, tvar_lambda, tvar_f_lambda)
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
        esp_acc, esp_n = [0.0, 0.0, 0.0], 0
        st_acc, st_n = [0.0, 0.0, 0.0], 0
        dl_sum, dl_n = 0.0, 0
        dc_sum, dc_n = 0.0, 0
        us_loss, us_corr, us_n = 0.0, 0.0, 0
        ua_loss, ua_corr, ua_n = 0.0, 0.0, 0
        ax_lay, ax_ov, ax_cnt, ax_rel, ax_n = 0.0, 0.0, 0.0, 0.0, 0
        rx_loss, rx_n, rx_corr, rx_cn = 0.0, 0, 0.0, 0
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
            esp_main = getattr(model, "last_esp", None)     # v18: the original-calibration ESP, before other passes
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
            if esp_place_lambda and batch_devices is devices and hasattr(batch, "layout"):
                ep_l, ep_a = esp_place_loss(model, batch)
                loss = loss + esp_place_lambda * ep_l
                ep_acc = [ep_acc[0] + ep_l.item(), ep_acc[1] + ep_a, ep_acc[2] + 1]
            if count_lambda:
                closs, rel2 = count_loss(model, batch, count_w)
                loss = loss + count_lambda * closs
                cnt_loss, cnt_rel2, cnt_n = cnt_loss + closs.item(), cnt_rel2 + rel2, cnt_n + 1
            if esp_lambda and batch_devices is devices:     # labels of the original calibration
                eloss, emse = esp_loss(model, batch, esp_w)
                loss = loss + esp_lambda * eloss
                if dur_corr_lambda and "dur_corr" in model.last_esp_feats:
                    loss = loss + dur_corr_lambda * model.last_esp_feats["dur_corr"].pow(2).mean()
                esp_acc = [a + b for a, b in zip(esp_acc, emse)]
                esp_n += 1
            if struct_lambda and batch_devices is devices:
                sloss, smse = struct_loss(model, batch)
                loss = loss + struct_lambda * sloss
                st_acc = [a + b for a, b in zip(st_acc, smse)]
                st_n += 1
            if usage_lambda and batch_devices is devices:   # maps exist for the original devices
                uloss, ucorr = usage_loss(model, batch)
                loss = loss + usage_lambda * uloss
                us_loss, us_corr, us_n = us_loss + uloss.item(), us_corr + ucorr, us_n + 1
            if delta_lambda:          # second pass on a random calibration variant
                j = aug_ids[int(aug_rng.integers(len(aug_ids)))]
                assign_o = model.last_assign
                log_var = model(batch, aug_devices[j])
                if react_lambda:
                    rloss, rcorr = reaction_loss(assign_o, model.last_assign, model.last_col_mask, batch, j)
                    loss = loss + react_lambda * rloss
                    rx_loss, rx_n = rx_loss + rloss.item(), rx_n + 1
                    if rcorr == rcorr:
                        rx_corr, rx_cn = rx_corr + rcorr, rx_cn + 1
                y_v = batch.y_aug[:, j].float().view(log_var.shape)
                y_o = batch.y_ob.float().view(log_var.shape)
                ok = ((y_v > 0.01) & (y_o > 0.01)).float()
                d_true = torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))
                d_pred = torch.clamp(log_var, max=0.0) - torch.clamp(log_pred, max=0.0)
                dloss = (ok * (d_pred - d_true) ** 2).sum() / ok.sum().clamp(min=1)
                loss = loss + delta_lambda * dloss
                if corr_lambda:
                    cs = []
                    for k in range(d_pred.size(-1)):
                        m = ok[:, k] > 0
                        if int(m.sum()) > 4:
                            a, b = d_pred[m, k], d_true[m, k]
                            a, b = a - a.mean(), b - b.mean()
                            cs.append((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))
                    if cs:
                        pcl = 1 - torch.stack(cs).mean()
                        loss = loss + corr_lambda * pcl
                        dc_sum, dc_n = dc_sum + pcl.item(), dc_n + 1
                if aug_aux:
                    lay0, cnt0 = batch.layout, getattr(batch, "counts", None)
                    batch.layout = batch.layout_aug[:, j]
                    if cnt0 is not None:
                        batch.counts = batch.counts_aug[:, j]
                    if layout_lambda and layout_kind == "region_dist":
                        aux_v, st_v = region_distance_loss(model, batch)
                        loss = loss + layout_lambda * aux_v
                        ax_lay, ax_n = ax_lay + float(aux_v), ax_n + 1
                        ax_ov += st_v.get("region_overlap", 0.0)
                    if count_lambda and cnt0 is not None:
                        closs_v, rel2_v = count_loss(model, batch, count_w)
                        loss = loss + count_lambda * closs_v
                        ax_cnt, ax_rel = ax_cnt + closs_v.item(), ax_rel + rel2_v
                    batch.layout = lay0
                    if cnt0 is not None:
                        batch.counts = cnt0
                if usage_aug_lambda:              # where the compiler puts the work on this variant
                    uid = batch.usage_aug_id.view(-1)
                    have = (uid >= 0).view(-1, 1, 1)
                    tm = tuple(torch.where(have, t[uid.clamp(min=0), j], torch.full_like(t[uid.clamp(min=0), j], -1.0))
                               for t in usage_aug)
                    if bool(have.any()):
                        ualoss, uacorr = usage_loss(model, batch, tm)
                        loss = loss + usage_aug_lambda * ualoss
                        ua_loss, ua_corr, ua_n = ua_loss + ualoss.item(), ua_corr + uacorr, ua_n + 1
                dl_sum, dl_n = dl_sum + dloss.item(), dl_n + 1
            if tvar_lambda and batch_devices is devices and esp_main is not None:
                dec_o, ldur_o = esp_main[0], esp_main[1]
                jt = int(tv_rng.integers(len(tvar_devices)))
                log_tv = model(batch, tvar_devices[jt])
                dec_v, ldur_v = model.last_esp[0], model.last_esp[1]
                td = batch.tdec.view(-1, batch.tdec.size(-2), batch.tdec.size(-1))      # (B, 1 + K, D)
                tu = batch.tdur.view_as(td)
                t_dec = torch.log1p(-td[:, 1 + jt].clamp(max=0)) - torch.log1p(-td[:, 0].clamp(max=0))
                t_dur = torch.log(tu[:, 1 + jt].clamp(min=1e-6)) - torch.log(tu[:, 0].clamp(min=1e-6))
                ok = torch.isfinite(t_dec) & torch.isfinite(t_dur)
                p_dec = torch.log1p(-dec_v.clamp(max=0)) - torch.log1p(-dec_o.clamp(max=0))
                p_dur = ldur_v - ldur_o
                n_ok = ok.sum().clamp(min=1)
                l_dec = torch.where(ok, (p_dec - torch.nan_to_num(t_dec)) ** 2, torch.zeros_like(p_dec)).sum() / n_ok
                l_dur = torch.where(ok, (p_dur - torch.nan_to_num(t_dur)) ** 2, torch.zeros_like(p_dur)).sum() / n_ok
                l_f = ((torch.clamp(log_tv, max=0.0) - torch.clamp(log_pred, max=0.0)) ** 2).mean()
                loss = loss + tvar_lambda * (l_dec + l_dur) + tvar_f_lambda * l_f
                tv_acc = [a + b for a, b in zip(tv_acc, (l_dec.item(), l_dur.item(), l_f.item()))]
                tv_n += 1
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
            if corr_lambda:
                history[-1]["train_delta_corr"] = 1 - dc_sum / max(1, dc_n)
                logger.info("[%s] epoch %04d  delta_corr=%.3f", tag, epoch, history[-1]["train_delta_corr"])
        if usage_lambda:
            history[-1]["train_usage_mse"] = us_loss / max(1, us_n)
            history[-1]["train_usage_cz_corr"] = us_corr / max(1, us_n)
            logger.info("[%s] epoch %04d  usage_mse=%.4f  usage_cz_corr=%.3f", tag, epoch,
                        history[-1]["train_usage_mse"], history[-1]["train_usage_cz_corr"])
        if react_lambda:
            history[-1]["train_react_mse"] = rx_loss / max(1, rx_n)
            history[-1]["train_react_corr"] = rx_corr / max(1, rx_cn)
            logger.info("[%s] epoch %04d  react_mse=%.4f  react_corr=%.3f", tag, epoch,
                        history[-1]["train_react_mse"], history[-1]["train_react_corr"])
        if aug_aux:
            history[-1]["train_aug_layout"] = ax_lay / max(1, ax_n)
            history[-1]["train_aug_region_overlap"] = ax_ov / max(1, ax_n)
            history[-1]["train_aug_count_mse"] = ax_cnt / max(1, ax_n)
            history[-1]["train_aug_count_rel_err_2q"] = ax_rel / max(1, ax_n)
            logger.info("[%s] epoch %04d  aug_layout=%.4f  aug_region_overlap=%.3f  aug_count_mse=%.4f  aug_count_rel_err_2q=%.3f",
                        tag, epoch, *(history[-1][k] for k in ("train_aug_layout", "train_aug_region_overlap",
                                                               "train_aug_count_mse", "train_aug_count_rel_err_2q")))
        if usage_aug_lambda:
            history[-1]["train_usage_aug_mse"] = ua_loss / max(1, ua_n)
            history[-1]["train_usage_aug_cz_corr"] = ua_corr / max(1, ua_n)
            logger.info("[%s] epoch %04d  usage_aug_mse=%.4f  usage_aug_cz_corr=%.3f", tag, epoch,
                        history[-1]["train_usage_aug_mse"], history[-1]["train_usage_aug_cz_corr"])
        if tvar_lambda:
            for k, v in zip(("train_tvar_decay_mse", "train_tvar_duration_mse", "train_tvar_f_mse"), tv_acc):
                history[-1][k] = v / max(1, tv_n)
            logger.info("[%s] epoch %04d  tvar decay_mse=%.4f  duration_mse=%.4f  f_mse=%.5f", tag, epoch,
                        *(history[-1][k] for k in ("train_tvar_decay_mse", "train_tvar_duration_mse", "train_tvar_f_mse")))
        if esp_place_lambda:
            history[-1]["train_esp_place_ce"] = ep_acc[0] / max(1, ep_acc[2])
            history[-1]["train_esp_place_acc"] = ep_acc[1] / max(1, ep_acc[2])
            logger.info("[%s] epoch %04d  ESP placement ce=%.3f  exact qubit=%.3f", tag, epoch,
                        history[-1]["train_esp_place_ce"], history[-1]["train_esp_place_acc"])
            ep_acc = [0.0, 0.0, 0]
        if esp_lambda:
            for k, v in zip(("train_esp_decay_mse", "train_log_duration_mse", "train_log_depth_mse"), esp_acc):
                history[-1][k] = v / max(1, esp_n)
            logger.info("[%s] epoch %04d  esp_decay_mse=%.4f  log_duration_mse=%.4f  log_depth_mse=%.4f", tag, epoch,
                        *(history[-1][k] for k in ("train_esp_decay_mse", "train_log_duration_mse", "train_log_depth_mse")))
        if struct_lambda:
            for k, v in zip(("train_struct_tmax_mse", "train_struct_mmax_mse", "train_struct_omax_mse"), st_acc):
                history[-1][k] = v / max(1, st_n)
            logger.info("[%s] epoch %04d  struct_mse tmax=%.4f  mmax=%.4f  omax=%.4f", tag, epoch,
                        *(history[-1][k] for k in ("train_struct_tmax_mse", "train_struct_mmax_mse", "train_struct_omax_mse")))
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
