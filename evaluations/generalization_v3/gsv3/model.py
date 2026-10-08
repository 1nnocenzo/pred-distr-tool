"""Predictors of the v3 study.

``pooled``  the v2 ``DeviceAwarePredictor`` (imported unchanged from ``gsv2``), used
            to measure the effect of the stabilised training alone.

``xattn``   :class:`QubitCrossAttentionPredictor`, end-to-end from the raw circuit
            and the raw device description::

    gate DAG ──GraphConvolutionSage (node level)──► gate embeddings g
                     │ scatter over each gate's operand qubits
                     ▼
    logical qubits ──GINE over the interaction graph (edges = multi-qubit gates)──► q
                     │ cross-attention (queries q, keys/values = physical qubits)
                     ▼
    device d ──GINE over the coupling map (+ Laplacian PE)──► physical qubits p_d
                     │
    per gate g, per device d:  cost(g, d) = softplus(MLP[g, ctx_mean, ctx_range, d̄, g·ctx_mean]) ≥ 0
                     ▼
    log F(c, d) = − Σ_g cost(g, d)

The cross-attention is a soft, learned layout: each logical qubit attends to the
physical qubits of the device; the contexts of a gate's operands then tell the
cost head where on the chip the gate lands (and so how far apart its qubits are,
i.e. the routing it will need).  No compilation, routing estimate or other
hand-made feature is used.  ``log F`` is a sum of non-negative per-gate terms,
which keeps ``F <= 1`` by construction and mirrors the product form of the
expected fidelity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter, to_dense_batch

from . import paths

paths.ensure_imports()

from encoding import get_gnn_input_features  # noqa: E402  (src/model/encoding.py)
from gnn import GraphConvolutionSage  # noqa: E402  (src/model/gnn.py)
from gsv2.model import build_model as build_v2_model  # noqa: E402

logger = logging.getLogger(__name__)

#: Fixed (untuned) sizes of the v3 components; the circuit encoder keeps best_params.json.
XATTN_DEFAULTS: dict[str, Any] = {
    "device_hidden": 64,      # v2 pooled model only
    "device_layers": 3,
    "fusion_dim": 128,
    "qubit_layers": 2,
    "attn_heads": 4,
    "lap_pe": 8,
    "cost_bias_init": -6.0,   # softplus(-6) ≈ 2.5e-3 per gate at initialisation
}

MODEL_KINDS = ("pooled", "xattn", "phys", "sinkhorn")


def load_hparams(params_path: Path | str | None = None) -> dict[str, Any]:
    params_path = Path(params_path or paths.REPO_ROOT / "src" / "model" / "best_params.json")
    with open(params_path) as handle:
        params = json.load(handle)
    merged = {**XATTN_DEFAULTS, **params}
    logger.info("Hyper-parameters (%s + v3 defaults): %s", params_path, merged)
    return merged


class NodeSage(GraphConvolutionSage):
    """``GraphConvolutionSage`` that returns node embeddings instead of a pooled vector."""

    def forward(self, data) -> torch.Tensor:  # noqa: D401
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            if self.bidirectional:
                x_new = self._apply_conv_bidir(conv, x, edge_index)
            else:
                x_new = conv(x, edge_index)
            x_new = self.norms[i](x_new, batch=batch)
            x_new = self.conv_activation(x_new, **self.conv_act_kwargs)
            x_new = self.drop(x_new)
            x = x_new if i < self._residual_start else x + x_new
        return x


def _gine(dim: int, edge_dim: int) -> GINEConv:
    return GINEConv(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(), nn.Linear(dim, dim)),
                    edge_dim=edge_dim)


class QubitCrossAttentionPredictor(nn.Module):
    """Returns ``log F`` of shape ``(B, n_devices)``; always ``<= 0``."""

    def __init__(self, params: dict[str, Any], dev_node_dim: int, dev_edge_dim: int) -> None:
        super().__init__()
        d = params["fusion_dim"]
        self.circuit_encoder = NodeSage(
            in_feats=get_gnn_input_features(),
            hidden_dim=params["hidden_dim"],
            num_conv_wo_resnet=params["num_conv_wo_resnet"],
            num_resnet_layers=params["num_resnet_layers"],
            dropout_p=params["dropout"],
            dropedge_p=0.0,
            bidirectional=params["bidirectional"],
            combine=params.get("combine", "sum"),
            readout=params.get("readout", "meanmaxsum"),
            conv_activation=F.leaky_relu,
        )
        self.gate_proj = nn.Linear(self.circuit_encoder.out_dim, d)

        self.qubit_in = nn.Linear(2 * d, d)
        self.qubit_convs = nn.ModuleList(_gine(d, d) for _ in range(params["qubit_layers"]))

        self.dev_in = nn.Linear(dev_node_dim, d)
        self.dev_convs = nn.ModuleList(
            _gine(d, dev_edge_dim) for _ in range(params["device_layers"]))

        self.attn = nn.MultiheadAttention(d, params["attn_heads"], batch_first=True)

        layers: list[nn.Module] = []
        last = 5 * d
        for width in params["mlp"]:
            layers += [nn.Linear(last, width), nn.LeakyReLU()]
            last = width
        out = nn.Linear(last, 1)
        nn.init.constant_(out.bias, params["cost_bias_init"])
        layers.append(out)
        self.cost = nn.Sequential(*layers)

    # -- circuit side ------------------------------------------------------

    def _qubits(self, circuits, g: torch.Tensor):
        """Logical-qubit embeddings ``(Q, d)`` and global operand indices ``(G, 3)``."""
        n_q = circuits.n_qubits.view(-1).long()
        offsets = torch.cumsum(n_q, 0) - n_q
        gq = circuits.gate_qubits
        valid = gq >= 0
        gq_glob = torch.where(valid, gq + offsets[circuits.batch].unsqueeze(1), -1)
        n_total = int(n_q.sum())

        gate_ids = torch.arange(g.size(0), device=g.device).unsqueeze(1).expand_as(gq)
        src_g, dst_q = gate_ids[valid], gq_glob[valid]
        q_mean = scatter(g[src_g], dst_q, dim=0, dim_size=n_total, reduce="mean")
        q_max = scatter(g[src_g], dst_q, dim=0, dim_size=n_total, reduce="max")
        q = F.leaky_relu(self.qubit_in(torch.cat([q_mean, q_max], dim=-1)))

        # Interaction graph: one edge per operand pair of every multi-qubit gate.
        src, dst, eg = [], [], []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            m = valid[:, a] & valid[:, b]
            if m.any():
                src += [gq_glob[m, a], gq_glob[m, b]]
                dst += [gq_glob[m, b], gq_glob[m, a]]
                ids = gate_ids[m, 0]
                eg += [ids, ids]
        if src:
            edge_index = torch.stack([torch.cat(src), torch.cat(dst)])
            edge_attr = g[torch.cat(eg)]
            for conv in self.qubit_convs:
                q = q + F.leaky_relu(conv(q, edge_index, edge_attr))
        return q, gq_glob, valid

    # -- device side -------------------------------------------------------

    def _devices(self, devices):
        x = F.leaky_relu(self.dev_in(devices.x))
        for conv in self.dev_convs:
            x = x + F.leaky_relu(conv(x, devices.edge_index, devices.edge_attr))
        dense, mask = to_dense_batch(x, devices.batch)            # (D, P, d), (D, P)
        glob = scatter(x, devices.batch, dim=0, reduce="mean")     # (D, d)
        return dense, mask, glob

    def forward(self, circuits, devices) -> torch.Tensor:
        g = self.gate_proj(self.circuit_encoder(circuits))         # (G, d)
        q, gq_glob, valid = self._qubits(circuits, g)              # (Q, d)
        phys, mask, dev_glob = self._devices(devices)
        n_dev = phys.size(0)

        query = q.unsqueeze(0).expand(n_dev, -1, -1)               # (D, Q, d)
        ctx, _ = self.attn(query, phys, phys, key_padding_mask=~mask, need_weights=False)
        ctx = ctx.transpose(0, 1)                                  # (Q, D, d)

        slots = ctx[gq_glob.clamp(min=0)]                          # (G, 3, D, d)
        vmask = valid.unsqueeze(-1).unsqueeze(-1)
        count = valid.sum(1).clamp(min=1).view(-1, 1, 1).float()
        c_mean = (slots * vmask).sum(1) / count                    # (G, D, d)
        c_max = slots.masked_fill(~vmask, -1e4).amax(1)
        c_min = slots.masked_fill(~vmask, 1e4).amin(1)
        c_range = torch.where(valid.sum(1).view(-1, 1, 1) > 1, c_max - c_min,
                              torch.zeros_like(c_max))

        g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
        d_e = dev_glob.unsqueeze(0).expand(g.size(0), -1, -1)
        z = torch.cat([g_e, c_mean, c_range, d_e, g_e * c_mean], dim=-1)
        cost = F.softplus(self.cost(z).squeeze(-1))                # (G, D)
        n_circ = circuits.n_qubits.view(-1).size(0)
        return -scatter(cost, circuits.batch, dim=0, dim_size=n_circ, reduce="sum")


class PhysicsHeadPredictor(QubitCrossAttentionPredictor):
    """v4 ``phys``: learned compilation counts × the device's calibrated errors.

    The expected fidelity is ``Π_op (1 - e_op)`` over the operations of the
    *compiled* circuit, i.e. ``log F = -Σ_op ε_op`` with ``ε = -log(1-e)``.  Instead
    of a free per-gate cost, the head predicts for every gate and device how many
    native operations it becomes — ``n_1q``, ``n_2q`` (routing SWAPs included) and
    ``n_meas`` — and weights them with the calibrated errors of the physical
    qubits the cross-attention places the gate's operands on::

        cost(g, d) = Σ_k s_k · n_k(g, d) · ε̄_k(g, d)      k ∈ {1q, readout, 2q}

    ``ε̄_k`` is the attention-weighted error of the operands' physical qubits
    (``devices.phys``: 1q error, readout error, mean incident CZ error);
    ``s_k = exp(log_scale_k)`` are three learned global factors (init 1) that absorb
    the gap between the native gate set and the calibrated quantities.  The model
    learns compilation (counts, placement); the error magnitudes come from the
    device, so the prediction cannot drift to "large circuit ⇒ F ≈ 0" without
    predicting the operations that would cause it.
    """

    N_KINDS = 3  # 1q, readout, 2q — order of devices.phys

    def __init__(self, params: dict[str, Any], dev_node_dim: int, dev_edge_dim: int) -> None:
        super().__init__(params, dev_node_dim, dev_edge_dim)
        d = params["fusion_dim"]
        layers: list[nn.Module] = []
        last = 5 * d
        for width in params["mlp"]:
            layers += [nn.Linear(last, width), nn.LeakyReLU()]
            last = width
        out = nn.Linear(last, self.N_KINDS)
        # softplus(0.5413) = 1: one native operation of each kind per gate at init.
        nn.init.constant_(out.bias, 0.5413)
        layers.append(out)
        self.cost = nn.Sequential(*layers)
        self.log_scale = nn.Parameter(torch.zeros(self.N_KINDS))

    def forward(self, circuits, devices) -> torch.Tensor:
        g = self.gate_proj(self.circuit_encoder(circuits))
        q, gq_glob, valid = self._qubits(circuits, g)
        phys, mask, dev_glob = self._devices(devices)
        n_dev = phys.size(0)

        query = q.unsqueeze(0).expand(n_dev, -1, -1)
        ctx, attn = self.attn(query, phys, phys, key_padding_mask=~mask,
                              need_weights=True, average_attn_weights=True)
        ctx = ctx.transpose(0, 1)                                   # (Q, D, d)
        eps_dense, _ = to_dense_batch(devices.phys, devices.batch)  # (D, P, 3)
        q_eps = torch.einsum("dqp,dpk->qdk", attn, eps_dense)       # (Q, D, 3)

        idx = gq_glob.clamp(min=0)
        vmask = valid.unsqueeze(-1).unsqueeze(-1)
        count = valid.sum(1).clamp(min=1).view(-1, 1, 1).float()
        slots = ctx[idx]
        c_mean = (slots * vmask).sum(1) / count
        c_max = slots.masked_fill(~vmask, -1e4).amax(1)
        c_min = slots.masked_fill(~vmask, 1e4).amin(1)
        c_range = torch.where(valid.sum(1).view(-1, 1, 1) > 1, c_max - c_min,
                              torch.zeros_like(c_max))
        g_eps = (q_eps[idx] * vmask).sum(1) / count                  # (G, D, 3)

        g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
        d_e = dev_glob.unsqueeze(0).expand(g.size(0), -1, -1)
        z = torch.cat([g_e, c_mean, c_range, d_e, g_e * c_mean], dim=-1)
        n_ops = F.softplus(self.cost(z))                            # (G, D, 3)
        cost = (n_ops * g_eps * self.log_scale.exp()).sum(-1)        # (G, D)
        n_circ = circuits.n_qubits.view(-1).size(0)
        return -scatter(cost, circuits.batch, dim=0, dim_size=n_circ, reduce="sum")


#: v5 placement knobs (read with ``params.get`` so that v3/v4 configurations, and
#: therefore their checkpoints, are unchanged).
SINKHORN_DEFAULTS: dict[str, Any] = {"sinkhorn_tau": 0.3, "sinkhorn_iters": 20}


def _hop_distances(n: int, edge_index: torch.Tensor) -> torch.Tensor:
    """All-pairs shortest-path length (in couplers) on one coupling map."""
    inf = float(n + 1)
    dist = torch.full((n, n), inf)
    dist.fill_diagonal_(0.0)
    dist[edge_index[0], edge_index[1]] = 1.0
    for k in range(n):  # Floyd–Warshall; n <= 27
        dist = torch.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    return dist


class SinkhornPlacementPredictor(PhysicsHeadPredictor):
    """v5 ``sinkhorn``: the v4 physics head with a (soft) one-to-one placement.

    In v4 every logical qubit attends to the physical qubits independently, so two
    logical qubits may sit on the same physical qubit and the chip distance between
    the operands of a two-qubit gate — what decides the routing SWAPs — is blurred.

    v5 scores every (logical qubit, physical qubit) pair of a circuit on a device and
    normalises the score matrix with Sinkhorn iterations (log domain, temperature
    ``sinkhorn_tau``) until every logical qubit has total mass 1 and every physical
    qubit at most 1 (unused physical qubits go to dummy rows): a differentiable
    relaxation of a one-to-one layout.  It is a layer of the network, learned from
    the fidelity targets; nothing is compiled.

    With the placement ``A`` the head gets, for each gate, the *expected chip
    distance* between its operands, ``A_a^T H A_b`` (``H`` = hop distances on the
    coupling map, raw device structure), and, as in v4, the placement-weighted
    calibrated errors.  ``log F = -Σ_g Σ_k s_k n_k ε_k`` is unchanged.
    """

    def __init__(self, params: dict[str, Any], dev_node_dim: int, dev_edge_dim: int,
                 devices) -> None:
        super().__init__(params, dev_node_dim, dev_edge_dim)
        d = params["fusion_dim"]
        self.tau = params.get("sinkhorn_tau", SINKHORN_DEFAULTS["sinkhorn_tau"])
        self.iters = params.get("sinkhorn_iters", SINKHORN_DEFAULTS["sinkhorn_iters"])
        self.w_q = nn.Linear(d, d)
        self.w_p = nn.Linear(d, d)
        self.dummy_score = nn.Parameter(torch.zeros(()))
        # Count head: v4 inputs + [expected distance, log1p(expected distance)].
        layers: list[nn.Module] = []
        last = 5 * d + 2
        for width in params["mlp"]:
            layers += [nn.Linear(last, width), nn.LeakyReLU()]
            last = width
        out = nn.Linear(last, self.N_KINDS)
        nn.init.constant_(out.bias, 0.5413)
        layers.append(out)
        self.cost = nn.Sequential(*layers)
        # Hop-distance matrices of the (fixed) devices, padded to (D, P, P).
        sizes = torch.bincount(devices.batch).tolist()
        p_max, offset, mats = max(sizes), 0, []
        for n in sizes:
            sel = (devices.edge_index[0] >= offset) & (devices.edge_index[0] < offset + n)
            h = _hop_distances(n, devices.edge_index[:, sel] - offset)
            mats.append(F.pad(h, (0, p_max - n, 0, p_max - n)))
            offset += n
        self.register_buffer("hops", torch.stack(mats), persistent=False)

    def _sinkhorn(self, scores: torch.Tensor, n_logical: torch.Tensor,
                  n_phys: torch.Tensor) -> torch.Tensor:
        """``scores`` (B, D, Qm, P) -> soft assignment (B, D, Qm, P) for real rows."""
        b, n_dev, q_max, p_max = scores.shape
        n_rows = p_max  # real rows + dummy rows, square per device
        rows = torch.arange(n_rows, device=scores.device)
        cols = torch.arange(p_max, device=scores.device)
        real_row = rows.view(1, 1, -1) < n_logical.view(-1, 1, 1)               # (B, 1, R)
        used_row = rows.view(1, 1, -1) < n_phys.view(1, -1, 1)                  # (1, D, R)
        valid_col = cols.view(1, -1) < n_phys.view(-1, 1)                       # (D, P)
        # Dummy rows (unused physical qubits) share one learned score.
        dummy = self.dummy_score.expand(b, n_dev, n_rows - q_max, p_max)
        head = torch.where(real_row[..., :q_max].unsqueeze(-1), scores,
                           self.dummy_score.expand_as(scores))
        full = torch.cat([head, dummy], dim=2)
        mask = (used_row.unsqueeze(-1) & valid_col.view(1, n_dev, 1, p_max)
                ).expand(b, -1, -1, -1)
        # Masked cells get a large *finite* negative value: with -inf, logsumexp over an
        # all-masked row/column yields NaN gradients even where torch.where discards it.
        neg = torch.full_like(full, -1e4)
        log_a = torch.where(mask, full / self.tau, neg)
        for _ in range(self.iters):
            log_a = torch.where(mask, log_a - torch.logsumexp(log_a, -1, keepdim=True), neg)
            log_a = torch.where(mask, log_a - torch.logsumexp(log_a, -2, keepdim=True), neg)
        assign = torch.exp(log_a) * mask
        return assign[:, :, :q_max] * real_row[..., :q_max].unsqueeze(-1)

    def forward(self, circuits, devices) -> torch.Tensor:
        g = self.gate_proj(self.circuit_encoder(circuits))
        q, gq_glob, valid = self._qubits(circuits, g)
        phys, mask, dev_glob = self._devices(devices)                # (D, P, d)
        n_dev, p_max = phys.size(0), phys.size(1)
        n_q = circuits.n_qubits.view(-1).long()
        n_circ = n_q.size(0)
        q_batch = torch.repeat_interleave(torch.arange(n_circ, device=g.device), n_q)
        q_dense, q_mask = to_dense_batch(q, q_batch)                 # (B, Qm, d)

        scores = torch.einsum("bqk,dpk->bdqp", self.w_q(q_dense), self.w_p(phys))
        scores = scores / phys.size(-1) ** 0.5
        assign = self._sinkhorn(scores, n_q, mask.sum(1))           # (B, D, Qm, P)

        ctx = torch.einsum("bdqp,dpk->bdqk", assign, phys)           # (B, D, Qm, d)
        eps_dense, _ = to_dense_batch(devices.phys, devices.batch)   # (D, P, 3)
        q_eps = torch.einsum("bdqp,dpk->bdqk", assign, eps_dense)     # (B, D, Qm, 3)
        a_h = torch.einsum("bdqp,dpr->bdqr", assign, self.hops[:, :p_max, :p_max])

        gb = circuits.batch
        local = circuits.gate_qubits.clamp(min=0)                    # (G, 3)
        vmask = valid.unsqueeze(-1).unsqueeze(-1)                    # (G, 3, 1, 1)
        count = valid.sum(1).clamp(min=1).view(-1, 1, 1).float()
        slots = ctx[gb.unsqueeze(1), :, local]                       # (G, 3, D, d)
        c_mean = (slots * vmask).sum(1) / count
        c_max = slots.masked_fill(~vmask, -1e4).amax(1)
        c_min = slots.masked_fill(~vmask, 1e4).amin(1)
        multi = valid.sum(1).view(-1, 1, 1) > 1
        c_range = torch.where(multi, c_max - c_min, torch.zeros_like(c_max))
        g_eps = (q_eps[gb.unsqueeze(1), :, local] * vmask).sum(1) / count   # (G, D, 3)

        # Expected chip distance between operand pairs (mean over the pairs).
        dist = torch.zeros(g.size(0), n_dev, device=g.device)
        n_pairs = torch.zeros(g.size(0), 1, device=g.device)
        for a, b_ in ((0, 1), (0, 2), (1, 2)):
            m = (valid[:, a] & valid[:, b_]).float().unsqueeze(-1)
            d_ab = (a_h[gb, :, local[:, a]] * assign[gb, :, local[:, b_]]).sum(-1)
            dist = dist + m * d_ab
            n_pairs = n_pairs + m
        dist = dist / n_pairs.clamp(min=1)                           # (G, D)

        g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
        d_e = dev_glob.unsqueeze(0).expand(g.size(0), -1, -1)
        z = torch.cat([g_e, c_mean, c_range, d_e, g_e * c_mean,
                       dist.unsqueeze(-1), torch.log1p(dist).unsqueeze(-1)], dim=-1)
        n_ops = F.softplus(self.cost(z))                             # (G, D, 3)
        cost = (n_ops * g_eps * self.log_scale.exp()).sum(-1)
        return -scatter(cost, gb, dim=0, dim_size=n_circ, reduce="sum")


def build_model(kind: str, params: dict[str, Any], devices, device: torch.device) -> nn.Module:
    if kind == "pooled":
        return build_v2_model(params, devices, device)
    if kind == "sinkhorn":
        return SinkhornPlacementPredictor(
            params, dev_node_dim=devices.x.size(1), dev_edge_dim=devices.edge_attr.size(1),
            devices=devices.cpu(),
        ).to(device)
    if kind in ("xattn", "phys"):
        cls = QubitCrossAttentionPredictor if kind == "xattn" else PhysicsHeadPredictor
        return cls(
            params, dev_node_dim=devices.x.size(1), dev_edge_dim=devices.edge_attr.size(1)
        ).to(device)
    raise ValueError(f"unknown model kind {kind!r}")
