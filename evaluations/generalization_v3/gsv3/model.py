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
import math
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter, softmax, to_dense_batch

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

MODEL_KINDS = ("pooled", "xattn", "phys", "sinkhorn", "phys_uniform", "sinkhorn_rf", "sinkhorn_rf_rel", "sinkhorn_rf_route", "sinkhorn_rf_rel_route", "sinkhorn_rf_route_czc", "sinkhorn_rf_rel_route_czc", "sinkhorn_rf_route_usage", "sinkhorn_rf_route_czc_ep", "sinkhorn_rf_route_czc_ep3", "sinkhorn_rf_route_czc_mlps", "sinkhorn_rf_route_czc_ep3_mlps", "sinkhorn_rf_route_czc_vf2", "sinkhorn_rf_route_czc_esp", "sinkhorn_rf_route_czc_espa", "sinkhorn_rf_route_czc_espi", "sinkhorn_rf_route_czc_esps", "sinkhorn_rf_route_pgnn_usage")


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
        x_in, e_in = devices.x, devices.edge_attr
        if getattr(self, "relative_dev", False):
            x_in, e_in = relative_device_features(devices)
        x = F.leaky_relu(self.dev_in(x_in))
        for conv in self.dev_convs:
            x = x + F.leaky_relu(conv(x, devices.edge_index, e_in))
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

    def __init__(self, params: dict[str, Any], dev_node_dim: int, dev_edge_dim: int,
                 uniform_attention: bool = False) -> None:
        super().__init__(params, dev_node_dim, dev_edge_dim)
        # ``phys_uniform`` (diagnostic): no placement at all — every logical qubit
        # attends uniformly to the device's qubits, in the errors and in the context.
        self.uniform_attention = uniform_attention
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
        if self.uniform_attention:
            d = phys.size(-1)
            attn = mask.float().unsqueeze(1).expand(-1, q.size(0), -1) / mask.sum(1).view(-1, 1, 1)
            v = F.linear(phys, self.attn.in_proj_weight[2 * d:], self.attn.in_proj_bias[2 * d:])
            v_mean = (v * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)        # (D, d)
            ctx = self.attn.out_proj(v_mean).unsqueeze(0).expand(q.size(0), -1, -1)    # (Q, D, d)
        else:
            ctx, attn = self.attn(query, phys, phys, key_padding_mask=~mask,
                                  need_weights=True, average_attn_weights=True)
            ctx = ctx.transpose(0, 1)                               # (Q, D, d)
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
        # v8: predicted native-operation counts per circuit (B, D, 3), for the count loss.
        self.last_counts = scatter(n_ops, circuits.batch, dim=0, dim_size=n_circ, reduce="sum")
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


class PlacedCircuitGNN(nn.Module):
    """v12: message passing on the placed circuit, refining the v11 usage maps.

    Graph = the device's physical qubits.  Node p: v11 usage prior U_r(p), U_m(p), occupancy, total
    CZ demand, raw errors, the device embedding h_p.  Coupling edges: the v11 CZ prior U_cz(e), the
    coupler error, the direct demand between its two ends.  Demand edges: every pair (p, r) with
    CZ demand between the logical qubits placed there (A^T W A), gated by (demand, hop distance), so
    messages follow both the chip and the circuit's needs (where routing comes from).
    Outputs are corrections, zero at init (the model starts exactly as v11 up to ~1e-5):
      U_r' = U_r · exp(δ_r),  U_m' = U_m · exp(δ_m),  U_cz' = U_cz · exp(δ_cz) + softplus(γ)  (γ₀ = −12).
    """

    def __init__(self, d_dev: int, hidden: int = 64, layers: int = 4) -> None:
        super().__init__()
        h = hidden
        self.node_in = nn.Linear(7, h)
        self.dev_in = nn.Linear(d_dev, h, bias=False)
        self.edge_in = nn.Linear(3, h)
        self.edge_mlp = nn.ModuleList(nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h)) for _ in range(layers))
        self.dem_gate = nn.ModuleList(nn.Linear(3, h) for _ in range(layers))
        self.dem_val = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(layers))
        self.node_mlp = nn.ModuleList(nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h)) for _ in range(layers))
        self.node_ln = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.edge_ln = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.node_out = nn.Linear(h, 2)
        self.edge_out = nn.Linear(h, 2)
        for lin in (self.node_out, self.edge_out):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        nn.init.constant_(self.edge_out.bias[1:], -12.0)

    def forward(self, u_r, u_m, u_cz, occ, pair_m, hops, dev_h, eps_q, eps_e, inc_a, inc_b, qmask, emask):
        """u_r, u_m, occ (B, D, P); u_cz (B, D, E); pair_m (B, D, P, P); hops (D, P, P); dev_h (D, P, d);
        eps_q (D, P, 3); eps_e (D, E); inc_a / inc_b (D, E, P) one-hot ends of each coupler."""
        qm, em = qmask.float(), emask.float()
        dem = pair_m + pair_m.transpose(-1, -2)                                  # symmetric demand
        x = torch.stack([torch.log1p(u_r), torch.log1p(u_m), occ, torch.log1p(dem.sum(-1)),
                         *(100 * eps_q).unsqueeze(0).expand(u_r.size(0), -1, -1, -1).unbind(-1)], -1)
        hn = (self.node_in(x) + self.dev_in(dev_h).unsqueeze(0)) * qm.unsqueeze(0).unsqueeze(-1)
        d_e = torch.einsum("bdpr,dep,der->bde", dem, inc_a, inc_b)               # direct demand per coupler
        xe = torch.stack([torch.log1p(u_cz), (100 * eps_e).unsqueeze(0).expand_as(u_cz), torch.log1p(d_e)], -1)
        he = self.edge_in(xe) * em.unsqueeze(0).unsqueeze(-1)
        gfeat = torch.stack([torch.log1p(dem), (hops / 4).unsqueeze(0).expand_as(dem), torch.ones_like(dem)], -1)
        inc = inc_a + inc_b
        for l in range(len(self.edge_mlp)):
            ends = torch.einsum("bdph,dep->bdeh", hn, inc)
            me = self.edge_mlp[l](torch.cat([ends, he], -1)) * em.unsqueeze(0).unsqueeze(-1)
            he = self.edge_ln[l](he + me) * em.unsqueeze(0).unsqueeze(-1)
            agg_c = torch.einsum("bdeh,dep->bdph", me, inc)
            g = self.dem_gate[l](gfeat)                                          # (B, D, P, P, h)
            agg_d = torch.einsum("bdprh,bdrh->bdph", g, self.dem_val[l](hn)) / hn.size(2)
            hn = self.node_ln[l](hn + self.node_mlp[l](torch.cat([hn, agg_c, agg_d], -1))) * qm.unsqueeze(0).unsqueeze(-1)
        dn = self.node_out(hn).clamp(-5, 5)
        de = self.edge_out(he)
        u_r2 = u_r * torch.exp(dn[..., 0])
        u_m2 = u_m * torch.exp(dn[..., 1])
        u_cz2 = (u_cz * torch.exp(de[..., 0].clamp(-5, 5)) + F.softplus(de[..., 1])) * em.unsqueeze(0)
        return u_r2, u_m2, u_cz2


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

    v6 uses the same network and adds, in training only, a cross-entropy between
    ``A`` and the initial layout the compiler chose (``training.layout_loss``), so
    that the placement becomes "where the compiler will put the qubits".  v7 keeps
    the network too but supervises only family-agnostic properties of that layout:
    which physical qubits are used and how far apart interacting qubits are
    (``training.region_distance_loss``).
    """

    def __init__(self, params: dict[str, Any], dev_node_dim: int, dev_edge_dim: int,
                 devices, routing_flag: bool = False, routing_est: bool = False,
                 cz_constrained: bool = False, usage_head: bool = False, err_prior: int = 0,
                 mlp_scorer: bool = False, vf2_prior: bool = False, placed_gnn: bool = False,
                 esp_head: bool = False) -> None:
        super().__init__(params, dev_node_dim, dev_edge_dim)
        d = params["fusion_dim"]
        # ``sinkhorn_rf``: one more count-head input per (circuit, device), 1 if the circuit's
        # interaction graph embeds in the device coupling graph (no routing SWAP needed).
        self.routing_flag = routing_flag
        self._rf_cache: dict = {}
        # ``sinkhorn_rf_route`` (v8s): + a cheap routing-cost estimate per (circuit, device):
        # greedy placement of the interaction graph on the coupling map, then
        # log1p(Σ_pairs count · max(hops − 1, 0) / multi-qubit gates).
        self.routing_est = routing_est
        self._re_cache: dict = {}
        # ``*_czc`` (v8k/v9k): the CZ count is not free. Per multi-qubit gate
        #   cz = base(gate type, parameters) · (1 + κ · extra hops per 2q gate · (1 − routing-free)),
        # base depends on the gate alone (no circuit or device context) and κ ≥ 0 comes from the
        # count MLP, so a circuit that needs no routing gets exactly its decomposition count.
        self.cz_constrained = cz_constrained
        # ``*_usage`` (v11): the head works on usage maps instead of per-gate mean errors.
        #   U_r(p), U_m(p): the gates' r / measure counts spread over the physical qubits of their
        #                   operands through the placement A;
        #   U_cz(e):        each operand pair's CZ count spread over the couplers of the shortest
        #                   paths between its placed qubits, lower-error paths preferred
        #                   (softmax(-beta * path error), beta learned);
        #   log F = -(s_r Σ_p U_r ε_r + s_m Σ_p U_m ε_m + s_cz Σ_e U_cz ε_cz)   — exact for true maps.
        self.usage_head = usage_head
        # ``*_ep``: explicit preference for low-error physical qubits in the placement scores,
        #   score(q, p) -= w(q) * z_p,   z_p = error of p (r + readout + mean CZ) as a z-score within
        #   its device, w(q) = softplus(linear(h_q)) >= 0, started near 0 (behaves as without it).
        #   ``*_ep3``: one weight per error kind (1q, readout, mean CZ), each with its own z-score.
        self.err_prior = int(err_prior)
        if self.err_prior:
            self.err_w = nn.Linear(d, self.err_prior)
            nn.init.zeros_(self.err_w.weight)
            nn.init.constant_(self.err_w.bias, float(params.get("err_prior_init", -4.0)))   # softplus(-1.05) = 0.3
        # ``*_vf2``: the compiler's own layout score (Qiskit VF2Layout / VF2PostLayout) as a prior.
        #   cost(q, p) = s_r n_1q(q) ε_r(p) + s_m n_meas(q) ε_m(p) + s_cz n_cz(q) ε̄_cz(p)
        #   = the log-fidelity the circuit would lose if logical q sat on physical p, with the head's
        #   own scales s, gate counts read from the circuit (CZ via the constrained base count) and
        #   the unstandardised errors of p (the 2q term linearised: mean CZ of p's couplers).
        #   score(q, p) -= β · cost(q, p), β = exp(learned), one for routing-free circuits (VF2 finds
        #   such a layout and picks it on this score) and one for the others (Sabre, then
        #   VF2PostLayout rescoring).  Unlike ``*_ep3`` the weights are not per qubit and the errors
        #   are not z-scored: the score is in the same units as the fidelity it predicts.
        self.vf2_prior = vf2_prior
        if vf2_prior:
            assert cz_constrained, "the vf2 prior uses the constrained CZ base count"
            b0 = math.log(float(params.get("vf2_beta_init", 10.0)))
            self.vf2_log_beta = nn.Parameter(torch.tensor([b0, b0]))
        # ``*_mlps``: placement score = dot product + MLP(a, b, a ⊙ b) on the projected logical (a) and
        # physical (b) embeddings, so errors can combine non-linearly; last layer zero-initialised.
        self.mlp_scorer = mlp_scorer
        if mlp_scorer:
            hdim = 64
            self.sc_a, self.sc_b = nn.Linear(d, hdim), nn.Linear(d, hdim, bias=False)
            self.sc_ab = nn.Parameter(torch.randn(hdim, d) / d)
            self.sc_out = nn.Linear(hdim, 1)
            nn.init.zeros_(self.sc_out.weight); nn.init.zeros_(self.sc_out.bias)
        if usage_head:
            assert cz_constrained, "the usage head builds on the constrained CZ count"
            self.log_beta = nn.Parameter(torch.tensor(math.log(50.0)))
            self._paths: dict = {}
        # ``*_pgnn_usage`` (v12): a network on the placed circuit refines the v11 usage maps
        self.placed_gnn = PlacedCircuitGNN(d) if placed_gnn else None
        if placed_gnn:
            assert usage_head, "the placed-circuit network refines the usage maps"
        if cz_constrained:
            assert routing_flag and routing_est, "the constrained CZ count needs both routing inputs"
            self.cz_base = nn.Linear(N_GATE_LOCAL, 1)
            nn.init.constant_(self.cz_base.bias, 0.5413)   # softplus -> 1 CZ per gate at init
        self.tau = params.get("sinkhorn_tau", SINKHORN_DEFAULTS["sinkhorn_tau"])
        # the temperature travels with the weights (models fine-tuned at another tau); checkpoints
        # without it keep the constructor's value (see _load_from_state_dict)
        self.register_buffer("tau_saved", torch.tensor(float(self.tau)))
        self.iters = params.get("sinkhorn_iters", SINKHORN_DEFAULTS["sinkhorn_iters"])
        self.w_q = nn.Linear(d, d)
        self.w_p = nn.Linear(d, d)
        self.dummy_score = nn.Parameter(torch.zeros(()))
        # Count head: v4 inputs + [expected distance, log1p(expected distance)].
        layers: list[nn.Module] = []
        last = 5 * d + 2 + int(routing_flag) + int(routing_est)
        for width in params["mlp"]:
            layers += [nn.Linear(last, width), nn.LeakyReLU()]
            last = width
        out = nn.Linear(last, self.N_KINDS)
        nn.init.constant_(out.bias, 0.5413)
        layers.append(out)
        self.cost = nn.Sequential(*layers)
        # ``*_esp`` (v13): duration and depth of the compiled circuit and its estimated success
        # probability (MQT): log ESP = log F − Σ_active qubits idle_q / min(T1, T2) of their physical qubit,
        # idle_q = duration − busy_q.  busy_q = the count head's gates on q × the gate durations; the head
        # predicts the duration as busiest qubit × (1 + softplus(·)) (never below it) and log depth.
        # Inputs: circuit embedding, device embedding, log1p of the total counts, busiest qubit, active qubits.
        # ``*_espa`` (v14, analytic): the same three targets from the circuit's structure alone, no circuit or
        # device embedding (see ``esp_features``), so an unseen family gets the rule learned on the others.
        self.esp_head = {False: False, True: "v13"}.get(esp_head, esp_head)
        if self.esp_head:
            n_in = 2 * d + 5 if self.esp_head == "v13" else self.N_ESPA_IN + {"a": 0, "i": 1, "s": 3, "f": 3}[self.esp_head]
            self.esp_mlp = self.new_esp_mlp(n_in, self.esp_head)
        # Hop-distance matrices of the (fixed) devices, padded to (D, P, P).
        sizes = torch.bincount(devices.batch).tolist()
        p_max, offset, mats = max(sizes), 0, []
        for n in sizes:
            sel = (devices.edge_index[0] >= offset) & (devices.edge_index[0] < offset + n)
            h = _hop_distances(n, devices.edge_index[:, sel] - offset)
            mats.append(F.pad(h, (0, p_max - n, 0, p_max - n)))
            offset += n
        self.register_buffer("hops", torch.stack(mats), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        key = prefix + "tau_saved"
        had = key in state_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
        if not had and key in missing_keys:
            missing_keys.remove(key)
        self.tau = float(self.tau_saved)

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
        self._dev_h = phys                                           # v12: device embeddings for the placed-circuit network
        n_dev, p_max = phys.size(0), phys.size(1)
        n_q = circuits.n_qubits.view(-1).long()
        n_circ = n_q.size(0)
        q_batch = torch.repeat_interleave(torch.arange(n_circ, device=g.device), n_q)
        q_dense, q_mask = to_dense_batch(q, q_batch)                 # (B, Qm, d)

        scores = torch.einsum("bqk,dpk->bdqp", self.w_q(q_dense), self.w_p(phys))
        scores = scores / phys.size(-1) ** 0.5
        if self.mlp_scorer:
            a, b_ = self.w_q(q_dense), self.w_p(phys)                              # (B, Qm, d), (D, P, d)
            hid = (self.sc_a(a).unsqueeze(1).unsqueeze(3) + self.sc_b(b_).unsqueeze(0).unsqueeze(2)
                   + torch.einsum("bqk,dpk,hk->bdqph", a, b_, self.sc_ab))
            scores = scores + self.sc_out(F.relu(hid)).squeeze(-1)
        if self.err_prior:
            ph, _ = to_dense_batch(devices.phys, devices.batch)                    # (D, P, 3)
            e_p = ph.sum(-1, keepdim=True) if self.err_prior == 1 else ph           # (D, P, K)
            mf = mask.float().unsqueeze(-1)
            mu = (e_p * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True)
            sd = (((e_p - mu) ** 2 * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True)).sqrt().clamp(min=1e-6)
            z = (e_p - mu) / sd * mf                                                # (D, P, K)
            w = F.softplus(self.err_w(q_dense))                                     # (B, Qm, K)
            scores = scores - torch.einsum("bqk,dpk->bdqp", w, z)
            self.last_err_w = w
        if self.vf2_prior:
            scores = scores - self._vf2_cost(circuits, devices, g, q_dense.size(1), n_circ, valid)
        assign = self._sinkhorn(scores, n_q, mask.sum(1))           # (B, D, Qm, P)
        # Kept for the v6 layout loss (training.layout_loss); not used by the head.
        self.last_assign, self.last_q_mask = assign, q_mask

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
        # Kept for the v7 layout loss (expected operand distance per gate, column mask).
        self.last_dist, self.last_col_mask = dist, mask

        g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
        d_e = dev_glob.unsqueeze(0).expand(g.size(0), -1, -1)
        feats = [g_e, c_mean, c_range, d_e, g_e * c_mean,
                 dist.unsqueeze(-1), torch.log1p(dist).unsqueeze(-1)]
        if self.routing_flag:
            rf = routing_free(circuits, devices, self._rf_cache).to(g.device)   # (B, D)
            feats.append(rf[gb].unsqueeze(-1))
        if self.routing_est:
            re_ = routing_estimate(circuits, devices, self._re_cache).to(g.device)  # (B, D)
            feats.append(re_[gb].unsqueeze(-1))
        z = torch.cat(feats, dim=-1)
        n_ops = F.softplus(self.cost(z))                             # (G, D, 3)
        if self.cz_constrained:
            base = F.softplus(self.cz_base(circuits.x[:, :N_GATE_LOCAL])) * multi.view(-1, 1)  # (G, 1)
            extra = torch.expm1(re_)[gb] * (1 - rf[gb])              # (G, D) extra hops per 2q gate
            cz = base * (1 + n_ops[..., 2] * extra)
            n_ops = torch.cat([n_ops[..., :2], cz.unsqueeze(-1)], dim=-1)
        n_ops_esp = n_ops
        if self.esp_head in ("i", "s", "f"):
            # compilation never adds or removes a measurement -> one per logical measure gate.  ``*_espi`` (v16) uses
            # it for F too; ``*_esps`` (v17) only in the ESP head, so F keeps the counts it was trained with
            meas = (circuits.x[:, GATE2IDX_MEASURE] > 0.5).float().view(-1, 1).expand(-1, n_dev)
            n_ops_esp = torch.cat([n_ops[..., :1], meas.unsqueeze(-1), n_ops[..., 2:]], dim=-1)
            if self.esp_head == "i":
                n_ops = n_ops_esp
        self.last_counts = scatter(n_ops, gb, dim=0, dim_size=n_circ, reduce="sum")   # v8
        self.last_n_ops = n_ops                                      # (G, D, 3) per gate: r, measure, cz
        if self.esp_head and hasattr(devices, "coh"):     # device batches built elsewhere may lack T1/T2
            self.last_esp_feats = self.esp_features(circuits, devices, g, gb, n_circ, n_ops_esp, valid, local, assign, dev_glob)
            self.last_esp = self.esp_from_features(self.esp_mlp, self.last_esp_feats, self.esp_head)
        if self.usage_head:
            return self._usage_log_f(circuits, devices, assign, n_ops, valid, local, gb, n_circ)
        cost = (n_ops * g_eps * self.log_scale.exp()).sum(-1)
        return -scatter(cost, gb, dim=0, dim_size=n_circ, reduce="sum")

    N_ESPA_IN = 10

    @staticmethod
    def new_esp_mlp(n_in: int, kind: str) -> nn.Sequential:
        n_out = {"i": 3, "s": 4, "f": 4}.get(kind, 2)      # espi: + idle factor; esps: + idle factor, ancilla idle
        mlp = nn.Sequential(nn.Linear(n_in, 128), nn.LeakyReLU(), nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, n_out))
        nn.init.zeros_(mlp[-1].weight)
        with torch.no_grad():   # v13: duration ≈ 1.7 × busiest qubit, depth ≈ 55; espa/espi: 1.7 × gate time, 2 × operations
            mlp[-1].bias.copy_(torch.tensor([0.0, 4.0]) if kind == "v13" else
                               torch.tensor([0.0, 0.0, 0.0, -4.0]) if kind in ("s", "f") else torch.zeros(n_out))
        return mlp

    def esp_features(self, circuits, devices, g, gb, n_circ, n_ops, valid, local, assign, dev_glob) -> dict:
        """Everything the ESP heads read except their MLP (fixed for a frozen backbone, so probes can cache it).

        Per logical qubit q, the count head's gates spread over their operands (1-qubit gates and measurements
        split over the operands, every CZ occupies two): busy_q (µs, measurements included), gate time t_q
        (µs, no measurements), measurements m_q, operations o_q.  inv_t = 1 / min(T1, T2) through the placement."""
        from .devices import GATE_US
        if self.esp_head == "f":
            assign = hard_placement(assign)
        b, n_dev, q_max, _ = assign.shape
        cnt = valid.sum(1).clamp(min=1).float().unsqueeze(1)                      # operands per gate
        two = torch.where(cnt > 1, 2.0 / cnt, torch.zeros_like(cnt))
        t_g = self.gate_times(devices, assign, valid, local, gb)                   # (G, D, 3) µs: r, measure, cz
        per = torch.stack([n_ops[..., 0] * t_g[..., 0] / cnt + n_ops[..., 2] * t_g[..., 2] * two,
                           n_ops[..., 1] / cnt,
                           (n_ops[..., 0] + n_ops[..., 1]) / cnt + n_ops[..., 2] * two,
                           n_ops[..., 1] * t_g[..., 1] / cnt], -1)   # (G, D, 4)
        acc = torch.zeros(b * q_max, n_dev, 4, device=g.device)
        for k in range(local.size(1)):
            m = valid[:, k]
            acc = acc.index_add(0, (gb * q_max + local[:, k])[m], per[m])
        t_q, m_q, o_q, mt_q = acc.view(b, q_max, n_dev, 4).permute(0, 2, 1, 3).unbind(-1)   # (B, D, Qm)
        busy = t_q + mt_q
        active = (busy > 0).float()
        coh, _ = to_dense_batch(devices.coh, devices.batch)                          # (D, P) 1/µs
        f = {"busy": busy, "active": active, "inv_t": torch.einsum("bdqp,dp->bdq", assign, coh),
             "counts": self.last_counts, "tmax": t_q.amax(-1), "mmax": m_q.amax(-1), "omax": o_q.amax(-1),
             "mtmax": mt_q.amax(-1)}
        if self.esp_head == "v13" or getattr(self, "esp_all_features", False):   # probes cache both heads' inputs
            g_pool = scatter(g, gb, dim=0, dim_size=n_circ, reduce="mean")
            f["emb"] = torch.cat([g_pool.unsqueeze(1).expand(-1, n_dev, -1),
                                  dev_glob.unsqueeze(0).expand(n_circ, -1, -1)], -1)
        if self.esp_head in ("a", "i", "s", "f") or getattr(self, "esp_all_features", False):
            f["rf"] = routing_free(circuits, devices, self._rf_cache).to(g.device)
            f["re"] = routing_estimate(circuits, devices, self._re_cache).to(g.device)
            f["n_phys"] = torch.bincount(devices.batch).float().to(g.device).expand(n_circ, -1)
        if self.esp_head in ("s", "f"):
            if not hasattr(self, "_lvl_cache"):
                self._lvl_cache = {}
            f.update(dag_schedule(circuits, n_ops, self._lvl_cache, gate_t=t_g))
        if self.esp_head == "f":
            # v19: the MLP only reads time-free inputs (the same schedule with the uniform reference durations), the
            # formula gets the real ones -> a change of gate times or T1 / T2 moves the ESP only through the formula
            ref = dag_schedule(circuits, n_ops, self._lvl_cache, gate_t=None)
            f.update({k + "_ref": v for k, v in ref.items() if k != "idle_q"})
            per_r = torch.stack([n_ops[..., 0] * GATE_US[0] / cnt + n_ops[..., 2] * GATE_US[2] * two,
                                 n_ops[..., 1] * GATE_US[1] / cnt], -1)
            acc_r = torch.zeros(b * q_max, n_dev, 2, device=g.device)
            for k in range(local.size(1)):
                m = valid[:, k]
                acc_r = acc_r.index_add(0, (gb * q_max + local[:, k])[m], per_r[m])
            t_r, mt_r = acc_r.view(b, q_max, n_dev, 2).permute(0, 2, 1, 3).unbind(-1)
            f["tmax_ref"], f["mtmax_ref"] = t_r.amax(-1), mt_r.amax(-1)
        return f

    @staticmethod
    def gate_times(devices, assign, valid, local, gb) -> torch.Tensor:
        """v18: (G, D, 3) durations in µs of r, measure and CZ for every logical gate on every device: each physical
        qubit's times (``devices.tg``, uniform GATE_US when absent) through the placement, averaged over the gate's
        operands.  With uniform times this is exactly GATE_US, so models trained before v18 are unchanged."""
        from .devices import GATE_US
        b, n_dev, q_max, _ = assign.shape
        n_g = local.size(0)
        if not hasattr(devices, "tg"):
            return assign.new_tensor(GATE_US).expand(n_g, n_dev, 3)
        tg, _ = to_dense_batch(devices.tg, devices.batch)                            # (D, P, 3)
        tq = torch.einsum("bdqp,dpk->bqdk", assign, tg).reshape(b * q_max, n_dev, 3)
        cnt = valid.sum(1).clamp(min=1).float().view(-1, 1, 1)
        acc = assign.new_zeros(n_g, n_dev, 3)
        for k in range(local.size(1)):
            acc = acc + tq[gb * q_max + local[:, k]] * valid[:, k].float().view(-1, 1, 1)
        return acc / cnt

    @staticmethod
    def esp_from_features(mlp, f: dict, kind: str):
        """(ESP decay = log ESP − log F, log duration in µs, log depth), each (B, D).

        v13:  duration = busiest qubit (measurements included) × (1 + softplus(a)); log depth = b; the MLP reads the
              circuit and device embeddings, the totals, the busiest qubit and the active qubits.
        espa: duration = T_meas · max_q m_q + max_q t_q · (1 + softplus(a))  (measurement layers + gate critical path);
              depth = max(1, max_q o_q) · exp(softplus(b))  (never below the busiest qubit's operations);
              the MLP reads only log1p of the totals, max t_q, max m_q, max o_q, active qubits, the routing-free flag,
              the routing estimate and the device size."""
        from .devices import GATE_US
        busy, active = f["busy"], f["active"]
        if kind == "v13":
            bmax = busy.amax(-1)
            z = torch.cat([f["emb"], torch.log1p(f["counts"]), torch.log1p(bmax).unsqueeze(-1),
                           torch.log1p(active.sum(-1)).unsqueeze(-1)], -1)
            out = mlp(z)
            dur = bmax * (1 + F.softplus(out[..., 0])) + 1e-3
            logdep = out[..., 1]
        else:
            st = torch.stack([f["tmax"], f["mmax"], f["omax"], active.sum(-1)], -1)
            z = [torch.log1p(f["counts"]), torch.log1p(st), f["rf"].unsqueeze(-1), f["re"].unsqueeze(-1),
                 torch.log(f["n_phys"]).unsqueeze(-1)]
            if kind == "f":
                # v19 formula head: same inputs as v17 but at the reference durations; per-qubit decay
                st = torch.stack([f["tmax_ref"], f["mmax"], f["omax"], active.sum(-1)], -1)
                z = [torch.log1p(f["counts"]), torch.log1p(st), f["rf"].unsqueeze(-1), f["re"].unsqueeze(-1),
                     torch.log(f["n_phys"]).unsqueeze(-1), torch.log1p(f["dur_s_ref"]).unsqueeze(-1),
                     torch.log1p(f["idle_s_ref"]).unsqueeze(-1), torch.log1p(f["dep_s_ref"]).unsqueeze(-1)]
                out = mlp(torch.cat(z, -1))
                dur = f["dur_s"] * torch.exp(out[..., 0]) + 1e-3
                logdep = torch.log(f["dep_s"].clamp(min=1)) + out[..., 1]
                inv_m = (active * f["inv_t"]).sum(-1) / active.sum(-1).clamp(min=1)
                # Σ_q idle_q / min(T1, T2)_q on the logical qubits' own schedule; routing-only qubits as in v17
                own = (f["idle_q"] * f["inv_t"]).sum(-1)
                return -(own * torch.exp(out[..., 2]) + f["dur_s"] * F.softplus(out[..., 3]) * inv_m), torch.log(dur), logdep
            if kind == "s":
                z += [torch.log1p(f["dur_s"]).unsqueeze(-1), torch.log1p(f["idle_s"]).unsqueeze(-1),
                      torch.log1p(f["dep_s"]).unsqueeze(-1)]
                out = mlp(torch.cat(z, -1))
                # v17: the schedule of the circuit's own DAG is the base; the MLP only corrects it.  Idle adds a term
                # for physical qubits that only routing SWAPs touch (not in the logical DAG): duration × softplus(·).
                dur = f["dur_s"] * torch.exp(out[..., 0]) + 1e-3
                logdep = torch.log(f["dep_s"].clamp(min=1)) + out[..., 1]
                idle = (f["idle_s"] + 1e-3) * torch.exp(out[..., 2]) + f["dur_s"] * F.softplus(out[..., 3])
                inv_m = (active * f["inv_t"]).sum(-1) / active.sum(-1).clamp(min=1)
                return -idle * inv_m, torch.log(dur), logdep
            if kind == "i":
                # structural idle time: every active qubit waits for the busiest gate chain and the most measurements
                base = (((f["tmax"] + f["mtmax"]).unsqueeze(-1) - busy).clamp(min=0) * active).sum(-1)
                z.append(torch.log1p(base).unsqueeze(-1))
            out = mlp(torch.cat(z, -1))
            dur = f["mtmax"] + f["tmax"] * (1 + F.softplus(out[..., 0])) + 1e-3
            logdep = torch.log(f["omax"].clamp(min=1)) + F.softplus(out[..., 1])
            if kind == "i":
                # v16: total idle time predicted directly (not as duration − busy, a small difference of large
                # numbers) times the mean 1 / min(T1, T2) of the placed active qubits
                idle = (base + 1e-3) * torch.exp(out[..., 2])
                inv_m = (active * f["inv_t"]).sum(-1) / active.sum(-1).clamp(min=1)
                return -idle * inv_m, torch.log(dur), logdep
        decay = -(((dur.unsqueeze(-1) - busy).clamp(min=0) * active) * f["inv_t"]).sum(-1)
        return decay, torch.log(dur), logdep

    def _vf2_cost(self, circuits, devices, g, q_max, n_circ, valid):
        """(B, D, Qm, P) β · estimated log-fidelity lost by putting logical q on physical p."""
        gb = circuits.batch
        local = circuits.gate_qubits.clamp(min=0)
        n_op = valid.sum(1)
        meas = circuits.x[:, GATE2IDX_MEASURE] > 0.5
        one = ((n_op == 1) & ~meas).float()
        cz = F.softplus(self.cz_base(circuits.x[:, :N_GATE_LOCAL])).squeeze(-1) * (n_op > 1)
        per = torch.stack([one, meas.float(), cz], -1)              # (G, 3) per operand
        cnt = torch.zeros(n_circ * q_max, 3, device=g.device)
        for k in range(local.size(1)):
            m = valid[:, k]
            cnt = cnt.index_add(0, (gb * q_max + local[:, k])[m], per[m])
        cnt = cnt.view(n_circ, q_max, 3) * self.log_scale.exp()
        eps, _ = to_dense_batch(devices.phys, devices.batch)       # (D, P, 3)
        cost = torch.einsum("bqk,dpk->bdqp", cnt, eps)
        rf = routing_free(circuits, devices, self._rf_cache).to(g.device)     # (B, D)
        beta = self.vf2_log_beta.exp()
        b_ = rf * beta[0] + (1 - rf) * beta[1]
        self.last_vf2_beta = beta
        return b_.view(n_circ, -1, 1, 1) * cost

    def _incidence(self, devices, p_max: int, e_max: int):
        """(D, E, P) one-hot first / second end of each coupler (zero rows for padding)."""
        n_c = devices.n_coupler.view(-1).tolist()
        dev = devices.coupler_pairs.device
        inc_a = torch.zeros(len(n_c), e_max, p_max, device=dev)
        inc_b = torch.zeros_like(inc_a)
        off = 0
        for d, E in enumerate(n_c):
            pr = devices.coupler_pairs[off: off + E]
            off += E
            inc_a[d, torch.arange(E, device=dev), pr[:, 0]] = 1.0
            inc_b[d, torch.arange(E, device=dev), pr[:, 1]] = 1.0
        return inc_a, inc_b

    # ------------------------------------------------------------------ v11 usage maps
    def _route_kernel(self, devices, p_max: int):
        """(D, P, P, E) share of a CZ between physical qubits (p, r) on each coupler: mean over
        the shortest-hop paths p -> r weighted by softmax(-beta * path error), divided by the
        path length (so a pair's shares sum to 1); plus (D, E) errors and (D, E) mask."""
        n_c = devices.n_coupler.view(-1).tolist()
        sizes = torch.bincount(devices.batch).tolist()
        e_max = max(n_c)
        beta = self.log_beta.exp()
        dev = devices.coupler_eps.device
        K = torch.zeros(len(n_c), p_max * p_max, e_max, device=dev)
        eps = torch.zeros(len(n_c), e_max, device=dev)
        emask = torch.zeros(len(n_c), e_max, dtype=torch.bool, device=dev)
        off = 0
        for d, (n, E) in enumerate(zip(sizes, n_c)):
            pairs = devices.coupler_pairs[off: off + E]
            e_d = devices.coupler_eps[off: off + E]
            off += E
            key = tuple(map(tuple, pairs.tolist()))
            if key not in self._paths:
                self._paths[key] = _shortest_paths(n, key)
            pid, s_t, inc, hops = (t.to(dev) for t in self._paths[key])
            err = inc @ e_d                                          # (Npath,)
            w = softmax(-beta * err, pid)                            # per (s, t) pair
            share = (w / hops).unsqueeze(-1) * inc                   # (Npath, E)
            idx = s_t[:, 0] * p_max + s_t[:, 1]
            idx_rev = s_t[:, 1] * p_max + s_t[:, 0]
            K[d, :, :E] = K[d, :, :E].index_add(0, idx, share).index_add(0, idx_rev, share)
            eps[d, :E] = e_d
            emask[d, :E] = True
        return K.view(len(n_c), p_max, p_max, e_max), eps, emask

    def _usage_log_f(self, circuits, devices, assign, n_ops, valid, local, gb, n_circ):
        b, n_dev, q_max, p_max = assign.shape
        cnt = valid.sum(1).clamp(min=1).float()                      # operands per gate
        n_r = torch.zeros(b * q_max, n_dev, device=assign.device)
        n_m = torch.zeros_like(n_r)
        for k in range(local.size(1)):
            m = valid[:, k]
            idx = (gb * q_max + local[:, k])[m]
            n_r = n_r.index_add(0, idx, (n_ops[..., 0] / cnt.unsqueeze(1))[m])
            n_m = n_m.index_add(0, idx, (n_ops[..., 1] / cnt.unsqueeze(1))[m])
        u_r = torch.einsum("bqd,bdqp->bdp", n_r.view(b, q_max, n_dev), assign)
        u_m = torch.einsum("bqd,bdqp->bdp", n_m.view(b, q_max, n_dev), assign)
        n2 = torch.zeros(b * q_max * q_max, n_dev, device=assign.device)
        npair = sum((valid[:, a] & valid[:, c]).float() for a, c in ((0, 1), (0, 2), (1, 2))).clamp(min=1)
        for a, c in ((0, 1), (0, 2), (1, 2)):
            m = valid[:, a] & valid[:, c]
            idx = (gb * q_max * q_max + local[:, a] * q_max + local[:, c])[m]
            n2 = n2.index_add(0, idx, (n_ops[..., 2] / npair.unsqueeze(1))[m])
        pair_m = torch.einsum("bdqp,bqsd,bdsr->bdpr", assign, n2.view(b, q_max, q_max, n_dev), assign)
        K, eps_e, emask = self._route_kernel(devices, p_max)
        u_cz = torch.einsum("bdpr,dpre->bde", pair_m, K)
        eps_q, qmask = to_dense_batch(devices.phys, devices.batch)  # (D, P, 3)
        if self.placed_gnn is not None:
            inc_a, inc_b = self._incidence(devices, p_max, eps_e.size(1))
            occ = assign.sum(2)
            u_r, u_m, u_cz = self.placed_gnn(u_r, u_m, u_cz, occ, pair_m, self.hops[:, :p_max, :p_max],
                                             self._dev_h, eps_q, eps_e, inc_a, inc_b, qmask, emask)
        s = self.log_scale.exp()
        log_f = -(s[0] * (u_r * eps_q[..., 0]).sum(-1) + s[1] * (u_m * eps_q[..., 1]).sum(-1)
                  + s[2] * (u_cz * eps_e).sum(-1))
        self.last_usage = (u_r, u_m, u_cz, qmask, emask)
        return log_f


# Gate-local inputs of the constrained CZ base: type one-hot (28), sin/cos of 3 parameters (6),
# arity, control qubits, number of parameters — nothing about the circuit or the device.
N_GATE_LOCAL = 28 + 6 + 3
GATE2IDX_MEASURE = 27   # one-hot column of "measure" (last of fast_encoding.UNIQUE)
N_RELATIVE = 5   # per-device z-scores of the first 5 qubit features (1q err, readout, T1, T2, degree)


def relative_device_features(devices):
    """v9: append to each qubit its first ``N_RELATIVE`` features as z-scores WITHIN its own
    device, and to each coupler its CZ error as a z-score within its device.  The compiler
    picks qubits by relative quality ("the better ones of this chip"); relative features give
    the same signal whatever the absolute calibration, so they should transfer to calibrations
    the model has never seen.  Affine-invariant, so standardised inputs give the same values."""
    b = devices.batch
    x = devices.x[:, :N_RELATIVE]
    mean = scatter(x, b, dim=0, reduce="mean")
    std = (scatter((x - mean[b]) ** 2, b, dim=0, reduce="mean")).sqrt().clamp(min=1e-6)
    xr = (x - mean[b]) / std[b]
    eb = b[devices.edge_index[0]]
    e = devices.edge_attr
    emean = scatter(e, eb, dim=0, reduce="mean", dim_size=int(b.max()) + 1)
    estd = (scatter((e - emean[eb]) ** 2, eb, dim=0, reduce="mean", dim_size=int(b.max()) + 1)).sqrt().clamp(min=1e-6)
    er = (e - emean[eb]) / estd[eb]
    return torch.cat([devices.x, xr], dim=1), torch.cat([e, er], dim=1)


def _greedy_route_cost(pairs: dict, H, pdeg) -> float:
    """Σ count · max(hops − 1, 0) of the interaction graph under a greedy placement."""
    w: dict = {}
    for (a, b), c in pairs.items():
        w.setdefault(a, {})[b] = c
        w.setdefault(b, {})[a] = c
    pos, free = {}, set(range(len(pdeg)))
    for q in sorted(w, key=lambda q: -sum(w[q].values())):
        placed = [(pos[o], c) for o, c in w[q].items() if o in pos]
        if not placed:
            best = max(free, key=lambda p: pdeg[p])
        else:
            best = min(free, key=lambda p: (sum(c * H[p][po] for po, c in placed), -pdeg[p]))
        pos[q] = best
        free.discard(best)
    return float(sum(c * max(H[pos[a]][pos[b]] - 1, 0) for (a, b), c in pairs.items()))


def _shortest_paths(n: int, pairs: tuple, cap: int = 64):
    """All shortest-hop paths between every two qubits of a coupling map (at most ``cap`` per
    pair, in a fixed order): (pair id, (s, t), edge incidence (Npath, E), hops) as tensors."""
    import collections
    adj = collections.defaultdict(list)
    for e, (a, c) in enumerate(pairs):
        adj[a].append((c, e)); adj[c].append((a, e))
    pid, st, rows, hops = [], [], [], []
    k = 0
    for s0 in range(n):
        dist = {s0: 0}; front = [s0]; preds = collections.defaultdict(list)
        while front:
            nxt = []
            for u in front:
                for v, e in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1; nxt.append(v)
                    if dist[v] == dist[u] + 1:
                        preds[v].append((u, e))
            front = nxt
        for t in range(s0 + 1, n):
            if t not in dist:
                continue
            paths = [[]]
            stack = [(t, [])]
            paths = []
            while stack and len(paths) < cap:
                v, es = stack.pop()
                if v == s0:
                    paths.append(es); continue
                for u, e in preds[v]:
                    stack.append((u, es + [e]))
            for es in paths:
                row = [0.0] * len(pairs)
                for e in es:
                    row[e] = 1.0
                pid.append(k); st.append((s0, t)); rows.append(row); hops.append(float(len(es)))
            k += 1
    return (torch.tensor(pid), torch.tensor(st), torch.tensor(rows), torch.tensor(hops))


def routing_estimate(circuits, devices, cache: dict | None = None) -> torch.Tensor:
    """``(B, D)``: log1p(greedy routing cost / number of multi-qubit gates), cached by
    (circuit name, coupling map).  Cost ≈ 0.5 ms per circuit for 3 topologies (Python)."""
    n_q = circuits.n_qubits.view(-1).tolist()
    gq = circuits.gate_qubits.cpu()
    gb = circuits.batch.cpu()
    names = getattr(circuits, "circuit_name", None)
    ei = devices.edge_index.cpu()
    sizes = torch.bincount(devices.batch.cpu()).tolist()
    chips, offset = [], 0
    for n in sizes:
        sel = (ei[0] >= offset) & (ei[0] < offset + n)
        edges = tuple(sorted({(min(a, b), max(a, b)) for a, b in (ei[:, sel] - offset).t().tolist()}))
        offset += n
        chips.append((n, edges))
    tables = {}
    out = torch.zeros(len(n_q), len(chips))
    starts = torch.searchsorted(gb, torch.arange(len(n_q) + 1)).tolist()
    for b in range(len(n_q)):
        keys = [(names[b], c[1]) if names is not None else None for c in chips]
        if cache is not None and all(k is not None and k in cache for k in keys):
            out[b] = torch.tensor([cache[k] for k in keys])
            continue
        pairs: dict = {}
        multi = 0
        for row in gq[starts[b]: starts[b + 1]].tolist():
            q = [x for x in row if x >= 0]
            if len(q) > 1:
                multi += 1
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    k = (min(q[i], q[j]), max(q[i], q[j]))
                    pairs[k] = pairs.get(k, 0) + 1
        for dd, (n, edges) in enumerate(chips):
            if edges not in tables:
                adj = [[] for _ in range(n)]
                for a, c in edges:
                    adj[a].append(c); adj[c].append(a)
                H = []
                for s0 in range(n):
                    dist = [n + 1] * n; dist[s0] = 0; front = [s0]
                    while front:
                        nxt = []
                        for u in front:
                            for v in adj[u]:
                                if dist[v] > dist[u] + 1:
                                    dist[v] = dist[u] + 1; nxt.append(v)
                        front = nxt
                    H.append(dist)
                tables[edges] = (H, [len(a) for a in adj])
            val = 0.0
            if pairs:
                val = float(np.log1p(_greedy_route_cost(pairs, *tables[edges]) / max(multi, 1)))
            out[b, dd] = val
            if keys[dd] is not None and cache is not None:
                cache[keys[dd]] = val
    return out


def routing_free(circuits, devices, cache: dict | None = None) -> torch.Tensor:
    """``(B, D)``: 1 if circuit b needs no routing on device d, i.e. its interaction graph
    (pairs of logical qubits sharing a gate) is a subgraph of the coupling graph — the
    perfect layout Qiskit's VF2Layout looks for at level 2.  rustworkx VF2 with a call
    limit (undecided -> 0).  Cached by (circuit name, coupling map) when names exist."""
    import rustworkx as rx
    n_q = circuits.n_qubits.view(-1).tolist()
    gq = circuits.gate_qubits.cpu()
    gb = circuits.batch.cpu()
    names = getattr(circuits, "circuit_name", None)
    if cache is not None and cache.get("_dev_id") == id(devices):
        chips = cache["_chips"]
    else:
        chips = None
    ei = devices.edge_index.cpu() if chips is None else None
    if chips is None:
        chips = []
        sizes = torch.bincount(devices.batch.cpu()).tolist()
        offset = 0
        for n in sizes:
            sel = (ei[0] >= offset) & (ei[0] < offset + n)
            edges = tuple(sorted({(min(a, b), max(a, b)) for a, b in (ei[:, sel] - offset).t().tolist()}))
            offset += n
            chips.append(edges)
        if cache is not None:
            cache["_dev_id"], cache["_chips"] = id(devices), chips
    chip_graphs = {}
    out = torch.zeros(len(n_q), len(chips))
    starts = torch.searchsorted(gb, torch.arange(len(n_q) + 1)).tolist()
    for b in range(len(n_q)):
        keys = [(names[b], edges) if names is not None else None for edges in chips]
        if cache is not None and all(k is not None and k in cache for k in keys):
            out[b] = torch.tensor([cache[k] for k in keys])
            continue
        pairs = set()
        for row in gq[starts[b]: starts[b + 1]].tolist():
            q = [x for x in row if x >= 0]
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    pairs.add((min(q[i], q[j]), max(q[i], q[j])))
        for dd, edges in enumerate(chips):
            key = keys[dd]
            if key is not None and cache is not None and key in cache:
                out[b, dd] = cache[key]
                continue
            if edges not in chip_graphs:
                c = rx.PyGraph(); c.add_nodes_from(range(1 + max(max(e) for e in edges))); c.add_edges_from_no_data(list(edges))
                chip_graphs[edges] = c
            if not pairs:
                val = 1.0
            else:
                used = sorted({q for p in pairs for q in p})
                pos = {q: i for i, q in enumerate(used)}
                g = rx.PyGraph(); g.add_nodes_from(range(len(used)))
                g.add_edges_from_no_data([(pos[a], pos[c]) for a, c in pairs])
                val = float(len(used) <= chip_graphs[edges].num_nodes() and rx.is_subgraph_isomorphic(
                    chip_graphs[edges], g, id_order=False, induced=False, call_limit=20_000))
            out[b, dd] = val
            if key is not None and cache is not None:
                cache[key] = val
    return out



def _dag_levels(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Topological level of every gate of ONE circuit (longest path in gates from a source), Kahn's algorithm."""
    src, dst = edge_index
    indeg = np.bincount(dst, minlength=n)
    out = [[] for _ in range(n)]
    for a, b in zip(src.tolist(), dst.tolist()):
        out[a].append(b)
    lvl = np.zeros(n, dtype=np.int64)
    stack = [i for i in range(n) if indeg[i] == 0]
    indeg = indeg.copy()
    while stack:
        a = stack.pop()
        for b in out[a]:
            lvl[b] = max(lvl[b], lvl[a] + 1)
            indeg[b] -= 1
            if indeg[b] == 0:
                stack.append(b)
    return lvl


@torch.no_grad()
def hard_placement(assign: torch.Tensor) -> torch.Tensor:
    """v19: one physical qubit per logical qubit (Hungarian on the Sinkhorn matrix, per circuit and device), with the
    soft matrix's gradient (straight-through).  The soft matrix spreads a logical qubit over ~5 physical ones, which
    averages away any per-qubit change of gate times or T1 / T2.  Padding rows (all zero) stay zero."""
    from scipy.optimize import linear_sum_assignment
    a = assign.detach().float().cpu().numpy()
    hard = np.zeros_like(a)
    for i in range(a.shape[0]):
        for d in range(a.shape[1]):
            m = a[i, d]
            rows = np.flatnonzero(m.sum(1) > 0)
            cols = np.flatnonzero(m.sum(0) > 0)
            if len(rows) == 0 or len(cols) < len(rows):
                hard[i, d] = m
                continue
            r, c = linear_sum_assignment(-m[np.ix_(rows, cols)])
            hard[i, d, rows[r], cols[c]] = 1.0
    hard = torch.from_numpy(hard).to(assign)
    return hard + assign - assign.detach()


def dag_schedule(circuits, n_ops: torch.Tensor, cache: dict | None = None, gate_t: torch.Tensor | None = None) -> dict:
    """v17: ASAP schedule of each circuit's logical gate DAG with every gate weighted by its compiled cost.

    n_ops (G, D, 3) = r / measure / CZ per gate and device (measurements exact upstream).  Each operand of a gate is
    busy for  r · 42 ns / operands + CZ · 130 ns  (a measurement: 15 µs) and the gate starts when all its
    predecessors on the same wires are done.  Returns (B, D) tensors: duration µs (longest path in time), depth
    (longest path in operations), idle µs = Σ_active logical qubits (duration − busy).  No gradient: the counts are
    trained by the count loss; the ESP head learns a correction on top."""
    from .devices import GATE_US
    dev = n_ops.device
    no = n_ops.detach().float().cpu().numpy()
    if gate_t is None:                     # v18: per-gate durations through the placement (G, D, 3)
        t1, tm, tcz = GATE_US
    else:
        gt = gate_t.detach().float().cpu().numpy()
        t1, tm, tcz = gt[..., 0], gt[..., 1], gt[..., 2]
    G, D = no.shape[0], no.shape[1]
    gq = circuits.gate_qubits.cpu().numpy()
    gb = circuits.batch.cpu().numpy()
    meas = circuits.x[:, GATE2IDX_MEASURE].cpu().numpy() > 0.5
    n_op = (gq >= 0).sum(1).clip(min=1)
    t_gate = np.where(meas[:, None], tm, no[..., 0] * t1 / n_op[:, None] + no[..., 2] * tcz)   # scalars or (G, D)
    o_gate = np.where(meas[:, None], 1.0, no[..., 0] / n_op[:, None] + no[..., 2])
    ei = circuits.edge_index.cpu().numpy()
    # levels per circuit (cached by name), then global
    names = getattr(circuits, "circuit_name", None)
    n_circ = int(circuits.n_qubits.view(-1).size(0))
    starts = np.concatenate([[0], np.cumsum(np.bincount(gb, minlength=n_circ))])
    lvl = np.zeros(G, dtype=np.int64)
    for b in range(n_circ):
        s, e = starts[b], starts[b + 1]
        key = names[b] if names is not None else None
        if cache is not None and key is not None and key in cache:
            lvl[s:e] = cache[key]
            continue
        sel = (ei[0] >= s) & (ei[0] < e)
        lv = _dag_levels(ei[:, sel] - s, e - s)
        lvl[s:e] = lv
        if cache is not None and key is not None:
            cache[key] = lv
    fin_t, fin_o = np.zeros((G, D)), np.zeros((G, D))
    st_t, st_o = np.zeros((G, D)), np.zeros((G, D))
    e_lvl = lvl[ei[1]]
    order = np.argsort(e_lvl, kind="stable")
    e_src, e_dst, e_lv = ei[0][order], ei[1][order], e_lvl[order]
    e_cut = np.searchsorted(e_lv, np.arange(lvl.max() + 2))
    g_order = np.argsort(lvl, kind="stable")
    g_cut = np.searchsorted(lvl[g_order], np.arange(lvl.max() + 2))
    for l in range(int(lvl.max()) + 1):
        a, b = e_cut[l], e_cut[l + 1]
        if b > a:
            np.maximum.at(st_t, e_dst[a:b], fin_t[e_src[a:b]])
            np.maximum.at(st_o, e_dst[a:b], fin_o[e_src[a:b]])
        idx = g_order[g_cut[l]: g_cut[l + 1]]
        fin_t[idx] = st_t[idx] + t_gate[idx]
        fin_o[idx] = st_o[idx] + o_gate[idx]
    dur = np.zeros((n_circ, D)); dep = np.zeros((n_circ, D))
    np.maximum.at(dur, gb, fin_t); np.maximum.at(dep, gb, fin_o)
    q_max = int(circuits.n_qubits.max())
    busy = np.zeros((n_circ * q_max, D))
    for k in range(gq.shape[1]):
        m = gq[:, k] >= 0
        np.add.at(busy, gb[m] * q_max + gq[m, k], t_gate[m])
    busy = busy.reshape(n_circ, q_max, D)
    idle_q = (dur[:, None, :] - busy).clip(min=0) * (busy > 0)                 # (B, Qm, D)
    idle = idle_q.sum(1)
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
    return {"dur_s": f(dur), "dep_s": f(dep), "idle_s": f(idle), "idle_q": f(idle_q.transpose(0, 2, 1))}

SINKHORN_KINDS = ("sinkhorn", "sinkhorn_rf", "sinkhorn_rf_rel", "sinkhorn_rf_route", "sinkhorn_rf_rel_route", "sinkhorn_rf_route_czc", "sinkhorn_rf_rel_route_czc", "sinkhorn_rf_route_usage", "sinkhorn_rf_route_czc_ep", "sinkhorn_rf_route_czc_ep3", "sinkhorn_rf_route_czc_mlps", "sinkhorn_rf_route_czc_ep3_mlps", "sinkhorn_rf_route_czc_vf2", "sinkhorn_rf_route_pgnn_usage")


def split_esp_kind(kind: str) -> tuple[str, str | bool]:
    """``<sinkhorn kind>_esp`` (v13 ESP head) / ``<sinkhorn kind>_espa`` (v14 analytic head) -> (base kind, head)."""
    for sfx, head in (("_espa", "a"), ("_espi", "i"), ("_esps", "s"), ("_espf", "f"), ("_esp", "v13")):
        if kind.endswith(sfx) and kind[: -len(sfx)] in SINKHORN_KINDS:
            return kind[: -len(sfx)], head
    return kind, False


def is_sinkhorn_kind(kind: str) -> bool:
    return split_esp_kind(kind)[0] in SINKHORN_KINDS


def build_model(kind: str, params: dict[str, Any], devices, device: torch.device) -> nn.Module:
    kind, esp = split_esp_kind(kind)
    if kind == "pooled":
        return build_v2_model(params, devices, device)
    if kind in SINKHORN_KINDS:
        rel = kind in ("sinkhorn_rf_rel", "sinkhorn_rf_rel_route", "sinkhorn_rf_rel_route_czc")
        model = SinkhornPlacementPredictor(
            params, dev_node_dim=devices.x.size(1) + (N_RELATIVE if rel else 0),
            dev_edge_dim=devices.edge_attr.size(1) * (2 if rel else 1),
            devices=devices.clone().cpu(),   # clone: PyG .cpu() moves the batch in place
            routing_flag=kind != "sinkhorn",
            routing_est="route" in kind,
            cz_constrained="_czc" in kind or kind.endswith("_usage"),
            usage_head=kind.endswith("_usage"),
            err_prior=3 if "_ep3" in kind else (1 if kind.endswith("_ep") else 0),
            mlp_scorer="_mlps" in kind, vf2_prior=kind.endswith("_vf2"), placed_gnn="_pgnn" in kind,
            esp_head=esp,
        )
        model.relative_dev = rel
        return model.to(device)
    if kind == "phys_uniform":
        return PhysicsHeadPredictor(
            params, dev_node_dim=devices.x.size(1), dev_edge_dim=devices.edge_attr.size(1),
            uniform_attention=True,
        ).to(device)
    if kind in ("xattn", "phys"):
        cls = QubitCrossAttentionPredictor if kind == "xattn" else PhysicsHeadPredictor
        return cls(
            params, dev_node_dim=devices.x.size(1), dev_edge_dim=devices.edge_attr.size(1)
        ).to(device)
    raise ValueError(f"unknown model kind {kind!r}")
