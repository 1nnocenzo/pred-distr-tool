"""Device-aware log-fidelity predictor.

::

    circuit DAG ──GraphConvolutionSage──► h_c ─┐                    (tuned, from gnn.py)
                                               ├─► MLP ─► log F(c,d)
    device d   ──DeviceEncoder (GINE)───► h_d ─┘
               (coupling map + calibration)

The circuit encoder is the tuned ``GraphConvolutionSage`` of ``src/model/gnn.py``
with the hyper-parameters of ``best_params.json``; the MLP head keeps the tuned
widths.  The device encoder and the fusion width are new and fixed
(:data:`DEVICE_ENCODER_DEFAULTS`): they are not tuned on any split.

The head output is ``log F`` directly (linear, unbounded): the fidelity of a
compiled circuit is a product of per-operation terms, so its log is an
approximately additive sum the network can model.  The output is capped at 0
(``F <= 1``) only at inference, in :func:`training.predict`; a saturating
output such as ``logsigmoid`` was avoided because the encoder's ``sum`` readout
grows with circuit size and would push it into its zero-gradient region.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool

from .paths import ensure_v1_imports

ensure_v1_imports()

from genstudy import paths as v1_paths  # noqa: E402

v1_paths.ensure_model_imports()

from encoding import get_gnn_input_features  # noqa: E402  (src/model/encoding.py)
from gnn import GraphConvolutionSage  # noqa: E402  (src/model/gnn.py)

logger = logging.getLogger(__name__)

#: Fixed (untuned) sizes of the new components.
DEVICE_ENCODER_DEFAULTS: dict[str, Any] = {
    "device_hidden": 64,
    "device_layers": 3,
    "fusion_dim": 128,
}


def load_hparams(params_path: Path | str | None = None) -> dict[str, Any]:
    """Tuned hyper-parameters of the circuit encoder and head, plus the v2 defaults."""
    params_path = Path(params_path or v1_paths.MODEL_DIR / "best_params.json")
    with open(params_path) as handle:
        params = json.load(handle)
    merged = {**DEVICE_ENCODER_DEFAULTS, **params}
    logger.info("Hyper-parameters (%s + v2 defaults): %s", params_path, merged)
    return merged


class DeviceEncoder(nn.Module):
    """GINE stack over a device coupling graph with calibration features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.inp = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden)),
                edge_dim=edge_dim,
            )
            for _ in range(layers)
        )
        self.out_dim = hidden * 3  # mean | max | sum readout

    def forward(self, graphs) -> torch.Tensor:
        x = F.leaky_relu(self.inp(graphs.x))
        for conv in self.convs:
            x = x + F.leaky_relu(conv(x, graphs.edge_index, graphs.edge_attr))
        b = graphs.batch
        return torch.cat(
            [global_mean_pool(x, b), global_max_pool(x, b), global_add_pool(x, b)], dim=-1
        )


class DeviceAwarePredictor(nn.Module):
    """Scores every (circuit, device) pair; returns uncapped ``log F``, shape ``(B, n_devices)``."""

    def __init__(self, params: dict[str, Any], node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.circuit_encoder = GraphConvolutionSage(
            in_feats=get_gnn_input_features(),
            hidden_dim=params["hidden_dim"],
            num_conv_wo_resnet=params["num_conv_wo_resnet"],
            num_resnet_layers=params["num_resnet_layers"],
            dropout_p=params["dropout"],
            dropedge_p=params.get("dropedge_p", 0),
            bidirectional=params["bidirectional"],
            combine=params.get("combine", "sum"),
            use_sag_pool=params.get("sag_pool", False),
            sag_ratio=params.get("sag_ratio", 0.9),
            readout=params.get("readout", "meanmaxsum"),
            conv_activation=F.leaky_relu,
        )
        self.device_encoder = DeviceEncoder(
            node_dim, edge_dim, params["device_hidden"], params["device_layers"]
        )
        d = params["fusion_dim"]
        self.proj_c = nn.Linear(self.circuit_encoder.graph_emb_dim, d)
        self.proj_d = nn.Linear(self.device_encoder.out_dim, d)

        layers: list[nn.Module] = []
        last = 3 * d
        for width in params["mlp"]:
            layers += [nn.Linear(last, width), nn.LeakyReLU()]
            last = width
        layers.append(nn.Linear(last, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, circuits, devices) -> torch.Tensor:
        hc = self.proj_c(self.circuit_encoder(circuits))          # (B, d)
        hd = self.proj_d(self.device_encoder(devices))            # (D, d)
        n_c, n_d = hc.size(0), hd.size(0)
        hc_e = hc.unsqueeze(1).expand(n_c, n_d, -1)
        hd_e = hd.unsqueeze(0).expand(n_c, n_d, -1)
        z = torch.cat([hc_e, hd_e, hc_e * hd_e], dim=-1)          # (B, D, 3d)
        return self.head(z).squeeze(-1)                           # (B, D) = log F


def build_model(params: dict[str, Any], devices, device: torch.device) -> DeviceAwarePredictor:
    return DeviceAwarePredictor(
        params, node_dim=devices.x.size(1), edge_dim=devices.edge_attr.size(1)
    ).to(device)
