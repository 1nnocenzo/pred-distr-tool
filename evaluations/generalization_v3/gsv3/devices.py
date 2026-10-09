"""Device graphs for v3: v2's graphs (read-only), optionally with Laplacian positional encodings.

The positional encoding is the first ``k`` non-trivial eigenvectors of each
coupling map's normalised Laplacian: the standard way to let a message-passing
network tell apart, and measure distances between, nodes of a graph.  It is
computed from the coupling map alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.data import Batch

from . import paths

paths.ensure_imports()

from gsv2.devices import load_device_batch  # noqa: E402

logger = logging.getLogger(__name__)


def _laplacian_pe(n: int, edge_index: torch.Tensor, k: int) -> torch.Tensor:
    adj = torch.zeros(n, n, dtype=torch.float64)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(1)
    inv_sqrt = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
    lap = torch.eye(n, dtype=torch.float64) - inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    _, vecs = torch.linalg.eigh(lap)
    pe = vecs[:, 1 : k + 1]
    # Fixed sign convention (largest-magnitude entry positive) for reproducibility.
    sign = torch.sign(pe.gather(0, pe.abs().argmax(0, keepdim=True)))
    pe = pe * torch.where(sign == 0, torch.ones_like(sign), sign)
    if pe.size(1) < k:
        pe = torch.cat([pe, torch.zeros(n, k - pe.size(1), dtype=pe.dtype)], dim=1)
    return pe.float()


def _physical_errors(raw: dict) -> torch.Tensor:
    """``(n_qubits, 3)``: unstandardised ``-log(1-e)`` of [1q gate, readout, mean incident CZ].

    These are the calibration numbers of the device itself (the ones the ground
    truth multiplies), used by the ``phys`` head; the mean incident CZ error of a
    qubit is the average over its couplers.
    """
    nodes = torch.tensor(raw["nodes"], dtype=torch.float32)
    n = nodes.size(0)
    cz_sum, cz_cnt = torch.zeros(n), torch.zeros(n)
    for (a, b), e in zip(raw["edges"], raw["edge_feats"]):
        for q in (a, b):
            cz_sum[q] += e
            cz_cnt[q] += 1
    cz_mean = torch.where(cz_cnt > 0, cz_sum / cz_cnt.clamp(min=1), cz_sum.new_tensor(0.0))
    return torch.stack([nodes[:, 0], nodes[:, 1], cz_mean], dim=1)


def load_devices(path: Path, device_names: tuple[str, ...], lap_pe: int = 0,
                 physical_errors: bool = False) -> Batch:
    batch = load_device_batch(path, device_names)
    if lap_pe <= 0 and not physical_errors:
        return batch
    graphs = batch.to_data_list()
    raw = torch.load(path, weights_only=False)["raw"] if physical_errors else None
    for g, name in zip(graphs, device_names):
        if lap_pe > 0:
            g.x = torch.cat([g.x, _laplacian_pe(g.num_nodes, g.edge_index, lap_pe)], dim=1)
        if physical_errors:
            g.phys = _physical_errors(raw[name])
            assert g.phys.size(0) == g.num_nodes, name
    logger.info("Device graphs: node features %d (Laplacian PE %d), physical errors %s",
                graphs[0].x.size(1), lap_pe, physical_errors)
    return Batch.from_data_list(graphs)


def variant_batch(raw_variants: dict, keys: list[str], lap_pe: int, path: Path | None = None) -> Batch:
    """Device graphs of calibration variants, standardised like the training graphs.

    ``raw_variants[key]`` = ``{"nodes", "edges", "edge_feats"}`` as written by
    ``gsv2.devices._backend_graph`` (same coupling map as the original device, only the
    errors differ).  Node/edge features are standardised with the statistics of the
    three ORIGINAL devices, exactly as at training time (v8b calibration augmentation).
    """
    from torch_geometric.data import Data
    path = path or paths.V2_DEVICE_GRAPHS
    orig = torch.load(path, weights_only=False)["raw"]
    all_nodes = torch.tensor([r for v in orig.values() for r in v["nodes"]])
    all_edges = torch.tensor([e for v in orig.values() for e in v["edge_feats"]]).unsqueeze(1)
    n_mean, n_std = all_nodes.mean(0), all_nodes.std(0).clamp_min(1e-8)
    e_mean, e_std = all_edges.mean(0), all_edges.std(0).clamp_min(1e-8)
    graphs = []
    for k in keys:
        r = raw_variants[k]
        x = (torch.tensor(r["nodes"]) - n_mean) / n_std
        ea = ((torch.tensor(r["edge_feats"]).unsqueeze(1) - e_mean) / e_std).float()
        src = [a for a, b in r["edges"]] + [b for a, b in r["edges"]]
        dst = [b for a, b in r["edges"]] + [a for a, b in r["edges"]]
        g = Data(x=x.float(), edge_index=torch.tensor([src, dst], dtype=torch.long),
                 edge_attr=torch.cat([ea, ea]))
        if lap_pe > 0:
            g.x = torch.cat([g.x, _laplacian_pe(g.num_nodes, g.edge_index, lap_pe)], dim=1)
        g.phys = _physical_errors({"nodes": r["nodes"], "edges": [tuple(e) for e in r["edges"]],
                                   "edge_feats": r["edge_feats"]})
        graphs.append(g)
    return Batch.from_data_list(graphs)
