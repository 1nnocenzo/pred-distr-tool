"""Device graphs: coupling map + calibration, as PyG ``Data`` objects.

Each target device of the scheduling study (``EQE1_Top``, ``EQE1_Bottom``,
``QExa20``) is described by the same Qiskit ``BackendV2`` used to compute the
ground-truth fidelities (``compileCircuits/createDevice.py``):

* **nodes** = physical qubits, features
  ``[-log(1-e_r), -log(1-e_readout), log10 T1, log10 T2, degree]``;
* **edges** = coupling-map links (both directions), feature ``[-log(1-e_cz)]``.

Error rates enter as ``-log(1-e)`` because the expected fidelity is a product of
``(1-e)`` terms, so these are the additive per-operation contributions to
``-log F``.  Every feature is standardised jointly over the three devices; the
statistics describe the fixed device set, not the circuit data, so they carry
no information about any split.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)

NODE_FEATURES = ("neglog_1q", "neglog_readout", "log10_t1", "log10_t2", "degree")
EDGE_FEATURES = ("neglog_cz",)


def _neglog(error: float | None) -> float | None:
    if error is None:
        return None
    return -math.log(max(1.0 - float(error), 1e-12))


def _mean_fill(values: list[float | None]) -> list[float]:
    known = [v for v in values if v is not None and math.isfinite(v)]
    fill = sum(known) / len(known) if known else 0.0
    return [v if v is not None and math.isfinite(v) else fill for v in values]


def _backend_graph(backend: Any) -> tuple[list[list[float]], list[tuple[int, int]], list[float]]:
    """Raw (unnormalised) node features, undirected edges and edge features."""
    target = backend.target
    qubits = list(getattr(backend, "active_qubits", None) or range(backend.num_qubits))
    position = {q: i for i, q in enumerate(qubits)}

    edges: dict[tuple[int, int], float | None] = {}
    for qargs, props in target["cz"].items():
        a, b = qargs
        if a not in position or b not in position:
            continue
        key = (min(position[a], position[b]), max(position[a], position[b]))
        err = _neglog(props.error if props is not None else None)
        # Keep the worse direction if both are calibrated.
        if key not in edges or (err is not None and (edges[key] is None or err > edges[key])):
            edges[key] = err
    edge_list = sorted(edges)
    edge_feats = _mean_fill([edges[e] for e in edge_list])

    degree = [0] * len(qubits)
    for a, b in edge_list:
        degree[a] += 1
        degree[b] += 1

    def _qubit_error(name: str, q: int) -> float | None:
        props = target[name].get((q,)) if name in target.operation_names else None
        return _neglog(props.error) if props is not None else None

    def _log10(v: float | None) -> float | None:
        return math.log10(v) if v and v > 0 else None

    cols = [
        _mean_fill([_qubit_error("r", q) for q in qubits]),
        _mean_fill([_qubit_error("measure", q) for q in qubits]),
        _mean_fill([_log10(getattr(backend.qubit_properties[q], "t1", None)) for q in qubits]),
        _mean_fill([_log10(getattr(backend.qubit_properties[q], "t2", None)) for q in qubits]),
        [float(d) for d in degree],
    ]
    node_feats = [list(row) for row in zip(*cols)]
    return node_feats, edge_list, edge_feats


def build_device_graphs(compile_circuits_dir: Path, device_names: tuple[str, ...]) -> dict[str, Any]:
    """Instantiate the backends from ``createDevice.py`` and turn them into graphs."""
    if str(compile_circuits_dir) not in sys.path:
        sys.path.insert(0, str(compile_circuits_dir))
    import createDevice  # noqa: E402  (compileCircuits/createDevice.py)

    constructors = {
        "EQE1_Top": createDevice.EQE1TopBackend,
        "EQE1_Bottom": createDevice.EQE1BottomBackend,
        "QExa20": createDevice.QExa20Backend,
    }
    raw = {}
    for name in device_names:
        backend = constructors[name]()
        raw[name] = _backend_graph(backend)
        logger.info("%s: %d qubits, %d couplers", name, len(raw[name][0]), len(raw[name][1]))

    # Joint standardisation over all devices.
    all_nodes = torch.tensor([row for nodes, _, _ in raw.values() for row in nodes])
    all_edges = torch.tensor([v for _, _, efs in raw.values() for v in efs]).unsqueeze(1)
    n_mean, n_std = all_nodes.mean(0), all_nodes.std(0).clamp_min(1e-8)
    e_mean, e_std = all_edges.mean(0), all_edges.std(0).clamp_min(1e-8)

    graphs = {}
    for name, (nodes, edge_list, edge_feats) in raw.items():
        x = (torch.tensor(nodes) - n_mean) / n_std
        ea = (torch.tensor(edge_feats).unsqueeze(1) - e_mean) / e_std
        src = [a for a, b in edge_list] + [b for a, b in edge_list]
        dst = [b for a, b in edge_list] + [a for a, b in edge_list]
        graphs[name] = Data(
            x=x.float(),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.cat([ea, ea]).float(),
            device_name=name,
        )
    return {
        "device_names": list(device_names),
        "graphs": graphs,
        "node_features": list(NODE_FEATURES),
        "edge_features": list(EDGE_FEATURES),
        "raw": {n: {"nodes": r[0], "edges": r[1], "edge_feats": r[2]} for n, r in raw.items()},
        "compile_circuits_dir": str(compile_circuits_dir),
    }


def save_device_graphs(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Wrote %s", path)


def load_device_batch(path: Path, device_names: tuple[str, ...]) -> Batch:
    """Load the pre-built graphs as one ``Batch`` in model-output order."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing — run scripts/build_device_graphs.py first."
        )
    payload = torch.load(path, weights_only=False)
    if tuple(payload["device_names"]) != tuple(device_names):
        raise ValueError(
            f"Device order in {path} is {payload['device_names']}, expected {list(device_names)}"
        )
    return Batch.from_data_list([payload["graphs"][n] for n in device_names])
