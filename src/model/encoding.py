"""Circuit-to-graph encoding for the GNN fidelity predictor.

Extracted verbatim from ``mqt.predictor.ml.helper`` (predictor-gnn) so the
evaluation pipeline does not depend on the full mqt-predictor package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from qiskit import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def get_openqasm3_gates() -> list[str]:
    """Returns a list of all quantum gates within the openQASM 3.0 standard header."""
    # according to https://openqasm.com/language/standard_library.html#standard-library
    # Snapshot from OpenQASM 3.0 specification (version 3.0)
    # Verify against latest spec when Qiskit or OpenQASM updates
    return [
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "p",
        "rx",
        "ry",
        "rz",
        "u",
        "cx",
        "cy",
        "cz",
        "ch",
        "cp",
        "crx",
        "cry",
        "crz",
        "cu",
        "swap",
        "ccx",
        "cswap",
        "if_else",
    ]


def create_dag(qc: QuantumCircuit) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Creates and returns the feature-annotated DAG of the quantum circuit.

    This is particularly useful for GNN-based models. It is needed to convert the quantum circuit
    into a graph representation. Then, each node is annotated with features such as one-hot gate encoding,
    sin/cos of parameters, arity, number of controls, number of parameters, critical path flag, fan-in and fan-out.

    Arguments:
        qc: The quantum circuit to be converted to a DAG.

    Returns:
        node_vector: features per node = [one-hot gate, sin/cos params, arity, controls,
        num_params, critical_flag, fan_in, fan_out]
        edge_index: 2 for E tensor of edges (src, dst)
        number_of_gates: number of nodes in the DAG.
    """
    # 0) cleanup & DAG
    pm = PassManager(RemoveBarriers())
    qc = pm.run(qc)
    try:
        qc = transpile(qc, optimization_level=0, basis_gates=get_openqasm3_gates())
    except Exception as e:
        msg = f"Transpilation failed: {e}"
        raise ValueError(msg) from e
    dag = circuit_to_dag(qc)

    unique_gates = [*get_openqasm3_gates(), "measure"]
    gate2idx = {g: i for i, g in enumerate(unique_gates)}
    number_gates = len(unique_gates)

    def _safe_float(val: object, default: float = 0.0) -> float:
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    # --- parameters sin/cos (max 3 param) ---
    def param_vector(node: DAGOpNode, dim: int = 3) -> list[float]:
        """Return [sin(p1), cos(p1), sin(p2), cos(p2), sin(p3), cos(p3)]."""
        raw_params = getattr(node.op, "params", [])
        params = [_safe_float(v) for v in raw_params[:dim]]
        params += [0.0] * (dim - len(params))

        out: list[float] = []
        for p in params:
            out.extend([np.sin(p), np.cos(p)])
        return out  # len = 2*dim

    nodes = list(dag.op_nodes())
    number_nodes = len(nodes)

    # prealloc
    onehots = torch.zeros((number_nodes, number_gates), dtype=torch.float32)
    num_params = torch.zeros((number_nodes, 1), dtype=torch.float32)
    params = torch.zeros((number_nodes, 6), dtype=torch.float32)
    arity = torch.zeros((number_nodes, 1), dtype=torch.float32)
    controls = torch.zeros((number_nodes, 1), dtype=torch.float32)
    fan_in = torch.zeros((number_nodes, 1), dtype=torch.float32)
    fan_out = torch.zeros((number_nodes, 1), dtype=torch.float32)

    for i, node in enumerate(nodes):
        name = node.op.name
        if name not in unique_gates:
            msg = f"Unknown gate: {name}"
            raise ValueError(msg)
        onehots[i, gate2idx[name]] = 1.0
        params[i] = torch.tensor(param_vector(node), dtype=torch.float32)
        arity[i] = float(len(node.qargs))
        controls[i] = float(getattr(node.op, "num_ctrl_qubits", 0))
        num_params[i] = float(len(getattr(node.op, "params", [])))
        preds = [p for p in dag.predecessors(node) if isinstance(p, DAGOpNode)]
        succs = [s for s in dag.successors(node) if isinstance(s, DAGOpNode)]
        fan_in[i] = len(preds)
        fan_out[i] = len(succs)

    # edges DAG
    idx_map = {node: i for i, node in enumerate(nodes)}
    edges: list[list[int]] = []
    for src, dst, _ in dag.edges():
        if src in idx_map and dst in idx_map:
            edges.append([idx_map[src], idx_map[dst]])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- critical path detection ---
    topo_nodes = list(dag.topological_op_nodes())
    if not topo_nodes:
        # No operation nodes: return node features with zero critical flags
        critical_flag = torch.zeros((number_nodes, 1), dtype=torch.float32)
        node_vector = torch.cat([onehots, params, arity, controls, num_params, critical_flag, fan_in, fan_out], dim=1)
        return node_vector, edge_index, number_nodes

    dist_in = dict.fromkeys(topo_nodes, 0)
    for node in topo_nodes:
        preds = [p for p in dag.predecessors(node) if isinstance(p, DAGOpNode)]
        if preds:
            dist_in[node] = max(dist_in.get(p, 0) + 1 for p in preds)

    dist_out = dict.fromkeys(topo_nodes, 0)
    for node in reversed(topo_nodes):
        succs = [s for s in dag.successors(node) if isinstance(s, DAGOpNode)]
        if succs:
            dist_out[node] = max(dist_out.get(s, 0) + 1 for s in succs)

    critical_len = max(dist_in.get(n, 0) + dist_out.get(n, 0) for n in topo_nodes)

    critical_flag = torch.zeros((number_nodes, 1), dtype=torch.float32)
    for i, node in enumerate(nodes):
        # set critical flag to 1 if on critical path
        if dist_in.get(node, 0) + dist_out.get(node, 0) == critical_len:
            critical_flag[i] = 1.0

    # final concat of features
    node_vector = torch.cat([onehots, params, arity, controls, num_params, critical_flag, fan_in, fan_out], dim=1)

    return node_vector, edge_index, number_nodes


def get_gnn_input_features() -> int:
    """Calculate GNN input feature dimension.

    Components:
    - len(get_openqasm3_gates()) + 1: one-hot gate encoding (including 'measure')
    - 6: sin/cos for up to 3 parameters
    - 1: arity
    - 1: controls
    - 1: num_params
    - 1: critical_flag
    - 2: fan_in, fan_out
    """
    return len(get_openqasm3_gates()) + 1 + 6 + 1 + 1 + 1 + 1 + 2
