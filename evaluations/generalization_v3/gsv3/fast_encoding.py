"""Fast circuit-to-graph encoding for the v3+ predictors (same graph as the dataset).

The dataset graphs were built by ``build_qubit_dataset._process``: transpile to the
OpenQASM 3 basis at level 0, then ``encoding.create_dag`` (which removes barriers and
transpiles *again*, builds a Qiskit DAG and calls ``dag.predecessors`` /
``dag.successors`` for every gate), then a third transpile + ``circuit_to_dag`` for the
gate operands.  Everything those steps compute can be read from one pass over the
instruction list:

* nodes = the instructions (barriers removed), in circuit order (= ``dag.op_nodes()``);
* DAG edges = consecutive instructions on the same wire (qubit or clbit), one per wire,
  so two gates sharing two qubits are joined by two edges, as in the DAG;
* fan-in / fan-out = number of distinct predecessor / successor instructions;
* critical flag = on a longest path (node count), computed forward and backward.

``encode(qc)`` returns ``(x, edge_index, gate_qubits, n_qubits)`` equal to the dataset
graph (checked by ``evaluations/timing_v8/check_fast_encoding.py``).  The transpile to the
basis is skipped when the circuit is already in it (``assume_basis=True`` skips the check).
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import transpile

from . import paths

paths.ensure_imports()

from encoding import get_openqasm3_gates  # noqa: E402  (src/model/encoding.py)

BASIS = get_openqasm3_gates()
BASIS_SET = frozenset(BASIS)
UNIQUE = [*BASIS, "measure"]
GATE2IDX = {g: i for i, g in enumerate(UNIQUE)}
N_FEATURES = len(UNIQUE) + 6 + 1 + 1 + 1 + 1 + 2
MAX_OPERANDS = 3


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def to_basis(qc, assume_basis: bool = False):
    """Barriers removed; transpiled to the OpenQASM 3 basis unless already in it."""
    names = {inst.operation.name for inst in qc.data}
    if assume_basis or names <= BASIS_SET | {"measure", "barrier"}:
        return _without_barriers(qc) if "barrier" in names else qc
    return transpile(_without_barriers(qc) if "barrier" in names else qc,
                     optimization_level=0, basis_gates=BASIS)


def _without_barriers(qc):
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name != "barrier":
            out._append(inst)
    return out


def encode(qc, assume_basis: bool = False):
    qc = to_basis(qc, assume_basis)
    qidx = {q: i for i, q in enumerate(qc.qubits)}
    cidx = {c: len(qidx) + i for i, c in enumerate(qc.clbits)}
    data = [inst for inst in qc.data if inst.operation.name != "barrier"]
    n = len(data)
    off = len(UNIQUE)
    feats = np.zeros((n, N_FEATURES), dtype=np.float32)
    gq = np.full((n, MAX_OPERANDS), -1, dtype=np.int64)
    last: dict[int, int] = {}
    src: list[int] = []
    dst: list[int] = []
    preds: list[set[int]] = [set() for _ in range(n)]
    succs: list[set[int]] = [set() for _ in range(n)]
    gate_col = np.empty(n, dtype=np.int64)
    par = np.zeros((n, 3), dtype=np.float64)
    for i, inst in enumerate(data):
        op = inst.operation
        name = op.name
        g = GATE2IDX.get(name)
        if g is None:
            raise ValueError(f"Unknown gate: {name}")
        gate_col[i] = g
        raw = getattr(op, "params", [])
        for k, v in enumerate(raw[:3]):
            par[i, k] = _float(v)
        q = [qidx[b] for b in inst.qubits]
        if len(q) > MAX_OPERANDS:
            raise ValueError(f"gate {name} on {len(q)} qubits")
        gq[i, : len(q)] = q
        feats[i, off + 6] = len(q)
        feats[i, off + 7] = getattr(op, "num_ctrl_qubits", 0)
        feats[i, off + 8] = len(raw)
        for w in q + [cidx[c] for c in inst.clbits]:
            j = last.get(w)
            if j is not None:
                src.append(j)
                dst.append(i)
                preds[i].add(j)
                succs[j].add(i)
            last[w] = i
    if n:
        feats[np.arange(n), gate_col] = 1.0
        feats[:, off: off + 6: 2] = np.sin(par)
        feats[:, off + 1: off + 6: 2] = np.cos(par)
        feats[:, off + 10] = [len(p) for p in preds]
        feats[:, off + 11] = [len(s) for s in succs]
        # critical path (instructions are in topological order)
        din = [0] * n
        for i in range(n):
            if preds[i]:
                din[i] = max(din[j] for j in preds[i]) + 1
        dout = [0] * n
        for i in range(n - 1, -1, -1):
            if succs[i]:
                dout[i] = max(dout[j] for j in succs[i]) + 1
        tot = np.asarray(din) + np.asarray(dout)
        feats[:, off + 9] = tot == tot.max()
    x = torch.from_numpy(feats)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    return x, edge_index, torch.from_numpy(gq), qc.num_qubits


def to_data(qc, assume_basis: bool = False):
    """``torch_geometric`` ``Data`` ready for the v3+ models (no target)."""
    from torch_geometric.data import Data
    x, ei, gq, nq = encode(qc, assume_basis)
    return Data(x=x, edge_index=ei, gate_qubits=gq, n_qubits=nq, num_nodes=x.size(0))
