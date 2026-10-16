"""Circuit -> gate DAG (PyG ``Data``), one pass over the instructions.

Nodes = instructions without barriers, in circuit order, after a level-0 transpile to the OpenQASM 3 basis.
Edges = consecutive instructions on the same wire (qubit or clbit), one per wire.
Features (40): gate one-hot (28 = basis + measure), sin / cos of 3 parameters, arity, control qubits, number of
parameters, critical-path flag, fan-in, fan-out.  ``gate_qubits`` (N, 3) = logical operands, -1 padded.
"""
from __future__ import annotations

import numpy as np
import torch

BASIS = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "p", "rx", "ry", "rz", "u", "cx", "cy", "cz", "ch",
         "cp", "crx", "cry", "crz", "cu", "swap", "ccx", "cswap", "if_else"]
UNIQUE = [*BASIS, "measure"]
GATE2IDX = {g: i for i, g in enumerate(UNIQUE)}
MEASURE = GATE2IDX["measure"]
N_FEATURES = len(UNIQUE) + 6 + 6
MAX_OPERANDS = 3


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _without_barriers(qc):
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name != "barrier":
            out._append(inst)
    return out


def to_basis(qc):
    """Barriers removed; transpiled to the OpenQASM 3 basis unless already in it.  This is also the circuit the
    labels compile, so the model sees exactly what is compiled."""
    from qiskit import transpile
    names = {inst.operation.name for inst in qc.data}
    qc = _without_barriers(qc) if "barrier" in names else qc
    if names <= set(BASIS) | {"measure", "barrier"}:
        return qc
    return transpile(qc, optimization_level=0, basis_gates=BASIS)


def encode(qc):
    """``(x (N, 40), edge_index (2, E), gate_qubits (N, 3), n_qubits)`` of a basis circuit (``to_basis``)."""
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
        g = GATE2IDX.get(op.name)
        if g is None:
            raise ValueError(f"unknown gate: {op.name}")
        gate_col[i] = g
        raw = getattr(op, "params", [])
        for k, v in enumerate(raw[:3]):
            par[i, k] = _float(v)
        q = [qidx[b] for b in inst.qubits]
        if len(q) > MAX_OPERANDS:
            raise ValueError(f"gate {op.name} on {len(q)} qubits")
        gq[i, : len(q)] = q
        feats[i, off + 6] = len(q)
        feats[i, off + 7] = getattr(op, "num_ctrl_qubits", 0)
        feats[i, off + 8] = len(raw)
        for w in q + [cidx[c] for c in inst.clbits]:
            j = last.get(w)
            if j is not None:
                src.append(j); dst.append(i)
                preds[i].add(j); succs[j].add(i)
            last[w] = i
    if n:
        feats[np.arange(n), gate_col] = 1.0
        feats[:, off: off + 6: 2] = np.sin(par)
        feats[:, off + 1: off + 6: 2] = np.cos(par)
        feats[:, off + 10] = [len(p) for p in preds]
        feats[:, off + 11] = [len(s) for s in succs]
        din = [0] * n                              # critical path: instructions are in topological order
        for i in range(n):
            if preds[i]:
                din[i] = max(din[j] for j in preds[i]) + 1
        dout = [0] * n
        for i in range(n - 1, -1, -1):
            if succs[i]:
                dout[i] = max(dout[j] for j in succs[i]) + 1
        tot = np.asarray(din) + np.asarray(dout)
        feats[:, off + 9] = tot == tot.max()
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(feats), ei, torch.from_numpy(gq), qc.num_qubits


def gate_times(x: torch.Tensor, gate_qubits: torch.Tensor, table: dict) -> np.ndarray:
    """(N,) µs each gate occupies each of its operand wires on the logical schedule: the decomposition table's
    r count (split over the operands) and cz count; a measurement 15 µs."""
    from .config import TCZ, TM, T1
    tr = np.array([table.get(u, {"r": 0})["r"] for u in UNIQUE], float)
    tc = np.array([table.get(u, {"cz": 0})["cz"] for u in UNIQUE], float)
    oh = x[:, : len(UNIQUE)].numpy()
    meas = x[:, MEASURE].numpy() > 0.5
    n_op = np.clip((gate_qubits.numpy() >= 0).sum(1), 1, None)
    return np.where(meas, TM, (oh @ tr) * T1 / n_op + (oh @ tc) * TCZ).astype(np.float32)
