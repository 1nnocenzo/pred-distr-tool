"""Devices: Qiskit backends (createDevice.py), calibration variants, and their graphs for the model.

Raw graph of a backend (``raw_graph``, JSON-serialisable, written by labels.py so training never needs Qiskit):
  nodes       per active qubit [-log(1-e_r), -log(1-e_readout), log10 T1, log10 T2, degree]
  edges       couplers as sorted node pairs;  edge_feats  -log(1-e_cz) (worse direction)
Model batch (``device_batch``): node / edge features standardised with the statistics of the three ORIGINAL devices
(so a variant differs from its device only where its calibration differs) + 8 Laplacian eigenvectors; ``phys``
(P, 3) unstandardised errors [r, readout, mean incident cz]; ``coh`` (P,) 1/min(T1, T2) in 1/µs.

Calibration variants (``aug<j>``, from the earlier v8b/v8d study): only r / cz / measure errors change, never the
coupling map, T1/T2 or the durations.  j < 8: jitter σ 0.25 (0-2), shuffle (3-4), 2 bad qubits ×5 (5-7); j ≥ 8 cycles
over j mod 10: 0-3 jitter σ U(0.15, 0.4), 4 shuffle, 5-6 1-3 bad qubits ×U(3, 8), 7-8 per-kind scale U(0.6, 1.6) +
jitter 0.15, 9 jitter 0.25 + bad qubits.  Seeds 10 000 + 1 000·device + j.
"""
from __future__ import annotations

import copy
import math
from collections import deque

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from . import config  # noqa: F401  (puts createDevice on sys.path)
from .config import DEVICES

GATES = ("r", "cz", "measure")
MAX_ERROR = 0.5
LAP_PE = 8


# ------------------------------------------------------------------------------------------------ backends
def backends(calib: str = "orig") -> dict:
    """{device: BackendV2} for calibration ``orig`` or ``aug<j>``."""
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    base = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    for be in base.values():               # MQT's ESP reads T1/T2 from the target
        t = be.target
        if t.qubit_properties is None:
            t.qubit_properties = list(be.qubit_properties)
    if calib == "orig":
        return base
    j = int(calib.removeprefix("aug"))
    return {d: make_aug(be, j, i) for i, (d, be) in enumerate(base.items())}


def active_qubits(be) -> list[int]:
    return list(getattr(be, "active_qubits", None) or range(be.num_qubits))


def _set_error(t, gate, qargs, error):
    props = t[gate][qargs]
    t.update_instruction_properties(gate, qargs, type(props)(duration=props.duration,
                                                             error=float(min(max(error, 0.0), MAX_ERROR))))


def _errors(t, gate):
    return [q for q, p in t[gate].items() if p is not None and p.error is not None]


def _jitter(t, rng, sigma, scale=None):
    for gate in GATES:
        qargs = _errors(t, gate)
        e = np.array([t[gate][q].error for q in qargs], float)
        for q, v in zip(qargs, e * np.exp(rng.normal(0.0, sigma, e.shape)) * (scale[gate] if scale else 1.0)):
            _set_error(t, gate, q, v)


def _shuffle(t, rng):
    for gate in GATES:
        qargs = _errors(t, gate)
        for q, v in zip(qargs, rng.permutation([t[gate][q].error for q in qargs])):
            _set_error(t, gate, q, v)


def _bad(t, rng, n_qubits, factor, n_edges):
    qubits = sorted({q[0] for q, p in t["r"].items() if p is not None})
    bad = set(rng.choice(qubits, size=n_qubits, replace=False).tolist())
    couplers = _errors(t, "cz")
    bad_edges = {tuple(couplers[i]) for i in rng.choice(len(couplers), size=n_edges, replace=False)}
    for gate in ("r", "measure"):
        for q in _errors(t, gate):
            if q[0] in bad:
                _set_error(t, gate, q, t[gate][q].error * factor)
    for q in couplers:
        if q[0] in bad or q[1] in bad or tuple(q) in bad_edges:
            _set_error(t, "cz", q, t["cz"][q].error * factor)


def make_aug(backend, j: int, dev_index: int):
    be = copy.deepcopy(backend)
    t = be.target
    rng = np.random.default_rng(10_000 + 1_000 * dev_index + j)
    if j < 8:
        if j <= 2:
            _jitter(t, rng, 0.25)
        elif j <= 4:
            _shuffle(t, rng)
        else:
            _bad(t, rng, 2, 5.0, 1)
        return be
    kind = j % 10
    if kind <= 3:
        _jitter(t, rng, rng.uniform(0.15, 0.4))
    elif kind == 4:
        _shuffle(t, rng)
    elif kind <= 6:
        _bad(t, rng, int(rng.integers(1, 4)), rng.uniform(3, 8), int(rng.integers(0, 3)))
    elif kind <= 8:
        _jitter(t, rng, 0.15, {g: rng.uniform(0.6, 1.6) for g in GATES})
    else:
        _jitter(t, rng, 0.25)
        _bad(t, rng, int(rng.integers(1, 3)), rng.uniform(3, 6), 1)
    return be


# ----------------------------------------------------------------------------------------------- raw graphs
def _neglog(e):
    return None if e is None else -math.log(max(1.0 - float(e), 1e-12))


def _mean_fill(vals):
    known = [v for v in vals if v is not None and math.isfinite(v)]
    fill = sum(known) / len(known) if known else 0.0
    return [v if v is not None and math.isfinite(v) else fill for v in vals]


def raw_graph(be) -> dict:
    t = be.target
    qubits = active_qubits(be)
    pos = {q: i for i, q in enumerate(qubits)}
    edges: dict = {}
    for (a, b), props in t["cz"].items():
        if a not in pos or b not in pos:
            continue
        k = (min(pos[a], pos[b]), max(pos[a], pos[b]))
        e = _neglog(props.error if props is not None else None)
        if k not in edges or (e is not None and (edges[k] is None or e > edges[k])):
            edges[k] = e
    elist = sorted(edges)
    deg = [0] * len(qubits)
    for a, b in elist:
        deg[a] += 1; deg[b] += 1

    def q_err(name, q):
        p = t[name].get((q,)) if name in t.operation_names else None
        return _neglog(p.error) if p is not None else None

    def log10(v):
        return math.log10(v) if v and v > 0 else None
    qp = be.qubit_properties
    cols = [_mean_fill([q_err("r", q) for q in qubits]), _mean_fill([q_err("measure", q) for q in qubits]),
            _mean_fill([log10(getattr(qp[q], "t1", None)) for q in qubits]),
            _mean_fill([log10(getattr(qp[q], "t2", None)) for q in qubits]), [float(d) for d in deg]]
    return {"nodes": [list(r) for r in zip(*cols)], "edges": [list(e) for e in elist],
            "edge_feats": _mean_fill([edges[e] for e in elist])}


# ------------------------------------------------------------------------------------------------ model batch
def _laplacian_pe(n: int, edge_index: torch.Tensor, k: int = LAP_PE) -> torch.Tensor:
    adj = torch.zeros(n, n, dtype=torch.float64)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(1)
    inv = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
    _, vecs = torch.linalg.eigh(torch.eye(n, dtype=torch.float64) - inv[:, None] * adj * inv[None, :])
    pe = vecs[:, 1: k + 1]
    sign = torch.sign(pe.gather(0, pe.abs().argmax(0, keepdim=True)))      # fixed sign convention
    pe = pe * torch.where(sign == 0, torch.ones_like(sign), sign)
    return pe.float()


def device_batch(raws: dict, orig: dict) -> Batch:
    """``raws`` {device: raw graph} of one calibration (DEVICES order) -> PyG Batch; ``orig`` = the raw graphs of
    the original calibration (standardisation statistics)."""
    all_n = torch.tensor([r for v in orig.values() for r in v["nodes"]])
    all_e = torch.tensor([e for v in orig.values() for e in v["edge_feats"]]).unsqueeze(1)
    n_mu, n_sd = all_n.mean(0), all_n.std(0).clamp_min(1e-8)
    e_mu, e_sd = all_e.mean(0), all_e.std(0).clamp_min(1e-8)
    graphs = []
    for d in DEVICES:
        r = raws[d]
        nodes = torch.tensor(r["nodes"], dtype=torch.float64)
        ea = ((torch.tensor(r["edge_feats"]).unsqueeze(1) - e_mu) / e_sd).float()
        src = [a for a, b in r["edges"]] + [b for a, b in r["edges"]]
        dst = [b for a, b in r["edges"]] + [a for a, b in r["edges"]]
        ei = torch.tensor([src, dst], dtype=torch.long)
        x = ((nodes.float() - n_mu) / n_sd).float()
        n = x.size(0)
        cz_sum, cz_cnt = torch.zeros(n), torch.zeros(n)
        for (a, b), e in zip(r["edges"], r["edge_feats"]):
            for q in (a, b):
                cz_sum[q] += e; cz_cnt[q] += 1
        phys = torch.stack([nodes[:, 0].float(), nodes[:, 1].float(), cz_sum / cz_cnt.clamp(min=1)], 1)
        coh = (10.0 ** (-torch.minimum(nodes[:, 2], nodes[:, 3])) * 1e-6).float()     # 1/µs
        graphs.append(Data(x=torch.cat([x, _laplacian_pe(n, ei)], 1), edge_index=ei, edge_attr=torch.cat([ea, ea]),
                           phys=phys, coh=coh))
    return Batch.from_data_list(graphs)


def chips(raws: dict) -> list:
    """Per device (n, adjacency, hop matrix H, BFS parents par[t][u] = next hop from u towards t)."""
    out = []
    for d in DEVICES:
        n = len(raws[d]["nodes"])
        adj = [[] for _ in range(n)]
        for a, c in raws[d]["edges"]:
            adj[a].append(c); adj[c].append(a)
        H, par = [], []
        for s in range(n):
            dist = [n + 1] * n; dist[s] = 0; p = [-1] * n; q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] > dist[u] + 1:
                        dist[v] = dist[u] + 1; p[v] = u; q.append(v)
            H.append(dist); par.append(p)
        out.append((n, adj, H, par))
    return out
