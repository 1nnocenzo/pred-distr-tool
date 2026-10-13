"""v23: approximate router for the ESP head (no learning, no compiler pass).

Greedy placement of the interaction graph, gates in program order, a 2-qubit gate on non-adjacent qubits moves its
first operand along a shortest path with SWAPs (3 CZ + 4 r each), ASAP on the physical qubits; per-gate r / CZ from
the fixed decomposition table.  Per (circuit, device): duration (µs, reference gate times), number of SWAPs, and the
idle time of every physical qubit it touches (duration − busy), so the ESP decay can use the physical qubits' own
T1 / T2.  Deterministic in (circuit, coupling map): cached by name.  Same algorithm as esp/esp_router.py.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .devices import GATE_US

T1, TM, TCZ = GATE_US
SWAP_R = 4


def chips(devices):
    ei = devices.edge_index.cpu()
    out, offset = [], 0
    import torch
    for n in torch.bincount(devices.batch.cpu()).tolist():
        sel = (ei[0] >= offset) & (ei[0] < offset + n)
        edges = tuple(sorted({(min(a, b), max(a, b)) for a, b in (ei[:, sel] - offset).t().tolist()}))
        offset += n
        adj = [[] for _ in range(n)]
        for a, c in edges:
            adj[a].append(c); adj[c].append(a)
        H, par = [], []
        for s0 in range(n):
            dist = [n + 1] * n; dist[s0] = 0; p = [-1] * n; q = deque([s0])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] > dist[u] + 1:
                        dist[v] = dist[u] + 1; p[v] = u; q.append(v)
            H.append(dist); par.append(p)
        out.append((edges, n, adj, H, par))
    return out


def _place(pairs, n_logical, H, adj):
    pdeg = [len(a) for a in adj]
    w: dict = {}
    for (a, b), c in pairs.items():
        w.setdefault(a, {})[b] = c; w.setdefault(b, {})[a] = c
    pos, free = {}, set(range(len(pdeg)))
    for q in sorted(w, key=lambda q: -sum(w[q].values())):
        placed = [(pos[o], c) for o, c in w[q].items() if o in pos]
        best = max(free, key=lambda p: pdeg[p]) if not placed else \
            min(free, key=lambda p: (sum(c * H[p][po] for po, c in placed), -pdeg[p]))
        pos[q] = best; free.discard(best)
    for q in range(n_logical):
        if q not in pos and free:
            best = max(free, key=lambda p: pdeg[p]); pos[q] = best; free.discard(best)
    return pos


def route(gates, n_logical, chip):
    """gates: [(logical qubits, r, cz, is_measure)] in program order -> (duration µs, SWAPs, idle per physical qubit)."""
    _, n, adj, H, par = chip
    pairs: dict = {}
    for qs, *_ in gates:
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                k = (min(qs[i], qs[j]), max(qs[i], qs[j])); pairs[k] = pairs.get(k, 0) + 1
    pos = _place(pairs, n_logical, H, adj)
    at = {p: q for q, p in pos.items()}
    avail = np.zeros(n); busy = np.zeros(n)
    t_swap = 3 * TCZ + SWAP_R / 2 * T1
    n_swap = 0
    for qs, r, cz, meas in gates:
        if meas:
            p = pos[qs[0]]; avail[p] += TM; busy[p] += TM; continue
        if len(qs) >= 2:
            for o in qs[1:]:
                anchor, po = pos[qs[0]], pos[o]
                if H[po][anchor] <= 1:
                    continue
                u = po
                while H[u][anchor] > 1:
                    v = par[anchor][u]
                    s = max(avail[u], avail[v]) + t_swap
                    avail[u] = avail[v] = s; busy[u] += t_swap; busy[v] += t_swap
                    qu, qv = at.get(u), at.get(v)
                    at[u], at[v] = qv, qu
                    if qu is not None:
                        pos[qu] = v
                    if qv is not None:
                        pos[qv] = u
                    n_swap += 1
                    u = v
        ps = [pos[q] for q in qs]
        t = r * T1 / len(qs) + cz * TCZ
        s = max(avail[p] for p in ps) + t
        for p in ps:
            avail[p] = s; busy[p] += t
    dur = float(avail.max())
    idle = np.where(busy > 0, dur - busy, 0.0).clip(min=0)
    return dur, n_swap, idle


def circuit_gates(g, tab_r, tab_cz, n_u, measure_idx):
    oh = g.x[:, :n_u].cpu().numpy()
    meas = g.x[:, measure_idx].cpu().numpy() > 0.5
    gq = g.gate_qubits.cpu().numpy()
    r, cz = oh @ tab_r, oh @ tab_cz
    return [([int(x) for x in gq[k] if x >= 0], float(r[k]), float(cz[k]), bool(meas[k])) for k in range(len(gq))
            if (gq[k] >= 0).any()]
