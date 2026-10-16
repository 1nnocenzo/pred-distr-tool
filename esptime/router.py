"""Routing features computed without a compiler, from the gate graph and the coupling maps only.

They depend on the coupling map, not on the calibration, so dataset.py computes them once per circuit:
  rf       (D,)      routing-free: the interaction graph embeds in the coupling graph (rustworkx VF2, as Qiskit's
                     VF2Layout at level 2; undecided within the call limit -> 0)
  route    (D,)      log1p(greedy placement's Σ pair count · (hops − 1) / multi-qubit gates)
  prior_s  (D, N, 3) SWAPs our greedy router puts on each operand wire before each gate
  hops     (D, N)    operand distance − 1 under the greedy router's current placement
  wf       (N, 9)    per operand slot: log1p(multi-qubit gates on its wire), log1p(distinct partners), position
  prev, lvl          previous gate on each operand wire, topological level (the level-wise schedule)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import T_SWAP


def pair_counts(wires) -> dict:
    pairs: dict = {}
    for qs in wires:
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                k = (min(qs[i], qs[j]), max(qs[i], qs[j]))
                pairs[k] = pairs.get(k, 0) + 1
    return pairs


def greedy_place(pairs: dict, n_logical: int, H, adj) -> dict:
    """Most-connected logical qubit first, each on the free physical qubit closest to its placed partners."""
    pdeg = [len(a) for a in adj]
    w: dict = {}
    for (a, b), c in pairs.items():
        w.setdefault(a, {})[b] = c
        w.setdefault(b, {})[a] = c
    pos, free = {}, set(range(len(pdeg)))
    for q in sorted(w, key=lambda q: -sum(w[q].values())):
        placed = [(pos[o], c) for o, c in w[q].items() if o in pos]
        best = max(free, key=lambda p: pdeg[p]) if not placed else \
            min(free, key=lambda p: (sum(c * H[p][po] for po, c in placed), -pdeg[p]))
        pos[q] = best
        free.discard(best)
    for q in range(n_logical):
        if q not in pos:
            best = max(free, key=lambda p: pdeg[p])
            pos[q] = best
            free.discard(best)
    return pos


def path(par, s, t):
    """Shortest path s -> t from the BFS parents of t's tree."""
    p, u = [s], s
    while u != t:
        u = par[t][u]
        p.append(u)
    return p


def prior_route(wires, t, meas, n_log, chip):
    """Greedy router in program order: a gate on non-adjacent qubits moves its other operands next to the first along
    shortest paths with SWAPs.  Returns (s (N, 3) SWAPs per operand slot, hops (N,), duration µs)."""
    n, adj, H, par = chip
    N = len(wires)
    s = np.zeros((N, 3), np.float32)
    hops = np.zeros(N, np.float32)
    if n_log > n:
        return s, hops, 0.0
    pos = greedy_place(pair_counts(wires), n_log, H, adj)
    at = {p: q for q, p in pos.items()}
    avail = [0.0] * n
    pending = defaultdict(int)
    for gi, qs in enumerate(wires):
        if not qs:
            continue
        if len(qs) >= 2:
            hops[gi] = max(H[pos[a]][pos[b]] for a in qs for b in qs if a != b) - 1
            anchor = pos[qs[0]]
            for o in qs[1:]:
                po = pos[o]
                if H[po][anchor] <= 1:
                    continue
                pth = path(par, po, anchor)
                for u, v in zip(pth[:-2], pth[1:-1]):
                    st = max(avail[u], avail[v]) + T_SWAP
                    avail[u] = avail[v] = st
                    qu, qv = at.get(u), at.get(v)
                    for w in (qu, qv):
                        if w is not None:
                            pending[w] += 1
                    at[u], at[v] = qv, qu
                    if qu is not None:
                        pos[qu] = v
                    if qv is not None:
                        pos[qv] = u
                anchor = pos[qs[0]]
        if len(qs) >= 2 or meas[gi]:
            for k, w in enumerate(qs):
                if pending[w]:
                    s[gi, k] += pending[w]
                    pending[w] = 0
        ps = [pos[q] for q in qs]
        st = max(avail[p] for p in ps) + t[gi]
        for p in ps:
            avail[p] = st
    return s, hops, max(avail)


def route_estimate(pairs: dict, n_multi: int, n_log: int, chip) -> float:
    n, adj, H, _ = chip
    if not pairs or n_log > n:
        return 0.0
    pos = greedy_place(pairs, n_log, H, adj)
    cost = sum(c * max(H[pos[a]][pos[b]] - 1, 0) for (a, b), c in pairs.items())
    return float(np.log1p(cost / max(n_multi, 1)))


def routing_free(pairs: dict, chip, call_limit: int = 20_000) -> bool:
    import rustworkx as rx
    n, adj, _, _ = chip
    if not pairs:
        return True
    used = sorted({q for p in pairs for q in p})
    if len(used) > n:
        return False
    c = rx.PyGraph()
    c.add_nodes_from(range(n))
    c.add_edges_from_no_data([(a, b) for a in range(n) for b in adj[a] if a < b])
    idx = {q: i for i, q in enumerate(used)}
    g = rx.PyGraph()
    g.add_nodes_from(range(len(used)))
    g.add_edges_from_no_data([(idx[a], idx[b]) for a, b in pairs])
    return bool(rx.is_subgraph_isomorphic(c, g, id_order=False, induced=False, call_limit=call_limit))


def wire_feats(gq: np.ndarray, n_q: int) -> np.ndarray:
    N = len(gq)
    multi = (gq >= 0).sum(1) >= 2
    cnt = np.zeros(n_q)
    partners = [set() for _ in range(n_q)]
    for i in np.nonzero(multi)[0]:
        qs = [q for q in gq[i] if q >= 0]
        for q in qs:
            cnt[q] += 1
            partners[q].update(x for x in qs if x != q)
    seen = np.zeros(n_q)
    out = np.zeros((N, 9), np.float32)
    for i in range(N):
        for k in range(3):
            q = gq[i, k]
            if q < 0:
                continue
            out[i, 3 * k] = np.log1p(cnt[q])
            out[i, 3 * k + 1] = np.log1p(len(partners[q]))
            out[i, 3 * k + 2] = seen[q] / max(cnt[q], 1)
        if multi[i]:
            for q in gq[i]:
                if q >= 0:
                    seen[q] += 1
    return out


def wire_prev(gq: np.ndarray):
    """(N, 3) previous gate on each operand wire (-1: none), (N,) level = longest chain of wire predecessors."""
    N = len(gq)
    prev = np.full((N, 3), -1, np.int64)
    lvl = np.zeros(N, np.int64)
    last = {}
    for i in range(N):
        best = -1
        for k in range(gq.shape[1]):
            q = int(gq[i, k])
            if q < 0:
                continue
            p = last.get(q, -1)
            prev[i, k] = p
            if p >= 0:
                best = max(best, int(lvl[p]))
            last[q] = i
        lvl[i] = best + 1
    return prev, lvl


def features(gq: np.ndarray, n_q: int, t: np.ndarray, meas: np.ndarray, chips: list) -> dict:
    """Every routing input of one circuit (see module doc), as numpy arrays."""
    wires = [[int(x) for x in row if x >= 0] for row in gq]
    pairs = pair_counts(wires)
    n_multi = sum(len(w) >= 2 for w in wires)
    ps, hp, rf, re_ = [], [], [], []
    for c in chips:
        s_, h_, _ = prior_route(wires, t, meas, n_q, c)
        ps.append(s_); hp.append(h_)
        rf.append(routing_free(pairs, c)); re_.append(route_estimate(pairs, n_multi, n_q, c))
    prev, lvl = wire_prev(gq)
    return {"prior_s": np.stack(ps), "hops": np.stack(hp), "rf": np.array(rf, np.float32),
            "route": np.array(re_, np.float32), "wf": wire_feats(gq, n_q), "prev": prev, "lvl": lvl}
