#!/usr/bin/env python3
"""Prototype of a cheap routing-cost feature: is it informative, and what does it cost?

For each circuit and device topology: the interaction graph (pairs of logical qubits sharing a
gate, with multiplicity) is placed greedily on the coupling map (qubits by decreasing weighted
degree; each goes to the free physical qubit closest to its already-placed partners), then
    est = Σ_pairs  count(pair) · max(hops − 1, 0)      (hop distance on the chip beyond adjacency)
normalised by the number of logical two-qubit gates.  Compared with the true routing overhead
of the compiled circuit (cz count / logical 2q count), per family; timed per circuit.
    python routing_estimate.py
"""
import collections
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
OFF = 28


def hop_matrix(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    H = np.full((n, n), n + 1, dtype=np.int32)
    for s in range(n):
        H[s, s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if H[s, v] > H[s, u] + 1:
                        H[s, v] = H[s, u] + 1
                        nxt.append(v)
            frontier = nxt
    return H, np.array([len(a) for a in adj])


def estimate(pairs, H, pdeg):
    """pairs: {(a, b): count}; returns Σ count·max(hops−1, 0) under a greedy placement."""
    w = collections.defaultdict(dict)
    for (a, b), c in pairs.items():
        w[a][b] = c; w[b][a] = c
    order = sorted(w, key=lambda q: -sum(w[q].values()))
    pos, free = {}, set(range(H.shape[0]))
    for q in order:
        placed = [(pos[o], c) for o, c in w[q].items() if o in pos]
        if not placed:
            best = max(free, key=lambda p: pdeg[p])
        else:
            best = min(free, key=lambda p: (sum(c * H[p, po] for po, c in placed), -pdeg[p]))
        pos[q] = best; free.discard(best)
    return sum(c * max(int(H[pos[a], pos[b]]) - 1, 0) for (a, b), c in pairs.items())


def main():
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    chips = {d: hop_matrix(len(raw[d]["nodes"]), [tuple(e) for e in raw[d]["edges"]]) for d in DEVICES}
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    counts = torch.load(EVAL / "compile_check/results/layouts_all/compiler_counts.pt", weights_only=False)
    rows, times = [], []
    for d in data:
        n = d.circuit_name
        if n not in counts:
            continue
        logical2q = int((d.x[:, OFF + 6] == 2).sum())
        if logical2q == 0:
            continue
        t0 = time.perf_counter()
        pairs = collections.Counter()
        for row in d.gate_qubits.tolist():
            q = [x for x in row if x >= 0]
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    pairs[(min(q[i], q[j]), max(q[i], q[j]))] += 1
        est = [estimate(pairs, *chips[dev]) / logical2q for dev in DEVICES]
        times.append(time.perf_counter() - t0)
        for i in range(3):
            rows.append((n.split("/")[0], est[i], float(counts[n][i, 2]) / logical2q,
                         int(re.search(r"_q(\d+)_", n).group(1))))
    fam = np.array([r[0] for r in rows]); est = np.array([r[1] for r in rows]); ratio = np.array([r[2] for r in rows])
    from scipy.stats import spearmanr
    t = np.array(times) * 1e3
    print(f"{len(times)} circuits × 3 topologies; time per circuit (all 3 topologies): median {np.median(t):.2f} ms, "
          f"p99 {np.percentile(t, 99):.2f} ms, max {t.max():.1f} ms")
    print(f"all: Spearman(est, true cz / logical 2q) = {spearmanr(est, ratio).correlation:+.2f}")
    for f in ("qaoa", "randomcircuit", "qft", "ae", "vqe_two_local", "vqe_real_amp", "qnn", "iqpe"):
        s = fam == f
        if s.sum() > 20:
            print(f"   {f:16} n={s.sum():5d}  Spearman {spearmanr(est[s], ratio[s]).correlation:+.2f}"
                  f"  mean est {est[s].mean():.2f}  mean true ratio {ratio[s].mean():.2f}")


if __name__ == "__main__":
    main()
