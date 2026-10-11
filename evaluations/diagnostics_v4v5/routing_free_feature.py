#!/usr/bin/env python3
"""Can we tell, before compiling, that a circuit needs no routing on a device?

For every dataset circuit and device:
  truth    : routing-free if compiled CZ count == logical 2-qubit gate count (compiler
             counts from compile_check; 3-qubit gates count as their 3 pairs ... ignored:
             circuits with gates on 3 qubits are reported separately);
  exact    : the interaction graph (pairs of logical qubits sharing a gate) is a
             subgraph (monomorphism) of the device coupling graph — rustworkx VF2,
             what Qiskit's VF2Layout looks for at level 2 — with a call limit;
  proxies  : max degree <= 2 / acyclic (forest) / path, of the interaction graph.
Reports precision / recall of each predictor of "routing-free", its time per circuit,
and how many circuits of each family are routing-free.
    python routing_free_feature.py
"""
import collections
import sys
import time
from pathlib import Path

import numpy as np
import rustworkx as rx
import torch

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
CALL_LIMIT = 20_000


def interaction_graph(d):
    gq = d.gate_qubits
    g = rx.PyGraph()
    g.add_nodes_from(range(int(d.n_qubits)))
    pairs = set()
    for row in gq.tolist():
        q = [x for x in row if x >= 0]
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                pairs.add((min(q[i], q[j]), max(q[i], q[j])))
    g.add_edges_from_no_data(sorted(pairs))
    used = [v for v in g.node_indices() if g.degree(v) > 0]
    return g.subgraph(used), any(sum(x >= 0 for x in row) == 3 for row in gq.tolist())


def main():
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    counts = torch.load(EVAL / "compile_check/results/layouts_all/compiler_counts.pt", weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    chips = {}
    for d in DEVICES:
        c = rx.PyGraph(); c.add_nodes_from(range(len(raw[d]["nodes"]))); c.add_edges_from_no_data([tuple(e) for e in raw[d]["edges"]])
        chips[d] = c
    off = 28  # arity column = off + 6
    rows, per_fam = [], collections.defaultdict(lambda: [0, 0])
    t_exact, t_proxy = [], []
    for d in data:
        name = d.circuit_name
        if name not in counts:
            continue
        g, has3 = interaction_graph(d)
        logical2q = int((d.x[:, off + 6] == 2).sum())
        if logical2q == 0 or has3:
            continue
        t0 = time.perf_counter()
        deg = max(g.degree(v) for v in g.node_indices()) if g.num_nodes() else 0
        forest = g.num_edges() == g.num_nodes() - len(rx.connected_components(g))
        path = forest and deg <= 2
        t_proxy.append(time.perf_counter() - t0)
        for i, dev in enumerate(DEVICES):
            cz = float(counts[name][i, 2])
            truth = cz <= logical2q + 0.5
            t0 = time.perf_counter()
            exact = rx.is_subgraph_isomorphic(chips[dev], g, id_order=False, induced=False, call_limit=CALL_LIMIT)
            t_exact.append(time.perf_counter() - t0)
            rows.append((name.split("/")[0], dev, truth, exact, deg <= 2, forest, path))
            per_fam[name.split("/")[0]][0] += truth
            per_fam[name.split("/")[0]][1] += 1
    a = np.array([r[2:] for r in rows], dtype=bool)
    truth = a[:, 0]
    print(f"{len(rows)} (circuit, device) pairs without 3-qubit gates; routing-free in truth: {truth.mean():.1%}\n")
    print(f"{'predictor':28}{'says free':>10}{'precision':>11}{'recall':>8}{'accuracy':>10}")
    for k, lab in enumerate(["exact (VF2 subgraph)", "max degree <= 2", "forest (no cycles)", "path"], start=1):
        p = a[:, k]
        prec = (p & truth).sum() / max(p.sum(), 1); rec = (p & truth).sum() / max(truth.sum(), 1)
        print(f"{lab:28}{p.mean():>10.1%}{prec:>11.3f}{rec:>8.3f}{(p == truth).mean():>10.3f}")
    print(f"\ntime per circuit: exact {np.median(t_exact)*1e3:.3f} ms median, {np.percentile(t_exact, 99)*1e3:.2f} ms p99, "
          f"{max(t_exact)*1e3:.1f} ms max (per device); proxies {np.median(t_proxy)*1e3:.3f} ms")
    print("\nrouting-free share by family (truth) and how the exact test sees it:")
    fams = sorted(per_fam, key=lambda f: -per_fam[f][1])
    for f in fams:
        sel = np.array([r[0] == f for r in rows])
        print(f"  {f:28} n={sel.sum():5d}  truth free {truth[sel].mean():6.1%}  exact free {a[sel, 1].mean():6.1%}"
              f"  agree {(a[sel, 1] == truth[sel]).mean():6.1%}")


if __name__ == "__main__":
    main()
