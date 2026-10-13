#!/usr/bin/env python3
"""Approximate router for the duration of routed circuits, no learning and no compilation (v22 candidate).

Why: the formula head schedules the LOGICAL gate DAG; with the compiler's exact counts it still underestimates routed
circuits by 20–33% (esp_count_oracle.py), because routing SWAPs serialise physical qubits the logical circuit never
links.  The learned correction fixes that on the training families (control routed 4.2%, compiler seed floor 4.6%,
compile_check/duration_anatomy.py) but not on new ones (qaoa / variational routed ~15%).  A family-agnostic estimate of
the SWAP serialisation should generalise.  The exact qubit is not needed (and not learnable: v6, v12, v20).

Router, per (circuit, device): greedy placement of the interaction graph (as routing_estimate); gates in program
order; a 2-qubit gate on non-adjacent qubits first moves its first operand along a shortest path with SWAPs (each
SWAP occupies both physical qubits for 3 CZ + SWAP_R r), updating the placement; a 3-qubit gate brings the other
operands next to the first.  ASAP on the physical qubits; gate times from the fixed per-gate decomposition table
(results/decomp_table.json), r 42 ns, CZ 130 ns, measure 15 µs.
Report: per split and routing group, median error / bias of the router duration vs the true duration of the stored
compilation (esp_labels.pt), raw and after one scale fitted on the control split's routed circuits (in-distribution
only, then applied to the other splits), and the logical-DAG schedule with the same table (what v21t uses) for
comparison.
    python esp_router.py [--max-circuits 0]          # -> results/report_esp_router.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3 import fast_encoding as fe  # noqa: E402
from gsv3.devices import GATE_US, variant_batch  # noqa: E402
from gsv3.model import GATE2IDX_MEASURE, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LABELS = EVAL / "compile_check/results/esp_labels.pt"
SPLITS = {"LOFO qaoa": "leave_one_family_out/qaoa", "LOFO qnn": "leave_one_family_out/qnn",
          "LOGO variational": "leave_one_group_out/variational", "control": "random_control/random_seed5"}
T1, TM, TCZ = GATE_US
SWAP_R = 4            # 1-qubit gates of a SWAP in the CZ basis besides its 3 CZ (split over the two qubits)


def chips(devices):
    ei = devices.edge_index.cpu()
    out, offset = [], 0
    for n in torch.bincount(devices.batch.cpu()).tolist():
        sel = (ei[0] >= offset) & (ei[0] < offset + n)
        edges = sorted({(min(a, b), max(a, b)) for a, b in (ei[:, sel] - offset).t().tolist()})
        offset += n
        adj = [[] for _ in range(n)]
        for a, c in edges:
            adj[a].append(c); adj[c].append(a)
        H, nxt = [], []
        for s0 in range(n):                                   # BFS: hops and next hop towards every target
            dist = [n + 1] * n; dist[s0] = 0; par = [-1] * n; q = deque([s0])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] > dist[u] + 1:
                        dist[v] = dist[u] + 1; par[v] = u; q.append(v)
            H.append(dist); nxt.append(par)
        out.append((n, adj, H, nxt))
    return out


def path(nxt, s, t):
    """Shortest path s -> t (list of physical qubits) from the BFS parents of t's tree (par rooted at t)."""
    p, u = [s], s
    while u != t:
        u = nxt[t][u]
        p.append(u)
    return p


def greedy_place(pairs, n_logical, H, adj):
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
        if q not in pos:
            best = max(free, key=lambda p: pdeg[p]); pos[q] = best; free.discard(best)
    return pos


def route_schedule(gates, n_logical, chip):
    """gates: list of (logical qubits, r count, cz count, is_measure) in program order -> (duration µs, SWAPs)."""
    n, adj, H, par = chip
    pairs: dict = {}
    for qs, *_ in gates:
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                k = (min(qs[i], qs[j]), max(qs[i], qs[j])); pairs[k] = pairs.get(k, 0) + 1
    pos = greedy_place(pairs, n_logical, H, adj)
    at = {p: q for q, p in pos.items()}                       # physical -> logical
    avail = [0.0] * n
    t_swap = 3 * TCZ + SWAP_R / 2 * T1
    n_swap = 0
    for qs, r, cz, meas in gates:
        if meas:
            p = pos[qs[0]]; avail[p] += TM; continue
        if len(qs) >= 2:
            anchor = pos[qs[0]]
            for o in qs[1:]:
                po = pos[o]
                if H[po][anchor] <= 1:
                    continue
                pth = path(par, po, anchor)[:-1]             # move o until adjacent to the anchor
                for u, v in zip(pth[:-1], pth[1:]):
                    s = max(avail[u], avail[v]) + t_swap
                    avail[u] = avail[v] = s
                    qu, qv = at.get(u), at.get(v)
                    at[u], at[v] = qv, qu
                    if qu is not None:
                        pos[qu] = v
                    if qv is not None:
                        pos[qv] = u
                    n_swap += 1
                anchor = pos[qs[0]]
        ps = [pos[q] for q in qs]
        t = r * T1 / len(qs) + cz * TCZ
        s = max(avail[p] for p in ps) + t
        for p in ps:
            avail[p] = s
    return max(avail), n_swap


def logical_schedule(gates, n_logical):
    avail = [0.0] * n_logical
    for qs, r, cz, meas in gates:
        t = TM if meas else r * T1 / len(qs) + cz * TCZ
        s = max(avail[q] for q in qs) + t
        for q in qs:
            avail[q] = s
    return max(avail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-circuits", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    chs = chips(dev)
    U = list(fe.UNIQUE)
    tab = json.loads((HERE / "results/decomp_table.json").read_text())
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    rows = []
    rf_cache: dict = {}
    bad_order = [0]
    for split, sd in SPLITS.items():
        names = [n for n in json.loads((EVAL / f"generalization_v10/results/v8kdfi10/{sd}/predictions.json").read_text())
                 if n in lab and n in data]
        if args.max_circuits:
            names = names[: args.max_circuits]
        for s in range(0, len(names), 64):
            nm = names[s: s + 64]
            b = next(iter(DataLoader([data[n] for n in nm], batch_size=len(nm))))
            rf = routing_free(b, dev, rf_cache).numpy() > 0.5
            for i, n in enumerate(nm):
                g = data[n]
                oh = g.x[:, : len(U)].numpy()
                meas = g.x[:, GATE2IDX_MEASURE].numpy() > 0.5
                gq = g.gate_qubits.numpy()
                ei = g.edge_index.numpy()
                if ei.shape[1] and not (ei[0] < ei[1]).all():
                    bad_order[0] += 1
                gates = [([int(x) for x in gq[k] if x >= 0], float(oh[k] @ tr), float(oh[k] @ tc), bool(meas[k]))
                         for k in range(len(gq))]
                gates = [x for x in gates if x[0]]
                nl = int(g.n_qubits)
                true = lab[n]["duration"].numpy() * 1e6
                lg = logical_schedule(gates, nl)
                for di in range(len(DEVS)):
                    rt, nsw = route_schedule(gates, nl, chs[di])
                    rows.append((split, bool(rf[i, di]), float(true[di]), rt, lg, nsw))
            print(split, s + len(nm), "/", len(names), flush=True)
    print("circuits whose graph edges are not all forward (program order assumed):", bad_order[0], flush=True)
    split_a = np.array([r[0] for r in rows]); rf_a = np.array([r[1] for r in rows])
    t, rt, lg = (np.array([r[k] for r in rows]) for k in (2, 3, 4))
    cal = (split_a == "control") & ~rf_a
    k_rt = float(np.median(np.log(t[cal] / rt[cal])))
    k_lg = float(np.median(np.log(t[cal] / lg[cal])))
    q = lambda e: f"{np.median(e):.3f} [{np.percentile(e, 90):.3f}]"
    L = ["# Approximate router: duration of routed circuits without learning or compiling", "",
         f"Greedy placement + shortest-path SWAPs (3 CZ + {SWAP_R} r each), ASAP on physical qubits, fixed decomposition "
         "table; vs the true duration of the stored compilation. Error = |est / true − 1|, median [p90]; bias = median "
         f"est / true. Scale fitted on the control split's routed circuits only: router × {np.exp(k_rt):.3f}, logical "
         f"schedule × {np.exp(k_lg):.3f}.", "",
         "| split | group | cases | router: error | router: bias | router × control scale: error | logical schedule: error | "
         "logical: bias | logical × control scale: error | corr(log est, log true), router / logical |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for split in [*SPLITS, "all"]:
        for grp in ("routed", "routing-free"):
            sel = ((split_a == split) if split != "all" else np.ones_like(rf_a)) & (rf_a == (grp == "routing-free"))
            if sel.sum() < 20:
                continue
            e = lambda est, k=0.0: np.abs(est[sel] * np.exp(k) / t[sel] - 1)
            c = lambda est: np.corrcoef(np.log(est[sel]), np.log(t[sel]))[0, 1]
            L.append(f"| {split} | {grp} | {int(sel.sum())} | {q(e(rt))} | {np.median(rt[sel] / t[sel]):.2f} | {q(e(rt, k_rt))} | "
                     f"{q(e(lg))} | {np.median(lg[sel] / t[sel]):.2f} | {q(e(lg, k_lg))} | {c(rt):.3f} / {c(lg):.3f} |")
    out = HERE / f"results/report_esp_router{args.tag}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
