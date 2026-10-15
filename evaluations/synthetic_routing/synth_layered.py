#!/usr/bin/env python3
"""Synthetic circuits of the class the routed-duration model misses (v27 data).

Why (2026-10-15, route_stage v25 results): v25 halved the error on unseen synthetic structures (13.8% -> 7.6%) and on
control (5.3% -> 3.3%) but not on qaoa (13.9%, bias 0.86), and randomcircuit, seen in training, is as bad (13%, 0.87).
Both are dense, irregular, LOW-PARALLELISM circuits that repeat the same pairs many times (2q gates per 2q layer / n/2:
qaoa 0.46, randomcircuit 0.32; gates per pair 2.5 / 4.2), where routing SWAPs land on the critical path.  The v22
synthetic circuits are built as parallel matchings (parallelism 0.9): that class is almost absent from training.
Two new structures (family ``synth``, names ``synth/synth_<struct>_q<n>_v<idx>``, idx from 100000, so the existing
circuits and their labels are untouched):
  layered    a random graph (density 0.25-0.85); 1-5 layers, each applying ONE 2-qubit rotation type (rzz, rxx, ryy, cp,
             crz; one per circuit, 80%) to EVERY edge in a random order (not a matching), then a 1-qubit layer — the
             generic shape of qaoa-like circuits, not the qaoa template (random gate types and angles, random graphs).
  seqdense   2-qubit gates on random pairs of a dense graph (density 0.5-1), drawn one after another with repetitions
             (no matching), mixed gate types, 1-qubit gates in between — the shape of randomcircuit.
Steps, all on CPU and without the cluster (outputs next to, never over, the bundle's files):
  gen     QASM -> circuits/synth/, graphs -> results/synth2_graphs.pt          (merge after the tasks: --merge)
  sabre   Sabre labels of these circuits (compile_check/sabre_labels.py, same code) -> results/sabre_synth2/task*.jsonl
  prep    stage-1 items as route_stage/prep.py -> route_stage/results/stage1_synth2.pt
    python synth_layered.py gen --task-id 0 --n-tasks 8 --n-circuits 6000 ; python synth_layered.py gen --merge
    python synth_layered.py sabre --task-id 0 --n-tasks 8 ;                   python synth_layered.py prep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EVAL / "compile_check"))
sys.path.insert(0, str(EVAL / "generalization_v3"))
sys.path.append(str(EVAL.parent))                  # createDevice.py at the repository root when ~/compileCircuits is absent
CIRC = HERE / "circuits/synth"
RES = HERE / "results"
SABRE_OUT = RES / "sabre_synth2"
FIRST = 100_000
STRUCTS2 = ("layered", "seqdense")
ROT2 = ("rzz", "rxx", "ryy", "cp", "crz")
ONE_Q = ("rx", "ry", "rz", "h", "sx", "x")
TWO_Q_MIX = ("cx", "cz", "cp", "crz", "rzz", "rxx", "ryy", "swap", "cy", "crx")


def _graph(n: int, density: float, rng) -> list[tuple[int, int]]:
    """Connected random graph with about ``density`` of all pairs."""
    perm = rng.permutation(n)
    edges = {(min(perm[i], perm[i + 1]), max(perm[i], perm[i + 1])) for i in range(n - 1)}      # spanning path
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < density:
                edges.add((a, b))
    return sorted((int(a), int(b)) for a, b in edges)


def _gate(qc, name: str, qs, rng):
    angles = {"rzz": 1, "rxx": 1, "ryy": 1, "cp": 1, "crz": 1, "crx": 1, "rx": 1, "ry": 1, "rz": 1}.get(name, 0)
    getattr(qc, name)(*[float(a) for a in rng.uniform(0.1, 2 * math.pi, angles)], *qs)


def make_circuit(idx: int):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(2_000_003 + idx)
    struct = STRUCTS2[idx % len(STRUCTS2)]
    n = int(rng.integers(6, 21))
    qc = QuantumCircuit(n)
    if struct == "layered":
        edges = _graph(n, float(rng.uniform(0.25, 0.85)), rng)
        main = ROT2[int(rng.integers(len(ROT2)))]
        layers = int(rng.integers(1, 6))
        one = ONE_Q[int(rng.integers(len(ONE_Q)))]
        for q in range(n):
            qc.h(q)
        for _ in range(layers):
            if qc.size() > 1500:
                break
            for k in rng.permutation(len(edges)):
                a, b = edges[k]
                name = main if rng.random() < 0.8 else ROT2[int(rng.integers(len(ROT2)))]
                _gate(qc, name, [a, b] if rng.random() < 0.5 else [b, a], rng)
            for q in range(n):
                _gate(qc, one, [q], rng)
    else:
        edges = _graph(n, float(rng.uniform(0.5, 1.0)), rng)
        target = int(rng.integers(3 * n, 15 * n + 1))
        p1 = float(rng.uniform(0.1, 0.6))
        n2 = 0
        while n2 < target and qc.size() < 1500:
            a, b = edges[int(rng.integers(len(edges)))]
            name = TWO_Q_MIX[int(rng.integers(len(TWO_Q_MIX)))]
            _gate(qc, name, [a, b] if rng.random() < 0.5 else [b, a], rng)
            n2 += 1
            for q in (a, b):
                if rng.random() < p1:
                    _gate(qc, ONE_Q[int(rng.integers(len(ONE_Q)))], [q], rng)
    qc.measure_all()
    return f"synth/synth_{struct}_q{n}_v{idx:06d}", qc


def gen(args) -> None:
    import torch
    from qiskit import qasm3
    from gsv3.fast_encoding import to_data
    out = RES / "synth2_graphs"
    if args.merge:
        graphs = []
        for f in sorted(out.glob("task*.pt")):
            graphs += torch.load(f, weights_only=False)
        torch.save(graphs, RES / "synth2_graphs.pt")
        print(f"{len(graphs)} graphs -> {RES / 'synth2_graphs.pt'}")
        return
    CIRC.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    idxs = [i for i in range(FIRST, FIRST + args.n_circuits) if zlib.crc32(str(i).encode()) % args.n_tasks == args.task_id]
    graphs = []
    for k, idx in enumerate(idxs):
        name, qc = make_circuit(idx)
        path = CIRC / f"{name.split('/')[1]}.qasm"
        path.write_text(qasm3.dumps(qc))
        g = to_data(qasm3.loads(path.read_text()))              # exactly the stored circuit, as synth_routing.merge
        g.circuit_name = name
        graphs.append(g)
        if k % 500 == 0:
            print(f"gen {k}/{len(idxs)} {name}", flush=True)
    torch.save(graphs, out / f"task{args.task_id:03d}.pt")


def sabre(args) -> None:
    """compile_check/sabre_labels.run on the new circuits only (same labelling code, own output directory)."""
    import sabre_labels as sl
    names = sorted(f"synth/{p.stem}" for p in CIRC.glob("synth_*.qasm") if p.stem.split("_")[1] in STRUCTS2)
    sl.names_list = lambda a: [(n, CIRC / f"{n.split('/')[1]}.qasm") for n in names
                               if zlib.crc32(n.encode()) % a.n_tasks == a.task_id]
    args.out, args.sample = SABRE_OUT, 0
    sl.run(args)


def prep(args) -> None:
    """route_stage/prep.py's items for the new circuits (chips from the tracked raw device graphs)."""
    import torch
    from collections import defaultdict
    sys.path.insert(0, str(EVAL / "route_stage"))
    import prep as P
    import stage1 as S
    tab = json.loads((EVAL / "esp/results/decomp_table.json").read_text())
    from gsv3 import fast_encoding as fe
    U = list(fe.UNIQUE)
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    chs = S.raw_chips()
    recs = {}
    for f in sorted(SABRE_OUT.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    recs[r["circuit"]] = r
    graphs = torch.load(RES / "synth2_graphs.pt", weights_only=False)
    out, skip = {}, defaultdict(int)
    D = len(P.DEVS)
    for g in graphs:
        rec = recs.get(g.circuit_name)
        if rec is None:
            skip["no label"] += 1; continue
        N = g.num_nodes
        if rec["n_nodes"] != N:
            skip["node count"] += 1; continue
        wires, t, meas = P.node_times(g, tr, tc)
        n_log = int(g.n_qubits)
        item = {"s": np.zeros((D, N, 3), np.float32), "wait": np.zeros((D, N), np.float32),
                "m": np.zeros((D, N), bool), "prior_s": np.zeros((D, N, 3), np.float32), "hops": np.zeros((D, N), np.float32),
                "rf": np.zeros(D, bool), "final": np.zeros(D, np.float32), "replay": np.full(D, np.nan, np.float32),
                "prior_dur": np.zeros(D, np.float32), "busy": np.zeros((D, n_log), np.float32), "aligned": np.ones(D, bool)}
        rows = {r["device"]: r for r in rec["rows"]}
        for di, d in enumerate(P.DEVS):
            r = rows.get(d)
            if r is None:
                skip["missing device"] += 1; continue
            item["rf"][di] = r["routing_free"]
            item["final"][di] = r["final"]
            ps, hp, pdur = P.prior_route(wires, t, meas, n_log, chs[di])
            item["prior_s"][di], item["hops"][di], item["prior_dur"][di] = ps, hp, pdur
            if r["routing_free"] or "swaps" not in r:
                continue
            for node, slot, n in r["swaps"]:
                item["s"][di, node, slot] = n
            tn = {k: (a, b) for k, a, b in r["t_node"]}
            item["m"][di, list(tn)] = True
            item["wait"][di] = P.wait_targets(wires, t, item["s"][di], tn)
            item["replay"][di] = r["replay"]
            item["busy"][di, : len(r["busy"])] = r["busy"]
            item["aligned"][di] = r["aligned"]
        out[g.circuit_name] = {k: torch.from_numpy(v) for k, v in item.items()}
    dst = EVAL / "route_stage/results/stage1_synth2.pt"
    torch.save(out, dst)
    routed = sum(int((~v["rf"]).sum()) for v in out.values())
    print(f"saved {len(out)} circuits ({routed} routed cases); skipped {dict(skip)} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gen", "sabre", "prep"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n-circuits", type=int, default=6000)
    ap.add_argument("--merge", action="store_true", help="gen: merge the task graph files into results/synth2_graphs.pt")
    args = ap.parse_args()
    {"gen": gen, "sabre": sabre, "prep": prep}[args.cmd](args)


if __name__ == "__main__":
    main()
