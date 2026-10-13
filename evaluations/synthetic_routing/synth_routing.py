#!/usr/bin/env python3
"""Synthetic training circuits for the duration of routed circuits (v22).

Why: on families never seen in training the duration of routed circuits is off by ~15% (LOFO qaoa, LOGO variational;
compiler seed floor ~5%, compile_check/duration_anatomy.py), and the 90% upper bound calibrated on the control split
covers only 35–53% there (esp/esp_interval.py): the learned correction recognises families instead of routing.  These
circuits show the model how routing stretches the schedule over many generic interaction structures.

Circuits (deterministic in their index): 3–20 qubits; an interaction graph of one structure — chain, ring, star, grid,
tree, bipartite, sparse / medium / dense random graph, random long-range pairs, complete; layers of random 1-qubit
gates and 2-qubit gates on random edges of that graph (one dominant 2-qubit gate type per circuit, some others, rare
ccx), final measurements on every qubit (as all v10 circuits, qaoa included since compile_check/qaoa_measure.py).  Gates from the dataset's OpenQASM 3 basis.  No qaoa cost / mixer
layers and no variational ansatz templates, so the held-out families stay unseen.  Family name ``synth``: only ever
added to the training side of a split (run_experiment.py --extra-train), never to a test set.

Labels exactly as the dataset's (compile_check): Qiskit preset level 2, seed 0, original calibration, per device:
F (MQT expected fidelity, compile_layouts.fidelity), layout (index among the active qubits), counts (r, measure, cz),
and ESP / depth / duration / decay (compile_check/esp_labels.measures, QExa20 T1/T2 unit fix at merge).
    python synth_routing.py run --task-id 0 --n-tasks 16 --n-circuits 15000
    python synth_routing.py merge          # -> results/synth_{graphs,counts,layouts,esp_labels}.pt
Files: circuits/synth/<name>.qasm (logical circuits), results/qpy/synth/<name>.qpy (compilations, device order),
results/task*.jsonl (raw labels).
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
sys.path.insert(0, str(EVAL / "compile_check"))
CIRC = HERE / "circuits"
RES = HERE / "results"
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
STRUCTS = ("chain", "ring", "star", "grid", "tree", "bipartite", "sparse", "medium", "dense", "longrange", "complete")
ONE_Q = ("x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "p", "rx", "ry", "rz", "u")
TWO_Q = ("cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "cu", "swap")
PARAMS = {"p": 1, "rx": 1, "ry": 1, "rz": 1, "u": 3, "cp": 1, "crx": 1, "cry": 1, "crz": 1, "cu": 4}


def interaction_graph(struct: str, n: int, rng) -> list[tuple[int, int]]:
    perm = rng.permutation(n)                                   # logical labels shuffled: no index-order shortcut
    e: set = set()
    if struct == "chain":
        e = {(i, i + 1) for i in range(n - 1)}
    elif struct == "ring":
        e = {(i, (i + 1) % n) for i in range(n)}
    elif struct == "star":
        hubs = 1 if n < 8 else 2
        e = {(h, j) for h in range(hubs) for j in range(hubs, n)}
    elif struct == "grid":
        w = max(2, int(round(math.sqrt(n))))
        e = {(i, i + 1) for i in range(n - 1) if (i + 1) % w} | {(i, i + w) for i in range(n - w)}
    elif struct == "tree":
        e = {(int(rng.integers(0, i)), i) for i in range(1, n)}
    elif struct == "bipartite":
        a = max(1, n // 2)
        e = {(i, j) for i in range(a) for j in range(a, n) if rng.random() < 0.5}
    elif struct in ("sparse", "medium", "dense"):
        p = {"sparse": 0.15, "medium": 0.35, "dense": 0.7}[struct]
        e = {(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p}
    elif struct == "longrange":
        e = {(i, j) for i in range(n) for j in range(i + 1, n) if j - i >= max(2, n // 3) and rng.random() < 0.4}
    elif struct == "complete":
        e = {(i, j) for i in range(n) for j in range(i + 1, n)}
    if not e:
        e = {(0, 1)}
    return sorted((int(perm[a]), int(perm[b])) for a, b in e)


def make_circuit(idx: int):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(1_000_003 + idx)
    struct = STRUCTS[idx % len(STRUCTS)]
    n = int(rng.integers(3, 21))
    edges = interaction_graph(struct, n, rng)
    main2 = TWO_Q[int(rng.integers(len(TWO_Q)))]
    target_2q = int(rng.integers(n, 15 * n + 1))
    p1 = float(rng.uniform(0.2, 0.9))                            # share of qubits getting a 1-qubit gate per layer
    qc = QuantumCircuit(n)
    n2 = 0

    def gate(name, qs):
        k = PARAMS.get(name, 0)
        getattr(qc, name)(*[float(a) for a in rng.uniform(0.1, 2 * math.pi, k)], *qs)

    while n2 < target_2q and qc.size() < 1500:
        for q in range(n):
            if rng.random() < p1:
                gate(ONE_Q[int(rng.integers(len(ONE_Q)))], [q])
        busy: set = set()
        for k in rng.permutation(len(edges)):
            a, b = edges[k]
            if a in busy or b in busy or rng.random() < 0.3:
                continue
            busy |= {a, b}
            if n >= 3 and rng.random() < 0.02:
                c = next((q for q in rng.permutation(n) if q not in (a, b)), None)
                if c is not None:
                    qc.ccx(a, b, int(c)); n2 += 3; continue
            name = main2 if rng.random() < 0.7 else TWO_Q[int(rng.integers(len(TWO_Q)))]
            gate(name, [b, a] if rng.random() < 0.5 else [a, b])
            n2 += 1
    qc.measure_all()      # as every circuit of the v10 data (qaoa got its final measurements: qaoa_measure.py)
    return f"synth/synth_{struct}_q{n}_v{idx:05d}", qc


def run(args) -> None:
    from qiskit import qasm3, qpy
    from compile_check import compile_once
    from compile_layouts import fidelity
    from esp_labels import measures
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    backends = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    for be in backends.values():                         # T1/T2 in the target for MQT's ESP (as esp_labels.py)
        t = be.target
        if t.qubit_properties is None:
            qp = be.qubit_properties
            t.qubit_properties = list(qp) if not callable(qp) else [qp(q) for q in range(t.num_qubits)]
    node = {d: {q: i for i, q in enumerate(getattr(b, "active_qubits", None) or range(b.num_qubits))}
            for d, b in backends.items()}
    idxs = [i for i in range(args.n_circuits) if zlib.crc32(str(i).encode()) % args.n_tasks == args.task_id]
    RES.mkdir(parents=True, exist_ok=True)
    out_file = RES / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t0 = time.time()
    with out_file.open("a") as fh:
        for k, idx in enumerate(idxs):
            name, qc = make_circuit(idx)
            if name in done:
                continue
            rec = {"circuit": name, "layout": {}, "F": {}, "counts": {}, "esp": {}}
            try:
                qpath = CIRC / f"{name}.qasm"
                qpath.parent.mkdir(parents=True, exist_ok=True)
                qpath.write_text(qasm3.dumps(qc))
                qc = qasm3.loads(qpath.read_text())              # exactly the stored circuit
                compiled = []
                for dev, be in backends.items():
                    tc, _ = compile_once(qc, be, 2, 0)
                    compiled.append(tc)
                    lay = tc.layout.initial_index_layout(filter_ancillas=True)
                    rec["layout"][dev] = [node[dev].get(p, -1) for p in lay]
                    rec["F"][dev] = fidelity(tc, be.target)
                    ops = tc.count_ops()
                    rec["counts"][dev] = [int(sum(v for g, v in ops.items() if g not in ("cz", "measure", "barrier", "delay"))),
                                          int(ops.get("measure", 0)), int(ops.get("cz", 0))]
                    rec["esp"][dev] = measures(tc, be.target)
                cpath = RES / "qpy" / f"{name}.qpy"
                cpath.parent.mkdir(parents=True, exist_ok=True)
                with cpath.open("wb") as f:
                    qpy.dump(compiled, f)
            except Exception as e:                               # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if k % 100 == 0:
                print(f"[{time.time() - t0:6.0f} s] {k}/{len(idxs)} {name}", flush=True)


def merge(args) -> None:
    import torch
    sys.path.insert(0, str(EVAL / "generalization_v3"))
    from gsv3 import paths
    paths.ensure_imports()
    from gsv3.fast_encoding import to_data
    from qiskit import qasm3
    from esp_labels import T_FIX
    graphs, counts, layouts, esp, err = [], {}, {}, {}, 0
    for f in sorted(RES.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            name = r["circuit"]
            if name in counts:                                   # computed twice (smoke run + task): keep one
                continue
            if "error" in r or any(r["F"][d] is None for d in DEVICES):
                err += 1
                continue
            g = to_data(qasm3.loads((CIRC / f"{name}.qasm").read_text()))
            g.y = torch.tensor([[float(r["F"][d]) for d in DEVICES]])
            g.circuit_name = name
            graphs.append(g)
            counts[name] = torch.tensor([r["counts"][d] for d in DEVICES], dtype=torch.float)
            layouts[name] = torch.tensor([r["layout"][d] for d in DEVICES], dtype=torch.long).t().contiguous()
            rec = {k: torch.tensor([float(r["esp"][d][k]) for d in DEVICES]) for k in ("F", "ESP", "depth", "depth_2q", "duration", "decay")}
            rec["decay"] = rec["decay"] / torch.tensor([T_FIX.get(d, 1.0) for d in DEVICES])
            rec["ESP"] = rec["F"] * torch.exp(rec["decay"])
            esp[name] = rec
    for obj, fn in ((graphs, "synth_graphs.pt"), (counts, "synth_counts.pt"), (layouts, "synth_layouts.pt"),
                    (esp, "synth_esp_labels.pt")):
        torch.save(obj, RES / fn)
    rf = sum(1 for g in graphs)
    print(f"{rf} circuits ({err} failed) -> {RES}/synth_*.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "merge"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n-circuits", type=int, default=15000)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else merge(args)


if __name__ == "__main__":
    main()
