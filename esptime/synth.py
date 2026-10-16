#!/usr/bin/env python3
"""Synthetic training circuits for routing (family ``synth``: training side only, never in a test set).

generic  (index 0 .. n-1)        3-20 qubits, an interaction graph of one structure (chain, ring, star, grid, tree,
         bipartite, sparse / medium / dense random, long-range pairs, complete); layers of random 1-qubit gates and
         2-qubit gates on random edges as parallel matchings (one dominant 2-qubit type, rare ccx).
layered  (index 100000 + ...)    the dense, irregular, LOW-parallelism class the generic circuits lack (qaoa-like and
         randomcircuit-like, without their templates): ``layered`` = 1-5 layers of one 2-qubit rotation type on EVERY
         edge of a random graph in random order; ``seqdense`` = 2-qubit gates on random pairs drawn one after another.
Every circuit ends with measurements on all qubits.  Deterministic in the index.
    python -m esptime.synth --n-generic 15000 --n-layered 6000      # -> work/circuits/synth/*.qasm
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from .config import WORK

OUT = WORK / "circuits/synth"
STRUCTS = ("chain", "ring", "star", "grid", "tree", "bipartite", "sparse", "medium", "dense", "longrange", "complete")
ONE_Q = ("x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "p", "rx", "ry", "rz", "u")
TWO_Q = ("cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "cu", "swap")
N_PARAMS = {"p": 1, "rx": 1, "ry": 1, "rz": 1, "u": 3, "cp": 1, "crx": 1, "cry": 1, "crz": 1, "cu": 4,
            "rzz": 1, "rxx": 1, "ryy": 1}
FIRST_LAYERED = 100_000
STRUCTS2 = ("layered", "seqdense")
ROT2 = ("rzz", "rxx", "ryy", "cp", "crz")
ONE_Q2 = ("rx", "ry", "rz", "h", "sx", "x")
TWO_Q_MIX = ("cx", "cz", "cp", "crz", "rzz", "rxx", "ryy", "swap", "cy", "crx")


def _gate(qc, name, qs, rng):
    getattr(qc, name)(*[float(a) for a in rng.uniform(0.1, 2 * math.pi, N_PARAMS.get(name, 0))], *qs)


def interaction_graph(struct: str, n: int, rng) -> list[tuple[int, int]]:
    perm = rng.permutation(n)                                   # shuffled labels: no index-order shortcut
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


def generic(idx: int):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(1_000_003 + idx)
    struct = STRUCTS[idx % len(STRUCTS)]
    n = int(rng.integers(3, 21))
    edges = interaction_graph(struct, n, rng)
    main2 = TWO_Q[int(rng.integers(len(TWO_Q)))]
    target_2q = int(rng.integers(n, 15 * n + 1))
    p1 = float(rng.uniform(0.2, 0.9))
    qc = QuantumCircuit(n)
    n2 = 0
    while n2 < target_2q and qc.size() < 1500:
        for q in range(n):
            if rng.random() < p1:
                _gate(qc, ONE_Q[int(rng.integers(len(ONE_Q)))], [q], rng)
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
            _gate(qc, name, [b, a] if rng.random() < 0.5 else [a, b], rng)
            n2 += 1
    qc.measure_all()
    return f"synth_{struct}_q{n}_v{idx:06d}", qc


def _random_graph(n: int, density: float, rng) -> list[tuple[int, int]]:
    perm = rng.permutation(n)
    edges = {(min(perm[i], perm[i + 1]), max(perm[i], perm[i + 1])) for i in range(n - 1)}      # connected
    edges |= {(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < density}
    return sorted((int(a), int(b)) for a, b in edges)


def layered(idx: int):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(2_000_003 + idx)
    struct = STRUCTS2[idx % len(STRUCTS2)]
    n = int(rng.integers(6, 21))
    qc = QuantumCircuit(n)
    if struct == "layered":
        edges = _random_graph(n, float(rng.uniform(0.25, 0.85)), rng)
        main = ROT2[int(rng.integers(len(ROT2)))]
        one = ONE_Q2[int(rng.integers(len(ONE_Q2)))]
        for q in range(n):
            qc.h(q)
        for _ in range(int(rng.integers(1, 6))):
            if qc.size() > 1500:
                break
            for k in rng.permutation(len(edges)):
                a, b = edges[k]
                name = main if rng.random() < 0.8 else ROT2[int(rng.integers(len(ROT2)))]
                _gate(qc, name, [a, b] if rng.random() < 0.5 else [b, a], rng)
            for q in range(n):
                _gate(qc, one, [q], rng)
    else:
        edges = _random_graph(n, float(rng.uniform(0.5, 1.0)), rng)
        target = int(rng.integers(3 * n, 15 * n + 1))
        p1 = float(rng.uniform(0.1, 0.6))
        for _ in range(target):
            if qc.size() >= 1500:
                break
            a, b = edges[int(rng.integers(len(edges)))]
            _gate(qc, TWO_Q_MIX[int(rng.integers(len(TWO_Q_MIX)))], [a, b] if rng.random() < 0.5 else [b, a], rng)
            for q in (a, b):
                if rng.random() < p1:
                    _gate(qc, ONE_Q2[int(rng.integers(len(ONE_Q2)))], [q], rng)
    qc.measure_all()
    return f"synth_{struct}_q{n}_v{idx:06d}", qc


def struct_of(name: str) -> str:
    """``synth/synth_<struct>_q<n>_v<idx>`` -> struct (the LOSO splits)."""
    return name.split("/")[-1].split("_")[1]


def main():
    from qiskit import qasm3
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-generic", type=int, default=15000)
    ap.add_argument("--n-layered", type=int, default=6000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(generic, i) for i in range(args.n_generic)] + \
           [(layered, FIRST_LAYERED + i) for i in range(args.n_layered)]
    for k, (fn, idx) in enumerate(jobs):
        name, qc = fn(idx)
        (OUT / f"{name}.qasm").write_text(qasm3.dumps(qc))
        if k % 1000 == 0:
            print(f"{k}/{len(jobs)} {name}", flush=True)
    print(f"{len(jobs)} circuits -> {OUT}")


if __name__ == "__main__":
    main()
