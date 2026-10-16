#!/usr/bin/env python3
"""Our own simplified Sabre router (v28 input features; no compiler pass at prediction time).

Why: the greedy router (prep.prior_route, esp_router) is far from Sabre (its duration alone 56% off on qaoa, 1.56x), so
its features (prior SWAPs, hops, contention) carry little routing information.  Qiskit's own Sabre with one trial is
10-28x cheaper than level 2 but is a compiler pass; this is a re-implementation of the same heuristic (Li, Ding, Xie,
ASPLOS 2019, as in Qiskit's SabreSwap 'decay' mode), simplified:
  * layout: our greedy placement, refined by ONE reverse pass (route the reversed circuit, start from where it ends);
  * routing: one forward pass, no trials.  Front layer F = the 2-qubit gates whose wire predecessors are done; while no
    gate of F is adjacent, apply the SWAP (on an edge touching a qubit of F) minimising
        max(decay_a, decay_b) * ( mean_F dist + W * mean_E dist ),   E = the next EXT 2-qubit gates (lookahead),
    decay += DELTA on swapped qubits, reset every RESET swaps; ties -> lowest edge index (deterministic).
Outputs per gate node (same layout as prep.prior_route, so they drop into stage 1): SWAPs attributed to the operand wires
(pending SWAPs of a logical qubit counted at its next multi-qubit gate or measurement), hops at the moment the gate
enters the front layer, the contention wait of the physical ASAP schedule, and the router's duration.
    python sabre_lite.py check --family qaoa --n 15     # vs Qiskit's Sabre (SWAPs, duration) and time, on data/circuits.zip
Result (2026-10-16, NEGATIVE, not used): qaoa, 36 routed cases, 8-20 qubits, vs Qiskit level 2: SWAPs 2.40x Sabre's
(greedy 3.95x), router duration 1.77x the compiled one (greedy 1.79x), 70 ms per (circuit, device) in Python (greedy
1.7 ms).  Qiskit's own one-trial Sabre: 1.17x SWAPs in 3.7 ms.  A competitive re-implementation needs Rust / numba-level
engineering; decision pending between Qiskit's light Sabre as a declared input and staying compiler-free (v27).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "esp"))
T1, TM, TCZ = 0.042, 15.0, 0.130
T_SWAP = 3 * TCZ + 2 * T1
EXT, W, DELTA, RESET = 20, 0.5, 0.001, 5


def _pairs(wires):
    pairs: dict = {}
    for qs in wires:
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                k = (min(qs[i], qs[j]), max(qs[i], qs[j])); pairs[k] = pairs.get(k, 0) + 1
    return pairs


def _route(wires, pos, chip, record=None):
    """One Sabre pass over ``wires`` (gate operand lists in program order) from layout ``pos`` (logical -> physical).
    Returns the final layout.  With ``record`` (dict of lists), records the events for the schedule:
    ('g', gate index, hops) when a gate executes, ('s', u, v) for a SWAP on physical qubits u, v."""
    n, adj, H, _ = chip
    pos = dict(pos)
    at = {p: q for q, p in pos.items()}
    N = len(wires)
    # wire predecessors: a gate is ready when every operand wire's previous gate is done
    nxt_on_wire = {}
    preds = [0] * N
    succ = [[] for _ in range(N)]
    last = {}
    for i, qs in enumerate(wires):
        for q in qs:
            if q in last:
                succ[last[q]].append(i); preds[i] += 1
            last[q] = i
    ready = [i for i in range(N) if preds[i] == 0 and wires[i]]
    two_q = [len(qs) >= 2 for qs in wires]
    done = [False] * N
    order2 = [i for i in range(N) if two_q[i]]          # 2q gates in program order, for the lookahead
    rank = {g: k for k, g in enumerate(order2)}
    decay = defaultdict(lambda: 1.0)
    n_swaps_since = 0
    entered = {}
    front: set = set()
    stack = list(ready)

    def dist(g):
        qs = wires[g]
        return max(H[pos[a]][pos[b]] for a in qs for b in qs if a != b)

    def release(i):
        for j in succ[i]:
            preds[j] -= 1
            if preds[j] == 0 and wires[j]:
                stack.append(j)

    guard = 0
    while stack or front:
        progressed = True
        while progressed:
            progressed = False
            while stack:                                  # non-2q gates execute at once; 2q gates join the front
                i = stack.pop()
                if two_q[i]:
                    front.add(i)
                    entered.setdefault(i, dist(i) - 1)
                else:
                    done[i] = True
                    if record is not None:
                        record["ev"].append(("g", i, 0))
                    release(i)
                    progressed = True
            for g in sorted(front):
                if dist(g) <= 1:
                    front.discard(g); done[g] = True
                    if record is not None:
                        record["ev"].append(("g", g, entered.get(g, 0)))
                    release(g)
                    progressed = True
                    decay.clear(); n_swaps_since = 0
        if not front:
            break
        # lookahead set: the next EXT not-done 2q gates after the front, in program order
        start = min(rank[g] for g in front)
        ext = []
        for k in range(start, len(order2)):
            g = order2[k]
            if not done[g] and g not in front:
                ext.append(g)
                if len(ext) >= EXT:
                    break
        fq = {pos[q] for g in front for q in wires[g]}
        cand = sorted({(min(p, r), max(p, r)) for p in fq for r in adj[p]})
        best, best_s = None, None
        for (u, v) in cand:
            qu, qv = at.get(u), at.get(v)
            # tentative swap
            if qu is not None: pos[qu] = v
            if qv is not None: pos[qv] = u
            hf = sum(dist(g) for g in front) / len(front)
            he = (sum(dist(g) for g in ext) / len(ext)) if ext else 0.0
            s = max(decay[u], decay[v]) * (hf + W * he)
            if qu is not None: pos[qu] = u
            if qv is not None: pos[qv] = v
            if best_s is None or s < best_s - 1e-12:
                best, best_s = (u, v), s
        u, v = best
        qu, qv = at.get(u), at.get(v)
        at[u], at[v] = qv, qu
        if qu is not None: pos[qu] = v
        if qv is not None: pos[qv] = u
        decay[u] += DELTA; decay[v] += DELTA
        n_swaps_since += 1
        if n_swaps_since >= RESET:
            decay.clear(); n_swaps_since = 0
        if record is not None:
            record["ev"].append(("s", u, v))
        guard += 1
        if guard > 50 * (N + 10):                         # safety: never loop forever
            break
    return pos


def sabre_lite(gq: np.ndarray, t: np.ndarray, meas: np.ndarray, n_log: int, chip):
    """(s (N,3) SWAPs per operand slot, hops (N,), contention wait (N,) µs, duration µs) of our Sabre-like routing."""
    from esp_router import greedy_place
    n, adj, H, _ = chip
    wires = [[int(x) for x in row if x >= 0] for row in gq]
    N = len(wires)
    s = np.zeros((N, 3), np.float32); hops = np.zeros(N, np.float32); wait = np.zeros(N, np.float32)
    if n_log > n:
        return s, hops, wait, float("nan")
    pos = greedy_place(_pairs(wires), n_log, H, adj)
    pos = _route(wires[::-1], pos, chip)                  # one reverse pass refines the initial layout (Sabre layout)
    rec = {"ev": []}
    _route(wires, pos, chip, rec)
    # physical ASAP schedule of the routed event sequence, SWAPs attributed to the wires they move
    at = {p: q for q, p in pos.items()}
    avail = [0.0] * n
    last = defaultdict(float)
    pending = defaultdict(int)
    for ev in rec["ev"]:
        if ev[0] == "s":
            _, u, v = ev
            st = max(avail[u], avail[v]) + T_SWAP
            avail[u] = avail[v] = st
            qu, qv = at.get(u), at.get(v)
            for w in (qu, qv):
                if w is not None:
                    pending[w] += 1
            at[u], at[v] = qv, qu
            continue
        _, gi, hp = ev
        qs = wires[gi]
        hops[gi] = hp
        own = {}
        if len(qs) >= 2 or meas[gi]:
            for k, w in enumerate(qs):
                own[w] = pending[w]
                s[gi, k] = pending[w]
                pending[w] = 0
        cur = {q: p for p, q in at.items() if q is not None}
        ps = [cur[q] for q in qs]
        ready = max(last[w] + own.get(w, 0) * T_SWAP for w in qs)
        st = max(avail[p] for p in ps)
        wait[gi] = st - ready
        e = st + t[gi]
        for p in ps:
            avail[p] = e
        for w in qs:
            last[w] = e
    return s, hops, wait, float(max(avail))


def check(args) -> None:
    """Our Sabre-lite vs Qiskit's Sabre (level 2, seed 0) on circuits of data/circuits.zip: SWAP count, router duration vs
    the compiled duration, time per (circuit, device)."""
    import json
    import random
    import zipfile
    sys.path[:0] = [str(EVAL.parent), str(EVAL / "compile_check"), str(EVAL / "generalization_v3"), str(HERE)]
    from qiskit import qasm3, transpile
    from qiskit.transpiler import generate_preset_pass_manager
    from v8b_variants import variants
    from gsv3 import fast_encoding as fe
    from gsv3.fast_encoding import to_data
    import stage1 as S
    from prep import prior_route
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    chs = S.raw_chips()
    tab = json.loads((EVAL / "esp/results/decomp_table.json").read_text()); U = list(fe.UNIQUE)
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float); tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    z = zipfile.ZipFile(EVAL.parent / "data/circuits.zip")
    names = sorted(x for x in z.namelist() if f"/{args.family}/" in x and x.endswith(".qasm"))
    random.Random(1).shuffle(names)
    rows, k = [], 0
    for x in names:
        qc = qasm3.loads(z.read(x).decode())
        if args.family == "qaoa" and "measure" not in qc.count_ops():
            qc.measure_all()
        if not (8 <= qc.num_qubits <= 20):
            continue
        g = to_data(qc)
        gq = g.gate_qubits.numpy(); t = S.node_t(g, tr, tc); meas = g.x[:, S.GATE2IDX_MEASURE].numpy() > 0.5
        qb = fe.to_basis(qc)
        for di, (d, be) in enumerate(base.items()):
            pm = generate_preset_pass_manager(optimization_level=2, target=be.target, seed_transpiler=0)
            routed = {}

            def cb(**kw):
                if type(kw["pass_"]).__name__ == "SabreLayout":
                    routed["c"] = kw["dag"]
            tcomp = pm.run(qb, callback=cb)
            if "c" not in routed:
                continue                                   # routing-free on this device
            sw_true = sum(1 for nd in routed["c"].op_nodes() if nd.op.name == "swap")
            sc = transpile(tcomp, target=be.target, scheduling_method="asap", optimization_level=0,
                           initial_layout=None, routing_method=None, layout_method=None)
            dur_true = sc.estimate_duration(target=be.target) * 1e6
            t0 = time.perf_counter(); s, hp, w, dur = sabre_lite(gq, t, meas, int(g.n_qubits), chs[di]); tl = time.perf_counter() - t0
            t0 = time.perf_counter(); ps, _, pdur = prior_route([[int(q) for q in r if q >= 0] for r in gq], t, meas, int(g.n_qubits), chs[di])
            tg = time.perf_counter() - t0
            rows.append((s.sum() / max(sw_true, 1), dur / dur_true, ps.sum() / max(sw_true, 1), pdur / dur_true, tl, tg))
        k += 1
        if k >= args.n:
            break
    r = np.array(rows)
    med = lambda c: np.median(r[:, c]); err = lambda c: np.median(np.abs(r[:, c] - 1))
    print(f"{args.family}: {len(r)} routed (circuit, device) cases, 8-20 qubits")
    print(f"  Sabre-lite: SWAPs / Sabre's {med(0):.2f} | router duration / compiled duration {med(1):.2f} (|err| {err(1):.0%}) | {med(4)*1e3:.1f} ms")
    print(f"  greedy    : SWAPs / Sabre's {med(2):.2f} | router duration / compiled duration {med(3):.2f} (|err| {err(3):.0%}) | {med(5)*1e3:.1f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--family", default="qaoa")
    ap.add_argument("--n", type=int, default=15)
    check(ap.parse_args())
