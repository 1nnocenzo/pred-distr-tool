"""Fast path for routing-free (circuit, device) pairs: F and ESP of every calibration without recompiling.

For a routing-free pair the compiled structure (native r / measure / CZ counts per logical qubit and pair, the
ASAP schedule) does not depend on the calibration; the calibration only decides WHERE the compiler's VF2Layout
puts it, by an average-error score (compile_check/results/pipeline_anatomy*/report.md).  So, per circuit and
topology, ONCE:
  1. switch: is the interaction graph a subgraph of the coupling graph (the VF2Layout decision);
  2. structure: one level-2 compilation on the reference calibration, pulled back to logical qubits through its
     layout (r, measure, CZ per pair, idle time per qubit) + the counts VF2Layout scores (1q / 2q ops of the
     logical circuit);
  3. embeddings: every embedding of the interaction graph in the coupling graph (rustworkx VF2, capped).
Per calibration (microseconds-milliseconds, no compiler):
  4. score every cached embedding with VF2Layout's score  Σ count · log(1 − average error), take the best,
     place the 1q-only qubits on the best remaining qubits (VF2Layout's free-qubit rule);
  5. log F = Σ r · log(1 − e_r) + measure · log(1 − e_m) + CZ · log(1 − e_cz) on that placement (MQT's
     expected fidelity), decay = − Σ_active idle_q / min(T1, T2) (MQT's ESP, checked to the last digit).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# QExa20's T1/T2 in createDevice.py are 10x too small; the ESP labels divide its decay by this factor
# (compile_check/esp_labels.py T_FIX).
T_FIX = {"QExa20": 10.0}
EMB_CALL_LIMIT = 2_000_000      # VF2 extension steps for the embedding enumeration
MAX_EMB = 20_000                # embeddings cached per (structure, topology)
DIRECTIVES = {"barrier", "delay", "id"}


# --------------------------------------------------------------------------------------------- calibration
@dataclass
class Calib:
    """Calibration of one device as arrays over physical qubits (indices of the backend target)."""
    n: int
    lr: np.ndarray            # (n,) log(1 - e_r), nan where the gate is missing
    lm: np.ndarray            # (n,) log(1 - e_measure)
    lcz: np.ndarray           # (n, n) log(1 - e_cz), symmetric, nan off the coupling map
    inv_t: np.ndarray         # (n,) 1 / min(T1, T2) in 1/s, divided by T_FIX
    s1: np.ndarray            # (n,) log(1 - average 1q error): VF2Layout's node score
    s2: np.ndarray            # (n, n) log(1 - average 2q error): VF2Layout's edge score
    edges: list = field(default_factory=list)


def qubit_props(backend) -> list:
    qp = backend.qubit_properties
    return [qp(q) for q in range(backend.target.num_qubits)] if callable(qp) else list(qp)


def calib(backend, name: str = "", avg_1q: str = "all") -> Calib:
    """Arrays of a backend's calibration.  ``avg_1q``: which 1q operations VF2Layout's average error map
    averages over ("all" = every 1q instruction with an error, as Qiskit's build_average_error_map)."""
    t = backend.target
    n = t.num_qubits
    lr, lm = np.full(n, np.nan), np.full(n, np.nan)
    lcz = np.full((n, n), np.nan)
    e1_sum, e1_cnt = np.zeros(n), np.zeros(n)
    e2_sum, e2_cnt = np.zeros((n, n)), np.zeros((n, n))
    for op in t.operation_names:
        for qargs, props in t[op].items():
            if qargs is None or props is None or props.error is None:
                continue
            e = float(props.error)
            if len(qargs) == 1:
                q = qargs[0]
                if op == "r":
                    lr[q] = np.log1p(-e)
                elif op == "measure":
                    lm[q] = np.log1p(-e)
                if avg_1q == "all" or op in avg_1q.split(","):
                    e1_sum[q] += e; e1_cnt[q] += 1
            elif len(qargs) == 2:
                a, b = qargs
                if op == "cz":
                    lcz[a, b] = lcz[b, a] = np.log1p(-e)
                e2_sum[a, b] += e; e2_cnt[a, b] += 1
    # undirected: VF2Layout with strict_direction=False scores an edge by the direction the target has
    e2_sum, e2_cnt = e2_sum + e2_sum.T, e2_cnt + e2_cnt.T
    with np.errstate(invalid="ignore", divide="ignore"):
        s1 = np.where(e1_cnt > 0, np.log1p(-e1_sum / np.maximum(e1_cnt, 1)), np.nan)
        s2 = np.where(e2_cnt > 0, np.log1p(-e2_sum / np.maximum(e2_cnt, 1)), np.nan)
    qp = qubit_props(backend)
    tmin = np.array([min(p.t1, p.t2) if p is not None and p.t1 and p.t2 else np.inf for p in qp], float)
    inv_t = 1.0 / tmin / T_FIX.get(name.split("/")[0], 1.0)
    edges = sorted({(min(a, b), max(a, b)) for a, b in t["cz"].keys() if a != b})
    return Calib(n, lr, lm, lcz, inv_t, s1, s2, edges)


# ----------------------------------------------------------------------------------------------- switch
def interaction_pairs(qc) -> tuple[dict, dict]:
    """1q op count per qubit and 2q op count per pair of a (logical) circuit, as VF2Layout counts them
    (every non-directive operation; gates on 3+ qubits contribute all their pairs)."""
    c1, c2 = {}, {}
    for inst in qc.data:
        if inst.operation.name in DIRECTIVES:
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) == 1:
            c1[qs[0]] = c1.get(qs[0], 0) + 1
        else:
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    k = (min(qs[i], qs[j]), max(qs[i], qs[j]))
                    c2[k] = c2.get(k, 0) + 1
    return c1, c2


def coupling_graph(c: Calib):
    import rustworkx as rx
    g = rx.PyGraph()
    g.add_nodes_from(range(c.n))
    g.add_edges_from_no_data(c.edges)
    return g


def routing_free_switch(qc, c: Calib, call_limit: int = 20_000) -> bool:
    """The switch: does the circuit's interaction graph embed in the coupling graph (no routing needed)?
    Same test as diagnostics_v4v5/routing_free_feature.py (recall 0.999 against the compiler)."""
    import rustworkx as rx
    _, c2 = interaction_pairs(qc)
    if not c2:
        return True
    nodes = sorted({q for k in c2 for q in k})
    idx = {q: i for i, q in enumerate(nodes)}
    g = rx.PyGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from_no_data([(idx[a], idx[b]) for a, b in c2])
    return bool(rx.is_subgraph_isomorphic(coupling_graph(c), g, id_order=False, induced=False, call_limit=call_limit))


# -------------------------------------------------------------------------------------------- structure
@dataclass
class Structure:
    """Calibration-independent structure of one routing-free compilation, on logical qubits 0..k-1."""
    qubits: list              # logical qubit index (of the input circuit) of each row
    r: np.ndarray             # (k,) native r gates per logical qubit
    m: np.ndarray             # (k,) measurements
    cz: list                  # [(u, v, count)] native CZ per logical pair (rows)
    idle: np.ndarray          # (k,) ASAP idle time (s) of active qubits, 0 for inactive ones
    active: np.ndarray        # (k,) bool: the qubit carries an operation
    c1: np.ndarray            # (k,) 1q ops of the logical circuit (VF2Layout's node counts)
    c2: list                  # [(u, v, count)] 2q ops of the logical circuit (VF2Layout's edge counts)
    duration: float           # ASAP duration (s)
    layout: dict              # reference layout: logical qubit -> physical qubit
    t_compile: float = 0.0
    emb: np.ndarray | None = None       # (K, n_int) cached embeddings of the interacting rows
    int_rows: np.ndarray | None = None  # rows of the interacting qubits (columns of emb)
    t_emb: float = 0.0


def schedule(tc, target) -> tuple[dict, dict, float]:
    """ASAP schedule of a compiled circuit, as MQT's ESP: busy time per physical qubit and the duration."""
    clock, busy = {}, {}
    for inst in tc.data:
        name = inst.operation.name
        if name == "id":
            continue
        qs = tuple(tc.find_bit(q).index for q in inst.qubits)
        if name == "barrier":            # a sync point in Qiskit's estimate_duration (MQT's ESP duration)
            s = max(clock.get(q, 0.0) for q in qs)
            clock.update({q: s for q in qs})
            continue
        d = float(target[name][qs].duration)
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
            busy[q] = busy.get(q, 0.0) + d
    return busy, clock, (max(clock.values()) if clock else 0.0)


def compile_reference(qc, backend, seed: int):
    """Level-2 compilation (as the labels) with the logical circuit VF2Layout scored and its decision."""
    from qiskit.converters import dag_to_circuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
    stop = {}

    def cb(**kw):
        if kw["pass_"].name() == "VF2Layout":
            stop["virt"] = dag_to_circuit(kw["dag"])
            stop["found"] = kw["property_set"]["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND

    pm = generate_preset_pass_manager(optimization_level=2, target=backend.target, seed_transpiler=seed)
    t0 = time.perf_counter()
    tc = pm.run(qc, callback=cb)
    return tc, time.perf_counter() - t0, stop.get("found", False), stop.get("virt")


def structure(tc, virt, target, t_compile: float = 0.0) -> Structure:
    """Pull a routing-free compilation back to the logical qubits of its layout."""
    v2p = tc.layout.final_index_layout(filter_ancillas=True) if tc.layout is not None else list(range(tc.num_qubits))
    p2v = {p: v for v, p in enumerate(v2p)}
    busy, _, dur = schedule(tc, target)
    c1v, c2v = interaction_pairs(virt) if virt is not None else ({}, {})
    used = set(c1v) | {q for k in c2v for q in k}
    r, m, cz = {}, {}, {}
    for inst in tc.data:
        name = inst.operation.name
        qs = [tc.find_bit(q).index for q in inst.qubits]
        if name in DIRECTIVES:
            continue
        vs = [p2v.get(p) for p in qs]
        if any(v is None for v in vs):
            raise ValueError("operation on an ancilla: the compilation was routed")
        used.update(vs)
        if name == "r":
            r[vs[0]] = r.get(vs[0], 0) + 1
        elif name == "measure":
            m[vs[0]] = m.get(vs[0], 0) + 1
        elif name == "cz":
            k = (min(vs), max(vs)); cz[k] = cz.get(k, 0) + 1
        else:
            raise ValueError(f"unexpected native operation {name}")
    qubits = sorted(used)
    row = {v: i for i, v in enumerate(qubits)}
    act = np.array([v2p[v] in busy for v in qubits])
    idle = np.array([dur - busy.get(v2p[v], 0.0) if v2p[v] in busy else 0.0 for v in qubits])
    return Structure(
        qubits=qubits,
        r=np.array([r.get(v, 0) for v in qubits], float), m=np.array([m.get(v, 0) for v in qubits], float),
        cz=[(row[a], row[b], c) for (a, b), c in sorted(cz.items())], idle=idle, active=act,
        c1=np.array([c1v.get(v, 0) for v in qubits], float),
        c2=[(row[a], row[b], c) for (a, b), c in sorted(c2v.items()) if a in row and b in row],
        duration=dur, layout={v: v2p[v] for v in qubits}, t_compile=t_compile)


def embeddings(s: Structure, c: Calib, call_limit: int = EMB_CALL_LIMIT, max_emb: int = MAX_EMB) -> Structure:
    """Cache every embedding (up to ``max_emb``) of the interacting logical qubits in the coupling graph."""
    import rustworkx as rx
    t0 = time.perf_counter()
    int_rows = sorted({u for u, v, _ in s.c2} | {v for u, v, _ in s.c2} | {u for u, v, _ in s.cz} | {v for u, v, _ in s.cz})
    pos = {r_: i for i, r_ in enumerate(int_rows)}
    if not int_rows:
        s.emb, s.int_rows, s.t_emb = np.zeros((1, 0), int), np.array([], int), time.perf_counter() - t0
        return s
    g = rx.PyGraph()
    g.add_nodes_from(int_rows)
    pairs = {(pos[u], pos[v]) for u, v, _ in s.c2} | {(pos[u], pos[v]) for u, v, _ in s.cz}
    g.add_edges_from_no_data(sorted(pairs))
    out = []
    for mp in rx.vf2_mapping(coupling_graph(c), g, subgraph=True, induced=False, id_order=False,
                             call_limit=call_limit):
        row = np.empty(len(int_rows), int)
        for p, i in mp.items():          # coupling-graph node -> interaction-graph node
            row[i] = p
        out.append(row)
        if len(out) >= max_emb:
            break
    if not out:                          # the reference layout is always an embedding
        out = [np.array([s.layout[s.qubits[r_]] for r_ in int_rows])]
    s.emb, s.int_rows, s.t_emb = np.stack(out), np.array(int_rows), time.perf_counter() - t0
    return s


# ----------------------------------------------------------------------------------------- per calibration
def _pairs(lst):
    if not lst:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    a = np.array(lst, float)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]


def place(s: Structure, c: Calib, score: str = "vf2", max_ties: int = 64) -> list[np.ndarray]:
    """Physical qubit of every row of ``s`` on calibration ``c``: one array per embedding tied for the best score
    (the compiler breaks ties by its search order, which a seed shuffles).  ``score``: "vf2" = VF2Layout's
    average-error score (what the compiler does), "true" = the highest true log F (an upper bound)."""
    k = len(s.qubits)
    pos = {r_: i for i, r_ in enumerate(s.int_rows)}
    E = s.emb                                          # (K, n_int)
    if E.shape[1]:
        if score == "vf2":
            sc = np.nansum(s.c1[s.int_rows] * c.s1[E], 1)
            u, v, w = _pairs(s.c2)
        else:
            sc = (s.r[s.int_rows] * np.nan_to_num(c.lr[E]) + s.m[s.int_rows] * np.nan_to_num(c.lm[E])).sum(1)
            u, v, w = _pairs(s.cz)
        if len(u):
            sc = sc + (w * (c.s2 if score == "vf2" else c.lcz)[E[:, [pos[x] for x in u]], E[:, [pos[x] for x in v]]]).sum(1)
        sc = np.where(np.isfinite(sc), sc, -np.inf)
        top = sc.max()
        tied = np.flatnonzero(sc >= top - 1e-12 * max(1.0, abs(top)))[:max_ties]
        bests = [E[i] for i in tied]
    else:
        bests = [np.zeros(0, int)]
    key = c.s1 if score == "vf2" else np.nan_to_num(c.lr, nan=-np.inf) + np.nan_to_num(c.lm, nan=-np.inf)
    weight = s.c1 if score == "vf2" else s.r + s.m
    order = [q for q in np.argsort(-np.nan_to_num(key, nan=-np.inf), kind="stable") if np.isfinite(key[q])]
    out = []
    for best in bests:
        phys = np.full(k, -1)
        phys[s.int_rows] = best
        # 1q-only qubits: most-used first, each on the best remaining qubit (VF2Layout's free-qubit mapping)
        free = [i for i in range(k) if phys[i] < 0]
        if free:
            taken = set(phys[phys >= 0].tolist())
            cand = [q for q in order if q not in taken]
            for i, q in zip(sorted(free, key=lambda i: -weight[i]), cand):
                phys[i] = q
        out.append(phys)
    return out


def vf2_score(s: Structure, c: Calib, phys: np.ndarray) -> float:
    """VF2Layout's average-error score (log of its fidelity estimate) of one placement of every row."""
    sc = float(np.nansum(s.c1 * c.s1[phys]))
    u, v, w = _pairs(s.c2)
    if len(u):
        sc += float((w * c.s2[phys[u], phys[v]]).sum())
    return sc


def evaluate(s: Structure, c: Calib, phys: np.ndarray) -> tuple[float, float]:
    """(log F, decay) of the structure on placement ``phys``: MQT's expected fidelity and ESP decay."""
    logf = float((s.r * np.nan_to_num(c.lr[phys])).sum() + (s.m * np.nan_to_num(c.lm[phys])).sum())
    u, v, w = _pairs(s.cz)
    if len(u):
        logf += float((w * c.lcz[phys[u], phys[v]]).sum())
    decay = -float((s.idle * c.inv_t[phys])[s.active].sum())
    return logf, decay


def predict(s: Structure, c: Calib, score: str = "vf2") -> dict:
    """F and ESP on calibration ``c``; with ties in the score, the mean over the tied placements (the compiler's
    pick among them is arbitrary).  ``vals`` keeps every tied (log F, decay) for ``best_of``."""
    t0 = time.perf_counter()
    ph = place(s, c, score)
    vals = [evaluate(s, c, p) for p in ph]
    F = float(np.mean([np.exp(a) for a, _ in vals]))
    E = float(np.mean([np.exp(a + b) for a, b in vals]))
    return {"F": F, "ESP": E, "logF": float(np.log(max(F, 1e-300))), "decay": float(np.log(max(E, 1e-300)) - np.log(max(F, 1e-300))),
            "n_ties": len(vals), "vals": vals,
            "layout": {int(s.qubits[i]): int(p) for i, p in enumerate(ph[0])}, "t": time.perf_counter() - t0}


def best_of(preds: list[dict]) -> dict:
    """Expected F (and ESP of the winner) of the best of several compiler seeds, each picking uniformly among
    its tied placements, as labels made of the best of K seeds."""
    sets = [p["vals"] for p in preds]
    size = int(np.prod([len(v) for v in sets]))
    rng = np.random.default_rng(0)
    if size <= 4096:
        import itertools
        combos = list(itertools.product(*sets))
    else:
        combos = [tuple(v[rng.integers(len(v))] for v in sets) for _ in range(4096)]
    F, E = [], []
    for cmb in combos:
        a, b = max(cmb, key=lambda x: x[0])
        F.append(np.exp(a)); E.append(np.exp(a + b))
    return {"F": float(np.mean(F)), "ESP": float(np.mean(E))}
