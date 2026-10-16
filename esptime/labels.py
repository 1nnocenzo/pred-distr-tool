#!/usr/bin/env python3
"""All training targets from ONE Qiskit compilation per (circuit, calibration, device).

The compiled circuit is the basis circuit the model encodes (encoding.to_basis), Qiskit 2.3.1 preset level 2,
seed_transpiler 0.  Per (circuit, calibration, device):
  logF, decay   MQT expected fidelity (log, unrounded) and log ESP − log F (MQT estimated_success_probability)
  dur           ASAP duration of the compiled circuit, µs
  layout        device-graph node of every logical qubit (initial layout, ancillas dropped)
  counts        native r / measure / cz;  rf: Qiskit's VF2Layout found a perfect layout (no routing)
  busy          [[node, µs], ...] gate time of every active physical qubit (MQT's exec time: the ESP's idle is
                duration − busy on these qubits)
Route targets (original calibration only; Sabre's routing does not depend on the errors): Sabre's routed
circuit walked against the model's gate nodes -> SWAPs per (node, operand slot), each matched node's [start, end] in
the routed schedule (table gate times), the replayed routed duration and whether every gate was matched.

    python -m esptime.labels run --task-id 0 --n-tasks 32 [--calibs orig,aug0-9] [--source mqt|synth]
    python -m esptime.labels merge          # -> work/labels.pt + work/devices.json
    python -m esptime.labels decomp         # regenerate esptime/decomp_table.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

from .config import CIRCUITS_ZIP, DECOMP_TABLE, DEVICES, EXCLUDE, T1, TCZ, TM, T_SWAP, WORK, decomp_table

OUT = WORK / "labels"
TAB = None


# ------------------------------------------------------------------------------------------------ circuits
def circuit_sources(source: str, per_family: int = 0) -> list[tuple[str, str]]:
    """[(name, qasm text loader key)]: ``mqt`` = data/circuits.zip, ``synth`` = work/circuits/synth/*.qasm."""
    if source == "synth":
        return [(f"synth/{p.stem}", str(p)) for p in sorted((WORK / "circuits/synth").glob("*.qasm"))]
    with zipfile.ZipFile(CIRCUITS_ZIP) as z:
        files = sorted(n for n in z.namelist() if n.endswith(".qasm"))
    out = []
    for f in files:
        parts = f.split("/")
        name = f"{parts[-2]}/{Path(parts[-1]).stem}"
        if name.split("/")[0] not in EXCLUDE:
            out.append((name, f))
    if per_family:                       # samples: k circuits of every family, deterministic in the name
        by = defaultdict(list)
        for x in sorted(out, key=lambda x: zlib.crc32(b"sample" + x[0].encode())):
            by[x[0].split("/")[0]].append(x)
        out = sorted(y for v in by.values() for y in v[:per_family])
    return out


def read_qasm(key: str, z: zipfile.ZipFile | None):
    from qiskit import qasm3
    text = z.read(key).decode() if z is not None and not key.startswith("/") else Path(key).read_text()
    return qasm3.loads(text)


def parse_calibs(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        if part.startswith("aug") and "-" in part:
            a, b = part[3:].split("-")
            out += [f"aug{j}" for j in range(int(a), int(b) + 1)]
        elif part:
            out.append(part)
    return out


# ------------------------------------------------------------------------------------------------ Sabre walk
def gate_time(name: str, n_q: int) -> float:
    if name == "measure":
        return TM
    t = TAB.get(name)
    if t is None:
        return TCZ * (n_q - 1) * 2 + T1 if n_q > 1 else T1
    return t["r"] * T1 / max(t.get("qubits", n_q), 1) + t["cz"] * TCZ


def walk_routed(nodes, n_log, routed, init_layout):
    """Attach the SWAPs of Sabre's routed circuit (physical qubits) to the model's nodes.  ElidePermutations (level
    2) removes logical swap gates, which relabels the wires; 3-qubit gates may match several routed gates."""
    perm = list(range(n_log))
    per_wire = defaultdict(list)                     # wire -> [(node, slot, wires of the node, name)]
    for i, (name, qs) in enumerate(nodes):
        if name == "swap" and len(qs) == 2:
            a, b = qs
            perm[a], perm[b] = perm[b], perm[a]
            continue
        ws = [perm[q] for q in qs]
        for k, w in enumerate(ws):
            per_wire[w].append((i, k, frozenset(ws), name))
    at = {p: v for v, p in init_layout.items()}     # physical -> wire
    ptr, used, pending = defaultdict(int), defaultdict(int), defaultdict(int)
    labels: dict = {}
    aligned = True
    avail = defaultdict(float)
    t_node: dict = {}
    for inst in routed.data:
        name = inst.operation.name
        ps = [routed.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "swap":
            a, b = ps
            s = max(avail[a], avail[b]) + T_SWAP
            avail[a] = avail[b] = s
            wa, wb = at.get(a), at.get(b)
            for w in (wa, wb):
                if w is not None:
                    pending[w] += 1
            at[a], at[b] = wb, wa
            for p in (a, b):
                if at[p] is None:
                    del at[p]
            continue
        t_g = gate_time(name, len(ps))
        s = max(avail[p] for p in ps) + t_g
        for p in ps:
            avail[p] = s
        ws = [at.get(p) for p in ps]
        if any(w is None for w in ws) or (len(ws) < 2 and name != "measure"):
            continue                                 # 1-qubit gates are re-synthesised: not matched
        S = frozenset(ws)
        for w in ws:
            lst = per_wire[w]
            j = ptr[w]
            if j < len(lst) and used[w] and not (len(lst[j][2]) > 2 and S <= lst[j][2]):
                j += 1; used[w] = 0
            while j < len(lst) and not (S <= lst[j][2] and (lst[j][3] == "measure") == (name == "measure")
                                        and (len(lst[j][2]) > 1 or name == "measure")):
                j += 1; used[w] = 0
            if j >= len(lst):
                aligned = False
                continue
            ptr[w] = j; used[w] += 1
            node, slot = lst[j][0], lst[j][1]
            tn = t_node.setdefault(node, [s - t_g, s])
            tn[0], tn[1] = min(tn[0], s - t_g), max(tn[1], s)
            if pending[w]:
                labels[(node, slot)] = labels.get((node, slot), 0) + pending[w]
                pending[w] = 0
    return labels, aligned, max(avail.values()) if avail else 0.0, t_node


# ------------------------------------------------------------------------------------------------ one case
def compile_case(qb, nodes, be, route: bool) -> dict:
    from mqt.predictor.reward import estimated_success_probability, expected_fidelity
    from qiskit import transpile
    from qiskit.converters import dag_to_circuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
    from .devices import active_qubits
    target = be.target
    pm = generate_preset_pass_manager(optimization_level=2, target=target, seed_transpiler=0)
    st: dict = {}

    def cb(**kw):
        ps_ = kw["property_set"]
        if ps_["VF2Layout_stop_reason"] is not None:
            st["vf2"] = ps_["VF2Layout_stop_reason"]
        if route and type(kw["pass_"]).__name__ == "SabreLayout":
            st["routed"] = dag_to_circuit(kw["dag"])
            st["vbits"] = ps_["layout"].get_virtual_bits()
    tc = pm.run(qb, callback=cb)
    f = expected_fidelity(tc, target, precision=300)
    esp = estimated_success_probability(tc, target, precision=300)
    sc = transpile(tc, target=target, scheduling_method="asap", optimization_level=0,
                   initial_layout=None, routing_method=None, layout_method=None)
    ops = tc.count_ops()
    node = {q: i for i, q in enumerate(active_qubits(be))}
    busy = defaultdict(float)
    for inst in tc.data:
        if inst.operation.name in ("barrier", "id", "delay"):
            continue
        qs = tuple(tc.find_bit(q).index for q in inst.qubits)
        for q in qs:
            busy[q] += target[inst.operation.name][qs].duration * 1e6
    rec = {"logF": math.log(f) if f > 0 else None,
           "decay": (math.log(esp) - math.log(f)) if esp > 0 and f > 0 else None,
           "dur": float(sc.estimate_duration(target=target)) * 1e6,
           "layout": [node.get(p, -1) for p in tc.layout.initial_index_layout(filter_ancillas=True)],
           "counts": [int(sum(v for k, v in ops.items() if k not in ("cz", "measure", "barrier", "delay"))),
                      int(ops.get("measure", 0)), int(ops.get("cz", 0))],
           "rf": st.get("vf2") == VF2LayoutStopReason.SOLUTION_FOUND,
           "busy": [[node.get(q, -1), round(v, 4)] for q, v in sorted(busy.items())]}
    if route and not rec["rf"] and "routed" in st:
        idx = {q: i for i, q in enumerate(qb.qubits)}
        init = {idx[v]: p for v, p in st["vbits"].items() if v in idx}
        sw, ok, replay, t_node = walk_routed(nodes, qb.num_qubits, st["routed"], init)
        rec.update(swaps=[[k[0], k[1], v] for k, v in sorted(sw.items())], aligned=ok, replay=replay,
                   t_node=[[k, round(v[0], 4), round(v[1], 4)] for k, v in sorted(t_node.items())])
    return rec


# ------------------------------------------------------------------------------------------------ commands
def run(args) -> None:
    global TAB
    from .devices import backends
    from .encoding import to_basis
    TAB = decomp_table()
    calibs = parse_calibs(args.calibs)
    bes = {c: backends(c) for c in calibs}
    items = [x for x in circuit_sources(args.source, args.per_family)
             if zlib.crc32(x[0].encode()) % args.n_tasks == args.task_id]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.source}_{args.calibs.replace(',', '_')}_task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    z = zipfile.ZipFile(CIRCUITS_ZIP) if args.source == "mqt" else None
    t0 = time.time()
    with out_file.open("a") as fh:
        for i, (name, key) in enumerate(items):
            if name in done:
                continue
            rec = {"circuit": name, "calib": {}}
            try:
                qb = to_basis(read_qasm(key, z))
                if any(inst.operation.name == "if_else" for inst in qb.data):
                    raise ValueError("dynamic circuit")
                nodes = [(inst.operation.name, [qb.find_bit(q).index for q in inst.qubits]) for inst in qb.data]
                rec.update(n_nodes=len(nodes), nq=qb.num_qubits)
                for c in calibs:
                    rec["calib"][c] = {d: compile_case(qb, nodes, bes[c][d], route=c == "orig") for d in DEVICES}
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 50 == 0:
                print(f"[{time.time() - t0:6.0f} s] {i}/{len(items)} {name} {rec.get('error', '')}", flush=True)


def merge(args) -> None:
    import torch
    from .devices import backends, raw_graph
    out, calibs, err = {}, set(), 0
    for f in sorted(args.out.glob("*task*.jsonl")):
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if "error" in r:
                err += 1
                continue
            cur = out.setdefault(r["circuit"], {"n_nodes": r["n_nodes"], "nq": r["nq"], "calib": {}})
            cur["calib"].update(r["calib"])
            calibs |= set(r["calib"])
    WORK.mkdir(exist_ok=True)
    torch.save(out, WORK / "labels.pt")
    raws = {c: {d: raw_graph(be) for d, be in backends(c).items()} for c in sorted(calibs | {"orig"})}
    (WORK / "devices.json").write_text(json.dumps(raws))
    print(f"{len(out)} circuits ({err} failed), calibrations {sorted(calibs)} -> {WORK / 'labels.pt'}, devices.json")


def decomp(args) -> None:
    """Native r / cz count of every basis gate transpiled alone at level 2 on adjacent qubits of EQE1_Top."""
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import get_standard_gate_name_mapping
    from .devices import backends
    from .encoding import UNIQUE
    target = backends("orig")["EQE1_Top"].target
    nb = defaultdict(set)
    for x, y in target.build_coupling_map().get_edges():
        nb[x].add(y); nb[y].add(x)
    b = next(q for q in nb if len(nb[q]) >= 2)
    a, c = sorted(nb[b])[:2]
    mapping = get_standard_gate_name_mapping()
    rng = np.random.default_rng(0)
    tab = {}
    for name in UNIQUE:
        if name in ("measure", "if_else") or name not in mapping:
            continue
        g = mapping[name]
        if g.params:
            g = g.__class__(*[float(x) for x in rng.uniform(0.3, 2.8, len(g.params))])
        qc = QuantumCircuit(g.num_qubits)
        qc.append(g, list(range(g.num_qubits)))
        t = transpile(qc, target=target, optimization_level=2, initial_layout=[a, b, c][: g.num_qubits], seed_transpiler=0)
        ops = t.count_ops()
        tab[name] = {"r": int(sum(v for k, v in ops.items() if k not in ("cz", "measure", "barrier", "delay"))),
                     "cz": int(ops.get("cz", 0)), "qubits": g.num_qubits}
    DECOMP_TABLE.write_text(json.dumps(tab, indent=1))
    print(json.dumps(tab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "merge", "decomp"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--calibs", default="orig", help="orig and/or variants, e.g. orig,aug0-9")
    ap.add_argument("--source", choices=("mqt", "synth"), default="mqt")
    ap.add_argument("--per-family", type=int, default=0, help="smoke tests: first k circuits per family")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    {"run": run, "merge": merge, "decomp": decomp}[args.cmd](args)


if __name__ == "__main__":
    main()
