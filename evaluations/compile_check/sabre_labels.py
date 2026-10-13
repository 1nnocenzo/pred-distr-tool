#!/usr/bin/env python3
"""Stage-1 labels: the SWAPs Qiskit's SabreLayout inserts, attached to the gates of the model's circuit graph.

Why: on routed circuits the compiled duration is set by Sabre (optimisation after it changes the duration by 2–3% on
the dataset families, compile_check/results/routing_anatomy/report.md), but the model only ever saw the final counts
and duration, so it had to infer the routing (and does not transfer to unseen families).  The two-stage model learns
Sabre's output directly: per gate of the logical circuit, how many SWAPs Sabre puts on each of its wires just before
it.  The compiler runs here only to make labels, never at prediction time.

Per (circuit, device), Qiskit 2.3.1 preset level 2, seed 0, original calibration (as the labels):
  swaps       [[node, slot, n], ...]: n SWAPs on the wire of operand ``slot`` of model node ``node`` (graph node order:
              the circuit's instructions without barriers, fast_encoding.encode) since that wire's previous multi-qubit
              gate or measurement; ``n_swap`` total, ``n_swap_anc`` SWAPs with an empty physical qubit
  aligned     every routed multi-qubit gate / measurement matched to a model node (wire order is kept by routing;
              logical SWAP gates removed by ElidePermutations relabel the wires; 3-qubit gates may match several
              routed gates)
  t_node      [[node, start, end], ...] µs: each matched multi-qubit gate / measurement in the replayed routed schedule
              (stage-1 target: the routed schedule itself, which per-wire SWAP counts alone do not give: a SWAP
              waits for both wires); ``busy`` per logical wire (gates + SWAPs), for the idle part of the ESP
  durations   µs: ``logical`` = ASAP of the model's nodes with the fixed decomposition table (what v21t schedules),
              ``wire`` = the same with each node's labelled SWAPs as a delay on their wires (the stage-1 formulation),
              ``replay`` = Sabre's routed circuit itself scheduled with the same table (exact SWAP placement),
              ``naive`` = Qiskit ASAP of the routed circuit translated without optimisation, ``final`` = the label's
              compilation; plus final CZ / r and the routing-free flag
    python sabre_labels.py run --task-id 0 --n-tasks 48 [--sample 0] --out results/sabre_labels
    python sabre_labels.py report --out results/sabre_labels
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from duration_anatomy import DEVICES, EXCLUDE  # noqa: E402

SYN = EVAL / "synthetic_routing/circuits/synth"
TAB = json.loads((EVAL / "esp/results/decomp_table.json").read_text())
T1, TM, TCZ = 0.042, 15.0, 0.130                    # gsv3.devices.GATE_US (µs)
SWAP_R = 4
T_SWAP = 3 * TCZ + SWAP_R / 2 * T1


def gate_time(name: str, n_q: int) -> float:
    if name == "measure":
        return TM
    t = TAB.get(name)
    if t is None:
        return TCZ * (n_q - 1) * 2 + T1 if n_q > 1 else T1
    return t["r"] * T1 / max(t.get("qubits", n_q), 1) + t["cz"] * TCZ


def schedule(nodes, n_wires, delays=None) -> float:
    """ASAP on logical wires; ``delays[(node, slot)]`` SWAPs before the node on that wire."""
    avail = [0.0] * n_wires
    for i, (name, qs) in enumerate(nodes):
        if delays:
            for k, q in enumerate(qs):
                avail[q] += delays.get((i, k), 0) * T_SWAP
        s = max(avail[q] for q in qs) + gate_time(name, len(qs))
        for q in qs:
            avail[q] = s
    return max(avail) if avail else 0.0


def label_case(nodes, n_log, routed, init_layout, elide: bool):
    """Walk Sabre's routed circuit (physical qubits) and attach its SWAPs to the model's nodes."""
    # model nodes per wire, in the frame of the routed circuit's virtual qubits (logical SWAPs elided -> relabel)
    perm = list(range(n_log))
    per_wire = defaultdict(list)                     # dag wire -> [(node, slot, set of dag wires, name)]
    for i, (name, qs) in enumerate(nodes):
        if elide and name == "swap" and len(qs) == 2:
            a, b = qs
            perm[a], perm[b] = perm[b], perm[a]
            continue
        ws = [perm[q] for q in qs]
        for k, w in enumerate(ws):
            per_wire[w].append((i, k, frozenset(ws), name))
    at = {p: v for v, p in init_layout.items()}      # physical -> virtual wire
    ptr = defaultdict(int)
    used = defaultdict(int)                          # matches of the node at ptr[w]
    pending = defaultdict(int)
    labels: dict = {}
    n_swap = n_anc = 0
    aligned = True
    avail = defaultdict(float)
    busy = defaultdict(float)                       # per virtual wire: gate + SWAP time
    t_node: dict = {}                               # node -> [start, end] in the routed schedule
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
                    busy[w] += T_SWAP
            n_swap += 1
            if wa is None or wb is None:
                n_anc += 1
            for w in (wa, wb):
                if w is not None:
                    pending[w] += 1
            at[a], at[b] = wb, wa
            if at[a] is None:
                del at[a]
            if at[b] is None:
                del at[b]
            continue
        t_g = gate_time(name, len(ps))
        s = max(avail[p] for p in ps) + t_g
        for p in ps:
            avail[p] = s
        ws = [at.get(p) for p in ps]
        for w in ws:
            if w is not None:
                busy[w] += t_g
        if any(w is None for w in ws):
            continue
        if len(ws) < 2 and name != "measure":
            continue                                 # 1-qubit gates are re-synthesised: not aligned
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
    trailing = sum(pending.values())
    return labels, n_swap, n_anc, aligned, trailing, max(avail.values()) if avail else 0.0, t_node, busy


def names_list(args):
    from compile_check import DATA
    real = sorted(str(p.relative_to(DATA).with_suffix("")) for p in DATA.rglob("*.qasm"))
    real = [n for n in real if not n.split("/")[0].startswith(EXCLUDE)]
    syn = [f"synth/{p.stem}" for p in sorted(SYN.glob("*.qasm"))]
    out = [(n, DATA / f"{n}.qasm") for n in real] + [(n, SYN / f"{n.split('/')[1]}.qasm") for n in syn]
    if args.sample:
        rng = random.Random(0)
        fam = defaultdict(list)
        for x in out:
            fam[x[0].split("/")[0]].append(x)
        out = [y for v in fam.values() for y in rng.sample(v, min(args.sample, len(v)))]
    return [x for x in out if zlib.crc32(x[0].encode()) % args.n_tasks == args.task_id]


def run(args) -> None:
    from qiskit import qasm3, transpile
    from qiskit.converters import dag_to_circuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
    from gsv3 import fast_encoding as fe
    from v8b_variants import variants
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    items = names_list(args)
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t_start = time.time()
    with out_file.open("a") as fh:
        for ci, (name, path) in enumerate(items):
            if name in done:
                continue
            rec = {"circuit": name, "rows": []}
            try:
                qc = qasm3.loads(path.read_text())
                qb = fe.to_basis(qc)
                nodes = [(inst.operation.name, [qb.find_bit(q).index for q in inst.qubits])
                         for inst in qb.data if inst.operation.name != "barrier"]
                n_log = qb.num_qubits
                rec.update(n_nodes=len(nodes), nq=n_log, logical=schedule(nodes, n_log))
                for d in DEVICES:
                    be = base[d]
                    pm = generate_preset_pass_manager(optimization_level=2, target=be.target, seed_transpiler=0)
                    elide = True                     # level 2 runs ElidePermutations in the init stage
                    st = {}

                    def cb(**kw):
                        ps_ = kw["property_set"]
                        if ps_["VF2Layout_stop_reason"] is not None:
                            st["vf2"] = ps_["VF2Layout_stop_reason"]
                        if type(kw["pass_"]).__name__ == "SabreLayout":
                            st["routed"] = dag_to_circuit(kw["dag"])
                            st["vbits"] = ps_["layout"].get_virtual_bits()
                    tc = pm.run(qb, callback=cb)
                    row = {"device": d, "routing_free": st.get("vf2") == VF2LayoutStopReason.SOLUTION_FOUND}
                    sc = transpile(tc, target=be.target, scheduling_method="asap", optimization_level=0,
                                   initial_layout=None, routing_method=None, layout_method=None)
                    ops = tc.count_ops()
                    row.update(final=float(sc.estimate_duration(target=be.target)) * 1e6,
                               cz=int(ops.get("cz", 0)), r=int(ops.get("r", 0)))
                    if not row["routing_free"] and "routed" in st:
                        rc = st["routed"]
                        # virtual wire index of each logical qubit: the circuit's own qubits keep their index
                        idx = {q: i for i, q in enumerate(qb.qubits)}
                        init = {idx[v]: p for v, p in st["vbits"].items() if v in idx}
                        labels, n_sw, n_anc, ok, trail, replay, t_node, busy = label_case(nodes, n_log, rc, init, elide)
                        nv = transpile(rc, target=be.target, optimization_level=0, layout_method="trivial", routing_method="none")
                        nsc = transpile(nv, target=be.target, scheduling_method="asap", optimization_level=0,
                                        initial_layout=None, routing_method=None, layout_method=None)
                        row.update(swaps=[[k[0], k[1], v] for k, v in sorted(labels.items())], n_swap=n_sw,
                                   n_swap_anc=n_anc, aligned=ok, trailing=trail, replay=replay,
                                   t_node=[[k, round(v[0], 3), round(v[1], 3)] for k, v in sorted(t_node.items())],
                                   busy=[round(busy.get(w, 0.0), 3) for w in range(n_log)],
                                   wire=schedule(nodes, n_log, labels),
                                   naive=float(nsc.estimate_duration(target=be.target)) * 1e6)
                    rec["rows"].append(row)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if ci % 50 == 0:
                print(ci, len(items), name, f"{time.time() - t_start:.0f}s", rec.get("error", ""), flush=True)


SPLIT_OF = {"qaoa": "qaoa", "vqe_su2": "variational", "vqe_real_amp": "variational", "vqe_two_local": "variational",
            "qnn": "variational", "synth": "synth"}


def report(args) -> None:
    recs = [json.loads(l) for f in sorted(args.out.glob("task*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    rows = [{**r, "fam": c["circuit"].split("/")[0], "logical": c["logical"], "nq": c["nq"]} for c in recs for r in c.get("rows", [])]
    rt = [r for r in rows if not r["routing_free"] and "swaps" in r]
    groups = {"qaoa": lambda r: r["fam"] == "qaoa",
              "variational (vqe_*, qnn)": lambda r: SPLIT_OF.get(r["fam"]) == "variational",
              "other families": lambda r: r["fam"] not in SPLIT_OF,
              "synth": lambda r: r["fam"] == "synth"}

    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = np.abs(a / b - 1)
        return f"{np.median(e):.3f} [{np.percentile(e, 90):.3f}] bias {np.median(a / b):.2f}" if len(e) else "–"
    L = ["# Sabre labels: stage-1 targets and how much of the duration they explain", "",
         f"{len(recs)} circuits ({sum('error' in c for c in recs)} failed) × 3 devices; {len(rt)} routed cases. "
         "Duration error vs the final compiled duration: median [p90], bias = median ratio.", "",
         "| | " + " | ".join(groups) + " |", "| --- |" + " --- |" * len(groups)]

    def line(lab, fn):
        L.append(f"| {lab} | " + " | ".join(fn([r for r in rt if g(r)]) for g in groups.values()) + " |")
    line("routed cases", lambda g: str(len(g)))
    line("aligned", lambda g: f"{np.mean([r['aligned'] for r in g]):.3f}" if g else "–")
    line("SWAPs median [p90]", lambda g: f"{np.median([r['n_swap'] for r in g]):.0f} [{np.percentile([r['n_swap'] for r in g], 90):.0f}]" if g else "–")
    line("SWAPs with an ancilla (share)", lambda g: f"{sum(r['n_swap_anc'] for r in g) / max(sum(r['n_swap'] for r in g), 1):.3f}" if g else "–")
    line("SWAPs left unattached (share)", lambda g: f"{sum(r['trailing'] for r in g) / max(sum(r['n_swap'] for r in g), 1):.3f}" if g else "–")
    line("logical schedule (no SWAPs)", lambda g: rel([r["logical"] for r in g], [r["final"] for r in g]))
    line("**wire schedule with Sabre's SWAPs (stage-1 oracle)**", lambda g: rel([r["wire"] for r in g], [r["final"] for r in g]))
    line("replay of the routed circuit (table times)", lambda g: rel([r["replay"] for r in g], [r["final"] for r in g]))
    line("Qiskit routed, not optimised", lambda g: rel([r["naive"] for r in g], [r["final"] for r in g]))
    line("wire schedule vs replay", lambda g: rel([r["wire"] for r in g], [r["replay"] for r in g]))
    try:      # are the compilations of the basis circuit those of the dataset labels?
        import torch
        lab = torch.load(HERE / "results/esp_labels.pt", weights_only=False)
        rel_ = [abs(r["final"] / (float(lab[c["circuit"]]["duration"][DEVICES.index(r["device"])]) * 1e6) - 1)
                for c in recs if c["circuit"] in lab for r in c.get("rows", [])]
        L += ["", f"Final duration vs the dataset label (esp_labels.pt): {len(rel_)} cases, "
              f"within 0.1%: {np.mean(np.array(rel_) < 1e-3):.3f}, median |Δ| {np.median(rel_):.4f}"]
    except Exception as e:  # noqa: BLE001
        L += ["", f"(label check skipped: {e!r})"]
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0, help="circuits per family (0 = all)")
    ap.add_argument("--out", type=Path, default=HERE / "results/sabre_labels")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
