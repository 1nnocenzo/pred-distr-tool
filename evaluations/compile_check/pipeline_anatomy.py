#!/usr/bin/env python3
"""Which steps of the level-2 compilation depend on the calibration?

Qiskit 2.3.1, preset level 2 (as the labels): VF2Layout (perfect embedding, picked by an average-error
score) else SabreLayout + SabreSwap (topology and seed only), then VF2PostLayout (re-embeds the routed
circuit on the lowest-error isomorphic subgraph), then optimisation (CZ-only basis).  Hypothesis: for a
fixed compiler seed the compiled structure does not depend on the errors; a calibration change only
moves its embedding, which VF2PostLayout computes.  Per (circuit, device, calibration, seed):
  routing_free      VF2Layout found a perfect layout
  same_structure    compiled usage graph (r / measure per qubit, CZ per coupler) isomorphic to the one
                    compiled on the original calibration with the same seed
  F, F_emul         true F on the variant; F of the original compilation re-embedded by VF2PostLayout
                    with the variant's errors (no re-routing), and its time vs the full compilation
    python pipeline_anatomy.py run --task-id 0 --n-tasks 4 --out results/pipeline_anatomy
    python pipeline_anatomy.py report --out results/pipeline_anatomy
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EVAL = HERE.parent
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
JS = (80, 84, 85, 89)          # local v8d variants: jitter, shuffle, bad qubits, jitter + bad qubits
SEEDS = (0, 1, 2)
EXCLUDE = ("grover", "iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code")


def totals(tc) -> dict:
    ops = tc.count_ops()
    return {k: int(ops.get(k, 0)) for k in ("cz", "r", "measure")}


def usage_graph(tc):
    import rustworkx as rx
    n = tc.num_qubits
    r, m, cz = [0] * n, [0] * n, {}
    for inst in tc.data:
        name = inst.operation.name
        qs = [tc.find_bit(q).index for q in inst.qubits]
        if name == "cz":
            k = (min(qs), max(qs)); cz[k] = cz.get(k, 0) + 1
        elif name == "r":
            r[qs[0]] += 1
        elif name == "measure":
            m[qs[0]] += 1
    used = sorted({q for q in range(n) if r[q] or m[q]} | {q for k in cz for q in k})
    g = rx.PyGraph()
    idx = {q: g.add_node((r[q], m[q])) for q in used}
    for (a, b), c in cz.items():
        g.add_edge(idx[a], idx[b], c)
    return g


def same_structure(g1, g2) -> bool:
    import rustworkx as rx
    if g1.num_nodes() != g2.num_nodes() or g1.num_edges() != g2.num_edges():
        return False
    return rx.is_isomorphic(g1, g2, node_matcher=lambda a, b: a == b, edge_matcher=lambda a, b: a == b,
                            call_limit=1_000_000)


def reembed(tc, target):
    """The compiled circuit ``tc`` moved by VF2PostLayout to the best-scoring embedding under ``target``."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import VF2PostLayout
    from qiskit.transpiler.preset_passmanagers.common import get_vf2_limits
    call_limit, max_trials = get_vf2_limits(2, None, None)
    pm = PassManager([VF2PostLayout(target, seed=-1, call_limit=call_limit, max_trials=max_trials,
                                    strict_direction=False)])
    t0 = time.perf_counter()
    pm.run(tc)
    post = pm.property_set["post_layout"]
    if post is None:                     # no better embedding: the circuit stays where it is
        return tc, time.perf_counter() - t0, str(pm.property_set["VF2PostLayout_stop_reason"])
    phys = {tc.find_bit(q).index: post[q] for q in tc.qubits if q in post.get_virtual_bits()}
    new = QuantumCircuit(tc.num_qubits, tc.num_clbits)
    for inst in tc.data:
        if inst.operation.name == "barrier":
            continue
        qs = [phys.get(tc.find_bit(q).index, tc.find_bit(q).index) for q in inst.qubits]
        if inst.operation.name == "cz" and tuple(qs) not in target["cz"]:
            qs = qs[::-1]                # CZ is symmetric; the target lists one direction (GateDirection's job)
        new.append(inst.operation, [new.qubits[q] for q in qs], [new.clbits[tc.find_bit(c).index] for c in inst.clbits])
    return new, time.perf_counter() - t0, "moved"


def permute(tc, P: dict, target):
    """``tc`` with physical qubit p moved to P[p] (others unchanged); CZ turned to the target's direction."""
    from qiskit import QuantumCircuit
    new = QuantumCircuit(tc.num_qubits, tc.num_clbits)
    for inst in tc.data:
        if inst.operation.name == "barrier":
            continue
        qs = [P.get(tc.find_bit(q).index, tc.find_bit(q).index) for q in inst.qubits]
        if inst.operation.name == "cz" and tuple(qs) not in target["cz"]:
            qs = qs[::-1]
        new.append(inst.operation, [new.qubits[q] for q in qs], [new.clbits[tc.find_bit(c).index] for c in inst.clbits])
    return new


def by_index(layout, circ) -> dict:
    """Layout (Qubit of ``circ`` -> physical int) as {qubit index: physical}."""
    if layout is None:
        return {}
    return {circ.find_bit(q).index: p for q, p in layout.get_virtual_bits().items() if q in set(circ.qubits)}


def stage_emulation(ref, target):
    """Re-run only the compiler's error-aware choice, on the circuit it actually scores, with ``target``'s
    errors, and move the original compilation accordingly.  Routing-free: VF2Layout on the logical circuit
    (after the init stage).  Routed: VF2PostLayout on the routed circuit (before optimisation)."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import VF2Layout, VF2PostLayout
    from qiskit.transpiler.preset_passmanagers.common import get_vf2_limits
    t0 = time.perf_counter()
    if ref["rf"]:
        pm = PassManager([VF2Layout(seed=-1, call_limit=(5_000_000, 10_000), target=target)])
        pm.run(ref["virt"])
        new = by_index(pm.property_set["layout"], ref["virt"])
        P = {ref["lay"][v]: new[v] for v in ref["lay"] if v in new}
    else:
        call_limit, max_trials = get_vf2_limits(2, None, None)
        pm = PassManager([VF2PostLayout(target, seed=-1, call_limit=call_limit, max_trials=max_trials, strict_direction=False)])
        pm.run(ref["routed"])
        new = by_index(pm.property_set["post_layout"], ref["routed"])
        old = ref["post"]
        n = ref["routed"].num_qubits
        P = {old.get(r, r): new.get(r, r) for r in range(n) if r in old or r in new}
    return permute(ref["tc"], P, target), time.perf_counter() - t0


def run(args) -> None:
    from qiskit import qasm3
    from compile_check import DATA, compile_once
    from compile_layouts import fidelity
    from v8b_variants import make_aug, variants
    from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
    from qiskit.converters import dag_to_circuit
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    vs = {}
    for i, d in enumerate(DEVICES):
        vs[(d, "orig")] = base[d]
        for j in JS:
            vs[(d, j)] = make_aug(base[d], j, i)
    test = sorted(json.loads((EVAL / "generalization_v9/results/v8kd40/random_control/random_seed5/predictions.json").read_text()))
    test = [n for n in test if not n.split("/")[0].startswith(EXCLUDE)]
    names = sorted(random.Random(0).sample(test, args.n_circuits))
    names = [n for n in names if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    with out_file.open("a") as fh:
        for ci, name in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name, "rows": []}
            try:
                qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
                for d in DEVICES:
                    for s in SEEDS:
                        ref = None
                        for key in ["orig", *JS]:
                            be = vs[(d, key)]
                            stop = {}

                            def cb(**kw):
                                ps = kw["property_set"]
                                name_ = kw["pass_"].name()
                                if name_ == "VF2Layout":          # analysis pass: kw["dag"] is the logical circuit it scored
                                    stop["virt"] = dag_to_circuit(kw["dag"])
                                    stop["lay"] = by_index(ps["layout"], stop["virt"]) if ps["layout"] is not None else {}
                                elif name_ == "VF2PostLayout":    # kw["dag"] is the routed circuit before optimisation
                                    stop["routed"] = dag_to_circuit(kw["dag"])
                                    stop["post_l"] = by_index(ps["post_layout"], stop["routed"])
                                if ps["VF2Layout_stop_reason"] is not None:
                                    stop["vf2"] = ps["VF2Layout_stop_reason"]
                                if ps["VF2PostLayout_stop_reason"] is not None:
                                    stop["post"] = str(ps["VF2PostLayout_stop_reason"])
                            from qiskit.transpiler import generate_preset_pass_manager
                            pm = generate_preset_pass_manager(optimization_level=2, target=be.target, seed_transpiler=s)
                            t0 = time.perf_counter()
                            tc = pm.run(qc, callback=cb)
                            t_full = time.perf_counter() - t0
                            f = float(fidelity(tc, be.target))
                            g = usage_graph(tc)
                            row = {"dev": d, "cal": key, "seed": s, "F": f, "t_full": t_full, "tot": totals(tc),
                                   "post": stop.get("post"),
                                   "routing_free": stop.get("vf2") == VF2LayoutStopReason.SOLUTION_FOUND}
                            if key == "orig":
                                ref = (tc, g)
                                stage_ref = {"rf": row["routing_free"], "tc": tc, "virt": stop.get("virt"),
                                             "lay": stop.get("lay", {}), "routed": stop.get("routed"),
                                             "post": stop.get("post_l", {})}
                            else:
                                row["same_structure"] = same_structure(ref[1], g)
                                emb, t_emb, how = reembed(ref[0], be.target)
                                try:
                                    st_c, t_st = stage_emulation(stage_ref, be.target)
                                    row.update(F_stage=float(fidelity(st_c, be.target)), t_stage=t_st)
                                except Exception as e:
                                    row["stage_error"] = f"{type(e).__name__}: {e}"[:200]
                                row.update(F_emul=float(fidelity(emb, be.target)), t_emul=t_emb, emul=how,
                                           F_stay=float(fidelity(ref[0], be.target)))
                            rec["rows"].append(row)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {ci + 1}/{len(names)} {name}", flush=True)


def report(args) -> None:
    rows, err, seen = [], 0, set()
    for f in sorted(args.out.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if r["circuit"] in seen:           # a circuit written twice (resubmitted task): keep the first
                continue
            seen.add(r["circuit"])
            err += "error" in r
            for x in r.get("rows", []):
                x["circuit"] = r["circuit"]; rows.append(x)
    var = [x for x in rows if x["cal"] != "orig"]
    orig = {(x["circuit"], x["dev"], x["seed"]): x for x in rows if x["cal"] == "orig"}
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
    L = ["# Level-2 pipeline anatomy: what a calibration change moves", "",
         f"{len({x['circuit'] for x in rows})} control test circuits ({err} failed) × 3 devices × orig + local v8d "
         f"variants {list(JS)} × seeds {list(SEEDS)}; Qiskit 2.3.1 preset level 2 (as the labels).", "",
         "| | all | routing-free | routed |", "| --- | --- | --- | --- |"]
    groups = {"all": var, "routing-free": [x for x in var if orig[(x["circuit"], x["dev"], x["seed"])]["routing_free"]],
              "routed": [x for x in var if not orig[(x["circuit"], x["dev"], x["seed"])]["routing_free"]]}
    def cell(fn):
        return " | ".join(fn(g) for g in groups.values())
    L.append(f"| cases (circuit × device × variant × seed) | {cell(lambda g: str(len(g)))} |")
    def share(fn):
        return lambda g: "{:.1%}".format(np.mean([fn(x) for x in g])) if g else "–"
    same = share(lambda x: x["same_structure"])
    rf_kept = share(lambda x: x["routing_free"] == orig[(x["circuit"], x["dev"], x["seed"])]["routing_free"])
    L.append(f"| same compiled structure as on the original calibration (up to qubit relabelling) | {cell(same)} |")
    L.append(f"| routing-free status unchanged | {cell(rf_kept)} |")
    tot_kept = share(lambda x: x["tot"] == orig[(x["circuit"], x["dev"], x["seed"])]["tot"])
    L.append(f"| same total CZ / r / measure counts | {cell(tot_kept)} |")
    dcz = lambda g: "{:+.2f}".format(np.mean([x["tot"]["cz"] - orig[(x["circuit"], x["dev"], x["seed"])]["tot"]["cz"] for x in g])) if g else "–"
    L.append(f"| mean change of the CZ count | {cell(dcz)} |")
    for lab, key in (("compiler's error-aware step re-run on the circuit it scores (VF2Layout / VF2PostLayout)", "F_stage"),
                     ("final circuit re-embedded by VF2PostLayout", "F_emul"), ("old compilation kept as is", "F_stay")):
        def c(g, key=key):
            g = [x for x in g if key in x and x["F"] > 0.01 and orig[(x["circuit"], x["dev"], x["seed"])]["F"] > 0.01]
            if len(g) < 3:
                return "–"
            dt = [np.log(x["F"]) - np.log(orig[(x["circuit"], x["dev"], x["seed"])]["F"]) for x in g]
            de = [np.log(max(x[key], 1e-12)) - np.log(orig[(x["circuit"], x["dev"], x["seed"])]["F"]) for x in g]
            exact = np.mean([abs(x[key] - x["F"]) < 1e-6 * max(x["F"], 1e-12) + 1e-12 for x in g])
            return f"corr {corr(de, dt):.3f}, exact {exact:.0%}"
        L.append(f"| Δlog F, same seed: {lab} vs recompiled | {cell(c)} |")
    tf = np.array([x["t_full"] for x in var]); te = np.array([x["t_emul"] for x in var])
    L += ["", f"Time per case, median: full compilation {np.median(tf) * 1e3:.1f} ms, VF2PostLayout re-embedding "
          f"{np.median(te) * 1e3:.1f} ms (mean {tf.mean() * 1e3:.1f} vs {te.mean() * 1e3:.1f} ms)."]
    # best of 3 seeds (the labels): does the winning seed change, and how well does the re-embedding track it?
    best = {}
    for x in rows:
        k = (x["circuit"], x["dev"], x["cal"])
        best.setdefault(k, []).append(x)
    sw, dt, de = [], [], []
    for (c, d, cal), xs in best.items():
        if cal == "orig" or len(xs) != len(SEEDS):
            continue
        o = best.get((c, d, "orig"))
        if not o or len(o) != len(SEEDS):
            continue
        so = max(o, key=lambda x: x["F"]); sv = max(xs, key=lambda x: x["F"])
        sw.append(so["seed"] != sv["seed"])
        if so["F"] > 0.01 and sv["F"] > 0.01:
            dt.append(np.log(sv["F"]) - np.log(so["F"]))
            de.append(np.log(max(max(x.get("F_stage", x["F_emul"]) for x in xs), 1e-12)) - np.log(so["F"]))
    L += ["", f"Best of {len(SEEDS)} seeds (as the labels): the winning seed changes in {np.mean(sw):.0%} of the "
          f"(circuit, device, variant) cases; Δlog F of the best stage-emulated original compilations vs the best "
          f"recompilation: corr {corr(de, dt):.3f} ({len(dt)} cases). Model local tracking on v8d variants: 0.74; "
          f"compiler noise ceiling (best of 3 vs best of 2): 0.94."]
    (args.out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--n-circuits", type=int, default=200)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "results/pipeline_anatomy")
    a = ap.parse_args()
    run(a) if a.mode == "run" else report(a)
