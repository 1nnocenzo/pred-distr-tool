#!/usr/bin/env python3
"""How does the level-2 compiler route, and what of it sets the duration and the ESP?  (first step toward emulating it)

Qiskit 2.3.1 preset level 2, seed 0, original calibration (as the labels).  For routed circuits SabreLayout picks the
initial layout and routes inside the layout stage (topology and seed only), VF2PostLayout then moves the routed
circuit to the lowest-error isomorphic embedding, translation and optimisation follow.  Per (circuit, device):
  sabre            SWAPs inserted by SabreLayout; its pass time; its trial settings
  naive            the routed circuit translated without optimisation: CZ / r counts and ASAP duration
  final            the label's compilation: CZ / r counts and ASAP duration (what the predictor must hit)
  idle             idle time per active physical qubit in the final schedule (Σ duration − busy), split into qubits
                   that hold a logical qubit and qubits only SWAPs pass through (routing ancillas): the ESP decay
  times            VF2Layout, SabreLayout, VF2PostLayout and the whole compilation (ms)
Circuits: random test circuits of LOFO qaoa, LOGO variational, the control split, and synthetic routing circuits.
    python routing_anatomy.py run --task-id 0 --n-tasks 8 --out results/routing_anatomy
    python routing_anatomy.py report --out results/routing_anatomy
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
from duration_anatomy import DEVICES, EXCLUDE, SPLITS  # noqa: E402

SYN = EVAL / "synthetic_routing/circuits"


def scheduled(tc, target):
    from qiskit import transpile
    return transpile(tc, target=target, scheduling_method="asap", optimization_level=0,
                     initial_layout=None, routing_method=None, layout_method=None)


def busy_idle(sc, target):
    """Per active physical qubit: busy time (µs) of its operations; total duration."""
    dt = target.dt or 1.0
    dur = float(sc.estimate_duration(target=target)) * 1e6
    busy = defaultdict(float)
    for inst in sc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qs = tuple(sc.find_bit(q).index for q in inst.qubits)
        props = target[name].get(qs) if name in target.operation_names else None
        d = (props.duration if props is not None and props.duration is not None else 0.0) * 1e6
        for q in qs:
            busy[q] += d
    return dur, dict(busy)


def counts(tc):
    ops = tc.count_ops()
    return {"cz": int(ops.get("cz", 0)), "r": int(ops.get("r", 0)), "measure": int(ops.get("measure", 0))}


def run(args) -> None:
    from qiskit import qasm3, transpile
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
    from compile_check import DATA
    from v8b_variants import variants
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    for be in base.values():
        t = be.target
        if t.qubit_properties is None:
            qp = be.qubit_properties
            t.qubit_properties = list(qp) if not callable(qp) else [qp(q) for q in range(t.num_qubits)]
    names = []
    for split, sd in SPLITS.items():
        test = sorted(json.loads((EVAL / f"generalization_v10/results/v8kdfi10/{sd}/predictions.json").read_text()))
        test = [n for n in test if not n.split("/")[0].startswith(EXCLUDE)]
        names += [(split, n, DATA / f"{n}.qasm") for n in sorted(random.Random(0).sample(test, min(args.n_per_split, len(test))))]
    syn = sorted(p for p in (SYN / "synth").glob("*.qasm"))
    names += [("synth", f"synth/{p.stem}", p) for p in sorted(random.Random(0).sample(syn, min(args.n_per_split, len(syn))))]
    names = [x for x in names if zlib.crc32(x[1].encode()) % args.n_tasks == args.task_id]
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    with out_file.open("a") as fh:
        for ci, (split, name, path) in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name, "split": split, "rows": []}
            try:
                qc = qasm3.loads(path.read_text())
                rec["nq"] = qc.num_qubits
                for d in DEVICES:
                    be = base[d]
                    pm = generate_preset_pass_manager(optimization_level=2, target=be.target, seed_transpiler=0)
                    st = {"t": defaultdict(float)}

                    def cb(**kw):
                        nm = type(kw["pass_"]).__name__
                        st["t"][nm] += kw["time"] * 1e3
                        ps = kw["property_set"]
                        if ps["VF2Layout_stop_reason"] is not None:
                            st["vf2"] = ps["VF2Layout_stop_reason"]
                        if nm == "SabreLayout":
                            from qiskit.converters import dag_to_circuit
                            st["routed"] = dag_to_circuit(kw["dag"])
                            p = kw["pass_"]
                            st["sabre"] = {k: getattr(p, k, None) for k in ("max_iterations", "swap_trials", "layout_trials", "skip_routing")}
                    t0 = time.perf_counter()
                    tc = pm.run(qc, callback=cb)
                    t_all = (time.perf_counter() - t0) * 1e3
                    row = {"device": d, "routing_free": st.get("vf2") == VF2LayoutStopReason.SOLUTION_FOUND,
                           "t_total": t_all, "t_pass": {k: v for k, v in st["t"].items() if v > 0.05}}
                    sc = scheduled(tc, be.target)
                    dur, busy = busy_idle(sc, be.target)
                    row.update(final=counts(tc), dur_final=dur)
                    if not row["routing_free"] and "routed" in st:
                        rc = st["routed"]
                        row["swaps"] = int(rc.count_ops().get("swap", 0))
                        row["sabre"] = st["sabre"]
                        nv = transpile(rc, target=be.target, optimization_level=0, layout_method="trivial", routing_method="none")
                        row.update(naive=counts(nv), dur_naive=float(scheduled(nv, be.target).estimate_duration(target=be.target)) * 1e6)
                        # physical qubits that hold a logical qubit at the start vs routing ancillas
                        lay = tc.layout.initial_index_layout(filter_ancillas=True) if tc.layout is not None else []
                        hold = set(int(p) for p in lay)
                        idle_l = sum(dur - b for q, b in busy.items() if q in hold)
                        idle_a = sum(dur - b for q, b in busy.items() if q not in hold)
                        row.update(idle_logical=idle_l, idle_ancilla=idle_a, n_active=len(busy),
                                   n_ancilla=sum(q not in hold for q in busy))
                    rec["rows"].append(row)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(ci, name, len(rec["rows"]), rec.get("error", ""), flush=True)


def report(args) -> None:
    recs = [json.loads(l) for f in sorted(args.out.glob("task*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    rows = [{**r, "split": c["split"], "nq": c.get("nq", 0), "circuit": c["circuit"]} for c in recs for r in c["rows"]]
    rt = [r for r in rows if not r["routing_free"] and "swaps" in r]
    splits = ["LOFO qaoa", "LOGO variational", "control", "synth"]

    def q(x, f="{:.2f}"):
        x = np.asarray(x, float)
        return (f + " [" + f + "]").format(np.median(x), np.percentile(x, 90)) if len(x) else "–"

    L = ["# Routing anatomy: what the compiler's routing does to CZ count, duration and idle time", "",
         f"Qiskit 2.3.1 preset level 2, seed 0, original calibration (as the labels); {len(recs)} circuits × 3 devices, "
         f"{len(rt)} routed cases. Median [p90].", ""]
    s0 = next((r["sabre"] for r in rt if r.get("sabre")), {})
    L += [f"SabreLayout settings: {s0}", "",
          "| | " + " | ".join(splits) + " |", "| --- |" + " --- |" * len(splits)]

    def line(lab, fn, f="{:.2f}"):
        cells = []
        for s in splits:
            g = [r for r in rt if r["split"] == s]
            cells.append(q([fn(r) for r in g], f) if g else "–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("| routed cases | " + " | ".join(str(sum(r["split"] == s for r in rt)) for s in splits) + " |")
    line("qubits", lambda r: r["nq"], "{:.0f}")
    line("SWAPs inserted by Sabre", lambda r: r["swaps"], "{:.0f}")
    line("SWAPs per logical qubit", lambda r: r["swaps"] / max(r["nq"], 1))
    line("share of naive CZ from SWAPs", lambda r: 3 * r["swaps"] / max(r["naive"]["cz"], 1))
    line("final CZ / naive CZ (optimisation)", lambda r: r["final"]["cz"] / max(r["naive"]["cz"], 1))
    line("final r / naive r", lambda r: r["final"]["r"] / max(r["naive"]["r"], 1))
    line("final duration / naive duration", lambda r: r["dur_final"] / max(r["dur_naive"], 1e-9))
    line("final duration µs", lambda r: r["dur_final"], "{:.1f}")
    line("idle share of qubit-time (Σ idle / (n_active · duration))",
         lambda r: (r["idle_logical"] + r["idle_ancilla"]) / max(r["n_active"] * r["dur_final"], 1e-9))
    line("share of idle on routing ancillas", lambda r: r["idle_ancilla"] / max(r["idle_logical"] + r["idle_ancilla"], 1e-9))
    line("routing ancillas", lambda r: r["n_ancilla"], "{:.0f}")
    line("compile ms (total)", lambda r: r["t_total"], "{:.1f}")
    line("SabreLayout ms", lambda r: r["t_pass"].get("SabreLayout", 0.0), "{:.1f}")
    line("VF2PostLayout ms", lambda r: r["t_pass"].get("VF2PostLayout", 0.0), "{:.1f}")
    # which passes take the time, routed cases, all splits
    tot = defaultdict(list)
    for r in rt:
        for k, v in r["t_pass"].items():
            tot[k].append(v)
    top = sorted(tot.items(), key=lambda kv: -np.mean(kv[1]))[:10]
    L += ["", "Mean time per pass, routed cases (ms): " + ", ".join(f"{k} {np.mean(v):.1f}" for k, v in top), ""]
    err = sum("error" in c for c in recs)
    if err:
        L.append(f"Failed circuits: {err}")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n-per-split", type=int, default=120)
    ap.add_argument("--out", type=Path, default=HERE / "results/routing_anatomy")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
