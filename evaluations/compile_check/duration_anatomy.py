#!/usr/bin/env python3
"""How exactly can the duration of a routed circuit be known, and at what cost?  (for a scheduler)

Per sampled test circuit and device (the labels' duration: ASAP schedule of the level-2 compilation, seed 0, original
calibration, as esp_labels.py):
  seed floor     duration of the level-2 compilation with seeds 1 and 2 vs seed 0: the spread of the compiler itself,
                 below which no predictor of "the" duration can go
  calibration    seed 0 on local calibration variants (v8d 80 jitter, 89 jitter + bad qubits): does a duration
                 computed once stay right after a calibration change
  level 1        seed 0 at optimization level 1 (cheaper): duration error vs the label and compile time
  time           compile time of every compilation (the cost of compiling instead of predicting)
Circuits: random test circuits of LOFO qaoa, LOGO variational and the control split (generalization_v10, seed 5).
    python duration_anatomy.py run --task-id 0 --n-tasks 8 --out results/duration_anatomy
    python duration_anatomy.py report --out results/duration_anatomy
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
JS = (80, 89)
EXCLUDE = ("grover", "iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code")
SPLITS = {"LOFO qaoa": "leave_one_family_out/qaoa", "LOGO variational": "leave_one_group_out/variational",
          "control": "random_control/random_seed5"}


def duration(tc, target) -> float:
    from qiskit import transpile
    sched = transpile(tc, target=target, scheduling_method="asap", optimization_level=0,
                      initial_layout=None, routing_method=None, layout_method=None)
    return float(sched.estimate_duration(target=target)) * 1e6     # µs


def run(args) -> None:
    from qiskit import qasm3
    from qiskit.transpiler import generate_preset_pass_manager
    from compile_check import DATA
    from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
    from v8b_variants import make_aug, variants
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    for be in base.values():                       # T1/T2 in the target, as esp_labels.py (needed by the scheduler)
        t = be.target
        if t.qubit_properties is None:
            qp = be.qubit_properties
            t.qubit_properties = list(qp) if not callable(qp) else [qp(q) for q in range(t.num_qubits)]
    vs = {(d, "orig"): base[d] for d in DEVICES}
    for i, d in enumerate(DEVICES):
        for j in JS:
            vs[(d, j)] = make_aug(base[d], j, i)
    names = []
    for split, sd in SPLITS.items():
        test = sorted(json.loads((EVAL / f"generalization_v10/results/v8kdfi10/{sd}/predictions.json").read_text()))
        test = [n for n in test if not n.split("/")[0].startswith(EXCLUDE)]
        names += [(split, n) for n in sorted(random.Random(0).sample(test, min(args.n_per_split, len(test))))]
    names = [x for x in names if zlib.crc32(x[1].encode()) % args.n_tasks == args.task_id]
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    with out_file.open("a") as fh:
        for ci, (split, name) in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name, "split": split, "rows": []}
            try:
                qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
                for d in DEVICES:
                    for key, seed, lvl in [("orig", 0, 2), ("orig", 1, 2), ("orig", 2, 2), *((j, 0, 2) for j in JS), ("orig", 0, 1)]:
                        be = vs[(d, key)]
                        pm = generate_preset_pass_manager(optimization_level=lvl, target=be.target, seed_transpiler=seed)
                        stop = {}

                        def cb(**kw):
                            r_ = kw["property_set"]["VF2Layout_stop_reason"]
                            if r_ is not None:
                                stop["vf2"] = r_
                        t0 = time.perf_counter()
                        tc = pm.run(qc, callback=cb)
                        t_c = time.perf_counter() - t0
                        ops = tc.count_ops()
                        rec["rows"].append({"dev": d, "cal": key, "seed": seed, "level": lvl, "t_compile": t_c,
                                            "dur": duration(tc, vs[(d, "orig")].target), "cz": int(ops.get("cz", 0)),
                                            "routing_free": stop.get("vf2") == VF2LayoutStopReason.SOLUTION_FOUND})
            except Exception as e:                              # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {ci + 1}/{len(names)} {name}", flush=True)


def report(args) -> None:
    import torch
    lab = torch.load(HERE / "results/esp_labels.pt", weights_only=False)
    recs = [json.loads(l) for f in sorted(args.out.glob("task*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    rows = []
    for r in recs:
        if "error" in r or r["circuit"] not in lab:
            continue
        true = lab[r["circuit"]]["duration"].numpy() * 1e6
        by = {(x["dev"], x["cal"], x["seed"], x["level"]): x for x in r["rows"]}
        for di, d in enumerate(DEVICES):
            ref = by[(d, "orig", 0, 2)]
            rows.append({"split": r["split"], "routed": not ref["routing_free"],
                         "label_check": abs(ref["dur"] / true[di] - 1),
                         "seed": max(abs(by[(d, "orig", s, 2)]["dur"] / ref["dur"] - 1) for s in (1, 2)),
                         "seed_mean": np.mean([abs(by[(d, "orig", s, 2)]["dur"] / ref["dur"] - 1) for s in (1, 2)]),
                         "calib": max(abs(by[(d, j, 0, 2)]["dur"] / ref["dur"] - 1) for j in JS),
                         "l1": abs(by[(d, "orig", 0, 1)]["dur"] / ref["dur"] - 1),
                         "t2": ref["t_compile"] * 1e3, "t1": by[(d, "orig", 0, 1)]["t_compile"] * 1e3})
    L = ["# Duration of the compiled circuit: compiler noise floor, calibration stability, cost (for a scheduler)", "",
         f"{len({r['circuit'] for r in recs})} test circuits × 3 devices; Qiskit preset passes as the labels; duration = ASAP "
         "schedule on the original calibration (µs). Relative differences |a/b − 1|, median [p90].", ""]
    q = lambda x: f"{np.median(x):.3f} [{np.percentile(x, 90):.3f}]"
    L += ["| split | group | cases | label reproduced (seed 0) | other compiler seed (mean of 2) | worst of 2 seeds | "
          "seed 0 after a local calibration change (worst of 2) | level 1 vs level 2 | compile ms level 2, median [p90] | level 1 ms |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for split in [*SPLITS, "all"]:
        for grp in ("routing-free", "routed", "all"):
            s = [x for x in rows if (split == "all" or x["split"] == split)
                 and (grp == "all" or x["routed"] == (grp == "routed"))]
            if len(s) < 10:
                continue
            g = lambda k: np.array([x[k] for x in s])
            L.append(f"| {split} | {grp} | {len(s)} | {q(g('label_check'))} | {q(g('seed_mean'))} | {q(g('seed'))} | "
                     f"{q(g('calib'))} | {q(g('l1'))} | {np.median(g('t2')):.1f} [{np.percentile(g('t2'), 90):.1f}] | "
                     f"{np.median(g('t1')):.1f} [{np.percentile(g('t1'), 90):.1f}] |")
    (args.out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n-per-split", type=int, default=100)
    ap.add_argument("--out", type=Path, default=HERE / "results/duration_anatomy")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
