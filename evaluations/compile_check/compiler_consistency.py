#!/usr/bin/env python3
"""How consistent is the compiler with itself?  The ceiling for a learned placement.

For a sample of control-split test circuits, compile on the original devices and on local v8d
calibration variants with K compiler seeds each, keeping every seed's initial layout and fidelity.
Then measure, between two seeds of the same compiler:
  * layout agreement     share of logical qubits on the same physical qubit; occupancy correlation
  * reaction agreement   corr(Δu seed a, Δu seed b), Δu = change of physical-qubit occupancy from the
                         original calibration to a variant (the model reaches 0.30–0.34 against the
                         best-of-3 compiler layout, placement_probe.py)
  * label agreement      corr of Δlog F between seeds (and best-of-2 vs a third seed)
    python compiler_consistency.py run --task-id T --n-tasks N --out DIR
    python compiler_consistency.py report --out DIR
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(HERE))
EVAL = HERE.parent
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
VARIANTS = (11, 19, 50, 55, 56, 61, 65, 72, 73, 82)    # local kinds, as placement_probe.py
K = 5


def sample(n=300):
    test = sorted(json.loads((EVAL / "generalization_v9/results/v8kd40/random_control/random_seed5/predictions.json").read_text()))
    test = [t for t in test if not t.startswith(("grover/", "iqpe/", "dynamic_qft/"))]
    rng = np.random.default_rng(1)
    return sorted(rng.choice(test, n, replace=False).tolist())


def run(args):
    from qiskit import qasm3
    import v8b_variants as V
    from compile_check import DATA, compile_once
    from compile_layouts import fidelity
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    base = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    backends = {f"{d}/orig": be for d, be in base.items()}
    for i, (d, be) in enumerate(base.items()):
        for j in VARIANTS:
            backends[f"{d}/aug{j}"] = V.make_aug(be, j, i)
    node = {k: {q: i for i, q in enumerate(getattr(be, "active_qubits", None) or range(be.num_qubits))} for k, be in backends.items()}
    names = [n for n in sample() if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"task{args.task_id:03d}.jsonl").open("w") as fh:
        for name in names:
            qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
            rec = {"circuit": name, "F": {}, "layout": {}}
            for key, be in backends.items():
                fs, lays = [], []
                for s in range(K):
                    tc, _ = compile_once(qc, be, 2, s)
                    fs.append(fidelity(tc, be.target))
                    lays.append([node[key].get(p, -1) for p in tc.layout.initial_index_layout(filter_ancillas=True)])
                rec["F"][key], rec["layout"][key] = fs, lays
            fh.write(json.dumps(rec) + "\n"); fh.flush()


def occ(lay, n_phys):
    u = np.zeros(n_phys)
    for p in lay:
        if p >= 0:
            u[p] = 1.0
    return u


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan


def report(args):
    recs = [json.loads(l) for f in sorted(args.out.glob("task*.jsonl")) for l in f.read_text().splitlines()]
    n_phys = {"EQE1_Top": 26, "EQE1_Bottom": 27, "QExa20": 20}
    same, occ_c, react, react_best, lab_c, lab_b3 = [], [], [], [], [], []
    for r in recs:
        for d in DEVICES:
            P = n_phys[d]
            lo = r["layout"][f"{d}/orig"]
            for a, b in itertools.combinations(range(K), 2):
                la, lb = np.array(lo[a]), np.array(lo[b])
                same.append(float((la == lb).mean())); occ_c.append(corr(occ(la, P), occ(lb, P)))
            best_o = int(np.argmax(r["F"][f"{d}/orig"]))
            for j in VARIANTS:
                lv = r["layout"][f"{d}/aug{j}"]
                du = [occ(lv[s], P) - occ(lo[s], P) for s in range(K)]
                for a, b in itertools.combinations(range(K), 2):
                    if np.abs(du[a]).sum() > 0 and np.abs(du[b]).sum() > 0:
                        react.append(corr(du[a], du[b]))
                # best-of-3 reaction (as the v8d labels: seeds 0-2) vs an independent seed pair (3, 4)
                bv = int(np.argmax(r["F"][f"{d}/aug{j}"][:3])); bo = int(np.argmax(r["F"][f"{d}/orig"][:3]))
                d_best = occ(lv[bv], P) - occ(lo[bo], P)
                for s in (3, 4):
                    if np.abs(d_best).sum() > 0 and np.abs(du[s]).sum() > 0:
                        react_best.append(corr(d_best, du[s]))
                fo, fv = np.array(r["F"][f"{d}/orig"]), np.array(r["F"][f"{d}/aug{j}"])
                if fo.min() > 0.01 and fv.min() > 0.01:
                    lab_c.append((np.log(fv[0] / fo[0]), np.log(fv[1] / fo[1])))
                    lab_b3.append((np.log(fv[:3].max() / fo[:3].max()), np.log(fv[3:].max() / fo[3:].max())))
    lc, lb = np.array(lab_c), np.array(lab_b3)
    L = ["# Compiler self-consistency (the ceiling for a learned placement)", "",
         f"{len(recs)} control test circuits × 3 devices × (original + {len(VARIANTS)} local variants) × {K} compiler seeds.", "",
         "| quantity | between two compiler seeds | model (placement_probe.py) |", "| --- | --- | --- |",
         f"| logical qubits on the same physical qubit, original calibration | {np.nanmean(same):.0%} | – |",
         f"| occupancy correlation, original calibration | {np.nanmean(occ_c):.2f} | – |",
         f"| reaction: corr(Δu seed a, Δu seed b), original → variant | {np.nanmean(react):.2f} | 0.30–0.34 (vs best of 3) |",
         f"| reaction: best of seeds 0–2 vs seed 3 or 4 | {np.nanmean(react_best):.2f} | 0.30–0.34 |",
         f"| Δlog F, seed 0 vs seed 1 | {corr(lc[:, 0], lc[:, 1]):.2f} | – |",
         f"| Δlog F, best of seeds 0–2 vs best of 3–4 | {corr(lb[:, 0], lb[:, 1]):.2f} | local tracking 0.57–0.61 |", "",
         "If the compiler's own reaction agreement is close to the model's, the placement is already near what is "
         "predictable; if it is much higher, the placement can still learn."]
    (args.out / "report.md").write_text("\n".join(L))
    out2 = EVAL / "diagnostics_v4v5/results/report_compiler_consistency.md"
    out2.write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
