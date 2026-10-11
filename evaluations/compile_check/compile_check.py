#!/usr/bin/env python3
"""Recompile circuits several times and check every result: fidelity, validity, equivalence.

For a stratified sample of circuits of one family (default ``qaoa``) and each of the
three devices, the circuit is compiled exactly as the labels were made
(``generate_preset_pass_manager(optimization_level=L, target=...)`` + MQT
``expected_fidelity``), but with several ``seed_transpiler`` values, optionally at
level 3, and optionally with a given ``initial_layout`` (e.g. the placement proposed
by a model, ``--layouts``).  Every compiled circuit is checked:

* **valid**: every instruction is supported by the target on exactly those physical
  qubits (native gate set + coupling map);
* **equivalent**: the compiled circuit implements the original one.  Both are run as
  statevectors on random product input states (``--equiv-trials``), the compiled one
  on the physical qubits it actually uses, with the input prepared on the initial
  layout and the output read on the final layout (routing permutation); idle
  ancillas must stay in |0>.  Overlap |<expected|actual>|^2 >= 1 - 1e-6 passes
  (global phase ignored).

A compilation counts only if it is both valid and equivalent.  Output: one JSON line
per (circuit, device, variant).

Run from ``~/compileCircuits`` (``createDevice.py`` and the label JSON live there),
on a compute node::

    python compile_check.py --task-id $SLURM_ARRAY_TASK_ID --n-tasks 16 --out out_dir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit.library import UGate
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(CC))
from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend  # noqa: E402
from mqt.predictor.reward import expected_fidelity  # noqa: E402

LABELS = CC / "expected_fidelity_results_benchmark_30k.json"
DATA = CC / "benchmark_dataset_30k"
TOL = 1e-6
SIM = AerSimulator(method="statevector")


def stratified_sample(family: str, per_size: int, seed: int) -> list[str]:
    names = sorted(p.stem for p in (DATA / family).glob("*.qasm"))
    by_q: dict[int, list[str]] = {}
    for n in names:
        by_q.setdefault(int(re.search(r"_q(\d+)_", n).group(1)), []).append(n)
    rng = np.random.default_rng(seed)
    out = []
    for q in sorted(by_q):
        pick = rng.choice(len(by_q[q]), size=min(per_size, len(by_q[q])), replace=False)
        out += [f"{family}/{by_q[q][i]}" for i in sorted(pick)]
    return out


def is_valid(tc: QuantumCircuit, target) -> bool:
    for inst in tc.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        qargs = tuple(tc.find_bit(q).index for q in inst.qubits)
        if not target.instruction_supported(name, qargs):
            return False
    return True


def _statevector(qc: QuantumCircuit) -> np.ndarray:
    qc = qc.copy()
    qc.save_statevector()
    res = SIM.run(transpile(qc, SIM, optimization_level=0)).result()
    return np.asarray(res.get_statevector())


def equivalence(orig: QuantumCircuit, tc: QuantumCircuit, trials: int, rng) -> float:
    """Worst overlap |<expected|actual>|^2 over random product inputs (1.0 = equivalent)."""
    n = orig.num_qubits
    init = tc.layout.initial_index_layout(filter_ancillas=True)
    final = tc.layout.final_index_layout(filter_ancillas=True)
    used = set(init) | set(final)
    for inst in tc.data:
        used |= {tc.find_bit(q).index for q in inst.qubits}
    phys = sorted(used)
    loc = {p: i for i, p in enumerate(phys)}
    body = tc.remove_final_measurements(inplace=False)
    worst = 1.0
    for _ in range(trials):
        angles = rng.uniform(0, 2 * np.pi, size=(n, 3))
        actual = QuantumCircuit(len(phys))
        expected = QuantumCircuit(len(phys))
        for v in range(n):
            actual.append(UGate(*angles[v]), [loc[init[v]]])
        # the compiled body, restricted to the physical qubits it uses
        mapped = QuantumCircuit(len(phys))
        for inst in body.data:
            mapped.append(inst.operation, [loc[body.find_bit(q).index] for q in inst.qubits])
        actual.compose(mapped, inplace=True)
        logical = QuantumCircuit(n)
        for v in range(n):
            logical.append(UGate(*angles[v]), [v])
        logical.compose(orig.remove_final_measurements(inplace=False), inplace=True)
        expected.compose(logical, qubits=[loc[final[v]] for v in range(n)], inplace=True)
        a, e = _statevector(actual), _statevector(expected)
        worst = min(worst, float(abs(np.vdot(e, a)) ** 2))
    return worst


def compile_once(qc, backend, level, seed, layout=None):
    pm = generate_preset_pass_manager(optimization_level=level, target=backend.target,
                                      seed_transpiler=seed, initial_layout=layout)
    t0 = time.perf_counter()
    tc = pm.run(qc)
    return tc, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", default="qaoa")
    ap.add_argument("--per-size", type=int, default=16, help="circuits per qubit count")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--seeds-l2", type=int, default=5)
    ap.add_argument("--seeds-l3", type=int, default=3)
    ap.add_argument("--layouts", type=Path, default=None,
                    help="JSON {variant: {circuit: {device: [phys qubit of logical i]}}}")
    ap.add_argument("--equiv-trials", type=int, default=2)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N circuits of this task")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labels = json.loads(LABELS.read_text())
    backends = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(),
                "QExa20": QExa20Backend()}
    layouts = json.loads(args.layouts.read_text()) if args.layouts else {}
    names = stratified_sample(args.family, args.per_size, args.sample_seed)[args.task_id::args.n_tasks]
    if args.limit:
        names = names[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"task{args.task_id:03d}.jsonl"
    done = set()
    if out_file.is_file():
        done = {(r["circuit"], r["device"], r["variant"])
                for r in map(json.loads, out_file.read_text().splitlines())}
    rng = np.random.default_rng(1000 + args.task_id)

    variants = [(f"L2_s{s}", 2, s, None) for s in range(args.seeds_l2)]
    variants += [(f"L3_s{s}", 3, s, None) for s in range(args.seeds_l3)]
    with out_file.open("a") as fh:
        for name in names:
            qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
            for dev, be in backends.items():
                extra = [(f"{v}_L2_s{s}", 2, s, layouts[v][name][dev])
                         for v in layouts if name in layouts[v] and dev in layouts[v][name]
                         for s in range(args.seeds_l2)]
                for tag, level, seed, layout in variants + extra:
                    if (name, dev, tag) in done:
                        continue
                    rec = {"circuit": name, "device": dev, "variant": tag,
                           "label": labels[name][0][dev]["fidelity"]}
                    try:
                        tc, ct = compile_once(qc, be, level, seed, layout)
                        rec.update(fidelity=float(expected_fidelity(tc, be.target)),
                                   compile_time=ct, valid=is_valid(tc, be.target),
                                   n_2q=sum(1 for i in tc.data if i.operation.num_qubits == 2),
                                   layout=tc.layout.initial_index_layout(filter_ancillas=True))
                        t0 = time.perf_counter()
                        rec["overlap"] = equivalence(qc, tc, args.equiv_trials, rng)
                        rec["equiv_time"] = time.perf_counter() - t0
                        rec["equivalent"] = rec["overlap"] >= 1 - TOL
                    except Exception as e:  # keep going, record the failure
                        rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {name} done", flush=True)


if __name__ == "__main__":
    main()
