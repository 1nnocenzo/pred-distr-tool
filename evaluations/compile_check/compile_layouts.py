#!/usr/bin/env python3
"""Compiler layouts for the whole dataset: the targets of the v6 placement loss.

Every circuit of the label file is compiled on each device exactly as the labels
were (Qiskit preset pass manager, level 2) with a fixed ``seed_transpiler`` and the
initial layout is stored as **device-graph node indices** (node ``i`` = backend qubit
``active_qubits[i]``, the order of the predictor's device graphs), together with the
fidelity of that compilation and whether every instruction is native and on a coupler.
Equivalence is not re-checked here (``compile_check.py`` did, on 12,961 compilations).

Output: one JSON line per circuit, ``{"circuit", "layout": {dev: [node of logical i]},
"fidelity": {dev: F}, "valid": {dev: bool}}``; ``merge`` writes ``compiler_layouts.pt``
``{circuit: LongTensor (n_qubits, n_devices)}`` in the predictor's device order.
With ``--save-qpy`` the compiled circuits are kept too (one QPY file per circuit, the
three devices in ``DEVICES`` order): the input of a later ESP / runtime computation.

    python compile_layouts.py run --task-id 0 --n-tasks 16 [--families grover | --exclude grover] --out DIR
    python compile_layouts.py merge --out DIR
    python compile_layouts.py counts --out DIR [--workers 32]   # v8 count targets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")   # = genstudy.data.DEVICE_NAMES


def _fidelity_recursive(qc, target, qubit_map=None):
    """Copy of ``_fidelity_recursive`` in ``compileCircuits/testComp-compilation-time.py``
    (the labels' fallback for dynamic circuits): MQT's per-gate formula, recursing into
    control-flow blocks and taking the worst-case branch."""
    if qubit_map is None:
        qubit_map = list(range(qc.num_qubits))
    res = 1.0
    for inst in qc.data:
        op, qargs = inst.operation, inst.qubits
        if op.name == "barrier":
            continue
        phys = [qubit_map[qc.find_bit(q).index] for q in qargs]
        blocks = getattr(op, "blocks", None)
        if blocks:
            res *= min(_fidelity_recursive(b, target, phys) for b in blocks)
            continue
        if len(phys) == 1:
            res *= 1 - target[op.name][(phys[0],)].error
        else:
            res *= 1 - target[op.name][tuple(phys[:2])].error
    return res


def fidelity(tc, target) -> float:
    """Expected fidelity exactly as the labels: MQT, else the recursive fallback."""
    from mqt.predictor.reward import expected_fidelity
    try:
        return float(expected_fidelity(tc, target))
    except KeyError:
        return float(round(_fidelity_recursive(tc, target), 10))


def run(args) -> None:
    from qiskit import qasm3, qpy
    from compile_check import DATA, LABELS, compile_once, is_valid
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend

    backends = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(),
                "QExa20": QExa20Backend()}
    node = {d: {q: i for i, q in enumerate(getattr(b, "active_qubits", None) or range(b.num_qubits))}
            for d, b in backends.items()}
    names = sorted(json.loads(LABELS.read_text()))
    fams = set(args.families.split(",")) if args.families else None
    excl = set(args.exclude.split(",")) if args.exclude else set()
    names = [n for n in names if (fams is None or n.split("/")[0] in fams)
             and n.split("/")[0] not in excl]
    # Deterministic sharding that spreads each family over the tasks.
    names = [n for n in names if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.families or 'all'}{'-' + args.exclude if args.exclude else ''}".replace(",", "+")
    out_file = args.out / f"{tag}_task{args.task_id:03d}.jsonl"
    done = set()
    if out_file.is_file():
        done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()}
    with out_file.open("a") as fh:
        for i, name in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name, "layout": {}, "fidelity": {}, "valid": {}}
            try:
                qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
                compiled = []
                for dev, be in backends.items():
                    tc, _ = compile_once(qc, be, 2, args.seed)
                    tc.name = f"{name}@{dev}"
                    compiled.append(tc)
                    lay = tc.layout.initial_index_layout(filter_ancillas=True)
                    rec["layout"][dev] = [node[dev].get(p, -1) for p in lay]
                    rec["valid"][dev] = is_valid(tc, be.target)
                    rec["fidelity"][dev] = fidelity(tc, be.target)
                if args.save_qpy:
                    # Compiled circuits (device order = DEVICES), for ESP / runtime later.
                    qpy_path = args.out / "qpy" / f"{name}.qpy"
                    qpy_path.parent.mkdir(parents=True, exist_ok=True)
                    with qpy_path.open("wb") as qf:
                        qpy.dump(compiled, qf)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(names)} {name}", flush=True)


def fill_fidelity(args) -> None:
    """Recompute missing (null) fidelities from the stored QPY files -> fidelity_fill.jsonl."""
    from qiskit import qpy
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    targets = {"EQE1_Top": EQE1TopBackend().target, "EQE1_Bottom": EQE1BottomBackend().target,
               "QExa20": QExa20Backend().target}
    todo = [r["circuit"] for f in sorted(args.out.glob("*_task*.jsonl"))
            for r in map(json.loads, f.read_text().splitlines())
            if "error" not in r and None in r["fidelity"].values()]
    with (args.out / "fidelity_fill.jsonl").open("w") as fh:
        for name in todo:
            with (args.out / "qpy" / f"{name}.qpy").open("rb") as qf:
                circuits = qpy.load(qf)
            fh.write(json.dumps({"circuit": name, "fidelity": {
                d: fidelity(c, targets[d]) for d, c in zip(DEVICES, circuits)}}) + "\n")
    print(f"{len(todo)} circuits filled from QPY -> {args.out / 'fidelity_fill.jsonl'}")


KINDS = {"r": 0, "measure": 1, "cz": 2}   # order of devices.phys: 1q, readout, 2q


def _counts(qc):
    """(r, measure, cz) of a compiled circuit; control flow: the costliest branch."""
    import numpy as np
    c = np.zeros(3)
    for inst in qc.data:
        op = inst.operation
        blocks = getattr(op, "blocks", None)
        if blocks:
            c += max((_counts(b) for b in blocks), key=lambda v: v.sum())
        elif op.name in KINDS:
            c[KINDS[op.name]] += 1
    return c


def _counts_file(path):
    from qiskit import qpy
    with open(path, "rb") as fh:
        return [_counts(c).tolist() for c in qpy.load(fh)]


def counts(args) -> None:
    """True native-operation counts per circuit and device from qpy/ -> compiler_counts.pt."""
    import torch
    from multiprocessing import Pool
    skip = set(args.exclude.split(",")) if args.exclude else set()
    files = sorted(f for f in (args.out / "qpy").rglob("*.qpy") if f.parent.name not in skip)
    names = [str(f.relative_to(args.out / "qpy").with_suffix("")) for f in files]
    with Pool(args.workers) as pool:
        res = pool.map(_counts_file, files, chunksize=16)
    out = {n: torch.tensor(r, dtype=torch.float32) for n, r in zip(names, res)}   # (D, 3)
    torch.save(out, args.out / "compiler_counts.pt")
    print(f"{len(out)} circuits -> {args.out / 'compiler_counts.pt'}")


def merge(args) -> None:
    import torch
    out, bad = {}, 0
    for f in sorted(args.out.glob("*_task*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" in r or len(r["layout"]) != len(DEVICES):
                bad += 1
                continue
            cols = [r["layout"][d] for d in DEVICES]
            if any(-1 in c for c in cols):
                bad += 1
                continue
            out[r["circuit"]] = torch.tensor(cols, dtype=torch.long).t().contiguous()
    torch.save(out, args.out / "compiler_layouts.pt")
    print(f"{len(out)} circuits with layouts, {bad} without -> {args.out / 'compiler_layouts.pt'}")

    # Fidelity of the very compilation stored in qpy/ (nulls filled by fill-fidelity).
    fill_path = args.out / "fidelity_fill.jsonl"
    fill = {r["circuit"]: r["fidelity"] for r in map(json.loads, fill_path.read_text().splitlines())
            } if fill_path.is_file() else {}
    fid, missing = {}, 0
    for f in sorted(args.out.glob("*_task*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if "error" in r:
                continue
            v = {d: (r["fidelity"].get(d) if r["fidelity"].get(d) is not None
                     else fill.get(r["circuit"], {}).get(d)) for d in DEVICES}
            missing += sum(x is None for x in v.values())
            fid[r["circuit"]] = v
    (args.out / "fidelity_seed0.json").write_text(json.dumps(fid))
    print(f"{len(fid)} circuits -> {args.out / 'fidelity_seed0.json'} ({missing} values still missing)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "merge", "fill-fidelity", "counts"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--families", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-qpy", action="store_true",
                    help="also store the compiled circuits (<out>/qpy/<family>/<name>.qpy)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"run": run, "merge": merge, "fill-fidelity": fill_fidelity, "counts": counts}[args.cmd](args)


if __name__ == "__main__":
    main()
