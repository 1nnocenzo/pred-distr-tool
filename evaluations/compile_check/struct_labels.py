#!/usr/bin/env python3
"""Per-qubit structure labels of the stored compilations (``layouts_all/qpy``, seed 0, original calibration).

Per (circuit, device), over the active physical qubits of the compiled circuit (1q gate 42 ns, CZ 130 ns):
  tmax  largest gate time of a qubit in µs (measurements excluded)
  mmax  largest number of measurements on a qubit
  omax  largest number of operations on a qubit (1q, CZ, measurements)
  active number of qubits with at least one operation
the quantities the analytic ESP head (``*_espa``) builds from the count head; ``--struct-lambda`` supervises them.
    python struct_labels.py run --task-id 0 --n-tasks 8
    python struct_labels.py merge          # -> results/struct_labels.pt {circuit: FloatTensor(3 devices, 4)}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
QPY = HERE / "results/layouts_all/qpy"
OUT = HERE / "results/struct_labels"
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
DYNAMIC = {"iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code", "grover"}
KEYS = ("tmax", "mmax", "omax", "active")
T1Q, TCZ = 0.042, 0.130


def structure(qc) -> dict:
    n = qc.num_qubits
    t, m, o = [0.0] * n, [0] * n, [0] * n
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        for q in (qc.find_bit(b).index for b in inst.qubits):
            o[q] += 1
            if name == "measure":
                m[q] += 1
            else:
                t[q] += TCZ if name == "cz" else T1Q
    return {"tmax": max(t), "mmax": max(m), "omax": max(o), "active": sum(x > 0 for x in o)}


def run(args) -> None:
    from qiskit import qpy
    files = sorted(f for f in QPY.rglob("*.qpy") if f.parent.name not in DYNAMIC)
    files = [f for f in files if zlib.crc32(str(f.relative_to(QPY)).encode()) % args.n_tasks == args.task_id]
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t0 = time.time()
    with out_file.open("a") as fh:
        for i, f in enumerate(files):
            name = str(f.relative_to(QPY).with_suffix(""))
            if name in done:
                continue
            rec = {"circuit": name}
            try:
                with f.open("rb") as h:
                    circs = qpy.load(h)
                assert len(circs) == len(DEVICES), len(circs)
                for d, qc in zip(DEVICES, circs):
                    rec[d] = structure(qc)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                fh.flush(); print(f"[{time.time() - t0:6.0f} s] {i}/{len(files)} {name}", flush=True)


def merge(args) -> None:
    import torch
    out, err = {}, 0
    for f in sorted(OUT.glob("task*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" in r:
                err += 1
                continue
            out[r["circuit"]] = torch.tensor([[float(r[d][k]) for k in KEYS] for d in DEVICES])
    torch.save(out, HERE / "results/struct_labels.pt")
    print(f"{len(out)} circuits, {err} errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "merge"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    run(a) if a.mode == "run" else merge(a)
