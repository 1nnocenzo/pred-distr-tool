#!/usr/bin/env python3
"""Add the missing final measurements to the qaoa circuits of the benchmark dataset.

MQT Bench's ``qaoa.create_circuit`` is the only benchmark without ``qc.measure_all()`` (every other family calls
it), so the 2,964 qaoa circuits of ``benchmark_dataset_30k`` end without measurements.  This script:
  1. copies the original qaoa QASM files to ``benchmark_dataset_30k_qaoa_nomeas/qaoa`` (kept untouched);
  2. rewrites each ``benchmark_dataset_30k/qaoa/*.qasm`` as the same circuit + ``measure_all()`` (barrier, then a
     measurement of every qubit into a ``meas`` register, as the other families);
  3. checks every new file: it re-loads, has the same gates plus exactly n_qubits measurements.
    python qaoa_measure.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

CC = Path.home() / "compileCircuits"
SRC = CC / "benchmark_dataset_30k" / "qaoa"
BACKUP = CC / "benchmark_dataset_30k_qaoa_nomeas" / "qaoa"


def main():
    from qiskit import qasm3
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = sorted(SRC.glob("*.qasm"))
    if not BACKUP.exists():
        if not args.dry_run:
            BACKUP.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SRC, BACKUP)
        print(f"backup: {len(files)} files -> {BACKUP}")
    else:
        print(f"backup already there ({len(list(BACKUP.glob('*.qasm')))} files)")
    bad = 0
    for i, f in enumerate(files):
        src = (BACKUP / f.name).read_text() if BACKUP.exists() else f.read_text()
        orig = qasm3.loads(src)
        assert orig.count_ops().get("measure", 0) == 0, f"{f.name} already measured"
        m = re.search(r"^qubit\[(\d+)\] (\w+);$", src, re.M)
        n, reg = int(m.group(1)), m.group(2)
        # textual edit, as the other families are written (a Qiskit round trip renames the custom rzz gate)
        text = src[:m.start()] + f"bit[{n}] meas;\n" + src[m.start():]
        text = text.rstrip("\n") + "\n" + f"barrier {', '.join(f'{reg}[{k}]' for k in range(n))};\n" + \
            "".join(f"meas[{k}] = measure {reg}[{k}];\n" for k in range(n))
        back = qasm3.loads(text)
        ops0, ops1 = Counter(orig.count_ops()), Counter(back.count_ops())
        ok = (ops1.get("measure", 0) == orig.num_qubits == n and ops1.get("barrier", 0) == ops0.get("barrier", 0) + 1 and
              all(ops1.get(k, 0) == v for k, v in ops0.items()) and set(ops1) == set(ops0) | {"measure", "barrier"} and
              back.num_qubits == n)
        if not ok:
            bad += 1
            print("MISMATCH", f.name, dict(ops0), dict(ops1))
            continue
        if not args.dry_run:
            f.write_text(text)
        if i % 500 == 0:
            print(i, f.name, dict(ops1), flush=True)
    print(f"{len(files)} files, {bad} mismatches, dry-run={args.dry_run}")


if __name__ == "__main__":
    main()
