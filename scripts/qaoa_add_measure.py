#!/usr/bin/env python3
"""Add the missing final measurements to the qaoa circuits of data/circuits.zip, in place (one-off, idempotent).

MQT Bench's qaoa generator is the only one without ``measure_all()``: the 2,964 qaoa circuits of the zip ended
without measurements, so their labels had no 15 µs readout (every other family has it).  Each qaoa file without a
measurement gets, as the other families are written (same textual edit as esp-depth's qaoa_measure.py):
``bit[n] meas;`` before the qubit register, then ``barrier`` on every qubit and ``meas[k] = measure q[k];``.
Every new file is re-loaded and checked: same gates, one more barrier, exactly n_qubits measurements.  Every other
entry of the zip is copied byte for byte.  The guard ``esptime.labels.check_circuits`` keeps it fixed.

    python scripts/qaoa_add_measure.py [--dry-run] [--jobs N]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ZIP = Path(__file__).resolve().parents[1] / "data" / "circuits.zip"


def add_measure(src: str) -> str:
    regs = re.findall(r"^qubit\[(\d+)\] (\w+);$", src, re.M)
    assert len(regs) == 1, f"expected one qubit register, found {regs}"
    m = re.search(r"^qubit\[(\d+)\] (\w+);$", src, re.M)
    n, reg = int(m.group(1)), m.group(2)
    text = src[:m.start()] + f"bit[{n}] meas;\n" + src[m.start():]
    return text.rstrip("\n") + "\n" + f"barrier {', '.join(f'{reg}[{k}]' for k in range(n))};\n" + \
        "".join(f"meas[{k}] = measure {reg}[{k}];\n" for k in range(n))


def check(src: str, new: str) -> bool:
    from qiskit import qasm3
    a, b = qasm3.loads(src), qasm3.loads(new)
    o0, o1 = Counter(a.count_ops()), Counter(b.count_ops())
    return (o1.get("measure", 0) == a.num_qubits == b.num_qubits and o1.get("barrier", 0) == o0.get("barrier", 0) + 1
            and all(o1.get(k, 0) == v for k, v in o0.items()) and set(o1) == set(o0) | {"measure", "barrier"})


def _one(src: str) -> str | None:
    new = add_measure(src)
    return new if check(src, new) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    tmp = ZIP.with_suffix(".zip.tmp")
    with zipfile.ZipFile(ZIP) as zin:
        todo = {i.filename: zin.read(i).decode() for i in zin.infolist()
                if "/qaoa/" in i.filename and i.filename.endswith(".qasm") and b"measure" not in zin.read(i)}
    names = sorted(todo)
    with Pool(args.jobs) as pool:                    # the qasm3 parser is slow: ~30 min on one core
        new = dict(zip(names, pool.map(_one, [todo[n] for n in names], chunksize=4)))
    bad = [n for n in names if new[n] is None]
    fixed = len(names) - len(bad)
    with zipfile.ZipFile(ZIP) as zin, zipfile.ZipFile(tmp, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info)
            if new.get(info.filename):
                data = new[info.filename].encode()
            zout.writestr(info, data)
    if bad or args.dry_run:
        tmp.unlink()
        print(f"{fixed} qaoa files would be fixed, {len(bad)} failed the check {bad[:5]}; zip unchanged")
        sys.exit(1 if bad else 0)
    tmp.replace(ZIP)
    print(f"{fixed} qaoa files now end with measurements -> {ZIP}")


if __name__ == "__main__":
    main()
