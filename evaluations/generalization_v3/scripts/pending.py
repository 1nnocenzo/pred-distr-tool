#!/usr/bin/env python3
"""List the planned splits that are unfinished AND not being trained by any job.

The plan is a text file with one split per line (``#`` comments allowed)::

    RUN  EXPERIMENT  SPLIT  MODEL  ALPHA  SEED  [EXTRA ARGS...]
    v1_s6      lofo  qaoa  v1     -    6
    xattn_a05  logo  vqe   xattn  0.5  5
    v4_phys    lofo  qnn   phys   0    5   --loss mixed --val-mode random

Runs live under ``$RESULTS_ROOT`` (default: the v3 results directory).

For each line: done if ``results/RUN/<exp dir>/SPLIT/metrics.json`` exists, busy if
another process holds the split's ``.lock`` (non-blocking flock, released at once),
otherwise free.  Prints ``FREE <the plan line>`` for each free split and a final
``SUMMARY`` line.  Nothing is created or modified.

    python pending.py [PLAN]        (default: results/plan_v3.txt, the original 48 splits)
"""

from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS_ROOT") or Path(__file__).resolve().parents[1] / "results")
EXP_DIRS = {"control": "random_control", "lofo": "leave_one_family_out",
            "size": "size_extrapolation", "logo": "leave_one_group_out"}


def read_plan(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            fields = line.split()
            if len(fields) < 6 or fields[1] not in EXP_DIRS:
                raise ValueError(f"bad plan line: {line!r}")
            rows.append(fields)
    return rows


def is_locked(split_dir: Path) -> bool:
    lock = split_dir / ".lock"
    if not lock.exists():
        return False
    with open(lock, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def main() -> int:
    plan = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "plan_v3.txt"
    rows = read_plan(plan)
    n_done = n_busy = n_free = 0
    for run, exp, split, *rest in rows:
        d = RESULTS / run / EXP_DIRS[exp] / split
        if (d / "metrics.json").is_file():
            n_done += 1
        elif d.is_dir() and is_locked(d):
            n_busy += 1
        else:
            n_free += 1
            print("FREE", run, exp, split, *rest)
    print(f"SUMMARY done={n_done} busy={n_busy} free={n_free} total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
