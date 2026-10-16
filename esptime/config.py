"""Paths and fixed constants shared by every module."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CIRCUITS_ZIP = DATA / "circuits.zip"           # MQT Bench QASM files, benchmark_dataset_30k/<family>/<name>.qasm
WORK = Path(os.environ.get("ESPTIME_WORK", ROOT / "work"))   # generated data (git-ignored): circuits, labels, runs
DECOMP_TABLE = Path(__file__).resolve().parent / "decomp_table.json"

if str(ROOT) not in sys.path:                   # createDevice.py lives at the repository root
    sys.path.insert(0, str(ROOT))

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")

# Gate durations in µs (createDevice.py: uniform on every device): r, measure, cz
T1, TM, TCZ = 0.042, 15.0, 0.130
GATE_US = (T1, TM, TCZ)
T_SWAP = 3 * TCZ + 2 * T1                       # a SWAP in the native basis: 3 CZ + 4 r split over its 2 qubits

# Families left out: dynamic circuits (if_else, no static schedule) and grover (compilation memory)
EXCLUDE = ("grover", "iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code")
VARIATIONAL = ("vqe_su2", "vqe_real_amp", "vqe_two_local", "qnn")


def decomp_table() -> dict:
    """Native r / cz count of every gate type transpiled alone at level 2 (no routing): labels.py decomp."""
    return json.loads(DECOMP_TABLE.read_text())
