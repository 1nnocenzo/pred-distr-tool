"""
Create Qiskit BackendV2 devices for IQM backends with calibration-derived
gate errors, readout errors, and qubit properties (T1, T2).

Devices
-------
  EQE1        – 53 qubits, calibration from JSON (data/2026-03-02T04_14_02.290607Z.json)
  EQE1_Top    – 26 qubits, upper half of EQE1 (horizontal cut through the lattice centre)
  EQE1_Bottom – 27 qubits, lower half of EQE1
  QExa20      – 20 qubits, calibration from CSV  (data/qpu_dcdb_results.csv)

Topology source: hpcqc-scripts/backend_info.txt
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit import Measure
from qiskit.circuit.library import CZGate, IGate, RGate
from qiskit.providers import BackendV2, Options, QubitProperties
from qiskit.transpiler import InstructionProperties, Target

# ── helpers: fill missing values with category mean ─────────────────────────
def _fill_qubit_mean(d: dict[int, float], qubit_range: range) -> dict[int, float]:
    """Return *d* with every missing qubit index filled by the mean of known values."""
    if not d:
        return d
    mean_val = sum(d.values()) / len(d)
    return {q: d.get(q, mean_val) for q in qubit_range}


def _fill_edge_mean(
    d: dict[tuple[int, int], float],
    edges: list[list[int]],
) -> dict[tuple[int, int], float]:
    """Return *d* with every missing coupling-map edge filled by the mean."""
    if not d:
        return d
    mean_val = sum(d.values()) / len(d)
    out = dict(d)
    for e in edges:
        key = (e[0], e[1])
        rev = (e[1], e[0])
        if key not in out and rev not in out:
            out[key] = mean_val
    return out


# ── gate durations (seconds) ────────────────────────────────────────────────
# Neither calibration source (IQM JSON / QExa20 CSV) reports gate durations —
# they carry only fidelities and T1/T2 — so these are fixed nominal values,
# identical across the three devices. They reproduce exactly the
# expected_runtime values in src/model/expected_fidelity_results_benchmark.json
# (all 24070 distinct values are integer combinations of these three, with no
# exceptions), so runtime labels stay comparable with that reference run.
# Retune here if per-device pulse timings become available.
R_DURATION = 42e-9        # PRX / r single-qubit gate
CZ_DURATION = 130e-9      # CZ two-qubit gate
MEASURE_DURATION = 15e-6  # readout

# ── paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
JSON_PATH = BASE / "data" / "2026-03-02T04_14_02.290607Z.json"

# ── device constants (from backend_info.txt) ────────────────────────────────
BACKEND_NAME = "EQE1"
NUM_QUBITS = 53          # 54 physical labels minus QB34 (absent)
COUPLING_MAP = [
    [0, 1], [0, 4], [3, 2], [3, 4], [3, 9], [5, 1], [5, 4], [5, 6], [5, 11],
    [8, 2], [8, 7], [8, 9], [8, 16], [10, 4], [10, 9], [10, 11], [10, 18],
    [12, 6], [12, 11], [12, 13], [12, 20], [15, 7], [15, 14], [15, 16], [15, 23],
    [17, 9], [17, 16], [17, 18], [17, 25], [19, 11], [19, 18], [19, 20], [19, 27],
    [21, 13], [21, 20], [21, 29], [22, 14], [22, 23], [24, 16], [24, 23], [24, 25],
    [24, 32], [26, 18], [26, 25], [26, 27], [26, 33], [28, 20], [28, 27], [28, 29],
    [28, 35], [30, 29], [30, 37], [31, 23], [31, 32], [31, 38], [34, 33], [34, 35],
    [34, 42], [36, 29], [36, 35], [36, 37], [36, 44], [39, 38], [39, 40], [39, 45],
    [41, 33], [41, 40], [41, 42], [41, 47], [43, 35], [43, 42], [43, 44], [43, 49],
    [46, 40], [46, 45], [46, 47], [46, 50], [48, 42], [48, 47], [48, 49], [48, 52],
    [51, 47], [51, 50], [51, 52],
]


# ── helpers to parse the IQM JSON ───────────────────────────────────────────
def _load_observations(path: Path) -> dict[str, float]:
    """Return {dut_field: value} from the JSON calibration file."""
    with open(path) as fh:
        payload = json.load(fh)
    return {obs["dut_field"]: float(obs["value"]) for obs in payload["observations"]}


def _qb_to_idx(label: str) -> int:
    """Convert 1-based 'QB<n>' label to 0-based qubit index (QB1→0, QB54→53).
    QB34 does not exist; labels ≥35 are shifted down by 1."""
    n = int(label)
    return n - 1 if n <= 34 else n - 2        # skip the gap at QB34


_RE_1Q_GATE = re.compile(
    r"metrics\.rb\.prx\.drag_crf_sx\.QB(\d+)\.fidelity:par=d2$"
)
_RE_2Q_CZ_CLIFFORD = re.compile(
    r"metrics\.rb\.clifford\.uz_cz\.QB(\d+)__QB(\d+)\.fidelity:par=d2$"
)
_RE_READOUT_ERR = re.compile(
    r"metrics\.ssro\.measure_fidelity\.constant\.QB(\d+)\.(error_0_to_1|error_1_to_0)$"
)
_RE_T1 = re.compile(r"characterization\.model\.QB(\d+)\.t1_time$")
_RE_T2 = re.compile(r"characterization\.model\.QB(\d+)\.t2_echo_time$")


def _parse_calibration(obs: dict[str, float]):
    """Extract per-qubit and per-edge calibration dictionaries."""
    # 1-qubit gate error  (1 − fidelity)
    r_gate_error: dict[int, float] = {}
    # 2-qubit CZ gate error  (1 − fidelity)
    cz_gate_error: dict[tuple[int, int], float] = {}
    # Readout errors
    readout_e01: dict[int, float] = {}
    readout_e10: dict[int, float] = {}
    # Qubit coherence
    t1: dict[int, float] = {}
    t2: dict[int, float] = {}

    for field, value in obs.items():
        m = _RE_1Q_GATE.match(field)
        if m:
            idx = _qb_to_idx(m.group(1))
            r_gate_error[idx] = 1.0 - value
            continue

        m = _RE_2Q_CZ_CLIFFORD.match(field)
        if m:
            q0, q1 = _qb_to_idx(m.group(1)), _qb_to_idx(m.group(2))
            cz_gate_error[(q0, q1)] = 1.0 - value
            continue

        m = _RE_READOUT_ERR.match(field)
        if m:
            idx = _qb_to_idx(m.group(1))
            if m.group(2) == "error_0_to_1":
                readout_e01[idx] = value
            else:
                readout_e10[idx] = value
            continue

        m = _RE_T1.match(field)
        if m:
            idx = _qb_to_idx(m.group(1))
            t1[idx] = value              # already in seconds
            continue

        m = _RE_T2.match(field)
        if m:
            idx = _qb_to_idx(m.group(1))
            t2[idx] = value              # already in seconds
            continue

    return r_gate_error, cz_gate_error, readout_e01, readout_e10, t1, t2


# ── build the Target ────────────────────────────────────────────────────────
def _build_target(
    r_gate_error: dict[int, float],
    cz_gate_error: dict[tuple[int, int], float],
    readout_e01: dict[int, float],
    readout_e10: dict[int, float],
) -> Target:
    """Construct a Qiskit Target with per-qubit/edge error properties."""
    target = Target(num_qubits=NUM_QUBITS)

    all_qubits = list(range(NUM_QUBITS))

    # -- id gate (no error) --------------------------------------------------
    id_props = {(q,): InstructionProperties(error=0.0, duration=0.0) for q in all_qubits}
    target.add_instruction(IGate(), id_props, name="id")

    # -- r gate (1-qubit rotation, error from RB) ----------------------------
    r_props = {}
    for q in all_qubits:
        err = r_gate_error.get(q)          # None if not calibrated
        r_props[(q,)] = InstructionProperties(error=err, duration=R_DURATION)
    target.add_instruction(RGate(0, 0), r_props, name="r")

    # -- cz gate (2-qubit, error from Clifford RB) --------------------------
    cz_props = {}
    for edge in COUPLING_MAP:
        q0, q1 = edge
        err = cz_gate_error.get((q0, q1)) or cz_gate_error.get((q1, q0))
        cz_props[(q0, q1)] = InstructionProperties(error=err, duration=CZ_DURATION)
    target.add_instruction(CZGate(), cz_props, name="cz")

    # -- measure (readout error = average of e01 and e10) --------------------
    meas_props = {}
    for q in all_qubits:
        e01 = readout_e01.get(q)
        e10 = readout_e10.get(q)
        if e01 is not None and e10 is not None:
            err = (e01 + e10) / 2.0
        else:
            err = e01 or e10              # use whichever is available
        meas_props[(q,)] = InstructionProperties(error=err, duration=MEASURE_DURATION)
    target.add_instruction(Measure(), meas_props, name="measure")

    return target


# ── BackendV2 implementation ───────────────────────────────────────────────
class EQE1Backend(BackendV2):
    """Qiskit BackendV2 for IQM EQE1 with real calibration data."""

    def __init__(self, json_path: Path = JSON_PATH):
        super().__init__(
            name=BACKEND_NAME,
            description="IQM EQE1 53-qubit superconducting QPU",
        )

        # Load and parse calibration
        obs = _load_observations(json_path)
        (
            self._r_gate_error,
            self._cz_gate_error,
            self._readout_e01,
            self._readout_e10,
            self._t1,
            self._t2,
        ) = _parse_calibration(obs)

        # Fill missing values with category means
        qr = range(NUM_QUBITS)
        self._r_gate_error  = _fill_qubit_mean(self._r_gate_error, qr)
        self._cz_gate_error = _fill_edge_mean(self._cz_gate_error, COUPLING_MAP)
        self._readout_e01   = _fill_qubit_mean(self._readout_e01, qr)
        self._readout_e10   = _fill_qubit_mean(self._readout_e10, qr)
        self._t1            = _fill_qubit_mean(self._t1, qr)
        self._t2            = _fill_qubit_mean(self._t2, qr)

        # Build Target (gate catalogue with errors)
        self._target = _build_target(
            self._r_gate_error,
            self._cz_gate_error,
            self._readout_e01,
            self._readout_e10,
        )

        # Qubit properties (T1, T2)
        self._qubit_properties = [
            QubitProperties(
                t1=self._t1.get(q),
                t2=self._t2.get(q),
            )
            for q in range(NUM_QUBITS)
        ]

    # Required BackendV2 properties ------------------------------------------
    @property
    def target(self) -> Target:
        return self._target

    @property
    def num_qubits(self) -> int:
        return NUM_QUBITS

    @property
    def max_circuits(self) -> int | None:
        return None

    @property
    def qubit_properties(self) -> list[QubitProperties]:
        return self._qubit_properties

    @classmethod
    def _default_options(cls):
        from qiskit.providers import Options
        return Options(shots=1024)

    def run(self, circuits, **kwargs):
        raise NotImplementedError(
            "EQE1Backend is a description-only backend for noise modelling; "
            "use a simulator or the real IQM backend to execute circuits."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  EQE1 sub-devices  (horizontal cut through the lattice centre)
# ═══════════════════════════════════════════════════════════════════════════
# The 53-qubit EQE1 diamond lattice is laid out in 9 rows:
#   Row 0: Q0-Q1    (2)     Row 5: Q31-Q37  (7)
#   Row 1: Q2-Q6    (5)     Row 6: Q38-Q44  (7)
#   Row 2: Q7-Q13   (7)     Row 7: Q45-Q49  (5)
#   Row 3: Q14-Q21  (8)     Row 8: Q50-Q52  (3)
#   Row 4: Q22-Q30  (9) ← widest row
#
# A horizontal cut splits the device through the middle of row 4:
#   EQE1_Top    – rows 0-3 + upper part of row 4  (EQE1 qubits Q0-Q25,  26 qubits)
#   EQE1_Bottom – lower part of row 4 + rows 5-8  (EQE1 qubits Q26-Q52, 27 qubits)
#
# Both sub-devices use 0-based contiguous qubit indexing.
# EQE1_Top    uses num_qubits = 26  (indices 0-25;  EQE1 Q0-Q25,  offset 0).
# EQE1_Bottom uses num_qubits = 27  (indices 0-26;  EQE1 Q26-Q52, offset 26).
#
# Severed edges (7): (19,27) (21,29) (24,32) (26,18) (26,25) (28,20) (31,23)

EQE1_TOP_NAME = "EQE1_Top"
EQE1_TOP_NUM_QUBITS = 26
EQE1_TOP_ACTIVE_QUBITS = list(range(0, 26))     # original EQE1 indices
EQE1_TOP_COUPLING_MAP = [
    [0, 1], [0, 4], [3, 2], [3, 4], [3, 9], [5, 1], [5, 4], [5, 6], [5, 11],
    [8, 2], [8, 7], [8, 9], [8, 16], [10, 4], [10, 9], [10, 11], [10, 18],
    [12, 6], [12, 11], [12, 13], [12, 20], [15, 7], [15, 14], [15, 16], [15, 23],
    [17, 9], [17, 16], [17, 18], [17, 25], [19, 11], [19, 18], [19, 20],
    [21, 13], [21, 20], [22, 14], [22, 23], [24, 16], [24, 23], [24, 25],
]
print(len(EQE1_TOP_COUPLING_MAP))  # 36 edges
EQE1_BOT_NAME = "EQE1_Bottom"
EQE1_BOT_NUM_QUBITS = 27                        # 0-based re-indexed (EQE1 Q26-Q52 → 0-26)
EQE1_BOT_ACTIVE_QUBITS = list(range(0, 27))     # 0-based sub-device indices
EQE1_BOT_QUBIT_OFFSET = 26                      # original EQE1 index = sub-device index + 26
EQE1_BOT_COUPLING_MAP = [
    [0, 1], [0, 7], [2, 1], [2, 3], [2, 9], [4, 3], [4, 11],
    [5, 6], [5, 12], [8, 7], [8, 9], [8, 16],
    [10, 3], [10, 9], [10, 11], [10, 18],
    [13, 12], [13, 14], [13, 19],
    [15, 7], [15, 14], [15, 16], [15, 21],
    [17, 9], [17, 16], [17, 18], [17, 23],
    [20, 14], [20, 19], [20, 21], [20, 24],
    [22, 16], [22, 21], [22, 23], [22, 26],
    [25, 21], [25, 24], [25, 26],
]

print(len(EQE1_BOT_COUPLING_MAP))  # 37 edges (7 severed from the original EQE1 coupling map)
# ── helpers for sub-device construction ─────────────────────────────────────
def _filter_qubit_dict(
    d: dict[int, float], active_set: set[int],
) -> dict[int, float]:
    """Keep only entries whose qubit index is in *active_set*."""
    return {q: v for q, v in d.items() if q in active_set}


def _filter_edge_dict(
    d: dict[tuple[int, int], float], active_set: set[int],
) -> dict[tuple[int, int], float]:
    """Keep only edges whose both endpoints are in *active_set*."""
    return {
        (a, b): v
        for (a, b), v in d.items()
        if a in active_set and b in active_set
    }


def _build_eqe1_sub_target(
    num_qubits: int,
    coupling_map: list[list[int]],
    active_qubits: list[int],
    r_gate_error: dict[int, float],
    cz_gate_error: dict[tuple[int, int], float],
    readout_e01: dict[int, float],
    readout_e10: dict[int, float],
) -> Target:
    """Construct a Qiskit Target for an EQE1 sub-device.

    Gate / measurement properties are only created for *active_qubits*.
    """
    target = Target(num_qubits=num_qubits)

    id_props = {(q,): InstructionProperties(error=0.0, duration=0.0) for q in active_qubits}
    target.add_instruction(IGate(), id_props, name="id")

    r_props = {
        (q,): InstructionProperties(error=r_gate_error.get(q), duration=R_DURATION)
        for q in active_qubits
    }
    target.add_instruction(RGate(0, 0), r_props, name="r")

    cz_props = {}
    for edge in coupling_map:
        q0, q1 = edge
        err = cz_gate_error.get((q0, q1)) or cz_gate_error.get((q1, q0))
        cz_props[(q0, q1)] = InstructionProperties(error=err, duration=CZ_DURATION)
    target.add_instruction(CZGate(), cz_props, name="cz")

    meas_props = {}
    for q in active_qubits:
        e01 = readout_e01.get(q)
        e10 = readout_e10.get(q)
        if e01 is not None and e10 is not None:
            err = (e01 + e10) / 2.0
        else:
            err = e01 or e10
        meas_props[(q,)] = InstructionProperties(error=err, duration=MEASURE_DURATION)
    target.add_instruction(Measure(), meas_props, name="measure")

    return target


# ── BackendV2 implementation (sub-devices) ──────────────────────────────────
class _EQE1SubBackend(BackendV2):
    """Base class for an EQE1 sub-device (top or bottom half).

    Sub-devices use 0-based contiguous qubit indexing.  A *qubit_offset*
    maps sub-device index → original EQE1 index for calibration lookup
    (original_index = sub_device_index + qubit_offset).
    """

    def __init__(
        self,
        name: str,
        description: str,
        num_qubits: int,
        active_qubits: list[int],
        coupling_map: list[list[int]],
        json_path: Path = JSON_PATH,
        qubit_offset: int = 0,
    ):
        super().__init__(name=name, description=description)
        self._num_qubits = num_qubits
        self._active_qubits = active_qubits
        self._qubit_offset = qubit_offset

        # Load and parse the full EQE1 calibration
        obs = _load_observations(json_path)
        (r_err, cz_err, e01, e10, t1_raw, t2_raw) = _parse_calibration(obs)

        # Original EQE1 qubit indices for calibration data lookup
        original_active = {q + qubit_offset for q in active_qubits}

        # Filter calibration data using original EQE1 indices
        r_filt   = _filter_qubit_dict(r_err, original_active)
        cz_filt  = _filter_edge_dict(cz_err, original_active)
        e01_filt = _filter_qubit_dict(e01, original_active)
        e10_filt = _filter_qubit_dict(e10, original_active)
        t1_filt  = _filter_qubit_dict(t1_raw, original_active)
        t2_filt  = _filter_qubit_dict(t2_raw, original_active)

        # Remap to 0-based sub-device indices
        def _remap_q(d):
            return {q - qubit_offset: v for q, v in d.items()}
        def _remap_e(d):
            return {(a - qubit_offset, b - qubit_offset): v
                    for (a, b), v in d.items()}

        self._r_gate_error  = _remap_q(r_filt)
        self._cz_gate_error = _remap_e(cz_filt)
        self._readout_e01   = _remap_q(e01_filt)
        self._readout_e10   = _remap_q(e10_filt)
        self._t1            = _remap_q(t1_filt)
        self._t2            = _remap_q(t2_filt)

        # Fill missing values with category means (only for active qubits)
        qr = active_qubits
        self._r_gate_error  = _fill_qubit_mean(self._r_gate_error, qr)
        self._cz_gate_error = _fill_edge_mean(self._cz_gate_error, coupling_map)
        self._readout_e01   = _fill_qubit_mean(self._readout_e01, qr)
        self._readout_e10   = _fill_qubit_mean(self._readout_e10, qr)
        self._t1            = _fill_qubit_mean(self._t1, qr)
        self._t2            = _fill_qubit_mean(self._t2, qr)

        self._target = _build_eqe1_sub_target(
            num_qubits, coupling_map, active_qubits,
            self._r_gate_error, self._cz_gate_error,
            self._readout_e01, self._readout_e10,
        )

        # Qubit properties list covers 0..num_qubits-1.
        active_set = set(active_qubits)
        self._qubit_properties = [
            QubitProperties(t1=self._t1.get(q), t2=self._t2.get(q))
            if q in active_set
            else QubitProperties()
            for q in range(num_qubits)
        ]

    # Required BackendV2 properties ------------------------------------------
    @property
    def target(self) -> Target:
        return self._target

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def active_qubits(self) -> list[int]:
        """0-based sub-device qubit indices."""
        return list(self._active_qubits)

    @property
    def qubit_offset(self) -> int:
        """Offset to reconstruct original EQE1 index: eqe1_idx = sub_idx + offset."""
        return self._qubit_offset

    @property
    def original_qubit_indices(self) -> list[int]:
        """Original EQE1 qubit indices corresponding to this sub-device."""
        return [q + self._qubit_offset for q in self._active_qubits]

    @property
    def max_circuits(self) -> int | None:
        return None

    @property
    def qubit_properties(self) -> list[QubitProperties]:
        return self._qubit_properties

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, circuits, **kwargs):
        raise NotImplementedError(
            f"{self.name} is a description-only backend for noise modelling; "
            "use a simulator or the real IQM backend to execute circuits."
        )


class EQE1TopBackend(_EQE1SubBackend):
    """Upper half of the EQE1 device (EQE1 qubits Q0-Q25, 26 qubits)."""

    def __init__(self, json_path: Path = JSON_PATH):
        super().__init__(
            name=EQE1_TOP_NAME,
            description="IQM EQE1 top half – 26 active qubits Q0-Q25 (rows 0-3 + upper row 4)",
            num_qubits=EQE1_TOP_NUM_QUBITS,
            active_qubits=EQE1_TOP_ACTIVE_QUBITS,
            coupling_map=EQE1_TOP_COUPLING_MAP,
            json_path=json_path,
        )


class EQE1BottomBackend(_EQE1SubBackend):
    """Lower half of the EQE1 device (EQE1 qubits Q26-Q52, 27 qubits).

    Re-indexed to 0-based (num_qubits = 27, indices 0-26).
    Original EQE1 index = sub-device index + 26.
    """

    def __init__(self, json_path: Path = JSON_PATH):
        super().__init__(
            name=EQE1_BOT_NAME,
            description="IQM EQE1 bottom half – 27 qubits (lower row 4 + rows 5-8), re-indexed 0-26",
            num_qubits=EQE1_BOT_NUM_QUBITS,
            active_qubits=EQE1_BOT_ACTIVE_QUBITS,
            coupling_map=EQE1_BOT_COUPLING_MAP,
            json_path=json_path,
            qubit_offset=EQE1_BOT_QUBIT_OFFSET,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  QExa20 Backend  (20 qubits, calibration from CSV time-series)
# ═══════════════════════════════════════════════════════════════════════════
CSV_PATH = BASE / "data" / "qpu_dcdb_results.csv"

QEXA20_NAME = "QExa20"
QEXA20_NUM_QUBITS = 20
QEXA20_COUPLING_MAP = [
    [1, 0], [1, 4], [3, 0], [3, 2], [3, 4], [3, 8],
    [5, 4], [5, 6], [5, 10], [7, 2], [7, 8], [7, 12],
    [9, 4], [9, 8], [9, 10], [9, 14], [11, 6], [11, 10], [11, 16],
    [13, 8], [13, 12], [13, 14], [13, 17],
    [15, 10], [15, 14], [15, 16], [15, 19],
    [18, 14], [18, 17], [18, 19],
]

# Regex for TC-<a>-<b> edge labels (1-based qubit IDs)
_RE_TC_EDGE = re.compile(r"^TC-(\d+)-(\d+)$")


def _load_csv_calibration(csv_path: Path):
    """Parse the QExa20 CSV calibration into per-qubit / per-edge dicts.

    CSV raw integer values are in 1e-4 units (fidelity, error) or µs·10⁴ (T1/T2).
    """
    df = pd.read_csv(
        csv_path,
        sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)',
        engine="python",
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    sp = df["sensor"].astype(str).str.rsplit("/", n=2, expand=True)
    df["entity"] = sp[1]
    df["metric_raw"] = sp[2]
    df["value_scaled"] = df["value"] / 10000.0

    means = df.groupby(["entity", "metric_raw"])["value_scaled"].mean()

    # -- 1-qubit gate error (from fidelity_1qb_gates_averaged) ---------------
    r_gate_error: dict[int, float] = {}
    fid_1q = means.xs("fidelity_1qb_gates_averaged", level="metric_raw")
    for entity, fid in fid_1q.items():
        m = re.match(r"^QB(\d+)$", entity)
        if m:
            q = int(m.group(1)) - 1          # 1-based → 0-based
            r_gate_error[q] = 1.0 - fid

    # -- 2-qubit CZ gate error (from cz_gate_fidelity) ----------------------
    cz_gate_error: dict[tuple[int, int], float] = {}
    cz_fid = means.xs("cz_gate_fidelity", level="metric_raw")
    for entity, fid in cz_fid.items():
        m = _RE_TC_EDGE.match(entity)
        if m:
            q0, q1 = int(m.group(1)) - 1, int(m.group(2)) - 1
            cz_gate_error[(q0, q1)] = 1.0 - fid

    # -- Readout errors -------------------------------------------------------
    readout_e01: dict[int, float] = {}
    readout_e10: dict[int, float] = {}
    for metric, target_dict in [("error_0_1", readout_e01), ("error_1_0", readout_e10)]:
        try:
            sub = means.xs(metric, level="metric_raw")
        except KeyError:
            continue
        for entity, val in sub.items():
            m = re.match(r"^QB(\d+)$", entity)
            if m:
                q = int(m.group(1)) - 1
                target_dict[q] = val

    # -- T1 / T2 (value_scaled is in µs, convert to seconds) -----------------
    t1: dict[int, float] = {}
    t2: dict[int, float] = {}
    for metric, target_dict in [("t1_time", t1), ("t2_echo_time", t2)]:
        try:
            sub = means.xs(metric, level="metric_raw")
        except KeyError:
            continue
        for entity, val in sub.items():
            m = re.match(r"^QB(\d+)$", entity)
            if m:
                q = int(m.group(1)) - 1
                target_dict[q] = val * 1e-6   # µs → s

    return r_gate_error, cz_gate_error, readout_e01, readout_e10, t1, t2


def _build_qexa20_target(
    r_gate_error: dict[int, float],
    cz_gate_error: dict[tuple[int, int], float],
    readout_e01: dict[int, float],
    readout_e10: dict[int, float],
) -> Target:
    """Construct a Qiskit Target for QExa20."""
    target = Target(num_qubits=QEXA20_NUM_QUBITS)
    all_qubits = list(range(QEXA20_NUM_QUBITS))

    # -- id gate --------------------------------------------------------------
    id_props = {(q,): InstructionProperties(error=0.0, duration=0.0) for q in all_qubits}
    target.add_instruction(IGate(), id_props, name="id")

    # -- r gate (1-qubit, error from RB) --------------------------------------
    r_props = {}
    for q in all_qubits:
        r_props[(q,)] = InstructionProperties(error=r_gate_error.get(q), duration=R_DURATION)
    target.add_instruction(RGate(0, 0), r_props, name="r")

    # -- cz gate (2-qubit, error from CZ gate fidelity) ----------------------
    cz_props = {}
    for edge in QEXA20_COUPLING_MAP:
        q0, q1 = edge
        err = cz_gate_error.get((q0, q1)) or cz_gate_error.get((q1, q0))
        cz_props[(q0, q1)] = InstructionProperties(error=err, duration=CZ_DURATION)
    target.add_instruction(CZGate(), cz_props, name="cz")

    # -- measure (readout error = average of e01 and e10) ---------------------
    meas_props = {}
    for q in all_qubits:
        e01 = readout_e01.get(q)
        e10 = readout_e10.get(q)
        if e01 is not None and e10 is not None:
            err = (e01 + e10) / 2.0
        else:
            err = e01 or e10
        meas_props[(q,)] = InstructionProperties(error=err, duration=MEASURE_DURATION)
    target.add_instruction(Measure(), meas_props, name="measure")

    return target


class QExa20Backend(BackendV2):
    """Qiskit BackendV2 for IQM QExa20 with real calibration data from CSV."""

    def __init__(self, csv_path: Path = CSV_PATH):
        super().__init__(
            name=QEXA20_NAME,
            description="IQM QExa20 20-qubit superconducting QPU",
        )

        (
            self._r_gate_error,
            self._cz_gate_error,
            self._readout_e01,
            self._readout_e10,
            self._t1,
            self._t2,
        ) = _load_csv_calibration(csv_path)

        # Fill missing values with category means
        qr = range(QEXA20_NUM_QUBITS)
        self._r_gate_error  = _fill_qubit_mean(self._r_gate_error, qr)
        self._cz_gate_error = _fill_edge_mean(self._cz_gate_error, QEXA20_COUPLING_MAP)
        self._readout_e01   = _fill_qubit_mean(self._readout_e01, qr)
        self._readout_e10   = _fill_qubit_mean(self._readout_e10, qr)
        self._t1            = _fill_qubit_mean(self._t1, qr)
        self._t2            = _fill_qubit_mean(self._t2, qr)

        self._target = _build_qexa20_target(
            self._r_gate_error,
            self._cz_gate_error,
            self._readout_e01,
            self._readout_e10,
        )

        self._qubit_properties = [
            QubitProperties(
                t1=self._t1.get(q),
                t2=self._t2.get(q),
            )
            for q in range(QEXA20_NUM_QUBITS)
        ]

    @property
    def target(self) -> Target:
        return self._target

    @property
    def num_qubits(self) -> int:
        return QEXA20_NUM_QUBITS

    @property
    def max_circuits(self) -> int | None:
        return None

    @property
    def qubit_properties(self) -> list[QubitProperties]:
        return self._qubit_properties

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, circuits, **kwargs):
        raise NotImplementedError(
            "QExa20Backend is a description-only backend for noise modelling; "
            "use a simulator or the real IQM backend to execute circuits."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test when run as a script
# ═══════════════════════════════════════════════════════════════════════════
def _print_backend_summary(backend: BackendV2):
    """Pretty-print gate errors, readout errors, T1/T2 for a BackendV2."""
    target = backend.target
    # Determine which qubits to display (sub-devices expose active_qubits)
    qubits = getattr(backend, 'active_qubits', None) or list(range(backend.num_qubits))
    label = f"{len(qubits)} active qubits" if hasattr(backend, 'active_qubits') else f"{backend.num_qubits} qubits"
    print(f"\nBackend: {backend.name}  ({label})")
    print("=" * 60)

    # ── Qubit properties (T1, T2) ──────────────────────────────────────────
    print("\n--- Qubit properties (T1, T2) ---")
    for q in qubits:
        qp = backend.qubit_properties[q]
        t1_us = f"{qp.t1 * 1e6:.1f} µs" if qp.t1 else "N/A"
        t2_us = f"{qp.t2 * 1e6:.1f} µs" if qp.t2 else "N/A"
        print(f"  Q{q:2d}:  T1 = {t1_us:>12s}   T2 = {t2_us:>12s}")

    # ── 1-qubit gate (r) errors ────────────────────────────────────────────
    print("\n--- 1-qubit gate 'r' errors ---")
    for q in qubits:
        ip = target["r"].get((q,))
        if ip and ip.error is not None:
            print(f"  Q{q:2d}:  error = {ip.error:.6f}")

    # ── 2-qubit gate (cz) errors ──────────────────────────────────────────
    print("\n--- 2-qubit gate 'cz' errors ---")
    for qargs, ip in sorted(target["cz"].items()):
        err_str = f"{ip.error:.6f}" if ip.error is not None else "N/A"
        print(f"  {qargs}:  error = {err_str}")

    # ── Measurement (readout) errors ──────────────────────────────────────
    print("\n--- Measurement readout errors ---")
    for q in qubits:
        ip = target["measure"].get((q,))
        if ip and ip.error is not None:
            print(f"  Q{q:2d}:  readout error = {ip.error:.6f}")
    print()


if __name__ == "__main__":
    print("Loading EQE1 backend …")
    eqe1 = EQE1Backend()
    _print_backend_summary(eqe1)

    print("Loading EQE1_Top backend …")
    eqe1_top = EQE1TopBackend()
    _print_backend_summary(eqe1_top)

    print("Loading EQE1_Bottom backend …")
    eqe1_bot = EQE1BottomBackend()
    _print_backend_summary(eqe1_bot)

    print("Loading QExa20 backend …")
    qexa20 = QExa20Backend()
    _print_backend_summary(qexa20)

    # ── helper to write a backend summary to a text file ──────────────────
    def _write_backend_info(path: Path, backend: BackendV2):
        qubits = getattr(backend, 'active_qubits', None) or list(range(backend.num_qubits))
        label = f"{len(qubits)} active qubits" if hasattr(backend, 'active_qubits') else f"{backend.num_qubits} qubits"
        with open(path, "w") as f:
            f.write(f"Backend: {backend.name}  ({label})\n")
            f.write("=" * 60 + "\n\n")
            f.write("--- Qubit properties (T1, T2) ---\n")
            for q in qubits:
                qp = backend.qubit_properties[q]
                t1_us = f"{qp.t1 * 1e6:.1f} µs" if qp.t1 else "N/A"
                t2_us = f"{qp.t2 * 1e6:.1f} µs" if qp.t2 else "N/A"
                f.write(f"  Q{q:2d}:  T1 = {t1_us:>12s}   T2 = {t2_us:>12s}\n")
            f.write("\n--- 1-qubit gate 'r' errors ---\n")
            for q in qubits:
                ip = backend.target["r"].get((q,))
                if ip and ip.error is not None:
                    f.write(f"  Q{q:2d}:  error = {ip.error:.6f}\n")
            f.write("\n--- 2-qubit gate 'cz' errors ---\n")
            for qargs, ip in sorted(backend.target["cz"].items()):
                err_str = f"{ip.error:.6f}" if ip.error is not None else "N/A"
                f.write(f"  {qargs}:  error = {err_str}\n")
            f.write("\n--- Measurement readout errors ---\n")
            for q in qubits:
                ip = backend.target["measure"].get((q,))
                if ip and ip.error is not None:
                    f.write(f"  Q{q:2d}:  readout error = {ip.error:.6f}\n")

    # save summaries for quick reference
    _write_backend_info(BASE / "EQE1_backend_info.txt", eqe1)
    _write_backend_info(BASE / "EQE1_Top_backend_info.txt", eqe1_top)
    _write_backend_info(BASE / "EQE1_Bottom_backend_info.txt", eqe1_bot)
    _write_backend_info(BASE / "QExa20_backend_info.txt", qexa20)