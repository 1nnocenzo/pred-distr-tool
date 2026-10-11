"""Calibration-shift study: same coupling maps, different gate errors, no retraining.

Question: if a device keeps its coupling map but its calibration changes (as it
does from one day to the next), do the trained predictors follow the change?

Device variants are built from the three original backends
(``compileCircuits/createDevice.py``) by changing only the ``error`` of the
``r``, ``cz`` and ``measure`` instructions of their Qiskit ``Target``; durations,
T1/T2 and the coupling map are untouched:

``orig``        unchanged (reference; must reproduce the dataset fidelities)
``x0.5``/``x2`` every error scaled by 0.5 / 2 (global quality change)
``shuffle``     per-qubit 1q/readout errors permuted across qubits and CZ errors
                permuted across couplers (same distribution, different places:
                tests whether the placement moves to the good qubits)
``jitterA/B``   every error multiplied by a log-normal factor (sigma 0.3) — a
                plausible day-to-day drift

Ground truth is computed exactly as for the dataset: Qiskit preset pass manager at
``optimization_level=2`` on the variant's target, then the expected-fidelity
formula (worst-case branch for control flow).
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

COMPILE_DIR = Path.home() / "compileCircuits"
DEVICE_NAMES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
VARIANTS = ("orig", "x0.5", "x2", "shuffle", "jitterA", "jitterB")
GATES = ("r", "cz", "measure")
MAX_ERROR = 0.5


def base_backends() -> dict:
    if str(COMPILE_DIR) not in sys.path:
        sys.path.insert(0, str(COMPILE_DIR))
    import createDevice  # noqa: E402
    return {"EQE1_Top": createDevice.EQE1TopBackend(),
            "EQE1_Bottom": createDevice.EQE1BottomBackend(),
            "QExa20": createDevice.QExa20Backend()}


def _set_error(target, gate, qargs, error):
    props = target[gate][qargs]
    new = type(props)(duration=props.duration, error=float(min(max(error, 0.0), MAX_ERROR)))
    target.update_instruction_properties(gate, qargs, new)


def make_variant(backend, variant: str, seed: int = 0):
    """Deep copy of ``backend`` whose target has the ``variant`` calibration."""
    be = copy.deepcopy(backend)
    if variant == "orig":
        return be
    t = be.target
    rng = np.random.default_rng(seed)
    for gate in GATES:
        qargs = [q for q, p in t[gate].items() if p is not None and p.error is not None]
        errors = np.array([t[gate][q].error for q in qargs], dtype=float)
        if variant == "x0.5":
            new = errors * 0.5
        elif variant == "x2":
            new = errors * 2.0
        elif variant == "shuffle":
            new = rng.permutation(errors)
        elif variant.startswith("jitter"):
            new = errors * np.exp(rng.normal(0.0, 0.3, size=errors.shape))
        else:
            raise ValueError(variant)
        for q, e in zip(qargs, new):
            _set_error(t, gate, q, e)
    return be


def all_variants() -> dict[str, object]:
    """``{"<device>/<variant>": backend}`` for every device and variant."""
    out = {}
    seeds = {"shuffle": 11, "jitterA": 21, "jitterB": 22}
    for name, be in base_backends().items():
        for i, v in enumerate(VARIANTS):
            out[f"{name}/{v}"] = make_variant(be, v, seed=seeds.get(v, 0) + 100 * i)
    return out


# --- ground truth (same formula as compileCircuits/testComp-random-EQE1-complete.py) ---

def _fidelity_recursive(qc, device, qubit_map=None) -> float:
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
            res *= min(_fidelity_recursive(b, device, phys) for b in blocks)
            continue
        key = (phys[0],) if len(phys) == 1 else tuple(phys[:2])
        res *= 1 - device[op.name][key].error
    return res


def compile_and_score(qc, backend) -> float:
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=2, target=backend.target)
    tc = pm.run(qc)
    return float(round(_fidelity_recursive(tc, backend.target), 10))


def neglog(e: float) -> float:
    return -math.log(max(1.0 - e, 1e-12))
