#!/usr/bin/env python3
"""Check that the three device targets declare gate durations.

Regression guard for the bug that made every expected_runtime come out 0.0:
InstructionProperties were built with error= only, so the critical-path
estimate in testComp-compilation-time.py summed nothing but zeros.

Run:  python test_device_durations.py
"""

from createDevice import (
    CZ_DURATION,
    MEASURE_DURATION,
    R_DURATION,
    EQE1BottomBackend,
    EQE1TopBackend,
    QExa20Backend,
)

EXPECTED = {"r": R_DURATION, "cz": CZ_DURATION, "measure": MEASURE_DURATION}


def main() -> None:
    # Values reverse-engineered from src/model/expected_fidelity_results_benchmark.json:
    # every one of its 24070 distinct expected_runtime values is a non-negative
    # integer combination of exactly these three.
    assert (R_DURATION, CZ_DURATION, MEASURE_DURATION) == (42e-9, 130e-9, 15e-6)

    for backend in (EQE1TopBackend(), EQE1BottomBackend(), QExa20Backend()):
        target = backend.target
        for name, expected in EXPECTED.items():
            assert name in target.operation_names, f"{backend.name}: no '{name}' instruction"
            durations = [p.duration for p in target[name].values() if p is not None]
            assert durations, f"{backend.name}/{name}: no InstructionProperties"
            assert all(d == expected for d in durations), (
                f"{backend.name}/{name}: expected duration {expected}, got {set(durations)}"
            )
        print(f"{backend.name}: r/cz/measure durations OK")

    # A GHZ-shaped critical path (2 single-qubit gates + 1 CZ per step) must add
    # 214 ns per step, the increment observed across the ghz family in the
    # reference results file.
    assert 2 * R_DURATION + CZ_DURATION == 214e-9
    print("GHZ per-step increment 214 ns OK")


if __name__ == "__main__":
    main()
