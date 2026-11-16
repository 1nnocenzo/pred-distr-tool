"""Every MQT circuit of data/circuits.zip ends with measurements (qaoa had none: labels without the 15 µs readout).

    python tests/test_circuits.py      (a few seconds, no qiskit needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esptime.labels import check_circuits, measured  # noqa: E402


def test_zip_measured():
    check_circuits()


def test_stale_record():
    ok = {"calib": {"orig": {"EQE1_Top": {"counts": [10, 3, 4]}}}}
    old = {"calib": {"orig": {"EQE1_Top": {"counts": [10, 0, 4]}}}}
    assert measured(ok) and not measured(old) and not measured({**ok, "error": "x"})


if __name__ == "__main__":
    test_stale_record()
    test_zip_measured()
    print("circuits OK: every MQT circuit has measurements")
