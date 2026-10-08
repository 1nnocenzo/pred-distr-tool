"""Path resolution for the v2 study.

The v1 library (``evaluations/generalization/genstudy``) is reused for dataset
loading, splits, metrics, I/O and reporting; :func:`ensure_v1_imports` makes it
importable.  All v2 output goes under ``evaluations/generalization_v2/results``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``evaluations/generalization_v2/``
STUDY_DIR = Path(__file__).resolve().parents[1]
#: Repository root.
REPO_ROOT = STUDY_DIR.parents[1]
#: The v1 study, whose ``genstudy`` package and results are reused read-only.
V1_STUDY_DIR = REPO_ROOT / "evaluations" / "generalization"
V1_RESULTS_DIR = V1_STUDY_DIR / "results"
#: Everything this study writes lands here.
RESULTS_DIR = STUDY_DIR / "results"
#: Pre-built device graphs (see ``scripts/build_device_graphs.py``).
DEVICE_GRAPHS_PATH = RESULTS_DIR / "device_graphs.pt"

#: Environment variable pointing at the compileCircuits folder (dataset + createDevice.py).
COMPILE_CIRCUITS_ENV = "COMPILE_CIRCUITS_DIR"


def ensure_v1_imports() -> None:
    """Put the v1 study directory on ``sys.path`` so ``import genstudy`` works."""
    sp = str(V1_STUDY_DIR)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def find_compile_circuits_dir(explicit: Path | str | None = None) -> Path:
    """Locate the compileCircuits folder that holds ``createDevice.py``."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(COMPILE_CIRCUITS_ENV):
        candidates.append(Path(os.environ[COMPILE_CIRCUITS_ENV]))
    candidates += [REPO_ROOT.parent / "compileCircuits", Path.home() / "compileCircuits"]
    for cand in candidates:
        if (cand / "createDevice.py").is_file():
            return cand.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"createDevice.py not found. Tried:\n  {tried}\n"
        f"Pass --compile-circuits-dir or set ${COMPILE_CIRCUITS_ENV}."
    )
