"""Path resolution for the v3 study.

v3 reuses, read-only, the v1 library (``evaluations/generalization/genstudy``:
dataset, splits, metrics, I/O), the v2 package (``evaluations/generalization_v2/gsv2``:
device graphs, the v2 model, validation split, balanced sampler, split lock) and
``src/model`` (circuit encoding, ``GraphConvolutionSage``).  Nothing outside
``evaluations/generalization_v3/`` is written.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``evaluations/generalization_v3/``
STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_DIR.parents[1]
V1_STUDY_DIR = REPO_ROOT / "evaluations" / "generalization"
V2_STUDY_DIR = REPO_ROOT / "evaluations" / "generalization_v2"
V2_RESULTS_DIR = V2_STUDY_DIR / "results"
#: Device graphs built by v2 (read-only input).
V2_DEVICE_GRAPHS = V2_RESULTS_DIR / "device_graphs.pt"

RESULTS_DIR = STUDY_DIR / "results"
#: Qubit-annotated copy of the graph dataset (see ``scripts/build_qubit_dataset.py``).
DATA_DIR = STUDY_DIR / "data"


def ensure_imports() -> None:
    """Make ``genstudy`` (v1), ``gsv2`` (v2) and ``src/model`` importable."""
    for p in (V1_STUDY_DIR, V2_STUDY_DIR, REPO_ROOT / "src", REPO_ROOT / "src" / "model"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
