"""Path resolution for the generalization study.

The study deliberately contains no machine-specific absolute paths: every
location is either derived from the repository layout or taken from an
environment variable / command-line flag, so the code is reproducible on a
different host or cluster.

Layout assumed inside the repository::

    src/model/                     gnn.py, encoding.py, best_params.json
    evaluations/pipeline/          scheduling benchmark (read-only here)
    evaluations/generalization/    this study; results/ is written here only
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``evaluations/generalization/``
STUDY_DIR = Path(__file__).resolve().parents[1]
#: Repository root.
REPO_ROOT = STUDY_DIR.parents[1]
#: Trained-model artefacts and the circuit encoder shared with the pipeline.
MODEL_DIR = REPO_ROOT / "src" / "model"
#: The existing scheduling benchmark; imported, never written to.
PIPELINE_DIR = REPO_ROOT / "evaluations" / "pipeline"
#: Everything this study produces lands here (parallel to pipeline/results).
RESULTS_DIR = STUDY_DIR / "results"

#: Name of the environment variable pointing at the pre-built graph dataset.
DATASET_DIR_ENV = "COMPILE_CIRCUITS_DIR"

#: File that identifies a usable dataset directory.
_DATASET_STAMP = "graph_dataset_{figure_of_merit}.pt"


def _dataset_candidates(explicit: Path | str | None) -> list[Path]:
    """Ordered search locations for the pre-built PyG dataset."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get(DATASET_DIR_ENV)
    if env:
        candidates.append(Path(env))
    candidates += [
        REPO_ROOT / "data" / "graph_dataset",
        REPO_ROOT / "data",
        REPO_ROOT.parent / "compileCircuits",
        Path.home() / "compileCircuits",
    ]
    return candidates


def find_dataset_dir(
    explicit: Path | str | None = None,
    figure_of_merit: str = "expected_fidelity",
) -> Path:
    """Return the first directory that holds the pre-built graph dataset.

    Args:
        explicit: Directory given on the command line; tried first.
        figure_of_merit: Dataset flavour, i.e. the ``graph_dataset_<fom>.pt``
            suffix written by ``create_pt_file_new.py``.

    Raises:
        FileNotFoundError: If no candidate contains the dataset, listing every
            location that was tried.
    """
    stamp = _DATASET_STAMP.format(figure_of_merit=figure_of_merit)
    tried: list[Path] = []
    for cand in _dataset_candidates(explicit):
        tried.append(cand)
        if (cand / stamp).is_file():
            return cand.resolve()
    tried_str = "\n  ".join(str(t) for t in tried)
    raise FileNotFoundError(
        f"Could not locate '{stamp}'. Tried:\n  {tried_str}\n"
        f"Pass --dataset-dir or set ${DATASET_DIR_ENV} to the directory holding "
        "the dataset produced by create_pt_file_new.py."
    )


def ensure_model_imports() -> None:
    """Put ``src/`` and ``src/model/`` on ``sys.path``.

    ``gnn.py`` and ``encoding.py`` are flat modules inside ``src/model/``, which
    is also how ``evaluations/pipeline/gnn_device_policy.py`` imports them; doing
    the same here keeps a single source of truth for the architecture and the
    circuit encoding.
    """
    for p in (REPO_ROOT / "src", MODEL_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def ensure_pipeline_imports() -> None:
    """Put ``evaluations/pipeline/`` on ``sys.path`` for the scheduling re-run."""
    ensure_model_imports()
    sp = str(PIPELINE_DIR)
    if sp not in sys.path:
        sys.path.insert(0, sp)
