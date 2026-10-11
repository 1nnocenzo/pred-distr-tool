"""Generalization study of the GNN fidelity predictor.

The scheduling evaluation in ``evaluations/pipeline`` trains and tests on a
random split that is stratified by fidelity.  Because the benchmark set contains
thousands of near-identical circuits per family, that split places
near-duplicates on both sides, so the reported in-distribution score is an
optimistic estimate of what the predictor does on a workload it has never seen.

This package quantifies that gap with three leakage-free conditions, all sharing
one training protocol and the tuned hyper-parameters of Table
``tab:ml4qc:scheduling-hparams`` (never re-tuned on held-out data):

1. **Leave-one-family-out** — train on every family but one, test on the held-out
   family (:mod:`~genstudy.data.GraphDataset.family_holdout_split`).
2. **Size extrapolation** — train on circuits of at most *n* qubits, test on the
   larger ones (:mod:`~genstudy.data.GraphDataset.size_holdout_split`).
3. **Random-split control** — the original split, retrained with this same code
   so the three numbers are comparable (:mod:`~genstudy.data.GraphDataset.random_split`).

The held-out predictions are then replayed through the unmodified scheduling
benchmark, which shows whether the policy stays close to GT-Weighted when the
fidelities it consumes come from a model that never saw the workload.

Module map:

:mod:`genstudy.paths`        repository/dataset path resolution
:mod:`genstudy.data`         dataset loading, family/size parsing, split construction
:mod:`genstudy.model`        frozen architecture, hyper-parameters, training loop
:mod:`genstudy.metrics`      MAE / RMSE / R^2 and scheduler-relevant metrics
:mod:`genstudy.experiment`   run one split end-to-end and persist artefacts
:mod:`genstudy.report`       CSV, LaTeX tables and figures
:mod:`genstudy.io`           JSON/CSV/logging helpers
"""

from __future__ import annotations

__all__ = [
    "data",
    "experiment",
    "io",
    "metrics",
    "model",
    "paths",
    "report",
]

__version__ = "1.0.0"
