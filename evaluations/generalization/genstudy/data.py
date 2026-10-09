"""Dataset access and split construction for the generalization study.

The study re-uses the artefacts built by ``create_pt_file_new.py``:

``graph_dataset_<fom>.pt``
    A list of PyG ``Data`` objects, one per circuit, carrying

    * ``x``, ``edge_index``  — the feature-annotated DAG (see ``src/model/encoding.py``),
    * ``y``                  — shape ``(1, 3)``, the ground-truth fidelities in
      ``DEVICE_NAMES`` order,
    * ``circuit_name``       — the ``"family/name_q<N>_v<id>"`` stem.

``names_list_<fom>.npy``
    The same stems in dataset order; used only if ``circuit_name`` is absent.

Two split families are provided, both of which avoid the near-duplicate leakage
of a random split:

``family_holdout``
    Leave-one-family-out: all circuits of one benchmark family form the test
    set, every other family is training data.

``size_holdout``
    Size extrapolation: circuits with at most *n* qubits are training data,
    larger ones (up to the dataset maximum) form the test set.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Model output order; must match ``create_pt_file_new.py``'s ``DEVICES``.
DEVICE_NAMES: tuple[str, ...] = ("EQE1_Top", "EQE1_Bottom", "QExa20")

#: Default dataset flavour.
FIGURE_OF_MERIT = "expected_fidelity"

#: Families requested for leave-one-family-out, in descending dataset size.
#: Small families are excluded by default because a few dozen test circuits do
#: not support a meaningful per-family R^2.
LARGE_FAMILIES: tuple[str, ...] = (
    "vqe_two_local",
    "vqe_real_amp",
    "vqe_su2",
    "qaoa",
    "randomcircuit",
    "qft",
    "qnn",
    "ae",
    "qftentangled",
    "iqpe",
)

_QUBIT_RE = re.compile(r"_q(\d+)_")


def family_of(circuit_name: str) -> str:
    """``"vqe_su2/vqe_su2_q12_v0003"`` -> ``"vqe_su2"``."""
    return circuit_name.split("/", 1)[0]


def num_qubits_of(circuit_name: str) -> int | None:
    """Qubit count encoded in the circuit file name, or ``None`` if absent."""
    match = _QUBIT_RE.search(circuit_name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class Split:
    """One train/test partition of the dataset.

    Attributes:
        name: Short identifier used for result directories (e.g. ``vqe_su2``).
        kind: ``"family_holdout"`` | ``"size_holdout"`` | ``"random"``.
        train: Dataset positions used for training (and validation).
        test: Dataset positions held out for testing.
        description: Human-readable summary recorded in the result files.
    """

    name: str
    kind: str
    train: tuple[int, ...]
    test: tuple[int, ...]
    description: str

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(
                f"Split '{self.name}' has {len(overlap)} circuits in both train and test."
            )
        if not self.train or not self.test:
            raise ValueError(f"Split '{self.name}' has an empty train or test side.")


class GraphDataset:
    """In-memory view of the pre-built graph dataset with split helpers."""

    def __init__(self, data_list: Sequence[Any], names: Sequence[str]) -> None:
        if len(data_list) != len(names):
            raise ValueError(
                f"{len(data_list)} graphs but {len(names)} names — must be aligned."
            )
        self.data: list[Any] = list(data_list)
        self.names: list[str] = [str(n) for n in names]
        self.families: list[str] = [family_of(n) for n in self.names]
        self.qubits: list[int | None] = [num_qubits_of(n) for n in self.names]

        missing = sum(1 for q in self.qubits if q is None)
        if missing:
            logger.warning(
                "%d/%d circuit names carry no '_q<N>_' tag; they are excluded from "
                "the size-extrapolation split.", missing, len(self.names),
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def load(
        cls,
        dataset_dir: Path | str,
        figure_of_merit: str = FIGURE_OF_MERIT,
    ) -> "GraphDataset":
        """Load ``graph_dataset_<fom>.pt`` (+ the name list fallback)."""
        dataset_dir = Path(dataset_dir)
        pt_path = dataset_dir / f"graph_dataset_{figure_of_merit}.pt"
        names_path = dataset_dir / f"names_list_{figure_of_merit}.npy"

        logger.info("Loading graph dataset: %s", pt_path)
        data_list = torch.load(pt_path, weights_only=False)
        logger.info("Loaded %d graphs", len(data_list))

        names = [getattr(d, "circuit_name", None) for d in data_list]
        if any(n is None for n in names):
            if not names_path.is_file():
                raise FileNotFoundError(
                    f"Some graphs lack 'circuit_name' and {names_path} is missing."
                )
            names = [str(n) for n in np.load(names_path, allow_pickle=True)]
            logger.info("Recovered circuit names from %s", names_path)

        return cls(data_list, names)

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def family_sizes(self) -> dict[str, int]:
        """``{family: n_circuits}``, descending by size."""
        return dict(Counter(self.families).most_common())

    def qubit_range(self) -> tuple[int, int]:
        present = [q for q in self.qubits if q is not None]
        return (min(present), max(present)) if present else (0, 0)

    def summary(self) -> dict[str, Any]:
        lo, hi = self.qubit_range()
        return {
            "n_circuits": len(self),
            "n_families": len(set(self.families)),
            "qubit_min": lo,
            "qubit_max": hi,
            "family_sizes": self.family_sizes(),
            "devices": list(DEVICE_NAMES),
        }

    def subset(self, indices: Iterable[int]) -> list[Any]:
        return [self.data[i] for i in indices]

    def names_for(self, indices: Iterable[int]) -> list[str]:
        return [self.names[i] for i in indices]

    def targets_for(self, indices: Iterable[int]) -> np.ndarray:
        """Ground-truth fidelity matrix ``(n, 3)`` for the given positions."""
        rows = [self.data[i].y.reshape(-1).numpy() for i in indices]
        return np.asarray(rows, dtype=np.float64)

    # -- splits ------------------------------------------------------------

    def select_families(
        self,
        requested: Sequence[str] | None = None,
        min_size: int = 0,
    ) -> list[str]:
        """Resolve the list of families to hold out.

        ``requested`` may be ``None``/``("large",)`` for :data:`LARGE_FAMILIES`,
        or ``("all",)`` for every family in the dataset.  Families that are not
        present (e.g. categories dropped during DAG pre-processing) or smaller
        than ``min_size`` are reported and skipped rather than failing the run.
        """
        sizes = self.family_sizes()
        if requested is None or tuple(requested) in ((), ("large",)):
            wanted = list(LARGE_FAMILIES)
        elif tuple(requested) == ("all",):
            wanted = list(sizes)
        else:
            wanted = list(requested)

        selected, missing, too_small = [], [], []
        for fam in wanted:
            if fam not in sizes:
                missing.append(fam)
            elif sizes[fam] < min_size:
                too_small.append((fam, sizes[fam]))
            else:
                selected.append(fam)

        if missing:
            logger.warning(
                "Requested families absent from the dataset (skipped): %s",
                ", ".join(missing),
            )
        if too_small:
            logger.warning(
                "Families below --min-family-size=%d (skipped): %s",
                min_size,
                ", ".join(f"{f} ({n})" for f, n in too_small),
            )
        if not selected:
            raise ValueError("No usable families selected for leave-one-family-out.")
        logger.info("Holding out %d families: %s", len(selected), ", ".join(selected))
        return selected

    def family_holdout_split(self, family: str) -> Split:
        """Train on every other family, test on ``family``."""
        test = tuple(i for i, f in enumerate(self.families) if f == family)
        train = tuple(i for i, f in enumerate(self.families) if f != family)
        return Split(
            name=family,
            kind="family_holdout",
            train=train,
            test=test,
            description=(
                f"Leave-one-family-out: test = all {len(test)} '{family}' circuits, "
                f"train = {len(train)} circuits from "
                f"{len(set(self.families)) - 1} other families."
            ),
        )

    def size_holdout_split(self, max_train_qubits: int) -> Split:
        """Train on circuits with ``<= max_train_qubits`` qubits, test on larger ones."""
        train = tuple(
            i for i, q in enumerate(self.qubits) if q is not None and q <= max_train_qubits
        )
        test = tuple(
            i for i, q in enumerate(self.qubits) if q is not None and q > max_train_qubits
        )
        lo, hi = self.qubit_range()
        return Split(
            name=f"qmax_{max_train_qubits}",
            kind="size_holdout",
            train=train,
            test=test,
            description=(
                f"Size extrapolation: train = {len(train)} circuits with "
                f"{lo}-{max_train_qubits} qubits, test = {len(test)} circuits with "
                f"{max_train_qubits + 1}-{hi} qubits."
            ),
        )

    def random_split(
        self,
        test_fraction: float = 0.3,
        seed: int = 5,
        stratify_bins: int = 5,
    ) -> Split:
        """Reproduce the original random split, stratified by fidelity.

        This is the *control* condition: it is the split used to obtain the
        published in-distribution score, trained here with the same code and the
        same hyper-parameters so that the leave-one-family-out and size
        extrapolation numbers can be compared against it directly.
        """
        from sklearn.model_selection import train_test_split

        y_first = self.targets_for(range(len(self)))[:, 0]
        indices = np.arange(len(self), dtype=np.int64)

        # Quantile bins on the first device's fidelity, with singleton bins
        # merged into the nearest populated one (mirrors the original script).
        # Interior edges are de-duplicated: a skewed fidelity distribution can
        # make two quantiles coincide, and np.digitize rejects non-monotonic bins.
        quantiles = np.quantile(y_first, np.linspace(0.0, 1.0, stratify_bins + 1))
        edges = np.unique(quantiles[1:-1])
        n_bins = len(edges) + 1
        strat = np.clip(np.digitize(y_first, edges), 0, n_bins - 1)
        counts = np.bincount(strat, minlength=n_bins)
        remap = np.arange(n_bins)
        populated = [j for j, c in enumerate(counts) if c >= 2]
        for i in np.where(counts < 2)[0]:
            if populated:
                remap[i] = min(populated, key=lambda j: abs(j - i))
        strat = remap[strat]

        train_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=seed, stratify=strat
        )
        return Split(
            name=f"random_seed{seed}",
            kind="random",
            train=tuple(int(i) for i in train_idx),
            test=tuple(int(i) for i in test_idx),
            description=(
                f"Random split stratified by fidelity (test_fraction={test_fraction}, "
                f"seed={seed}): train = {len(train_idx)}, test = {len(test_idx)} "
                "circuits; families and sizes appear on both sides."
            ),
        )

    # -- train/validation sub-split ---------------------------------------

    @staticmethod
    def train_val_indices(
        train: Sequence[int], val_fraction: float, seed: int
    ) -> tuple[list[int], list[int]]:
        """Split the training positions into train/validation.

        The held-out side of a :class:`Split` never enters ``train``, so model
        selection (early stopping) cannot see the unseen family or the larger
        circuits.
        """
        rng = np.random.default_rng(seed)
        perm = np.array(train, dtype=np.int64)
        rng.shuffle(perm)
        n_val = max(1, int(round(len(perm) * val_fraction)))
        return perm[n_val:].tolist(), perm[:n_val].tolist()
