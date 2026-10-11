"""Leave-one-group-out (LOGO): hold out a family together with its sibling families.

Leave-one-family-out is optimistic for families that have close relatives in the
training set (the three VQE ansätze, the QFT-based circuits): holding out
``vqe_su2`` still trains on ``vqe_real_amp`` and ``vqe_two_local``.  LOGO holds out
whole groups of structurally related MQT Bench families, so the test measures
generalisation to a genuinely new *kind* of circuit.

Groups (by benchmark construction):

``vqe``          the three VQE ansätze.
``fourier``      everything built on the QFT or on phase estimation: QFT and its
                 variants, the QFT-based arithmetic, QPE (exact, inexact, iterative)
                 and amplitude estimation (which ends with an inverse QFT).
``variational``  ``vqe`` + ``qnn`` (the MQT ``qnn`` uses a RealAmplitudes ansatz,
                 like ``vqe_real_amp``) — the strictest variational hold-out.

``qaoa`` and ``randomcircuit`` have no sibling family, so their LOGO split equals
the LOFO split and is not repeated.
"""

from __future__ import annotations

from . import paths

paths.ensure_imports()

from genstudy.data import GraphDataset, Split  # noqa: E402

GROUPS: dict[str, tuple[str, ...]] = {
    "vqe": ("vqe_two_local", "vqe_real_amp", "vqe_su2"),
    "fourier": ("qft", "qftentangled", "dynamic_qft", "draper_qft_adder",
                "rg_qft_multiplier", "qpeexact", "qpeinexact", "iqpe", "ae"),
    "variational": ("vqe_two_local", "vqe_real_amp", "vqe_su2", "qnn"),
}


def group_holdout_split(dataset: GraphDataset, group: str) -> Split:
    members = set(GROUPS[group])
    present = sorted(members & set(dataset.families))
    test = tuple(i for i, f in enumerate(dataset.families) if f in members)
    train = tuple(i for i, f in enumerate(dataset.families) if f not in members)
    return Split(
        name=group,
        kind="group_holdout",
        train=train,
        test=test,
        description=(
            f"Leave-one-group-out: test = all {len(test)} circuits of group '{group}' "
            f"({', '.join(present)}), train = {len(train)} circuits from the other "
            f"{len(set(dataset.families)) - len(present)} families."
        ),
    )
