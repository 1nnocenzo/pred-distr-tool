"""
QMS — Quantum Meta-Scheduler (minimal extract).

Only the policy engine is included here: the ``MultiDevicePolicy``
interface, the round-robin baseline, and the policy registry.  This is
the subset needed to run the dry-run device-selection experiments in
``evaluations/pipeline``; the full dispatcher, backend adapters, queue
store, and monitor live in the original QMS repository.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
