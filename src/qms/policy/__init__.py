"""
Policy engine — pluggable scheduling algorithms for circuit dispatch.

Multi-device policies (device selection from a unified queue):

- ``RoundRobinMultiDevicePolicy``: Cycle circuits across devices.
- ``GNNDevicePolicy``:             GNN-guided device selection (registered
                                   on import by ``gnn_device_policy``).

Custom policies can be created by subclassing ``MultiDevicePolicy`` and
registered with ``register_multi_policy``.
"""

from qms.policy.base import (
    MultiDevicePolicy,
    DeviceState,
    MultiDeviceSnapshot,
)
from qms.policy.multi_round_robin import RoundRobinMultiDevicePolicy

__all__ = [
    "MultiDevicePolicy",
    "DeviceState",
    "MultiDeviceSnapshot",
    "RoundRobinMultiDevicePolicy",
]


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

_BUILTIN_MULTI: dict[str, type[MultiDevicePolicy]] = {
    "round_robin_multi": RoundRobinMultiDevicePolicy,
}

_CUSTOM_MULTI: dict[str, type[MultiDevicePolicy]] = {}


def get_multi_policy(name: str) -> type[MultiDevicePolicy]:
    """Look up a multi-device policy class by name."""
    key = name.lower()
    if key in _CUSTOM_MULTI:
        return _CUSTOM_MULTI[key]
    if key in _BUILTIN_MULTI:
        return _BUILTIN_MULTI[key]
    available = sorted(set(_BUILTIN_MULTI) | set(_CUSTOM_MULTI))
    raise KeyError(f"Unknown multi-device policy '{name}'.  Available: {', '.join(available)}")


def register_multi_policy(name: str, policy_cls: type[MultiDevicePolicy]) -> None:
    """Register a custom multi-device scheduling policy."""
    if not issubclass(policy_cls, MultiDevicePolicy):
        raise TypeError(f"{policy_cls.__name__} must subclass MultiDevicePolicy.")
    _CUSTOM_MULTI[name.lower()] = policy_cls
