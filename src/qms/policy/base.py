"""
Base classes for scheduling policies.

``MultiDevicePolicy`` — primary interface.
    select(MultiDeviceSnapshot) -> dict[str, list[str]]
    Selects which device each circuit goes to AND orders circuits within each
    device.  Device selection from a unified queue is the core contribution;
    within-device ordering is a secondary concern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeviceState:
    """Runtime state of one backend device.

    Attributes:
        running_count: Circuits currently executing on this device.
        max_concurrent: Maximum concurrent circuits this device supports.
        calibration: Latest calibration data from the backend.
    """

    running_count: int = 0
    max_concurrent: int = 1
    calibration: dict[str, Any] = field(default_factory=dict)

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrent - self.running_count)


@dataclass(frozen=True)
class MultiDeviceSnapshot:
    """Immutable snapshot of a unified circuit queue across all devices.

    The pending queue is shared — no circuit is pre-assigned to a device.
    The policy decides both *which device* each circuit goes to and *in what
    order* it is submitted on that device.

    Attributes:
        pending: Unified (job_id, metadata) queue ordered by arrival time.
        devices: Per-device state keyed by device name.
    """

    pending: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    devices: dict[str, DeviceState] = field(default_factory=dict)

    @property
    def total_available_slots(self) -> int:
        return sum(d.available_slots for d in self.devices.values())

    def available_devices(self) -> list[str]:
        """Device names that have at least one free slot."""
        return [name for name, state in self.devices.items() if state.available_slots > 0]


class MultiDevicePolicy(ABC):
    """Policy whose primary role is device selection across a unified queue.

    ``select()`` returns a dict mapping each device name to an ordered list of
    job IDs to dispatch on that device.

    Contracts:
    - Each job ID appears in at most one device's list.
    - Per-device list length <= that device's ``available_slots``.
    - Total dispatched <= ``snapshot.total_available_slots``.

    Within-device ordering (secondary role) is expressed through the order of
    job IDs in each device's list.
    """

    def configure(self, config: dict[str, Any]) -> None:
        """Accept policy-specific configuration before the dispatch loop."""

    @abstractmethod
    def select(self, snapshot: MultiDeviceSnapshot) -> dict[str, list[str]]:
        """Choose which circuits to dispatch and on which device.

        Args:
            snapshot: Unified queue and all device states.

        Returns:
            {device_name: [job_id, ...]} — ordered dispatch lists per device.
        """
        ...
