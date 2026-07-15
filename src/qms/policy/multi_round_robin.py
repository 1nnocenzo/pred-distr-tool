"""Multi-device Round-Robin policy — cycle circuits across devices."""

from __future__ import annotations

from qms.policy.base import MultiDevicePolicy, MultiDeviceSnapshot


class RoundRobinMultiDevicePolicy(MultiDevicePolicy):
    """Assign circuits to devices in a round-robin cycle, arrival-order first.

    Circuits are taken from the unified queue in arrival order and distributed
    evenly across devices with available slots by cycling through them.
    No fidelity information is used.
    """

    def select(self, snapshot: MultiDeviceSnapshot) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {dev: [] for dev in snapshot.devices}
        available = [dev for dev, state in snapshot.devices.items() if state.available_slots > 0]
        slots = {dev: snapshot.devices[dev].available_slots for dev in available}

        cycle_idx = 0
        for job_id, _ in snapshot.pending:
            if not available:
                break
            dev = available[cycle_idx % len(available)]
            result[dev].append(job_id)
            slots[dev] -= 1
            if slots[dev] == 0:
                available.pop(cycle_idx % len(available))
                # Don't advance index — the next element has shifted into place
                if available:
                    cycle_idx = cycle_idx % len(available)
            else:
                cycle_idx = (cycle_idx + 1) % len(available)

        return result
