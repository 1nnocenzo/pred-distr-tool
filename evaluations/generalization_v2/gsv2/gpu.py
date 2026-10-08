"""Device selection for the v2 entry points: the GPU when usable, the CPU otherwise."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def select_device() -> torch.device:
    """Return ``cuda:0`` if a working GPU is present, ``cpu`` otherwise.

    ``torch.cuda.is_available()`` alone is not enough: a device can be listed
    but fail on first use, so a tiny kernel is run as well.
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA device available — running on the CPU.")
        return torch.device("cpu")
    try:
        probe = torch.ones(8, device="cuda") * 2
        torch.cuda.synchronize()
        assert float(probe.sum()) == 16.0
    except Exception:
        logger.exception("The CUDA device is listed but unusable — running on the CPU.")
        return torch.device("cpu")
    device = torch.device("cuda:0")
    logger.info("GPU: %s", torch.cuda.get_device_name(device))
    return device
