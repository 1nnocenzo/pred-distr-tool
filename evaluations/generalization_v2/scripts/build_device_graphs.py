#!/usr/bin/env python3
"""Build the device graphs (coupling map + calibration) used by the v2 predictor.

Instantiates the EQE1_Top, EQE1_Bottom and QExa20 backends from
``compileCircuits/createDevice.py`` — the same targets the ground-truth
fidelities were computed on — and writes ``results/device_graphs.pt``.

Example::

    python evaluations/generalization_v2/scripts/build_device_graphs.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from gsv2 import paths
from gsv2.devices import build_device_graphs, save_device_graphs

paths.ensure_v1_imports()

from genstudy.data import DEVICE_NAMES  # noqa: E402
from genstudy.io import save_json, setup_logging  # noqa: E402

logger = logging.getLogger("build_device_graphs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--compile-circuits-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=paths.DEVICE_GRAPHS_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging(args.output.parent / "build_device_graphs.log")
    logger.info("Command line: %s", " ".join(sys.argv))

    if args.output.is_file() and not args.overwrite:
        logger.info("%s exists — nothing to do (use --overwrite to rebuild)", args.output)
        return 0

    compile_dir = paths.find_compile_circuits_dir(args.compile_circuits_dir)
    payload = build_device_graphs(compile_dir, DEVICE_NAMES)
    save_device_graphs(payload, args.output)
    save_json(payload["raw"], args.output.with_suffix(".raw.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
