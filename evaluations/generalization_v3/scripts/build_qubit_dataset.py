#!/usr/bin/env python3
"""Add the raw qubit operands of every gate to the graph dataset.

The original dataset (``compileCircuits/graph_dataset_expected_fidelity.pt``)
keeps, for each gate node, its type, parameters, arity, ... but not *which*
logical qubits it acts on.  That is raw circuit information (no compilation, no
heuristic) and the v3 model needs it to relate the circuit's qubits to the
device's qubits.

For every circuit the script re-runs the exact pre-processing of
``create_pt_file_new.py`` (``qasm3.load`` -> ``transpile(opt=0, openqasm3
basis)`` -> ``create_dag``), checks that the recomputed node features equal the
stored ``x`` (same nodes, same order), and stores:

* ``gate_qubits``  ``(num_nodes, 3)`` long, local logical-qubit indices of each
  gate's operands, ``-1``-padded;
* ``n_qubits``     number of logical qubits of the circuit.

The original dataset file is only read.  The output goes to
``evaluations/generalization_v3/data/`` with the same file names, so v1's
``GraphDataset.load`` reads it unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import _bootstrap  # noqa: F401

from gsv3 import paths  # noqa: F401  (sets sys.path)

import numpy as np
import torch
from qiskit import qasm3, transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from encoding import create_dag, get_openqasm3_gates  # src/model/encoding.py
from genstudy.io import save_json, setup_logging

logger = logging.getLogger("build_qubit_dataset")

MAX_OPERANDS = 3
FOM = "expected_fidelity"


def _process(args: tuple[str, str]):
    """Return ``(name, x, gate_qubits, n_qubits, error)`` for one circuit."""
    name, qasm_path = args
    try:
        circ = qasm3.load(qasm_path)
        # Same two steps as create_pt_file_new.py ...
        tcirc = transpile(circ, optimization_level=0, basis_gates=get_openqasm3_gates())
        x, _, n_nodes = create_dag(tcirc)
        # ... and the same DAG create_dag builds internally, to read the operands.
        qc = PassManager(RemoveBarriers()).run(tcirc)
        qc = transpile(qc, optimization_level=0, basis_gates=get_openqasm3_gates())
        dag = circuit_to_dag(qc)
        nodes = list(dag.op_nodes())
        if len(nodes) != n_nodes:
            return name, None, None, None, f"node count {len(nodes)} != {n_nodes}"
        gq = torch.full((len(nodes), MAX_OPERANDS), -1, dtype=torch.long)
        for i, node in enumerate(nodes):
            idx = [dag.find_bit(q).index for q in node.qargs][:MAX_OPERANDS]
            if len(node.qargs) > MAX_OPERANDS:
                return name, None, None, None, f"gate {node.op.name} on {len(node.qargs)} qubits"
            gq[i, : len(idx)] = torch.tensor(idx, dtype=torch.long)
        return name, x, gq, dag.num_qubits(), None
    except Exception as exc:  # noqa: BLE001
        return name, None, None, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--source-dir", type=Path, default=Path.home() / "compileCircuits")
    p.add_argument("--qasm-dir", type=Path, default=None,
                   help="Default: <source-dir>/benchmark_dataset_30k")
    p.add_argument("--output-dir", type=Path, default=paths.DATA_DIR)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--limit", type=int, default=None, help="Only the first N circuits (smoke test).")
    args = p.parse_args()
    qasm_dir = args.qasm_dir or args.source_dir / "benchmark_dataset_30k"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir / "build_qubit_dataset.log")
    logging.getLogger("qiskit").setLevel(logging.WARNING)

    src = args.source_dir / f"graph_dataset_{FOM}.pt"
    logger.info("Reading %s (read-only)", src)
    data = torch.load(src, weights_only=False)
    if args.limit:
        data = data[: args.limit]
    jobs = [(d.circuit_name, str(qasm_dir / f"{d.circuit_name}.qasm")) for d in data]
    logger.info("%d circuits, %d workers", len(jobs), args.workers)

    results = {}
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_process, jobs, chunksize=16), start=1):
            results[res[0]] = res
            if k % 1000 == 0:
                logger.info("  %d / %d", k, len(jobs))

    out, failures, mismatches = [], {}, []
    for d in data:
        name, x, gq, nq, err = results[d.circuit_name]
        if err is not None:
            failures[name] = err
            continue
        if x.shape != d.x.shape or not torch.allclose(x, d.x, atol=1e-5):
            mismatches.append(name)
            continue
        d = d.clone()
        d.gate_qubits = gq
        d.n_qubits = int(nq)
        out.append(d)

    logger.info("kept %d / %d  (failures %d, feature mismatches %d)",
                len(out), len(data), len(failures), len(mismatches))
    if failures:
        logger.warning("first failures: %s", list(failures.items())[:5])
    if mismatches:
        logger.warning("first mismatches: %s", mismatches[:5])

    torch.save(out, args.output_dir / f"graph_dataset_{FOM}.pt")
    np.save(args.output_dir / f"names_list_{FOM}.npy",
            np.array([d.circuit_name for d in out], dtype=object))
    save_json({"source": str(src), "n_source": len(data), "n_kept": len(out),
               "failures": failures, "feature_mismatches": mismatches},
              args.output_dir / "build_report.json")
    logger.info("Wrote %s", args.output_dir / f"graph_dataset_{FOM}.pt")
    # A dataset that silently lost circuits would change the splits.
    return 0 if not failures and not mismatches else 2


if __name__ == "__main__":
    sys.exit(main())
