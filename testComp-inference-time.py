#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GNN-side counterpart of testComp-compilation-time.py: for every circuit in
benchmark_dataset_30k/ (QASM3), record the per-circuit cost of the proposed
fast path — predicting per-device fidelity with the trained GNN instead of
compiling on every backend:

  - encode_time    : wall-clock time to build the feature-annotated DAG
                     (create_dag) and wrap it as a PyG Data object.
  - inference_time : wall-clock time of one GNN forward pass (batch size 1,
                     CPU), matching how the dispatcher invokes the model.

Symmetric with the sister script: qasm3.load is excluded from both timers
(the sister excludes it from compilation_time too), model construction /
checkpoint loading happens once outside the loop (like the prebuilt pass
managers there), and one warm-up forward pass is discarded to absorb
process/JIT startup cost.

Also recorded (excluded from the timers): the predicted per-device fidelities
and num_qubits, so timing_analysis.py can pick the predicted-best device's
compilation time from the sister script's JSON and bin results by width.

Run from the directory containing benchmark_dataset_30k/:
  python testComp-inference-time.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "evaluations" / "pipeline"))

import torch
from torch_geometric.data import Data
from qiskit import qasm3

from gnn_device_policy import _GNNPredictor, DEVICE_NAMES
from encoding import create_dag

# -------------------------------
# Configuration (mirrors sister script)
# -------------------------------

IN_DIR = "data/benchmark_dataset_30k"
RESULTS_JSON = "evaluations/pipeline/timing_results/inference_time_results_benchmark_30k.json"

# Must match testComp-compilation-time.py: a circuit measured on only one side
# of the comparison is dropped by timing_analysis.py anyway.
EXCLUDED_FAMILIES = {"grover"}


def list_qasm_files(folder: str):
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(
        p.relative_to(root) for p in root.rglob("*.qasm")
        if p.relative_to(root).parts[0] not in EXCLUDED_FAMILIES
    )


# -------------------------------
# Main: encode, predict, time, save JSON
# -------------------------------

qasm_files = list_qasm_files(IN_DIR)
if not qasm_files:
    # Do NOT write an empty results file here: the usual cause is running from
    # the wrong working directory, and truncating a completed run to {} loses
    # hours of compute.
    raise SystemExit(
        f"[FATAL] No .qasm files under '{IN_DIR}' (cwd={os.getcwd()}). "
        f"Run from the repository root, with the dataset extracted to '{IN_DIR}'."
    )

# Model built once, outside the loop (analogous to the prebuilt pass managers).
predictor = _GNNPredictor()

# One-time warm-up forward pass (discarded): absorbs torch/JIT startup cost so
# it is not billed to the first circuit. Each circuit below is measured once.
try:
    _qc = qasm3.load(os.path.join(IN_DIR, qasm_files[0]))
    _x, _ei, _ = create_dag(_qc)
    predictor.predict(Data(x=_x, edge_index=_ei))
    del _qc, _x, _ei
    print("[WARMUP] Discarded one warm-up encode+forward pass (process startup).")
except Exception as e:
    print(f"[WARMUP] Skipped warm-up ({e})")

results = {}
total = len(qasm_files)
for idx, rel_path in enumerate(qasm_files, start=1):
    in_path = os.path.join(IN_DIR, rel_path)
    src_tag = str(rel_path.with_suffix(""))

    try:
        circ = qasm3.load(in_path)
    except Exception as e:
        print(f"[SKIP] Failed to load {rel_path}: {e}")
        continue

    try:
        t0 = time.perf_counter()
        node_features, edge_index, _ = create_dag(circ)
        data = Data(x=node_features, edge_index=edge_index)
        encode_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        predicted = predictor.predict(data)
        inference_time = time.perf_counter() - t1
    except Exception as e:
        print(f"[ERR] {idx}/{total} {src_tag}: {e}")
        continue

    results[src_tag] = {
        "predicted": predicted,
        "num_qubits": circ.num_qubits,
        "encode_time": encode_time,
        "inference_time": inference_time,
    }
    print(
        f"[OK] {idx}/{total} {src_tag}: pred_best={DEVICE_NAMES[max(range(len(predicted)), key=predicted.__getitem__)]} "
        f"encode_time={encode_time * 1e3:.3f}ms inference_time={inference_time * 1e3:.3f}ms"
    )

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[DONE] Wrote '{RESULTS_JSON}' with {len(results)} entries on {datetime.now(timezone.utc).isoformat()}Z")
