#!/usr/bin/env python3
"""Does the compiled 1-qubit count follow the circuit's runs of 1-qubit gates?  (reverse engineering, no model)

Level-2 compilation merges consecutive 1-qubit gates on a wire.  Per circuit, from the logical gate graph:
  runs      maximal runs of 1-qubit gates on a wire (between multi-qubit gates / the wire's ends)
  n1q       1-qubit gates;  n2q[type]  multi-qubit gates per type
compared with the true compiled r and CZ counts (compiler_counts_v8.pt, routing-free circuits only, so no SWAPs),
with a least-squares rule fitted on every family but one and tested on the held-out family:
  r  ≈ a · runs + Σ_type b_type · n2q[type]          vs the naive  r ≈ c · n1q + Σ_type b_type · n2q[type]
    python esp_runs.py         # -> results/report_esp_runs.md
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3 import fast_encoding as fe  # noqa: E402
from gsv3.model import GATE2IDX_MEASURE, routing_free  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import load_hparams  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

U = list(fe.UNIQUE)
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")


def features(g):
    x = g.x[:, :len(U)].numpy()
    gq = g.gate_qubits.numpy()
    n_op = (gq >= 0).sum(1)
    meas = x[:, GATE2IDX_MEASURE] > 0.5
    one = (n_op == 1) & ~meas
    src, dst = g.edge_index.numpy()
    # a 1q gate starts a run unless one of its predecessors on its wire is a 1q gate on the same qubit
    q_of = np.where(one, gq[:, 0], -1)
    starts = one.copy()
    for a, b in zip(src, dst):
        if one[b] and one[a] and q_of[a] == q_of[b]:
            starts[b] = False
    two_types = np.zeros(len(U))
    multi = n_op >= 2
    two_types += x[multi].sum(0)
    return float(starts.sum()), float(one.sum()), two_types


def main():
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    cnt = torch.load(EVAL / "compile_check/results/compiler_counts_v8.pt", weights_only=False)
    params = load_hparams()
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    rows = []
    cache = {}
    data = [g for g in data if g.circuit_name in cnt]
    for s in range(0, len(data), 256):
        gs = data[s: s + 256]
        rf = routing_free(next(iter(DataLoader(gs, batch_size=len(gs)))), dev, cache).numpy()
        for g, f in zip(gs, rf):
            if f[0] < 0.5:                       # EQE1_Top routing-free: no SWAPs in its compiled counts
                continue
            runs, n1, two = features(g)
            c = cnt[g.circuit_name].numpy()[0]
            rows.append((g.circuit_name.split("/")[0], runs, n1, two, c[0], c[2]))
    fam = np.array([r[0] for r in rows])
    runs = np.array([r[1] for r in rows]); n1 = np.array([r[2] for r in rows])
    two = np.stack([r[3] for r in rows]); r_true = np.array([r[4] for r in rows]); cz_true = np.array([r[5] for r in rows])
    used = two.sum(0) > 0
    two = two[:, used]
    L = ["# Compiled 1-qubit count vs runs of 1-qubit gates (routing-free circuits on EQE1_Top, no model)", "",
         f"{len(rows)} circuits. Rule fitted on all families but the held-out one (least squares, no intercept).", "",
         "| held-out family | n | true r / runs (median) | true r / 1q gates (median) | error, runs rule | error, naive rule |",
         "| --- | --- | --- | --- | --- | --- |"]
    for f_ in sorted(set(fam)):
        te = fam == f_
        if te.sum() < 20:
            continue
        tr = ~te
        out = []
        for base in (runs, n1):
            A = np.column_stack([base, two])
            w, *_ = np.linalg.lstsq(A[tr], r_true[tr], rcond=None)
            pred = A[te] @ w
            out.append(float(np.median(np.abs(pred / np.maximum(r_true[te], 1) - 1))))
        L.append(f"| {f_} | {int(te.sum())} | {np.median(r_true[te] / np.maximum(runs[te], 1)):.2f} | "
                 f"{np.median(r_true[te] / np.maximum(n1[te], 1)):.2f} | {out[0]:.2f} | {out[1]:.2f} |")
    (HERE / "results/report_esp_runs.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
