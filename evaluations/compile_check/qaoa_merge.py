#!/usr/bin/env python3
"""Merge the regenerated qaoa data (with final measurements) into every file the models are trained on.

Inputs: results/qaoa_meas/{layouts, v8b_orig, v8d, v8u_orig, v8u, esp_labels, struct_labels, struct_labels_v2}
(produced by compile_layouts.py / v8b_variants.py / esp_labels.py / struct_labels.py with --families qaoa).
Every target file is first copied to ``<name>_nomeas<ext>`` (once; never overwritten), then its qaoa entries are
replaced; non-qaoa entries are untouched.  The graph dataset's qaoa graphs are rebuilt from the new QASM with the
same steps as the original pipeline (create_pt_file_new.py + build_qubit_dataset.py), with
y = the expected fidelity of the new seed-0 level-2 compilation (results/qaoa_meas/layouts).
    python qaoa_merge.py check     # what would change, nothing written
    python qaoa_merge.py apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
R = HERE / "results/qaoa_meas"
sys.path.insert(0, str(EVAL / "generalization_v3"))
sys.path.insert(0, str(EVAL / "generalization_v3/scripts"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
QASM = Path.home() / "compileCircuits/benchmark_dataset_30k"

# target file -> new qaoa-only file
TARGETS = {
    HERE / "results/compiler_layouts_v6pilot.pt": R / "layouts/compiler_layouts.pt",
    HERE / "results/compiler_counts_v8.pt": R / "layouts/compiler_counts.pt",
    HERE / "results/v8b_orig/v8b_orig_labels.pt": R / "v8b_orig/v8b_orig_labels.pt",
    HERE / "results/v8d/v8d_labels.pt": R / "v8d/v8d_labels.pt",
    HERE / "results/esp_labels.pt": R / "esp_labels.pt",
    HERE / "results/struct_labels.pt": R / "struct_labels_v2.pt",          # first 4 columns (tmax, mmax, omax, active)
    HERE / "results/struct_labels_v2.pt": R / "struct_labels_v2.pt",
    HERE / "results/v8u_orig/usage_labels.pt": R / "v8u_orig/usage_labels.pt",
    HERE / "results/v8u/usage_labels.pt": R / "v8u/usage_labels.pt",
}
V8B = HERE / "results/v8b/v8b_labels.pt"          # = the first 8 v8d variants (same definitions, seeds, k)
DATASET = paths.DATA_DIR / "graph_dataset_expected_fidelity.pt"


def backup(p: Path):
    b = p.with_name(p.stem + "_nomeas" + p.suffix)
    if not b.exists():
        shutil.copy2(p, b)
    return b


def is_qaoa(n: str) -> bool:
    return n.startswith("qaoa/")


def new_graph(old, F):
    """Rebuild one qaoa graph from its new QASM as the original pipeline did."""
    from build_qubit_dataset import _process
    from mqt.predictor.ml.helper import create_dag, create_feature_vector, get_openqasm3_gates
    from qiskit import qasm3, transpile
    name = old.circuit_name
    path = QASM / f"{name}.qasm"
    _, x, gq, nq, err = _process((name, str(path)))
    if err:
        raise RuntimeError(f"{name}: {err}")
    circ = qasm3.load(str(path))
    tcirc = transpile(circ, optimization_level=0, basis_gates=get_openqasm3_gates())
    x2, ei, n_nodes = create_dag(tcirc)
    assert torch.equal(torch.as_tensor(x2), torch.as_tensor(x)), name
    g = old.clone()
    g.x, g.edge_index, g.gate_qubits, g.n_qubits, g.num_nodes = torch.as_tensor(x), torch.as_tensor(ei), gq, nq, n_nodes
    g.y = torch.tensor([F], dtype=torch.float32)
    g.scores_dict = dict(zip(DEVICES, map(float, F)))
    g.target_label = DEVICES[int(np.argmax(F))]
    g.feature_vec = torch.tensor(create_feature_vector(circ), dtype=torch.float32).unsqueeze(0)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "apply"])
    ap.add_argument("--stage", choices=["A", "B", "all"], default="all",
                    help="A: everything from-scratch training needs (dataset, layouts, counts, orig labels, ESP, structure, "
                         "orig usage); B: the calibration variants (v8d, v8b, variant usage maps)")
    args = ap.parse_args()
    VARIANT_FILES = {"v8d_labels.pt"}
    def in_stage(tgt):
        var = tgt.name in VARIANT_FILES or (tgt.name == "usage_labels.pt" and tgt.parent.name == "v8u")
        return args.stage == "all" or (args.stage == "B") == var
    apply = args.cmd == "apply"
    # label files
    for tgt, src in TARGETS.items():
        if not in_stage(tgt):
            continue
        if not src.is_file():
            print(f"MISSING new file {src.relative_to(HERE)} -> {tgt.name} not updated")
            continue
        old, new = torch.load(tgt, weights_only=False), torch.load(src, weights_only=False)
        if tgt.name == "struct_labels.pt":
            new = {k: v[:, :4] for k, v in new.items()}
        q_old = [k for k in old if is_qaoa(k)]
        q_new = [k for k in new if is_qaoa(k)]
        if tgt.name == "usage_labels.pt":          # usage maps exist only on the pilot subset: keep the same circuits
            q_new = [k for k in q_new if k in old]
        missing = sorted(set(q_old) - set(q_new))
        print(f"{tgt.relative_to(HERE)}: {len(old)} entries, qaoa old {len(q_old)} new {len(q_new)}, old qaoa without a new "
              f"entry {len(missing)} (dropped)")
        if apply:
            backup(tgt)
            out = {k: v for k, v in old.items() if not is_qaoa(k)}
            out.update({k: new[k] for k in q_new})
            torch.save(out, tgt)
    # v8b labels = first 8 v8d variants
    d = torch.load(R / "v8d/v8d_labels.pt", weights_only=False) if (R / "v8d/v8d_labels.pt").is_file() and args.stage != "A" else None
    if d is not None:
        old = torch.load(V8B, weights_only=False)
        print(f"v8b/v8b_labels.pt: qaoa old {sum(map(is_qaoa, old))} new {sum(map(is_qaoa, d))} (first 8 v8d variants)")
        if apply:
            backup(V8B)
            out = {k: v for k, v in old.items() if not is_qaoa(k)}
            out.update({k: {"F": v["F"][:8], "layout": v["layout"][:, :8], "counts": v["counts"][:8]} for k, v in d.items() if is_qaoa(k)})
            torch.save(out, V8B)
    # graph dataset
    if args.stage == "B":
        return
    lay = {}
    for f in sorted((R / "layouts").glob("*_task*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r and all(r["fidelity"].get(dv) is not None for dv in DEVICES):
                lay[r["circuit"]] = [float(r["fidelity"][dv]) for dv in DEVICES]
    data = torch.load(DATASET, weights_only=False)
    q_idx = [i for i, g in enumerate(data) if is_qaoa(g.circuit_name)]
    have = [i for i in q_idx if data[i].circuit_name in lay]
    print(f"dataset: {len(data)} graphs, qaoa {len(q_idx)}, with a new label {len(have)}")
    if have:
        g0 = new_graph(data[have[0]], lay[data[have[0]].circuit_name])
        o = data[have[0]]
        print(f"  example {o.circuit_name}: gates {o.x.size(0)} -> {g0.x.size(0)}, y {o.y.tolist()} -> {g0.y.tolist()}")
    if apply:
        backup(DATASET)
        from multiprocessing import Pool
        with Pool(16) as pool:
            new = pool.starmap(new_graph, [(data[i], lay[data[i].circuit_name]) for i in have], chunksize=8)
        for i, g in zip(have, new):
            data[i] = g
        dropped = set(q_idx) - set(have)
        data = [g for i, g in enumerate(data) if i not in dropped]
        torch.save(data, DATASET)
        backup(paths.DATA_DIR / "names_list_expected_fidelity.npy")
        np.save(paths.DATA_DIR / "names_list_expected_fidelity.npy", np.array([g.circuit_name for g in data]))
        print(f"dataset written: {len(data)} graphs ({len(dropped)} qaoa without a new label dropped)")


if __name__ == "__main__":
    main()
