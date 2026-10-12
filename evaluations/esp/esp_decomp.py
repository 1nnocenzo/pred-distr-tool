#!/usr/bin/env python3
"""How much of the compiled gate counts is a fixed per-gate decomposition, and how much is left to learn?

1. Table: every gate type of the encoding, alone, transpiled at level 2 on adjacent qubits of EQE1_Top (no
   routing, generic angles) -> native 1-qubit (r) and CZ count per gate type.
2. Per circuit, Σ of the table over its gates (r_dec, cz_dec) vs the true compiled counts (compiler_counts_v8.pt):
   per family the merge factor true / table (routing-free circuits: no SWAPs), its spread across families, the
   variance of log true counts the table explains, and the held-out rule  r ≈ r_dec × (median factor of the other
   families)  per family (what a fixed table + one learned factor would give, against the model's count error).
3. Routed circuits: the share of CZ that routing adds on top of the decomposition.
    python esp_decomp.py       # -> results/report_esp_decomp.md (+ decomp_table.json)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
sys.path.insert(0, str(EVAL / "compile_check"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3 import fast_encoding as fe  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import GATE2IDX_MEASURE, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

U = list(fe.UNIQUE)
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")


def table():
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import get_standard_gate_name_mapping
    from v8b_variants import variants
    be = {k.split("/")[0]: v for k, v in variants(orig=True).items()}["EQE1_Top"]
    target = be.target
    cm = target.build_coupling_map()
    nb = defaultdict(set)
    for x, y in cm.get_edges():
        nb[x].add(y); nb[y].add(x)
    b = next(q for q in nb if len(nb[q]) >= 2)                     # a qubit with two neighbours: path a - b - c
    a, c = sorted(nb[b])[:2]
    path = [a, b, c]
    mapping = get_standard_gate_name_mapping()
    rng = np.random.default_rng(0)
    out = {}
    for name in U:
        if name in ("measure", "if_else", "barrier") or name not in mapping:
            continue
        g = mapping[name]
        if g.params:
            g = g.__class__(*[float(x) for x in rng.uniform(0.3, 2.8, len(g.params))])
        n = g.num_qubits
        qc = QuantumCircuit(n)
        qc.append(g, list(range(n)))
        try:
            t = transpile(qc, target=target, optimization_level=2, initial_layout=path[:n], seed_transpiler=0)
        except Exception as e:                                      # noqa: BLE001
            print("skip", name, e)
            continue
        ops = t.count_ops()
        cz = int(ops.get("cz", 0))
        r = int(sum(v for k, v in ops.items() if k not in ("cz", "measure", "barrier", "delay")))
        out[name] = {"r": r, "cz": cz, "qubits": n}
    return out


def main():
    tab = table()
    (HERE / "results/decomp_table.json").write_text(json.dumps(tab, indent=1))
    print(json.dumps(tab))
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    cnt = torch.load(EVAL / "compile_check/results/compiler_counts_v8.pt", weights_only=False)
    data = [g for g in data if g.circuit_name in cnt]
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    params = load_hparams()
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    rows, cache = [], {}
    for s in range(0, len(data), 256):
        gs = data[s: s + 256]
        rf = routing_free(next(iter(DataLoader(gs, batch_size=len(gs)))), dev, cache).numpy()
        for g, f in zip(gs, rf):
            oh = g.x[:, :len(U)].numpy()
            r_dec, cz_dec = float((oh @ tr).sum()), float((oh @ tc).sum())
            c = cnt[g.circuit_name].numpy()[0]                     # EQE1_Top
            rows.append((g.circuit_name.split("/")[0], f[0] > 0.5, r_dec, cz_dec, c[0], c[2]))
    fam = np.array([r[0] for r in rows]); rf = np.array([r[1] for r in rows])
    r_dec = np.array([r[2] for r in rows]); cz_dec = np.array([r[3] for r in rows])
    r_t = np.array([r[4] for r in rows]); cz_t = np.array([r[5] for r in rows])
    ok = (r_dec > 0) & (r_t > 0)
    L = ["# Fixed per-gate decomposition vs the true compiled counts (EQE1_Top, no model)", "",
         "Table: each gate type alone, level 2, adjacent qubits, generic angles (results/decomp_table.json). "
         "Merge factor = true / Σ table over the circuit's gates.", ""]
    rf_ok = ok & rf
    lr, ld = np.log(r_t[rf_ok]), np.log(r_dec[rf_ok])
    r2 = lambda p, y: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    k = np.median(lr - ld)
    L += [f"Routing-free circuits: {int(rf_ok.sum())}. Variance of log true r explained by log table r (one global factor): "
          f"R² {r2(ld + k, lr):.3f}; CZ: R² {r2(np.log(cz_dec[rf_ok & (cz_dec > 0) & (cz_t > 0)] + 0) + np.median(np.log(cz_t[rf_ok & (cz_dec > 0) & (cz_t > 0)]) - np.log(cz_dec[rf_ok & (cz_dec > 0) & (cz_t > 0)])), np.log(cz_t[rf_ok & (cz_dec > 0) & (cz_t > 0)])):.3f}", "",
          "| family | routing-free circuits | r merge factor, median [p10, p90] | CZ true / table, median | held-out rule: r error | table alone: r error |",
          "| --- | --- | --- | --- | --- | --- |"]
    for f_ in sorted(set(fam[rf_ok])):
        te = rf_ok & (fam == f_)
        if te.sum() < 10:
            continue
        others = rf_ok & (fam != f_)
        fac_o = np.median(r_t[others] / r_dec[others])
        fac = r_t[te] / r_dec[te]
        cz_sel = te & (cz_dec > 0)
        czf = f"{np.median(cz_t[cz_sel] / cz_dec[cz_sel]):.2f}" if cz_sel.sum() else "–"
        L.append(f"| {f_} | {int(te.sum())} | {np.median(fac):.2f} [{np.percentile(fac, 10):.2f}, {np.percentile(fac, 90):.2f}] | {czf} | "
                 f"{np.median(np.abs(r_dec[te] * fac_o / r_t[te] - 1)):.2f} | {np.median(np.abs(r_dec[te] / r_t[te] - 1)):.2f} |")
    ro = ok & ~rf & (cz_dec > 0)
    L += ["", "## Routed circuits: CZ added by routing", "", "| family | routed circuits | true CZ / table CZ, median [p10, p90] |", "| --- | --- | --- |"]
    for f_ in sorted(set(fam[ro])):
        te = ro & (fam == f_)
        if te.sum() < 10:
            continue
        q = cz_t[te] / cz_dec[te]
        L.append(f"| {f_} | {int(te.sum())} | {np.median(q):.2f} [{np.percentile(q, 10):.2f}, {np.percentile(q, 90):.2f}] |")
    (HERE / "results/report_esp_decomp.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
