#!/usr/bin/env python3
"""Exhaustive compilation vs the v8 predictor, as a function of the number of devices.

Devices: the 3 original backends plus the 24 calibration variants of v8b (same coupling
maps, different errors), i.e. up to 27 targets.  For a stratified sample of dataset
circuits, already loaded in memory (as in the compile-time benchmark):

  exhaustive(N) = sum over N devices of [preset pass manager L2 run + expected fidelity]
                  (pass managers built once per device, outside the timer)
  predictor(N)  = graph encoding (old: build_qubit_dataset._process minus QASM parsing;
                  new: gsv3.fast_encoding) + one v8a forward pass over the N devices,
                  batch 1, and amortised over batches of 64

Everything single-threaded (torch threads = 1) unless --threads is given.
    python timing_scaling.py [--per-family 10] [--threads 1]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(EVAL / "generalization_v3" / "scripts"))
sys.path.insert(0, str(EVAL / "generalization_v3"))
sys.path.insert(0, str(EVAL / "compile_check"))
sys.path.insert(0, str(CC))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import load_devices, variant_batch  # noqa: E402
from gsv3.fast_encoding import encode  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402
from genstudy.data import DEVICE_NAMES  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

N_DEVICES = (3, 6, 12, 18, 27)


def old_encode(qc):
    """build_qubit_dataset._process without the QASM parsing (3 transpiles + 2 DAGs)."""
    from qiskit import transpile
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    from encoding import create_dag, get_openqasm3_gates
    tc = transpile(qc, optimization_level=0, basis_gates=get_openqasm3_gates())
    x, ei, n = create_dag(tc)
    q2 = transpile(PassManager(RemoveBarriers()).run(tc), optimization_level=0, basis_gates=get_openqasm3_gates())
    dag = circuit_to_dag(q2)
    _ = [[dag.find_bit(q).index for q in nd.qargs] for nd in dag.op_nodes()]
    return x, ei


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--model", default=str(EVAL / "generalization_v8/results/v8a_c001/random_control/random_seed5/model.pth"))
    ap.add_argument("--kinds", default="sinkhorn", help="comma-separated model kinds timed on the same circuits "
                    "(weights from --model, missing heads left random: timing only)")
    ap.add_argument("--largest", type=int, default=0, help="per family, the K circuits with the most gates instead of a random sample")
    ap.add_argument("--compile-devices", type=int, default=27, help="compile on the first K targets only; "
                    "exhaustive(N > K) extrapolated with the mean per target")
    ap.add_argument("--no-old-encoding", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    torch.set_num_threads(args.threads)

    from qiskit import qasm3
    from qiskit.transpiler import generate_preset_pass_manager
    from compile_layouts import fidelity
    import v8b_variants

    # devices: originals + v8b variants, in a fixed order
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    base = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    var = v8b_variants.variants()
    keys = list(base) + [f"{d}/aug{j}" for j in range(8) for d in DEVICE_NAMES]
    backends = {**base, **var}
    pms = {k: generate_preset_pass_manager(optimization_level=2, target=backends[k].target) for k in keys}

    params = load_hparams()
    raw = json.loads((EVAL / "compile_check/results/v8b/variants_raw.json").read_text())
    # originals rebuilt with the same function as the variants (identical to the training
    # graphs, cf. the 'orig graph check' of calibration_shift/run_shift.py), so they batch together
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    all_dev = variant_batch(raw, keys, params["lap_pe"])
    ref = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
    for i in range(3):
        a, b = all_dev.get_example(i), ref.get_example(i)
        assert torch.allclose(a.x, b.x, atol=1e-4) and torch.allclose(a.phys, b.phys, atol=1e-6), DEVICE_NAMES[i]
    models, dev_sets = {}, {}
    state = torch.load(args.model, map_location="cpu")
    for n in N_DEVICES:
        dev_sets[n] = Batch.from_data_list(all_dev.to_data_list()[:n])
        for k in kinds:
            m = build_model(k, params, dev_sets[n], torch.device("cpu"))
            m.load_state_dict(state, strict=False)
            models[k, n] = m.eval()

    # stratified sample of dataset circuits (no grover: not in the dataset)
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    by_fam = defaultdict(list)
    for d in data:
        by_fam[d.circuit_name.split("/")[0]].append(d.circuit_name)
    rng = np.random.default_rng(0)
    if args.largest:
        size = {d.circuit_name: d.x.size(0) for d in data}
        names = sorted(n for v in by_fam.values() for n in sorted(v, key=lambda c: -size[c])[: args.largest])
    else:
        names = sorted(n for v in by_fam.values() for n in rng.choice(v, size=min(args.per_family, len(v)), replace=False))
    print(f"{len(names)} circuits, kinds {kinds}, compile on {args.compile_devices} targets", flush=True)
    qdir = CC / "benchmark_dataset_30k"

    rows, graphs = [], []
    with torch.no_grad():
        for name in names:
            qc = qasm3.loads((qdir / f"{name}.qasm").read_text())
            t_old = float("nan")
            if not args.no_old_encoding:
                t0 = time.perf_counter(); old_encode(qc); t_old = time.perf_counter() - t0
            t0 = time.perf_counter(); x, ei, gq, nq = encode(qc); t_new = time.perf_counter() - t0
            g = Data(x=x, edge_index=ei, gate_qubits=gq, n_qubits=nq, num_nodes=x.size(0))
            graphs.append(g)
            b = Batch.from_data_list([g])
            fwd = {}
            for k_ in kinds:
                for n in N_DEVICES:
                    models[k_, n](b, dev_sets[n])           # warm the shapes
                    t0 = time.perf_counter(); models[k_, n](b, dev_sets[n]); fwd[f"{k_}|{n}"] = time.perf_counter() - t0
            comp = []
            for k in keys[: args.compile_devices]:
                t0 = time.perf_counter()
                tc = pms[k].run(qc)
                fidelity(tc, backends[k].target)
                comp.append(time.perf_counter() - t0)
            rows.append({"circuit": name, "q": int(re.search(r"_q(\d+)_", name).group(1)),
                         "gates": int(x.size(0)), "enc_old": t_old, "enc_new": t_new, "fwd": fwd, "compile": comp})
            print(f"{name:40} gates {x.size(0):6d} enc {t_new*1e3:6.1f} ms  " + "  ".join(
                  f"fwd3 {k_.split('_')[-1]} {fwd[f'{k_}|3']*1e3:6.1f} ms" for k_ in kinds) +
                  f"  compile/device {np.mean(comp)*1e3:8.1f} ms", flush=True)
        batched = {}
        for k_ in kinds:
            for n in N_DEVICES:
                t0 = time.perf_counter()
                for b in DataLoader(graphs, batch_size=64):
                    models[k_, n](b, dev_sets[n])
                batched[f"{k_}|{n}"] = (time.perf_counter() - t0) / len(graphs)

    out = HERE / f"timing_scaling_threads{args.threads}{args.tag}.json"
    out.write_text(json.dumps({"rows": rows, "batched": batched, "keys": keys}))

    enc_new = np.array([r["enc_new"] for r in rows])
    cd = args.compile_devices
    exh_n = lambda r, n: sum(r["compile"][: min(n, cd)]) + max(n - cd, 0) * float(np.mean(r["compile"]))
    print(f"\n{len(rows)} circuits, torch threads {args.threads}; encoding median {np.median(enc_new)*1e3:.1f} ms")
    for k_ in kinds:
        print(f"\n== {k_}")
        print(f"{'devices':>8}{'exhaustive':>13}{'pred b=1':>11}{'pred b=64':>11}{'x (b=1)':>9}{'x (b=64)':>10}{'faster b=1':>12}")
        for n in N_DEVICES:
            exh = np.array([exh_n(r, n) for r in rows])
            p1 = enc_new + np.array([r["fwd"][f"{k_}|{n}"] for r in rows])
            p64 = enc_new + batched[f"{k_}|{n}"]
            print(f"{n:>8}{np.median(exh)*1e3:>11.1f}ms{np.median(p1)*1e3:>9.1f}ms{np.median(p64)*1e3:>9.1f}ms"
                  f"{np.median(exh/p1):>8.1f}x{np.median(exh/p64):>9.1f}x{np.mean(exh > p1):>11.0%}")
        gates = np.array([r["gates"] for r in rows])
        print("  by gates, 3 and 27 devices (batch 1):")
        for lo, hi in ((0, 100), (100, 500), (500, 2000), (2000, 10 ** 9)):
            s_ = (gates >= lo) & (gates < hi)
            if not s_.any():
                continue
            for n in (3, 27):
                exh = np.array([exh_n(r, n) for r in rows])[s_]
                p1 = (enc_new + np.array([r["fwd"][f"{k_}|{n}"] for r in rows]))[s_]
                print(f"  gates {lo:>5}-{hi if hi < 10**9 else 'max':<5} ({int(s_.sum()):3d}) N={n:>2}: exhaustive {np.median(exh)*1e3:9.1f} ms"
                      f"  predictor {np.median(p1)*1e3:7.1f} ms  x{np.median(exh/p1):6.1f}")


if __name__ == "__main__":
    main()
