#!/usr/bin/env python3
"""Light Sabre as the routing input of stage 1 (v28): Qiskit's SabreLayout with ONE layout trial, ONE iteration and ONE
routing trial (level-2 init stage before it, as the labels), instead of our greedy router.

Why (2026-10-16): the greedy router is far from Sabre (qaoa: 3.95x Sabre's SWAPs, its duration ~1.8x the compiled one), so
its features carry little routing information.  The light Sabre costs 3.5-6.7 ms per (circuit, device) vs 38-188 ms for
level 2's Sabre (qft / qaoa / randomcircuit, 8-20 qubits) with 0.84-1.58x Sabre's SWAPs.  Declared in the paper as a
cheap approximate routing pass at prediction time; the full compiler's Sabre stays a training label only.
Its SWAPs are attached to the model's nodes with compile_check/sabre_labels.label_case (the code of the labels), so the
features have the format of the targets: SWAPs per operand slot, the routed schedule (t_node) and its duration (replay).
    python sabre_light.py run --task-id 0 --n-tasks 8      # every circuit of stage1.pt (+ stage1_synth2.pt)
    python sabre_light.py merge                            # -> results/sabre_light.pt
    python sabre_light.py check --n 300                    # light vs true Sabre labels and vs the greedy router
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path[:0] = [str(HERE), str(EVAL / "compile_check"), str(EVAL / "generalization_v3"), str(EVAL / "synthetic_routing")]
sys.path.append(str(EVAL.parent))                  # createDevice.py at the repository root when ~/compileCircuits is absent
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
OUT = HERE / "results/sabre_light"
DATA_DIRS = (Path.home() / "compileCircuits/benchmark_dataset_30k", EVAL.parent / "data/benchmark_dataset_30k")


def circuit(name: str):
    """The logical circuit of a stage-1 name, exactly as its graph was made."""
    from qiskit import qasm3
    fam, stem = name.split("/")
    if fam == "synth":
        idx = int(stem.rsplit("_v", 1)[1])
        if idx >= 100_000:
            import synth_layered as gen
        else:
            import synth_routing as gen
        _, qc = gen.make_circuit(idx)
        return qasm3.loads(qasm3.dumps(qc))                     # the generators store and reload the QASM
    for d in DATA_DIRS:
        p = d / f"{name}.qasm"
        if p.is_file():
            qc = qasm3.loads(p.read_text())
            if fam == "qaoa" and "measure" not in qc.count_ops():
                qc.measure_all()                                 # data/circuits.zip has the old qaoa (qaoa_measure.py)
            return qc
    raise FileNotFoundError(f"{name}.qasm not in {[str(d) for d in DATA_DIRS]} (unpack data/circuits.zip)")


_PM: dict = {}


def light_case(qb, nodes, n_log, backend):
    """Light Sabre on one device: (labels {(node, slot): n}, n_swap, aligned, replay µs, t_node {node: (start, end)})."""
    from qiskit.converters import dag_to_circuit
    from qiskit.transpiler import PassManager, StagedPassManager, generate_preset_pass_manager
    from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements, SabreLayout
    from sabre_labels import label_case
    key = id(backend)
    if key not in _PM:                                   # built once per device (building a preset pass manager is slow)
        full = generate_preset_pass_manager(optimization_level=2, target=backend.target, seed_transpiler=0)
        cm = backend.target.build_coupling_map()
        # as level 2's layout stage (barrier before the final measurements, then Sabre), with 1 trial and 1 iteration
        _PM[key] = StagedPassManager(stages=["init", "layout"], init=full.init, layout=PassManager([
            BarrierBeforeFinalMeasurements(), SabreLayout(cm, seed=0, max_iterations=1, layout_trials=1, swap_trials=1)]))
    pm = _PM[key]
    st = {}

    def cb(**kw):
        if type(kw["pass_"]).__name__ == "SabreLayout":
            st["routed"] = dag_to_circuit(kw["dag"])
            st["vbits"] = kw["property_set"]["layout"].get_virtual_bits()
    pm.run(qb, callback=cb)
    idx = {q: i for i, q in enumerate(qb.qubits)}
    init = {idx[v]: p for v, p in st["vbits"].items() if v in idx}
    labels, n_sw, _, ok, _, replay, t_node, _ = label_case(nodes, n_log, st["routed"], init, True)
    return labels, n_sw, ok, replay, t_node


def names_all():
    import torch
    names = list(torch.load(HERE / "results/stage1.pt", weights_only=False))
    s2 = HERE / "results/stage1_synth2.pt"
    if s2.is_file():
        names += list(torch.load(s2, weights_only=False))
    return sorted(set(names))


def run(args) -> None:
    from gsv3 import fast_encoding as fe
    from v8b_variants import variants
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    names = [n for n in names_all() if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t0 = time.time()
    with out_file.open("a") as fh:
        for ci, name in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name, "rows": {}}
            try:
                qb = fe.to_basis(circuit(name))
                nodes = [(inst.operation.name, [qb.find_bit(q).index for q in inst.qubits])
                         for inst in qb.data if inst.operation.name != "barrier"]
                rec["n_nodes"] = len(nodes)
                for d in DEVS:
                    ts = time.perf_counter()
                    labels, n_sw, ok, replay, t_node = light_case(qb, nodes, qb.num_qubits, base[d])
                    rec["rows"][d] = {"swaps": [[k[0], k[1], v] for k, v in sorted(labels.items())], "n_swap": n_sw,
                                      "aligned": ok, "replay": replay, "t": time.perf_counter() - ts,
                                      "t_node": [[k, round(v[0], 3), round(v[1], 3)] for k, v in sorted(t_node.items())]}
            except Exception as e:                                         # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if ci % 200 == 0:
                print(f"[{time.time() - t0:6.0f} s] {ci}/{len(names)} {name}", flush=True)


def merge(args) -> None:
    """-> results/sabre_light.pt {circuit: {"n_nodes", "s": (D, N, 3), "replay": (D,), "t_node": [D lists], "t": (D,)}}"""
    import torch
    out, err = {}, 0
    for f in sorted(OUT.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if "error" in r or len(r["rows"]) != len(DEVS):
                err += 1
                continue
            N = r["n_nodes"]
            s = np.zeros((len(DEVS), N, 3), np.float32)
            for di, d in enumerate(DEVS):
                for node, slot, n in r["rows"][d]["swaps"]:
                    s[di, node, slot] = n
            out[r["circuit"]] = {"n_nodes": N, "s": torch.from_numpy(s),
                                 "replay": torch.tensor([r["rows"][d]["replay"] for d in DEVS], dtype=torch.float),
                                 "t_node": [r["rows"][d]["t_node"] for d in DEVS],
                                 "t": torch.tensor([r["rows"][d]["t"] for d in DEVS], dtype=torch.float)}
    torch.save(out, HERE / "results/sabre_light.pt")
    tt = torch.cat([v["t"] for v in out.values()]) if out else torch.zeros(1)
    print(f"{len(out)} circuits ({err} failed) -> {HERE / 'results/sabre_light.pt'}; light Sabre time per (circuit, device): "
          f"median {tt.median().item() * 1e3:.1f} ms, p90 {tt.quantile(0.9).item() * 1e3:.1f} ms")


def check(args) -> None:
    """Light Sabre vs the true Sabre labels (stage1.pt) and vs the greedy prior, on routed cases of a sample."""
    import random
    import torch
    s1 = torch.load(HERE / "results/stage1.pt", weights_only=False)
    light = torch.load(HERE / "results/sabre_light.pt", weights_only=False)
    names = [n for n in sorted(light) if n in s1]
    random.Random(0).shuffle(names)
    rows = []
    for n in names[: args.n]:
        lab, li = s1[n], light[n]
        if li["n_nodes"] != lab["s"].shape[1]:
            continue
        for d in range(len(DEVS)):
            if bool(lab["rf"][d]) or not bool(lab["aligned"][d]) or not np.isfinite(float(lab["replay"][d])):
                continue
            st = float(lab["s"][d].sum())
            if st < 1:
                continue
            rows.append((n.split("/")[0], float(li["s"][d].sum()) / st, float(lab["prior_s"][d].sum()) / st,
                         float(li["replay"][d]) / float(lab["replay"][d]), float(lab["prior_dur"][d]) / float(lab["replay"][d])))
    fams = sorted({r[0] for r in rows})
    print(f"{len(rows)} routed cases; ratios to the true Sabre labels (median, and median |ratio - 1| for durations)")
    for f in ["all"] + fams:
        sel = [r for r in rows if f == "all" or r[0] == f]
        if len(sel) < 5:
            continue
        a = np.array([r[1:] for r in sel])
        print(f"  {f:16} n={len(sel):5}  SWAPs light {np.median(a[:, 0]):.2f} / greedy {np.median(a[:, 1]):.2f} | "
              f"routed schedule light {np.median(a[:, 2]):.2f} (|err| {np.median(abs(a[:, 2] - 1)):.0%}) / "
              f"greedy {np.median(a[:, 3]):.2f} (|err| {np.median(abs(a[:, 3] - 1)):.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "merge", "check"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    {"run": run, "merge": merge, "check": check}[a.cmd](a)
