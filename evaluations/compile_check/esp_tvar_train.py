#!/usr/bin/env python3
"""Training labels for v18: ESP decay and duration of every circuit on random T1 / T2 and gate-duration variants.

esp_tvariants.py showed that Qiskit level 2 never moves the layout or the counts when only T1, T2 or the gate
durations change (457 circuits × 3 devices × 12 variants, 100%), and that rescheduling the original compilation
gives exactly the variant's ESP and duration.  So each label here is the stored original compilation (seed 0, as
esp_labels.py) rescheduled on the variant: no recompilation.  F is unchanged by construction.

Variant j of device d (seed 70 000 + 1 000·d + j, disjoint from the test variants of esp_tvariants.py):
  T1, T2 × lognormal(σ ~ U(0.1, 0.4)) per qubit, and with p 0.3 two or three qubits × U(0.15, 0.4);
  every r / cz / measure duration × lognormal(σ ~ U(0.05, 0.3)) per qubit or coupler; with p 0.3 three to five
  couplers with the CZ × U(1.5, 3); with p 0.3 every measurement × U(0.7, 2).
    python esp_tvar_train.py run --task-id T --n-tasks N          # -> results/esp_tvar_train/task*.jsonl
    python esp_tvar_train.py merge                                # -> esp_tvar_labels.pt + tvar_raw.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "compileCircuits"))
sys.path.insert(0, str(HERE))
OUT = HERE / "results/esp_tvar_train"
QPY_DIRS = (HERE / "results/layouts_all/qpy", HERE / "results/qaoa_meas/layouts/qpy")   # qaoa: with measurements
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
DYNAMIC = {"iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code", "grover"}
N_VAR = 16
T_FIX = {"QExa20": 10.0}


def make_train_tvar(backend, j: int, dev_index: int):
    from esp_tvariants import _qprops, _scale_dur, _scale_t
    be = copy.deepcopy(backend)
    t = _qprops(be)
    rng = np.random.default_rng(70_000 + 1_000 * dev_index + j)
    nq = t.num_qubits
    s = rng.uniform(0.1, 0.4)
    fac = {q: float(np.exp(rng.normal(0, s))) for q in range(nq)}
    if rng.random() < 0.3:
        for q in rng.choice(nq, size=int(rng.integers(2, 4)), replace=False):
            fac[int(q)] *= float(rng.uniform(0.15, 0.4))
    _scale_t(t, fac)
    s = rng.uniform(0.05, 0.3)
    for g in ("r", "cz", "measure"):
        _scale_dur(t, g, {q: float(np.exp(rng.normal(0, s))) for q, p in t[g].items() if p is not None})
    if rng.random() < 0.3:
        cz = [q for q, p in t["cz"].items() if p is not None]
        pick = rng.choice(len(cz), size=int(rng.integers(3, 6)), replace=False)
        _scale_dur(t, "cz", {cz[i]: float(rng.uniform(1.5, 3.0)) for i in pick})
    if rng.random() < 0.3:
        f = float(rng.uniform(0.7, 2.0))
        _scale_dur(t, "measure", {q: f for q, p in t["measure"].items() if p is not None})
    return be


def train_tvariants() -> dict:
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    from esp_tvariants import _qprops
    base = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    out = {}
    for i, (d, be) in enumerate(base.items()):
        _qprops(be)
        out[f"{d}/orig"] = be
        for j in range(N_VAR):
            out[f"{d}/t{j}"] = make_train_tvar(be, j, i)
    return out


def esp_dur(qc, target) -> tuple[float, float]:
    """(decay = log ESP − log F, duration in s) of a compiled circuit on ``target`` (as esp_labels.measures)."""
    from mqt.predictor.reward import estimated_success_probability, expected_fidelity
    from qiskit import transpile
    f = expected_fidelity(qc, target, precision=300)
    esp = estimated_success_probability(qc, target, precision=300)
    sched = transpile(qc, target=target, scheduling_method="asap", optimization_level=0,
                      initial_layout=None, routing_method=None, layout_method=None)
    decay = math.log(esp) - math.log(f) if esp > 0 and f > 0 else float("nan")
    return decay, float(sched.estimate_duration(target=target))


def _files():
    files = {}
    for root in QPY_DIRS:
        for f in root.rglob("*.qpy"):
            if f.parent.name in DYNAMIC:
                continue
            files[str(f.relative_to(root).with_suffix(""))] = f      # the qaoa_meas copy overrides the old qaoa
    return files


def run(args) -> None:
    from qiskit import qpy
    vs = train_tvariants()
    files = _files()
    names = sorted(n for n in files if zlib.crc32(n.encode()) % args.n_tasks == args.task_id)
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t0 = time.time()
    with out_file.open("a") as fh:
        for i, name in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name}
            try:
                with files[name].open("rb") as q:
                    circs = qpy.load(q)
                assert len(circs) == len(DEVICES), len(circs)
                for d, qc in zip(DEVICES, circs):
                    rec[d] = [esp_dur(qc, vs[f"{d}/{v}"].target) for v in ["orig"] + [f"t{j}" for j in range(N_VAR)]]
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 20 == 0:
                print(f"[{time.time() - t0:6.0f} s] {i}/{len(names)} {name}", flush=True)


def raw_graphs() -> dict:
    """Device graphs of the training variants (T1 / T2 from the target, per-qubit durations as ``t``)."""
    from types import SimpleNamespace
    sys.path.insert(0, str(HERE.parent / "generalization_v3"))
    sys.path.insert(0, str(HERE.parent / "esp"))
    import gsv3  # noqa: F401
    from gsv2.devices import _backend_graph
    from esp_ttrack import _times
    out = {}
    for k, be in train_tvariants().items():
        t = be.target
        shim = SimpleNamespace(target=t, qubit_properties=t.qubit_properties, num_qubits=be.num_qubits,
                               active_qubits=getattr(be, "active_qubits", None))
        out[k] = dict(zip(("nodes", "edges", "edge_feats"), _backend_graph(shim)))
        out[k]["t"] = _times(be, t)
    return out


def merge(args) -> None:
    """-> esp_tvar_labels.pt {circuit: {"decay": (1 + N_VAR, 3), "duration": (1 + N_VAR, 3) µs}} (row 0 = original;
    QExa20's decay divided by T_FIX as esp_labels) and tvar_raw.json (device graphs, keys <dev>/orig, <dev>/t<j>)."""
    import torch
    out, err = {}, 0
    fix = torch.tensor([T_FIX.get(d, 1.0) for d in DEVICES])
    for f in sorted(OUT.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if "error" in r:
                err += 1
                continue
            dec = torch.tensor([[r[d][v][0] for d in DEVICES] for v in range(1 + N_VAR)], dtype=torch.float64)
            dur = torch.tensor([[r[d][v][1] for d in DEVICES] for v in range(1 + N_VAR)], dtype=torch.float64)
            out[r["circuit"]] = {"decay": (dec / fix).float(), "duration": (dur * 1e6).float()}
    torch.save(out, HERE / "results/esp_tvar_labels.pt")
    (HERE / "results/tvar_raw.json").write_text(json.dumps(raw_graphs()))
    print(f"{len(out)} circuits ({err} failed) -> results/esp_tvar_labels.pt, results/tvar_raw.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "merge"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    run(a) if a.mode == "run" else merge(a)
