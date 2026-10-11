#!/usr/bin/env python3
"""ESP, depth and duration labels from the stored compilations (``layouts_all/qpy``).

Each QPY file holds one circuit compiled at level 2 (seed 0, original calibration) on the three devices
(``DEVICES`` order).  Per (circuit, device), with MQT's functions as the fidelity labels:
  F         expected_fidelity (check against the label pipeline)
  ESP       estimated_success_probability: F × Π_active exp(−idle_q / min(T1_q, T2_q)), ASAP schedule
  depth     circuit depth without barriers;  depth_2q: depth counting only 2-qubit gates
  duration  ASAP-scheduled duration in seconds (r 42 ns, cz 130 ns, measure 15 µs, uniform per device)
  decay     log ESP − log F = −Σ_active idle_q / min(T1_q, T2_q)  (unrounded; ESP stored unrounded too)
Note: QExa20's T1/T2 come from createDevice.py's CSV parsing (raw / 1e4 read as µs → median T1 3.5 µs, 10× below
EQE1); if the raw unit is ns the decay of QExa20 is 10× smaller — it scales exactly as 1 / T, so rescale, no rerun.
    python esp_labels.py run --task-id 0 --n-tasks 16
    python esp_labels.py merge          # -> results/esp_labels.pt {circuit: {key: FloatTensor(3)}}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(CC))
QPY = HERE / "results/layouts_all/qpy"
OUT = HERE / "results/esp_labels"
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
DYNAMIC = {"iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code", "grover"}
# T1/T2 unit fix: createDevice.py reads QExa20's DCDB T1/T2 as 1e-4 µs units, but they are ns (median T1 34.9 µs,
# T2 echo 15.1 µs, T2* 2.9 µs as ns, vs 41 / 18 / 3.9 µs reported for this Q-Exa system in Deng et al.,
# arXiv:2508.21231); the decay scales exactly as 1 / T, so merge divides QExa20's by this factor.
T_FIX = {"QExa20": 10.0}
KEYS = ("F", "ESP", "depth", "depth_2q", "duration", "decay")


def depth_2q(qc) -> int:
    return qc.depth(filter_function=lambda inst: inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


def measures(qc, target) -> dict:
    from mqt.predictor.reward import estimated_success_probability, expected_fidelity
    from qiskit import transpile
    f = expected_fidelity(qc, target)                       # as the labels (rounded to 1e-10)
    f_hp = expected_fidelity(qc, target, precision=300)     # unrounded, for the decay of tiny F
    esp = estimated_success_probability(qc, target, precision=300)
    sched = transpile(qc, target=target, scheduling_method="asap", optimization_level=0,
                      initial_layout=None, routing_method=None, layout_method=None)
    return {"F": f, "ESP": esp, "depth": qc.depth(filter_function=lambda inst: inst.operation.name != "barrier"),
            "depth_2q": depth_2q(qc), "duration": float(sched.estimate_duration(target=target)),
            "decay": (math.log(esp) - math.log(f_hp)) if esp > 0 and f_hp > 0 else float("nan")}


def run(args) -> None:
    from qiskit import qpy
    from v8b_variants import variants
    tg = {}
    for k, be in variants(orig=True).items():
        t = be.target
        if t.qubit_properties is None:       # the backends keep T1/T2 in qubit_properties; MQT's ESP reads the target's
            qp = be.qubit_properties
            t.qubit_properties = list(qp) if not callable(qp) else [qp(q) for q in range(t.num_qubits)]
            assert len(t.qubit_properties) == t.num_qubits, (k, len(t.qubit_properties), t.num_qubits)
        tg[k.split("/")[0]] = t
    files = sorted(QPY.rglob("*.qpy"))
    files = [f for f in files if f.parent.name not in DYNAMIC]
    files = [f for f in files if zlib.crc32(str(f.relative_to(QPY)).encode()) % args.n_tasks == args.task_id]
    if args.limit:
        files = files[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t0 = time.time()
    with out_file.open("a") as fh:
        for i, f in enumerate(files):
            name = str(f.relative_to(QPY).with_suffix(""))
            if name in done:
                continue
            rec = {"circuit": name}
            try:
                with f.open("rb") as q:
                    circs = qpy.load(q)
                assert len(circs) == len(DEVICES), len(circs)
                for d, qc in zip(DEVICES, circs):
                    rec[d] = measures(qc, tg[d])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 100 == 0:
                print(f"[{time.time() - t0:6.0f} s] {i}/{len(files)} {name}", flush=True)


def merge(args) -> None:
    import torch
    out, err = {}, 0
    for f in sorted(OUT.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if "error" in r:
                err += 1
                continue
            rec = {k: torch.tensor([float(r[d][k]) for d in DEVICES]) for k in KEYS}
            fix = torch.tensor([T_FIX.get(d, 1.0) for d in DEVICES])
            rec["decay"] = rec["decay"] / fix
            rec["ESP"] = rec["F"] * torch.exp(rec["decay"])      # with the corrected decay (F as the labels)
            out[r["circuit"]] = rec
    torch.save(out, HERE / "results/esp_labels.pt")
    print(f"{len(out)} circuits ({err} failed) -> {HERE / 'results/esp_labels.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "merge"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a) if a.mode == "run" else merge(a)
