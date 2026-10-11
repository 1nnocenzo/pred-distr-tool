#!/usr/bin/env python3
"""Inference cost of the usage-map head (v11) vs v8kd40, and where v11 spends its time.

Same circuits and devices as timing_scaling.py (its JSON gives the per-device compile times):
forward pass per circuit at batch 1 and amortised at batch 64, for N = 3 … 27 devices,
1 thread; speed-up = exhaustive compile + fidelity over N devices / (encoding + forward).
Then a torch-profiler table of one training step (batch 32, forward + backward) per model.
    python timing_v11.py [--threads 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import load_devices, variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402
from genstudy.data import DEVICE_NAMES  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

ROOT = EVAL / "generalization_v9/results"
MODELS = {"v8kd40": ("sinkhorn_rf_route_czc", "v8kd40"), "v11": ("sinkhorn_rf_route_usage", "v11")}
NS = (3, 6, 12, 18, 27)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ref = json.loads((HERE / "timing_scaling_threads1.json").read_text())
    params = load_hparams()
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    rows = [r for r in ref["rows"] if r["circuit"] in data]
    raw = json.loads((EVAL / "compile_check/results/v8b/variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    keys = ref["keys"]
    dev_n = {n: variant_batch(raw, keys[:n], params["lap_pe"]) for n in NS}
    dev3 = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
    out = {"n_circuits": len(rows)}
    for name, (kind, run) in MODELS.items():
        state = torch.load(ROOT / run / "random_control/random_seed5/model.pth", map_location="cpu")
        graphs = [data[r["circuit"]] for r in rows]
        res = {}
        with torch.no_grad():
            for n in NS:
                model = build_model(kind, params, dev_n[n], torch.device("cpu"))   # hop matrices per device set
                model.load_state_dict(state); model.eval()
                model(next(iter(DataLoader(graphs[:2], batch_size=2))), dev_n[n])   # warm caches
                t = []
                for g in graphs:
                    b = next(iter(DataLoader([g], batch_size=1)))
                    t0 = time.perf_counter(); model(b, dev_n[n]); t.append(time.perf_counter() - t0)
                bt = []
                for b in DataLoader(graphs, batch_size=64):
                    t0 = time.perf_counter(); model(b, dev_n[n]); bt.append((time.perf_counter() - t0) / b.num_graphs)
                comp = np.array([sum(r["compile"][:n]) for r in rows])
                enc = np.array([r["enc_new"] for r in rows])
                res[n] = {"fwd_ms": 1e3 * float(np.median(t)), "batched_ms": 1e3 * float(np.mean(bt)),
                          "speedup_median": float(np.median(comp / (enc + np.array(t)))),
                          "compile_ms": 1e3 * float(np.median(comp))}
                print(f"{name:7} N={n:2d}: forward {res[n]['fwd_ms']:7.1f} ms (batch 64: {res[n]['batched_ms']:5.1f} ms/circuit) | "
                      f"compile {res[n]['compile_ms']:8.1f} ms | speed-up {res[n]['speedup_median']:5.1f}x", flush=True)
        out[name] = res
    # one training step per model, profiled
    from torch.profiler import ProfilerActivity, profile
    batch = next(iter(DataLoader([data[r["circuit"]] for r in rows[:32]], batch_size=32)))
    for name, (kind, run) in MODELS.items():
        model = build_model(kind, params, dev3, torch.device("cpu")); model.train()
        model(batch, dev3).sum().backward()          # warm-up (path cache, routing caches)
        t0 = time.perf_counter()
        for _ in range(3):
            model.zero_grad(); model(batch, dev3).sum().backward()
        step = (time.perf_counter() - t0) / 3
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            model.zero_grad(); model(batch, dev3).sum().backward()
        print(f"\n== {name}: training step (batch 32, fwd+bwd) {step * 1e3:.0f} ms")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=14))
        import cProfile, io, pstats
        pr = cProfile.Profile(); pr.enable()
        model.zero_grad(); model(batch, dev3).sum().backward()
        pr.disable(); sio = io.StringIO()
        pstats.Stats(pr, stream=sio).sort_stats("cumulative").print_stats(18)
        print("\n".join(l for l in sio.getvalue().splitlines() if "/gsv3/" in l or "cumtime" in l))
        out[name]["train_step_ms"] = step * 1e3
    (HERE / f"timing_v11_threads{args.threads}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
