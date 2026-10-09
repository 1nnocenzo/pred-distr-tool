"""Inference time of the v8 predictor vs compiling + scoring on every device.

For the circuits of compileCircuits/compilation_time_results_benchmark_30k.json (real
compile_time + fidelity_time per device, preset pass manager L2), measure:
  encode  : QASM -> graph (same steps as build_qubit_dataset._process: load, transpile
            to the OpenQASM3 basis at level 0, DAG features, gate operands)
  forward : v8a model, all 3 devices at once, CPU; one circuit at a time (batch 1)
            and amortised over batches of 64
Compare with the exhaustive path: sum over the 3 devices of compile + fidelity time.
    python timing_v8.py [--threads 1] [--limit N]
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, torch

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL / "generalization_v3" / "scripts"))
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402  (sys.path for src/model, genstudy)
paths.ensure_imports()
import build_qubit_dataset as bqd  # noqa: E402
from gsv3.devices import load_devices  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402
from genstudy.data import DEVICE_NAMES  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default=str(EVAL / "generalization_v8/results/v8a_c001/random_control/random_seed5/model.pth"))
args = ap.parse_args()
torch.set_num_threads(args.threads)

comp = json.loads((Path.home() / "compileCircuits/compilation_time_results_benchmark_30k.json").read_text())
data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
names = sorted(n for n in comp if n in data)
if args.limit:
    names = names[:: max(1, len(names) // args.limit)]
qasm = Path.home() / "compileCircuits/benchmark_dataset_30k"

params = load_hparams()
devices = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
model = build_model("sinkhorn", params, devices, torch.device("cpu"))
model.load_state_dict(torch.load(args.model, map_location="cpu")); model.eval()

rows = []
with torch.no_grad():
    model(data[names[0]].clone().__class__.from_dict(data[names[0]].to_dict()) if False else next(iter(DataLoader([data[names[0]]], batch_size=1))), devices)  # warm-up
    for n in names:
        t0 = time.perf_counter(); res = bqd._process((n, str(qasm / f"{n}.qasm"))); enc = time.perf_counter() - t0
        b = next(iter(DataLoader([data[n]], batch_size=1)))
        t0 = time.perf_counter(); model(b, devices); fwd = time.perf_counter() - t0
        e = comp[n][0]
        exh = sum(e[d]["compilation_time"] + e[d].get("fidelity_time", 0.0) for d in DEVICE_NAMES)
        comp1 = {d: e[d]["compilation_time"] + e[d].get("fidelity_time", 0.0) for d in DEVICE_NAMES}
        rows.append({"circuit": n, "q": int(re.search(r"_q(\d+)_", n).group(1)), "encode": enc, "forward": fwd,
                     "exhaustive": exh, "compile_best": comp1[DEVICE_NAMES[int(np.argmax([e[d]["fidelity"] for d in DEVICE_NAMES]))]],
                     "ok": res[4] is None})
    graphs = [data[n] for n in names]
    t0 = time.perf_counter()
    for b in DataLoader(graphs, batch_size=64):
        model(b, devices)
    batched = (time.perf_counter() - t0) / len(graphs)

out = Path(__file__).resolve().parent / f"timing_v8_threads{args.threads}.json"
out.write_text(json.dumps({"rows": rows, "batched_forward_per_circuit": batched}))
enc = np.array([r["encode"] for r in rows]); fwd = np.array([r["forward"] for r in rows])
exh = np.array([r["exhaustive"] for r in rows]); q = np.array([r["q"] for r in rows])
gro = np.array([r["circuit"].startswith("grover/") for r in rows])
print(f"{len(rows)} circuits, CPU threads {args.threads}")
print(f"encode   median {np.median(enc)*1e3:8.1f} ms   mean {enc.mean()*1e3:8.1f} ms")
print(f"forward  median {np.median(fwd)*1e3:8.1f} ms   mean {fwd.mean()*1e3:8.1f} ms   (batch 1, 3 devices)")
print(f"forward  batched (64) per circuit {batched*1e3:.2f} ms")
print(f"exhaustive (compile+fidelity, 3 devices) median {np.median(exh)*1e3:8.1f} ms  mean {exh.mean()*1e3:8.1f} ms")
pred = enc + fwd
print(f"speedup exhaustive / (encode+forward): median {np.median(exh/pred):.1f}x   share of circuits where predictor is faster {np.mean(exh > pred):.0%}")
print(f"without grover: median exhaustive {np.median(exh[~gro])*1e3:.1f} ms, median speedup {np.median(exh[~gro]/pred[~gro]):.1f}x, faster in {np.mean(exh[~gro] > pred[~gro]):.0%}")
for lo, hi in ((2, 5), (6, 10), (11, 15), (16, 20)):
    s = (q >= lo) & (q <= hi)
    if s.any():
        print(f"  q{lo:>2}-{hi:<2} n={s.sum():4d}  encode {np.median(enc[s])*1e3:7.1f} ms  forward {np.median(fwd[s])*1e3:6.1f} ms"
              f"  exhaustive {np.median(exh[s])*1e3:9.1f} ms  speedup {np.median(exh[s]/pred[s]):7.1f}x")
