#!/usr/bin/env python3
"""Calibration shift with a low-noise ground truth (best of K compiler seeds) and
moderate changes.  Same 457-circuit sample as ``run_shift.py``; results in
``results_k/`` (the original ``results/`` are left untouched).

Variants: the original ones (orig, x0.5, x2, shuffle, jitterA, jitterB) plus
moderate changes of the gate errors:
``x0.7`` ``x0.8`` ``x1.2`` ``x1.3``  every error scaled together by ±20–30 %;
``jitterS``                          every error times an independent log-normal
                                     factor with sigma 0.2 (typically ±20 %).

Ground truth: each (circuit, variant) compiled with ``seed_transpiler`` 0..K-1;
``best`` = max over the seeds (what a good compilation reaches), plus the spread.
The noise ceiling of the tracking is estimated by split halves: the change vs
orig measured with seeds {0, 1} and with seeds {2, 3} are two independent estimates;
their correlation r gives the reliability, and sqrt(r) bounds the correlation any
model can reach with a single estimate (Spearman–Brown for the best of all K is higher).

    python shift_k.py truth [--k 5 --workers 32]
    python shift_k.py predict
    python shift_k.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cshift  # noqa: E402
import run_shift  # noqa: E402

OUT = HERE / "results_k"
EXTRA = {"x0.7": 0.7, "x0.8": 0.8, "x1.2": 1.2, "x1.3": 1.3}
GROUPS = {"large scale (x0.5, x2)": ("x0.5", "x2"),
          "moderate scale (x0.7-x1.3)": ("x0.7", "x0.8", "x1.2", "x1.3"),
          "local (shuffle, jitter)": ("shuffle", "jitterA", "jitterB", "jitterS")}
_VARIANTS = None
K = 5


def make_extra(backend, variant: str, seed: int):
    import copy
    be = copy.deepcopy(backend)
    t = be.target
    rng = np.random.default_rng(seed)
    for gate in cshift.GATES:
        qargs = [q for q, p in t[gate].items() if p is not None and p.error is not None]
        errors = np.array([t[gate][q].error for q in qargs], dtype=float)
        if variant in EXTRA:
            new = errors * EXTRA[variant]
        elif variant == "jitterS":
            new = errors * np.exp(rng.normal(0.0, 0.2, size=errors.shape))
        else:
            raise ValueError(variant)
        for q, e in zip(qargs, new):
            cshift._set_error(t, gate, q, e)
    return be


def all_variants() -> dict:
    out = cshift.all_variants()
    for i, (name, be) in enumerate(cshift.base_backends().items()):
        for v in (*EXTRA, "jitterS"):
            out[f"{name}/{v}"] = make_extra(be, v, seed=23 + 100 * i)
    return out


def _init():
    global _VARIANTS
    _VARIANTS = all_variants()


def _score(name: str):
    from qiskit import qasm3
    from qiskit.transpiler import generate_preset_pass_manager
    qc = qasm3.load(str(cshift.COMPILE_DIR / "benchmark_dataset_30k" / f"{name}.qasm"))
    out = {}
    for key, be in _VARIANTS.items():
        fs = []
        for s in range(K):
            try:
                pm = generate_preset_pass_manager(optimization_level=2, target=be.target, seed_transpiler=s)
                fs.append(float(round(cshift._fidelity_recursive(pm.run(qc), be.target), 10)))
            except Exception:  # noqa: BLE001
                fs.append(None)
        out[key] = fs
    return name, out


def cmd_truth(args):
    global K
    K = args.k
    OUT.mkdir(exist_ok=True)
    sample = json.loads((run_shift.RESULTS / "sample.json").read_text())
    import gsv3  # noqa: F401  (puts gsv2 on sys.path)
    from gsv2.devices import _backend_graph  # noqa: E402
    variants = all_variants()
    raw = {k: dict(zip(("nodes", "edges", "edge_feats"), _backend_graph(be))) for k, be in variants.items()}
    (OUT / "variants_raw.json").write_text(json.dumps(raw))
    truth, t0 = {}, time.time()
    with Pool(args.workers, initializer=_init) as pool:
        for i, (name, res) in enumerate(pool.imap_unordered(_score, sample, chunksize=1), 1):
            truth[name] = res
            if i % 25 == 0:
                print(f"  {i}/{len(sample)}  {time.time() - t0:.0f}s", flush=True)
    (OUT / "truth.json").write_text(json.dumps(truth))
    print("wrote", OUT / "truth.json")


def cmd_predict(args):
    import torch
    from torch_geometric.loader import DataLoader
    import gsv3  # noqa: F401
    from gsv3 import paths
    from gsv3.model import build_model, load_hparams
    sample = json.loads((run_shift.RESULTS / "sample.json").read_text())
    raw = json.loads((OUT / "variants_raw.json").read_text())
    keys = list(raw)
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt",
                                                  weights_only=False)}
    graphs = [data[n] for n in sample]
    params = load_hparams()
    devices = run_shift._variant_batch(raw, keys, params["lap_pe"])
    preds = {}
    for mname, (kind, path) in run_shift.MODELS.items():
        if not path.is_file():
            continue
        model = build_model(kind, params, devices, torch.device("cpu"))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            P = torch.cat([torch.exp(torch.clamp(model(b, devices), max=0.0))
                           for b in DataLoader(graphs, batch_size=16)]).numpy()
        preds[mname] = {n: dict(zip(keys, map(float, row))) for n, row in zip(sample, P)}
        print("predicted", mname, flush=True)
    (OUT / "predictions.json").write_text(json.dumps(preds))


def _r2(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return 1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)


def cmd_report(args):
    truth = json.loads((OUT / "truth.json").read_text())
    preds = json.loads((OUT / "predictions.json").read_text())
    names = sorted(truth)
    keys = list(truth[names[0]])
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    def best(n, k, seeds=None):
        v = [f for i, f in enumerate(truth[n][k]) if f is not None and (seeds is None or i in seeds)]
        return max(v) if v else None

    out(f"## Calibration shift, best-of-{len(truth[names[0]][keys[0]])} ground truth")
    out("noise: mean std of F over the compiler seeds; ceiling = sqrt(split-half reliability) of the")
    out("change vs orig (seeds {0,1} vs {2,3}); tracking = corr(pred dlogF, best-of-K dlogF), F > 0.01")
    models = sorted(preds)
    per = {}   # (model, variant) -> (r2, trk)
    out(f"\n{'variant':22}{'noise':>7}{'ceiling':>8}" + "".join(f"{m.replace('_sinkhorn', ''):>16}" for m in models))
    for k in keys:
        dev, v = k.split("/")
        ref = f"{dev}/orig"
        ok = [n for n in names if best(n, k) is not None and best(n, ref) is not None]
        t = np.array([best(n, k) for n in ok])
        noise = np.mean([np.std([f for f in truth[n][k] if f is not None]) for n in ok])
        keep = [n for n in ok if best(n, k) > 0.01 and best(n, ref) > 0.01]
        dt = np.log([best(n, k) / best(n, ref) for n in keep])
        ceil = float("nan")
        if v != "orig":
            a = [np.log(best(n, k, {0, 1}) / best(n, ref, {0, 1})) for n in keep]
            b = [np.log(best(n, k, {2, 3}) / best(n, ref, {2, 3})) for n in keep]
            r = np.corrcoef(a, b)[0, 1]
            ceil = np.sqrt(max(r, 0.0))
        cells = []
        for m in models:
            p = np.array([preds[m][n][k] for n in ok])
            r2 = _r2(p, t)
            trk = float("nan")
            if v != "orig":
                dp = np.log([max(preds[m][n][k], 1e-6) / max(preds[m][n][ref], 1e-6) for n in keep])
                trk = np.corrcoef(dp, dt)[0, 1]
            per[(m, k)] = (r2, trk)
            cells.append(f"{r2:7.3f}/{trk:+.2f}" if v != "orig" else f"{r2:7.3f}      ")
        out(f"{k:22}{noise:>7.4f}{ceil:>8.2f}" + "".join(f"{c:>16}" for c in cells))
    out("\ncells: R² vs best-of-K truth / tracking")
    out("\n## Summary per group of variants (mean over the 3 devices): R² mean, worst R², tracking mean")
    for g, vs in GROUPS.items():
        out(f"\n### {g}")
        for m in models:
            ks = [k for k in keys if k.split("/")[1] in vs]
            r2s = [per[(m, k)][0] for k in ks]
            trk = [per[(m, k)][1] for k in ks]
            out(f"{m:20} R² {np.mean(r2s):.3f}  worst {np.min(r2s):.3f}  tracking {np.nanmean(trk):+.2f}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["truth", "predict", "report"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    {"truth": cmd_truth, "predict": cmd_predict, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
