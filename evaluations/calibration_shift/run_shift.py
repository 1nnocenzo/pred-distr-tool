#!/usr/bin/env python3
"""Calibration-shift study (see cshift.py).  Run on a compute node.

    python run_shift.py truth   [--n-per-family 40] [--workers 32]
    python run_shift.py predict
    python run_shift.py report

``truth``    picks a sample of the *control* test set (circuits no control model saw
             in training), compiles each circuit on every device variant and
             scores it; also stores the variants' raw device graphs.
``predict``  runs the trained control models (v3 xattn, v4 phys, v5 sinkhorn) on
             the variants, with the device features standardised with the
             statistics of the three ORIGINAL devices, as at training time.
``report``   accuracy per variant and how well each model follows the change
             with respect to ``orig``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EVAL / "generalization_v3"))
import cshift  # noqa: E402

RESULTS = HERE / "results"
EXCLUDE = {"iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code",
           "shors_nine_qubit_code"}           # dynamic circuits: see generalization_v4 README
def _control(root: str, run: str, seed: int) -> Path:
    name = run if seed == 5 else f"{run}_s{seed}"
    return EVAL / root / "results" / name / "random_control" / f"random_seed{seed}" / "model.pth"


# name: (kind, control model).  "@sN" marks the seed; report() aggregates seeds.
MODELS = {"v3_xattn@s5": ("xattn", _control("generalization_v3", "xattn_a05", 5))}
for _s in (5, 6, 7):
    MODELS[f"v4_phys@s{_s}"] = ("phys", _control("generalization_v4", "v4_phys", _s))
    MODELS[f"v5_sinkhorn@s{_s}"] = ("sinkhorn", _control("generalization_v5", "v5_sinkhorn", _s))
    MODELS[f"v5b_sinkhorn@s{_s}"] = ("sinkhorn", _control("generalization_v5", "v5b_sinkhorn", _s))
    for _lam in ("v6_l001", "v6_l01"):   # v6: Sinkhorn + compiler-layout loss (missing seeds skipped)
        MODELS[f"{_lam}@s{_s}"] = ("sinkhorn", _control("generalization_v6", _lam, _s))
    MODELS[f"v4u_phys@s{_s}"] = ("phys_uniform", _control("generalization_v6", "v4u_phys", _s))
    for _lam in ("v7_l001", "v7_l01"):   # v7: Sinkhorn + region/distance layout loss
        MODELS[f"{_lam}@s{_s}"] = ("sinkhorn", _control("generalization_v7", _lam, _s))
    for _c in ("v8a_c001", "v8a_c01"):   # v8a: v7 (λ 0.1) + count loss
        MODELS[f"{_c}@s{_s}"] = ("sinkhorn", _control("generalization_v8", _c, _s))

_VARIANTS = None


def _init_worker():
    global _VARIANTS
    _VARIANTS = cshift.all_variants()


def _score(name: str):
    from qiskit import qasm3
    qc = qasm3.load(str(cshift.COMPILE_DIR / "benchmark_dataset_30k" / f"{name}.qasm"))
    out = {}
    for key, be in _VARIANTS.items():
        t0 = time.perf_counter()
        try:
            out[key] = (cshift.compile_and_score(qc, be), time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            out[key] = (None, repr(exc))
    return name, out


def cmd_truth(args):
    import gsv3  # noqa: F401  (sys.path for genstudy / gsv2)
    from genstudy.data import GraphDataset
    from gsv2.devices import _backend_graph
    ds = GraphDataset.load(EVAL / "generalization_v3" / "data")
    split = ds.random_split(test_fraction=0.3, seed=5)       # the control split
    rng = np.random.default_rng(0)
    by_fam = defaultdict(list)
    for i in split.test:
        d = ds.data[i]
        if ds.families[i] in EXCLUDE or sum(d.compilation_time_dict.values()) > 0.5:
            continue
        by_fam[ds.families[i]].append(ds.names[i])
    sample = sorted(n for names in by_fam.values()
                    for n in rng.permutation(names)[: args.n_per_family])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sample.json").write_text(json.dumps(sample, indent=1))
    print(f"sample: {len(sample)} circuits from {len(by_fam)} families", flush=True)

    variants = cshift.all_variants()
    raw = {k: dict(zip(("nodes", "edges", "edge_feats"), _backend_graph(be)))
           for k, be in variants.items()}
    (RESULTS / "variants_raw.json").write_text(json.dumps(raw))

    truth, t0 = {}, time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for k, (name, res) in enumerate(pool.imap_unordered(_score, sample, chunksize=2), 1):
            truth[name] = res
            if k % 50 == 0:
                print(f"  {k}/{len(sample)}  {time.time() - t0:.0f}s", flush=True)
    (RESULTS / "truth.json").write_text(json.dumps(truth))
    print("wrote", RESULTS / "truth.json")


def _variant_batch(raw_all: dict, keys: list[str], lap_pe: int):
    import torch
    from torch_geometric.data import Batch, Data
    from gsv3.devices import _laplacian_pe, _physical_errors
    # Standardisation statistics of the three ORIGINAL devices (as in gsv2.devices).
    orig = torch.load(EVAL / "generalization_v2/results/device_graphs.pt", weights_only=False)["raw"]
    all_nodes = torch.tensor([r for v in orig.values() for r in v["nodes"]])
    all_edges = torch.tensor([e for v in orig.values() for e in v["edge_feats"]]).unsqueeze(1)
    n_mean, n_std = all_nodes.mean(0), all_nodes.std(0).clamp_min(1e-8)
    e_mean, e_std = all_edges.mean(0), all_edges.std(0).clamp_min(1e-8)
    graphs = []
    for k in keys:
        r = raw_all[k]
        x = (torch.tensor(r["nodes"]) - n_mean) / n_std
        ea = ((torch.tensor(r["edge_feats"]).unsqueeze(1) - e_mean) / e_std).float()
        src = [a for a, b in r["edges"]] + [b for a, b in r["edges"]]
        dst = [b for a, b in r["edges"]] + [a for a, b in r["edges"]]
        g = Data(x=x.float(), edge_index=torch.tensor([src, dst], dtype=torch.long),
                 edge_attr=torch.cat([ea, ea]))
        g.x = torch.cat([g.x, _laplacian_pe(g.num_nodes, g.edge_index, lap_pe)], dim=1)
        g.phys = _physical_errors({"nodes": r["nodes"], "edges": [tuple(e) for e in r["edges"]],
                                   "edge_feats": r["edge_feats"]})
        graphs.append(g)
    return Batch.from_data_list(graphs)


def cmd_predict(args):
    import torch
    from torch_geometric.loader import DataLoader
    import gsv3  # noqa: F401
    from gsv3 import paths
    from gsv3.model import build_model, load_hparams
    sample = json.loads((RESULTS / "sample.json").read_text())
    raw_all = json.loads((RESULTS / "variants_raw.json").read_text())
    keys = list(raw_all)
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt",
                                                  weights_only=False)}
    graphs = [data[n] for n in sample]
    params = load_hparams()
    devices = _variant_batch(raw_all, keys, params["lap_pe"])

    # Sanity check: the 'orig' variants must equal the training-time device graphs.
    from gsv3.devices import load_devices
    from genstudy.data import DEVICE_NAMES
    ref = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"],
                       physical_errors=True)
    for i, name in enumerate(DEVICE_NAMES):
        j = keys.index(f"{name}/orig")
        a, b = devices.get_example(j), ref.get_example(i)
        ok = torch.allclose(a.x, b.x, atol=1e-4) and torch.allclose(a.phys, b.phys, atol=1e-6)
        print(f"orig graph check {name}: {'OK' if ok else 'MISMATCH'}", flush=True)

    preds = {}
    for mname, (kind, path) in MODELS.items():
        if not path.is_file():
            print(f"skip {mname}: {path} not found", flush=True)
            continue
        model = build_model(kind, params, devices, torch.device("cpu"))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        out = []
        with torch.no_grad():
            for b in DataLoader(graphs, batch_size=16):
                out.append(torch.exp(torch.clamp(model(b, devices), max=0.0)))
        P = torch.cat(out).numpy()
        preds[mname] = {n: dict(zip(keys, map(float, row))) for n, row in zip(sample, P)}
        print(f"predicted {mname}", flush=True)
    (RESULTS / "predictions.json").write_text(json.dumps(preds))
    print("wrote", RESULTS / "predictions.json")


def _r2(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return 1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)


def cmd_report(args):
    import torch
    truth = json.loads((RESULTS / "truth.json").read_text())
    preds = json.loads((RESULTS / "predictions.json").read_text())
    data = {d.circuit_name: d.y.view(-1).numpy() for d in torch.load(
        EVAL / "generalization_v3/data/graph_dataset_expected_fidelity.pt", weights_only=False)}
    names = sorted(truth)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    # 1) compiler noise floor: recompiled 'orig' vs the dataset value.
    out("## Compiler noise floor (recompiling on the original devices vs. the dataset)")
    for i, dev in enumerate(cshift.DEVICE_NAMES):
        t = [truth[n][f"{dev}/orig"][0] for n in names if truth[n][f"{dev}/orig"][0] is not None]
        d = [data[n][i] for n in names if truth[n][f"{dev}/orig"][0] is not None]
        out(f"{dev:12} R2 {_r2(t, d):.4f}  MAE {np.mean(np.abs(np.array(t) - d)):.4f}")

    # 2) accuracy per variant, and tracking of the change w.r.t. orig.
    all_preds = preds
    preds = {m.split("@")[0]: v for m, v in all_preds.items() if m.endswith("@s5")}
    out("\n## Accuracy per variant (R2 / MAE) and tracking of the change vs. orig (seed 5)")
    out("tracking = corr(predicted log F(variant) - log F(orig), true ...), on circuits with F > 0.01")
    for dev in cshift.DEVICE_NAMES:
        out(f"\n### {dev}")
        out(f"{'variant':9}" + "".join(f"{m:>34}" for m in preds))
        for v in cshift.VARIANTS:
            key, ref = f"{dev}/{v}", f"{dev}/orig"
            cells = []
            for m in preds:
                ok = [n for n in names if truth[n][key][0] is not None and truth[n][ref][0] is not None]
                t = np.array([truth[n][key][0] for n in ok]); p = np.array([preds[m][n][key] for n in ok])
                cell = f"R2 {_r2(p, t):.3f} MAE {np.mean(np.abs(p - t)):.3f}"
                if v != "orig":
                    keep = [i for i, n in enumerate(ok) if truth[n][key][0] > 0.01 and truth[n][ref][0] > 0.01]
                    dt = np.log([truth[ok[i]][key][0] / truth[ok[i]][ref][0] for i in keep])
                    dp = np.log([max(preds[m][ok[i]][key], 1e-6) / max(preds[m][ok[i]][ref], 1e-6) for i in keep])
                    corr = np.corrcoef(dp, dt)[0, 1] if len(keep) > 2 and dt.std() > 0 else float("nan")
                    cell += f" trk {corr:+.2f}"
                cells.append(cell)
            out(f"{v:9}" + "".join(f"{c:>34}" for c in cells))

    # 3) device choice among ALL 18 variants (as if a scheduler saw 18 machines).
    out("\n## Device choice among all variants: accuracy / mean regret")
    keys = [f"{d}/{v}" for d in cshift.DEVICE_NAMES for v in cshift.VARIANTS]
    for m in preds:
        acc, reg = [], []
        for n in names:
            t = np.array([truth[n][k][0] if truth[n][k][0] is not None else -1 for k in keys])
            p = np.array([preds[m][n][k] for k in keys])
            acc.append(np.argmax(p) == np.argmax(t)); reg.append(t.max() - t[np.argmax(p)])
        out(f"{m:12} accuracy {np.mean(acc):.3f}  regret {np.mean(reg):.4f}")
    # 4) Seed-aggregated summary: one row per model version, mean ± std over seeds.
    out("\n## Summary over seeds (mean ± std over the available seeds)")
    out("worst R2 = min over all 18 device variants; scale = x0.5/x2; local = shuffle/jitter")
    def stats(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return "-"
        return f"{np.mean(vals):.3f}" + (f"±{np.std(vals, ddof=1):.3f}" if len(vals) > 1 else "") + f"({len(vals)})"
    per = defaultdict(lambda: defaultdict(list))
    for m, P in all_preds.items():
        base = m.split("@")[0]
        r2s, trk_scale, trk_local = [], [], []
        for dev in cshift.DEVICE_NAMES:
            ref = f"{dev}/orig"
            for v in cshift.VARIANTS:
                key = f"{dev}/{v}"
                ok = [n for n in names if truth[n][key][0] is not None and truth[n][ref][0] is not None]
                t = np.array([truth[n][key][0] for n in ok]); p = np.array([P[n][key] for n in ok])
                r2s.append(_r2(p, t))
                if v == "orig":
                    continue
                keep = [n for n in ok if truth[n][key][0] > 0.01 and truth[n][ref][0] > 0.01]
                dt = np.log([truth[n][key][0] / truth[n][ref][0] for n in keep])
                dp = np.log([max(P[n][key], 1e-6) / max(P[n][ref], 1e-6) for n in keep])
                c = np.corrcoef(dp, dt)[0, 1] if len(keep) > 2 and dt.std() > 0 else float("nan")
                (trk_scale if v in ("x0.5", "x2") else trk_local).append(c)
        acc, reg = [], []
        for n in names:
            t = np.array([truth[n][k][0] if truth[n][k][0] is not None else -1 for k in keys])
            p = np.array([P[n][k] for k in keys])
            acc.append(np.argmax(p) == np.argmax(t)); reg.append(t.max() - t[np.argmax(p)])
        for k, v in (("worst R2", min(r2s)), ("mean R2", float(np.mean(r2s))),
                     ("tracking scale", float(np.nanmean(trk_scale))),
                     ("tracking local", float(np.nanmean(trk_local))),
                     ("choice acc", float(np.mean(acc))), ("choice regret", float(np.mean(reg)))):
            per[base][k].append(v)
    cols = ["worst R2", "mean R2", "tracking scale", "tracking local", "choice acc", "choice regret"]
    out(f"{'model':14}" + "".join(f"{c:>22}" for c in cols))
    for base, d in per.items():
        out(f"{base:14}" + "".join(f"{stats(d[c]):>22}" for c in cols))
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    print("\nwrote", RESULTS / "report.md")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("command", choices=["truth", "predict", "report"])
    p.add_argument("--n-per-family", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    {"truth": cmd_truth, "predict": cmd_predict, "report": cmd_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
