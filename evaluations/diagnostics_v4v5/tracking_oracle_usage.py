#!/usr/bin/env python3
"""Oracle study for the usage-map head (v11), on the pilot circuits that have usage maps.

Same question as tracking_oracle.py (what limits local calibration tracking?), but the model's
own forward is used and only patched where an oracle replaces a learned part:
  model          as is
  + layout       Sinkhorn placement replaced by the compiler's initial layout (one-hot)
  + counts       per-circuit totals of r / measure / cz rescaled to the compiled ones (v11 only)
  + both         layout and counts
  true maps      the head evaluated on the compiler's usage maps (v11 only): the maps come from the
                 same compilations as the labels, so this must give ~1.0 (a check of the code path)
Circuits: pilot circuits (merge_usage.py) in the control test split; calibrations: local v8d
variants (jitter, shuffle, bad qubits, aug >= 8).  Tracking = mean over (variant, device) of
corr(Δlog F predicted, Δlog F true), F > 0.01, as in tracking_oracle.py.
    python tracking_oracle_usage.py [--n-circuits 300] [--n-variants 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

ROOT = EVAL / "generalization_v9/results"
V8D = EVAL / "compile_check/results/v8d"
ORIG = EVAL / "compile_check/results/v8b_orig"
USAGE = EVAL / "compile_check/results/v8u/usage_labels.pt"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LOCAL_KINDS = {0, 1, 2, 3, 4, 5, 6, 9}
MODELS = {"v8kd40": ("sinkhorn_rf_route_czc", "v8kd40"), "v11 (phase 1)": ("sinkhorn_rf_route_usage", "v11"),
          "v11d40": ("sinkhorn_rf_route_usage", "v11d40")}


def predict(model, batch, devices, layout=None, counts=None, maps=None):
    """log F (B, D) with the requested oracles patched into the model for this call."""
    orig_sink, orig_usage = model._sinkhorn, getattr(model, "_usage_log_f", None)
    if layout is not None:
        def sink(scores, n_q, n_phys):
            a = orig_sink(scores, n_q, n_phys)
            lay = layout[:, :, : a.size(2)]
            onehot = F.one_hot(lay.clamp(min=0), a.size(3)).float()
            return torch.where((lay >= 0).unsqueeze(-1), onehot, a)
        model._sinkhorn = sink
    if orig_usage is not None and (counts is not None or maps is not None):
        def usage(circuits, devs, assign, n_ops, valid, local, gb, n_circ):
            if counts is not None:
                tot = scatter(n_ops, gb, dim=0, dim_size=n_circ, reduce="sum")
                ratio = torch.where(tot > 1e-6, counts / tot.clamp(min=1e-6), torch.ones_like(tot))
                n_ops = n_ops * ratio[gb]
            out = orig_usage(circuits, devs, assign, n_ops, valid, local, gb, n_circ)
            if maps is not None:
                from torch_geometric.utils import to_dense_batch
                eps_q, _ = to_dense_batch(devs.phys, devs.batch)
                _, eps_e, _ = model._route_kernel(devs, eps_q.size(1))
                ur, um, uc = maps
                P, E = eps_q.size(1), eps_e.size(1)
                out = -((ur[..., :P] * eps_q[..., 0]).sum(-1) + (um[..., :P] * eps_q[..., 1]).sum(-1)
                        + (uc[..., :E] * eps_e).sum(-1))
            return out
        model._usage_log_f = usage
    try:
        with torch.no_grad():
            return model(batch, devices)
    finally:
        model._sinkhorn = orig_sink
        if orig_usage is not None:
            model._usage_log_f = orig_usage


def dense_layout(graphs, lays):
    qm = max(int(g.n_qubits) for g in graphs)
    out = torch.full((len(graphs), len(DEVS), qm), -1, dtype=torch.long)
    for i, l in enumerate(lays):
        out[i, :, : l.shape[0]] = l.t()
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=300)
    ap.add_argument("--n-variants", type=int, default=10)
    ap.add_argument("--split", default="random_control/random_seed5", help="model directory under each run")
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    rng = np.random.default_rng(0)
    params = load_hparams()
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    use = torch.load(USAGE, weights_only=False)
    lab = torch.load(V8D / "v8d_labels.pt", weights_only=False)
    orig = torch.load(ORIG / "v8b_orig_labels.pt", weights_only=False)
    test = set(json.loads((ROOT / "v8kd40/random_control/random_seed5/predictions.json").read_text()))
    pool = [data[n] for n in sorted(use) if n in test and n in data and n in lab and n in orig
            and "F_aug" in use[n] and lab[n]["layout"].shape[0] == int(data[n].n_qubits)]
    pool = [pool[i] for i in rng.choice(len(pool), min(args.n_circuits, len(pool)), replace=False)]
    js = sorted(rng.choice([j for j in range(8, 100) if j % 10 in LOCAL_KINDS], args.n_variants, replace=False).tolist())
    raw = json.loads((V8D / "variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    dev_o = variant_batch(raw, list(DEVS), params["lap_pe"])
    dev_v = {j: variant_batch(raw, [f"{d}/aug{j}" for d in DEVS], params["lap_pe"]) for j in js}
    log(f"{len(pool)} pilot test circuits, variants {js}")
    L = ["# Oracle study on the usage-map head (v11)", "",
         f"{len(pool)} pilot circuits of the control test split; {len(js)} local v8d calibrations (aug {js}).",
         "Tracking: mean over (variant, device) of corr(Δlog F pred, Δlog F true), F > 0.01.", "",
         "| model | model | + layout | + counts | + both | true maps |", "| --- | --- | --- | --- | --- | --- |"]
    for name, (kind, run) in MODELS.items():
        path = ROOT / run / args.split / "model.pth"
        if not path.is_file():
            L.append(f"| {name} | not trained yet | | | | |"); continue
        model = build_model(kind, params, dev_o, torch.device("cpu"))
        model.load_state_dict(torch.load(path, map_location="cpu")); model.eval()
        is_usage = kind.endswith("_usage")
        settings = ["model", "+ layout"] + (["+ counts", "+ both", "true maps"] if is_usage else [])
        dp = {s: {} for s in settings}; dt = {}
        for start in range(0, len(pool), 32):
            gs = pool[start: start + 32]
            b = next(iter(DataLoader(gs, batch_size=len(gs))))
            y_o = torch.stack([orig[g.circuit_name]["F"][0] for g in gs]).float()
            lay_o = dense_layout(gs, [orig[g.circuit_name]["layout"][:, 0] for g in gs])
            cnt_o = torch.stack([orig[g.circuit_name]["counts"][0] for g in gs]).float()
            map_o = tuple(torch.stack([use[g.circuit_name][k] for g in gs]) for k in ("r", "m", "cz"))
            base = {}
            for s in settings:
                base[s] = predict(model, b, dev_o, layout=lay_o if s in ("+ layout", "+ both") else None,
                                  counts=cnt_o if s in ("+ counts", "+ both") else None,
                                  maps=map_o if s == "true maps" else None)
            for j in js:
                y_v = torch.stack([lab[g.circuit_name]["F"][j] for g in gs]).float()
                keep = (y_v > 0.01) & (y_o > 0.01)
                d_true = torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))
                lay_v = dense_layout(gs, [lab[g.circuit_name]["layout"][:, j] for g in gs])
                cnt_v = torch.stack([lab[g.circuit_name]["counts"][j] for g in gs]).float()
                map_v = tuple(torch.stack([use[g.circuit_name][k + "_aug"][j] for g in gs]) for k in ("r", "m", "cz"))
                for d in range(len(DEVS)):
                    dt.setdefault((j, d), []).append(d_true[:, d][keep[:, d]].numpy())
                for s in settings:
                    lv = predict(model, b, dev_v[j], layout=lay_v if s in ("+ layout", "+ both") else None,
                                 counts=cnt_v if s in ("+ counts", "+ both") else None,
                                 maps=map_v if s == "true maps" else None)
                    dpred = torch.clamp(lv, max=0) - torch.clamp(base[s], max=0)
                    for d in range(len(DEVS)):
                        dp[s].setdefault((j, d), []).append(dpred[:, d][keep[:, d]].numpy())
        dtc = {k: np.concatenate(v) for k, v in dt.items()}
        cells = []
        for s in ["model", "+ layout", "+ counts", "+ both", "true maps"]:
            if s not in settings:
                cells.append("–"); continue
            pc = {k: np.concatenate(v) for k, v in dp[s].items()}
            cells.append(f"{np.nanmean([corr(pc[k], dtc[k]) for k in dtc if len(dtc[k]) > 20]):.2f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
        log(L[-1])
    L += ["", "true maps = the head's formula on the compiler's own maps (same compilations as the labels): "
          "~1.0 by construction, a check that the head and the maps agree."]
    out = HERE / "results" / f"report_tracking_oracle_usage{'' if args.split.startswith('random') else '_' + args.split.replace('/', '_')}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
