#!/usr/bin/env python3
"""Which ingredient of the decay is wrong?  decay ≈ −(n_active · duration − Σ_q busy_q) · mean 1/min(T1, T2).

esp_budget.py showed that the true duration does not fix the ESP of the v8k-family models.  The decay also needs
how many qubits are active and their total busy time.  Per (circuit, device), with the true values from the stored
compilation (struct_labels_v2.pt: active, busy_sum; esp_labels.pt: duration, decay):
  1. the simplified formula with every ingredient TRUE vs the true decay (is the simplification good enough?);
  2. predicted vs true n_active, Σ busy, duration (ratio, per family and routing-free or not);
  3. the simplified formula with the model's ingredients, then each one swapped for the truth -> ESP regret.
    python esp_activity.py --run v8kda10 [--seeds 5]      # -> results/report_esp_activity_<run>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(HERE))
from esp_eval import DEVS, LABELS, SPLITS, r2  # noqa: E402
from esp_probe import features, split_dir  # noqa: E402
from esp_budget import KINDS, ROOT, choice  # noqa: E402
from gsv3 import paths  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

STRUCT2 = EVAL / "compile_check/results/struct_labels_v2.pt"


@torch.no_grad()
def collect(model, data, lab, st, dev, names):
    feats = features(model, data, names, dev)
    coh = np.array([float(dev.coh[dev.batch == d].mean()) for d in range(len(DEVS))])
    rows = defaultdict(list)
    for f in feats:
        nm = f["names"]
        dec, ldur, _ = model.esp_from_features(model.esp_mlp, f, model.esp_head)
        b = next(iter(DataLoader([data[n] for n in nm], batch_size=len(nm))))
        rows["rf"].append(routing_free(b, dev, model._rf_cache).numpy())
        rows["act"].append(f["active"].sum(-1).numpy())
        rows["busy"].append(f["busy"].sum(-1).numpy())
        rows["dur"].append(torch.exp(ldur).numpy())
        rows["dec"].append(dec.numpy())
        rows["logf"].append(f["logf"].numpy())
        rows["names"] += nm
    p = {k: (np.concatenate(v) if k != "names" else v) for k, v in rows.items()}
    S = np.stack([st[n].numpy() if n in st else np.full((len(DEVS), 5), np.nan) for n in p["names"]])
    T = {k: np.stack([lab[n][k].numpy() for n in p["names"]]) for k in ("F", "ESP", "duration", "decay")}
    return p, S, T, coh


def analyse(p, S, T, coh):
    ok = np.isfinite(S[..., 0]).all(1) & np.isfinite(T["decay"]).all(1)
    act_t, busy_t, dur_t = S[..., 3], S[..., 4], T["duration"] * 1e6
    simp = lambda a, d, b: -np.maximum(a * d - b, 0) * coh[None, :]
    out = {"n": int(ok.sum())}
    s_true = simp(act_t, dur_t, busy_t)
    out["simplified (all true) vs true decay: corr"] = float(np.corrcoef(s_true[ok].ravel(), T["decay"][ok].ravel())[0, 1])
    out["simplified (all true) vs true decay: R²"] = r2(s_true[ok].ravel(), T["decay"][ok].ravel())
    rat = lambda a, b: float(np.median(a[ok] / np.maximum(b[ok], 1e-9)))
    out["n_active pred / true (median)"] = rat(p["act"], act_t)
    out["Σ busy pred / true (median)"] = rat(p["busy"], busy_t)
    out["duration pred / true (median)"] = rat(p["dur"], dur_t)
    out["share with n_active wrong"] = float((np.abs(p["act"][ok] - act_t[ok]) > 0.5).mean())
    for lab_, sel in (("routing-free", p["rf"] > 0.5), ("routed", p["rf"] <= 0.5)):
        s = ok[:, None] & sel
        if s.sum() > 20:
            out[f"n_active pred − true, {lab_} (mean)"] = float((p["act"][s] - act_t[s]).mean())
            out[f"n_active true − logical, {lab_} (mean)"] = float((act_t[s] - p["act"][s]).mean())
    fhat = np.exp(np.minimum(p["logf"], 0))
    esp = T["ESP"][ok]
    variants = {"model decay (per-qubit head)": p["dec"], "simplified, model ingredients": simp(p["act"], p["dur"], p["busy"]),
                "… true n_active": simp(act_t, p["dur"], p["busy"]), "… true Σ busy": simp(p["act"], p["dur"], busy_t),
                "… true duration": simp(p["act"], dur_t, p["busy"]), "… true n_active + duration": simp(act_t, dur_t, p["busy"]),
                "… true n_active + Σ busy": simp(act_t, p["dur"], busy_t), "simplified, all true": s_true, "true decay": T["decay"]}
    out["regret"] = {k: choice((fhat * np.exp(v))[ok], esp)[1] for k, v in variants.items()}
    out["decay R²"] = {k: r2(v[ok].ravel(), T["decay"][ok].ravel()) for k, v in variants.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v8kda10")
    ap.add_argument("--seeds", default="5")
    args = ap.parse_args()
    torch.set_num_threads(8)
    kind = KINDS.get(args.run, "sinkhorn_rf_route_czc_espa")
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    st = torch.load(STRUCT2, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    res = defaultdict(list)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for split, sd in SPLITS.items():
            d = split_dir(ROOT, args.run, seed, sd)
            if d is None:
                continue
            model = build_model(kind, params, dev, torch.device("cpu"))
            model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
            names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data]
            m = analyse(*collect(model, data, lab, st, dev, names))
            res[split].append(m)
            print(args.run, seed, split, json.dumps(m), flush=True)
    (HERE / f"results/report_esp_activity_{args.run}.json").write_text(json.dumps(res))
    mean = lambda xs: float(np.mean(xs))
    L = [f"# Decay ingredients: {args.run}", "", "decay ≈ −(n_active · duration − Σ busy) · mean 1/min(T1, T2) per device; mean over seeds.", "",
         "| | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
    keys = [k for k in next(iter(res.values()))[0] if not isinstance(next(iter(res.values()))[0][k], dict) and k != "n"]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{mean([m[k] for m in res[s] if k in m]):.3f}" if any(k in m for m in res[s]) else "" for s in res) + " |")
    for sec in ("regret", "decay R²"):
        L += ["", f"## {sec}", "", "| decay used | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
        for v in next(iter(res.values()))[0][sec]:
            L.append(f"| {v} | " + " | ".join(f"{mean([m[sec][v] for m in res[s]]):.4f}" for s in res) + " |")
    (HERE / f"results/report_esp_activity_{args.run}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
