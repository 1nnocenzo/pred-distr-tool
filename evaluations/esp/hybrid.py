#!/usr/bin/env python3
"""Hybrid rule: compile where it is cheap (routing-free on that device), predict the rest.

Per test circuit and device, the model's routing-free flag (VF2 subgraph test, the decision Qiskit's VF2Layout
makes) decides: routing-free -> compile (the true label: F, ESP), otherwise -> the model's prediction.  The device
is chosen on the mixed scores.  Per split and seed: regret and right-device share for F (v8kdc10, predictions.json)
and ESP (an ESP run, its own heads), model only vs hybrid vs compile everything, and the compilations per circuit.
Flag check: among flagged routing-free pairs, the share whose compiled circuit uses more physical qubits than the
logical circuit (ancillas = routing happened).
    python hybrid.py [--esp-run v8kdi10]            # -> results/report_hybrid.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(HERE))
from esp_eval import DEVS, LABELS, SPLITS, predict  # noqa: E402
from esp_probe import split_dir  # noqa: E402
from esp_budget import KINDS, ROOT, choice  # noqa: E402
from gsv3 import paths  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

STRUCT2 = EVAL / "compile_check/results/struct_labels_v2.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-run", default="v8kdc10")
    ap.add_argument("--esp-run", default="v8kdi10")
    ap.add_argument("--seeds", default="5,6,7")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    st = torch.load(STRUCT2, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    kind = KINDS.get(args.esp_run, "sinkhorn_rf_route_czc_espa")
    res, rf_cache = defaultdict(list), {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        for split, sd in SPLITS.items():
            df, de = split_dir(ROOT, args.f_run, seed, sd), split_dir(ROOT, args.esp_run, seed, sd)
            if df is None or de is None:
                continue
            predF = json.loads((df / "predictions.json").read_text())
            names = [n for n in predF if n in lab and n in data and n in st]
            model = build_model(kind, params, dev, torch.device("cpu"))
            model.load_state_dict(torch.load(de / "model.pth", map_location="cpu")); model.eval()
            p = predict(model, data, names, dev)
            rf = np.concatenate([routing_free(next(iter(DataLoader([data[n] for n in names[s:s + 256]], batch_size=len(names[s:s + 256])))), dev, rf_cache).numpy()
                                 for s in range(0, len(names), 256)]) > 0.5
            Fh = np.array([predF[n] for n in names], float)
            Ft = np.stack([data[n].y.reshape(-1).numpy() for n in names])
            Et = np.stack([lab[n]["ESP"].numpy() for n in names])
            Eh = np.exp(np.minimum(p["logf"], 0)) * np.exp(p["decay"])
            logical = np.array([len(np.unique(data[n].gate_qubits[data[n].gate_qubits >= 0].numpy())) for n in names])
            act_true = np.stack([st[n].numpy()[:, 3] for n in names])
            okF = (Ft >= 0).all(1)
            okE = np.isfinite(Et).all(1)
            m = {"pairs routing-free": float(rf.mean()), "circuits all routing-free": float(rf.all(1).mean()),
                 "compilations per circuit (hybrid)": float(rf.sum(1).mean()),
                 "flagged routing-free but ancillas used": float((act_true[rf] > logical[:, None].repeat(3, 1)[rf]).mean()) if rf.any() else 0.0}
            for tag, ok, true, hat in (("F", okF, Ft, Fh), ("ESP", okE, Et, Eh)):
                hyb = np.where(rf, true, hat)
                for lab_, sc in (("model", hat), ("hybrid", hyb)):
                    rd, rg = choice(sc[ok], true[ok])
                    m[f"{tag} regret, {lab_}"], m[f"{tag} right device, {lab_}"] = rg, rd
            res[split].append(m)
            print(seed, split, json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
    keys = list(next(iter(res.values()))[0])
    L = [f"# Hybrid rule: compile routing-free (circuit, device) pairs, predict the rest", "",
         f"F from {args.f_run}, ESP from {args.esp_run}; mean over seeds. 'Compile everything' has regret 0 and 3 compilations per circuit.", "",
         "| | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{np.mean([m[k] for m in res[s]]):.4f}" for s in res) + " |")
    (HERE / f"results/report_hybrid{os.environ.get('GS_TAG', '')}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
