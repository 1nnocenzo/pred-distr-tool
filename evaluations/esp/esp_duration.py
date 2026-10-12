#!/usr/bin/env python3
"""How large is the error on the predicted circuit duration (QPU time of one shot), for a scheduler?

Per run, split and seed (test circuits with ESP labels): true duration of the stored compilation (ASAP schedule,
µs), predicted duration; relative error |pred/true − 1| (median, p90), absolute error in µs (median, p90), bias
(median pred/true), share of cases within 10% / 25%, per device too.
    python esp_duration.py --runs v8kdi10,v8kda10,v8kde10      # -> results/report_esp_duration{TAG}.md (+ .json)
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
sys.path.insert(0, str(HERE))
from esp_eval import DEVS, LABELS, SPLITS, predict  # noqa: E402
from esp_probe import split_dir  # noqa: E402
from esp_budget import KINDS, ROOT  # noqa: E402
from gsv3 import paths  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

KIND_DEFAULT = {"v8kde10": "sinkhorn_rf_route_czc_esp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="v8kdi10,v8kda10,v8kde10")
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    out = {}
    for run in args.runs.split(","):
        kind = KIND_DEFAULT.get(run, KINDS.get(run, "sinkhorn_rf_route_czc_espa"))
        per = defaultdict(lambda: defaultdict(list))
        for seed in [int(s) for s in args.seeds.split(",")]:
            for split, sd in SPLITS.items():
                d = split_dir(ROOT, run, seed, sd)
                if d is None:
                    continue
                model = build_model(kind, params, dev, torch.device("cpu"))
                model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
                names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data]
                p = predict(model, data, names, dev)
                true = np.stack([lab[n]["duration"].numpy() for n in names]) * 1e6
                pred = np.exp(p["logdur"])
                per[split]["true"].append(true); per[split]["pred"].append(pred)
        res = {}
        for split, v in per.items():
            t, pr = np.concatenate(v["true"]), np.concatenate(v["pred"])
            rel, ab = np.abs(pr / t - 1), np.abs(pr - t)
            row = {"cases": int(t.size), "true duration median µs": float(np.median(t)),
                   "rel. error median": float(np.median(rel)), "rel. error p90": float(np.percentile(rel, 90)),
                   "abs. error median µs": float(np.median(ab)), "abs. error p90 µs": float(np.percentile(ab, 90)),
                   "bias (median pred/true)": float(np.median(pr / t)),
                   "within 10%": float((rel < 0.10).mean()), "within 25%": float((rel < 0.25).mean())}
            for di, dn in enumerate(DEVS):
                row[f"rel. error median, {dn}"] = float(np.median(rel[:, di]))
            res[split] = row
            print(run, split, json.dumps({k: round(x, 4) for k, x in row.items()}), flush=True)
        out[run] = res
    (HERE / f"results/report_esp_duration{args.tag}.json").write_text(json.dumps(out))
    L = ["# Predicted circuit duration (QPU time of one shot, ASAP schedule): how large is the error?", "",
         "Seeds 5–7 pooled; (circuit, device) cases; relative error = |pred/true − 1|.", ""]
    for run, res in out.items():
        keys = list(next(iter(res.values())))
        L += [f"## {run}", "", "| | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(f"{res[s][k]:.3f}" if not k.startswith("cases") else str(res[s][k]) for s in res) + " |")
        L.append("")
    (HERE / f"results/report_esp_duration{args.tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
