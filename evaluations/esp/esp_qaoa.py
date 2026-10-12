#!/usr/bin/env python3
"""Why is the duration of qaoa (no measurements, much routing) still off?  Reverse engineering on one run.

For the LOFO qaoa test circuits of an ESP run: duration error split by routing-free / routed and by circuit size,
and against the error of the predicted gate counts (compiler_counts_v8.pt: r / measure / CZ per device) and of the
busiest qubit's gate time (struct_labels_v2.pt).  Also depth / busiest-qubit operations, true vs predicted.
    python esp_qaoa.py --run v8kdi10 [--seeds 5,6,7] [--split "LOFO qaoa"]   # -> results/report_esp_qaoa_<run>.md
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
from esp_eval import DEVS, LABELS, SPLITS  # noqa: E402
from esp_probe import split_dir  # noqa: E402
from esp_budget import KINDS, ROOT  # noqa: E402
from gsv3 import paths  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

COUNTS = EVAL / "compile_check/results/compiler_counts_v8.pt"
STRUCT2 = EVAL / "compile_check/results/struct_labels_v2.pt"


@torch.no_grad()
def collect(model, data, dev, names):
    out = defaultdict(list)
    for s in range(0, len(names), 64):
        nm = names[s: s + 64]
        b = next(iter(DataLoader([data[n] for n in nm], batch_size=len(nm))))
        model(b, dev)
        _, ldur, ldep = model.last_esp
        f = model.last_esp_feats
        out["dur"].append(torch.exp(ldur).numpy()); out["dep"].append(torch.exp(ldep).numpy())
        out["counts"].append(model.last_counts.numpy())
        out["tmax"].append(f["tmax"].numpy()); out["omax"].append(f["omax"].numpy())
        out["rf"].append(routing_free(b, dev, model._rf_cache).numpy())
        out["nq"].append(b.n_qubits.view(-1).numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v8kdi10")
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--split", default="LOFO qaoa")
    args = ap.parse_args()
    torch.set_num_threads(8)
    kind = KINDS.get(args.run, "sinkhorn_rf_route_czc_espa")
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    cnt = torch.load(COUNTS, weights_only=False)
    st = torch.load(STRUCT2, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    rows = defaultdict(list)
    for seed in [int(s) for s in args.seeds.split(",")]:
        d = split_dir(ROOT, args.run, seed, SPLITS[args.split])
        model = build_model(kind, params, dev, torch.device("cpu"))
        model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
        names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data and n in cnt and n in st]
        p = collect(model, data, dev, names)
        rows["dur_r"].append(np.log(p["dur"] / (np.stack([lab[n]["duration"].numpy() for n in names]) * 1e6)))
        T = np.stack([st[n].numpy() for n in names])                      # tmax, mmax, omax, active, busy_sum
        C = np.stack([cnt[n].numpy() for n in names])                     # r, measure, cz
        rows["dep_r"].append(np.log(p["dep"] / np.maximum(np.stack([lab[n]["depth"].numpy() for n in names]), 1)))
        rows["cz_r"].append(np.log((p["counts"][..., 2] + 1) / (C[..., 2] + 1)))
        rows["r_r"].append(np.log((p["counts"][..., 0] + 1) / (C[..., 0] + 1)))
        rows["tmax_r"].append(np.log((p["tmax"] + 1e-3) / (T[..., 0] + 1e-3)))
        rows["omax_r"].append(np.log((p["omax"] + 1) / (T[..., 2] + 1)))
        rows["depth_over_omax_true"].append(np.stack([lab[n]["depth"].numpy() for n in names]) / np.maximum(T[..., 2], 1))
        rows["depth_over_omax_pred"].append(p["dep"] / np.maximum(p["omax"], 1))
        rows["rf"].append(p["rf"]); rows["nq"].append(np.repeat(p["nq"][:, None], len(DEVS), 1))
        rows["true_cz"].append(C[..., 2])
    R = {k: np.concatenate(v).ravel() for k, v in rows.items()}
    med = lambda x: float(np.median(np.abs(np.exp(x) - 1)))
    sgn = lambda x: float(np.median(np.exp(x)))
    L = [f"# Reverse engineering: {args.run}, {args.split}", "", f"{len(R['dur_r'])} (circuit, device, seed) cases.", "",
         "| group | cases | duration error | duration pred/true (median) | CZ count pred/true | 1q count pred/true | busiest-qubit gate time pred/true |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    groups = [("all", np.ones_like(R["rf"], bool)), ("routing-free", R["rf"] > 0.5), ("routed", R["rf"] <= 0.5)]
    for lo, hi in ((1, 6), (6, 11), (11, 16), (16, 21), (21, 99)):
        groups.append((f"{lo}–{hi - 1} qubits", (R["nq"] >= lo) & (R["nq"] < hi)))
    for name, s in groups:
        if s.sum() < 10:
            continue
        L.append(f"| {name} | {int(s.sum())} | {med(R['dur_r'][s]):.2f} | {sgn(R['dur_r'][s]):.2f} | {sgn(R['cz_r'][s]):.2f} | "
                 f"{sgn(R['r_r'][s]):.2f} | {sgn(R['tmax_r'][s]):.2f} |")
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    L += ["", f"corr(log duration ratio, log CZ ratio) = {c(R['dur_r'], R['cz_r']):.2f}; "
          f"with log busiest-qubit gate time ratio = {c(R['dur_r'], R['tmax_r']):.2f}; with log 1q ratio = {c(R['dur_r'], R['r_r']):.2f}", "",
          "## Depth", "", "| | value |", "| --- | --- |",
          f"| depth error (median) | {med(R['dep_r']):.2f} |", f"| depth pred/true (median) | {sgn(R['dep_r']):.2f} |",
          f"| busiest-qubit operations pred/true (median) | {sgn(R['omax_r']):.2f} |",
          f"| depth / busiest-qubit ops, true: median [p10, p90] | {np.median(R['depth_over_omax_true']):.2f} [{np.percentile(R['depth_over_omax_true'], 10):.2f}, {np.percentile(R['depth_over_omax_true'], 90):.2f}] |",
          f"| depth / busiest-qubit ops, predicted: median [p10, p90] | {np.median(R['depth_over_omax_pred']):.2f} [{np.percentile(R['depth_over_omax_pred'], 10):.2f}, {np.percentile(R['depth_over_omax_pred'], 90):.2f}] |",
          f"| corr(log depth ratio, log busiest-ops ratio) | {c(R['dep_r'], R['omax_r']):.2f} |"]
    out = HERE / f"results/report_esp_qaoa_{args.run}_{args.split.replace(' ', '_')}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
