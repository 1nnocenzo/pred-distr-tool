#!/usr/bin/env python3
"""Routed circuits only: where do the duration and ESP errors come from?  (the user's priority, 2026-10-13)

Per run, the routed (circuit, device) cases of every split (seeds pooled), against the stored compilation's labels
(esp_labels.pt): duration error |pred/true − 1| and bias (median pred/true), ESP error on log ESP (|Δ log ESP|,
median) and ESP R², binned by
  routing overhead  true CZ count / CZ count of the circuit's gates decomposed alone (decomp_table.json) − 1
  circuit size      logical qubits
  device
so a scheduler can see which routed circuits are trustworthy.
    GS_RESULTS_ROOT=generalization_v10 python esp_routed.py --runs v8kdfi10      # -> results/report_esp_routed_<tag>.md
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
from gsv3 import fast_encoding as fe  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

COUNTS = EVAL / "compile_check/results/compiler_counts_v8.pt"


@torch.no_grad()
def collect(model, data, dev, names):
    out = defaultdict(list)
    for s in range(0, len(names), 64):
        nm = names[s: s + 64]
        b = next(iter(DataLoader([data[n] for n in nm], batch_size=len(nm))))
        logf = model(b, dev)
        decay, ldur, _ = model.last_esp
        out["dur"].append(torch.exp(ldur).numpy())
        out["logesp"].append((logf.clamp(max=0) + decay).numpy())
        out["rf"].append(routing_free(b, dev, model._rf_cache).numpy() > 0.5)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="v8kdfi10")
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    cnt = torch.load(COUNTS, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    U = list(fe.UNIQUE)
    tab = json.loads((HERE / "results/decomp_table.json").read_text())
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    L = ["# Routed circuits: duration and ESP errors by routing overhead, size and device", "",
         "Routed (circuit, device) cases only (not routing-free on that device), seeds pooled. Duration error = median "
         "|pred/true − 1|, bias = median pred/true; ESP error = median |log ESP_pred − log ESP_true| (×100 ≈ % error); "
         "routing overhead = true CZ / CZ of the gates decomposed alone − 1.", ""]
    for run in args.runs.split(","):
        rows = defaultdict(list)
        for seed in [int(s) for s in args.seeds.split(",")]:
            for split, sd in SPLITS.items():
                d = split_dir(ROOT, run, seed, sd)
                if d is None:
                    continue
                model = build_model(KINDS[run], params, dev, torch.device("cpu"))
                model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
                names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data and n in cnt]
                p = collect(model, data, dev, names)
                t_dur = np.stack([lab[n]["duration"].numpy() for n in names]) * 1e6
                t_esp = np.stack([lab[n]["ESP"].numpy() for n in names])
                czt = np.stack([cnt[n].numpy()[:, 2] for n in names])
                czd = np.array([float((data[n].x[:, : len(U)].numpy() @ tc).sum()) for n in names])
                nq = np.array([int(data[n].n_qubits) for n in names])
                for di in range(len(DEVS)):
                    sel = ~p["rf"][:, di]
                    rows["split"] += [split] * int(sel.sum())
                    rows["dev"] += [DEVS[di]] * int(sel.sum())
                    rows["r_dur"] += list(p["dur"][sel, di] / t_dur[sel, di])
                    rows["d_lesp"] += list(p["logesp"][sel, di] - np.log(np.clip(t_esp[sel, di], 1e-300, None)))
                    rows["t_lesp"] += list(np.log(np.clip(t_esp[sel, di], 1e-300, None)))
                    rows["p_esp"] += list(np.exp(p["logesp"][sel, di])); rows["t_esp"] += list(t_esp[sel, di])
                    rows["ovh"] += list(czt[sel, di] / np.maximum(czd[sel], 1) - 1)
                    rows["nq"] += list(nq[sel])
                print(run, seed, split, len(names), flush=True)
        R = {k: np.array(v) for k, v in rows.items()}
        ok = R["t_esp"] > 1e-3                              # ESP error only where ESP is not ~0 (as esp_eval's log ESP R²)

        fin = np.isfinite(R["p_esp"]) & np.isfinite(R["d_lesp"])

        def line(name, s):
            if s.sum() < 30:
                return None
            se = s & ok & fin
            s = s & fin if (s & fin).sum() >= 30 else s
            r2 = 1 - ((R["p_esp"][s] - R["t_esp"][s]) ** 2).sum() / max(((R["t_esp"][s] - R["t_esp"][s].mean()) ** 2).sum(), 1e-12)
            return (f"| {name} | {int(s.sum())} | {np.median(np.abs(R['r_dur'][s] - 1)):.3f} | {np.percentile(np.abs(R['r_dur'][s] - 1), 90):.3f} | "
                    f"{np.median(R['r_dur'][s]):.2f} | {np.median(np.abs(R['d_lesp'][se])) if se.sum() else float('nan'):.3f} | {r2:.3f} |")
        head = ["| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"## {run}", ""]
        for split in SPLITS:
            if split not in set(R["split"]):
                continue
            S = R["split"] == split
            L += [f"### {split}", ""] + head
            groups = [("all routed", S)]
            for lo, hi in ((0, 0.25), (0.25, 0.75), (0.75, 1.5), (1.5, 99)):
                groups.append((f"routing overhead {lo}–{hi}", S & (R["ovh"] >= lo) & (R["ovh"] < hi)))
            for lo, hi in ((2, 6), (6, 11), (11, 16), (16, 21)):
                groups.append((f"{lo}–{hi - 1} qubits", S & (R["nq"] >= lo) & (R["nq"] < hi)))
            for dn in DEVS:
                groups.append((dn, S & (R["dev"] == dn)))
            L += [x for x in (line(n, s) for n, s in groups) if x] + [""]
    out = HERE / f"results/report_esp_routed{args.tag}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
