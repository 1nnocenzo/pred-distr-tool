#!/usr/bin/env python3
"""Does the model know how big the compiled circuit will be?  Predicted vs true native-gate counts.

The count head predicts, per gate and device, how many r / measure / cz operations the gate
becomes; summed over the circuit these are the predicted counts (trained by the v8a count
loss).  True counts come from the compiled circuits (compile_check, Qiskit level 2).
  A) unseen families (LOFO qaoa, LOFO qnn, LOGO variational test sets): ratio predicted/true,
     by kind and by circuit size; for cz also against the logical two-qubit gate count
     (= what the compiler would emit with no routing and no optimisation).
  B) calibration changes (control model, calibration-shift sample, the 8 v8b variants): does the
     predicted change of the cz count follow the compiler's (SWAPs added or removed)?
    python counts_check.py [--model v8a_c001] [--kind sinkhorn]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import load_devices, variant_batch  # noqa: E402
from gsv3.groups import GROUPS  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402
from genstudy.data import DEVICE_NAMES  # noqa: E402

COUNTS = EVAL / "compile_check/results/layouts_all/compiler_counts.pt"
V8B = EVAL / "compile_check/results/v8b"
V8B_ORIG = EVAL / "compile_check/results/v8b_orig/v8b_orig_labels.pt"
KIND = ("r", "measure", "cz")
OFF = 28   # arity column in the gate features = OFF + 6


def predict_counts(model, graphs, devices):
    out = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=32):
            model(b, devices)
            out.append(model.last_counts)
    return torch.cat(out).numpy()          # (N, D, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v8a_c001")
    ap.add_argument("--kind", default="sinkhorn")
    ap.add_argument("--root", default="generalization_v8")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    devices = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    counts = torch.load(COUNTS, weights_only=False)
    root = EVAL / args.root / "results" / args.model

    print(f"== {args.model}: predicted / true native-gate counts (median ratio, mean of log2 ratio)\n")
    print("A) unseen families")
    splits = {"LOFO qaoa": ("leave_one_family_out/qaoa", {"qaoa"}),
              "LOFO qnn": ("leave_one_family_out/qnn", {"qnn"}),
              "LOGO variational": ("leave_one_group_out/variational", set(GROUPS["variational"]))}
    for name, (rel, fams) in splits.items():
        mp = root / rel / "model.pth"
        if not mp.is_file():
            print(f"   {name}: no model yet"); continue
        model = build_model(args.kind, params, devices, torch.device("cpu"))
        model.load_state_dict(torch.load(mp, map_location="cpu")); model.eval()
        gs = [d for d in data if d.circuit_name.split("/")[0] in fams and d.circuit_name in counts]
        P = predict_counts(model, gs, devices)
        T = np.stack([counts[d.circuit_name].numpy() for d in gs])          # (N, D, 3)
        L2 = np.array([int((d.x[:, OFF + 6] == 2).sum()) for d in gs])     # logical 2q gates
        nq = np.array([int(re.search(r"_q(\d+)_", d.circuit_name).group(1)) for d in gs])
        fam = np.array([d.circuit_name.split("/")[0] for d in gs])
        print(f"   {name} ({len(gs)} circuits)")
        for f in sorted(set(fam)) if len(set(fam)) > 1 else [None]:
            s = fam == f if f else np.ones(len(gs), bool)
            cells = []
            for k in range(3):
                t, p = T[s][..., k], P[s][..., k]
                ok = t > 0
                r = p[ok] / t[ok]
                cells.append(f"{KIND[k]} {np.median(r):.2f}" if ok.any() else f"{KIND[k]} –")
            cz_route = T[s][..., 2].mean() / max(L2[s].mean(), 1e-9)
            print(f"     {(f or 'all'):16} {'  '.join(cells)}   | true cz / logical 2q {cz_route:.2f}, predicted {P[s][..., 2].mean() / max(L2[s].mean(), 1e-9):.2f}")
        for lo, hi in ((2, 5), (6, 10), (11, 15), (16, 20)):
            s = (nq >= lo) & (nq <= hi)
            if not s.any():
                continue
            r = [np.median(P[s][..., k][T[s][..., k] > 0] / T[s][..., k][T[s][..., k] > 0]) if (T[s][..., k] > 0).any() else np.nan for k in range(3)]
            print(f"       q{lo:>2}-{hi:<2} r {r[0]:.2f}  meas {r[1]:.2f}  cz {r[2]:.2f}")

    print("\nB) calibration changes (control model, 8 v8b variants per device)")
    mp = root / "random_control/random_seed5/model.pth"
    labels = torch.load(V8B / "v8b_labels.pt", weights_only=False)
    orig = torch.load(V8B_ORIG, weights_only=False)
    raw = json.loads((V8B / "variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    dev_o = variant_batch(raw, list(DEVICE_NAMES), params["lap_pe"])
    model = build_model(args.kind, params, dev_o, torch.device("cpu"))
    model.load_state_dict(torch.load(mp, map_location="cpu")); model.eval()
    sample = json.loads((EVAL / "calibration_shift/results/sample.json").read_text())
    by = {d.circuit_name: d for d in data}
    gs = [by[n] for n in sample if n in labels and n in orig and n in by]
    P0 = predict_counts(model, gs, dev_o)
    dp, dt, t0 = [], [], []
    for j in range(8):
        Pj = predict_counts(model, gs, variant_batch(raw, [f"{d}/aug{j}" for d in DEVICE_NAMES], params["lap_pe"]))
        for i, g in enumerate(gs):
            for d in range(3):
                to = float(orig[g.circuit_name]["counts"][0, d, 2]); tv = float(labels[g.circuit_name]["counts"][j, d, 2])
                dt.append(tv - to); dp.append(Pj[i, d, 2] - P0[i, d, 2]); t0.append(to)
    dp, dt, t0 = map(np.array, (dp, dt, t0))
    ch = dt != 0
    print(f"   cases {len(dt)}; the compiler changes the cz count in {ch.mean():.0%} of them "
          f"(mean |Δcz| {np.abs(dt[ch]).mean():.1f}, {np.abs(dt[ch] / np.maximum(t0[ch], 1)).mean():.0%} of the count)")
    print(f"   predicted Δcz: mean |Δ| {np.abs(dp).mean():.2f}, corr with true Δcz {np.corrcoef(dp, dt)[0, 1]:+.2f}, "
          f"slope {np.polyfit(dt, dp, 1)[0]:+.3f}")


if __name__ == "__main__":
    main()
