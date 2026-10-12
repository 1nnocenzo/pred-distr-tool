#!/usr/bin/env python3
"""v19: how often the hard placement of the formula head (Hungarian on the Sinkhorn matrix) is the compiler's qubit.

The formula head reads the gate times and T1 / T2 of the physical qubit each logical qubit is put on, so a per-qubit
change is followed only where that qubit is the compiler's.  Per split (test circuits of each split model, seeds
5–7), over the (logical qubit, device) pairs with a known compiler layout (seed 0, original calibration):
  exact          hard qubit = compiler's qubit
  argmax         soft argmax = compiler's qubit (what the v6 layout loss measured)
  p(true)        mean soft probability on the compiler's qubit
  in region      hard qubit is one the compiler used for some logical qubit of the circuit
  hops           mean chip distance between the hard qubit and the compiler's
    python placement_acc.py --run v8kdg10 --kind sinkhorn_rf_route_czc_espf    # -> results/placement_acc_<run>.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import esp_eval as E
from gsv3.model import build_model, hard_placement, load_hparams

LAYOUTS = E.EVAL / "compile_check/results/compiler_layouts_v6pilot.pt"


@torch.no_grad()
def split_stats(model, data, names, lay, hops):
    acc = {k: [] for k in ("exact", "argmax", "p(true)", "in region", "hops")}
    per_dev = {d: [] for d in E.DEVS}
    for s in range(0, len(names), 64):
        chunk = names[s: s + 64]
        b = next(iter(DataLoader([data[n] for n in chunk], batch_size=len(chunk))))
        model(b, model._dev)
        soft = model.last_assign                       # (B, D, Qm, P)
        hard = hard_placement(soft).argmax(-1)         # (B, D, Qm)
        arg = soft.argmax(-1)
        for i, n in enumerate(chunk):
            t = lay.get(n)
            if t is None:
                continue
            for d in range(len(E.DEVS)):
                tq = t[:, d].numpy()
                k = tq >= 0
                if not k.any():
                    continue
                h, a_ = hard[i, d, : len(tq)].numpy()[k], arg[i, d, : len(tq)].numpy()[k]
                tt = tq[k]
                acc["exact"].append(h == tt); per_dev[E.DEVS[d]].append(h == tt)
                acc["argmax"].append(a_ == tt)
                acc["p(true)"].append(soft[i, d, np.flatnonzero(k), tt].numpy())
                acc["in region"].append(np.isin(h, tt))
                acc["hops"].append(hops[d][h, tt])
    out = {k: float(np.concatenate(v).mean()) for k, v in acc.items()}
    out.update({f"exact {d}": float(np.concatenate(v).mean()) for d, v in per_dev.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v8kdg10")
    ap.add_argument("--kind", default="sinkhorn_rf_route_czc_espf")
    ap.add_argument("--root", type=Path, default=E.EVAL / "generalization_v10/results")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(E.paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    raw = torch.load(E.paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = E.variant_batch(raw, list(E.DEVS), params["lap_pe"])
    lay = torch.load(LAYOUTS, weights_only=False)
    res = {}
    for split, sd in E.SPLITS.items():
        rows = []
        for seed in (5, 6, 7):
            sfx = "" if seed == 5 else f"_s{seed}"
            d = args.root / f"{args.run}{sfx}" / sd.format(s=seed)
            if not (d / "model.pth").is_file():
                continue
            names = [n for n in json.loads((d / "predictions.json").read_text()) if n in data and n in lay]
            model = build_model(args.kind, params, dev, torch.device("cpu"))
            model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
            model._dev = dev
            hops = [h.numpy() for h in model.hops]
            rows.append(split_stats(model, data, names, lay, hops) | {"n": len(names)})
            print(split, seed, rows[-1], flush=True)
        if rows:
            res[split] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}
    keys = [k for k in next(iter(res.values())) if k != "n"]
    L = [f"# Hard placement vs the compiler's layout: {args.run}", "",
         "Test circuits of each split, (logical qubit, device) pairs with a known compiler layout (seed 0, original "
         "calibration). Mean ± std over seeds 5–7. Random guess on a 20–27-qubit chip: exact ≈ 0.04–0.05.", "",
         "| | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
    for k in keys:
        fmt = "{:.2f} ± {:.2f}" if k == "hops" else "{:.3f} ± {:.3f}"
        L.append(f"| {k} | " + " | ".join(fmt.format(*res[s][k]) for s in res) + " |")
    out = E.HERE / "results" / f"placement_acc_{args.run}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
