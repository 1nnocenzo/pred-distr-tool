#!/usr/bin/env python3
"""v11 usage-map labels from the pilot compilations (``v8b_variants.py run --usage``).

For every circuit, the usage maps of the best of K compilations, in device-graph order:
  r, measure (n_dev, P_max) per physical qubit · cz (n_dev, E_max) per coupler, couplers in the
  order of the device's raw ``edges`` (= ``coupler_pairs`` of gsv3.devices).
Original devices from ``--orig-dir`` (key ``<dev>/orig``); variants ``<dev>/aug<j>`` from
``--var-dir`` (A = n_aug).  Writes ``usage_labels.pt``:
  {circuit: {"F": (D,), "r": (D, P), "m": (D, P), "cz": (D, E),            original devices
             "F_aug": (A, D), "r_aug": (A, D, P), "m_aug": (A, D, P), "cz_aug": (A, D, E)}}
Also checks the head's formula on the labels: −Σ U·ε against log F of the best compilation.
    python merge_usage.py --orig-dir results/v8u_orig --var-dir results/v8u --n-aug 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "generalization_v3"))
from gsv3 import paths  # noqa: E402

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")


def maps(u: dict, edge_idx: dict, p_max: int, e_max: int):
    r = torch.zeros(p_max); m = torch.zeros(p_max); cz = torch.zeros(e_max)
    r[: len(u["r"])] = torch.tensor(u["r"], dtype=torch.float32)
    m[: len(u["measure"])] = torch.tensor(u["measure"], dtype=torch.float32)
    for a, b, c in u["cz"]:
        cz[edge_idx[(min(a, b), max(a, b))]] += c          # KeyError = CZ on a non-coupler: a bug
    return r, m, cz


def read(directory: Path) -> dict:
    out = {}
    for f in sorted(directory.glob("task*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r and "usage" in r:
                out[r["circuit"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", type=Path, required=True)
    ap.add_argument("--var-dir", type=Path, default=None)
    ap.add_argument("--n-aug", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    p_max = max(len(raw[d]["nodes"]) for d in DEVICES)
    e_max = max(len(raw[d]["edges"]) for d in DEVICES)
    edge_idx = [{(min(a, b), max(a, b)): i for i, (a, b) in enumerate(raw[d]["edges"])} for d in DEVICES]
    eps_q = [torch.tensor(raw[d]["nodes"], dtype=torch.float64)[:, :2] for d in DEVICES]
    eps_e = [torch.tensor(raw[d]["edge_feats"], dtype=torch.float64) for d in DEVICES]
    orig = read(args.orig_dir)
    var = read(args.var_dir) if args.var_dir else {}
    labels, check, dropped = {}, [], 0
    for name, rec in orig.items():
        ent = {"F": torch.tensor([max(rec["F"][f"{d}/orig"]) for d in DEVICES])}
        rr, mm, cc, bad = [], [], [], False
        for i, d in enumerate(DEVICES):
            r, m, cz = maps(rec["usage"][f"{d}/orig"], edge_idx[i], p_max, e_max)
            rr.append(r); mm.append(m); cc.append(cz)
            n, e = len(eps_q[i]), len(eps_e[i])
            pred = -(r[:n].double() @ eps_q[i][:, 0] + m[:n].double() @ eps_q[i][:, 1] + cz[:e].double() @ eps_e[i])
            if ent["F"][i] > 1e-6:      # below, the labels are rounded to 10 decimals
                check.append((float(pred), float(np.log(ent["F"][i]))))
                bad |= abs(float(pred) - float(np.log(ent["F"][i]))) > 1e-3
        if bad:     # maps that do not reproduce the label: dynamic circuits (if_else branches)
            dropped += 1
            continue
        ent.update(r=torch.stack(rr), m=torch.stack(mm), cz=torch.stack(cc))
        v = var.get(name)
        if v is not None and all(f"{d}/aug{j}" in v["usage"] for d in DEVICES for j in range(args.n_aug)):
            F, R, M, C = [], [], [], []
            for j in range(args.n_aug):
                F.append([max(v["F"][f"{d}/aug{j}"]) for d in DEVICES])
                per = [maps(v["usage"][f"{d}/aug{j}"], edge_idx[i], p_max, e_max) for i, d in enumerate(DEVICES)]
                R.append(torch.stack([x[0] for x in per])); M.append(torch.stack([x[1] for x in per]))
                C.append(torch.stack([x[2] for x in per]))
            ent.update(F_aug=torch.tensor(F), r_aug=torch.stack(R), m_aug=torch.stack(M), cz_aug=torch.stack(C))
        labels[name] = ent
    out = args.out or (args.var_dir or args.orig_dir) / "usage_labels.pt"
    torch.save(labels, out)
    c = np.array(check)
    err = np.abs(c[:, 0] - c[:, 1])
    print(f"{len(labels)} circuits ({sum('F_aug' in e for e in labels.values())} with variants) -> {out}; "
          f"{dropped} dropped: maps do not reproduce log F (dynamic circuits)")
    print(f"formula check before dropping (F > 1e-6): |−Σ U·ε − log F| median {np.median(err):.2e}, "
          f"99th pct {np.percentile(err, 99):.2e}, max {err.max():.2e}  ({len(c)} cases)")


if __name__ == "__main__":
    main()
