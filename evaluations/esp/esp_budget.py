#!/usr/bin/env python3
"""Error budget of the ESP prediction: which link of  ESP = F · exp(decay),
decay = −Σ_active (duration − busy_q) / min(T1, T2)_q,  loses the most?

For an end-to-end ``*_espa`` run, on each split's test circuits, the model's own prediction is compared with
oracle swaps, one link at a time (true values from esp_labels.pt and struct_labels.pt):
  model            F̂ · exp(decaŷ)
  true F           F  · exp(decaŷ)          -> what is left once F is perfect
  true decay       F̂ · exp(decay)           -> what is left once the decay is perfect
  true duration    F̂ · exp(decay from the TRUE duration, the model's busy times and placement)
  true structure   F̂ · exp(decay from the head fed the TRUE busiest-qubit gate time / measurements / operations)
  F only / true F only    ESP ≈ F̂ / ESP ≈ F (how differently F and ESP rank the devices)
plus duration / depth with the true structure, the structure error per family, and the size of the decay.
    python esp_budget.py --run v8kda10 [--seeds 5,6,7]      # -> results/report_esp_budget_<run>.md (+ .json)
"""
from __future__ import annotations

import argparse
import json
import math
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
from gsv3 import paths  # noqa: E402
from gsv3.devices import GATE_US, variant_batch  # noqa: E402
from gsv3.model import SinkhornPlacementPredictor, build_model, load_hparams  # noqa: E402

STRUCT = EVAL / "compile_check/results/struct_labels.pt"
ROOT = EVAL / "generalization_v9/results"
KINDS = {"v5_espa": "sinkhorn_espa", "v8a_espa": "sinkhorn_espa", "v8kdi10": "sinkhorn_rf_route_czc_espi"}


def choice(score, esp):
    """right-device share and ESP regret when choosing argmax(score)."""
    pick = score.argmax(1)
    return float((pick == esp.argmax(1)).mean()), float((esp.max(1) - esp[np.arange(len(esp)), pick]).mean())


@torch.no_grad()
def run_split(model, data, lab, st, dev, names):
    feats = features(model, data, names, dev)
    coh_mean = [float(dev.coh[dev.batch == d].mean()) for d in range(len(DEVS))]
    out = defaultdict(list)
    for f in feats:
        nm = f["names"]
        dec, ldur, ldep = model.esp_from_features(model.esp_mlp, f, model.esp_head)
        tS = torch.stack([st[n] if n in st else torch.full((len(DEVS), 4), math.nan) for n in nm])   # (B, D, 4)
        f2 = dict(f)
        for i, k in enumerate(("tmax", "mmax", "omax")):
            f2[k] = torch.where(torch.isfinite(tS[..., i]), tS[..., i], f[k])
        dec_s, ldur_s, ldep_s = model.esp_from_features(model.esp_mlp, f2, model.esp_head)
        dur_t = torch.stack([lab[n]["duration"].float() * 1e6 for n in nm])
        dec_d = -(((dur_t.unsqueeze(-1) - f["busy"]).clamp(min=0) * f["active"]) * f["inv_t"]).sum(-1)
        # device-average 1 / min(T1, T2) instead of the placed qubits' (what a uniform placement reads)
        inv_u = f["inv_t"].new_tensor(coh_mean).view(1, -1, 1).expand_as(f["inv_t"])
        dur_m = torch.exp(ldur)
        dec_u = -(((dur_m.unsqueeze(-1) - f["busy"]).clamp(min=0) * f["active"]) * inv_u).sum(-1)
        dec_du = -(((dur_t.unsqueeze(-1) - f["busy"]).clamp(min=0) * f["active"]) * inv_u).sum(-1)
        for k, v in (("logf", f["logf"]), ("dec", dec), ("ldur", ldur), ("ldep", ldep), ("dec_s", dec_s),
                     ("ldur_s", ldur_s), ("ldep_s", ldep_s), ("dec_d", dec_d), ("dec_u", dec_u), ("dec_du", dec_du), ("tS", tS),
                     ("pS", torch.stack([f["tmax"], f["mmax"], f["omax"]], -1))):
            out[k].append(v.numpy())
        out["names"] += nm
    return {k: (np.concatenate(v) if k != "names" else v) for k, v in out.items()}


def budget(p, lab):
    names = p["names"]
    T = {k: np.stack([lab[n][k].numpy() for n in names]) for k in ("F", "ESP", "duration", "depth", "decay")}
    ok = np.isfinite(T["decay"]).all(1) & np.isfinite(p["dec_d"]).all(1)
    fhat = np.exp(np.minimum(p["logf"], 0))
    esp = T["ESP"][ok]
    variants = {"model": fhat * np.exp(p["dec"]), "true F": T["F"] * np.exp(p["dec"]),
                "true decay": fhat * np.exp(T["decay"]), "true duration": fhat * np.exp(p["dec_d"]),
                "true structure": fhat * np.exp(p["dec_s"]), "device-average T": fhat * np.exp(p["dec_u"]),
                "true duration + device-average T": fhat * np.exp(p["dec_du"]), "F only": fhat, "true F only": T["F"]}
    m = {"n": int(ok.sum())}
    for k, v in variants.items():
        v = v[ok]
        big = esp > 1e-3
        rd, rg = choice(v, esp)
        m[k] = {"R² ESP": r2(v.ravel(), esp.ravel()),
                "R² log ESP": r2(np.log(np.maximum(v[big], 1e-12)), np.log(esp[big])),
                "right device": rd, "regret": rg}
    ld, lp = np.log(T["duration"] * 1e6), np.log(np.maximum(T["depth"], 1))
    err = lambda a, b: float(np.median(np.abs(np.exp(a - b) - 1)))
    m["duration"] = {"model": err(p["ldur"], ld), "true structure": err(p["ldur_s"], ld),
                     "R² model": r2(p["ldur"].ravel(), ld.ravel()), "R² true structure": r2(p["ldur_s"].ravel(), ld.ravel())}
    m["depth"] = {"model": err(p["ldep"], lp), "true structure": err(p["ldep_s"], lp)}
    m["decay"] = {"mean |true|": float(np.abs(T["decay"][ok]).mean()), "mean |pred|": float(np.abs(p["dec"][ok]).mean()),
                  "corr": float(np.corrcoef(p["dec"][ok].ravel(), T["decay"][ok].ravel())[0, 1]),
                  "mean |true| per device": np.abs(T["decay"][ok]).mean(0).round(4).tolist()}
    # structure error per family
    fam = np.array([n.split("/")[0] for n in names])
    tS, pS = p["tS"][..., :3], p["pS"]
    fm = {}
    for f_ in sorted(set(fam)):
        s = (fam == f_)[:, None] & np.isfinite(tS[..., 0])
        if s.sum() < 30:
            continue
        rel = np.abs(pS[s] / np.maximum(tS[s], 1e-6) - 1)
        fm[f_] = {"n": int(s.sum()), "tmax err": float(np.median(rel[:, 0])), "mmax err": float(np.median(np.abs(pS[s][:, 1] - tS[s][:, 1]))),
                  "omax err": float(np.median(rel[:, 2])),
                  "duration err": err(p["ldur"][s], ld[s]), "duration err, true structure": err(p["ldur_s"][s], ld[s]),
                  "true mmax": float(np.median(tS[s][:, 1])), "pred mmax": float(np.median(pS[s][:, 1]))}
    m["families"] = fm
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v8kda10")
    ap.add_argument("--seeds", default="5,6,7")
    args = ap.parse_args()
    torch.set_num_threads(8)
    kind = KINDS.get(args.run, "sinkhorn_rf_route_czc_espa")
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    st = torch.load(STRUCT, weights_only=False)
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
            m = budget(run_split(model, data, lab, st, dev, names), lab)
            res[split].append(m)
            print(args.run, seed, split, {k: round(m[k]["regret"], 4) for k in ("model", "true F", "true decay", "true duration", "true structure", "device-average T", "true duration + device-average T", "F only", "true F only")},
                  m["duration"], flush=True)
    (HERE / f"results/report_esp_budget_{args.run}.json").write_text(json.dumps(res))
    mean = lambda xs: float(np.mean(xs))
    L = [f"# ESP error budget: {args.run}", "", "Mean over seeds. Each row swaps one link for the truth; regret = mean ESP lost by the chosen device.", ""]
    for metric in ("regret", "right device", "R² ESP", "R² log ESP"):
        L += [f"## {metric}", "", "| variant | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
        for v in ("model", "true F", "true decay", "true duration", "true structure", "device-average T",
                  "true duration + device-average T", "F only", "true F only"):
            L.append(f"| {v} | " + " | ".join(f"{mean([m[v][metric] for m in res[s]]):.4f}" for s in res) + " |")
        L.append("")
    L += ["## Duration and depth: model vs the head fed the true per-qubit structure", "",
          "| | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
    for k, sub in (("duration error, model", ("duration", "model")), ("duration error, true structure", ("duration", "true structure")),
                   ("R² log duration, model", ("duration", "R² model")), ("R² log duration, true structure", ("duration", "R² true structure")),
                   ("depth error, model", ("depth", "model")), ("depth error, true structure", ("depth", "true structure")),
                   ("mean |decay|, true", ("decay", "mean |true|")), ("mean |decay|, predicted", ("decay", "mean |pred|")),
                   ("corr(decay pred, true)", ("decay", "corr"))):
        L.append(f"| {k} | " + " | ".join(f"{mean([m[sub[0]][sub[1]] for m in res[s]]):.3f}" for s in res) + " |")
    L += ["", "## Per family (test families of every split, mean over seeds)", "",
          "| family | n | tmax error | mmax abs. error (pred / true median) | omax error | duration error | duration error, true structure |",
          "| --- |" + " --- |" * 6]
    fams = defaultdict(list)
    for s in res:
        for m in res[s]:
            for f_, v in m["families"].items():
                fams[(s, f_)].append(v)
    for (s, f_), vs in sorted(fams.items()):
        if s == "control":
            continue
        g = lambda k: mean([v[k] for v in vs])
        L.append(f"| {f_} ({s}) | {int(g('n'))} | {g('tmax err'):.2f} | {g('mmax err'):.2f} ({g('pred mmax'):.2f} / {g('true mmax'):.2f}) | "
                 f"{g('omax err'):.2f} | {g('duration err'):.2f} | {g('duration err, true structure'):.2f} |")
    (HERE / f"results/report_esp_budget_{args.run}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
