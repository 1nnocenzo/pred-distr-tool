#!/usr/bin/env python3
"""How well does a ``*_esp`` model predict duration, depth and estimated success probability (ESP)?

For each seed and standard split of a run (default v8kde10), on that split's test circuits with ESP labels
(compile_check/results/esp_labels.pt: the stored seed-0 compilation, QExa20 T1/T2 unit fix):
  R² of F (sanity, as metrics.json), of log duration and log depth, median relative error of duration / depth;
  ESP: R² of ESP = F · exp(decay) and of log ESP (ESP > 1e-3), against the baseline "ESP ≈ predicted F"
  (no decay); device choice by ESP: right-device share and ESP regret, model vs choosing by predicted F.
    python esp_eval.py [--run v8kde10]          # -> results/report_esp_<run>.md
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from torch_geometric.loader import DataLoader  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

ROOT = EVAL / os.environ.get("GS_RESULTS_ROOT", "generalization_v9") / "results"
LABELS = EVAL / "compile_check/results/esp_labels.pt"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
SPLITS = {"LOFO qaoa": "leave_one_family_out/qaoa", "LOFO qnn": "leave_one_family_out/qnn",
          "LOGO variational": "leave_one_group_out/variational", "control": "random_control/random_seed{s}"}


def r2(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    return float(1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12))


@torch.no_grad()
def predict(model, data, names, dev):
    out = {k: [] for k in ("logf", "decay", "logdur", "logdep")}
    for s in range(0, len(names), 64):
        b = next(iter(DataLoader([data[n] for n in names[s: s + 64]], batch_size=len(names[s: s + 64]))))
        logf = model(b, dev)
        decay, logdur, logdep = model.last_esp
        for k, v in zip(out, (logf, decay, logdur, logdep)):
            out[k].append(v.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate(run, seed, split_dir, params, data, lab, dev, root=ROOT, kind="sinkhorn_rf_route_czc_esp"):
    sfx = "" if seed == 5 else f"_s{seed}"
    d = root / f"{run}{sfx}" / split_dir.format(s=seed)
    if not (d / "model.pth").is_file():
        return None
    names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data]
    model = build_model(kind, params, dev, torch.device("cpu"))
    model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
    return metrics(predict(model, data, names, dev), names, data, lab)


def metrics(p, names, data, lab):
    """``p``: per-circuit arrays (B, D) of logf and, when ESP heads exist, decay / logdur / logdep."""
    t = {k: np.stack([lab[n][k].numpy() for n in names]) for k in ("F", "ESP", "duration", "depth", "decay")}
    y = np.stack([data[n].y.reshape(-1).numpy() for n in names])
    f_pred = np.exp(np.minimum(p["logf"], 0))
    heads = "decay" in p
    esp_pred = f_pred * np.exp(p["decay"]) if heads else f_pred
    dur_t, dep_t = np.log(t["duration"] * 1e6), np.log(np.maximum(t["depth"], 1))
    ok = np.isfinite(t["decay"])
    big = ok & (t["ESP"] > 1e-3)
    m = {"n": len(names), "R² F": r2(f_pred[y >= 0], y[y >= 0])}
    if heads:
        m.update({"R² log duration": r2(p["logdur"].ravel(), dur_t.ravel()),
                  "duration error (median)": float(np.median(np.abs(np.exp(p["logdur"] - dur_t) - 1))),
                  "R² log depth": r2(p["logdep"].ravel(), dep_t.ravel()),
                  "depth error (median)": float(np.median(np.abs(np.exp(p["logdep"] - dep_t) - 1))),
                  "R² ESP": r2(esp_pred[ok], t["ESP"][ok])})
    m.update({"R² ESP, baseline F": r2(f_pred[ok], t["ESP"][ok]),
              "R² log ESP (ESP > 1e-3)": r2(np.log(np.maximum(esp_pred[big], 1e-12)), np.log(t["ESP"][big]))})
    rows = ok.all(1)
    true_best = t["ESP"][rows].argmax(1)
    for lab_, score in ((("model", esp_pred[rows]),) if heads else ()) + (("baseline F", f_pred[rows]),):
        pick = score.argmax(1)
        m[f"right device by ESP, {lab_}"] = float((pick == true_best).mean())
        m[f"ESP regret, {lab_}"] = float((t["ESP"][rows].max(1) - t["ESP"][rows][np.arange(rows.sum()), pick]).mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v8kde10")
    ap.add_argument("--root", type=Path, default=ROOT, help="results root of the run")
    ap.add_argument("--kind", default="sinkhorn_rf_route_czc_esp", help="model kind (…_esp or …_espa)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    res = {}
    for name, sd in SPLITS.items():
        for seed in (5, 6, 7):
            m = evaluate(args.run, seed, sd, params, data, lab, dev, args.root, args.kind)
            if m is not None:
                res.setdefault(name, []).append(m)
                print(name, seed, {k: round(v, 4) for k, v in m.items()}, flush=True)
    keys = [k for k in next(iter(res.values()))[0] if k != "n"]
    L = [f"# ESP, duration and depth predicted by {args.run}", "",
         "Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, "
         "ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.", "",
         "| | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
    for k in keys:
        cells = []
        for name in res:
            v = np.array([m[k] for m in res[name]])
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append(f"| seeds | " + " | ".join(str(len(res[n])) for n in res) + " |")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"report_esp_{args.run}{os.environ.get('GS_TAG', '')}.md").write_text("\n".join(L))
    (HERE / "results" / f"esp_eval_{args.run}{os.environ.get('GS_TAG', '')}.json").write_text(json.dumps({"run": args.run, "kind": args.kind, "res": res}))
    print("\n".join(L))


if __name__ == "__main__":
    main()
