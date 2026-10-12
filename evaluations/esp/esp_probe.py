#!/usr/bin/env python3
"""ESP / duration / depth on top of every fidelity-model version (frozen backbones).

For each version (every run of calibration_shift/run_shift.py MODELS with the four standard splits, + v1) and each
seed / split, on that split's test circuits (labels: compile_check/results/esp_labels.pt, as esp_eval.py):
  F only   (every version): ESP ≈ the version's predicted F (predictions.json) -> device choice by ESP, regret.
  probes   (Sinkhorn versions): the fidelity model frozen, only an ESP head trained on the split's training circuits
           (all circuits outside its test set), so F is the version's own and the head reads what that version
           computes: ``v13`` (v8kde10's head: circuit / device embeddings) and ``a`` (v14 analytic head: structure
           only, see SinkhornPlacementPredictor.esp_from_features).  Features are computed once (one forward pass),
           then the MLP trains in seconds.
    python esp_probe.py run --run v8kdc10 [--seeds 5,6,7]     # -> results/probe/<run>_s<seed>_<split>.json
    python esp_probe.py report                                # -> results/report_esp_versions.md (+ .json)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EVAL / "calibration_shift"))
from esp_eval import DEVS, LABELS, SPLITS, metrics  # noqa: E402  (imports gsv3 too)
from gsv3 import paths  # noqa: E402
from gsv3.model import build_model, is_sinkhorn_kind, load_hparams  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

OUT = HERE / "results/probe"
HEADS = ("v13", "a")
SKIP = {"sinkhorn_rf_route_pgnn_usage"}


def versions() -> dict[str, tuple[str, Path]]:
    """run name -> (kind, results root): run_shift's MODELS (seed 5 entries) + v1, kept if its seed-5 (or any)
    directory has the four standard splits."""
    import run_shift
    out = {"v1": ("pooled", EVAL / "generalization_v3/results")}
    for key, (kind, path) in run_shift.MODELS.items():
        run = path.parents[2].name
        base = run[:-3] if run[-3:] in ("_s6", "_s7") else run
        out.setdefault(base, (kind, path.parents[3]))
    keep = {}
    for run, (kind, root) in out.items():
        if kind in SKIP or "_esp" in kind:
            continue
        if any(split_dir(root, run, s, sd) for s in (5, 6, 7) for sd in SPLITS.values()):
            n = sum(bool(split_dir(root, run, s, sd)) for s in (5, 6, 7) for sd in SPLITS.values())
            if n >= 4:
                keep[run] = (kind, root)
    return keep


def split_dir(root: Path, run: str, seed: int, sd: str) -> Path | None:
    for name in ((run, f"{run}_s5") if seed == 5 else (f"{run}_s{seed}",)):
        d = root / name / sd.format(s=seed)
        if (d / "predictions.json").is_file():
            return d
    return None


@torch.no_grad()
def features(model, data, names, dev, bs=64):
    """Per batch of circuits: the frozen model's log F and the ESP heads' inputs."""
    out = []
    for s in range(0, len(names), bs):
        nm = names[s: s + bs]
        b = next(iter(DataLoader([data[n] for n in nm], batch_size=len(nm))))
        logf = model(b, dev)
        out.append({"names": nm, "logf": logf, **{k: v.detach() for k, v in model.last_esp_feats.items()}})
    return out


def esp_target(lab, names):
    """(B, D, 3): decay, log duration (µs), log depth of the stored compilation; NaN if missing."""
    rows = []
    for n in names:
        r = lab.get(n)
        if r is None:
            rows.append(torch.full((len(DEVS), 3), math.nan))
            continue
        rows.append(torch.stack([r["decay"].float(), torch.log(r["duration"].float() * 1e6),
                                 torch.log(r["depth"].float().clamp(min=1))], -1))
    return torch.stack(rows)


def train_head(model, feats, lab, head, epochs=40, lr=1e-3, seed=0):
    """Only the ESP MLP, on cached features; loss = training.esp_loss (log1p(−decay), log duration, log depth)."""
    torch.manual_seed(seed)
    n_in = feats[0]["emb"].size(-1) + 5 if head == "v13" else model.N_ESPA_IN
    mlp = model.new_esp_mlp(n_in, head)
    targets = [esp_target(lab, f["names"]) for f in feats]
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(feats))
    hist = []
    for ep in range(epochs):
        tot, nb = np.zeros(3), 0
        for i in np.random.default_rng(seed + ep).permutation(len(feats)):
            pred = torch.stack(model.esp_from_features(mlp, feats[i], head), -1)
            true = targets[i]
            known = torch.isfinite(true)
            p = torch.cat([torch.log1p(-pred[..., :1].clamp(max=0)), pred[..., 1:]], -1)
            t = torch.cat([torch.log1p(-true[..., :1].clamp(max=0)), true[..., 1:]], -1)
            err = torch.where(known, (p - torch.nan_to_num(t)) ** 2, torch.zeros_like(p))
            mse = err.sum((0, 1)) / known.sum((0, 1)).clamp(min=1)
            opt.zero_grad(); mse.sum().backward(); opt.step(); sched.step()
            tot += mse.detach().numpy(); nb += 1
        hist.append((tot / nb).round(5).tolist())
    return mlp, hist


@torch.no_grad()
def apply_head(model, mlp, feats, head):
    out = {k: [] for k in ("logf", "decay", "logdur", "logdep")}
    for f in feats:
        d, du, de = model.esp_from_features(mlp, f, head)
        for k, v in zip(out, (f["logf"], d, du, de)):
            out[k].append(v.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run(args):
    torch.set_num_threads(args.threads)
    kind, root = versions()[args.run] if args.kind is None else (args.kind, Path(args.root))
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for split, sd in SPLITS.items():
            d = split_dir(root, args.run, seed, sd)
            dest = OUT / f"{args.run}_s{seed}_{split.replace(' ', '_')}.json"
            if d is None or (dest.is_file() and not args.force):
                continue
            t0 = time.time()
            preds = json.loads((d / "predictions.json").read_text())
            test = [n for n in preds if n in lab and n in data]
            res = {"run": args.run, "kind": kind, "seed": seed, "split": split, "dir": str(d.relative_to(EVAL))}
            # F only, from the stored predictions (every version)
            logf = np.log(np.clip(np.array([preds[n] for n in test], float), 1e-12, 1))
            res["F only"] = metrics({"logf": logf}, test, data, lab)
            if is_sinkhorn_kind(kind) and (d / "model.pth").is_file():
                model = build_model(kind + "_espa", params, dev, torch.device("cpu"))
                missing, unexpected = model.load_state_dict(torch.load(d / "model.pth", map_location="cpu"), strict=False)
                assert all(k.startswith("esp_mlp.") for k in missing) and not unexpected, (missing, unexpected)
                model.eval(); model.esp_all_features = True
                for p_ in model.parameters():
                    p_.requires_grad_(False)
                test_set = set(test)
                train = [n for n in data if n not in test_set and n not in preds and n in lab]
                ft, fe = features(model, data, train, dev), features(model, data, test, dev)
                for head in HEADS:
                    mlp, hist = train_head(model, ft, lab, head, epochs=args.epochs)
                    res[f"probe {head}"] = metrics(apply_head(model, mlp, fe, head), test, data, lab)
                    res[f"probe {head} train mse"] = hist[-1]
            res["seconds"] = round(time.time() - t0, 1)
            dest.write_text(json.dumps(res, indent=1))
            print(args.run, seed, split, {k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()})
                                         for k, v in res.items() if k.startswith(("F only", "probe a", "probe v13"))
                                         and isinstance(v, dict)}, f"{res['seconds']} s", flush=True)


def report(args):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    (HERE / "results/report_esp_versions.json").write_text(json.dumps(rows))
    by = {}
    for r in rows:
        by.setdefault(r["run"], {}).setdefault(r["split"], []).append(r)
    L = ["# ESP, duration and depth across fidelity-model versions", "",
         "Mean over seeds (n). F only: ESP ≈ predicted F. probe v13 / a: frozen version + trained ESP head "
         "(v8kde10's head / analytic head). Regret = mean ESP lost by the chosen device.", ""]
    for metric, src in (("ESP regret", None), ("R² log depth", None), ("depth error (median)", None),
                        ("R² log duration", None), ("R² ESP", None)):
        L += [f"## {metric}", "", "| run | " + " | ".join(f"{s} {h}" for s in SPLITS for h in ("F", "v13", "a")) + " |",
              "| --- |" + " --- |" * (3 * len(SPLITS))]
        for run in by:
            cells = []
            for s in SPLITS:
                rs = by[run].get(s, [])
                for h, key in (("F", "F only"), ("v13", "probe v13"), ("a", "probe a")):
                    k = metric
                    if h == "F":
                        k = {"ESP regret": "ESP regret, baseline F", "R² ESP": "R² ESP, baseline F"}.get(metric)
                    v = [r[key][k] for r in rs if key in r and k and k in r[key]]
                    cells.append(f"{np.mean(v):.3f} ({len(v)})" if v else "")
            L.append(f"| {run} | " + " | ".join(cells) + " |")
        L.append("")
    (HERE / "results/report_esp_versions.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "list"))
    ap.add_argument("--run")
    ap.add_argument("--kind", default=None, help="override (with --root) for runs outside run_shift's MODELS")
    ap.add_argument("--root", default=None)
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "list":
        for name, (kind, root) in versions().items():
            print(name, kind, root.relative_to(EVAL))
    else:
        {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
