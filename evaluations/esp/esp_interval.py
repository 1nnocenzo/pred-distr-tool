#!/usr/bin/env python3
"""Duration with a guaranteed bound, for a scheduler: split-conformal intervals on the predicted circuit duration.

Per run and seed, the ratio true / predicted duration of the control split's test circuits (families seen in
training) gives, separately for routing-free and routed circuits, the factors that make
  upper bound      pred × u          with P(true ≤ bound) = 90% / 95%   (one-sided: a scheduler must not overrun)
  interval         [pred × a, pred × b]  holding 80% of the cases       (two-sided)
The control circuits are split in two halves (calibrate on one, check on the other: in-distribution coverage); the
same factors are then applied to the LOFO qaoa / LOFO qnn / LOGO variational models of the same seed: does the
coverage hold on families never seen?  Reported: coverage, median width (bound / pred), and the unbiased point
estimate pred × median ratio.
    GS_RESULTS_ROOT=generalization_v10 python esp_interval.py --runs v8kdfi10,v8kdft10   # -> results/report_esp_interval.md
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esp_eval import DEVS, LABELS, SPLITS  # noqa: E402
from esp_probe import split_dir  # noqa: E402
from esp_budget import KINDS, ROOT  # noqa: E402
from gsv3 import paths  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

LEVELS = (0.90, 0.95)


@torch.no_grad()
def collect(model, data, dev, names):
    dur, rf = [], []
    for s in range(0, len(names), 64):
        b = next(iter(DataLoader([data[n] for n in names[s: s + 64]], batch_size=len(names[s: s + 64]))))
        model(b, dev)
        dur.append(torch.exp(model.last_esp[1]).numpy())
        rf.append(routing_free(b, dev, model._rf_cache).numpy() > 0.5)
    return np.concatenate(dur), np.concatenate(rf)


def factors(r):
    """r = log(true / pred) of the calibration cases -> conformal factors (finite-sample quantiles)."""
    n = len(r)
    qk = lambda a: float(np.quantile(r, min(1.0, np.ceil((n + 1) * a) / n), method="higher"))
    return {**{f"u{int(a * 100)}": qk(a) for a in LEVELS}, "lo80": qk(0.10), "hi80": qk(0.90), "med": float(np.median(r))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="v8kdfi10,v8kdft10")
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    res = {}
    for run in args.runs.split(","):
        acc = defaultdict(lambda: defaultdict(list))      # (split, group) -> metric -> values per case
        for seed in [int(s) for s in args.seeds.split(",")]:
            pr = {}
            for split, sd in SPLITS.items():
                d = split_dir(ROOT, run, seed, sd)
                if d is None:
                    continue
                model = build_model(KINDS[run], params, dev, torch.device("cpu"))
                model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
                names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data]
                p, rf = collect(model, data, dev, names)
                t = np.stack([lab[n]["duration"].numpy() for n in names]) * 1e6
                pr[split] = (np.log(t / p).ravel(), rf.ravel(), np.array([zlib.crc32(n.encode()) % 2 for n in names]).repeat(len(DEVS)))
                print(run, seed, split, len(names), flush=True)
            if "control" not in pr:
                continue
            r_c, rf_c, half = pr["control"]
            for grp, gsel in (("routing-free", True), ("routed", False)):
                fac_all = factors(r_c[rf_c == gsel])
                for split, (r, rf, h) in pr.items():
                    if split == "control":           # cross-fit: calibrate on one half, check on the other
                        for k in (0, 1):
                            cal, te = (rf == gsel) & (h == k), (rf == gsel) & (h != k)
                            if cal.sum() < 20 or te.sum() < 20:
                                continue
                            f = factors(r[cal])
                            for m, v in metrics(r[te], f).items():
                                acc[(split, grp)][m] += v
                        continue
                    sel = rf == gsel
                    if sel.sum() < 20:
                        continue
                    for m, v in metrics(r[sel], fac_all).items():
                        acc[(split, grp)][m] += v
        agg = lambda m, v: float(np.median(v) if m.startswith("err") else np.mean(v))
        res[run] = {f"{s}|{g}": {m: agg(m, v) for m, v in ms.items()} | {"n": len(ms["cov90"])}
                    for (s, g), ms in acc.items()}
        for k, v in res[run].items():
            print(run, k, json.dumps({m: round(x, 3) for m, x in v.items()}), flush=True)
    (HERE / f"results/report_esp_interval{args.tag}.json").write_text(json.dumps(res))
    L = ["# Duration bounds for a scheduler: conformal factors from the control split", "",
         "Factors calibrated on the control split's test circuits (families seen in training; cross-fitted halves for the "
         "control row), separately for routing-free and routed circuits, then applied to the models of the other splits "
         "(same seed). Seeds pooled. Coverage = share of cases with the true duration inside; width = bound / predicted "
         "duration (median); error after bias fix = median |pred × median ratio / true − 1|.", ""]
    for run, rs in res.items():
        L += [f"## {run}", "", "| split | group | cases | coverage, 90% upper bound | width (×pred) | coverage, 95% upper bound | width | "
              "coverage, 80% interval | interval [lo, hi] (×pred) | error, raw | error after bias fix |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key in sorted(rs, key=lambda k: (list(SPLITS).index(k.split("|")[0]), k.split("|")[1])):
            v = rs[key]; s, g = key.split("|")
            L.append(f"| {s} | {g} | {v['n']} | {v['cov90']:.3f} | {v['w90']:.2f} | {v['cov95']:.3f} | {v['w95']:.2f} | "
                     f"{v['cov80']:.3f} | [{v['lo80']:.2f}, {v['hi80']:.2f}] | {v['err']:.3f} | {v['err_fix']:.3f} |")
        L.append("")
    (HERE / f"results/report_esp_interval{args.tag}.md").write_text("\n".join(L))
    print("\n".join(L))


def metrics(r, f) -> dict:
    """Per-case values (lists, pooled over seeds by the caller) for r = log(true / pred) under factors f."""
    n = len(r)
    one = lambda x: [float(x)] * n
    return {"cov90": list((r <= f["u90"]).astype(float)), "cov95": list((r <= f["u95"]).astype(float)),
            "cov80": list(((r >= f["lo80"]) & (r <= f["hi80"])).astype(float)),
            "w90": one(np.exp(f["u90"])), "w95": one(np.exp(f["u95"])),
            "lo80": one(np.exp(f["lo80"])), "hi80": one(np.exp(f["hi80"])),
            "err": list(np.abs(np.exp(-r) - 1)), "err_fix": list(np.abs(np.exp(f["med"] - r) - 1))}


if __name__ == "__main__":
    main()
