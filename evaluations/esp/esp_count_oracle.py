#!/usr/bin/env python3
"""How good would the predicted circuit duration be with the right gate counts?  (ceiling of the count route)

The formula head (v19 / v19i) computes the duration as an ASAP schedule of the logical gate DAG, every gate weighted by
its predicted compiled r / CZ count (``dag_schedule``), times a learned correction.  Here the same schedule is run with
per-gate counts from different sources, against the true duration of the stored compilation (esp_labels.pt):
  model       the model's own duration (schedule × learned correction)
  pred        schedule with the predicted per-gate counts (no correction)
  pred→tot    predicted per-gate counts rescaled so each circuit's r and CZ totals are the compiler's (compiler_counts_v8)
  pred→cz     only the CZ total made exact;   pred→r  only the r total made exact
  table       fixed per-gate decomposition (results/decomp_table.json: each gate type alone, level 2), no model
  table→tot   the table's per-gate counts rescaled to the compiler's totals
"→tot" rows are oracles: the gap that is left with exact totals is the limit of scheduling the LOGICAL DAG (where in
the circuit the counts fall, SWAPs that serialise qubits the logical circuit does not link).  Measurements exact.
    GS_RESULTS_ROOT=generalization_v10 python esp_count_oracle.py --runs v8kdi10,v8kdfi10   # -> results/report_esp_count_oracle.md
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
from gsv3.model import GATE2IDX_MEASURE, build_model, dag_schedule, load_hparams, routing_free  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.utils import scatter  # noqa: E402

COUNTS = EVAL / "compile_check/results/compiler_counts_v8.pt"
SOURCES = ("model", "pred", "pred→tot", "pred→cz", "pred→r", "table", "table→tot")


def to_totals(n, gb, n_circ, true, col, eligible):
    """Rescale column ``col`` of the per-gate counts n (G, D, 3) so each circuit's total is ``true`` (B, D); a circuit
    whose predicted total is ~0 gets the true total spread evenly over its eligible gates."""
    n = n.clone()
    x = n[..., col] * eligible.view(-1, 1)
    tot = scatter(x, gb, dim=0, dim_size=n_circ, reduce="sum")
    k = eligible.float()
    n_el = scatter(k, gb, dim=0, dim_size=n_circ, reduce="sum").clamp(min=1).view(-1, 1)
    scaled = x * (true / tot.clamp(min=1e-9))[gb]
    even = (true / n_el)[gb] * k.view(-1, 1)
    n[..., col] = torch.where((tot > 1e-6)[gb], scaled, even)
    return n


@torch.no_grad()
def collect(model, data, dev, names, cnt, tr, tc, n_u):
    out = defaultdict(list)
    cache = {}
    for s in range(0, len(names), 64):
        nm = names[s: s + 64]
        b = next(iter(DataLoader([data[n] for n in nm], batch_size=len(nm))))
        model(b, dev)
        n_circ = len(nm)
        gb = b.batch
        meas = (b.x[:, GATE2IDX_MEASURE] > 0.5)
        n_dev = len(DEVS)
        m_col = meas.float().view(-1, 1).expand(-1, n_dev)
        n_pred = model.last_n_ops.clone()
        n_pred[..., 1] = m_col                                          # measurements exact (as the ESP heads)
        oh = b.x[:, :n_u].float()
        n_tab = torch.stack([(oh @ tr).view(-1, 1).expand(-1, n_dev), m_col, (oh @ tc).view(-1, 1).expand(-1, n_dev)], -1)
        n_tab = n_tab * (~meas).float().view(-1, 1, 1) + torch.stack([torch.zeros_like(m_col), m_col, torch.zeros_like(m_col)], -1) * meas.float().view(-1, 1, 1)
        C = torch.stack([cnt[n].float() for n in nm])                   # (B, D, 3) r, measure, cz
        n_op = (b.gate_qubits >= 0).sum(1)
        el_r, el_cz = ~meas, (~meas) & (n_op >= 2)
        variants = {
            "pred": n_pred,
            "pred→tot": to_totals(to_totals(n_pred, gb, n_circ, C[..., 0], 0, el_r), gb, n_circ, C[..., 2], 2, el_cz),
            "pred→cz": to_totals(n_pred, gb, n_circ, C[..., 2], 2, el_cz),
            "pred→r": to_totals(n_pred, gb, n_circ, C[..., 0], 0, el_r),
            "table": n_tab,
            "table→tot": to_totals(to_totals(n_tab, gb, n_circ, C[..., 0], 0, el_r), gb, n_circ, C[..., 2], 2, el_cz),
        }
        out["model"].append(torch.exp(model.last_esp[1]).numpy())
        for k, v in variants.items():
            out[k].append(dag_schedule(b, v, cache)["dur_s"].numpy())
        out["rf"].append(routing_free(b, dev, model._rf_cache).numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def stats(pr, t):
    rel = np.abs(pr / t - 1)
    lp, lt = np.log(pr + 1e-9), np.log(t)
    return {"err": float(np.median(rel)), "p90": float(np.percentile(rel, 90)), "bias": float(np.median(pr / t)),
            "w10": float((rel < 0.10).mean()), "r2": float(1 - ((lp - lt) ** 2).sum() / ((lt - lt.mean()) ** 2).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="v8kdi10,v8kdfi10")
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N circuits per split")
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
    tr = torch.tensor([tab.get(u, {"r": 0})["r"] for u in U], dtype=torch.float)
    tc = torch.tensor([tab.get(u, {"cz": 0})["cz"] for u in U], dtype=torch.float)
    res = {}
    for run in args.runs.split(","):
        kind = KINDS[run]
        per = defaultdict(lambda: defaultdict(list))
        for seed in [int(s) for s in args.seeds.split(",")]:
            for split, sd in SPLITS.items():
                d = split_dir(ROOT, run, seed, sd)
                if d is None:
                    continue
                model = build_model(kind, params, dev, torch.device("cpu"))
                model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
                names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data and n in cnt]
                if args.limit:
                    names = names[: args.limit]
                p = collect(model, data, dev, names, cnt, tr, tc, len(U))
                per[split]["true"].append(np.stack([lab[n]["duration"].numpy() for n in names]) * 1e6)
                for k in (*SOURCES, "rf"):
                    per[split][k].append(p[k])
                print(run, seed, split, len(names), flush=True)
        res[run] = {}
        for split, v in per.items():
            t = np.concatenate(v["true"]).ravel()
            rf = np.concatenate(v["rf"]).ravel() > 0.5
            res[run][split] = {}
            for grp, sel in (("all", np.ones_like(rf)), ("routing-free", rf), ("routed", ~rf)):
                if sel.sum() < 20:
                    continue
                res[run][split][grp] = {"n": int(sel.sum()),
                                        **{k: stats(np.concatenate(v[k]).ravel()[sel], t[sel]) for k in SOURCES}}
            print(run, split, json.dumps({k: round(x["err"], 4) for k, x in res[run][split]["all"].items() if k != "n"}), flush=True)
    (HERE / f"results/report_esp_count_oracle{args.tag}.json").write_text(json.dumps(res))
    L = ["# Duration with the right gate counts: ceiling of the count route", "",
         "ASAP schedule of the logical gate DAG with per-gate counts from each source (see esp_count_oracle.py); "
         "true duration of the stored compilation. Seeds pooled; relative error |pred/true − 1|. "
         "\"→tot\" = per-gate counts rescaled to the compiler's true r / CZ totals of the circuit (oracle).", ""]
    for run, rs in res.items():
        for grp in ("all", "routing-free", "routed"):
            L += [f"## {run}: {grp}", "", "| source | " + " | ".join(f"{s} (n {rs[s][grp]['n']})" for s in rs if grp in rs[s]) + " |",
                  "| --- |" + " --- |" * sum(grp in rs[s] for s in rs)]
            for k in SOURCES:
                L.append(f"| {k} | " + " | ".join(
                    f"{rs[s][grp][k]['err']:.3f} (p90 {rs[s][grp][k]['p90']:.2f}, bias {rs[s][grp][k]['bias']:.2f}, R² {rs[s][grp][k]['r2']:.2f})"
                    for s in rs if grp in rs[s]) + " |")
            L.append("")
    (HERE / f"results/report_esp_count_oracle{args.tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
