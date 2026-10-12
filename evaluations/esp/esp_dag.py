#!/usr/bin/env python3
"""Physical duration / idle time / depth from the circuit's own gate DAG (no learned ratios).

The circuit graph has an edge between consecutive gates on the same qubit (program order).  With every gate
weighted by its predicted compiled cost (the count head's r / CZ per gate and device; measurements exact from the
circuit), an ASAP schedule of the logical DAG gives:
  duration   longest path in time       (r 42 ns, CZ 130 ns, measurement 15 µs)
  depth      longest path in operations  (r and CZ split over the gate's operands)
  idle       Σ_active qubits (duration − busy_q),  busy_q = Σ time of the gates on q
  decay      −idle · mean 1/min(T1, T2) of the placed qubits  ->  ESP = F̂ · exp(decay)
compared with the truth (esp_labels.pt) and with the model's own ESP head, per split.  No training.
    python esp_dag.py --run v8kda10 --seed 5 [--max-circuits 1500]   # -> results/report_esp_dag_<run>.md
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
from esp_eval import DEVS, LABELS, SPLITS, r2  # noqa: E402
from esp_probe import split_dir  # noqa: E402
from esp_budget import KINDS, ROOT, choice  # noqa: E402
from gsv3 import paths  # noqa: E402
from gsv3.devices import GATE_US, variant_batch  # noqa: E402
from gsv3.model import GATE2IDX_MEASURE, build_model, load_hparams  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

T1Q, TM, TCZ = GATE_US


def schedule(g, n_ops):
    """ASAP on the logical DAG.  g: one circuit graph; n_ops: (G, D, 3) r / meas / cz per gate.
    Returns per device: duration µs, depth ops, idle µs, busy per qubit."""
    G, D = n_ops.shape[0], n_ops.shape[1]
    meas = g.x[:, GATE2IDX_MEASURE].numpy() > 0.5
    gq = g.gate_qubits.numpy()
    n_operands = (gq >= 0).sum(1).clip(min=1)
    r, cz = n_ops[..., 0], n_ops[..., 2]
    t_gate = np.where(meas[:, None], TM, r * T1Q / n_operands[:, None] + cz * TCZ)        # time on each operand
    o_gate = np.where(meas[:, None], 1.0, r / n_operands[:, None] + cz)                   # ops on each operand
    src, dst = g.edge_index.numpy()
    preds = defaultdict(list)
    for a, b in zip(src, dst):
        if a < b:
            preds[b].append(a)
    fin_t, fin_o = np.zeros((G, D)), np.zeros((G, D))
    for i in range(G):
        p = preds.get(i)
        st_t = fin_t[p].max(0) if p else 0.0
        st_o = fin_o[p].max(0) if p else 0.0
        fin_t[i] = st_t + t_gate[i]
        fin_o[i] = st_o + o_gate[i]
    dur, dep = fin_t.max(0), fin_o.max(0)
    nq = int(g.n_qubits)
    busy = np.zeros((nq, D))
    for k in range(gq.shape[1]):
        m = gq[:, k] >= 0
        np.add.at(busy, gq[m, k], t_gate[m])
    active = (busy > 0)
    idle = ((dur[None, :] - busy).clip(min=0) * active).sum(0)
    return dur, dep, idle, busy, active


@torch.no_grad()
def run(model, data, lab, dev, names):
    out = defaultdict(list)
    for s in range(0, len(names), 64):
        nm = names[s: s + 64]
        b = next(iter(DataLoader([data[n] for n in nm], batch_size=len(nm))))
        logf = model(b, dev)
        dec_m, ldur_m, ldep_m = model.last_esp
        inv_t, act_m = model.last_esp_feats["inv_t"], model.last_esp_feats["active"]
        inv_m = ((act_m * inv_t).sum(-1) / act_m.sum(-1).clamp(min=1)).numpy()          # (B, D)
        n_ops = model.last_n_ops.numpy()
        gb = b.batch.numpy()
        for j, n in enumerate(nm):
            dur, dep, idle, _, _ = schedule(data[n], n_ops[gb == j])
            for k, v in (("dur", dur), ("dep", dep), ("idle", idle), ("inv", inv_m[j])):
                out[k].append(v)
        for k, v in (("logf", logf), ("dec_m", dec_m), ("ldur_m", ldur_m), ("ldep_m", ldep_m)):
            out[k].append(v.numpy())
    p = {k: (np.stack(v) if k in ("dur", "dep", "idle", "inv") else np.concatenate(v)) for k, v in out.items()}
    T = {k: np.stack([lab[n][k].numpy() for n in names]) for k in ("F", "ESP", "duration", "depth", "decay")}
    return p, T


def metrics(p, T):
    ld, lp = np.log(T["duration"] * 1e6), np.log(np.maximum(T["depth"], 1))
    err = lambda a, b: float(np.median(np.abs(np.exp(a - b) - 1)))
    ok = np.isfinite(T["decay"]).all(1)
    fhat = np.exp(np.minimum(p["logf"], 0))
    dec_dag = -p["idle"] * p["inv"]
    m = {"duration error, DAG": err(np.log(p["dur"] + 1e-3), ld), "duration error, model head": err(p["ldur_m"], ld),
         "R² log duration, DAG": r2(np.log(p["dur"] + 1e-3).ravel(), ld.ravel()), "R² log duration, model head": r2(p["ldur_m"].ravel(), ld.ravel()),
         "depth error, DAG": err(np.log(np.maximum(p["dep"], 1)), lp), "depth error, model head": err(p["ldep_m"], lp),
         "R² log depth, DAG": r2(np.log(np.maximum(p["dep"], 1)).ravel(), lp.ravel()), "R² log depth, model head": r2(p["ldep_m"].ravel(), lp.ravel()),
         "R² decay, DAG": r2(dec_dag[ok].ravel(), T["decay"][ok].ravel()), "R² decay, model head": r2(p["dec_m"][ok].ravel(), T["decay"][ok].ravel())}
    esp = T["ESP"][ok]
    for lab_, d in (("DAG", dec_dag), ("model head", p["dec_m"]), ("true decay", T["decay"])):
        rd, rg = choice((fhat * np.exp(d))[ok], esp)
        m[f"ESP regret, {lab_}"], m[f"right device, {lab_}"] = rg, rd
    rd, rg = choice(fhat[ok], esp)
    m["ESP regret, F only"], m["right device, F only"] = rg, rd
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v8kda10")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--max-circuits", type=int, default=1500)
    args = ap.parse_args()
    torch.set_num_threads(8)
    kind = KINDS.get(args.run, "sinkhorn_rf_route_czc_espa")
    params = load_hparams()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(LABELS, weights_only=False)
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    res = {}
    rng = np.random.default_rng(0)
    for split, sd in SPLITS.items():
        d = split_dir(ROOT, args.run, args.seed, sd)
        if d is None:
            continue
        model = build_model(kind, params, dev, torch.device("cpu"))
        model.load_state_dict(torch.load(d / "model.pth", map_location="cpu")); model.eval()
        names = [n for n in json.loads((d / "predictions.json").read_text()) if n in lab and n in data]
        if len(names) > args.max_circuits:
            names = [names[i] for i in sorted(rng.choice(len(names), args.max_circuits, replace=False))]
        res[split] = metrics(*run(model, data, lab, dev, names))
        res[split]["n"] = len(names)
        print(split, json.dumps({k: round(v, 4) for k, v in res[split].items()}), flush=True)
    keys = list(next(iter(res.values())))
    L = [f"# Physical schedule on the circuit DAG vs the ESP head: {args.run} (seed {args.seed})", "",
         "Per-gate cost from the model's count head (measurements exact), ASAP on the logical gate DAG; no training.", "",
         "| | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{res[s][k]:.4f}" if k != "n" else str(res[s][k]) for s in res) + " |")
    (HERE / f"results/report_esp_dag_{args.run}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
