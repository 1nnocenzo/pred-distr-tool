#!/usr/bin/env python3
"""Why does the placement not follow the qubits?  A mechanistic look at the Sinkhorn placement.

For control-split test circuits and local v8d calibration variants (jitter, shuffle, bad qubits),
compare how the model's placement and the compiler's layout react to the calibration change:

 1. reactivity   share of logical qubits whose argmax physical qubit changes (model vs compiler);
                 change of occupancy u_p = Σ_q A[q, p] per physical qubit, model vs compiler
                 (correlation over the qubits of each case, and its magnitude);
 2. direction    corr(Δu_p, Δε_p) where ε_p = r + readout + mean CZ error of qubit p: negative =
                 moves away from qubits that got worse; for the model and for the compiler;
 3. inputs       how much the device embeddings h_p and the placement scores depend on the errors:
                 same device with every error replaced by the device mean (errors flattened) vs
                 real errors: 1 − cos(h, h_flat) (0 = the errors do not enter the embeddings);
 4. sharpness    row max of A and its entropy; local tracking when the Sinkhorn temperature is
                 lowered at inference (no retraining).
    python placement_probe.py [--n-circuits 300] [--n-variants 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

ROOT = EVAL / "generalization_v9/results"
V8D = EVAL / "compile_check/results/v8d"
ORIG = EVAL / "compile_check/results/v8b_orig"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LOCAL_KINDS = {0, 1, 2, 3, 4, 5, 6, 9}
MODELS = {"v8k (no fine-tuning)": ("sinkhorn_rf_route_czc", "v8k"), "v8kd40": ("sinkhorn_rf_route_czc", "v8kd40"),
          "v11d40": ("sinkhorn_rf_route_usage", "v11d40"), "v8kdl40": ("sinkhorn_rf_route_czc", "v8kdl40"),
          "v8kdr40": ("sinkhorn_rf_route_czc", "v8kdr40"), "v8kdp40": ("sinkhorn_rf_route_czc_ep", "v8kdp40"),
          "v8kdrp40": ("sinkhorn_rf_route_czc_ep", "v8kdrp40"), "v8kdc10": ("sinkhorn_rf_route_czc", "v8kdc10")}
TAUS = (0.3, 0.15, 0.07, 0.03)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def occupancy_from_layout(lay, n_dev, p_max):
    """lay (n, D) node indices -> (D, P) 0/1 occupancy."""
    u = torch.zeros(n_dev, p_max)
    for d in range(n_dev):
        for p in lay[:, d].tolist():
            if p >= 0:
                u[d, p] = 1.0
    return u


def flatten_errors(raw_dev: dict) -> dict:
    """Same device with every qubit's r / readout error and every coupler's CZ error set to the mean."""
    nodes = np.array(raw_dev["nodes"], dtype=float)
    nodes[:, 0] = nodes[:, 0].mean(); nodes[:, 1] = nodes[:, 1].mean()
    e = np.array(raw_dev["edge_feats"], dtype=float)
    return {"nodes": nodes.tolist(), "edges": raw_dev["edges"], "edge_feats": [float(e.mean())] * len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=300)
    ap.add_argument("--n-variants", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    rng = np.random.default_rng(0)
    params = load_hparams()
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(V8D / "v8d_labels.pt", weights_only=False)
    orig = torch.load(ORIG / "v8b_orig_labels.pt", weights_only=False)
    test = sorted(json.loads((ROOT / "v8kd40/random_control/random_seed5/predictions.json").read_text()))
    pool = [data[n] for n in test if n in data and n in lab and n in orig and lab[n]["layout"].shape[0] == int(data[n].n_qubits)]
    pool = [pool[i] for i in rng.choice(len(pool), min(args.n_circuits, len(pool)), replace=False)]
    js = sorted(rng.choice([j for j in range(8, 100) if j % 10 in LOCAL_KINDS], args.n_variants, replace=False).tolist())
    raw = json.loads((V8D / "variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    for d in DEVS:
        raw[f"{d}/flat"] = flatten_errors(raw[d])
    dev_o = variant_batch(raw, list(DEVS), params["lap_pe"])
    dev_f = variant_batch(raw, [f"{d}/flat" for d in DEVS], params["lap_pe"])
    dev_v = {j: variant_batch(raw, [f"{d}/aug{j}" for d in DEVS], params["lap_pe"]) for j in js}
    eps = lambda dv: to_dense_batch(dv.phys.sum(-1), dv.batch)[0]          # (D, P) r + readout + mean CZ
    eps_o = eps(dev_o)
    L = ["# Why does the placement not follow the qubits?", "",
         f"{len(pool)} control test circuits × {len(js)} local v8d calibrations (aug {js}) × 3 devices.", ""]
    rows1, rows3, rows4 = [], [], []
    comp_stats = None
    for name, (kind, run) in MODELS.items():
        model = build_model(kind, params, dev_o, torch.device("cpu"))
        model.load_state_dict(torch.load(ROOT / run / "random_control/random_seed5/model.pth", map_location="cpu"))
        model.eval()
        # 3. inputs: embeddings with real vs flattened errors
        with torch.no_grad():
            h_o, mask, _ = model._devices(dev_o)
            h_f, _, _ = model._devices(dev_f)
        cos = torch.nn.functional.cosine_similarity(h_o, h_f, dim=-1)[mask]
        emb_dep = float((1 - cos).mean())
        moves_m, moves_c, nq = 0, 0, 0
        cu, cdir_m, cdir_c, mag_m, mag_c, rowmax, ent = [], [], [], [], [], [], []
        trk = {t: {} for t in TAUS}; dt = {}
        for start in range(0, len(pool), 32):
            gs = pool[start: start + 32]
            b = next(iter(DataLoader(gs, batch_size=len(gs))))
            y_o = torch.stack([orig[g.circuit_name]["F"][0] for g in gs]).float()
            with torch.no_grad():
                base = {}
                for t in TAUS:
                    model.tau = t
                    base[t] = model(b, dev_o)
                    if t == TAUS[0]:
                        A_o, qm = model.last_assign.clone(), model.last_q_mask.clone()
                model.tau = TAUS[0]
            p_max = A_o.size(-1)
            lay_o = [orig[g.circuit_name]["layout"][:, 0] for g in gs]
            for j in js:
                y_v = torch.stack([lab[g.circuit_name]["F"][j] for g in gs]).float()
                keep = (y_v > 0.01) & (y_o > 0.01)
                d_true = torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))
                for d in range(3):
                    dt.setdefault((j, d), []).append(d_true[:, d][keep[:, d]].numpy())
                eps_v = eps(dev_v[j])
                with torch.no_grad():
                    for t in TAUS:
                        model.tau = t
                        lv = model(b, dev_v[j])
                        if t == TAUS[0]:
                            A_v = model.last_assign.clone()
                        dp = torch.clamp(lv, max=0) - torch.clamp(base[t], max=0)
                        for d in range(3):
                            trk[t].setdefault((j, d), []).append(dp[:, d][keep[:, d]].numpy())
                    model.tau = TAUS[0]
                for i, g in enumerate(gs):
                    n = int(g.n_qubits)
                    lv_ = lab[g.circuit_name]["layout"][:, j]                      # (n, D)
                    uc_o = occupancy_from_layout(lay_o[i], 3, p_max); uc_v = occupancy_from_layout(lv_, 3, p_max)
                    for d in range(3):
                        P = int(dev_o.batch.eq(d).sum())
                        um_o = A_o[i, d, :n, :P].sum(0); um_v = A_v[i, d, :n, :P].sum(0)
                        dum, duc = (um_v - um_o).numpy(), (uc_v[d, :P] - uc_o[d, :P]).numpy()
                        de = (eps_v[d, :P] - eps_o[d, :P]).numpy()
                        am_o = A_o[i, d, :n, :P].argmax(-1); am_v = A_v[i, d, :n, :P].argmax(-1)
                        ok = (lay_o[i][:, d] >= 0) & (lv_[:, d] >= 0)
                        moves_m += int((am_o != am_v)[ok].sum()); moves_c += int((lay_o[i][:, d] != lv_[:, d])[ok].sum()); nq += int(ok.sum())
                        if np.abs(duc).sum() > 0:
                            cu.append(corr(dum, duc))
                        if de.std() > 0:
                            cdir_m.append(corr(dum, de)); cdir_c.append(corr(duc, de))
                        mag_m.append(np.abs(dum).sum() / max(n, 1)); mag_c.append(np.abs(duc).sum() / max(n, 1))
                        rows_ = A_v[i, d, :n, :P]
                        rowmax.append(float(rows_.max(-1).values.mean()))
                        ent.append(float((-(rows_.clamp(min=1e-12)).log() * rows_).sum(-1).mean() / np.log(P)))
        dtc = {k: np.concatenate(v) for k, v in dt.items()}
        trk_t = {t: np.nanmean([corr(np.concatenate(trk[t][k]), dtc[k]) for k in dtc if len(dtc[k]) > 20]) for t in TAUS}
        rows1.append(f"| {name} | {moves_m / max(nq, 1):.0%} | {moves_c / max(nq, 1):.0%} | {np.nanmean(mag_m):.2f} | {np.nanmean(mag_c):.2f} | {np.nanmean(cu):+.2f} | {np.nanmean(cdir_m):+.2f} | {np.nanmean(cdir_c):+.2f} |")
        rows3.append(f"| {name} | {emb_dep:.3f} | {np.mean(rowmax):.2f} | {np.mean(ent):.2f} |")
        rows4.append(f"| {name} | " + " | ".join(f"{trk_t[t]:.2f}" for t in TAUS) + " |")
        log(rows1[-1]); log(rows3[-1]); log(rows4[-1])
    L += ["## 1–2. Does the placement react, and in the right direction?", "",
          "| model | qubits moved, model | compiler | occupancy change per logical qubit, model | compiler | corr(Δu model, Δu compiler) | corr(Δu, Δε), model | compiler |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"] + rows1 + [
          "", "Δu = change of occupancy of each physical qubit (model: Σ_q A; compiler: 0/1 initial layout); Δε = change of r + readout + mean CZ error of the qubit. Negative corr(Δu, Δε) = leaves qubits that got worse.", "",
          "## 3–4. What drives the placement, and how sharp is it?", "",
          "| model | embedding change when errors are flattened (1 − cos) | mean row max of A | row entropy / log P |", "| --- | --- | --- | --- |"] + rows3 + [
          "", "## 4. Local tracking with a lower Sinkhorn temperature at inference (no retraining)", "",
          "| model | " + " | ".join(f"τ = {t}" for t in TAUS) + " |", "| --- | " + " | ".join("---" for _ in TAUS) + " |"] + rows4
    out = HERE / "results" / "report_placement_probe.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
