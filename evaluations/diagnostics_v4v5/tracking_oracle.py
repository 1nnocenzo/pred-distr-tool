#!/usr/bin/env python3
"""Why is the local calibration tracking low?  Oracle substitutions on the v8d calibrations.

The physics head computes log F = -Σ_g Σ_k n_k(g, d) · ε̄_k(g, d) · s_k from two learned parts:
the placement A (which physical qubits each logical qubit uses) and the counts n_k (native
r / measure / cz per gate).  For every (circuit, device, local calibration variant) of v8d
we know the compiler's truth: the initial layout and the native counts of the best of 3
compilations, on the original devices (v8b_orig) and on the variant.  Tracking = corr of
predicted vs true Δlog F = log F_variant − log F_original (F > 0.01), as in training.

  model          the model as is
  + layout       A replaced by the compiler's layout (one-hot) on each calibration
  + counts       per-circuit totals of n_r, n_measure, n_cz rescaled to the compiled ones
  + both         layout and counts from the compiler → what remains is the head itself
  fixed          both oracles, but layout and counts from the ORIGINAL compilation also on
                 the variant: the error change alone, no recompilation

Rows: test vs training circuits of the control split, seen vs unseen calibrations (v8kd8 saw
aug0-7 only; v8kd saw aug0-99), plus the label noise (agreement between compiler seeds).
    python tracking_oracle.py [--n-circuits 1000] [--n-variants 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, to_dense_batch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import N_GATE_LOCAL, build_model, load_hparams, routing_estimate, routing_free  # noqa: E402

V8D = EVAL / "compile_check/results/v8d"
ORIG = EVAL / "compile_check/results/v8b_orig"
ROOT = EVAL / "generalization_v9/results"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LOCAL_KINDS = {0, 1, 2, 3, 4, 5, 6, 9}          # v8d j % 10: jitter, shuffle, bad qubits (not global scale)
MODELS = {"v8k": "v8k", "v8kd8 (8 calib.)": "v8kd8", "v8kd (100 calib.)": "v8kd"}
KIND = "sinkhorn_rf_route_czc"


def coupler_matrix(raw_dev, p_max):
    """(P, P) mean CZ error (−log(1−e)) per edge along the lowest-error path between two
    physical qubits: the error of the coupler itself when they are adjacent."""
    n = len(raw_dev["nodes"])
    INF = 1e9
    w = np.full((n, n), INF); h = np.zeros((n, n))
    np.fill_diagonal(w, 0.0)
    for (a, b), e in zip(raw_dev["edges"], raw_dev["edge_feats"]):
        w[a, b] = w[b, a] = e; h[a, b] = h[b, a] = 1
    for k in range(n):
        alt = w[:, k:k + 1] + w[k:k + 1, :]
        better = alt < w
        w = np.where(better, alt, w); h = np.where(better, h[:, k:k + 1] + h[k:k + 1, :], h)
    out = np.zeros((p_max, p_max))
    out[:n, :n] = np.where(h > 0, w / np.maximum(h, 1), 0.0)
    return torch.tensor(out, dtype=torch.float32)


def oracle_forward(m, circuits, devices, layout=None, counts=None, coupler=None):
    """SinkhornPlacementPredictor.forward with optional oracles.  ``layout`` (B, D, Qm) node
    index per logical qubit (−1 = keep the model's row); ``counts`` (B, D, 3) true totals.
    Returns (log F (B, D), argmax placement (B, D, Qm))."""
    g = m.gate_proj(m.circuit_encoder(circuits))
    q, gq_glob, valid = m._qubits(circuits, g)
    phys, mask, dev_glob = m._devices(devices)
    n_dev, p_max = phys.size(0), phys.size(1)
    n_q = circuits.n_qubits.view(-1).long()
    n_circ = n_q.size(0)
    q_batch = torch.repeat_interleave(torch.arange(n_circ), n_q)
    q_dense, q_mask = to_dense_batch(q, q_batch)
    scores = torch.einsum("bqk,dpk->bdqp", m.w_q(q_dense), m.w_p(phys)) / phys.size(-1) ** 0.5
    assign = m._sinkhorn(scores, n_q, mask.sum(1))
    if layout is not None:
        lay = layout[:, :, : assign.size(2)]
        ok = (lay >= 0).unsqueeze(-1)
        onehot = F.one_hot(lay.clamp(min=0), p_max).float()
        assign = torch.where(ok, onehot, assign) * q_mask.view(n_circ, 1, -1, 1)
    ctx = torch.einsum("bdqp,dpk->bdqk", assign, phys)
    eps_dense, _ = to_dense_batch(devices.phys, devices.batch)
    q_eps = torch.einsum("bdqp,dpk->bdqk", assign, eps_dense)
    a_h = torch.einsum("bdqp,dpr->bdqr", assign, m.hops[:, :p_max, :p_max])
    gb = circuits.batch
    local = circuits.gate_qubits.clamp(min=0)
    vmask = valid.unsqueeze(-1).unsqueeze(-1)
    count = valid.sum(1).clamp(min=1).view(-1, 1, 1).float()
    slots = ctx[gb.unsqueeze(1), :, local]
    c_mean = (slots * vmask).sum(1) / count
    c_max = slots.masked_fill(~vmask, -1e4).amax(1)
    c_min = slots.masked_fill(~vmask, 1e4).amin(1)
    multi = valid.sum(1).view(-1, 1, 1) > 1
    c_range = torch.where(multi, c_max - c_min, torch.zeros_like(c_max))
    g_eps = (q_eps[gb.unsqueeze(1), :, local] * vmask).sum(1) / count
    if coupler is not None:          # CZ error of the coupler (path) actually used, not the qubit mean
        pe = torch.zeros(g.size(0), n_dev); npair = torch.zeros(g.size(0), 1)
        for a, b_ in ((0, 1), (0, 2), (1, 2)):
            mm = (valid[:, a] & valid[:, b_]).float().unsqueeze(-1)
            A1 = assign[gb, :, local[:, a]]; A2 = assign[gb, :, local[:, b_]]      # (G, D, P)
            pe = pe + mm * torch.einsum("gdp,dpr,gdr->gd", A1, coupler[:, :p_max, :p_max], A2)
            npair = npair + mm
        g_eps = g_eps.clone()
        g_eps[..., 2] = torch.where(multi.view(-1, 1), pe / npair.clamp(min=1), g_eps[..., 2])
    dist = torch.zeros(g.size(0), n_dev)
    n_pairs = torch.zeros(g.size(0), 1)
    for a, b_ in ((0, 1), (0, 2), (1, 2)):
        mm = (valid[:, a] & valid[:, b_]).float().unsqueeze(-1)
        dist = dist + mm * (a_h[gb, :, local[:, a]] * assign[gb, :, local[:, b_]]).sum(-1)
        n_pairs = n_pairs + mm
    dist = dist / n_pairs.clamp(min=1)
    g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
    d_e = dev_glob.unsqueeze(0).expand(g.size(0), -1, -1)
    feats = [g_e, c_mean, c_range, d_e, g_e * c_mean, dist.unsqueeze(-1), torch.log1p(dist).unsqueeze(-1)]
    rf = routing_free(circuits, devices, m._rf_cache)
    re_ = routing_estimate(circuits, devices, m._re_cache)
    feats += [rf[gb].unsqueeze(-1), re_[gb].unsqueeze(-1)]
    n_ops = F.softplus(m.cost(torch.cat(feats, dim=-1)))
    base = F.softplus(m.cz_base(circuits.x[:, :N_GATE_LOCAL])) * multi.view(-1, 1)
    cz = base * (1 + n_ops[..., 2] * torch.expm1(re_)[gb] * (1 - rf[gb]))
    n_ops = torch.cat([n_ops[..., :2], cz.unsqueeze(-1)], dim=-1)
    if counts is not None:
        tot = scatter(n_ops, gb, dim=0, dim_size=n_circ, reduce="sum")
        ratio = torch.where(tot > 1e-6, counts / tot.clamp(min=1e-6), torch.ones_like(tot))
        n_ops = n_ops * ratio[gb]
    cost = (n_ops * g_eps * m.log_scale.exp()).sum(-1)
    return -scatter(cost, gb, dim=0, dim_size=n_circ, reduce="sum"), assign.argmax(-1)


def dense_layout(graphs, lay_list):
    """list of (n, D) node-index tensors -> (B, D, Qm) padded with −1."""
    qm = max(int(g.n_qubits) for g in graphs)
    out = torch.full((len(graphs), len(DEVS), qm), -1, dtype=torch.long)
    for i, l in enumerate(lay_list):
        out[i, :, : l.shape[0]] = l.t()
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def tracking(dp, dt):
    """mean over (variant, device) cells of corr(Δpred, Δtrue), and the pooled slope."""
    cells = [corr(dp[k], dt[k]) for k in dp if len(dp[k]) > 20]
    allp = np.concatenate(list(dp.values())); allt = np.concatenate(list(dt.values()))
    return float(np.nanmean(cells)), float(np.polyfit(allt, allp, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=1000)
    ap.add_argument("--n-variants", type=int, default=16)
    ap.add_argument("--coupler", action="store_true", help="only unseen calibrations, settings with coupler errors")
    ap.add_argument("--only", default=None, help="run directory of a single model (v8k, v8kd8, v8kd)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)
    rng = np.random.default_rng(0)
    params = load_hparams()
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    lab = torch.load(V8D / "v8d_labels.pt", weights_only=False)
    orig = torch.load(ORIG / "v8b_orig_labels.pt", weights_only=False)
    raw = json.loads((V8D / "variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    log("data loaded")
    test = set(json.loads((ROOT / "v8kd/random_control/random_seed5/predictions.json").read_text()))
    ok = [d for d in data if d.circuit_name in lab and d.circuit_name in orig
          and lab[d.circuit_name]["layout"].shape[0] == int(d.n_qubits)]
    pools = {"test": [d for d in ok if d.circuit_name in test], "train": [d for d in ok if d.circuit_name not in test]}
    pools = {k: [v[i] for i in rng.choice(len(v), min(args.n_circuits, len(v)), replace=False)] for k, v in pools.items()}
    unseen = [j for j in range(8, 100) if j % 10 in LOCAL_KINDS]
    variants = {"unseen": sorted(rng.choice(unseen, args.n_variants, replace=False).tolist()),
                "seen by v8kd8": [0, 1, 2, 3, 4, 5, 6, 7]}
    dev_o = variant_batch(raw, list(DEVS), params["lap_pe"])
    dev_v = {j: variant_batch(raw, [f"{d}/aug{j}" for d in DEVS], params["lap_pe"])
             for js in variants.values() for j in js}
    settings = ("model", "+ layout", "+ counts", "+ both", "fixed")
    if args.coupler:
        settings = ("model", "+ coupler", "+ layout + coupler", "+ both", "+ both + coupler")
        variants = {"unseen": variants["unseen"]}
    p_max = max(len(r["nodes"]) for r in raw.values())
    cpl_o = torch.stack([coupler_matrix(raw[d], p_max) for d in DEVS])
    cpl_v = {j: torch.stack([coupler_matrix(raw[f"{d}/aug{j}"], p_max) for d in DEVS]) for j in dev_v}
    LAY = ("+ layout", "+ both", "+ layout + coupler", "+ both + coupler")
    CNT = ("+ counts", "+ both", "+ both + coupler")
    L = ["# Why is local tracking low? Oracle substitutions (v8d calibrations, control split)", "",
         f"{args.n_circuits} circuits per pool; unseen = {len(variants['unseen'])} local v8d variants "
         f"(jitter / shuffle / bad qubits, aug ≥ 8); seen by v8kd8 = aug0-7.  Tracking: mean over "
         "(variant, device) of corr(Δlog F pred, Δlog F true), F > 0.01; slope: pooled regression of pred on true.", ""]

    # ---- label noise: agreement between compiler seeds on the same Δ ----
    names = {d.circuit_name for v in pools.values() for d in v}
    per_seed = {}
    for f in sorted(V8D.glob("task*.jsonl")) + sorted(ORIG.glob("task*.jsonl")):
        for line in f.read_text().splitlines():
            head = line[:200]
            name = head.split('"circuit": "', 1)[-1].split('"', 1)[0]
            if name not in names:
                continue
            r = json.loads(line)
            if "error" not in r:
                per_seed.setdefault(r["circuit"], {}).update(r["F"])
    a, b, c3, c3b = [], [], [], []
    for n, fs in per_seed.items():
        for j in variants["unseen"]:
            for d in DEVS:
                v, o = fs.get(f"{d}/aug{j}"), fs.get(f"{d}/orig")
                if not v or not o or len(v) < 3 or len(o) < 3 or min(v + o) <= 0.01:
                    continue
                a.append(np.log(v[0] / o[0])); b.append(np.log(v[1] / o[1]))
                c3.append(np.log(max(v[:2]) / max(o[:2]))); c3b.append(np.log(v[2] / o[2]))
    log("label noise done")
    L += ["## Label noise", "",
          f"corr(Δ seed 0, Δ seed 1) = {corr(a, b):.2f};  corr(Δ best of seeds 0-1, Δ seed 2) = {corr(c3, c3b):.2f}  "
          f"({len(a)} cases).  The labels are best of 3, so their own reliability is a bit higher: "
          "a perfect predictor cannot exceed roughly this agreement.", ""]

    for mname, run in MODELS.items():
        if args.only and run != args.only:
            continue
        model = build_model(KIND, params, dev_o, torch.device("cpu"))
        model.load_state_dict(torch.load(ROOT / run / "random_control/random_seed5/model.pth", map_location="cpu"))
        model.eval()
        L += [f"## {mname}", "", "| circuits | calibrations | " + " | ".join(settings) + " | placement moves (model / compiler) |",
              "| --- | --- | " + " | ".join("---" for _ in settings) + " | --- |"]
        for pool, graphs in pools.items():
            for vname, js in variants.items():
                if vname != "unseen" and run == "v8k":
                    continue
                dp = {s: {} for s in settings}; dt = {}
                moved_m, moved_c, n_q_tot = 0, 0, 0
                for start in range(0, len(graphs), 32):
                    gs = graphs[start: start + 32]
                    batch = next(iter(DataLoader(gs, batch_size=len(gs))))
                    lay_o = dense_layout(gs, [orig[g.circuit_name]["layout"][:, 0] for g in gs])
                    cnt_o = torch.stack([orig[g.circuit_name]["counts"][0] for g in gs]).float()
                    y_o = torch.stack([orig[g.circuit_name]["F"][0] for g in gs]).float()
                    with torch.no_grad():
                        base_o = {s: oracle_forward(model, batch, dev_o,
                                                    lay_o if s in LAY + ("fixed",) else None,
                                                    cnt_o if s in CNT + ("fixed",) else None,
                                                    cpl_o if "coupler" in s else None)
                                  for s in settings}
                        for j in js:
                            lay_v = dense_layout(gs, [lab[g.circuit_name]["layout"][:, j] for g in gs])
                            cnt_v = torch.stack([lab[g.circuit_name]["counts"][j] for g in gs]).float()
                            y_v = torch.stack([lab[g.circuit_name]["F"][j] for g in gs]).float()
                            keep = (y_v > 0.01) & (y_o > 0.01)
                            d_true = (torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6)))
                            for d in range(len(DEVS)):
                                dt.setdefault((j, d), []).append(d_true[:, d][keep[:, d]].numpy())
                            for s in settings:
                                lv = lay_o if s == "fixed" else (lay_v if s in LAY else None)
                                cv = cnt_o if s == "fixed" else (cnt_v if s in CNT else None)
                                lp_v, am_v = oracle_forward(model, batch, dev_v[j], lv, cv,
                                                           cpl_v[j] if "coupler" in s else None)
                                lp_o, am_o = base_o[s]
                                dpred = torch.clamp(lp_v, max=0) - torch.clamp(lp_o, max=0)
                                for d in range(len(DEVS)):
                                    dp[s].setdefault((j, d), []).append(dpred[:, d][keep[:, d]].numpy())
                                if s == "model":
                                    qm = lay_v.size(2)
                                    real = lay_v >= 0
                                    moved_m += int(((am_v[:, :, :qm] != am_o[:, :, :qm]) & real).sum())
                                    moved_c += int(((lay_v != lay_o) & real & (lay_o >= 0)).sum())
                                    n_q_tot += int(real.sum())
                dtc = {k: np.concatenate(v) for k, v in dt.items()}
                cells = []
                for s in settings:
                    t, sl = tracking({k: np.concatenate(v) for k, v in dp[s].items()}, dtc)
                    cells.append(f"{t:.2f} (slope {sl:.2f})")
                log(f"{mname} {pool} {vname}")
                L.append(f"| {pool} | {vname} | " + " | ".join(cells) +
                         f" | {moved_m / max(n_q_tot, 1):.0%} / {moved_c / max(n_q_tot, 1):.0%} |")
                print(L[-1], flush=True)
        L.append("")
    out = HERE / "results" / f"report_tracking_oracle{'_coupler' if args.coupler else ''}{'_' + args.only if args.only else ''}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
