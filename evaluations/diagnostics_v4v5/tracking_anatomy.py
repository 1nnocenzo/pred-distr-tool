#!/usr/bin/env python3
"""Where and why does the model miss local calibration changes?

Part 1 — where (calibration-shift test: 457 unseen circuits, truth = best of 5 compilations, local
variants shuffle / jitterA / jitterB / jitterS; v8kd40 and v8kdl40, 3 seeds each):
  tracking (corr of predicted vs true Δlog F, F > 0.01) and regret after the change, split by
  family, number of qubits, variant, device and routing need (compiled CZ / logical 2q gates).

Part 2 — why (v8kdl40, control seed 5; device side on the original devices + 100 v8d variants each):
  a. PCA of the physical-qubit embeddings h_p: variance per component and what it correlates with
     (position on the chip, degree, 1q / readout / mean CZ error, T1, T2);
     linear R² of h_p from the errors alone vs from position + degree alone;
  b. probes: ridge regression from h_p to the qubit's total error and to its rank within the device
     (held-out calibrations);
  c. placement scores: share of |Δscore| when only one kind of error is taken from the variant
     (1q, readout, CZ, T1/T2), and the same question for the compiler: least-squares weights of
     Δ(occupancy) on Δε_1q, Δε_readout, Δε_cz for the compiler and for the model (v8d layouts);
  d. Δlog F decomposition on the test variants: errors read at fixed placement (assignment frozen at
     the original calibration) vs the rest (placement and counts moving), when the model gets the
     direction of the change right vs wrong.
    python tracking_anatomy.py [--compare v8kd40,v8kdl40] [--main v8kdl40:sinkhorn_rf_route_czc]
Part 1 compares the runs of --compare (every seed found in the calibration test); parts 2 and 3
look inside the --main model (control split, seed 5).  Standard evaluation of a new candidate:
run it after the candidate's calibration test, with the current main model in --compare.
"""
from __future__ import annotations

import os
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
sys.path.insert(0, str(EVAL / "calibration_shift"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

ROOT = EVAL / os.environ.get("GS_RESULTS_ROOT", "generalization_v9") / "results"
RK = EVAL / "calibration_shift" / os.environ.get("GS_CALIB_DIR", "results_k")
V8D = EVAL / "compile_check/results/v8d"
ORIG = EVAL / "compile_check/results/v8b_orig"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
VARS = ("shuffle", "jitterA", "jitterB", "jitterS")
GROUPS: dict = {}          # run -> its seeds in the calibration test (set in main from --compare)
OFF = 28


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 and a.std() > 0 and b.std() > 0 else float("nan")


def r2_ridge(X, y, train, test, lam=1e-2):
    X = np.c_[X, np.ones(len(X))]
    A = X[train].T @ X[train] + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[train].T @ y[train])
    p = X[test] @ w
    return float(1 - ((y[test] - p) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum())


def part1(L, data, orig_counts):
    T = json.loads((RK / "truth.json").read_text()); P = json.loads((RK / "predictions.json").read_text())
    best = lambda n, k: max([f for f in T[n][k] if f is not None], default=None)
    names = [n for n in T if all(best(n, f"{d}/{v}") is not None for d in DEVS for v in VARS + ("orig",))]
    nq = {n: int(re.search(r"_q(\d+)_", n).group(1)) for n in names}
    l2 = {n: int((data[n].x[:, OFF + 6] >= 2).sum()) for n in names if n in data}
    def route(n, di):
        c = orig_counts.get(n)
        if c is None or l2.get(n, 0) == 0:
            return "no 2q gates"
        r = float(c["counts"][0, di, 2]) / l2[n]
        return "no routing (CZ ≈ 2q gates)" if r <= 1.05 else ("light routing (≤ 2×)" if r <= 2 else "heavy routing (> 2×)")
    keys = {"family": lambda n, d, v: n.split("/")[0], "qubits": lambda n, d, v: f"{(nq[n] - 1) // 5 * 5 + 1:02d}–{(nq[n] - 1) // 5 * 5 + 5:02d}",
            "variant": lambda n, d, v: v, "device": lambda n, d, v: d, "routing": lambda n, d, v: route(n, DEVS.index(d))}
    L += ["## 1. Where the model misses (calibration-shift test, local variants)", "",
          "Tracking = corr(Δlog F predicted, true), F > 0.01, pooled over the cases of each group (so it is higher than the "
          "official metric, a mean over (variant, device) cells: compare groups with each other, not with it); mean over seeds. "
          "|ΔF| = mean true change of F. Groups with fewer than 60 cases are left out.", ""]
    for kname, kf in keys.items():
        L += [f"| {kname} | cases | mean \\|ΔF\\| | " + " | ".join(f"tracking {g}" for g in GROUPS) + " | " + " | ".join(f"error on ΔF {g}" for g in GROUPS) + " |",
              "| --- | --- | --- | " + " | ".join(["---"] * (2 * len(GROUPS))) + " |"]
        cells = defaultdict(lambda: {"dt": [], "dp": defaultdict(list)})
        for n in names:
            for d in DEVS:
                fo = best(n, f"{d}/orig")
                for v in VARS:
                    fv = best(n, f"{d}/{v}")
                    if fo <= 0.01 or fv <= 0.01:
                        continue
                    key = kf(n, d, v)
                    cells[key]["dt"].append((np.log(fv / fo), fv - fo))
                    for g, ms in GROUPS.items():
                        for m in ms:
                            if m in P:
                                po, pv = P[m][n][f"{d}/orig"], P[m][n][f"{d}/{v}"]
                                cells[key]["dp"][(g, m)].append((np.log(max(pv, 1e-6) / max(po, 1e-6)), pv - po))
        for key in sorted(cells):
            c = cells[key]
            if len(c["dt"]) < 60:
                continue
            dt = np.array(c["dt"])
            trk, err = [], []
            for g, ms in GROUPS.items():
                tg, eg = [], []
                for m in ms:
                    dp = np.array(c["dp"].get((g, m), []))
                    if len(dp):
                        tg.append(corr(dp[:, 0], dt[:, 0])); eg.append(float(np.abs(dp[:, 1] - dt[:, 1]).mean()))
                trk.append(f"{np.nanmean(tg):.2f}"); err.append(f"{np.mean(eg):.4f}")
            L.append(f"| {key} | {len(dt)} | {np.abs(dt[:, 1]).mean():.4f} | " + " | ".join(trk) + " | " + " | ".join(err) + " |")
        L.append("")


def device_tables(raw, keys):
    """Per (device-variant) physical-qubit features: errors (r, readout, mean CZ), T1, T2, degree."""
    out = []
    for k in keys:
        r = raw[k]
        nodes = np.array(r["nodes"], dtype=float)
        n = len(nodes)
        cz = np.zeros(n); cnt = np.zeros(n)
        for (a, b), e in zip(r["edges"], r["edge_feats"]):
            cz[a] += e; cz[b] += e; cnt[a] += 1; cnt[b] += 1
        cz = np.where(cnt > 0, cz / np.maximum(cnt, 1), 0)
        out.append(np.c_[nodes[:, 0], nodes[:, 1], cz, nodes[:, 2], nodes[:, 3], nodes[:, 4]])
    return out


def part2(L, model, params, data, raw, lab, orig):
    L += [f"## 2. Why: inside the placement ({model.run_name}, control, seed 5)", ""]
    # a/b. embeddings over the original devices and the 100 v8d variants of each
    keys = [d for d in DEVS] + [f"{d}/aug{j}" for j in range(100) for d in DEVS]
    H, F, G, R = [], [], [], []
    for start in range(0, len(keys), 30):
        ks = keys[start: start + 30]
        dv = variant_batch(raw, ks, params["lap_pe"])
        with torch.no_grad():
            h, mask, _ = model._devices(dv)
        feats = device_tables(raw, ks)
        pe = to_dense_batch(dv.x[:, -params["lap_pe"]:], dv.batch)[0]
        for i, k in enumerate(ks):
            n = int(mask[i].sum())
            H.append(h[i, :n].numpy()); f = feats[i]; F.append(f)
            G.append(np.c_[pe[i, :n].numpy(), f[:, 5:6]])
            tot = f[:, :3].sum(1)
            R.append(np.c_[np.full(n, start + i), tot, np.argsort(np.argsort(tot)) / max(n - 1, 1)])
    H, F, G, R = np.vstack(H), np.vstack(F), np.vstack(G), np.vstack(R)
    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    pcs = Hc @ Vt[:6].T
    names = ["1q error", "readout error", "mean CZ error", "T1", "T2", "degree"]
    L += ["### a. PCA of the physical-qubit embeddings", "",
          f"{len(H)} physical qubits (3 devices × 101 calibrations). Variance explained by the first 6 components: "
          + ", ".join(f"{x:.0%}" for x in ev[:6]) + ".", "",
          "| component | " + " | ".join(names) + " | position (best of 8 Laplacian PE) |", "| --- | " + " | ".join("---" for _ in names) + " | --- |"]
    for c in range(6):
        row = [f"{corr(pcs[:, c], F[:, j]):+.2f}" for j in range(6)]
        row.append(f"{max(abs(corr(pcs[:, c], G[:, j])) for j in range(G.shape[1] - 1)):.2f}")
        L.append(f"| PC{c + 1} ({ev[c]:.0%}) | " + " | ".join(row) + " |")
    rng = np.random.default_rng(0)
    cal = R[:, 0].astype(int); cals = np.unique(cal); test_c = set(rng.choice(cals, len(cals) // 4, replace=False).tolist())
    te = np.array([c in test_c for c in cal]); tr = ~te
    r2_err = np.mean([r2_ridge(F[:, :5], H[:, j], tr, te) for j in range(H.shape[1])])
    r2_pos = np.mean([r2_ridge(G, H[:, j], tr, te) for j in range(H.shape[1])])
    L += ["", f"Linear R² of the embedding (mean over its {H.shape[1]} dimensions, held-out calibrations): from the errors and T1/T2 alone "
          f"**{r2_err:.2f}**, from position and degree alone **{r2_pos:.2f}**.", "",
          "### b. Can the error be read back from the embedding?", "",
          "| probe target (ridge from h_p, held-out calibrations) | R² |", "| --- | --- |",
          f"| total error of the qubit (1q + readout + mean CZ) | {r2_ridge(H, R[:, 1], tr, te):.2f} |",
          f"| its rank within the device (0 = best, 1 = worst) | {r2_ridge(H, R[:, 2], tr, te):.2f} |",
          f"| 1q error | {r2_ridge(H, F[:, 0], tr, te):.2f} |", f"| readout error | {r2_ridge(H, F[:, 1], tr, te):.2f} |",
          f"| mean CZ error | {r2_ridge(H, F[:, 2], tr, te):.2f} |", ""]
    # c. placement scores and occupancy, what drives the change
    test = sorted(json.loads((ROOT / model.run_name / "random_control/random_seed5/predictions.json").read_text()))
    pool = [data[n] for n in test if n in data and n in lab and n in orig and lab[n]["layout"].shape[0] == int(data[n].n_qubits)]
    pool = [pool[i] for i in rng.choice(len(pool), 300, replace=False)]
    js = [11, 19, 50, 55, 56, 61, 65, 72, 73, 82]
    dev_o = variant_batch(raw, list(DEVS), params["lap_pe"])

    def swap(kind, j):
        """original devices with only one kind of error taken from variant j"""
        ks = []
        for d in DEVS:
            o, v = raw[d], raw[f"{d}/aug{j}"]
            nodes = np.array(o["nodes"], dtype=float); vn = np.array(v["nodes"], dtype=float)
            ef = list(o["edge_feats"])
            if kind == "1q":
                nodes[:, 0] = vn[:, 0]
            elif kind == "readout":
                nodes[:, 1] = vn[:, 1]
            elif kind == "T1/T2":
                nodes[:, 2:4] = vn[:, 2:4]
            elif kind == "CZ":
                ef = list(v["edge_feats"])
            raw[f"{d}/swap"] = {"nodes": nodes.tolist(), "edges": o["edges"], "edge_feats": ef}
            ks.append(f"{d}/swap")
        return variant_batch(raw, ks, params["lap_pe"])

    def scores(b, dv):
        with torch.no_grad():
            model(b, dv)
            return model.last_assign.sum(2)                      # occupancy (B, D, P)
    kinds = ("1q", "readout", "CZ", "T1/T2")
    share = defaultdict(list)
    Xc, yc, ym = [], [], []
    eps_o = [np.c_[f[:, 0], f[:, 1], f[:, 2]] for f in device_tables(raw, list(DEVS))]
    for start in range(0, len(pool), 32):
        gs = pool[start: start + 32]
        b = next(iter(DataLoader(gs, batch_size=len(gs))))
        u_o = scores(b, dev_o)
        for j in js:
            dv = variant_batch(raw, [f"{d}/aug{j}" for d in DEVS], params["lap_pe"])
            du = (scores(b, dv) - u_o).abs().sum(-1)            # (B, D)
            parts = {k: (scores(b, swap(k, j)) - u_o).abs().sum(-1) for k in kinds}
            tot = sum(parts.values()).clamp(min=1e-9)
            for k in kinds:
                share[k].append(float((parts[k] / tot).mean()))
            share["sum of parts / full change"].append(float((tot / du.clamp(min=1e-9)).mean()))
            eps_v = [np.c_[f[:, 0], f[:, 1], f[:, 2]] for f in device_tables(raw, [f"{d}/aug{j}" for d in DEVS])]
            u_v = scores(b, dv)
            for i, g in enumerate(gs):
                n = int(g.n_qubits)
                lo = orig[g.circuit_name]["layout"][:, 0]; lv = lab[g.circuit_name]["layout"][:, j]
                for di in range(3):
                    P = len(eps_o[di])
                    if (lo[:, di] < 0).any() or (lv[:, di] < 0).any():
                        continue
                    oc = np.zeros(P); vc = np.zeros(P)
                    oc[lo[:, di].numpy()] = 1; vc[lv[:, di].numpy()] = 1
                    Xc.append(eps_v[di] - eps_o[di]); yc.append(vc - oc); ym.append((u_v[i, di, :P] - u_o[i, di, :P]).numpy())
    Xc, yc, ym = np.vstack(Xc), np.concatenate(yc), np.concatenate(ym)
    Xs = Xc / Xc.std(0)
    wc = np.linalg.lstsq(np.c_[Xs, np.ones(len(Xs))], yc, rcond=None)[0][:3]
    wm = np.linalg.lstsq(np.c_[Xs, np.ones(len(Xs))], ym, rcond=None)[0][:3]
    L += ["### c. What moves the placement, and what moves the compiler", "",
          "Share of the change of the model's qubit occupancy when only one kind of error is taken from the variant "
          "(300 test circuits × 10 local v8d calibrations):", "",
          "| error changed alone | share of the occupancy change |", "| --- | --- |"]
    L += [f"| {k} | {np.mean(share[k]):.0%} |" for k in kinds]
    L += [f"| (sum of the single-kind changes / change with all errors) | {np.mean(share['sum of parts / full change']):.2f} |", "",
          "Least-squares weight of Δ(occupancy of a physical qubit) on the standardised change of its errors "
          "(negative = leaves a qubit whose error grew):", "",
          "| | Δ 1q error | Δ readout error | Δ mean CZ error |", "| --- | --- | --- | --- |",
          f"| compiler | {wc[0]:+.3f} | {wc[1]:+.3f} | {wc[2]:+.3f} |", f"| model | {wm[0]:+.3f} | {wm[1]:+.3f} | {wm[2]:+.3f} |", ""]
    return pool


def part3(L, model, params, data, raw):
    """d. errors at fixed placement vs the rest, on the calibration-shift test variants."""
    import run_shift
    T = json.loads((RK / "truth.json").read_text())
    best = lambda n, k: max([f for f in T[n][k] if f is not None], default=None)
    names = [n for n in T if n in data and all(best(n, f"{d}/{v}") is not None for d in DEVS for v in VARS + ("orig",))]
    vraw = json.loads((RK / "variants_raw.json").read_text())
    dev_o = run_shift._variant_batch(vraw, [f"{d}/orig" for d in DEVS], params["lap_pe"])
    orig_sink = model._sinkhorn
    rows = []
    for start in range(0, len(names), 32):
        ns = names[start: start + 32]
        b = next(iter(DataLoader([data[n] for n in ns], batch_size=len(ns))))
        with torch.no_grad():
            lo = model(b, dev_o); A_o = model.last_assign
        for v in VARS:
            dv = run_shift._variant_batch(vraw, [f"{d}/{v}" for d in DEVS], params["lap_pe"])
            with torch.no_grad():
                lv = model(b, dv)
                model._sinkhorn = lambda *a, **k: A_o            # placement frozen at the original calibration
                lfix = model(b, dv)
                model._sinkhorn = orig_sink
            for i, n in enumerate(ns):
                for di, d in enumerate(DEVS):
                    fo, fv = best(n, f"{d}/orig"), best(n, f"{d}/{v}")
                    if fo <= 0.01 or fv <= 0.01:
                        continue
                    rows.append((np.log(fv / fo), float(lv[i, di] - lo[i, di]), float(lfix[i, di] - lo[i, di])))
    a = np.array(rows)
    dt, dp, dfix = a[:, 0], a[:, 1], a[:, 2]
    rest = dp - dfix
    right = np.sign(dp) == np.sign(dt)
    big = np.abs(dt) > 0.03
    L += ["### d. Errors read at fixed placement vs placement and counts moving (calibration-shift test)", "",
          f"{len(a)} (circuit, device, local variant) cases. Δ predicted = errors-at-fixed-placement term + rest.", "",
          "| | corr with true Δlog F | mean \\|term\\| | share of \\|Δ predicted\\| |", "| --- | --- | --- | --- |",
          f"| errors at fixed placement | {corr(dfix, dt):.2f} | {np.abs(dfix).mean():.4f} | {np.abs(dfix).mean() / np.abs(dp).mean():.0%} |",
          f"| placement and counts moving | {corr(rest, dt):.2f} | {np.abs(rest).mean():.4f} | {np.abs(rest).mean() / np.abs(dp).mean():.0%} |",
          f"| full prediction | {corr(dp, dt):.2f} | {np.abs(dp).mean():.4f} | – |",
          f"| true change | – | {np.abs(dt).mean():.4f} | – |", "",
          f"Direction right in {right.mean():.0%} of the cases ({right[big].mean():.0%} when |Δlog F| > 0.03). "
          f"When it is wrong and the change is large, the fixed-placement term has the true sign in "
          f"{(np.sign(dfix) == np.sign(dt))[~right & big].mean():.0%} of the cases and the moving term in "
          f"{(np.sign(rest) == np.sign(dt))[~right & big].mean():.0%}; the true change is "
          f"{'mostly a gain' if (dt[~right & big] > 0).mean() > 0.5 else 'mostly a loss'} "
          f"({(dt[~right & big] > 0).mean():.0%} gains).", ""]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="v8kd40,v8kdl40", help="runs compared in part 1")
    ap.add_argument("--main", default="v8kdl40:sinkhorn_rf_route_czc", help="run:kind looked into in parts 2-3")
    args = ap.parse_args()
    P = json.loads((RK / "predictions.json").read_text())
    for r in args.compare.split(","):
        GROUPS[r] = [m for m in (f"{r}@s{s}" for s in (5, 6, 7)) if m in P]
    run, kind = args.main.split(":")
    torch.set_num_threads(8)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    params = load_hparams()
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    lab = torch.load(V8D / "v8d_labels.pt", weights_only=False)
    orig = torch.load(ORIG / "v8b_orig_labels.pt", weights_only=False)
    raw = json.loads((V8D / "variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    L = ["# Anatomy of the local calibration tracking", "",
         f"Compared: {', '.join(f'{g} ({len(m)} seeds)' for g, m in GROUPS.items())}; looked into: {run}.", ""]
    part1(L, data, orig); log("part 1 done")
    dev = variant_batch(raw, list(DEVS), params["lap_pe"])
    model = build_model(kind, params, dev, torch.device("cpu"))
    model.load_state_dict(torch.load(ROOT / run / "random_control/random_seed5/model.pth", map_location="cpu")); model.eval()
    model.run_name = run
    part2(L, model, params, data, raw, lab, orig); log("part 2 done")
    part3(L, model, params, data, raw); log("part 3 done")
    out = HERE / "results" / f"report_tracking_anatomy_{run}{os.environ.get('GS_TAG', '')}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
