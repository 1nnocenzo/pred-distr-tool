#!/usr/bin/env python3
"""Using the predictor to decide where (and whether) to compile: every scenario, one rule.

Rule (adaptive top-k): rank the candidate backends by predicted fidelity; compile on the first and on
every backend predicted within δ of it; run on the best of those (its true fidelity is then known).
δ = 0 trusts the model (1 compilation), δ = ∞ is the exhaustive search.  Sweeping δ gives a cost /
quality curve: compilations per circuit vs fidelity lost against the truly best backend.

Backends: the 39 calibrations of the calibration-shift test (3 chips × original + 12 variants), truth
= best of 5 compilations, 457 unseen circuits.

  A. cold start       a new circuit, choose among a fleet of N backends (N = 3, 9, 27, 33; fleets
                      sampled at random, 100 per N, the same in every section)
  B. calibration update  the circuit runs on chip d; the calibration of all 3 chips changes (local,
                      moderate, large); the starting device is the truly best one or the model's pick.
                      Policies: keep; switch to the model's first; re-check (compile all 3) only where
                      the model predicts a change; adaptive top-k on the new calibration (the current
                      device always included); re-check all.
Random baseline: compile on k random backends (k = the same mean cost).
    python decision_policies.py --models v8a_c001,v8kd40,v8kdl40 [--tag name]
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RK = HERE / os.environ.get("GS_CALIB_DIR", "results_k")
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
KINDS = {"local": ("shuffle", "jitterA", "jitterB", "jitterS"), "moderate": ("x0.7", "x0.8", "x1.2", "x1.3"),
         "large": ("x0.5", "x2")}
DELTAS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 1.0)


def topk_policy(t, p, delta, must=None):
    """t, p: (C, B) true / predicted F. Returns (fidelity lost, compilations per circuit)."""
    top = p.max(1, keepdims=True)
    comp = p >= top - delta
    if must is not None:
        comp[np.arange(len(t)), must] = True
    got = np.where(comp, t, -np.inf).max(1)
    return float((t.max(1) - got).mean()), float(comp.sum(1).mean())


def random_policy(t, k, rng, must=None, n=50):
    C, B = t.shape
    losses = []
    for _ in range(n):
        kk = np.floor(k) + (rng.random(C) < (k - np.floor(k)))
        sel = np.zeros((C, B), bool)
        for i in range(C):
            idx = rng.permutation(B)[: int(max(kk[i], 1))]
            sel[i, idx] = True
        if must is not None:
            sel[np.arange(C), must] = True
        losses.append((t.max(1) - np.where(sel, t, -np.inf).max(1)).mean())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8a_c001,v8kd40,v8kdl40")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    T = json.loads((RK / "truth.json").read_text()); P = json.loads((RK / "predictions.json").read_text())
    best = lambda n, k: max([f for f in T[n][k] if f is not None], default=None)
    variants = ["orig"] + [v for vs in KINDS.values() for v in vs]
    keys = [f"{d}/{v}" for d in DEVS for v in variants]
    names = [n for n in T if all(best(n, k) is not None for k in keys)]
    Tm = np.array([[best(n, k) for k in keys] for n in names])
    runs = args.models.split(",")
    seeds = {r: [m for m in (f"{r}@s{s}" for s in (5, 6, 7)) if m in P] for r in runs}
    Pm = {m: np.array([[P[m][n][k] for k in keys] for n in names]) for ms in seeds.values() for m in ms}
    # "<run>_ens": the seed ensemble of <run> (geometric mean of the seeds' F), evaluated as one model
    for r in runs:
        if r.endswith("_ens"):
            base = [m for m in (f"{r[:-4]}@s{s}" for s in (5, 6, 7)) if m in P]
            Pm[f"{r}@ens"] = np.exp(np.mean([np.log(np.clip([[P[m][n][k] for k in keys] for n in names], 1e-12, None))
                                             for m in base], 0))
            seeds[r] = [f"{r}@ens"]
    rng = np.random.default_rng(0)
    # one set of fleets for every section (small fleets vary a lot: 100 random draws per N)
    FLEETS = {N: [np.arange(len(keys))] if N == len(keys) else [rng.choice(len(keys), N, replace=False) for _ in range(100)]
              for N in (3, 9, 27, len(keys))}
    out = {"cold": {}, "update": {}}
    L = ["# Where to compile: one rule for every scenario", "",
         f"{len(names)} unseen circuits; {len(keys)} backends (3 chips × original + 10 calibration variants); truth = best of 5 compilations. "
         "Adaptive top-k: compile on every backend predicted within δ of the predicted best and run on the best of them. "
         "Cost = compilations per circuit; lost = mean true fidelity below the truly best backend. Models: mean over seeds.", ""]

    # A. cold start
    L += ["## A. Cold start: choose among N backends", "",
          "| N | exhaustive cost | model only (δ = 0): lost | cost for lost < 0.001 | random at that cost: lost | " +
          " | ".join(f"{r}: δ 0 / 0.005 / 0.01 lost (cost)" for r in runs) + " |",
          "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in runs) + " |"]
    for N in (3, 9, 27, len(keys)):
        fleets = FLEETS[N]
        res = {}
        for r, ms in seeds.items():
            rows = []
            for dl in DELTAS:
                vals = [topk_policy(Tm[:, f], Pm[m][:, f], dl) for m in ms for f in fleets]
                rows.append((dl, float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))))
            res[r] = rows
        res["random"] = [(float(k), float(np.mean([random_policy(Tm[:, f], k, rng, n=5) for f in fleets[:5]])), float(k))
                         for k in sorted({1, 2, 3, max(1, N // 4), max(1, N // 2), N}) if k <= N]
        out["cold"][N] = res
        main_r = runs[-1]
        ok = [x for x in res[main_r] if x[1] < 0.001]
        c_ok = ok[0][2] if ok else float("nan")
        rand = float(np.mean([random_policy(Tm[:, f], c_ok, rng, n=10) for f in fleets[:5]])) if ok else float("nan")
        cells = [" / ".join(f"{x[1]:.4f}" for x in res[r][:4:1] if x[0] in (0.0, 0.005, 0.01)) +
                 f" ({res[r][3][2]:.1f})" for r in runs]
        L.append(f"| {N} | {N} | {res[main_r][0][1]:.4f} | {c_ok:.1f} ({main_r}) | {rand:.4f} | " + " | ".join(cells) + " |")
    L += ["", f"\"cost for lost < 0.001\" and the random column refer to {runs[-1]}; the last columns give δ = 0, 0.005, 0.01 and, in brackets, the cost at δ = 0.01.", ""]

    # B. calibration update, 3 chips
    L += ["## B. Calibration update on the 3 chips", ""]
    io = [keys.index(f"{d}/orig") for d in DEVS]
    for kind, vs in KINDS.items():
        rows_t = np.concatenate([Tm[:, [keys.index(f"{d}/{v}") for d in DEVS]] for v in vs])
        to = np.concatenate([Tm[:, io] for _ in vs])
        oracle = rows_t.max(1)
        idx = np.arange(len(rows_t))
        L += [f"### {kind.capitalize()} changes ({', '.join(vs)})", "",
              "| start | policy | " + " | ".join(f"{r}: lost (compilations)" for r in runs) + " |", "| --- | --- | " + " | ".join("---" for _ in runs) + " |"]
        res = {}
        for start in ("truly best", "model's pick"):
            pol = {}
            for r, ms in seeds.items():
                acc = {}
                for m in ms:
                    pv = np.concatenate([Pm[m][:, [keys.index(f"{d}/{v}") for d in DEVS]] for v in vs])
                    po = np.concatenate([Pm[m][:, io] for _ in vs])
                    old = to.argmax(1) if start == "truly best" else po.argmax(1)
                    lost_old = oracle - rows_t[idx, old]
                    pick = pv.argmax(1)
                    acc.setdefault("keep (0)", []).append((lost_old.mean(), 0.0))
                    acc.setdefault("switch to the model's first (1)", []).append(((oracle - rows_t[idx, pick]).mean(), 1.0))
                    fl = pick != old
                    acc.setdefault("re-check where it predicts a change", []).append((np.where(fl, 0, lost_old).mean(), 1 + 2 * fl.mean()))
                    for dl in (0.005, 0.01, 0.02):
                        acc.setdefault(f"adaptive top-k, δ {dl}, current included", []).append(topk_policy(rows_t, pv, dl, must=old))
                    acc.setdefault("re-check all (3)", []).append((0.0, 3.0))
                pol[r] = {k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for k, v in acc.items()}
            res[start] = pol
            for k in pol[runs[0]]:
                L.append(f"| {start} | {k} | " + " | ".join(f"{pol[r][k][0]:.4f} ({pol[r][k][1]:.2f})" for r in runs) + " |")
        out["update"][kind] = res
        L.append("")
    # C. fine δ sweep: fidelity lost, compilations and share of circuits that compile more than one backend
    FINE = (0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1)
    sweep = {"cold": {}, "update": {}}
    for N in (3, 9, 33):
        fleets = FLEETS[N]
        sweep["cold"][N] = {}
        for r, ms in seeds.items():
            pts = []
            for dl in FINE:
                v = []
                for m in ms:
                    for f in fleets:
                        lost, cost = topk_policy(Tm[:, f], Pm[m][:, f], dl)
                        pf = Pm[m][:, f]
                        more = float(((pf >= pf.max(1, keepdims=True) - dl).sum(1) > 1).mean())
                        v.append((lost, cost, more))
                pts.append((dl, *[float(np.mean([x[i] for x in v])) for i in range(3)]))
            sweep["cold"][N][r] = pts
    for kind, vs in KINDS.items():
        rows_t = np.concatenate([Tm[:, [keys.index(f"{d}/{v}") for d in DEVS]] for v in vs])
        sweep["update"][kind] = {}
        for r, ms in seeds.items():
            pts = []
            for dl in FINE:
                v = []
                for m in ms:
                    pv = np.concatenate([Pm[m][:, [keys.index(f"{d}/{x}") for d in DEVS]] for x in vs])
                    po = np.concatenate([Pm[m][:, io] for _ in vs])
                    old = po.argmax(1)
                    lost, cost = topk_policy(rows_t, pv, dl, must=old)
                    comp = pv >= pv.max(1, keepdims=True) - dl
                    comp[np.arange(len(old)), old] = True
                    v.append((lost, cost, float((comp.sum(1) > 1).mean())))
                pts.append((dl, *[float(np.mean([x[i] for x in v])) for i in range(3)]))
            sweep["update"][kind][r] = pts
    out["sweep"] = sweep
    L += ["## C. Choosing δ", "", "Start of an update = the model's previous pick. 'Recompiled' = circuits compiled on more than one backend.", "",
          "| δ | " + " | ".join(f"cold N={N}: lost / compilations" for N in (3, 9, 33)) + " | " +
          " | ".join(f"update {k}: lost / recompiled" for k in KINDS) + " |", "| --- | " + " | ".join("---" for _ in range(6)) + " |"]
    main_r = runs[-1]
    for i, dl in enumerate(FINE):
        c = [f"{sweep['cold'][N][main_r][i][1]:.4f} / {sweep['cold'][N][main_r][i][2]:.2f}" for N in (3, 9, 33)]
        u = [f"{sweep['update'][k][main_r][i][1]:.4f} / {sweep['update'][k][main_r][i][3]:.0%}" for k in KINDS]
        L.append(f"| {dl} | " + " | ".join(c + u) + " |")
    L += ["", f"Model: {main_r}.", ""]
    tag = f"_{args.tag}" if args.tag else ""
    (RK / f"decision_policies{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    (RK / f"decision_policies{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
