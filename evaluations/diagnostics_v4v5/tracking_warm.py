#!/usr/bin/env python3
"""Tracking when the circuit was already compiled on the old calibration ("warm" start).

After a calibration change, staying with the existing compilation changes log F exactly by
    Δ_stay = −(Σ_p U_r(p) Δε_r(p) + Σ_p U_m(p) Δε_m(p) + Σ_e U_cz(e) Δε_cz(e)),
U = usage maps of the old compilation, ε = −log(1 − e).  The true change (best of 3 recompilations
on the new calibration) also contains the compiler's re-optimisation gain.  Same 300 pilot circuits
and 10 local v8d calibrations as tracking_oracle_usage.py (same random draws), so the numbers
compare with its model column (v8kdl40: 0.69 there).
    python tracking_warm.py
"""
from __future__ import annotations

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
from gsv3.devices import _physical_errors  # noqa: E402

ROOT = EVAL / "generalization_v9/results"
V8D = EVAL / "compile_check/results/v8d"
ORIG = EVAL / "compile_check/results/v8b_orig"
USAGE = EVAL / "compile_check/results/v8u/usage_labels.pt"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LOCAL_KINDS = {0, 1, 2, 3, 4, 5, 6, 9}


def eps(raw):
    ph = _physical_errors(raw).numpy()
    return ph[:, 0], ph[:, 1], np.asarray(raw["edge_feats"], dtype=float).reshape(-1)


def main():
    rng = np.random.default_rng(0)
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    use = torch.load(USAGE, weights_only=False)
    lab = torch.load(V8D / "v8d_labels.pt", weights_only=False)
    orig = torch.load(ORIG / "v8b_orig_labels.pt", weights_only=False)
    test = set(json.loads((ROOT / "v8kd40/random_control/random_seed5/predictions.json").read_text()))
    pool = [n for n in sorted(use) if n in test and n in data and n in lab and n in orig
            and "F_aug" in use[n] and lab[n]["layout"].shape[0] == int(data[n].n_qubits)]
    pool = [pool[i] for i in rng.choice(len(pool), min(300, len(pool)), replace=False)]
    js = sorted(rng.choice([j for j in range(8, 100) if j % 10 in LOCAL_KINDS], 10, replace=False).tolist())
    raw = json.loads((V8D / "variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    E = {d: eps(raw[d]) for d in DEVS}
    # check of the formula on the old compilation: −Σ U ε vs log F of the best compilation
    chk = []
    for n in pool:
        for di, d in enumerate(DEVS):
            u = use[n]; er, em, ec = E[d]
            lf = -(u["r"][di, :len(er)].numpy() @ er + u["m"][di, :len(em)].numpy() @ em + u["cz"][di, :len(ec)].numpy() @ ec)
            if float(u["F"][di]) > 1e-3:
                chk.append(abs(lf - np.log(float(u["F"][di]))))
    rows = {}
    for j in js:
        for di, d in enumerate(DEVS):
            er0, em0, ec0 = E[d]; er1, em1, ec1 = eps(raw[f"{d}/aug{j}"])
            st, tr, fo_l = [], [], []
            for n in pool:
                fo, fv = float(orig[n]["F"][0][di]), float(lab[n]["F"][j, di])
                if fo <= 0.01 or fv <= 0.01:
                    continue
                u = use[n]
                ds = -(u["r"][di, :len(er0)].numpy() @ (er1 - er0) + u["m"][di, :len(em0)].numpy() @ (em1 - em0)
                       + u["cz"][di, :len(ec0)].numpy() @ (ec1 - ec0))
                st.append(ds); tr.append(np.log(fv / fo))
            rows[(j, d)] = (np.array(st), np.array(tr))
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    trk = np.mean([corr(s, t) for s, t in rows.values()])
    s_all = np.concatenate([s for s, _ in rows.values()]); t_all = np.concatenate([t for _, t in rows.values()])
    gain = t_all - s_all
    L = ["# Tracking from an existing compilation (warm start)", "",
         f"{len(pool)} pilot circuits, local v8d calibrations {js}. Formula check on the old compilation: "
         f"median |−ΣUε − log F| = {np.median(chk):.1e}.", "",
         "| predictor of Δlog F | local tracking (mean over calibration × device) |", "| --- | --- |",
         f"| stay on the old compilation, exact (Δ_stay) | {trk:.2f} |",
         "| v8kdl40 from scratch (tracking_oracle_usage.py, same cases) | 0.69 |", "",
         f"Re-optimisation gain of the compiler (true Δ − Δ_stay): mean {gain.mean():+.4f}, "
         f"positive in {(gain > 0).mean():.0%} of the cases; |true Δ| mean {np.abs(t_all).mean():.4f}, "
         f"|Δ_stay| mean {np.abs(s_all).mean():.4f}; pooled corr(Δ_stay, true Δ) {corr(s_all, t_all):.2f}.", ""]
    (HERE / "results" / "report_tracking_warm.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
