#!/usr/bin/env python3
"""Where do duration / depth come from?  Labels + the stored compilations, no model.

On a sample of circuits per family (stored seed-0 QPY, original calibration), per device, the TRUE per-qubit
structure of the compiled circuit: gate time t_q (1q 42 ns, CZ 130 ns), measurements m_q, operations o_q.
  1. per family: duration / depth spread (std of the logs: why R² of log duration can be very negative on a family
     whose durations hardly vary), share with measurements, true duration / (15 µs · max m_q + max t_q) and
     depth / max o_q (are the analytic head's ratios family-independent?);
  2. ceiling of the analytic head: its rule fitted with the TRUE structure on all families but the held-out ones
     (qaoa, qnn, variational group), scored on those -> what the head could reach with exact counts.
    python esp_diag.py [--per-family 60]      # -> results/report_esp_diag.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
QPY = EVAL / "compile_check/results/layouts_all/qpy"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
DYNAMIC = {"iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code", "grover"}
VARIATIONAL = {"qnn", "vqe_real_amp", "vqe_su2", "vqe_two_local"}
T1Q, TCZ, TM = 0.042, 0.130, 15.0


def structure(qc):
    """(t_q, m_q, o_q) arrays over the active qubits of a compiled circuit."""
    n = qc.num_qubits
    t, m, o = np.zeros(n), np.zeros(n), np.zeros(n)
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        for q in qs:
            o[q] += 1
            if name == "measure":
                m[q] += 1
            elif name == "cz":
                t[q] += TCZ
            else:
                t[q] += T1Q
    act = o > 0
    return t[act], m[act], o[act]


def r2(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(1 - ((p - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=60)
    args = ap.parse_args()
    import torch
    from qiskit import qpy
    lab = torch.load(EVAL / "compile_check/results/esp_labels.pt", weights_only=False)
    rows = []
    rng = random.Random(0)
    for fam_dir in sorted(p for p in QPY.iterdir() if p.is_dir() and p.name not in DYNAMIC):
        files = sorted(fam_dir.glob("*.qpy"))
        rng.shuffle(files)
        for f in files[: args.per_family]:
            name = f"{fam_dir.name}/{f.stem}"
            if name not in lab:
                continue
            with f.open("rb") as h:
                circs = qpy.load(h)
            for d, qc in enumerate(circs):
                t, m, o = structure(qc)
                if len(t) == 0:
                    continue
                rows.append({"family": fam_dir.name, "circuit": name, "dev": DEVS[d],
                             "dur": float(lab[name]["duration"][d]) * 1e6, "depth": float(lab[name]["depth"][d]),
                             "tmax": t.max(), "mmax": m.max(), "omax": o.max(), "active": len(t),
                             "n_meas": m.sum(), "busy_max": (t + TM * m).max()})
        print(fam_dir.name, len(rows), flush=True)
    fam = np.array([r["family"] for r in rows])
    X = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k not in ("family", "circuit", "dev")}
    base = TM * X["mmax"] + X["tmax"]
    ratio_d, ratio_p = X["dur"] / np.maximum(base, 1e-3), X["depth"] / np.maximum(X["omax"], 1)
    L = ["# Duration and depth: what the labels say (stored compilations, no model)", "",
         f"{len(rows)} (circuit, device) pairs, up to {args.per_family} circuits per family.", "",
         "| family | n | with measurements | median duration µs | std log duration | median depth | std log depth | "
         "duration / (15 µs·max m + max t) median [p10, p90] | depth / max ops median [p10, p90] |",
         "| --- |" + " --- |" * 8]
    for f_ in sorted(set(fam)):
        s = fam == f_
        q = lambda a: f"{np.median(a):.2f} [{np.percentile(a, 10):.2f}, {np.percentile(a, 90):.2f}]"
        L.append(f"| {f_} | {s.sum()} | {np.mean(X['mmax'][s] > 0):.2f} | {np.median(X['dur'][s]):.1f} | "
                 f"{np.log(X['dur'][s]).std():.2f} | {np.median(X['depth'][s]):.0f} | {np.log(X['depth'][s]).std():.2f} | "
                 f"{q(ratio_d[s])} | {q(ratio_p[s])} |")
    # 2. ceiling: per held-out split, fit log ratio ~ linear(features) on the other families
    feats = np.stack([np.log1p(X["tmax"]), np.log1p(X["mmax"]), np.log1p(X["omax"]), np.log1p(X["active"]),
                      np.log1p(X["n_meas"]), np.ones(len(rows))], 1)
    L += ["", "## Ceiling of the analytic rule with the TRUE structure (linear fit of the log ratios on the other families)", "",
          "| held out | n | R² log duration | duration error (median) | R² log depth | depth error (median) | "
          "R² log duration, ratio fixed at train median | R² log depth, ratio fixed |", "| --- |" + " --- |" * 7]
    for name, test in (("qaoa", {"qaoa"}), ("qnn", {"qnn"}), ("variational group", VARIATIONAL)):
        te = np.isin(fam, list(test)); tr = ~te
        out = [name, str(te.sum())]
        fixed = []
        for y_ratio, lo, tgt in ((ratio_d, base, X["dur"]), (ratio_p, np.maximum(X["omax"], 1), X["depth"])):
            w, *_ = np.linalg.lstsq(feats[tr], np.log(y_ratio[tr]), rcond=None)
            pred = np.log(lo[te]) + feats[te] @ w
            out += [f"{r2(pred, np.log(tgt[te])):.3f}", f"{np.median(np.abs(np.exp(pred - np.log(tgt[te])) - 1)):.3f}"]
            fixed.append(f"{r2(np.log(lo[te]) + np.median(np.log(y_ratio[tr])), np.log(tgt[te])):.3f}")
        L.append("| " + " | ".join(out + fixed) + " |")
    (HERE / "results/report_esp_diag.md").write_text("\n".join(L))
    (HERE / "results/esp_diag_rows.json").write_text(json.dumps(rows, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
