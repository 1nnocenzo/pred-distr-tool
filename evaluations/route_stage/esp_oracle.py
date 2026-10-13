#!/usr/bin/env python3
"""How well does the routed schedule give the ESP decay?  (oracle for the ESP part of the two-stage model)

MQT's ESP = F · Π_active qubits exp(−idle_q / min(T1, T2)_q), idle_q = duration − busy_q (ASAP schedule of the compiled
circuit).  Stage 1 gives the duration and the busy time per logical wire (gates + SWAPs), not the physical qubit each
wire sits on.  Oracle here, per routed case: Sabre's replayed schedule (sabre_labels: replay, busy) scaled to the final
duration, and 1 / min(T1, T2) averaged over (a) the whole device, (b) the best-T qubits as many as the circuit uses;
target = the label's decay (compile_check/results/esp_labels.pt: log ESP − log F).  A scale k is fitted on the control
families (every family except qaoa / vqe_* / qnn / synth) and applied to the rest.
    python esp_oracle.py      # -> results/esp_oracle.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "compile_check"))
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
VAR = ("vqe_su2", "vqe_real_amp", "vqe_two_local", "qnn")


def main():
    from v8b_variants import variants
    from esp_labels import T_FIX      # QExa20 T1/T2 unit fix: the labels' decay is divided by it
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    inv_t = {}
    for d, be in base.items():
        qp = be.qubit_properties
        qp = list(qp) if not callable(qp) else [qp(q) for q in range(be.target.num_qubits)]
        tm = np.array([min(p.t1, p.t2) for p in qp if p is not None and p.t1 and p.t2]) * 1e6   # µs
        inv_t[d] = np.sort(1.0 / tm) / T_FIX.get(d, 1.0)
    lab = torch.load(EVAL / "compile_check/results/esp_labels.pt", weights_only=False)
    recs = [json.loads(l) for f in sorted((EVAL / "compile_check/results/sabre_labels").glob("task*.jsonl"))
            for l in f.read_text().splitlines() if l.strip()]
    rows = []
    for c in recs:
        n = c["circuit"]
        if n not in lab or "rows" not in c:
            continue
        fam = n.split("/")[0]
        grp = "qaoa" if fam == "qaoa" else "variational" if fam in VAR else "control"
        for r in c["rows"]:
            if r["routing_free"] or "busy" not in r:
                continue
            di = DEVS.index(r["device"])
            dec = float(lab[n]["decay"][di])
            if not np.isfinite(dec):
                continue
            sc = r["final"] / max(r["replay"], 1e-9)
            busy = np.array(r["busy"]) * sc
            act = busy > 0
            n_act = int(act.sum())
            idle = float((r["final"] - busy[act]).clip(min=0).sum())
            it = inv_t[r["device"]]
            rows.append({"grp": grp, "dec": dec, "x_mean": idle * it.mean(), "x_best": idle * it[:max(n_act, 1)].mean(),
                         "x_dur": r["final"] * n_act * it.mean()})
    L = ["# ESP decay from the routed schedule (oracle)", "",
         "Routed cases; x = Σ idle (µs) × mean 1/min(T1,T2) (whole device, or the best-T qubits as many as used); "
         "decay ≈ −k·x with k fitted on control (median ratio). Error = |Δ log ESP| median [p90] (×100 ≈ %).", "",
         "| group | n | true decay median | k·x_mean: err | k·x_best: err | duration-only (n·dur): err |", "| --- | --- | --- | --- | --- | --- |"]
    ctrl = [r for r in rows if r["grp"] == "control"]
    ks = {k: float(np.median([-r["dec"] / max(r[k], 1e-12) for r in ctrl])) for k in ("x_mean", "x_best", "x_dur")}
    for g in ("control", "qaoa", "variational"):
        sel = [r for r in rows if r["grp"] == g]
        if not sel:
            continue
        dec = np.array([r["dec"] for r in sel])

        def e(k):
            p = -ks[k] * np.array([r[k] for r in sel])
            a = np.abs(p - dec)
            return f"{np.median(a):.3f} [{np.percentile(a, 90):.3f}]"
        L.append(f"| {g} | {len(sel)} | {np.median(dec):.3f} | {e('x_mean')} | {e('x_best')} | {e('x_dur')} |")
    L += ["", "k: " + ", ".join(f"{k} {v:.3f}" for k, v in ks.items())]
    (HERE / "results/esp_oracle.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
