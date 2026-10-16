#!/usr/bin/env python3
"""Formula check: how far are the model's duration and ESP formulas from the labels when every learned input is
replaced by the compiler's truth (no network involved)?

Duration (µs), vs the label (ASAP of the compiled circuit):
  D0 logical       schedule of the gate graph, decomposition-table gate times, no SWAPs
  D1 +SWAPs        + Sabre's SWAPs before each gate on each operand wire (the route stage's rate target)
  D2 +waits        + the teacher-forced extra waits (the route stage's wait target) = the stage-1 oracle
  D3 replay        Sabre's routed circuit itself scheduled with table times (what D2 approximates)
  "after c2 = const" divides by the group's median ratio: what a constant stage 2 would leave (c2 is learned per circuit)
Decay = log ESP − log F (nats = error of log ESP when F is exact), vs the label:
  E0 MQT exact     −Σ_active physical p (duration − busy_p) / min(T1,T2)_p, from the compiled circuit (checks the
                   label definition: the idle time is counted once)
  E1 formula       the model's formula with true inputs: idle of every LOGICAL qubit on the routed wires (gate times +
                   true SWAPs) up to the label duration, T of the physical qubit of the compiler's initial layout;
                   −(α · Σ_q idle_q / T_q + β · duration · mean 1/T): pure (α = 1, β = 0) and with one global α, β
                   fitted in log space (the head predicts e^a, softplus(b) per case, so it can do better than that)
  E2 no placement  E1 with every qubit at the device's mean 1/T (what the hard placement adds)
  E3 duration      E1 with the duration of D2 (no stage 2) and with the label duration +5% (sensitivity)
  E4 counts        the model's formula since 2026-10-16: −[N_act · duration − busy] · mean 1/T, busy = r · 42 ns +
                   2 · cz · 130 ns + measure · 15 µs from the true counts (= Σ_p busy_p exactly), T over the qubits
                   of the initial layout; N_act = true active physical qubits, or the logical ones (no extra qubits)
    python -m esptime.oracle            # work/dataset.pt + work/labels.pt -> work/oracle/report.md, rows.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .config import DEVICES, T1, TCZ, TM, T_SWAP, VARIATIONAL, WORK
from .dataset import family, schedule_np

ESP_MIN = np.log(1e-3)


def inv_t(raw: dict) -> np.ndarray:
    """1/min(T1, T2) per device-graph node, 1/µs (nodes[:, 2:4] = log10 T1, T2 in s)."""
    n = np.array(raw["nodes"], float)
    return 10.0 ** (-np.minimum(n[:, 2], n[:, 3])) * 1e-6


def group(name: str) -> str:
    f = family(name)
    return "qaoa" if f == "qaoa" else "variational" if f in VARIATIONAL else "synth" if f == "synth" else "other"


def rows_of(it, lab, invts) -> list[dict]:
    gq = it.gate_qubits.numpy()
    wires = [[int(q) for q in row if q >= 0] for row in gq]
    t = it.t.numpy()
    logical = schedule_np(wires, t)
    out = []
    for d, dev in enumerate(DEVICES):
        if not bool(it.ok[0, d]):
            continue
        rec = lab["calib"]["orig"][dev]
        rf, routed = bool(it.rf[0, d]), bool(it.routed[0, d])
        s = it.s[:, d].numpy()
        w = it.wait[:, d].numpy()
        dur = float(it.dur[0, d])
        r = {"name": it.name, "group": group(it.name), "d": d, "rf": rf, "routed": routed, "dur": dur,
             "D0": logical, "decay": float(it.decay[0, d]), "logesp": float(it.logF[0, d] + it.decay[0, d])}
        if routed:
            r.update(D1=schedule_np(wires, t, s), D2=schedule_np(wires, t, s, w), D3=float(it.replay[0, d]))
        it_d = invts[d]
        # E0: MQT's own definition on the physical qubits of the compiled circuit
        busy_p = np.array([[n, b] for n, b in rec["busy"] if n >= 0])
        n_r, n_m, n_cz = rec["counts"]
        busy_tot = n_r * T1 + n_m * TM + 2 * n_cz * TCZ
        lay0 = it.layout[:, d].numpy()
        inv_lay = float(it_d[lay0[lay0 >= 0]].mean())
        n_log_act = len({q for w in wires for q in w})
        r["E4_trueN"] = -(len(busy_p) * dur - busy_tot) * inv_lay
        r["E4_logN"] = -(n_log_act * dur - busy_tot) * inv_lay
        r["E4_trueN_p5"] = -(len(busy_p) * 1.05 * dur - busy_tot) * inv_lay
        r["E0_own"] = float(((dur - busy_p[:, 1]).clip(min=0) * it_d[busy_p[:, 0].astype(int)]).sum())
        r["idle_phys"], r["busy_phys"], r["n_phys"] = float((dur - busy_p[:, 1]).clip(min=0).sum()), float(busy_p[:, 1].sum()), len(busy_p)
        # E1: logical wires on the routed schedule, T of the initial layout
        busy_q = np.zeros(int(it.n_qubits))
        for i, qs in enumerate(wires):
            for k, q in enumerate(qs):
                busy_q[q] += t[i] + (s[i, k] * T_SWAP if routed else 0.0)
        act = busy_q > 0
        r["idle_log"], r["busy_log"], r["n_log"] = float((dur - busy_q[act]).clip(min=0).sum()), float(busy_q[act].sum()), int(act.sum())
        lay = it.layout[:, d].numpy()
        tq = np.where(lay >= 0, it_d[lay.clip(min=0)], it_d.mean())
        r["inv_m"] = float(tq[act].mean())
        r["inv_dev"] = float(it_d.mean())
        for key, du in (("", dur), ("_D2", r.get("D2", logical)), ("_p5", 1.05 * dur)):
            idle = (du - busy_q).clip(min=0) * act
            r["own" + key] = float((idle * tq).sum())
            r["own_nopl" + key] = float(idle.sum() * it_d.mean())
            r["dur" + key] = du
        out.append(r)
    return out


def err(p, t) -> dict:
    e = np.abs(p / t - 1)
    return {"median": round(float(np.median(e)), 4), "p90": round(float(np.percentile(e, 90)), 4),
            "bias": round(float(np.median(p / t)), 4), "n": int(len(e))}


def fit(rows, own_key, dur_key, inv_key):
    """α, β > 0 of −decay ≈ α · own + β · duration · mean 1/T, fitted in log space (the model's loss is relative:
    Huber on log(−decay + 0.01))."""
    from scipy.optimize import minimize
    X = np.array([[r[own_key], r[dur_key] * r[inv_key]] for r in rows])
    y = np.log(-np.array([r["decay"] for r in rows]) + 1e-2)
    obj = lambda c: float(np.mean((np.log(X @ np.exp(c) + 1e-2) - y) ** 2))
    coef = np.exp(minimize(obj, np.array([0.0, -4.0]), method="Nelder-Mead").x)
    return coef, -(X @ coef)


def dstat(pred, rows) -> dict:
    lab = np.array([r["decay"] for r in rows])
    big = np.array([r["logesp"] > ESP_MIN for r in rows])
    e = np.abs(pred - lab)
    return {"abs_median_esp_gt_1e-3": round(float(np.median(e[big])), 4) if big.any() else None,
            "abs_p90_esp_gt_1e-3": round(float(np.percentile(e[big], 90)), 4) if big.any() else None,
            "rel_median": round(float(np.median(e / np.maximum(np.abs(lab), 1e-9))), 4), "n": int(len(e)),
            "n_esp_gt_1e-3": int(big.sum())}


def main():
    blob = torch.load(WORK / "dataset.pt", weights_only=False)
    labels = torch.load(WORK / "labels.pt", weights_only=False)
    invts = [inv_t(blob["devices"]["orig"][d]) for d in DEVICES]
    rows = [r for it in blob["items"] for r in rows_of(it, labels[it.name], invts)]
    res: dict = {"n_circuits": len(blob["items"]), "n_cases": len(rows), "duration": {}, "decay": {}}
    rt, rf_ = [r for r in rows if r["routed"]], [r for r in rows if r["rf"]]
    dl = lambda sel, k: np.array([r[k] for r in sel])
    for grp, sel, keys in (("routing-free", rf_, ("D0",)), ("routed", rt, ("D0", "D1", "D2", "D3"))):
        if not sel:
            continue
        lab = dl(sel, "dur")
        g = {k: err(dl(sel, k), lab) for k in keys}
        for k in keys:
            ratio = dl(sel, k) / lab
            g[k]["after_c2_const"] = round(float(np.median(np.abs(ratio / np.median(ratio) - 1))), 4)
        if grp == "routed":
            g["D2_vs_replay"] = err(dl(sel, "D2"), dl(sel, "D3"))
            g["by_group_D3"] = {gg: err(dl(ss, "D3"), dl(ss, "dur")) for gg in ("qaoa", "variational", "other", "synth")
                                if (ss := [r for r in sel if r["group"] == gg])}
        res["duration"][grp] = g
    own0 = dl(rows, "E0_own")
    res["decay"]["E0 MQT exact"] = dstat(-own0, rows)
    for grp, sel in (("routing-free", rf_), ("routed", rt)):
        if sel:
            q = lambda k: np.array([r[k] for r in sel])
            res["idle " + grp] = {"logical/physical idle, median": round(float(np.median(q("idle_log") / np.maximum(q("idle_phys"), 1e-9))), 3),
                                  "p10": round(float(np.percentile(q("idle_log") / np.maximum(q("idle_phys"), 1e-9), 10)), 3),
                                  "logical/physical busy, median": round(float(np.median(q("busy_log") / q("busy_phys"))), 3),
                                  "share with more active physical than logical qubits": round(float(np.mean(q("n_phys") > q("n_log"))), 3)}
    for grp, sel in (("all", rows), ("routing-free", rf_), ("routed", rt)):
        if len(sel) < 3:
            continue
        g = {}
        g["E1 pure formula (alpha=1, beta=0)"] = dstat(-dl(sel, "own"), sel)
        coef, pred = fit(sel, "own", "dur", "inv_m")
        g["E1 fitted"] = {**dstat(pred, sel), "alpha": round(float(coef[0]), 3), "beta": round(float(coef[1]), 3)}
        _, pred = fit(sel, "own_nopl", "dur", "inv_dev")
        g["E2 no placement, fitted"] = dstat(pred, sel)
        X = lambda ok, dk: np.array([[r[ok], r[dk] * r["inv_m"]] for r in sel])
        g["E3 duration of D2 (no c2)"] = dstat(-(X("own_D2", "dur_D2") @ coef), sel)
        g["E3 duration +5%"] = dstat(-(X("own_p5", "dur_p5") @ coef), sel)
        g["E4 counts, true N_act"] = dstat(dl(sel, "E4_trueN"), sel)
        g["E4 counts, N_act = logical"] = dstat(dl(sel, "E4_logN"), sel)
        g["E4 counts, true N_act, duration +5%"] = dstat(dl(sel, "E4_trueN_p5"), sel)
        res["decay"][grp] = g
    res["decay"]["by_group E1 fitted (all)"] = {}
    coef, _ = fit(rows, "own", "dur", "inv_m")
    for gg in ("qaoa", "variational", "other", "synth"):
        ss = [r for r in rows if r["group"] == gg]
        if ss:
            X = np.array([[r["own"], r["dur"] * r["inv_m"]] for r in ss])
            res["decay"]["by_group E1 fitted (all)"][gg] = dstat(-(X @ coef), ss)
    out = WORK / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1))
    (out / "rows.json").write_text(json.dumps(rows, default=float))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
