#!/usr/bin/env python3
"""Calibration variants that change T1 / T2 and gate durations (not the errors), for ESP and duration tracking.

The error variants (v8b_variants.py) leave T1, T2 and the gate durations untouched, so the calibration test can
only measure the tracking of F.  Here every variant keeps the errors and changes only:
  tjit     T1 and T2 of every qubit × lognormal(σ 0.3)
  tbad     2–3 qubits with T1 and T2 × 0.15–0.3
  djit     duration of every r / cz / measure instruction × lognormal(σ 0.2), per qubit and coupler
  czslow   3–5 couplers with the CZ 2–3× slower
  mslow    measurement 1.3–2× slower on every qubit (a slower readout calibration)
  mix      tjit + djit
Seeds 50 000 + 1 000·device + j: disjoint from the error variants.

Per (circuit, device): the circuit is compiled (level 2, seed 0, as the ESP labels) on the original device and on
every variant.  Step 1 (does the layout move?): initial layout and native counts of each variant compilation vs the
original.  Step 2 (labels): F, ESP, decay, duration, depth of the variant compilation, and ESP / duration of the
ORIGINAL compilation rescheduled on the variant (the cheap label if the layout never moves).  QExa20's decay is
stored raw: divide by esp_labels.T_FIX at analysis, as esp_labels.merge does.
    python esp_tvariants.py run --task-id T --n-tasks N [--limit 50]
    python esp_tvariants.py report          # -> results/esp_tvariants/report.md
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(HERE))
OUT = HERE / "results/esp_tvariants"
TRUTH = HERE.parent / "calibration_shift/results_k_v10/truth.json"
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
KINDS = ("tjit", "tbad", "djit", "czslow", "mslow", "mix")
N_PER_KIND = 2


def _qprops(be):
    t = be.target
    if t.qubit_properties is None:
        qp = be.qubit_properties
        t.qubit_properties = list(qp) if not callable(qp) else [qp(q) for q in range(t.num_qubits)]
    return t


def _scale_t(t, factors: dict) -> None:
    from qiskit.providers import QubitProperties
    new = []
    for q, p in enumerate(t.qubit_properties):
        f = factors.get(q, 1.0)
        new.append(p if p is None or f == 1.0 else
                   QubitProperties(t1=p.t1 * f if p.t1 else p.t1, t2=p.t2 * f if p.t2 else p.t2, frequency=p.frequency))
    t.qubit_properties = new


def _scale_dur(t, gate: str, factors: dict) -> None:
    for q, p in list(t[gate].items()):
        if p is None or p.duration is None or q not in factors:
            continue
        t.update_instruction_properties(gate, q, type(p)(duration=float(p.duration * factors[q]), error=p.error))


def make_tvar(backend, kind: str, j: int, dev_index: int):
    be = copy.deepcopy(backend)
    t = _qprops(be)
    rng = np.random.default_rng(50_000 + 1_000 * dev_index + 10 * KINDS.index(kind) + j)
    nq = t.num_qubits
    if kind in ("tjit", "mix"):
        _scale_t(t, {q: float(np.exp(rng.normal(0, 0.3))) for q in range(nq)})
    if kind == "tbad":
        bad = rng.choice(nq, size=int(rng.integers(2, 4)), replace=False)
        _scale_t(t, {int(q): float(rng.uniform(0.15, 0.3)) for q in bad})
    if kind in ("djit", "mix"):
        for g in ("r", "cz", "measure"):
            _scale_dur(t, g, {q: float(np.exp(rng.normal(0, 0.2))) for q, p in t[g].items() if p is not None})
    if kind == "czslow":
        cz = [q for q, p in t["cz"].items() if p is not None]
        pick = rng.choice(len(cz), size=int(rng.integers(3, 6)), replace=False)
        _scale_dur(t, "cz", {cz[i]: float(rng.uniform(2, 3)) for i in pick})
    if kind == "mslow":
        f = float(rng.uniform(1.3, 2.0))
        _scale_dur(t, "measure", {q: f for q, p in t["measure"].items() if p is not None})
    return be


def tvariants() -> dict:
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    base = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    out = {}
    for i, (d, be) in enumerate(base.items()):
        _qprops(be)
        out[f"{d}/orig"] = be
        for kind in KINDS:
            for j in range(N_PER_KIND):
                out[f"{d}/{kind}{j}"] = make_tvar(be, kind, j, i)
    return out


def _measures(tc, target) -> dict:
    from esp_labels import depth_2q, measures
    m = measures(tc, target)
    m["depth_2q"] = depth_2q(tc)
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in m.items()}


def run(args) -> None:
    from qiskit import qasm3
    from compile_check import DATA, compile_once
    from compile_layouts import _counts
    vs = tvariants()
    names = sorted(json.loads(TRUTH.read_text()))
    names = [n for n in names if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    if args.limit:
        names = names[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t0 = time.time()
    with out_file.open("a") as fh:
        for i, name in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name}
            try:
                qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
                for d in DEVICES:
                    orig_tc, _ = compile_once(qc, vs[f"{d}/orig"], 2, 0)
                    orig_lay = list(orig_tc.layout.initial_index_layout(filter_ancillas=True))
                    orig_cnt = _counts(orig_tc).tolist()
                    rec[f"{d}/orig"] = {"layout": orig_lay, "counts": orig_cnt, **_measures(orig_tc, vs[f"{d}/orig"].target)}
                    for key, be in vs.items():
                        if not key.startswith(d + "/") or key.endswith("/orig"):
                            continue
                        tc, _ = compile_once(qc, be, 2, 0)
                        lay = list(tc.layout.initial_index_layout(filter_ancillas=True))
                        r = {"same_layout": lay == orig_lay, "same_counts": _counts(tc).tolist() == orig_cnt,
                             **_measures(tc, be.target)}
                        rs = _measures(orig_tc, be.target)           # original compilation, rescheduled
                        r["ESP_resched"], r["duration_resched"], r["decay_resched"] = rs["ESP"], rs["duration"], rs["decay"]
                        rec[key] = r
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 10 == 0:
                print(f"[{time.time() - t0:6.0f} s] {i}/{len(names)} {name}", flush=True)


def report(args) -> None:
    from esp_labels import T_FIX
    recs = list({r["circuit"]: r for r in (json.loads(l) for f in sorted(OUT.glob("task0*.jsonl"))
                                            for l in f.read_text().splitlines() if l.strip())}.values())
    ok = [r for r in recs if "error" not in r]
    L = ["# T1 / T2 and gate-duration variants: does the compilation move, how much do ESP and duration change?", "",
         f"{len(ok)} circuits of the calibration test ({len(recs) - len(ok)} failed), 3 devices, "
         f"{len(KINDS)} kinds × {N_PER_KIND} variants, level 2, seed 0.", "",
         "| kind | same layout | same counts | median abs Δlog ESP | p90 abs Δlog ESP | median abs Δlog duration | "
         "p90 abs Δlog duration | rescheduled = recompiled (ESP, rel. diff < 1e-6) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for kind in KINDS:
        sl, sc, de, dd, eq = [], [], [], [], []
        for r in ok:
            for d in DEVICES:
                o = r[f"{d}/orig"]
                fix = T_FIX.get(d, 1.0)
                for j in range(N_PER_KIND):
                    v = r.get(f"{d}/{kind}{j}")
                    if not v or v.get("decay") is None or o.get("decay") is None:
                        continue
                    sl.append(v["same_layout"]); sc.append(v["same_counts"])
                    de.append(abs((v["decay"] - o["decay"]) / fix))          # Δlog ESP (F is unchanged)
                    dd.append(abs(math.log(v["duration"] / o["duration"])))
                    if v.get("ESP_resched") and v.get("ESP"):
                        eq.append(abs(v["ESP_resched"] / v["ESP"] - 1) < 1e-6)
        if not sl:
            continue
        p = lambda x, q: f"{np.percentile(x, q):.4f}"
        L.append(f"| {kind} | {np.mean(sl):.1%} | {np.mean(sc):.1%} | {p(de, 50)} | {p(de, 90)} | {p(dd, 50)} | {p(dd, 90)} | "
                 f"{np.mean(eq):.1%} |")
    L += ["", "Δlog ESP = change of the decay term (the errors, so F, are unchanged); QExa20's decay divided by "
          "esp_labels.T_FIX. If the layout never moves, the labels of any T1/T2/duration variant are the original "
          "compilation rescheduled: no recompilation needed."]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a) if a.mode == "run" else report(a)
