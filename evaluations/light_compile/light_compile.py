#!/usr/bin/env python3
"""Duration / ESP / F per (circuit, device) without the full compiler's Sabre search (2026-10-16).

Rule, per device:
  switch     routing-free?  Cheap necessary conditions first (more qubits or couplers than the chip, a degree the chip
             lacks, an odd cycle on a bipartite chip -> routed), then VF2 subgraph search with a call limit (hitting
             it counts as routed: the safe side).  The same question as Qiskit's VF2Layout, the first layout pass of
             level 2, which tries up to 5e6 states.
  exact      routing-free or small (<= SMALL 2-qubit gates): Qiskit level 2 as the labels; exact, a few ms.
  light      otherwise: level 2 with the light Sabre (SabreLayout 1 layout trial, 1 iteration, 1 routing trial
             instead of 20 / 2 / 20) and VF2Layout's limit at CALL_LIMIT; F / ESP / duration of that compiled circuit
             are computed exactly (fastpath's MQT formulas), so the only error is the cheaper routing.
  model      (option) the route-stage model on the light Sabre (stage1.py --prior sabre-light, v28 checkpoint) for
             the duration; needs a trained checkpoint, not used by `eval` unless --ckpt is given.
    python light_compile.py eval --task-id 0 --n-tasks 4 --n 400     # reference, seed floor, light, switch, times
    python light_compile.py report                                   # -> results/report.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path[:0] = [str(EVAL / "fastpath"), str(EVAL / "compile_check"), str(EVAL / "generalization_v3"), str(EVAL / "route_stage")]
sys.path.append(str(EVAL.parent))
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
OUT = HERE / "results"
SMALL = 0                      # 2-qubit gates up to which a ROUTED circuit is compiled exactly (2026-10-16: 50 costs
                               # 45% more time for 8.4% -> 4.9% routed duration error on the rest; 0 = light for every routed case)
CALL_LIMIT = 2_000             # VF2 states of the switch (2026-10-16: max 2.9 ms; 2.6% of routing-free cases missed
                               # before the cheap checks)
SKIP = {"grover", "iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code"}


# ------------------------------------------------------------------------------------------------ devices
class Device:
    """One device: backend, calibration arrays, coupling graph facts and the two cached pass managers."""

    def __init__(self, name: str, backend):
        import rustworkx as rx
        import fastpath as fp
        from qiskit.transpiler import generate_preset_pass_manager
        self.name, self.backend = name, backend
        self.c = fp.calib(backend, name)
        self.g = fp.coupling_graph(self.c)
        self.bipartite = rx.is_bipartite(self.g)
        self.deg = sorted((self.g.degree(i) for i in self.g.node_indices()), reverse=True)
        t0 = time.perf_counter()
        self.pm_full = generate_preset_pass_manager(optimization_level=2, target=backend.target, seed_transpiler=0)
        self.t_pm = time.perf_counter() - t0
        self.pm_light = light_pm(backend.target, seed=0)
        t = backend.target
        self.dur_of = {}
        for op in ("r", "measure", "cz"):
            ds = {float(p.duration) for p in t[op].values() if p is not None and p.duration is not None}
            assert len(ds) == 1, f"{name}: {op} durations not uniform ({sorted(ds)[:3]}); physics() assumes one per op"
            self.dur_of[op] = ds.pop()

    def compile(self, qc, light: bool):
        return (self.pm_light if light else self.pm_full).run(qc)


def _passes(obj, seen=None):
    """Every pass object inside a (staged) pass manager / flow controller."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _passes(o, seen)
        return
    if hasattr(obj, "run") and type(obj).__module__.startswith("qiskit.transpiler.passes"):
        yield obj
    for attr in ("_tasks", "tasks", "stages", "_pass_manager"):
        v = getattr(obj, attr, None)
        if v is None:
            continue
        if attr == "stages":
            v = [getattr(obj, s, None) for s in v]
        yield from _passes(v, seen)


def light_pm(target, seed: int = 0, call_limit: int = 20_000):
    """Level 2 with the light Sabre and a bounded VF2Layout (no routing-free case lost at 20k on the dataset)."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.transpiler.passes import SabreLayout, VF2Layout
    pm = generate_preset_pass_manager(optimization_level=2, target=target, seed_transpiler=seed)
    n_sabre = 0
    for p in _passes(pm):
        if isinstance(p, SabreLayout):
            p.max_iterations, p.layout_trials, p.swap_trials = 1, 1, 1
            n_sabre += 1
        elif isinstance(p, VF2Layout):
            p.call_limit = (call_limit, p.call_limit[1]) if isinstance(p.call_limit, tuple) else call_limit
    assert n_sabre == 1, f"expected one SabreLayout in level 2, found {n_sabre}"
    return pm


def devices():
    from v8b_variants import variants
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    return [Device(d, base[d]) for d in DEVS]


# ------------------------------------------------------------------------------------------------ switch
def interaction_graph(qc):
    import rustworkx as rx
    import fastpath as fp
    _, c2 = fp.interaction_pairs(qc)
    nodes = sorted({q for k in c2 for q in k})
    idx = {q: i for i, q in enumerate(nodes)}
    g = rx.PyGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from_no_data([(idx[a], idx[b]) for a, b in c2])
    return g


def routing_free(ig, dev: Device, call_limit: int = CALL_LIMIT) -> tuple[bool, str]:
    """(routing-free?, how it was decided)."""
    import rustworkx as rx
    if ig.num_edges() == 0:
        return True, "no 2q gates"
    if ig.num_nodes() > dev.c.n or ig.num_edges() > dev.g.num_edges():
        return False, "size"
    deg = sorted((ig.degree(i) for i in ig.node_indices()), reverse=True)
    if any(a > b for a, b in zip(deg, dev.deg)):
        return False, "degree"
    if dev.bipartite and not rx.is_bipartite(ig):
        return False, "odd cycle"
    ok = rx.is_subgraph_isomorphic(dev.g, ig, id_order=False, induced=False, call_limit=call_limit)
    return bool(ok), "vf2"


# ----------------------------------------------------------------------------------------------- physics
def physics(tc, dev: Device) -> dict:
    """F, ESP (MQT's formulas, QExa20's T fix; equal to mqt.predictor.reward to 1e-12) and the ASAP duration
    (barriers are sync points, as Qiskit's estimate_duration) of a compiled circuit."""
    c, dur_of = dev.c, dev.dur_of
    idx = {q: i for i, q in enumerate(tc.qubits)}
    clock = np.zeros(tc.num_qubits)
    busy = np.zeros(tc.num_qubits)
    act = np.zeros(tc.num_qubits, bool)
    logf, n_cz = 0.0, 0
    for inst in tc.data:
        name = inst.operation.name
        qs = [idx[q] for q in inst.qubits]
        if name == "barrier":
            clock[qs] = clock[qs].max()
            continue
        if name in ("id", "delay"):
            continue
        if name == "r":
            logf += c.lr[qs[0]]; d = dur_of["r"]
        elif name == "measure":
            logf += c.lm[qs[0]]; d = dur_of["measure"]
        elif name == "cz":
            logf += c.lcz[qs[0], qs[1]]; d = dur_of["cz"]; n_cz += 1
        else:
            d = float(dev.backend.target[name][tuple(qs)].duration)
        t = clock[qs].max() + d
        clock[qs] = t
        busy[qs] += d
        act[qs] = True
    dur = float(clock.max()) if len(clock) else 0.0
    decay = -float(((dur - busy) * c.inv_t)[act].sum())
    return {"logF": float(logf), "logESP": float(logf + decay), "duration": dur, "cz": n_cz}


def n2q_logical(qc) -> int:
    return sum(1 for i in qc.data if i.operation.num_qubits == 2 and i.operation.name not in ("barrier",))


def predict(qc, devs, small: int = SMALL) -> list[dict]:
    """The rule of the module doc for one circuit on every device."""
    ig = interaction_graph(qc)
    n2 = n2q_logical(qc)
    out = []
    for dev in devs:
        t0 = time.perf_counter()
        rf, how = routing_free(ig, dev)
        light = not (rf or n2 <= small)
        tc = dev.compile(qc, light)
        out.append({**physics(tc, dev), "rf": rf, "how": how, "path": "light" if light else "exact",
                    "t": time.perf_counter() - t0})
    return out


# ------------------------------------------------------------------------------------------------- eval
def circuit_files(data_dir: Path):
    fs = [p for p in sorted(data_dir.glob("*/*.qasm")) if p.parent.name not in SKIP]
    random.Random(0).shuffle(fs)
    return fs


def load(p: Path):
    from qiskit import qasm3
    qc = qasm3.loads(p.read_text())
    if p.parent.name == "qaoa" and "measure" not in qc.count_ops():
        qc.measure_all()                              # data/circuits.zip has the old qaoa (no measurements)
    return qc


def eval_(args) -> None:
    """Per (circuit, device): reference compile (seed 0, as the labels), seed 1 (the compiler's own noise),
    the light compile, the switch, each with its time; one jsonl line per circuit."""
    devs = devices()
    full1 = {d.name: full_pm(d, 1) for d in devs}
    files = [p for p in circuit_files(args.data_dir) if zlib.crc32(p.stem.encode()) % args.n_tasks == args.task_id]
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"eval_task{args.task_id:02d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    n_ok = len(done)
    with out_file.open("a") as fh:
        for p in files:
            if n_ok >= args.n // args.n_tasks:
                break
            name = f"{p.parent.name}/{p.stem}"
            if name in done:
                continue
            try:
                qc = load(p)
                if qc.num_qubits > 20:
                    continue
                ig = interaction_graph(qc)
                n2 = n2q_logical(qc)
                rows = {}
                for dev in devs:
                    r = {}
                    t0 = time.perf_counter(); rf, how = routing_free(ig, dev); r["t_switch"] = time.perf_counter() - t0
                    t0 = time.perf_counter(); rf20k = routing_free(ig, dev, 20_000)[0]; r["t_switch20k"] = time.perf_counter() - t0
                    r.update(rf=rf, how=how, rf20k=rf20k)
                    dev.pm_full.run(qc); dev.pm_light.run(qc)              # warm caches (both timings fair)
                    t0 = time.perf_counter(); tc = dev.pm_full.run(qc); r["t_full"] = time.perf_counter() - t0
                    t0 = time.perf_counter(); r["full"] = physics(tc, dev); r["t_physics"] = time.perf_counter() - t0
                    r["vf2_found"] = _no_swap(tc)
                    r["seed1"] = physics(full1[dev.name].run(qc), dev)
                    t0 = time.perf_counter(); tl = dev.pm_light.run(qc); r["t_light"] = time.perf_counter() - t0
                    r["light"] = physics(tl, dev)
                    rows[dev.name] = r
                fh.write(json.dumps({"circuit": name, "n_qubits": qc.num_qubits, "n2q": n2, "size": qc.size(), "rows": rows}) + "\n")
                fh.flush()
                n_ok += 1
            except Exception as e:                                         # noqa: BLE001
                print(f"{name}: {type(e).__name__}: {e}"[:300], flush=True)
            if n_ok % 20 == 0:
                print(f"{n_ok} circuits", flush=True)


def full_pm(dev: Device, seed: int):
    from qiskit.transpiler import generate_preset_pass_manager
    return generate_preset_pass_manager(optimization_level=2, target=dev.backend.target, seed_transpiler=seed)


def _no_swap(tc) -> bool:
    """Routing-free compilation: no SWAP permuted the qubits (identity routing permutation)."""
    lay = tc.layout
    if lay is None or lay.final_layout is None:
        return True
    perm = lay.routing_permutation()
    return list(perm) == list(range(len(perm)))


# ----------------------------------------------------------------------------------------------- report
def report(args) -> None:
    recs = [json.loads(l) for f in sorted(OUT.glob("eval_task*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    rows = [(r["circuit"].split("/")[0], r["n2q"], r["n_qubits"], d, x) for r in recs for d, x in r["rows"].items()]
    L = []
    w = L.append

    def err(sel, key):
        """median / p90 relative duration error, median / p90 |Δ log ESP|, median |Δ log F| vs the reference."""
        if not sel:
            return None
        dd = np.array([abs(x[key]["duration"] / x["full"]["duration"] - 1) for *_, x in sel])
        de = np.array([abs(x[key]["logESP"] - x["full"]["logESP"]) for *_, x in sel])
        df = np.array([abs(x[key]["logF"] - x["full"]["logF"]) for *_, x in sel])
        bias = np.array([x[key]["duration"] / x["full"]["duration"] for *_, x in sel])
        rel = de / np.array([max(abs(x["full"]["logESP"]), 1e-9) for *_, x in sel])
        return (f"{np.median(dd):.1%} / {np.percentile(dd, 90):.1%} | {np.median(bias):.2f} | {np.median(de):.2f} / "
                f"{np.percentile(de, 90):.2f} | {np.median(rel):.1%} | {np.median(df):.3f}")

    def t_path(x, small):        # the pipeline: switch + one compilation + physics
        return x["t_switch"] + (x["t_full"] if (x["rf"] or x["n2q"] <= small) else x["t_light"]) + x["t_physics"]

    def t_ref(x):                # the reference: full compilation + physics
        return x["t_full"] + x["t_physics"]

    for (fam, n2, nq, d, x) in rows:
        x["n2q"] = n2
    w(f"# Pipeline: switch -> exact compile (routing-free / small) or light-Sabre compile (2026-10-16)\n")
    w(f"{len(recs)} circuits ({len(rows)} circuit-device cases; random sample of the dataset, <= 20 qubits, static circuits),"
      f" one CPU thread. Reference = Qiskit level 2 seed 0 (as the labels). Seed 1 = the compiler's own noise floor.\n")
    rt = [r for r in rows if not r[4]["rf20k"]]
    rfc = [r for r in rows if r[4]["rf20k"]]
    w("## Switch\n")
    agree = np.mean([x["rf20k"] == x["vf2_found"] for *_, x in rows])
    lost = sum(1 for *_, x in rows if x["rf20k"] and not x["rf"])
    ts = np.array([x["t_switch"] for *_, x in rows]) * 1e3
    t20 = np.array([x["t_switch20k"] for *_, x in rows]) * 1e3
    from collections import Counter
    hows = Counter(x["how"] for *_, x in rows)
    w(f"- agreement with the compiler's decision (VF2Layout found a layout, no SWAPs): {agree:.1%}")
    w(f"- with the cheap checks + limit {CALL_LIMIT}: median {np.median(ts):.3f} ms, p99 {np.percentile(ts, 99):.2f} ms, "
      f"max {ts.max():.2f} ms; routing-free cases lost vs limit 20k: {lost} of {len(rfc)}")
    w(f"- limit 20k without the checks would cost: max {t20.max():.2f} ms")
    w(f"- decided by: {dict(hows)}\n")
    w("## Errors vs the reference (seed 0)\n")
    le = np.array([x["full"]["logESP"] for *_, x in rt])
    w(f"Routed cases have tiny ESP (median log ESP {np.median(le):.1f}, i.e. ESP ~ {np.exp(np.median(le)):.0e}): a 5% change of the "
      f"duration moves log ESP by ~1, so ESP errors are given in log and relative to abs log ESP too.\n")
    w("| cases | what | duration err median / p90 | duration ratio | abs Δlog ESP median / p90 | abs Δlog ESP / abs log ESP | abs Δlog F median |")
    w("|---|---|---|---|---|---|---|")
    big = [r for r in rt if r[4]["full"]["logESP"] > np.log(1e-3)]
    for lbl, sel in (("routed", rt), ("routed, ESP > 1e-3 (as the model reports)", big), ("routing-free", rfc)):
        w(f"| {lbl} (n={len(sel)}) | light compile | {err(sel, 'light')} |")
        w(f"| {lbl} | compiler seed 1 (noise floor) | {err(sel, 'seed1')} |")
    w("\nFor comparison, the models on routed cases (vs the labels, ESP error only where ESP > 1e-3): v8kdi10 + v19i "
      "duration 12.5% (LOFO qaoa), 12.6% (LOGO variational), 4.5% (control); abs Δlog ESP 0.59 / 0.65 / 0.30 "
      "(esp/results/report_esp_routed_v10_v19i.md); route stage v27 duration 8.8% / 5.4% / 3.2%.")
    w("\nRouted, per family (light compile; seed-1 floor = the compiler's own spread):\n")
    w("| family | n | duration err median / p90 | ratio | abs Δlog ESP median / p90 | rel. | abs Δlog F | floor duration | floor abs Δlog ESP |")
    w("|---|---|---|---|---|---|---|---|---|")
    fams = Counter(r[0] for r in rt)
    for f, n in fams.most_common():
        sel = [r for r in rt if r[0] == f]
        if n < 5:
            continue
        fl = np.median([abs(x["seed1"]["duration"] / x["full"]["duration"] - 1) for *_, x in sel])
        fe_ = np.median([abs(x["seed1"]["logESP"] - x["full"]["logESP"]) for *_, x in sel])
        w(f"| {f} | {n} | {err(sel, 'light')} | {fl:.1%} | {fe_:.2f} |")
    w("\n## Time per circuit (3 devices, sum)\n")
    by_c = {}
    for (fam, n2, nq, d, x) in rows:
        by_c.setdefault((fam, n2, nq), []).append(x)
    w("| SMALL (exact up to this many 2q gates) | pipeline total | full compile total | saving | routed-case duration err median | share compiled exactly |")
    w("|---|---|---|---|---|---|")
    T_full = sum(t_ref(x) for *_, x in rows)
    for small in (0, 20, 50, 100, 200):
        T = sum(t_path(x, small) for *_, x in rows)
        ex = [x for *_, x in rows if x["rf"] or x["n2q"] <= small]
        rr = [x for *_, x in rows if not (x["rf"] or x["n2q"] <= small)]
        e = np.median([abs(x["light"]["duration"] / x["full"]["duration"] - 1) for x in rr]) if rr else 0
        w(f"| {small} | {T:.1f} s | {T_full:.1f} s | {1 - T / T_full:.0%} | {e:.1%} | {len(ex) / len(rows):.0%} |")
    w("\nPer circuit size (sum over the 3 devices, median; SMALL = %d):\n" % SMALL)
    w("| 2q gates | n | full compile | pipeline | ratio | of which switch | light compile alone |")
    w("|---|---|---|---|---|---|---|")
    for lo, hi in ((0, 20), (20, 50), (50, 200), (200, 600), (600, 10 ** 9)):
        cs = [xs for (f, n2, nq), xs in by_c.items() if lo <= n2 < hi]
        if not cs:
            continue
        tf = np.array([sum(t_ref(x) for x in xs) for xs in cs]) * 1e3
        tp = np.array([sum(t_path(x, SMALL) for x in xs) for xs in cs]) * 1e3
        tsw = np.array([sum(x["t_switch"] for x in xs) for xs in cs]) * 1e3
        tl = np.array([sum(x["t_light"] for x in xs) for xs in cs]) * 1e3
        w(f"| {lo}-{hi if hi < 10 ** 9 else '…'} | {len(cs)} | {np.median(tf):.1f} ms | {np.median(tp):.1f} ms | "
          f"{np.median(tp / tf):.2f} | {np.median(tsw):.2f} ms | {np.median(tl):.1f} ms |")
    w("\nPer family (routed families, sum over the 3 devices, median):\n")
    w("| family | n | full compile | pipeline | ratio |")
    w("|---|---|---|---|---|")
    for f in ("qaoa", "randomcircuit", "qft", "qftentangled", "qnn", "vqe_su2", "vqe_real_amp", "vqe_two_local", "ae", "qpeexact", "graphstate", "ghz", "dj"):
        cs = [xs for (ff, n2, nq), xs in by_c.items() if ff == f]
        if len(cs) < 3:
            continue
        tf = np.array([sum(t_ref(x) for x in xs) for xs in cs]) * 1e3
        tp = np.array([sum(t_path(x, SMALL) for x in xs) for xs in cs]) * 1e3
        w(f"| {f} | {len(cs)} | {np.median(tf):.1f} ms | {np.median(tp):.1f} ms | {np.median(tp / tf):.2f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("eval", "report"))
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--data-dir", type=Path, default=EVAL.parent / "data/benchmark_dataset_30k")
    a = ap.parse_args()
    {"eval": eval_, "report": report}[a.cmd](a)
