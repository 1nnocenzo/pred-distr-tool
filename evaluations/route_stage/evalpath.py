#!/usr/bin/env python3
"""End-to-end test of the v28 path on the dataset's test splits: duration, ESP, F and time per (circuit, device).

The path at prediction time, per circuit and device (no compilation for routed cases):
  switch      light_compile.routing_free (size / degree / odd-cycle checks + VF2, limit 2000)
  routing-free  -> Qiskit level 2, exact (a few ms; F / ESP / duration of the result, MQT's formulas)
  routed      -> light Sabre (sabre_light.light_case: placement + SWAPs on physical qubits)
                 -> the v28 model (stage1.py --prior sabre-light) on that input: SWAPs and waits of the full Sabre
                    -> duration (schedule of Poisson-sampled SWAPs, median of MC, × stage 2)
                 -> F from the F model, as before (v8kdc10's predictions.json of the same split and seed, --f-pred; the
                    F the hybrid / ESP studies used)
                 -> ESP = F · Π_active exp(−(duration − busy_q) / min(T1, T2)_q): the ESP head's formula (gsv3/model.py,
                    v19 formula head) with the v28 duration, on the light Sabre's physical qubits and busy times (its
                    gates converted to native r / cz / measure with esp/results/decomp_table.json, plus the SWAPs the
                    model adds or removes vs the light Sabre, their time spread over the active qubits)
                 Without --f-pred: F from that table (biased low: it counts r gates before the optimisation merges them
                 and ignores VF2PostLayout; 2026-10-16 smoke: log F −0.5 to −3 vs the compile), reported as such.
Baselines on the same routed cases: "light table" (the light Sabre alone: its replayed schedule, table F, no model) and
"light compile" (level 2 with the light Sabre, exact physics; evaluations/light_compile).  Truth: the labels
(compile_check/results/esp_labels.pt: F, decay, duration of the level-2 compilation, seed 0).
Times (one thread): every step of the path, including the encoding and the greedy-router hops the model reads,
against the full level-2 compile (and the light compile) of the same case.
    python evalpath.py run --split qaoa --ckpt results/v28_qaoa_s5/model.pt --task-id 0 --n-tasks 4
    python evalpath.py report                     # every results/evalpath/*.jsonl -> results/evalpath/report.md
    python evalpath.py smoke --data-dir DIR       # untrained weights, live inputs only, a few real circuits (no dataset)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path[:0] = [str(HERE), str(EVAL / "light_compile"), str(EVAL / "fastpath"), str(EVAL / "compile_check")]
import stage1 as S  # noqa: E402
import prep as P  # noqa: E402
import sabre_light as SL  # noqa: E402
import light_compile as LC  # noqa: E402
from gsv3 import fast_encoding as fe  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

OUT = HERE / "results/evalpath"
SWAP_R, SWAP_CZ = 4, 3                      # a routing SWAP in native gates (3 cz + 2 r per qubit, T_SWAP = 3 TCZ + 2 T1)


def tables():
    tab = json.loads((EVAL / "esp/results/decomp_table.json").read_text())
    U = list(fe.UNIQUE)
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    tab.update({"r": {"r": 1, "cz": 0}, "cz": {"r": 0, "cz": 1}, "swap": {"r": SWAP_R, "cz": SWAP_CZ}})
    return tab, tr, tc


# --------------------------------------------------------------------------------------------- live inputs
def live_item(g, light: list, tr, tc, chips):
    """The v28 model's input built at prediction time exactly as stage1.build (greedy-router hops, light Sabre SWAPs per
    slot and its teacher-forced waits); ``light`` = [light_case(...) per device].  Returns (item, seconds of the
    feature steps)."""
    t0 = time.perf_counter()
    gq = g.gate_qubits.numpy()
    N, n_q = len(gq), int(g.n_qubits)
    wires, t, meas = P.node_times(g, tr, tc)
    D = len(S.DEVS)
    hops = np.zeros((D, N), np.float32)
    s = np.zeros((D, N, 3), np.float32)
    pw = np.zeros((N, D), np.float32)
    for d in range(D):
        _, hops[d], _ = P.prior_route(wires, t, meas, n_q, chips[d])
        lab, _, _, _, tn = light[d]
        for (node, slot), v in lab.items():
            s[d, node, slot] = v
        if tn:
            pw[:, d] = P.wait_targets(wires, t, s[d], {k: tuple(v) for k, v in tn.items()})
    prev, lvl = S.wire_prev(gq)
    dev_f = torch.cat([torch.log1p(torch.from_numpy(hops).clamp(min=0)).t(),
                       torch.log1p(torch.from_numpy(s)).permute(1, 0, 2).reshape(-1, 9),
                       torch.log1p(torch.from_numpy(pw).clamp(min=0) / S.T_SWAP)], 1).float()
    item = Data(prev=torch.from_numpy(prev), lvl=torch.from_numpy(lvl), pw=torch.from_numpy(pw), x=g.x.float(),
                edge_index=g.edge_index, gate_qubits=g.gate_qubits, n_qubits=g.n_qubits,
                wf=torch.from_numpy(S.wire_feats(gq, n_q)), dev_f=dev_f, t=torch.from_numpy(S.node_t(g, tr, tc)), num_nodes=N)
    return item, time.perf_counter() - t0, s


def model_out(model, item):
    """rate (N, D, 3), wait (N, D) µs, c2 (D,) for one circuit, CPU, no grad."""
    b = item.clone()
    b.batch = torch.zeros(item.num_nodes, dtype=torch.long)
    b.ptr = torch.tensor([0, item.num_nodes])
    with torch.no_grad():
        rate, wait, c2 = model(b)
    return rate.numpy(), wait.numpy() * S.T_SWAP, c2.view(-1).numpy()


def model_duration(item, rate, wait, c2, d: int, mc: int, rng) -> float:
    """Routed duration (µs) on device d: median over mc Poisson samples of the SWAPs (as stage1.evaluate's pred_mc)."""
    gq = item.gate_qubits.numpy(); t = item.t.numpy()
    meas = item.x[:, S.GATE2IDX_MEASURE].numpy() > 0.5
    elig = ((gq >= 0).sum(1) >= 2) | meas
    slot = (gq >= 0) & elig[:, None]
    r = rate[:, d] * slot
    if mc:
        p1 = float(np.median(S.schedule_k(gq, t, rng.poisson(np.repeat(r[..., None], mc, -1)), wait[:, d], elig)))
    else:
        p1 = S.schedule_np(gq, t, r, wait[:, d], elig)
    return p1 * math.exp(float(c2[d])), float(r.sum())


def table_physics(routed, dev, tab) -> dict:
    """log F and per-physical-qubit busy time (µs) of the light Sabre's routed circuit, gates -> native counts by the
    decomposition table (r split over the operands, cz on the gate's coupler, SWAP = 3 cz + 4 r)."""
    c = dev.c
    idx = {q: i for i, q in enumerate(routed.qubits)}
    busy = np.zeros(c.n)
    act = np.zeros(c.n, bool)
    logf, miss = 0.0, Counter()
    for inst in routed.data:
        name = inst.operation.name
        if name in ("barrier", "id", "delay"):
            continue
        qs = [idx[q] for q in inst.qubits]
        act[qs] = True
        if name == "measure":
            logf += c.lm[qs[0]]; busy[qs[0]] += S.TM
            continue
        e = tab.get(name)
        if e is None:                                    # e.g. 2q "unitary" left by the init stage: generic KAK
            miss[name] += 1
            e = {"r": 2, "cz": 0} if len(qs) == 1 else {"r": 8, "cz": 3}
        nr, ncz = float(e["r"]), float(e["cz"])
        for q in qs:
            logf += nr / len(qs) * np.nan_to_num(c.lr[q])
            busy[q] += nr / len(qs) * S.T1 + ncz * S.TCZ
        if ncz and len(qs) == 2:
            logf += ncz * np.nan_to_num(c.lcz[qs[0], qs[1]])
        elif ncz:
            logf += ncz * float(np.nanmean(c.lcz))
    return {"logF": float(logf), "busy": busy, "active": act, "missing": dict(miss)}


def esp_from(logf: float, busy: np.ndarray, act: np.ndarray, dur_us: float, dev) -> float:
    idle = np.clip(dur_us - busy, 0, None) * 1e-6
    return float(logf - (idle * dev.c.inv_t)[act].sum())


def swap_logf(dev) -> float:
    c = dev.c
    return SWAP_CZ * float(np.nanmean(c.lcz[np.isfinite(c.lcz)])) + SWAP_R * float(np.nanmean(c.lr))


# ------------------------------------------------------------------------------------------------- one case
def predict_circuit(name, qc, g, model, devs, tab, tr, tc, chips, mc: int, rng, baselines: bool, time_full: bool,
                    fpred=None):
    """Every step for one circuit on every device; returns per-device rows with values and times."""
    T = {}
    t0 = time.perf_counter(); qb = fe.to_basis(qc); g_live = fe.to_data(qc); T["encode"] = time.perf_counter() - t0
    g = g_live if g is None else g                    # the dataset's graph when given (the model was trained on it)
    nodes = [(i.operation.name, [qb.find_bit(q).index for q in i.qubits]) for i in qb.data if i.operation.name != "barrier"]
    if len(nodes) != g.num_nodes:
        raise ValueError(f"node count {len(nodes)} != graph {g.num_nodes}")
    ig = LC.interaction_graph(qc)
    rows = {}
    light, keep, t_light = [], [], []
    rf_all = []
    for dev in devs:
        t0 = time.perf_counter(); rf, how = LC.routing_free(ig, dev); t_sw = time.perf_counter() - t0
        rf_all.append((rf, how, t_sw))
        k = {}
        t0 = time.perf_counter()
        light.append(SL.light_case(qb, nodes, qb.num_qubits, dev.backend, k) if not rf else ({}, 0, True, 0.0, {}))
        t_light.append(time.perf_counter() - t0)
        keep.append(k)
    routed_any = any(not r[0] for r in rf_all)
    item = rate = None
    t_feat = t_model = 0.0
    if routed_any:
        item, t_feat, s_light = live_item(g, light, tr, tc, chips)
        t0 = time.perf_counter(); rate, wait, c2 = model_out(model, item); t_model = time.perf_counter() - t0
    for di, dev in enumerate(devs):
        rf, how, t_sw = rf_all[di]
        # the encoding is only needed when some device is routed (the model reads the graph)
        r = {"rf": rf, "how": how, "t_switch": t_sw, "t_encode": T["encode"] / len(devs) if routed_any else 0.0}
        if rf:                                                     # exact path
            t0 = time.perf_counter(); tc_ = dev.pm_full.run(qc); ph = LC.physics(tc_, dev)
            r.update(path="exact", t_path=time.perf_counter() - t0, pred={"logF": ph["logF"], "logESP": ph["logESP"],
                                                                          "duration": ph["duration"] * 1e6})
        else:
            t0 = time.perf_counter()
            dur, sw_model = model_duration(item, rate, wait, c2, di, mc, rng)
            t_sched = time.perf_counter() - t0
            t0 = time.perf_counter()
            tp = table_physics(keep[di]["routed"], dev, tab)
            sw_light = float(s_light[di].sum())
            d_sw = sw_model - sw_light
            logf_tab = tp["logF"] + d_sw * swap_logf(dev)
            logf_m = math.log(max(float(fpred[di]), 1e-300)) if fpred is not None else logf_tab
            busy_m = tp["busy"] + np.where(tp["active"], 2 * d_sw * S.T_SWAP / max(tp["active"].sum(), 1), 0)
            logesp_m = esp_from(logf_m, busy_m, tp["active"], dur, dev)
            r["f_source"] = "v8kdc10" if fpred is not None else "table"
            r["model_tableF"] = {"logF": logf_tab, "logESP": esp_from(logf_tab, busy_m, tp["active"], dur, dev), "duration": dur}
            t_phys = time.perf_counter() - t0
            replay = float(light[di][3])
            r.update(path="model", n_sw_light=sw_light, n_sw_model=sw_model, missing=tp["missing"],
                     t_light=t_light[di], t_features=t_feat / len(devs), t_model=t_model / len(devs), t_sched=t_sched,
                     t_phys=t_phys,
                     t_path=t_light[di] + t_feat / len(devs) + t_model / len(devs) + t_sched + t_phys,
                     pred={"logF": logf_m, "logESP": logesp_m, "duration": dur},
                     light_table={"logF": tp["logF"], "logESP": esp_from(tp["logF"], tp["busy"], tp["active"], replay, dev),
                                  "duration": replay})
            if baselines:
                t0 = time.perf_counter(); tl = dev.pm_light.run(qc); ph = LC.physics(tl, dev)
                r["t_light_compile"] = time.perf_counter() - t0
                r["light_compile"] = {"logF": ph["logF"], "logESP": ph["logESP"], "duration": ph["duration"] * 1e6}
        r["t_path"] += t_sw + r["t_encode"]
        if time_full:
            t0 = time.perf_counter(); tf = dev.pm_full.run(qc); ph = LC.physics(tf, dev)
            r["t_full"] = time.perf_counter() - t0            # reference = full compile + the same physics
            r["full_seed0"] = {"logF": ph["logF"], "logESP": ph["logESP"], "duration": ph["duration"] * 1e6}
        rows[dev.name] = r
    return rows


def load_model(ckpt: Path, f_in: int, hidden: int, layers: int):
    model = S.Stage1(f_in, hidden, layers, contention=True)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def labels_of(lab, d: int) -> dict:
    lf = math.log(max(float(lab["F"][d]), 1e-300))
    return {"logF": lf, "logESP": lf + float(lab["decay"][d]), "duration": float(lab["duration"][d]) * 1e6}


# ---------------------------------------------------------------------------------------------------- run
def run(args) -> None:
    torch.set_num_threads(1)
    tab, tr, tc = tables()
    devs = LC.devices()
    chips = S.raw_chips()
    ds = torch.load(args.esp_labels, weights_only=False)
    graphs = {g.circuit_name: g for g in torch.load(S.paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    _, test = S.split_names(args.split, sorted(n for n in graphs if n in ds))
    test = [n for n in test if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    if args.max_circuits:
        test = test[: args.max_circuits // args.n_tasks + 1]
    f_in = next(iter(graphs.values())).x.size(1) + 9 + 12 + 3
    model = load_model(args.ckpt, f_in, args.hidden, args.layers)
    tag = args.tag or (args.ckpt.parent.name if args.ckpt else "untrained")
    fpred = None
    if args.f_pred is not None and args.f_pred.is_file():
        fpred = json.loads(args.f_pred.read_text())
        print(f"F from {args.f_pred} ({len(fpred)} circuits)", flush=True)
    else:
        print(f"no F predictions ({args.f_pred}): F from the decomposition table (biased low)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"{tag}_{args.split}_task{args.task_id:02d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    rng = np.random.default_rng(args.seed)
    print(f"{tag} {args.split}: {len(test)} test circuits in this task ({len(done)} done)", flush=True)
    t_start = time.time()
    with out_file.open("a") as fh:
        for ci, name in enumerate(test):
            if name in done:
                continue
            try:
                qc = SL.circuit(name)
                fp_ = fpred.get(name) if fpred is not None else None
                if fpred is not None and fp_ is None:
                    continue                                       # not in the F model's test set
                rows = predict_circuit(name, qc, graphs[name], model, devs, tab, tr, tc, chips, args.mc, rng,
                                       args.baselines, ci % args.time_full_every == 0, fp_)
                lab = ds[name]
                for di, d in enumerate(S.DEVS):
                    rows[d]["label"] = labels_of(lab, di)
                fh.write(json.dumps({"circuit": name, "split": args.split, "model": tag, "n_qubits": qc.num_qubits,
                                     "rows": rows}, default=float) + "\n")
                fh.flush()
            except Exception as e:                                                 # noqa: BLE001
                print(f"{name}: {type(e).__name__}: {e}"[:300], flush=True)
            if ci % 50 == 0:
                print(f"[{time.time() - t_start:6.0f} s] {ci}/{len(test)} {name}", flush=True)


def smoke(args) -> None:
    """Untrained weights, live inputs only (no stage1.pt / labels): every step runs and gives finite numbers."""
    torch.set_num_threads(1)
    tab, tr, tc = tables()
    devs = LC.devices()
    chips = S.raw_chips()
    SL.DATA_DIRS = (args.data_dir,) + tuple(SL.DATA_DIRS)
    rng = np.random.default_rng(0)
    model = None
    n = 0
    for fam in ("qaoa", "randomcircuit", "qft", "vqe_su2", "ghz", "qnn", "ae"):
        for p in sorted((args.data_dir / fam).glob("*.qasm"))[:2]:
            qc = SL.circuit(f"{fam}/{p.stem}")
            if qc.num_qubits > 20:
                continue
            g = fe.to_data(qc)
            if model is None:
                model = load_model(None, g.x.size(1) + 9 + 12 + 3, 96, 6)
            rows = predict_circuit(f"{fam}/{p.stem}", qc, g, model, devs, tab, tr, tc, chips, 8, rng, True, True,
                                   [0.5, 0.5, 0.5] if n % 2 else None)
            for d, r in rows.items():
                vals = [r["pred"][k] for k in ("logF", "logESP", "duration")]
                assert all(np.isfinite(v) for v in vals), (p.stem, d, r)
                if r["path"] == "model":
                    lt = r["light_table"]; lc = r["light_compile"]
                    print(f"{fam:14} {qc.num_qubits:2}q {d:11} model: dur {r['pred']['duration']:8.2f} µs  light table "
                          f"{lt['duration']:8.2f}  light compile {lc['duration']:8.2f}  full {r['full_seed0']['duration']:8.2f} | "
                          f"logF table {lt['logF']:7.2f} compile {lc['logF']:7.2f} full {r['full_seed0']['logF']:7.2f} | "
                          f"t path {r['t_path'] * 1e3:6.1f} ms full {r['t_full'] * 1e3:6.1f} ms  miss {r['missing']}")
                else:
                    print(f"{fam:14} {qc.num_qubits:2}q {d:11} exact ({r['how']}): dur {r['pred']['duration']:8.2f} µs "
                          f"full {r['full_seed0']['duration']:8.2f}  t {r['t_path'] * 1e3:.1f} ms")
                n += 1
    print(f"smoke: {n} cases, all finite: ok")


# ------------------------------------------------------------------------------------------------- report
def report(args) -> None:
    recs = [json.loads(l) for f in sorted(OUT.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    L = []
    w = L.append
    w("# v28 path end to end: switch -> exact (routing-free) | light Sabre + v28 model (routed)\n")
    w("Truth = the labels (level 2, seed 0). Duration err = median |pred/true − 1|; ESP / F errors = median abs Δ log; "
      "'ESP > 1e-3' = cases whose true ESP is above 1e-3 (as the earlier model reports). Seeds pooled. One CPU thread.\n")
    groups = defaultdict(list)
    for r in recs:
        model = r["model"].split("_s")[0] if r["model"] != "untrained" else "untrained"
        for d, x in r["rows"].items():
            groups[(model, r["split"])].append((r["circuit"].split("/")[0], d, x))

    def stats(sel, key):
        sel = [x for x in sel if key in x]
        if not sel:
            return "–"
        dd = np.array([abs(x[key]["duration"] / x["label"]["duration"] - 1) for x in sel if x["label"]["duration"] > 0])
        bias = np.array([x[key]["duration"] / x["label"]["duration"] for x in sel if x["label"]["duration"] > 0])
        de = np.array([abs(x[key]["logESP"] - x["label"]["logESP"]) for x in sel])
        big = np.array([abs(x[key]["logESP"] - x["label"]["logESP"]) for x in sel if x["label"]["logESP"] > math.log(1e-3)])
        rel = np.array([abs(x[key]["logESP"] - x["label"]["logESP"]) / max(abs(x["label"]["logESP"]), 1e-9) for x in sel])
        df = np.array([abs(x[key]["logF"] - x["label"]["logF"]) for x in sel])
        return (f"{np.median(dd):.1%} / {np.percentile(dd, 90):.1%} | {np.median(bias):.2f} | "
                f"{(np.median(big) if len(big) else float('nan')):.2f} (n={len(big)}) | {np.median(de):.2f} | {np.median(rel):.1%} | "
                f"{np.median(df):.3f}")

    hdr = ("| what | n | duration err median / p90 | ratio | abs Δlog ESP, ESP > 1e-3 | abs Δlog ESP all | rel. | abs Δlog F |\n"
           "|---|---|---|---|---|---|---|---|")
    for (model, split), rows in sorted(groups.items()):
        xs = [x for *_, x in rows]
        rt = [x for x in xs if x["path"] == "model"]
        ex = [x for x in xs if x["path"] == "exact"]
        w(f"## {model}, split {split} ({len(rows) // 3} circuits)\n")
        w(hdr)
        src = Counter(x.get("f_source", "?") for x in rt)
        w(f"| routed: v28 duration, F {'/'.join(src)}, ESP formula | {len(rt)} | {stats(rt, 'pred')} |")
        w(f"| routed: v28 duration, F from the table | {len(rt)} | {stats(rt, 'model_tableF')} |")
        w(f"| routed: light Sabre alone (table) | {len(rt)} | {stats(rt, 'light_table')} |")
        w(f"| routed: light compile (exact physics) | {sum('light_compile' in x for x in rt)} | {stats(rt, 'light_compile')} |")
        w(f"| routed: full compile seed 0, own run (≈ labels) | {sum('full_seed0' in x for x in rt)} | {stats(rt, 'full_seed0')} |")
        w(f"| routing-free: exact | {len(ex)} | {stats(ex, 'pred')} |")
        lost = sum(1 for x in rt if x["how"] == "vf2" and x.get("n_sw_light", 1) == 0)
        w(f"\nSwitch: {Counter(x['how'] for x in xs)}; routed by the switch with 0 light-Sabre SWAPs (possibly routing-free): {lost}")
        miss = Counter()
        for x in rt:
            miss.update(x.get("missing", {}))
        if miss:
            w(f"Gates without a table entry (counted as a generic gate: 1q 2 r, 2q 3 cz + 8 r): {dict(miss)}")
        w("\nRouted, per family (v28 model):\n")
        w(hdr.replace("| what |", "| family |"))
        for fam, n in Counter(f for f, d, x in rows if x["path"] == "model").most_common(12):
            sel = [x for f, d, x in rows if f == fam and x["path"] == "model"]
            w(f"| {fam} | {n} | {stats(sel, 'pred')} |")
        w("\nRouted, per device (v28 model):\n")
        w(hdr.replace("| what |", "| device |"))
        for d in S.DEVS:
            sel = [x for f, dd, x in rows if dd == d and x["path"] == "model"]
            w(f"| {d} | {len(sel)} | {stats(sel, 'pred')} |")
        tf = [x for x in xs if "t_full" in x]
        if tf:
            w("\nTime per (circuit, device), median ms (cases with a timed full compile):\n")
            w("| cases | n | path | full compile | path / full (median of ratios) | sum path / sum full | light compile |")
            w("|---|---|---|---|---|---|---|")
            for lbl, sel in (("routed", [x for x in tf if x["path"] == "model"]), ("routing-free", [x for x in tf if x["path"] == "exact"]),
                             ("all", tf)):
                if not sel:
                    continue
                tp = np.array([x["t_path"] for x in sel]) * 1e3
                tfu = np.array([x["t_full"] for x in sel]) * 1e3
                tl = [x["t_light_compile"] * 1e3 for x in sel if "t_light_compile" in x]
                w(f"| {lbl} | {len(sel)} | {np.median(tp):.1f} | {np.median(tfu):.1f} | {np.median(tp / tfu):.2f} | "
                  f"{tp.sum() / tfu.sum():.2f} | {np.median(tl) if tl else float('nan'):.1f} |")
            rsel = [x for x in tf if x["path"] == "model"]
            if rsel:
                parts = ("t_encode", "t_switch", "t_light", "t_features", "t_model", "t_sched", "t_phys")
                w("\nRouted path, median ms per step: " + ", ".join(
                    f"{k[2:]} {np.median([x[k] for x in rsel]) * 1e3:.2f}" for k in parts))
        w("")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "smoke"))
    ap.add_argument("--split", default="qaoa")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--mc", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--max-circuits", type=int, default=0)
    ap.add_argument("--baselines", action="store_true", help="also the light compile on routed cases")
    ap.add_argument("--time-full-every", type=int, default=1, help="time the full compile on every k-th circuit")
    ap.add_argument("--f-pred", type=Path, default=None, help="predictions.json of the F model (v8kdc10) for this split / seed")
    ap.add_argument("--esp-labels", type=Path, default=EVAL / "compile_check/results/esp_labels.pt")
    ap.add_argument("--data-dir", type=Path, default=EVAL.parent / "data/benchmark_dataset_30k")
    a = ap.parse_args()
    {"run": run, "report": report, "smoke": smoke}[a.cmd](a)
