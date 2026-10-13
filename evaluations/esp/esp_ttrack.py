#!/usr/bin/env python3
"""Do the ESP models follow changes of T1 / T2 and of the gate durations? (calibration test for ESP and duration)

Truth: compile_check/esp_tvariants.py (457 calibration-test circuits × 3 devices × 12 variants that change only
T1 / T2 or the gate durations; the errors, so F, are unchanged).  Models: the control-split models of the ESP heads
(seeds 5–7), as the calibration test of F uses.  Per variant kind, over (circuit, device, variant):
  tracking   corr(predicted Δ, true Δ), Δ = variant − original, for log ESP (= the decay term) and log duration
  slope      least-squares slope of predicted Δ on true Δ (1 = right size, 0 = no reaction)
  |Δ| true   median size of the true change; spurious Δlog F = median |predicted Δlog F| (should be 0)
  regret     ESP lost by keeping the device chosen on the original calibration vs re-predicting on the variant
    python esp_ttrack.py --root evaluations/generalization_v10/results     # -> esp/results/report_esp_ttrack.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
sys.path.insert(0, str(EVAL / "compile_check"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from torch_geometric.loader import DataLoader  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
TV = EVAL / "compile_check/results/esp_tvariants"
MODELS = {"v8kdi10 (v16), fixed gate times": ("v8kdi10", "sinkhorn_rf_route_czc_espi", False),
          "v8kdi10 (v16) + per-qubit gate times, no training": ("v8kdi10", "sinkhorn_rf_route_czc_espi", True),
          "v8kdg10 (v17), fixed gate times": ("v8kdg10", "sinkhorn_rf_route_czc_esps", False),
          "v8kdg10 (v17) + per-qubit gate times, no training (v18 zero-shot)": ("v8kdg10", "sinkhorn_rf_route_czc_esps", True),
          "v8kdt10 (v18): v17 fine-tuned on T1 / T2 / gate-time variants": ("v8kdt10", "sinkhorn_rf_route_czc_esps", True),
          "v8kdt10b (v18b): as v18, change loss × 0.1, F invariance × 3": ("v8kdt10b", "sinkhorn_rf_route_czc_esps", True),
          "v8kdg10 (v17) weights in the formula head, hard placement, coupler CZ times (v19 zero-shot)": ("v8kdg10", "sinkhorn_rf_route_czc_espf", True),
          "v8kdg10 (v17) weights in the formula head, compiler's true layout (oracle)": ("v8kdg10", "sinkhorn_rf_route_czc_espf", True, "oracle"),
          "v8kdf10 (v19): formula head trained, no T / gate-time variants in training": ("v8kdf10", "sinkhorn_rf_route_czc_espf", True),
          "v8kdf10 (v19), compiler's true layout (oracle)": ("v8kdf10", "sinkhorn_rf_route_czc_espf", True, "oracle"),
          "v8kdp10 (v20): v19 + ESP-only placement trained on the exact layout": ("v8kdp10", "sinkhorn_rf_route_czc_espp", True),
          "v8kdfi10 (v19i): the formula head on v8kdi10": ("v8kdfi10", "sinkhorn_rf_route_czc_espfi", True),
          "v8kdfw10 (v21w: v19i + r count weight 1)": ("v8kdfw10", "sinkhorn_rf_route_czc_espfi", True),
          "v8kdfs10 (v21s: v19i + duration kept on the schedule)": ("v8kdfs10", "sinkhorn_rf_route_czc_espfi", True),
          "v8kdfr10 (v21: v19i + r count weight 1 + duration on the schedule)": ("v8kdfr10", "sinkhorn_rf_route_czc_espfi", True),
          "v8kdft10 (v21t: decomposition table for routing-free circuits, r count weight 1)": ("v8kdft10", "sinkhorn_rf_route_czc_espft", True),
          "v8kdfx10 (v22: v21t + 15k synthetic routing circuits in training)": ("v8kdfx10", "sinkhorn_rf_route_czc_espft", True)}
T_FIX = {"QExa20": 10.0}


def _times(be, t) -> list:
    """Per active qubit [r, measure, mean incident CZ] durations in µs (v18 ``raw["t"]``)."""
    qubits = list(getattr(be, "active_qubits", None) or range(be.num_qubits))
    pos = {q: i for i, q in enumerate(qubits)}
    cz = {i: [] for i in range(len(qubits))}
    for (a, b), p in t["cz"].items():
        if p is not None and p.duration and a in pos and b in pos:
            cz[pos[a]].append(p.duration); cz[pos[b]].append(p.duration)
    out = []
    for q in qubits:
        r, m = t["r"].get((q,)), t["measure"].get((q,))
        out.append([r.duration * 1e6, m.duration * 1e6, float(np.mean(cz[pos[q]])) * 1e6 if cz[pos[q]] else 0.130])
    return out


def _coupler_times(be, t, edges) -> list:
    """v19: CZ duration in µs of every coupler of the device graph (``edges``: local pairs), mean over directions."""
    qubits = list(getattr(be, "active_qubits", None) or range(be.num_qubits))
    pos = {q: i for i, q in enumerate(qubits)}
    acc = {}
    for (a, b), p in t["cz"].items():
        if p is not None and p.duration and a in pos and b in pos:
            acc.setdefault((min(pos[a], pos[b]), max(pos[a], pos[b])), []).append(p.duration * 1e6)
    return [float(np.mean(acc[tuple(e)])) if tuple(e) in acc else 0.130 for e in edges]


def oracle_layouts(names, recs, keys) -> dict:
    """{circuit: (n_qubits, len(keys)) device-graph node of each logical qubit in the compiler's layout (seed 0,
    original calibration; the T / duration variants never move it), -1 where unknown."""
    from esp_tvariants import tvariants
    bes = tvariants()
    pos = {}
    for d in DEVS:
        be = bes[f"{d}/orig"]
        pos[d] = {q: i for i, q in enumerate(list(getattr(be, "active_qubits", None) or range(be.num_qubits)))}
    out = {}
    for n in names:
        r = recs[n]
        lays = [r.get(f"{k.split('/')[0]}/orig", {}).get("layout") for k in keys]
        nq = max((len(l) for l in lays if l), default=0)
        t = torch.full((nq, len(keys)), -1, dtype=torch.long)
        for c, (k, l) in enumerate(zip(keys, lays)):
            for q, b in enumerate(l or []):
                t[q, c] = pos[k.split("/")[0]].get(b, -1)
        out[n] = t
    return out


def raw_graphs(with_t: bool = False) -> dict:
    """Raw device graphs of every T / duration variant; T1 / T2 read from the target, where the variants change them.
    ``with_t``: also the per-qubit gate durations (v18), else the model uses the uniform GATE_US."""
    from esp_tvariants import tvariants
    import gsv3  # noqa: F401
    from gsv2.devices import _backend_graph
    out = {}
    for k, be in tvariants().items():
        t = be.target
        shim = SimpleNamespace(target=t, qubit_properties=t.qubit_properties, num_qubits=be.num_qubits,
                               active_qubits=getattr(be, "active_qubits", None))
        out[k] = dict(zip(("nodes", "edges", "edge_feats"), _backend_graph(shim)))
        if with_t:
            out[k]["t"] = _times(be, t)
            out[k]["tc"] = _coupler_times(be, t, out[k]["edges"])
    return out


def truth() -> tuple[list, dict]:
    recs = {}
    for f in sorted(TV.glob("task0*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                recs[r["circuit"]] = r
    return sorted(recs), recs


@torch.no_grad()
def predict(model, graphs, dev):
    out = {k: [] for k in ("logf", "decay", "logdur")}
    for b in DataLoader(graphs, batch_size=32):
        logf = model(b, dev)
        decay, logdur, _ = model.last_esp
        for k, v in zip(out, (logf, decay, logdur)):
            out[k].append(v.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=EVAL / "generalization_v10/results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    raw = raw_graphs()
    raw_t = raw_graphs(with_t=True)
    keys = list(raw)
    ref = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    for d in DEVS:         # sanity: the original devices must match the training-time graphs
        a, b = np.array(raw[f"{d}/orig"]["nodes"]), np.array(ref[d]["nodes"])
        print(f"orig graph {d}: {'OK' if a.shape == b.shape and np.allclose(a, b, atol=1e-6) else 'MISMATCH'}", flush=True)
    devs = {False: variant_batch(raw, keys, params["lap_pe"]), True: variant_batch(raw_t, keys, params["lap_pe"])}
    names, recs = truth()
    data = {g.circuit_name: g for g in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    names = [n for n in names if n in data]
    kinds = sorted({k.split("/")[1].rstrip("0123456789") for k in keys if not k.endswith("/orig")},
                   key=lambda s: ("tjit", "tbad", "djit", "czslow", "mslow", "mix").index(s))
    L = [f"# Do the ESP heads follow T1 / T2 and gate-duration changes? {args.tag}", "",
         f"{len(names)} calibration-test circuits × 3 devices; per kind 2 variants per device; control-split models, "
         "mean over seeds 5–7. Δ = variant − original. Only T1 / T2 or durations change, so the true Δlog F is 0.", ""]
    oracle = oracle_layouts(names, recs, keys)
    for mname, (run, kind, use_t, *opt) in MODELS.items():
        dev = devs[use_t]
        res = {}
        for seed in (5, 6, 7):
            sfx = "" if seed == 5 else f"_s{seed}"
            path = args.root / f"{run}{sfx}" / f"random_control/random_seed{seed}/model.pth"
            if not path.is_file():
                print("skip", path, flush=True)
                continue
            model = build_model(kind, params, dev, torch.device("cpu"))
            model.load_state_dict(torch.load(path, map_location="cpu")); model.eval()
            if "oracle" in opt:
                model.esp_oracle = oracle
            p = predict(model, [data[n] for n in names], dev)
            col = {k: i for i, k in enumerate(keys)}
            for kd in kinds:
                pe, te, pd, td, pf, keep, swap = [], [], [], [], [], [], []
                for j in range(2):
                    tr_e, pr_e, tr_esp_v, pr_esp_v, pr_esp_o = [], [], [], [], []
                    for i, n in enumerate(names):
                        r = recs[n]
                        ev, eo, rv, ro = [], [], [], []
                        for d in DEVS:
                            o, v = r[f"{d}/orig"], r.get(f"{d}/{kd}{j}")
                            if not v or v.get("decay") is None or o.get("decay") is None:
                                break
                            fix = T_FIX.get(d, 1.0)
                            co, cv = col[f"{d}/orig"], col[f"{d}/{kd}{j}"]
                            te.append((v["decay"] - o["decay"]) / fix); pe.append(p["decay"][i, cv] - p["decay"][i, co])
                            td.append(math.log(v["duration"] / o["duration"])); pd.append(p["logdur"][i, cv] - p["logdur"][i, co])
                            pf.append(abs(p["logf"][i, cv] - p["logf"][i, co]))
                            ev.append(v["F"] * math.exp(v["decay"] / fix))
                            eo.append(o["F"] * math.exp(o["decay"] / fix))
                            fo = math.exp(min(p["logf"][i, co], 0)); fv = math.exp(min(p["logf"][i, cv], 0))
                            ro.append(fo * math.exp(p["decay"][i, co])); rv.append(fv * math.exp(p["decay"][i, cv]))
                        else:
                            ev, rv, ro = np.array(ev), np.array(rv), np.array(ro)
                            keep.append(ev.max() - ev[int(np.argmax(ro))])
                            swap.append(ev.max() - ev[int(np.argmax(rv))])
                pe, te, pd, td = map(np.array, (pe, te, pd, td))
                c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
                s = lambda a, b: float((a * b).sum() / max((b * b).sum(), 1e-12))
                res.setdefault(kd, []).append([c(pe, te), s(pe, te), float(np.median(np.abs(te))),
                                               c(pd, td), s(pd, td), float(np.median(np.abs(td))),
                                               float(np.median(pf)), float(np.mean(keep)), float(np.mean(swap))])
            print("done", mname, seed, flush=True)
        L += [f"## {mname}", "", "| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | "
              "duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | "
              "ESP regret, re-predict |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for kd in kinds:
            if kd not in res:
                continue
            m = np.nanmean(np.array(res[kd]), axis=0)
            f = lambda x, k=3: "–" if not np.isfinite(x) else f"{x:.{k}f}"
            L.append(f"| {kd} | {f(m[0], 2)} | {f(m[1], 2)} | {f(m[2])} | {f(m[3], 2)} | {f(m[4], 2)} | {f(m[5])} | "
                     f"{f(m[6], 4)} | {f(m[7], 4)} | {f(m[8], 4)} |")
        L.append("")
    L += ["Kinds: tjit = T1/T2 × lognormal(0.3) on every qubit; tbad = 2–3 qubits with T × 0.15–0.3; djit = every "
          "gate duration × lognormal(0.2); czslow = 3–5 couplers with the CZ 2–3× slower; mslow = readout 1.3–2× slower; "
          "mix = tjit + djit. A tracking of – means the model's prediction does not move at all (no input carries "
          "that change)."]
    out = HERE / f"results/report_esp_ttrack{args.tag}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
