#!/usr/bin/env python3
"""Fast path for routing-free (circuit, device) pairs (fastpath.py): validation against the compiler and
predictions for the hybrid evaluation.

  validate  sample of test circuits of the four standard splits; per device and compiler seed: the reference
            compilation on the original calibration (structure + cached embeddings, once), then for the original
            and the local v8d calibrations aug 80-99: fast-path F / ESP (no compiler) vs a full level-2
            recompilation (MQT F / ESP, as the labels), layouts, times.
  predict   every test circuit of the four splits (seeds 5, 6, 7 of --run), original devices, seed 0 (as
            esp_labels.pt and the dataset labels): switch, compiler's VF2Layout decision, fast F / ESP.
  report    validate -> results/report_validate.md;  merge -> results/fastpath_pred.json (for esp/hybrid.py).
    python run_fastpath.py validate --task-id 0 --n-tasks 8
    python run_fastpath.py predict  --task-id 0 --n-tasks 8
    python run_fastpath.py report ; python run_fastpath.py merge
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EVAL / "compile_check"))
sys.path.append(str(EVAL.parent))       # createDevice.py at the repository root when ~/compileCircuits is absent
import fastpath as fp  # noqa: E402

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
SPLITS = {"LOFO qaoa": "leave_one_family_out/qaoa", "LOFO qnn": "leave_one_family_out/qnn",
          "LOGO variational": "leave_one_group_out/variational", "control": "random_control/random_seed{s}"}
EXCLUDE = ("grover", "iqpe", "dynamic_qft", "ghz_dynamic", "seven_qubit_steane_code", "shors_nine_qubit_code")
LOCAL = tuple(range(80, 100))          # v8d local variants (jitter / shuffle / bad qubits / mixes)


FAMILY_SPLIT = {"qaoa": "LOFO qaoa", "qnn": "LOFO qnn", "vqe_two_local": "LOGO variational",
                "vqe_real_amp": "LOGO variational", "vqe_su2": "LOGO variational"}
DATA_DIRS = (Path.home() / "compileCircuits/benchmark_dataset_30k", EVAL.parent / "data/benchmark_dataset_30k")


def data_dir(arg: Path | None) -> Path:
    for d in ((arg,) if arg else DATA_DIRS):
        if d is not None and d.is_dir():
            return d
    raise SystemExit(f"no circuit directory: pass --data-dir (tried {[str(d) for d in DATA_DIRS]})")


def all_names(data: Path) -> dict:
    """{circuit: split} without the models' predictions.json: every circuit of esp_labels.pt (else every QASM file),
    split by family as the hold-outs define them (qaoa, qnn, vqe_* + qnn); the other families stand for control."""
    lab = EVAL / "compile_check/results/esp_labels.pt"
    if lab.is_file():
        import torch
        names = list(torch.load(lab, weights_only=False))
    else:
        names = [str(p.relative_to(data).with_suffix("")) for p in data.glob("*/*.qasm")]
    return {n: FAMILY_SPLIT.get(n.split("/")[0], "control") for n in sorted(names)
            if not n.split("/")[0].startswith(EXCLUDE)}


def test_names(run: str, seeds=(5, 6, 7)) -> dict:
    """{circuit: split} of the test circuits of the standard splits (union over seeds)."""
    root = EVAL / "generalization_v9/results"
    out = {}
    for seed in seeds:
        for split, sd in SPLITS.items():
            for rn in ((run, f"{run}_s5") if seed == 5 else (f"{run}_s{seed}",)):
                f = root / rn / sd.format(s=seed) / "predictions.json"
                if f.is_file():
                    for n in json.loads(f.read_text()):
                        if not n.split("/")[0].startswith(EXCLUDE):
                            out.setdefault(n, split)
                    break
    return out


def mqt(tc, backend, name: str) -> tuple[float, float]:
    """MQT F and ESP of a compilation, as the labels (T1/T2 from the backend, QExa20 decay / T_FIX)."""
    from mqt.predictor.reward import estimated_success_probability, expected_fidelity
    t = backend.target
    if t.qubit_properties is None:
        t.qubit_properties = fp.qubit_props(backend)
    f = float(expected_fidelity(tc, t, precision=300))
    e = float(estimated_success_probability(tc, t, precision=300))
    if f > 0 and e > 0:
        e = f * float(np.exp((np.log(e) - np.log(f)) / fp.T_FIX.get(name, 1.0)))
    return f, e


def load_backends(aug: tuple) -> dict:
    from v8b_variants import make_aug, variants
    base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
    out = {}
    for i, d in enumerate(DEVICES):
        out[(d, "orig")] = base[d]
        for j in aug:
            out[(d, j)] = make_aug(base[d], j, i)
    return out


def fast_rows(qc, tc, tcomp, virt, bes, cal, d, s, aug, mode):
    """Fast-path predictions of one routing-free (circuit, device, seed) on the original calibration and the
    variants ``aug``; in validate mode with the full recompilation of each as truth."""
    st = fp.embeddings(fp.structure(tc, virt, bes[(d, "orig")].target, tcomp), cal[(d, "orig")])
    extra = {"n_emb": int(st.emb.shape[0]), "t_emb": st.t_emb, "duration": st.duration}
    rows = {}
    for key in ["orig", *aug]:
        p = fp.predict(st, cal[(d, key)])
        row = {"F": p["F"], "ESP": p["ESP"], "vals": p["vals"], "ties": p["n_ties"], "t": p["t"]}
        if mode == "validate":
            tv, tf = (tc, tcomp) if key == "orig" else fp.compile_reference(qc, bes[(d, key)], s)[:2]
            ft, et = mqt(tv, bes[(d, key)], d)
            lay = tv.layout.final_index_layout(filter_ancillas=True)
            row.update(F_true=ft, ESP_true=et, t_full=tf,
                       same_layout=all(lay[v] == p["layout"][v] for v in p["layout"]))
            # why the placement differs: the compiler's placement scored and evaluated with the cached structure
            c = cal[(d, key)]
            ph_t = np.array([lay[v] for v in st.qubits])
            ph_f = np.array([p["layout"][int(v)] for v in st.qubits])
            lt, dt = fp.evaluate(st, c, ph_t)
            row.update(d_score=fp.vf2_score(st, c, ph_f) - fp.vf2_score(st, c, ph_t), score_true=fp.vf2_score(st, c, ph_t),
                       in_cache=bool((st.emb == ph_t[st.int_rows]).all(1).any()) if st.emb.shape[1] else True,
                       logF_struct=lt, decay_struct=dt)
        elif key == "orig":
            row["F_true"], row["ESP_true"] = mqt(tc, bes[(d, key)], d)
        rows[str(key)] = row
    return rows, extra


def run(args) -> None:
    from qiskit import qasm3
    DATA = data_dir(args.data_dir)
    tests = test_names(args.run) if args.circuits in ("auto", "tests") else {}
    if not tests:
        if args.circuits == "tests":
            raise SystemExit(f"no predictions.json of {args.run} under generalization_v9/results")
        tests = all_names(DATA)
        print(f"no predictions.json of {args.run}: {len(tests)} circuits from esp_labels.pt / {DATA}, split by family", flush=True)
    names = sorted(tests)
    if args.mode == "validate":
        per = defaultdict(list)
        for n in names:
            per[tests[n]].append(n)
        rnd = random.Random(0)
        names = sorted(x for sp in per.values() for x in rnd.sample(sorted(sp), min(args.n_per_split, len(sp))))
    names = [n for n in names if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    aug = LOCAL if args.mode == "validate" else ()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    bes = load_backends(aug)
    cal = {k: fp.calib(be, k[0]) for k, be in bes.items()}
    out = args.out / args.mode
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"task{args.task_id:03d}.jsonl"
    done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()} if out_file.is_file() else set()
    t_start = time.time()
    with out_file.open("a") as fh:
        for ci, name in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name, "split": tests[name], "dev": {}}
            try:
                qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
                if name.startswith("qaoa/") and "measure" not in qc.count_ops():
                    qc.measure_all()     # data/circuits.zip has the old qaoa without final measurements (qaoa_measure.py)
                for d in DEVICES:
                    t0 = time.perf_counter()
                    sw = fp.routing_free_switch(qc, cal[(d, "orig")])
                    r = {"switch": sw, "t_switch": time.perf_counter() - t0, "seeds": {}}
                    if sw or args.mode == "validate":
                        for s in seeds:
                            tc, tcomp, found, virt = fp.compile_reference(qc, bes[(d, "orig")], s)
                            x = {"found": found, "t_compile": tcomp}
                            if found:
                                try:
                                    x["cal"], extra = fast_rows(qc, tc, tcomp, virt, bes, cal, d, s, aug, args.mode)
                                    x.update(extra)
                                except Exception as e:                      # noqa: BLE001
                                    x["error"] = f"{type(e).__name__}: {e}"[:300]
                            r["seeds"][str(s)] = x
                    rec["dev"][d] = r
            except Exception as e:                                          # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if ci % 20 == 0:
                print(f"[{time.time() - t_start:6.0f} s] {ci + 1}/{len(names)} {name}", flush=True)


def read(directory: Path) -> list:
    seen, out = set(), []
    for f in sorted(directory.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r["circuit"] not in seen:
                seen.add(r["circuit"]); out.append(r)
    return out


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def report(args) -> None:
    recs = read(args.out / "validate")
    seeds = [str(s) for s in args.seeds.split(",")]
    err = sum("error" in r for r in recs)
    L = ["# Fast path for routing-free circuits: validation against the compiler", "",
         f"{len(recs)} test circuits of the four splits ({err} failed), 3 devices, compiler seeds {seeds}; "
         f"original calibration + local v8d variants aug {LOCAL[0]}-{LOCAL[-1]}. Fast path = one reference "
         "compilation per (circuit, device, seed) on the original calibration, then per calibration only the "
         "VF2Layout score over the cached embeddings (fastpath.py). Truth = full level-2 recompilation, MQT F / ESP "
         "(as the labels). Ties in the score: mean over the tied placements.", ""]
    # switch vs the compiler's VF2Layout decision
    sw = [(x["switch"], x["seeds"][seeds[0]]["found"]) for r in recs if "error" not in r for x in r["dev"].values()
          if seeds[0] in x["seeds"]]
    sw = np.array(sw, bool)
    if len(sw):
        tp = (sw[:, 0] & sw[:, 1]).sum()
        L += ["## Switch (VF2 subgraph test, no compiler) vs the compiler's VF2Layout decision", "",
              "| pairs | routing-free (compiler) | switch says free | precision | recall | time, median |",
              "| --- | --- | --- | --- | --- | --- |",
              f"| {len(sw)} | {sw[:, 1].mean():.1%} | {sw[:, 0].mean():.1%} | {tp / max(sw[:, 0].sum(), 1):.3f} | "
              f"{tp / max(sw[:, 1].sum(), 1):.3f} | "
              f"{np.median([x['t_switch'] for r in recs if 'error' not in r for x in r['dev'].values()]) * 1e3:.3f} ms |", ""]
    # per seed, routing-free pairs
    rows = []      # (split, dev, seed, key, F, Ft, E, Et, same_layout, t, t_full, F_orig, Ft_orig, ties)
    best = defaultdict(dict)
    for r in recs:
        for d, x in r.get("dev", {}).items():
            for s, y in x["seeds"].items():
                if not y.get("found") or "cal" not in y:
                    continue
                o = y["cal"]["orig"]
                for key, z in y["cal"].items():
                    a0, b0 = z["vals"][0]
                    rows.append((r["split"], d, s, key, z["F"], z["F_true"], z["ESP"], z["ESP_true"], z["same_layout"],
                                 z["t"], z["t_full"], o["F"], o["F_true"], z["ties"], np.exp(a0), np.exp(a0 + b0),
                                 np.exp(o["vals"][0][0])))
                    best[(r["circuit"], d, key)][s] = z
    if not rows:
        (args.out / "report_validate.md").write_text("\n".join(L)); print("\n".join(L)); return
    A = np.array([x[4:8] + x[9:] for x in rows], float)
    lay = np.array([x[8] for x in rows], bool)
    split = np.array([x[0] for x in rows]); dev = np.array([x[1] for x in rows]); key = np.array([x[3] for x in rows])
    Fm, Ft, Em, Et, t, tfull, Fom, Fto, ties, F1, E1, Fo1 = A.T

    def stats(sel, tag, first=False):
        F, E, Fo = (F1, E1, Fo1) if first else (Fm, Em, Fom)
        ok = sel & (F > 1e-6) & (Ft > 1e-6)
        lf = np.abs(np.log(F[ok]) - np.log(Ft[ok]))
        okE = ok & (E > 1e-9) & (Et > 1e-9)
        le = np.abs(np.log(E[okE]) - np.log(Et[okE]))
        var = ok & (key != "orig") & (Fo > 1e-6) & (Fto > 1e-6)
        dP, dT = np.log(F[var]) - np.log(Fo[var]), np.log(Ft[var]) - np.log(Fto[var])
        return (f"| {tag} | {ok.sum()} | {np.median(lf):.1e} / {np.percentile(lf, 90):.1e} | {np.mean(lf < 1e-6):.1%} | "
                f"{np.median(le):.1e} / {np.percentile(le, 90):.1e} | {corr(dP, dT):.3f} | {lay[sel].mean():.1%} | "
                f"{np.mean(ties[sel] > 1):.1%} |")
    for first in (False, True):
        L += [f"## Per compiler seed, ties {'broken by the first embedding found' if first else 'averaged'} "
              "(fast path vs the recompilation with the same seed)", "",
              "|Δ log F| and |Δ log ESP|: median / p90. Tracking = corr of Δ log F (variant − original) fast vs true, "
              "local variants only. Same layout = the compiler's placement reproduced exactly (first tied embedding). "
              "Ties = share of cases with several placements tied for the best VF2 score.", "",
              "| group | cases | abs log F error | exact (< 1e-6) | abs log ESP error | tracking | same layout | ties |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        L.append(stats(np.ones(len(Fm), bool), "all", first))
        for sp in SPLITS:
            if (split == sp).any():
                L.append(stats(split == sp, sp, first))
        for d in DEVICES:
            if (dev == d).any():
                L.append(stats(dev == d, d, first))
        L.append(stats(key == "orig", "original calibration", first))
        L.append(stats(key != "orig", "local variants", first))
        L.append("")
    # best of the seeds, as the recompiled labels
    bo = []
    for (c, d, k), per in best.items():
        if set(per) != set(seeds):
            continue
        pr = fp.best_of([{"vals": [tuple(v) for v in per[s]["vals"]]} for s in seeds])
        tb = max(per.values(), key=lambda z: z["F_true"])
        bo.append((k, pr["F"], tb["F_true"], pr["ESP"], tb["ESP_true"], c, d))
    # why the fast path misses: compiler's placement vs the fast path's, per seed
    diag = [z for r in recs for x in r.get("dev", {}).values() for y in x["seeds"].values()
            if y.get("found") and "cal" in y for z in y["cal"].values() if "d_score" in z]
    if diag:
        def lerr(z):
            return abs(np.log(max(z["F"], 1e-300)) - np.log(max(z["F_true"], 1e-300)))
        tol = lambda z: 1e-9 * max(1.0, abs(z["score_true"]))
        cats = (("same placement as the compiler", lambda z: z["same_layout"]),
                ("tie: same VF2 score, the compiler took another tied placement",
                 lambda z: not z["same_layout"] and abs(z["d_score"]) <= tol(z)),
                ("fast path scores higher: the compiler's search stopped early (call limit)",
                 lambda z: not z["same_layout"] and z["d_score"] > tol(z)),
                ("fast path scores lower, compiler's placement in the cache",
                 lambda z: not z["same_layout"] and z["d_score"] < -tol(z) and z["in_cache"]),
                ("fast path scores lower, compiler's placement NOT in the cache (embedding cap)",
                 lambda z: not z["same_layout"] and z["d_score"] < -tol(z) and not z["in_cache"]))
        L += ["## Why the placement differs", "",
              "Each case: the compiler's own placement scored with VF2Layout's score and looked up in the cached embeddings. "
              "Bias = median signed log F error (fast − true).", "",
              "| cause | share of cases | abs log F error, median / p90 | bias |", "| --- | --- | --- | --- |"]
        for lab, fn in cats:
            g = [z for z in diag if fn(z)]
            if g:
                e = np.array([lerr(z) for z in g])
                b = np.median([np.log(max(z["F"], 1e-300)) - np.log(max(z["F_true"], 1e-300)) for z in g])
                L.append(f"| {lab} | {len(g) / len(diag):.1%} | {np.median(e):.1e} / {np.percentile(e, 90):.1e} | {b:+.1e} |")
            else:
                L.append(f"| {lab} | 0.0% | – | – |")
        sc = np.array([abs(z["logF_struct"] - np.log(max(z["F_true"], 1e-300))) for z in diag if z["F_true"] > 1e-6])
        L += ["", f"Structure check (cached structure evaluated on the compiler's own placement vs the true F): "
              f"|Δ log F| median {np.median(sc):.1e}, p90 {np.percentile(sc, 90):.1e}, exact (< 1e-6) {np.mean(sc < 1e-6):.1%}. "
              "Not exact = the compiled structure itself changes with the calibration (not a placement error).", ""]
    if bo:
        orig = {(c, d): (f, ft) for k, f, ft, e, et, c, d in bo if k == "orig"}
        x = [(np.log(f) - np.log(orig[(c, d)][0]), np.log(ft) - np.log(orig[(c, d)][1]))
             for k, f, ft, e, et, c, d in bo if k != "orig" and (c, d) in orig and min(f, ft, *orig[(c, d)]) > 1e-6]
        lf = np.array([abs(np.log(f) - np.log(ft)) for k, f, ft, *_ in bo if min(f, ft) > 1e-6])
        le = np.array([abs(np.log(e) - np.log(et)) for k, f, ft, e, et, *_ in bo if min(e, et) > 1e-9])
        x = np.array(x)
        L += ["", f"## Best of seeds {seeds} (as the recompiled labels)", "",
              "| cases | abs log F error, median / p90 | abs log ESP error, median / p90 | tracking (corr Δ log F, local variants) |",
              "| --- | --- | --- | --- |",
              f"| {len(bo)} | {np.median(lf):.1e} / {np.percentile(lf, 90):.1e} | {np.median(le):.1e} / "
              f"{np.percentile(le, 90):.1e} | {corr(x[:, 0], x[:, 1]) if len(x) else float('nan'):.3f} |",
              "", "Model for reference (v8kdc10, calibration test): local tracking 0.66 ± 0.02 on all circuits; "
              "0.83 on routing-free cases in the tracking anatomy."]
    # time
    temb = [y["t_emb"] for r in recs for x in r.get("dev", {}).values() for y in x["seeds"].values() if "t_emb" in y]
    tcomp = [y["t_compile"] for r in recs for x in r.get("dev", {}).values() for y in x["seeds"].values() if y.get("found")]
    nemb = [y["n_emb"] for r in recs for x in r.get("dev", {}).values() for y in x["seeds"].values() if "n_emb" in y]
    L += ["", "## Time per (circuit, device), routing-free", "",
          "| step | median | p90 | max |", "| --- | --- | --- | --- |"]
    for lab, v in (("reference compilation (once per seed)", tcomp), ("embedding enumeration (once)", temb),
                   ("fast path per calibration (score + F + ESP)", t[key != "orig"]),
                   ("full recompilation per calibration", tfull[key != "orig"])):
        v = np.asarray(v, float) * 1e3
        if len(v):
            L.append(f"| {lab} | {np.median(v):.2f} ms | {np.percentile(v, 90):.2f} ms | {v.max():.1f} ms |")
    if nemb:
        L += ["", f"Cached embeddings per structure: median {int(np.median(nemb))}, max {max(nemb)} "
              f"(cap {fp.MAX_EMB}; {np.mean(np.array(nemb) >= fp.MAX_EMB):.1%} at the cap)."]
    (args.out / "report_validate.md").write_text("\n".join(L))
    print("\n".join(L))


def merge(args) -> None:
    """predict -> fastpath_pred.json {circuit: {"switch": [D], "found": [D], "F": [D], "ESP": [D],
    "F_true": [D], "ESP_true": [D]}} (None where the pair is routed); seed 0 only."""
    recs = read(args.out / "predict")
    s = args.seeds.split(",")[0]
    out, err = {}, 0
    for r in recs:
        if "error" in r:
            err += 1
            continue
        e = {k: [] for k in ("switch", "found", "F", "ESP", "F_true", "ESP_true", "t_compile", "t_emb", "t")}
        for d in DEVICES:
            x = r["dev"][d]
            y = x["seeds"].get(s, {})
            o = y.get("cal", {}).get("orig", {})
            e["switch"].append(bool(x["switch"])); e["found"].append(bool(y.get("found", False)))
            for k in ("F", "ESP", "F_true", "ESP_true", "t"):
                e[k].append(o.get(k))
            e["t_compile"].append(y.get("t_compile")); e["t_emb"].append(y.get("t_emb"))
        out[r["circuit"]] = e
    f = args.out / "fastpath_pred.json"
    f.write_text(json.dumps(out))
    rf = np.array([v["found"] for v in out.values()], bool)
    print(f"{len(out)} circuits ({err} failed) -> {f}; routing-free pairs {rf.mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "predict", "report", "merge"])
    ap.add_argument("--run", default="v8kdc10", help="run whose predictions.json define the test circuits")
    ap.add_argument("--n-per-split", type=int, default=60, help="validate: circuits sampled per split")
    ap.add_argument("--seeds", default=None, help="compiler seeds (validate: 0,1,2; predict: 0)")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--data-dir", type=Path, default=None, help="QASM circuits (default ~/compileCircuits/benchmark_dataset_30k, "
                    "else data/benchmark_dataset_30k)")
    ap.add_argument("--circuits", choices=["auto", "tests", "all"], default="auto",
                    help="auto = the test circuits of --run's predictions.json if present, else all circuits by family")
    a = ap.parse_args()
    if a.seeds is None:
        a.seeds = "0" if a.mode in ("predict", "merge") else "0,1,2"
    {"validate": run, "predict": run, "report": report, "merge": merge}[a.mode](a)
