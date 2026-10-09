#!/usr/bin/env python3
"""Decide which v8r add-ons go into the final v9 (v9f) and launch it — run unattended by v9f_supervisor.sh.

Each add-on is judged on seed 5 against the run that differs from it only by that add-on:
    cz-weighted count loss    v8r_cz  vs  v8r
    routing-cost estimate     v8s     vs  v8r_cz
    relative device features  v9      vs  v8r
An add-on is kept if, on the three unseen-family splits (LOFO qaoa, LOFO qnn, LOGO variational),
the mean test R² improves by ≥ MIN_GAIN, no split loses more than MAX_SPLIT_LOSS, the control R²
loses at most MAX_CONTROL_LOSS and the local calibration tracking at most MAX_TRACK_LOSS.
The kept add-ons define v9f; if it equals a run that already exists (e.g. only the cz loss → v8r_cz)
only its missing seeds 6 and 7 are launched, otherwise v9f is trained on seeds 5, 6, 7.
Nothing is launched if no add-on is kept or a result is missing.  Writes generalization_v9/DECISION.md
(shown in RESULTS.md) and generalization_v9/results/decision.json (read by calibration_shift/run_shift.py).
    python decide_v9f.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
REPO = EVAL.parent
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "generalization_v3" / "scripts"))
import results_report  # noqa: E402
import summarize  # noqa: E402

MIN_GAIN, MAX_SPLIT_LOSS, MAX_CONTROL_LOSS, MAX_TRACK_LOSS = 0.005, 0.03, 0.003, 0.03
OOD = [("lofo", "qaoa"), ("lofo", "qnn"), ("logo", "variational")]
TESTS = [("cz", "cz-weighted count loss", "v8r_cz", "v8r"),
         ("route", "routing-cost estimate", "v8s", "v8r_cz"),
         ("rel", "relative device features", "v9", "v8r")]
# (rel, route, cz) -> run that already exists with exactly these add-ons (seed 5 trained)
EXISTING = {(False, False, False): "v8r", (False, False, True): "v8r_cz",
            (False, True, True): "v8s", (True, False, False): "v9"}
KIND = {(False, False): "sinkhorn_rf", (True, False): "sinkhorn_rf_rel",
        (False, True): "sinkhorn_rf_route", (True, True): "sinkhorn_rf_rel_route"}
LP = "evaluations/compile_check/results/compiler_layouts_v6pilot.pt"
CP = "evaluations/compile_check/results/compiler_counts_v8.pt"
SPLITS = [("lofo", "qaoa"), ("lofo", "qnn"), ("logo", "variational"), ("control", "x")]
OUT_MD = HERE / "DECISION.md"
OUT_JSON = HERE / "results" / "decision.json"


def metrics(values, track, cfg):
    m = {}
    for exp, name in OOD:
        v = values.get((exp, name, None), {}).get(cfg, {}).get(5)
        m[name] = v["r2"] if v else None
    v = values.get(("control", "random_seed5", None), {}).get(cfg, {}).get(5)
    m["control"] = v["r2"] if v else None
    t = track.get(cfg, {}).get("local_trk")
    m["tracking"] = float(sum(t) / len(t)) if t else None
    return m


def judge(cand, ref):
    if any(x is None for x in (*cand.values(), *ref.values())):
        return None, "missing results"
    d = {k: cand[k] - ref[k] for k in cand}
    gain = sum(d[n] for _, n in OOD) / len(OOD)
    worst = min(d[n] for _, n in OOD)
    ok = (gain >= MIN_GAIN and worst >= -MAX_SPLIT_LOSS and d["control"] >= -MAX_CONTROL_LOSS
          and d["tracking"] >= -MAX_TRACK_LOSS)
    return ok, (f"mean ΔR² unseen {gain:+.3f} (worst {worst:+.3f}), Δcontrol {d['control']:+.3f}, "
                f"Δtracking {d['tracking']:+.2f}")


def earliest_partition():
    best = None
    for part, limit in (("cpu_sapphire_ext", "24:00:00"), ("cpu_sapphire", "12:00:00")):
        r = subprocess.run(["sbatch", "--test-only", "-p", part, "-c", "8", "--mem=16G", "-t", limit,
                            "--wrap=true"], capture_output=True, text=True, cwd=REPO)
        m = re.search(r"start at (\S+)", r.stdout + r.stderr)
        if m and (best is None or m.group(1) < best[0]):
            best = (m.group(1), part, limit)
    return best[1:] if best else ("cpu_sapphire", "12:00:00")


def launch(run, kind, cz, seeds, dry):
    count = "--count-lambda 0.1 --count-weights 0.1,0.1,1" if cz else "--count-lambda 0.01"
    extra = (f"--loss mixed --val-mode random --layout-lambda 0.1 --layout-loss region_dist "
             f"--layouts-path {LP} {count} --counts-path {CP}")
    jobs, controls = [], []
    for s in seeds:
        name = run if s == 5 else f"{run}_s{s}"
        for exp, sp in SPLITS:
            part, limit = earliest_partition()
            tag = f"random_seed{s}" if exp == "control" else sp
            env = dict(RUN=name, MODEL=kind, ALPHA="0", SEED=str(s), EXPERIMENTS=exp, FAMILIES=sp,
                       SPLIT_GROUPS=sp, RESULTS_ROOT="evaluations/generalization_v9/results", EXTRA_ARGS=extra)
            cmd = ["sbatch", "--parsable", "-p", part, f"--time={limit}", f"--job-name=cpu_{name}_{tag}",
                   "--export=ALL", "evaluations/generalization_v3/slurm/train_cpu.sh"]
            if dry:
                jid = "DRY"
            else:
                import os
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env={**os.environ, **env})
                jid = r.stdout.strip().split(";")[0] or f"FAILED: {r.stderr.strip()}"
            jobs.append((name, tag, part, jid))
            if exp == "control" and jid.isdigit():
                controls.append(jid)
    if controls and not dry:
        r = subprocess.run(["sbatch", "--parsable", f"--dependency=afterany:{':'.join(controls)}",
                            "--cpus-per-task=8", "--mem=32G", "--time=01:00:00",
                            "evaluations/calibration_shift/shift_k.sh", "predict_report"],
                           capture_output=True, text=True, cwd=REPO)
        jobs.append(("calibration test", "after the controls", "cpu_sapphire", r.stdout.strip()))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    values = summarize.collect()
    track = results_report.calibration()
    M = {c: metrics(values, track, c) for c in ("v8r", "v8r_cz", "v8s", "v9")}
    L = [f"# v9 decision — {dt.datetime.now():%Y-%m-%d %H:%M}", "",
         f"Rule (seed 5): keep an add-on if mean ΔR² on unseen families ≥ {MIN_GAIN}, no split < −{MAX_SPLIT_LOSS}, "
         f"Δcontrol ≥ −{MAX_CONTROL_LOSS}, Δlocal tracking ≥ −{MAX_TRACK_LOSS}.", "",
         "| run | LOFO qaoa | LOFO qnn | LOGO variational | control | local tracking |", "| --- | --- | --- | --- | --- | --- |"]
    f = lambda x, p=3: "–" if x is None else f"{x:.{p}f}"
    for c, m in M.items():
        L.append(f"| {c} | {f(m['qaoa'])} | {f(m['qnn'])} | {f(m['variational'])} | {f(m['control'])} | {f(m['tracking'], 2)} |")
    L += ["", "| add-on | test | verdict | numbers |", "| --- | --- | --- | --- |"]
    keep, missing = {}, False
    for key, label, cand, ref in TESTS:
        ok, txt = judge(M[cand], M[ref])
        missing |= ok is None
        keep[key] = bool(ok)
        L.append(f"| {label} | {cand} vs {ref} | {'missing' if ok is None else ('**keep**' if ok else 'drop')} | {txt} |")
    L.append("")
    combo = (keep["rel"], keep["route"], keep["cz"])
    if missing:
        L.append("**Not launched:** some results are missing (failed or still running jobs). Decide by hand.")
        jobs, decision = [], {"status": "missing"}
    elif not any(combo):
        L.append("**Not launched:** no add-on helps; v8r (3 seeds) stays the candidate.")
        jobs, decision = [], {"status": "none"}
    else:
        kind = KIND[(keep["rel"], keep["route"])]
        run = EXISTING.get(combo, "v9f")
        seeds = (6, 7) if run in EXISTING.values() else (5, 6, 7)
        L.append(f"**Launched:** `{run}` = v8r + " + ", ".join(lbl for k, lbl, *_ in TESTS if keep[k])
                 + f" (kind `{kind}`), seeds {', '.join(map(str, seeds))}.")
        jobs = launch(run, kind, keep["cz"], seeds, args.dry_run)
        decision = {"status": "launched", "run": run, "kind": kind, "seeds": list(seeds), "keep": keep}
    if jobs:
        L += ["", "| run | split | partition | job |", "| --- | --- | --- | --- |"]
        L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in jobs]
    L.append("")
    if not args.dry_run:
        OUT_MD.write_text("\n".join(L))
        OUT_JSON.write_text(json.dumps(decision, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
