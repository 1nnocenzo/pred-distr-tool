#!/usr/bin/env python3
"""Old (qaoa without measurements) vs retrained (generalization_v10) models, on the standard evaluation.

Per run (seeds 5-7 pooled as mean ± std, number of seeds in brackets):
  generalization: test R² of F, device-choice regret, right-device share per standard split (metrics.json);
  calibration test: local / moderate / large tracking and R² after the change (results_k vs results_k_v10 report);
  ESP (if the ESP reports exist): R² ESP, duration error, ESP regret (esp/results/esp_eval_<run>[_v10].json).
    python compare_v10.py [--runs v8k,v8kdc10,v8kdg10,v8kdi10]     # -> results/compare_v10.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
NEW = EVAL / "generalization_v10/results"
SPLITS = [("LOFO qaoa", "leave_one_family_out/qaoa"), ("LOFO qnn", "leave_one_family_out/qnn"),
          ("LOGO variational", "leave_one_group_out/variational"), ("control", "random_control/random_seed{s}")]
GROUPS = ("local", "moderate", "large")


def old_root(run: str) -> Path:
    plan = json.loads((HERE / "plan.json").read_text())
    for r in plan:
        if re.sub(r"_s[567]$", "", r["run"]) == run:
            return EVAL.parent / r["orig_root"]
    return EVAL / "generalization_v9/results"


def gen(root: Path, run: str):
    out = defaultdict(lambda: defaultdict(list))
    for s in (5, 6, 7):
        name = run if s == 5 else f"{run}_s{s}"
        for lab, sd in SPLITS:
            f = root / name / sd.format(s=s) / "metrics.json"
            if not f.is_file() and s == 5:
                f = root / f"{run}_s5" / sd.format(s=s) / "metrics.json"
            if f.is_file():
                t = json.loads(f.read_text())["test"]
                out[lab]["R²"].append(t["r2"]); out[lab]["regret"].append(t["fidelity_regret_mean"])
                out[lab]["right device"].append(t["device_choice_accuracy"])
    return out


def calib(rk: Path, run: str):
    rep = rk / "report.md"
    if not rep.is_file():
        return {}
    txt = rep.read_text().split("## Summary per group of variants")[-1]
    out = {}
    for sec in re.split(r"\n### ", txt)[1:]:
        g = next((g for g in GROUPS if sec.startswith(g)), None)
        if g is None:
            continue
        vals = [(float(a), float(b)) for a, b in re.findall(rf"^{re.escape(run)}@s\d\s+R² ([\d.]+)\s+worst [\d.]+\s+tracking ([+-][\d.]+)", sec, re.M)]
        if vals:
            out[g] = (np.mean([v[1] for v in vals]), np.mean([v[0] for v in vals]), len(vals))
    return out


def ms(v):
    v = np.array(v)
    return "–" if not len(v) else (f"{v.mean():.3f} ± {v.std():.3f} ({len(v)})" if len(v) > 1 else f"{v.mean():.3f} (1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="v8k,v8kdc10,v8kdg10,v8kdi10")
    args = ap.parse_args()
    L = ["# Retrained on qaoa with final measurements (generalization_v10) vs the original models", "",
         "Mean ± std over seeds (n). Old = trained and tested on qaoa without measurements. LOFO qaoa: the test circuits "
         "themselves changed (now measured), so old and new numbers there measure different circuits.", ""]
    for run in args.runs.split(","):
        o, n = gen(old_root(run), run), gen(NEW, run)
        L += [f"## {run}", "", "| split | metric | old | new |", "| --- | --- | --- | --- |"]
        for lab, _ in SPLITS:
            for k in ("R²", "regret", "right device"):
                L.append(f"| {lab} | {k} | {ms(o[lab][k])} | {ms(n[lab][k])} |")
        co, cn = calib(EVAL / "calibration_shift/results_k", run), calib(EVAL / "calibration_shift/results_k_v10", run)
        if co or cn:
            L += ["", "| calibration test | old tracking | new tracking | old R² after change | new R² after change |", "| --- | --- | --- | --- | --- |"]
            for g in GROUPS:
                a, b = co.get(g), cn.get(g)
                f = lambda x, i: f"{x[i]:.3f}" if x else "–"
                L.append(f"| {g} | {f(a, 0)} | {f(b, 0)} | {f(a, 1)} | {f(b, 1)} |")
        eo, en = EVAL / f"esp/results/esp_eval_{run}.json", EVAL / f"esp/results/esp_eval_{run}_v10.json"
        if en.is_file():
            ro, rn = (json.loads(eo.read_text())["res"] if eo.is_file() else {}), json.loads(en.read_text())["res"]
            L += ["", "| ESP | split | old | new |", "| --- | --- | --- | --- |"]
            for k in ("R² ESP", "duration error (median)", "depth error (median)", "ESP regret, model", "R² F"):
                for lab, _ in SPLITS:
                    vo = [m[k] for m in ro.get(lab, []) if k in m]; vn = [m[k] for m in rn.get(lab, []) if k in m]
                    L.append(f"| {k} | {lab} | {ms(vo)} | {ms(vn)} |")
        L.append("")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/compare_v10.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
