#!/usr/bin/env python3
"""Collect the v24 routing-stage results into results/report_v24.md.

Rows: v24 seed 5 (mean rates, and `_mc` = sampled SWAPs), v24b over seeds (mean ± std), and the scale diagnostics
(results/diag_scale_*.json). Errors are median |pred/true − 1| on routed cases; "vs Sabre" = the training target
(Sabre's own compilation), "vs dataset" = esp_labels.pt, the target of the v19i–v22 scorecard.
    python report_v24.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SPLITS = ("control", "qaoa", "variational", "loso_dense-complete", "loso_layered-seqdense")


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(vals, pct=True):
    v = [x for x in vals if x is not None]
    if not v:
        return "–"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.2f}")
    return f(v[0]) if len(v) == 1 else f"{f(np.mean(v))} ± {f(np.std(v)).rstrip('%')}" + ("%" if pct else "")


def model_rows():
    runs = defaultdict(lambda: defaultdict(list))          # (label) -> split -> [eval dict]
    for p in sorted(list(RES.glob("v24*/eval_*.json")) + list(RES.glob("v25*/eval_*.json")) + list(RES.glob("v26*/eval_*.json")) + list(RES.glob("v27*/eval_*.json")) + list(RES.glob("v28*/eval_*.json")) + list(RES.glob("v29*/eval_*.json"))):
        m = re.match(r"eval_(.+?)(_mc)?\.json$", p.name)
        if not m or m.group(1) not in SPLITS:
            continue
        run = re.sub(r"_s\d+$", "", p.parent.name)
        run = next((r for r in ("v29", "v28", "v27", "v26", "v25", "v24b") if run.startswith(r)), "v24")
        mc = bool(m.group(2)) or (run in ("v24b", "v25", "v26", "v27", "v28", "v29"))
        runs[(run, mc)][m.group(1)].append(json.loads(p.read_text()))
    return runs


def v29_section(runs) -> list[str]:
    """v29 (unified.py): one model for F, duration and ESP; ESP / F / device choice and the end-to-end time."""
    by_split = runs.get(("v29", True), {})
    if not by_split:
        return []
    L = ["", "## v29: one model (F backbone v8kdfi10 + route stage + ESP on the same placement)", "",
         "abs Δlog ESP where the true ESP > 1e-3 (as the earlier ESP reports); routed / all = by the label's routing flag. "
         "References on routed cases: v8kdfi10 abs Δlog ESP 0.53 (qaoa) / 0.65 (variational) / 0.30 (control) "
         "(esp/results/report_esp_routed_v10_v19i.md); light compile 0.29, compiler seed floor 0.20 "
         "(light_compile/results/report.md).", "",
         "| split | seeds | R² F | R² ESP | abs Δlog ESP, ESP > 1e-3: all | routed | routing-free | rel. abs Δlog ESP, routed (all ESP) | "
         "right device by ESP | ESP regret | routed duration vs dataset (sampled) | switch agreement |",
         "|" + " --- |" * 12]
    for s in SPLITS:
        ev = by_split.get(s, [])
        if not ev:
            continue
        e = lambda grp, k: fmt([g(x, "esp", grp, k) for x in ev], False)
        L.append(f"| {s} | {len(ev)} | {e('all', 'R2 F')} | {e('all', 'R2 ESP')} | {e('all', 'abs dlog ESP median (ESP>1e-3)')} | "
                 f"{e('routed', 'abs dlog ESP median (ESP>1e-3)')} | {e('routing-free', 'abs dlog ESP median (ESP>1e-3)')} | "
                 f"{fmt([g(x, 'esp', 'routed', 'rel abs dlog ESP median (all)') for x in ev])} | "
                 f"{fmt([g(x, 'device_by_esp', 'right') for x in ev])} | {fmt([g(x, 'device_by_esp', 'regret') for x in ev], False)} | "
                 f"{fmt([g(x, 'routed', 'vs_dataset', 'pred_mc', 'err') for x in ev])} | {fmt([g(x, 'switch', 'agreement') for x in ev])} |")
    L += ["", "### Time per circuit, whole prediction (3 devices, batch 1, one thread, median ms)", "",
          "features = greedy router on the 3 chips + wire features; forward = F, placement, switch, route stage, ESP (one "
          "model); mc = sampled schedule. Light / full compile: light_compile/results/report.md (other circuits, same bins).", "",
          "| split | 2q gates | n | features | forward | mc | total | light compile | full compile |", "|" + " --- |" * 9]
    for s in SPLITS:
        for x in by_split.get(s, [])[:1]:
            for k, t in (g(x, "timing", "by_2q") or {}).items():
                L.append(f"| {s} | {k} | {t['n']} | {t['feat_ms']:.1f} | {t['forward_ms']:.1f} | {t['mc_ms']:.1f} | "
                         f"**{t['total_ms']:.1f}** | {t['light_compile_ms']:.1f} | {t['full_compile_ms']:.1f} |")
    return L


def main():
    L = ["# v24 routing stage: routed-circuit duration", "",
         "Median |pred/true − 1| on routed cases [n seeds]; bias = median pred/true. *mean rates* = schedule of the "
         "predicted SWAP rates; *sampled* = median schedule over Poisson samples of the SWAPs. "
         "vs dataset = esp_labels.pt (the v19i–v22 target).", ""]
    runs = model_rows()
    hdr = "| model | split | seeds | vs Sabre: mean rates | sampled | sampled, no stage 2 | bias (sampled) | with true SWAPs | vs dataset: sampled | Sabre vs dataset | SWAP count ratio (median / mean) | routing-free |"
    L += [hdr, "|" + " --- |" * (hdr.count("|") - 1)]
    for (run, mc), by_split in sorted(runs.items()):
        for s in SPLITS:
            ev = by_split.get(s, [])
            if not ev:
                continue
            col = lambda *k: [g(e, "routed", *k) for e in ev]
            L.append(f"| {run}{' (sampled)' if mc and run == 'v24' else ''} | {s} | {len(ev)} | {fmt(col('pred', 'err'))} | "
                     f"{fmt(col('pred_mc', 'err'))} | {fmt(col('pred1_mc', 'err'))} | {fmt(col('pred_mc', 'bias'), False)} | "
                     f"{fmt(col('oracle_sw', 'err'))} | {fmt(col('vs_dataset', 'pred_mc', 'err'))} | "
                     f"{fmt(col('vs_dataset', 'sabre_final', 'err'))} | {fmt(col('swaps', 'bias'), False)} / {fmt(col('swaps', 'mean_ratio'), False)} | "
                     f"{fmt([g(e, 'routing-free', 'pred', 'err') for e in ev])} |")
    L += v29_section(runs)
    diags = sorted(RES.glob("diag_scale_*.json"))
    diags = [p for p in diags if "smoke" not in p.name]
    if diags:
        L += ["", "## Scale diagnostic (seed-5 checkpoints)", "",
              "| split | set | median true SWAPs | median pred | mean pred / mean true | share of pred on zero slots | "
              "mean rates | sampled (median) | pred SWAPs + true waits | true SWAPs + pred waits | true SWAPs + waits |",
              "|" + " --- |" * 11]
        for p in diags:
            d = json.loads(p.read_text())
            for st in ("test", "train"):
                t, du = g(d, st, "totals"), g(d, st, "duration_vs_final")
                if not t:
                    continue
                e = lambda k: f"{100 * du[k]['err']:.1f}% ({du[k]['bias']:.2f})"
                L.append(f"| {d['split']} | {st} | {t['median_true']:.0f} | {t['median_pred']:.0f} | {t['mean_pred/mean_true']:.2f} | "
                         f"{t['share_pred_on_zero_slots']:.2f} | {e('mean')} | {e('mc_med')} | {e('s_pred_w_true')} | "
                         f"{e('s_true_w_pred')} | {e('oracle_sw')} |")
        L += ["", "Duration cells: median error (bias). Per-slot calibration tables are in the json files."]
    out = RES / "report_v24.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
