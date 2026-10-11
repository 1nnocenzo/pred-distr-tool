#!/usr/bin/env python3
"""Recompile only when needed: what the predictor saves after a calibration change.

Scenario (calibration-shift test: 457 unseen circuits, 3 devices, truth = best of 5 compilations):
each circuit runs on the device that was best under the old calibration; a new calibration arrives
(the same variant on all devices).  Which circuits must have their device choice re-checked, i.e. be
recompiled on every device?

  keep          never re-check: stay on the old device                            cost 0
  re-check all  recompile every circuit on every device                            cost 100%
  model switch  move to the device the model ranks first now (one compilation, as before)
  cautious      move only if the model predicts the new first device beats the old one by > δ
  model-guided  re-check only the circuits where the model predicts the best device changes or the
                predicted margin between its top two devices is below a threshold; the others keep
                their old device.  The threshold is swept: share re-checked vs fidelity lost.
  random        re-check the same share of circuits at random (mean of 200 draws)

Fidelity lost = mean over circuits of (true best F under the new calibration − true F of the device
used).  Reported per kind of change (local, moderate, large) and as a curve.
    python recompile_policy.py --models v8a_c001,v8kd40,v8kdl40 [--tag name]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RK = HERE / "results_k"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
KINDS = {"local": ("shuffle", "jitterA", "jitterB", "jitterS"),
         "moderate": ("x0.7", "x0.8", "x1.2", "x1.3"), "large": ("x0.5", "x2")}
SHARES = (0.05, 0.10, 0.20, 0.30, 0.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8a_c001,v8kd40,v8kdl40")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    T = json.loads((RK / "truth.json").read_text()); P = json.loads((RK / "predictions.json").read_text())
    best = lambda n, k: max([f for f in T[n][k] if f is not None], default=None)
    allv = [v for vs in KINDS.values() for v in vs]
    names = [n for n in T if all(best(n, f"{d}/{v}") is not None for d in DEVS for v in allv + ["orig"])]
    runs = args.models.split(",")
    seeds = {r: [m for m in (f"{r}@s{s}" for s in (5, 6, 7)) if m in P] for r in runs}
    rng = np.random.default_rng(0)
    out = {"n_circuits": len(names), "kinds": {}}
    L = ["# Recompile only when needed", "",
         f"{len(names)} unseen circuits × 3 devices; each circuit starts on the device that is truly best under the old "
         "calibration. Fidelity lost = true best F under the new calibration − true F of the device used. "
         "Re-checking a circuit = compiling it on every device again. Models: mean over the seeds listed.", ""]
    for kind, vs in KINDS.items():
        rows = []          # per (circuit, variant): true new F (3), old best index, per model: predicted new F (3)
        for n in names:
            old = int(np.argmax([best(n, f"{d}/orig") for d in DEVS]))
            for v in vs:
                rows.append((n, v, np.array([best(n, f"{d}/{v}") for d in DEVS]), old))
        tnew = np.stack([r[2] for r in rows]); old = np.array([r[3] for r in rows])
        oracle = tnew.max(1)
        loss_keep = float((oracle - tnew[np.arange(len(rows)), old]).mean())
        changed = float((tnew.argmax(1) != old).mean())
        res = {"keep": loss_keep, "best_changes": changed, "models": {}}
        # random re-check curve: re-checked circuits lose 0, others keep
        keep_loss_each = oracle - tnew[np.arange(len(rows)), old]
        res["random"] = {f"{s:.2f}": float(np.mean([keep_loss_each[rng.random(len(rows)) >= s].sum() / len(rows) for _ in range(200)]))
                         for s in SHARES}
        for r, ms in seeds.items():
            per = []
            for m in ms:
                pnew = np.stack([np.array([P[m][n][f"{d}/{v}"] for d in DEVS]) for n, v, _, _ in rows])
                pick = pnew.argmax(1)
                switch = float((oracle - tnew[np.arange(len(rows)), pick]).mean())
                gain = pnew[np.arange(len(rows)), pick] - pnew[np.arange(len(rows)), old]
                caut = {}
                for dlt in (0.005, 0.01, 0.02):
                    use = np.where(gain > dlt, pick, old)
                    caut[f"{dlt}"] = (float((oracle - tnew[np.arange(len(rows)), use]).mean()), float((use != old).mean()))
                srt = np.sort(pnew, 1)
                margin = srt[:, -1] - srt[:, -2]
                flag = pick != old
                # priority to re-check: predicted change first, then small predicted margin
                prio = np.where(flag, -1.0, margin)
                order = np.argsort(prio)
                curve = {}
                for s in SHARES:
                    k = int(round(s * len(rows)))
                    re = np.zeros(len(rows), bool); re[order[:k]] = True
                    curve[f"{s:.2f}"] = float(np.where(re, 0.0, keep_loss_each).mean())
                share_flag = float(flag.mean())
                loss_flag = float(np.where(flag, 0.0, keep_loss_each).mean())
                per.append({"switch": switch, "curve": curve, "flag_share": share_flag, "flag_loss": loss_flag, "caut": caut})
            res["models"][r] = {"switch": float(np.mean([p["switch"] for p in per])),
                                "flag_share": float(np.mean([p["flag_share"] for p in per])),
                                "flag_loss": float(np.mean([p["flag_loss"] for p in per])),
                                "curve": {s: float(np.mean([p["curve"][s] for p in per])) for s in per[0]["curve"]},
                                "cautious": {d: [float(np.mean([p["caut"][d][i] for p in per])) for i in (0, 1)] for d in per[0]["caut"]},
                                "seeds": len(per)}
        out["kinds"][kind] = res
        L += [f"## {kind.capitalize()} calibration changes ({', '.join(vs)})", "",
              f"The best device changes in {changed:.0%} of (circuit, change). Keeping the old device loses {loss_keep:.4f} of F on average.", "",
              "| policy | circuits re-checked | fidelity lost | vs keeping |", "| --- | --- | --- | --- |",
              f"| keep the old device | 0% | {loss_keep:.4f} | – |", "| re-check all | 100% | 0 | −100% |"]
        rel = lambda x: f"{x / loss_keep - 1:+.0%}" if loss_keep > 1e-4 else "–"
        for r, m in res["models"].items():
            L.append(f"| {r}: switch to its new first device (no re-check) | 0% | {m['switch']:.4f} | {rel(m['switch'])} |")
            for d, (lo, mv) in m["cautious"].items():
                L.append(f"| {r}: switch only if predicted gain > {d} (moves {mv:.0%}) | 0% | {lo:.4f} | {rel(lo)} |")
            L.append(f"| {r}: re-check where it predicts a change | {m['flag_share']:.0%} | {m['flag_loss']:.4f} | {rel(m['flag_loss'])} |")
        L += ["", "Model-guided re-check of a fixed share of circuits (predicted changes first, then the smallest predicted margins), vs random:", "",
              "| share re-checked | random | " + " | ".join(res["models"]) + " |", "| --- | --- | " + " | ".join("---" for _ in res["models"]) + " |"]
        for s in SHARES:
            k = f"{s:.2f}"
            L.append(f"| {s:.0%} | {res['random'][k]:.4f} | " + " | ".join(f"{m['curve'][k]:.4f}" for m in res["models"].values()) + " |")
        L.append("")
    tag = f"_{args.tag}" if args.tag else ""
    (RK / f"recompile_policy{tag}.json").write_text(json.dumps(out, indent=1))
    (RK / f"recompile_policy{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
