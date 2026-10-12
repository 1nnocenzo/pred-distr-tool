#!/usr/bin/env python3
"""Local tracking of each seed vs the mean of the seeds' log F (a seed ensemble), calibration-shift test.

Tracking as in run_shift.py: mean over (local variant, device) of corr(predicted Δlog F, true Δlog F),
circuits with F > 0.01 on both calibrations, truth = best of 5 compilations.
    python ensemble_tracking.py --models v8kd40,v8kdl40
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np

RK = Path(__file__).resolve().parent / os.environ.get("GS_CALIB_DIR", "results_k")
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LOCAL = ("shuffle", "jitterA", "jitterB", "jitterS")


def tracking(T, pred):
    best = lambda n, k: max([f for f in T[n][k] if f is not None], default=None)
    cs = []
    for d in DEVS:
        for v in LOCAL:
            dp, dt = [], []
            for n in T:
                fo, fv = best(n, f"{d}/orig"), best(n, f"{d}/{v}")
                if fo is None or fv is None or fo <= 0.01 or fv <= 0.01 or n not in pred:
                    continue
                dp.append(pred[n][f"{d}/{v}"] - pred[n][f"{d}/orig"])
                dt.append(np.log(fv / fo))
            cs.append(np.corrcoef(dp, dt)[0, 1])
    return float(np.mean(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8kd40,v8kdl40")
    args = ap.parse_args()
    T = json.loads((RK / "truth.json").read_text()); P = json.loads((RK / "predictions.json").read_text())
    log = lambda m: {n: {k: float(np.log(max(x, 1e-12))) for k, x in v.items()} for n, v in P[m].items()}
    L = ["| model | seeds | tracking per seed | mean | seed ensemble |", "| --- | --- | --- | --- | --- |"]
    for r in args.models.split(","):
        ms = [m for m in (f"{r}@s{s}" for s in (5, 6, 7)) if m in P]
        if not ms:
            continue
        lg = [log(m) for m in ms]
        single = [tracking(T, x) for x in lg]
        ens = {n: {k: float(np.mean([x[n][k] for x in lg])) for k in lg[0][n]} for n in lg[0] if all(n in x for x in lg)}
        L.append(f"| {r} | {len(ms)} | {' / '.join(f'{x:.3f}' for x in single)} | {np.mean(single):.3f} | {tracking(T, ens):.3f} |")
    (RK / "ensemble_tracking.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
