#!/usr/bin/env python3
"""Summarise ``compile_check.py`` output: label noise, best-of-K and model layouts.

Only compilations that are both valid on the device and equivalent to the original
circuit are used; the others are counted and reported.

    python analyze.py results/qaoa_seeds [results/qaoa_layouts]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")


def load(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        for f in sorted(d.glob("task*.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def main() -> None:
    rows = load([Path(a) for a in sys.argv[1:]])
    bad = [r for r in rows if "error" in r or not (r.get("valid") and r.get("equivalent"))]
    ok = [r for r in rows if r not in bad]
    print(f"compilations: {len(rows)}   valid+equivalent: {len(ok)}   rejected: {len(bad)}")
    for r in bad[:5]:
        print("  rejected:", r["circuit"], r["device"], r["variant"], r.get("error"),
              "valid", r.get("valid"), "overlap", r.get("overlap"))

    F = defaultdict(dict)       # (circuit, device) -> variant -> fidelity
    label = {}
    for r in ok:
        F[(r["circuit"], r["device"])][r["variant"]] = r["fidelity"]
        label[(r["circuit"], r["device"])] = r["label"]
    keys = sorted(F)
    nq = {k: int(re.search(r"_q(\d+)_", k[0]).group(1)) for k in keys}
    groups = sorted({re.sub(r"_s\d+$", "", v) for k in keys for v in F[k]})

    def best(k, g):
        vals = [f for v, f in F[k].items() if re.sub(r"_s\d+$", "", v) == g]
        return max(vals) if vals else None

    print("\nLabel noise: spread of the L2 seeds around the label (same pipeline, different seed)")
    sel = [k for k in keys if sum(v.startswith("L2_") for v in F[k]) >= 2]
    l2 = {k: np.array([f for v, f in F[k].items() if v.startswith("L2_")]) for k in sel}
    for lo, hi in ((2, 4), (5, 8), (9, 13), (14, 20)):
        ks = [k for k in sel if lo <= nq[k] <= hi]
        if not ks:
            continue
        lab = np.array([label[k] for k in ks])
        print(f"  q{lo:>2}-{hi:<2} n={len(ks):4d}  mean label {lab.mean():.3f}  "
              f"seed std {np.mean([l2[k].std() for k in ks]):.4f}  "
              f"max-min {np.mean([np.ptp(l2[k]) for k in ks]):.4f}  "
              f"best L2 - label {np.mean([l2[k].max() - label[k] for k in ks]):+.4f}")

    print("\nBest of each variant group vs label (mean over circuit x device; same pairs)")
    for lo, hi in ((2, 4), (5, 8), (9, 13), (14, 20), (2, 20)):
        ks = [k for k in keys if lo <= nq[k] <= hi and all(best(k, g) is not None for g in groups)]
        if not ks:
            continue
        lab = np.array([label[k] for k in ks])
        cells = "  ".join(f"{g}: {np.mean([best(k, g) for k in ks]) - lab.mean():+.4f}" for g in groups)
        print(f"  q{lo:>2}-{hi:<2} n={len(ks):4d} label {lab.mean():.3f} | {cells}")

    print("\nBest device per circuit: how often it changes vs the label's choice")
    circuits = sorted({k[0] for k in keys})
    for g in groups:
        flips, n = 0, 0
        for c in circuits:
            if not all((c, d) in F for d in DEVICES):
                continue
            b = [best((c, d), g) for d in DEVICES]
            if None in b:
                continue
            lab = [label[(c, d)] for d in DEVICES]
            if max(lab) < 1e-3:   # all ~0: the choice is meaningless
                continue
            n += 1
            flips += int(np.argmax(b) != np.argmax(lab))
        if n:
            print(f"  {g:22s} {flips}/{n} circuits ({flips / n:.0%})")

    print("\nModel layouts: fraction of (circuit, device) where the layout beats the label / best-of-K")
    model_groups = [g for g in groups if not g.startswith(("L2", "L3"))]
    for g in model_groups:
        ks = [k for k in keys if best(k, g) is not None and label[k] > 1e-3]
        if not ks:
            continue
        bk = [max(f for v, f in F[k].items() if v.startswith(("L2_", "L3_"))) for k in ks]
        m = [best(k, g) for k in ks]
        print(f"  {g:22s} n={len(ks):4d}  > label {np.mean([a > label[k] + 1e-9 for a, k in zip(m, ks)]):.0%}"
              f"  > best-of-K {np.mean([a > b + 1e-9 for a, b in zip(m, bk)]):.0%}"
              f"  mean (layout - label) {np.mean([a - label[k] for a, k in zip(m, ks)]):+.4f}")


if __name__ == "__main__":
    main()
