#!/usr/bin/env python3
"""Wait until every retrained model of the given runs exists, then submit an evaluation script.

The trainings may come from GPU packs or from CPU duplicates (whichever started first; the split lock makes the
other skip), so evaluations wait for the files, not for particular jobs.  A model counts as done when its
model.pth and metrics.json exist (run_experiment writes metrics.json last).
    python wait_models.py --runs v8k,v8kdc10 --then "sbatch ... evaluations/retrain/eval_best.sh" [--every 120] [--max-hours 12]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="base run names (all seeds), or 'tier1'")
    ap.add_argument("--then", required=True)
    ap.add_argument("--every", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=12)
    args = ap.parse_args()
    rows = json.loads((HERE / "plan.json").read_text())
    if args.runs == "tier1":
        want = [r for r in rows if r["tier"] == 1]
    else:
        names = set(args.runs.split(","))
        want = [r for r in rows if re.sub(r"_s[567]$", "", r["run"]) in names]
    paths = [REPO / r["out"] for r in want]
    t0 = time.time()
    while True:
        done = [p for p in paths if p.is_file() and (p.parent / "metrics.json").is_file()]
        print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(paths)} models", flush=True)
        if len(done) == len(paths):
            break
        if time.time() - t0 > args.max_hours * 3600:
            print("timeout; submitting anyway with the models available", flush=True)
            break
        time.sleep(args.every)
    out = subprocess.run(args.then, shell=True, cwd=REPO, capture_output=True, text=True)
    print("submitted:", out.stdout.strip(), out.stderr.strip(), flush=True)


if __name__ == "__main__":
    main()
