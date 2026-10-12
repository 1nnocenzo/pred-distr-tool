#!/usr/bin/env python3
"""v19 (run v8kdf10): v17 (v8kdg10, generalization_v10) with the formula ESP head (kind ``*_espf``: the MLP reads the
schedule at the reference gate times, the formula the real ones; per-qubit decay; hard placement; coupler CZ times),
fine-tuned 20 epochs from v17 with v17's own recipe and labels.  No T / gate-time variants in training: the tracking
test stays a clean test.  4 splits × seeds 5–7, one CPU job each; then the ESP evaluation, the duration report, the
T / duration tracking test and the placement accuracy.
    python launch_v19.py [--dry-run]"""
import argparse, json, shlex, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--exclude", default="compute-3-1,compute-7-5,compute-7-6")
ap.add_argument("--run", default="v8kdf10"); ap.add_argument("--partition", default="cpu_sapphire")
a = ap.parse_args()
plan = json.loads((HERE / "plan.json").read_text())
rows = [r for r in plan if r["run"] in ("v8kdg10", "v8kdg10_s6", "v8kdg10_s7")]
ids = []
for r in rows:
    argv = list(r["argv"])
    i = argv.index("--model"); argv[i + 1] = argv[i + 1].replace("_esps", "_espf")
    i = argv.index("--init-from"); argv[i + 1] = r["out"]                   # start from v17 itself
    i = argv.index("--results-dir"); argv[i + 1] = argv[i + 1].replace("v8kdg10", a.run)
    i = argv.index("--epochs"); argv[i + 1] = "20"
    name = f"v19_{r['run'].replace('v8kdg10', a.run)}_{r['split']}"
    wrap = ("export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; bash evaluations/retrain/run_one.sh "
            + shlex.join(["python3", "-u"] + argv))
    cmd = (f"sbatch --parsable -p {a.partition} --exclude={a.exclude} -c 8 --mem=10G -t 03:00:00 --job-name={name} "
           f"-o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(wrap)}")
    if a.dry_run:
        print(cmd[:600]); continue
    ids.append(subprocess.check_output(cmd, shell=True, cwd=REPO, text=True).strip())
if ids:
    env = ('eval "$($HOME/miniforge3/bin/conda shell.bash hook)"; conda activate pred-distr; '
           'export PYTHONPATH=$PWD/src OMP_NUM_THREADS=8 GS_RESULTS_ROOT=generalization_v10; ')
    ev = (env + f"python3 -u evaluations/esp/esp_eval.py --run {a.run} --kind sinkhorn_rf_route_czc_espf "
          "--root evaluations/generalization_v10/results; "
          f"python3 -u evaluations/esp/esp_duration.py --runs v8kdg10,{a.run} --tag _v10_v19; "
          "python3 -u evaluations/esp/esp_ttrack.py --tag _v10_v19; "
          f"python3 -u evaluations/esp/placement_acc.py --run {a.run}")
    e = subprocess.check_output(f"sbatch --parsable --dependency=afterany:{':'.join(ids)} -p {a.partition} "
                                f"--exclude={a.exclude} -c 8 --mem=14G -t 02:00:00 --job-name=eval_{a.run} "
                                f"-o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(ev)}",
                                shell=True, cwd=REPO, text=True).strip()
    line = f"v19 {a.run} (formula ESP head) trainings {':'.join(ids)} eval {e}"
    print(line); (HERE / "eval_jobs.txt").open("a").write(line + "\n")
