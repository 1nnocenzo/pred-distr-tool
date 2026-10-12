#!/usr/bin/env python3
"""v20 (run v8kdp10): v19 (v8kdf10) + an ESP-only placement (kind ``*_espp``) trained with cross-entropy on the
compiler's exact layout (--esp-place-lambda 1); its inputs are detached, so F's placement does not move.  20 epochs from
each split's v19 model, each job after that v19 training (--after: the 12 v19 job ids in launch_v19's order); then the
same evaluations as v19.
    python launch_v20.py --after id1:id2:...:id12 [--dry-run]"""
import argparse, json, shlex, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--exclude", default="compute-3-1,compute-7-5,compute-7-6")
ap.add_argument("--run", default="v8kdp10"); ap.add_argument("--after", default=""); ap.add_argument("--partition", default="cpu_sapphire")
a = ap.parse_args()
plan = json.loads((HERE / "plan.json").read_text())
rows = [r for r in plan if r["run"] in ("v8kdg10", "v8kdg10_s6", "v8kdg10_s7")]
ids = []
after = a.after.split(":") if a.after else []
for j, r in enumerate(rows):
    argv = list(r["argv"])
    i = argv.index("--model"); argv[i + 1] = argv[i + 1].replace("_esps", "_espp")
    i = argv.index("--init-from"); argv[i + 1] = r["out"].replace("/v8kdg10", "/v8kdf10")   # from v19
    i = argv.index("--results-dir"); argv[i + 1] = argv[i + 1].replace("v8kdg10", a.run)
    i = argv.index("--epochs"); argv[i + 1] = "20"
    argv += ["--esp-place-lambda", "1"]
    name = f"v20_{r['run'].replace('v8kdg10', a.run)}_{r['split']}"
    wrap = ("export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; bash evaluations/retrain/run_one.sh "
            + shlex.join(["python3", "-u"] + argv))
    dep = f"--dependency=afterok:{after[j]} " if after else ""
    cmd = (f"sbatch --parsable {dep}-p {a.partition} --exclude={a.exclude} -c 8 --mem=10G -t 03:00:00 --job-name={name} "
           f"-o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(wrap)}")
    if a.dry_run:
        print(cmd[:600]); continue
    ids.append(subprocess.check_output(cmd, shell=True, cwd=REPO, text=True).strip())
if ids:
    env = ('eval "$($HOME/miniforge3/bin/conda shell.bash hook)"; conda activate pred-distr; '
           'export PYTHONPATH=$PWD/src OMP_NUM_THREADS=8 GS_RESULTS_ROOT=generalization_v10; ')
    ev = (env + f"python3 -u evaluations/esp/esp_eval.py --run {a.run} --kind sinkhorn_rf_route_czc_espp "
          "--root evaluations/generalization_v10/results; "
          f"python3 -u evaluations/esp/esp_duration.py --runs v8kdg10,v8kdf10,{a.run} --tag _v10_v20; "
          "python3 -u evaluations/esp/esp_ttrack.py --tag _v10_v20; "
          f"python3 -u evaluations/esp/placement_acc.py --run {a.run}")
    e = subprocess.check_output(f"sbatch --parsable --dependency=afterany:{':'.join(ids)} -p {a.partition} "
                                f"--exclude={a.exclude} -c 8 --mem=14G -t 02:00:00 --job-name=eval_{a.run} "
                                f"-o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(ev)}",
                                shell=True, cwd=REPO, text=True).strip()
    line = f"v20 {a.run} (ESP-only placement) after {a.after} trainings {':'.join(ids)} eval {e}"
    print(line); (HERE / "eval_jobs.txt").open("a").write(line + "\n")
