#!/usr/bin/env python3
"""v18 (run v8kdt10): v17 (v8kdg10, generalization_v10) fine-tuned 20 epochs with the T1 / T2 / gate-time change loss
(--tvar-path, labels from compile_check/esp_tvar_train.py), same recipe as v17 otherwise; 4 splits × seeds 5–7, one
CPU job each.  Then the ESP evaluation and the T / duration tracking test.
    python launch_v18.py --after <merge job> [--dry-run]
v18b (run v8kdt10b): same with a lighter change loss and a stronger F invariance
    python launch_v18.py --run v8kdt10b --tvar-lambda 0.1 --tvar-f-lambda 3"""
import argparse, json, shlex, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--after", default=""); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--exclude", default="compute-3-1,compute-7-5,compute-7-6")
ap.add_argument("--run", default="v8kdt10"); ap.add_argument("--tvar-lambda", default="1"); ap.add_argument("--tvar-f-lambda", default="1")
a = ap.parse_args()
plan = json.loads((HERE / "plan.json").read_text())
rows = [r for r in plan if r["run"] in ("v8kdg10", "v8kdg10_s6", "v8kdg10_s7")]
ids = []
for r in rows:
    argv = list(r["argv"])
    i = argv.index("--init-from"); argv[i + 1] = r["out"]                   # start from v17 itself
    i = argv.index("--results-dir"); argv[i + 1] = argv[i + 1].replace("v8kdg10", a.run)
    i = argv.index("--epochs"); argv[i + 1] = "20"
    argv += ["--tvar-path", "evaluations/compile_check/results", "--tvar-lambda", a.tvar_lambda, "--tvar-f-lambda", a.tvar_f_lambda]
    name = f"v18_{r['run'].replace('v8kdg10', a.run)}_{r['split']}"
    wrap = ("export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; bash evaluations/retrain/run_one.sh "
            + shlex.join(["python3", "-u"] + argv))
    dep = f"--dependency=afterok:{a.after}" if a.after else ""
    cmd = (f"sbatch --parsable {dep} -p cpu_sapphire --exclude={a.exclude} -c 8 --mem=10G -t 03:00:00 --job-name={name} "
           f"-o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(wrap)}")
    if a.dry_run:
        print(cmd[:400]); continue
    ids.append(subprocess.check_output(cmd, shell=True, cwd=REPO, text=True).strip())
if ids:
    env = 'eval "$($HOME/miniforge3/bin/conda shell.bash hook)"; conda activate pred-distr; export PYTHONPATH=$PWD/src OMP_NUM_THREADS=8; '
    ev = (env + f"python3 -u evaluations/esp/esp_eval.py --run {a.run} --kind sinkhorn_rf_route_czc_esps --root evaluations/generalization_v10/results; "
          "python3 -u evaluations/esp/esp_ttrack.py --tag _v10_v18")
    e = subprocess.check_output(f"sbatch --parsable --dependency=afterany:{':'.join(ids)} -p cpu_sapphire --exclude={a.exclude} "
                                f"-c 8 --mem=14G -t 01:00:00 --job-name=eval_{a.run} -o evaluations/retrain/logs/%x_%j.log "
                                f"--wrap={shlex.quote(ev)}", shell=True, cwd=REPO, text=True).strip()
    line = f"v18 {a.run} (tvar {a.tvar_lambda}, F {a.tvar_f_lambda}) trainings {':'.join(ids)} eval {e} (after {a.after})"
    print(line); (HERE / "eval_jobs.txt").open("a").write(line + "\n")
