#!/usr/bin/env python3
"""v21: duration from the right counts.  v19i (v8kdfi10) fine-tuned 20 epochs from its own split models, v19i's recipe,
with one or both of
  --r-weight W        weight of the 1-qubit (r) count in the count loss (v19i: 0.1, CZ 1): the schedule that gives the
                      duration is built from these counts, and the duration loss never reaches them (dag_schedule is
                      detached), so this is the only signal that makes the r counts right
  --dur-corr L        --dur-corr-lambda: L2 penalty on the learned log-duration correction, so the duration stays the
                      count-based schedule instead of a correction fitted to the training families
Then the standard ESP evaluation, duration report, T / gate-time tracking test, count oracle and the F calibration
test (merged into results_k_v10), all chained.  4 splits × seeds 5–7, one CPU job each.
    python launch_v21.py --run v8kdfr10 --r-weight 1 --dur-corr 1 [--dry-run]
v21t (run v8kdft10, kind *_espft): routing-free circuits are scheduled with the fixed per-gate decomposition table and
no learned correction (count oracle, esp_count_oracle.py: 0.5% median error on variational / control), routed ones
keep the learned counts and correction; with --r-weight 1.
    python launch_v21.py --run v8kdft10 --kind sinkhorn_rf_route_czc_espft --r-weight 1 --label v21t"""
import argparse, json, shlex, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--exclude", default="compute-3-1,compute-7-5,compute-7-6")
ap.add_argument("--run", required=True); ap.add_argument("--partition", default="cpu_sapphire")
ap.add_argument("--r-weight", type=float, default=None); ap.add_argument("--dur-corr", type=float, default=None)
ap.add_argument("--label", default="v21")
ap.add_argument("--init-run", default="v8kdfi10", help="run whose split models are the starting point")
ap.add_argument("--extra-train", default=None, help="v22: synthetic_routing/results (training side only)")
ap.add_argument("--after", default="", help="sbatch dependency (afterok) for the trainings, e.g. the synthetic merge job")
ap.add_argument("--guard-for", default="", help="colon list of this run's training job ids (launch order): submit one "
                "resume job per split after it (afterany); a finished split is skipped, a timed-out one resumes from its checkpoint")
ap.add_argument("--time", default="03:00:00")
ap.add_argument("--nodes", default="", help="comma list: pin the trainings round-robin to these idle nodes (-w)")
ap.add_argument("--kind", default="sinkhorn_rf_route_czc_espfi",
                help="*_espft (v21t): fixed decomposition table for the schedule of routing-free circuits")
a = ap.parse_args()
KIND = a.kind
plan = json.loads((HERE / "plan.json").read_text())
rows = [r for r in plan if r["run"] in ("v8kdi10", "v8kdi10_s6", "v8kdi10_s7")]
ids = []
guard = [x for x in a.guard_for.split(":") if x]
nodes = [n for n in a.nodes.split(",") if n]
for k, r in enumerate(rows):
    argv = list(r["argv"])
    i = argv.index("--model"); argv[i + 1] = KIND
    init = r["out"].replace("/v8kdi10", "/" + a.init_run)                     # start from that run's split model
    i = argv.index("--init-from"); argv[i + 1] = init
    i = argv.index("--results-dir"); argv[i + 1] = argv[i + 1].replace("v8kdi10", a.run)
    i = argv.index("--epochs"); argv[i + 1] = "20"
    if a.r_weight is not None:
        i = argv.index("--count-weights"); w = argv[i + 1].split(","); w[0] = str(a.r_weight); argv[i + 1] = ",".join(w)
    if a.dur_corr is not None:
        argv += ["--dur-corr-lambda", str(a.dur_corr)]
    if a.extra_train:
        argv += ["--extra-train", a.extra_train]
    assert a.after or (REPO / init).is_file(), init
    name = f"{a.label}_{r['run'].replace('v8kdi10', a.run)}_{r['split']}"
    wrap = ("export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; bash evaluations/retrain/run_one.sh "
            + shlex.join(["python3", "-u"] + argv))
    where = f"-w {nodes[k % len(nodes)]}" if nodes else f"--exclude={a.exclude}"
    dep = f"--dependency=afterany:{guard[k]} " if guard else (f"--dependency=afterok:{a.after} " if a.after else "")
    if guard:
        name = "g" + name
    cmd = (f"sbatch --parsable {dep}-p {a.partition} {where} -c 8 --mem=12G -t {a.time} --job-name={name} "
           f"-o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(wrap)}")
    if a.dry_run:
        print(cmd[:900]); continue
    ids.append(subprocess.check_output(cmd, shell=True, cwd=REPO, text=True).strip())
if ids and guard:
    line = f"{a.label} {a.run} resume guards {':'.join(ids)} (after {a.guard_for})"
    print(line); (HERE / "eval_jobs.txt").open("a").write(line + "\n")
elif ids:
    env = ('eval "$($HOME/miniforge3/bin/conda shell.bash hook)"; conda activate pred-distr; '
           'export PYTHONPATH=$PWD/src OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= GS_RESULTS_ROOT=generalization_v10; ')
    ev = (env + f"python3 -u evaluations/esp/esp_eval.py --run {a.run} --kind {KIND} "
          "--root evaluations/generalization_v10/results; "
          f"python3 -u evaluations/esp/esp_duration.py --runs v8kdi10,v8kdfi10,{a.run} --tag _v10_{a.run}; "
          f"python3 -u evaluations/esp/esp_ttrack.py --tag _v10_{a.run}; "
          f"python3 -u evaluations/esp/esp_count_oracle.py --runs {a.run} --tag _v10_{a.run}; "
          f"GS_CALIB_DIR=results_k_v10 GS_MODELS={a.run} bash -c 'cd evaluations/calibration_shift && "
          "python3 -u shift_k.py predict && python3 -u shift_k.py report'")
    e = subprocess.check_output(f"sbatch --parsable --dependency=afterany:{':'.join(ids)} -p {a.partition} "
                                f"--exclude={a.exclude} -c 8 --mem=14G -t 03:00:00 --job-name=eval_{a.run} "
                                f"-o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(ev)}",
                                shell=True, cwd=REPO, text=True).strip()
    line = (f"{a.label} {a.run} ({KIND}, from v19i; r count weight {a.r_weight}, duration-correction penalty {a.dur_corr}, "
            f"extra train {a.extra_train}) "
            f"trainings {':'.join(ids)} eval {e}")
    print(line); (HERE / "eval_jobs.txt").open("a").write(line + "\n")
