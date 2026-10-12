#!/usr/bin/env python3
"""Retrain every model of the study on the regenerated data (qaoa with its final measurements).

Recipes come from the original runs themselves: every ``Command line:`` of the ``run.log`` files under
``evaluations/generalization_v*/results``.  For each (run, split) whose ``model.pth`` exists, the latest command that
trained it is kept, restricted to the standard splits (LOFO qaoa, LOFO qnn, LOGO variational, control), split
into one command per split, and pointed to ``evaluations/generalization_v10/results/<run>`` (``--init-from`` too,
so fine-tunings start from the retrained parents).  Levels follow ``--init-from`` (0 = from scratch); tier 1 =
the paper's models and the ESP candidates.

    python retrain_all.py plan                       # -> plan.json + summary (nothing submitted)
    python retrain_all.py submit --after JOBID[:JOBID] [--tier 1|2|all] [--per-pack 12] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "evaluations"
HERE = Path(__file__).resolve().parent
NEW_ROOT = "evaluations/generalization_v10/results"
STANDARD = {"lofo": {"qaoa", "qnn"}, "logo": {"variational"}}
EXP_DIR = {"lofo": "leave_one_family_out", "logo": "leave_one_group_out", "control": "random_control"}
TIER1 = {"v1", "v1_s5", "v4_phys", "v5_sinkhorn", "v8a_c001", "v8k", "v8kd40", "v8kdl40", "v8kdc10",
         "v8kde10", "v8kda10", "v8kdi10", "v8kdg10", "v8kds10",
         "v5_espa", "v8a_espa", "v8kd40_espa", "v8kdl40_espa", "v8kdc10_espa"}
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def base_run(run: str) -> str:
    return re.sub(r"_s[567]$", "", run)


def collect():
    """{(run, exp, split): (timestamp, argv)} latest command per trained split."""
    best = {}
    for log in sorted(EVAL.glob("generalization_v*/results/*/*/run.log")):
        if "generalization_v10" in str(log):
            continue
        for line in log.read_text(errors="replace").splitlines():
            if "Command line:" not in line:
                continue
            m = TS.match(line)
            ts = m.group(1) if m else ""
            argv = shlex.split(line.split("Command line:", 1)[1])
            if not argv or not argv[0].endswith("run_experiment.py") or len(argv) < 2:
                continue
            exp = argv[1]
            if exp not in EXP_DIR:
                continue
            opts = parse(argv[2:])
            rd = opts.get("--results-dir")
            if not rd:
                continue
            run = Path(rd).name
            seed = int(opts.get("--seed", 5))
            if exp == "control":
                splits = [f"random_seed{seed}"]
            else:
                key = "--families" if exp == "lofo" else "--groups"
                splits = [s for s in opts.get(key, "").split(",") if s in STANDARD[exp]]
            for sp in splits:
                d = REPO / rd / EXP_DIR[exp] / sp
                if not (d / "model.pth").is_file():
                    continue
                k = (rd, exp, sp)
                if k not in best or ts >= best[k][0]:
                    best[k] = (ts, argv)
    return best


def fill_missing(best):
    """Splits with a model.pth but no recorded command: rebuild from the same run's command on another split
    (same experiment), else from another seed of the same base run (seed, run suffix, control split changed)."""
    by_run = defaultdict(list)
    for (rd, exp, sp), v in best.items():
        by_run[rd].append((exp, sp, v))
    added = 0
    for mp in sorted(EVAL.glob("generalization_v*/results/*/*/*/model.pth")):
        if "generalization_v10" in str(mp):
            continue
        sp, expd, run, root = mp.parent.name, mp.parent.parent.name, mp.parents[2].name, mp.parents[3]
        exp = {v: k for k, v in EXP_DIR.items()}.get(expd)
        if exp is None or (exp != "control" and sp not in STANDARD[exp]) or (exp == "control" and not sp.startswith("random_seed")):
            continue
        rd = str(root.relative_to(REPO) / run)
        if (rd, exp, sp) in best:
            continue
        seed = int(sp.replace("random_seed", "")) if exp == "control" else int(re.search(r"_s(\d)$", run).group(1)) if re.search(r"_s\d$", run) else 5
        tmpl = next(((ts, argv) for e, s2, (ts, argv) in by_run.get(rd, []) if e == exp), None)
        if tmpl is None:                        # another seed of the same base run, same experiment
            for rd2, items in by_run.items():
                if Path(rd2).parent == Path(rd).parent and base_run(Path(rd2).name) == base_run(run):
                    tmpl = next(((ts, argv) for e, s2, (ts, argv) in items if e == exp), None)
                    if tmpl:
                        break
        if tmpl is None:
            print("no template for", rd, exp, sp)
            continue
        ts, argv = tmpl
        o = list(argv)
        for i, a in enumerate(o):
            if a == "--seed":
                o[i + 1] = str(seed)
            if a == "--results-dir":
                o[i + 1] = rd
            if a == "--init-from":
                parts = list(Path(o[i + 1]).parts)
                k = parts.index("results")
                parts[k + 1] = base_run(parts[k + 1]) + ("" if seed == 5 else f"_s{seed}")
                if parts[k + 2] == "random_control":
                    parts[k + 3] = f"random_seed{seed}"
                parts[k + 3] = parts[k + 3] if parts[k + 2] == "random_control" else sp
                o[i + 1] = str(Path(*parts))
        best[(rd, exp, sp)] = ("fill", o)
        added += 1
    print("filled from templates:", added)
    return best


def parse(args):
    out, i = {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[a] = args[i + 1]; i += 2
        else:
            out[a] = True; i += 1
    return out


def rewrite(rd: str, exp: str, sp: str, argv):
    """argv of one split, results and init-from in the new root."""
    run = Path(rd).name
    out, i, init = [argv[0], exp], 2, None
    args = argv[2:]
    skip = {"--results-dir", "--families", "--groups"}
    while i - 2 < len(args):
        a = args[i - 2]
        val = args[i - 1] if i - 1 < len(args) and not args[i - 1].startswith("--") else None
        if a in skip:
            i += 2 if val is not None else 1
            continue
        if a == "--init-from" and val is not None:
            p = Path(val)
            parts = p.parts
            k = parts.index("results")
            init = str(Path(NEW_ROOT, *parts[k + 1:]))
            out += [a, init]
        else:
            out += [a] + ([val] if val is not None else [])
        i += 2 if val is not None else 1
    fam = sp if exp != "control" else "x"
    out += ["--families", fam, "--groups", fam, "--results-dir", f"{NEW_ROOT}/{run}"]
    return out, init


def plan():
    best = fill_missing(collect())
    rows = []
    for (rd, exp, sp), (ts, argv) in sorted(best.items()):
        new_argv, init = rewrite(rd, exp, sp, argv)
        run = Path(rd).name
        model_path = f"{NEW_ROOT}/{run}/{EXP_DIR[exp]}/{sp}/model.pth"
        rows.append({"run": run, "orig_root": str(Path(rd).parent), "exp": exp, "split": sp, "ts": ts, "argv": new_argv,
                     "init": init, "out": model_path, "tier": 1 if run in TIER1 or base_run(run) in TIER1 else 2})
    by_out = {r["out"]: r for r in rows}
    # levels
    def level(r, seen=()):
        if r["init"] is None:
            return 0
        par = by_out.get(r["init"])
        if par is None:
            r["missing_parent"] = True
            return 0
        return 1 + level(par)
    for r in rows:
        r["level"] = level(r)
    names = defaultdict(set)
    for r in rows:
        names[r["run"]].add(r["orig_root"])
    clashes = {k: v for k, v in names.items() if len(v) > 1}
    (HERE / "plan.json").write_text(json.dumps(rows, indent=1))
    n = defaultdict(int)
    for r in rows:
        n[(r["tier"], r["level"])] += 1
    print(f"{len(rows)} trainings, {len({r['run'] for r in rows})} runs; by (tier, level): {dict(sorted(n.items()))}")
    print("missing parents:", sum(r.get("missing_parent", False) for r in rows), "| run-name clashes across roots:", clashes)
    return rows


def depstr(after: str, parents: set) -> str:
    """after-ok on the data jobs (never train on half-merged data), after-any on the parent trainings."""
    parts = []
    if after:
        parts.append("afterok:" + after)
    if parents:
        parts.append("afterany:" + ":".join(sorted(parents)))
    return f"--dependency={','.join(parts)}" if parts else ""


def submit(args):
    rows = json.loads((HERE / "plan.json").read_text())
    if args.tier != "all":
        rows = [r for r in rows if r["tier"] == int(args.tier) or (int(args.tier) == 2 and False)]
    packs_dir = HERE / "packs"; packs_dir.mkdir(exist_ok=True)
    job_of_out, submitted = {}, []
    cpu_tiers = {int(t) for t in args.cpu_tiers.split(",") if t}
    for tier in sorted({r["tier"] for r in rows}):
        for lvl in sorted({r["level"] for r in rows}):
            sel = [r for r in rows if r["tier"] == tier and r["level"] == lvl]
            if tier in cpu_tiers:          # one CPU job per training
                for r in sel:
                    name = f"t{tier}_l{lvl}_{r['run']}_{r['split']}"
                    deps = {job_of_out[r["init"]]} if r["init"] in job_of_out else set()
                    dep = depstr(args.after, deps)
                    inner = " ".join(shlex.quote(a) for a in ["python3", "-u"] + r["argv"])
                    t = "12:00:00" if r["init"] is None else "06:00:00"
                    wrap = f"export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; bash evaluations/retrain/run_one.sh {inner}"
                    cmd = (f"sbatch --parsable {dep} -p cpu_sapphire --exclude=compute-3-1 -c 8 --mem=8G -t {t} "
                           f"--job-name=rt_{name[:60]} -o evaluations/retrain/logs/%x_%j.log --wrap={shlex.quote(wrap)}")
                    jid = f"<{name}>" if args.dry_run else subprocess.check_output(cmd, shell=True, cwd=REPO, text=True).strip()
                    if args.dry_run:
                        print(cmd[:300])
                    job_of_out[r["out"]] = jid
                    submitted.append({"pack": name, "job": jid, "n": 1, "deps": sorted(deps)})
                continue
            for p in range(0, len(sel), args.per_pack):
                chunk = sel[p: p + args.per_pack]
                name = f"t{tier}_l{lvl}_{p // args.per_pack:02d}"
                lines = []
                for r in chunk:
                    cmd = " ".join(shlex.quote(a) for a in ["python3", "-u"] + r["argv"])
                    lines.append(f"bash evaluations/retrain/run_one.sh {cmd}")
                f = packs_dir / f"{name}.txt"
                f.write_text("\n".join(lines) + "\n")
                deps = {job_of_out[r["init"]] for r in chunk if r["init"] in job_of_out}
                dep = depstr(args.after, deps)
                big = any(r["init"] is None for r in chunk)
                t = "08:00:00" if big else "04:00:00"
                cmd = (f"sbatch --parsable {dep} -p {args.partition} --gres=gpu:{args.gpus} -c {2 * len(chunk)} "
                       f"--mem={4 * len(chunk) + 8}G -t {t} --job-name=rt_{name} evaluations/generalization_v3/slurm/run_packed.sh {f}")
                if args.dry_run:
                    jid = f"<{name}>"
                    print(cmd)
                else:
                    jid = subprocess.check_output(cmd, shell=True, cwd=REPO, text=True).strip()
                for r in chunk:
                    job_of_out[r["out"]] = jid
                submitted.append({"pack": name, "job": jid, "n": len(chunk), "deps": sorted(deps)})
    (HERE / ("submitted_dry.json" if args.dry_run else "submitted.json")).write_text(json.dumps(submitted, indent=1))
    print(f"{len(submitted)} packs, {sum(s['n'] for s in submitted)} trainings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "submit"])
    ap.add_argument("--after", default="")
    ap.add_argument("--tier", default="all")
    ap.add_argument("--per-pack", type=int, default=12)
    ap.add_argument("--partition", default="gpu_h200")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--cpu-tiers", default="2", help="tiers run as one CPU job per training (others: GPU packs)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan() if args.cmd == "plan" else submit(args)


if __name__ == "__main__":
    main()
