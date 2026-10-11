# pred-distr-tool — generalization study of the expected-fidelity predictor

A graph network predicts the expected fidelity of a quantum circuit on each of three IQM-style
devices (EQE1_Top, EQE1_Bottom, QExa20) without compiling. The current work is the
generalization study for a paper (npj Quantum Information collection, deadline 2026-12-20),
on branch `generalization-study` (open PR; commit locally, push only when asked).

Several co-authors may use Claude here: they may write in Italian, English or German; answer in
the language of the message. Reports and documents for co-authors are in English, visual, with
little text. Be proactive: launch and chain experiments, report results with numbers.

## Where things are

| Path | What |
| --- | --- |
| `src/` | original tool (v1 model, encoding); `PYTHONPATH=src` for every script |
| `evaluations/generalization_v3/gsv3/` | current model code: `model.py` (all kinds), `training.py` (losses), `devices.py` (device graphs, calibration variants), `fast_encoding.py` |
| `evaluations/generalization_v3/scripts/run_experiment.py` | training entry point (lofo / logo / control / size) |
| `evaluations/generalization_v3/slurm/train_cpu.sh` | SLURM launcher, configured by env vars (below) |
| `evaluations/generalization_vN/results/<run>[_sS]/<exp>/<split>/` | results: `metrics.json`, `history.json`, `model.pth`; seed 5 has no `_s` suffix |
| `evaluations/compile_check/` | recompilations and labels: `v8b_variants.py` (calibration variants, `--n-aug`, `--usage`), `merge_usage.py`, `results/` (`v8d/v8d_labels.pt`, `v8b_orig/`, `v8u/usage_labels.pt`) |
| `evaluations/calibration_shift/` | calibration-shift test: `run_shift.py` (MODELS registry), `shift_k.sh predict_report` → `results_k/report.md` |
| `evaluations/diagnostics_v4v5/` | diagnostics and oracle studies (`tracking_oracle.py`, `tracking_oracle_usage.py`, `counts_check.py`, …), reports in `results/` |
| `evaluations/results_report.py` | writes `evaluations/RESULTS.md` + `figures/`; `results_loop.sh` reruns it every 15 min as a SLURM job |
| `evaluations/explorer/fidelity_predictor_explorer.html` | source of the interactive explorer artifact |
| `evaluations/TODO.md` | findings and open items |

## Models (state on 2026-10-11)

- **Main model: v8kd40** (kind `sinkhorn_rf_route_czc`, runs `generalization_v9/results/v8kd40[_s6|_s7]`):
  v8k trained from scratch, then 40 epochs of change-loss fine-tuning on 100 exactly compiled
  calibrations per device. 3 seeds: R² qaoa 0.965 ± 0.031, qnn 0.855 ± 0.063, variational
  0.918 ± 0.025, control 0.996; local calibration tracking 0.60 ± 0.02, global 0.95–0.98.
- **v8k**: v8s + CZ count constrained to the gate decomposition when no routing is needed
  (`cz = base(gate) · (1 + κ · hops)`).
- **In progress: v11** (kind `sinkhorn_rf_route_usage`, runs `v11`, `v11d40`): usage-map head,
  CZ per coupler (shortest paths, lower-error preferred, learned temperature) and r / measure
  per physical qubit; `log F = −Σ U·ε` is exact for true maps. Decision on 2026-10-23: keep if
  local tracking clearly > 0.6 without losing R² on the held-out families.
- Previous: v1 (circuit-only, first model) → v4 physics head → v5 Sinkhorn → v7 layout loss →
  v8a count loss (previous main) → v8r routing-free flag → v8s routing estimate + cz-weighted
  loss → v8k. Dropped: v5b, v6, v8b, v9 / v9c / v9k (relative device features), v8r_cz alone.
- Why tracking stopped at 0.6: oracle study (`diagnostics_v4v5/results/report_tracking_oracle*.md`):
  not epochs or counts; the head (initial layout + gate totals) cannot describe routing.

## Running experiments

```bash
# one split per job; FAMILIES/SPLIT_GROUPS pick the split; control uses FAMILIES=x
LP=evaluations/compile_check/results/compiler_layouts_v6pilot.pt
CP=evaluations/compile_check/results/compiler_counts_v8.pt
X="--loss mixed --val-mode random --layout-lambda 0.1 --layout-loss region_dist --layouts-path $LP \
   --count-lambda 0.1 --count-weights 0.1,0.1,1 --counts-path $CP"
RUN=v8k_s6 MODEL=sinkhorn_rf_route_czc ALPHA=0 SEED=6 EXPERIMENTS=lofo FAMILIES=qaoa SPLIT_GROUPS=qaoa \
  RESULTS_ROOT=evaluations/generalization_v9/results EXTRA_ARGS="$X" \
  sbatch -p cpu_sapphire --time=08:00:00 --job-name=cpu_v8k_s6_qaoa --export=ALL \
  evaluations/generalization_v3/slurm/train_cpu.sh
```

- Standard splits: `lofo qaoa`, `lofo qnn`, `logo variational`, `control x` (split name `random_seedS`); seeds 5, 6, 7.
- Fine-tuning (change loss): add `--calib-aug evaluations/compile_check/results/v8d --orig-best
  evaluations/compile_check/results/v8b_orig/v8b_orig_labels.pt --delta-lambda 1 --keep-last
  --epochs 40 --patience 100 --lr 1e-4 --init-from <split model.pth>` and `--mem=8G`.
- v11: add `--usage-lambda 0.05 --usage-path evaluations/compile_check/results/v8u_orig/usage_labels.pt`.
- A new model needs: its kind in `gsv3/model.py` (`MODEL_KINDS`, `build_model`) and in the kind
  lists of `run_experiment.py`; a line in `calibration_shift/run_shift.py` MODELS; an entry in
  `results_report.py` MODELS.
- Calibration test after a control: `sbatch --dependency=afterany:<control job> --cpus-per-task=8
  --mem=12G --time=01:00:00 evaluations/calibration_shift/shift_k.sh predict_report`.
- Chain dependent steps with `--dependency` (`afterok` for fine-tuning from a trained model,
  `afterany` for reports) so results arrive without supervision.
- Training resumes from `checkpoint.pt` when a job is resubmitted with the same RUN.

## Cluster rules (PoliTo HPC)

- Never load data or train on the login node: use `srun` / `sbatch` (smoke tests too). `/tmp` is
  node-local: put scripts that run on compute nodes under `$HOME`.
- Partitions: `cpu_sapphire` (backfill; `sbatch --test-only` start estimates are pessimistic),
  `cpu_sapphire_ext` (often full). GPUs are usually busy for days.
- Some shared nodes run ~30× slower (compute-5-6, compute-5-8 seen): check the first epochs; if
  slow, cancel and resubmit with `-w <idle node>` (from `sinfo -p cpu_sapphire -o "%n %C %t"`).
  Raising a running job's time limit is not allowed.
- Request memory near the measured peak (the cluster warns about RAM efficiency): CPU training
  6G, change-loss fine-tuning 8G, compile task 2G, calibration test 12G. `sacct`/`seff` do not
  work: measure with `sstat -j <id>.batch -o MaxRSS`.
- No job arrays; submit tasks in a loop.

## Data facts

- Labels: expected fidelity of the Qiskit level-2 compilation (MQT formula); best of 3 compiler seeds for
  recompiled sets. Every recompilation is verified (native gates, existing couplers, equivalence).
- Grover is excluded from recompilations (compilation memory too heavy).
- Usage maps reproduce log F to 1e-7 for static circuits with F > 1e-3; dynamic circuits
  (if_else: iqpe, dynamic_qft) are dropped from the maps; labels below F ≈ 1e-6 are rounded.
- Model dataset: 17,056 circuits vs 19,465 labels (gap upstream, in `create_pt_file_new.py`).

## Reports (update when results change; the user asks for it)

- Report for co-authors (Claude Docs): https://claude.ai/code/artifact/3e13c3ec-9da1-44d5-ad88-6444eebd3961
- Interactive explorer: https://claude.ai/artifact/MYZFQqQ894n3HhmG7rXT9b (source in `evaluations/explorer/`)
- Notion page: id 3f410bbf-6295-811c-b81d-ee07c38f18d9 (private draft)
- `evaluations/RESULTS.md` updates itself every 15 min while `results_loop.sh` runs.

## Commits

- Small logical commits on `generalization-study`; never push unless asked; no `Co-Authored-By` trailer.
- Commit finished results only (metrics, history, run logs; `.gitignore` drops `*.pth`, predictions,
  locks). Leave the user's own `.gitignore` edits and root-level scratch files alone.
