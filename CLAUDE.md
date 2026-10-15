# pred-distr-tool — generalization study of the expected-fidelity predictor

A graph network predicts the expected fidelity of a quantum circuit on each of three IQM-style
devices (EQE1_Top, EQE1_Bottom, QExa20) without compiling. The current work is the
generalization study for a paper (npj Quantum Information collection, deadline 2026-12-20),
on branch `generalization-study` (open PR); Claude commits and pushes its finished work itself.

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
| `evaluations/fastpath/` | two-model plan, routing-free side: `fastpath.py` (switch, cached embeddings, VF2-score placement, F / ESP per calibration without compiling), `run_fastpath.py` (validate / predict / report / merge), `launch.sh` (SLURM: tasks + report + `esp/hybrid.py --fast`), `run_local.sh` (same without SLURM; works without the models' predictions.json) |
| `evaluations/results_report.py` | writes `evaluations/RESULTS.md` + `figures/`; `results_loop.sh` reruns it every 15 min as a SLURM job |
| `evaluations/explorer/fidelity_predictor_explorer.html` | source of the interactive explorer artifact |
| `evaluations/TODO.md` | findings and open items |

## Models (state on 2026-10-11)

- **Main model: v8kdc10** (adopted 2026-10-11; kind `sinkhorn_rf_route_czc`, runs `generalization_v9/results/v8kdc10[_s6|_s7]`):
  the v8kdl40 recipe + `--delta-corr-lambda 0.1` (1 − Pearson of predicted vs true Δ per batch and device).
  3 seeds: R² qaoa 0.970 ± 0.023, qnn 0.847 ± 0.123, variational 0.939 ± 0.018 (qnn subset 0.93), control 0.996;
  local tracking 0.66 ± 0.02 (ensemble 0.69, an option at 3× inference), global 0.95–0.98; regret after a local
  change re-predicted 0.0037; placement probe: reaction agreement with the compiler 0.49 (v8kdl40 0.52), tracking
  on the probe sample 0.74 (v8kdl40 0.70). The previous best models stay in the paper's comparison (storytelling):
  v1 → v8a → v8kd40 → v8kdl40 → v8kdc10.
- **Previous main model: v8kd40** (kind `sinkhorn_rf_route_czc`, runs `generalization_v9/results/v8kd40[_s6|_s7]`):
  v8k trained from scratch, then 40 epochs of change-loss fine-tuning on 100 exactly compiled
  calibrations per device. 3 seeds: R² qaoa 0.965 ± 0.031, qnn 0.855 ± 0.063, variational
  0.918 ± 0.025, control 0.996; local calibration tracking 0.60 ± 0.02, global 0.95–0.98.
  "Tracking" is a correlation (not R²) between predicted and true Δlog F under calibration
  changes; in practice re-predicting after a local change cuts the device-choice regret by 21%
  (0.0062 → 0.0049) while R² of F after the change stays 0.99.
- **v8k**: v8s + CZ count constrained to the gate decomposition when no routing is needed
  (`cz = base(gate) · (1 + κ · hops)`).
- **v8kdl40** (runs `v8kdl40[_s6|_s7]`): the v8kd40 fine-tuning plus the
  layout and count losses on each calibration variant (`--aug-aux`). 3 seeds: local tracking
  0.63 ± 0.02, regret after a calibration change −43% when re-predicting, generalization as v8kd40
  (one qnn seed low). Superseded by v8kdc10.
- Done (2026-10-11): per-error-kind placement prior (`*_czc_ep3`, v8kdq40b), MLP placement scorer
  (`*_czc_mlps`, v8kdm40b), v8kdl40 with placement lr ×10 (v8kdl40p), fine-tuning grid
  `v8kg_t{15,30}_d{1,3}_e{40,80}` (Sinkhorn tau × change-loss weight × epochs, control only).
- **Running (2026-10-11 evening): compiler-score placement prior** (kind `sinkhorn_rf_route_czc_vf2`, runs
  v8kdv40 β₀ = 10 and v8kdv40s β₀ = 3, v8kdl40 recipe + placement lr ×10): the placement score is reduced by
  β · the log-fidelity a logical qubit would lose on a physical one (its 1q / measure / CZ counts × that qubit's
  errors, the head's scales; Qiskit VF2Layout's score, linearised), β learned separately for routing-free
  circuits and the others. Side variant for the paper comparison; v8kdl40 untouched.
- Placement experiments so far all give local tracking 0.60–0.63 (grid on tau / change-loss weight / epochs,
  lr ×10, per-kind prior, MLP scorer): the placement's error-kind weights never move (readout −0.020,
  CZ −0.029 vs compiler −0.024 / −0.016), so the limit is the training signal.
- **Correlation change loss (2026-10-11): v8kdc10** = v8kdl40 recipe + `--delta-corr-lambda 0.1` (1 − Pearson of
  Δ per batch and device). Local tracking 0.63 / 0.67 / 0.67 (mean 0.66 vs 0.63; ensemble 0.69); weights 0.02 /
  0.3 / 1 no better. Cold start N = 33 model-only loss 0.0054 vs 0.0064. Adopted as main model (above).
- **v12 placed-circuit network (2026-10-11)**, kind `sinkhorn_rf_route_pgnn_usage` (`PlacedCircuitGNN` in model.py,
  design `evaluations/DESIGN_v12_placed_circuit_gnn.md`, phase 0 `evaluations/v12/phase0_true_layout.py` + `phase0.sh`,
  reports `evaluations/v12/results/report_phase0*.md`). Frozen v11dl40 + network, 577 test circuits, unseen
  calibrations: with the compiler's true layout 0.714 → **0.857** (30k steps, no memorisation: seen 0.839);
  on the learned placement no gain (0.57–0.72 vs 0.737 without network), whether trained on true, learned or
  mixed placement. So with a good routing model the placement is the bottleneck again.
  **Phase 1 (running 2026-10-11 evening):** `evaluations/v12/phase1_joint.py` + `phase1.sh`: whole v11dl40
  fine-tuned (lr 1e-4) with the 30k phase-0 network (lr 3e-4), placement loss on the compiler's layout of each
  calibration (`--place-loss ce|region|both`, λ 0.1), scheduled sampling true → learned layout over the first
  60% of 20k steps; reports `results/report_phase1_{ce,region,both}.md`. Caveat: the phase-0/1 "unseen"
  calibrations (v8d aug 80-99) are unseen by the network only; v11dl40 was fine-tuned on all 100 v8d variants,
  so its 0.737 is inflated. Only the calibration test (disjoint variants, best of 5) judges v12 against others. Start: learned placement 0.574,
  argmax agreement with the compiler 0.12. Go to the standard evaluation if the learned-placement row clearly
  beats 0.737 with R² of F unchanged (0.990).
- **Pipeline anatomy (2026-10-11, `compile_check/pipeline_anatomy.py`, reports `results/pipeline_anatomy*/report.md`):**
  Qiskit 2.3.1 level 2: Sabre layout/routing and optimisation ignore the errors; only VF2Layout (routing-free) and
  VF2PostLayout (routed) pick WHERE the fixed structure goes by an average-error score. 200 test circuits × 4 local
  variants × 3 seeds: CZ count and routing-free status never change with the calibration; re-running only that VF2
  step on the circuit it scores (logical / routed before optimisation) reproduces the recompilation's Δlog F with
  corr 1.000 (exact 88%), best of 3 too. Time: routing-free full compile 2.3 ms median (≈ the VF2 step alone),
  routed 40 ms vs 8 ms for the VF2 step. Ideas: count head calibration-blind; placement computed with the VF2
  score instead of learned; routing-free circuits are cheap enough to compile outright.
- **Not adopted:** v11 usage-map head (`sinkhorn_rf_route_usage`): exact for true maps, tracking 0.59
  after fine-tuning, weaker on qaoa; kept as the base for the multi-device extension. Also dropped:
  placement reaction loss (v8kdr40), single-weight error preference (v8kdp40).
- Why tracking is limited (`diagnostics_v4v5/results/report_tracking_anatomy_*.md`): the embeddings
  carry every error, but the placement moves half as much as the compiler (reaction agreement 0.52
  vs 0.87 between compiler seeds) and over-weights CZ vs readout; weakest on the EQE1 chips and on
  circuits with 1–5 qubits.
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

## Standard evaluation of a candidate model

Every new candidate is judged on all of these, against the current main model, before it is adopted:
1. R², device-choice regret and right-device share on the four splits (LOFO qaoa, LOFO qnn,
   LOGO variational with its qnn subset, control), 3 seeds before adoption.
2. Calibration test (`shift_k.sh predict_report`): local tracking (a correlation of Δlog F, not R²),
   moderate / large tracking, R² of F after the change, and the regret when re-predicting after a
   change vs keeping the old choice.
3. `diagnostics_v4v5/tracking_anatomy.sh --compare <main>,<candidate> --main <candidate>:<kind>`:
   where it misses (device, circuit size, family, variant, routing) and why (embeddings, placement
   reaction vs the compiler's, which error kinds move the placement, fixed-placement vs moving terms).
4. `diagnostics_v4v5/placement_probe.sh` and, for new heads, the oracle studies (`tracking_oracle*.py`).
5. Use case for the paper: `calibration_shift/decision_policies.py --models <main>,<candidate>` after the
   calibration test (one adaptive top-k rule: compile on the backends predicted within δ of the best).
   Cold start on fleets of 3–33 backends and calibration updates (local / moderate / large, starting
   from the truly best or the model's device; section C sweeps δ 0–0.1 for any tolerance, 100 shared fleets). v8kdl40: loss < 0.001 with ~40% of the compilations;
   after an update δ 0.02 cuts the loss of not reacting 3–5× with 2.1 of 3 compilations; never switch
   devices on the model alone after global changes. `recompile_policy.py` is the earlier, narrower version.
6. Seed ensemble, reported as an option NEXT TO the main model (not as the main model: 3× inference):
   `calibration_shift/ensemble_tracking.py --models <main>` and `decision_policies.py --models <main>,<main>_ens`.
   Always give its cost too (model 10–40 ms per circuit for 3–27 devices, ×3 for the ensemble; one
   compilation ≈ 21 ms median). v8kdl40: tracking 0.63 → 0.67, 15–25% less fidelity lost at the same δ,
   about the same total time at a fixed loss target on median circuits.
Chain 2–6 after the candidate's control job so they run without supervision.
- Data scaling (running 2026-10-11): the oracle study shows test circuits track at 0.73 on calibrations seen in
  training vs 0.59 on unseen ones, and 8 → 100 calibrations per device raised tracking 0.45 → 0.59. Learning curve
  `v8kc{12,25,50,q4}[_s6]`, `v11c{12,25,50,q4}` (`--aug-variants 0..k-1`, `--aug-circuit-mod 4` = 100 calibrations,
  ~1/4 of the circuits). If tracking still rises at 100, generate more calibrations with fewer circuits each.

## Cluster rules (PoliTo HPC)

- Never load data or train on the login node: use `srun` / `sbatch` (smoke tests too). `/tmp` is
  node-local: put scripts that run on compute nodes under `$HOME`.
- Partitions: `cpu_sapphire` (backfill; `sbatch --test-only` start estimates are pessimistic),
  `cpu_sapphire_ext` (often full). GPUs are often busy; when `gpu_h200` (compute-7-12) is free use
  it: `sbatch -p gpu_h200 --gres=gpu:2 -c 8 --mem=32G evaluations/generalization_v3/slurm/train_gpu.sh`
  (same env vars as train_cpu.sh; `pick_gpu.sh` skips its broken GPU 0000:19:00.0). The v11
  change-loss fine-tuning runs at 19 s/epoch there vs ~4.5 min on 8 CPU cores.
- **Before every launch, compare queues and start times on all usable partitions** (`sbatch --test-only`
  on gpu_h200, gpu_a100, gpu_a40, gpu_a40_ext, cpu_sapphire, cpu_sapphire_ext,
  cpu_skylake, cpu_skylake_ext) and pick the fastest; re-check queued jobs and move them
  (`scontrol update JobId=… Partition=…`). Tell the user where each job runs.
  `evaluations/generalization_v3/slurm/pick_partition.sh <time> <mem>` prints the fastest GPU option.
  V100 partitions do not work with this env (torch 2.11+cu128 has no sm_70 kernels).
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

- Small logical commits on the working branch; no `Co-Authored-By` trailer.
- Claude pushes its own commits (`git push -u origin <branch>`) once the work is done and checked, so the
  user can pull and launch on the cluster; say which branch was pushed. Branches:
  `generalization-study` (open PR), `esp-depth` (ESP / duration / depth heads, branched from it).
- Commit finished results only (metrics, history, run logs; `.gitignore` drops `*.pth`, predictions,
  locks). Leave the user's own `.gitignore` edits and root-level scratch files alone.
