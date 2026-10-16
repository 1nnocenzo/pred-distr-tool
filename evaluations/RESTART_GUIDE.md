# Restart guide: the essential files only (state 2026-10-16)

This guide covers only what you need to rebuild and run the current system. It leaves out the experiment history
and the diagnostics. For each essential file it says what the file is used for and what it implements, then gives
the order in which to run things. Every command runs from the repository root with `PYTHONPATH=src`.

## 0. What the system does

From a circuit (OpenQASM, 2–20 qubits) and the calibration of each device (EQE1_Top 26 q, EQE1_Bottom 27 q,
QExa20 20 q; native gates `r`, `cz`, `measure`), it predicts the following **without compiling**:

- **F**, the expected fidelity (MQT): the product of (1 − error) over the native operations of the circuit compiled
  by Qiskit 2.3.1 at level 2;
- **duration**, the ASAP schedule with r 42 ns, cz 130 ns, measure 15 µs;
- **ESP** = F · exp(−Σ_q idle_q / min(T1, T2)_q), the MQT estimated success probability.

Three components make up the current system:

| component | for | status |
| --- | --- | --- |
| **F model** `sinkhorn_rf_route_czc` (v8kdc10; v8kdfi10 = the same plus an ESP head) | F on any calibration, without compiling | main model |
| **Fast path** | exact F / ESP / duration of routing-free circuits, with no compiler per calibration | validated |
| **v29 unified model** (frozen F backbone + route stage + ESP formula) | duration and ESP of routed circuits, in one forward pass | written and smoke-tested; trains off-cluster |
| *(reference)* light compile | level 2 with a 1-trial Sabre, exact physics | baseline that v29 must beat on time |

---

## 1. Minimal tree

```
createDevice.py                      environment.yml
data/  circuits.zip  2026-03-02T04_14_02.290607Z.json  qpu_dcdb_results.csv
src/model/  encoding.py  gnn.py  best_params.json
evaluations/
  generalization/genstudy/           __init__ data experiment io metrics paths cli model
  generalization_v2/gsv2/            __init__ devices experiment gpu model paths training
  generalization_v2/scripts/         build_device_graphs.py
  generalization_v2/results/         device_graphs.raw.json  (device_graphs.pt, built)
  generalization_v3/gsv3/            __init__ paths devices fast_encoding model router training experiment groups
  generalization_v3/scripts/         _bootstrap.py  build_qubit_dataset.py  run_experiment.py
  generalization_v3/slurm/           train_cpu.sh  train_gpu.sh  pick_partition.sh  pick_gpu.sh
  compile_check/                     compile_check.py  duration_anatomy.py  compile_layouts.py  v8b_variants.py
                                     v8b_merge.sh  v8b_merge_orig.sh  v8d_merge.sh  esp_labels.py  sabre_labels.py
                                     qaoa_measure.py  qaoa_merge.py
  esp/                               esp_decomp.py  esp_router.py  results/decomp_table.json
  synthetic_routing/                 synth_routing.py  synth_layered.py
  route_stage/                       prep.py  stage1.py  unified.py  test_stage1.py  test_unified.py  run_local.sh  report_v24.py
  calibration_shift/                 cshift.py  run_shift.py  shift_k.py  shift_k.sh  decision_policies.py  ensemble_tracking.py
  fastpath/                          fastpath.py  run_fastpath.py
  light_compile/                     light_compile.py  run_local.sh
```

Everything else in `evaluations/` is experiment history: v4–v12 variants, diagnostics, the explorer, retrain packs,
old reports and v24–v28 analysis scripts. Nothing above imports it. One exception: `evaluations/retrain/plan.json`
holds the exact command line of every trained run, which is useful as a recipe reference.

---

## 2. Files, one by one

### 2.1 Devices and circuits (repository root, `data/`, `src/`)

| file | used for | implements |
| --- | --- | --- |
| `createDevice.py` | every compilation and every label | Qiskit `BackendV2` targets: EQE1 (53 q, coupling map, IQM JSON calibration) cut into EQE1_Top / EQE1_Bottom, QExa20 (CSV); gate errors, readout, T1/T2 (missing values filled with the mean), fixed durations. **Caveat:** QExa20's T1/T2 come out 10× too small; the ESP code divides QExa20's decay by 10 (`T_FIX`). This is an assumption to verify |
| `data/circuits.zip` | the benchmark circuits (MQT Bench, 35 families) | QASM files `family/name.qasm`. It contains the **old** qaoa without final measurements; `qaoa_measure.py` adds them |
| `data/*.json`, `data/*.csv` | calibration inputs | EQE1 and QExa20 calibrations |
| `src/model/encoding.py` | the dataset graphs | circuit → gate DAG: nodes = instructions (barriers removed, transpiled at level 0 to the 27-gate OpenQASM 3 basis); 40 features per node (gate one-hot 28, sin/cos of 3 parameters, arity, controls, number of parameters, critical-path flag, fan-in, fan-out); edges = consecutive gates on a wire |
| `src/model/gnn.py` | circuit encoder | `GraphConvolutionSage` (SAGE convolutions, bidirectional, residual layers) |
| `src/model/best_params.json` | encoder hyper-parameters | hidden 142, 3 conv + 1 residual, bidirectional, head MLP [512, 256, 128] |
| `environment.yml` | environment | conda env `pred-distr`: Python 3.11, torch 2.11, torch-geometric 2.7, Qiskit 2.3.1, mqt |

### 2.2 Shared library (`generalization/genstudy`, `generalization_v2/gsv2`)

These are needed because `gsv3` imports them.

| file | implements |
| --- | --- |
| `genstudy/data.py` | `GraphDataset` (loads `graph_dataset_expected_fidelity.pt`: graphs with `y` = F on the 3 devices, `circuit_name`), family / size / random splits |
| `genstudy/metrics.py` | R², MAE, RMSE, device-choice accuracy, regret |
| `genstudy/experiment.py`, `io.py`, `paths.py`, `cli.py` | running one split and writing `metrics.json` / `history.json`; JSON / CSV helpers; paths; shared CLI arguments |
| `genstudy/model.py` | the v1 model (imported by the experiment drivers) |
| `gsv2/devices.py` | `_backend_graph` (backend → raw node features −log(1−e_r), −log(1−e_ro), log10 T1, log10 T2, degree; edges −log(1−e_cz)); `load_device_batch` |
| `gsv2/training.py` | base protocol (batch 32, epochs, patience), `log_target`, `balanced_sampler`, `validation_split` |
| `gsv2/experiment.py`, `gpu.py`, `paths.py`, `model.py` | split driver, CPU/GPU choice, paths, the v2 model (`pooled` kind, imported by gsv3) |
| `generalization_v2/scripts/build_device_graphs.py` | builds `generalization_v2/results/device_graphs.pt` from `createDevice.py` (the tracked `device_graphs.raw.json` is a copy of the raw data) |

### 2.3 Model code (`generalization_v3/gsv3`)

| file | implements |
| --- | --- |
| `paths.py` | repository paths and `ensure_imports()`, which puts `genstudy`, `gsv2` and `src/model` on `sys.path` |
| `fast_encoding.py` | `encode` / `to_data` / `to_basis`: the same graph as `encoding.py` in one pass over the instructions (2.2 ms vs 11.2 ms); `UNIQUE` = gate vocabulary |
| `devices.py` | `load_devices` (device graphs + 8 Laplacian eigenvectors + `phys` errors, `coh` = 1/min(T1,T2) with the QExa20 fix, `tg` / `tcz` gate durations, coupler lists); `variant_batch` (calibration variants standardised like the originals); `GATE_US` |
| `model.py` | **all the models**. The current one is `SinkhornPlacementPredictor`: circuit encoder (`NodeSage`) → gate embeddings g → logical-qubit embeddings (mean/max + 2 GINE layers on the interaction graph) → device encoder (3 GINE layers) → placement scores + **Sinkhorn** (τ 0.3, 20 iterations, soft one-to-one layout A) → per gate: operand context, placement-weighted errors, expected operand distance Aᵀ H A, routing-free flag, routing estimate → count MLP → native counts (r, measure, cz); `_czc`: cz = base(gate) · (1 + κ · extra hops · (1 − routing_free)) → **log F = − Σ_g Σ_k s_k n_k ε_k**. Helpers: `routing_free` (VF2 subgraph test, as Qiskit's VF2Layout), `routing_estimate` (greedy router cost), `hard_placement` (Hungarian), `dag_schedule` (ASAP of the logical DAG), the ESP heads `esp_features` / `esp_from_features` (suffix `_espfi` = v19 formula head), `build_model(kind, …)`, `split_esp_kind`. v29 hooks: `last_gate_emb`, `skip_esp` |
| `router.py` | approximate greedy router (used by the `_espfr` head) |
| `training.py` | `train_model` (Adam, clipping, early stopping, checkpoints and resume, calibration augmentation, change loss) and the losses: mixed MSE on F + log F, `layout_loss` / `region_distance_loss`, `count_loss`, change loss Δlog F and its correlation version, `esp_loss` |
| `experiment.py` | `run_split`: train, predict and score one split |
| `groups.py` | LOGO groups (`variational` = vqe_* + qnn) and `group_holdout_split` |

### 2.4 Scripts and launchers (`generalization_v3/scripts`, `slurm`)

| file | used for |
| --- | --- |
| `scripts/build_qubit_dataset.py` | adds `gate_qubits` / `n_qubits` to the original dataset and writes `generalization_v3/data/graph_dataset_expected_fidelity.pt` |
| `scripts/run_experiment.py` | **training entry point**: `control` / `lofo` / `logo` / `size`, `--model <kind>` and every loss flag (§3.3) |
| `scripts/_bootstrap.py` | import paths for the scripts |
| `slurm/train_cpu.sh`, `train_gpu.sh` | SLURM launchers configured by env vars (`RUN MODEL ALPHA SEED EXPERIMENTS FAMILIES SPLIT_GROUPS RESULTS_ROOT EXTRA_ARGS`); one split per job, per-split lock, resume from `checkpoint.pt` |
| `slurm/pick_partition.sh`, `pick_gpu.sh` | fastest partition; skips the broken GPU on gpu_h200 |

### 2.5 Labels (`compile_check/`, `esp/`, `synthetic_routing/`)

| file | produces | how |
| --- | --- | --- |
| `compile_check.py` | (helper library) | `compile_once` (Qiskit L2, seed, optional layout), `is_valid` (native gates on existing couplers), `DATA`, `LABELS` |
| `duration_anatomy.py` | (helper) | `DEVICES`, `EXCLUDE` for `sabre_labels.py` |
| `compile_layouts.py` | `compiler_layouts*.pt`, `compiler_counts_v8.pt`, QPY of the compilations | compiles every circuit on the 3 devices (L2, fixed seed); initial layout as device-graph node indices; `counts` = native r / measure / cz |
| `v8b_variants.py` + `v8b_merge.sh`, `v8b_merge_orig.sh`, `v8d_merge.sh` | `v8d/` (100 calibration variants per device: labels, layouts, counts, `variants_raw.json`), `v8b_orig/v8b_orig_labels.pt` | `variants()` builds perturbed calibrations (jitter, shuffle, bad qubits, global scale); every circuit compiled with 3 seeds, the best gives the label. `variants()` is also imported by other scripts |
| `esp_labels.py` | `esp_labels.pt` {circuit: F, ESP, depth, duration, decay} | from the stored compilations (seed 0), MQT `estimated_success_probability`, ASAP; QExa20 decay ÷ 10 at merge |
| `sabre_labels.py` | `results/sabre_labels/` | per routed (circuit, device): the SWAPs Sabre inserts on each operand wire of each gate, the routed schedule and the durations (logical / replay / final). Training targets only |
| `qaoa_measure.py`, `qaoa_merge.py` | regenerated qaoa | adds the final measurements and merges the new qaoa into every data file |
| `esp/esp_decomp.py` → `esp/results/decomp_table.json` | r / cz count of every gate type transpiled alone | used by the schedules (route stage, v29, the `_espft` head) |
| `esp/esp_router.py` | (library) | `chips` (hops and BFS parents per chip), `greedy_place`, `path`: the greedy router used for the route-stage input features |
| `synthetic_routing/synth_routing.py` | 15k synthetic circuits (chain, ring, star, grid, random graphs…) with dataset-style labels | extra training data for routing; family `synth`, training side only |
| `synthetic_routing/synth_layered.py` | 6k circuits of the dense, irregular, low-parallelism class (qaoa-like); `stage1_synth2.pt` | the v27 data that brought qaoa from 13.9% to 8.8% |

### 2.6 Duration and ESP of routed circuits (`route_stage/`)

| file | implements |
| --- | --- |
| `prep.py` | builds `results/stage1.pt`: per circuit and device the Sabre targets (SWAPs per slot, teacher-forced waits, matched gates, routing-free flag, final / replay durations) and the greedy router's input features (`prior_route`: SWAPs per slot, hops) |
| `stage1.py` | **route stage** (v24–v27): `build` (PyG items), `Stage1` (input MLP + 6 message-passing layers over predecessors / successors / global → per gate and device: SWAP rate per operand wire (Poisson), extra wait (Huber); pooled stage 2 = log(final / routed schedule)), `losses`, `schedule_torch` (level-wise differentiable ASAP), `duration_loss`, `schedule_np` / `schedule_k` (reference schedules), `evaluate`, `split_names` (standard splits from the v8kdfi10 lists, LOSO on synthetic structures), `load_all` (`--synth2` = v27) |
| `unified.py` | **v29, one model for F, duration and ESP**: frozen v8kdfi10 F network (its ESP head off) → log F, gate embeddings, operand distances, placement; `Stage1` reading those as extra inputs → SWAPs, waits, stage 2; routing-free pairs (the backbone's VF2 test) get no SWAPs; duration = schedule × e^stage2 (MC: 24 Poisson samples in one torch pass, `mc_schedule`); ESP: idle per logical qubit on the routed schedule × 1/min(T1,T2) of the physical qubit chosen by the **same** hard placement as F, scaled by a small learned head. `train` / `eval`, `summary` (duration, R² F / ESP, abs Δlog ESP, device choice by ESP), `timing` (whole prediction per circuit by 2q-gate bin) |
| `test_stage1.py`, `test_unified.py` | smoke tests, no data needed (10 s / 30 s); run them after any code change |
| `run_local.sh` | runs everything without SLURM: `test29`, `smoke29`, `train29` (3 seeds × 5 splits), `synth2`, `train27`, `report` |
| `report_v24.py` | `results/report_v24.md`: duration table of every run plus the v29 section (ESP, F, time) |
| `RUNBOOK.md` | off-cluster commands and adoption criteria |

### 2.7 Evaluation and use (`calibration_shift/`, `fastpath/`, `light_compile/`)

| file | implements |
| --- | --- |
| `calibration_shift/cshift.py` | calibration variants of the backends (×0.5, ×2, shuffle, jitter) and `compile_and_score` |
| `calibration_shift/run_shift.py` | **MODELS registry** (each model to test: kind and checkpoint), prediction on variant devices, report helpers |
| `calibration_shift/shift_k.py` + `shift_k.sh` | calibration test: 457 unseen circuits × 39 calibrations, truth = best of 5 seeds; **tracking** = correlation of predicted vs true Δlog F; `truth` / `predict` / `report` → `results_k/report.md` |
| `calibration_shift/decision_policies.py` | use case: adaptive top-k rule (compile on the devices predicted within δ of the best); cold start on fleets of 3–33 backends and calibration updates |
| `calibration_shift/ensemble_tracking.py` | seed ensemble (mean of 3 seeds' log F) |
| `fastpath/fastpath.py` | routing-free pairs without a compiler: `routing_free_switch`, `compile_reference` (once), `structure`, `embeddings` (VF2, capped), `place` (Qiskit VF2Layout's average-error score), `evaluate` (MQT F and ESP, exact) |
| `fastpath/run_fastpath.py` | validation against recompilation and predictions for the hybrid evaluation |
| `light_compile/light_compile.py` | switch (cheap checks + VF2) → routing-free: exact L2; routed: L2 with the 1-trial Sabre; `physics` (exact F / ESP / duration); `eval` / `report` with timings |

---

## 3. Run order

### 3.1 Environment
```bash
conda env create -f environment.yml && conda activate pred-distr
export PYTHONPATH=$PWD/src
```

### 3.2 Data and labels (cluster, SLURM jobs submitted in a loop; each label script has `run --task-id i --n-tasks N` then `merge`)
1. `python createDevice.py`, then check QExa20's T1/T2.
2. `python evaluations/generalization_v2/scripts/build_device_graphs.py --output evaluations/generalization_v2/results/device_graphs.pt`
3. Original dataset: `create_pt_file_new.py` in `~/compileCircuits` (outside this repo), then
   `python evaluations/generalization_v3/scripts/build_qubit_dataset.py`. For the new qaoa: `qaoa_measure.py`, then `qaoa_merge.py`.
4. `compile_check/compile_layouts.py run … --exclude grover --save-qpy` → `merge` → `counts`.
5. `compile_check/v8b_variants.py run --k 3 --n-aug 100 --out results/v8d` → `raw` → `v8d_merge.sh`, `v8b_merge_orig.sh`.
6. `compile_check/esp_labels.py run` → `merge`.
7. `esp/esp_decomp.py` (→ `decomp_table.json`), `compile_check/sabre_labels.py run`, `synthetic_routing/synth_routing.py run` → `merge`,
   `route_stage/run_local.sh synth2`, `route_stage/prep.py --out results/stage1.pt`.

### 3.3 F model (4 standard splits × seeds 5, 6, 7)
```bash
LP=evaluations/compile_check/results/compiler_layouts_v6pilot.pt
CP=evaluations/compile_check/results/compiler_counts_v8.pt
BASE="--loss mixed --val-mode random --layout-lambda 0.1 --layout-loss region_dist --layouts-path $LP \
      --count-lambda 0.1 --count-weights 0.1,0.1,1 --counts-path $CP"
R=evaluations/generalization_v10/results
# stage 1, from scratch (control: EXPERIMENTS=control FAMILIES=x; also lofo qnn, logo variational)
RUN=v8k MODEL=sinkhorn_rf_route_czc ALPHA=0 SEED=5 EXPERIMENTS=lofo FAMILIES=qaoa SPLIT_GROUPS=qaoa \
  RESULTS_ROOT=$R EXTRA_ARGS="$BASE" sbatch -p cpu_sapphire --time=08:00:00 --mem=6G --export=ALL \
  evaluations/generalization_v3/slurm/train_cpu.sh
# stage 2, fine-tuning on the 100 calibration variants (--dependency=afterok:<v8k job>, --mem=8G)
FT="--calib-aug evaluations/compile_check/results/v8d --orig-best evaluations/compile_check/results/v8b_orig/v8b_orig_labels.pt \
    --delta-lambda 1 --keep-last --epochs 40 --patience 100 --lr 1e-4 --aug-aux --delta-corr-lambda 0.1 \
    --init-from $R/v8k/<split dir>/model.pth"
RUN=v8kdc10 MODEL=sinkhorn_rf_route_czc ... EXTRA_ARGS="$BASE $FT"
# F + ESP backbone for v29: v8kdi10 = MODEL=sinkhorn_rf_route_czc_espi, plus
#   "--esp-path evaluations/compile_check/results/esp_labels.pt --esp-lambda 0.1" (from v8k);
#   v8kdfi10 = MODEL=sinkhorn_rf_route_czc_espfi, --epochs 20, --init-from the v8kdi10 of the same split
```
Exact command lines for every split and seed: `evaluations/retrain/plan.json` (field `argv`).

### 3.4 Evaluate the F model
```bash
# add the model to calibration_shift/run_shift.py MODELS, then:
sbatch --dependency=afterany:<control job> --cpus-per-task=8 --mem=12G --time=01:00:00 \
  evaluations/calibration_shift/shift_k.sh predict_report
python evaluations/calibration_shift/decision_policies.py --models v8kdc10
```
Split metrics (R², regret, right device) are in each `metrics.json`.

### 3.5 Routed duration and ESP (v29, off-cluster; GPU recommended)
```bash
cd evaluations/route_stage
PYTHONPATH=../../src python test_unified.py          # must end with "all checks passed"
bash run_local.sh smoke29                            # 2 epochs on 300 circuits; check the epoch time
DEVICE=cuda GPUS="0 1" JOBS=4 bash run_local.sh train29
bash run_local.sh report                             # results/report_v24.md, section "v29"
```
v29 needs: `stage1.pt`, `stage1_synth2.pt`, `synth2_graphs.pt`, the dataset and synthetic graphs, `esp_labels.pt`,
`decomp_table.json`, and the 9 `model.pth` files of v8kdfi10
(`generalization_v10/results/v8kdfi10{,_s6,_s7}/{leave_one_family_out/qaoa, leave_one_group_out/variational, random_control/random_seed{5|6|7}}/`).

### 3.6 Fast path and light compile
```bash
bash evaluations/fastpath/run_local.sh               # routing-free validation (or fastpath/launch.sh on SLURM)
bash evaluations/light_compile/run_local.sh          # baseline: errors and times per size bin
```

---

## 4. Reference numbers

| what | value |
| --- | --- |
| v8kdc10, R² of F (3 seeds) | qaoa 0.970, qnn 0.847, variational 0.939, control 0.996 |
| v8kdc10, local calibration tracking | 0.66 (seed ensemble 0.69); fast path on routing-free pairs 0.94 |
| routed duration, median error vs the dataset | v27 qaoa 6.4%; light compile 5.4% (qaoa 6.6%); compiler floor (seed 1 vs seed 0) 3.9% (qaoa 4.8%) |
| routed ESP, abs Δlog ESP where ESP > 1e-3 | v8kdfi10 0.30–0.65; light compile 0.29; floor 0.20 (log units: 0.7 means ×2, not 0.7%) |
| time per circuit, 3 devices, by 2q gates 0–20 / 20–50 / 50–200 / 200–600 / 600+ | full compile 33 / 86 / 531 / 1845 / 2786 ms; light compile 19 / 37 / 69 / 132 / 190 ms; F model ~13 ms |

## 5. Known problems to keep in mind

1. Routed ESP is extremely sensitive to the duration: Δlog ESP ≈ (Δduration / duration) · |log ESP|, with log ESP ≈ −17.
2. Labels are mixed: F is the best of 3 seeds, while ESP and duration use seed 0 only.
3. The encoding decomposes `rzz` (qaoa) into cx·rz·cx, which leaves ~4.6% irreducible noise on qaoa durations.
4. The QExa20 T1/T2 ×10 fix is an assumption.
5. About 30 model versions were selected on the same test splits: freeze the model and evaluate it once on a test
   set that was never used for any choice.
