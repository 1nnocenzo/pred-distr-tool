# Generalization of the fidelity predictor

This study quantifies how well the GNN fidelity predictor transfers to workloads
it has never seen. It is **parallel to** `evaluations/pipeline/` and writes only
into `evaluations/generalization/results/` — the existing scheduling results are
never modified.

## Motivation

The published evaluation splits the benchmark set at random over circuits,
stratified by fidelity. The set is dominated by large, internally homogeneous
families:

| family | circuits | family | circuits |
|---|---|---|---|
| `vqe_two_local` | 3 449 | `qft` | 760 |
| `vqe_real_amp` | 2 861 | `qnn` | 760 |
| `vqe_su2` | 2 858 | `ae` | 758 |
| `qaoa` | 2 706 | `qftentangled` | 746 |
| `randomcircuit` | 1 234 | `iqpe` | 570 |

A random split therefore places near-identical circuits on both sides, so the
in-distribution score (R² ≈ 0.996) is an optimistic estimate of behaviour on an
unseen workload. Three leakage-free conditions are evaluated instead:

1. **Leave-one-family-out** — train on every family but one, test on the held-out
   family. One fold per family, for the ten large families by default.
2. **Size extrapolation** — train on circuits with at most *n* qubits (default
   *n* = 14), test on the larger ones (up to 20).
3. **Random-split control** — the original split, retrained with *this* code so
   the held-out numbers and the in-distribution number differ only by the split,
   not by the training driver.

All three use the tuned hyper-parameters of Table `tab:ml4qc:scheduling-hparams`
(`src/model/best_params.json`) and one frozen optimisation protocol
(`genstudy.model.TRAIN_PROTOCOL`: Adam, MSE, batch 32, ≤ 1000 epochs, early
stopping on validation MSE with patience 30). **Nothing is re-tuned on held-out
data**, and the validation split used for early stopping is drawn from the
training families only.

Finally, the held-out predictions are replayed through the unmodified scheduling
benchmark (`evaluations/pipeline/run_gnn_dispatch.py`), which shows whether the
policy stays close to GT-Weighted — the same scoring rule supplied with
ground-truth fidelities — when the fidelities it consumes come from a model that
never saw the workload.

## Reported metrics

Per held-out split, overall and per device (`EQE1_Top`, `EQE1_Bottom`, `QExa20`):
MAE, RMSE, R². R² is averaged uniformly over the three device outputs and is
accompanied by `target_std`, because a family whose fidelities are nearly
constant has almost no variance to explain and its R² must be read with that in
mind.

Two scheduler-relevant quantities are added: `device_choice_accuracy` (how often
the predictor's arg-max device is the ground-truth best device) and
`fidelity_regret_mean` (mean fidelity lost by following the predictor's choice).
The scheduling replay reports mean achieved fidelity versus the weight *w*, the
device agreement rate and the regret against both Oracle and GT-Weighted.

## Inputs

Reused from the compilation campaign (`compileCircuits/`, built by
`create_pt_file_new.py`):

| file | content |
|---|---|
| `graph_dataset_expected_fidelity.pt` | one PyG `Data` per circuit: DAG features, `y = [fid_EQE1_Top, fid_EQE1_Bottom, fid_QExa20]`, `circuit_name` |
| `names_list_expected_fidelity.npy` | circuit stems in dataset order (fallback if `circuit_name` is absent) |

The directory is located in this order: `--dataset-dir`,
`$COMPILE_CIRCUITS_DIR`, `data/graph_dataset/`, `data/`, `../compileCircuits/`,
`~/compileCircuits/`. Set the environment variable to make a run portable:

```bash
export COMPILE_CIRCUITS_DIR=/path/to/compileCircuits
```

From the repository itself: `src/model/gnn.py` (architecture),
`src/model/encoding.py` (circuit encoding), `src/model/best_params.json`
(hyper-parameters) and `src/model/expected_fidelity_results_benchmark.json`
(ground truth for the scheduling replay).

## Layout

```
evaluations/generalization/
  genstudy/                 importable library (no side effects on import)
    paths.py                repository + dataset path resolution
    data.py                 dataset loading, family/size parsing, split construction
    model.py                frozen architecture, hyper-parameters, training loop
    metrics.py              MAE / RMSE / R² + scheduler-relevant metrics
    experiment.py           run one split end-to-end, persist artefacts
    report.py               CSV, LaTeX tables, figures
    io.py                   JSON/CSV/logging helpers
    cli.py                  shared command-line flags
  scripts/
    run_random_split_control.py
    run_leave_one_family_out.py
    run_size_extrapolation.py
    run_scheduling_eval.py
    make_report.py
  slurm/
    submit_generalization.sh   whole study on one GPU node (partition gpu_a100)
  results/                     all output (git-ignored)
```

## Running

### On the cluster

```bash
sbatch evaluations/generalization/slurm/submit_generalization.sh
```

The job runs the control, the ten leave-one-family-out folds, the size
extrapolation, both scheduling replays and the report.

**Resuming.** Twelve trainings on ~15 k graphs can exceed the 24 h partition
limit, so progress is saved at two levels:

- *Per split* — a split whose `metrics.json` exists is skipped.
- *Per epoch* — the full training state (weights, optimiser, best-so-far
  weights, patience counter, history) is written to `<split>/checkpoint.pt`
  after every epoch and reloaded on restart, so an interrupted fold continues
  from its last epoch instead of from scratch. The file is written atomically
  (temp + rename) and deleted once the split completes; a corrupt or stale
  checkpoint is reported and the split trains from scratch rather than aborting
  the job.

Resubmitting the same command therefore always continues the study. To let it
continue automatically past the time limit, chain jobs with `afterany`:

```bash
first=$(sbatch --parsable evaluations/generalization/slurm/submit_generalization.sh)
next=$(sbatch --parsable --dependency=afterany:$first evaluations/generalization/slurm/submit_generalization.sh)
sbatch --dependency=afterany:$next evaluations/generalization/slurm/submit_generalization.sh
```

Each link resumes where the previous one stopped; a link that finds the study
already complete exits in minutes. Note that the training-batch order is not
restored across a resume (the shuffled `DataLoader` restarts), so a resumed fold
is not bit-identical to an uninterrupted one — the protocol, the split and the
early-stopping state are.

Folds can also be spread over several jobs:

```bash
FAMILIES=vqe_two_local,vqe_real_amp sbatch evaluations/generalization/slurm/submit_generalization.sh
```

Other overrides: `MAX_TRAIN_QUBITS`, `WEIGHTS`, `RUN_CONTROL=0`, `SMOKE=1`,
`CONDA_ENV`, `COMPILE_CIRCUITS_DIR`.

`SMOKE=1` trains 2 epochs per split and writes to `results_smoke/` instead of
`results/`, so a smoke run can never be mistaken for finished work by the resume
logic:

```bash
SMOKE=1 FAMILIES=qft WEIGHTS=0.0,0.5,1.0 \
  sbatch --time=00:30:00 evaluations/generalization/slurm/submit_generalization.sh
```

### Locally / step by step

```bash
# End-to-end smoke test of every code path (2 epochs per split, minutes not hours)
python evaluations/generalization/scripts/run_leave_one_family_out.py \
    --families qft --epochs 2 --patience 1 --no-save-model

# Full experiments
python evaluations/generalization/scripts/run_random_split_control.py
python evaluations/generalization/scripts/run_leave_one_family_out.py          # --families large
python evaluations/generalization/scripts/run_size_extrapolation.py            # --max-train-qubits 14
python evaluations/generalization/scripts/run_scheduling_eval.py --per-family \
    --fidelity-weights 0.0:1.0:0.1
python evaluations/generalization/scripts/make_report.py
```

Every script supports `--help`, `--dataset-dir`, `--results-dir`, `--device` and
the training flags; defaults reproduce the reported configuration.

## Outputs

```
results/
  random_control/random_seed5/        metrics.json, predictions.json, model.pth, history.json
  leave_one_family_out/
    <family>/                         same four artefacts per fold
    heldout_predictions.json          union over folds: one leakage-free prediction per circuit
    folds.json, dataset_summary.json, run.log
  size_extrapolation/
    qmax_14/                          metrics.json includes a per-test-size breakdown
  scheduling/
    leave_one_family_out/             dispatch_results.json, dispatch_summary.csv,
                                      cross_policy_summary.csv, sweep_summary.csv,
                                      by_family/<family>/...
    size_extrapolation_qmax_14/
  report/
    summary_*.csv                     tidy tables (one row per split and device)
    table_*.tex                       booktabs tables, ready for \input{}
    fig_*.pdf / fig_*.png             figures
    report_summary.json               headline numbers (control vs held-out)
```

`results/` is git-ignored: the JSON, checkpoints and figures are reproducible
from the scripts, and the per-fold checkpoints are large. Commit the contents of
`report/` alongside the manuscript if the numbers need to be archived.

## Reading the numbers

- Compare each held-out family against the control row of
  `table_leave_one_family_out.tex`. A large drop in R² with a comparatively small
  rise in MAE means the predictor still ranks devices usefully while losing
  calibration on the absolute fidelity; `device_choice_accuracy` separates the
  two cases, and it is the quantity the scheduler depends on.
- A family whose `target_std` is small can show a strongly negative R² while its
  MAE stays low. Report MAE/RMSE for those families and treat R² as uninformative
  rather than as evidence of failure.
- In the scheduling replay, the gap to **GT-Weighted** at the same *w* is the
  meaningful quantity: it isolates the cost of imperfect predictions from the cost
  of the policy's load-balancing term, since both policies use the identical
  scoring rule.
