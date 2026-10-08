# Generalization study v2: device-aware log-fidelity predictor

Follow-up to `evaluations/generalization/` (v1). v1 showed that the fidelity
predictor keeps the absolute fidelity level on unseen families but loses the
*ranking* between devices: on `vqe_su2` and `qaoa` its device-choice regret is
worse than always picking the single best device. v2 keeps the same splits,
seed, epoch budget and tuned circuit encoder, and changes three things:

1. **Log-fidelity target.** The model predicts `log F`; the loss is the MSE to
   `log(max(F, 1e-3))`. Expected fidelity is a product of per-operation
   `(1 - e)` terms, so `log F` is roughly additive and extrapolates better
   across families and sizes. Fidelities below `1e-3` are equivalent for
   scheduling and are clipped.
2. **Device-graph encoder.** Each device (`EQE1_Top`, `EQE1_Bottom`, `QExa20`) is
   a graph built from the same `BackendV2` targets used for the ground truth
   (`compileCircuits/createDevice.py`). Its nodes are qubits with
   `-log(1-e_r)`, `-log(1-e_readout)`, `log10 T1`, `log10 T2` and degree; its
   edges are couplers with `-log(1-e_cz)`. A GINE encoder embeds each device,
   and the head scores every (circuit embedding, device embedding) pair
   instead of emitting three fixed outputs. No compilation is needed at
   inference.
3. **Family-balanced sampling and out-of-distribution validation.** Training
   circuits are drawn with weight `n_family^-0.5`. Early stopping uses
   validation data that mimics the test condition:
   - LOFO: whole medium/small families. Families above half the validation
     budget always stay in training.
   - Size extrapolation: the largest training sizes (12–14 qubits for
     `qmax_14`).
   - Random control: a random subset, so it remains an in-distribution
     reference.

The circuit encoder (`GraphConvolutionSage` from `src/model/gnn.py`) and the
head widths use the tuned `src/model/best_params.json`. The device encoder (64
hidden, 3 layers) and fusion width (128) are fixed and untuned. Nothing is tuned
on held-out data.

## GPU only

Every entry point calls `gsv2.gpu.require_cuda()`. This checks that a CUDA
device exists and that a test kernel runs on it, and exits with status 3
otherwise. The SLURM script also checks `nvidia-smi` and excludes
`compute-7-12`, the node whose GPU failed in the v1 run. Unlike v1, the study
never falls back to the CPU.

## Running

```bash
sbatch evaluations/generalization_v2/slurm/submit_v2.sh
# subset / smoke test
FAMILIES=qaoa,vqe_su2 EXPERIMENTS=lofo sbatch evaluations/generalization_v2/slurm/submit_v2.sh
SMOKE=1 FAMILIES=qft EXPERIMENTS=lofo sbatch --time=00:30:00 evaluations/generalization_v2/slurm/submit_v2.sh
```

The job runs these steps:

1. `build_device_graphs.py`
2. `run_experiment.py {control,lofo,size}`
3. v1's `run_scheduling_eval.py` on the v2 predictions
4. `make_report.py`, which compares v1 and v2 on the same splits

Resuming and concurrency work as follows:

- Finished splits are skipped.
- Interrupted splits resume from a per-epoch checkpoint. The checkpoint is
  ignored if it was written with a different configuration.
- Each split is trained under an exclusive `flock`. A second job that reaches a
  split already in training skips it instead of training it in parallel, so
  chaining with `--dependency=afterany` or resubmitting is safe.

## Outputs

```
results/
  device_graphs.pt, device_graphs.raw.json
  random_control/random_seed5/        metrics.json, predictions.json, history.json, model.pth
  leave_one_family_out/<family>/      same; + heldout_predictions.json, folds.json
  size_extrapolation/qmax_14/         same; metrics.json has test_by_qubits
  scheduling/...                      v1 scheduling replay on v2 predictions
  report/
    comparison.csv                    v1 vs v2 per split (MAE, R², device acc., regret, baselines)
    table_comparison.tex
    fig_regret_lofo.pdf/png
    comparison_summary.json           LOFO means, folds beating the best fixed device
```

Each `metrics.json` also records which families (or qubit sizes) were used for
validation (`"validation"`), and the test `log_mse`.
