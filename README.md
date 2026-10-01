# Fidelity-Aware Scheduling of Quantum Circuits on Multi-QPU Systems

Minimal, cleaned-up version of the research repository: a GNN predicts the
expected fidelity of a circuit on each device of a multi-QPU system
(`EQE1_Top`, `EQE1_Bottom`, `QExa20`), and a scheduling policy uses those
predictions to select the best device for every circuit in a unified queue.
Dry-run experiments compare the GNN policy against Oracle, GT-Weighted, and
Round-Robin baselines on ground-truth fidelities.

## Structure

```
data/
  circuits.zip                  # benchmark dataset (OpenQASM 3, family/name.qasm)
src/
  model/
    gnn.py                      # GNN architecture
    encoding.py                 # circuit -> DAG graph encoding (extracted from predictor-gnn)
    best_model.pth              # trained checkpoint       (placeholder, awaiting new model)
    best_params.json            # model hyper-parameters   (placeholder, awaiting new model)
    expected_fidelity_results_benchmark.json  # ground-truth per-device fidelities
    test_circuit_names.json     # held-out test split (evaluation subset)
  qms/                          # minimal extract of QMS (policy engine only)
    policy/                     # MultiDevicePolicy base, registry, round-robin baseline
  predictor-gnn/                # (not tracked) local mqt-predictor fork, model
                                # training only; the pipeline does not import from it
evaluations/
  pipeline/
    gnn_device_policy.py        # GNN device-selection policy (registers "gnn_device")
    run_gnn_dispatch.py         # dry-run benchmark driver
    plot_benchmark.py           # figures from dispatch_results.json
  training/                     # (training evaluations)
```

## Setup

```bash
unzip data/circuits.zip -d data/    # -> data/benchmark_dataset_30k/
pip install torch torch_geometric qiskit qiskit-qasm3-import numpy matplotlib
```

## Run experiments

```bash
# Quick check (10 circuits, single weight)
python evaluations/pipeline/run_gnn_dispatch.py --max-circuits 10

# Full sweep over fidelity weights on the test split
python evaluations/pipeline/run_gnn_dispatch.py --fidelity-weights 0.0:1.0:0.1

# Drop circuits whose best ground-truth fidelity is below a threshold
python evaluations/pipeline/run_gnn_dispatch.py --min-best-fidelity 0.01 --fidelity-weights 0.0:1.0:0.1

# Plots (reads evaluations/pipeline/results/dispatch_results.json by default)
python evaluations/pipeline/plot_benchmark.py
```

By default the queue is sorted by circuit name, which delivers each family as a
contiguous block and penalizes the greedy scheduler. The thesis sweep
(`results_v4_shuffled/`) uses a shuffled queue and reuses saved predictions:

```bash
python evaluations/pipeline/run_gnn_dispatch.py --fidelity-weights 0.0:1.0:0.1 \
    --predictions-path evaluations/pipeline/results_v3_dense/fidelity_gnn_predicted.json \
    --shuffle-seed 0 --output-dir evaluations/pipeline/results_v4_shuffled

# Greedy vs optimal assignment under the same per-device load
python evaluations/pipeline/optimal_assignment.py \
    --results-dir evaluations/pipeline/results_v4_shuffled --queue-seed 0 --n-perm 0
python evaluations/pipeline/optimal_assignment.py \
    --results-dir evaluations/pipeline/results_v3_dense        # sorted queue + 20 permutations
python evaluations/pipeline/plot_optimal_assignment.py
```

Results (JSON + CSV + log) land in `evaluations/pipeline/results/`; plots in
`evaluations/pipeline/results/plots/`.

By default the benchmark is restricted to the held-out test split
(`--names-path src/model/test_circuit_names.json`); the oracle and all metrics
use ground truth, while the GNN policy decides from its own predictions.
