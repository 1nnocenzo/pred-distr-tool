"""Mean predicted vs. true fidelity per qubit count for one family (run on a compute node).

    python bias_by_qubits.py FAMILY PRED_JSON [PRED_JSON ...]
"""
import json, re, sys
import numpy as np, torch
fam, files = sys.argv[1], sys.argv[2:]
data = torch.load("evaluations/generalization_v3/data/graph_dataset_expected_fidelity.pt", weights_only=False)
Y = {d.circuit_name: d.y.view(-1).numpy() for d in data if d.circuit_name.startswith(fam + "/")}
for f in files:
    p = json.load(open(f)); names = [n for n in p if n in Y]
    q = np.array([int(re.search(r"_q(\d+)_", n).group(1)) for n in names])
    P = np.array([p[n] for n in names]).mean(1); T = np.array([Y[n] for n in names]).mean(1)
    print(f.split("results/")[1].rsplit("/", 1)[0])
    print("   q   " + " ".join(f"{k:>5}" for k in range(2, 21, 2)))
    print("  true " + " ".join(f"{T[q == k].mean():5.2f}" for k in range(2, 21, 2)))
    print("  pred " + " ".join(f"{P[q == k].mean():5.2f}" for k in range(2, 21, 2)))
