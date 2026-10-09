"""fast_encoding.encode must give the dataset graph up to node order (Qiskit's transpile
re-emits instructions in its own topological order).  Two checks per circuit:
1. structure: each node identified as (first operand qubit, k-th instruction on it), an
   order-independent id; features, operands and the edge multiset compared through it;
2. model: v8a predictions on both graphs (the check that matters).
    python check_fast_encoding.py [step]
"""
import collections, sys
from pathlib import Path
import torch
EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths; paths.ensure_imports()
from gsv3.fast_encoding import encode
from gsv3.devices import load_devices
from gsv3.model import build_model, load_hparams
from genstudy.data import DEVICE_NAMES
from torch_geometric.data import Batch, Data
from qiskit import qasm3

def node_ids(gq):
    seen = collections.Counter(); ids = []
    for row in gq.tolist():
        w = row[0]; ids.append((w, seen[w]))
        for q in row:
            if q >= 0: seen[q] += 1
    return ids

def canon(x, ei, gq):
    ids = node_ids(gq)
    nodes = {ids[i]: (tuple(round(v, 4) for v in x[i].tolist()), tuple(gq[i].tolist())) for i in range(len(ids))}
    edges = collections.Counter((ids[a], ids[b]) for a, b in ei.t().tolist())
    return nodes, edges

data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
step = int(sys.argv[1]) if len(sys.argv) > 1 else 5
params = load_hparams()
devices = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
model = build_model("sinkhorn", params, devices, torch.device("cpu"))
model.load_state_dict(torch.load(EVAL / "generalization_v8/results/v8a_c001/random_control/random_seed5/model.pth", map_location="cpu")); model.eval()
qdir = Path.home() / "compileCircuits/benchmark_dataset_30k"
bad = collections.Counter(); n = 0; first = []; maxdiff = 0.0
for d in data[::step]:
    qc = qasm3.loads((qdir / f"{d.circuit_name}.qasm").read_text())
    try:
        x, ei, gq, nq = encode(qc)
    except Exception as e:
        bad["error"] += 1; n += 1; first.append((d.circuit_name, repr(e)[:100])); continue
    n += 1
    if nq != int(d.n_qubits) or x.shape != d.x.shape:
        bad["shape"] += 1; first.append((d.circuit_name, "shape")); continue
    a, b = canon(x, ei, gq), canon(d.x, d.edge_index, d.gate_qubits)
    if a[0] != b[0]: bad["nodes"] += 1; first.append((d.circuit_name, "nodes")) if len(first) < 8 else None
    if a[1] != b[1]: bad["edges"] += 1; first.append((d.circuit_name, "edges")) if len(first) < 8 else None
    with torch.no_grad():
        p1 = model(Batch.from_data_list([Data(x=x, edge_index=ei, gate_qubits=gq, n_qubits=nq, num_nodes=x.size(0))]), devices)
        p2 = model(Batch.from_data_list([Data(x=d.x, edge_index=d.edge_index, gate_qubits=d.gate_qubits, n_qubits=int(d.n_qubits), num_nodes=d.x.size(0))]), devices)
    diff = float((torch.exp(p1.clamp(max=0)) - torch.exp(p2.clamp(max=0))).abs().max())
    maxdiff = max(maxdiff, diff)
    if diff > 1e-4: bad["prediction"] += 1
print(f"checked {n} circuits; mismatches: {dict(bad) or 'none'}; max |F_fast - F_dataset| = {maxdiff:.2e}")
for f in first[:8]: print("  ", f)
