import collections, sys
from pathlib import Path
import torch
EVAL = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths; paths.ensure_imports()
from gsv3.fast_encoding import encode
from qiskit import qasm3
data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
for name in ["bv/bv_q4_v0000", "qaoa/qaoa_q6_v0003"]:
    d = data[name]; qc = qasm3.loads((Path.home() / f"compileCircuits/benchmark_dataset_30k/{name}.qasm").read_text())
    x, ei, gq, nq = encode(qc)
    print(name, "shapes", tuple(x.shape), tuple(d.x.shape), "edges", ei.shape[1], d.edge_index.shape[1], "nq", nq, int(d.n_qubits))
    rows = lambda X, G: collections.Counter(tuple([round(v, 4) for v in r] + g) for r, g in zip(X.tolist(), G.tolist()))
    print("  same multiset of (features, operands) rows:", rows(x, gq) == rows(d.x, d.gate_qubits))
    print("  first 4 gates fast:", [int(r.argmax()) for r in x[:4]], gq[:4].tolist())
    print("  first 4 gates data:", [int(r.argmax()) for r in d.x[:4]], d.gate_qubits[:4].tolist())
