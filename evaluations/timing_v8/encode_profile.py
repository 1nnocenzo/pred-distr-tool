"""Where does the graph-building time go?  Stages of build_qubit_dataset._process, timed
separately on the circuits of the compilation-time benchmark."""
import json, re, sys, time
from pathlib import Path
import numpy as np, torch
EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL / "generalization_v3" / "scripts")); sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths; paths.ensure_imports()
from qiskit import qasm3, transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from encoding import create_dag, get_openqasm3_gates
comp = json.loads((Path.home() / "compileCircuits/compilation_time_results_benchmark_30k.json").read_text())
data = {d.circuit_name for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
names = sorted(n for n in comp if n in data)
qdir = Path.home() / "compileCircuits/benchmark_dataset_30k"
basis = get_openqasm3_gates()
T = {k: [] for k in ("parse", "transpile0", "create_dag", "transpile0_again", "dag_operands", "already_in_basis")}
for n in names:
    s = (qdir / f"{n}.qasm").read_text()
    t = time.perf_counter(); circ = qasm3.loads(s); T["parse"].append(time.perf_counter() - t)
    t = time.perf_counter(); tc = transpile(circ, optimization_level=0, basis_gates=basis); T["transpile0"].append(time.perf_counter() - t)
    T["already_in_basis"].append(all(i.operation.name in basis or i.operation.name in ("barrier", "measure") for i in circ.data))
    t = time.perf_counter(); create_dag(tc); T["create_dag"].append(time.perf_counter() - t)
    t = time.perf_counter(); qc = PassManager(RemoveBarriers()).run(tc); qc = transpile(qc, optimization_level=0, basis_gates=basis); T["transpile0_again"].append(time.perf_counter() - t)
    t = time.perf_counter(); dag = circuit_to_dag(qc); [[dag.find_bit(q).index for q in nd.qargs] for nd in dag.op_nodes()]; T["dag_operands"].append(time.perf_counter() - t)
tot = sum(np.array(T[k]) for k in ("parse", "transpile0", "create_dag", "transpile0_again", "dag_operands"))
print(f"{len(names)} circuits; median total {np.median(tot)*1e3:.1f} ms")
for k in ("parse", "transpile0", "create_dag", "transpile0_again", "dag_operands"):
    v = np.array(T[k]); print(f"  {k:18} median {np.median(v)*1e3:7.2f} ms   share of total {v.sum()/tot.sum():5.1%}")
print(f"  circuits already in the OpenQASM3 basis before transpile0: {np.mean(T['already_in_basis']):.0%}")

rows = json.loads((Path(__file__).resolve().parent / "timing_v8_threads8.json").read_text())
fwd = {r["circuit"]: r["forward"] for r in rows["rows"]}; exh = {r["circuit"]: r["exhaustive"] for r in rows["rows"]}
batched = rows["batched_forward_per_circuit"]
enc = np.array(T["transpile0"]) + np.array(T["create_dag"]) + np.array(T["dag_operands"])   # one transpile, no parsing
enc_inbasis = np.array(T["create_dag"]) + np.array(T["dag_operands"])                         # circuit already in the basis
F = np.array([fwd[n] for n in names]); E = np.array([exh[n] for n in names])
q = np.array([int(re.search(r"_q(\d+)_", n).group(1)) for n in names])
print("\nFair comparison (circuit already loaded, as for the compile timing; 3 devices):")
for lab, e in (("encode (1 transpile)", enc), ("encode, circuit already in basis", enc_inbasis)):
    for flab, f in (("forward batch 1", F), ("forward batched 64", np.full_like(F, batched))):
        P = e + f
        print(f"  {lab:34} + {flab:19}: median {np.median(P)*1e3:6.1f} ms  vs exhaustive {np.median(E)*1e3:6.1f} ms"
              f"  speedup median {np.median(E/P):4.1f}x  faster in {np.mean(E > P):4.0%}")
P = enc + F
for lo, hi in ((2, 5), (6, 10), (11, 15), (16, 20)):
    s_ = (q >= lo) & (q <= hi)
    print(f"  q{lo:>2}-{hi:<2} encode {np.median(enc[s_])*1e3:5.1f} ms + forward {np.median(F[s_])*1e3:5.1f} ms vs exhaustive {np.median(E[s_])*1e3:6.1f} ms  speedup {np.median(E[s_]/P[s_]):4.1f}x")
dev_unit = np.median(E) / 3
for nd in (3, 5, 10, 20):
    print(f"  {nd:2d} devices: exhaustive ~{dev_unit*nd*1e3:6.1f} ms vs predictor ~{np.median(enc)*1e3 + np.median(F)*1e3*(1 + 0.05*(nd-3)):5.1f} ms (forward cost grows little with devices; to be measured)")
