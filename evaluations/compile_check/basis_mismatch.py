#!/usr/bin/env python3
"""Why Sabre labels and dataset labels differ on some families: the same circuit compiled (Qiskit level 2, seed 0)
from the original QASM (the dataset labels) and after fast_encoding.to_basis (sabre_labels.py). Reads data/circuits.zip;
qaoa gets measure_all() as on the cluster.  2026-10-15: qaoa 75 cases: ratio 1.043, |diff| median 4.6%, p90 15.4%,
99% differ (the sabre_labels mismatch); vqe_su2 0.6% / 0.7%; bmw_quark_copula 6.0% / 21.8%.
    PYTHONPATH=src python evaluations/compile_check/basis_mismatch.py qaoa 25
"""
sys.path[:0] = [".", "src", "evaluations/compile_check", "evaluations/generalization_v3"]
from qiskit import qasm3, transpile
from qiskit.transpiler import generate_preset_pass_manager
from v8b_variants import variants
from gsv3 import fast_encoding as fe
base = {k.split("/")[0]: be for k, be in variants(orig=True).items()}
z = zipfile.ZipFile("data/circuits.zip")
fam = sys.argv[1]; n = int(sys.argv[2])
names = sorted(x for x in z.namelist() if f"/{fam}/" in x and x.endswith(".qasm"))
random.Random(0).shuffle(names)
def dur(tc, be):
    sc = transpile(tc, target=be.target, scheduling_method="asap", optimization_level=0, initial_layout=None, routing_method=None, layout_method=None)
    return sc.estimate_duration(target=be.target) * 1e6, tc.count_ops().get("cz", 0)
rows = []
for x in names[:n]:
    qc = qasm3.loads(z.read(x).decode())
    if fam == "qaoa" and "measure" not in qc.count_ops():
        qc.measure_all()                      # as qaoa_measure.py did on the cluster
    if qc.num_qubits > 20: continue
    qb = fe.to_basis(qc)
    for d, be in base.items():
        pm = generate_preset_pass_manager(optimization_level=2, target=be.target, seed_transpiler=0)
        a, cza = dur(pm.run(qc), be)                 # dataset labels: the original circuit
        pm = generate_preset_pass_manager(optimization_level=2, target=be.target, seed_transpiler=0)
        b, czb = dur(pm.run(qb), be)                 # Sabre labels: the circuit after to_basis
        rows.append((b / a, czb / max(cza, 1)))
r = np.array(rows)
e = np.abs(r[:, 0] - 1)
print(f"{fam}: {len(r)} cases; duration to_basis / original: median {np.median(r[:,0]):.3f}, |diff| median {np.median(e):.1%}, "
      f"p90 {np.percentile(e, 90):.1%}, differ > 0.1% in {np.mean(e > 1e-3):.0%}; CZ ratio median {np.median(r[:,1]):.3f}")
