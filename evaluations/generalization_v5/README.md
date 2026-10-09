# Generalization study v5: physics head + Sinkhorn placement

v5 is v4 (`phys`) with one change: how logical qubits are placed on the chip
inside the network. The code is in the shared `generalization_v3/gsv3`
package, model kind `sinkhorn` (`SinkhornPlacementPredictor`).

## Motivation

On `qaoa` (LOFO, seed 5), v4 is the best model on small (3–4 qubit) and large
(≥10 qubit) circuits, but has errors of up to 0.21 at 5–8 qubits. That is
where the fidelity collapses because routing starts to add SWAPs. Qiskit
(`optimization_level=2`) places qubits as follows:

- **VF2Layout first:** if a layout exists in which every interacting pair is
  directly coupled, it picks the one with the lowest errors.
- **SabreLayout otherwise:** it minimises SWAPs.
- **VF2PostLayout after routing:** it moves to lower-error qubits when the
  structure allows it.

The number of SWAPs depends on the chip distance between interacting logical
qubits. In v4 each logical qubit attends to the physical qubits independently,
so two logical qubits can sit on the same physical qubit, and that distance is
blurred.

## The change: placement only, no compilation

- For each circuit and device, the network scores every (logical, physical)
  qubit pair.
- **Sinkhorn** iterations normalise the scores (log domain, temperature 0.3,
  20 iterations) into a soft one-to-one assignment: each logical qubit has
  mass 1, each physical qubit at most 1, and unused physical qubits go to
  dummy rows.
- This is a differentiable layer learned from the fidelity targets. Nothing
  is compiled.
- The count head also receives, for each gate, the **expected chip distance**
  between its operands, `A_aᵀ H A_b`. `H` is the hop-distance matrix of the
  coupling map, which is raw device structure.
- Everything else is as in v4:
  - placement-weighted calibrated errors;
  - `log F = −Σ_g Σ_k s_k n_k ε_k`;
  - mixed loss;
  - random validation;
  - α = 0.

## Checks (compute node)

- Assignment: row sums 1.000, column sums ≤ 0.974.
- Hop-distance diameters: 9, 11 and 6.
- A 2-epoch run on `qnn` trains stably, with no NaN after replacing `-inf`
  masking by a finite value.
- Cost per epoch is about the same as v4: around 100 s on 8 CPU cores.

## Pilot plan (`results/plan_v5_pilot.txt`, seed 5, 8 splits)

| experiment | splits |
|---|---|
| control | random_seed5 (sanity check: must stay ≈ 0.995) |
| LOFO | qaoa, qnn, iqpe, randomcircuit |
| LOGO | vqe, fourier, variational |

Compare it with `v4_phys` seed 5. If the pilot helps, extend it to seeds 6–7
and to the full LOFO.

```bash
PLAN=evaluations/generalization_v5/results/plan_v5_pilot.txt \
RESULTS_ROOT=evaluations/generalization_v5/results SBATCH_EXTRA="--time=06:00:00" \
  sbatch --job-name=v5_babysit \
  --output=evaluations/generalization_v5/results/slurm/babysit_%j.log \
  evaluations/generalization_v3/slurm/babysit.sh
```
