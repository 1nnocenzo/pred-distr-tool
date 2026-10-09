# Generalization study v4: physics-structured head for unseen circuit families

v4 targets the families that have no sibling in the training set (`qaoa`,
`qnn`, `iqpe`, `randomcircuit`) and the leave-one-group-out (LOGO) hold-outs.
On those, v1 and v3 still fail badly. v3 has R² 0.32 on `qnn` and −0.94 on
`iqpe`.

## Why v3 fails on `qnn` and `iqpe`

These families are shallow: their fidelity decays smoothly with size, from
about 0.95 at 2 qubits to about 0.27 at 20, and is never close to zero. Almost
every other family is at F ≈ 0 by 20 qubits. The models learn the shortcut
"many qubits ⇒ F ≈ 0", and on these families they are wrong by 0.15–0.30.

## What v4 changes

The code is in the shared `generalization_v3/gsv3` package, as model kind
`phys`.

1. **Physics-structured head (`PhysicsHeadPredictor`).**
   - The expected fidelity is `Π_op (1 − e_op)` over the *compiled* operations,
     so `log F = −Σ_op ε_op`.
   - The model predicts, for each gate and each device, how many native
     operations it becomes: `n_1q`, `n_2q` (routing SWAPs included) and
     `n_meas`.
   - Each count is multiplied by the **calibrated** error of the physical
     qubits that the cross-attention places the gate's operands on. These are
     the device's real `−log(1−e)` values for 1q gates, readout and mean
     incident CZ.
   - The model learns compilation (counts and placement); the error magnitudes
     come from the device. Nothing is compiled at inference, and no hand-made
     feature is added.
2. **Mixed loss:** `MSE(F) + 0.1·MSE(log F)`. The linear term is what R²
   measures, and v1, trained on it, was best on the QFT families. The log term
   keeps the additive structure informative at low F.
3. **Random (in-distribution) validation, as in v1, selected on linear MSE.**
   Validating on unrelated held-out families made the selected epoch almost
   random in v3, sometimes epoch 1 or 2.
4. **α = 0:** no family balancing, because balancing hurt the VQE families in v3.

A first 2-epoch smoke test on `qnn` gave R² 0.80, compared with −0.39 for v1
and 0.32 for v3. The maximum gradient norm was about 4, compared with 30–90 in
v3. This is a single short run, not a result.

## Runs (plan: `results/plan_v4.txt`)

| config | model | loss | validation |
|---|---|---|---|
| `v4_phys` | `phys` | mixed | random |
| `v4_xattn_mixed` | `xattn` (v3 head) | mixed | random |

`v4_xattn_mixed` is the ablation. It separates the effect of the physics head
from that of the loss and validation changes.

Each config is evaluated on:

- LOFO on all 10 large families, seed 5;
- LOGO on `vqe`, `fourier` and `variational` (see `gsv3/groups.py`), seeds 5–7;
- extra seeds 6–7 on the LOFO families without siblings;
- the random control and size extrapolation, seed 5.

In parallel, `generalization_v3/results/plan_logo.txt` runs v1, `xattn_a05`
and `xattn_a00` on the same LOGO groups and seeds, for the baseline.

## Running

The runs are driven by a SLURM supervisor. Everything runs on compute nodes,
never on the login node.

```bash
PLAN=evaluations/generalization_v4/results/plan_v4.txt \
RESULTS_ROOT=evaluations/generalization_v4/results SBATCH_EXTRA="--time=12:00:00" \
  sbatch --output=evaluations/generalization_v4/results/slurm/babysit_%j.log \
  evaluations/generalization_v3/slurm/babysit.sh

# progress / results (read-only, fine on the login node)
RESULTS_ROOT=evaluations/generalization_v4/results \
  python3 evaluations/generalization_v3/scripts/pending.py evaluations/generalization_v4/results/plan_v4.txt
python3 evaluations/generalization_v3/scripts/summarize.py
```

`summarize.py` aggregates v1, v3 and v4 over seeds (mean ± std) for R², MAE
and regret. For LOGO it also reports each family inside the held-out group.
The tables are written to `results/summary_<metric>.csv`.

## Known limitation: dynamic circuits (`iqpe`), set aside for now

All models strongly underestimate `iqpe`, and v4 `phys` does so most. For
example, LOGO `fourier` at "10 qubits" has true F 0.63 and v4 predicts 0.06.
The cause is the encoding, not generalisation:

- `iqpe_qN` uses **2 qubits**; N is the number of precision bits. It contains
  N−1 mid-circuit measurements and many classically conditioned blocks:
  `iqpe_q10_v0000` has 45 `if (c[k]) { x | rz }`.
- The graph turns each `if` block into one opaque `if_else` node, so its body
  is invisible to the model. The ground truth (`_fidelity_recursive` in
  `compileCircuits/testComp-random-EQE1-complete.py`) **counts** the body,
  using the worst-case branch.
- `iqpe` holds 570 of the 594 dynamic circuits and 39 900 of the 41 262
  `if_else` nodes. Once it is held out, training sees almost no `if_else`:
  24 circuits in LOFO, 5 in LOGO `fourier`.
- The qubit count in the file name, used by the size split and the per-qubit
  breakdowns, is not the real number of qubits for `iqpe`.

Decision (2026-10-08): `iqpe` is reported separately as a known limitation and
excluded from the generalisation conclusions. For LOGO `fourier`, read the
per-family numbers or the group R² without `iqpe`.

Possible fix for later: inline the bodies of control-flow blocks as ordinary
gate nodes with a "conditional" flag, which mirrors the ground-truth
accounting, then rebuild the dataset (about 3 min) and rerun the dynamic
splits.
