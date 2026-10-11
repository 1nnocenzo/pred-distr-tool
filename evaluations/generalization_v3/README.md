# Generalization study v3: stable training + qubit-aware cross-attention predictor

Follow-up to `generalization_v2/`. The analysis of the first five v2
leave-one-family-out (LOFO) splits showed two problems:

1. **Unstable training.** In 4 of 5 splits the training loss diverged between
   epochs 24 and 46. The train log-MSE went from about 0.05 to between 10 and
   9860. Early stopping then kept a model from before the divergence. That
   model was undertrained, with best epoch between 17 and 36, compared with
   100–180 in v1. This explains most of v2's R² drop relative to v1, for
   example vqe_su2 going from 0.88 to 0.69.
2. **A systematic offset per held-out family**, present in v1 too. Ranking
   within the family is good (Spearman 0.97–0.99), but all of the family's
   predictions are shifted by an almost constant amount. For vqe_su2 in v2,
   removing the mean bias alone raises R² from 0.69 to 0.88. The model gets
   the *compilation cost* of unseen gate patterns wrong. It cannot learn that
   cost well because:
   - the circuit encoding does not record which logical qubit each gate acts on;
   - circuit and device meet only as two pooled vectors, so the model cannot
     relate the circuit's interaction structure to the coupling map, which is
     what routing depends on.

The model must stay end-to-end: raw circuit + raw devices → fidelity, without
compiling and without hand-made features such as routing estimates. v3 removes
the two obstacles without adding any heuristic:

| run | model | training | dataset |
|---|---|---|---|
| `pooled_stab_a05` | v2 `DeviceAwarePredictor`, unchanged | grad-clip 1.0, lr 3e-4, α=0.5 | original |
| `pooled_stab_a00` | same | same, α=0 (no family balancing) | original |
| `xattn_a05` | `QubitCrossAttentionPredictor` | grad-clip 1.0, lr 3e-4, α=0.5 | + gate operands |
| `xattn_a00` | same | same, α=0 | + gate operands |

- `pooled_*` isolates the effect of stabilisation alone, and of family
  balancing, which may hurt the VQE families by down-weighting their sibling
  families.
- `xattn_*` is the new model (see `gsv3/model.py`):
  1. The circuit encoder is the tuned `GraphConvolutionSage`, used at node level.
  2. Logical qubits are built from the gates that touch them, then message
     passing runs over the interaction graph from the circuit.
  3. Cross-attention to the device's physical qubits gives a learned, soft layout.
     The device side is GINE over the coupling map plus Laplacian positional
     encodings.
  4. Each gate gets a cost ≥ 0, and `log F = −Σ cost`.

Everything else matches v2: loss (MSE on `log max(F, 1e-3)`), splits, seed,
validation modes, batch size, patience and the circuit-encoder
hyper-parameters.

## Isolation from v2

- v3 writes only under `evaluations/generalization_v3/`.
- It imports `genstudy` (v1), `gsv2` (v2) and `src/model` read-only.
- The qubit dataset is a new file in `data/`. The original
  `~/compileCircuits/graph_dataset_expected_fidelity.pt` is only read.
- `build_qubit_dataset.py` re-runs the original pre-processing and checks that
  every recomputed node-feature matrix equals the stored `x`. It exits with an
  error if a circuit fails or differs, so the splits are identical to v1 and v2.

## Running

From the repository root:

```bash
d=$(sbatch --parsable evaluations/generalization_v3/slurm/build_dataset.sh)
RUN=pooled_stab_a05 MODEL=pooled ALPHA=0.5 sbatch --job-name=pooled_stab_a05 evaluations/generalization_v3/slurm/train.sh
RUN=xattn_a05 MODEL=xattn ALPHA=0.5 sbatch --job-name=xattn_a05 --dependency=afterok:$d evaluations/generalization_v3/slurm/train.sh
# comparison with v1/v2 (works on partial results)
python evaluations/generalization_v3/scripts/compare.py --metric r2
```

`train.sh` requests a GPU (`gpu_a40`, `--gres=gpu:1`) and aborts if CUDA is
not usable. Unlike v2, there is no silent CPU fallback.
