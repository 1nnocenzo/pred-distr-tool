# TODO: generalization study and paper

Target: npj Quantum Information collection "Foundations and Advances in HPC/AI-Quantum
Co-Design" (deadline 20 Dec 2026; check the collection's eligibility rules).

## In progress (2026-10-08)
- [ ] v5b (v5 + 150-epoch budget, cosine LR, SWA, Sinkhorn temperature 1.0 → 0.3):
      3 seeds on LOGO variational, LOFO qaoa/qnn, control. Goal: reduce the seed
      instability of v5 (variational 0.798 ± 0.103 vs v4 0.896 ± 0.002).
- [ ] Control models of v4/v5 with seeds 6–7 → repeat the calibration-shift test
      with 3 seeds per model (`evaluations/calibration_shift/run_shift.py predict/report`,
      add the new models to `MODELS`).
- [ ] Choose the main model (v4 vs v5/v5b), with seed ensembles (`generalization_v4/scripts/ensemble_seeds.py`).

## Next
- [ ] **Target ESP instead of plain expected fidelity**: include T1/T2 and gate
      durations (already in the `createDevice.py` targets). Extend the physics head with
      a learned per-gate duration/idle time × the device's real 1/T1, 1/T2.
      Needs the dataset recompiled (keep the compiled circuits this time).
- [ ] **Heterogeneous devices**: add 2–3 devices with different topology / size /
      native gates (e.g. Qiskit fake IBM backends) next to the current three; recompile
      in the same pass as the ESP targets; rerun LOFO/LOGO/calibration shift.
- [ ] **Runtime estimation** (same durations as the ESP).
- [ ] **Timing analysis**: predictor (graph build + model) vs compile + fidelity on the
      same circuits, including the heavy ones (grover, shor…); CPU and GPU, single and
      batched; scaling with the number of devices. Merge with the timing work already
      in `evaluations/pipeline/timing_results/`.
- [ ] **End-to-end scheduling**: queue simulation where the predictor chooses the device;
      compare with simple policies (best fixed device, random, …).

## Later / open issues
- [ ] Dynamic circuits (`iqpe`): inline the bodies of `if_else` blocks in the circuit
      graph (worst-case branch, as the ground truth does); rebuild the dataset; rerun
      the dynamic splits. See `generalization_v4/README.md`, "Known limitation".
- [ ] `qaoa` (5–8 qubits, routing-heavy): all models ≈ 0.88–0.93 R²; v5 worse.
- [ ] Training on device variants (calibration augmentation) so the placement learns
      to follow *local* calibration changes (shuffle/jitter tracking is ≤ 0.65 today).
- [ ] Regularisation (dropout / weight decay) against seed instability: postponed.
- [ ] Validation closer to hardware (noisy simulation or real runs) — likely asked by
      reviewers, since the ground truth is a calibration-based estimate.
- [ ] Release code + data for the paper.

## Notes for the paper
- Comparison with MQT Predictor: already in the QSW paper (our model is better at
  fidelity estimation and uses the device as input, so it can follow new calibrations).
  Cite it and recall the main numbers; no new experiment needed.
- Present the three devices as a small HPC centre; two of them come from the same chip
  (EQE1 Top/Bottom).
