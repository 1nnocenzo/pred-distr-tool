# TODO: generalization study and paper

Target: npj Quantum Information collection "Foundations and Advances in HPC/AI-Quantum
Co-Design" (deadline 20 Dec 2026; check the collection's eligibility rules).

## In progress (2026-10-08)
- [ ] v5b (v5 + 150-epoch budget, cosine LR, SWA, Sinkhorn temperature 1.0 → 0.3):
      3 seeds on LOGO variational, LOFO qaoa/qnn, control. Goal: reduce the seed
      instability of v5 (variational 0.798 ± 0.103 vs v4 0.896 ± 0.002).
      Variational done: 0.800 ± 0.046 (spread halved, mean unchanged; its qnn subset
      −0.93) → no gain on variational. qaoa/qnn/control pending.
- [ ] **v6: Sinkhorn placement supervised by the compiler's layout** (auxiliary
      cross-entropy on the assignment `A` vs the initial layout chosen by Qiskit L2,
      `--layout-lambda`). Layout targets: `evaluations/compile_check/compile_layouts.py`
      over the whole dataset (grover separately, ~19 CPU-h). Measure: qaoa bias,
      seed spread, qnn; and with `compile_check` how close its layouts get to the
      compiler's. If close → extra paper section ("the predictor also proposes a layout").
- [ ] Control models of v4/v5 with seeds 6–7 → repeat the calibration-shift test
      with 3 seeds per model (`evaluations/calibration_shift/run_shift.py predict/report`,
      add the new models to `MODELS`).
- [ ] Choose the main model. **Keep both v4_phys and v5** (decision 2026-10-08):
      v4 generalises better to unseen families (qaoa 0.882, variational 0.896 ± 0.002),
      v5 survives calibration changes (calibration shift seed 5: v5 R² ≥ 0.908 on every
      variant; v4 drops to 0.518 on QExa20 ×2, 0.436 on QExa20 jitterA, 0.658 on
      EQE1_Bottom shuffle) but v4 picks the device better (acc 0.694 / regret 0.0040
      vs 0.654 / 0.0048). 3-seed calibration shift (+ v6) queued (`calib_predict`).
      v5b discarded (variational 0.800 ± 0.046, qaoa 0.609 ± 0.289, qnn 0.906 ± 0.023 vs v5 0.922 ± 0.040).
- [ ] **Why v4 generalises and v5 survives calibration changes** (diagnostics on the
      trained models, no retraining):
      1. channel swap in the calibration shift: variant calibration only in the device
         features (attention + count MLP) vs only in the physics-head errors, for v4
         and v5 → where the v4 collapse comes from (hypothesis: out-of-range device
         features into the count MLP);
      2. v5 with the placement forced uniform → if predictions barely move, v5 is
         robust because it uses the device-average error (consistent with its weak
         shuffle tracking: −0.07 / 0.45 / 0.06), not because it follows the change;
      3. predicted vs true compiled counts (`r`, `cz`) from the new QPY files, v4 vs v5,
         on qaoa and on the calibration variants → is the v5 qaoa bias in the 2q/SWAP
         count (hypothesis: Sinkhorn coupling + expected-distance input)?
      4. sharpness of the v4 attention (max weight, logical qubits sharing a physical
         one) on original vs variant devices, against the error.
      **Results 2026-10-08** (`diagnostics_v4v5/results/`, seed-5 models):
      - *Neither model really places qubits.* v4's attention is degenerate: every
        logical qubit of every circuit puts 0.50 of its weight on the same physical
        qubit of a device (node 8 on EQE1_Top, 10 on QExa20) and 0.25 on a second,
        independent of circuit and calibration. v5's Sinkhorn placement is near uniform
        (max row probability 0.175).
      - *v4 calibration collapse* = its errors come from that fixed handful of qubits.
        Channel swap: on EQE1 shuffle the collapse is entirely in the physics-head
        errors (errors only 0.658 = full 0.658, features only 0.976); on QExa20 ×2 /
        jitterA it needs both channels (full 0.518 / 0.436, each alone ≥ 0.92 / 0.55).
        v5 averages over the whole device, so local changes barely move it.
      - *v5 qaoa bias comes from its learned placement*: with a uniform placement the
        LOFO-qaoa model goes from R² 0.823 / bias +0.085 to **0.910 / +0.027**. On the
        calibration sample the uniform placement costs 0.02–0.1 R² (it helps on shuffle).
      - *Parts of −log F on qaoa (vs compiled QPY)*: both models put the cost in the
        wrong places — a readout part although qaoa has no measurements, and a 2q
        part 3–8× too small (q5–8: true 2q 1.20, v4 0.31, v5 0.47). The heads fit F, not
        the physical decomposition.
      - Scale changes (×0.5/×2) are carried by the physics-head errors in both models
        (features only: R² 0.66–0.85), as intended.
      - **3-seed calibration shift (2026-10-08 evening): the v4 collapse is seed 5 only.**
        Worst R² over the 18 variants: v4 s5 0.436, s6 0.928, s7 0.915 (mean R² 0.888 /
        0.984 / 0.984); v5 s5/s6/s7 0.908 / 0.911 / 0.969. Summary: v4 worst 0.760 ± 0.280,
        mean 0.952 ± 0.055; v5 worst 0.929 ± 0.034, mean 0.984 ± 0.004; device choice
        v4 0.675 / regret 0.004, v5 0.650 / 0.006. The v4 attention is degenerate in every
        seed (most logical qubits' top choice = one physical qubit: s6 node 7 / 6, s7
        nodes 22–24 / 6), so the collapse depends on *which* reference qubits a seed
        picked and how the variant changes them — a seed lottery, not a systematic flaw.
        The channel/uniform diagnostics above are on seed 5: repeat on s6/s7.
- [ ] **v6 pilot results (seed 5, λ = 0.1)**: variational 0.774 (λ 0.01: 0.844; v4 0.895,
      v5 0.901), qaoa 0.822 (v4 0.893, v5 0.823). The placement learns the compiler's
      layout but **only within families**: exact-qubit agreement on test circuits 86%
      (control, same families) vs 7–8% on unseen families (qaoa, variational; random
      layout 5%), while confident (max row prob 0.50 vs 0.22 for v5) → it reads the
      errors of the wrong qubits; bias −0.08 on variational. (`diagnostics_v4v5/
      v6_variational.py`, `v6_layout_acc.py`.) Compiled v6 qaoa layouts: compile_check
      running (`compile_check/results/qaoa_layouts_v6`); they are more compact than
      v5's (2.6 vs 3.1–3.8 hops).
- [ ] **v7 (pilot running)**: same network, layout loss on family-agnostic properties
      only (`--layout-loss region_dist`): BCE on which physical qubits are used (column
      sums of `A`) + MSE on log1p of the operand chip distance of every multi-qubit gate
      vs the compiler's. Seed 5, λ 0.01 / 0.1, 4 splits, `generalization_v7/results`;
      calibration shift queued. Check: test-set region overlap / distance error on unseen
      families, R² / bias on qaoa and variational.
- [ ] `phys_uniform` (v4 without placement, retrained; `generalization_v6/results/v4u_phys`):
      separates "placement" from "calibration of the head" in the uniform-attention test.
- [ ] **Pilot results, seed 5 (2026-10-09)** — test R² qaoa / qnn / variational / control;
      calibration shift worst R² / mean R² / local tracking / device choice:
      | model | qaoa | qnn | variational | control | calib worst | mean | local trk | choice |
      | v4_phys s5 | 0.893 | 0.844 | 0.895 | 0.994 | 0.436 | 0.888 | ~0.03 | ~0.68 |
      | v5 s5 | 0.823 | 0.968 | 0.901 | 0.997 | 0.908 | 0.983 | ~0.10 | ~0.65 |
      | v6 λ0.01 | 0.549 | 0.645 | 0.844 | 0.996 | 0.889 | 0.977 | 0.238 | 0.536 |
      | v6 λ0.1 | 0.822 | 0.721 | 0.774 | 0.995 | 0.898 | 0.976 | 0.241 | 0.667 |
      | phys_uniform | 0.740 | 0.897 | 0.756 | 0.994 | 0.932 | 0.979 | −0.159 | 0.772 |
      | **v7 λ0.01** | 0.810 | 0.781 | **0.894** | 0.997 | **0.967** | **0.988** | 0.234 | 0.711 |
      | **v7 λ0.1** | **0.934** | 0.615 | 0.830 | 0.996 | 0.964 | 0.987 | **0.367** | **0.740** |
      v7 best on calibration (worst R² ≥ 0.96, highest local tracking) and on qaoa (λ 0.1)
      or variational (λ 0.01), but qnn drops (0.62–0.78 vs v5 0.968). Next: 3 seeds of
      v7 (both λ), and understand the qnn drop. v6 discarded.
- [ ] **Low-noise calibration shift** (`calibration_shift/shift_k.py`, `results_k/`; truth =
      best of 5 compiler seeds; + moderate variants ×0.7/0.8/1.2/1.3 and jitterS σ 0.2):
      - moderate global ±20–30 %: every model R² ≈ 0.99, tracking 0.96–0.99 → solved;
      - local (shuffle/jitter): split-half noise ceiling now 0.6–0.99 (mostly 0.8–0.9),
        so low tracking is a **model** limitation: v4/v5/phys_uniform ≈ 0 (−0.14…+0.16),
        v6 0.27–0.34, **v7 λ0.1 0.43** (best; abs R² 0.988, worst 0.980).
        (The earlier "ceiling ≈ 0" estimate from single compiles was too pessimistic.)
- [ ] **qnn drop of v7** (`diagnostics_v4v5/qnn_drop.py`): v7's layout on qnn transfers
      (distance error 0.15–0.18 vs v5 0.55, region 0.84 vs 0.59), but its count head
      over-predicts the 1q part (q11–14: v7 λ0.1 0.27 vs true 0.14) → bias −0.08/−0.14.
      v5 gets qnn right by a wrong but lucky allocation (1q ≈ 0.03, 2q 0.40 vs true 0.22).
      Learned scales s_k ≈ 1 everywhere: the error is in the counts n_k. Fix idea:
      **supervise the counts** (true r / measure / cz per circuit and device from the
      QPY files) so the decomposition becomes physical.
- [ ] **v8a (v7 λ0.1 + count loss 0.01), 3 seeds (2026-10-09)** — candidate main model:
      qaoa 0.892 ± 0.081 (0.930 / 0.947 / 0.800), qnn 0.843 ± 0.037, variational
      0.880 ± 0.037, control 0.996; local calibration tracking **0.46 ± 0.03**
      (v4 ≈ 0, v5 ≈ 0.05), local worst R² 0.965–0.978, moderate ±20–30 % R² 0.99.
      Weak spot: seed 6 worst R² 0.762 on the large-scale (×0.5/×2) variants; seed 7
      lower on qaoa (0.800). Count weight 0.1 is worse everywhere (discarded).
      v4 for reference: 0.882 ± 0.026 / 0.670 ± 0.139 / 0.896 ± 0.002; v5: 0.709 ± 0.105 /
      0.922 ± 0.040 / 0.798 ± 0.103.
- [ ] **v8b (v8a + calibration variants in training, seed 5)**: qaoa 0.844 (device
      choice 57 % vs 30 %, regret 0.0135), variational 0.801; qnn/control running;
      calibration test queued. So far: better device ranking, worse absolute fidelity on
      unseen families. Next if confirmed: p(orig) 0.5.
- [ ] **Timing (2026-10-09, `evaluations/timing_v8/`)**: QASM 3 parsing was 77 % of the old
      encode time; the compile benchmark times a circuit already in memory, so compare
      the same way. New `gsv3/fast_encoding.py` (one pass over the instructions, ≤ 1
      transpile, identical graphs on 3,412 circuits, |ΔF| ≤ 6e-7): encode 11.2 → 2.2 ms.
      Exhaustive compile + fidelity vs predictor (1 thread, same 288 circuits, devices =
      3 originals + 24 calibration variants): 3 devices 66.7 vs 12.9 ms (5.4×), 6: 7.8×,
      12: 10.7×, 18: 12.2×, 27: 567 vs 44 ms (13.2×); q16–20 at 27 devices 2.96 s vs
      80 ms (37×). Exhaustive grows ~21 ms/device, the predictor ~1.2 ms/device.
      To do: GPU / multi-thread batched inference; heavy circuits (shor, large qft).
- [ ] Option, depending on v6/v7: **train on calibration variants** (compile the training
      circuits on perturbed devices → labels and layouts that move with the
      calibration), so the placement learns to follow local changes., with seed ensembles (`generalization_v4/scripts/ensemble_seeds.py`).

## Next
- [ ] **Target ESP instead of plain expected fidelity**: include T1/T2 and gate
      durations (already in the `createDevice.py` targets). Extend the physics head with
      a learned per-gate duration/idle time × the device's real 1/T1, 1/T2.
      Needs the dataset recompiled (keep the compiled circuits this time).
      In the same pass: **best-of-K labels** (several seeds and/or L3) and the
      compiler's layouts, stored per circuit and device (see findings below).
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

## Grover excluded from the recompilation (2026-10-09)
- `grover` is left out of the new compilation (layouts, QPY, counts, v8b variants): the
  memory required to compile it would be very heavy. Its compiled circuits reach
  ~2.7 GB each as QPY (889 circuits ≈ 490 GB), jobs with 8 GB of memory were killed
  (out of memory), and its expected fidelity is ≈ 0 on every device, so it carries no
  information for the fidelity/ESP targets. The QPY files were deleted; the original
  grover labels in the dataset are untouched. To state in the paper.

## Findings 2026-10-08 (compile_check, qaoa sample of 304 circuits)
- All 12,961 recompilations valid on the device and equivalent to the original
  circuit (statevector check on random inputs; negative controls fail as expected).
- **Label noise**: one L2 compile per label. Other seeds move F by up to ~0.05 at
  5–8 qubits; best of 5 L2 seeds +0.011 on average, best of 3 L3 seeds +0.016
  (+0.030 at 5–8 qubits); the best device changes in 7% (L2) / 18% (L3) of circuits.
  On 1592 circuits recompiled for the timing study the best device changed in 16%.
- **v5 placement is not a layout**: max row probability of `A` ≈ 0.17–0.20; its
  rounded layouts, compiled, lose ~0.12 F vs the label (random layout: −0.19) and
  never beat best-of-K. So the v5 qaoa overestimate is a miscalibrated count head,
  not a better layout found by the model.
- Scripts: `evaluations/compile_check/` (`compile_check.py`, `extract_layouts.py`,
  `analyze.py`).

## Later / open issues
- [ ] Dynamic circuits (`iqpe`): inline the bodies of `if_else` blocks in the circuit
      graph (worst-case branch, as the ground truth does); rebuild the dataset; rerun
      the dynamic splits. See `generalization_v4/README.md`, "Known limitation".
- [ ] `qaoa` (5–8 qubits, routing-heavy): all models ≈ 0.88–0.93 R²; v5 worse.
- [ ] (moved up: "train on calibration variants", under In progress.)
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
