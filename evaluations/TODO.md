# TODO: generalization study and paper

Target: npj Quantum Information collection "Foundations and Advances in HPC/AI-Quantum
Co-Design" (deadline 20 Dec 2026; check the collection's eligibility rules).

## In progress (2026-10-08)
- **Light compile instead of the routed model (2026-10-16, `light_compile/`, 800 circuits × 3 devices, 1 thread):** switch
  (cheap checks + VF2 limit 2000: median 0.04 ms, max 9 ms, 99.2% agreement with the compiler, 23 / 1007 routing-free
  cases lost) → routing-free exact; routed → level 2 with the light Sabre, physics computed exactly. Routed vs level 2
  seed 0: duration 5.4% (seed-1 floor 3.9%), qaoa 6.6% (floor 4.8%; v27 model 8.8%); abs Δlog ESP where ESP > 1e-3: 0.29
  (floor 0.20; v8kdi10 + v19i model 0.30-0.65). Time −70% overall, 0.15-0.27× the full compile above 50 2q gates.
  Open: is the route-stage model still needed (only if it beats this at lower cost); best-of-3 labels vs one light run.
- **v29 (2026-10-16, to launch, replaces v28): ONE model for F, duration and ESP** (`route_stage/unified.py`, RUNBOOK
  "v29"): frozen v8kdfi10 F backbone + v27 route stage on its gate embeddings + ESP formula on the backbone's hard placement
  (same placement for F and decay; no second F model call; vectorised MC schedule). Why: the v28 path ran two models with
  two placements (F model's for F, light Sabre's for the decay), and v28's light-Sabre data were never produced. Adopt if
  routed ESP beats v8kdfi10 (abs Δlog ESP 0.53 / 0.65 / 0.30) toward the light compile (0.29) at a lower time per circuit
  than the light compile in the same size bin. Needs the 9 v8kdfi10 model.pth from the cluster.
- **v28 (2026-10-16, parked: needs results/sabre_light.pt): light Sabre as the routing input of the route stage** (`route_stage/sabre_light.py`,
  `stage1.py --prior sabre-light`). The greedy router was far from Sabre (routed schedule 65% off on test circuits); Qiskit's
  SabreLayout with 1 trial / 1 iteration / 1 routing trial after the level-2 init stage gives SWAPs 1.08-1.20x the full
  Sabre's and a routed schedule within 4-8%, at ~6-10 ms per (circuit, device) (randomcircuit ~40 ms) vs 35-220 ms for
  level 2's Sabre. Launch: `run_local.sh light` then `train28` (RUNBOOK). The self-written Python Sabre (`sabre_lite.py`,
  70 ms, 2.4x SWAPs) is dropped.
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
- [ ] **Priority (2026-10-13): ESP + execution time, routed circuits first.** Routing-free durations are
      solved by the fixed decomposition table (v21t zero-shot 0.5–1.5% median on new families); routed
      circuits on new families are ~15% off, biased low, and a 90% bound calibrated on seen families covers
      only 35% (qaoa) / 53% (variational) — compiler seed floor ~5% (`compile_check/duration_anatomy.py`,
      `esp/esp_count_oracle.py`, `esp/esp_interval.py`). Running: v21w, v21t, v22 (+15k synthetic routing
      circuits, `synthetic_routing/`). Then: bounds calibrated on held-out structures, not on seen families.
- [ ] **Routed circuits, plan agreed 2026-10-13 (next session):** (v19i routed breakdown,
      `esp/results/report_esp_routed_v10_v19i.md`: duration error grows with routing overhead on new
      families — qaoa 4.6% → 11% → 20%, variational up to 44%, bias down to 0.56 — while control stays 3–8%;
      ESP of routed circuits off ×1.7–1.9 on new families and ~30% on control.)
      1. Development benchmark: leave-one-structure-out on `synthetic_routing/` (choose variants there; qaoa /
         variational = final test only).
      2. v23 = v22 + router features in the duration correction (logical schedule, `esp/esp_router.py` SWAP
         schedule, SWAP count, predicted routing overhead).
      3. ESP of routed circuits from the router's physical schedule: idle per physical qubit touched by SWAPs /
         its own min(T1, T2), small learned correction.
      5. Quantile (90%) head + conformal calibrated on held-out synthetic structures; report coverage on LOFO/LOGO.
      6. Next synthetic batch biased to heavy routing (overhead > 0.75) and 11–20 qubits; up-weight them in the loss.
      NOT to do: Qiskit's SabreSwap pass as a feature (user does not agree: it is part of the compiler).
      Results expected from the queue: v21w / v21t / v22 evaluations, intervals (`report_esp_interval_v10_v22.md`),
      routed breakdowns (`report_esp_routed_v10_v21.md`, `_v22.md`).
- [ ] **v24: two-stage routed duration (2026-10-13, `route_stage/`)**. Stage 1 = graph network that learns
      Sabre's routing (SWAPs per operand wire + extra wait per gate, Poisson / Huber), supervised by
      `compile_check/sabre_labels.py`; the duration is the ASAP schedule with those SWAPs; stage 2 = pooled
      log-correction for translation + optimisation. Sabre is a training target only, never an input
      (no compiler at prediction time).
      - Sabre labels, full dataset (`compile_check/results/sabre_labels/report.md`): 32,895 circuits, 0 failed,
        71,222 routed cases. Even with Sabre's true SWAPs the wire schedule is 12–22% off, biased low
        (0.78–0.88); replaying the routed circuit is 1–6% → the rest is translation / optimisation (stage 2).
        **Open:** the final duration matches `esp_labels.pt` within 0.1% in only 58% of cases (median |Δ| 0).
        By family (`claude_scratch/label_mismatch.py`, `route_stage/slurm/label_mismatch_in1988088.log`): exact for
        ae, qft, qftentangled, qnn, vqe_real_amp, vqe_two_local, ghz, wstate, qpe; tiny for vqe_su2 (p90 0.8%);
        large for **qaoa routed (99% differ, p90 16%)**, randomcircuit routed (p90 9%), bmw_quark (p90 32%).
        → v24 trains/evaluates on Sabre's own `final`, the v19i–v22 scorecard on `esp_labels.pt`: on qaoa the two
        targets differ, so compare v24 vs v22 on one target only. Cause to find (different qaoa circuit version
        or compiler settings/seed in sabre_labels.py vs the stored compilations).
      - Stage 1, seed 5, 20 epochs (val loss still falling), routed duration median error [p90] / bias:
        control 0.084 [0.263] / 0.98, qaoa 0.153 [0.329] / 0.85 (without stage 2: 0.127), variational
        0.094 [0.307] / 0.93, LOSO dense+complete 0.201 / 0.80. Routing-free 0.003–0.022.
        SWAP total over-counted ×1.4–2.0 (median ratio) while log-correlation is 0.94–0.99, and yet the
        duration is low → scale/schedule mismatch. Stage 2 hurts on qaoa (0.127 → 0.153).
        Not comparable with v22 (v22 pools 3 seeds; v24 is seed 5 only) — evaluate on the same test first.
      - ESP oracle (`route_stage/results/esp_oracle.md`): decay = k · Σ idle × mean 1/min(T1,T2) of the device
        gives |Δ log ESP| median 1.4 / 1.3 / 0.7 (control / qaoa / variational; log units, i.e. ESP off by ×2–4, NOT %: routed
        decay median −16, all routed cases) vs 2.2–2.6 from n·duration;
        the best-T qubits are worse than the device mean. → with the right schedule, ESP follows.
      - **Next (queued, `route_stage/diag_scale.py`, jobs 1996862–4):** per-slot rate calibration, mass on
        zero-SWAP slots (softplus floor), mean vs median of the totals, duration from Poisson samples vs from
        mean rates (Jensen: schedule of the means < mean of the schedules), attribution SWAPs vs waits.
        Then fix (MC sampling at prediction, or zero-inflated rate / direct duration loss), 40 epochs,
        3 seeds, evaluation on the same test as v22.
- [ ] **v25 (2026-10-15): why routed durations are under-predicted, and the fix** (`route_stage/stage1.py --dur-lambda`).
      Diagnosis from `route_stage/results/report_analysis_v24.md`: NOT the SWAP / CZ count (predicted SWAPs + true waits
      5.2% qaoa / 4.5% LOSO, ≈ the 3.6% / 4.0% ceiling) but the waits: true SWAPs + predicted waits 12% / 18% (bias 0.88 /
      0.84); critical-path waits true 18.7% / 22.9% of the duration, predicted 5.4% / 2.3%; per-gate wait corr 0.12 / 0.00.
      The wait targets are teacher-forced per gate (64% negative on qaoa), so a per-gate Huber averages them to ~0.
      v25 = v24b + Huber on log(schedule / Sabre's replay) through a differentiable level-wise schedule
      (`schedule_torch`, identical to `schedule_np`): SWAPs and waits learn the critical path. Run off-cluster:
      `run_local.sh smoke25` then `JOBS=3 run_local.sh train25` + `report`.
      **v26 (written, to launch together with v25):** v25 + `--contention`: per gate the wait our greedy router pays
      when its SWAPs wait for / block the logical qubits on their chip path (`router_contention`, no compiler), as input
      and as the base of the predicted wait (the network learns the correction). Commands in `route_stage/RUNBOOK.md`.
- [ ] **v27 results (2026-10-16): adopt as the routed-duration stage.** v25 + `synth_layered.py` circuits (dense, irregular,
      low-parallelism). Routed duration, sampled + stage 2, 3 seeds (± std ≤ 0.3 points): qaoa 13.9% -> **8.8%** (p90 30% ->
      20%, bias 0.86 -> 0.92), vs dataset 10.9% -> **6.4%** (Sabre-vs-dataset floor 4.6%); SWAP count bias 0.88 -> 1.03;
      stage 2 now helps qaoa (10.5% -> 8.8%) instead of hurting. control 3.3% -> 3.2%, variational 5.7% -> 5.4%, LOSO
      dense,complete 7.6% -> 7.4%. New split LOSO layered,seqdense (class held out) 11.8%: the gain needs the class in
      training. Next: ESP of routed circuits from the v27 schedule (oracle |Δlog ESP| 0.7-1.4, log units = ×2-4, not %); bias still 0.92 on qaoa.
- [ ] **v25 / v26 results (2026-10-15):** v25 (duration loss) routed duration, sampled + stage 2, 3 seeds: LOSO dense,complete
      13.8% -> 7.6% (bias 0.87 -> 1.05), control 5.3% -> 3.3%, variational 5.4% -> 5.7%, qaoa 13.7% -> 13.9% (bias 0.86).
      v26 (router contention) = v25 everywhere, dropped: our greedy router is far from Sabre (its duration alone 56% off
      on qaoa, 1.56x). qaoa: stage 2 adds ~4 points (10.0% -> 13.9%, wrong sign: final/replay 1.07 on heavy routing), stage 1
      ~7% low. randomcircuit (seen) 13%, bias 0.87: a CLASS (dense, irregular, low parallelism, repeated pairs), not a
      family. **v27 (to launch):** + `synth_layered.py` circuits of that class (route_stage/RUNBOOK.md). Sabre inside the
      predictor: 75-87% of a full compile (qaoa 63 of 72 ms) -> not worth it; only as 'compile once' mode.
- [ ] **qaoa label mismatch explained (2026-10-15, `compile_check/basis_mismatch.py`):** `sabre_labels.py` compiles the
      circuit AFTER `fast_encoding.to_basis` (gates outside the OpenQASM 3 basis are decomposed at level 0: qaoa `rzz` ->
      cx·rz·cx, bmw_quark `rxx`, randomcircuit's exotic gates), the dataset labels compile the ORIGINAL QASM. Same CZ
      count, different 1q optimisation / schedule: qaoa ratio 1.043, |diff| median 4.6%, p90 15.4%, 99% of cases (=
      the reported mismatch); vqe_su2 0.6%; bmw_quark_copula 6.0% / 21.8%. Consequences: (1) the model's input loses
      the information that a gate was `rzz` -> irreducible noise of that size on those families' durations unless the
      encoding gets node types for rzz / rxx (re-encoding + retraining); (2) stage 1 can keep Sabre's SWAPs, but stage 2
      should target the dataset label: `stage1.py --c2-target dataset` (`C2_TARGET=dataset run_local.sh train25/26`).
      Also: `data/circuits.zip` has the OLD qaoa without final measurements; `run_fastpath.py` now adds them.
- [ ] **More devices and topologies**: generalise ESP and duration to more chips (other coupling maps,
      sizes, native gates, gate times); new labels for the dataset and the synthetic circuits on them;
      leave-one-device-out next to LOFO / LOGO.
- [ ] **Test reuse**: ~30 versions were selected on the same splits. Before the paper: freeze the model,
      build a final test never used for any choice (a held-out family or newly generated circuits), evaluate
      the frozen model and 2–3 references on it once; report seed spreads and say when two models are
      indistinguishable (qnn varies ±0.05–0.08 across seeds).
- [ ] **Two-model plan (2026-10-15), steps 1-2 written, to launch** (`bash evaluations/fastpath/launch.sh`):
      routing-free pairs (switch = VF2 subgraph test) -> fast path (`fastpath/`): one reference compilation per circuit
      and topology, embeddings cached, per calibration only VF2Layout's average-error score + MQT F / ESP (no
      compiler; local smoke test: same layout as Qiskit except exact score ties, log F / ESP exact to 1e-15 when
      the layout matches, 0.4-0.7 ms per calibration vs 2-3 ms recompiling); routed pairs -> the model. Outputs:
      `fastpath/results/report_validate.md` (vs recompilation on aug 80-99, 3 seeds, tracking, time) and
      `esp/results/report_hybrid_fast.md` (regret / right device: model, compile-hybrid, fast-hybrid).
      Step 3 after the results: routed model (v24 + ESP from the schedule) trained on routed pairs only.
- [ ] **Hybrid rule: compile what is cheap, predict the rest** (from the pipeline anatomy, 2026-10-11,
      `compile_check/results/pipeline_anatomy_stage/report.md`): routing-free circuits compile in 2.3 ms
      (median, ≈ the VF2 step alone) and their calibration response is exactly the compiler's VF2Layout
      choice → compile them, predict only the routed ones (40 ms median compile). First check that the
      model's routing-free flag matches the compiler's VF2Layout decision; then score with the standard
      evaluation (tracking, regret, decision policies) — no retraining. Later: predict the routed structure
      (calibration-invariant) and compute the placement with the VF2 score.
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
- [ ] **Future work (after the paper): reinforcement-learning placement and routing**, idea of
      2026-10-11. Why: local calibration tracking is limited by the placement, a soft Sinkhorn average,
      while the compiler makes a hard, noise-scored choice among many near-equivalent layouts (worst on
      the larger EQE1 chips and on small circuits). An agent that places qubits and inserts SWAPs step
      by step fits that discrete choice, and adapts to calibration changes by construction.
      - Reward: the exact expected fidelity from usage maps, `log F = −Σ U·ε` (verified to 1e-7,
        `compile_check/merge_usage.py`), so no compiler labels are needed per calibration; train on
        many calibrations generated on the fly.
      - Plan: (1) imitation of the compiler's layouts and usage maps (v8d layouts, v8u maps on 300
        calibrations); (2) RL fine-tuning with the fidelity reward; (3) the agent becomes the placement
        module, the usage-map head (v11) computes F; one rollout per device for device choice.
      - Caveats: it predicts the fidelity *its own* compilation reaches, not Qiskit's (needs imitation
        to stay aligned, or a reframed goal: a noise-aware learned compiler); several steps per circuit
        reduce the 5–13× speed-up; RL on combinatorial problems is unstable. Prior art to check:
        MQT Predictor (RL over compilation flows), IBM's RL-based transpiler passes (routing).
      - Intermediate step inside the current predictor (paper scope): a noise score of each candidate
        placement as prior (readout + CZ errors of the used qubits and couplers, as VF2Layout) and a
        near-argmax placement in training (annealed τ or straight-through).

## Notes for the paper
- Comparison with MQT Predictor: already in the QSW paper (our model is better at
  fidelity estimation and uses the device as input, so it can follow new calibrations).
  Cite it and recall the main numbers; no new experiment needed.
- Present the three devices as a small HPC centre; two of them come from the same chip
  (EQE1 Top/Bottom).
