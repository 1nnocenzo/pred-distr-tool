# v12 design: a network on the placed circuit (idea, 2026-10-11)

Status: proposal, not implemented. Goal: raise local calibration tracking (0.63 per seed, 0.67 as a
3-seed ensemble for v8kdl40) by predicting *where the compiled circuit spends its gates on the chip*
after the placement, with a message-passing network on the device graph.

## Why here

What the studies so far rule out and point to:

| evidence | source | consequence |
| --- | --- | --- |
| true compiler layout does not help (old head 0.59 → 0.57; v11 0.64 → 0.64) | `report_tracking_oracle*.md` | the placement is not the bottleneck |
| v11 with true layout and true counts 0.74; with true usage maps 1.00 | `report_tracking_oracle_usage.md` | the gap is how gates spread over qubits and couplers after the placement (routing) |
| placement ablations (grid, lr ×10, per-kind prior, MLP scorer, VF2 prior) all 0.60–0.63 | anatomy reports | architecture of the placement exhausted |
| sharper placement at inference is worse (0.61 → 0.44 hard) | `report_placement_probe.md` §4 | keep the soft placement |
| staying on the old compilation tracks at 0.57 (model 0.69, same cases); the compiler recovers most of the damage | `report_tracking_warm.md` | tracking = predicting the compiler's re-optimisation |
| 12 → 100 calibrations per device: 0.49 → 0.63, saturating | learning curve (`v8kc*`) | more calibrations of the same 3 chips give little |

Today, after the placement, native counts come from an MLP per gate (mean context of its qubits,
operand distance) and, in v11, CZs are spread over couplers by a fixed rule (shortest paths, softmax
of path error). Neither sees the placed circuit as a whole on the chip; routing (paths, congestion,
avoiding bad couplers) is a global problem on the device graph.

## Architecture (kind `sinkhorn_rf_route_usage_pgnn`, built on v11)

Per (circuit, device), after the Sinkhorn placement A (Qm × P):

**Placed-circuit graph** on the physical qubits:
- node p: occupancy Σ_q A[q,p]; expected 1q, measure and 2q load (logical per-qubit counts, CZ via the
  constrained base count, mapped through A); device embedding h_p (already carries the errors);
  raw ε_r, ε_m, mean ε_cz.
- coupling edges (p, r): coupler error ε_cz(e); direct CZ demand A^T W A (p, r), W = logical interaction
  counts.
- demand edges for non-adjacent pairs with demand > 0: demand, hop distance. Messages then flow
  along both the chip and the circuit's needs.

**Network:** 4 layers of edge-conditioned message passing (GINE-like, hidden 64), edge types embedded.

**Outputs (anchored, zero-initialised corrections, so it starts as v11):**
- Û_r(p) = U_r^prior(p) · exp(δ_r(p)), Û_m(p) likewise (prior = counts mapped through A);
- Û_cz(e) = softplus(log U_cz^prior(e) + δ_cz(e)), prior = v11 route kernel applied to the demand;
- log F = −(s_r Σ Û_r ε_r + s_m Σ Û_m ε_m + s_cz Σ Û_cz ε_cz): exact for true maps (checked to 1e-7).

Cost: P ≤ 27, E ≤ 39, tiny next to the circuit encoder; inference time to measure (target: stays well
below compilation, see `timing_v8/timing_v11.py`).

## Supervision

- Usage maps of the best compilation (`compile_check/results/v8u/usage_labels.pt`): 2040 circuits × 3
  devices × (original + 100 calibration variants) ≈ 6·10⁵ maps. Skewed to vqe_* / qaoa / randomcircuit;
  dynamic circuits excluded.
- Losses: F (as now), change loss on variants (+ correlation term if `--delta-corr-lambda` helps),
  map loss on original and variant maps (log1p MSE per qubit / coupler + CZ-map correlation).

## Plan

0. **Go / no-go, ~½ day.** Train the post-placement network alone with the compiler's true layout as
   A (teacher forcing) to predict the variant maps; calibrations aug 0–79 for training, aug 80–99 and the
   calibration-shift test for evaluation; held-out circuits = control test split. Measure tracking from
   its maps. Go only if it clearly beats the 0.74 "true layout + true counts" oracle on unseen
   calibrations (it sees the same inputs plus the routing context). Also report seen vs unseen
   calibrations, to catch memorisation of the 3 chips (the 0.73 vs 0.59 gap of the current model).
1. **Integrate, ~1 day.** Learned placement + network, fine-tuned from v11dl40 with the v8kdl40 recipe.
2. **Standard evaluation** (CLAUDE.md 1–6) against v8kdl40, then 3 seeds if it wins.

## Risks

- Memorisation of the 3 chips' calibrations (phase 0 measures it directly).
- Map labels only for 2040 pilot circuits, few for small families: the map loss may bias toward vqe/qaoa.
- The learned placement feeding it is imperfect; the oracles say this matters little (0.64 → 0.64).
- If phase 0 fails, the remaining structural lever is device variety (many synthetic chips), which
  costs weeks of compilation.
