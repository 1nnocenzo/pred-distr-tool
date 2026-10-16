# Points 1 + 2 vs the previous model: local comparison (2026-10-16)

Sample: 357 circuits (12 per family + 60 synthetic, circuits deeper than 3000 layers left out), original
calibration, control split (285 train, 15 validation, 57 test circuits = 171 cases), 25 epochs on CPU, last epoch,
seeds 5 and 6. Very small: read trends, not digits. ESP errors in nats on cases with ESP > 1e-3 (96 per run, of
which only 24 routed).

| | previous model (s5 / s6) | new: count formula + final-duration loss (s5 / s6) |
| --- | --- | --- |
| duration, routed, median relative error | 11.0% / 11.4% | **5.9% / 5.6%** |
| duration, routed, p90 | 37% / 35% | **25% / 29%** |
| duration, routing-free | 1.5% / 1.9% | 1.2% / 1.7% |
| ESP, all | 0.248 / 0.221 | **0.159 / 0.164** |
| ESP, routing-free (72 cases) | 0.196 / 0.175 | **0.106 / 0.101** |
| ESP, routed (24 cases) | **0.436 / 0.324** | 0.557 / 0.663 |
| abs Δlog F | 0.31 / 0.21 | 0.37 / 0.19 |
| device choice by ESP: right / regret | 0.49 / 0.0152, 0.51 / 0.0129 | **0.68 / 0.0052, 0.58 / 0.0111** |

Routed ESP diagnosis (new model, routed cases): decay biased −0.18 to −0.21 nats (too much decay) because
- the count head predicts 0.37-0.48× the true r and 0.54-0.69× the true cz -> busy time −11%, idle +16-20%;
- extra physical qubits predicted 0.4 vs 0.25 true; device-mean 1/T is ×1.12 the compiler's qubits.
Busy time from the route stage (logical wires + predicted SWAPs) instead of the counts removes the bias (−0.21 ->
0.00) but not the spread (0.34-0.40). With 24 test cases the rest is not readable: judge on the full data.

Also found: model selection on the total validation loss (auxiliaries included, 15 circuits) kept epochs 4-6 of 25;
on the target terms it is still noisy (val F jumps). Default is now the last epoch (`--select best` optional).

## Busy time from the route stage, and the count loss weight (same sample and settings)

| | counts (s5 / s6) | `--esp-busy route` (s5 / s6) | counts, λ_count = 1 (s5) |
| --- | --- | --- | --- |
| duration routed | 5.9% / 5.6% | 6.4% / 6.6% | 7.1% |
| ESP all | 0.159 / 0.164 | 0.198 / **0.133** | 0.186 |
| ESP routing-free | 0.106 / 0.101 | 0.151 / **0.083** | 0.130 |
| ESP routed (24 cases) | 0.557 / 0.663 | **0.350 / 0.420** | 0.618 |
| device choice: right / regret | 0.68 / 0.0052, 0.58 / 0.0111 | 0.61 / 0.0071, 0.65 / 0.0073 | 0.67 / 0.0092 |

Why the counts are poor (diagnosis on the TRAINING circuits of the counts model):
- not a hard target: the decomposition table alone gives r 0.97-1.00 and cz 1.00 (routing-free) / 0.62 (routed) of the
  true counts, cz 1.08 with 3 CZ per true SWAP; the learned head gives r 0.46, cz 0.58 on routed circuits;
- the count head starts from 1 r / 1 cz per gate, not from the table, and its routed cz uses a circuit-level greedy
  hop estimate, not the route stage's SWAPs;
- weak signal: 225 optimiser steps in total (count loss still falling); the F loss (MSE on F + 0.1 MSE on log F)
  hardly sees routed circuits (small F): their F is overestimated x3.5 on training circuits (log F +1.26);
- not the scales s_k (stay at 1.0) and not the placement's errors (ratio 1.00-1.04 to the compiler's qubits).

## Structured counts (table x correction + route-stage SWAPs) and the F-loss fix

| | `route`, old count head (s5 / s6) | structured, F loss clipped (`route` s5 / s6, `counts` s5) | structured + exact log F loss (`route` s5 / s6, `counts` s5) |
| --- | --- | --- | --- |
| counts on routed test circuits, r / cz (pred/true) | 0.38 / 0.54 (counts model) | 0.75-0.81 / 0.85-0.89 | **0.84-0.85 / 0.98-1.01** |
| log F error on routed training circuits (median) | +1.26 (counts model) | +1.18 | **-0.04 / -0.10** |
| abs Δlog F, all test cases | 0.38 / 0.19 | 0.22 / 0.21 / 0.20 | **0.19 / 0.19 / 0.18** |
| duration routed (median / p90) | 6.4% / 26% | 5-7% / 25-27% | 6.1-6.8% / **20-24%** |
| ESP all | 0.198 / 0.133 | 0.248 / 0.224 / 0.198 | 0.229 / 0.217 / 0.190 |
| ESP routed (24 cases): decay part | 0.42 / 0.49 | | **0.32 / 0.43 / 0.28** |
| device choice by ESP: regret | 0.0071 / 0.0073 | 0.018-0.021 | 0.0195-0.0196 |

The clipped F target (log max(F, 1e-3)) had pushed the count correction to 0.5x of the table on routed circuits;
with the exact log F the correction settles at 0.7-0.8 and F on routed circuits is right. The device-choice regret
is identical across the three runs (0.0167 by F): a few of the 57 test circuits dominate it, not readable here.
Default kept: `--esp-busy counts` (best decay error, one seed); compare both on the full data.
