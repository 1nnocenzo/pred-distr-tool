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
