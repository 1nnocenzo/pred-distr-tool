# Anatomy of the local calibration tracking

Compared: v8kdl40 (3 seeds), v8kdc10 (3 seeds); looked into: v8kdc10.

## 1. Where the model misses (calibration-shift test, local variants)

Tracking = corr(Δlog F predicted, true), F > 0.01, pooled over the cases of each group (so it is higher than the official metric, a mean over (variant, device) cells: compare groups with each other, not with it); mean over seeds. |ΔF| = mean true change of F. Groups with fewer than 60 cases are left out.

| family | cases | mean \|ΔF\| | tracking v8kdl40 | tracking v8kdc10 | error on ΔF v8kdl40 | error on ΔF v8kdc10 |
| --- | --- | --- | --- | --- | --- | --- |
| ae | 245 | 0.0294 | 0.89 | 0.90 | 0.0139 | 0.0144 |
| bmw_quark_cardinality | 68 | 0.0255 | 0.64 | 0.62 | 0.0168 | 0.0169 |
| bv | 72 | 0.0182 | 0.68 | 0.56 | 0.0148 | 0.0163 |
| dj | 60 | 0.0259 | 0.53 | 0.56 | 0.0224 | 0.0225 |
| ghz | 60 | 0.0159 | 0.86 | 0.84 | 0.0102 | 0.0088 |
| graphstate | 72 | 0.0181 | 0.64 | 0.66 | 0.0156 | 0.0147 |
| hhl | 62 | 0.0398 | 0.83 | 0.85 | 0.0246 | 0.0248 |
| qaoa | 252 | 0.0276 | 0.78 | 0.80 | 0.0193 | 0.0203 |
| qft | 243 | 0.0292 | 0.82 | 0.84 | 0.0142 | 0.0138 |
| qftentangled | 158 | 0.0358 | 0.89 | 0.90 | 0.0163 | 0.0161 |
| qnn | 480 | 0.0199 | 0.81 | 0.85 | 0.0121 | 0.0115 |
| randomcircuit | 215 | 0.0376 | 0.69 | 0.68 | 0.0247 | 0.0251 |
| vqe_real_amp | 417 | 0.0231 | 0.84 | 0.84 | 0.0138 | 0.0136 |
| vqe_su2 | 420 | 0.0234 | 0.81 | 0.81 | 0.0133 | 0.0131 |
| vqe_two_local | 436 | 0.0208 | 0.84 | 0.83 | 0.0118 | 0.0116 |
| wstate | 72 | 0.0180 | 0.74 | 0.77 | 0.0115 | 0.0116 |

| qubits | cases | mean \|ΔF\| | tracking v8kdl40 | tracking v8kdc10 | error on ΔF v8kdl40 | error on ΔF v8kdc10 |
| --- | --- | --- | --- | --- | --- | --- |
| 01–05 | 1224 | 0.0275 | 0.66 | 0.68 | 0.0173 | 0.0183 |
| 06–10 | 1204 | 0.0290 | 0.83 | 0.85 | 0.0156 | 0.0147 |
| 11–15 | 603 | 0.0192 | 0.75 | 0.74 | 0.0126 | 0.0121 |
| 16–20 | 478 | 0.0203 | 0.80 | 0.80 | 0.0123 | 0.0121 |

| variant | cases | mean \|ΔF\| | tracking v8kdl40 | tracking v8kdc10 | error on ΔF v8kdl40 | error on ΔF v8kdc10 |
| --- | --- | --- | --- | --- | --- | --- |
| jitterA | 881 | 0.0225 | 0.68 | 0.68 | 0.0152 | 0.0155 |
| jitterB | 881 | 0.0184 | 0.53 | 0.59 | 0.0150 | 0.0148 |
| jitterS | 884 | 0.0171 | 0.73 | 0.76 | 0.0094 | 0.0091 |
| shuffle | 863 | 0.0448 | 0.76 | 0.77 | 0.0215 | 0.0214 |

| device | cases | mean \|ΔF\| | tracking v8kdl40 | tracking v8kdc10 | error on ΔF v8kdl40 | error on ΔF v8kdc10 |
| --- | --- | --- | --- | --- | --- | --- |
| EQE1_Bottom | 1160 | 0.0227 | 0.65 | 0.68 | 0.0159 | 0.0159 |
| EQE1_Top | 1147 | 0.0259 | 0.70 | 0.70 | 0.0171 | 0.0168 |
| QExa20 | 1202 | 0.0281 | 0.89 | 0.89 | 0.0129 | 0.0129 |

| routing | cases | mean \|ΔF\| | tracking v8kdl40 | tracking v8kdc10 | error on ΔF v8kdl40 | error on ΔF v8kdc10 |
| --- | --- | --- | --- | --- | --- | --- |
| heavy routing (> 2×) | 958 | 0.0351 | 0.78 | 0.80 | 0.0194 | 0.0190 |
| light routing (≤ 2×) | 711 | 0.0280 | 0.79 | 0.79 | 0.0177 | 0.0188 |
| no routing (CZ ≈ 2q gates) | 1804 | 0.0196 | 0.83 | 0.83 | 0.0119 | 0.0115 |

## 2. Why: inside the placement (v8kdc10, control, seed 5)

### a. PCA of the physical-qubit embeddings

7373 physical qubits (3 devices × 101 calibrations). Variance explained by the first 6 components: 23%, 12%, 7%, 6%, 6%, 5%.

| component | 1q error | readout error | mean CZ error | T1 | T2 | degree | position (best of 8 Laplacian PE) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PC1 (23%) | -0.18 | -0.15 | -0.72 | -0.34 | -0.34 | +0.50 | 0.17 |
| PC2 (12%) | +0.51 | +0.29 | +0.28 | -0.82 | -0.81 | +0.07 | 0.22 |
| PC3 (7%) | +0.03 | +0.17 | +0.42 | +0.28 | +0.31 | +0.68 | 0.34 |
| PC4 (6%) | -0.59 | -0.54 | +0.29 | +0.01 | -0.07 | +0.05 | 0.16 |
| PC5 (6%) | -0.21 | -0.19 | +0.16 | +0.11 | +0.10 | -0.29 | 0.28 |
| PC6 (5%) | +0.05 | -0.05 | +0.06 | -0.04 | -0.00 | -0.13 | 0.52 |

Linear R² of the embedding (mean over its 128 dimensions, held-out calibrations): from the errors and T1/T2 alone **0.30**, from position and degree alone **0.26**.

### b. Can the error be read back from the embedding?

| probe target (ridge from h_p, held-out calibrations) | R² |
| --- | --- |
| total error of the qubit (1q + readout + mean CZ) | 0.99 |
| its rank within the device (0 = best, 1 = worst) | 0.87 |
| 1q error | 0.99 |
| readout error | 0.99 |
| mean CZ error | 0.97 |

### c. What moves the placement, and what moves the compiler

Share of the change of the model's qubit occupancy when only one kind of error is taken from the variant (300 test circuits × 10 local v8d calibrations):

| error changed alone | share of the occupancy change |
| --- | --- |
| 1q | 8% |
| readout | 38% |
| CZ | 54% |
| T1/T2 | 0% |
| (sum of the single-kind changes / change with all errors) | 1.48 |

Least-squares weight of Δ(occupancy of a physical qubit) on the standardised change of its errors (negative = leaves a qubit whose error grew):

| | Δ 1q error | Δ readout error | Δ mean CZ error |
| --- | --- | --- | --- |
| compiler | -0.009 | -0.024 | -0.016 |
| model | -0.010 | -0.017 | -0.031 |

### d. Errors read at fixed placement vs placement and counts moving (calibration-shift test)

3509 (circuit, device, local variant) cases. Δ predicted = errors-at-fixed-placement term + rest.

| | corr with true Δlog F | mean \|term\| | share of \|Δ predicted\| |
| --- | --- | --- | --- |
| errors at fixed placement | 0.69 | 0.1223 | 142% |
| placement and counts moving | -0.30 | 0.0717 | 83% |
| full prediction | 0.77 | 0.0859 | – |
| true change | – | 0.0987 | – |

Direction right in 78% of the cases (86% when |Δlog F| > 0.03). When it is wrong and the change is large, the fixed-placement term has the true sign in 32% of the cases and the moving term in 43%; the true change is mostly a gain (50% gains).
