# Anatomy of the local calibration tracking

## 1. Where the model misses (calibration-shift test, local variants)

Tracking = corr(Δlog F predicted, true), F > 0.01, pooled per group; mean over 3 seeds. |ΔF| = mean true change of F. Groups with fewer than 60 cases are left out.

| family | cases | mean \|ΔF\| | tracking v8kd40 | tracking v8kdl40 | error on ΔF v8kd40 | error on ΔF v8kdl40 |
| --- | --- | --- | --- | ------ | --- |
| ae | 245 | 0.0294 | 0.86 | 0.89 | 0.0169 | 0.0139 |
| bmw_quark_cardinality | 68 | 0.0255 | 0.70 | 0.64 | 0.0166 | 0.0168 |
| bv | 72 | 0.0182 | 0.72 | 0.68 | 0.0148 | 0.0148 |
| dj | 60 | 0.0259 | 0.55 | 0.53 | 0.0219 | 0.0224 |
| ghz | 60 | 0.0159 | 0.88 | 0.86 | 0.0099 | 0.0102 |
| graphstate | 72 | 0.0181 | 0.68 | 0.64 | 0.0161 | 0.0156 |
| hhl | 62 | 0.0398 | 0.82 | 0.83 | 0.0280 | 0.0246 |
| qaoa | 252 | 0.0276 | 0.72 | 0.78 | 0.0244 | 0.0193 |
| qft | 243 | 0.0292 | 0.77 | 0.82 | 0.0180 | 0.0142 |
| qftentangled | 158 | 0.0358 | 0.92 | 0.89 | 0.0174 | 0.0163 |
| qnn | 480 | 0.0199 | 0.82 | 0.81 | 0.0123 | 0.0121 |
| randomcircuit | 215 | 0.0376 | 0.63 | 0.69 | 0.0274 | 0.0247 |
| vqe_real_amp | 417 | 0.0231 | 0.81 | 0.84 | 0.0159 | 0.0138 |
| vqe_su2 | 420 | 0.0234 | 0.81 | 0.81 | 0.0152 | 0.0133 |
| vqe_two_local | 436 | 0.0208 | 0.82 | 0.84 | 0.0131 | 0.0118 |
| wstate | 72 | 0.0180 | 0.81 | 0.74 | 0.0115 | 0.0115 |

| qubits | cases | mean \|ΔF\| | tracking v8kd40 | tracking v8kdl40 | error on ΔF v8kd40 | error on ΔF v8kdl40 |
| --- | --- | --- | --- | ------ | --- |
| 01–05 | 1224 | 0.0275 | 0.58 | 0.66 | 0.0222 | 0.0173 |
| 06–10 | 1204 | 0.0290 | 0.81 | 0.83 | 0.0163 | 0.0156 |
| 11–15 | 603 | 0.0192 | 0.72 | 0.75 | 0.0128 | 0.0126 |
| 16–20 | 478 | 0.0203 | 0.80 | 0.80 | 0.0118 | 0.0123 |

| variant | cases | mean \|ΔF\| | tracking v8kd40 | tracking v8kdl40 | error on ΔF v8kd40 | error on ΔF v8kdl40 |
| --- | --- | --- | --- | ------ | --- |
| jitterA | 881 | 0.0225 | 0.61 | 0.68 | 0.0176 | 0.0152 |
| jitterB | 881 | 0.0184 | 0.52 | 0.53 | 0.0166 | 0.0150 |
| jitterS | 884 | 0.0171 | 0.71 | 0.73 | 0.0103 | 0.0094 |
| shuffle | 863 | 0.0448 | 0.72 | 0.76 | 0.0243 | 0.0215 |

| device | cases | mean \|ΔF\| | tracking v8kd40 | tracking v8kdl40 | error on ΔF v8kd40 | error on ΔF v8kdl40 |
| --- | --- | --- | --- | ------ | --- |
| EQE1_Bottom | 1160 | 0.0227 | 0.66 | 0.65 | 0.0170 | 0.0159 |
| EQE1_Top | 1147 | 0.0259 | 0.63 | 0.70 | 0.0200 | 0.0171 |
| QExa20 | 1202 | 0.0281 | 0.87 | 0.89 | 0.0146 | 0.0129 |

| routing | cases | mean \|ΔF\| | tracking v8kd40 | tracking v8kdl40 | error on ΔF v8kd40 | error on ΔF v8kdl40 |
| --- | --- | --- | --- | ------ | --- |
| heavy routing (> 2×) | 958 | 0.0351 | 0.75 | 0.78 | 0.0216 | 0.0194 |
| light routing (≤ 2×) | 711 | 0.0280 | 0.77 | 0.79 | 0.0214 | 0.0177 |
| no routing (CZ ≈ 2q gates) | 1804 | 0.0196 | 0.81 | 0.83 | 0.0129 | 0.0119 |

## 2. Why: inside the placement (v8kdl40, control, seed 5)

### a. PCA of the physical-qubit embeddings

7373 physical qubits (3 devices × 101 calibrations). Variance explained by the first 6 components: 23%, 12%, 7%, 6%, 5%, 5%.

| component | 1q error | readout error | mean CZ error | T1 | T2 | degree | position (best of 8 Laplacian PE) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PC1 (23%) | -0.16 | -0.14 | -0.69 | -0.38 | -0.39 | +0.50 | 0.16 |
| PC2 (12%) | -0.48 | -0.28 | -0.33 | +0.80 | +0.80 | +0.00 | 0.19 |
| PC3 (7%) | -0.08 | -0.14 | -0.41 | -0.19 | -0.23 | -0.70 | 0.39 |
| PC4 (6%) | +0.25 | +0.27 | +0.05 | +0.12 | +0.17 | -0.03 | 0.25 |
| PC5 (5%) | -0.52 | -0.53 | +0.34 | +0.13 | +0.09 | -0.17 | 0.30 |
| PC6 (5%) | +0.27 | +0.13 | -0.10 | -0.05 | -0.00 | -0.19 | 0.45 |

Linear R² of the embedding (mean over its 128 dimensions, held-out calibrations): from the errors and T1/T2 alone **0.29**, from position and degree alone **0.25**.

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
| 1q | 7% |
| readout | 39% |
| CZ | 54% |
| T1/T2 | 0% |
| (sum of the single-kind changes / change with all errors) | 1.54 |

Least-squares weight of Δ(occupancy of a physical qubit) on the standardised change of its errors (negative = leaves a qubit whose error grew):

| | Δ 1q error | Δ readout error | Δ mean CZ error |
| --- | --- | --- | --- |
| compiler | -0.009 | -0.024 | -0.016 |
| model | -0.009 | -0.019 | -0.028 |

### d. Errors read at fixed placement vs placement and counts moving (calibration-shift test)

3509 (circuit, device, local variant) cases. Δ predicted = errors-at-fixed-placement term + rest.

| | corr with true Δlog F | mean \|term\| | share of \|Δ predicted\| |
| --- | --- | --- | --- |
| errors at fixed placement | 0.68 | 0.1225 | 148% |
| placement and counts moving | -0.28 | 0.0716 | 86% |
| full prediction | 0.76 | 0.0830 | – |
| true change | – | 0.0987 | – |

Direction right in 79% of the cases (86% when |Δlog F| > 0.03). When it is wrong and the change is large, the fixed-placement term has the true sign in 36% of the cases and the moving term in 43%; the true change is mostly a gain (51% gains).
