# Anatomy of the local calibration tracking

Compared: v8kdc10 (3 seeds), v8kde10 (3 seeds), v8kda10 (3 seeds); looked into: v8kda10.

## 1. Where the model misses (calibration-shift test, local variants)

Tracking = corr(Δlog F predicted, true), F > 0.01, pooled over the cases of each group (so it is higher than the official metric, a mean over (variant, device) cells: compare groups with each other, not with it); mean over seeds. |ΔF| = mean true change of F. Groups with fewer than 60 cases are left out.

| family | cases | mean \|ΔF\| | tracking v8kdc10 | tracking v8kde10 | tracking v8kda10 | error on ΔF v8kdc10 | error on ΔF v8kde10 | error on ΔF v8kda10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ae | 245 | 0.0294 | 0.90 | 0.92 | 0.91 | 0.0144 | 0.0145 | 0.0142 |
| bmw_quark_cardinality | 68 | 0.0255 | 0.62 | 0.59 | 0.68 | 0.0169 | 0.0159 | 0.0162 |
| bv | 72 | 0.0182 | 0.56 | 0.71 | 0.73 | 0.0163 | 0.0133 | 0.0125 |
| dj | 60 | 0.0259 | 0.56 | 0.61 | 0.51 | 0.0225 | 0.0221 | 0.0223 |
| ghz | 60 | 0.0159 | 0.84 | 0.87 | 0.69 | 0.0088 | 0.0082 | 0.0090 |
| graphstate | 72 | 0.0181 | 0.66 | 0.71 | 0.58 | 0.0147 | 0.0136 | 0.0151 |
| hhl | 62 | 0.0398 | 0.85 | 0.87 | 0.85 | 0.0248 | 0.0256 | 0.0240 |
| qaoa | 252 | 0.0276 | 0.80 | 0.82 | 0.80 | 0.0203 | 0.0199 | 0.0205 |
| qft | 243 | 0.0292 | 0.84 | 0.87 | 0.86 | 0.0138 | 0.0135 | 0.0142 |
| qftentangled | 158 | 0.0358 | 0.90 | 0.92 | 0.91 | 0.0161 | 0.0150 | 0.0158 |
| qnn | 480 | 0.0199 | 0.85 | 0.80 | 0.81 | 0.0115 | 0.0116 | 0.0118 |
| randomcircuit | 215 | 0.0376 | 0.68 | 0.72 | 0.69 | 0.0251 | 0.0244 | 0.0252 |
| vqe_real_amp | 417 | 0.0231 | 0.84 | 0.83 | 0.83 | 0.0136 | 0.0137 | 0.0137 |
| vqe_su2 | 420 | 0.0234 | 0.81 | 0.82 | 0.81 | 0.0131 | 0.0129 | 0.0129 |
| vqe_two_local | 436 | 0.0208 | 0.83 | 0.82 | 0.82 | 0.0116 | 0.0114 | 0.0114 |
| wstate | 72 | 0.0180 | 0.77 | 0.73 | 0.71 | 0.0116 | 0.0107 | 0.0116 |

| qubits | cases | mean \|ΔF\| | tracking v8kdc10 | tracking v8kde10 | tracking v8kda10 | error on ΔF v8kdc10 | error on ΔF v8kde10 | error on ΔF v8kda10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01–05 | 1224 | 0.0275 | 0.68 | 0.71 | 0.67 | 0.0183 | 0.0181 | 0.0181 |
| 06–10 | 1204 | 0.0290 | 0.85 | 0.87 | 0.86 | 0.0147 | 0.0143 | 0.0148 |
| 11–15 | 603 | 0.0192 | 0.74 | 0.71 | 0.71 | 0.0121 | 0.0117 | 0.0116 |
| 16–20 | 478 | 0.0203 | 0.80 | 0.80 | 0.79 | 0.0121 | 0.0120 | 0.0123 |

| variant | cases | mean \|ΔF\| | tracking v8kdc10 | tracking v8kde10 | tracking v8kda10 | error on ΔF v8kdc10 | error on ΔF v8kde10 | error on ΔF v8kda10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jitterA | 881 | 0.0225 | 0.68 | 0.69 | 0.67 | 0.0155 | 0.0152 | 0.0152 |
| jitterB | 881 | 0.0184 | 0.59 | 0.60 | 0.60 | 0.0148 | 0.0147 | 0.0152 |
| jitterS | 884 | 0.0171 | 0.76 | 0.77 | 0.77 | 0.0091 | 0.0092 | 0.0090 |
| shuffle | 863 | 0.0448 | 0.77 | 0.79 | 0.77 | 0.0214 | 0.0204 | 0.0210 |

| device | cases | mean \|ΔF\| | tracking v8kdc10 | tracking v8kde10 | tracking v8kda10 | error on ΔF v8kdc10 | error on ΔF v8kde10 | error on ΔF v8kda10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EQE1_Bottom | 1160 | 0.0227 | 0.68 | 0.73 | 0.71 | 0.0159 | 0.0158 | 0.0154 |
| EQE1_Top | 1147 | 0.0259 | 0.70 | 0.73 | 0.72 | 0.0168 | 0.0162 | 0.0166 |
| QExa20 | 1202 | 0.0281 | 0.89 | 0.89 | 0.87 | 0.0129 | 0.0128 | 0.0133 |

| routing | cases | mean \|ΔF\| | tracking v8kdc10 | tracking v8kde10 | tracking v8kda10 | error on ΔF v8kdc10 | error on ΔF v8kde10 | error on ΔF v8kda10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heavy routing (> 2×) | 958 | 0.0351 | 0.80 | 0.83 | 0.81 | 0.0190 | 0.0183 | 0.0188 |
| light routing (≤ 2×) | 711 | 0.0280 | 0.79 | 0.81 | 0.78 | 0.0188 | 0.0182 | 0.0184 |
| no routing (CZ ≈ 2q gates) | 1804 | 0.0196 | 0.83 | 0.81 | 0.83 | 0.0115 | 0.0116 | 0.0116 |

## 2. Why: inside the placement (v8kda10, control, seed 5)

### a. PCA of the physical-qubit embeddings

7373 physical qubits (3 devices × 101 calibrations). Variance explained by the first 6 components: 25%, 12%, 7%, 6%, 6%, 5%.

| component | 1q error | readout error | mean CZ error | T1 | T2 | degree | position (best of 8 Laplacian PE) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PC1 (25%) | -0.21 | -0.17 | -0.72 | -0.31 | -0.32 | +0.47 | 0.17 |
| PC2 (12%) | +0.48 | +0.25 | +0.32 | -0.80 | -0.78 | +0.12 | 0.20 |
| PC3 (7%) | -0.13 | +0.05 | +0.44 | +0.34 | +0.37 | +0.66 | 0.31 |
| PC4 (6%) | -0.54 | -0.46 | +0.18 | -0.05 | -0.13 | +0.02 | 0.24 |
| PC5 (6%) | +0.42 | +0.26 | -0.18 | -0.17 | -0.15 | +0.37 | 0.40 |
| PC6 (5%) | -0.19 | +0.11 | +0.02 | +0.07 | +0.02 | +0.10 | 0.42 |

Linear R² of the embedding (mean over its 128 dimensions, held-out calibrations): from the errors and T1/T2 alone **0.30**, from position and degree alone **0.26**.

### b. Can the error be read back from the embedding?

| probe target (ridge from h_p, held-out calibrations) | R² |
| --- | --- |
| total error of the qubit (1q + readout + mean CZ) | 0.99 |
| its rank within the device (0 = best, 1 = worst) | 0.87 |
| 1q error | 1.00 |
| readout error | 0.99 |
| mean CZ error | 0.96 |

### c. What moves the placement, and what moves the compiler

Share of the change of the model's qubit occupancy when only one kind of error is taken from the variant (300 test circuits × 10 local v8d calibrations):

| error changed alone | share of the occupancy change |
| --- | --- |
| 1q | 8% |
| readout | 38% |
| CZ | 54% |
| T1/T2 | 0% |
| (sum of the single-kind changes / change with all errors) | 1.52 |

Least-squares weight of Δ(occupancy of a physical qubit) on the standardised change of its errors (negative = leaves a qubit whose error grew):

| | Δ 1q error | Δ readout error | Δ mean CZ error |
| --- | --- | --- | --- |
| compiler | -0.009 | -0.024 | -0.016 |
| model | -0.010 | -0.018 | -0.029 |

### d. Errors read at fixed placement vs placement and counts moving (calibration-shift test)

3509 (circuit, device, local variant) cases. Δ predicted = errors-at-fixed-placement term + rest.

| | corr with true Δlog F | mean \|term\| | share of \|Δ predicted\| |
| --- | --- | --- | --- |
| errors at fixed placement | 0.66 | 0.1146 | 142% |
| placement and counts moving | -0.22 | 0.0671 | 83% |
| full prediction | 0.76 | 0.0806 | – |
| true change | – | 0.0987 | – |

Direction right in 79% of the cases (87% when |Δlog F| > 0.03). When it is wrong and the change is large, the fixed-placement term has the true sign in 31% of the cases and the moving term in 44%; the true change is mostly a gain (57% gains).
