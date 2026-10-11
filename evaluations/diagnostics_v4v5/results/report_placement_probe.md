# Why does the placement not follow the qubits?

300 control test circuits × 10 local v8d calibrations (aug [11, 19, 50, 55, 56, 61, 65, 72, 73, 82]) × 3 devices.

## 1–2. Does the placement react, and in the right direction?

| model | qubits moved, model | compiler | occupancy change per logical qubit, model | compiler | corr(Δu model, Δu compiler) | corr(Δu, Δε), model | compiler |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v8k (no fine-tuning) | 31% | 49% | 0.24 | 0.41 | +0.30 | -0.21 | -0.23 |
| v8kd40 | 29% | 49% | 0.22 | 0.41 | +0.33 | -0.24 | -0.23 |
| v11d40 | 31% | 49% | 0.21 | 0.41 | +0.34 | -0.23 | -0.23 |
| v8kdl40 | 28% | 49% | 0.25 | 0.41 | +0.52 | -0.30 | -0.23 |
| v8kdr40 | 30% | 49% | 0.28 | 0.41 | +0.53 | -0.30 | -0.23 |
| v8kdp40 | 28% | 49% | 0.24 | 0.41 | +0.52 | -0.29 | -0.23 |
| v8kdrp40 | 29% | 49% | 0.26 | 0.41 | +0.53 | -0.28 | -0.23 |

Δu = change of occupancy of each physical qubit (model: Σ_q A; compiler: 0/1 initial layout); Δε = change of r + readout + mean CZ error of the qubit. Negative corr(Δu, Δε) = leaves qubits that got worse.

## 3–4. What drives the placement, and how sharp is it?

| model | embedding change when errors are flattened (1 − cos) | mean row max of A | row entropy / log P |
| --- | --- | --- | --- |
| v8k (no fine-tuning) | 0.218 | 0.56 | 0.32 |
| v8kd40 | 0.177 | 0.54 | 0.35 |
| v11d40 | 0.181 | 0.47 | 0.41 |
| v8kdl40 | 0.197 | 0.43 | 0.49 |
| v8kdr40 | 0.202 | 0.43 | 0.50 |
| v8kdp40 | 0.192 | 0.43 | 0.49 |
| v8kdrp40 | 0.201 | 0.43 | 0.49 |

## 4. Local tracking with a lower Sinkhorn temperature at inference (no retraining)

| model | τ = 0.3 | τ = 0.15 | τ = 0.07 | τ = 0.03 |
| --- | --- | --- | --- | --- |
| v8k (no fine-tuning) | 0.37 | 0.22 | 0.10 | 0.05 |
| v8kd40 | 0.64 | 0.55 | 0.39 | 0.28 |
| v11d40 | 0.62 | 0.53 | 0.36 | 0.22 |
| v8kdl40 | 0.70 | 0.64 | 0.47 | 0.30 |
| v8kdr40 | 0.70 | 0.63 | 0.45 | 0.25 |
| v8kdp40 | 0.68 | 0.58 | 0.42 | 0.25 |
| v8kdrp40 | 0.68 | 0.58 | 0.45 | 0.29 |