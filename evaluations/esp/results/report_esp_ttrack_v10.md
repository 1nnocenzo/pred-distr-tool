# Do the ESP heads follow T1 / T2 and gate-duration changes? _v10

457 calibration-test circuits × 3 devices; per kind 2 variants per device; control-split models, mean over seeds 5–7. Δ = variant − original. Only T1 / T2 or durations change, so the true Δlog F is 0.

## v8kdi10 (v16)

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.84 | 0.70 | 0.162 | – | 0.00 | 0.000 | 0.0072 | 0.0041 | 0.0040 |
| tbad | 0.73 | 0.65 | 0.856 | – | 0.00 | 0.000 | 0.0116 | 0.0065 | 0.0063 |
| djit | -0.01 | -0.00 | 1.693 | -0.01 | 0.00 | 0.171 | 0.0000 | 0.0068 | 0.0068 |
| czslow | 0.01 | 0.00 | 0.081 | -0.01 | 0.00 | 0.014 | 0.0000 | 0.0032 | 0.0032 |
| mslow | 0.00 | 0.00 | 0.000 | -0.01 | 0.00 | 0.335 | 0.0000 | 0.0039 | 0.0039 |
| mix | 0.37 | 0.08 | 1.698 | 0.00 | 0.00 | 0.152 | 0.0057 | 0.0143 | 0.0145 |

## v8kdg10 (v17)

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.86 | 0.76 | 0.162 | – | 0.00 | 0.000 | 0.0088 | 0.0037 | 0.0034 |
| tbad | 0.75 | 0.73 | 0.856 | – | 0.00 | 0.000 | 0.0143 | 0.0060 | 0.0052 |
| djit | 0.02 | 0.00 | 1.693 | – | 0.00 | 0.171 | 0.0000 | 0.0054 | 0.0054 |
| czslow | 0.01 | 0.00 | 0.081 | – | 0.00 | 0.014 | 0.0000 | 0.0028 | 0.0028 |
| mslow | 0.02 | 0.00 | 0.000 | – | 0.00 | 0.335 | 0.0000 | 0.0036 | 0.0036 |
| mix | 0.40 | 0.09 | 1.698 | 0.01 | 0.00 | 0.152 | 0.0073 | 0.0138 | 0.0137 |

Kinds: tjit = T1/T2 × lognormal(0.3) on every qubit; tbad = 2–3 qubits with T × 0.15–0.3; djit = every gate duration × lognormal(0.2); czslow = 3–5 couplers with the CZ 2–3× slower; mslow = readout 1.3–2× slower; mix = tjit + djit. A tracking of – means the model's prediction does not move at all (no input carries that change).
