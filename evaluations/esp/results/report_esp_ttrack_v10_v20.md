# Do the ESP heads follow T1 / T2 and gate-duration changes? _v10_v20

457 calibration-test circuits × 3 devices; per kind 2 variants per device; control-split models, mean over seeds 5–7. Δ = variant − original. Only T1 / T2 or durations change, so the true Δlog F is 0.

## v8kdi10 (v16), fixed gate times

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.83 | 0.69 | 0.162 | – | 0.00 | 0.000 | 0.0072 | 0.0050 | 0.0052 |
| tbad | 0.73 | 0.65 | 0.856 | – | 0.00 | 0.000 | 0.0116 | 0.0070 | 0.0071 |
| djit | -0.04 | -0.00 | 1.693 | 0.02 | -0.00 | 0.171 | 0.0000 | 0.0079 | 0.0079 |
| czslow | -0.01 | -0.00 | 0.081 | -0.01 | -0.00 | 0.014 | 0.0000 | 0.0045 | 0.0045 |
| mslow | -0.03 | -0.00 | 0.000 | 0.02 | -0.00 | 0.335 | 0.0000 | 0.0053 | 0.0053 |
| mix | 0.36 | 0.07 | 1.698 | 0.01 | -0.00 | 0.152 | 0.0057 | 0.0147 | 0.0147 |

## v8kdi10 (v16) + per-qubit gate times, no training

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.83 | 0.69 | 0.162 | – | 0.00 | 0.000 | 0.0072 | 0.0050 | 0.0052 |
| tbad | 0.73 | 0.65 | 0.856 | – | 0.00 | 0.000 | 0.0116 | 0.0070 | 0.0071 |
| djit | 0.59 | 0.82 | 1.693 | 0.55 | 0.56 | 0.171 | 0.0000 | 0.0079 | 0.0071 |
| czslow | 0.29 | 0.15 | 0.081 | 0.61 | 0.56 | 0.014 | 0.0000 | 0.0045 | 0.0045 |
| mslow | 0.85 | 1.96 | 0.000 | 0.99 | 1.00 | 0.335 | 0.0000 | 0.0053 | 0.0063 |
| mix | 0.76 | 0.69 | 1.698 | 0.66 | 0.65 | 0.152 | 0.0057 | 0.0147 | 0.0085 |

## v8kdg10 (v17), fixed gate times

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.86 | 0.76 | 0.162 | – | 0.00 | 0.000 | 0.0088 | 0.0038 | 0.0037 |
| tbad | 0.75 | 0.72 | 0.856 | – | 0.00 | 0.000 | 0.0143 | 0.0060 | 0.0053 |
| djit | 0.01 | 0.00 | 1.693 | – | 0.00 | 0.171 | 0.0000 | 0.0054 | 0.0054 |
| czslow | 0.01 | 0.00 | 0.081 | – | 0.00 | 0.014 | 0.0000 | 0.0031 | 0.0031 |
| mslow | 0.03 | 0.00 | 0.000 | – | 0.00 | 0.335 | 0.0000 | 0.0039 | 0.0039 |
| mix | 0.40 | 0.09 | 1.698 | 0.01 | 0.00 | 0.152 | 0.0073 | 0.0139 | 0.0139 |

## v8kdg10 (v17) + per-qubit gate times, no training (v18 zero-shot)

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.86 | 0.76 | 0.162 | – | 0.00 | 0.000 | 0.0088 | 0.0038 | 0.0037 |
| tbad | 0.75 | 0.72 | 0.856 | – | 0.00 | 0.000 | 0.0143 | 0.0060 | 0.0053 |
| djit | 0.56 | 0.35 | 1.693 | 0.64 | 0.37 | 0.171 | 0.0000 | 0.0054 | 0.0072 |
| czslow | 0.41 | 0.17 | 0.081 | 0.58 | 0.31 | 0.014 | 0.0000 | 0.0031 | 0.0034 |
| mslow | 0.85 | 1.41 | 0.000 | 0.98 | 0.97 | 0.335 | 0.0000 | 0.0039 | 0.0045 |
| mix | 0.73 | 0.39 | 1.698 | 0.71 | 0.41 | 0.152 | 0.0073 | 0.0139 | 0.0087 |

## v8kdt10 (v18): v17 fine-tuned on T1 / T2 / gate-time variants

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.66 | 0.90 | 0.162 | – | 0.00 | 0.000 | 0.0163 | 0.0060 | 0.0053 |
| tbad | 0.70 | 0.81 | 0.856 | – | 0.00 | 0.000 | 0.0208 | 0.0071 | 0.0117 |
| djit | 0.55 | 0.48 | 1.693 | 0.64 | 0.55 | 0.171 | 0.0000 | 0.0077 | 0.0098 |
| czslow | 0.55 | 0.38 | 0.081 | 0.64 | 0.62 | 0.014 | 0.0000 | 0.0059 | 0.0060 |
| mslow | 0.70 | 2.02 | 0.000 | 0.98 | 1.10 | 0.335 | 0.0000 | 0.0065 | 0.0087 |
| mix | 0.79 | 0.71 | 1.698 | 0.69 | 0.76 | 0.152 | 0.0125 | 0.0149 | 0.0084 |

## v8kdt10b (v18b): as v18, change loss × 0.1, F invariance × 3

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.87 | 0.78 | 0.162 | – | 0.00 | 0.000 | 0.0046 | 0.0042 | 0.0040 |
| tbad | 0.73 | 0.72 | 0.856 | – | 0.00 | 0.000 | 0.0076 | 0.0059 | 0.0052 |
| djit | 0.57 | 0.55 | 1.693 | 0.67 | 0.62 | 0.171 | 0.0000 | 0.0057 | 0.0092 |
| czslow | 0.50 | 0.37 | 0.081 | 0.60 | 0.55 | 0.014 | 0.0000 | 0.0037 | 0.0038 |
| mslow | 0.77 | 2.40 | 0.000 | 0.98 | 1.10 | 0.335 | 0.0000 | 0.0045 | 0.0066 |
| mix | 0.79 | 0.62 | 1.698 | 0.70 | 0.68 | 0.152 | 0.0042 | 0.0143 | 0.0080 |

## v8kdg10 (v17) weights in the formula head, hard placement, coupler CZ times (v19 zero-shot)

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.79 | 0.77 | 0.162 | – | 0.00 | 0.000 | 0.0088 | 0.0038 | 0.0034 |
| tbad | 0.73 | 0.70 | 0.856 | – | 0.00 | 0.000 | 0.0143 | 0.0055 | 0.0049 |
| djit | 0.67 | 0.99 | 1.693 | 0.84 | 0.92 | 0.171 | 0.0000 | 0.0049 | 0.0029 |
| czslow | 0.47 | 0.39 | 0.081 | 0.60 | 0.56 | 0.014 | 0.0000 | 0.0033 | 0.0032 |
| mslow | 0.86 | 1.64 | 0.000 | 0.99 | 1.05 | 0.335 | 0.0000 | 0.0040 | 0.0041 |
| mix | 0.80 | 0.95 | 1.698 | 0.82 | 0.89 | 0.152 | 0.0073 | 0.0141 | 0.0028 |

## v8kdg10 (v17) weights in the formula head, compiler's true layout (oracle)

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.91 | 0.85 | 0.162 | – | 0.00 | 0.000 | 0.0088 | 0.0032 | 0.0027 |
| tbad | 0.78 | 0.73 | 0.856 | – | 0.00 | 0.000 | 0.0143 | 0.0047 | 0.0021 |
| djit | 0.70 | 1.01 | 1.693 | 0.89 | 0.94 | 0.171 | 0.0000 | 0.0044 | 0.0016 |
| czslow | 0.49 | 0.38 | 0.081 | 0.66 | 0.56 | 0.014 | 0.0000 | 0.0026 | 0.0025 |
| mslow | 0.87 | 1.67 | 0.000 | 0.99 | 1.05 | 0.335 | 0.0000 | 0.0034 | 0.0034 |
| mix | 0.84 | 0.91 | 1.698 | 0.87 | 0.89 | 0.152 | 0.0073 | 0.0136 | 0.0018 |

## v8kdf10 (v19): formula head trained, no T / gate-time variants in training

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.79 | 0.75 | 0.162 | – | 0.00 | 0.000 | 0.0085 | 0.0042 | 0.0036 |
| tbad | 0.73 | 0.70 | 0.856 | – | 0.00 | 0.000 | 0.0130 | 0.0062 | 0.0050 |
| djit | 0.68 | 0.99 | 1.693 | 0.85 | 0.93 | 0.171 | 0.0000 | 0.0054 | 0.0026 |
| czslow | 0.45 | 0.39 | 0.081 | 0.59 | 0.55 | 0.014 | 0.0000 | 0.0038 | 0.0033 |
| mslow | 0.86 | 1.69 | 0.000 | 0.99 | 1.05 | 0.335 | 0.0000 | 0.0043 | 0.0043 |
| mix | 0.81 | 0.93 | 1.698 | 0.83 | 0.89 | 0.152 | 0.0072 | 0.0143 | 0.0024 |

## v8kdf10 (v19), compiler's true layout (oracle)

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.92 | 0.84 | 0.162 | – | 0.00 | 0.000 | 0.0085 | 0.0037 | 0.0029 |
| tbad | 0.80 | 0.76 | 0.856 | – | 0.00 | 0.000 | 0.0130 | 0.0053 | 0.0022 |
| djit | 0.70 | 1.00 | 1.693 | 0.89 | 0.94 | 0.171 | 0.0000 | 0.0050 | 0.0016 |
| czslow | 0.50 | 0.38 | 0.081 | 0.66 | 0.56 | 0.014 | 0.0000 | 0.0030 | 0.0024 |
| mslow | 0.86 | 1.69 | 0.000 | 0.99 | 1.05 | 0.335 | 0.0000 | 0.0038 | 0.0037 |
| mix | 0.84 | 0.91 | 1.698 | 0.87 | 0.89 | 0.152 | 0.0072 | 0.0139 | 0.0017 |

## v8kdp10 (v20): v19 + ESP-only placement trained on the exact layout

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.76 | 0.78 | 0.162 | – | 0.00 | 0.000 | 0.0085 | 0.0046 | 0.0039 |
| tbad | 0.68 | 0.63 | 0.856 | – | 0.00 | 0.000 | 0.0131 | 0.0073 | 0.0057 |
| djit | 0.67 | 0.95 | 1.693 | 0.82 | 0.92 | 0.171 | 0.0000 | 0.0053 | 0.0029 |
| czslow | 0.44 | 0.34 | 0.081 | 0.60 | 0.52 | 0.014 | 0.0000 | 0.0041 | 0.0039 |
| mslow | 0.86 | 1.62 | 0.000 | 0.99 | 1.05 | 0.335 | 0.0000 | 0.0045 | 0.0048 |
| mix | 0.80 | 0.86 | 1.698 | 0.80 | 0.88 | 0.152 | 0.0069 | 0.0147 | 0.0029 |

## v8kdfi10 (v19i): the formula head on v8kdi10

| kind | ESP tracking | ESP slope | median abs true Δlog ESP | duration tracking | duration slope | median abs true Δlog duration | spurious Δlog F | ESP regret, keep old choice | ESP regret, re-predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tjit | 0.72 | 0.75 | 0.162 | – | 0.00 | 0.000 | 0.0075 | 0.0045 | 0.0038 |
| tbad | 0.72 | 0.74 | 0.856 | – | 0.00 | 0.000 | 0.0125 | 0.0064 | 0.0054 |
| djit | 0.70 | 0.98 | 1.693 | 0.84 | 0.92 | 0.171 | 0.0000 | 0.0059 | 0.0027 |
| czslow | 0.47 | 0.37 | 0.081 | 0.58 | 0.55 | 0.014 | 0.0000 | 0.0039 | 0.0037 |
| mslow | 0.87 | 1.16 | 0.000 | 0.99 | 1.05 | 0.335 | 0.0000 | 0.0045 | 0.0046 |
| mix | 0.78 | 0.91 | 1.698 | 0.82 | 0.89 | 0.152 | 0.0057 | 0.0138 | 0.0026 |

Kinds: tjit = T1/T2 × lognormal(0.3) on every qubit; tbad = 2–3 qubits with T × 0.15–0.3; djit = every gate duration × lognormal(0.2); czslow = 3–5 couplers with the CZ 2–3× slower; mslow = readout 1.3–2× slower; mix = tjit + djit. A tracking of – means the model's prediction does not move at all (no input carries that change).
