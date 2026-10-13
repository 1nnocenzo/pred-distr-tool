# Routed circuits: duration and ESP errors by routing overhead, size and device

Routed (circuit, device) cases only (not routing-free on that device), seeds pooled. Duration error = median |pred/true − 1|, bias = median pred/true; ESP error = median |log ESP_pred − log ESP_true| (×100 ≈ % error); routing overhead = true CZ / CZ of the gates decomposed alone − 1.

## v8kdi10

### LOFO qaoa

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 20142 | 0.125 | 0.299 | 0.91 | 0.594 | nan |
| routing overhead 0–0.25 | 117 | 0.014 | 0.068 | 0.99 | 0.138 | -216.117 |
| routing overhead 0.25–0.75 | 9369 | 0.100 | 0.243 | 0.93 | 0.611 | 0.790 |
| routing overhead 0.75–1.5 | 10656 | 0.154 | 0.338 | 0.88 | nan | nan |
| 2–5 qubits | 1404 | 0.075 | 0.194 | 0.93 | 0.246 | 0.881 |
| 6–10 qubits | 7020 | 0.110 | 0.263 | 0.91 | 0.775 | 0.624 |
| 11–15 qubits | 7020 | 0.148 | 0.314 | 0.88 | 1.747 | nan |
| 16–20 qubits | 4698 | 0.142 | 0.339 | 0.91 | nan | nan |
| EQE1_Top | 6714 | 0.136 | 0.314 | 0.88 | 0.452 | 0.910 |
| EQE1_Bottom | 6714 | 0.114 | 0.282 | 0.93 | 0.589 | 0.771 |
| QExa20 | 6714 | 0.122 | 0.300 | 0.90 | 0.809 | nan |

### LOFO qnn

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 120 | 0.038 | 0.067 | 1.02 | nan | -6.255 |
| routing overhead 0.25–0.75 | 120 | 0.038 | 0.067 | 1.02 | nan | -6.255 |
| 16–20 qubits | 120 | 0.038 | 0.067 | 1.02 | nan | -6.255 |
| QExa20 | 120 | 0.038 | 0.067 | 1.02 | nan | -6.255 |

### LOGO variational

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 28557 | 0.126 | 0.392 | 0.94 | 0.646 | nan |
| routing overhead 0–0.25 | 4704 | 0.108 | 0.256 | 0.96 | 0.913 | 0.884 |
| routing overhead 0.25–0.75 | 6453 | 0.082 | 0.231 | 1.01 | 0.842 | 0.581 |
| routing overhead 0.75–1.5 | 15423 | 0.143 | 0.400 | 0.92 | 0.490 | nan |
| routing overhead 1.5–99 | 1977 | 0.404 | 0.559 | 0.60 | nan | nan |
| 2–5 qubits | 5571 | 0.059 | 0.176 | 1.02 | 0.428 | 0.425 |
| 6–10 qubits | 7821 | 0.113 | 0.302 | 0.99 | 0.886 | 0.649 |
| 11–15 qubits | 8811 | 0.173 | 0.451 | 0.86 | 1.012 | nan |
| 16–20 qubits | 6354 | 0.207 | 0.469 | 0.81 | 1.486 | nan |
| EQE1_Top | 9105 | 0.129 | 0.366 | 0.94 | 0.578 | 0.811 |
| EQE1_Bottom | 9105 | 0.133 | 0.412 | 0.96 | 0.632 | 0.755 |
| QExa20 | 10347 | 0.119 | 0.394 | 0.93 | 0.836 | nan |

### control

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 23884 | 0.045 | 0.138 | 0.99 | 0.304 | nan |
| routing overhead 0–0.25 | 2045 | 0.034 | 0.137 | 1.00 | 0.563 | nan |
| routing overhead 0.25–0.75 | 12321 | 0.041 | 0.132 | 0.99 | 0.282 | nan |
| routing overhead 0.75–1.5 | 8848 | 0.051 | 0.138 | 0.99 | 0.279 | nan |
| routing overhead 1.5–99 | 575 | 0.067 | 0.166 | 0.94 | 1.791 | nan |
| 2–5 qubits | 3975 | 0.026 | 0.100 | 0.99 | 0.244 | 0.824 |
| 6–10 qubits | 7611 | 0.045 | 0.145 | 0.99 | 0.325 | nan |
| 11–15 qubits | 7161 | 0.048 | 0.144 | 0.99 | 0.461 | nan |
| 16–20 qubits | 5137 | 0.056 | 0.145 | 0.99 | 0.691 | nan |
| EQE1_Top | 7837 | 0.046 | 0.144 | 0.99 | 0.282 | 0.920 |
| EQE1_Bottom | 7837 | 0.044 | 0.142 | 0.99 | 0.271 | 0.931 |
| QExa20 | 8210 | 0.044 | 0.125 | 0.99 | 0.388 | nan |

## v8kdfi10

### LOFO qaoa

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 20142 | 0.153 | 0.330 | 0.86 | 0.528 | nan |
| routing overhead 0–0.25 | 117 | 0.046 | 0.089 | 0.97 | 0.294 | -439.485 |
| routing overhead 0.25–0.75 | 9369 | 0.110 | 0.244 | 0.91 | 0.541 | 0.666 |
| routing overhead 0.75–1.5 | 10656 | 0.202 | 0.380 | 0.81 | nan | nan |
| 2–5 qubits | 1404 | 0.076 | 0.226 | 0.93 | 0.226 | 0.802 |
| 6–10 qubits | 7020 | 0.114 | 0.249 | 0.90 | 0.800 | 0.402 |
| 11–15 qubits | 7020 | 0.192 | 0.367 | 0.82 | 2.430 | nan |
| 16–20 qubits | 4698 | 0.185 | 0.376 | 0.82 | nan | nan |
| EQE1_Top | 6714 | 0.185 | 0.375 | 0.82 | 0.461 | 0.870 |
| EQE1_Bottom | 6714 | 0.123 | 0.298 | 0.89 | 1.109 | 0.545 |
| QExa20 | 6714 | 0.155 | 0.302 | 0.86 | 0.400 | nan |

### LOFO qnn

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 120 | 0.032 | 0.067 | 1.01 | nan | -2.907 |
| routing overhead 0.25–0.75 | 120 | 0.032 | 0.067 | 1.01 | nan | -2.907 |
| 16–20 qubits | 120 | 0.032 | 0.067 | 1.01 | nan | -2.907 |
| QExa20 | 120 | 0.032 | 0.067 | 1.01 | nan | -2.907 |

### LOGO variational

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 28557 | 0.155 | 0.416 | 0.94 | 0.634 | nan |
| routing overhead 0–0.25 | 4704 | 0.123 | 0.286 | 1.03 | 0.955 | 0.879 |
| routing overhead 0.25–0.75 | 6453 | 0.106 | 0.285 | 1.05 | 0.783 | 0.640 |
| routing overhead 0.75–1.5 | 15423 | 0.178 | 0.409 | 0.87 | 0.470 | nan |
| routing overhead 1.5–99 | 1977 | 0.437 | 0.561 | 0.56 | nan | nan |
| 2–5 qubits | 5571 | 0.083 | 0.217 | 1.04 | 0.399 | 0.514 |
| 6–10 qubits | 7821 | 0.137 | 0.341 | 0.98 | 0.902 | 0.695 |
| 11–15 qubits | 8811 | 0.212 | 0.461 | 0.84 | 1.008 | nan |
| 16–20 qubits | 6354 | 0.215 | 0.473 | 0.83 | 1.460 | nan |
| EQE1_Top | 9105 | 0.154 | 0.395 | 0.92 | 0.550 | 0.844 |
| EQE1_Bottom | 9105 | 0.171 | 0.445 | 0.96 | 0.639 | 0.789 |
| QExa20 | 10347 | 0.145 | 0.400 | 0.94 | 0.770 | nan |

### control

| group | cases | duration error | duration p90 | duration bias | ESP error (log, ESP > 1e-3) | ESP R² |
| --- | --- | --- | --- | --- | --- | --- |
| all routed | 23884 | 0.042 | 0.137 | 1.01 | 0.278 | nan |
| routing overhead 0–0.25 | 2045 | 0.032 | 0.112 | 1.01 | 0.498 | nan |
| routing overhead 0.25–0.75 | 12321 | 0.039 | 0.132 | 1.00 | 0.250 | nan |
| routing overhead 0.75–1.5 | 8848 | 0.049 | 0.138 | 1.01 | 0.262 | nan |
| routing overhead 1.5–99 | 575 | 0.081 | 0.175 | 0.93 | 2.248 | nan |
| 2–5 qubits | 3975 | 0.028 | 0.095 | 1.01 | 0.248 | 0.843 |
| 6–10 qubits | 7611 | 0.046 | 0.149 | 1.01 | 0.268 | nan |
| 11–15 qubits | 7161 | 0.047 | 0.144 | 1.00 | 0.419 | nan |
| 16–20 qubits | 5137 | 0.047 | 0.131 | 1.00 | 0.651 | nan |
| EQE1_Top | 7837 | 0.044 | 0.134 | 0.99 | 0.250 | 0.935 |
| EQE1_Bottom | 7837 | 0.045 | 0.144 | 1.01 | 0.236 | 0.940 |
| QExa20 | 8210 | 0.039 | 0.130 | 1.01 | 0.352 | nan |
