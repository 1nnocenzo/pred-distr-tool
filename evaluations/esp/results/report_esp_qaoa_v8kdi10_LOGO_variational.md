# Reverse engineering: v8kdi10, LOGO variational

89352 (circuit, device, seed) cases.

| group | cases | duration error | duration pred/true (median) | CZ count pred/true | 1q count pred/true | busiest-qubit gate time pred/true |
| --- | --- | --- | --- | --- | --- | --- |
| all | 89352 | 0.06 | 0.98 | 0.96 | 0.97 | 1.24 |
| routing-free | 60795 | 0.05 | 0.98 | 0.96 | 0.90 | 1.35 |
| routed | 28557 | 0.11 | 0.98 | 1.05 | 1.17 | 0.85 |
| 1–5 qubits | 19404 | 0.03 | 1.01 | 0.97 | 1.17 | 1.13 |
| 6–10 qubits | 24255 | 0.06 | 0.99 | 0.96 | 1.09 | 1.22 |
| 11–15 qubits | 24147 | 0.08 | 0.96 | 0.96 | 0.91 | 1.31 |
| 16–20 qubits | 21546 | 0.11 | 0.92 | 0.96 | 0.78 | 1.29 |

corr(log duration ratio, log CZ ratio) = 0.53; with log busiest-qubit gate time ratio = 0.56; with log 1q ratio = 0.60

## Depth

| | value |
| --- | --- |
| depth error (median) | 0.37 |
| depth pred/true (median) | 0.99 |
| busiest-qubit operations pred/true (median) | 1.37 |
| depth / busiest-qubit ops, true: median [p10, p90] | 1.83 [1.05, 4.00] |
| depth / busiest-qubit ops, predicted: median [p10, p90] | 1.35 [1.06, 2.27] |
| corr(log depth ratio, log busiest-ops ratio) | 0.67 |