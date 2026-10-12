# Reverse engineering: v8kdi10, LOFO qaoa

24354 (circuit, device, seed) cases.

| group | cases | duration error | duration pred/true (median) | CZ count pred/true | 1q count pred/true | busiest-qubit gate time pred/true |
| --- | --- | --- | --- | --- | --- | --- |
| all | 24354 | 0.36 | 0.66 | 0.93 | 0.62 | 0.58 |
| routing-free | 4212 | 0.57 | 0.50 | 0.75 | 0.53 | 0.56 |
| routed | 20142 | 0.33 | 0.69 | 0.96 | 0.63 | 0.59 |
| 1–5 qubits | 5616 | 0.52 | 0.55 | 0.77 | 0.58 | 0.57 |
| 6–10 qubits | 7020 | 0.33 | 0.68 | 1.04 | 0.75 | 0.66 |
| 11–15 qubits | 7020 | 0.31 | 0.70 | 0.94 | 0.62 | 0.57 |
| 16–20 qubits | 4698 | 0.35 | 0.68 | 0.80 | 0.46 | 0.50 |

corr(log duration ratio, log CZ ratio) = 0.68; with log busiest-qubit gate time ratio = 0.89; with log 1q ratio = 0.65

## Depth

| | value |
| --- | --- |
| depth error (median) | 0.43 |
| depth pred/true (median) | 0.59 |
| busiest-qubit operations pred/true (median) | 0.53 |
| depth / busiest-qubit ops, true: median [p10, p90] | 1.76 [1.11, 2.29] |
| depth / busiest-qubit ops, predicted: median [p10, p90] | 1.93 [1.07, 3.03] |
| corr(log depth ratio, log busiest-ops ratio) | 0.87 |