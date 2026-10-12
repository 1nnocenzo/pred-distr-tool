# Fixed per-gate decomposition vs the true compiled counts (EQE1_Top, no model)

Table: each gate type alone, level 2, adjacent qubits, generic angles (results/decomp_table.json). Merge factor = true / Σ table over the circuit's gates.

Routing-free circuits: 8294. Variance of log true r explained by log table r (one global factor): R² 0.828; CZ: R² 0.964

| family | routing-free circuits | r merge factor, median [p10, p90] | CZ true / table, median | held-out rule: r error | table alone: r error |
| --- | --- | --- | --- | --- | --- |
| ae | 40 | 0.13 [0.11, 0.14] | 1.00 | 6.90 | 6.90 |
| bmw_quark_cardinality | 19 | 0.66 [0.64, 0.71] | 1.00 | 0.52 | 0.52 |
| dynamic_qft | 19 | 8.50 [3.10, 13.90] | – | 0.88 | 0.88 |
| ghz | 19 | 1.00 [1.00, 1.00] | 1.00 | 0.00 | 0.00 |
| iqpe | 570 | 1.22 [0.65, 1.79] | 0.88 | 0.32 | 0.32 |
| qaoa | 468 | 0.48 [0.14, 0.52] | 1.00 | 1.06 | 1.06 |
| qft | 40 | 0.63 [0.52, 0.68] | 1.50 | 0.58 | 0.58 |
| qftentangled | 40 | 0.37 [0.34, 0.41] | 0.67 | 1.69 | 1.69 |
| qnn | 760 | 0.71 [0.34, 0.92] | 1.00 | 0.41 | 0.41 |
| randomcircuit | 133 | 0.29 [0.20, 0.65] | 0.55 | 2.50 | 2.50 |
| vqe_real_amp | 1920 | 1.00 [1.00, 1.00] | 1.00 | 0.04 | 0.00 |
| vqe_su2 | 1920 | 0.95 [0.70, 1.00] | 1.00 | 0.05 | 0.05 |
| vqe_two_local | 2293 | 1.00 [0.91, 1.00] | 1.00 | 0.03 | 0.00 |
| wstate | 19 | 1.00 [1.00, 1.00] | 1.00 | 0.00 | 0.00 |

## Routed circuits: CZ added by routing

| family | routed circuits | true CZ / table CZ, median [p10, p90] |
| --- | --- | --- |
| ae | 718 | 1.61 [1.43, 1.75] |
| bv | 10 | 1.60 [1.17, 1.86] |
| dj | 15 | 1.95 [1.50, 2.56] |
| graphstate | 10 | 1.50 [1.31, 1.91] |
| hhl | 41 | 1.62 [1.44, 1.78] |
| qaoa | 2238 | 1.80 [1.29, 2.12] |
| qft | 720 | 1.62 [1.45, 1.73] |
| qftentangled | 706 | 1.59 [1.38, 1.71] |
| qpeexact | 17 | 1.62 [1.44, 1.75] |
| qpeinexact | 18 | 1.61 [1.45, 1.73] |
| randomcircuit | 1101 | 1.43 [1.15, 1.70] |
| vqe_real_amp | 941 | 2.10 [1.20, 2.38] |
| vqe_su2 | 938 | 2.04 [1.20, 2.38] |
| vqe_two_local | 1156 | 2.00 [1.20, 2.33] |