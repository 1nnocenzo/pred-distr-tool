# Pipeline: switch -> exact compile (routing-free / small) or light-Sabre compile (2026-10-16)

800 circuits (2400 circuit-device cases; random sample of the dataset, <= 20 qubits, static circuits), one CPU thread. Reference = Qiskit level 2 seed 0 (as the labels). Seed 1 = the compiler's own noise floor.

## Switch

- agreement with the compiler's decision (VF2Layout found a layout, no SWAPs): 99.2%
- with the cheap checks + limit 2000: median 0.038 ms, p99 0.75 ms, max 9.27 ms; routing-free cases lost vs limit 20k: 23 of 1007
- limit 20k without the checks would cost: max 10.09 ms
- decided by: {'vf2': 1020, 'degree': 310, 'size': 701, 'odd cycle': 366, 'no 2q gates': 3}

## Errors vs the reference (seed 0)

Routed cases have tiny ESP (median log ESP -17.2, i.e. ESP ~ 3e-08): a 5% change of the duration moves log ESP by ~1, so ESP errors are given in log and relative to abs log ESP too.

| cases | what | duration err median / p90 | duration ratio | abs Δlog ESP median / p90 | abs Δlog ESP / abs log ESP | abs Δlog F median |
|---|---|---|---|---|---|---|
| routed (n=1393) | light compile | 5.4% / 17.8% | 1.03 | 2.19 / 15.23 | 12.3% | 0.504 |
| routed | compiler seed 1 (noise floor) | 3.9% / 13.8% | 1.00 | 1.70 / 10.58 | 9.0% | 0.368 |
| routed, ESP > 1e-3 (as the model reports) (n=406) | light compile | 1.7% / 9.5% | 1.00 | 0.29 / 1.94 | 11.4% | 0.050 |
| routed, ESP > 1e-3 (as the model reports) | compiler seed 1 (noise floor) | 0.9% / 6.1% | 1.00 | 0.20 / 1.12 | 8.4% | 0.036 |
| routing-free (n=1007) | light compile | 0.0% / 0.0% | 1.00 | 0.00 / 0.00 | 0.0% | 0.000 |
| routing-free | compiler seed 1 (noise floor) | 0.0% / 0.0% | 1.00 | 0.00 / 0.00 | 0.0% | 0.000 |

For comparison, the models on routed cases (vs the labels, ESP error only where ESP > 1e-3): v8kdi10 + v19i duration 12.5% (LOFO qaoa), 12.6% (LOGO variational), 4.5% (control); abs Δlog ESP 0.59 / 0.65 / 0.30 (esp/results/report_esp_routed_v10_v19i.md); route stage v27 duration 8.8% / 5.4% / 3.2%.

Routed, per family (light compile; seed-1 floor = the compiler's own spread):

| family | n | duration err median / p90 | ratio | abs Δlog ESP median / p90 | rel. | abs Δlog F | floor duration | floor abs Δlog ESP |
|---|---|---|---|---|---|---|---|---|
| qaoa | 354 | 6.6% / 17.3% | 1.06 | 2.22 / 11.47 | 11.6% | 0.541 | 4.8% | 1.93 |
| randomcircuit | 234 | 3.9% / 13.7% | 1.02 | 6.86 / 47.63 | 10.1% | 1.577 | 3.9% | 5.08 |
| vqe_su2 | 178 | 3.4% / 19.4% | 1.02 | 1.60 / 12.48 | 13.8% | 0.276 | 2.0% | 0.82 |
| vqe_real_amp | 172 | 4.0% / 24.0% | 1.04 | 1.74 / 11.46 | 21.0% | 0.280 | 2.1% | 0.82 |
| vqe_two_local | 145 | 3.7% / 16.9% | 1.03 | 0.84 / 7.14 | 13.4% | 0.135 | 1.5% | 0.79 |
| qft | 123 | 6.9% / 22.0% | 1.03 | 2.27 / 15.41 | 14.2% | 0.747 | 5.0% | 2.40 |
| ae | 81 | 4.8% / 15.3% | 1.01 | 4.69 / 14.02 | 8.7% | 0.454 | 4.9% | 1.90 |
| qftentangled | 81 | 8.2% / 20.4% | 1.00 | 1.62 / 9.75 | 9.4% | 0.665 | 9.1% | 3.66 |
| hhl | 6 | 1.7% / 12.2% | 1.01 | 0.72 / 2.76 | 5.1% | 0.554 | 4.1% | 0.57 |

## Time per circuit (3 devices, sum)

| SMALL (exact up to this many 2q gates) | pipeline total | full compile total | saving | routed-case duration err median | share compiled exactly |
|---|---|---|---|---|---|
| 0 | 93.0 s | 310.6 s | 70% | 4.9% | 46% |
| 20 | 128.0 s | 310.6 s | 59% | 6.2% | 63% |
| 50 | 170.5 s | 310.6 s | 45% | 8.4% | 78% |
| 100 | 216.7 s | 310.6 s | 30% | 9.6% | 88% |
| 200 | 253.1 s | 310.6 s | 19% | 9.5% | 95% |

Per circuit size (sum over the 3 devices, median; SMALL = 0):

| 2q gates | n | full compile | pipeline | ratio | of which switch | light compile alone |
|---|---|---|---|---|---|---|
| 0-20 | 232 | 33.5 ms | 26.6 ms | 1.01 | 0.18 ms | 19.1 ms |
| 20-50 | 143 | 86.1 ms | 53.7 ms | 0.55 | 0.22 ms | 36.8 ms |
| 50-200 | 104 | 531.4 ms | 112.3 ms | 0.27 | 0.07 ms | 68.9 ms |
| 200-600 | 28 | 1844.9 ms | 254.9 ms | 0.16 | 0.03 ms | 131.9 ms |
| 600-… | 4 | 2785.8 ms | 403.2 ms | 0.15 | 0.03 ms | 190.1 ms |

Per family (routed families, sum over the 3 devices, median):

| family | n | full compile | pipeline | ratio |
|---|---|---|---|---|
| qaoa | 60 | 466.9 ms | 176.2 ms | 0.39 |
| randomcircuit | 76 | 1119.7 ms | 289.8 ms | 0.27 |
| qft | 18 | 287.4 ms | 287.4 ms | 1.00 |
| qftentangled | 16 | 273.0 ms | 84.4 ms | 0.31 |
| qnn | 17 | 32.6 ms | 33.0 ms | 1.01 |
| vqe_su2 | 96 | 57.0 ms | 42.9 ms | 1.01 |
| vqe_real_amp | 101 | 55.1 ms | 39.0 ms | 1.01 |
| vqe_two_local | 98 | 31.6 ms | 26.9 ms | 1.01 |
| ae | 14 | 271.6 ms | 97.4 ms | 0.30 |
| graphstate | 3 | 10.3 ms | 7.8 ms | 1.02 |
