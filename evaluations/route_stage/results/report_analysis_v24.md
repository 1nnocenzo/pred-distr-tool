# v24b analysis: routed-circuit duration

Rows of the 3 v24b seeds pooled (each seed scores the same test cases). Cells: median |pred/true − 1| / p90 (median pred/true), vs Sabre's final duration unless noted. *pred1* = stage 1 only, *pred* = with stage 2, *_mc* = median schedule over 24 Poisson samples of the SWAPs.

## qaoa (20142 routed / 4212 routing-free cases, pooled seeds)

### Error by stage

| what | med / p90 (bias) |
| --- | --- |
| Sabre final vs dataset label (label mismatch) | 4.6% / 16% (1.04) |
| Sabre's routed schedule replayed (no optimisation) | 5.4% / 9% (0.95) |
|   + stage 2 (on the true schedule) | 8.2% / 14% (0.92) |
| true SWAPs + true waits (representation ceiling) | 3.6% / 7% (0.97) |
| true SWAPs, no waits | 17.4% / 35% (0.83) |
| predicted SWAPs + true waits | 5.2% / 10% (0.95) |
| true SWAPs + predicted waits | 12.0% / 27% (0.88) |
| stage 1, mean rates | 13.3% / 31% (0.87) |
| stage 1, sampled | 10.8% / 27% (0.89) |
| stage 1 sampled + stage 2 | 14.0% / 30% (0.86) |
| stage 1 sampled vs dataset | 8.6% / 21% (0.93) |
| stage 1 sampled + stage 2 vs dataset | 10.6% / 25% (0.90) |

### Where the error is (stage 1 sampled vs with stage 2)

| device | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| EQE1_Top | 6714 | 10.8% / 25% (0.89) | 13.7% / 29% (0.86) | 6% |
| EQE1_Bottom | 6714 | 10.3% / 28% (0.90) | 13.2% / 30% (0.87) | 12% |
| QExa20 | 6714 | 11.3% / 27% (0.89) | 15.0% / 31% (0.85) | 12% |

| qubits | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–6 | 1404 | 3.2% / 13% (0.97) | 10.8% / 24% (0.89) | 0% |
| 6–11 | 7020 | 6.1% / 11% (0.98) | 6.1% / 15% (0.95) | 0% |
| 11–16 | 7020 | 18.3% / 27% (0.82) | 21.5% / 32% (0.78) | 13% |
| 16–21 | 4698 | 21.5% / 31% (0.78) | 22.7% / 33% (0.77) | 23% |

| 2q gates | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–20 | 2817 | 4.3% / 10% (1.01) | 4.1% / 14% (0.98) | 0% |
| 20–50 | 4563 | 6.9% / 11% (0.98) | 7.3% / 16% (0.94) | 0% |
| 50–100 | 5985 | 10.6% / 22% (0.89) | 13.5% / 24% (0.87) | 3% |
| 100–200 | 5247 | 22.2% / 29% (0.78) | 25.0% / 33% (0.75) | 17% |
| 200–500 | 1530 | 27.5% / 34% (0.72) | 30.4% / 36% (0.70) | 62% |

| SWAPs per 2q gate | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0.25–0.5 | 1179 | 4.4% / 8% (1.04) | 1.4% / 4% (0.99) | 0% |
| 0.5–1 | 7014 | 6.7% / 11% (0.97) | 8.2% / 16% (0.92) | 0% |
| 1–2 | 11949 | 19.0% / 29% (0.81) | 21.8% / 32% (0.78) | 17% |

| routing share of duration (1 − logical / final) | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| -inf–0.2 | 5886 | 4.2% / 10% (1.02) | 3.6% / 12% (0.98) | 0% |
| 0.2–0.4 | 7164 | 9.5% / 16% (0.91) | 12.9% / 21% (0.87) | 0% |
| 0.4–0.6 | 7092 | 24.0% / 30% (0.76) | 26.9% / 34% (0.73) | 28% |

### Representation ceiling: true SWAPs + true waits vs Sabre's replayed schedule

Cases with unmatched gates: 0%; with negative waits: 100%.

| by | bin | n | oracle (SWAPs + waits) | replay | median extra error, points |
| --- | --- | --- | --- | --- | --- |
| unmatched share of gates | 0–1e-09 | 20142 | 3.6% / 7% (0.97) | 5.4% / 9% (0.95) | -1.5 |
| negative waits / duration | 0.02–0.05 | 1122 | 0.9% / 2% (1.00) | 1.4% / 3% (0.99) | -0.8 |
| negative waits / duration | 0.05–0.1 | 4659 | 3.1% / 7% (1.02) | 3.0% / 7% (0.99) | +0.2 |
| negative waits / duration | ≥0.1 | 14361 | 4.5% / 7% (0.96) | 6.2% / 9% (0.94) | -1.7 |

### Waits

|  | value |
| --- | --- |
| matched gates with a positive / negative wait | 25.6% / 64.2% |
| positive waits / duration (median over cases) | 44.5% |
| net waits on the critical path / duration: true / predicted (median) | 18.7% / 5.4% |
| negative waits / duration (median) | 12.2% |
| predicted SWAPs on that path / true | 0.77 |
| |wait error| on the critical path / all |wait error| | 36.0% (path holds 20% of the gates) |
| per-gate corr(pred, true wait): all / critical / off-path | 0.12 / 0.22 / 0.08 |
| true waits that are zero: predicted |wait| mean (µs) | 0.097 |
| SWAP mass on slots with no SWAP | 50% |

### Stage 2 by routing overhead (SWAPs per 2q gate)

| overhead | n | replay | replay + stage 2 | stage 1 sampled | + stage 2 | stage-2 factor (median) | true final / replay (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25–0.5 | 1179 | 2.6% / 5% (1.03) | 2.6% / 5% (0.97) | 4.4% / 8% (1.04) | 1.4% / 4% (0.99) | 0.962 | 0.975 |
| 0.5–1 | 7014 | 3.3% / 7% (0.97) | 6.6% / 13% (0.93) | 6.7% / 11% (0.97) | 8.2% / 16% (0.92) | 0.965 | 1.031 |
| 1–2 | 11949 | 6.7% / 9% (0.93) | 9.5% / 14% (0.91) | 19.0% / 29% (0.81) | 21.8% / 32% (0.78) | 0.969 | 1.072 |

## variational (28557 routed / 60795 routing-free cases, pooled seeds)

### Error by stage

| what | med / p90 (bias) |
| --- | --- |
| Sabre final vs dataset label (label mismatch) | 0.0% / 1% (1.00) |
| Sabre's routed schedule replayed (no optimisation) | 1.1% / 4% (0.99) |
|   + stage 2 (on the true schedule) | 3.6% / 12% (1.01) |
| true SWAPs + true waits (representation ceiling) | 8.5% / 28% (1.06) |
| true SWAPs, no waits | 10.8% / 32% (0.89) |
| predicted SWAPs + true waits | 8.4% / 32% (1.07) |
| true SWAPs + predicted waits | 7.0% / 26% (0.93) |
| stage 1, mean rates | 9.8% / 33% (0.91) |
| stage 1, sampled | 5.7% / 21% (0.96) |
| stage 1 sampled + stage 2 | 5.3% / 17% (0.97) |
| stage 1 sampled vs dataset | 5.7% / 21% (0.96) |
| stage 1 sampled + stage 2 vs dataset | 5.5% / 17% (0.97) |

### Where the error is (stage 1 sampled vs with stage 2)

| device | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| EQE1_Top | 9105 | 5.1% / 20% (0.97) | 5.3% / 17% (0.98) | 9% |
| EQE1_Bottom | 9105 | 6.1% / 20% (0.96) | 5.6% / 15% (0.97) | 8% |
| QExa20 | 10347 | 5.9% / 23% (0.95) | 5.2% / 17% (0.96) | 13% |

| qubits | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–6 | 5571 | 3.6% / 8% (0.97) | 4.4% / 9% (0.96) | 0% |
| 6–11 | 7821 | 9.2% / 25% (0.91) | 6.3% / 21% (0.95) | 17% |
| 11–16 | 8811 | 6.9% / 22% (0.96) | 5.7% / 17% (0.99) | 12% |
| 16–21 | 6354 | 5.7% / 18% (0.98) | 5.4% / 15% (1.00) | 7% |

| 2q gates | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–20 | 7206 | 2.6% / 7% (0.98) | 3.7% / 8% (0.96) | 0% |
| 20–50 | 7410 | 4.1% / 13% (0.98) | 3.9% / 11% (0.98) | 1% |
| 50–100 | 5292 | 8.0% / 24% (0.97) | 7.4% / 20% (0.99) | 17% |
| 100–200 | 4806 | 12.9% / 25% (0.88) | 9.7% / 23% (0.96) | 19% |
| 200–500 | 3843 | 14.9% / 26% (0.85) | 8.7% / 20% (0.95) | 26% |

| SWAPs per 2q gate | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–0.25 | 10542 | 2.6% / 8% (1.00) | 3.2% / 8% (0.99) | 2% |
| 0.25–0.5 | 4557 | 4.6% / 14% (0.95) | 5.1% / 12% (0.95) | 2% |
| 0.5–1 | 10959 | 11.7% / 25% (0.89) | 9.0% / 21% (0.95) | 18% |
| 1–2 | 2499 | 13.7% / 25% (0.87) | 8.1% / 20% (0.96) | 22% |

| routing share of duration (1 − logical / final) | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| -inf–0.2 | 14328 | 2.8% / 8% (0.99) | 3.5% / 9% (0.98) | 1% |
| 0.2–0.4 | 3294 | 7.9% / 15% (0.92) | 6.6% / 14% (0.97) | 0% |
| 0.4–0.6 | 6951 | 13.3% / 25% (0.87) | 10.3% / 22% (0.95) | 20% |
| 0.6–0.8 | 3984 | 16.9% / 27% (0.83) | 9.4% / 21% (0.92) | 31% |

| family | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| qnn | 120 | 2.6% / 5% (0.98) | 5.6% / 10% (0.94) | 0% |
| vqe_real_amp | 8820 | 5.6% / 21% (0.96) | 5.2% / 16% (0.97) | 10% |
| vqe_su2 | 8793 | 7.3% / 23% (0.95) | 6.5% / 18% (0.96) | 14% |
| vqe_two_local | 10824 | 4.6% / 19% (0.97) | 4.6% / 15% (0.98) | 7% |

### Representation ceiling: true SWAPs + true waits vs Sabre's replayed schedule

Cases with unmatched gates: 98%; with negative waits: 99%.

| by | bin | n | oracle (SWAPs + waits) | replay | median extra error, points |
| --- | --- | --- | --- | --- | --- |
| unmatched share of gates | 0–1e-09 | 495 | 0.8% / 1% (1.00) | 0.6% / 1% (0.99) | -0.0 |
| unmatched share of gates | 0.05–0.2 | 5949 | 19.6% / 46% (1.19) | 1.9% / 4% (1.00) | +17.6 |
| unmatched share of gates | 0.2–0.5 | 21060 | 7.2% / 19% (1.05) | 1.1% / 4% (0.99) | +5.8 |
| unmatched share of gates | ≥0.5 | 1053 | 8.8% / 13% (0.92) | 0.7% / 2% (1.00) | +8.2 |
| negative waits / duration | 0–1e-09 | 300 | 0.8% / 1% (0.99) | 0.8% / 1% (0.99) | -0.0 |
| negative waits / duration | 1e-09–0.02 | 597 | 5.1% / 9% (0.95) | 1.0% / 1% (0.99) | +3.9 |
| negative waits / duration | 0.02–0.05 | 1548 | 3.5% / 9% (0.96) | 0.8% / 1% (0.99) | +2.9 |
| negative waits / duration | 0.05–0.1 | 924 | 1.9% / 5% (1.02) | 0.9% / 2% (1.00) | +1.2 |
| negative waits / duration | ≥0.1 | 25188 | 9.7% / 31% (1.08) | 1.3% / 4% (0.99) | +8.3 |

### Waits

|  | value |
| --- | --- |
| matched gates with a positive / negative wait | 35.8% / 49.5% |
| positive waits / duration (median over cases) | 23.0% |
| net waits on the critical path / duration: true / predicted (median) | 2.4% / 2.3% |
| negative waits / duration (median) | 101.2% |
| predicted SWAPs on that path / true | 1.31 |
| |wait error| on the critical path / all |wait error| | 19.5% (path holds 27% of the gates) |
| per-gate corr(pred, true wait): all / critical / off-path | -0.00 / 0.03 / -0.00 |
| true waits that are zero: predicted |wait| mean (µs) | 0.074 |
| SWAP mass on slots with no SWAP | 62% |

### Stage 2 by routing overhead (SWAPs per 2q gate)

| overhead | n | replay | replay + stage 2 | stage 1 sampled | + stage 2 | stage-2 factor (median) | true final / replay (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0–0.25 | 10542 | 0.8% / 2% (1.00) | 1.8% / 4% (0.98) | 2.6% / 8% (1.00) | 3.2% / 8% (0.99) | 0.985 | 1.005 |
| 0.25–0.5 | 4557 | 1.2% / 5% (0.99) | 2.1% / 5% (0.99) | 4.6% / 14% (0.95) | 5.1% / 12% (0.95) | 0.998 | 1.008 |
| 0.5–1 | 10959 | 2.3% / 5% (0.99) | 7.7% / 14% (1.08) | 11.7% / 25% (0.89) | 9.0% / 21% (0.95) | 1.086 | 1.009 |
| 1–2 | 2499 | 1.8% / 5% (1.00) | 10.7% / 18% (1.11) | 13.7% / 25% (0.87) | 8.1% / 20% (0.96) | 1.114 | 1.003 |

## control (23850 routed / 20511 routing-free cases, pooled seeds)

### Error by stage

| what | med / p90 (bias) |
| --- | --- |
| Sabre final vs dataset label (label mismatch) | 0.0% / 6% (1.00) |
| Sabre's routed schedule replayed (no optimisation) | 3.7% / 12% (0.99) |
|   + stage 2 (on the true schedule) | 1.2% / 4% (1.00) |
| true SWAPs + true waits (representation ceiling) | 7.0% / 19% (0.98) |
| true SWAPs, no waits | 15.1% / 35% (0.85) |
| predicted SWAPs + true waits | 7.1% / 19% (0.98) |
| true SWAPs + predicted waits | 9.2% / 26% (0.97) |
| stage 1, mean rates | 9.4% / 27% (0.97) |
| stage 1, sampled | 8.4% / 26% (0.98) |
| stage 1 sampled + stage 2 | 5.3% / 26% (1.01) |
| stage 1 sampled vs dataset | 8.8% / 26% (1.01) |
| stage 1 sampled + stage 2 vs dataset | 7.1% / 26% (1.02) |

### Where the error is (stage 1 sampled vs with stage 2)

| device | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| EQE1_Top | 7830 | 8.5% / 25% (0.98) | 5.9% / 24% (1.01) | 10% |
| EQE1_Bottom | 7830 | 8.5% / 25% (0.98) | 4.9% / 26% (1.01) | 10% |
| QExa20 | 8190 | 8.0% / 26% (0.99) | 5.1% / 27% (1.02) | 11% |

| qubits | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–6 | 3888 | 4.1% / 14% (0.99) | 3.5% / 10% (1.00) | 0% |
| 6–11 | 7812 | 8.0% / 21% (0.99) | 5.5% / 21% (1.01) | 5% |
| 11–16 | 7065 | 10.7% / 30% (0.97) | 6.8% / 31% (1.02) | 16% |
| 16–21 | 5085 | 10.3% / 35% (0.98) | 5.3% / 35% (1.02) | 18% |

| 2q gates | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–20 | 4749 | 4.4% / 15% (1.02) | 3.7% / 12% (1.01) | 2% |
| 20–50 | 5091 | 8.4% / 25% (1.00) | 5.7% / 25% (1.03) | 9% |
| 50–100 | 5550 | 7.9% / 28% (0.97) | 5.1% / 29% (1.02) | 12% |
| 100–200 | 5742 | 10.5% / 25% (0.94) | 5.5% / 25% (0.99) | 10% |
| 200–500 | 2718 | 16.0% / 35% (0.97) | 12.4% / 36% (1.01) | 23% |

| SWAPs per 2q gate | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–0.25 | 3297 | 14.6% / 46% (1.15) | 14.5% / 45% (1.14) | 33% |
| 0.25–0.5 | 2949 | 6.2% / 19% (1.02) | 3.2% / 10% (1.01) | 2% |
| 0.5–1 | 10929 | 8.4% / 22% (0.98) | 5.5% / 19% (1.00) | 5% |
| 1–2 | 6588 | 7.7% / 26% (0.94) | 4.5% / 26% (1.01) | 10% |
| ≥2 | 87 | 23.1% / 41% (0.77) | 33.0% / 43% (0.67) | 43% |

| routing share of duration (1 − logical / final) | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| -inf–0.2 | 8931 | 6.6% / 29% (1.05) | 5.1% / 29% (1.03) | 13% |
| 0.2–0.4 | 6795 | 6.8% / 18% (0.96) | 5.3% / 18% (0.99) | 2% |
| 0.4–0.6 | 6750 | 9.8% / 25% (0.92) | 4.6% / 24% (1.00) | 9% |
| 0.6–0.8 | 1374 | 20.8% / 45% (1.16) | 19.7% / 45% (1.15) | 38% |

| family | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| ae | 1917 | 16.6% / 23% (1.17) | 2.5% / 8% (1.00) | 4% |
| bmw_quark_copula | 36 | 29.5% / 39% (0.70) | 35.4% / 44% (0.65) | 56% |
| bv | 18 | 4.0% / 5% (0.96) | 4.5% / 6% (0.96) | 0% |
| dj | 54 | 4.5% / 7% (0.96) | 3.3% / 6% (0.97) | 0% |
| draper_qft_adder | 18 | 9.6% / 20% (0.90) | 9.0% / 13% (0.93) | 0% |
| full_adder | 27 | 11.2% / 16% (0.89) | 13.9% / 18% (0.86) | 0% |
| hhl | 108 | 15.7% / 27% (0.84) | 10.4% / 21% (0.90) | 15% |
| modular_adder | 36 | 17.6% / 38% (1.18) | 23.5% / 48% (1.24) | 25% |
| qaoa | 5976 | 4.0% / 10% (0.98) | 3.2% / 8% (1.03) | 0% |
| qft | 1872 | 8.0% / 15% (0.92) | 3.2% / 8% (1.00) | 0% |
| qftentangled | 1989 | 9.3% / 17% (0.91) | 2.6% / 9% (1.00) | 1% |
| qnn | 45 | 1.3% / 3% (1.00) | 1.6% / 4% (0.99) | 0% |
| qpeexact | 45 | 13.5% / 32% (0.87) | 11.7% / 28% (0.88) | 20% |
| qpeinexact | 54 | 21.1% / 29% (0.79) | 17.7% / 25% (0.82) | 15% |
| randomcircuit | 3213 | 12.8% / 26% (0.87) | 13.1% / 28% (0.87) | 12% |
| vbe_ripple_carry_adder | 18 | 5.0% / 8% (0.95) | 9.1% / 12% (0.91) | 0% |
| vqe_real_amp | 2586 | 10.8% / 31% (1.07) | 11.1% / 31% (1.08) | 17% |
| vqe_su2 | 2649 | 12.0% / 51% (1.06) | 12.1% / 51% (1.07) | 30% |
| vqe_two_local | 3153 | 10.3% / 33% (1.08) | 10.5% / 33% (1.08) | 20% |

### Representation ceiling: true SWAPs + true waits vs Sabre's replayed schedule

Cases with unmatched gates: 72%; with negative waits: 99%.

| by | bin | n | oracle (SWAPs + waits) | replay | median extra error, points |
| --- | --- | --- | --- | --- | --- |
| unmatched share of gates | 0–1e-09 | 6672 | 3.5% / 7% (0.97) | 5.1% / 8% (0.95) | -1.5 |
| unmatched share of gates | 1e-09–0.05 | 2610 | 8.6% / 15% (0.91) | 4.6% / 10% (0.96) | +0.8 |
| unmatched share of gates | 0.05–0.2 | 7614 | 11.3% / 24% (0.97) | 7.4% / 17% (0.99) | +0.0 |
| unmatched share of gates | 0.2–0.5 | 6546 | 7.2% / 19% (1.05) | 1.1% / 5% (0.99) | +5.3 |
| unmatched share of gates | ≥0.5 | 408 | 8.9% / 13% (0.92) | 1.5% / 6% (1.00) | +8.2 |
| negative waits / duration | 0–1e-09 | 201 | 2.0% / 5% (0.99) | 2.0% / 5% (0.99) | -0.0 |
| negative waits / duration | 1e-09–0.02 | 519 | 5.1% / 14% (0.96) | 3.2% / 14% (0.99) | +0.0 |
| negative waits / duration | 0.02–0.05 | 1170 | 2.7% / 12% (0.98) | 1.6% / 12% (0.99) | +0.0 |
| negative waits / duration | 0.05–0.1 | 2193 | 3.1% / 8% (1.01) | 2.9% / 7% (0.99) | +0.0 |
| negative waits / duration | ≥0.1 | 19767 | 8.1% / 20% (0.98) | 4.2% / 12% (0.98) | +0.0 |

### Waits

|  | value |
| --- | --- |
| matched gates with a positive / negative wait | 27.2% / 37.8% |
| positive waits / duration (median over cases) | 32.0% |
| net waits on the critical path / duration: true / predicted (median) | 5.3% / 5.5% |
| negative waits / duration (median) | 43.0% |
| predicted SWAPs on that path / true | 0.99 |
| |wait error| on the critical path / all |wait error| | 27.1% (path holds 28% of the gates) |
| per-gate corr(pred, true wait): all / critical / off-path | 0.84 / 0.83 / 0.84 |
| true waits that are zero: predicted |wait| mean (µs) | 0.115 |
| SWAP mass on slots with no SWAP | 21% |

### Stage 2 by routing overhead (SWAPs per 2q gate)

| overhead | n | replay | replay + stage 2 | stage 1 sampled | + stage 2 | stage-2 factor (median) | true final / replay (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0–0.25 | 3297 | 0.8% / 2% (1.00) | 0.7% / 2% (1.00) | 14.6% / 46% (1.15) | 14.5% / 45% (1.14) | 1.000 | 1.004 |
| 0.25–0.5 | 2949 | 2.6% / 17% (1.00) | 1.0% / 3% (1.00) | 6.2% / 19% (1.02) | 3.2% / 10% (1.01) | 0.997 | 0.997 |
| 0.5–1 | 10929 | 4.0% / 13% (0.98) | 1.5% / 4% (1.00) | 8.4% / 22% (0.98) | 5.5% / 19% (1.00) | 1.013 | 1.021 |
| 1–2 | 6588 | 6.4% / 10% (0.94) | 1.3% / 4% (1.00) | 7.7% / 26% (0.94) | 4.5% / 26% (1.01) | 1.066 | 1.062 |
| ≥2 | 87 | 6.4% / 18% (1.06) | 2.2% / 6% (1.02) | 23.1% / 41% (0.77) | 33.0% / 43% (0.67) | 0.954 | 0.940 |

## loso:dense,complete (23448 routed / 1077 routing-free cases, pooled seeds)

### Error by stage

| what | med / p90 (bias) |
| --- | --- |
| Sabre final vs dataset label (label mismatch) | – |
| Sabre's routed schedule replayed (no optimisation) | 3.3% / 12% (1.01) |
|   + stage 2 (on the true schedule) | 2.3% / 6% (1.01) |
| true SWAPs + true waits (representation ceiling) | 4.0% / 13% (1.00) |
| true SWAPs, no waits | 20.3% / 40% (0.81) |
| predicted SWAPs + true waits | 4.5% / 13% (0.99) |
| true SWAPs + predicted waits | 17.9% / 37% (0.84) |
| stage 1, mean rates | 20.2% / 41% (0.81) |
| stage 1, sampled | 14.9% / 35% (0.87) |
| stage 1 sampled + stage 2 | 13.7% / 33% (0.87) |

### Where the error is (stage 1 sampled vs with stage 2)

| device | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| EQE1_Top | 7815 | 14.8% / 35% (0.87) | 13.6% / 33% (0.87) | 10% |
| EQE1_Bottom | 7815 | 15.2% / 35% (0.87) | 13.8% / 33% (0.87) | 11% |
| QExa20 | 7818 | 14.9% / 35% (0.87) | 13.8% / 33% (0.86) | 10% |

| qubits | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–6 | 3366 | 8.8% / 27% (1.08) | 5.0% / 13% (0.99) | 5% |
| 6–11 | 6372 | 7.1% / 17% (0.96) | 7.7% / 18% (0.93) | 0% |
| 11–16 | 6903 | 17.4% / 29% (0.83) | 16.9% / 28% (0.83) | 2% |
| 16–21 | 6807 | 29.3% / 41% (0.71) | 28.2% / 39% (0.72) | 30% |

| 2q gates | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–20 | 1512 | 4.2% / 15% (1.02) | 3.5% / 9% (1.00) | 0% |
| 20–50 | 5151 | 8.0% / 21% (0.99) | 6.0% / 15% (0.96) | 3% |
| 50–100 | 7416 | 11.9% / 26% (0.90) | 11.9% / 24% (0.88) | 1% |
| 100–200 | 7263 | 23.9% / 36% (0.76) | 23.2% / 35% (0.77) | 13% |
| 200–500 | 2106 | 35.8% / 44% (0.64) | 34.6% / 42% (0.65) | 56% |

| SWAPs per 2q gate | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| 0–0.25 | 2175 | 9.2% / 26% (1.09) | 4.2% / 11% (1.02) | 5% |
| 0.25–0.5 | 3882 | 5.7% / 18% (1.03) | 5.0% / 14% (0.98) | 2% |
| 0.5–1 | 7419 | 8.8% / 18% (0.92) | 9.8% / 20% (0.90) | 0% |
| 1–2 | 9903 | 27.0% / 39% (0.73) | 25.8% / 38% (0.74) | 21% |
| ≥2 | 69 | 42.0% / 46% (0.58) | 40.3% / 44% (0.60) | 94% |

| routing share of duration (1 − logical / final) | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| -inf–0.2 | 7881 | 5.5% / 19% (1.03) | 4.6% / 12% (0.98) | 2% |
| 0.2–0.4 | 6315 | 11.4% / 19% (0.89) | 12.3% / 20% (0.88) | 0% |
| 0.4–0.6 | 7641 | 26.0% / 35% (0.74) | 24.9% / 34% (0.75) | 10% |
| 0.6–0.8 | 1611 | 39.4% / 45% (0.61) | 37.9% / 43% (0.62) | 87% |

| family | n | pred1_mc: med / p90 (bias) | pred_mc: med / p90 (bias) | share in top-10% errors |
| --- | --- | --- | --- | --- |
| synth_complete | 11889 | 15.5% / 36% (0.87) | 13.9% / 34% (0.86) | 11% |
| synth_dense | 11559 | 14.4% / 34% (0.88) | 13.6% / 33% (0.87) | 9% |

### Representation ceiling: true SWAPs + true waits vs Sabre's replayed schedule

Cases with unmatched gates: 95%; with negative waits: 98%.

| by | bin | n | oracle (SWAPs + waits) | replay | median extra error, points |
| --- | --- | --- | --- | --- | --- |
| unmatched share of gates | 0–1e-09 | 1062 | 3.7% / 14% (1.01) | 3.7% / 16% (1.02) | +0.0 |
| unmatched share of gates | 1e-09–0.05 | 9108 | 4.0% / 11% (0.98) | 3.2% / 11% (1.00) | +0.0 |
| unmatched share of gates | 0.05–0.2 | 10623 | 3.6% / 11% (0.99) | 3.0% / 11% (1.01) | +0.1 |
| unmatched share of gates | 0.2–0.5 | 513 | 4.1% / 23% (1.03) | 4.2% / 24% (1.04) | +0.4 |
| unmatched share of gates | ≥0.5 | 2142 | 7.2% / 22% (1.07) | 6.0% / 16% (1.06) | +1.5 |
| negative waits / duration | 0–1e-09 | 570 | 4.0% / 15% (1.01) | 4.0% / 15% (1.01) | +0.0 |
| negative waits / duration | 1e-09–0.02 | 1323 | 3.5% / 16% (1.01) | 3.5% / 16% (1.01) | +0.0 |
| negative waits / duration | 0.02–0.05 | 1392 | 3.8% / 12% (1.00) | 3.5% / 12% (1.00) | +0.0 |
| negative waits / duration | 0.05–0.1 | 2943 | 3.9% / 12% (1.00) | 3.6% / 12% (1.00) | +0.0 |
| negative waits / duration | ≥0.1 | 17220 | 4.0% / 12% (0.99) | 3.2% / 12% (1.01) | +0.3 |

### Waits

|  | value |
| --- | --- |
| matched gates with a positive / negative wait | 31.3% / 14.7% |
| positive waits / duration (median over cases) | 53.8% |
| net waits on the critical path / duration: true / predicted (median) | 22.9% / 2.3% |
| negative waits / duration (median) | 20.1% |
| predicted SWAPs on that path / true | 0.85 |
| |wait error| on the critical path / all |wait error| | 31.0% (path holds 13% of the gates) |
| per-gate corr(pred, true wait): all / critical / off-path | 0.44 / 0.42 / 0.45 |
| true waits that are zero: predicted |wait| mean (µs) | 0.119 |
| SWAP mass on slots with no SWAP | 49% |

### Stage 2 by routing overhead (SWAPs per 2q gate)

| overhead | n | replay | replay + stage 2 | stage 1 sampled | + stage 2 | stage-2 factor (median) | true final / replay (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0–0.25 | 2175 | 6.8% / 22% (1.07) | 2.7% / 8% (1.00) | 9.2% / 26% (1.09) | 4.2% / 11% (1.02) | 0.942 | 0.937 |
| 0.25–0.5 | 3882 | 6.0% / 20% (1.06) | 2.5% / 7% (1.01) | 5.7% / 18% (1.03) | 5.0% / 14% (0.98) | 0.954 | 0.944 |
| 0.5–1 | 7419 | 3.3% / 10% (1.02) | 2.3% / 6% (1.01) | 8.8% / 18% (0.92) | 9.8% / 20% (0.90) | 0.993 | 0.983 |
| 1–2 | 9903 | 2.6% / 7% (0.99) | 2.1% / 6% (1.00) | 27.0% / 39% (0.73) | 25.8% / 38% (0.74) | 1.018 | 1.012 |
| ≥2 | 69 | 2.1% / 4% (0.98) | 2.4% / 6% (1.01) | 42.0% / 46% (0.58) | 40.3% / 44% (0.60) | 1.025 | 1.020 |

## Device choice by duration (circuits with ≥ 2 devices and at least one routed)

Fastest device picked by the prediction vs the true fastest (Sabre final; last column vs the dataset label). Right = picked device within 0.1% of the fastest (ties count as right). Regret = extra duration of the picked device, mean; random = mean regret of a random device. Pairs in right order: pairs more than 1% apart. Ratio error = median |log(pred_a/pred_b) − log(true_a/true_b)|.

| split | model | circuits × seeds | with a routed device | right device | regret | random regret | pairs in right order | ratio error | vs dataset: right / regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qaoa | pred1_mc | 8118 | 6714 | 36% | 4.77% | 4.38% | 51% | 6.9% | 37% / 5.39% |
| qaoa | pred_mc | 8118 | 6714 | 36% | 4.80% | 4.38% | 51% | 6.9% | 37% / 5.45% |
| variational | pred1_mc | 29784 | 10347 | 49% | 4.20% | 3.70% | 54% | 3.4% | 49% / 4.18% |
| variational | pred_mc | 29784 | 10347 | 48% | 4.19% | 3.70% | 54% | 3.4% | 49% / 4.18% |
| control | pred1_mc | 14787 | 8190 | 59% | 1.80% | 4.31% | 68% | 4.4% | 57% / 2.29% |
| control | pred_mc | 14787 | 8190 | 60% | 1.77% | 4.31% | 68% | 4.4% | 58% / 2.22% |
| loso:dense,complete | pred1_mc | 8175 | 7818 | 42% | 3.60% | 4.36% | 56% | 5.4% | – |
| loso:dense,complete | pred_mc | 8175 | 7818 | 42% | 3.66% | 4.36% | 55% | 5.4% | – |
