# Level-2 pipeline anatomy: what a calibration change moves

200 control test circuits (0 failed) × 3 devices × orig + local v8d variants [80, 84, 85, 89] × seeds [0, 1, 2]; Qiskit 2.3.1 preset level 2 (as the labels).

| | all | routing-free | routed |
| --- | --- | --- | --- |
| cases (circuit × device × variant × seed) | 7200 | 3168 | 4032 |
| same compiled structure as on the original calibration (up to qubit relabelling) | 84.9% | 92.9% | 78.6% |
| routing-free status unchanged | 100.0% | 100.0% | 100.0% |
| same total CZ / r / measure counts | 87.5% | 93.8% | 82.5% |
| mean change of the CZ count | +0.00 | +0.00 | +0.00 |
| Δlog F, same seed: compiler's error-aware step re-run on the circuit it scores (VF2Layout / VF2PostLayout) vs recompiled | corr 1.000, exact 88% | corr 1.000, exact 93% | corr 1.000, exact 81% |
| Δlog F, same seed: final circuit re-embedded by VF2PostLayout vs recompiled | corr 0.688, exact 32% | corr 0.853, exact 37% | corr 0.628, exact 23% |
| Δlog F, same seed: old compilation kept as is vs recompiled | corr 0.509, exact 29% | corr 0.471, exact 25% | corr 0.545, exact 34% |

Time per case, median: full compilation 4.8 ms, VF2PostLayout re-embedding 2.9 ms (mean 32.2 vs 7.1 ms).

Best of 3 seeds (as the labels): the winning seed changes in 17% of the (circuit, device, variant) cases; Δlog F of the best stage-emulated original compilations vs the best recompilation: corr 1.000 (1668 cases). Model local tracking on v8d variants: 0.74; compiler noise ceiling (best of 3 vs best of 2): 0.94.