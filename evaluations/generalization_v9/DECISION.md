# v9 decision — 2026-10-10 04:46

Rule (seed 5): keep an add-on if mean ΔR² on unseen families ≥ 0.005, no split < −0.03, Δcontrol ≥ −0.003, Δlocal tracking ≥ −0.03.

| run | LOFO qaoa | LOFO qnn | LOGO variational | control | local tracking |
| --- | --- | --- | --- | --- | --- |
| v8r | 0.743 | 0.948 | 0.919 | 0.996 | 0.47 |
| v8r_cz | 0.918 | 0.974 | 0.878 | 0.996 | 0.43 |
| v8s | 0.939 | 0.958 | 0.841 | 0.996 | 0.43 |
| v9 | 0.823 | 0.954 | 0.945 | 0.997 | 0.41 |

| add-on | test | verdict | numbers |
| --- | --- | --- | --- |
| cz-weighted count loss | v8r_cz vs v8r | drop | mean ΔR² unseen +0.053 (worst -0.041), Δcontrol -0.000, Δtracking -0.04 |
| routing-cost estimate | v8s vs v8r_cz | drop | mean ΔR² unseen -0.011 (worst -0.037), Δcontrol +0.000, Δtracking +0.00 |
| relative device features | v9 vs v8r | drop | mean ΔR² unseen +0.037 (worst +0.006), Δcontrol +0.000, Δtracking -0.06 |

**Not launched:** no add-on helps; v8r (3 seeds) stays the candidate.
