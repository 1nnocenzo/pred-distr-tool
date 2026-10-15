# Fast path for routing-free circuits: validation against the compiler

240 test circuits of the four splits (0 failed), 3 devices, compiler seeds ['0', '1', '2']; original calibration + local v8d variants aug 80-99. Fast path = one reference compilation per (circuit, device, seed) on the original calibration, then per calibration only the VF2Layout score over the cached embeddings (fastpath.py). Truth = full level-2 recompilation, MQT F / ESP (as the labels). Ties in the score: mean over the tied placements.

## Switch (VF2 subgraph test, no compiler) vs the compiler's VF2Layout decision

| pairs | routing-free (compiler) | switch says free | precision | recall | time, median |
| --- | --- | --- | --- | --- | --- |
| 720 | 43.1% | 43.1% | 1.000 | 1.000 | 0.220 ms |

## Per compiler seed, ties averaged (fast path vs the recompilation with the same seed)

|Δ log F| and |Δ log ESP|: median / p90. Tracking = corr of Δ log F (variant − original) fast vs true, local variants only. Same layout = the compiler's placement reproduced exactly (first tied embedding). Ties = share of cases with several placements tied for the best VF2 score.

| group | cases | abs log F error | exact (< 1e-6) | abs log ESP error | tracking | same layout | ties |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 19530 | 1.4e-03 / 2.2e-02 | 19.2% | 2.3e-03 / 3.5e-02 | 0.940 | 36.6% | 68.6% |
| LOFO qaoa | 1512 | 1.6e-03 / 1.2e-02 | 19.8% | 2.4e-03 / 1.7e-02 | 0.976 | 86.7% | 87.5% |
| LOFO qnn | 11151 | 2.0e-03 / 2.4e-02 | 12.3% | 3.6e-03 / 4.3e-02 | 0.925 | 28.0% | 63.8% |
| LOGO variational | 6300 | 6.2e-04 / 1.7e-02 | 29.5% | 7.7e-04 / 2.7e-02 | 0.943 | 35.7% | 74.5% |
| control | 567 | 2.2e-03 / 4.4e-02 | 37.6% | 3.3e-03 / 8.7e-02 | 0.992 | 81.5% | 45.5% |
| EQE1_Top | 6615 | 1.4e-03 / 2.8e-02 | 17.3% | 2.4e-03 / 5.7e-02 | 0.942 | 34.1% | 59.7% |
| EQE1_Bottom | 6615 | 8.0e-04 / 7.9e-03 | 20.1% | 1.2e-03 / 2.4e-02 | 0.897 | 39.0% | 62.6% |
| QExa20 | 6300 | 3.9e-03 / 2.4e-02 | 20.1% | 4.7e-03 / 3.7e-02 | 0.963 | 36.6% | 84.0% |
| original calibration | 930 | 1.1e-03 / 1.9e-02 | 18.1% | 1.8e-03 / 2.4e-02 | nan | 37.1% | 68.1% |
| local variants | 18600 | 1.5e-03 / 2.2e-02 | 19.2% | 2.3e-03 / 3.6e-02 | 0.940 | 36.5% | 68.6% |

## Per compiler seed, ties broken by the first embedding found (fast path vs the recompilation with the same seed)

|Δ log F| and |Δ log ESP|: median / p90. Tracking = corr of Δ log F (variant − original) fast vs true, local variants only. Same layout = the compiler's placement reproduced exactly (first tied embedding). Ties = share of cases with several placements tied for the best VF2 score.

| group | cases | abs log F error | exact (< 1e-6) | abs log ESP error | tracking | same layout | ties |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 19530 | 1.0e-03 / 2.7e-02 | 37.2% | 1.7e-03 / 4.6e-02 | 0.937 | 36.6% | 68.6% |
| LOFO qaoa | 1512 | 9.4e-16 / 5.0e-03 | 66.7% | 8.9e-16 / 7.8e-03 | 0.989 | 86.7% | 87.5% |
| LOFO qnn | 11151 | 2.3e-03 / 3.3e-02 | 25.6% | 4.3e-03 / 5.4e-02 | 0.916 | 28.0% | 63.8% |
| LOGO variational | 6300 | 1.4e-04 / 2.2e-02 | 48.4% | 1.4e-04 / 3.2e-02 | 0.942 | 35.7% | 74.5% |
| control | 567 | 2.0e-15 / 1.4e-02 | 61.9% | 7.1e-15 / 6.9e-02 | 0.992 | 81.5% | 45.5% |
| EQE1_Top | 6615 | 1.2e-03 / 2.7e-02 | 33.9% | 2.3e-03 / 5.8e-02 | 0.942 | 34.1% | 59.7% |
| EQE1_Bottom | 6615 | 5.4e-04 / 8.5e-03 | 38.9% | 8.6e-04 / 2.4e-02 | 0.896 | 39.0% | 62.6% |
| QExa20 | 6300 | 2.5e-03 / 3.7e-02 | 38.9% | 3.0e-03 / 5.3e-02 | 0.956 | 36.6% | 84.0% |
| original calibration | 930 | 7.2e-04 / 2.1e-02 | 42.6% | 1.4e-03 / 2.6e-02 | nan | 37.1% | 68.1% |
| local variants | 18600 | 1.0e-03 / 2.8e-02 | 36.9% | 1.8e-03 / 4.7e-02 | 0.937 | 36.5% | 68.6% |


## Best of seeds ['0', '1', '2'] (as the recompiled labels)

| cases | abs log F error, median / p90 | abs log ESP error, median / p90 | tracking (corr Δ log F, local variants) |
| --- | --- | --- | --- |
| 6510 | 1.2e-03 / 2.7e-02 | 2.0e-03 / 4.4e-02 | 0.939 |

Model for reference (v8kdc10, calibration test): local tracking 0.66 ± 0.02 on all circuits; 0.83 on routing-free cases in the tracking anatomy.

## Time per (circuit, device), routing-free

| step | median | p90 | max |
| --- | --- | --- | --- |
| reference compilation (once per seed) | 2.59 ms | 4.12 ms | 24.9 ms |
| embedding enumeration (once) | 23.63 ms | 82.19 ms | 142.5 ms |
| fast path per calibration (score + F + ESP) | 0.88 ms | 2.91 ms | 12.5 ms |
| full recompilation per calibration | 2.44 ms | 4.15 ms | 195.9 ms |

Cached embeddings per structure: median 7628, max 20000 (cap 20000; 25.5% at the cap).