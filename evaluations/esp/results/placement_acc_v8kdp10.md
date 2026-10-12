# Hard placement vs the compiler's layout: v8kdp10

Test circuits of each split, (logical qubit, device) pairs with a known compiler layout (seed 0, original calibration). Mean ± std over seeds 5–7. Random guess on a 20–27-qubit chip: exact ≈ 0.04–0.05.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| exact | 0.089 ± 0.004 | 0.167 ± 0.018 | 0.067 ± 0.002 | 0.240 ± 0.011 |
| argmax | 0.092 ± 0.003 | 0.177 ± 0.013 | 0.078 ± 0.004 | 0.234 ± 0.009 |
| p(true) | 0.077 ± 0.001 | 0.112 ± 0.006 | 0.069 ± 0.004 | 0.152 ± 0.008 |
| in region | 0.760 ± 0.008 | 0.864 ± 0.016 | 0.749 ± 0.016 | 0.897 ± 0.003 |
| hops | 2.60 ± 0.03 | 2.43 ± 0.14 | 2.92 ± 0.01 | 1.99 ± 0.06 |
| exact EQE1_Top | 0.067 ± 0.005 | 0.141 ± 0.017 | 0.058 ± 0.006 | 0.237 ± 0.011 |
| exact EQE1_Bottom | 0.091 ± 0.007 | 0.183 ± 0.029 | 0.067 ± 0.002 | 0.248 ± 0.010 |
| exact QExa20 | 0.110 ± 0.012 | 0.176 ± 0.016 | 0.077 ± 0.002 | 0.235 ± 0.013 |
