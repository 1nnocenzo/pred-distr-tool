# Hard placement vs the compiler's layout: v8kdf10

Test circuits of each split, (logical qubit, device) pairs with a known compiler layout (seed 0, original calibration). Mean ± std over seeds 5–7. Random guess on a 20–27-qubit chip: exact ≈ 0.04–0.05.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| exact | 0.089 ± 0.008 | 0.139 ± 0.013 | 0.068 ± 0.004 | 0.158 ± 0.002 |
| argmax | 0.091 ± 0.006 | 0.142 ± 0.015 | 0.070 ± 0.004 | 0.160 ± 0.003 |
| p(true) | 0.077 ± 0.003 | 0.110 ± 0.011 | 0.064 ± 0.003 | 0.127 ± 0.002 |
| in region | 0.752 ± 0.011 | 0.881 ± 0.005 | 0.771 ± 0.004 | 0.936 ± 0.005 |
| hops | 2.61 ± 0.05 | 2.49 ± 0.13 | 2.91 ± 0.04 | 2.32 ± 0.06 |
| exact EQE1_Top | 0.070 ± 0.010 | 0.102 ± 0.032 | 0.058 ± 0.004 | 0.133 ± 0.019 |
| exact EQE1_Bottom | 0.091 ± 0.007 | 0.133 ± 0.042 | 0.070 ± 0.006 | 0.171 ± 0.023 |
| exact QExa20 | 0.107 ± 0.011 | 0.183 ± 0.032 | 0.077 ± 0.005 | 0.170 ± 0.014 |
