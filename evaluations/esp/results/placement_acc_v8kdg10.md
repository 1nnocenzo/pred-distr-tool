# Hard placement vs the compiler's layout: v8kdg10

Test circuits of each split, (logical qubit, device) pairs with a known compiler layout (seed 0, original calibration). Mean ± std over seeds 5–7. Random guess on a 20–27-qubit chip: exact ≈ 0.04–0.05.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| exact | 0.091 ± 0.004 | 0.130 ± 0.006 | 0.069 ± 0.004 | 0.157 ± 0.004 |
| argmax | 0.090 ± 0.004 | 0.132 ± 0.010 | 0.071 ± 0.004 | 0.160 ± 0.005 |
| p(true) | 0.076 ± 0.003 | 0.107 ± 0.009 | 0.064 ± 0.002 | 0.125 ± 0.004 |
| in region | 0.753 ± 0.017 | 0.877 ± 0.011 | 0.771 ± 0.007 | 0.932 ± 0.004 |
| hops | 2.60 ± 0.07 | 2.53 ± 0.13 | 2.90 ± 0.04 | 2.33 ± 0.06 |
| exact EQE1_Top | 0.069 ± 0.006 | 0.103 ± 0.032 | 0.059 ± 0.005 | 0.129 ± 0.018 |
| exact EQE1_Bottom | 0.090 ± 0.011 | 0.118 ± 0.024 | 0.071 ± 0.004 | 0.170 ± 0.023 |
| exact QExa20 | 0.115 ± 0.006 | 0.168 ± 0.039 | 0.077 ± 0.005 | 0.171 ± 0.016 |
