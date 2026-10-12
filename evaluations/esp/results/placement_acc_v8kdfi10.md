# Hard placement vs the compiler's layout: v8kdfi10

Test circuits of each split, (logical qubit, device) pairs with a known compiler layout (seed 0, original calibration). Mean ± std over seeds 5–7. Random guess on a 20–27-qubit chip: exact ≈ 0.04–0.05.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| exact | 0.089 ± 0.005 | 0.127 ± 0.010 | 0.068 ± 0.004 | 0.157 ± 0.001 |
| argmax | 0.088 ± 0.004 | 0.131 ± 0.011 | 0.071 ± 0.004 | 0.159 ± 0.001 |
| p(true) | 0.075 ± 0.003 | 0.108 ± 0.010 | 0.065 ± 0.004 | 0.126 ± 0.002 |
| in region | 0.748 ± 0.006 | 0.882 ± 0.007 | 0.774 ± 0.005 | 0.932 ± 0.004 |
| hops | 2.62 ± 0.03 | 2.57 ± 0.10 | 2.88 ± 0.05 | 2.33 ± 0.06 |
| exact EQE1_Top | 0.070 ± 0.003 | 0.098 ± 0.025 | 0.060 ± 0.006 | 0.132 ± 0.020 |
| exact EQE1_Bottom | 0.085 ± 0.006 | 0.118 ± 0.023 | 0.069 ± 0.005 | 0.171 ± 0.020 |
| exact QExa20 | 0.113 ± 0.007 | 0.164 ± 0.041 | 0.075 ± 0.002 | 0.167 ± 0.012 |
