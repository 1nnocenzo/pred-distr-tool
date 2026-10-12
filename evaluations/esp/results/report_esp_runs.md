# Compiled 1-qubit count vs runs of 1-qubit gates (routing-free circuits on EQE1_Top, no model)

8294 circuits. Rule fitted on all families but the held-out one (least squares, no intercept).

| held-out family | n | true r / runs (median) | true r / 1q gates (median) | error, runs rule | error, naive rule |
| --- | --- | --- | --- | --- | --- |
| ae | 40 | 1.75 | 0.97 | 1.24 | 1.55 |
| iqpe | 570 | 20.54 | 3.40 | 0.75 | 0.35 |
| qaoa | 468 | 4.50 | 2.80 | 2.05 | 0.86 |
| qft | 40 | 6.00 | 4.38 | 0.37 | 0.50 |
| qftentangled | 40 | 3.17 | 2.45 | 1.03 | 0.71 |
| qnn | 760 | 6.06 | 1.78 | 0.37 | 0.50 |
| randomcircuit | 133 | 3.00 | 1.83 | 0.83 | 0.77 |
| vqe_real_amp | 1920 | 3.90 | 3.15 | 0.50 | 0.12 |
| vqe_su2 | 1920 | 6.45 | 2.90 | 0.36 | 0.18 |
| vqe_two_local | 2293 | 2.64 | 1.50 | 0.44 | 0.17 |