# ESP, duration and depth predicted by v8kdl40_espa

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.965 ± 0.027 | 0.870 ± 0.012 | 0.929 ± 0.008 | 0.994 ± 0.000 |
| R² log duration | 0.307 ± 0.079 | -8.776 ± 2.090 | -0.279 ± 0.257 | 0.983 ± 0.001 |
| duration error (median) | 0.677 ± 0.198 | 0.197 ± 0.012 | 0.346 ± 0.070 | 0.052 ± 0.002 |
| R² log depth | 0.742 ± 0.087 | 0.580 ± 0.252 | 0.659 ± 0.050 | 0.993 ± 0.000 |
| depth error (median) | 0.429 ± 0.080 | 0.352 ± 0.162 | 0.376 ± 0.030 | 0.052 ± 0.003 |
| R² ESP | 0.890 ± 0.003 | 0.324 ± 0.144 | 0.444 ± 0.042 | 0.980 ± 0.001 |
| R² ESP, baseline F | 0.740 ± 0.077 | 0.731 ± 0.015 | 0.449 ± 0.107 | 0.676 ± 0.006 |
| R² log ESP (ESP > 1e-3) | 0.707 ± 0.022 | -2.602 ± 1.621 | -9.223 ± 3.612 | 0.949 ± 0.001 |
| right device by ESP, model | 0.489 ± 0.038 | 0.508 ± 0.118 | 0.541 ± 0.045 | 0.796 ± 0.008 |
| ESP regret, model | 0.007 ± 0.001 | 0.020 ± 0.005 | 0.019 ± 0.002 | 0.004 ± 0.000 |
| right device by ESP, baseline F | 0.263 ± 0.031 | 0.681 ± 0.130 | 0.468 ± 0.111 | 0.575 ± 0.013 |
| ESP regret, baseline F | 0.009 ± 0.002 | 0.010 ± 0.005 | 0.025 ± 0.007 | 0.007 ± 0.001 |
| seeds | 3 | 3 | 3 | 3 |