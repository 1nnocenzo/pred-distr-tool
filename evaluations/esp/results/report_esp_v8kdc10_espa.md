# ESP, duration and depth predicted by v8kdc10_espa

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.966 ± 0.019 | 0.847 ± 0.036 | 0.937 ± 0.013 | 0.994 ± 0.000 |
| R² log duration | 0.337 ± 0.040 | -9.612 ± 2.124 | 0.081 ± 0.219 | 0.979 ± 0.000 |
| duration error (median) | 0.568 ± 0.175 | 0.203 ± 0.040 | 0.252 ± 0.046 | 0.063 ± 0.001 |
| R² log depth | 0.744 ± 0.115 | 0.236 ± 0.178 | 0.694 ± 0.043 | 0.993 ± 0.001 |
| depth error (median) | 0.411 ± 0.093 | 0.584 ± 0.083 | 0.369 ± 0.026 | 0.060 ± 0.001 |
| R² ESP | 0.892 ± 0.016 | 0.286 ± 0.027 | 0.521 ± 0.054 | 0.977 ± 0.002 |
| R² ESP, baseline F | 0.736 ± 0.068 | 0.741 ± 0.041 | 0.424 ± 0.081 | 0.682 ± 0.003 |
| R² log ESP (ESP > 1e-3) | 0.730 ± 0.032 | -3.470 ± 0.654 | -6.601 ± 3.683 | 0.940 ± 0.003 |
| right device by ESP, model | 0.461 ± 0.021 | 0.461 ± 0.119 | 0.555 ± 0.053 | 0.769 ± 0.010 |
| ESP regret, model | 0.008 ± 0.001 | 0.023 ± 0.007 | 0.018 ± 0.003 | 0.005 ± 0.000 |
| right device by ESP, baseline F | 0.253 ± 0.040 | 0.604 ± 0.064 | 0.537 ± 0.102 | 0.565 ± 0.013 |
| ESP regret, baseline F | 0.011 ± 0.004 | 0.014 ± 0.002 | 0.020 ± 0.005 | 0.008 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |