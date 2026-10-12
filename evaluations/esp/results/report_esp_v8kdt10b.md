# ESP, duration and depth predicted by v8kdt10b

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.949 ± 0.022 | 0.873 ± 0.084 | 0.924 ± 0.013 | 0.996 ± 0.001 |
| R² log duration | 0.790 ± 0.084 | 0.671 ± 0.217 | 0.815 ± 0.036 | 0.992 ± 0.000 |
| duration error (median) | 0.136 ± 0.016 | 0.033 ± 0.017 | 0.091 ± 0.015 | 0.029 ± 0.001 |
| R² log depth | 0.810 ± 0.100 | 0.613 ± 0.025 | 0.621 ± 0.063 | 0.993 ± 0.000 |
| depth error (median) | 0.377 ± 0.095 | 0.245 ± 0.029 | 0.526 ± 0.069 | 0.060 ± 0.003 |
| R² ESP | 0.968 ± 0.015 | 0.962 ± 0.004 | 0.903 ± 0.002 | 0.982 ± 0.001 |
| R² ESP, baseline F | 0.679 ± 0.072 | 0.696 ± 0.098 | 0.486 ± 0.095 | 0.674 ± 0.011 |
| R² log ESP (ESP > 1e-3) | 0.734 ± 0.098 | 0.968 ± 0.006 | 0.372 ± 0.088 | 0.928 ± 0.006 |
| right device by ESP, model | 0.509 ± 0.020 | 0.744 ± 0.101 | 0.543 ± 0.074 | 0.793 ± 0.030 |
| ESP regret, model | 0.006 ± 0.000 | 0.008 ± 0.004 | 0.019 ± 0.004 | 0.004 ± 0.001 |
| right device by ESP, baseline F | 0.346 ± 0.071 | 0.854 ± 0.065 | 0.477 ± 0.062 | 0.532 ± 0.011 |
| ESP regret, baseline F | 0.007 ± 0.000 | 0.003 ± 0.001 | 0.024 ± 0.003 | 0.008 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |