# ESP, duration and depth predicted by v8kda10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.962 ± 0.029 | 0.848 ± 0.020 | 0.938 ± 0.006 | 0.994 ± 0.000 |
| R² log duration | 0.340 ± 0.082 | -7.624 ± 1.536 | -0.191 ± 0.402 | 0.982 ± 0.000 |
| duration error (median) | 0.554 ± 0.185 | 0.178 ± 0.021 | 0.341 ± 0.115 | 0.059 ± 0.001 |
| R² log depth | 0.731 ± 0.092 | 0.322 ± 0.393 | 0.676 ± 0.025 | 0.993 ± 0.001 |
| depth error (median) | 0.407 ± 0.077 | 0.527 ± 0.227 | 0.387 ± 0.016 | 0.063 ± 0.004 |
| R² ESP | 0.885 ± 0.018 | 0.403 ± 0.101 | 0.460 ± 0.067 | 0.978 ± 0.002 |
| R² ESP, baseline F | 0.726 ± 0.080 | 0.754 ± 0.018 | 0.471 ± 0.131 | 0.675 ± 0.003 |
| R² log ESP (ESP > 1e-3) | 0.663 ± 0.080 | -2.504 ± 1.746 | -8.191 ± 4.147 | 0.938 ± 0.002 |
| right device by ESP, model | 0.433 ± 0.027 | 0.528 ± 0.126 | 0.553 ± 0.073 | 0.770 ± 0.006 |
| ESP regret, model | 0.008 ± 0.001 | 0.020 ± 0.003 | 0.018 ± 0.004 | 0.005 ± 0.000 |
| right device by ESP, baseline F | 0.237 ± 0.033 | 0.809 ± 0.086 | 0.502 ± 0.140 | 0.554 ± 0.019 |
| ESP regret, baseline F | 0.012 ± 0.002 | 0.006 ± 0.004 | 0.024 ± 0.009 | 0.008 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |