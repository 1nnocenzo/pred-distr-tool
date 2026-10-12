# ESP, duration and depth predicted by v8kdt10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.948 ± 0.022 | 0.920 ± 0.024 | 0.925 ± 0.014 | 0.994 ± 0.001 |
| R² log duration | 0.756 ± 0.062 | -1.174 ± 1.823 | 0.743 ± 0.029 | 0.985 ± 0.004 |
| duration error (median) | 0.158 ± 0.011 | 0.089 ± 0.052 | 0.107 ± 0.001 | 0.045 ± 0.005 |
| R² log depth | 0.786 ± 0.086 | 0.323 ± 0.221 | 0.627 ± 0.048 | 0.988 ± 0.002 |
| depth error (median) | 0.406 ± 0.079 | 0.445 ± 0.169 | 0.528 ± 0.036 | 0.095 ± 0.012 |
| R² ESP | 0.965 ± 0.011 | 0.883 ± 0.072 | 0.878 ± 0.004 | 0.974 ± 0.004 |
| R² ESP, baseline F | 0.679 ± 0.063 | 0.654 ± 0.047 | 0.454 ± 0.093 | 0.668 ± 0.013 |
| R² log ESP (ESP > 1e-3) | 0.748 ± 0.013 | 0.803 ± 0.194 | -0.094 ± 0.379 | 0.924 ± 0.014 |
| right device by ESP, model | 0.490 ± 0.037 | 0.632 ± 0.042 | 0.567 ± 0.072 | 0.769 ± 0.027 |
| ESP regret, model | 0.006 ± 0.000 | 0.013 ± 0.003 | 0.019 ± 0.004 | 0.006 ± 0.001 |
| right device by ESP, baseline F | 0.398 ± 0.056 | 0.833 ± 0.056 | 0.454 ± 0.099 | 0.559 ± 0.011 |
| ESP regret, baseline F | 0.007 ± 0.000 | 0.004 ± 0.001 | 0.025 ± 0.006 | 0.007 ± 0.001 |
| seeds | 3 | 3 | 3 | 3 |