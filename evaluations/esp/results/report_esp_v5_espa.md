# ESP, duration and depth predicted by v5_espa

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.694 ± 0.128 | 0.941 ± 0.013 | 0.825 ± 0.070 | 0.996 ± 0.000 |
| R² log duration | 0.286 ± 0.023 | -1.748 ± 1.896 | 0.089 ± 0.054 | 0.989 ± 0.002 |
| duration error (median) | 0.521 ± 0.056 | 0.085 ± 0.052 | 0.240 ± 0.020 | 0.028 ± 0.001 |
| R² log depth | 0.282 ± 0.392 | 0.667 ± 0.206 | 0.590 ± 0.173 | 0.995 ± 0.001 |
| depth error (median) | 0.639 ± 0.116 | 0.246 ± 0.114 | 0.386 ± 0.021 | 0.039 ± 0.002 |
| R² ESP | 0.866 ± 0.051 | 0.738 ± 0.068 | 0.433 ± 0.013 | 0.984 ± 0.001 |
| R² ESP, baseline F | 0.153 ± 0.212 | 0.634 ± 0.021 | 0.641 ± 0.028 | 0.680 ± 0.018 |
| R² log ESP (ESP > 1e-3) | 0.473 ± 0.158 | 0.428 ± 0.214 | -8.005 ± 0.382 | 0.963 ± 0.002 |
| right device by ESP, model | 0.573 ± 0.035 | 0.850 ± 0.084 | 0.522 ± 0.013 | 0.825 ± 0.008 |
| ESP regret, model | 0.006 ± 0.001 | 0.004 ± 0.004 | 0.019 ± 0.001 | 0.003 ± 0.000 |
| right device by ESP, baseline F | 0.541 ± 0.044 | 0.868 ± 0.084 | 0.329 ± 0.033 | 0.579 ± 0.014 |
| ESP regret, baseline F | 0.007 ± 0.001 | 0.002 ± 0.002 | 0.044 ± 0.009 | 0.009 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |