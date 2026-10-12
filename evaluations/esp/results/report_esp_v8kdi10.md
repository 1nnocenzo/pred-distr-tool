# ESP, duration and depth predicted by v8kdi10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.926 ± 0.043 | 0.909 ± 0.013 | 0.952 ± 0.003 | 0.993 ± 0.001 |
| R² log duration | 0.839 ± 0.046 | 0.422 ± 0.505 | 0.834 ± 0.020 | 0.991 ± 0.000 |
| duration error (median) | 0.356 ± 0.054 | 0.031 ± 0.014 | 0.064 ± 0.002 | 0.023 ± 0.000 |
| R² log depth | 0.720 ± 0.086 | 0.687 ± 0.197 | 0.662 ± 0.017 | 0.992 ± 0.000 |
| depth error (median) | 0.410 ± 0.082 | 0.235 ± 0.102 | 0.368 ± 0.016 | 0.066 ± 0.001 |
| R² ESP | 0.925 ± 0.033 | 0.968 ± 0.011 | 0.896 ± 0.002 | 0.985 ± 0.001 |
| R² ESP, baseline F | 0.546 ± 0.083 | 0.678 ± 0.020 | 0.500 ± 0.054 | 0.659 ± 0.022 |
| R² log ESP (ESP > 1e-3) | 0.636 ± 0.150 | 0.900 ± 0.061 | 0.531 ± 0.056 | 0.950 ± 0.008 |
| right device by ESP, model | 0.452 ± 0.045 | 0.674 ± 0.126 | 0.568 ± 0.081 | 0.799 ± 0.011 |
| ESP regret, model | 0.008 ± 0.000 | 0.012 ± 0.006 | 0.018 ± 0.004 | 0.004 ± 0.000 |
| right device by ESP, baseline F | 0.088 ± 0.011 | 0.726 ± 0.124 | 0.555 ± 0.069 | 0.556 ± 0.017 |
| ESP regret, baseline F | 0.038 ± 0.001 | 0.008 ± 0.005 | 0.019 ± 0.004 | 0.008 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |