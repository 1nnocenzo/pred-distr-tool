# ESP, duration and depth predicted by v8a_espa

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.891 ± 0.053 | 0.773 ± 0.064 | 0.821 ± 0.036 | 0.996 ± 0.000 |
| R² log duration | 0.199 ± 0.107 | -3.687 ± 1.837 | 0.054 ± 0.086 | 0.985 ± 0.002 |
| duration error (median) | 0.588 ± 0.243 | 0.083 ± 0.012 | 0.276 ± 0.038 | 0.031 ± 0.001 |
| R² log depth | 0.781 ± 0.099 | 0.313 ± 0.163 | 0.646 ± 0.025 | 0.995 ± 0.000 |
| depth error (median) | 0.374 ± 0.051 | 0.304 ± 0.044 | 0.356 ± 0.067 | 0.040 ± 0.000 |
| R² ESP | 0.805 ± 0.035 | 0.389 ± 0.051 | 0.413 ± 0.039 | 0.971 ± 0.009 |
| R² ESP, baseline F | 0.535 ± 0.135 | 0.742 ± 0.036 | 0.564 ± 0.062 | 0.675 ± 0.012 |
| R² log ESP (ESP > 1e-3) | 0.651 ± 0.055 | -0.664 ± 0.326 | -7.260 ± 1.317 | 0.958 ± 0.000 |
| right device by ESP, model | 0.477 ± 0.062 | 0.546 ± 0.082 | 0.563 ± 0.024 | 0.826 ± 0.010 |
| ESP regret, model | 0.006 ± 0.001 | 0.017 ± 0.004 | 0.017 ± 0.001 | 0.003 ± 0.001 |
| right device by ESP, baseline F | 0.380 ± 0.032 | 0.663 ± 0.168 | 0.536 ± 0.055 | 0.549 ± 0.009 |
| ESP regret, baseline F | 0.011 ± 0.001 | 0.011 ± 0.007 | 0.024 ± 0.004 | 0.010 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |