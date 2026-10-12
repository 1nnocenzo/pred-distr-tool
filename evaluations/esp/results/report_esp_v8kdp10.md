# ESP, duration and depth predicted by v8kdp10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.948 ± 0.023 | 0.859 ± 0.077 | 0.905 ± 0.022 | 0.996 ± 0.000 |
| R² log duration | 0.805 ± 0.075 | 0.846 ± 0.030 | 0.779 ± 0.031 | 0.995 ± 0.000 |
| duration error (median) | 0.122 ± 0.011 | 0.023 ± 0.005 | 0.119 ± 0.027 | 0.018 ± 0.002 |
| R² log depth | 0.817 ± 0.091 | 0.599 ± 0.032 | 0.603 ± 0.029 | 0.995 ± 0.000 |
| depth error (median) | 0.365 ± 0.096 | 0.259 ± 0.039 | 0.547 ± 0.069 | 0.049 ± 0.000 |
| R² ESP | 0.962 ± 0.009 | 0.965 ± 0.002 | 0.904 ± 0.006 | 0.987 ± 0.001 |
| R² ESP, baseline F | 0.684 ± 0.070 | 0.709 ± 0.090 | 0.462 ± 0.183 | 0.682 ± 0.008 |
| R² log ESP (ESP > 1e-3) | 0.779 ± 0.088 | 0.929 ± 0.020 | 0.193 ± 0.175 | 0.949 ± 0.004 |
| right device by ESP, model | 0.532 ± 0.066 | 0.814 ± 0.032 | 0.484 ± 0.046 | 0.808 ± 0.006 |
| ESP regret, model | 0.005 ± 0.001 | 0.004 ± 0.002 | 0.023 ± 0.003 | 0.004 ± 0.000 |
| right device by ESP, baseline F | 0.415 ± 0.087 | 0.797 ± 0.087 | 0.465 ± 0.126 | 0.559 ± 0.022 |
| ESP regret, baseline F | 0.007 ± 0.000 | 0.004 ± 0.002 | 0.022 ± 0.006 | 0.008 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |