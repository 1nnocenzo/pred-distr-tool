# ESP, duration and depth predicted by v8kdf10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.946 ± 0.028 | 0.871 ± 0.075 | 0.918 ± 0.015 | 0.996 ± 0.001 |
| R² log duration | 0.780 ± 0.083 | 0.762 ± 0.057 | 0.789 ± 0.041 | 0.994 ± 0.000 |
| duration error (median) | 0.132 ± 0.008 | 0.031 ± 0.006 | 0.118 ± 0.028 | 0.019 ± 0.002 |
| R² log depth | 0.815 ± 0.087 | 0.594 ± 0.061 | 0.596 ± 0.036 | 0.994 ± 0.000 |
| depth error (median) | 0.371 ± 0.097 | 0.272 ± 0.039 | 0.553 ± 0.071 | 0.053 ± 0.000 |
| R² ESP | 0.958 ± 0.015 | 0.958 ± 0.010 | 0.911 ± 0.006 | 0.987 ± 0.001 |
| R² ESP, baseline F | 0.686 ± 0.083 | 0.705 ± 0.088 | 0.455 ± 0.115 | 0.676 ± 0.008 |
| R² log ESP (ESP > 1e-3) | 0.754 ± 0.087 | 0.919 ± 0.026 | 0.389 ± 0.157 | 0.947 ± 0.001 |
| right device by ESP, model | 0.505 ± 0.058 | 0.703 ± 0.011 | 0.546 ± 0.069 | 0.798 ± 0.014 |
| ESP regret, model | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.019 ± 0.004 | 0.004 ± 0.000 |
| right device by ESP, baseline F | 0.372 ± 0.069 | 0.807 ± 0.056 | 0.474 ± 0.126 | 0.544 ± 0.022 |
| ESP regret, baseline F | 0.007 ± 0.000 | 0.004 ± 0.001 | 0.024 ± 0.007 | 0.008 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |