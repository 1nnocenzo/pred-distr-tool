# ESP, duration and depth predicted by v8kdfi10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.964 ± 0.011 | 0.904 ± 0.006 | 0.954 ± 0.005 | 0.994 ± 0.000 |
| R² log duration | 0.864 ± 0.049 | 0.412 ± 0.128 | 0.747 ± 0.073 | 0.988 ± 0.001 |
| duration error (median) | 0.129 ± 0.027 | 0.047 ± 0.006 | 0.095 ± 0.026 | 0.023 ± 0.004 |
| R² log depth | 0.802 ± 0.060 | 0.057 ± 0.058 | 0.666 ± 0.050 | 0.991 ± 0.001 |
| depth error (median) | 0.378 ± 0.081 | 0.561 ± 0.090 | 0.412 ± 0.048 | 0.070 ± 0.008 |
| R² ESP | 0.961 ± 0.008 | 0.937 ± 0.012 | 0.911 ± 0.009 | 0.986 ± 0.001 |
| R² ESP, baseline F | 0.722 ± 0.042 | 0.681 ± 0.005 | 0.575 ± 0.051 | 0.674 ± 0.003 |
| R² log ESP (ESP > 1e-3) | 0.763 ± 0.060 | 0.908 ± 0.027 | 0.610 ± 0.102 | 0.938 ± 0.003 |
| right device by ESP, model | 0.419 ± 0.018 | 0.740 ± 0.048 | 0.566 ± 0.037 | 0.780 ± 0.009 |
| ESP regret, model | 0.006 ± 0.000 | 0.009 ± 0.003 | 0.018 ± 0.002 | 0.004 ± 0.000 |
| right device by ESP, baseline F | 0.276 ± 0.020 | 0.862 ± 0.090 | 0.508 ± 0.055 | 0.566 ± 0.018 |
| ESP regret, baseline F | 0.009 ± 0.002 | 0.004 ± 0.003 | 0.019 ± 0.003 | 0.008 ± 0.001 |
| seeds | 3 | 3 | 3 | 3 |