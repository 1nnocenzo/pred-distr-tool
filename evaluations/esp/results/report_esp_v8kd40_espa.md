# ESP, duration and depth predicted by v8kd40_espa

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.943 ± 0.055 | 0.878 ± 0.007 | 0.926 ± 0.004 | 0.994 ± 0.000 |
| R² log duration | 0.251 ± 0.118 | -7.782 ± 2.037 | -0.051 ± 0.378 | 0.984 ± 0.000 |
| duration error (median) | 0.664 ± 0.229 | 0.169 ± 0.009 | 0.280 ± 0.069 | 0.052 ± 0.002 |
| R² log depth | 0.752 ± 0.042 | 0.272 ± 0.174 | 0.677 ± 0.062 | 0.993 ± 0.000 |
| depth error (median) | 0.450 ± 0.040 | 0.571 ± 0.100 | 0.359 ± 0.009 | 0.055 ± 0.001 |
| R² ESP | 0.885 ± 0.012 | 0.443 ± 0.130 | 0.486 ± 0.052 | 0.980 ± 0.001 |
| R² ESP, baseline F | 0.675 ± 0.117 | 0.719 ± 0.004 | 0.457 ± 0.078 | 0.679 ± 0.006 |
| R² log ESP (ESP > 1e-3) | 0.699 ± 0.060 | -2.856 ± 2.329 | -8.745 ± 5.456 | 0.947 ± 0.003 |
| right device by ESP, model | 0.435 ± 0.056 | 0.562 ± 0.200 | 0.527 ± 0.045 | 0.791 ± 0.013 |
| ESP regret, model | 0.008 ± 0.001 | 0.019 ± 0.009 | 0.020 ± 0.002 | 0.004 ± 0.001 |
| right device by ESP, baseline F | 0.189 ± 0.076 | 0.694 ± 0.281 | 0.480 ± 0.069 | 0.565 ± 0.010 |
| ESP regret, baseline F | 0.014 ± 0.004 | 0.011 ± 0.013 | 0.024 ± 0.003 | 0.008 ± 0.000 |
| seeds | 3 | 3 | 3 | 3 |