# ESP, duration and depth predicted by v8kdg10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.943 ± 0.036 | 0.892 ± 0.053 | 0.917 ± 0.010 | 0.996 ± 0.000 |
| R² log duration | 0.753 ± 0.101 | 0.601 ± 0.272 | 0.814 ± 0.035 | 0.993 ± 0.000 |
| duration error (median) | 0.148 ± 0.013 | 0.040 ± 0.019 | 0.104 ± 0.033 | 0.022 ± 0.002 |
| R² log depth | 0.791 ± 0.084 | 0.510 ± 0.094 | 0.607 ± 0.003 | 0.993 ± 0.001 |
| depth error (median) | 0.408 ± 0.071 | 0.337 ± 0.059 | 0.559 ± 0.018 | 0.057 ± 0.001 |
| R² ESP | 0.958 ± 0.021 | 0.959 ± 0.005 | 0.903 ± 0.007 | 0.986 ± 0.001 |
| R² ESP, baseline F | 0.680 ± 0.101 | 0.687 ± 0.062 | 0.461 ± 0.125 | 0.679 ± 0.007 |
| R² log ESP (ESP > 1e-3) | 0.711 ± 0.102 | 0.918 ± 0.029 | 0.447 ± 0.105 | 0.940 ± 0.005 |
| right device by ESP, model | 0.500 ± 0.067 | 0.664 ± 0.037 | 0.535 ± 0.057 | 0.806 ± 0.005 |
| ESP regret, model | 0.006 ± 0.000 | 0.011 ± 0.002 | 0.020 ± 0.003 | 0.004 ± 0.000 |
| right device by ESP, baseline F | 0.382 ± 0.122 | 0.692 ± 0.107 | 0.491 ± 0.086 | 0.556 ± 0.012 |
| ESP regret, baseline F | 0.007 ± 0.002 | 0.008 ± 0.004 | 0.024 ± 0.004 | 0.009 ± 0.001 |
| seeds | 3 | 3 | 3 | 3 |