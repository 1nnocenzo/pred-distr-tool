# ESP, duration and depth predicted by v8kde10

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| R² F | 0.976 ± 0.010 | 0.848 ± 0.014 | 0.942 ± 0.007 | 0.994 ± 0.000 |
| R² log duration | 0.336 ± 0.100 | -9.172 ± 2.755 | -0.502 ± 0.407 | 0.980 ± 0.003 |
| duration error (median) | 0.657 ± 0.276 | 0.186 ± 0.076 | 0.366 ± 0.146 | 0.057 ± 0.003 |
| R² log depth | 0.832 ± 0.043 | 0.054 ± 0.189 | 0.293 ± 0.126 | 0.993 ± 0.001 |
| depth error (median) | 0.306 ± 0.071 | 0.737 ± 0.101 | 0.961 ± 0.161 | 0.061 ± 0.007 |
| R² ESP | 0.902 ± 0.022 | 0.361 ± 0.101 | 0.448 ± 0.062 | 0.978 ± 0.000 |
| R² ESP, baseline F | 0.759 ± 0.036 | 0.753 ± 0.007 | 0.477 ± 0.107 | 0.676 ± 0.010 |
| R² log ESP (ESP > 1e-3) | 0.721 ± 0.029 | -3.095 ± 0.977 | -13.119 ± 4.038 | 0.937 ± 0.005 |
| right device by ESP, model | 0.459 ± 0.025 | 0.566 ± 0.182 | 0.551 ± 0.059 | 0.771 ± 0.005 |
| ESP regret, model | 0.007 ± 0.001 | 0.018 ± 0.010 | 0.018 ± 0.003 | 0.005 ± 0.000 |
| right device by ESP, baseline F | 0.222 ± 0.021 | 0.757 ± 0.150 | 0.552 ± 0.057 | 0.576 ± 0.010 |
| ESP regret, baseline F | 0.012 ± 0.003 | 0.008 ± 0.006 | 0.019 ± 0.003 | 0.008 ± 0.001 |
| seeds | 3 | 3 | 3 | 3 |