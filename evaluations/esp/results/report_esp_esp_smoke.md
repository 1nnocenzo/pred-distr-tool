# ESP, duration and depth predicted by esp_smoke

Test circuits of each split with labels from the stored seed-0 compilations (MQT estimated_success_probability, ASAP schedule; QExa20 T1/T2 unit fix). Mean ± std over the available seeds; error = median |pred/true − 1|.

| | control |
| --- | --- |
| R² F | 0.988 |
| R² log duration | 0.911 |
| duration error (median) | 0.115 |
| R² log depth | 0.486 |
| depth error (median) | 0.792 |
| R² ESP | 0.767 |
| R² ESP, baseline F | 0.593 |
| R² log ESP (ESP > 1e-3) | 0.629 |
| right device by ESP, model | 0.765 |
| ESP regret, model | 0.005 |
| right device by ESP, baseline F | 0.516 |
| ESP regret, baseline F | 0.015 |
| seeds | 1 |