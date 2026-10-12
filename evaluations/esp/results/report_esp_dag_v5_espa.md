# Physical schedule on the circuit DAG vs the ESP head: v5_espa (seed 5)

Per-gate cost from the model's count head (measurements exact), ASAP on the logical gate DAG; no training.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| duration error, DAG | 0.7817 | 0.0725 | 0.0898 | 0.0996 |
| duration error, model head | 0.5597 | 0.0228 | 0.2556 | 0.0296 |
| R² log duration, DAG | -0.2182 | -0.2233 | 0.4015 | 0.5710 |
| R² log duration, model head | 0.2923 | 0.4748 | 0.0973 | 0.9855 |
| depth error, DAG | 0.8055 | 0.4995 | 0.5911 | 0.4951 |
| depth error, model head | 0.5036 | 0.1580 | 0.4297 | 0.0380 |
| R² log depth, DAG | -0.4813 | -0.5172 | 0.2476 | 0.6807 |
| R² log depth, model head | 0.6742 | 0.8396 | 0.6790 | 0.9958 |
| R² decay, DAG | -0.1602 | 0.5303 | 0.0591 | 0.2202 |
| R² decay, model head | 0.6713 | 0.5257 | 0.3705 | 0.9723 |
| ESP regret, DAG | 0.0066 | 0.0002 | 0.0227 | 0.0032 |
| right device, DAG | 0.5828 | 0.9737 | 0.3773 | 0.7244 |
| ESP regret, model head | 0.0055 | 0.0008 | 0.0164 | 0.0038 |
| right device, model head | 0.6003 | 0.9211 | 0.5375 | 0.8099 |
| ESP regret, true decay | 0.0004 | 0.0000 | 0.0060 | 0.0005 |
| right device, true decay | 0.8474 | 1.0000 | 0.7406 | 0.9004 |
| ESP regret, F only | 0.0068 | 0.0004 | 0.0349 | 0.0078 |
| right device, F only | 0.5164 | 0.9605 | 0.3003 | 0.5625 |
| n | 1500 | 760 | 1500 | 1500 |