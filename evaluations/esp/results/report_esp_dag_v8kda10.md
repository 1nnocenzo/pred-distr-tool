# Physical schedule on the circuit DAG vs the ESP head: v8kda10 (seed 5)

Per-gate cost from the model's count head (measurements exact), ASAP on the logical gate DAG; no training.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| duration error, DAG | 0.2736 | 0.0364 | 0.0547 | 0.0845 |
| duration error, model head | 0.8055 | 0.1960 | 0.3424 | 0.0575 |
| R² log duration, DAG | 0.8813 | 0.6815 | 0.7042 | 0.9168 |
| R² log duration, model head | 0.2634 | -9.7208 | -0.2141 | 0.9817 |
| depth error, DAG | 0.3533 | 0.4293 | 0.4341 | 0.4302 |
| depth error, model head | 0.2990 | 0.3156 | 0.3845 | 0.0581 |
| R² log depth, DAG | 0.7671 | 0.0455 | 0.6806 | 0.8044 |
| R² log depth, model head | 0.8383 | 0.7099 | 0.7201 | 0.9936 |
| R² decay, DAG | 0.7237 | 0.8925 | 0.2459 | 0.7153 |
| R² decay, model head | 0.8917 | -6.0926 | -0.1534 | 0.9684 |
| ESP regret, DAG | 0.0072 | 0.0144 | 0.0153 | 0.0029 |
| right device, DAG | 0.4500 | 0.6171 | 0.5804 | 0.7555 |
| ESP regret, model head | 0.0066 | 0.0187 | 0.0154 | 0.0039 |
| right device, model head | 0.4546 | 0.6092 | 0.5784 | 0.7717 |
| ESP regret, true decay | 0.0002 | 0.0000 | 0.0010 | 0.0003 |
| right device, true decay | 0.8070 | 0.9974 | 0.8479 | 0.9088 |
| ESP regret, F only | 0.0142 | 0.0107 | 0.0182 | 0.0074 |
| right device, F only | 0.2204 | 0.6882 | 0.5623 | 0.5640 |
| n | 1500 | 760 | 1500 | 1500 |