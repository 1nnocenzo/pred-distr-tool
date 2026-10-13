# Duration bounds for a scheduler: conformal factors from the control split

Factors calibrated on the control split's test circuits (families seen in training; cross-fitted halves for the control row), separately for routing-free and routed circuits, then applied to the models of the other splits (same seed). Seeds pooled. Coverage = share of cases with the true duration inside; width = bound / predicted duration (median); error after bias fix = median |pred × median ratio / true − 1|.

## v8kdfi10

| split | group | cases | coverage, 90% upper bound | width (×pred) | coverage, 95% upper bound | width | coverage, 80% interval | interval [lo, hi] (×pred) | error, raw | error after bias fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOFO qaoa | routed | 20142 | 0.348 | 1.09 | 0.439 | 1.13 | 0.296 | [0.90, 1.09] | 0.153 | 0.155 |
| LOFO qaoa | routing-free | 4212 | 0.380 | 1.02 | 0.436 | 1.03 | 0.205 | [0.97, 1.02] | 0.054 | 0.058 |
| LOFO qnn | routed | 120 | 0.992 | 1.09 | 1.000 | 1.13 | 0.992 | [0.90, 1.09] | 0.032 | 0.042 |
| LOFO qnn | routing-free | 6720 | 0.996 | 1.02 | 0.999 | 1.03 | 0.272 | [0.97, 1.02] | 0.049 | 0.046 |
| LOGO variational | routed | 28557 | 0.532 | 1.09 | 0.591 | 1.13 | 0.313 | [0.90, 1.09] | 0.155 | 0.157 |
| LOGO variational | routing-free | 60795 | 0.593 | 1.02 | 0.620 | 1.03 | 0.186 | [0.97, 1.02] | 0.072 | 0.073 |
| control | routed | 23884 | 0.900 | 1.09 | 0.950 | 1.13 | 0.801 | [0.90, 1.09] | 0.042 | 0.042 |
| control | routing-free | 20315 | 0.900 | 1.02 | 0.950 | 1.03 | 0.800 | [0.97, 1.02] | 0.013 | 0.012 |
