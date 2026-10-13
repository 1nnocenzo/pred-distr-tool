# Approximate router: duration of routed circuits without learning or compiling

Greedy placement + shortest-path SWAPs (3 CZ + 4 r each), ASAP on physical qubits, fixed decomposition table; vs the true duration of the stored compilation. Error = |est / true − 1|, median [p90]; bias = median est / true. Scale fitted on the control split's routed circuits only: router × 0.705, logical schedule × 1.419.

| split | group | cases | router: error | router: bias | router × control scale: error | logical schedule: error | logical: bias | logical × control scale: error | corr(log est, log true), router / logical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOFO qaoa | routed | 450 | 0.635 [0.833] | 1.63 | 0.153 [0.292] | 0.192 [0.301] | 0.81 | 0.146 [0.378] | 0.983 / 0.989 |
| LOFO qnn | routing-free | 450 | 0.016 [0.024] | 1.02 | 0.284 [0.286] | 0.015 [0.022] | 1.01 | 0.440 [0.451] | 0.884 / 0.937 |
| LOGO variational | routing-free | 450 | 0.016 [0.024] | 1.02 | 0.284 [0.286] | 0.015 [0.022] | 1.01 | 0.440 [0.451] | 0.884 / 0.937 |
| control | routed | 234 | 0.418 [1.417] | 1.42 | 0.245 [0.704] | 0.295 [0.569] | 0.70 | 0.240 [0.389] | 0.949 / 0.875 |
| control | routing-free | 216 | 0.012 [0.534] | 1.01 | 0.290 [0.298] | 0.005 [0.023] | 1.00 | 0.419 [0.452] | 0.724 / 0.993 |
| all | routed | 684 | 0.633 [0.902] | 1.63 | 0.185 [0.341] | 0.201 [0.437] | 0.80 | 0.171 [0.378] | 0.951 / 0.882 |
| all | routing-free | 1116 | 0.015 [0.039] | 1.01 | 0.284 [0.290] | 0.014 [0.022] | 1.01 | 0.440 [0.451] | 0.753 / 0.990 |