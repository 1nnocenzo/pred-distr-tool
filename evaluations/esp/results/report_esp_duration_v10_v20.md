# Predicted circuit duration (QPU time of one shot, ASAP schedule): how large is the error?

Seeds 5–7 pooled; (circuit, device) cases; relative error = |pred/true − 1|.

## v8kdg10

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| cases | 24354 | 6840 | 89352 | 44199 |
| true duration median µs | 40.756 | 17.430 | 18.718 | 20.780 |
| rel. error median | 0.148 | 0.038 | 0.097 | 0.026 |
| rel. error p90 | 0.441 | 0.073 | 0.316 | 0.085 |
| abs. error median µs | 6.297 | 0.669 | 1.971 | 0.642 |
| abs. error p90 µs | 33.329 | 1.251 | 8.357 | 4.556 |
| bias (median pred/true) | 0.894 | 1.038 | 1.085 | 1.006 |
| within 10% | 0.378 | 0.999 | 0.509 | 0.932 |
| within 25% | 0.685 | 1.000 | 0.839 | 0.998 |
| rel. error median, EQE1_Top | 0.145 | 0.046 | 0.081 | 0.026 |
| rel. error median, EQE1_Bottom | 0.141 | 0.038 | 0.094 | 0.025 |
| rel. error median, QExa20 | 0.158 | 0.033 | 0.121 | 0.028 |

## v8kdf10

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| cases | 24354 | 6840 | 89352 | 44199 |
| true duration median µs | 40.756 | 17.430 | 18.718 | 20.780 |
| rel. error median | 0.133 | 0.031 | 0.114 | 0.019 |
| rel. error p90 | 0.431 | 0.054 | 0.338 | 0.076 |
| abs. error median µs | 5.478 | 0.541 | 2.321 | 0.435 |
| abs. error p90 µs | 32.207 | 0.920 | 8.908 | 4.253 |
| bias (median pred/true) | 0.918 | 1.031 | 1.108 | 1.007 |
| within 10% | 0.396 | 1.000 | 0.451 | 0.947 |
| within 25% | 0.707 | 1.000 | 0.815 | 0.998 |
| rel. error median, EQE1_Top | 0.130 | 0.037 | 0.099 | 0.019 |
| rel. error median, EQE1_Bottom | 0.126 | 0.034 | 0.108 | 0.019 |
| rel. error median, QExa20 | 0.143 | 0.024 | 0.137 | 0.019 |

## v8kdp10

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| cases | 24354 | 6840 | 89352 | 44199 |
| true duration median µs | 40.756 | 17.430 | 18.718 | 20.780 |
| rel. error median | 0.121 | 0.023 | 0.115 | 0.018 |
| rel. error p90 | 0.404 | 0.045 | 0.350 | 0.072 |
| abs. error median µs | 5.000 | 0.405 | 2.346 | 0.412 |
| abs. error p90 µs | 31.094 | 0.759 | 9.168 | 3.913 |
| bias (median pred/true) | 0.926 | 1.022 | 1.112 | 1.004 |
| within 10% | 0.415 | 1.000 | 0.448 | 0.956 |
| within 25% | 0.741 | 1.000 | 0.801 | 0.999 |
| rel. error median, EQE1_Top | 0.120 | 0.026 | 0.105 | 0.018 |
| rel. error median, EQE1_Bottom | 0.120 | 0.026 | 0.113 | 0.019 |
| rel. error median, QExa20 | 0.124 | 0.017 | 0.126 | 0.017 |
