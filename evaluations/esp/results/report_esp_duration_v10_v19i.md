# Predicted circuit duration (QPU time of one shot, ASAP schedule): how large is the error?

Seeds 5–7 pooled; (circuit, device) cases; relative error = |pred/true − 1|.

## v8kdi10

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| cases | 24354 | 6840 | 89352 | 44199 |
| true duration median µs | 40.756 | 17.430 | 18.718 | 20.780 |
| rel. error median | 0.104 | 0.049 | 0.069 | 0.031 |
| rel. error p90 | 0.282 | 0.091 | 0.250 | 0.116 |
| abs. error median µs | 3.935 | 0.839 | 1.330 | 0.840 |
| abs. error p90 µs | 19.134 | 1.646 | 7.940 | 5.840 |
| bias (median pred/true) | 0.921 | 1.049 | 0.975 | 1.005 |
| within 10% | 0.482 | 0.933 | 0.638 | 0.864 |
| within 25% | 0.857 | 1.000 | 0.900 | 0.985 |
| rel. error median, EQE1_Top | 0.112 | 0.055 | 0.068 | 0.032 |
| rel. error median, EQE1_Bottom | 0.096 | 0.051 | 0.071 | 0.031 |
| rel. error median, QExa20 | 0.104 | 0.041 | 0.067 | 0.031 |

## v8kdfi10

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| cases | 24354 | 6840 | 89352 | 44199 |
| true duration median µs | 40.756 | 17.430 | 18.718 | 20.780 |
| rel. error median | 0.124 | 0.048 | 0.088 | 0.024 |
| rel. error p90 | 0.312 | 0.086 | 0.306 | 0.101 |
| abs. error median µs | 4.666 | 0.842 | 1.693 | 0.552 |
| abs. error p90 µs | 22.884 | 1.459 | 8.814 | 5.671 |
| bias (median pred/true) | 0.887 | 1.048 | 0.990 | 1.004 |
| within 10% | 0.409 | 0.955 | 0.548 | 0.899 |
| within 25% | 0.802 | 1.000 | 0.851 | 0.989 |
| rel. error median, EQE1_Top | 0.146 | 0.050 | 0.087 | 0.023 |
| rel. error median, EQE1_Bottom | 0.110 | 0.048 | 0.090 | 0.024 |
| rel. error median, QExa20 | 0.126 | 0.046 | 0.087 | 0.024 |
