# Predicted circuit duration (QPU time of one shot, ASAP schedule): how large is the error?

Seeds 5–7 pooled; (circuit, device) cases; relative error = |pred/true − 1|.

## v8kdfi10

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| cases | 24354 | 6840 | 89352 | 44199 |
| true duration median µs | 40.756 | 17.430 | 18.718 | 20.780 |
| rel. error median | 0.124 | 0.015 | 0.006 | 0.016 |
| rel. error p90 | 0.312 | 0.028 | 0.249 | 0.104 |
| abs. error median µs | 4.666 | 0.257 | 0.127 | 0.404 |
| abs. error p90 µs | 22.884 | 0.429 | 8.213 | 5.679 |
| bias (median pred/true) | 0.900 | 1.015 | 0.998 | 1.000 |
| within 10% | 0.423 | 1.000 | 0.792 | 0.891 |
| within 25% | 0.802 | 1.000 | 0.901 | 0.989 |
| rel. error median, EQE1_Top | 0.150 | 0.015 | 0.005 | 0.016 |
| rel. error median, EQE1_Bottom | 0.108 | 0.015 | 0.005 | 0.016 |
| rel. error median, QExa20 | 0.127 | 0.016 | 0.007 | 0.016 |
