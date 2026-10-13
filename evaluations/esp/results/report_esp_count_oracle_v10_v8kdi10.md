# Duration with the right gate counts: ceiling of the count route

ASAP schedule of the logical gate DAG with per-gate counts from each source (see esp_count_oracle.py); true duration of the stored compilation. Seeds pooled; relative error |pred/true − 1|. "→tot" = per-gate counts rescaled to the compiler's true r / CZ totals of the circuit (oracle).

## v8kdi10: all

| source | LOFO qaoa (n 24354) | LOFO qnn (n 6840) | LOGO variational (n 89352) | control (n 44199) |
| --- | --- | --- | --- | --- |
| model | 0.104 (p90 0.28, bias 0.92, R² 0.91) | 0.049 (p90 0.09, bias 1.05, R² 0.34) | 0.069 (p90 0.25, bias 0.97, R² 0.78) | 0.031 (p90 0.12, bias 1.00, R² 0.98) |
| pred | 0.293 (p90 0.52, bias 0.71, R² 0.45) | 0.044 (p90 0.07, bias 0.96, R² 0.47) | 0.051 (p90 0.34, bias 0.98, R² 0.61) | 0.063 (p90 0.33, bias 0.94, R² 0.86) |
| pred→tot | 0.188 (p90 0.34, bias 0.81, R² 0.82) | 0.017 (p90 0.06, bias 0.98, R² 0.78) | 0.035 (p90 0.18, bias 1.01, R² 0.91) | 0.038 (p90 0.24, bias 0.99, R² 0.95) |
| pred→cz | 0.275 (p90 0.46, bias 0.73, R² 0.58) | 0.046 (p90 0.08, bias 0.95, R² 0.45) | 0.047 (p90 0.27, bias 0.98, R² 0.79) | 0.071 (p90 0.29, bias 0.93, R² 0.91) |
| pred→r | 0.203 (p90 0.38, bias 0.80, R² 0.77) | 0.016 (p90 0.06, bias 0.99, R² 0.79) | 0.034 (p90 0.23, bias 1.01, R² 0.85) | 0.037 (p90 0.23, bias 0.99, R² 0.95) |
| table | 0.209 (p90 0.47, bias 0.79, R² 0.63) | 0.015 (p90 0.03, bias 1.01, R² 0.92) | 0.007 (p90 0.48, bias 1.00, R² 0.34) | 0.051 (p90 0.50, bias 0.97, R² 0.65) |
| table→tot | 0.224 (p90 0.41, bias 0.78, R² 0.71) | 0.004 (p90 0.04, bias 1.00, R² 0.91) | 0.005 (p90 0.32, bias 1.00, R² 0.77) | 0.016 (p90 0.39, bias 0.99, R² 0.82) |

## v8kdi10: routing-free

| source | LOFO qaoa (n 4212) | LOFO qnn (n 6720) | LOGO variational (n 60795) | control (n 20315) |
| --- | --- | --- | --- | --- |
| model | 0.055 (p90 0.11, bias 0.95, R² 0.45) | 0.049 (p90 0.09, bias 1.05, R² 0.27) | 0.054 (p90 0.16, bias 0.98, R² 0.21) | 0.019 (p90 0.09, bias 1.01, R² 0.82) |
| pred | 0.060 (p90 0.14, bias 0.94, R² 0.14) | 0.044 (p90 0.07, bias 0.96, R² 0.46) | 0.040 (p90 0.09, bias 0.98, R² 0.74) | 0.042 (p90 0.08, bias 0.96, R² 0.79) |
| pred→tot | 0.025 (p90 0.06, bias 0.98, R² 0.85) | 0.017 (p90 0.06, bias 0.98, R² 0.77) | 0.028 (p90 0.11, bias 1.03, R² 0.74) | 0.015 (p90 0.06, bias 1.00, R² 0.91) |
| pred→cz | 0.048 (p90 0.11, bias 0.95, R² 0.48) | 0.046 (p90 0.08, bias 0.95, R² 0.42) | 0.037 (p90 0.09, bias 0.99, R² 0.75) | 0.044 (p90 0.08, bias 0.96, R² 0.77) |
| pred→r | 0.039 (p90 0.09, bias 0.96, R² 0.67) | 0.015 (p90 0.06, bias 0.99, R² 0.79) | 0.023 (p90 0.10, bias 1.02, R² 0.77) | 0.017 (p90 0.06, bias 1.00, R² 0.90) |
| table | 0.037 (p90 0.13, bias 1.04, R² 0.53) | 0.015 (p90 0.03, bias 1.02, R² 0.93) | 0.005 (p90 0.02, bias 1.00, R² 0.99) | 0.005 (p90 0.03, bias 1.00, R² 0.96) |
| table→tot | 0.023 (p90 0.06, bias 0.98, R² 0.87) | 0.004 (p90 0.04, bias 1.00, R² 0.91) | 0.004 (p90 0.01, bias 1.00, R² 1.00) | 0.004 (p90 0.01, bias 1.00, R² 0.98) |

## v8kdi10: routed

| source | LOFO qaoa (n 20142) | LOFO qnn (n 120) | LOGO variational (n 28557) | control (n 23884) |
| --- | --- | --- | --- | --- |
| model | 0.125 (p90 0.30, bias 0.91, R² 0.85) | 0.038 (p90 0.07, bias 1.02, R² -6.96) | 0.126 (p90 0.39, bias 0.94, R² 0.66) | 0.045 (p90 0.14, bias 0.99, R² 0.97) |
| pred | 0.337 (p90 0.53, bias 0.66, R² 0.10) | 0.111 (p90 0.13, bias 0.89, R² -56.78) | 0.153 (p90 0.55, bias 0.93, R² 0.26) | 0.136 (p90 0.43, bias 0.87, R² 0.72) |
| pred→tot | 0.217 (p90 0.35, bias 0.78, R² 0.70) | 0.060 (p90 0.11, bias 0.94, R² -21.61) | 0.066 (p90 0.28, bias 0.96, R² 0.86) | 0.109 (p90 0.29, bias 0.95, R² 0.90) |
| pred→cz | 0.316 (p90 0.47, bias 0.68, R² 0.30) | 0.091 (p90 0.11, bias 0.91, R² -39.38) | 0.125 (p90 0.43, bias 0.91, R² 0.62) | 0.150 (p90 0.35, bias 0.85, R² 0.82) |
| pred→r | 0.237 (p90 0.39, bias 0.76, R² 0.62) | 0.072 (p90 0.13, bias 0.93, R² -30.92) | 0.113 (p90 0.37, bias 0.96, R² 0.74) | 0.092 (p90 0.30, bias 0.96, R² 0.90) |
| table | 0.263 (p90 0.48, bias 0.74, R² 0.39) | 0.050 (p90 0.07, bias 0.95, R² -10.93) | 0.199 (p90 0.63, bias 0.80, R² -0.30) | 0.270 (p90 0.57, bias 0.73, R² 0.29) |
| table→tot | 0.269 (p90 0.42, bias 0.73, R² 0.52) | 0.019 (p90 0.10, bias 0.98, R² -10.61) | 0.127 (p90 0.44, bias 0.87, R² 0.54) | 0.225 (p90 0.44, bias 0.77, R² 0.64) |
