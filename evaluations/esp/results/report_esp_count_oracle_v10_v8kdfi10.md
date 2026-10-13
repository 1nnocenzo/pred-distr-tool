# Duration with the right gate counts: ceiling of the count route

ASAP schedule of the logical gate DAG with per-gate counts from each source (see esp_count_oracle.py); true duration of the stored compilation. Seeds pooled; relative error |pred/true − 1|. "→tot" = per-gate counts rescaled to the compiler's true r / CZ totals of the circuit (oracle).

## v8kdfi10: all

| source | LOFO qaoa (n 24354) | LOFO qnn (n 6840) | LOGO variational (n 89352) | control (n 44199) |
| --- | --- | --- | --- | --- |
| model | 0.124 (p90 0.31, bias 0.89, R² 0.86) | 0.048 (p90 0.09, bias 1.05, R² 0.41) | 0.088 (p90 0.31, bias 0.99, R² 0.75) | 0.024 (p90 0.10, bias 1.00, R² 0.99) |
| pred | 0.288 (p90 0.51, bias 0.71, R² 0.48) | 0.041 (p90 0.07, bias 0.96, R² 0.56) | 0.050 (p90 0.34, bias 0.97, R² 0.61) | 0.062 (p90 0.33, bias 0.94, R² 0.86) |
| pred→tot | 0.167 (p90 0.33, bias 0.83, R² 0.83) | 0.014 (p90 0.06, bias 0.99, R² 0.80) | 0.034 (p90 0.17, bias 1.01, R² 0.92) | 0.037 (p90 0.22, bias 0.99, R² 0.96) |
| pred→cz | 0.267 (p90 0.46, bias 0.73, R² 0.59) | 0.042 (p90 0.07, bias 0.96, R² 0.54) | 0.045 (p90 0.27, bias 0.97, R² 0.79) | 0.070 (p90 0.28, bias 0.93, R² 0.91) |
| pred→r | 0.184 (p90 0.37, bias 0.82, R² 0.79) | 0.013 (p90 0.06, bias 0.99, R² 0.81) | 0.033 (p90 0.22, bias 1.01, R² 0.86) | 0.037 (p90 0.20, bias 0.99, R² 0.96) |
| table | 0.209 (p90 0.47, bias 0.79, R² 0.63) | 0.015 (p90 0.03, bias 1.01, R² 0.92) | 0.007 (p90 0.48, bias 1.00, R² 0.34) | 0.051 (p90 0.50, bias 0.97, R² 0.65) |
| table→tot | 0.224 (p90 0.41, bias 0.78, R² 0.71) | 0.004 (p90 0.04, bias 1.00, R² 0.91) | 0.005 (p90 0.32, bias 1.00, R² 0.77) | 0.016 (p90 0.39, bias 0.99, R² 0.82) |

## v8kdfi10: routing-free

| source | LOFO qaoa (n 4212) | LOFO qnn (n 6720) | LOGO variational (n 60795) | control (n 20315) |
| --- | --- | --- | --- | --- |
| model | 0.054 (p90 0.13, bias 0.96, R² 0.30) | 0.049 (p90 0.09, bias 1.05, R² 0.35) | 0.072 (p90 0.22, bias 1.00, R² -0.14) | 0.013 (p90 0.04, bias 1.00, R² 0.96) |
| pred | 0.061 (p90 0.14, bias 0.94, R² 0.14) | 0.040 (p90 0.07, bias 0.96, R² 0.55) | 0.040 (p90 0.09, bias 0.98, R² 0.75) | 0.039 (p90 0.08, bias 0.96, R² 0.81) |
| pred→tot | 0.026 (p90 0.06, bias 0.97, R² 0.86) | 0.013 (p90 0.06, bias 0.99, R² 0.80) | 0.027 (p90 0.10, bias 1.02, R² 0.78) | 0.015 (p90 0.05, bias 1.00, R² 0.91) |
| pred→cz | 0.048 (p90 0.11, bias 0.95, R² 0.48) | 0.042 (p90 0.07, bias 0.96, R² 0.53) | 0.036 (p90 0.08, bias 0.98, R² 0.78) | 0.040 (p90 0.08, bias 0.96, R² 0.80) |
| pred→r | 0.039 (p90 0.09, bias 0.96, R² 0.67) | 0.013 (p90 0.05, bias 0.99, R² 0.81) | 0.022 (p90 0.09, bias 1.02, R² 0.82) | 0.017 (p90 0.06, bias 1.00, R² 0.91) |
| table | 0.037 (p90 0.13, bias 1.04, R² 0.53) | 0.015 (p90 0.03, bias 1.02, R² 0.93) | 0.005 (p90 0.02, bias 1.00, R² 0.99) | 0.005 (p90 0.03, bias 1.00, R² 0.96) |
| table→tot | 0.023 (p90 0.06, bias 0.98, R² 0.87) | 0.004 (p90 0.04, bias 1.00, R² 0.91) | 0.004 (p90 0.01, bias 1.00, R² 1.00) | 0.004 (p90 0.01, bias 1.00, R² 0.98) |

## v8kdfi10: routed

| source | LOFO qaoa (n 20142) | LOFO qnn (n 120) | LOGO variational (n 28557) | control (n 23884) |
| --- | --- | --- | --- | --- |
| model | 0.153 (p90 0.33, bias 0.86, R² 0.78) | 0.032 (p90 0.07, bias 1.01, R² -5.50) | 0.155 (p90 0.42, bias 0.94, R² 0.64) | 0.042 (p90 0.14, bias 1.01, R² 0.98) |
| pred | 0.330 (p90 0.52, bias 0.67, R² 0.14) | 0.103 (p90 0.13, bias 0.90, R² -50.05) | 0.141 (p90 0.55, bias 0.93, R² 0.25) | 0.137 (p90 0.42, bias 0.87, R² 0.73) |
| pred→tot | 0.200 (p90 0.34, bias 0.80, R² 0.73) | 0.056 (p90 0.12, bias 0.94, R² -21.89) | 0.066 (p90 0.29, bias 0.96, R² 0.86) | 0.102 (p90 0.27, bias 0.95, R² 0.91) |
| pred→cz | 0.313 (p90 0.47, bias 0.69, R² 0.33) | 0.089 (p90 0.11, bias 0.91, R² -39.05) | 0.119 (p90 0.43, bias 0.91, R² 0.61) | 0.155 (p90 0.33, bias 0.85, R² 0.83) |
| pred→r | 0.223 (p90 0.38, bias 0.78, R² 0.65) | 0.068 (p90 0.13, bias 0.93, R² -29.37) | 0.104 (p90 0.37, bias 0.96, R² 0.74) | 0.082 (p90 0.26, bias 0.96, R² 0.92) |
| table | 0.263 (p90 0.48, bias 0.74, R² 0.39) | 0.050 (p90 0.07, bias 0.95, R² -10.93) | 0.199 (p90 0.63, bias 0.80, R² -0.30) | 0.270 (p90 0.57, bias 0.73, R² 0.29) |
| table→tot | 0.269 (p90 0.42, bias 0.73, R² 0.52) | 0.019 (p90 0.10, bias 0.98, R² -10.61) | 0.127 (p90 0.44, bias 0.87, R² 0.54) | 0.225 (p90 0.44, bias 0.77, R² 0.64) |
