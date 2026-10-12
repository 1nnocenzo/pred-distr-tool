# End-to-end ESP models

Mean ± std over seeds (3 each). Error = median |pred/true − 1|. R² F of the source version = the same version before the ESP fine-tuning.

## R² F

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.694 ± 0.128 | 0.941 ± 0.013 | 0.825 ± 0.070 | 0.996 ± 0.000 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.891 ± 0.053 | 0.773 ± 0.064 | 0.821 ± 0.036 | 0.996 ± 0.000 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.943 ± 0.055 | 0.878 ± 0.007 | 0.926 ± 0.004 | 0.994 ± 0.000 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.965 ± 0.027 | 0.870 ± 0.012 | 0.929 ± 0.008 | 0.994 ± 0.000 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.966 ± 0.019 | 0.847 ± 0.036 | 0.937 ± 0.013 | 0.994 ± 0.000 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.976 ± 0.010 | 0.848 ± 0.014 | 0.942 ± 0.007 | 0.994 ± 0.000 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.962 ± 0.029 | 0.848 ± 0.020 | 0.938 ± 0.006 | 0.994 ± 0.000 |

## R² log depth

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.282 ± 0.392 | 0.667 ± 0.206 | 0.590 ± 0.173 | 0.995 ± 0.001 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.781 ± 0.099 | 0.313 ± 0.163 | 0.646 ± 0.025 | 0.995 ± 0.000 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.752 ± 0.042 | 0.272 ± 0.174 | 0.677 ± 0.062 | 0.993 ± 0.000 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.742 ± 0.087 | 0.580 ± 0.252 | 0.659 ± 0.050 | 0.993 ± 0.000 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.744 ± 0.115 | 0.236 ± 0.178 | 0.694 ± 0.043 | 0.993 ± 0.001 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.832 ± 0.043 | 0.054 ± 0.189 | 0.293 ± 0.126 | 0.993 ± 0.001 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.731 ± 0.092 | 0.322 ± 0.393 | 0.676 ± 0.025 | 0.993 ± 0.001 |

## depth error (median)

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.639 ± 0.116 | 0.246 ± 0.114 | 0.386 ± 0.021 | 0.039 ± 0.002 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.374 ± 0.051 | 0.304 ± 0.044 | 0.356 ± 0.067 | 0.040 ± 0.000 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.450 ± 0.040 | 0.571 ± 0.100 | 0.359 ± 0.009 | 0.055 ± 0.001 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.429 ± 0.080 | 0.352 ± 0.162 | 0.376 ± 0.030 | 0.052 ± 0.003 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.411 ± 0.093 | 0.584 ± 0.083 | 0.369 ± 0.026 | 0.060 ± 0.001 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.306 ± 0.071 | 0.737 ± 0.101 | 0.961 ± 0.161 | 0.061 ± 0.007 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.407 ± 0.077 | 0.527 ± 0.227 | 0.387 ± 0.016 | 0.063 ± 0.004 |

## R² log duration

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.286 ± 0.023 | -1.748 ± 1.896 | 0.089 ± 0.054 | 0.989 ± 0.002 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.199 ± 0.107 | -3.687 ± 1.837 | 0.054 ± 0.086 | 0.985 ± 0.002 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.251 ± 0.118 | -7.782 ± 2.037 | -0.051 ± 0.378 | 0.984 ± 0.000 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.307 ± 0.079 | -8.776 ± 2.090 | -0.279 ± 0.257 | 0.983 ± 0.001 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.337 ± 0.040 | -9.612 ± 2.124 | 0.081 ± 0.219 | 0.979 ± 0.000 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.336 ± 0.100 | -9.172 ± 2.755 | -0.502 ± 0.407 | 0.980 ± 0.003 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.340 ± 0.082 | -7.624 ± 1.536 | -0.191 ± 0.402 | 0.982 ± 0.000 |

## duration error (median)

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.521 ± 0.056 | 0.085 ± 0.052 | 0.240 ± 0.020 | 0.028 ± 0.001 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.588 ± 0.243 | 0.083 ± 0.012 | 0.276 ± 0.038 | 0.031 ± 0.001 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.664 ± 0.229 | 0.169 ± 0.009 | 0.280 ± 0.069 | 0.052 ± 0.002 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.677 ± 0.198 | 0.197 ± 0.012 | 0.346 ± 0.070 | 0.052 ± 0.002 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.568 ± 0.175 | 0.203 ± 0.040 | 0.252 ± 0.046 | 0.063 ± 0.001 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.657 ± 0.276 | 0.186 ± 0.076 | 0.366 ± 0.146 | 0.057 ± 0.003 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.554 ± 0.185 | 0.178 ± 0.021 | 0.341 ± 0.115 | 0.059 ± 0.001 |

## R² ESP

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.866 ± 0.051 | 0.738 ± 0.068 | 0.433 ± 0.013 | 0.984 ± 0.001 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.805 ± 0.035 | 0.389 ± 0.051 | 0.413 ± 0.039 | 0.971 ± 0.009 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.885 ± 0.012 | 0.443 ± 0.130 | 0.486 ± 0.052 | 0.980 ± 0.001 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.890 ± 0.003 | 0.324 ± 0.144 | 0.444 ± 0.042 | 0.980 ± 0.001 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.892 ± 0.016 | 0.286 ± 0.027 | 0.521 ± 0.054 | 0.977 ± 0.002 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.902 ± 0.022 | 0.361 ± 0.101 | 0.448 ± 0.062 | 0.978 ± 0.000 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.885 ± 0.018 | 0.403 ± 0.101 | 0.460 ± 0.067 | 0.978 ± 0.002 |

## right device by ESP, model

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.573 ± 0.035 | 0.850 ± 0.084 | 0.522 ± 0.013 | 0.825 ± 0.008 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.477 ± 0.062 | 0.546 ± 0.082 | 0.563 ± 0.024 | 0.826 ± 0.010 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.435 ± 0.056 | 0.562 ± 0.200 | 0.527 ± 0.045 | 0.791 ± 0.013 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.489 ± 0.038 | 0.508 ± 0.118 | 0.541 ± 0.045 | 0.796 ± 0.008 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.461 ± 0.021 | 0.461 ± 0.119 | 0.555 ± 0.053 | 0.769 ± 0.010 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.459 ± 0.025 | 0.566 ± 0.182 | 0.551 ± 0.059 | 0.771 ± 0.005 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.433 ± 0.027 | 0.528 ± 0.126 | 0.553 ± 0.073 | 0.770 ± 0.006 |

## ESP regret, model

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.006 ± 0.001 | 0.004 ± 0.004 | 0.019 ± 0.001 | 0.003 ± 0.000 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.006 ± 0.001 | 0.017 ± 0.004 | 0.017 ± 0.001 | 0.003 ± 0.001 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.008 ± 0.001 | 0.019 ± 0.009 | 0.020 ± 0.002 | 0.004 ± 0.001 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.007 ± 0.001 | 0.020 ± 0.005 | 0.019 ± 0.002 | 0.004 ± 0.000 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.008 ± 0.001 | 0.023 ± 0.007 | 0.018 ± 0.003 | 0.005 ± 0.000 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.007 ± 0.001 | 0.018 ± 0.010 | 0.018 ± 0.003 | 0.005 ± 0.000 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.008 ± 0.001 | 0.020 ± 0.003 | 0.018 ± 0.004 | 0.005 ± 0.000 |

## ESP regret, baseline F

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.007 ± 0.001 | 0.002 ± 0.002 | 0.044 ± 0.009 | 0.009 ± 0.000 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.011 ± 0.001 | 0.011 ± 0.007 | 0.024 ± 0.004 | 0.010 ± 0.000 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.014 ± 0.004 | 0.011 ± 0.013 | 0.024 ± 0.003 | 0.008 ± 0.000 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.009 ± 0.002 | 0.010 ± 0.005 | 0.025 ± 0.007 | 0.007 ± 0.001 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.011 ± 0.004 | 0.014 ± 0.002 | 0.020 ± 0.005 | 0.008 ± 0.000 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.012 ± 0.003 | 0.008 ± 0.006 | 0.019 ± 0.003 | 0.008 ± 0.001 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.012 ± 0.002 | 0.006 ± 0.004 | 0.024 ± 0.009 | 0.008 ± 0.000 |

## R² F, source version

| model | trained from | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- | --- |
| v5 + ESP | v5 checkpoint, 40 ep | 0.707 ± 0.086 | 0.934 ± 0.028 | 0.797 ± 0.084 | 0.996 ± 0.000 |
| v8a + ESP | v8a checkpoint, 40 ep | 0.892 ± 0.066 | 0.871 ± 0.026 | 0.880 ± 0.030 | 0.996 ± 0.000 |
| v8kd40 + ESP | v8kd40 checkpoint, 40 ep | 0.964 ± 0.026 | 0.877 ± 0.040 | 0.919 ± 0.020 | 0.996 ± 0.000 |
| v8kdl40 + ESP | v8kdl40 checkpoint, 40 ep | 0.969 ± 0.022 | 0.863 ± 0.084 | 0.934 ± 0.010 | 0.996 ± 0.000 |
| v8kdc10 + ESP | v8kdc10 checkpoint, 40 ep | 0.970 ± 0.019 | 0.872 ± 0.080 | 0.940 ± 0.014 | 0.995 ± 0.000 |
| v8kde10 (v13 head) | v8k checkpoint, v8kdc10 recipe | 0.970 ± 0.019 | 0.872 ± 0.080 | 0.940 ± 0.014 | 0.995 ± 0.000 |
| v8kda10 (analytic) | v8k checkpoint, v8kdc10 recipe | 0.970 ± 0.019 | 0.872 ± 0.080 | 0.940 ± 0.014 | 0.995 ± 0.000 |
