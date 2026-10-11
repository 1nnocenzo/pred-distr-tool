## Compiler noise floor (recompiling on the original devices vs. the dataset)
EQE1_Top     R2 0.9973  MAE 0.0057
EQE1_Bottom  R2 0.9939  MAE 0.0082
QExa20       R2 0.9989  MAE 0.0039

## Accuracy per variant (R2 / MAE) and tracking of the change vs. orig (seed 5)
tracking = corr(predicted log F(variant) - log F(orig), true ...), on circuits with F > 0.01

### EQE1_Top
variant                            v3_xattn                           v4_phys                       v5_sinkhorn                      v5b_sinkhorn                           v6_l001                            v6_l01                          v4u_phys                           v7_l001                            v7_l01
orig                     R2 0.995 MAE 0.012                R2 0.987 MAE 0.015                R2 0.995 MAE 0.011                R2 0.996 MAE 0.010                R2 0.995 MAE 0.012                R2 0.994 MAE 0.013                R2 0.987 MAE 0.016                R2 0.995 MAE 0.011                R2 0.995 MAE 0.011
x0.5           R2 0.836 MAE 0.110 trk -0.62      R2 0.991 MAE 0.023 trk +0.95      R2 0.990 MAE 0.025 trk +0.95      R2 0.953 MAE 0.043 trk +0.94      R2 0.981 MAE 0.040 trk +0.94      R2 0.984 MAE 0.032 trk +0.90      R2 0.989 MAE 0.025 trk +0.95      R2 0.991 MAE 0.024 trk +0.95      R2 0.989 MAE 0.027 trk +0.94
x2             R2 0.724 MAE 0.107 trk -0.00      R2 0.974 MAE 0.015 trk +0.94      R2 0.990 MAE 0.015 trk +0.93      R2 0.957 MAE 0.035 trk +0.81      R2 0.938 MAE 0.036 trk +0.79      R2 0.925 MAE 0.044 trk +0.72      R2 0.972 MAE 0.016 trk +0.94      R2 0.982 MAE 0.021 trk +0.92      R2 0.966 MAE 0.027 trk +0.86
shuffle        R2 0.982 MAE 0.025 trk -0.10      R2 0.799 MAE 0.100 trk +0.22      R2 0.981 MAE 0.026 trk -0.07      R2 0.980 MAE 0.028 trk -0.00      R2 0.986 MAE 0.023 trk +0.15      R2 0.984 MAE 0.024 trk +0.14      R2 0.974 MAE 0.026 trk +0.16      R2 0.982 MAE 0.025 trk +0.01      R2 0.987 MAE 0.022 trk +0.31
jitterA        R2 0.984 MAE 0.023 trk -0.18      R2 0.976 MAE 0.027 trk +0.06      R2 0.984 MAE 0.023 trk -0.04      R2 0.985 MAE 0.022 trk -0.16      R2 0.981 MAE 0.025 trk -0.19      R2 0.981 MAE 0.026 trk +0.09      R2 0.973 MAE 0.029 trk -0.50      R2 0.985 MAE 0.023 trk +0.10      R2 0.987 MAE 0.023 trk +0.35
jitterB        R2 0.990 MAE 0.018 trk +0.13      R2 0.970 MAE 0.035 trk -0.35      R2 0.991 MAE 0.018 trk +0.33      R2 0.993 MAE 0.016 trk +0.37      R2 0.991 MAE 0.019 trk +0.48      R2 0.990 MAE 0.020 trk +0.29      R2 0.986 MAE 0.020 trk +0.42      R2 0.990 MAE 0.018 trk +0.30      R2 0.991 MAE 0.019 trk +0.44

### EQE1_Bottom
variant                            v3_xattn                           v4_phys                       v5_sinkhorn                      v5b_sinkhorn                           v6_l001                            v6_l01                          v4u_phys                           v7_l001                            v7_l01
orig                     R2 0.991 MAE 0.016                R2 0.985 MAE 0.017                R2 0.993 MAE 0.013                R2 0.995 MAE 0.012                R2 0.993 MAE 0.014                R2 0.993 MAE 0.014                R2 0.984 MAE 0.018                R2 0.995 MAE 0.012                R2 0.993 MAE 0.013
x0.5           R2 0.829 MAE 0.114 trk -0.01      R2 0.988 MAE 0.025 trk +0.91      R2 0.988 MAE 0.026 trk +0.92      R2 0.980 MAE 0.037 trk +0.89      R2 0.981 MAE 0.039 trk +0.88      R2 0.986 MAE 0.032 trk +0.87      R2 0.987 MAE 0.028 trk +0.90      R2 0.990 MAE 0.024 trk +0.90      R2 0.993 MAE 0.020 trk +0.91
x2             R2 0.728 MAE 0.108 trk -0.69      R2 0.978 MAE 0.016 trk +0.93      R2 0.979 MAE 0.023 trk +0.89      R2 0.987 MAE 0.016 trk +0.85      R2 0.889 MAE 0.038 trk +0.80      R2 0.898 MAE 0.054 trk +0.67      R2 0.978 MAE 0.017 trk +0.92      R2 0.989 MAE 0.015 trk +0.91      R2 0.979 MAE 0.021 trk +0.86
shuffle        R2 0.977 MAE 0.031 trk +0.01      R2 0.658 MAE 0.136 trk +0.65      R2 0.932 MAE 0.053 trk +0.45      R2 0.987 MAE 0.021 trk +0.19      R2 0.985 MAE 0.022 trk +0.45      R2 0.981 MAE 0.025 trk +0.47      R2 0.968 MAE 0.036 trk -0.67      R2 0.982 MAE 0.027 trk +0.52      R2 0.984 MAE 0.022 trk +0.43
jitterA        R2 0.992 MAE 0.017 trk -0.08      R2 0.984 MAE 0.022 trk +0.24      R2 0.991 MAE 0.018 trk +0.18      R2 0.990 MAE 0.017 trk -0.30      R2 0.990 MAE 0.019 trk +0.36      R2 0.990 MAE 0.019 trk +0.13      R2 0.986 MAE 0.021 trk -0.33      R2 0.991 MAE 0.018 trk +0.37      R2 0.989 MAE 0.020 trk +0.33
jitterB        R2 0.989 MAE 0.018 trk -0.10      R2 0.986 MAE 0.019 trk +0.30      R2 0.991 MAE 0.017 trk +0.02      R2 0.993 MAE 0.016 trk +0.19      R2 0.993 MAE 0.017 trk +0.19      R2 0.992 MAE 0.016 trk +0.23      R2 0.986 MAE 0.022 trk +0.23      R2 0.994 MAE 0.014 trk +0.25      R2 0.992 MAE 0.016 trk +0.38

### QExa20
variant                            v3_xattn                           v4_phys                       v5_sinkhorn                      v5b_sinkhorn                           v6_l001                            v6_l01                          v4u_phys                           v7_l001                            v7_l01
orig                     R2 0.994 MAE 0.012                R2 0.992 MAE 0.012                R2 0.998 MAE 0.009                R2 0.998 MAE 0.008                R2 0.997 MAE 0.009                R2 0.995 MAE 0.012                R2 0.989 MAE 0.014                R2 0.998 MAE 0.007                R2 0.996 MAE 0.010
x0.5           R2 0.796 MAE 0.123 trk -0.21      R2 0.993 MAE 0.018 trk +0.98      R2 0.995 MAE 0.018 trk +0.98      R2 0.995 MAE 0.016 trk +0.97      R2 0.993 MAE 0.023 trk +0.98      R2 0.983 MAE 0.029 trk +0.96      R2 0.992 MAE 0.020 trk +0.98      R2 0.996 MAE 0.015 trk +0.99      R2 0.994 MAE 0.019 trk +0.98
x2             R2 0.666 MAE 0.106 trk +0.12      R2 0.518 MAE 0.109 trk +0.94      R2 0.995 MAE 0.009 trk +0.94      R2 0.995 MAE 0.009 trk +0.96      R2 0.981 MAE 0.014 trk +0.93      R2 0.952 MAE 0.026 trk +0.76      R2 0.975 MAE 0.014 trk +0.96      R2 0.991 MAE 0.011 trk +0.92      R2 0.964 MAE 0.018 trk +0.90
shuffle        R2 0.938 MAE 0.048 trk +0.23      R2 0.788 MAE 0.087 trk +0.34      R2 0.908 MAE 0.056 trk +0.06      R2 0.943 MAE 0.046 trk -0.01      R2 0.942 MAE 0.041 trk +0.34      R2 0.965 MAE 0.030 trk +0.46      R2 0.932 MAE 0.052 trk -0.57      R2 0.967 MAE 0.031 trk +0.41      R2 0.978 MAE 0.026 trk +0.65
jitterA        R2 0.985 MAE 0.024 trk +0.11      R2 0.436 MAE 0.171 trk -0.04      R2 0.994 MAE 0.014 trk +0.41      R2 0.989 MAE 0.020 trk -0.05      R2 0.975 MAE 0.027 trk +0.26      R2 0.991 MAE 0.017 trk +0.16      R2 0.985 MAE 0.021 trk -0.09      R2 0.968 MAE 0.031 trk +0.20      R2 0.993 MAE 0.015 trk +0.18
jitterB        R2 0.990 MAE 0.018 trk +0.11      R2 0.985 MAE 0.023 trk -0.04      R2 0.992 MAE 0.017 trk -0.03      R2 0.989 MAE 0.020 trk -0.22      R2 0.991 MAE 0.017 trk +0.08      R2 0.991 MAE 0.018 trk +0.20      R2 0.982 MAE 0.025 trk -0.09      R2 0.993 MAE 0.016 trk -0.03      R2 0.991 MAE 0.016 trk +0.23

## Device choice among all variants: accuracy / mean regret
v3_xattn     accuracy 0.079  regret 0.1559
v4_phys      accuracy 0.694  regret 0.0040
v5_sinkhorn  accuracy 0.654  regret 0.0048
v5b_sinkhorn accuracy 0.510  regret 0.0091
v6_l001      accuracy 0.536  regret 0.0092
v6_l01       accuracy 0.667  regret 0.0068
v4u_phys     accuracy 0.772  regret 0.0033
v7_l001      accuracy 0.711  regret 0.0049
v7_l01       accuracy 0.740  regret 0.0046

## Summary over seeds (mean ± std over the available seeds)
worst R2 = min over all 18 device variants; scale = x0.5/x2; local = shuffle/jitter
model                       worst R2               mean R2        tracking scale        tracking local            choice acc         choice regret
v3_xattn                    0.666(1)              0.910(1)             -0.234(1)              0.015(1)              0.079(1)              0.156(1)
v4_phys               0.760±0.280(3)        0.952±0.055(3)        0.947±0.005(3)        0.027±0.136(3)        0.675±0.026(3)        0.004±0.000(3)
v5_sinkhorn           0.929±0.034(3)        0.984±0.004(3)        0.911±0.022(3)        0.099±0.051(3)        0.650±0.131(3)        0.006±0.003(3)
v5b_sinkhorn          0.887±0.049(3)        0.977±0.006(3)        0.836±0.085(3)        0.032±0.084(3)        0.508±0.224(3)        0.011±0.006(3)
v6_l001                     0.889(1)              0.977(1)              0.885(1)              0.238(1)              0.536(1)              0.009(1)
v6_l01                      0.898(1)              0.976(1)              0.814(1)              0.241(1)              0.667(1)              0.007(1)
v4u_phys                    0.932(1)              0.979(1)              0.945(1)             -0.159(1)              0.772(1)              0.003(1)
v7_l001                     0.967(1)              0.988(1)              0.930(1)              0.234(1)              0.711(1)              0.005(1)
v7_l01                      0.964(1)              0.987(1)              0.910(1)              0.367(1)              0.740(1)              0.005(1)
