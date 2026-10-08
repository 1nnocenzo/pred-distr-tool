## Compiler noise floor (recompiling on the original devices vs. the dataset)
EQE1_Top     R2 0.9973  MAE 0.0057
EQE1_Bottom  R2 0.9939  MAE 0.0082
QExa20       R2 0.9989  MAE 0.0039

## Accuracy per variant (R2 / MAE) and tracking of the change vs. orig
tracking = corr(predicted log F(variant) - log F(orig), true ...), on circuits with F > 0.01

### EQE1_Top
variant                            v3_xattn                           v4_phys                       v5_sinkhorn
orig                     R2 0.995 MAE 0.012                R2 0.987 MAE 0.015                R2 0.995 MAE 0.011
x0.5           R2 0.836 MAE 0.110 trk -0.62      R2 0.991 MAE 0.023 trk +0.95      R2 0.990 MAE 0.025 trk +0.95
x2             R2 0.724 MAE 0.107 trk -0.00      R2 0.974 MAE 0.015 trk +0.94      R2 0.990 MAE 0.015 trk +0.93
shuffle        R2 0.982 MAE 0.025 trk -0.10      R2 0.799 MAE 0.100 trk +0.22      R2 0.981 MAE 0.026 trk -0.07
jitterA        R2 0.984 MAE 0.023 trk -0.18      R2 0.976 MAE 0.027 trk +0.06      R2 0.984 MAE 0.023 trk -0.04
jitterB        R2 0.990 MAE 0.018 trk +0.13      R2 0.970 MAE 0.035 trk -0.35      R2 0.991 MAE 0.018 trk +0.33

### EQE1_Bottom
variant                            v3_xattn                           v4_phys                       v5_sinkhorn
orig                     R2 0.991 MAE 0.016                R2 0.985 MAE 0.017                R2 0.993 MAE 0.013
x0.5           R2 0.829 MAE 0.114 trk -0.01      R2 0.988 MAE 0.025 trk +0.91      R2 0.988 MAE 0.026 trk +0.92
x2             R2 0.728 MAE 0.108 trk -0.69      R2 0.978 MAE 0.016 trk +0.93      R2 0.979 MAE 0.023 trk +0.89
shuffle        R2 0.977 MAE 0.031 trk +0.01      R2 0.658 MAE 0.136 trk +0.65      R2 0.932 MAE 0.053 trk +0.45
jitterA        R2 0.992 MAE 0.017 trk -0.08      R2 0.984 MAE 0.022 trk +0.24      R2 0.991 MAE 0.018 trk +0.18
jitterB        R2 0.989 MAE 0.018 trk -0.10      R2 0.986 MAE 0.019 trk +0.30      R2 0.991 MAE 0.017 trk +0.02

### QExa20
variant                            v3_xattn                           v4_phys                       v5_sinkhorn
orig                     R2 0.994 MAE 0.012                R2 0.992 MAE 0.012                R2 0.998 MAE 0.009
x0.5           R2 0.796 MAE 0.123 trk -0.21      R2 0.993 MAE 0.018 trk +0.98      R2 0.995 MAE 0.018 trk +0.98
x2             R2 0.666 MAE 0.106 trk +0.12      R2 0.518 MAE 0.109 trk +0.94      R2 0.995 MAE 0.009 trk +0.94
shuffle        R2 0.938 MAE 0.048 trk +0.23      R2 0.788 MAE 0.087 trk +0.34      R2 0.908 MAE 0.056 trk +0.06
jitterA        R2 0.985 MAE 0.024 trk +0.11      R2 0.436 MAE 0.171 trk -0.04      R2 0.994 MAE 0.014 trk +0.41
jitterB        R2 0.990 MAE 0.018 trk +0.11      R2 0.985 MAE 0.023 trk -0.04      R2 0.992 MAE 0.017 trk -0.03

## Device choice among all variants: accuracy / mean regret
v3_xattn     accuracy 0.079  regret 0.1559
v4_phys      accuracy 0.694  regret 0.0040
v5_sinkhorn  accuracy 0.654  regret 0.0048
