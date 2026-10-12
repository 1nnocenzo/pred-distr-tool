# Where to compile: one rule for every scenario

457 unseen circuits; 33 backends (3 chips × original + 10 calibration variants); truth = best of 5 compilations. Adaptive top-k: compile on every backend predicted within δ of the predicted best and run on the best of them. Cost = compilations per circuit; lost = mean true fidelity below the truly best backend. Models: mean over seeds.

## A. Cold start: choose among N backends

| N | exhaustive cost | model only (δ = 0): lost | cost for lost < 0.001 | random at that cost: lost | v8kdc10: δ 0 / 0.005 / 0.01 lost (cost) | v8kde10: δ 0 / 0.005 / 0.01 lost (cost) | v8kda10: δ 0 / 0.005 / 0.01 lost (cost) | v8kda10_ens: δ 0 / 0.005 / 0.01 lost (cost) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 0.0011 | 1.3 (v8kda10_ens) | 0.0399 | 0.0012 / 0.0009 / 0.0007 (1.6) | 0.0011 / 0.0009 / 0.0007 (1.6) | 0.0012 / 0.0009 / 0.0007 (1.6) | 0.0011 / 0.0008 / 0.0006 (1.6) |
| 9 | 9 | 0.0021 | 3.5 (v8kda10_ens) | 0.0409 | 0.0021 / 0.0016 / 0.0013 (2.3) | 0.0023 / 0.0018 / 0.0014 (2.3) | 0.0023 / 0.0018 / 0.0015 (2.4) | 0.0021 / 0.0017 / 0.0012 (2.4) |
| 27 | 27 | 0.0046 | 10.6 (v8kda10_ens) | 0.0326 | 0.0046 / 0.0033 / 0.0024 (2.1) | 0.0058 / 0.0044 / 0.0037 (2.2) | 0.0047 / 0.0037 / 0.0030 (2.3) | 0.0046 / 0.0035 / 0.0026 (2.3) |
| 33 | 33 | 0.0057 | 12.4 (v8kda10_ens) | 0.0284 | 0.0054 / 0.0038 / 0.0027 (2.1) | 0.0072 / 0.0053 / 0.0044 (2.2) | 0.0058 / 0.0045 / 0.0037 (2.2) | 0.0057 / 0.0043 / 0.0031 (2.3) |

"cost for lost < 0.001" and the random column refer to v8kda10_ens; the last columns give δ = 0, 0.005, 0.01 and, in brackets, the cost at δ = 0.01.

## B. Calibration update on the 3 chips

### Local changes (shuffle, jitterA, jitterB, jitterS)

| start | policy | v8kdc10: lost (compilations) | v8kde10: lost (compilations) | v8kda10: lost (compilations) | v8kda10_ens: lost (compilations) |
| --- | --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0050 (0.00) | 0.0050 (0.00) | 0.0050 (0.00) | 0.0050 (0.00) |
| truly best | switch to the model's first (1) | 0.0037 (1.00) | 0.0040 (1.00) | 0.0047 (1.00) | 0.0040 (1.00) |
| truly best | re-check where it predicts a change | 0.0013 (1.69) | 0.0012 (1.76) | 0.0012 (1.74) | 0.0012 (1.69) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0013 (1.91) | 0.0012 (1.95) | 0.0013 (1.95) | 0.0013 (1.95) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0011 (2.01) | 0.0010 (2.04) | 0.0011 (2.04) | 0.0010 (2.05) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0007 (2.16) | 0.0007 (2.18) | 0.0008 (2.19) | 0.0006 (2.20) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0068 (0.00) | 0.0066 (0.00) | 0.0067 (0.00) | 0.0064 (0.00) |
| model's pick | switch to the model's first (1) | 0.0037 (1.00) | 0.0040 (1.00) | 0.0047 (1.00) | 0.0040 (1.00) |
| model's pick | re-check where it predicts a change | 0.0023 (1.65) | 0.0022 (1.72) | 0.0028 (1.66) | 0.0024 (1.63) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0022 (1.87) | 0.0022 (1.90) | 0.0029 (1.88) | 0.0024 (1.90) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0019 (1.97) | 0.0018 (2.00) | 0.0025 (1.98) | 0.0019 (2.00) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0013 (2.13) | 0.0013 (2.15) | 0.0016 (2.15) | 0.0013 (2.17) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Moderate changes (x0.7, x0.8, x1.2, x1.3)

| start | policy | v8kdc10: lost (compilations) | v8kde10: lost (compilations) | v8kda10: lost (compilations) | v8kda10_ens: lost (compilations) |
| --- | --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0027 (1.00) | 0.0030 (1.00) | 0.0033 (1.00) | 0.0030 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.50) | 0.0000 (1.58) | 0.0000 (1.54) | 0.0000 (1.47) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.82) | 0.0000 (1.85) | 0.0000 (1.85) | 0.0000 (1.84) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.92) | 0.0000 (1.95) | 0.0000 (1.96) | 0.0000 (1.96) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.10) | 0.0000 (2.13) | 0.0000 (2.14) | 0.0000 (2.14) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0025 (0.00) | 0.0023 (0.00) | 0.0027 (0.00) | 0.0021 (0.00) |
| model's pick | switch to the model's first (1) | 0.0027 (1.00) | 0.0030 (1.00) | 0.0033 (1.00) | 0.0030 (1.00) |
| model's pick | re-check where it predicts a change | 0.0018 (1.37) | 0.0018 (1.38) | 0.0021 (1.38) | 0.0017 (1.32) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0017 (1.75) | 0.0015 (1.77) | 0.0019 (1.76) | 0.0015 (1.77) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0014 (1.86) | 0.0013 (1.89) | 0.0016 (1.88) | 0.0012 (1.89) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0008 (2.06) | 0.0009 (2.08) | 0.0011 (2.09) | 0.0009 (2.10) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Large changes (x0.5, x2)

| start | policy | v8kdc10: lost (compilations) | v8kde10: lost (compilations) | v8kda10: lost (compilations) | v8kda10_ens: lost (compilations) |
| --- | --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0049 (1.00) | 0.0049 (1.00) | 0.0051 (1.00) | 0.0042 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.57) | 0.0000 (1.60) | 0.0000 (1.62) | 0.0000 (1.54) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.78) | 0.0000 (1.80) | 0.0000 (1.81) | 0.0000 (1.79) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.96) | 0.0000 (1.97) | 0.0000 (1.98) | 0.0000 (2.00) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.17) | 0.0000 (2.18) | 0.0000 (2.21) | 0.0000 (2.22) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0026 (0.00) | 0.0024 (0.00) | 0.0026 (0.00) | 0.0019 (0.00) |
| model's pick | switch to the model's first (1) | 0.0049 (1.00) | 0.0049 (1.00) | 0.0051 (1.00) | 0.0042 (1.00) |
| model's pick | re-check where it predicts a change | 0.0014 (1.59) | 0.0012 (1.68) | 0.0016 (1.60) | 0.0012 (1.54) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0013 (1.77) | 0.0013 (1.79) | 0.0016 (1.78) | 0.0013 (1.75) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0011 (1.94) | 0.0011 (1.95) | 0.0014 (1.94) | 0.0011 (1.96) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0007 (2.15) | 0.0009 (2.15) | 0.0011 (2.16) | 0.0008 (2.18) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

## C. Choosing δ

Start of an update = the model's previous pick. 'Recompiled' = circuits compiled on more than one backend.

| δ | cold N=3: lost / compilations | cold N=9: lost / compilations | cold N=33: lost / compilations | update local: lost / recompiled | update moderate: lost / recompiled | update large: lost / recompiled |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0011 / 1.00 | 0.0021 / 1.00 | 0.0057 / 1.00 | 0.0028 / 32% | 0.0017 / 16% | 0.0014 / 27% |
| 0.001 | 0.0010 / 1.20 | 0.0020 / 1.12 | 0.0054 / 1.11 | 0.0027 / 46% | 0.0017 / 35% | 0.0014 / 41% |
| 0.002 | 0.0009 / 1.30 | 0.0019 / 1.28 | 0.0049 / 1.19 | 0.0026 / 50% | 0.0017 / 39% | 0.0014 / 44% |
| 0.003 | 0.0009 / 1.38 | 0.0018 / 1.46 | 0.0048 / 1.25 | 0.0025 / 52% | 0.0016 / 41% | 0.0014 / 47% |
| 0.005 | 0.0008 / 1.48 | 0.0017 / 1.83 | 0.0043 / 1.37 | 0.0024 / 56% | 0.0015 / 45% | 0.0013 / 52% |
| 0.0075 | 0.0007 / 1.55 | 0.0014 / 2.12 | 0.0034 / 1.64 | 0.0021 / 60% | 0.0014 / 50% | 0.0011 / 58% |
| 0.01 | 0.0006 / 1.60 | 0.0012 / 2.37 | 0.0031 / 2.26 | 0.0019 / 64% | 0.0012 / 55% | 0.0011 / 62% |
| 0.015 | 0.0005 / 1.72 | 0.0010 / 3.04 | 0.0023 / 5.24 | 0.0017 / 69% | 0.0011 / 63% | 0.0010 / 69% |
| 0.02 | 0.0004 / 1.80 | 0.0008 / 3.45 | 0.0019 / 7.79 | 0.0013 / 75% | 0.0009 / 69% | 0.0008 / 74% |
| 0.03 | 0.0002 / 1.92 | 0.0005 / 3.96 | 0.0014 / 9.96 | 0.0007 / 83% | 0.0006 / 79% | 0.0007 / 83% |
| 0.05 | 0.0001 / 2.07 | 0.0002 / 4.55 | 0.0005 / 12.40 | 0.0003 / 92% | 0.0003 / 93% | 0.0003 / 93% |
| 0.1 | 0.0000 / 2.36 | 0.0000 / 5.61 | 0.0000 / 16.85 | 0.0000 / 99% | 0.0000 / 99% | 0.0000 / 99% |

Model: v8kda10_ens.
