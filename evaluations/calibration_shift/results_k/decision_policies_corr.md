# Where to compile: one rule for every scenario

457 unseen circuits; 33 backends (3 chips × original + 10 calibration variants); truth = best of 5 compilations. Adaptive top-k: compile on every backend predicted within δ of the predicted best and run on the best of them. Cost = compilations per circuit; lost = mean true fidelity below the truly best backend. Models: mean over seeds.

## A. Cold start: choose among N backends

| N | exhaustive cost | model only (δ = 0): lost | cost for lost < 0.001 | random at that cost: lost | v8kdl40: δ 0 / 0.005 / 0.01 lost (cost) | v8kdc10: δ 0 / 0.005 / 0.01 lost (cost) | v8kdl40_ens: δ 0 / 0.005 / 0.01 lost (cost) | v8kdc10_ens: δ 0 / 0.005 / 0.01 lost (cost) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 0.0010 | 1.0 (v8kdc10_ens) | 0.0518 | 0.0011 / 0.0008 / 0.0007 (1.6) | 0.0012 / 0.0009 / 0.0007 (1.6) | 0.0009 / 0.0007 / 0.0005 (1.6) | 0.0010 / 0.0007 / 0.0005 (1.6) |
| 9 | 9 | 0.0016 | 2.3 (v8kdc10_ens) | 0.0614 | 0.0020 / 0.0016 / 0.0013 (2.3) | 0.0021 / 0.0016 / 0.0013 (2.3) | 0.0016 / 0.0011 / 0.0009 (2.3) | 0.0016 / 0.0011 / 0.0009 (2.3) |
| 27 | 27 | 0.0041 | 11.1 (v8kdc10_ens) | 0.0310 | 0.0051 / 0.0038 / 0.0031 (2.1) | 0.0046 / 0.0033 / 0.0024 (2.1) | 0.0044 / 0.0030 / 0.0023 (2.0) | 0.0041 / 0.0026 / 0.0020 (2.2) |
| 33 | 33 | 0.0049 | 13.1 (v8kdc10_ens) | 0.0263 | 0.0064 / 0.0047 / 0.0037 (2.0) | 0.0054 / 0.0038 / 0.0027 (2.1) | 0.0054 / 0.0037 / 0.0028 (2.0) | 0.0049 / 0.0030 / 0.0023 (2.1) |

"cost for lost < 0.001" and the random column refer to v8kdc10_ens; the last columns give δ = 0, 0.005, 0.01 and, in brackets, the cost at δ = 0.01.

## B. Calibration update on the 3 chips

### Local changes (shuffle, jitterA, jitterB, jitterS)

| start | policy | v8kdl40: lost (compilations) | v8kdc10: lost (compilations) | v8kdl40_ens: lost (compilations) | v8kdc10_ens: lost (compilations) |
| --- | --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0050 (0.00) | 0.0050 (0.00) | 0.0050 (0.00) | 0.0050 (0.00) |
| truly best | switch to the model's first (1) | 0.0038 (1.00) | 0.0037 (1.00) | 0.0033 (1.00) | 0.0032 (1.00) |
| truly best | re-check where it predicts a change | 0.0014 (1.67) | 0.0013 (1.69) | 0.0015 (1.61) | 0.0013 (1.63) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0014 (1.91) | 0.0013 (1.91) | 0.0013 (1.91) | 0.0012 (1.91) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0011 (2.01) | 0.0011 (2.01) | 0.0011 (2.00) | 0.0010 (2.00) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0008 (2.16) | 0.0007 (2.16) | 0.0007 (2.16) | 0.0007 (2.16) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0067 (0.00) | 0.0068 (0.00) | 0.0063 (0.00) | 0.0064 (0.00) |
| model's pick | switch to the model's first (1) | 0.0038 (1.00) | 0.0037 (1.00) | 0.0033 (1.00) | 0.0032 (1.00) |
| model's pick | re-check where it predicts a change | 0.0023 (1.67) | 0.0023 (1.65) | 0.0021 (1.63) | 0.0020 (1.63) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0022 (1.88) | 0.0022 (1.87) | 0.0018 (1.88) | 0.0019 (1.87) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0018 (1.98) | 0.0019 (1.97) | 0.0015 (1.98) | 0.0017 (1.97) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0013 (2.13) | 0.0013 (2.13) | 0.0010 (2.15) | 0.0012 (2.14) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Moderate changes (x0.7, x0.8, x1.2, x1.3)

| start | policy | v8kdl40: lost (compilations) | v8kdc10: lost (compilations) | v8kdl40_ens: lost (compilations) | v8kdc10_ens: lost (compilations) |
| --- | --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0025 (1.00) | 0.0027 (1.00) | 0.0019 (1.00) | 0.0020 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.49) | 0.0000 (1.50) | 0.0000 (1.42) | 0.0000 (1.41) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.81) | 0.0000 (1.82) | 0.0000 (1.80) | 0.0000 (1.80) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.92) | 0.0000 (1.92) | 0.0000 (1.91) | 0.0000 (1.91) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.11) | 0.0000 (2.10) | 0.0000 (2.11) | 0.0000 (2.09) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0023 (0.00) | 0.0025 (0.00) | 0.0017 (0.00) | 0.0019 (0.00) |
| model's pick | switch to the model's first (1) | 0.0025 (1.00) | 0.0027 (1.00) | 0.0019 (1.00) | 0.0020 (1.00) |
| model's pick | re-check where it predicts a change | 0.0016 (1.37) | 0.0018 (1.37) | 0.0014 (1.33) | 0.0016 (1.31) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0014 (1.75) | 0.0017 (1.75) | 0.0012 (1.74) | 0.0013 (1.74) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0012 (1.87) | 0.0014 (1.86) | 0.0010 (1.86) | 0.0010 (1.87) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0009 (2.07) | 0.0008 (2.06) | 0.0004 (2.09) | 0.0007 (2.06) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Large changes (x0.5, x2)

| start | policy | v8kdl40: lost (compilations) | v8kdc10: lost (compilations) | v8kdl40_ens: lost (compilations) | v8kdc10_ens: lost (compilations) |
| --- | --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0051 (1.00) | 0.0049 (1.00) | 0.0045 (1.00) | 0.0042 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.58) | 0.0000 (1.57) | 0.0000 (1.52) | 0.0000 (1.48) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.77) | 0.0000 (1.78) | 0.0000 (1.77) | 0.0000 (1.79) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.94) | 0.0000 (1.96) | 0.0000 (1.94) | 0.0000 (1.96) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.16) | 0.0000 (2.17) | 0.0000 (2.17) | 0.0000 (2.17) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0024 (0.00) | 0.0026 (0.00) | 0.0019 (0.00) | 0.0020 (0.00) |
| model's pick | switch to the model's first (1) | 0.0051 (1.00) | 0.0049 (1.00) | 0.0045 (1.00) | 0.0042 (1.00) |
| model's pick | re-check where it predicts a change | 0.0012 (1.62) | 0.0014 (1.59) | 0.0012 (1.57) | 0.0013 (1.50) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0012 (1.76) | 0.0013 (1.77) | 0.0011 (1.75) | 0.0012 (1.76) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0011 (1.92) | 0.0011 (1.94) | 0.0010 (1.90) | 0.0008 (1.93) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0008 (2.13) | 0.0007 (2.15) | 0.0006 (2.15) | 0.0006 (2.15) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

## C. Choosing δ

Start of an update = the model's previous pick. 'Recompiled' = circuits compiled on more than one backend.

| δ | cold N=3: lost / compilations | cold N=9: lost / compilations | cold N=33: lost / compilations | update local: lost / recompiled | update moderate: lost / recompiled | update large: lost / recompiled |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0010 / 1.00 | 0.0016 / 1.00 | 0.0049 / 1.00 | 0.0024 / 32% | 0.0016 / 15% | 0.0014 / 25% |
| 0.001 | 0.0009 / 1.20 | 0.0015 / 1.11 | 0.0047 / 1.08 | 0.0022 / 44% | 0.0015 / 32% | 0.0014 / 40% |
| 0.002 | 0.0008 / 1.29 | 0.0015 / 1.23 | 0.0044 / 1.16 | 0.0021 / 48% | 0.0015 / 36% | 0.0013 / 43% |
| 0.003 | 0.0008 / 1.37 | 0.0014 / 1.41 | 0.0043 / 1.26 | 0.0020 / 50% | 0.0014 / 38% | 0.0013 / 46% |
| 0.005 | 0.0007 / 1.47 | 0.0011 / 1.76 | 0.0030 / 1.45 | 0.0019 / 53% | 0.0013 / 42% | 0.0012 / 51% |
| 0.0075 | 0.0006 / 1.54 | 0.0010 / 2.06 | 0.0027 / 1.69 | 0.0017 / 57% | 0.0010 / 47% | 0.0010 / 56% |
| 0.01 | 0.0005 / 1.60 | 0.0009 / 2.34 | 0.0023 / 2.14 | 0.0017 / 61% | 0.0010 / 51% | 0.0008 / 59% |
| 0.015 | 0.0004 / 1.71 | 0.0006 / 2.94 | 0.0016 / 4.26 | 0.0014 / 67% | 0.0008 / 58% | 0.0006 / 65% |
| 0.02 | 0.0003 / 1.81 | 0.0005 / 3.52 | 0.0013 / 8.20 | 0.0012 / 72% | 0.0007 / 64% | 0.0006 / 68% |
| 0.03 | 0.0002 / 1.93 | 0.0003 / 4.03 | 0.0008 / 10.62 | 0.0008 / 80% | 0.0004 / 76% | 0.0003 / 77% |
| 0.05 | 0.0001 / 2.08 | 0.0001 / 4.62 | 0.0003 / 13.13 | 0.0003 / 90% | 0.0002 / 87% | 0.0002 / 90% |
| 0.1 | 0.0000 / 2.38 | 0.0000 / 5.66 | 0.0000 / 17.18 | 0.0000 / 99% | 0.0000 / 99% | 0.0000 / 99% |

Model: v8kdc10_ens.
