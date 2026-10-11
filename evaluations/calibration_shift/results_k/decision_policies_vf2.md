# Where to compile: one rule for every scenario

457 unseen circuits; 33 backends (3 chips × original + 10 calibration variants); truth = best of 5 compilations. Adaptive top-k: compile on every backend predicted within δ of the predicted best and run on the best of them. Cost = compilations per circuit; lost = mean true fidelity below the truly best backend. Models: mean over seeds.

## A. Cold start: choose among N backends

| N | exhaustive cost | model only (δ = 0): lost | cost for lost < 0.001 | random at that cost: lost | v8kdl40: δ 0 / 0.005 / 0.01 lost (cost) | v8kdv40: δ 0 / 0.005 / 0.01 lost (cost) | v8kdv40s: δ 0 / 0.005 / 0.01 lost (cost) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 0.0013 | 1.4 (v8kdv40s) | 0.0349 | 0.0011 / 0.0008 / 0.0007 (1.6) | 0.0013 / 0.0010 / 0.0008 (1.6) | 0.0013 / 0.0010 / 0.0008 (1.6) |
| 9 | 9 | 0.0023 | 3.1 (v8kdv40s) | 0.0465 | 0.0020 / 0.0016 / 0.0013 (2.3) | 0.0024 / 0.0019 / 0.0015 (2.2) | 0.0023 / 0.0017 / 0.0013 (2.1) |
| 27 | 27 | 0.0057 | 10.3 (v8kdv40s) | 0.0335 | 0.0051 / 0.0038 / 0.0031 (2.1) | 0.0058 / 0.0043 / 0.0027 (1.9) | 0.0057 / 0.0039 / 0.0028 (1.8) |
| 33 | 33 | 0.0072 | 12.0 (v8kdv40s) | 0.0295 | 0.0064 / 0.0047 / 0.0037 (2.0) | 0.0072 / 0.0052 / 0.0031 (2.0) | 0.0072 / 0.0048 / 0.0033 (1.9) |

"cost for lost < 0.001" and the random column refer to v8kdv40s; the last columns give δ = 0, 0.005, 0.01 and, in brackets, the cost at δ = 0.01.

## B. Calibration update on the 3 chips

### Local changes (shuffle, jitterA, jitterB, jitterS)

| start | policy | v8kdl40: lost (compilations) | v8kdv40: lost (compilations) | v8kdv40s: lost (compilations) |
| --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0050 (0.00) | 0.0050 (0.00) | 0.0050 (0.00) |
| truly best | switch to the model's first (1) | 0.0038 (1.00) | 0.0046 (1.00) | 0.0042 (1.00) |
| truly best | re-check where it predicts a change | 0.0014 (1.67) | 0.0014 (1.76) | 0.0016 (1.72) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0014 (1.91) | 0.0013 (1.93) | 0.0014 (1.95) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0011 (2.01) | 0.0010 (2.04) | 0.0011 (2.05) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0008 (2.16) | 0.0006 (2.19) | 0.0007 (2.19) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0067 (0.00) | 0.0076 (0.00) | 0.0071 (0.00) |
| model's pick | switch to the model's first (1) | 0.0038 (1.00) | 0.0046 (1.00) | 0.0042 (1.00) |
| model's pick | re-check where it predicts a change | 0.0023 (1.67) | 0.0029 (1.71) | 0.0028 (1.66) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0022 (1.88) | 0.0028 (1.89) | 0.0025 (1.91) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0018 (1.98) | 0.0024 (2.00) | 0.0022 (2.01) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0013 (2.13) | 0.0017 (2.16) | 0.0015 (2.16) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Moderate changes (x0.7, x0.8, x1.2, x1.3)

| start | policy | v8kdl40: lost (compilations) | v8kdv40: lost (compilations) | v8kdv40s: lost (compilations) |
| --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0025 (1.00) | 0.0036 (1.00) | 0.0032 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.49) | 0.0000 (1.66) | 0.0000 (1.59) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.81) | 0.0000 (1.83) | 0.0000 (1.84) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.92) | 0.0000 (1.93) | 0.0000 (1.95) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.11) | 0.0000 (2.11) | 0.0000 (2.13) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0023 (0.00) | 0.0034 (0.00) | 0.0027 (0.00) |
| model's pick | switch to the model's first (1) | 0.0025 (1.00) | 0.0036 (1.00) | 0.0032 (1.00) |
| model's pick | re-check where it predicts a change | 0.0016 (1.37) | 0.0027 (1.33) | 0.0022 (1.36) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0014 (1.75) | 0.0025 (1.75) | 0.0020 (1.76) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0012 (1.87) | 0.0022 (1.86) | 0.0018 (1.88) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0009 (2.07) | 0.0016 (2.05) | 0.0015 (2.09) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Large changes (x0.5, x2)

| start | policy | v8kdl40: lost (compilations) | v8kdv40: lost (compilations) | v8kdv40s: lost (compilations) |
| --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0051 (1.00) | 0.0061 (1.00) | 0.0066 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.58) | 0.0000 (1.81) | 0.0000 (1.73) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.77) | 0.0000 (1.81) | 0.0000 (1.83) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.94) | 0.0000 (1.95) | 0.0000 (1.97) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.16) | 0.0000 (2.17) | 0.0000 (2.18) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0024 (0.00) | 0.0036 (0.00) | 0.0029 (0.00) |
| model's pick | switch to the model's first (1) | 0.0051 (1.00) | 0.0061 (1.00) | 0.0066 (1.00) |
| model's pick | re-check where it predicts a change | 0.0012 (1.62) | 0.0022 (1.59) | 0.0017 (1.67) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0012 (1.76) | 0.0022 (1.77) | 0.0016 (1.79) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0011 (1.92) | 0.0017 (1.92) | 0.0014 (1.93) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0008 (2.13) | 0.0013 (2.13) | 0.0009 (2.16) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

## C. Choosing δ

Start of an update = the model's previous pick. 'Recompiled' = circuits compiled on more than one backend.

| δ | cold N=3: lost / compilations | cold N=9: lost / compilations | cold N=33: lost / compilations | update local: lost / recompiled | update moderate: lost / recompiled | update large: lost / recompiled |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0013 / 1.00 | 0.0023 / 1.00 | 0.0072 / 1.00 | 0.0032 / 33% | 0.0023 / 18% | 0.0021 / 34% |
| 0.001 | 0.0012 / 1.20 | 0.0022 / 1.12 | 0.0068 / 1.08 | 0.0030 / 47% | 0.0022 / 32% | 0.0020 / 45% |
| 0.002 | 0.0012 / 1.28 | 0.0020 / 1.22 | 0.0059 / 1.24 | 0.0028 / 50% | 0.0021 / 38% | 0.0019 / 48% |
| 0.003 | 0.0011 / 1.35 | 0.0019 / 1.37 | 0.0054 / 1.33 | 0.0027 / 52% | 0.0021 / 40% | 0.0018 / 51% |
| 0.005 | 0.0010 / 1.44 | 0.0017 / 1.63 | 0.0048 / 1.45 | 0.0025 / 56% | 0.0020 / 44% | 0.0016 / 56% |
| 0.0075 | 0.0009 / 1.52 | 0.0015 / 1.96 | 0.0037 / 1.68 | 0.0023 / 60% | 0.0019 / 48% | 0.0015 / 60% |
| 0.01 | 0.0008 / 1.57 | 0.0013 / 2.14 | 0.0033 / 1.89 | 0.0022 / 63% | 0.0018 / 52% | 0.0014 / 64% |
| 0.015 | 0.0006 / 1.67 | 0.0010 / 2.53 | 0.0024 / 2.71 | 0.0019 / 68% | 0.0016 / 61% | 0.0011 / 69% |
| 0.02 | 0.0005 / 1.77 | 0.0008 / 3.08 | 0.0014 / 4.79 | 0.0015 / 73% | 0.0015 / 67% | 0.0009 / 74% |
| 0.03 | 0.0003 / 1.91 | 0.0006 / 3.81 | 0.0010 / 9.47 | 0.0008 / 82% | 0.0010 / 78% | 0.0006 / 80% |
| 0.05 | 0.0002 / 2.08 | 0.0003 / 4.52 | 0.0006 / 12.05 | 0.0005 / 91% | 0.0006 / 90% | 0.0003 / 90% |
| 0.1 | 0.0001 / 2.37 | 0.0001 / 5.58 | 0.0004 / 16.28 | 0.0001 / 99% | 0.0002 / 99% | 0.0001 / 98% |

Model: v8kdv40s.
