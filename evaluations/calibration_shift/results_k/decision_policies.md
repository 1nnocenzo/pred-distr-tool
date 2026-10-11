# Where to compile: one rule for every scenario

457 unseen circuits; 33 backends (3 chips × original + 10 calibration variants); truth = best of 5 compilations. Adaptive top-k: compile on every backend predicted within δ of the predicted best and run on the best of them. Cost = compilations per circuit; lost = mean true fidelity below the truly best backend. Models: mean over seeds.

## A. Cold start: choose among N backends

| N | exhaustive cost | model only (δ = 0): lost | cost for lost < 0.001 | random at that cost: lost | v8a_c001: δ 0 / 0.005 / 0.01 lost (cost) | v8kd40: δ 0 / 0.005 / 0.01 lost (cost) | v8kdl40: δ 0 / 0.005 / 0.01 lost (cost) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 0.0011 | 1.3 (v8kdl40) | 0.0403 | 0.0018 / 0.0014 / 0.0012 (1.6) | 0.0014 / 0.0011 / 0.0008 (1.6) | 0.0011 / 0.0008 / 0.0007 (1.6) |
| 9 | 9 | 0.0020 | 3.3 (v8kdl40) | 0.0437 | 0.0025 / 0.0019 / 0.0015 (2.1) | 0.0021 / 0.0016 / 0.0013 (2.2) | 0.0020 / 0.0016 / 0.0013 (2.3) |
| 27 | 27 | 0.0051 | 10.6 (v8kdl40) | 0.0327 | 0.0052 / 0.0041 / 0.0033 (1.6) | 0.0042 / 0.0031 / 0.0026 (1.9) | 0.0051 / 0.0038 / 0.0031 (2.1) |
| 33 | 33 | 0.0064 | 12.4 (v8kdl40) | 0.0283 | 0.0060 / 0.0047 / 0.0038 (1.6) | 0.0049 / 0.0035 / 0.0030 (1.9) | 0.0064 / 0.0047 / 0.0037 (2.0) |

"cost for lost < 0.001" and the random column refer to v8kdl40; the last columns give δ = 0, 0.005, 0.01 and, in brackets, the cost at δ = 0.01.

## B. Calibration update on the 3 chips

### Local changes (shuffle, jitterA, jitterB, jitterS)

| start | policy | v8a_c001: lost (compilations) | v8kd40: lost (compilations) | v8kdl40: lost (compilations) |
| --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0050 (0.00) | 0.0050 (0.00) | 0.0050 (0.00) |
| truly best | switch to the model's first (1) | 0.0067 (1.00) | 0.0049 (1.00) | 0.0038 (1.00) |
| truly best | re-check where it predicts a change | 0.0021 (1.80) | 0.0016 (1.73) | 0.0014 (1.67) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0025 (1.90) | 0.0017 (1.91) | 0.0014 (1.91) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0023 (2.00) | 0.0014 (2.01) | 0.0011 (2.01) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0017 (2.16) | 0.0009 (2.16) | 0.0008 (2.16) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0074 (0.00) | 0.0062 (0.00) | 0.0067 (0.00) |
| model's pick | switch to the model's first (1) | 0.0067 (1.00) | 0.0049 (1.00) | 0.0038 (1.00) |
| model's pick | re-check where it predicts a change | 0.0040 (1.62) | 0.0026 (1.67) | 0.0023 (1.67) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0042 (1.86) | 0.0026 (1.87) | 0.0022 (1.88) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0037 (1.96) | 0.0022 (1.97) | 0.0018 (1.98) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0029 (2.13) | 0.0017 (2.12) | 0.0013 (2.13) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Moderate changes (x0.7, x0.8, x1.2, x1.3)

| start | policy | v8a_c001: lost (compilations) | v8kd40: lost (compilations) | v8kdl40: lost (compilations) |
| --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0034 (1.00) | 0.0027 (1.00) | 0.0025 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.68) | 0.0000 (1.59) | 0.0000 (1.49) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.83) | 0.0000 (1.82) | 0.0000 (1.81) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.93) | 0.0000 (1.93) | 0.0000 (1.92) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.11) | 0.0000 (2.11) | 0.0000 (2.11) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0029 (0.00) | 0.0019 (0.00) | 0.0023 (0.00) |
| model's pick | switch to the model's first (1) | 0.0034 (1.00) | 0.0027 (1.00) | 0.0025 (1.00) |
| model's pick | re-check where it predicts a change | 0.0021 (1.41) | 0.0016 (1.35) | 0.0016 (1.37) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0019 (1.76) | 0.0014 (1.74) | 0.0014 (1.75) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0016 (1.88) | 0.0012 (1.87) | 0.0012 (1.87) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0012 (2.07) | 0.0009 (2.07) | 0.0009 (2.07) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

### Large changes (x0.5, x2)

| start | policy | v8a_c001: lost (compilations) | v8kd40: lost (compilations) | v8kdl40: lost (compilations) |
| --- | --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0059 (1.00) | 0.0051 (1.00) | 0.0051 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.95) | 0.0000 (1.82) | 0.0000 (1.58) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.81) | 0.0000 (1.78) | 0.0000 (1.77) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.93) | 0.0000 (1.97) | 0.0000 (1.94) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.13) | 0.0000 (2.18) | 0.0000 (2.16) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0028 (0.00) | 0.0021 (0.00) | 0.0024 (0.00) |
| model's pick | switch to the model's first (1) | 0.0059 (1.00) | 0.0051 (1.00) | 0.0051 (1.00) |
| model's pick | re-check where it predicts a change | 0.0013 (1.84) | 0.0010 (1.75) | 0.0012 (1.62) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0015 (1.79) | 0.0010 (1.79) | 0.0012 (1.76) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0013 (1.92) | 0.0009 (1.95) | 0.0011 (1.92) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0009 (2.12) | 0.0008 (2.15) | 0.0008 (2.13) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) | 0.0000 (3.00) |

## C. Choosing δ

Start of an update = the model's previous pick. 'Recompiled' = circuits compiled on more than one backend.

| δ | cold N=3: lost / compilations | cold N=9: lost / compilations | cold N=33: lost / compilations | update local: lost / recompiled | update moderate: lost / recompiled | update large: lost / recompiled |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0011 / 1.00 | 0.0020 / 1.00 | 0.0064 / 1.00 | 0.0027 / 34% | 0.0016 / 19% | 0.0014 / 31% |
| 0.001 | 0.0010 / 1.20 | 0.0019 / 1.12 | 0.0059 / 1.07 | 0.0026 / 46% | 0.0016 / 33% | 0.0014 / 43% |
| 0.002 | 0.0010 / 1.29 | 0.0018 / 1.24 | 0.0056 / 1.16 | 0.0025 / 49% | 0.0015 / 37% | 0.0014 / 46% |
| 0.003 | 0.0009 / 1.36 | 0.0017 / 1.42 | 0.0052 / 1.24 | 0.0024 / 51% | 0.0015 / 40% | 0.0013 / 49% |
| 0.005 | 0.0008 / 1.46 | 0.0016 / 1.75 | 0.0047 / 1.41 | 0.0022 / 54% | 0.0014 / 44% | 0.0012 / 53% |
| 0.0075 | 0.0008 / 1.54 | 0.0014 / 2.07 | 0.0042 / 1.62 | 0.0021 / 58% | 0.0013 / 48% | 0.0011 / 58% |
| 0.01 | 0.0007 / 1.59 | 0.0013 / 2.30 | 0.0037 / 2.02 | 0.0018 / 62% | 0.0012 / 52% | 0.0011 / 61% |
| 0.015 | 0.0005 / 1.70 | 0.0010 / 2.81 | 0.0028 / 3.84 | 0.0016 / 67% | 0.0010 / 59% | 0.0009 / 66% |
| 0.02 | 0.0004 / 1.78 | 0.0008 / 3.26 | 0.0022 / 6.31 | 0.0013 / 72% | 0.0009 / 66% | 0.0008 / 71% |
| 0.03 | 0.0003 / 1.91 | 0.0005 / 3.88 | 0.0012 / 9.96 | 0.0009 / 79% | 0.0007 / 77% | 0.0006 / 78% |
| 0.05 | 0.0001 / 2.07 | 0.0003 / 4.53 | 0.0006 / 12.44 | 0.0005 / 89% | 0.0004 / 89% | 0.0003 / 88% |
| 0.1 | 0.0000 / 2.37 | 0.0001 / 5.60 | 0.0001 / 16.67 | 0.0001 / 99% | 0.0000 / 99% | 0.0000 / 99% |

Model: v8kdl40.
