# Where to compile: one rule for every scenario

457 unseen circuits; 33 backends (3 chips × original + 10 calibration variants); truth = best of 5 compilations. Adaptive top-k: compile on every backend predicted within δ of the predicted best and run on the best of them. Cost = compilations per circuit; lost = mean true fidelity below the truly best backend. Models: mean over seeds.

## A. Cold start: choose among N backends

| N | exhaustive cost | model only (δ = 0): lost | cost for lost < 0.001 | random at that cost: lost | v8kdl40: δ 0 / 0.005 / 0.01 lost (cost) | v8kdl40_ens: δ 0 / 0.005 / 0.01 lost (cost) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 0.0009 | 1.0 (v8kdl40_ens) | 0.0518 | 0.0011 / 0.0008 / 0.0007 (1.6) | 0.0009 / 0.0007 / 0.0005 (1.6) |
| 9 | 9 | 0.0016 | 2.3 (v8kdl40_ens) | 0.0623 | 0.0020 / 0.0016 / 0.0013 (2.3) | 0.0016 / 0.0011 / 0.0009 (2.3) |
| 27 | 27 | 0.0044 | 10.8 (v8kdl40_ens) | 0.0319 | 0.0051 / 0.0038 / 0.0031 (2.1) | 0.0044 / 0.0030 / 0.0023 (2.0) |
| 33 | 33 | 0.0054 | 12.8 (v8kdl40_ens) | 0.0275 | 0.0064 / 0.0047 / 0.0037 (2.0) | 0.0054 / 0.0037 / 0.0028 (2.0) |

"cost for lost < 0.001" and the random column refer to v8kdl40_ens; the last columns give δ = 0, 0.005, 0.01 and, in brackets, the cost at δ = 0.01.

## B. Calibration update on the 3 chips

### Local changes (shuffle, jitterA, jitterB, jitterS)

| start | policy | v8kdl40: lost (compilations) | v8kdl40_ens: lost (compilations) |
| --- | --- | --- | --- |
| truly best | keep (0) | 0.0050 (0.00) | 0.0050 (0.00) |
| truly best | switch to the model's first (1) | 0.0038 (1.00) | 0.0033 (1.00) |
| truly best | re-check where it predicts a change | 0.0014 (1.67) | 0.0015 (1.61) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0014 (1.91) | 0.0013 (1.91) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0011 (2.01) | 0.0011 (2.00) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0008 (2.16) | 0.0007 (2.16) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0067 (0.00) | 0.0063 (0.00) |
| model's pick | switch to the model's first (1) | 0.0038 (1.00) | 0.0033 (1.00) |
| model's pick | re-check where it predicts a change | 0.0023 (1.67) | 0.0021 (1.63) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0022 (1.88) | 0.0018 (1.88) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0018 (1.98) | 0.0015 (1.98) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0013 (2.13) | 0.0010 (2.15) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) |

### Moderate changes (x0.7, x0.8, x1.2, x1.3)

| start | policy | v8kdl40: lost (compilations) | v8kdl40_ens: lost (compilations) |
| --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0025 (1.00) | 0.0019 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.49) | 0.0000 (1.42) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.81) | 0.0000 (1.80) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.92) | 0.0000 (1.91) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.11) | 0.0000 (2.11) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0023 (0.00) | 0.0017 (0.00) |
| model's pick | switch to the model's first (1) | 0.0025 (1.00) | 0.0019 (1.00) |
| model's pick | re-check where it predicts a change | 0.0016 (1.37) | 0.0014 (1.33) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0014 (1.75) | 0.0012 (1.74) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0012 (1.87) | 0.0010 (1.86) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0009 (2.07) | 0.0004 (2.09) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) |

### Large changes (x0.5, x2)

| start | policy | v8kdl40: lost (compilations) | v8kdl40_ens: lost (compilations) |
| --- | --- | --- | --- |
| truly best | keep (0) | 0.0000 (0.00) | 0.0000 (0.00) |
| truly best | switch to the model's first (1) | 0.0051 (1.00) | 0.0045 (1.00) |
| truly best | re-check where it predicts a change | 0.0000 (1.58) | 0.0000 (1.52) |
| truly best | adaptive top-k, δ 0.005, current included | 0.0000 (1.77) | 0.0000 (1.77) |
| truly best | adaptive top-k, δ 0.01, current included | 0.0000 (1.94) | 0.0000 (1.94) |
| truly best | adaptive top-k, δ 0.02, current included | 0.0000 (2.16) | 0.0000 (2.17) |
| truly best | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) |
| model's pick | keep (0) | 0.0024 (0.00) | 0.0019 (0.00) |
| model's pick | switch to the model's first (1) | 0.0051 (1.00) | 0.0045 (1.00) |
| model's pick | re-check where it predicts a change | 0.0012 (1.62) | 0.0012 (1.57) |
| model's pick | adaptive top-k, δ 0.005, current included | 0.0012 (1.76) | 0.0011 (1.75) |
| model's pick | adaptive top-k, δ 0.01, current included | 0.0011 (1.92) | 0.0010 (1.90) |
| model's pick | adaptive top-k, δ 0.02, current included | 0.0008 (2.13) | 0.0006 (2.15) |
| model's pick | re-check all (3) | 0.0000 (3.00) | 0.0000 (3.00) |

## C. Choosing δ

Start of an update = the model's previous pick. 'Recompiled' = circuits compiled on more than one backend.

| δ | cold N=3: lost / compilations | cold N=9: lost / compilations | cold N=33: lost / compilations | update local: lost / recompiled | update moderate: lost / recompiled | update large: lost / recompiled |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0009 / 1.00 | 0.0016 / 1.00 | 0.0054 / 1.00 | 0.0024 / 31% | 0.0014 / 16% | 0.0014 / 29% |
| 0.001 | 0.0008 / 1.20 | 0.0015 / 1.12 | 0.0053 / 1.08 | 0.0023 / 44% | 0.0014 / 32% | 0.0014 / 42% |
| 0.002 | 0.0008 / 1.29 | 0.0014 / 1.25 | 0.0047 / 1.15 | 0.0021 / 48% | 0.0013 / 36% | 0.0013 / 44% |
| 0.003 | 0.0007 / 1.37 | 0.0013 / 1.42 | 0.0044 / 1.22 | 0.0021 / 49% | 0.0013 / 38% | 0.0012 / 46% |
| 0.005 | 0.0007 / 1.47 | 0.0011 / 1.77 | 0.0037 / 1.40 | 0.0018 / 53% | 0.0012 / 42% | 0.0011 / 51% |
| 0.0075 | 0.0006 / 1.55 | 0.0010 / 2.09 | 0.0033 / 1.61 | 0.0015 / 57% | 0.0012 / 46% | 0.0010 / 56% |
| 0.01 | 0.0005 / 1.60 | 0.0009 / 2.30 | 0.0028 / 1.99 | 0.0015 / 60% | 0.0010 / 50% | 0.0010 / 59% |
| 0.015 | 0.0004 / 1.70 | 0.0007 / 2.84 | 0.0021 / 3.72 | 0.0012 / 67% | 0.0007 / 59% | 0.0009 / 65% |
| 0.02 | 0.0003 / 1.79 | 0.0006 / 3.34 | 0.0013 / 7.15 | 0.0010 / 72% | 0.0004 / 67% | 0.0006 / 71% |
| 0.03 | 0.0002 / 1.92 | 0.0003 / 3.89 | 0.0009 / 9.96 | 0.0005 / 81% | 0.0003 / 77% | 0.0004 / 80% |
| 0.05 | 0.0001 / 2.08 | 0.0002 / 4.57 | 0.0001 / 12.77 | 0.0002 / 91% | 0.0001 / 90% | 0.0001 / 88% |
| 0.1 | 0.0000 / 2.37 | 0.0000 / 5.62 | 0.0000 / 16.90 | 0.0000 / 99% | 0.0000 / 99% | 0.0000 / 99% |

Model: v8kdl40_ens.
