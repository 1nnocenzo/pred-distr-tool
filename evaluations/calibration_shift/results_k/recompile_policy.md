# Recompile only when needed

457 unseen circuits × 3 devices; each circuit starts on the device that is truly best under the old calibration. Fidelity lost = true best F under the new calibration − true F of the device used. Re-checking a circuit = compiling it on every device again. Models: mean over the seeds listed.

## Local calibration changes (shuffle, jitterA, jitterB, jitterS)

The best device changes in 29% of (circuit, change). Keeping the old device loses 0.0050 of F on average.

| policy | circuits re-checked | fidelity lost | vs keeping |
| --- | --- | --- | --- |
| keep the old device | 0% | 0.0050 | – |
| re-check all | 100% | 0 | −100% |
| v8a_c001: switch to its new first device (no re-check) | 0% | 0.0067 | +35% |
| v8a_c001: switch only if predicted gain > 0.005 (moves 15%) | 0% | 0.0064 | +29% |
| v8a_c001: switch only if predicted gain > 0.01 (moves 12%) | 0% | 0.0060 | +21% |
| v8a_c001: switch only if predicted gain > 0.02 (moves 8%) | 0% | 0.0056 | +14% |
| v8a_c001: re-check where it predicts a change | 40% | 0.0021 | -57% |
| v8kd40: switch to its new first device (no re-check) | 0% | 0.0049 | -2% |
| v8kd40: switch only if predicted gain > 0.005 (moves 17%) | 0% | 0.0045 | -9% |
| v8kd40: switch only if predicted gain > 0.01 (moves 13%) | 0% | 0.0043 | -13% |
| v8kd40: switch only if predicted gain > 0.02 (moves 9%) | 0% | 0.0042 | -14% |
| v8kd40: re-check where it predicts a change | 37% | 0.0016 | -68% |
| v8kdl40: switch to its new first device (no re-check) | 0% | 0.0038 | -23% |
| v8kdl40: switch only if predicted gain > 0.005 (moves 15%) | 0% | 0.0037 | -26% |
| v8kdl40: switch only if predicted gain > 0.01 (moves 11%) | 0% | 0.0036 | -27% |
| v8kdl40: switch only if predicted gain > 0.02 (moves 7%) | 0% | 0.0036 | -27% |
| v8kdl40: re-check where it predicts a change | 33% | 0.0014 | -71% |

Model-guided re-check of a fixed share of circuits (predicted changes first, then the smallest predicted margins), vs random:

| share re-checked | random | v8a_c001 | v8kd40 | v8kdl40 |
| --- | --- | --- | --- | --- |
| 5% | 0.0047 | 0.0045 | 0.0045 | 0.0044 |
| 10% | 0.0045 | 0.0042 | 0.0041 | 0.0037 |
| 20% | 0.0039 | 0.0034 | 0.0031 | 0.0028 |
| 30% | 0.0035 | 0.0028 | 0.0021 | 0.0016 |
| 50% | 0.0025 | 0.0020 | 0.0016 | 0.0014 |

## Moderate calibration changes (x0.7, x0.8, x1.2, x1.3)

The best device changes in 2% of (circuit, change). Keeping the old device loses 0.0000 of F on average.

| policy | circuits re-checked | fidelity lost | vs keeping |
| --- | --- | --- | --- |
| keep the old device | 0% | 0.0000 | – |
| re-check all | 100% | 0 | −100% |
| v8a_c001: switch to its new first device (no re-check) | 0% | 0.0034 | – |
| v8a_c001: switch only if predicted gain > 0.005 (moves 12%) | 0% | 0.0027 | – |
| v8a_c001: switch only if predicted gain > 0.01 (moves 8%) | 0% | 0.0022 | – |
| v8a_c001: switch only if predicted gain > 0.02 (moves 5%) | 0% | 0.0015 | – |
| v8a_c001: re-check where it predicts a change | 34% | 0.0000 | – |
| v8kd40: switch to its new first device (no re-check) | 0% | 0.0027 | – |
| v8kd40: switch only if predicted gain > 0.005 (moves 10%) | 0% | 0.0021 | – |
| v8kd40: switch only if predicted gain > 0.01 (moves 8%) | 0% | 0.0017 | – |
| v8kd40: switch only if predicted gain > 0.02 (moves 5%) | 0% | 0.0011 | – |
| v8kd40: re-check where it predicts a change | 29% | 0.0000 | – |
| v8kdl40: switch to its new first device (no re-check) | 0% | 0.0025 | – |
| v8kdl40: switch only if predicted gain > 0.005 (moves 10%) | 0% | 0.0019 | – |
| v8kdl40: switch only if predicted gain > 0.01 (moves 7%) | 0% | 0.0016 | – |
| v8kdl40: switch only if predicted gain > 0.02 (moves 4%) | 0% | 0.0010 | – |
| v8kdl40: re-check where it predicts a change | 24% | 0.0000 | – |

Model-guided re-check of a fixed share of circuits (predicted changes first, then the smallest predicted margins), vs random:

| share re-checked | random | v8a_c001 | v8kd40 | v8kdl40 |
| --- | --- | --- | --- | --- |
| 5% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 20% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 50% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Large calibration changes (x0.5, x2)

The best device changes in 7% of (circuit, change). Keeping the old device loses 0.0000 of F on average.

| policy | circuits re-checked | fidelity lost | vs keeping |
| --- | --- | --- | --- |
| keep the old device | 0% | 0.0000 | – |
| re-check all | 100% | 0 | −100% |
| v8a_c001: switch to its new first device (no re-check) | 0% | 0.0059 | – |
| v8a_c001: switch only if predicted gain > 0.005 (moves 23%) | 0% | 0.0051 | – |
| v8a_c001: switch only if predicted gain > 0.01 (moves 17%) | 0% | 0.0042 | – |
| v8a_c001: switch only if predicted gain > 0.02 (moves 10%) | 0% | 0.0028 | – |
| v8a_c001: re-check where it predicts a change | 47% | 0.0000 | – |
| v8kd40: switch to its new first device (no re-check) | 0% | 0.0051 | – |
| v8kd40: switch only if predicted gain > 0.005 (moves 17%) | 0% | 0.0039 | – |
| v8kd40: switch only if predicted gain > 0.01 (moves 13%) | 0% | 0.0031 | – |
| v8kd40: switch only if predicted gain > 0.02 (moves 9%) | 0% | 0.0023 | – |
| v8kd40: re-check where it predicts a change | 41% | 0.0000 | – |
| v8kdl40: switch to its new first device (no re-check) | 0% | 0.0051 | – |
| v8kdl40: switch only if predicted gain > 0.005 (moves 17%) | 0% | 0.0040 | – |
| v8kdl40: switch only if predicted gain > 0.01 (moves 13%) | 0% | 0.0034 | – |
| v8kdl40: switch only if predicted gain > 0.02 (moves 8%) | 0% | 0.0020 | – |
| v8kdl40: re-check where it predicts a change | 29% | 0.0000 | – |

Model-guided re-check of a fixed share of circuits (predicted changes first, then the smallest predicted margins), vs random:

| share re-checked | random | v8a_c001 | v8kd40 | v8kdl40 |
| --- | --- | --- | --- | --- |
| 5% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 20% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 50% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
