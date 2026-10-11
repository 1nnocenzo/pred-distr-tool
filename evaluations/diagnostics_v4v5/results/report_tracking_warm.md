# Tracking from an existing compilation (warm start)

300 pilot circuits, local v8d calibrations [11, 19, 50, 55, 56, 61, 65, 72, 73, 82]. Formula check on the old compilation: median |−ΣUε − log F| = 2.2e-08.

| predictor of Δlog F | local tracking (mean over calibration × device) |
| --- | --- |
| stay on the old compilation, exact (Δ_stay) | 0.57 |
| v8kdl40 from scratch (tracking_oracle_usage.py, same cases) | 0.69 |

Re-optimisation gain of the compiler (true Δ − Δ_stay): mean +0.1882, positive in 66% of the cases; |true Δ| mean 0.1071, |Δ_stay| mean 0.2811; pooled corr(Δ_stay, true Δ) 0.60.
