# Why is local tracking low? Oracle substitutions (v8d calibrations, control split)

300 circuits per pool; unseen = 10 local v8d variants (jitter / shuffle / bad qubits, aug ≥ 8); seen by v8kd8 = aug0-7.  Tracking: mean over (variant, device) of corr(Δlog F pred, Δlog F true), F > 0.01; slope: pooled regression of pred on true.

## Label noise

corr(Δ seed 0, Δ seed 1) = 0.80;  corr(Δ best of seeds 0-1, Δ seed 2) = 0.86  (13188 cases).  The labels are best of 3, so their own reliability is a bit higher: a perfect predictor cannot exceed roughly this agreement.

## v8kd8 (8 calib.)

| circuits | calibrations | model | + layout | + counts | + both | fixed | placement moves (model / compiler) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | unseen | 0.45 (slope 0.56) | 0.53 (slope 0.63) | 0.40 (slope 0.47) | 0.54 (slope 0.64) | 0.41 (slope 0.74) | 31% / 53% |
| test | seen by v8kd8 | 0.81 (slope 0.88) | 0.63 (slope 0.68) | 0.67 (slope 0.58) | 0.59 (slope 0.59) | 0.50 (slope 1.20) | 42% / 56% |
| train | unseen | 0.47 (slope 0.61) | 0.49 (slope 0.59) | 0.42 (slope 0.50) | 0.52 (slope 0.63) | 0.45 (slope 0.69) | 31% / 52% |
| train | seen by v8kd8 | 0.83 (slope 0.88) | 0.62 (slope 0.69) | 0.70 (slope 0.60) | 0.54 (slope 0.58) | 0.50 (slope 1.01) | 42% / 56% |
