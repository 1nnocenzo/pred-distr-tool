# Why is local tracking low? Oracle substitutions (v8d calibrations, control split)

300 circuits per pool; unseen = 10 local v8d variants (jitter / shuffle / bad qubits, aug ≥ 8); seen by v8kd8 = aug0-7.  Tracking: mean over (variant, device) of corr(Δlog F pred, Δlog F true), F > 0.01; slope: pooled regression of pred on true.

## Label noise

corr(Δ seed 0, Δ seed 1) = 0.80;  corr(Δ best of seeds 0-1, Δ seed 2) = 0.86  (13188 cases).  The labels are best of 3, so their own reliability is a bit higher: a perfect predictor cannot exceed roughly this agreement.

## v8k

| circuits | calibrations | model | + layout | + counts | + both | fixed | placement moves (model / compiler) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | unseen | 0.35 (slope 0.45) | 0.53 (slope 0.68) | 0.34 (slope 0.46) | 0.53 (slope 0.78) | 0.42 (slope 0.92) | 31% / 53% |
| train | unseen | 0.40 (slope 0.50) | 0.50 (slope 0.66) | 0.41 (slope 0.51) | 0.51 (slope 0.77) | 0.44 (slope 0.85) | 30% / 52% |
