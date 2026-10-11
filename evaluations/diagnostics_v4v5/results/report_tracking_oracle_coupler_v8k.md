# Why is local tracking low? Oracle substitutions (v8d calibrations, control split)

300 circuits per pool; unseen = 10 local v8d variants (jitter / shuffle / bad qubits, aug ≥ 8); seen by v8kd8 = aug0-7.  Tracking: mean over (variant, device) of corr(Δlog F pred, Δlog F true), F > 0.01; slope: pooled regression of pred on true.

## Label noise

corr(Δ seed 0, Δ seed 1) = 0.80;  corr(Δ best of seeds 0-1, Δ seed 2) = 0.86  (13188 cases).  The labels are best of 3, so their own reliability is a bit higher: a perfect predictor cannot exceed roughly this agreement.

## v8k

| circuits | calibrations | model | + coupler | + layout + coupler | + both | + both + coupler | placement moves (model / compiler) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | unseen | 0.35 (slope 0.45) | 0.31 (slope 0.35) | 0.48 (slope 0.62) | 0.53 (slope 0.78) | 0.57 (slope 0.73) | 31% / 53% |
| train | unseen | 0.40 (slope 0.50) | 0.36 (slope 0.40) | 0.46 (slope 0.62) | 0.51 (slope 0.77) | 0.58 (slope 0.73) | 30% / 52% |
