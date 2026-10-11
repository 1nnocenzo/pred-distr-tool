# Why is local tracking low? Oracle substitutions (v8d calibrations, control split)

300 circuits per pool; unseen = 10 local v8d variants (jitter / shuffle / bad qubits, aug ≥ 8); seen by v8kd8 = aug0-7.  Tracking: mean over (variant, device) of corr(Δlog F pred, Δlog F true), F > 0.01; slope: pooled regression of pred on true.

## Label noise

corr(Δ seed 0, Δ seed 1) = 0.80;  corr(Δ best of seeds 0-1, Δ seed 2) = 0.86  (13188 cases).  The labels are best of 3, so their own reliability is a bit higher: a perfect predictor cannot exceed roughly this agreement.

## v8kd (100 calib.)

| circuits | calibrations | model | + coupler | + layout + coupler | + both | + both + coupler | placement moves (model / compiler) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | unseen | 0.59 (slope 0.67) | 0.50 (slope 0.55) | 0.49 (slope 0.55) | 0.56 (slope 0.66) | 0.59 (slope 0.61) | 30% / 53% |
| train | unseen | 0.61 (slope 0.71) | 0.51 (slope 0.56) | 0.50 (slope 0.53) | 0.54 (slope 0.63) | 0.61 (slope 0.59) | 30% / 52% |
