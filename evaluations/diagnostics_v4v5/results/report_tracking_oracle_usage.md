# Oracle study on the usage-map head (v11)

300 pilot circuits of the control test split; 10 local v8d calibrations (aug [11, 19, 50, 55, 56, 61, 65, 72, 73, 82]).
Tracking: mean over (variant, device) of corr(Δlog F pred, Δlog F true), F > 0.01.

| model | model | + layout | + counts | + both | true maps |
| --- | --- | --- | --- | --- | --- |
| v8kd40 | 0.63 | 0.55 | – | – | – |
| v11 (phase 1) | 0.28 | 0.59 | 0.27 | 0.70 | 1.00 |
| v11d40 | 0.64 | 0.64 | 0.52 | 0.74 | 1.00 |
| v11du40 | 0.65 | 0.65 | 0.53 | 0.74 | 1.00 |
| v8kdl40 | 0.69 | 0.59 | – | – | – |
| v11dl40 | 0.68 | 0.64 | 0.59 | 0.72 | 1.00 |

true maps = the head's formula on the compiler's own maps (same compilations as the labels): ~1.0 by construction, a check that the head and the maps agree.