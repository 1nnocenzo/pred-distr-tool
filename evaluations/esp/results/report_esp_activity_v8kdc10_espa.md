# Decay ingredients: v8kdc10_espa

decay ≈ −(n_active · duration − Σ busy) · mean 1/min(T1, T2) per device; mean over seeds.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| simplified (all true) vs true decay: corr | 0.997 | 0.995 | 0.997 | 0.996 |
| simplified (all true) vs true decay: R² | 0.991 | 0.984 | 0.992 | 0.991 |
| n_active pred / true (median) | 1.000 | 1.000 | 1.000 | 1.000 |
| Σ busy pred / true (median) | 3.837 | 0.806 | 0.733 | 0.982 |
| duration pred / true (median) | 1.966 | 1.066 | 1.182 | 1.002 |
| share with n_active wrong | 0.252 | 0.000 | 0.127 | 0.182 |
| n_active pred − true, routing-free (mean) | 0.000 | 0.000 | 0.000 | 0.000 |
| n_active true − logical, routing-free (mean) | 0.000 | 0.000 | 0.000 | 0.000 |
| n_active pred − true, routed (mean) | -0.369 | 0.000 | -0.450 | -0.428 |
| n_active true − logical, routed (mean) | 0.369 | 0.000 | 0.450 | 0.428 |

## regret

| decay used | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model decay (per-qubit head) | 0.0064 | 0.0286 | 0.0168 | 0.0043 |
| simplified, model ingredients | 0.0059 | 0.0248 | 0.0164 | 0.0101 |
| … true n_active | 0.0039 | 0.0248 | 0.0139 | 0.0088 |
| … true Σ busy | 0.0054 | 0.0271 | 0.0239 | 0.0140 |
| … true duration | 0.0163 | 0.0363 | 0.0202 | 0.0212 |
| … true n_active + duration | 0.0163 | 0.0363 | 0.0177 | 0.0191 |
| … true n_active + Σ busy | 0.0053 | 0.0271 | 0.0217 | 0.0127 |
| simplified, all true | 0.0023 | 0.0227 | 0.0122 | 0.0025 |
| true decay | 0.0002 | 0.0015 | 0.0010 | 0.0003 |

## decay R²

| decay used | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model decay (per-qubit head) | 0.9121 | -5.0343 | -0.1690 | 0.9700 |
| simplified, model ingredients | 0.9154 | -5.4034 | -0.2754 | 0.9614 |
| … true n_active | 0.8855 | -5.4034 | -0.3228 | 0.9475 |
| … true Σ busy | 0.3734 | -1.6223 | -1.1079 | 0.9495 |
| … true duration | 0.4089 | -1.3688 | 0.5941 | 0.9490 |
| … true n_active + duration | 0.5183 | -1.3688 | 0.5491 | 0.9515 |
| … true n_active + Σ busy | 0.1625 | -1.6223 | -1.1805 | 0.9467 |
| simplified, all true | 0.9912 | 0.9839 | 0.9924 | 0.9909 |
| true decay | 1.0000 | 1.0000 | 1.0000 | 1.0000 |