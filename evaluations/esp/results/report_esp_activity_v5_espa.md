# Decay ingredients: v5_espa

decay ≈ −(n_active · duration − Σ busy) · mean 1/min(T1, T2) per device; mean over seeds.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| simplified (all true) vs true decay: corr | 0.997 | 0.995 | 0.997 | 0.996 |
| simplified (all true) vs true decay: R² | 0.991 | 0.984 | 0.992 | 0.991 |
| n_active pred / true (median) | 1.000 | 1.000 | 1.000 | 1.000 |
| Σ busy pred / true (median) | 3.404 | 0.926 | 0.709 | 0.986 |
| duration pred / true (median) | 1.760 | 1.015 | 1.160 | 1.004 |
| share with n_active wrong | 0.252 | 0.000 | 0.127 | 0.182 |
| n_active pred − true, routing-free (mean) | 0.000 | 0.000 | 0.000 | 0.000 |
| n_active true − logical, routing-free (mean) | 0.000 | 0.000 | 0.000 | 0.000 |
| n_active pred − true, routed (mean) | -0.369 | 0.000 | -0.450 | -0.428 |
| n_active true − logical, routed (mean) | 0.369 | 0.000 | 0.450 | 0.428 |

## regret

| decay used | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model decay (per-qubit head) | 0.0055 | 0.0008 | 0.0172 | 0.0040 |
| simplified, model ingredients | 0.0062 | 0.0125 | 0.0219 | 0.0071 |
| … true n_active | 0.0045 | 0.0125 | 0.0195 | 0.0055 |
| … true Σ busy | 0.0073 | 0.0156 | 0.0243 | 0.0085 |
| … true duration | 0.0067 | 0.0051 | 0.0297 | 0.0090 |
| … true n_active + duration | 0.0067 | 0.0051 | 0.0274 | 0.0074 |
| … true n_active + Σ busy | 0.0050 | 0.0156 | 0.0221 | 0.0066 |
| simplified, all true | 0.0017 | 0.0001 | 0.0195 | 0.0039 |
| true decay | 0.0005 | 0.0000 | 0.0052 | 0.0004 |

## decay R²

| decay used | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model decay (per-qubit head) | 0.6650 | 0.5257 | 0.3076 | 0.9730 |
| simplified, model ingredients | 0.7182 | 0.1717 | -0.1652 | 0.9637 |
| … true n_active | 0.6950 | 0.1717 | -0.3543 | 0.9458 |
| … true Σ busy | 0.6209 | 0.7370 | -0.9738 | 0.9586 |
| … true duration | 0.7116 | 0.6355 | 0.5656 | 0.9764 |
| … true n_active + duration | 0.7785 | 0.6355 | 0.5992 | 0.9736 |
| … true n_active + Σ busy | 0.5049 | 0.7370 | -1.2637 | 0.9581 |
| simplified, all true | 0.9912 | 0.9839 | 0.9924 | 0.9909 |
| true decay | 1.0000 | 1.0000 | 1.0000 | 1.0000 |