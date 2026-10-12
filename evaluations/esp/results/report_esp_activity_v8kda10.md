# Decay ingredients: v8kda10

decay ≈ −(n_active · duration − Σ busy) · mean 1/min(T1, T2) per device; mean over seeds.

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| simplified (all true) vs true decay: corr | 0.997 | 0.995 | 0.997 | 0.996 |
| simplified (all true) vs true decay: R² | 0.991 | 0.984 | 0.992 | 0.991 |
| n_active pred / true (median) | 1.000 | 1.000 | 1.000 | 1.000 |
| Σ busy pred / true (median) | 3.846 | 0.876 | 0.805 | 0.984 |
| duration pred / true (median) | 1.937 | 1.175 | 1.283 | 0.997 |
| share with n_active wrong | 0.252 | 0.000 | 0.127 | 0.182 |
| n_active pred − true, routing-free (mean) | 0.000 | 0.000 | 0.000 | 0.000 |
| n_active true − logical, routing-free (mean) | 0.000 | 0.000 | 0.000 | 0.000 |
| n_active pred − true, routed (mean) | -0.369 | 0.000 | -0.450 | -0.428 |
| n_active true − logical, routed (mean) | 0.369 | 0.000 | 0.450 | 0.428 |

## regret

| decay used | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model decay (per-qubit head) | 0.0065 | 0.0187 | 0.0154 | 0.0045 |
| simplified, model ingredients | 0.0088 | 0.0196 | 0.0151 | 0.0098 |
| … true n_active | 0.0072 | 0.0196 | 0.0127 | 0.0087 |
| … true Σ busy | 0.0077 | 0.0171 | 0.0195 | 0.0138 |
| … true duration | 0.0142 | 0.0389 | 0.0245 | 0.0207 |
| … true n_active + duration | 0.0142 | 0.0389 | 0.0224 | 0.0184 |
| … true n_active + Σ busy | 0.0064 | 0.0171 | 0.0169 | 0.0122 |
| simplified, all true | 0.0032 | 0.0183 | 0.0120 | 0.0029 |
| true decay | 0.0003 | 0.0000 | 0.0010 | 0.0003 |

## decay R²

| decay used | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model decay (per-qubit head) | 0.8902 | -6.0926 | -0.3842 | 0.9698 |
| simplified, model ingredients | 0.8934 | -6.4702 | -1.1090 | 0.9632 |
| … true n_active | 0.8773 | -6.4702 | -1.3661 | 0.9498 |
| … true Σ busy | 0.3754 | -4.0067 | -3.0301 | 0.9451 |
| … true duration | 0.4279 | -0.0557 | 0.6891 | 0.9511 |
| … true n_active + duration | 0.5286 | -0.0557 | 0.6712 | 0.9508 |
| … true n_active + Σ busy | 0.1796 | -4.0067 | -3.4170 | 0.9456 |
| simplified, all true | 0.9912 | 0.9839 | 0.9924 | 0.9909 |
| true decay | 1.0000 | 1.0000 | 1.0000 | 1.0000 |