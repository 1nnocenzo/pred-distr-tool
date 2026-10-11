
## 2. v5 with a uniform placement
learned placement: mean max row probability 0.175 (uniform would be ~1/26)
calibration sample, all variants: mean |F_learned - F_uniform| = 0.0451, corr = 0.9849
variant                 R² learned  R² uniform
EQE1_Top/orig                0.995       0.973
EQE1_Top/x0.5                0.990       0.977
EQE1_Top/x2                  0.990       0.948
EQE1_Top/shuffle             0.981       0.978
EQE1_Top/jitterA             0.984       0.948
EQE1_Top/jitterB             0.991       0.977
EQE1_Bottom/orig             0.993       0.923
EQE1_Bottom/x0.5             0.988       0.969
EQE1_Bottom/x2               0.979       0.901
EQE1_Bottom/shuffle          0.932       0.964
EQE1_Bottom/jitterA          0.991       0.934
EQE1_Bottom/jitterB          0.991       0.938
QExa20/orig                  0.998       0.935
QExa20/x0.5                  0.995       0.960
QExa20/x2                    0.995       0.898
QExa20/shuffle               0.908       0.972
QExa20/jitterA               0.994       0.883
QExa20/jitterB               0.992       0.938
LOFO qaoa test: R² learned 0.823  uniform 0.910   bias learned +0.085  uniform +0.027
