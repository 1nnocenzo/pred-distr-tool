
## 4. v4 attention: sharpness (mean max weight per logical qubit), share of logical
qubits whose top physical qubit is used by another one (over a batch of 16 circuits), R²
variant                max weight shared top      R²
EQE1_Top/orig               0.500      0.993   0.987
EQE1_Top/x0.5               0.500      0.993   0.991
EQE1_Top/x2                 0.500      0.993   0.974
EQE1_Top/shuffle            0.500      0.993   0.799
EQE1_Top/jitterA            0.500      0.993   0.976
EQE1_Top/jitterB            0.500      0.993   0.970
EQE1_Bottom/orig            0.500      0.993   0.985
EQE1_Bottom/x0.5            0.500      0.993   0.988
EQE1_Bottom/x2              0.500      0.993   0.978
EQE1_Bottom/shuffle         0.500      0.993   0.658
EQE1_Bottom/jitterA         0.500      0.993   0.984
EQE1_Bottom/jitterB         0.500      0.993   0.986
QExa20/orig                 0.500      0.993   0.992
QExa20/x0.5                 0.500      0.993   0.993
QExa20/x2                   0.500      0.993   0.518
QExa20/shuffle              0.500      0.993   0.788
QExa20/jitterA              0.500      0.993   0.436
QExa20/jitterB              0.500      0.993   0.985
