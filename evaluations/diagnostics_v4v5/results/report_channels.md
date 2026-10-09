## 1. Channel swap (calibration shift, control models seed 5): R² vs the variant's truth
full = variant everywhere; features_only = variant only in placement/MLP features;
errors_only = variant only in the physics-head errors.  orig rows are identical by construction.

### v4_phys
variant                          full  features_only    errors_only
EQE1_Top/x0.5                   0.991          0.826          0.991
EQE1_Top/x2                     0.974          0.726          0.975
EQE1_Top/shuffle                0.799          0.976          0.801
EQE1_Top/jitterA                0.976          0.975          0.976
EQE1_Top/jitterB                0.970          0.983          0.970
EQE1_Bottom/x0.5                0.988          0.835          0.988
EQE1_Bottom/x2                  0.978          0.746          0.978
EQE1_Bottom/shuffle             0.658          0.976          0.658
EQE1_Bottom/jitterA             0.984          0.985          0.984
EQE1_Bottom/jitterB             0.986          0.985          0.986
QExa20/x0.5                     0.993          0.773          0.993
QExa20/x2                       0.518          0.924          0.984
QExa20/shuffle                  0.788          0.969          0.979
QExa20/jitterA                  0.436          0.548          0.986
QExa20/jitterB                  0.985          0.986          0.985

### v5_sinkhorn
variant                          full  features_only    errors_only
EQE1_Top/x0.5                   0.990          0.798          0.992
EQE1_Top/x2                     0.990          0.696          0.995
EQE1_Top/shuffle                0.981          0.981          0.986
EQE1_Top/jitterA                0.984          0.980          0.985
EQE1_Top/jitterB                0.991          0.989          0.992
EQE1_Bottom/x0.5                0.988          0.792          0.991
EQE1_Bottom/x2                  0.979          0.822          0.990
EQE1_Bottom/shuffle             0.932          0.969          0.933
EQE1_Bottom/jitterA             0.991          0.992          0.990
EQE1_Bottom/jitterB             0.991          0.990          0.991
QExa20/x0.5                     0.995          0.768          0.994
QExa20/x2                       0.995          0.662          0.997
QExa20/shuffle                  0.908          0.971          0.983
QExa20/jitterA                  0.994          0.994          0.993
QExa20/jitterB                  0.992          0.991          0.992
