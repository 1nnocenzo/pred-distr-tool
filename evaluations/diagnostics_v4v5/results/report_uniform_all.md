## 5. Learned vs uniform placement, v4 and v5 (seed 5)
calibration: control model on the calibration-shift sample, R² per variant summarised as
mean / worst over the 18 device variants; held-out splits: R² and bias (mean F_pred - F_true).
model       placement        calib mean calib worst         qaoa R2    bias          qnn R2    bias  variational R2    bias  var/qnn R2
v4_phys     learned               0.888       0.436           0.893  +0.052           0.844  -0.066           0.895  -0.002       0.945
v4_phys     uniform errors        0.843       0.737           0.923  -0.022          -0.497  -0.149           0.785  -0.065       0.444
v4_phys     uniform all           0.837       0.732           0.912  -0.023          -0.371  -0.176           0.797  -0.060       0.370
v5_sinkhorn learned               0.983       0.908           0.823  +0.085           0.968  -0.027           0.901  -0.025       0.484
v5_sinkhorn uniform               0.945       0.883           0.910  +0.027           0.193  -0.179           0.839  -0.023      -0.093
