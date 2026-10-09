
## 3. LOFO qaoa: parts of -log F, model vs truth from the compiled circuits
2706/2706 circuits with compiled QPY.  Values: mean over circuits x devices.
truth: compilation stored in qpy/ (seed 0), so it differs slightly from the label compile.

q2-4 (n=468)   true F 0.881   true counts r/meas/cz = 46/0/7
                    1q   readout        2q     total   pred F   bias F
truth            0.036     0.000     0.095     0.131
v4_phys          0.022     0.090     0.059     0.171    0.847   -0.034
v5_sinkhorn      0.038     0.022     0.080     0.141    0.871   -0.010

q5-8 (n=624)   true F 0.318   true counts r/meas/cz = 334/0/70
                    1q   readout        2q     total   pred F   bias F
truth            0.300     0.000     1.202     1.502
v4_phys          0.089     0.454     0.308     0.851    0.471   +0.151
v5_sinkhorn      0.179     0.238     0.472     0.888    0.486   +0.166

q9-13 (n=780)   true F 0.035   true counts r/meas/cz = 1041/0/252
                    1q   readout        2q     total   pred F   bias F
truth            1.084     0.000     5.938     7.022
v4_phys          0.228     1.502     0.976     2.706    0.100   +0.064
v5_sinkhorn      0.466     1.118     1.644     3.228    0.159   +0.123

q14-20 (n=834)   true F 0.001   true counts r/meas/cz = 2125/0/554
                    1q   readout        2q     total   pred F   bias F
truth            2.829     0.000    15.520    18.349
v4_phys          0.414     3.000     1.874     5.288    0.015   +0.014
v5_sinkhorn      0.894     1.976     3.316     6.187    0.044   +0.043
