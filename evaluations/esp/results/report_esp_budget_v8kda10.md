# ESP error budget: v8kda10

Mean over seeds. Each row swaps one link for the truth; regret = mean ESP lost by the chosen device.

## regret

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.0075 | 0.0200 | 0.0177 | 0.0047 |
| true F | 0.0057 | 0.0182 | 0.0147 | 0.0034 |
| true decay | 0.0009 | 0.0000 | 0.0031 | 0.0003 |
| true duration | 0.0104 | 0.0260 | 0.0197 | 0.0159 |
| true structure | 0.0114 | 0.0254 | 0.0206 | 0.0137 |
| device-average T | 0.0093 | 0.0247 | 0.0197 | 0.0102 |
| true duration + device-average T | 0.0105 | 0.0361 | 0.0234 | 0.0208 |
| F only | 0.0117 | 0.0056 | 0.0236 | 0.0082 |
| true F only | 0.0095 | 0.0010 | 0.0065 | 0.0065 |

## right device

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.4334 | 0.5281 | 0.5535 | 0.7702 |
| true F | 0.6510 | 0.5601 | 0.6155 | 0.8379 |
| true decay | 0.8113 | 0.9982 | 0.8276 | 0.9072 |
| true duration | 0.5139 | 0.3785 | 0.5427 | 0.5634 |
| true structure | 0.4458 | 0.3868 | 0.5400 | 0.5686 |
| device-average T | 0.3928 | 0.4149 | 0.4794 | 0.6147 |
| true duration + device-average T | 0.5359 | 0.1632 | 0.4622 | 0.4446 |
| F only | 0.2372 | 0.8088 | 0.5016 | 0.5542 |
| true F only | 0.2822 | 0.9342 | 0.7262 | 0.5734 |

## R² ESP

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.8815 | 0.4033 | 0.4586 | 0.9770 |
| true F | 0.8985 | 0.4594 | 0.4604 | 0.9815 |
| true decay | 0.9932 | 0.9845 | 0.9894 | 0.9982 |
| true duration | 0.8099 | 0.6339 | 0.7316 | 0.9413 |
| true structure | 0.7780 | 0.6753 | 0.7470 | 0.9148 |
| device-average T | 0.8519 | 0.2715 | 0.3957 | 0.9678 |
| true duration + device-average T | 0.8156 | 0.5439 | 0.6842 | 0.9274 |
| F only | 0.7188 | 0.7541 | 0.4703 | 0.6680 |
| true F only | 0.8058 | 0.4473 | 0.6025 | 0.6720 |

## R² log ESP

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.6633 | -2.5039 | -8.1912 | 0.9382 |
| true F | 0.7230 | -2.0068 | -8.2854 | 0.9446 |
| true decay | 0.9687 | 0.9728 | 0.9751 | 0.9967 |
| true duration | 0.2471 | 0.3761 | 0.1815 | 0.7642 |
| true structure | 0.1870 | 0.7284 | 0.3672 | 0.7048 |
| device-average T | 0.6807 | -2.9826 | -8.8282 | 0.9173 |
| true duration + device-average T | 0.2647 | 0.2854 | 0.0972 | 0.7200 |
| F only | 0.0874 | 0.3051 | -0.0235 | 0.0956 |
| true F only | 0.2447 | 0.0538 | 0.0671 | 0.1043 |

## Duration and depth: model vs the head fed the true per-qubit structure

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| duration error, model | 0.554 | 0.178 | 0.341 | 0.059 |
| duration error, true structure | 0.251 | 0.085 | 0.096 | 0.088 |
| R² log duration, model | 0.340 | -7.624 | -0.191 | 0.982 |
| R² log duration, true structure | 0.937 | -1.348 | 0.844 | 0.962 |
| depth error, model | 0.407 | 0.527 | 0.387 | 0.063 |
| depth error, true structure | 0.250 | 0.158 | 0.285 | 0.191 |
| mean |decay|, true | 10.462 | 1.021 | 3.638 | 6.408 |
| mean |decay|, predicted | 7.838 | 2.850 | 7.035 | 6.446 |
| corr(decay pred, true) | 0.905 | 0.923 | 0.760 | 0.984 |

## Per family (test families of every split, mean over seeds)

| family | n | tmax error | mmax abs. error (pred / true median) | omax error | duration error | duration error, true structure |
| --- | --- | --- | --- | --- | --- | --- |
| qaoa (LOFO qaoa) | 8118 | 0.41 | 1.44 (1.44 / 0.00) | 0.47 | 0.55 | 0.25 |
| qnn (LOFO qnn) | 2280 | 0.47 | 0.30 (1.28 / 1.00) | 0.51 | 0.18 | 0.08 |
| qnn (LOGO variational) | 2280 | 0.44 | 0.51 (1.51 / 1.00) | 0.50 | 0.36 | 0.08 |
| vqe_real_amp (LOGO variational) | 8583 | 0.27 | 0.43 (1.40 / 1.00) | 0.35 | 0.26 | 0.11 |
| vqe_su2 (LOGO variational) | 8574 | 0.39 | 0.73 (1.71 / 1.00) | 0.47 | 0.45 | 0.10 |
| vqe_two_local (LOGO variational) | 10347 | 0.49 | 0.53 (1.52 / 1.00) | 0.68 | 0.36 | 0.09 |