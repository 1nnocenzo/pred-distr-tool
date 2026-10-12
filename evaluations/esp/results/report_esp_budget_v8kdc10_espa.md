# ESP error budget: v8kdc10_espa

Mean over seeds. Each row swaps one link for the truth; regret = mean ESP lost by the chosen device.

## regret

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.0079 | 0.0229 | 0.0184 | 0.0046 |
| true F | 0.0061 | 0.0217 | 0.0157 | 0.0035 |
| true decay | 0.0011 | 0.0005 | 0.0013 | 0.0003 |
| true duration | 0.0103 | 0.0162 | 0.0208 | 0.0138 |
| true structure | 0.0101 | 0.0200 | 0.0218 | 0.0120 |
| device-average T | 0.0087 | 0.0272 | 0.0197 | 0.0107 |
| true duration + device-average T | 0.0109 | 0.0281 | 0.0222 | 0.0199 |
| F only | 0.0109 | 0.0137 | 0.0198 | 0.0078 |
| true F only | 0.0095 | 0.0010 | 0.0065 | 0.0065 |

## right device

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.4613 | 0.4614 | 0.5555 | 0.7689 |
| true F | 0.6717 | 0.4899 | 0.6107 | 0.8320 |
| true decay | 0.8267 | 0.9614 | 0.8608 | 0.9061 |
| true duration | 0.5335 | 0.5684 | 0.5338 | 0.6010 |
| true structure | 0.4991 | 0.4829 | 0.5231 | 0.5947 |
| device-average T | 0.4523 | 0.3654 | 0.4968 | 0.6150 |
| true duration + device-average T | 0.5580 | 0.2798 | 0.4746 | 0.4569 |
| F only | 0.2528 | 0.6044 | 0.5374 | 0.5647 |
| true F only | 0.2822 | 0.9342 | 0.7262 | 0.5734 |

## R² ESP

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.8890 | 0.2855 | 0.5201 | 0.9766 |
| true F | 0.9070 | 0.3364 | 0.5167 | 0.9811 |
| true decay | 0.9926 | 0.9839 | 0.9906 | 0.9981 |
| true duration | 0.8052 | 0.5811 | 0.6857 | 0.9325 |
| true structure | 0.7771 | 0.6454 | 0.6974 | 0.9104 |
| device-average T | 0.8621 | 0.1436 | 0.4547 | 0.9663 |
| true duration + device-average T | 0.8102 | 0.4846 | 0.6277 | 0.9147 |
| F only | 0.7286 | 0.7412 | 0.4227 | 0.6748 |
| true F only | 0.8058 | 0.4473 | 0.6025 | 0.6720 |

## R² log ESP

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.7304 | -3.4704 | -6.6008 | 0.9403 |
| true F | 0.7905 | -2.9172 | -6.7860 | 0.9460 |
| true decay | 0.9691 | 0.9729 | 0.9749 | 0.9966 |
| true duration | 0.2447 | 0.0499 | -0.2287 | 0.7110 |
| true structure | 0.1676 | 0.6470 | 0.1451 | 0.6953 |
| device-average T | 0.7391 | -4.0430 | -7.2409 | 0.9149 |
| true duration + device-average T | 0.2610 | -0.0546 | -0.3792 | 0.6521 |
| F only | 0.0926 | 0.2864 | -0.0550 | 0.1000 |
| true F only | 0.2447 | 0.0538 | 0.0671 | 0.1043 |

## Duration and depth: model vs the head fed the true per-qubit structure

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| duration error, model | 0.568 | 0.203 | 0.252 | 0.063 |
| duration error, true structure | 0.240 | 0.085 | 0.095 | 0.088 |
| R² log duration, model | 0.337 | -9.612 | 0.081 | 0.979 |
| R² log duration, true structure | 0.942 | -1.406 | 0.840 | 0.961 |
| depth error, model | 0.411 | 0.584 | 0.369 | 0.060 |
| depth error, true structure | 0.228 | 0.141 | 0.293 | 0.165 |
| mean |decay|, true | 10.462 | 1.021 | 3.638 | 6.408 |
| mean |decay|, predicted | 7.974 | 3.193 | 6.180 | 6.420 |
| corr(decay pred, true) | 0.916 | 0.899 | 0.722 | 0.985 |

## Per family (test families of every split, mean over seeds)

| family | n | tmax error | mmax abs. error (pred / true median) | omax error | duration error | duration error, true structure |
| --- | --- | --- | --- | --- | --- | --- |
| qaoa (LOFO qaoa) | 8118 | 0.41 | 1.48 (1.48 / 0.00) | 0.47 | 0.57 | 0.24 |
| qnn (LOFO qnn) | 2280 | 0.39 | 0.30 (1.29 / 1.00) | 0.43 | 0.20 | 0.09 |
| qnn (LOGO variational) | 2280 | 0.64 | 0.35 (1.29 / 1.00) | 0.71 | 0.26 | 0.08 |
| vqe_real_amp (LOGO variational) | 8583 | 0.29 | 0.36 (1.33 / 1.00) | 0.38 | 0.21 | 0.11 |
| vqe_su2 (LOGO variational) | 8574 | 0.42 | 0.54 (1.52 / 1.00) | 0.52 | 0.31 | 0.10 |
| vqe_two_local (LOGO variational) | 10347 | 0.54 | 0.42 (1.39 / 1.00) | 0.73 | 0.27 | 0.09 |