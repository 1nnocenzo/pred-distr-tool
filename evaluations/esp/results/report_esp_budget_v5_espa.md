# ESP error budget: v5_espa

Mean over seeds. Each row swaps one link for the truth; regret = mean ESP lost by the chosen device.

## regret

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.0061 | 0.0039 | 0.0189 | 0.0035 |
| true F | 0.0047 | 0.0038 | 0.0155 | 0.0024 |
| true decay | 0.0005 | 0.0005 | 0.0079 | 0.0003 |
| true duration | 0.0067 | 0.0043 | 0.0224 | 0.0044 |
| true structure | 0.0072 | 0.0048 | 0.0225 | 0.0060 |
| device-average T | 0.0083 | 0.0161 | 0.0234 | 0.0061 |
| true duration + device-average T | 0.0067 | 0.0103 | 0.0275 | 0.0085 |
| F only | 0.0073 | 0.0021 | 0.0438 | 0.0087 |
| true F only | 0.0095 | 0.0010 | 0.0065 | 0.0065 |

## right device

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.5727 | 0.8504 | 0.5222 | 0.8247 |
| true F | 0.7103 | 0.8636 | 0.6023 | 0.8841 |
| true decay | 0.8543 | 0.9548 | 0.7446 | 0.9162 |
| true duration | 0.6394 | 0.8662 | 0.5182 | 0.7895 |
| true structure | 0.5873 | 0.8636 | 0.5172 | 0.7352 |
| device-average T | 0.4582 | 0.5294 | 0.4303 | 0.7311 |
| true duration + device-average T | 0.6067 | 0.7167 | 0.4091 | 0.6813 |
| F only | 0.5408 | 0.8680 | 0.3292 | 0.5792 |
| true F only | 0.2822 | 0.9342 | 0.7262 | 0.5734 |

## R² ESP

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.8626 | 0.7384 | 0.4321 | 0.9836 |
| true F | 0.9159 | 0.7802 | 0.4885 | 0.9862 |
| true decay | 0.9815 | 0.9926 | 0.9390 | 0.9991 |
| true duration | 0.5320 | 0.9208 | 0.5682 | 0.9768 |
| true structure | 0.5314 | 0.9038 | 0.4902 | 0.9305 |
| device-average T | 0.8255 | 0.6035 | 0.3272 | 0.9736 |
| true duration + device-average T | 0.5656 | 0.8617 | 0.4767 | 0.9648 |
| F only | 0.1300 | 0.6337 | 0.6402 | 0.6728 |
| true F only | 0.8058 | 0.4473 | 0.6025 | 0.6720 |

## R² log ESP

| variant | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| model | 0.4731 | 0.4276 | -8.0052 | 0.9635 |
| true F | 0.7480 | 0.5098 | -6.8065 | 0.9696 |
| true decay | 0.8825 | 0.9915 | 0.8585 | 0.9970 |
| true duration | -0.1055 | 0.8604 | -1.2710 | 0.9495 |
| true structure | -0.0799 | 0.7341 | -2.5543 | 0.2086 |
| device-average T | 0.5400 | 0.0580 | -9.9516 | 0.9425 |
| true duration + device-average T | -0.0399 | 0.7844 | -1.9246 | 0.9246 |
| F only | -0.3601 | 0.1781 | 0.1419 | 0.1089 |
| true F only | 0.2447 | 0.0538 | 0.0671 | 0.1043 |

## Duration and depth: model vs the head fed the true per-qubit structure

| | LOFO qaoa | LOFO qnn | LOGO variational | control |
| --- | --- | --- | --- | --- |
| duration error, model | 0.521 | 0.085 | 0.240 | 0.028 |
| duration error, true structure | 0.202 | 0.068 | 0.089 | 0.136 |
| R² log duration, model | 0.286 | -1.748 | 0.089 | 0.989 |
| R² log duration, true structure | 0.967 | -0.641 | 0.241 | 0.779 |
| depth error, model | 0.639 | 0.246 | 0.386 | 0.039 |
| depth error, true structure | 0.461 | 0.237 | 1.570 | 0.782 |
| mean |decay|, true | 10.462 | 1.021 | 3.638 | 6.408 |
| mean |decay|, predicted | 5.662 | 1.717 | 6.376 | 6.254 |
| corr(decay pred, true) | 0.881 | 0.940 | 0.769 | 0.986 |

## Per family (test families of every split, mean over seeds)

| family | n | tmax error | mmax abs. error (pred / true median) | omax error | duration error | duration error, true structure |
| --- | --- | --- | --- | --- | --- | --- |
| qaoa (LOFO qaoa) | 8118 | 0.85 | 1.44 (1.44 / 0.00) | 0.83 | 0.52 | 0.20 |
| qnn (LOFO qnn) | 2280 | 0.30 | 0.19 (1.19 / 1.00) | 0.30 | 0.08 | 0.07 |
| qnn (LOGO variational) | 2280 | 0.50 | 0.20 (1.15 / 1.00) | 0.47 | 0.16 | 0.04 |
| vqe_real_amp (LOGO variational) | 8583 | 0.49 | 0.30 (1.24 / 1.00) | 0.47 | 0.20 | 0.09 |
| vqe_su2 (LOGO variational) | 8574 | 0.63 | 0.52 (1.49 / 1.00) | 0.59 | 0.30 | 0.13 |
| vqe_two_local (LOGO variational) | 10347 | 0.58 | 0.44 (1.43 / 1.00) | 0.56 | 0.33 | 0.09 |