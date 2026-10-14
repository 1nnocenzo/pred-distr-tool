# v24 routing stage: routed-circuit duration

Median |pred/true − 1| on routed cases [n seeds]; bias = median pred/true. *mean rates* = schedule of the predicted SWAP rates; *sampled* = median schedule over Poisson samples of the SWAPs. vs dataset = esp_labels.pt (the v19i–v22 target).

| model | split | seeds | vs Sabre: mean rates | sampled | sampled, no stage 2 | bias (sampled) | with true SWAPs | vs dataset: sampled | Sabre vs dataset | SWAP count ratio (median / mean) | routing-free |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v24 | control | 1 | 8.4% | – | – | – | 6.9% | – | – | 1.44 / – | 0.3% |
| v24 | qaoa | 1 | 15.3% | – | – | – | 3.6% | – | – | 1.70 / – | 2.1% |
| v24 | variational | 1 | 9.4% | – | – | – | 8.1% | – | – | 1.94 / – | 1.5% |
| v24 | loso_dense-complete | 1 | 20.1% | – | – | – | 4.0% | – | – | 1.96 / – | 2.2% |
| v24 (sampled) | control | 1 | 8.4% | 7.4% | 10.1% | 0.99 | 7.0% | 7.8% | 0.0% | 1.00 / 1.00 | 0.3% |
| v24 (sampled) | qaoa | 1 | 15.3% | 12.5% | 9.8% | 0.87 | 3.6% | 9.8% | 4.6% | 0.88 / 0.87 | 2.1% |
| v24 (sampled) | variational | 1 | 9.4% | 5.6% | 7.5% | 0.98 | 8.5% | 5.7% | 0.0% | 1.22 / 1.15 | 1.5% |
| v24 (sampled) | loso_dense-complete | 1 | 20.1% | 14.3% | 15.0% | 0.86 | 4.0% | – | – | 1.08 / 1.06 | 2.2% |
| v24b | control | 3 | 5.2% ± 0.1% | 5.3% ± 0.2% | 8.3% ± 0.1% | 1.01 ± 0.00 | 7.0% ± 0.0% | 7.1% ± 0.1% | 0.0% ± 0.0% | 1.00 ± 0.00 / 1.00 ± 0.00 | 0.4% ± 0.0% |
| v24b | qaoa | 3 | 16.2% ± 2.6% | 13.7% ± 2.5% | 10.6% ± 1.1% | 0.86 ± 0.03 | 3.6% ± 0.0% | 10.8% ± 2.2% | 4.6% ± 0.0% | 0.87 ± 0.05 / 0.86 ± 0.05 | 2.4% ± 1.0% |
| v24b | variational | 3 | 7.9% ± 0.5% | 5.4% ± 0.6% | 5.7% ± 0.6% | 0.97 ± 0.01 | 8.5% ± 0.0% | 5.5% ± 0.6% | 0.0% ± 0.0% | 1.32 ± 0.10 / 1.28 ± 0.10 | 2.9% ± 0.2% |
| v24b | loso_dense-complete | 3 | 19.6% ± 0.5% | 13.8% ± 0.4% | 14.9% ± 0.4% | 0.87 ± 0.00 | 4.0% ± 0.0% | – | – | 1.10 ± 0.02 / 1.08 ± 0.03 | 2.4% ± 0.0% |

## Scale diagnostic (seed-5 checkpoints)

| split | set | median true SWAPs | median pred | mean pred / mean true | share of pred on zero slots | mean rates | sampled (median) | pred SWAPs + true waits | true SWAPs + pred waits | true SWAPs + waits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | test | 44 | 43 | 1.00 | 0.27 | 11.0% (0.97) | 9.9% (0.99) | 6.9% (0.98) | 10.8% (0.98) | 6.8% (0.98) |
| control | train | 34 | 36 | 1.00 | 0.40 | 12.9% (0.96) | 11.0% (0.99) | 7.8% (1.04) | 11.6% (0.97) | 7.6% (1.05) |
| qaoa | test | 70 | 62 | 0.87 | 0.50 | 12.5% (0.88) | 9.6% (0.91) | 4.9% (0.95) | 11.5% (0.89) | 3.5% (0.97) |
| qaoa | train | 37 | 36 | 1.00 | 0.43 | 13.6% (0.95) | 11.3% (0.97) | 7.5% (1.04) | 12.4% (0.97) | 7.5% (1.04) |
| variational | test | 8 | 16 | 1.15 | 0.62 | 10.9% (0.90) | 7.6% (0.96) | 8.1% (1.06) | 7.5% (0.92) | 8.7% (1.06) |
| variational | train | 26 | 28 | 1.00 | 0.39 | 10.7% (0.97) | 9.7% (1.00) | 7.8% (1.04) | 9.9% (0.99) | 8.1% (1.05) |

Duration cells: median error (bias). Per-slot calibration tables are in the json files.
