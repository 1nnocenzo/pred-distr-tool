# v24 routing stage: routed-circuit duration

Median |pred/true − 1| on routed cases [n seeds]; bias = median pred/true. *mean rates* = schedule of the predicted SWAP rates; *sampled* = median schedule over Poisson samples of the SWAPs. vs dataset = esp_labels.pt (the v19i–v22 target).

| model | split | seeds | vs Sabre: mean rates | sampled | sampled, no stage 2 | bias (sampled) | with true SWAPs | vs dataset: sampled | Sabre vs dataset | SWAP count ratio (median / mean) | routing-free |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v24 | control | 1 | 8.4% | – | – | – | 6.9% | – | – | 1.44 / – | 0.3% |
| v24 | qaoa | 1 | 15.3% | – | – | – | 3.6% | – | – | 1.70 / – | 2.1% |
| v24 | variational | 1 | 9.4% | – | – | – | 8.1% | – | – | 1.94 / – | 1.5% |
| v24 | loso_dense-complete | 1 | 20.1% | – | – | – | 4.0% | – | – | 1.96 / – | 2.2% |
