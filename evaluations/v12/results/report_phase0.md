# v12 phase 0: placed-circuit network with the compiler's layout

Frozen v11dl40 (control model), compiler's true layout as the placement; network trained on 1295 circuits × calibrations aug 0-79; evaluated on 577 control test circuits.

| | unseen calibrations (aug 80-99, local) | seen calibrations (local, in aug 8-79) |
| --- | --- | --- |
| v11 head, learned placement (the model as it is) | 0.737 | 0.692 |
| v11 head, true layout, network off | 0.714 | 0.683 |
| network, step 1000 | 0.749 | 0.717 |
| network, step 2000 | 0.723 | 0.713 |
| network, step 3000 | 0.760 | 0.717 |
| network, step 4000 | 0.765 | 0.736 |

Reference (report_tracking_oracle_usage.md, 300 pilot test circuits, 10 local calibrations): v11dl40 learned 0.68, + true layout 0.64, + true layout and counts 0.72; v8kdl40 0.69.

Go if the network clearly beats ~0.74 on unseen calibrations; a large seen − unseen gap means it memorises the 3 chips' calibrations.