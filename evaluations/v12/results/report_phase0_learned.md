# v12 phase 0: placed-circuit network with the compiler's layout

Frozen v11dl40 (control model), compiler's true layout as the placement; network trained on 1295 circuits × calibrations aug 0-79; evaluated on 577 control test circuits.

| | unseen calibrations (aug 80-99, local) | seen calibrations (local, in aug 8-79) |
| --- | --- | --- |
| v11 head, learned placement (the model as it is) | 0.737 | 0.692 |
| v11 head, true layout, network off | 0.714 | 0.683 |
| network, step 5000 | 0.713 | 0.680 |
| network on the learned placement, step 5000 | 0.702 | 0.676 |
| network, step 10000 | 0.712 | 0.676 |
| network on the learned placement, step 10000 | 0.708 | 0.681 |
| network, step 15000 | 0.707 | 0.667 |
| network on the learned placement, step 15000 | 0.708 | 0.683 |
| network, step 20000 | 0.707 | 0.680 |
| network on the learned placement, step 20000 | 0.698 | 0.677 |

Reference (report_tracking_oracle_usage.md, 300 pilot test circuits, 10 local calibrations): v11dl40 learned 0.68, + true layout 0.64, + true layout and counts 0.72; v8kdl40 0.69.

Go if the network clearly beats ~0.74 on unseen calibrations; a large seen − unseen gap means it memorises the 3 chips' calibrations.