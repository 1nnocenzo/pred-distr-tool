# v12 phase 1: placement and placed-circuit network trained together

Fine-tuned v11dl40 (control model) + network (init: phase0_network_30k.pth); placement loss both × 0.1 on the compiler's layout of each calibration; scheduled sampling p_true 1 → 0 over the first 60% of 20000 steps; 1295 training circuits × calibrations aug 0-79; 577 control test circuits.

| | tracking, unseen calibrations (aug 80-99, local) | tracking, seen calibrations | placement agreement (unseen) | R² of F, original calibration |
| --- | --- | --- | --- | --- |
| start (v11dl40 + phase-0 network), learned placement | 0.574 | 0.529 | 0.116 | 0.9899 |
| start (v11dl40 + phase-0 network), compiler's layout | 0.857 | 0.839 | 0.116 | 0.9963 |
| step 5000, learned placement | 0.648 | 0.615 | 0.236 | 0.9962 |
| step 5000, compiler's layout | 0.901 | 0.883 | 0.236 | 0.9978 |
| step 10000, learned placement | 0.675 | 0.651 | 0.262 | 0.9965 |
| step 10000, compiler's layout | 0.905 | 0.898 | 0.262 | 0.9976 |
| step 15000, learned placement | 0.690 | 0.678 | 0.277 | 0.9970 |
| step 15000, compiler's layout | 0.885 | 0.873 | 0.277 | 0.9963 |
| step 20000, learned placement | 0.690 | 0.695 | 0.285 | 0.9968 |
| step 20000, compiler's layout | 0.867 | 0.856 | 0.285 | 0.9962 |

Reference (phase 0, same circuits and calibrations): v11dl40 learned placement 0.737 unseen / 0.692 seen; frozen v11dl40 + network on the compiler's layout 0.857 / 0.839.

Go to the standard evaluation if the learned-placement row clearly beats 0.737 on unseen calibrations with R² of F unchanged.