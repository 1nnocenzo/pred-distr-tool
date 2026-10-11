# v12 phase 1: placement and placed-circuit network trained together

Fine-tuned v11dl40 (control model) + network (init: phase0_network_30k.pth); placement loss region × 0.1 on the compiler's layout of each calibration; scheduled sampling p_true 1 → 0 over the first 60% of 20000 steps; 1295 training circuits × calibrations aug 0-79; 577 control test circuits.

| | tracking, unseen calibrations (aug 80-99, local) | tracking, seen calibrations | placement agreement (unseen) | R² of F, original calibration |
| --- | --- | --- | --- | --- |
| start (v11dl40 + phase-0 network), learned placement | 0.574 | 0.529 | 0.116 | 0.9899 |
| start (v11dl40 + phase-0 network), compiler's layout | 0.857 | 0.839 | 0.116 | 0.9963 |
| step 5000, learned placement | 0.699 | 0.660 | 0.113 | 0.9966 |
| step 5000, compiler's layout | 0.902 | 0.887 | 0.113 | 0.9978 |
| step 10000, learned placement | 0.711 | 0.699 | 0.112 | 0.9973 |
| step 10000, compiler's layout | 0.901 | 0.894 | 0.112 | 0.9978 |
| step 15000, learned placement | 0.705 | 0.707 | 0.112 | 0.9977 |
| step 15000, compiler's layout | 0.886 | 0.876 | 0.112 | 0.9957 |
| step 20000, learned placement | 0.718 | 0.727 | 0.112 | 0.9978 |
| step 20000, compiler's layout | 0.880 | 0.869 | 0.112 | 0.9945 |

Reference (phase 0, same circuits and calibrations): v11dl40 learned placement 0.737 unseen / 0.692 seen; frozen v11dl40 + network on the compiler's layout 0.857 / 0.839.

Go to the standard evaluation if the learned-placement row clearly beats 0.737 on unseen calibrations with R² of F unchanged.