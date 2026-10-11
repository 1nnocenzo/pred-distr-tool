# v12 phase 1: placement and placed-circuit network trained together

Fine-tuned v11dl40 (control model) + network (init: phase0_network_30k.pth); placement loss both × 1.0 on the compiler's layout of each calibration; scheduled sampling p_true 1 → 0 over the first 60% of 20000 steps (batch 64 over 2 GPU(s)); 1295 training circuits × calibrations aug 0-79; 577 control test circuits.

| | tracking, unseen calibrations (aug 80-99, local) | tracking, seen calibrations | placement agreement (unseen) | R² of F, original calibration |
| --- | --- | --- | --- | --- |
| start (v11dl40 + phase-0 network), learned placement | 0.574 | 0.529 | 0.116 | 0.9899 |
| start (v11dl40 + phase-0 network), compiler's layout | 0.857 | 0.839 | 0.116 | 0.9963 |
| step 5000, learned placement | 0.647 | 0.619 | 0.267 | 0.9956 |
| step 5000, compiler's layout | 0.889 | 0.877 | 0.267 | 0.9974 |
| step 10000, learned placement | 0.647 | 0.622 | 0.291 | 0.9950 |
| step 10000, compiler's layout | 0.886 | 0.879 | 0.291 | 0.9966 |
| step 15000, learned placement | 0.628 | 0.636 | 0.300 | 0.9952 |
| step 15000, compiler's layout | 0.856 | 0.852 | 0.300 | 0.9952 |
| step 20000, learned placement | 0.615 | 0.627 | 0.305 | 0.9955 |
| step 20000, compiler's layout | 0.857 | 0.848 | 0.305 | 0.9965 |

Reference (phase 0, same circuits and calibrations): v11dl40 learned placement 0.737 unseen / 0.692 seen; frozen v11dl40 + network on the compiler's layout 0.857 / 0.839.

Go to the standard evaluation if the learned-placement row clearly beats 0.737 on unseen calibrations with R² of F unchanged.