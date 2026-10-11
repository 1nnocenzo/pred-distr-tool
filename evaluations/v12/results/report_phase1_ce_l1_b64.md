# v12 phase 1: placement and placed-circuit network trained together

Fine-tuned v11dl40 (control model) + network (init: phase0_network_30k.pth); placement loss ce × 1.0 on the compiler's layout of each calibration; scheduled sampling p_true 1 → 0 over the first 60% of 20000 steps (batch 64 over 2 GPU(s)); 1295 training circuits × calibrations aug 0-79; 577 control test circuits.

| | tracking, unseen calibrations (aug 80-99, local) | tracking, seen calibrations | placement agreement (unseen) | R² of F, original calibration |
| --- | --- | --- | --- | --- |
| start (v11dl40 + phase-0 network), learned placement | 0.574 | 0.529 | 0.116 | 0.9899 |
| start (v11dl40 + phase-0 network), compiler's layout | 0.857 | 0.839 | 0.116 | 0.9963 |
| step 5000, learned placement | 0.622 | 0.594 | 0.271 | 0.9958 |
| step 5000, compiler's layout | 0.893 | 0.884 | 0.271 | 0.9978 |
| step 10000, learned placement | 0.642 | 0.636 | 0.289 | 0.9951 |
| step 10000, compiler's layout | 0.889 | 0.875 | 0.289 | 0.9970 |
| step 15000, learned placement | 0.638 | 0.646 | 0.299 | 0.9957 |
| step 15000, compiler's layout | 0.866 | 0.846 | 0.299 | 0.9964 |
| step 20000, learned placement | 0.620 | 0.628 | 0.303 | 0.9956 |
| step 20000, compiler's layout | 0.850 | 0.827 | 0.303 | 0.9963 |

Reference (phase 0, same circuits and calibrations): v11dl40 learned placement 0.737 unseen / 0.692 seen; frozen v11dl40 + network on the compiler's layout 0.857 / 0.839.

Go to the standard evaluation if the learned-placement row clearly beats 0.737 on unseen calibrations with R² of F unchanged.