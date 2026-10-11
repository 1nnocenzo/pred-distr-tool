# Compiler self-consistency (the ceiling for a learned placement)

300 control test circuits × 3 devices × (original + 10 local variants) × 5 compiler seeds.

| quantity | between two compiler seeds | model (placement_probe.py) |
| --- | --- | --- |
| logical qubits on the same physical qubit, original calibration | 58% | – |
| occupancy correlation, original calibration | 0.80 | – |
| reaction: corr(Δu seed a, Δu seed b), original → variant | 0.87 | 0.30–0.34 (vs best of 3) |
| reaction: best of seeds 0–2 vs seed 3 or 4 | 0.86 | 0.30–0.34 |
| Δlog F, seed 0 vs seed 1 | 0.86 | – |
| Δlog F, best of seeds 0–2 vs best of 3–4 | 0.94 | local tracking 0.57–0.61 |

If the compiler's own reaction agreement is close to the model's, the placement is already near what is predictable; if it is much higher, the placement can still learn.