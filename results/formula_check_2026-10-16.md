# Formula check (2026-10-16): duration and ESP formulas with the compiler's true inputs

`python -m esptime.oracle` on 366 circuits (12 per family + 60 synthetic), original calibration, 1069 (circuit, device)
cases (313 routing-free, 756 routed). No network: every learned input is replaced by the truth. Numbers: `formula_check_2026-10-16.json`.

## Duration: relative error vs the label (median / p90)

| inputs | routing-free | routed |
| --- | --- | --- |
| logical schedule, table gate times | **0.8%** / 8.4% | 25% / 56% (bias 0.76) |
| + Sabre's true SWAPs per wire | | 14% / 37% |
| + true waits (= route stage targets) | | **6.4%** / 17% (4.9% with one constant c2) |
| Sabre's routed circuit replayed | | 4.5% / 12% |

## ESP: |Δ log ESP| in nats when F is exact (cases with ESP > 1e-3; 0.7 nats = ×2)

| formula inputs | routing-free | routed |
| --- | --- | --- |
| MQT definition recomputed from the compiled circuit | 0.000 | 0.000 |
| model formula, all inputs true (α = 1, β = 0) | **0.006** / p90 0.18 | **0.26** / p90 1.25 |
| same, without the true placement (device-mean T) | 0.055 | 0.43 |
| same, duration of the SWAP+wait schedule (no stage 2) | 0.051 | 0.34 |
| same, duration +5% | 0.22 | 0.44 |

Reference (earlier study): light compile 0.29, compiler seed floor 0.20, earlier ESP head 0.30-0.65.

## What it says

1. Labels: MQT's ESP counts the idle time once; decay = −Σ_active physical qubits idle/T exactly.
2. Routing-free: both formulas are essentially exact; the error will come from the learned placement and counts.
3. Routed duration: SWAPs alone leave 14%, the waits halve it; ~5% is left for stage 2 (c2) per circuit.
4. Routed ESP: the formula's own floor is 0.26 nats, about the light compile's 0.29. Cause: idle on logical wires
   is 11% below the physical idle in median (p10 −45%); 40% of the routed cases activate more physical qubits than
   logical ones, and logical busy time is 5% above the optimised circuit's.
5. ESP accuracy is set by the duration: +5% duration costs +0.18 nats routed and +0.21 routing-free, more than the
   formula floor. The placement adds 0.05-0.17.
