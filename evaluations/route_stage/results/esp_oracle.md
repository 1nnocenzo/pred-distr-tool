# ESP decay from the routed schedule (oracle)

Routed cases; x = Σ idle (µs) × mean 1/min(T1,T2) (whole device, or the best-T qubits as many as used); decay ≈ −k·x with k fitted on control (median ratio). Error = |Δ log ESP| median [p90] (×100 ≈ %).

| group | n | true decay median | k·x_mean: err | k·x_best: err | duration-only (n·dur): err |
| --- | --- | --- | --- | --- | --- |
| control | 11456 | -16.289 | 1.434 [8.433] | 2.272 [14.177] | 2.565 [15.838] |
| qaoa | 6905 | -12.939 | 1.253 [7.309] | 1.580 [14.870] | 2.461 [6.872] |
| variational | 10170 | -6.293 | 0.678 [2.921] | 1.034 [7.799] | 2.189 [7.028] |

k: x_mean 1.114, x_best 1.646, x_dur 0.618
