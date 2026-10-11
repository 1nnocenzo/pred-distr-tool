## 4b. v4 attention per circuit (control model seed 5, calibration-shift sample)
Script: `attention_per_circuit.py`.  Head-averaged attention of each logical qubit.

variant            top1 weight  top2 weight  distinct top / qubits  most attended node
EQE1_Top/orig      0.500        0.250        0.144                  8  (100% of qubits)
EQE1_Top/shuffle   0.500        0.250        0.144                  8  (100%)
QExa20/orig        0.500        0.250        0.144                  10 (100%)
QExa20/x2          0.500        0.250        0.144                  10 (100%)
QExa20/jitterA     0.500        0.250        0.144                  10 (100%)

Every logical qubit of every circuit puts half of its weight on the same physical
qubit of a device (and a quarter on a second one), whatever the circuit and the
calibration: the v4 "placement" is a fixed reference set of a few qubits.
