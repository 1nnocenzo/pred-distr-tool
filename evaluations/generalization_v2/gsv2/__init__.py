"""Device-aware, log-fidelity predictor for the generalization study (v2).

Builds on ``evaluations/generalization/genstudy`` (splits, metrics, I/O,
reporting) and changes only the predictor and its training:

1. the model predicts ``log F`` instead of ``F``;
2. each device is described by its coupling graph and calibration data and
   encoded by a small GNN, so the predictor scores (circuit, device) pairs
   instead of emitting three fixed outputs;
3. training batches are sampled with family-balanced weights and early
   stopping uses whole held-out families (or the largest circuits, for size
   extrapolation) as validation data.
"""
