"""GNN-guided multi-device dispatch policy.

Core contribution: device selection from a unified circuit queue using
per-circuit, per-device fidelity predictions from the trained GNN.

Each pending circuit is assigned to the device that maximises its predicted
fidelity, subject to a load-share penalty that prevents all circuits from
piling onto a single device.  Within each device, circuits are ordered by
descending fidelity score (secondary role).

- There is **one shared queue** for all devices.
- QMS calls ``select()`` once per dispatch tick and receives back a
  ``dict[device_name, list[job_ids]]`` — device selection and ordering in
  a single call.
- No external ``assign_circuits()`` step is needed; routing is fully inside
  the policy.

Configuration keys (via ``configure()``):
    fidelity_weight    float  0 = max load balance, 1 = pure fidelity. Default 0.7.
    fidelity_threshold float  Hard minimum fidelity; circuit is skipped if all
                              devices score below this. Default 0.0.
    max_batch_size     int    Max circuits dispatched per select() call
                              (summed across all devices). Default 8.
    checkpoint_path    str    Path to GNN weights (optional).
    params_path        str    Path to GNN hyper-params (optional).

Usage::

    import gnn_device_policy  # auto-registers "gnn_device" with QMS
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Path setup: make src/qms and src/model importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "src" / "model"

for _p in (str(_REPO_ROOT / "src"), str(_MODEL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn import GNN  # noqa: E402
from encoding import create_dag, get_gnn_input_features  # noqa: E402
from qms.policy.base import MultiDevicePolicy, MultiDeviceSnapshot  # noqa: E402
from qms.policy import register_multi_policy  # noqa: E402

logger = logging.getLogger(__name__)

# Devices the GNN was trained on — order matches model output indices.
DEVICE_NAMES = ["EQE1_Top", "EQE1_Bottom", "QExa20"]
NUM_DEVICES = len(DEVICE_NAMES)


# ---------------------------------------------------------------------------
# GNN predictor
# ---------------------------------------------------------------------------

class _GNNPredictor:
    """Loads the trained GNN checkpoint and predicts per-device fidelity."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        params_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        checkpoint_path = Path(checkpoint_path or _MODEL_DIR / "best_model.pth")
        params_path = Path(params_path or _MODEL_DIR / "best_params.json")
        self.device = torch.device(device)

        with open(params_path) as f:
            params = json.load(f)

        # Architecture knobs are made explicit here to avoid silent drift
        # from gnn.py defaults.  If the checkpoint was tuned with e.g.
        # sag_pool=True, add the key to best_params.json and it will flow
        # through.
        self.model = GNN(
            in_feats=get_gnn_input_features(),
            hidden_dim=params["hidden_dim"],
            num_conv_wo_resnet=params["num_conv_wo_resnet"],
            num_resnet_layers=params["num_resnet_layers"],
            mlp_units=params["mlp"],
            dropout_p=params["dropout"],
            bidirectional=params["bidirectional"],
            output_dim=NUM_DEVICES,
            use_sag_pool=params.get("sag_pool", False),
            sag_ratio=params.get("sag_ratio", 0.7),
            combine=params.get("combine", "sum"),
            readout=params.get("readout", "mean"),
        )
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, data: Data) -> list[float]:
        """Return predicted fidelities [EQE1_Top, EQE1_Bottom, QExa20]."""
        data = data.to(self.device)
        if data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(data)
        return out.squeeze(0).tolist()


def _circuit_to_data(qasm_path: str | Path) -> Data:
    """Load an OpenQASM 3 file and convert to a PyG Data object."""
    from qiskit import qasm3
    qc = qasm3.load(str(qasm_path))
    node_features, edge_index, _ = create_dag(qc)
    return Data(x=node_features, edge_index=edge_index)


# ---------------------------------------------------------------------------
# GNN device selection policy
# ---------------------------------------------------------------------------

class GNNDevicePolicy(MultiDevicePolicy):
    """Select devices for circuits using GNN-predicted fidelity.

    Primary — device selection:
        Each circuit is scored against every available device using its
        raw predicted fidelity, penalised by the device's current share of
        dispatched load, so a device attracts circuits in proportion to
        its genuine fidelity edge over the alternatives.

    Secondary — within-device ordering:
        Circuits assigned to the same device are ordered by descending
        fidelity score on that device.

    Fidelity predictions are read from ``predicted_fidelities`` in job
    metadata if present (pre-computed path); otherwise the GNN is invoked
    live using ``qasm_path`` or ``circuit_source`` + job_id.
    """

    def __init__(self) -> None:
        self._fidelity_weight: float = 0.7
        self._fidelity_threshold: float = 0.0
        self._max_batch_size: int = 8
        self._checkpoint_path: str | None = None
        self._params_path: str | None = None
        self._predictor: _GNNPredictor | None = None

    def configure(self, config: dict[str, Any]) -> None:
        self._fidelity_weight = float(config.get("fidelity_weight", 0.7))
        self._fidelity_threshold = float(config.get("fidelity_threshold", 0.0))
        self._max_batch_size = int(config.get("max_batch_size", 8))
        self._checkpoint_path = config.get("checkpoint_path")
        self._params_path = config.get("params_path")

    # -- internals --

    def _get_predictor(self) -> _GNNPredictor:
        if self._predictor is None:
            self._predictor = _GNNPredictor(
                checkpoint_path=self._checkpoint_path,
                params_path=self._params_path,
            )
        return self._predictor

    @staticmethod
    def _resolve_qasm_path(job_id: str, meta: dict[str, Any]) -> Path | None:
        if "qasm_path" in meta:
            p = Path(meta["qasm_path"])
            return p if p.is_absolute() else _REPO_ROOT / p
        if "circuit_source" in meta:
            base = Path(meta["circuit_source"])
            if not base.is_absolute():
                base = _REPO_ROOT / base
            return base / f"{job_id}.qasm"
        return None

    def _get_fidelities(
        self, predictor: _GNNPredictor, job_id: str, meta: dict[str, Any],
    ) -> list[float] | None:
        """Return per-device fidelity list, or None if unavailable."""
        pre = meta.get("predicted_fidelities")
        if pre is not None:
            return [float(f) for f in pre]

        qasm_path = self._resolve_qasm_path(job_id, meta)
        if qasm_path is None or not qasm_path.exists():
            logger.warning("No QASM file for job %s; skipping.", job_id)
            return None
        try:
            data = _circuit_to_data(qasm_path)
            return predictor.predict(data)
        except Exception:
            logger.exception("GNN inference failed for job %s", job_id)
            return None

    # -- round-robin fallback --

    def _round_robin(
        self, snapshot: MultiDeviceSnapshot, available: list[str],
    ) -> dict[str, list[str]]:
        """Assign circuits to devices in round-robin order (no GNN)."""
        slots = {dev: snapshot.devices[dev].available_slots for dev in available}
        total_slots = sum(slots.values())
        result: dict[str, list[str]] = {dev: [] for dev in snapshot.devices}
        dev_cycle = [d for d in available if slots[d] > 0]
        idx = 0
        dispatched = 0
        for job_id, _meta in snapshot.pending:
            if dispatched >= min(self._max_batch_size, total_slots) or not dev_cycle:
                break
            dev = dev_cycle[idx % len(dev_cycle)]
            result[dev].append(job_id)
            slots[dev] -= 1
            if slots[dev] <= 0:
                dev_cycle = [d for d in dev_cycle if slots[d] > 0]
            idx += 1
            dispatched += 1
        return result

    # -- core select --

    def select(self, snapshot: MultiDeviceSnapshot) -> dict[str, list[str]]:
        """Assign pending circuits to devices and order within each device.

        Device selection (primary):
            Each circuit is assigned to::

                argmax_d  w * fid[d] - (1 - w) * load_share[d]

            where ``fid[d]`` is the raw predicted fidelity (in [0, 1]) and
            ``load_share[d]`` is the fraction of circuits dispatched to
            ``d`` so far in this pass (0 while nothing has been
            dispatched).  See the inline comment in the loop for why this
            trades off fidelity against balance smoothly across the whole
            weight range.

            - At w=0: picks the least-loaded device (pure round-robin).
            - At w=1: picks the highest-fidelity device (argmax of the
              policy's own predictions).
            - Intermediate w: devices fill until per-device load shares
              offset the fidelity gaps, so the allocation moves
              continuously from balanced to fidelity-optimal as w grows.

        Within-device ordering (secondary):
            Circuits assigned to a device are sorted by descending fidelity
            on that device.
        """
        available = snapshot.available_devices()
        if not available:
            return {dev: [] for dev in snapshot.devices}

        w = self._fidelity_weight

        # Pure round-robin when fidelity is irrelevant — skip GNN entirely.
        if w == 0.0:
            return self._round_robin(snapshot, available)

        # Map device name → index in DEVICE_NAMES for fidelity lookup
        dev_to_idx: dict[str, int] = {
            name: DEVICE_NAMES.index(name)
            for name in available
            if name in DEVICE_NAMES
        }
        if not dev_to_idx:
            return {dev: [] for dev in snapshot.devices}

        predictor = self._get_predictor()
        slots = {dev: snapshot.devices[dev].available_slots for dev in available}
        load = {dev: 0 for dev in available}
        total_slots = sum(slots.values())

        # Accumulate per-device assignments: (job_id, fidelity_on_device)
        assigned: dict[str, list[tuple[str, float]]] = {dev: [] for dev in snapshot.devices}
        dispatched_count = 0

        for job_id, meta in snapshot.pending:
            if dispatched_count >= min(self._max_batch_size, total_slots):
                break

            fids_all = self._get_fidelities(predictor, job_id, meta)
            if fids_all is None:
                continue

            # Per-device fidelity for available devices only
            dev_fids: dict[str, float] = {
                dev: fids_all[idx]
                for dev, idx in dev_to_idx.items()
                if slots[dev] > 0
            }
            if not dev_fids:
                break  # all slots filled

            # Hard threshold: skip if no device meets minimum fidelity
            if max(dev_fids.values()) < self._fidelity_threshold:
                continue

            # Score:  w * fid[d] - (1 - w) * load_share[d]
            #
            # Raw fidelities (already in [0, 1]) trade off against the
            # device's share of the circuits dispatched so far, so both
            # terms live on the same scale without per-circuit min-max
            # normalisation.  Keeping raw magnitudes matters: a device with
            # a 0.001 fidelity edge barely outbids the load term, while a
            # 0.3 edge dominates it.  (The previous formula normalised
            # fidelity per circuit, which erased those magnitudes and made
            # every circuit's decision a step function of w; combined with
            # the quartic (1-w)**4 coefficient this confined the entire
            # balance→fidelity transition to w ≈ 0.28–0.55.)
            #
            # Because the penalty grows with the actual share imbalance,
            # devices fill until the marginal share gap offsets the fidelity
            # gap — at equilibrium, for any two devices a, b::
            #
            #     share[a] - share[b]  ≈  w / (1 - w) * (fid[a] - fid[b])
            #
            # so the allocation shifts continuously across the whole
            # w ∈ (0, 1) range instead of flipping en masse in a narrow
            # band.  Dividing by the dispatched total also makes the
            # penalty scale-invariant in queue length.
            total_load = dispatched_count

            def _score(d: str) -> float:
                share = load[d] / total_load if total_load > 0 else 0.0
                return w * dev_fids[d] - (1.0 - w) * share

            best_dev = max(dev_fids, key=_score)

            assigned[best_dev].append((job_id, dev_fids[best_dev]))
            slots[best_dev] -= 1
            load[best_dev] += 1
            dispatched_count += 1

        # Secondary: sort each device's list by descending fidelity
        result: dict[str, list[str]] = {}
        for dev in snapshot.devices:
            pairs = sorted(assigned[dev], key=lambda t: t[1], reverse=True)
            result[dev] = [job_id for job_id, _ in pairs]

        return result


# ---------------------------------------------------------------------------
# Auto-register with QMS when this module is imported
# ---------------------------------------------------------------------------
register_multi_policy("gnn_device", GNNDevicePolicy)
