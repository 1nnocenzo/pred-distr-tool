"""GNN-guided multi-device dispatch policy.

Core contribution: device selection from a unified circuit queue using
per-circuit, per-device fidelity predictions from the trained GNN.

Each pending circuit is assigned to the device that maximises its predicted
fidelity, subject to a spread-scaled load penalty that prevents all circuits
from piling onto a single device.  Within each device, circuits are ordered
by descending fidelity score (secondary role).

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
        predicted fidelity, penalised by a spread-scaled load term so that
        circuits with a strong device preference resist redistribution while
        device-agnostic circuits absorb load freely.

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
            For each circuit, both fidelity and load are min-max normalised
            across the available devices **at the moment of decision** into
            [0, 1].  Let::

                fid_norm[d]  = (fid[d]  - fid_min ) / (fid_max  - fid_min )
                load_norm[d] = (load[d] - load_min) / (load_max - load_min)

            (spans < 1e-9 collapse to 0.)  The circuit is assigned to::

                argmax_d  w * fid_norm[d] - (1 - w)**4 * load_norm[d]

            Both terms live on the same [0, 1] scale, and the load
            coefficient ``(1-w)**4`` is asymmetric on purpose: it decays
            faster than the linear fidelity coefficient ``w``, which shifts
            the crossover for the anti-correlated case (best-fid device is
            also most-loaded) towards lower weights, so even the low end of
            the sweep shows a visible departure from pure round-robin.

            - At w=0: picks the least-loaded device (pure round-robin).
            - At w=1: picks the highest-fidelity device (pure oracle).
            - Intermediate w: the crossover weight per circuit depends on
              where its preferred device sits in the current load order at
              the moment of decision.

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

            # Min-max normalise fidelity and load across available devices so
            # both terms share the [0, 1] range regardless of absolute scale.
            fid_vals = list(dev_fids.values())
            fid_min, fid_max = min(fid_vals), max(fid_vals)
            fid_span = fid_max - fid_min

            load_vals = [load[d] for d in dev_fids]
            load_min, load_max = min(load_vals), max(load_vals)
            load_span = load_max - load_min

            load_coef = (1.0 - w) ** 4 # used to be 3
            def _score(d: str) -> float:
                fid_norm = (dev_fids[d] - fid_min) / fid_span if fid_span > 0.0 else 0.0
                load_norm = (load[d] - load_min) / load_span if load_span > 1e-9 else 0.0
                return w * fid_norm - load_coef * load_norm

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
