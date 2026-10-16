"""End-to-end check on a handful of real compilations, no prepared data needed (~1 min on CPU).

    python -m pytest -q tests/test_model.py      (or: python tests/test_model.py)
Compiles 6 small circuits on the 3 devices (labels.compile_case), builds the items (dataset.build_item), runs the
model forward / losses / backward with and without a calibration variant, and checks the schedules and labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esptime import labels as LB  # noqa: E402
from esptime import losses as L  # noqa: E402
from esptime.config import DEVICES, decomp_table  # noqa: E402
from esptime.dataset import build_item, schedule_np  # noqa: E402
from esptime.devices import backends, chips, device_batch, raw_graph  # noqa: E402
from esptime.encoding import to_basis  # noqa: E402
from esptime.model import ESPTimeModel, schedule_torch  # noqa: E402
from esptime.synth import generic, layered  # noqa: E402


def _circuits():
    from qiskit import QuantumCircuit
    ghz = QuantumCircuit(5)
    ghz.h(0)
    for i in range(4):
        ghz.cx(i, i + 1)
    ghz.measure_all()
    star = QuantumCircuit(8)                     # needs routing on every device
    for i in range(1, 8):
        star.cx(0, i)
    for i in range(7):
        star.rzz(0.3, i, i + 1)
    star.measure_all()
    return [("t/ghz", ghz), ("t/star", star)] + [("synth/" + n, qc) for n, qc in
                                                 (generic(3), generic(8), layered(100_000), layered(100_001))]


def build():
    LB.TAB = decomp_table()
    bes = {c: backends(c) for c in ("orig", "aug0")}
    raws = {c: {d: raw_graph(be) for d, be in bes[c].items()} for c in bes}
    chs, table = chips(raws["orig"]), decomp_table()
    items = []
    for name, qc in _circuits():
        qb = to_basis(qc)
        nodes = [(i.operation.name, [qb.find_bit(q).index for q in i.qubits]) for i in qb.data]
        lab = {"n_nodes": len(nodes), "nq": qb.num_qubits,
               "calib": {c: {d: LB.compile_case(qb, nodes, bes[c][d], route=c == "orig") for d in DEVICES} for c in bes}}
        for c in bes:
            for d in DEVICES:
                r = lab["calib"][c][d]
                assert r["decay"] is not None and r["decay"] <= 1e-12 and r["logF"] <= 0 and r["dur"] > 0, (name, c, d, r)
        items.append(build_item(name, qb, lab, ["orig", "aug0"], chs, table))
    return items, raws


def test_pipeline():
    from torch_geometric.loader import DataLoader
    torch.manual_seed(0)
    items, raws = build()
    assert any(bool(it.routed.any()) for it in items), "no routed case among the test circuits"
    dev = device_batch(raws["orig"], raws["orig"])
    dev_v = device_batch(raws["aug0"], raws["orig"])
    assert torch.allclose(dev.coh, dev_v.coh) and not torch.allclose(dev.x, dev_v.x)
    b = next(iter(DataLoader(items, batch_size=len(items))))
    # the level-wise torch schedule = the sequential one (no SWAPs, no waits)
    zero = torch.zeros(b.num_nodes, len(DEVICES), 3)
    st = schedule_torch(b, zero, torch.zeros(b.num_nodes, len(DEVICES)))
    for i, it in enumerate(items):
        wires = [[int(q) for q in row if q >= 0] for row in it.gate_qubits.numpy()]
        assert abs(float(st[i, 0]) - schedule_np(wires, it.t.numpy())) < 1e-3
    # Sabre's SWAPs + teacher-forced waits reproduce the replayed routed schedule
    sch = schedule_torch(b, b.s, b.wait / L.T_SWAP)
    m = b.routed
    rel = (sch[m] / b.replay[m] - 1).abs()
    assert float(rel.median()) < 0.02 and float(rel.max()) < 0.1, rel      # 1q gates are re-synthesised by routing
    model = ESPTimeModel(dev)
    out = model(b, dev, mc=4)
    for k in ("logf", "dur", "dur_mc", "decay", "logesp"):
        assert out[k].shape == (len(items), len(DEVICES)) and torch.isfinite(out[k]).all(), k
    assert (out["logf"] <= 0).all() and (out["decay"] <= 0).all() and (out["dur"] > 0).all()
    assert (out["n_extra"] >= 0).all() and (out["n_extra"][b.rf_pred > 0.5] == 0).all()   # none without routing
    assert torch.isfinite(b.n_extra).all() and (b.n_extra[b.rf] == 0).all()
    loss, parts = L.total(out, b, model.backbone.hops)
    out_r = ESPTimeModel(dev, esp_busy="route")(b, dev)
    assert torch.isfinite(out_r["decay"]).all() and (out_r["decay"] <= 0).all()
    out_v = model(b, dev_v)
    loss = loss + L.calib_loss(out, out_v, b, 0)
    loss.backward()
    assert all(np.isfinite(v) for v in parts.values()), parts
    grads = {n.split(".")[0]: p.grad is not None and bool(torch.isfinite(p.grad).all())
             for n, p in model.named_parameters() if p.requires_grad}
    assert grads["backbone"] and grads["route"] and grads["esp"], grads
    print("parts", {k: round(v, 4) for k, v in parts.items()})


if __name__ == "__main__":
    test_pipeline()
    print("all checks passed")
