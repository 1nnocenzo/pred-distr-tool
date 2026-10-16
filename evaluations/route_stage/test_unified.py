#!/usr/bin/env python3
"""Smoke test of unified.py (v29) on small real circuits with synthetic labels and the quirks of the data (no dataset,
no trained weights; ~30 s on CPU): routing-free device (NaN replay), a device neither routed nor routing-free, a circuit
without F / decay labels, a NaN decay, measurements, 1/2/3-qubit gates.  Checks: the backbone's F is unchanged by
switching its ESP head off; MC schedule over D·K columns == stage1.schedule_k on the same samples; finite outputs,
losses and gradients; frozen backbone gets no gradient; one optimiser step; evaluate / summary / timing; checkpoint
round trip without the backbone weights.  Run before pushing changes to unified.py or stage1.py:
    PYTHONPATH=src python evaluations/route_stage/test_unified.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified as U  # noqa: E402
S = U.S
from qiskit import QuantumCircuit  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402


def circuit(rng, n):
    qc = QuantumCircuit(n, n)
    for _ in range(int(rng.integers(8, 30))):
        k = rng.random()
        qs = [int(x) for x in rng.choice(n, size=3, replace=False)]
        if k < 0.35:
            qc.cx(qs[0], qs[1])
        elif k < 0.45 and n >= 3:
            qc.ccx(*qs)
        elif k < 0.7:
            qc.rz(float(rng.uniform(0, 3)), qs[0])
        else:
            qc.h(qs[0])
    qc.measure(range(n), range(n))
    return qc


def make_item(rng, name, chips, tr, tc, labelled=True):
    n = int(rng.integers(3, 8))
    g = S.fe.to_data(circuit(rng, n))
    g.name = name
    it = U.live_item(g, chips, tr, tc)
    N, D = it.num_nodes, len(S.DEVS)
    gq = it.gate_qubits.numpy()
    logical = S.schedule_np(gq, it.t.numpy(), np.zeros((N, 3)), np.zeros(N), np.zeros(N, bool))
    it.circuit_name = it.name = name
    it.s = torch.from_numpy(rng.poisson(0.3, (N, D, 3)).astype(np.float32))
    it.wait, it.m = torch.randn(N, D), torch.rand(N, D) < 0.5
    it.rf, it.routed = torch.tensor([[True, False, False]]), torch.tensor([[False, True, False]])
    it.replay = torch.tensor([[float("nan"), logical * 1.3, float("nan")]])
    it.final = torch.tensor([[logical, logical * 1.4, logical * 1.2]])
    it.c2, it.ok2 = torch.zeros(1, D), torch.tensor([[True, True, False]])
    it.logical, it.prior_dur = torch.tensor([[logical]]), torch.full((1, D), logical * 1.5)
    return it, g


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    params, devices = U.device_batch()
    from esp_router import chips as mk_chips
    chips = mk_chips(devices)
    import json
    tab = json.loads((U.EVAL / "esp/results/decomp_table.json").read_text())
    UU = list(S.fe.UNIQUE)
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in UU], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in UU], float)
    pairs = [make_item(rng, f"c{i}", chips, tr, tc) for i in range(6)]
    items = [p[0] for p in pairs]
    data = {it.name: it for it in items}
    ds = {f"c{i}": {"F": torch.tensor([0.5, 0.3, 0.2]), "decay": torch.tensor([-0.5, -3.0, -1.0]),
                    "duration": torch.tensor([20e-6, 30e-6, 25e-6])} for i in range(4)}
    ds["c3"]["decay"] = torch.tensor([float("nan"), -3.0, -1.0])               # NaN decay; c4, c5 unlabelled
    U.attach_labels(data, ds)
    assert not items[4].esp_ok.any() and not bool(items[3].esp_ok[0, 0]), "bad labels not masked"

    # backbone: F unchanged by skip_esp
    fm = U.backbone(params, devices, None, "cpu", allow_untrained=True)
    b = next(iter(DataLoader(items, batch_size=len(items))))
    fm.eval()
    with torch.no_grad():
        f_skip = fm(b, devices)
        fm.skip_esp = False
        f_full = fm(b, devices)
        fm.skip_esp = True
    assert torch.allclose(f_skip, f_full), "skip_esp changed F"

    model = U.Unified(fm, items[0].x.size(1) + 9 + 12, hidden=32, layers=2)
    model.train()
    out = model(b, devices, mc=4, gen=torch.Generator().manual_seed(0))
    for k in ("logf", "dur", "dur_mc", "decay", "logesp"):
        assert torch.isfinite(out[k]).all(), f"{k} not finite"
    assert (out["decay"] <= 0).all(), "positive decay"

    # MC over D·K columns == schedule_k on the same samples
    nr = (~out["rf"])[b.batch]
    rate_e = (out["rate"] * nr.unsqueeze(-1)).detach()
    wait_e = torch.where(nr, out["wait"], torch.zeros_like(out["wait"])).detach()
    K = 3
    N, D, _ = rate_e.shape
    smp = torch.poisson(rate_e.unsqueeze(-1).expand(N, D, 3, K).contiguous())
    tor = S.schedule_torch(b, smp.permute(0, 1, 3, 2).reshape(N, D * K, 3), wait_e.unsqueeze(-1).expand(N, D, K).reshape(N, D * K)).view(-1, D, K)
    ptr, diff = b.ptr.tolist(), 0.0
    for i, it in enumerate(items):
        a, e = ptr[i], ptr[i + 1]
        gq, t = it.gate_qubits.numpy(), it.t.numpy()
        meas = it.x[:, S.GATE2IDX_MEASURE].numpy() > 0.5
        elig = ((gq >= 0).sum(1) >= 2) | meas
        slot = ((gq >= 0) & elig[:, None])[..., None]
        for d in range(D):
            ref = S.schedule_k(gq, t, smp[a:e, d].numpy() * slot, wait_e[a:e, d].numpy() * S.T_SWAP, elig)
            diff = max(diff, float(np.abs(ref - tor[i, d].numpy()).max()))
    assert diff < 1e-3, f"MC schedule mismatch {diff}"

    # losses, gradients, frozen backbone, one step
    l_s, l_w, l_c = S.losses(b, out["rate"], out["wait"], out["c2"])
    l_d = S.duration_loss(b, out["rate"], out["wait"])
    l_e = U.esp_loss(b, out)
    for nm, v in (("swap", l_s), ("wait", l_w), ("stage2", l_c), ("duration", l_d), ("esp", l_e)):
        assert torch.isfinite(v), f"{nm} loss not finite: {v}"
    loss = l_s + 0.3 * l_w + l_c + l_d + l_e
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(); loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in {bad}"
    assert all(p.grad is None for p in model.fmod.parameters()), "frozen backbone got gradients"
    assert model.esp[-1].weight.grad is not None and model.esp[-1].weight.grad.abs().sum() > 0, "ESP head got no gradient"
    opt.step()
    # esp_grad=route: the decay loss reaches the route stage
    m2 = U.Unified(fm, items[0].x.size(1) + 9 + 12, hidden=32, layers=2, esp_grad="route")
    U.esp_loss(b, m2(b, devices)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m2.route.head.parameters()), "route got no ESP gradient"

    rows = U.evaluate(model, devices, items, bs=4, mc=4, ds=ds)
    assert rows and all(np.isfinite(r["pred"]) and np.isfinite(r["logESP"]) for r in rows), "evaluate rows not finite"
    summ = U.summary(rows)
    assert "esp" in summ and "routed" in summ, f"summary incomplete: {list(summ)}"
    tres, trows = U.timing(model, devices, items, 3, 4)
    assert tres["n"] == 3 and tres["live_vs_stored_dev_f_max_abs_diff"] < 1e-5, tres
    # checkpoint without the backbone, loaded into a fresh model on the same backbone
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.pt"
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("fmod.")}, p)
        m3 = U.Unified(fm, items[0].x.size(1) + 9 + 12, hidden=32, layers=2)
        missing, unexpected = m3.load_state_dict(torch.load(p), strict=False)
        assert not unexpected and all(k.startswith("fmod.") for k in missing), (missing, unexpected)
        m3.eval(); model.eval()
        with torch.no_grad():
            assert torch.allclose(m3(b, devices)["logesp"], model(b, devices)["logesp"]), "checkpoint round trip differs"
    print(f"F unchanged by skip_esp; MC diff {diff:.1e}; losses swap {l_s.item():.3f} wait {l_w.item():.3f} "
          f"stage2 {l_c.item():.3f} duration {l_d.item():.3f} esp {l_e.item():.3f}; {len(rows)} rows; "
          f"timing {tres['all_total_ms_median']:.1f} ms median (untrained, tiny circuits)")
    print("all checks passed")


if __name__ == "__main__":
    main()
