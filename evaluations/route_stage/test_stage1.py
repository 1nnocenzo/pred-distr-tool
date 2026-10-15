#!/usr/bin/env python3
"""Smoke test of stage1.py on synthetic items with the quirks of the real data (no dataset needed, ~10 s on CPU):
routing-free devices with NaN replay, a device that is neither routed nor routing-free, measurements, 1/2/3-qubit
gates, gates without operands.  Checks: torch schedule == numpy schedule, finite losses and gradients for v24 / v25 /
v26, one optimiser step, evaluate() rows.  Run before pushing changes to stage1.py:
    PYTHONPATH=src python evaluations/route_stage/test_stage1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1 as S  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

F_X = S.GATE2IDX_MEASURE + 8


def item(rng, name, contention):
    nq, N = int(rng.integers(3, 8)), int(rng.integers(15, 60))
    gq = np.full((N, 3), -1)
    x = torch.zeros(N, F_X)
    for i in range(N):
        k = int(rng.choice([0, 1, 2, 2, 3]))
        if k:
            gq[i, :k] = rng.choice(nq, size=k, replace=False)
        if k == 1 and rng.random() < 0.2:
            x[i, S.GATE2IDX_MEASURE] = 1
    prev, lvl = S.wire_prev(gq)
    D = len(S.DEVS)
    rf = torch.tensor([[True, False, False]])                      # device 0 routing-free (replay NaN)
    routed = torch.tensor([[False, True, False]])                  # device 2: neither (not aligned)
    replay = torch.tensor([[float("nan"), 20.0, float("nan")]])
    t = rng.uniform(0.04, 0.4, N).astype(np.float32)
    d = dict(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), gate_qubits=torch.from_numpy(gq), n_qubits=nq,
             prev=torch.from_numpy(prev), lvl=torch.from_numpy(lvl), wf=torch.zeros(N, 9),
             dev_f=torch.zeros(N, 12 + (3 if contention else 0)), t=torch.from_numpy(t),
             c2=torch.zeros(1, D), ok2=torch.tensor([[True, True, False]]), logical=torch.tensor([[10.0]]),
             s=torch.from_numpy(rng.poisson(0.3, (N, D, 3)).astype(np.float32)), wait=torch.randn(N, D),
             m=torch.rand(N, D) < 0.5, routed=routed, rf=rf, final=torch.tensor([[10.0, 22.0, 15.0]]), replay=replay,
             prior_dur=torch.full((1, D), 18.0), name=name, num_nodes=N)
    if contention:
        d["pw"] = torch.rand(N, D) * S.T_SWAP
    return Data(**d), gq, t, x


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for contention in (False, True):
        items = [item(rng, f"c{i}", contention) for i in range(6)]
        b = next(iter(DataLoader([it for it, *_ in items], batch_size=6)))
        model = S.Stage1(F_X + 9 + 12 + (3 if contention else 0), hidden=32, layers=2, contention=contention)
        rate, wait, c2 = model(b)
        # torch schedule == numpy schedule on the model's own outputs
        dur = S.schedule_torch(b, rate, wait)
        ptr = b.ptr.tolist()
        diff = 0.0
        for i, (_, gq, t, x) in enumerate(items):
            a, e = ptr[i], ptr[i + 1]
            meas = x[:, S.GATE2IDX_MEASURE].numpy() > 0.5
            elig = ((gq >= 0).sum(1) >= 2) | meas
            slot = (gq >= 0) & elig[:, None]
            for d in range(len(S.DEVS)):
                ref = S.schedule_np(gq, t, rate[a:e, d].detach().numpy() * slot, wait[a:e, d].detach().numpy() * S.T_SWAP, elig)
                diff = max(diff, abs(ref - float(dur[i, d])))
        assert diff < 1e-3, f"schedule mismatch {diff}"
        l_s, l_w, l_c = S.losses(b, rate, wait, c2)
        l_d = S.duration_loss(b, rate, wait)
        loss = l_s + 0.3 * l_w + l_c + l_d
        for nm, v in (("swap", l_s), ("wait", l_w), ("stage2", l_c), ("duration", l_d)):
            assert torch.isfinite(v), f"{nm} loss not finite: {v}"
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad(); loss.backward()
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, f"non-finite gradients in {bad}"
        opt.step()
        rows = S.evaluate(model, [it for it, *_ in items], bs=4, mc=2)
        assert rows and all(np.isfinite(r["pred"]) for r in rows), "evaluate rows not finite"
        print(f"contention={contention}: schedule diff {diff:.1e}, losses swap {l_s.item():.3f} wait {l_w.item():.3f} "
              f"stage2 {l_c.item():.3f} duration {l_d.item():.3f}, {len(rows)} eval rows: ok")
    print("all checks passed")


if __name__ == "__main__":
    main()
