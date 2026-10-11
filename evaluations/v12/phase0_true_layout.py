#!/usr/bin/env python3
"""v12 phase 0 (go / no-go): can a network on the placed circuit predict how the compiler spreads
the gates over the chip, given the compiler's own layout?

The trained v11dl40 model is frozen; its placement is replaced by the compiler's true layout (best
compilation of each calibration), and only the placed-circuit network (gsv3.model.PlacedCircuitGNN)
is trained, on the usage maps and Δlog F of calibration variants aug 0-79 and the training circuits
of the control split.  Evaluation on the control test circuits:
  unseen calibrations  local variants among aug 80-99   (the number that decides)
  seen calibrations    10 local variants among aug 8-79  (memorisation check)
Reference points on the same cases: the frozen v11 head with the true layout (network off), and
the oracle "true layout + true counts" of report_tracking_oracle_usage.md (0.72 for v11dl40).
Go if the network clearly beats 0.74 on unseen calibrations.
    python phase0_true_layout.py [--steps 4000] [--batch 32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from torch_geometric.loader import DataLoader  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

ROOT = EVAL / "generalization_v9/results"
V8D = EVAL / "compile_check/results/v8d"
ORIG = EVAL / "compile_check/results/v8b_orig"
USAGE = EVAL / "compile_check/results/v8u/usage_labels.pt"
DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LOCAL_KINDS = {0, 1, 2, 3, 4, 5, 6, 9}
KIND = "sinkhorn_rf_route_pgnn_usage"


def dense_layout(graphs, lays):
    qm = max(int(g.n_qubits) for g in graphs)
    out = torch.full((len(graphs), len(DEVS), qm), -1, dtype=torch.long)
    for i, l in enumerate(lays):
        out[i, :, : l.shape[0]] = l.t()
    return out


def run(model, b, devs, lay):
    """log F (B, D) with the compiler's layout as the placement."""
    orig_sink = model._sinkhorn

    def sink(scores, n_q, n_phys):
        a = orig_sink(scores, n_q, n_phys)
        l = lay[:, :, : a.size(2)].to(a.device)
        onehot = F.one_hot(l.clamp(min=0), a.size(3)).float()
        return torch.where((l >= 0).unsqueeze(-1), onehot, a)
    model._sinkhorn = sink
    try:
        return model(b, devs)
    finally:
        model._sinkhorn = orig_sink


def map_loss(model, true_maps):
    u_r, u_m, u_cz, qmask, emask = model.last_usage
    P, E = u_r.size(-1), u_cz.size(-1)
    tr, tm, tc = (t.to(u_r.device) for t in true_maps)
    q = qmask.float().unsqueeze(0); e = emask.float().unsqueeze(0)
    l = ((torch.log1p(u_r) - torch.log1p(tr[..., :P])) ** 2 * q).sum() / q.sum() / u_r.size(0)
    l = l + ((torch.log1p(u_m) - torch.log1p(tm[..., :P])) ** 2 * q).sum() / q.sum() / u_r.size(0)
    l = l + ((torch.log1p(u_cz) - torch.log1p(tc[..., :E])) ** 2 * e).sum() / e.sum() / u_r.size(0)
    return l


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--map-lambda", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--base", default="v11dl40", help="frozen v11 run (control model)")
    ap.add_argument("--tag", default="", help="suffix of the report / network file")
    ap.add_argument("--train-placement", choices=["true", "learned", "mixed"], default="true",
                    help="placement fed to the network in training: the compiler's layout, the model's own, or 50/50")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    params = load_hparams()
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    use = torch.load(USAGE, weights_only=False)
    lab = torch.load(V8D / "v8d_labels.pt", weights_only=False)
    orig = torch.load(ORIG / "v8b_orig_labels.pt", weights_only=False)
    test = set(json.loads((ROOT / "v8kd40/random_control/random_seed5/predictions.json").read_text()))
    pool = [n for n in sorted(use) if n in data and n in lab and n in orig and "F_aug" in use[n]
            and lab[n]["layout"].shape[0] == int(data[n].n_qubits) and orig[n]["layout"].shape[0] == int(data[n].n_qubits)]
    tr_c = [n for n in pool if n not in test]
    te_c = [n for n in pool if n in test]
    js_train = list(range(80))
    js_unseen = [j for j in range(80, 100) if j % 10 in LOCAL_KINDS]
    js_seen = sorted(rng.choice([j for j in range(8, 80) if j % 10 in LOCAL_KINDS], 10, replace=False).tolist())
    raw = json.loads((V8D / "variants_raw.json").read_text())
    raw.update(torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"])
    dev_o = variant_batch(raw, list(DEVS), params["lap_pe"])
    dev_v = {j: variant_batch(raw, [f"{d}/aug{j}" for d in DEVS], params["lap_pe"]).to(dev) for j in range(100)}
    model = build_model(KIND, params, dev_o, dev)
    missing = model.load_state_dict(torch.load(ROOT / args.base / "random_control/random_seed5/model.pth", map_location="cpu"), strict=False)
    assert all(k.startswith("placed_gnn.") for k in missing.missing_keys), missing
    dev_o = dev_o.to(dev)
    for n_, p in model.named_parameters():
        p.requires_grad = n_.startswith("placed_gnn.")
    model.eval()            # frozen parts in eval mode (dropout off); the new network has none
    gnn = model.placed_gnn
    opt = torch.optim.AdamW(gnn.parameters(), lr=args.lr, weight_decay=1e-4)
    log(f"{len(tr_c)} training / {len(te_c)} test circuits with maps; unseen calibrations {js_unseen}, seen {js_seen}; "
        f"{sum(p.numel() for p in gnn.parameters())} parameters in the network; device {dev}")

    def batch_of(names, j):
        gs = [data[n] for n in names]
        b = next(iter(DataLoader(gs, batch_size=len(gs)))).to(dev)
        y_o = torch.stack([orig[n]["F"][0] for n in names]).float().to(dev)
        y_v = torch.stack([lab[n]["F"][j] for n in names]).float().to(dev)
        lay_o = dense_layout(gs, [orig[n]["layout"][:, 0] for n in names])
        lay_v = dense_layout(gs, [lab[n]["layout"][:, j] for n in names])
        m_o = tuple(torch.stack([use[n][k] for n in names]) for k in ("r", "m", "cz"))
        m_v = tuple(torch.stack([use[n][k + "_aug"][j] for n in names]) for k in ("r", "m", "cz"))
        return b, y_o, y_v, lay_o, lay_v, m_o, m_v

    @torch.no_grad()
    def evaluate(js, network_on=True, true_layout=True):
        saved = model.placed_gnn
        if not network_on:
            model.placed_gnn = None
        cells = []
        try:
            for j in js:
                dp, dt = [[] for _ in DEVS], [[] for _ in DEVS]
                for s in range(0, len(te_c), 64):
                    names = te_c[s: s + 64]
                    b, y_o, y_v, lay_o, lay_v, _, _ = batch_of(names, j)
                    if not true_layout:
                        lay_o, lay_v = torch.full_like(lay_o, -1), torch.full_like(lay_v, -1)
                    lo = run(model, b, dev_o, lay_o); lv = run(model, b, dev_v[j], lay_v)
                    keep = (y_v > 0.01) & (y_o > 0.01)
                    d_pred = (torch.clamp(lv, max=0) - torch.clamp(lo, max=0)).cpu()
                    d_true = (torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))).cpu()
                    for d in range(len(DEVS)):
                        k = keep[:, d].cpu()
                        dp[d].append(d_pred[k, d].numpy()); dt[d].append(d_true[k, d].numpy())
                for d in range(len(DEVS)):
                    a, t = np.concatenate(dp[d]), np.concatenate(dt[d])
                    if len(t) > 20:
                        cells.append(corr(a, t))
        finally:
            model.placed_gnn = saved
        return float(np.nanmean(cells))

    L = ["# v12 phase 0: placed-circuit network with the compiler's layout", "",
         f"Frozen {args.base} (control model), compiler's true layout as the placement; network trained on "
         f"{len(tr_c)} circuits × calibrations aug 0-79; evaluated on {len(te_c)} control test circuits.", "",
         "| | unseen calibrations (aug 80-99, local) | seen calibrations (local, in aug 8-79) |", "| --- | --- | --- |"]
    L.append(f"| v11 head, learned placement (the model as it is) | {evaluate(js_unseen, False, False):.3f} | {evaluate(js_seen, False, False):.3f} |")
    log(L[-1])
    base_u, base_s = evaluate(js_unseen, False), evaluate(js_seen, False)
    L.append(f"| v11 head, true layout, network off | {base_u:.3f} | {base_s:.3f} |")
    log(L[-1])
    model.train(False)
    gnn.train()
    hist = []
    for step in range(1, args.steps + 1):
        names = [tr_c[i] for i in rng.choice(len(tr_c), args.batch, replace=False)]
        j = int(rng.choice(js_train))
        b, y_o, y_v, lay_o, lay_v, m_o, m_v = batch_of(names, j)
        if args.train_placement == "learned" or (args.train_placement == "mixed" and rng.random() < 0.5):
            lay_o, lay_v = torch.full_like(lay_o, -1), torch.full_like(lay_v, -1)
        lo = run(model, b, dev_o, lay_o); l_map = map_loss(model, m_o)
        lv = run(model, b, dev_v[j], lay_v); l_map = l_map + map_loss(model, m_v)
        ok = ((y_v > 0.01) & (y_o > 0.01)).float()
        d_pred = torch.clamp(lv, max=0) - torch.clamp(lo, max=0)
        d_true = torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))
        l_d = (ok * (d_pred - d_true) ** 2).sum() / ok.sum().clamp(min=1)
        okv = (y_v > 1e-4).float()
        l_abs = (okv * (torch.clamp(lv, max=0) - torch.log(y_v.clamp(min=1e-6))) ** 2).sum() / okv.sum().clamp(min=1)
        loss = l_d + l_abs + args.map_lambda * l_map
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        opt.step()
        hist.append((l_d.item(), l_abs.item(), l_map.item()))
        if step % 200 == 0:
            h = np.mean(hist[-200:], 0)
            log(f"step {step}: delta {h[0]:.4f}  abs {h[1]:.4f}  maps {h[2]:.4f}")
        if step % args.eval_every == 0 or step == args.steps:
            gnn.eval()
            u, s = evaluate(js_unseen), evaluate(js_seen)
            gnn.train()
            L.append(f"| network, step {step} | {u:.3f} | {s:.3f} |")
            log(L[-1])
            L.append(f"| network on the learned placement, step {step} | {evaluate(js_unseen, True, False):.3f} | "
                     f"{evaluate(js_seen, True, False):.3f} |")
            log(L[-1])
    L += ["", "Reference (report_tracking_oracle_usage.md, 300 pilot test circuits, 10 local calibrations): "
          "v11dl40 learned 0.68, + true layout 0.64, + true layout and counts 0.72; v8kdl40 0.69.", "",
          "Go if the network clearly beats ~0.74 on unseen calibrations; a large seen − unseen gap means it "
          "memorises the 3 chips' calibrations."]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"report_phase0{args.tag}.md").write_text("\n".join(L))
    torch.save(gnn.state_dict(), HERE / "results" / f"phase0_network{args.tag}.pth")
    print("\n".join(L))


if __name__ == "__main__":
    main()
