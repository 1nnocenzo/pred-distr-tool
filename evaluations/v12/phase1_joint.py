#!/usr/bin/env python3
"""v12 phase 1: train the placement and the placed-circuit network together.

Phase 0 showed the network lifts local tracking 0.714 -> 0.857 with the compiler's layout but gives
nothing on the model's own placement (0.57-0.72 vs 0.737).  Here the whole v11dl40 model is
fine-tuned with the network (started from the phase-0 network):
  * the placement is supervised on the compiler's layout of every calibration (original + variant):
    ``ce``  = v6 cross-entropy -log A[q, layout(q)];  ``region`` = BCE of the occupancy Σ_q A[q, p]
    vs the compiler's used qubits (the v7 region term);  ``both`` = their sum;
  * scheduled sampling: the network is fed the compiler's layout with probability p_true, annealed
    1 -> 0 over the first ``--anneal`` fraction of the steps, then only the learned placement;
  * losses as phase 0 (Δlog F on variants, |log F|, usage maps) + λ_place · placement loss.
Evaluation on the control test circuits, unseen calibrations aug 80-99 (local) and seen ones:
local tracking on the learned placement (the number that decides, vs 0.737 for v11dl40), on the
compiler's layout, placement agreement (argmax = compiler's qubit), R² of F on the original calibration.
Data parallel over the working GPUs (one process each, gradients all-reduced): every rank draws the same
circuits / calibration / teacher choice and takes its share of the batch, so the recipe is the same as on one GPU.
    python phase1_joint.py [--steps 20000] [--place-loss ce] [--tag _ce]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from phase0_true_layout import (DEVS, KIND, LOCAL_KINDS, ORIG, ROOT, USAGE, V8D, corr, dense_layout,
                                map_loss, paths)
from torch_geometric.loader import DataLoader
from gsv3.devices import variant_batch
from gsv3.model import build_model, load_hparams

HERE = Path(__file__).resolve().parent


def run(model, b, devs, lay, teacher: bool):
    """log F (B, D); the learned placement A is kept in ``model.v12_learned`` for the placement loss.
    ``teacher``: the head and network get the compiler's layout instead of A."""
    orig_sink = model._sinkhorn

    def sink(scores, n_q, n_phys):
        a = orig_sink(scores, n_q, n_phys)
        model.v12_learned = a
        if not teacher:
            return a
        l = lay[:, :, : a.size(2)].to(a.device)
        onehot = F.one_hot(l.clamp(min=0), a.size(3)).float()
        return torch.where((l >= 0).unsqueeze(-1), onehot, a)
    model._sinkhorn = sink
    try:
        return model(b, devs)
    finally:
        model._sinkhorn = orig_sink


def place_loss(a, lay, kind):
    """a (B, D, Qm, P) learned placement, lay (B, D, Qm') compiler's layout (-1 = no qubit)."""
    lay = lay[:, :, : a.size(2)].to(a.device)
    known = lay >= 0
    out = a.new_zeros(())
    if kind in ("ce", "both"):
        picked = a.gather(-1, lay.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        out = out + (-torch.log(picked.clamp_min(1e-9)) * known).sum() / known.sum().clamp(min=1)
    if kind in ("region", "both"):
        p = a.size(3)
        used = torch.zeros(*a.shape[:2], p + 1, device=a.device)
        used.scatter_(2, torch.where(known, lay, torch.full_like(lay, p)), 1.0)
        used = used[..., :p]
        occ = a.sum(2).clamp(1e-6, 1 - 1e-6)
        valid = (a.sum(2) > 0) | (used > 0)          # columns of the device (padding columns get A = 0)
        bce = -(used * occ.log() + (1 - used) * (1 - occ).log())
        out = out + (bce * valid).sum() / valid.sum().clamp(min=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="model (placement, encoders, head)")
    ap.add_argument("--lr-net", type=float, default=3e-4, help="placed-circuit network")
    ap.add_argument("--map-lambda", type=float, default=0.1)
    ap.add_argument("--place-lambda", type=float, default=0.1)
    ap.add_argument("--place-loss", choices=["ce", "region", "both"], default="ce")
    ap.add_argument("--anneal", type=float, default=0.6, help="fraction of the steps over which p_true goes 1 -> 0")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--base", default="v11dl40", help="v11 run (control model) to fine-tune")
    ap.add_argument("--init-net", default="phase0_network_30k.pth", help="phase-0 network in results/ ('' = zero-init)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    world = max(1, torch.cuda.device_count())
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_JOB_ID", "0")) % 2000))
        mp.spawn(worker, args=(world, args), nprocs=world)
    else:
        worker(0, 1, args)


def worker(rank, world, args):
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)          # same stream on every rank: same circuits, calibration, teacher
    t0 = time.time()
    log = (lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)) if rank == 0 else (lambda m: None)
    log(f"args {vars(args)}")
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
    if args.init_net:
        model.placed_gnn.load_state_dict(torch.load(HERE / "results" / args.init_net, map_location="cpu"))
    dev_o = dev_o.to(dev)
    model.eval()            # dropout is 0 and there is no batch norm: eval mode only fixes the behaviour
    net_p = [p for n_, p in model.named_parameters() if n_.startswith("placed_gnn.")]
    rest_p = [p for n_, p in model.named_parameters() if not n_.startswith("placed_gnn.")]
    opt = torch.optim.AdamW([{"params": rest_p, "lr": args.lr}, {"params": net_p, "lr": args.lr_net}], weight_decay=1e-4)
    log(f"{len(tr_c)} training / {len(te_c)} test circuits; unseen {js_unseen}, seen {js_seen}; "
        f"{sum(p.numel() for p in net_p)} network + {sum(p.numel() for p in rest_p)} model parameters; {world} GPU process(es)")
    all_p = net_p + rest_p

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
    def evaluate(js, teacher):
        """(local tracking, placement agreement on the variants, R² of F on the original calibration)."""
        cells, agree, n_q = [], 0.0, 0.0
        fo_p, fo_t = [], []
        for j in js:
            dp, dt = [[] for _ in DEVS], [[] for _ in DEVS]
            for s in range(0, len(te_c), 64):
                names = te_c[s: s + 64]
                b, y_o, y_v, lay_o, lay_v, _, _ = batch_of(names, j)
                lo = run(model, b, dev_o, lay_o, teacher)
                lv = run(model, b, dev_v[j], lay_v, teacher)
                a = model.v12_learned
                l = lay_v[:, :, : a.size(2)].to(dev)
                k = l >= 0
                agree += float(((a.argmax(-1) == l) & k).sum()); n_q += float(k.sum())
                if j == js[0]:
                    fo_p.append(torch.exp(torch.clamp(lo, max=0)).cpu()); fo_t.append(y_o.cpu())
                keep = (y_v > 0.01) & (y_o > 0.01)
                d_pred = (torch.clamp(lv, max=0) - torch.clamp(lo, max=0)).cpu()
                d_true = (torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))).cpu()
                for d in range(len(DEVS)):
                    kd = keep[:, d].cpu()
                    dp[d].append(d_pred[kd, d].numpy()); dt[d].append(d_true[kd, d].numpy())
            for d in range(len(DEVS)):
                x, t = np.concatenate(dp[d]), np.concatenate(dt[d])
                if len(t) > 20:
                    cells.append(corr(x, t))
        p, t = torch.cat(fo_p).numpy().ravel(), torch.cat(fo_t).numpy().ravel()
        r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        return float(np.nanmean(cells)), agree / max(n_q, 1), float(r2)

    def report_rows(label):
        rows = []
        for teacher, what in ((False, "learned placement"), (True, "compiler's layout")):
            u, s = evaluate(js_unseen, teacher), evaluate(js_seen, teacher)
            rows.append(f"| {label}, {what} | {u[0]:.3f} | {s[0]:.3f} | {u[1]:.3f} | {u[2]:.4f} |")
            log(rows[-1])
        return rows

    L = ["# v12 phase 1: placement and placed-circuit network trained together", "",
         f"Fine-tuned {args.base} (control model) + network (init: {args.init_net or 'zero'}); placement loss "
         f"{args.place_loss} × {args.place_lambda} on the compiler's layout of each calibration; scheduled sampling "
         f"p_true 1 → 0 over the first {args.anneal:.0%} of {args.steps} steps (batch {args.batch} over {world} GPU(s)); {len(tr_c)} training circuits × "
         f"calibrations aug 0-79; {len(te_c)} control test circuits.", "",
         "| | tracking, unseen calibrations (aug 80-99, local) | tracking, seen calibrations | placement agreement (unseen) | R² of F, original calibration |",
         "| --- | --- | --- | --- | --- |"]
    def eval_rows(label):
        rows = report_rows(label) if rank == 0 else []
        if world > 1:
            dist.barrier()
        return rows

    L += eval_rows("start (v11dl40 + phase-0 network)")
    hist = []
    for step in range(1, args.steps + 1):
        p_true = max(0.0, 1.0 - step / max(1.0, args.anneal * args.steps))
        teacher = bool(rng.random() < p_true)
        names = [tr_c[i] for i in rng.choice(len(tr_c), args.batch, replace=False)][rank::world]
        j = int(rng.choice(js_train))
        b, y_o, y_v, lay_o, lay_v, m_o, m_v = batch_of(names, j)
        lo = run(model, b, dev_o, lay_o, teacher)
        l_map, l_pl = map_loss(model, m_o), place_loss(model.v12_learned, lay_o, args.place_loss)
        lv = run(model, b, dev_v[j], lay_v, teacher)
        l_map = l_map + map_loss(model, m_v)
        l_pl = l_pl + place_loss(model.v12_learned, lay_v, args.place_loss)
        ok = ((y_v > 0.01) & (y_o > 0.01)).float()
        d_pred = torch.clamp(lv, max=0) - torch.clamp(lo, max=0)
        d_true = torch.log(y_v.clamp(min=1e-6)) - torch.log(y_o.clamp(min=1e-6))
        l_d = (ok * (d_pred - d_true) ** 2).sum() / ok.sum().clamp(min=1)
        oko, okv = (y_o > 1e-4).float(), (y_v > 1e-4).float()
        l_abs = ((oko * (torch.clamp(lo, max=0) - torch.log(y_o.clamp(min=1e-6))) ** 2).sum() / oko.sum().clamp(min=1)
                 + (okv * (torch.clamp(lv, max=0) - torch.log(y_v.clamp(min=1e-6))) ** 2).sum() / okv.sum().clamp(min=1))
        loss = l_d + l_abs + args.map_lambda * l_map + args.place_lambda * l_pl
        opt.zero_grad(); loss.backward()
        if world > 1:                   # mean gradient over the ranks (each saw its share of the batch)
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in all_p]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for p_, g in zip(all_p, grads):
                p_.grad = flat[o: o + g.numel()].view_as(g); o += g.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if rank == 0:
            hist.append((l_d.item(), l_abs.item(), l_map.item(), l_pl.item()))
        if step % 200 == 0 and rank == 0:
            h = np.mean(hist[-200:], 0)
            log(f"step {step}: p_true {p_true:.2f}  delta {h[0]:.4f}  abs {h[1]:.4f}  maps {h[2]:.4f}  place {h[3]:.4f}")
        if step % args.eval_every == 0 or step == args.steps:
            L += eval_rows(f"step {step}")
    if rank != 0:
        dist.destroy_process_group()
        return
    L += ["", "Reference (phase 0, same circuits and calibrations): v11dl40 learned placement 0.737 unseen / 0.692 seen; "
          "frozen v11dl40 + network on the compiler's layout 0.857 / 0.839.", "",
          "Go to the standard evaluation if the learned-placement row clearly beats 0.737 on unseen calibrations "
          "with R² of F unchanged."]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"report_phase1{args.tag}.md").write_text("\n".join(L))
    torch.save(model.state_dict(), HERE / "results" / f"phase1_model{args.tag}.pth")
    print("\n".join(L))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
