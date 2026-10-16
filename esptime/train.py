#!/usr/bin/env python3
"""Train and evaluate the model on one split; one run directory per (split, seed).

    python -m esptime.train --split control --seed 5 --out work/runs/control_s5 [--calib-every 4] [--device cuda]
    python -m esptime.train --split lofo:qaoa --eval-only --out work/runs/qaoa_s5
Splits: control, lofo:<family> (lofo:qaoa, lofo:qnn), logo:variational, loso:<synthetic structures>.
Writes model.pt (last epoch, or --select best), checkpoint.pt (resume), history.json, metrics.json (+ timing).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from . import losses as L
from .config import DEVICES, WORK
from .dataset import split_names
from .devices import device_batch
from .model import ESPTimeModel, mc_schedule, n_params

SIZE_BINS = ((0, 20), (20, 50), (50, 200), (200, 600), (600, 10 ** 9))
# Reference (earlier study, 800 circuits, sum over the 3 devices, median ms, one thread): light compile (level 2,
# 1-trial Sabre) and full level-2 compile, by 2q-gate count
COMPILE_MS = {(0, 20): (19.1, 33.5), (20, 50): (36.8, 86.1), (50, 200): (68.9, 531.4), (200, 600): (131.9, 1844.9),
              (600, 10 ** 9): (190.1, 2785.8)}


# Validation terms the best model is selected on: the targets (F, final duration, ESP decay), not the auxiliaries
SELECT = ("f", "final", "decay")


def n2q(it) -> int:
    return int(((it.gate_qubits >= 0).sum(1) >= 2).sum())


def bin_name(lo, hi) -> str:
    return f"{lo}-{hi if hi < 10 ** 9 else ''}"


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict(model, dev, items, bs=32, mc=0, device="cpu", seed=0) -> list[dict]:
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for s0 in range(0, len(items), bs):
        chunk = items[s0: s0 + bs]
        b = next(iter(DataLoader(chunk, batch_size=len(chunk)))).to(device)
        out = {k: v.cpu() for k, v in model(b, dev, mc=mc, gen=gen).items() if k in
               ("logf", "dur", "dur_mc", "logesp", "decay")}
        for i, it in enumerate(chunk):
            for d in range(len(DEVICES)):
                r = {"name": it.name, "d": d, "n2q": n2q(it), "rf": bool(it.rf[0, d]), "rf_pred": bool(it.rf_pred[0, d] > 0.5),
                     "ok": bool(it.ok[0, d]), "dur_lab": float(it.dur[0, d]), "dur": float(out["dur"][i, d]),
                     "logF": float(out["logf"][i, d]), "logESP": float(out["logesp"][i, d]),
                     "logF_lab": float(it.logF[0, d]), "logESP_lab": float(it.logF[0, d] + it.decay[0, d])}
                if mc:
                    r["dur_mc"] = float(out["dur_mc"][i, d])
                rows.append(r)
    return rows


def _err(p, t):
    e = np.abs(p / t - 1)
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "bias": float(np.median(p / t)), "n": int(len(p))}


def _r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12))


def summary(rows) -> dict:
    rows = [r for r in rows if r["ok"] and r["dur_lab"] > 0]
    if not rows:
        return {}
    out = {"n_cases": len(rows), "switch_agreement": float(np.mean([r["rf"] == r["rf_pred"] for r in rows]))}
    for key in ("dur", "dur_mc"):                  # dur: mean SWAP rates (what the losses train); dur_mc: sampled SWAPs
        if key not in rows[0]:
            continue
        dur = {}
        for grp, sel in (("all", rows), ("routed", [r for r in rows if not r["rf"]]), ("routing-free", [r for r in rows if r["rf"]])):
            if sel:
                dur[grp] = _err(np.array([r[key] for r in sel]), np.array([r["dur_lab"] for r in sel]))
        dur["routed_by_2q"] = {bin_name(lo, hi): _err(np.array([r[key] for r in ss]), np.array([r["dur_lab"] for r in ss]))
                               for lo, hi in SIZE_BINS if (ss := [r for r in rows if not r["rf"] and lo <= r["n2q"] < hi])}
        out[f"duration ({key} vs label, relative error)"] = dur
    lf, lft = np.array([r["logF"] for r in rows]), np.array([r["logF_lab"] for r in rows])
    le, let = np.array([r["logESP"] for r in rows]), np.array([r["logESP_lab"] for r in rows])
    big = let > math.log(1e-3)
    out["F"] = {"R2": _r2(np.exp(lf), np.exp(lft)), "abs_dlog_median": float(np.median(np.abs(lf - lft)))}
    rt = np.array([not r["rf"] for r in rows])
    out["ESP"] = {"R2": _r2(np.exp(le), np.exp(let)),
                  "abs_dlog_median_esp_gt_1e-3_routed": float(np.median(np.abs(le - let)[big & rt])) if (big & rt).any() else None,
                  "abs_dlog_median_esp_gt_1e-3_routing_free": float(np.median(np.abs(le - let)[big & ~rt])) if (big & ~rt).any() else None,
                  "abs_dlog_median_esp_gt_1e-3": float(np.median(np.abs(le - let)[big])) if big.any() else None,
                  "abs_dlog_p90_esp_gt_1e-3": float(np.percentile(np.abs(le - let)[big], 90)) if big.any() else None,
                  "R2_log_esp_gt_1e-3": _r2(le[big], let[big]) if big.sum() > 2 else None}
    by = defaultdict(dict)
    for r in rows:
        by[r["name"]][r["d"]] = r
    for what, kp, kt in (("ESP", "logESP", "logESP_lab"), ("F", "logF", "logF_lab")):
        right, regret = [], []
        for dd in by.values():
            if len(dd) == len(DEVICES):
                t = np.exp([dd[d][kt] for d in range(len(DEVICES))])
                k = int(np.argmax([dd[d][kp] for d in range(len(DEVICES))]))
                right.append(t[k] >= 0.99 * t.max()); regret.append(t.max() - t[k])
        if right:
            out[f"device_by_{what}"] = {"n": len(right), "right": float(np.mean(right)), "regret": float(np.mean(regret))}
    return out


@torch.no_grad()
def timing(model, dev, items, n: int, mc: int, seed: int = 0) -> dict:
    """Whole prediction per circuit (batch 1, one thread, CPU): QASM parse + basis + encoding + router features +
    one forward (+ the MC schedule), by 2q-gate bin, next to the compile reference."""
    from .config import CIRCUITS_ZIP, decomp_table
    from .dataset import build_item
    from .devices import chips
    from .labels import circuit_sources, read_qasm
    from .encoding import to_basis
    model.eval()
    th = torch.get_num_threads()
    torch.set_num_threads(1)
    raws = json.loads((WORK / "devices.json").read_text())
    chs, table = chips(raws["orig"]), decomp_table()
    keys = dict(circuit_sources("mqt"))
    z = zipfile.ZipFile(CIRCUITS_ZIP)
    rng = np.random.default_rng(seed)
    sel = [items[i] for i in rng.permutation(len(items)) if items[i].name in keys][:n]
    gen = torch.Generator().manual_seed(seed)
    fake = {"n_nodes": None, "calib": {"orig": {}}}
    res = []
    for it in sel:
        t0 = time.perf_counter()
        qb = to_basis(read_qasm(keys[it.name], z))
        fake["n_nodes"] = len(qb.data)
        li = build_item(it.name, qb, fake, ["orig"], chs, table)
        t1 = time.perf_counter()
        b = next(iter(DataLoader([li], batch_size=1)))
        model(b, dev, mc=mc, gen=gen)
        t2 = time.perf_counter()
        res.append({"n2q": n2q(it), "inputs": t1 - t0, "model": t2 - t1, "total": t2 - t0})
    torch.set_num_threads(th)
    tab = {}
    for lo, hi in SIZE_BINS:
        ss = [r for r in res if lo <= r["n2q"] < hi]
        if ss:
            tab[bin_name(lo, hi)] = {"n": len(ss), **{k + "_ms": float(np.median([r[k] for r in ss]) * 1e3)
                                                      for k in ("inputs", "model", "total")},
                                     "light_compile_ms": COMPILE_MS[(lo, hi)][0], "full_compile_ms": COMPILE_MS[(lo, hi)][1]}
    return {"n": len(res), "mc": mc, "by_2q_ms": tab}


# ------------------------------------------------------------------------------------------------ training
def depth_batches(items, bs: int, rng: random.Random) -> list[list[int]]:
    """Batches of circuits of similar depth (sorted by topological levels with jitter), in random order: the
    schedule runs one step per level, so a deep circuit would otherwise set the cost of every batch it lands in."""
    depth = [int(it.lvl.max()) + 1 if it.num_nodes else 0 for it in items]
    order = sorted(range(len(items)), key=lambda i: depth[i] * rng.uniform(0.8, 1.25))
    batches = [order[i: i + bs] for i in range(0, len(order), bs)]
    rng.shuffle(batches)
    return batches


def run_epoch(model, dev, devs_v, loader, opt=None, sched=None, args=None, rng=None) -> dict:
    train = opt is not None
    model.train(train)
    tot, nb = defaultdict(float), 0
    hops = model.backbone.hops
    with torch.set_grad_enabled(train):
        for bi, b in enumerate(loader):
            b = b.to(args.device)
            out = model(b, dev)
            loss, parts = L.total(out, b, hops, args.lambdas)
            if devs_v and args.calib_every and bi % args.calib_every == 0:
                k = rng.randrange(len(devs_v))
                out_v = model(b, devs_v[k])
                ok_v = b.ok_v[:, k]
                lc = (L.calib_loss(out, out_v, b, k) + L.f_loss(out_v["logf"], b, b.logF_v[:, k], ok_v)
                      + L.decay_loss(out_v["decay"], b.decay_v[:, k], ok_v))
                loss = loss + args.lambdas["calib"] * lc
                parts["calib"] = float(lc)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
            tot["loss"] += float(loss)
            for k, v in parts.items():
                tot[k] += v
            nb += 1
    return {k: v / max(nb, 1) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="control")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="backbone (F network) learning rate")
    ap.add_argument("--lr-route", type=float, default=2e-3, help="route stage and ESP head learning rate")
    ap.add_argument("--esp-grad", choices=("head", "all"), default="head",
                    help="head: the decay loss trains only the ESP head; all: also the route stage")
    ap.add_argument("--calib-every", type=int, default=0, help="variant pass + change loss every k-th batch (0: off)")
    ap.add_argument("--lambda", dest="lam", action="append", default=[], help="loss weight override, e.g. dur=0.5")
    ap.add_argument("--init", type=Path, default=None, help="start from these weights")
    ap.add_argument("--freeze-backbone", action="store_true", help="train only the route stage and ESP head (v29)")
    ap.add_argument("--data", type=Path, default=WORK / "dataset.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--mc", type=int, default=24, help="evaluation: Poisson samples of the SWAPs (0: mean rates)")
    ap.add_argument("--time-n", type=int, default=300, help="circuits timed end to end (0: none)")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--select", choices=("last", "best"), default="last",
                    help="model kept: the last epoch (OneCycle ends at lr ~ 0) or the best validation F + final + decay")
    ap.add_argument("--max-levels", type=int, default=0, help="leave out circuits deeper than this (local runs; 0: all)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    args.lambdas = {**L.LAMBDAS, **{k: float(v) for k, v in (s.split("=") for s in args.lam)}}
    args.out = args.out or WORK / "runs" / f"{args.split.replace(':', '_').replace(',', '-')}_s{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)

    blob = torch.load(args.data, weights_only=False)
    data = {it.name: it for it in blob["items"]
            if not args.max_levels or not it.num_nodes or int(it.lvl.max()) < args.max_levels}
    raws, calibs = blob["devices"], blob["calibs"]
    dev = device_batch(raws["orig"], raws["orig"]).to(args.device)
    devs_v = [device_batch(raws[c], raws["orig"]).to(args.device) for c in calibs if c != "orig"]
    train_n, test_n = split_names(args.split, sorted(data))
    perm = np.random.default_rng(args.seed).permutation(len(train_n))
    n_val = max(1, len(train_n) // 20)
    val_n, train_n = [train_n[i] for i in perm[:n_val]], [train_n[i] for i in perm[n_val:]]
    train_n = train_n[: args.max_train or None]
    test_n = test_n[: args.max_test or None]
    print(f"split {args.split}: train {len(train_n)}, val {len(val_n)}, test {len(test_n)}; "
          f"{len(devs_v)} calibration variants", flush=True)

    model = ESPTimeModel(dev, esp_grad=args.esp_grad).to(args.device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    print(f"model: {n_params(model):,} parameters", flush=True)

    if not args.eval_only:
        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        groups = ([{"params": bb, "lr": args.lr}] if bb else []) + [{"params": rest, "lr": args.lr_route}]
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        steps = args.epochs * math.ceil(len(train_n) / args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps)
        ck = args.out / "checkpoint.pt"
        start, best, hist = 0, float("inf"), []
        if ck.is_file():
            st = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            start, best, hist = st["epoch"] + 1, st["best"], st["hist"]
            print(f"resumed at epoch {start}", flush=True)
        tr_items, va_items = [data[n] for n in train_n], [data[n] for n in val_n]
        for ep in range(start, args.epochs):
            t0 = time.time()
            tr = run_epoch(model, dev, devs_v, DataLoader(tr_items, batch_sampler=depth_batches(tr_items, args.bs, rng)),
                           opt, sched, args, rng)
            va = run_epoch(model, dev, [], DataLoader(va_items, batch_size=args.bs), args=args)
            hist.append({"epoch": ep, "sec": round(time.time() - t0, 1), **{f"train_{k}": v for k, v in tr.items()},
                         **{f"val_{k}": v for k, v in va.items()}})
            print(json.dumps(hist[-1]), flush=True)
            sel = sum(va.get(k, 0.0) for k in SELECT)          # model selection on the target terms only
            hist[-1]["val_select"] = sel
            if args.select == "last" or sel < best:
                best = sel
                torch.save(model.state_dict(), args.out / "model.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "best": best, "hist": hist}, ck)
        (args.out / "history.json").write_text(json.dumps(hist, indent=1))

    model.load_state_dict(torch.load(args.out / "model.pt", map_location="cpu"))
    rows = predict(model, dev, [data[n] for n in test_n], args.bs, args.mc, args.device, args.seed)
    res = {"split": args.split, "seed": args.seed, "n_test_circuits": len(test_n), "args": {k: str(v) for k, v in vars(args).items()},
           **summary(rows)}
    if args.time_n:
        res["timing"] = timing(model.cpu(), dev.cpu(), [data[n] for n in test_n], args.time_n, args.mc, args.seed)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=1))
    torch.save(rows, args.out / "rows.pt")
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
