#!/usr/bin/env python3
"""v29: ONE model for F, duration and ESP (in place of v28, whose light-Sabre input data do not exist yet).

Until v28 the ESP path ran two networks: the F model (v8kdc10 / v8kdfi10, its own learned placement) and the route stage
(v24-v28, durations from the logical qubits, the light Sabre's physical qubits for the decay).  F and the decay came from
two different placements and the F forward was a second model call.  v29 is a single forward:

  backbone   the F network of v8kdfi10 (kind sinkhorn_rf_route_czc_espfi, generalization_v10: same data and splits as
             the route stage), FROZEN by default: F, its placement and its calibration tracking are exactly v8kdfi10's
             (its own ESP head is switched off, ``skip_esp``).  It also gives every gate its embedding and the expected
             chip distance between its operands under the placement.
  route      the v27 route stage (stage1.Stage1: SWAPs per operand wire, waits, stage 2; greedy router features, synth2
             circuits, duration loss) with the backbone's gate embedding and operand distances as extra node inputs.
             Routing-free pairs (the backbone's own VF2 test, no compiler) get no SWAPs and no waits.
  ESP        log ESP = log F − [exp(a) · Σ_q idle_q / min(T1, T2)_q + softplus(b) · duration · mean 1/min(T1, T2)]
             idle_q = duration − busy_q on the routed schedule (gates + the SWAPs on q's wire), min(T1, T2)_q of the
             physical qubit the BACKBONE's hard placement puts q on (the same placement as F); the second term covers
             physical qubits only SWAPs touch.  a, b from a small head (zero-initialised: a = 0, b small), trained on
             the decay labels (esp_labels.pt).  By default the ESP loss trains only that head (``--esp-grad head``);
             ``route`` lets it reach the SWAPs / waits / stage 2 too.

No compiler at prediction time.  Inputs per circuit: the gate graph, the greedy router (prep.prior_route), the device
graphs.  Data: the v27 bundle (stage1.pt, stage1_synth2.pt, graphs, synth graphs, esp_labels.pt) + the v8kdfi10 weights
(generalization_v10/results/v8kdfi10[_s6|_s7]/<split>/model.pth, from the cluster; LOSO splits use the control model).
    python unified.py train --split qaoa --seed 5 --epochs 40 --out results/v29_qaoa_s5 --synth2 --dur-lambda 1 --mc 24 \
        --esp-labels ../compile_check/results/esp_labels.pt --time-n 300
    python unified.py eval --split qaoa --ckpt results/v29_qaoa_s5/model.pt ...
Rows: duration (Sabre's final and the dataset label), F, ESP vs the labels; timing of the whole prediction per circuit
(greedy router features + one forward + MC schedule, batch 1, one thread) by circuit size, next to the light compile.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "esp")]
import stage1 as S  # noqa: E402  (sets the gsv3 import paths)
from gsv3 import paths  # noqa: E402
from gsv3.model import build_model, hard_placement, load_hparams, routing_free  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.utils import scatter, to_dense_batch  # noqa: E402

EVAL = HERE.parent
BACKBONE_KIND = "sinkhorn_rf_route_czc_espfi"
F_ROOT = EVAL / "generalization_v10/results"
F_DIRS = {"qaoa": "leave_one_family_out/qaoa", "variational": "leave_one_group_out/variational",
          "control": "random_control/random_seed{seed}"}
SIZE_BINS = ((0, 20), (20, 50), (50, 200), (200, 600), (600, 10 ** 9))
# light_compile/results/report.md (800 circuits, sum over the 3 devices, median, one thread): light compile alone and
# full level-2 compile per circuit, by 2q-gate count, for the timing table
LIGHT_MS = {(0, 20): (19.1, 33.5), (20, 50): (36.8, 86.1), (50, 200): (68.9, 531.4), (200, 600): (131.9, 1844.9),
            (600, 10 ** 9): (190.1, 2785.8)}


# ----------------------------------------------------------------------------------------------- backbone / devices
def device_batch():
    """The three devices as the F model sees them (variant_batch; the raw JSON copy when device_graphs.pt is absent)."""
    params = load_hparams()
    gpath = paths.V2_DEVICE_GRAPHS
    if gpath.is_file():
        raw = torch.load(gpath, weights_only=False)["raw"]
    else:
        raw = json.loads(S.RAW_DEV.read_text())
        gpath = HERE / "results/device_graphs_from_raw_json.pt"
        gpath.parent.mkdir(parents=True, exist_ok=True)
        if not gpath.is_file():
            torch.save({"raw": raw}, gpath)
    return params, variant_batch(raw, list(S.DEVS), params["lap_pe"], gpath)


def f_checkpoint(split: str, seed: int, run: str = "v8kdfi10", root: Path = F_ROOT) -> Path:
    sd = F_DIRS.get(split, F_DIRS["control"]).format(seed=seed)       # loso: the control model (no synthetic F labels)
    return root / (run if seed == 5 else f"{run}_s{seed}") / sd / "model.pth"


def backbone(params, devices, ckpt: Path | None, device: str, allow_untrained: bool = False):
    fm = build_model(BACKBONE_KIND, params, devices, torch.device(device))
    if ckpt is not None and ckpt.is_file():
        fm.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"F backbone: {ckpt}", flush=True)
    elif allow_untrained:
        print(f"F backbone UNTRAINED ({ckpt} missing): smoke test only", flush=True)
    else:
        raise SystemExit(f"missing F backbone weights {ckpt} (copy v8kdfi10 model.pth from the cluster)")
    fm.skip_esp = True                    # its own ESP head is replaced by the route stage + formula below
    return fm


# ---------------------------------------------------------------------------------------------------------- model
class Unified(nn.Module):
    def __init__(self, fmod, f_in: int, hidden: int = 96, layers: int = 6, train_f: bool = False, esp_grad: str = "head"):
        super().__init__()
        self.fmod, self.train_f, self.esp_grad = fmod, train_f, esp_grad
        if not train_f:
            for p in fmod.parameters():
                p.requires_grad_(False)
        d = fmod.w_q.in_features
        self.emb = nn.Sequential(nn.Linear(d, 32), nn.SiLU())
        self.route = S.Stage1(f_in + 32 + len(S.DEVS), hidden, layers)
        self.esp = nn.Sequential(nn.Linear(2 * hidden + 16 + 5, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.esp[-1].weight)
        with torch.no_grad():
            self.esp[-1].bias.copy_(torch.tensor([0.0, -4.0]))

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_f:
            self.fmod.eval()
        return self

    def forward(self, b, devices, mc: int = 0, gen=None) -> dict:
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_f):
            logf = self.fmod(b, devices)                                             # (B, D)
        fm = self.fmod
        g, dist, assign = fm.last_gate_emb, fm.last_dist, fm.last_assign
        if not self.train_f:
            g, dist, assign = g.detach(), dist.detach(), assign.detach()
        extra = torch.cat([self.emb(g), torch.log1p(dist.clamp(min=0))], 1)
        rate, wait, c2 = self.route(b, extra)
        rf = routing_free(b, devices, fm._rf_cache).to(rate.device) > 0.5          # (B, D) the switch, no compiler
        nr = (~rf)[b.batch]                                                          # (N, D)
        rate_e = rate * nr.unsqueeze(-1)
        wait_e = torch.where(nr, wait, torch.zeros_like(wait))
        sched = S.schedule_torch(b, rate_e, wait_e)                                  # (B, D) µs, mean rates
        dur = sched * torch.exp(c2)
        out = {"logf": logf, "rate": rate, "wait": wait, "c2": c2, "rf": rf, "dur": dur}
        if mc:
            out["dur_mc"] = mc_schedule(b, rate_e.detach(), wait_e.detach(), mc, gen) * torch.exp(c2.detach())
        # ---- ESP on the backbone's placement
        dur_e, rate_b = (dur.detach(), rate_e.detach()) if self.esp_grad == "head" else (dur, rate_e)
        gq = b.gate_qubits
        slot = gq >= 0
        meas = b.x[:, S.GATE2IDX_MEASURE] > 0.5
        elig = (slot.sum(1) >= 2) | meas
        n_c = int(b.num_graphs)
        q_max = assign.size(2)
        sw_t = rate_b * (slot & elig.view(-1, 1)).unsqueeze(1) * S.T_SWAP            # (N, D, 3) SWAP time per wire
        busy = rate.new_zeros(n_c * q_max, rate.size(1))
        for k in range(gq.size(1)):
            m = slot[:, k]
            busy = busy.index_add(0, (b.batch * q_max + gq[:, k].clamp(min=0))[m], (b.t.view(-1, 1) + sw_t[:, :, k])[m])
        busy = busy.view(n_c, q_max, -1).transpose(1, 2)                             # (B, D, Qm)
        active = busy > 0
        idle = torch.where(active, (dur_e.unsqueeze(-1) - busy).clamp(min=0), torch.zeros_like(busy))
        coh, _ = to_dense_batch(devices.coh, devices.batch)                         # (D, P) 1/µs
        inv_t = torch.einsum("bdqp,dp->bdq", hard_placement(assign).detach(), coh.to(busy))
        own = (idle * inv_t).sum(-1)
        n_act = active.sum(-1).clamp(min=1)
        inv_m = (active * inv_t).sum(-1) / n_act
        h = self.route.last_h
        pooled = torch.cat([scatter(h, b.batch, 0, reduce="mean"), scatter(h, b.batch, 0, reduce="max")], 1)
        if self.esp_grad == "head":
            pooled = pooled.detach()
        n_dev = rate.size(1)
        nsw = scatter((rate_b * (slot & elig.view(-1, 1)).unsqueeze(1)).sum(-1), b.batch, 0, dim_size=n_c, reduce="sum")
        z = torch.cat([pooled.unsqueeze(1).expand(-1, n_dev, -1), self.route.dev.weight.unsqueeze(0).expand(n_c, -1, -1),
                       torch.stack([torch.log1p(dur_e), torch.log1p(idle.sum(-1)), torch.log1p(n_act.float()),
                                    torch.log1p(nsw), rf.float()], -1)], -1)
        o = self.esp(z)
        decay = -(own * torch.exp(o[..., 0]) + dur_e * F.softplus(o[..., 1]) * inv_m)
        out.update(decay=decay, logesp=logf + decay)
        return out


def mc_schedule(b, rate_e, wait_e, K: int, gen=None):
    """Median over K schedules of Poisson-sampled SWAPs, all samples in one level-wise pass (schedule_torch over D·K
    columns) instead of stage1.schedule_k's Python loop over the gates: (B, D)."""
    N, D, _ = rate_e.shape
    s = torch.poisson(rate_e.unsqueeze(-1).expand(N, D, 3, K).contiguous(), generator=gen)
    s = s.permute(0, 1, 3, 2).reshape(N, D * K, 3)
    w = wait_e.unsqueeze(-1).expand(N, D, K).reshape(N, D * K)
    return S.schedule_torch(b, s, w).view(-1, D, K).median(-1).values


# ------------------------------------------------------------------------------------------------------------ data
def attach_labels(data: dict, ds: dict | None) -> None:
    """Per item: circuit_name (the backbone's caches), log F and decay labels (esp_labels.pt) with their mask."""
    n = 0
    for name, it in data.items():
        it.circuit_name = name
        lab = ds.get(name) if ds is not None else None
        D = len(S.DEVS)
        lf, dec, ok = torch.zeros(1, D), torch.zeros(1, D), torch.zeros(1, D, dtype=torch.bool)
        if lab is not None:
            f = lab["F"].double().view(-1)
            de = lab["decay"].double().view(-1)
            ok = (torch.isfinite(f) & (f > 0) & torch.isfinite(de) & (de <= 0)).view(1, -1)
            lf = torch.where(ok.view(-1), torch.log(f.clamp(min=1e-300)), torch.zeros_like(f)).float().view(1, -1)
            dec = torch.where(ok.view(-1), de, torch.zeros_like(de)).float().view(1, -1)
            n += int(ok.any())
        it.logf_lab, it.dec_lab, it.esp_ok = lf, dec, ok
    print(f"F / decay labels on {n}/{len(data)} circuits", flush=True)


def esp_loss(b, out):
    """Huber on log(−decay): the decay spans 1e-3 .. 50, its relative error is what moves log ESP in proportion."""
    m = b.esp_ok
    p = torch.log(-out["decay"] + 1e-2)
    t = torch.log(-torch.where(m, b.dec_lab, torch.zeros_like(b.dec_lab)) + 1e-2)
    err = torch.where(m, p - t, torch.zeros_like(p))
    return F.huber_loss(err, torch.zeros_like(err), reduction="none", delta=0.2).sum() / m.sum().clamp(min=1)


# ------------------------------------------------------------------------------------------------------- evaluation
def n2q(it) -> int:
    return int(((it.gate_qubits >= 0).sum(1) >= 2).sum())


@torch.no_grad()
def evaluate(model, devices, items, bs=32, mc=0, ds=None, device="cpu", seed=0):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for s0 in range(0, len(items), bs):
        chunk = items[s0: s0 + bs]
        b = next(iter(DataLoader(chunk, batch_size=len(chunk)))).to(device)
        out = {k: v.cpu() for k, v in model(b, devices, mc=mc, gen=gen).items() if torch.is_tensor(v) and v.dim() == 2}
        for i, it in enumerate(chunk):
            lab_ds = ds.get(it.name) if ds is not None else None
            for d in range(len(S.DEVS)):
                rf_l, rt_l, ok = bool(it.rf[0, d]), bool(it.routed[0, d]), bool(it.esp_ok[0, d])
                if not (rf_l or rt_l or ok):
                    continue
                r = {"name": it.name, "d": d, "n2q": n2q(it), "rf": rf_l, "routed": rt_l, "rf_pred": bool(out["rf"][i, d]),
                     "final": float(it.final[0, d]), "pred": float(out["dur"][i, d]),
                     "logF": float(out["logf"][i, d]), "logESP": float(out["logesp"][i, d]), "decay": float(out["decay"][i, d])}
                if mc:
                    r["pred_mc"] = float(out["dur_mc"][i, d])
                if lab_ds is not None:
                    r["final_ds"] = float(lab_ds["duration"][d]) * 1e6
                if ok:
                    r["logF_lab"] = float(it.logf_lab[0, d]); r["decay_lab"] = float(it.dec_lab[0, d])
                    r["logESP_lab"] = r["logF_lab"] + r["decay_lab"]
                rows.append(r)
    return rows


def _err(p, fin):
    e = np.abs(p / fin - 1)
    return {"err": float(np.median(e)), "p90": float(np.percentile(e, 90)), "bias": float(np.median(p / fin)), "n": int(len(p))}


def _r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12))


def summary(rows):
    out = {}
    mc = rows and "pred_mc" in rows[0]
    dk = ("pred", "pred_mc") if mc else ("pred",)
    rows_rf = [r for r in rows if r["rf"] or r["routed"]]
    out["switch"] = {"agreement": float(np.mean([r["rf"] == r["rf_pred"] for r in rows_rf])) if rows_rf else None}
    for grp, sel in (("routed", [r for r in rows if r["routed"]]), ("routing-free", [r for r in rows if r["rf"]])):
        if not sel:
            continue
        fin = np.array([r["final"] for r in sel])
        res = {"n": len(sel), **{k: _err(np.array([r[k] for r in sel]), fin) for k in dk}}
        ds = [r for r in sel if r.get("final_ds", 0) > 0]
        if ds:
            fd = np.array([r["final_ds"] for r in ds])
            res["vs_dataset"] = {k: _err(np.array([r[k] for r in ds]), fd) for k in dk}
        if grp == "routed":
            res["by_2q"] = {}
            for lo, hi in SIZE_BINS:
                ss = [r for r in ds if lo <= r["n2q"] < hi]
                if ss:
                    res["by_2q"][f"{lo}-{hi if hi < 10 ** 9 else ''}"] = _err(
                        np.array([r[dk[-1]] for r in ss]), np.array([r["final_ds"] for r in ss]))
        out[grp] = res
    for grp, sel in (("all", rows), ("routed", [r for r in rows if r["routed"]]), ("routing-free", [r for r in rows if r["rf"]])):
        ls = [r for r in sel if "logESP_lab" in r and np.isfinite(r["logESP"])]
        if not ls:
            continue
        lf, lft = np.array([r["logF"] for r in ls]), np.array([r["logF_lab"] for r in ls])
        le, let = np.array([r["logESP"] for r in ls]), np.array([r["logESP_lab"] for r in ls])
        big = let > math.log(1e-3)
        e = np.abs(le - let)
        out.setdefault("esp", {})[grp] = {
            "n": len(ls), "R2 F": _r2(np.exp(lf), np.exp(lft)), "abs dlog F median": float(np.median(np.abs(lf - lft))),
            "R2 ESP": _r2(np.exp(le), np.exp(let)),
            "abs dlog ESP median (ESP>1e-3)": float(np.median(e[big])) if big.any() else None,
            "abs dlog ESP p90 (ESP>1e-3)": float(np.percentile(e[big], 90)) if big.any() else None,
            "R2 log ESP (ESP>1e-3)": _r2(le[big], let[big]) if big.sum() > 2 else None,
            "abs dlog ESP median (all)": float(np.median(e)),
            "rel abs dlog ESP median (all)": float(np.median(e / np.maximum(np.abs(let), 1e-9)))}
    # device choice by ESP (circuits with all devices labelled; ties within 1% count as right, as the v24 analysis)
    by = defaultdict(dict)
    for r in rows:
        if "logESP_lab" in r and np.isfinite(r["logESP"]):
            by[r["name"]][r["d"]] = r
    right, regret = [], []
    for name, dd in by.items():
        if len(dd) < len(S.DEVS):
            continue
        t = np.array([math.exp(dd[d]["logESP_lab"]) for d in range(len(S.DEVS))])
        p = np.array([dd[d]["logESP"] for d in range(len(S.DEVS))])
        k = int(np.argmax(p))
        right.append(t[k] >= 0.99 * t.max()); regret.append(t.max() - t[k])
    if right:
        out["device_by_esp"] = {"n": len(right), "right": float(np.mean(right)), "regret": float(np.mean(regret))}
    return out


# ----------------------------------------------------------------------------------------------------------- timing
def live_item(it, chips, tr, tc):
    """The inputs the model reads, rebuilt from the gate graph alone (as at prediction time)."""
    from prep import prior_route
    gq = it.gate_qubits.numpy()
    wires = [[int(x) for x in row if x >= 0] for row in gq]
    t = S.node_t(it, tr, tc)
    meas = it.x[:, S.GATE2IDX_MEASURE].numpy() > 0.5
    n_log = int(it.n_qubits)
    ps, hp = [], []
    for c in chips:
        if n_log > c[0]:
            ps.append(np.zeros((len(gq), 3), np.float32)); hp.append(np.zeros(len(gq), np.float32)); continue
        s_, h_, _ = prior_route(wires, t, meas, n_log, c)
        ps.append(s_); hp.append(h_)
    ps, hp = torch.from_numpy(np.stack(ps)), torch.from_numpy(np.stack(hp))
    dev_f = torch.cat([torch.log1p(hp.clamp(min=0)).t(), torch.log1p(ps.float()).permute(1, 0, 2).reshape(-1, 9)], 1)
    prev, lvl = S.wire_prev(gq)
    return Data(x=it.x, edge_index=it.edge_index, gate_qubits=it.gate_qubits, n_qubits=it.n_qubits,
                wf=torch.from_numpy(S.wire_feats(gq, n_log)), dev_f=dev_f.float(), t=torch.from_numpy(t),
                prev=torch.from_numpy(prev), lvl=torch.from_numpy(lvl), circuit_name=f"__timed__{it.name}",
                num_nodes=it.num_nodes)


@torch.no_grad()
def timing(model, devices, items, n: int, mc: int, seed: int = 0):
    """Whole prediction per circuit, batch 1, one thread, empty caches: features (greedy router on the 3 devices, wire
    features, levels) + one forward (F, placement, VF2 switch, route stage, ESP) + the MC schedule."""
    from esp_router import chips as mk_chips
    model.eval()
    th = torch.get_num_threads()
    torch.set_num_threads(1)
    chips = mk_chips(devices)
    tab = json.loads((EVAL / "esp/results/decomp_table.json").read_text())
    U = list(S.fe.UNIQUE)
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    rng = np.random.default_rng(seed)
    sel = [items[i] for i in rng.permutation(len(items))[:n]]
    gen = torch.Generator().manual_seed(seed)
    rows, dmax = [], 0.0
    for it in sel:
        model.fmod._rf_cache.clear(); model.fmod._re_cache.clear()
        t0 = time.perf_counter()
        li = live_item(it, chips, tr, tc)
        t1 = time.perf_counter()
        b = next(iter(DataLoader([li], batch_size=1)))
        t2 = time.perf_counter()
        out = model(b, devices)
        t3 = time.perf_counter()
        nr = (~out["rf"])[b.batch]
        if mc:
            mc_schedule(b, out["rate"] * nr.unsqueeze(-1), torch.where(nr, out["wait"], torch.zeros_like(out["wait"])), mc, gen)
        t4 = time.perf_counter()
        dmax = max(dmax, float((li.dev_f - it.dev_f[:, :li.dev_f.size(1)]).abs().max()))
        rows.append({"name": it.name, "n2q": n2q(it), "feat": t1 - t0, "collate": t2 - t1, "forward": t3 - t2, "mc": t4 - t3,
                     "total": t4 - t0, "routed_any": bool((~out["rf"]).any())})
    torch.set_num_threads(th)
    res = {"n": len(rows), "mc": mc, "live_vs_stored_dev_f_max_abs_diff": dmax, "by_2q": {}}
    for lo, hi in SIZE_BINS:
        ss = [r for r in rows if lo <= r["n2q"] < hi]
        if ss:
            res["by_2q"][f"{lo}-{hi if hi < 10 ** 9 else ''}"] = {
                "n": len(ss), **{k + "_ms": float(np.median([r[k] for r in ss]) * 1e3) for k in ("feat", "forward", "mc", "total")},
                "light_compile_ms": LIGHT_MS[(lo, hi)][0], "full_compile_ms": LIGHT_MS[(lo, hi)][1]}
    res["all_total_ms_median"] = float(np.median([r["total"] for r in rows]) * 1e3) if rows else None
    return res, rows


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--split", default="qaoa")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--wait-lambda", type=float, default=0.3)
    ap.add_argument("--c2-lambda", type=float, default=1.0)
    ap.add_argument("--dur-lambda", type=float, default=1.0, help="v25 duration loss (as v27)")
    ap.add_argument("--dur-every", type=int, default=1)
    ap.add_argument("--esp-lambda", type=float, default=1.0, help="weight of the decay loss")
    ap.add_argument("--esp-grad", choices=("head", "route"), default="head",
                    help="head: the decay loss trains only the ESP head; route: also the SWAPs / waits / stage 2")
    ap.add_argument("--train-f", action="store_true", help="also fine-tune the F backbone (default: frozen = v8kdfi10's F)")
    ap.add_argument("--c2-target", choices=("sabre", "dataset"), default="sabre")
    ap.add_argument("--synth2", action="store_true", help="v27's low-parallelism synthetic circuits on the training side")
    ap.add_argument("--f-run", default="v8kdfi10")
    ap.add_argument("--f-ckpt", type=Path, default=None, help="F backbone weights (default: f_checkpoint(split, seed))")
    ap.add_argument("--allow-untrained-f", action="store_true", help="smoke tests without the backbone weights")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "results/v29")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=HERE / "results/stage1.pt")
    ap.add_argument("--mc", type=int, default=24)
    ap.add_argument("--esp-labels", type=Path, default=EVAL / "compile_check/results/esp_labels.pt")
    ap.add_argument("--time-n", type=int, default=300, help="circuits timed end to end (batch 1, one thread); 0: none")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-train", type=int, default=0, help="smoke tests: first N training circuits")
    ap.add_argument("--max-test", type=int, default=0, help="smoke tests: first N test circuits")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = torch.load(args.esp_labels, weights_only=False) if args.esp_labels and args.esp_labels.is_file() else None
    if ds is None:
        print(f"no esp labels at {args.esp_labels}: the ESP loss and scores are off", flush=True)
    data = S.load_all(args.data, synth2=args.synth2)
    if args.c2_target == "dataset":
        S.retarget_stage2(data, ds)
    attach_labels(data, ds)
    train_n, test_n = S.split_names(args.split, sorted(data))
    if args.max_test:
        test_n = test_n[:args.max_test]
    rng = np.random.default_rng(args.seed)
    train_n = list(rng.permutation(train_n))
    n_val = max(1, len(train_n) // 20)
    val_n, train_n = train_n[:n_val], train_n[n_val:]
    if args.max_train:
        train_n = train_n[:args.max_train]
    print(f"split {args.split}: train {len(train_n)}, val {len(val_n)}, test {len(test_n)}", flush=True)
    params, devices = device_batch()
    devices = devices.to(args.device)
    fm = backbone(params, devices, args.f_ckpt or f_checkpoint(args.split, args.seed, args.f_run), args.device,
                  args.allow_untrained_f)
    f_in = data[train_n[0]].x.size(1) + 9 + 12
    model = Unified(fm, f_in, args.hidden, args.layers, args.train_f, args.esp_grad).to(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "train":
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * math.ceil(len(train_n) / args.bs))
        tr_items = [data[n] for n in train_n]
        best, hist = float("inf"), []
        for ep in range(args.epochs):
            model.train()
            t0 = time.time(); tot = defaultdict(float); nb = 0
            for bi, b in enumerate(DataLoader(tr_items, batch_size=args.bs, shuffle=True)):
                b = b.to(args.device)
                out = model(b, devices)
                l_s, l_w, l_c = S.losses(b, out["rate"], out["wait"], out["c2"])
                loss = l_s + args.wait_lambda * l_w + args.c2_lambda * l_c
                if args.dur_lambda and bi % args.dur_every == 0:
                    l_d = S.duration_loss(b, out["rate"], out["wait"])
                    loss = loss + args.dur_lambda * l_d
                    tot["d"] += l_d.item(); tot["nd"] += 1
                if args.esp_lambda and ds is not None:
                    l_e = esp_loss(b, out)
                    loss = loss + args.esp_lambda * l_e
                    tot["e"] += l_e.item()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step()
                tot["s"] += l_s.item(); tot["w"] += l_w.item(); tot["c"] += l_c.item(); nb += 1
            model.eval()
            with torch.no_grad():
                vl = []
                for b in DataLoader([data[n] for n in val_n], batch_size=args.bs):
                    b = b.to(args.device)
                    out = model(b, devices)
                    l_s, l_w, l_c = S.losses(b, out["rate"], out["wait"], out["c2"])
                    v_ = l_s + args.wait_lambda * l_w + args.c2_lambda * l_c
                    if args.dur_lambda:
                        v_ = v_ + args.dur_lambda * S.duration_loss(b, out["rate"], out["wait"])
                    if args.esp_lambda and ds is not None:
                        v_ = v_ + args.esp_lambda * esp_loss(b, out)
                    vl.append(v_.item())
            v = float(np.mean(vl))
            hist.append({"epoch": ep, "train_s": tot["s"] / nb, "train_w": tot["w"] / nb, "train_c": tot["c"] / nb,
                         "train_d": tot["d"] / max(tot["nd"], 1), "train_e": tot["e"] / nb, "val": v, "sec": time.time() - t0})
            print(json.dumps(hist[-1]), flush=True)
            if v < best:
                best = v
                torch.save({k: t for k, t in model.state_dict().items() if args.train_f or not k.startswith("fmod.")},
                           args.out / "model.pt")
        (args.out / "history.json").write_text(json.dumps(hist))
    ckpt = args.ckpt or args.out / "model.pt"
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    bad = [k for k in missing if not k.startswith("fmod.")] + list(unexpected)
    assert not bad, f"checkpoint mismatch: {bad[:5]}"
    rows = evaluate(model, devices, [data[n] for n in test_n], bs=args.bs, mc=args.mc, ds=ds, device=args.device, seed=args.seed)
    res = {"split": args.split, "seed": args.seed, "n_test_circuits": len(test_n), "ckpt": str(ckpt), "mc": args.mc,
           "f_backbone": str(args.f_ckpt or f_checkpoint(args.split, args.seed, args.f_run)), **summary(rows)}
    if args.time_n:
        tres, trows = timing(model.cpu(), devices.cpu(), [data[n] for n in test_n], args.time_n, args.mc, args.seed)
        res["timing"] = tres
        torch.save(trows, args.out / f"timing_rows_{S_stem(args.split)}.pt")
    stem = S_stem(args.split)
    (args.out / f"eval_{stem}.json").write_text(json.dumps(res, indent=1))
    torch.save(rows, args.out / f"rows_{stem}.pt")
    print(json.dumps(res, indent=1))


def S_stem(split: str) -> str:
    return split.replace(':', '_').replace(',', '-')


if __name__ == "__main__":
    main()
