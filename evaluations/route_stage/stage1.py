#!/usr/bin/env python3
"""Stage 1 (v24): a graph network that learns Sabre's routing on the model's gate graph.

Per gate and device it predicts the SWAPs Sabre puts on each operand's wire just before the gate (Poisson rate) and
the extra wait of the gate (synchronisation with the SWAP partners), both supervised by compile_check/sabre_labels.py
through prep.py.  The routed duration is then the ASAP schedule of the gates with those SWAPs and waits (``schedule``);
routing-free cases keep the plain schedule (exact up to the decomposition table).  No compiler at prediction time:
the only routing input is our own greedy router (``prior_s``, ``hops``).
Stage 2 (same network, pooled head): log(final duration / routed schedule) per circuit and device, i.e. what
translation + optimisation do to the routed schedule (target on Sabre's replayed schedule; routing-free: on the
logical schedule).  Prediction = schedule(stage-1 SWAPs, waits) × exp(stage 2).

Splits (``--split``):
  loso:<structs>   development: test = synthetic circuits of those structures; qaoa and the variational group
                   (vqe_*, qnn) are left out of training entirely, so the final splits stay unseen
  qaoa | variational | control   the standard splits' test circuits (generalization_v10, seed-5 lists); for qaoa /
                   variational every circuit of the held-out families is out of training
    python stage1.py train --split loso:dense,complete --epochs 30 --out results/s1_loso
    python stage1.py eval --split qaoa --ckpt results/s1_qaoa/model.pt
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
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3 import fast_encoding as fe  # noqa: E402
from gsv3.model import GATE2IDX_MEASURE  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.utils import scatter  # noqa: E402

DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
T1, TM, TCZ = 0.042, 15.0, 0.130
T_SWAP = 3 * TCZ + 2 * T1
VAR_FAMS = ("vqe_su2", "vqe_real_amp", "vqe_two_local", "qnn")
SPLIT_LISTS = {"qaoa": "leave_one_family_out/qaoa", "variational": "leave_one_group_out/variational",
               "control": "random_control/random_seed5"}


def wire_feats(gq: np.ndarray, n_q: int) -> np.ndarray:
    """(N, 9): per operand slot, log1p(multi-qubit gates on its wire), log1p(distinct partners), position on the wire."""
    N = len(gq)
    multi = (gq >= 0).sum(1) >= 2
    cnt = np.zeros(n_q); partners = [set() for _ in range(n_q)]
    for i in np.nonzero(multi)[0]:
        qs = [q for q in gq[i] if q >= 0]
        for q in qs:
            cnt[q] += 1; partners[q].update(x for x in qs if x != q)
    seen = np.zeros(n_q)
    out = np.zeros((N, 9), np.float32)
    for i in range(N):
        for k in range(3):
            q = gq[i, k]
            if q < 0:
                continue
            out[i, 3 * k] = np.log1p(cnt[q]); out[i, 3 * k + 1] = np.log1p(len(partners[q]))
            out[i, 3 * k + 2] = seen[q] / max(cnt[q], 1)
        if multi[i]:
            for q in gq[i]:
                if q >= 0:
                    seen[q] += 1
    return out


RAW_DEV = EVAL / "generalization_v2/results/device_graphs.raw.json"


def raw_chips():
    """(n, adj, H, next-hop parents) per device from the tracked raw device graphs (same node order as the model's)."""
    from collections import deque
    raw = json.loads(RAW_DEV.read_text())
    out = []
    for d in DEVS:
        n = len(raw[d]["nodes"])
        edges = sorted({(min(a, b), max(a, b)) for a, b in raw[d]["edges"]})
        adj = [[] for _ in range(n)]
        for a, c in edges:
            adj[a].append(c); adj[c].append(a)
        H, nxt = [], []
        for s0 in range(n):
            dist = [n + 1] * n; dist[s0] = 0; par = [-1] * n; q = deque([s0])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] > dist[u] + 1:
                        dist[v] = dist[u] + 1; par[v] = u; q.append(v)
            H.append(dist); nxt.append(par)
        out.append((n, adj, H, nxt))
    return out


def router_contention(gq: np.ndarray, t: np.ndarray, meas: np.ndarray, n_log: int, chip) -> np.ndarray:
    """v26: per gate, the wait our own greedy router (prep.prior_route, no compiler) pays for contention: its gate start
    on the chip minus the start the wire model would give (previous end on each operand wire + that wire's SWAPs).
    Non-zero when a SWAP had to wait for, or blocked, the logical qubit sitting on its path (SWAP partners)."""
    sys.path.insert(0, str(EVAL / "esp"))
    from esp_router import greedy_place, path
    n, adj, H, par = chip
    wires = [[int(x) for x in row if x >= 0] for row in gq]
    pairs: dict = {}
    for qs in wires:
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                k = (min(qs[i], qs[j]), max(qs[i], qs[j])); pairs[k] = pairs.get(k, 0) + 1
    if n_log > n:
        return np.zeros(len(wires), np.float32)
    pos = greedy_place(pairs, n_log, H, adj)
    at = {p: q for q, p in pos.items()}
    avail = [0.0] * n
    last = defaultdict(float)                      # end of the last gate of each logical wire (router schedule)
    pending = defaultdict(int)
    out = np.zeros(len(wires), np.float32)
    for gi, qs in enumerate(wires):
        if not qs:
            continue
        if len(qs) >= 2:
            anchor = pos[qs[0]]
            for o in qs[1:]:
                po = pos[o]
                if H[po][anchor] <= 1:
                    continue
                pth = path(par, po, anchor)
                for u, v in zip(pth[:-2], pth[1:-1]):
                    st = max(avail[u], avail[v]) + T_SWAP
                    avail[u] = avail[v] = st
                    qu, qv = at.get(u), at.get(v)
                    for w in (qu, qv):
                        if w is not None:
                            pending[w] += 1
                    at[u], at[v] = qv, qu
                    if qu is not None:
                        pos[qu] = v
                    if qv is not None:
                        pos[qv] = u
                anchor = pos[qs[0]]
        own = {}
        if len(qs) >= 2 or meas[gi]:
            for w in qs:
                own[w] = pending[w]; pending[w] = 0
        ready = max(last[w] + own.get(w, 0) * T_SWAP for w in qs)
        ps = [pos[q] for q in qs]
        st = max(avail[p] for p in ps)
        out[gi] = st - ready
        e = st + t[gi]
        for p in ps:
            avail[p] = e
        for w in qs:
            last[w] = e
    return out


def wire_prev(gq: np.ndarray):
    """(N, 3) index of the previous gate on each operand's wire (-1: none) and (N,) topological level (longest chain
    of wire predecessors): the gates of one level can be scheduled together (v25 differentiable schedule)."""
    N = len(gq)
    prev = np.full((N, 3), -1, np.int64)
    lvl = np.zeros(N, np.int64)
    last = {}
    for i in range(N):
        best = -1
        for k in range(gq.shape[1]):
            q = int(gq[i, k])
            if q < 0:
                continue
            p = last.get(q, -1)
            prev[i, k] = p
            if p >= 0:
                best = max(best, int(lvl[p]))
            last[q] = i
        lvl[i] = best + 1
    return prev, lvl


def node_t(g, tr, tc):
    U = len(fe.UNIQUE)
    oh = g.x[:, :U].numpy()
    meas = g.x[:, GATE2IDX_MEASURE].numpy() > 0.5
    gq = g.gate_qubits.numpy()
    n_op = np.clip((gq >= 0).sum(1), 1, None)
    return np.where(meas, TM, (oh @ tr) * T1 / n_op + (oh @ tc) * TCZ).astype(np.float32)


def build(graphs, s1, tr, tc, contention=None):
    """PyG items with the stage-1 features and targets (routing-free devices are masked out of the loss)."""
    items = []
    for g in graphs:
        lab = s1.get(g.circuit_name)
        if lab is None:
            continue
        gq = g.gate_qubits.numpy()
        n_q = int(g.n_qubits)
        dev_f = torch.cat([torch.log1p(lab["hops"].clamp(min=0)).t(), torch.log1p(lab["prior_s"]).permute(1, 0, 2).reshape(-1, 9)], 1)
        routed = (~lab["rf"]) & torch.isfinite(lab["replay"]) & lab["aligned"]
        t_np = node_t(g, tr, tc)
        meas = g.x[:, GATE2IDX_MEASURE].numpy() > 0.5
        logical = schedule_np(gq, t_np, np.zeros((len(gq), 3)), np.zeros(len(gq)), np.zeros(len(gq), bool))
        base = torch.where(routed, lab["replay"], torch.full_like(lab["replay"], logical))
        ok2 = (routed | lab["rf"]) & (lab["final"] > 0)
        prev, lvl = wire_prev(gq)
        extra = {}
        if contention is not None:
            pw = contention.get(g.circuit_name)
            extra["pw"] = (pw if pw is not None else torch.zeros(len(gq), len(DEVS))).float()
            dev_f = torch.cat([dev_f.float(), torch.log1p(extra["pw"].clamp(min=0) / T_SWAP)], 1)
        items.append(Data(prev=torch.from_numpy(prev), lvl=torch.from_numpy(lvl), **extra, x=g.x.float(), edge_index=g.edge_index, gate_qubits=g.gate_qubits, n_qubits=g.n_qubits,
                          wf=torch.from_numpy(wire_feats(gq, n_q)), dev_f=dev_f.float(), t=torch.from_numpy(t_np),
                          c2=torch.where(ok2, torch.log(lab["final"].clamp(min=1e-3) / base.clamp(min=1e-3)), torch.zeros_like(base)).view(1, -1).float(),
                          ok2=ok2.view(1, -1), logical=torch.tensor([[logical]], dtype=torch.float),
                          s=lab["s"].permute(1, 0, 2).float(), wait=lab["wait"].t().float(), m=lab["m"].t(),
                          routed=routed.view(1, -1), rf=lab["rf"].view(1, -1), final=lab["final"].view(1, -1),
                          replay=lab["replay"].view(1, -1), prior_dur=lab["prior_dur"].view(1, -1),
                          name=g.circuit_name, num_nodes=g.num_nodes))
    return items


class Stage1(nn.Module):
    def __init__(self, f_in: int, hidden: int = 96, layers: int = 6, n_dev: int = 3, contention: bool = False):
        super().__init__()
        self.contention = contention                  # v26: wait = router's contention wait + learned correction
        self.inp = nn.Sequential(nn.Linear(f_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                                     for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dev = nn.Embedding(n_dev, 16)
        self.head = nn.Sequential(nn.Linear(hidden + 16 + 4 + int(contention), hidden), nn.SiLU(), nn.Linear(hidden, 4))
        self.head2 = nn.Sequential(nn.Linear(2 * hidden + 16, hidden), nn.SiLU(), nn.Linear(hidden, 1))   # stage 2
        self.n_dev = n_dev

    def forward(self, b):
        x = torch.cat([b.x, b.wf, b.dev_f], 1)
        h = self.inp(x)
        src, dst = b.edge_index
        for lay, nrm in zip(self.layers, self.norms):
            fwd = scatter(h[src], dst, 0, dim_size=h.size(0), reduce="mean")          # from predecessors
            bwd = scatter(h[dst], src, 0, dim_size=h.size(0), reduce="mean")          # from successors
            glob = scatter(h, b.batch, 0, reduce="mean")[b.batch]
            h = nrm(h + lay(torch.cat([h, fwd, bwd, glob], 1)))
        N = h.size(0)
        outs = []
        for d in range(self.n_dev):
            df = torch.cat([b.dev_f[:, d: d + 1], b.dev_f[:, 3 + 3 * d: 6 + 3 * d]]
                           + ([b.dev_f[:, 12 + d: 13 + d]] if self.contention else []), 1)
            z = torch.cat([h, self.dev.weight[d].expand(N, -1), df], 1)
            outs.append(self.head(z))
        o = torch.stack(outs, 1)                                                     # (N, D, 4)
        pooled = torch.cat([scatter(h, b.batch, 0, reduce="mean"), scatter(h, b.batch, 0, reduce="max")], 1)
        c2 = torch.cat([self.head2(torch.cat([pooled, self.dev.weight[d].expand(pooled.size(0), -1)], 1))
                        for d in range(self.n_dev)], 1)                              # (B, D) log(final / base)
        wait = o[..., 3] + (b.pw / T_SWAP if self.contention else 0.0)
        return F.softplus(o[..., :3]), wait, c2                                       # SWAP rates, wait / T_SWAP, stage 2


def losses(b, rate, wait, c2):
    gq = b.gate_qubits
    slot_ok = (gq >= 0)                                                                # (N, 3)
    n_op = slot_ok.sum(1)
    meas = b.x[:, GATE2IDX_MEASURE] > 0.5
    elig = ((n_op >= 2) | meas).view(-1, 1)                                            # gates that carry SWAP labels
    routed = b.routed[b.batch]                                                         # (N, D)
    ms = (routed & elig).unsqueeze(-1) & slot_ok.unsqueeze(1)                         # (N, D, 3)
    l_s = (F.poisson_nll_loss(rate, b.s, log_input=False, reduction="none") * ms).sum() / ms.sum().clamp(min=1)
    mw = routed & b.m
    l_w = (F.huber_loss(wait, b.wait / T_SWAP, reduction="none", delta=1.0) * mw).sum() / mw.sum().clamp(min=1)
    l_c = (F.huber_loss(c2, b.c2, reduction="none", delta=0.2) * b.ok2).sum() / b.ok2.sum().clamp(min=1)
    return l_s, l_w, l_c


def schedule_torch(b, rate, wait):
    """schedule_np in torch, all circuits of the batch and all devices at once, level by level (differentiable in the
    SWAP rates and the waits): (B, D) durations of the mean-rate schedule."""
    gq = b.gate_qubits
    N, D = rate.shape[0], rate.shape[1]
    off = b.ptr[:-1][b.batch].unsqueeze(1)                                            # local -> batch gate index
    has = b.prev >= 0
    gprev = torch.where(has, b.prev + off, torch.zeros_like(b.prev))
    slot = gq >= 0
    meas = b.x[:, GATE2IDX_MEASURE] > 0.5
    elig = (slot.sum(1) >= 2) | meas
    end = rate.new_zeros(N, D)
    neg = torch.finfo(rate.dtype).min / 4
    for L in range(int(b.lvl.max()) + 1):
        idx = (b.lvl == L).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        last = torch.where(has[idx].unsqueeze(1), end[gprev[idx]].transpose(1, 2), rate.new_zeros(()))   # (n, D, 3)
        sl = slot[idx].unsqueeze(1)
        sw = rate[idx] * (sl & elig[idx].view(-1, 1, 1)) * T_SWAP
        ready = torch.where(sl, last + sw, rate.new_full((), neg)).amax(-1)               # (n, D)
        st = ready + torch.where(elig[idx].unsqueeze(1), wait[idx] * T_SWAP, rate.new_zeros(()))
        e = torch.maximum(st, torch.where(sl, last, rate.new_full((), neg)).amax(-1)) + b.t[idx].unsqueeze(1)
        end = end.index_put((idx,), e)
    return scatter(end, b.batch, 0, reduce="max")                                    # (B, D)


def duration_loss(b, rate, wait):
    """v25: Huber on log(schedule of the predicted SWAPs and waits / Sabre's replayed routed schedule), routed cases:
    trains SWAPs and waits on what sets the duration (the critical path), not gate by gate."""
    dur = schedule_torch(b, rate, wait)
    m = b.routed & torch.isfinite(b.replay) & (b.replay > 0)
    err = torch.log(dur.clamp(min=1e-3)) - torch.log(b.replay.clamp(min=1e-3))
    return (F.huber_loss(err, torch.zeros_like(err), reduction="none", delta=0.2) * m).sum() / m.sum().clamp(min=1)


def schedule_np(gq, t, s, wait, use_wait):
    """Sequential ASAP over the gates: SWAPs before a gate occupy its wires, plus the gate's wait."""
    last = {}
    end = 0.0
    for i in range(len(gq)):
        qs = [int(q) for q in gq[i] if q >= 0]
        if not qs:
            continue
        ready = max(last.get(q, 0.0) + s[i, k] * T_SWAP for k, q in enumerate(qs))
        st = ready + (wait[i] if use_wait[i] else 0.0)
        e = max(st, max(last.get(q, 0.0) for q in qs)) + t[i]
        for q in qs:
            last[q] = e
        end = max(end, e)
    return end


def schedule_k(gq, t, s, wait, use_wait):
    """schedule_np vectorised over K samples of the SWAPs: s (N, 3, K) -> (K,) durations."""
    K = s.shape[-1]
    last = {}
    end = np.zeros(K)
    zero = np.zeros(K)
    for i in range(len(gq)):
        qs = [int(q) for q in gq[i] if q >= 0]
        if not qs:
            continue
        ready = np.max([last.get(q, zero) + s[i, k] * T_SWAP for k, q in enumerate(qs)], 0)
        st = ready + (wait[i] if use_wait[i] else 0.0)
        e = np.maximum(st, np.max([last.get(q, zero) for q in qs], 0)) + t[i]
        for q in qs:
            last[q] = e
        end = np.maximum(end, e)
    return end


@torch.no_grad()
def evaluate(model, items, bs=32, mc=0, ds=None, device="cpu", seed=0):
    """Per (circuit, device) rows. mc > 0 adds pred_mc / pred1_mc = median over mc schedules of Poisson-sampled SWAPs
    (the schedule of the mean rates under-estimates the longest path); ds = esp_labels dict -> final_ds (µs)."""
    model.eval()
    rng = np.random.default_rng(seed)
    rows = []
    for s0 in range(0, len(items), bs):
        b = next(iter(DataLoader(items[s0: s0 + bs], batch_size=bs))).to(device)
        rate, wait, c2 = (o.cpu() for o in model(b))
        ptr = b.ptr.tolist()
        for i in range(len(ptr) - 1):
            a, e = ptr[i], ptr[i + 1]
            it = items[s0 + i]
            gq = it.gate_qubits.numpy(); t = it.t.numpy()
            meas = it.x[:, GATE2IDX_MEASURE].numpy() > 0.5
            elig = ((gq >= 0).sum(1) >= 2) | meas
            slot = (gq >= 0) & elig[:, None]                                       # real operand slots that carry SWAPs
            lab_ds = ds.get(it.name) if ds is not None else None
            for d in range(len(DEVS)):
                f_ds = {"final_ds": float(lab_ds["duration"][d]) * 1e6} if lab_ds is not None else {}
                k2 = math.exp(float(c2[i, d]))
                if bool(it.rf[0, d]):
                    zero = np.zeros((len(gq), 3))
                    p0 = schedule_np(gq, t, zero, np.zeros(len(gq)), elig)
                    row = {"name": it.name, "d": d, "rf": True, "final": float(it.final[0, d]), "pred1": p0, "pred": p0 * k2}
                    if mc:
                        row.update(pred1_mc=p0, pred_mc=p0 * k2)
                    rows.append({**row, **f_ds})
                    continue
                if not bool(it.routed[0, d]):
                    continue
                r = rate[a:e, d].numpy() * slot
                w = wait[a:e, d].numpy() * T_SWAP
                s_true = it.s[:, d].numpy() * slot; w_true = it.wait[:, d].numpy()
                p1 = schedule_np(gq, t, r, w, elig)
                row = {"name": it.name, "d": d, "rf": False, "final": float(it.final[0, d]),
                       "replay": float(it.replay[0, d]), "prior": float(it.prior_dur[0, d]),
                       "pred1": p1, "pred": p1 * k2,
                       "replay_c2": float(it.replay[0, d]) * k2,
                       "pred_nowait": schedule_np(gq, t, r, w, np.zeros(len(gq), bool)),
                       "oracle_s": schedule_np(gq, t, s_true, w_true, np.zeros(len(gq), bool)),
                       "oracle_sw": schedule_np(gq, t, s_true, w_true, elig),
                       "sw_pred": float(r.sum()), "sw_true": float(s_true.sum())}
                if mc:
                    pk = float(np.median(schedule_k(gq, t, rng.poisson(np.repeat(r[..., None], mc, -1)), w, elig)))
                    row.update(pred1_mc=pk, pred_mc=pk * k2)
                rows.append({**row, **f_ds})
    return rows


def _err(p, fin):
    e = np.abs(p / fin - 1)
    return {"err": float(np.median(e)), "p90": float(np.percentile(e, 90)), "bias": float(np.median(p / fin)),
            "w10": float((e < 0.1).mean())}


def summary(rows):
    """Errors vs Sabre's final duration (the training target); with final_ds also vs the dataset label (esp_labels.pt),
    the target of the v19i-v22 scorecard."""
    out = {}
    mc = rows and "pred_mc" in rows[0]
    rt = [r for r in rows if not r["rf"]]
    for grp, sel in (("routed", rt), ("routing-free", [r for r in rows if r["rf"]])):
        if not sel:
            continue
        fin = np.array([r["final"] for r in sel])
        res = {"n": len(sel)}
        keys = ("pred", "pred1", "pred_nowait", "oracle_s", "oracle_sw", "replay", "replay_c2", "prior") if grp == "routed" else ("pred", "pred1")
        keys += ("pred_mc", "pred1_mc") if mc else ()
        for k in keys:
            res[k] = _err(np.array([r[k] for r in sel]), fin)
        if grp == "routed":
            sp, st = np.array([r["sw_pred"] for r in sel]), np.array([r["sw_true"] for r in sel])
            res["swaps"] = {"err": float(np.median(np.abs(sp / np.maximum(st, 1) - 1))), "bias": float(np.median(sp / np.maximum(st, 1))),
                            "mean_ratio": float(sp.mean() / max(st.mean(), 1e-9)),
                            "corr_log": float(np.corrcoef(np.log1p(sp), np.log1p(st))[0, 1])}
        ds = [r for r in sel if r.get("final_ds", 0) > 0]
        if ds:
            fd = np.array([r["final_ds"] for r in ds])
            res["vs_dataset"] = {"n": len(ds), "sabre_final": _err(np.array([r["final"] for r in ds]), fd),
                                 **{k: _err(np.array([r[k] for r in ds]), fd) for k in ("pred", "pred1") + (("pred_mc", "pred1_mc") if mc else ())}}
        out[grp] = res
    return out


def split_names(split: str, names: list[str]):
    fam = lambda n: n.split("/")[0]
    if split.startswith("loso:"):
        structs = split[5:].split(",")
        test = [n for n in names if fam(n) == "synth" and n.split("/")[1].split("_")[1] in structs]
        train = [n for n in names if fam(n) != "qaoa" and fam(n) not in VAR_FAMS and n not in set(test)]
        return train, test
    lst = json.loads((EVAL / f"generalization_v10/results/v8kdfi10/{SPLIT_LISTS[split]}/predictions.json").read_text())
    test = [n for n in lst if n in set(names)]
    if split == "control":
        tset = set(test)
        return [n for n in names if n not in tset], test
    held = {fam(n) for n in test}
    return [n for n in names if fam(n) not in held], test


def contention_table(graphs, tr, tc, cache: Path):
    """{circuit: (N, D) router contention wait} for every graph, computed once and cached (a few minutes)."""
    if cache.is_file():
        return torch.load(cache, weights_only=False)
    chs = raw_chips()
    out = {}
    t0 = time.time()
    for i, g in enumerate(graphs):
        gq = g.gate_qubits.numpy()
        t = node_t(g, tr, tc)
        meas = g.x[:, GATE2IDX_MEASURE].numpy() > 0.5
        out[g.circuit_name] = torch.from_numpy(np.stack([router_contention(gq, t, meas, int(g.n_qubits), c) for c in chs], 1))
        if i % 2000 == 0:
            print(f"contention {i}/{len(graphs)} ({time.time() - t0:.0f} s)", flush=True)
    torch.save(out, cache)
    return out


def load_all(path: Path, contention: bool = False):
    s1 = torch.load(path, weights_only=False)
    graphs = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    graphs += torch.load(EVAL / "synthetic_routing/results/synth_graphs.pt", weights_only=False)
    graphs = [g for g in graphs if g.circuit_name in s1]
    tab = json.loads((EVAL / "esp/results/decomp_table.json").read_text())
    U = list(fe.UNIQUE)
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    ct = contention_table(graphs, tr, tc, path.parent / "contention.pt") if contention else None
    items = build(graphs, s1, tr, tc, ct)
    return {it.name: it for it in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--split", default="loso:dense,complete")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--wait-lambda", type=float, default=0.3)
    ap.add_argument("--c2-lambda", type=float, default=1.0)
    ap.add_argument("--dur-lambda", type=float, default=0.0,
                    help="v25: weight of the loss on the routed duration through the differentiable schedule")
    ap.add_argument("--dur-every", type=int, default=1, help="v25: duration loss on every k-th training batch (cost)")
    ap.add_argument("--contention", action="store_true",
                    help="v26: our greedy router's contention wait per gate as input and as the base of the predicted wait")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "results/s1")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=HERE / "results/stage1.pt")
    ap.add_argument("--mc", type=int, default=0, help="eval: also schedule this many Poisson samples of the SWAPs (median)")
    ap.add_argument("--esp-labels", type=Path, default=None, help="eval: also score against the dataset durations (esp_labels.pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="", help="suffix of the eval / rows file names")
    ap.add_argument("--max-test", type=int, default=0, help="smoke tests: evaluate only the first N test circuits")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = load_all(args.data, args.contention)
    train_n, test_n = split_names(args.split, sorted(data))
    if args.max_test:
        test_n = test_n[:args.max_test]
    rng = np.random.default_rng(args.seed)
    train_n = list(rng.permutation(train_n))
    n_val = max(1, len(train_n) // 20)
    val_n, train_n = train_n[:n_val], train_n[n_val:]
    print(f"split {args.split}: train {len(train_n)}, val {len(val_n)}, test {len(test_n)}", flush=True)
    f_in = data[train_n[0]].x.size(1) + 9 + 12 + (3 if args.contention else 0)
    model = Stage1(f_in, args.hidden, args.layers, contention=args.contention).to(args.device)
    if args.contention:      # signal check: router contention wait vs Sabre's (teacher-forced) wait on routed matched gates
        pw, tw = [], []
        for n in train_n[:2000]:
            it = data[n]
            m = it.m & it.routed
            pw.append(it.pw[m]); tw.append(it.wait[m])
        pw, tw = torch.cat(pw).numpy(), torch.cat(tw).numpy()
        if len(pw) > 2:
            print(f"contention check: corr(router wait, Sabre wait) {np.corrcoef(pw, tw)[0, 1]:.3f}, "
                  f"router > 0 on {np.mean(pw > 1e-9):.1%} of routed gates", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "train":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * math.ceil(len(train_n) / args.bs))
        tr_items = [data[n] for n in train_n]
        best, hist = float("inf"), []
        for ep in range(args.epochs):
            model.train()
            t0 = time.time(); tot = defaultdict(float); nb = 0
            for bi, b in enumerate(DataLoader(tr_items, batch_size=args.bs, shuffle=True)):
                b = b.to(args.device)
                rate, wait, c2 = model(b)
                l_s, l_w, l_c = losses(b, rate, wait, c2)
                loss = l_s + args.wait_lambda * l_w + args.c2_lambda * l_c
                if args.dur_lambda and bi % args.dur_every == 0:
                    l_d = duration_loss(b, rate, wait)
                    loss = loss + args.dur_lambda * l_d
                    tot["d"] += l_d.item(); tot["nd"] += 1
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot["s"] += l_s.item(); tot["w"] += l_w.item(); tot["c"] += l_c.item(); nb += 1
            model.eval()
            with torch.no_grad():
                vl = []
                for b in DataLoader([data[n] for n in val_n], batch_size=args.bs):
                    b = b.to(args.device)
                    rate, wait, c2 = model(b)
                    l_s, l_w, l_c = losses(b, rate, wait, c2)
                    v_ = l_s + args.wait_lambda * l_w + args.c2_lambda * l_c
                    if args.dur_lambda:
                        v_ = v_ + args.dur_lambda * duration_loss(b, rate, wait)
                    vl.append(v_.item())
            v = float(np.mean(vl))
            hist.append({"epoch": ep, "train_s": tot["s"] / nb, "train_w": tot["w"] / nb, "train_c": tot["c"] / nb,
                         "train_d": tot["d"] / max(tot["nd"], 1), "val": v,
                         "sec": time.time() - t0})
            print(json.dumps(hist[-1]), flush=True)
            if v < best:
                best = v; torch.save(model.state_dict(), args.out / "model.pt")
        (args.out / "history.json").write_text(json.dumps(hist))
    ckpt = args.ckpt or args.out / "model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    ds = torch.load(args.esp_labels, weights_only=False) if args.esp_labels else None
    rows = evaluate(model, [data[n] for n in test_n], mc=args.mc, ds=ds, device=args.device, seed=args.seed)
    res = {"split": args.split, "n_test_circuits": len(test_n), "ckpt": str(ckpt), "mc": args.mc, **summary(rows)}
    stem = args.split.replace(':', '_').replace(',', '-') + args.tag
    (args.out / f"eval_{stem}.json").write_text(json.dumps(res, indent=1))
    torch.save(rows, args.out / f"rows_{stem}.pt")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
