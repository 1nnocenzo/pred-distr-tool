"""One model for F, duration and ESP (the v29 design, trained jointly).

  backbone  gate DAG --SAGE--> gate embeddings g --> logical-qubit embeddings (mean/max + 2 GINE on the interaction
            graph); device graph --3 GINE--> physical-qubit embeddings; placement scores + Sinkhorn (τ 0.3, 20 iter.)
            = soft one-to-one layout A.  Per gate and device: operand context, placement-weighted errors, expected
            operand distance Aᵀ H A, routing-free flag, routing estimate -> native counts (r, measure exact, cz =
            base(gate) · (1 + κ · extra hops · (1 − routing-free))) -> log F = − Σ_g Σ_k s_k n_k ε_k.
  route     gate-graph network (inputs: gate features, wire features, greedy-router SWAPs / hops, the backbone's
            gate embedding and operand distances) -> per gate and device: SWAP rate per operand wire (Poisson),
            extra wait; pooled: c2 = log(final duration / routed schedule).  Routing-free pairs get no SWAPs / waits.
            duration = level-wise ASAP schedule(SWAPs, waits) × e^c2.
  ESP       log ESP = log F + decay,  decay = −e^a · [(N_log + N_extra) · duration − busy] · mean 1/min(T1,T2):
            the idle time summed over the active physical qubits.  busy = r · 42 ns + 2 · cz · 130 ns + measure · 15 µs
            from the backbone's counts (exact for true counts); N_log = logical qubits with a gate, N_extra = physical
            qubits only routing activates (predicted, 0 when routing-free); T: routing-free -> the physical qubits the
            backbone's HARD placement (Hungarian on A, the same placement as F) gives the logical qubits, routed -> the
            device mean; a, N_extra from a small head.  Formula check with true inputs, routed: 0.06 nats with the
            compiler's layout, 0.145 with the device mean (the learned placement gave ~0.4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, GraphNorm, SAGEConv
from torch_geometric.utils import scatter, to_dense_batch

from .config import T1, TCZ, TM, T_SWAP
from .encoding import MEASURE, N_FEATURES, N_GATE_LOCAL

HP = {"hidden": 142, "plain_layers": 3, "res_layers": 1, "d": 128, "qubit_layers": 2, "device_layers": 3,
      "mlp": (512, 256, 128), "tau": 0.3, "iters": 20, "route_hidden": 96, "route_layers": 6, "dev_emb": 16}
SOFTPLUS_ONE = 0.5413                       # softplus(0.5413) = 1


# ------------------------------------------------------------------------------------------------ backbone
class CircuitEncoder(nn.Module):
    """Bidirectional SAGE (forward and reverse edges averaged), GraphNorm, residual after the first layers."""

    def __init__(self, in_dim: int, hidden: int, plain: int, res: int):
        super().__init__()
        self.convs = nn.ModuleList(SAGEConv(in_dim if i == 0 else hidden, hidden) for i in range(plain + res))
        self.norms = nn.ModuleList(GraphNorm(hidden) for _ in range(plain + res))
        self.res_start = plain

    def forward(self, x, edge_index, batch):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = F.leaky_relu(norm((conv(x, edge_index) + conv(x, edge_index.flip(0))) / 2, batch))
            x = h if i < self.res_start else x + h
        return x


def _gine(dim: int, edge_dim: int) -> GINEConv:
    return GINEConv(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(), nn.Linear(dim, dim)), edge_dim=edge_dim)


def _mlp(n_in: int, widths, n_out: int, bias: float) -> nn.Sequential:
    layers, last = [], n_in
    for w in widths:
        layers += [nn.Linear(last, w), nn.LeakyReLU()]
        last = w
    out = nn.Linear(last, n_out)
    nn.init.constant_(out.bias, bias)
    return nn.Sequential(*layers, out)


def hop_matrices(devices) -> torch.Tensor:
    """(D, P, P) hop distances on each coupling map (padded)."""
    sizes = torch.bincount(devices.batch).tolist()
    p_max, off, mats = max(sizes), 0, []
    for n in sizes:
        sel = (devices.edge_index[0] >= off) & (devices.edge_index[0] < off + n)
        ei = devices.edge_index[:, sel] - off
        h = torch.full((n, n), float(n + 1))
        h.fill_diagonal_(0.0)
        h[ei[0], ei[1]] = 1.0
        for k in range(n):                                  # Floyd–Warshall, n <= 27
            h = torch.minimum(h, h[:, k:k + 1] + h[k:k + 1, :])
        mats.append(F.pad(h, (0, p_max - n, 0, p_max - n)))
        off += n
    return torch.stack(mats)


class Backbone(nn.Module):
    """F network with a Sinkhorn placement (the v8k / v8kdfi10 F model without its ESP head)."""

    def __init__(self, devices, hp=HP):
        super().__init__()
        d = hp["d"]
        self.enc = CircuitEncoder(N_FEATURES, hp["hidden"], hp["plain_layers"], hp["res_layers"])
        self.gate_proj = nn.Linear(hp["hidden"], d)
        self.qubit_in = nn.Linear(2 * d, d)
        self.qubit_convs = nn.ModuleList(_gine(d, d) for _ in range(hp["qubit_layers"]))
        self.dev_in = nn.Linear(devices.x.size(1), d)
        self.dev_convs = nn.ModuleList(_gine(d, devices.edge_attr.size(1)) for _ in range(hp["device_layers"]))
        self.w_q, self.w_p = nn.Linear(d, d), nn.Linear(d, d)
        self.dummy_score = nn.Parameter(torch.zeros(()))
        self.tau, self.iters = hp["tau"], hp["iters"]
        self.cost = _mlp(5 * d + 4, hp["mlp"], 3, SOFTPLUS_ONE)          # count head: r, measure, κ (cz)
        self.cz_base = nn.Linear(N_GATE_LOCAL, 1)
        nn.init.constant_(self.cz_base.bias, SOFTPLUS_ONE)
        self.log_scale = nn.Parameter(torch.zeros(3))
        self.register_buffer("hops", hop_matrices(devices.cpu()), persistent=False)
        self.d = d

    def _qubits(self, b, g):
        n_q = b.n_qubits.view(-1).long()
        offsets = torch.cumsum(n_q, 0) - n_q
        gq = b.gate_qubits
        valid = gq >= 0
        glob = torch.where(valid, gq + offsets[b.batch].unsqueeze(1), -1)
        gate_ids = torch.arange(g.size(0), device=g.device).unsqueeze(1).expand_as(gq)
        src, dst = gate_ids[valid], glob[valid]
        n_tot = int(n_q.sum())
        q = F.leaky_relu(self.qubit_in(torch.cat([scatter(g[src], dst, 0, dim_size=n_tot, reduce="mean"),
                                                  scatter(g[src], dst, 0, dim_size=n_tot, reduce="max")], -1)))
        s_, d_, eg = [], [], []
        for a, c in ((0, 1), (0, 2), (1, 2)):               # interaction graph: one edge per operand pair
            m = valid[:, a] & valid[:, c]
            if m.any():
                s_ += [glob[m, a], glob[m, c]]
                d_ += [glob[m, c], glob[m, a]]
                eg += [gate_ids[m, 0]] * 2
        if s_:
            ei, ea = torch.stack([torch.cat(s_), torch.cat(d_)]), g[torch.cat(eg)]
            for conv in self.qubit_convs:
                q = q + F.leaky_relu(conv(q, ei, ea))
        return q, valid

    def _devices(self, dev):
        x = F.leaky_relu(self.dev_in(dev.x))
        for conv in self.dev_convs:
            x = x + F.leaky_relu(conv(x, dev.edge_index, dev.edge_attr))
        dense, mask = to_dense_batch(x, dev.batch)
        return dense, mask, scatter(x, dev.batch, 0, reduce="mean")

    def _sinkhorn(self, scores, n_logical, n_phys):
        """(B, D, Qm, P) scores -> soft one-to-one assignment; unused physical qubits go to dummy rows."""
        b, n_dev, q_max, p_max = scores.shape
        rows = torch.arange(p_max, device=scores.device)
        real = rows.view(1, 1, -1) < n_logical.view(-1, 1, 1)
        used = rows.view(1, 1, -1) < n_phys.view(1, -1, 1)
        col = rows.view(1, -1) < n_phys.view(-1, 1)
        head = torch.where(real[..., :q_max].unsqueeze(-1), scores, self.dummy_score.expand_as(scores))
        full = torch.cat([head, self.dummy_score.expand(b, n_dev, p_max - q_max, p_max)], 2)
        mask = (used.unsqueeze(-1) & col.view(1, n_dev, 1, p_max)).expand(b, -1, -1, -1)
        neg = torch.full_like(full, -1e4)                   # finite: -inf gives NaN gradients in logsumexp
        la = torch.where(mask, full / self.tau, neg)
        for _ in range(self.iters):
            la = torch.where(mask, la - torch.logsumexp(la, -1, keepdim=True), neg)
            la = torch.where(mask, la - torch.logsumexp(la, -2, keepdim=True), neg)
        return (torch.exp(la) * mask)[:, :, :q_max] * real[..., :q_max].unsqueeze(-1)

    def forward(self, b, dev) -> dict:
        g = self.gate_proj(self.enc(b.x, b.edge_index, b.batch))           # (G, d)
        q, valid = self._qubits(b, g)
        phys, pmask, dev_glob = self._devices(dev)                         # (D, P, d)
        n_dev, p_max = phys.shape[:2]
        n_q = b.n_qubits.view(-1).long()
        n_c = n_q.numel()
        q_dense, q_mask = to_dense_batch(q, torch.repeat_interleave(torch.arange(n_c, device=g.device), n_q))
        scores = torch.einsum("bqk,dpk->bdqp", self.w_q(q_dense), self.w_p(phys)) / self.d ** 0.5
        assign = self._sinkhorn(scores, n_q, pmask.sum(1))                 # (B, D, Qm, P)
        ctx = torch.einsum("bdqp,dpk->bdqk", assign, phys)
        eps, _ = to_dense_batch(dev.phys, dev.batch)                       # (D, P, 3)
        q_eps = torch.einsum("bdqp,dpk->bdqk", assign, eps)
        a_h = torch.einsum("bdqp,dpr->bdqr", assign, self.hops[:, :p_max, :p_max])
        gb, local = b.batch, b.gate_qubits.clamp(min=0)
        vmask = valid.unsqueeze(-1).unsqueeze(-1)
        cnt = valid.sum(1).clamp(min=1).view(-1, 1, 1).float()
        slots = ctx[gb.unsqueeze(1), :, local]                             # (G, 3, D, d)
        c_mean = (slots * vmask).sum(1) / cnt
        multi = valid.sum(1).view(-1, 1, 1) > 1
        c_range = torch.where(multi, slots.masked_fill(~vmask, -1e4).amax(1) - slots.masked_fill(~vmask, 1e4).amin(1),
                              torch.zeros_like(c_mean))
        g_eps = (q_eps[gb.unsqueeze(1), :, local] * vmask).sum(1) / cnt    # (G, D, 3)
        dist = torch.zeros(g.size(0), n_dev, device=g.device)              # expected operand distance (pair mean)
        n_pairs = torch.zeros(g.size(0), 1, device=g.device)
        for a, c in ((0, 1), (0, 2), (1, 2)):
            m = (valid[:, a] & valid[:, c]).float().unsqueeze(-1)
            dist = dist + m * (a_h[gb, :, local[:, a]] * assign[gb, :, local[:, c]]).sum(-1)
            n_pairs = n_pairs + m
        dist = dist / n_pairs.clamp(min=1)
        rf, route = b.rf_pred.view(n_c, n_dev), b.route.view(n_c, n_dev)
        g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
        z = torch.cat([g_e, c_mean, c_range, dev_glob.unsqueeze(0).expand(g.size(0), -1, -1), g_e * c_mean,
                       dist.unsqueeze(-1), torch.log1p(dist).unsqueeze(-1), rf[gb].unsqueeze(-1), route[gb].unsqueeze(-1)], -1)
        n_ops = F.softplus(self.cost(z))                                   # (G, D, 3)
        base = F.softplus(self.cz_base(b.x[:, :N_GATE_LOCAL])) * multi.view(-1, 1)
        cz = base * (1 + n_ops[..., 2] * torch.expm1(route)[gb] * (1 - rf[gb]))
        meas = (b.x[:, MEASURE] > 0.5).float().view(-1, 1).expand(-1, n_dev)   # compilation keeps the measurements
        n_ops = torch.stack([n_ops[..., 0], meas, cz], -1)
        cost = (n_ops * g_eps * self.log_scale.exp()).sum(-1)
        return {"logf": -scatter(cost, gb, 0, dim_size=n_c, reduce="sum"), "g": g, "dist": dist, "assign": assign,
                "q_mask": q_mask, "col_mask": pmask, "counts": scatter(n_ops, gb, 0, dim_size=n_c, reduce="sum"),
                "dev_glob": dev_glob}


# ------------------------------------------------------------------------------------------------ route stage
class RouteStage(nn.Module):
    def __init__(self, f_in: int, hidden: int, layers: int, dev_emb: int):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                                    for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden + dev_emb + 4, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        self.head2 = nn.Sequential(nn.Linear(2 * hidden + dev_emb, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, b, extra, dev_e):
        """-> SWAP rates (N, D, 3), wait / T_SWAP (N, D), c2 (B, D), node states (N, H)."""
        N, n_dev = b.hops.shape
        df = torch.cat([torch.log1p(b.hops.clamp(min=0)).unsqueeze(-1), torch.log1p(b.prior_s)], -1)   # (N, D, 4)
        h = self.inp(torch.cat([b.x, b.wf, df.flatten(1), extra], 1))
        src, dst = b.edge_index
        for lay, nrm in zip(self.layers, self.norms):
            fwd = scatter(h[src], dst, 0, dim_size=N, reduce="mean")
            bwd = scatter(h[dst], src, 0, dim_size=N, reduce="mean")
            glob = scatter(h, b.batch, 0, reduce="mean")[b.batch]
            h = nrm(h + lay(torch.cat([h, fwd, bwd, glob], 1)))
        o = self.head(torch.cat([h.unsqueeze(1).expand(-1, n_dev, -1), dev_e.unsqueeze(0).expand(N, -1, -1), df], -1))
        pooled = torch.cat([scatter(h, b.batch, 0, reduce="mean"), scatter(h, b.batch, 0, reduce="max")], 1)
        c2 = self.head2(torch.cat([pooled.unsqueeze(1).expand(-1, n_dev, -1),
                                   dev_e.unsqueeze(0).expand(pooled.size(0), -1, -1)], -1)).squeeze(-1)
        return F.softplus(o[..., :3]), o[..., 3], c2, h


def swap_slots(b):
    """(N, 3) operand slots that carry SWAPs: real operands of multi-qubit gates and measurements."""
    slot = b.gate_qubits >= 0
    elig = (slot.sum(1) >= 2) | (b.x[:, MEASURE] > 0.5)
    return slot, elig


def schedule_torch(b, rate, wait):
    """(B, D) ASAP duration µs with rate (N, D, 3) SWAPs before each gate on each operand wire and wait (N, D) ×
    T_SWAP extra start delay; one pass per topological level, differentiable in rate and wait."""
    N, D = rate.shape[:2]
    off = b.ptr[:-1][b.batch].unsqueeze(1)
    has = b.prev >= 0
    gprev = torch.where(has, b.prev + off, torch.zeros_like(b.prev))
    slot, elig = swap_slots(b)
    end = rate.new_zeros(N, D)
    neg = torch.finfo(rate.dtype).min / 4
    for L in range(int(b.lvl.max()) + 1 if N else 0):
        idx = (b.lvl == L).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        last = torch.where(has[idx].unsqueeze(1), end[gprev[idx]].transpose(1, 2), rate.new_zeros(()))   # (n, D, 3)
        sl = slot[idx].unsqueeze(1)
        sw = rate[idx] * (sl & elig[idx].view(-1, 1, 1)) * T_SWAP
        ready = torch.where(sl, last + sw, rate.new_full((), neg)).amax(-1)
        st = ready + torch.where(elig[idx].unsqueeze(1), wait[idx] * T_SWAP, rate.new_zeros(()))
        e = torch.maximum(st, torch.where(sl, last, rate.new_full((), neg)).amax(-1)) + b.t[idx].unsqueeze(1)
        end = end.index_put((idx,), e)
    return scatter(end, b.batch, 0, dim_size=b.num_graphs, reduce="max")


def mc_schedule(b, rate, wait, K: int, gen=None):
    """Median over K schedules of Poisson-sampled SWAPs (the mean-rate schedule underestimates the longest path)."""
    N, D, _ = rate.shape
    s = torch.poisson(rate.unsqueeze(-1).expand(N, D, 3, K).contiguous(), generator=gen)
    s = s.permute(0, 1, 3, 2).reshape(N, D * K, 3)
    w = wait.unsqueeze(-1).expand(N, D, K).reshape(N, D * K)
    return schedule_torch(b, s, w).view(-1, D, K).median(-1).values


@torch.no_grad()
def hard_placement(assign: torch.Tensor) -> torch.Tensor:
    """One physical qubit per logical qubit: Hungarian on the Sinkhorn matrix, per circuit and device."""
    from scipy.optimize import linear_sum_assignment
    a = assign.detach().float().cpu().numpy()
    hard = np.zeros_like(a)
    for i in range(a.shape[0]):
        for d in range(a.shape[1]):
            m = a[i, d]
            rows, cols = np.flatnonzero(m.sum(1) > 0), np.flatnonzero(m.sum(0) > 0)
            if len(rows) == 0 or len(cols) < len(rows):
                hard[i, d] = m
                continue
            r, c = linear_sum_assignment(-m[np.ix_(rows, cols)])
            hard[i, d, rows[r], cols[c]] = 1.0
    return torch.from_numpy(hard).to(assign)


# ------------------------------------------------------------------------------------------------ the model
class ESPTimeModel(nn.Module):
    def __init__(self, devices, hp=HP, esp_grad: str = "head"):
        super().__init__()
        self.backbone = Backbone(devices, hp)
        n_dev, d, H, de = int(devices.num_graphs), hp["d"], hp["route_hidden"], hp["dev_emb"]
        self.esp_grad = esp_grad
        self.emb = nn.Sequential(nn.Linear(d, 32), nn.SiLU())
        self.dev_proj = nn.Linear(d, de)
        self.route = RouteStage(N_FEATURES + 9 + 4 * n_dev + 32 + n_dev, H, hp["route_layers"], de)
        self.esp = nn.Sequential(nn.Linear(2 * H + de + 5, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.esp[-1].weight)
        with torch.no_grad():
            self.esp[-1].bias.copy_(torch.tensor([0.0, -2.0]))        # scale e^0 = 1, extra qubits softplus(-2) = 0.13

    def forward(self, b, dev, mc: int = 0, gen=None) -> dict:
        out = self.backbone(b, dev)
        dist = out["dist"]
        extra = torch.cat([self.emb(out["g"]), torch.log1p(dist.clamp(min=0))], 1)
        dev_e = self.dev_proj(out["dev_glob"])
        rate, wait, c2, h = self.route(b, extra, dev_e)
        n_c, n_dev = b.num_graphs, rate.size(1)
        rf = b.rf_pred.view(n_c, n_dev) > 0.5                                   # VF2 switch, no compiler
        nr = (~rf)[b.batch]
        rate_e = rate * nr.unsqueeze(-1)
        wait_e = torch.where(nr, wait, torch.zeros_like(wait))
        dur = schedule_torch(b, rate_e, wait_e) * torch.exp(c2)                 # (B, D) µs
        out.update(rate=rate, wait=wait, c2=c2, dur=dur)
        if mc:
            out["dur_mc"] = mc_schedule(b, rate_e.detach(), wait_e.detach(), mc, gen) * torch.exp(c2.detach())
        # ---- ESP, count formula: the idle time summed over the active PHYSICAL qubits is N_act · duration − Σ_p busy_p,
        # and Σ_p busy_p is fixed by the compiled counts (r · 42 ns + 2 · cz · 130 ns + measure · 15 µs, exact);
        # N_act = active logical qubits + the extra physical qubits routing activates (predicted, 0 if routing-free);
        # 1/min(T1, T2): routing-free -> mean over the physical qubits the hard placement gives the active logical qubits;
        # routed -> mean over the device (the learned placement is further from the compiler's qubits than the device
        # mean on routed circuits, and routing moves the qubits anyway; formula check: 0.145 nats with true inputs)
        head_only = self.esp_grad == "head"
        dur_e = dur.detach() if head_only else dur
        counts = out["counts"].detach() if head_only else out["counts"]
        busy_tot = counts @ counts.new_tensor([T1, TM, 2 * TCZ])                     # (B, D) µs
        q_max = out["assign"].size(2)
        slot, elig = swap_slots(b)
        used = rate.new_zeros(n_c * q_max).index_fill(
            0, (b.batch.unsqueeze(1) * q_max + b.gate_qubits.clamp(min=0))[slot], 1.0).view(n_c, q_max)
        coh, cmask = to_dense_batch(dev.coh, dev.batch)                          # (D, P) 1/µs
        inv_t = torch.einsum("bdqp,dp->bdq", hard_placement(out["assign"]), coh.to(rate))
        n_log = used.sum(-1, keepdim=True).clamp(min=1)                           # (B, 1)
        inv_dev = (coh.sum(1) / cmask.sum(1)).to(rate)                           # (D,)
        inv_m = torch.where(rf, (inv_t * used.unsqueeze(1)).sum(-1) / n_log, inv_dev.expand(n_c, -1))   # (B, D)
        pooled = torch.cat([scatter(h, b.batch, 0, reduce="mean"), scatter(h, b.batch, 0, reduce="max")], 1)
        if head_only:
            pooled = pooled.detach()
        nsw = scatter((rate_e.detach() * (slot & elig.view(-1, 1)).unsqueeze(1)).sum(-1), b.batch, 0, dim_size=n_c,
                      reduce="sum")
        z = torch.cat([pooled.unsqueeze(1).expand(-1, n_dev, -1), dev_e.unsqueeze(0).expand(n_c, -1, -1),
                       torch.stack([torch.log1p(dur_e), torch.log1p(busy_tot), torch.log1p(n_log.expand(-1, n_dev)),
                                    torch.log1p(nsw), rf.float()], -1)], -1)
        o = self.esp(z)
        n_extra = F.softplus(o[..., 1]) * (~rf)
        idle_tot = ((n_log + n_extra) * dur_e - busy_tot).clamp(min=0)
        decay = -idle_tot * inv_m * torch.exp(o[..., 0])
        out["n_extra"] = n_extra
        out.update(decay=decay, logesp=out["logf"] + decay)
        return out


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

