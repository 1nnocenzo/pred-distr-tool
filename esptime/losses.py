"""Training losses.  Every term is masked to the labels it has; ``total`` sums them with the weights of ``LAMBDAS``.

  f        MSE on F + 0.1 · Huber on log F (exact label, not clipped)        (backbone)
  region   compiler's layout: occupancy BCE + operand-distance MSE (log1p)    (backbone placement)
  count    native counts, MSE on log1p, kind weights r 1, cz 1 (measure is exact)   (backbone count head; the ESP's
           busy time is computed from these counts)
  swap     Poisson NLL of Sabre's SWAPs per operand slot, routed cases        (route stage)
  wait     Huber on the teacher-forced wait / T_SWAP, matched gates
  c2       Huber on log(final duration / base)
  dur      Huber on log(schedule of predicted SWAPs and waits / Sabre's routed schedule), routed cases
  final    Huber on log(predicted duration / label), every case: end to end through SWAPs, waits and c2, so that
           c2 corrects the PREDICTED schedule (its own target c2 is measured on Sabre's schedule)
  decay    Huber on log(−decay + 0.01)                                        (ESP head; its relative error moves log ESP)
  extra    Poisson NLL of the extra physical qubits routing activates (ESP head)
  calib    on a calibration variant: MSE of the predicted vs true change of log F and of log ESP
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .config import T_SWAP
from .model import schedule_torch, swap_slots

LAMBDAS = {"f": 1.0, "region": 0.1, "count": 0.3, "swap": 1.0, "wait": 0.3, "c2": 1.0, "dur": 1.0, "final": 1.0,
           "decay": 1.0, "extra": 0.3, "calib": 1.0}
COUNT_W = (1.0, 0.0, 1.0)


def _masked_mean(err, m):
    return (err * m).sum() / m.sum().clamp(min=1)


def f_loss(logf, b, logf_lab=None, ok=None):
    logf_lab = b.logF if logf_lab is None else logf_lab
    ok = (b.ok if ok is None else ok).float()
    y = torch.exp(logf_lab)
    lin = (torch.exp(logf.clamp(max=0)) - y) ** 2
    # log term on the exact (unrounded) log F label, no clip: the old recipe's log max(F, 1e-3) taught every circuit
    # with F < 1e-3 to predict log F = -6.9, which halved the counts of deep routed circuits (their ESP needs them)
    log = F.huber_loss(logf, logf_lab, reduction="none", delta=1.0)
    return _masked_mean(lin + 0.1 * log, ok)


def region_loss(out, b, hops):
    """Occupancy of each physical qubit vs the compiler's used qubits (BCE) + expected vs true operand distance."""
    assign, q_mask, col_mask = out["assign"], out["q_mask"], out["col_mask"]
    n_dev, p_max = col_mask.shape
    n_q = b.n_qubits.view(-1).long()
    n_c = n_q.numel()
    q_batch = torch.repeat_interleave(torch.arange(n_c, device=n_q.device), n_q)
    target, _ = to_dense_batch(b.layout, q_batch, fill_value=-1, max_num_nodes=assign.size(2))   # (B, Qm, D)
    known_q = (target >= 0) & q_mask.unsqueeze(-1)
    known_c = (known_q | ~q_mask.unsqueeze(-1)).all(1)
    used = torch.zeros(n_c, n_dev, p_max + 1, device=assign.device)
    used.scatter_(2, torch.where(known_q, target, torch.full_like(target, p_max)).permute(0, 2, 1), 1.0)
    used = used[..., :p_max]
    occ = assign.sum(2).clamp(1e-6, 1 - 1e-6)
    cell = (known_c.unsqueeze(-1) & col_mask.unsqueeze(0)).float()
    region = _masked_mean(-(used * occ.log() + (1 - used) * (1 - occ).log()), cell)
    offsets = torch.cumsum(n_q, 0) - n_q
    valid = b.gate_qubits >= 0
    glob = torch.where(valid, b.gate_qubits + offsets[b.batch].unsqueeze(1), torch.zeros_like(b.gate_qubits))
    dev = torch.arange(n_dev, device=assign.device).view(1, -1)
    true_d = torch.zeros(glob.size(0), n_dev, device=assign.device)
    n_pairs = torch.zeros_like(true_d)
    for a, c in ((0, 1), (0, 2), (1, 2)):
        la, lc = b.layout[glob[:, a]], b.layout[glob[:, c]]
        ok = (valid[:, a] & valid[:, c]).unsqueeze(1) & (la >= 0) & (lc >= 0)
        true_d = true_d + ok * hops[dev, la.clamp(min=0), lc.clamp(min=0)]
        n_pairs = n_pairs + ok
    dist = _masked_mean((torch.log1p(out["dist"]) - torch.log1p(true_d / n_pairs.clamp(min=1))) ** 2, n_pairs > 0)
    return region + dist


def count_loss(out, b):
    pred, true = out["counts"], b.counts.view_as(out["counts"])
    err = (torch.log1p(pred) - torch.log1p(true.clamp(min=0))) ** 2 * pred.new_tensor(COUNT_W)
    return _masked_mean(err, true >= 0)


def route_losses(out, b):
    rate, wait = out["rate"], out["wait"]
    slot, elig = swap_slots(b)
    routed = b.routed[b.batch]                                                  # (N, D)
    ms = (routed & elig.view(-1, 1)).unsqueeze(-1) & slot.unsqueeze(1)
    swap = _masked_mean(F.poisson_nll_loss(rate, b.s, log_input=False, reduction="none"), ms)
    w = _masked_mean(F.huber_loss(wait, b.wait / T_SWAP, reduction="none", delta=1.0), routed & b.m)
    c2 = _masked_mean(F.huber_loss(out["c2"], b.c2, reduction="none", delta=0.2), b.ok2)
    sched = schedule_torch(b, rate, wait)
    m = b.routed & torch.isfinite(b.replay) & (b.replay > 0)
    replay = torch.where(m, b.replay, torch.ones_like(b.replay))
    err = torch.where(m, torch.log(sched.clamp(min=1e-3)) - torch.log(replay.clamp(min=1e-3)), torch.zeros_like(sched))
    dur = _masked_mean(F.huber_loss(err, torch.zeros_like(err), reduction="none", delta=0.2), m)
    ok = b.ok & (b.dur > 0)
    e = torch.where(ok, torch.log(out["dur"].clamp(min=1e-3)) - torch.log(b.dur.clamp(min=1e-3)), torch.zeros_like(b.dur))
    final = _masked_mean(F.huber_loss(e, torch.zeros_like(e), reduction="none", delta=0.2), ok)
    return {"swap": swap, "wait": w, "c2": c2, "dur": dur, "final": final}


def decay_loss(decay, dec_lab, ok):
    p = torch.log(-decay + 1e-2)
    t = torch.log(-torch.where(ok, dec_lab, torch.zeros_like(dec_lab)) + 1e-2)
    err = torch.where(ok, p - t, torch.zeros_like(p))
    return _masked_mean(F.huber_loss(err, torch.zeros_like(err), reduction="none", delta=0.2), ok)


def calib_loss(out, out_v, b, k: int):
    """Change of log F and log ESP from the original calibration to variant k, predicted vs true (both F > 0.01)."""
    ok = b.ok & b.ok_v[:, k] & (b.logF > -4.6) & (b.logF_v[:, k] > -4.6)
    d_f = (out_v["logf"] - out["logf"]) - (b.logF_v[:, k] - b.logF)
    d_e = (out_v["logesp"] - out["logesp"]) - (b.logF_v[:, k] + b.decay_v[:, k] - b.logF - b.decay)
    return _masked_mean(d_f ** 2 + d_e ** 2, ok)


def total(out, b, hops, lambdas=LAMBDAS) -> tuple[torch.Tensor, dict]:
    parts = {"f": f_loss(out["logf"], b), "region": region_loss(out, b, hops), "count": count_loss(out, b),
             **route_losses(out, b), "decay": decay_loss(out["decay"], b.decay, b.ok),
             "extra": _masked_mean(F.poisson_nll_loss(out["n_extra"], b.n_extra, log_input=False, reduction="none"), b.ok)}
    loss = sum(lambdas[k] * v for k, v in parts.items() if lambdas.get(k))
    return loss, {k: float(v) for k, v in parts.items()}
