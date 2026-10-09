#!/usr/bin/env python3
"""Why does v4_phys generalise to new families while v5_sinkhorn survives calibration changes?

Four diagnostics on the trained seed-5 models (no retraining).  Run on a compute node::

    python diagnose.py all            # or: channels | uniform | parts | attention

``channels``  calibration-shift sample (``calibration_shift/results``), control models.
              Each device variant is fed to the model in four ways: unchanged (orig),
              full variant, variant only in the *device features* (node/edge features
              seen by the placement and the count MLP; physics-head errors from orig),
              variant only in the *physics-head errors* (features from orig).  R² vs the
              variant's true fidelity tells which channel breaks a model.
``uniform``   v5 with its Sinkhorn placement replaced by a uniform one (every logical
              qubit spread evenly over the device), on the calibration sample and on
              the LOFO-qaoa test set: if nothing moves, v5 effectively uses the
              device-average error.
``parts``     LOFO qaoa, v4 vs v5: the model's ``log F`` is a sum of three parts
              (1q gates, readout, 2q gates incl. routing); the true parts are computed
              from the compiled circuits (``compile_check/results/layouts_all/qpy``)
              with the actual error of each operation on its physical qubits.  Which
              part carries the v5 overestimate?
``attention`` v4 attention sharpness (max weight per logical qubit, share of logical
              qubits whose top physical qubit is shared) on orig vs variants, and its
              link with the error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, to_dense_batch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "calibration_shift"))
sys.path.insert(0, str(EVAL / "generalization_v3"))
import run_shift  # noqa: E402  (also puts gsv3 / genstudy on sys.path)
from gsv3 import paths  # noqa: E402
from gsv3.devices import load_devices  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

paths.ensure_imports()
from genstudy.data import DEVICE_NAMES  # noqa: E402

OUT = HERE / "results"
CS = EVAL / "calibration_shift" / "results"
QPY = EVAL / "compile_check" / "results" / "layouts_all" / "qpy"
LOFO = {"v4_phys": EVAL / "generalization_v4/results/v4_phys/leave_one_family_out/qaoa/model.pth",
        "v5_sinkhorn": EVAL / "generalization_v5/results/v5_sinkhorn/leave_one_family_out/qaoa/model.pth"}
CONTROL = {"v4_phys": run_shift._control("generalization_v4", "v4_phys", 5),
           "v5_sinkhorn": run_shift._control("generalization_v5", "v5_sinkhorn", 5)}
KIND = {"v4_phys": "phys", "v5_sinkhorn": "sinkhorn"}
PARTS = ("1q", "readout", "2q")


def r2(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return float(1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12))


def load_model(name, path, devices, params):
    m = build_model(KIND[name], params, devices, torch.device("cpu"))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


# -- forward passes that expose the parts of log F (copies of the model code) ----------

def v4_parts(model, circuits, devices, uniform=None):
    """PhysicsHeadPredictor.forward, returning per-circuit parts (B, D, 3) and attention.

    ``uniform="errors"``: the physics-head errors use a uniform attention (device mean),
    the context fed to the count MLP keeps the learned attention; ``uniform="all"``:
    both (the context becomes the attention output for uniform weights,
    ``out_proj(mean_p v_proj(phys_p))``, identical for every logical qubit).
    """
    g = model.gate_proj(model.circuit_encoder(circuits))
    q, gq_glob, valid = model._qubits(circuits, g)
    phys, mask, dev_glob = model._devices(devices)
    n_dev = phys.size(0)
    query = q.unsqueeze(0).expand(n_dev, -1, -1)
    ctx, attn = model.attn(query, phys, phys, key_padding_mask=~mask,
                           need_weights=True, average_attn_weights=True)
    ctx = ctx.transpose(0, 1)
    if uniform:
        attn = mask.float().unsqueeze(1).expand_as(attn) / mask.sum(1).view(-1, 1, 1).float()
    if uniform == "all":
        d = phys.size(-1)
        v = F.linear(phys, model.attn.in_proj_weight[2 * d:], model.attn.in_proj_bias[2 * d:])
        v_mean = (v * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)     # (D, d)
        ctx = model.attn.out_proj(v_mean).unsqueeze(0).expand(q.size(0), -1, -1)
    eps_dense, _ = to_dense_batch(devices.phys, devices.batch)
    q_eps = torch.einsum("dqp,dpk->qdk", attn, eps_dense)
    idx = gq_glob.clamp(min=0)
    vmask = valid.unsqueeze(-1).unsqueeze(-1)
    count = valid.sum(1).clamp(min=1).view(-1, 1, 1).float()
    slots = ctx[idx]
    c_mean = (slots * vmask).sum(1) / count
    c_max = slots.masked_fill(~vmask, -1e4).amax(1)
    c_min = slots.masked_fill(~vmask, 1e4).amin(1)
    c_range = torch.where(valid.sum(1).view(-1, 1, 1) > 1, c_max - c_min, torch.zeros_like(c_max))
    g_eps = (q_eps[idx] * vmask).sum(1) / count
    g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
    d_e = dev_glob.unsqueeze(0).expand(g.size(0), -1, -1)
    z = torch.cat([g_e, c_mean, c_range, d_e, g_e * c_mean], dim=-1)
    n_ops = F.softplus(model.cost(z))
    part = n_ops * g_eps * model.log_scale.exp()                       # (G, D, 3)
    n_circ = circuits.n_qubits.view(-1).size(0)
    return scatter(part, circuits.batch, dim=0, dim_size=n_circ, reduce="sum"), attn


def v5_parts(model, circuits, devices, uniform=False):
    """SinkhornPlacementPredictor.forward, returning parts (B, D, 3) and the placement."""
    g = model.gate_proj(model.circuit_encoder(circuits))
    q, gq_glob, valid = model._qubits(circuits, g)
    phys, mask, dev_glob = model._devices(devices)
    n_dev, p_max = phys.size(0), phys.size(1)
    n_q = circuits.n_qubits.view(-1).long()
    n_circ = n_q.size(0)
    q_batch = torch.repeat_interleave(torch.arange(n_circ), n_q)
    q_dense, q_mask = to_dense_batch(q, q_batch)
    scores = torch.einsum("bqk,dpk->bdqp", model.w_q(q_dense), model.w_p(phys)) / phys.size(-1) ** 0.5
    assign = model._sinkhorn(scores, n_q, mask.sum(1))
    if uniform:
        n_phys = mask.sum(1).float()                                    # (D,)
        assign = (q_mask.view(n_circ, 1, -1, 1) & mask.view(1, n_dev, 1, p_max)).float() \
            / n_phys.view(1, n_dev, 1, 1)
    ctx = torch.einsum("bdqp,dpk->bdqk", assign, phys)
    eps_dense, _ = to_dense_batch(devices.phys, devices.batch)
    q_eps = torch.einsum("bdqp,dpk->bdqk", assign, eps_dense)
    a_h = torch.einsum("bdqp,dpr->bdqr", assign, model.hops[:, :p_max, :p_max])
    gb = circuits.batch
    local = circuits.gate_qubits.clamp(min=0)
    vmask = valid.unsqueeze(-1).unsqueeze(-1)
    count = valid.sum(1).clamp(min=1).view(-1, 1, 1).float()
    slots = ctx[gb.unsqueeze(1), :, local]
    c_mean = (slots * vmask).sum(1) / count
    c_max = slots.masked_fill(~vmask, -1e4).amax(1)
    c_min = slots.masked_fill(~vmask, 1e4).amin(1)
    multi = valid.sum(1).view(-1, 1, 1) > 1
    c_range = torch.where(multi, c_max - c_min, torch.zeros_like(c_max))
    g_eps = (q_eps[gb.unsqueeze(1), :, local] * vmask).sum(1) / count
    dist = torch.zeros(g.size(0), n_dev)
    n_pairs = torch.zeros(g.size(0), 1)
    for a, b_ in ((0, 1), (0, 2), (1, 2)):
        m = (valid[:, a] & valid[:, b_]).float().unsqueeze(-1)
        dist = dist + m * (a_h[gb, :, local[:, a]] * assign[gb, :, local[:, b_]]).sum(-1)
        n_pairs = n_pairs + m
    dist = dist / n_pairs.clamp(min=1)
    g_e = g.unsqueeze(1).expand(-1, n_dev, -1)
    d_e = dev_glob.unsqueeze(0).expand(g.size(0), -1, -1)
    z = torch.cat([g_e, c_mean, c_range, d_e, g_e * c_mean,
                   dist.unsqueeze(-1), torch.log1p(dist).unsqueeze(-1)], dim=-1)
    n_ops = F.softplus(model.cost(z))
    part = n_ops * g_eps * model.log_scale.exp()
    return scatter(part, gb, dim=0, dim_size=n_circ, reduce="sum"), assign


def parts_of(name, model, graphs, devices, uniform=False, check=True):
    out, extra = [], []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=16):
            if name == "v4_phys":
                p, a = v4_parts(model, b, devices, uniform=uniform or None)
            else:
                p, a = v5_parts(model, b, devices, uniform=bool(uniform))
            if check and not uniform:
                ref = model(b, devices)
                assert torch.allclose(-p.sum(-1), ref, atol=1e-4), "parts do not add up to the model output"
            out.append(p)
            extra.append(a)
    return torch.cat(out).numpy(), extra


def fid(parts):
    return np.exp(np.minimum(-parts.sum(-1), 0.0))


# -- calibration-shift inputs -----------------------------------------------------------

def cs_inputs(params):
    sample = json.loads((CS / "sample.json").read_text())
    truth = json.loads((CS / "truth.json").read_text())
    raw_all = json.loads((CS / "variants_raw.json").read_text())
    keys = list(raw_all)
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt",
                                                  weights_only=False)}
    graphs = [data[n] for n in sample]
    full = run_shift._variant_batch(raw_all, keys, params["lap_pe"])
    return sample, truth, keys, graphs, full


def hybrid(full, keys, mode):
    """Copy of the variant batch with one channel taken from the device's 'orig' variant."""
    from torch_geometric.data import Batch
    gs = full.to_data_list()
    out = []
    for k, g in zip(keys, gs):
        o = gs[keys.index(k.split("/")[0] + "/orig")]
        h = g.clone()
        if mode == "features_only":          # variant features, orig physics-head errors
            h.phys = o.phys.clone()
        elif mode == "errors_only":          # orig features, variant physics-head errors
            h.x, h.edge_attr = o.x.clone(), o.edge_attr.clone()
        out.append(h)
    return Batch.from_data_list(out)


def table_r2(pred, truth, sample, keys):
    res = {}
    for j, k in enumerate(keys):
        ok = [i for i, n in enumerate(sample) if truth[n][k][0] is not None]
        res[k] = r2(pred[ok, j], [truth[sample[i]][k][0] for i in ok])
    return res


def cmd_channels(params, report):
    sample, truth, keys, graphs, full = cs_inputs(params)
    modes = {"full": full, "features_only": hybrid(full, keys, "features_only"),
             "errors_only": hybrid(full, keys, "errors_only")}
    res = {}
    for name, path in CONTROL.items():
        res[name] = {}
        for mode, dev in modes.items():
            model = load_model(name, path, dev, params)
            parts, _ = parts_of(name, model, graphs, dev, check=(mode == "full"))
            res[name][mode] = table_r2(fid(parts), truth, sample, keys)
    report("## 1. Channel swap (calibration shift, control models seed 5): R² vs the variant's truth")
    report("full = variant everywhere; features_only = variant only in placement/MLP features;")
    report("errors_only = variant only in the physics-head errors.  orig rows are identical by construction.")
    for name in res:
        report(f"\n### {name}")
        report(f"{'variant':22}" + "".join(f"{m:>15}" for m in modes))
        for k in keys:
            if k.endswith("/orig"):
                continue
            report(f"{k:22}" + "".join(f"{res[name][m][k]:>15.3f}" for m in modes))
    return res


def cmd_uniform(params, report):
    sample, truth, keys, graphs, full = cs_inputs(params)
    model = load_model("v5_sinkhorn", CONTROL["v5_sinkhorn"], full, params)
    p_learn, assigns = parts_of("v5_sinkhorn", model, graphs, full)
    p_unif, _ = parts_of("v5_sinkhorn", model, graphs, full, uniform=True)
    fl, fu = fid(p_learn), fid(p_unif)
    rl, ru = table_r2(fl, truth, sample, keys), table_r2(fu, truth, sample, keys)
    maxrow = float(np.mean([a.max(-1).values[a.sum(-1) > 0].mean().item() for a in assigns]))
    report("\n## 2. v5 with a uniform placement")
    report(f"learned placement: mean max row probability {maxrow:.3f} (uniform would be ~1/26)")
    report(f"calibration sample, all variants: mean |F_learned - F_uniform| = {np.abs(fl - fu).mean():.4f}, "
           f"corr = {np.corrcoef(fl.ravel(), fu.ravel())[0, 1]:.4f}")
    report(f"{'variant':22}{'R² learned':>12}{'R² uniform':>12}")
    for k in keys:
        report(f"{k:22}{rl[k]:>12.3f}{ru[k]:>12.3f}")
    # LOFO qaoa (generalisation): learned vs uniform placement
    devices = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
    names, graphs_q, T = qaoa_test()
    m = load_model("v5_sinkhorn", LOFO["v5_sinkhorn"], devices, params)
    a, _ = parts_of("v5_sinkhorn", m, graphs_q, devices)
    u, _ = parts_of("v5_sinkhorn", m, graphs_q, devices, uniform=True)
    report(f"LOFO qaoa test: R² learned {r2(fid(a).ravel(), T.ravel()):.3f}  uniform {r2(fid(u).ravel(), T.ravel()):.3f}"
           f"   bias learned {np.mean(fid(a) - T):+.3f}  uniform {np.mean(fid(u) - T):+.3f}")


def qaoa_test():
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    graphs = [d for d in data if d.circuit_name.startswith("qaoa/")]
    return [d.circuit_name for d in graphs], graphs, np.array([d.y.view(-1).numpy() for d in graphs])


def true_parts(names):
    """(N, D, 3) true -log(1-e) sums per kind from the stored compiled circuits; NaN if missing."""
    from qiskit import qpy
    sys.path.insert(0, str(Path.home() / "compileCircuits"))
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    targets = [EQE1TopBackend().target, EQE1BottomBackend().target, QExa20Backend().target]
    kind = {"r": 0, "measure": 1, "cz": 2}
    out = np.full((len(names), 3, 3), np.nan)
    counts = np.full((len(names), 3, 3), np.nan)
    for i, n in enumerate(names):
        f = QPY / f"{n}.qpy"
        if not f.is_file():
            continue
        with f.open("rb") as fh:
            circuits = qpy.load(fh)
        for d, (tc, tg) in enumerate(zip(circuits, targets)):
            s, c = np.zeros(3), np.zeros(3)
            for inst in tc.data:
                k = kind.get(inst.operation.name)
                if k is None:
                    continue
                qargs = tuple(tc.find_bit(q).index for q in inst.qubits)
                s[k] += -np.log(max(1 - tg[inst.operation.name][qargs].error, 1e-12))
                c[k] += 1
            out[i, d], counts[i, d] = s, c
    return out, counts


def cmd_parts(params, report):
    devices = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
    names, graphs, T = qaoa_test()
    tp, tc = true_parts(names)
    ok = ~np.isnan(tp[:, 0, 0])
    nq = np.array([int(re.search(r"_q(\d+)_", n).group(1)) for n in names])
    report("\n## 3. LOFO qaoa: parts of -log F, model vs truth from the compiled circuits")
    report(f"{ok.sum()}/{len(names)} circuits with compiled QPY.  Values: mean over circuits x devices.")
    report("truth: compilation stored in qpy/ (seed 0), so it differs slightly from the label compile.")
    preds = {}
    for name, path in LOFO.items():
        model = load_model(name, path, devices, params)
        preds[name], _ = parts_of(name, model, graphs, devices)
    for lo, hi in ((2, 4), (5, 8), (9, 13), (14, 20)):
        sel = ok & (nq >= lo) & (nq <= hi)
        report(f"\nq{lo}-{hi} (n={sel.sum()})   true F {np.exp(-tp[sel].sum(-1)).mean():.3f}   "
               f"true counts r/meas/cz = " + "/".join(f"{tc[sel][..., k].mean():.0f}" for k in range(3)))
        report(f"{'':12}" + "".join(f"{p:>10}" for p in PARTS) + f"{'total':>10}{'pred F':>9}{'bias F':>9}")
        report(f"{'truth':12}" + "".join(f"{tp[sel][..., k].mean():>10.3f}" for k in range(3))
               + f"{tp[sel].sum(-1).mean():>10.3f}")
        for name, p in preds.items():
            report(f"{name:12}" + "".join(f"{p[sel][..., k].mean():>10.3f}" for k in range(3))
                   + f"{p[sel].sum(-1).mean():>10.3f}{fid(p[sel]).mean():>9.3f}{np.mean(fid(p[sel]) - T[sel]):>+9.3f}")


def cmd_attention(params, report):
    sample, truth, keys, graphs, full = cs_inputs(params)
    model = load_model("v4_phys", CONTROL["v4_phys"], full, params)
    parts, attns = parts_of("v4_phys", model, graphs, full)
    f = fid(parts)
    sharp, shared = defaultdict(list), defaultdict(list)
    for a in attns:                                        # (D, Q_batch, P)
        top = a.argmax(-1)
        for j in range(a.size(0)):
            sharp[keys[j]].append(a[j].max(-1).values.mean().item())
            shared[keys[j]].append(1 - top[j].unique().numel() / top.size(1))
    report("\n## 4. v4 attention: sharpness (mean max weight per logical qubit), share of logical")
    report("qubits whose top physical qubit is used by another one (over a batch of 16 circuits), R²")
    report(f"{'variant':22}{'max weight':>11}{'shared top':>11}{'R²':>8}")
    rr = table_r2(f, truth, sample, keys)
    for k in keys:
        report(f"{k:22}{np.mean(sharp[k]):>11.3f}{np.mean(shared[k]):>11.3f}{rr[k]:>8.3f}")


SPLIT_MODEL = {  # split -> (model results dir relative to EVAL, experiment dir)
    "qaoa": "leave_one_family_out/qaoa", "qnn": "leave_one_family_out/qnn",
    "variational": "leave_one_group_out/variational"}
RUN_DIR = {"v4_phys": "generalization_v4/results/v4_phys",
           "v5_sinkhorn": "generalization_v5/results/v5_sinkhorn"}
VARIANTS_U = {"v4_phys": [("learned", None), ("uniform errors", "errors"), ("uniform all", "all")],
              "v5_sinkhorn": [("learned", None), ("uniform", True)]}


def split_test(split):
    from gsv3.groups import GROUPS
    fams = set(GROUPS[split]) if split in GROUPS else {split}
    data = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    graphs = [d for d in data if d.circuit_name.split("/")[0] in fams]
    return ([d.circuit_name for d in graphs], graphs, np.array([d.y.view(-1).numpy() for d in graphs]))


def cmd_uniform_all(params, report):
    """Learned vs uniform placement for v4 and v5: calibration shift + held-out splits."""
    sample, truth, keys, graphs, full = cs_inputs(params)
    devices = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
    tests = {s: split_test(s) for s in SPLIT_MODEL}
    report("## 5. Learned vs uniform placement, v4 and v5 (seed 5)")
    report("calibration: control model on the calibration-shift sample, R² per variant summarised as")
    report("mean / worst over the 18 device variants; held-out splits: R² and bias (mean F_pred - F_true).")
    head = f"{'model':12}{'placement':16}{'calib mean':>11}{'calib worst':>12}"
    head += "".join(f"{s + ' R2':>16}{'bias':>8}" for s in SPLIT_MODEL) + f"{'var/qnn R2':>12}"
    report(head)
    for name in ("v4_phys", "v5_sinkhorn"):
        for label, u in VARIANTS_U[name]:
            model = load_model(name, CONTROL[name], full, params)
            p, _ = parts_of(name, model, graphs, full, uniform=u)
            rr = list(table_r2(fid(p), truth, sample, keys).values())
            row = f"{name:12}{label:16}{np.mean(rr):>11.3f}{np.min(rr):>12.3f}"
            for s, rel in SPLIT_MODEL.items():
                names, g, T = tests[s]
                m = load_model(name, EVAL / RUN_DIR[name] / rel / "model.pth", devices, params)
                f = fid(parts_of(name, m, g, devices, uniform=u)[0])
                row += f"{r2(f.ravel(), T.ravel()):>16.3f}{np.mean(f - T):>+8.3f}"
                if s == "variational":
                    q = [i for i, n in enumerate(names) if n.startswith("qnn/")]
                    sub = r2(f[q].ravel(), T[q].ravel())
            report(row + f"{sub:>12.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["all", "channels", "uniform", "parts", "attention", "uniform_all"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    params = load_hparams()
    lines = []

    def report(s=""):
        print(s, flush=True)
        lines.append(s)

    todo = ["channels", "uniform", "parts", "attention", "uniform_all"] if args.what == "all" else [args.what]
    for w in todo:
        globals()[f"cmd_{w}"](params, report)
    (OUT / f"report_{args.what}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
