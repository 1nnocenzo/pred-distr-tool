#!/usr/bin/env python3
"""Why do the models follow local calibration changes so poorly?  Decomposition of the change.

Circuits: the calibration-shift sample (control test circuits). Calibrations: the 8 v8b variants
per device, whose fidelity (best of 3) and compiler layout are known for every circuit, against
the original calibration (best of 3).

TRUE change of log F, split in two:
  fixed   : the circuit compiled on the ORIGINAL calibration (QPY, seed 0), its exact cost
            recomputed with the variant's errors (same layout, same gates);
  relayout: true change − fixed = what the compiler gains or loses by compiling again
            (other qubits, other SWAPs).
MODEL change of log F, split in three by freezing parts of the forward pass:
  errors  : variant errors in the physics head only (placement and counts frozen at original);
  counts  : + the count MLP sees the variant device (placement still frozen);
  place   : + the placement is recomputed on the variant.
Also: bias / slope of predicted vs true change, sign agreement, and how much the placement
moves vs how much the compiler's layout moves.
    python calib_decomp.py [--model v8a_c001] [--seed 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, to_dense_batch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402
from genstudy.data import DEVICE_NAMES  # noqa: E402

V8B = EVAL / "compile_check/results/v8b"
V8B_ORIG = EVAL / "compile_check/results/v8b_orig/v8b_orig_labels.pt"
QPY = EVAL / "compile_check/results/layouts_all/qpy"
SAMPLE = EVAL / "calibration_shift/results/sample.json"
N_AUG = 8
KINDS = {"r": 0, "measure": 1, "cz": 2}


def forward_parts(model, circuits, devices, assign=None, counts_devices=None):
    """Sinkhorn forward returning log F (B, D), the placement and the per-circuit parts.
    ``assign`` freezes the placement; ``counts_devices`` = devices whose features feed the
    placement/context/count MLP (default ``devices``); ``devices.phys`` always gives the errors."""
    cdev = counts_devices if counts_devices is not None else devices
    g = model.gate_proj(model.circuit_encoder(circuits))
    q, gq_glob, valid = model._qubits(circuits, g)
    phys, mask, dev_glob = model._devices(cdev)
    n_dev, p_max = phys.size(0), phys.size(1)
    n_q = circuits.n_qubits.view(-1).long()
    n_circ = n_q.size(0)
    q_batch = torch.repeat_interleave(torch.arange(n_circ), n_q)
    q_dense, q_mask = to_dense_batch(q, q_batch)
    if assign is None:
        scores = torch.einsum("bqk,dpk->bdqp", model.w_q(q_dense), model.w_p(phys)) / phys.size(-1) ** 0.5
        assign = model._sinkhorn(scores, n_q, mask.sum(1))
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
    feats = [g_e, c_mean, c_range, d_e, g_e * c_mean, dist.unsqueeze(-1), torch.log1p(dist).unsqueeze(-1)]
    if getattr(model, "routing_flag", False):
        from gsv3.model import routing_free
        feats.append(routing_free(circuits, cdev, model._rf_cache)[gb].unsqueeze(-1))
    n_ops = F.softplus(model.cost(torch.cat(feats, -1)))
    part = n_ops * g_eps * model.log_scale.exp()
    parts = scatter(part, gb, dim=0, dim_size=n_circ, reduce="sum")
    return -parts.sum(-1), assign


def exact_cost(ops, raw):
    """−log F of a compiled circuit (lists of node ids) under the errors of a raw device graph."""
    nodes = np.array(raw["nodes"], float)
    E = {tuple(sorted(e)): v for e, v in zip(map(tuple, raw["edges"]), raw["edge_feats"])}
    one, meas, cz = ops
    return nodes[one, 0].sum() + nodes[meas, 1].sum() + sum(E[p] for p in cz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v8a_c001")
    ap.add_argument("--kind", default="sinkhorn")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(8)
    params = load_hparams()
    raw_var = json.loads((V8B / "variants_raw.json").read_text())
    raw_orig = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    allraw = {**raw_var, **{d: raw_orig[d] for d in DEVICE_NAMES}}
    dev_o = variant_batch(allraw, list(DEVICE_NAMES), params["lap_pe"])
    dev_v = [variant_batch(allraw, [f"{d}/aug{j}" for d in DEVICE_NAMES], params["lap_pe"]) for j in range(N_AUG)]
    # hybrid: original features (placement, context, counts) with the variant's errors in the head
    root = EVAL / "generalization_v8/results" / (args.model if args.seed == 5 else f"{args.model}_s{args.seed}")
    model = build_model(args.kind, params, dev_o, torch.device("cpu"))
    model.load_state_dict(torch.load(root / "random_control" / f"random_seed{args.seed}" / "model.pth", map_location="cpu"))
    model.eval()

    sample = json.loads(SAMPLE.read_text())
    labels = torch.load(V8B / "v8b_labels.pt", weights_only=False)
    orig = torch.load(V8B_ORIG, weights_only=False)
    data = {d.circuit_name: d for d in torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)}
    names = [n for n in sample if n in labels and n in orig and n in data and (QPY / f"{n}.qpy").is_file()]
    from qiskit import qpy
    sys.path.insert(0, str(Path.home() / "compileCircuits"))
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    bes = [EQE1TopBackend(), EQE1BottomBackend(), QExa20Backend()]
    nodemap = [{q: i for i, q in enumerate(getattr(b, "active_qubits", None) or range(b.num_qubits))} for b in bes]

    rows = []
    with torch.no_grad():
        for start in range(0, len(names), 16):
            chunk = names[start:start + 16]
            b = Batch.from_data_list([data[n] for n in chunk])
            lo, A_o = forward_parts(model, b, dev_o)
            for j in range(N_AUG):
                l_err, _ = forward_parts(model, b, dev_v[j], assign=A_o, counts_devices=dev_o)
                l_cnt, _ = forward_parts(model, b, dev_v[j], assign=A_o)
                l_full, A_v = forward_parts(model, b, dev_v[j])
                for i, n in enumerate(chunk):
                    nq = int(data[n].n_qubits)
                    for d in range(3):
                        fo = float(orig[n]["F"].view(-1)[d]); fv = float(labels[n]["F"][j, d])
                        if fo <= 0.01 or fv <= 0.01:
                            continue
                        po = A_o[i, d, :nq].argmax(-1); pv = A_v[i, d, :nq].argmax(-1)
                        tv = 0.5 * (A_o[i, d, :nq] - A_v[i, d, :nq]).abs().sum(-1).mean()
                        lay_o = orig[n]["layout"][:, 0, d]; lay_v = labels[n]["layout"][:, j, d]
                        rows.append({"n": n, "d": d, "j": j, "true": np.log(fv / fo),
                                     "m_err": float(l_err[i, d] - lo[i, d]), "m_cnt": float(l_cnt[i, d] - lo[i, d]),
                                     "m_full": float(l_full[i, d] - lo[i, d]),
                                     "place_moved": float((po != pv).float().mean()), "place_tv": float(tv),
                                     "comp_moved": float((lay_o != lay_v).float().mean())})
    # exact cost at the original compiled layout
    cache = {}
    for r in rows:
        key = r["n"]
        if key not in cache:
            with (QPY / f"{key}.qpy").open("rb") as fh:
                circs = qpy.load(fh)
            ops = []
            for d, tc in enumerate(circs):
                one, meas, cz = [], [], []
                for inst in tc.data:
                    k = KINDS.get(inst.operation.name)
                    if k is None:
                        continue
                    q = [nodemap[d][tc.find_bit(x).index] for x in inst.qubits]
                    (one if k == 0 else meas if k == 1 else cz).append(q[0] if k < 2 else tuple(sorted(q)))
                ops.append((np.array(one, int), np.array(meas, int), cz))
            cache[key] = ops
        o = cache[key][r["d"]]
        dname = DEVICE_NAMES[r["d"]]
        r["fixed"] = -(exact_cost(o, allraw[f"{dname}/aug{r['j']}"]) - exact_cost(o, allraw[dname]))
        r["relayout"] = r["true"] - r["fixed"]

    a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "n"}
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    slope = lambda x, y: float(np.polyfit(y, x, 1)[0])     # predicted on true
    print(f"model {args.model} seed {args.seed}: {len(rows)} (circuit, device, variant) with F > 0.01\n")
    print("1) What the true change is made of")
    print(f"   std true {a['true'].std():.3f} | fixed-layout part std {a['fixed'].std():.3f}, corr with true {corr(a['fixed'], a['true']):.2f}"
          f" | relayout part std {a['relayout'].std():.3f}, mean {a['relayout'].mean():+.3f} (compiler gains by re-compiling)")
    print("\n2) Model change vs truth (corr / slope pred-on-true / mean pred vs mean true)")
    for k, lab in (("m_err", "errors only (placement, counts frozen)"), ("m_cnt", "+ counts see the variant"), ("m_full", "+ placement recomputed")):
        print(f"   {lab:40} corr true {corr(a[k], a['true']):+.2f}  corr fixed {corr(a[k], a['fixed']):+.2f}"
              f"  corr relayout {corr(a[k], a['relayout']):+.2f}  slope {slope(a[k], a['true']):+.2f}"
              f"  mean {a[k].mean():+.3f} vs {a['true'].mean():+.3f}  std {a[k].std():.3f}")
    part_place = a["m_full"] - a["m_cnt"]
    print(f"   placement contribution alone: std {part_place.std():.3f}, corr relayout {corr(part_place, a['relayout']):+.2f}")
    big = np.abs(a["true"]) > 0.05
    print(f"\n3) Sign (|true dlogF| > 0.05, n={big.sum()}): true worse in {np.mean(a['true'][big] < 0):.0%}, "
          f"model predicts worse in {np.mean(a['m_full'][big] < 0):.0%}, same sign {np.mean(np.sign(a['m_full'][big]) == np.sign(a['true'][big])):.0%}")
    print("\n4) Does the placement move when the compiler's layout moves?")
    print(f"   compiler: share of logical qubits moved {a['comp_moved'].mean():.2f} | model argmax moved {a['place_moved'].mean():.2f}"
          f" | model total-variation shift {a['place_tv'].mean():.3f} | corr(model shift, compiler moved) {corr(a['place_tv'], a['comp_moved']):+.2f}")
    names_v = ["jitter", "jitter", "jitter", "shuffle", "shuffle", "bad qubits", "bad qubits", "bad qubits"]
    print("\n5) By variant type: corr(model, true) / corr(fixed, true) / compiler moved / model moved")
    for t in ("jitter", "shuffle", "bad qubits"):
        s = np.isin(a["j"], [i for i, v in enumerate(names_v) if v == t])
        print(f"   {t:11} {corr(a['m_full'][s], a['true'][s]):+.2f} / {corr(a['fixed'][s], a['true'][s]):+.2f}"
              f" / {a['comp_moved'][s].mean():.2f} / {a['place_moved'][s].mean():.2f}")


if __name__ == "__main__":
    main()
