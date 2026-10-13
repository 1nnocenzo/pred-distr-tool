#!/usr/bin/env python3
"""Stage-1 dataset: per circuit and device, Sabre's SWAPs and routed schedule on the model's gate graph + a router prior.

Inputs: the model's circuit graphs (dataset + synthetic_routing), compile_check/results/sabre_labels (sabre_labels.py),
the device graphs.  Per circuit, tensors over its N gate nodes and the D = 3 devices:
  s       (D, N, 3) SWAPs Sabre puts on the wire of each operand slot just before the gate (target)
  wait    (D, N)    extra wait of the gate in Sabre's routed schedule beyond  max over its wires of
                    (previous gate's end + its SWAPs × T_SWAP)  — the synchronisation a SWAP pays for the partner wire
                    it waits for (target; teacher-forced: previous ends taken from the routed schedule)
  m       (D, N)    gate matched in the routed circuit (multi-qubit gates and measurements)
  prior_s (D, N, 3) the same SWAPs from our own greedy router (esp_router-style, no compiler): input feature
  hops    (D, N)    operand distance − 1 on the chip at that gate under the greedy router's current placement: input
  rf, final, replay, prior_dur (D)   routing-free flag, label duration, replayed Sabre schedule, prior router duration
    python prep.py --out results/stage1.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL / "esp"))
sys.path.insert(0, str(EVAL / "generalization_v3"))
from gsv3 import paths  # noqa: E402
paths.ensure_imports()
from gsv3 import fast_encoding as fe  # noqa: E402
from gsv3.devices import variant_batch  # noqa: E402
from gsv3.model import GATE2IDX_MEASURE, load_hparams  # noqa: E402
from esp_router import chips, greedy_place, path  # noqa: E402

DEVS = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LAB = EVAL / "compile_check/results/sabre_labels"
SYN = EVAL / "synthetic_routing/results/synth_graphs.pt"
T1, TM, TCZ = 0.042, 15.0, 0.130
T_SWAP = 3 * TCZ + 2 * T1


def node_times(g, tr, tc):
    U = len(fe.UNIQUE)
    oh = g.x[:, :U].numpy()
    meas = g.x[:, GATE2IDX_MEASURE].numpy() > 0.5
    gq = g.gate_qubits.numpy()
    n_op = np.clip((gq >= 0).sum(1), 1, None)
    t = np.where(meas, TM, (oh @ tr) * T1 / n_op + (oh @ tc) * TCZ)
    return [[int(x) for x in row if x >= 0] for row in gq], t, meas


def prior_route(wires, t, meas, n_log, chip):
    """Greedy router (esp_router.route_schedule) with per-gate attribution: (s (N, 3), hops (N,), duration)."""
    n, adj, H, par = chip
    pairs: dict = {}
    for qs in wires:
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                k = (min(qs[i], qs[j]), max(qs[i], qs[j])); pairs[k] = pairs.get(k, 0) + 1
    pos = greedy_place(pairs, n_log, H, adj)
    at = {p: q for q, p in pos.items()}
    avail = [0.0] * n
    N = len(wires)
    s = np.zeros((N, 3), np.float32)
    hops = np.zeros(N, np.float32)
    pending = defaultdict(int)
    for gi, qs in enumerate(wires):
        if not qs:
            continue
        if len(qs) >= 2:
            hops[gi] = max(H[pos[a]][pos[b]] for a in qs for b in qs if a != b) - 1
            anchor = pos[qs[0]]
            for o in qs[1:]:
                po = pos[o]
                if H[po][anchor] <= 1:
                    continue
                for u, v in zip(path(par, po, anchor)[:-2], path(par, po, anchor)[1:-1]):
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
        if len(qs) >= 2 or meas[gi]:
            for k, w in enumerate(qs):
                if pending[w]:
                    s[gi, k] += pending[w]; pending[w] = 0
        ps = [pos[q] for q in qs]
        st = max(avail[p] for p in ps) + t[gi]
        for p in ps:
            avail[p] = st
    return s, hops, max(avail)


def wait_targets(wires, t, s, t_node):
    """Teacher-forced wait per matched gate (see module doc); unmatched gates follow the plain rule."""
    N = len(wires)
    end = {}
    last = defaultdict(float)
    wait = np.zeros(N, np.float32)
    for gi, qs in enumerate(wires):
        if not qs:
            continue
        ready = max(last[q] + s[gi, k] * T_SWAP for k, q in enumerate(qs))
        if gi in t_node:
            st, en = t_node[gi]
            wait[gi] = st - ready
            e = en
        else:
            e = ready + t[gi]
        for q in qs:
            last[q] = e
        end[gi] = e
    return wait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results/stage1.pt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    params = load_hparams()
    raw = torch.load(paths.V2_DEVICE_GRAPHS, weights_only=False)["raw"]
    chs = chips(variant_batch(raw, list(DEVS), params["lap_pe"]))
    tab = json.loads((EVAL / "esp/results/decomp_table.json").read_text())
    U = list(fe.UNIQUE)
    tr = np.array([tab.get(u, {"r": 0})["r"] for u in U], float)
    tc = np.array([tab.get(u, {"cz": 0})["cz"] for u in U], float)
    recs = {}
    for f in sorted(LAB.glob("task*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    recs[r["circuit"]] = r
    graphs = torch.load(paths.DATA_DIR / "graph_dataset_expected_fidelity.pt", weights_only=False)
    graphs += torch.load(SYN, weights_only=False)
    out, skip = {}, defaultdict(int)
    for gi_, g in enumerate(graphs):
        name = g.circuit_name
        rec = recs.get(name)
        if rec is None:
            skip["no label"] += 1; continue
        N = g.num_nodes
        if rec["n_nodes"] != N:
            skip["node count"] += 1; continue
        wires, t, meas = node_times(g, tr, tc)
        n_log = int(g.n_qubits)
        D = len(DEVS)
        item = {"s": np.zeros((D, N, 3), np.float32), "wait": np.zeros((D, N), np.float32),
                "m": np.zeros((D, N), bool), "prior_s": np.zeros((D, N, 3), np.float32), "hops": np.zeros((D, N), np.float32),
                "rf": np.zeros(D, bool), "final": np.zeros(D, np.float32), "replay": np.full(D, np.nan, np.float32),
                "prior_dur": np.zeros(D, np.float32), "busy": np.zeros((D, n_log), np.float32), "aligned": np.ones(D, bool)}
        rows = {r["device"]: r for r in rec["rows"]}
        for di, d in enumerate(DEVS):
            r = rows.get(d)
            if r is None:
                skip["missing device"] += 1; continue
            item["rf"][di] = r["routing_free"]
            item["final"][di] = r["final"]
            ps, hp, pdur = prior_route(wires, t, meas, n_log, chs[di])
            item["prior_s"][di], item["hops"][di], item["prior_dur"][di] = ps, hp, pdur
            if r["routing_free"] or "swaps" not in r:
                continue
            for node, slot, n in r["swaps"]:
                item["s"][di, node, slot] = n
            tn = {k: (a, b) for k, a, b in r["t_node"]}
            item["m"][di, list(tn)] = True
            item["wait"][di] = wait_targets(wires, t, item["s"][di], tn)
            item["replay"][di] = r["replay"]
            item["busy"][di, : len(r["busy"])] = r["busy"]
            item["aligned"][di] = r["aligned"]
        out[name] = {k: torch.from_numpy(v) for k, v in item.items()}
        if gi_ % 2000 == 0:
            print(gi_, len(graphs), dict(skip), flush=True)
        if args.limit and len(out) >= args.limit:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print("saved", len(out), "circuits;", dict(skip), "->", args.out)


if __name__ == "__main__":
    main()
