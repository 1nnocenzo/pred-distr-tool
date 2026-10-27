#!/usr/bin/env python3
"""Model dataset: one PyG item per labelled circuit, every input and target the model needs.

Inputs   x, edge_index, gate_qubits, n_qubits (encoding); t (N,) logical gate time µs; router features (router.py):
         rf_pred, route (1, D), prior_s, hops, wf, prev, lvl.
Targets  (original calibration, (1, D) unless noted; ``ok`` masks the known ones)
         logF, decay, dur µs, counts (1, D, 3), layout (n_qubits, D) device-graph node, rf (VF2Layout's verdict)
         route: s (N, D, 3) Sabre SWAPs per operand slot, wait (N, D) teacher-forced extra wait of each matched gate
         (/T_SWAP in the loss), m (N, D) matched, routed (1, D), replay (1, D) Sabre's routed schedule µs,
         c2 (1, D) = log(dur / base) with base = replay (routed) or the logical schedule (routing-free)
         n_extra (1, D) active physical qubits of the compiled circuit beyond the logical qubits with a gate (the ESP's
         idle runs over all active physical qubits; routing may activate extra ones)
         variants: logF_v, decay_v (1, K, D) on the K calibration variants of labels.pt (``calibs``)
    python -m esptime.dataset [--jobs 8]          # work/labels.pt + work/devices.json -> work/dataset.pt
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time
import zipfile
import zlib
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.data import Data

from . import router
from .config import CIRCUITS_ZIP, DEVICES, T_SWAP, VARIATIONAL, WORK, decomp_table
from .devices import chips
from .encoding import MEASURE, encode, gate_times, to_basis
from .labels import circuit_sources, read_qasm

D = len(DEVICES)


def schedule_np(wires, t, s=None, wait=None) -> float:
    """Sequential ASAP on logical wires: SWAPs before a gate occupy its wires, plus the gate's own wait."""
    last: dict = {}
    end = 0.0
    for i, qs in enumerate(wires):
        if not qs:
            continue
        ready = max(last.get(q, 0.0) + (s[i, k] * T_SWAP if s is not None else 0.0) for k, q in enumerate(qs))
        st = ready + (wait[i] if wait is not None else 0.0)
        e = max(st, max(last.get(q, 0.0) for q in qs)) + t[i]
        for q in qs:
            last[q] = e
        end = max(end, e)
    return end


def wait_targets(wires, t, s, t_node) -> np.ndarray:
    """Per matched gate: start in Sabre's routed schedule − (max over its wires of previous end + its SWAPs), with
    the previous ends taken from the routed schedule (teacher forcing).  Unmatched gates follow the plain rule."""
    last = defaultdict(float)
    wait = np.zeros(len(wires), np.float32)
    for gi, qs in enumerate(wires):
        if not qs:
            continue
        ready = max(last[q] + s[gi, k] * T_SWAP for k, q in enumerate(qs))
        if gi in t_node:
            st, e = t_node[gi]
            wait[gi] = st - ready
        else:
            e = ready + t[gi]
        for q in qs:
            last[q] = e
    return wait


def _vec(rec: dict, key: str, default=float("nan")) -> torch.Tensor:
    return torch.tensor([[default if rec.get(d, {}).get(key) is None else float(rec[d][key]) for d in DEVICES]])


def build_item(name: str, qb, lab: dict, calibs: list[str], chs, table) -> Data:
    x, ei, gq, nq = encode(qb)
    if x.size(0) != lab["n_nodes"]:
        raise ValueError(f"node count {x.size(0)} != labels {lab['n_nodes']}")
    t = gate_times(x, gq, table)
    meas = x[:, MEASURE].numpy() > 0.5
    gqn = gq.numpy()
    wires = [[int(q) for q in row if q >= 0] for row in gqn]
    f = router.features(gqn, nq, t, meas, chs)
    N = x.size(0)
    o = lab["calib"]["orig"]
    logf, dec, dur = _vec(o, "logF"), _vec(o, "decay"), _vec(o, "dur")
    ok = torch.isfinite(logf) & torch.isfinite(dec) & (dec <= 0) & torch.isfinite(dur)
    layout = torch.full((nq, D), -1, dtype=torch.long)
    counts = torch.full((1, D, 3), -1.0)
    rf = torch.zeros(1, D, dtype=torch.bool)
    s = np.zeros((N, D, 3), np.float32)
    wait = np.zeros((N, D), np.float32)
    m = np.zeros((N, D), bool)
    routed = torch.zeros(1, D, dtype=torch.bool)
    n_extra = torch.zeros(1, D)
    n_log_act = len({q for w in wires for q in w})
    replay = torch.full((1, D), float("nan"))
    logical = schedule_np(wires, t)
    for di, d in enumerate(DEVICES):
        r = o.get(d)
        if r is None:
            continue
        layout[: len(r["layout"]), di] = torch.tensor(r["layout"][:nq])
        counts[0, di] = torch.tensor(r["counts"], dtype=torch.float)
        rf[0, di] = bool(r["rf"])
        if "busy" in r:
            n_extra[0, di] = max(sum(n >= 0 for n, _ in r["busy"]) - n_log_act, 0)
        if "swaps" in r and r.get("aligned"):
            for node, slot, n in r["swaps"]:
                s[node, di, slot] = n
            tn = {k: (a, b) for k, a, b in r["t_node"]}
            m[list(tn), di] = True
            wait[:, di] = wait_targets(wires, t, s[:, di], tn)
            replay[0, di] = r["replay"]
            routed[0, di] = True
    base = torch.where(routed, replay, torch.full_like(replay, logical))
    ok2 = (routed | rf) & torch.isfinite(dur) & (dur > 0)
    c2 = torch.where(ok2, torch.log(dur.clamp(min=1e-3) / base.nan_to_num(1.0).clamp(min=1e-3)), torch.zeros_like(dur))
    var = [c for c in calibs if c != "orig"]
    logf_v = torch.full((1, len(var), D), float("nan"))
    dec_v = torch.full_like(logf_v, float("nan"))
    for k, c in enumerate(var):
        rv = lab["calib"].get(c)
        if rv is not None:
            logf_v[0, k], dec_v[0, k] = _vec(rv, "logF")[0], _vec(rv, "decay")[0]
    return Data(x=x, edge_index=ei, gate_qubits=gq, n_qubits=nq, num_nodes=N, name=name,
                t=torch.from_numpy(t), wf=torch.from_numpy(f["wf"]), prev=torch.from_numpy(f["prev"]),
                lvl=torch.from_numpy(f["lvl"]), prior_s=torch.from_numpy(f["prior_s"]).permute(1, 0, 2).contiguous(),
                hops=torch.from_numpy(f["hops"]).t().contiguous(), rf_pred=torch.from_numpy(f["rf"]).view(1, -1),
                route=torch.from_numpy(f["route"]).view(1, -1),
                logF=logf.nan_to_num(0.0), decay=dec.nan_to_num(0.0), dur=dur.nan_to_num(0.0), ok=ok,
                counts=counts, layout=layout, rf=rf,
                s=torch.from_numpy(s), wait=torch.from_numpy(wait), m=torch.from_numpy(m),
                routed=routed, replay=replay, c2=c2, ok2=ok2,
                n_extra=n_extra, logF_v=logf_v.nan_to_num(0.0), decay_v=dec_v.nan_to_num(0.0),
                ok_v=torch.isfinite(logf_v) & torch.isfinite(dec_v) & (dec_v <= 0))


_CTX: dict = {}


def _init(raws, calibs):
    _CTX.update(chs=chips(raws), calibs=calibs, table=decomp_table(), zip=zipfile.ZipFile(CIRCUITS_ZIP))


def _work(job):
    name, key, lab = job
    try:
        qb = to_basis(read_qasm(key, None if name.startswith("synth/") else _CTX["zip"]))
        buf = io.BytesIO()  # bytes, not tensors: torch would pass each tensor as a file descriptor ("received 0 items of ancdata")
        torch.save(build_item(name, qb, lab, _CTX["calibs"], _CTX["chs"], _CTX["table"]), buf)
        return buf.getvalue(), None
    except Exception as e:  # noqa: BLE001
        return None, f"{name}: {type(e).__name__}: {e}"[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(WORK / "dataset.pt"))
    args = ap.parse_args()
    labels = torch.load(WORK / "labels.pt", weights_only=False)
    raws = json.loads((WORK / "devices.json").read_text())
    calibs = sorted({c for r in labels.values() for c in r["calib"]}, key=lambda c: (c != "orig", len(c), c))
    jobs = [(n, k, labels[n]) for src in ("mqt", "synth") for n, k in circuit_sources(src) if n in labels]
    t0 = time.time()
    items, errs = [], []
    with Pool(args.jobs, initializer=_init, initargs=(raws["orig"], calibs)) as pool:
        for i, (it, err) in enumerate(pool.imap(_work, jobs, chunksize=16)):
            items.append(torch.load(io.BytesIO(it), weights_only=False)) if it is not None else errs.append(err)
            if i % 2000 == 0:
                print(f"[{time.time() - t0:5.0f} s] {i}/{len(jobs)}", flush=True)
    torch.save({"items": items, "calibs": calibs, "devices": raws}, args.out)
    print(f"{len(items)} items, {len(errs)} failed {errs[:3]}, calibrations {calibs} -> {args.out}")


# ------------------------------------------------------------------------------------------------ splits
def family(name: str) -> str:
    return name.split("/")[0]


def split_names(split: str, names: list[str]) -> tuple[list[str], list[str]]:
    """(train, test).  control: 20% of the real circuits by name hash; lofo:<family>; logo:variational (vqe_*, qnn);
    loso:<s1,s2> (development): synthetic structures held out, qaoa and the variational group out of training.
    Synthetic circuits are only ever on the training side."""
    real = [n for n in names if family(n) != "synth"]
    if split == "control":
        test = [n for n in real if zlib.crc32(n.encode()) % 5 == 0]
        held = set(test)
        return [n for n in names if n not in held], test
    if split.startswith("lofo:") or split == "logo:variational":
        fams = set(VARIATIONAL) if split == "logo:variational" else {split[5:]}
        return [n for n in names if family(n) not in fams], [n for n in real if family(n) in fams]
    if split.startswith("loso:"):
        from .synth import struct_of
        structs = set(split[5:].split(","))
        test = [n for n in names if family(n) == "synth" and struct_of(n) in structs]
        out = set(VARIATIONAL) | {"qaoa"}
        return [n for n in names if family(n) not in out and n not in set(test)], test
    raise ValueError(f"unknown split {split!r}")


if __name__ == "__main__":
    main()
