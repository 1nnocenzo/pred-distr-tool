#!/usr/bin/env python3
"""v8b training data: the dataset circuits compiled on perturbed calibrations.

Per device, ``N_AUG`` = 8 calibration variants (seeds disjoint from the calibration-shift
test variants, which use seeds 11-23 + 100·i):

``aug0``-``aug2``  jitter: every error × log-normal factor, sigma 0.25 (daily drift)
``aug3``-``aug4``  shuffle: errors permuted across qubits / couplers
``aug5``-``aug7``  bad qubits: 2 random qubits get their 1q and readout errors and the
                   errors of all their couplers ×5, plus 1 random coupler ×5

Only the ``r``, ``cz`` and ``measure`` errors change (durations, T1/T2, coupling map do
not).  Each circuit is compiled on every variant with ``seed_transpiler`` 0..K-1 exactly
as the labels (Qiskit preset pass manager, level 2; MQT expected fidelity with the
recursive fallback); the best compilation gives the label, its initial layout (device-graph
node indices) and its native-operation counts (r, measure, cz).  Grover is excluded.

    python v8b_variants.py run --task-id T --n-tasks N --k 3 --out DIR
    python v8b_variants.py raw --out DIR        # device graphs of the variants (for training)
    python v8b_variants.py merge --out DIR      # -> v8b_labels.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(HERE))

DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
GATES = ("r", "cz", "measure")
N_AUG = 8
MAX_ERROR = 0.5


def _set_error(target, gate, qargs, error):
    props = target[gate][qargs]
    new = type(props)(duration=props.duration, error=float(min(max(error, 0.0), MAX_ERROR)))
    target.update_instruction_properties(gate, qargs, new)


def make_aug(backend, j: int, dev_index: int):
    be = copy.deepcopy(backend)
    t = be.target
    rng = np.random.default_rng(10_000 + 1_000 * dev_index + j)
    if j <= 4:
        for gate in GATES:
            qargs = [q for q, p in t[gate].items() if p is not None and p.error is not None]
            errors = np.array([t[gate][q].error for q in qargs], dtype=float)
            new = (errors * np.exp(rng.normal(0.0, 0.25, size=errors.shape)) if j <= 2
                   else rng.permutation(errors))
            for q, e in zip(qargs, new):
                _set_error(t, gate, q, e)
        return be
    qubits = sorted({q[0] for q, p in t["r"].items() if p is not None})
    bad = set(rng.choice(qubits, size=2, replace=False).tolist())
    couplers = [q for q, p in t["cz"].items() if p is not None and p.error is not None]
    bad_edge = couplers[rng.integers(len(couplers))]
    for gate in ("r", "measure"):
        for q, p in t[gate].items():
            if p is not None and p.error is not None and q[0] in bad:
                _set_error(t, gate, q, p.error * 5)
    for q in couplers:
        p = t["cz"][q]
        if q[0] in bad or q[1] in bad or set(q) == set(bad_edge):
            _set_error(t, "cz", q, p.error * 5)
    return be


def variants(orig: bool = False) -> dict:
    """The 8 variants per device, or with ``orig`` the unchanged devices (key ``<dev>/orig``)."""
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    base = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    if orig:
        return {f"{d}/orig": be for d, be in base.items()}
    return {f"{d}/aug{j}": make_aug(be, j, i) for i, (d, be) in enumerate(base.items()) for j in range(N_AUG)}


def run(args) -> None:
    from qiskit import qasm3
    from compile_check import DATA, LABELS, compile_once
    from compile_layouts import _counts, fidelity
    vs = variants(orig=args.orig)
    node = {k: {q: i for i, q in enumerate(getattr(be, "active_qubits", None) or range(be.num_qubits))}
            for k, be in vs.items()}
    names = sorted(n for n in json.loads(LABELS.read_text()) if not n.startswith("grover/"))
    names = [n for n in names if zlib.crc32(n.encode()) % args.n_tasks == args.task_id]
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"task{args.task_id:03d}.jsonl"
    done = set()
    if out_file.is_file():
        done = {json.loads(l)["circuit"] for l in out_file.read_text().splitlines() if l.strip()}
    with out_file.open("a") as fh:
        for i, name in enumerate(names):
            if name in done:
                continue
            rec = {"circuit": name, "F": {}, "layout": {}, "counts": {}}
            try:
                qc = qasm3.loads((DATA / f"{name}.qasm").read_text())
                for key, be in vs.items():
                    best, best_tc = None, None
                    fs = []
                    for s in range(args.k):
                        tc, _ = compile_once(qc, be, 2, s)
                        f = fidelity(tc, be.target)
                        fs.append(f)
                        if best is None or f > best:
                            best, best_tc = f, tc
                    rec["F"][key] = fs
                    lay = best_tc.layout.initial_index_layout(filter_ancillas=True)
                    rec["layout"][key] = [node[key].get(p, -1) for p in lay]
                    rec["counts"][key] = _counts(best_tc).tolist()
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(names)} {name}", flush=True)


def raw(args) -> None:
    sys.path.insert(0, str(HERE.parent / "generalization_v3"))
    import gsv3  # noqa: F401  (puts gsv2 on sys.path)
    from gsv2.devices import _backend_graph
    vs = variants()
    out = {k: dict(zip(("nodes", "edges", "edge_feats"), _backend_graph(be))) for k, be in vs.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "variants_raw.json").write_text(json.dumps(out))
    print(f"{len(out)} variant device graphs -> {args.out / 'variants_raw.json'}")


def merge(args) -> None:
    """-> v8b_labels.pt {circuit: {"F": (N_AUG, 3) best fidelity, "layout": (n, N_AUG, 3),
    "counts": (N_AUG, 3, 3)}} in DEVICES order (with --orig: v8b_orig_labels.pt, N_AUG = 1)."""
    import torch
    out, bad = {}, 0
    keys = ([[f"{d}/orig" for d in DEVICES]] if args.orig
            else [[f"{d}/aug{j}" for d in DEVICES] for j in range(N_AUG)])
    for f in sorted(args.out.glob("task*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if "error" in r or len(r["F"]) != len(keys) * len(DEVICES):
                bad += 1
                continue
            F = torch.tensor([[max(r["F"][k]) for k in row] for row in keys])
            lay = torch.tensor([[r["layout"][k] for k in row] for row in keys]).permute(2, 0, 1)
            cnt = torch.tensor([[r["counts"][k] for k in row] for row in keys])
            out[r["circuit"]] = {"F": F, "layout": lay.contiguous(), "counts": cnt}
    name = "v8b_orig_labels.pt" if args.orig else "v8b_labels.pt"
    torch.save(out, args.out / name)
    print(f"{len(out)} circuits, {bad} skipped -> {args.out / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "raw", "merge"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--k", type=int, default=3, help="compiler seeds per (circuit, variant)")
    ap.add_argument("--orig", action="store_true", help="the unchanged devices instead of the variants")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"run": run, "raw": raw, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
