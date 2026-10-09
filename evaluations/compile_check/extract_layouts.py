#!/usr/bin/env python3
"""Turn the Sinkhorn placement of trained v5 models into real layouts for ``compile_check.py``.

For every circuit of the ``compile_check`` sample and every device, the soft
assignment ``A`` (logical x physical) of a trained ``sinkhorn`` model is rounded to a
one-to-one layout with the Hungarian algorithm (max total assignment probability);
graph node ``i`` of a device is backend qubit ``active_qubits[i]``.  Logical qubit
``i`` is qubit ``i`` of the QASM file (the dataset keeps that order).

Also written, per (circuit, device): the mean soft chip distance between the operands
of the two-qubit gates (what the model uses, ``A_a^T H A_b``) and the same distance
under the rounded layout, plus how sharp ``A`` is (mean max row probability).

    python extract_layouts.py --run generalization_v5/results/v5_sinkhorn \
        [--run ..._s6 --run ..._s7] --out layouts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.loader import DataLoader

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL / "generalization_v3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsv3 import paths  # noqa: E402
from gsv3.devices import load_devices  # noqa: E402
from gsv3.model import build_model, load_hparams  # noqa: E402

paths.ensure_imports()
from genstudy import paths as v1_paths  # noqa: E402
from genstudy.data import DEVICE_NAMES, FIGURE_OF_MERIT, GraphDataset  # noqa: E402

sys.path.insert(0, str(Path.home() / "compileCircuits"))
from compile_check import stratified_sample  # noqa: E402
from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="results dir of a sinkhorn run (repeatable, e.g. one per seed)")
    ap.add_argument("--family", default="qaoa")
    ap.add_argument("--per-size", type=int, default=16)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = stratified_sample(args.family, args.per_size, args.sample_seed)
    params = load_hparams(None)
    devices = load_devices(paths.V2_DEVICE_GRAPHS, DEVICE_NAMES, lap_pe=params["lap_pe"],
                           physical_errors=True)
    dataset = GraphDataset.load(v1_paths.find_dataset_dir(paths.DATA_DIR, FIGURE_OF_MERIT),
                                FIGURE_OF_MERIT)
    index = {n: i for i, n in enumerate(dataset.names)}
    missing = [n for n in names if n not in index]
    if missing:
        print(f"{len(missing)} sample circuits are not in the model dataset, skipped: {missing}")
    names = [n for n in names if n in index]
    graphs = dataset.subset([index[n] for n in names])
    backends = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(),
                "QExa20": QExa20Backend()}
    active = {d: list(getattr(b, "active_qubits", None) or range(b.num_qubits))
              for d, b in backends.items()}

    layouts, diag = {}, []
    for run in args.run:
        variant = run.name  # e.g. v5_sinkhorn_s6
        model = build_model("sinkhorn", params, devices, torch.device("cpu"))
        model.load_state_dict(torch.load(run / "leave_one_family_out" / args.family / "model.pth",
                                         map_location="cpu"))
        model.eval()
        captured = []
        orig = model._sinkhorn
        model._sinkhorn = lambda *a, _o=orig: captured.append(_o(*a)) or captured[-1]
        n_phys = torch.bincount(devices.batch).tolist()
        hops = model.hops
        layouts[variant] = {}
        loader = DataLoader(list(graphs), batch_size=32, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                model(batch, devices)
                assign = captured[0]                                   # (B, D, Qm, P)
                for b, data in enumerate(batch.to_data_list()):
                    name, n = data.circuit_name, int(data.n_qubits)
                    gq = data.gate_qubits
                    pairs = gq[(gq[:, 0] >= 0) & (gq[:, 1] >= 0)][:, :2].numpy()
                    layouts[variant][name] = {}
                    for d, dev in enumerate(DEVICE_NAMES):
                        a = assign[b, d, :n, : n_phys[d]].numpy()
                        rows, cols = linear_sum_assignment(-a)
                        nodes = cols[np.argsort(rows)]
                        layouts[variant][name][dev] = [int(active[dev][c]) for c in nodes]
                        h = hops[d, : n_phys[d], : n_phys[d]].numpy()
                        soft = float(np.mean([a[i] @ h @ a[j] for i, j in pairs])) if len(pairs) else 0.0
                        hard = float(np.mean([h[nodes[i], nodes[j]] for i, j in pairs])) if len(pairs) else 0.0
                        diag.append({"variant": variant, "circuit": name, "device": dev, "n": n,
                                     "soft_dist": soft, "hard_dist": hard,
                                     "max_row_prob": float(a.max(1).mean())})
        print(f"{variant}: {len(layouts[variant])} circuits", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(layouts))
    args.out.with_name(args.out.stem + "_diag.jsonl").write_text(
        "\n".join(json.dumps(r) for r in diag) + "\n")


if __name__ == "__main__":
    main()
