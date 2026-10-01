"""Greedy scheduler vs. optimal assignment under the same per-device load.

The fidelity-aware policy (``gnn_device_policy.py``) assigns circuits one at a
time, so its outcome depends on queue order and on the greedy heuristic
itself.  This script separates the gap between Round-Robin and Oracle into

    optimal-GT  -> greedy-GT    cost of the greedy heuristic
    greedy-GT   -> greedy-GNN   cost of prediction error

For every fidelity weight ``w`` the greedy policy is replayed offline (same
score as ``gnn_device_policy.py``) and its per-device circuit counts are
recorded.  Under those same counts, the assignment that maximises total
fidelity is solved exactly as a transportation LP (circuits -> devices with
fixed device capacities; the constraint matrix is totally unimodular, so the
LP optimum is integral and equals the ``linear_sum_assignment`` optimum with
each device replicated into slots).  The optimum is computed with
ground-truth fidelities (optimal-GT) and with predicted fidelities
(optimal-GNN); both are scored on ground truth.

Everything is evaluated on the sorted queue used by ``run_gnn_dispatch.py``
(circuits ordered by name, hence grouped by family) and on random
permutations of the queue, to measure how much of the greedy-vs-optimal gap
is due to queue order.

Usage::

    # Sorted queue as reference, plus 20 random permutations
    python evaluations/pipeline/optimal_assignment.py \
        --results-dir evaluations/pipeline/results_v3_dense

    # Reference = the shuffled queue of ``run_gnn_dispatch.py --shuffle-seed 0``
    python evaluations/pipeline/optimal_assignment.py \
        --results-dir evaluations/pipeline/results_v4_shuffled --queue-seed 0 --n-perm 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

DEVICE_NAMES = ["EQE1_Top", "EQE1_Bottom", "QExa20"]
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]


def greedy(fids: np.ndarray, w: float) -> np.ndarray:
    """Replay ``GNNDevicePolicy.select`` and return the device index per circuit."""
    n, d = fids.shape
    out = np.empty(n, dtype=int)
    if w == 0.0:
        out[:] = np.arange(n) % d  # round-robin path of the policy
        return out
    load = np.zeros(d)
    for i in range(n):
        share = load / i if i > 0 else load
        score = w * fids[i] - (1.0 - w) * share
        best = int(np.argmax(score))  # first max on ties, as Python's max()
        out[i] = best
        load[best] += 1
    return out


def optimal(fids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Max-total-fidelity assignment with exactly ``counts[d]`` circuits on device d."""
    n, d = fids.shape
    var = np.arange(n * d)
    one_device = coo_matrix((np.ones(n * d), (np.repeat(np.arange(n), d), var)), shape=(n, n * d))
    capacity = coo_matrix((np.ones(n * d), (np.tile(np.arange(d), n), var)), shape=(d, n * d))
    res = linprog(
        -fids.ravel(),
        A_eq=vstack([one_device, capacity]).tocsr(),
        b_eq=np.concatenate([np.ones(n), counts]),
        bounds=(0, 1),
        method="highs",
    )
    x = res.x.reshape(n, d)
    if not np.allclose(x, np.round(x), atol=1e-6):
        raise RuntimeError("transportation LP returned a fractional solution")
    return x.argmax(1)


def mean_fid(gt: np.ndarray, assign: np.ndarray) -> float:
    return float(gt[np.arange(len(assign)), assign].mean())


def load_cv(assign: np.ndarray, d: int) -> float:
    counts = np.bincount(assign, minlength=d)
    return float(counts.std(ddof=1) / counts.mean())


_GT: np.ndarray
_PR: np.ndarray


def _init(gt: np.ndarray, pr: np.ndarray) -> None:
    global _GT, _PR
    _GT, _PR = gt, pr


def evaluate_order(order: np.ndarray) -> list[dict[str, float]]:
    """Greedy and optimal policies for every weight, with the queue in ``order``."""
    gt, pr = _GT, _PR
    n, d = gt.shape
    rows = []
    for w in WEIGHTS:
        g_gnn = np.empty(n, dtype=int)
        g_gnn[order] = greedy(pr[order], w)
        g_gt = np.empty(n, dtype=int)
        g_gt[order] = greedy(gt[order], w)
        c_gnn = np.bincount(g_gnn, minlength=d)
        c_gt = np.bincount(g_gt, minlength=d)
        rows.append({
            "greedy_gnn": mean_fid(gt, g_gnn),
            "greedy_gt": mean_fid(gt, g_gt),
            "optimal_gnn": mean_fid(gt, optimal(pr, c_gnn)),
            "optimal_gt": mean_fid(gt, optimal(gt, c_gt)),
            "optimal_gt_gnn_load": mean_fid(gt, optimal(gt, c_gnn)),
            "cv_gnn": load_cv(g_gnn, d),
            "cv_gt": load_cv(g_gt, d),
        })
    return rows


POLICIES = ("greedy_gnn", "greedy_gt", "optimal_gnn", "optimal_gt", "optimal_gt_gnn_load")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path,
                        default=Path("evaluations/pipeline/results_v3_dense"))
    parser.add_argument("--n-perm", type=int, default=20,
                        help="random queue permutations for the order-sensitivity check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--queue-seed", type=int, default=None,
                        help="reference queue = sorted names shuffled with this seed, "
                             "as in run_gnn_dispatch.py --shuffle-seed (default: sorted)")
    args = parser.parse_args()

    gt_raw = json.loads((args.results_dir / "fidelity_ground_truth.json").read_text())
    pr_raw = json.loads((args.results_dir / "fidelity_gnn_predicted.json").read_text())
    names = sorted(set(gt_raw) & set(pr_raw))  # same canonical order as the driver
    gt = np.array([gt_raw[k] for k in names], dtype=float)
    pr = np.array([pr_raw[k] for k in names], dtype=float)
    n, d = gt.shape

    ref_order = np.arange(n)
    if args.queue_seed is not None:
        shuffled = list(names)
        random.Random(args.queue_seed).shuffle(shuffled)
        index = {k: i for i, k in enumerate(names)}
        ref_order = np.array([index[k] for k in shuffled])

    # Round-Robin depends on queue order too: normalise by the reference queue's
    rr_assign = np.empty(n, dtype=int)
    rr_assign[ref_order] = np.arange(n) % d
    rr = mean_fid(gt, rr_assign)
    oracle = mean_fid(gt, gt.argmax(1))
    gap = oracle - rr
    print(f"{n} circuits  round-robin={rr:.4f}  oracle={oracle:.4f}")

    rng = np.random.default_rng(args.seed)
    orders = [ref_order] + [rng.permutation(n) for _ in range(args.n_perm)]
    with Pool(args.workers, initializer=_init, initargs=(gt, pr)) as pool:
        runs = pool.map(evaluate_order, orders)
    sorted_run, perm_runs = runs[0], runs[1:]

    rows = []
    for i, w in enumerate(WEIGHTS):
        row: dict[str, float] = {"w": w}
        for key in (*POLICIES, "cv_gnn", "cv_gt"):
            row[key] = sorted_run[i][key]
            vals = np.array([r[i][key] for r in perm_runs] or [np.nan])
            row[f"{key}_perm_mean"] = float(vals.mean())
            row[f"{key}_perm_std"] = float(vals.std(ddof=1)) if len(perm_runs) > 1 else float("nan")
        for key in POLICIES:
            row[f"{key}_norm"] = (row[key] - rr) / gap
            row[f"{key}_perm_norm"] = (row[f"{key}_perm_mean"] - rr) / gap
        rows.append(row)
        print(f"w={w:.1f}  " + "  ".join(
            f"{k}={row[k + '_norm']:6.1%}/{row[k + '_perm_norm']:6.1%}" for k in POLICIES
        ) + f"  CV gnn={row['cv_gnn']:.3f}/{row['cv_gnn_perm_mean']:.3f}")

    out = args.results_dir / "optimal_assignment.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.6f}" for k, v in r.items()})
    meta = {"n_circuits": n, "round_robin": rr, "oracle": oracle,
            "n_perm": args.n_perm, "seed": args.seed, "queue_seed": args.queue_seed}
    (args.results_dir / "optimal_assignment_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
