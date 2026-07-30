#!/usr/bin/env python3
"""GNN device-selection dispatch — dry-run benchmark via the QMS policy engine.

Wires together:
1. GNNDevicePolicy registration (automatic on import).
2. Ground-truth fidelities for the benchmark dataset.
3. GNN fidelity predictions for the same circuits.
4. A simulated dispatch pass per policy (oracle / gt_weighted / gnn / round_robin).

The oracle and all metrics use ground truth; the GNN policy makes its
device-selection decisions from its own predictions.

Usage::

    # Dry-run on the test split, single weight
    python evaluations/pipeline/run_gnn_dispatch.py --max-circuits 10

    # Sweep over fidelity weights
    python evaluations/pipeline/run_gnn_dispatch.py --fidelity-weights 0.0:1.0:0.1

    # Keep only circuits whose best ground-truth fidelity is >= 0.01
    python evaluations/pipeline/run_gnn_dispatch.py --min-best-fidelity 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_MODEL_DIR = _REPO_ROOT / "src" / "model"

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Import registers "gnn_device" in the multi-device policy registry
# (and puts src/ and src/model/ on sys.path)
import gnn_device_policy  # noqa: F401, E402
from gnn_device_policy import (  # noqa: E402
    DEVICE_NAMES,
    GNNDevicePolicy,
    _GNNPredictor,
    _circuit_to_data,
)

from qms.policy import get_multi_policy  # noqa: E402
from qms.policy.base import DeviceState, MultiDevicePolicy, MultiDeviceSnapshot  # noqa: E402
from qms.policy.multi_round_robin import RoundRobinMultiDevicePolicy  # noqa: E402

logger = logging.getLogger("run_gnn_dispatch")

_LOG_FMT = "%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    """Configure root logger with console + file output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    root.handlers[0].setFormatter(logging.Formatter(_LOG_FMT))

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    root.addHandler(fh)

    # Quieten noisy Qiskit loggers
    for noisy in ("qiskit.passmanager", "qiskit.compiler", "qiskit.transpiler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Logging to %s", log_path)


# ---------------------------------------------------------------------------
# Fidelity prediction / loading
# ---------------------------------------------------------------------------

def predict_all_fidelities(
    circuit_dir: Path,
    stems: list[str],
) -> dict[str, list[float]]:
    """Run GNN inference for each stem ("family/name") found in circuit_dir.

    Returns {stem: [fid_dev0, fid_dev1, fid_dev2]}.
    """
    predictor = _GNNPredictor()

    logger.info("Predicting fidelities for %d circuits ...", len(stems))
    fidelities: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for i, stem in enumerate(stems, 1):
        qasm_path = circuit_dir / f"{stem}.qasm"
        if not qasm_path.exists():
            logger.warning("Circuit file %s not found; skipping.", qasm_path)
            continue
        try:
            data = _circuit_to_data(qasm_path)
            fidelities[stem] = predictor.predict(data)
        except Exception:
            logger.warning("Skipping %s (GNN inference failed)", stem, exc_info=True)
        if i % 50 == 0:
            logger.info("  predicted %d / %d", i, len(stems))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Predictions done: %d circuits in %.1f s (%.1f circuits/s)",
        len(fidelities), elapsed, len(fidelities) / max(elapsed, 1e-9),
    )
    return fidelities


def load_ground_truth_fidelities(
    gt_path: Path,
    names_path: Path | None = None,
    min_best_fidelity: float | None = None,
    max_circuits: int | None = None,
) -> dict[str, list[float]]:
    """Load ground-truth per-device fidelities from a pre-computed JSON file.

    Expected format::

        {
          "family/circuit_name": [
            {"EQE1_Top": {"fidelity": 0.8, ...}, "EQE1_Bottom": {...}, "QExa20": {...}},
            "EQE1_Top",   <- best device (ignored here)
            0.8           <- best fidelity (ignored here)
          ],
          ...
        }

    Args:
        gt_path: Path to the ground-truth JSON file.
        names_path: Optional JSON file with a list of circuit names; when
            given, only those circuits are kept (e.g. the held-out test split).
        min_best_fidelity: Optional threshold; circuits whose best fidelity
            across all devices is below it are dropped.
        max_circuits: Optional cap on the number of circuits (after filtering).
    """
    with open(gt_path) as f:
        raw: dict[str, list] = json.load(f)

    if names_path is not None:
        with open(names_path) as f:
            names = set(json.load(f))
        raw = {stem: entry for stem, entry in raw.items() if stem in names}
        logger.info(
            "Restricted ground truth to %d circuits listed in %s",
            len(raw), names_path,
        )

    fidelities: dict[str, list[float]] = {}
    for stem, entry in sorted(raw.items()):
        dev_map: dict[str, dict[str, Any]] = entry[0]
        fids = [float(dev_map[d]["fidelity"]) for d in DEVICE_NAMES]
        if min_best_fidelity is not None and max(fids) < min_best_fidelity:
            continue
        fidelities[stem] = fids

    if min_best_fidelity is not None:
        logger.info(
            "Kept %d circuits with best fidelity >= %.4f",
            len(fidelities), min_best_fidelity,
        )

    if max_circuits:
        fidelities = dict(list(fidelities.items())[:max_circuits])

    logger.info(
        "Loaded ground-truth fidelities for %d circuits from %s",
        len(fidelities), gt_path,
    )
    return fidelities


# ---------------------------------------------------------------------------
# Weight parsing
# ---------------------------------------------------------------------------

def _parse_weight_list(value: str) -> list[float]:
    """Parse "0.0,0.3,0.7" or "0.0:1.0:0.1" into a list of floats."""
    if ":" in value:
        parts = value.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        weights, w = [], start
        while w <= stop + 1e-9:
            weights.append(round(w, 4))
            w += step
        return weights
    return [float(x.strip()) for x in value.split(",")]


# ---------------------------------------------------------------------------
# Dispatch simulation (dry-run)
# ---------------------------------------------------------------------------

POLICY_ORDER = ["oracle", "gt_weighted", "gnn", "round_robin"]


def _simulate_oracle(
    fidelities: dict[str, list[float]],
) -> dict[str, list[str]]:
    """Oracle policy: assign each circuit to the device with the highest fidelity."""
    assignments: dict[str, list[str]] = {name: [] for name in DEVICE_NAMES}
    for stem, fids in fidelities.items():
        best_idx = max(range(len(fids)), key=lambda i: fids[i])
        assignments[DEVICE_NAMES[best_idx]].append(stem)
    return assignments


def _build_snapshot(
    fidelities: dict[str, list[float]],
) -> MultiDeviceSnapshot:
    """Build a MultiDeviceSnapshot with all circuits pending and unlimited slots."""
    pending: list[tuple[str, dict[str, Any]]] = [
        (stem, {"predicted_fidelities": fids})
        for stem, fids in fidelities.items()
    ]
    devices = {
        name: DeviceState(running_count=0, max_concurrent=len(pending))
        for name in DEVICE_NAMES
    }
    return MultiDeviceSnapshot(pending=pending, devices=devices)


def _simulate_dispatch(
    policy: MultiDevicePolicy,
    fidelities: dict[str, list[float]],
) -> dict[str, list[str]]:
    """Simulate one full dispatch pass through the unified queue."""
    snapshot = _build_snapshot(fidelities)
    return policy.select(snapshot)


def _compute_metrics(
    assignments: dict[str, list[str]],
    fidelities: dict[str, list[float]],
) -> dict[str, Any]:
    """Compute per-device and overall fidelity metrics for one weight."""
    device_metrics: dict[str, dict[str, Any]] = {}
    all_fids: list[float] = []

    for dev_idx, dev_name in enumerate(DEVICE_NAMES):
        stems = assignments.get(dev_name, [])
        fids_on_dev = [fidelities[s][dev_idx] for s in stems if s in fidelities]

        if fids_on_dev:
            device_metrics[dev_name] = {
                "count": len(fids_on_dev),
                "mean": statistics.mean(fids_on_dev),
                "median": statistics.median(fids_on_dev),
                "stdev": statistics.stdev(fids_on_dev) if len(fids_on_dev) > 1 else 0.0,
                "min": min(fids_on_dev),
                "max": max(fids_on_dev),
            }
        else:
            device_metrics[dev_name] = {
                "count": 0, "mean": 0.0, "median": 0.0,
                "stdev": 0.0, "min": 0.0, "max": 0.0,
            }
        all_fids.extend(fids_on_dev)

    counts = [device_metrics[d]["count"] for d in DEVICE_NAMES]
    mean_count = statistics.mean(counts) if counts else 0
    cv = (statistics.stdev(counts) / mean_count) if mean_count > 0 and len(counts) > 1 else 0.0

    return {
        "overall": {
            "total_circuits": len(all_fids),
            "mean": statistics.mean(all_fids) if all_fids else 0.0,
            "median": statistics.median(all_fids) if all_fids else 0.0,
            "stdev": statistics.stdev(all_fids) if len(all_fids) > 1 else 0.0,
            "min": min(all_fids) if all_fids else 0.0,
            "max": max(all_fids) if all_fids else 0.0,
        },
        "per_device": device_metrics,
        "load_balance_cv": cv,
    }


# ---------------------------------------------------------------------------
# Sweep + reporting
# ---------------------------------------------------------------------------

def _run_single_weight(
    fidelities: dict[str, list[float]],
    fidelity_weight: float,
    gnn_fidelities: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    """Run Oracle + GT-Weighted + GNN + Round-Robin for one weight.

    Args:
        fidelities: Ground-truth fidelities used by the oracle and for all
            metric computation.
        fidelity_weight: Weight parameter for the GNN policy.
        gnn_fidelities: The GNN policy makes device-selection decisions based
            on these (its own predictions) instead of *fidelities*.  This
            separates "what the GNN sees" from "what actually happens",
            giving a realistic benchmark.
    """
    _policy_cfg = {
        "fidelity_weight": fidelity_weight,
        # -inf, not 0.0: the regressor can predict small negative fidelities
        # for circuits that are dead on every device, and a 0.0 threshold
        # dropped 16 of them, scoring the GNN on 5101 circuits against the
        # other policies' 5117.
        "fidelity_threshold": float("-inf"),
        "max_batch_size": len(fidelities),
    }

    gnn = GNNDevicePolicy()
    gnn.configure(_policy_cfg)

    # Ground-truth weighted: same scoring formula as GNN but with perfect
    # fidelity knowledge.
    gt_weighted = GNNDevicePolicy()
    gt_weighted.configure(_policy_cfg)

    # Oracle is weight-independent but included at every weight for easy plotting
    dispatch_map: dict[str, dict[str, list[str]]] = {
        "oracle": _simulate_oracle(fidelities),
        "gt_weighted": _simulate_dispatch(gt_weighted, fidelities),
        "gnn": _simulate_dispatch(gnn, gnn_fidelities),
        "round_robin": _simulate_dispatch(RoundRobinMultiDevicePolicy(), fidelities),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, dispatch_result in dispatch_map.items():
        for dev_name in DEVICE_NAMES:
            logger.info(
                "  [%s] %s  ->  %d circuits",
                name, dev_name, len(dispatch_result.get(dev_name, [])),
            )

        # Metrics always evaluated against ground-truth fidelities
        metrics = _compute_metrics(dispatch_result, fidelities)
        logger.info(
            "  [%s] overall mean fidelity: %.4f  (load CV: %.3f)",
            name, metrics["overall"]["mean"], metrics["load_balance_cv"],
        )
        results[name] = metrics

    # Cross-policy comparisons against the oracle and GT-weighted baselines
    results["cross_policy"] = _compute_cross_policy_metrics(
        dispatch_map["oracle"], dispatch_map["gnn"], fidelities,
    )
    cp = results["cross_policy"]
    logger.info(
        "  [cross-policy oracle↔gnn] agreement: %.1f%%  mean regret: %.4f",
        cp["device_agreement_rate"] * 100, cp["regret"]["mean"],
    )

    results["cross_policy_gt_weighted"] = _compute_cross_policy_metrics(
        dispatch_map["gt_weighted"], dispatch_map["gnn"], fidelities,
    )
    cpg = results["cross_policy_gt_weighted"]
    logger.info(
        "  [cross-policy gt_weighted↔gnn] agreement: %.1f%%  mean regret: %.4f",
        cpg["device_agreement_rate"] * 100, cpg["regret"]["mean"],
    )

    return results


def _invert_assignments(assignments: dict[str, list[str]]) -> dict[str, str]:
    """Convert {device: [circuits]} -> {circuit: device}."""
    return {stem: dev for dev, stems in assignments.items() for stem in stems}


def _compute_cross_policy_metrics(
    oracle_assignments: dict[str, list[str]],
    gnn_assignments: dict[str, list[str]],
    fidelities: dict[str, list[float]],
) -> dict[str, Any]:
    """Compare GNN device assignments against a baseline.

    Returns dict with:
        device_agreement_rate: fraction of circuits assigned to same device
        regret: {mean, median, max, zero_regret_pct, values}
        confusion_matrix: {matrix (list-of-lists), labels}
    """
    oracle_map = _invert_assignments(oracle_assignments)
    gnn_map = _invert_assignments(gnn_assignments)

    # Only compare circuits present in both
    common = sorted(set(oracle_map) & set(gnn_map) & set(fidelities))

    dev_to_idx = {name: i for i, name in enumerate(DEVICE_NAMES)}
    n_dev = len(DEVICE_NAMES)

    agrees = 0
    regrets: list[float] = []
    confusion: list[list[int]] = [[0] * n_dev for _ in range(n_dev)]

    for stem in common:
        o_dev = oracle_map[stem]
        g_dev = gnn_map[stem]
        fids = fidelities[stem]

        if o_dev == g_dev:
            agrees += 1

        o_fid = fids[dev_to_idx[o_dev]]
        g_fid = fids[dev_to_idx[g_dev]]
        regrets.append(o_fid - g_fid)

        confusion[dev_to_idx[o_dev]][dev_to_idx[g_dev]] += 1

    n = len(common)
    return {
        "device_agreement_rate": agrees / n if n else 0.0,
        "regret": {
            "mean": statistics.mean(regrets) if regrets else 0.0,
            "median": statistics.median(regrets) if regrets else 0.0,
            "max": max(regrets) if regrets else 0.0,
            "zero_regret_pct": sum(1 for r in regrets if r == 0.0) / n if n else 0.0,
            "values": regrets,
        },
        "confusion_matrix": {
            "matrix": confusion,
            "labels": list(DEVICE_NAMES),
        },
    }


def run_sweep(
    fidelities: dict[str, list[float]],
    fidelity_weights: list[float],
    gnn_fidelities: dict[str, list[float]],
) -> dict[float, dict[str, dict[str, Any]]]:
    """Run all policies for each weight and collect metrics."""
    all_results: dict[float, dict[str, dict[str, Any]]] = {}
    for w in fidelity_weights:
        logger.info("=== Fidelity weight %.2f ===", w)
        all_results[w] = _run_single_weight(fidelities, w, gnn_fidelities)
    return all_results


def print_report(all_results: dict[float, dict[str, dict[str, Any]]]) -> None:
    """Print a formatted table for each weight, plus a sweep summary."""
    for weight in sorted(all_results):
        results = all_results[weight]

        header = (
            f"{'Policy':<14} {'Circuits':>8} {'Mean Fid':>9} {'Median':>9} "
            f"{'Stdev':>8} {'Min':>8} {'Max':>8} {'Load CV':>8}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(f"  Fidelity weight = {weight:.2f}")
        print(sep)
        print(header)
        print(sep)

        for name in POLICY_ORDER:
            if name not in results:
                continue
            o = results[name]["overall"]
            cv = results[name]["load_balance_cv"]
            print(
                f"{name:<14} {o['total_circuits']:>8d} "
                f"{o['mean']:>9.4f} {o['median']:>9.4f} {o['stdev']:>8.4f} "
                f"{o['min']:>8.4f} {o['max']:>8.4f} {cv:>8.3f}"
            )
        print(sep)

        # Per-device breakdown
        dev_header = (
            f"  {'Policy':<14} {'Device':<14} {'Count':>6} "
            f"{'Mean Fid':>9} {'Median':>9} {'Min':>8} {'Max':>8}"
        )
        dev_sep = "-" * len(dev_header)
        print(f"\n  Per-device breakdown:\n{dev_sep}\n{dev_header}\n{dev_sep}")
        for name in POLICY_ORDER:
            if name not in results:
                continue
            for dev_name in DEVICE_NAMES:
                d = results[name]["per_device"][dev_name]
                print(
                    f"  {name:<14} {dev_name:<14} {d['count']:>6d} "
                    f"{d['mean']:>9.4f} {d['median']:>9.4f} "
                    f"{d['min']:>8.4f} {d['max']:>8.4f}"
                )
            print()
        print(dev_sep)

        if "gnn" in results and "round_robin" in results:
            rr_mean = results["round_robin"]["overall"]["mean"]
            gnn_mean = results["gnn"]["overall"]["mean"]
            if rr_mean > 0:
                pct = (gnn_mean - rr_mean) / rr_mean * 100
                print(f"\n  GNN vs Round-Robin: {pct:+.2f}%")

        # Cross-policy metrics (GNN vs baselines)
        for cp_key, cp_title, cp_row_label, cp_col_label in [
            ("cross_policy", "GNN vs Oracle (fidelity-only)", "Oracle", "GNN"),
            ("cross_policy_gt_weighted", "GNN vs GT-Weighted", "GT-Weighted", "GNN"),
        ]:
            if cp_key not in results:
                continue
            cp = results[cp_key]
            reg = cp["regret"]
            cm = cp["confusion_matrix"]
            labels = cm["labels"]
            matrix = cm["matrix"]

            print(f"\n  --- {cp_title} comparison ---")
            print(f"  Device agreement rate:  {cp['device_agreement_rate']:.1%}")
            print(
                f"  Fidelity regret:  mean={reg['mean']:.4f}  "
                f"median={reg['median']:.4f}  max={reg['max']:.4f}  "
                f"zero-regret={reg['zero_regret_pct']:.1%}"
            )

            col_w = max(len(l) for l in labels) + 2
            row_label_w = col_w
            print(f"\n  Confusion matrix (rows={cp_row_label}, cols={cp_col_label}):")
            print(f"  {'':<{row_label_w}}" + "".join(f"{l:>{col_w}}" for l in labels))
            for i, row_label in enumerate(labels):
                row_str = "".join(f"{matrix[i][j]:>{col_w}d}" for j in range(len(labels)))
                print(f"  {row_label:<{row_label_w}}{row_str}")

        print()

    # Sweep summary
    if len(all_results) > 1:
        sweep_hdr = (
            f"{'Weight':>8} {'Oracle':>9} {'GT-Wtd':>9}"
            f" {'GNN Mean':>9} {'RR Mean':>9} "
            f"{'GNN vs RR':>10} {'GNN vs Orc':>11} {'GNN vs GTW':>11}"
            f" {'Agree%':>8} {'MnRegret':>9}"
        )
        sep = "-" * len(sweep_hdr)
        print(f"\n{sep}")
        print("  SWEEP SUMMARY — Mean fidelity by fidelity weight")
        print(sep)
        print(sweep_hdr)
        print(sep)
        for w in sorted(all_results):
            r = all_results[w]
            orc = r["oracle"]["overall"]["mean"]
            gtw = r["gt_weighted"]["overall"]["mean"]
            g = r["gnn"]["overall"]["mean"]
            rr = r["round_robin"]["overall"]["mean"]
            pct_rr = (g - rr) / rr * 100 if rr else 0.0
            pct_orc = (g - orc) / orc * 100 if orc else 0.0
            pct_gtw = (g - gtw) / gtw * 100 if gtw else 0.0
            cp = r["cross_policy"]
            line = (
                f"{w:>8.2f} {orc:>9.4f} {gtw:>9.4f}"
                f" {g:>9.4f} {rr:>9.4f} {pct_rr:>+9.2f}% {pct_orc:>+10.2f}% {pct_gtw:>+10.2f}%"
                f" {cp['device_agreement_rate']*100:>7.1f}% {cp['regret']['mean']:>9.4f}"
            )
            print(line)
        print(sep)
        print()


def save_results(
    all_results: dict[float, dict[str, dict[str, Any]]],
    fidelities: dict[str, list[float]],
    output_dir: Path,
    gnn_fidelities: dict[str, list[float]],
) -> None:
    """Persist results as JSON and CSV, plus the fidelity data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fidelities
    gt_path = output_dir / "fidelity_ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(fidelities, f, indent=2)
    logger.info("Ground-truth fidelities saved to %s", gt_path)

    gnn_path = output_dir / "fidelity_gnn_predicted.json"
    with open(gnn_path, "w") as f:
        json.dump(gnn_fidelities, f, indent=2)
    logger.info("GNN-predicted fidelities saved to %s", gnn_path)

    # Save full results JSON (multi-policy format, compatible with plot_benchmark.py)
    json_path = output_dir / "dispatch_results.json"
    with open(json_path, "w") as f:
        json.dump({f"{w:.2f}": v for w, v in all_results.items()}, f, indent=2)
    logger.info("Full results saved to %s", json_path)

    # Save CSV summary
    csv_path = output_dir / "dispatch_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fidelity_weight", "policy", "device", "count",
            "mean_fidelity", "median_fidelity", "stdev", "min", "max",
        ])
        for w in sorted(all_results):
            for name in POLICY_ORDER:
                if name not in all_results[w]:
                    continue
                for dev_name in DEVICE_NAMES:
                    d = all_results[w][name]["per_device"][dev_name]
                    writer.writerow([
                        f"{w:.2f}", name, dev_name, d["count"],
                        f"{d['mean']:.6f}", f"{d['median']:.6f}",
                        f"{d['stdev']:.6f}", f"{d['min']:.6f}", f"{d['max']:.6f}",
                    ])
                o = all_results[w][name]["overall"]
                writer.writerow([
                    f"{w:.2f}", name, "OVERALL", o["total_circuits"],
                    f"{o['mean']:.6f}", f"{o['median']:.6f}",
                    f"{o['stdev']:.6f}", f"{o['min']:.6f}", f"{o['max']:.6f}",
                ])
    logger.info("Summary CSV saved to %s", csv_path)

    # Save cross-policy CSV
    cp_keys = [
        ("cross_policy", "oracle"),
        ("cross_policy_gt_weighted", "gt_weighted"),
    ]
    cp_csv_path = output_dir / "cross_policy_summary.csv"
    with open(cp_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fidelity_weight", "baseline", "agreement_rate",
            "regret_mean", "regret_median", "regret_max", "zero_regret_pct",
        ])
        for w in sorted(all_results):
            for cpk, baseline_name in cp_keys:
                if cpk not in all_results[w]:
                    continue
                cp = all_results[w][cpk]
                reg = cp["regret"]
                writer.writerow([
                    f"{w:.2f}",
                    baseline_name,
                    f"{cp['device_agreement_rate']:.6f}",
                    f"{reg['mean']:.6f}",
                    f"{reg['median']:.6f}",
                    f"{reg['max']:.6f}",
                    f"{reg['zero_regret_pct']:.6f}",
                ])
    logger.info("Cross-policy CSV saved to %s", cp_csv_path)


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def dry_run(
    fidelities: dict[str, list[float]],
    fidelity_weights: list[float],
    output_dir: Path,
    gnn_fidelities: dict[str, list[float]],
) -> None:
    """Simulate device selection across a sweep of weights — no QPU execution."""
    # Verify registry works
    cls = get_multi_policy("gnn_device")
    logger.info("Registry check: get_multi_policy('gnn_device') -> %s", cls.__name__)

    all_results = run_sweep(fidelities, fidelity_weights, gnn_fidelities)
    print_report(all_results)
    save_results(all_results, fidelities, output_dir, gnn_fidelities)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run benchmark of GNN-guided device selection.",
    )
    parser.add_argument(
        "--circuit-dir", type=Path,
        default=_REPO_ROOT / "data" / "benchmark_dataset_30k",
        help="Directory with the benchmark circuits (family/name.qasm layout).",
    )
    parser.add_argument(
        "--gt-path", type=Path,
        default=_MODEL_DIR / "expected_fidelity_results_benchmark.json",
        help="Ground-truth fidelity JSON (oracle + metrics).",
    )
    parser.add_argument(
        "--names-path", type=Path,
        default=_MODEL_DIR / "test_circuit_names.json",
        help="JSON list of circuit names to evaluate (default: held-out test split).",
    )
    parser.add_argument(
        "--min-best-fidelity", type=float, default=None,
        metavar="F",
        help="Drop circuits whose best ground-truth fidelity is below F.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_SCRIPT_DIR / "results",
    )
    parser.add_argument("--max-circuits", type=int, default=None)
    parser.add_argument(
        "--fidelity-weights", type=str, default="0.7",
        help="Comma-separated or start:stop:step (default: 0.7)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.output_dir / "dispatch.log")

    fidelity_weights = _parse_weight_list(args.fidelity_weights)

    # Ground truth: oracle + metrics
    fidelities = load_ground_truth_fidelities(
        args.gt_path,
        names_path=args.names_path,
        min_best_fidelity=args.min_best_fidelity,
        max_circuits=args.max_circuits,
    )
    if not fidelities:
        logger.error("No circuits found — nothing to dispatch.")
        sys.exit(1)

    # GNN predictions: what the GNN policy sees when making decisions
    logger.info(
        "Running GNN inference on circuits in %s "
        "(GNN policy will use its own predictions, not ground truth)",
        args.circuit_dir,
    )
    gnn_fidelities = predict_all_fidelities(args.circuit_dir, sorted(fidelities))

    # Restrict both to the intersection of available circuits, in a canonical
    # (sorted) order so the unified-queue iteration is the same across both
    # dicts — otherwise RR-style cycling desyncs and produces different
    # per-device assignments at low weights.
    common = sorted(set(fidelities) & set(gnn_fidelities))
    if len(common) < len(fidelities) or len(common) < len(gnn_fidelities):
        logger.info(
            "Restricted to %d circuits present in both ground truth and circuit dir "
            "(gt=%d, gnn=%d)",
            len(common), len(fidelities), len(gnn_fidelities),
        )
    fidelities = {k: fidelities[k] for k in common}
    gnn_fidelities = {k: gnn_fidelities[k] for k in common}

    if not fidelities:
        logger.error("No circuits found — nothing to dispatch.")
        sys.exit(1)

    logger.info("Benchmarking %d circuits", len(fidelities))

    dry_run(fidelities, fidelity_weights, args.output_dir, gnn_fidelities)


if __name__ == "__main__":
    main()
