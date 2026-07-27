#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compile every circuit in benchmark_dataset_30k/ (QASM3) against the EQE1_Top,
EQE1_Bottom and QExa20 IQM-like targets at optimization_level=2, and record for
each (circuit, backend) pair the *real* cost of the brute-force baseline:

  - compilation_time : wall-clock time of pm.run(circ) ONLY. Unlike the sister
                       script (testComp-random-EQE1-complete.py) the pass manager
                       is built once per backend, outside the circuit loop, so its
                       construction overhead (which does not depend on the circuit)
                       is NOT counted here.
  - fidelity_time    : wall-clock time to evaluate the expected-fidelity formula
                       (compute_expected_log_fidelity) on the transpiled circuit.

The per-backend brute-force cost is (compilation_time + fidelity_time); the
per-circuit brute-force cost is the sum of that over the three backends. This is
the quantity we contrast against a GNN that predicts fidelity directly, without
compiling on every backend.

Also recorded (excluded from the timers): fidelity, log_fidelity and
expected_runtime, so the output JSON stays a superset of
expected_fidelity_results_benchmark_30k.json.

Fidelity is accumulated in log space and the per-circuit argmax is taken over
log_fidelity: the linear product underflows to exactly 0.0 beyond ~1e4 gates,
which would turn the oracle choice into a tie broken by dict ordering.

The grover family is excluded (see EXCLUDED_FAMILIES).

Requirements:
  - qiskit
  - createDevice.py (EQE1Top/EQE1Bottom/QExa20 BackendV2 targets)

Run:
  python testComp-compilation-time.py
"""

import math
import os
from pathlib import Path
import json
import time
from datetime import datetime, timezone
from qiskit import qasm3
from qiskit.transpiler import generate_preset_pass_manager

from createDevice import EQE1TopBackend, EQE1BottomBackend, QExa20Backend

# -------------------------------
# Configuration
# -------------------------------

IN_DIR = "data/benchmark_dataset_30k"                                # folder with .qasm inputs (recursive)
RESULTS_JSON = "evaluations/pipeline/timing_results/compilation_time_results_benchmark_30k.json"    # final JSON path
OPT_LEVEL = 2                                                   # single compilation level

# Benchmark families excluded from the sweep. grover's multi-controlled gates
# blow up ~21600x in transpilation (a 20-qubit instance reaches 3e7 gates and
# ~870 s per circuit), which is 92% of the total sweep cost while contributing
# no usable fidelity labels -- even in log space those circuits are so deep
# that every device is equally hopeless.
EXCLUDED_FAMILIES = {"grover"}


# -------------------------------
# Helpers
# -------------------------------

def list_qasm_files(folder: str):
    """Return all .qasm files under folder (recursive, excluding EXCLUDED_FAMILIES),
    as paths relative to folder."""
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(
        p.relative_to(root) for p in root.rglob("*.qasm")
        if p.relative_to(root).parts[0] not in EXCLUDED_FAMILIES
    )


def _log_fidelity_recursive(qc, device, qubit_map=None):
    """
    Expected fidelity of `qc` on `device`, accumulated in LOG space:
    sum of log(1 - error) over gates instead of the product of (1 - error).

    Same per-gate formula as mqt.predictor.reward.expected_fidelity, with two
    deliberate differences:

      - log space. The product underflows to exactly 0.0 for any circuit with
        more than ~1e4 gates, and mqt's version additionally rounds to 10
        decimals, so every deep circuit collapses to 0.0 and the per-device
        argmax degenerates into a tie broken by dict order. In log space the
        ranking stays exact however deep the circuit gets.
      - recursion into control-flow blocks (if_else/while_loop/for_loop/
        switch_case). The upstream function only walks the top-level
        instruction list and assumes every non-barrier op is a plain 1-/2-qubit
        gate with calibration data, so it raises a bare KeyError on any of the
        dynamic-circuit benchmarks (dynamic_qft, iqpe, the QEC codes, ...).

    `qubit_map[i]` gives the physical device qubit index for `qc`'s i-th
    qubit; `None` means `qc` already uses physical indices (the top-level,
    routed circuit). Branches take the fidelity-minimizing (worst-case) body,
    symmetric with estimate_expected_runtime's worst-case branch handling,
    since the real branch taken is runtime data-dependent.

    Returns the natural log of the expected fidelity (<= 0.0), or -inf if any
    gate has error 1.0.
    """
    if qubit_map is None:
        qubit_map = list(range(qc.num_qubits))

    res = 0.0
    for qc_instruction in qc.data:
        instruction, qargs = qc_instruction.operation, qc_instruction.qubits
        gate_type = instruction.name
        if gate_type == "barrier":
            continue

        physical_indices = [qubit_map[qc.find_bit(q).index] for q in qargs]

        blocks = getattr(instruction, "blocks", None)
        if blocks:
            # worst case == minimum log-fidelity, same ordering as min fidelity
            res += min(_log_fidelity_recursive(block, device, physical_indices) for block in blocks)
            continue

        if len(physical_indices) == 1:
            error = device[gate_type][physical_indices[0],].error
        else:
            error = device[gate_type][tuple(physical_indices[:2])].error
        if error >= 1.0:
            return float("-inf")
        res += math.log1p(-error)

    return res


def compute_expected_log_fidelity(tcirc, be):
    """
    Log expected fidelity of the transpiled circuit on backend `be`.

    Single implementation for every circuit (no mqt fast path / fallback
    split): besides avoiding the underflow described above, it keeps the
    measured fidelity_time comparable across circuit families, which it would
    not be if control-flow circuits used a different implementation from the
    rest.
    """
    try:
        return float(_log_fidelity_recursive(tcirc, be.target))
    except Exception as e:
        raise RuntimeError(
            f"Log-fidelity evaluation failed for backend '{be.name}'. Original error: {e}"
        )


def _instruction_duration(target, name):
    """Fixed per-gate duration (seconds) declared on `target` for instruction `name`."""
    if name not in target.operation_names:
        return 0.0
    for ip in target[name].values():
        if ip is not None and ip.duration is not None:
            return ip.duration
    return 0.0


def estimate_expected_runtime(circuit, target):
    """
    Estimate total circuit execution time (seconds) via ALAP-style
    critical-path scheduling over `target`'s per-gate durations.

    Qiskit's built-in scheduling passes reject circuits containing
    control-flow ops (if_else/while_loop/...), which several benchmark
    categories rely on (dynamic_qft, ghz_dynamic, iqpe, QEC codes), so the
    critical path is computed directly here instead. Control-flow bodies
    are counted once (branches take their max duration, loops a single
    iteration), since real branch/repetition counts are runtime data-dependent.
    """
    def run_block(qc):
        busy = {}
        for instr in qc.data:
            op = instr.operation
            bits = list(instr.qubits) + list(instr.clbits)
            t0 = max((busy.get(b, 0.0) for b in bits), default=0.0)
            if op.name == "barrier":
                dur = 0.0
            elif getattr(op, "blocks", None):
                dur = max((run_block(block) for block in op.blocks), default=0.0)
            else:
                dur = _instruction_duration(target, op.name)
            t_end = t0 + dur
            for b in bits:
                busy[b] = t_end
        return max(busy.values(), default=0.0)

    return run_block(circuit)


backends = {}
backends["EQE1_Top"] = EQE1TopBackend()
backends["EQE1_Bottom"] = EQE1BottomBackend()
backends["QExa20"] = QExa20Backend()

# Build each pass manager once, outside the circuit loop: it depends only on
# be.target, not on the circuit, so its construction must NOT be counted in the
# per-circuit compilation_time.
pass_managers = {
    be_name: generate_preset_pass_manager(optimization_level=OPT_LEVEL, target=be.target)
    for be_name, be in backends.items()
}

# -------------------------------
# Main: compile, score, save JSON
# -------------------------------

qasm_files = list_qasm_files(IN_DIR)

if not qasm_files:
    print(f"[INFO] No .qasm files in '{IN_DIR}'. Writing empty results and exiting.")
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)
    raise SystemExit(0)

# One-time global warm-up: the very first pm.run() in the process pays a one-off
# import/JIT cost (~15ms) that has nothing to do with any circuit's real compile
# time. Absorb it here (discarded) so it is not billed to the first real circuit's
# first backend. This is NOT per-circuit averaging -- each circuit is still measured
# exactly once below; this only excludes process-startup cost.
try:
    _warm = qasm3.load(os.path.join(IN_DIR, qasm_files[0]))
    for _pm in pass_managers.values():
        _pm.run(_warm)
    del _warm
    print("[WARMUP] Discarded one warm-up transpile per backend (process/JIT startup).")
except Exception as e:
    print(f"[WARMUP] Skipped warm-up ({e})")

# Load checkpoint: resume from existing results if present.
# Entries from before the log-fidelity / gate-duration fixes are dropped rather
# than resumed: they carry expected_runtime=0.0 (durations were missing from the
# device targets), no log_fidelity, and an argmax that degenerates to dict order
# on deep circuits. Excluded families are dropped too, so a checkpoint written
# before an exclusion does not survive in the output.
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    results = {
        tag: entry for tag, entry in loaded.items()
        if tag.split("/")[0] not in EXCLUDED_FAMILIES
        and entry[0] and all("log_fidelity" in be_data for be_data in entry[0].values())
    }
    print(
        f"[RESUME] Loaded {len(loaded)} entries from '{RESULTS_JSON}'; "
        f"kept {len(results)} (dropped {len(loaded) - len(results)}: excluded family or pre-fix schema)"
    )
else:
    results = {}

total = len(qasm_files)
for idx, rel_path in enumerate(qasm_files, start=1):
    in_path = os.path.join(IN_DIR, rel_path)
    src_tag = str(rel_path.with_suffix(""))

    if src_tag in results:
        print(f"[SKIP] {idx}/{total} {src_tag}: already in checkpoint")
        continue

    try:
        circ = qasm3.load(in_path)
    except Exception as e:
        print(f"[SKIP] Failed to load {rel_path}: {e}")
        continue
    size_hl = circ.size()
    depth_hl = circ.depth()

    per_backend = {}
    for be_name, be in backends.items():
        try:
            # Time ONLY pm.run: the pass manager is prebuilt above.
            t0 = time.perf_counter()
            tcirc = pass_managers[be_name].run(circ)
            compilation_time = time.perf_counter() - t0

            # Time the expected-fidelity formula evaluation (second half of the
            # brute-force baseline "compile + estimate fidelity").
            t1 = time.perf_counter()
            logF = compute_expected_log_fidelity(tcirc, be)
            fidelity_time = time.perf_counter() - t1

            # Linear fidelity kept for continuity with the reference results
            # file; it underflows to 0.0 on deep circuits (and math.exp(-inf)
            # is 0.0), which is exactly why log_fidelity is what the argmax
            # below uses.
            F = math.exp(logF)

            # Excluded from the timers: expected runtime estimate.
            expected_runtime = estimate_expected_runtime(tcirc, be.target)

            per_backend[be_name] = {
                "fidelity": F,
                "log_fidelity": logF,
                "size": tcirc.size(),
                "depth": tcirc.depth(),
                "size_hl": size_hl,
                "depth_hl": depth_hl,
                "compilation_time": compilation_time,
                "fidelity_time": fidelity_time,
                "expected_runtime": expected_runtime,
            }
            print(
                f"[OK] {idx}/{total} {src_tag} @ {be_name} L{OPT_LEVEL}: log_fidelity={logF:.4f} "
                f"(fidelity={F:.6f}) compile_time={compilation_time * 1e3:.3f}ms "
                f"fidelity_time={fidelity_time * 1e3:.3f}ms "
                f"expected_runtime={expected_runtime * 1e6:.2f}us"
            )

        except Exception as e:
            print(f"[ERR] {idx}/{total} {src_tag} @ {be_name} L{OPT_LEVEL}: {e}")

    if per_backend:
        # argmax over log_fidelity, not fidelity: the latter is 0.0 for every
        # deep circuit, which would make this a tie broken by dict order.
        argmax_backend = max(per_backend, key=lambda k: per_backend[k]["log_fidelity"])
        argmax_fid = per_backend[argmax_backend]["fidelity"]
        results[src_tag] = [per_backend, argmax_backend, argmax_fid]
        brute_force_cost = sum(
            d["compilation_time"] + d["fidelity_time"] for d in per_backend.values()
        )
        argmax_runtime = per_backend[argmax_backend]["expected_runtime"]
        print(
            f"[WIN] {idx}/{total} {src_tag}: {argmax_backend} with "
            f"log_fidelity={per_backend[argmax_backend]['log_fidelity']:.4f} "
            f"(fidelity={argmax_fid:.6f}) | expected_runtime={argmax_runtime * 1e6:.2f}us "
            f"| brute_force_cost={brute_force_cost * 1e3:.3f}ms"
        )
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

# Save single JSON with all circuits
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[DONE] Wrote '{RESULTS_JSON}' with {len(results)} entries on {datetime.now(timezone.utc).isoformat()}Z")