#!/usr/bin/env python3
"""Can a physics head that uses the *mean incident CZ error* of each operand follow local
calibration changes?  Oracle check, no model involved.

For the calibration-shift sample (circuits compiled on the original devices, stored as
QPY), with the layout and the gate list fixed, compute -log F on every local variant
(shuffle, jitterA, jitterB, jitterS) in two ways:

  exact : sum over the operations of -log(1-e) with the error of the qubit / coupler the
          operation actually runs on (r, measure: per qubit; cz: per coupler);
  proxy : the same, but each cz uses the mean of the two operands' mean incident CZ
          error, i.e. what the v4-v8 physics head multiplies (devices.phys[:, 2]).

Tracking = corr over circuits of the change vs orig.  ``proxy vs exact`` is the best a
perfect proxy-based model could do at fixed layout; ``exact vs truth`` and ``proxy vs
truth`` compare with the real change (best of 5 recompilations, which also re-lays out).
    python coupler_proxy_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[1]
CC = Path.home() / "compileCircuits"
sys.path.insert(0, str(CC))
QPY = EVAL / "compile_check/results/layouts_all/qpy"
RK = EVAL / "calibration_shift/results_k"
DEVICES = ("EQE1_Top", "EQE1_Bottom", "QExa20")
LOCAL = ("shuffle", "jitterA", "jitterB", "jitterS")


def main():
    from qiskit import qpy
    from createDevice import EQE1BottomBackend, EQE1TopBackend, QExa20Backend
    backends = {"EQE1_Top": EQE1TopBackend(), "EQE1_Bottom": EQE1BottomBackend(), "QExa20": QExa20Backend()}
    node = {d: {q: i for i, q in enumerate(getattr(b, "active_qubits", None) or range(b.num_qubits))}
            for d, b in backends.items()}
    raw = json.loads((RK / "variants_raw.json").read_text())
    truth = json.loads((RK / "truth.json").read_text())

    def tables(key):
        r = raw[key]
        nodes = np.array(r["nodes"], float)
        e1, em = nodes[:, 0], nodes[:, 1]                      # -log(1-e): 1q, readout
        E = {tuple(sorted(e)): v for e, v in zip(map(tuple, r["edges"]), r["edge_feats"])}
        inc = [[] for _ in range(len(nodes))]
        for (a, b), v in E.items():
            inc[a].append(v); inc[b].append(v)
        m = np.array([np.mean(x) if x else 0.0 for x in inc])
        return e1, em, E, m

    tab = {k: tables(k) for k in raw if k.split("/")[1] in ("orig", *LOCAL)}
    best = lambda n, k: max([f for f in truth[n][k] if f is not None], default=None)

    ops = {}   # (circuit, device) -> (list of 1q nodes, list of meas nodes, list of cz node pairs)
    for n in sorted(truth):
        f = QPY / f"{n}.qpy"
        if not f.is_file():
            continue
        with f.open("rb") as fh:
            circuits = qpy.load(fh)
        for d, tc in zip(DEVICES, circuits):
            one, meas, cz = [], [], []
            for inst in tc.data:
                q = [node[d][tc.find_bit(b).index] for b in inst.qubits]
                name = inst.operation.name
                if name == "r":
                    one.append(q[0])
                elif name == "measure":
                    meas.append(q[0])
                elif name == "cz":
                    cz.append(tuple(sorted(q)))
            ops[(n, d)] = (np.array(one, int), np.array(meas, int), cz)

    def cost(key, o, proxy):
        e1, em, E, m = tab[key]
        one, meas, cz = o
        c2 = sum((m[a] + m[b]) / 2 if proxy else E[(a, b)] for a, b in cz)
        return e1[one].sum() + em[meas].sum() + c2, c2

    print(f"{len(ops)} (circuit, device) pairs with compiled circuits\n")
    print(f"{'variant':22}{'n':>5}{'proxy vs exact':>16}{'2q only: proxy vs exact':>25}{'exact vs truth':>16}{'proxy vs truth':>16}")
    summary = {"pe": [], "pe2": [], "et": [], "pt": []}
    for d in DEVICES:
        for v in LOCAL:
            k, r = f"{d}/{v}", f"{d}/orig"
            de, dp, de2, dp2, dt = [], [], [], [], []
            for (n, dd), o in ops.items():
                if dd != d:
                    continue
                ce_v, c2e_v = cost(k, o, False); ce_o, c2e_o = cost(r, o, False)
                cp_v, c2p_v = cost(k, o, True); cp_o, c2p_o = cost(r, o, True)
                de.append(ce_v - ce_o); dp.append(cp_v - cp_o); de2.append(c2e_v - c2e_o); dp2.append(c2p_v - c2p_o)
                bt, bo = best(n, k), best(n, r)
                dt.append(-np.log(bt / bo) if bt and bo and bt > 0.01 and bo > 0.01 else np.nan)
            de, dp, de2, dp2, dt = map(np.array, (de, dp, de2, dp2, dt))
            ok = ~np.isnan(dt)
            pe = np.corrcoef(dp, de)[0, 1]; pe2 = np.corrcoef(dp2, de2)[0, 1]
            et = np.corrcoef(de[ok], dt[ok])[0, 1]; pt = np.corrcoef(dp[ok], dt[ok])[0, 1]
            for key, val in zip(summary, (pe, pe2, et, pt)):
                summary[key].append(val)
            print(f"{k:22}{len(de):>5}{pe:>16.2f}{pe2:>25.2f}{et:>16.2f}{pt:>16.2f}")
    print(f"\n{'mean over local variants':27}{np.mean(summary['pe']):>16.2f}{np.mean(summary['pe2']):>25.2f}"
          f"{np.mean(summary['et']):>16.2f}{np.mean(summary['pt']):>16.2f}")
    print("(truth = change of -log F, best of 5 recompilations, circuits with F > 0.01 on both)")


if __name__ == "__main__":
    main()
