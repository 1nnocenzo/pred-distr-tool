#!/usr/bin/env python3
"""Write evaluations/RESULTS.md (+ figures/) from the current results — the study's results page.

Everything is recomputed from the result files at every run:
  * generalization  : per-split test metrics of every run (generalization_v*/results/**/metrics.json,
                      via generalization_v3/scripts/summarize.collect), mean ± std over seeds;
  * calibration     : low-noise calibration shift (calibration_shift/results_k: best-of-5 truth,
                      predictions of the control models);
  * timing          : timing_v8/timing_scaling_threads1.json.
The model descriptions and the main-model choice are the only hand-written parts (MODELS, MAIN).

    python evaluations/results_report.py            # once
    sbatch evaluations/results_loop.sh               # every 15 minutes, as a SLURM job
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL / "generalization_v3" / "scripts"))
import summarize  # noqa: E402

OUT = EVAL / "RESULTS.md"
FIG = EVAL / "figures"
MAIN = "v8a_c001"

# config -> (short name, one-line idea, status). Hand-written: the only non-computed text.
MODELS = {
    "v4_phys": ("v4", "physics head + per-qubit attention", "baseline"),
    "v5_sinkhorn": ("v5", "one-to-one Sinkhorn placement + chip distance", "baseline"),
    "v5b_sinkhorn": ("v5b", "v5 + fixed budget, cosine LR, SWA", "discarded"),
    "v4u_phys": ("phys_uniform", "v4 without placement (diagnostic)", "diagnostic"),
    "v6_l01": ("v6", "v5 + loss on the compiler's exact layout", "discarded"),
    "v7_l01": ("v7", "v5 + loss on used region and qubit distances", "superseded"),
    "v8a_c001": ("v8a", "v7 + loss on the true gate counts", "**main model**"),
    "v8b_c001": ("v8b", "v8a + calibration variants in training", "discarded"),
    "v8c_d1": ("v8c", "v8a fine-tuned with a loss on the change of log F", "pilot"),
    "v8c_d1_v4": ("v8c (4 var.)", "v8c with 4 of the 8 calibration variants", "test"),
    "v8r": ("v8r", "v8a + routing-free flag (VF2 subgraph test)", "3 seeds running"),
    "v9": ("v9", "v8r + per-device relative qubit features", "running"),
    "v8r_cz": ("v8r_cz", "v8r + count loss weighted on cz (0.1 vs 0.01)", "running"),
    "v8s": ("v8s", "v8r_cz + greedy routing-cost estimate as input", "running"),
    "v9f": ("v9f", "v8r + the add-ons kept by the v9 decision", "running"),
}
COMPARE = ["v4_phys", "v5_sinkhorn", MAIN]           # the 3-seed comparison
SPLITS = [("lofo", "qaoa", None), ("lofo", "qnn", None), ("logo", "variational", None),
          ("logo", "variational", "qnn"), ("control", None, None)]
SPLIT_NAME = {("lofo", "qaoa", None): "LOFO qaoa", ("lofo", "qnn", None): "LOFO qnn",
              ("logo", "variational", None): "LOGO variational",
              ("logo", "variational", "qnn"): "variational: qnn", ("control", None, None): "control"}
GREY_DARK, GREY_LIGHT, ACCENT, INK, INK2 = "#7d7c78", "#c3c2b7", "#2a78d6", "#0b0b0b", "#52514e"


def model_color(cfg: str) -> str:
    return ACCENT if cfg == MAIN else (GREY_DARK if cfg == "v4_phys" else GREY_LIGHT)


def gen_values(values, cfg, split, metric="r2"):
    exp, name, fam = split
    out = []
    for key, per in values.items():
        if key[0] != exp or key[2] != fam:
            continue
        if name is None:            # control: random_seedN
            if not key[1].startswith("random_seed"):
                continue
        elif key[1] != name:
            continue
        out += [v[metric] for v in per.get(cfg, {}).values() if v.get(metric) is not None]
    return out


def fmt(v):
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.3f} (1)"
    return f"{statistics.fmean(v):.3f} ± {statistics.stdev(v):.3f} ({len(v)})"


# --------------------------------------------------------------------------- calibration

def calibration():
    rk = EVAL / "calibration_shift/results_k"
    if not (rk / "truth.json").is_file() or not (rk / "predictions.json").is_file():
        return {}
    T = json.loads((rk / "truth.json").read_text())
    P = json.loads((rk / "predictions.json").read_text())
    names = sorted(T)
    devs = ("EQE1_Top", "EQE1_Bottom", "QExa20")
    best = lambda n, k: max([f for f in T[n][k] if f is not None], default=None)
    groups = {"local": ("shuffle", "jitterA", "jitterB", "jitterS"),
              "moderate": ("x0.7", "x0.8", "x1.2", "x1.3"), "large": ("x0.5", "x2")}
    res = defaultdict(lambda: defaultdict(list))   # cfg -> stat -> [per seed]
    for m, pred in P.items():
        cfg = m.split("@")[0]
        for g, vs in groups.items():
            r2s, trk = [], []
            rows_all = []
            for d in devs:
                ref = f"{d}/orig"
                for v in vs:
                    k = f"{d}/{v}"
                    ok = [n for n in names if best(n, k) is not None and best(n, ref) is not None]
                    t = np.array([best(n, k) for n in ok]); p = np.array([pred[n][k] for n in ok])
                    r2s.append(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
                    keep = [n for n in ok if best(n, k) > 0.01 and best(n, ref) > 0.01]
                    dt_ = np.log([best(n, k) / best(n, ref) for n in keep])
                    dp = np.log([max(pred[n][k], 1e-6) / max(pred[n][ref], 1e-6) for n in keep])
                    trk.append(np.corrcoef(dp, dt_)[0, 1])
                    rows_all += [(best(n, k), best(n, ref), pred[n][k], pred[n][ref]) for n in ok]
            a = np.array(rows_all)
            big = np.abs(a[:, 0] - a[:, 1]) > 0.02
            rel = np.abs((a[:, 2] - a[:, 3]) - (a[:, 0] - a[:, 1]))[big] / np.abs(a[:, 0] - a[:, 1])[big]
            res[cfg][f"{g}_r2"].append(float(np.mean(r2s)))
            res[cfg][f"{g}_worst"].append(float(np.min(r2s)))
            res[cfg][f"{g}_trk"].append(float(np.nanmean(trk)))
            res[cfg][f"{g}_rel"].append(float(rel.mean()))
            res[cfg][f"{g}_mae"].append(float(np.abs(a[:, 2] - a[:, 0]).mean()))
    return res


# --------------------------------------------------------------------------- figures

def fig_generalization(values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = SPLITS[:3] + SPLITS[4:]
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=150)
    w = 0.26
    for i, cfg in enumerate(COMPARE):
        for j, s in enumerate(splits):
            v = gen_values(values, cfg, s)
            if not v:
                continue
            m = statistics.fmean(v); sd = statistics.stdev(v) if len(v) > 1 else 0
            x = j + (i - 1) * w
            ax.errorbar(x, m, yerr=sd, fmt="o", ms=7, color=model_color(cfg), ecolor=model_color(cfg),
                        elinewidth=2, capsize=0, zorder=3)
            ax.text(x + 0.05, m, f"{m:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(splits)), [SPLIT_NAME[s] for s in splits], fontsize=9, color=INK)
    ax.set_ylim(0.5, 1.03); ax.set_ylabel("test R² (dot = mean, line = ± std, 3 seeds)", fontsize=8, color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", zorder=0); ax.tick_params(colors=INK2, labelsize=8)
    for j in range(1, len(splits)):
        ax.axvline(j - 0.5, color="#efeee9", lw=1, zorder=0)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=model_color(c)) for c in COMPARE]
    ax.legend(handles, [MODELS[c][0] for c in COMPARE], frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.set_title("v8a is close to the best model on every split", fontsize=11, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "generalization.png"); plt.close(fig)


def fig_calibration(cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfgs = [c for c in MODELS if c in cal]
    if not cfgs:
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.2), dpi=150)
    for i, c in enumerate(cfgs):
        v = cal[c]["local_trk"]; m = statistics.fmean(v); sd = statistics.stdev(v) if len(v) > 1 else 0
        ax.barh(i, m, 0.62, color=model_color(c), zorder=2)
        if sd:
            ax.errorbar(m, i, xerr=sd, color=INK2, lw=1, capsize=2, zorder=3)
        ax.text(max(m, 0) + sd + 0.015, i, f"{m:.2f}" + (f" ({len(v)} seeds)" if len(v) > 1 else ""),
                va="center", fontsize=7, color=INK2)
    ax.axvline(0.85, color=INK2, ls="--", lw=1)
    ax.text(0.845, len(cfgs) - 0.4, "noise ceiling ≈ 0.85", ha="right", fontsize=7, color=INK2)
    ax.axvline(0, color="#b0afa9", lw=0.8)
    ax.set_yticks(range(len(cfgs)), [MODELS[c][0] for c in cfgs], fontsize=8, color=INK)
    ax.invert_yaxis(); ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel("tracking of local calibration changes (corr. predicted vs true Δlog F)", fontsize=8, color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title("Only the v7/v8 family follows changes of single qubits", fontsize=11, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "calibration_tracking.png"); plt.close(fig)


def timing():
    f = EVAL / "timing_v8/timing_scaling_threads1.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    rows = d["rows"]
    enc = np.array([r["enc_new"] for r in rows])
    out = []
    for n in (3, 6, 12, 18, 27):
        exh = np.array([sum(r["compile"][:n]) for r in rows])
        pr = enc + np.array([r["fwd"][str(n)] if str(n) in r["fwd"] else r["fwd"][n] for r in rows])
        out.append((n, float(np.median(exh)) * 1e3, float(np.median(pr)) * 1e3, float(np.median(exh / pr))))
    return out, float(np.median(enc)) * 1e3, float(np.median([r["enc_old"] for r in rows])) * 1e3, len(rows)


def fig_timing(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = t[0]
    fig, ax = plt.subplots(figsize=(8.4, 3.2), dpi=150)
    n = [r[0] for r in rows]
    ax.plot(n, [r[1] for r in rows], "-o", color=GREY_DARK, lw=2, ms=4, label="exhaustive compile + fidelity")
    ax.plot(n, [r[2] for r in rows], "-o", color=ACCENT, lw=2, ms=4, label="predictor (encode + v8 forward)")
    ax.text(n[-1], rows[-1][1] + 15, f"{rows[-1][1]:.0f} ms", ha="right", fontsize=8, color=INK)
    ax.text(n[-1], rows[-1][2] + 25, f"{rows[-1][2]:.0f} ms  ({rows[-1][3]:.1f}× faster)", ha="right", fontsize=8, color=ACCENT)
    ax.set_xticks(n); ax.set_xlabel("number of devices", fontsize=8, color=INK2)
    ax.set_ylabel("ms per circuit (median)", fontsize=8, color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0"); ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Exhaustive compilation grows with the devices, the predictor barely", fontsize=11, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "timing.png"); plt.close(fig)


# --------------------------------------------------------------------------- page

def main() -> None:
    FIG.mkdir(exist_ok=True)
    values = summarize.collect()
    cal = calibration()
    tim = timing()
    fig_generalization(values)
    fig_calibration(cal)
    if tim:
        fig_timing(tim)

    L = []
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    L += [f"# Results — fidelity predictor study", "",
          f"*Generated by `evaluations/results_report.py` on {now} from the result files; do not edit by hand.*", ""]
    L += [f"**Main model: {MODELS[MAIN][0]}** — {MODELS[MAIN][1]}.", ""]

    L += ["## Models", "", "| Model | Idea | Status |", "| --- | --- | --- |"]
    L += [f"| {s} | {idea} | {st} |" for c, (s, idea, st) in MODELS.items()]
    L += [""]

    L += ["## Generalization to unseen families", "", "![generalization](figures/generalization.png)", "",
          "Test R², mean ± std (number of seeds).", "",
          "| Model | " + " | ".join(SPLIT_NAME[s] for s in SPLITS) + " |",
          "| --- | " + " | ".join("---" for _ in SPLITS) + " |"]
    for c in MODELS:
        cells = [fmt(gen_values(values, c, s)) for s in SPLITS]
        if all(x == "–" for x in cells):
            continue
        name = f"**{MODELS[c][0]}**" if c == MAIN else MODELS[c][0]
        L += [f"| {name} | " + " | ".join(cells) + " |"]
    L += [""]

    if cal:
        L += ["## Calibration changes (low-noise test)", "", "![calibration](figures/calibration_tracking.png)", "",
              "457 unseen circuits, truth = best of 5 compilations, control models. *Local* = single qubits/couplers "
              "change (shuffle, jitter σ 0.2–0.3); *moderate* = all errors ×0.7–1.3. Relative error 1.0 = as good as "
              "predicting no change. Mean over seeds.", "",
              "| Model | seeds | local: tracking | local: worst R² | local: rel. error of ΔF | local: MAE of F | moderate: R² | moderate: tracking |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in MODELS:
            if c not in cal:
                continue
            s = cal[c]; mean = lambda k: statistics.fmean(s[k])
            name = f"**{MODELS[c][0]}**" if c == MAIN else MODELS[c][0]
            L += [f"| {name} | {len(s['local_trk'])} | {mean('local_trk'):.2f} | {min(s['local_worst']):.3f} | "
                  f"{mean('local_rel'):.2f} | {mean('local_mae'):.3f} | {mean('moderate_r2'):.3f} | {mean('moderate_trk'):.2f} |"]
        L += [""]

    if tim:
        rows, enc_new, enc_old, n = tim
        L += ["## Inference time vs exhaustive compilation", "", "![timing](figures/timing.png)", "",
              f"{n} circuits, 1 thread, circuit already in memory; devices = 3 originals + calibration variants. "
              f"Graph encoding {enc_old:.1f} → {enc_new:.1f} ms (`gsv3/fast_encoding.py`).", "",
              "| Devices | Exhaustive (ms) | Predictor (ms) | Speed-up (median per circuit) |", "| --- | --- | --- | --- |"]
        L += [f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f}× |" for r in rows]
        L += [""]

    dec = EVAL / "generalization_v9" / "DECISION.md"
    if dec.is_file():
        L += ["#" + dec.read_text().strip(), ""]
    L += ["Details, diagnostics and decisions: `evaluations/TODO.md`.", ""]
    OUT.write_text("\n".join(L))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
