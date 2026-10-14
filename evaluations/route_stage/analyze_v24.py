#!/usr/bin/env python3
"""Where v24b loses accuracy on routed circuits: per-case rows of the trained checkpoints and a report.

rows:   for one split and seed, every test (circuit, device) with circuit features (family, qubits, gates, logical
        depth, routing overhead), the durations of every stage (dataset label, Sabre final, replay, representation
        oracle, SWAP-only and wait-only substitutions, stage 1 mean / sampled, stage 2), wait statistics (sign, share,
        critical path of the oracle schedule, unmatched gates) -> results/v24b_<split>_s<seed>/rows_ext.pt
report: pools the rows of all seeds -> results/report_analysis_v24.md (waterfall, breakdowns, tail, representation
        ceiling, waits, stage 2 vs overhead, device ranking)
    python analyze_v24.py rows --split qaoa --seed 5 --device cuda
    python analyze_v24.py report
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import stage1 as S

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SPLITS = {"qaoa": "qaoa", "variational": "variational", "control": "control", "loso:dense,complete": "loso_dense_complete"}


def fam(n):
    f = n.split("/")[0]
    return f"synth_{n.split('/')[1].split('_')[1]}" if f == "synth" else f


def critical_path(gq, t, s, wait, use_wait):
    """S.schedule_np that also returns the gates on the longest path (backtracked through the binding wire)."""
    last, owner = {}, {}
    end, pred = np.zeros(len(gq)), np.full(len(gq), -1)
    for i in range(len(gq)):
        qs = [int(q) for q in gq[i] if q >= 0]
        if not qs:
            continue
        cand = [last.get(q, 0.0) + s[i, k] * S.T_SWAP for k, q in enumerate(qs)]
        kb = int(np.argmax(cand))
        st = cand[kb] + (wait[i] if use_wait[i] else 0.0)
        lq = [last.get(q, 0.0) for q in qs]
        if max(lq) > st:                                   # negative wait: the plain wire end binds
            kb = int(np.argmax(lq)); st = lq[kb]
        pred[i] = owner.get(qs[kb], -1)
        end[i] = st + t[i]
        for q in qs:
            last[q] = end[i]; owner[q] = i
    cp = np.zeros(len(gq), bool)
    i = int(np.argmax(end))
    while i >= 0:
        cp[i] = True; i = pred[i]
    return float(end.max()), cp


@torch.no_grad()
def rows_for(model, items, ds, device, mc, rng, bs=32):
    out, gates = [], []
    for s0 in range(0, len(items), bs):
        b = next(iter(DataLoader(items[s0: s0 + bs], batch_size=bs))).to(device)
        rate, wait, c2 = (o.cpu() for o in model(b))
        ptr = b.ptr.tolist()
        for i in range(len(ptr) - 1):
            a, e = ptr[i], ptr[i + 1]
            it = items[s0 + i]
            gq = it.gate_qubits.numpy(); t = it.t.numpy()
            meas = it.x[:, S.GATE2IDX_MEASURE].numpy() > 0.5
            n_op = (gq >= 0).sum(1)
            elig = (n_op >= 2) | meas
            slot = (gq >= 0) & elig[:, None]
            zero = np.zeros((len(gq), 3)); nw = np.zeros(len(gq))
            logical = S.schedule_np(gq, t, zero, nw, elig)
            lab = ds.get(it.name) if ds is not None else None
            base = {"name": it.name, "fam": fam(it.name), "nq": int(it.n_qubits), "ng": len(gq), "n2q": int((n_op >= 2).sum()),
                    "logical": logical}
            for d in range(len(S.DEVS)):
                k2 = math.exp(float(c2[i, d]))
                row = {**base, "d": d, "rf": bool(it.rf[0, d]), "final": float(it.final[0, d]),
                       "final_ds": float(lab["duration"][d]) * 1e6 if lab is not None else float("nan")}
                if row["rf"]:
                    row.update(pred1=logical, pred1_mc=logical, pred=logical * k2, pred_mc=logical * k2)
                    out.append(row); continue
                if not bool(it.routed[0, d]):
                    continue
                r = rate[a:e, d].numpy() * slot
                w = wait[a:e, d].numpy() * S.T_SWAP
                st_ = it.s[:, d].numpy() * slot; wt = it.wait[:, d].numpy(); m = it.m[:, d].numpy()
                osw, cp = critical_path(gq, t, st_, wt, elig)
                p1 = S.schedule_np(gq, t, r, w, elig)
                pk = float(np.median(S.schedule_k(gq, t, rng.poisson(np.repeat(r[..., None], mc, -1)), w, elig)))
                mm = m & elig
                row.update(
                    replay=float(it.replay[0, d]), replay_c2=float(it.replay[0, d]) * k2, oracle_sw=osw,
                    oracle_s=S.schedule_np(gq, t, st_, wt, np.zeros(len(gq), bool)),
                    s_pred_w_true=S.schedule_np(gq, t, r, wt, elig), s_true_w_pred=S.schedule_np(gq, t, st_, w, elig),
                    pred1=p1, pred1_mc=pk, pred=p1 * k2, pred_mc=pk * k2, k2=k2,
                    sw_true=float(st_.sum()), sw_pred=float(r.sum()), sw_pred_zero=float(r[st_ == 0].sum()),
                    n_elig=int(elig.sum()), n_unmatched=int((elig & ~m).sum()),
                    w_pos=float(wt[mm & (wt > 1e-6)].sum()), w_neg=float(-wt[mm & (wt < -1e-6)].sum()),
                    n_wpos=int((mm & (wt > 1e-6)).sum()), n_wneg=int((mm & (wt < -1e-6)).sum()), n_m=int(mm.sum()),
                    w_pred_sum=float(w[elig].sum()), n_cp=int(cp.sum()),
                    w_true_cp=float(wt[cp & elig].sum()), w_pred_cp=float(w[cp & elig].sum()),
                    w_abs_err_cp=float(np.abs(w - wt)[cp & mm].sum()), w_abs_err_off=float(np.abs(w - wt)[~cp & mm].sum()),
                    sw_true_cp=float(st_[cp].sum()), sw_pred_cp=float(r[cp].sum()))
                out.append(row)
                idx = np.nonzero(mm)[0]
                if len(idx):
                    idx = rng.choice(idx, min(40, len(idx)), replace=False)
                    gates += [(row["fam"], d, float(wt[j]), float(w[j]), bool(cp[j])) for j in idx]
    return out, gates


def cmd_rows(args):
    torch.set_num_threads(args.threads)
    data = S.load_all(HERE / "results/stage1.pt")
    _, test_n = S.split_names(args.split, sorted(data))
    out = RES / f"v24b_{SPLITS[args.split]}_s{args.seed}"
    model = S.Stage1(data[test_n[0]].x.size(1) + 9 + 12).to(args.device)
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
    model.eval()
    ds = torch.load(S.EVAL / "compile_check/results/esp_labels.pt", weights_only=False)
    items = [data[n] for n in test_n][: args.max_test or None]
    rows, gates = rows_for(model, items, ds, args.device, args.mc, np.random.default_rng(args.seed))
    torch.save({"rows": rows, "gates": gates}, out / "rows_ext.pt")
    print(f"{args.split} s{args.seed}: {len(rows)} rows, {len(gates)} gate samples -> {out / 'rows_ext.pt'}", flush=True)


# ---------------------------------------------------------------- report
def e(p, f):
    return np.abs(p / f - 1)


def cell(p, f):
    if len(f) == 0:
        return "–"
    x = e(p, f)
    return f"{100 * np.median(x):.1f}% / {100 * np.percentile(x, 90):.0f}% ({np.median(p / f):.2f})"


def md(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def bins_of(x, edges, fmt="{:g}"):
    lab = [f"{fmt.format(a)}–{fmt.format(b)}" if np.isfinite(b) else f"≥{fmt.format(a)}" for a, b in zip(edges[:-1], edges[1:])]
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(lab) - 1), lab


def breakdown(rt, key, idx, lab, models=("pred1_mc", "pred_mc")):
    f = col(rt, "final")
    err = e(col(rt, "pred1_mc"), f)
    tail = err >= np.percentile(err, 90)
    out = []
    for b, name in enumerate(lab):
        m = idx == b
        if m.sum() < 10:
            continue
        out.append([name, int(m.sum()), *[cell(col(rt, k)[m], f[m]) for k in models], f"{100 * tail[m].mean():.0f}%"])
    return md([key, "n", *[f"{k}: med / p90 (bias)" for k in models], "share in top-10% errors"], out)


def cmd_report(_args):
    L = ["# v24b analysis: routed-circuit duration", "",
         "Rows of the 3 v24b seeds pooled (each seed scores the same test cases). Cells: median |pred/true − 1| / p90 "
         "(median pred/true), vs Sabre's final duration unless noted. *pred1* = stage 1 only, *pred* = with stage 2, "
         "*_mc* = median schedule over 24 Poisson samples of the SWAPs.", ""]
    rk_rows = []
    for split, stem in SPLITS.items():
        rows, gates = [], []
        for f in sorted(RES.glob(f"v24b_{stem}_s*/rows_ext.pt")):
            d = torch.load(f, weights_only=False); gates += d["gates"]
            rows += [{**r, "seed": f.parent.name} for r in d["rows"]]
        if not rows:
            continue
        rt = [r for r in rows if not r["rf"]]
        f = col(rt, "final")
        L += [f"## {split} ({len(rt)} routed / {len(rows) - len(rt)} routing-free cases, pooled seeds)", ""]
        # A. waterfall
        hasds = np.isfinite(col(rt, "final_ds"))
        wf = [["Sabre final vs dataset label (label mismatch)", cell(f[hasds], col(rt, "final_ds")[hasds]) if hasds.any() else "–"],
              ["Sabre's routed schedule replayed (no optimisation)", cell(col(rt, "replay"), f)],
              ["  + stage 2 (on the true schedule)", cell(col(rt, "replay_c2"), f)],
              ["true SWAPs + true waits (representation ceiling)", cell(col(rt, "oracle_sw"), f)],
              ["true SWAPs, no waits", cell(col(rt, "oracle_s"), f)],
              ["predicted SWAPs + true waits", cell(col(rt, "s_pred_w_true"), f)],
              ["true SWAPs + predicted waits", cell(col(rt, "s_true_w_pred"), f)],
              ["stage 1, mean rates", cell(col(rt, "pred1"), f)],
              ["stage 1, sampled", cell(col(rt, "pred1_mc"), f)],
              ["stage 1 sampled + stage 2", cell(col(rt, "pred_mc"), f)]]
        if hasds.any():
            fd = col(rt, "final_ds")[hasds]
            wf += [["stage 1 sampled vs dataset", cell(col(rt, "pred1_mc")[hasds], fd)],
                   ["stage 1 sampled + stage 2 vs dataset", cell(col(rt, "pred_mc")[hasds], fd)]]
        L += ["### Error by stage", "", md(["what", "med / p90 (bias)"], wf), ""]
        # B. breakdowns
        ovh = col(rt, "sw_true") / np.maximum(col(rt, "n2q"), 1)
        L += ["### Where the error is (stage 1 sampled vs with stage 2)", ""]
        L += [breakdown(rt, "device", col(rt, "d").astype(int), list(S.DEVS)), ""]
        L += [breakdown(rt, "qubits", *bins_of(col(rt, "nq"), [0, 6, 11, 16, 21, np.inf])), ""]
        L += [breakdown(rt, "2q gates", *bins_of(col(rt, "n2q"), [0, 20, 50, 100, 200, 500, np.inf])), ""]
        L += [breakdown(rt, "SWAPs per 2q gate", *bins_of(ovh, [0, 0.25, 0.5, 1, 2, np.inf])), ""]
        L += [breakdown(rt, "routing share of duration (1 − logical / final)", *bins_of(1 - col(rt, "logical") / f, [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], "{:.1f}")), ""]
        fams = sorted({r["fam"] for r in rt})
        if len(fams) > 1:
            fi = np.array([fams.index(r["fam"]) for r in rt])
            L += [breakdown(rt, "family", fi, fams), ""]
        # C. representation ceiling
        unm = col(rt, "n_unmatched") / np.maximum(col(rt, "n_elig"), 1)
        neg = col(rt, "w_neg") / f
        ce = e(col(rt, "oracle_sw"), f) - e(col(rt, "replay"), f)
        rep = []
        for name, x, edges in (("unmatched share of gates", unm, [0, 1e-9, 0.05, 0.2, 0.5, np.inf]),
                               ("negative waits / duration", neg, [0, 1e-9, 0.02, 0.05, 0.1, np.inf])):
            idx, lab = bins_of(x, edges)
            for b, nm in enumerate(lab):
                m = idx == b
                if m.sum() >= 10:
                    rep.append([name, nm, int(m.sum()), cell(col(rt, "oracle_sw")[m], f[m]), cell(col(rt, "replay")[m], f[m]),
                                f"{100 * np.median(ce[m]):+.1f}"])
        L += ["### Representation ceiling: true SWAPs + true waits vs Sabre's replayed schedule", "",
              f"Cases with unmatched gates: {100 * (unm > 0).mean():.0f}%; with negative waits: {100 * (neg > 0).mean():.0f}%.", "",
              md(["by", "bin", "n", "oracle (SWAPs + waits)", "replay", "median extra error, points"], rep), ""]
        # D. waits
        g = np.array([(x[2], x[3], x[4]) for x in gates], float) if gates else np.zeros((0, 3))
        cpm = g[:, 2] > 0.5 if len(g) else np.zeros(0, bool)

        def corr(a, b):
            return f"{np.corrcoef(a, b)[0, 1]:.2f}" if len(a) > 2 and a.std() > 0 and b.std() > 0 else "–"
        wst = [["matched gates with a positive / negative wait", f"{100 * col(rt, 'n_wpos').sum() / col(rt, 'n_m').sum():.1f}% / {100 * col(rt, 'n_wneg').sum() / col(rt, 'n_m').sum():.1f}%"],
               ["positive waits / duration (median over cases)", f"{100 * np.median(col(rt, 'w_pos') / f):.1f}%"],
               ["net waits on the critical path / duration: true / predicted (median)", f"{100 * np.median(col(rt, 'w_true_cp') / f):.1f}% / {100 * np.median(col(rt, 'w_pred_cp') / f):.1f}%"],
               ["negative waits / duration (median)", f"{100 * np.median(col(rt, 'w_neg') / f):.1f}%"],
               ["predicted SWAPs on that path / true", f"{np.median(col(rt, 'sw_pred_cp') / np.maximum(col(rt, 'sw_true_cp'), 1e-6)):.2f}"],
               ["|wait error| on the critical path / all |wait error|", f"{100 * col(rt, 'w_abs_err_cp').sum() / max(col(rt, 'w_abs_err_cp').sum() + col(rt, 'w_abs_err_off').sum(), 1e-9):.1f}% (path holds {100 * np.median(col(rt, 'n_cp') / col(rt, 'ng')):.0f}% of the gates)"],
               ["per-gate corr(pred, true wait): all / critical / off-path", f"{corr(g[:, 0], g[:, 1])} / {corr(g[cpm, 0], g[cpm, 1])} / {corr(g[~cpm, 0], g[~cpm, 1])}"],
               ["true waits that are zero: predicted |wait| mean (µs)", f"{np.abs(g[np.abs(g[:, 0]) < 1e-6, 1]).mean():.3f}" if len(g) else "–"],
               ["SWAP mass on slots with no SWAP", f"{100 * col(rt, 'sw_pred_zero').sum() / col(rt, 'sw_pred').sum():.0f}%"]]
        L += ["### Waits", "", md(["", "value"], wst), ""]
        # E. stage 2 vs overhead
        idx, lab = bins_of(ovh, [0, 0.25, 0.5, 1, 2, np.inf])
        s2 = []
        for b, nm in enumerate(lab):
            m = idx == b
            if m.sum() >= 10:
                s2.append([nm, int(m.sum()), cell(col(rt, "replay")[m], f[m]), cell(col(rt, "replay_c2")[m], f[m]),
                           cell(col(rt, "pred1_mc")[m], f[m]), cell(col(rt, "pred_mc")[m], f[m]),
                           f"{np.median(col(rt, 'k2')[m]):.3f}", f"{np.median(f[m] / col(rt, 'replay')[m]):.3f}"])
        L += ["### Stage 2 by routing overhead (SWAPs per 2q gate)", "",
              md(["overhead", "n", "replay", "replay + stage 2", "stage 1 sampled", "+ stage 2", "stage-2 factor (median)",
                  "true final / replay (median)"], s2), ""]
        # F. device ranking: per circuit and seed, the devices with a label (routed or routing-free)
        by = defaultdict(list)
        for r in rows:
            if r["rf"] or "replay" in r:
                by[(r["name"], r["seed"])].append(r)
        rk = {k: [] for k in ("pred1_mc", "pred_mc")}
        rk_ds = {k: [] for k in ("pred1_mc", "pred_mc")}
        for key, rs in by.items():
            if len(rs) < 2:
                continue
            fin = np.array([r["final"] for r in rs])
            fds = np.array([r["final_ds"] for r in rs])
            for k in rk:
                p = np.array([r[k] for r in rs])
                i, j = np.triu_indices(len(rs), 1)
                far = np.abs(fin[i] / fin[j] - 1) > 0.01                                 # ties (identical chips) left out
                rk[k].append((fin[np.argmin(p)] <= fin.min() * 1.001, fin[np.argmin(p)] / fin.min() - 1,
                              np.mean(np.sign(p[i] - p[j])[far] == np.sign(fin[i] - fin[j])[far]) if far.any() else np.nan,
                              np.median(np.abs(np.log(p[i] / p[j]) - np.log(fin[i] / fin[j]))), all(r["rf"] for r in rs),
                              np.mean(fin / fin.min() - 1)))
                if np.isfinite(fds).all():
                    rk_ds[k].append((fds[np.argmin(p)] <= fds.min() * 1.001, fds[np.argmin(p)] / fds.min() - 1, 0, 0, all(r["rf"] for r in rs), 0))
        for k, v in rk.items():
            if not v:
                continue
            a = np.array(v, float)
            mixed = a[:, 4] < 0.5
            vd = np.array(rk_ds[k], float) if rk_ds[k] else None
            rk_rows.append([split, k, len(a), int(mixed.sum()), f"{100 * a[mixed, 0].mean():.0f}%", f"{100 * a[mixed, 1].mean():.2f}%",
                            f"{100 * a[mixed, 5].mean():.2f}%", f"{100 * np.nanmean(a[mixed, 2]):.0f}%", f"{100 * np.median(a[mixed, 3]):.1f}%",
                            f"{100 * vd[vd[:, 4] < 0.5, 0].mean():.0f}% / {100 * vd[vd[:, 4] < 0.5, 1].mean():.2f}%" if vd is not None and (vd[:, 4] < 0.5).any() else "–"])
    L += ["## Device choice by duration (circuits with ≥ 2 devices and at least one routed)", "",
          "Fastest device picked by the prediction vs the true fastest (Sabre final; last column vs the dataset label). "
          "Right = picked device within 0.1% of the fastest (ties count as right). Regret = extra duration of the picked device, mean; "
          "random = mean regret of a random device. Pairs in right order: pairs more than 1% apart. "
          "Ratio error = median |log(pred_a/pred_b) − log(true_a/true_b)|.", "",
          md(["split", "model", "circuits × seeds", "with a routed device", "right device", "regret", "random regret", "pairs in right order",
              "ratio error", "vs dataset: right / regret"], rk_rows), ""]
    (RES / "report_analysis_v24.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("rows", "report"))
    ap.add_argument("--split", default="qaoa", choices=list(SPLITS))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mc", type=int, default=24)
    ap.add_argument("--max-test", type=int, default=0)
    args = ap.parse_args()
    cmd_rows(args) if args.cmd == "rows" else cmd_report(args)


if __name__ == "__main__":
    main()
