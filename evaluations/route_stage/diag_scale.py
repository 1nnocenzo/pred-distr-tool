#!/usr/bin/env python3
"""Why does v24 stage 1 over-count SWAPs (×1.4–2) yet under-estimate the routed duration?

On a trained stage-1 checkpoint, for routed test (and a train sample of) cases:
  - per-slot calibration of the SWAP rate (mean predicted vs mean true, by rate bin) and the predicted mass on slots
    that carry no SWAP (floor of the softplus over many slots);
  - mean vs median calibration of the per-case SWAP total (Poisson NLL fits the mean; a right-skewed total has
    median ratio > 1 even when the mean is right);
  - duration from the mean rates (as stage1.evaluate) vs from Poisson samples of the SWAPs (median / mean over K
    schedules): if the samples close the gap, the under-estimate is Jensen (schedule of the means < mean schedule);
  - attribution: predicted SWAPs + true waits, true SWAPs + predicted waits.
    python diag_scale.py --split qaoa --ckpt results/v24_qaoa/model.pt --out results/diag_scale_qaoa.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import stage1 as S

BINS = [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, np.inf]


def err(p, f):
    e = np.abs(p / f - 1)
    return {"err": float(np.median(e)), "p90": float(np.percentile(e, 90)), "bias": float(np.median(p / f)), "w10": float((e < 0.1).mean())}


@torch.no_grad()
def run(model, items, K, rng, bs=32):
    model.eval()
    cal_r, cal_s = [], []
    cases = []
    for s0 in range(0, len(items), bs):
        b = next(iter(DataLoader(items[s0: s0 + bs], batch_size=bs)))
        rate, wait, _ = model(b)
        ptr = b.ptr.tolist()
        for i in range(len(ptr) - 1):
            a, e = ptr[i], ptr[i + 1]
            it = items[s0 + i]
            gq = it.gate_qubits.numpy(); t = it.t.numpy()
            meas = it.x[:, S.GATE2IDX_MEASURE].numpy() > 0.5
            elig = ((gq >= 0).sum(1) >= 2) | meas
            slot = (gq >= 0) & elig[:, None]
            for d in range(len(S.DEVS)):
                if not bool(it.routed[0, d]):
                    continue
                r = rate[a:e, d].numpy() * slot
                w = wait[a:e, d].numpy() * S.T_SWAP
                s_true = it.s[:, d].numpy() * slot; w_true = it.wait[:, d].numpy()
                cal_r.append(r[slot]); cal_s.append(s_true[slot])
                samp = rng.poisson(np.repeat(r[..., None], K, -1))
                dk = S.schedule_k(gq, t, samp, w, elig)
                cases.append({"final": float(it.final[0, d]), "replay": float(it.replay[0, d]),
                              "mean": S.schedule_np(gq, t, r, w, elig),
                              "mc_med": float(np.median(dk)), "mc_mean": float(dk.mean()),
                              "s_pred_w_true": S.schedule_np(gq, t, r, w_true, elig),
                              "s_true_w_pred": S.schedule_np(gq, t, s_true, w, elig),
                              "oracle_sw": S.schedule_np(gq, t, s_true, w_true, elig),
                              "sum_r": float(r.sum()), "sum_s": float(s_true.sum()),
                              "sum_r_on_zero": float(r[s_true == 0].sum()), "n_slots": int(slot.sum()),
                              "sum_samp_med": float(np.median(samp.sum((0, 1))))})
    return np.concatenate(cal_r), np.concatenate(cal_s), cases


def report(cal_r, cal_s, cases):
    out = {"n_cases": len(cases), "n_slots": int(len(cal_r))}
    tab = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (cal_r >= lo) & (cal_r < hi)
        if m.any():
            tab.append({"bin": f"[{lo},{hi})", "slots": int(m.sum()), "mean_rate": float(cal_r[m].mean()),
                        "mean_true": float(cal_s[m].mean()), "share_pred_mass": float(cal_r[m].sum() / cal_r.sum())})
    out["slot_calibration"] = tab
    sr = np.array([c["sum_r"] for c in cases]); ss = np.array([c["sum_s"] for c in cases])
    rz = np.array([c["sum_r_on_zero"] for c in cases])
    out["totals"] = {"mean_pred/mean_true": float(sr.mean() / ss.mean()), "median_ratio": float(np.median(sr / np.maximum(ss, 1))),
                     "median_true": float(np.median(ss)), "median_pred": float(np.median(sr)),
                     "share_pred_on_zero_slots": float(rz.sum() / sr.sum()),
                     "median_slots": float(np.median([c["n_slots"] for c in cases])),
                     "share_true_slots_nonzero": float((cal_s > 0).mean())}
    f = np.array([c["final"] for c in cases])
    out["duration_vs_final"] = {k: err(np.array([c[k] for c in cases]), f)
                                for k in ("mean", "mc_med", "mc_mean", "s_pred_w_true", "s_true_w_pred", "oracle_sw", "replay")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1500, help="test circuits (random subset)")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)
    data = S.load_all(S.HERE / "results/stage1.pt")
    train_n, test_n = S.split_names(a.split, sorted(data))
    f_in = data[test_n[0]].x.size(1) + 9 + 12
    model = S.Stage1(f_in)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    res = {"split": a.split, "ckpt": str(a.ckpt), "K": a.K}
    for name, pool, n in (("test", test_n, a.n), ("train", train_n, a.n_train)):
        sel = list(rng.choice(pool, min(n, len(pool)), replace=False))
        res[name] = report(*run(model, [data[x] for x in sel], a.K, rng))
        print(name, json.dumps(res[name], indent=1), flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
