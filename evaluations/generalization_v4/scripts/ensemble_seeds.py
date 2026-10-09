"""Seed ensembles: R² of each seed vs. the average of the seeds' predictions (run on a compute node)."""
import json, torch, numpy as np, os
E = "evaluations"
data = torch.load(E + "/generalization_v3/data/graph_dataset_expected_fidelity.pt", weights_only=False)
Y = {d.circuit_name: d.y.view(-1).numpy() for d in data}
def r2(p, t): return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
def path(m, s, exp, sp):
    if m == "v1":
        if s == 5 and exp == "leave_one_family_out":
            return f"{E}/generalization/results/{exp}/{sp}/predictions.json"
        return f"{E}/generalization_v3/results/v1_s{s}/{exp}/{sp}/predictions.json"
    root = "generalization_v4" if m.startswith("v4") else "generalization_v3"
    suf = "" if s == 5 else f"_s{s}"
    return f"{E}/{root}/results/{m}{suf}/{exp}/{sp}/predictions.json"
splits = [("leave_one_group_out", "variational"), ("leave_one_group_out", "vqe"),
          ("leave_one_group_out", "fourier"), ("leave_one_family_out", "qnn"),
          ("leave_one_family_out", "iqpe"), ("leave_one_family_out", "qaoa")]
print(f"{'model':15}{'split':13}{'n':>3}  {'R2 single seeds':24}{'ensemble':>9}   family inside group: singles -> ensemble")
for m in ["v1", "xattn_a00", "xattn_a05", "v4_xattn_mixed", "v4_phys"]:
    for exp, sp in splits:
        preds = [json.load(open(p)) for s in (5, 6, 7) if os.path.isfile(p := path(m, s, exp, sp))]
        if len(preds) < 2:
            continue
        names = sorted(set.intersection(*[set(p) for p in preds]))
        T = np.array([Y[n] for n in names]); P = [np.array([p[n] for n in names]) for p in preds]
        ens = np.mean(P, 0)
        singles = " ".join(f"{r2(p.ravel(), T.ravel()):.3f}" for p in P)
        extra = ""
        if sp in ("variational", "fourier"):
            fam = "qnn" if sp == "variational" else "iqpe"
            idx = [i for i, n in enumerate(names) if n.startswith(fam + "/")]
            extra = f"{fam}: " + " ".join(f"{r2(p[idx].ravel(), T[idx].ravel()):.2f}" for p in P) \
                    + f" -> {r2(ens[idx].ravel(), T[idx].ravel()):.2f}"
        print(f"{m:15}{sp:13}{len(P):>3}  {singles:24}{r2(ens.ravel(), T.ravel()):>9.3f}   {extra}")
