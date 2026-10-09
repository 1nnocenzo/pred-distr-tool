"""Why is v6 (lambda 0.1) worse on LOGO variational?  Bias, placement on the test set,
parts of -log F vs the compiled circuits, for v4 / v5 / v6 (seed 5)."""
import sys, re, numpy as np, torch
sys.path.insert(0, "/home/atudisco/pred-distr-tool/evaluations/diagnostics_v4v5")
import diagnose as D
from torch_geometric.loader import DataLoader
EVAL = D.EVAL
params = D.load_hparams()
devices = D.load_devices(D.paths.V2_DEVICE_GRAPHS, D.DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
names, graphs, T = D.split_test("variational")
fam = np.array([n.split("/")[0] for n in names])
lay = torch.load(EVAL / "compile_check/results/layouts_all/compiler_layouts.pt", weights_only=False)
tp, tc = D.true_parts(names)
ok = ~np.isnan(tp[:, 0, 0])
runs = {"v4_phys": ("v4_phys", EVAL / "generalization_v4/results/v4_phys"),
        "v5": ("v5_sinkhorn", EVAL / "generalization_v5/results/v5_sinkhorn"),
        "v6_l001": ("v5_sinkhorn", EVAL / "generalization_v6/results/v6_l001"),
        "v6_l01": ("v5_sinkhorn", EVAL / "generalization_v6/results/v6_l01")}
print(f"test circuits {len(names)}, with compiled QPY {ok.sum()}")
for f in sorted(set(fam)):
    s = ok & (fam == f)
    print(f"  truth {f:14} F {np.exp(-tp[s].sum(-1)).mean():.3f}  parts 1q/meas/2q " + "/".join(f"{tp[s][..., k].mean():.3f}" for k in range(3))
          + "  counts r/meas/cz " + "/".join(f"{tc[s][..., k].mean():.0f}" for k in range(3)))
for label, (kind, root) in runs.items():
    model = D.load_model(kind, root / "leave_one_group_out/variational/model.pth", devices, params)
    parts, extra = D.parts_of(kind, model, graphs, devices)
    F = D.fid(parts)
    line = f"\n{label:8} R2 {D.r2(F.ravel(), T.ravel()):.3f} bias {np.mean(F - T):+.3f}"
    print(line)
    for f in sorted(set(fam)):
        s = fam == f; so = s & ok
        print(f"  {f:14} R2 {D.r2(F[s].ravel(), T[s].ravel()):6.3f} bias {np.mean(F[s] - T[s]):+.3f}  parts 1q/meas/2q "
              + "/".join(f"{parts[so][..., k].mean():.3f}" for k in range(3)))
    if kind == "v5_sinkhorn":   # placement on the test set vs the compiler's layout
        hit = {f: 0 for f in set(fam)}; tot = {f: 0 for f in set(fam)}; maxp = []
        i = 0
        for a in extra:                                   # (B, D, Qm, P)
            for b in range(a.size(0)):
                n = names[i]; L = lay.get(n); nq = int(graphs[i].n_qubits); i += 1
                maxp.append(a[b, :, :nq].max(-1).values.mean().item())
                if L is None:
                    continue
                pred = a[b, :, :nq].argmax(-1).T           # (nq, D)
                hit[n.split("/")[0]] += int((pred == L).sum()); tot[n.split("/")[0]] += L.numel()
        print("  placement on TEST: max row prob %.3f; argmax = compiler's qubit: " % np.mean(maxp)
              + "  ".join(f"{f} {hit[f] / max(tot[f], 1):.2f}" for f in sorted(hit)))
