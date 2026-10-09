"""Why does v7 lose on LOFO qnn?  Bias by size, parts of -log F vs the compiled circuits,
operand-distance error, for v4 / v5 / phys_uniform / v7 (seed 5)."""
import sys, re, numpy as np, torch
sys.path.insert(0, "/home/atudisco/pred-distr-tool/evaluations/diagnostics_v4v5")
import diagnose as D
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(D.EVAL / "generalization_v3"))
from gsv3.training import region_distance_loss
EVAL = D.EVAL
params = D.load_hparams()
devices = D.load_devices(D.paths.V2_DEVICE_GRAPHS, D.DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
names, graphs, T = D.split_test("qnn")
nq = np.array([int(re.search(r"_q(\d+)_", n).group(1)) for n in names])
tp, tc = D.true_parts(names)
lay = torch.load(EVAL / "compile_check/results/layouts_all/compiler_layouts.pt", weights_only=False)
runs = {"v4_phys": ("v4_phys", "phys", EVAL / "generalization_v4/results/v4_phys"),
        "v5": ("v5_sinkhorn", "sinkhorn", EVAL / "generalization_v5/results/v5_sinkhorn"),
        "phys_uniform": ("v4_phys", "phys_uniform", EVAL / "generalization_v6/results/v4u_phys"),
        "v7_l001": ("v5_sinkhorn", "sinkhorn", EVAL / "generalization_v7/results/v7_l001"),
        "v7_l01": ("v5_sinkhorn", "sinkhorn", EVAL / "generalization_v7/results/v7_l01")}
bands = [(lo, hi) for lo, hi in ((2, 4), (5, 7), (8, 10), (11, 14), (15, 20)) if ((nq >= lo) & (nq <= hi)).any()]
print(f"qnn test: {len(names)} circuits, qubits {nq.min()}-{nq.max()}")
for lo, hi in bands:
    s = (nq >= lo) & (nq <= hi)
    print(f"  truth q{lo}-{hi}: n={s.sum()} F {T[s].mean():.3f}  parts 1q/meas/2q " +
          "/".join(f"{tp[s][..., k].mean():.3f}" for k in range(3)) +
          "  counts r/meas/cz " + "/".join(f"{tc[s][..., k].mean():.0f}" for k in range(3)))
for label, (fwd, kind, root) in runs.items():
    model = D.build_model(kind, params, devices, torch.device("cpu"))
    model.load_state_dict(torch.load(root / "leave_one_family_out/qnn/model.pth", map_location="cpu"))
    model.eval()
    if kind == "phys_uniform":
        with torch.no_grad():
            F = torch.cat([torch.exp(torch.clamp(model(b, devices), max=0.0))
                           for b in DataLoader(graphs, batch_size=16)]).numpy()
        parts = None
    else:
        parts, _ = D.parts_of(fwd, model, graphs, devices)
        F = D.fid(parts)
    print(f"\n{label:13} R2 {D.r2(F.ravel(), T.ravel()):.3f}  bias {np.mean(F - T):+.3f}  MAE {np.abs(F - T).mean():.3f}")
    for lo, hi in bands:
        s = (nq >= lo) & (nq <= hi)
        line = f"  q{lo}-{hi:<3} R2 {D.r2(F[s].ravel(), T[s].ravel()):6.3f} bias {np.mean(F[s] - T[s]):+.3f}"
        if parts is not None:
            line += "  parts 1q/meas/2q " + "/".join(f"{parts[s][..., k].mean():.3f}" for k in range(3))
        print(line)
    if kind == "sinkhorn":
        gs = []
        for g in graphs:
            L = lay.get(g.circuit_name)
            if L is not None:
                g = g.clone(); g.layout = L; gs.append(g)
        tot = {"region_overlap": 0.0, "dist_mse": 0.0}; n = 0
        with torch.no_grad():
            for b in DataLoader(gs, batch_size=16):
                model(b, devices); _, st = region_distance_loss(model, b)
                for k in tot: tot[k] += st[k]
                n += 1
        print(f"  layout on qnn test: region overlap {tot['region_overlap']/n:.3f}  distance error {tot['dist_mse']/n:.3f}")
