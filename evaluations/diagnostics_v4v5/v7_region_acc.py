"""Do the v7 layout properties transfer to unseen families?  On train vs test circuits:
region overlap (share of compiler-used qubits covered by the predicted occupancy), and
error of the operand chip distance (|log1p d_model - log1p d_compiler|), vs v5 and v6."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/atudisco/pred-distr-tool/evaluations/diagnostics_v4v5")
import diagnose as D
from torch_geometric.loader import DataLoader
from genstudy.data import GraphDataset
sys.path.insert(0, str(D.EVAL / "generalization_v3"))
from gsv3.training import region_distance_loss
EVAL = D.EVAL
params = D.load_hparams()
devices = D.load_devices(D.paths.V2_DEVICE_GRAPHS, D.DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
lay = torch.load(EVAL / "compile_check/results/layouts_all/compiler_layouts.pt", weights_only=False)
ds = GraphDataset.load(EVAL / "generalization_v3" / "data")
fam = np.array(ds.families)
rng = np.random.default_rng(0)
VAR = ["qnn", "vqe_real_amp", "vqe_su2", "vqe_two_local"]

def state(root, rel):
    p = root / rel / "model.pth"
    if p.is_file():
        return torch.load(p, map_location="cpu"), "final"
    ck = torch.load(root / rel / "checkpoint.pt", map_location="cpu", weights_only=False)
    return ck["best_state"], f"ckpt best ep {ck['best_epoch']}"

def measure(model, idx, k=1200):
    pick = rng.choice(len(idx), size=min(k, len(idx)), replace=False)
    graphs = []
    for i in pick:
        g = ds.data[idx[i]].clone(); L = lay.get(g.circuit_name)
        if L is None or L.shape[0] != int(g.n_qubits):
            continue
        g.layout = L; graphs.append(g)
    tot = {"region_overlap": 0.0, "dist_mse": 0.0, "region_bce": 0.0}; n = 0
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=16):
            model(b, devices)
            _, st = region_distance_loss(model, b)
            for key in tot:
                tot[key] += st[key]
            n += 1
    return {key: v / n for key, v in tot.items()}

runs = [("v5", EVAL / "generalization_v5/results/v5_sinkhorn"),
        ("v6 l01", EVAL / "generalization_v6/results/v6_l01"),
        ("v7 l001", EVAL / "generalization_v7/results/v7_l001"),
        ("v7 l01", EVAL / "generalization_v7/results/v7_l01")]
splits = [("LOFO qaoa", "leave_one_family_out/qaoa", np.where(fam != "qaoa")[0], np.where(fam == "qaoa")[0]),
          ("LOGO variational", "leave_one_group_out/variational",
           np.where(~np.isin(fam, VAR))[0], np.where(np.isin(fam, VAR))[0])]
print(f"{'split':18}{'model':9}{'state':20}{'set':6}{'region overlap':>15}{'dist err (log1p MSE)':>22}")
for sname, rel, tr, te in splits:
    for mname, root in runs:
        model = D.build_model("sinkhorn", params, devices, torch.device("cpu"))
        st, tag = state(root, rel); model.load_state_dict(st); model.eval()
        for lab, idx in (("train", tr), ("test", te)):
            r = measure(model, idx)
            print(f"{sname:18}{mname:9}{tag:20}{lab:6}{r['region_overlap']:>15.3f}{r['dist_mse']:>22.3f}", flush=True)
