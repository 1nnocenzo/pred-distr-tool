"""Does the v6 placement generalise?  Agreement with the compiler's layout (exact qubit,
within 1 hop, mean hop distance) on train vs test circuits, for several splits."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/atudisco/pred-distr-tool/evaluations/diagnostics_v4v5")
import diagnose as D
from torch_geometric.loader import DataLoader
from genstudy.data import GraphDataset
EVAL = D.EVAL
params = D.load_hparams()
devices = D.load_devices(D.paths.V2_DEVICE_GRAPHS, D.DEVICE_NAMES, lap_pe=params["lap_pe"], physical_errors=True)
lay = torch.load(EVAL / "compile_check/results/layouts_all/compiler_layouts.pt", weights_only=False)
ds = GraphDataset.load(EVAL / "generalization_v3" / "data")
rng = np.random.default_rng(0)

def state(path):
    if path.name == "checkpoint.pt":
        return torch.load(path, map_location="cpu", weights_only=False)["best_state"]
    return torch.load(path, map_location="cpu")

def agreement(model, graphs):
    exact = near = n = 0; hop = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=16):
            _, a = D.v5_parts(model, b, devices)
            for i, g in enumerate(b.to_data_list()):
                L = lay.get(g.circuit_name)
                if L is None:
                    continue
                nq = int(g.n_qubits)
                pred = a[i, :, :nq].argmax(-1)                     # (D, nq)
                for d in range(pred.size(0)):
                    h = model.hops[d][pred[d], L[:, d]]            # hop distance to compiler's qubit
                    exact += int((h == 0).sum()); near += int((h <= 1).sum()); n += nq
                    hop.append(h.float().mean().item())
    return exact / n, near / n, float(np.mean(hop))

def subset(idx, k=1500):
    idx = list(idx)
    pick = rng.choice(len(idx), size=min(k, len(idx)), replace=False)
    return [ds.data[idx[i]] for i in pick]

ctrl = ds.random_split(test_fraction=0.3, seed=5)
fam = np.array(ds.families)
splits = {
    "control (same families)": (EVAL / "generalization_v6/results/v6_l01/random_control/random_seed5/checkpoint.pt",
                                ctrl.train, ctrl.test),
    "LOFO qaoa": (EVAL / "generalization_v6/results/v6_l01/leave_one_family_out/qaoa/model.pth",
                  np.where(fam != "qaoa")[0], np.where(fam == "qaoa")[0]),
    "LOGO variational": (EVAL / "generalization_v6/results/v6_l01/leave_one_group_out/variational/model.pth",
                         np.where(~np.isin(fam, ["qnn", "vqe_real_amp", "vqe_su2", "vqe_two_local"]))[0],
                         np.where(np.isin(fam, ["qnn", "vqe_real_amp", "vqe_su2", "vqe_two_local"]))[0]),
}
print(f"{'split (v6 lambda 0.1)':26}{'set':6}{'exact':>8}{'<=1 hop':>9}{'mean hops':>11}")
for name, (path, tr, te) in splits.items():
    model = D.build_model("sinkhorn", params, devices, torch.device("cpu"))
    model.load_state_dict(state(path)); model.eval()
    for lab, idx in (("train", tr), ("test", te)):
        e, n1, h = agreement(model, subset(idx))
        print(f"{name:26}{lab:6}{e:>8.3f}{n1:>9.3f}{h:>11.2f}", flush=True)
# reference: a random one-to-one layout
e = n1 = n = 0; hs = []
for g in subset(np.where(fam == "qaoa")[0], 500):
    L = lay.get(g.circuit_name); nq = int(g.n_qubits)
    if L is None: continue
    for d, P in enumerate(torch.bincount(devices.batch).tolist()):
        r = torch.tensor(rng.choice(P, nq, replace=False))
        hh = torch.as_tensor(model.hops[d][r, L[:, d]])
        e += int((hh == 0).sum()); n1 += int((hh <= 1).sum()); n += nq; hs.append(hh.float().mean().item())
print(f"{'random layout (qaoa)':26}{'':6}{e / n:>8.3f}{n1 / n:>9.3f}{np.mean(hs):>11.2f}")
