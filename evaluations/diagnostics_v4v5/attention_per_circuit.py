import sys, collections, numpy as np, torch
sys.path.insert(0, "/home/atudisco/pred-distr-tool/evaluations/diagnostics_v4v5")
import diagnose as D
params = D.load_hparams()
sample, truth, keys, graphs, full = D.cs_inputs(params)
model = D.load_model("v4_phys", D.CONTROL["v4_phys"], full, params)
from torch_geometric.loader import DataLoader
with torch.no_grad():
    for k in ["EQE1_Top/orig", "EQE1_Top/shuffle", "QExa20/orig", "QExa20/x2", "QExa20/jitterA"]:
        j = keys.index(k); tops = collections.Counter(); distinct = []; w1 = []; w2 = []
        for b in DataLoader(graphs, batch_size=16):
            _, a = D.v4_parts(model, b, full)
            a = a[j]                                   # (Q, P)
            n_q = b.n_qubits.view(-1).tolist(); s = 0
            for n in n_q:
                blk = a[s:s + n]; s += n
                t = blk.argmax(-1); tops.update(t.tolist())
                distinct.append(t.unique().numel() / n)
                srt = blk.sort(-1, descending=True).values
                w1.append(srt[:, 0].mean().item()); w2.append(srt[:, 1].mean().item())
        tot = sum(tops.values()); top3 = [(q, round(c / tot, 3)) for q, c in tops.most_common(3)]
        print(f"{k:18} top1 weight {np.mean(w1):.3f} top2 {np.mean(w2):.3f}  distinct top/qubit per circuit {np.mean(distinct):.3f}  most used physical qubits {top3}", flush=True)
# physics-head errors of the most attended qubits, orig vs shuffle
for k in ["EQE1_Top/orig", "EQE1_Top/shuffle"]:
    g = full.get_example(keys.index(k)); print(k, "phys errors of node 0..4:", g.phys[:5].numpy().round(4).tolist())
