#!/usr/bin/env python3
"""Train and evaluate a v3 predictor on the v1/v2 splits.

Experiments (same splits, seed and validation modes as v2):

``control``  random split stratified by fidelity (validation: random)
``lofo``     leave-one-family-out over the large families (validation: whole families)
``size``     train on <= n qubits, test on larger circuits (validation: largest sizes)

Each run writes to ``--results-dir`` (one directory per configuration).  Finished
splits are skipped, interrupted ones resume, and splits locked by another job
are skipped.

Examples::

    # v2 model, stabilised training
    python evaluations/generalization_v3/scripts/run_experiment.py lofo --model pooled \\
        --results-dir evaluations/generalization_v3/results/pooled_stab_a05
    # qubit cross-attention model (needs the qubit-annotated dataset)
    python evaluations/generalization_v3/scripts/run_experiment.py lofo --model xattn \\
        --results-dir evaluations/generalization_v3/results/xattn_a05
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from gsv3 import paths
from gsv3.devices import load_devices
from gsv3.experiment import run_split
from gsv3.groups import GROUPS, group_holdout_split
from gsv3.model import MODEL_KINDS, SINKHORN_KINDS, load_hparams, split_esp_kind
from gsv3.training import TRAIN_PROTOCOL

paths.ensure_imports()

from genstudy import paths as v1_paths  # noqa: E402
from genstudy.data import DEVICE_NAMES, FIGURE_OF_MERIT, GraphDataset  # noqa: E402
from genstudy.experiment import merge_predictions, split_is_complete  # noqa: E402
from genstudy.io import save_json, setup_logging  # noqa: E402
from gsv2.experiment import split_lock  # noqa: E402
from gsv2.gpu import select_device  # noqa: E402

logger = logging.getLogger("run_experiment")

EXPERIMENTS = {
    "control": ("random_control", "random"),
    "lofo": ("leave_one_family_out", "family"),
    "size": ("size_extrapolation", "size"),
    "logo": ("leave_one_group_out", "family"),
}

#: ``v1`` = the original v1 predictor (genstudy: linear target, 3 fixed outputs,
#: random validation split), run unchanged through ``genstudy.experiment.run_split``.
ALL_MODELS = (*MODEL_KINDS, *(k + s for k in SINKHORN_KINDS for s in ("_esp", "_espa", "_espi", "_esps", "_espf", "_espp", "_espfi", "_espft") if k + s not in MODEL_KINDS), "v1")


def attach_layouts(dataset, path: Path) -> None:
    """v6: ``d.layout`` = compiler layout ``(n_qubits, n_devices)``, ``-1`` where missing."""
    import torch
    layouts = torch.load(path, weights_only=False)
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        n = int(d.n_qubits)
        lay = layouts.get(name)
        if lay is not None and lay.shape == (n, len(DEVICE_NAMES)):
            d.layout, found = lay.clone(), found + 1
        else:
            d.layout = torch.full((n, len(DEVICE_NAMES)), -1, dtype=torch.long)
    logger.info("Compiler layouts (%s): %d/%d circuits", path, found, len(dataset.data))


def attach_counts(dataset, path: Path) -> None:
    """v8: ``d.counts`` = true (r, measure, cz) per device ``(1, n_devices, 3)``, ``-1`` if missing."""
    import torch
    counts = torch.load(path, weights_only=False)
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        c = counts.get(name)
        if c is not None and c.shape == (len(DEVICE_NAMES), 3):
            d.counts, found = c.unsqueeze(0).float(), found + 1
        else:
            d.counts = torch.full((1, len(DEVICE_NAMES), 3), -1.0)
    logger.info("Compiler counts (%s): %d/%d circuits", path, found, len(dataset.data))


def attach_esp(dataset, path: Path) -> None:
    """v13: ``d.esp`` = (ESP decay = log ESP − log F, log duration in µs, log depth) per device
    ``(1, n_devices, 3)`` of the stored seed-0 compilation (compile_check/esp_labels.py); NaN if missing."""
    import math
    import torch
    labels = torch.load(path, weights_only=False)
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        r = labels.get(name)
        if r is not None:
            dur = torch.log(r["duration"].float() * 1e6)
            dep = torch.log(r["depth"].float().clamp(min=1))
            d.esp, found = torch.stack([r["decay"].float(), dur, dep], -1).unsqueeze(0), found + 1
        else:
            d.esp = torch.full((1, len(DEVICE_NAMES), 3), math.nan)
    logger.info("ESP / duration / depth labels (%s): %d/%d circuits", path, found, len(dataset.data))


def attach_tvar(dataset, directory: Path) -> None:
    """v18: ``d.tdec`` / ``d.tdur`` = (1, 1 + K, n_devices) ESP decay and duration (µs) of the stored compilation
    rescheduled on the original devices (row 0) and on K T1 / T2 / gate-duration variants
    (compile_check/esp_tvar_train.py: the layout never moves, so rescheduling is exact); NaN if missing."""
    import math
    import torch
    labels = torch.load(Path(directory) / "esp_tvar_labels.pt", weights_only=False)
    k = next(iter(labels.values()))["decay"].size(0)
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        r = labels.get(name)
        if r is not None:
            d.tdec, d.tdur, found = r["decay"].float().unsqueeze(0), r["duration"].float().unsqueeze(0), found + 1
        else:
            d.tdec = torch.full((1, k, len(DEVICE_NAMES)), math.nan)
            d.tdur = torch.full((1, k, len(DEVICE_NAMES)), math.nan)
    logger.info("T / duration variant labels (%s): %d/%d circuits, %d variants", directory, found, len(dataset.data), k - 1)


def attach_struct(dataset, path: Path) -> None:
    """v15: ``d.struct`` = (tmax µs, mmax, omax, active) per device ``(1, n_devices, 4)`` of the stored seed-0
    compilation (compile_check/struct_labels.py); NaN if missing."""
    import math
    import torch
    labels = torch.load(path, weights_only=False)
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        r = labels.get(name)
        d.struct = r.float().unsqueeze(0) if r is not None else torch.full((1, len(DEVICE_NAMES), 4), math.nan)
        found += r is not None
    logger.info("Per-qubit structure labels (%s): %d/%d circuits", path, found, len(dataset.data))


def attach_usage(dataset, path: Path) -> None:
    """v11: usage maps of the best compilation on the original devices (merge_usage.py):
    ``usage_r``, ``usage_m`` (1, D, P_max) per physical qubit and ``usage_cz`` (1, D, E_max) per
    coupler; ``-1`` everywhere for circuits without maps (the loss skips them)."""
    import torch
    labels = torch.load(path, weights_only=False)
    any_ = next(iter(labels.values()))
    p_max, e_max = any_["r"].shape[1], any_["cz"].shape[1]
    found = 0
    for d, name in zip(dataset.data, dataset.names):
        r = labels.get(name)
        if r is not None:
            d.usage_r, d.usage_m, d.usage_cz = (r[k].unsqueeze(0).float() for k in ("r", "m", "cz"))
            found += 1
        else:
            d.usage_r = torch.full((1, len(DEVICE_NAMES), p_max), -1.0)
            d.usage_m = torch.full((1, len(DEVICE_NAMES), p_max), -1.0)
            d.usage_cz = torch.full((1, len(DEVICE_NAMES), e_max), -1.0)
    logger.info("Usage maps (%s): %d/%d circuits", path, found, len(dataset.data))


def attach_usage_aug_ids(dataset, path: Path) -> None:
    """v11 change-loss pass: ``d.usage_aug_id`` = row of the circuit in the variant maps of
    ``path`` (sorted names with variants, as gsv3.training builds the table), ``-1`` if none."""
    import torch
    labels = torch.load(path, weights_only=False)
    ids = {n: i for i, n in enumerate(sorted(n for n, e in labels.items() if "r_aug" in e))}
    for d, name in zip(dataset.data, dataset.names):
        d.usage_aug_id = torch.tensor([ids.get(name, -1)])
    logger.info("Variant usage maps (%s): %d/%d circuits", path, sum(n in ids for n in dataset.names), len(dataset.data))


def attach_calib_aug(dataset, directory: Path, circuit_mod: int = 0) -> None:
    """v8b: per-variant labels ``y_aug`` (1, A, D), layouts ``layout_aug`` (n, A, D) and
    counts ``counts_aug`` (1, A, D, 3) from ``v8b_labels.pt`` (v8d: ``v8d_labels.pt``, 100
    variants per device); ``-1`` where missing.  ``circuit_mod`` > 1 keeps the variant labels of
    ~1/circuit_mod of the circuits only (crc32 of the name), to test circuits vs calibrations."""
    import zlib
    import torch
    path = Path(directory) / "v8d_labels.pt"
    labels = torch.load(path if path.is_file() else Path(directory) / "v8b_labels.pt", weights_only=False)
    n_aug = next(iter(labels.values()))["F"].shape[0]
    n_dev, found = len(DEVICE_NAMES), 0
    for d, name in zip(dataset.data, dataset.names):
        n = int(d.n_qubits)
        r = labels.get(name)
        if circuit_mod > 1 and zlib.crc32(b"augc" + name.encode()) % circuit_mod:
            r = None
        if r is not None and r["layout"].shape[0] == n:
            d.y_aug = r["F"].unsqueeze(0).float()
            d.layout_aug = r["layout"].long()
            d.counts_aug = r["counts"].unsqueeze(0).float()
            found += 1
        else:
            d.y_aug = torch.full((1, n_aug, n_dev), -1.0)
            d.layout_aug = torch.full((n, n_aug, n_dev), -1, dtype=torch.long)
            d.counts_aug = torch.full((1, n_aug, n_dev, 3), -1.0)
    logger.info("Calibration variants (%s): %d variant sets, %d/%d circuits",
                directory, n_aug, found, len(dataset.data))


def add_extra_train(dataset, splits, directory: Path, args):
    """v22: append the synthetic circuits of ``directory`` (synthetic_routing/synth_routing.py: synth_graphs.pt and its
    counts / layouts / ESP labels) to the TRAINING side of every split, after the splits are drawn, so the test sets
    (and the control split's random draw) are exactly those of the other models.  Their labels are attached with the
    same functions as the dataset's; variant / best-of-K labels they do not have are filled as missing."""
    import dataclasses
    import torch
    from genstudy.data import family_of, num_qubits_of
    directory = Path(directory)
    graphs = torch.load(directory / "synth_graphs.pt", weights_only=False)
    ref = dataset.data[0]
    for g in graphs:      # the dataset graphs' bookkeeping fields (batching needs the same keys); not model inputs
        f = g.y.view(-1).tolist()
        g.target_label = DEVICE_NAMES[int(max(range(len(f)), key=f.__getitem__))]
        g.scores_dict = dict(zip(DEVICE_NAMES, f))
        g.compilation_time_dict = {d: float("nan") for d in DEVICE_NAMES}
        g.expected_runtime_dict = {d: float("nan") for d in DEVICE_NAMES}
        g.dag_preprocessing_time = float("nan")
        g.feature_vec = torch.zeros_like(ref.feature_vec)
    mini = argparse.Namespace(data=graphs, names=[g.circuit_name for g in graphs])
    if args.layout_lambda:
        attach_layouts(mini, directory / "synth_layouts.pt")
    if args.calib_aug is not None:
        attach_calib_aug(mini, args.calib_aug, args.aug_circuit_mod)
    if args.delta_lambda:
        for d in graphs:
            d.y_ob = torch.full((1, len(DEVICE_NAMES)), -1.0)
    if args.count_lambda:
        attach_counts(mini, directory / "synth_counts.pt")
    if args.esp_path is not None:
        attach_esp(mini, directory / "synth_esp_labels.pt")
    for flag in ("tvar_path", "struct_path", "usage_path", "usage_aug_path"):
        if getattr(args, flag, None) is not None:
            raise SystemExit(f"--extra-train does not provide the labels of --{flag.replace('_', '-')}")
    missing = set(ref.keys()) - set(graphs[0].keys())
    if missing:
        raise SystemExit(f"--extra-train graphs lack {sorted(missing)}")
    start = len(dataset.data)
    dataset.data.extend(graphs)
    dataset.names.extend(mini.names)
    dataset.families.extend(family_of(n) for n in mini.names)
    dataset.qubits.extend(num_qubits_of(n) for n in mini.names)
    new = tuple(range(start, len(dataset.data)))
    logger.info("Extra training circuits (%s): %d, added to the training side of %d split(s)", directory, len(new), len(splits))
    return [dataclasses.replace(sp, train=tuple(sp.train) + new) for sp in splits]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--model", choices=ALL_MODELS, default=TRAIN_PROTOCOL["model"])
    p.add_argument("--families", default="large")
    p.add_argument("--groups", default="vqe,fourier,variational",
                   help=f"logo: comma-separated groups among {sorted(GROUPS)}.")
    p.add_argument("--min-family-size", type=int, default=50)
    p.add_argument("--max-train-qubits", default="14")
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--val-mode", choices=["random", "family", "size"], default=None)

    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Default: the v3 qubit dataset for xattn, the original one for pooled.")
    p.add_argument("--params-path", type=Path, default=None)
    p.add_argument("--device-graphs", type=Path, default=paths.V2_DEVICE_GRAPHS)

    for key in ("batch_size", "patience", "seed"):
        p.add_argument(f"--{key.replace('_', '-')}", type=int, default=TRAIN_PROTOCOL[key])
    p.add_argument("--epochs", type=int, default=TRAIN_PROTOCOL["num_epochs"])
    for key in ("val_fraction", "balance_alpha", "log_eps", "lr", "grad_clip"):
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=TRAIN_PROTOCOL[key])

    p.add_argument("--loss", choices=["log", "mixed"], default=TRAIN_PROTOCOL["loss_kind"],
                   help="log: MSE on log F (v2/v3); mixed: MSE on F + mix_lambda * MSE on log F.")
    p.add_argument("--mix-lambda", type=float, default=TRAIN_PROTOCOL["mix_lambda"])
    p.add_argument("--epoch-budget", type=int, default=None,
                   help="v5b: fixed epochs + cosine LR + SWA final model (no early stopping).")
    p.add_argument("--swa-frac", type=float, default=None,
                   help="Fraction of the budget averaged by SWA (default 0.25).")
    p.add_argument("--placement-lr-mult", type=float, default=None,
                   help="Fine-tuning: learning rate multiplier for the placement parameters.")
    p.add_argument("--err-prior-init", type=float, default=None,
                   help="Initial bias of the low-error preference weight (softplus; -4 = 0.018, -1.05 = 0.3).")
    p.add_argument("--vf2-beta-init", type=float, default=None,
                   help="*_vf2: initial weight of the compiler-score prior (default 10).")
    p.add_argument("--sinkhorn-tau", type=float, default=None,
                   help="Sinkhorn temperature of the placement (default 0.3); saved with the model.")
    p.add_argument("--tau-start", type=float, default=None,
                   help="Sinkhorn temperature at epoch 1, annealed to sinkhorn_tau over 60%% of the budget.")
    p.add_argument("--layout-lambda", type=float, default=None,
                   help="v6 (sinkhorn only): weight of the cross-entropy between the placement "
                        "and the compiler's initial layout (needs --layouts-path).")
    p.add_argument("--layout-loss", choices=["assign", "region_dist"], default=None,
                   help="v6 'assign' (CE on the exact compiler layout, default) or v7 'region_dist' "
                        "(which physical qubits are used + operand distances).")
    p.add_argument("--count-lambda", type=float, default=None,
                   help="v8: weight of the loss on the predicted native-operation counts (needs --counts-path).")
    p.add_argument("--esp-path", type=Path, default=None,
                   help="v13: ESP / duration / depth labels (compile_check/results/esp_labels.pt)")
    p.add_argument("--esp-lambda", type=float, default=None, help="v13: weight of the ESP / duration / depth loss")
    p.add_argument("--extra-train", type=Path, default=None,
                   help="v22: directory with synth_graphs.pt + labels (synthetic_routing/results), training side only")
    p.add_argument("--dur-corr-lambda", type=float, default=None,
                   help="v21: L2 penalty on the formula head's learned log-duration correction (duration -> schedule)")
    p.add_argument("--esp-place-lambda", type=float, default=None,
                   help="v20 (*_espp): weight of the cross-entropy of the ESP head's own placement on the compiler's exact layout")
    p.add_argument("--tvar-path", type=Path, default=None,
                   help="v18: directory with esp_tvar_labels.pt and tvar_raw.json (compile_check/esp_tvar_train.py)")
    p.add_argument("--tvar-lambda", type=float, default=None,
                   help="v18: weight of the change loss on ESP decay and log duration under T1 / T2 / gate-time variants")
    p.add_argument("--tvar-f-lambda", type=float, default=None,
                   help="v18: weight of the invariance of log F under those variants (true change 0; default = tvar-lambda)")
    p.add_argument("--struct-path", type=Path, default=None,
                   help="v15: per-qubit structure labels (compile_check/results/struct_labels.pt)")
    p.add_argument("--struct-lambda", type=float, default=None, help="v15: weight of the per-qubit structure loss (*_espa)")
    p.add_argument("--esp-weights", default=None,
                   help="v13: per-target weights 'decay,duration,depth' of the ESP loss (default 1,1,1)")
    p.add_argument("--count-weights", default=None,
                   help="Per-kind weights of the count loss as 'r,measure,cz' (e.g. 0.1,0.1,1).")
    p.add_argument("--counts-path", type=Path, default=None,
                   help="compiler_counts.pt from evaluations/compile_check/compile_layouts.py counts.")
    p.add_argument("--usage-lambda", type=float, default=None,
                   help="v11: weight of the loss on the usage maps (needs --usage-path and a *_usage model).")
    p.add_argument("--usage-path", type=Path, default=None,
                   help="usage_labels.pt from evaluations/compile_check/merge_usage.py.")
    p.add_argument("--react-lambda", type=float, default=None,
                   help="Change-loss pass: weight of the loss on how the placement's qubit occupancy "
                        "moves from the original to the variant calibration vs the compiler's.")
    p.add_argument("--aug-aux", action="store_true",
                   help="Change-loss pass: also the layout and count losses on the calibration variant "
                        "(against its compiled layout and counts in --calib-aug).")
    p.add_argument("--usage-aug-lambda", type=float, default=None,
                   help="v11 fine-tuning: weight of the usage-map loss on the calibration variant of the "
                        "change-loss pass (needs --delta-lambda and --usage-aug-path).")
    p.add_argument("--usage-aug-path", type=Path, default=None,
                   help="usage_labels.pt with variant maps (merge_usage.py --var-dir).")
    p.add_argument("--calib-aug", type=Path, default=None,
                   help="v8b: directory with v8b_labels.pt and variants_raw.json (calibration "
                        "variants compiled by evaluations/compile_check/v8b_variants.py).")
    p.add_argument("--aug-p-orig", type=float, default=None,
                   help="v8b: probability that a batch uses the original calibration (default 0.25).")
    p.add_argument("--delta-lambda", type=float, default=None,
                   help="Change loss: weight of the fit of dlogF (variant vs original calibration); "
                        "needs --calib-aug and --orig-best.")
    p.add_argument("--orig-best", type=Path, default=None,
                   help="v8b_orig_labels.pt: best-of-K fidelity on the original devices (y_ob).")
    p.add_argument("--delta-corr-lambda", type=float, default=None,
                   help="Change loss: weight of 1 - Pearson(Δ predicted, Δ true) per batch and device.")
    p.add_argument("--aug-circuit-mod", type=int, default=0,
                   help="Keep the calibration-variant labels of ~1/N of the circuits only (data-scaling test).")
    p.add_argument("--aug-variants", default=None,
                   help="Comma-separated subset of the calibration variant sets to use (default all).")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Start from this model.pth (fine-tuning).")
    p.add_argument("--keep-last", action="store_true",
                   help="Keep the last-epoch weights instead of the best-validation ones.")
    p.add_argument("--layouts-path", type=Path, default=None,
                   help="compiler_layouts.pt from evaluations/compile_check/compile_layouts.py merge.")
    p.add_argument("--select-metric", choices=["mse", "log_mse"],
                   default=TRAIN_PROTOCOL["select_metric"],
                   help="Validation metric for early stopping / model selection.")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-save-model", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    exp_name, default_val_mode = EXPERIMENTS[args.experiment]
    val_mode = args.val_mode or default_val_mode

    experiment_dir = args.results_dir / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(experiment_dir / "run.log")
    logger.info("Command line: %s", " ".join(sys.argv))
    device = select_device()

    protocol = {**TRAIN_PROTOCOL, "batch_size": args.batch_size, "num_epochs": args.epochs,
                "patience": args.patience, "seed": args.seed, "val_fraction": args.val_fraction,
                "balance_alpha": args.balance_alpha, "log_eps": args.log_eps, "lr": args.lr,
                "grad_clip": args.grad_clip, "model": args.model,
                "select_metric": args.select_metric, "loss_kind": args.loss,
                "mix_lambda": args.mix_lambda}
    # Opt-in v5b keys: only present when requested, so older configurations (and
    # their checkpoints) are unchanged.
    for key in ("epoch_budget", "swa_frac", "tau_start", "layout_lambda", "layout_loss", "count_lambda",
                "aug_p_orig", "delta_lambda", "aug_variants", "count_weights", "usage_lambda", "usage_aug_lambda", "react_lambda", "placement_lr_mult", "delta_corr_lambda", "esp_lambda", "esp_weights", "struct_lambda", "tvar_lambda", "tvar_f_lambda", "esp_place_lambda", "dur_corr_lambda"):
        if getattr(args, key) is not None:
            protocol[key] = getattr(args, key)
    if args.model == "v1":
        from genstudy import experiment as v1_experiment
        from genstudy import model as v1_model
        params = v1_model.load_hparams(args.params_path)
        devices = None
    else:
        params = load_hparams(args.params_path)
        if args.sinkhorn_tau is not None:
            params["sinkhorn_tau"] = args.sinkhorn_tau
        if args.err_prior_init is not None:
            params["err_prior_init"] = args.err_prior_init
        if args.vf2_beta_init is not None:
            params["vf2_beta_init"] = args.vf2_beta_init
        qubit_aware = split_esp_kind(args.model)[0] in ("xattn", "phys", "sinkhorn", "phys_uniform", "sinkhorn_rf", "sinkhorn_rf_rel", "sinkhorn_rf_route", "sinkhorn_rf_rel_route", "sinkhorn_rf_route_czc", "sinkhorn_rf_rel_route_czc", "sinkhorn_rf_route_usage", "sinkhorn_rf_route_czc_ep", "sinkhorn_rf_route_czc_ep3", "sinkhorn_rf_route_czc_mlps", "sinkhorn_rf_route_czc_ep3_mlps", "sinkhorn_rf_route_czc_vf2", "sinkhorn_rf_route_czc_esp", "sinkhorn_rf_route_czc_espa", "sinkhorn_rf_route_pgnn_usage")
        devices = load_devices(args.device_graphs, DEVICE_NAMES,
                               lap_pe=params["lap_pe"] if qubit_aware else 0,
                               physical_errors=split_esp_kind(args.model)[0] in ("phys", "sinkhorn", "phys_uniform", "sinkhorn_rf", "sinkhorn_rf_rel", "sinkhorn_rf_route", "sinkhorn_rf_rel_route", "sinkhorn_rf_route_czc", "sinkhorn_rf_rel_route_czc", "sinkhorn_rf_route_usage", "sinkhorn_rf_route_czc_ep", "sinkhorn_rf_route_czc_ep3", "sinkhorn_rf_route_czc_mlps", "sinkhorn_rf_route_czc_ep3_mlps", "sinkhorn_rf_route_czc_vf2", "sinkhorn_rf_route_czc_esp", "sinkhorn_rf_route_czc_espa", "sinkhorn_rf_route_pgnn_usage"))

    if args.dataset_dir is None and split_esp_kind(args.model)[0] in ("xattn", "phys", "sinkhorn", "phys_uniform", "sinkhorn_rf", "sinkhorn_rf_rel", "sinkhorn_rf_route", "sinkhorn_rf_rel_route", "sinkhorn_rf_route_czc", "sinkhorn_rf_rel_route_czc", "sinkhorn_rf_route_usage", "sinkhorn_rf_route_czc_ep", "sinkhorn_rf_route_czc_ep3", "sinkhorn_rf_route_czc_mlps", "sinkhorn_rf_route_czc_ep3_mlps", "sinkhorn_rf_route_czc_vf2", "sinkhorn_rf_route_czc_esp", "sinkhorn_rf_route_czc_espa", "sinkhorn_rf_route_pgnn_usage"):
        args.dataset_dir = paths.DATA_DIR
    dataset_dir = v1_paths.find_dataset_dir(args.dataset_dir, FIGURE_OF_MERIT)
    dataset = GraphDataset.load(dataset_dir, FIGURE_OF_MERIT)
    if args.layout_lambda:
        if split_esp_kind(args.model)[0] not in ("sinkhorn", "sinkhorn_rf", "sinkhorn_rf_rel", "sinkhorn_rf_route", "sinkhorn_rf_rel_route", "sinkhorn_rf_route_czc", "sinkhorn_rf_rel_route_czc", "sinkhorn_rf_route_usage", "sinkhorn_rf_route_czc_ep", "sinkhorn_rf_route_czc_ep3", "sinkhorn_rf_route_czc_mlps", "sinkhorn_rf_route_czc_ep3_mlps", "sinkhorn_rf_route_czc_vf2", "sinkhorn_rf_route_czc_esp", "sinkhorn_rf_route_czc_espa", "sinkhorn_rf_route_pgnn_usage") or args.layouts_path is None:
            raise SystemExit("--layout-lambda needs --model sinkhorn and --layouts-path")
        attach_layouts(dataset, args.layouts_path)
    if args.calib_aug is not None:
        protocol["calib_aug"] = str(args.calib_aug)
        attach_calib_aug(dataset, args.calib_aug, args.aug_circuit_mod)
        protocol["aug_circuit_mod"] = args.aug_circuit_mod
    if args.init_from is not None:
        protocol["init_from"] = str(args.init_from)
    if args.keep_last:
        protocol["keep_last"] = True
    if args.aug_aux:
        if not args.delta_lambda:
            raise SystemExit("--aug-aux needs --delta-lambda")
        protocol["aug_aux"] = True
    if args.delta_lambda:
        if args.orig_best is None or args.calib_aug is None:
            raise SystemExit("--delta-lambda needs --calib-aug and --orig-best")
        import torch
        ob = torch.load(args.orig_best, weights_only=False)
        found = 0
        for d, name in zip(dataset.data, dataset.names):
            r = ob.get(name)
            d.y_ob = r["F"].float().view(1, -1) if r is not None else torch.full((1, len(DEVICE_NAMES)), -1.0)
            found += r is not None
        logger.info("Original-calibration best-of-K labels (%s): %d/%d circuits", args.orig_best, found, len(dataset.data))
    if args.count_lambda:
        if split_esp_kind(args.model)[0] not in ("sinkhorn", "sinkhorn_rf", "sinkhorn_rf_rel", "sinkhorn_rf_route", "sinkhorn_rf_rel_route", "sinkhorn_rf_route_czc", "sinkhorn_rf_rel_route_czc", "sinkhorn_rf_route_usage", "sinkhorn_rf_route_czc_ep", "sinkhorn_rf_route_czc_ep3", "sinkhorn_rf_route_czc_mlps", "sinkhorn_rf_route_czc_ep3_mlps", "sinkhorn_rf_route_czc_vf2", "sinkhorn_rf_route_czc_esp", "sinkhorn_rf_route_czc_espa", "sinkhorn_rf_route_pgnn_usage", "phys", "phys_uniform") or args.counts_path is None:
            raise SystemExit("--count-lambda needs a physics-head model and --counts-path")
        attach_counts(dataset, args.counts_path)
    if args.esp_lambda:
        if not args.model.endswith(("_esp", "_espa", "_espi", "_esps", "_espf", "_espp", "_espfi", "_espft")) or args.esp_path is None:
            raise SystemExit("--esp-lambda needs a *_esp / *_espa model and --esp-path")
        attach_esp(dataset, args.esp_path)
    if args.tvar_lambda:
        if not args.model.endswith(("_espa", "_espi", "_esps", "_espf", "_espp", "_espfi", "_espft")) or args.tvar_path is None:
            raise SystemExit("--tvar-lambda needs an *_espa / *_espi / *_esps model and --tvar-path")
        attach_tvar(dataset, args.tvar_path)
        protocol["tvar_path"] = str(args.tvar_path)
    if args.struct_lambda:
        if not args.model.endswith(("_espa", "_espi", "_esps", "_espf", "_espp", "_espfi", "_espft")) or args.struct_path is None:
            raise SystemExit("--struct-lambda needs a *_espa model and --struct-path")
        attach_struct(dataset, args.struct_path)
    if args.usage_lambda:
        if not split_esp_kind(args.model)[0].endswith("_usage") or args.usage_path is None:
            raise SystemExit("--usage-lambda needs a *_usage model and --usage-path")
        attach_usage(dataset, args.usage_path)
    if args.usage_aug_lambda:
        if not args.delta_lambda or args.usage_aug_path is None:
            raise SystemExit("--usage-aug-lambda needs --delta-lambda and --usage-aug-path")
        protocol["usage_path_aug"] = str(args.usage_aug_path)
        attach_usage_aug_ids(dataset, args.usage_aug_path)
    save_json(dataset.summary(), experiment_dir / "dataset_summary.json")

    if args.experiment == "control":
        splits = [dataset.random_split(test_fraction=args.test_fraction, seed=args.seed)]
        group_by = "family"
    elif args.experiment == "lofo":
        requested = None if args.families in ("", "large") else tuple(
            f.strip() for f in args.families.split(",") if f.strip())
        families = dataset.select_families(requested, min_size=args.min_family_size)
        splits = [dataset.family_holdout_split(f) for f in families]
        group_by = "qubits"
    elif args.experiment == "logo":
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        splits = [group_holdout_split(dataset, g) for g in groups]
        group_by = "family"
    else:
        lo, hi = dataset.qubit_range()
        cutoffs = [int(v) for v in args.max_train_qubits.split(",") if v.strip()]
        splits = [dataset.size_holdout_split(n) for n in cutoffs if lo <= n < hi]
        group_by = "qubits"

    if args.extra_train is not None:
        protocol["extra_train"] = str(args.extra_train)
        splits = add_extra_train(dataset, splits, args.extra_train, args)

    completed, skipped = [], []
    for position, split in enumerate(splits, start=1):
        split_dir = experiment_dir / split.name
        if not args.overwrite and split_is_complete(split_dir):
            logger.info("[%d/%d] %s already done — skipping", position, len(splits), split.name)
            completed.append(split.name)
            continue
        logger.info("[%d/%d] %s", position, len(splits), split.name)
        if args.model == "v1":
            with split_lock(split_dir) as acquired:
                result = v1_experiment.run_split(
                    dataset, split, params, split_dir, dataset_dir=dataset_dir,
                    device=device, seed=args.seed, group_by=group_by,
                    batch_size=args.batch_size, num_epochs=args.epochs,
                    patience=args.patience,
                    save_model=not args.no_save_model, resume=not args.overwrite,
                ) if acquired else None
            if result is None:
                logger.warning("[%s] locked by another job — skipping", split.name)
        else:
            result = run_split(dataset, split, devices, params, protocol, split_dir,
                               device=device, val_mode=val_mode, group_by=group_by,
                               save_model=not args.no_save_model,
                               resume=not args.overwrite)
        (completed if result is not None else skipped).append(split.name)

    if skipped:
        logger.warning("Splits held by another job (not run here): %s", ", ".join(skipped))
    if args.experiment == "lofo" and not skipped:
        save_json({"experiment": exp_name, "families": completed, "val_mode": val_mode},
                  experiment_dir / "folds.json")
        save_json(merge_predictions([experiment_dir / f for f in completed]),
                  experiment_dir / "heldout_predictions.json")
    logger.info("%s finished: %d complete, %d skipped", exp_name, len(completed), len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
