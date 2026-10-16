# Routed-circuit duration (v24): what to run and evaluate off the cluster

State on 2026-10-13. The PoliTo CPU queues are blocked (fair-share), so the v24 work moves to another machine.
Everything here runs without SLURM: `run_local.sh` chains the steps; `report_v24.py` writes `results/report_v24.md`.

## 1. Get code and data

1. Code: push branch `esp-depth`, then `git clone` / `git pull` it on the other machine.
2. Data not in git (2.1 GB, paths relative to the repo root): `~/v24_data_bundle.tar` on the cluster
   (sha256 in `~/v24_data_bundle.tar.sha256`; list in `bundle_files.txt`).
   ```bash
   scp <cluster>:v24_data_bundle.tar* . && sha256sum -c v24_data_bundle.tar.sha256
   tar -xf v24_data_bundle.tar -C <repo root>
   ```
3. Environment: `conda env create -f environment.yml` (env `pred-distr`: Python 3.11, torch 2.11, torch-geometric 2.7).
   Without a CUDA GPU, use the CPU build of torch; the code runs on CPU by default.

## 2. Run

From the repo root, with `conda activate pred-distr`:

```bash
bash evaluations/route_stage/run_local.sh smoke     # ~10 min: every code path on a few circuits; stop if it fails
bash evaluations/route_stage/run_local.sh diag      # scale diagnostic, 3 splits (seed-5 checkpoints, no training)
bash evaluations/route_stage/run_local.sh mc        # seed-5 checkpoints re-evaluated: sampled SWAPs + dataset target
bash evaluations/route_stage/run_local.sh report    # -> evaluations/route_stage/results/report_v24.md
JOBS=3 bash evaluations/route_stage/run_local.sh train   # v24b: 40 epochs x 3 seeds x 4 splits, then `report` again
```

- Settings: `THREADS` (torch threads per process, default 8), `JOBS` (trainings in parallel), `DEVICE=cuda` for a GPU,
  `EPOCHS`, `SEEDS`, `MC`. Logs in `evaluations/route_stage/local_logs/`.
- Time on the cluster (8 cores): ~90 s per epoch → ~1 h per training at 40 epochs, 12 trainings. RAM per process not
  measured (whole dataset in memory: expect several GB); check the first run before raising `JOBS`.
- `train` skips runs that already have their eval file, so it can be restarted.

## TO LAUNCH (2026-10-16): v28 = v27 with the light Sabre as routing input

The greedy router (prior SWAPs, hops) is far from Sabre: on test circuits its routed schedule is 65% off the true one, the
light Sabre's 4-8% (qaoa / qft / randomcircuit / synthetic; SWAPs 1.10-1.20x). Light Sabre = Qiskit's level-2 init stage +
barrier + SabreLayout with 1 trial / 1 iteration / 1 routing trial (`sabre_light.py`), ~6-10 ms per (circuit, device)
including the alignment to the model's nodes (randomcircuit ~40 ms). Its SWAPs per slot replace the greedy ones and its
waits are the base of the predicted wait. Declared in the paper as a cheap approximate routing pass at prediction time.

```bash
cd evaluations/route_stage
PYTHONPATH=../../src python test_stage1.py                     # 10 s
JOBS=8 bash run_local.sh light                                  # CPU: every circuit -> results/sabre_light.pt + check
DEVICE=cuda GPUS="0 1" JOBS=4 bash run_local.sh train28         # needs synth2 (stage1_synth2.pt) as v27
bash run_local.sh report
```

Then the end-to-end test of the v28 path (CPU; switch -> exact compile | light Sabre + v28; duration, ESP, F, times vs the
labels and vs the full / light compile of the same case; F from v8kdc10's predictions.json of the same split and seed,
as the earlier ESP studies; without it F comes from the decomposition table and is biased low):

```bash
JOBS=8 EP_TASKS=4 bash run_local.sh evalpath     # -> results/evalpath/report.md (EP_MAX=300 for a quick look per split)
```

Adopt v28 if it beats v27 (qaoa 8.8%, control 3.2%, variational 5.4%, LOSO 7.4% / 11.8%) without losing elsewhere.
`light_check.log` gives the light Sabre's own schedule error per family (the floor it brings).

## TO LAUNCH (2026-10-15 evening): v27 = v25 + the low-parallelism synthetic class

Why: v25 fixed unseen synthetic structures (13.8% -> 7.6%) and control (5.3% -> 3.3%) but not qaoa (13.9%, bias 0.86);
randomcircuit, seen in training, is as bad. Both are dense, irregular, low-parallelism circuits repeating the same pairs
(parallelism qaoa 0.46 / randomcircuit 0.32 vs 0.9 for the old synthetic circuits): that class was missing from training.
`synthetic_routing/synth_layered.py` generates it (structures `layered` and `seqdense`, parallelism 0.36-0.38, density
0.59, 2.5-4 gates per pair; without waits their schedule is 24% off, with true SWAPs + waits 4.4%: same signature as qaoa).
Separate files, the bundle's data are not touched.

```bash
cd evaluations/route_stage
PYTHONPATH=../../src python test_stage1.py                 # 10 s
JOBS=8 bash run_local.sh synth2                             # CPU: 6000 circuits -> Sabre labels -> results/stage1_synth2.pt
DEVICE=cuda GPUS="0 1" JOBS=4 bash run_local.sh train27     # 3 seeds x (4 splits + loso:layered,seqdense)
bash run_local.sh report
```

`N_SYNTH2` changes the number of circuits (default 6000). Adopt v27 if qaoa drops clearly below v25's 13.9% (and its
bias towards 1) with control / variational / LOSO not worse. Caveat for the paper: `layered` is the generic shape of
qaoa-like circuits (random graphs, random rotation types), not the qaoa template; the new split loso:layered,seqdense
shows how well the class itself generalises.

## TO LAUNCH (2026-10-15): v25 and v26, both

Why: routed durations are under-predicted because of the waits, not the SWAPs (TODO.md, "v25"). Two fixes, run both
(same seeds and splits as v24b, so `report` puts v24b / v25 / v26 side by side):

```bash
cd evaluations/route_stage
bash run_local.sh smoke25                 # ~5 min: duration loss runs and falls
JOBS=3 bash run_local.sh train25          # v25 = v24b + loss on the routed duration (differentiable schedule)
bash run_local.sh contention              # once, ~15 min: our greedy router's contention wait per gate -> results/contention.pt
bash run_local.sh smoke26                 # ~5 min; prints corr(router wait, Sabre wait): the feature's signal
JOBS=3 bash run_local.sh train26          # v26 = v25 + contention as input and as the base of each wait
bash run_local.sh report                  # results/report_v24.md
```

| run | what changes | look at | adopt if |
| --- | --- | --- | --- |
| v25 | `--dur-lambda 1`: Huber on log(schedule / Sabre replay) through `schedule_torch` | routed error and bias on qaoa and LOSO | qaoa / LOSO < 10% and bias ≈ 1, control and variational not worse |
| v26 | v25 + `--contention`: wait = router contention wait + learned correction | the heavy-routing bins (2q gates 100–500, SWAPs per 2q gate 1–2, routing share 0.4–0.8) | those bins < 12% where v25 stays > 15% |

Before launching after a code change: `PYTHONPATH=../../src python test_stage1.py` (~10 s, no data needed).
`train25` / `train26` skip runs that already have their eval file; `DUR_EVERY=2` halves the cost of the duration loss
if an epoch is too slow (the schedule is computed level by level).

## 3. To do and to evaluate

| # | What | Where | Look at | Decision |
| --- | --- | --- | --- | --- |
| 1 | Scale diagnostic (`diag`) | here | `report_v24.md`, section "Scale diagnostic" | sampled ≈ true durations → the under-estimate is the schedule of mean rates (Jensen); keep sampling at prediction. Large share of predicted SWAPs on zero slots → zero-inflated rate or a direct duration loss (tell Claude). |
| 2 | Seed-5 re-evaluation (`mc`) | here | columns *sampled* vs *mean rates*, SWAP count ratio | The SWAP count of the first run was inflated by a metric bug (empty operand slots were summed; fixed): check the true ratio first. |
| 3 | v24b, 40 epochs × 3 seeds (`train`) | here | *sampled* and *vs dataset* columns, mean ± std | Candidate if routed error on qaoa / variational beats v21t (15% / 15%) on the dataset target, with routing-free unchanged (≤ 2%). |
| 4 | Compare with v21t / v22 on one target | here | *vs dataset: sampled* vs `esp/results/report_esp_routed_v10_v22.md` | v24 trains on Sabre's own compilation; qaoa differs from the dataset labels (p90 16%). Compare only *vs dataset*. Same test circuits; v22's "routed" comes from its own flag, so case counts differ slightly. |
| 5 | Stage 2 on unseen families | here | *sampled* vs *sampled, no stage 2* on qaoa | Stage 2 hurt qaoa (12.7% → 15.3%). If it still does, drop it or keep it only where the routing overhead is low. |
| 6 | Why qaoa labels differ (Sabre relabel vs stored compilations) | here or cluster | `compile_check/sabre_labels.py` vs `compile_check/esp_labels.py` (both level 2, seed 0) | Suspects: qaoa regenerated with final measurements after the stored compilations, or a different Qiskit version / target. Decides which duration target the paper uses. |
| 7 | ESP of routed circuits from the predicted schedule | after 3 | `results/esp_oracle.md` (oracle: 0.7–1.4% log-ESP error with the true schedule) | Apply the oracle formula (k · idle × chip-mean 1/min(T1,T2)) to the v24b schedule; compare with v21t ESP on routed cases. Not scripted yet. |
| 8 | v23 evaluation (router features, v8kdfy10) | cluster (queued, job 1994408) | `esp/results/report_esp_*v8kdfy10*` | Depends on calibration-shift data and `.pth` models; stays on the cluster. |

Not to do: Qiskit's SabreSwap (or any compiler pass) as a model input. Sabre is a training label only.
