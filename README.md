# esptime: execution time and ESP of a quantum circuit, without compiling

One graph network predicts, for a circuit (OpenQASM, 2–20 qubits) on each of three IQM devices
(EQE1_Top 26 q, EQE1_Bottom 27 q, QExa20 20 q; native gates `r`, `cz`, `measure`):

| output | definition (labels: Qiskit 2.3.1 level 2, seed 0, on the basis circuit the model encodes) |
| --- | --- |
| **duration** | ASAP schedule of the compiled circuit, r 42 ns, cz 130 ns, measure 15 µs |
| **ESP** | MQT estimated success probability = F · exp(decay), decay = −Σ_q idle_q / min(T1, T2)_q |
| F (auxiliary) | MQT expected fidelity = Π (1 − error) over the compiled operations |

It is the v29 design of the earlier study (branch `esp-depth`), trained as one model:

```
gate DAG ─► backbone: SAGE encoder ─► logical qubits ─► Sinkhorn placement on the device graph ─► counts ─► log F
                │ gate embeddings, operand distances, hard placement
                ▼
        route stage: SWAPs per operand wire + waits + c2 ─► level-wise ASAP schedule ─► duration
                ▼
        ESP head: idle = (active qubits · duration − busy time from the counts) × mean 1/min(T1,T2) ─► decay
                  ─► log ESP = log F + decay      (T: placed qubits if routing-free, device mean if routed)
```
Devices enter as calibration graphs (per-qubit errors, T1/T2, couplers), so calibration variants are just other graphs.

## Layout

| path | what |
| --- | --- |
| `createDevice.py`, `data/` | backends and calibrations (QExa20 T1/T2 ×10 unit fix: `QEXA20_T_SCALE`, an assumption), `circuits.zip` = MQT Bench |
| `esptime/config.py` | paths (`ESPTIME_WORK`, default `work/`), devices, gate times, excluded families |
| `esptime/encoding.py` | circuit → gate DAG (40 features per gate); `decomp_table.json` = r / cz per gate type |
| `esptime/devices.py` | backends, calibration variants `aug<j>`, raw and standardised device graphs |
| `esptime/labels.py` | **one compilation per (circuit, calibration, device)** → F, decay, duration, layout, counts, routing-free, Sabre SWAPs per gate |
| `esptime/synth.py` | 21k synthetic routing circuits (training side only) |
| `esptime/router.py` | compiler-free routing inputs: VF2 routing-free test, greedy router SWAPs / hops, wire features |
| `esptime/dataset.py` | labels + circuits → `work/dataset.pt`; splits |
| `esptime/model.py` | `ESPTimeModel` = backbone + route stage + ESP head |
| `esptime/losses.py` | all losses and their weights (`LAMBDAS`) |
| `esptime/train.py` | train / evaluate / time one split → `work/runs/<split>_s<seed>/metrics.json` |
| `esptime/oracle.py` | formula check: duration and ESP formulas with the compiler's true inputs (`results/formula_check_*.md`) |
| `tests/test_model.py` | 6 real compilations → items → forward / losses / backward checks (~1 min) |
| `scripts/smoke.sh` | the whole pipeline on ~100 circuits (~15 min, CPU) |
| `scripts/run_local.sh` | the whole pipeline on one machine, no SLURM |
| `scripts/pipeline.sh`, `job.sh` | the whole pipeline as a chain of SLURM jobs |

## Run on a server without SLURM

```bash
conda env create -f environment.yml && conda activate pred-distr      # once
bash scripts/run_local.sh test         # ~15 s, must print "all checks passed"
bash scripts/run_local.sh smoke        # whole pipeline on ~90 circuits (~10 min)
JOBS=32 bash scripts/run_local.sh labels      # compilations (the long step), resumable
JOBS=32 bash scripts/run_local.sh dataset
DEVICE=cuda GPUS="0 1" TRAIN_JOBS=2 bash scripts/run_local.sh train   # 4 splits x seeds 5, 6, 7
bash scripts/run_local.sh report       # -> work/runs/summary.md
```
Settings and options: header of `scripts/run_local.sh`. Use `nohup ... &` or `tmux` for the long steps.

## Run (step by step / SLURM)

```bash
conda env create -f environment.yml && conda activate pred-distr
python tests/test_model.py                 # "all checks passed"
bash scripts/smoke.sh                      # end to end on a few circuits

# full data (cluster): N_TASKS label jobs per source, then merge -> dataset -> 4 splits x 3 seeds
bash scripts/pipeline.sh
# or step by step
python -m esptime.synth                                     # work/circuits/synth/*.qasm
python -m esptime.labels run --source mqt --calibs orig,aug0-19 --task-id 0 --n-tasks 1
python -m esptime.labels run --source synth --calibs orig,aug0-19
python -m esptime.labels merge                              # work/labels.pt, work/devices.json
python -m esptime.dataset --jobs 8                          # work/dataset.pt
python -m esptime.train --split lofo:qaoa --seed 5 --calib-every 4 [--device cuda]
```

Splits: `control` (20% of the real circuits by name hash), `lofo:qaoa`, `lofo:qnn`, `logo:variational`
(vqe_* + qnn), `loso:<synthetic structures>` (development). Synthetic circuits are only ever in training.

`metrics.json`: duration error (median / p90 / bias vs the label; routed, routing-free, routed by 2q-gate bin),
R² of F and ESP, |Δlog ESP| for ESP > 1e-3, device choice by ESP and by F (right share, regret), routing-free switch
agreement, time per circuit by 2q-gate bin next to the light / full compile.

Useful options of `train`: `--calib-every k` (variant pass + change loss every k-th batch), `--esp-grad all` (the
decay loss also trains the route stage), `--lambda dur=0.5` (loss weights), `--init ckpt --freeze-backbone` (v29's
two-stage recipe), `--mc 24` (Poisson samples of the SWAPs at evaluation).

## Reference numbers (earlier study, other labels: compare trends, not digits)

| | value |
| --- | --- |
| routed duration, median error | v27 route stage qaoa 6.4%; light compile 5.4%; compiler seed floor 3.9% |
| routed ESP, abs Δlog ESP (ESP > 1e-3) | F model's ESP head 0.30–0.65; light compile 0.29; seed floor 0.20 |
| time per circuit, 3 devices, by 2q gates 0–20 / 20–50 / 50–200 / 200–600 / 600+ | full compile 33 / 86 / 531 / 1845 / 2786 ms; light compile 19 / 37 / 69 / 132 / 190 ms |

## Known caveats

1. Routed ESP is very sensitive to the duration: Δlog ESP ≈ (Δduration / duration) · |log ESP|.
2. QExa20's T1/T2 ×10 (`createDevice.QEXA20_T_SCALE`) is an assumption to verify.
3. The earlier study selected ~30 model versions on the same test splits: freeze a model before a final test.
