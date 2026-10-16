# pred-distr-tool / esptime — one model for execution time and ESP

Fresh scaffold (branch `esp-time-model`, from `esp-depth` on 2026-10-16): one graph network predicts, without compiling,
the duration and the ESP (and F) of a circuit on EQE1_Top, EQE1_Bottom and QExa20. It ports the v29 design of the
earlier study (backbone F network + route stage + ESP formula head), trained jointly. The earlier study's code,
results and guides (`evaluations/RESTART_GUIDE.md`, `GUIDA_COMPLETA.md`) are on branch `esp-depth`.

Several co-authors may use Claude here: they may write in Italian, English or German; answer in the language of the
message. Reports for co-authors are in English, visual, with little text. Be proactive: launch and chain experiments,
report results with numbers. Do not over-engineer: this is a base for new experiments.

## Where things are

See `README.md` (layout, run commands, splits, metrics). Generated data goes to `work/` (`ESPTIME_WORK`), git-ignored.

## Design decisions (2026-10-16, with the user)

- Model = v29 stack: Sinkhorn backbone (F, placement) + route stage (SWAPs, waits, c2 → schedule → duration) + ESP
  head on the backbone's hard placement. Joint training in one stage (`--init` + `--freeze-backbone` = v29 recipe).
- Devices are calibration graphs (input), not fixed output heads.
- Outputs: duration, log F, decay; ESP = F · exp(decay) by construction.
- ESP count formula (2026-10-16, after the formula check): decay = −e^a · [(N_log + N_extra) · duration − busy] ·
  mean 1/T, busy from the predicted counts (exact identity), N_extra predicted; T from the hard placement when
  routing-free, device mean when routed (the learned placement was worse than the device mean there).
  Duration also trained end to end (`final` loss); model selection on the target terms (F, final, decay).
- Labels: one Qiskit L2 compilation (seed 0) of the BASIS circuit the model encodes; F, ESP, duration, layout, counts
  and Sabre SWAP targets all come from it (no more mixed seed-0 / best-of-3 labels, no qaoa rzz label gap).
- Included: synthetic routing circuits, timing benchmark, calibration variants (+ change loss on log F and log ESP).

## Workflow

- First: `python tests/test_model.py` after any code change; `bash scripts/smoke.sh` after pipeline changes.
- Then work on correctness of duration and ESP against the labels (metrics.json), 4 splits × seeds 5, 6, 7.

## Cluster rules (PoliTo HPC)

- Never load data or train on the login node: use `srun` / `sbatch` (smoke tests too). `/tmp` is node-local.
- Before every launch compare start times on the usable partitions (`sbatch --test-only` on gpu_h200, gpu_a100,
  gpu_a40, gpu_a40_ext, cpu_sapphire, cpu_sapphire_ext, cpu_skylake, cpu_skylake_ext) and pick the fastest; tell the
  user where each job runs. On gpu_h200 skip the broken GPU 0000:19:00.0. V100 partitions do not work (torch 2.11+cu128).
- Some shared nodes run ~30× slower (compute-5-6, compute-5-8): check the first epochs; resubmit with `-w <idle node>`.
- Request memory near the measured peak (`sstat -j <id>.batch -o MaxRSS`; `sacct`/`seff` do not work). Compile task
  ~2G. Training / dataset memory not measured yet in this scaffold: measure on the first runs.
- No job arrays; submit tasks in a loop (`scripts/pipeline.sh`). Chain steps with `--dependency`.

## Commits

- Small logical commits; never push unless asked; no `Co-Authored-By` trailer; the user does the pushes (tell them
  which branch and why).
- Commit code and finished small results only (metrics.json, history.json); never `work/`, `*.pt`, `*.pth`.
