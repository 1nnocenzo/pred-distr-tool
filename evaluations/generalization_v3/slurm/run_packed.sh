#!/bin/bash
# Several trainings in ONE GPU job.  The trainings are CPU-bound on one core each (GPU at 40-70%,
# ~3 GB of GPU memory), so a GPU runs several at once at little cost per run.
# Each non-blank, non-# line of the commands file is one training (env vars + script), run in the
# background on the working GPUs of the allocation, round-robin; the CPUs are split evenly.
#
#     sbatch -p gpu_h200 --gres=gpu:2 -c 8 --mem=24G evaluations/generalization_v3/slurm/run_packed.sh cmds.txt
#   cmds.txt, e.g.:
#     RUN=v8kdc10 MODEL=sinkhorn_rf_route_czc ... bash evaluations/generalization_v3/slurm/train_gpu.sh
#     bash evaluations/v12/phase1.sh --place-loss ce --tag _ce
#
# On gpu_h200 (broken GPU 0000:19:00.0) ask --gres=gpu:2: the broken one is skipped, a working
# second GPU is used.  Memory: ~3.5 GB of RAM per training.  Logs: results/slurm/packed_<job>_<i>.log.

#SBATCH --job-name=packed
#SBATCH --output=evaluations/generalization_v3/results/slurm/packed_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -uo pipefail
CMDS="${1:?usage: run_packed.sh <commands file>}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"; conda activate "${CONDA_ENV:-pred-distr}"
LOGDIR=evaluations/generalization_v3/results/slurm

good=()
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
  if CUDA_VISIBLE_DEVICES="$gpu" timeout 120 python3 -c \
      "import torch; x = torch.ones(8, device='cuda') * 2; torch.cuda.synchronize(); assert float(x.sum()) == 16.0" \
      >/dev/null 2>&1; then
    good+=("$gpu")
  else
    echo "GPU index $gpu is not usable — skipping."
  fi
done
[ ${#good[@]} -gt 0 ] || { echo "No usable GPU"; exit 3; }

mapfile -t lines < <(grep -Ev '^[[:space:]]*(#|$)' "$CMDS")
n=${#lines[@]}
threads=$(( ${SLURM_CPUS_PER_TASK:-8} / n )); [ "$threads" -ge 1 ] || threads=1
echo "$(date -u +%T) $n trainings on GPUs ${good[*]} ($(hostname)), $threads threads each"

pids=()
for i in "${!lines[@]}"; do
  gpu=${good[$(( i % ${#good[@]} ))]}
  log="$LOGDIR/packed_${SLURM_JOB_ID:-local}_$i.log"
  echo "[$i] GPU $gpu -> $log: ${lines[$i]}"
  ( export CUDA_VISIBLE_DEVICES="$gpu" SLURM_CPUS_PER_TASK="$threads" \
           OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
    eval "${lines[$i]}" ) > "$log" 2>&1 &
  pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "$(date -u +%T) [$i] done"; else echo "$(date -u +%T) [$i] FAILED"; rc=1; fi
done
exit $rc
