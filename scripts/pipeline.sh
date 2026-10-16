#!/bin/bash
# Submit the whole pipeline as a chain of SLURM jobs (run on the login node; nothing heavy runs here).
#   labels (N_TASKS jobs per source, original calibration + CALIBS) -> merge -> dataset -> train SPLITS x SEEDS
# Settings: N_TASKS (64) CALIBS (orig,aug0-19) SPLITS SEEDS P_CPU (cpu_sapphire) P_TRAIN (cpu_sapphire)
#           TRAIN_SBATCH (extra sbatch options for training, e.g. "--gres=gpu:1") TRAIN_ARGS (e.g. "--device cuda")
# Pick the partitions first (CLAUDE.md: compare start times with sbatch --test-only).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/slurm
N_TASKS=${N_TASKS:-64}
CALIBS=${CALIBS:-orig,aug0-19}
SPLITS=${SPLITS:-"control lofo:qaoa lofo:qnn logo:variational"}
SEEDS=${SEEDS:-"5 6 7"}
P_CPU=${P_CPU:-cpu_sapphire}
P_TRAIN=${P_TRAIN:-cpu_sapphire}
sub() { sbatch --parsable "$@"; }

synth=$(sub -p "$P_CPU" --mem=2G --time=01:00:00 -J synth scripts/job.sh synth)
ids=()
for src in mqt synth; do
  dep=""; [ "$src" = synth ] && dep="--dependency=afterok:$synth"
  for t in $(seq 0 $((N_TASKS - 1))); do
    ids+=("$(sub -p "$P_CPU" $dep --cpus-per-task=1 --mem=2G --time=12:00:00 -J "lab_${src}_$t" \
      scripts/job.sh labels run --source "$src" --calibs "$CALIBS" --task-id "$t" --n-tasks "$N_TASKS")")
  done
done
merge=$(sub -p "$P_CPU" --dependency="afterany:$(IFS=:; echo "${ids[*]}")" --mem=8G --time=01:00:00 -J merge \
  scripts/job.sh labels merge)
data=$(sub -p "$P_CPU" --dependency="afterok:$merge" --cpus-per-task=8 --mem=24G --time=04:00:00 -J dataset \
  scripts/job.sh dataset --jobs 8)
for split in $SPLITS; do
  for seed in $SEEDS; do
    sub -p "$P_TRAIN" ${TRAIN_SBATCH:-} --dependency="afterok:$data" --mem=24G --time=24:00:00 -J "tr_${split}_$seed" \
      scripts/job.sh train --split "$split" --seed "$seed" --calib-every 4 ${TRAIN_ARGS:-} > /dev/null
  done
done
echo "submitted: synth $synth, ${#ids[@]} label tasks, merge $merge, dataset $data, training on $P_TRAIN"
