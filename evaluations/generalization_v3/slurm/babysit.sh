#!/bin/bash
# Autonomous supervisor for a plan of v3/v4 splits; independent of any interactive session.
#
# PLAN is a text file read by scripts/pending.py (one split per line:
# RUN EXPERIMENT SPLIT MODEL ALPHA SEED).  Every minute the supervisor lists the
# planned splits that are unfinished and not held by any job.  A split that stays
# free for 3 consecutive checks gets a CPU job (train_cpu.sh restricted to exactly
# that split; it resumes from the split's checkpoint if any).  It never submits
# twice for a split whose job is still queued or running, alternates the CPU
# partitions, and gives up on a split after MAX_ATTEMPTS submissions.  When every
# split is done it runs scripts/summarize.py on the plan and exits.
#
#     PLAN=evaluations/generalization_v3/results/plan_logo.txt \
#       sbatch evaluations/generalization_v3/slurm/babysit.sh
#
# RESULTS_ROOT (exported to pending.py and to the training jobs) selects where the
# plan's runs live, e.g. evaluations/generalization_v4/results.
#
# Log: evaluations/generalization_v3/results/slurm/babysit_<jobid>.log

#SBATCH --job-name=v3_babysit
#SBATCH --output=evaluations/generalization_v3/results/slurm/babysit_%j.log
#SBATCH --error=evaluations/generalization_v3/results/slurm/babysit_%j.log
#SBATCH --partition=cpu_sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

set -uo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "$HOME/miniforge3/bin/conda" ]; then
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
else
  eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
fi
conda activate "${CONDA_ENV:-pred-distr}"
cd "$REPO_DIR"

V3=evaluations/generalization_v3
PLAN="${PLAN:?set PLAN=<plan file>}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
read -r -a PARTITIONS <<<"${CPU_PARTITIONS:-cpu_sapphire}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$V3/slurm/train_cpu.sh}"
read -r -a SBATCH_EXTRA <<<"${SBATCH_EXTRA:-}"   # e.g. "--time=12:00:00"
declare -A streak attempts jobid
n_submitted=0

log() { echo "$(date '+%F %T')  $*"; }
job_alive() { [ -n "${1:-}" ] && squeue -h -j "$1" 2>/dev/null | grep -q .; }

submit_for() {  # run exp split model alpha seed [extra args]
  local run="$1" exp="$2" split="$3" model="$4" alpha="$5" seed="$6" extra="${7:-}" fam=large grp=vqe
  [ "$alpha" = "-" ] && alpha=0
  [ "$exp" = lofo ] && fam="$split"
  [ "$exp" = logo ] && grp="$split"
  # n_submitted is advanced by the caller: this function runs in a $(...) subshell.
  local part="${PARTITIONS[$(( n_submitted % ${#PARTITIONS[@]} ))]}"
  RUN="$run" MODEL="$model" ALPHA="$alpha" SEED="$seed" EXPERIMENTS="$exp" \
    FAMILIES="$fam" SPLIT_GROUPS="$grp" EXTRA_ARGS="$extra" \
    sbatch --parsable --partition="$part" --job-name="cpu_${run}_${split}" \
    "${SBATCH_EXTRA[@]}" "$TRAIN_SCRIPT"
}

log "supervisor started on $(hostname), plan $PLAN"
while true; do
  out=$(python3 "$V3/scripts/pending.py" "$PLAN")
  summary=$(grep '^SUMMARY' <<<"$out")
  if grep -q ' free=0 ' <<<"$summary" && grep -q ' busy=0 ' <<<"$summary"; then
    log "all splits done: $summary"
    break
  fi

  declare -A seen=()
  while read -r _ run exp split model alpha seed extra; do
    [ -z "${run:-}" ] && continue
    key="$run $exp $split"; seen[$key]=1
    streak[$key]=$(( ${streak[$key]:-0} + 1 ))
    if [ "${streak[$key]}" -ge 3 ] && ! job_alive "${jobid[$key]:-}"; then
      if [ "${attempts[$key]:-0}" -ge "$MAX_ATTEMPTS" ]; then
        [ "${streak[$key]}" -eq 3 ] && log "GIVING UP on $key after $MAX_ATTEMPTS attempts"
        continue
      fi
      j=$(submit_for "$run" "$exp" "$split" "$model" "$alpha" "$seed" "${extra:-}")
      n_submitted=$(( n_submitted + 1 ))
      attempts[$key]=$(( ${attempts[$key]:-0} + 1 ))
      jobid[$key]="$j"
      log "free: $key -> job $j (attempt ${attempts[$key]})"
    fi
  done < <(grep '^FREE' <<<"$out")
  for key in "${!streak[@]}"; do [ -z "${seen[$key]:-}" ] && streak[$key]=0; done
  unset seen
  # Hourly progress line.
  [ "$(date +%M)" = "00" ] && log "progress: $summary"
  sleep 60
done

if [ -f "$V3/scripts/summarize.py" ]; then
  python3 "$V3/scripts/summarize.py" "$PLAN"
fi
log "supervisor finished"
