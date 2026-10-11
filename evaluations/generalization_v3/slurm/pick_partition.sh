#!/bin/bash
# Prints the "-p <partition> --gres=gpu:N" options of the GPU partition whose estimated start time
# (sbatch --test-only) is the earliest for a job of the given time limit and memory.
#   OPTS=$(evaluations/generalization_v3/slurm/pick_partition.sh 03:00:00 32G)
T="${1:-03:00:00}"; MEM="${2:-32G}"; best=""; bestt="9999"
for p in gpu_a100 gpu_a40_ext gpu_h200 gpu_a40; do   # no V100: torch 2.11+cu128 has no sm_70 kernels
  g=1; [ "$p" = gpu_h200 ] && g=2          # compute-7-12 has a broken GPU: ask 2, pick_gpu.sh keeps the good one
  xx=""; [ "$p" = gpu_v100 ] && xx="--exclude=compute-1-29,compute-1-31"
  s=$(sbatch --test-only -p "$p" $xx --gres=gpu:$g -c 8 --mem="$MEM" -t "$T" --wrap=true 2>&1 | grep -oE 'start at \S+' | cut -d' ' -f3)
  x=""; [ "$p" = gpu_v100 ] && x=" --exclude=compute-1-29,compute-1-31"   # their V100s fail the CUDA probe
  [ -n "$s" ] && [[ "$s" < "$bestt" ]] && { bestt="$s"; best="-p $p --gres=gpu:$g$x"; }
done
echo "$best"
