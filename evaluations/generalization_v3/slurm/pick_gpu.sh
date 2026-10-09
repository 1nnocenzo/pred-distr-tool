# Sourced by the v3 SLURM scripts.  Keeps, among the GPUs SLURM assigned, the
# first one on which a CUDA kernel actually runs, and exports it as
# CUDA_VISIBLE_DEVICES.  compute-7-12 has a broken GPU (0000:19:00.0) that SLURM
# still hands out; requesting --gres=gpu:2 and picking the working one avoids it.

pick_working_gpu() {
  local candidates="${CUDA_VISIBLE_DEVICES:-}"
  local gpu
  for gpu in ${candidates//,/ }; do
    if CUDA_VISIBLE_DEVICES="$gpu" timeout 120 python3 -c \
        "import torch; x = torch.ones(8, device='cuda') * 2; torch.cuda.synchronize(); assert float(x.sum()) == 16.0" \
        >/dev/null 2>&1; then
      export CUDA_VISIBLE_DEVICES="$gpu"
      echo "Using GPU index $gpu: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
      return 0
    fi
    echo "GPU index $gpu is not usable — skipping." >&2
  done
  echo "No usable CUDA device among '${candidates}' — aborting." >&2
  return 3
}
