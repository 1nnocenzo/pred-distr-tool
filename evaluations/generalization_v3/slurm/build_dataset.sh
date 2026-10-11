#!/bin/bash
# Build the qubit-annotated copy of the graph dataset (CPU only).
# Reads ~/compileCircuits/graph_dataset_expected_fidelity.pt and the .qasm files;
# writes evaluations/generalization_v3/data/.  Submit from the repository root:
#
#     sbatch evaluations/generalization_v3/slurm/build_dataset.sh

#SBATCH --job-name=gen_v3_data
#SBATCH --output=evaluations/generalization_v3/results/slurm/build_dataset_%j.log
#SBATCH --error=evaluations/generalization_v3/results/slurm/build_dataset_%j.err
#SBATCH --partition=cpu_sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "$HOME/miniforge3/bin/conda" ]; then
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
else
  eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
fi
conda activate "${CONDA_ENV:-pred-distr}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO_DIR"

python3 -u evaluations/generalization_v3/scripts/build_qubit_dataset.py \
  --workers "${SLURM_CPUS_PER_TASK:-32}"
