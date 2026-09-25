#!/bin/bash
# Create artifacts/logs before submitting.
#   mkdir -p artifacts/logs
#   sbatch --export=ALL,UV_GROUP=train slurm/template_tamia.sbatch python train.py
# gpubase_bynode_b1 allocates a whole four-H100 node.
#SBATCH --job-name=lilpipe
#SBATCH --output=artifacts/logs/slurm-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=01:00:00

set -euo pipefail

UV_GROUP=${UV_GROUP:?set UV_GROUP to the dependency group to install}

# Shared caches and outputs.
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export WANDB_DIR="$PWD/artifacts/wandb"

# Job-local environment.
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WANDB_DIR"

module load httpproxy/1.0
uv sync --frozen --group "$UV_GROUP"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
exec "$@"
