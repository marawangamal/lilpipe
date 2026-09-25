#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

# Shared caches and outputs.
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export WANDB_DIR="$PWD/artifacts/tamia/logs"

# Job-local environment.
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"

export PATH="$HOME/.local/bin:$PATH"

config=${1:?missing training config}

mkdir -p "$WANDB_DIR"

module load httpproxy/1.0
uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
axolotl train "$config"
