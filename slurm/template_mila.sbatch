#!/bin/bash
# Create artifacts/logs before submitting.
#   mkdir -p artifacts/logs
#   sbatch --export=ALL,UV_GROUP=train slurm/template_mila.sbatch python train.py
#SBATCH --job-name=lilpipe
#SBATCH --output=artifacts/logs/slurm-%j.out
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

UV_GROUP=${UV_GROUP:?set UV_GROUP to the dependency group to install}

# Shared model cache and outputs.
export HF_HOME="$SCRATCH/.cache/huggingface"
# Offline mode avoids shared-cluster Hub rate limits and makes cache requirements explicit.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DIR="$PWD/artifacts/wandb"

# Job-local uv cache and environment; Mila may set SLURM_TMPDIR=/tmp.
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WANDB_DIR"

uv sync --frozen --group "$UV_GROUP"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
exec "$@"
