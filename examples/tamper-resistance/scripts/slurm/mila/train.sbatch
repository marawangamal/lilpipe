#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
# Offline mode avoids shared-cluster Hub rate limits and makes cache requirements explicit.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export WANDB_DIR="$PWD/artifacts/mila/logs"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WANDB_DIR"

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
config=${1:?missing training config}
axolotl train "$config"
if [[ ${2:-} == "--merge" ]]; then
  axolotl merge-lora "$config"
fi
