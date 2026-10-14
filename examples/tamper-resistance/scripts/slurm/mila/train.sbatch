#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
# Offline mode avoids shared-cluster Hub rate limits and makes cache requirements explicit.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_CONCURRENT_INSTALLS=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export WANDB_DIR="$PWD/artifacts/mila/logs"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_DATA_DIR="$SCRATCH/.cache/wandb/staging"
export WANDB_MODE=offline
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WANDB_DIR"
mkdir -p "$UV_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

flock "$UV_CACHE_DIR/.lilpipe-install.lock" \
  uv sync --offline --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
config=${1:?missing training config}
axolotl train "$config"
if [[ ${2:-} == "--merge" ]]; then
  axolotl merge-lora "$config"
fi
