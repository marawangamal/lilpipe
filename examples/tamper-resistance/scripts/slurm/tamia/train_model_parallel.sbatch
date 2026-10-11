#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export WANDB_DIR="$PWD/artifacts/tamia/logs"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_DATA_DIR="$SCRATCH/.cache/wandb/staging"
export WANDB_MODE=offline
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

config=${1:?missing training config}

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

module load httpproxy/1.0
uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
# The paper uses one process with a model dispatched across visible GPUs.
axolotl train "$config" --launcher python
