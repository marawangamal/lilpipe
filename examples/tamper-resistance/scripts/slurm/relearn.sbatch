#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-train"
export PYTHONPATH="$PWD"
export WANDB_DIR="$PWD/artifacts/logs"

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
axolotl merge-lora "$1"
axolotl train "$2" --launcher python
