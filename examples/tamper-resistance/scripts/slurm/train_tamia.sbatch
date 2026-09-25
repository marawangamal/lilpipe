#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-train"
export PYTHONPATH="$PWD"
export WANDB_DIR="$PWD/artifacts/tamia/logs"
export WANDB_MODE=offline

config=${1:?missing training config}
tamia_config="$SLURM_TMPDIR/$(basename "$config")"
sed 's#artifacts/mila/#artifacts/tamia/#g' "$config" > "$tamia_config"

uv sync --offline --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:?}" \
  -m axolotl.cli.train "$tamia_config"
