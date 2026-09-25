#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export WANDB_DIR="$PWD/artifacts/mila/logs"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WANDB_DIR"

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
if [[ $# -ge 4 ]]; then
  axolotl merge-lora "$1" --lora-model-dir "$3" --output-dir "$4"
else
  axolotl merge-lora "$1"
fi
config=${2:?missing relearning config}
axolotl train "$config"
