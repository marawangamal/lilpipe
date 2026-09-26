#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

config=${1:?missing training config}
adapter=${2:?missing selected adapter checkpoint}
output=${3:?missing merged output directory}

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
axolotl merge-lora "$config" --lora-model-dir "$adapter" --output-dir "$output"
