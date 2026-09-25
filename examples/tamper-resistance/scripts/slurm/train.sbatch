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
if [[ "$1" == configs/*/z7b/* ]]; then
  torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:?}" \
    -m axolotl.cli.train "$1"
else
  axolotl train "$1" --launcher python
fi
if [[ ${2:-} == "--merge" ]]; then
  axolotl merge-lora "$1"
fi
