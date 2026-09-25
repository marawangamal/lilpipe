#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-train"
export PYTHONPATH="$PWD"
export WANDB_DIR="$PWD/artifacts/mila/logs"

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
if [[ $# -ge 4 ]]; then
  axolotl merge-lora "$1" --lora-model-dir "$3" --output-dir "$4"
else
  axolotl merge-lora "$1"
fi
if [[ "$2" == configs/relearn/z7b/* ]]; then
  torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:?}" \
    -m axolotl.cli.train "$2"
else
  axolotl train "$2" --launcher python
fi
