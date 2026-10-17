#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/optuna-%j.out
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <optuna.yml>" >&2
  exit 2
fi

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=120
export WANDB_DIR="$PWD/artifacts/tamia/logs"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_DATA_DIR="$SCRATCH/.cache/wandb/staging"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$UV_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
module load httpproxy/1.0
uv sync --frozen --group train --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python scripts/optuna_hpo.py worker "$1"
