#!/bin/bash
#SBATCH --output=artifacts/mila/logs/optuna-%A_%a.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export WANDB_MODE=offline
export WANDB_DIR="$PWD/artifacts/mila/logs"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$UV_CACHE_DIR" "$WANDB_DIR"
install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group train --group eval
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python scripts/optuna_hpo.py worker "$1"
