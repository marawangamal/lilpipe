#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
#SBATCH --time=03:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
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

install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group train
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
config=${1:?missing training config}
shift
merge=false
if [[ ${1:-} == "--merge" ]]; then
  merge=true
  shift
fi
axolotl train "$config" "$@"
if [[ "$merge" == true ]]; then
  axolotl merge-lora "$config"
fi
