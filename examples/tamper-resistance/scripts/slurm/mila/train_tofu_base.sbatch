#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
#SBATCH --time=12:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_CONCURRENT_INSTALLS=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export WANDB_DIR="$PWD/artifacts/mila/logs"
export WANDB_MODE=offline
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group train --group eval
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

config=${1:?missing training config}
evaluation=${2:?missing evaluation path}
axolotl train "$config"
model=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")
python scripts/evaluate_tofu.py "$model" "$evaluation"
