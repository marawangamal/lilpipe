#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
#SBATCH --time=03:00:00
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
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_DATA_DIR="$SCRATCH/.cache/wandb/staging"
export WANDB_MODE=offline
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

attack_config=${1:?missing attack config}
alignment_config=${2:?missing alignment config}
aligned_adapter=${3:?missing aligned adapter}
merged_base=${4:?missing merged base destination}
merged_model="$merged_base/merged"
shift 4

mkdir -p "$UV_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group train
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# Migrate merge markers written by the earlier wrapper, but only when the
# nested Axolotl output has the files required for a complete merged model.
if [[ -f "$merged_base/.merge-complete" \
  && -f "$merged_model/config.json" \
  && -f "$merged_model/model.safetensors.index.json" ]]; then
  mv "$merged_base/.merge-complete" "$merged_model/.merge-complete"
fi

if [[ ! -f "$merged_model/.merge-complete" ]]; then
  if [[ -e "$merged_base" ]]; then
    echo "incomplete merged base already exists: $merged_base" >&2
    exit 1
  fi
  axolotl merge-lora "$alignment_config" \
    --lora-model-dir "$aligned_adapter" \
    --output-dir "$merged_base"
  touch "$merged_model/.merge-complete"
fi

axolotl train "$attack_config" "$@"
