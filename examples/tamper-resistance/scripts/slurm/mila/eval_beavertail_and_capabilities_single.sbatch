#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
#SBATCH --time=03:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_CONCURRENT_INSTALLS=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

model_id=${1:?missing model id}
model=${2:?missing model}
method=${3:?missing method}
result_root=${4:?missing result root}

mkdir -p "$UV_CACHE_DIR"
install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group eval
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python scripts/evaluate_beavertail_and_capabilities.py single \
    "$model_id" \
    "$model" \
    "$method" \
    "$result_root"
