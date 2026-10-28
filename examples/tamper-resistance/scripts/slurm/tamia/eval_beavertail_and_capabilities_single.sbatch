#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

model_id=${1:?missing model id}
model=${2:?missing model}
method=${3:?missing method}
result_root=${4:?missing result root}

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python scripts/evaluate_beavertail_and_capabilities.py single \
    "$model_id" \
    "$model" \
    "$method" \
    "$result_root"
