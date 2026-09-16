#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-train"
export PYTHONPATH="$PWD"

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python scripts/steering/task_vector.py "$1"
