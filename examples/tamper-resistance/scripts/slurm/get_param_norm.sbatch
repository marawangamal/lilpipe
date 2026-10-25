#!/bin/bash
# Create artifacts/logs before submitting.
#   mkdir -p artifacts/logs
#   sbatch scripts/slurm/get_param_norm.sbatch [--model MODEL]
#SBATCH --job-name=param-norm
#SBATCH --output=artifacts/logs/slurm-%j.out
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
exec python scripts/analysis/get_param_norm.py "$@"
